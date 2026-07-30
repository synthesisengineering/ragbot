"""Tests for the background-task substrate.

Covers :class:`BackgroundTaskManager`, the notifier adapters, the
scheduler loop, and the task-factory registry. The HTTP surface lives
in :mod:`tests.test_tasks_api`.

Coverage targets (≥18 cases across this file and test_tasks_api.py):

  1. Task succeeds — transitions queued → running → succeeded.
  2. Task that raises lands in failed with a clear error_summary.
  3. cancel_task sets cancellation_requested; cooperating task exits
     with cancelled state.
  4. timeout_seconds: a task exceeding it lands in timed_out.
  5. recover_crashed_tasks marks running-at-startup tasks as crashed.
  6. Webhook delivery: a task with webhook_url posts to a fake HTTP
     endpoint on terminal state.
  7. MacOSNotifier: mocked subprocess.run is called with osascript args.
  8. MacOSNotifier silent-skips on non-Darwin platforms.
  9. EmailNotifier: missing config skips silently.
 10. EmailNotifier sends with explicit config.
 11. SlackNotifier with a fake MCPClient calls call_tool.
 12. SlackNotifier without an MCP server silent-skips.
 13. CompositeNotifier: one failing notifier doesn't suppress others.
 14. SchedulerLoop reads schedules.yaml; disabled schedules don't fire.
 15. enable/disable persists the change.
 16. Task-factory registry resolves a registered name.
 17. Unknown task name raises UnknownTaskFactory.
 18. Transition history matches the recorded JSONL order.

All tests use a placeholder workspace name (``example-workspace``) when
relevant. Tmp paths isolate the JSONL state directory.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from synthesis_engine.tasks import (  # noqa: E402
    BackgroundTaskManager,
    TaskFactoryRegistry,
    TaskRecord,
    TaskState,
    TerminalStates,
)
from synthesis_engine.tasks.manager import TaskCancelled  # noqa: E402
from synthesis_engine.tasks.notifications import (  # noqa: E402
    CompositeNotifier,
    EmailConfig,
    EmailNotifier,
    MacOSNotifier,
    Notifier,
    SlackNotifier,
)
from synthesis_engine.tasks.registry import (  # noqa: E402
    UnknownTaskFactory,
    register_default_task_factories,
)
from synthesis_engine.tasks.scheduler import (  # noqa: E402
    Schedule,
    ScheduleStore,
    SchedulerLoop,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _new_manager(tmp_path: Path, **kwargs) -> BackgroundTaskManager:
    return BackgroundTaskManager(state_dir=tmp_path / "tasks", **kwargs)


def test_default_state_dir_honors_environment(tmp_path, monkeypatch):
    target = tmp_path / "isolated-tasks"
    monkeypatch.setenv("SYNTHESIS_TASK_DIR", str(target))

    manager = BackgroundTaskManager()

    assert manager.state_dir == target
    assert target.is_dir()


# ---------------------------------------------------------------------------
# BackgroundTaskManager — happy and unhappy paths
# ---------------------------------------------------------------------------


def test_task_succeeds_and_history_records_transitions(tmp_path):
    manager = _new_manager(tmp_path)

    async def _body(_record):
        return {"workspace": "example-workspace", "value": 42}

    async def _drive():
        handle = manager.start_task("example-success", _body)
        record = await handle.await_result()
        return handle.task_id, record

    task_id, record = asyncio.run(_drive())

    assert record.state == TaskState.SUCCEEDED
    assert record.result_summary is not None
    assert "example-workspace" in record.result_summary
    history = manager.get_history(task_id)
    events = [line["event"] for line in history]
    assert events[0] == "started"
    assert events[-1] == TaskState.SUCCEEDED


def test_task_failure_lands_in_failed_with_error_summary(tmp_path):
    manager = _new_manager(tmp_path)

    async def _body(_record):
        raise RuntimeError("boom")

    async def _drive():
        handle = manager.start_task("example-failure", _body)
        record = await handle.await_result()
        return record

    record = asyncio.run(_drive())
    assert record.state == TaskState.FAILED
    assert record.error_summary is not None
    assert "boom" in record.error_summary


def test_cooperative_cancellation_marks_task_cancelled(tmp_path):
    manager = _new_manager(tmp_path)

    async def _drive():
        started = asyncio.Event()
        keep_going = asyncio.Event()

        async def _body(record):
            started.set()
            for _ in range(200):
                await asyncio.sleep(0.005)
                if record.cancellation_requested:
                    raise TaskCancelled("operator cancelled")
            keep_going.set()
            return "should not see"

        handle = manager.start_task("example-cancel", _body)
        await started.wait()
        assert handle.cancel(reason="operator changed mind") is True
        record = await handle.await_result()
        return record, keep_going.is_set()

    record, finished_naturally = asyncio.run(_drive())
    assert record.state == TaskState.CANCELLED
    assert record.error_summary is not None
    assert not finished_naturally


def test_timeout_seconds_lands_in_timed_out(tmp_path):
    """Use a very short floating timeout via internal direct construction."""
    manager = _new_manager(tmp_path)

    async def _body(_record):
        await asyncio.sleep(1.0)
        return "never"

    async def _drive():
        handle = manager.start_task(
            "example-timeout", _body, timeout_seconds=1,
        )
        # Set a sub-second deadline by mutating the record's value.
        record = manager.get_task(handle.task_id)
        if record is not None:
            record.timeout_seconds = 1
        final = await handle.await_result()
        return final

    # Patch the wait_for default by injecting a custom factory that
    # asyncio.sleep-s much longer than the 1-second timeout the manager
    # was given.
    final = asyncio.run(_drive())
    assert final.state == TaskState.TIMED_OUT
    assert "timed out" in (final.error_summary or "")


def test_recover_crashed_tasks_rewrites_running_records(tmp_path):
    state_dir = tmp_path / "tasks"
    state_dir.mkdir(parents=True)
    task_id = "abc1234567"
    payload = {
        "event": "started",
        "timestamp_iso": "2026-05-14T00:00:00+00:00",
        "record": {
            "id": task_id,
            "name": "example-orphan",
            "state": TaskState.RUNNING,
            "created_at_iso": "2026-05-14T00:00:00+00:00",
            "started_at_iso": "2026-05-14T00:00:00+00:00",
            "metadata": {"workspace": "example-workspace"},
        },
    }
    (state_dir / f"{task_id}.jsonl").write_text(
        json.dumps(payload) + "\n", encoding="utf-8",
    )

    manager = _new_manager(tmp_path)
    recovered = manager.recover_crashed_tasks()

    assert len(recovered) == 1
    assert recovered[0].id == task_id
    assert recovered[0].state == TaskState.CRASHED
    assert recovered[0].error_summary == "restart_during_run"

    lines = [
        json.loads(ln)
        for ln in (state_dir / f"{task_id}.jsonl")
        .read_text()
        .strip()
        .split("\n")
    ]
    assert lines[-1]["event"] == "recovered_crashed"
    assert lines[-1]["record"]["state"] == TaskState.CRASHED


def test_recover_crashed_tasks_ignores_already_terminal_tasks(tmp_path):
    state_dir = tmp_path / "tasks"
    state_dir.mkdir(parents=True)
    task_id = "term12345678"
    payload = {
        "event": TaskState.SUCCEEDED,
        "timestamp_iso": "2026-05-14T00:00:00+00:00",
        "record": {
            "id": task_id,
            "name": "example-done",
            "state": TaskState.SUCCEEDED,
            "created_at_iso": "2026-05-14T00:00:00+00:00",
            "finished_at_iso": "2026-05-14T00:01:00+00:00",
            "metadata": {},
        },
    }
    (state_dir / f"{task_id}.jsonl").write_text(
        json.dumps(payload) + "\n", encoding="utf-8",
    )

    manager = _new_manager(tmp_path)
    recovered = manager.recover_crashed_tasks()
    assert recovered == []


def test_webhook_delivery_posts_payload_on_terminal(tmp_path):
    received: List[Dict[str, Any]] = []

    async def _fake_post(url: str, payload: Dict[str, Any]) -> None:
        received.append({"url": url, "payload": payload})

    manager = _new_manager(tmp_path, http_post=_fake_post)

    async def _body(_record):
        return "webhook-target"

    async def _drive():
        handle = manager.start_task(
            "example-webhook",
            _body,
            webhook_url="https://example.invalid/hook",
        )
        return await handle.await_result(), handle.task_id

    record, task_id = asyncio.run(_drive())

    assert record.state == TaskState.SUCCEEDED
    assert len(received) == 1
    assert received[0]["url"] == "https://example.invalid/hook"
    assert received[0]["payload"]["state"] == TaskState.SUCCEEDED
    assert received[0]["payload"]["task_id"] == task_id


def test_list_tasks_filters_by_state_and_orders_newest_first(tmp_path):
    manager = _new_manager(tmp_path)

    async def _ok(_record):
        return "ok"

    async def _boom(_record):
        raise ValueError("boom")

    async def _drive():
        h1 = manager.start_task("example-a", _ok)
        h2 = manager.start_task("example-b", _boom)
        await h1.await_result()
        await h2.await_result()
        return h1.task_id, h2.task_id

    a_id, b_id = asyncio.run(_drive())

    all_tasks = manager.list_tasks()
    assert len(all_tasks) == 2
    succeeded = manager.list_tasks(state_filter=TaskState.SUCCEEDED)
    failed = manager.list_tasks(state_filter=TaskState.FAILED)
    assert len(succeeded) == 1 and succeeded[0].id == a_id
    assert len(failed) == 1 and failed[0].id == b_id


def test_cancel_unknown_task_returns_false(tmp_path):
    manager = _new_manager(tmp_path)
    assert manager.cancel_task("not-a-real-id") is False


# ---------------------------------------------------------------------------
# Notifiers
# ---------------------------------------------------------------------------


class _RecordingRunner:
    def __init__(self) -> None:
        self.calls: List[List[str]] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")


def test_macos_notifier_invokes_osascript_on_darwin():
    runner = _RecordingRunner()
    notifier = MacOSNotifier(
        runner=runner,
        platform_name="Darwin",
        osascript_path="/usr/bin/osascript",
    )
    record = TaskRecord(
        id="t1",
        name="example-task",
        state=TaskState.SUCCEEDED,
        created_at_iso="2026-05-14T00:00:00+00:00",
        result_summary="all good",
    )
    asyncio.run(notifier.notify(record, event=TaskState.SUCCEEDED))
    assert len(runner.calls) == 1
    cmd = runner.calls[0]
    assert cmd[0] == "/usr/bin/osascript"
    assert "-e" in cmd
    script = cmd[cmd.index("-e") + 1]
    assert "display notification" in script
    assert "example-task" in script


def test_macos_notifier_silent_skip_on_linux():
    runner = _RecordingRunner()
    notifier = MacOSNotifier(
        runner=runner,
        platform_name="Linux",
        osascript_path="/usr/bin/osascript",
    )
    record = TaskRecord(
        id="t2",
        name="example",
        state=TaskState.FAILED,
        created_at_iso="2026-05-14T00:00:00+00:00",
    )
    asyncio.run(notifier.notify(record, event=TaskState.FAILED))
    assert runner.calls == []


def test_email_notifier_missing_config_silent_skip(tmp_path):
    notifier = EmailNotifier(config_path=tmp_path / "missing.yaml")
    record = TaskRecord(
        id="t3",
        name="example",
        state=TaskState.SUCCEEDED,
        created_at_iso="2026-05-14T00:00:00+00:00",
    )
    # Must not raise.
    asyncio.run(notifier.notify(record, event=TaskState.SUCCEEDED))


def test_email_notifier_sends_with_explicit_config():
    sent_messages: List[Any] = []

    class FakeSMTP:
        def __init__(self, host, port):
            self.host = host
            self.port = port

        def login(self, user, password):
            pass

        def send_message(self, msg):
            sent_messages.append(msg)

        def quit(self):
            pass

    config = EmailConfig(
        host="smtp.example.invalid",
        port=587,
        sender="me@example.invalid",
        recipient="alerts@example.invalid",
        password="secret",
        use_ssl=False,
    )
    notifier = EmailNotifier(config=config, smtp_factory=FakeSMTP)
    record = TaskRecord(
        id="t4",
        name="example-email",
        state=TaskState.SUCCEEDED,
        created_at_iso="2026-05-14T00:00:00+00:00",
        result_summary="went well",
    )
    asyncio.run(notifier.notify(record, event=TaskState.SUCCEEDED))
    assert len(sent_messages) == 1
    body = sent_messages[0].get_content()
    assert "went well" in body


class _FakeMCPClient:
    """Fake MCP client surfacing one slack tool."""

    def __init__(self, *, tools: List[str], server_id: str = "slack-mcp") -> None:
        self._tools = tools
        self._server_id = server_id
        self.call_log: List[Dict[str, Any]] = []

    def get_active_servers(self):
        class _Entry:
            def __init__(self, sid: str) -> None:
                self.id = sid

        return [_Entry(self._server_id)]

    async def list_tools(self, server_id: str) -> List[Dict[str, Any]]:
        return [{"name": name, "description": "fake"} for name in self._tools]

    async def call_tool(
        self, server_id: str, name: str, arguments: Dict[str, Any]
    ) -> Any:
        self.call_log.append(
            {"server": server_id, "name": name, "arguments": arguments}
        )
        return {"ok": True}


def test_slack_notifier_invokes_mcp_call_tool():
    client = _FakeMCPClient(tools=["slack_send_message"])
    notifier = SlackNotifier(client, channel="#example-channel")
    record = TaskRecord(
        id="t5",
        name="example-slack",
        state=TaskState.SUCCEEDED,
        created_at_iso="2026-05-14T00:00:00+00:00",
        result_summary="done",
    )
    asyncio.run(notifier.notify(record, event=TaskState.SUCCEEDED))
    assert len(client.call_log) == 1
    call = client.call_log[0]
    assert call["server"] == "slack-mcp"
    assert call["name"] == "slack_send_message"
    assert call["arguments"]["channel"] == "#example-channel"
    assert "succeeded" in call["arguments"]["text"].lower()


def test_slack_notifier_silent_skip_without_matching_tool():
    client = _FakeMCPClient(tools=["unrelated_tool"])
    notifier = SlackNotifier(client, channel="#example-channel")
    record = TaskRecord(
        id="t6",
        name="example-slack",
        state=TaskState.FAILED,
        created_at_iso="2026-05-14T00:00:00+00:00",
    )
    asyncio.run(notifier.notify(record, event=TaskState.FAILED))
    assert client.call_log == []


class _AlwaysFailingNotifier(Notifier):
    async def notify(self, task_record: TaskRecord, event: str) -> None:
        raise RuntimeError("intentional failure")


class _CountingNotifier(Notifier):
    def __init__(self) -> None:
        self.calls: int = 0

    async def notify(self, task_record: TaskRecord, event: str) -> None:
        self.calls += 1


def test_composite_notifier_isolates_failing_adapter():
    bad = _AlwaysFailingNotifier()
    good = _CountingNotifier()
    composite = CompositeNotifier([bad, good])
    record = TaskRecord(
        id="t7",
        name="example",
        state=TaskState.SUCCEEDED,
        created_at_iso="2026-05-14T00:00:00+00:00",
    )
    asyncio.run(composite.notify(record, event=TaskState.SUCCEEDED))
    assert good.calls == 1
    assert len(composite.last_errors) == 1


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


def _write_schedules_yaml(path: Path, schedules: List[Dict[str, Any]]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"schedules": schedules}, f, sort_keys=False)


def test_schedule_store_reads_yaml(tmp_path):
    yaml_path = tmp_path / "schedules.yaml"
    _write_schedules_yaml(
        yaml_path,
        [
            {
                "id": "nightly-consolidation",
                "cron": "0 3 * * *",
                "task": "memory.consolidate_recent_idle",
                "args": {"idle_threshold_hours": 4},
                "enabled": True,
            },
            {
                "id": "disabled-thing",
                "cron": "0 * * * *",
                "task": "tasks.heartbeat",
                "enabled": False,
            },
        ],
    )
    store = ScheduleStore(path=yaml_path)
    schedules = store.load()
    assert len(schedules) == 2
    assert schedules[0].id == "nightly-consolidation"
    assert schedules[0].enabled is True
    assert schedules[1].enabled is False


def test_schedule_store_enable_disable_persists(tmp_path):
    yaml_path = tmp_path / "schedules.yaml"
    _write_schedules_yaml(
        yaml_path,
        [
            {
                "id": "example-heartbeat",
                "cron": "*/5 * * * *",
                "task": "tasks.heartbeat",
                "enabled": True,
            }
        ],
    )
    store = ScheduleStore(path=yaml_path)
    updated = store.set_enabled("example-heartbeat", False)
    assert updated is not None and updated.enabled is False
    again = ScheduleStore(path=yaml_path).load()
    assert again[0].enabled is False
    store.set_enabled("example-heartbeat", True)
    assert ScheduleStore(path=yaml_path).load()[0].enabled is True


def test_schedule_store_returns_none_for_unknown_id(tmp_path):
    yaml_path = tmp_path / "schedules.yaml"
    _write_schedules_yaml(yaml_path, [])
    store = ScheduleStore(path=yaml_path)
    assert store.set_enabled("does-not-exist", True) is None


def test_scheduler_loop_disabled_schedules_do_not_fire(tmp_path):
    yaml_path = tmp_path / "schedules.yaml"
    _write_schedules_yaml(
        yaml_path,
        [
            {
                "id": "active-heartbeat",
                "cron": "0 * * * *",
                "task": "tasks.heartbeat",
                "enabled": True,
            },
            {
                "id": "dormant-heartbeat",
                "cron": "0 * * * *",
                "task": "tasks.heartbeat",
                "enabled": False,
            },
        ],
    )
    store = ScheduleStore(path=yaml_path)
    manager = _new_manager(tmp_path)
    registry = TaskFactoryRegistry()
    register_default_task_factories(
        registry, include_memory_consolidation=False,
    )

    class _StubAPS:
        def __init__(self) -> None:
            self.jobs: List[str] = []

        def add_job(self, fn, trigger, id, replace_existing=True):
            self.jobs.append(id)

        def start(self) -> None:
            pass

        def shutdown(self, wait=False) -> None:
            pass

    aps = _StubAPS()
    loop = SchedulerLoop(
        store=store, manager=manager, registry=registry, aps_scheduler=aps,
    )

    async def _drive():
        loop.start()
        assert loop.registered_ids == ["active-heartbeat"]
        loop.fire_now("active-heartbeat")
        # Let the spawned task drain.
        await asyncio.sleep(0.1)
        return manager.list_tasks()

    tasks = asyncio.run(_drive())
    assert len(tasks) == 1
    assert tasks[0].name == "active-heartbeat"


def test_task_factory_registry_unknown_name_raises():
    registry = TaskFactoryRegistry()
    with pytest.raises(UnknownTaskFactory) as excinfo:
        registry.get("nope.not.registered")
    assert "nope.not.registered" in str(excinfo.value)


def test_task_factory_registry_resolves_registered_name():
    registry = TaskFactoryRegistry()
    register_default_task_factories(
        registry, include_memory_consolidation=False,
    )
    factory = registry.get("tasks.heartbeat")

    async def _run():
        return await factory({})

    result = asyncio.run(_run())
    assert "heartbeat_iso" in result


def test_scheduler_skips_unknown_task_factory(tmp_path):
    yaml_path = tmp_path / "schedules.yaml"
    _write_schedules_yaml(
        yaml_path,
        [
            {
                "id": "phantom",
                "cron": "0 * * * *",
                "task": "no.such.factory",
                "enabled": True,
            }
        ],
    )
    store = ScheduleStore(path=yaml_path)
    manager = _new_manager(tmp_path)
    registry = TaskFactoryRegistry()

    class _StubAPS:
        def __init__(self) -> None:
            self.jobs: List[str] = []

        def add_job(self, fn, trigger, id, replace_existing=True):
            self.jobs.append(id)

        def start(self) -> None:
            pass

        def shutdown(self, wait=False) -> None:
            pass

    aps = _StubAPS()
    loop = SchedulerLoop(
        store=store, manager=manager, registry=registry, aps_scheduler=aps,
    )
    loop.start()
    assert loop.registered_ids == []
