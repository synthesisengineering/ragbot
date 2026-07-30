"""Tests for the FastAPI lifespan handler in ``src/api/main.py``.

Phase 5 wires the lifespan handler to:

1. Call :meth:`BackgroundTaskManager.recover_crashed_tasks` before any new
   task starts, so a task left in ``running`` state by a prior process is
   marked ``crashed`` deterministically.
2. Register the built-in task factories (``tasks.heartbeat`` +
   ``memory.consolidate_recent_idle``) on the process-singleton registry.
3. Construct a :class:`SchedulerLoop` and call ``start()`` iff
   ``RAGBOT_SCHEDULER`` is truthy. The loop instance lands on
   ``app.state.scheduler_loop`` and is stopped during shutdown.
4. Touch the MCP client singleton so misconfiguration surfaces at startup.

Shutdown:

* Stops the scheduler if one was started.
* Calls :func:`synthesis_engine.observability.shutdown_tracer` so the
  OTEL exporter flushes pending spans.

The tests build a fresh FastAPI app per case (via ``importlib.reload`` or
direct construction) and exercise the lifespan via the FastAPI TestClient's
context-manager protocol.

Placeholder workspace names (``example-workspace``) appear where relevant.

Coverage (≥6 cases):

  1. ``recover_crashed_tasks`` runs on startup; an existing JSONL file with
     a ``running``-state task gets a ``crashed`` line appended.
  2. ``register_default_task_factories`` runs on startup; ``tasks.heartbeat``
     and ``memory.consolidate_recent_idle`` are registered.
  3. Scheduler does NOT start when ``RAGBOT_SCHEDULER`` is unset.
  4. Scheduler DOES start when ``RAGBOT_SCHEDULER=1``; the loop lands on
     ``app.state.scheduler_loop`` and the router-singleton.
  5. Shutdown calls ``scheduler.stop()`` and ``shutdown_tracer()`` so the
     scheduler is no longer running afterwards.
  6. Lifespan handler is idempotent under repeated startup (per-test
     isolation): a second TestClient context manager on a fresh app stops
     the previous scheduler cleanly without raising.
  7. Lifespan does NOT crash if the manager / registry / scheduler raises;
     warnings are logged but startup continues.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from synthesis_engine.tasks import (  # noqa: E402
    BackgroundTaskManager,
    TaskRecord,
    TaskState,
    set_default_manager,
)
from synthesis_engine.tasks.registry import (  # noqa: E402
    TaskFactoryRegistry,
    set_default_registry,
)
from synthesis_engine.tasks.scheduler import SchedulerLoop  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes / stubs
# ---------------------------------------------------------------------------


class _FakeScheduler:
    """Stand-in for SchedulerLoop that records lifecycle calls.

    We use a fake instead of a real :class:`SchedulerLoop` so the tests
    do not depend on apscheduler being installed and do not have to wait
    for the cron clock to tick. The fake mirrors the substrate's contract:
    ``start()`` flips ``is_running``; ``stop()`` flips it back.
    """

    def __init__(self) -> None:
        self.start_calls = 0
        self.stop_calls = 0
        self._running = False
        self.registered_ids: List[str] = []

    def start(self) -> None:
        self.start_calls += 1
        self._running = True

    def stop(self) -> None:
        self.stop_calls += 1
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_substrate_singletons(monkeypatch):
    """Clear process-wide manager/registry between tests for isolation."""
    set_default_manager(None)
    set_default_registry(None)
    yield
    set_default_manager(None)
    set_default_registry(None)


@pytest.fixture
def _isolated_state_dir(tmp_path, monkeypatch):
    """Point the BackgroundTaskManager's default state dir at a tmp path.

    The default manager honors ``SYNTHESIS_TASK_DIR`` specifically so tests
    can isolate task state without changing every unrelated home-directory
    lookup in the process.
    """
    state_dir = tmp_path / "tasks"
    monkeypatch.setenv("SYNTHESIS_TASK_DIR", str(state_dir))
    return state_dir


def _build_app_with_lifespan() -> FastAPI:
    """Build a fresh FastAPI app whose lifespan handler is the Phase 5 hook.

    We construct the app from scratch (rather than importing
    ``api.main.app``) so each test runs the lifespan once on a clean state.
    """
    # Import lazily so the substrate-reset fixture runs first.
    from api.main import lifespan

    app = FastAPI(lifespan=lifespan)
    return app


def _write_running_task_jsonl(state_dir: Path, task_id: str) -> Path:
    """Write a JSONL state file that looks like a ``running``-at-startup task.

    The recovery walker is supposed to flip this to ``crashed`` on the
    next process start.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / f"{task_id}.jsonl"
    record_dict = {
        "id": task_id,
        "name": "example-prior-run",
        "state": TaskState.RUNNING,
        "created_at_iso": "2026-05-13T12:00:00+00:00",
        "started_at_iso": "2026-05-13T12:00:01+00:00",
        "finished_at_iso": None,
        "result_summary": None,
        "error_summary": None,
        "webhook_url": None,
        "cancellation_requested": False,
        "timeout_seconds": None,
        "metadata": {"workspace": "example-workspace"},
    }
    line = {
        "event": "started",
        "timestamp_iso": "2026-05-13T12:00:01+00:00",
        "record": record_dict,
    }
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(line, sort_keys=True) + "\n")
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_lifespan_recovers_crashed_tasks_on_startup(_isolated_state_dir):
    """A running-at-startup JSONL gets a crashed line appended."""
    task_id = "prior-task-deadbeef"
    path = _write_running_task_jsonl(_isolated_state_dir, task_id)

    app = _build_app_with_lifespan()
    with TestClient(app):
        # Reading the file mid-context confirms recovery already happened.
        with open(path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        # 1 original "started" line + 1 "recovered_crashed" line.
        assert len(lines) == 2
        assert lines[0]["event"] == "started"
        assert lines[1]["event"] == "recovered_crashed"
        assert lines[1]["record"]["state"] == TaskState.CRASHED
        assert lines[1]["record"]["error_summary"] == "restart_during_run"

    # After teardown the file is unchanged from inside the context — the
    # crashed state is durable.
    with open(path, "r", encoding="utf-8") as f:
        post_lines = [json.loads(line) for line in f if line.strip()]
    assert len(post_lines) == 2


def test_lifespan_registers_default_task_factories(_isolated_state_dir):
    """Heartbeat and memory.consolidate_recent_idle land on the registry."""
    app = _build_app_with_lifespan()
    with TestClient(app):
        from synthesis_engine.tasks.registry import get_default_registry

        registry = get_default_registry()
        names = registry.names()
        assert "tasks.heartbeat" in names
        assert "memory.consolidate_recent_idle" in names


def test_scheduler_does_not_start_without_env_var(
    _isolated_state_dir, monkeypatch,
):
    """``RAGBOT_SCHEDULER`` unset → no SchedulerLoop is constructed."""
    monkeypatch.delenv("RAGBOT_SCHEDULER", raising=False)
    app = _build_app_with_lifespan()
    with TestClient(app):
        assert getattr(app.state, "scheduler_loop", None) is None


def test_scheduler_starts_when_env_var_truthy(
    _isolated_state_dir, monkeypatch,
):
    """``RAGBOT_SCHEDULER=1`` → SchedulerLoop is built and start() is called."""
    # Substitute the SchedulerLoop class with our fake so this test does
    # not depend on apscheduler being installed/wired in the env.
    import synthesis_engine.tasks.scheduler as scheduler_mod

    fakes_created: List[_FakeScheduler] = []

    def _factory(*args, **kwargs):
        f = _FakeScheduler()
        fakes_created.append(f)
        return f

    monkeypatch.setattr(scheduler_mod, "SchedulerLoop", _factory)
    monkeypatch.setenv("RAGBOT_SCHEDULER", "1")

    app = _build_app_with_lifespan()
    with TestClient(app):
        loop_obj = getattr(app.state, "scheduler_loop", None)
        assert loop_obj is not None
        assert isinstance(loop_obj, _FakeScheduler)
        assert loop_obj.start_calls == 1
        assert loop_obj.is_running is True
        # Also wired onto the tasks-router singleton.
        from api.routers import tasks as tasks_router

        assert tasks_router.get_scheduler_loop() is loop_obj


def test_shutdown_stops_scheduler_and_flushes_tracer(
    _isolated_state_dir, monkeypatch,
):
    """Scheduler.stop() and shutdown_tracer() both fire on context exit.

    The lifespan's tracer-ownership semantics: if a tracer is already
    initialised when startup runs, the lifespan defers to the host and
    leaves shutdown to the host as well. This test simulates a fresh
    process where the lifespan owns the tracer — that's the production
    path — by patching ``get_tracer_provider`` to report no prior
    provider. We then assert the lifespan calls ``shutdown_tracer``.
    """

    import synthesis_engine.tasks.scheduler as scheduler_mod
    import synthesis_engine.observability as observability_mod

    fakes_created: List[_FakeScheduler] = []

    def _factory(*args, **kwargs):
        f = _FakeScheduler()
        fakes_created.append(f)
        return f

    shutdown_calls = {"count": 0}

    def _fake_shutdown_tracer() -> None:
        shutdown_calls["count"] += 1

    monkeypatch.setattr(scheduler_mod, "SchedulerLoop", _factory)
    # Patch the symbol on the api.main import target — the lifespan handler
    # imports it lazily inside the function body.
    monkeypatch.setattr(
        observability_mod, "shutdown_tracer", _fake_shutdown_tracer,
    )
    # Simulate a fresh process: the lifespan's get_tracer_provider() call
    # returns None, so the lifespan concludes it owns the tracer and runs
    # shutdown_tracer at app exit. Without this patch, conftest's
    # session-scoped tracer is already in place and the lifespan correctly
    # defers — which is the behaviour we want in tests, just not the
    # scenario this specific test is asserting.
    monkeypatch.setattr(
        observability_mod, "get_tracer_provider", lambda: None,
    )
    monkeypatch.setenv("RAGBOT_SCHEDULER", "1")

    app = _build_app_with_lifespan()
    with TestClient(app):
        pass  # startup + shutdown run via the context manager.

    assert len(fakes_created) == 1
    fake = fakes_created[0]
    assert fake.stop_calls == 1
    assert fake.is_running is False
    assert shutdown_calls["count"] == 1
    # And the tasks-router singleton is cleared.
    from api.routers import tasks as tasks_router

    assert tasks_router.get_scheduler_loop() is None


def test_lifespan_idempotent_across_repeated_app_construction(
    _isolated_state_dir, monkeypatch,
):
    """Two TestClient contexts on two fresh apps both complete cleanly."""
    import synthesis_engine.tasks.scheduler as scheduler_mod

    fakes_created: List[_FakeScheduler] = []

    def _factory(*args, **kwargs):
        f = _FakeScheduler()
        fakes_created.append(f)
        return f

    monkeypatch.setattr(scheduler_mod, "SchedulerLoop", _factory)
    monkeypatch.setenv("RAGBOT_SCHEDULER", "1")

    for _ in range(2):
        app = _build_app_with_lifespan()
        with TestClient(app):
            pass

    assert len(fakes_created) == 2
    for f in fakes_created:
        assert f.start_calls == 1
        assert f.stop_calls == 1


def test_lifespan_survives_subsystem_failures(
    _isolated_state_dir, monkeypatch, caplog,
):
    """A scheduler that raises on start() is logged but doesn't crash startup."""
    import synthesis_engine.tasks.scheduler as scheduler_mod

    class _ExplodingScheduler:
        def start(self) -> None:
            raise RuntimeError("boom")

        def stop(self) -> None:
            pass

        @property
        def is_running(self) -> bool:
            return False

        @property
        def registered_ids(self) -> List[str]:
            return []

    monkeypatch.setattr(
        scheduler_mod, "SchedulerLoop", lambda: _ExplodingScheduler(),
    )
    monkeypatch.setenv("RAGBOT_SCHEDULER", "1")

    app = _build_app_with_lifespan()
    # Startup must NOT raise; the failure is logged as a warning.
    with caplog.at_level("WARNING"):
        with TestClient(app):
            pass
    # At least one warning mentioning the scheduler failure should appear.
    assert any(
        "Scheduler startup failed" in record.getMessage()
        for record in caplog.records
    ), f"Expected scheduler startup warning, got: {[r.getMessage() for r in caplog.records]}"
