"""MCP Tasks wrappers (``synthesis_engine.mcp.tasks``) against mcp 2.x.

The 2.x SDK has no client helpers for tasks, so the wrappers send the
typed 2025-11-25 requests themselves. ``tasks/get``, ``tasks/result``,
``tasks/list`` and ``tasks/cancel`` run over real JSON-RPC against an
in-process low-level server on a 2025-11-25 connection, which is what
``ClientSession.initialize`` negotiates in Ragbot's registry. Task
creation goes through ``tools/call``, whose result a 2.x server always
validates as a ``CallToolResult``; it is checked as a request contract
instead.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

import pytest

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import mcp.types as t  # noqa: E402
from mcp import Client  # noqa: E402
from mcp.server.lowlevel import Server  # noqa: E402

from synthesis_engine.mcp import tasks  # noqa: E402

STAMP = "2026-11-06T00:00:00Z"


def _task(task_id: str, status: str, poll_interval: int = 1) -> Dict[str, Any]:
    return dict(
        task_id=task_id,
        status=status,
        created_at=STAMP,
        last_updated_at=STAMP,
        ttl=60_000,
        poll_interval=poll_interval,
    )


class TaskServer:
    """In-process server whose task ``slow`` completes on its third poll."""

    def __init__(self) -> None:
        self.polls = 0
        self.cancelled: List[str] = []
        self.list_cursors: List[Any] = []
        self.server = Server("tasks-fixture")
        self.server.add_request_handler("tasks/get", t.GetTaskRequestParams, self.get)
        self.server.add_request_handler("tasks/result", t.GetTaskPayloadRequestParams, self.result)
        self.server.add_request_handler("tasks/list", t.PaginatedRequestParams, self.list)
        self.server.add_request_handler("tasks/cancel", t.CancelTaskRequestParams, self.cancel)

    async def get(self, ctx, params: t.GetTaskRequestParams) -> t.GetTaskResult:
        self.polls += 1
        status = "completed" if self.polls >= 3 else "working"
        return t.GetTaskResult(**_task(params.task_id, status))

    async def result(self, ctx, params: t.GetTaskPayloadRequestParams) -> t.CallToolResult:
        return t.CallToolResult(
            content=[t.TextContent(type="text", text=f"done {params.task_id}")],
            structured_content={"task": params.task_id},
        )

    async def list(self, ctx, params: t.PaginatedRequestParams) -> t.ListTasksResult:
        self.list_cursors.append(params.cursor)
        if params.cursor is None:
            return t.ListTasksResult(tasks=[t.Task(**_task("a", "working"))], next_cursor="page-2")
        return t.ListTasksResult(tasks=[t.Task(**_task("b", "completed"))])

    async def cancel(self, ctx, params: t.CancelTaskRequestParams) -> t.CancelTaskResult:
        self.cancelled.append(params.task_id)
        return t.CancelTaskResult(**_task(params.task_id, "cancelled"))


@pytest.fixture
def fixture_server() -> TaskServer:
    return TaskServer()


def _client(fixture: TaskServer) -> Client:
    # "legacy" is the initialize handshake, i.e. the 2025-11-25 protocol.
    return Client(fixture.server, raise_exceptions=True, mode="legacy")


@pytest.mark.asyncio
async def test_connection_negotiates_the_handshake_protocol(fixture_server):
    async with _client(fixture_server) as client:
        assert client.session.protocol_version == "2025-11-25"


@pytest.mark.asyncio
async def test_get_status_returns_the_task_envelope(fixture_server):
    async with _client(fixture_server) as client:
        status = await tasks.get_status(client.session, "slow")
    assert status.task_id == "slow"
    assert status.status == "working"


@pytest.mark.asyncio
async def test_poll_until_done_reports_every_status_and_returns_the_terminal_one(fixture_server):
    seen: List[str] = []

    async def on_status(status: t.GetTaskResult) -> None:
        seen.append(status.status)

    async with _client(fixture_server) as client:
        final = await tasks.poll_until_done(client.session, "slow", on_status=on_status)
    assert final.status == "completed"
    assert seen == ["working", "working", "completed"]


@pytest.mark.asyncio
async def test_subscribe_yields_until_terminal(fixture_server):
    pushed: List[str] = []

    async def on_status(status: t.GetTaskResult) -> None:
        pushed.append(status.status)

    async with _client(fixture_server) as client:
        yielded = [s.status async for s in tasks.subscribe(client.session, "slow", on_status=on_status)]
    assert yielded == ["working", "working", "completed"]
    assert pushed == yielded


@pytest.mark.asyncio
async def test_get_result_parses_the_payload_into_the_requested_type(fixture_server):
    async with _client(fixture_server) as client:
        payload = await tasks.get_result(client.session, "slow")
    assert isinstance(payload, t.CallToolResult)
    assert payload.content[0].text == "done slow"
    assert payload.structured_content == {"task": "slow"}


@pytest.mark.asyncio
async def test_list_tasks_passes_the_cursor_through(fixture_server):
    async with _client(fixture_server) as client:
        first = await tasks.list_tasks(client.session)
        second = await tasks.list_tasks(client.session, cursor=first.next_cursor)
    assert [x.task_id for x in first.tasks] == ["a"]
    assert [x.task_id for x in second.tasks] == ["b"]
    assert fixture_server.list_cursors == [None, "page-2"]


@pytest.mark.asyncio
async def test_cancel_reaches_the_server(fixture_server):
    async with _client(fixture_server) as client:
        result = await tasks.cancel(client.session, "slow")
    assert result.status == "cancelled"
    assert fixture_server.cancelled == ["slow"]


@pytest.mark.asyncio
async def test_call_tool_as_task_sends_task_metadata_and_expects_a_create_result():
    captured: Dict[str, Any] = {}

    class RecordingSession:
        async def send_request(self, request, result_type):
            captured["request"] = request
            captured["result_type"] = result_type
            return t.CreateTaskResult(task=t.Task(**_task("new", "working")))

    result = await tasks.call_tool_as_task(
        RecordingSession(), "long_job", {"n": 3}, ttl_ms=1234, meta={"origin": "test"}
    )
    request = captured["request"]
    wire = request.model_dump(by_alias=True, exclude_none=True)
    assert wire["method"] == "tools/call"
    assert wire["params"]["name"] == "long_job"
    assert wire["params"]["arguments"] == {"n": 3}
    assert wire["params"]["task"] == {"ttl": 1234}
    assert wire["params"]["_meta"] == {"origin": "test"}
    assert captured["result_type"] is t.CreateTaskResult
    assert result.task.task_id == "new"
