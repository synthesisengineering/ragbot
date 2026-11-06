"""MCP Tasks support (2025-11-25 spec, SEP-1686).

Any MCP request can be promoted to a *task* — a durable handle the
client polls for status, fetches the payload from when complete, and
cancels if it changes its mind.

The mcp 2.x SDK ships the task message types but no client helpers for
them: the 1.x ``session.experimental`` namespace is gone, and the SDK's
method tables leave ``tasks/*`` out of the core protocol because later
protocol revisions move tasks into an extension. Ragbot's sessions
negotiate 2025-11-25 through ``ClientSession.initialize``, where tasks
are core methods, so this module sends the typed requests itself through
``ClientSession.send_request``. Callers get task semantics without
building JSON-RPC envelopes.

JSON-RPC methods covered:

* ``tools/call`` with ``task`` metadata — promote a tool call to a task;
  the server answers with a :class:`CreateTaskResult`.
* ``tasks/get`` — current status of a task.
* ``tasks/result`` — payload once the task has reached ``completed``.
* ``tasks/cancel`` — request cancellation; the server moves the task to
  ``cancelled`` (or leaves it terminal if it has already finished).
* ``tasks/list`` — enumerate the server's tasks.

Status pushes (``notifications/tasks/status``) arrive through the
session's message handler. :func:`poll_until_done` and :func:`subscribe`
poll instead, honoring the server's suggested interval, which works
whether or not a server pushes.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Dict, Optional, TypeVar

import anyio
from mcp import ClientSession
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    CancelTaskRequest,
    CancelTaskRequestParams,
    CancelTaskResult,
    CreateTaskResult,
    GetTaskPayloadRequest,
    GetTaskPayloadRequestParams,
    GetTaskRequest,
    GetTaskRequestParams,
    GetTaskResult,
    ListTasksRequest,
    ListTasksResult,
    PaginatedRequestParams,
    TaskMetadata,
)


T = TypeVar("T")


TaskStatus = str  # "working" | "input_required" | "completed" | "failed" | "cancelled"
TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})

# Poll interval when the server does not suggest one, in milliseconds.
DEFAULT_POLL_INTERVAL_MS = 500


async def call_tool_as_task(
    session: ClientSession,
    name: str,
    arguments: Optional[Dict[str, Any]] = None,
    *,
    ttl_ms: int = 60_000,
    meta: Optional[Dict[str, Any]] = None,
) -> CreateTaskResult:
    """Promote a tool call to a task.

    Returns a :class:`CreateTaskResult` whose ``task`` field carries the
    task id (and the rest of the task envelope). Use :func:`get_status`,
    :func:`get_result`, and :func:`cancel` to manage the lifetime, or
    :func:`poll_until_done` for the common "wait for the result" path.
    """
    params = CallToolRequestParams(
        name=name,
        arguments=arguments or {},
        task=TaskMetadata(ttl=ttl_ms),
        _meta=meta,
    )
    return await session.send_request(CallToolRequest(params=params), CreateTaskResult)


async def get_status(session: ClientSession, task_id: str) -> GetTaskResult:
    """Return the current status envelope for ``task_id``."""
    request = GetTaskRequest(params=GetTaskRequestParams(task_id=task_id))
    return await session.send_request(request, GetTaskResult)


async def get_result(
    session: ClientSession,
    task_id: str,
    result_type: type[T] = CallToolResult,  # type: ignore[assignment]
) -> T:
    """Fetch the payload for a completed task.

    ``result_type`` is the pydantic model the payload deserialises into
    — :class:`CallToolResult` for tool-promoted tasks, the appropriate
    result type for other primitives. The default is :class:`CallToolResult`
    because tool promotion is by far the most common path.
    """
    request = GetTaskPayloadRequest(params=GetTaskPayloadRequestParams(task_id=task_id))
    return await session.send_request(request, result_type)


async def cancel(session: ClientSession, task_id: str) -> CancelTaskResult:
    """Request cancellation of ``task_id``."""
    request = CancelTaskRequest(params=CancelTaskRequestParams(task_id=task_id))
    return await session.send_request(request, CancelTaskResult)


async def list_tasks(
    session: ClientSession,
    *,
    cursor: Optional[str] = None,
) -> ListTasksResult:
    """Return one page of tasks (the cursor lets you walk further)."""
    request = ListTasksRequest(
        params=PaginatedRequestParams(cursor=cursor) if cursor else None
    )
    return await session.send_request(request, ListTasksResult)


async def _poll(session: ClientSession, task_id: str) -> AsyncIterator[GetTaskResult]:
    """Yield status envelopes for ``task_id`` until one is terminal.

    Sleeps between polls for the server-suggested ``poll_interval``
    (milliseconds), or :data:`DEFAULT_POLL_INTERVAL_MS` when none is given.
    The terminal envelope is always the last value yielded.
    """
    while True:
        status = await get_status(session, task_id)
        yield status
        if status.status in TERMINAL_STATES:
            return
        interval_ms = status.poll_interval or DEFAULT_POLL_INTERVAL_MS
        await anyio.sleep(interval_ms / 1000)


async def poll_until_done(
    session: ClientSession,
    task_id: str,
    *,
    on_status: Optional[Callable[[GetTaskResult], Awaitable[None]]] = None,
) -> GetTaskResult:
    """Poll ``task_id`` until it reaches a terminal state.

    Honors the server-provided poll interval on each status response;
    falls back to 500 ms when the server doesn't suggest one. The
    optional ``on_status`` callback fires for every status, suitable for
    streaming progress updates to a UI.
    """
    last: Optional[GetTaskResult] = None
    async for status in _poll(session, task_id):
        last = status
        if on_status is not None:
            await on_status(status)
    assert last is not None, "polling yielded no statuses"
    return last


async def subscribe(
    session: ClientSession,
    task_id: str,
    *,
    on_status: Callable[[GetTaskResult], Awaitable[None]],
) -> AsyncIterator[GetTaskResult]:
    """Async iterator over status updates for ``task_id``.

    A thin convenience around the polling loop for callers that want the
    iterator shape (e.g., to ``async for`` over progress in a UI handler).
    """
    async for status in _poll(session, task_id):
        await on_status(status)
        yield status


__all__ = [
    "DEFAULT_POLL_INTERVAL_MS",
    "TaskStatus",
    "TERMINAL_STATES",
    "call_tool_as_task",
    "cancel",
    "get_result",
    "get_status",
    "list_tasks",
    "poll_until_done",
    "subscribe",
]
