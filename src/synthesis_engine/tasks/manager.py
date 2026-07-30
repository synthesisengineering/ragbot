"""BackgroundTaskManager — substrate primitive for long-running async work.

Design contract
===============

A task's lifecycle is recorded as an append-only JSONL stream at
``~/.synthesis/tasks/{task_id}.jsonl`` (one line per state transition).
Tests and isolated runtimes can redirect the root with
``SYNTHESIS_TASK_DIR``.
The newest line determines the current state; every previous line is
preserved as the transition history. The stream is the SINGLE SOURCE
OF TRUTH on disk; the in-memory dictionary is a cache that lets the
manager hand callers a fresh :class:`TaskRecord` without re-reading
JSONL on every poll.

Why JSONL rather than a database
--------------------------------

The substrate has to operate on a freshly-provisioned machine with no
running services. JSONL on disk is the same pattern the audit log
uses (``synthesis_engine.policy.audit``) — atomic append per line,
robust to mid-write crashes, easy to ``tail`` / ``grep`` for forensic
inspection.

Crash recovery
--------------

Every started task records a ``running`` line at start. If the process
exits without the manager writing a terminal line, that ``running``
entry remains the newest line on disk. On the next process start,
:meth:`recover_crashed_tasks` walks the directory, finds every task
whose newest line is ``running``, and appends a ``crashed`` line with
reason ``restart_during_run`` so callers polling
``GET /api/tasks/{id}`` see a deterministic terminal state. This is
the contract the user-facing UI relies on: a task that started can
ALWAYS be observed in a terminal state eventually.

Cancellation
------------

Cooperation, never force-kill. The manager sets
``task.cancellation_requested = True`` on the in-memory record and
sends an :class:`asyncio.CancelledError` to the underlying coroutine
only as a last resort (when ``force=True`` is passed). Tasks that
honour the cooperative protocol check ``cancellation_requested`` at
safe points and raise :class:`TaskCancelled`. The manager catches
that exception and records the ``cancelled`` terminal state.

Webhook delivery
----------------

When a task carries a ``webhook_url``, the manager POSTs a JSON body
to that URL on every TERMINAL transition. Delivery is best-effort:
network failures are logged but do not affect the recorded state.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)


logger = logging.getLogger("synthesis_engine.tasks.manager")


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------


class TaskState:
    """Canonical state strings.

    Plain string constants rather than an Enum because the JSONL format
    serializes state as a bare string and an Enum value adds zero
    correctness on top of a short whitelist. Test code can compare with
    plain string literals without depending on the import path of the
    enum.
    """

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    CRASHED = "crashed"

    ALL: Tuple[str, ...] = (
        QUEUED,
        RUNNING,
        SUCCEEDED,
        FAILED,
        CANCELLED,
        TIMED_OUT,
        CRASHED,
    )


TerminalStates: Tuple[str, ...] = (
    TaskState.SUCCEEDED,
    TaskState.FAILED,
    TaskState.CANCELLED,
    TaskState.TIMED_OUT,
    TaskState.CRASHED,
)


class TaskCancelled(Exception):
    """Raised by a cooperating task body when it observes a cancel request.

    The substrate translates this into the ``cancelled`` terminal state
    on the JSONL stream. Regular Python :class:`asyncio.CancelledError`
    is also accepted (caught and reported as ``cancelled``) so callers
    who rely on the native asyncio cancel protocol get the same
    terminal record.
    """


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class TaskRecord:
    """Immutable view of a task's state at a moment in time."""

    id: str
    name: str
    state: str
    created_at_iso: str
    started_at_iso: Optional[str] = None
    finished_at_iso: Optional[str] = None
    result_summary: Optional[str] = None
    error_summary: Optional[str] = None
    webhook_url: Optional[str] = None
    cancellation_requested: bool = False
    timeout_seconds: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TaskRecord":
        return cls(
            id=str(raw["id"]),
            name=str(raw.get("name", "")),
            state=str(raw.get("state", TaskState.QUEUED)),
            created_at_iso=str(raw.get("created_at_iso", "")),
            started_at_iso=raw.get("started_at_iso"),
            finished_at_iso=raw.get("finished_at_iso"),
            result_summary=raw.get("result_summary"),
            error_summary=raw.get("error_summary"),
            webhook_url=raw.get("webhook_url"),
            cancellation_requested=bool(raw.get("cancellation_requested", False)),
            timeout_seconds=raw.get("timeout_seconds"),
            metadata=dict(raw.get("metadata") or {}),
        )

    @property
    def is_terminal(self) -> bool:
        return self.state in TerminalStates


# ---------------------------------------------------------------------------
# Handle
# ---------------------------------------------------------------------------


class TaskHandle:
    """Caller-facing wrapper around a started task.

    The handle hides the underlying :class:`asyncio.Task` so callers
    cannot reach in and mutate state behind the manager's back. Use
    :meth:`await_result` to block until terminal, :meth:`cancel` to
    request cooperative cancellation, and :attr:`task_id` to identify
    the task in subsequent manager calls.
    """

    def __init__(
        self,
        manager: "BackgroundTaskManager",
        task_id: str,
        asyncio_task: asyncio.Task,
    ) -> None:
        self._manager = manager
        self._task_id = task_id
        self._asyncio_task = asyncio_task

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def asyncio_task(self) -> asyncio.Task:
        return self._asyncio_task

    async def await_result(self) -> TaskRecord:
        """Block until the task reaches a terminal state and return its record."""
        try:
            await self._asyncio_task
        except (asyncio.CancelledError, TaskCancelled, Exception):
            # Errors are already captured into the JSONL stream by the
            # manager's wrapper. Swallow here so callers observe the
            # terminal state via the record, not via re-raises.
            pass
        record = self._manager.get_task(self._task_id)
        if record is None:
            raise RuntimeError(
                f"Task {self._task_id} disappeared from the manager registry."
            )
        return record

    def cancel(self, reason: str = "user_cancelled") -> bool:
        return self._manager.cancel_task(self._task_id, reason=reason)


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


def _default_state_dir() -> Path:
    override = os.environ.get("SYNTHESIS_TASK_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".synthesis" / "tasks"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    """Append one JSON line to ``path`` atomically.

    Uses the same os.O_APPEND + single ``os.write`` pattern as the audit
    log so concurrent writers do not interleave bytes mid-line.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    fd = os.open(str(path), flags, 0o600)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    out: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                text = raw.strip()
                if not text:
                    continue
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping malformed task JSONL line in %s", path
                    )
                    continue
                if isinstance(parsed, dict):
                    out.append(parsed)
    except OSError as exc:
        logger.warning("Failed to read task JSONL at %s: %s", path, exc)
    return out


def _last_line(path: Path) -> Optional[Dict[str, Any]]:
    lines = _read_jsonl(path)
    return lines[-1] if lines else None


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


CoroFactory = Callable[["TaskRecord"], Awaitable[Any]]
"""Factory signature.

The factory receives the live :class:`TaskRecord` so the task body can
inspect ``cancellation_requested`` and other metadata without having
to capture the manager via closure. Implementations are expected to
periodically check ``record.cancellation_requested`` at safe points
and raise :class:`TaskCancelled` when it flips to True.
"""


class BackgroundTaskManager:
    """Substrate-level manager for long-running asyncio tasks.

    A single manager instance is the canonical entry point for one
    process. Tests construct a fresh manager per test (with a tmp_path
    state dir) so JSONL streams do not leak between tests.
    """

    def __init__(
        self,
        *,
        state_dir: Optional[Path] = None,
        notifier: Optional[Any] = None,
        http_post: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self._state_dir = Path(state_dir) if state_dir else _default_state_dir()
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._records: Dict[str, TaskRecord] = {}
        self._asyncio_tasks: Dict[str, asyncio.Task] = {}
        self._notifier = notifier
        self._http_post = http_post or _default_http_post
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()

    # ----- public API --------------------------------------------------------

    @property
    def state_dir(self) -> Path:
        """Directory that owns this manager's durable task streams."""
        return self._state_dir

    def start_task(
        self,
        name: str,
        coro_factory: CoroFactory,
        *,
        timeout_seconds: Optional[int] = None,
        webhook_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TaskHandle:
        """Register and start a new task. Returns the handle immediately."""
        task_id = uuid.uuid4().hex
        record = TaskRecord(
            id=task_id,
            name=name,
            state=TaskState.RUNNING,
            created_at_iso=_now_iso(),
            started_at_iso=_now_iso(),
            timeout_seconds=timeout_seconds,
            webhook_url=webhook_url,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._records[task_id] = record
        self._write_transition(record, event="started")

        async def _runner() -> None:
            await self._drive(record, coro_factory)

        asyncio_task = asyncio.create_task(_runner(), name=f"task:{name}:{task_id[:8]}")
        with self._lock:
            self._asyncio_tasks[task_id] = asyncio_task

        def _on_done(_t: asyncio.Task) -> None:
            with self._lock:
                self._asyncio_tasks.pop(task_id, None)

        asyncio_task.add_done_callback(_on_done)
        return TaskHandle(self, task_id, asyncio_task)

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        """Return the most-recent state record for ``task_id`` or None."""
        with self._lock:
            cached = self._records.get(task_id)
        if cached is not None:
            return cached
        record = self._reconstruct(task_id)
        if record is not None:
            with self._lock:
                self._records[task_id] = record
        return record

    def list_tasks(
        self,
        state_filter: Optional[str] = None,
        limit: int = 100,
    ) -> List[TaskRecord]:
        """Return every known task, optionally filtered by state."""
        records: List[TaskRecord] = []
        # Copy the in-memory cache first.
        with self._lock:
            cached_ids = set(self._records)
            for rec in self._records.values():
                records.append(rec)
        # Walk the state dir for anything we haven't cached yet.
        if self._state_dir.is_dir():
            for entry in self._state_dir.iterdir():
                if not entry.is_file() or not entry.name.endswith(".jsonl"):
                    continue
                tid = entry.stem
                if tid in cached_ids:
                    continue
                rec = self._reconstruct(tid)
                if rec is not None:
                    records.append(rec)
        if state_filter is not None:
            records = [r for r in records if r.state == state_filter]
        # Newest-first by created_at_iso.
        records.sort(key=lambda r: r.created_at_iso, reverse=True)
        return records[:limit]

    def get_history(self, task_id: str) -> List[Dict[str, Any]]:
        """Return every JSONL line for ``task_id`` in chronological order."""
        return _read_jsonl(self._jsonl_path(task_id))

    def cancel_task(self, task_id: str, *, reason: str = "user_cancelled") -> bool:
        """Request cooperative cancellation.

        Returns True if the task was running and a cancellation has
        been requested; False if the task is unknown or already
        terminal.
        """
        with self._lock:
            record = self._records.get(task_id)
        if record is None:
            record = self._reconstruct(task_id)
            if record is not None:
                with self._lock:
                    self._records[task_id] = record
        if record is None or record.is_terminal:
            return False
        record.cancellation_requested = True
        record.metadata.setdefault("cancel_reason", reason)
        with self._lock:
            self._records[task_id] = record
        # Append a soft-cancel event line so observers see the request
        # even before the cooperating coroutine acknowledges it.
        _atomic_append_jsonl(
            self._jsonl_path(task_id),
            {
                "event": "cancel_requested",
                "timestamp_iso": _now_iso(),
                "reason": reason,
                "record": record.to_dict(),
            },
        )
        return True

    def recover_crashed_tasks(self) -> List[TaskRecord]:
        """Mark every running-at-startup record as ``crashed``.

        Walks the JSONL state directory at construction time of a new
        process and rewrites any task whose newest line is
        ``state == running`` to ``crashed`` with reason
        ``restart_during_run``. Returns the list of records that were
        rewritten.
        """
        recovered: List[TaskRecord] = []
        if not self._state_dir.is_dir():
            return recovered
        for entry in self._state_dir.iterdir():
            if not entry.is_file() or not entry.name.endswith(".jsonl"):
                continue
            task_id = entry.stem
            record = self._reconstruct(task_id)
            if record is None:
                continue
            if record.state == TaskState.RUNNING:
                record.state = TaskState.CRASHED
                record.finished_at_iso = _now_iso()
                record.error_summary = "restart_during_run"
                with self._lock:
                    self._records[task_id] = record
                self._write_transition(record, event="recovered_crashed")
                recovered.append(record)
        return recovered

    # ----- internal driving --------------------------------------------------

    async def _drive(self, record: TaskRecord, coro_factory: CoroFactory) -> None:
        timeout = record.timeout_seconds
        try:
            coro = coro_factory(record)
            if not asyncio.iscoroutine(coro) and not isinstance(coro, asyncio.Future):
                # Allow callable returning awaitable (defensive).
                raise TypeError(
                    "coro_factory must return an awaitable (coroutine or Future)."
                )
            if timeout and timeout > 0:
                try:
                    result = await asyncio.wait_for(coro, timeout=timeout)
                except asyncio.TimeoutError:
                    await self._transition_terminal(
                        record,
                        state=TaskState.TIMED_OUT,
                        error_summary=f"timed out after {timeout}s",
                    )
                    return
            else:
                result = await coro
        except TaskCancelled as exc:
            await self._transition_terminal(
                record,
                state=TaskState.CANCELLED,
                error_summary=str(exc) or "cancelled",
            )
            return
        except asyncio.CancelledError as exc:
            await self._transition_terminal(
                record,
                state=TaskState.CANCELLED,
                error_summary=str(exc) or "cancelled",
            )
            return
        except Exception as exc:  # noqa: BLE001 — terminal state catch-all
            logger.exception("Background task %s failed", record.id)
            await self._transition_terminal(
                record,
                state=TaskState.FAILED,
                error_summary=repr(exc),
            )
            return
        await self._transition_terminal(
            record,
            state=TaskState.SUCCEEDED,
            result_summary=_summarise_result(result),
        )

    async def _transition_terminal(
        self,
        record: TaskRecord,
        *,
        state: str,
        result_summary: Optional[str] = None,
        error_summary: Optional[str] = None,
    ) -> None:
        record.state = state
        record.finished_at_iso = _now_iso()
        if result_summary is not None:
            record.result_summary = result_summary
        if error_summary is not None:
            record.error_summary = error_summary
        with self._lock:
            self._records[record.id] = record
        self._write_transition(record, event=state)
        await self._dispatch_webhook(record)
        await self._dispatch_notifier(record)

    def _write_transition(self, record: TaskRecord, *, event: str) -> None:
        _atomic_append_jsonl(
            self._jsonl_path(record.id),
            {
                "event": event,
                "timestamp_iso": _now_iso(),
                "record": record.to_dict(),
            },
        )

    async def _dispatch_webhook(self, record: TaskRecord) -> None:
        if not record.webhook_url:
            return
        try:
            await self._http_post(
                record.webhook_url,
                {
                    "task_id": record.id,
                    "name": record.name,
                    "state": record.state,
                    "result_summary": record.result_summary,
                    "error_summary": record.error_summary,
                    "finished_at_iso": record.finished_at_iso,
                },
            )
        except Exception as exc:  # noqa: BLE001 — webhook failure must not propagate
            logger.warning(
                "Webhook delivery for task %s failed: %s",
                record.id,
                exc,
            )

    async def _dispatch_notifier(self, record: TaskRecord) -> None:
        if self._notifier is None:
            return
        try:
            await self._notifier.notify(record, event=record.state)
        except Exception as exc:  # noqa: BLE001 — notifier failure must not propagate
            logger.warning(
                "Notifier dispatch for task %s failed: %s",
                record.id,
                exc,
            )

    # ----- internals ---------------------------------------------------------

    def _jsonl_path(self, task_id: str) -> Path:
        return self._state_dir / f"{task_id}.jsonl"

    def _reconstruct(self, task_id: str) -> Optional[TaskRecord]:
        path = self._jsonl_path(task_id)
        last = _last_line(path)
        if last is None:
            return None
        raw = last.get("record")
        if not isinstance(raw, dict):
            return None
        return TaskRecord.from_dict(raw)


# ---------------------------------------------------------------------------
# Result summarisation
# ---------------------------------------------------------------------------


def _summarise_result(value: Any, limit: int = 500) -> str:
    """Stringify a task result for the JSONL record.

    Tasks may return arbitrary objects; we capture a short stringified
    form so the record stays small. The caller's webhook/notifier
    payload uses this same summary.
    """
    try:
        if value is None:
            return "ok"
        if isinstance(value, str):
            text = value
        else:
            text = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001 — never let summarisation crash the driver
        text = repr(value)
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


# ---------------------------------------------------------------------------
# Webhook default HTTP client
# ---------------------------------------------------------------------------


async def _default_http_post(url: str, payload: Dict[str, Any]) -> None:
    """Best-effort HTTP POST using stdlib only.

    We avoid pulling ``httpx`` or ``aiohttp`` into the substrate's hard
    deps; webhook delivery is best-effort by design. The post runs in a
    thread via ``asyncio.to_thread`` so the event loop is never blocked.
    """
    import urllib.error
    import urllib.request

    def _do_post() -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"content-type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                resp.read()
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"webhook POST failed: {exc}") from exc

    await asyncio.to_thread(_do_post)


# ---------------------------------------------------------------------------
# Process-singleton
# ---------------------------------------------------------------------------


_DEFAULT_MANAGER: Optional[BackgroundTaskManager] = None


def get_default_manager() -> Optional[BackgroundTaskManager]:
    """Return the installed process-wide :class:`BackgroundTaskManager`."""
    return _DEFAULT_MANAGER


def set_default_manager(manager: Optional[BackgroundTaskManager]) -> None:
    """Install (or clear) the process-wide :class:`BackgroundTaskManager`."""
    global _DEFAULT_MANAGER
    _DEFAULT_MANAGER = manager


def default_manager() -> BackgroundTaskManager:
    """Resolve the process-wide manager, constructing one on first use."""
    manager = get_default_manager()
    if manager is None:
        manager = BackgroundTaskManager()
        set_default_manager(manager)
    return manager


__all__ = [
    "BackgroundTaskManager",
    "TaskCancelled",
    "TaskHandle",
    "TaskRecord",
    "TaskState",
    "TerminalStates",
    "default_manager",
    "get_default_manager",
    "set_default_manager",
]
