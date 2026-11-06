"""Scheduled memory consolidation — the "dreaming" pattern.

Where :func:`synthesis_engine.memory.consolidation.consolidate_session` is
the single-call distillation primitive, :class:`MemoryConsolidator` is
the scheduled orchestrator that wraps it. It runs between sessions,
discovers candidate sessions, distils each one, writes the extracted
facts into the entity graph with provenance that pins the run, and
records an audit entry per pass.

Pattern (May 2026 research preview): the agent "dreams" between
conversations. While the agent is idle, the consolidator reviews what
was learned, packs durable facts into the entity graph (Tier 2 of the
three-tier memory stack), and surfaces them in subsequent retrieval —
the agent wakes up remembering yesterday's facts as long-term knowledge,
not as a verbatim transcript it has to re-read.

Design choices
==============

Provenance shape
    Every consolidated fact carries
    ``consolidation:session={id}:model={id}:run_at={iso}``. The model id
    and timestamp let the same session be re-consolidated with a stronger
    model later; both passes' provenance accumulate on the relations
    rather than overwriting.

Idempotency
    Running the same session through the same model id is a no-op for
    the writer: the existing relation row carries an identical
    provenance string, and the consolidator skips the upsert. Running
    again with a different model id IS allowed; the new pass adds a
    fresh row with its own provenance.

Discovery
    :meth:`consolidate_recent_idle` is the cron-friendly entry point.
    It walks the checkpoint store, finds sessions whose latest
    checkpoint is older than ``idle_threshold_hours`` and that don't
    yet have a consolidation entry for the current model id, and
    consolidates each. Suitable for a four-times-a-day routine.

Audit
    Every consolidation pass — single-session or batch element —
    records one :class:`AuditEntry` with op_type
    ``memory_consolidation``. The history endpoint and the
    ``ragbot memory consolidation-history`` CLI command tail these
    entries.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from ..llm import LLMBackend
from ..policy.audit import AuditEntry, record as record_audit
from .base import Memory
from .consolidation import consolidate_session as _consolidate_one
from .models import Provenance

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


@dataclass
class ConsolidationReport:
    """Outcome of one session's consolidation pass."""

    session_id: str
    workspace: str
    model_id: str
    run_at_iso: str
    entities_added: int = 0
    relations_added: int = 0
    entities_existing: int = 0
    relations_existing: int = 0
    skipped: bool = False
    skip_reason: Optional[str] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "workspace": self.workspace,
            "model_id": self.model_id,
            "run_at_iso": self.run_at_iso,
            "entities_added": self.entities_added,
            "relations_added": self.relations_added,
            "entities_existing": self.entities_existing,
            "relations_existing": self.relations_existing,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class BatchReport:
    """Aggregate outcome of a multi-session consolidation pass."""

    started_at_iso: str
    finished_at_iso: str
    model_id: str
    dry_run: bool = False
    sessions_consolidated: int = 0
    sessions_skipped: int = 0
    sessions_errored: int = 0
    total_entities_added: int = 0
    total_relations_added: int = 0
    duration_seconds: float = 0.0
    per_session: List[ConsolidationReport] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "started_at_iso": self.started_at_iso,
            "finished_at_iso": self.finished_at_iso,
            "model_id": self.model_id,
            "dry_run": self.dry_run,
            "sessions_consolidated": self.sessions_consolidated,
            "sessions_skipped": self.sessions_skipped,
            "sessions_errored": self.sessions_errored,
            "total_entities_added": self.total_entities_added,
            "total_relations_added": self.total_relations_added,
            "duration_seconds": self.duration_seconds,
            "per_session": [r.to_dict() for r in self.per_session],
        }


# ---------------------------------------------------------------------------
# Checkpoint discovery — protocol-friendly so tests inject in-memory stores
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CheckpointDiscovery:
    """A discovered session candidate from the checkpoint store."""

    session_id: str
    last_modified: datetime
    checkpoint_count: int


def _discover_checkpoints_fs(
    base_dir: Path, since: Optional[datetime] = None, until: Optional[datetime] = None
) -> List[_CheckpointDiscovery]:
    """Walk the filesystem checkpoint store and return session candidates.

    For each ``base_dir/{task_id}/`` directory we record the session id
    (the directory name) and the modification time of the
    newest checkpoint file inside.
    """
    if not base_dir.is_dir():
        return []
    found: List[_CheckpointDiscovery] = []
    for task_dir in base_dir.iterdir():
        if not task_dir.is_dir():
            continue
        latest_mtime = 0.0
        count = 0
        for entry in task_dir.iterdir():
            if not entry.is_file():
                continue
            if not re.match(r"^\d{4,}\.json$", entry.name):
                continue
            try:
                stat = entry.stat()
            except OSError:
                continue
            if stat.st_mtime > latest_mtime:
                latest_mtime = stat.st_mtime
            count += 1
        if count == 0:
            continue
        last_dt = datetime.fromtimestamp(latest_mtime, tz=timezone.utc)
        if since is not None and last_dt < since:
            continue
        if until is not None and last_dt > until:
            continue
        found.append(
            _CheckpointDiscovery(
                session_id=task_dir.name,
                last_modified=last_dt,
                checkpoint_count=count,
            )
        )
    found.sort(key=lambda d: d.last_modified)
    return found


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


_PROVENANCE_RE = re.compile(
    r"^consolidation:session=([^:]+):model=([^:]+):run_at=(.+)$"
)


def _build_provenance_source(session_id: str, model_id: str, run_at_iso: str) -> str:
    """Render the canonical provenance string for a consolidation pass."""
    return (
        f"consolidation:session={session_id}"
        f":model={model_id}"
        f":run_at={run_at_iso}"
    )


def _parse_provenance_source(source: str) -> Optional[Tuple[str, str, str]]:
    """Inverse of :func:`_build_provenance_source`; None on non-matches."""
    m = _PROVENANCE_RE.match(source or "")
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def _existing_consolidation_runs(
    memory: Memory,
    workspace: str,
    session_id: str,
) -> List[Tuple[str, str]]:
    """Return list of ``(model_id, run_at_iso)`` already recorded.

    Scans the workspace's relations and returns every consolidation
    provenance string we find for the given session id. Used for the
    "skip if this (session, model) pair already consolidated" guard.
    """
    runs: List[Tuple[str, str]] = []
    # We don't know which entities are involved without first running
    # the consolidator; instead, walk relations in the workspace and
    # filter by provenance source. The graph is small enough for this
    # in the workspaces consolidation targets (per-session counts in
    # the low hundreds, not millions).
    try:
        seed_ids: List[UUID] = []
        # Use list_entities + query_graph: list workspace entities and
        # use them as seeds. The depth=1 graph traversal will surface
        # every relation that touches at least one workspace entity.
        for ent in memory.list_entities(workspace, limit=500, offset=0):
            if ent.id is not None:
                seed_ids.append(ent.id)
        if not seed_ids:
            return []
        relations = memory.query_graph(
            workspace, seed_entity_ids=seed_ids, depth=1, limit=1000
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "consolidation idempotency lookup failed for session=%s: %s",
            session_id,
            exc,
        )
        return []
    for rel in relations:
        prov = getattr(rel, "provenance", None)
        if prov is None:
            continue
        parsed = _parse_provenance_source(prov.source or "")
        if parsed is None:
            continue
        sid, mid, run_at = parsed
        if sid == session_id:
            runs.append((mid, run_at))
    return runs


# ---------------------------------------------------------------------------
# Memory consolidator
# ---------------------------------------------------------------------------


class MemoryConsolidator:
    """Scheduled wrapper around :func:`consolidate_session`.

    The consolidator is the substrate-level scheduler primitive. A cron
    runner (Phase 4 Agent D) drives it via :meth:`consolidate_recent_idle`;
    the REST API drives it via :meth:`consolidate_session` or
    :meth:`consolidate_batch`; the CLI drives it via the same methods.

    Construction parameters are intentionally explicit so tests can
    swap the LLM backend (FakeLLMBackend), the checkpoint store (the
    filesystem store under a tmp_path), and the memory backend (an
    in-memory stub).
    """

    @property
    def default_model_id(self) -> str:
        """Model used when a caller names none: the default provider's bulk tier."""
        from ..config import get_default_model_for_tier

        return get_default_model_for_tier("bulk")

    def __init__(
        self,
        memory: Memory,
        *,
        llm_backend: Optional[LLMBackend] = None,
        checkpoint_base_dir: Optional[Path] = None,
        default_workspace: str = "personal",
        clock: Optional[Any] = None,
    ) -> None:
        self.memory = memory
        self.llm_backend = llm_backend
        self.checkpoint_base_dir = (
            Path(checkpoint_base_dir) if checkpoint_base_dir else None
        )
        self.default_workspace = default_workspace
        # ``clock`` is a callable returning a tz-aware datetime; lets
        # tests freeze time without monkeypatching datetime at module
        # scope.
        self._clock = clock or (lambda: datetime.now(tz=timezone.utc))

    # ------------------------------------------------------------------
    # Single-session consolidation
    # ------------------------------------------------------------------

    async def consolidate_session(
        self,
        session_id: str,
        model_id: Optional[str] = None,
        *,
        workspace: Optional[str] = None,
        dry_run: bool = False,
    ) -> ConsolidationReport:
        """Consolidate one session into the entity graph.

        Reads the session payload from the memory backend's
        :meth:`Memory.get_session`, then distils via
        :func:`consolidate_session` with a fresh provenance string keyed
        to ``model_id`` and ``run_at``. Writes are skipped on
        ``dry_run=True`` so callers can preview what would be added.
        """
        run_at_dt = self._clock()
        run_at_iso = run_at_dt.isoformat()
        effective_model = model_id or self.default_model_id
        session = self.memory.get_session(session_id)
        ws = workspace or (
            session.workspace if session is not None else self.default_workspace
        )

        report = ConsolidationReport(
            session_id=session_id,
            workspace=ws,
            model_id=effective_model,
            run_at_iso=run_at_iso,
        )
        started = run_at_dt
        try:
            if session is None:
                report.skipped = True
                report.skip_reason = "session_not_found"
                return report

            # Idempotency guard: skip if a prior run with the same
            # (session, model) pair has already landed. We don't gate on
            # run_at — re-running with the same model produces an
            # identical provenance string and the same extraction, so
            # there is nothing to add.
            prior_runs = _existing_consolidation_runs(self.memory, ws, session_id)
            same_model_runs = [r for r in prior_runs if r[0] == effective_model]
            if same_model_runs and not dry_run:
                report.skipped = True
                report.skip_reason = "already_consolidated_with_same_model"
                report.entities_existing = sum(
                    1 for _ in self.memory.list_entities(ws, limit=500)
                )
                return report

            # In dry_run mode we still call the extractor (and the LLM)
            # but skip the writes. We do this by composing a memory
            # facade that captures upserts without committing them. The
            # simplest implementation: use a transient TransparentMemory
            # wrapper that records but doesn't forward.
            if dry_run:
                facade = _DryRunMemory(self.memory)
                extraction = _consolidate_one(
                    facade,
                    session_id,
                    ws,
                    llm_backend=self.llm_backend,
                    model=effective_model,
                )
                report.entities_added = facade.entities_attempted
                report.relations_added = facade.relations_attempted
                report.duration_seconds = (self._clock() - started).total_seconds()
                return report

            # Provenance pinning happens AFTER consolidate_session by
            # rewriting the source strings on the relations and
            # attribute values written. The existing consolidation
            # helper writes provenance ``consolidation:session={id}``;
            # we want the richer model-and-run-at form. We achieve this
            # via a thin wrapper Memory that intercepts upserts and
            # restamps provenance.
            wrapper = _RestampedMemory(
                self.memory,
                provenance_source=_build_provenance_source(
                    session_id, effective_model, run_at_iso
                ),
            )
            _consolidate_one(
                wrapper,
                session_id,
                ws,
                llm_backend=self.llm_backend,
                model=effective_model,
            )
            report.entities_added = wrapper.entities_added
            report.relations_added = wrapper.relations_added
            report.entities_existing = wrapper.entities_existing
            report.relations_existing = wrapper.relations_existing

            # Record audit. The audit entry includes both counts AND
            # the rendered provenance source so a forensic reader can
            # join the audit log back to the entity graph by string.
            try:
                record_audit(
                    AuditEntry.build(
                        op_type="memory_consolidation",
                        workspaces=[ws],
                        tools=[],
                        model_id=effective_model,
                        outcome="allowed",
                        args_summary="{}",
                        metadata={
                            "session_id": session_id,
                            "entities_added": report.entities_added,
                            "relations_added": report.relations_added,
                            "entities_existing": report.entities_existing,
                            "relations_existing": report.relations_existing,
                            "run_at_iso": run_at_iso,
                            "provenance_source": wrapper.provenance_source,
                            "dry_run": False,
                        },
                        timestamp=run_at_dt,
                    )
                )
            except Exception as exc:  # pragma: no cover - audit must not break the pass
                logger.warning(
                    "audit record failed for consolidation (session=%s): %s",
                    session_id,
                    exc,
                )
        except Exception as exc:
            logger.exception(
                "consolidate_session(%s) failed", session_id
            )
            report.error = repr(exc)
            return report
        finally:
            report.duration_seconds = max(
                (self._clock() - started).total_seconds(), 0.0
            )

        return report

    # ------------------------------------------------------------------
    # Batch consolidation
    # ------------------------------------------------------------------

    async def consolidate_batch(
        self,
        since_iso: Optional[str] = None,
        until_iso: Optional[str] = None,
        model_id: Optional[str] = None,
        *,
        workspace: Optional[str] = None,
        dry_run: bool = False,
        session_ids: Optional[List[str]] = None,
    ) -> BatchReport:
        """Consolidate every session whose checkpoint mtime is in window.

        ``session_ids`` may be passed directly to bypass discovery —
        useful for tests and for the API endpoint when an explicit list
        is supplied.
        """
        started = self._clock()
        effective_model = model_id or self.default_model_id
        ws = workspace or self.default_workspace

        if session_ids is None:
            since_dt = _parse_iso(since_iso) if since_iso else None
            until_dt = _parse_iso(until_iso) if until_iso else None
            candidates = self._discover_candidates(since=since_dt, until=until_dt)
            session_ids = [c.session_id for c in candidates]

        report = BatchReport(
            started_at_iso=started.isoformat(),
            finished_at_iso=started.isoformat(),
            model_id=effective_model,
            dry_run=dry_run,
        )

        for sid in session_ids:
            r = await self.consolidate_session(
                sid, model_id=effective_model, workspace=ws, dry_run=dry_run
            )
            report.per_session.append(r)
            if r.error is not None:
                report.sessions_errored += 1
            elif r.skipped:
                report.sessions_skipped += 1
            else:
                report.sessions_consolidated += 1
                report.total_entities_added += r.entities_added
                report.total_relations_added += r.relations_added

        finished = self._clock()
        report.finished_at_iso = finished.isoformat()
        report.duration_seconds = max(
            (finished - started).total_seconds(), 0.0
        )
        return report

    # ------------------------------------------------------------------
    # Idle-based discovery
    # ------------------------------------------------------------------

    async def consolidate_recent_idle(
        self,
        idle_threshold_hours: float = 4.0,
        model_id: Optional[str] = None,
        *,
        workspace: Optional[str] = None,
        dry_run: bool = False,
    ) -> BatchReport:
        """Consolidate sessions whose latest checkpoint is older than threshold.

        Skips sessions whose latest checkpoint is more recent than the
        threshold (still active) and sessions that have already been
        consolidated with the same model id (idempotency).
        """
        started = self._clock()
        effective_model = model_id or self.default_model_id
        ws = workspace or self.default_workspace
        cutoff = started - timedelta(hours=max(idle_threshold_hours, 0.0))

        candidates = self._discover_candidates(since=None, until=cutoff)

        report = BatchReport(
            started_at_iso=started.isoformat(),
            finished_at_iso=started.isoformat(),
            model_id=effective_model,
            dry_run=dry_run,
        )

        for cand in candidates:
            r = await self.consolidate_session(
                cand.session_id,
                model_id=effective_model,
                workspace=ws,
                dry_run=dry_run,
            )
            report.per_session.append(r)
            if r.error is not None:
                report.sessions_errored += 1
            elif r.skipped:
                report.sessions_skipped += 1
            else:
                report.sessions_consolidated += 1
                report.total_entities_added += r.entities_added
                report.total_relations_added += r.relations_added

        finished = self._clock()
        report.finished_at_iso = finished.isoformat()
        report.duration_seconds = max(
            (finished - started).total_seconds(), 0.0
        )
        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _discover_candidates(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> List[_CheckpointDiscovery]:
        """Return session candidates from the configured checkpoint store.

        Subclasses may override; the default walks the filesystem store
        at ``self.checkpoint_base_dir`` (or
        ``~/.synthesis/agent-checkpoints``).
        """
        base = self.checkpoint_base_dir or (
            Path.home() / ".synthesis" / "agent-checkpoints"
        )
        return _discover_checkpoints_fs(base, since=since, until=until)


# ---------------------------------------------------------------------------
# Internal: provenance-restamping memory wrapper
# ---------------------------------------------------------------------------


class _RestampedMemory(Memory):
    """A Memory facade that rewrites provenance source strings on upsert.

    The underlying :func:`consolidate_session` writes provenance
    ``consolidation:session={id}``. The scheduled consolidator needs the
    richer ``consolidation:session={id}:model={id}:run_at={iso}`` form,
    and it needs to detect "this exact provenance already exists" to
    suppress duplicate writes when re-running with the same model.

    The wrapper:
      * intercepts ``upsert_entity`` and ``upsert_relation``
      * rewrites the provenance ``source`` to the canonical string
      * counts entities/relations actually added vs already present
      * forwards every other Memory method untouched.
    """

    backend_name = "restamped"

    def __init__(self, inner: Memory, *, provenance_source: str) -> None:
        self._inner = inner
        self.provenance_source = provenance_source
        self.entities_added = 0
        self.relations_added = 0
        self.entities_existing = 0
        self.relations_existing = 0

    # ----- upserts that need provenance restamping ----------------------

    def upsert_entity(self, entity):
        from .models import AttributeValue  # local import to avoid cycle

        # Restamp each attribute's provenance to the canonical source.
        attrs: Dict[str, AttributeValue] = {}
        for k, av in entity.attributes.items():
            new_prov = av.provenance.model_copy(
                update={"source": self.provenance_source}
            )
            attrs[k] = av.model_copy(update={"provenance": new_prov})
        restamped = entity.model_copy(update={"attributes": attrs})

        before = self._inner.get_entity(
            workspace=entity.workspace, type=entity.type, name=entity.name
        )
        result = self._inner.upsert_entity(restamped)
        if before is None:
            self.entities_added += 1
        else:
            self.entities_existing += 1
        return result

    def upsert_relation(self, relation, *, supersedes=None):
        prov = relation.provenance
        if prov is None:
            new_prov = Provenance(source=self.provenance_source, confidence=0.9)
        else:
            new_prov = prov.model_copy(update={"source": self.provenance_source})
        restamped = relation.model_copy(update={"provenance": new_prov})

        # Best-effort dedupe: if a relation with this exact provenance
        # source already exists between the same endpoints with the
        # same type, skip the write. This is the second half of the
        # idempotency contract.
        if self._relation_already_exists(restamped):
            self.relations_existing += 1
            return relation

        result = self._inner.upsert_relation(restamped, supersedes=supersedes)
        self.relations_added += 1
        return result

    def _relation_already_exists(self, relation) -> bool:
        try:
            existing = self._inner.query_graph(
                relation.workspace,
                seed_entity_ids=[relation.from_entity],
                depth=1,
                limit=200,
            )
        except Exception:  # pragma: no cover - defensive
            return False
        for r in existing:
            if str(r.from_entity) != str(relation.from_entity):
                continue
            if str(r.to_entity) != str(relation.to_entity):
                continue
            if r.type != relation.type:
                continue
            prov = getattr(r, "provenance", None)
            if prov is None:
                continue
            if prov.source == self.provenance_source:
                return True
        return False

    # ----- everything else: straight passthrough -----------------------

    def get_entity(self, *args, **kwargs):
        return self._inner.get_entity(*args, **kwargs)

    def list_entities(self, *args, **kwargs):
        return self._inner.list_entities(*args, **kwargs)

    def get_relation(self, *args, **kwargs):
        return self._inner.get_relation(*args, **kwargs)

    def query_graph(self, *args, **kwargs):
        return self._inner.query_graph(*args, **kwargs)

    def get_session(self, *args, **kwargs):
        return self._inner.get_session(*args, **kwargs)

    def set_session(self, *args, **kwargs):
        return self._inner.set_session(*args, **kwargs)

    def get_user(self, *args, **kwargs):
        return self._inner.get_user(*args, **kwargs)

    def set_user(self, *args, **kwargs):
        return self._inner.set_user(*args, **kwargs)

    def search_vector(self, *args, **kwargs):
        return self._inner.search_vector(*args, **kwargs)

    def search_three_tier(self, *args, **kwargs):
        return self._inner.search_three_tier(*args, **kwargs)


class _DryRunMemory(Memory):
    """Memory facade that counts attempted writes without forwarding them.

    Used by the dry-run code path so callers can preview what would
    land in the entity graph without actually mutating it. Reads pass
    through to the underlying backend so the LLM extractor still sees
    real session payloads.
    """

    backend_name = "dryrun"

    def __init__(self, inner: Memory) -> None:
        self._inner = inner
        self.entities_attempted = 0
        self.relations_attempted = 0

    def upsert_entity(self, entity):
        self.entities_attempted += 1
        # Return a synthetic id so the caller can chain into relation
        # writes; the upsert never lands.
        from uuid import uuid4 as _uuid4

        return entity.model_copy(update={"id": _uuid4()})

    def upsert_relation(self, relation, *, supersedes=None):
        self.relations_attempted += 1
        from uuid import uuid4 as _uuid4

        return relation.model_copy(update={"id": _uuid4()})

    def get_entity(self, *args, **kwargs):
        return self._inner.get_entity(*args, **kwargs)

    def list_entities(self, *args, **kwargs):
        return self._inner.list_entities(*args, **kwargs)

    def get_relation(self, *args, **kwargs):
        return self._inner.get_relation(*args, **kwargs)

    def query_graph(self, *args, **kwargs):
        return self._inner.query_graph(*args, **kwargs)

    def get_session(self, *args, **kwargs):
        return self._inner.get_session(*args, **kwargs)

    def set_session(self, *args, **kwargs):
        return self._inner.set_session(*args, **kwargs)

    def get_user(self, *args, **kwargs):
        return self._inner.get_user(*args, **kwargs)

    def set_user(self, *args, **kwargs):
        return self._inner.set_user(*args, **kwargs)

    def search_vector(self, *args, **kwargs):
        return self._inner.search_vector(*args, **kwargs)

    def search_three_tier(self, *args, **kwargs):
        return self._inner.search_three_tier(*args, **kwargs)


# ---------------------------------------------------------------------------
# History reader
# ---------------------------------------------------------------------------


def read_consolidation_history(
    limit: int = 100,
    log_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Tail recent ``memory_consolidation`` audit entries as JSON-safe dicts.

    Backs both the API endpoint and the CLI ``consolidation-history``
    command. Filters the audit log (which carries every cross-workspace
    op type) down to memory_consolidation entries.
    """
    from ..policy.audit import read_recent as _read_recent

    # Over-read: the audit log mixes op types so we may need to scan
    # more than ``limit`` raw entries to find ``limit`` consolidation
    # entries. We read up to 10x as a forensic ceiling.
    raw = _read_recent(limit=max(limit * 10, limit), log_path=log_path)
    entries = [e for e in raw if e.op_type == "memory_consolidation"]
    return [e.to_dict() for e in entries[-limit:]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_iso(value: str) -> datetime:
    """Parse an ISO-8601 string; assumes UTC if no tz is supplied."""
    # ``datetime.fromisoformat`` handles offsets natively from 3.11+.
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


__all__ = [
    "BatchReport",
    "ConsolidationReport",
    "MemoryConsolidator",
    "read_consolidation_history",
]
