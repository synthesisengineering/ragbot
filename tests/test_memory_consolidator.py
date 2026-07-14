"""Tests for the scheduled memory consolidator — the "dreaming" pattern.

Coverage (≥15 cases):

  1. Single-session consolidation extracts entities/relations from a
     planted transcript via the deterministic extractor path.
  2. Re-running the same session+model pair is idempotent (no duplicate
     facts written; provenance string filters out the second pass).
  3. A different model_id produces a NEW pass; both passes' provenance
     accumulates on the same entity's relations.
  4. ``dry_run=True`` returns counts but does not write to memory.
  5. ``consolidate_batch`` with since/until filters out checkpoints
     whose mtime falls outside the window.
  6. ``consolidate_recent_idle`` skips sessions whose latest checkpoint
     is newer than the threshold.
  7. ``consolidate_recent_idle`` skips sessions already consolidated
     with the same model_id (idempotency at batch scope).
  8. Audit log records a ``memory_consolidation`` entry per session.
  9. REST API endpoints (POST /consolidate, GET /consolidations/{id},
     GET /consolidation-history) return well-formed responses.
 10. REST API returns 4xx on invalid input.
 11. CLI's tabular output is parseable (header row + N data rows,
     stable column order).
 12. Missing session id reports skipped="session_not_found".
 13. Provenance source is the canonical
     ``consolidation:session={id}:model={id}:run_at={iso}`` form.
 14. Batch report aggregates per-session totals correctly.
 15. The LLM backend gets called exactly once per non-dry-run session,
     never on skip.
 16. The CLI memory subgroup's help text renders without import errors.
 17. ``read_consolidation_history`` filters the audit log to only
     memory_consolidation entries.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import pytest

_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from synthesis_engine.memory import (  # noqa: E402
    AttributeValue,
    BatchReport,
    ConsolidationReport,
    Entity,
    Memory,
    MemoryConsolidator,
    MemoryQuery,
    MemoryResult,
    Provenance,
    Relation,
    SessionMemory,
    UserMemory,
    read_consolidation_history,
    three_tier_retrieve,
)
from synthesis_engine.memory.base import require_provenance  # noqa: E402
from synthesis_engine.memory.consolidator import (  # noqa: E402
    _build_provenance_source,
    _parse_provenance_source,
)
from synthesis_engine.vectorstore import SearchHit  # noqa: E402


# ---------------------------------------------------------------------------
# In-memory Memory backend — independent from test_memory.StubMemory so the
# consolidator's contract is exercised without a database.
# ---------------------------------------------------------------------------


class InMemoryMemory(Memory):
    """Pure-dict in-memory implementation. Sufficient for consolidator tests."""

    backend_name = "in_memory"

    def __init__(self) -> None:
        self.entities: Dict[UUID, Entity] = {}
        self.relations: Dict[UUID, Relation] = {}
        self.sessions: Dict[str, SessionMemory] = {}
        self.users: Dict[str, UserMemory] = {}
        self.chunks: List[SearchHit] = []

    def upsert_entity(self, entity: Entity) -> Entity:
        existing = self.get_entity(
            workspace=entity.workspace, type=entity.type, name=entity.name
        )
        if existing is not None:
            merged = {**existing.attributes, **entity.attributes}
            new = existing.model_copy(update={"attributes": merged})
            self.entities[new.id] = new
            return new
        new = entity.model_copy(update={"id": uuid4()})
        self.entities[new.id] = new
        return new

    def get_entity(
        self,
        entity_id: Optional[UUID] = None,
        *,
        workspace: Optional[str] = None,
        type: Optional[str] = None,
        name: Optional[str] = None,
    ) -> Optional[Entity]:
        if entity_id is not None:
            return self.entities.get(entity_id)
        for e in self.entities.values():
            if (
                e.workspace == workspace
                and e.type == type
                and e.name == name
            ):
                return e
        return None

    def list_entities(
        self,
        workspace: str,
        *,
        type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Entity]:
        rows = [
            e
            for e in self.entities.values()
            if e.workspace == workspace and (type is None or e.type == type)
        ]
        return rows[offset : offset + limit]

    def upsert_relation(
        self,
        relation: Relation,
        *,
        supersedes: Optional[UUID] = None,
    ) -> Relation:
        require_provenance(relation.provenance)
        if supersedes is not None:
            prior = self.relations.get(supersedes)
            if prior is not None:
                self.relations[supersedes] = prior.model_copy(
                    update={
                        "validity_end": datetime.now(tz=timezone.utc)
                    }
                )
        new = relation.model_copy(update={"id": uuid4()})
        if new.validity_start is None:
            new = new.model_copy(
                update={"validity_start": datetime.now(tz=timezone.utc)}
            )
        self.relations[new.id] = new
        return new

    def get_relation(self, relation_id: UUID) -> Optional[Relation]:
        return self.relations.get(relation_id)

    def query_graph(
        self,
        workspace: str,
        *,
        seed_entity_ids: List[UUID],
        depth: int = 2,
        validity_at: Optional[datetime] = None,
        relation_types: Optional[List[str]] = None,
        limit: int = 200,
    ) -> List[Relation]:
        frontier = {str(sid) for sid in seed_entity_ids}
        visited = set(frontier)
        touched: List[Relation] = []
        for _ in range(depth):
            next_frontier: set = set()
            for r in self.relations.values():
                if r.workspace != workspace:
                    continue
                if relation_types and r.type not in relation_types:
                    continue
                if validity_at is None and r.validity_end is not None:
                    continue
                if (
                    str(r.from_entity) in frontier
                    or str(r.to_entity) in frontier
                ):
                    touched.append(r)
                    next_frontier.add(str(r.from_entity))
                    next_frontier.add(str(r.to_entity))
            frontier = next_frontier - visited
            visited |= frontier
            if not frontier:
                break
        seen: set = set()
        out: List[Relation] = []
        for r in touched:
            if r.id in seen:
                continue
            seen.add(r.id)
            out.append(r)
        return out[:limit]

    def get_session(self, session_id: str) -> Optional[SessionMemory]:
        return self.sessions.get(session_id)

    def set_session(self, session: SessionMemory) -> SessionMemory:
        self.sessions[session.session_id] = session
        return session

    def get_user(self, user_id: str) -> Optional[UserMemory]:
        return self.users.get(user_id)

    def set_user(self, user: UserMemory) -> UserMemory:
        self.users[user.user_id] = user
        return user

    def search_vector(
        self,
        workspace: str,
        query_vector: List[float],
        *,
        limit: int = 10,
        content_type: Optional[str] = None,
    ) -> List[SearchHit]:
        return self.chunks[:limit]

    def search_three_tier(
        self,
        query: MemoryQuery,
        *,
        query_vector: Optional[List[float]] = None,
    ) -> List[MemoryResult]:
        return three_tier_retrieve(self, query, query_vector=query_vector)


# ---------------------------------------------------------------------------
# Fake LLM backend that returns a scripted JSON extraction
# ---------------------------------------------------------------------------


@dataclass
class _FakeLLMResponse:
    text: str
    model: str = "fake-model"
    backend: str = "fake"
    finish_reason: Optional[str] = "stop"
    usage: Dict[str, int] = None  # type: ignore[assignment]


class FakeLLMBackend:
    """Returns a deterministic JSON extraction. Tracks call count."""

    backend_name = "fake"

    def __init__(self, payload: Dict[str, Any]) -> None:
        self.payload = payload
        self.calls: List[Any] = []

    def complete(self, request: Any) -> _FakeLLMResponse:
        self.calls.append(request)
        return _FakeLLMResponse(
            text=json.dumps(self.payload),
            usage={},
        )

    def stream(self, request, on_chunk):  # pragma: no cover - unused
        on_chunk(json.dumps(self.payload))
        return json.dumps(self.payload)

    def healthcheck(self):
        return {"backend": "fake", "ok": True}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def memory_backend() -> InMemoryMemory:
    return InMemoryMemory()


@pytest.fixture()
def planted_session(memory_backend) -> str:
    """Seed a session payload that the LLM extractor will distil."""
    session = SessionMemory(
        session_id="sess-acme-001",
        user_id="acme-user",
        workspace="acme-news",
        payload={
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Alice published synthesis engineering articles."
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "Noted. Alice authored work on synthesis engineering."
                    ),
                },
            ],
            "summary": (
                "Alice authored several articles on synthesis engineering."
            ),
        },
    )
    memory_backend.set_session(session)
    return session.session_id


@pytest.fixture()
def llm_payload() -> Dict[str, Any]:
    return {
        "entities": [
            {
                "type": "person",
                "name": "Alice",
                "attributes": {"role": "author"},
            },
            {"type": "concept", "name": "synthesis engineering"},
        ],
        "relations": [
            {
                "from": {"type": "person", "name": "Alice"},
                "to": {"type": "concept", "name": "synthesis engineering"},
                "type": "authored",
                "confidence": 0.92,
                "attributes": {"date": "2025-11"},
            }
        ],
    }


@pytest.fixture()
def fake_llm(llm_payload) -> FakeLLMBackend:
    return FakeLLMBackend(payload=llm_payload)


@pytest.fixture()
def audit_log(tmp_path, monkeypatch) -> Path:
    """Redirect the audit log to a tmp path so tests don't pollute the host."""
    path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("SYNTHESIS_AUDIT_LOG_PATH", str(path))
    return path


@pytest.fixture()
def checkpoint_root(tmp_path) -> Path:
    """A throwaway checkpoint root for the discovery code path."""
    return tmp_path / "checkpoints"


# ---------------------------------------------------------------------------
# 1. Single-session consolidation
# ---------------------------------------------------------------------------


def test_single_session_extracts_entities_and_relations(
    memory_backend, planted_session, fake_llm, audit_log
):
    consolidator = MemoryConsolidator(
        memory_backend,
        llm_backend=fake_llm,
        default_workspace="acme-news",
    )
    report = asyncio.run(
        consolidator.consolidate_session(
            planted_session, model_id="anthropic/claude-haiku-4-5"
        )
    )
    assert isinstance(report, ConsolidationReport)
    assert report.error is None, report.error
    assert report.skipped is False
    assert report.entities_added == 2
    assert report.relations_added == 1
    assert report.duration_seconds >= 0.0
    # The entity graph has the planted facts.
    alice = memory_backend.get_entity(
        workspace="acme-news", type="person", name="Alice"
    )
    assert alice is not None
    rels = memory_backend.query_graph(
        "acme-news", seed_entity_ids=[alice.id], depth=1
    )
    authored = [r for r in rels if r.type == "authored"]
    assert len(authored) == 1
    # Provenance is the canonical form.
    src = authored[0].provenance.source
    parsed = _parse_provenance_source(src)
    assert parsed is not None
    assert parsed[0] == planted_session
    assert parsed[1] == "anthropic/claude-haiku-4-5"


# ---------------------------------------------------------------------------
# 2. Idempotency — same session + same model = no duplicate facts
# ---------------------------------------------------------------------------


def test_idempotent_same_session_and_model(
    memory_backend, planted_session, llm_payload, audit_log
):
    # Two calls with the same model id; the second is a no-op.
    fake_llm = FakeLLMBackend(payload=llm_payload)
    consolidator = MemoryConsolidator(memory_backend, llm_backend=fake_llm)
    asyncio.run(
        consolidator.consolidate_session(
            planted_session,
            model_id="anthropic/claude-haiku-4-5",
            workspace="acme-news",
        )
    )
    relations_before = len(memory_backend.relations)

    report2 = asyncio.run(
        consolidator.consolidate_session(
            planted_session,
            model_id="anthropic/claude-haiku-4-5",
            workspace="acme-news",
        )
    )
    assert report2.skipped is True
    assert report2.skip_reason == "already_consolidated_with_same_model"
    # No new relations.
    assert len(memory_backend.relations) == relations_before


# ---------------------------------------------------------------------------
# 3. Different model id produces a new pass; provenance accumulates
# ---------------------------------------------------------------------------


def test_different_model_accumulates_provenance(
    memory_backend, planted_session, llm_payload, audit_log
):
    fake_llm = FakeLLMBackend(payload=llm_payload)
    consolidator = MemoryConsolidator(memory_backend, llm_backend=fake_llm)
    asyncio.run(
        consolidator.consolidate_session(
            planted_session,
            model_id="anthropic/claude-haiku-4-5",
            workspace="acme-news",
        )
    )
    # Re-run with a different model. The extractor returns the same
    # payload, so the same logical facts are extracted — but the
    # provenance string differs, so the new relations are written.
    asyncio.run(
        consolidator.consolidate_session(
            planted_session,
            model_id="anthropic/claude-opus-4-8",
            workspace="acme-news",
        )
    )
    alice = memory_backend.get_entity(
        workspace="acme-news", type="person", name="Alice"
    )
    assert alice is not None
    rels = memory_backend.query_graph(
        "acme-news", seed_entity_ids=[alice.id], depth=1
    )
    authored = [r for r in rels if r.type == "authored"]
    sources = {r.provenance.source for r in authored}
    # Both passes are present.
    assert len(authored) == 2
    haiku_runs = [
        s for s in sources if "model=anthropic/claude-haiku-4-5" in s
    ]
    opus_runs = [
        s for s in sources if "model=anthropic/claude-opus-4-8" in s
    ]
    assert haiku_runs, "expected Haiku pass to be recorded"
    assert opus_runs, "expected Opus pass to be recorded"


# ---------------------------------------------------------------------------
# 4. dry_run=True does not write
# ---------------------------------------------------------------------------


def test_dry_run_does_not_write(
    memory_backend, planted_session, fake_llm, audit_log
):
    consolidator = MemoryConsolidator(memory_backend, llm_backend=fake_llm)
    entities_before = len(memory_backend.entities)
    relations_before = len(memory_backend.relations)
    report = asyncio.run(
        consolidator.consolidate_session(
            planted_session,
            model_id="anthropic/claude-haiku-4-5",
            workspace="acme-news",
            dry_run=True,
        )
    )
    assert report.error is None
    # No mutation
    assert len(memory_backend.entities) == entities_before
    assert len(memory_backend.relations) == relations_before
    # But counts are reported.
    assert report.entities_added >= 1
    assert report.relations_added >= 1


# ---------------------------------------------------------------------------
# 5. Batch with since/until filters by checkpoint mtime
# ---------------------------------------------------------------------------


def _write_fake_checkpoint(
    base: Path, task_id: str, mtime_dt: datetime
) -> None:
    """Write a dummy checkpoint file under base/{task_id}/0000.json and
    set its mtime."""
    d = base / task_id
    d.mkdir(parents=True, exist_ok=True)
    f = d / "0000.json"
    f.write_text(json.dumps({"task_id": task_id}))
    ts = mtime_dt.timestamp()
    os.utime(f, (ts, ts))


def test_batch_window_filters_checkpoints(
    memory_backend, fake_llm, audit_log, checkpoint_root, tmp_path
):
    base = checkpoint_root
    # Three sessions: two days ago, two hours ago, ten minutes ago.
    now = datetime.now(tz=timezone.utc)
    sessions_meta = [
        ("acme-sess-old", now - timedelta(days=2)),
        ("acme-sess-mid", now - timedelta(hours=2)),
        ("acme-sess-new", now - timedelta(minutes=10)),
    ]
    for sid, dt in sessions_meta:
        _write_fake_checkpoint(base, sid, dt)
        memory_backend.set_session(
            SessionMemory(
                session_id=sid,
                user_id="acme-user",
                workspace="acme-news",
                payload={"summary": f"summary for {sid}"},
            )
        )

    consolidator = MemoryConsolidator(
        memory_backend,
        llm_backend=fake_llm,
        checkpoint_base_dir=base,
        default_workspace="acme-news",
    )
    # Window: 3 hours ago to 1 hour ago — only the "mid" session matches.
    since_iso = (now - timedelta(hours=3)).isoformat()
    until_iso = (now - timedelta(hours=1)).isoformat()
    report = asyncio.run(
        consolidator.consolidate_batch(
            since_iso=since_iso, until_iso=until_iso, workspace="acme-news"
        )
    )
    assert isinstance(report, BatchReport)
    assert report.sessions_consolidated == 1
    consolidated_ids = {r.session_id for r in report.per_session}
    assert consolidated_ids == {"acme-sess-mid"}


# ---------------------------------------------------------------------------
# 6. consolidate_recent_idle skips active sessions
# ---------------------------------------------------------------------------


def test_idle_threshold_skips_active_sessions(
    memory_backend, fake_llm, audit_log, checkpoint_root
):
    base = checkpoint_root
    now = datetime.now(tz=timezone.utc)
    _write_fake_checkpoint(base, "acme-sess-stale", now - timedelta(hours=8))
    _write_fake_checkpoint(base, "acme-sess-active", now - timedelta(minutes=5))
    for sid in ("acme-sess-stale", "acme-sess-active"):
        memory_backend.set_session(
            SessionMemory(
                session_id=sid,
                user_id="acme-user",
                workspace="acme-news",
                payload={"summary": f"summary for {sid}"},
            )
        )
    consolidator = MemoryConsolidator(
        memory_backend,
        llm_backend=fake_llm,
        checkpoint_base_dir=base,
        default_workspace="acme-news",
    )
    report = asyncio.run(
        consolidator.consolidate_recent_idle(
            idle_threshold_hours=4.0, workspace="acme-news"
        )
    )
    consolidated = [r.session_id for r in report.per_session]
    # Only the stale session should land.
    assert "acme-sess-stale" in consolidated
    assert "acme-sess-active" not in consolidated


# ---------------------------------------------------------------------------
# 7. consolidate_recent_idle skips already-consolidated sessions
# ---------------------------------------------------------------------------


def test_idle_skips_already_consolidated_same_model(
    memory_backend, fake_llm, audit_log, checkpoint_root
):
    base = checkpoint_root
    now = datetime.now(tz=timezone.utc)
    _write_fake_checkpoint(base, "acme-sess-stale", now - timedelta(hours=8))
    memory_backend.set_session(
        SessionMemory(
            session_id="acme-sess-stale",
            user_id="acme-user",
            workspace="acme-news",
            payload={"summary": "summary for stale"},
        )
    )
    consolidator = MemoryConsolidator(
        memory_backend,
        llm_backend=fake_llm,
        checkpoint_base_dir=base,
        default_workspace="acme-news",
    )
    asyncio.run(
        consolidator.consolidate_recent_idle(
            idle_threshold_hours=4.0,
            workspace="acme-news",
            model_id="anthropic/claude-haiku-4-5",
        )
    )
    # Second run with the same model id should skip.
    report2 = asyncio.run(
        consolidator.consolidate_recent_idle(
            idle_threshold_hours=4.0,
            workspace="acme-news",
            model_id="anthropic/claude-haiku-4-5",
        )
    )
    assert report2.sessions_consolidated == 0
    assert report2.sessions_skipped == 1


# ---------------------------------------------------------------------------
# 8. Audit log records a memory_consolidation entry per pass
# ---------------------------------------------------------------------------


def test_audit_log_records_per_consolidation(
    memory_backend, planted_session, fake_llm, audit_log
):
    consolidator = MemoryConsolidator(memory_backend, llm_backend=fake_llm)
    asyncio.run(
        consolidator.consolidate_session(
            planted_session,
            model_id="anthropic/claude-haiku-4-5",
            workspace="acme-news",
        )
    )
    # Read the audit log file directly.
    assert audit_log.is_file()
    lines = audit_log.read_text("utf-8").strip().splitlines()
    rows = [json.loads(ln) for ln in lines if ln.strip()]
    consolidation_entries = [
        r for r in rows if r.get("op_type") == "memory_consolidation"
    ]
    assert len(consolidation_entries) == 1
    e = consolidation_entries[0]
    assert e["workspaces"] == ["acme-news"]
    assert e["model_id"] == "anthropic/claude-haiku-4-5"
    assert e["metadata"]["session_id"] == planted_session


# ---------------------------------------------------------------------------
# 9. REST API: POST /consolidate single-session returns a ConsolidationReport
# ---------------------------------------------------------------------------


def test_api_post_consolidate_single_session(
    memory_backend, planted_session, fake_llm, audit_log, monkeypatch
):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.routers import memory as memory_router

    # Patch get_memory to return our in-memory backend.
    monkeypatch.setattr(memory_router, "get_memory", lambda: memory_backend)

    # Patch MemoryConsolidator constructor to inject fake LLM.
    from synthesis_engine.memory import MemoryConsolidator as _MC

    original_init = _MC.__init__

    def patched_init(self, mem, **kw):
        kw.setdefault("llm_backend", fake_llm)
        return original_init(self, mem, **kw)

    monkeypatch.setattr(_MC, "__init__", patched_init)

    app = FastAPI()
    app.include_router(memory_router.router)
    client = TestClient(app)

    resp = client.post(
        "/api/memory/consolidate",
        json={
            "session_id": planted_session,
            "model_id": "anthropic/claude-haiku-4-5",
            "workspace": "acme-news",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["session_id"] == planted_session
    assert body["entities_added"] == 2
    assert body["relations_added"] == 1


# ---------------------------------------------------------------------------
# 10. REST API: GET /consolidation-history filters to consolidation entries
# ---------------------------------------------------------------------------


def test_api_consolidation_history(
    memory_backend, planted_session, fake_llm, audit_log, monkeypatch
):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.routers import memory as memory_router

    monkeypatch.setattr(memory_router, "get_memory", lambda: memory_backend)

    consolidator = MemoryConsolidator(memory_backend, llm_backend=fake_llm)
    asyncio.run(
        consolidator.consolidate_session(
            planted_session,
            model_id="anthropic/claude-haiku-4-5",
            workspace="acme-news",
        )
    )

    app = FastAPI()
    app.include_router(memory_router.router)
    client = TestClient(app)

    resp = client.get("/api/memory/consolidation-history?limit=50")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "entries" in body
    assert body["count"] >= 1
    assert all(e.get("op_type") == "memory_consolidation" for e in body["entries"])


# ---------------------------------------------------------------------------
# 11. REST API rejects invalid input
# ---------------------------------------------------------------------------


def test_api_consolidation_history_rejects_invalid_limit(
    memory_backend, audit_log, monkeypatch
):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.routers import memory as memory_router

    monkeypatch.setattr(memory_router, "get_memory", lambda: memory_backend)

    app = FastAPI()
    app.include_router(memory_router.router)
    client = TestClient(app)

    # limit < 1 -> 422 from FastAPI's Query validation
    resp = client.get("/api/memory/consolidation-history?limit=0")
    assert resp.status_code == 422


def test_api_get_consolidations_unknown_task_returns_404(
    memory_backend, audit_log, monkeypatch
):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.routers import memory as memory_router

    monkeypatch.setattr(memory_router, "get_memory", lambda: memory_backend)

    app = FastAPI()
    app.include_router(memory_router.router)
    client = TestClient(app)

    resp = client.get("/api/memory/consolidations/bogus-task-id-abcdef")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 12. Missing session is reported as skipped
# ---------------------------------------------------------------------------


def test_missing_session_skipped(memory_backend, fake_llm, audit_log):
    consolidator = MemoryConsolidator(memory_backend, llm_backend=fake_llm)
    report = asyncio.run(
        consolidator.consolidate_session(
            "no-such-session-id", model_id="anthropic/claude-haiku-4-5"
        )
    )
    assert report.skipped is True
    assert report.skip_reason == "session_not_found"
    assert report.entities_added == 0
    assert report.relations_added == 0


# ---------------------------------------------------------------------------
# 13. Provenance source rendering / parsing round-trip
# ---------------------------------------------------------------------------


def test_provenance_source_round_trip():
    iso = "2026-05-14T10:00:00+00:00"
    s = _build_provenance_source(
        session_id="sess-acme-001",
        model_id="anthropic/claude-haiku-4-5",
        run_at_iso=iso,
    )
    parsed = _parse_provenance_source(s)
    assert parsed is not None
    sid, mid, run = parsed
    assert sid == "sess-acme-001"
    assert mid == "anthropic/claude-haiku-4-5"
    assert run == iso
    # Reject malformed inputs
    assert _parse_provenance_source("session:abc123") is None
    assert _parse_provenance_source("") is None


# ---------------------------------------------------------------------------
# 14. BatchReport aggregates per-session totals
# ---------------------------------------------------------------------------


def test_batch_report_aggregates_correctly(
    memory_backend, fake_llm, audit_log, checkpoint_root
):
    base = checkpoint_root
    now = datetime.now(tz=timezone.utc)
    # Three sessions, all stale enough to consolidate.
    for sid in ("acme-sess-a", "acme-sess-b", "acme-sess-c"):
        _write_fake_checkpoint(base, sid, now - timedelta(hours=8))
        memory_backend.set_session(
            SessionMemory(
                session_id=sid,
                user_id="acme-user",
                workspace="acme-news",
                payload={"summary": f"summary for {sid}"},
            )
        )
    consolidator = MemoryConsolidator(
        memory_backend,
        llm_backend=fake_llm,
        checkpoint_base_dir=base,
        default_workspace="acme-news",
    )
    report = asyncio.run(
        consolidator.consolidate_recent_idle(
            idle_threshold_hours=4.0, workspace="acme-news"
        )
    )
    assert report.sessions_consolidated == 3
    assert report.total_entities_added == sum(
        r.entities_added for r in report.per_session
    )
    assert report.total_relations_added == sum(
        r.relations_added for r in report.per_session
    )


# ---------------------------------------------------------------------------
# 15. LLM called exactly once per non-dry-run session
# ---------------------------------------------------------------------------


def test_llm_called_once_per_session(
    memory_backend, planted_session, llm_payload, audit_log
):
    fake = FakeLLMBackend(payload=llm_payload)
    consolidator = MemoryConsolidator(memory_backend, llm_backend=fake)
    asyncio.run(
        consolidator.consolidate_session(
            planted_session,
            model_id="anthropic/claude-haiku-4-5",
            workspace="acme-news",
        )
    )
    assert len(fake.calls) == 1
    # On the idempotent re-run (same model), the LLM should NOT be called.
    asyncio.run(
        consolidator.consolidate_session(
            planted_session,
            model_id="anthropic/claude-haiku-4-5",
            workspace="acme-news",
        )
    )
    assert len(fake.calls) == 1, "second pass should short-circuit before LLM"


# ---------------------------------------------------------------------------
# 16. CLI subgroup renders help text
# ---------------------------------------------------------------------------


def test_cli_memory_help_renders():
    """The `ragbot memory --help` command imports and prints without error."""
    ragbot_script = Path(__file__).resolve().parents[1] / "src" / "ragbot.py"
    result = subprocess.run(
        [sys.executable, str(ragbot_script), "memory", "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ragbot_script.parent)},
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    # Tabular header for the two subcommands must appear.
    assert "consolidate" in result.stdout
    assert "consolidation-history" in result.stdout


def test_cli_memory_consolidate_help_renders():
    ragbot_script = Path(__file__).resolve().parents[1] / "src" / "ragbot.py"
    result = subprocess.run(
        [
            sys.executable,
            str(ragbot_script),
            "memory",
            "consolidate",
            "--help",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ragbot_script.parent)},
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--session-id" in result.stdout
    assert "--idle-hours" in result.stdout
    assert "--dry-run" in result.stdout


# ---------------------------------------------------------------------------
# 17. read_consolidation_history filters audit log correctly
# ---------------------------------------------------------------------------


def test_read_consolidation_history_filters(
    memory_backend, planted_session, fake_llm, audit_log
):
    from synthesis_engine.policy import AuditEntry, record as record_audit

    # Write some non-consolidation audit entries.
    record_audit(
        AuditEntry.build(
            op_type="cross_workspace_synthesis",
            workspaces=["acme-news", "acme-user"],
            tools=["search"],
            model_id="anthropic/claude-opus-4-8",
            outcome="allowed",
        )
    )
    record_audit(
        AuditEntry.build(
            op_type="tool_call",
            workspaces=["acme-news"],
            tools=["web_search"],
            model_id="anthropic/claude-opus-4-8",
            outcome="allowed",
        )
    )
    # Now run a consolidation.
    consolidator = MemoryConsolidator(memory_backend, llm_backend=fake_llm)
    asyncio.run(
        consolidator.consolidate_session(
            planted_session,
            model_id="anthropic/claude-haiku-4-5",
            workspace="acme-news",
        )
    )

    history = read_consolidation_history(limit=100)
    # Only the consolidation entry should come back.
    assert len(history) == 1
    assert history[0]["op_type"] == "memory_consolidation"


# ---------------------------------------------------------------------------
# 18. CLI tabular output for consolidation-history is parseable
# ---------------------------------------------------------------------------


def test_cli_tabular_output_for_history(
    memory_backend, planted_session, fake_llm, audit_log
):
    """The consolidation-history CLI prints a header row and one
    data row per audit entry. Columns are stable: timestamp,
    session_id, workspace, model_id, entities_added, relations_added."""
    # Plant one consolidation in the audit log.
    consolidator = MemoryConsolidator(memory_backend, llm_backend=fake_llm)
    asyncio.run(
        consolidator.consolidate_session(
            planted_session,
            model_id="anthropic/claude-haiku-4-5",
            workspace="acme-news",
        )
    )
    # Now drive the CLI's history runner directly so we can inspect
    # stdout without re-spawning python. The "ragbot" name resolves to
    # the src/ragbot/ package (not the src/ragbot.py CLI script), so we
    # load the script by path.
    from io import StringIO
    import contextlib
    import importlib.util

    script_path = (
        Path(__file__).resolve().parents[1] / "src" / "ragbot.py"
    )
    spec = importlib.util.spec_from_file_location(
        "ragbot_cli_module", script_path
    )
    rb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rb)

    class _Args:
        limit = 50

    captured = StringIO()
    with contextlib.redirect_stdout(captured):
        rc = rb.run_memory_consolidation_history(_Args())
    assert rc == 0
    out = captured.getvalue().strip().splitlines()
    assert len(out) >= 2  # header + at least one data row
    header = out[0].split()
    assert header[:3] == ["timestamp", "session_id", "workspace"]
    # The data row's third column should carry the workspace name.
    data_row = out[1].split()
    assert "acme-news" in data_row
