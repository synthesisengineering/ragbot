"""Tests for the content-tier + RLS hardening added in migration 0003.

These are unit tests against a fake psycopg connection/cursor — no live
Postgres required. They verify:

  * The tier-resolution helper defaults to the safe (non-archive) tier set.
  * search() / keyword_search() / scroll_documents() bind the RLS session
    context (app.current_workspace) and filter on content_tier before
    running their query.
  * delete_collection() branches correctly between a full wipe and a
    tier-scoped clear, and never relies on FK cascade to clean up chunks.
  * list_collections() binds the admin escape hatch instead of a
    workspace-scoped context, since it has no single workspace to scope to.
  * rag.py threads content_tier through index_content() into the Point
    objects handed to upsert_points(), and rag.search_archive() only ever
    asks for the 'archive-only' tier.

A live-database test class (gated on RAGBOT_PGVECTOR_TEST_URL, same
convention as tests/test_vectorstore.py) proves the RLS policies actually
enforce isolation — not just that the code calls set_config with the
right arguments.
"""

from __future__ import annotations

import os
import sys
import uuid
from unittest.mock import MagicMock, patch

import pytest

_SRC = os.path.join(os.path.dirname(__file__), '..', 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from synthesis_engine.vectorstore import (  # noqa: E402
    CONTENT_TIERS,
    DEFAULT_VISIBLE_CONTENT_TIERS,
    Point,
)
from synthesis_engine.vectorstore.pgvector_backend import (  # noqa: E402
    PgvectorBackend,
    _resolve_tiers,
)


# ---------------------------------------------------------------------------
# Fake psycopg connection/cursor harness
# ---------------------------------------------------------------------------


class FakeCursor:
    def __init__(self, fetchall_result=None, fetchone_result=None):
        self.executed = []  # list of (sql, params)
        self._fetchall_result = fetchall_result or []
        self._fetchone_result = fetchone_result

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        return self

    def fetchall(self):
        return self._fetchall_result

    def fetchone(self):
        return self._fetchone_result

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConnection:
    def __init__(self, cursor: FakeCursor):
        self._cursor = cursor
        self.committed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _backend_with_fake_cursor(fetchall_result=None, fetchone_result=None):
    """Build a PgvectorBackend whose _connection() yields a fake cursor
    that records every execute() call, without touching a real DB."""
    backend = PgvectorBackend(dsn="postgresql://fake/fake")
    cursor = FakeCursor(fetchall_result=fetchall_result, fetchone_result=fetchone_result)
    conn = FakeConnection(cursor)
    backend._connection = MagicMock(return_value=conn)  # type: ignore[method-assign]
    return backend, cursor


# ---------------------------------------------------------------------------
# Tier resolution
# ---------------------------------------------------------------------------


class TestResolveTiers:
    def test_none_defaults_to_safe_tiers(self):
        assert _resolve_tiers(None) == list(DEFAULT_VISIBLE_CONTENT_TIERS)

    def test_default_excludes_archive_only(self):
        assert "archive-only" not in DEFAULT_VISIBLE_CONTENT_TIERS
        assert "archive-only" in CONTENT_TIERS

    def test_explicit_tiers_are_used_verbatim(self):
        assert _resolve_tiers(["archive-only"]) == ["archive-only"]
        assert _resolve_tiers(["frequent-access", "archive-only"]) == [
            "frequent-access", "archive-only",
        ]


class TestPointDefaults:
    def test_point_default_tier_is_topic_specific(self):
        p = Point(chunk_uid="x", vector=[0.0] * 4, text="hi")
        assert p.content_tier == "topic-specific"
        assert p.content_hash is None
        assert p.embedding_model_version is None


# ---------------------------------------------------------------------------
# search() / keyword_search() / scroll_documents() — RLS context + tier filter
# ---------------------------------------------------------------------------


class TestSearchTierFiltering:
    def test_search_sets_workspace_context_before_query(self):
        backend, cursor = _backend_with_fake_cursor(fetchall_result=[])
        backend.search("ws-a", [0.1, 0.2], limit=5)

        assert len(cursor.executed) == 2
        set_config_sql, set_config_params = cursor.executed[0]
        assert "set_config" in set_config_sql
        assert set_config_params == ("ws-a",)

    def test_search_defaults_to_safe_tiers(self):
        backend, cursor = _backend_with_fake_cursor(fetchall_result=[])
        backend.search("ws-a", [0.1, 0.2], limit=5)

        query_sql, query_params = cursor.executed[1]
        assert "content_tier = ANY(%s)" in query_sql
        # tiers list is passed positionally; find it by type (a list of str)
        tier_arg = [p for p in query_params if isinstance(p, list) and p and isinstance(p[0], str)]
        assert tier_arg == [list(DEFAULT_VISIBLE_CONTENT_TIERS)]

    def test_search_honors_explicit_archive_tier(self):
        backend, cursor = _backend_with_fake_cursor(fetchall_result=[])
        backend.search("ws-a", [0.1, 0.2], limit=5, content_tiers=["archive-only"])

        _, query_params = cursor.executed[1]
        tier_arg = [p for p in query_params if isinstance(p, list) and p and isinstance(p[0], str)]
        assert tier_arg == [["archive-only"]]

    def test_keyword_search_defaults_to_safe_tiers(self):
        backend, cursor = _backend_with_fake_cursor(fetchall_result=[])
        backend.keyword_search("ws-a", "hello")

        query_sql, query_params = cursor.executed[1]
        assert "content_tier = ANY(%s)" in query_sql
        tier_arg = [p for p in query_params if isinstance(p, list) and p and isinstance(p[0], str)]
        assert tier_arg == [list(DEFAULT_VISIBLE_CONTENT_TIERS)]

    def test_scroll_documents_defaults_to_safe_tiers(self):
        backend, cursor = _backend_with_fake_cursor(fetchall_result=[])
        backend.scroll_documents("ws-a")

        query_sql, query_params = cursor.executed[1]
        assert "content_tier = ANY(%s)" in query_sql
        tier_arg = [p for p in query_params if isinstance(p, list) and p and isinstance(p[0], str)]
        assert tier_arg == [list(DEFAULT_VISIBLE_CONTENT_TIERS)]


# ---------------------------------------------------------------------------
# delete_collection() — full wipe vs tier-scoped clear
# ---------------------------------------------------------------------------


class TestDeleteCollection:
    def test_full_wipe_deletes_chunks_documents_and_workspace_row(self):
        backend, cursor = _backend_with_fake_cursor()
        ok = backend.delete_collection("ws-a")
        assert ok is True

        statements = [sql for sql, _ in cursor.executed]
        assert any("set_config('app.current_workspace'" in s for s in statements)
        assert any(s.strip().startswith("DELETE FROM chunks WHERE workspace = %s")
                   and "content_tier" not in s for s in statements)
        assert any("DELETE FROM documents WHERE workspace = %s" in s for s in statements)
        assert any("DELETE FROM workspaces WHERE name = %s" in s for s in statements)

    def test_tier_scoped_clear_leaves_workspace_row_intact(self):
        backend, cursor = _backend_with_fake_cursor()
        ok = backend.delete_collection("ws-a", content_tier="archive-only")
        assert ok is True

        statements = [sql for sql, _ in cursor.executed]
        assert any("DELETE FROM chunks WHERE workspace = %s AND content_tier = %s" in s
                   for s in statements)
        assert any("DELETE FROM documents" in s and "NOT IN" in s for s in statements)
        # The workspace row itself must never be touched in tier-scoped mode.
        assert not any("DELETE FROM workspaces" in s for s in statements)

    def test_tier_scoped_clear_params_include_tier(self):
        backend, cursor = _backend_with_fake_cursor()
        backend.delete_collection("ws-a", content_tier="archive-only")

        chunk_delete = next(
            params for sql, params in cursor.executed
            if sql.strip().startswith("DELETE FROM chunks WHERE workspace = %s AND content_tier")
        )
        assert chunk_delete == ("ws-a", "archive-only")


class TestListCollectionsAdminBypass:
    def test_list_collections_sets_admin_flag_not_workspace(self):
        backend, cursor = _backend_with_fake_cursor(fetchall_result=[("ws-a",)])
        backend.list_collections()

        set_config_sql, set_config_params = cursor.executed[0]
        assert "app.rls_admin" in set_config_sql
        assert set_config_params is None  # 'on' is a literal in the SQL, not bound


# ---------------------------------------------------------------------------
# upsert_points — writes the new columns
# ---------------------------------------------------------------------------


class TestUpsertPointsWritesTierColumns:
    def test_chunk_insert_includes_tier_hash_and_version(self):
        backend, cursor = _backend_with_fake_cursor(fetchone_result=(1,))
        point = Point(
            chunk_uid="c1", vector=[0.0] * 4, text="hi",
            source_path="/tmp/a.md", content_tier="archive-only",
            content_hash="deadbeef", embedding_model_version="model@1.2.3",
        )
        written = backend.upsert_points("ws-a", [point])
        assert written == 1

        chunk_insert = next(
            (sql, params) for sql, params in cursor.executed
            if sql.strip().startswith("\n                            INSERT INTO chunks")
            or "INSERT INTO chunks" in sql
        )
        sql, params = chunk_insert
        assert "content_tier" in sql
        assert "content_hash" in sql
        assert "embedding_model_version" in sql
        assert "archive-only" in params
        assert "deadbeef" in params
        assert "model@1.2.3" in params


# ---------------------------------------------------------------------------
# rag.py wiring — index_content / search_archive
# ---------------------------------------------------------------------------


class TestRagIndexContentTierThreading:
    def test_index_content_tags_points_with_requested_tier(self):
        import rag

        fake_chunk = MagicMock()
        fake_chunk.metadata = {
            'filename': 'a.md', 'title': 'A', 'chunk_index': 0,
            'char_start': 0, 'char_end': 5, 'source_file': '/tmp/a.md',
        }
        fake_chunk.text = "hello"

        fake_vs = MagicMock()
        fake_vs.upsert_points.return_value = 1
        fake_model = MagicMock()
        fake_model.encode.return_value.tolist.return_value = [0.0] * 4

        with patch.object(rag, '_load_chunking', return_value=True), \
             patch.object(rag, 'VECTOR_STORE_AVAILABLE', True), \
             patch.object(rag, 'get_vector_store', return_value=fake_vs), \
             patch.object(rag, '_get_embedding_model', return_value=fake_model), \
             patch.object(rag, '_get_embedding_dimension', return_value=4), \
             patch.object(rag, 'init_collection', return_value=True), \
             patch.object(rag, 'ChunkConfig', MagicMock()), \
             patch.object(rag, 'chunk_files', return_value=[fake_chunk]):
            result = rag.index_content(
                'ws-a', ['/tmp'], content_type='archive', content_tier='archive-only',
            )

        assert result['indexed'] == 1
        assert result['content_tier'] == 'archive-only'
        points = fake_vs.upsert_points.call_args[0][1]
        assert points[0].content_tier == 'archive-only'
        assert points[0].content_hash is not None

    def test_default_content_tier_is_topic_specific(self):
        import rag
        import inspect
        sig = inspect.signature(rag.index_content)
        assert sig.parameters['content_tier'].default == 'topic-specific'


class TestSearchArchiveOnlyReachesArchiveTier:
    def test_search_archive_requests_only_archive_tier(self):
        import rag

        fake_vs = MagicMock()
        fake_vs.search.return_value = []
        fake_vs.keyword_search.return_value = []
        fake_model = MagicMock()
        fake_model.encode.return_value.tolist.return_value = [0.0] * 4

        with patch.object(rag, 'VECTOR_STORE_AVAILABLE', True), \
             patch.object(rag, 'get_vector_store', return_value=fake_vs), \
             patch.object(rag, '_get_embedding_model', return_value=fake_model):
            result = rag.search_archive('ws-a', 'find my old notes')

        assert result == ""  # no hits -> empty string, not an error
        _, search_kwargs = fake_vs.search.call_args
        assert search_kwargs['content_tiers'] == ['archive-only']
        _, kw_kwargs = fake_vs.keyword_search.call_args
        assert kw_kwargs['content_tiers'] == ['archive-only']

    def test_get_relevant_context_never_passes_content_tiers(self):
        """get_relevant_context()'s signature must be untouched by this
        change — no include_archive param, no content_tiers threading."""
        import rag
        import inspect
        sig = inspect.signature(rag.get_relevant_context)
        assert 'include_archive' not in sig.parameters
        assert 'content_tiers' not in sig.parameters


# ---------------------------------------------------------------------------
# Live integration (gated) — proves RLS actually isolates workspaces
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("RAGBOT_PGVECTOR_TEST_URL"),
    reason="Set RAGBOT_PGVECTOR_TEST_URL to a postgres DSN to enable live RLS tests.",
)
class TestRLSLive:
    """Live proof that a session scoped to workspace A cannot read
    workspace B's rows, even via a raw query that 'forgot' the
    WHERE clause — the exact property migration 0003 exists to add.
    """

    def setup_method(self):
        from synthesis_engine.vectorstore import reset_vector_store
        reset_vector_store()

    def teardown_method(self):
        from synthesis_engine.vectorstore import reset_vector_store
        reset_vector_store()

    def test_forgotten_where_clause_still_isolates_workspaces(self, monkeypatch):
        from synthesis_engine.vectorstore import get_vector_store, Point

        monkeypatch.setenv("RAGBOT_DATABASE_URL", os.environ["RAGBOT_PGVECTOR_TEST_URL"])
        vs = get_vector_store()
        assert vs is not None

        ws_a = f"rls_a_{uuid.uuid4().hex[:8]}"
        ws_b = f"rls_b_{uuid.uuid4().hex[:8]}"
        vec = [1.0] + [0.0] * 383

        try:
            vs.init_collection(ws_a, vector_size=384)
            vs.init_collection(ws_b, vector_size=384)
            vs.upsert_points(ws_a, [Point(
                chunk_uid="a1", vector=vec, text="workspace A secret",
                source_path="/tmp/a1",
            )])
            vs.upsert_points(ws_b, [Point(
                chunk_uid="b1", vector=vec, text="workspace B secret",
                source_path="/tmp/b1",
            )])

            # Raw connection, scoped to ws_a, running a query that
            # "forgot" the WHERE workspace = ... clause entirely.
            with vs._connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT set_config('app.current_workspace', %s, true)",
                        (ws_a,),
                    )
                    cur.execute("SELECT workspace, text FROM chunks")
                    rows = cur.fetchall()

            workspaces_seen = {r[0] for r in rows}
            assert ws_a in workspaces_seen
            assert ws_b not in workspaces_seen
        finally:
            vs.delete_collection(ws_a)
            vs.delete_collection(ws_b)
