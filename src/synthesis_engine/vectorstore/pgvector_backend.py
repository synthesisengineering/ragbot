"""Pgvector backend.

PostgreSQL with the ``pgvector`` extension. Implements the :class:`VectorStore`
interface with:

    * Vector ANN search via cosine distance (HNSW index).
    * Native full-text search via tsvector + GIN (replaces in-process BM25).
    * Single shared schema across workspaces, scoped by a ``workspace`` column.
    * Connection pooling via psycopg's ``ConnectionPool``.
    * Idempotent migration runner that applies SQL files in
      ``vectorstore/migrations/`` on first use.

The schema is documented in ``migrations/0001_initial.sql``.
"""

from __future__ import annotations

import logging
import os
import re
import secrets
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..observability import record_retrieval_result, retrieval_span
from . import DEFAULT_VISIBLE_CONTENT_TIERS, Point, SearchHit, VectorStore

logger = logging.getLogger(__name__)


def _resolve_tiers(content_tiers: Optional[Sequence[str]]) -> List[str]:
    """Resolve the effective tier filter for a query.

    ``None`` means "the caller didn't ask for a specific tier" — apply the
    safe default (excludes archive-only). A caller that wants archive-tier
    content must pass an explicit sequence including it.
    """
    if content_tiers is None:
        return list(DEFAULT_VISIBLE_CONTENT_TIERS)
    return list(content_tiers)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _resolve_database_url() -> Optional[str]:
    """Resolve the APPLICATION (runtime) connection string from env vars.

    This is the DSN used for every query the app issues at runtime —
    search, upsert, memory reads/writes, everything except applying
    migrations. In any deployment that has adopted the RLS hardening in
    ``0003_content_tiers.sql``, this should point at the restricted,
    non-superuser ``ragbot_app`` role that migration creates — see
    ``docs/rls-and-roles.md``. Migrations themselves run over a separate,
    intentionally more-privileged DSN; see ``_resolve_migration_database_url``.

    Priority:
      1. ``RAGBOT_DATABASE_URL``  (preferred — explicit, ragbot-scoped)
      2. ``DATABASE_URL``         (general fallback)
    """

    return (
        os.environ.get("RAGBOT_DATABASE_URL")
        or os.environ.get("DATABASE_URL")
    )


def _resolve_migration_database_url() -> Optional[str]:
    """Resolve the ELEVATED connection string used to apply migrations.

    Migrations need privileges the restricted ``ragbot_app`` runtime role
    deliberately does not have: ``CREATE ROLE``, ``ALTER TABLE ... ENABLE/
    FORCE ROW LEVEL SECURITY``, ``CREATE EXTENSION``, schema-level GRANTs.
    Reusing the app's runtime DSN for migrations would either fail outright
    (once that role is genuinely restricted) or — on a fresh database —
    be a chicken-and-egg problem, since the restricted role doesn't exist
    until a migration creates it.

    Falls back to the application DSN (`_resolve_database_url`) when unset,
    which reproduces today's single-DSN behavior for setups that haven't
    split the two roles yet (including a brand-new database where
    ``RAGBOT_DATABASE_URL`` itself must be the elevated/owner role, since
    nothing else can create ``ragbot_app`` for the first time).
    """

    return os.environ.get("RAGBOT_MIGRATION_DATABASE_URL") or _resolve_database_url()


# Placeholder token embedded in migration SQL (see 0003_content_tiers.sql)
# where the `ragbot_app` role's password needs to be substituted at
# apply-time. Migrations are executed as plain multi-statement SQL text
# (see `_run_migrations`), not through psycopg's parameter binding, so a
# secret a migration needs can't be passed as a normal bind parameter —
# this is the mechanism instead.
_APP_ROLE_PASSWORD_PLACEHOLDER = "__RAGBOT_APP_DB_PASSWORD__"


def _resolve_app_role_password() -> str:
    """Resolve the password to provision for the restricted `ragbot_app`
    Postgres role created idempotently by migration 0003.

    Read from ``RAGBOT_APP_DB_PASSWORD`` so operators can keep it in sync
    with the connection string ``RAGBOT_DATABASE_URL`` should use in any
    real deployment (docker-compose.yml and .env.example derive both from
    the same value). If unset, a random password is generated so the role
    is still created with a working — if unknown — credential rather than
    a predictable one. Nothing will be able to log in as ``ragbot_app``
    until ``RAGBOT_APP_DB_PASSWORD`` and ``RAGBOT_DATABASE_URL`` are set to
    match, so this path logs loudly rather than silently degrading.
    """

    pw = os.environ.get("RAGBOT_APP_DB_PASSWORD")
    if pw:
        return pw
    generated = secrets.token_urlsafe(24)
    logger.warning(
        "RAGBOT_APP_DB_PASSWORD is not set; generated a random password for "
        "the 'ragbot_app' restricted role created by migration "
        "0003_content_tiers.sql. Set RAGBOT_APP_DB_PASSWORD explicitly (and "
        "point RAGBOT_DATABASE_URL at postgresql://ragbot_app:<that "
        "password>@.../<db>) so the row-level-security hardening actually "
        "applies to the app's runtime connection — see docs/rls-and-roles.md."
    )
    return generated


def _substitute_migration_secrets(sql: str, conn) -> str:
    """Replace secret placeholders in migration SQL with real values.

    Currently the only placeholder is `_APP_ROLE_PASSWORD_PLACEHOLDER`. A
    no-op (returns ``sql`` unchanged) for every migration file that doesn't
    reference it, which is all of them except 0003_content_tiers.sql.
    """

    if _APP_ROLE_PASSWORD_PLACEHOLDER not in sql:
        return sql

    from psycopg import sql as psycopg_sql  # local import, mirrors _to_jsonb style

    password = _resolve_app_role_password()
    literal = psycopg_sql.Literal(password).as_string(conn)
    return sql.replace(_APP_ROLE_PASSWORD_PLACEHOLDER, literal)


def _migrations_dir() -> Path:
    return Path(__file__).resolve().parent / "migrations"


# ---------------------------------------------------------------------------
# Backend implementation
# ---------------------------------------------------------------------------


class PgvectorBackend(VectorStore):
    backend_name = "pgvector"

    def __init__(
        self,
        dsn: str,
        *,
        migration_dsn: Optional[str] = None,
        min_size: int = 1,
        max_size: int = 8,
    ) -> None:
        # `dsn` is the APPLICATION (runtime) connection string — everything
        # the app pool serves (search/upsert/memory ops). `migration_dsn` is
        # the ELEVATED connection string migrations run over; it defaults to
        # `dsn` so a single-DSN setup (no role split configured) keeps
        # working exactly as before. See `_resolve_migration_database_url`.
        self.dsn = dsn
        self.migration_dsn = migration_dsn or dsn
        self._pool = None
        self._pool_lock = threading.Lock()
        self._migrated = False
        self._min_size = min_size
        self._max_size = max_size

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "PgvectorBackend":
        """Build a backend from env vars, raising if no DSN is configured."""

        dsn = _resolve_database_url()
        if not dsn:
            raise RuntimeError(
                "Pgvector backend requires RAGBOT_DATABASE_URL (or DATABASE_URL). "
                "Set it to a postgres:// connection string and ensure the "
                "pgvector extension is installable on the target server."
            )
        migration_dsn = _resolve_migration_database_url()
        backend = cls(dsn, migration_dsn=migration_dsn)
        # Migrations MUST run before the app pool is opened: on a fresh
        # database (or the first run after adopting the RLS hardening in
        # 0003_content_tiers.sql), the restricted `ragbot_app` role that
        # `dsn` may point at doesn't exist yet — only the migration
        # (running over the elevated `migration_dsn`) creates it. Opening
        # the app pool first would fail to authenticate.
        backend._run_migrations()
        # Eagerly validate app-pool connectivity. Fail fast if the database
        # is unreachable so get_vector_store() returns None and callers in
        # rag.py degrade to chat-only without RAG.
        backend._ensure_pool()
        return backend

    # ------------------------------------------------------------------
    # pool / migrations
    # ------------------------------------------------------------------

    def _ensure_pool(self):
        if self._pool is not None:
            return self._pool
        with self._pool_lock:
            if self._pool is not None:
                return self._pool
            try:
                from psycopg_pool import ConnectionPool  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "psycopg[pool] is required for the pgvector backend. "
                    "Install it via the project's requirements.txt."
                ) from exc

            self._pool = ConnectionPool(
                conninfo=self.dsn,
                min_size=self._min_size,
                max_size=self._max_size,
                kwargs={"autocommit": False},
                # Open lazily so import-time test failures don't blow up the
                # whole module; the first acquire() validates the connection.
                open=False,
            )
            self._pool.open()
            return self._pool

    def _connection(self):
        """Acquire a pooled connection with the pgvector type registered."""

        # Local import so the module loads even if pgvector isn't installed.
        from pgvector.psycopg import register_vector  # type: ignore

        pool = self._ensure_pool()
        ctx = pool.connection()
        # The context manager returned by `pool.connection()` will yield a
        # connection on entry; we wrap it so register_vector runs each time.
        return _VectorConn(ctx, register_vector)

    def _run_migrations(self) -> None:
        if self._migrated:
            return
        migrations = sorted(_migrations_dir().glob("*.sql"))
        if not migrations:
            logger.warning("No migrations found for pgvector backend.")
            self._migrated = True
            return

        import psycopg  # local import: keep module importable without psycopg installed

        # Migrations run over a DEDICATED, non-pooled connection using the
        # elevated `migration_dsn` — deliberately NOT the shared app pool
        # (`self._connection()` / `self.dsn`). In any deployment that has
        # adopted the RLS hardening below, the app pool connects as the
        # restricted `ragbot_app` role, which lacks the privileges these
        # migrations need (CREATE ROLE, ALTER TABLE ... ENABLE/FORCE ROW
        # LEVEL SECURITY, CREATE EXTENSION) — and on a fresh database,
        # `ragbot_app` doesn't exist yet for the pool to even authenticate
        # as until this method's first run creates it.
        with psycopg.connect(self.migration_dsn, autocommit=False) as conn:
            with conn.cursor() as cur:
                for path in migrations:
                    sql = path.read_text()
                    sql = _substitute_migration_secrets(sql, conn)
                    logger.info("Applying migration %s", path.name)
                    cur.execute(sql)
            conn.commit()
        self._migrated = True

    # ------------------------------------------------------------------
    # Row-Level Security session context (migration 0003)
    # ------------------------------------------------------------------
    #
    # RLS policies on `documents` and `chunks` (see 0003_content_tiers.sql)
    # require `app.current_workspace` (or the `app.rls_admin` escape
    # hatch) to be set for the current transaction before any query
    # touches those tables — otherwise the policies default-deny and the
    # query silently sees zero rows. `SET LOCAL`-equivalent semantics
    # (via `set_config(..., is_local=true)`) are used so the setting
    # reverts at the end of the transaction and never leaks to the next
    # borrower of a pooled connection.
    #
    # `set_config()` (a normal function call) is used instead of the bare
    # `SET` statement because psycopg's parameter binding does not support
    # `SET x = %s` — SET is a utility statement, not a normal expression
    # position, so the value can't be passed as a bind parameter. See
    # https://www.psycopg.org/docs/usage.html and the info faq re: SET.

    def _set_workspace_context(self, cur, workspace: str) -> None:
        """Bind `app.current_workspace` for the current transaction."""
        cur.execute(
            "SELECT set_config('app.current_workspace', %s, true)",
            (workspace,),
        )

    def _set_admin_context(self, cur) -> None:
        """Bind the `app.rls_admin` escape hatch for genuinely
        cross-workspace admin queries (e.g. list_collections()).

        Only call this for operations that have no single workspace to
        scope to. Everything else should use `_set_workspace_context`.
        """
        cur.execute("SELECT set_config('app.rls_admin', 'on', true)")

    # ------------------------------------------------------------------
    # interface implementations
    # ------------------------------------------------------------------

    def init_collection(self, workspace: str, vector_size: int = 384) -> bool:
        # Schema is shared; "init" simply ensures the workspace row exists
        # and migrations are applied. The vector_size argument is informational
        # (the schema fixes the embedding dimension; mismatches will error
        # when upserting, which is the right loud failure).
        try:
            self._run_migrations()
            with self._connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO workspaces (name) VALUES (%s) "
                        "ON CONFLICT (name) DO NOTHING",
                        (workspace,),
                    )
                conn.commit()
            return True
        except Exception as exc:
            logger.error("PgvectorBackend.init_collection failed: %s", exc)
            return False

    def upsert_points(self, workspace: str, points: List[Point]) -> int:
        if not points:
            return 0
        # Group points by source_path so each unique document gets a single row.
        documents_by_path: Dict[str, Point] = {}
        for p in points:
            if not p.source_path:
                continue
            documents_by_path.setdefault(p.source_path, p)

        try:
            with self._connection() as conn:
                with conn.cursor() as cur:
                    self._set_workspace_context(cur, workspace)

                    # Upsert documents, returning their ids.
                    doc_ids: Dict[str, int] = {}
                    for path, ref in documents_by_path.items():
                        cur.execute(
                            """
                            INSERT INTO documents
                                (workspace, source_path, filename, title,
                                 content_type, embedding_model, indexed_at)
                            VALUES (%s, %s, %s, %s, %s, %s, now())
                            ON CONFLICT (workspace, source_path) DO UPDATE
                                SET filename       = EXCLUDED.filename,
                                    title          = EXCLUDED.title,
                                    content_type   = EXCLUDED.content_type,
                                    embedding_model = EXCLUDED.embedding_model,
                                    indexed_at     = now()
                            RETURNING id
                            """,
                            (
                                workspace,
                                path,
                                ref.filename or os.path.basename(path),
                                ref.title,
                                ref.content_type,
                                ref.embedding_model,
                            ),
                        )
                        doc_ids[path] = cur.fetchone()[0]

                    # Upsert chunks. Re-indexing replaces existing chunk rows
                    # for a (workspace, chunk_uid) pair.
                    written = 0
                    for p in points:
                        document_id = doc_ids.get(p.source_path) if p.source_path else None
                        if document_id is None:
                            # No source file? Create a synthetic document row.
                            cur.execute(
                                """
                                INSERT INTO documents
                                    (workspace, source_path, filename, title,
                                     content_type, embedding_model, indexed_at)
                                VALUES (%s, %s, %s, %s, %s, %s, now())
                                ON CONFLICT (workspace, source_path) DO UPDATE
                                    SET indexed_at = now()
                                RETURNING id
                                """,
                                (
                                    workspace,
                                    p.chunk_uid,  # use chunk_uid as the synthetic source_path
                                    p.filename or "",
                                    p.title,
                                    p.content_type,
                                    p.embedding_model,
                                ),
                            )
                            document_id = cur.fetchone()[0]

                        cur.execute(
                            """
                            INSERT INTO chunks
                                (document_id, workspace, chunk_index, chunk_uid,
                                 text, char_start, char_end,
                                 embedding, embedding_model,
                                 content_type, filename, title, metadata,
                                 content_tier, content_hash, embedding_model_version)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (workspace, chunk_uid) DO UPDATE
                                SET text            = EXCLUDED.text,
                                    chunk_index     = EXCLUDED.chunk_index,
                                    char_start      = EXCLUDED.char_start,
                                    char_end        = EXCLUDED.char_end,
                                    embedding       = EXCLUDED.embedding,
                                    embedding_model = EXCLUDED.embedding_model,
                                    content_type    = EXCLUDED.content_type,
                                    filename        = EXCLUDED.filename,
                                    title           = EXCLUDED.title,
                                    metadata        = EXCLUDED.metadata,
                                    document_id     = EXCLUDED.document_id,
                                    content_tier    = EXCLUDED.content_tier,
                                    content_hash    = EXCLUDED.content_hash,
                                    embedding_model_version = EXCLUDED.embedding_model_version
                            """,
                            (
                                document_id,
                                workspace,
                                p.chunk_index,
                                p.chunk_uid,
                                p.text,
                                p.char_start,
                                p.char_end,
                                p.vector,
                                p.embedding_model,
                                p.content_type,
                                p.filename,
                                p.title,
                                _to_jsonb(p.metadata),
                                p.content_tier,
                                p.content_hash,
                                p.embedding_model_version,
                            ),
                        )
                        written += 1
                conn.commit()
            return written
        except Exception as exc:
            logger.error("PgvectorBackend.upsert_points failed: %s", exc)
            return 0

    def search(
        self,
        workspace: str,
        query_vector: List[float],
        limit: int = 10,
        content_type: Optional[str] = None,
        content_tiers: Optional[Sequence[str]] = None,
    ) -> List[SearchHit]:
        tiers = _resolve_tiers(content_tiers)
        with retrieval_span(
            workspace=workspace,
            k=limit,
            tier="vector",
            backend=self.backend_name,
            content_type=content_type,
            extra={"synthesis.retrieval.vector_dim": len(query_vector or [])},
        ) as span:
            try:
                with self._connection() as conn:
                    with conn.cursor() as cur:
                        self._set_workspace_context(cur, workspace)
                        if content_type:
                            cur.execute(
                                """
                                SELECT text,
                                       1 - (embedding <=> %s::vector) AS score,
                                       chunk_uid, chunk_index, char_start, char_end,
                                       filename, title, content_type, metadata
                                FROM chunks
                                WHERE workspace = %s AND content_type = %s
                                  AND content_tier = ANY(%s)
                                ORDER BY embedding <=> %s::vector
                                LIMIT %s
                                """,
                                (query_vector, workspace, content_type, tiers, query_vector, limit),
                            )
                        else:
                            cur.execute(
                                """
                                SELECT text,
                                       1 - (embedding <=> %s::vector) AS score,
                                       chunk_uid, chunk_index, char_start, char_end,
                                       filename, title, content_type, metadata
                                FROM chunks
                                WHERE workspace = %s
                                  AND content_tier = ANY(%s)
                                ORDER BY embedding <=> %s::vector
                                LIMIT %s
                                """,
                                (query_vector, workspace, tiers, query_vector, limit),
                            )
                        hits = [_row_to_hit(row) for row in cur.fetchall()]
                        top_score = hits[0].score if hits else None
                        record_retrieval_result(
                            span,
                            result_count=len(hits),
                            top_score=top_score,
                            workspace=workspace,
                            tier="vector",
                        )
                        return hits
            except Exception as exc:
                logger.error("PgvectorBackend.search failed: %s", exc)
                record_retrieval_result(
                    span, result_count=0, workspace=workspace, tier="vector",
                )
                return []

    def keyword_search(
        self,
        workspace: str,
        query: str,
        limit: int = 10,
        content_type: Optional[str] = None,
        content_tiers: Optional[Sequence[str]] = None,
    ) -> List[SearchHit]:
        # Use websearch_to_tsquery: forgiving parser, handles bare strings
        # like "show me my biography" without forcing & between terms.
        tiers = _resolve_tiers(content_tiers)
        with retrieval_span(
            workspace=workspace,
            query=query,
            k=limit,
            tier="keyword",
            backend=self.backend_name,
            content_type=content_type,
        ) as span:
            try:
                with self._connection() as conn:
                    with conn.cursor() as cur:
                        self._set_workspace_context(cur, workspace)
                        if content_type:
                            cur.execute(
                                """
                                SELECT text,
                                       ts_rank(text_search, websearch_to_tsquery('english', %s)) AS score,
                                       chunk_uid, chunk_index, char_start, char_end,
                                       filename, title, content_type, metadata
                                FROM chunks
                                WHERE workspace = %s
                                  AND content_type = %s
                                  AND content_tier = ANY(%s)
                                  AND text_search @@ websearch_to_tsquery('english', %s)
                                ORDER BY score DESC
                                LIMIT %s
                                """,
                                (query, workspace, content_type, tiers, query, limit),
                            )
                        else:
                            cur.execute(
                                """
                                SELECT text,
                                       ts_rank(text_search, websearch_to_tsquery('english', %s)) AS score,
                                       chunk_uid, chunk_index, char_start, char_end,
                                       filename, title, content_type, metadata
                                FROM chunks
                                WHERE workspace = %s
                                  AND content_tier = ANY(%s)
                                  AND text_search @@ websearch_to_tsquery('english', %s)
                                ORDER BY score DESC
                                LIMIT %s
                                """,
                                (query, workspace, tiers, query, limit),
                            )
                        hits = [_row_to_hit(row) for row in cur.fetchall()]
                        top_score = hits[0].score if hits else None
                        record_retrieval_result(
                            span,
                            result_count=len(hits),
                            top_score=top_score,
                            workspace=workspace,
                            tier="keyword",
                        )
                        return hits
            except Exception as exc:
                logger.error("PgvectorBackend.keyword_search failed: %s", exc)
                record_retrieval_result(
                    span, result_count=0, workspace=workspace, tier="keyword",
                )
                return []

    def scroll_documents(
        self,
        workspace: str,
        limit: int = 1000,
        content_type: Optional[str] = None,
        content_tiers: Optional[Sequence[str]] = None,
    ) -> List[SearchHit]:
        tiers = _resolve_tiers(content_tiers)
        try:
            with self._connection() as conn:
                with conn.cursor() as cur:
                    self._set_workspace_context(cur, workspace)
                    if content_type:
                        cur.execute(
                            """
                            SELECT text, 0.0 AS score,
                                   chunk_uid, chunk_index, char_start, char_end,
                                   filename, title, content_type, metadata
                            FROM chunks
                            WHERE workspace = %s AND content_type = %s
                              AND content_tier = ANY(%s)
                            ORDER BY id
                            LIMIT %s
                            """,
                            (workspace, content_type, tiers, limit),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT text, 0.0 AS score,
                                   chunk_uid, chunk_index, char_start, char_end,
                                   filename, title, content_type, metadata
                            FROM chunks
                            WHERE workspace = %s
                              AND content_tier = ANY(%s)
                            ORDER BY id
                            LIMIT %s
                            """,
                            (workspace, tiers, limit),
                        )
                    return [_row_to_hit(row) for row in cur.fetchall()]
        except Exception as exc:
            logger.error("PgvectorBackend.scroll_documents failed: %s", exc)
            return []

    def delete_collection(self, workspace: str, content_tier: Optional[str] = None) -> bool:
        try:
            with self._connection() as conn:
                with conn.cursor() as cur:
                    self._set_workspace_context(cur, workspace)
                    if content_tier is None:
                        # Full wipe — historical behavior. Chunks are deleted
                        # explicitly (not left to FK ON DELETE CASCADE) so
                        # the delete happens under the same RLS-scoped
                        # session context as everything else in this method,
                        # rather than depending on cascade-trigger/RLS
                        # interaction that we haven't verified against a
                        # live database.
                        cur.execute("DELETE FROM chunks WHERE workspace = %s", (workspace,))
                        cur.execute("DELETE FROM documents WHERE workspace = %s", (workspace,))
                        cur.execute("DELETE FROM workspaces WHERE name = %s", (workspace,))
                    else:
                        # Tier-scoped clear: remove only chunks in this tier,
                        # then drop any documents left with zero chunks so
                        # `documents` doesn't accumulate orphaned rows. The
                        # workspace row itself is left intact — other tiers
                        # may still have content.
                        cur.execute(
                            "DELETE FROM chunks WHERE workspace = %s AND content_tier = %s",
                            (workspace, content_tier),
                        )
                        cur.execute(
                            """
                            DELETE FROM documents
                            WHERE workspace = %s
                              AND id NOT IN (
                                  SELECT DISTINCT document_id FROM chunks WHERE workspace = %s
                              )
                            """,
                            (workspace, workspace),
                        )
                conn.commit()
            return True
        except Exception as exc:
            logger.error("PgvectorBackend.delete_collection failed: %s", exc)
            return False

    def list_collections(self) -> List[str]:
        try:
            with self._connection() as conn:
                with conn.cursor() as cur:
                    # Genuinely cross-workspace enumeration — no single
                    # workspace to scope `app.current_workspace` to, so use
                    # the admin escape hatch instead.
                    self._set_admin_context(cur)
                    cur.execute(
                        "SELECT DISTINCT workspace FROM chunks "
                        "UNION SELECT name FROM workspaces "
                        "ORDER BY 1"
                    )
                    return [row[0] for row in cur.fetchall()]
        except Exception as exc:
            logger.error("PgvectorBackend.list_collections failed: %s", exc)
            return []

    def get_collection_info(self, workspace: str) -> Optional[Dict[str, Any]]:
        try:
            with self._connection() as conn:
                with conn.cursor() as cur:
                    self._set_workspace_context(cur, workspace)
                    # Deliberately NOT tier-filtered: this reports total
                    # indexed volume for the workspace (ops visibility),
                    # across every tier including archive-only. Retrieval
                    # safety is enforced in search()/keyword_search()/
                    # scroll_documents(), not in this count.
                    cur.execute(
                        """
                        SELECT count(*) AS chunk_count,
                               count(DISTINCT document_id) AS document_count,
                               max(updated_at) AS last_indexed
                        FROM chunks
                        WHERE workspace = %s
                        """,
                        (workspace,),
                    )
                    row = cur.fetchone()
            if not row or row[0] == 0:
                return None
            return {
                "backend": self.backend_name,
                "workspace": workspace,
                "count": row[0],
                "document_count": row[1],
                "last_indexed": row[2].isoformat() if row[2] else None,
            }
        except Exception as exc:
            logger.error("PgvectorBackend.get_collection_info failed: %s", exc)
            return None

    def healthcheck(self) -> Dict[str, Any]:
        try:
            with self._connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
                    row = cur.fetchone()
                    cur.execute("SELECT count(*) FROM workspaces")
                    workspaces_count = cur.fetchone()[0]
            return {
                "backend": self.backend_name,
                "ok": row is not None,
                "pgvector_version": row[0] if row else None,
                "workspaces": workspaces_count,
            }
        except Exception as exc:
            return {"backend": self.backend_name, "ok": False, "reason": str(exc)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _VectorConn:
    """Thin wrapper that registers pgvector types on each acquired connection."""

    def __init__(self, ctx, register_vector):
        self._ctx = ctx
        self._register = register_vector
        self._conn = None

    def __enter__(self):
        self._conn = self._ctx.__enter__()
        self._register(self._conn)
        return self._conn

    def __exit__(self, exc_type, exc, tb):
        return self._ctx.__exit__(exc_type, exc, tb)


def _row_to_hit(row) -> SearchHit:
    (
        text,
        score,
        chunk_uid,
        chunk_index,
        char_start,
        char_end,
        filename,
        title,
        content_type,
        metadata,
    ) = row
    md: Dict[str, Any] = dict(metadata or {})
    md.setdefault("text", text)
    md.setdefault("chunk_uid", chunk_uid)
    md.setdefault("chunk_index", chunk_index)
    md.setdefault("char_start", char_start)
    md.setdefault("char_end", char_end)
    md.setdefault("filename", filename)
    md.setdefault("title", title)
    md.setdefault("content_type", content_type)
    md.setdefault("source_file", md.get("source_file"))
    return SearchHit(text=text or "", score=float(score) if score is not None else 0.0, metadata=md)


def _to_jsonb(value: Any):
    """Adapt a Python dict for psycopg JSONB binding.

    psycopg requires explicit Jsonb wrapping; the default text adapter does
    not coerce dict → jsonb. Wrap here so call sites stay readable.
    """

    from psycopg.types.json import Jsonb  # type: ignore

    return Jsonb(value or {})
