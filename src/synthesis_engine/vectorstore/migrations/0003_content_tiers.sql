-- Ragbot pgvector backend — content tiers + RLS hardening (migration 0003).
--
-- Design notes:
--
-- * **Retrieval tiers, independent of content_type.** `content_type`
--   already describes WHAT a chunk is (datasets, runbooks, skill, ...).
--   `content_tier` is an orthogonal dimension describing HOW EAGERLY it
--   should be retrieved:
--
--     'frequent-access' — reserved for future use. Nothing indexes into
--                          this tier today; instructions/runbooks stay in
--                          the system prompt by design (unchanged by this
--                          migration).
--     'topic-specific'  — the existing source/datasets/ content. Default
--                          value, so every row written before this
--                          migration classifies correctly with zero
--                          backfill.
--     'archive-only'    — personal-archive/ and, in owner context,
--                          private-repo content. Excluded from every
--                          default retrieval path (search(),
--                          keyword_search(), scroll_documents() all
--                          default their new content_tiers parameter to
--                          DEFAULT_VISIBLE_CONTENT_TIERS, which excludes
--                          this tier). Reachable only through an explicit
--                          content_tiers=['archive-only'] call — see
--                          rag.search_archive(). Mirrors the MemGPT/Zep
--                          pattern of gating archival memory behind an
--                          explicit search call rather than default
--                          retrieval.
--
-- * **Drift-detection columns.** `content_hash` (sha256 of the chunk's
--   embedded text) and `embedding_model_version` (model name + embedding
--   library version) support future selective re-embedding: re-embed only
--   rows whose source hash changed, and never silently mix embedding-model
--   vintages inside one searchable index. Populated for new rows going
--   forward; existing rows are left NULL rather than backfilled (backfill
--   would require re-embedding the entire corpus, which is out of scope
--   for a metadata-only migration).
--
-- * **Row-Level Security — defense in depth, not a replacement.** Every
--   query in pgvector_backend.py already filters explicitly with
--   `WHERE workspace = %s`. That is application-enforced: a bug in a
--   future query (or a query written by a different synthesis-engine
--   consumer plugged into the same schema) that forgets the WHERE clause
--   would silently cross workspace boundaries. RLS adds a database-level
--   guarantee on top: even a query that forgets the filter sees zero rows
--   unless the session has explicitly bound `app.current_workspace` (via
--   `set_config(..., is_local=true)`, called at the start of every
--   workspace-scoped method in pgvector_backend.py) to match.
--
--   `FORCE ROW LEVEL SECURITY` is used so the policies apply even to the
--   table-owning role the application connects as (by default Postgres
--   exempts the owner from RLS, which would make this a no-op for the
--   app's own connection — the exact case we're defending against).
--
--   A second, admin-only policy (`app.rls_admin`) exists for the one
--   genuinely cross-workspace operation, list_collections(), which by
--   definition has no single workspace to scope to. Both policies are
--   PERMISSIVE (Postgres default), so they combine with OR: a row is
--   visible if EITHER the workspace matches OR the admin flag is set.
--   Neither GUC is set by default, so an un-scoped connection sees
--   nothing from either table — fail-closed.
--
-- Idempotent: every ALTER/CREATE uses IF NOT EXISTS or an existence
-- check; constraints and policies are dropped-and-recreated by name so
-- re-running this migration is safe.

-- ---------------------------------------------------------------------------
-- content_tier + drift-detection columns on chunks
-- ---------------------------------------------------------------------------

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_tier TEXT NOT NULL DEFAULT 'topic-specific';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_hash TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_model_version TEXT;

-- Fixed, small vocabulary (unlike relations.type in 0002, which is an
-- intentionally open DSL). An unrecognized tier value would silently fall
-- outside DEFAULT_VISIBLE_CONTENT_TIERS and become invisible by default —
-- a fail-safe direction — but the CHECK constraint catches the mistake
-- loudly at write time instead of silently misfiling content into an
-- unknown tier.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'chunks_content_tier_check'
    ) THEN
        ALTER TABLE chunks ADD CONSTRAINT chunks_content_tier_check
            CHECK (content_tier IN ('frequent-access', 'topic-specific', 'archive-only'));
    END IF;
END $$;

-- Tier-scoped queries are always paired with a workspace filter in
-- practice (WHERE workspace = %s AND content_tier = ANY(%s)), so the
-- composite index is what the query planner actually uses. A plain
-- content_tier index is added too per the literal ask, useful for any
-- future cross-workspace tier scan (e.g. an operator query like "how much
-- archive-only content exists across every workspace").
CREATE INDEX IF NOT EXISTS idx_chunks_content_tier
    ON chunks (content_tier);
CREATE INDEX IF NOT EXISTS idx_chunks_workspace_content_tier
    ON chunks (workspace, content_tier);

-- ---------------------------------------------------------------------------
-- Row-Level Security hardening on documents + chunks
-- ---------------------------------------------------------------------------

ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents FORCE ROW LEVEL SECURITY;
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE chunks FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS documents_workspace_isolation ON documents;
CREATE POLICY documents_workspace_isolation ON documents
    USING (workspace = current_setting('app.current_workspace', true));

DROP POLICY IF EXISTS documents_admin_bypass ON documents;
CREATE POLICY documents_admin_bypass ON documents
    USING (current_setting('app.rls_admin', true) = 'on');

DROP POLICY IF EXISTS chunks_workspace_isolation ON chunks;
CREATE POLICY chunks_workspace_isolation ON chunks
    USING (workspace = current_setting('app.current_workspace', true));

DROP POLICY IF EXISTS chunks_admin_bypass ON chunks;
CREATE POLICY chunks_admin_bypass ON chunks
    USING (current_setting('app.rls_admin', true) = 'on');

-- Record migration as applied.
INSERT INTO schema_migrations (version, description)
VALUES (3, 'content tiers (frequent-access/topic-specific/archive-only) + drift-detection columns + RLS hardening on documents/chunks')
ON CONFLICT (version) DO NOTHING;
