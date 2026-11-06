-- Ragbot pgvector backend — embedding dimension 1024 (migration 0004).
--
-- The embedding model moved from all-MiniLM-L6-v2 (384 dimensions) to
-- microsoft/harrier-oss-v1-0.6b (1,024 dimensions); see the `embedding:`
-- block of engines.yaml. The vector columns follow it.
--
-- Design notes:
--
-- * **Chunks are cleared, then rebuilt by re-indexing.** Vectors from two
--   models live in unrelated spaces: no conversion maps one onto the other,
--   and a 384-dimension row cannot be stored in a 1,024-dimension column. Every
--   chunk row is written by indexing source files (`ragbot index`,
--   `ragbot skills index`), so clearing `chunks` and `documents` loses nothing
--   a re-index does not rebuild. The `workspaces` rows are kept.
--
-- * **Entity embeddings are cleared; entities are kept.** `entities.embedding`
--   is optional and caller-supplied. The entities, their attributes and their
--   relations are memory, not derived data, and stay untouched.
--
-- * **Runs only when the dimension differs.** The runner executes every
--   migration file on every start, so each change is guarded by the column's
--   current dimension (pgvector stores it as the type modifier). On a database
--   already at 1,024 this file changes nothing; on a fresh one it converts the
--   empty tables 0001 and 0002 just created.
--
-- * **The HNSW index is rebuilt for the new type.** Same parameters as 0001.

DO $$
BEGIN
    IF (SELECT atttypmod FROM pg_attribute
        WHERE attrelid = 'chunks'::regclass AND attname = 'embedding') <> 1024 THEN
        DROP INDEX IF EXISTS idx_chunks_embedding_hnsw;
        TRUNCATE chunks, documents;
        ALTER TABLE chunks ALTER COLUMN embedding TYPE VECTOR(1024);
        CREATE INDEX idx_chunks_embedding_hnsw
            ON chunks USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
    END IF;

    IF (SELECT atttypmod FROM pg_attribute
        WHERE attrelid = 'entities'::regclass AND attname = 'embedding') <> 1024 THEN
        UPDATE entities SET embedding = NULL WHERE embedding IS NOT NULL;
        ALTER TABLE entities ALTER COLUMN embedding TYPE VECTOR(1024);
    END IF;
END
$$;

-- Record migration as applied.
INSERT INTO schema_migrations (version, description)
VALUES (4, 'embedding dimension 1024 (harrier-oss-v1-0.6b): chunks cleared for re-index, entity embeddings cleared')
ON CONFLICT (version) DO NOTHING;
