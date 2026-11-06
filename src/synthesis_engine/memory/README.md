# synthesis_engine.memory — three-tier memory architecture

The memory layer of `synthesis_engine` goes beyond pure vector RAG. It
ships a three-tier stack with explicit provenance and bi-temporal facts,
and a pluggable backend interface so alternative implementations
(Mem0, Letta, Zep/Graphiti, custom stores) can slot in without changes
to retrieval or the API router.

## Architecture

```
   Tier 1   Vector RAG over chunks            (synthesis_engine.vectorstore)
   Tier 2   Entity graph with bi-temporal     entities + relations
            relations and provenance
   Tier 3a  Session memory                    session_memory
   Tier 3b  User-scoped persistent memory     user_memory
```

The three tiers are merged into a single ranked list by
`three_tier_retrieve`. Each result is tagged with the tier of origin and
carries the provenance that supports it.

## Write path

Every relation REQUIRES a `Provenance` value with at minimum
`source` and `confidence`. The interface refuses to write facts without
this, so the agent loop and the consolidation pass cannot drop
attribution.

Per-attribute provenance on entities is encoded in the attributes jsonb:

```python
Entity(
    workspace="ws",
    type="person",
    name="Alice",
    attributes={
        "role": AttributeValue(
            value="engineer",
            provenance=Provenance(source="manual:admin", confidence=1.0),
        ),
    },
)
```

Relation provenance lives on the row itself:

```python
Relation(
    workspace="ws",
    from_entity=alice.id,
    to_entity=acme.id,
    type="works_at",
    provenance=Provenance(
        source="session:abc-123",
        agent_run_id=uuid.UUID("..."),
        message_id="msg-7",
        confidence=0.92,
    ),
)
```

## Bi-temporal supersession

Relations are immutable append-only. A new fact never deletes an older
one. Two writes happen inside one transaction when superseding:

1. The prior row's `validity_end` is set to `now()`.
2. The new row is inserted with `validity_start = now()`,
   `validity_end = NULL`.
3. An audit `supersedes` relation is recorded in the same transaction.

```python
r1 = memory.upsert_relation(Relation(..., type="works_at", from=Alice, to=Acme))
# ... time passes ...
r2 = memory.upsert_relation(
    Relation(..., type="works_at", from=Alice, to=Beta),
    supersedes=r1.id,
)
```

Queries default to current facts (`validity_end IS NULL`). Time-travel
queries supply a `validity_at` timestamp:

```python
# What did the graph say at midpoint?
historical = memory.query_graph(ws, seed_entity_ids=[alice.id], validity_at=t1+5d)
```

### Why bi-temporal append-only instead of tombstoned rows

The two approaches considered:

* **Approach A — bi-temporal append-only with `validity_start` /
  `validity_end`.** Old rows stay; their validity window closes. Audit
  trail is automatic. Time-travel queries are a `WHERE ... AND
  validity_start <= ts AND (validity_end IS NULL OR validity_end > ts)`
  predicate. Matches the Mem0 / Zep / Graphiti shape, so backends can
  swap.

* **Approach B — tombstoned rows with `superseded_by_id` self-reference.**
  Conceptually simple ("give me rows where superseded_by_id is null"),
  but no native `as-of` query — you'd walk the linked list with a
  recursive CTE. Worse: concurrent supersessions race on a pointer
  rather than collapsing onto a clean validity window.

We chose A. The two-write transaction is the small cost; the audit
trail, time-travel queries, and Mem0/Zep compatibility are the gains.

## Consolidation ("dreaming") pass

Between sessions, `consolidate_session(memory, session_id, workspace)`:

1. Reads the session's payload.
2. Calls the configured LLM backend to extract a JSON document of
   `{entities: [...], relations: [...]}`.
3. Writes each entity and relation with provenance
   `source="consolidation:session=<session_id>"`.

The pass is designed to be invoked by Ragenie scheduled routines, a
cron job, or any other scheduler. It is idempotent: rerunning over the
same session re-extracts; the upsert path deduplicates entities by
`(workspace, type, name)` and appends dated relations (it never
deletes).

For tests and rule-based consolidators, pass an `extractor` callable
that returns the same JSON shape without invoking an LLM.

## Schema

Migrations: `vectorstore/migrations/0002_memory_entities.sql`; `0004_embedding_dimensions.sql` moved the embedding column to 1,024 dimensions.

```
entities         (id, workspace, type, name, attributes jsonb,
                  embedding vector(1024) nullable, created_at, updated_at)
                 UNIQUE (workspace, type, name)
                 HNSW on embedding WHERE embedding IS NOT NULL
                 GIN on attributes

relations        (id, workspace, from_entity, to_entity, type,
                  attributes jsonb, validity_start, validity_end,
                  provenance jsonb NOT NULL, created_at)
                 FKs to entities ON DELETE CASCADE
                 Partial btree on (workspace, from_entity, to_entity)
                   WHERE validity_end IS NULL    (hot path: current facts)
                 GiST on tstzrange(validity_start, COALESCE(validity_end, infinity))
                   for `validity_at` time-travel queries

session_memory   (session_id, user_id, workspace, payload jsonb, ts)
user_memory      (user_id, payload jsonb, ts)
```

Example rows:

```text
entities
  id=...  workspace='personal'  type='person'      name='Alice'
          attributes={"role":{"value":"engineer", "provenance":{...},
                              "updated_at":"2026-05-13T11:42:00Z"}}

relations
  id=r1   workspace='personal'  from=Alice  to=Acme  type='works_at'
          validity_start='2024-01-01'  validity_end='2026-05-10'
          provenance={"source":"session:abc","confidence":0.95}
  id=r2   workspace='personal'  from=Alice  to=Beta  type='works_at'
          validity_start='2026-05-10'  validity_end=NULL  (current)
          provenance={"source":"session:def","confidence":0.95}
```

## Migration ordering

Migrations are applied numerically by `PgvectorBackend._run_migrations`.
The runner is idempotent — every `CREATE TABLE`, `CREATE INDEX`, and
`INSERT INTO schema_migrations` is `IF NOT EXISTS` / `ON CONFLICT DO
NOTHING`.

* `0001_initial.sql` — workspaces, documents, chunks (Tier 1).
* `0002_memory_entities.sql` — entities, relations, session_memory,
  user_memory (Tier 2, Tier 3).

`0002` depends on the `set_updated_at` function defined in `0001` and
on the `vector` extension; running them out of order would fail
deterministically rather than silently.

## Pluggability story

The `Memory` ABC in `base.py` defines the per-tier reads and writes
every backend implements. The default `PgvectorMemory` ships in
`pgvector_memory.py` and reuses the existing pgvector connection pool.

To add an alternative backend (Mem0, Letta, Zep/Graphiti, etc.):

```python
from synthesis_engine.memory.base import Memory

class Mem0Memory(Memory):
    backend_name = "mem0"

    def upsert_entity(self, entity): ...
    def upsert_relation(self, relation, *, supersedes=None): ...
    def query_graph(self, workspace, *, seed_entity_ids, depth=2,
                    validity_at=None, relation_types=None, limit=200): ...
    # ... etc ...

    def search_three_tier(self, query, *, query_vector=None):
        # Either delegate to retrieval.three_tier_retrieve(self, query),
        # which composes the per-tier reads above, or call the backend's
        # own native merged-query endpoint.
        ...
```

Register the new backend behind the `RAGBOT_MEMORY_BACKEND` env var (or
its substrate equivalent in another runtime). The API router and the
agent loop do not need to change.

## API surface

```
GET  /api/memory/entities?workspace=X[&type=Y][&limit=N][&offset=M]
GET  /api/memory/entities/{id}        -> entity + incoming/outgoing relations
POST /api/memory/entities             -> upsert (admin-style)
GET  /api/memory/query?q=...&workspace=X[&session_id=...][&user_id=...]
GET  /api/memory/session/{session_id}
PUT  /api/memory/session/{session_id}
```

All endpoints scope by `workspace` (same convention as the rest of
ragbot). Authentication is intentionally out of scope at this layer;
the runtime that mounts the router enforces the calling identity.

## Substrate cleanliness

`synthesis_engine.memory` does not depend on any ragbot module. It
depends only on:

* `synthesis_engine.vectorstore` for the Tier 1 vector substrate and
  the shared pgvector connection pool.
* `synthesis_engine.llm` for the consolidation pass.
* `pydantic` for the typed boundary.

Ragenie, synthesis-console, and any other runtime built on
`synthesis_engine` consume `from synthesis_engine.memory import ...`
directly.
