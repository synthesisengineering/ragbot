# Row-level security and the two Postgres roles

Migration `0003_content_tiers.sql` adds Row-Level Security (RLS) policies on
`documents` and `chunks` that isolate rows by workspace at the database
layer, as a second line of defense behind the `WHERE workspace = %s` filter
every query in `pgvector_backend.py` already applies explicitly. This
document explains why that only works if the app's runtime connection uses
the right Postgres role, and how to configure it.

## The rule that makes this necessary

PostgreSQL has a hard, non-configurable rule: **row security is always
disabled for superusers.** `ALTER TABLE ... FORCE ROW LEVEL SECURITY` does
not override this — `FORCE` only extends RLS enforcement to the table
*owner* when the owner is not a superuser. It has no effect on a superuser
connection at all.

The official `postgres` and `pgvector/pgvector` Docker images always create
`POSTGRES_USER` as a superuser. If the application's runtime queries connect
as that same role — which is what `RAGBOT_DATABASE_URL` pointed at before
this change — every RLS policy on `documents`/`chunks` is a silent no-op:
`ENABLE ROW LEVEL SECURITY`, `FORCE ROW LEVEL SECURITY`, and the isolation
policy itself all evaluate as if they weren't there. A raw query that
"forgets" its `WHERE workspace = ...` clause (a bug, a future consumer of
the same schema, a copy-paste mistake) sees every workspace's rows, not
zero. This is not a bug in the migration's SQL — the policies are written
correctly — it is a property of which role executes the query.

## The fix: two roles, two purposes

| Role | Created by | Used for | Privileges |
|---|---|---|---|
| `ragbot` (or whatever `POSTGRES_USER` is set to) | Postgres init (docker image) | Applying migrations only | Superuser |
| `ragbot_app` | Migration `0003_content_tiers.sql`, idempotently | The app's runtime queries — search, upsert, memory reads/writes | `SELECT`/`INSERT`/`UPDATE`/`DELETE` on `workspaces`, `documents`, `chunks`, `entities`, `relations`, `session_memory`, `user_memory`; `USAGE`/`SELECT` on sequences. No `CREATE`, no role/extension management, no other schemas. `NOSUPERUSER`, `NOBYPASSRLS`. |

Migrations legitimately need elevated privileges that `ragbot_app` must not
have: `CREATE ROLE` (to create `ragbot_app` itself), `ALTER TABLE ... ENABLE/
FORCE ROW LEVEL SECURITY`, `CREATE EXTENSION`, and schema-level `GRANT`s.
Because `ragbot_app` is not a superuser and not the table owner, it is fully
subject to RLS the moment `ENABLE ROW LEVEL SECURITY` is set — `FORCE` isn't
even required for it, though the migration keeps `FORCE` in place as
defense-in-depth in case a future deployment ever points the app connection
at the owning role instead.

## Env vars

- **`RAGBOT_DATABASE_URL`** — the application's runtime DSN. In any real
  deployment this should be a `postgresql://ragbot_app:<password>@.../<db>`
  connection string.
- **`RAGBOT_MIGRATION_DATABASE_URL`** — the elevated DSN migrations run
  over. Falls back to `RAGBOT_DATABASE_URL` when unset — required for a
  brand-new database, since nothing else can create `ragbot_app` the first
  time. Once both roles exist, set this explicitly to the superuser/owner
  connection string and keep `RAGBOT_DATABASE_URL` pointed at `ragbot_app`.
- **`RAGBOT_APP_DB_PASSWORD`** — the password provisioned for `ragbot_app`
  the first time migrations run (`CREATE ROLE ragbot_app LOGIN PASSWORD
  ...`). Keep it in sync with the password embedded in
  `RAGBOT_DATABASE_URL`. If unset, a random password is generated and logged
  as a warning — the role still gets created, but nothing will be able to
  authenticate as it until this is set to match `RAGBOT_DATABASE_URL`.

`docker-compose.yml` and `.env.example` wire up all three for local dev with
the correct split by default, so a fresh `docker compose up -d` demonstrates
the intended pattern rather than silently defaulting back to a superuser
connection.

## Never do this

Never point `RAGBOT_DATABASE_URL` (the app's runtime DSN) at the same role
used for `RAGBOT_MIGRATION_DATABASE_URL`, once `ragbot_app` exists. Doing so
reintroduces the exact bypass this document describes — the RLS policies
stay in the schema, but the connecting role ignores them.

## Verifying it live

`tests/test_content_tiers.py::TestRLSLive::test_forgotten_where_clause_still_isolates_workspaces`
is the live proof: it opens a raw connection scoped to workspace A via
`set_config('app.current_workspace', ...)`, runs `SELECT workspace, text
FROM chunks` with no `WHERE` clause at all, and asserts workspace B's rows
are absent. It's gated behind `RAGBOT_PGVECTOR_TEST_URL`; point that at the
`ragbot_app` role (with `RAGBOT_MIGRATION_DATABASE_URL` and
`RAGBOT_APP_DB_PASSWORD` also set, e.g. in the same shell) to exercise the
real, hardened path rather than the superuser role, which will pass this
test for the wrong reason — it can see workspace B too, it just happens not
to be asked to in the test's own scoped queries elsewhere in the suite.
