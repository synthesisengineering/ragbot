# Dependabot Triage — 2026-11 (queue clearance)

Triage record for the 20 Dependabot pull requests open on
`synthesisengineering/ragbot` on 2026-11-06, between 24 and 64 days old. Earlier
passes: [2026-05](dependabot-triage-2026-05.md) (alert-driven) and
[2026-09](dependabot-triage-2026-09.md) (CI enablement). This document is the
source of truth for the decisions below.

## Summary

| Bucket | Count |
| --- | --- |
| Open Dependabot PRs at pass start | 20 (10 pip, 10 npm) |
| Merged | 19 |
| Closed without merge | 1 (#62) |
| Web advisories (`npm audit`) on `main` before the pass | 22: 1 critical, 6 high, 15 moderate |
| Web advisories after this pass | 0 |
| Python advisories (`pip-audit`) after this pass | 13, all `filelock`, blocked upstream (see below) |

Both ecosystems were at their `open-pull-requests-limit` of 10 when the pass
began.

## CI freshness

A green check is evidence only for the tree it ran against.

| Group | CI ran | Against | Changed on `main` since | Treated as |
| --- | --- | --- | --- | --- |
| pip (#95–#106) | 2026-09-21; #101 2026-09-25 | `7b75de3`; #101 `a360a14` | `a360a14` changed source and `requirements.txt` | stale; verified locally |
| npm (#117, #118, #123–#129) | 2026-10-05 and 2026-10-13 | `e64c3e7` | `README.md` only | current |
| #62 | 2026-09-20 | `dab5d13` | — | failing |

## Python group — verified locally, then merged

Two fresh Python 3.12 environments, one from `main` and one from `main` with all
ten floor bumps applied, each installed with `make install`:

- `pip freeze` output identical between the two. Every new floor sits at or
  below what pip already resolves, so no floor dragged another package
  backwards, which is the failure mode recorded in the 2026-09 addendum.
- `pip check` clean in both.
- `make test-fast`: 1015 passed, 13 skipped in both.

#101 (`langchain`) changes the line adjacent to #98 (`langchain-anthropic`), so
it conflicted after #98 merged. Dependabot rebased it, raising its target from
`>=1.4.2` to `>=1.4.8`, and CI passed on the rebased branch before merge.

Merged: #95 `pytest-asyncio>=1.4.0`, #96 `pytest>=9.1.1`,
#98 `langchain-anthropic>=1.7.2`, #100 `boto3>=1.43.97`, #101 `langchain>=1.4.8`,
#102 `transformers>=5.17.0`, #103 `anthropic>=1.6.0`,
#104 `prometheus-client>=0.26.0`, #105 `uvicorn[standard]>=0.53.0`,
#106 `langchain-google-genai>=4.4.0`.

## Web group

Six PRs touched non-adjacent lines and merged without conflict. Their combined
tree, merged exactly as GitHub would, passed `npm ci`, lint, the suite and
`next build` before any of them merged: #128 `jsdom` 30.1.2, #126 `@types/node`
26.6.5, #127 `@next/eslint-plugin-next` 16.4.0, #129 `@vitejs/plugin-react`
6.1.2, #123 `eslint` 10.12.0, #125 `typescript-eslint` 8.71.1.

#117 and #118 conflicted with those six, so Dependabot rebased them. The rebase
also moved both from 5.0.3 to 5.1.0, which clears GHSA-p458-7748-mcp3; 5.0.3
would have left it open. Both merged on green CI.

### #124 — `next`: green in CI, but the build failed

As opened, #124 proposed `next` 16.4.0. CI was green because CI ran lint and
the unit suite but never `next build`. On 2026-11-06, `next build` on 16.4.0
exited 1: Next.js refuses to build a version with known security advisories,
and three published advisories cover `>=16.0.0 <=16.4.0`:

- GHSA-2464-fhr4-w9xw — unauthenticated remote code execution in the Image
  Optimization API with SVG files
- GHSA-673q-mjm6-r5xr — unauthenticated remote code execution in `use cache`
  functions returning typed arrays
- GHSA-p3m5-59mp-jv5r — denial of service in the App Router through Server
  Actions

`main` was on 16.3.5, which those three cover along with eight more, among them
GHSA-vcvr-r3jv-pc5j (remote code execution in `next/og` `ImageResponse`).

Dependabot's rebase moved #124 to 16.4.4, which carries none of these. It
passed `npm ci`, lint, the suite and `next build` together with #117 and #118
locally, and passed CI, before merging. This pass then moves `next` to 16.4.5,
the current release.

**CI change.** The web job now runs `npm run build`. A `next` version that
cannot build should fail CI, not pass it. Next.js cited the npm advisory
database for its refusal, so a version that builds today can stop building once
an advisory against it is published. That failure is the signal the step exists
to give.

### #62 — `typescript` 7: closed

`typescript-eslint` 8.72.1, the latest release, declares a peer range of
`typescript >=4.8.4 <6.1.0`, so `npm ci` fails with `ERESOLVE` against
TypeScript 7. Nothing in this repository can change that. Installing with
`--legacy-peer-deps` would only hide the conflict.

This pass instead moves TypeScript from 5.9.3 to 6.0.3, the newest release the
lint toolchain accepts. `tsc --noEmit`, lint and `next build` pass with no
`tsconfig.json` change. The range is `~6.0.3` rather than `^6.0.3`, so a future
6.1 cannot enter the lockfile ahead of `typescript-eslint`'s support for it. TypeScript 7 becomes
possible once a `typescript-eslint` release accepts it.

## Transitive web advisories

Fixed in the same branch, following the standing policy: overrides in
`web/package.json` for transitive packages, not ignores.

| Package | Before | After | How |
| --- | --- | --- | --- |
| `sharp` | 0.35.4 (`<0.35.5`, librsvg) | 0.35.5 | arrived with `next` 16.4.4 (#124); override retargeted from `0.35.3` to `0.35.5` so it cannot pin a vulnerable version |
| `brace-expansion` | 5.0.9 | 5.0.12 | override retargeted; the old target was itself vulnerable |
| `minimatch` | 10.2.6 | 11.0.5 | override `>=10.0.0 <11.0.1` → `>=11.0.1 <12`; GHSA-g3hv-75wg-m249 covers 7.0.0–11.0.0 with no backport |
| `postcss-selector-parser` | 6.0.10 | 6.1.5 | new override; `@tailwindcss/typography` pins 6.0.10 exactly |
| `source-map-js`, `vite`, `zod`, `undici` | various | in-range patches | lockfile refresh within existing ranges |

`minimatch` 11 keeps 10's export map and engine range. Its consumers are ESLint
and `typescript-estree`; lint reports the same 24 files and zero findings before
and after.

The lockfile drops `@rolldown/binding-wasm32-wasi` and its `@emnapi` and
`@napi-rs` dependencies. That is upstream: `rolldown` 1.2.16 no longer lists
the wasm32 binding. It is not the macOS lockfile omission recorded in 2026-09.

## Open: `filelock` (Python), blocked by LiteLLM's cap

`pip-audit` reports 13 CVEs against `filelock` 3.32.7, fixed only in 4.x
(4.0.9 for one, 4.2.1 for the rest). `filelock` is transitive, through `torch`,
`huggingface_hub` and `litellm`.

LiteLLM 1.108.1, the latest release, declares `filelock>=3.16.1,<4.0`. Every
release from 1.104.0 on carries that cap. Adding
`filelock>=4.2.1` to `requirements.txt` does resolve, and that is the problem:
the resolver moves LiteLLM back from 1.108.1 to 1.103.10, the newest release
without the cap. It is the openai 3 case from 2026-09 again. The result would
pair LiteLLM with a `filelock` major its own later releases forbid, and would
fix the default gateway at an older version.

After #154 and the LiteLLM 1.108.0 floor, the same `filelock>=4.2.1` floor no
longer resolves at all. openai 3 needs LiteLLM 1.108.0 or later, and every
LiteLLM from 1.104.0 on caps `filelock` below 4 (`uv pip compile` reports the
requirements as unsatisfiable). LiteLLM 1.109.0rc1, a pre-release published
2026-11-04, declares `filelock<5`. The first stable release carrying that
change clears the cap.

Two ways to resolve it, neither one a dependency bump:

1. A stable LiteLLM release that accepts `filelock` 4 (1.109.0rc1 already
   does). Then add `filelock>=4.2.1` to `requirements.txt` and confirm `pip
   freeze` moves nothing else backwards.
2. Make the `direct` backend the default and drop LiteLLM from
   `requirements.txt`. The 2026-09 triage already named this as a decision to
   make on its own merits.

## Verification commands

```bash
make install && python -m pip check && make test-fast
cd web && npm ci && npm run lint && npm run test && npm run build && npm audit
```

## Second batch — same day

Clearing the queue freed both ecosystems' slots, and Dependabot opened 15 more
pull requests (#132–#146) at 05:32–05:34 UTC. They were triaged the same
morning.

CI on all 15 ran after #131 merged, against the `main` they would land on, and
included the new `next build` step, so it was treated as current. 13 were green.

**Python (6), verified locally, then merged.** Fresh Python 3.12 environments
from `main` and from `main` with all six floors gave identical `pip freeze`,
clean `pip check`, and 1015 passed / 13 skipped in both. Merged #133
`boto3>=1.43.123`, #135 `fastapi>=0.162.0`,
#137 `opentelemetry-exporter-otlp-proto-grpc>=1.45.2`, #139 `psycopg>=3.3.7`
and #141 `langchain-anthropic>=1.7.9`. #142 `uvicorn[standard]>=0.54.0` edits
the line after #135's, so it conflicted and was sent back to Dependabot for a
rebase.

**Web (7), verified as a combined tree, then merged.** The seven green npm PRs
touch no shared lines. Merged together on top of `main`, they passed `npm ci`,
lint (24 files, 0 findings), 27 tests and `next build`, and `npm audit` found 0
vulnerabilities. Merged #132 `@types/node` 26.6.6, #134 `eslint` 10.14.0,
#136 `jsdom` 30.1.3, #138 `@testing-library/user-event` 14.6.8, #143
`tailwindcss` 4.3.4, #145 `react-syntax-highlighter` 16.1.2 and #146
`@vitejs/plugin-react` 6.2.1.

### #140 and #144 — `react` and `react-dom` opened separately: closed, fixed together

Each bumped one package to 19.3.1 and left the other at 19.3.0. Every web test
file then failed to load with React's own check: `react` and `react-dom` must
have the exact same version. Neither pull request could pass alone, and the two
conflict on adjacent `package.json` lines, so they cannot be merged as a pair
either.

Both packages are now bumped to 19.3.1 in one change, verified with `npm ci`,
lint, tests, `next build` and `npm audit` (0). A Dependabot `groups` entry in
`.github/dependabot.yml` puts the two in one pull request from now on. The group
covers only these two packages, because only they must match exactly. #117 and
#118 showed `vitest` and `@vitest/ui` passing as separate pull requests.

## Third batch — same day

Dependabot opened eight more pull requests (#148–#155) at 14:46 UTC. CI on all
eight ran against current `main`.

**Web (2), merged.** #152 `typescript-eslint` 8.72.1 and #155
`@tailwindcss/postcss` 4.3.4. Merged together on `main` they passed `npm ci`,
lint, tests and `next build`, and `npm audit` found 0 vulnerabilities.

**Python (5), verified locally, then merged.** #149 `pydantic>=2.14.2`,
#150 `pytest>=9.2.0`, #151 `transformers>=5.22.1`,
#153 `opentelemetry-sdk>=1.45.2` and #154 `openai>=3.42.1,<4`. Fresh Python 3.12
environments from `main` and from `main` with all five differed in exactly one
package, openai 2.54.0 → 3.46.0. LiteLLM stayed at 1.108.1, `pip check` was
clean, and both ran 1015 passed / 13 skipped.

### #154 — openai 3: the cap's reason is gone

`requirements.txt` capped openai at `<3` because LiteLLM capped it there (see the
2026-09 addendum). LiteLLM 1.108.0 is the first release that declares
`openai<4.0.0`; 1.102.1 through 1.106.7 all say `<3.0.0`. With an openai 3
floor, the resolver can no longer reach back to a LiteLLM that predates openai
3, because the LiteLLM floor already sits above those releases.

The checks went past the suite. `tests/test_transport_contract_repair.py` drives
the real `openai.OpenAI` client through a mock HTTP transport with Ragbot's own
request builder, and it passed on 3.46.0. A separate check against 3.46.0 covered
the paths the suite does not: module-level `openai.api_key`, the
`organization` argument, and streaming `chat.completions.create`.

Follow-up in this pass: the comment block above the openai line, which explained
a cap that no longer exists, now explains the pairing. The LiteLLM floor rises
from 1.102.1 to 1.108.0, so the requirement states the real constraint.

### #148 — mcp 2.x: closed, then ported

`mcp>=2.3.0,<3` failed to import in six test modules. mcp 2.x is a redesign
rather than a rename:

- the client's experimental tasks namespace is gone;
- `RequestContext` moved;
- the streamable HTTP client was renamed;
- `AnyUrl` left `mcp.types`;
- `ClientSession.call_tool` gained an input-required and request-state flow.

#148 was closed because a requirement bump could not carry that. The port
landed in its own pull request the same day; the "mcp 2.x port" section
below records it.

## mcp 2.x port

Ragbot moved from mcp 1.27.1 to 2.3.0, the current release. What changed in
Ragbot, by area:

- **Client primitives.** List calls take `params=PaginatedRequestParams(...)`
  rather than `cursor=`. Resource URIs are plain strings. Callbacks receive a
  `ClientRequestContext`. Types expose snake_case attributes (`next_cursor`,
  `is_error`, `structured_content`); camelCase still works when constructing
  them.
- **Tasks.** The SDK has no client task helpers, and its method tables leave
  `tasks/*` out of the core protocol. Ragbot's sessions negotiate 2025-11-25,
  where tasks are core methods, so `synthesis_engine.mcp.tasks` sends the typed
  `tools/call`-with-task, `tasks/get`, `tasks/result`, `tasks/list` and
  `tasks/cancel` requests itself, and polls at the server's suggested interval.
- **Streamable HTTP client.** `streamable_http_client` takes an
  `httpx2.AsyncClient`; Ragbot builds it with the server's headers, timeout and
  OAuth provider. The OAuth callback now returns an `AuthorizationCodeResult`
  and carries the RFC 9207 `iss` parameter.
- **Ragbot's MCP server.** The low-level server takes handlers as constructor
  arguments and no longer validates or packages tool calls. The handler now
  validates arguments against each tool's input schema, returns structured
  content with its JSON text, validates it against the output schema, and turns
  failures into `is_error` results — what 1.x did implicitly.
- **Proxy.** The relay passes pages through by cursor and returns tool results
  whole. Under 1.x it collected every page and returned only a tool result's
  content, dropping `isError` and structured content.

Defects found while porting, fixed in the same change:

- **Tool errors read as success in the agent loop.** `_coerce_tool_output`
  read `isError`, which 2.x types do not have. A failed MCP tool call would
  have reached the agent as successful output. It now reads `is_error`; a test
  pins it, and the same input on the pre-port code returns the error text as
  output.
- **HTTP bearer resolution.** The token travelled to the tool handler through
  a `ContextVar` set in the ASGI layer, which the SDK's session tasks do not
  reliably see. The handler now resolves the token from the HTTP request it is
  serving, and refuses an HTTP request it cannot authenticate.
- **HTTP lifecycle.** `asgi_app_factory` forwarded ASGI lifespan events to a
  session manager that was never started. It now wraps the SDK's
  `streamable_http_app`, which runs the manager for the app's lifetime, and
  serves at `/mcp`.
- **Reflected HTML in the OAuth callback page.** The loopback page wrote the
  redirect's `error` parameter into HTML unescaped; it is now escaped.
- **Frontend field names.** `web/src/lib/api.ts` declares `input_schema` and
  `mime_type`, but 1.x types serialized as `inputSchema` and `mimeType`, so the
  resource MIME badge never rendered. 2.x serializes snake_case; tests pin the
  shape the web client reads.

Verification: 1035 tests pass and 13 skip, up from 1015 on `main`. New tests
cover task polling and paging over real JSON-RPC, proxy pass-through, tool-call
validation and error results, unknown-resource errors, fail-closed bearer
resolution, and a Streamable HTTP round trip through Uvicorn that checks a 401
without a token and `allowed_tools` enforcement with one.

## Full upgrade to the latest releases

After the queue was clear, every external dependency moved to its latest
stable release, at the owner's direction: no backward compatibility, and no
dependency tied to an old version.

### Runtimes and images

| Component | Before | After | Note |
| --- | --- | --- | --- |
| Python (CI, backend image) | 3.12 | 3.14 | 3.15 (released 2026-10-09) lacks Linux wheels for PyYAML 6.0.3, fastuuid and orjson; 3.14 has wheels for all 175 packages on amd64 and arm64 |
| Backend image compiler | `gcc` installed | none | every dependency installs from a wheel |
| Node (CI, web image) | 22 | 26 | 26 became the active LTS on 2026-10-28 and bundles npm 11, so the image no longer installs npm separately |
| PostgreSQL | pgvector `pg16` | pgvector `pg18` (pgvector 0.8.7) | new volume at `/var/lib/postgresql`; `scripts/migrate-postgres-16-to-18.sh` copies an existing database |
| Jaeger | `all-in-one:1.76.0` (v1, end of life) | `jaegertracing/jaeger:2.23.0` | v2 receives OTLP on 4317/4318 by default |
| GitHub Actions | checkout, setup-python, setup-node v7 | unchanged | already the latest majors |

### Python packages

- Every direct requirement's floor is now the latest release, resolved on
  Python 3.14. Raising the floors moved nothing: `pip freeze` was identical
  before and after.
- `google-generativeai` was removed. It is end of life, nothing imported it,
  and no other dependency required it. Its pin on
  `google-ai-generativelanguage==0.6.15` had held protobuf at 5.29.6; protobuf
  is now 7.36.2, and google-api-core, proto-plus and grpcio-status are current.
- `appdirs`, unreleased since 2020, was replaced by its successor
  `platformdirs`, which has the same `user_data_dir` call and returns the same
  paths.

### Held back upstream

Seven packages still install below their latest release. In every case the cap
sits in the latest release of the package that depends on them, so nothing in
Ragbot can lift it:

| Package | Installed | Latest | Capped by |
| --- | --- | --- | --- |
| filelock | 3.32.7 | 4.2.9 | litellm 1.108.1 (`<4.0`); 13 CVEs, see above |
| huggingface-hub | 1.33.0 | 2.4.0 | litellm 1.108.1 (`<2.0`) |
| importlib-metadata | 8.9.0 | 10.1.0 | litellm 1.108.1 (`<9.0`) |
| mpmath | 1.3.0 | 1.4.1 | sympy 1.14.0 (`<1.4`) |
| python-dateutil | 2.9.0.post0 | 4.0.0 | botocore 1.43.125 (`<3.0.0`) |
| tenacity | 9.1.4 | 9.2.1 | google-genai 3.0.0 (`<9.2.0`) |
| uuid-utils | 0.19.0 | 1.2.0 | langsmith 0.14.17 (`<1.0`) |

LiteLLM holds three of the seven. Its 1.109.0rc1 pre-release already lifts the
filelock cap.

### Web packages

Every package is on its latest release, with manifest floors raised to match,
including jsdom 31. The one exception is TypeScript, which stays on 6.0.3:
typescript-eslint 8.72.1, and its canary build, still declare
`typescript <6.1.0`. `npm audit` reports 0 vulnerabilities.

### Code and test fixes found on the new runtimes

- `synthesis_engine/agent/sandbox.py` used `asyncio.get_event_loop()` inside a
  coroutine; it now uses `asyncio.get_running_loop()`.
- Two class-scoped fixtures in `tests/test_engines_yaml.py` were instance
  methods, which pytest 9 deprecates and pytest 10 removes; they are now
  classmethods.
