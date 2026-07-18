# Changelog

All notable changes to Ragbot are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) loosely, and
versioning follows [Semantic Versioning](https://semver.org/).

For the prose narratives accompanying major releases, see
[`docs/release-notes-v3.4.0.md`](docs/release-notes-v3.4.0.md) and
the equivalents for prior versions when added.

## v3.7.0 — 2026-07-17

### Added

- **Content-tier retrieval metadata and an archive tier.** `chunks` gains a `content_tier`
  column (`frequent-access` / `topic-specific` / `archive-only`), defaulting existing rows to
  `topic-specific` (no backfill needed — that's exactly what `source/datasets/` already was),
  plus `content_hash`/`embedding_model_version` for future drift detection. A new indexing path
  (`ragbot index --workspace X --tier archive`, gated behind `RAGBOT_OWNER_CONTEXT=1` like the
  existing private-repo inclusion) covers `personal-archive/`-style content that was previously
  invisible to Ragbot entirely. Archive-tier content is excluded from the default search path —
  reachable only through a new, separate `search_archive()` function; `get_relevant_context()`'s
  signature and default behavior are unchanged.
- **`ragbot mcp serve`** — a CLI entry point (stdio transport) for the previously-dormant
  `RagbotMCPServer`, which exposes `workspace_search`/`workspace_search_multi` over pgvector to
  MCP-aware coding agents (Claude Code, Cursor, etc.). See `docs/mcp-server-activation.md` for
  what's still manual (bearer-token generation, client-side registration).
- **`docs/rls-and-roles.md`** — explains the two-role database pattern below and how to verify
  it live.

### Security

- **Row-Level Security on `documents`/`chunks`, with a restricted application role.** Workspace
  isolation was previously enforced only by application code always including a `WHERE
  workspace = %s` clause — a real gap if any future query path forgot it. RLS policies now
  enforce this at the database level too. Getting this to actually work required a second fix
  found only by testing against a live database: Postgres unconditionally disables RLS for
  superuser roles (`FORCE ROW LEVEL SECURITY` does not override this), and the `ragbot` role
  created by the standard Postgres Docker image is a superuser by default — so the policies
  alone would have been silently inert. A new restricted `ragbot_app` role (no superuser, no
  bypassrls, exactly the grants runtime queries need) now handles all app-level queries;
  migrations still run under the superuser role via a separate `RAGBOT_MIGRATION_DATABASE_URL`.
  `docker-compose.yml`/`.env.example` default to this split for local dev.

## v3.6.0 — 2026-07-14

### Changed

- **One tier vocabulary everywhere: `category: small|medium|large` →
  `tier: judgment|routine|bulk`.** engines.yaml, the loader
  (`get_all_models`, `get_model_by_tier` — formerly
  `get_model_by_category`), the `/api/models` responses, the web model
  picker (badges and search now keyed on `tier`), the compiler
  (`resolve_model` tiers, `compile-config.yaml` targets take `model_tier`;
  `flagship` remains the is_flagship-based selector, distinct from the
  judgment tier), and every test. The words are identical to the
  synthesis-model-tiers skill's tiers.yaml — role labels and catalog
  labels are one vocabulary, not two. The labels name the WORK a model
  is trusted with (judgment calls where being wrong is expensive;
  routine rule-following execution; high-volume bulk), not the artifact
  — deliberately outside vendor marketing vocabulary, which claims words
  like "frontier" for entire model generations (see the skill's
  `references/naming-rationale.md` for the criteria and the rejected
  candidates). Content categories (datasets/runbooks/instructions) are
  an unrelated concept and are unchanged. Breaking for external
  `compile-config.yaml` files: rename the `model_category` key to
  `model_tier` (values `small`/`medium`/`large` become
  `bulk`/`routine`/`judgment`; `flagship` is unchanged).

### Added

- **Tier-consistency enforcement** (`tests/test_engines_yaml.py`):
  `TestTierVocabulary` requires every model to declare a valid tier and
  rejects any reappearance of the legacy `category` field;
  `TestTierRoleConsistency` cross-checks the installed
  synthesis-model-tiers `tiers.yaml` — every provider configured here
  must have a role block there, and every model named in a role list
  must exist here under the same tier. Drift between the role table and
  this catalog is now a test failure instead of an observation. The
  cross-check skips cleanly on machines (and public CI) without the
  skill installed; `SYNTHESIS_TIERS_FILE` overrides the search path.

## v3.5.2 — 2026-07-14

### Security

- **Web dev/test toolchain upgraded, clearing all 14 open Dependabot alerts**
  (1 critical, 4 high, 5 moderate, 4 low — every one in
  `web/package-lock.json`; the Python requirements had none).
  vitest/@vitest/ui `^3.2.4` → `^3.2.6` (resolved 3.2.7; the critical
  GHSA-5xrq-8626-4rwp — arbitrary file read/execute while the Vitest UI
  server listens), vite 8.0.13 → 8.1.4 and the vitest-nested vite line
  7.3.3 → 7.3.6 (`server.fs.deny` bypass, launch-editor NTLMv2 hash
  disclosure), undici 7.25.0 → 7.28.0 (six advisories, including SOCKS5
  cross-origin request routing and a TLS-certificate-validation bypass),
  esbuild 0.27.7 → 0.28.1, js-yaml 4.1.1 → 4.3.0, @babel/core 7.29.0 →
  7.29.7. `npm audit` now reports 0 vulnerabilities. Runtime dependencies
  were unaffected — every flagged package is build/test tooling.

### Fixed

- **`npm run lint` crashed with `TypeError: expand is not a function`.**
  The security overrides in `web/package.json` set floors with no upper
  bound, so the brace-expansion override (`>=1.1.13`) resolved to the 5.x
  line, whose export shape minimatch v3 (CJS) cannot call. Every override
  is now bounded to the major line it patches (e.g. `>=1.1.13 <2`);
  minimatch v3 gets a nested brace-expansion 1.1.16 while minimatch v10
  keeps the hoisted 5.x it actually wants.
- **14 `react-hooks/set-state-in-effect` errors** (a rule enabled by
  eslint-config-next 16 via eslint-plugin-react-hooks 7) that accumulated
  invisibly while lint was crashing. All fixed with restructures rather
  than suppressions: render-phase adjustments for prop-driven state
  (ModelPicker ⌘K open signal and focus-index clamp, PolicyPanel workspace
  seeding, SettingsPanel per-workspace resets, SkillsPanel expanded-row
  reset), mount fetches inlined with cancellation guards (McpServersPanel,
  PolicyPanel policies/check/audit feeds), loading indicators derived from
  fetched-state keys instead of flags flipped synchronously inside effects
  (SkillsPanel list, PolicyPanel cross-workspace check), dropdown-open
  resets moved into the open event (ModelPicker), and
  ShortcutsHelpOverlay's refresh-on-open effect deleted outright (the
  registry subscription it duplicated already covers every change).
  Verified by the web suite (24/24), a clean `next build`, and driving the
  panels live in the browser.

## v3.5.1 — 2026-07-14

### Changed

- **engines.yaml refreshed to the July 2026 model lines** (verified against the
  providers' live docs and APIs the same day). OpenAI: GPT-5.6 Sol / Terra /
  Luna replace the GPT-5.4/5.5 line (default: `gpt-5.6-terra`). Anthropic:
  Claude Fable 5 (new flagship), Claude Opus 4.8, and Claude Sonnet 5 replace
  Opus 4.7 / Sonnet 4.6 (default: `claude-sonnet-5`); Claude Haiku 4.5 stays
  and now correctly advertises its extended-thinking support.

### Fixed

- **Claude 4.8+/5.x `temperature` rejection.** The Anthropic API now returns
  400 "`temperature` is deprecated for this model" for the new generation.
  Both LLM backends and the thinking resolver stop sending `temperature` for
  Claude ≥ 4.8 (including the thinking paths); Claude 4.7 keeps the
  adaptive-shape + temperature=1 behavior, and pre-4.7 models keep the
  reasoning_effort + temperature=1 path.
- **Version-aware Claude generation detection.** The "Claude 4.7 or newer"
  checks in `ragbot.core` and both `synthesis_engine.llm` backends were
  hardcoded substring lists (`opus-4-7`, `sonnet-4-7`, ...) that silently
  stopped matching the moment engines.yaml moved past the 4.7 line — Fable 5
  / Opus 4.8 / Sonnet 5 would have been routed through LiteLLM's
  `reasoning_effort` mapper, which the API rejects for 4.7+. Replaced with a
  shared, version-parsing predicate in `synthesis_engine.llm.base`.
- **Test output no longer dumps request kwargs on HTTP failures**
  (`pytest.ini` with `--tb=short`): a full traceback through LiteLLM's HTTP
  handler printed live credential kwargs into test logs.

## v3.5.0 — 2026-05-15

Substrate cleanup. Pgvector is the only vector backend, the agent loop
wires at startup, OTLP metric and trace export are independently
configurable, app loggers surface under uvicorn, and the regression
suite no longer fails on dev machines without heavy ML dependencies.
Breaking change: `RAGBOT_VECTOR_BACKEND=qdrant` and the bundled Qdrant
backend are removed. Operators who ran v3.4 with the Qdrant opt-in
must reindex their workspaces into pgvector before upgrading.

### Removed

- **Qdrant vector backend.** Deleted `synthesis_engine.vectorstore.QdrantBackend`,
  the embedded `qdrant_data/` storage path, the `qdrant-client` dependency,
  the `RAGBOT_VECTOR_BACKEND` environment variable, and the `ragbot-qdrant`
  Docker volume. Dead `_qdrant_client` / `_get_qdrant_client` /
  `get_qdrant_point_id` helpers in `rag.py` and `chunking/` removed.
  The `VectorStore` ABC at `synthesis_engine.vectorstore` is retained so
  substrate consumers outside Ragbot can plug in alternative backends
  behind the same contract.

### Changed

- **Agent loop wires at startup.** The FastAPI lifespan now constructs an
  `AgentLoop` with the lifespan's LLM backend, the resolved MCP client, and
  a `FilesystemCheckpointStore`, then calls
  `api.routers.agent.set_default_loop()` to register the singleton. The
  `/api/agent/run` endpoint resolves against a real loop on a fresh install
  — through v3.4 it returned `"Agent loop is not configured"`. Shutdown
  clears the singleton.
- **OTLP metric export is independently configurable.** The substrate now
  honours the OTEL standard per-signal env-var hierarchy:
  `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` (per-signal override; accepts the
  literal `"none"` to disable metric OTLP export) falls back to
  `OTEL_EXPORTER_OTLP_ENDPOINT`. The bundled docker-compose stack sets the
  metrics endpoint to `"none"` because Jaeger only accepts traces — the
  `UNIMPLEMENTED` errors from earlier deployments are gone. Prometheus
  exposition at `/api/metrics` is unaffected.
- **App loggers surface under uvicorn.** `src/api/main.py` calls
  `logging.basicConfig` at module-import time before uvicorn takes over,
  so `api.main`, `api.routers.*`, and `synthesis_engine.*` log lines now
  flow to `docker logs ragbot-api` alongside uvicorn's own access logs.
  Override the level with `RAGBOT_LOG_LEVEL`.
- **`get_vector_store()` returns `None` when pgvector is unreachable**
  instead of falling back through a backend chain. Callers in `rag.py`
  treat `None` as "RAG unavailable; chat-only mode," so the user-facing
  failure mode is graceful.

### Fixed

- **`/api/agent/run` returns 503 "not configured"** — fixed by the agent
  loop wiring change above.
- **OTLP metric export prints `UNIMPLEMENTED`** — fixed by the per-signal
  endpoint split.
- **App-namespace logger lines invisible in container logs** — fixed by
  `logging.basicConfig` in `src/api/main.py`.
- **`test_sentence_transformers_imports_cleanly` fails on dev machines
  without sentence_transformers installed** — wrapped both Bug5
  regression tests in `pytest.importorskip("sentence_transformers")` so
  they skip cleanly on lightweight dev installs and still run in Docker /
  CI where the dependency is present.

### Test-suite delta

- v3.4.0 baseline: 871 passing, 25 skipped, 4 failing (3 Qdrant tests + 1
  sentence_transformers env gap).
- v3.5.0: 850 passing, 14 skipped, 0 failing (Qdrant tests gone with the
  backend; sentence_transformers tests skip cleanly).

## v3.4.0 — 2026-05-14

Ragbot becomes the conversational reference runtime of synthesis engineering.

v3.4 is the next-major-features release for Ragbot. It moves the project
from a polished 2024-paradigm chat-with-RAG product to a 2026-shaped
conversational AI runtime: explicit agent loop, first-class MCP in both
directions, an executable skills runtime, cross-workspace synthesis with
visible confidentiality boundaries, durable memory beyond vector RAG,
and the production-grade signals (observability, replay, eval harness,
background tasks, scheduled routines) that make the architecture legible
to engineering leadership. The synthesis-engineering positioning that the
architecture had quietly named for a year is now visible in the README,
the ragbot.ai homepage, and the in-product chrome.

### Headliners

**Agent loop runtime.** Ragbot is now an execution surface, not just a
chat surface. A hand-rolled graph-state agent loop replaces the
single-turn `prompt → retrieve → call LLM → return` path. The agent can
decide between answering directly, dispatching retrieval, calling a
tool, running a skill, or fanning out to sub-agents. Plan-and-Execute is
the default compound-question pattern with explicit replanning on
failure. Permission gates fail closed at the tool boundary. State
checkpoints are durable and replayable. The chat-only no-tools mode
remains available for users who want it.

**First-class MCP — client and server.** As a client, Ragbot covers all
six MCP primitives (tools, resources, prompts, Roots, Sampling,
Elicitation) and supports MCP Tasks for long-running calls. As a server,
Ragbot exposes workspace search, document retrieval, skill execution,
and audit recent as MCP tools and resources so Claude Code, Cursor,
ChatGPT desktop, and other MCP-aware clients can call into Ragbot's
knowledge surface. OAuth 2.1 with Dynamic Client Registration is
supported for remote servers; bearer-token auth and per-token
`allowed_tools` glob filtering are supported on the server side.

**Skills as runtime.** `SKILL.md` is now Ragbot's native extensibility
format with progressive disclosure (names + descriptions in the system
prompt, full body on selection, scripts and templates on tool call).
Skills written for Claude Code, Codex CLI, Cursor, or Gemini CLI run on
Ragbot without modification. Six starter skills ship in the box:
`workspace-search-with-citations`, `draft-and-revise`,
`fact-check-claims`, `summarize-document`, `agent-self-review`, and
`cross-workspace-synthesis`. The `npx skills add
synthesisengineering/synthesis-skills` install path turns the 32 public
synthesis-skills into runnable capabilities, not just searchable
content.

**Synthesis ecosystem positioning and rebrand.** Ragbot is officially
the reference runtime for the conversational interaction primitive
inside synthesis engineering, with sibling reference implementations
covering direct manipulation (synthesis-console), procedural execution
(Ragenie), and the portable capability format (synthesis-skills). The
README hero, the ragbot.ai homepage, the in-product footer chrome, and
`llms.txt` all lead with the synthesis framing. Vermillion (#b8312f) is
Ragbot's accent ink in the historical-inks palette of the synthesis
family, joining Prussian blue (synthesis-engineering), iron-gall green
(synthesis-coding), walnut brown (synthesis-writing), and lapis
ultramarine (Ragenie).

**Cross-workspace synthesis.** Multi-workspace chat is now first-class.
Select 2+ workspaces in the UI; the agent sees a per-workspace context
budget and per-workspace confidentiality tag. Per-workspace
`routing.yaml` governs which models can be called for each workspace
(local-only for `client-confidential`, frontier for `personal`, and so
on). Every cross-workspace operation is logged to an append-only audit
trail with timestamp, workspaces involved, tools called, and model
used. The `cross-workspace-synthesize` starter skill walks the agent
through per-workspace budget math, the four-level confidentiality
strictness order with pairwise mix table, and the
`[workspace:document_id]` citation format.

### Production-grade signals

- **Memory beyond RAG.** A three-layer memory stack: vector RAG over
  pgvector (existing), entity-graph memory with provenance and temporal
  validity (new `nodes` and `edges` tables), and session/working memory
  (per-user persistent prefs and in-flight context). A consolidation
  pass between sessions distills durable facts from the previous
  session and writes them into the entity graph. Pluggable: Mem0 and
  Letta integrations available as optional swap-ins behind the
  abstraction.
- **MCP server.** `synthesis_engine.mcp_server` exposes Ragbot via both
  stdio (for desktop integrations) and HTTP/SSE (via
  `StreamableHTTPSessionManager`) transports. Per-token `allowed_tools`
  glob filtering and bearer-token auth configured via
  `~/.synthesis/mcp-server.yaml`. Five exposed tools: `workspace_search`,
  `workspace_search_multi` (confidentiality gate fires before retrieval),
  `document_get`, `skill_run`, `agent_run_start`. Three exposed
  resources: workspaces, skills, audit-recent.
- **Replay CLI.** `ragbot agent replay <task_id>` deterministically
  re-runs a session from any checkpoint, with `--show-trace` and
  `--save-output` flags. `ragbot agent list-sessions` and `ragbot agent
  checkpoints <task_id>` for inspection. A stable hash over
  `{current_state, final_answer, step_results}` (timestamps excluded)
  gates regression detection.
- **Eval regressions.** `tests/evals/regressions/` captures canonical
  bug shapes (sub-agent dispatch max-parallel, disabled sandbox
  actionable error, permission deny blocks tool, cross-workspace
  air-gapped isolation, replay determinism canary). `make eval` runs the
  full eval suite with a scorecard renderer.
- **Background tasks.** `synthesis_engine.tasks` provides a
  `BackgroundTaskManager` with JSONL persistence at
  `~/.synthesis/tasks/{id}.jsonl`, cooperative cancellation
  (`TaskCancelled` raised at safe points; no force-kill), crash recovery
  on startup, webhook delivery per-task, and three notifier adapters
  (macOS via `osascript`, email via SMTP, Slack via MCP). A scheduler is
  opt-in via `RAGBOT_SCHEDULER=1`, reading `~/.synthesis/schedules.yaml`.
- **Keyboard shortcuts.** A coherent shortcut layer covers the 2026
  expected interactions: ⌘K (model picker), ⌘J (workspace switch), ⌘/
  (message history search), ⌘N (new chat), ⌘B (background current run),
  ⌘. (cancel), ⌘? (help overlay with focus trap and Escape close).
  Platform-aware key matching (Meta on macOS, Ctrl elsewhere); strict
  exact-modifier matching so ⌘⌥K doesn't accidentally fire ⌘K.
- **Observability.** OpenTelemetry traces by default with semantic
  GenAI attributes on every model call, retrieval step, guardrail
  check, and tool dispatch. `OTEL_EXPORTER_OTLP_ENDPOINT` ships traces
  to Phoenix, Langfuse, Datadog, or Honeycomb. Prometheus exposition at
  `/api/metrics` and cache-stats JSON at `/api/metrics/cache`. Prompt
  caching with `cache_control` annotations on the static system-prompt
  prefix.

### Open-weights additions

`engines.yaml` adds the four open-weights families that became serious
local agent defaults in 2026:

- **Llama 4** (Meta) — sizes documented in the new sizing matrix at
  [`docs/open-weights-sizing.md`](docs/open-weights-sizing.md).
- **Qwen3** (Alibaba) — the practical local agent default; 27B size is
  the recommended balance of capability and footprint on Apple Silicon
  with the MLX backend.
- **DeepSeek-V3** — strong reasoning at competitive sizes.
- **Mistral Large** — Mistral's open-weights flagship.

Updated **Gemma 4** entries with notes on the Ollama 0.19 MLX backend
(~2x decode speedup on Apple Silicon). The full sizing matrix maps
model families to recommended hardware tiers (laptop, prosumer desktop,
Mac Studio-class, workstation), VRAM/unified-memory requirements, and
target inference profiles.

### Breaking changes

v3.4 is the next-major-features release. Breaking changes are
intentional and visible. If you are upgrading from v3.3, the items
below require migration steps.

- **`synthesis_engine` is now a public substrate library.** The runtime
  code under `src/synthesis_engine/` is the supported import surface for
  building synthesis-engineering products on top of Ragbot's primitives.
  `src/ragbot/` is now ragbot-runtime-specific code only. Imports under
  `from ragbot.X` for substrate types have moved to `from
  synthesis_engine.X`. The `RagbotError` exception base class has been
  renamed to `SynthesisError` across the substrate; all five subclasses
  are renamed correspondingly.
- **`routing.yaml` is a new per-workspace convention.** Each workspace
  may declare a `routing.yaml` at its root with
  `allowed_models`/`denied_models` globs, a `confidentiality` tag
  (`public` / `personal` / `client-confidential` / `air-gapped`), and a
  `fallback_behavior` (`DENY` / `DOWNGRADE_TO_LOCAL` / `WARN`). The
  cross-workspace agent runtime enforces the strictest applicable
  policy. Workspaces with no `routing.yaml` default to `personal` with
  no model restrictions, preserving prior behavior.
- **`identity.yaml` is a new convention at `~/.synthesis/`.** Declares
  `personal_workspaces` (the list of workspaces treated as universal
  for skill scoping) and `personal_remote_patterns` (used by the
  synthesis-git-hooks policy to classify a repo as personal vs strict).
  Required for the new workspace-scoped skills discovery to know which
  workspaces are personal vs scoped.
- **Agent loop API signature.** `AgentLoop.run()` accepts new kwargs:
  `workspaces` (list, not single value), `workspace_roots`,
  `routing_enforced`, and `cross_workspace_budget_tokens`. The legacy
  single-workspace shape continues to work — a single-element list
  preserves prior behavior — but consumers calling the agent loop
  directly should update to the multi-workspace shape.

### Migration notes

If you are upgrading from v3.3:

1. **Update imports.** Replace `from ragbot.X` substrate imports with
   `from synthesis_engine.X`. The same applies to `RagbotError` →
   `SynthesisError`.
2. **Create `~/.synthesis/identity.yaml`.** A minimal example:
   ```yaml
   personal_workspaces:
     - acme-user
   personal_remote_patterns:
     - "github.com:acme-user/"
   ```
   The `synthesis-git-hooks` skill in `synthesis-skills` includes a
   commented example config. Install via `npx skills add
   synthesisengineering/synthesis-skills --skill synthesis-git-hooks`.
3. **Add `routing.yaml` to confidential workspaces.** A minimal example
   for a `client-confidential` workspace:
   ```yaml
   confidentiality: client-confidential
   allowed_models:
     - "ollama/*"
     - "ollama_chat/*"
   denied_models:
     - "claude-*"
     - "gpt-*"
     - "gemini-*"
   fallback_behavior: DENY
   ```
4. **Install the synthesis-git-hooks engine.** Replace any repo-local
   `.githooks/pre-commit` with the universal engine at
   `~/.synthesis/git-hooks/` (installed via the synthesis-git-hooks
   skill above). Configure `~/.synthesis/git-hook-config.yaml` with your
   client names, internal URLs, and personal-remote patterns.
5. **Install the anti-shortcut catalog.** `~/.synthesis/anti-shortcut-catalog.yaml`
   is consumed by the synthesis-anti-shortcuts skill and by the
   pre-commit hooks to detect lazy-shortcut costume vocabulary in code
   and prose. Install via `npx skills add
   synthesisengineering/synthesis-skills --skill synthesis-anti-shortcuts`.

### Acknowledgments

Thanks to everyone who reviewed proposals, surfaced issues, and pushed
back on lazy shortcuts during the v3.4 development cycle. The lessons
distilled during development inform the
[synthesis-anti-shortcuts](https://github.com/synthesisengineering/synthesis-skills/tree/main/synthesis-anti-shortcuts)
skill, which any SKILL.md-compatible AI coding agent can install.

---

## v3.3.0 — 2026-05

Local Gemma 4 via Ollama as first-class. Redesigned single-rich-dropdown
model picker with Pinned/Recent, type-ahead search, `⌘K` shortcut, and
capability badges. User preferences API persisting to
`~/.synthesis/ragbot.yaml`. Bug fix for non-flagship GPT-5.x and Gemini
returning empty content on long-context RAG calls. LiteLLM pinned
`>=1.83.0` to exclude the March-2026 supply-chain incident range.

## v3.2.0 — 2026-04

Demo mode (`RAGBOT_DEMO=1`) with hard-isolated discovery and a bundled
sample workspace and skill. `/health` and `/api/config` report
`demo_mode`. Twenty new tests locking in the discovery isolation
contract.

## v3.1.0 — 2026-04

LLM backend abstraction (`RAGBOT_LLM_BACKEND={litellm|direct}`). Web UI
controls for reasoning effort and the cross-workspace skills toggle.
`/api/chat` accepts `thinking_effort` and `additional_workspaces`.

## v3.0.0 — 2026-04

Pgvector by default with native FTS via tsvector + GIN. Agent Skills as
first-class content with `ragbot skills {list,info,index}` CLI.
Workspace-rooted layout discovered across `~/workspaces/*/ai-knowledge-*`
and via `~/.synthesis/console.yaml`. Reasoning-effort wiring for Claude
4.x adaptive thinking, GPT-5.5 reasoning, and Gemini 3.x thinking
levels.

## Earlier versions

Pre-v3.0 history is recorded in commit messages and the README "What's
New" sections. Older releases predate this CHANGELOG file.
