# Ragbot — the open-source reference runtime for conversational synthesis engineering.

Ragbot is the chat-led runtime of [synthesis engineering](https://synthesisengineering.org) — the open methodology for systematic human-AI collaboration on complex work. It is the workbench where humans and AI do the crafts of synthesis one collaborative turn at a time, on the user's own workspace, with skills, agent execution, durable memory, and bi-directional MCP. Workspace-aware. Local-first with frontier fallback. Production-grade observability. MIT-licensed.

Developed by [Rajiv Pant](https://github.com/rajivpant). See [INSTALL.md](INSTALL.md) for setup, [CONFIGURE.md](CONFIGURE.md) for keys and providers, and [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR.

## What's New in v3.5 (May 2026)

Ragbot v3.5.0 (May 15, 2026) is a substrate-cleanup release that ships the day after v3.4. Pgvector is now the only vector backend (the Qdrant code path is removed entirely, no shim). The agent loop wires at startup so `/api/agent/run` resolves against a real loop on a fresh install. OTLP trace and metric endpoints are independently configurable; the bundled docker-compose stack no longer prints `UNIMPLEMENTED` errors. App-namespace log lines now surface in `docker logs ragbot-api`. The regression suite no longer fails on dev machines without `sentence_transformers` installed. See [CHANGELOG.md](CHANGELOG.md) for the full delta — including the breaking change for operators who ran v3.4 with `RAGBOT_VECTOR_BACKEND=qdrant` (reindex into pgvector before upgrading).

## What's New in v3.4 (May 2026)

Ragbot v3.4.0 (May 14, 2026) is the next-major-features release. It moves the project from a polished 2024-paradigm chat-with-RAG product to a 2026-shaped conversational AI runtime: explicit agent loop, first-class MCP in both directions, an executable skills runtime, cross-workspace synthesis with visible confidentiality boundaries, durable memory beyond vector RAG, and the production-grade signals that make the architecture legible to engineering leadership.

[Read the v3.4.0 release notes →](docs/release-notes-v3.4.0.md) · [GitHub release →](https://github.com/synthesisengineering/ragbot/releases/tag/v3.4.0)

- **Agent loop runtime.** Hand-rolled FSM — no LangGraph, no CrewAI, no AutoGen. Multi-step planning, sub-agent dispatch, sandboxed code execution (E2B / Daytona / DisabledSandbox), durable checkpoints, deterministic replay via `ragbot agent replay <task_id>`. 45 tests across the agent loop core and capabilities surface.
- **First-class MCP — client and server.** All six primitives (tools, resources, prompts, Roots, Sampling, Elicitation) plus Tasks. OAuth 2.1 + Dynamic Client Registration for remote servers. As a server, Ragbot exposes its workspace surface to Claude Code, Cursor, ChatGPT desktop, Gemini CLI, and any MCP-aware peer over stdio or HTTP/SSE.
- **Skills as runtime.** Progressive-disclosure execution of SKILL.md capabilities — Ragbot becomes the third compatible runtime for the SKILL.md format (after Claude Code and Codex CLI). Six starter skills ship in the box; `npx skills add synthesisengineering/synthesis-skills` pulls in 32 more.
- **Cross-workspace synthesis.** Multi-workspace chat with per-workspace `routing.yaml`, four-level confidentiality (`public` / `personal` / `client-confidential` / `air-gapped`), and an append-only audit log at `~/.synthesis/cross-workspace-audit.jsonl`. The confidentiality gate fires before retrieval, so denied workspace combinations never read content. Citations name the source workspace explicitly.
- **Three-tier memory beyond RAG.** Vector RAG + entity graph + session/working memory. A consolidation pass between sessions distills durable facts from the previous session into the entity graph — Anthropic's "Dreaming" pattern. Mem0 and Letta integrations are available behind the abstraction.
- **Production-grade observability.** OpenTelemetry GenAI semantic conventions on every model call, retrieval step, guardrail check, and tool dispatch. OTLP gRPC export to a bundled Jaeger in the docker-compose stack; `OTEL_EXPORTER_OTLP_ENDPOINT` redirects to Phoenix, Langfuse, Datadog, or Honeycomb. Prometheus exposition at `/api/metrics`; cache stats at `/api/metrics/cache`.
- **Keyboard shortcut layer.** Coherent set across the web UI: `⌘K` model picker, `⌘J` workspace switch, `⌘/` message-history search, `⌘N` new chat, `⌘B` background the current run, `⌘.` cancel the current run, `⌘?` help overlay.
- **Open-weights additions.** Llama 4, Qwen3, DeepSeek-V3, Mistral Large, and updated Gemma 4 entries in `engines.yaml` with Ollama 0.19 MLX backend notes. Full hardware-sizing matrix in [`docs/open-weights-sizing.md`](docs/open-weights-sizing.md).

![Ragbot v3.4 — Jaeger trace tree for a chat request showing retrieval and chat-completion children](docs/screenshots/observability-trace.png)

![Ragbot v3.4 — cross-workspace policy panel with per-workspace confidentiality and model-routing verdicts](docs/screenshots/policy-panel.png)

## What's New in v3.3

Ragbot v3.3 (May 2026) adds first-class local model support and a
redesigned model picker:

- **Local Gemma 4 via Ollama.** A new `ollama` engine ships Google's Gemma 4
  family (E4B, 26B MoE, 31B Dense) as first-class models alongside Anthropic,
  OpenAI, and Google. No API key required; LiteLLM routes via the
  `ollama_chat/` prefix. The Docker stack reaches host Ollama via
  `host.docker.internal:11434` out of the box (configurable with
  `OLLAMA_API_BASE` if Ollama runs elsewhere on your LAN).
- **Redesigned model picker.** A single rich dropdown replaces the
  three-step Provider → Category → Model cascade. Display names
  (`Claude Opus 4.7` instead of `claude-opus-4-7`). Pinned and Recent
  sections at the top. Type-ahead search. `⌘K` / `Ctrl+K` global shortcut.
  Per-row badges for tier (Fast / Balanced / Powerful), context window,
  🧠 thinking, 🏠 local.
- **User preferences API.** New `/api/preferences/pinned-models` and
  `/api/preferences/recent-models` endpoints persist your model selections
  across sessions in `~/.synthesis/ragbot.yaml`.
- **Thinking control moved adjacent to Model.** Renders inline below the
  picker, only for thinking-capable models — so the control isn't there
  when it has no effect.
- **Bug fix:** non-flagship GPT-5.x and Gemini models no longer return
  empty content on long-context RAG calls. The default `reasoning_effort`
  for these models is now the lowest declared mode (`minimal`) rather than
  unset, so the provider's own reasoning default doesn't consume the entire
  output-token budget.
- **Security.** LiteLLM pinned `>=1.83.0` in requirements (excludes the
  compromised 1.82.7 / 1.82.8 range from the March 2026 supply-chain
  incident).

![Ragbot v3.3 — settings panel with the new model picker](screenshots/v3.3/01-hero-model-picker.png)

The model picker, opened — one dropdown grouped by provider with Pinned,
Recent, type-ahead search, and capability badges:

![Ragbot v3.3 — model picker open with Pinned, Recent, and by-provider sections](screenshots/v3.3/02-model-picker-open.png)

Type-ahead filters across the full list. Searching `gemma` surfaces all
three local Gemma 4 variants:

![Ragbot v3.3 — model picker with type-ahead search filter](screenshots/v3.3/03-model-picker-filter.png)

Selecting a local model surfaces the 🏠 badge on the trigger and hides the
Thinking control (Gemma 4 doesn't expose thinking effort):

![Ragbot v3.3 — local Gemma 4 31B selected](screenshots/v3.3/04-local-model-selected.png)

Advanced panel — per-provider API-key status, including the new
local-only Ollama entry:

![Ragbot v3.3 — advanced settings with all four providers](screenshots/v3.3/05-advanced-settings.png)

## What's New in v3.2

Ragbot v3.2 (April 2026) ships a one-command demo mode and a refreshed
screenshot set captured against it:

- **Demo mode** via `RAGBOT_DEMO=1` (or `ragbot --demo`) ships a small
  bundled workspace and skill in `demo/`, hard-isolates discovery from
  the user's real workspaces, and surfaces an unmistakable banner in
  the Web UI. Anyone can clone the repo, set an API key, and have a
  working chat with RAG retrieval in under a minute — no database or
  workspace setup needed.
- **`/health` and `/api/config`** report `demo_mode` so any consumer
  (UI, ops dashboards, screenshot tools) can render the right
  affordances.
- **Twenty new tests** lock in the discovery isolation contract so a
  future change can't accidentally let real workspace or skill names
  leak through when demo mode is on.

![Ragbot v3.2 demo — settings panel and welcome state](screenshots/v3.2/01-hero-settings-panel.png)

A live conversation in demo mode showing RAG retrieval citing the
bundled sample documents:

![Ragbot v3.2 chat with RAG response](screenshots/v3.2/02-chat-with-response.png)

## What's New in v3.1

Ragbot v3.1 (April 2026) adds an LLM backend abstraction that decouples ragbot from any single provider gateway:

- **Swappable LLM backends** via `RAGBOT_LLM_BACKEND={litellm|direct}`. The default `litellm` backend keeps the broadest provider/model coverage; the `direct` backend calls Anthropic, OpenAI, and google-genai SDKs directly with no third-party dependency. Adding alternatives (Bifrost, Portkey, OpenRouter) is a one-file change.
- **Web UI controls** for reasoning effort and the cross-workspace skills toggle, alongside the existing workspace/model picker.
- **`/api/chat`** accepts `thinking_effort` and `additional_workspaces` fields directly.

Strategic note on LiteLLM in 2026: it remains a defensible default because of provider/model coverage, but the March-2026 supply-chain incident (versions 1.82.7–1.82.8) and the API-compatibility lag for Claude 4.7+'s `thinking.type.adaptive` shape are real frictions. Pinning `>=1.83.0` avoids the compromised range; the abstraction layer makes a future swap a configuration change, not a code rewrite.

## What's New in v3.0

Ragbot v3.0 (April 2026) ships three major upgrades over v2:

- **Pgvector by default.** PostgreSQL with the `pgvector` extension is the vector backend. Native full-text search via `tsvector` + GIN replaces in-process BM25. (As of v3.5, Qdrant support was removed entirely — pgvector is the only shipped backend; the `VectorStore` ABC still allows substrate consumers to plug in alternatives.)
- **Agent Skills as first-class content.** Ragbot discovers and indexes Agent Skills (`SKILL.md` plus references and scripts) from `~/.synthesis/skills`, `~/.claude/skills`, and plugin caches. New `ragbot skills {list,info,index}` CLI. The compiler can include skills via a `sources.skills` block in `compile-config.yaml`.
- **Workspace-rooted layout.** AI Knowledge repos are discovered across `~/workspaces/*/ai-knowledge-*` and via the synthesis-engineering shared `~/.synthesis/console.yaml` source list. Configuration moved to `~/.synthesis/` (legacy `~/.config/ragbot/` falls through).

Plus reasoning-effort wiring (Claude 4.x adaptive thinking, GPT-5.5 reasoning, Gemini 3.x thinking levels) — see `--thinking-effort` and `RAGBOT_THINKING_EFFORT`.

## Synthesis Engineering Ecosystem

Ragbot is a reference implementation of the synthesis-engineering methodology, focused on the **conversational** interaction primitive. Sibling reference implementations cover other primitives: **synthesis-console** for direct manipulation (browse and edit), **Ragenie** for the procedural primitive (workflow definition with autonomous execution), and **synthesis-skills** as the portable capability format consumed by every runtime and by external SKILL.md-compatible agents (Claude Code, Codex CLI, Cursor, Gemini CLI). The family of reference implementations will grow as the methodology and the AI landscape evolve. All implementations share the `~/.synthesis/` config home, the `ai-knowledge-*` workspace model, and a Python substrate library, and they integrate through Model Context Protocol (MCP) calls and a filesystem-as-source-of-truth contract.

## Development Methodology

Ragbot is developed using **Synthesis Engineering** (also known as **Synthesis Coding**)—a systematic approach that combines human architectural expertise with AI-assisted implementation. This methodology ensures that while AI accelerates development velocity, engineers maintain architectural authority, enforce quality standards, and deeply understand every component of the system.

Key principles applied in Ragbot's development:
- Human-defined architecture with AI-accelerated implementation
- Systematic quality assurance regardless of code origin
- Context preservation across development sessions
- Iterative refinement based on real-world usage

Learn more about this approach:
- [Synthesis Engineering: The Professional Practice](https://rajiv.com/blog/2025/11/09/synthesis-engineering-the-professional-practice-emerging-in-ai-assisted-development/)
- [The Organizational Framework](https://rajiv.com/blog/2025/11/09/the-synthesis-engineering-framework-how-organizations-build-production-software-with-ai/)
- [Technical Implementation with Claude Code](https://rajiv.com/blog/2025/11/09/synthesis-engineering-with-claude-code-technical-implementation-and-workflows/)

Code Contributors & Collaborators
- [Alexandria Redmon](https://github.com/alexdredmon)
- [Ishaan Jaffer](https://www.linkedin.com/in/reffajnaahsi/)
- [Trace Wax](https://github.com/tracedwax)
- [Jim Mortko](https://github.com/jskills)
- [Vik Pant](https://www.linkedin.com/in/vikpant/)

How to Contribute

Your code contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for important safety guidelines (especially about not committing personal data), then fork the repository and submit a pull request with your improvements.

[![](https://img.shields.io/badge/Chat_with_Ragbot.ai-Ask_Cody-%238A16D7?labelColor=%23383838)](https://sourcegraph.com/github.com/synthesisengineering/ragbot)

Quick Start
-----------

Get Ragbot running in 5 minutes:

### Option 1: Quick Start with Example Data (Fastest)

```bash
# 1. Clone this repository
git clone https://github.com/synthesisengineering/ragbot.git
cd ragbot

# 2. Set up your API keys
cp .env.docker .env
# Edit .env and add at least one API key (OpenAI, Anthropic, or Gemini)

# 3. Get starter templates from ai-knowledge-ragbot
git clone https://github.com/rajivpant/ai-knowledge-ragbot.git ~/ai-knowledge-ragbot
cp -r ~/ai-knowledge-ragbot/source/datasets/templates/ datasets/my-data/
cp ~/ai-knowledge-ragbot/source/instructions/templates/default.md instructions/

# 4. Customize with your information
# Edit the files in datasets/my-data/ with your personal details

# 5. Start Ragbot with Docker
docker-compose up -d

# 6. Access the web interface
open http://localhost:3000
```

### Option 2: Using Your Own Data Repository (Recommended for Production)

If you want to keep your data in a separate directory or private repository:

```bash
# 1. Clone Ragbot
git clone https://github.com/synthesisengineering/ragbot.git
cd ragbot

# 2. Create your data directory
mkdir ~/ragbot-data
# Or clone your private data repo: git clone <your-private-repo> ~/ragbot-data

# 3. Set up Docker override
cp docker-compose.override.example.yml docker-compose.override.yml
# Edit docker-compose.override.yml to point to your data directory

# 4. Organize your data (get templates from ai-knowledge-ragbot)
git clone https://github.com/rajivpant/ai-knowledge-ragbot.git ~/ai-knowledge-ragbot
cp -r ~/ai-knowledge-ragbot/source/datasets/templates/* ~/ragbot-data/datasets/
cp ~/ai-knowledge-ragbot/source/instructions/templates/default.md ~/ragbot-data/instructions/

# 5. Configure API keys
cp .env.docker .env
# Edit .env with your API keys

# 6. Start Ragbot
docker-compose up -d
```

### What's Next?

- 📖 **Knowledge Base:** Get templates and runbooks from [ai-knowledge-ragbot](https://github.com/rajivpant/ai-knowledge-ragbot)
- 🎓 **Understand the philosophy:** Read [docs/DATA_ORGANIZATION.md](docs/DATA_ORGANIZATION.md)
- 🐳 **Docker deployment:** See [README-DOCKER.md](README-DOCKER.md) for deployment guide
- 🤝 **Contributing safely:** Read [CONTRIBUTING.md](CONTRIBUTING.md) before contributing
- ⚙️ **Detailed setup:** Follow the [installation guide](INSTALL.md) and [configuration guide](CONFIGURE.md)

Try the Demo (v3.2+)
--------------------

Want to evaluate ragbot end-to-end without setting up a workspace, a
database, or any data? Use demo mode:

```bash
git clone https://github.com/synthesisengineering/ragbot.git
cd ragbot
cp .env.example .env
# Edit .env to set at least one API key (Anthropic, OpenAI, or Google).

# Install Python deps (in your preferred virtualenv):
pip install -r requirements.txt

# Start the bundled Postgres + ragbot stack:
docker compose up -d

# Run any subcommand in demo mode:
RAGBOT_DEMO=1 python3 src/ragbot.py db status
RAGBOT_DEMO=1 python3 src/ragbot.py skills list
RAGBOT_DEMO=1 python3 src/ragbot.py chat -p "What is ragbot?"
```

Or for the Web UI:

```bash
RAGBOT_DEMO=1 python3 -m uvicorn src.api.main:app --port 8000 &
cd web && npm install && NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
# Open http://localhost:3000 — you'll see a yellow "🎭 Demo mode" banner.
```

Demo mode hard-isolates discovery to the bundled `demo/ai-knowledge-demo/`
workspace and `demo/skills/ragbot-demo-skill/`. Real workspaces on the
host are invisible to discovery while `RAGBOT_DEMO=1` is set, so the
demo is safe to drive in front of an audience or to capture
screenshots from. Unset the env var to return to your real workspaces.

Configuration Layout (v3+)
-------------------------

Ragbot's user configuration lives under `~/.synthesis/` (a shared home for synthesis-engineering tools — keys are reused by other tools in the family without duplication):

```
~/.synthesis/
├── keys.yaml         # API keys (per-provider; per-workspace overrides supported)
├── ragbot.yaml       # Ragbot user prefs (default_workspace, etc.)
└── console.yaml      # Synthesis-console source list (optional; ragbot reads it for repo discovery)
```

Legacy `~/.config/ragbot/{keys,config}.yaml` is still read as a fallback so existing setups keep working.

Vector Backend
--------------

Ragbot's default vector store is PostgreSQL with the `pgvector` extension. The schema is shared across workspaces (one `chunks` table with a `workspace` column, an HNSW vector index for cosine ANN, and a generated `tsvector` + GIN index for native full-text search). Migrations are applied idempotently on first connection.

For local development without Docker, install pgvector for your PostgreSQL and point `RAGBOT_DATABASE_URL` at it. With Docker Compose, the bundled `postgres` service starts automatically. See [CONFIGURE.md](CONFIGURE.md) for both paths.

Use `ragbot db status` to confirm pgvector connectivity, list collections, and verify migrations are applied.

Agent Skills
------------

Ragbot indexes Agent Skills (directories containing `SKILL.md`) as first-class content. The full directory tree is honoured — `references/**/*.md` and bundled scripts (`*.py`, `*.sh`, etc.) are all indexed and become queryable via RAG.

```bash
ragbot skills list                   # show all discovered skills
ragbot skills info <skill-name>      # full details for one skill
ragbot skills index                  # index all skills into the 'skills' workspace
```

Discovery sources (later wins on name collision):

1. `~/.synthesis/skills/` (synthesis-engineering shared install)
2. `~/.claude/skills/` (Claude Code private)
3. `~/.claude/plugins/cache/<vendor>/skills/` (plugin-installed)
4. Per-workspace roots declared in `compile-config.yaml` (`sources.skills.roots`)

When the `skills` workspace has indexed content, `ragbot chat` automatically merges its results with the user's selected workspace via cross-workspace retrieval. Disable per-call with `--no-skills` or programmatically via `additional_workspaces=[]`.

Model Tiers
-----------

Every model in `engines.yaml` declares a `tier` — **judgment**, **routine**, or **bulk**. The labels name the work a model is trusted with, not the artifact: judgment-tier models handle work where being wrong is expensive; routine-tier models are the balanced default for rule-following execution; bulk-tier models take high-volume, low-stakes operations (Ragbot's RAG pipeline uses the same provider's bulk-tier model for query planning and other auxiliary calls). A fourth selector, `flagship` (`is_flagship: true`), marks each provider's showcase model and drives default thinking effort — distinct from tier, since a provider can have several flagship-class local models and a judgment-tier model that isn't its flagship.

The vocabulary is shared with the [synthesis-model-tiers skill](https://github.com/synthesisengineering/synthesis-skills/tree/main/synthesis-model-tiers), whose `tiers.yaml` holds the cross-provider ordered-preference table; `tests/test_engines_yaml.py` enforces that the two files agree wherever the skill is installed. The words are deliberately chosen from outside vendor marketing vocabulary — the industry applies "frontier" to entire model generations, so work-based labels are the ones no vendor can collide with. Full naming rationale: the skill's `references/naming-rationale.md`.

Compiler targets in `compile-config.yaml` select models the same way, via `model_tier: flagship | judgment | routine | bulk`.

Reasoning / Thinking Modes
--------------------------

Models that advertise thinking support in `engines.yaml` (Claude Fable 5 / Opus 4.8 / Sonnet 5 / Haiku 4.5, GPT-5.6 Sol/Terra, Gemini 3.x) are wired through LiteLLM's `reasoning_effort` parameter. Defaults: flagship models → `medium`, non-flagship with thinking → `off`, models without thinking metadata → silent (no params sent).

```bash
ragbot chat --thinking-effort high -p "explain this..."          # explicit high effort
RAGBOT_THINKING_EFFORT=low ragbot chat -p "explain this..."      # globally low
ragbot chat --thinking-effort off -p "..."                       # disable on a flagship
```

RAG (Retrieval-Augmented Generation)
------------------------------------

Ragbot implements a **production-grade, multi-stage RAG pipeline** based on research from leading AI systems including Perplexity, ChatGPT, Claude, and Gemini. Unlike simple RAG implementations, Ragbot uses sophisticated techniques proven to significantly improve retrieval accuracy.

### Architecture Overview

```
Query → Phase 1 → Phase 2 → Phase 3 → Generate → Phase 4 → Response
        Foundation  Query     Hybrid    Response   Verify    with
                   Intel    Retrieval             & CRAG   Confidence
```

**Four-Phase Pipeline:**

| Phase | Description | Key Techniques |
|-------|-------------|----------------|
| **Phase 1** | Foundation | Query preprocessing, full document retrieval, 16K context budget |
| **Phase 2** | Query Intelligence | LLM planner, multi-query expansion (5-7 variations), HyDE |
| **Phase 3** | Hybrid Retrieval | BM25 + Vector search, Reciprocal Rank Fusion, LLM reranking |
| **Phase 4** | Verification | Hallucination detection, confidence scoring, CRAG loop |

### Research-Backed Impact

Based on benchmarks from Anthropic, Microsoft, and other research:

| Technique | Impact |
|-----------|--------|
| Contextual embeddings | **35% fewer retrieval failures** |
| Hybrid search + reranking | **67% fewer retrieval failures** |
| Query rewriting (multi-query) | **+21 NDCG points** |

### Phase 1: Foundation Layer

- **Query Preprocessing**: Expands contractions ("what's" → "what is"), extracts key terms
- **Document Detection**: Recognizes "show me my biography" style queries
- **Full Document Retrieval**: Returns complete documents instead of fragments when appropriate
- **Enhanced Embeddings**: Includes filename and title in embeddings for better matching

### Phase 2: Query Intelligence

- **LLM Query Planner**: Analyzes intent, determines retrieval strategy
- **Multi-Query Expansion**: Generates 5-7 query variations for better recall
- **HyDE (Hypothetical Document Embeddings)**: Generates hypothetical answers for semantic search
- **Provider-Agnostic**: Uses fast model from same provider as user's selection

### Phase 3: Hybrid Retrieval

- **Dual Search**: Combines semantic (vector) and lexical (BM25) search
- **Reciprocal Rank Fusion**: Merges results from both search methods
- **LLM Reranking**: Scores relevance 0-10, reorders by combined score
- **Result**: Best of both semantic understanding and exact keyword matching

### Phase 4: Response Verification

- **Claim Extraction**: Identifies factual claims in generated responses
- **Evidence Matching**: Checks each claim against retrieved context
- **Confidence Scoring**: 0.0-1.0 score based on claim verification
- **CRAG (Corrective RAG)**: Re-retrieves for low-confidence responses (<0.7)

### Using RAG in the Web UI

1. Select a workspace in the sidebar
2. Click **"Index Workspace"** in Advanced Settings to build the index (first time only)
3. Enable **"Enable RAG"** checkbox
4. Adjust **"RAG context tokens"** slider to control how much context is retrieved

### Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| Enable RAG | On | Toggle RAG-augmented responses |
| RAG context tokens | 16000 | Maximum tokens for retrieved context |
| Confidence threshold | 0.7 | CRAG triggers below this score |
| Embedding model | all-MiniLM-L6-v2 | 384-dimension embeddings |

### Technical Details

- **Vector Database**: PostgreSQL with the `pgvector` extension (default DSN via `RAGBOT_DATABASE_URL`; bundled `postgres` service in `docker-compose.yml`)
- **Embedding Model**: sentence-transformers `all-MiniLM-L6-v2` (80MB, 384 dimensions)
- **Chunking**: ~500 tokens per chunk with 50-token overlap
- **Similarity**: Cosine distance for semantic matching

For the complete technical architecture, see [docs/rag-architecture.md](docs/rag-architecture.md).

AI Knowledge Integration
------------------------

Ragbot integrates with the **AI Knowledge** ecosystem for managing knowledge bases across multiple workspaces.

### Open Source Knowledge Base

The **[ai-knowledge-ragbot](https://github.com/rajivpant/ai-knowledge-ragbot)** repository contains open-source runbooks, templates, and guides that ship with Ragbot:

- **Instruction templates** - Starter configurations for AI assistants
- **Dataset templates** - Personal and professional profile templates
- **Runbooks** - Procedures for content creation, communication, system configuration
- **Guides** - Reference materials for working with AI

Personal ai-knowledge repos can inherit from ai-knowledge-ragbot to get these shared resources while adding private content.

### AI Knowledge System

The AI Knowledge system manages content across multiple workspaces using a three-part architecture:

| Operation | Where | When |
|-----------|-------|------|
| Knowledge concatenation (`all-knowledge.md`) | CI/CD (GitHub Actions) | Every push to `source/` |
| Instruction compilation | Local (`ragbot compile`) | When instructions change (rare) |
| RAG indexing | Local (`ragbot index`) | When content changes + RAG needed |

**Key concept:** Edit `source/` files directly. Knowledge concatenation is automatic. See [docs/compilation-guide.md](docs/compilation-guide.md) for details.

```
ai-knowledge-{workspace}/
├── source/                    # Your source files (authoritative)
│   ├── instructions/          # WHO - Identity, persona, rules
│   ├── runbooks/             # HOW - Procedures, workflows
│   └── datasets/             # WHAT - Reference knowledge
├── compiled/                  # Auto-generated
│   └── {project}/
│       └── instructions/      # LLM-specific (claude.md, chatgpt.md, gemini.md)
└── all-knowledge.md           # Concatenated knowledge (CI/CD via GitHub Actions)
```

Quick examples:

```bash
# Compile instructions for a project
ragbot compile --project {name} --no-llm

# Index workspace for RAG
ragbot index --workspace {name}
```

For detailed setup instructions, see the [LLM Project Setup Guide](https://github.com/rajivpant/ai-knowledge-ragbot/blob/main/source/runbooks/system-config/llm-project-setup.md).

### Convention-Based Discovery

Ragbot automatically discovers AI Knowledge repositories by convention:

1. Mount your `ai-knowledge` parent directory to `/app/ai-knowledge`
2. Ragbot scans for directories matching `ai-knowledge-{workspace}`
3. Each discovered repo provides instructions and knowledge for that workspace

### Docker Setup with AI Knowledge

```yaml
# docker-compose.override.yml
services:
  ragbot-web:
    volumes:
      - ${HOME}/workspaces:/root/workspaces:ro
      - ./workspaces:/app/workspaces:ro
```

### Workspace Configuration

Create `workspace.yaml` files to customize workspace behavior:

```yaml
# workspaces/my-project/workspace.yaml
name: My Project
description: Project-specific AI assistant
status: active
type: work
inherits_from:
  - personal  # Inherit from personal workspace
```

### Content Loading Strategy

| Content Type | Loading Method | Use Case |
|--------------|----------------|----------|
| Instructions | Always loaded | Core identity and behavior |
| Datasets | Direct or RAG | Small: direct, Large: RAG |
| Runbooks | RAG retrieval | Retrieved when relevant |

Supported AI Models
-------------------

Ragbot supports models from Anthropic (Claude), OpenAI (GPT and reasoning models), Google (Gemini), and local open-weights models via Ollama. The authoritative list — model IDs, context windows, thinking-mode support, tier badges, and defaults — lives in [engines.yaml](engines.yaml). v3.3's redesigned model picker reads from the same file at runtime, so what you see in the UI matches what is configured in the repo.

Adding or updating models is an `engines.yaml` change, not a code change. See the [v3.3 release notes](#whats-new-in-v33) for the local-model integration details.

### Open-weights model support

Ragbot v3.4 ships expanded local-model coverage out of the box. The `ollama` engine in [engines.yaml](engines.yaml) includes Gemma 4 (E4B, 26B MoE, 31B Dense), Llama 4 (Scout, Maverick), Qwen3.6 (27B Dense, 35B-A3B MoE), DeepSeek V3.2, and Mistral (Small 4, Medium 3.5, Large 3). Each entry carries MLX-backend notes, license info, recommended quantization tags, and a real-world parameter count.

**Choosing a model for your Mac.** Open-weights inference on Apple Silicon is bound by unified memory and bandwidth. A 16 GB Mac mini runs Gemma 4 E4B; a Mac Studio with 256 GB unified memory runs DeepSeek V3.2 and Mistral Large 3. The [sizing matrix at `docs/sizing-matrix.md`](docs/sizing-matrix.md) maps every model in `engines.yaml` to each Mac hardware profile (Mac mini 16/32/64 GB, MacBook Air 24/32 GB, MacBook Pro M4 Pro 48 / M5 Max 128 GB, Mac Studio M3 Ultra 192/256 GB), with per-model memory footprints at FP16 and Q4, expected MLX tokens-per-second, and `comfortable / tight / Q4-only / won't fit` verdicts. Start there if you're trying to pick a model for your Mac, or pick a Mac for a model.

Installation, Configuration, and Personalization
------------------------------------------------
Read the [installation guide](INSTALL.md) and the [configuration and personaliation guide](CONFIGURE.md).

Using the Web version
---------------------
_The screenshots below were captured against the bundled demo workspace.
The model-picker and advanced-panel shots reflect v3.3 (May 2026); the
chat-and-skills shots are still v3.2 captures since those features didn't
change. The full sets are at [`screenshots/v3.3/`](screenshots/v3.3/) and
[`screenshots/v3.2/`](screenshots/v3.2/)._

**Settings panel and welcome state (v3.3)**

![Ragbot Web UI — settings panel with the new model picker](screenshots/v3.3/01-hero-model-picker.png)

**Model picker open — Pinned, Recent, and by-provider sections (v3.3)**

![Ragbot Web UI — model picker open](screenshots/v3.3/02-model-picker-open.png)

**A chat that retrieves from the bundled sample documents (v3.2)**

![Ragbot Web UI — chat with RAG-augmented response](screenshots/v3.2/02-chat-with-response.png)

**Advanced settings expanded (v3.3)**

![Ragbot Web UI — advanced settings expanded, four providers configured](screenshots/v3.3/05-advanced-settings.png)

**Cross-workspace skills auto-include (v3.2)**

A follow-up question retrieves from the bundled demo skill via the cross-workspace fan-out:

![Ragbot Web UI — cross-workspace skills retrieval](screenshots/v3.2/04-skills-via-cross-workspace.png)

<!--
Legacy screenshots from earlier ragbot versions, kept on disk for archival
reference but no longer rendered in the README. The v3.2 captures above
reflect the current Web UI.
  - screenshots/Screenshot 2023-06-16 at 10.53.12 PM.png
  - screenshots/Screenshot 2024-04-10 at 10.46.02 PM.png
  - screenshots/Screenshot 2023-08-02 at 10.30.37 PM.png
-->


Using the command line interface
--------------------------------

The CLI uses workspaces with RAG (Retrieval-Augmented Generation) and automatically loads LLM-specific instructions based on the model you're using.

### Key CLI Options

```
ragbot chat [options]

Input Options:
  -p, --prompt PROMPT          Prompt text
  -f, --prompt_file FILE       Read prompt from file
  -i, --interactive            Interactive mode with history
  --stdin                      Read prompt from stdin

Workspace & Knowledge:
  -profile NAME                Workspace to use (auto-loads instructions and enables RAG)
  --rag / --no-rag            Enable/disable RAG retrieval (default: enabled)

Model Selection:
  -e {openai,anthropic,google} Engine/provider
  -m MODEL                     Model name (or 'flagship' for best)

Custom Instructions:
  -c PATH [PATH ...]           Explicit instruction files (overrides auto-loading)
  -nc                          Disable all instructions
```

### Using Workspaces with RAG

The recommended way to use the CLI is with workspaces:

```bash
# Chat with a workspace - instructions auto-loaded, RAG enabled
ragbot chat -profile personal -p "What are my travel preferences?"

# Use Anthropic Claude (loads claude.md instructions)
ragbot chat -profile personal -e anthropic -p "Summarize my work history"

# Use OpenAI GPT-5.2 (loads chatgpt.md instructions)
ragbot chat -profile personal -e openai -m gpt-5.2 -p "Summarize my work history"

# Use Google Gemini (loads gemini.md instructions)
ragbot chat -profile personal -e google -p "Summarize my work history"
```

### LLM-Specific Instructions

The system automatically loads the correct instruction file based on the LLM:

| Engine | Instruction File |
|--------|------------------|
| anthropic | `compiled/{workspace}/instructions/claude.md` |
| openai | `compiled/{workspace}/instructions/chatgpt.md` |
| google | `compiled/{workspace}/instructions/gemini.md` |

### Interactive Mode

Maintain conversation history across multiple prompts:

```bash
ragbot chat -profile personal -i

> Tell me about my professional background
Ragbot.AI: [response based on RAG-retrieved knowledge]

> Summarize it in 3 bullet points
Ragbot.AI: [continues with context]

> /save session.json
Conversation saved to ...

> /quit
```

### CLI Reference

Full help output for `ragbot chat`:

```console
$ ragbot chat --help
usage: ragbot chat [-h] [-ls] [-p PROMPT | -f PROMPT_FILE | -i | --stdin]
                 [-profile PROFILE] [-c [CUSTOM_INSTRUCTIONS ...]] [-nc]
                 [--rag] [--no-rag]
                 [-e {openai,anthropic,google}] [-m MODEL] [-t TEMPERATURE]
                 [-mt MAX_TOKENS] [-l LOAD]

Ragbot.AI is an augmented brain and assistant. Learn more at https://ragbot.ai

options:
  -h, --help            show this help message and exit
  -ls, --list-saved     List all the currently saved JSON files.
  -p, --prompt          The user's input prompt
  -f, --prompt_file     Read prompt from a file
  -i, --interactive     Enable interactive mode with conversation history
  --stdin               Read prompt from stdin
  -profile              Workspace name (enables RAG and auto-loads instructions)
  -c                    Custom instruction file paths (overrides auto-loading)
  -nc                   Disable custom instructions
  --rag                 Enable RAG retrieval (default)
  --no-rag              Disable RAG - instructions only
  -e {openai,anthropic,google}  LLM engine/provider
  -m MODEL              Model name or 'flagship'
  -t TEMPERATURE        Creativity (0-2)
  -mt MAX_TOKENS        Max response tokens
  -l LOAD               Load previous session from file
```

### Knowledge Retrieval (RAG)

Knowledge is retrieved via RAG (Retrieval-Augmented Generation) from indexed workspace content:

```bash
ragbot chat -profile personal -p "What are my travel preferences?"
# RAG enabled for workspace: personal
# [Response based on retrieved knowledge]
```


---

Ragbot is by **Synthesis Engineering** · [synthesisengineering.org](https://synthesisengineering.org) · [synthesiscoding.org](https://synthesiscoding.org) · MIT License
