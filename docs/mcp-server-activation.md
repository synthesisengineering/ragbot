# Activating RagbotMCPServer

Ragbot ships an MCP **server** (`synthesis_engine.mcp_server.RagbotMCPServer`) — distinct from the MCP **client** wiring in `synthesis_engine.mcp` (the `/api/mcp/*` endpoints, which are Ragbot calling *out* to other MCP servers). This document is about the server side: exposing Ragbot's own primitives — `workspace_search`, `workspace_search_multi`, `document_get`, `skill_run`, `agent_run_start` — to an external MCP client such as Claude Code, Cursor, or another agent.

The server itself and its tool dispatch are fully built and unit-tested (`tests/test_mcp_server.py`). As of this change, it also has a CLI entry point: `ragbot mcp serve`. It is **not** registered with any client and **not** running anywhere by default — this document is what a user does to actually turn it on.

## Two transports, two trust models

| Transport | CLI command | Auth | Who uses it |
|---|---|---|---|
| **stdio** | `ragbot mcp serve` | None — process-local trust | Claude Code, Cursor, and other desktop clients that spawn the server as a subprocess |
| **HTTP/SSE** | *(no CLI subcommand yet — `RagbotMCPServer.serve_http()` exists in code but isn't wired to `ragbot mcp`)* | Bearer token via `~/.synthesis/mcp-server.yaml` | Remote clients, or anything that can't spawn a local process |

**Important correction to a common assumption:** the bearer-token config at `~/.synthesis/mcp-server.yaml` is **not** needed to register `ragbot mcp serve` with Claude Code. Stdio is process-local — whatever spawned the process (Claude Code) is the trust boundary, and `RagbotMCPServer.serve_stdio()` never consults the bearer-token config at all (see `synthesis_engine/mcp_server/server.py::serve_stdio`). The token file only matters if/when the HTTP/SSE transport gets its own CLI subcommand — that's a real follow-up, not something this change builds, and there is no token to generate for the stdio path described below.

## Step 1 — Confirm the server starts

From the repo root, with a reachable Postgres (`RAGBOT_DATABASE_URL` set):

```bash
ragbot mcp serve
```

This blocks in the foreground and speaks MCP over stdin/stdout. Ctrl-C (or the client disconnecting) stops it. If the memory backend (pgvector) isn't reachable, the command fails fast with a clear error pointing at `ragbot db status` — it does not start a half-wired server.

## Step 2 — Register it with Claude Code

Claude Code's local (stdio) MCP servers are registered either via `claude mcp add`, or by hand in the relevant MCP config JSON (project-level `.mcp.json`, or the user-level config, depending on scope). The registration is a command + args pair — no different in shape from any other local MCP server:

```json
{
  "mcpServers": {
    "ragbot": {
      "command": "ragbot",
      "args": ["mcp", "serve"],
      "env": {
        "RAGBOT_DATABASE_URL": "postgresql://ragbot_app:<password>@localhost:5433/ragbot"
      }
    }
  }
}
```

Adjust `command` to an absolute path (or activate the right virtualenv/interpreter) if `ragbot` isn't on `PATH` in the environment Claude Code spawns from. Set any other env vars the process needs to resolve workspaces the same way the CLI does (`RAGBOT_OWNER_CONTEXT=1` if you want private-repo and archive-tier content reachable through `workspace_search`/`workspace_search_multi` — see the caveat below).

`RAGBOT_DATABASE_URL` above should point at the restricted `ragbot_app` role, not the Postgres superuser role — see [`docs/rls-and-roles.md`](rls-and-roles.md) for why that split exists and matters even for a read-mostly consumer like this MCP server.

Exact registration mechanics (CLI flag vs. hand-edited JSON, project-scope vs. user-scope) are Claude Code's own concern and may vary by version — this is the one step in this document that's a live action for the user to perform and verify against their installed Claude Code version, not something this change can pre-validate.

## Step 3 — What the tools expose

* **`workspace_search`** / **`workspace_search_multi`** — three-tier retrieval (vector + entity-graph + session/user memory) over one or more workspaces. Routes through the same cross-workspace confidentiality gate (`AIR_GAPPED` never mixes, `CLIENT_CONFIDENTIAL` never mixes with `PUBLIC`, etc.) that the chat path enforces via `routing.yaml`.
* **`document_get`** — fetch one document's full reconstructed content by id (source path / filename), not a search — the caller already knows what they want.
* **`skill_run`** — invoke a skill-declared tool via `SkillRuntime`, gated by the skill's own visibility (must be visible from the requesting workspace) and its frontmatter permission gates.
* **`agent_run_start`** — kick off a background agent run (`AgentLoop.drive_to_terminal`) across one or more workspaces, returning a task id and status URL.

Every tool call still passes through the existing `PermissionRegistry` and the cross-workspace policy gate — MCP is a new transport into the same policy surface, not a bypass of it.

### Content-tier caveat (this change)

`workspace_search` / `workspace_search_multi` retrieve through the same default-tier filter as the chat path: **archive-tier content (`personal-archive/`, and private repos in owner context) is never returned**, by construction — the pgvector backend excludes the `archive-only` tier unless a caller explicitly asks for it (see `rag.search_archive()`). There is no MCP tool wired to `search_archive()` today; an external MCP client cannot reach archive-tier content through this server at all, regardless of `RAGBOT_OWNER_CONTEXT`. If archive-tier access via MCP is ever wanted, that's a new tool definition in `synthesis_engine/mcp_server/tools.py`, not a flag on the existing ones — deliberately, so archive-tier reachability stays an explicit, auditable surface rather than something that quietly rides along on an existing tool's default behavior.

## What this change does NOT do

* Does not generate a bearer token or write anything to `~/.synthesis/mcp-server.yaml` — that file does not exist yet on a fresh install, and creating it is a live decision for whoever wants the HTTP/SSE transport, not something to automate.
* Does not register `ragbot mcp serve` in any Claude Code config — Step 2 above is a manual action.
* Does not wire a CLI subcommand for `serve_http()` — stdio was the explicit target ("that's what Claude Code speaks for local MCP servers"); HTTP/SSE activation is a separate, not-yet-scoped follow-up.
