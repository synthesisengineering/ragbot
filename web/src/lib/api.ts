/**
 * API client for Ragbot FastAPI backend
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Types
export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export type ThinkingEffort = 'auto' | 'off' | 'minimal' | 'low' | 'medium' | 'high';

export interface VectorBackendInfo {
  backend?: string;
  ok?: boolean;
  pgvector_version?: string;
  workspaces?: number;
  [key: string]: unknown;
}

export interface ChatRequest {
  prompt: string;
  workspace?: string;
  model?: string;
  temperature?: number;
  max_tokens?: number;
  use_rag?: boolean;
  rag_max_tokens?: number;
  history?: Message[];
  stream?: boolean;
  /**
   * Reasoning / thinking effort. Defaults: flagship → "medium", non-flagship → "off",
   * models without thinking metadata → ignored. Override here or via the
   * RAGBOT_THINKING_EFFORT env on the server.
   */
  thinking_effort?: ThinkingEffort;
  /**
   * Extra workspaces to query alongside the primary one. When omitted, the
   * "skills" workspace is auto-included if it has indexed content. Pass an
   * empty array to opt out, or a list of workspace names for explicit fan-out.
   */
  additional_workspaces?: string[];
}

export interface WorkspaceInfo {
  name: string;
  dir_name: string;
  description?: string;
  status: string;
  type: string;
  inherits_from: string[];
  has_instructions: boolean;
  has_datasets: boolean;
  has_source: boolean;
  repo_path?: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  provider: string;
  context_window: number;
  supports_streaming: boolean;
  supports_system_role: boolean;
  available?: boolean;
  tier?: string;
  max_output_tokens?: number;
  temperature?: number;
  max_temperature?: number;
  /** Human-readable label (e.g. "Claude Opus 4.7"). Falls back to `name` if absent. */
  display_name?: string;
  /** True when the model exposes a thinking-effort control. */
  supports_thinking?: boolean;
  /** True for local-runtime providers (Ollama). No API key needed; no cloud egress. */
  is_local?: boolean;
  is_flagship?: boolean;
}

export interface ConfigResponse {
  version: string;
  ai_knowledge_root?: string;
  workspace_count: number;
  rag_available: boolean;
  default_model: string;
  default_workspace?: string;
  api_keys: Record<string, boolean>;
  workspaces_with_keys: string[];
  vector_backend?: VectorBackendInfo;
  /**
   * True when the server is running with RAGBOT_DEMO=1. The UI shows a
   * yellow banner and the discovery layer hard-isolates from real
   * workspaces/skills on the host.
   */
  demo_mode?: boolean;
}

export interface KeyStatus {
  has_key: boolean;
  source: 'workspace' | 'default' | null;
  has_workspace_key: boolean;
  has_default_key: boolean;
}

export type KeysStatusResponse = Record<string, KeyStatus>;

// API functions
export async function getConfig(): Promise<ConfigResponse> {
  const res = await fetch(`${API_BASE}/api/config`);
  if (!res.ok) throw new Error('Failed to fetch config');
  return res.json();
}

export async function getKeysStatus(workspace?: string): Promise<KeysStatusResponse> {
  const url = workspace
    ? `${API_BASE}/api/config/keys?workspace=${encodeURIComponent(workspace)}`
    : `${API_BASE}/api/config/keys`;
  const res = await fetch(url);
  if (!res.ok) throw new Error('Failed to fetch keys status');
  return res.json();
}

export async function getWorkspaces(): Promise<WorkspaceInfo[]> {
  const res = await fetch(`${API_BASE}/api/workspaces`);
  if (!res.ok) throw new Error('Failed to fetch workspaces');
  const data = await res.json();
  return data.workspaces;
}

export async function getModels(): Promise<{ models: ModelInfo[]; default_model: string; api_keys_configured?: Record<string, boolean> }> {
  // Use /api/models/all to get all models regardless of API key status
  const res = await fetch(`${API_BASE}/api/models/all`);
  if (!res.ok) throw new Error('Failed to fetch models');
  return res.json();
}

export interface ProviderInfo {
  id: string;
  name: string;
  is_local?: boolean;
}

export async function getProviders(): Promise<{ providers: ProviderInfo[] }> {
  const res = await fetch(`${API_BASE}/api/models/providers`);
  if (!res.ok) throw new Error('Failed to fetch providers');
  return res.json();
}

// ----- Preferences (pinned / recent models) -----

export async function getPinnedModels(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/api/preferences/pinned-models`);
  if (!res.ok) throw new Error('Failed to fetch pinned models');
  const data = await res.json();
  return data.model_ids ?? [];
}

export async function setPinnedModels(modelIds: string[]): Promise<string[]> {
  const res = await fetch(`${API_BASE}/api/preferences/pinned-models`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_ids: modelIds }),
  });
  if (!res.ok) throw new Error('Failed to update pinned models');
  const data = await res.json();
  return data.model_ids ?? [];
}

export async function getRecentModels(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/api/preferences/recent-models`);
  if (!res.ok) throw new Error('Failed to fetch recent models');
  const data = await res.json();
  return data.model_ids ?? [];
}

export async function recordRecentModel(modelId: string): Promise<string[]> {
  const res = await fetch(`${API_BASE}/api/preferences/recent-models`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id: modelId }),
  });
  if (!res.ok) throw new Error('Failed to record recent model');
  const data = await res.json();
  return data.model_ids ?? [];
}

export async function getTemperatureSettings(): Promise<Record<string, number>> {
  const res = await fetch(`${API_BASE}/api/models/temperature-settings`);
  if (!res.ok) throw new Error('Failed to fetch temperature settings');
  return res.json();
}

export interface IndexStatus {
  indexed: boolean;
  workspace?: string;
  chunk_count?: number;
  last_indexed?: string | null;
  error?: string;
}

export async function getIndexStatus(workspace: string): Promise<IndexStatus> {
  const res = await fetch(`${API_BASE}/api/workspaces/${workspace}/index`);
  if (!res.ok) throw new Error('Failed to get index status');
  return res.json();
}

export async function healthCheck(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`);
    return res.ok;
  } catch {
    return false;
  }
}

/**
 * Yielded values from chatStream. Most yields are token strings; structured
 * events (the initial task announcement, a mid-stream cancellation notice)
 * surface as discriminated-union objects so callers can plumb them into
 * UI state.
 *
 * Background: the SSE wire format ships these as distinct ``event:`` types
 * (``task``, ``message``, ``cancelled``, ``done``, ``error``). Mapping them
 * onto a single async iterator keeps the consumer's code (``for await``)
 * uniform; consumers that only care about tokens can ``typeof === 'string'``-
 * guard and ignore everything else.
 */
export type ChatStreamEvent =
  | string
  | { type: 'task'; task_id: string }
  | { type: 'cancelled'; task_id: string; reason: string };

/**
 * Send a chat message and stream the response.
 *
 * Yields:
 *  - ``string`` — a token of generated content.
 *  - ``{type: 'task', task_id}`` — emitted exactly once, as the first
 *    event from the stream. Lets the caller wire ⌘B / ⌘. to the live
 *    task id before any tokens arrive.
 *  - ``{type: 'cancelled', task_id, reason}`` — emitted at most once,
 *    before the stream ends, when the server transitions the task to
 *    ``cancelled``. The caller is responsible for surfacing the
 *    cancellation in the UI.
 *
 * Errors:
 *  - HTTP non-2xx → ``Error`` thrown synchronously.
 *  - Mid-stream backend failure → ``Error`` thrown from the iterator.
 */
export async function* chatStream(
  request: ChatRequest,
): AsyncGenerator<ChatStreamEvent> {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'text/event-stream',
    },
    body: JSON.stringify({
      ...request,
      stream: true,
    }),
  });

  if (!res.ok) {
    throw new Error(`Chat failed: ${res.statusText}`);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error('No response body');

  const decoder = new TextDecoder();
  let buffer = '';
  // SSE frames may span ``event:`` and ``data:`` lines; we accumulate the
  // most-recent ``event`` label and apply it to the next ``data`` line.
  let currentEvent = 'message';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    // SSE frames are separated by '\n\n'; lines within a frame by '\n'.
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (line.startsWith('event: ')) {
        currentEvent = line.slice(7).trim() || 'message';
        continue;
      }
      if (line === '') {
        // Frame delimiter — reset to default for the next frame.
        currentEvent = 'message';
        continue;
      }
      if (line.startsWith('data: ')) {
        const data = line.slice(6);
        if (data === '[DONE]') return;

        try {
          const parsed = JSON.parse(data);
          switch (currentEvent) {
            case 'task':
              if (parsed.task_id) {
                yield { type: 'task', task_id: parsed.task_id };
              }
              break;
            case 'cancelled':
              yield {
                type: 'cancelled',
                task_id: parsed.task_id,
                reason: parsed.reason || 'cancelled',
              };
              return;
            case 'done':
              return;
            case 'error':
              throw new Error(parsed.error || 'chat stream failed');
            case 'message':
            default:
              if (parsed.content) {
                yield parsed.content;
              }
              if (parsed.error) {
                throw new Error(parsed.error);
              }
              break;
          }
        } catch (e) {
          // JSON parse failures are skipped; thrown ``Error`` instances
          // (from the explicit throw above) propagate to the caller.
          if (e instanceof Error && e.message !== 'Unexpected end of JSON input') {
            // Distinguish "couldn't parse" from "server-reported error".
            if (currentEvent === 'error') throw e;
          }
        }
      }
    }
  }
}

/**
 * Send a chat message without streaming
 */
export async function chat(request: ChatRequest): Promise<string> {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      ...request,
      stream: false,
    }),
  });

  if (!res.ok) {
    throw new Error(`Chat failed: ${res.statusText}`);
  }

  const data = await res.json();
  return data.response;
}

/**
 * Trigger workspace indexing
 */
export async function indexWorkspace(name: string, force = false): Promise<void> {
  const res = await fetch(`${API_BASE}/api/workspaces/${name}/index`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ force }),
  });

  if (!res.ok) {
    throw new Error(`Indexing failed: ${res.statusText}`);
  }
}

// ----- MCP Servers -----
//
// Mirrors the `synthesis_engine.mcp` substrate. The backend persists user
// edits to `~/.synthesis/mcp.yaml` and surfaces live connection state from
// the in-process registry.

export type McpTransport = 'stdio' | 'http' | 'sse';
export type McpAuthMode = 'none' | 'oauth' | 'bearer';
export type McpConnectionState =
  | 'disconnected'
  | 'connecting'
  | 'connected'
  | 'error';

export interface McpAuthConfig {
  mode: McpAuthMode;
  client_id_metadata_url?: string | null;
  client_name?: string;
  redirect_port?: number;
  scope?: string | null;
  /** Bearer token; backend redacts on read. Only present when the user types one in. */
  token?: string | null;
}

export interface McpServer {
  id: string;
  name: string;
  description?: string;
  transport: McpTransport;
  // stdio
  command?: string | null;
  args?: string[];
  env?: Record<string, string>;
  cwd?: string | null;
  // http / sse
  url?: string | null;
  headers?: Record<string, string>;
  // workspace gating
  workspace_allow?: string[] | null;
  workspace_deny?: string[];
  // auth
  auth?: McpAuthConfig;
  // lifecycle
  enabled: boolean;
  timeout_seconds?: number;
  // live state (server-populated; absent on POST/PUT requests)
  connection_state?: McpConnectionState;
  last_error?: string | null;
  tool_count?: number;
  resource_count?: number;
  prompt_count?: number;
}

export interface McpTool {
  name: string;
  title?: string | null;
  description?: string | null;
  input_schema?: Record<string, unknown> | null;
}

export interface McpResource {
  uri: string;
  name?: string | null;
  description?: string | null;
  mime_type?: string | null;
}

export interface McpPrompt {
  name: string;
  title?: string | null;
  description?: string | null;
  arguments?: Array<{
    name: string;
    description?: string | null;
    required?: boolean;
  }>;
}

export interface McpServersResponse {
  servers: McpServer[];
}

/**
 * Payload accepted by `addMcpServer`. Mirrors `McpServer` but excludes the
 * runtime/live-state fields the backend computes.
 */
export type McpServerInput = Omit<
  McpServer,
  'connection_state' | 'last_error' | 'tool_count' | 'resource_count' | 'prompt_count'
>;

export async function listMcpServers(): Promise<McpServer[]> {
  const res = await fetch(`${API_BASE}/api/mcp/servers`);
  if (!res.ok) throw new Error(`Failed to list MCP servers: ${res.statusText}`);
  const data = (await res.json()) as McpServersResponse;
  return data.servers ?? [];
}

export async function addMcpServer(server: McpServerInput): Promise<McpServer> {
  const res = await fetch(`${API_BASE}/api/mcp/servers`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(server),
  });
  if (!res.ok) {
    const detail = await safeErrorDetail(res);
    throw new Error(`Failed to add MCP server: ${detail}`);
  }
  return res.json();
}

export async function removeMcpServer(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/api/mcp/servers/${encodeURIComponent(id)}`, {
    method: 'DELETE',
  });
  if (!res.ok) {
    const detail = await safeErrorDetail(res);
    throw new Error(`Failed to remove MCP server: ${detail}`);
  }
}

/**
 * Flip the `enabled` flag on a server. Returns the updated server record so
 * the caller can refresh its row without a separate `listMcpServers()` call.
 */
export async function toggleMcpServer(id: string): Promise<McpServer> {
  const res = await fetch(
    `${API_BASE}/api/mcp/servers/${encodeURIComponent(id)}/toggle`,
    { method: 'POST' },
  );
  if (!res.ok) {
    const detail = await safeErrorDetail(res);
    throw new Error(`Failed to toggle MCP server: ${detail}`);
  }
  return res.json();
}

/**
 * Trigger the OAuth browser flow for a remote server. The backend launches
 * `LocalBrowserOAuthFlow`, opens the system browser, and listens on the
 * configured loopback port. Returns once tokens have been stored or the
 * flow errors. The UI can also poll `listMcpServers()` to watch the
 * connection state transition through `connecting → connected`.
 */
export async function startMcpOAuth(id: string): Promise<{ ok: boolean; error?: string }> {
  const res = await fetch(
    `${API_BASE}/api/mcp/servers/${encodeURIComponent(id)}/oauth`,
    { method: 'POST' },
  );
  if (!res.ok) {
    const detail = await safeErrorDetail(res);
    return { ok: false, error: detail };
  }
  return res.json();
}

export async function listMcpTools(id: string): Promise<McpTool[]> {
  const res = await fetch(
    `${API_BASE}/api/mcp/servers/${encodeURIComponent(id)}/tools`,
  );
  if (!res.ok) throw new Error(`Failed to list tools: ${res.statusText}`);
  const data = await res.json();
  return data.tools ?? [];
}

export async function listMcpResources(id: string): Promise<McpResource[]> {
  const res = await fetch(
    `${API_BASE}/api/mcp/servers/${encodeURIComponent(id)}/resources`,
  );
  if (!res.ok) throw new Error(`Failed to list resources: ${res.statusText}`);
  const data = await res.json();
  return data.resources ?? [];
}

export async function listMcpPrompts(id: string): Promise<McpPrompt[]> {
  const res = await fetch(
    `${API_BASE}/api/mcp/servers/${encodeURIComponent(id)}/prompts`,
  );
  if (!res.ok) throw new Error(`Failed to list prompts: ${res.statusText}`);
  const data = await res.json();
  return data.prompts ?? [];
}

// ----- Agent Skills -----
//
// Mirrors the `synthesis_engine.skills` substrate. The list endpoint
// honours an optional `?workspace=` filter that applies the inheritance
// chain in `my-projects.yaml`. The detail endpoint returns the full
// SKILL.md body and file list. The run endpoint dispatches the skill's
// first declared tool (or its body prompt) through the agent loop and
// returns a task id to poll.

/** Visibility scope for a skill — universal vs. one-or-more workspaces. */
export interface SkillScope {
  universal: boolean;
  workspaces: string[];
}

/** A tool declared in a skill's frontmatter. */
export interface SkillToolDef {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  script: string | null;
}

/** A file inside a skill's directory tree (compact view). */
export interface SkillFileEntry {
  relative_path: string;
  kind: 'skill_md' | 'reference' | 'script' | 'other';
  is_text: boolean;
  char_count: number;
}

/** Compact view used by GET /api/skills. */
export interface SkillSummary {
  name: string;
  description: string;
  scope: SkillScope;
  source_path: string;
  version: string | null;
  tools: SkillToolDef[];
  tool_count: number;
  reference_count: number;
  script_count: number;
}

/** Full view used by GET /api/skills/{name}. */
export interface SkillDetail extends SkillSummary {
  body: string;
  frontmatter: Record<string, unknown>;
  tool_permissions: Record<string, string>;
  files: SkillFileEntry[];
}

/** Body for POST /api/skills/{name}/run. */
export interface SkillRunRequest {
  workspace: string;
  input?: Record<string, unknown>;
  file?: string;
  model?: string;
}

export interface SkillRunResponse {
  task_id: string;
  status: 'running' | 'done' | 'error';
  skill?: string;
  result?: string;
  error?: string;
  workspace?: string;
}

/**
 * List discovered skills. When ``workspace`` is supplied, the response
 * is filtered through the inheritance chain in my-projects.yaml so the
 * caller only sees skills visible from that workspace plus any
 * universals. When omitted, every discovered skill is returned with its
 * scope tag visible.
 */
export async function listSkills(workspace?: string): Promise<SkillSummary[]> {
  const url = workspace
    ? `${API_BASE}/api/skills?workspace=${encodeURIComponent(workspace)}`
    : `${API_BASE}/api/skills`;
  const res = await fetch(url);
  if (!res.ok) {
    const detail = await safeErrorDetail(res);
    throw new Error(`Failed to list skills: ${detail}`);
  }
  const data = await res.json();
  return data.skills ?? [];
}

/** Fetch the full SKILL.md body and file list for one skill. */
export async function getSkill(
  name: string,
  workspace?: string,
): Promise<SkillDetail> {
  const url = workspace
    ? `${API_BASE}/api/skills/${encodeURIComponent(name)}?workspace=${encodeURIComponent(workspace)}`
    : `${API_BASE}/api/skills/${encodeURIComponent(name)}`;
  const res = await fetch(url);
  if (!res.ok) {
    const detail = await safeErrorDetail(res);
    throw new Error(`Failed to load skill: ${detail}`);
  }
  return res.json();
}

/**
 * Dispatch a skill's first declared tool (or its body prompt) via the
 * agent loop. The response returns a task id the caller polls via
 * ``getSkillRun()``. A 403 is returned when the skill is not visible
 * from the supplied workspace.
 */
export async function runSkill(
  name: string,
  params: SkillRunRequest,
): Promise<SkillRunResponse> {
  const res = await fetch(`${API_BASE}/api/skills/${encodeURIComponent(name)}/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) {
    const detail = await safeErrorDetail(res);
    throw new Error(`Failed to run skill: ${detail}`);
  }
  return res.json();
}

/** Poll the in-process task table for a skill run's terminal state. */
export async function getSkillRun(taskId: string): Promise<SkillRunResponse> {
  const res = await fetch(`${API_BASE}/api/skills/runs/${encodeURIComponent(taskId)}`);
  if (!res.ok) {
    const detail = await safeErrorDetail(res);
    throw new Error(`Failed to fetch skill run: ${detail}`);
  }
  return res.json();
}

/** Pull a structured `detail` from a FastAPI error, falling back to status text. */
async function safeErrorDetail(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (body && typeof body.detail === 'string') return body.detail;
    if (body && body.detail) return JSON.stringify(body.detail);
  } catch {
    /* fall through */
  }
  return res.statusText || `HTTP ${res.status}`;
}

// ----- Cross-workspace policy ------------------------------------------------
//
// Mirrors the `synthesis_engine.policy` substrate. The router exposes the
// loaded RoutingPolicy per workspace, a dry-run cross-workspace boundary
// check, and the recent audit log feed.

export type Confidentiality =
  | 'PUBLIC'
  | 'PERSONAL'
  | 'CLIENT_CONFIDENTIAL'
  | 'AIR_GAPPED';

export type FallbackBehavior = 'deny' | 'downgrade_to_local' | 'warn';

export interface WorkspacePolicy {
  workspace: string;
  workspace_root: string;
  routing_yaml_path: string;
  routing_yaml_exists: boolean;
  confidentiality: Confidentiality;
  allowed_models: string[];
  denied_models: string[];
  local_only: boolean;
  fallback_behavior: FallbackBehavior;
}

export interface CrossWorkspaceBoundary {
  from_workspace: string;
  to_workspace: string;
  allowed: boolean;
  reason: string;
}

export interface ModelRoutingVerdict {
  workspace: string;
  allowed: boolean;
  reason: string;
  fallback_behavior: FallbackBehavior;
  suggested_fallback: string | null;
}

export interface ModelRoutingResult {
  requested_model: string;
  aggregate_allowed: boolean;
  denying_workspace_count: number;
  verdicts: ModelRoutingVerdict[];
}

export interface CrossWorkspaceCheck {
  workspaces: string[];
  allowed: boolean;
  effective_confidentiality: Confidentiality;
  requires_audit: boolean;
  reason: string;
  boundaries: CrossWorkspaceBoundary[];
  policies: Record<
    string,
    {
      confidentiality: Confidentiality;
      fallback_behavior: FallbackBehavior;
      local_only: boolean;
    }
  >;
  model_routing?: ModelRoutingResult;
}

export interface AuditEntry {
  timestamp_iso: string;
  op_type: string;
  workspaces: string[];
  tools: string[];
  model_id: string;
  outcome: string;
  args_summary: string;
  metadata: Record<string, unknown>;
}

export interface AuditRecentResponse {
  entries: AuditEntry[];
  limit: number;
  count: number;
}

export async function getWorkspacePolicy(
  workspace: string,
): Promise<WorkspacePolicy> {
  const res = await fetch(
    `${API_BASE}/api/policy/workspaces/${encodeURIComponent(workspace)}`,
  );
  if (!res.ok) {
    const detail = await safeErrorDetail(res);
    throw new Error(`Failed to load policy for ${workspace}: ${detail}`);
  }
  return res.json();
}

export async function checkCrossWorkspace(
  workspaces: string[],
  options?: { requestedModel?: string },
): Promise<CrossWorkspaceCheck> {
  const params = new URLSearchParams();
  params.set('workspaces', workspaces.join(','));
  if (options?.requestedModel) {
    params.set('requested_model', options.requestedModel);
  }
  const res = await fetch(
    `${API_BASE}/api/policy/cross-workspace-check?${params.toString()}`,
  );
  if (!res.ok) {
    const detail = await safeErrorDetail(res);
    throw new Error(`Cross-workspace check failed: ${detail}`);
  }
  return res.json();
}

export async function getAuditRecent(
  limit: number = 100,
): Promise<AuditRecentResponse> {
  const res = await fetch(
    `${API_BASE}/api/policy/audit/recent?limit=${encodeURIComponent(String(limit))}`,
  );
  if (!res.ok) {
    const detail = await safeErrorDetail(res);
    throw new Error(`Failed to load audit feed: ${detail}`);
  }
  return res.json();
}

// ----- Agent task control (Phase 4) -----
//
// The keyboard shortcuts ⌘B (background) and ⌘. (cancel) target an in-flight
// agent task by id. The server-side router exposes idempotent endpoints; on
// 404 the helpers throw with a clear message so the UI can surface a
// "no active agent run" toast.

export interface TaskControlResponse {
  task_id: string;
  status: string;
}

/**
 * Background the named agent task. The server detaches the task from any
 * SSE/streaming subscriber but continues running it. Idempotent: backgrounding
 * an already-backgrounded task is a no-op.
 */
export async function backgroundAgentTask(taskId: string): Promise<TaskControlResponse> {
  const res = await fetch(
    `${API_BASE}/api/agent/sessions/${encodeURIComponent(taskId)}/background`,
    { method: 'POST' },
  );
  if (!res.ok) {
    const detail = await safeErrorDetail(res);
    throw new Error(`Failed to background task: ${detail}`);
  }
  return res.json();
}

/**
 * Cancel the named agent task. Server moves the task to a cancelled terminal
 * state and stops further tool/model calls. Idempotent.
 */
export async function cancelAgentTask(taskId: string): Promise<TaskControlResponse> {
  const res = await fetch(
    `${API_BASE}/api/agent/sessions/${encodeURIComponent(taskId)}/cancel`,
    { method: 'POST' },
  );
  if (!res.ok) {
    const detail = await safeErrorDetail(res);
    throw new Error(`Failed to cancel task: ${detail}`);
  }
  return res.json();
}
