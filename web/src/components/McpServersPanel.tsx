'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  addMcpServer,
  listMcpPrompts,
  listMcpResources,
  listMcpServers,
  listMcpTools,
  removeMcpServer,
  startMcpOAuth,
  toggleMcpServer,
  type McpAuthMode,
  type McpPrompt,
  type McpResource,
  type McpServer,
  type McpServerInput,
  type McpTool,
  type McpTransport,
} from '@/lib/api';

/**
 * MCP Servers management panel.
 *
 * Lives inside SettingsPanel as a collapsible section. Lists configured
 * servers from `~/.synthesis/mcp.yaml`, surfaces their live connection
 * state from the in-process registry, and lets the user add, toggle,
 * remove, and (for remote servers) re-authorize via OAuth.
 */
export function McpServersPanel({ disabled }: { disabled?: boolean }) {
  const [servers, setServers] = useState<McpServer[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [showAddForm, setShowAddForm] = useState(false);
  const [busyId, setBusyId] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const data = await listMcpServers();
      setServers(data);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load MCP servers');
    } finally {
      setLoading(false);
    }
  }, []);

  // Initial load runs inline (not via refresh) so unmount cancels the state
  // writes; refresh() serves the event handlers below.
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const data = await listMcpServers();
        if (!cancelled) {
          setServers(data);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : 'Failed to load MCP servers');
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const handleToggle = async (id: string) => {
    setBusyId(id);
    try {
      await toggleMcpServer(id);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Toggle failed');
    } finally {
      setBusyId(null);
    }
  };

  const handleRemove = async (id: string) => {
    if (!confirm(`Remove MCP server "${id}"?`)) return;
    setBusyId(id);
    try {
      await removeMcpServer(id);
      if (expandedId === id) setExpandedId(null);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Remove failed');
    } finally {
      setBusyId(null);
    }
  };

  const handleOAuth = async (id: string) => {
    setBusyId(id);
    try {
      const r = await startMcpOAuth(id);
      if (!r.ok && r.error) setError(r.error);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'OAuth failed');
    } finally {
      setBusyId(null);
    }
  };

  const handleAdd = async (server: McpServerInput) => {
    try {
      await addMcpServer(server);
      setShowAddForm(false);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Add failed');
      throw e;
    }
  };

  if (loading) {
    return (
      <div className="p-4 animate-pulse">
        <div className="h-8 w-64 bg-gray-200 dark:bg-gray-700 rounded mb-2"></div>
        <div className="h-12 w-full bg-gray-200 dark:bg-gray-700 rounded"></div>
      </div>
    );
  }

  return (
    <div className="px-4 py-3 space-y-3">
      {error && (
        <div className="text-sm text-red-700 dark:text-red-300 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg px-3 py-2 flex items-start justify-between gap-2">
          <span className="break-words">{error}</span>
          <button
            onClick={() => setError(null)}
            className="text-red-500 hover:text-red-700 flex-shrink-0"
            aria-label="Dismiss error"
          >
            ✕
          </button>
        </div>
      )}

      {servers.length === 0 ? (
        <div className="text-sm text-gray-500 dark:text-gray-400 italic">
          No MCP servers configured. Add one below to extend Ragbot with external tools.
        </div>
      ) : (
        <ul className="space-y-2">
          {servers.map((s) => (
            <McpServerRow
              key={s.id}
              server={s}
              expanded={expandedId === s.id}
              busy={busyId === s.id}
              disabled={disabled}
              onToggleExpand={() => setExpandedId(expandedId === s.id ? null : s.id)}
              onToggleEnabled={() => handleToggle(s.id)}
              onRemove={() => handleRemove(s.id)}
              onOAuth={() => handleOAuth(s.id)}
            />
          ))}
        </ul>
      )}

      <div className="flex justify-end">
        {showAddForm ? null : (
          <button
            onClick={() => setShowAddForm(true)}
            disabled={disabled}
            className="px-3 py-1.5 text-sm rounded-lg bg-accent text-white
                       hover:bg-accent-dark disabled:opacity-50
                       transition-colors"
          >
            + Add server
          </button>
        )}
      </div>

      {showAddForm && (
        <AddMcpServerForm
          onCancel={() => setShowAddForm(false)}
          onSubmit={handleAdd}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Server row
// ---------------------------------------------------------------------------

interface McpServerRowProps {
  server: McpServer;
  expanded: boolean;
  busy: boolean;
  disabled?: boolean;
  onToggleExpand: () => void;
  onToggleEnabled: () => void;
  onRemove: () => void;
  onOAuth: () => void;
}

function McpServerRow({
  server,
  expanded,
  busy,
  disabled,
  onToggleExpand,
  onToggleEnabled,
  onRemove,
  onOAuth,
}: McpServerRowProps) {
  const isRemote = server.transport === 'http' || server.transport === 'sse';
  const usesOAuth = isRemote && server.auth?.mode === 'oauth';
  const target = server.transport === 'stdio' ? server.command : server.url;

  return (
    <li className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900">
      <div className="flex items-center gap-2 px-3 py-2">
        <button
          onClick={onToggleExpand}
          className="text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 text-xs w-4"
          aria-label={expanded ? 'Collapse' : 'Expand'}
        >
          {expanded ? '▼' : '▶'}
        </button>

        <div className="flex flex-col min-w-0 flex-1">
          <div className="flex items-center gap-2 min-w-0">
            <span className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
              {server.name}
            </span>
            <ConnectionStatePill state={server.connection_state} />
            <span className="text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400">
              {server.transport}
            </span>
          </div>
          {target && (
            <span className="text-xs text-gray-500 dark:text-gray-400 truncate font-mono">
              {target}
            </span>
          )}
          {server.last_error && server.connection_state === 'error' && (
            <span className="text-xs text-red-600 dark:text-red-400 truncate" title={server.last_error}>
              {server.last_error}
            </span>
          )}
        </div>

        <div className="flex items-center gap-1 flex-shrink-0">
          {usesOAuth && (
            <button
              onClick={onOAuth}
              disabled={busy || disabled}
              className="px-2 py-1 text-xs rounded border border-accent text-accent
                         hover:bg-accent hover:text-white
                         disabled:opacity-50 transition-colors"
              title="Authorize via OAuth in your browser"
            >
              🔐 Authorize
            </button>
          )}
          <button
            onClick={onToggleEnabled}
            disabled={busy || disabled}
            className={`px-2 py-1 text-xs rounded border transition-colors disabled:opacity-50
              ${server.enabled
                ? 'border-accent bg-accent-light text-accent-dark dark:text-accent'
                : 'border-gray-300 dark:border-gray-600 text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
              }`}
            title={server.enabled ? 'Disable server' : 'Enable server'}
          >
            {server.enabled ? 'Enabled' : 'Disabled'}
          </button>
          <button
            onClick={onRemove}
            disabled={busy || disabled}
            className="px-2 py-1 text-xs rounded border border-gray-300 dark:border-gray-600
                       text-gray-500 hover:bg-red-50 hover:border-red-300 hover:text-red-600
                       dark:hover:bg-red-900/20 dark:hover:text-red-400
                       disabled:opacity-50 transition-colors"
            title="Remove server"
          >
            🗑️
          </button>
        </div>
      </div>

      {expanded && <McpCapabilitiesPanel server={server} />}
    </li>
  );
}

function ConnectionStatePill({ state }: { state: McpServer['connection_state'] }) {
  // Connected uses the vermillion accent so the eye lands on the success
  // state first; other states stay neutral or red.
  const map: Record<NonNullable<McpServer['connection_state']>, { label: string; cls: string }> = {
    connected: {
      label: 'Connected',
      cls: 'bg-accent-light text-accent-dark dark:text-accent border-accent',
    },
    connecting: {
      label: 'Connecting',
      cls: 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 border-gray-300 dark:border-gray-600 animate-pulse',
    },
    disconnected: {
      label: 'Disconnected',
      cls: 'bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-400 border-gray-300 dark:border-gray-600',
    },
    error: {
      label: 'Error',
      cls: 'bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300 border-red-300 dark:border-red-800',
    },
  };
  const v = map[state ?? 'disconnected'];
  return (
    <span
      className={`text-[10px] font-medium px-1.5 py-0.5 rounded border ${v.cls}`}
    >
      {v.label}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Per-server capabilities (tools / resources / prompts)
// ---------------------------------------------------------------------------

function McpCapabilitiesPanel({ server }: { server: McpServer }) {
  const [section, setSection] = useState<'tools' | 'resources' | 'prompts'>('tools');
  const [tools, setTools] = useState<McpTool[] | null>(null);
  const [resources, setResources] = useState<McpResource[] | null>(null);
  const [prompts, setPrompts] = useState<McpPrompt[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const isConnected = server.connection_state === 'connected';

  // Lazy-load when the active section is empty and the server is connected.
  useEffect(() => {
    if (!isConnected) return;
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setLoadError(null);
      try {
        if (section === 'tools' && tools === null) {
          const data = await listMcpTools(server.id);
          if (!cancelled) setTools(data);
        } else if (section === 'resources' && resources === null) {
          const data = await listMcpResources(server.id);
          if (!cancelled) setResources(data);
        } else if (section === 'prompts' && prompts === null) {
          const data = await listMcpPrompts(server.id);
          if (!cancelled) setPrompts(data);
        }
      } catch (e) {
        if (!cancelled) {
          setLoadError(e instanceof Error ? e.message : 'Load failed');
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, [section, isConnected, server.id, tools, resources, prompts]);

  if (!isConnected) {
    return (
      <div className="px-3 pb-3 text-xs text-gray-500 dark:text-gray-400 italic">
        Connect to the server to inspect its tools, resources, and prompts.
      </div>
    );
  }

  return (
    <div className="px-3 pb-3 border-t border-gray-100 dark:border-gray-800">
      <div className="flex gap-1 mt-2 mb-2">
        {(['tools', 'resources', 'prompts'] as const).map((s) => {
          const count =
            s === 'tools'
              ? (tools?.length ?? server.tool_count)
              : s === 'resources'
                ? (resources?.length ?? server.resource_count)
                : (prompts?.length ?? server.prompt_count);
          return (
            <button
              key={s}
              onClick={() => setSection(s)}
              className={`px-2 py-1 text-xs rounded transition-colors capitalize
                ${section === s
                  ? 'bg-accent-light text-accent-dark dark:text-accent border border-accent'
                  : 'border border-transparent text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800'
                }`}
            >
              {s}
              {typeof count === 'number' && (
                <span className="ml-1 text-[10px] opacity-70">({count})</span>
              )}
            </button>
          );
        })}
      </div>

      {loadError && (
        <div className="text-xs text-red-600 dark:text-red-400 py-1">{loadError}</div>
      )}
      {loading && (
        <div className="text-xs text-gray-500 dark:text-gray-400 py-1">Loading…</div>
      )}

      {section === 'tools' && tools && <ToolsList tools={tools} />}
      {section === 'resources' && resources && <ResourcesList resources={resources} />}
      {section === 'prompts' && prompts && <PromptsList prompts={prompts} />}
    </div>
  );
}

function ToolsList({ tools }: { tools: McpTool[] }) {
  if (tools.length === 0) {
    return <div className="text-xs text-gray-500 italic">No tools advertised.</div>;
  }
  return (
    <ul className="space-y-1 max-h-48 overflow-y-auto">
      {tools.map((t) => (
        <li key={t.name} className="text-xs">
          <span className="font-mono text-gray-900 dark:text-gray-100">{t.name}</span>
          {t.description && (
            <span className="text-gray-500 dark:text-gray-400 ml-2">— {t.description}</span>
          )}
        </li>
      ))}
    </ul>
  );
}

function ResourcesList({ resources }: { resources: McpResource[] }) {
  if (resources.length === 0) {
    return <div className="text-xs text-gray-500 italic">No resources advertised.</div>;
  }
  return (
    <ul className="space-y-1 max-h-48 overflow-y-auto">
      {resources.map((r) => (
        <li key={r.uri} className="text-xs">
          <span className="font-mono text-gray-900 dark:text-gray-100 break-all">{r.uri}</span>
          {r.name && <span className="text-gray-500 dark:text-gray-400 ml-2">{r.name}</span>}
          {r.mime_type && (
            <span className="text-gray-400 dark:text-gray-500 ml-1">[{r.mime_type}]</span>
          )}
        </li>
      ))}
    </ul>
  );
}

function PromptsList({ prompts }: { prompts: McpPrompt[] }) {
  if (prompts.length === 0) {
    return <div className="text-xs text-gray-500 italic">No prompts advertised.</div>;
  }
  return (
    <ul className="space-y-1 max-h-48 overflow-y-auto">
      {prompts.map((p) => (
        <li key={p.name} className="text-xs">
          <span className="font-mono text-gray-900 dark:text-gray-100">{p.name}</span>
          {p.description && (
            <span className="text-gray-500 dark:text-gray-400 ml-2">— {p.description}</span>
          )}
        </li>
      ))}
    </ul>
  );
}

// ---------------------------------------------------------------------------
// Add server form
// ---------------------------------------------------------------------------

interface AddMcpServerFormProps {
  onCancel: () => void;
  onSubmit: (server: McpServerInput) => Promise<void>;
}

function AddMcpServerForm({ onCancel, onSubmit }: AddMcpServerFormProps) {
  const [id, setId] = useState('');
  const [name, setName] = useState('');
  const [transport, setTransport] = useState<McpTransport>('stdio');
  const [command, setCommand] = useState('');
  const [argsRaw, setArgsRaw] = useState('');
  const [url, setUrl] = useState('');
  const [authMode, setAuthMode] = useState<McpAuthMode>('none');
  const [bearerToken, setBearerToken] = useState('');
  const [clientIdMetadataUrl, setClientIdMetadataUrl] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);

  const isStdio = transport === 'stdio';

  const validId = useMemo(() => /^[A-Za-z0-9._-]+$/.test(id), [id]);
  const canSubmit = useMemo(() => {
    if (!id.trim() || !name.trim() || !validId) return false;
    if (isStdio) return command.trim().length > 0;
    return url.trim().length > 0;
  }, [id, name, validId, isStdio, command, url]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSubmit) return;
    setSubmitting(true);
    setFormError(null);
    try {
      const args = argsRaw.trim()
        ? argsRaw.split(/\s+/).filter((s) => s.length > 0)
        : [];
      const payload: McpServerInput = {
        id: id.trim(),
        name: name.trim(),
        transport,
        enabled: true,
        ...(isStdio
          ? { command: command.trim(), args }
          : { url: url.trim() }),
        ...(isStdio
          ? {}
          : {
              auth: {
                mode: authMode,
                ...(authMode === 'bearer' && bearerToken
                  ? { token: bearerToken }
                  : {}),
                ...(authMode === 'oauth' && clientIdMetadataUrl
                  ? { client_id_metadata_url: clientIdMetadataUrl }
                  : {}),
              },
            }),
      };
      await onSubmit(payload);
    } catch (e) {
      setFormError(e instanceof Error ? e.message : 'Add failed');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="rounded-lg border border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 p-3 space-y-3"
    >
      <div className="text-sm font-medium text-gray-700 dark:text-gray-300">Add MCP server</div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <Field label="ID" hint={validId || !id ? 'alphanumeric, -, _, .' : 'invalid characters'}>
          <input
            value={id}
            onChange={(e) => setId(e.target.value)}
            placeholder="fs-local"
            className="w-full rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-accent"
            required
          />
        </Field>
        <Field label="Name">
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Local Filesystem"
            className="w-full rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-accent"
            required
          />
        </Field>
      </div>

      <Field label="Transport">
        <div className="flex gap-1">
          {(['stdio', 'http', 'sse'] as const).map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setTransport(t)}
              className={`flex-1 px-2 py-1.5 text-xs rounded border transition-colors capitalize
                ${transport === t
                  ? 'bg-accent-light border-accent text-accent-dark dark:text-accent'
                  : 'border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700'
                }`}
            >
              {t}
            </button>
          ))}
        </div>
      </Field>

      {isStdio ? (
        <>
          <Field label="Command">
            <input
              value={command}
              onChange={(e) => setCommand(e.target.value)}
              placeholder="npx"
              className="w-full rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-accent"
              required
            />
          </Field>
          <Field label="Args" hint="space-separated">
            <input
              value={argsRaw}
              onChange={(e) => setArgsRaw(e.target.value)}
              placeholder="-y @modelcontextprotocol/server-filesystem /path"
              className="w-full rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-accent"
            />
          </Field>
        </>
      ) : (
        <>
          <Field label="URL">
            <input
              type="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://mcp.example.com/v1"
              className="w-full rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-accent"
              required
            />
          </Field>
          <Field label="Auth">
            <select
              value={authMode}
              onChange={(e) => setAuthMode(e.target.value as McpAuthMode)}
              className="w-full rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-accent"
            >
              <option value="none">None</option>
              <option value="oauth">OAuth 2.1</option>
              <option value="bearer">Bearer token</option>
            </select>
          </Field>
          {authMode === 'bearer' && (
            <Field label="Bearer token">
              <input
                type="password"
                value={bearerToken}
                onChange={(e) => setBearerToken(e.target.value)}
                placeholder="ya29.…"
                className="w-full rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-accent"
              />
            </Field>
          )}
          {authMode === 'oauth' && (
            <Field
              label="Client ID Metadata URL"
              hint="optional CIMD document; leave blank to use Dynamic Client Registration"
            >
              <input
                type="url"
                value={clientIdMetadataUrl}
                onChange={(e) => setClientIdMetadataUrl(e.target.value)}
                placeholder="https://app.example.com/mcp/client.json"
                className="w-full rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 px-2 py-1 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-accent"
              />
            </Field>
          )}
        </>
      )}

      {formError && (
        <div className="text-xs text-red-600 dark:text-red-400">{formError}</div>
      )}

      <div className="flex justify-end gap-2">
        <button
          type="button"
          onClick={onCancel}
          disabled={submitting}
          className="px-3 py-1.5 text-sm rounded-lg border border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 transition-colors"
        >
          Cancel
        </button>
        <button
          type="submit"
          disabled={!canSubmit || submitting}
          className="px-3 py-1.5 text-sm rounded-lg bg-accent text-white hover:bg-accent-dark disabled:opacity-50 transition-colors"
        >
          {submitting ? 'Adding…' : 'Add server'}
        </button>
      </div>
    </form>
  );
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-xs font-medium text-gray-600 dark:text-gray-400 flex items-center justify-between">
        {label}
        {hint && <span className="text-[10px] text-gray-400 dark:text-gray-500">{hint}</span>}
      </span>
      {children}
    </label>
  );
}
