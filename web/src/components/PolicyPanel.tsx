'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  checkCrossWorkspace,
  getAuditRecent,
  getWorkspacePolicy,
  getWorkspaces,
  type AuditEntry,
  type Confidentiality,
  type CrossWorkspaceCheck,
  type WorkspaceInfo,
  type WorkspacePolicy,
} from '@/lib/api';

/**
 * Cross-workspace policy panel.
 *
 * Lives inside SettingsPanel as a collapsible section, mirroring the
 * pattern McpServersPanel and SkillsPanel established. Surfaces three
 * orthogonal substrate concerns:
 *
 *   1. Per-workspace routing policy (confidentiality chip + tooltip
 *      to the source routing.yaml path).
 *   2. Effective confidentiality for the union of selected workspaces
 *      with a model-routing summary table when a model id is provided.
 *   3. The recent audit-log feed (last N entries) with timestamp, op,
 *      workspaces, model, outcome.
 *
 * The panel is read-only — it does not mutate the substrate. Operators
 * edit routing.yaml directly; the panel observes what the substrate
 * sees.
 */
interface PolicyPanelProps {
  /** The currently active primary workspace, if any. */
  workspace: string | undefined;
  /** The currently selected model id, surfaced to the routing table. */
  model: string | undefined;
  disabled?: boolean;
}

const CONFIDENTIALITY_STYLES: Record<Confidentiality, string> = {
  PUBLIC:
    'bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300',
  PERSONAL:
    'bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300',
  CLIENT_CONFIDENTIAL:
    'bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300',
  AIR_GAPPED:
    'bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300',
};

const FALLBACK_STYLES: Record<string, string> = {
  deny: 'text-red-600 dark:text-red-400',
  downgrade_to_local: 'text-amber-600 dark:text-amber-400',
  warn: 'text-gray-600 dark:text-gray-400',
};

export function PolicyPanel({
  workspace,
  model,
  disabled,
}: PolicyPanelProps) {
  const [workspaces, setWorkspaces] = useState<WorkspaceInfo[]>([]);
  // Seed the multi-select with the active workspace (if any) so the panel
  // renders something useful on first paint.
  const [selected, setSelected] = useState<string[]>(
    workspace ? [workspace] : [],
  );
  const [policies, setPolicies] = useState<Record<string, WorkspacePolicy>>({});
  const [check, setCheck] = useState<CrossWorkspaceCheck | null>(null);
  const [auditEntries, setAuditEntries] = useState<AuditEntry[]>([]);
  const [loadingWorkspaces, setLoadingWorkspaces] = useState(true);
  const [loadingAudit, setLoadingAudit] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // The (selection, model) pair the current `check` answers for. The
  // "refreshing" indicator derives from comparing it against the pair on
  // screen — no loading flag needs to be set from inside an effect.
  const [checkedKey, setCheckedKey] = useState('');
  const wantKey = useMemo(
    () => JSON.stringify([selected, model ?? null]),
    [selected, model],
  );
  const loadingCheck = selected.length > 0 && checkedKey !== wantKey;

  // Whenever the workspace prop changes (e.g., user picks one in the main
  // settings dropdown), include it in the selection if it isn't already.
  // Render-phase adjustment (react.dev "adjusting state when a prop changes").
  const [prevWorkspace, setPrevWorkspace] = useState(workspace);
  if (prevWorkspace !== workspace) {
    setPrevWorkspace(workspace);
    if (workspace && !selected.includes(workspace)) {
      setSelected([workspace, ...selected]);
    }
  }

  // Initial workspace list.
  useEffect(() => {
    let cancelled = false;
    getWorkspaces()
      .then((list) => {
        if (!cancelled) setWorkspaces(list);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : 'Failed to load workspaces');
      })
      .finally(() => {
        if (!cancelled) setLoadingWorkspaces(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Load per-workspace policies whenever the selection changes. An empty
  // selection has nothing to fetch; toggleWorkspace clears the fetched
  // state when it empties the selection.
  useEffect(() => {
    if (selected.length === 0) return;
    let cancelled = false;
    void (async () => {
      const next: Record<string, WorkspacePolicy> = {};
      for (const ws of selected) {
        try {
          next[ws] = await getWorkspacePolicy(ws);
        } catch (e) {
          if (cancelled) return;
          setError(e instanceof Error ? e.message : `Failed: ${ws}`);
        }
      }
      if (!cancelled) setPolicies(next);
    })();
    return () => {
      cancelled = true;
    };
  }, [selected]);

  // Cross-workspace boundary check + model-routing summary.
  useEffect(() => {
    if (selected.length === 0) return;
    let cancelled = false;
    void (async () => {
      try {
        const result = await checkCrossWorkspace(selected, {
          requestedModel: model,
        });
        if (!cancelled) {
          setCheck(result);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : 'Cross-workspace check failed');
          setCheck(null);
        }
      } finally {
        if (!cancelled) setCheckedKey(wantKey);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [selected, model, wantKey]);

  // Audit-log feed (last 100). The Refresh button reuses refreshAudit; the
  // initial load runs inline so unmount cancels the state writes.
  const refreshAudit = useCallback(async () => {
    setLoadingAudit(true);
    try {
      const res = await getAuditRecent(100);
      // Newest-first for the feed (the backend returns natural file
      // order, i.e., oldest-first); reversing here keeps the UI fresh.
      setAuditEntries([...res.entries].reverse());
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Audit fetch failed');
    } finally {
      setLoadingAudit(false);
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const res = await getAuditRecent(100);
        if (!cancelled) setAuditEntries([...res.entries].reverse());
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : 'Audit fetch failed');
        }
      } finally {
        if (!cancelled) setLoadingAudit(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const toggleWorkspace = (dir: string) => {
    const next = selected.includes(dir)
      ? selected.filter((n) => n !== dir)
      : [...selected, dir];
    setSelected(next);
    if (next.length === 0) {
      // Event-driven counterpart of the fetch effects above, which skip
      // empty selections instead of clearing state themselves.
      setPolicies({});
      setCheck(null);
      setCheckedKey('');
    }
  };

  const sortedAvailable = useMemo(
    () => [...workspaces].sort((a, b) => a.name.localeCompare(b.name)),
    [workspaces],
  );

  if (loadingWorkspaces && workspaces.length === 0) {
    return (
      <div className="p-4 animate-pulse">
        <div className="h-8 w-64 bg-gray-200 dark:bg-gray-700 rounded mb-2" />
        <div className="h-4 w-96 bg-gray-200 dark:bg-gray-700 rounded" />
      </div>
    );
  }

  return (
    <div className="p-4 space-y-4">
      {error && (
        <div
          className="text-sm text-red-700 dark:text-red-300 bg-red-50 dark:bg-red-900/30 rounded px-3 py-2"
          role="alert"
        >
          {error}
        </div>
      )}

      {/* Workspace multi-select */}
      <section>
        <h3 className="text-sm font-semibold text-gray-800 dark:text-gray-200 mb-2">
          Active workspaces
        </h3>
        <div className="flex flex-wrap gap-2">
          {sortedAvailable.map((ws) => {
            const checked = selected.includes(ws.dir_name);
            return (
              <button
                key={ws.dir_name}
                type="button"
                disabled={disabled}
                onClick={() => toggleWorkspace(ws.dir_name)}
                className={`px-3 py-1 text-xs rounded-full border transition-colors ${
                  checked
                    ? 'bg-accent-light border-accent text-accent-dark dark:text-accent'
                    : 'border-gray-300 dark:border-gray-600 text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                }`}
                aria-pressed={checked}
              >
                {checked ? '✓ ' : ''}
                {ws.name}
              </button>
            );
          })}
        </div>
      </section>

      {/* Per-workspace policy chips */}
      {selected.length > 0 && (
        <section>
          <h3 className="text-sm font-semibold text-gray-800 dark:text-gray-200 mb-2">
            Per-workspace policy
          </h3>
          <div className="space-y-1">
            {selected.map((ws) => {
              const policy = policies[ws];
              if (!policy) {
                return (
                  <div
                    key={ws}
                    className="text-xs text-gray-500 dark:text-gray-400"
                  >
                    {ws} — loading…
                  </div>
                );
              }
              return (
                <div
                  key={ws}
                  className="flex items-center gap-2 text-xs"
                >
                  <span className="font-medium text-gray-700 dark:text-gray-300 w-40 truncate">
                    {ws}
                  </span>
                  <span
                    className={`px-2 py-0.5 rounded ${
                      CONFIDENTIALITY_STYLES[policy.confidentiality] ?? ''
                    }`}
                    title={`source: ${policy.routing_yaml_path}${
                      policy.routing_yaml_exists ? '' : ' (default — no file)'
                    }`}
                  >
                    {policy.confidentiality}
                  </span>
                  <span
                    className={`px-1 py-0.5 rounded text-[10px] ${
                      FALLBACK_STYLES[policy.fallback_behavior] ?? ''
                    }`}
                    title="fallback_behavior when a model is denied"
                  >
                    fallback: {policy.fallback_behavior}
                  </span>
                  {policy.local_only && (
                    <span
                      className="px-1 py-0.5 rounded text-[10px] bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300"
                      title="local_only=true; only local models permitted"
                    >
                      local_only
                    </span>
                  )}
                  {!policy.routing_yaml_exists && (
                    <span
                      className="text-[10px] text-gray-400"
                      title="No routing.yaml on disk; using default policy"
                    >
                      (default)
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* Aggregate cross-workspace check */}
      {check && (
        <section>
          <h3 className="text-sm font-semibold text-gray-800 dark:text-gray-200 mb-2">
            Effective confidentiality
          </h3>
          <div className="flex items-center gap-2 text-xs">
            <span
              className={`px-2 py-0.5 rounded font-medium ${
                CONFIDENTIALITY_STYLES[check.effective_confidentiality] ?? ''
              }`}
            >
              {check.effective_confidentiality}
            </span>
            {check.allowed ? (
              <span className="text-green-700 dark:text-green-400">
                ✓ allowed
              </span>
            ) : (
              <span className="text-red-700 dark:text-red-400">✗ denied</span>
            )}
            {check.requires_audit && (
              <span
                className="text-amber-700 dark:text-amber-400"
                title="This mix is logged to the cross-workspace audit trail."
              >
                ⓘ audit required
              </span>
            )}
            {loadingCheck && (
              <span className="text-gray-400">refreshing…</span>
            )}
          </div>
          <div className="text-[11px] text-gray-500 dark:text-gray-400 mt-1">
            {check.reason}
          </div>
        </section>
      )}

      {/* Model-routing summary table */}
      {check?.model_routing && (
        <section>
          <h3 className="text-sm font-semibold text-gray-800 dark:text-gray-200 mb-2">
            Model routing — {check.model_routing.requested_model}
          </h3>
          <table className="w-full text-xs">
            <thead className="text-left text-gray-500 dark:text-gray-400">
              <tr>
                <th className="py-1 pr-2 font-normal">Workspace</th>
                <th className="py-1 pr-2 font-normal">Verdict</th>
                <th className="py-1 pr-2 font-normal">Fallback</th>
                <th className="py-1 font-normal">Reason</th>
              </tr>
            </thead>
            <tbody>
              {check.model_routing.verdicts.map((v) => (
                <tr
                  key={v.workspace}
                  className="border-t border-gray-100 dark:border-gray-700"
                >
                  <td className="py-1 pr-2 font-medium text-gray-700 dark:text-gray-300">
                    {v.workspace}
                  </td>
                  <td className="py-1 pr-2">
                    {v.allowed ? (
                      <span className="text-green-700 dark:text-green-400">
                        ✓ allowed
                      </span>
                    ) : (
                      <span className="text-red-700 dark:text-red-400">
                        ✗ denied
                      </span>
                    )}
                  </td>
                  <td
                    className={`py-1 pr-2 ${
                      FALLBACK_STYLES[v.fallback_behavior] ?? ''
                    }`}
                  >
                    {v.fallback_behavior}
                  </td>
                  <td className="py-1 text-gray-600 dark:text-gray-400">
                    {v.reason}
                  </td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr className="border-t border-gray-200 dark:border-gray-600">
                <td className="py-1 pr-2 font-medium text-gray-700 dark:text-gray-300">
                  Aggregate
                </td>
                <td className="py-1 pr-2" colSpan={3}>
                  {check.model_routing.aggregate_allowed ? (
                    <span className="text-green-700 dark:text-green-400">
                      ✓ every workspace allows {check.model_routing.requested_model}
                    </span>
                  ) : (
                    <span className="text-red-700 dark:text-red-400">
                      ✗ {check.model_routing.denying_workspace_count} workspace(s) deny this model
                    </span>
                  )}
                </td>
              </tr>
            </tfoot>
          </table>
        </section>
      )}

      {/* Audit-log feed */}
      <section>
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-sm font-semibold text-gray-800 dark:text-gray-200">
            Audit log (last 100)
          </h3>
          <button
            type="button"
            disabled={disabled || loadingAudit}
            onClick={() => void refreshAudit()}
            className="text-xs px-2 py-1 rounded border border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50"
          >
            {loadingAudit ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
        {auditEntries.length === 0 ? (
          <div className="text-xs text-gray-500 dark:text-gray-400">
            No entries yet. The audit log records every cross-workspace
            run, model call, and confidentiality boundary decision.
          </div>
        ) : (
          <div className="max-h-64 overflow-y-auto border border-gray-200 dark:border-gray-700 rounded">
            <table className="w-full text-[11px]">
              <thead className="sticky top-0 bg-gray-50 dark:bg-gray-800 text-left text-gray-500 dark:text-gray-400">
                <tr>
                  <th className="py-1 px-2 font-normal">Time (UTC)</th>
                  <th className="py-1 px-2 font-normal">Op</th>
                  <th className="py-1 px-2 font-normal">Workspaces</th>
                  <th className="py-1 px-2 font-normal">Model</th>
                  <th className="py-1 px-2 font-normal">Outcome</th>
                </tr>
              </thead>
              <tbody>
                {auditEntries.map((entry, idx) => (
                  <tr
                    key={`${entry.timestamp_iso}-${idx}`}
                    className="border-t border-gray-100 dark:border-gray-700"
                  >
                    <td className="py-1 px-2 font-mono text-gray-600 dark:text-gray-400">
                      {entry.timestamp_iso}
                    </td>
                    <td className="py-1 px-2 text-gray-700 dark:text-gray-300">
                      {entry.op_type}
                    </td>
                    <td className="py-1 px-2 text-gray-700 dark:text-gray-300">
                      {entry.workspaces.join(', ')}
                    </td>
                    <td className="py-1 px-2 font-mono text-gray-600 dark:text-gray-400">
                      {entry.model_id || '—'}
                    </td>
                    <td className="py-1 px-2">
                      {entry.outcome === 'denied' ||
                      entry.outcome === 'error' ? (
                        <span className="text-red-700 dark:text-red-400">
                          {entry.outcome}
                        </span>
                      ) : entry.outcome === 'downgraded' ? (
                        <span className="text-amber-700 dark:text-amber-400">
                          {entry.outcome}
                        </span>
                      ) : (
                        <span className="text-green-700 dark:text-green-400">
                          {entry.outcome}
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
