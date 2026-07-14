'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  getSkill,
  listSkills,
  type SkillDetail,
  type SkillSummary,
} from '@/lib/api';

/**
 * Agent Skills management panel.
 *
 * Lives inside SettingsPanel as a collapsible section, mirroring the
 * pattern McpServersPanel established. Lists skills discovered by the
 * substrate, filters by the active workspace (so workspace-scoped skills
 * surface only where they're meant to), and lets the user expand a row
 * to read the SKILL.md body and inspect the tools declared.
 *
 * The filter has two modes:
 *   - "active workspace" (default) — calls GET /api/skills?workspace=W,
 *     applying the inheritance chain.
 *   - "all skills" — calls GET /api/skills, returning every discovered
 *     skill with its scope tag visible.
 *
 * Switching workspaces externally (via the workspace dropdown in the
 * settings panel) automatically re-fetches the active-workspace view.
 */
interface SkillsPanelProps {
  workspace: string | undefined;
  disabled?: boolean;
}

export function SkillsPanel({ workspace, disabled }: SkillsPanelProps) {
  type FilterMode = 'workspace' | 'all';
  const [filter, setFilter] = useState<FilterMode>('workspace');
  const [skills, setSkills] = useState<SkillSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [details, setDetails] = useState<Record<string, SkillDetail>>({});
  const [detailLoading, setDetailLoading] = useState<string | null>(null);
  // The workspace filter the current `skills` list answers for (null until
  // the first fetch lands). List-loading state derives from comparing it
  // against the filter on screen; manual refreshes track their own flag.
  const [loadedFor, setLoadedFor] = useState<string | null>(null);
  const [manualRefreshing, setManualRefreshing] = useState(false);

  // Active workspace filter resolves to:
  //   - the supplied workspace name when filter === 'workspace' and one
  //     is selected,
  //   - undefined (no filter) when filter === 'all' OR no workspace is
  //     selected and the user wants the workspace view (we cannot send
  //     `?workspace=` empty; the backend would 422 it).
  const activeWorkspace = useMemo(() => {
    if (filter === 'all') return undefined;
    return workspace;
  }, [filter, workspace]);

  const loading = manualRefreshing || loadedFor !== (activeWorkspace ?? '');

  // Manual refresh for the 🔄 button.
  const refresh = useCallback(async () => {
    setManualRefreshing(true);
    try {
      const data = await listSkills(activeWorkspace);
      setSkills(data);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load skills');
    } finally {
      setManualRefreshing(false);
    }
  }, [activeWorkspace]);

  // Fetch the list whenever the effective filter changes (including the
  // initial mount). `loading` above derives from loadedFor, so nothing
  // needs to flip a flag before the fetch starts.
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const data = await listSkills(activeWorkspace);
        if (!cancelled) {
          setSkills(data);
          setError(null);
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : 'Failed to load skills');
        }
      } finally {
        if (!cancelled) setLoadedFor(activeWorkspace ?? '');
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [activeWorkspace]);

  // Reset the expanded state when the underlying list changes so the UI
  // does not show a stale body for a skill that fell out of view.
  // Render-phase adjustment (react.dev "adjusting state when a prop changes").
  const [prevActiveWorkspace, setPrevActiveWorkspace] = useState(activeWorkspace);
  if (prevActiveWorkspace !== activeWorkspace) {
    setPrevActiveWorkspace(activeWorkspace);
    setExpanded(null);
  }

  const handleExpand = async (name: string) => {
    if (expanded === name) {
      setExpanded(null);
      return;
    }
    setExpanded(name);
    if (!details[name]) {
      setDetailLoading(name);
      try {
        const detail = await getSkill(name, activeWorkspace);
        setDetails((prev) => ({ ...prev, [name]: detail }));
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Failed to load skill detail');
      } finally {
        setDetailLoading(null);
      }
    }
  };

  if (loading && skills.length === 0) {
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

      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-xs text-gray-500 dark:text-gray-400">Show:</span>
        <FilterButton
          active={filter === 'workspace'}
          disabled={disabled || !workspace}
          onClick={() => setFilter('workspace')}
          title={
            workspace
              ? `Show skills visible from workspace "${workspace}"`
              : 'Select a workspace to filter'
          }
        >
          {workspace ? `workspace: ${workspace}` : 'workspace (none selected)'}
        </FilterButton>
        <FilterButton
          active={filter === 'all'}
          disabled={disabled}
          onClick={() => setFilter('all')}
          title="Show every discovered skill regardless of scope"
        >
          all skills
        </FilterButton>
        <button
          onClick={() => void refresh()}
          disabled={disabled || loading}
          className="ml-auto px-2 py-1 text-xs rounded border border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50 transition-colors"
          title="Refresh"
        >
          {loading ? '...' : '🔄'}
        </button>
      </div>

      {skills.length === 0 ? (
        <div className="text-sm text-gray-500 dark:text-gray-400 italic">
          {filter === 'workspace' && workspace
            ? `No skills visible from workspace "${workspace}".`
            : 'No skills discovered. Install skills under ~/.synthesis/skills, ~/.claude/skills, or per-workspace skill collections.'}
        </div>
      ) : (
        <ul className="space-y-2">
          {skills.map((s) => (
            <SkillRow
              key={s.name}
              skill={s}
              expanded={expanded === s.name}
              detail={details[s.name]}
              detailLoading={detailLoading === s.name}
              onToggleExpand={() => handleExpand(s.name)}
            />
          ))}
        </ul>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Filter button
// ---------------------------------------------------------------------------

interface FilterButtonProps {
  active: boolean;
  disabled?: boolean;
  onClick: () => void;
  title?: string;
  children: React.ReactNode;
}

function FilterButton({
  active,
  disabled,
  onClick,
  title,
  children,
}: FilterButtonProps) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={`px-2 py-1 text-xs rounded border transition-colors disabled:opacity-50
        ${active
          ? 'bg-accent-light border-accent text-accent-dark dark:text-accent'
          : 'border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700'
        }`}
    >
      {children}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Skill row
// ---------------------------------------------------------------------------

interface SkillRowProps {
  skill: SkillSummary;
  expanded: boolean;
  detail: SkillDetail | undefined;
  detailLoading: boolean;
  onToggleExpand: () => void;
}

function SkillRow({
  skill,
  expanded,
  detail,
  detailLoading,
  onToggleExpand,
}: SkillRowProps) {
  return (
    <li className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900">
      <button
        onClick={onToggleExpand}
        className="w-full flex items-start gap-2 px-3 py-2 text-left hover:bg-gray-50 dark:hover:bg-gray-800/60 transition-colors"
        aria-expanded={expanded}
      >
        <span className="text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 text-xs w-4 mt-0.5">
          {expanded ? '▼' : '▶'}
        </span>

        <div className="flex flex-col min-w-0 flex-1 gap-1">
          <div className="flex items-center gap-2 min-w-0 flex-wrap">
            <span className="text-sm font-semibold text-gray-900 dark:text-gray-100 truncate">
              {skill.name}
            </span>
            <ScopeChip scope={skill.scope} />
            {skill.tool_count > 0 && (
              <span className="text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400">
                {skill.tool_count} tool{skill.tool_count === 1 ? '' : 's'}
              </span>
            )}
          </div>
          {skill.description && (
            <span className="text-xs text-gray-600 dark:text-gray-400 line-clamp-2">
              {skill.description}
            </span>
          )}
          <span className="text-[10px] text-gray-400 dark:text-gray-500 truncate font-mono">
            {skill.source_path}
          </span>
        </div>
      </button>

      {expanded && (
        <SkillBodyPanel
          skill={skill}
          detail={detail}
          loading={detailLoading}
        />
      )}
    </li>
  );
}

function ScopeChip({ scope }: { scope: SkillSummary['scope'] }) {
  if (scope.universal) {
    return (
      <span className="text-[10px] font-medium px-1.5 py-0.5 rounded border bg-accent-light text-accent-dark dark:text-accent border-accent">
        universal
      </span>
    );
  }
  if (scope.workspaces.length === 1) {
    return (
      <span className="text-[10px] font-medium px-1.5 py-0.5 rounded border bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-200 border-gray-300 dark:border-gray-600">
        workspace: {scope.workspaces[0]}
      </span>
    );
  }
  return (
    <span
      className="text-[10px] font-medium px-1.5 py-0.5 rounded border bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-200 border-gray-300 dark:border-gray-600"
      title={scope.workspaces.join(', ')}
    >
      workspaces: {scope.workspaces.length}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Expanded body
// ---------------------------------------------------------------------------

function SkillBodyPanel({
  skill,
  detail,
  loading,
}: {
  skill: SkillSummary;
  detail: SkillDetail | undefined;
  loading: boolean;
}) {
  if (loading) {
    return (
      <div className="px-3 pb-3 border-t border-gray-100 dark:border-gray-800 pt-2 text-xs text-gray-500 dark:text-gray-400">
        Loading SKILL.md…
      </div>
    );
  }
  if (!detail) {
    return (
      <div className="px-3 pb-3 border-t border-gray-100 dark:border-gray-800 pt-2 text-xs text-gray-500 dark:text-gray-400 italic">
        Could not load skill detail.
      </div>
    );
  }
  return (
    <div className="px-3 pb-3 border-t border-gray-100 dark:border-gray-800 pt-2 space-y-3">
      {skill.tool_count > 0 && (
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-1">
            Tools
          </div>
          <ul className="space-y-1">
            {detail.tools.map((t) => (
              <li key={t.name} className="text-xs">
                <span className="font-mono text-gray-900 dark:text-gray-100">
                  {t.name}
                </span>
                {t.description && (
                  <span className="text-gray-500 dark:text-gray-400 ml-2">
                    — {t.description}
                  </span>
                )}
                {t.script && (
                  <span className="text-[10px] text-gray-400 dark:text-gray-500 ml-2 font-mono">
                    [{t.script}]
                  </span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div>
        <div className="text-[11px] font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-1">
          SKILL.md body
        </div>
        <pre className="text-xs whitespace-pre-wrap font-mono bg-gray-50 dark:bg-gray-950 rounded p-2 max-h-96 overflow-y-auto text-gray-800 dark:text-gray-200">
          {detail.body || '(empty body)'}
        </pre>
      </div>

      {detail.files.length > 0 && (
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-wider text-gray-500 dark:text-gray-400 mb-1">
            Files ({detail.files.length})
          </div>
          <ul className="space-y-0.5 max-h-32 overflow-y-auto">
            {detail.files.map((f) => (
              <li key={f.relative_path} className="text-[11px] font-mono text-gray-600 dark:text-gray-400">
                <span className="inline-block w-20 text-gray-400 dark:text-gray-500">
                  [{f.kind}]
                </span>
                {f.relative_path}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
