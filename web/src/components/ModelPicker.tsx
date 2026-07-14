'use client';

import { useState, useEffect, useMemo, useRef } from 'react';
import {
  getModels,
  getProviders,
  getPinnedModels,
  setPinnedModels,
  getRecentModels,
  type ModelInfo,
  type ProviderInfo,
} from '@/lib/api';

interface ModelPickerProps {
  value: string | undefined;
  onChange: (id: string) => void;
  disabled?: boolean;
  /** Imperative open/close handle for ⌘K / external triggers. */
  openSignal?: number;
}

interface OpenChangeProps {
  onOpenChange?: (open: boolean) => void;
}

const TIER_LABEL: Record<string, string> = {
  bulk: 'Fast',
  routine: 'Balanced',
  judgment: 'Powerful',
};

const TIER_BADGE_CLASS: Record<string, string> = {
  bulk: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300',
  routine: 'bg-sky-100 text-sky-700 dark:bg-sky-900/40 dark:text-sky-300',
  judgment: 'bg-violet-100 text-violet-700 dark:bg-violet-900/40 dark:text-violet-300',
};

function formatContext(tokens: number): string {
  if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(0)}M`;
  if (tokens >= 1_000) return `${Math.round(tokens / 1_000)}K`;
  return `${tokens}`;
}

function displayNameFor(m: ModelInfo): string {
  return m.display_name && m.display_name.trim() ? m.display_name : m.name;
}

function matchesSearch(m: ModelInfo, provider: ProviderInfo | undefined, q: string): boolean {
  if (!q) return true;
  const lower = q.toLowerCase();
  const hay = [
    displayNameFor(m),
    m.name,
    m.id,
    provider?.name ?? '',
    m.provider,
    TIER_LABEL[m.tier || 'routine'] || '',
    m.is_local ? 'local' : '',
    m.supports_thinking ? 'thinking' : '',
  ].join(' ').toLowerCase();
  return hay.includes(lower);
}

interface Section {
  key: string;
  label: string;
  badge?: string;
  models: ModelInfo[];
}

function buildSections(
  models: ModelInfo[],
  providers: ProviderInfo[],
  pinnedIds: string[],
  recentIds: string[],
  query: string,
): Section[] {
  const providerById = new Map(providers.map((p) => [p.id, p]));
  const byId = new Map(models.map((m) => [m.id, m]));
  const filter = (m: ModelInfo) => matchesSearch(m, providerById.get(m.provider), query);

  const pinnedModels = pinnedIds
    .map((id) => byId.get(id))
    .filter((m): m is ModelInfo => m !== undefined)
    .filter(filter);
  const pinnedIdSet = new Set(pinnedIds);

  const recentModels = recentIds
    .map((id) => byId.get(id))
    .filter((m): m is ModelInfo => m !== undefined && !pinnedIdSet.has(m.id))
    .slice(0, 5)
    .filter(filter);

  const sections: Section[] = [];
  if (pinnedModels.length > 0) {
    sections.push({ key: '__pinned', label: 'Pinned', badge: '📌', models: pinnedModels });
  }
  if (recentModels.length > 0) {
    sections.push({ key: '__recent', label: 'Recent', badge: '🕐', models: recentModels });
  }

  for (const p of providers) {
    const list = models.filter((m) => m.provider === p.id).filter(filter);
    if (list.length === 0) continue;
    sections.push({
      key: p.id,
      label: p.name,
      badge: p.is_local ? 'Local' : undefined,
      models: list,
    });
  }
  return sections;
}

export function ModelPicker({ value, onChange, disabled, openSignal, onOpenChange }: ModelPickerProps & OpenChangeProps) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [pinned, setPinned] = useState<string[]>([]);
  const [recent, setRecent] = useState<string[]>([]);
  const [focusedIndex, setFocusedIndex] = useState(0);
  const [defaultModelId, setDefaultModelId] = useState<string>('');
  const containerRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // Notify parent of open state changes (used by Chat.tsx for the ⌘K shortcut).
  useEffect(() => {
    onOpenChange?.(open);
  }, [open, onOpenChange]);

  // Open the dropdown with a clean slate (query and keyboard focus reset
  // belong to the open *event*, not to an effect watching `open`).
  const openPicker = () => {
    setQuery('');
    setFocusedIndex(0);
    setOpen(true);
  };

  // External trigger (e.g., ⌘K): increment openSignal to request open.
  // Render-phase adjustment (react.dev "adjusting state when a prop changes").
  const [prevSignal, setPrevSignal] = useState(openSignal);
  if (openSignal !== prevSignal) {
    setPrevSignal(openSignal);
    if (openSignal !== undefined && !disabled) openPicker();
  }

  // Load all data once on mount.
  useEffect(() => {
    let cancelled = false;
    Promise.all([getModels(), getProviders(), getPinnedModels(), getRecentModels()])
      .then(([modelsResp, providersResp, pinnedIds, recentIds]) => {
        if (cancelled) return;
        setModels(modelsResp.models);
        setDefaultModelId(modelsResp.default_model);
        setProviders(providersResp.providers);
        setPinned(pinnedIds);
        setRecent(recentIds);
        // If no model selected yet, adopt the server's default.
        if (!value && modelsResp.default_model) {
          onChange(modelsResp.default_model);
        }
      })
      .catch((err) => {
        console.error('ModelPicker failed to load:', err);
      });
    return () => {
      cancelled = true;
    };
    // Run once; subsequent changes are handled via explicit refreshes below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Refresh recent/pinned when the dropdown opens (cheap, keeps state fresh
  // after a chat has recorded a new entry), and focus the search field.
  useEffect(() => {
    if (!open) return;
    getPinnedModels().then(setPinned).catch(() => {});
    getRecentModels().then(setRecent).catch(() => {});
    // Focus search after the panel mounts.
    queueMicrotask(() => searchRef.current?.focus());
  }, [open]);

  // Close on outside click.
  useEffect(() => {
    if (!open) return;
    const onDocClick = (e: MouseEvent) => {
      if (!containerRef.current?.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', onDocClick);
    return () => document.removeEventListener('mousedown', onDocClick);
  }, [open]);

  const sections = useMemo(
    () => buildSections(models, providers, pinned, recent, query),
    [models, providers, pinned, recent, query],
  );

  // Flatten into a list of (sectionKey, model) for keyboard navigation.
  const flatRows = useMemo(() => {
    const rows: { sectionKey: string; model: ModelInfo }[] = [];
    for (const section of sections) {
      for (const model of section.models) {
        rows.push({ sectionKey: section.key, model });
      }
    }
    return rows;
  }, [sections]);

  // Keep focusedIndex in bounds as filtering changes the row count
  // (render-phase adjustment; the setState restarts render immediately).
  if (focusedIndex >= flatRows.length && flatRows.length > 0) {
    setFocusedIndex(flatRows.length - 1);
  }

  const selectedModel = useMemo(
    () => models.find((m) => m.id === value) ?? models.find((m) => m.id === defaultModelId),
    [models, value, defaultModelId],
  );

  const selectModel = (m: ModelInfo) => {
    if (m.available === false) return; // unavailable: do not select
    onChange(m.id);
    setOpen(false);
  };

  const togglePinned = async (id: string) => {
    const next = pinned.includes(id) ? pinned.filter((x) => x !== id) : [...pinned, id];
    setPinned(next);
    try {
      const persisted = await setPinnedModels(next);
      setPinned(persisted);
    } catch (err) {
      console.error('Failed to persist pinned models:', err);
      // Roll back optimistic update.
      try {
        const live = await getPinnedModels();
        setPinned(live);
      } catch {
        /* ignore */
      }
    }
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      e.preventDefault();
      setOpen(false);
      return;
    }
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setFocusedIndex((i) => Math.min(i + 1, flatRows.length - 1));
      return;
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      setFocusedIndex((i) => Math.max(i - 1, 0));
      return;
    }
    if (e.key === 'Enter') {
      e.preventDefault();
      const row = flatRows[focusedIndex];
      if (row) selectModel(row.model);
    }
  };

  // Scroll the focused row into view.
  useEffect(() => {
    if (!open) return;
    const node = listRef.current?.querySelector<HTMLDivElement>(`[data-row-index="${focusedIndex}"]`);
    node?.scrollIntoView({ block: 'nearest' });
  }, [focusedIndex, open]);

  const triggerLabel = selectedModel ? displayNameFor(selectedModel) : 'Select model';
  const triggerProviderLabel = selectedModel
    ? providers.find((p) => p.id === selectedModel.provider)?.name ?? selectedModel.provider
    : '';

  return (
    <div className="flex flex-col gap-1" ref={containerRef}>
      <label className="text-sm font-medium text-gray-700 dark:text-gray-300 flex items-center justify-between">
        <span>Model</span>
        <kbd className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-400 border border-gray-200 dark:border-gray-700">
          ⌘K
        </kbd>
      </label>
      <button
        type="button"
        onClick={() => {
          if (disabled) return;
          if (open) {
            setOpen(false);
          } else {
            openPicker();
          }
        }}
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
        className="rounded-lg border border-gray-300 dark:border-gray-600
                   bg-white dark:bg-gray-800 px-3 py-2 text-left
                   focus:outline-none focus:ring-2 focus:ring-accent
                   disabled:opacity-50 disabled:cursor-not-allowed
                   flex items-center justify-between gap-2"
      >
        <span className="flex flex-col min-w-0">
          <span className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">{triggerLabel}</span>
          <span className="text-xs text-gray-500 dark:text-gray-400 flex items-center gap-1.5">
            {triggerProviderLabel}
            {selectedModel?.tier && (
              <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${TIER_BADGE_CLASS[selectedModel.tier] || ''}`}>
                {TIER_LABEL[selectedModel.tier] || selectedModel.tier}
              </span>
            )}
            {selectedModel && (
              <span className="text-gray-400">· {formatContext(selectedModel.context_window)}</span>
            )}
            {selectedModel?.supports_thinking && <span title="Supports thinking">🧠</span>}
            {selectedModel?.is_local && <span title="Runs locally">🏠</span>}
          </span>
        </span>
        <span className="text-gray-400 flex-shrink-0">{open ? '▲' : '▼'}</span>
      </button>

      {open && (
        <div
          role="listbox"
          aria-label="Model picker"
          onKeyDown={onKeyDown}
          className="relative z-50"
        >
          <div className="absolute mt-1 w-full min-w-[22rem] max-h-[28rem]
                          rounded-lg border border-gray-200 dark:border-gray-700
                          bg-white dark:bg-gray-800 shadow-lg overflow-hidden flex flex-col">
            <div className="p-2 border-b border-gray-200 dark:border-gray-700">
              <input
                ref={searchRef}
                type="text"
                value={query}
                onChange={(e) => {
                  setQuery(e.target.value);
                  setFocusedIndex(0);
                }}
                onKeyDown={onKeyDown}
                placeholder="Search models…  (try 'opus', 'gemma', 'local')"
                className="w-full rounded border border-gray-300 dark:border-gray-600
                           bg-white dark:bg-gray-900 px-2 py-1.5 text-sm
                           focus:outline-none focus:ring-2 focus:ring-accent"
              />
            </div>

            <div ref={listRef} className="overflow-y-auto flex-1">
              {sections.length === 0 ? (
                <div className="p-4 text-sm text-gray-500 dark:text-gray-400 text-center">
                  No models match &ldquo;{query}&rdquo;.
                </div>
              ) : (
                sections.map((section) => {
                  // Pinned and Recent group headers get the Synthesis
                  // Engineering vermillion accent — a small visual reminder
                  // of Ragbot's lineage. Provider section headers remain
                  // neutral.
                  const isCuratedSection =
                    section.key === '__pinned' || section.key === '__recent';
                  return (
                  <div key={section.key} className="py-1">
                    <div
                      className={`px-3 pt-2 pb-1 text-[10px] font-semibold uppercase tracking-wider flex items-center gap-1.5 ${
                        isCuratedSection
                          ? 'text-[color:var(--accent)] dark:text-[color:var(--accent)]'
                          : 'text-gray-500 dark:text-gray-400'
                      }`}
                    >
                      {section.badge && <span aria-hidden>{section.badge}</span>}
                      <span>{section.label}</span>
                    </div>
                    {section.models.map((model) => {
                      const idx = flatRows.findIndex((r) => r.model.id === model.id && r.sectionKey === section.key);
                      const isFocused = idx === focusedIndex;
                      const isSelected = model.id === value;
                      const isPinned = pinned.includes(model.id);
                      const isAvailable = model.available !== false;
                      const tierLabel = TIER_LABEL[model.tier || 'routine'] || model.tier;
                      return (
                        <div
                          key={`${section.key}/${model.id}`}
                          data-row-index={idx}
                          role="option"
                          aria-selected={isSelected}
                          onMouseEnter={() => setFocusedIndex(idx)}
                          onClick={() => selectModel(model)}
                          // The selected row gets a vermillion left border and
                          // an accent-light wash. Focus highlight (hover/keyboard)
                          // stays neutral so we don't fight the selection cue.
                          style={
                            isSelected
                              ? {
                                  borderLeft: '3px solid var(--accent)',
                                  paddingLeft: 'calc(0.75rem - 3px)',
                                  backgroundColor: 'var(--accent-light)',
                                }
                              : undefined
                          }
                          className={`flex items-center gap-2 px-3 py-1.5 text-sm cursor-pointer
                                      ${isFocused && !isSelected ? 'bg-gray-100 dark:bg-gray-700/40' : ''}
                                      ${!isAvailable ? 'opacity-60 cursor-not-allowed' : ''}
                                      ${isSelected ? 'font-semibold text-[color:var(--accent-dark)] dark:text-[color:var(--accent)]' : 'text-gray-900 dark:text-gray-100'}`}
                        >
                          <span className="w-4 text-center" aria-hidden>
                            {isSelected ? '●' : '○'}
                          </span>
                          <span className="flex-1 truncate">{displayNameFor(model)}</span>
                          {tierLabel && (
                            <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${TIER_BADGE_CLASS[model.tier || 'routine'] || ''}`}>
                              {tierLabel}
                            </span>
                          )}
                          <span className="text-[11px] text-gray-500 dark:text-gray-400 tabular-nums w-12 text-right">
                            {formatContext(model.context_window)}
                          </span>
                          <span className="w-4 text-center" title={model.supports_thinking ? 'Supports thinking' : ''}>
                            {model.supports_thinking ? '🧠' : ''}
                          </span>
                          <span className="w-4 text-center" title={model.is_local ? 'Runs locally' : ''}>
                            {model.is_local ? '🏠' : ''}
                          </span>
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              togglePinned(model.id);
                            }}
                            title={isPinned ? 'Unpin' : 'Pin'}
                            className={`w-5 text-center text-xs rounded hover:bg-gray-200 dark:hover:bg-gray-700
                                        ${isPinned ? 'opacity-100' : 'opacity-40 hover:opacity-100'}`}
                          >
                            {isPinned ? '📌' : '📍'}
                          </button>
                          {!isAvailable && (
                            <span className="text-[10px] text-gray-500 ml-1" title="No API key configured">
                              🔒 No key
                            </span>
                          )}
                        </div>
                      );
                    })}
                  </div>
                  );
                })
              )}
            </div>

            <div className="px-3 py-1.5 text-[10px] text-gray-500 dark:text-gray-400 border-t border-gray-200 dark:border-gray-700 flex justify-between">
              <span>↑↓ navigate · Enter select · Esc close</span>
              <span>{flatRows.length} model{flatRows.length === 1 ? '' : 's'}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
