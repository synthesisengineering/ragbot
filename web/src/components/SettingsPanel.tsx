'use client';

import { useState, useEffect } from 'react';
import {
  getModels,
  getWorkspaces,
  getIndexStatus,
  indexWorkspace,
  getConfig,
  getProviders,
  getTemperatureSettings,
  getKeysStatus,
  type ModelInfo,
  type WorkspaceInfo,
  type IndexStatus,
  type ProviderInfo,
  type KeysStatusResponse,
  type KeyStatus,
  type ThinkingEffort,
} from '@/lib/api';
import { ModelPicker } from './ModelPicker';
import { McpServersPanel } from './McpServersPanel';
import { PolicyPanel } from './PolicyPanel';
import { SkillsPanel } from './SkillsPanel';

interface SettingsPanelProps {
  workspace: string | undefined;
  onWorkspaceChange: (workspace: string | undefined) => void;
  model: string | undefined;
  onModelChange: (model: string | undefined) => void;
  temperature: number;
  onTemperatureChange: (temp: number) => void;
  useRag: boolean;
  onUseRagChange: (useRag: boolean) => void;
  ragMaxTokens: number;
  onRagMaxTokensChange: (tokens: number) => void;
  maxTokens: number;
  onMaxTokensChange: (tokens: number) => void;
  conversationTurns: number;
  conversationTokens: number;
  onClearChat: () => void;
  disabled?: boolean;
  // v3 reasoning + cross-workspace controls (optional — caller may omit
  // for backwards compatibility; the panel hides them gracefully).
  thinkingEffort?: ThinkingEffort;
  onThinkingEffortChange?: (effort: ThinkingEffort | undefined) => void;
  includeSkills?: boolean;
  onIncludeSkillsChange?: (include: boolean) => void;
  // External open-the-picker signal (incremented by Chat.tsx on ⌘K).
  openModelPickerSignal?: number;
}

const MAX_TOKEN_OPTIONS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536];

export function SettingsPanel({
  workspace,
  onWorkspaceChange,
  model,
  onModelChange,
  temperature,
  onTemperatureChange,
  useRag,
  onUseRagChange,
  ragMaxTokens,
  onRagMaxTokensChange,
  maxTokens,
  onMaxTokensChange,
  conversationTurns,
  conversationTokens,
  onClearChat,
  disabled,
  thinkingEffort,
  onThinkingEffortChange,
  includeSkills,
  onIncludeSkillsChange,
  openModelPickerSignal,
}: SettingsPanelProps) {
  const [workspaces, setWorkspaces] = useState<WorkspaceInfo[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [temperaturePresets, setTemperaturePresets] = useState<Record<string, number>>({});
  const [keysStatus, setKeysStatus] = useState<KeysStatusResponse>({});
  const [indexStatus, setIndexStatus] = useState<IndexStatus | null>(null);
  const [indexing, setIndexing] = useState(false);
  const [loading, setLoading] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showMcp, setShowMcp] = useState(false);
  const [showSkills, setShowSkills] = useState(false);
  const [showPolicy, setShowPolicy] = useState(false);

  // Key source overrides - user can override workspace keys with default keys
  const [keyOverrides, setKeyOverrides] = useState<Record<string, 'auto' | 'default'>>({});

  // Load workspaces, models, providers, and config from API (all from engines.yaml)
  useEffect(() => {
    async function load() {
      try {
        const [wsData, modelData, configData, providersData, tempSettings] = await Promise.all([
          getWorkspaces(),
          getModels(),
          getConfig(),
          getProviders(),
          getTemperatureSettings(),
        ]);
        setWorkspaces(wsData);
        setModels(modelData.models);
        setProviders(providersData.providers);
        setTemperaturePresets(tempSettings);

        // Set default workspace if none selected and config has one
        const defaultWorkspaceName = configData.default_workspace;
        if (!workspace && defaultWorkspaceName) {
          const defaultWs = wsData.find(
            (w) =>
              w.dir_name === defaultWorkspaceName ||
              w.name.toLowerCase() === defaultWorkspaceName.toLowerCase(),
          );
          if (defaultWs) {
            onWorkspaceChange(defaultWs.dir_name);
          }
        }

        // Set default model if none selected. ModelPicker will also adopt
        // the server default on its own load, but doing it here keeps the
        // Thinking control rendering decision deterministic on first paint.
        if (!model && modelData.default_model) {
          onModelChange(modelData.default_model);
        }
      } catch (e) {
        console.error('Failed to load settings data:', e);
      } finally {
        setLoading(false);
      }
    }
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Reset per-workspace UI state the moment the workspace changes.
  // Render-phase adjustment (react.dev "adjusting state when a prop changes").
  const [prevWorkspace, setPrevWorkspace] = useState(workspace);
  if (prevWorkspace !== workspace) {
    setPrevWorkspace(workspace);
    setKeyOverrides({});
    setIndexStatus(null);
  }

  // Load keys status when workspace changes
  useEffect(() => {
    let cancelled = false;
    getKeysStatus(workspace)
      .then((status) => {
        if (!cancelled) setKeysStatus(status);
      })
      .catch(() => {
        if (!cancelled) setKeysStatus({});
      });
    return () => {
      cancelled = true;
    };
  }, [workspace]);

  // Load index status when workspace changes (no workspace: the reset
  // above already cleared it).
  useEffect(() => {
    if (!workspace) return;
    let cancelled = false;
    getIndexStatus(workspace)
      .then((status) => {
        if (!cancelled) setIndexStatus(status);
      })
      .catch(() => {
        if (!cancelled) setIndexStatus(null);
      });
    return () => {
      cancelled = true;
    };
  }, [workspace]);

  // Get effective key source for a provider (considering overrides)
  const getEffectiveKeySource = (providerId: string): KeyStatus['source'] => {
    const status = keysStatus[providerId];
    if (!status) return null;

    const override = keyOverrides[providerId];
    if (override === 'default' && status.has_default_key) {
      return 'default';
    }
    return status.source;
  };

  const handleIndex = async () => {
    if (!workspace) return;
    setIndexing(true);
    try {
      await indexWorkspace(workspace, true);
      const status = await getIndexStatus(workspace);
      setIndexStatus(status);
    } catch (e) {
      console.error('Indexing failed:', e);
    } finally {
      setIndexing(false);
    }
  };

  const selectedWorkspace = workspaces.find((w) => w.dir_name === workspace);
  const selectedModel = models.find((m) => m.id === model);
  const showThinkingControl =
    onThinkingEffortChange !== undefined && Boolean(selectedModel?.supports_thinking);

  if (loading) {
    return (
      <div className="p-4 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
        <div className="animate-pulse flex gap-4">
          <div className="h-10 w-48 bg-gray-200 dark:bg-gray-700 rounded"></div>
          <div className="h-10 w-48 bg-gray-200 dark:bg-gray-700 rounded"></div>
          <div className="h-10 w-48 bg-gray-200 dark:bg-gray-700 rounded"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
      {/* Settings header — identifies Ragbot's lineage. */}
      <div className="px-4 pt-3 pb-1 border-b border-[color:var(--accent-light)] dark:border-gray-700">
        <div className="max-w-6xl mx-auto flex items-baseline gap-2">
          <span className="text-sm font-semibold text-gray-700 dark:text-gray-300">
            Settings
          </span>
          <span className="text-xs text-gray-500 dark:text-gray-400">
            · Ragbot — by{' '}
            <span className="text-[color:var(--accent)] font-medium">
              Synthesis Engineering
            </span>
          </span>
        </div>
      </div>

      {/* Main Settings Row */}
      <div className="px-4 py-3">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 max-w-6xl mx-auto">
          {/* Workspace */}
          <div className="flex flex-col gap-1">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Workspace
            </label>
            <select
              value={workspace || ''}
              onChange={(e) => onWorkspaceChange(e.target.value || undefined)}
              disabled={disabled}
              aria-label="Workspace"
              data-shortcut-target="workspace"
              className="rounded-lg border border-gray-300 dark:border-gray-600
                         bg-white dark:bg-gray-800 px-3 py-2 text-sm
                         focus:outline-none focus:ring-2 focus:ring-accent"
            >
              <option value="">Select workspace...</option>
              {workspaces.map((ws) => (
                <option key={ws.dir_name} value={ws.dir_name}>
                  {ws.name}
                </option>
              ))}
            </select>
            {selectedWorkspace && (
              <div className="text-xs text-gray-500 flex items-center gap-1">
                {selectedWorkspace.has_datasets ? '✅' : '⚠️'}
                <span>{selectedWorkspace.has_datasets ? 'Has datasets' : 'No datasets'}</span>
                {selectedWorkspace.has_instructions && <span className="text-green-600">• Instructions</span>}
              </div>
            )}
          </div>

          {/* Model + Thinking — single rich picker; Thinking renders only when supported. */}
          <div className="flex flex-col gap-2">
            <ModelPicker
              value={model}
              onChange={(id) => onModelChange(id)}
              disabled={disabled}
              openSignal={openModelPickerSignal}
            />
            {showThinkingControl && (
              <div className="flex items-center gap-2">
                <span className="text-xs font-medium text-gray-700 dark:text-gray-300">🧠 Thinking</span>
                <select
                  value={thinkingEffort ?? 'auto'}
                  onChange={(e) => {
                    const v = e.target.value as ThinkingEffort;
                    onThinkingEffortChange?.(v === 'auto' ? undefined : v);
                  }}
                  disabled={disabled}
                  className="rounded border border-gray-300 dark:border-gray-600
                             bg-white dark:bg-gray-800 px-2 py-1 text-xs flex-1"
                  title="Reasoning effort. Defaults: flagship → medium, others → off."
                >
                  <option value="auto">auto</option>
                  <option value="off">off</option>
                  <option value="minimal">minimal</option>
                  <option value="low">low</option>
                  <option value="medium">medium</option>
                  <option value="high">high</option>
                </select>
              </div>
            )}
          </div>

          {/* Temperature - presets from engines.yaml */}
          <div className="flex flex-col gap-1">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Creativity
            </label>
            <div className="flex gap-1">
              {Object.entries(temperaturePresets).map(([name, val]) => (
                <button
                  key={name}
                  onClick={() => onTemperatureChange(val)}
                  disabled={disabled}
                  className={`flex-1 px-2 py-2 text-xs rounded-lg border transition-colors capitalize
                    ${Math.abs(temperature - val) < 0.05
                      ? 'bg-accent-light border-accent text-accent-dark dark:text-accent'
                      : 'border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700'
                    }`}
                >
                  {name}
                </button>
              ))}
            </div>
            <div className="text-xs text-gray-500 text-center">
              Temperature: {temperature.toFixed(2)}
            </div>
          </div>

          {/* Conversation Stats */}
          <div className="flex flex-col gap-1">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Conversation
            </label>
            <div className="flex gap-2 items-center">
              <div className="flex-1 bg-white dark:bg-gray-900 rounded-lg border border-gray-300 dark:border-gray-600 px-3 py-2 text-center">
                <div className="text-lg font-semibold text-gray-900 dark:text-white">{conversationTurns}</div>
                <div className="text-xs text-gray-500">turns</div>
              </div>
              <div className="flex-1 bg-white dark:bg-gray-900 rounded-lg border border-gray-300 dark:border-gray-600 px-3 py-2 text-center">
                <div className="text-lg font-semibold text-gray-900 dark:text-white">
                  {conversationTokens > 1000 ? `${(conversationTokens / 1000).toFixed(1)}K` : conversationTokens}
                </div>
                <div className="text-xs text-gray-500">tokens</div>
              </div>
              <button
                onClick={onClearChat}
                disabled={disabled || conversationTurns === 0}
                className="px-3 py-2 text-sm rounded-lg border border-gray-300 dark:border-gray-600
                           hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50
                           transition-colors"
                title="Clear chat"
              >
                🗑️
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* RAG Section */}
      <div className="px-4 py-3 border-t border-gray-200 dark:border-gray-700">
        <div className="max-w-6xl mx-auto">
          <div className="flex items-center gap-4 flex-wrap">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">🔍 RAG</span>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={useRag}
                  onChange={(e) => onUseRagChange(e.target.checked)}
                  disabled={disabled || !selectedWorkspace?.has_datasets}
                  className="rounded border-gray-300 text-accent focus:ring-accent"
                />
                <span className="text-sm text-gray-600 dark:text-gray-400">Enable</span>
              </label>
            </div>

            {useRag && (
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-500">Context:</span>
                <input
                  type="range"
                  min={500}
                  max={8000}
                  step={500}
                  value={ragMaxTokens}
                  onChange={(e) => onRagMaxTokensChange(Number(e.target.value))}
                  disabled={disabled}
                  className="w-24"
                />
                <span className="text-xs text-gray-600 dark:text-gray-400 w-16">{ragMaxTokens} tok</span>
              </div>
            )}

            {workspace && (
              <div className="flex items-center gap-2">
                {indexStatus?.indexed ? (
                  <span className="text-xs text-green-600">✅ {indexStatus.chunk_count || 0} chunks</span>
                ) : (
                  <span className="text-xs text-amber-600">⚠️ Not indexed</span>
                )}
                <button
                  onClick={handleIndex}
                  disabled={disabled || indexing}
                  className="px-2 py-1 text-xs rounded border border-gray-300 dark:border-gray-600
                             hover:bg-gray-100 dark:hover:bg-gray-700 disabled:opacity-50"
                >
                  {indexing ? '...' : indexStatus?.indexed ? '🔄 Rebuild' : '📚 Index'}
                </button>
              </div>
            )}

            {/* Cross-workspace skills auto-include toggle (v3+) */}
            {onIncludeSkillsChange !== undefined && (
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">🧩 Skills</span>
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={includeSkills ?? true}
                    onChange={(e) => onIncludeSkillsChange(e.target.checked)}
                    disabled={disabled}
                    className="rounded border-gray-300 text-accent focus:ring-accent"
                  />
                  <span className="text-sm text-gray-600 dark:text-gray-400">Auto-include</span>
                </label>
              </div>
            )}

            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="ml-auto px-2 py-1 text-xs text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
            >
              {showAdvanced ? '▼ Advanced' : '▶ Advanced'}
            </button>
          </div>
        </div>
      </div>

      {/* MCP Servers — collapsible section. Default-collapsed to keep the
          settings panel compact; click the header to reveal the panel. */}
      <div className="border-t border-gray-200 dark:border-gray-700">
        <button
          type="button"
          onClick={() => setShowMcp((v) => !v)}
          className="w-full px-4 py-2 flex items-center gap-2 text-left hover:bg-gray-100 dark:hover:bg-gray-700/40 transition-colors"
          aria-expanded={showMcp}
        >
          <span className="text-gray-400 text-xs w-3">{showMcp ? '▼' : '▶'}</span>
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
            🔌 MCP Servers
          </span>
          <span className="text-xs text-gray-500 dark:text-gray-400">
            external tools, resources, and prompts via Model Context Protocol
          </span>
        </button>
        {showMcp && (
          <div className="max-w-6xl mx-auto">
            <McpServersPanel disabled={disabled} />
          </div>
        )}
      </div>

      {/* Agent Skills — collapsible section. Same default-collapsed pattern
          as MCP servers. The panel filters by the active workspace by
          default so workspace-scoped skills only surface where they're
          meant to. */}
      <div className="border-t border-gray-200 dark:border-gray-700">
        <button
          type="button"
          onClick={() => setShowSkills((v) => !v)}
          className="w-full px-4 py-2 flex items-center gap-2 text-left hover:bg-gray-100 dark:hover:bg-gray-700/40 transition-colors"
          aria-expanded={showSkills}
        >
          <span className="text-gray-400 text-xs w-3">{showSkills ? '▼' : '▶'}</span>
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
            🧩 Agent Skills
          </span>
          <span className="text-xs text-gray-500 dark:text-gray-400">
            workspace-scoped procedural capabilities (SKILL.md)
          </span>
        </button>
        {showSkills && (
          <div className="max-w-6xl mx-auto">
            <SkillsPanel workspace={workspace} disabled={disabled} />
          </div>
        )}
      </div>

      {/* Cross-workspace Policy — collapsible section. Surfaces the
          synthesis-engine policy substrate: per-workspace routing rules,
          effective confidentiality across the selected workspace mix,
          model-routing verdicts, and the recent audit-log feed. */}
      <div className="border-t border-gray-200 dark:border-gray-700">
        <button
          type="button"
          onClick={() => setShowPolicy((v) => !v)}
          className="w-full px-4 py-2 flex items-center gap-2 text-left hover:bg-gray-100 dark:hover:bg-gray-700/40 transition-colors"
          aria-expanded={showPolicy}
        >
          <span className="text-gray-400 text-xs w-3">{showPolicy ? '▼' : '▶'}</span>
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
            🛡️ Cross-workspace Policy
          </span>
          <span className="text-xs text-gray-500 dark:text-gray-400">
            confidentiality, model routing, audit log
          </span>
        </button>
        {showPolicy && (
          <div className="max-w-6xl mx-auto">
            <PolicyPanel
              workspace={workspace}
              model={model}
              disabled={disabled}
            />
          </div>
        )}
      </div>

      {/* Advanced Settings */}
      {showAdvanced && (
        <div className="px-4 py-3 border-t border-gray-200 dark:border-gray-700 bg-gray-100 dark:bg-gray-900">
          <div className="max-w-6xl mx-auto">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="flex flex-col gap-1">
                <label className="text-xs font-medium text-gray-600 dark:text-gray-400">
                  Max Response Tokens
                </label>
                <select
                  value={maxTokens}
                  onChange={(e) => onMaxTokensChange(Number(e.target.value))}
                  disabled={disabled}
                  className="rounded border border-gray-300 dark:border-gray-600
                             bg-white dark:bg-gray-800 px-2 py-1 text-sm"
                >
                  {MAX_TOKEN_OPTIONS.map((opt) => (
                    <option key={opt} value={opt}>
                      {opt.toLocaleString()}
                    </option>
                  ))}
                </select>
              </div>

              <div className="flex flex-col gap-1">
                <label className="text-xs font-medium text-gray-600 dark:text-gray-400">
                  Model Info
                </label>
                <div className="text-xs text-gray-500 space-y-0.5">
                  <div>Provider: {providers.find((p) => p.id === selectedModel?.provider)?.name || selectedModel?.provider || '-'}</div>
                  <div>Model: {selectedModel?.display_name || selectedModel?.name || 'None'}</div>
                  <div>Context: {selectedModel ? `${(selectedModel.context_window / 1000).toFixed(0)}K` : '-'}</div>
                </div>
              </div>

              {/* API Keys with detailed status */}
              <div className="flex flex-col gap-1 md:col-span-2">
                <label className="text-xs font-medium text-gray-600 dark:text-gray-400">
                  API Keys for {workspace || 'default'}
                </label>
                <div className="bg-white dark:bg-gray-800 rounded border border-gray-200 dark:border-gray-700 p-2">
                  <div className="grid grid-cols-3 gap-2 text-xs">
                    {providers.map((p) => {
                      const status = keysStatus[p.id];
                      const effectiveSource = getEffectiveKeySource(p.id);
                      const canOverride = status?.has_workspace_key && status?.has_default_key;
                      const isLocalProvider = Boolean(p.is_local);

                      return (
                        <div key={p.id} className="flex flex-col gap-1">
                          <div className="flex items-center gap-1">
                            {status?.has_key ? (
                              <span className="text-green-600">✅</span>
                            ) : (
                              <span className="text-red-500">❌</span>
                            )}
                            <span className={status?.has_key ? 'text-gray-700 dark:text-gray-300' : 'text-gray-400'}>
                              {p.name}
                            </span>
                            {isLocalProvider && (
                              <span className="text-[10px] text-gray-500" title="Runs locally; no API key required">
                                · 🏠 local
                              </span>
                            )}
                          </div>

                          {status?.has_key && !isLocalProvider && (
                            <div className="flex items-center gap-1 pl-5">
                              <span
                                className={`text-[10px] px-1 py-0.5 rounded ${
                                  effectiveSource === 'workspace'
                                    ? 'bg-accent-light text-accent-dark dark:text-accent'
                                    : 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-400'
                                }`}
                              >
                                {effectiveSource}
                              </span>

                              {canOverride && (
                                <button
                                  onClick={() =>
                                    setKeyOverrides((prev) => ({
                                      ...prev,
                                      [p.id]: prev[p.id] === 'default' ? 'auto' : 'default',
                                    }))
                                  }
                                  className="text-[10px] text-accent hover:text-accent-dark ml-1"
                                  title={keyOverrides[p.id] === 'default' ? 'Use workspace key' : 'Use default key'}
                                >
                                  [{keyOverrides[p.id] === 'default' ? 'use workspace' : 'use default'}]
                                </button>
                              )}
                            </div>
                          )}

                          {!status?.has_key && !isLocalProvider && (
                            <div className="text-[10px] text-gray-400 pl-5">
                              no key configured
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
