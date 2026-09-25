import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { thinkingOptions } from '../src/lib/thinking';
import { SettingsPanel } from '../src/components/SettingsPanel';
import type { ModelInfo } from '../src/lib/api';
const sample = { id: 'openai/gpt-6-astra', name: 'gpt-6-astra', provider: 'openai', context_window: 1050000, supports_streaming: true, supports_system_role: true, supports_thinking: true, thinking: { strict: true, modes: ['low', 'medium', 'high', 'xhigh', 'max'], default: 'medium' } } satisfies ModelInfo;
vi.mock('../src/lib/api', () => ({
  getModels: async () => ({ models: [sample], default_model: sample.id }),
  getWorkspaces: async () => [], getConfig: async () => ({}),
  getProviders: async () => ({ providers: [] }), getTemperatureSettings: async () => ({}),
  getKeysStatus: async () => ({}), getIndexStatus: async () => null, indexWorkspace: vi.fn(),
}));
vi.mock('../src/components/ModelPicker', () => ({ ModelPicker: () => null }));
vi.mock('../src/components/McpServersPanel', () => ({ McpServersPanel: () => null }));
vi.mock('../src/components/PolicyPanel', () => ({ PolicyPanel: () => null }));
vi.mock('../src/components/SkillsPanel', () => ({ SkillsPanel: () => null }));
describe('catalog reasoning selector', () => {
  it('offers the exact verified modes', () => {
    expect(thinkingOptions(sample).map(x => x.value)).toEqual(['auto', 'low', 'medium', 'high', 'xhigh', 'max']);
  });
  it('retains an unsupported explicit selection visibly', () => {
    expect(thinkingOptions(sample, 'off').at(-1)).toEqual({ value: 'off', label: 'off (unsupported by this model)', disabled: true });
  });
  it('actual settings panel preserves extra high without choosing a default', async () => {
    const change = vi.fn();
    render(<SettingsPanel workspace={undefined} onWorkspaceChange={vi.fn()} model={sample.id} onModelChange={vi.fn()} temperature={0.5} onTemperatureChange={vi.fn()} maxTokens={8192} onMaxTokensChange={vi.fn()} useRag={false} onUseRagChange={vi.fn()} ragMaxTokens={16000} onRagMaxTokensChange={vi.fn()} thinkingEffort="xhigh" onThinkingEffortChange={change} />);
    const select = await screen.findByLabelText('Thinking effort');
    expect(select).toHaveValue('xhigh');
    expect(screen.queryByRole('option', { name: 'off', exact: true })).toBeNull();
    expect(change).not.toHaveBeenCalled();
  });
});
