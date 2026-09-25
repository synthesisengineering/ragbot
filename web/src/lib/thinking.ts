import type { ModelInfo, ThinkingEffort } from './api';

/** Catalog preference never replaces an explicit selection, including stale ones. */
export function thinkingOptions(model: ModelInfo | undefined, selected?: ThinkingEffort) {
  const modes = model?.thinking?.strict
    ? (model.thinking.modes ?? [])
    : ['off', 'minimal', 'low', 'medium', 'high'];
  const values = ['auto', ...modes];
  const options = values.map(value => ({ value, label: value, disabled: false }));
  if (selected && !values.includes(selected)) {
    options.push({ value: selected, label: `${selected} (unsupported by this model)`, disabled: true });
  }
  return options;
}
