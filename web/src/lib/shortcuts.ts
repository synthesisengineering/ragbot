/**
 * Keyboard shortcut registry (Phase 4, Ragbot v3.4).
 *
 * A small typed registry that decouples shortcut *intent* (id + description +
 * key spec) from shortcut *binding* (the window keydown listener that React
 * mounts in `Chat.tsx`). One singleton instance, `keyboardShortcuts`, holds
 * every registered shortcut. The React provider in `Chat.tsx` wires
 * `window.addEventListener('keydown', e => keyboardShortcuts.dispatch(e))`
 * on mount, and individual feature components call `register()` /
 * `unregister()` for their own shortcuts. This lets the help overlay
 * (`ShortcutsHelpOverlay.tsx`) enumerate every shortcut from a single source
 * of truth.
 *
 * Platform awareness: the spec is two key arrays — `mac: ['Meta', 'k']` and
 * `other: ['Control', 'k']` — and `dispatch()` picks the right one based on
 * `isMac()`. The same physical combination (⌘K on mac, Ctrl+K on win/linux)
 * triggers the same intent.
 *
 * Scope: `'global'` shortcuts fire even when a text input has focus.
 * `'overlay-safe'` shortcuts fire only when no text input has focus, which
 * lets letter-key combos coexist with normal typing without grabbing
 * keystrokes mid-message.
 */

/** Modifier key names normalised across browsers. */
export type ModifierKey = 'Meta' | 'Control' | 'Alt' | 'Shift';

/**
 * A keyboard shortcut declaration. Both `mac` and `other` are arrays where
 * every element except the last is a modifier (`'Meta'`, `'Control'`,
 * `'Alt'`, `'Shift'`) and the last element is the literal `KeyboardEvent.key`
 * value (case-insensitive — `'k'` and `'K'` both match the K key).
 */
export interface KeyboardShortcut {
  /** Stable identifier — also used as the registry key. */
  id: string;
  /** Platform-aware key sequences. The last element is the non-modifier key. */
  keys: { mac: string[]; other: string[] };
  /** Human-readable description shown in the help overlay. */
  description: string;
  /**
   * `'global'` — fires regardless of focus target (use for application-wide
   * actions like ⌘N, ⌘J, ⌘B, ⌘.).
   * `'overlay-safe'` — fires only when no text input / textarea / content-
   * editable element is focused. Reserved for shortcuts that would otherwise
   * eat normal typing.
   */
  scope: 'global' | 'overlay-safe';
  /** Action invoked when the combo matches. Receives the raw event. */
  handler: (event: KeyboardEvent) => void;
}

/**
 * Returns true when the current host is macOS. Uses the modern UA-Client-Hints
 * platform when available, falls back to `navigator.platform`, and finally to
 * a `userAgent` substring match. SSR-safe: returns `false` when `navigator`
 * is undefined.
 */
export function isMac(): boolean {
  if (typeof navigator === 'undefined') return false;
  // UA-Client-Hints (modern Chromium): `navigator.userAgentData.platform`.
  const uaData = (navigator as unknown as { userAgentData?: { platform?: string } }).userAgentData;
  if (uaData && typeof uaData.platform === 'string') {
    return uaData.platform.toLowerCase().includes('mac');
  }
  const platform = (navigator.platform || '').toLowerCase();
  if (platform.includes('mac')) return true;
  // Fallback for browsers that hide platform.
  const ua = (navigator.userAgent || '').toLowerCase();
  return ua.includes('mac');
}

/** Returns true when the keydown target is a text input / textarea / contentEditable. */
function isTextTarget(el: EventTarget | null): boolean {
  if (!(el instanceof HTMLElement)) return false;
  const tag = el.tagName;
  if (tag === 'INPUT') {
    // Allow shortcuts inside hidden/checkbox/button inputs.
    const type = (el as HTMLInputElement).type.toLowerCase();
    const typingTypes = ['text', 'search', 'url', 'email', 'tel', 'password', 'number', 'date'];
    return typingTypes.includes(type) || type === '';
  }
  if (tag === 'TEXTAREA') return true;
  if (el.isContentEditable) return true;
  return false;
}

/**
 * Returns true when the keydown event's modifier flags + key name match the
 * given key spec. The spec's last element is the key name (case-insensitive);
 * preceding elements are required modifiers. Modifiers NOT listed in the spec
 * must NOT be pressed (so `['Meta', 'k']` matches ⌘K but not ⌘⌥K — that would
 * be `['Meta', 'Alt', 'k']`).
 */
function matches(event: KeyboardEvent, spec: string[]): boolean {
  if (spec.length === 0) return false;
  const key = spec[spec.length - 1].toLowerCase();
  if (event.key.toLowerCase() !== key) return false;

  const mods = new Set(spec.slice(0, -1).map((m) => m.toLowerCase()));
  const wantMeta = mods.has('meta');
  const wantCtrl = mods.has('control');
  const wantAlt = mods.has('alt');
  const wantShift = mods.has('shift');

  if (Boolean(event.metaKey) !== wantMeta) return false;
  if (Boolean(event.ctrlKey) !== wantCtrl) return false;
  if (Boolean(event.altKey) !== wantAlt) return false;
  // Shift is special-cased: many keys (like '?') already imply Shift via the
  // character produced. We require Shift only when the spec asks for it, but
  // we don't fail if Shift is unexpectedly pressed for keys where it's
  // implicit. Treat shift symmetrically: spec wins.
  if (Boolean(event.shiftKey) !== wantShift) return false;
  return true;
}

/**
 * Returns the active key spec for a shortcut on the current platform.
 * Exported for use by the help overlay and tests.
 */
export function activeSpec(shortcut: KeyboardShortcut): string[] {
  return isMac() ? shortcut.keys.mac : shortcut.keys.other;
}

/**
 * Formats a key spec for display. On macOS this uses Unicode symbols
 * (⌘, ⌥, ⇧, ⌃); on other platforms it uses spelled-out names joined by '+'.
 * The non-modifier key is shown in uppercase for visual consistency.
 */
export function formatKeys(keys: string[]): string {
  if (keys.length === 0) return '';
  const onMac = isMac();
  const macSymbols: Record<string, string> = {
    meta: '⌘', // ⌘
    control: '⌃', // ⌃
    alt: '⌥', // ⌥
    shift: '⇧', // ⇧
  };
  const otherNames: Record<string, string> = {
    meta: 'Meta',
    control: 'Ctrl',
    alt: 'Alt',
    shift: 'Shift',
  };
  const mods = keys.slice(0, -1);
  const main = keys[keys.length - 1];
  // Display non-letter keys as-is, letters uppercased.
  const mainDisplay = main.length === 1 ? main.toUpperCase() : main;

  if (onMac) {
    const parts = mods.map((m) => macSymbols[m.toLowerCase()] ?? m);
    return parts.join('') + mainDisplay;
  }
  const parts = mods.map((m) => otherNames[m.toLowerCase()] ?? m);
  return [...parts, mainDisplay].join('+');
}

/**
 * The registry. Encapsulates the shortcut map plus the dispatch loop. One
 * singleton, `keyboardShortcuts`, is exported below. Tests can construct
 * their own instances to keep state isolated.
 */
export class ShortcutRegistry {
  private shortcuts = new Map<string, KeyboardShortcut>();
  // Listeners notified when the registry membership changes — used by the
  // help overlay to re-render its list without polling.
  private changeListeners = new Set<() => void>();

  /**
   * Add a shortcut. Re-registering with the same id replaces the previous
   * entry — this lets components re-mount safely without leaking handlers.
   */
  register(shortcut: KeyboardShortcut): void {
    this.shortcuts.set(shortcut.id, shortcut);
    this.notifyChange();
  }

  /** Remove a shortcut by id. No-op if not present. */
  unregister(id: string): void {
    if (this.shortcuts.delete(id)) {
      this.notifyChange();
    }
  }

  /** Return every registered shortcut in insertion order. */
  getAll(): KeyboardShortcut[] {
    return Array.from(this.shortcuts.values());
  }

  /** Look up a single shortcut by id (used by tests). */
  get(id: string): KeyboardShortcut | undefined {
    return this.shortcuts.get(id);
  }

  /**
   * Dispatch a keydown event through the registry. Returns true if a
   * registered shortcut handled the event (and the handler was invoked),
   * false otherwise. `'overlay-safe'` shortcuts are skipped when a text
   * target has focus; `'global'` shortcuts always fire when matched.
   */
  dispatch(event: KeyboardEvent): boolean {
    const focusedOnText = isTextTarget(event.target);
    for (const shortcut of this.shortcuts.values()) {
      if (shortcut.scope === 'overlay-safe' && focusedOnText) continue;
      const spec = activeSpec(shortcut);
      if (matches(event, spec)) {
        shortcut.handler(event);
        return true;
      }
    }
    return false;
  }

  /** Subscribe to membership changes. Returns an unsubscribe function. */
  onChange(listener: () => void): () => void {
    this.changeListeners.add(listener);
    return () => {
      this.changeListeners.delete(listener);
    };
  }

  /** Clear all registered shortcuts. Test helper. */
  clear(): void {
    if (this.shortcuts.size === 0) return;
    this.shortcuts.clear();
    this.notifyChange();
  }

  private notifyChange(): void {
    for (const listener of this.changeListeners) {
      try {
        listener();
      } catch (err) {
        // A bad listener must not break dispatch for others.
        console.error('Shortcut registry change listener threw:', err);
      }
    }
  }
}

/** Singleton instance — Chat.tsx wires the keydown listener; features register. */
export const keyboardShortcuts = new ShortcutRegistry();
