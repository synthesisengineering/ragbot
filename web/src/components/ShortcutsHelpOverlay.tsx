'use client';

/**
 * Help overlay listing every registered keyboard shortcut.
 *
 * Triggered by ⌘? (Ctrl+? on Windows/Linux). The overlay reads directly from
 * the `keyboardShortcuts` singleton in `lib/shortcuts.ts` so the list always
 * mirrors the actual registry — no manual sync.
 *
 * Accessibility:
 *  - Renders as a modal dialog with `role="dialog"` + `aria-modal="true"`.
 *  - Escape closes; clicking the backdrop closes.
 *  - Focus is moved into the overlay on open and trapped between the close
 *    button and the dialog itself so Tab cycles within the dialog.
 *  - The shortcut combo is rendered as a `<kbd>` element with platform-aware
 *    formatting via `formatKeys()`.
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import {
  keyboardShortcuts,
  formatKeys,
  activeSpec,
  type KeyboardShortcut,
} from '@/lib/shortcuts';

export interface ShortcutsHelpOverlayProps {
  open: boolean;
  onClose: () => void;
}

export function ShortcutsHelpOverlay({ open, onClose }: ShortcutsHelpOverlayProps) {
  // Local mirror of the shortcut list. We subscribe to registry changes so
  // the overlay reflects shortcuts that mount/unmount while it's open.
  const [shortcuts, setShortcuts] = useState<KeyboardShortcut[]>(() =>
    keyboardShortcuts.getAll(),
  );
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const closeButtonRef = useRef<HTMLButtonElement | null>(null);

  // Re-read the registry whenever it changes. The subscription is live from
  // mount (even while the overlay is hidden), so the mirror is always
  // current by the time the user presses ⌘? — no refresh-on-open needed.
  useEffect(() => {
    const unsubscribe = keyboardShortcuts.onChange(() => {
      setShortcuts(keyboardShortcuts.getAll());
    });
    return unsubscribe;
  }, []);

  // Move focus into the overlay on open so screen readers announce it and
  // Tab navigation stays inside the dialog.
  useEffect(() => {
    if (!open) return;
    // Wait one tick so the dialog is mounted before focusing.
    const id = window.setTimeout(() => {
      closeButtonRef.current?.focus();
    }, 0);
    return () => window.clearTimeout(id);
  }, [open]);

  // Escape closes; Tab/Shift+Tab traps focus inside the dialog.
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLDivElement>) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        e.stopPropagation();
        onClose();
        return;
      }
      if (e.key === 'Tab') {
        const root = dialogRef.current;
        if (!root) return;
        const focusables = root.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
        );
        if (focusables.length === 0) return;
        const first = focusables[0];
        const last = focusables[focusables.length - 1];
        const active = document.activeElement as HTMLElement | null;
        if (e.shiftKey) {
          if (active === first) {
            e.preventDefault();
            last.focus();
          }
        } else {
          if (active === last) {
            e.preventDefault();
            first.focus();
          }
        }
      }
    },
    [onClose],
  );

  if (!open) return null;

  return (
    <div
      data-testid="shortcuts-help-overlay"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      onClick={(e) => {
        // Backdrop click closes; clicks on the dialog itself are ignored.
        if (e.target === e.currentTarget) onClose();
      }}
      onKeyDown={handleKeyDown}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="shortcuts-help-title"
        className="bg-white dark:bg-gray-900 rounded-lg shadow-xl max-w-2xl w-full max-h-[80vh] overflow-y-auto"
      >
        <div className="flex items-center justify-between px-5 py-3 border-b border-gray-200 dark:border-gray-700">
          <h2
            id="shortcuts-help-title"
            className="text-lg font-semibold text-gray-900 dark:text-white"
          >
            Keyboard Shortcuts
          </h2>
          <button
            ref={closeButtonRef}
            type="button"
            onClick={onClose}
            aria-label="Close shortcuts help"
            className="text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 text-xl leading-none w-8 h-8 rounded hover:bg-gray-100 dark:hover:bg-gray-800"
          >
            ×
          </button>
        </div>

        <div className="px-5 py-4">
          {shortcuts.length === 0 ? (
            <p className="text-sm text-gray-500 dark:text-gray-400">
              No keyboard shortcuts registered.
            </p>
          ) : (
            <ul className="divide-y divide-gray-100 dark:divide-gray-800">
              {shortcuts.map((s) => {
                const spec = activeSpec(s);
                const label = formatKeys(spec);
                return (
                  <li
                    key={s.id}
                    className="flex items-center justify-between py-2 gap-4"
                  >
                    <div className="flex-1 min-w-0">
                      <div className="text-sm text-gray-900 dark:text-gray-100">
                        {s.description}
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
                        <span
                          aria-label={`Scope: ${s.scope}`}
                          className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-medium ${
                            s.scope === 'global'
                              ? 'bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300'
                              : 'bg-amber-50 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300'
                          }`}
                        >
                          {s.scope}
                        </span>
                      </div>
                    </div>
                    <kbd
                      aria-label={`Keyboard combo: ${label}`}
                      className="font-mono text-sm px-2 py-1 rounded border border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-800 text-gray-700 dark:text-gray-200 whitespace-nowrap"
                    >
                      {label}
                    </kbd>
                  </li>
                );
              })}
            </ul>
          )}
        </div>

        <div className="px-5 py-3 border-t border-gray-200 dark:border-gray-700 text-xs text-gray-500 dark:text-gray-400">
          Press{' '}
          <kbd className="font-mono px-1 py-0.5 rounded border border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-800">
            Esc
          </kbd>{' '}
          to close.{' '}
          <span className="ml-2">
            <span className="inline-block px-1 py-0.5 rounded bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 mr-1">
              global
            </span>{' '}
            fires anywhere;
            <span className="inline-block px-1 py-0.5 rounded bg-amber-50 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 mx-1">
              overlay-safe
            </span>{' '}
            yields to text inputs.
          </span>
        </div>
      </div>
    </div>
  );
}
