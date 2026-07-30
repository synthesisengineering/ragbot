"""Demo-mode helpers.

When ``RAGBOT_DEMO=1`` (or ``true``/``yes``), ragbot:

* Discovers exactly one workspace — the bundled ``demo/ai-knowledge-demo/`` —
  and ignores everything else (the user's ``~/.synthesis/console.yaml``,
  the ``~/workspaces/*/ai-knowledge-*`` glob, legacy flat parents).
* Discovers exactly one skill root — the bundled ``demo/skills/`` —
  and ignores ``~/.synthesis/skills``, ``~/.claude/skills``,
  ``~/.agents/skills``, and both clients' plugin caches.
* Reports ``demo_mode: true`` from ``/health`` and ``/api/config`` so the
  Web UI can render an unmistakable banner.

The motivation is twofold: an evaluator can run ``ragbot --demo`` and
see ragbot work end-to-end without setup, and a maintainer can capture
screenshots without any risk that a real workspace name leaks into the
frame.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


# The canonical workspace names used in demo mode. Exposed so other code
# (chat default-workspace, vector-store namespace, screenshots) doesn't
# duplicate the literal.
DEMO_WORKSPACE_NAME = "demo"
# Demo skills go into their own workspace so the demo's cross-workspace
# fan-out can NOT pull from a real ``skills`` workspace that happens to
# share the same vector store on the host.
DEMO_SKILLS_WORKSPACE_NAME = "demo_skills"


def is_demo_mode() -> bool:
    """Return True if ``RAGBOT_DEMO`` is set to a truthy value."""

    raw = os.environ.get("RAGBOT_DEMO", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def demo_data_root() -> Optional[Path]:
    """Locate the bundled ``demo/`` directory inside the ragbot repo.

    Returns the absolute path or ``None`` if the directory is missing
    (which would only happen for a misconfigured installation).
    """

    # ragbot/src/ragbot/demo.py → ragbot/demo/ is two ``parents`` up.
    candidates = [
        Path(__file__).resolve().parents[2] / "demo",
        # Fallback for installed-wheel layouts that may differ.
        Path(__file__).resolve().parents[3] / "demo",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def demo_workspace_path() -> Optional[Path]:
    """Path to the bundled demo workspace, or None if missing."""

    root = demo_data_root()
    if root is None:
        return None
    candidate = root / "ai-knowledge-demo"
    return candidate if candidate.is_dir() else None


def demo_skills_path() -> Optional[Path]:
    """Path to the bundled demo skills directory, or None if missing."""

    root = demo_data_root()
    if root is None:
        return None
    candidate = root / "skills"
    return candidate if candidate.is_dir() else None
