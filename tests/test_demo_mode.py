"""Tests for demo mode (v3.2+).

Demo mode hard-isolates ragbot's discovery and skill-discovery layers
to the bundled ``demo/`` directory inside the repo. These tests lock
in that isolation so a future change can't accidentally let real
workspace or skill names leak through when ``RAGBOT_DEMO=1``.
"""

from __future__ import annotations

import os
import sys

import pytest

_SRC = os.path.join(os.path.dirname(__file__), '..', 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from ragbot.demo import (  # noqa: E402
    DEMO_SKILLS_WORKSPACE_NAME,
    DEMO_WORKSPACE_NAME,
    demo_data_root,
    demo_skills_path,
    demo_workspace_path,
    is_demo_mode,
)


# ---------------------------------------------------------------------------
# is_demo_mode
# ---------------------------------------------------------------------------


class TestIsDemoMode:
    @pytest.mark.parametrize("value,expected", [
        ("1", True),
        ("true", True),
        ("True", True),
        ("YES", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("no", False),
        ("", False),
    ])
    def test_recognises_truthy_and_falsy(self, value, expected, monkeypatch):
        monkeypatch.setenv("RAGBOT_DEMO", value)
        assert is_demo_mode() is expected

    def test_unset_env_means_off(self, monkeypatch):
        monkeypatch.delenv("RAGBOT_DEMO", raising=False)
        assert is_demo_mode() is False


# ---------------------------------------------------------------------------
# Bundled demo paths
# ---------------------------------------------------------------------------


class TestBundledDemoPaths:
    def test_demo_data_root_exists(self):
        root = demo_data_root()
        assert root is not None
        assert root.is_dir()

    def test_demo_workspace_has_compile_config(self):
        workspace = demo_workspace_path()
        assert workspace is not None
        assert (workspace / "compile-config.yaml").is_file()

    def test_demo_workspace_has_source_files(self):
        workspace = demo_workspace_path()
        assert workspace is not None
        assert (workspace / "source" / "datasets").is_dir()
        assert (workspace / "source" / "instructions").is_dir()

    def test_demo_skills_directory_has_at_least_one_skill(self):
        skills = demo_skills_path()
        assert skills is not None
        # At least one subdirectory containing SKILL.md.
        skill_mds = list(skills.glob("*/SKILL.md"))
        assert skill_mds, "demo/skills/ should ship at least one SKILL.md"


# ---------------------------------------------------------------------------
# Discovery isolation
# ---------------------------------------------------------------------------


class TestDiscoveryIsolation:
    def setup_method(self):
        from synthesis_engine.workspaces import resolve_repo_index  # noqa: F401
        # No cached state to reset; resolve_repo_index reads env each call.

    def test_resolve_repo_index_returns_only_demo_in_demo_mode(self, monkeypatch):
        monkeypatch.setenv("RAGBOT_DEMO", "1")
        # Even with a configured base path that points at real workspaces,
        # demo mode must override.
        monkeypatch.setenv("RAGBOT_BASE_PATH", "/Users/someone/ai-knowledge")

        from synthesis_engine.workspaces import resolve_repo_index

        index = resolve_repo_index()
        assert set(index.keys()) == {DEMO_WORKSPACE_NAME}, (
            "Demo mode must return ONLY the demo workspace; got "
            f"{set(index.keys())}"
        )
        # And the path must point at the bundled demo dir.
        assert index[DEMO_WORKSPACE_NAME].endswith("/demo/ai-knowledge-demo")

    def test_resolve_repo_index_normal_mode_still_works(self, monkeypatch):
        monkeypatch.delenv("RAGBOT_DEMO", raising=False)
        # Use an explicit base_path so this test doesn't depend on the
        # host's ~/.synthesis/console.yaml or workspace glob.
        from synthesis_engine.workspaces import resolve_repo_index

        index = resolve_repo_index("/nonexistent")
        # Empty result is fine; the assertion is "didn't crash and didn't
        # short-circuit to demo".
        assert DEMO_WORKSPACE_NAME not in index


class TestSkillDiscoveryIsolation:
    def test_demo_mode_returns_only_bundled_skills(self, monkeypatch):
        """Demo mode hides USER-installed skills but keeps bundled v3.4 content.

        The substrate's demo filter (see ``ragbot/_demo_registration.py::
        _skill_roots_filter``) pairs two bundled roots:

        * The synthesis-engine starter pack — the six universal skills that
          ship with the substrate (summarize-document, fact-check-claims,
          agent-self-review, draft-and-revise, workspace-search-with-citations,
          cross-workspace-synthesis). These are part of "what v3.4 ships with,"
          not user content, so demo mode INCLUDES them.
        * The bundled ``demo/skills/`` directory (ragbot-demo-skill).

        Demo mode HIDES the user's installed skills (workspace skill packs,
        plugin skills, the operator's private skill repo). That isolation
        is the property this test asserts: the visible set must be EXACTLY
        the bundled content and nothing else.
        """

        monkeypatch.setenv("RAGBOT_DEMO", "1")
        from synthesis_engine.skills import discover_skills

        skills = discover_skills()
        names = {s.name for s in skills}
        expected = {
            # Starter pack (substrate-bundled).
            "summarize-document",
            "fact-check-claims",
            "agent-self-review",
            "draft-and-revise",
            "workspace-search-with-citations",
            "cross-workspace-synthesis",
            # Ragbot's demo bundle.
            "ragbot-demo-skill",
        }
        assert names == expected, (
            "Demo mode must return exactly the bundled starter pack + "
            f"ragbot-demo-skill; got {names}"
        )

    def test_normal_mode_unaffected(self, monkeypatch, tmp_path):
        monkeypatch.delenv("RAGBOT_DEMO", raising=False)
        # Empty user-supplied roots → empty result (no leak from defaults
        # which we don't want this test depending on).
        from synthesis_engine.skills import discover_skills

        skills = discover_skills(roots=[str(tmp_path)])
        assert skills == []


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestDemoConstants:
    def test_demo_workspace_name_is_stable(self):
        # Other code (CLI filters, API banner, test fixtures) depends on
        # this exact value. Lock it.
        assert DEMO_WORKSPACE_NAME == "demo"

    def test_demo_skills_workspace_name_is_distinct(self):
        # Must NOT equal the canonical 'skills' workspace, otherwise the
        # demo's auto-index pollutes the real skills collection on the
        # same vector store.
        assert DEMO_SKILLS_WORKSPACE_NAME != "skills"
        assert DEMO_SKILLS_WORKSPACE_NAME == "demo_skills"


# ---------------------------------------------------------------------------
# First-run auto-index (src/ragbot.py::_ensure_demo_indexed)
# ---------------------------------------------------------------------------


def _load_cli():
    """Load src/ragbot.py by path: ``import ragbot`` resolves to the package
    of the same name, not the CLI script (see tests/test_mcp_cli_wiring.py)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ragbot_cli_demo_under_test", os.path.join(_SRC, "ragbot.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestEnsureDemoIndexed:
    def test_indexes_bundled_content_and_skills_when_empty(self):
        from unittest.mock import patch

        import rag

        cli = _load_cli()
        with patch.object(rag, 'get_index_status', return_value=(False, 0)), \
             patch.object(rag, 'index_content', return_value={'indexed': 1}) as index_content, \
             patch.object(rag, 'index_skills', return_value={}) as index_skills:
            cli._ensure_demo_indexed()

        indexed = [call.args[0] for call in index_content.call_args_list]
        assert indexed and set(indexed) == {DEMO_WORKSPACE_NAME}
        assert index_skills.call_args.kwargs['workspace_name'] == DEMO_SKILLS_WORKSPACE_NAME

    def test_skips_when_already_indexed(self):
        from unittest.mock import patch

        import rag

        cli = _load_cli()
        with patch.object(rag, 'get_index_status', return_value=(True, 4)), \
             patch.object(rag, 'index_content') as index_content, \
             patch.object(rag, 'index_skills') as index_skills:
            cli._ensure_demo_indexed()

        index_content.assert_not_called()
        index_skills.assert_not_called()
