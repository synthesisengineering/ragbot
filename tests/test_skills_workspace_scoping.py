"""Tests for workspace-scoped skill discovery (Phase 2 Agent A)."""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path
from typing import Dict, List

import pytest

# Add src/ to sys.path so the tests can import the package under test.
_SRC = os.path.join(os.path.dirname(__file__), '..', 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from synthesis_engine.skills import (  # noqa: E402
    Skill,
    SkillScope,
    discover_skills,
    get_skills_for_workspace,
    parse_skill,
    resolve_skill_roots,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_skill(root: Path, name: str, frontmatter: str = "") -> Path:
    """Create a minimal skill directory under ``root`` with the given name."""
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    body = (
        "---\n"
        f"name: {name}\n"
        "description: test fixture\n"
        f"{frontmatter}"
        "---\n\n"
        f"# {name}\n\nbody.\n"
    )
    (skill_dir / "SKILL.md").write_text(body)
    return skill_dir


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch) -> Path:
    """Patch ``Path.home``, ``HOME``, and ``SYNTHESIS_IDENTITY_CONFIG``.

    The synthetic layout mirrors the operator's actual workspace shape so
    the new discovery roots ``~/workspaces/acme-user/synthesis-skills``,
    ``~/workspaces/acme-user/synthesis-skills-acme-user``, and the
    ``~/workspaces/<W>/synthesis-skills-<W>`` glob can all be exercised
    against tmp_path without ever touching the real home directory.

    The fixture also writes a synthetic ``identity.yaml`` declaring
    ``acme-user`` as a personal workspace, so the discovery chain
    includes the open-source ``synthesis-skills`` checkout under
    ``~/workspaces/acme-user/`` and the path-convention rule treats
    ``synthesis-skills-acme-user`` as universal.
    """
    home = tmp_path / "fakehome"
    home.mkdir()

    # Materialise the conventional sub-tree so resolve_skill_roots finds
    # every kind of root.
    (home / ".synthesis" / "skills").mkdir(parents=True)
    (home / ".claude" / "skills").mkdir(parents=True)
    (home / ".agents" / "skills").mkdir(parents=True)
    (home / "workspaces" / "acme-user" / "synthesis-skills").mkdir(parents=True)
    (home / "workspaces" / "acme-user" / "synthesis-skills-acme-user").mkdir(parents=True)

    identity = home / ".synthesis" / "identity.yaml"
    identity.write_text("personal_workspaces:\n  - acme-user\n")

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("SYNTHESIS_IDENTITY_CONFIG", str(identity))
    return home


# ---------------------------------------------------------------------------
# SkillScope construction from frontmatter
# ---------------------------------------------------------------------------


class TestSkillScopeFromFrontmatter:
    def test_universal_string(self) -> None:
        scope = SkillScope.from_frontmatter("universal", fallback=None)
        assert scope.universal is True
        assert scope.workspaces == ()

    def test_workspace_list(self) -> None:
        scope = SkillScope.from_frontmatter(["a", "b"], fallback=None)
        assert scope.universal is False
        assert scope.workspaces == ("a", "b")

    def test_workspaces_mapping(self) -> None:
        scope = SkillScope.from_frontmatter(
            {"workspaces": ["a", "b"]},
            fallback=None,
        )
        assert scope.universal is False
        assert scope.workspaces == ("a", "b")

    def test_single_workspace_string(self) -> None:
        scope = SkillScope.from_frontmatter("acme-news", fallback=None)
        assert scope.universal is False
        assert scope.workspaces == ("acme-news",)

    def test_explicit_universal_true_mapping(self) -> None:
        # universal=True wins even when workspaces is also set.
        scope = SkillScope.from_frontmatter(
            {"universal": True, "workspaces": ["acme-news"]},
            fallback=None,
        )
        assert scope.universal is True
        assert scope.workspaces == ()

    def test_missing_value_falls_back(self) -> None:
        fallback = SkillScope.for_workspaces(("acme-news",))
        assert SkillScope.from_frontmatter(None, fallback=fallback) == fallback
        assert SkillScope.from_frontmatter("", fallback=fallback) == fallback

    def test_dedup_preserves_first_occurrence_order(self) -> None:
        scope = SkillScope.from_frontmatter(["a", "b", "a", "c"], fallback=None)
        assert scope.workspaces == ("a", "b", "c")


# ---------------------------------------------------------------------------
# SkillScope construction from path convention
# ---------------------------------------------------------------------------


class TestSkillScopeFromPath:
    def test_workspace_scoped_path(self) -> None:
        path = "/Users/acme-user/workspaces/acme-news/synthesis-skills-acme-news/foo"
        scope = SkillScope.from_path_convention(path)
        assert scope.universal is False
        assert scope.workspaces == ("acme-news",)

    def test_non_convention_path_is_universal(self) -> None:
        # No matching workspaces/<W>/synthesis-skills-<W> segment.
        path = "/Users/acme-user/.claude/skills/some-skill"
        scope = SkillScope.from_path_convention(path)
        assert scope.universal is True

    def test_mismatched_pair_is_universal(self) -> None:
        # Inner name doesn't match the workspace token; we won't guess.
        path = "/Users/x/workspaces/foo/synthesis-skills-bar/skill"
        scope = SkillScope.from_path_convention(path)
        assert scope.universal is True

    def test_case_insensitive(self) -> None:
        path = "/Users/x/Workspaces/Beta-Media/SYNTHESIS-SKILLS-beta-media/skill"
        scope = SkillScope.from_path_convention(path)
        assert scope.universal is False
        # Preserves the original casing of the workspace token from the
        # first segment.
        assert scope.workspaces == ("Beta-Media",)

    def test_personal_workspace_path_becomes_universal(
        self, tmp_path, monkeypatch
    ) -> None:
        """A path under a workspace declared as personal in
        ``~/.synthesis/identity.yaml`` is treated as universal even though
        the path convention would normally scope it.

        Operators put their personal authoring/workflow skills in
        ``~/workspaces/<self>/synthesis-skills-<self>/``; those should be
        visible from every workspace they run in. The identity config
        declares which ``<self>`` names qualify.
        """
        identity = tmp_path / "identity.yaml"
        identity.write_text("personal_workspaces:\n  - acme-user\n")
        monkeypatch.setenv("SYNTHESIS_IDENTITY_CONFIG", str(identity))

        path = "/Users/x/workspaces/acme-user/synthesis-skills-acme-user/skill"
        scope = SkillScope.from_path_convention(path)
        assert scope.universal is True
        assert scope.workspaces == ()

        # And confirm the non-personal case still scopes normally.
        other = "/Users/x/workspaces/acme-news/synthesis-skills-acme-news/skill"
        scope_other = SkillScope.from_path_convention(other)
        assert scope_other.universal is False
        assert scope_other.workspaces == ("acme-news",)


# ---------------------------------------------------------------------------
# Frontmatter overrides path convention
# ---------------------------------------------------------------------------


class TestFrontmatterOverridesPath:
    def test_frontmatter_universal_wins_over_workspace_path(
        self, tmp_path: Path
    ) -> None:
        # Build a skill at a workspace-suggestive path that declares
        # ``scope: universal`` explicitly.
        skill_dir = (
            tmp_path / "workspaces" / "acme-news" / "synthesis-skills-acme-news"
            / "shared-skill"
        )
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(textwrap.dedent("""\
            ---
            name: shared-skill
            description: shared by everyone
            scope: universal
            ---

            # shared-skill
            body.
            """))
        skill = parse_skill(str(skill_dir))
        assert skill is not None
        assert skill.scope.universal is True
        assert skill.scope.workspaces == ()

    def test_path_convention_used_when_frontmatter_silent(
        self, tmp_path: Path
    ) -> None:
        skill_dir = (
            tmp_path / "workspaces" / "acme-news" / "synthesis-skills-acme-news"
            / "private-skill"
        )
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(textwrap.dedent("""\
            ---
            name: private-skill
            description: implicit scope
            ---

            # private-skill
            body.
            """))
        skill = parse_skill(str(skill_dir))
        assert skill is not None
        assert skill.scope.universal is False
        assert skill.scope.workspaces == ("acme-news",)


# ---------------------------------------------------------------------------
# discover_skills against the new sources
# ---------------------------------------------------------------------------


class TestDiscoveryAgainstNewSources:
    def test_discovers_from_all_new_sources(self, fake_home: Path) -> None:
        # Plant one skill in each new source.
        _write_skill(
            fake_home / "workspaces" / "acme-user" / "synthesis-skills",
            "open-source-skill",
        )
        _write_skill(
            fake_home / "workspaces" / "acme-user" / "synthesis-skills-acme-user",
            "acme-user-personal-skill",
        )
        # Per-workspace glob source.
        (fake_home / "workspaces" / "acme-news" / "synthesis-skills-acme-news").mkdir(
            parents=True
        )
        _write_skill(
            fake_home / "workspaces" / "acme-news" / "synthesis-skills-acme-news",
            "acme-news-only-skill",
        )

        skills = discover_skills()
        names = {s.name for s in skills}
        assert "open-source-skill" in names
        assert "acme-user-personal-skill" in names
        assert "acme-news-only-skill" in names

    def test_workspace_glob_filters_mismatched_pairs(self, fake_home: Path) -> None:
        # ~/workspaces/foo/synthesis-skills-bar/ must NOT be discovered;
        # the inner name doesn't match the workspace token.
        bad = fake_home / "workspaces" / "foo" / "synthesis-skills-bar"
        bad.mkdir(parents=True)
        _write_skill(bad, "should-not-appear")

        skills = discover_skills()
        assert "should-not-appear" not in {s.name for s in skills}

    def test_native_private_sources_discovered(self, fake_home: Path) -> None:
        # Shared, Claude Code-private, and Codex/agent-neutral paths all
        # remain part of the chain.
        _write_skill(fake_home / ".synthesis" / "skills", "synthesis-skill")
        _write_skill(fake_home / ".claude" / "skills", "claude-skill")
        _write_skill(fake_home / ".agents" / "skills", "codex-skill")

        skills = discover_skills()
        names = {s.name for s in skills}
        assert "synthesis-skill" in names
        assert "claude-skill" in names
        assert "codex-skill" in names

    def test_discovers_both_client_plugin_cache_layouts(
        self, fake_home: Path
    ) -> None:
        claude_plugins = (
            fake_home / ".claude" / "plugins" / "cache"
            / "example-marketplace" / "example-plugin" / "1.0.0" / "skills"
        )
        codex_plugins = (
            fake_home / ".codex" / "plugins" / "cache"
            / "example-marketplace" / "example-plugin" / "1.0.0" / "skills"
        )
        _write_skill(claude_plugins, "claude-plugin-skill")
        _write_skill(codex_plugins, "codex-plugin-skill")

        names = {skill.name for skill in discover_skills()}
        assert "claude-plugin-skill" in names
        assert "codex-plugin-skill" in names

    def test_agent_neutral_private_copy_overrides_claude_copy(
        self, fake_home: Path
    ) -> None:
        _write_skill(fake_home / ".claude" / "skills", "shared-private")
        codex_copy = _write_skill(
            fake_home / ".agents" / "skills", "shared-private"
        )

        discovered = {
            skill.name: skill for skill in discover_skills()
        }
        assert discovered["shared-private"].path == str(codex_copy)

    def test_discover_skills_returns_all_regardless_of_scope(
        self, fake_home: Path
    ) -> None:
        # The unfiltered API surface must keep returning every skill so
        # the CLI's "list all" path works.
        _write_skill(
            fake_home / "workspaces" / "acme-user" / "synthesis-skills",
            "universal-one",
        )
        (fake_home / "workspaces" / "beta-media" / "synthesis-skills-beta-media").mkdir(
            parents=True
        )
        _write_skill(
            fake_home / "workspaces" / "beta-media" / "synthesis-skills-beta-media",
            "beta-media-only",
        )

        skills = discover_skills()
        names = {s.name for s in skills}
        assert {"universal-one", "beta-media-only"} <= names


# ---------------------------------------------------------------------------
# get_skills_for_workspace
# ---------------------------------------------------------------------------


class TestGetSkillsForWorkspace:
    def test_returns_universal_only_when_no_workspace_scoped_skills(
        self, fake_home: Path
    ) -> None:
        # Two universal skills, one under the open-source root (path
        # gives universal scope) and one under the personal-skills root
        # that opts in to universal scope via explicit frontmatter
        # (overrides the workspace-suggestive path).
        _write_skill(
            fake_home / "workspaces" / "acme-user" / "synthesis-skills",
            "universal-a",
        )
        _write_skill(
            fake_home / "workspaces" / "acme-user" / "synthesis-skills-acme-user",
            "universal-b",
            frontmatter="scope: universal\n",
        )

        skills = get_skills_for_workspace("personal", inheritance_config={})
        names = {s.name for s in skills}
        # Both planted universals are visible from a workspace without
        # any scoped skills. The bundled starter pack is also visible
        # (universal scope), but only the planted skills are asserted
        # against here; the workspace-scoping behavior is what this test
        # exercises, not the starter-pack content.
        assert {"universal-a", "universal-b"} <= names

    def test_returns_universal_plus_workspace_when_scope_matches(
        self, fake_home: Path
    ) -> None:
        _write_skill(
            fake_home / "workspaces" / "acme-user" / "synthesis-skills",
            "universal-x",
        )
        (fake_home / "workspaces" / "acme-news" / "synthesis-skills-acme-news").mkdir(
            parents=True
        )
        _write_skill(
            fake_home / "workspaces" / "acme-news" / "synthesis-skills-acme-news",
            "acme-news-private",
        )

        skills = get_skills_for_workspace("acme-news", inheritance_config={})
        names = {s.name for s in skills}
        # The universal-scoped skill is visible; the workspace-scoped
        # skill is visible because the workspace matches. Additional
        # bundled universals (the starter pack) ride along.
        assert {"universal-x", "acme-news-private"} <= names

    def test_filters_out_other_workspaces_skills(self, fake_home: Path) -> None:
        # acme-news and beta-media each have a private skill. A run in the
        # personal workspace must see neither (only universals).
        for ws in ("acme-news", "beta-media"):
            (fake_home / "workspaces" / ws / f"synthesis-skills-{ws}").mkdir(
                parents=True
            )
            _write_skill(
                fake_home / "workspaces" / ws / f"synthesis-skills-{ws}",
                f"{ws}-private",
            )
        _write_skill(
            fake_home / "workspaces" / "acme-user" / "synthesis-skills",
            "universal-shared",
        )

        skills = get_skills_for_workspace("personal", inheritance_config={})
        names = {s.name for s in skills}
        assert "universal-shared" in names
        assert "acme-news-private" not in names
        assert "beta-media-private" not in names

    def test_inheritance_pulls_in_ancestor_workspace_skills(
        self, fake_home: Path
    ) -> None:
        # csa-team inherits from acme-news. A run in csa-team must see
        # acme-news-scoped skills.
        for ws in ("acme-news", "csa-team"):
            (fake_home / "workspaces" / ws / f"synthesis-skills-{ws}").mkdir(
                parents=True
            )
        _write_skill(
            fake_home / "workspaces" / "acme-news" / "synthesis-skills-acme-news",
            "acme-news-skill",
        )
        _write_skill(
            fake_home / "workspaces" / "csa-team" / "synthesis-skills-csa-team",
            "csa-team-skill",
        )

        inheritance_config: Dict = {
            "projects": {
                "acme-news": {"inherits_from": []},
                "csa-team": {"inherits_from": ["acme-news"]},
            }
        }

        skills = get_skills_for_workspace(
            "csa-team", inheritance_config=inheritance_config
        )
        names = {s.name for s in skills}
        assert "acme-news-skill" in names
        assert "csa-team-skill" in names

    def test_inheritance_does_not_leak_sibling_workspaces(
        self, fake_home: Path
    ) -> None:
        # Even with an inheritance config in play, a sibling workspace
        # (beta-media) must not leak into a csa-team run.
        for ws in ("acme-news", "beta-media", "csa-team"):
            (fake_home / "workspaces" / ws / f"synthesis-skills-{ws}").mkdir(
                parents=True
            )
            _write_skill(
                fake_home / "workspaces" / ws / f"synthesis-skills-{ws}",
                f"{ws}-skill",
            )

        inheritance_config: Dict = {
            "projects": {
                "acme-news": {"inherits_from": []},
                "beta-media": {"inherits_from": []},
                "csa-team": {"inherits_from": ["acme-news"]},
            }
        }

        skills = get_skills_for_workspace(
            "csa-team", inheritance_config=inheritance_config
        )
        names = {s.name for s in skills}
        assert "beta-media-skill" not in names
        assert "acme-news-skill" in names
        assert "csa-team-skill" in names

    def test_missing_inheritance_config_uses_single_workspace_chain(
        self, fake_home: Path
    ) -> None:
        # When no my-projects.yaml is reachable, the fallback chain is
        # just (workspace_name,). Universal + own-scope skills must work.
        _write_skill(
            fake_home / "workspaces" / "acme-user" / "synthesis-skills",
            "universal-z",
        )
        (fake_home / "workspaces" / "solo" / "synthesis-skills-solo").mkdir(
            parents=True
        )
        _write_skill(
            fake_home / "workspaces" / "solo" / "synthesis-skills-solo",
            "solo-skill",
        )

        # inheritance_config=None and no my-projects.yaml exists in any
        # ai-knowledge-* repo under fake_home → graceful single-workspace
        # chain.
        skills = get_skills_for_workspace("solo", inheritance_config=None)
        names = {s.name for s in skills}
        assert "universal-z" in names
        assert "solo-skill" in names

    def test_multi_workspace_scoped_skill_visible_from_either(
        self, fake_home: Path
    ) -> None:
        # A skill scoped to BOTH acme-news and beta-media (via explicit
        # frontmatter on a skill that lives in a neutral location).
        shared_root = fake_home / ".synthesis" / "skills"
        shared_dir = shared_root / "media-publisher-workflow"
        shared_dir.mkdir(parents=True)
        (shared_dir / "SKILL.md").write_text(textwrap.dedent("""\
            ---
            name: media-publisher-workflow
            description: shared between two media workspaces
            scope: [acme-news, beta-media]
            ---

            # media-publisher-workflow
            body.
            """))

        # Add per-workspace dirs so the workspaces themselves resolve.
        for ws in ("acme-news", "beta-media"):
            (fake_home / "workspaces" / ws / f"synthesis-skills-{ws}").mkdir(
                parents=True
            )

        from_acme = get_skills_for_workspace("acme-news", inheritance_config={})
        from_beta = get_skills_for_workspace("beta-media", inheritance_config={})
        assert "media-publisher-workflow" in {s.name for s in from_acme}
        assert "media-publisher-workflow" in {s.name for s in from_beta}

        # And invisible to an unrelated workspace.
        personal = get_skills_for_workspace("personal", inheritance_config={})
        assert "media-publisher-workflow" not in {s.name for s in personal}

    def test_sorted_by_name_and_no_duplicate_names(self, fake_home: Path) -> None:
        # Two universal skills in different roots — sort + dedup by name.
        _write_skill(
            fake_home / "workspaces" / "acme-user" / "synthesis-skills",
            "zeta",
        )
        _write_skill(
            fake_home / "workspaces" / "acme-user" / "synthesis-skills-acme-user",
            "alpha",
        )

        skills = get_skills_for_workspace("personal", inheritance_config={})
        names = [s.name for s in skills]
        assert names == sorted(names)
        assert len(names) == len(set(names))
