"""Tests for personal-archive/ resolution and its owner-context gating.

`personal-archive/` is a top-level directory (sibling of source/, compiled/)
in an ai-knowledge-* repo holding lower-priority reference/archive content.
It must behave exactly like the existing -private repo filtering (ADR-014):
invisible by default, visible only when RAGBOT_OWNER_CONTEXT=1 — but the
check is independent of the private-repo-suffix filter, since a repo need
not be named `*-private` to have an archive directory that should stay
hidden by default.
"""

from __future__ import annotations

import os
import sys

import pytest

_SRC = os.path.join(os.path.dirname(__file__), '..', 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from synthesis_engine.workspaces import (  # noqa: E402
    _build_repo_metadata,
    is_owner_context,
    resolve_workspace_paths,
    workspace_to_profile,
)


@pytest.fixture
def repo_with_archive(tmp_path):
    """A minimal ai-knowledge-* repo with a source/datasets/ dir AND a
    personal-archive/ dir, neither of which is named `*-private`."""
    repo = tmp_path / "ai-knowledge-example"
    (repo / "source" / "datasets").mkdir(parents=True)
    (repo / "source" / "datasets" / "note.md").write_text("hello")
    (repo / "personal-archive").mkdir()
    (repo / "personal-archive" / "old-note.md").write_text("archived")
    return repo


class TestOwnerContextAlias:
    def test_public_alias_matches_env(self, monkeypatch):
        monkeypatch.delenv("RAGBOT_OWNER_CONTEXT", raising=False)
        assert is_owner_context() is False
        monkeypatch.setenv("RAGBOT_OWNER_CONTEXT", "1")
        assert is_owner_context() is True


class TestBuildRepoMetadataArchiveGating:
    def test_archive_invisible_without_owner_context(self, tmp_path, monkeypatch, repo_with_archive):
        monkeypatch.delenv("RAGBOT_OWNER_CONTEXT", raising=False)
        meta = _build_repo_metadata("example", str(repo_with_archive))
        assert meta is not None
        assert meta["archive"] is None
        assert meta["has_archive"] is False
        # Datasets remain visible — only archive is gated.
        assert meta["has_datasets"] is True

    def test_archive_visible_with_owner_context(self, tmp_path, monkeypatch, repo_with_archive):
        monkeypatch.setenv("RAGBOT_OWNER_CONTEXT", "1")
        meta = _build_repo_metadata("example", str(repo_with_archive))
        assert meta is not None
        assert meta["archive"] == str(repo_with_archive / "personal-archive")
        assert meta["has_archive"] is True

    def test_repo_without_archive_dir_is_unaffected(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RAGBOT_OWNER_CONTEXT", "1")
        repo = tmp_path / "ai-knowledge-plain"
        (repo / "source" / "datasets").mkdir(parents=True)
        meta = _build_repo_metadata("plain", str(repo))
        assert meta is not None
        assert meta["archive"] is None
        assert meta["has_archive"] is False

    def test_archive_gate_independent_of_private_suffix(self, tmp_path, monkeypatch):
        """A repo named *-private with an archive dir: has_archive should
        still require owner context on its own — it isn't automatically
        true just because the repo is already private-suffixed, and it
        isn't skipped either. (The repo itself would already have been
        filtered out of discovery upstream in resolve_repo_index() unless
        owner context is on; this test exercises _build_repo_metadata in
        isolation, which is called only for repos that survived that
        filter.)"""
        monkeypatch.setenv("RAGBOT_OWNER_CONTEXT", "1")
        repo = tmp_path / "ai-knowledge-example-private"
        (repo / "source" / "datasets").mkdir(parents=True)
        (repo / "personal-archive").mkdir()
        meta = _build_repo_metadata("example-private", str(repo))
        assert meta["has_archive"] is True


class TestResolveWorkspacePathsArchive:
    def test_archive_key_present_and_populated(self, monkeypatch, repo_with_archive):
        monkeypatch.setenv("RAGBOT_OWNER_CONTEXT", "1")
        meta = _build_repo_metadata("example", str(repo_with_archive))
        workspace = {
            'dir_name': 'example',
            'config': {'inherits_from': []},
            'ai_knowledge': meta,
        }
        resolved = resolve_workspace_paths(workspace, all_workspaces=None)
        assert resolved['archive'] == [meta['archive']]
        assert resolved['datasets']

    def test_archive_empty_without_owner_context(self, monkeypatch, repo_with_archive):
        monkeypatch.delenv("RAGBOT_OWNER_CONTEXT", raising=False)
        meta = _build_repo_metadata("example", str(repo_with_archive))
        workspace = {
            'dir_name': 'example',
            'config': {'inherits_from': []},
            'ai_knowledge': meta,
        }
        resolved = resolve_workspace_paths(workspace, all_workspaces=None)
        assert resolved['archive'] == []

    def test_workspace_to_profile_carries_archive(self, monkeypatch, repo_with_archive):
        monkeypatch.setenv("RAGBOT_OWNER_CONTEXT", "1")
        meta = _build_repo_metadata("example", str(repo_with_archive))
        workspace = {
            'name': 'Example', 'dir_name': 'example',
            'config': {'inherits_from': []},
            'ai_knowledge': meta,
        }
        profile = workspace_to_profile(workspace, all_workspaces=None)
        assert profile['archive'] == [meta['archive']]

    def test_inheritance_propagates_archive_from_parent(self, monkeypatch, tmp_path):
        monkeypatch.setenv("RAGBOT_OWNER_CONTEXT", "1")
        parent_repo = tmp_path / "ai-knowledge-parent"
        (parent_repo / "source" / "datasets").mkdir(parents=True)
        (parent_repo / "personal-archive").mkdir()
        parent_meta = _build_repo_metadata("parent", str(parent_repo))

        child_repo = tmp_path / "ai-knowledge-child"
        (child_repo / "source" / "datasets").mkdir(parents=True)
        child_meta = _build_repo_metadata("child", str(child_repo))

        parent_ws = {
            'dir_name': 'parent', 'config': {'inherits_from': []},
            'ai_knowledge': parent_meta,
        }
        child_ws = {
            'dir_name': 'child', 'config': {'inherits_from': ['parent']},
            'ai_knowledge': child_meta,
        }
        resolved = resolve_workspace_paths(child_ws, all_workspaces=[parent_ws, child_ws])
        assert resolved['archive'] == [parent_meta['archive']]
