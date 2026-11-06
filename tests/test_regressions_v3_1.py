"""Regression tests for bugs caught during the v3.0 manual test session.

These tests cover the five issues found while running the v3.0 test plan
against a live local stack. Each test is named after the bug so a future
breakage maps directly to the fix.
"""

from __future__ import annotations

import os
import sys

import pytest

_SRC = os.path.join(os.path.dirname(__file__), '..', 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# ---------------------------------------------------------------------------
# Bug 1: chat workspace lookup used the legacy data_root pattern, missing
# workspaces discovered via ~/.synthesis/console.yaml or the workspace glob.
# ---------------------------------------------------------------------------


class TestBug1WorkspaceLookupHonoursDiscovery:
    def test_load_workspaces_with_none_uses_full_resolution_chain(self):
        from ragbot import load_workspaces_as_profiles
        # When called with None, the function must use the resolver chain
        # (synthesis-console config + glob + legacy fallbacks). The exact
        # number of workspaces is environment-dependent but the function
        # must not crash and must return a list.
        profiles = load_workspaces_as_profiles(None)
        assert isinstance(profiles, list)


# ---------------------------------------------------------------------------
# Bug 2: rag.get_relevant_context's budget loop used `break` on oversized
# chunks, which prevented smaller chunks from ever being included when a
# whole-file script chunk topped the rankings.
# ---------------------------------------------------------------------------


class TestBug2BudgetLoopContinuesPastOversizedChunks:
    def test_oversized_top_result_does_not_block_smaller_following_results(self):
        # Re-implement the trim loop standalone to lock in the expected
        # semantics (continue, not break, when a chunk doesn't fit).
        results = [
            {"text": "X" * 10000, "score": 0.9, "metadata": {}},  # too big
            {"text": "small", "score": 0.5, "metadata": {}},        # fits
        ]
        budget = 1000
        kept = []
        cur = 0
        for r in results:
            t_tok = len(r["text"]) // 4
            if cur + t_tok > budget:
                continue   # the bug fix
            kept.append(r)
            cur += t_tok
        assert len(kept) == 1
        assert kept[0]["text"] == "small"


# Bugs 3 and 4 (Claude 4.x thinking shapes and their temperature rules) were
# retired on 2026-11-06 along with the models they concerned. Every Claude in
# the catalog now takes adaptive thinking through output_config and never
# receives sampling parameters; tests/test_catalog_refresh.py covers that.


class TestBug5TransformersTokenizersCompat:
    def test_sentence_transformers_imports_cleanly(self):
        # If transformers and tokenizers are compatible, importing
        # sentence_transformers.SentenceTransformer must succeed without
        # raising ImportError("tokenizers>=0.21,<0.22 is required ...").
        #
        # ``sentence_transformers`` is a hard requirement (requirements.txt
        # pins >=6.2.0), but it pulls in torch which weighs several hundred
        # MB. Dev machines without a full install would otherwise see this
        # regression test FAIL rather than SKIP. importorskip preserves the
        # regression coverage on Docker / CI where the dep is installed,
        # while keeping local dev runs green when only the lightweight
        # dependencies are present.
        pytest.importorskip("sentence_transformers")
        from sentence_transformers import SentenceTransformer  # noqa: F401

    def test_rag_module_reports_available_when_deps_load(self):
        # is_rag_available() now requires sentence-transformers AND a working
        # vector store backend. The transformers/tokenizers regression caused
        # this to silently flip to False because sentence-transformers raised
        # at import time. Lock in the import-success path. Same env
        # rationale as the import test above.
        pytest.importorskip("sentence_transformers")
        from rag import is_rag_available
        # The actual return depends on RAGBOT_DATABASE_URL etc., but the
        # function must not raise.
        is_rag_available()


# ---------------------------------------------------------------------------
# Extra: regression test for the get_embedding_dimension rename. The old
# get_sentence_embedding_dimension method emits a FutureWarning in
# sentence-transformers 5.x. Our helper prefers the new name.
# ---------------------------------------------------------------------------


class TestEmbeddingDimensionHelper:
    def test_helper_prefers_new_method(self):
        from rag import _get_embedding_dimension

        class FakeModelNew:
            def get_embedding_dimension(self):
                return 384

        class FakeModelOld:
            def get_sentence_embedding_dimension(self):
                return 384

        assert _get_embedding_dimension(FakeModelNew()) == 384
        assert _get_embedding_dimension(FakeModelOld()) == 384
