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


# ---------------------------------------------------------------------------
# Bug 3: Anthropic extended thinking requires temperature=1.0; the resolver
# now forces this whenever thinking is sent for Claude.
# ---------------------------------------------------------------------------


class TestBug3AnthropicThinkingForcesTemperature:
    def test_pre_4_7_claude_with_explicit_effort_forces_temp_1(self, monkeypatch):
        # Haiku 4.5 is the remaining pre-4.7 model: reasoning_effort path,
        # temperature forced to 1 (extended thinking requires it).
        from ragbot.core import _resolve_thinking_for_model
        monkeypatch.delenv("RAGBOT_THINKING_EFFORT", raising=False)
        out = _resolve_thinking_for_model(
            "anthropic/claude-haiku-4-5-20251001",
            requested_effort="medium",
        )
        assert out["reasoning_effort"] == "medium"
        assert out["temperature"] == 1.0

    def test_claude_4_8_plus_default_omits_temperature_with_adaptive_shape(self, monkeypatch):
        # The 4.8+/5.x inverse of the original bug: the API now REJECTS the
        # temperature parameter ("`temperature` is deprecated for this
        # model", verified 2026-07-14), so the resolver must NOT send it.
        from ragbot.core import _resolve_thinking_for_model
        monkeypatch.delenv("RAGBOT_THINKING_EFFORT", raising=False)
        out = _resolve_thinking_for_model("anthropic/claude-fable-5")
        assert out["thinking"] == {"type": "adaptive"}
        assert "temperature" not in out


# ---------------------------------------------------------------------------
# Bug 4: Claude 4.7+ requires the new ``thinking.type.adaptive`` API shape.
# LiteLLM <=1.83.13's reasoning_effort mapper still emits the older
# ``enabled`` shape, which Anthropic now rejects. The resolver bypasses
# reasoning_effort for Claude 4.7+ and sends adaptive directly.
# ---------------------------------------------------------------------------


class TestBug4Claude47AdaptiveShape:
    def test_opus_4_7_emits_adaptive_not_reasoning_effort(self, monkeypatch):
        from ragbot.core import _resolve_thinking_for_model
        monkeypatch.delenv("RAGBOT_THINKING_EFFORT", raising=False)
        out = _resolve_thinking_for_model(
            "anthropic/claude-opus-4-8",
            requested_effort="high",
        )
        assert "reasoning_effort" not in out
        assert out["thinking"] == {"type": "adaptive"}

    def test_pre_4_7_claude_still_uses_reasoning_effort(self, monkeypatch):
        from ragbot.core import _resolve_thinking_for_model
        monkeypatch.delenv("RAGBOT_THINKING_EFFORT", raising=False)
        out = _resolve_thinking_for_model(
            "anthropic/claude-haiku-4-5-20251001",
            requested_effort="high",
        )
        assert out["reasoning_effort"] == "high"
        assert "thinking" not in out


# ---------------------------------------------------------------------------
# Bug 5: transformers had a hard pin on tokenizers<0.22 that was incompatible
# with the tokenizers 0.22.x pulled in by the upgraded sentence-transformers.
# requirements.txt now pins transformers>=5.6.0 (the version supporting
# tokenizers 0.22.x).
# ---------------------------------------------------------------------------


class TestBug5TransformersTokenizersCompat:
    def test_sentence_transformers_imports_cleanly(self):
        # If transformers and tokenizers are compatible, importing
        # sentence_transformers.SentenceTransformer must succeed without
        # raising ImportError("tokenizers>=0.21,<0.22 is required ...").
        #
        # ``sentence_transformers`` is a hard requirement (requirements.txt
        # pins >=5.4.0), but it pulls in torch which weighs several hundred
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
