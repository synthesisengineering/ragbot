"""Tests for the LiteLLM thinking / reasoning_effort wiring (loose end)."""

from __future__ import annotations

import os
import sys

import pytest

_SRC = os.path.join(os.path.dirname(__file__), '..', 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from ragbot.core import _resolve_thinking_for_model, _normalise_effort  # noqa: E402


class TestNormaliseEffort:
    @pytest.mark.parametrize("raw,expected", [
        ("high", "high"),
        ("HIGH", "high"),
        (" medium ", "medium"),
        ("low", "low"),
        ("minimal", "minimal"),
        ("off", "off"),
        ("auto", "auto"),
        ("default", "auto"),
        ("nope", None),
        (None, None),
        ("", None),
    ])
    def test_normalise(self, raw, expected):
        assert _normalise_effort(raw) == expected


class TestResolveThinkingForModel:
    """Behavioural rules:

    * Catalog cloud models carry strict contracts. Their effort defaults to
      the catalog's ``thinking.default``; an explicit value must be one of
      the model's ``modes`` and is never replaced by another value.
    * Local (non-strict) models: a flagship with thinking metadata defaults
      to ``medium``; anything else sends no thinking parameter, leaving the
      model's own default in place.
    * A model without thinking metadata never receives thinking params.
    * Explicit override (per-call) wins over the env var, which wins over
      the catalog default.
    """

    @pytest.mark.parametrize("model,default", [
        ("anthropic/claude-fable-5-1", "high"),
        ("anthropic/claude-opus-5-5", "medium"),
        ("anthropic/claude-sonnet-5-5", "high"),
        ("anthropic/claude-haiku-5-5", "medium"),
        ("openai/gpt-6-astra", "medium"),
        ("openai/gpt-6.1-sol", "medium"),
        ("openai/gpt-6.1-terra", "medium"),
        ("openai/gpt-6-luna", "medium"),
        ("gemini/gemini-4-argon", "medium"),
        ("gemini/gemini-3.8-flash", "medium"),
        ("gemini/gemini-3.5-flash-lite", "minimal"),
    ])
    def test_cloud_models_default_to_catalog_effort(self, monkeypatch, model, default):
        monkeypatch.delenv("RAGBOT_THINKING_EFFORT", raising=False)
        assert _resolve_thinking_for_model(model) == {"reasoning_effort": default}

    def test_explicit_effort_is_kept(self, monkeypatch):
        monkeypatch.delenv("RAGBOT_THINKING_EFFORT", raising=False)
        out = _resolve_thinking_for_model("anthropic/claude-sonnet-5-5", requested_effort="max")
        assert out == {"reasoning_effort": "max"}

    def test_env_var_overrides_catalog_default(self, monkeypatch):
        monkeypatch.setenv("RAGBOT_THINKING_EFFORT", "low")
        assert _resolve_thinking_for_model("anthropic/claude-haiku-5-5") == {"reasoning_effort": "low"}

    def test_per_call_override_beats_env_var(self, monkeypatch):
        monkeypatch.setenv("RAGBOT_THINKING_EFFORT", "high")
        out = _resolve_thinking_for_model("anthropic/claude-haiku-5-5", requested_effort="low")
        assert out == {"reasoning_effort": "low"}

    @pytest.mark.parametrize("effort", ["off", "ridiculous", "minimal"])
    def test_unsupported_effort_raises_for_cloud_models(self, monkeypatch, effort):
        monkeypatch.delenv("RAGBOT_THINKING_EFFORT", raising=False)
        with pytest.raises(ValueError):
            _resolve_thinking_for_model("anthropic/claude-fable-5-1", requested_effort=effort)

    def test_local_non_flagship_defaults_to_no_thinking_param(self, monkeypatch):
        # Qwen3.8 thinks by default on its own; sending nothing keeps that.
        monkeypatch.delenv("RAGBOT_THINKING_EFFORT", raising=False)
        assert _resolve_thinking_for_model("ollama_chat/qwen3.8:27b") == {}

    def test_local_explicit_effort_passes_through(self, monkeypatch):
        monkeypatch.delenv("RAGBOT_THINKING_EFFORT", raising=False)
        out = _resolve_thinking_for_model("ollama_chat/qwen3.8:27b", requested_effort="high")
        assert out == {"reasoning_effort": "high"}

    def test_local_per_call_off_sends_nothing(self, monkeypatch):
        monkeypatch.setenv("RAGBOT_THINKING_EFFORT", "high")
        assert _resolve_thinking_for_model("ollama_chat/qwen3.8:27b", requested_effort="off") == {}

    def test_local_flagship_with_thinking_defaults_to_medium(self, monkeypatch):
        import synthesis_engine.config as config

        monkeypatch.delenv("RAGBOT_THINKING_EFFORT", raising=False)
        monkeypatch.setattr(config, "get_model_info", lambda model: {
            "id": model, "is_flagship": True,
            "thinking": {"supported": True, "mode": "hybrid", "modes": ["thinking", "instruct"]},
        })
        assert _resolve_thinking_for_model("ollama_chat/example:1b") == {"reasoning_effort": "medium"}

    def test_model_without_thinking_metadata_ignores_overrides(self, monkeypatch):
        # Gemma 4 entries carry no thinking block.
        monkeypatch.delenv("RAGBOT_THINKING_EFFORT", raising=False)
        assert _resolve_thinking_for_model("ollama_chat/gemma4:31b", requested_effort="high") == {}
