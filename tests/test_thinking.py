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

    * Flagship model with thinking metadata → default ``medium``.
    * Non-flagship model with thinking metadata:
        - if engines.yaml declares a discrete ``modes:`` list and ``off`` is
          NOT among them (OpenAI / Gemini style, where reasoning is always
          on), default to the LOWEST listed mode (typically ``minimal``) so
          the provider's own reasoning default doesn't consume the entire
          output budget on long-context calls.
        - otherwise (Claude with ``mode: adaptive`` or no modes listed) →
          default to ``off`` (send no thinking params; provider's neutral
          default applies).
    * Model without thinking metadata → never pass thinking params.
    * Explicit override (per-call) wins over both env and engines.yaml default.
    * Env-var override wins over engines.yaml default but loses to per-call.
    """

    def test_flagship_fable_5_uses_adaptive_thinking_shape(self, monkeypatch):
        # Claude 4.7+ (and all 5.x) requires the ``thinking.type.adaptive``
        # shape; LiteLLM's reasoning_effort mapper still emits the older
        # ``thinking.type.enabled`` form which the API rejects.
        monkeypatch.delenv("RAGBOT_THINKING_EFFORT", raising=False)
        out = _resolve_thinking_for_model("anthropic/claude-fable-5")
        assert out == {"thinking": {"type": "adaptive"}}

    def test_claude_4_8_and_5_x_explicit_effort_use_adaptive_shape(self, monkeypatch):
        # The 4.7+ detection is version-aware, not a hardcoded id list:
        # Opus 4.8 and Sonnet 5 must take the adaptive-shape path too.
        monkeypatch.delenv("RAGBOT_THINKING_EFFORT", raising=False)
        out = _resolve_thinking_for_model("anthropic/claude-opus-4-8", requested_effort="high")
        assert out == {"thinking": {"type": "adaptive"}}
        out = _resolve_thinking_for_model("anthropic/claude-sonnet-5", requested_effort="medium")
        assert out == {"thinking": {"type": "adaptive"}}

    def test_non_flagship_with_thinking_defaults_to_off(self, monkeypatch):
        monkeypatch.delenv("RAGBOT_THINKING_EFFORT", raising=False)
        out = _resolve_thinking_for_model("anthropic/claude-sonnet-5")
        assert out == {}
        # Opus 4.8 stays supported but is no longer the flagship (Fable 5 is),
        # so its default is off as well.
        out = _resolve_thinking_for_model("anthropic/claude-opus-4-8")
        assert out == {}

    def test_model_without_thinking_metadata_returns_empty(self, monkeypatch):
        monkeypatch.delenv("RAGBOT_THINKING_EFFORT", raising=False)
        # GPT-5.6 Luna carries no thinking block in engines.yaml.
        out = _resolve_thinking_for_model("openai/gpt-5.6-luna")
        assert out == {}

    def test_per_call_override_for_pre_4_7_claude_uses_reasoning_effort(self, monkeypatch):
        # Pre-4.7 Claude (Haiku 4.5, extended thinking) still uses
        # reasoning_effort via LiteLLM's mapper, and extended thinking on
        # Anthropic requires temperature=1.
        monkeypatch.delenv("RAGBOT_THINKING_EFFORT", raising=False)
        out = _resolve_thinking_for_model("anthropic/claude-haiku-4-5-20251001", requested_effort="high")
        assert out == {"reasoning_effort": "high", "temperature": 1.0}

    def test_per_call_off_disables_flagship_default(self, monkeypatch):
        monkeypatch.delenv("RAGBOT_THINKING_EFFORT", raising=False)
        out = _resolve_thinking_for_model("anthropic/claude-fable-5", requested_effort="off")
        assert out == {}

    def test_env_var_overrides_engines_yaml_default(self, monkeypatch):
        # Haiku defaults to off; env says low. (Haiku is pre-4.7, so the
        # effort level stays visible in the emitted params.)
        monkeypatch.setenv("RAGBOT_THINKING_EFFORT", "low")
        out = _resolve_thinking_for_model("anthropic/claude-haiku-4-5-20251001")
        assert out == {"reasoning_effort": "low", "temperature": 1.0}

    def test_per_call_override_beats_env_var(self, monkeypatch):
        monkeypatch.setenv("RAGBOT_THINKING_EFFORT", "high")
        out = _resolve_thinking_for_model("anthropic/claude-haiku-4-5-20251001", requested_effort="low")
        assert out == {"reasoning_effort": "low", "temperature": 1.0}

    def test_models_without_thinking_metadata_ignore_overrides(self, monkeypatch):
        monkeypatch.delenv("RAGBOT_THINKING_EFFORT", raising=False)
        # Luna has no thinking block — even an explicit high should be silent.
        out = _resolve_thinking_for_model(
            "openai/gpt-5.6-luna",
            requested_effort="high",
        )
        assert out == {}

    def test_unknown_effort_value_falls_through_to_engines_default(self, monkeypatch):
        monkeypatch.delenv("RAGBOT_THINKING_EFFORT", raising=False)
        out = _resolve_thinking_for_model(
            "anthropic/claude-fable-5",
            requested_effort="ridiculous",
        )
        # Falls back to engines.yaml default (flagship → adaptive thinking
        # shape; Fable 5 is 4.8+ so no temperature is sent).
        assert out == {"thinking": {"type": "adaptive"}}

    def test_gemini_flagship_defaults_to_medium(self, monkeypatch):
        monkeypatch.delenv("RAGBOT_THINKING_EFFORT", raising=False)
        out = _resolve_thinking_for_model("gemini/gemini-3.1-pro-preview")
        # Non-Anthropic provider — no temperature override needed.
        assert out == {"reasoning_effort": "medium"}

    def test_openai_flagship_defaults_to_medium(self, monkeypatch):
        monkeypatch.delenv("RAGBOT_THINKING_EFFORT", raising=False)
        out = _resolve_thinking_for_model("openai/gpt-5.6-sol")
        assert out == {"reasoning_effort": "medium"}

    def test_openai_non_flagship_defaults_to_minimal(self, monkeypatch):
        """GPT-5.6 Terra (non-flagship, `modes: [minimal, low, medium, high]`,
        no `off`) should default to the lowest listed mode so the provider's
        own reasoning default doesn't consume the output-token budget on
        long-context calls."""
        monkeypatch.delenv("RAGBOT_THINKING_EFFORT", raising=False)
        out = _resolve_thinking_for_model("openai/gpt-5.6-terra")
        assert out == {"reasoning_effort": "minimal"}

    def test_gemini_non_flagship_defaults_to_minimal(self, monkeypatch):
        """Gemini 3 Flash (non-flagship, same `modes:` shape as GPT-5.6 Terra)
        gets the same lowest-mode default treatment."""
        monkeypatch.delenv("RAGBOT_THINKING_EFFORT", raising=False)
        out = _resolve_thinking_for_model("gemini/gemini-3-flash-preview")
        assert out == {"reasoning_effort": "minimal"}

    def test_per_call_off_still_disables_non_flagship_with_modes(self, monkeypatch):
        """User can still pick ``off`` explicitly on a non-flagship GPT/Gemini
        model — the per-call override beats the engines.yaml default policy."""
        monkeypatch.delenv("RAGBOT_THINKING_EFFORT", raising=False)
        out = _resolve_thinking_for_model("openai/gpt-5.6-terra", requested_effort="off")
        assert out == {}
