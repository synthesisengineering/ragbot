"""Tests for the LLM-backend abstraction (Phase 3.1)."""

from __future__ import annotations

import os
import sys

import pytest

_SRC = os.path.join(os.path.dirname(__file__), '..', 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from synthesis_engine.llm import (  # noqa: E402
    LLMBackend,
    LLMRequest,
    LLMResponse,
    LLMUnavailableError,
    get_llm_backend,
    reset_llm_backend,
)


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


class TestLLMRequest:
    def test_defaults_are_safe(self):
        r = LLMRequest(model="anthropic/claude-sonnet-5-5", messages=[])
        assert r.temperature is None
        assert r.max_tokens == 4096
        assert r.api_key is None
        assert r.thinking is None
        assert r.reasoning_effort is None
        assert r.extra == {}


class TestLLMResponse:
    def test_response_has_expected_fields(self):
        r = LLMResponse(text="hi", model="x", backend="litellm")
        assert r.text == "hi"
        assert r.model == "x"
        assert r.backend == "litellm"
        assert r.usage == {}


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


class TestBackendSelection:
    def setup_method(self):
        reset_llm_backend()

    def teardown_method(self):
        reset_llm_backend()

    def test_default_is_litellm(self, monkeypatch):
        monkeypatch.delenv("RAGBOT_LLM_BACKEND", raising=False)
        b = get_llm_backend()
        assert isinstance(b, LLMBackend)
        assert b.backend_name == "litellm"

    def test_explicit_litellm(self, monkeypatch):
        monkeypatch.setenv("RAGBOT_LLM_BACKEND", "litellm")
        b = get_llm_backend()
        assert b.backend_name == "litellm"

    def test_unknown_value_falls_back_to_litellm(self, monkeypatch):
        monkeypatch.setenv("RAGBOT_LLM_BACKEND", "nonsense")
        b = get_llm_backend()
        assert b.backend_name == "litellm"

    def test_direct_backend_when_env_set(self, monkeypatch):
        monkeypatch.setenv("RAGBOT_LLM_BACKEND", "direct")
        b = get_llm_backend()
        # If anthropic/openai/google-genai are all installed, direct backend
        # constructs cleanly. If a SDK is missing, the resolver falls back to
        # litellm. Either is acceptable; both are valid LLMBackend instances.
        assert b.backend_name in {"direct", "litellm"}

    def test_singleton_caching(self, monkeypatch):
        monkeypatch.delenv("RAGBOT_LLM_BACKEND", raising=False)
        first = get_llm_backend()
        second = get_llm_backend()
        assert first is second

    def test_reset_clears_cache(self, monkeypatch):
        monkeypatch.delenv("RAGBOT_LLM_BACKEND", raising=False)
        first = get_llm_backend()
        reset_llm_backend()
        second = get_llm_backend()
        assert first is not second


# ---------------------------------------------------------------------------
# LiteLLM backend kwargs builder
# ---------------------------------------------------------------------------


class TestLiteLLMKwargsBuilder:
    """Verify the request → litellm.completion kwargs translation.

    Catalog models also pass through their verified request contract
    (tests/test_catalog_refresh.py). These cases cover the provider rules
    that apply to every model id, including ids engines.yaml does not list.
    """

    LOCAL = "ollama_chat/gemma4:31b"

    def _build(self, model=LOCAL, **overrides):
        from synthesis_engine.llm.litellm_backend import _build_completion_kwargs
        req = LLMRequest(model=model, messages=[{"role": "user", "content": "hi"}])
        for k, v in overrides.items():
            setattr(req, k, v)
        return _build_completion_kwargs(req)

    def test_max_tokens_for_non_openai(self):
        out = self._build(max_tokens=512)
        assert out["max_tokens"] == 512
        assert "max_completion_tokens" not in out

    @pytest.mark.parametrize("model", ["openai/gpt-6.1-sol", "openai/a-future-model"])
    def test_openai_takes_max_completion_tokens(self, model):
        out = self._build(model=model, max_tokens=512)
        assert out["max_completion_tokens"] == 512
        assert "max_tokens" not in out

    def test_temperature_passes_for_local_models(self):
        assert self._build(temperature=0.3)["temperature"] == 0.3

    def test_anthropic_never_receives_temperature(self):
        # API version 2026-10-21 rejects sampling parameters for every model.
        out = self._build(model="anthropic/a-future-claude", temperature=0.3)
        assert "temperature" not in out

    def test_anthropic_effort_uses_adaptive_thinking_and_output_config(self):
        out = self._build(model="anthropic/a-future-claude", reasoning_effort="medium")
        assert out["thinking"] == {"type": "adaptive"}
        assert out["output_config"] == {"effort": "medium"}
        assert "reasoning_effort" not in out
        assert "temperature" not in out

    def test_explicit_thinking_passes_through(self):
        out = self._build(model="anthropic/a-future-claude", thinking={"type": "adaptive"})
        assert out["thinking"] == {"type": "adaptive"}
        assert "temperature" not in out

    def test_gemini_reasoning_effort_does_not_force_temperature(self):
        out = self._build(model="gemini/gemini-4-argon", reasoning_effort="medium")
        assert out["reasoning_effort"] == "medium"
        assert "temperature" not in out

    def test_extra_kwargs_pass_through_and_override(self):
        out = self._build(extra={"top_p": 0.9, "max_tokens": 999})
        assert out["top_p"] == 0.9
        # extra overrides the default-built max_tokens.
        assert out["max_tokens"] == 999

    def test_openai_organization_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_ORGANIZATION", "org-payer123")
        assert self._build(model="openai/gpt-6.1-sol")["organization"] == "org-payer123"

    def test_openai_organization_absent_without_env(self, monkeypatch):
        monkeypatch.delenv("OPENAI_ORGANIZATION", raising=False)
        assert "organization" not in self._build(model="openai/gpt-6.1-sol")

    def test_openai_organization_ignored_for_other_providers(self, monkeypatch):
        monkeypatch.setenv("OPENAI_ORGANIZATION", "org-payer123")
        assert "organization" not in self._build(model="anthropic/claude-sonnet-5-5")

    def test_openai_organization_extra_override_wins(self, monkeypatch):
        monkeypatch.setenv("OPENAI_ORGANIZATION", "org-payer123")
        out = self._build(model="gpt-6.1-sol", extra={"organization": "org-explicit"})
        assert out["organization"] == "org-explicit"


# ---------------------------------------------------------------------------
# OpenAI organization env contract (issue #6)
# ---------------------------------------------------------------------------


class TestOpenAIOrganization:
    def test_blank_env_is_unset(self, monkeypatch):
        from synthesis_engine.llm.base import openai_organization_from_env
        monkeypatch.setenv("OPENAI_ORGANIZATION", "   ")
        assert openai_organization_from_env() is None

    def test_missing_env_is_unset(self, monkeypatch):
        from synthesis_engine.llm.base import openai_organization_from_env
        monkeypatch.delenv("OPENAI_ORGANIZATION", raising=False)
        assert openai_organization_from_env() is None

    def test_direct_backend_passes_organization(self, monkeypatch):
        import sys
        import types
        from synthesis_engine.llm.direct_backend import DirectBackend

        seen = {}

        class FakeOpenAI:
            def __init__(self, **kwargs):
                seen.update(kwargs)

        fake = types.ModuleType("openai")
        fake.OpenAI = FakeOpenAI
        monkeypatch.setitem(sys.modules, "openai", fake)
        monkeypatch.setenv("OPENAI_ORGANIZATION", "org-payer123")

        backend = DirectBackend()
        backend._get_openai_client(api_key="sk-test")
        assert seen == {"api_key": "sk-test", "organization": "org-payer123"}

    def test_direct_backend_omits_organization_without_env(self, monkeypatch):
        import sys
        import types
        from synthesis_engine.llm.direct_backend import DirectBackend

        seen = {}

        class FakeOpenAI:
            def __init__(self, **kwargs):
                seen.update(kwargs)

        fake = types.ModuleType("openai")
        fake.OpenAI = FakeOpenAI
        monkeypatch.setitem(sys.modules, "openai", fake)
        monkeypatch.delenv("OPENAI_ORGANIZATION", raising=False)

        backend = DirectBackend()
        backend._get_openai_client(api_key="sk-test")
        assert seen == {"api_key": "sk-test"}


# ---------------------------------------------------------------------------
# DirectBackend healthcheck (no live calls)
# ---------------------------------------------------------------------------


class TestDirectBackendHealthcheck:
    def test_reports_provider_availability(self, monkeypatch):
        monkeypatch.setenv("RAGBOT_LLM_BACKEND", "direct")
        reset_llm_backend()
        b = get_llm_backend()
        if b.backend_name != "direct":
            pytest.skip("DirectBackend not available; provider SDK missing.")
        h = b.healthcheck()
        assert h["backend"] == "direct"
        assert isinstance(h.get("providers"), dict)
        # At least one provider should be available in the test environment.
        assert any(p["available"] for p in h["providers"].values())
        reset_llm_backend()
