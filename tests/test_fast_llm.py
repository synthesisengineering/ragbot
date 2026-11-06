"""Retrieval helper calls go through the LLM backend at the least reasoning.

Query planning, expansion, HyDE, reranking and verification ask the bulk-tier
model for a few lines of JSON or prose. On 2026-11-06 a local Gemma 4 12B,
which thinks by default, spent the reranker's whole 500-token budget
reasoning and returned an empty reply, so reranking never ran. The helper
also bypassed the backend: it sent sampling parameters to every provider and
looked API keys up under the environment-variable name, which the keystore
does not use. These tests pin the request the helper now builds.
"""

from __future__ import annotations

import logging
import os
import sys

import pytest

_SRC = os.path.join(os.path.dirname(__file__), '..', 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import rag  # noqa: E402
import synthesis_engine.keystore as keystore  # noqa: E402
import synthesis_engine.llm as llm  # noqa: E402
from synthesis_engine.config import get_model_info  # noqa: E402
from synthesis_engine.llm import LLMResponse, OutputBudgetExhaustedError  # noqa: E402
from synthesis_engine.llm.litellm_backend import _build_completion_kwargs  # noqa: E402
from synthesis_engine.llm.model_parameters import least_reasoning_effort  # noqa: E402


class _RecordingBackend:
    """Stands in for the configured backend and records each request."""

    backend_name = 'recording'

    def __init__(self, reply='{"scores": [7]}', error=None):
        self.requests = []
        self._reply = reply
        self._error = error

    def complete(self, request):
        self.requests.append(request)
        if self._error is not None:
            raise self._error
        return LLMResponse(text=self._reply, model=request.model, backend=self.backend_name)


@pytest.fixture
def backend(monkeypatch):
    fake = _RecordingBackend()
    monkeypatch.setattr(llm, 'get_llm_backend', lambda: fake)
    return fake


@pytest.fixture
def keys(monkeypatch):
    asked = []

    def get_api_key(provider, workspace=None):
        asked.append((provider, workspace))
        return None if provider == 'ollama' else f'key-for-{provider}'

    monkeypatch.setattr(keystore, 'get_api_key', get_api_key)
    return asked


class TestLeastReasoningEffort:
    @pytest.mark.parametrize('model, effort', [
        ('openai/gpt-6-luna', 'none'),
        ('anthropic/claude-haiku-5-5', 'low'),
        ('gemini/gemini-3.5-flash-lite', 'minimal'),
        ('ollama_chat/gemma4:12b', 'none'),
        ('ollama_chat/gemma4:e4b', 'none'),
    ])
    def test_bulk_models_get_their_least_reasoning(self, model, effort):
        assert least_reasoning_effort(get_model_info(model)) == effort

    def test_model_without_thinking_support_gets_no_parameter(self):
        assert least_reasoning_effort({'id': 'ollama_chat/example:1b'}) is None
        assert least_reasoning_effort({'thinking': {'supported': False}}) is None

    def test_unknown_strict_mode_is_an_error(self):
        info = {'id': 'x/y', 'thinking': {'supported': True, 'strict': True, 'modes': ['low', 'turbo']}}
        with pytest.raises(ValueError):
            least_reasoning_effort(info)


class TestFastLlmRequest:
    def test_local_helper_switches_thinking_off(self, backend, keys):
        reply = rag._call_fast_llm('rank these', user_model='ollama_chat/gemma4:26b', workspace='personal')

        assert reply == '{"scores": [7]}'
        (request,) = backend.requests
        assert request.model == 'ollama_chat/gemma4:12b'
        assert request.reasoning_effort == 'none'
        assert request.max_tokens == get_model_info('ollama_chat/gemma4:12b')['default_max_tokens']
        assert request.api_key is None
        assert keys == [('ollama', 'personal')]
        # LiteLLM receives the effort that maps to Ollama's think: false.
        kwargs = _build_completion_kwargs(request)
        assert kwargs['reasoning_effort'] == 'none'
        assert kwargs['temperature'] == 0.3

    def test_anthropic_helper_uses_keystore_key_and_no_sampling(self, backend, keys):
        rag._call_fast_llm('plan this', user_model='anthropic/claude-sonnet-5-5', workspace='personal')

        (request,) = backend.requests
        assert request.model == 'anthropic/claude-haiku-5-5'
        assert request.api_key == 'key-for-anthropic'
        assert keys == [('anthropic', 'personal')]
        kwargs = _build_completion_kwargs(request)
        assert 'temperature' not in kwargs
        assert kwargs['thinking'] == {'type': 'adaptive'}
        assert kwargs['output_config'] == {'effort': 'low'}

    def test_openai_helper_reasons_least(self, backend, keys):
        rag._call_fast_llm('expand this', user_model='openai/gpt-6.1-sol')

        (request,) = backend.requests
        assert request.model == 'openai/gpt-6-luna'
        assert request.api_key == 'key-for-openai'
        assert _build_completion_kwargs(request)['reasoning_effort'] == 'none'

    def test_reply_is_stripped(self, monkeypatch, keys):
        monkeypatch.setattr(llm, 'get_llm_backend', lambda: _RecordingBackend(reply='  {"a": 1}\n'))
        assert rag._call_fast_llm('x', user_model='ollama_chat/gemma4:26b') == '{"a": 1}'

    def test_exhausted_budget_returns_none_and_logs_the_cause(self, monkeypatch, keys, caplog):
        error = OutputBudgetExhaustedError('gemma4:12b used its whole 4096-token output budget')
        monkeypatch.setattr(llm, 'get_llm_backend', lambda: _RecordingBackend(error=error))

        with caplog.at_level(logging.WARNING, logger='rag'):
            assert rag._call_fast_llm('x', user_model='ollama_chat/gemma4:26b') is None

        assert 'ollama_chat/gemma4:12b' in caplog.text
        assert 'output budget' in caplog.text


def test_litellm_backend_suppresses_debug_banners(monkeypatch):
    import litellm
    from synthesis_engine.llm.litellm_backend import LiteLLMBackend

    monkeypatch.setattr(litellm, 'suppress_debug_info', False)
    monkeypatch.setattr(litellm, 'drop_params', litellm.drop_params)
    LiteLLMBackend()
    assert litellm.suppress_debug_info is True


def test_litellm_sends_ollama_think_false_for_none():
    """The fix relies on this LiteLLM mapping; pin it so an upgrade that
    changes it fails here rather than in a live chat."""
    from litellm.llms.ollama.chat.transformation import OllamaChatConfig

    params = OllamaChatConfig().map_openai_params(
        non_default_params={'reasoning_effort': 'none'},
        optional_params={},
        model='gemma4:12b',
        drop_params=False,
    )
    assert params['think'] is False
