"""A completion that hits its output limit before writing a reply is an error.

Reasoning tokens count against ``max_tokens``. A reasoning model given a small
budget can spend all of it thinking and return empty content with a length
finish reason (observed on 2026-11-06 with a local Gemma 4 model and a
300-token budget). These tests pin that every backend path raises
``OutputBudgetExhaustedError`` instead of returning an empty reply, and that
``chat_stream`` passes a streaming failure to its caller instead of ending the
stream quietly.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

_SRC = os.path.join(os.path.dirname(__file__), '..', 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from synthesis_engine.exceptions import ChatError  # noqa: E402
from synthesis_engine.llm import LLMRequest, OutputBudgetExhaustedError  # noqa: E402
from synthesis_engine.llm.base import require_reply  # noqa: E402

_LOCAL_MODEL = 'ollama_chat/gemma4:26b'


def _request(model=_LOCAL_MODEL, max_tokens=300):
    return LLMRequest(model=model, messages=[{'role': 'user', 'content': 'hi'}],
                      max_tokens=max_tokens)


class _GoogleFinishReason:
    """Stands in for google-genai's FinishReason enum member."""

    def __init__(self, name):
        self.name = name


class TestRequireReply:
    @pytest.mark.parametrize('finish_reason', [
        'length', 'LENGTH', 'max_tokens', _GoogleFinishReason('MAX_TOKENS'),
    ])
    def test_empty_reply_at_the_output_limit_raises(self, finish_reason):
        with pytest.raises(OutputBudgetExhaustedError, match='300-token output budget'):
            require_reply('', finish_reason, _request())

    @pytest.mark.parametrize('finish_reason', ['length', 'max_tokens'])
    def test_partial_reply_at_the_limit_is_returned(self, finish_reason):
        # A truncated answer is still an answer; only an empty one is hidden.
        require_reply('Partial answer', finish_reason, _request())

    @pytest.mark.parametrize('finish_reason', [None, 'stop', 'end_turn', _GoogleFinishReason('STOP')])
    def test_other_finish_reasons_are_left_alone(self, finish_reason):
        require_reply('', finish_reason, _request())

    def test_message_names_the_model_and_the_remedy(self):
        with pytest.raises(OutputBudgetExhaustedError) as info:
            require_reply('', 'length', _request(max_tokens=512))
        message = str(info.value)
        assert _LOCAL_MODEL in message and '512-token' in message
        assert 'max_tokens' in message and 'thinking_effort' in message


# ---------------------------------------------------------------------------
# LiteLLM backend
# ---------------------------------------------------------------------------


def _litellm_backend(fake_completion):
    pytest.importorskip('litellm')
    from synthesis_engine.llm.litellm_backend import LiteLLMBackend

    backend = LiteLLMBackend()
    backend._litellm = SimpleNamespace(completion=fake_completion)
    return backend


def _litellm_response(content, finish_reason):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content),
                                 finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=25, completion_tokens=300, total_tokens=325),
        model=_LOCAL_MODEL,
    )


def _litellm_chunk(content=None, finish_reason=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content),
                                 finish_reason=finish_reason)],
        usage=None,
    )


class TestLiteLLMBackend:
    def test_complete_raises_when_reasoning_used_the_budget(self):
        backend = _litellm_backend(lambda **kw: _litellm_response('', 'length'))
        with pytest.raises(OutputBudgetExhaustedError):
            backend.complete(_request())

    def test_complete_returns_a_reply(self):
        backend = _litellm_backend(lambda **kw: _litellm_response('pgvector stores vectors.', 'stop'))
        assert backend.complete(_request()).text == 'pgvector stores vectors.'

    def test_stream_raises_when_no_text_arrived_before_the_limit(self):
        def completion(**kw):
            return iter([_litellm_chunk(), _litellm_chunk(finish_reason='length')])

        backend = _litellm_backend(completion)
        with pytest.raises(OutputBudgetExhaustedError):
            backend.stream(_request(), on_chunk=lambda text: None)

    def test_stream_returns_text_that_arrived(self):
        def completion(**kw):
            return iter([_litellm_chunk('Hello'), _litellm_chunk(finish_reason='length')])

        backend = _litellm_backend(completion)
        seen = []
        assert backend.stream(_request(), on_chunk=seen.append) == 'Hello'
        assert seen == ['Hello']


# ---------------------------------------------------------------------------
# Direct backend (OpenAI and Google paths; Anthropic shares the check)
# ---------------------------------------------------------------------------


def _direct_backend():
    from synthesis_engine.llm.direct_backend import DirectBackend

    return DirectBackend()


class TestDirectBackend:
    def test_openai_complete_raises_on_an_empty_length_stop(self):
        backend = _direct_backend()
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=''), finish_reason='length')],
            usage=None, model='gpt-test',
        )
        client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: response)))
        backend._get_openai_client = lambda api_key: client
        backend._build_openai_kwargs = lambda request: {}
        with pytest.raises(OutputBudgetExhaustedError):
            backend._openai_complete(_request(model='openai/gpt-test'))

    def test_openai_stream_raises_on_an_empty_length_stop(self):
        backend = _direct_backend()
        chunks = [
            SimpleNamespace(usage=None, choices=[SimpleNamespace(
                delta=SimpleNamespace(content=None), finish_reason=None)]),
            SimpleNamespace(usage=None, choices=[SimpleNamespace(
                delta=SimpleNamespace(content=None), finish_reason='length')]),
        ]
        client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: iter(chunks))))
        backend._get_openai_client = lambda api_key: client
        backend._build_openai_kwargs = lambda request: {}
        with pytest.raises(OutputBudgetExhaustedError):
            backend._openai_stream(_request(model='openai/gpt-test'), on_chunk=lambda t: None)

    def test_google_complete_reads_the_candidate_finish_reason(self):
        backend = _direct_backend()
        response = SimpleNamespace(
            text=None, usage_metadata=None,
            candidates=[SimpleNamespace(finish_reason=_GoogleFinishReason('MAX_TOKENS'))],
        )
        client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: response))
        backend._get_google_client = lambda api_key: client
        backend._build_google_request = lambda request: ('gemini-test', [], None)
        with pytest.raises(OutputBudgetExhaustedError):
            backend._google_complete(_request(model='gemini/gemini-test'))

    def test_google_stream_reads_the_last_candidate_finish_reason(self):
        backend = _direct_backend()
        chunks = [
            SimpleNamespace(text=None, usage_metadata=None, candidates=[]),
            SimpleNamespace(text=None, usage_metadata=None,
                            candidates=[SimpleNamespace(finish_reason=_GoogleFinishReason('MAX_TOKENS'))]),
        ]
        client = SimpleNamespace(models=SimpleNamespace(
            generate_content_stream=lambda **kw: iter(chunks)))
        backend._get_google_client = lambda api_key: client
        backend._build_google_request = lambda request: ('gemini-test', [], None)
        with pytest.raises(OutputBudgetExhaustedError):
            backend._google_stream(_request(model='gemini/gemini-test'), on_chunk=lambda t: None)


# ---------------------------------------------------------------------------
# chat_stream: failures in the streaming thread reach the caller
# ---------------------------------------------------------------------------


class _FailingStreamBackend:
    backend_name = 'fake'

    def __init__(self, error, before=()):
        self._error = error
        self._before = before

    def stream(self, request, on_chunk):
        for text in self._before:
            on_chunk(text)
        raise self._error


class TestChatStreamPropagatesFailures:
    def _stream(self, monkeypatch, backend):
        import synthesis_engine.llm as llm
        from ragbot import core

        monkeypatch.setattr(llm, 'get_llm_backend', lambda: backend)
        return core.chat_stream(
            'hi', model=_LOCAL_MODEL, use_rag=False, auto_load_instructions=False,
            max_tokens=300,
        )

    def test_exhausted_budget_surfaces_as_chat_error(self, monkeypatch):
        error = OutputBudgetExhaustedError('budget spent on reasoning')
        stream = self._stream(monkeypatch, _FailingStreamBackend(error))
        with pytest.raises(ChatError, match='budget spent on reasoning') as info:
            list(stream)
        assert info.value.__cause__ is error

    def test_text_before_a_failure_is_still_delivered(self, monkeypatch):
        stream = self._stream(
            monkeypatch, _FailingStreamBackend(RuntimeError('connection reset'), before=('Hel', 'lo')),
        )
        received = []
        with pytest.raises(ChatError, match='connection reset'):
            for text in stream:
                received.append(text)
        assert received == ['Hel', 'lo']
