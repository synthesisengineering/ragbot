"""LiteLLM backend (default).

Wraps :func:`litellm.completion`. Provides:

* Provider-agnostic completion via LiteLLM's OpenAI-compatible API.
* Handles the GPT-5.x ``max_completion_tokens`` parameter rename.
* Honours the Anthropic constraint that extended thinking requires
  ``temperature=1``.
* Routes Claude 4.7+ around LiteLLM's still-pre-adaptive thinking shape
  (LiteLLM <=1.83.13 emits ``thinking.type.enabled`` for newer Claude
  models, which the API now rejects). For 4.7+ we send the new
  ``thinking={"type": "adaptive"}`` directly.
* Auto-applies Anthropic ``cache_control`` to the system prompt and any
  large reusable context blocks for prompt-cache discipline.
* Wraps every completion in an OTEL ``chat_completion_span`` with full
  GenAI-semantic attributes and emits cache-hit metrics off the response.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict

from ..observability import (
    chat_completion_span,
    record_llm_response,
)
from ..observability.attributes import (
    PROVIDER_ANTHROPIC,
    PROVIDER_GOOGLE,
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI,
    PROVIDER_UNKNOWN,
)
from .base import LLMBackend, LLMRequest, LLMResponse, LLMUnavailableError
from .cache_control import (
    CacheConfig,
    apply_cache_control_to_messages,
    extract_cache_metadata,
    is_anthropic_model,
    is_eligible_for_cache,
)

logger = logging.getLogger(__name__)


def _is_anthropic_model(model: str) -> bool:
    # Back-compat alias for the cache_control predicate.
    return is_anthropic_model(model)


def _provider_for(model: str) -> str:
    m = (model or "").lower()
    if m.startswith("anthropic/") or "claude" in m:
        return PROVIDER_ANTHROPIC
    if m.startswith("openai/") or m.startswith("gpt") or m.startswith("o1") or m.startswith("o3"):
        return PROVIDER_OPENAI
    if m.startswith("gemini/") or "gemini" in m:
        return PROVIDER_GOOGLE
    if m.startswith("ollama/") or m.startswith("ollama_chat/"):
        return PROVIDER_OLLAMA
    return PROVIDER_UNKNOWN


# Version-aware Claude generation predicates are shared across backends and
# the chat core — see synthesis_engine.llm.base. (A hardcoded substring list
# here silently regressed the moment engines.yaml moved past the 4.7 line.)
from .base import claude_rejects_temperature as _claude_rejects_temperature
from .base import is_claude_4_7_or_newer as _is_claude_4_7_or_newer


def _build_completion_kwargs(req: LLMRequest) -> Dict[str, Any]:
    """Translate a generic LLMRequest into litellm.completion kwargs.

    Applies Anthropic cache_control when eligible. The cache_control
    rewrite is intentionally upstream of the rest of the kwargs builder
    so the system prompt and large context blocks carry their cache
    markers no matter what other knobs are set downstream.
    """

    cache_cfg = CacheConfig.from_extra(req.extra)
    messages = req.messages
    if is_eligible_for_cache(req.model, cache_cfg):
        messages, _stats = apply_cache_control_to_messages(messages, cache_cfg)

    kwargs: Dict[str, Any] = {
        "model": req.model,
        "messages": messages,
        "api_key": req.api_key,
    }
    if req.temperature is not None and not (
        _is_anthropic_model(req.model) and _claude_rejects_temperature(req.model)
    ):
        # Claude 4.8+ / 5.x rejects the temperature parameter outright.
        kwargs["temperature"] = req.temperature

    # GPT-5.x uses max_completion_tokens, not max_tokens.
    model_lower = req.model.lower()
    if "gpt-5" in model_lower or "gpt5" in model_lower:
        kwargs["max_completion_tokens"] = req.max_tokens
    else:
        kwargs["max_tokens"] = req.max_tokens

    # Reasoning / thinking handling.
    if req.thinking is not None:
        # Caller provided the provider-native shape; pass it through.
        kwargs["thinking"] = req.thinking
        if _is_anthropic_model(req.model) and not _claude_rejects_temperature(req.model):
            kwargs["temperature"] = 1.0
    elif req.reasoning_effort:
        if _is_anthropic_model(req.model) and _is_claude_4_7_or_newer(req.model):
            # LiteLLM through 1.83.13 sends the older ``thinking.type.enabled``
            # shape via reasoning_effort. Claude 4.7+ rejects that — use
            # the new adaptive shape directly.
            kwargs["thinking"] = {"type": "adaptive"}
            if not _claude_rejects_temperature(req.model):
                # 4.7 requires temp=1 alongside thinking; 4.8+/5.x rejects
                # the parameter entirely.
                kwargs["temperature"] = 1.0
        else:
            kwargs["reasoning_effort"] = req.reasoning_effort
            if _is_anthropic_model(req.model):
                # Claude 4.5/4.6 with reasoning_effort still requires temp=1.
                kwargs["temperature"] = 1.0

    # Free-form passthrough (last so it can override anything above).
    # We strip the substrate-internal cache_control config keys so they
    # don't reach LiteLLM as unknown kwargs.
    if req.extra:
        passthrough = {
            k: v
            for k, v in req.extra.items()
            if k not in {
                "cache_control_enabled",
                "cache_min_block_tokens",
                "cache_ttl",
            }
        }
        if passthrough:
            kwargs.update(passthrough)

    return kwargs


class LiteLLMBackend(LLMBackend):
    backend_name = "litellm"

    def __init__(self) -> None:
        try:
            import litellm  # type: ignore
        except ImportError as exc:
            raise LLMUnavailableError(
                "litellm is not installed; pip install -r requirements.txt"
            ) from exc

        self._litellm = litellm
        # Drop unsupported params at the LiteLLM layer; we still build kwargs
        # carefully ourselves above to avoid round-tripping wrong values.
        litellm.drop_params = True

    def complete(self, request: LLMRequest) -> LLMResponse:
        cache_cfg = CacheConfig.from_extra(request.extra)
        provider = _provider_for(request.model)

        with chat_completion_span(
            model=request.model,
            provider=provider,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=False,
            backend_name=self.backend_name,
            reasoning_effort=request.reasoning_effort,
            cache_control_enabled=(
                cache_cfg.enabled if is_anthropic_model(request.model) else None
            ),
        ) as span:
            kwargs = _build_completion_kwargs(request)
            response = self._litellm.completion(**kwargs)

            # LiteLLM responses behave like dicts but expose attribute access
            # for the OpenAI-compatible nested fields. Use attribute access
            # (with dict fallback) so we work across versions.
            choices = getattr(response, "choices", None) or response.get("choices", [])
            choice = choices[0] if choices else None

            text = ""
            finish_reason = None
            if choice is not None:
                # ``choice.message.content`` (object access) is the canonical path.
                msg = getattr(choice, "message", None)
                if msg is None and isinstance(choice, dict):
                    msg = choice.get("message")
                if msg is not None:
                    content = getattr(msg, "content", None)
                    if content is None and isinstance(msg, dict):
                        content = msg.get("content")
                    text = content or ""
                finish_reason = (
                    getattr(choice, "finish_reason", None)
                    or (choice.get("finish_reason") if isinstance(choice, dict) else None)
                )

            usage_obj = getattr(response, "usage", None) or response.get("usage", {})
            prompt_tok = getattr(usage_obj, "prompt_tokens", None)
            if prompt_tok is None and isinstance(usage_obj, dict):
                prompt_tok = usage_obj.get("prompt_tokens")
            completion_tok = getattr(usage_obj, "completion_tokens", None)
            if completion_tok is None and isinstance(usage_obj, dict):
                completion_tok = usage_obj.get("completion_tokens")
            total_tok = getattr(usage_obj, "total_tokens", None)
            if total_tok is None and isinstance(usage_obj, dict):
                total_tok = usage_obj.get("total_tokens")

            # Cache accounting (Anthropic). Other providers will report
            # zeros and the cache metrics simply will not increment.
            cache_meta = extract_cache_metadata(usage_obj)

            response_model = (
                getattr(response, "model", None)
                or response.get("model", request.model)
            )

            record_llm_response(
                span,
                model=request.model,
                provider=provider,
                response_model=response_model,
                # The Anthropic API reports *uncached* input tokens
                # under input_tokens / prompt_tokens. The cache_meta
                # extractor uses the same value, so passing it once
                # keeps the recorder's hit-ratio math correct.
                input_tokens=cache_meta.uncached_input_tokens or int(prompt_tok or 0),
                output_tokens=cache_meta.output_tokens or int(completion_tok or 0),
                finish_reason=finish_reason,
                cache_read_tokens=cache_meta.cache_read_input_tokens,
                cache_creation_tokens=cache_meta.cache_creation_input_tokens,
            )

            usage_dict: Dict[str, int] = {
                "prompt_tokens": int(prompt_tok or 0),
                "completion_tokens": int(completion_tok or 0),
                "total_tokens": int(total_tok or 0),
            }
            if cache_meta.cache_read_input_tokens:
                usage_dict["cache_read_input_tokens"] = cache_meta.cache_read_input_tokens
            if cache_meta.cache_creation_input_tokens:
                usage_dict["cache_creation_input_tokens"] = cache_meta.cache_creation_input_tokens

            return LLMResponse(
                text=text,
                model=response_model,
                backend=self.backend_name,
                finish_reason=finish_reason,
                usage=usage_dict,
            )

    def stream(
        self,
        request: LLMRequest,
        on_chunk: Callable[[str], None],
    ) -> str:
        cache_cfg = CacheConfig.from_extra(request.extra)
        provider = _provider_for(request.model)
        with chat_completion_span(
            model=request.model,
            provider=provider,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=True,
            backend_name=self.backend_name,
            reasoning_effort=request.reasoning_effort,
            cache_control_enabled=(
                cache_cfg.enabled if is_anthropic_model(request.model) else None
            ),
        ) as span:
            kwargs = _build_completion_kwargs(request)
            response = self._litellm.completion(**kwargs, stream=True)
            chunks = []
            final_usage = None
            for chunk in response:
                if not chunk:
                    continue
                # Capture usage when it arrives on the final chunk
                # (LiteLLM emits ``usage`` on the last delta).
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    final_usage = chunk_usage
                if not getattr(chunk, "choices", None):
                    continue
                delta = chunk.choices[0].delta if chunk.choices else None
                text = getattr(delta, "content", None)
                if text:
                    chunks.append(text)
                    try:
                        on_chunk(text)
                    except Exception as exc:  # pragma: no cover
                        logger.warning("on_chunk callback raised: %s", exc)
            assembled = "".join(chunks)
            cache_meta = extract_cache_metadata(final_usage)
            record_llm_response(
                span,
                model=request.model,
                provider=provider,
                input_tokens=cache_meta.uncached_input_tokens,
                output_tokens=cache_meta.output_tokens,
                cache_read_tokens=cache_meta.cache_read_input_tokens,
                cache_creation_tokens=cache_meta.cache_creation_input_tokens,
            )
            return assembled

    def healthcheck(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {"backend": self.backend_name, "ok": True}
        try:
            import litellm  # type: ignore

            ver = getattr(litellm, "__version__", None) or getattr(
                litellm, "version", None,
            )
            if ver:
                info["litellm_version"] = str(ver)
        except Exception:
            pass
        return info
