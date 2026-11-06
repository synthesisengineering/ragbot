"""Span context managers + functional recorders for synthesis_engine.

Each public function below produces a span with the right OTEL GenAI
attributes pre-populated, plus the synthesis-specific extensions. Spans
auto-record exceptions and set the span status to ERROR on raise; on
success the caller can call the matching ``record_*`` helper to push
the response/result data onto the span and into the metric registry.

Usage pattern (LLM call):

    with chat_completion_span(
        model="anthropic/claude-opus-5-5",
        provider="anthropic",
        max_tokens=4096,
        temperature=1.0,
    ) as span:
        resp = backend.complete(request)
        record_llm_response(
            span,
            input_tokens=resp.usage.get("prompt_tokens"),
            output_tokens=resp.usage.get("completion_tokens"),
            finish_reason=resp.finish_reason,
            cache_read_tokens=resp.usage.get("cache_read_input_tokens"),
            cache_creation_tokens=resp.usage.get("cache_creation_input_tokens"),
        )

The recorder is split from the context manager so callers can do work
inside the span (e.g., post-process the response) before the recorder
is called. It also lets the recorder be a no-op-safe function: passing
``None`` for fields the backend did not provide is supported.
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any, Dict, Iterator, Optional

from .attributes import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_MAX_TOKENS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_STREAM,
    GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_TOOL_NAME,
    GEN_AI_TOOL_TYPE,
    GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
    GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    OP_CHAT,
    OP_RETRIEVAL,
    SYNTHESIS_AGENT_ITERATION,
    SYNTHESIS_CACHE_CONTROL_ENABLED,
    SYNTHESIS_CACHE_HIT_RATIO,
    SYNTHESIS_GUARDRAIL_NAME,
    SYNTHESIS_GUARDRAIL_OUTCOME,
    SYNTHESIS_LLM_BACKEND,
    SYNTHESIS_REASONING_EFFORT,
    SYNTHESIS_RETRIEVAL_BACKEND,
    SYNTHESIS_RETRIEVAL_CONTENT_TYPE,
    SYNTHESIS_RETRIEVAL_K,
    SYNTHESIS_RETRIEVAL_QUERY_LENGTH,
    SYNTHESIS_RETRIEVAL_RESULT_COUNT,
    SYNTHESIS_RETRIEVAL_TOP_SCORE,
    SYNTHESIS_RETRIEVER_TIER,
    SYNTHESIS_SESSION_ID,
    SYNTHESIS_TOOL_NAME,
    SYNTHESIS_WORKSPACE,
)
from .metrics import _record_cache_event, _record_otel_and_prom
from .tracer import get_tracer

# Synthesis-specific span attribute for the chat surface (used by chat_request_span).
# Kept as a string literal because attributes.py reserves the SYNTHESIS_* namespace
# for declared, stable extensions; chat-request is a router-level wrap and lives
# on the consumer side of the boundary.
_CHAT_USE_RAG_ATTR = "synthesis.chat.use_rag"
_CHAT_STREAM_ATTR = "synthesis.chat.stream"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Span status helpers
# ---------------------------------------------------------------------------


def _set_span_error(span, exc: BaseException) -> None:
    try:
        from opentelemetry.trace import Status, StatusCode

        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR, str(exc)))
    except Exception:  # pragma: no cover - defensive
        pass


def _set_span_ok(span) -> None:
    try:
        from opentelemetry.trace import Status, StatusCode

        span.set_status(Status(StatusCode.OK))
    except Exception:  # pragma: no cover - defensive
        pass


def _strip_none(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


# ---------------------------------------------------------------------------
# chat_request_span — router-level parent span
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def chat_request_span(
    *,
    workspace: Optional[str] = None,
    model: Optional[str] = None,
    use_rag: Optional[bool] = None,
    stream: Optional[bool] = None,
    session_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Iterator[Any]:
    """Wrap an incoming chat API request with a root-level span.

    The chat endpoint (``POST /api/chat``) handles a single user turn:
    retrieval, prompt construction, LLM completion, and response framing.
    Each of those substeps emits its own span (:func:`retrieval_span`,
    :func:`chat_completion_span`, etc.) — without a parent context they
    end up as disconnected single-span traces. ``chat_request_span``
    provides the parent so the full request shows up as one trace tree
    in Jaeger / Honeycomb / Phoenix.

    The span is named ``chat.request``. Attributes:

    * ``synthesis.workspace`` — the workspace the user selected.
    * ``gen_ai.request.model`` — the requested model id.
    * ``synthesis.chat.use_rag`` — whether retrieval is enabled for this turn.
    * ``synthesis.chat.stream`` — whether the response is streamed.
    * ``synthesis.session.id`` — optional session correlation id.

    Use as a context manager at the router boundary; the inner spans
    propagate automatically via OTEL's active-context machinery.
    """

    tracer = get_tracer("synthesis_engine.chat")
    span_name = "chat.request"

    attrs: Dict[str, Any] = {}
    if workspace:
        attrs[SYNTHESIS_WORKSPACE] = workspace
    if model:
        attrs[GEN_AI_REQUEST_MODEL] = model
    if use_rag is not None:
        attrs[_CHAT_USE_RAG_ATTR] = bool(use_rag)
    if stream is not None:
        attrs[_CHAT_STREAM_ATTR] = bool(stream)
    if session_id:
        attrs[SYNTHESIS_SESSION_ID] = session_id
    if extra:
        attrs.update(extra)

    with tracer.start_as_current_span(span_name, attributes=attrs) as span:
        try:
            yield span
            _set_span_ok(span)
        except BaseException as exc:
            _set_span_error(span, exc)
            raise


# ---------------------------------------------------------------------------
# chat_completion_span
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def chat_completion_span(
    model: str,
    provider: str,
    *,
    operation: str = OP_CHAT,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    stream: Optional[bool] = None,
    backend_name: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    session_id: Optional[str] = None,
    cache_control_enabled: Optional[bool] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Iterator[Any]:
    """Wrap an LLM completion call with a GenAI-conformant span.

    The span is named per the OTEL GenAI spec: ``"<operation> <model>"``
    (e.g., ``"chat anthropic/claude-opus-5-5"``).

    Parameters mirror the LLMRequest data contract; only the ones the
    caller has are required. Unknown fields can be passed via ``extra``.

    The decorated block has access to the span instance via ``as span:`` and
    should call :func:`record_llm_response` with the response data to
    populate the usage/cache attributes.
    """

    tracer = get_tracer("synthesis_engine.llm")
    span_name = f"{operation} {model}"
    started = time.monotonic()

    attrs: Dict[str, Any] = {
        GEN_AI_OPERATION_NAME: operation,
        GEN_AI_PROVIDER_NAME: provider,
        GEN_AI_REQUEST_MODEL: model,
        # gen_ai.system is the deprecated alias; keep emitting it for
        # backends that still consume it (Datadog v1.36 era, some Phoenix
        # versions). The OTEL spec keeps both during the transition.
        GEN_AI_SYSTEM: provider,
    }
    if max_tokens is not None:
        attrs[GEN_AI_REQUEST_MAX_TOKENS] = int(max_tokens)
    if temperature is not None:
        attrs[GEN_AI_REQUEST_TEMPERATURE] = float(temperature)
    if stream is not None:
        attrs[GEN_AI_REQUEST_STREAM] = bool(stream)
    if backend_name is not None:
        attrs[SYNTHESIS_LLM_BACKEND] = backend_name
    if reasoning_effort:
        attrs[SYNTHESIS_REASONING_EFFORT] = reasoning_effort
    if session_id:
        attrs[SYNTHESIS_SESSION_ID] = session_id
    if cache_control_enabled is not None:
        attrs[SYNTHESIS_CACHE_CONTROL_ENABLED] = bool(cache_control_enabled)
    if extra:
        attrs.update(extra)

    with tracer.start_as_current_span(span_name, attributes=attrs) as span:
        outcome = "ok"
        try:
            yield span
            _set_span_ok(span)
        except BaseException as exc:
            outcome = "error"
            _set_span_error(span, exc)
            raise
        finally:
            duration = time.monotonic() - started
            _record_otel_and_prom(
                "llm_operation_duration",
                duration,
                otel_attrs={
                    GEN_AI_PROVIDER_NAME: provider,
                    GEN_AI_REQUEST_MODEL: model,
                    GEN_AI_OPERATION_NAME: operation,
                },
                prom_labels=(provider, model, operation),
            )
            _record_otel_and_prom(
                "llm_requests",
                1,
                otel_attrs={
                    GEN_AI_PROVIDER_NAME: provider,
                    GEN_AI_REQUEST_MODEL: model,
                    GEN_AI_OPERATION_NAME: operation,
                    "outcome": outcome,
                },
                prom_labels=(provider, model, operation, outcome),
            )


# ---------------------------------------------------------------------------
# record_llm_response — usage + cache accounting
# ---------------------------------------------------------------------------


def record_llm_response(
    span: Any,
    *,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    finish_reason: Optional[str] = None,
    cache_read_tokens: Optional[int] = None,
    cache_creation_tokens: Optional[int] = None,
    response_model: Optional[str] = None,
) -> None:
    """Stamp response data onto the span + record metrics.

    Cache accounting:
      input_tokens reported by Anthropic is the **uncached** input token
      count. cache_read_tokens and cache_creation_tokens are reported
      separately. The hit-ratio metric is computed from all three so the
      ratio is meaningful regardless of how the provider partitions the
      counts.

    For OpenAI / Google models without cache accounting, the cache fields
    will be None and the cache metrics simply do not get incremented.
    """

    if input_tokens is not None:
        span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, int(input_tokens))
    if output_tokens is not None:
        span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, int(output_tokens))
    if response_model:
        span.set_attribute(GEN_AI_RESPONSE_MODEL, response_model)
    if finish_reason:
        # OTEL spec is finish_reasons: string[] — wrap in a list.
        span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, [str(finish_reason)])
    if cache_read_tokens is not None:
        span.set_attribute(GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, int(cache_read_tokens))
    if cache_creation_tokens is not None:
        span.set_attribute(
            GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS, int(cache_creation_tokens),
        )

    # Cache hit ratio: read / (read + creation + uncached_input).
    if any(
        v is not None for v in (cache_read_tokens, cache_creation_tokens, input_tokens)
    ):
        read = int(cache_read_tokens or 0)
        creation = int(cache_creation_tokens or 0)
        uncached = int(input_tokens or 0)
        total = read + creation + uncached
        if total > 0:
            ratio = read / total
            span.set_attribute(SYNTHESIS_CACHE_HIT_RATIO, ratio)

        # Push into the rolling cache stats deque and the cumulative counters.
        # We only push when a model is known.
        effective_model = (
            model
            or _safe_get_attr(span, GEN_AI_REQUEST_MODEL)
            or response_model
            or "unknown"
        )
        effective_provider = (
            provider
            or _safe_get_attr(span, GEN_AI_PROVIDER_NAME)
            or "unknown"
        )
        _record_cache_event(
            model=effective_model,
            cache_read_tokens=read,
            cache_creation_tokens=creation,
            uncached_input_tokens=uncached,
        )
        if read:
            _record_otel_and_prom(
                "cache_read_input_tokens",
                read,
                otel_attrs={GEN_AI_PROVIDER_NAME: effective_provider, GEN_AI_REQUEST_MODEL: effective_model},
                prom_labels=(effective_provider, effective_model),
            )
        if creation:
            _record_otel_and_prom(
                "cache_creation_input_tokens",
                creation,
                otel_attrs={GEN_AI_PROVIDER_NAME: effective_provider, GEN_AI_REQUEST_MODEL: effective_model},
                prom_labels=(effective_provider, effective_model),
            )
        if uncached:
            _record_otel_and_prom(
                "cache_uncached_input_tokens",
                uncached,
                otel_attrs={GEN_AI_PROVIDER_NAME: effective_provider, GEN_AI_REQUEST_MODEL: effective_model},
                prom_labels=(effective_provider, effective_model),
            )

    # Token-usage histogram (broken down by kind: input / output).
    effective_model = model or _safe_get_attr(span, GEN_AI_REQUEST_MODEL) or "unknown"
    effective_provider = provider or _safe_get_attr(span, GEN_AI_PROVIDER_NAME) or "unknown"
    if input_tokens is not None:
        _record_otel_and_prom(
            "llm_token_usage",
            int(input_tokens),
            otel_attrs={
                GEN_AI_PROVIDER_NAME: effective_provider,
                GEN_AI_REQUEST_MODEL: effective_model,
                "token_kind": "input",
            },
            prom_labels=(effective_provider, effective_model, "input"),
        )
    if output_tokens is not None:
        _record_otel_and_prom(
            "llm_token_usage",
            int(output_tokens),
            otel_attrs={
                GEN_AI_PROVIDER_NAME: effective_provider,
                GEN_AI_REQUEST_MODEL: effective_model,
                "token_kind": "output",
            },
            prom_labels=(effective_provider, effective_model, "output"),
        )


def record_cache_hit(
    span: Any,
    *,
    model: str,
    provider: str,
    cache_read_tokens: int,
    cache_creation_tokens: int = 0,
    uncached_input_tokens: int = 0,
) -> None:
    """Standalone cache event recorder.

    Used for paths where the LLM response has been collected upstream
    (e.g., from a stream's final event) and we want to record just the
    cache accounting without going through record_llm_response.
    """

    total = cache_read_tokens + cache_creation_tokens + uncached_input_tokens
    if total > 0:
        span.set_attribute(
            SYNTHESIS_CACHE_HIT_RATIO,
            cache_read_tokens / total,
        )
    _record_cache_event(
        model=model,
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=cache_creation_tokens,
        uncached_input_tokens=uncached_input_tokens,
    )


# ---------------------------------------------------------------------------
# retrieval_span
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def retrieval_span(
    workspace: str,
    *,
    query: Optional[str] = None,
    k: Optional[int] = None,
    tier: str = "vector",
    backend: Optional[str] = None,
    content_type: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Iterator[Any]:
    """Wrap a vector / keyword retrieval call.

    The query text is NOT recorded by default (privacy default; OTEL
    GenAI spec marks gen_ai.retrieval.query.text as Opt-In). The query
    length is recorded instead so query-distribution dashboards work
    without leaking content.

    A caller that explicitly opts in (e.g., for offline eval traces)
    can pass the query in ``extra`` under ``gen_ai.retrieval.query.text``.
    """

    tracer = get_tracer("synthesis_engine.retrieval")
    span_name = f"retrieval {workspace}"
    started = time.monotonic()

    attrs: Dict[str, Any] = {
        GEN_AI_OPERATION_NAME: OP_RETRIEVAL,
        SYNTHESIS_WORKSPACE: workspace,
        SYNTHESIS_RETRIEVER_TIER: tier,
    }
    if query is not None:
        attrs[SYNTHESIS_RETRIEVAL_QUERY_LENGTH] = len(query)
    if k is not None:
        attrs[SYNTHESIS_RETRIEVAL_K] = int(k)
    if backend:
        attrs[SYNTHESIS_RETRIEVAL_BACKEND] = backend
    if content_type:
        attrs[SYNTHESIS_RETRIEVAL_CONTENT_TYPE] = content_type
    if extra:
        attrs.update(extra)

    with tracer.start_as_current_span(span_name, attributes=attrs) as span:
        try:
            yield span
            _set_span_ok(span)
        except BaseException as exc:
            _set_span_error(span, exc)
            raise
        finally:
            duration = time.monotonic() - started
            _record_otel_and_prom(
                "retrieval_latency",
                duration,
                otel_attrs={
                    SYNTHESIS_WORKSPACE: workspace,
                    SYNTHESIS_RETRIEVER_TIER: tier,
                    SYNTHESIS_RETRIEVAL_BACKEND: backend or "unknown",
                },
                prom_labels=(workspace, tier, backend or "unknown"),
            )


def record_retrieval_result(
    span: Any,
    *,
    result_count: int,
    top_score: Optional[float] = None,
    workspace: Optional[str] = None,
    tier: Optional[str] = None,
) -> None:
    span.set_attribute(SYNTHESIS_RETRIEVAL_RESULT_COUNT, int(result_count))
    if top_score is not None:
        span.set_attribute(SYNTHESIS_RETRIEVAL_TOP_SCORE, float(top_score))

    workspace = workspace or _safe_get_attr(span, SYNTHESIS_WORKSPACE) or "unknown"
    tier = tier or _safe_get_attr(span, SYNTHESIS_RETRIEVER_TIER) or "unknown"
    _record_otel_and_prom(
        "retrieval_result_count",
        int(result_count),
        otel_attrs={
            SYNTHESIS_WORKSPACE: workspace,
            SYNTHESIS_RETRIEVER_TIER: tier,
        },
        prom_labels=(workspace, tier),
    )


# ---------------------------------------------------------------------------
# tool_span
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def tool_span(
    tool_name: str,
    *,
    tool_type: str = "function",
    upstream_tool_name: Optional[str] = None,
    call_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Iterator[Any]:
    tracer = get_tracer("synthesis_engine.tools")
    span_name = f"execute_tool {tool_name}"

    attrs: Dict[str, Any] = {
        GEN_AI_OPERATION_NAME: "execute_tool",
        GEN_AI_TOOL_NAME: upstream_tool_name or tool_name,
        GEN_AI_TOOL_TYPE: tool_type,
        SYNTHESIS_TOOL_NAME: tool_name,
    }
    if call_id:
        attrs["gen_ai.tool.call.id"] = call_id
    if extra:
        attrs.update(extra)

    with tracer.start_as_current_span(span_name, attributes=attrs) as span:
        outcome = "ok"
        try:
            yield span
            _set_span_ok(span)
        except BaseException as exc:
            outcome = "error"
            _set_span_error(span, exc)
            raise
        finally:
            _record_otel_and_prom(
                "tool_calls",
                1,
                otel_attrs={SYNTHESIS_TOOL_NAME: tool_name, "outcome": outcome},
                prom_labels=(tool_name, outcome),
            )


def record_tool_result(
    span: Any,
    *,
    success: bool = True,
    result_summary: Optional[str] = None,
) -> None:
    span.set_attribute("synthesis.tool.success", success)
    if result_summary:
        span.set_attribute("synthesis.tool.result_summary", result_summary)


# ---------------------------------------------------------------------------
# guardrail_span
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def guardrail_span(
    guardrail_name: str,
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Iterator[Any]:
    tracer = get_tracer("synthesis_engine.guardrails")
    span_name = f"guardrail {guardrail_name}"

    attrs: Dict[str, Any] = {SYNTHESIS_GUARDRAIL_NAME: guardrail_name}
    if extra:
        attrs.update(extra)

    with tracer.start_as_current_span(span_name, attributes=attrs) as span:
        outcome = "allow"
        try:
            yield span
            # The block sets the outcome via span attribute; we read it back.
            recorded = _safe_get_attr(span, SYNTHESIS_GUARDRAIL_OUTCOME)
            outcome = recorded or outcome
            _set_span_ok(span)
        except BaseException as exc:
            outcome = "error"
            _set_span_error(span, exc)
            raise
        finally:
            _record_otel_and_prom(
                "guardrail_outcomes",
                1,
                otel_attrs={
                    SYNTHESIS_GUARDRAIL_NAME: guardrail_name,
                    SYNTHESIS_GUARDRAIL_OUTCOME: outcome,
                },
                prom_labels=(guardrail_name, outcome),
            )


# ---------------------------------------------------------------------------
# agent_iteration_span
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def agent_iteration_span(
    iteration: int,
    *,
    session_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Iterator[Any]:
    tracer = get_tracer("synthesis_engine.agent")
    span_name = f"agent.iteration {iteration}"

    attrs: Dict[str, Any] = {SYNTHESIS_AGENT_ITERATION: int(iteration)}
    if session_id:
        attrs[SYNTHESIS_SESSION_ID] = session_id
    if extra:
        attrs.update(extra)

    with tracer.start_as_current_span(span_name, attributes=attrs) as span:
        try:
            yield span
            _set_span_ok(span)
        except BaseException as exc:
            _set_span_error(span, exc)
            raise
        finally:
            _record_otel_and_prom(
                "agent_iterations",
                1,
                otel_attrs={SYNTHESIS_SESSION_ID: session_id or "anonymous"},
                prom_labels=(session_id or "anonymous",),
            )


# ---------------------------------------------------------------------------
# Internal: safe attribute introspection
# ---------------------------------------------------------------------------


def _safe_get_attr(span: Any, key: str) -> Optional[str]:
    """Read an attribute off the active span, if accessible.

    OTEL's ReadableSpan exposes ``.attributes`` (a dict) on the SDK side;
    the API-only Span does not. Tests use the SDK side so this works
    there; production hot paths use this only for backfill fields when
    they aren't passed in directly.
    """

    try:
        attrs = getattr(span, "attributes", None)
        if attrs is not None:
            val = attrs.get(key)
            if val is not None:
                return str(val)
    except Exception:  # pragma: no cover - defensive
        pass
    return None


__all__ = [
    "chat_request_span",
    "chat_completion_span",
    "record_llm_response",
    "record_cache_hit",
    "retrieval_span",
    "record_retrieval_result",
    "tool_span",
    "record_tool_result",
    "guardrail_span",
    "agent_iteration_span",
]
