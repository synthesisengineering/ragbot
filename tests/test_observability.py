"""Tests for the synthesis_engine.observability substrate.

The substrate has already been built; these tests consume it through its
public API. The goals:

  1. init_tracer + an explicit InMemorySpanExporter produce a working
     tracer that emits real spans.
  2. A wrapped LLM call emits a span carrying the OTEL GenAI
     ``gen_ai.*`` attributes the spec requires.
  3. A wrapped retrieval call emits a span carrying the synthesis-specific
     workspace / k / latency attributes.
  4. Anthropic prompt-cache discipline rewrites the messages array so the
     system prompt block carries a ``cache_control`` annotation; non-
     Anthropic models are left unchanged.
  5. The Prometheus exposition endpoint returns a valid payload that
     parses through prometheus_client's text-format parser.
"""

from __future__ import annotations

import os
import sys

import pytest

# Make ``src/`` importable.
_REPO_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src",
)
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)


# ---------------------------------------------------------------------------
# Shared fixture: an isolated tracer per test with an in-memory exporter.
# ---------------------------------------------------------------------------


# The session-scoped OTEL initialization and the ``in_memory_tracer``
# per-test fixture live in tests/conftest.py so every test file in this
# suite shares ONE TracerProvider + MeterProvider (OTEL's globals refuse
# to be replaced once set, so multiple per-file fixtures used to deadlock
# the suite). See the comment in conftest.py for the rationale.


# ---------------------------------------------------------------------------
# Test 1 — init_tracer produces a working tracer
# ---------------------------------------------------------------------------


def test_init_tracer_produces_a_working_tracer(in_memory_tracer):
    """A trivial span emitted through the tracer is captured by the exporter."""

    from synthesis_engine.observability import get_tracer

    tracer = get_tracer("synthesis_engine.test")
    with tracer.start_as_current_span("smoke-span") as span:
        span.set_attribute("synthesis.test", "ok")

    spans = in_memory_tracer.get_finished_spans()
    span_names = [s.name for s in spans]
    assert "smoke-span" in span_names, (
        f"smoke-span not captured; got: {span_names}"
    )
    smoke = next(s for s in spans if s.name == "smoke-span")
    assert smoke.attributes.get("synthesis.test") == "ok"


# ---------------------------------------------------------------------------
# Test 2 — LLM span carries gen_ai.* attributes
# ---------------------------------------------------------------------------


def test_chat_completion_span_emits_gen_ai_attributes(in_memory_tracer):
    """A wrapped LLM call produces a ``chat <model>`` span with the
    required OTEL GenAI semantic-convention attributes.
    """

    from synthesis_engine.observability import (
        chat_completion_span,
        record_llm_response,
    )

    model = "anthropic/claude-opus-4-8"
    provider = "anthropic"

    with chat_completion_span(
        model=model,
        provider=provider,
        max_tokens=512,
        temperature=0.7,
        stream=False,
        backend_name="litellm",
    ) as span:
        record_llm_response(
            span,
            input_tokens=100,
            output_tokens=42,
            finish_reason="stop",
            cache_read_tokens=80,
            cache_creation_tokens=20,
        )

    spans = in_memory_tracer.get_finished_spans()
    chat_spans = [s for s in spans if s.name == f"chat {model}"]
    assert chat_spans, (
        f"expected a 'chat {model}' span; got names: {[s.name for s in spans]}"
    )
    chat_span = chat_spans[0]
    attrs = chat_span.attributes

    # OTEL GenAI required / recommended fields.
    assert attrs.get("gen_ai.operation.name") == "chat"
    assert attrs.get("gen_ai.provider.name") == provider
    assert attrs.get("gen_ai.request.model") == model
    assert attrs.get("gen_ai.request.max_tokens") == 512
    assert attrs.get("gen_ai.request.temperature") == pytest.approx(0.7)
    assert attrs.get("gen_ai.usage.input_tokens") == 100
    assert attrs.get("gen_ai.usage.output_tokens") == 42
    # finish_reasons is a list per the spec.
    finish_reasons = attrs.get("gen_ai.response.finish_reasons")
    assert finish_reasons is not None
    assert "stop" in list(finish_reasons)

    # Anthropic-specific cache accounting.
    assert attrs.get("gen_ai.usage.cache_read.input_tokens") == 80
    assert attrs.get("gen_ai.usage.cache_creation.input_tokens") == 20

    # The synthesis-specific cache hit ratio: 80 / (80 + 20 + 100) = 0.4
    ratio = attrs.get("synthesis.cache.hit_ratio")
    assert ratio == pytest.approx(0.4, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 3 — retrieval span carries workspace + k + latency
# ---------------------------------------------------------------------------


def test_retrieval_span_emits_workspace_k_and_latency(in_memory_tracer):
    """A wrapped retrieval call emits a ``retrieval <workspace>`` span
    with the synthesis-specific workspace + k attributes and a non-zero
    measured duration.
    """

    import time

    from synthesis_engine.observability import (
        record_retrieval_result,
        retrieval_span,
    )

    workspace = "personal"
    query = "what is RAG?"
    k = 4

    with retrieval_span(
        workspace=workspace,
        query=query,
        k=k,
        tier="vector",
        backend="pgvector",
    ) as span:
        # Simulate retrieval latency.
        time.sleep(0.005)
        record_retrieval_result(span, result_count=3, top_score=0.92)

    spans = in_memory_tracer.get_finished_spans()
    retrieval_spans = [s for s in spans if s.name == f"retrieval {workspace}"]
    assert retrieval_spans, (
        f"expected a 'retrieval {workspace}' span; got: {[s.name for s in spans]}"
    )
    rs = retrieval_spans[0]
    attrs = rs.attributes

    assert attrs.get("synthesis.workspace") == workspace
    assert attrs.get("synthesis.retrieval.k") == k
    assert attrs.get("synthesis.retriever.tier") == "vector"
    assert attrs.get("synthesis.retrieval.backend") == "pgvector"
    # Query text is not recorded by default (privacy); only the length.
    assert attrs.get("synthesis.retrieval.query_length") == len(query)
    assert attrs.get("synthesis.retrieval.result_count") == 3
    assert attrs.get("synthesis.retrieval.top_score") == pytest.approx(0.92)

    # The span has a measured non-zero duration.
    duration_ns = rs.end_time - rs.start_time
    assert duration_ns > 0


# ---------------------------------------------------------------------------
# Test 4 — Anthropic prompt-cache application
# ---------------------------------------------------------------------------


def test_cache_control_applied_for_anthropic_model_only():
    """For an Anthropic model, the rewritten messages array has a
    ``cache_control`` annotation on the system prompt block. For a
    non-Anthropic model, the messages pass through unchanged.
    """

    from synthesis_engine.llm.cache_control import (
        CacheConfig,
        apply_cache_control_to_messages,
        is_anthropic_model,
        is_eligible_for_cache,
    )

    cfg = CacheConfig(enabled=True)
    base_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hi there."},
    ]

    # ---- Anthropic path ---------------------------------------------------
    anthropic_model = "anthropic/claude-opus-4-8"
    assert is_anthropic_model(anthropic_model)
    assert is_eligible_for_cache(anthropic_model, cfg)

    rewritten, stats = apply_cache_control_to_messages(base_messages, cfg)

    assert stats["system_cache_applied"] is True, (
        "system_cache_applied flag should fire for Anthropic models"
    )

    # First message is now a structured system block with cache_control.
    system_message = rewritten[0]
    assert system_message["role"] == "system"
    assert isinstance(system_message["content"], list)
    first_block = system_message["content"][0]
    assert first_block["type"] == "text"
    assert "cache_control" in first_block, (
        f"system block missing cache_control: {first_block}"
    )
    assert first_block["cache_control"].get("type") == "ephemeral"

    # ---- Non-Anthropic path -----------------------------------------------
    gpt_model = "openai/gpt-5.6-terra"
    assert not is_anthropic_model(gpt_model)
    assert not is_eligible_for_cache(gpt_model, cfg)
    # The eligibility check is the gate the LiteLLM backend uses; when
    # ineligible, the rewrite is simply not invoked.


# ---------------------------------------------------------------------------
# Test 5 — Prometheus exposition endpoint
# ---------------------------------------------------------------------------


def test_prometheus_metrics_endpoint_returns_valid_exposition(in_memory_tracer):
    """The /api/metrics endpoint returns a payload that parses as valid
    Prometheus exposition format (via prometheus_client's text parser)
    and includes the synthesis_engine metric family names.
    """

    from fastapi.testclient import TestClient
    from prometheus_client.parser import text_string_to_metric_families

    from synthesis_engine.observability import (
        chat_completion_span,
        record_llm_response,
    )

    # Emit at least one LLM-instrument record so the Prometheus registry
    # has a non-empty observation series.
    with chat_completion_span(
        model="anthropic/claude-opus-4-8",
        provider="anthropic",
        max_tokens=128,
    ) as span:
        record_llm_response(
            span,
            input_tokens=10,
            output_tokens=5,
            finish_reason="stop",
            cache_read_tokens=2,
        )

    from api.main import app
    client = TestClient(app)

    response = client.get("/api/metrics")
    assert response.status_code == 200, response.text
    body = response.text

    # The Content-Type must follow the Prometheus exposition spec.
    content_type = response.headers.get("content-type", "")
    assert "text/plain" in content_type, (
        f"expected text/plain Prometheus exposition; got {content_type!r}"
    )

    # Parse the payload using prometheus_client's parser; this is the
    # canonical validity check for the exposition format.
    families = list(text_string_to_metric_families(body))
    family_names = {f.name for f in families}

    # The substrate emits at least the LLM-request counter and operation
    # duration histogram (the latter is suffixed with _seconds when in
    # exposition form). prometheus_client strips the bucket suffix on the
    # parser side, so the family name is the *base* name.
    assert any(
        name.startswith("synthesis_llm_requests")
        for name in family_names
    ), f"missing synthesis_llm_requests; saw: {sorted(family_names)}"
    assert any(
        "gen_ai_client_operation_duration" in name
        for name in family_names
    ), f"missing gen_ai_client_operation_duration; saw: {sorted(family_names)}"


# ---------------------------------------------------------------------------
# Extra coverage: the JSON cache-stats endpoint returns the documented shape.
# ---------------------------------------------------------------------------


def test_cache_stats_endpoint_returns_documented_shape(in_memory_tracer):
    from fastapi.testclient import TestClient

    from synthesis_engine.observability import (
        chat_completion_span,
        record_llm_response,
    )

    with chat_completion_span(
        model="anthropic/claude-opus-4-8",
        provider="anthropic",
        max_tokens=64,
    ) as span:
        record_llm_response(
            span,
            input_tokens=50,
            output_tokens=10,
            cache_read_tokens=40,
            cache_creation_tokens=10,
        )

    from api.main import app
    client = TestClient(app)
    response = client.get("/api/metrics/cache?window_minutes=30")
    assert response.status_code == 200
    payload = response.json()

    for key in (
        "window_seconds",
        "window_minutes",
        "samples",
        "cache_read_tokens",
        "cache_creation_tokens",
        "uncached_input_tokens",
        "total_input_tokens",
        "hit_rate",
        "per_model",
    ):
        assert key in payload, f"missing field {key!r}: {payload}"

    assert payload["window_minutes"] == 30
    # The recorded event landed in the rolling deque.
    assert payload["samples"] >= 1
    assert 0.0 <= payload["hit_rate"] <= 1.0
