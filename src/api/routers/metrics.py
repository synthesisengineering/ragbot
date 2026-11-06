"""Metrics API endpoints.

Two endpoints, both intentionally narrow:

  GET /api/metrics
      Prometheus exposition format. Backed by the prometheus_client
      registry that the synthesis_engine observability substrate maintains
      in parallel with its OTEL meter provider. Scrapeable by Prometheus
      directly — no collector required.

  GET /api/metrics/cache
      JSON view of the rolling prompt-cache hit rate over the last N
      minutes (default 60). Convenience endpoint for the web UI's
      observability panel and for ad-hoc health checks; not a substitute
      for the full Prometheus surface.

Both endpoints lazily ensure the substrate is initialised, so the first
request after process start triggers a one-time ``init_tracer`` without
forcing every caller to wire startup hooks.
"""

from __future__ import annotations

import os
import sys

from fastapi import APIRouter, Query, Response

# Add src directory to path so synthesis_engine is importable when this
# module is loaded from the ``api`` package.
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from synthesis_engine.observability import (  # noqa: E402
    cache_stats_capture,
    init_tracer,
    render_prometheus,
)
from synthesis_engine.observability.tracer import is_initialized  # noqa: E402


router = APIRouter(prefix="/api/metrics", tags=["metrics"])


def _ensure_initialized() -> None:
    """Lazy-init the substrate on first metrics scrape.

    The substrate is silent (no-op) until ``init_tracer`` runs. For the
    metrics endpoints to return a usable payload, the parallel Prometheus
    registry has to be bound. We initialise with a no-op exporter (the
    default), so this triggers the Prometheus binding without sending
    spans anywhere unless an OTLP endpoint is configured via env.
    """

    if not is_initialized():
        service = os.environ.get("OTEL_SERVICE_NAME") or "ragbot"
        init_tracer(service_name=service)


@router.get("", include_in_schema=True)
async def get_prometheus_metrics() -> Response:
    """Return Prometheus exposition format for the synthesis_engine registry.

    Content-Type follows the canonical Prometheus content-type (set by the
    ``prometheus_client`` library).
    """

    _ensure_initialized()
    body, content_type = render_prometheus()
    return Response(content=body, media_type=content_type)


@router.get("/cache")
async def get_cache_stats(
    window_minutes: int = Query(
        60,
        ge=1,
        le=24 * 60,
        description="Rolling window in minutes. Default: 60.",
    ),
) -> dict:
    """Return prompt-cache hit-rate statistics for the last N minutes.

    Response shape:

        {
          "window_seconds": 3600,
          "window_minutes": 60,
          "samples": 42,
          "cache_read_tokens": 12345,
          "cache_creation_tokens": 6789,
          "uncached_input_tokens": 234,
          "total_input_tokens": 19368,
          "hit_rate": 0.6373,
          "per_model": {
            "anthropic/claude-opus-5-5": {
              "samples": 30,
              "cache_read_tokens": 9000,
              "cache_creation_tokens": 5000,
              "uncached_input_tokens": 100
            },
            ...
          }
        }

    A sample of 0 with hit_rate of 0.0 means no LLM calls have been
    observed in the window; it is NOT an error.
    """

    _ensure_initialized()
    window_seconds = float(window_minutes * 60)
    stats = cache_stats_capture(window_seconds=window_seconds)
    stats["window_minutes"] = window_minutes
    return stats
