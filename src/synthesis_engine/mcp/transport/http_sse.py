"""HTTP/SSE transport adapters.

Covers two related but distinct wire transports:

* **Streamable HTTP** — single endpoint, full duplex via HTTP POSTs from
  the client and SSE streams from the server. Used when
  ``server.transport == "http"``.
* **Legacy SSE** — the original MCP HTTP transport. A persistent
  ``GET /sse`` opens the read channel and the client sends each message
  as a separate POST. Retained because many MCP servers still offer it
  as their primary endpoint. Used when ``server.transport == "sse"``.

Both delegate to the official SDK helpers. They compose
``OAuthClientProvider`` (when configured) and any static headers from the
server entry into the HTTP client the SDK uses, so the caller need not
know which auth path is in play.

mcp 2.x takes a ready-made ``httpx2.AsyncClient`` for Streamable HTTP
instead of separate header, timeout and auth arguments. This module
builds that client, owns its lifetime, and keeps the read timeout long
enough for a server to hold a response stream open.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Any

import httpx2
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client

from ..auth import static_headers_for
from ..config import MCPServerConfig

# A server may keep a response stream open between events; the read timeout
# bounds the silence between them, not the whole call. Matches the SDK's own
# default for streamed responses.
STREAM_READ_TIMEOUT_SECONDS = 300.0


def _http_client(server: MCPServerConfig, oauth_provider: Any | None) -> httpx2.AsyncClient:
    """Build the HTTP client for a Streamable HTTP connection to ``server``."""
    headers = static_headers_for(server)
    return httpx2.AsyncClient(
        headers=headers or None,
        timeout=httpx2.Timeout(
            float(server.timeout_seconds), read=STREAM_READ_TIMEOUT_SECONDS
        ),
        auth=oauth_provider,
    )


@contextlib.asynccontextmanager
async def open_http_transport(
    server: MCPServerConfig,
    *,
    oauth_provider: Any | None = None,
) -> AsyncIterator[tuple[Any, Any]]:
    """Open a Streamable HTTP transport against ``server.url``."""
    if not server.url:
        raise ValueError(f"http server {server.id!r} has no 'url' configured")
    async with _http_client(server, oauth_provider) as http_client:
        async with streamable_http_client(
            server.url, http_client=http_client
        ) as (read_stream, write_stream):
            yield read_stream, write_stream


@contextlib.asynccontextmanager
async def open_sse_transport(
    server: MCPServerConfig,
    *,
    oauth_provider: Any | None = None,
) -> AsyncIterator[tuple[Any, Any]]:
    """Open a legacy SSE transport against ``server.url``."""
    if not server.url:
        raise ValueError(f"sse server {server.id!r} has no 'url' configured")
    headers = static_headers_for(server)
    async with sse_client(
        server.url,
        headers=headers or None,
        timeout=float(server.timeout_seconds),
        sse_read_timeout=STREAM_READ_TIMEOUT_SECONDS,
        auth=oauth_provider,
    ) as (read_stream, write_stream):
        yield read_stream, write_stream


__all__ = ["open_http_transport", "open_sse_transport"]
