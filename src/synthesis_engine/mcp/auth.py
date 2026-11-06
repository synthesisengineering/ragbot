"""OAuth 2.1 + Dynamic Client Registration support for remote MCP servers.

This module wires the official SDK's :class:`mcp.client.auth.OAuthClientProvider`
into the substrate so any HTTP/SSE server with ``auth.mode == "oauth"`` runs
the full MCP authorization handshake from the 2025-11-25 spec:

1. The client makes an unauthenticated MCP request and receives 401.
2. The client discovers Protected Resource Metadata (RFC 9728) — either
   from the ``WWW-Authenticate`` header or by probing the well-known URI.
3. The client fetches the authorization server's metadata (RFC 8414 or
   OpenID Connect Discovery 1.0).
4. The client identifies itself one of three ways:
   * **Client ID Metadata Document** (CIMD; preferred per the spec) — an
     HTTPS URL is used as the client ID, pointing at a JSON document the
     authorization server fetches.
   * **Dynamic Client Registration** (RFC 7591) — the client POSTs to the
     authorization server's ``/register`` endpoint.
   * Pre-registered credentials (not implemented here; would need explicit
     config support).
5. PKCE-protected authorization-code flow with RFC 8707 ``resource``
   indicator binding.
6. Tokens stored on disk under ``~/.synthesis/mcp/tokens/<server_id>.json``,
   refreshed on demand.

Local stdio servers ignore this module entirely — the spec is explicit
that stdio transports should rely on environment-supplied credentials,
not OAuth.

The :class:`DiskTokenStorage` implementation satisfies the SDK's
``TokenStorage`` protocol, persisting tokens and registered client info
to disk so a user does not have to re-authorize on every restart.

The :class:`LocalBrowserOAuthFlow` orchestrates the user-facing half: it
opens the system browser at the authorization URL and runs a one-shot
``http.server`` on ``localhost`` to capture the redirect with the
authorization code. This is the standard MCP pattern and works on every
desktop OS without any additional plumbing.
"""

from __future__ import annotations

import asyncio
import html
import http.server
import logging
import os
import socket
import threading
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.shared.auth import (
    AuthorizationCodeResult,
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthToken,
)

from synthesis_engine.exceptions import SynthesisError

from .config import MCPServerConfig, mcp_state_dir


logger = logging.getLogger("synthesis_engine.mcp.auth")


class MCPAuthError(SynthesisError):
    """OAuth flow failed (registration, authorization, token exchange)."""


# ---------------------------------------------------------------------------
# DiskTokenStorage
# ---------------------------------------------------------------------------


def _tokens_dir() -> Path:
    """Return ``~/.synthesis/mcp/tokens`` (created on demand)."""
    d = mcp_state_dir() / "tokens"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _token_path(server_id: str) -> Path:
    return _tokens_dir() / f"{server_id}.tokens.json"


def _client_info_path(server_id: str) -> Path:
    return _tokens_dir() / f"{server_id}.client.json"


class DiskTokenStorage(TokenStorage):
    """Persist tokens and registered client metadata to disk.

    Files are written into ``~/.synthesis/mcp/tokens/`` with restrictive
    permissions (0600). The format matches the SDK's pydantic models so
    a future migration to a different storage backend can simply round-
    trip these JSON files.
    """

    def __init__(self, server_id: str):
        self.server_id = server_id

    # ------ tokens ----------------------------------------------------------

    async def get_tokens(self) -> Optional[OAuthToken]:
        p = _token_path(self.server_id)
        if not p.exists():
            return None
        try:
            return OAuthToken.model_validate_json(p.read_text())
        except Exception as exc:  # pragma: no cover - corruption recovery
            logger.warning(
                "discarding corrupt token file for server %s: %s",
                self.server_id, exc,
            )
            try:
                p.unlink()
            except OSError:
                pass
            return None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        p = _token_path(self.server_id)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(tokens.model_dump_json(exclude_none=True))
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, p)

    # ------ client info -----------------------------------------------------

    async def get_client_info(self) -> Optional[OAuthClientInformationFull]:
        p = _client_info_path(self.server_id)
        if not p.exists():
            return None
        try:
            return OAuthClientInformationFull.model_validate_json(p.read_text())
        except Exception as exc:  # pragma: no cover - corruption recovery
            logger.warning(
                "discarding corrupt client info for server %s: %s",
                self.server_id, exc,
            )
            return None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        p = _client_info_path(self.server_id)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(client_info.model_dump_json(exclude_none=True))
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, p)


# ---------------------------------------------------------------------------
# Browser-based authorization flow
# ---------------------------------------------------------------------------


@dataclass
class _CallbackResult:
    code: Optional[str] = None
    state: Optional[str] = None
    iss: Optional[str] = None
    error: Optional[str] = None


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    """One-shot HTTP handler that captures the OAuth redirect."""

    # populated by LocalBrowserOAuthFlow before serve_forever
    _bucket: Optional[_CallbackResult] = None
    _done: Optional[threading.Event] = None

    def do_GET(self) -> None:  # noqa: N802 - http.server signature
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        bucket = self._bucket
        if bucket is not None:
            bucket.code = (qs.get("code") or [None])[0]
            bucket.state = (qs.get("state") or [None])[0]
            # RFC 9207 issuer identification; the SDK checks it against the
            # authorization server it expected when the server sends one.
            bucket.iss = (qs.get("iss") or [None])[0]
            bucket.error = (qs.get("error") or [None])[0]
        if bucket and bucket.error:
            self.send_response(400)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            # The error string comes from the redirect's query string, so it is
            # escaped before it reaches the page.
            self.wfile.write(
                f"<html><body><h1>Authorization failed</h1>"
                f"<p>{html.escape(bucket.error)}</p></body></html>".encode()
            )
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                b"<html><body><h1>Authorization complete</h1>"
                b"<p>You may close this window and return to the application.</p>"
                b"</body></html>"
            )
        if self._done is not None:
            self._done.set()

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - http.server
        # silence default stderr logging
        logger.debug("callback %s", format % args)


def _find_free_port(preferred: int) -> int:
    """Return ``preferred`` if it binds; otherwise an OS-assigned port."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", preferred))
        s.close()
        return preferred
    except OSError:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        return port


class LocalBrowserOAuthFlow:
    """Handle the browser-side half of the MCP OAuth dance.

    Two callable surfaces, both wired into :class:`OAuthClientProvider`:

    * :meth:`open_in_browser` is the ``redirect_handler`` — it receives a
      fully-formed authorization URL and is responsible for getting the
      user there.
    * :meth:`await_callback` is the ``callback_handler`` — it blocks
      until the loopback HTTP server receives the redirect and returns an
      :class:`AuthorizationCodeResult` (code, state and issuer).

    The local server is bound only after :meth:`open_in_browser` is
    invoked, on the configured port (or any free port if it is taken),
    and is torn down immediately after the callback is consumed.
    """

    def __init__(self, redirect_port: int = 33418, auto_open: bool = True):
        self.redirect_port = redirect_port
        self.auto_open = auto_open
        self._server: Optional[http.server.HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._done = threading.Event()
        self._bucket = _CallbackResult()
        self._actual_port: Optional[int] = None

    @property
    def redirect_uri(self) -> str:
        port = self._actual_port or self.redirect_port
        return f"http://127.0.0.1:{port}/callback"

    def _start_server(self) -> None:
        if self._server is not None:
            return
        port = _find_free_port(self.redirect_port)
        self._actual_port = port

        handler_cls = type(
            "ConfiguredHandler",
            (_CallbackHandler,),
            {"_bucket": self._bucket, "_done": self._done},
        )
        self._server = http.server.HTTPServer(("127.0.0.1", port), handler_cls)
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._thread.start()
        logger.info("callback listener bound at %s", self.redirect_uri)

    def _stop_server(self) -> None:
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:  # pragma: no cover - defensive
                pass
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    async def open_in_browser(self, authorization_url: str) -> None:
        """``redirect_handler`` body: launch the system browser."""
        self._start_server()
        logger.info("opening browser for OAuth: %s", authorization_url)
        if self.auto_open:
            # webbrowser.open is sync; spinning it off keeps the event loop
            # responsive even on platforms where it blocks briefly.
            await asyncio.to_thread(webbrowser.open, authorization_url)

    async def await_callback(self) -> AuthorizationCodeResult:
        """``callback_handler`` body: wait for the redirect, return its result."""
        # blocking wait on the threading.Event without holding the event loop.
        deadline = time.monotonic() + 300.0
        while not self._done.is_set():
            if time.monotonic() > deadline:
                self._stop_server()
                raise MCPAuthError("OAuth callback timed out after 5 minutes")
            await asyncio.sleep(0.1)
        try:
            if self._bucket.error:
                raise MCPAuthError(f"authorization failed: {self._bucket.error}")
            if not self._bucket.code:
                raise MCPAuthError("authorization succeeded but no code was returned")
            return AuthorizationCodeResult(
                code=self._bucket.code,
                state=self._bucket.state,
                iss=self._bucket.iss,
            )
        finally:
            self._stop_server()


# ---------------------------------------------------------------------------
# Build an OAuthClientProvider for a configured server
# ---------------------------------------------------------------------------


def build_oauth_provider(
    server: MCPServerConfig,
    *,
    auto_open_browser: bool = True,
) -> Optional[OAuthClientProvider]:
    """Construct an :class:`OAuthClientProvider` for ``server`` or return None.

    Returns ``None`` for servers that do not use OAuth (``auth.mode != "oauth"``),
    for ``bearer`` mode (handled with static headers elsewhere), and for stdio
    servers (which the spec says should not use this flow).
    """
    if server.transport == "stdio":
        return None
    auth = server.auth
    if auth.mode != "oauth":
        return None
    if not server.url:
        raise MCPAuthError(f"server {server.id} is OAuth-enabled but has no URL")

    flow = LocalBrowserOAuthFlow(
        redirect_port=auth.redirect_port,
        auto_open=auto_open_browser,
    )
    redirect_uri = flow.redirect_uri

    client_metadata = OAuthClientMetadata(
        redirect_uris=[redirect_uri],
        client_name=auth.client_name,
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        scope=auth.scope,
        token_endpoint_auth_method="none",
    )

    storage = DiskTokenStorage(server.id)

    return OAuthClientProvider(
        server_url=server.url,
        client_metadata=client_metadata,
        storage=storage,
        redirect_handler=flow.open_in_browser,
        callback_handler=flow.await_callback,
        client_metadata_url=auth.client_id_metadata_url,
    )


def static_headers_for(server: MCPServerConfig) -> dict[str, str]:
    """Resolve the static header set for ``server`` (bearer + user headers)."""
    headers = dict(server.headers or {})
    if server.auth.mode == "bearer" and server.auth.token:
        headers["Authorization"] = f"Bearer {server.auth.token}"
    return headers


__all__ = [
    "DiskTokenStorage",
    "LocalBrowserOAuthFlow",
    "MCPAuthError",
    "build_oauth_provider",
    "static_headers_for",
]
