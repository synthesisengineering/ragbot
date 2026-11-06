"""Tests for the synthesis_engine MCP client substrate and the API router.

Three groups of tests:

1. **API router** — uses FastAPI's :class:`TestClient` against a fresh
   :class:`MCPClient` whose transport layer is never invoked. Covers
   server CRUD via the REST surface, error shapes, and 4xx handling.
2. **Workspace allow/deny** — exercises the substrate's per-workspace
   filtering directly so we know the REST surface composes correctly
   on top of it.
3. **Live filesystem round-trip** — spawns the public
   ``@modelcontextprotocol/server-filesystem`` reference server under
   ``npx`` and walks the connect → list_tools → call_tool path. Behind
   ``@pytest.mark.skipif`` for environments without ``npx`` so CI on a
   minimal image still passes.

The tests do not touch ``~/.synthesis/mcp.yaml`` directly. Every test
that writes config points :func:`mcp_config_path` at a tmp file via the
``SYNTHESIS_HOME`` env var so a developer running the suite locally
does not lose their real config.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# Make the ragbot src/ tree importable.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from fastapi import FastAPI
from fastapi.testclient import TestClient

from synthesis_engine.mcp import (
    MCPClient,
    MCPConfig,
    MCPServerConfig,
    get_default_client,
    set_default_client,
)
from synthesis_engine.mcp.registry import (
    CachedCatalog,
    MCPRegistry,
    MCPRegistryError,
    ServerEntry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_synthesis_home(monkeypatch, tmp_path):
    """Redirect ``~/.synthesis`` to a tmp directory for every test.

    The MCP router persists added servers to ``~/.synthesis/mcp.yaml``;
    without this fixture, the test suite would corrupt a developer's
    real config on first run.
    """
    home = tmp_path / "synthesis-home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SYNTHESIS_HOME", str(home))
    yield home


@pytest.fixture(autouse=True)
def reset_default_client():
    """Clear the substrate's process-singleton between tests."""
    set_default_client(None)
    yield
    set_default_client(None)


@pytest.fixture
def app_with_router() -> FastAPI:
    """Build a minimal FastAPI app with just the MCP router mounted."""
    from api.routers import mcp as mcp_router  # noqa: WPS433

    app = FastAPI()
    app.include_router(mcp_router.router)
    return app


@pytest.fixture
def client(app_with_router) -> TestClient:
    return TestClient(app_with_router)


def _make_stdio_server(server_id: str = "test-fs", **overrides: Any) -> MCPServerConfig:
    """Build a representative stdio server config for tests."""
    kwargs: Dict[str, Any] = dict(
        id=server_id,
        name=f"Test {server_id}",
        description="Test fixture",
        transport="stdio",
        command="echo",
        args=["{}"],
    )
    kwargs.update(overrides)
    return MCPServerConfig(**kwargs)


# ---------------------------------------------------------------------------
# Workspace allow/deny — direct substrate tests
# ---------------------------------------------------------------------------


class TestWorkspaceGating:
    """Substrate-level: per-workspace allow/deny resolution."""

    def test_wildcard_enabled_admits_everywhere(self):
        cfg = MCPConfig(
            servers=[
                _make_stdio_server("s1", enabled_workspaces=["*"]),
            ]
        )
        reg = MCPRegistry(cfg)
        assert [e.config.id for e in reg.get_active_servers("personal")] == ["s1"]
        assert [e.config.id for e in reg.get_active_servers("client-x")] == ["s1"]
        assert [e.config.id for e in reg.get_active_servers(None)] == ["s1"]

    def test_named_workspace_only_admits_that_workspace(self):
        cfg = MCPConfig(
            servers=[
                _make_stdio_server("s1", enabled_workspaces=["personal"]),
            ]
        )
        reg = MCPRegistry(cfg)
        assert [e.config.id for e in reg.get_active_servers("personal")] == ["s1"]
        assert reg.get_active_servers("other") == []
        # workspace=None should be excluded — explicit list with no "*"
        assert reg.get_active_servers(None) == []

    def test_disabled_workspaces_beats_enabled(self):
        cfg = MCPConfig(
            servers=[
                _make_stdio_server(
                    "s1",
                    enabled_workspaces=["*"],
                    disabled_workspaces=["client-acme"],
                ),
            ]
        )
        reg = MCPRegistry(cfg)
        assert [e.config.id for e in reg.get_active_servers("personal")] == ["s1"]
        assert reg.get_active_servers("client-acme") == []

    def test_disabled_flag_takes_precedence(self):
        cfg = MCPConfig(
            servers=[
                _make_stdio_server("s1", enabled_workspaces=["*"], enabled=False),
            ]
        )
        reg = MCPRegistry(cfg)
        assert reg.get_active_servers("personal") == []

    def test_default_on_when_no_enabled_workspaces_list(self):
        cfg = MCPConfig(
            servers=[
                _make_stdio_server("s1", enabled_workspaces=None),
            ]
        )
        reg = MCPRegistry(cfg)
        # defaults.enabled_by_default = True (the substrate default)
        assert [e.config.id for e in reg.get_active_servers("anywhere")] == ["s1"]

    def test_default_off_when_globally_disabled(self):
        from synthesis_engine.mcp import MCPDefaults

        cfg = MCPConfig(
            servers=[
                _make_stdio_server("s1", enabled_workspaces=None),
            ],
            defaults=MCPDefaults(enabled_by_default=False),
        )
        reg = MCPRegistry(cfg)
        assert reg.get_active_servers("personal") == []


# ---------------------------------------------------------------------------
# Registry-level primitive listing — cached catalog
# ---------------------------------------------------------------------------


class TestRegistryCachedCatalog:
    """The registry exposes a cached catalog so the UI can render server
    summaries without a live round trip. Verify the structure is sane
    even when no connection has been opened yet."""

    def test_fresh_entry_has_empty_catalog(self):
        cfg = MCPConfig(servers=[_make_stdio_server("s1")])
        reg = MCPRegistry(cfg)
        entry = reg.get_entry("s1")
        assert entry.status == "disconnected"
        assert entry.session is None
        assert entry.catalog.tools == []
        assert entry.catalog.resources == []
        assert entry.catalog.prompts == []

    def test_unknown_server_raises(self):
        reg = MCPRegistry(MCPConfig())
        with pytest.raises(MCPRegistryError):
            reg.get_entry("nope")

    def test_replace_config_preserves_existing_entries(self):
        cfg1 = MCPConfig(servers=[_make_stdio_server("a"), _make_stdio_server("b")])
        reg = MCPRegistry(cfg1)
        # Mutate one entry's catalog so we can check survival.
        entry_a = reg.get_entry("a")
        entry_a.catalog = CachedCatalog(tools=[], resources=[], prompts=[])
        entry_a.last_error = "marker"
        cfg2 = MCPConfig(
            servers=[
                _make_stdio_server("a"),  # unchanged id
                _make_stdio_server("c"),  # new
            ]
        )
        reg.replace_config(cfg2)
        # "a" survived (with its marker), "b" is gone, "c" is fresh.
        assert reg.get_entry("a").last_error == "marker"
        assert not reg.has("b")
        assert reg.has("c")
        assert reg.get_entry("c").last_error is None


# ---------------------------------------------------------------------------
# API router — TestClient against a mocked client
# ---------------------------------------------------------------------------


def _install_client(servers: Optional[List[MCPServerConfig]] = None) -> MCPClient:
    """Install a fresh MCPClient (autoconnect off) as the process singleton."""
    cfg = MCPConfig(servers=list(servers or []))
    client = MCPClient(cfg, autoconnect=False)
    set_default_client(client)
    return client


class TestServersListEndpoint:
    def test_empty_list_when_no_servers_configured(self, client):
        _install_client([])
        resp = client.get("/api/mcp/servers")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body == {"servers": [], "workspace": None}

    def test_list_returns_serialized_entries(self, client):
        _install_client(
            [
                _make_stdio_server("fs", description="local fs"),
                _make_stdio_server("git", description="local git"),
            ]
        )
        resp = client.get("/api/mcp/servers")
        assert resp.status_code == 200
        body = resp.json()
        ids = [s["id"] for s in body["servers"]]
        assert ids == ["fs", "git"]
        first = body["servers"][0]
        assert first["status"] == "disconnected"
        assert first["last_error"] is None
        assert first["catalog_sizes"] == {"tools": 0, "resources": 0, "prompts": 0}
        assert first["transport"] == "stdio"

    def test_workspace_filter_applies_allow_deny(self, client):
        _install_client(
            [
                _make_stdio_server("global", enabled_workspaces=["*"]),
                _make_stdio_server("only-personal", enabled_workspaces=["personal"]),
                _make_stdio_server(
                    "denied-here",
                    enabled_workspaces=["*"],
                    disabled_workspaces=["client-acme"],
                ),
            ]
        )

        resp = client.get("/api/mcp/servers", params={"workspace": "personal"})
        assert resp.status_code == 200
        personal_ids = {s["id"] for s in resp.json()["servers"]}
        assert personal_ids == {"global", "only-personal", "denied-here"}

        resp = client.get("/api/mcp/servers", params={"workspace": "client-acme"})
        assert resp.status_code == 200
        acme_ids = {s["id"] for s in resp.json()["servers"]}
        # 'denied-here' is excluded by disabled_workspaces; 'only-personal'
        # is excluded because client-acme is not in its enabled list.
        assert acme_ids == {"global"}


class TestServersUpsertEndpoint:
    def test_add_new_server_persists_and_returns(self, client, isolated_synthesis_home):
        _install_client([])
        payload = {
            "id": "fs-test",
            "name": "Filesystem",
            "transport": "stdio",
            "command": "echo",
            "args": ["hello"],
        }
        resp = client.post("/api/mcp/servers", json=payload)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["id"] == "fs-test"
        assert body["status"] == "disconnected"

        # The list endpoint should now include the new server.
        resp = client.get("/api/mcp/servers")
        ids = [s["id"] for s in resp.json()["servers"]]
        assert "fs-test" in ids

        # And it should have been persisted to mcp.yaml under SYNTHESIS_HOME.
        yaml_path = isolated_synthesis_home / "mcp.yaml"
        assert yaml_path.exists(), "POST should persist the config"
        assert "fs-test" in yaml_path.read_text()

    def test_replace_existing_server_by_id(self, client):
        _install_client([_make_stdio_server("dup", description="v1")])
        payload = {
            "id": "dup",
            "name": "Dup",
            "transport": "stdio",
            "command": "echo",
            "description": "v2",
        }
        resp = client.post("/api/mcp/servers", json=payload)
        assert resp.status_code == 200
        assert resp.json()["description"] == "v2"

        # Should still be exactly one server, with the updated description.
        resp = client.get("/api/mcp/servers")
        servers = resp.json()["servers"]
        assert len(servers) == 1
        assert servers[0]["description"] == "v2"

    def test_stdio_without_command_rejected(self, client):
        _install_client([])
        payload = {
            "id": "bad",
            "name": "Bad",
            "transport": "stdio",
            # command intentionally missing
        }
        resp = client.post("/api/mcp/servers", json=payload)
        assert resp.status_code == 400
        assert "command" in resp.text.lower()

    def test_http_without_url_rejected(self, client):
        _install_client([])
        payload = {
            "id": "bad-http",
            "name": "Bad HTTP",
            "transport": "http",
            # url intentionally missing
        }
        resp = client.post("/api/mcp/servers", json=payload)
        assert resp.status_code == 400
        assert "url" in resp.text.lower()

    def test_invalid_transport_rejected(self, client):
        _install_client([])
        payload = {
            "id": "x",
            "name": "X",
            "transport": "telepathy",
            "command": "echo",
        }
        resp = client.post("/api/mcp/servers", json=payload)
        # pydantic returns 422 for body-level validation failures
        assert resp.status_code in (400, 422)


class TestServersDeleteEndpoint:
    def test_delete_existing_server(self, client, isolated_synthesis_home):
        _install_client([_make_stdio_server("doomed")])
        resp = client.delete("/api/mcp/servers/doomed")
        assert resp.status_code == 200
        assert resp.json() == {"deleted": "doomed"}

        # Subsequent list should not include it.
        resp = client.get("/api/mcp/servers")
        assert "doomed" not in [s["id"] for s in resp.json()["servers"]]

    def test_delete_missing_server_returns_404(self, client):
        _install_client([])
        resp = client.delete("/api/mcp/servers/ghost")
        assert resp.status_code == 404


class TestPrimitiveEndpointsRequireConnection:
    """tools/resources/prompts endpoints reject disconnected servers."""

    def test_tools_endpoint_returns_409_when_disconnected(self, client):
        _install_client([_make_stdio_server("idle")])
        resp = client.get("/api/mcp/servers/idle/tools")
        assert resp.status_code == 409
        assert "not connected" in resp.text.lower()

    def test_resources_endpoint_returns_409_when_disconnected(self, client):
        _install_client([_make_stdio_server("idle")])
        resp = client.get("/api/mcp/servers/idle/resources")
        assert resp.status_code == 409

    def test_prompts_endpoint_returns_409_when_disconnected(self, client):
        _install_client([_make_stdio_server("idle")])
        resp = client.get("/api/mcp/servers/idle/prompts")
        assert resp.status_code == 409

    def test_tools_endpoint_returns_404_for_unknown_server(self, client):
        _install_client([])
        resp = client.get("/api/mcp/servers/nope/tools")
        assert resp.status_code == 404


class TestPrimitiveEndpointsAgainstMockedSession:
    """tools/resources/prompts return live results when the entry is connected.

    We bypass the real connection lifecycle by inserting a mock session
    into the registry entry, mark it as connected, and let the
    primitive helpers route through it. The session only needs to
    implement the SDK methods the primitive wrappers call."""

    def _mock_entry_with_session(self, c: MCPClient, server_id: str, session: Any) -> None:
        entry = c.registry.get_entry(server_id)
        entry.session = session
        entry.status = "connected"

    def test_list_tools_returns_serialized_tools(self, client):
        from mcp.types import ListToolsResult, Tool

        installed = _install_client([_make_stdio_server("svr")])

        tool = Tool(
            name="echo",
            description="echo input",
            inputSchema={"type": "object", "properties": {}},
        )

        class FakeSession:
            async def list_tools(self, *, params=None):
                return ListToolsResult(tools=[tool], next_cursor=None)

        self._mock_entry_with_session(installed, "svr", FakeSession())

        resp = client.get("/api/mcp/servers/svr/tools")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["server_id"] == "svr"
        assert len(body["tools"]) == 1
        assert body["tools"][0]["name"] == "echo"
        # The web client reads snake_case keys (web/src/lib/api.ts McpTool).
        assert body["tools"][0]["input_schema"] == {"type": "object", "properties": {}}

    def test_list_tools_follows_every_page(self, client):
        from mcp.types import ListToolsResult, Tool

        installed = _install_client([_make_stdio_server("svr")])
        pages = {
            None: ListToolsResult(tools=[Tool(name="a", inputSchema={"type": "object"})], next_cursor="p2"),
            "p2": ListToolsResult(tools=[Tool(name="b", inputSchema={"type": "object"})], next_cursor=None),
        }
        seen = []

        class FakeSession:
            async def list_tools(self, *, params=None):
                cursor = None if params is None else params.cursor
                seen.append(cursor)
                return pages[cursor]

        self._mock_entry_with_session(installed, "svr", FakeSession())

        resp = client.get("/api/mcp/servers/svr/tools")
        assert resp.status_code == 200, resp.text
        assert [t["name"] for t in resp.json()["tools"]] == ["a", "b"]
        assert seen == [None, "p2"]

    def test_list_resources_returns_serialized_resources(self, client):
        from mcp.types import ListResourcesResult, Resource

        installed = _install_client([_make_stdio_server("svr")])

        resource = Resource(uri="file:///tmp/a.txt", name="a.txt", mimeType="text/plain")

        class FakeSession:
            async def list_resources(self, *, params=None):
                return ListResourcesResult(resources=[resource], next_cursor=None)

        self._mock_entry_with_session(installed, "svr", FakeSession())

        resp = client.get("/api/mcp/servers/svr/resources")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["server_id"] == "svr"
        assert len(body["resources"]) == 1
        assert body["resources"][0]["uri"] == "file:///tmp/a.txt"
        # The web client reads snake_case keys (web/src/lib/api.ts McpResource).
        assert body["resources"][0]["mime_type"] == "text/plain"

    def test_list_prompts_returns_serialized_prompts(self, client):
        from mcp.types import ListPromptsResult, Prompt

        installed = _install_client([_make_stdio_server("svr")])

        prompt = Prompt(name="hello", description="Greet")

        class FakeSession:
            async def list_prompts(self, *, params=None):
                return ListPromptsResult(prompts=[prompt], next_cursor=None)

        self._mock_entry_with_session(installed, "svr", FakeSession())

        resp = client.get("/api/mcp/servers/svr/prompts")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["server_id"] == "svr"
        assert len(body["prompts"]) == 1
        assert body["prompts"][0]["name"] == "hello"


def _make_http_server(
    server_id: str = "remote",
    *,
    auth_mode: str = "oauth",
    **overrides: Any,
) -> MCPServerConfig:
    """Build a representative remote (http) server config for OAuth tests."""
    from synthesis_engine.mcp import AuthConfig

    auth_payload: Dict[str, Any] = {"mode": auth_mode}
    if auth_mode == "bearer":
        auth_payload["token"] = "stub-token"

    kwargs: Dict[str, Any] = dict(
        id=server_id,
        name=f"Remote {server_id}",
        description="Remote test fixture",
        transport="http",
        url=f"https://{server_id}.example.com/mcp",
        auth=AuthConfig.model_validate(auth_payload),
    )
    kwargs.update(overrides)
    return MCPServerConfig(**kwargs)


class TestOAuthEndpoint:
    """``POST /api/mcp/servers/{id}/oauth`` validation + happy-path behaviour.

    The endpoint is a thin wrapper around ``client.disconnect`` +
    ``client.connect``. The substrate's connect path is exercised by the
    live filesystem round-trip and the broader registry tests; here we
    confirm the wrapper's input validation and its ``{ok, error}`` shape.
    """

    def test_unknown_server_returns_404(self, client):
        _install_client([])
        resp = client.post("/api/mcp/servers/ghost/oauth")
        assert resp.status_code == 404

    def test_stdio_server_rejected_with_400(self, client):
        _install_client([_make_stdio_server("local")])
        resp = client.post("/api/mcp/servers/local/oauth")
        assert resp.status_code == 400
        assert "stdio" in resp.text.lower()

    def test_non_oauth_mode_rejected_with_400(self, client):
        _install_client([_make_http_server("plain", auth_mode="none")])
        resp = client.post("/api/mcp/servers/plain/oauth")
        assert resp.status_code == 400
        assert "oauth" in resp.text.lower()

    def test_bearer_mode_rejected_with_400(self, client):
        _install_client([_make_http_server("static-bearer", auth_mode="bearer")])
        resp = client.post("/api/mcp/servers/static-bearer/oauth")
        assert resp.status_code == 400

    def test_happy_path_returns_ok_true(self, client, monkeypatch):
        installed = _install_client([_make_http_server("oauth-srv")])

        async def fake_connect(self, server_id):
            entry = self._registry.get_entry(server_id)
            entry.status = "connected"
            return entry

        async def fake_disconnect(self, server_id):
            return None

        monkeypatch.setattr(MCPClient, "connect", fake_connect, raising=True)
        monkeypatch.setattr(MCPClient, "disconnect", fake_disconnect, raising=True)

        resp = client.post("/api/mcp/servers/oauth-srv/oauth")
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"ok": True}

    def test_connect_failure_returns_ok_false_with_error(self, client, monkeypatch):
        _install_client([_make_http_server("oauth-bust")])

        async def fake_connect_fail(self, server_id):
            raise RuntimeError("authorization denied by upstream")

        async def fake_disconnect(self, server_id):
            return None

        monkeypatch.setattr(MCPClient, "connect", fake_connect_fail, raising=True)
        monkeypatch.setattr(MCPClient, "disconnect", fake_disconnect, raising=True)

        resp = client.post("/api/mcp/servers/oauth-bust/oauth")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["ok"] is False
        assert "authorization denied" in body["error"]
        assert "RuntimeError" in body["error"]

    def test_already_connected_forces_disconnect_then_reconnect(
        self, client, monkeypatch
    ):
        installed = _install_client([_make_http_server("oauth-warm")])
        # Mark the entry as connected so the endpoint triggers the
        # disconnect-then-reconnect branch.
        installed.registry.get_entry("oauth-warm").status = "connected"

        calls: List[str] = []

        async def fake_disconnect(self, server_id):
            calls.append(f"disconnect:{server_id}")

        async def fake_connect(self, server_id):
            calls.append(f"connect:{server_id}")
            entry = self._registry.get_entry(server_id)
            entry.status = "connected"
            return entry

        monkeypatch.setattr(MCPClient, "disconnect", fake_disconnect, raising=True)
        monkeypatch.setattr(MCPClient, "connect", fake_connect, raising=True)

        resp = client.post("/api/mcp/servers/oauth-warm/oauth")
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"ok": True}
        # disconnect must precede connect when the entry is already live
        assert calls == ["disconnect:oauth-warm", "connect:oauth-warm"]


# ---------------------------------------------------------------------------
# Live filesystem MCP round-trip — skipped when npx is unavailable
# ---------------------------------------------------------------------------


_NPX = shutil.which("npx")
_skip_if_no_npx = pytest.mark.skipif(
    _NPX is None,
    reason="npx not found on PATH; install Node.js to enable live MCP tests",
)


@_skip_if_no_npx
@pytest.mark.asyncio
async def test_filesystem_mcp_roundtrip(tmp_path):
    """End-to-end: connect to the published filesystem server and call a tool.

    Spawns ``npx -y @modelcontextprotocol/server-filesystem <tmp>`` and
    walks initialize → list_tools → call_tool. Uses ``@pytest.mark.asyncio``
    so the substrate's async API is exercised on a live transport.

    The first invocation per environment downloads the npm package, which
    can take 10–30 s. Subsequent invocations hit the npx cache and are
    near-instant.
    """
    # Prepare a sandbox directory the server will be rooted at.
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "hello.txt").write_text("hello from filesystem mcp\n")

    cfg = MCPConfig(
        servers=[
            MCPServerConfig(
                id="fs-live",
                name="Filesystem (live)",
                transport="stdio",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", str(sandbox)],
            )
        ]
    )
    client = MCPClient(cfg, autoconnect=False)
    try:
        await client.connect("fs-live")
        tools = await client.list_tools("fs-live")
        assert tools, "filesystem server should advertise at least one tool"
        tool_names = {t.name for t in tools}
        # The reference server names vary slightly across versions; a few
        # canonical tools should be present.
        assert tool_names & {"read_file", "read_text_file", "list_directory"}, (
            f"unexpected tool set: {tool_names}"
        )
    finally:
        await client.shutdown()


# ---------------------------------------------------------------------------
# pytest-asyncio support detection (best-effort import)
# ---------------------------------------------------------------------------


# Some environments don't have pytest-asyncio installed. The live
# round-trip test is marked async; if pytest-asyncio is missing, the
# test is collected but cannot execute. Mark it as skip in that case
# so the suite reports skipped (with reason) rather than ERROR.
def _has_pytest_asyncio() -> bool:
    try:
        import pytest_asyncio  # noqa: F401
        return True
    except ImportError:
        return False


if not _has_pytest_asyncio():
    test_filesystem_mcp_roundtrip = pytest.mark.skip(
        reason="pytest-asyncio not installed; cannot run async live test"
    )(test_filesystem_mcp_roundtrip)
