"""Tests for :mod:`synthesis_engine.mcp_server`.

The tests are organised in four groups:

1. **Server construction** — assert that the server wires every
   declared tool and resource and that the tool/resource schemas are
   well-formed.
2. **Tool dispatch (direct)** — drive :func:`dispatch_tool` against
   fake collaborators. This is the fastest layer to test because it
   skips the MCP framing entirely.
3. **Auth** — bearer-token resolution, ``allowed_tools`` gating, stdio
   bypass, and the HTTP fail-closed behaviour when the config is
   missing or malformed.
4. **End-to-end** — connects ``mcp.Client`` to the real SDK Server
   in-process, then walks list_tools → call_tool → list_resources →
   read_resource. This is the strongest guarantee that the registered
   handlers behave like an MCP server would for an external client.
5. **Tool-call semantics and HTTP** — input and output schema
   validation, dispatch errors as tool errors, unknown resources as
   ``INVALID_PARAMS``, fail-closed bearer resolution, and a real
   Streamable HTTP round trip through Uvicorn with bearer tokens.

All tests use placeholder workspace names (``acme-news``, ``acme-user``,
``beta-media``) — ragbot is a public repo and client names never appear
here.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pytest


# Make the ragbot src/ tree importable.
_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


from synthesis_engine.agent.permissions import (  # noqa: E402
    PermissionRegistry,
    PermissionResult,
)
from synthesis_engine.memory.models import (  # noqa: E402
    MemoryQuery,
    MemoryResult,
    Provenance,
)
from synthesis_engine.memory.retrieval import RetrievedBlock  # noqa: E402
from synthesis_engine.mcp_server import (  # noqa: E402
    AUDIT_RESOURCE_URI,
    BearerToken,
    MCPServerAuthConfig,
    MCPServerAuthError,
    RagbotMCPServer,
    SKILL_RESOURCE_PREFIX,
    ServerDependencies,
    TOOL_DEFINITIONS,
    ToolDispatchError,
    WORKSPACE_RESOURCE_PREFIX,
    dispatch_tool,
    load_auth_config,
)
from synthesis_engine.mcp_server.tools import ToolDispatchContext  # noqa: E402
from synthesis_engine.policy.routing import (  # noqa: E402
    Confidentiality,
    FallbackBehavior,
    RoutingPolicy,
)


# ---------------------------------------------------------------------------
# Placeholder workspace fixtures (public-repo safe)
# ---------------------------------------------------------------------------


# Three workspaces with deliberately different confidentiality tags so
# the cross-workspace gate has something to enforce.
ROUTING_POLICIES: Dict[str, RoutingPolicy] = {
    "acme-news": RoutingPolicy(confidentiality=Confidentiality.PUBLIC),
    "acme-user": RoutingPolicy(confidentiality=Confidentiality.PERSONAL),
    "beta-media": RoutingPolicy(
        confidentiality=Confidentiality.CLIENT_CONFIDENTIAL,
        fallback_behavior=FallbackBehavior.DENY,
    ),
    "gamma-secret": RoutingPolicy(confidentiality=Confidentiality.AIR_GAPPED),
}


def _routing_for(ws: str) -> RoutingPolicy:
    return ROUTING_POLICIES.get(ws, RoutingPolicy())


def _list_workspaces() -> List[str]:
    return list(ROUTING_POLICIES.keys())


# ---------------------------------------------------------------------------
# Fake memory backend
# ---------------------------------------------------------------------------


@dataclass
class FakeMemoryBackend:
    """Minimal stand-in for :class:`Memory` — the retrievers we use take
    a backend handle but never dereference it because we also pass in
    fake retrieve_single / retrieve_multi callables."""

    name: str = "fake"


# ---------------------------------------------------------------------------
# Fake retrievers
# ---------------------------------------------------------------------------


def fake_retrieve_single(memory: Any, query: MemoryQuery, **_kwargs: Any) -> List[MemoryResult]:
    """Return one result per tier so the schema fields all get exercised."""
    return [
        MemoryResult(
            tier="vector",
            score=0.9,
            text=f"vec hit for {query.text!r} in {query.workspace}",
            metadata={"source_workspace": query.workspace},
            provenance=Provenance(source=f"vector:{query.workspace}"),
        ),
        MemoryResult(
            tier="session",
            score=0.7,
            text="session memory snippet",
            metadata={"source_workspace": query.workspace},
        ),
    ]


def fake_retrieve_multi(
    memory: Any,
    workspaces: List[str],
    query: str,
    *,
    total_budget_tokens: int = 6000,
    **_kwargs: Any,
) -> List[RetrievedBlock]:
    blocks: List[RetrievedBlock] = []
    for rank, ws in enumerate(workspaces, start=1):
        mr = MemoryResult(
            tier="vector",
            score=0.8,
            text=f"vector hit in {ws} for {query!r}",
            metadata={"source_workspace": ws},
        )
        blocks.append(
            RetrievedBlock(
                source_workspace=ws,
                result=mr,
                estimated_tokens=len(mr.text) // 4 + 1,
                workspace_rank=rank,
            )
        )
    return blocks


# ---------------------------------------------------------------------------
# Fake document store
# ---------------------------------------------------------------------------


FAKE_DOCS: Dict[Tuple[str, str], Dict[str, Any]] = {
    ("acme-news", "doc-1"): {
        "content": "Lorem ipsum about acme-news.",
        "metadata": {"source": "fake-store"},
    },
    ("acme-user", "doc-7"): {
        "content": "Personal note in acme-user.",
        "metadata": {"source": "fake-store"},
    },
}


def fake_document_getter(workspace: str, document_id: str) -> Dict[str, Any]:
    return FAKE_DOCS[(workspace, document_id)]


# ---------------------------------------------------------------------------
# Fake skill runtime + skill objects
# ---------------------------------------------------------------------------


@dataclass
class FakeSkill:
    name: str
    description: str = ""
    body: str = ""


@dataclass
class FakeSkillRuntime:
    """Mimics the subset of :class:`SkillRuntime` the dispatcher uses."""

    visible: Dict[str, List[FakeSkill]] = field(default_factory=dict)
    activations: List[str] = field(default_factory=list)
    dispatches: List[Tuple[str, str, Dict[str, Any]]] = field(default_factory=list)
    deny_skill: Optional[str] = None
    raise_on_register: Optional[str] = None

    def register_skill_tools(self, skill_name: str) -> List[Any]:
        if self.raise_on_register and skill_name == self.raise_on_register:
            raise KeyError(f"unknown skill {skill_name}")
        self.activations.append(skill_name)
        return []

    async def _dispatch_skill_tool(
        self,
        *,
        loop: Any,
        state: Any,
        step: Any,
        inputs: Dict[str, Any],
        skill_name: str,
        tool_name: str,
    ) -> Dict[str, Any]:
        if self.deny_skill == f"{skill_name}.{tool_name}":
            raise PermissionError("skill says deny")
        self.dispatches.append((skill_name, tool_name, dict(inputs)))
        return {
            "skill": skill_name,
            "tool": tool_name,
            "arguments": dict(inputs),
            "description": f"fake {skill_name}.{tool_name}",
        }


# ---------------------------------------------------------------------------
# Fake agent-run starter
# ---------------------------------------------------------------------------


async def fake_agent_run_starter(
    workspaces: Tuple[str, ...],
    task: str,
    rubric: Optional[str],
) -> Tuple[str, str]:
    # Deterministic id so tests can assert on the exact value.
    task_id = f"task-{'-'.join(workspaces)}-{abs(hash(task)) % 10_000}"
    return task_id, f"/api/agent/sessions/{task_id}"


# ---------------------------------------------------------------------------
# Build dependencies + server
# ---------------------------------------------------------------------------


def build_dependencies(
    *,
    skill_runtime: Optional[FakeSkillRuntime] = None,
    permission_registry: Optional[PermissionRegistry] = None,
    visible_skills: Optional[Dict[str, List[FakeSkill]]] = None,
) -> ServerDependencies:
    visible = visible_skills or {
        "acme-news": [FakeSkill(name="news-skill", description="news ops")],
        "acme-user": [
            FakeSkill(name="user-skill", description="personal helper"),
            FakeSkill(name="news-skill", description="news ops"),
        ],
        "beta-media": [],
        "gamma-secret": [],
    }

    def _skills_for(ws: str) -> List[FakeSkill]:
        return visible.get(ws, [])

    return ServerDependencies(
        memory=FakeMemoryBackend(),
        skill_runtime=skill_runtime or FakeSkillRuntime(visible=visible),
        skills_visible_for=_skills_for,
        list_workspaces=_list_workspaces,
        routing_policies=_routing_for,
        document_getter=fake_document_getter,
        agent_run_starter=fake_agent_run_starter,
        permission_registry=permission_registry,
        retrieve_single=fake_retrieve_single,
        retrieve_multi=fake_retrieve_multi,
    )


@pytest.fixture
def server() -> RagbotMCPServer:
    deps = build_dependencies()
    return RagbotMCPServer(dependencies=deps)


@pytest.fixture
def dispatch_ctx(server: RagbotMCPServer) -> ToolDispatchContext:
    # Build a dispatch context the same way the server would for a stdio
    # request (bearer=None).
    return server._build_dispatch_context(bearer=None)


@pytest.fixture(autouse=True)
def isolated_audit_log(monkeypatch, tmp_path):
    """Redirect the audit log to a per-test tmp file."""
    log_path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("SYNTHESIS_AUDIT_LOG_PATH", str(log_path))
    yield log_path


# ---------------------------------------------------------------------------
# Group 1 — Construction & schemas
# ---------------------------------------------------------------------------


def test_server_constructs_with_all_five_tools(server: RagbotMCPServer) -> None:
    names = sorted(t.name for t in server.tools())
    assert names == sorted(
        [
            "agent_run_start",
            "document_get",
            "skill_run",
            "workspace_search",
            "workspace_search_multi",
        ]
    )


def test_tool_definitions_include_input_and_output_schemas() -> None:
    for tool in TOOL_DEFINITIONS:
        assert tool.input_schema, f"tool {tool.name} has no input schema"
        assert tool.output_schema, (
            f"tool {tool.name} has no output schema; MCP server tools "
            f"must publish structured-output schemas"
        )
        # Both schemas must be valid JSON Schema "type: object" shapes.
        assert tool.input_schema.get("type") == "object"
        assert tool.output_schema.get("type") == "object"


# ---------------------------------------------------------------------------
# Group 2 — Tool dispatch (direct)
# ---------------------------------------------------------------------------


def test_workspace_search_returns_memory_results(
    dispatch_ctx: ToolDispatchContext,
) -> None:
    result = asyncio.run(
        dispatch_tool(
            "workspace_search",
            {"workspace": "acme-news", "query": "anything", "k": 5},
            dispatch_ctx,
        )
    )
    assert result["workspace"] == "acme-news"
    assert len(result["blocks"]) == 2
    tiers = sorted(b["tier"] for b in result["blocks"])
    assert tiers == ["session", "vector"]


def test_workspace_search_multi_denies_air_gapped_mix(
    dispatch_ctx: ToolDispatchContext,
) -> None:
    with pytest.raises(ToolDispatchError) as excinfo:
        asyncio.run(
            dispatch_tool(
                "workspace_search_multi",
                {
                    "workspaces": ["acme-news", "gamma-secret"],
                    "query": "anything",
                },
                dispatch_ctx,
            )
        )
    assert excinfo.value.code == "confidentiality"
    assert "AIR_GAPPED" in str(excinfo.value)


def test_workspace_search_multi_denies_client_confidential_plus_public(
    dispatch_ctx: ToolDispatchContext,
) -> None:
    with pytest.raises(ToolDispatchError) as excinfo:
        asyncio.run(
            dispatch_tool(
                "workspace_search_multi",
                {
                    "workspaces": ["acme-news", "beta-media"],
                    "query": "anything",
                },
                dispatch_ctx,
            )
        )
    assert excinfo.value.code == "confidentiality"
    assert "PUBLIC" in str(excinfo.value) or "CLIENT_CONFIDENTIAL" in str(
        excinfo.value
    )


def test_workspace_search_multi_records_audit_on_personal_plus_client(
    dispatch_ctx: ToolDispatchContext, isolated_audit_log: Path
) -> None:
    result = asyncio.run(
        dispatch_tool(
            "workspace_search_multi",
            {
                "workspaces": ["acme-user", "beta-media"],
                "query": "synthesis",
            },
            dispatch_ctx,
        )
    )
    assert result["requires_audit"] is True
    assert result["audit_entry_id"] is not None
    # An audit line was written to disk.
    contents = isolated_audit_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(contents) == 1
    entry = json.loads(contents[0])
    assert entry["op_type"] == "mcp.workspace_search_multi"
    assert sorted(entry["workspaces"]) == ["acme-user", "beta-media"]


def test_document_get_returns_content(
    dispatch_ctx: ToolDispatchContext,
) -> None:
    result = asyncio.run(
        dispatch_tool(
            "document_get",
            {"workspace": "acme-news", "document_id": "doc-1"},
            dispatch_ctx,
        )
    )
    assert result["content"].startswith("Lorem ipsum")
    assert result["metadata"]["source"] == "fake-store"


def test_document_get_unknown_id_raises_not_found(
    dispatch_ctx: ToolDispatchContext,
) -> None:
    with pytest.raises(ToolDispatchError) as excinfo:
        asyncio.run(
            dispatch_tool(
                "document_get",
                {"workspace": "acme-news", "document_id": "nope"},
                dispatch_ctx,
            )
        )
    assert excinfo.value.code == "not_found"


def test_skill_run_invokes_runtime(
    server: RagbotMCPServer,
) -> None:
    ctx = server._build_dispatch_context(bearer=None)
    result = asyncio.run(
        dispatch_tool(
            "skill_run",
            {
                "workspace": "acme-news",
                "skill_name": "news-skill",
                "tool_name": "summarise",
                "input": {"text": "hello"},
            },
            ctx,
        )
    )
    assert result["skill"] == "news-skill"
    assert result["tool"] == "summarise"
    fake_runtime: FakeSkillRuntime = server.dependencies.skill_runtime
    assert ("news-skill", "summarise", {"text": "hello"}) in fake_runtime.dispatches


def test_skill_run_denies_when_skill_not_visible(
    server: RagbotMCPServer,
) -> None:
    ctx = server._build_dispatch_context(bearer=None)
    with pytest.raises(ToolDispatchError) as excinfo:
        asyncio.run(
            dispatch_tool(
                "skill_run",
                {
                    "workspace": "beta-media",
                    "skill_name": "news-skill",
                    "tool_name": "summarise",
                    "input": {},
                },
                ctx,
            )
        )
    assert excinfo.value.code == "forbidden"


def test_agent_run_start_returns_task_id_and_status_url(
    dispatch_ctx: ToolDispatchContext,
) -> None:
    result = asyncio.run(
        dispatch_tool(
            "agent_run_start",
            {
                "workspace_or_workspaces": "acme-news",
                "task": "summarise the latest news",
            },
            dispatch_ctx,
        )
    )
    assert result["status"] == "running"
    assert result["task_id"].startswith("task-acme-news-")
    assert result["status_url"].endswith(result["task_id"])


def test_agent_run_start_multi_workspace_passes_gate(
    dispatch_ctx: ToolDispatchContext,
) -> None:
    result = asyncio.run(
        dispatch_tool(
            "agent_run_start",
            {
                "workspace_or_workspaces": ["acme-user", "beta-media"],
                "task": "compose a brief",
            },
            dispatch_ctx,
        )
    )
    assert result["status"] == "running"


def test_agent_run_start_denies_air_gapped_mix(
    dispatch_ctx: ToolDispatchContext,
) -> None:
    with pytest.raises(ToolDispatchError) as excinfo:
        asyncio.run(
            dispatch_tool(
                "agent_run_start",
                {
                    "workspace_or_workspaces": ["acme-news", "gamma-secret"],
                    "task": "anything",
                },
                dispatch_ctx,
            )
        )
    assert excinfo.value.code == "confidentiality"


def test_unknown_tool_raises_dispatch_error(
    dispatch_ctx: ToolDispatchContext,
) -> None:
    with pytest.raises(ToolDispatchError) as excinfo:
        asyncio.run(
            dispatch_tool("nonexistent_tool", {}, dispatch_ctx)
        )
    assert excinfo.value.code == "unknown_tool"


# ---------------------------------------------------------------------------
# Group 3 — Auth
# ---------------------------------------------------------------------------


def _make_auth_yaml(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_bearer_token_allows_glob() -> None:
    token = BearerToken(
        name="cursor", token="secret", allowed_tools=("workspace_*",)
    )
    assert token.allows_tool("workspace_search") is True
    assert token.allows_tool("workspace_search_multi") is True
    assert token.allows_tool("skill_run") is False


def test_bearer_token_wildcard_allows_everything() -> None:
    token = BearerToken(name="all", token="x", allowed_tools=("*",))
    assert token.allows_tool("workspace_search") is True
    assert token.allows_tool("agent_run_start") is True


def test_bearer_token_empty_allows_nothing() -> None:
    token = BearerToken(name="none", token="x", allowed_tools=())
    assert token.allows_tool("workspace_search") is False


def test_auth_config_authenticates_bearer() -> None:
    cfg = MCPServerAuthConfig(
        tokens=(
            BearerToken(name="a", token="aaa", allowed_tools=("*",)),
            BearerToken(name="b", token="bbb", allowed_tools=("workspace_*",)),
        )
    )
    assert cfg.authenticate_bearer("Bearer aaa").name == "a"
    assert cfg.authenticate_bearer("bbb").name == "b"
    assert cfg.authenticate_bearer("nope") is None
    assert cfg.authenticate_bearer(None) is None
    assert cfg.authenticate_bearer("") is None


def test_load_auth_config_http_mode_fails_when_missing(tmp_path: Path) -> None:
    missing = tmp_path / "absent.yaml"
    with pytest.raises(MCPServerAuthError) as excinfo:
        load_auth_config(require=True, config_path=missing)
    assert "missing" in str(excinfo.value).lower() or "absent" in str(
        excinfo.value
    ).lower() or "create" in str(excinfo.value).lower()


def test_load_auth_config_stdio_mode_returns_none_when_missing(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "absent.yaml"
    cfg = load_auth_config(require=False, config_path=missing)
    assert cfg is None


def test_load_auth_config_parses_yaml(tmp_path: Path) -> None:
    body = textwrap.dedent(
        """
        bearer_tokens:
          - name: claude-code-local
            token: AAA
            allowed_tools: ["workspace_search", "skill_run"]
          - name: cursor-laptop
            token: BBB
            allowed_tools: ["*"]
        """
    ).strip()
    cfg_path = tmp_path / "mcp-server.yaml"
    _make_auth_yaml(cfg_path, body)
    cfg = load_auth_config(require=True, config_path=cfg_path)
    assert cfg is not None
    assert len(cfg.tokens) == 2
    claude = cfg.authenticate_bearer("Bearer AAA")
    assert claude is not None
    assert claude.allows_tool("workspace_search") is True
    assert claude.allows_tool("agent_run_start") is False
    cursor = cfg.authenticate_bearer("BBB")
    assert cursor is not None
    assert cursor.allows_tool("agent_run_start") is True


def test_load_auth_config_rejects_duplicate_tokens(tmp_path: Path) -> None:
    body = textwrap.dedent(
        """
        bearer_tokens:
          - name: a
            token: SAME
            allowed_tools: ["*"]
          - name: b
            token: SAME
            allowed_tools: ["*"]
        """
    ).strip()
    cfg_path = tmp_path / "mcp-server.yaml"
    _make_auth_yaml(cfg_path, body)
    with pytest.raises(MCPServerAuthError):
        load_auth_config(require=True, config_path=cfg_path)


def test_load_auth_config_rejects_empty_token_list(tmp_path: Path) -> None:
    body = "bearer_tokens: []\n"
    cfg_path = tmp_path / "mcp-server.yaml"
    _make_auth_yaml(cfg_path, body)
    with pytest.raises(MCPServerAuthError):
        load_auth_config(require=True, config_path=cfg_path)


def test_dispatch_enforces_bearer_token_allowed_tools(
    server: RagbotMCPServer,
) -> None:
    restricted = BearerToken(
        name="restricted", token="r", allowed_tools=("workspace_*",)
    )
    ctx = server._build_dispatch_context(bearer=restricted)
    # Allowed.
    asyncio.run(
        dispatch_tool(
            "workspace_search",
            {"workspace": "acme-news", "query": "hello"},
            ctx,
        )
    )
    # Forbidden — agent_run_start does not match the glob.
    with pytest.raises(ToolDispatchError) as excinfo:
        asyncio.run(
            dispatch_tool(
                "agent_run_start",
                {
                    "workspace_or_workspaces": "acme-news",
                    "task": "do a thing",
                },
                ctx,
            )
        )
    assert excinfo.value.code == "forbidden"


def test_stdio_mode_skips_bearer_check(
    server: RagbotMCPServer,
) -> None:
    # bearer=None mirrors stdio mode. Every tool dispatches without an
    # ``allowed_tools`` check.
    ctx = server._build_dispatch_context(bearer=None)
    for name in (
        "workspace_search",
        "document_get",
        "skill_run",
        "agent_run_start",
    ):
        # Pick representative valid arguments per tool.
        if name == "workspace_search":
            args: Dict[str, Any] = {
                "workspace": "acme-news",
                "query": "x",
            }
        elif name == "document_get":
            args = {"workspace": "acme-news", "document_id": "doc-1"}
        elif name == "skill_run":
            args = {
                "workspace": "acme-news",
                "skill_name": "news-skill",
                "tool_name": "summarise",
                "input": {},
            }
        else:
            args = {
                "workspace_or_workspaces": "acme-news",
                "task": "t",
            }
        asyncio.run(dispatch_tool(name, args, ctx))


# ---------------------------------------------------------------------------
# Group 3b — Permission registry
# ---------------------------------------------------------------------------


def test_permission_registry_gate_can_deny_tool(
    monkeypatch,
) -> None:
    registry = PermissionRegistry()
    registry.register(
        "workspace_search",
        lambda _ctx: PermissionResult.deny("policy denies search"),
    )
    deps = build_dependencies(permission_registry=registry)
    s = RagbotMCPServer(dependencies=deps)
    ctx = s._build_dispatch_context(bearer=None)
    with pytest.raises(ToolDispatchError) as excinfo:
        asyncio.run(
            dispatch_tool(
                "workspace_search",
                {"workspace": "acme-news", "query": "x"},
                ctx,
            )
        )
    assert excinfo.value.code == "permission"


# ---------------------------------------------------------------------------
# Group 4 — Resources
# ---------------------------------------------------------------------------


def test_resource_listing_includes_workspaces_skills_and_audit(
    server: RagbotMCPServer,
) -> None:
    resources = asyncio.run(_get_resource_list(server))
    uris = sorted(str(r.uri) for r in resources)
    # Workspace resources for every known workspace.
    for ws in ("acme-news", "acme-user", "beta-media", "gamma-secret"):
        assert any(uri.endswith(f"/workspaces/{ws}") for uri in uris), (
            f"missing workspace resource for {ws!r}; got {uris!r}"
        )
    # At least one skill resource (acme-news has news-skill).
    assert any(uri.endswith("/skills/acme-news/news-skill") for uri in uris)
    # Audit resource present exactly once.
    audit_uris = [u for u in uris if u.endswith("audit/recent")]
    assert len(audit_uris) == 1


def test_reading_workspace_resource_returns_metadata(
    server: RagbotMCPServer,
) -> None:
    contents = list(
        _read_resource(
            server, f"{WORKSPACE_RESOURCE_PREFIX}acme-news"
        )
    )
    assert len(contents) == 1
    payload = json.loads(contents[0].content)
    assert payload["workspace"] == "acme-news"
    assert payload["confidentiality"] == "PUBLIC"


def test_reading_unknown_resource_raises(
    server: RagbotMCPServer,
) -> None:
    with pytest.raises(LookupError):
        list(_read_resource(server, "synthesis://nope/whatever"))


def test_reading_skill_resource_returns_body(
    server: RagbotMCPServer,
) -> None:
    # First, give the news-skill a body so the read has something to surface.
    server.dependencies.skills_visible_for("acme-news")[0].body = (
        "# news skill body\n\nDo news things."
    )
    contents = list(
        _read_resource(
            server, f"{SKILL_RESOURCE_PREFIX}acme-news/news-skill"
        )
    )
    assert any("news skill body" in c.content for c in contents)


def test_reading_audit_resource_returns_jsonl(
    server: RagbotMCPServer, isolated_audit_log: Path
) -> None:
    # Trigger an audit-logging op so the resource has something to surface.
    ctx = server._build_dispatch_context(bearer=None)
    asyncio.run(
        dispatch_tool(
            "workspace_search_multi",
            {
                "workspaces": ["acme-user", "beta-media"],
                "query": "audit",
            },
            ctx,
        )
    )
    contents = list(_read_resource(server, AUDIT_RESOURCE_URI))
    assert len(contents) == 1
    body = contents[0].content
    # One JSON line per audit entry; we expect at least one.
    assert body.strip()
    for line in body.strip().splitlines():
        json.loads(line)


# ---------------------------------------------------------------------------
# Group 5 — End-to-end (in-memory MCP client + server)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_to_end_in_memory_client_calls_workspace_search(
    server: RagbotMCPServer,
) -> None:
    """Spin up an in-process client + server pair and call workspace_search.

    ``mcp.Client(server)`` is the SDK's in-process transport. This
    exercises the full framing path: the registered ``list_tools`` and
    ``call_tool`` handlers see real ``tools/call`` envelopes and produce
    real :class:`CallToolResult` payloads.
    """
    from mcp import Client

    async with Client(server.server, raise_exceptions=True) as client:
        tools_listing = await client.list_tools()
        names = {t.name for t in tools_listing.tools}
        assert {
            "workspace_search",
            "workspace_search_multi",
            "document_get",
            "skill_run",
            "agent_run_start",
        } <= names

        result = await client.call_tool(
            "workspace_search",
            arguments={"workspace": "acme-news", "query": "hello"},
        )
        assert result.is_error is False
        # Structured output is preserved, alongside its JSON text.
        assert result.structured_content is not None
        sc = result.structured_content
        assert json.loads(result.content[0].text) == sc
        assert sc["workspace"] == "acme-news"
        assert len(sc["blocks"]) == 2


@pytest.mark.asyncio
async def test_end_to_end_in_memory_client_reads_audit_resource(
    server: RagbotMCPServer, isolated_audit_log: Path
) -> None:
    """Round-trip a resource read over the in-process transport."""
    from mcp import Client

    # Pre-populate the audit log via a dispatch call.
    ctx = server._build_dispatch_context(bearer=None)
    await dispatch_tool(
        "workspace_search_multi",
        {
            "workspaces": ["acme-user", "beta-media"],
            "query": "via mcp",
        },
        ctx,
    )

    async with Client(server.server, raise_exceptions=True) as client:
        listing = await client.list_resources()
        uris = [str(r.uri) for r in listing.resources]
        assert AUDIT_RESOURCE_URI in uris

        read = await client.read_resource(AUDIT_RESOURCE_URI)
        assert read.contents
        # The audit resource returns text content (JSONL).
        text = "".join(getattr(c, "text", "") for c in read.contents)
        assert text.strip()
        first = json.loads(text.strip().splitlines()[0])
        assert first["op_type"] == "mcp.workspace_search_multi"


# ---------------------------------------------------------------------------
# Internal helpers (resource list / read shims)
# ---------------------------------------------------------------------------


async def _get_resource_list(server: RagbotMCPServer):
    from mcp import Client

    async with Client(server.server, raise_exceptions=True) as client:
        result = await client.list_resources()
    return result.resources


def _read_resource(server: RagbotMCPServer, uri: str):
    """Synchronous wrapper to drive the resource resolver directly."""
    from synthesis_engine.mcp_server.resources import read_resource_contents

    return read_resource_contents(server._resource_provider, uri)


# ---------------------------------------------------------------------------
# Group 6 — Tool-call semantics the server supplies itself (mcp 2.x)
#
# The 1.x SDK validated arguments, packaged structured output and turned
# exceptions into tool errors on the server's behalf. The 2.x low-level
# server leaves all of that to the handler; these tests pin it down.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_tool_rejects_arguments_that_fail_the_input_schema(
    server: RagbotMCPServer,
) -> None:
    from mcp import Client

    async with Client(server.server, raise_exceptions=True) as client:
        result = await client.call_tool("workspace_search", arguments={"workspace": "acme-news"})
    assert result.is_error is True
    assert result.content[0].text.startswith("Input validation error:")
    assert "query" in result.content[0].text


@pytest.mark.asyncio
async def test_call_tool_reports_dispatch_errors_as_tool_errors(
    server: RagbotMCPServer,
) -> None:
    from mcp import Client

    async with Client(server.server, raise_exceptions=True) as client:
        result = await client.call_tool("no_such_tool", arguments={})
    assert result.is_error is True
    assert result.content[0].text.startswith("[")


@pytest.mark.asyncio
async def test_call_tool_rejects_output_that_fails_the_output_schema(
    server: RagbotMCPServer, monkeypatch
) -> None:
    from mcp import Client

    import synthesis_engine.mcp_server.server as server_module

    async def bad_dispatch(name, arguments, ctx):
        return {"workspace": "acme-news"}  # missing required "blocks"

    monkeypatch.setattr(server_module, "dispatch_tool", bad_dispatch)
    async with Client(server.server, raise_exceptions=True) as client:
        result = await client.call_tool(
            "workspace_search", arguments={"workspace": "acme-news", "query": "q"}
        )
    assert result.is_error is True
    assert result.content[0].text.startswith("Output validation error:")


@pytest.mark.asyncio
async def test_reading_unknown_resource_over_mcp_is_invalid_params(
    server: RagbotMCPServer,
) -> None:
    from mcp import Client, MCPError
    from mcp.types import INVALID_PARAMS

    async with Client(server.server) as client:
        with pytest.raises(MCPError) as excinfo:
            await client.read_resource("synthesis://nope/whatever")
    assert excinfo.value.error.code == INVALID_PARAMS


@pytest.mark.asyncio
async def test_http_request_without_auth_config_fails_closed(
    server: RagbotMCPServer,
) -> None:
    """A request that carries an HTTP request but no auth config is refused.

    Stdio and in-process sessions have no HTTP request and run with
    process-local trust. Anything that arrives over HTTP must resolve a
    bearer token; without an auth config there is nothing to resolve it
    against, so the call is an error rather than a silent stdio-trust run.
    """
    from types import SimpleNamespace

    from mcp.types import CallToolRequestParams

    ctx = SimpleNamespace(request=SimpleNamespace(headers={"authorization": "Bearer x"}))
    result = await server._on_call_tool(
        ctx,
        CallToolRequestParams(
            name="workspace_search", arguments={"workspace": "acme-news", "query": "q"}
        ),
    )
    assert result.is_error is True
    assert "[unauthorized]" in result.content[0].text


@pytest.mark.asyncio
async def test_http_transport_enforces_bearer_tokens_end_to_end(
    server: RagbotMCPServer,
) -> None:
    """Serve over real Streamable HTTP and check both auth layers.

    Without a token the ASGI layer answers 401. With a token restricted to
    ``workspace_*`` the allowed tool runs and ``agent_run_start`` comes
    back as a ``[forbidden]`` tool error, which shows the handler resolved
    the token from the request it was actually serving.
    """
    import socket

    import httpx2
    import uvicorn
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    auth = MCPServerAuthConfig(
        tokens=(BearerToken(name="search", token="search-token", allowed_tools=("workspace_*",)),)
    )
    app = server.asgi_app_factory(auth_config=auth)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    uv = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on")
    )
    serve_task = asyncio.create_task(uv.serve())
    try:
        for _ in range(200):
            if uv.started:
                break
            await asyncio.sleep(0.025)
        assert uv.started, "uvicorn did not start"
        url = f"http://127.0.0.1:{port}/mcp"

        async with httpx2.AsyncClient() as http:
            unauthenticated = await http.post(url, json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
        assert unauthenticated.status_code == 401

        async with httpx2.AsyncClient(headers={"Authorization": "Bearer search-token"}) as http:
            async with streamable_http_client(url, http_client=http) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    allowed = await session.call_tool(
                        "workspace_search", {"workspace": "acme-news", "query": "hello"}
                    )
                    assert allowed.is_error is False
                    assert allowed.structured_content["workspace"] == "acme-news"
                    denied = await session.call_tool(
                        "agent_run_start",
                        {"workspace_or_workspaces": "acme-news", "task": "do a thing"},
                    )
                    assert denied.is_error is True
                    assert "[forbidden]" in denied.content[0].text
    finally:
        uv.should_exit = True
        await serve_task
