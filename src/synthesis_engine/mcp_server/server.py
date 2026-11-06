"""RagbotMCPServer — wraps :class:`mcp.server.lowlevel.Server`.

This module is the transport-level adapter. It binds the tool and
resource handlers defined in :mod:`.tools` and :mod:`.resources` to a
:class:`mcp.server.lowlevel.Server`, then exposes two ways to serve
that server to an external client:

* :meth:`RagbotMCPServer.serve_stdio` — blocks on stdin/stdout. The
  standard transport for desktop integrations (Claude Code, Cursor,
  ChatGPT desktop). Auth is process-local; the parent process is
  trusted.

* :meth:`RagbotMCPServer.serve_http` — serves the same handlers over
  Streamable HTTP at ``/mcp``. Bearer-token auth is enforced via the
  per-server config at ``~/.synthesis/mcp-server.yaml``. The method
  raises :class:`MCPServerAuthError` and refuses to start when the
  config is missing or malformed.

The class is a thin shell. Transport, framing and JSON-RPC are the
SDK's responsibility. The mcp 2.x low-level server leaves tool-call
semantics to the handler, so this module validates arguments against
each tool's input schema, returns structured content alongside its JSON
text, validates it against the output schema, and turns failures into
``is_error`` results — the behaviour the 1.x SDK supplied implicitly.
Every tool-level concern (dispatch, permission, policy) lives in
:mod:`.tools`. Every resource-level concern (URI parsing,
content materialisation) lives in :mod:`.resources`. The server is the
glue that puts those layers behind one MCP-compliant socket.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from dataclasses import dataclass
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
)

import jsonschema
from mcp.server.context import ServerRequestContext
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError
from mcp.types import (
    INVALID_PARAMS,
    BlobResourceContents,
    CallToolRequestParams,
    CallToolResult,
    ListResourcesResult,
    ListToolsResult,
    PaginatedRequestParams,
    ReadResourceRequestParams,
    ReadResourceResult,
    TextContent,
    TextResourceContents,
    Tool,
)

from ..policy.audit import read_recent as audit_read_recent  # noqa: F401  # re-export for tests
from .auth import (
    BearerToken,
    MCPServerAuthConfig,
    MCPServerAuthError,
    load_auth_config,
)
from .resources import (
    ResourceProvider,
    list_all_resources,
    read_resource_contents,
)
from .tools import (
    TOOL_DEFINITIONS,
    ToolDispatchContext,
    ToolDispatchError,
    dispatch_tool,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Server dependencies
# ---------------------------------------------------------------------------


@dataclass
class ServerDependencies:
    """Runtime collaborators required by :class:`RagbotMCPServer`.

    All collaborators are passed in by the caller (typically a FastAPI
    lifespan handler in production, or a test fixture). The server
    never imports the substrate's process-singletons; this keeps the
    server testable in isolation and makes the wiring explicit at the
    composition root.

    Attributes:
        memory:               Substrate :class:`Memory` backend.
        skill_runtime:        :class:`SkillRuntime` for ``skill_run``.
        skills_visible_for:   Callable ``workspace -> List[Skill]``.
        list_workspaces:      Callable ``() -> Iterable[str]`` returning
                              every known workspace name.
        routing_policies:     Callable ``workspace -> RoutingPolicy``.
        document_getter:      Callable ``(workspace, document_id) -> dict``.
        agent_run_starter:    Async callable ``(workspaces, task, rubric)
                              -> (task_id, status_url)``.
        permission_registry:  Optional :class:`PermissionRegistry` for
                              cross-cutting policy.
        audit_limit:          Max audit entries surfaced through the
                              ``synthesis://audit/recent`` resource.
        status_url_template:  Template for the status URL returned from
                              ``agent_run_start``.
        retrieve_single:      Test seam for the single-workspace
                              retriever. Defaults to the real one.
        retrieve_multi:       Test seam for the multi-workspace
                              retriever. Defaults to the real one.
    """

    memory: Any
    skill_runtime: Any
    skills_visible_for: Callable[[str], Iterable[Any]]
    list_workspaces: Callable[[], Iterable[str]]
    routing_policies: Callable[[str], Any]
    document_getter: Callable[[str, str], Dict[str, Any]]
    agent_run_starter: Callable[
        [Tuple[str, ...], str, Optional[str]], Awaitable[Tuple[str, str]]
    ]
    permission_registry: Optional[Any] = None
    audit_limit: int = 200
    status_url_template: str = "/api/agent/sessions/{task_id}"
    retrieve_single: Optional[Callable[..., List[Any]]] = None
    retrieve_multi: Optional[Callable[..., List[Any]]] = None


# ---------------------------------------------------------------------------
# RagbotMCPServer
# ---------------------------------------------------------------------------


class RagbotMCPServer:
    """Expose Ragbot's primitives as an MCP server.

    Construction
    ------------

    The server is constructed with a :class:`ServerDependencies`
    instance. The dependencies object is the single seam between the
    substrate (memory, skills, policy, agent loop) and this adapter.
    """

    DEFAULT_NAME = "ragbot"

    def __init__(
        self,
        dependencies: ServerDependencies,
        *,
        name: str = DEFAULT_NAME,
        version: str = "1.0.0",
    ) -> None:
        self._deps = dependencies
        self._tools_by_name: Dict[str, Tool] = {t.name: t for t in TOOL_DEFINITIONS}
        # Set by :meth:`asgi_app_factory`. HTTP requests resolve their bearer
        # token against it; stdio and in-process sessions never consult it.
        self._auth_config: Optional[MCPServerAuthConfig] = None
        self._resource_provider = ResourceProvider(
            list_workspaces=dependencies.list_workspaces,
            skills_for=lambda ws: list(dependencies.skills_visible_for(ws)),
            routing_policy_for=dependencies.routing_policies,
            audit_limit=dependencies.audit_limit,
        )
        self._server: Server = Server(
            name=name,
            version=version,
            get_tool_input_schema=self._tool_input_schema,
            on_list_tools=self._on_list_tools,
            on_call_tool=self._on_call_tool,
            on_list_resources=self._on_list_resources,
            on_read_resource=self._on_read_resource,
        )

    # ------ public introspection -------------------------------------------

    @property
    def server(self) -> Server:
        """The underlying SDK :class:`Server` instance.

        Exposed so tests can connect to it in-process with
        ``mcp.Client(server.server)`` and inspect its capabilities.
        """
        return self._server

    @property
    def dependencies(self) -> ServerDependencies:
        return self._deps

    def tools(self) -> Tuple[Tool, ...]:
        """Return the static tool definitions surfaced by this server."""
        return TOOL_DEFINITIONS

    # ------ protocol handlers ----------------------------------------------

    def _tool_input_schema(self, name: str) -> Optional[Mapping[str, Any]]:
        tool = self._tools_by_name.get(name)
        return None if tool is None else tool.input_schema

    async def _on_list_tools(
        self, ctx: ServerRequestContext[Any], params: Optional[PaginatedRequestParams]
    ) -> ListToolsResult:
        return ListToolsResult(tools=list(TOOL_DEFINITIONS))

    async def _on_call_tool(
        self, ctx: ServerRequestContext[Any], params: CallToolRequestParams
    ) -> CallToolResult:
        """Validate, dispatch and package one ``tools/call``.

        Every failure becomes an ``is_error`` result rather than a JSON-RPC
        error, so a client sees the tool's own message: schema violations,
        dispatch refusals (``[code] message``) and unexpected exceptions.
        """
        name = params.name
        arguments = params.arguments or {}
        tool = self._tools_by_name.get(name)
        try:
            if tool is not None:
                try:
                    jsonschema.validate(instance=arguments, schema=tool.input_schema)
                except jsonschema.ValidationError as exc:
                    return _error_result(f"Input validation error: {exc.message}")
            bearer, authorized = self._bearer_for(ctx)
            if not authorized:
                return _error_result(
                    "[unauthorized] Missing or invalid bearer token for this request."
                )
            dispatch_ctx = self._build_dispatch_context(bearer)
            try:
                structured = await dispatch_tool(name, arguments, dispatch_ctx)
            except ToolDispatchError as exc:
                return _error_result(f"[{exc.code}] {exc}")
            if tool is not None and tool.output_schema is not None:
                try:
                    jsonschema.validate(instance=structured, schema=tool.output_schema)
                except jsonschema.ValidationError as exc:
                    return _error_result(f"Output validation error: {exc.message}")
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(structured, indent=2))],
                structured_content=structured,
                is_error=False,
            )
        except Exception as exc:
            logger.exception("tool %s failed", name)
            return _error_result(str(exc))

    async def _on_list_resources(
        self, ctx: ServerRequestContext[Any], params: Optional[PaginatedRequestParams]
    ) -> ListResourcesResult:
        return ListResourcesResult(resources=list_all_resources(self._resource_provider))

    async def _on_read_resource(
        self, ctx: ServerRequestContext[Any], params: ReadResourceRequestParams
    ) -> ReadResourceResult:
        uri = str(params.uri)
        try:
            items = list(read_resource_contents(self._resource_provider, uri))
        except LookupError as exc:
            raise MCPError(code=INVALID_PARAMS, message=str(exc)) from exc
        contents: List[TextResourceContents | BlobResourceContents] = []
        for item in items:
            if isinstance(item.content, bytes):
                contents.append(
                    BlobResourceContents(
                        uri=uri,
                        blob=base64.b64encode(item.content).decode("ascii"),
                        mime_type=item.mime_type or "application/octet-stream",
                        _meta=item.meta,
                    )
                )
            else:
                contents.append(
                    TextResourceContents(
                        uri=uri,
                        text=item.content,
                        mime_type=item.mime_type or "text/plain",
                        _meta=item.meta,
                    )
                )
        return ReadResourceResult(contents=contents)

    def _bearer_for(
        self, ctx: ServerRequestContext[Any]
    ) -> Tuple[Optional[BearerToken], bool]:
        """Resolve the bearer token for one request; return ``(token, authorized)``.

        ``ctx.request`` is the HTTP request on the Streamable HTTP
        transport and ``None`` for stdio and in-process sessions. Stdio is
        process-local trust, so it dispatches with no token. An HTTP
        request must carry a token that resolves against the auth config;
        the ASGI layer already refuses requests without one, and this
        check fails closed in case a request reaches the handler some
        other way.
        """
        request = ctx.request
        if request is None:
            return None, True
        if self._auth_config is None:
            return None, False
        headers = getattr(request, "headers", None) or {}
        bearer = self._auth_config.authenticate_bearer(headers.get("authorization"))
        return bearer, bearer is not None

    def _build_dispatch_context(
        self, bearer: Optional[BearerToken]
    ) -> ToolDispatchContext:
        """Compose a :class:`ToolDispatchContext` for one request."""
        deps = self._deps
        return ToolDispatchContext(
            memory=deps.memory,
            skill_runtime=deps.skill_runtime,
            skills_visible_for=lambda ws: list(deps.skills_visible_for(ws)),
            document_getter=deps.document_getter,
            agent_run_starter=deps.agent_run_starter,
            routing_policies=deps.routing_policies,
            permission_registry=deps.permission_registry,
            bearer_token=bearer,
            retrieve_single=deps.retrieve_single,
            retrieve_multi=deps.retrieve_multi,
            status_url_template=deps.status_url_template,
        )

    # ------ transport: stdio -----------------------------------------------

    async def serve_stdio(self) -> None:
        """Serve the MCP server over stdin/stdout. Auth is bypassed.

        Stdio is process-local: the client (Claude Code, Cursor, etc.)
        spawned this process and shares its security boundary. No
        bearer token is consulted; requests carry no HTTP request, which
        is how :meth:`_bearer_for` recognises the stdio-trust path.
        """
        async with stdio_server() as (read_stream, write_stream):
            await self._server.run(
                read_stream,
                write_stream,
                self._server.create_initialization_options(),
            )

    # ------ transport: HTTP/SSE --------------------------------------------

    async def serve_http(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        auth_config: Optional[MCPServerAuthConfig] = None,
        config_path: Optional[str] = None,
    ) -> None:
        """Serve the MCP server over Streamable HTTP with bearer-token auth.

        Args:
            host: Bind host. Defaults to localhost so a misconfigured
                deployment does not silently expose the server to the
                public internet.
            port: Bind port.
            auth_config: Pre-loaded auth config. When omitted the
                method calls :func:`load_auth_config` with
                ``require=True``; the call raises
                :class:`MCPServerAuthError` when
                ``~/.synthesis/mcp-server.yaml`` is missing or
                malformed.
            config_path: Optional override for the auth-config path.
                Passed through to :func:`load_auth_config`.

        Raises:
            MCPServerAuthError: when the HTTP-mode auth config cannot
                be loaded. Fail-closed; the server does not start.

        Note:
            The endpoint is ``http://<host>:<port>/mcp``. The default
            in-process Uvicorn binding is the simplest standalone path;
            use :meth:`asgi_app_factory` to embed the server inside an
            existing ASGI app.
        """
        if auth_config is None:
            cfg_path = (
                None if config_path is None else os.path.expanduser(config_path)
            )
            from pathlib import Path

            auth_config = load_auth_config(
                require=True,
                config_path=Path(cfg_path) if cfg_path else None,
            )
        # An empty token list would have raised inside load_auth_config
        # when require=True; assert here for defence in depth.
        if auth_config is None or not auth_config.tokens:
            raise MCPServerAuthError(
                "HTTP/SSE transport requires at least one bearer token. "
                "Add one to ~/.synthesis/mcp-server.yaml and restart."
            )

        # Lazy import: uvicorn is only needed for the default in-process
        # binding. Embedding the server in an existing ASGI app uses
        # :meth:`asgi_app_factory` and never imports uvicorn here.
        import uvicorn

        app = self.asgi_app_factory(auth_config=auth_config, host=host)
        config = uvicorn.Config(
            app=app, host=host, port=port, log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()

    def asgi_app_factory(
        self,
        *,
        auth_config: MCPServerAuthConfig,
        host: str = "127.0.0.1",
        path: str = "/mcp",
    ) -> Callable[..., Awaitable[None]]:
        """Return an ASGI callable that serves this MCP server at ``path``.

        The SDK's Streamable HTTP app owns the transport and its session
        manager lifecycle (the ASGI lifespan starts and stops it). This
        factory puts a bearer-token check in front of it: requests whose
        ``Authorization`` header does not resolve against ``auth_config``
        get a 401 with a clear remediation message and never reach the
        transport. The tool handler resolves the same header again for
        each call to apply the token's ``allowed_tools``.

        ``host`` is the address the server binds to; the SDK uses it for
        DNS-rebinding protection on local binds.
        """
        self._auth_config = auth_config
        inner = self._server.streamable_http_app(
            streamable_http_path=path, host=host
        )

        async def asgi_app(scope: Dict[str, Any], receive: Any, send: Any) -> None:
            if scope.get("type") != "http":
                await inner(scope, receive, send)
                return
            headers = {
                k.decode("latin-1").lower(): v.decode("latin-1")
                for k, v in (scope.get("headers") or [])
            }
            if auth_config.authenticate_bearer(headers.get("authorization")) is None:
                body = (
                    b'{"error":"unauthorized",'
                    b'"detail":"Missing or invalid bearer token. '
                    b'Authorization: Bearer <token>."}'
                )
                await send(
                    {
                        "type": "http.response.start",
                        "status": 401,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"www-authenticate", b'Bearer realm="ragbot-mcp"'),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": body})
                return
            await inner(scope, receive, send)

        return asgi_app


def _error_result(message: str) -> CallToolResult:
    """A ``tools/call`` result reporting ``message`` as a tool error."""
    return CallToolResult(
        content=[TextContent(type="text", text=message)], is_error=True
    )


# ---------------------------------------------------------------------------
# Convenience: build a default agent-run starter from an AgentLoop
# ---------------------------------------------------------------------------


def make_agent_run_starter_from_loop(
    loop: Any,
    *,
    status_url_template: str = "/api/agent/sessions/{task_id}",
) -> Callable[
    [Tuple[str, ...], str, Optional[str]], Awaitable[Tuple[str, str]]
]:
    """Build an ``agent_run_starter`` callable wrapping an :class:`AgentLoop`.

    The returned async callable matches the contract expected by
    :class:`ServerDependencies.agent_run_starter`. It mirrors the
    background-task pattern in :mod:`api.routers.agent`: build an
    initial :class:`GraphState`, save it via the loop's checkpoint
    store, kick off ``drive_to_terminal`` as a background asyncio task,
    and return the new task id.
    """
    from ..agent import GraphState  # local import to avoid cycle

    async def _starter(
        workspaces: Tuple[str, ...],
        task: str,
        rubric: Optional[str],
    ) -> Tuple[str, str]:
        state = GraphState.new(task)
        if workspaces:
            state.metadata["workspaces"] = list(workspaces)
        if rubric:
            state.metadata["rubric"] = rubric
            state.metadata["pending_grade"] = True
        state.add_turn(state.current_state, "MCP-initiated agent run.")
        await loop.checkpoint_store.save(state)

        async def _drive() -> None:
            try:
                await loop.drive_to_terminal(state)
            except Exception:  # pragma: no cover - defensive
                logger.exception(
                    "MCP-initiated agent run %s raised", state.task_id
                )

        asyncio.create_task(_drive())
        status_url = status_url_template.format(task_id=state.task_id)
        return state.task_id, status_url

    return _starter


__all__ = [
    "RagbotMCPServer",
    "ServerDependencies",
    "make_agent_run_starter_from_loop",
]
