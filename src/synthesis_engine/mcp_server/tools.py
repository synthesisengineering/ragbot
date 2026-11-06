"""MCP tool definitions exposed by :class:`RagbotMCPServer`.

The five tools surfaced to remote MCP clients:

* :data:`TOOL_WORKSPACE_SEARCH`        — single-workspace three-tier search.
* :data:`TOOL_WORKSPACE_SEARCH_MULTI`  — multi-workspace search with the
  cross-workspace confidentiality gate.
* :data:`TOOL_DOCUMENT_GET`            — fetch one document by id.
* :data:`TOOL_SKILL_RUN`               — invoke a skill-declared tool
  through :class:`SkillRuntime`.
* :data:`TOOL_AGENT_RUN_START`         — start an agent loop in the
  background and return a task id + status url.

Every tool routes through the same permission registry and policy gates
the rest of Ragbot uses. The MCP server is a transport adapter on top
of the substrate, not a parallel implementation of any of these
primitives.

Tool definitions
================

The :class:`mcp.types.Tool` instances live in :data:`TOOL_DEFINITIONS`.
Each definition includes a complete JSON Schema for both ``inputSchema``
and ``outputSchema`` so an MCP client knows the call shape without
having to read documentation.

Dispatch contract
=================

:func:`dispatch_tool` is the single entry point the server's
``call_tool`` handler forwards to. It takes a tool name, the validated
arguments, and a :class:`ToolDispatchContext` carrying the runtime
dependencies (memory, skill runtime, agent-run starter, policy
registries, bearer-token entry for HTTP mode). The function:

1. Looks up the tool by name; raises :class:`ToolDispatchError` for
   unknown names.
2. (HTTP mode only) Confirms the bearer token is allowed to call this
   tool. Stdio mode skips this check because the transport is
   process-local.
3. Forwards to the per-tool implementation, returning the structured
   output as a plain ``dict`` so the SDK's ``call_tool`` decorator can
   serialize it as both a JSON ``structuredContent`` and a
   pretty-printed ``TextContent`` block.

Errors raised inside the per-tool implementations bubble out as
:class:`ToolDispatchError` so the SDK turns them into MCP error
responses. The error message is suitable for surfacing directly to the
end user.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from mcp.types import Tool

from ..agent.permissions import PermissionRegistry, ToolCallContext
from ..policy.audit import AuditEntry, record as audit_record
from ..policy.confidentiality import (
    ACTIVE_WORKSPACES_METADATA_KEY,
    ROUTING_POLICIES_METADATA_KEY,
    check_cross_workspace_op,
)
from ..policy.routing import RoutingPolicy
from .auth import BearerToken


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ToolDispatchError(Exception):
    """Raised when a tool call cannot be dispatched.

    Carries a stable ``code`` so an MCP client can map the failure
    without parsing the human-readable text. Codes used in this module:

    * ``unknown_tool``      — the tool name has no implementation.
    * ``forbidden``         — the bearer token is not authorised for
      this tool.
    * ``not_found``         — the resource (document, skill, workspace)
      does not exist.
    * ``confidentiality``   — the cross-workspace gate denied the op.
    * ``permission``        — the underlying permission registry denied
      the op.
    * ``invalid_argument``  — the arguments fail a runtime check the
      JSON Schema did not catch.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# Dispatch context
# ---------------------------------------------------------------------------


# Type alias for the agent-run starter the dispatcher calls when
# ``agent_run_start`` is invoked. The concrete implementation lives in the
# server (it builds an asyncio task), but the contract here is just
# ``(workspaces, task, rubric) -> (task_id, status_url)``.
AgentRunStarter = Callable[
    [Tuple[str, ...], str, Optional[str]],
    Awaitable[Tuple[str, str]],
]


# Type alias for the document-fetch hook. The substrate's memory layer
# does not yet have a uniform document-by-id API; the server takes a
# small callable so tests can inject a fake without having to spin up a
# full Memory backend. Returns a dict that becomes the tool's structured
# output, or raises ``KeyError`` for not-found.
DocumentGetter = Callable[[str, str], Dict[str, Any]]


@dataclass
class ToolDispatchContext:
    """Runtime dependencies the tool implementations require.

    The context is built once by :class:`RagbotMCPServer` and passed
    through on every dispatch. Keeping it as a dataclass means the
    transport layer (stdio vs HTTP) can swap in a different bearer-token
    entry per request without rebuilding the rest of the dependencies.

    Attributes:
        memory:               Substrate memory backend. The
                              ``three_tier_retrieve`` /
                              ``three_tier_retrieve_multi`` functions
                              consume it; tests pass a fake that
                              implements just the methods the retriever
                              touches.
        skill_runtime:        :class:`SkillRuntime` instance for
                              ``skill_run`` dispatch.
        skills_visible_for:   Callable ``workspace -> List[Skill]`` used
                              by ``skill_run`` to gate which skills a
                              given workspace can see. The default
                              implementation goes through
                              ``get_skills_for_workspace``.
        document_getter:      Callable ``(workspace, document_id) ->
                              dict`` for ``document_get``. Raises
                              ``KeyError`` when the document is missing.
        agent_run_starter:    Async callable kicking off an agent run
                              and returning ``(task_id, status_url)``.
        routing_policies:     Callable ``workspace -> RoutingPolicy``
                              used by the confidentiality gate. The
                              default reads ``routing.yaml`` from the
                              workspace root via :func:`load_routing_policy`.
        permission_registry:  Optional :class:`PermissionRegistry`. When
                              set, every tool dispatch passes through
                              it before the per-tool implementation runs.
                              Tools that are not registered fall through
                              to the registry's default-allow path for
                              the substrate's read-only operations.
        bearer_token:         The :class:`BearerToken` entry that
                              authenticated the current request, or
                              ``None`` for stdio mode. Per-tool dispatch
                              enforces this token's ``allowed_tools``
                              glob before running the implementation.
        retrieve_single:      Hook to run a single-workspace retrieval.
                              Defaults to
                              ``synthesis_engine.memory.three_tier_retrieve``;
                              tests pass a fake to avoid wiring a real
                              pgvector backend.
        retrieve_multi:       Hook to run a multi-workspace retrieval.
                              Defaults to
                              ``synthesis_engine.memory.three_tier_retrieve_multi``.
        status_url_template:  Format string used to render the
                              ``status_url`` returned from
                              ``agent_run_start``. The string is
                              ``str.format``-ed with ``task_id`` as the
                              only key. Defaults to the agent router's
                              REST shape.
    """

    memory: Any
    skill_runtime: Any
    skills_visible_for: Callable[[str], List[Any]]
    document_getter: DocumentGetter
    agent_run_starter: AgentRunStarter
    routing_policies: Callable[[str], RoutingPolicy]
    permission_registry: Optional[PermissionRegistry] = None
    bearer_token: Optional[BearerToken] = None
    retrieve_single: Optional[Callable[..., List[Any]]] = None
    retrieve_multi: Optional[Callable[..., List[Any]]] = None
    status_url_template: str = "/api/agent/sessions/{task_id}"


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


_RESULT_BLOCK_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "tier": {
            "type": "string",
            "description": (
                "Which retrieval tier produced this block. One of "
                "'vector', 'graph', 'session', 'user'."
            ),
        },
        "score": {"type": "number", "description": "Tier-relative ranking score."},
        "text": {"type": "string", "description": "Rendered text for prompting."},
        "metadata": {
            "type": "object",
            "additionalProperties": True,
            "description": "Tier-specific metadata (source_workspace, etc.).",
        },
        "provenance": {
            "type": ["object", "null"],
            "additionalProperties": True,
            "description": "Where the block came from (source, agent_run_id, confidence).",
        },
    },
    "required": ["tier", "score", "text"],
    "additionalProperties": False,
}


_CROSS_WORKSPACE_BLOCK_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "source_workspace": {
            "type": "string",
            "description": "The workspace the block was retrieved from.",
        },
        "estimated_tokens": {
            "type": "integer",
            "description": "Approximate token cost of the block's text.",
        },
        "workspace_rank": {
            "type": "integer",
            "description": "1-based rank within the source workspace.",
        },
        "result": _RESULT_BLOCK_SCHEMA,
    },
    "required": ["source_workspace", "result"],
    "additionalProperties": False,
}


TOOL_WORKSPACE_SEARCH = Tool(
    name="workspace_search",
    description=(
        "Three-tier retrieval over one workspace's memory. Returns "
        "ranked blocks from the vector, entity-graph, session, and "
        "user-memory tiers."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "workspace": {
                "type": "string",
                "minLength": 1,
                "description": "Workspace name (e.g., 'acme-news').",
            },
            "query": {
                "type": "string",
                "minLength": 1,
                "description": "Natural-language query.",
            },
            "k": {
                "type": "integer",
                "minimum": 1,
                "maximum": 200,
                "default": 10,
                "description": "Max vector-tier results to consider.",
            },
        },
        "required": ["workspace", "query"],
        "additionalProperties": False,
    },
    outputSchema={
        "type": "object",
        "properties": {
            "workspace": {"type": "string"},
            "blocks": {
                "type": "array",
                "items": _RESULT_BLOCK_SCHEMA,
            },
        },
        "required": ["workspace", "blocks"],
        "additionalProperties": False,
    },
)


TOOL_WORKSPACE_SEARCH_MULTI = Tool(
    name="workspace_search_multi",
    description=(
        "Multi-workspace three-tier retrieval. Applies the cross-"
        "workspace confidentiality gate (AIR_GAPPED never mixes; "
        "CLIENT_CONFIDENTIAL never mixes with PUBLIC; "
        "PERSONAL + CLIENT_CONFIDENTIAL is allowed but audited). "
        "Returns the merged blocks plus the effective confidentiality "
        "and a reference to the audit entry."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "workspaces": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
                "description": "Workspaces to search across.",
            },
            "query": {
                "type": "string",
                "minLength": 1,
                "description": "Natural-language query.",
            },
            "budget": {
                "type": "integer",
                "minimum": 100,
                "maximum": 100000,
                "default": 6000,
                "description": "Total token budget across workspaces.",
            },
        },
        "required": ["workspaces", "query"],
        "additionalProperties": False,
    },
    outputSchema={
        "type": "object",
        "properties": {
            "blocks": {
                "type": "array",
                "items": _CROSS_WORKSPACE_BLOCK_SCHEMA,
            },
            "effective_confidentiality": {
                "type": "string",
                "description": (
                    "Strictest confidentiality across the participating "
                    "workspaces (PUBLIC, PERSONAL, CLIENT_CONFIDENTIAL, "
                    "or AIR_GAPPED)."
                ),
            },
            "audit_entry_id": {
                "type": ["string", "null"],
                "description": (
                    "Audit-trail identifier when the op required an "
                    "audit record; null otherwise."
                ),
            },
            "requires_audit": {"type": "boolean"},
        },
        "required": ["blocks", "effective_confidentiality", "requires_audit"],
        "additionalProperties": False,
    },
)


TOOL_DOCUMENT_GET = Tool(
    name="document_get",
    description=(
        "Fetch a single document from a workspace by id. Returns the "
        "document's text content and metadata."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "workspace": {"type": "string", "minLength": 1},
            "document_id": {"type": "string", "minLength": 1},
        },
        "required": ["workspace", "document_id"],
        "additionalProperties": False,
    },
    outputSchema={
        "type": "object",
        "properties": {
            "workspace": {"type": "string"},
            "document_id": {"type": "string"},
            "content": {"type": "string"},
            "metadata": {
                "type": "object",
                "additionalProperties": True,
            },
        },
        "required": ["workspace", "document_id", "content"],
        "additionalProperties": False,
    },
)


TOOL_SKILL_RUN = Tool(
    name="skill_run",
    description=(
        "Invoke a skill-declared tool through the SkillRuntime. The "
        "skill must be visible from the requesting workspace; "
        "permission gates from the skill's frontmatter are enforced."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "workspace": {"type": "string", "minLength": 1},
            "skill_name": {"type": "string", "minLength": 1},
            "tool_name": {
                "type": "string",
                "minLength": 1,
                "description": (
                    "The skill's tool name. The runtime resolves "
                    "skill::<skill_name>::<tool_name> internally."
                ),
            },
            "input": {
                "type": "object",
                "description": (
                    "Tool input. Validated against the skill's declared "
                    "parameter schema by the runtime."
                ),
                "additionalProperties": True,
                "default": {},
            },
        },
        "required": ["workspace", "skill_name", "tool_name"],
        "additionalProperties": False,
    },
    outputSchema={
        "type": "object",
        "properties": {
            "skill": {"type": "string"},
            "tool": {"type": "string"},
            "result": {
                "description": "Whatever the skill-tool executor returned.",
            },
        },
        "required": ["skill", "tool"],
        "additionalProperties": True,
    },
)


TOOL_AGENT_RUN_START = Tool(
    name="agent_run_start",
    description=(
        "Start a Ragbot agent loop in the background. Returns "
        "immediately with a task_id and a status_url the client can "
        "poll for the FSM's terminal state."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "workspace_or_workspaces": {
                "oneOf": [
                    {"type": "string", "minLength": 1},
                    {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "minItems": 1,
                    },
                ],
                "description": (
                    "Single workspace name, or a list for a cross-"
                    "workspace synthesis."
                ),
            },
            "task": {
                "type": "string",
                "minLength": 1,
                "description": "Natural-language task description.",
            },
            "rubric": {
                "type": ["string", "null"],
                "description": (
                    "Optional grading rubric. When set, the loop drives "
                    "to DONE_GRADED rather than DONE."
                ),
            },
        },
        "required": ["workspace_or_workspaces", "task"],
        "additionalProperties": False,
    },
    outputSchema={
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
            "status_url": {"type": "string"},
            "status": {"type": "string", "enum": ["running"]},
        },
        "required": ["task_id", "status_url", "status"],
        "additionalProperties": False,
    },
)


TOOL_DEFINITIONS: Tuple[Tool, ...] = (
    TOOL_WORKSPACE_SEARCH,
    TOOL_WORKSPACE_SEARCH_MULTI,
    TOOL_DOCUMENT_GET,
    TOOL_SKILL_RUN,
    TOOL_AGENT_RUN_START,
)


# Lookup map: tool name → Tool definition. Used by the dispatcher.
_TOOL_BY_NAME: Dict[str, Tool] = {t.name: t for t in TOOL_DEFINITIONS}


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _enforce_bearer_token(ctx: ToolDispatchContext, tool_name: str) -> None:
    """Apply per-token ``allowed_tools`` gating. No-op for stdio (bearer=None)."""
    if ctx.bearer_token is None:
        return
    if not ctx.bearer_token.allows_tool(tool_name):
        raise ToolDispatchError(
            "forbidden",
            f"Bearer token {ctx.bearer_token.name!r} is not permitted to "
            f"call tool {tool_name!r}.",
        )


def _enforce_permission_registry(
    ctx: ToolDispatchContext,
    tool_name: str,
    arguments: Dict[str, Any],
    *,
    workspaces: Optional[List[str]] = None,
) -> None:
    """Run the permission registry's gate for ``tool_name``.

    When the registry has no gate for the tool, the default-deny rule
    for unknown writes / default-allow rule for unknown reads kicks in.
    The MCP tools surfaced here are all read-only or run-through-a-
    further-gate, so the call is safe to make even when the registry is
    not explicitly configured.

    Metadata fields the cross-workspace gate consults are populated when
    ``workspaces`` is supplied so ``workspace_search_multi`` rides the
    same gate the agent loop uses.
    """
    registry = ctx.permission_registry
    if registry is None:
        return

    metadata: Dict[str, Any] = {}
    if workspaces is not None:
        metadata[ACTIVE_WORKSPACES_METADATA_KEY] = list(workspaces)
        metadata[ROUTING_POLICIES_METADATA_KEY] = {
            ws: ctx.routing_policies(ws) for ws in workspaces
        }
    verdict = registry.check(
        tool_name,
        arguments=arguments,
        context=ToolCallContext(
            tool_name=tool_name,
            arguments=arguments,
            server_id="ragbot-mcp-server",
            metadata=metadata,
        ),
    )
    if not verdict.allowed:
        raise ToolDispatchError(
            "permission",
            f"Permission denied for {tool_name!r}: {verdict.reason}",
        )


def _result_to_dict(result: Any) -> Dict[str, Any]:
    """Render a :class:`MemoryResult` as the JSON-schema shape we surface."""
    # MemoryResult is a pydantic BaseModel; model_dump() is the right call.
    if hasattr(result, "model_dump"):
        raw = result.model_dump(mode="json")
    else:
        raw = dict(result)
    # The schema we publish is a tighter subset; drop any keys the schema
    # does not advertise so the SDK's outputSchema validation passes.
    return {
        "tier": raw.get("tier"),
        "score": float(raw.get("score", 0.0)),
        "text": raw.get("text", ""),
        "metadata": raw.get("metadata", {}) or {},
        "provenance": raw.get("provenance"),
    }


def _block_to_dict(block: Any) -> Dict[str, Any]:
    """Render a :class:`RetrievedBlock` as the cross-workspace schema shape."""
    return {
        "source_workspace": block.source_workspace,
        "estimated_tokens": int(getattr(block, "estimated_tokens", 0) or 0),
        "workspace_rank": int(getattr(block, "workspace_rank", 0) or 0),
        "result": _result_to_dict(block.result),
    }


# ---- per-tool implementations ---------------------------------------------


async def _impl_workspace_search(
    ctx: ToolDispatchContext, arguments: Dict[str, Any]
) -> Dict[str, Any]:
    workspace = arguments["workspace"]
    query = arguments["query"]
    k = int(arguments.get("k", 10))

    _enforce_permission_registry(
        ctx, "workspace_search", arguments, workspaces=[workspace]
    )

    # Resolve the retriever lazily so tests can swap it without touching
    # the substrate's process-wide ``get_memory`` singleton.
    if ctx.retrieve_single is None:
        from ..memory import three_tier_retrieve  # local import to avoid cycle

        retriever = three_tier_retrieve
    else:
        retriever = ctx.retrieve_single

    # The MemoryQuery type owns the field defaults; build a minimal one here.
    from ..memory import MemoryQuery

    mq = MemoryQuery(text=query, workspace=workspace, vector_k=k)
    try:
        results = retriever(ctx.memory, mq)
    except Exception as exc:  # pragma: no cover - defensive
        raise ToolDispatchError(
            "invalid_argument", f"workspace_search failed: {exc!r}"
        ) from exc
    return {
        "workspace": workspace,
        "blocks": [_result_to_dict(r) for r in results],
    }


async def _impl_workspace_search_multi(
    ctx: ToolDispatchContext, arguments: Dict[str, Any]
) -> Dict[str, Any]:
    workspaces_raw = arguments["workspaces"]
    workspaces = [str(w) for w in workspaces_raw if str(w).strip()]
    query = arguments["query"]
    budget = int(arguments.get("budget", 6000))

    if not workspaces:
        raise ToolDispatchError(
            "invalid_argument",
            "workspace_search_multi requires at least one workspace.",
        )

    # Cross-workspace policy gate first (before any retrieval), so the
    # caller doesn't accidentally see a single-workspace's data when a
    # mix would have been denied.
    policies = {ws: ctx.routing_policies(ws) for ws in workspaces}
    check = check_cross_workspace_op(workspaces, policies)
    if not check.allowed:
        raise ToolDispatchError("confidentiality", check.reason)

    _enforce_permission_registry(
        ctx, "workspace_search_multi", arguments, workspaces=workspaces
    )

    if ctx.retrieve_multi is None:
        from ..memory import three_tier_retrieve_multi  # local import

        retriever = three_tier_retrieve_multi
    else:
        retriever = ctx.retrieve_multi

    try:
        blocks = retriever(
            ctx.memory,
            workspaces,
            query,
            total_budget_tokens=budget,
        )
    except Exception as exc:  # pragma: no cover - defensive
        raise ToolDispatchError(
            "invalid_argument", f"workspace_search_multi failed: {exc!r}"
        ) from exc

    audit_entry_id: Optional[str] = None
    if check.requires_audit:
        # Record an audit entry; the audit-record helper does not return
        # an id, so we synthesize one from the timestamp + workspaces for
        # the response. The full entry is the source of truth.
        entry = AuditEntry.build(
            op_type="mcp.workspace_search_multi",
            workspaces=workspaces,
            tools=["workspace_search_multi"],
            outcome="allowed",
            metadata={
                "bearer_token_name": (
                    ctx.bearer_token.name if ctx.bearer_token else "stdio"
                ),
                "effective_confidentiality": (
                    check.effective_confidentiality.name
                ),
            },
        )
        try:
            audit_record(entry)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to record audit entry: %s", exc)
        audit_entry_id = f"{entry.timestamp_iso}::{','.join(workspaces)}"

    return {
        "blocks": [_block_to_dict(b) for b in blocks],
        "effective_confidentiality": check.effective_confidentiality.name,
        "audit_entry_id": audit_entry_id,
        "requires_audit": check.requires_audit,
    }


async def _impl_document_get(
    ctx: ToolDispatchContext, arguments: Dict[str, Any]
) -> Dict[str, Any]:
    workspace = arguments["workspace"]
    document_id = arguments["document_id"]

    _enforce_permission_registry(
        ctx, "document_get", arguments, workspaces=[workspace]
    )
    try:
        doc = ctx.document_getter(workspace, document_id)
    except KeyError as exc:
        raise ToolDispatchError(
            "not_found",
            f"Document {document_id!r} not found in workspace {workspace!r}.",
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise ToolDispatchError(
            "invalid_argument", f"document_get failed: {exc!r}"
        ) from exc
    # Normalise the shape so the outputSchema validation is deterministic.
    return {
        "workspace": workspace,
        "document_id": document_id,
        "content": str(doc.get("content", "")),
        "metadata": doc.get("metadata", {}) or {},
    }


async def _impl_skill_run(
    ctx: ToolDispatchContext, arguments: Dict[str, Any]
) -> Dict[str, Any]:
    workspace = arguments["workspace"]
    skill_name = arguments["skill_name"]
    tool_name = arguments["tool_name"]
    tool_input = arguments.get("input", {}) or {}

    _enforce_permission_registry(
        ctx, "skill_run", arguments, workspaces=[workspace]
    )

    # Workspace-visibility check: the skill must be in the workspace's
    # inheritance chain. Stops a stdio client from running a personal
    # skill against a client-confidential workspace by accident.
    visible = {getattr(s, "name", None) for s in ctx.skills_visible_for(workspace)}
    if skill_name not in visible:
        raise ToolDispatchError(
            "forbidden",
            f"Skill {skill_name!r} is not visible from workspace "
            f"{workspace!r}.",
        )

    runtime = ctx.skill_runtime
    if runtime is None:
        raise ToolDispatchError(
            "invalid_argument",
            "SkillRuntime is not wired; skill_run is unavailable.",
        )

    # Register the skill's tools (idempotent) and dispatch through the
    # runtime's _dispatch_skill_tool path. The dispatcher's contract
    # accepts a tiny synthetic state/step pair — we surface only the
    # subset of attributes the skill runtime touches.
    try:
        runtime.register_skill_tools(skill_name)
    except Exception as exc:
        raise ToolDispatchError(
            "not_found",
            f"Skill {skill_name!r} could not be activated: {exc!r}",
        ) from exc

    @dataclass
    class _Step:
        target: str
        step_id: str = "mcp-skill_run"

    @dataclass
    class _State:
        task_id: Optional[str] = None

    step = _Step(target=f"skill::{skill_name}::{tool_name}")
    state = _State(task_id=None)

    try:
        output = await runtime._dispatch_skill_tool(
            loop=None,
            state=state,
            step=step,
            inputs=tool_input,
            skill_name=skill_name,
            tool_name=tool_name,
        )
    except PermissionError as exc:
        raise ToolDispatchError(
            "permission",
            f"Skill {skill_name}.{tool_name} denied: {exc}",
        ) from exc
    except RuntimeError as exc:
        raise ToolDispatchError(
            "not_found",
            f"Skill {skill_name}.{tool_name} unavailable: {exc}",
        ) from exc

    return {"skill": skill_name, "tool": tool_name, "result": output}


async def _impl_agent_run_start(
    ctx: ToolDispatchContext, arguments: Dict[str, Any]
) -> Dict[str, Any]:
    raw_workspaces = arguments["workspace_or_workspaces"]
    if isinstance(raw_workspaces, str):
        workspaces: Tuple[str, ...] = (raw_workspaces,)
    elif isinstance(raw_workspaces, (list, tuple)):
        workspaces = tuple(str(w) for w in raw_workspaces if str(w).strip())
    else:
        raise ToolDispatchError(
            "invalid_argument",
            "workspace_or_workspaces must be a string or a non-empty list.",
        )
    if not workspaces:
        raise ToolDispatchError(
            "invalid_argument", "workspace_or_workspaces must be non-empty."
        )

    task = arguments["task"]
    rubric = arguments.get("rubric")

    # Cross-workspace gate for multi-workspace agent runs.
    if len(workspaces) > 1:
        policies = {ws: ctx.routing_policies(ws) for ws in workspaces}
        check = check_cross_workspace_op(list(workspaces), policies)
        if not check.allowed:
            raise ToolDispatchError("confidentiality", check.reason)

    _enforce_permission_registry(
        ctx,
        "agent_run_start",
        arguments,
        workspaces=list(workspaces),
    )
    starter = ctx.agent_run_starter
    if starter is None:
        raise ToolDispatchError(
            "invalid_argument",
            "agent_run_starter is not wired; agent_run_start is unavailable.",
        )

    try:
        task_id, status_url = await starter(workspaces, task, rubric)
    except Exception as exc:  # pragma: no cover - defensive
        raise ToolDispatchError(
            "invalid_argument", f"agent_run_start failed: {exc!r}"
        ) from exc
    return {"task_id": task_id, "status_url": status_url, "status": "running"}


_IMPLEMENTATIONS: Dict[
    str, Callable[[ToolDispatchContext, Dict[str, Any]], Awaitable[Dict[str, Any]]]
] = {
    TOOL_WORKSPACE_SEARCH.name: _impl_workspace_search,
    TOOL_WORKSPACE_SEARCH_MULTI.name: _impl_workspace_search_multi,
    TOOL_DOCUMENT_GET.name: _impl_document_get,
    TOOL_SKILL_RUN.name: _impl_skill_run,
    TOOL_AGENT_RUN_START.name: _impl_agent_run_start,
}


# ---------------------------------------------------------------------------
# Public dispatch entry point
# ---------------------------------------------------------------------------


async def dispatch_tool(
    name: str,
    arguments: Dict[str, Any],
    ctx: ToolDispatchContext,
) -> Dict[str, Any]:
    """Dispatch one MCP tool call.

    Args:
        name: The tool name as advertised in :data:`TOOL_DEFINITIONS`.
        arguments: The validated argument dict from the MCP client.
        ctx: The current request's :class:`ToolDispatchContext`.

    Returns:
        The structured-output dict matching the tool's ``outputSchema``.

    Raises:
        ToolDispatchError: with a stable ``code`` for unknown tools,
            forbidden bearer tokens, denied permissions, missing
            resources, and confidentiality-gate denials.
    """
    if name not in _TOOL_BY_NAME:
        raise ToolDispatchError("unknown_tool", f"Unknown tool: {name!r}")
    _enforce_bearer_token(ctx, name)
    impl = _IMPLEMENTATIONS[name]
    return await impl(ctx, arguments or {})


__all__ = [
    "TOOL_AGENT_RUN_START",
    "TOOL_DEFINITIONS",
    "TOOL_DOCUMENT_GET",
    "TOOL_SKILL_RUN",
    "TOOL_WORKSPACE_SEARCH",
    "TOOL_WORKSPACE_SEARCH_MULTI",
    "ToolDispatchContext",
    "ToolDispatchError",
    "dispatch_tool",
]
