"""Production wiring for :class:`RagbotMCPServer`.

The server itself (``.server``) is a thin adapter that takes a fully-built
:class:`ServerDependencies` and exposes it over stdio or HTTP. Tests build
their own fake dependencies (see ``tests/test_mcp_server.py``); the FastAPI
app's lifespan handler (``api/main.py``) wires an :class:`AgentLoop` and an
LLM backend for the HTTP/chat path. Neither of those wires up a
`ServerDependencies` instance today — this module is that missing piece,
used by the ``ragbot mcp serve`` CLI entry point (see ``src/ragbot.py``).

Every collaborator resolved here is the same process-wide singleton (or
the same construction pattern) already used elsewhere in the codebase:

    * memory              -> synthesis_engine.memory.get_memory()
    * skill_runtime        -> SkillRuntime over ALL discovered skills
                              (per-call workspace visibility is enforced
                              separately by skills_visible_for, matching
                              tools.py's _impl_skill_run contract)
    * skills_visible_for   -> synthesis_engine.skills.get_skills_for_workspace
    * list_workspaces      -> synthesis_engine.workspaces.resolve_repo_index
    * routing_policies     -> synthesis_engine.policy.routing.load_routing_policy
    * document_getter      -> a thin scroll_documents()-based lookup
    * agent_run_starter    -> make_agent_run_starter_from_loop() over a
                              fresh AgentLoop (same construction as
                              api/main.py's lifespan step 6, minus the MCP
                              client — stdio mode has no HTTP app to source
                              one from)

retrieve_single / retrieve_multi are deliberately left ``None`` so
tools.py's lazy default (``three_tier_retrieve`` / ``three_tier_retrieve_multi``
against the real ``memory``) is used — that is the real behavior, not a
test seam.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _list_workspaces() -> List[str]:
    from ..workspaces import resolve_repo_index

    return sorted(resolve_repo_index().keys())


def _routing_policy_for(workspace: str):
    from ..policy.routing import RoutingPolicy, load_routing_policy
    from ..workspaces import resolve_repo_index

    root = resolve_repo_index().get(workspace)
    if not root:
        return RoutingPolicy()
    try:
        return load_routing_policy(root)
    except Exception as exc:  # noqa: BLE001 — fall back to the safe default
        logger.warning(
            "Failed to load routing.yaml for workspace %s: %s", workspace, exc,
        )
        return RoutingPolicy()


def _document_getter(workspace: str, document_id: str) -> Dict[str, Any]:
    """Fetch a document's full content by source path / filename.

    Mirrors rag.py's find_full_document reconstruction (chunks ordered by
    char_start, overlap trimmed) but keyed by an exact document_id instead
    of a fuzzy hint — the MCP document_get contract is "fetch by id," not
    "search for the best match." Raises KeyError when nothing matches, per
    tools.py's _impl_document_get contract.
    """
    from ..vectorstore import get_vector_store

    vs = get_vector_store()
    if vs is None:
        raise KeyError(document_id)

    hits = vs.scroll_documents(workspace, limit=5000)
    matches = [
        h for h in hits
        if h.metadata.get("source_file") == document_id
        or h.metadata.get("source_path") == document_id
        or h.metadata.get("filename") == document_id
    ]
    if not matches:
        raise KeyError(document_id)

    ordered = sorted(matches, key=lambda h: h.metadata.get("char_start") or 0)
    content_parts: List[str] = []
    last_end = 0
    for hit in ordered:
        start = hit.metadata.get("char_start") or 0
        end = hit.metadata.get("char_end") or (start + len(hit.text))
        if start >= last_end:
            content_parts.append(hit.text)
        else:
            overlap = last_end - start
            if overlap < len(hit.text):
                content_parts.append(hit.text[overlap:])
        last_end = end

    return {
        "content": "".join(content_parts),
        "metadata": dict(ordered[0].metadata),
    }


def build_default_dependencies():
    """Wire a :class:`ServerDependencies` instance from live substrate
    singletons, for use by ``ragbot mcp serve``.

    Raises RuntimeError if the memory backend (pgvector) is unreachable —
    fail loud at startup rather than serving a half-wired MCP server.
    """
    from ..agent import AgentLoop, FilesystemCheckpointStore
    from ..agent.permissions import PermissionRegistry
    from ..llm import get_llm_backend
    from ..memory import get_memory
    from ..skills import discover_skills, get_skills_for_workspace
    from ..skills.loader import SkillLoader
    from ..skills.runtime import SkillRuntime
    from .server import ServerDependencies, make_agent_run_starter_from_loop

    memory = get_memory()
    if memory is None:
        raise RuntimeError(
            "Memory backend unavailable (RAGBOT_DATABASE_URL not set or "
            "the database is unreachable). `ragbot mcp serve` requires a "
            "working pgvector backend — run `ragbot db status` to diagnose."
        )

    permission_registry = PermissionRegistry()
    # Global loader over every discovered skill. Per-call workspace
    # visibility is enforced separately in tools.py's _impl_skill_run via
    # skills_visible_for BEFORE the runtime is asked to activate anything,
    # so this loader only needs to be able to activate skills that are
    # visible from *some* workspace, not pre-filtered to one.
    loader = SkillLoader(discover_skills())
    skill_runtime = SkillRuntime(loader, permission_registry)

    llm_backend = get_llm_backend()
    checkpoint_store = FilesystemCheckpointStore()
    agent_loop = AgentLoop(
        llm_backend=llm_backend,
        mcp_client=None,
        checkpoint_store=checkpoint_store,
        default_mcp_server="local",
    )
    agent_run_starter = make_agent_run_starter_from_loop(agent_loop)

    return ServerDependencies(
        memory=memory,
        skill_runtime=skill_runtime,
        skills_visible_for=get_skills_for_workspace,
        list_workspaces=_list_workspaces,
        routing_policies=_routing_policy_for,
        document_getter=_document_getter,
        agent_run_starter=agent_run_starter,
        permission_registry=permission_registry,
    )


__all__ = ["build_default_dependencies"]
