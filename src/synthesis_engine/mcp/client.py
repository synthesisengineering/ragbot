"""Top-level MCP client class managing multiple server connections.

:class:`MCPClient` is the substrate's public face. A runtime instantiates
it once per process, hands it the loaded :class:`MCPConfig`, and consumes
its async surface to:

* Connect/disconnect specific servers, or every server enabled for a
  given workspace.
* List tools/resources/prompts on a connected server.
* Invoke tools (sync or via Tasks for long-running calls).
* Read resources, subscribe to updates, get prompts.

Internally, :class:`MCPClient` owns the :class:`MCPRegistry` and threads
sampling/elicitation/roots callbacks through to every new session. It is
intentionally a thin orchestration class — the protocol-level work lives
in the per-primitive modules — because the substrate's job is to keep
the runtime decoupled from SDK churn, not to re-implement the protocol.

Lifecycle::

    client = MCPClient(config, sampling_callback=..., elicitation_callback=...)
    await client.start()                          # opens auto-enabled servers
    tools = await client.list_tools("fs-local")
    result = await client.call_tool("fs-local", "read_file", {"path": "..."})
    await client.shutdown()

Or via async context manager::

    async with MCPClient(config) as client:
        ...

The class is safe to use across multiple workspaces in a single process
— per-workspace gating is consulted on each ``get_active_servers`` call.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from mcp.types import (
    CallToolResult,
    CancelTaskResult,
    CreateTaskResult,
    GetPromptResult,
    GetTaskResult,
    ListTasksResult,
    Prompt,
    ReadResourceResult,
    Resource,
    ResourceTemplate,
    Tool,
)

from . import tasks as mcp_tasks
from .config import MCPConfig, load_mcp_config
from .primitives import (
    ElicitationCallback,
    RootsProvider,
    SamplingCallback,
    call_tool,
    default_elicitation_handler,
    default_sampling_handler,
    get_prompt,
    list_prompts,
    list_resource_templates,
    list_resources,
    list_tools,
    read_resource,
    subscribe_resource,
    unsubscribe_resource,
)
from .registry import MCPRegistry, ServerEntry


logger = logging.getLogger("synthesis_engine.mcp.client")


class MCPClient:
    """High-level MCP client wrapping the registry and primitive helpers."""

    def __init__(
        self,
        config: Optional[MCPConfig] = None,
        *,
        sampling_callback: SamplingCallback = default_sampling_handler,
        elicitation_callback: ElicitationCallback = default_elicitation_handler,
        roots_provider: Optional[RootsProvider] = None,
        autoconnect: bool = True,
    ):
        self._registry = MCPRegistry(
            config or load_mcp_config(),
            sampling_callback=sampling_callback,
            elicitation_callback=elicitation_callback,
            roots_provider=roots_provider,
        )
        self._autoconnect = autoconnect

    # ------ lifecycle -------------------------------------------------------

    async def start(self, *, workspace: Optional[str] = None) -> None:
        """Open every server enabled for ``workspace`` (if ``autoconnect``)."""
        if self._autoconnect:
            await self._registry.connect_all_enabled(workspace=workspace)

    async def shutdown(self) -> None:
        """Disconnect every live session."""
        await self._registry.disconnect_all()

    async def __aenter__(self) -> "MCPClient":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.shutdown()

    # ------ config / registry introspection ---------------------------------

    @property
    def registry(self) -> MCPRegistry:
        return self._registry

    @property
    def config(self) -> MCPConfig:
        return self._registry.config

    def replace_config(self, config: MCPConfig) -> None:
        """Swap in a new config (existing live sessions preserved)."""
        self._registry.replace_config(config)

    def get_active_servers(
        self, workspace: Optional[str] = None
    ) -> List[ServerEntry]:
        """Return entries enabled for ``workspace``."""
        return self._registry.get_active_servers(workspace)

    def list_servers(self) -> List[ServerEntry]:
        """Return every configured entry, regardless of workspace gating."""
        return self._registry.all_entries()

    # ------ connection control ---------------------------------------------

    async def connect(self, server_id: str) -> ServerEntry:
        return await self._registry.connect(server_id)

    async def disconnect(self, server_id: str) -> None:
        await self._registry.disconnect(server_id)

    async def toggle(self, server_id: str) -> ServerEntry:
        """If connected, disconnect; otherwise, connect. Returns the entry."""
        entry = self._registry.get_entry(server_id)
        if entry.status == "connected":
            await self._registry.disconnect(server_id)
        else:
            await self._registry.connect(server_id)
        return self._registry.get_entry(server_id)

    # ------ primitives — tools ---------------------------------------------

    async def list_tools(self, server_id: str) -> List[Tool]:
        return await list_tools(self._registry.require_session(server_id))

    async def call_tool(
        self,
        server_id: str,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        *,
        timeout_seconds: Optional[float] = None,
    ) -> CallToolResult:
        return await call_tool(
            self._registry.require_session(server_id),
            name=name,
            arguments=arguments,
            timeout_seconds=timeout_seconds,
        )

    # ------ primitives — resources ----------------------------------------

    async def list_resources(self, server_id: str) -> List[Resource]:
        return await list_resources(self._registry.require_session(server_id))

    async def list_resource_templates(self, server_id: str) -> List[ResourceTemplate]:
        return await list_resource_templates(
            self._registry.require_session(server_id)
        )

    async def read_resource(self, server_id: str, uri: str) -> ReadResourceResult:
        return await read_resource(self._registry.require_session(server_id), uri)

    async def subscribe_resource(self, server_id: str, uri: str) -> None:
        await subscribe_resource(self._registry.require_session(server_id), uri)

    async def unsubscribe_resource(self, server_id: str, uri: str) -> None:
        await unsubscribe_resource(self._registry.require_session(server_id), uri)

    # ------ primitives — prompts ------------------------------------------

    async def list_prompts(self, server_id: str) -> List[Prompt]:
        return await list_prompts(self._registry.require_session(server_id))

    async def get_prompt(
        self,
        server_id: str,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> GetPromptResult:
        return await get_prompt(
            self._registry.require_session(server_id),
            name=name,
            arguments=arguments,
        )

    # ------ tasks -----------------------------------------------------------

    async def call_tool_as_task(
        self,
        server_id: str,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        *,
        ttl_ms: int = 60_000,
        meta: Optional[Dict[str, Any]] = None,
    ) -> CreateTaskResult:
        return await mcp_tasks.call_tool_as_task(
            self._registry.require_session(server_id),
            name=name,
            arguments=arguments,
            ttl_ms=ttl_ms,
            meta=meta,
        )

    async def task_status(self, server_id: str, task_id: str) -> GetTaskResult:
        return await mcp_tasks.get_status(
            self._registry.require_session(server_id), task_id
        )

    async def task_result(
        self,
        server_id: str,
        task_id: str,
        result_type: type = CallToolResult,
    ) -> Any:
        return await mcp_tasks.get_result(
            self._registry.require_session(server_id), task_id, result_type
        )

    async def cancel_task(self, server_id: str, task_id: str) -> CancelTaskResult:
        return await mcp_tasks.cancel(
            self._registry.require_session(server_id), task_id
        )

    async def list_tasks(
        self, server_id: str, *, cursor: Optional[str] = None
    ) -> ListTasksResult:
        return await mcp_tasks.list_tasks(
            self._registry.require_session(server_id), cursor=cursor
        )

    async def wait_for_task(
        self,
        server_id: str,
        task_id: str,
        *,
        on_status=None,
    ) -> GetTaskResult:
        return await mcp_tasks.poll_until_done(
            self._registry.require_session(server_id),
            task_id,
            on_status=on_status,
        )


# ---------------------------------------------------------------------------
# Process-singleton helper (substrate convention)
# ---------------------------------------------------------------------------

_default_client: Optional[MCPClient] = None


def get_default_client() -> Optional[MCPClient]:
    """Return the process-wide default :class:`MCPClient`, if installed."""
    return _default_client


def set_default_client(client: Optional[MCPClient]) -> None:
    """Install (or clear) the process-wide default :class:`MCPClient`.

    Runtimes that want a single shared registry across many call sites
    (chat handlers, agent loops, settings panels) call this once during
    startup and ``get_default_client()`` everywhere else.
    """
    global _default_client
    _default_client = client


__all__ = [
    "MCPClient",
    "get_default_client",
    "set_default_client",
]
