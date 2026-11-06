"""In-process registry of configured MCP servers and live connections.

The registry is the single source of truth for the substrate's view of
*which servers exist*, *whether each is connected*, *what the last
recorded error was*, and *what its discovered capabilities/catalog
look like*. The runtime queries the registry to render the settings
panel and to iterate active connections during an agent turn.

State held per entry:

* ``config`` — the validated :class:`MCPServerConfig` from ``mcp.yaml``.
* ``status`` — ``disconnected | connecting | connected | error``.
* ``session`` — the active :class:`mcp.ClientSession` while connected.
* ``capabilities`` — the server's advertised capabilities from
  ``initialize``.
* ``last_error`` — the last exception (as a string) on failure.
* ``cached_tools`` / ``cached_resources`` / ``cached_prompts`` — the
  most recent catalog responses, so the UI does not have to round-
  trip on every render.

Concurrency: all mutating operations on a single :class:`ServerEntry`
go through an asyncio lock. The registry-level dict is guarded by a
separate lock for cross-server operations like "list everything."
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from mcp import ClientSession
from mcp.types import (
    Implementation,
    Prompt,
    Resource,
    ServerCapabilities,
    Tool,
)

from synthesis_engine.exceptions import SynthesisError

from .auth import build_oauth_provider
from .config import MCPConfig, MCPServerConfig
from .primitives.elicitation import ElicitationCallback, default_elicitation_handler
from .primitives.prompts import list_prompts
from .primitives.resources import list_resources
from .primitives.roots import RootsProvider
from .primitives.sampling import SamplingCallback, default_sampling_handler
from .primitives.tools import list_tools
from .transport import open_transport


logger = logging.getLogger("synthesis_engine.mcp.registry")

CLIENT_INFO = Implementation(
    name="synthesis-engine",
    version="0.1.0",
    title="synthesis-engine MCP client",
)


class MCPRegistryError(SynthesisError):
    """Registry-layer failure (unknown server id, connection failed, etc.)."""


ServerStatus = str  # "disconnected" | "connecting" | "connected" | "error"


@dataclass
class CachedCatalog:
    """Most recent catalog responses cached by the registry."""

    tools: List[Tool] = field(default_factory=list)
    resources: List[Resource] = field(default_factory=list)
    prompts: List[Prompt] = field(default_factory=list)


@dataclass
class ServerEntry:
    """One server's state inside the registry."""

    config: MCPServerConfig
    status: ServerStatus = "disconnected"
    session: Optional[ClientSession] = None
    capabilities: Optional[ServerCapabilities] = None
    last_error: Optional[str] = None
    catalog: CachedCatalog = field(default_factory=CachedCatalog)
    # Internal: tasks holding the transport + session open. We cannot
    # ``await`` an ``async with`` and store the entered streams without
    # a background driver coroutine, so each connection runs as a task
    # that we cancel on disconnect.
    _driver_task: Optional[asyncio.Task[None]] = None
    _ready: Optional[asyncio.Event] = None
    _shutdown: Optional[asyncio.Event] = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class MCPRegistry:
    """In-process catalog of configured servers and their live connections.

    The registry is async-friendly but does not own its own event loop;
    it expects to be driven from a running loop (FastAPI's, a CLI's,
    a test's). All public methods are async to keep the signature stable.

    Typical lifecycle, from a runtime's perspective::

        registry = MCPRegistry(config)
        await registry.connect_all_enabled(workspace="personal")
        ...
        servers = registry.get_active_servers(workspace="personal")
        for entry in servers:
            tools = await registry.tools(entry.config.id)
            ...
        await registry.disconnect_all()

    Per-workspace filtering is applied by :meth:`get_active_servers`. The
    underlying connections are workspace-agnostic — workspace gating is
    a *policy* decision, not a *connection* decision, because a user may
    have multiple workspaces open and switch between them.
    """

    def __init__(
        self,
        config: Optional[MCPConfig] = None,
        *,
        sampling_callback: SamplingCallback = default_sampling_handler,
        elicitation_callback: ElicitationCallback = default_elicitation_handler,
        roots_provider: Optional[RootsProvider] = None,
    ):
        self._config = config or MCPConfig()
        self._entries: Dict[str, ServerEntry] = {}
        self._lock = asyncio.Lock()
        self._sampling_callback = sampling_callback
        self._elicitation_callback = elicitation_callback
        self._roots_provider = roots_provider
        for s in self._config.servers:
            self._entries[s.id] = ServerEntry(config=s)

    # ------ config sync -----------------------------------------------------

    def replace_config(self, config: MCPConfig) -> None:
        """Swap in a new config, preserving live connections for unchanged servers."""
        new_ids = {s.id for s in config.servers}
        old_ids = set(self._entries.keys())

        # Drop entries for removed servers (the caller is responsible for
        # disconnecting them first; we just clear the bookkeeping).
        for removed in old_ids - new_ids:
            self._entries.pop(removed, None)

        # Add new entries, update existing ones.
        for s in config.servers:
            if s.id in self._entries:
                # Keep status/session if the connection is alive, replace config.
                self._entries[s.id].config = s
            else:
                self._entries[s.id] = ServerEntry(config=s)

        self._config = config

    @property
    def config(self) -> MCPConfig:
        return self._config

    def get_entry(self, server_id: str) -> ServerEntry:
        entry = self._entries.get(server_id)
        if entry is None:
            raise MCPRegistryError(f"no server configured with id {server_id!r}")
        return entry

    def has(self, server_id: str) -> bool:
        return server_id in self._entries

    def all_entries(self) -> List[ServerEntry]:
        return list(self._entries.values())

    # ------ workspace gating ------------------------------------------------

    def get_active_servers(self, workspace: Optional[str]) -> List[ServerEntry]:
        """Return the entries enabled for ``workspace``.

        Discovery-filter pattern hook: if a runtime has registered a
        discovery filter for the ``mcp_servers`` scope, its return value
        wins outright. Otherwise, the policy lives entirely in the per-
        server :meth:`MCPServerConfig.is_enabled_for_workspace`.

        The filter return contract: either ``None`` (no override; use
        the default policy) or a list of server ids to include. Server
        ids not present in the registry are silently dropped.
        """
        from synthesis_engine.discovery import apply_discovery_filter

        default = [
            e for e in self._entries.values()
            if e.config.is_enabled_for_workspace(
                workspace, self._config.defaults.enabled_by_default
            )
        ]
        result = apply_discovery_filter("mcp_servers", default)
        # If the filter returned a list of ids, translate to entries.
        if result is not default and isinstance(result, list) and all(
            isinstance(x, str) for x in result
        ):
            return [self._entries[i] for i in result if i in self._entries]
        return result

    # ------ lifecycle -------------------------------------------------------

    async def connect(self, server_id: str) -> ServerEntry:
        """Open a session against ``server_id``.

        Idempotent: if the entry is already connected, returns it untouched.
        On failure, ``status`` becomes ``error`` and ``last_error`` carries
        the message; the exception is re-raised.
        """
        entry = self.get_entry(server_id)
        async with entry._lock:
            if entry.status == "connected":
                return entry
            entry.status = "connecting"
            entry.last_error = None
            entry._ready = asyncio.Event()
            entry._shutdown = asyncio.Event()
            try:
                # The driver task owns the transport and session contexts
                # for their entire lifetime. We wait on _ready for the
                # session to be initialized before returning.
                entry._driver_task = asyncio.create_task(
                    self._drive(entry), name=f"mcp-driver:{server_id}"
                )
                # Wait for either ready or task completion (error).
                await asyncio.wait(
                    [
                        asyncio.create_task(entry._ready.wait()),
                        entry._driver_task,
                    ],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if entry._driver_task.done() and entry.status != "connected":
                    # Driver exited before signalling ready; surface the cause.
                    exc = entry._driver_task.exception()
                    if exc is not None:
                        raise exc
                    raise MCPRegistryError(
                        f"connection for {server_id!r} exited before initializing"
                    )
                return entry
            except Exception as exc:
                entry.status = "error"
                entry.last_error = f"{type(exc).__name__}: {exc}"
                logger.exception("connect failed for %s", server_id)
                raise

    async def _drive(self, entry: ServerEntry) -> None:
        """Background coroutine: own the transport + session until shutdown."""
        cfg = entry.config
        oauth = build_oauth_provider(cfg)
        try:
            async with open_transport(cfg, oauth_provider=oauth) as (read, write):
                async with ClientSession(
                    read,
                    write,
                    sampling_callback=self._sampling_callback,
                    elicitation_callback=self._elicitation_callback,
                    list_roots_callback=(
                        self._roots_provider.list_roots_callback
                        if self._roots_provider is not None else None
                    ),
                    client_info=CLIENT_INFO,
                ) as session:
                    init = await session.initialize()
                    entry.session = session
                    entry.capabilities = init.capabilities
                    entry.status = "connected"
                    entry.last_error = None
                    assert entry._ready is not None
                    entry._ready.set()
                    # Pre-warm the catalog for the UI. Failures here are
                    # non-fatal — a server may advertise capabilities it
                    # then chooses to refuse, and we don't want catalog
                    # quirks to bring down the connection.
                    try:
                        await self._refresh_catalog(entry, session)
                    except Exception as cache_exc:
                        logger.warning(
                            "catalog prefetch failed for %s: %s",
                            cfg.id, cache_exc,
                        )
                    # Wait until the orchestrator asks us to shut down.
                    assert entry._shutdown is not None
                    await entry._shutdown.wait()
        except Exception as exc:
            entry.status = "error"
            entry.last_error = f"{type(exc).__name__}: {exc}"
            if entry._ready is not None and not entry._ready.is_set():
                entry._ready.set()  # unblock connect()
            raise
        finally:
            entry.session = None
            if entry.status != "error":
                entry.status = "disconnected"

    async def _refresh_catalog(self, entry: ServerEntry, session: ClientSession) -> None:
        """Refresh the cached catalog using the server's advertised capabilities."""
        caps = entry.capabilities
        catalog = CachedCatalog()
        if caps is not None and caps.tools is not None:
            with contextlib.suppress(Exception):
                catalog.tools = await list_tools(session)
        if caps is not None and caps.resources is not None:
            with contextlib.suppress(Exception):
                catalog.resources = await list_resources(session)
        if caps is not None and caps.prompts is not None:
            with contextlib.suppress(Exception):
                catalog.prompts = await list_prompts(session)
        entry.catalog = catalog

    async def disconnect(self, server_id: str) -> None:
        """Close the session for ``server_id`` and tear down the transport."""
        entry = self.get_entry(server_id)
        async with entry._lock:
            if entry.status not in ("connected", "connecting"):
                return
            if entry._shutdown is not None:
                entry._shutdown.set()
            if entry._driver_task is not None:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(entry._driver_task, timeout=5.0)
                entry._driver_task = None
            entry._shutdown = None
            entry._ready = None
            entry.session = None
            if entry.status != "error":
                entry.status = "disconnected"

    async def connect_all_enabled(self, workspace: Optional[str] = None) -> None:
        """Open sessions for every server enabled for ``workspace``."""
        targets = self.get_active_servers(workspace)
        results = await asyncio.gather(
            *(self.connect(e.config.id) for e in targets),
            return_exceptions=True,
        )
        for entry, res in zip(targets, results):
            if isinstance(res, Exception):
                logger.warning(
                    "auto-connect failed for %s: %s",
                    entry.config.id, res,
                )

    async def disconnect_all(self) -> None:
        """Close every active session."""
        await asyncio.gather(
            *(self.disconnect(e.config.id) for e in self._entries.values()),
            return_exceptions=True,
        )

    # ------ helpers ---------------------------------------------------------

    def require_session(self, server_id: str) -> ClientSession:
        """Return the live session for ``server_id`` or raise.

        Higher-level call paths use this; callers that want to handle
        the "not connected yet" case themselves should consult
        :meth:`get_entry` instead.
        """
        entry = self.get_entry(server_id)
        if entry.status != "connected" or entry.session is None:
            raise MCPRegistryError(
                f"server {server_id!r} is not connected (status={entry.status})"
            )
        return entry.session


__all__ = [
    "CachedCatalog",
    "MCPRegistry",
    "MCPRegistryError",
    "ServerEntry",
    "ServerStatus",
]
