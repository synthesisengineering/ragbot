"""Roots primitive.

Roots are client-offered URI or filesystem boundaries inside which the
server is permitted to operate. In MCP terminology this is a *client
feature offered to servers*: the server inquires, the client answers.

JSON-RPC methods covered:

* ``roots/list`` — the server-initiated inquiry. The SDK's
  ``ClientSession.list_roots_callback`` is the hook; we wrap it via the
  :class:`RootsProvider` protocol so callers don't have to know the SDK's
  internal signature.
* ``notifications/roots/list_changed`` — the client-to-server notification
  emitted whenever the root set mutates. The wire call is exposed via
  :meth:`RootsProvider.notify_changed`.

The substrate ships :class:`StaticRootsProvider` for the simple case
(known roots at startup). Runtimes with dynamic root sets (Ragbot
switching workspaces, Ragenie scoping a routine to an ai-knowledge path)
implement :class:`RootsProvider` themselves.
"""

from __future__ import annotations

import abc
from typing import Optional, Sequence

from mcp import ClientSession
from mcp.client.context import ClientRequestContext
from mcp.types import ListRootsResult, Root


class RootsProvider(abc.ABC):
    """Abstract source of root URIs the client is willing to expose."""

    @abc.abstractmethod
    async def list_roots(self) -> list[Root]:
        """Return the current root list."""

    async def list_roots_callback(
        self, context: ClientRequestContext
    ) -> ListRootsResult:
        """SDK-compatible callback: wraps :meth:`list_roots` into a result."""
        roots = await self.list_roots()
        return ListRootsResult(roots=list(roots))

    async def notify_changed(self, session: ClientSession) -> None:
        """Emit ``notifications/roots/list_changed`` on the session.

        Call this after the root set has mutated. The default
        implementation just delegates to the SDK; subclasses are free to
        override (for example, to coalesce rapid-fire updates).
        """
        await session.send_roots_list_changed()


class StaticRootsProvider(RootsProvider):
    """Yields a fixed list of roots provided at construction time.

    Update the set via :meth:`replace`, which atomically swaps the list
    and emits the ``list_changed`` notification on every session that
    has been registered via :meth:`bind`.
    """

    def __init__(self, roots: Optional[Sequence[Root]] = None):
        self._roots: list[Root] = list(roots or [])
        self._bound: list[ClientSession] = []

    async def list_roots(self) -> list[Root]:
        return list(self._roots)

    def bind(self, session: ClientSession) -> None:
        """Register a session for ``list_changed`` notifications."""
        if session not in self._bound:
            self._bound.append(session)

    def unbind(self, session: ClientSession) -> None:
        if session in self._bound:
            self._bound.remove(session)

    async def replace(self, roots: Sequence[Root]) -> None:
        """Atomically swap the root list and notify every bound session."""
        self._roots = list(roots)
        for session in list(self._bound):
            try:
                await self.notify_changed(session)
            except Exception:
                # one dead session must not break the others
                pass


__all__ = ["RootsProvider", "StaticRootsProvider"]
