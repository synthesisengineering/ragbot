"""Resources primitive.

Resources are read-only data exposed by an MCP server. They have URIs,
content types, and optional subscriptions for live-updating cases (a
filesystem server, for example, subscribes a resource to push file-
change notifications to the client).

JSON-RPC methods covered:

* ``resources/list`` — page through the resource catalog.
* ``resources/templates/list`` — page through URI templates (RFC 6570).
* ``resources/read`` — fetch the content of a specific resource.
* ``resources/subscribe`` — request live updates for a resource.
* ``resources/unsubscribe`` — release a subscription.
"""

from __future__ import annotations

from typing import Optional

from mcp import ClientSession
from mcp.types import (
    ListResourcesResult,
    ListResourceTemplatesResult,
    PaginatedRequestParams,
    ReadResourceResult,
    Resource,
    ResourceTemplate,
)


async def list_resources(session: ClientSession) -> list[Resource]:
    """Return every resource the server advertises, fully paged."""
    resources: list[Resource] = []
    cursor: Optional[str] = None
    while True:
        page: ListResourcesResult = await session.list_resources(
            params=PaginatedRequestParams(cursor=cursor) if cursor else None
        )
        resources.extend(page.resources)
        cursor = page.next_cursor
        if not cursor:
            return resources


async def list_resource_templates(session: ClientSession) -> list[ResourceTemplate]:
    """Return every URI template the server advertises, fully paged."""
    templates: list[ResourceTemplate] = []
    cursor: Optional[str] = None
    while True:
        page: ListResourceTemplatesResult = await session.list_resource_templates(
            params=PaginatedRequestParams(cursor=cursor) if cursor else None
        )
        templates.extend(page.resource_templates)
        cursor = page.next_cursor
        if not cursor:
            return templates


async def read_resource(session: ClientSession, uri: str) -> ReadResourceResult:
    """Read the content at ``uri``.

    The returned :class:`ReadResourceResult` holds one or more content
    blocks — text or binary — that the caller renders or hands off to a
    downstream LLM context window.
    """
    return await session.read_resource(uri)


async def subscribe_resource(session: ClientSession, uri: str) -> None:
    """Subscribe to live updates for ``uri``.

    The server sends ``notifications/resources/updated`` notifications
    when the resource changes. Consumers register a notification handler
    via the SDK's ``message_handler`` to receive them; the substrate's
    higher-level client wires this for you.
    """
    await session.subscribe_resource(uri)


async def unsubscribe_resource(session: ClientSession, uri: str) -> None:
    """Release a previously-established subscription on ``uri``."""
    await session.unsubscribe_resource(uri)


__all__ = [
    "list_resources",
    "list_resource_templates",
    "read_resource",
    "subscribe_resource",
    "unsubscribe_resource",
]
