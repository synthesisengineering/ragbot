"""Tools primitive.

Wraps the two ``tools/*`` JSON-RPC methods on the server side of the MCP
spec:

* ``tools/list`` — discover the tools the server offers, including
  input schemas, output schemas, and tool annotations.
* ``tools/call`` — invoke a named tool with arguments and receive its
  content blocks (and optional structured output) back.

The wrappers are thin: they accept and return SDK-native objects so a
caller working directly against :class:`mcp.types` keeps full schema
fidelity. The higher-level :class:`synthesis_engine.mcp.client.MCPClient`
adds the registry-lookup and exception-translation layer on top.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from mcp import ClientSession
from mcp.types import (
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
    Tool,
)


async def list_tools(session: ClientSession) -> list[Tool]:
    """Return the tool catalog the server advertises.

    Pages through ``cursor`` results transparently so the caller gets the
    full list in one shot. Servers that do not paginate return everything
    in the first page.
    """
    tools: list[Tool] = []
    cursor: Optional[str] = None
    while True:
        page: ListToolsResult = await session.list_tools(
            params=PaginatedRequestParams(cursor=cursor) if cursor else None
        )
        tools.extend(page.tools)
        cursor = page.next_cursor
        if not cursor:
            return tools


async def call_tool(
    session: ClientSession,
    name: str,
    arguments: Optional[Dict[str, Any]] = None,
    *,
    timeout_seconds: Optional[float] = None,
) -> CallToolResult:
    """Invoke a tool by name with ``arguments``.

    The returned :class:`CallToolResult` carries:

    * ``content`` — a list of content blocks (text, image, embedded resource).
    * ``structured_content`` — when the tool declares an output schema, the
      parsed structured object.
    * ``is_error`` — True if the tool reported a user-facing failure.

    The caller decides how to render content; the substrate does not
    presume a particular response shape because tools vary widely.
    """
    return await session.call_tool(
        name=name,
        arguments=arguments or {},
        read_timeout_seconds=timeout_seconds,
    )


def tool_names(tools: Iterable[Tool]) -> list[str]:
    """Convenience: extract just the tool names from a catalog."""
    return [t.name for t in tools]


__all__ = ["list_tools", "call_tool", "tool_names"]
