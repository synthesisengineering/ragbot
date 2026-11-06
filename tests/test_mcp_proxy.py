"""Stdio-to-HTTP proxy relay (``synthesis_engine.mcp.proxy``) on mcp 2.x.

The relay is a low-level server whose handlers forward to an upstream
``ClientSession``. These tests swap the upstream for a fake and drive the
relay in-process, checking that results pass through whole: a tool
result keeps ``is_error`` and its structured content, and paging follows
the caller's cursor one page at a time.
"""

from __future__ import annotations

import os
import sys
from typing import Any, List

import pytest

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import mcp.types as t  # noqa: E402
from mcp import Client  # noqa: E402

from synthesis_engine.mcp.config import MCPServerConfig  # noqa: E402
from synthesis_engine.mcp.proxy import StdioHTTPProxy  # noqa: E402


class FakeUpstream:
    def __init__(self) -> None:
        self.tool_cursors: List[Any] = []
        self.read_uris: List[str] = []

    async def list_tools(self, *, params=None):
        cursor = None if params is None else params.cursor
        self.tool_cursors.append(cursor)
        if cursor is None:
            return t.ListToolsResult(
                tools=[t.Tool(name="first", inputSchema={"type": "object"})], next_cursor="p2"
            )
        return t.ListToolsResult(tools=[t.Tool(name="second", inputSchema={"type": "object"})])

    async def call_tool(self, name, arguments=None):
        return t.CallToolResult(
            content=[t.TextContent(type="text", text=f"{name} failed")],
            structured_content={"reason": "upstream"},
            is_error=True,
        )

    async def read_resource(self, uri):
        self.read_uris.append(uri)
        return t.ReadResourceResult(
            contents=[t.TextResourceContents(uri=uri, text="hello", mime_type="text/plain")]
        )

    async def get_prompt(self, name, arguments=None):
        return t.GetPromptResult(
            messages=[
                t.PromptMessage(
                    role="user", content=t.TextContent(type="text", text=f"{name}:{arguments}")
                )
            ]
        )


@pytest.fixture
def relay():
    proxy = StdioHTTPProxy(
        MCPServerConfig(id="up", name="Upstream", transport="stdio", command="upstream-cmd")
    )
    upstream = FakeUpstream()

    async def ensure_upstream():
        return upstream

    proxy._ensure_upstream = ensure_upstream  # type: ignore[method-assign]
    return proxy, upstream


@pytest.mark.asyncio
async def test_tool_listing_follows_the_callers_cursor(relay):
    proxy, upstream = relay
    async with Client(proxy._mcp_app, raise_exceptions=True) as client:
        first = await client.list_tools()
        second = await client.list_tools(cursor=first.next_cursor)
    assert [x.name for x in first.tools] == ["first"]
    assert first.next_cursor == "p2"
    assert [x.name for x in second.tools] == ["second"]
    assert upstream.tool_cursors == [None, "p2"]


@pytest.mark.asyncio
async def test_tool_result_passes_through_whole(relay):
    proxy, _ = relay
    async with Client(proxy._mcp_app, raise_exceptions=True) as client:
        result = await client.call_tool("job", {"x": 1})
    assert result.is_error is True
    assert result.structured_content == {"reason": "upstream"}
    assert result.content[0].text == "job failed"


@pytest.mark.asyncio
async def test_resource_and_prompt_requests_reach_the_upstream(relay):
    proxy, upstream = relay
    async with Client(proxy._mcp_app, raise_exceptions=True) as client:
        read = await client.read_resource("file:///tmp/a.txt")
        prompt = await client.get_prompt("greet", {"who": "you"})
    assert upstream.read_uris == ["file:///tmp/a.txt"]
    assert read.contents[0].text == "hello"
    assert prompt.messages[0].content.text == "greet:{'who': 'you'}"
