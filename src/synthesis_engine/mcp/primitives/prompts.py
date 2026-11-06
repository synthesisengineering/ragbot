"""Prompts primitive.

Prompts are server-defined message templates that the user (or the agent)
can invoke to bootstrap a conversation. Each prompt declares its argument
schema; the server fills in the messages when ``prompts/get`` is called.

JSON-RPC methods covered:

* ``prompts/list`` — enumerate available prompts and their argument
  schemas.
* ``prompts/get`` — render a specific prompt with arguments and receive
  back a list of messages ready for the LLM.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from mcp import ClientSession
from mcp.types import (
    GetPromptResult,
    ListPromptsResult,
    PaginatedRequestParams,
    Prompt,
)


async def list_prompts(session: ClientSession) -> list[Prompt]:
    """Return every prompt the server advertises, fully paged."""
    prompts: list[Prompt] = []
    cursor: Optional[str] = None
    while True:
        page: ListPromptsResult = await session.list_prompts(
            params=PaginatedRequestParams(cursor=cursor) if cursor else None
        )
        prompts.extend(page.prompts)
        cursor = page.next_cursor
        if not cursor:
            return prompts


async def get_prompt(
    session: ClientSession,
    name: str,
    arguments: Optional[Dict[str, Any]] = None,
) -> GetPromptResult:
    """Render the prompt ``name`` with ``arguments``.

    The returned :class:`GetPromptResult` carries:

    * ``description`` — human-readable description (optional).
    * ``messages`` — list of :class:`PromptMessage` ready to forward to
      the LLM.
    """
    return await session.get_prompt(name=name, arguments=arguments or {})


__all__ = ["list_prompts", "get_prompt"]
