"""Sampling primitive.

Sampling is a *client feature offered to servers* — the server asks the
client to run an LLM completion on its behalf, optionally with a set of
tools the sampled model can use. The substrate exposes a
:class:`SamplingCallback` protocol so the runtime can plug its own LLM
backend in.

JSON-RPC method covered:

* ``sampling/createMessage`` — server-to-client request; the client must
  return a :class:`CreateMessageResult`.

The 2025-11-25 spec added ``tools`` and ``toolChoice`` parameters so a
server can hand the sampled model an explicit tool palette. Our default
handler refuses to sample (returns a polite error) until the runtime
plugs in its own implementation; that keeps the default behavior safe
without surprises.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mcp.client.context import ClientRequestContext
from mcp.types import (
    CreateMessageRequestParams,
    CreateMessageResult,
    ErrorData,
    INTERNAL_ERROR,
    METHOD_NOT_FOUND,
    TextContent,
)


@runtime_checkable
class SamplingCallback(Protocol):
    """Callback that performs a sampling request on the server's behalf."""

    async def __call__(
        self,
        context: ClientRequestContext,
        params: CreateMessageRequestParams,
    ) -> CreateMessageResult | ErrorData:
        """Run the sampling request and return either a result or an error."""


class RefusingSamplingHandler:
    """Default handler that refuses to sample.

    The MCP spec is explicit that the user must explicitly approve every
    sampling request. Without an opt-in handler from the runtime the
    safe default is to refuse — better to surface a method-not-found
    error than to silently invoke a model the user did not authorize.
    """

    async def __call__(
        self,
        context: ClientRequestContext,
        params: CreateMessageRequestParams,
    ) -> CreateMessageResult | ErrorData:
        return ErrorData(
            code=METHOD_NOT_FOUND,
            message=(
                "sampling/createMessage is not implemented by this client. "
                "Plug in a SamplingCallback to enable server-initiated LLM calls."
            ),
        )


default_sampling_handler: SamplingCallback = RefusingSamplingHandler()


def make_text_response(text: str, *, role: str = "assistant", model: str = "unknown") -> CreateMessageResult:
    """Convenience for callbacks that just want to return a text message."""
    return CreateMessageResult(
        role=role,
        content=TextContent(type="text", text=text),
        model=model,
        stopReason="endTurn",
    )


def internal_error(message: str) -> ErrorData:
    """Convenience for callbacks that want to bail with a 500-style error."""
    return ErrorData(code=INTERNAL_ERROR, message=message)


__all__ = [
    "SamplingCallback",
    "RefusingSamplingHandler",
    "default_sampling_handler",
    "make_text_response",
    "internal_error",
]
