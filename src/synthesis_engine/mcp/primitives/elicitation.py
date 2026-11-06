"""Elicitation primitive.

Elicitation is a *client feature offered to servers* in which the server
asks the user (via the client) for additional information mid-call. The
2025-11-25 spec recognises two modes:

* **Form mode** — the server supplies a JSON Schema describing the input
  shape; the client renders a form and returns the collected values.
* **URL mode** — the server supplies a URL; the client opens it in the
  user's browser. Useful when the server needs the user to complete a
  flow (OAuth, payment, captcha) without ever exposing credentials to
  the client.

JSON-RPC method covered:

* ``elicitation/create`` — server-to-client request.

This module defines the :class:`ElicitationCallback` protocol the runtime
plugs in. The substrate ships a refusing default for the same reason as
sampling — user consent is the default-deny.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mcp.client.context import ClientRequestContext
from mcp.types import (
    ElicitRequestParams,
    ElicitResult,
    ErrorData,
    METHOD_NOT_FOUND,
)


@runtime_checkable
class ElicitationCallback(Protocol):
    """Callback that gathers user input on the server's behalf."""

    async def __call__(
        self,
        context: ClientRequestContext,
        params: ElicitRequestParams,
    ) -> ElicitResult | ErrorData:
        """Run the elicitation and return either a result or an error."""


class RefusingElicitationHandler:
    """Default handler that refuses elicitation requests."""

    async def __call__(
        self,
        context: ClientRequestContext,
        params: ElicitRequestParams,
    ) -> ElicitResult | ErrorData:
        return ErrorData(
            code=METHOD_NOT_FOUND,
            message=(
                "elicitation/create is not implemented by this client. "
                "Plug in an ElicitationCallback to enable structured user prompts."
            ),
        )


default_elicitation_handler: ElicitationCallback = RefusingElicitationHandler()


def cancel_result() -> ElicitResult:
    """Convenience: return an elicitation result indicating the user cancelled."""
    return ElicitResult(action="cancel")


def decline_result() -> ElicitResult:
    """Convenience: return an elicitation result indicating the user declined."""
    return ElicitResult(action="decline")


def accept_result(content: dict) -> ElicitResult:
    """Convenience: return an elicitation result with collected content."""
    return ElicitResult(action="accept", content=content)


__all__ = [
    "ElicitationCallback",
    "RefusingElicitationHandler",
    "default_elicitation_handler",
    "cancel_result",
    "decline_result",
    "accept_result",
]
