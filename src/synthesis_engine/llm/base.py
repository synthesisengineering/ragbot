"""LLM-backend interface and request/response data contracts.

Every backend exchanges :class:`LLMRequest` going in and :class:`LLMResponse`
coming out. The contract is intentionally provider-agnostic — provider
quirks (max-token field naming, thinking-shape variants, etc.) are
absorbed inside backend implementations.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


#: Env var selecting the billed OpenAI organization for multi-org keys.
#: Honored by every backend for OpenAI models; unset means the SDK's own
#: default (or OPENAI_ORG_ID) applies.
OPENAI_ORGANIZATION_ENV = "OPENAI_ORGANIZATION"


def openai_organization_from_env() -> Optional[str]:
    """Return the configured OpenAI organization id, or None when unset.

    Blank values are treated as unset so a commented template line that
    got uncommented without a value cannot bill the wrong org.
    """
    value = os.environ.get(OPENAI_ORGANIZATION_ENV, "").strip()
    return value or None


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LLMUnavailableError(RuntimeError):
    """Raised when a backend cannot be constructed (missing deps, config, etc.)."""


class OutputBudgetExhaustedError(RuntimeError):
    """Raised when a model stops at its output limit before writing any reply.

    Reasoning tokens count against ``max_tokens``. A reasoning model given a
    small budget can spend all of it thinking and return empty content with a
    length finish reason. Passing that on as an empty reply would look like a
    model that had nothing to say.
    """


# Finish reasons meaning "stopped at the output limit", normalised to lower
# case: OpenAI and LiteLLM report ``length``, Anthropic ``max_tokens`` and
# google-genai the ``FinishReason.MAX_TOKENS`` enum.
_OUTPUT_LIMIT_FINISH_REASONS = frozenset({"length", "max_tokens"})


def _normalise_finish_reason(finish_reason: Any) -> Optional[str]:
    if finish_reason is None:
        return None
    name = getattr(finish_reason, "name", None)  # google-genai enums
    return str(name if name is not None else finish_reason).lower()


def require_reply(text: str, finish_reason: Any, request: "LLMRequest") -> None:
    """Raise OutputBudgetExhaustedError when a completion produced no text
    because it hit its output limit.

    Every backend calls this for both completion and streaming, so an empty
    reply never reaches a caller as if the model had finished.
    """
    if text:
        return
    if _normalise_finish_reason(finish_reason) in _OUTPUT_LIMIT_FINISH_REASONS:
        raise OutputBudgetExhaustedError(
            f"{request.model} used its whole {request.max_tokens}-token output "
            "budget without writing a reply; reasoning counts against that "
            "budget. Raise max_tokens or lower thinking_effort."
        )


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass
class LLMRequest:
    """A request to the configured LLM backend.

    Attributes:
        model: Provider-prefixed model id (e.g., ``anthropic/claude-sonnet-5-5``).
        messages: OpenAI-style chat messages. Each item: ``{role, content}``.
        temperature: 0..2; None defers to the model's engines.yaml default.
        max_tokens: Maximum tokens to generate in the response.
        api_key: Optional provider API key. None lets the backend fall back
            to env / shared keystore.
        thinking: Provider-native thinking shape (e.g.,
            ``{"type": "adaptive"}`` for Claude 4.7+). Mutually exclusive
            with ``reasoning_effort`` per call; backends decide which to
            forward when both are set.
        reasoning_effort: Cross-provider shorthand (``minimal|low|medium|high``).
            Backends translate to the provider-native shape.
        extra: Free-form passthrough for provider-specific knobs not yet
            modelled here. Stays out of the typed surface to avoid pinning
            us to any one provider's API.
    """

    model: str
    messages: List[Dict[str, str]]
    temperature: Optional[float] = None
    max_tokens: int = 4096
    api_key: Optional[str] = None
    thinking: Optional[Dict[str, Any]] = None
    reasoning_effort: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """A non-streaming response from the configured backend."""

    text: str
    model: str
    backend: str
    finish_reason: Optional[str] = None
    usage: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------


class LLMBackend(ABC):
    """The contract every backend implements."""

    backend_name: str = "abstract"

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Run a non-streaming completion. Returns the assembled response."""

    @abstractmethod
    def stream(
        self,
        request: LLMRequest,
        on_chunk: Callable[[str], None],
    ) -> str:
        """Run a streaming completion.

        ``on_chunk`` is invoked with each text delta as it arrives. The
        full assembled string is also returned so callers that want both
        progressive UI updates AND a final assembled value get them in
        one call.
        """

    @abstractmethod
    def healthcheck(self) -> Dict[str, Any]:
        """Return ``{backend, ok: bool, ...detail}``. Used by /health."""
