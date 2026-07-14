"""LLM-backend interface and request/response data contracts.

Every backend exchanges :class:`LLMRequest` going in and :class:`LLMResponse`
coming out. The contract is intentionally provider-agnostic — provider
quirks (max-token field naming, thinking-shape variants, etc.) are
absorbed inside backend implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LLMUnavailableError(RuntimeError):
    """Raised when a backend cannot be constructed (missing deps, config, etc.)."""


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass
class LLMRequest:
    """A request to the configured LLM backend.

    Attributes:
        model: Provider-prefixed model id (e.g., ``anthropic/claude-sonnet-4-6``).
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

def claude_generation(model: str):
    """Parse (major, minor) from a Claude model id, or return None.

    Handles ids like ``claude-opus-4-8``, ``claude-sonnet-5``,
    ``claude-fable-5``, ``claude-haiku-4-5-20251001``, with or without a
    provider prefix. Single-number generations compare as (N, 0). Date
    suffixes are not mistaken for minor versions.
    """
    import re as _re

    m = _re.search(r"claude-(?:[a-z]+-)?(\d+)(?:-(\d+))?", model.lower())
    if not m:
        return None
    major = int(m.group(1))
    minor_raw = m.group(2)
    minor = int(minor_raw) if minor_raw is not None and len(minor_raw) <= 2 else 0
    return (major, minor)


def is_claude_4_7_or_newer(model: str) -> bool:
    """Claude 4.7+ (incl. all 5.x) uses the ``thinking.type.adaptive`` API shape."""
    gen = claude_generation(model)
    return gen is not None and gen >= (4, 7)


def claude_rejects_temperature(model: str) -> bool:
    """Claude 4.8+ (incl. all 5.x) rejects the ``temperature`` parameter.

    The API returns 400 "`temperature` is deprecated for this model" for
    non-default values (verified 2026-07-14 on claude-sonnet-5 and
    claude-opus-4-8); omitting the parameter is always safe. Claude 4.7 and
    older (e.g. Haiku 4.5) still accept it.
    """
    gen = claude_generation(model)
    return gen is not None and gen >= (4, 8)
