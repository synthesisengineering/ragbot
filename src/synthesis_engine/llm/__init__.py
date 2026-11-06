"""LLM-backend abstraction for Ragbot.

Two backends ship:

    - litellm  — default. Wraps :mod:`litellm`. Best provider/model coverage,
                 handles the long tail of provider quirks. Pinned >=1.83.0
                 (post-March-2026 supply-chain incident) in requirements.txt.
    - direct   — opt-in. Calls each provider's official SDK directly
                 (anthropic, openai, google-genai). Smaller surface area,
                 shorter dependency chain, no third-party gateway. Useful for
                 users who want to retire LiteLLM, or for benchmarking.

Selection via the ``RAGBOT_LLM_BACKEND`` env var (default: ``litellm``).
The interface is provider-agnostic: a single :class:`LLMBackend` ABC with
``complete()``, ``stream()``, and ``healthcheck()`` methods that exchange
``LLMRequest`` and ``LLMResponse`` dataclasses.

Adding a new backend (e.g., Bifrost, Portkey, OpenRouter) is a single new
file implementing :class:`LLMBackend`, plus one selection arm in
``get_llm_backend()`` below.

Routing-aware backend
=====================

In addition to the cached singleton, the module exposes
:func:`get_routed_llm_backend`. Given a set of per-workspace
:class:`RoutingPolicy` objects and a requested model id, the routed
backend resolves the model under the strictest applicable policy:

    * If every policy admits ``requested_model``, the backend returns the
      pair ``(backend, requested_model)`` unchanged.
    * If any policy denies the model and that policy's
      ``fallback_behavior`` is ``DENY``, a :class:`ModelDeniedError` is
      raised carrying the denying workspace and the structured reason.
    * If the strictest fallback is ``DOWNGRADE_TO_LOCAL``, the backend
      resolves to the strictest workspace's first allowed local model
      and returns it; the caller proceeds with the resolved id.
    * If the strictest fallback is ``WARN``, a warning is logged and the
      requested model is returned (the operator opted into a noisy-but-
      permissive policy).

The "strictest" workspace is the one with the highest
:class:`Confidentiality`. Ties are resolved by iterating ``policies``
in dict-insertion order so the call site controls the precedence.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Tuple

from ..config import get_local_model_ids
from ..exceptions import ConfigurationError
from ..policy import (
    AllowanceCheck,
    Confidentiality,
    FallbackBehavior,
    RoutingPolicy,
    is_model_allowed,
)
from .base import (
    LLMBackend,
    LLMRequest,
    LLMResponse,
    LLMUnavailableError,
)

logger = logging.getLogger(__name__)


_BACKEND: Optional[LLMBackend] = None


class ModelDeniedError(Exception):
    """A workspace's routing policy denied the requested model.

    Carries the structured fields a UI / API surface needs to render
    the denial without parsing free-form prose:

    * ``requested_model`` — the model id the caller asked for.
    * ``denying_workspace`` — the workspace whose policy refused.
    * ``reason`` — the human-readable reason from
      :class:`AllowanceCheck`.
    """

    def __init__(
        self,
        *,
        requested_model: str,
        denying_workspace: str,
        reason: str,
    ) -> None:
        self.requested_model = requested_model
        self.denying_workspace = denying_workspace
        self.reason = reason
        super().__init__(
            f"workspace {denying_workspace!r} denied model "
            f"{requested_model!r}: {reason}"
        )

    def to_dict(self) -> Dict[str, str]:
        """Return a JSON-friendly view for API / audit surfaces."""
        return {
            "requested_model": self.requested_model,
            "denying_workspace": self.denying_workspace,
            "reason": self.reason,
        }


def _resolve_backend_name() -> str:
    name = os.environ.get("RAGBOT_LLM_BACKEND", "litellm").strip().lower()
    if name not in ("litellm", "direct"):
        logger.warning(
            "Unknown RAGBOT_LLM_BACKEND=%r, falling back to litellm.", name,
        )
        return "litellm"
    return name


def get_llm_backend(refresh: bool = False) -> LLMBackend:
    """Return the configured LLM backend (cached singleton).

    When ``refresh`` is True, the cache is dropped and the backend is
    rebuilt — useful for tests that swap env vars.
    """

    global _BACKEND
    if _BACKEND is not None and not refresh:
        return _BACKEND

    name = _resolve_backend_name()
    if name == "direct":
        try:
            from .direct_backend import DirectBackend  # noqa: WPS433
            _BACKEND = DirectBackend()
            return _BACKEND
        except Exception as exc:  # pragma: no cover - construction failure path
            logger.warning(
                "Direct backend unavailable (%s); falling back to litellm.", exc,
            )

    from .litellm_backend import LiteLLMBackend  # noqa: WPS433
    _BACKEND = LiteLLMBackend()
    return _BACKEND


def reset_llm_backend() -> None:
    """Drop the cached backend (test hook)."""

    global _BACKEND
    _BACKEND = None


# ---------------------------------------------------------------------------
# Routing-aware resolution
# ---------------------------------------------------------------------------


def _pick_local_fallback(policy: RoutingPolicy) -> str:
    """Pick the first allowed local-model id under ``policy``.

    Prefers a concrete (non-glob) model id from
    ``policy.allowed_models`` so the downgrade respects the operator's
    declared shortlist. Glob entries (anything containing ``*`` or
    ``?``) are skipped here because the resolved id has to be a
    dispatchable model name, not a pattern. Falls back to the local
    models engines.yaml configures (its default local model first) when
    the policy is allowlist-empty or contains only globs. We don't raise
    from this helper so the routing decision stays in one place.
    """

    def _is_concrete(model_id: str) -> bool:
        return not any(ch in model_id for ch in ("*", "?", "["))

    # Try the policy's own allowed_models first; pick the first concrete
    # entry that the policy actually admits.
    for candidate in policy.allowed_models:
        if not _is_concrete(candidate):
            continue
        verdict = is_model_allowed(policy, candidate)
        if verdict.allowed:
            return candidate

    # Fall back to the configured local models. Re-check each through
    # is_model_allowed so the fallback respects the policy's own deny
    # rules (a policy can deny one local family while requiring
    # local_only, in which case the next candidate wins).
    local_models = get_local_model_ids()
    for candidate in local_models:
        verdict = is_model_allowed(policy, candidate)
        if verdict.allowed:
            return candidate

    # As a last resort, return the default local model. The caller will
    # see ``is_model_allowed=False`` and can decide whether to deny or
    # proceed with a warning; we don't raise from this helper so the
    # routing decision stays in one place.
    if not local_models:
        raise ConfigurationError(
            "A workspace policy asked to downgrade to a local model, but "
            "engines.yaml configures no local models."
        )
    return local_models[0]


def get_routed_llm_backend(
    workspace_policies: Dict[str, RoutingPolicy],
    requested_model: str,
    *,
    backend: Optional[LLMBackend] = None,
) -> Tuple[LLMBackend, str]:
    """Return ``(backend, resolved_model)`` after applying routing policy.

    Args:
        workspace_policies: Mapping of workspace name → loaded
            :class:`RoutingPolicy`. An empty mapping is treated as the
            single-workspace / unconstrained case and returns the
            requested model unchanged.
        requested_model: The model id the agent loop wants to dispatch.
        backend: Optional explicit backend. When ``None``, the cached
            singleton from :func:`get_llm_backend` is used.

    Returns:
        ``(backend, resolved_model)`` — the resolved model is the
        requested id when every policy admits it; otherwise it is the
        downgraded local fallback selected from the strictest workspace.

    Raises:
        :class:`ModelDeniedError`: when at least one policy denies the
            requested model AND the strictest such policy specifies
            ``fallback_behavior == DENY``.
    """

    resolved_backend = backend if backend is not None else get_llm_backend()

    if not workspace_policies:
        return resolved_backend, requested_model

    # Collect per-workspace verdicts. Deny verdicts go through a second
    # pass to pick the strictest denial so the fallback behaviour comes
    # from the most-restrictive workspace, not the first one to refuse.
    verdicts: Dict[str, AllowanceCheck] = {
        name: is_model_allowed(policy, requested_model)
        for name, policy in workspace_policies.items()
    }
    denying = [name for name, v in verdicts.items() if not v.allowed]

    if not denying:
        return resolved_backend, requested_model

    # Pick the strictest denying workspace. Strictness is confidentiality
    # first, then dict-insertion order. This means an AIR_GAPPED denial
    # wins over a PUBLIC denial regardless of which workspace appeared
    # first in the input mapping.
    strictest_denier: Optional[str] = None
    strictest_level: Confidentiality = Confidentiality.PUBLIC
    for name in denying:
        policy = workspace_policies[name]
        if strictest_denier is None or policy.confidentiality > strictest_level:
            strictest_denier = name
            strictest_level = policy.confidentiality
    assert strictest_denier is not None  # denying non-empty by construction

    strictest_policy = workspace_policies[strictest_denier]
    verdict = verdicts[strictest_denier]
    behavior = strictest_policy.fallback_behavior

    if behavior == FallbackBehavior.DENY:
        raise ModelDeniedError(
            requested_model=requested_model,
            denying_workspace=strictest_denier,
            reason=verdict.reason,
        )

    if behavior == FallbackBehavior.DOWNGRADE_TO_LOCAL:
        local_model = _pick_local_fallback(strictest_policy)
        logger.info(
            "Routing policy: model %r denied by workspace %r (%s); "
            "downgrading to local model %r.",
            requested_model, strictest_denier, verdict.reason, local_model,
        )
        return resolved_backend, local_model

    # WARN: log loudly and proceed with the requested model. The
    # operator opted into a permissive-but-noisy policy.
    logger.warning(
        "Routing policy: model %r denied by workspace %r (%s); "
        "proceeding anyway because fallback_behavior=WARN.",
        requested_model, strictest_denier, verdict.reason,
    )
    return resolved_backend, requested_model


__all__ = [
    "LLMBackend",
    "LLMRequest",
    "LLMResponse",
    "LLMUnavailableError",
    "ModelDeniedError",
    "get_llm_backend",
    "get_routed_llm_backend",
    "reset_llm_backend",
]
