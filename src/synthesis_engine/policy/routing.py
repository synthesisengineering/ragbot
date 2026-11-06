"""Per-workspace model-routing policy.

This module is the synthesis-engine substrate's source of truth for
"which LLM model is allowed to handle a request originating in this
workspace." It is consulted by the LLM-backend abstraction
(:mod:`synthesis_engine.llm`) before any model call is dispatched.

Every workspace MAY declare a ``routing.yaml`` at the workspace's root
(the same directory that already holds ``compile-config.yaml`` and the
centralized ``my-projects.yaml``). When present, the file looks like:

.. code-block:: yaml

    # routing.yaml — per-workspace model routing policy
    #
    # confidentiality: one of public, personal, client_confidential, air_gapped
    #   public               — no restrictions; safe to route anywhere
    #   personal             — operator-only data; frontier models OK
    #   client_confidential  — client data; restrict to approved models
    #   air_gapped           — must NEVER leave local infrastructure
    confidentiality: client_confidential

    # allowed_models: glob-matched IDs that are explicitly permitted
    allowed_models:
      - anthropic/claude-*
      - openai/gpt-5*

    # denied_models: glob-matched IDs that are explicitly forbidden.
    # denied_models takes precedence over allowed_models.
    denied_models:
      - "*-preview*"

    # local_only: when true, the workspace can ONLY route to local models
    # (the Ollama models in engines.yaml, any ollama_chat/ or ollama/ id, or
    # any id containing :local).
    # When local_only is true, allowed_models / denied_models are still
    # consulted as further restrictions.
    local_only: false

    # fallback_behavior: what to do when is_model_allowed returns False.
    #   deny                 — refuse the call
    #   downgrade_to_local   — silently switch to a local model
    #   warn                 — log a warning but proceed
    fallback_behavior: deny

The defaults (``confidentiality=public``, ``allowed_models=()``,
``denied_models=()``, ``local_only=False``, ``fallback_behavior=warn``)
are applied to any workspace that does not ship a ``routing.yaml``.
A one-time log warning fires when the default is taken so operators
know they are running unconfigured.

Glob support: both ``allowed_models`` and ``denied_models`` accept
fnmatch-style globs (``openai/*``, ``anthropic/claude-*``,
``*-preview*``). Matching is case-sensitive and applies to the full
model identifier (provider prefix included).
"""

from __future__ import annotations

import enum
import fnmatch
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is a hard dep
    yaml = None  # type: ignore[assignment]

from ..exceptions import ConfigurationError

logger = logging.getLogger(__name__)


# Tracks which workspace roots have already emitted the "missing routing.yaml"
# warning so a long-running process doesn't repeat the same warning every
# turn. The set is intentionally global; tests reset it via the
# ``_clear_warning_cache`` helper.
_WARNED_ROOTS: set = set()


# ---------------------------------------------------------------------------
# Example YAML (embedded so the schema is discoverable from a REPL)
# ---------------------------------------------------------------------------


EXAMPLE_ROUTING_YAML: str = """\
# routing.yaml — per-workspace model routing policy
#
# Place this file at the root of an ai-knowledge-<workspace> repo. It is
# read on demand by synthesis_engine.policy.routing.load_routing_policy.

# confidentiality: one of public, personal, client_confidential, air_gapped
#   public               — no restrictions
#   personal             — operator-only data; frontier models OK
#   client_confidential  — client data; restrict to approved models
#   air_gapped           — must NEVER leave local infrastructure
confidentiality: client_confidential

# allowed_models: glob-matched IDs that are explicitly permitted.
# Empty list (default) means "no allow-list restriction" — i.e., any
# model not on denied_models is allowed.
allowed_models:
  - anthropic/claude-*
  - openai/gpt-5*

# denied_models: glob-matched IDs that are explicitly forbidden.
# denied_models takes precedence over allowed_models.
denied_models:
  - "*-preview*"

# local_only: when true, only local models (the Ollama models in
# engines.yaml, any ollama_chat/ or ollama/ id, or any id containing
# ':local') are permitted.
local_only: false

# fallback_behavior: what to do when a disallowed model is requested.
#   deny                 — refuse the call
#   downgrade_to_local   — silently switch to a local model
#   warn                 — log a warning but proceed
fallback_behavior: deny
"""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Confidentiality(enum.IntEnum):
    """Workspace confidentiality tag.

    Ordered by strictness (ascending). Higher values are stricter. The
    cross-workspace mixing rule in :mod:`.confidentiality` uses this
    ordering directly: the effective confidentiality of a mixed op is
    ``max(participating_confidentialities)``.
    """

    PUBLIC = 0
    PERSONAL = 10
    CLIENT_CONFIDENTIAL = 20
    AIR_GAPPED = 30

    @classmethod
    def from_string(cls, value: str) -> "Confidentiality":
        """Parse a YAML string into a Confidentiality. Unknown → AIR_GAPPED.

        The fail-closed default for unknown tags is the STRICTEST tag,
        not the loosest. An operator who types ``confidentiality: top-secret``
        in their routing.yaml will see their workspace lock down rather
        than silently relax to PUBLIC.
        """
        if not isinstance(value, str):
            return cls.AIR_GAPPED
        key = value.strip().lower().replace("-", "_")
        mapping = {
            "public": cls.PUBLIC,
            "personal": cls.PERSONAL,
            "client_confidential": cls.CLIENT_CONFIDENTIAL,
            "air_gapped": cls.AIR_GAPPED,
        }
        if key in mapping:
            return mapping[key]
        logger.warning(
            "Unknown confidentiality tag %r; defaulting to AIR_GAPPED "
            "(fail-closed).",
            value,
        )
        return cls.AIR_GAPPED


class FallbackBehavior(enum.Enum):
    """What to do when a disallowed model is requested."""

    DENY = "deny"
    DOWNGRADE_TO_LOCAL = "downgrade_to_local"
    WARN = "warn"

    @classmethod
    def from_string(cls, value: str) -> "FallbackBehavior":
        """Parse a YAML string. Unknown → DENY (fail-closed)."""
        if not isinstance(value, str):
            return cls.DENY
        key = value.strip().lower().replace("-", "_")
        for member in cls:
            if member.value == key:
                return member
        logger.warning(
            "Unknown fallback_behavior %r; defaulting to DENY (fail-closed).",
            value,
        )
        return cls.DENY


# ---------------------------------------------------------------------------
# RoutingPolicy + AllowanceCheck
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RoutingPolicy:
    """Per-workspace routing policy.

    Constructed by :func:`load_routing_policy`. Frozen because the policy
    is consulted across many call sites and downstream caching depends
    on identity-stability.
    """

    confidentiality: Confidentiality = Confidentiality.PUBLIC
    allowed_models: Tuple[str, ...] = field(default_factory=tuple)
    denied_models: Tuple[str, ...] = field(default_factory=tuple)
    local_only: bool = False
    fallback_behavior: FallbackBehavior = FallbackBehavior.WARN


@dataclass(frozen=True)
class AllowanceCheck:
    """Verdict on whether one model is allowed under one policy."""

    allowed: bool
    reason: str
    suggested_fallback: Optional[str] = None


def _is_local_model(model_id: str) -> bool:
    """Does ``model_id`` refer to a locally served model? See
    :func:`synthesis_engine.config.is_local_model_id`."""
    from ..config import is_local_model_id

    return is_local_model_id(model_id)


def _suggest_local_fallback() -> Optional[str]:
    """The default local model from engines.yaml, for downgrade messaging.

    ``None`` when engines.yaml configures no local model.
    """
    from ..config import get_local_model_ids

    local = get_local_model_ids()
    return local[0] if local else None


def is_model_allowed(policy: RoutingPolicy, model_id: str) -> AllowanceCheck:
    """Check whether ``model_id`` is permitted under ``policy``.

    Rules, in order:

    1. ``denied_models`` glob match → DENIED. (denied wins over allowed)
    2. ``local_only=True`` + non-local model → DENIED.
    3. Non-empty ``allowed_models`` + no glob match → DENIED.
    4. Otherwise → ALLOWED.

    Returns an :class:`AllowanceCheck` carrying a human-readable reason
    and (when relevant) a suggested local fallback the caller can use to
    implement ``FallbackBehavior.DOWNGRADE_TO_LOCAL``.
    """

    if not isinstance(model_id, str) or not model_id:
        return AllowanceCheck(
            allowed=False,
            reason="model_id is empty or non-string.",
        )

    # Rule 1: denied wins.
    for pattern in policy.denied_models:
        if fnmatch.fnmatchcase(model_id, pattern):
            return AllowanceCheck(
                allowed=False,
                reason=(
                    f"model {model_id!r} matches denied_models pattern "
                    f"{pattern!r}."
                ),
                suggested_fallback=_suggest_local_fallback(),
            )

    # Rule 2: local_only restriction.
    if policy.local_only and not _is_local_model(model_id):
        return AllowanceCheck(
            allowed=False,
            reason=(
                f"workspace policy is local_only=True; model {model_id!r} "
                "is not local."
            ),
            suggested_fallback=_suggest_local_fallback(),
        )

    # Rule 3: allowlist enforcement (only when allowed_models is non-empty).
    if policy.allowed_models:
        for pattern in policy.allowed_models:
            if fnmatch.fnmatchcase(model_id, pattern):
                return AllowanceCheck(
                    allowed=True,
                    reason=(
                        f"model {model_id!r} matches allowed_models pattern "
                        f"{pattern!r}."
                    ),
                )
        return AllowanceCheck(
            allowed=False,
            reason=(
                f"model {model_id!r} matches no entry in allowed_models "
                f"{list(policy.allowed_models)!r}."
            ),
            suggested_fallback=_suggest_local_fallback(),
        )

    # Rule 4: nothing restricts it.
    return AllowanceCheck(
        allowed=True,
        reason=f"model {model_id!r} is allowed (no restriction applies).",
    )


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


_DEFAULT_POLICY = RoutingPolicy(
    confidentiality=Confidentiality.PUBLIC,
    allowed_models=(),
    denied_models=(),
    local_only=False,
    fallback_behavior=FallbackBehavior.WARN,
)


def _coerce_glob_list(raw, field_name: str) -> Tuple[str, ...]:
    """Validate a list-of-strings field; raises ConfigurationError on shape mismatch."""
    if raw is None:
        return ()
    if isinstance(raw, str):
        return (raw,)
    if not isinstance(raw, (list, tuple)):
        raise ConfigurationError(
            f"routing.yaml field {field_name!r} must be a list of strings, "
            f"got {type(raw).__name__}."
        )
    out: list = []
    for item in raw:
        if not isinstance(item, str):
            raise ConfigurationError(
                f"routing.yaml field {field_name!r} contains a non-string "
                f"entry {item!r}."
            )
        if item.strip():
            out.append(item.strip())
    return tuple(out)


def load_routing_policy(workspace_root: str) -> RoutingPolicy:
    """Load the per-workspace routing policy.

    Reads ``routing.yaml`` from the workspace root. When the file is
    absent, returns the default policy (PUBLIC + WARN) and emits a
    one-time log warning per workspace_root so operators know they are
    running unconfigured.

    Raises :class:`ConfigurationError` when the file exists but cannot
    be parsed or contains a field of the wrong type.
    """

    root = Path(os.path.expanduser(workspace_root))
    path = root / "routing.yaml"

    if not path.is_file():
        marker = str(path)
        if marker not in _WARNED_ROOTS:
            _WARNED_ROOTS.add(marker)
            logger.warning(
                "No routing.yaml at %s — running with default policy "
                "(confidentiality=PUBLIC, fallback_behavior=WARN). "
                "Add routing.yaml to declare workspace confidentiality.",
                marker,
            )
        return _DEFAULT_POLICY

    if yaml is None:  # pragma: no cover - PyYAML is a hard dep
        raise ConfigurationError(
            "PyYAML is required to read routing.yaml but is unavailable."
        )

    try:
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
    except (yaml.YAMLError, OSError) as exc:
        raise ConfigurationError(
            f"Failed to read routing.yaml at {path}: {exc}"
        ) from exc

    if raw is None:
        return _DEFAULT_POLICY
    if not isinstance(raw, dict):
        raise ConfigurationError(
            f"routing.yaml at {path} must contain a mapping at the top "
            f"level, got {type(raw).__name__}."
        )

    confidentiality = Confidentiality.from_string(
        raw.get("confidentiality", "public")
    )
    allowed_models = _coerce_glob_list(raw.get("allowed_models"), "allowed_models")
    denied_models = _coerce_glob_list(raw.get("denied_models"), "denied_models")
    local_only_raw = raw.get("local_only", False)
    if not isinstance(local_only_raw, bool):
        raise ConfigurationError(
            f"routing.yaml field 'local_only' must be a boolean, got "
            f"{type(local_only_raw).__name__}."
        )
    fallback_behavior = FallbackBehavior.from_string(
        raw.get("fallback_behavior", "warn")
    )

    return RoutingPolicy(
        confidentiality=confidentiality,
        allowed_models=allowed_models,
        denied_models=denied_models,
        local_only=local_only_raw,
        fallback_behavior=fallback_behavior,
    )


# ---------------------------------------------------------------------------
# Test hooks
# ---------------------------------------------------------------------------


def _clear_warning_cache() -> None:
    """Reset the per-root "missing routing.yaml" warning memo (test hook)."""
    _WARNED_ROOTS.clear()


__all__ = [
    "AllowanceCheck",
    "Confidentiality",
    "EXAMPLE_ROUTING_YAML",
    "FallbackBehavior",
    "RoutingPolicy",
    "is_model_allowed",
    "load_routing_policy",
]
