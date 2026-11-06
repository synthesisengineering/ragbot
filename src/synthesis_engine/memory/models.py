"""Pydantic data models for the synthesis_engine memory layer.

These models are the contract every memory backend exchanges. They are
substrate-level types: callers may import them directly without depending
on any particular backend (pgvector, Mem0, Letta, Zep, etc.).

Naming aligns with the 2026 memory-architecture vocabulary so external
backends can map without renaming fields:

    * Entity.attributes carry per-attribute provenance via AttributeValue
    * Relation has validity_start / validity_end (Mem0, Zep/Graphiti shape)
    * Provenance includes source, agent_run_id, message_id, confidence
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class Provenance(BaseModel):
    """Where a fact came from, who wrote it, and how confident we are.

    Provenance is REQUIRED on every relation written through the memory
    interface. The agent loop (sub-phase 1.3) supplies this on every
    upsert; the consolidation pass supplies it when distilling session
    facts into the entity graph.
    """

    model_config = ConfigDict(extra="allow")

    source: str = Field(
        ...,
        description=(
            "Free-text source identifier. Examples: 'session:abc123', "
            "'agent_run:f0e9...', 'tool:web_search', 'manual:admin', "
            "'consolidation:session=abc123'."
        ),
    )
    agent_run_id: Optional[UUID] = Field(
        default=None,
        description="Run id of the agent invocation that produced the fact.",
    )
    message_id: Optional[str] = Field(
        default=None,
        description="Message id within a session, when applicable.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Estimated confidence in the fact (0.0..1.0).",
    )


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------


class AttributeValue(BaseModel):
    """A single attribute on an entity with per-attribute provenance.

    Entities carry attributes like {"role": ..., "email": ..., "birthday": ...}.
    Each attribute is recorded with the value, the provenance for the
    claim, and when it was last updated. Per-attribute provenance avoids
    forcing every attribute change through a relation row when the
    attribute is intrinsic to the entity rather than relational.
    """

    model_config = ConfigDict(extra="allow")

    value: Any = Field(..., description="The attribute value (any JSON-compatible).")
    provenance: Provenance
    updated_at: Optional[datetime] = Field(
        default=None,
        description=(
            "When this attribute's value was last written. Server-set on "
            "upsert; clients may pass an explicit override (e.g., for "
            "back-dated facts during consolidation)."
        ),
    )


class Entity(BaseModel):
    """A long-lived noun in the workspace graph.

    Entity identity is by ``(workspace, type, name)`` — the same name and
    type within a workspace refer to the same entity, regardless of how
    many times it is upserted. The ``id`` is server-assigned on first
    insert and stable thereafter.
    """

    model_config = ConfigDict(extra="ignore")

    id: Optional[UUID] = Field(
        default=None,
        description="Server-assigned UUID. Omit on upsert; the backend sets it.",
    )
    workspace: str
    type: str = Field(
        ...,
        description=(
            "Entity type. Conventional values include 'person', 'concept', "
            "'document', 'decision', 'project', 'event'. The schema does "
            "not constrain the vocabulary."
        ),
    )
    name: str
    attributes: Dict[str, AttributeValue] = Field(
        default_factory=dict,
        description="Per-attribute value + provenance map.",
    )
    embedding: Optional[List[float]] = Field(
        default=None,
        description=(
            "Optional embedding of the entity's effective text (name + key "
            "attributes), at the dimension engines.yaml's embedding block "
            "declares. Omit when entities aren't embedded."
        ),
    )
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @field_validator("name")
    @classmethod
    def _name_nonempty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Entity.name must be non-empty.")
        return value.strip()


# ---------------------------------------------------------------------------
# Relation
# ---------------------------------------------------------------------------


class Relation(BaseModel):
    """A bi-temporal fact connecting two entities.

    Identity is by ``id``. A ``validity_end`` of ``None`` means "still
    current." When a relation is superseded, the backend updates the prior
    row's validity_end and inserts a new row inside one transaction.
    """

    model_config = ConfigDict(extra="ignore")

    id: Optional[UUID] = None
    workspace: str
    from_entity: UUID
    to_entity: UUID
    type: str = Field(
        ...,
        description=(
            "Relation type. Conventional values: 'authored', 'cites', "
            "'contradicts', 'supersedes', 'works_at', 'depends_on', "
            "'located_in'. Not enforced by schema."
        ),
    )
    attributes: Dict[str, Any] = Field(default_factory=dict)
    validity_start: Optional[datetime] = Field(
        default=None,
        description=(
            "When the fact began to hold. Server-set to now() if omitted."
        ),
    )
    validity_end: Optional[datetime] = Field(
        default=None,
        description="When the fact stopped holding. None means current.",
    )
    provenance: Provenance
    created_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Session and user memory
# ---------------------------------------------------------------------------


class SessionMemory(BaseModel):
    """Per-session working memory (Tier 3a in the three-tier stack).

    The agent loop reads and writes this between turns; the consolidation
    pass distils durable facts out of it into the entity graph (Tier 2)
    after the session is over.
    """

    model_config = ConfigDict(extra="ignore")

    session_id: str
    user_id: Optional[str] = None
    workspace: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class UserMemory(BaseModel):
    """Per-user persistent memory (Tier 3b in the three-tier stack).

    Long-lived per-user blocks. Examples: persona summary, recurring
    preferences, pinned facts surfaced into every session's system prompt.
    """

    model_config = ConfigDict(extra="ignore")

    user_id: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Retrieval shapes
# ---------------------------------------------------------------------------


MemoryTier = Literal["vector", "graph", "session", "user"]


class MemoryQuery(BaseModel):
    """Caller-facing query envelope for three-tier retrieval."""

    model_config = ConfigDict(extra="ignore")

    text: str = Field(..., description="The natural-language query.")
    workspace: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    vector_k: int = Field(default=10, ge=0, le=200)
    graph_depth: int = Field(default=2, ge=0, le=5)
    include_session: bool = True
    include_user: bool = True
    # When set, graph traversal filters to facts that were current at
    # ``validity_at``. When None, only currently-valid facts are returned.
    validity_at: Optional[datetime] = None


class MemoryResult(BaseModel):
    """A single retrieval result with provenance and tier-of-origin.

    The merged ranked result from ``three_tier_retrieve`` is
    ``List[MemoryResult]``. Each entry tells the caller (and downstream
    LLM via citation) exactly where the fact came from and how it was
    surfaced.
    """

    model_config = ConfigDict(extra="ignore")

    tier: MemoryTier
    score: float = Field(..., description="Tier-relative ranking score.")
    text: str = Field(..., description="Rendered text suitable for prompting.")
    entity_id: Optional[UUID] = None
    relation_id: Optional[UUID] = None
    provenance: Optional[Provenance] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

