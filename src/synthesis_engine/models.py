"""Substrate-level Pydantic models for synthesis-engineering runtimes.

These types are produced by code in this package — workspace discovery,
the engines.yaml-backed model registry, and future indexing helpers.
Runtimes consume these types directly: `from synthesis_engine.models
import WorkspaceInfo` (or `ModelInfo`, etc.).

Runtime-specific API request/response shapes (chat requests, index requests,
config responses) do NOT belong here. They live in the runtime package
that owns the HTTP/CLI surface they describe.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class WorkspaceInfo(BaseModel):
    """Information about a workspace.

    Workspaces are AI-knowledge repositories discovered by the substrate's
    `workspaces.discover_*` helpers. Every runtime built on synthesis_engine
    consumes this shape.
    """

    name: str = Field(..., description="Display name")
    dir_name: str = Field(..., description="Directory name")
    description: Optional[str] = None
    status: str = Field("active", description="Workspace status")
    type: str = Field("project", description="Workspace type")
    inherits_from: List[str] = Field(default_factory=list)
    has_instructions: bool = False
    has_datasets: bool = False
    has_archive: bool = False
    has_source: bool = False
    repo_path: Optional[str] = None


class WorkspaceList(BaseModel):
    """List of workspaces."""

    workspaces: List[WorkspaceInfo]
    count: int


class ModelInfo(BaseModel):
    """Information about an available LLM model.

    Model metadata is sourced from `engines.yaml`, which lives in
    `synthesis_engine.config`. Substrate-level because every runtime that
    routes LLM calls needs the same shape; runtimes may expose this type
    on their own API surfaces (e.g., Ragbot's `/api/models` returns
    a list of these inside a runtime-specific `ModelsResponse`).
    """

    id: str
    name: str
    provider: str
    context_window: int
    supports_streaming: bool = True
    supports_system_role: bool = True
    display_name: Optional[str] = None
    supports_thinking: bool = False
    is_local: bool = False
