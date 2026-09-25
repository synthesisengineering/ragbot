"""Pydantic models for the Ragbot runtime.

Ragbot-specific HTTP/CLI shapes only: chat requests/responses, index
requests, config responses, health responses, the `ModelsResponse`
envelope. Substrate-level types (`WorkspaceInfo`, `WorkspaceList`,
`ModelInfo`) live in `synthesis_engine.models` and are imported from
there directly — Ragbot does NOT re-export them.

Consumers (including API routers and tests) import substrate types from
`synthesis_engine.models` and runtime envelopes from `ragbot.models`.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

from synthesis_engine.models import ModelInfo


class MessageRole(str, Enum):
    """Role of a message in conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """A single message in conversation history."""
    role: MessageRole
    content: str


class ChatRequest(BaseModel):
    """Request model for chat API."""
    prompt: str = Field(..., description="User's message")
    workspace: Optional[str] = Field(None, description="Workspace name for context")
    model: str = Field(
        "anthropic/claude-sonnet-4-6",
        description="Model identifier (provider/name) recognised by LiteLLM.",
    )
    temperature: Optional[float] = Field(
        None,
        ge=0, le=2,
        description="Creativity level. None uses the per-model default from engines.yaml.",
    )
    max_tokens: int = Field(4096, ge=1, le=8192, description="Max response tokens")
    use_rag: bool = Field(True, description="Use RAG for context retrieval")
    rag_max_tokens: int = Field(16000, ge=0, description="Max tokens for RAG context")
    history: List[Message] = Field(default_factory=list, description="Conversation history")
    stream: bool = Field(True, description="Stream the response")
    thinking_effort: Optional[str] = Field(
        None,
        description=(
            "Reasoning effort: one of 'auto', 'off', 'minimal', 'low', 'medium', "
            "'high', 'xhigh', 'max', or 'none' where supported. None reads "
            "RAGBOT_THINKING_EFFORT env or applies the engines.yaml "
            "default. Verified contracts reject unsupported explicit effort without substitution."
        ),
    )
    additional_workspaces: Optional[List[str]] = Field(
        None,
        description=(
            "Extra workspaces to retrieve alongside the primary one. "
            "None = auto-include the 'skills' workspace if it has indexed content. "
            "Empty list = opt out of cross-workspace retrieval."
        ),
    )


class ChatResponse(BaseModel):
    """Response model for non-streaming chat."""
    response: str
    model: str
    workspace: Optional[str] = None
    tokens_used: Optional[int] = None


class IndexStatus(BaseModel):
    """RAG index status for a workspace."""
    workspace: str
    indexed: bool
    chunk_count: int = 0
    last_indexed: Optional[str] = None


class IndexRequest(BaseModel):
    """Request to index a workspace."""
    force: bool = Field(False, description="Force re-indexing even if up to date")


class ModelsResponse(BaseModel):
    """List of available models — Ragbot's `/api/models` response envelope.

    The `ModelInfo` element type is substrate-level (sourced from
    engines.yaml); the envelope shape is runtime-specific to Ragbot's API.
    """
    models: List[ModelInfo]
    default_model: str


class ApiKeyStatus(BaseModel):
    """API key configuration status."""
    provider: str
    configured: bool
    source: str = Field("default", description="Where the key comes from: 'default', 'workspace', or 'none'")


class ConfigResponse(BaseModel):
    """Configuration response."""
    version: str
    ai_knowledge_root: Optional[str] = None
    workspace_count: int = 0
    rag_available: bool = False
    default_model: str = "anthropic/claude-sonnet-4-6"
    default_workspace: Optional[str] = None
    api_keys: Dict[str, bool] = Field(
        default_factory=dict,
        description="Provider to configured status",
    )
    workspaces_with_keys: List[str] = Field(
        default_factory=list,
        description="Workspaces with custom API keys",
    )
    vector_backend: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Active vector backend health and metadata: "
            "{backend, ok, pgvector_version (if pgvector), workspaces_count, ...}."
        ),
    )
    demo_mode: bool = Field(
        False,
        description=(
            "True when RAGBOT_DEMO=1 is set on the server. The Web UI "
            "renders a banner and the discovery layer hard-isolates from "
            "the user's real workspaces and skills."
        ),
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    version: str
    rag_available: bool = False
    vector_backend: Dict[str, Any] = Field(
        default_factory=dict,
        description="Active vector backend health: {backend, ok, ...}",
    )
    demo_mode: bool = Field(
        False,
        description="True when RAGBOT_DEMO=1 is set on the server.",
    )
