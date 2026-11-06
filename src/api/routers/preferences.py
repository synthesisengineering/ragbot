"""User-preferences API endpoints.

Persists model-selection preferences in ~/.synthesis/ragbot.yaml so the web
UI's pinned and recently-used model lists survive across sessions and across
machines (when the synthesis-engineering config home is iCloud-synced or
otherwise replicated).

Endpoints:
    GET    /api/preferences/pinned-models   → {"model_ids": [...]}
    PUT    /api/preferences/pinned-models   body: {"model_ids": [...]}
    GET    /api/preferences/recent-models   → {"model_ids": [...]}
    POST   /api/preferences/recent-models   body: {"model_id": "..."}

Recent-models uses move-to-front semantics with a cap (RECENT_MODELS_CAP).
Pinned-models is a full-replace API; order is preserved, duplicates dropped.
"""

import os
import sys
from typing import List

from fastapi import APIRouter
from pydantic import BaseModel, Field

# Add src directory to path
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from ragbot import (
    get_pinned_models,
    set_pinned_models,
    get_recent_models,
    record_recent_model,
)


router = APIRouter(prefix="/api/preferences", tags=["preferences"])


class ModelIdList(BaseModel):
    """A list of canonical model IDs (e.g. ``anthropic/claude-opus-5-5``)."""
    model_ids: List[str] = Field(default_factory=list)


class RecordModelRequest(BaseModel):
    """Body for POST /api/preferences/recent-models."""
    model_id: str


@router.get("/pinned-models", response_model=ModelIdList)
async def list_pinned_models() -> ModelIdList:
    """Return the user's pinned model IDs in display order."""
    return ModelIdList(model_ids=get_pinned_models())


@router.put("/pinned-models", response_model=ModelIdList)
async def replace_pinned_models(payload: ModelIdList) -> ModelIdList:
    """Replace the pinned-models list. Duplicates dropped, order preserved."""
    set_pinned_models(payload.model_ids)
    return ModelIdList(model_ids=get_pinned_models())


@router.get("/recent-models", response_model=ModelIdList)
async def list_recent_models() -> ModelIdList:
    """Return recently-used model IDs, newest first, capped."""
    return ModelIdList(model_ids=get_recent_models())


@router.post("/recent-models", response_model=ModelIdList)
async def post_recent_model(payload: RecordModelRequest) -> ModelIdList:
    """Record a model use. Moves the entry to the front; caps the list."""
    record_recent_model(payload.model_id)
    return ModelIdList(model_ids=get_recent_models())
