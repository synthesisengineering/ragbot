"""Model API endpoints."""

import os
import sys
from fastapi import APIRouter

# Add src directory to path
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from ragbot import (
    get_all_models,
    get_available_models,
    get_default_model,
    get_model_info,
    get_providers,
    get_temperature_settings,
    check_api_keys,
    ModelsResponse,
)
from synthesis_engine.models import ModelInfo

router = APIRouter(prefix="/api/models", tags=["models"])


# Provider display labels - loaded dynamically from engines.yaml
PROVIDER_LABELS = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google",
    "ollama": "Ollama",
}

# Providers whose models run on the user's machine (no API key, no cloud egress).
# Surfaced on /api/models/providers and per-model so the UI can show a local badge.
LOCAL_PROVIDERS = {"ollama"}


@router.get("", response_model=ModelsResponse)
async def list_models():
    """List all available models grouped by provider."""
    models_by_provider = get_available_models()
    all_models = []

    for provider, models in models_by_provider.items():
        for model in models:
            all_models.append(ModelInfo(
                id=model["id"],
                name=model["name"],
                provider=provider,
                context_window=model["context_window"],
                supports_streaming=model.get("supports_streaming", True),
                supports_system_role=model.get("supports_system_role", True),
                display_name=model.get("display_name") or model["name"],
                supports_thinking=bool(model.get("supports_thinking", False)),
                is_local=bool(model.get("is_local", provider in LOCAL_PROVIDERS)),
            ))

    return ModelsResponse(
        models=all_models,
        default_model=get_default_model()
    )


@router.get("/all")
async def list_all_models():
    """List all models (including those without API keys configured)."""
    models_by_provider = get_all_models()
    api_keys = check_api_keys()
    all_models = []

    for provider, models in models_by_provider.items():
        for model in models:
            all_models.append({
                "id": model["id"],
                "name": model["name"],
                "provider": provider,
                "context_window": model["context_window"],
                "supports_streaming": model.get("supports_streaming", True),
                "supports_system_role": model.get("supports_system_role", True),
                "available": api_keys.get(provider, False),
                "tier": model.get("tier", "routine"),
                "max_output_tokens": model.get("max_output_tokens"),
                "temperature": model.get("temperature"),
                "max_temperature": model.get("max_temperature", 2),
                "display_name": model.get("display_name") or model["name"],
                "supports_thinking": bool(model.get("supports_thinking", False)),
                "is_local": bool(model.get("is_local", provider in LOCAL_PROVIDERS)),
                "is_flagship": bool(model.get("is_flagship", False)),
            })

    return {
        "models": all_models,
        "default_model": get_default_model(),
        "api_keys_configured": api_keys,
    }


@router.get("/providers")
async def list_providers():
    """List all configured providers from engines.yaml."""
    providers = get_providers()
    return {
        "providers": [
            {
                "id": p,
                "name": PROVIDER_LABELS.get(p, p.title()),
                "is_local": p in LOCAL_PROVIDERS,
            }
            for p in providers
        ]
    }


@router.get("/temperature-settings")
async def get_temp_settings():
    """Get temperature preset settings from engines.yaml."""
    return get_temperature_settings()


@router.get("/{model_id:path}")
async def get_model(model_id: str):
    """Get details about a specific model."""
    info = get_model_info(model_id)
    if not info:
        return {"error": f"Model not found: {model_id}"}
    return info
