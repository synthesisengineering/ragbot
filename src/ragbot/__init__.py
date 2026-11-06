"""Ragbot - AI Knowledge Assistant Library.

This package provides the core functionality for Ragbot:
- Chat engine with streaming support
- Workspace discovery and management
- Configuration loading
- Pydantic models for API integration

Example usage:
    from ragbot import chat, discover_workspaces, get_workspace

    # List available workspaces
    workspaces = discover_workspaces()

    # Get a specific workspace
    workspace = get_workspace("personal")

    # Chat with context from workspace
    response = chat(
        prompt="Hello!",
        workspace_name="personal",
        model="anthropic/claude-sonnet-5-5"
    )
"""

# Register demo-mode discovery filters with the substrate before anything
# else touches discovery. The filters themselves are dynamic — they read
# ``is_demo_mode()`` at call time, so toggling the env var between calls
# works without re-importing the package.
from . import _demo_registration  # noqa: F401  (side-effect import)

from synthesis_engine.config import (
    VERSION,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    load_engines_config,
    get_providers,
    get_provider_config,
    get_all_models,
    get_model_info,
    get_provider_for_model,
    get_default_model,
    get_available_models,
    get_temperature_settings,
    check_api_keys,
    load_yaml_config,
    load_data_config,
)

from synthesis_engine.keystore import (
    get_api_key,
    get_keystore,
    Keystore,
    KEYSTORE_PATH,
    CONFIG_DIR,
    USER_CONFIG_PATH,
    get_default_workspace,
    get_user_config,
    set_user_config,
    get_pinned_models,
    set_pinned_models,
    get_recent_models,
    record_recent_model,
    RECENT_MODELS_CAP,
    check_api_keys as keystore_check_api_keys,
)

def get_key_status(workspace: str = None):
    """Get detailed key status per provider for a workspace."""
    return get_keystore().get_key_status(workspace)

from .core import (
    chat,
    chat_stream,
    count_tokens,
    compact_history,
)

from synthesis_engine.workspaces import (
    discover_workspaces,
    discover_ai_knowledge_repos,
    find_ai_knowledge_root,
    resolve_workspace_paths,
    workspace_to_profile,
    load_workspaces_as_profiles,
    get_workspace,
    get_workspace_info,
    list_workspace_info,
    get_llm_specific_instruction_path,
    ENGINE_TO_INSTRUCTION_FILE,
)

from synthesis_engine.exceptions import (
    ConfigurationError,
    WorkspaceError,
    WorkspaceNotFoundError,
    ChatError,
    RAGError,
    IndexingError,
)

from .models import (
    MessageRole,
    Message,
    ChatRequest,
    ChatResponse,
    IndexStatus,
    IndexRequest,
    ModelsResponse,
    ConfigResponse,
    HealthResponse,
)

__version__ = VERSION

__all__ = [
    # Version
    "VERSION",
    "__version__",
    # Config (all from engines.yaml - the single source of truth)
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "load_engines_config",
    "get_providers",
    "get_provider_config",
    "get_all_models",
    "get_model_info",
    "get_provider_for_model",
    "get_default_model",
    "get_available_models",
    "get_temperature_settings",
    "check_api_keys",
    "load_yaml_config",
    "load_data_config",
    # Keystore & User Config
    "get_api_key",
    "get_keystore",
    "get_key_status",
    "Keystore",
    "KEYSTORE_PATH",
    "CONFIG_DIR",
    "USER_CONFIG_PATH",
    "get_default_workspace",
    "get_user_config",
    "set_user_config",
    "get_pinned_models",
    "set_pinned_models",
    "get_recent_models",
    "record_recent_model",
    "RECENT_MODELS_CAP",
    # Core
    "chat",
    "chat_stream",
    "count_tokens",
    "compact_history",
    # Workspaces
    "discover_workspaces",
    "discover_ai_knowledge_repos",
    "find_ai_knowledge_root",
    "resolve_workspace_paths",
    "workspace_to_profile",
    "load_workspaces_as_profiles",
    "get_workspace",
    "get_workspace_info",
    "list_workspace_info",
    "get_llm_specific_instruction_path",
    "ENGINE_TO_INSTRUCTION_FILE",
    # Exceptions (substrate hierarchy; base class SynthesisError lives in
    # synthesis_engine.exceptions and is not re-exported here — runtime code
    # that needs the base type imports it directly from the substrate).
    "ConfigurationError",
    "WorkspaceError",
    "WorkspaceNotFoundError",
    "ChatError",
    "RAGError",
    "IndexingError",
    # Models (runtime-specific shapes only; substrate types live in
    # synthesis_engine.models and are imported from there directly)
    "MessageRole",
    "Message",
    "ChatRequest",
    "ChatResponse",
    "IndexStatus",
    "IndexRequest",
    "ModelsResponse",
    "ConfigResponse",
    "HealthResponse",
]
