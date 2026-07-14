"""Configuration loading for Ragbot.

This module handles loading and managing configuration from various sources.
All model and provider configuration is loaded from engines.yaml - never hardcoded.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml

from .exceptions import ConfigurationError
from .keystore import get_api_key, check_api_keys as keystore_check_api_keys


def _read_version() -> str:
    """Read the project version.

    Resolution order:
      1. ``RAGBOT_VERSION`` env var (useful for development overrides).
      2. ``VERSION`` file at the repository root (single source of truth).
      3. Hard-coded fallback (only hit if neither of the above is available,
         e.g., the package is imported from an installed wheel without the
         VERSION file).
    """

    env = os.environ.get("RAGBOT_VERSION")
    if env:
        return env.strip()
    # repo root is two parents above this file: src/ragbot/config.py → repo
    candidates = [
        Path(__file__).resolve().parents[2] / "VERSION",
        Path(__file__).resolve().parents[3] / "VERSION",  # in case of layout shift
    ]
    for path in candidates:
        try:
            return path.read_text().strip() or "0.0.0"
        except OSError:
            continue
    return "0.0.0"


VERSION = _read_version()

# Default settings (fallbacks only - prefer engines.yaml values)
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.75
DEFAULT_MAX_INPUT_TOKENS = 128000

# Cached engines config
_engines_config: Optional[Dict[str, Any]] = None
_engines_config_path: Optional[str] = None


def _find_engines_yaml() -> str:
    """Find the engines.yaml file.

    Searches in order:
    1. RAGBOT_ENGINES_PATH environment variable
    2. Current working directory
    3. Ragbot package directory (src/ragbot/../..)
    4. ~/.synthesis/ (synthesis-engineering shared config home)
    5. ~/.config/ragbot/ (legacy fallback)

    Returns:
        Path to engines.yaml

    Raises:
        ConfigurationError: If engines.yaml cannot be found
    """
    search_paths = []

    # 1. Environment variable
    env_path = os.environ.get('RAGBOT_ENGINES_PATH')
    if env_path:
        search_paths.append(env_path)

    # 2. Current working directory
    search_paths.append(os.path.join(os.getcwd(), 'engines.yaml'))

    # 3. Ragbot package directory (for development)
    package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    search_paths.append(os.path.join(package_dir, 'engines.yaml'))

    # 4. Synthesis-engineering shared config home
    search_paths.append(os.path.expanduser('~/.synthesis/engines.yaml'))

    # 5. Legacy user config directory
    search_paths.append(os.path.expanduser('~/.config/ragbot/engines.yaml'))

    for path in search_paths:
        if os.path.exists(path):
            return path

    raise ConfigurationError(
        f"engines.yaml not found. Searched: {', '.join(search_paths)}"
    )


def load_engines_config(force_reload: bool = False) -> Dict[str, Any]:
    """Load the engines configuration from engines.yaml.

    This is the single source of truth for all model and provider configuration.

    Args:
        force_reload: Force reloading from disk even if cached

    Returns:
        Dictionary with engines configuration
    """
    global _engines_config, _engines_config_path

    if _engines_config is not None and not force_reload:
        return _engines_config

    path = _find_engines_yaml()
    _engines_config_path = path

    try:
        with open(path, 'r') as f:
            _engines_config = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in {path}: {e}")

    return _engines_config


def get_providers() -> List[str]:
    """Get list of configured provider names.

    Returns:
        List of provider names (e.g., ['openai', 'anthropic', 'google'])
    """
    config = load_engines_config()
    return [engine['name'] for engine in config.get('engines', [])]


def get_provider_config(provider: str) -> Optional[Dict[str, Any]]:
    """Get configuration for a specific provider.

    Args:
        provider: Provider name (e.g., 'anthropic')

    Returns:
        Provider configuration dict, or None if not found
    """
    config = load_engines_config()
    for engine in config.get('engines', []):
        if engine['name'] == provider:
            return engine
    return None


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to YAML file

    Returns:
        Dictionary with config values

    Raises:
        ConfigurationError: If file cannot be loaded
    """
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in {path}: {e}")


def load_data_config(data_root: str) -> Dict[str, Any]:
    """Load global configuration from workspaces/config.yaml.

    Args:
        data_root: Root directory containing the workspaces folder

    Returns:
        Dictionary with config values
    """
    config_path = os.path.join(data_root, 'workspaces', 'config.yaml')
    return load_yaml_config(config_path)


def normalize_model_id(provider: str, model_name: str) -> str:
    """Normalize model ID format for LiteLLM.

    Idempotent: if ``model_name`` is already prefixed with the correct
    provider scheme, it is returned unchanged. Otherwise the prefix is
    added. Safe to call on user-supplied model strings.

    LiteLLM requires provider prefixes to route correctly:
    - Anthropic: 'anthropic/{model}'
    - OpenAI: 'openai/{model}'
    - Ollama: 'ollama_chat/{model}' (local server, http://localhost:11434)
    - Google: Already prefixed with 'gemini/' in engines.yaml

    Args:
        provider: Provider name (from engines.yaml ``name`` field).
        model_name: Model name from engines.yaml, or a user-supplied
            override which may already include the provider prefix.

    Returns:
        Normalized model ID for LiteLLM API use.
    """
    if provider == 'anthropic':
        if model_name.startswith('anthropic/'):
            return model_name
        return f"anthropic/{model_name}"
    if provider == 'openai':
        if model_name.startswith('openai/'):
            return model_name
        return f"openai/{model_name}"
    if provider == 'ollama':
        if model_name.startswith('ollama_chat/') or model_name.startswith('ollama/'):
            return model_name
        return f"ollama_chat/{model_name}"
    # Google models already have gemini/ prefix in engines.yaml; other
    # providers pass through unchanged.
    return model_name


def get_all_models() -> Dict[str, List[Dict[str, Any]]]:
    """Get all available models grouped by provider.

    Loads from engines.yaml - the single source of truth.

    Returns:
        Dictionary mapping provider to list of model configs
    """
    config = load_engines_config()
    result = {}

    for engine in config.get('engines', []):
        provider = engine['name']
        models = []

        for model in engine.get('models', []):
            model_id = normalize_model_id(provider, model['name'])
            thinking_meta = model.get('thinking') or {}
            models.append({
                "id": model_id,
                "name": model['name'],
                # Human-readable label for UIs; falls back to the canonical name.
                "display_name": model.get('display_name') or model['name'],
                "tier": model.get('tier', 'routine'),
                "context_window": model.get('max_input_tokens', 128000),
                "max_output_tokens": model.get('max_output_tokens'),
                "supports_streaming": True,  # All current models support streaming
                "supports_system_role": model.get('supports_system_role', True),
                "temperature": model.get('temperature', DEFAULT_TEMPERATURE),
                "max_temperature": model.get('max_temperature', 1.0),
                "default_max_tokens": model.get('default_max_tokens', DEFAULT_MAX_TOKENS),
                "is_flagship": model.get('is_flagship', False),
                # Forward thinking metadata (supported flag, mode, modes,
                # features) so callers can decide how to set reasoning_effort.
                "thinking": thinking_meta,
                # Convenience boolean for frontends that just want to know
                # whether the model surfaces a thinking-effort control.
                "supports_thinking": bool(thinking_meta.get('supported')),
                # True for local providers (no API key required, runs on-device).
                "is_local": provider == 'ollama',
            })

        if models:
            result[provider] = models

    return result


def get_model_info(model_id: str) -> Optional[Dict[str, Any]]:
    """Get information about a specific model.

    Args:
        model_id: Model identifier (e.g., 'anthropic/claude-sonnet-4-5-20250929')

    Returns:
        Model configuration dict, or None if not found
    """
    all_models = get_all_models()
    for provider, models in all_models.items():
        for model in models:
            if model["id"] == model_id:
                return {**model, "provider": provider}
    return None


def get_provider_for_model(model_id: str) -> str:
    """Get the provider name for a model by looking it up in engines.yaml.

    This is the authoritative way to determine which provider a model belongs to.
    Uses engines.yaml as the single source of truth rather than pattern matching
    on model names (which would be fragile for future models like "opengpt").

    Args:
        model_id: Model identifier in any format:
            - Full litellm format: 'anthropic/claude-sonnet-4-5-20250929'
            - Just model name: 'claude-sonnet-4-5-20250929'
            - With provider prefix: 'openai/gpt-5.2'

    Returns:
        Provider name ('anthropic', 'openai', 'google') or 'anthropic' as fallback
    """
    config = load_engines_config()

    # Normalize: strip any provider prefix to get the raw model name
    raw_model = model_id
    for prefix in ['anthropic/', 'openai/', 'gemini/', 'google/']:
        if model_id.lower().startswith(prefix):
            raw_model = model_id[len(prefix):]
            break

    # Search engines.yaml for this model
    for engine in config.get('engines', []):
        provider = engine['name']
        for model in engine.get('models', []):
            model_name = model['name']
            # Handle Google models which have gemini/ prefix in engines.yaml
            if model_name.startswith('gemini/'):
                model_name_stripped = model_name[7:]  # Remove 'gemini/' prefix
            else:
                model_name_stripped = model_name

            # Match against raw model name or full model name
            if raw_model == model_name or raw_model == model_name_stripped:
                return provider
            # Also check if the input was the full litellm format
            if model_id == normalize_model_id(provider, model['name']):
                return provider

    # Fallback: check if the original model_id had a provider prefix we can trust
    if model_id.lower().startswith('anthropic/'):
        return 'anthropic'
    if model_id.lower().startswith('openai/'):
        return 'openai'
    if model_id.lower().startswith('gemini/') or model_id.lower().startswith('google/'):
        return 'google'

    # Ultimate fallback to default provider from config
    return config.get('default', 'anthropic')


def get_default_model() -> str:
    """Get the default model ID.

    Uses the default provider from engines.yaml, then that provider's default_model.

    Returns:
        Default model identifier
    """
    config = load_engines_config()
    default_provider = config.get('default', 'anthropic')

    provider_config = get_provider_config(default_provider)
    if provider_config:
        default_model = provider_config.get('default_model')
        if default_model:
            return normalize_model_id(default_provider, default_model)

    # Fallback: first model of default provider
    all_models = get_all_models()
    if default_provider in all_models and all_models[default_provider]:
        return all_models[default_provider][0]['id']

    # Ultimate fallback
    return "anthropic/claude-sonnet-4-5-20250929"


def get_temperature_settings() -> Dict[str, float]:
    """Get temperature preset settings from engines.yaml.

    Returns:
        Dictionary mapping preset name to temperature value
    """
    config = load_engines_config()
    return config.get('temperature_settings', {
        'precise': 0.25,
        'balanced': 0.50,
        'creative': 0.75,
    })


def check_api_keys(workspace: Optional[str] = None) -> Dict[str, bool]:
    """Check which API keys are configured.

    Args:
        workspace: Optional workspace name for workspace-specific keys

    Returns:
        Dictionary mapping provider to boolean availability
    """
    return keystore_check_api_keys(workspace)


def get_available_models(workspace: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Get models filtered by available API keys.

    Args:
        workspace: Optional workspace name for workspace-specific keys

    Returns:
        Dictionary mapping provider to list of available model configs
    """
    api_keys = check_api_keys(workspace)
    all_models = get_all_models()
    available = {}

    for provider, models in all_models.items():
        if api_keys.get(provider, False):
            available[provider] = models

    return available


def get_model_by_tier(provider: str, tier: str) -> Optional[str]:
    """Get a model ID for a specific provider and tier.

    This enables provider-agnostic model selection. Instead of hardcoding
    "claude-haiku" for fast operations, request tier="bulk" and get
    the appropriate fast model for that provider.

    Tiers defined in engines.yaml (same vocabulary as the
    synthesis-model-tiers skill's tiers.yaml — labels name the work):
    - "bulk": Fast, cost-effective models (Haiku, GPT-5.6 Luna, Flash Lite)
    - "routine": Balanced models (Sonnet, GPT-5.6 Terra, Flash)
    - "judgment": Most capable models (Fable, Opus, GPT-5.6 Sol, Gemini Pro)

    Args:
        provider: Provider name ('anthropic', 'openai', 'google')
        tier: Model tier ('bulk', 'routine', 'judgment')

    Returns:
        Model ID in LiteLLM format (e.g., 'anthropic/claude-haiku-4-5-20251001'),
        or None if no model found for that provider/tier
    """
    all_models = get_all_models()
    provider_models = all_models.get(provider, [])

    for model in provider_models:
        if model.get('tier') == tier:
            return model['id']

    return None


def get_fast_model_for_provider(model_id: str) -> Optional[str]:
    """Get the fast (bulk-tier) model for the same provider as model_id.

    This is used for auxiliary LLM calls (planner, reranker, etc.) where we
    want to use a fast model from the same provider as the user's selected
    model. This ensures consistent API key usage and billing.

    Example:
        get_fast_model_for_provider("anthropic/claude-fable-5")
        → "anthropic/claude-haiku-4-5-20251001"

        get_fast_model_for_provider("openai/gpt-5.6-sol")
        → "openai/gpt-5.6-luna"

    Args:
        model_id: The user's selected model ID

    Returns:
        Model ID for the fast model of the same provider,
        or None if not found
    """
    provider = get_provider_for_model(model_id)
    return get_model_by_tier(provider, 'bulk')
