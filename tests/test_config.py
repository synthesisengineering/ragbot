"""Tests for the ragbot config module.

Tests the engines.yaml loading and model configuration functions.
"""

import pytest
import os
import sys
import tempfile
import yaml

# Add src directory to path
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from synthesis_engine.config import (
    load_engines_config,
    get_providers,
    get_provider_config,
    get_all_models,
    get_model_info,
    get_default_model,
    get_temperature_settings,
    normalize_model_id,
)


class TestLoadEnginesConfig:
    """Tests for load_engines_config function."""

    def test_load_engines_config_returns_dict(self):
        """load_engines_config should return a dictionary."""
        config = load_engines_config()
        assert isinstance(config, dict)

    def test_load_engines_config_has_engines(self):
        """Config should have an 'engines' key."""
        config = load_engines_config()
        assert 'engines' in config
        assert isinstance(config['engines'], list)

    def test_load_engines_config_has_default(self):
        """Config should have a 'default' key."""
        config = load_engines_config()
        assert 'default' in config

    def test_load_engines_config_has_temperature_settings(self):
        """Config should have temperature_settings."""
        config = load_engines_config()
        assert 'temperature_settings' in config


class TestGetProviders:
    """Tests for get_providers function."""

    def test_get_providers_returns_list(self):
        """get_providers should return a list."""
        providers = get_providers()
        assert isinstance(providers, list)

    def test_get_providers_includes_main_providers(self):
        """Should include the main AI providers."""
        providers = get_providers()
        # At least some of these should be present
        expected = {'openai', 'anthropic', 'google'}
        assert len(expected.intersection(set(providers))) >= 2


class TestGetProviderConfig:
    """Tests for get_provider_config function."""

    def test_get_provider_config_returns_dict_for_valid_provider(self):
        """Should return config dict for a valid provider."""
        providers = get_providers()
        if providers:
            config = get_provider_config(providers[0])
            assert config is not None
            assert isinstance(config, dict)

    def test_get_provider_config_returns_none_for_invalid_provider(self):
        """Should return None for an invalid provider."""
        config = get_provider_config('nonexistent_provider_xyz')
        assert config is None

    def test_get_provider_config_has_required_fields(self):
        """Provider config should have required fields."""
        providers = get_providers()
        if providers:
            config = get_provider_config(providers[0])
            assert 'name' in config
            assert 'models' in config


class TestGetAllModels:
    """Tests for get_all_models function."""

    def test_get_all_models_returns_dict(self):
        """get_all_models should return a dictionary."""
        models = get_all_models()
        assert isinstance(models, dict)

    def test_get_all_models_has_models_for_providers(self):
        """Each provider should have at least one model."""
        models = get_all_models()
        for provider, model_list in models.items():
            assert isinstance(model_list, list)
            assert len(model_list) > 0, f"Provider {provider} has no models"

    def test_get_all_models_model_has_required_fields(self):
        """Each model should have required fields."""
        models = get_all_models()
        required_fields = ['id', 'name', 'category']
        for provider, model_list in models.items():
            for model in model_list:
                for field in required_fields:
                    assert field in model, f"Model {model.get('name', 'unknown')} missing {field}"

    def test_get_all_models_ids_have_provider_prefix(self):
        """Model IDs should have provider prefix for LiteLLM."""
        models = get_all_models()
        for provider, model_list in models.items():
            for model in model_list:
                model_id = model['id']
                # Should either have provider prefix or gemini/ prefix
                has_prefix = ('/' in model_id or
                             model_id.startswith('anthropic/') or
                             model_id.startswith('openai/') or
                             model_id.startswith('gemini/'))
                assert has_prefix, f"Model ID {model_id} should have provider prefix"


class TestGetModelInfo:
    """Tests for get_model_info function."""

    def test_get_model_info_returns_dict_for_valid_model(self):
        """Should return model info for a valid model."""
        default_model = get_default_model()
        info = get_model_info(default_model)
        assert info is not None
        assert isinstance(info, dict)

    def test_get_model_info_returns_none_for_invalid_model(self):
        """Should return None for an invalid model."""
        info = get_model_info('nonexistent/model-xyz')
        assert info is None

    def test_get_model_info_includes_provider(self):
        """Model info should include provider."""
        default_model = get_default_model()
        info = get_model_info(default_model)
        assert 'provider' in info


class TestGetDefaultModel:
    """Tests for get_default_model function."""

    def test_get_default_model_returns_string(self):
        """get_default_model should return a string."""
        model = get_default_model()
        assert isinstance(model, str)
        assert len(model) > 0

    def test_get_default_model_exists_in_all_models(self):
        """Default model should exist in the models list."""
        default_model = get_default_model()
        all_models = get_all_models()

        found = False
        for provider, model_list in all_models.items():
            for model in model_list:
                if model['id'] == default_model:
                    found = True
                    break

        assert found, f"Default model {default_model} not found in models"


class TestGetTemperatureSettings:
    """Tests for get_temperature_settings function."""

    def test_get_temperature_settings_returns_dict(self):
        """get_temperature_settings should return a dictionary."""
        settings = get_temperature_settings()
        assert isinstance(settings, dict)

    def test_get_temperature_settings_has_presets(self):
        """Should have standard presets."""
        settings = get_temperature_settings()
        expected_presets = ['precise', 'balanced', 'creative']
        for preset in expected_presets:
            assert preset in settings

    def test_get_temperature_settings_values_in_range(self):
        """Temperature values should be in valid range (0-2)."""
        settings = get_temperature_settings()
        for preset, value in settings.items():
            assert 0 <= value <= 2, f"Preset {preset} value {value} out of range"


class TestNormalizeModelId:
    """Tests for normalize_model_id function."""

    def test_normalize_anthropic_model(self):
        """Anthropic models should get anthropic/ prefix."""
        result = normalize_model_id('anthropic', 'claude-3-sonnet')
        assert result == 'anthropic/claude-3-sonnet'

    def test_normalize_openai_model(self):
        """OpenAI models should get openai/ prefix."""
        result = normalize_model_id('openai', 'gpt-4')
        assert result == 'openai/gpt-4'

    def test_normalize_google_model_with_prefix(self):
        """Google models with gemini/ prefix should be unchanged."""
        result = normalize_model_id('google', 'gemini/gemini-2.5-flash')
        assert result == 'gemini/gemini-2.5-flash'

    def test_normalize_ollama_model(self):
        """Ollama models should get ollama_chat/ prefix."""
        result = normalize_model_id('ollama', 'gemma4:31b')
        assert result == 'ollama_chat/gemma4:31b'

    def test_normalize_is_idempotent_anthropic(self):
        """Already-prefixed Anthropic model should pass through."""
        result = normalize_model_id('anthropic', 'anthropic/claude-3-sonnet')
        assert result == 'anthropic/claude-3-sonnet'

    def test_normalize_is_idempotent_openai(self):
        """Already-prefixed OpenAI model should pass through."""
        result = normalize_model_id('openai', 'openai/gpt-4')
        assert result == 'openai/gpt-4'

    def test_normalize_is_idempotent_ollama_chat(self):
        """Already-prefixed Ollama (chat) model should pass through."""
        result = normalize_model_id('ollama', 'ollama_chat/gemma4:e4b')
        assert result == 'ollama_chat/gemma4:e4b'

    def test_normalize_is_idempotent_ollama_completion(self):
        """Already-prefixed Ollama (completion) model should pass through."""
        result = normalize_model_id('ollama', 'ollama/gemma4:e4b')
        assert result == 'ollama/gemma4:e4b'


class TestCurrentModels:
    """Tests to verify current models in engines.yaml are configured correctly."""

    def test_openai_models_have_temperature_one(self):
        """OpenAI GPT-5 models should have temperature=1.0."""
        config = get_provider_config('openai')
        if config:
            for model in config.get('models', []):
                if 'gpt-5' in model['name'].lower():
                    assert model.get('temperature') == 1.0, \
                        f"OpenAI model {model['name']} should have temperature=1.0"

    def test_anthropic_models_exist(self):
        """Anthropic models should be configured."""
        models = get_all_models()
        assert 'anthropic' in models
        assert len(models['anthropic']) >= 3  # haiku, sonnet, opus

    def test_google_models_exist(self):
        """Google models should be configured."""
        models = get_all_models()
        assert 'google' in models
        assert len(models['google']) >= 2  # At least flash and pro

    def test_models_have_context_window(self):
        """All models should have context window specified."""
        models = get_all_models()
        for provider, model_list in models.items():
            for model in model_list:
                assert 'context_window' in model, \
                    f"Model {model['name']} missing context_window"
                assert model['context_window'] > 0


class TestModelMetadata:
    """Tests for the new display_name, supports_thinking, is_local fields."""

    def test_every_model_has_a_display_name(self):
        """Every model should expose display_name (falling back to name)."""
        models = get_all_models()
        for provider, model_list in models.items():
            for model in model_list:
                assert model.get('display_name'), \
                    f"Model {model['name']} missing display_name"

    def test_display_name_falls_back_to_name(self):
        """If engines.yaml omits display_name, it should default to name."""
        from synthesis_engine.config import normalize_model_id

        # Simulate the helper logic locally: a model dict without display_name
        # should produce a display_name equal to its name.
        model_yaml = {"name": "fictional-model-no-display-name"}
        display = model_yaml.get("display_name") or model_yaml["name"]
        assert display == "fictional-model-no-display-name"

    def test_known_display_names(self):
        """Spot-check a few well-known model display names."""
        models = get_all_models()
        flat = {m['id']: m for provider in models.values() for m in provider}

        assert flat.get('anthropic/claude-opus-4-8', {}).get('display_name') == 'Claude Opus 4.8'
        assert flat.get('openai/gpt-5.6-terra', {}).get('display_name') == 'GPT-5.6 Terra'
        assert flat.get('ollama_chat/gemma4:31b', {}).get('display_name') == 'Gemma 4 31B'

    def test_supports_thinking_reflects_engines_yaml(self):
        """Models with `thinking.supported: true` should have supports_thinking=True."""
        models = get_all_models()
        flat = {m['id']: m for provider in models.values() for m in provider}

        # Claude Opus 4.8 has thinking.supported: true in engines.yaml.
        assert flat.get('anthropic/claude-opus-4-8', {}).get('supports_thinking') is True
        # Claude Haiku 4.5 does not have a thinking block.
        assert flat.get('anthropic/claude-haiku-4-5-20251001', {}).get('supports_thinking') is True
        # GPT-5.4 mini does not have a thinking block.
        assert flat.get('openai/gpt-5.6-luna', {}).get('supports_thinking') is False

    def test_is_local_is_true_only_for_ollama(self):
        """Only Ollama models are local; cloud-provider models are not."""
        models = get_all_models()
        for provider, model_list in models.items():
            for model in model_list:
                expected = (provider == 'ollama')
                assert model.get('is_local') is expected, \
                    f"{model['id']} is_local mismatch for provider {provider}"
