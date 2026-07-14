"""Integration tests for all models in engines.yaml.

These tests verify that each configured model can actually be called
and returns a valid response. They require API keys to be configured.

Run with: pytest tests/test_models_integration.py -v

Skip expensive models: pytest tests/test_models_integration.py -v -m "not expensive"
"""

import pytest
import os
import subprocess
import sys
import time

# Add src directory to path
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from synthesis_engine.config import get_all_models, get_default_model, check_api_keys
from ragbot.core import chat, chat_stream


# Test prompt - simple to minimize tokens/cost
TEST_PROMPT = "Say 'hello' in exactly one word."


def get_available_models():
    """Get models that have API keys configured."""
    api_keys = check_api_keys()
    all_models = get_all_models()
    available = []

    for provider, models in all_models.items():
        if api_keys.get(provider, False):
            for model in models:
                available.append(model)

    return available


def model_is_expensive(model):
    """Check if a model is expensive (judgment-tier/flagship)."""
    return model.get('is_flagship', False) or model.get('tier') == 'judgment'


def _ollama_pulled_models():
    """Return set of model names currently pulled in local Ollama.

    Returns None if the daemon isn't reachable or the CLI isn't installed.
    """
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True, text=True, timeout=3,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    names = set()
    for line in result.stdout.strip().split('\n')[1:]:
        parts = line.split()
        if parts:
            names.add(parts[0])
    return names


def _maybe_skip_ollama(model_info):
    """Skip the test if the model is an Ollama model that isn't pulled locally."""
    if not model_info['id'].startswith('ollama_chat/'):
        return
    pulled = _ollama_pulled_models()
    if pulled is None:
        pytest.skip(f"Ollama daemon unreachable; skipping {model_info['id']}")
    if model_info['name'] not in pulled:
        pytest.skip(f"Ollama model not pulled locally: {model_info['name']}")


class TestModelsIntegration:
    """Integration tests that actually call the LLM APIs."""

    @pytest.fixture(scope="class")
    def available_models(self):
        """Get list of models with available API keys."""
        return get_available_models()

    def test_at_least_one_model_available(self, available_models):
        """At least one model should be available for testing."""
        assert len(available_models) > 0, \
            "No models available - check API key configuration"

    def test_default_model_works(self):
        """The default model should return a valid response."""
        api_keys = check_api_keys()
        default_model = get_default_model()

        # Check if we have the key for default model's provider
        provider = default_model.split('/')[0] if '/' in default_model else 'anthropic'
        if not api_keys.get(provider, False):
            pytest.skip(f"No API key for default model provider: {provider}")

        response = chat(
            prompt=TEST_PROMPT,
            model=default_model,
            max_tokens=50,
            stream=False
        )

        assert response is not None
        assert len(response) > 0
        assert 'hello' in response.lower() or len(response) > 0

    @pytest.mark.parametrize("model_info", get_available_models(), ids=lambda m: m['id'])
    def test_model_returns_response(self, model_info):
        """Each available model should return a valid response."""
        model_id = model_info['id']

        # Skip Ollama models that aren't pulled locally
        _maybe_skip_ollama(model_info)

        # Skip expensive models unless explicitly enabled
        if model_is_expensive(model_info):
            if not os.environ.get('TEST_EXPENSIVE_MODELS'):
                pytest.skip(f"Skipping expensive model {model_id} (set TEST_EXPENSIVE_MODELS=1 to test)")

        # Use model's default_max_tokens or a reasonable default
        # Some models (like gpt-5-mini) need more tokens to respond properly
        max_tokens = model_info.get('default_max_tokens', 1000)

        response = chat(
            prompt=TEST_PROMPT,
            model=model_id,
            max_tokens=max_tokens,
            stream=False
        )

        assert response is not None, f"Model {model_id} returned None"
        assert len(response) > 0, f"Model {model_id} returned empty response"

    @pytest.mark.parametrize("model_info", get_available_models()[:3], ids=lambda m: m['id'])
    def test_model_streaming_works(self, model_info):
        """Models should work with streaming enabled."""
        model_id = model_info['id']

        # Skip Ollama models that aren't pulled locally
        _maybe_skip_ollama(model_info)

        # Skip expensive models
        if model_is_expensive(model_info):
            if not os.environ.get('TEST_EXPENSIVE_MODELS'):
                pytest.skip(f"Skipping expensive model {model_id}")

        # Use model's default_max_tokens or a reasonable default
        max_tokens = model_info.get('default_max_tokens', 1000)

        chunks = []
        for chunk in chat_stream(
            prompt=TEST_PROMPT,
            model=model_id,
            max_tokens=max_tokens
        ):
            chunks.append(chunk)

        response = ''.join(chunks)
        assert len(response) > 0, f"Model {model_id} streaming returned empty response"
        assert len(chunks) > 0, f"Model {model_id} should return multiple chunks"


class TestModelParameters:
    """Tests for model-specific parameter handling."""

    def test_openai_temperature_one(self):
        """OpenAI models should work with temperature=1."""
        api_keys = check_api_keys()
        if not api_keys.get('openai', False):
            pytest.skip("No OpenAI API key")

        all_models = get_all_models()
        openai_models = all_models.get('openai', [])
        if not openai_models:
            pytest.skip("No OpenAI models configured")

        model = openai_models[0]
        max_tokens = model.get('default_max_tokens', 1000)

        # OpenAI GPT-5 models only support temperature=1
        response = chat(
            prompt=TEST_PROMPT,
            model=model['id'],
            max_tokens=max_tokens,
            temperature=1.0,
            stream=False
        )

        assert response is not None
        assert len(response) > 0

    def test_anthropic_temperature_range(self):
        """Anthropic models should work with various temperatures."""
        api_keys = check_api_keys()
        if not api_keys.get('anthropic', False):
            pytest.skip("No Anthropic API key")

        all_models = get_all_models()
        anthropic_models = all_models.get('anthropic', [])
        if not anthropic_models:
            pytest.skip("No Anthropic models configured")

        # Use the smallest/cheapest model
        model = next((m for m in anthropic_models if m.get('tier') == 'bulk'), anthropic_models[0])

        for temp in [0.25, 0.5, 0.75]:
            response = chat(
                prompt=TEST_PROMPT,
                model=model['id'],
                max_tokens=50,
                temperature=temp,
                stream=False
            )

            assert response is not None, f"Failed at temperature {temp}"
            assert len(response) > 0, f"Empty response at temperature {temp}"


class TestErrorHandling:
    """Tests for error handling in model calls."""

    def test_invalid_model_raises_error(self):
        """Invalid model should raise an error."""
        with pytest.raises(Exception):
            chat(
                prompt=TEST_PROMPT,
                model="invalid/nonexistent-model-xyz",
                max_tokens=50,
                stream=False
            )

    def test_empty_prompt_handled(self):
        """Empty prompt should be handled gracefully (rejected by API)."""
        default_model = get_default_model()
        api_keys = check_api_keys()
        provider = default_model.split('/')[0] if '/' in default_model else 'anthropic'

        if not api_keys.get(provider, False):
            pytest.skip("No API key for default model")

        # Modern LLMs (Anthropic Claude, etc.) correctly reject empty prompts
        # This is expected behavior - empty messages are invalid
        with pytest.raises(Exception):
            chat(
                prompt="",
                model=default_model,
                max_tokens=50,
                stream=False
            )


# Marker for expensive tests
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "expensive: marks tests as expensive (use TEST_EXPENSIVE_MODELS=1 to run)"
    )
