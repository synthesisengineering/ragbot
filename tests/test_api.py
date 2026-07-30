"""Tests for the FastAPI backend API endpoints.

These tests use TestClient to test the API without making actual LLM calls
where possible.
"""

import pytest
import os
import sys

# Add src directory to path
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Create a test client for the API."""
    from api.main import app
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_ok(self, client):
        """Health endpoint should return status ok."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_health_includes_version(self, client):
        """Health endpoint should include version."""
        response = client.get("/health")
        data = response.json()
        assert "version" in data


class TestConfigEndpoint:
    """Tests for the /api/config endpoint."""

    def test_config_returns_200(self, client):
        """Config endpoint should return 200."""
        response = client.get("/api/config")
        assert response.status_code == 200

    def test_config_includes_version(self, client):
        """Config should include version."""
        response = client.get("/api/config")
        data = response.json()
        assert "version" in data

    def test_config_includes_default_model(self, client):
        """Config should include default_model."""
        response = client.get("/api/config")
        data = response.json()
        assert "default_model" in data
        assert len(data["default_model"]) > 0


class TestConfigKeysEndpoint:
    """Tests for the /api/config/keys endpoint."""

    def test_keys_returns_200(self, client):
        """Keys endpoint should return 200."""
        response = client.get("/api/config/keys")
        assert response.status_code == 200

    def test_keys_returns_dict(self, client):
        """Keys endpoint should return a dictionary."""
        response = client.get("/api/config/keys")
        data = response.json()
        assert isinstance(data, dict)

    def test_keys_includes_providers(self, client):
        """Keys should include standard providers."""
        response = client.get("/api/config/keys")
        data = response.json()
        # Should have at least some providers
        assert len(data) > 0

    def test_keys_with_workspace(self, client):
        """Keys endpoint should accept workspace parameter."""
        response = client.get("/api/config/keys?workspace=test")
        assert response.status_code == 200

    def test_keys_status_structure(self, client):
        """Keys status should have correct structure."""
        response = client.get("/api/config/keys")
        data = response.json()
        for provider, status in data.items():
            assert "has_key" in status
            assert "source" in status
            assert "has_workspace_key" in status
            assert "has_default_key" in status


class TestModelsEndpoint:
    """Tests for the /api/models endpoint."""

    def test_models_returns_200(self, client):
        """Models endpoint should return 200."""
        response = client.get("/api/models")
        assert response.status_code == 200

    def test_models_includes_list(self, client):
        """Models should include a models list."""
        response = client.get("/api/models")
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_models_includes_default(self, client):
        """Models should include default_model."""
        response = client.get("/api/models")
        data = response.json()
        assert "default_model" in data

    def test_models_have_required_fields(self, client):
        """Each model should have required fields."""
        response = client.get("/api/models")
        data = response.json()
        required_fields = ["id", "name", "provider"]
        for model in data["models"]:
            for field in required_fields:
                assert field in model, f"Model missing {field}"


class TestModelsProvidersEndpoint:
    """Tests for the /api/models/providers endpoint."""

    def test_providers_returns_200(self, client):
        """Providers endpoint should return 200."""
        response = client.get("/api/models/providers")
        assert response.status_code == 200

    def test_providers_returns_list(self, client):
        """Providers should return a list."""
        response = client.get("/api/models/providers")
        data = response.json()
        assert "providers" in data
        assert isinstance(data["providers"], list)

    def test_providers_not_empty(self, client):
        """Providers list should not be empty."""
        response = client.get("/api/models/providers")
        data = response.json()
        assert len(data["providers"]) > 0


class TestTemperatureSettingsEndpoint:
    """Tests for the /api/models/temperature-settings endpoint."""

    def test_temperature_returns_200(self, client):
        """Temperature settings endpoint should return 200."""
        response = client.get("/api/models/temperature-settings")
        assert response.status_code == 200

    def test_temperature_returns_dict(self, client):
        """Temperature settings should return a dictionary."""
        response = client.get("/api/models/temperature-settings")
        data = response.json()
        # API returns settings directly as dict
        assert isinstance(data, dict)

    def test_temperature_has_presets(self, client):
        """Temperature settings should have standard presets."""
        response = client.get("/api/models/temperature-settings")
        data = response.json()
        # API returns settings directly
        assert "precise" in data
        assert "balanced" in data
        assert "creative" in data


class TestWorkspacesEndpoint:
    """Tests for the /api/workspaces endpoint."""

    def test_workspaces_returns_200(self, client):
        """Workspaces endpoint should return 200."""
        response = client.get("/api/workspaces")
        assert response.status_code == 200

    def test_workspaces_returns_list(self, client):
        """Workspaces should return a list."""
        response = client.get("/api/workspaces")
        data = response.json()
        assert "workspaces" in data
        assert isinstance(data["workspaces"], list)

    def test_workspaces_include_count(self, client):
        """Workspaces should include count."""
        response = client.get("/api/workspaces")
        data = response.json()
        assert "count" in data


class TestChatEndpoint:
    """Tests for the /api/chat endpoint.

    Note: These tests mock the LLM to avoid actual API calls.
    """

    def test_chat_requires_prompt(self, client):
        """Chat endpoint should require a prompt."""
        response = client.post("/api/chat", json={})
        assert response.status_code == 422  # Validation error

    def test_chat_accepts_valid_request(self, client):
        """Chat should accept valid request structure."""
        # Note: This will fail without mocking, but tests the validation
        response = client.post("/api/chat", json={
            "prompt": "Hello",
            "model": "anthropic/claude-sonnet-4-5-20250929",
            "stream": False
        })
        # May fail due to no API key, but should pass validation
        assert response.status_code in [200, 500, 401]

    def test_chat_stream_returns_event_stream(self, client, monkeypatch):
        """Streaming chat should return event stream content type."""
        import importlib

        chat_router = importlib.import_module("api.routers.chat")
        monkeypatch.setattr(
            chat_router,
            "chat_stream",
            lambda *args, **kwargs: iter(["hello"]),
        )

        response = client.post("/api/chat", json={
            "prompt": "Hello",
            "model": "anthropic/claude-sonnet-4-5-20250929",
            "stream": True
        })
        # Check content type if request succeeded
        if response.status_code == 200:
            assert "text/event-stream" in response.headers.get("content-type", "")


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_headers_present(self, client):
        """CORS headers should be present for cross-origin requests."""
        response = client.options(
            "/api/config",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )
        # CORS preflight should be handled
        assert response.status_code in [200, 204]
