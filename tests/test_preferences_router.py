"""Tests for the /api/preferences/* router.

Uses FastAPI's TestClient and an isolated tmp_path for the synthesis config
home so the user's real ~/.synthesis/ragbot.yaml is never touched.
"""

import os
import sys

import pytest

# Add src directory to path
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


pytest.importorskip("sse_starlette", reason="API tests require sse_starlette (installed in the Docker stack)")


@pytest.fixture()
def isolated_synthesis_home(tmp_path, monkeypatch):
    """Redirect the synthesis config home and reset the cached user config."""
    import synthesis_engine.keystore as ks

    monkeypatch.setattr(ks, "SYNTHESIS_DIR", tmp_path)
    monkeypatch.setattr(ks, "USER_CONFIG_PATH", tmp_path / "ragbot.yaml")
    monkeypatch.setattr(ks, "LEGACY_USER_CONFIG_PATH", tmp_path / "_legacy_does_not_exist.yaml")
    monkeypatch.setattr(ks, "_user_config", None)
    yield tmp_path
    monkeypatch.setattr(ks, "_user_config", None)


@pytest.fixture()
def client(isolated_synthesis_home):
    """A FastAPI TestClient backed by an isolated synthesis home."""
    from fastapi.testclient import TestClient

    from api.main import app
    return TestClient(app)


class TestPinnedModelsEndpoint:
    def test_get_returns_empty_initially(self, client):
        r = client.get("/api/preferences/pinned-models")
        assert r.status_code == 200
        assert r.json() == {"model_ids": []}

    def test_put_replaces_full_list(self, client):
        body = {"model_ids": ["anthropic/claude-opus-4-8", "ollama_chat/gemma4:31b"]}
        r = client.put("/api/preferences/pinned-models", json=body)
        assert r.status_code == 200
        assert r.json()["model_ids"] == body["model_ids"]

        # Round-trips.
        r2 = client.get("/api/preferences/pinned-models")
        assert r2.json() == body

    def test_put_dedupes_and_preserves_order(self, client):
        body = {"model_ids": ["a", "b", "a", "c", "b"]}
        r = client.put("/api/preferences/pinned-models", json=body)
        assert r.json()["model_ids"] == ["a", "b", "c"]


class TestRecentModelsEndpoint:
    def test_get_returns_empty_initially(self, client):
        r = client.get("/api/preferences/recent-models")
        assert r.status_code == 200
        assert r.json() == {"model_ids": []}

    def test_post_records_a_use(self, client):
        r = client.post(
            "/api/preferences/recent-models",
            json={"model_id": "anthropic/claude-sonnet-5"},
        )
        assert r.status_code == 200
        assert r.json()["model_ids"] == ["anthropic/claude-sonnet-5"]

    def test_post_moves_to_front_on_repeat(self, client):
        client.post("/api/preferences/recent-models", json={"model_id": "a"})
        client.post("/api/preferences/recent-models", json={"model_id": "b"})
        r = client.post("/api/preferences/recent-models", json={"model_id": "a"})
        assert r.json()["model_ids"] == ["a", "b"]

    def test_post_caps_at_recent_models_cap(self, client):
        from synthesis_engine.keystore import RECENT_MODELS_CAP

        for i in range(RECENT_MODELS_CAP + 5):
            client.post("/api/preferences/recent-models", json={"model_id": f"m-{i}"})
        r = client.get("/api/preferences/recent-models")
        ids = r.json()["model_ids"]
        assert len(ids) == RECENT_MODELS_CAP
        assert ids[0] == f"m-{RECENT_MODELS_CAP + 4}"  # newest first
