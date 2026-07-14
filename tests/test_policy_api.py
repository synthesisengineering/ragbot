"""Tests for the FastAPI cross-workspace policy router.

Exercises ``src/api/routers/policy.py`` end-to-end with the FastAPI
TestClient and a workspace-resolver hook so tests never touch the
operator's actual filesystem. Uses placeholder workspace names
(``acme-news``, ``acme-user``, ``beta-media``, ``air-gapped-ws``,
``client-conf-ws``) throughout — ragbot is a public repo.

Coverage:

  1. GET /api/policy/workspaces/{workspace} returns the policy for a
     well-formed routing.yaml.
  2. GET /api/policy/workspaces/{workspace} returns 404 when the
     workspace resolves to nothing.
  3. GET /api/policy/workspaces/{workspace} returns 400 when the
     routing.yaml is malformed.
  4. GET /api/policy/cross-workspace-check denies AIR_GAPPED mixed
     with any other tier.
  5. GET /api/policy/cross-workspace-check allows PUBLIC + PERSONAL,
     records requires_audit=false.
  6. GET /api/policy/cross-workspace-check allows PERSONAL +
     CLIENT_CONFIDENTIAL with requires_audit=true.
  7. GET /api/policy/cross-workspace-check denies CLIENT_CONFIDENTIAL
     + PUBLIC.
  8. GET /api/policy/cross-workspace-check returns 400 on empty
     workspaces query.
  9. GET /api/policy/cross-workspace-check returns 404 when any
     workspace is unresolved.
 10. GET /api/policy/cross-workspace-check with requested_model
     surfaces per-workspace verdicts.
 11. GET /api/policy/audit/recent returns empty when the log is missing.
 12. GET /api/policy/audit/recent returns entries with limit respected.
 13. GET /api/policy/example-routing-yaml returns the embedded schema.
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path
from typing import Dict, Optional

import pytest

# Make ``src/`` importable just like the other test modules do.
_REPO_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src",
)
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from synthesis_engine.policy import (  # noqa: E402
    AuditEntry,
    record as record_audit,
)
from synthesis_engine.policy.audit import (  # noqa: E402
    AUDIT_LOG_ENV,
    _reset_regex_cache,
)
from synthesis_engine.policy.routing import _clear_warning_cache  # noqa: E402

from api.routers import policy as policy_router  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_routing_yaml(root: Path, body: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "routing.yaml").write_text(textwrap.dedent(body))


def _install_resolver(
    workspace_roots: Dict[str, Optional[Path]],
) -> None:
    """Install a workspace resolver against the per-test root map."""

    def _resolver(name: str) -> Optional[str]:
        root = workspace_roots.get(name)
        if root is None:
            return None
        return str(root)

    policy_router.set_default_workspace_resolver(_resolver)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_policy_state(monkeypatch, tmp_path):
    """Reset the module-level caches across tests."""

    _clear_warning_cache()
    _reset_regex_cache()
    # Point the audit log at a per-test file via the env override.
    monkeypatch.setenv(AUDIT_LOG_ENV, str(tmp_path / "audit.jsonl"))
    policy_router.reset_workspace_resolver()
    yield
    _clear_warning_cache()
    _reset_regex_cache()
    policy_router.reset_workspace_resolver()


@pytest.fixture
def client():
    """FastAPI TestClient bound to the policy router."""

    app = FastAPI()
    app.include_router(policy_router.router)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def workspaces(tmp_path):
    """Build five workspace roots covering each confidentiality tier."""

    roots: Dict[str, Path] = {
        "acme-news": tmp_path / "acme-news",  # PUBLIC
        "acme-user": tmp_path / "acme-user",  # PERSONAL
        "beta-media": tmp_path / "beta-media",  # PERSONAL
        "client-conf-ws": tmp_path / "client-conf-ws",  # CLIENT_CONFIDENTIAL
        "air-gapped-ws": tmp_path / "air-gapped-ws",  # AIR_GAPPED
    }

    _write_routing_yaml(
        roots["acme-news"],
        """
        confidentiality: public
        fallback_behavior: warn
        """,
    )
    _write_routing_yaml(
        roots["acme-user"],
        """
        confidentiality: personal
        fallback_behavior: warn
        """,
    )
    _write_routing_yaml(
        roots["beta-media"],
        """
        confidentiality: personal
        fallback_behavior: downgrade_to_local
        allowed_models:
          - gemma/*
          - anthropic/claude-*
        """,
    )
    _write_routing_yaml(
        roots["client-conf-ws"],
        """
        confidentiality: client_confidential
        fallback_behavior: deny
        allowed_models:
          - anthropic/claude-*
        """,
    )
    _write_routing_yaml(
        roots["air-gapped-ws"],
        """
        confidentiality: air_gapped
        local_only: true
        fallback_behavior: deny
        """,
    )

    _install_resolver({name: root for name, root in roots.items()})
    return roots


# ---------------------------------------------------------------------------
# /workspaces/{workspace}
# ---------------------------------------------------------------------------


class TestGetWorkspacePolicy:
    def test_well_formed_policy_returns_200(self, client, workspaces):
        resp = client.get("/api/policy/workspaces/client-conf-ws")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["workspace"] == "client-conf-ws"
        assert body["confidentiality"] == "CLIENT_CONFIDENTIAL"
        assert body["fallback_behavior"] == "deny"
        assert body["allowed_models"] == ["anthropic/claude-*"]
        assert body["routing_yaml_exists"] is True

    def test_unknown_workspace_returns_404(self, client, workspaces):
        resp = client.get("/api/policy/workspaces/never-heard-of-it")
        assert resp.status_code == 404
        assert "never-heard-of-it" in resp.text

    def test_malformed_routing_yaml_returns_400(
        self, client, workspaces, tmp_path
    ):
        bad_root = tmp_path / "bad-ws"
        bad_root.mkdir(parents=True, exist_ok=True)
        # ``local_only`` must be a boolean per the schema; a string is
        # rejected by load_routing_policy.
        (bad_root / "routing.yaml").write_text(
            "confidentiality: public\nlocal_only: 'maybe'\n"
        )
        # Install a resolver that knows about both the good and the bad
        # workspace.
        def _resolver(name: str) -> Optional[str]:
            if name == "bad-ws":
                return str(bad_root)
            root = workspaces.get(name)
            return str(root) if root else None

        policy_router.set_default_workspace_resolver(_resolver)

        resp = client.get("/api/policy/workspaces/bad-ws")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /cross-workspace-check
# ---------------------------------------------------------------------------


class TestCrossWorkspaceCheck:
    def test_air_gapped_mix_is_denied(self, client, workspaces):
        resp = client.get(
            "/api/policy/cross-workspace-check",
            params={"workspaces": "air-gapped-ws,acme-news"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["allowed"] is False
        # AIR_GAPPED is the strictest tier — effective confidentiality
        # rises to it for any participating mix.
        assert body["effective_confidentiality"] == "AIR_GAPPED"
        assert "AIR_GAPPED" in body["reason"]

    def test_public_personal_mix_allowed_no_audit(self, client, workspaces):
        resp = client.get(
            "/api/policy/cross-workspace-check",
            params={"workspaces": "acme-news,acme-user"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["allowed"] is True
        assert body["effective_confidentiality"] == "PERSONAL"
        assert body["requires_audit"] is False

    def test_personal_client_confidential_requires_audit(
        self, client, workspaces
    ):
        resp = client.get(
            "/api/policy/cross-workspace-check",
            params={"workspaces": "acme-user,client-conf-ws"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["allowed"] is True
        assert body["effective_confidentiality"] == "CLIENT_CONFIDENTIAL"
        assert body["requires_audit"] is True

    def test_client_confidential_with_public_denied(self, client, workspaces):
        resp = client.get(
            "/api/policy/cross-workspace-check",
            params={"workspaces": "acme-news,client-conf-ws"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["allowed"] is False
        assert "CLIENT_CONFIDENTIAL" in body["reason"]
        # All boundaries surfaced.
        assert len(body["boundaries"]) == 1

    def test_empty_workspaces_returns_400(self, client, workspaces):
        resp = client.get(
            "/api/policy/cross-workspace-check",
            params={"workspaces": ",,,"},
        )
        assert resp.status_code == 400

    def test_unknown_workspace_returns_404(self, client, workspaces):
        resp = client.get(
            "/api/policy/cross-workspace-check",
            params={"workspaces": "acme-user,does-not-exist"},
        )
        assert resp.status_code == 404
        detail = resp.json().get("detail") or {}
        # FastAPI wraps the dict under ``detail``.
        if isinstance(detail, dict):
            assert "does-not-exist" in detail.get("workspaces", [])

    def test_requested_model_surfaces_per_workspace_verdicts(
        self, client, workspaces
    ):
        # ``client-conf-ws`` allows anthropic/claude-* only; an openai
        # model trips the deny rule there but acme-user has no allowlist
        # so the model passes.
        resp = client.get(
            "/api/policy/cross-workspace-check",
            params={
                "workspaces": "acme-user,client-conf-ws",
                "requested_model": "openai/gpt-5.6-terra",
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "model_routing" in body
        mr = body["model_routing"]
        assert mr["requested_model"] == "openai/gpt-5.6-terra"
        assert mr["aggregate_allowed"] is False
        assert mr["denying_workspace_count"] == 1
        verdicts = {v["workspace"]: v for v in mr["verdicts"]}
        assert verdicts["acme-user"]["allowed"] is True
        assert verdicts["client-conf-ws"]["allowed"] is False
        assert verdicts["client-conf-ws"]["fallback_behavior"] == "deny"


# ---------------------------------------------------------------------------
# /audit/recent
# ---------------------------------------------------------------------------


class TestAuditRecent:
    def test_missing_log_returns_empty(self, client):
        resp = client.get("/api/policy/audit/recent")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["entries"] == []
        assert body["count"] == 0

    def test_returns_entries_respecting_limit(self, client):
        # Seed five entries.
        for i in range(5):
            record_audit(
                AuditEntry.build(
                    op_type="cross_workspace_run_start",
                    workspaces=["acme-news", "acme-user"],
                    tools=[],
                    model_id="anthropic/claude-opus-4-8",
                    outcome="allowed",
                    metadata={"i": i},
                )
            )
        # No limit → all five.
        resp = client.get("/api/policy/audit/recent")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 5
        assert len(body["entries"]) == 5
        # File-natural order: oldest first.
        assert body["entries"][0]["metadata"]["i"] == 0
        assert body["entries"][-1]["metadata"]["i"] == 4

        # Limit honored.
        resp = client.get("/api/policy/audit/recent", params={"limit": 2})
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 2
        assert body["entries"][-1]["metadata"]["i"] == 4
        assert body["entries"][0]["metadata"]["i"] == 3


# ---------------------------------------------------------------------------
# /example-routing-yaml
# ---------------------------------------------------------------------------


def test_example_yaml_returns_embedded_schema(client):
    resp = client.get("/api/policy/example-routing-yaml")
    assert resp.status_code == 200
    body = resp.json()
    assert "example" in body
    assert "confidentiality" in body["example"]
    assert "fallback_behavior" in body["example"]
