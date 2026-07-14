"""Tests for synthesis_engine.policy — routing, confidentiality, audit."""

from __future__ import annotations

import json
import os
import textwrap
import threading
from pathlib import Path

import pytest

from synthesis_engine.agent.permissions import (
    PermissionRegistry,
    ToolCallContext,
)
from synthesis_engine.exceptions import ConfigurationError
from synthesis_engine.policy import (
    AllowanceCheck,
    AuditEntry,
    Confidentiality,
    ConfidentialityCheck,
    EXAMPLE_ROUTING_YAML,
    FallbackBehavior,
    RoutingPolicy,
    check_cross_workspace_op,
    cross_workspace_gate,
    is_model_allowed,
    load_routing_policy,
    read_recent,
    record,
    redact_args,
    register_cross_workspace_gate,
)
from synthesis_engine.policy.audit import (
    AUDIT_LOG_ENV,
    _reset_regex_cache,
)
from synthesis_engine.policy.confidentiality import (
    ACTIVE_WORKSPACES_METADATA_KEY,
    ROUTING_POLICIES_METADATA_KEY,
)
from synthesis_engine.policy.routing import _clear_warning_cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_routing_yaml(workspace_root: Path, body: str) -> None:
    workspace_root.mkdir(parents=True, exist_ok=True)
    (workspace_root / "routing.yaml").write_text(textwrap.dedent(body))


@pytest.fixture(autouse=True)
def _reset_module_state():
    """Reset cached state across tests to prevent cross-test pollution."""
    _clear_warning_cache()
    _reset_regex_cache()
    yield
    _clear_warning_cache()
    _reset_regex_cache()


# ---------------------------------------------------------------------------
# RoutingPolicy YAML round-trip
# ---------------------------------------------------------------------------


class TestRoutingPolicyLoad:
    def test_well_formed_yaml_round_trip(self, tmp_path):
        root = tmp_path / "acme-news"
        _write_routing_yaml(
            root,
            """
            confidentiality: client_confidential
            allowed_models:
              - anthropic/claude-*
              - openai/gpt-5*
            denied_models:
              - "*-preview*"
            local_only: false
            fallback_behavior: deny
            """,
        )
        policy = load_routing_policy(str(root))
        assert policy.confidentiality == Confidentiality.CLIENT_CONFIDENTIAL
        assert policy.allowed_models == ("anthropic/claude-*", "openai/gpt-5*")
        assert policy.denied_models == ("*-preview*",)
        assert policy.local_only is False
        assert policy.fallback_behavior == FallbackBehavior.DENY

    def test_missing_file_returns_default_policy(self, tmp_path, caplog):
        root = tmp_path / "acme-user"
        root.mkdir()
        with caplog.at_level("WARNING"):
            policy = load_routing_policy(str(root))
        assert policy.confidentiality == Confidentiality.PUBLIC
        assert policy.allowed_models == ()
        assert policy.denied_models == ()
        assert policy.local_only is False
        assert policy.fallback_behavior == FallbackBehavior.WARN
        # The missing-file warning fires.
        assert any("No routing.yaml" in r.message for r in caplog.records)

    def test_missing_file_warning_fires_only_once(self, tmp_path, caplog):
        root = tmp_path / "acme-user"
        root.mkdir()
        with caplog.at_level("WARNING"):
            load_routing_policy(str(root))
            load_routing_policy(str(root))
        warnings = [r for r in caplog.records if "No routing.yaml" in r.message]
        assert len(warnings) == 1

    def test_malformed_yaml_raises_configuration_error(self, tmp_path):
        root = tmp_path / "beta-media"
        root.mkdir()
        # Unclosed flow-style map — yaml.safe_load raises YAMLError on this.
        (root / "routing.yaml").write_text("confidentiality: {unterminated\n")
        with pytest.raises(ConfigurationError):
            load_routing_policy(str(root))

    def test_non_mapping_top_level_raises(self, tmp_path):
        root = tmp_path / "beta-media"
        root.mkdir()
        (root / "routing.yaml").write_text("- a\n- b\n")
        with pytest.raises(ConfigurationError):
            load_routing_policy(str(root))

    def test_non_bool_local_only_raises(self, tmp_path):
        root = tmp_path / "beta-media"
        _write_routing_yaml(
            root,
            """
            local_only: maybe
            """,
        )
        with pytest.raises(ConfigurationError):
            load_routing_policy(str(root))

    def test_non_string_glob_entry_raises(self, tmp_path):
        root = tmp_path / "beta-media"
        _write_routing_yaml(
            root,
            """
            allowed_models:
              - anthropic/claude-*
              - 42
            """,
        )
        with pytest.raises(ConfigurationError):
            load_routing_policy(str(root))

    def test_unknown_confidentiality_falls_back_to_air_gapped(self, tmp_path):
        root = tmp_path / "acme-user"
        _write_routing_yaml(
            root,
            """
            confidentiality: top-secret
            """,
        )
        policy = load_routing_policy(str(root))
        assert policy.confidentiality == Confidentiality.AIR_GAPPED

    def test_unknown_fallback_behavior_falls_back_to_deny(self, tmp_path):
        root = tmp_path / "acme-user"
        _write_routing_yaml(
            root,
            """
            fallback_behavior: shrug
            """,
        )
        policy = load_routing_policy(str(root))
        assert policy.fallback_behavior == FallbackBehavior.DENY

    def test_example_routing_yaml_is_parseable(self, tmp_path):
        root = tmp_path / "acme-news"
        root.mkdir()
        (root / "routing.yaml").write_text(EXAMPLE_ROUTING_YAML)
        # Should parse without raising.
        policy = load_routing_policy(str(root))
        assert isinstance(policy, RoutingPolicy)


# ---------------------------------------------------------------------------
# is_model_allowed
# ---------------------------------------------------------------------------


class TestIsModelAllowed:
    def test_denied_models_wins_over_allowed(self):
        policy = RoutingPolicy(
            allowed_models=("anthropic/claude-*",),
            denied_models=("anthropic/claude-opus-*-preview*",),
        )
        check = is_model_allowed(policy, "anthropic/claude-opus-4-8-preview")
        assert check.allowed is False
        assert "denied_models" in check.reason

    def test_allowed_models_honored(self):
        policy = RoutingPolicy(allowed_models=("openai/gpt-5*",))
        check = is_model_allowed(policy, "openai/gpt-5.6-sol")
        assert check.allowed is True
        assert "allowed_models" in check.reason

    def test_glob_matching_anthropic(self):
        policy = RoutingPolicy(allowed_models=("anthropic/claude-*",))
        for mid in (
            "anthropic/claude-sonnet-5",
            "anthropic/claude-opus-4-8",
            "anthropic/claude-haiku-4-5",
        ):
            check = is_model_allowed(policy, mid)
            assert check.allowed, mid

    def test_no_match_with_allowlist_denies(self):
        policy = RoutingPolicy(allowed_models=("anthropic/claude-*",))
        check = is_model_allowed(policy, "openai/gpt-5")
        assert check.allowed is False
        assert check.suggested_fallback is not None

    def test_empty_allowlist_allows_anything_not_denied(self):
        policy = RoutingPolicy()
        check = is_model_allowed(policy, "anthropic/claude-opus-4-8")
        assert check.allowed is True

    def test_local_only_denies_non_local(self):
        policy = RoutingPolicy(local_only=True)
        check = is_model_allowed(policy, "anthropic/claude-opus-4-8")
        assert check.allowed is False
        assert "local_only" in check.reason

    def test_local_only_heuristic_allows_local_families(self):
        policy = RoutingPolicy(local_only=True)
        for mid in (
            "gemma/gemma-3-27b",
            "qwen3/qwen-3-32b",
            "llama-3-70b-instruct",
            "deepseek/deepseek-v4",
            "anthropic/claude-4-7:local",
        ):
            check = is_model_allowed(policy, mid)
            assert check.allowed, mid

    def test_empty_model_id_is_denied(self):
        policy = RoutingPolicy()
        check = is_model_allowed(policy, "")
        assert check.allowed is False


# ---------------------------------------------------------------------------
# Confidentiality strictness ordering
# ---------------------------------------------------------------------------


class TestConfidentialityOrdering:
    def test_strictness_order(self):
        assert (
            Confidentiality.PUBLIC
            < Confidentiality.PERSONAL
            < Confidentiality.CLIENT_CONFIDENTIAL
            < Confidentiality.AIR_GAPPED
        )

    def test_max_yields_strictest(self):
        result = max(
            [
                Confidentiality.PUBLIC,
                Confidentiality.PERSONAL,
                Confidentiality.CLIENT_CONFIDENTIAL,
            ]
        )
        assert result == Confidentiality.CLIENT_CONFIDENTIAL

    def test_from_string_canonical_values(self):
        assert Confidentiality.from_string("public") == Confidentiality.PUBLIC
        assert Confidentiality.from_string("personal") == Confidentiality.PERSONAL
        assert (
            Confidentiality.from_string("client_confidential")
            == Confidentiality.CLIENT_CONFIDENTIAL
        )
        assert Confidentiality.from_string("air_gapped") == Confidentiality.AIR_GAPPED

    def test_from_string_handles_dash_variant(self):
        assert (
            Confidentiality.from_string("client-confidential")
            == Confidentiality.CLIENT_CONFIDENTIAL
        )


# ---------------------------------------------------------------------------
# Cross-workspace mix rules
# ---------------------------------------------------------------------------


def _policy(confidentiality: Confidentiality) -> RoutingPolicy:
    return RoutingPolicy(confidentiality=confidentiality)


class TestCrossWorkspaceMixRules:
    def test_air_gapped_with_anything_denied(self):
        policies = {
            "acme-secrets": _policy(Confidentiality.AIR_GAPPED),
            "acme-user": _policy(Confidentiality.PERSONAL),
        }
        check = check_cross_workspace_op(
            ["acme-secrets", "acme-user"], policies,
        )
        assert check.allowed is False
        assert "AIR_GAPPED" in check.reason

    def test_air_gapped_with_air_gapped_allowed(self):
        policies = {
            "acme-secrets": _policy(Confidentiality.AIR_GAPPED),
            "beta-secrets": _policy(Confidentiality.AIR_GAPPED),
        }
        check = check_cross_workspace_op(
            ["acme-secrets", "beta-secrets"], policies,
        )
        assert check.allowed is True
        assert check.effective_confidentiality == Confidentiality.AIR_GAPPED

    def test_client_confidential_with_public_denied(self):
        policies = {
            "acme-news": _policy(Confidentiality.CLIENT_CONFIDENTIAL),
            "beta-media": _policy(Confidentiality.PUBLIC),
        }
        check = check_cross_workspace_op(
            ["acme-news", "beta-media"], policies,
        )
        assert check.allowed is False
        assert "PUBLIC" in check.reason

    def test_personal_plus_client_confidential_allowed_with_audit(self):
        policies = {
            "acme-news": _policy(Confidentiality.CLIENT_CONFIDENTIAL),
            "acme-user": _policy(Confidentiality.PERSONAL),
        }
        check = check_cross_workspace_op(
            ["acme-news", "acme-user"], policies,
        )
        assert check.allowed is True
        assert check.requires_audit is True
        assert check.effective_confidentiality == Confidentiality.CLIENT_CONFIDENTIAL

    def test_public_plus_public_allowed_without_audit(self):
        policies = {
            "acme-public": _policy(Confidentiality.PUBLIC),
            "beta-public": _policy(Confidentiality.PUBLIC),
        }
        check = check_cross_workspace_op(
            ["acme-public", "beta-public"], policies,
        )
        assert check.allowed is True
        assert check.requires_audit is False

    def test_personal_plus_personal_allowed_without_audit(self):
        policies = {
            "acme-user": _policy(Confidentiality.PERSONAL),
            "beta-user": _policy(Confidentiality.PERSONAL),
        }
        check = check_cross_workspace_op(
            ["acme-user", "beta-user"], policies,
        )
        assert check.allowed is True
        assert check.requires_audit is False

    def test_single_workspace_short_circuits(self):
        policies = {
            "acme-user": _policy(Confidentiality.AIR_GAPPED),
        }
        check = check_cross_workspace_op(["acme-user"], policies)
        assert check.allowed is True
        assert check.boundaries == ()

    def test_missing_policy_fails_closed_to_air_gapped(self):
        policies = {
            "acme-news": _policy(Confidentiality.PUBLIC),
            # "beta-media" deliberately omitted.
        }
        check = check_cross_workspace_op(
            ["acme-news", "beta-media"], policies,
        )
        # With beta-media → AIR_GAPPED fallback, mix is denied.
        assert check.allowed is False

    def test_empty_active_returns_denied(self):
        check = check_cross_workspace_op([], {})
        assert check.allowed is False

    def test_three_way_mix_chooses_strictest(self):
        policies = {
            "p": _policy(Confidentiality.PUBLIC),
            "u": _policy(Confidentiality.PERSONAL),
            "c": _policy(Confidentiality.CLIENT_CONFIDENTIAL),
        }
        # p + c is denied (public + client_confidential).
        check = check_cross_workspace_op(["p", "u", "c"], policies)
        assert check.allowed is False


# ---------------------------------------------------------------------------
# cross_workspace_gate / PermissionRegistry integration
# ---------------------------------------------------------------------------


class TestCrossWorkspaceGate:
    def test_gate_allows_single_workspace(self):
        ctx = ToolCallContext(
            tool_name="memory.read",
            metadata={
                ACTIVE_WORKSPACES_METADATA_KEY: ["acme-user"],
                ROUTING_POLICIES_METADATA_KEY: {
                    "acme-user": _policy(Confidentiality.AIR_GAPPED),
                },
            },
        )
        verdict = cross_workspace_gate(ctx)
        assert verdict.allowed is True

    def test_gate_denies_air_gapped_mix(self):
        ctx = ToolCallContext(
            tool_name="memory.read",
            metadata={
                ACTIVE_WORKSPACES_METADATA_KEY: ["acme-secrets", "acme-user"],
                ROUTING_POLICIES_METADATA_KEY: {
                    "acme-secrets": _policy(Confidentiality.AIR_GAPPED),
                    "acme-user": _policy(Confidentiality.PERSONAL),
                },
            },
        )
        verdict = cross_workspace_gate(ctx)
        assert verdict.allowed is False
        assert "AIR_GAPPED" in verdict.reason

    def test_gate_denies_when_metadata_missing(self):
        ctx = ToolCallContext(tool_name="memory.read", metadata={})
        verdict = cross_workspace_gate(ctx)
        assert verdict.allowed is False
        assert "active_workspaces" in verdict.reason

    def test_registry_integration_allows_compatible_mix(self):
        registry = PermissionRegistry()
        register_cross_workspace_gate(registry, tool_name="memory.read")
        verdict = registry.check(
            "memory.read",
            context=ToolCallContext(
                tool_name="memory.read",
                metadata={
                    ACTIVE_WORKSPACES_METADATA_KEY: ["acme-news", "acme-user"],
                    ROUTING_POLICIES_METADATA_KEY: {
                        "acme-news": _policy(Confidentiality.CLIENT_CONFIDENTIAL),
                        "acme-user": _policy(Confidentiality.PERSONAL),
                    },
                },
            ),
        )
        assert verdict.allowed is True

    def test_gate_directly_surfaces_audit_required(self):
        """The audit-required signal is surfaced in the gate's own reason
        string; the PermissionRegistry collapses to its standard 'all gates
        allowed' message when every gate returns ALLOW. Callers that need
        the audit flag should consult :func:`check_cross_workspace_op`
        directly (which is what the agent loop will do)."""
        ctx = ToolCallContext(
            tool_name="memory.read",
            metadata={
                ACTIVE_WORKSPACES_METADATA_KEY: ["acme-news", "acme-user"],
                ROUTING_POLICIES_METADATA_KEY: {
                    "acme-news": _policy(Confidentiality.CLIENT_CONFIDENTIAL),
                    "acme-user": _policy(Confidentiality.PERSONAL),
                },
            },
        )
        verdict = cross_workspace_gate(ctx)
        assert verdict.allowed is True
        assert "audit required" in verdict.reason

    def test_registry_integration_denies_public_client_mix(self):
        registry = PermissionRegistry()
        register_cross_workspace_gate(registry, tool_name="memory.read")
        verdict = registry.check(
            "memory.read",
            context=ToolCallContext(
                tool_name="memory.read",
                metadata={
                    ACTIVE_WORKSPACES_METADATA_KEY: ["acme-news", "beta-media"],
                    ROUTING_POLICIES_METADATA_KEY: {
                        "acme-news": _policy(Confidentiality.CLIENT_CONFIDENTIAL),
                        "beta-media": _policy(Confidentiality.PUBLIC),
                    },
                },
            ),
        )
        assert verdict.allowed is False


# ---------------------------------------------------------------------------
# Audit log: append / read / rotation-safe / redaction
# ---------------------------------------------------------------------------


@pytest.fixture
def audit_path(tmp_path, monkeypatch):
    """Point the audit log at a tmp file via the env override."""
    path = tmp_path / "audit.jsonl"
    monkeypatch.setenv(AUDIT_LOG_ENV, str(path))
    return path


class TestAuditLog:
    def test_record_appends_one_line(self, audit_path):
        entry = AuditEntry.build(
            op_type="cross_workspace_synthesis",
            workspaces=["acme-news", "acme-user"],
            tools=["memory.read"],
            model_id="anthropic/claude-opus-4-8",
            outcome="allowed",
            args_summary="{}",
        )
        record(entry)
        assert audit_path.is_file()
        with open(audit_path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["op_type"] == "cross_workspace_synthesis"
        assert data["workspaces"] == ["acme-news", "acme-user"]

    def test_read_recent_returns_entries_in_order(self, audit_path):
        for i in range(5):
            record(
                AuditEntry.build(
                    op_type="tool_call",
                    workspaces=["acme-user"],
                    tools=[f"tool_{i}"],
                    model_id="local/gemma",
                    outcome="allowed",
                )
            )
        entries = read_recent(limit=10)
        assert len(entries) == 5
        # Natural file order: oldest first, newest last.
        assert entries[0].tools == ("tool_0",)
        assert entries[-1].tools == ("tool_4",)

    def test_read_recent_truncates_to_limit(self, audit_path):
        for i in range(20):
            record(
                AuditEntry.build(
                    op_type="tool_call",
                    workspaces=["w"],
                    tools=[f"tool_{i}"],
                )
            )
        entries = read_recent(limit=5)
        assert len(entries) == 5
        assert entries[-1].tools == ("tool_19",)
        assert entries[0].tools == ("tool_15",)

    def test_read_recent_missing_file_returns_empty(self, tmp_path, monkeypatch):
        nonexistent = tmp_path / "nope.jsonl"
        monkeypatch.setenv(AUDIT_LOG_ENV, str(nonexistent))
        assert read_recent() == []

    def test_read_recent_skips_corrupt_line(self, audit_path):
        record(
            AuditEntry.build(op_type="tool_call", workspaces=["w"], tools=["a"])
        )
        # Simulate a rotation glitch: append a corrupt line and another good
        # line.
        with open(audit_path, "a") as f:
            f.write("not valid json\n")
        record(
            AuditEntry.build(op_type="tool_call", workspaces=["w"], tools=["b"])
        )
        entries = read_recent(limit=10)
        # Corrupt middle line is skipped; bracket entries remain.
        assert [e.tools for e in entries] == [("a",), ("b",)]

    def test_record_thread_safe_under_concurrency(self, audit_path):
        n = 50
        threads = []

        def writer(i: int) -> None:
            record(
                AuditEntry.build(
                    op_type="tool_call",
                    workspaces=[f"w-{i}"],
                    tools=[f"tool_{i}"],
                )
            )

        for i in range(n):
            t = threading.Thread(target=writer, args=(i,))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

        # Every line should be intact JSON; no torn writes.
        with open(audit_path) as f:
            lines = f.readlines()
        assert len(lines) == n
        for ln in lines:
            json.loads(ln)  # raises on corruption

    def test_record_creates_parent_dirs(self, tmp_path, monkeypatch):
        nested = tmp_path / "a" / "b" / "audit.jsonl"
        monkeypatch.setenv(AUDIT_LOG_ENV, str(nested))
        record(AuditEntry.build(op_type="tool_call", workspaces=["w"]))
        assert nested.is_file()

    def test_env_override_honored(self, tmp_path, monkeypatch):
        override = tmp_path / "override.jsonl"
        monkeypatch.setenv(AUDIT_LOG_ENV, str(override))
        record(AuditEntry.build(op_type="tool_call", workspaces=["w"]))
        assert override.is_file()
        # Without override, the default home path would not exist on this
        # test machine — the override path proves the env var is used.

    def test_explicit_log_path_overrides_env(self, tmp_path, monkeypatch):
        env_path = tmp_path / "env.jsonl"
        arg_path = tmp_path / "arg.jsonl"
        monkeypatch.setenv(AUDIT_LOG_ENV, str(env_path))
        record(
            AuditEntry.build(op_type="tool_call", workspaces=["w"]),
            log_path=arg_path,
        )
        assert arg_path.is_file()
        assert not env_path.is_file()


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


class TestRedaction:
    def test_redacts_long_value(self):
        long_value = "x" * 250
        rendered = redact_args({"prompt": long_value})
        assert "<redacted>" in rendered
        assert long_value not in rendered

    def test_redacts_anthropic_key(self):
        rendered = redact_args(
            {"auth": "sk-ant-api-abcdef0123456789abcdef0123456789"}
        )
        assert "<redacted>" in rendered
        assert "sk-ant-api" not in rendered

    def test_redacts_openai_key(self):
        # OpenAI key pattern: sk-<20 alnum>T3BlbkFJ
        key = "sk-" + "a" * 20 + "T3BlbkFJ" + "xyz"
        rendered = redact_args({"token": key})
        assert "<redacted>" in rendered

    def test_does_not_redact_safe_string(self):
        rendered = redact_args({"query": "what's the weather"})
        assert "what's the weather" in rendered
        assert "<redacted>" not in rendered

    def test_redacts_nested_dict(self):
        rendered = redact_args(
            {
                "request": {
                    "headers": {
                        "Authorization": "Bearer sk-ant-api-zzzzzzzzzzzzzzzz",
                    },
                    "body": "ok",
                },
            }
        )
        assert "<redacted>" in rendered
        assert "sk-ant-api" not in rendered
        assert "ok" in rendered

    def test_redacts_list_values(self):
        rendered = redact_args(
            {"keys": ["sk-ant-api-zzzzzzzzzzzz", "fine-value"]}
        )
        assert "<redacted>" in rendered
        assert "fine-value" in rendered

    def test_empty_args_renders_empty_dict(self):
        assert redact_args(None) == "{}"
        assert redact_args({}) == "{}"

    def test_redaction_falls_back_when_git_hook_config_missing(
        self, tmp_path, monkeypatch
    ):
        """When the hook config is missing, the fallback regex set kicks in."""
        # Point HOME at an empty dir so git-hook-config.yaml is absent.
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
        _reset_regex_cache()
        # The fallback set still catches Anthropic keys.
        rendered = redact_args({"x": "sk-ant-api-aaaaaaaaaaaaaa"})
        assert "<redacted>" in rendered


# ---------------------------------------------------------------------------
# AuditEntry round-trip
# ---------------------------------------------------------------------------


class TestAuditEntryShape:
    def test_to_dict_from_dict_round_trip(self):
        entry = AuditEntry.build(
            op_type="model_call",
            workspaces=["acme-news"],
            tools=["llm.complete"],
            model_id="anthropic/claude-opus-4-8",
            outcome="denied",
            args_summary='{"prompt": "<redacted>"}',
            metadata={"effective_confidentiality": "CLIENT_CONFIDENTIAL"},
        )
        round_tripped = AuditEntry.from_dict(entry.to_dict())
        assert round_tripped == entry
