"""Tests for the AgentLoop's routing-enforced multi-workspace path.

Exercises:

* Cross-workspace boundary refusal at the start of a run (AIR_GAPPED +
  anything non-AIR_GAPPED).
* PUBLIC + PERSONAL mix succeeds with an audit start + terminal entry.
* PERSONAL + CLIENT_CONFIDENTIAL mix succeeds with requires_audit=True.
* Per-policy fallback behavior for model calls:
    - DENY: ModelDeniedError surfaces, plan step fails, audit records denial.
    - DOWNGRADE_TO_LOCAL: resolved to a local model; audit records downgrade.
    - WARN: proceeds with a warning; audit records ``allowed`` with a
      flag set.
* Single-workspace regression: ``workspaces=None`` preserves the
  existing behaviour exactly (no new metadata, no audit entries).
* Single-workspace regression: ``workspaces=["one"]`` is allowed and
  drives to DONE without writing audit entries that the legacy path
  would not have written, except the routing-enforced start/terminal
  pair (which is the cost of opting in).

Uses placeholder workspace names (``acme-news``, ``acme-user``,
``beta-media``, ``client-conf-ws``, ``air-gapped-ws``, ``acme-public``)
throughout — ragbot is a public repo.

Fake substrates are used everywhere; no real LLM or MCP traffic.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest

# Make ``src/`` importable just like the other test modules do.
_REPO_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src",
)
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)


from synthesis_engine.agent import (  # noqa: E402
    AgentLoop,
    AgentState,
    FilesystemCheckpointStore,
    PermissionRegistry,
    PermissionResult,
)
from synthesis_engine.llm import (  # noqa: E402
    ModelDeniedError,
    get_routed_llm_backend,
)
from synthesis_engine.policy import (  # noqa: E402
    Confidentiality,
    FallbackBehavior,
    RoutingPolicy,
    read_recent,
)
from synthesis_engine.policy.audit import (  # noqa: E402
    AUDIT_LOG_ENV,
    _reset_regex_cache,
)
from synthesis_engine.policy.routing import _clear_warning_cache  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes (mirrored from test_agent_loop / test_agent_api)
# ---------------------------------------------------------------------------


@dataclass
class FakeLLMResponse:
    text: str
    model: str = "fake-model"
    backend: str = "fake"
    finish_reason: Optional[str] = "stop"
    usage: Dict[str, int] = field(default_factory=dict)


class FakeLLMBackend:
    def __init__(
        self,
        scripted: Optional[List[str]] = None,
        *,
        by_marker: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self._scripted = list(scripted or [])
        self._by_marker = {k: list(v) for k, v in (by_marker or {}).items()}
        self.calls: List[Any] = []

    def complete(self, request: Any) -> FakeLLMResponse:
        self.calls.append(request)
        user_text = _user_text(request)
        for marker, responses in self._by_marker.items():
            if marker in user_text and responses:
                return FakeLLMResponse(
                    text=responses.pop(0),
                    model=_request_model(request),
                )
        if self._scripted:
            return FakeLLMResponse(
                text=self._scripted.pop(0),
                model=_request_model(request),
            )
        raise AssertionError(
            f"FakeLLMBackend: no scripted response (user_text="
            f"{user_text[:120]!r})"
        )


def _request_model(request: Any) -> str:
    if isinstance(request, dict):
        return str(request.get("model", "fake-model"))
    return getattr(request, "model", "fake-model")


def _user_text(request: Any) -> str:
    if isinstance(request, dict):
        msgs = request.get("messages") or []
    else:
        msgs = getattr(request, "messages", []) or []
    parts: List[str] = []
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "user":
            parts.append(str(m.get("content", "")))
    return "\n".join(parts)


class FakeMCPClient:
    def __init__(
        self,
        *,
        tools: Optional[List[str]] = None,
        responses: Optional[Dict[str, Callable[[Dict[str, Any]], Any]]] = None,
    ) -> None:
        self._tools = tools or []
        self._responses = responses or {}
        self.call_log: List[Dict[str, Any]] = []

    async def list_tools(self, server_id: str) -> List[Any]:
        return [{"name": n, "description": f"fake tool {n}"} for n in self._tools]

    async def call_tool(
        self,
        server_id: str,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Any:
        self.call_log.append(
            {"server": server_id, "name": name, "arguments": arguments}
        )
        handler = self._responses.get(name)
        if handler is None:
            return {"text": f"{name} ok", "args": arguments}
        return handler(arguments or {})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_routing_yaml(root: Path, body: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "routing.yaml").write_text(textwrap.dedent(body))


def _trivial_plan_json(target: str = "summarise") -> str:
    return json.dumps(
        {
            "steps": [
                {
                    "step_id": "s1",
                    "action_type": "TOOL_CALL",
                    "target": target,
                    "inputs": {"text": "hello"},
                    "description": f"Call {target}.",
                }
            ]
        }
    )


def _llm_plan_json(model_target: str = "openai/gpt-5.6-terra") -> str:
    return json.dumps(
        {
            "steps": [
                {
                    "step_id": "s1",
                    "action_type": "LLM_CALL",
                    "target": model_target,
                    "inputs": {"prompt": "say hello"},
                    "description": "LLM call gated by routing policy.",
                }
            ]
        }
    )


def _permissive_registry() -> PermissionRegistry:
    reg = PermissionRegistry()
    reg.register("*", lambda _ctx: PermissionResult.allow("test-permissive"))
    return reg


def _build_loop(
    *,
    tmp_path: Path,
    llm_backend: Optional[FakeLLMBackend] = None,
    mcp_responses: Optional[Dict[str, Callable[[Dict[str, Any]], Any]]] = None,
    registry: Optional[PermissionRegistry] = None,
) -> AgentLoop:
    return AgentLoop(
        llm_backend=llm_backend or FakeLLMBackend(),
        mcp_client=FakeMCPClient(
            tools=["summarise"], responses=mcp_responses or {},
        ),
        permission_registry=registry or _permissive_registry(),
        checkpoint_store=FilesystemCheckpointStore(
            base_dir=tmp_path / "checkpoints"
        ),
        default_mcp_server="local",
    )


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch, tmp_path):
    _clear_warning_cache()
    _reset_regex_cache()
    monkeypatch.setenv(AUDIT_LOG_ENV, str(tmp_path / "audit.jsonl"))
    yield
    _clear_warning_cache()
    _reset_regex_cache()


@pytest.fixture
def workspace_roots(tmp_path):
    """Build the standard placeholder workspace tree."""

    roots: Dict[str, Path] = {
        "acme-news": tmp_path / "ws" / "acme-news",
        "acme-user": tmp_path / "ws" / "acme-user",
        "acme-public": tmp_path / "ws" / "acme-public",
        "beta-media": tmp_path / "ws" / "beta-media",
        "client-conf-ws": tmp_path / "ws" / "client-conf-ws",
        "client-conf-warn": tmp_path / "ws" / "client-conf-warn",
        "client-conf-downgrade": tmp_path / "ws" / "client-conf-downgrade",
        "air-gapped-ws": tmp_path / "ws" / "air-gapped-ws",
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
        roots["acme-public"],
        """
        confidentiality: public
        fallback_behavior: warn
        """,
    )
    _write_routing_yaml(
        roots["beta-media"],
        """
        confidentiality: personal
        fallback_behavior: warn
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
        roots["client-conf-warn"],
        """
        confidentiality: client_confidential
        fallback_behavior: warn
        allowed_models:
          - anthropic/claude-*
        """,
    )
    _write_routing_yaml(
        roots["client-conf-downgrade"],
        """
        confidentiality: client_confidential
        fallback_behavior: downgrade_to_local
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
    return {name: str(path) for name, path in roots.items()}


# ---------------------------------------------------------------------------
# Routing-aware LLM backend (deliverable 2)
# ---------------------------------------------------------------------------


class TestGetRoutedLLMBackend:
    """Exercise ``get_routed_llm_backend`` directly so we know the gate
    behaves correctly without driving the full agent loop."""

    def _policy(
        self,
        confidentiality: Confidentiality,
        *,
        fallback: FallbackBehavior,
        allowed: tuple = (),
        denied: tuple = (),
        local_only: bool = False,
    ) -> RoutingPolicy:
        return RoutingPolicy(
            confidentiality=confidentiality,
            allowed_models=allowed,
            denied_models=denied,
            local_only=local_only,
            fallback_behavior=fallback,
        )

    def test_all_policies_allow_returns_requested(self):
        policies = {
            "acme-news": self._policy(
                Confidentiality.PUBLIC, fallback=FallbackBehavior.WARN,
            ),
            "acme-user": self._policy(
                Confidentiality.PERSONAL, fallback=FallbackBehavior.WARN,
            ),
        }
        backend = object()
        result_backend, model = get_routed_llm_backend(
            policies, "anthropic/claude-opus-4-8", backend=backend,
        )
        assert result_backend is backend
        assert model == "anthropic/claude-opus-4-8"

    def test_deny_raises_model_denied_error(self):
        policies = {
            "client-conf-ws": self._policy(
                Confidentiality.CLIENT_CONFIDENTIAL,
                fallback=FallbackBehavior.DENY,
                allowed=("anthropic/claude-*",),
            ),
        }
        with pytest.raises(ModelDeniedError) as exc:
            get_routed_llm_backend(
                policies, "openai/gpt-5.6-terra", backend=object(),
            )
        assert exc.value.denying_workspace == "client-conf-ws"
        assert "openai/gpt-5.6-terra" in exc.value.requested_model

    def test_downgrade_to_local_resolves_local_model(self):
        policies = {
            "client-conf-downgrade": self._policy(
                Confidentiality.CLIENT_CONFIDENTIAL,
                fallback=FallbackBehavior.DOWNGRADE_TO_LOCAL,
                allowed=("anthropic/claude-*", "gemma/gemma-4-27b"),
            ),
        }
        _, model = get_routed_llm_backend(
            policies, "openai/gpt-5.6-terra", backend=object(),
        )
        # Must resolve to a model the workspace allows; the allowlist
        # contains gemma so that's the natural pick.
        assert model == "gemma/gemma-4-27b"

    def test_warn_proceeds_with_requested_model(self, caplog):
        policies = {
            "client-conf-warn": self._policy(
                Confidentiality.CLIENT_CONFIDENTIAL,
                fallback=FallbackBehavior.WARN,
                allowed=("anthropic/claude-*",),
            ),
        }
        with caplog.at_level("WARNING"):
            _, model = get_routed_llm_backend(
                policies, "openai/gpt-5.6-terra", backend=object(),
            )
        assert model == "openai/gpt-5.6-terra"
        # The denial reason was logged.
        assert any(
            "openai/gpt-5.6-terra" in rec.getMessage() for rec in caplog.records
        )

    def test_strictest_denier_wins_over_first_denier(self):
        # Two denying workspaces: PERSONAL (WARN) and CLIENT_CONFIDENTIAL
        # (DENY). The DENY policy must win because it is strictest.
        policies = {
            "acme-user": self._policy(
                Confidentiality.PERSONAL,
                fallback=FallbackBehavior.WARN,
                denied=("openai/*",),
            ),
            "client-conf-ws": self._policy(
                Confidentiality.CLIENT_CONFIDENTIAL,
                fallback=FallbackBehavior.DENY,
                denied=("openai/*",),
            ),
        }
        with pytest.raises(ModelDeniedError) as exc:
            get_routed_llm_backend(
                policies, "openai/gpt-5.6-terra", backend=object(),
            )
        assert exc.value.denying_workspace == "client-conf-ws"


# ---------------------------------------------------------------------------
# AgentLoop.run() — multi-workspace path
# ---------------------------------------------------------------------------


class TestAgentLoopMultiWorkspace:
    @pytest.mark.asyncio
    async def test_public_personal_mix_succeeds_with_audit(
        self, tmp_path, workspace_roots
    ):
        loop = _build_loop(
            tmp_path=tmp_path,
            llm_backend=FakeLLMBackend(scripted=[_trivial_plan_json()]),
            mcp_responses={"summarise": lambda _a: {"text": "ok"}},
        )

        final = await loop.run(
            "summarise something",
            workspaces=["acme-news", "acme-user"],
            workspace_roots=workspace_roots,
        )

        assert final.current_state == AgentState.DONE
        assert final.metadata["routing_enforced"] is True
        assert final.metadata["active_workspaces"] == ["acme-news", "acme-user"]
        assert final.metadata["effective_confidentiality"] == "PERSONAL"
        # Start + terminal audit entries recorded.
        entries = read_recent(limit=10)
        op_types = [e.op_type for e in entries]
        assert "cross_workspace_run_start" in op_types
        assert "cross_workspace_run_terminal" in op_types

    @pytest.mark.asyncio
    async def test_air_gapped_mix_transitions_to_error(
        self, tmp_path, workspace_roots
    ):
        loop = _build_loop(tmp_path=tmp_path)

        final = await loop.run(
            "denied op",
            workspaces=["air-gapped-ws", "acme-public"],
            workspace_roots=workspace_roots,
        )
        assert final.current_state == AgentState.ERROR
        assert final.final_answer and "AIR_GAPPED" in final.final_answer
        # Audit log records the denial.
        entries = read_recent(limit=10)
        outcomes = [e.outcome for e in entries]
        assert "denied" in outcomes

    @pytest.mark.asyncio
    async def test_client_confidential_deny_surfaces_model_denied_error(
        self, tmp_path, workspace_roots
    ):
        # The plan asks for an LLM_CALL targeting an openai model, but
        # client-conf-ws's policy only allows anthropic/claude-* and its
        # fallback is DENY. The step fails; after MAX_REPLANS the loop
        # transitions to ERROR.
        plan = _llm_plan_json("openai/gpt-5.6-terra")
        loop = _build_loop(
            tmp_path=tmp_path,
            llm_backend=FakeLLMBackend(
                scripted=[plan, plan, plan, plan, plan]
            ),
        )

        final = await loop.run(
            "use a frontier model",
            workspaces=["client-conf-ws"],
            workspace_roots=workspace_roots,
        )

        # The model call denial cascades through retries + replans to ERROR.
        assert final.current_state == AgentState.ERROR
        # Audit log records the denial.
        entries = read_recent(limit=50)
        denied_model_calls = [
            e for e in entries
            if e.op_type == "model_call" and e.outcome == "denied"
        ]
        assert len(denied_model_calls) >= 1
        assert any(
            e.metadata.get("denying_workspace") == "client-conf-ws"
            for e in denied_model_calls
        )

    @pytest.mark.asyncio
    async def test_client_confidential_downgrade_resolves_local(
        self, tmp_path, workspace_roots
    ):
        plan = _llm_plan_json("openai/gpt-5.6-terra")
        # The plan parses the planner response; the executed LLM_CALL
        # step then calls back into the same fake — provide a follow-up
        # response so the step's complete() succeeds.
        loop = _build_loop(
            tmp_path=tmp_path,
            llm_backend=FakeLLMBackend(
                scripted=[plan, "model response after downgrade"],
            ),
        )

        final = await loop.run(
            "use a model",
            workspaces=["client-conf-downgrade"],
            workspace_roots=workspace_roots,
        )

        # The LLM call still ran (against a local model). DONE.
        assert final.current_state == AgentState.DONE
        # Audit log records the downgrade.
        entries = read_recent(limit=50)
        downgrades = [
            e for e in entries
            if e.op_type == "model_call" and e.outcome == "downgraded"
        ]
        assert len(downgrades) == 1
        assert downgrades[0].metadata["resolved_model"] != "openai/gpt-5.6-terra"

    @pytest.mark.asyncio
    async def test_client_confidential_warn_proceeds(
        self, tmp_path, workspace_roots, caplog
    ):
        plan = _llm_plan_json("openai/gpt-5.6-terra")
        loop = _build_loop(
            tmp_path=tmp_path,
            llm_backend=FakeLLMBackend(
                scripted=[plan, "model response with warning"],
            ),
        )

        with caplog.at_level("WARNING"):
            final = await loop.run(
                "use a model",
                workspaces=["client-conf-warn"],
                workspace_roots=workspace_roots,
            )

        assert final.current_state == AgentState.DONE
        # Audit log records an ``allowed`` outcome — the routing-aware
        # path proceeded with the requested model.
        entries = read_recent(limit=50)
        allowed_model_calls = [
            e for e in entries
            if e.op_type == "model_call" and e.outcome == "allowed"
        ]
        assert any(
            e.metadata.get("resolved_model") == "openai/gpt-5.6-terra"
            for e in allowed_model_calls
        )

    @pytest.mark.asyncio
    async def test_personal_client_confidential_requires_audit(
        self, tmp_path, workspace_roots
    ):
        loop = _build_loop(
            tmp_path=tmp_path,
            llm_backend=FakeLLMBackend(scripted=[_trivial_plan_json()]),
            mcp_responses={"summarise": lambda _a: {"text": "ok"}},
        )

        final = await loop.run(
            "join personal + client",
            workspaces=["acme-user", "client-conf-ws"],
            workspace_roots=workspace_roots,
        )

        assert final.current_state == AgentState.DONE
        check = final.metadata["cross_workspace_check"]
        assert check["allowed"] is True
        assert check["requires_audit"] is True
        assert check["effective_confidentiality"] == "CLIENT_CONFIDENTIAL"


# ---------------------------------------------------------------------------
# Single-workspace regression
# ---------------------------------------------------------------------------


class TestSingleWorkspaceRegression:
    """Confirm ``workspaces=None`` / single-entry preserves legacy behaviour."""

    @pytest.mark.asyncio
    async def test_workspaces_none_preserves_existing_behaviour(
        self, tmp_path, workspace_roots
    ):
        loop = _build_loop(
            tmp_path=tmp_path,
            llm_backend=FakeLLMBackend(scripted=[_trivial_plan_json()]),
            mcp_responses={"summarise": lambda _a: {"text": "ok"}},
        )

        final = await loop.run("summarise")  # no workspaces kwarg

        assert final.current_state == AgentState.DONE
        # No routing metadata recorded; this run is the legacy path.
        assert "routing_enforced" not in final.metadata
        assert "active_workspaces" not in final.metadata
        # No audit entries were written for this run.
        entries = read_recent(limit=10)
        cross_workspace_entries = [
            e for e in entries
            if e.op_type.startswith("cross_workspace_")
        ]
        assert cross_workspace_entries == []

    @pytest.mark.asyncio
    async def test_single_workspace_list_does_not_break_existing_behaviour(
        self, tmp_path, workspace_roots
    ):
        loop = _build_loop(
            tmp_path=tmp_path,
            llm_backend=FakeLLMBackend(scripted=[_trivial_plan_json()]),
            mcp_responses={"summarise": lambda _a: {"text": "ok"}},
        )

        final = await loop.run(
            "summarise one workspace",
            workspaces=["acme-news"],
            workspace_roots=workspace_roots,
        )

        # Single-workspace runs may opt into the routing-enforced path
        # by passing the workspace; the run still reaches DONE.
        assert final.current_state == AgentState.DONE
        # No cross-workspace boundary evaluation for a single workspace.
        check = final.metadata["cross_workspace_check"]
        assert check["allowed"] is True
        # The boundaries list is empty for a single-workspace op.
        assert check["boundaries"] == []

    @pytest.mark.asyncio
    async def test_rubric_path_still_works_without_workspaces(
        self, tmp_path, workspace_roots
    ):
        # Sanity: the rubric handling we preserved doesn't blow up when
        # ``workspaces`` is omitted; this is the canonical single-workspace
        # smoke test.
        loop = _build_loop(
            tmp_path=tmp_path,
            llm_backend=FakeLLMBackend(scripted=[_trivial_plan_json()]),
            mcp_responses={"summarise": lambda _a: {"text": "ok"}},
        )

        final = await loop.run("summarise it")
        assert final.current_state == AgentState.DONE
        assert "routing_enforced" not in final.metadata
