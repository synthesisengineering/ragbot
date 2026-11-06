"""Tests for synthesis_engine.agent — the agent-loop CORE.

The agent loop is a hand-rolled FSM (no LangGraph, no CrewAI). The tests
exercise:

  - State dataclass round-trip integrity (to_dict / from_dict)
  - The full state machine (INIT -> PLAN -> EXECUTE -> EVALUATE -> DONE)
  - Replanning on a failed step
  - Termination on max_iterations
  - The permission gate registry (DENY blocks, default ALLOW patterns)
  - Checkpoint serialisation round-trip
  - Replay equivalence from a checkpoint mid-run
  - Plan-structure validation (rejects malformed LLM JSON)
  - Instrumentation: spans land in the in-memory tracer
  - Memory-retriever wiring
  - Tool-output coercion of MCP CallToolResult shapes
  - $ref input resolution between steps

Fake substrates are used throughout so no real LLM or MCP I/O happens.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
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
    ActionType,
    AgentLoop,
    AgentState,
    ContextBlock,
    FilesystemCheckpointStore,
    GraphState,
    PermissionRegistry,
    PermissionResult,
    PlanStep,
    PlanValidationError,
    StepStatus,
    ToolCallContext,
)
from synthesis_engine.agent.checkpoints import InMemoryCheckpointStore  # noqa: E402
from synthesis_engine.agent.permissions import default_gate  # noqa: E402
from synthesis_engine.agent.planner import _parse_plan  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class FakeLLMResponse:
    """Minimal stand-in for ``synthesis_engine.llm.LLMResponse``."""

    text: str
    model: str = "fake-model"
    backend: str = "fake"
    finish_reason: Optional[str] = "stop"
    usage: Dict[str, int] = field(default_factory=dict)


class FakeLLMBackend:
    """LLM backend that returns pre-canned responses keyed by call signature.

    The signature for each call is a tuple of (model, role_summary,
    nth-call-for-this-key). The first matching key wins; an unmatched
    request raises so tests fail loudly rather than silently fall
    through to a default response.
    """

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
        # Decide which response to use.
        if self._scripted:
            text = self._scripted.pop(0)
            return FakeLLMResponse(text=text, model=_request_model(request))
        # Per-marker scripts: match on substrings present in the user
        # message (REPLAN / PLAN etc.).
        user_text = _user_text(request)
        for marker, responses in self._by_marker.items():
            if marker in user_text and responses:
                return FakeLLMResponse(
                    text=responses.pop(0), model=_request_model(request),
                )
        raise AssertionError(
            f"FakeLLMBackend: no scripted response available for request "
            f"(user_text={user_text!r}, marker_keys={list(self._by_marker)})"
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
    parts = []
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "user":
            parts.append(str(m.get("content", "")))
    return "\n".join(parts)


class FakeMCPClient:
    """In-memory MCP-like client that returns canned tool responses."""

    def __init__(
        self,
        *,
        tools: Optional[List[str]] = None,
        responses: Optional[Dict[str, Callable[[Dict[str, Any]], Any]]] = None,
        failure_count: int = 0,
    ) -> None:
        self._tools = tools or []
        self._responses = responses or {}
        self._failure_count = failure_count
        self.call_log: List[Dict[str, Any]] = []

    async def list_tools(self, server_id: str) -> List[Any]:
        return [{"name": name, "description": f"fake tool {name}"} for name in self._tools]

    async def call_tool(
        self,
        server_id: str,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Any:
        self.call_log.append(
            {"server": server_id, "name": name, "arguments": arguments}
        )
        if self._failure_count > 0:
            self._failure_count -= 1
            raise RuntimeError(f"Tool {name} failed (scripted).")
        handler = self._responses.get(name)
        if handler is None:
            return {"text": f"{name} ok", "args": arguments}
        result = handler(arguments or {})
        return result


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fs_checkpoint_store(tmp_path):
    """Filesystem checkpoint store rooted in a tmp directory."""
    return FilesystemCheckpointStore(base_dir=tmp_path / "checkpoints")


@pytest.fixture
def mem_checkpoint_store():
    return InMemoryCheckpointStore()


@pytest.fixture
def isolated_registry():
    """Empty registry — strict fail-closed defaults apply."""
    return PermissionRegistry()


@pytest.fixture
def permissive_registry():
    """Registry that auto-allows every tool except names ending in '_denied'.

    Used by tests that exercise the loop's happy paths without wanting
    to spell out an ALLOW gate per tool. Tests that exercise the deny
    behaviour use ``isolated_registry`` and register an explicit gate.
    """

    reg = PermissionRegistry()
    # Glob over every tool name and allow by default.
    reg.register(
        "*",
        lambda ctx: (
            PermissionResult.deny("test convention: explicit deny")
            if ctx.tool_name.endswith("_denied")
            else PermissionResult.allow("test-permissive default")
        ),
    )
    return reg


# ``in_memory_tracer`` lives in tests/conftest.py and is shared across the
# whole suite. Removing the per-file fixture avoids OTEL global-provider
# collisions when this file runs alongside test_observability.py.


# ---------------------------------------------------------------------------
# Canned plan responses
# ---------------------------------------------------------------------------


def _trivial_plan_json() -> str:
    return json.dumps(
        {
            "steps": [
                {
                    "step_id": "s1",
                    "action_type": "TOOL_CALL",
                    "target": "summarise",
                    "inputs": {"text": "hello"},
                    "description": "Summarise the input.",
                }
            ]
        }
    )


def _multi_step_plan_json() -> str:
    return json.dumps(
        {
            "steps": [
                {
                    "step_id": "s1",
                    "action_type": "TOOL_CALL",
                    "target": "fetch_doc",
                    "inputs": {"id": 42},
                    "description": "Fetch the doc.",
                },
                {
                    "step_id": "s2",
                    "action_type": "TOOL_CALL",
                    "target": "summarise",
                    "inputs": {"text": {"$ref": "s1.text"}},
                    "description": "Summarise the fetched doc.",
                },
            ]
        }
    )


def _fixed_plan_json() -> str:
    """Replan response that uses a tool guaranteed to succeed."""
    return json.dumps(
        {
            "steps": [
                {
                    "step_id": "fixed",
                    "action_type": "TOOL_CALL",
                    "target": "summarise",
                    "inputs": {"text": "after replan"},
                    "description": "Use a safe tool after replan.",
                }
            ]
        }
    )


# ---------------------------------------------------------------------------
# Test 1 — GraphState round-trip
# ---------------------------------------------------------------------------


def test_graph_state_round_trip_is_idempotent():
    state = GraphState.new("write a poem", max_iterations=11)
    state.plan = [
        PlanStep(
            step_id="s1",
            action_type=ActionType.TOOL_CALL,
            target="poet",
            inputs={"topic": "rain"},
        )
    ]
    state.retrieved_context = [ContextBlock(text="ctx", source="vector")]
    state.add_turn(AgentState.PLAN, "drafted plan")

    once = state.to_dict()
    rebuilt = GraphState.from_dict(once)
    twice = rebuilt.to_dict()

    assert once == twice, "to_dict/from_dict round-trip must be idempotent"
    assert rebuilt.plan[0].action_type == ActionType.TOOL_CALL
    assert rebuilt.retrieved_context[0].text == "ctx"
    assert rebuilt.turn_history[0].state == AgentState.PLAN


def test_graph_state_round_trip_via_json():
    state = GraphState.new("task")
    state.metadata["debug"] = True
    serialised = json.dumps(state.to_dict())
    rebuilt = GraphState.from_dict(json.loads(serialised))
    assert rebuilt.task_id == state.task_id
    assert rebuilt.metadata["debug"] is True


# ---------------------------------------------------------------------------
# Test 2 — happy path: INIT -> PLAN -> EXECUTE -> EVALUATE -> DONE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_drives_to_done(
    fs_checkpoint_store, permissive_registry
):
    llm = FakeLLMBackend(scripted=[_trivial_plan_json()])
    mcp = FakeMCPClient(
        tools=["summarise"],
        responses={"summarise": lambda args: {"text": f"summary of {args}"}},
    )

    loop = AgentLoop(
        llm_backend=llm,
        mcp_client=mcp,
        permission_registry=permissive_registry,
        checkpoint_store=fs_checkpoint_store,
        default_mcp_server="local",
    )

    final = await loop.run("summarise hello")

    assert final.current_state == AgentState.DONE
    assert final.plan[0].status == StepStatus.SUCCEEDED
    assert "summary of" in (final.final_answer or "")
    # The turn history must include every transition the FSM drove through.
    states_visited = [t.state for t in final.turn_history]
    assert AgentState.INIT in states_visited
    assert AgentState.PLAN in states_visited
    assert AgentState.EXECUTE in states_visited
    assert AgentState.EVALUATE in states_visited
    assert AgentState.DONE in states_visited


# ---------------------------------------------------------------------------
# Test 3 — replan recovers from a failed step
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_replans_on_failed_step(
    fs_checkpoint_store, permissive_registry
):
    # First plan tries 'broken_tool' which fails; replan uses 'summarise'.
    bad_plan = json.dumps(
        {
            "steps": [
                {
                    "step_id": "s1",
                    "action_type": "TOOL_CALL",
                    "target": "broken_tool",
                    "inputs": {},
                    "description": "Will fail.",
                }
            ]
        }
    )

    llm = FakeLLMBackend(scripted=[bad_plan, _fixed_plan_json()])
    mcp = FakeMCPClient(
        tools=["broken_tool", "summarise"],
        responses={
            "broken_tool": lambda _args: (_ for _ in ()).throw(
                RuntimeError("broken_tool exploded")
            ),
            "summarise": lambda args: {"text": "fixed result"},
        },
    )

    loop = AgentLoop(
        llm_backend=llm,
        mcp_client=mcp,
        permission_registry=permissive_registry,
        checkpoint_store=fs_checkpoint_store,
        default_mcp_server="local",
    )

    final = await loop.run("recover from failure")
    assert final.current_state == AgentState.DONE
    assert final.metadata.get("replans") == 1
    # Replan archive captures the original failing plan.
    assert final.metadata.get("replan_archive")
    # The new plan succeeded.
    assert final.plan[0].status == StepStatus.SUCCEEDED
    assert "fixed result" in (final.final_answer or "")


# ---------------------------------------------------------------------------
# Test 4 — max_iterations transitions to ERROR
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_transitions_to_error_after_max_iterations(
    fs_checkpoint_store, permissive_registry
):
    # Plan with five steps; each runs without error but max_iterations is 2
    # so the loop never reaches DONE.
    plan = {
        "steps": [
            {
                "step_id": f"s{i}",
                "action_type": "TOOL_CALL",
                "target": "summarise",
                "inputs": {"i": i},
                "description": f"step {i}",
            }
            for i in range(5)
        ]
    }
    llm = FakeLLMBackend(scripted=[json.dumps(plan)])
    mcp = FakeMCPClient(
        tools=["summarise"],
        responses={"summarise": lambda args: {"text": "ok"}},
    )

    loop = AgentLoop(
        llm_backend=llm,
        mcp_client=mcp,
        permission_registry=permissive_registry,
        checkpoint_store=fs_checkpoint_store,
        max_iterations=2,
        default_mcp_server="local",
    )

    final = await loop.run("ten step task")
    assert final.current_state == AgentState.ERROR
    assert final.error_message
    assert "max_iterations" in final.error_message


# ---------------------------------------------------------------------------
# Test 5 — permission gate DENY blocks tool dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_permission_deny_blocks_tool_call(
    fs_checkpoint_store, isolated_registry
):
    # Register an explicit deny gate for 'dangerous_tool'.
    isolated_registry.register(
        "dangerous_tool",
        lambda _ctx: PermissionResult.deny("policy: not allowed"),
    )

    plan = {
        "steps": [
            {
                "step_id": "s1",
                "action_type": "TOOL_CALL",
                "target": "dangerous_tool",
                "inputs": {},
                "description": "Should be blocked.",
            }
        ]
    }
    # Two LLM responses: the original plan and the replan (we want to
    # confirm the loop transitions to ERROR after exhausting replans).
    llm = FakeLLMBackend(
        scripted=[json.dumps(plan), json.dumps(plan), json.dumps(plan), json.dumps(plan)]
    )
    mcp = FakeMCPClient(tools=["dangerous_tool"], responses={})

    loop = AgentLoop(
        llm_backend=llm,
        mcp_client=mcp,
        permission_registry=isolated_registry,
        checkpoint_store=fs_checkpoint_store,
        default_mcp_server="local",
    )

    final = await loop.run("attempt dangerous op")
    assert final.current_state == AgentState.ERROR
    # The step recorded the permission-denied reason at some point.
    assert any(
        "Permission denied" in (s.error or "")
        for s in final.plan
        if s.error
    ) or any(
        "Permission denied" in (t.summary or "")
        for t in final.turn_history
    )


# ---------------------------------------------------------------------------
# Test 6 — default ALLOW for read-only patterns
# ---------------------------------------------------------------------------


def test_default_gate_allows_read_only_patterns():
    registry = PermissionRegistry()
    for name in [
        "fs.read_file",
        "db.list_tables",
        "api.get_status",
        "search_index",
        "describe_schema",
    ]:
        verdict = registry.check(name, arguments={})
        assert verdict.allowed, f"{name} should be auto-allowed; reason: {verdict.reason}"


def test_default_gate_denies_unknown_writes():
    registry = PermissionRegistry()
    for name in [
        "fs.write_file",
        "db.delete_row",
        "api.post_to_webhook",
        "shell_exec",
    ]:
        verdict = registry.check(name, arguments={})
        assert not verdict.allowed, (
            f"{name} should be denied by default; got allow ({verdict.reason})"
        )


# ---------------------------------------------------------------------------
# Test 7 — multiple gates: first non-ALLOW wins
# ---------------------------------------------------------------------------


def test_multiple_gates_first_non_allow_wins():
    registry = PermissionRegistry()
    registry.register("custom_tool", lambda _ctx: PermissionResult.allow())
    registry.register(
        "custom_tool",
        lambda _ctx: PermissionResult.deny("second gate vetoes"),
    )

    verdict = registry.check("custom_tool", arguments={})
    assert not verdict.allowed
    assert "second gate vetoes" in verdict.reason


def test_glob_gates_match_unrelated_tools():
    registry = PermissionRegistry()
    registry.register(
        "fs.*", lambda _ctx: PermissionResult.deny("fs locked down")
    )
    deny = registry.check("fs.read_file")
    # Glob deny beats the read-only default allow.
    assert not deny.allowed
    assert "fs locked down" in deny.reason

    allow = registry.check("api.get_status")
    assert allow.allowed


# ---------------------------------------------------------------------------
# Test 8 — checkpoint serialisation round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_checkpoint_round_trip(fs_checkpoint_store):
    state = GraphState.new("test checkpoint")
    state.plan = [
        PlanStep(
            step_id="s1",
            action_type=ActionType.LLM_CALL,
            target="anthropic/claude",
            inputs={"prompt": "hi"},
        )
    ]
    state.metadata["k"] = "v"

    idx = await fs_checkpoint_store.save(state)
    assert idx == 0

    state.iteration_count += 1
    idx2 = await fs_checkpoint_store.save(state)
    assert idx2 == 1

    listed = await fs_checkpoint_store.list_checkpoints(state.task_id)
    assert listed == [0, 1]

    rebuilt = await fs_checkpoint_store.load(state.task_id, 0)
    assert rebuilt.task_id == state.task_id
    assert rebuilt.plan[0].action_type == ActionType.LLM_CALL


# ---------------------------------------------------------------------------
# Test 9 — replay from checkpoint reproduces the final state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replay_from_checkpoint_reproduces_final_state(
    fs_checkpoint_store, permissive_registry
):
    # First run: deterministic plan, deterministic tool.
    plan = _trivial_plan_json()
    llm1 = FakeLLMBackend(scripted=[plan])
    mcp1 = FakeMCPClient(
        tools=["summarise"],
        responses={"summarise": lambda args: {"text": "deterministic"}},
    )
    loop1 = AgentLoop(
        llm_backend=llm1,
        mcp_client=mcp1,
        permission_registry=permissive_registry,
        checkpoint_store=fs_checkpoint_store,
        default_mcp_server="local",
    )
    final_a = await loop1.run("replay test")
    assert final_a.current_state == AgentState.DONE

    checkpoints = await fs_checkpoint_store.list_checkpoints(final_a.task_id)
    assert len(checkpoints) >= 3, (
        f"expected several checkpoints, got {len(checkpoints)}"
    )

    # Pick a mid-run checkpoint (after PLAN, before final EVALUATE).
    mid = checkpoints[1] if len(checkpoints) > 2 else checkpoints[0]

    llm2 = FakeLLMBackend(scripted=[plan])
    mcp2 = FakeMCPClient(
        tools=["summarise"],
        responses={"summarise": lambda args: {"text": "deterministic"}},
    )
    loop2 = AgentLoop(
        llm_backend=llm2,
        mcp_client=mcp2,
        permission_registry=permissive_registry,
        checkpoint_store=fs_checkpoint_store,
        default_mcp_server="local",
    )
    final_b = await loop2.replay(final_a.task_id, from_state_idx=mid)
    assert final_b.current_state == AgentState.DONE
    assert final_b.final_answer == final_a.final_answer
    # The successful step's output is the same on replay.
    assert final_b.step_results == final_a.step_results


# ---------------------------------------------------------------------------
# Test 10 — plan-structure validation
# ---------------------------------------------------------------------------


def test_plan_parser_rejects_non_json():
    with pytest.raises(PlanValidationError):
        _parse_plan("definitely not json")


def test_plan_parser_rejects_missing_steps_key():
    with pytest.raises(PlanValidationError):
        _parse_plan(json.dumps({"other": []}))


def test_plan_parser_rejects_empty_steps():
    with pytest.raises(PlanValidationError):
        _parse_plan(json.dumps({"steps": []}))


def test_plan_parser_rejects_unknown_action_type():
    payload = {
        "steps": [
            {
                "step_id": "x",
                "action_type": "WAT",
                "target": "t",
                "inputs": {},
            }
        ]
    }
    with pytest.raises(PlanValidationError):
        _parse_plan(json.dumps(payload))


def test_plan_parser_rejects_duplicate_step_ids():
    payload = {
        "steps": [
            {
                "step_id": "same",
                "action_type": "TOOL_CALL",
                "target": "t",
                "inputs": {},
            },
            {
                "step_id": "same",
                "action_type": "TOOL_CALL",
                "target": "t",
                "inputs": {},
            },
        ]
    }
    with pytest.raises(PlanValidationError):
        _parse_plan(json.dumps(payload))


def test_plan_parser_accepts_code_fence_wrapped_json():
    fenced = (
        "```json\n"
        + json.dumps(
            {
                "steps": [
                    {
                        "step_id": "s1",
                        "action_type": "TOOL_CALL",
                        "target": "t",
                        "inputs": {},
                    }
                ]
            }
        )
        + "\n```"
    )
    steps = _parse_plan(fenced)
    assert len(steps) == 1
    assert steps[0].step_id == "s1"


# ---------------------------------------------------------------------------
# Test 11 — instrumentation: agent_iteration_span lands per transition
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_emits_agent_iteration_spans(
    fs_checkpoint_store, permissive_registry, in_memory_tracer
):
    llm = FakeLLMBackend(scripted=[_trivial_plan_json()])
    mcp = FakeMCPClient(
        tools=["summarise"],
        responses={"summarise": lambda args: {"text": "ok"}},
    )
    loop = AgentLoop(
        llm_backend=llm,
        mcp_client=mcp,
        permission_registry=permissive_registry,
        checkpoint_store=fs_checkpoint_store,
        default_mcp_server="local",
    )
    await loop.run("trace me")

    spans = in_memory_tracer.get_finished_spans()
    iteration_spans = [s for s in spans if s.name.startswith("agent.iteration")]
    assert len(iteration_spans) >= 3, (
        f"expected at least three agent.iteration spans (INIT, PLAN, EXECUTE, "
        f"EVALUATE), got {len(iteration_spans)}: {[s.name for s in spans]}"
    )
    tool_spans = [s for s in spans if s.name.startswith("execute_tool")]
    assert tool_spans, "tool dispatch should have emitted an execute_tool span"


# ---------------------------------------------------------------------------
# Test 12 — memory retriever populates retrieved_context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_retriever_populates_context(
    fs_checkpoint_store, permissive_registry
):
    async def fake_retriever(task: str):
        return [
            ContextBlock(text=f"ctx for: {task}", source="vector", score=0.9),
        ]

    llm = FakeLLMBackend(scripted=[_trivial_plan_json()])
    mcp = FakeMCPClient(
        tools=["summarise"],
        responses={"summarise": lambda args: {"text": "ok"}},
    )
    loop = AgentLoop(
        llm_backend=llm,
        mcp_client=mcp,
        memory_retriever=fake_retriever,
        permission_registry=permissive_registry,
        checkpoint_store=fs_checkpoint_store,
        default_mcp_server="local",
    )

    final = await loop.run("research topic")
    assert final.retrieved_context
    assert final.retrieved_context[0].source == "vector"
    assert "research topic" in final.retrieved_context[0].text


# ---------------------------------------------------------------------------
# Test 13 — $ref input resolution wires outputs between steps
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ref_input_resolution_threads_outputs(
    fs_checkpoint_store, permissive_registry
):
    llm = FakeLLMBackend(scripted=[_multi_step_plan_json()])

    captured: List[Dict[str, Any]] = []

    def _summarise(args):
        captured.append(args)
        return {"text": "summary"}

    mcp = FakeMCPClient(
        tools=["fetch_doc", "summarise"],
        responses={
            "fetch_doc": lambda args: {"text": "doc body", "id": args.get("id")},
            "summarise": _summarise,
        },
    )
    loop = AgentLoop(
        llm_backend=llm,
        mcp_client=mcp,
        permission_registry=permissive_registry,
        checkpoint_store=fs_checkpoint_store,
        default_mcp_server="local",
    )

    final = await loop.run("two step task")
    assert final.current_state == AgentState.DONE
    # The summarise call should have received the doc body from the first step.
    assert captured and captured[0].get("text") == "doc body"


# ---------------------------------------------------------------------------
# Test 14 — step() advances by one transition only
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_step_advances_one_transition(
    fs_checkpoint_store, permissive_registry
):
    llm = FakeLLMBackend(scripted=[_trivial_plan_json()])
    mcp = FakeMCPClient(
        tools=["summarise"],
        responses={"summarise": lambda args: {"text": "ok"}},
    )
    loop = AgentLoop(
        llm_backend=llm,
        mcp_client=mcp,
        permission_registry=permissive_registry,
        checkpoint_store=fs_checkpoint_store,
        default_mcp_server="local",
    )

    state = GraphState.new("step me", max_iterations=10)
    state.add_turn(state.current_state, "seed")
    await fs_checkpoint_store.save(state)

    s1 = await loop.step(state)
    assert s1.current_state == AgentState.PLAN
    s2 = await loop.step(s1)
    assert s2.current_state == AgentState.EXECUTE
    s3 = await loop.step(s2)
    assert s3.current_state == AgentState.EVALUATE
    s4 = await loop.step(s3)
    assert s4.current_state == AgentState.DONE


# ---------------------------------------------------------------------------
# Test 15 — InMemoryCheckpointStore satisfies the protocol contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_in_memory_checkpoint_store_contract():
    store = InMemoryCheckpointStore()
    state = GraphState.new("inmem")
    idx = await store.save(state)
    assert idx == 0

    state.iteration_count += 1
    idx2 = await store.save(state)
    assert idx2 == 1

    loaded = await store.load(state.task_id, 0)
    assert loaded.task_id == state.task_id
    assert loaded.iteration_count == 0  # the original checkpoint

    listed = await store.list_checkpoints(state.task_id)
    assert listed == [0, 1]

    with pytest.raises(FileNotFoundError):
        await store.load(state.task_id, 99)


# ---------------------------------------------------------------------------
# Test 16 — invalid plan from LLM transitions to ERROR
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_plan_drives_to_error(
    fs_checkpoint_store, isolated_registry
):
    llm = FakeLLMBackend(scripted=["this is not JSON"])
    loop = AgentLoop(
        llm_backend=llm,
        permission_registry=isolated_registry,
        checkpoint_store=fs_checkpoint_store,
        default_mcp_server="local",
    )
    final = await loop.run("force invalid plan")
    assert final.current_state == AgentState.ERROR
    assert "invalid plan" in (final.error_message or "")


# ---------------------------------------------------------------------------
# Test 17 — terminal state is sticky; step() on DONE/ERROR is a no-op
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_step_is_noop_on_terminal_state(
    fs_checkpoint_store, isolated_registry
):
    loop = AgentLoop(
        llm_backend=FakeLLMBackend(),
        permission_registry=isolated_registry,
        checkpoint_store=fs_checkpoint_store,
    )
    state = GraphState.new("done already")
    state.current_state = AgentState.DONE
    out = await loop.step(state)
    assert out is state
    assert out.current_state == AgentState.DONE


# ---------------------------------------------------------------------------
# MCP tool results (mcp 2.x types)
# ---------------------------------------------------------------------------


def test_coerce_tool_output_raises_on_an_mcp_tool_error():
    """A ``CallToolResult`` with ``is_error`` must fail the step, not pass as output.

    mcp 2.x exposes the flag only as ``is_error``; reading the old
    ``isError`` attribute would return ``None`` and treat the failure as
    success.
    """
    from mcp.types import CallToolResult, TextContent

    from synthesis_engine.agent.loop import _coerce_tool_output

    result = CallToolResult(content=[TextContent(type="text", text="boom")], is_error=True)
    with pytest.raises(RuntimeError, match="boom"):
        _coerce_tool_output(result)


def test_coerce_tool_output_keeps_text_and_structured_content():
    from mcp.types import CallToolResult, TextContent

    from synthesis_engine.agent.loop import _coerce_tool_output

    result = CallToolResult(
        content=[TextContent(type="text", text="ok")],
        structured_content={"n": 1},
        is_error=False,
    )
    payload = _coerce_tool_output(result)
    assert payload["is_error"] is False
    assert payload["text"] == "ok"
    assert payload["structured_content"] == {"n": 1}
