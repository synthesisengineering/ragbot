"""Agent-loop driver: an explicit-FSM plan-and-execute runtime.

The :class:`AgentLoop` consumes the substrate components (LLM backend,
MCP client, memory retriever, permission registry, checkpoint store)
and drives a task from INIT to DONE (or ERROR) one state transition at
a time. Every transition produces a new :class:`GraphState` value;
that value is observably traced and durably checkpointed.

Architectural notes:

* The transition table is a plain dict of async functions. There is no
  framework, no decorator magic, no DAG library. The whole point of
  this substrate is to keep the agent-loop semantics inspectable.

* The same code path drives both fresh runs (``run()``) and replays
  (``replay()``). Replay simply loads a checkpoint and re-invokes the
  driver loop; deterministic substrates (fake LLMs, fake tool clients)
  reproduce the same final state.

* Every state transition is wrapped in an ``agent_iteration_span`` so
  one OTEL span per transition lands in the tracer. Tool calls are
  additionally wrapped in ``tool_span``; LLM calls in
  ``chat_completion_span``.

* The loop's "plan and execute with replan on failure" pattern is
  pinned to a small, explicit retry budget per step (``MAX_STEP_ATTEMPTS``)
  before the loop transitions to REPLAN. Once REPLAN consumes its own
  budget (``max_iterations`` overall, plus a separate
  ``MAX_REPLANS`` count), the loop transitions to ERROR.
"""

from __future__ import annotations

import asyncio
import logging
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    TYPE_CHECKING,
)

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from ..policy.confidentiality import ConfidentialityCheck
    from ..policy.routing import RoutingPolicy

from .checkpoints import CheckpointStore, FilesystemCheckpointStore
from .grading import SelfGrader
from .permissions import (
    PermissionRegistry,
    PermissionResult,
    ToolCallContext,
    get_default_registry,
)
from .planner import PlanValidationError, make_plan, replan
from .sandbox import DisabledSandbox, ExecutionResult, Sandbox
from .state import (
    ActionType,
    AgentState,
    ContextBlock,
    GraphState,
    PlanStep,
    StepStatus,
)

# Cross-workspace policy substrate (Phase 3 Agent A). Imported lazily
# inside the methods that need it because the policy package transitively
# imports ``synthesis_engine.agent.permissions``, and an eager import at
# the top of this module triggers a circular import (loop.py is loaded
# during ``synthesis_engine.agent`` package init, before that package's
# ``__init__.py`` finishes). The metadata-key constants are pure strings
# so they're safe to import eagerly via a small re-export module — but we
# keep the convention uniform and import everything lazily.

# Sentinel: populated on first use by ``_policy_imports`` below.
_POLICY_IMPORTS: Optional[Dict[str, Any]] = None


def _policy_imports() -> Dict[str, Any]:
    """Lazy import of the policy package to avoid a circular import.

    The loop is loaded during ``synthesis_engine.agent`` package init;
    ``synthesis_engine.policy`` imports back into
    ``synthesis_engine.agent.permissions``, so the policy package can
    only be imported after this module's ``__init__`` finishes. Every
    routing-aware method calls this helper instead of touching a
    module-level binding.
    """

    global _POLICY_IMPORTS
    if _POLICY_IMPORTS is not None:
        return _POLICY_IMPORTS
    from ..policy import (  # noqa: WPS433
        AuditEntry,
        RoutingPolicy,
        check_cross_workspace_op,
        load_routing_policy,
        record as record_audit,
        redact_args,
    )
    from ..policy.confidentiality import (  # noqa: WPS433
        ACTIVE_WORKSPACES_METADATA_KEY,
        ROUTING_POLICIES_METADATA_KEY,
        register_cross_workspace_gate,
    )

    _POLICY_IMPORTS = {
        "AuditEntry": AuditEntry,
        "RoutingPolicy": RoutingPolicy,
        "check_cross_workspace_op": check_cross_workspace_op,
        "load_routing_policy": load_routing_policy,
        "record_audit": record_audit,
        "redact_args": redact_args,
        "ACTIVE_WORKSPACES_METADATA_KEY": ACTIVE_WORKSPACES_METADATA_KEY,
        "ROUTING_POLICIES_METADATA_KEY": ROUTING_POLICIES_METADATA_KEY,
        "register_cross_workspace_gate": register_cross_workspace_gate,
    }
    return _POLICY_IMPORTS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


# Per-step retry budget before the loop escalates to REPLAN.
MAX_STEP_ATTEMPTS: int = 2

# How many REPLAN cycles the loop tolerates before giving up.
MAX_REPLANS: int = 3

# Default tool name surfaced to the permission registry for the two
# capability action types. Operators register gates against these names
# to allow / deny.
SUBAGENT_DISPATCH_TOOL_NAME: str = "subagent_dispatch"
SANDBOX_EXEC_TOOL_NAME: str = "sandbox_exec"


MemoryRetriever = Callable[[str], Awaitable[List[ContextBlock]]]
"""Callable signature for the optional memory retriever.

Production callers wire this to a partial over
``synthesis_engine.memory.three_tier_retrieve``; tests pass a fake.
"""


# ---------------------------------------------------------------------------
# Helpers — instrumentation imports that degrade gracefully
# ---------------------------------------------------------------------------


def _import_instrumentation():
    """Return the observability context managers, falling back to no-ops.

    The substrate's observability module is always available in the
    repo, but the agent loop should never crash because OTEL is
    misconfigured. We import lazily and fall through to no-op context
    managers if anything goes wrong.
    """

    import contextlib

    try:
        from synthesis_engine.observability import (  # noqa: WPS433
            agent_iteration_span,
            chat_completion_span,
            tool_span,
        )
        return agent_iteration_span, chat_completion_span, tool_span
    except Exception:  # pragma: no cover - defensive
        @contextlib.contextmanager
        def _noop_cm(*_args, **_kwargs):
            yield None

        return _noop_cm, _noop_cm, _noop_cm


# ---------------------------------------------------------------------------
# AgentLoop
# ---------------------------------------------------------------------------


class AgentLoop:
    """Drive a single :class:`GraphState` from INIT to a terminal state."""

    def __init__(
        self,
        *,
        llm_backend: Any,
        mcp_client: Optional[Any] = None,
        memory_retriever: Optional[MemoryRetriever] = None,
        permission_registry: Optional[PermissionRegistry] = None,
        checkpoint_store: Optional[CheckpointStore] = None,
        max_iterations: int = 30,
        planner_model: Optional[str] = None,
        default_mcp_server: Optional[str] = None,
        sandbox: Optional[Sandbox] = None,
        dispatcher: Optional[Any] = None,
        grader: Optional[SelfGrader] = None,
    ) -> None:
        self._llm = llm_backend
        self._mcp = mcp_client
        self._retriever = memory_retriever
        self._permissions = permission_registry or get_default_registry()
        self._checkpoints: CheckpointStore = (
            checkpoint_store or FilesystemCheckpointStore()
        )
        self._max_iterations = max_iterations
        self._planner_model = planner_model
        self._default_mcp_server = default_mcp_server
        # Fail-closed defaults: DisabledSandbox refuses every execute().
        # Callers opt into real backends by passing E2BSandbox() etc.
        self._sandbox: Sandbox = sandbox or DisabledSandbox()
        self._grader: Optional[SelfGrader] = grader

        # Per-run cross-workspace policy map. Populated by
        # :meth:`_configure_multi_workspace` on the routing-enforced
        # path; otherwise stays empty so single-workspace runs continue
        # to behave as before.
        self._workspace_policies: Dict[str, "RoutingPolicy"] = {}

        # Dispatcher needs a back-reference to this loop. We lazy-import
        # to avoid an import cycle (dispatch.py imports AgentLoop only
        # under TYPE_CHECKING).
        if dispatcher is None:
            from .dispatch import SubAgentDispatcher  # noqa: WPS433

            self._dispatcher = SubAgentDispatcher(parent_loop=self)
        else:
            self._dispatcher = dispatcher

        (
            self._agent_iteration_span,
            self._chat_completion_span,
            self._tool_span,
        ) = _import_instrumentation()

        # Transition dispatch table — explicit, no framework.
        self._transitions: Dict[
            AgentState,
            Callable[[GraphState], Awaitable[GraphState]],
        ] = {
            AgentState.INIT: self._handle_init,
            AgentState.PLAN: self._handle_plan,
            AgentState.EXECUTE: self._handle_execute,
            AgentState.EVALUATE: self._handle_evaluate,
            AgentState.REPLAN: self._handle_replan,
            AgentState.GRADE: self._handle_grade,
        }

    # ----- accessors (dispatcher reaches in for these) ----------------------

    @property
    def checkpoint_store(self) -> CheckpointStore:
        """Read-only handle to the configured checkpoint store."""
        return self._checkpoints

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    @property
    def sandbox(self) -> Sandbox:
        return self._sandbox

    @property
    def grader(self) -> Optional[SelfGrader]:
        return self._grader

    def subagent_span(
        self,
        *,
        parent_task_id: str,
        child_index: int,
    ):
        """Context manager wrapping one sub-agent run.

        The trace shows the fan-out as nested ``agent.iteration`` spans
        under the parent's current iteration span. Tests can match on
        the ``parent_task_id`` attribute to count children.
        """
        return self._agent_iteration_span(
            iteration=child_index,
            session_id=f"{parent_task_id}/sub/{child_index}",
            extra={
                "synthesis.agent.parent_task_id": parent_task_id,
                "synthesis.agent.child_index": int(child_index),
                "synthesis.agent.role": "subagent",
            },
        )

    # ----- public API -------------------------------------------------------

    async def run(
        self,
        task: str,
        *,
        rubric: Optional[str] = None,
        workspaces: Optional[List[str]] = None,
        workspace_roots: Optional[Dict[str, str]] = None,
        routing_enforced: bool = False,
        cross_workspace_budget_tokens: int = 6000,
    ) -> GraphState:
        """Drive a fresh task from INIT to a terminal state.

        Args:
            task: The user-facing task description.
            rubric: Optional rubric for the self-grading loop. When
                supplied the loop transitions ``DONE -> GRADE`` instead
                of treating DONE as terminal; the grader scores the
                answer, and the loop either transitions to
                ``DONE_GRADED`` (passed) or back to ``REPLAN`` with the
                grader's suggested revisions in the failure context.
                Requires that a :class:`SelfGrader` was passed at
                construction.
            workspaces: Ordered list of workspace names the loop is
                authorized to touch. The first entry is the primary
                workspace. When the list has length ≥ 2 (or when
                ``routing_enforced=True``) the loop enters the routing-
                enforced path: per-workspace :class:`RoutingPolicy`
                objects are loaded, ``check_cross_workspace_op`` is
                consulted before INIT proceeds, and every LLM call is
                gated by ``get_routed_llm_backend``. A length-1 list
                (or ``None``) preserves the existing single-workspace
                behaviour exactly.
            workspace_roots: Optional override mapping workspace name →
                absolute filesystem path of the workspace root. When
                omitted, the loop falls back to
                ``~/workspaces/<name>/ai-knowledge-<name>`` and then
                ``~/ai-knowledge-<name>``. Tests pass an explicit map.
            routing_enforced: When True, force the routing-enforced
                path even for length-0 or length-1 ``workspaces`` lists.
                Used for single-workspace runs that nonetheless want
                LLM-call gating against an explicit policy.
            cross_workspace_budget_tokens: Total token budget for the
                cross-workspace retrieval fan-out. Ignored on the
                single-workspace path.
        """

        state = GraphState.new(task, max_iterations=self._max_iterations)
        state.add_turn(state.current_state, "Initial state.")
        if rubric is not None:
            if self._grader is None:
                raise ValueError(
                    "AgentLoop was given a rubric but no SelfGrader was "
                    "wired. Pass grader=SelfGrader(...) to the loop."
                )
            state.metadata["rubric"] = rubric
            state.metadata["pending_grade"] = True
            state.metadata["grading_rounds"] = 0

        # Multi-workspace wiring. When the routing-enforced path is
        # active we (a) load policies, (b) evaluate the cross-workspace
        # boundary up front so an unsupported mix transitions to ERROR
        # before any model call, and (c) record audit entries for the
        # start AND for every model call AND for the terminal state.
        if self._is_routing_enforced(workspaces, routing_enforced):
            await self._configure_multi_workspace(
                state,
                workspaces=workspaces or [],
                workspace_roots=workspace_roots,
                budget_tokens=cross_workspace_budget_tokens,
            )

        await self._checkpoints.save(state)

        if state.current_state == AgentState.ERROR:
            # The cross-workspace gate refused; the start entry already
            # recorded the denial. Surface a terminal-state audit entry
            # too so callers reading the JSONL log see the closure.
            self._record_terminal_audit(state)
            return state

        result = await self._drive(state)

        # Terminal-state audit for the routing-enforced path. The
        # single-workspace path skips this so unchanged tests don't
        # see new entries on their unrelated audit log.
        if state.metadata.get("routing_enforced"):
            self._record_terminal_audit(result)
        return result

    async def drive_to_terminal(self, state: GraphState) -> GraphState:
        """Drive a pre-built state to a terminal state.

        Public entry that the sub-agent dispatcher uses: it builds the
        child's initial :class:`GraphState` itself (so it can mint a
        parent-linked task_id) and then asks the loop to drive it.
        """
        return await self._drive(state)

    async def step(self, state: GraphState) -> GraphState:
        """Execute exactly one transition and return the new state.

        Useful for tests and for stepping through an agent run from a
        debugger. The loop method that callers normally use is
        :meth:`run`.
        """

        if state.is_terminal():
            return state
        return await self._run_one_transition(state)

    async def replay(
        self,
        task_id: str,
        from_state_idx: Optional[int] = None,
    ) -> GraphState:
        """Resume a task from the given checkpoint and drive to a terminal state.

        Args:
            task_id: The task id whose checkpoint stream to replay.
            from_state_idx: Specific checkpoint index. None means the
                latest checkpoint.
        """

        if from_state_idx is None:
            state = await self._checkpoints.load_latest(task_id)
            if state is None:
                raise FileNotFoundError(
                    f"No checkpoints found for task {task_id}"
                )
        else:
            state = await self._checkpoints.load(task_id, from_state_idx)
        if state.is_terminal():
            return state
        return await self._drive(state)

    # ----- multi-workspace helpers ------------------------------------------

    @staticmethod
    def _is_routing_enforced(
        workspaces: Optional[List[str]], routing_enforced: bool,
    ) -> bool:
        """Decide whether the routing-enforced path applies for this run.

        Activates when ``routing_enforced=True`` OR when ``workspaces``
        has at least one entry. A length-1 list still activates so a
        single-workspace run can opt into policy enforcement; the
        cross-workspace boundary check is short-circuited for it inside
        :func:`check_cross_workspace_op`.

        ``workspaces=None`` AND ``routing_enforced=False`` (the default)
        preserves the existing behaviour exactly.
        """
        if routing_enforced:
            return True
        return bool(workspaces)

    async def _configure_multi_workspace(
        self,
        state: GraphState,
        *,
        workspaces: List[str],
        workspace_roots: Optional[Dict[str, str]],
        budget_tokens: int,
    ) -> None:
        """Load per-workspace policies + evaluate the cross-workspace boundary.

        Populates ``state.metadata`` with:

        * ``active_workspaces`` (List[str]) — de-duplicated, order-preserved.
        * ``routing_policies`` (Dict[str, RoutingPolicy]) — keyed by name.
        * ``effective_confidentiality`` (str) — strictest tag's name.
        * ``cross_workspace_check`` (Dict) — serialised ConfidentialityCheck.
        * ``routing_enforced`` (bool) — True once this method runs.
        * ``cross_workspace_budget_tokens`` (int).

        On a denied mix, sets ``state.current_state = ERROR`` and writes
        a clear ``final_answer`` so the caller does not need to peek at
        metadata to render the refusal.
        """

        seen: List[str] = []
        for name in workspaces:
            if isinstance(name, str) and name and name not in seen:
                seen.append(name)

        imports = _policy_imports()
        load_routing_policy = imports["load_routing_policy"]
        check_cross_workspace_op = imports["check_cross_workspace_op"]
        AuditEntry = imports["AuditEntry"]
        record_audit = imports["record_audit"]
        redact_args = imports["redact_args"]

        roots = dict(workspace_roots or {})
        policies: Dict[str, "RoutingPolicy"] = {}
        for name in seen:
            root = roots.get(name) or self._default_workspace_root(name)
            try:
                policies[name] = load_routing_policy(root)
            except Exception as exc:
                # A malformed routing.yaml is fail-closed: treat the
                # workspace as missing-policy, which the cross-workspace
                # check will then map to AIR_GAPPED. Record the load
                # failure on state.metadata so the operator can debug.
                logger.warning(
                    "Failed to load routing.yaml for workspace %r at %s: %s; "
                    "treating as missing policy (fail-closed).",
                    name, root, exc,
                )
                state.metadata.setdefault("routing_load_errors", {})[name] = str(exc)

        check: "ConfidentialityCheck" = check_cross_workspace_op(seen, policies)

        state.metadata["routing_enforced"] = True
        state.metadata["active_workspaces"] = list(seen)
        state.metadata["routing_policies_names"] = list(policies.keys())
        state.metadata["effective_confidentiality"] = (
            check.effective_confidentiality.name
        )
        state.metadata["cross_workspace_check"] = {
            "allowed": check.allowed,
            "effective_confidentiality": check.effective_confidentiality.name,
            "requires_audit": check.requires_audit,
            "reason": check.reason,
            "boundaries": [
                {
                    "from_workspace": b.from_workspace,
                    "to_workspace": b.to_workspace,
                    "allowed": b.allowed,
                    "reason": b.reason,
                }
                for b in check.boundaries
            ],
        }
        state.metadata["cross_workspace_budget_tokens"] = int(budget_tokens)

        # Stash live policies under a private key so the dispatchers can
        # find them without re-loading. We don't surface this in
        # to_dict() (RoutingPolicy is not JSON-friendly).
        self._workspace_policies = policies

        # Start-of-run audit entry.
        record_audit(
            AuditEntry.build(
                op_type="cross_workspace_run_start",
                workspaces=list(seen),
                tools=[],
                model_id="",
                outcome="allowed" if check.allowed else "denied",
                args_summary=redact_args({"task": state.original_task}),
                metadata={
                    "task_id": state.task_id,
                    "effective_confidentiality": check.effective_confidentiality.name,
                    "requires_audit": check.requires_audit,
                    "reason": check.reason,
                },
            )
        )

        if not check.allowed:
            state.current_state = AgentState.ERROR
            state.error_message = (
                f"Cross-workspace boundary refused this run: {check.reason}"
            )
            state.final_answer = state.error_message
            state.add_turn(state.current_state, state.error_message)
            return

        # Install the gate against the wildcard tool so every TOOL_CALL
        # routes through it. Single-workspace calls short-circuit inside
        # the gate so this does not change single-workspace semantics.
        try:
            register_cross_workspace_gate = imports[
                "register_cross_workspace_gate"
            ]
            register_cross_workspace_gate(self._permissions, tool_name="*")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Failed to register cross-workspace permission gate: %s", exc,
            )

    def _default_workspace_root(self, name: str) -> str:
        """Resolve a workspace name to its on-disk root.

        Tries ``~/workspaces/<name>/ai-knowledge-<name>`` first (the
        workspace-rooted layout the rest of Ragbot uses), then
        ``~/ai-knowledge-<name>`` (the legacy flat-parent layout).
        Falls back to ``~/.synthesis/workspaces/<name>`` so the lookup
        always returns a path even when neither exists; the policy
        loader emits a one-time warning when ``routing.yaml`` is missing.
        """
        from pathlib import Path

        candidates = [
            Path.home() / "workspaces" / name / f"ai-knowledge-{name}",
            Path.home() / f"ai-knowledge-{name}",
        ]
        for path in candidates:
            if path.is_dir():
                return str(path)
        return str(Path.home() / ".synthesis" / "workspaces" / name)

    def _record_terminal_audit(self, state: GraphState) -> None:
        """Record an audit entry for a terminal state of a routing-enforced run."""

        try:
            imports = _policy_imports()
            AuditEntry = imports["AuditEntry"]
            record_audit = imports["record_audit"]
            record_audit(
                AuditEntry.build(
                    op_type="cross_workspace_run_terminal",
                    workspaces=list(state.metadata.get("active_workspaces") or []),
                    tools=[],
                    model_id="",
                    outcome={
                        AgentState.DONE: "done",
                        AgentState.DONE_GRADED: "done_graded",
                        AgentState.ERROR: "error",
                    }.get(state.current_state, "unknown"),
                    args_summary="{}",
                    metadata={
                        "task_id": state.task_id,
                        "current_state": state.current_state.value,
                        "error_message": state.error_message,
                        "effective_confidentiality": state.metadata.get(
                            "effective_confidentiality"
                        ),
                    },
                )
            )
        except Exception as exc:  # pragma: no cover - audit must not crash the loop
            logger.warning("Failed to record terminal audit entry: %s", exc)

    def _gate_metadata(
        self, state: GraphState, step: PlanStep,
    ) -> Dict[str, Any]:
        """Build the permission-gate metadata for one tool dispatch.

        Always includes ``step_id``. On the routing-enforced path also
        carries ``active_workspaces`` + ``routing_policies`` so the
        cross-workspace gate registered against ``*`` can evaluate the
        boundary. Off the routing-enforced path the returned mapping is
        identical to the pre-extension behaviour.
        """
        meta: Dict[str, Any] = {"step_id": step.step_id}
        if state.metadata.get("routing_enforced"):
            imports = _policy_imports()
            meta[imports["ACTIVE_WORKSPACES_METADATA_KEY"]] = list(
                state.metadata.get("active_workspaces") or []
            )
            meta[imports["ROUTING_POLICIES_METADATA_KEY"]] = (
                self._workspace_policies
            )
        return meta

    def _record_model_audit(
        self,
        *,
        state: GraphState,
        requested_model: str,
        resolved_model: str,
        outcome: str,
        step_id: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an audit entry for one model call on a routing-enforced run."""

        try:
            imports = _policy_imports()
            AuditEntry = imports["AuditEntry"]
            record_audit = imports["record_audit"]
            record_audit(
                AuditEntry.build(
                    op_type="model_call",
                    workspaces=list(
                        state.metadata.get("active_workspaces") or []
                    ),
                    tools=[],
                    model_id=resolved_model,
                    outcome=outcome,
                    args_summary="{}",
                    metadata={
                        "task_id": state.task_id,
                        "step_id": step_id,
                        "requested_model": requested_model,
                        "resolved_model": resolved_model,
                        "effective_confidentiality": state.metadata.get(
                            "effective_confidentiality"
                        ),
                        **(extra or {}),
                    },
                )
            )
        except Exception as exc:  # pragma: no cover - audit must not crash the loop
            logger.warning("Failed to record model-call audit entry: %s", exc)

    # ----- internals --------------------------------------------------------

    async def _drive(self, state: GraphState) -> GraphState:
        """Loop until the state is terminal or max_iterations is hit."""

        while not state.is_terminal():
            if state.iteration_count >= state.max_iterations:
                state.current_state = AgentState.ERROR
                state.error_message = (
                    f"Reached max_iterations={state.max_iterations} "
                    "without terminating."
                )
                state.add_turn(
                    state.current_state, state.error_message,
                )
                await self._checkpoints.save(state)
                break
            state = await self._run_one_transition(state)
        return state

    async def _run_one_transition(self, state: GraphState) -> GraphState:
        """Run one transition, wrap it in instrumentation, persist a checkpoint."""

        with self._agent_iteration_span(
            iteration=state.iteration_count,
            session_id=state.task_id,
        ):
            handler = self._transitions.get(state.current_state)
            if handler is None:
                state.current_state = AgentState.ERROR
                state.error_message = (
                    f"No transition handler for state {state.current_state.value}"
                )
                state.add_turn(state.current_state, state.error_message)
            else:
                try:
                    state = await handler(state)
                except Exception as exc:
                    logger.exception(
                        "Unhandled exception in transition handler"
                    )
                    state.current_state = AgentState.ERROR
                    state.error_message = (
                        f"Transition handler raised: {exc!r}"
                    )
                    state.add_turn(
                        state.current_state, state.error_message,
                    )

        state.iteration_count += 1
        await self._checkpoints.save(state)
        return state

    # ----- transition handlers ----------------------------------------------

    async def _handle_init(self, state: GraphState) -> GraphState:
        """INIT -> PLAN (optionally with retrieved context attached).

        When the routing-enforced path is active AND at least one
        workspace is declared, retrieval uses
        :func:`memory.three_tier_retrieve_multi` so blocks tagged with
        their source workspace are surfaced; the agent's answer can then
        cite each block by origin. Single-workspace runs (the existing
        behaviour) call the wired ``self._retriever`` exactly as before.
        """

        multi_active = (
            state.metadata.get("routing_enforced")
            and len(state.metadata.get("active_workspaces") or []) >= 1
        )

        if multi_active:
            try:
                context = await self._retrieve_multi_workspace(state)
                state.retrieved_context = list(context or [])
            except Exception as exc:
                logger.warning(
                    "Cross-workspace memory retrieval failed: %s", exc,
                )
                state.metadata["retrieval_error"] = str(exc)
        elif self._retriever is not None:
            try:
                context = await self._retriever(state.original_task)
                state.retrieved_context = list(context or [])
            except Exception as exc:
                logger.warning("Memory retrieval failed: %s", exc)
                state.metadata["retrieval_error"] = str(exc)

        state.current_state = AgentState.PLAN
        state.add_turn(
            state.current_state,
            f"INIT done; {len(state.retrieved_context)} context block(s) retrieved.",
        )
        return state

    async def _retrieve_multi_workspace(
        self, state: GraphState,
    ) -> List[ContextBlock]:
        """Fan retrieval across the run's active workspaces.

        Imports the multi-workspace retriever lazily so test setups that
        do not configure a memory backend still work; the function falls
        back to the wired ``self._retriever`` when the memory module's
        global backend is not available. Each retrieved block is
        promoted to a :class:`ContextBlock` tagged with its source
        workspace so the planner can cite by origin.
        """

        workspaces: List[str] = list(
            state.metadata.get("active_workspaces") or []
        )
        if not workspaces:
            return []

        try:
            from ..memory import (  # noqa: WPS433
                get_memory,
                three_tier_retrieve_multi,
            )
        except Exception:
            # Memory module unavailable: fall back to the wired retriever.
            if self._retriever is not None:
                return await self._retriever(state.original_task)
            return []

        memory = get_memory()
        if memory is None:
            # No memory backend installed: fall back to the wired retriever.
            if self._retriever is not None:
                return await self._retriever(state.original_task)
            return []

        budget = int(
            state.metadata.get("cross_workspace_budget_tokens") or 6000
        )

        # The retriever is synchronous; run it inline. It is a cheap
        # in-process call (no network) so we don't need a thread pool.
        try:
            blocks = three_tier_retrieve_multi(
                memory,
                workspaces=workspaces,
                query=state.original_task,
                total_budget_tokens=budget,
            )
        except Exception as exc:
            logger.warning(
                "three_tier_retrieve_multi raised; falling back to "
                "single-workspace retriever. error=%r", exc,
            )
            if self._retriever is not None:
                return await self._retriever(state.original_task)
            return []

        out: List[ContextBlock] = []
        for block in blocks:
            out.append(
                ContextBlock(
                    text=block.text,
                    source=f"{block.result.tier}:{block.source_workspace}",
                    score=float(block.score),
                    provenance={
                        "source_workspace": block.source_workspace,
                        "tier": block.result.tier,
                        "workspace_rank": block.workspace_rank,
                        "estimated_tokens": block.estimated_tokens,
                    },
                )
            )
        return out

    async def _handle_plan(self, state: GraphState) -> GraphState:
        """PLAN -> EXECUTE: ask the LLM for a plan and store it."""

        tools = await self._list_available_tools()
        try:
            plan = await make_plan(
                state.original_task,
                available_tools=tools,
                retrieved_context=state.retrieved_context,
                llm_backend=self._llm,
                model=self._planner_model,
            )
        except PlanValidationError as exc:
            state.current_state = AgentState.ERROR
            state.error_message = f"Planner produced an invalid plan: {exc}"
            state.add_turn(state.current_state, state.error_message)
            return state

        state.plan = plan
        state.current_state = AgentState.EXECUTE
        state.add_turn(
            state.current_state,
            f"PLAN done; {len(plan)} step(s) drafted.",
        )
        return state

    async def _handle_execute(self, state: GraphState) -> GraphState:
        """EXECUTE -> EVALUATE: run the next pending step."""

        step = state.next_pending_step()
        if step is None:
            # Nothing pending: jump straight to EVALUATE so the loop
            # decides DONE vs REPLAN based on the recorded outcomes.
            state.current_state = AgentState.EVALUATE
            state.add_turn(
                state.current_state, "EXECUTE: no pending step; evaluating."
            )
            return state

        step.status = StepStatus.RUNNING
        step.attempts += 1

        # Resolve any $ref entries in the step inputs against prior outputs.
        resolved_inputs = _resolve_inputs(step.inputs, state.step_results)

        try:
            output = await self._dispatch_step(state, step, resolved_inputs)
            step.output = output
            step.status = StepStatus.SUCCEEDED
            state.step_results[step.step_id] = output
            summary = (
                f"EXECUTE step {step.step_id} ({step.action_type.value}) -> "
                "succeeded"
            )
        except PermissionError as exc:
            step.status = StepStatus.FAILED
            step.error = f"Permission denied: {exc}"
            summary = (
                f"EXECUTE step {step.step_id} permission-denied: {exc}"
            )
        except Exception as exc:
            logger.warning(
                "Step %s failed (attempt %s): %s",
                step.step_id,
                step.attempts,
                exc,
            )
            if step.attempts < MAX_STEP_ATTEMPTS:
                step.status = StepStatus.PENDING
                step.error = str(exc)
                summary = (
                    f"EXECUTE step {step.step_id} failed attempt "
                    f"{step.attempts}; will retry."
                )
            else:
                step.status = StepStatus.FAILED
                step.error = str(exc)
                summary = (
                    f"EXECUTE step {step.step_id} failed after "
                    f"{step.attempts} attempts: {exc}"
                )

        state.current_state = AgentState.EVALUATE
        state.add_turn(state.current_state, summary)
        return state

    async def _handle_evaluate(self, state: GraphState) -> GraphState:
        """EVALUATE -> DONE | EXECUTE | REPLAN."""

        # FAILED step: route to REPLAN (subject to the REPLAN budget).
        if state.has_unresolved_failure():
            replans = int(state.metadata.get("replans", 0))
            if replans >= MAX_REPLANS:
                state.current_state = AgentState.ERROR
                state.error_message = (
                    f"Exceeded MAX_REPLANS={MAX_REPLANS} without recovery."
                )
                state.add_turn(state.current_state, state.error_message)
                return state
            state.current_state = AgentState.REPLAN
            state.add_turn(
                state.current_state,
                f"EVALUATE: failure detected, transitioning to REPLAN "
                f"(replan #{replans + 1}).",
            )
            return state

        # Still steps to run? Loop back to EXECUTE.
        if state.next_pending_step() is not None:
            state.current_state = AgentState.EXECUTE
            state.add_turn(
                state.current_state, "EVALUATE: pending steps remain."
            )
            return state

        # All steps succeeded (or were skipped). The final answer is
        # the output of the last SUCCEEDED step by convention; an
        # explicit LLM_CALL with an answer-producing role can override
        # this by writing to ``metadata["final_answer"]`` itself.
        final = state.metadata.get("final_answer")
        if final is None:
            succeeded = [
                s for s in state.plan if s.status == StepStatus.SUCCEEDED
            ]
            if succeeded:
                final = succeeded[-1].output
        state.final_answer = _coerce_str(final)

        # If a rubric is pending, hand off to the grader before declaring
        # DONE terminal. Otherwise DONE is the final state.
        if state.metadata.get("pending_grade") and self._grader is not None:
            state.current_state = AgentState.GRADE
            state.add_turn(
                state.current_state,
                "EVALUATE: all steps complete; transitioning to GRADE.",
            )
            return state

        state.current_state = AgentState.DONE
        state.add_turn(state.current_state, "EVALUATE: all steps complete.")
        return state

    async def _handle_replan(self, state: GraphState) -> GraphState:
        """REPLAN -> EXECUTE with a new step list."""

        try:
            new_plan = await replan(
                state, llm_backend=self._llm, model=self._planner_model
            )
        except PlanValidationError as exc:
            state.current_state = AgentState.ERROR
            state.error_message = f"Replanner produced an invalid plan: {exc}"
            state.add_turn(state.current_state, state.error_message)
            return state

        # Track failed steps for audit, then replace plan with the new one.
        state.metadata["replans"] = int(state.metadata.get("replans", 0)) + 1
        archive = state.metadata.setdefault("replan_archive", [])
        archive.append([s.to_dict() for s in state.plan])

        state.plan = new_plan
        state.current_state = AgentState.EXECUTE
        state.add_turn(
            state.current_state,
            f"REPLAN done; new plan has {len(new_plan)} step(s).",
        )
        return state

    async def _handle_grade(self, state: GraphState) -> GraphState:
        """GRADE -> DONE_GRADED | REPLAN.

        Scores the loop's final answer against the rubric the caller
        supplied. A passing score transitions to DONE_GRADED. A failing
        score with revision budget remaining transitions to REPLAN with
        the grader's suggested revisions wired into the failure context;
        no remaining budget transitions to DONE_GRADED with
        ``passed=False`` recorded so the caller sees the verdict.
        """

        if self._grader is None:
            # Defensive: if we got here without a grader, the rubric
            # was wired without a grader (the run() guard should have
            # caught this, but we double-check).
            state.metadata["pending_grade"] = False
            state.current_state = AgentState.DONE_GRADED
            state.metadata["grading"] = {
                "score": 0.0,
                "passed": False,
                "error": "GRADE state entered without a SelfGrader wired.",
            }
            state.add_turn(state.current_state, "GRADE: no grader wired.")
            return state

        rubric = str(state.metadata.get("rubric") or "")
        result = await self._grader.grade(state, rubric)

        history = state.metadata.setdefault("grading_history", [])
        history.append(result.to_dict())
        state.metadata["grading"] = result.to_dict()

        rounds_so_far = int(state.metadata.get("grading_rounds", 0))
        budget = self._grader.max_revision_rounds

        if result.passed:
            state.metadata["pending_grade"] = False
            state.current_state = AgentState.DONE_GRADED
            state.add_turn(
                state.current_state,
                f"GRADE: passed (score={result.score:.2f}).",
            )
            return state

        if rounds_so_far >= budget:
            # Budget exhausted — accept the current answer with a
            # passed=False annotation. The caller sees the score and
            # rationale in state.metadata['grading'].
            state.metadata["pending_grade"] = False
            state.current_state = AgentState.DONE_GRADED
            state.add_turn(
                state.current_state,
                (
                    f"GRADE: score={result.score:.2f} below threshold; "
                    f"revision budget ({budget}) exhausted."
                ),
            )
            return state

        # Inject the grader's suggested revisions as a synthetic failed
        # step so the standard REPLAN path picks them up via
        # _summarise_failures.
        state.metadata["grading_rounds"] = rounds_so_far + 1
        feedback_step = PlanStep(
            step_id=f"grade-feedback-{rounds_so_far + 1}",
            action_type=ActionType.LLM_CALL,
            target="grader",
            inputs={"rubric": rubric},
            status=StepStatus.FAILED,
            error=(
                f"Grader returned score={result.score:.2f} (threshold "
                f"{self._grader.threshold}). Suggested revisions: "
                + "; ".join(result.suggested_revisions or [])
                + (f". Rationale: {result.rationale}" if result.rationale else "")
            ),
            description="Synthetic step capturing grader feedback for replan.",
        )
        # Reset prior plan to SKIPPED so the loop replans cleanly.
        for step in state.plan:
            if step.status in (StepStatus.PENDING, StepStatus.RUNNING):
                step.status = StepStatus.SKIPPED
        state.plan = list(state.plan) + [feedback_step]
        state.current_state = AgentState.REPLAN
        state.add_turn(
            state.current_state,
            (
                f"GRADE: score={result.score:.2f} below threshold; "
                f"transitioning to REPLAN (round "
                f"{rounds_so_far + 1}/{budget})."
            ),
        )
        return state

    # ----- dispatch ---------------------------------------------------------

    async def _dispatch_step(
        self,
        state: GraphState,
        step: PlanStep,
        inputs: Dict[str, Any],
    ) -> Any:
        """Dispatch one plan step through the right substrate."""

        if step.action_type == ActionType.TOOL_CALL:
            return await self._dispatch_tool_call(state, step, inputs)
        if step.action_type == ActionType.LLM_CALL:
            return await self._dispatch_llm_call(state, step, inputs)
        if step.action_type == ActionType.MEMORY_QUERY:
            return await self._dispatch_memory_query(state, step, inputs)
        if step.action_type == ActionType.SUBAGENT_DISPATCH:
            return await self._dispatch_subagent(state, step, inputs)
        if step.action_type == ActionType.SANDBOX_EXEC:
            return await self._dispatch_sandbox_exec(state, step, inputs)
        raise ValueError(
            f"Unknown ActionType {step.action_type!r} for step {step.step_id}"
        )

    async def _dispatch_subagent(
        self,
        state: GraphState,
        step: PlanStep,
        inputs: Dict[str, Any],
    ) -> Any:
        """Run N child loops in parallel and aggregate their outputs.

        The plan step must carry ``inputs.subtasks`` (a list of strings)
        and may carry ``inputs.max_parallel`` (int, default 4). Each
        child's final answer lands in the aggregated output as
        ``{"subtasks": [...], "results": [...]}``. Failures route
        through the standard step-retry / replan path.
        """

        from .dispatch import SubAgentFailure  # noqa: WPS433

        # Permission gate first — fail-closed.
        verdict: PermissionResult = self._permissions.check(
            SUBAGENT_DISPATCH_TOOL_NAME,
            arguments=inputs,
            context=ToolCallContext(
                tool_name=SUBAGENT_DISPATCH_TOOL_NAME,
                arguments=inputs,
                server_id="agent",
                task_id=state.task_id,
                metadata=self._gate_metadata(state, step),
            ),
        )
        if not verdict.allowed:
            raise PermissionError(verdict.reason)

        subtasks_raw = inputs.get("subtasks") or []
        if not isinstance(subtasks_raw, list) or not subtasks_raw:
            raise ValueError(
                f"SUBAGENT_DISPATCH step {step.step_id} requires a "
                "non-empty 'subtasks' list in inputs."
            )
        subtasks = [str(t) for t in subtasks_raw]
        max_parallel = int(inputs.get("max_parallel", 4) or 4)

        try:
            child_states = await self._dispatcher.dispatch(
                state, subtasks, max_parallel=max_parallel
            )
        except SubAgentFailure as exc:
            # Re-raise with a flat message; the loop's step-retry path
            # catches Exception and records it on the step.
            raise RuntimeError(str(exc)) from exc

        return {
            "subtasks": subtasks,
            "results": [
                {
                    "task_id": cs.task_id,
                    "subtask": cs.original_task,
                    "current_state": cs.current_state.value,
                    "final_answer": cs.final_answer,
                    "error_message": cs.error_message,
                }
                for cs in child_states
            ],
        }

    async def _dispatch_sandbox_exec(
        self,
        state: GraphState,
        step: PlanStep,
        inputs: Dict[str, Any],
    ) -> Any:
        """Run untrusted code inside the configured :class:`Sandbox`.

        Routes through the permission registry first (fail-closed),
        then calls ``self._sandbox.execute()``. The result lands in the
        plan step's output and is captured in a tool_span tagged with
        the sandbox provider and language.

        A non-zero exit code raises so the standard step-retry path
        records the failure. The exception message includes the
        sandbox's stderr so the replanner sees what went wrong.
        """

        # Permission gate first — fail-closed.
        verdict: PermissionResult = self._permissions.check(
            SANDBOX_EXEC_TOOL_NAME,
            arguments=inputs,
            context=ToolCallContext(
                tool_name=SANDBOX_EXEC_TOOL_NAME,
                arguments=inputs,
                server_id="agent",
                task_id=state.task_id,
                metadata=self._gate_metadata(state, step),
            ),
        )
        if not verdict.allowed:
            raise PermissionError(verdict.reason)

        code = str(inputs.get("code") or "")
        if not code:
            raise ValueError(
                f"SANDBOX_EXEC step {step.step_id} requires non-empty 'code'."
            )
        language = str(step.target or inputs.get("language") or "python")
        timeout_seconds = int(inputs.get("timeout_seconds", 30) or 30)
        raw_files = inputs.get("files") or {}
        files: Dict[str, bytes] = {}
        for path, data in raw_files.items():
            if isinstance(data, (bytes, bytearray)):
                files[str(path)] = bytes(data)
            elif isinstance(data, str):
                # Treat plain strings as utf-8 file bodies — convenient
                # for tests and for the LLM's likely output shape.
                files[str(path)] = data.encode("utf-8")
            else:
                raise ValueError(
                    f"SANDBOX_EXEC files[{path!r}] must be bytes or str, "
                    f"got {type(data).__name__}."
                )

        with self._tool_span(
            tool_name=SANDBOX_EXEC_TOOL_NAME,
            tool_type="sandbox",
            extra={
                "synthesis.sandbox.provider": self._sandbox.provider,
                "synthesis.sandbox.language": language,
            },
        ) as span:
            result: ExecutionResult = await self._sandbox.execute(
                code,
                language=language,
                timeout_seconds=timeout_seconds,
                files=files,
            )
            try:
                if span is not None:
                    span.set_attribute(
                        "synthesis.sandbox.exit_code", int(result.exit_code)
                    )
                    span.set_attribute(
                        "synthesis.sandbox.duration_seconds",
                        float(result.duration_seconds),
                    )
            except Exception:  # pragma: no cover - defensive
                pass

        if not result.success:
            raise RuntimeError(
                f"Sandbox ({result.provider}) exec failed with "
                f"exit_code={result.exit_code}: {result.stderr or 'no stderr'}"
            )
        return result.to_dict()

    async def _dispatch_tool_call(
        self,
        state: GraphState,
        step: PlanStep,
        inputs: Dict[str, Any],
    ) -> Any:
        if self._mcp is None:
            raise RuntimeError(
                "Plan requested a TOOL_CALL but no MCP client was wired."
            )

        # Permission gate. On the routing-enforced path we also pipe the
        # active workspaces + their loaded policies through the gate's
        # metadata so the cross-workspace gate registered against ``*``
        # can evaluate the boundary.
        server_id, tool_name = _split_tool_target(
            step.target, self._default_mcp_server
        )
        verdict: PermissionResult = self._permissions.check(
            tool_name,
            arguments=inputs,
            context=ToolCallContext(
                tool_name=tool_name,
                arguments=inputs,
                server_id=server_id,
                task_id=state.task_id,
                metadata=self._gate_metadata(state, step),
            ),
        )
        if not verdict.allowed:
            raise PermissionError(verdict.reason)

        # Dispatch under a tool_span.
        with self._tool_span(tool_name=tool_name) as span:
            result = await self._mcp.call_tool(
                server_id, tool_name, arguments=inputs
            )
            try:
                if span is not None:
                    span.set_attribute(
                        "synthesis.tool.server_id", server_id
                    )
            except Exception:  # pragma: no cover - defensive
                pass
            return _coerce_tool_output(result)

    async def _dispatch_llm_call(
        self,
        state: GraphState,
        step: PlanStep,
        inputs: Dict[str, Any],
    ) -> Any:
        messages = inputs.get("messages")
        if not messages:
            # Convenience: the LLM call may have been written as just a
            # `prompt` string; wrap it.
            prompt = inputs.get("prompt") or ""
            if not prompt:
                raise ValueError(
                    f"LLM_CALL step {step.step_id} needs 'messages' or 'prompt'."
                )
            messages = [{"role": "user", "content": prompt}]

        # Allow per-step model overrides; otherwise use the configured planner_model.
        requested_model = step.target or inputs.get("model") or "default"

        # Routing-policy gate. On the routing-enforced path consult the
        # per-workspace policies before dispatching. The gate may
        # downgrade the model (DOWNGRADE_TO_LOCAL), warn-and-proceed
        # (WARN), or raise a ModelDeniedError (DENY) — in which case the
        # standard step-retry path records the failure.
        model = requested_model
        if state.metadata.get("routing_enforced") and self._workspace_policies:
            try:
                from ..llm import (  # noqa: WPS433
                    ModelDeniedError,
                    get_routed_llm_backend,
                )

                _, resolved = get_routed_llm_backend(
                    self._workspace_policies,
                    requested_model,
                    backend=self._llm,
                )
                model = resolved
                self._record_model_audit(
                    state=state,
                    requested_model=requested_model,
                    resolved_model=resolved,
                    outcome=(
                        "allowed"
                        if resolved == requested_model
                        else "downgraded"
                    ),
                    step_id=step.step_id,
                )
            except ModelDeniedError as exc:
                self._record_model_audit(
                    state=state,
                    requested_model=requested_model,
                    resolved_model=requested_model,
                    outcome="denied",
                    step_id=step.step_id,
                    extra={
                        "denying_workspace": exc.denying_workspace,
                        "reason": exc.reason,
                    },
                )
                raise

        provider = inputs.get("provider") or _provider_from_model(model)
        max_tokens = int(inputs.get("max_tokens", 1024))
        temperature = inputs.get("temperature")

        with self._chat_completion_span(
            model=model,
            provider=provider,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            request = _build_llm_request(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            response = self._llm.complete(request)
            if asyncio.iscoroutine(response):
                response = await response
            return getattr(response, "text", response)

    async def _dispatch_memory_query(
        self,
        state: GraphState,
        step: PlanStep,
        inputs: Dict[str, Any],
    ) -> Any:
        if self._retriever is None:
            raise RuntimeError(
                "Plan requested a MEMORY_QUERY but no retriever was wired."
            )
        query = inputs.get("query") or inputs.get("text") or state.original_task
        blocks = await self._retriever(query)
        return [b.to_dict() if isinstance(b, ContextBlock) else b for b in blocks]

    async def _list_available_tools(self) -> List[Any]:
        if self._mcp is None:
            return []
        if self._default_mcp_server is None:
            # Without a default server, we don't know which to list;
            # return an empty list rather than guessing.
            return []
        try:
            return await self._mcp.list_tools(self._default_mcp_server)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not list MCP tools: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_tool_target(
    target: str, default_server: Optional[str]
) -> tuple:
    """Split a ``server::tool`` target into (server_id, tool_name).

    A bare tool name uses ``default_server`` as the server id. Raises
    ``ValueError`` if no server can be resolved.
    """

    if "::" in target:
        server_id, tool_name = target.split("::", 1)
        return server_id, tool_name
    if default_server:
        return default_server, target
    raise ValueError(
        f"Tool target {target!r} has no server prefix and the agent loop "
        "has no default_mcp_server configured."
    )


def _resolve_inputs(
    inputs: Dict[str, Any], step_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Resolve ``{"$ref": "step_id.output"}`` placeholders against prior results.

    Allows a plan step to reference the output of an earlier step. The
    syntax is ``{"$ref": "step_id"}`` for the whole output and
    ``{"$ref": "step_id.field"}`` for a dotted path into a dict output.
    Anything else passes through untouched.
    """

    def _resolve(value: Any) -> Any:
        if isinstance(value, dict):
            ref = value.get("$ref")
            if ref is not None and isinstance(ref, str):
                parts = ref.split(".")
                head = parts[0]
                cursor: Any = step_results.get(head)
                for p in parts[1:]:
                    if isinstance(cursor, dict):
                        cursor = cursor.get(p)
                    else:
                        cursor = None
                        break
                return cursor
            return {k: _resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_resolve(v) for v in value]
        return value

    return {k: _resolve(v) for k, v in inputs.items()}


def _build_llm_request(
    *,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: Optional[float],
) -> Any:
    try:
        from synthesis_engine.llm.base import LLMRequest  # noqa: WPS433

        return LLMRequest(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception:
        return {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }


def _provider_from_model(model: str) -> str:
    """Best-effort provider extraction from a model id (e.g., ``anthropic/...``)."""
    if "/" in model:
        return model.split("/", 1)[0]
    return "unknown"


def _coerce_tool_output(result: Any) -> Any:
    """Reduce an MCP CallToolResult to a serialisable form.

    ``CallToolResult`` carries a ``.content`` list of typed content
    blocks and an ``.is_error`` flag. For the agent state we capture the
    flag plus a list of text fragments; structured content is preserved
    under ``structured_content`` when present.
    """

    if result is None:
        return None
    if isinstance(result, (str, int, float, bool, list, dict)):
        return result

    payload: Dict[str, Any] = {}
    is_error = getattr(result, "is_error", None)
    if is_error is not None:
        payload["is_error"] = bool(is_error)
        if is_error:
            raise RuntimeError(
                f"Tool reported error: {_extract_text(result) or 'unknown'}"
            )
    text = _extract_text(result)
    if text:
        payload["text"] = text
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        payload["structured_content"] = structured
    return payload or _coerce_str(result)


def _extract_text(result: Any) -> str:
    """Pull a single text string out of an MCP CallToolResult."""

    content = getattr(result, "content", None)
    if not content:
        return ""
    parts: List[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)


def _coerce_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        import json

        return json.dumps(value, default=str)
    except Exception:
        return str(value)


__all__ = [
    "AgentLoop",
    "MAX_REPLANS",
    "MAX_STEP_ATTEMPTS",
    "MemoryRetriever",
    "SANDBOX_EXEC_TOOL_NAME",
    "SUBAGENT_DISPATCH_TOOL_NAME",
]
