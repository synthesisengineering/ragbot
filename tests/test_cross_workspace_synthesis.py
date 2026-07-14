"""Tests for the ``cross-workspace-synthesis`` starter-pack skill and its driver.

Phase 3 Agent C ships:

* a sixth starter-pack skill (``cross-workspace-synthesis``) that declares a
  ``cross_workspace_synthesize`` tool with a full JSON schema,
* a Python driver (:func:`run_cross_workspace_synthesis`) that the runtime
  hooks the tool to,
* an audit-trail surface visible inside every returned SynthesisReport.

These tests cover both surfaces. Skill parsing, scope, tool declarations,
and discovery wiring exercise the SKILL.md side. The driver tests exercise
the happy path, the confidentiality-denied path, the budget-passthrough
contract, and audit-on-every-path semantics. Placeholder workspace names
(``acme-news``, ``acme-user``, ``beta-media``) are used throughout —
ragbot is a public repo, so client-real names never appear in tests.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import pytest

# Add src/ to sys.path so the tests can import the package under test.
_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Import ``synthesis_engine.agent.permissions`` FIRST. There is a pre-existing
# load-order sensitivity in the substrate (introduced by Phase 3 Agent A):
# ``policy.confidentiality`` imports from ``..agent.permissions``, and
# ``agent.loop`` imports back from ``..policy``. When the policy package is
# entered first via ``policy/__init__.py``, the cycle fails. When the agent
# package's submodules (``loop.py`` in particular) load first via the
# ``permissions`` path, ``policy.confidentiality`` is loaded inside the agent
# package's transitive import and resolves cleanly. ``test_policy.py`` does
# the same thing for the same reason.
from synthesis_engine.agent.permissions import (  # noqa: E402,F401
    PermissionRegistry,
    ToolCallContext,
)
from synthesis_engine.policy import (  # noqa: E402
    AuditEntry,
    Confidentiality,
    RoutingPolicy,
)
from synthesis_engine.memory import (  # noqa: E402
    MemoryResult,
    RetrievedBlock,
)
from synthesis_engine.llm.base import (  # noqa: E402
    LLMBackend,
    LLMRequest,
    LLMResponse,
)
from synthesis_engine.policy.routing import _clear_warning_cache  # noqa: E402
from synthesis_engine.skills import (  # noqa: E402
    discover_skills_in_root,
    get_skills_for_workspace,
    parse_skill,
)
from synthesis_engine.skills.cross_workspace_driver import (  # noqa: E402
    Citation,
    ConfidentialityError,
    Finding,
    SynthesisReport,
    run_cross_workspace_synthesis,
)
from synthesis_engine.skills.starter_pack import (  # noqa: E402
    list_starter_skill_paths,
    starter_pack_root,
)


SKILL_NAME = "cross-workspace-synthesis"
SKILL_TOOL_NAME = "cross_workspace_synthesize"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_routing_warnings():
    """Keep the per-test 'missing routing.yaml' warning memo clean."""
    _clear_warning_cache()
    yield
    _clear_warning_cache()


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch) -> Path:
    """Isolate Path.home() so discovery does not see the operator's real skills."""
    home = tmp_path / "fakehome"
    home.mkdir()
    (home / ".synthesis" / "skills").mkdir(parents=True)
    (home / ".claude" / "skills").mkdir(parents=True)
    (home / "workspaces" / "acme-user" / "synthesis-skills").mkdir(parents=True)
    identity = home / ".synthesis" / "identity.yaml"
    identity.write_text("personal_workspaces:\n  - acme-user\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("SYNTHESIS_IDENTITY_CONFIG", str(identity))
    return home


# ---------------------------------------------------------------------------
# Fakes for the driver
# ---------------------------------------------------------------------------


class FakeLLMBackend(LLMBackend):
    """In-memory LLM backend that returns a configurable response.

    Records every request so tests can assert on the system + user prompt
    the driver assembled. The default response is a well-formed
    SynthesisReport JSON; tests override ``self.response_text`` to drive
    the malformed-output path.
    """

    backend_name = "fake"

    def __init__(self, response_text: Optional[str] = None) -> None:
        self.response_text = response_text or json.dumps(
            {
                "summary": (
                    "Both workspaces describe the migration plan but assign "
                    "different owners; the date agrees [acme-news:adr-0042]."
                ),
                "findings": [
                    {
                        "claim": "The migration is scheduled for May 2026.",
                        "supporting_citations": [
                            {
                                "workspace": "acme-news",
                                "document_id": "release-2026-05.md",
                                "snippet": "Migration kickoff: 2026-05-12.",
                            },
                            {
                                "workspace": "acme-user",
                                "document_id": "notes-2026-04-30.md",
                                "snippet": "Confirmed May for the migration.",
                            },
                        ],
                    },
                    {
                        "claim": "The migration owner is contested.",
                        "supporting_citations": [
                            {
                                "workspace": "acme-news",
                                "document_id": "adr-0042",
                                "snippet": "Owner: engineering lead.",
                            },
                            {
                                "workspace": "acme-user",
                                "document_id": "notes-2026-04-30.md",
                                "snippet": "Priya owns the migration.",
                            },
                        ],
                        "conflict_note": (
                            "acme-news says engineering lead; acme-user says Priya."
                        ),
                    },
                ],
                "citations": [
                    {
                        "workspace": "acme-news",
                        "document_id": "release-2026-05.md",
                        "snippet": "Migration kickoff: 2026-05-12.",
                    },
                    {
                        "workspace": "acme-news",
                        "document_id": "adr-0042",
                        "snippet": "Owner: engineering lead.",
                    },
                    {
                        "workspace": "acme-user",
                        "document_id": "notes-2026-04-30.md",
                        "snippet": "Confirmed May for the migration. Priya owns it.",
                    },
                ],
                "audit_trail": [],
                "effective_confidentiality": "PERSONAL",
            }
        )
        self.requests: List[LLMRequest] = []
        self.should_raise: bool = False

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self.should_raise:
            raise RuntimeError("simulated LLM failure")
        return LLMResponse(
            text=self.response_text,
            model=request.model,
            backend=self.backend_name,
            finish_reason="stop",
            usage={"input_tokens": 100, "output_tokens": 50},
        )

    def stream(self, request: LLMRequest, on_chunk: Callable[[str], None]) -> str:
        # Not exercised by the driver; provide a stub to satisfy the ABC.
        text = self.complete(request).text
        on_chunk(text)
        return text

    def healthcheck(self) -> Dict[str, Any]:
        return {"backend": self.backend_name, "ok": True}


def _result(text: str, score: float, metadata: Optional[Dict[str, Any]] = None) -> MemoryResult:
    return MemoryResult(
        tier="vector",
        score=score,
        text=text,
        metadata=metadata or {},
    )


def _block(
    workspace: str,
    text: str,
    score: float,
    rank: int,
    document_id: Optional[str] = None,
) -> RetrievedBlock:
    md: Dict[str, Any] = {"source_workspace": workspace}
    if document_id:
        md["document_id"] = document_id
    return RetrievedBlock(
        source_workspace=workspace,
        result=_result(text, score, metadata=md),
        estimated_tokens=max(1, len(text) // 4),
        workspace_rank=rank,
    )


class FakeMemoryRetriever:
    """Mimics ``three_tier_retrieve_multi`` against a per-workspace block table.

    Records the call so tests can assert that ``total_budget_tokens`` was
    threaded through. Raises ``RuntimeError`` for a workspace marked via
    :meth:`raise_for` to exercise the partial-failure path.
    """

    def __init__(self) -> None:
        self._blocks_by_workspace: Dict[str, List[RetrievedBlock]] = {}
        self.calls: List[Dict[str, Any]] = []

    def seed(self, workspace: str, blocks: List[RetrievedBlock]) -> None:
        self._blocks_by_workspace[workspace] = list(blocks)

    def __call__(
        self,
        memory: Any,
        workspaces: List[str],
        query: str,
        *,
        total_budget_tokens: int,
        **kwargs: Any,
    ) -> List[RetrievedBlock]:
        self.calls.append(
            {
                "workspaces": list(workspaces),
                "query": query,
                "total_budget_tokens": total_budget_tokens,
                "kwargs": dict(kwargs),
            }
        )
        out: List[RetrievedBlock] = []
        for w in workspaces:
            out.extend(self._blocks_by_workspace.get(w, []))
        return out


class FakeAuditWriter:
    """Captures audit entries in-memory instead of touching the JSONL file."""

    def __init__(self) -> None:
        self.entries: List[AuditEntry] = []

    def __call__(self, entry: AuditEntry) -> None:
        self.entries.append(entry)


# ---------------------------------------------------------------------------
# Skill parsing + frontmatter contract
# ---------------------------------------------------------------------------


class TestSkillParses:
    def test_skill_parses_without_errors(self) -> None:
        path = os.path.join(starter_pack_root(), SKILL_NAME)
        skill = parse_skill(path)
        assert skill is not None, "parse_skill returned None for cross-workspace-synthesis"
        assert skill.name == SKILL_NAME

    def test_scope_is_universal(self) -> None:
        path = os.path.join(starter_pack_root(), SKILL_NAME)
        skill = parse_skill(path)
        assert skill is not None
        assert skill.scope.universal is True
        assert skill.scope.workspaces == ()

    def test_license_is_apache_2_0(self) -> None:
        path = os.path.join(starter_pack_root(), SKILL_NAME)
        skill = parse_skill(path)
        assert skill is not None
        assert skill.frontmatter.get("license") == "Apache-2.0"

    def test_description_advertises_synthesis_triggers(self) -> None:
        path = os.path.join(starter_pack_root(), SKILL_NAME)
        skill = parse_skill(path)
        assert skill is not None
        desc = skill.description.lower()
        assert "synthesi" in desc
        assert "citation" in desc or "cite" in desc

    def test_body_is_substantial(self) -> None:
        """The skill body walks the LLM through four sections; verify length."""
        path = os.path.join(starter_pack_root(), SKILL_NAME)
        skill = parse_skill(path)
        assert skill is not None
        # The body covers budget, confidentiality, citation, and audit
        # sections. A lightweight stub would be far shorter than 4000 chars.
        assert len(skill.body) > 4000, (
            f"SKILL.md body is shorter than expected ({len(skill.body)} chars); "
            "the four-section walkthrough should be substantive."
        )

    def test_body_walks_through_four_sections(self) -> None:
        """The brief asks for explicit sections on budget, confidentiality,
        citation, and audit. Verify each section header is present in the
        body so a casual reordering does not silently drop one.
        """
        path = os.path.join(starter_pack_root(), SKILL_NAME)
        skill = parse_skill(path)
        assert skill is not None
        body = skill.body
        assert "Per-Workspace Context Budget" in body
        assert "Confidentiality Boundaries" in body
        assert "Citation Format" in body
        assert "Audit Trail Surfacing" in body


class TestToolDeclaration:
    def test_tool_declaration_present(self) -> None:
        path = os.path.join(starter_pack_root(), SKILL_NAME)
        skill = parse_skill(path)
        assert skill is not None
        tools = skill.frontmatter.get("tools")
        assert isinstance(tools, list)
        assert len(tools) == 1
        tool = tools[0]
        assert tool.get("name") == SKILL_TOOL_NAME

    def test_tool_has_full_input_schema(self) -> None:
        path = os.path.join(starter_pack_root(), SKILL_NAME)
        skill = parse_skill(path)
        assert skill is not None
        tool = skill.frontmatter["tools"][0]
        schema = tool.get("input_schema")
        assert isinstance(schema, dict)
        props = schema.get("properties") or {}
        # The brief mandates workspaces + query + total_budget_tokens.
        for required_prop in ("workspaces", "query", "total_budget_tokens"):
            assert required_prop in props, (
                f"input_schema missing property {required_prop!r}; got {sorted(props)}"
            )
        # workspaces must be an array with minItems >= 2.
        ws_schema = props["workspaces"]
        assert ws_schema.get("type") == "array"
        assert ws_schema.get("minItems", 0) >= 2
        # total_budget_tokens default must match the brief (8000).
        assert props["total_budget_tokens"].get("default") == 8000

    def test_tool_required_includes_workspaces_and_query(self) -> None:
        path = os.path.join(starter_pack_root(), SKILL_NAME)
        skill = parse_skill(path)
        assert skill is not None
        tool = skill.frontmatter["tools"][0]
        required = set(tool["input_schema"].get("required") or [])
        assert {"workspaces", "query"} <= required

    def test_tool_has_output_schema(self) -> None:
        path = os.path.join(starter_pack_root(), SKILL_NAME)
        skill = parse_skill(path)
        assert skill is not None
        tool = skill.frontmatter["tools"][0]
        schema = tool.get("output_schema")
        assert isinstance(schema, dict)
        out_props = schema.get("properties") or {}
        # The brief mandates the SynthesisReport shape: summary, findings,
        # citations, audit_trail, effective_confidentiality.
        for expected in (
            "summary",
            "findings",
            "citations",
            "audit_trail",
            "effective_confidentiality",
        ):
            assert expected in out_props


# ---------------------------------------------------------------------------
# Discovery wiring (regression: 5 -> 6 starter skills)
# ---------------------------------------------------------------------------


class TestStarterPackEnumeration:
    def test_list_starter_skill_paths_now_returns_six(self) -> None:
        paths = list_starter_skill_paths()
        assert len(paths) == 6, (
            f"Expected 6 starter skills after adding cross-workspace-synthesis; "
            f"got {len(paths)}: {[os.path.basename(p) for p in paths]}"
        )
        basenames = {os.path.basename(p) for p in paths}
        assert SKILL_NAME in basenames

    def test_discover_in_starter_pack_root_finds_new_skill(self) -> None:
        skills = discover_skills_in_root(starter_pack_root())
        names: Set[str] = {s.name for s in skills}
        assert SKILL_NAME in names, (
            f"Discovery did not find {SKILL_NAME}; saw {sorted(names)}"
        )

    def test_get_skills_for_workspace_surfaces_new_skill(
        self, fake_home: Path
    ) -> None:
        skills = get_skills_for_workspace("acme-news", inheritance_config={})
        names = {s.name for s in skills}
        assert SKILL_NAME in names, (
            f"{SKILL_NAME!r} not visible from acme-news; saw {sorted(names)}"
        )


# ---------------------------------------------------------------------------
# Driver — happy path
# ---------------------------------------------------------------------------


def _policy(level: Confidentiality) -> RoutingPolicy:
    return RoutingPolicy(confidentiality=level)


def _run(coro):
    """Tiny helper so tests don't pull pytest-asyncio in."""
    return asyncio.run(coro)


class TestDriverHappyPath:
    def test_two_workspace_synthesis_returns_structured_report(self) -> None:
        retriever = FakeMemoryRetriever()
        retriever.seed(
            "acme-news",
            [
                _block(
                    "acme-news",
                    "Migration kickoff: 2026-05-12.",
                    0.95,
                    1,
                    document_id="release-2026-05.md",
                ),
                _block(
                    "acme-news",
                    "Owner: engineering lead.",
                    0.85,
                    2,
                    document_id="adr-0042",
                ),
            ],
        )
        retriever.seed(
            "acme-user",
            [
                _block(
                    "acme-user",
                    "Confirmed May for the migration. Priya owns it.",
                    0.9,
                    1,
                    document_id="notes-2026-04-30.md",
                ),
            ],
        )
        llm = FakeLLMBackend()
        audit = FakeAuditWriter()

        report = _run(
            run_cross_workspace_synthesis(
                workspaces=["acme-news", "acme-user"],
                query="Who owns the migration and when does it ship?",
                llm_backend=llm,
                memory=object(),  # the fake retriever does not consult it
                routing_policies={
                    "acme-news": _policy(Confidentiality.PUBLIC),
                    "acme-user": _policy(Confidentiality.PERSONAL),
                },
                total_budget_tokens=8000,
                retriever=retriever,
                audit_writer=audit,
            )
        )

        assert isinstance(report, SynthesisReport)
        assert "migration" in report.summary.lower()
        # Citations come from both workspaces.
        workspaces_cited = {c.workspace for c in report.citations}
        assert workspaces_cited == {"acme-news", "acme-user"}
        assert len(report.findings) == 2
        # Findings carry their citations.
        for finding in report.findings:
            assert isinstance(finding, Finding)
            assert finding.supporting_citations, (
                "every finding must carry at least one citation"
            )
        # Effective confidentiality is the max(participants).
        assert report.effective_confidentiality == "PERSONAL"

    def test_audit_trail_lists_every_required_element(self) -> None:
        retriever = FakeMemoryRetriever()
        retriever.seed(
            "acme-news",
            [_block("acme-news", "fact 1", 0.9, 1, document_id="doc-a.md")],
        )
        retriever.seed(
            "acme-user",
            [_block("acme-user", "fact 2", 0.8, 1, document_id="doc-b.md")],
        )
        llm = FakeLLMBackend()
        audit = FakeAuditWriter()

        report = _run(
            run_cross_workspace_synthesis(
                workspaces=["acme-news", "acme-user"],
                query="What does each workspace say about the deploy?",
                llm_backend=llm,
                memory=object(),
                routing_policies={
                    "acme-news": _policy(Confidentiality.PUBLIC),
                    "acme-user": _policy(Confidentiality.PERSONAL),
                },
                retriever=retriever,
                audit_writer=audit,
                model_id="anthropic/claude-opus-4-8",
            )
        )

        joined = "\n".join(report.audit_trail)
        # Every element the SKILL.md mandates must be visible in the trail.
        assert "Workspaces consulted" in joined
        assert "acme-news" in joined and "acme-user" in joined
        assert "Per-workspace confidentiality" in joined
        assert "Effective confidentiality" in joined
        assert "PERSONAL" in joined
        assert "Tools fired" in joined
        assert "cross_workspace_synthesize" in joined
        assert "three_tier_retrieve_multi" in joined
        assert "llm.complete" in joined
        assert "Model id" in joined
        assert "anthropic/claude-opus-4-8" in joined
        assert "Per-workspace block counts" in joined

    def test_audit_entry_recorded_on_success(self) -> None:
        retriever = FakeMemoryRetriever()
        retriever.seed(
            "acme-news",
            [_block("acme-news", "fact 1", 0.9, 1, document_id="doc-a.md")],
        )
        retriever.seed(
            "acme-user",
            [_block("acme-user", "fact 2", 0.8, 1, document_id="doc-b.md")],
        )
        llm = FakeLLMBackend()
        audit = FakeAuditWriter()

        _run(
            run_cross_workspace_synthesis(
                workspaces=["acme-news", "acme-user"],
                query="x",
                llm_backend=llm,
                memory=object(),
                routing_policies={
                    "acme-news": _policy(Confidentiality.PUBLIC),
                    "acme-user": _policy(Confidentiality.PERSONAL),
                },
                retriever=retriever,
                audit_writer=audit,
            )
        )

        assert len(audit.entries) == 1
        entry = audit.entries[0]
        assert entry.op_type == "cross_workspace_synthesis"
        assert set(entry.workspaces) == {"acme-news", "acme-user"}
        assert entry.outcome == "allowed"
        assert "cross_workspace_synthesize" in entry.tools
        assert "three_tier_retrieve_multi" in entry.tools

    def test_audit_entry_marks_personal_plus_client_confidential_mix(self) -> None:
        """The PERSONAL + CLIENT_CONFIDENTIAL borderline is allowed but the
        substrate records the mix. The driver surfaces the audit flag in the
        success entry's outcome AND in the in-report audit trail."""

        retriever = FakeMemoryRetriever()
        retriever.seed(
            "acme-news",
            [_block("acme-news", "fact 1", 0.9, 1, document_id="cc-doc.md")],
        )
        retriever.seed(
            "acme-user",
            [_block("acme-user", "fact 2", 0.8, 1, document_id="user-doc.md")],
        )
        llm = FakeLLMBackend()
        audit = FakeAuditWriter()

        report = _run(
            run_cross_workspace_synthesis(
                workspaces=["acme-news", "acme-user"],
                query="x",
                llm_backend=llm,
                memory=object(),
                routing_policies={
                    "acme-news": _policy(Confidentiality.CLIENT_CONFIDENTIAL),
                    "acme-user": _policy(Confidentiality.PERSONAL),
                },
                retriever=retriever,
                audit_writer=audit,
            )
        )

        assert audit.entries[0].outcome == "allowed_with_audit"
        joined = "\n".join(report.audit_trail)
        assert "audit required" in joined.lower()
        assert report.effective_confidentiality == "CLIENT_CONFIDENTIAL"

    def test_budget_passes_through_to_retriever(self) -> None:
        retriever = FakeMemoryRetriever()
        retriever.seed(
            "acme-news",
            [_block("acme-news", "x", 0.9, 1, document_id="d.md")],
        )
        retriever.seed(
            "acme-user",
            [_block("acme-user", "y", 0.8, 1, document_id="e.md")],
        )
        llm = FakeLLMBackend()
        audit = FakeAuditWriter()

        _run(
            run_cross_workspace_synthesis(
                workspaces=["acme-news", "acme-user"],
                query="x",
                llm_backend=llm,
                memory=object(),
                routing_policies={
                    "acme-news": _policy(Confidentiality.PUBLIC),
                    "acme-user": _policy(Confidentiality.PERSONAL),
                },
                total_budget_tokens=1234,
                retriever=retriever,
                audit_writer=audit,
            )
        )

        assert len(retriever.calls) == 1
        call = retriever.calls[0]
        assert call["total_budget_tokens"] == 1234
        # Workspaces are forwarded in dedup order.
        assert call["workspaces"] == ["acme-news", "acme-user"]
        assert call["query"] == "x"


# ---------------------------------------------------------------------------
# Driver — confidentiality denial path
# ---------------------------------------------------------------------------


class TestConfidentialityDenial:
    def test_air_gapped_plus_public_raises_confidentiality_error(self) -> None:
        retriever = FakeMemoryRetriever()
        # Even though the retriever is seeded, the gate denies BEFORE
        # any retrieval happens — the seeded blocks should never be read.
        retriever.seed(
            "acme-secrets",
            [_block("acme-secrets", "secret", 0.9, 1, document_id="vault.md")],
        )
        retriever.seed(
            "beta-media",
            [_block("beta-media", "public", 0.9, 1, document_id="press.md")],
        )
        llm = FakeLLMBackend()
        audit = FakeAuditWriter()

        with pytest.raises(ConfidentialityError) as exc_info:
            _run(
                run_cross_workspace_synthesis(
                    workspaces=["acme-secrets", "beta-media"],
                    query="x",
                    llm_backend=llm,
                    memory=object(),
                    routing_policies={
                        "acme-secrets": _policy(Confidentiality.AIR_GAPPED),
                        "beta-media": _policy(Confidentiality.PUBLIC),
                    },
                    retriever=retriever,
                    audit_writer=audit,
                )
            )

        # The error carries the substrate's pairwise verdict.
        assert "AIR_GAPPED" in str(exc_info.value)
        assert exc_info.value.check.allowed is False
        # The retriever was NEVER called — the gate ran first.
        assert retriever.calls == []
        # The LLM was NEVER called either.
        assert llm.requests == []
        # The denial was audited.
        assert len(audit.entries) == 1
        assert audit.entries[0].outcome == "denied"
        assert set(audit.entries[0].workspaces) == {"acme-secrets", "beta-media"}

    def test_client_confidential_plus_public_raises(self) -> None:
        llm = FakeLLMBackend()
        audit = FakeAuditWriter()
        retriever = FakeMemoryRetriever()

        with pytest.raises(ConfidentialityError) as exc_info:
            _run(
                run_cross_workspace_synthesis(
                    workspaces=["acme-news", "beta-media"],
                    query="x",
                    llm_backend=llm,
                    memory=object(),
                    routing_policies={
                        "acme-news": _policy(Confidentiality.CLIENT_CONFIDENTIAL),
                        "beta-media": _policy(Confidentiality.PUBLIC),
                    },
                    retriever=retriever,
                    audit_writer=audit,
                )
            )

        assert "PUBLIC" in str(exc_info.value)
        assert audit.entries[0].outcome == "denied"


# ---------------------------------------------------------------------------
# Driver — input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_single_workspace_raises_value_error(self) -> None:
        """The skill is for ≥2 workspaces; single-workspace queries belong in
        workspace-search-with-citations. The driver rejects them before any
        gate or retrieval runs."""

        llm = FakeLLMBackend()
        audit = FakeAuditWriter()

        with pytest.raises(ValueError, match="at least two"):
            _run(
                run_cross_workspace_synthesis(
                    workspaces=["acme-news"],
                    query="x",
                    llm_backend=llm,
                    memory=object(),
                    routing_policies={
                        "acme-news": _policy(Confidentiality.PUBLIC),
                    },
                    audit_writer=audit,
                )
            )
        # Input validation runs before the audit log opens.
        assert audit.entries == []

    def test_duplicate_workspaces_collapse_to_single_and_raise(self) -> None:
        """`["acme-news", "acme-news"]` should dedup to one workspace and
        therefore raise — the substrate does not silently swallow the typo."""

        llm = FakeLLMBackend()
        audit = FakeAuditWriter()

        with pytest.raises(ValueError, match="at least two"):
            _run(
                run_cross_workspace_synthesis(
                    workspaces=["acme-news", "acme-news"],
                    query="x",
                    llm_backend=llm,
                    memory=object(),
                    routing_policies={
                        "acme-news": _policy(Confidentiality.PUBLIC),
                    },
                    audit_writer=audit,
                )
            )

    def test_zero_total_budget_raises(self) -> None:
        llm = FakeLLMBackend()
        audit = FakeAuditWriter()
        with pytest.raises(ValueError, match="positive"):
            _run(
                run_cross_workspace_synthesis(
                    workspaces=["acme-news", "acme-user"],
                    query="x",
                    llm_backend=llm,
                    memory=object(),
                    routing_policies={
                        "acme-news": _policy(Confidentiality.PUBLIC),
                        "acme-user": _policy(Confidentiality.PERSONAL),
                    },
                    total_budget_tokens=0,
                    audit_writer=audit,
                )
            )
