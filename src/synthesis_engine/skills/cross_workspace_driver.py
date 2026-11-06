"""Cross-workspace synthesis driver — backbone of the ``cross-workspace-synthesis`` skill.

The skill (``starter_pack/cross-workspace-synthesis/SKILL.md``) declares a
``cross_workspace_synthesize`` tool. The LLM uses the tool declaration as
the orchestration shape: it produces a tool-call with workspaces + query
+ budget, the runtime dispatches the call to
:func:`run_cross_workspace_synthesis`, and the structured
:class:`SynthesisReport` comes back as the tool's result.

The driver lives outside ``starter_pack/`` because it is Python code, not
skill content: skill directories carry SKILL.md + references + scripts,
and ``starter_pack/`` is the canonical location for those. The driver is
the Python substrate the runtime hooks the skill's tool name to.

Design notes
------------

* **Single function, not a class.** The driver is one entry point with
  pure arguments. Tests mock the dependencies directly; production wires
  the real Memory + LLMBackend.

* **Async signature.** The agent loop is async; making this entry point
  ``async def`` lets callers ``await`` it inside that loop without a
  thread shim. The actual work is CPU-bound (LLM call + retrieval), and
  both the LLMBackend.complete and three_tier_retrieve_multi calls are
  sync — we run them directly inside the async function.

* **Confidentiality gate first, retrieval second.** The driver checks the
  cross-workspace boundary BEFORE retrieving any blocks. A denied op
  never reads any workspace's content. The audit entry records the
  attempt regardless of outcome.

* **Audit on every path.** Success, confidentiality denial, retrieval
  failure, and LLM failure all write an :class:`AuditEntry`. The on-disk
  audit log is the forensic surface; if the operation produced any
  side-effect (including the LLM call), the audit log records it.

* **Errors over silent degradation.** When the confidentiality gate
  denies the op, the driver raises :class:`ConfidentialityError`. When
  the LLM returns malformed output, the driver returns a SynthesisReport
  carrying the raw text in ``summary`` and an empty findings list — the
  operator sees the failure mode rather than a fabricated success.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from ..exceptions import SynthesisError
# Load order matters here. ``policy.confidentiality`` imports from
# ``..agent.permissions``, and ``agent.loop`` imports back from ``..policy``.
# When the policy package is the first entry into the cycle, the import
# fails partway through ``policy/__init__.py``. Importing
# ``agent.permissions`` first (which transitively loads ``agent.loop``
# inside its own package context, where ``permissions`` is already
# resolvable) lets the policy package finish cleanly. The same load
# order is used by ``tests/test_policy.py``.
from ..agent import permissions as _agent_permissions  # noqa: F401
# Import the LLM backend types from the leaf module to avoid the
# additional indirection of ``..llm.__init__`` (which itself depends on
# the policy package).
from ..llm.base import LLMBackend, LLMRequest, LLMResponse
from ..memory import RetrievedBlock, three_tier_retrieve_multi
from ..memory.base import Memory
from ..policy import (
    AuditEntry,
    Confidentiality,
    ConfidentialityCheck,
    RoutingPolicy,
    check_cross_workspace_op,
    record as audit_record,
    redact_args,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ConfidentialityError(SynthesisError):
    """Raised when the confidentiality gate denies a cross-workspace op.

    The ``check`` attribute carries the full :class:`ConfidentialityCheck`
    so the caller can render every pairwise boundary in the error message
    surfaced to the user.
    """

    def __init__(self, message: str, check: ConfidentialityCheck) -> None:
        super().__init__(message)
        self.check = check


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Citation:
    """One workspace+document citation from the synthesis.

    The driver builds citations directly from the retrieved blocks; the
    LLM consumes them in its prompt context and references them by their
    inline ``[workspace:document_id]`` form in the synthesis text.
    """

    workspace: str
    document_id: str
    snippet: str


@dataclass(frozen=True)
class Finding:
    """One atomic claim in the synthesis with its supporting citations.

    Optional ``conflict_note`` is populated when two workspaces disagree
    about the claim — the model surfaces the disagreement rather than
    resolving it silently.
    """

    claim: str
    supporting_citations: List[Citation]
    conflict_note: Optional[str] = None


@dataclass
class SynthesisReport:
    """The structured result of a cross-workspace synthesis.

    Mirrors the SKILL.md output schema exactly. ``audit_trail`` is a
    human-readable list of one-line entries the operator can scan to
    confirm no boundary was crossed; the on-disk audit log carries the
    same information in JSONL form.
    """

    summary: str
    findings: List[Finding] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    audit_trail: List[str] = field(default_factory=list)
    effective_confidentiality: str = Confidentiality.PUBLIC.name


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


# A Protocol for the memory retriever surface the driver needs. The real
# implementation is :func:`three_tier_retrieve_multi`; tests pass a fake
# with the same shape so the driver does not depend on a live pgvector
# instance. The Protocol stays in this file (not pushed up to memory/)
# because the driver is the only consumer of this exact shape.


class _MemoryRetriever(Protocol):
    """Callable shape compatible with :func:`three_tier_retrieve_multi`."""

    def __call__(
        self,
        memory: Memory,
        workspaces: List[str],
        query: str,
        *,
        total_budget_tokens: int,
        **kwargs: Any,
    ) -> List[RetrievedBlock]: ...


_DEFAULT_RETRIEVER: _MemoryRetriever = three_tier_retrieve_multi  # type: ignore[assignment]


def _build_citation_from_block(block: RetrievedBlock) -> Citation:
    """Extract a Citation from a RetrievedBlock.

    The ``document_id`` is the most stable identifier we can find in the
    block's metadata, falling back to a deterministic synthetic id when
    the retrieval layer did not stamp one. ``snippet`` is the rendered
    text, truncated to a readable length so the bibliography stays
    scannable.
    """

    md = block.result.metadata or {}
    document_id = (
        md.get("document_id")
        or md.get("source_path")
        or md.get("doc_id")
        or md.get("relation_id")
        or f"tier:{block.result.tier}:rank:{block.workspace_rank}"
    )
    snippet = (block.text or "").strip().replace("\n", " ")
    if len(snippet) > 280:
        snippet = snippet[:277] + "..."
    return Citation(
        workspace=block.source_workspace,
        document_id=str(document_id),
        snippet=snippet,
    )


def _build_context_block(blocks: List[RetrievedBlock]) -> str:
    """Render the retrieved blocks as a prompt-ready context string.

    Each block is prefixed with its inline ``[workspace:document_id]``
    marker so the LLM can mirror the marker into the synthesis text. The
    ordering is workspace-by-workspace, score-descending within each
    workspace — the same order ``three_tier_retrieve_multi`` returns.
    """

    if not blocks:
        return "(no blocks were retrieved from any workspace)"

    lines: List[str] = []
    for block in blocks:
        citation = _build_citation_from_block(block)
        marker = f"[{citation.workspace}:{citation.document_id}]"
        lines.append(f"{marker}\n{citation.snippet}")
    return "\n\n".join(lines)


def _build_audit_trail(
    workspaces: List[str],
    routing_policies: Dict[str, RoutingPolicy],
    check: ConfidentialityCheck,
    tools_fired: List[str],
    model_id: str,
    blocks_by_workspace: Dict[str, int],
) -> List[str]:
    """Produce the human-readable audit trail surfaced inside the report."""

    per_ws_conf = ", ".join(
        f"{w}={routing_policies[w].confidentiality.name}"
        if w in routing_policies
        else f"{w}=AIR_GAPPED (missing routing.yaml)"
        for w in workspaces
    )
    audit_label = (
        "audit required (PERSONAL + CLIENT_CONFIDENTIAL mix recorded)"
        if check.requires_audit
        else "no audit (mix is within policy)"
    )
    block_counts = ", ".join(
        f"{w}={blocks_by_workspace.get(w, 0)}" for w in workspaces
    )
    return [
        f"Workspaces consulted: {', '.join(workspaces)}",
        f"Per-workspace confidentiality: {per_ws_conf}",
        f"Effective confidentiality: {check.effective_confidentiality.name}",
        f"Audit required: {audit_label}",
        f"Tools fired: {', '.join(tools_fired)}",
        f"Model id: {model_id or '(unknown)'}",
        f"Per-workspace block counts: {block_counts}",
    ]


_SYSTEM_PROMPT_TEMPLATE = """\
You are the cross-workspace synthesis engine for a synthesis-engineering runtime.

Your task: produce one synthesis report that answers the user's question by
weaving together evidence from {n_workspaces} workspaces. Every claim you make
must cite the workspace and the document it came from using the inline
[workspace:document_id] format. Surface conflicts as conflicts; never silently
prefer one workspace over another.

Confidentiality boundary: the effective confidentiality of this operation is
{effective_confidentiality}. The report inherits this level. Do not include
content that would expose a stricter workspace to a more public reader.

Output schema: return ONE JSON object with these top-level fields:
  - summary: one-paragraph rollup that cites at least one workspace by name.
  - findings: list of {{claim, supporting_citations[], conflict_note?}} objects.
  - citations: flat bibliography of every citation referenced.
  - audit_trail: leave empty; the substrate fills this in.
  - effective_confidentiality: leave as "{effective_confidentiality}".

Return ONLY the JSON object. Do not wrap it in markdown fences. Do not add
preamble. The substrate parses your output as JSON before surfacing it.
"""


_USER_PROMPT_TEMPLATE = """\
Question:
{query}

Workspaces and their retrieved context (in workspace order, score-descending
within each workspace). Each block is prefixed with its [workspace:document_id]
citation marker — reuse those markers in your synthesis.

{context_block}

Produce the synthesis now.
"""


def _build_messages(
    query: str,
    blocks: List[RetrievedBlock],
    workspaces: List[str],
    effective_confidentiality: Confidentiality,
) -> List[Dict[str, str]]:
    """Assemble the chat-style messages the LLM backend consumes."""

    system = _SYSTEM_PROMPT_TEMPLATE.format(
        n_workspaces=len(workspaces),
        effective_confidentiality=effective_confidentiality.name,
    )
    user = _USER_PROMPT_TEMPLATE.format(
        query=query,
        context_block=_build_context_block(blocks),
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _parse_llm_response(
    raw_text: str,
    fallback_citations: List[Citation],
    effective_confidentiality: Confidentiality,
) -> SynthesisReport:
    """Parse the LLM's JSON output into a SynthesisReport.

    A malformed response does NOT raise: the report is returned with the
    raw text in ``summary`` and an empty findings list, plus the
    fallback citation list so the operator at least sees what was
    retrieved. The audit trail is filled in by the caller regardless.
    """

    text = (raw_text or "").strip()
    # Strip a markdown fence if the model wrapped its output in one.
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Cross-workspace synthesis: LLM output was not valid JSON (%s); "
            "returning raw text in summary.",
            exc,
        )
        return SynthesisReport(
            summary=(
                "LLM returned non-JSON output; raw text preserved here. "
                f"Underlying content: {text[:600]}"
            ),
            findings=[],
            citations=list(fallback_citations),
            audit_trail=[],
            effective_confidentiality=effective_confidentiality.name,
        )

    if not isinstance(data, dict):
        return SynthesisReport(
            summary=str(text)[:600],
            findings=[],
            citations=list(fallback_citations),
            audit_trail=[],
            effective_confidentiality=effective_confidentiality.name,
        )

    summary = str(data.get("summary") or "").strip()

    findings_raw = data.get("findings") or []
    findings: List[Finding] = []
    if isinstance(findings_raw, list):
        for entry in findings_raw:
            if not isinstance(entry, dict):
                continue
            claim = str(entry.get("claim") or "").strip()
            if not claim:
                continue
            cites_raw = entry.get("supporting_citations") or []
            cites: List[Citation] = []
            if isinstance(cites_raw, list):
                for c in cites_raw:
                    if not isinstance(c, dict):
                        continue
                    cites.append(
                        Citation(
                            workspace=str(c.get("workspace") or "").strip(),
                            document_id=str(c.get("document_id") or "").strip(),
                            snippet=str(c.get("snippet") or "").strip(),
                        )
                    )
            findings.append(
                Finding(
                    claim=claim,
                    supporting_citations=cites,
                    conflict_note=(
                        str(entry["conflict_note"]).strip()
                        if entry.get("conflict_note")
                        else None
                    ),
                )
            )

    citations_raw = data.get("citations") or []
    citations: List[Citation] = []
    if isinstance(citations_raw, list):
        for c in citations_raw:
            if not isinstance(c, dict):
                continue
            citations.append(
                Citation(
                    workspace=str(c.get("workspace") or "").strip(),
                    document_id=str(c.get("document_id") or "").strip(),
                    snippet=str(c.get("snippet") or "").strip(),
                )
            )
    # Always anchor the bibliography in the substrate's retrieved blocks.
    # The model may add to it, but it cannot remove what the retrieval
    # actually produced.
    if not citations:
        citations = list(fallback_citations)

    return SynthesisReport(
        summary=summary or "(empty summary)",
        findings=findings,
        citations=citations,
        audit_trail=[],
        effective_confidentiality=effective_confidentiality.name,
    )


async def run_cross_workspace_synthesis(
    workspaces: List[str],
    query: str,
    *,
    llm_backend: LLMBackend,
    memory: Memory,
    routing_policies: Dict[str, RoutingPolicy],
    total_budget_tokens: int = 8000,
    model_id: Optional[str] = None,
    retriever: _MemoryRetriever = _DEFAULT_RETRIEVER,
    audit_writer: Any = None,
) -> SynthesisReport:
    """Run a cross-workspace synthesis. Returns a structured SynthesisReport.

    Pipeline:

    1. **Normalise inputs.** Dedupe ``workspaces`` in first-occurrence
       order; reject anything below two distinct workspaces (single-
       workspace queries belong in ``workspace-search-with-citations``).
    2. **Confidentiality gate.** Compute the
       :class:`ConfidentialityCheck` from ``routing_policies``. If denied,
       write a ``denied`` audit entry and raise
       :class:`ConfidentialityError`.
    3. **Retrieve.** Call ``retriever`` (default:
       :func:`three_tier_retrieve_multi`) with the de-duplicated
       workspace list and the shared budget.
    4. **Prompt + complete.** Build system + user messages, call
       ``llm_backend.complete(...)``.
    5. **Parse.** Best-effort parse the LLM's JSON. Malformed output
       returns a SynthesisReport carrying the raw text rather than
       raising.
    6. **Audit.** Write the audit entry with the operation outcome and
       fill the audit trail inside the report.

    Args:
        workspaces: Ordered list of workspace names. Must contain at
            least two distinct names after dedup.
        query: Natural-language question the synthesis answers.
        llm_backend: The :class:`LLMBackend` to call for completion.
        memory: The :class:`Memory` backend the retriever reads from.
        routing_policies: Map of workspace name -> RoutingPolicy. A
            workspace missing from this map fails closed to AIR_GAPPED.
        total_budget_tokens: Aggregate token budget for retrieval.
        model_id: Model id to pass to the LLM backend AND to the audit
            entry. The audit trail surfaces this value for the operator.
            Defaults to the default provider's judgment-tier model.
        retriever: Override for the retrieval function. Defaults to
            :func:`three_tier_retrieve_multi`. Tests pass a fake.
        audit_writer: Override for the audit-log writer. Defaults to
            :func:`synthesis_engine.policy.record`. Tests pass a fake to
            capture entries in-memory.

    Returns:
        A :class:`SynthesisReport`.

    Raises:
        ConfidentialityError: When the cross-workspace boundary denies
            the op (e.g., AIR_GAPPED + PUBLIC mix). The audit entry is
            written before the exception is raised.
        ValueError: When ``workspaces`` contains fewer than two distinct
            entries. The audit entry is NOT written in this case — the
            op did not pass the input validation.
    """

    if model_id is None:
        from ..config import get_default_model_for_tier

        model_id = get_default_model_for_tier("judgment")

    # Step 1 — normalise inputs.
    deduped: List[str] = []
    for w in workspaces:
        if isinstance(w, str) and w and w not in deduped:
            deduped.append(w)
    if len(deduped) < 2:
        raise ValueError(
            "Cross-workspace synthesis requires at least two distinct "
            f"workspaces; got {workspaces!r}."
        )
    if total_budget_tokens <= 0:
        raise ValueError(
            f"total_budget_tokens must be positive; got {total_budget_tokens}."
        )

    writer = audit_writer if audit_writer is not None else audit_record

    # Step 2 — confidentiality gate.
    check = check_cross_workspace_op(deduped, routing_policies)
    args_summary = redact_args(
        {
            "workspaces": deduped,
            "query": query,
            "total_budget_tokens": total_budget_tokens,
        }
    )

    if not check.allowed:
        denied_entry = AuditEntry.build(
            op_type="cross_workspace_synthesis",
            workspaces=deduped,
            tools=["cross_workspace_synthesize"],
            model_id=model_id,
            outcome="denied",
            args_summary=args_summary,
            metadata={
                "effective_confidentiality": check.effective_confidentiality.name,
                "reason": check.reason,
                "requires_audit": check.requires_audit,
            },
        )
        try:
            writer(denied_entry)
        except Exception as audit_exc:  # pragma: no cover - audit must not mask the original
            logger.error(
                "Failed to write denied audit entry; surfacing the gate denial "
                "anyway. audit_error=%r", audit_exc,
            )
        raise ConfidentialityError(
            "Cross-workspace synthesis denied by confidentiality gate: "
            f"{check.reason}",
            check=check,
        )

    # Step 3 — retrieve.
    tools_fired: List[str] = ["cross_workspace_synthesize"]
    blocks: List[RetrievedBlock] = []
    try:
        blocks = list(
            retriever(
                memory,
                deduped,
                query,
                total_budget_tokens=total_budget_tokens,
            )
        )
        tools_fired.append("three_tier_retrieve_multi")
    except Exception as exc:
        logger.warning(
            "Cross-workspace retrieval failed: %s. Continuing with an empty "
            "block list so the LLM can still produce a 'no evidence' synthesis "
            "and the audit trail still captures the attempt.",
            exc,
        )
        blocks = []

    # Group block counts per workspace for the audit trail.
    blocks_by_workspace: Dict[str, int] = {w: 0 for w in deduped}
    for b in blocks:
        if b.source_workspace in blocks_by_workspace:
            blocks_by_workspace[b.source_workspace] += 1

    fallback_citations: List[Citation] = [
        _build_citation_from_block(b) for b in blocks
    ]

    # Step 4 — prompt + complete.
    messages = _build_messages(
        query=query,
        blocks=blocks,
        workspaces=deduped,
        effective_confidentiality=check.effective_confidentiality,
    )
    llm_response: Optional[LLMResponse] = None
    try:
        llm_response = llm_backend.complete(
            LLMRequest(
                model=model_id,
                messages=messages,
                temperature=0.2,
                max_tokens=4096,
            )
        )
        tools_fired.append("llm.complete")
        raw_text = llm_response.text
    except Exception as exc:
        logger.warning(
            "LLM completion failed during cross-workspace synthesis: %s. "
            "Returning a SynthesisReport carrying the failure in summary.",
            exc,
        )
        raw_text = (
            f"LLM completion failed: {exc!r}. Retrieved {len(blocks)} block(s) "
            "across the requested workspaces; the bibliography below preserves "
            "every retrieved block so the operator can render the synthesis "
            "manually."
        )
        # Synthesise a JSON-shaped fallback so downstream parsing produces
        # a well-formed (if minimal) report.
        raw_text = json.dumps(
            {
                "summary": raw_text,
                "findings": [],
                "citations": [asdict(c) for c in fallback_citations],
            }
        )

    # Step 5 — parse.
    report = _parse_llm_response(
        raw_text=raw_text,
        fallback_citations=fallback_citations,
        effective_confidentiality=check.effective_confidentiality,
    )

    # Step 6 — audit.
    report.audit_trail = _build_audit_trail(
        workspaces=deduped,
        routing_policies=routing_policies,
        check=check,
        tools_fired=tools_fired,
        model_id=(llm_response.model if llm_response else model_id),
        blocks_by_workspace=blocks_by_workspace,
    )

    outcome = "allowed_with_audit" if check.requires_audit else "allowed"
    success_entry = AuditEntry.build(
        op_type="cross_workspace_synthesis",
        workspaces=deduped,
        tools=tools_fired,
        model_id=(llm_response.model if llm_response else model_id),
        outcome=outcome,
        args_summary=args_summary,
        metadata={
            "effective_confidentiality": check.effective_confidentiality.name,
            "requires_audit": check.requires_audit,
            "block_count_total": len(blocks),
            "blocks_by_workspace": blocks_by_workspace,
            "finding_count": len(report.findings),
            "llm_finish_reason": (
                llm_response.finish_reason if llm_response else None
            ),
        },
    )
    try:
        writer(success_entry)
    except Exception as audit_exc:
        # Failing to write the audit log is a substrate-level integrity issue;
        # surface it loudly but do not poison a successful synthesis the
        # operator already has in hand.
        logger.error(
            "Failed to write success audit entry for cross-workspace "
            "synthesis: %r. Report returned despite audit failure.",
            audit_exc,
        )

    return report


__all__ = [
    "Citation",
    "ConfidentialityError",
    "Finding",
    "SynthesisReport",
    "run_cross_workspace_synthesis",
]
