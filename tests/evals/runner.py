"""Eval runner — execute the YAML cases under ``tests/evals/cases/`` and
emit a markdown scorecard.

Usage:

    python -m tests.evals.runner [--quick] [--filter SUBSTRING] [--output PATH]

Design notes
------------

The runner is deliberately substrate-light. Each case is a YAML file with
the following minimum shape:

    id:          unique id within the suite
    category:    retrieval | tool_selection | refusal | multi_step_planning
    description: human-readable purpose
    prompt:      the user prompt under test (string)
    evaluator:   one of: keyword_match | citation_match | refusal_match |
                          tool_match | json_schema | exact
    expected:    payload for the evaluator (shape depends on evaluator)
    fixture:     optional path under ``tests/evals/fixtures/`` whose content
                 the case relies on (the runner makes the path available
                 to the evaluator as ``case.fixture_path``).

The runner does NOT make live LLM calls by default. Each case can declare
either an ``inline_response`` (the literal text the agent would produce —
used for deterministic harness tests) or a ``live: true`` flag (which is
skipped under ``--quick`` and skipped entirely if no API key is configured,
so the eval suite stays runnable in CI).

The scorecard is markdown that the Makefile target writes to
``tests/evals/last-scorecard.md``.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make ``src/`` importable so eval code can use the substrate types.
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]  # tests/evals/runner.py → tests/evals → tests → repo
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _maybe_yaml():
    try:
        import yaml  # type: ignore
        return yaml
    except ImportError:  # pragma: no cover
        return None


CASES_DIR = _ROOT / "tests" / "evals" / "cases"
REGRESSIONS_DIR = _ROOT / "tests" / "evals" / "regressions"
FIXTURES_DIR = _ROOT / "tests" / "evals" / "fixtures"
DEFAULT_OUTPUT = _ROOT / "tests" / "evals" / "last-scorecard.md"


# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RegressionMeta:
    """Optional metadata attached to a regression eval case.

    A regression case captures a previously-fixed bug as an eval so
    the harness fails when the bug re-emerges. The metadata records:

    * ``bug_id`` — a short stable identifier (the eval's own ``id``
      is usually the same string, but the explicit field documents
      intent).
    * ``description`` — one-paragraph description of the bug.
    * ``fix_commit`` — the commit (or PR) where the bug was fixed.
      Empty when the bug is still open or the case was authored
      ahead of the fix.
    * ``severity`` — one of "low", "medium", "high", "critical".
      Default "high" so a re-emergence catches reviewer attention.
    """

    bug_id: str
    description: str
    fix_commit: str = ""
    severity: str = "high"

    @classmethod
    def from_raw(cls, raw: Any) -> Optional["RegressionMeta"]:
        if not isinstance(raw, dict):
            return None
        bug_id = str(raw.get("bug_id") or "")
        if not bug_id:
            return None
        return cls(
            bug_id=bug_id,
            description=str(raw.get("description") or ""),
            fix_commit=str(raw.get("fix_commit") or ""),
            severity=str(raw.get("severity") or "high"),
        )


@dataclasses.dataclass
class EvalCase:
    """A single eval case loaded from a YAML file."""

    id: str
    category: str
    description: str
    prompt: str
    evaluator: str
    expected: Any
    fixture: Optional[str] = None
    inline_response: Optional[str] = None
    live: bool = False
    quick: bool = True  # included in eval-quick by default
    source_path: Optional[Path] = None
    regression: Optional[RegressionMeta] = None

    @property
    def fixture_path(self) -> Optional[Path]:
        if self.fixture:
            return FIXTURES_DIR / self.fixture
        return None

    @property
    def is_regression(self) -> bool:
        return self.regression is not None


def load_cases(filter_substring: Optional[str] = None) -> List[EvalCase]:
    """Load every YAML case under ``CASES_DIR`` and ``REGRESSIONS_DIR``.

    Both directories are walked; the regression directory's cases are
    flagged via the ``regression:`` block in their YAML, which the
    runner uses to surface failures more prominently in the scorecard.
    """

    yaml = _maybe_yaml()
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to run the eval suite. "
            "Install it with: pip install PyYAML"
        )

    cases: List[EvalCase] = []

    def _load_dir(root: Path, default_category: Optional[str]) -> None:
        if not root.exists():
            return
        for path in sorted(root.rglob("*.yaml")):
            try:
                with path.open("r", encoding="utf-8") as fp:
                    raw = yaml.safe_load(fp) or {}
            except Exception as exc:
                print(f"[warn] skipping {path}: {exc}", file=sys.stderr)
                continue

            # Category precedence:
            #   1. explicit `category:` in the file
            #   2. caller-supplied default (regressions land in
            #      "regression" by default so the scorecard groups them)
            #   3. parent directory name (legacy)
            category = (
                raw.get("category")
                or default_category
                or path.parent.name
            )
            try:
                case = EvalCase(
                    id=raw["id"],
                    category=category,
                    description=raw.get("description", ""),
                    prompt=raw["prompt"],
                    evaluator=raw["evaluator"],
                    expected=raw.get("expected"),
                    fixture=raw.get("fixture"),
                    inline_response=raw.get("inline_response"),
                    live=bool(raw.get("live", False)),
                    quick=bool(raw.get("quick", True)),
                    source_path=path,
                    regression=RegressionMeta.from_raw(raw.get("regression")),
                )
            except KeyError as exc:
                print(
                    f"[warn] case at {path} missing required field {exc}",
                    file=sys.stderr,
                )
                continue

            if filter_substring and filter_substring.lower() not in case.id.lower():
                continue
            cases.append(case)

    _load_dir(CASES_DIR, default_category=None)
    _load_dir(REGRESSIONS_DIR, default_category="regression")
    return cases


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class EvalResult:
    case_id: str
    category: str
    passed: bool
    score: float  # 0..1
    detail: str
    duration_s: float
    skipped: bool = False
    skip_reason: str = ""
    regression: Optional[RegressionMeta] = None

    @property
    def is_regression(self) -> bool:
        return self.regression is not None


def _evaluator_keyword_match(case: EvalCase, response: str) -> Tuple[bool, float, str]:
    """Pass if every keyword in ``expected.must_contain`` appears in the response.

    Case-insensitive. Optionally fail if any keyword in
    ``expected.must_not_contain`` appears.
    """

    expected = case.expected or {}
    must_contain = [str(x) for x in expected.get("must_contain", [])]
    must_not_contain = [str(x) for x in expected.get("must_not_contain", [])]
    text = (response or "").lower()

    missing = [k for k in must_contain if k.lower() not in text]
    forbidden = [k for k in must_not_contain if k.lower() in text]
    total = max(len(must_contain), 1)
    matched = total - len(missing)
    score = matched / total if must_contain else (0.0 if forbidden else 1.0)
    if forbidden:
        return False, score * 0.5, f"forbidden keywords present: {forbidden}"
    if missing:
        return False, score, f"missing keywords: {missing}"
    return True, 1.0, "all keywords matched"


def _evaluator_citation_match(case: EvalCase, response: str) -> Tuple[bool, float, str]:
    """Pass if the response cites every required chunk id.

    The expected shape:

        expected:
          required_citations: ["chunk-1", "chunk-3"]
          forbidden_citations: ["chunk-99"]  # optional
    """

    expected = case.expected or {}
    required = [str(x) for x in expected.get("required_citations", [])]
    forbidden = [str(x) for x in expected.get("forbidden_citations", [])]
    text = response or ""

    missing = [r for r in required if r not in text]
    bad = [f for f in forbidden if f in text]
    if bad:
        return False, 0.0, f"forbidden citations present: {bad}"
    if missing:
        return False, 1.0 - (len(missing) / max(len(required), 1)), f"missing citations: {missing}"
    return True, 1.0, "all citations present"


def _evaluator_refusal_match(case: EvalCase, response: str) -> Tuple[bool, float, str]:
    """Pass if the response is a refusal.

    Heuristic: the response contains at least one phrase from
    ``expected.refusal_markers`` (default set if empty) AND does NOT contain
    any phrase from ``expected.must_not_contain`` (e.g., the data the user
    is trying to extract).
    """

    expected = case.expected or {}
    markers = expected.get("refusal_markers") or [
        "i can't", "i cannot", "i won't", "i'm not able to",
        "unable to", "not appropriate", "decline", "refuse",
        "i don't have", "outside the scope", "i'm sorry",
    ]
    forbidden = expected.get("must_not_contain", [])
    text = (response or "").lower()

    has_refusal = any(str(m).lower() in text for m in markers)
    bad = [f for f in forbidden if str(f).lower() in text]
    if bad:
        return False, 0.0, f"response leaked forbidden content: {bad}"
    if not has_refusal:
        return False, 0.0, "no refusal marker detected"
    return True, 1.0, "refusal detected"


def _evaluator_tool_match(case: EvalCase, response: str) -> Tuple[bool, float, str]:
    """Pass if the response is a JSON tool call that selects the expected tool.

    The eval expects ``response`` to be a JSON string with a ``tool`` key
    matching ``expected.tool``. Optionally checks ``expected.required_args``.
    """

    expected = case.expected or {}
    tool_name = expected.get("tool")
    required_args = expected.get("required_args", [])

    try:
        payload = json.loads(response)
    except (TypeError, json.JSONDecodeError) as exc:
        return False, 0.0, f"response is not JSON: {exc}"

    chosen = payload.get("tool") or payload.get("name")
    if chosen != tool_name:
        return False, 0.0, f"chose tool {chosen!r}, expected {tool_name!r}"

    missing_args = [a for a in required_args if a not in (payload.get("arguments") or payload.get("args") or {})]
    if missing_args:
        return False, 0.5, f"missing required args: {missing_args}"
    return True, 1.0, f"chose {tool_name} with all required args"


def _evaluator_exact(case: EvalCase, response: str) -> Tuple[bool, float, str]:
    expected = (case.expected or {}).get("text", "")
    if (response or "").strip() == str(expected).strip():
        return True, 1.0, "exact match"
    return False, 0.0, "exact mismatch"


_EVALUATORS = {
    "keyword_match": _evaluator_keyword_match,
    "citation_match": _evaluator_citation_match,
    "refusal_match": _evaluator_refusal_match,
    "tool_match": _evaluator_tool_match,
    "exact": _evaluator_exact,
}


# ---------------------------------------------------------------------------
# Case execution
# ---------------------------------------------------------------------------


def _resolve_response(case: EvalCase) -> Tuple[Optional[str], Optional[str]]:
    """Return (response_text, skip_reason).

    The harness prefers ``inline_response`` (deterministic). When ``live: true``
    and an API key is available, a live LLM call is attempted; otherwise
    the case is skipped with an informative reason.
    """

    if case.inline_response is not None:
        return case.inline_response, None

    if not case.live:
        return None, "no inline_response and live=false"

    # Live path is intentionally minimal: we route through the substrate
    # LLM backend so the OTEL spans + cache_control are exercised end-to-end.
    has_keys = any(
        os.environ.get(k) for k in (
            "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY",
        )
    )
    if not has_keys:
        return None, "live case skipped — no API key in environment"

    try:
        from synthesis_engine.config import get_default_model_for_tier  # type: ignore
        from synthesis_engine.llm import get_llm_backend  # type: ignore
        from synthesis_engine.llm.base import LLMRequest  # type: ignore
    except Exception as exc:
        return None, f"live case skipped — backend unavailable: {exc}"

    try:
        backend = get_llm_backend()
        # RAGBOT_EVAL_MODEL overrides; otherwise the default provider's bulk
        # tier from engines.yaml.
        req = LLMRequest(
            model=os.environ.get("RAGBOT_EVAL_MODEL") or get_default_model_for_tier("bulk"),
            messages=[{"role": "user", "content": case.prompt}],
            max_tokens=512,
            temperature=0.0,
            api_key=None,
            extra={},
        )
        resp = backend.complete(req)
        return resp.text, None
    except Exception as exc:
        return None, f"live call failed: {exc}"


def run_case(case: EvalCase, *, quick: bool = False) -> EvalResult:
    started = time.monotonic()

    if quick and not case.quick:
        return EvalResult(
            case_id=case.id, category=case.category, passed=False, score=0.0,
            detail="excluded from quick subset", duration_s=0.0,
            skipped=True, skip_reason="not in quick subset",
            regression=case.regression,
        )

    response, skip_reason = _resolve_response(case)
    if response is None:
        return EvalResult(
            case_id=case.id, category=case.category, passed=False, score=0.0,
            detail=skip_reason or "no response", duration_s=time.monotonic() - started,
            skipped=True, skip_reason=skip_reason or "no response",
            regression=case.regression,
        )

    evaluator = _EVALUATORS.get(case.evaluator)
    if evaluator is None:
        return EvalResult(
            case_id=case.id, category=case.category, passed=False, score=0.0,
            detail=f"unknown evaluator {case.evaluator!r}",
            duration_s=time.monotonic() - started,
            regression=case.regression,
        )

    passed, score, detail = evaluator(case, response)
    return EvalResult(
        case_id=case.id, category=case.category, passed=passed, score=score,
        detail=detail, duration_s=time.monotonic() - started,
        regression=case.regression,
    )


# ---------------------------------------------------------------------------
# Scorecard rendering
# ---------------------------------------------------------------------------


def render_scorecard(
    results: List[EvalResult],
    *,
    quick: bool,
    started_at: str,
    duration_s: float,
) -> str:
    # Partition regression cases out so they get their own section at
    # the top of the scorecard — any re-emerged bug is urgent.
    regression_results = [r for r in results if r.is_regression]
    standard_results = [r for r in results if not r.is_regression]

    by_cat: Dict[str, List[EvalResult]] = {}
    for r in standard_results:
        by_cat.setdefault(r.category, []).append(r)

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    skipped = sum(1 for r in results if r.skipped)
    failed = total - passed - skipped
    pass_rate = (passed / max(total - skipped, 1)) * 100.0

    reg_failed = sum(
        1 for r in regression_results if not r.passed and not r.skipped
    )

    lines: List[str] = []
    lines.append("# Synthesis Engine — Eval Scorecard")
    lines.append("")
    lines.append(f"- **Run started:** {started_at}")
    lines.append(f"- **Mode:** {'quick' if quick else 'full'}")
    lines.append(f"- **Total cases:** {total}")
    lines.append(f"- **Passed:** {passed}")
    lines.append(f"- **Failed:** {failed}")
    lines.append(f"- **Skipped:** {skipped}")
    lines.append(f"- **Pass rate (excl. skipped):** {pass_rate:.1f}%")
    lines.append(f"- **Wall time:** {duration_s:.2f}s")
    if regression_results:
        lines.append(
            f"- **Regression cases:** {len(regression_results)} "
            f"({reg_failed} failed)"
        )
    lines.append("")

    # Regressions group goes first when present so a failure is
    # impossible to miss in a quick scan.
    if regression_results:
        urgency = "URGENT — regressions failed" if reg_failed else "all regressions held"
        lines.append(f"## Regression cases ({urgency})")
        lines.append("")
        lines.append(
            "| Case | Status | Bug ID | Severity | Fix Commit | Detail |"
        )
        lines.append(
            "|------|--------|--------|----------|------------|--------|"
        )
        for r in sorted(regression_results, key=lambda x: x.case_id):
            status = (
                "skip" if r.skipped else ("pass" if r.passed else "**FAIL**")
            )
            meta = r.regression
            assert meta is not None  # narrowed by is_regression
            commit = meta.fix_commit or "(none)"
            detail = r.detail.replace('|', '\\|')
            lines.append(
                f"| `{r.case_id}` | {status} | `{meta.bug_id}` "
                f"| {meta.severity} | `{commit}` | {detail} |"
            )
        lines.append("")

    lines.append("## Results by category")
    lines.append("")

    for category, cat_results in sorted(by_cat.items()):
        cat_passed = sum(1 for r in cat_results if r.passed)
        cat_skipped = sum(1 for r in cat_results if r.skipped)
        cat_total = len(cat_results)
        lines.append(
            f"### {category} — {cat_passed}/{cat_total - cat_skipped} passed"
            f"{f' ({cat_skipped} skipped)' if cat_skipped else ''}"
        )
        lines.append("")
        lines.append("| Case | Status | Score | Detail | Duration |")
        lines.append("|------|--------|-------|--------|----------|")
        for r in sorted(cat_results, key=lambda x: x.case_id):
            status = (
                "skip" if r.skipped
                else ("pass" if r.passed else "fail")
            )
            lines.append(
                f"| `{r.case_id}` | {status} | {r.score:.2f} "
                f"| {r.detail.replace('|', '\\|')} | {r.duration_s:.3f}s |"
            )
        lines.append("")

    lines.append("---")
    lines.append("Generated by `python -m tests.evals.runner`")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the synthesis_engine eval suite.")
    parser.add_argument("--quick", action="store_true", help="Run only quick-eligible cases.")
    parser.add_argument("--filter", default=None, help="Case-id substring filter.")
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT), help="Markdown scorecard output path.",
    )
    parser.add_argument(
        "--regressions-only",
        action="store_true",
        help="Run only the regression cases under tests/evals/regressions/.",
    )
    args = parser.parse_args(argv)

    started_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    started = time.monotonic()

    try:
        cases = load_cases(filter_substring=args.filter)
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    if args.regressions_only:
        cases = [c for c in cases if c.is_regression]

    if not cases:
        print("[warn] no eval cases found.", file=sys.stderr)

    results = [run_case(c, quick=args.quick) for c in cases]
    scorecard = render_scorecard(
        results,
        quick=args.quick,
        started_at=started_at,
        duration_s=time.monotonic() - started,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(scorecard, encoding="utf-8")

    # Also print a compact summary so CI logs show the verdict without
    # cat'ing the markdown.
    failed = sum(1 for r in results if not r.passed and not r.skipped)
    total = len(results)
    skipped = sum(1 for r in results if r.skipped)
    reg_failed = sum(
        1 for r in results
        if r.is_regression and not r.passed and not r.skipped
    )
    summary_parts = [
        f"{total - failed - skipped} passed",
        f"{failed} failed",
        f"{skipped} skipped",
    ]
    if reg_failed:
        summary_parts.append(f"REGRESSIONS_FAILED={reg_failed}")
    print(f"eval: {', '.join(summary_parts)} → {output}")
    # Non-zero exit only if a non-skipped case failed.
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
