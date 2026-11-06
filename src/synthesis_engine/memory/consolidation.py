"""Between-session consolidation pass (the "dreaming" pattern).

Reads a session's recent events, asks the configured LLM backend to
extract durable facts as ``(subject, predicate, object)`` triples with
confidence, and writes them into the entity graph with provenance
pointing back at the session.

Designed to be invoked by a scheduled job (Ragenie's routines, a cron
in a synthesis-console deployment, etc.). The function is synchronous
and idempotent: re-running it on the same session re-extracts but the
upsert path deduplicates by ``(workspace, type, name)`` for entities and
appends new dated relations (never deletes existing facts).

The LLM contract is intentionally narrow so any backend that returns
strict JSON works without bespoke parsing per provider.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from ..llm import LLMBackend, LLMRequest, get_llm_backend
from .base import Memory
from .models import (
    AttributeValue,
    Entity,
    Provenance,
    Relation,
)

logger = logging.getLogger(__name__)


_CONSOLIDATION_SYSTEM_PROMPT = """\
You are the consolidation pass for an agent memory system. Read the
session transcript or summary you are given and extract DURABLE facts —
the kind worth remembering after the session ends. Skip pleasantries,
greetings, hedges, and anything that is just the agent thinking aloud.

Return ONLY a JSON object of the form:

  {
    "entities": [
      {"type": "person", "name": "Alice", "attributes": {"role": "engineer"}},
      ...
    ],
    "relations": [
      {
        "from": {"type": "person", "name": "Alice"},
        "to": {"type": "concept", "name": "synthesis engineering"},
        "type": "authored",
        "confidence": 0.95,
        "attributes": {"date": "2026-05-12"}
      },
      ...
    ]
  }

Rules:
  - Use the same "type" + "name" to refer to the same entity across
    facts. Don't invent new entities for the same noun.
  - Confidence is 0.0..1.0 — a hedge.
  - Output JSON only. No prose, no markdown fence.
"""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def consolidate_session(
    memory: Memory,
    session_id: str,
    workspace: str,
    *,
    llm_backend: Optional[LLMBackend] = None,
    model: Optional[str] = None,
    extractor: Optional[Callable[[str], Dict[str, Any]]] = None,
    transcript: Optional[str] = None,
) -> Dict[str, Any]:
    """Distil a session's contents into the entity graph.

    Args:
        memory: The memory backend to read the session from and write
            entities/relations into.
        session_id: Identifier of the session to consolidate.
        workspace: Workspace the consolidated facts belong to.
        llm_backend: Optional explicit backend. Defaults to the cached
            singleton from ``synthesis_engine.llm.get_llm_backend()``.
        model: LLM model id (matches engines.yaml ids). Defaults to the
            default provider's bulk-tier model.
        extractor: Optional override that takes the session transcript
            and returns the extraction JSON directly. Used by tests and
            by deterministic rule-based consolidators that don't need
            an LLM. When None, the LLM backend is invoked.
        transcript: Optional explicit transcript string. When None the
            session payload is rendered into a transcript via
            :func:`render_session_transcript`.

    Returns:
        A summary dict with the entity and relation ids written and the
        raw extraction for downstream inspection.
    """

    session = memory.get_session(session_id)
    if session is None:
        raise ValueError(f"session_id={session_id!r} not found")

    text = transcript if transcript is not None else render_session_transcript(session.payload)
    if not text.strip():
        return {"entities": [], "relations": [], "raw": None}

    if extractor is not None:
        extraction = extractor(text)
    else:
        backend = llm_backend or get_llm_backend()
        if model is None:
            from ..config import get_default_model_for_tier

            model = get_default_model_for_tier("bulk")
        extraction = _llm_extract(backend, model, text)

    provenance = Provenance(
        source=f"consolidation:session={session_id}",
        message_id=session_id,
        confidence=0.9,
    )

    written_entities: List[str] = []
    name_to_id: Dict[tuple, str] = {}

    for raw_entity in extraction.get("entities", []) or []:
        try:
            ent_type = str(raw_entity["type"])
            ent_name = str(raw_entity["name"])
        except (KeyError, TypeError) as exc:
            logger.warning("skipping malformed entity in extraction: %s", exc)
            continue
        # Wrap each attribute with provenance.
        attr_map: Dict[str, AttributeValue] = {}
        for k, v in dict(raw_entity.get("attributes") or {}).items():
            attr_map[str(k)] = AttributeValue(value=v, provenance=provenance)
        ent = memory.upsert_entity(
            Entity(
                workspace=workspace,
                type=ent_type,
                name=ent_name,
                attributes=attr_map,
            )
        )
        if ent.id is None:
            continue
        name_to_id[(ent_type, ent_name)] = str(ent.id)
        written_entities.append(str(ent.id))

    written_relations: List[str] = []
    for raw_rel in extraction.get("relations", []) or []:
        try:
            from_ref = raw_rel["from"]
            to_ref = raw_rel["to"]
            rel_type = str(raw_rel["type"])
        except (KeyError, TypeError) as exc:
            logger.warning("skipping malformed relation in extraction: %s", exc)
            continue
        from_id = _resolve_ref(memory, workspace, from_ref, name_to_id, provenance)
        to_id = _resolve_ref(memory, workspace, to_ref, name_to_id, provenance)
        if from_id is None or to_id is None:
            continue
        confidence = float(raw_rel.get("confidence", 0.7))
        rel_provenance = Provenance(
            source=provenance.source,
            agent_run_id=provenance.agent_run_id,
            message_id=provenance.message_id,
            confidence=max(0.0, min(1.0, confidence)),
        )
        rel = memory.upsert_relation(
            Relation(
                workspace=workspace,
                from_entity=from_id,
                to_entity=to_id,
                type=rel_type,
                attributes=dict(raw_rel.get("attributes") or {}),
                provenance=rel_provenance,
            )
        )
        if rel.id is not None:
            written_relations.append(str(rel.id))

    return {
        "entities": written_entities,
        "relations": written_relations,
        "raw": extraction,
    }


# ---------------------------------------------------------------------------
# Transcript rendering
# ---------------------------------------------------------------------------


def render_session_transcript(payload: Dict[str, Any]) -> str:
    """Render a session payload into a transcript string for the LLM.

    Supports the agent loop's conventional payload shape:

        {
            "messages": [
                {"role": "user",      "content": "..."},
                {"role": "assistant", "content": "..."},
                ...
            ],
            "events": [
                {"type": "tool_call", "tool": "...", "args": {...}},
                {"type": "fact_observed", "subject": "...", ...},
                ...
            ],
            "summary": "..."
        }

    Anything not in the convention is included verbatim under a generic
    "extra" section so a custom payload shape doesn't lose data.
    """

    out: List[str] = []
    messages = payload.get("messages") or []
    if messages:
        out.append("=== messages ===")
        for m in messages:
            role = str(m.get("role", "?"))
            content = str(m.get("content", ""))
            out.append(f"[{role}] {content}")
    events = payload.get("events") or []
    if events:
        out.append("=== events ===")
        for e in events:
            out.append(json.dumps(e, default=str))
    summary = payload.get("summary")
    if summary:
        out.append("=== summary ===")
        out.append(str(summary))
    leftover = {k: v for k, v in payload.items() if k not in {"messages", "events", "summary"}}
    if leftover:
        out.append("=== extra ===")
        out.append(json.dumps(leftover, default=str))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------


def _llm_extract(backend: LLMBackend, model: str, transcript: str) -> Dict[str, Any]:
    request = LLMRequest(
        model=model,
        messages=[
            {"role": "system", "content": _CONSOLIDATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Session transcript:\n\n{transcript}\n\nReturn the JSON object.",
            },
        ],
        temperature=0.0,
        max_tokens=2048,
    )
    response = backend.complete(request)
    return _parse_extraction_json(response.text)


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_extraction_json(text: str) -> Dict[str, Any]:
    """Parse the LLM's response, tolerating accidental code fences."""

    if not text:
        return {"entities": [], "relations": []}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_BLOCK_RE.search(text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                logger.warning("consolidation extraction not valid JSON: %s", exc)
        return {"entities": [], "relations": []}


def _resolve_ref(
    memory: Memory,
    workspace: str,
    ref: Any,
    name_to_id: Dict[tuple, str],
    provenance: Provenance,
):
    """Resolve a ``{type, name}`` ref to a UUID, upserting on the fly."""

    if not isinstance(ref, dict):
        return None
    ent_type = str(ref.get("type", "")).strip()
    ent_name = str(ref.get("name", "")).strip()
    if not ent_type or not ent_name:
        return None
    key = (ent_type, ent_name)
    if key in name_to_id:
        return name_to_id[key]
    # Look up or upsert; the consolidation pass tolerates relations
    # referring to entities the extraction didn't explicitly list, so
    # the agent loop can write "Alice authored synthesis engineering"
    # without first declaring "Alice" as an entity in the same pass.
    existing = memory.get_entity(workspace=workspace, type=ent_type, name=ent_name)
    if existing and existing.id is not None:
        name_to_id[key] = str(existing.id)
        return existing.id
    created = memory.upsert_entity(
        Entity(workspace=workspace, type=ent_type, name=ent_name)
    )
    if created.id is None:
        return None
    name_to_id[key] = str(created.id)
    return created.id
