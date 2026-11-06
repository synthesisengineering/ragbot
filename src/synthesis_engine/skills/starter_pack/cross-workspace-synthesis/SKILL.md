---
name: cross-workspace-synthesis
description: "Produce a synthesis report from 2+ workspaces with explicit per-workspace citations and a visible audit trail. Use when asked to: synthesize, compare workspaces, cross-reference, multi-workspace research, cite from multiple workspaces."
license: "Apache-2.0"
scope: universal
depends_on: []
metadata:
  author: "Synthesis Engineering"
  version: "1.0.0"
  source_repo: "github.com/synthesisengineering/ragbot"
  source_type: "starter_pack"
tools:
  - name: cross_workspace_synthesize
    description: "Run a synthesis-engine retrieval across two or more workspaces under a shared token budget, then produce a single structured SynthesisReport whose every finding cites the workspace and document it came from. The tool also records an entry in the cross-workspace audit log so the operator has a forensic trail of which workspaces were consulted, which tools fired, which model produced the synthesis, and what the effective confidentiality of the operation was. Use this tool whenever the user asks for an answer that crosses a workspace boundary; do not use it for single-workspace queries (those go through workspace-search-with-citations)."
    input_schema:
      type: object
      properties:
        workspaces:
          type: array
          description: "Ordered list of workspace names to synthesize across. Must contain at least two distinct names. Duplicates are removed in first-occurrence order. Each workspace must have a loaded routing.yaml policy; workspaces with no policy fail closed to AIR_GAPPED and the operation is denied."
          items:
            type: string
            minLength: 1
          minItems: 2
        query:
          type: string
          description: "Natural-language question the synthesis report should answer. Pass the user's question verbatim; the retriever rewrites internally."
          minLength: 1
        total_budget_tokens:
          type: integer
          description: "Aggregate token budget for the retrieval step across all workspaces. The budget allocator starts with an equal split and redistributes unused budget from sparse workspaces to data-rich ones."
          default: 8000
          minimum: 1000
          maximum: 64000
      required:
        - workspaces
        - query
    output_schema:
      type: object
      description: "A SynthesisReport: structured object the agent renders into prose. The model SHOULD NOT mutate the schema; the schema is the operator's audit surface."
      properties:
        summary:
          type: string
          description: "One-paragraph rollup of the synthesis. Cites at least one workspace by name in the body."
        findings:
          type: array
          description: "Ordered findings extracted from the cross-workspace retrieval. Each finding is one atomic claim with the citations that support it."
          items:
            type: object
            properties:
              claim:
                type: string
                description: "Atomic claim, verbatim or lightly paraphrased from the synthesis."
              supporting_citations:
                type: array
                description: "Citations that support this finding. Each citation must reference at least one workspace; multi-workspace claims surface every workspace that contributed."
                items:
                  type: object
                  properties:
                    workspace:
                      type: string
                    document_id:
                      type: string
                    snippet:
                      type: string
                  required:
                    - workspace
                    - document_id
              conflict_note:
                type: string
                description: "Optional. Populated when two workspaces disagree about this finding. The model surfaces the disagreement instead of resolving it silently."
            required:
              - claim
              - supporting_citations
        citations:
          type: array
          description: "Flat list of every citation referenced by any finding. Suitable for the end-of-report bibliography block."
          items:
            type: object
            properties:
              workspace:
                type: string
              document_id:
                type: string
              snippet:
                type: string
            required:
              - workspace
              - document_id
              - snippet
        audit_trail:
          type: array
          description: "Operator-facing 'show your work' trail. Lists the workspaces consulted, the tools that fired, the model that produced the synthesis, and the effective confidentiality of the operation. Read this from top to bottom and confirm no boundary was crossed."
          items:
            type: string
        effective_confidentiality:
          type: string
          description: "The strictest confidentiality among the participating workspaces. PUBLIC, PERSONAL, CLIENT_CONFIDENTIAL, or AIR_GAPPED. Determines who the report is safe to share with."
          enum:
            - PUBLIC
            - PERSONAL
            - CLIENT_CONFIDENTIAL
            - AIR_GAPPED
      required:
        - summary
        - findings
        - citations
        - audit_trail
        - effective_confidentiality
---

# Cross-Workspace Synthesis

This skill takes a question and a list of workspaces, retrieves grounded context from each workspace under a shared token budget, and produces one synthesis report whose every claim cites the workspace and document it came from. The report ends with an audit trail the operator can scan to confirm no confidentiality boundary was crossed.

The skill exists because single-workspace search is necessary but insufficient for the kind of work synthesis engineers actually do. A real question — "how does our migration plan compare across our news and user workspaces?", "what do our acme-news playbooks say that beta-media's playbooks don't?", "where do the two workspaces agree on the answer, and where do they disagree?" — spans more than one workspace by construction. Stitching the answer together by running two separate searches and pasting the results loses three things the model is responsible for: a unified token budget, an explicit confidentiality verdict, and a record of which workspaces touched which output. This skill gives the model all three as a single operation.

## When to Apply

- The user asks a question that mentions two or more workspaces by name.
- The user asks for a comparison, a cross-reference, or a join across workspaces.
- The user asks "where do my workspaces agree" or "where do they disagree" about a topic.
- The user asks for a synthesis that draws on multiple bodies of grounded content the workspaces hold.
- The user is using a multi-workspace agent loop and the active workspace set has more than one entry.

## When NOT to Apply

- The question is single-workspace. Route to `workspace-search-with-citations` instead — its output schema is simpler and the audit overhead is unnecessary.
- The question is general knowledge no workspace can answer. The model answers from training and says so.
- The user wants creative generation rather than retrieval-grounded synthesis. Route to `draft-and-revise`.
- The active workspace set includes any AIR_GAPPED workspace alongside any other workspace. The operation will be denied at the confidentiality gate; tell the user explicitly that AIR_GAPPED data cannot mix and ask whether they want the AIR_GAPPED workspace queried alone.

## Section 1 — Per-Workspace Context Budget

The retrieval step allocates a single `total_budget_tokens` figure across every participating workspace. The default is 8000 tokens, which is generous enough for a 4-workspace synthesis on a Claude Opus 4.7 context window and tight enough that the model is forced to think about ranking.

The allocator's algorithm:

1. **Equal split first.** Every workspace starts with `total_budget_tokens / N` tokens, clamped down to a floor (typically 800 tokens) per workspace.
2. **Demand measurement.** The retriever produces candidate blocks per workspace. Each block has an estimated token cost. The "demand" for a workspace is the sum of its candidate block costs.
3. **Slack collection.** Workspaces whose demand is below their allocation surrender the slack to a redistribution pool.
4. **Slack redistribution.** Over-demand workspaces (whose candidate blocks exceed their initial allocation) receive a proportional share of the pool, weighted by `(demand - allocation)`.
5. **Greedy fill.** Each workspace's final budget is consumed in score-descending order. A workspace with one rich, very-long block still gets at least one block in the output even if that single block exceeds its budget — the floor wins over the cap when both can't be honored.

The implication for the model: the budget allocator already weighs workspaces by content density. The model SHOULD NOT re-weight by workspace size or by which workspace it "likes better." The model SHOULD weight findings by:

- **Relevance.** Does the block address the question directly?
- **Recency.** When the question is about current state, surface fresh blocks first.
- **Conflict.** When two workspaces disagree, both blocks earn weight — surface both in the finding's `conflict_note`.

A workspace that produces zero candidate blocks under the budget simply contributes nothing to the synthesis. The model surfaces the empty workspace in the audit trail (so the operator sees "acme-archive produced no relevant blocks") instead of silently dropping it.

## Section 2 — Confidentiality Boundaries

Every workspace has a confidentiality tag declared in its `routing.yaml`. Four canonical levels exist, ordered ascending by strictness:

1. **PUBLIC.** No restrictions. The workspace's content can be freely combined with any other.
2. **PERSONAL.** The operator's own data — notes, drafts, lessons. Frontier models OK. Mixes freely with other PERSONAL workspaces; mixes with CLIENT_CONFIDENTIAL with an audit entry.
3. **CLIENT_CONFIDENTIAL.** Client data. Restricted to approved models. Mixes with PERSONAL with an audit entry. NEVER mixes with PUBLIC; doing so is a leak path the substrate refuses to enable.
4. **AIR_GAPPED.** Must never leave local infrastructure. NEVER mixes with anything except another AIR_GAPPED workspace.

The pairwise rule table:

| Pair                                       | Verdict        |
|--------------------------------------------|----------------|
| AIR_GAPPED + AIR_GAPPED                    | allowed        |
| AIR_GAPPED + anything else                 | DENIED         |
| CLIENT_CONFIDENTIAL + PUBLIC               | DENIED         |
| CLIENT_CONFIDENTIAL + PERSONAL             | allowed, audit |
| CLIENT_CONFIDENTIAL + CLIENT_CONFIDENTIAL  | allowed        |
| PERSONAL + PERSONAL                        | allowed        |
| PUBLIC + PUBLIC                            | allowed        |
| PUBLIC + PERSONAL                          | allowed        |

The **effective confidentiality** of a multi-workspace operation is `max(participating_confidentialities)`. If the question synthesizes a CLIENT_CONFIDENTIAL workspace with a PERSONAL workspace, the resulting report inherits CLIENT_CONFIDENTIAL. The model treats the resulting report as if every byte of it came from the strictest participating workspace.

Three operational rules follow from this:

1. **Never quote from a stricter workspace into a body destined for a more public reader.** If the user asks for a synthesis but is reading it on behalf of a PUBLIC audience, and the synthesis would mix in a CLIENT_CONFIDENTIAL workspace, the model refuses with a clear explanation: "this synthesis would expose CLIENT_CONFIDENTIAL content to a PUBLIC reader; route the question through only the PUBLIC workspaces, or escalate the reader's clearance."
2. **Produce separate outputs per confidentiality level when asked to summarize across a mix.** If the user asks "summarize what every workspace says about X" and the workspaces span PUBLIC + PERSONAL + CLIENT_CONFIDENTIAL, the model returns three reports — one per level — instead of one report at the strictest level. The reader can then choose which to surface where.
3. **The audit trail records the effective confidentiality of every operation.** Even when the operation is allowed, the trail records the verdict so an operator scanning the log later can confirm no boundary was crossed.

The `cross_workspace_synthesize` tool will REFUSE to return a SynthesisReport when the confidentiality gate denies the operation. The refusal carries a clear `ConfidentialityError` with the pairwise reason. The model surfaces the refusal to the user verbatim — it does not retry with a smaller workspace set without the user's explicit go-ahead.

## Section 3 — Citation Format

Every claim in the synthesis cites the workspace AND the source document. The citation format is:

```
[workspace:document_id]
```

inline, immediately after the claim. Examples:

- "The migration is owned by the engineering lead [acme-news:adr-0042]."
- "Two workspaces describe the schema change; acme-news says it lands in May [acme-news:release-2026-05.md] while beta-media says it lands in June [beta-media:roadmap.yaml]."

Rules:

1. **One citation per atomic claim.** Compound claims cite each underlying source. "The decision is owned by Anjali and was made on April 12" splits into two citations: one for ownership, one for the date.
2. **Cite every workspace that contributed.** When multiple workspaces independently affirm the same claim, the citation block lists every contributing workspace. This is the model's evidence that the claim is corroborated rather than echoed.
3. **Surface conflicts as conflicts, not consensus.** When two workspaces disagree, the finding carries BOTH citations and a `conflict_note` field that names the disagreement. The model does not silently pick one source; the disagreement is the signal.
4. **End-of-report bibliography.** The `citations[]` field at the bottom of the report lists every citation referenced by any finding, with the snippet. This is the operator's lookup table: scan the bibliography, find the snippet, verify the finding against the source.

The model NEVER fabricates a citation. If a finding cannot be cited, the finding does not belong in the synthesis. Mention the gap explicitly in the summary: "the workspaces did not contain sufficient grounded content to answer the second half of the question."

## Section 4 — Audit Trail Surfacing

The bottom of every SynthesisReport carries an audit trail. The trail is the operator's "show your work" surface — they should be able to scan it and confirm no confidentiality boundary was crossed.

The trail contains, in order:

1. **Workspaces consulted.** The exact list passed in, in the order they were queried.
2. **Per-workspace confidentiality.** The confidentiality tag of each workspace.
3. **Effective confidentiality.** The max across the participants, in CAPITAL_CASE.
4. **Audit-required flag.** "Audit required" when the pair triggered the PERSONAL + CLIENT_CONFIDENTIAL borderline rule; "no audit" otherwise. (The audit log entry is written regardless; this flag is the operator-visible signal.)
5. **Tools fired.** Every tool the synthesis pipeline called. At minimum: `cross_workspace_synthesize` itself; downstream `three_tier_retrieve_multi` calls; the LLM completion call.
6. **Model id.** The model that produced the synthesis. The operator should be able to verify the model was on the workspace's `allowed_models` list.
7. **Per-workspace block count.** "acme-news contributed 4 blocks; acme-user contributed 2 blocks; beta-archive contributed 0 blocks." The operator scans this to spot a workspace that under-contributed.

The audit log entry is written to `~/.synthesis/cross-workspace-audit.jsonl` (or the path declared in `$SYNTHESIS_AUDIT_LOG_PATH`). The on-disk record carries the same fields as the in-report trail, plus a timestamp and a redacted `args_summary`. The model does NOT need to write the log entry — the `cross_workspace_synthesize` tool's Python driver handles that on every call, success or failure. The model's job is to surface the in-report trail so the operator does not have to dig through the JSONL file to confirm the operation was clean.

When the operation is DENIED at the confidentiality gate, the audit entry is still written, with `outcome: "denied"` and the pairwise reason. This is intentional: the substrate captures every attempt, not just successful ones, so a forensic review can reconstruct the operator's intent even when the operation never executed.

## Schema Example

A SynthesisReport that synthesizes a question across `acme-news` (PUBLIC) and `acme-user` (PERSONAL):

```json
{
  "summary": "Both workspaces describe the migration plan, but acme-news names a different owner than acme-user. The migration is scheduled for May; the owner is contested.",
  "findings": [
    {
      "claim": "The migration is scheduled for May 2026.",
      "supporting_citations": [
        {"workspace": "acme-news", "document_id": "release-2026-05.md", "snippet": "Migration kickoff: 2026-05-12."},
        {"workspace": "acme-user", "document_id": "notes/2026-04-30.md", "snippet": "Confirmed May for the migration."}
      ]
    },
    {
      "claim": "The migration owner is contested between two candidates.",
      "supporting_citations": [
        {"workspace": "acme-news", "document_id": "adr-0042", "snippet": "Owner: engineering lead."},
        {"workspace": "acme-user", "document_id": "notes/2026-04-30.md", "snippet": "We agreed Priya would own it."}
      ],
      "conflict_note": "acme-news names the engineering lead; acme-user names Priya. Resolve before publishing."
    }
  ],
  "citations": [
    {"workspace": "acme-news", "document_id": "release-2026-05.md", "snippet": "Migration kickoff: 2026-05-12."},
    {"workspace": "acme-news", "document_id": "adr-0042", "snippet": "Owner: engineering lead."},
    {"workspace": "acme-user", "document_id": "notes/2026-04-30.md", "snippet": "Confirmed May for the migration. We agreed Priya would own it."}
  ],
  "audit_trail": [
    "Workspaces consulted: acme-news, acme-user",
    "Per-workspace confidentiality: acme-news=PUBLIC, acme-user=PERSONAL",
    "Effective confidentiality: PERSONAL",
    "Audit required: no audit (PUBLIC + PERSONAL is within policy)",
    "Tools fired: cross_workspace_synthesize, three_tier_retrieve_multi, llm.complete",
    "Model id: anthropic/claude-fable-5-1",
    "Per-workspace block counts: acme-news=3, acme-user=2"
  ],
  "effective_confidentiality": "PERSONAL"
}
```

The summary opens with the cross-workspace claim. The findings each cite the contributing workspaces. The conflict is surfaced in a `conflict_note`, not resolved silently. The audit trail is human-readable on its own — no decoder needed.

## Chain of Thought the Model Should Follow

The synthesis is not a concatenation of two single-workspace search results. Two failure modes are common when the model treats it as one:

- **Echo bias.** Workspace A and workspace B both quote the same upstream source, and the model reports the finding as "two workspaces agree." It is one source, echoed twice. The model checks for shared document identifiers and de-emphasizes them.
- **Volume bias.** Workspace A has 4 blocks, workspace B has 1 block, and the model gives workspace A four times the weight. The blocks-per-workspace metric is a retrieval artifact, not a relevance signal.

The model's reasoning pattern, per finding:

1. **Read every block.** Do not skim. The synthesis quality is bounded by the model's grasp of the retrieved evidence.
2. **Identify the atomic claim.** What is the smallest assertion of fact the blocks support?
3. **Trace each claim to its citations.** Every claim cites at least one workspace + document. Compound claims cite each underlying source.
4. **Surface conflicts as conflicts.** If workspace A says X and workspace B says Y, the finding records both. The model does not resolve the conflict; the user does.
5. **Note gaps.** If the question has three parts and the workspaces grounded only two of them, the summary opens with the gap.

## Edge Cases

**One workspace produced no blocks.** Record the empty workspace in the audit trail. Do not exclude it from the participant list — the operator needs to see that the workspace was consulted but found nothing relevant.

**A workspace has no loaded `routing.yaml`.** Fails closed to AIR_GAPPED, which denies any mix. The model returns the denial verbatim and asks the operator to add a `routing.yaml` to the workspace root before retrying.

**The user asks for the report in a confidentiality level lower than `effective_confidentiality`.** Refuse. Explain that the synthesis inherits the strictest level and offer to produce a separate report restricted to the workspaces at or below the requested level.

**Three or more workspaces.** Same algorithm. The pairwise confidentiality check evaluates every pair; one denied pair denies the whole operation. The budget allocator scales naturally to N participants.

**An attempt that hits the AIR_GAPPED gate.** The audit log records the denial with `outcome: "denied"`. The report is not produced. The user sees the gate's reason verbatim and decides whether to retry with a narrower workspace set.

## Output Format

Default output is the SynthesisReport object defined in the tool's `output_schema`. The model renders the JSON when the caller asks for a structured output channel; otherwise the model formats the synthesis as Markdown with:

- the `summary` paragraph as the opening,
- the `findings[]` as a numbered list with bracketed `[workspace:document_id]` citations,
- the `citations[]` block as an end-of-report bibliography,
- the `audit_trail[]` as a "show your work" block under a horizontal rule,
- the `effective_confidentiality` as a label on the audit trail's first line.

The audit trail always appears at the bottom. The reader who skims the report sees the summary and findings; the reader who wants to audit sees the trail.

## Relationship to Other Starter Pack Skills

- `workspace-search-with-citations` is the single-workspace counterpart. Use it for one-workspace queries; this skill is the multi-workspace variant.
- `fact-check-claims` audits an existing draft. The combination — synthesize across workspaces, then fact-check the synthesis against the source workspaces — is the highest-confidence research pattern the starter pack supports.
- `summarize-document` consumes one document inline. This skill consumes multiple workspaces' worth of grounded content.
- `agent-self-review` invokes this skill on its own previous turn when the turn's question was cross-workspace.
