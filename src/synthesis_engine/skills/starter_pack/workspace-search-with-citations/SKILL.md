---
name: workspace-search-with-citations
description: "Search the active workspace's three-tier memory (recent turns, working memory, long-term vector store) and answer with explicit citations to source documents and character ranges. Use when asked to: find information, look up, search the workspace, retrieve, locate documents, what does the workspace say about, cite sources, where in the docs, ground the answer."
license: "Apache-2.0"
scope: universal
depends_on: []
metadata:
  author: "Synthesis Engineering"
  version: "1.0.0"
  source_repo: "github.com/synthesisengineering/ragbot"
  source_type: "starter_pack"
tools:
  - name: workspace_search
    description: "Retrieve the top-k most relevant chunks from the active workspace's three-tier memory. Returns each hit with the document identifier, the character range within that document, the surrounding text, and a similarity score."
    input_schema:
      type: object
      properties:
        query:
          type: string
          description: "Natural-language query. The retriever rewrites this internally for hybrid lexical and semantic search; pass the user's question verbatim."
        k:
          type: integer
          description: "Maximum number of chunks to return. Defaults to 10. Use higher values when the question is broad; use lower values when latency matters."
          default: 10
          minimum: 1
          maximum: 50
      required:
        - query
    output_schema:
      type: object
      properties:
        hits:
          type: array
          items:
            type: object
            properties:
              document_id:
                type: string
                description: "Stable identifier for the source document."
              char_range:
                type: array
                description: "Two-element [start, end] character offsets within the source document."
                items:
                  type: integer
              text:
                type: string
                description: "The chunk text as it appears in the source."
              score:
                type: number
                description: "Similarity score in [0, 1]. Higher is more relevant."
              source_workspace:
                type: string
                description: "Workspace the hit came from. Matters when cross-workspace fan-out is active."
            required:
              - document_id
              - char_range
              - text
              - score
      required:
        - hits
---

# Workspace Search with Citations

This skill makes Ragbot's three-tier memory retrieval usable by the model the way humans use a research library: ask a question, get back passages with the page numbers attached, and answer using those passages — quoting where useful and pointing back to the source every time.

The skill exists because retrieval without citations is indistinguishable from hallucination. A response that says "the deployment uses Cloud Run" is useful only if the reader can verify it. The verification path is the citation: which document, which character range, what does that passage actually say. The model owes the user that path on every factual claim.

## When to Apply

- The user asks a question that depends on workspace content (their notes, code, lessons, project context, prior decisions).
- The user asks "where does it say X?" or any variant that explicitly asks for a citation.
- The user asks for a summary or synthesis that draws on multiple documents.
- The user follows up on an earlier answer and the answer needs to stay grounded in the same sources.

## When NOT to Apply

- The question is general knowledge that the workspace cannot reasonably answer (math, definitions, public-domain facts). The model answers from training and says so.
- The question is procedural and answered by another skill (e.g., `summarize-document` when a document is given inline).
- The user explicitly asks for the model's opinion, draft, or generation — in which case `draft-and-revise` is the right tool. Citations still belong in any factual scaffolding inside the draft.

## Protocol

The skill is a four-step loop. Each step has a single responsibility; the model should not skip steps or compress them, because skipping is where ungrounded responses come from.

### 1. Read the question for retrievable claims

Before calling the tool, identify what the model would need to know from the workspace to answer well. Decompose compound questions into the underlying factual sub-questions. A question like "is the staging deployment using the same model as prod?" decomposes into "what model is staging using?" and "what model is prod using?" — two retrievals, not one.

When the decomposition is non-obvious, say so in the response. The user benefits from seeing the search strategy when the question is ambiguous.

### 2. Call `workspace_search` with focused queries

One call per sub-question, not one mega-query. Hybrid retrievers reward focused queries with better top-k results; mega-queries dilute the signal.

The `k` parameter trades coverage for latency. Use the default of 10 for most queries. Drop to 3-5 when the question is narrow and the right hit is likely near the top. Raise to 20+ when the user has explicitly asked for a comprehensive sweep ("everything the workspace says about X").

When a search returns zero hits or only weak hits (top score below 0.3), the workspace does not have a confident answer. The model says so, instead of inventing one. "I did not find relevant content in the workspace for X. The closest hit was a document on Y, which does not directly answer the question."

### 3. Read the hits before composing the answer

Each hit comes back with the chunk text. The model reads it. The model does not assume the chunk says what the chunk title or document identifier implies. Two common failure modes the model must avoid:

- Citing a hit that does not actually support the claim. This is the false-citation pattern; it is worse than no citation because it transfers the model's confidence onto a source that disagrees.
- Quoting the chunk text without checking that the surrounding sentences in the chunk do not undermine the quote. Chunks are excerpts; an excerpt out of context can flip meaning.

When a hit is ambiguous, the model says so. When a hit is on point, the model uses it.

### 4. Compose the answer with inline citations

Every factual claim that came from a hit is attributed inline. Use a compact format:

```
The staging environment uses claude-sonnet-5-5 [deploy-config.yaml:142-167]. Production
runs the same model on the same revision [deploy-config.yaml:142-167; release-notes.md:88-95].
```

The bracket contains the document identifier and the character range. When multiple hits support the same claim, list them comma-separated within the same bracket.

If the model is offering its own synthesis (not a direct quote), it still attributes the underlying facts. The synthesis claim itself does not need a citation — the supporting facts do.

If the model is uncertain or the hits disagree, the answer surfaces that explicitly. "The deploy config lists model X [...]; the release notes describe model Y [...]; the discrepancy was introduced on the 2026-04-12 release commit." A disagreement surfaced is more useful than a single source picked arbitrarily.

## Output Format

The default response shape is prose with inline citations. When the user asks for a structured list ("list every place the staging URL appears"), produce a table or bulleted list with one row per hit and the citation in its own column.

Citations are mandatory. A response that asserts a fact from the workspace without a citation is a defect. If the supporting hit was weak (low score, ambiguous chunk), the model says so in addition to citing it.

## Edge Cases

**No relevant hits.** Say so explicitly. Do not synthesize an answer from training data and present it as a workspace answer.

**Hits from multiple workspaces.** Ragbot's cross-workspace fan-out returns hits tagged with `source_workspace`. When citing, include the workspace name in the bracket. The user benefits from knowing which workspace the answer came from.

**Conflicting hits.** Surface the conflict. Do not pick a winner silently. The user is the one who decides which source supersedes the other.

**Stale hits.** If a hit references a date or version, and the user is asking about the current state, note the date in the answer. "Per the deploy config last updated 2026-03-22 [...], the model is X. If the deployment has changed since, this citation is stale."

## Relationship to Other Starter Pack Skills

- `summarize-document` consumes the citations from this skill when summarizing a document that lives in the workspace.
- `fact-check-claims` uses this skill's retrieval as its evidence base for each claim.
- `agent-self-review` checks that citations are present and that they actually support the claims they attach to — the groundedness rubric item.

The four skills work alone. They are stronger together: retrieval grounds drafts, drafts get fact-checked, fact-checks pass through self-review, and the loop closes on a response the user can verify end-to-end.
