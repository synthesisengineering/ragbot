# Ragbot RAG Architecture

Ragbot implements a **production-grade, multi-stage RAG pipeline** based on research from leading AI systems including Perplexity, ChatGPT, Claude, and Gemini. This document describes the complete retrieval and generation architecture.

## Overview

Unlike simple RAG implementations that follow a basic "embed query → search → answer" pattern, Ragbot uses a sophisticated multi-stage pipeline with distinct responsibilities at each stage:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RAGBOT RAG PIPELINE                                   │
│                                                                              │
│  ┌─────────┐   ┌─────────────┐   ┌──────────┐   ┌─────────┐   ┌──────────┐ │
│  │ Phase 1 │ → │   Phase 2   │ → │ Phase 3  │ → │Generate │ → │ Phase 4  │ │
│  │Foundation│   │Query Intel │   │ Hybrid   │   │Response │   │ Verify   │ │
│  └─────────┘   └─────────────┘   │ Retrieval│   └─────────┘   └──────────┘ │
│                                  └──────────┘                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Phase 1: Foundation Layer

**Purpose:** Intelligent query understanding and document-aware retrieval

### Components

1. **Query Preprocessing**
   - Contraction expansion ("what's" → "what is")
   - Stop word removal for key term extraction
   - Document lookup detection patterns

2. **Full Document Retrieval**
   - Detects queries like "show me my biography"
   - Returns complete documents instead of fragments
   - 16,000 token context budget (8x increase from typical 2K)

3. **Enhanced Embeddings**
   - Filename and title included in embedding text
   - "rajiv-pant-biography.md" → "rajiv pant biography" for semantic matching
   - Significantly improves document-name queries

### Impact
- Queries like "show me my biography" now return the complete, correct document
- Reduced fragmentation for targeted document requests

## Phase 2: Query Intelligence

**Purpose:** LLM-powered query understanding and expansion

### Components

1. **Query Planner** (uses provider's fast model)
   - Analyzes query intent and complexity
   - Determines retrieval strategy (full_document, semantic_chunks, hybrid)
   - Identifies filename hints and answer style

2. **Multi-Query Expansion**
   - Generates 5-7 query variations for better recall
   - Extracts key entities and concepts
   - Creates filename patterns for matching

3. **HyDE (Hypothetical Document Embeddings)**
   - Generates hypothetical answer to the query
   - Uses hypothetical answer for semantic search
   - Bridges the gap between questions and answers

### Provider-Agnostic Design
- Uses `engines.yaml` tiers (bulk, routine, judgment; plus the flagship selector)
- Automatically selects fast model from same provider as user's model
- Ensures consistent API key usage across pipeline

### Impact
- **+21 NDCG points** from query rewriting (Microsoft benchmarks)
- Better recall through multiple query perspectives
- Graceful fallback to Phase 1 when LLM unavailable

## Phase 3: Advanced Retrieval

**Purpose:** Hybrid search combining multiple retrieval strategies

### Components

1. **Vector Search**
   - Semantic similarity using sentence-transformers
   - `all-MiniLM-L6-v2` embeddings (384 dimensions)
   - Cosine distance for matching

2. **BM25 Keyword Search**
   - Classic term-frequency ranking
   - Complements semantic search for exact matches
   - Includes filename/title in indexable text

3. **Reciprocal Rank Fusion (RRF)**
   - Merges ranked lists from vector and BM25 search
   - Formula: `score = Σ(1 / (k + rank))` for each list
   - Balances semantic and lexical relevance

4. **LLM-Based Reranking**
   - Scores each result's relevance to query (0-10 scale)
   - Uses provider's fast model for efficiency
   - Combined score: 30% original + 70% LLM judgment

### Impact
- **67% fewer retrieval failures** from hybrid search (Anthropic benchmarks)
- Exact keyword matches no longer lost in semantic search
- More accurate final ranking through LLM judgment

## Phase 4: Response Verification

**Purpose:** Hallucination detection and corrective retrieval

### Components

1. **Response Verifier**
   - Extracts factual claims from generated response
   - Matches each claim against retrieved context
   - Classifies as SUPPORTED, UNSUPPORTED, or PARTIALLY_SUPPORTED

2. **Confidence Scoring**
   - Formula: `(supported + 0.5 * partial) / total - 0.1 * unsupported + bonus`
   - Bonus (+0.1) when no unsupported claims
   - Score range: 0.0 to 1.0

3. **CRAG (Corrective RAG) Loop**
   - Triggers when confidence < threshold (default: 0.7)
   - Generates targeted queries for unsupported claims
   - Retrieves additional context and re-verifies
   - Up to 2 correction attempts

### Verification Flow

```
Response Generated
       │
       ▼
┌─────────────────┐
│  Extract Claims │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Match Evidence  │
│ for Each Claim  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     < 0.7      ┌─────────────────┐
│  Calculate      │ ──────────────→│   CRAG Loop     │
│  Confidence     │                │  (up to 2x)     │
└────────┬────────┘                └─────────────────┘
         │ ≥ 0.7
         ▼
┌─────────────────┐
│  Return with    │
│  Confidence     │
└─────────────────┘
```

### Impact
- Catches hallucinations before reaching users
- Provides confidence scores for response quality assessment
- Self-correcting through targeted re-retrieval

## Performance Characteristics

| Stage | Latency | Notes |
|-------|---------|-------|
| Phase 1 | <50ms | Local preprocessing, no LLM |
| Phase 2 | 100-300ms | Fast model for planning/expansion |
| Phase 3 (search) | 50-100ms | Vector + BM25 search |
| Phase 3 (rerank) | 100-200ms | LLM reranking top 20 |
| Phase 4 (verify) | 150-300ms | Claim extraction + matching |
| Phase 4 (CRAG) | 500-1500ms | Only if triggered (<20% of requests) |

**Total typical latency:** 400-700ms (without CRAG)

## Comparison to Other Systems

| Feature | Ragbot | Basic RAG | Enterprise RAG |
|---------|--------|-----------|----------------|
| Query preprocessing | ✅ | ❌ | ✅ |
| Full document retrieval | ✅ | ❌ | Sometimes |
| LLM query planning | ✅ | ❌ | ✅ |
| Multi-query expansion | ✅ | ❌ | ✅ |
| HyDE | ✅ | ❌ | Sometimes |
| Hybrid search (BM25+Vector) | ✅ | ❌ | ✅ |
| Reciprocal Rank Fusion | ✅ | ❌ | ✅ |
| LLM reranking | ✅ | ❌ | ✅ |
| Response verification | ✅ | ❌ | Rare |
| CRAG (corrective retrieval) | ✅ | ❌ | Rare |
| Confidence scoring | ✅ | ❌ | Sometimes |
| Provider-agnostic | ✅ | Varies | ❌ |

## Configuration

### Enabling/Disabling Phases

```python
# In code
context = get_relevant_context(
    workspace_name="personal",
    query="What is my background?",
    user_model="anthropic/claude-sonnet-4",
    use_phase2=True,   # Query intelligence
    use_phase3=True    # Hybrid search + reranking
)

# Verification
result = verify_and_correct(
    query="...",
    response="...",
    context="...",
    workspace_name="personal",
    enable_verification=True,
    enable_crag=True,
    confidence_threshold=0.7
)
```

### Tuning Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_tokens` | 16000 | Maximum context budget |
| `limit` | 50 | Number of search results |
| `top_k` | 20 | Results to rerank |
| `confidence_threshold` | 0.7 | CRAG trigger threshold |
| `max_crag_attempts` | 2 | Maximum correction loops |

## Research Foundation

This architecture is based on combined research from:

- **Anthropic** - Contextual embeddings, RAG best practices
- **Microsoft Azure** - Hybrid search benchmarks, query rewriting impact
- **Perplexity** - Multi-stage pipeline architecture
- **OpenAI** - Query planning, HyDE technique
- **Google** - Reranking strategies, confidence scoring

Key findings from research:
- Contextual embeddings: **35% fewer retrieval failures**
- Hybrid search + reranking: **67% fewer retrieval failures**
- Query rewriting (10 variations): **+21 NDCG points**

## Source Code

The RAG implementation is in [src/rag.py](../src/rag.py):

- Lines 1-200: Phase 1 (Query preprocessing, document detection)
- Lines 200-1000: Phase 2 (Planner, multi-query, HyDE)
- Lines 280-700: Phase 3 (BM25, RRF, reranking)
- Lines 280-365: Phase 4 data structures
- Lines 1830-2250: Phase 4 implementation (verification, CRAG)

## Testing

Each phase has comprehensive unit tests:

- Phase 1: 22 tests
- Phase 2: 32 tests
- Phase 3: 32 tests
- Phase 4: 23 tests

Run tests with:
```bash
pytest tests/ -v
```
