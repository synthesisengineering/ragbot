# rag.py
# RAG (Retrieval-Augmented Generation) support for Ragbot
# Uses pgvector for vector storage and sentence-transformers for embeddings
# Uses shared chunking library for consistent text chunking
#
# Phase 1 Improvements (December 2025):
# - Increased context budget from 2K to 16K tokens
# - Full document retrieval for targeted queries
# - Query preprocessing (contraction expansion)
# - Enhanced filename/title matching
#
# Phase 2 Improvements (December 2025):
# - Query Planner stage using provider's fast model (tier="bulk")
# - Multi-query expansion (5-7 variations for better recall)
# - HyDE (Hypothetical Document Embeddings)
# - Provider-agnostic model selection via engines.yaml tiers
#
# Phase 3 Improvements (December 2025):
# - BM25/keyword search alongside vector search (hybrid retrieval)
# - Reciprocal Rank Fusion (RRF) for result merging
# - LLM-based reranking with provider's fast model
#
# Phase 4 Improvements (December 2025):
# - Response verification (hallucination detection)
# - Confidence scoring for responses
# - CRAG (Corrective RAG) loop for low-confidence responses
#
# Author: Rajiv Pant

import os
import re
import json
import math
import logging
from typing import Optional, Dict, List, Tuple, Any, Set

# Vector store abstraction (pgvector backend behind a common ABC).
# Importing at module load time so callers can rely on get_vector_store().
try:
    from synthesis_engine.vectorstore import (
        get_vector_store,
        Point as VectorStorePoint,
        SearchHit as VectorStoreHit,
    )
    VECTOR_STORE_AVAILABLE = True
except ImportError:  # pragma: no cover - import guard for partial installs
    get_vector_store = None  # type: ignore
    VectorStorePoint = None  # type: ignore
    VectorStoreHit = None  # type: ignore
    VECTOR_STORE_AVAILABLE = False
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from collections import Counter

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# Query Preprocessing (Phase 1 Improvements)
# =============================================================================

# Common contractions to expand for better keyword matching
CONTRACTIONS = {
    "what's": "what is",
    "where's": "where is",
    "who's": "who is",
    "how's": "how is",
    "that's": "that is",
    "there's": "there is",
    "here's": "here is",
    "it's": "it is",
    "let's": "let us",
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "couldn't": "could not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "i'm": "i am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "we'll": "we will",
    "they'll": "they will",
    "i'd": "i would",
    "you'd": "you would",
    "we'd": "we would",
    "they'd": "they would",
}

# Patterns that indicate a document lookup request (not semantic search)
DOCUMENT_LOOKUP_PATTERNS = [
    r"^show\s+(?:me\s+)?(?:my\s+|the\s+)?(.+)$",
    r"^display\s+(?:my\s+|the\s+)?(.+)$",
    r"^get\s+(?:me\s+)?(?:my\s+|the\s+)?(.+)$",
    r"^read\s+(?:my\s+|the\s+)?(.+)$",
    r"^open\s+(?:my\s+|the\s+)?(.+)$",
    r"^use\s+(?:the\s+)?(.+?)(?:\s+runbook)?$",
    r"^what(?:'s| is)\s+in\s+(?:my\s+|the\s+)?(.+)$",
    r"^what\s+does\s+(?:my\s+|the\s+)?(.+?)\s+(?:say|contain|have).*$",
]


def expand_contractions(query: str) -> str:
    """
    Expand contractions in a query for better keyword matching.

    Example: "what's in my biography" -> "what is in my biography"

    Args:
        query: Original user query

    Returns:
        Query with contractions expanded
    """
    result = query.lower()
    for contraction, expansion in CONTRACTIONS.items():
        # Use word boundaries to avoid partial matches
        result = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, result)
    return result


def detect_document_request(query: str) -> Tuple[bool, Optional[str]]:
    """
    Detect if a query is asking for a specific document by name.

    Args:
        query: User's query

    Returns:
        Tuple of (is_document_request, document_hint)
        document_hint is the extracted document name/pattern if detected
    """
    query_lower = query.lower().strip()

    for pattern in DOCUMENT_LOOKUP_PATTERNS:
        match = re.match(pattern, query_lower, re.IGNORECASE)
        if match:
            # Extract the document hint from the match
            doc_hint = match.group(1).strip()
            # Remove common suffixes that aren't part of the name
            doc_hint = re.sub(r'\s*(file|document|doc|content|runbook)s?\s*$', '', doc_hint)
            if doc_hint:
                return True, doc_hint

    return False, None


def preprocess_query(query: str) -> Dict[str, any]:
    """
    Preprocess a query for optimal retrieval.

    This implements Phase 1 query preprocessing:
    1. Expand contractions for keyword matching
    2. Detect document lookup requests
    3. Extract key terms for filename matching

    Args:
        query: Original user query

    Returns:
        Dict with:
        - original_query: The unchanged input
        - processed_query: Query with contractions expanded
        - is_document_request: Whether this looks like a document lookup
        - document_hint: Extracted document name pattern (if applicable)
        - search_terms: Key terms for filename matching
    """
    expanded = expand_contractions(query)
    is_doc_request, doc_hint = detect_document_request(query)

    # Extract meaningful search terms (remove stop words)
    stop_words = {'a', 'an', 'the', 'my', 'your', 'is', 'are', 'was', 'were',
                  'in', 'on', 'at', 'to', 'for', 'of', 'with', 'me', 'show',
                  'tell', 'give', 'get', 'find', 'what', 'where', 'how', 'when',
                  'display', 'open', 'read', 'use', 'about', 'does', 'do'}
    words = re.findall(r'\b[a-z]+\b', expanded.lower())
    search_terms = [w for w in words if w not in stop_words and len(w) > 2]

    return {
        'original_query': query,
        'processed_query': expanded,
        'is_document_request': is_doc_request,
        'document_hint': doc_hint,
        'search_terms': search_terms,
    }


# =============================================================================
# Phase 2: Query Intelligence (Planner, Multi-Query, HyDE)
# =============================================================================

# Planner prompt template - generates execution plan for the query
PLANNER_PROMPT = """You are a query planning assistant for a RAG (Retrieval-Augmented Generation) system.
Analyze the user's query and create an execution plan.

User query: "{query}"
Workspace context: Personal knowledge base with datasets (documents, notes) and runbooks (how-to guides)

Respond with JSON only (no markdown, no explanation):
{{
  "query_type": "document_lookup" | "factual_qa" | "procedural" | "multi_step",
  "retrieval_strategy": "full_document" | "semantic_chunks" | "hybrid",
  "filename_hints": ["hint1", "hint2"],
  "answer_style": "return_content" | "synthesize" | "list_sources",
  "complexity": "simple" | "moderate" | "complex"
}}

Query types:
- document_lookup: User wants a specific document ("show me my biography")
- factual_qa: User wants factual information ("what's my email address")
- procedural: User wants to know how to do something ("how do I write a blog post")
- multi_step: Complex query needing multiple retrievals

Retrieval strategies:
- full_document: Return entire document (for document_lookup)
- semantic_chunks: Standard RAG with relevant chunks
- hybrid: Combine semantic search with keyword matching"""

# Multi-query expansion prompt
MULTI_QUERY_PROMPT = """Generate search query variations to improve retrieval recall.

Original query: "{query}"
Query type: {query_type}

Generate 5-7 search query variations that:
1. Use different phrasings and synonyms
2. Include likely document/file name patterns
3. Extract key entities and concepts
4. Add related terms that might appear in relevant documents

Respond with JSON only (no markdown):
{{
  "queries": ["query1", "query2", "query3", "query4", "query5"],
  "key_entities": ["entity1", "entity2"],
  "filename_patterns": ["pattern1", "pattern2"]
}}"""

# HyDE (Hypothetical Document Embeddings) prompt
HYDE_PROMPT = """Generate a hypothetical document excerpt that would answer this query.
This will be used for semantic search - write content similar to what a matching document would contain.

Query: "{query}"

Write 2-3 sentences that a relevant document would contain. Be factual and specific.
Do not include any preamble or explanation, just the hypothetical content."""

# =============================================================================
# Phase 3: Advanced Retrieval (BM25, RRF, Reranking)
# =============================================================================

# Reranker prompt template - scores query-document relevance
RERANKER_PROMPT = """You are a relevance scoring assistant for a RAG system.
Score how relevant each document chunk is to the user's query.

User query: "{query}"

For each chunk below, output a relevance score from 0-10:
- 0-2: Not relevant (off-topic, wrong context)
- 3-4: Marginally relevant (tangentially related)
- 5-6: Somewhat relevant (contains related information)
- 7-8: Relevant (answers part of the query)
- 9-10: Highly relevant (directly answers the query)

Chunks to score:
{chunks}

Respond with JSON only (no markdown):
{{
  "scores": [score1, score2, score3, ...]
}}

Important: Return exactly {num_chunks} scores in the same order as the chunks."""


# =============================================================================
# Phase 4: Response Verification and CRAG
# =============================================================================

class ClaimStatus(Enum):
    """Status of a verified claim."""
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    PARTIALLY_SUPPORTED = "partially_supported"


@dataclass
class VerifiedClaim:
    """A single verified claim from a response."""
    claim: str
    status: ClaimStatus
    evidence: Optional[str]
    reasoning: str


@dataclass
class VerificationResult:
    """Result of verifying a response against context."""
    confidence: float  # 0.0 to 1.0
    is_grounded: bool
    claims: List[VerifiedClaim]
    suggested_corrections: List[str]


@dataclass
class CRAGResult:
    """Result of Corrective RAG loop."""
    final_response: str
    confidence: float
    attempts: int
    verification_history: List[VerificationResult]
    additional_context_used: bool


# Verifier prompt template - checks if response is grounded in context
VERIFIER_PROMPT = """You are a fact-checking assistant. Your task is to verify that the response is grounded in the provided context.

CONTEXT:
{context}

RESPONSE TO VERIFY:
{response}

For each factual claim in the response:
1. Find supporting evidence in the context
2. Mark as SUPPORTED, UNSUPPORTED, or PARTIALLY_SUPPORTED
3. Quote the relevant evidence if found

IMPORTANT:
- Only verify FACTUAL claims (not opinions, questions, or general statements)
- If the response says "I don't have information about X", that is NOT an unsupported claim
- If the response correctly summarizes context, mark claims as SUPPORTED
- Be conservative: only mark UNSUPPORTED if the claim clearly contradicts or has no basis in context

Respond with JSON only (no markdown):
{{
  "overall_confidence": 0.0-1.0,
  "is_grounded": true/false,
  "claims": [
    {{
      "claim": "The claim text",
      "status": "SUPPORTED" | "UNSUPPORTED" | "PARTIALLY_SUPPORTED",
      "evidence": "Quote from context if found, or null",
      "reasoning": "Why this is/isn't supported"
    }}
  ],
  "suggested_corrections": ["Correction1", "Correction2"]
}}"""

# CRAG query generation prompt - generates targeted queries for unsupported claims
CRAG_QUERY_PROMPT = """Generate targeted search queries to find evidence for these unsupported claims.

Original query: "{query}"

Unsupported claims that need evidence:
{claims}

Generate 2-3 specific search queries that would help find documents containing evidence for these claims.
Focus on key entities, facts, and concepts mentioned in the claims.

Respond with JSON only (no markdown):
{{
  "queries": ["query1", "query2", "query3"]
}}"""


def bm25_tokenize(text: str) -> List[str]:
    """Tokenize text for BM25 search.

    Simple tokenizer that:
    - Converts to lowercase
    - Splits on non-alphanumeric characters
    - Removes very short tokens
    - Removes common stop words

    Args:
        text: Text to tokenize

    Returns:
        List of tokens
    """
    # Simple tokenization: lowercase, split on non-alphanumeric
    tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())

    # Remove stop words and very short tokens
    stop_words = {
        'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'must',
        'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
        'it', 'its', 'this', 'that', 'these', 'those',
        'i', 'me', 'my', 'you', 'your', 'we', 'our', 'they', 'their',
        'what', 'which', 'who', 'whom', 'when', 'where', 'why', 'how',
        'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'not', 'only', 'same', 'so', 'than', 'too',
        'very', 'just', 'can', 'now', 'as', 'if', 'then', 'else', 'also'
    }

    return [t for t in tokens if len(t) > 1 and t not in stop_words]


class BM25Index:
    """Simple BM25 index for keyword search.

    BM25 (Best Matching 25) is a ranking function used for information retrieval.
    It considers term frequency, document frequency, and document length.

    This implementation is designed for in-memory use with Ragbot's RAG system.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """Initialize BM25 index.

        Args:
            k1: Term frequency saturation parameter (default: 1.5)
            b: Document length normalization parameter (default: 0.75)
        """
        self.k1 = k1
        self.b = b
        self.documents: List[Dict] = []  # Store original documents
        self.doc_tokens: List[List[str]] = []  # Tokenized documents
        self.doc_lengths: List[int] = []  # Document lengths
        self.avg_doc_length: float = 0.0
        self.doc_freqs: Dict[str, int] = Counter()  # Document frequencies
        self.term_freqs: List[Counter] = []  # Term frequencies per document

    def add_documents(self, documents: List[Dict], text_field: str = 'text'):
        """Add documents to the index.

        Args:
            documents: List of document dicts with text and metadata
            text_field: Field containing the text to index
        """
        for doc in documents:
            text = doc.get(text_field, '')
            # Also include filename and title in indexable text
            filename = doc.get('metadata', {}).get('filename', '')
            title = doc.get('metadata', {}).get('title', '')

            # Combine text with filename/title for better keyword matching
            full_text = f"{filename} {title} {text}"
            tokens = bm25_tokenize(full_text)

            self.documents.append(doc)
            self.doc_tokens.append(tokens)
            self.doc_lengths.append(len(tokens))

            # Count term frequencies for this document
            tf = Counter(tokens)
            self.term_freqs.append(tf)

            # Update document frequencies (how many docs contain each term)
            for term in set(tokens):
                self.doc_freqs[term] += 1

        # Update average document length
        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)

    def search(self, query: str, limit: int = 10) -> List[Tuple[Dict, float]]:
        """Search for documents matching the query.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of (document, score) tuples, sorted by score descending
        """
        if not self.documents:
            return []

        query_tokens = bm25_tokenize(query)
        if not query_tokens:
            return []

        n_docs = len(self.documents)
        scores = []

        for doc_idx in range(n_docs):
            score = 0.0
            doc_length = self.doc_lengths[doc_idx]
            tf = self.term_freqs[doc_idx]

            for term in query_tokens:
                if term not in self.doc_freqs:
                    continue

                # IDF: Inverse Document Frequency
                df = self.doc_freqs[term]
                idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)

                # TF: Term Frequency with BM25 normalization
                term_freq = tf.get(term, 0)
                tf_norm = (term_freq * (self.k1 + 1)) / (
                    term_freq + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                )

                score += idf * tf_norm

            if score > 0:
                scores.append((self.documents[doc_idx], score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:limit]


def reciprocal_rank_fusion(
    result_lists: List[List[Tuple[Dict, float]]],
    k: int = 60
) -> List[Tuple[Dict, float]]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion.

    RRF is a simple but effective method for combining ranked lists.
    Score = sum(1 / (k + rank)) for each list where the document appears.

    Args:
        result_lists: List of ranked result lists, each containing (document, score) tuples
        k: Constant to prevent high ranks from dominating (default: 60)

    Returns:
        Merged list of (document, rrf_score) tuples, sorted by RRF score
    """
    # Create unique key for each document
    def doc_key(doc: Dict) -> str:
        metadata = doc.get('metadata', doc)
        filename = metadata.get('filename', '')
        char_start = metadata.get('char_start', 0)
        return f"{filename}:{char_start}"

    # Calculate RRF scores
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict] = {}

    for result_list in result_lists:
        for rank, (doc, _original_score) in enumerate(result_list, start=1):
            key = doc_key(doc)
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (k + rank)
            doc_map[key] = doc

    # Sort by RRF score
    sorted_results = sorted(
        [(doc_map[key], score) for key, score in rrf_scores.items()],
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_results


def rerank_with_llm(
    query: str,
    results: List[Dict],
    user_model: Optional[str] = None,
    workspace: Optional[str] = None,
    top_k: int = 20
) -> List[Dict]:
    """Rerank search results using an LLM for relevance scoring.

    Uses the provider's fast model to score each result's relevance
    to the query, then reorders by LLM score.

    Args:
        query: Original user query
        results: List of search results with 'text' and 'metadata'
        user_model: User's selected model (for provider selection)
        workspace: Workspace name for API key resolution
        top_k: Number of top results to rerank (default: 20)

    Returns:
        Reranked results with 'llm_score' added
    """
    if not results:
        return results

    # Only rerank top_k results to control costs/latency
    to_rerank = results[:top_k]
    rest = results[top_k:]

    # Format chunks for the prompt
    chunk_texts = []
    for i, result in enumerate(to_rerank):
        text = result.get('text', '')[:500]  # Truncate for prompt size
        filename = result.get('metadata', {}).get('filename', 'unknown')
        chunk_texts.append(f"[{i+1}] {filename}: {text}")

    chunks_str = "\n\n".join(chunk_texts)

    # Build and send prompt
    prompt = RERANKER_PROMPT.format(
        query=query,
        chunks=chunks_str,
        num_chunks=len(to_rerank)
    )

    response = _call_fast_llm(prompt, user_model, workspace)

    if response:
        try:
            # Parse JSON response
            json_str = response
            if '```json' in json_str:
                json_str = json_str.split('```json')[1].split('```')[0]
            elif '```' in json_str:
                json_str = json_str.split('```')[1].split('```')[0]

            result_data = json.loads(json_str.strip())
            scores = result_data.get('scores', [])

            # Apply LLM scores
            for i, result in enumerate(to_rerank):
                if i < len(scores):
                    llm_score = float(scores[i])
                    result['llm_score'] = llm_score
                    # Combine with original score (weighted average)
                    original_score = result.get('score', 0.5)
                    result['combined_score'] = 0.3 * original_score + 0.7 * (llm_score / 10)
                else:
                    result['llm_score'] = 5.0  # Default middle score
                    result['combined_score'] = result.get('score', 0.5)

            # Sort by combined score
            to_rerank.sort(key=lambda x: x.get('combined_score', 0), reverse=True)

            logger.info(f"LLM reranking complete: top scores = {[r.get('llm_score', 0) for r in to_rerank[:5]]}")

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse reranker response: {e}")
            # Fall back to original order
            for result in to_rerank:
                result['llm_score'] = None
                result['combined_score'] = result.get('score', 0.5)
    else:
        # LLM unavailable - use original scores
        logger.info("LLM reranking skipped (no fast model available)")
        for result in to_rerank:
            result['llm_score'] = None
            result['combined_score'] = result.get('score', 0.5)

    # Combine reranked and rest
    return to_rerank + rest


def hybrid_search(
    workspace_name: str,
    query: str,
    limit: int = 50,
    content_type: Optional[str] = None,
    use_bm25: bool = True,
    use_rrf: bool = True
) -> List[Dict]:
    """Perform hybrid search combining vector and BM25 search.

    This is the core Phase 3 retrieval function. It:
    1. Runs vector (semantic) search
    2. Runs BM25 (keyword) search
    3. Merges results using Reciprocal Rank Fusion

    Args:
        workspace_name: Workspace to search
        query: Search query
        limit: Maximum number of results
        content_type: Filter by content type
        use_bm25: Enable BM25 keyword search
        use_rrf: Use RRF to merge results (vs simple concatenation)

    Returns:
        List of search results with 'text', 'score', 'metadata'
    """
    # Step 1: Vector search (existing implementation)
    vector_results = search(workspace_name, query, limit=limit, content_type=content_type)

    if not use_bm25:
        return vector_results

    # Step 2: BM25 search
    # First, we need to get all documents to build the BM25 index
    # For efficiency, we'll use the vector search results as the corpus
    # (This is a practical trade-off: full BM25 would require loading all docs)

    if not VECTOR_STORE_AVAILABLE:
        return vector_results
    vs = get_vector_store()
    if vs is None:
        return vector_results

    try:
        # Try native FTS first (pgvector). If the backend supports it, this
        # is faster, more accurate, and avoids loading the corpus into memory.
        bm25_results = []
        native_hits = vs.keyword_search(
            workspace_name,
            query,
            limit=limit,
            content_type=content_type,
        )
        if native_hits:
            bm25_results = [
                {
                    'text': h.text,
                    'score': h.score,
                    'metadata': dict(h.metadata),
                }
                for h in native_hits
            ]
        else:
            # Fallback: in-process BM25 over scrolled chunks. Reached only
            # when a backend declines to implement keyword_search natively
            # — pgvector always returns native FTS hits, so the fallback
            # never fires on Ragbot's bundled stack. Kept for backends
            # plugged in by other synthesis-engine consumers.
            # Limit to 500 to bound memory.
            scrolled = vs.scroll_documents(
                workspace_name,
                limit=500,
                content_type=content_type,
            )
            if not scrolled:
                return vector_results

            bm25_docs = [
                {'text': h.text, 'metadata': dict(h.metadata)}
                for h in scrolled
            ]
            if not bm25_docs:
                return vector_results

            bm25_index = BM25Index()
            bm25_index.add_documents(bm25_docs)
            bm25_raw = bm25_index.search(query, limit=limit)
            bm25_results = [
                {
                    'text': doc.get('text', ''),
                    'score': score,
                    'metadata': doc.get('metadata', {}),
                }
                for doc, score in bm25_raw
            ]

        logger.info(f"Hybrid search: {len(vector_results)} vector, {len(bm25_results)} BM25")

        # Step 3: Merge results
        if use_rrf and vector_results and bm25_results:
            # Prepare for RRF
            vector_list = [(r, r['score']) for r in vector_results]
            bm25_list = [(r, r['score']) for r in bm25_results]

            # Apply RRF
            merged = reciprocal_rank_fusion([vector_list, bm25_list])

            # Convert back to standard format
            final_results = []
            for doc, rrf_score in merged[:limit]:
                result = doc.copy()
                result['rrf_score'] = rrf_score
                # Preserve original score if present
                if 'score' not in result:
                    result['score'] = rrf_score
                final_results.append(result)

            return final_results
        else:
            # Simple merge (concatenate and dedupe)
            seen = set()
            merged = []

            for r in vector_results + bm25_results:
                key = (r['metadata'].get('filename', ''), r['metadata'].get('char_start', 0))
                if key not in seen:
                    seen.add(key)
                    merged.append(r)

            # Sort by score
            merged.sort(key=lambda x: x['score'], reverse=True)
            return merged[:limit]

    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        return vector_results


def _get_fast_model(user_model: Optional[str] = None) -> Optional[str]:
    """Get the fast model for the same provider as the user's model.

    Uses the 'bulk' tier from engines.yaml to get the fastest
    model from the same provider, ensuring consistent API key usage.

    Args:
        user_model: User's selected model ID (e.g., 'anthropic/claude-fable-5')

    Returns:
        Fast model ID for the same provider, or None if not found
    """
    try:
        # Import config functions - handle both package and standalone usage
        try:
            from synthesis_engine.config import get_fast_model_for_provider, get_default_model
        except ImportError:
            from .synthesis_engine.config import get_fast_model_for_provider, get_default_model

        if user_model:
            fast_model = get_fast_model_for_provider(user_model)
            if fast_model:
                return fast_model

        # Fallback to default provider's fast model
        default_model = get_default_model()
        return get_fast_model_for_provider(default_model)

    except Exception as e:
        logger.warning(f"Could not determine fast model: {e}")
        return None


def _call_fast_llm(prompt: str, user_model: Optional[str] = None,
                   workspace: Optional[str] = None) -> Optional[str]:
    """Call the fast LLM (bulk tier) for auxiliary operations.

    Uses the same provider as the user's model to ensure consistent
    API key usage and billing. Falls back gracefully if unavailable.

    Args:
        prompt: The prompt to send
        user_model: User's selected model ID
        workspace: Workspace name for API key resolution

    Returns:
        LLM response text, or None on failure
    """
    fast_model = _get_fast_model(user_model)
    if not fast_model:
        logger.warning("No fast model available for auxiliary LLM calls")
        return None

    try:
        import litellm

        # Get API key for the provider
        try:
            from synthesis_engine.config import get_provider_for_model
            from synthesis_engine.keystore import get_api_key
        except ImportError:
            from .synthesis_engine.config import get_provider_for_model
            from .synthesis_engine.keystore import get_api_key

        provider = get_provider_for_model(fast_model)

        # Get API key name from engines.yaml
        try:
            from synthesis_engine.config import get_provider_config
        except ImportError:
            from .synthesis_engine.config import get_provider_config

        provider_config = get_provider_config(provider)
        if provider_config:
            api_key_name = provider_config.get('api_key_name')
            api_key = get_api_key(api_key_name, workspace)
            if api_key:
                # Set API key for litellm
                if provider == 'anthropic':
                    litellm.api_key = api_key
                elif provider == 'openai':
                    os.environ['OPENAI_API_KEY'] = api_key
                elif provider == 'google':
                    os.environ['GOOGLE_API_KEY'] = api_key

        # Make the LLM call with minimal tokens (fast operation)
        response = litellm.completion(
            model=fast_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,  # Planner responses are short
            temperature=0.3,  # Low temperature for consistent planning
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.warning(f"Fast LLM call failed: {e}")
        return None


def plan_query(query: str, user_model: Optional[str] = None,
               workspace: Optional[str] = None) -> Dict[str, Any]:
    """Plan the retrieval strategy for a query using an LLM.

    This is Stage 1 of the Phase 2 pipeline. It uses a fast model
    to analyze the query and determine the best retrieval approach.

    Args:
        query: User's query
        user_model: User's selected model (to determine provider for fast model)
        workspace: Workspace name for API key resolution

    Returns:
        Dict with planning results:
        - query_type: Type of query (document_lookup, factual_qa, etc.)
        - retrieval_strategy: How to retrieve (full_document, semantic_chunks, hybrid)
        - filename_hints: Hints for document names
        - answer_style: How to format the answer
        - complexity: Query complexity
        - used_llm: Whether LLM planning was used (vs fallback)
    """
    # Try LLM-based planning first
    prompt = PLANNER_PROMPT.format(query=query)
    response = _call_fast_llm(prompt, user_model, workspace)

    if response:
        try:
            # Parse JSON response (handle potential markdown wrapping)
            json_str = response
            if '```json' in json_str:
                json_str = json_str.split('```json')[1].split('```')[0]
            elif '```' in json_str:
                json_str = json_str.split('```')[1].split('```')[0]

            plan = json.loads(json_str.strip())
            plan['used_llm'] = True
            logger.info(f"Query plan (LLM): type={plan.get('query_type')}, "
                       f"strategy={plan.get('retrieval_strategy')}")
            return plan

        except (json.JSONDecodeError, IndexError, KeyError) as e:
            logger.warning(f"Failed to parse planner response: {e}")

    # Fallback to Phase 1 heuristics if LLM fails
    phase1_result = preprocess_query(query)

    plan = {
        'query_type': 'document_lookup' if phase1_result['is_document_request'] else 'factual_qa',
        'retrieval_strategy': 'full_document' if phase1_result['is_document_request'] else 'semantic_chunks',
        'filename_hints': [phase1_result['document_hint']] if phase1_result['document_hint'] else [],
        'answer_style': 'return_content' if phase1_result['is_document_request'] else 'synthesize',
        'complexity': 'simple',
        'used_llm': False,
    }
    logger.info(f"Query plan (fallback): type={plan['query_type']}, "
               f"strategy={plan['retrieval_strategy']}")
    return plan


def expand_query(query: str, query_type: str = 'factual_qa',
                 user_model: Optional[str] = None,
                 workspace: Optional[str] = None) -> Dict[str, Any]:
    """Expand a query into multiple search variations for better recall.

    This is Stage 2a of the Phase 2 pipeline. It generates 5-7 query
    variations that will all be searched and results merged.

    Args:
        query: Original user query
        query_type: Type of query from planner
        user_model: User's selected model
        workspace: Workspace name

    Returns:
        Dict with:
        - queries: List of expanded query variations
        - key_entities: Extracted entities
        - filename_patterns: Likely filename patterns
        - used_llm: Whether LLM expansion was used
    """
    # Try LLM-based expansion
    prompt = MULTI_QUERY_PROMPT.format(query=query, query_type=query_type)
    response = _call_fast_llm(prompt, user_model, workspace)

    if response:
        try:
            # Parse JSON response
            json_str = response
            if '```json' in json_str:
                json_str = json_str.split('```json')[1].split('```')[0]
            elif '```' in json_str:
                json_str = json_str.split('```')[1].split('```')[0]

            expansion = json.loads(json_str.strip())
            expansion['used_llm'] = True
            logger.info(f"Query expansion (LLM): {len(expansion.get('queries', []))} variations")
            return expansion

        except (json.JSONDecodeError, IndexError, KeyError) as e:
            logger.warning(f"Failed to parse expansion response: {e}")

    # Fallback to Phase 1 term extraction
    phase1_result = preprocess_query(query)
    expanded_query = phase1_result['processed_query']
    search_terms = phase1_result['search_terms']

    # Generate simple variations
    queries = [
        expanded_query,  # Original (contractions expanded)
        ' '.join(search_terms),  # Just key terms
    ]

    # Add document hint as a query if present
    if phase1_result['document_hint']:
        queries.append(phase1_result['document_hint'])

    # Add term combinations
    if len(search_terms) >= 2:
        queries.append(f"{search_terms[0]} {search_terms[-1]}")

    return {
        'queries': queries,
        'key_entities': search_terms,
        'filename_patterns': [phase1_result['document_hint']] if phase1_result['document_hint'] else [],
        'used_llm': False,
    }


def generate_hyde_document(query: str, user_model: Optional[str] = None,
                           workspace: Optional[str] = None) -> Optional[str]:
    """Generate a hypothetical document for HyDE retrieval.

    HyDE (Hypothetical Document Embeddings) generates a hypothetical
    answer to the query, then embeds that answer for semantic search.
    This bridges the semantic gap between questions and answers.

    Args:
        query: User's query
        user_model: User's selected model
        workspace: Workspace name

    Returns:
        Hypothetical document text, or None if generation fails
    """
    prompt = HYDE_PROMPT.format(query=query)
    response = _call_fast_llm(prompt, user_model, workspace)

    if response:
        logger.info(f"Generated HyDE document ({len(response)} chars)")
        return response

    return None


def enhanced_preprocess_query(query: str, user_model: Optional[str] = None,
                              workspace: Optional[str] = None,
                              use_planner: bool = True,
                              use_multi_query: bool = True,
                              use_hyde: bool = True) -> Dict[str, Any]:
    """Enhanced query preprocessing with Phase 2 intelligence.

    Combines Phase 1 preprocessing with Phase 2 LLM-powered features:
    1. Query Planning (intent detection, strategy selection)
    2. Multi-Query Expansion (5-7 search variations)
    3. HyDE (hypothetical document for semantic search)

    Args:
        query: Original user query
        user_model: User's selected model (for provider-specific fast model)
        workspace: Workspace name for API key resolution
        use_planner: Enable LLM-based query planning
        use_multi_query: Enable multi-query expansion
        use_hyde: Enable HyDE document generation

    Returns:
        Dict with all preprocessing results:
        - (All Phase 1 fields)
        - plan: Query plan from planner
        - expanded_queries: List of query variations
        - hyde_document: Hypothetical answer document
        - phase2_enabled: Whether any Phase 2 features were used
    """
    # Start with Phase 1 preprocessing
    result = preprocess_query(query)

    # Add Phase 2 intelligence
    result['phase2_enabled'] = False

    # Stage 1: Query Planning
    if use_planner:
        plan = plan_query(query, user_model, workspace)
        result['plan'] = plan
        if plan.get('used_llm'):
            result['phase2_enabled'] = True

        # Use planner's hints to enrich results
        if plan.get('filename_hints'):
            result['search_terms'] = list(set(result['search_terms'] + plan['filename_hints']))

        # Override document detection if planner says document_lookup
        if plan.get('query_type') == 'document_lookup':
            result['is_document_request'] = True
            if plan.get('filename_hints') and not result['document_hint']:
                result['document_hint'] = plan['filename_hints'][0]
    else:
        result['plan'] = None

    # Stage 2a: Multi-Query Expansion
    if use_multi_query:
        query_type = result.get('plan', {}).get('query_type', 'factual_qa')
        expansion = expand_query(query, query_type, user_model, workspace)
        result['expanded_queries'] = expansion.get('queries', [result['processed_query']])
        if expansion.get('key_entities'):
            result['search_terms'] = list(set(result['search_terms'] + expansion['key_entities']))
        if expansion.get('used_llm'):
            result['phase2_enabled'] = True
    else:
        result['expanded_queries'] = [result['processed_query']]

    # Stage 2b: HyDE (only for non-document-lookup queries)
    if use_hyde and not result.get('is_document_request'):
        hyde_doc = generate_hyde_document(query, user_model, workspace)
        result['hyde_document'] = hyde_doc
        if hyde_doc:
            result['phase2_enabled'] = True
    else:
        result['hyde_document'] = None

    return result


# Lazy import for chunking - handle both relative and absolute imports
_chunking_loaded = False
chunk_file = None
chunk_files = None
ChunkConfig = None
Chunk = None


def _load_chunking():
    """Lazy load chunking module."""
    global _chunking_loaded, chunk_file, chunk_files, ChunkConfig, Chunk
    if _chunking_loaded:
        return True
    try:
        # Try relative import (when used as part of a package)
        from .chunking import chunk_file as cf, chunk_files as cfs, ChunkConfig as CC, Chunk as C
        chunk_file, chunk_files, ChunkConfig, Chunk = cf, cfs, CC, C
        _chunking_loaded = True
        return True
    except ImportError:
        try:
            # Try absolute import (when used standalone)
            from chunking import chunk_file as cf, chunk_files as cfs, ChunkConfig as CC, Chunk as C
            chunk_file, chunk_files, ChunkConfig, Chunk = cf, cfs, CC, C
            _chunking_loaded = True
            return True
        except ImportError:
            logger.warning("chunking module not available. Some RAG features may be limited.")
            return False

# Lazy imports - only load heavy dependencies when needed
_embedding_model = None


def _get_embedding_model():
    """Get or create embedding model (lazy initialization)."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer

            # Use a small, fast model that works well for retrieval
            # all-MiniLM-L6-v2 is 80MB and produces 384-dim embeddings
            model_name = os.environ.get('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
            logger.info(f"Loading embedding model: {model_name}")
            _embedding_model = SentenceTransformer(model_name)
        except ImportError:
            logger.warning("sentence-transformers not installed. RAG features disabled.")
            return None
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return None
    return _embedding_model


def _get_embedding_dimension(model) -> int:
    """Return the embedding dimension regardless of sentence-transformers version.

    The library renamed ``get_sentence_embedding_dimension`` →
    ``get_embedding_dimension`` in 5.x and emits a FutureWarning on the old
    name. We prefer the new method and fall back to the old one for older
    library versions.
    """

    if hasattr(model, "get_embedding_dimension"):
        return model.get_embedding_dimension()
    # Fallback for older sentence-transformers versions; suppress the
    # deprecation warning since this branch only runs on libraries that
    # don't yet expose the new name.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return model.get_sentence_embedding_dimension()


def is_rag_available() -> bool:
    """Check if RAG dependencies are available.

    RAG requires (a) sentence-transformers for embeddings, and (b) a working
    pgvector backend reachable via RAGBOT_DATABASE_URL. Returns False when
    either is missing.
    """

    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except ImportError:
        return False
    if not VECTOR_STORE_AVAILABLE:
        return False
    vs = get_vector_store()
    return vs is not None


def get_collection_name(workspace_name: str) -> str:
    """Generate a collection name for a workspace.

    Retained for back-compat with callers that still ask for a collection
    identifier; the pgvector backend uses the raw workspace name as the
    ``workspace`` column value and does not need the sanitised form.
    """
    safe_name = workspace_name.lower().replace(' ', '_').replace('-', '_')
    return f"ragbot_{safe_name}"


def init_collection(workspace_name: str, vector_size: int = 384) -> bool:
    """
    Initialize the workspace's storage in the configured vector backend.

    Args:
        workspace_name: Name of the workspace
        vector_size: Dimension of embedding vectors (384 for MiniLM)

    Returns:
        True if storage is ready, False otherwise
    """
    if not VECTOR_STORE_AVAILABLE:
        return False
    vs = get_vector_store()
    if vs is None:
        return False
    return vs.init_collection(workspace_name, vector_size=vector_size)


def index_content(workspace_name: str, content_paths: list, content_type: str = 'datasets') -> dict:
    """
    Index content into the configured vector store.

    Args:
        workspace_name: Name of the workspace
        content_paths: List of file/directory paths to index
        content_type: Type of content ('datasets', 'runbooks')

    Returns:
        Dictionary with indexing stats
    """
    # Load chunking module
    if not _load_chunking():
        return {'error': 'Chunking module not available', 'indexed': 0}

    if not VECTOR_STORE_AVAILABLE:
        return {'error': 'Vector store module unavailable', 'indexed': 0}

    vs = get_vector_store()
    model = _get_embedding_model()

    if vs is None or model is None:
        return {'error': 'RAG not available', 'indexed': 0}

    embedding_model_name = os.environ.get('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')

    # Ensure storage exists for this workspace
    if not init_collection(workspace_name, _get_embedding_dimension(model)):
        return {'error': 'Failed to initialize collection', 'indexed': 0}

    # Configure chunking for RAG (smaller chunks, title extraction)
    config = ChunkConfig(
        chunk_size=500,
        chunk_overlap=50,
        extract_title=True,
        category=content_type
    )

    # Chunk all files
    chunks = chunk_files(content_paths, config)

    # Generate embeddings and build VectorStore Point objects
    points = []
    for idx, chunk in enumerate(chunks):
        filename = chunk.metadata.get('filename', '')
        title = chunk.metadata.get('title', '')

        # Build embedding text with document context for better semantic matching
        embedding_parts = []
        if filename:
            readable_filename = filename.rsplit('.', 1)[0].replace('-', ' ').replace('_', ' ')
            embedding_parts.append(f"Document: {readable_filename}")
        if title:
            embedding_parts.append(f"Title: {title}")
        embedding_parts.append(chunk.text)

        embedding_text = '\n'.join(embedding_parts)
        embedding = model.encode(embedding_text).tolist()

        # Strip large/duplicate fields from metadata (text + structural fields are stored
        # in dedicated columns / payload keys; the rest goes into the JSONB metadata column).
        chunk_meta = {
            k: v
            for k, v in chunk.metadata.items()
            if k not in ('text', 'filename', 'title', 'content_type', 'category',
                         'source_file', 'char_start', 'char_end', 'chunk_index')
        }

        chunk_uid = f"{workspace_name}-{idx}"

        points.append(VectorStorePoint(
            chunk_uid=chunk_uid,
            vector=embedding,
            text=chunk.text,
            chunk_index=chunk.metadata.get('chunk_index', idx),
            char_start=chunk.metadata.get('char_start'),
            char_end=chunk.metadata.get('char_end'),
            filename=filename or None,
            title=title or None,
            content_type=content_type,
            source_path=chunk.metadata.get('source_file'),
            embedding_model=embedding_model_name,
            metadata=chunk_meta,
        ))

    written = vs.upsert_points(workspace_name, points) if points else 0

    return {
        'backend': vs.backend_name,
        'collection': get_collection_name(workspace_name),
        'workspace': workspace_name,
        'indexed': written,
        'content_type': content_type,
    }


def find_full_document(workspace_name: str, document_hint: str,
                       search_terms: List[str]) -> Optional[Dict]:
    """
    Find and retrieve a complete document by name/hint.

    This is used when query preprocessing detects a document lookup request
    (e.g., "show me my biography"). Instead of returning chunks, we find
    the best matching document and return its full content.

    Args:
        workspace_name: Name of the workspace
        document_hint: Extracted document name hint (e.g., "biography")
        search_terms: Additional search terms from the query

    Returns:
        Dict with 'content', 'filename', 'source_file' if found, None otherwise
    """
    if not VECTOR_STORE_AVAILABLE:
        return None
    vs = get_vector_store()
    if vs is None:
        return None

    try:
        # Pull all chunks for this workspace via the abstraction. The cap of
        # 5,000 bounds the in-memory document map for full-document recovery.
        hits = vs.scroll_documents(workspace_name, limit=5000)
        if not hits:
            return None

        # Group chunks by source file and score each file
        file_chunks: Dict[str, List] = {}
        for hit in hits:
            source_file = hit.metadata.get('source_file', '') or hit.metadata.get('source_path', '')
            if source_file:
                file_chunks.setdefault(source_file, []).append(hit.metadata)

        # Score each file based on how well it matches the hint and terms
        file_scores = []
        hint_words = set(document_hint.lower().replace('-', ' ').replace('_', ' ').split())
        term_set = set(search_terms)

        for source_file, chunks in file_chunks.items():
            filename = chunks[0].get('filename', '') if chunks else ''
            title = chunks[0].get('title', '') if chunks else ''

            # Convert filename to word set
            filename_clean = filename.lower().rsplit('.', 1)[0] if filename else ''
            filename_words = set(filename_clean.replace('-', ' ').replace('_', ' ').split())

            # Convert title to word set
            title_words = set(title.lower().split()) if title else set()

            score = 0

            # Score based on hint matching
            hint_in_filename = hint_words & filename_words
            hint_in_title = hint_words & title_words
            if hint_in_filename:
                score += 10 * len(hint_in_filename)  # Strong boost for filename match
            if hint_in_title:
                score += 5 * len(hint_in_title)  # Good boost for title match

            # Score based on search terms
            terms_in_filename = term_set & filename_words
            terms_in_title = term_set & title_words
            if terms_in_filename:
                score += 3 * len(terms_in_filename)
            if terms_in_title:
                score += 2 * len(terms_in_title)

            # Check for substring match (e.g., "bio" in "biography")
            if document_hint.lower() in filename_clean:
                score += 15  # Strong boost for substring match

            if score > 0:
                file_scores.append((source_file, filename, score, chunks))

        if not file_scores:
            return None

        # Sort by score and get best match
        file_scores.sort(key=lambda x: x[2], reverse=True)
        best_source, best_filename, best_score, best_chunks = file_scores[0]

        logger.info(f"Full document match: {best_filename} (score: {best_score})")

        # Reconstruct full document from chunks
        # Sort chunks by char_start to maintain order
        sorted_chunks = sorted(best_chunks, key=lambda c: c.get('char_start', 0))

        # Merge chunks, removing overlapping content
        full_content = ""
        last_end = 0
        for chunk in sorted_chunks:
            chunk_start = chunk.get('char_start', 0)
            chunk_text = chunk.get('text', '')

            if chunk_start >= last_end:
                # No overlap, append full chunk
                full_content += chunk_text
            else:
                # Overlap - only add non-overlapping part
                overlap = last_end - chunk_start
                if overlap < len(chunk_text):
                    full_content += chunk_text[overlap:]

            last_end = chunk.get('char_end', chunk_start + len(chunk_text))

        return {
            'content': full_content.strip(),
            'filename': best_filename,
            'source_file': best_source,
            'score': best_score,
            'title': sorted_chunks[0].get('title', '') if sorted_chunks else '',
            'content_type': sorted_chunks[0].get('content_type', 'datasets') if sorted_chunks else 'datasets'
        }

    except Exception as e:
        logger.error(f"Full document retrieval failed: {e}")
        return None


def search(workspace_name: str, query: str, limit: int = 5,
           content_type: Optional[str] = None,
           use_preprocessing: bool = True) -> list:
    """
    Search for relevant content using semantic similarity.

    Args:
        workspace_name: Name of the workspace to search
        query: Search query
        limit: Maximum number of results
        content_type: Filter by content type ('datasets', 'runbooks', or None for all)
        use_preprocessing: If True, preprocess query (expand contractions, extract terms)

    Returns:
        List of search results with text and metadata
    """
    if not VECTOR_STORE_AVAILABLE:
        return []
    vs = get_vector_store()
    model = _get_embedding_model()

    if vs is None or model is None:
        return []

    # Preprocess query for better matching
    if use_preprocessing:
        query_info = preprocess_query(query)
        search_query = query_info['processed_query']
        search_terms = query_info['search_terms']
    else:
        search_query = query
        search_terms = set(query.lower().split())

    try:
        # Generate query embedding using preprocessed query
        query_vector = model.encode(search_query).tolist()

        hits = vs.search(
            workspace_name,
            query_vector=query_vector,
            limit=limit,
            content_type=content_type,
        )

        if not hits:
            return []

        # Format results
        formatted = []
        for hit in hits:
            text = hit.text
            if not text:
                try:
                    source_file = hit.metadata.get('source_file', '') or hit.metadata.get('source_path', '')
                    char_start = hit.metadata.get('char_start', 0) or 0
                    char_end = hit.metadata.get('char_end', 0) or 0
                    with open(source_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        text = content[char_start:char_end]
                except Exception:
                    text = "[Content not available]"

            formatted.append({
                'text': text,
                'score': hit.score,
                'metadata': dict(hit.metadata),
            })

        # Re-rank: boost results where query terms appear in filename or title
        # This improves results for queries like "show me my biography" where
        # semantic search might not prioritize exact document name matches
        # Use preprocessed search_terms which have contractions expanded and stop words removed
        query_terms = set(search_terms) if isinstance(search_terms, list) else search_terms
        for item in formatted:
            # `.get(..., '')` doesn't help when the key is present-but-None
            # (which happens for chunks indexed without a title). Coerce.
            filename = (item['metadata'].get('filename') or '').lower()
            title = (item['metadata'].get('title') or '').lower()

            # Check for exact term matches in filename
            filename_words = set(filename.replace('-', ' ').replace('_', ' ').replace('.md', '').split())
            title_words = set(title.split()) if title else set()

            # Boost score if query terms appear in filename or title
            matching_filename_terms = query_terms & filename_words
            matching_title_terms = query_terms & title_words

            if matching_filename_terms:
                # Significant boost for filename matches (increased from 0.3)
                item['score'] += 0.5 * len(matching_filename_terms)
            if matching_title_terms:
                # Good boost for title matches (increased from 0.2)
                item['score'] += 0.3 * len(matching_title_terms)

        # Re-sort by boosted score
        formatted.sort(key=lambda x: x['score'], reverse=True)

        return formatted
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []


def search_across_workspaces(workspaces: List[str], query: str, limit: int = 5,
                             content_type: Optional[str] = None,
                             use_preprocessing: bool = True) -> list:
    """Vector search across multiple workspaces, RRF-merged.

    For each workspace, run the standard ``search`` and tag every result with
    its source workspace. Then fuse rankings via reciprocal rank fusion so a
    cross-workspace top-N reflects rank consistency, not raw scores (which
    aren't directly comparable across workspaces with different corpus
    distributions).

    Returns the top ``limit`` results, each annotated with
    ``metadata['source_workspace']``.
    """
    if not workspaces:
        return []
    if len(workspaces) == 1:
        # Single-workspace path is identical to plain ``search``; preserves
        # the original score for callers that depend on it.
        results = search(workspaces[0], query, limit=limit,
                         content_type=content_type,
                         use_preprocessing=use_preprocessing)
        for r in results:
            r['metadata']['source_workspace'] = workspaces[0]
        return results

    per_workspace = []
    for ws in workspaces:
        ws_results = search(ws, query, limit=limit,
                            content_type=content_type,
                            use_preprocessing=use_preprocessing)
        if not ws_results:
            continue
        for r in ws_results:
            r['metadata']['source_workspace'] = ws
        per_workspace.append([(r, r.get('score', 0.0)) for r in ws_results])

    if not per_workspace:
        return []

    fused = reciprocal_rank_fusion(per_workspace)
    output = []
    for doc, rrf_score in fused[:limit]:
        result = doc.copy()
        result['rrf_score'] = rrf_score
        if 'score' not in result:
            result['score'] = rrf_score
        output.append(result)
    return output


def get_relevant_context(workspace_name, query: str,
                         max_tokens: int = 16000,
                         user_model: Optional[str] = None,
                         use_phase2: bool = True,
                         use_phase3: bool = True,
                         additional_workspaces: Optional[List[str]] = None) -> str:
    """
    Get relevant context for a query, formatted for LLM consumption.

    This is the main entry point for RAG-augmented prompts.

    Phase 1 Improvements:
    - Increased default max_tokens from 2000 to 16000 (8x increase)
    - Full document retrieval for targeted queries ("show me my biography")
    - Query preprocessing with contraction expansion
    - Enhanced filename/title matching

    Phase 2 Improvements:
    - LLM-powered query planning using provider's fast model
    - Multi-query expansion (5-7 variations for better recall)
    - HyDE (Hypothetical Document Embeddings)
    - Provider-agnostic model selection via engines.yaml categories

    Phase 3 Improvements:
    - Hybrid search (vector + BM25 keyword search)
    - Reciprocal Rank Fusion for result merging
    - LLM-based reranking with provider's fast model

    Args:
        workspace_name: Name of the workspace
        query: User's query
        max_tokens: Maximum tokens for retrieved context (default: 16000)
        user_model: User's selected model (for provider-specific fast model)
        use_phase2: Enable Phase 2 LLM-powered features (default: True)
        use_phase3: Enable Phase 3 hybrid search and reranking (default: True)

    Returns:
        Formatted context string to include in the prompt
    """
    # Cross-workspace fan-out: when additional_workspaces is provided (or
    # auto-detected via the skills workspace), retrieve context from each in
    # turn and concatenate the formatted blocks. Each block stays workspace-
    # scoped so chunk identity, char ranges, and full-document retrieval
    # behave correctly within their own corpus.
    auto_extra: List[str] = []
    if additional_workspaces is None:
        # Auto-include the canonical "skills" workspace when it has content
        # and it's not already the primary workspace.
        # Demo mode swaps in the demo-scoped skills workspace so the fan-out
        # cannot pull real skill content from the host vector store.
        try:
            from ragbot.demo import (
                is_demo_mode,
                DEMO_SKILLS_WORKSPACE_NAME,
            )
            skills_ws = DEMO_SKILLS_WORKSPACE_NAME if is_demo_mode() else 'skills'
        except ImportError:
            skills_ws = 'skills'
        if VECTOR_STORE_AVAILABLE and workspace_name != skills_ws:
            vs = get_vector_store()
            if vs is not None:
                info = vs.get_collection_info(skills_ws)
                if info and (info.get('count') or 0) > 0:
                    auto_extra = [skills_ws]
        effective_extra = auto_extra
    else:
        effective_extra = [w for w in additional_workspaces if w != workspace_name]

    if effective_extra:
        budget_per_ws = max(1024, max_tokens // (1 + len(effective_extra)))
        primary = get_relevant_context(
            workspace_name, query,
            max_tokens=budget_per_ws,
            user_model=user_model,
            use_phase2=use_phase2,
            use_phase3=use_phase3,
            additional_workspaces=[],   # break recursion
        )
        blocks = [primary] if primary else []
        for ws in effective_extra:
            extra_block = get_relevant_context(
                ws, query,
                max_tokens=budget_per_ws,
                user_model=user_model,
                use_phase2=use_phase2,
                use_phase3=use_phase3,
                additional_workspaces=[],
            )
            if extra_block:
                blocks.append(f"<!-- workspace:{ws} -->\n{extra_block}")
        return '\n\n'.join(blocks) if blocks else ""

    # Step 1: Preprocess the query (Phase 1 or Phase 2)
    if use_phase2:
        query_info = enhanced_preprocess_query(
            query,
            user_model=user_model,
            workspace=workspace_name,
            use_planner=True,
            use_multi_query=True,
            use_hyde=True
        )
        logger.info(f"Phase 2 preprocessing: enabled={query_info.get('phase2_enabled')}")
    else:
        query_info = preprocess_query(query)
        query_info['expanded_queries'] = [query_info['processed_query']]
        query_info['hyde_document'] = None

    # Step 2: For document lookup requests, try full document retrieval first
    if query_info['is_document_request'] and query_info['document_hint']:
        logger.info(f"Document request detected: '{query_info['document_hint']}'")

        full_doc = find_full_document(
            workspace_name,
            query_info['document_hint'],
            query_info['search_terms']
        )

        if full_doc:
            # Check if full document fits in budget
            doc_tokens = len(full_doc['content']) // 4
            if doc_tokens <= max_tokens:
                logger.info(f"Returning full document: {full_doc['filename']} ({doc_tokens} tokens)")
                content_type = full_doc.get('content_type', 'datasets')
                filename = full_doc.get('filename', 'unknown')
                title = full_doc.get('title', '')

                header = f"[{content_type}: {filename}]"
                if title:
                    header = f"[{content_type}: {filename} - {title}]"

                return f"<retrieved_context>\n{header}\n\n{full_doc['content']}\n</retrieved_context>"
            else:
                logger.info(f"Full document too large ({doc_tokens} tokens > {max_tokens}), using chunks")

    # Step 3: Multi-query search with result fusion
    # Phase 2: Search with multiple query variations and merge results
    # Phase 3: Use hybrid search (vector + BM25) with RRF
    all_results = []
    seen_chunks = set()  # Track by (filename, char_start) to avoid duplicates

    expanded_queries = query_info.get('expanded_queries', [query])

    # Search with each expanded query
    for expanded_query in expanded_queries[:7]:  # Limit to 7 queries max
        # Phase 3: Use hybrid search if enabled
        if use_phase3:
            results = hybrid_search(
                workspace_name, expanded_query, limit=50,
                use_bm25=True, use_rrf=True
            )
        else:
            results = search(workspace_name, expanded_query, limit=50)

        for result in results:
            # Create a unique key for this chunk
            filename = result['metadata'].get('filename', '')
            char_start = result['metadata'].get('char_start', 0)
            chunk_key = (filename, char_start)

            if chunk_key not in seen_chunks:
                seen_chunks.add(chunk_key)
                all_results.append(result)

    # Step 3b: HyDE search (if available and not document lookup)
    hyde_doc = query_info.get('hyde_document')
    if hyde_doc:
        logger.info("Running HyDE search with hypothetical document")
        if use_phase3:
            hyde_results = hybrid_search(
                workspace_name, hyde_doc, limit=30,
                use_bm25=True, use_rrf=True
            )
        else:
            hyde_results = search(workspace_name, hyde_doc, limit=30, use_preprocessing=False)

        for result in hyde_results:
            filename = result['metadata'].get('filename', '')
            char_start = result['metadata'].get('char_start', 0)
            chunk_key = (filename, char_start)

            if chunk_key not in seen_chunks:
                seen_chunks.add(chunk_key)
                # Slightly boost HyDE results as they bridge semantic gap
                result['score'] *= 1.1
                all_results.append(result)

    # Sort all results by score
    all_results.sort(key=lambda x: x.get('rrf_score', x.get('score', 0)), reverse=True)

    # Step 3c: LLM-based reranking (Phase 3)
    if use_phase3 and all_results:
        logger.info(f"Running LLM reranking on {len(all_results)} results")
        all_results = rerank_with_llm(
            query, all_results,
            user_model=user_model,
            workspace=workspace_name,
            top_k=20  # Only rerank top 20 for efficiency
        )
        # Re-sort by combined score after reranking
        all_results.sort(key=lambda x: x.get('combined_score', x.get('score', 0)), reverse=True)

    if not all_results:
        return ""

    # Build context string within token budget
    context_parts = []
    current_tokens = 0
    seen_files = set()

    for result in all_results:
        text = result['text']
        # Rough token estimate
        text_tokens = len(text) // 4

        # Skip oversized chunks rather than terminating: results are
        # ranked by score, so an oversized top-ranked chunk would lock
        # out smaller, lower-ranked chunks that still fit. This matters
        # for skill-script chunks (whole-file) that can be much larger
        # than typical markdown chunks.
        if current_tokens + text_tokens > max_tokens:
            continue

        source = result['metadata'].get('filename', 'unknown')
        content_type = result['metadata'].get('content_type', 'content')
        score = result['score']
        title = result['metadata'].get('title', '')

        # Track which files we've included
        seen_files.add(source)

        header = f"[{content_type}: {source} (relevance: {score:.2f})]"
        if title and source not in title:
            header = f"[{content_type}: {source} - {title} (relevance: {score:.2f})]"

        context_parts.append(f"{header}\n{text}\n")
        current_tokens += text_tokens

    if not context_parts:
        return ""

    # Add summary of sources and Phase 2/3 info
    phase_notes = []
    if use_phase2 and query_info.get('phase2_enabled'):
        phase_notes.append(f"Phase2: {len(expanded_queries)} queries, HyDE={'yes' if hyde_doc else 'no'}")

    if use_phase3:
        # Check if any results have reranking scores
        has_reranking = any(r.get('llm_score') is not None for r in all_results[:10])
        has_rrf = any(r.get('rrf_score') is not None for r in all_results[:10])
        phase3_parts = ["Phase3: hybrid"]
        if has_rrf:
            phase3_parts.append("RRF")
        if has_reranking:
            phase3_parts.append("reranked")
        phase_notes.append(" + ".join(phase3_parts))

    phase_note = f"<!-- {', '.join(phase_notes)} -->\n" if phase_notes else ""

    sources_note = f"<!-- Sources: {', '.join(sorted(seen_files)[:10])} -->\n"

    return "<retrieved_context>\n" + phase_note + sources_note + "\n---\n".join(context_parts) + "</retrieved_context>"


def index_workspace(workspace_name: str, ai_knowledge_paths: dict) -> dict:
    """
    Index all content for a workspace from ai-knowledge paths.

    Args:
        workspace_name: Name of the workspace
        ai_knowledge_paths: Dict with 'instructions', 'datasets' paths

    Returns:
        Indexing stats
    """
    stats = {'workspace': workspace_name, 'datasets': 0, 'runbooks': 0}

    # Index datasets/knowledge
    datasets_path = ai_knowledge_paths.get('datasets')
    if datasets_path and os.path.exists(datasets_path):
        result = index_content(workspace_name, [datasets_path], 'datasets')
        stats['datasets'] = result.get('indexed', 0)

    # Note: We don't index instructions - they should always be in the system prompt
    # Runbooks could be indexed if they exist in a separate path

    return stats


def get_index_status(workspace_name: str) -> tuple[bool, int]:
    """
    Get the index status for a workspace.

    Args:
        workspace_name: Name of the workspace

    Returns:
        Tuple of (is_indexed, chunk_count) where chunk_count is the number of
        chunk rows in the workspace's pgvector collection.
    """
    if not VECTOR_STORE_AVAILABLE:
        return False, 0
    vs = get_vector_store()
    if vs is None:
        return False, 0
    try:
        info = vs.get_collection_info(workspace_name)
        if not info:
            return False, 0
        count = int(info.get('count') or 0)
        return count > 0, count
    except Exception as exc:
        logger.error("Failed to get index status: %s", exc)
        return False, 0


def _build_skill_chunks(skill, embedding_model_name: str, model) -> List["VectorStorePoint"]:
    """Generate vector-store Points for every indexable file inside a skill.

    Each chunk is tagged with ``skill_name`` plus a content_type derived
    from the file's classification (``skill``, ``skill_reference``,
    ``skill_script``, ``skill_other``). Markdown content is normally
    chunked; scripts/other-text are stored as a single chunk per file
    (their value is queryability, not semantic mid-document chunking).
    """
    if not _load_chunking():
        return []
    if not VECTOR_STORE_AVAILABLE:
        return []

    from synthesis_engine.skills.model import SkillFileKind  # local to avoid module-load cost

    points: List[VectorStorePoint] = []

    kind_to_content_type = {
        SkillFileKind.SKILL_MD: 'skill',
        SkillFileKind.REFERENCE: 'skill_reference',
        SkillFileKind.SCRIPT: 'skill_script',
        SkillFileKind.OTHER: 'skill_other',
    }

    cfg_md = ChunkConfig(
        chunk_size=500, chunk_overlap=50,
        extract_title=True, category='skill',
    )

    for sf in skill.files:
        if not sf.is_text or not sf.content:
            continue
        content_type = kind_to_content_type.get(sf.kind, 'skill_other')
        ext = os.path.splitext(sf.relative_path)[1].lower()
        is_markdown = ext in ('.md', '.markdown')

        # Markdown gets the standard chunker. Scripts and other text become
        # a single whole-file chunk so retrieval lands the entire artifact.
        if is_markdown:
            chunks = chunk_files([sf.absolute_path], cfg_md)
        else:
            # Synthesise a one-chunk record without invoking the chunker so
            # we keep the original file content intact.
            class _PseudoChunk:
                def __init__(self, text, metadata):
                    self.text = text
                    self.metadata = metadata
            md = {
                'filename': os.path.basename(sf.relative_path),
                'title': os.path.basename(sf.relative_path),
                'category': 'skill_script' if sf.kind is SkillFileKind.SCRIPT else 'skill_other',
                'source_file': sf.absolute_path,
                'chunk_index': 0,
                'char_start': 0,
                'char_end': len(sf.content),
            }
            chunks = [_PseudoChunk(sf.content, md)]

        for idx, chunk in enumerate(chunks):
            filename = chunk.metadata.get('filename', os.path.basename(sf.relative_path))
            title = chunk.metadata.get('title') or skill.name

            embedding_parts = []
            if filename:
                readable = filename.rsplit('.', 1)[0].replace('-', ' ').replace('_', ' ')
                embedding_parts.append(f"Skill: {skill.name}")
                embedding_parts.append(f"Document: {readable}")
            if title and title != filename:
                embedding_parts.append(f"Title: {title}")
            if skill.description:
                embedding_parts.append(f"Skill description: {skill.description[:200]}")
            embedding_parts.append(chunk.text)

            embedding = model.encode('\n'.join(embedding_parts)).tolist()

            chunk_uid = f"{skill.name}::{sf.relative_path}::{idx}"

            metadata = {
                'skill_name': skill.name,
                'skill_path': skill.path,
                'skill_version': skill.version,
                'skill_relative_path': sf.relative_path,
                'skill_file_kind': sf.kind.value,
                'skill_description': skill.description,
            }

            points.append(VectorStorePoint(
                chunk_uid=chunk_uid,
                vector=embedding,
                text=chunk.text,
                chunk_index=idx,
                char_start=chunk.metadata.get('char_start'),
                char_end=chunk.metadata.get('char_end'),
                filename=filename,
                title=title,
                content_type=content_type,
                source_path=sf.absolute_path,
                embedding_model=embedding_model_name,
                metadata=metadata,
            ))

    return points


def index_skills(workspace_name: str = 'skills', skill_roots: Optional[List[str]] = None,
                 only: Optional[List[str]] = None, force: bool = False) -> dict:
    """Index discovered Agent Skills into the configured vector store.

    Args:
        workspace_name: Vector-store workspace to index into (default ``skills``).
            Skills typically live in a dedicated workspace so they can be
            queried alongside any user workspace via cross-workspace search.
        skill_roots: Override the discovery roots. ``None`` uses defaults.
        only: Optional list of skill names to limit indexing to.
        force: If True, clear the workspace before indexing.

    Returns:
        Dict with summary stats ``{backend, workspace, skills_indexed,
        chunks_indexed, skipped, skill_names}``.
    """
    if not VECTOR_STORE_AVAILABLE:
        return {'error': 'Vector store module unavailable', 'skills_indexed': 0, 'chunks_indexed': 0}

    from synthesis_engine.skills import discover_skills

    vs = get_vector_store()
    model = _get_embedding_model()
    if vs is None or model is None:
        return {'error': 'RAG not available', 'skills_indexed': 0, 'chunks_indexed': 0}

    embedding_model_name = os.environ.get('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')

    if not init_collection(workspace_name, _get_embedding_dimension(model)):
        return {'error': 'Failed to initialize collection', 'skills_indexed': 0}

    if force:
        vs.delete_collection(workspace_name)
        init_collection(workspace_name, _get_embedding_dimension(model))

    skills = discover_skills(roots=skill_roots) if skill_roots else discover_skills()
    if only:
        wanted = set(only)
        skills = [s for s in skills if s.name in wanted]

    total_chunks = 0
    indexed_names: List[str] = []
    skipped: List[str] = []
    for skill in skills:
        points = _build_skill_chunks(skill, embedding_model_name, model)
        if not points:
            skipped.append(skill.name)
            continue
        written = vs.upsert_points(workspace_name, points)
        total_chunks += written
        indexed_names.append(skill.name)

    return {
        'backend': vs.backend_name,
        'workspace': workspace_name,
        'skills_indexed': len(indexed_names),
        'chunks_indexed': total_chunks,
        'skipped': skipped,
        'skill_names': indexed_names,
    }


def index_workspace_by_name(workspace_name: str, force: bool = False) -> int:
    """
    Index a workspace by name, automatically discovering its paths.

    This is a convenience wrapper that discovers workspace paths from
    the ai-knowledge repository structure.

    Args:
        workspace_name: Name/dir_name of the workspace
        force: If True, clear existing index first

    Returns:
        Number of documents indexed
    """
    # Import here to avoid circular imports
    import sys
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    try:
        from ragbot import get_workspace
        workspace = get_workspace(workspace_name)
    except Exception as e:
        logger.error(f"Failed to get workspace {workspace_name}: {e}")
        return 0

    if force:
        clear_collection(workspace_name)

    # Get datasets paths (may be a list or a single path)
    datasets = workspace.get('datasets', [])
    if isinstance(datasets, str):
        datasets = [datasets]

    # Filter to existing paths
    datasets = [p for p in datasets if p and os.path.exists(p)]

    if not datasets:
        logger.warning(f"No dataset paths found for workspace {workspace_name}")
        return 0

    # Index the content directly
    result = index_content(workspace_name, datasets, 'datasets')
    return result.get('indexed', 0)


# =============================================================================
# Phase 4: Response Verification Implementation
# =============================================================================

def verify_response(
    query: str,
    response: str,
    context: str,
    user_model: Optional[str] = None,
    workspace: Optional[str] = None
) -> Optional[VerificationResult]:
    """
    Verify that a response is grounded in the retrieved context.

    Uses the provider's fast model to extract claims from the response
    and check each one against the context for supporting evidence.

    Args:
        query: Original user query
        response: Generated response to verify
        context: Retrieved context that was used to generate the response
        user_model: User's selected model (for provider selection)
        workspace: Workspace name for API key resolution

    Returns:
        VerificationResult with confidence score and claim details,
        or None if verification fails
    """
    if not response or not context:
        logger.warning("Cannot verify: missing response or context")
        return None

    # Truncate context if too long (keep verification prompt manageable)
    max_context_chars = 8000
    truncated_context = context[:max_context_chars]
    if len(context) > max_context_chars:
        truncated_context += "\n... [context truncated for verification]"

    # Build verification prompt
    prompt = VERIFIER_PROMPT.format(
        context=truncated_context,
        response=response
    )

    # Call fast LLM for verification
    llm_response = _call_fast_llm(prompt, user_model, workspace)

    if not llm_response:
        logger.warning("Verification skipped: no LLM response")
        return None

    try:
        # Parse JSON response
        json_str = llm_response
        if '```json' in json_str:
            json_str = json_str.split('```json')[1].split('```')[0]
        elif '```' in json_str:
            json_str = json_str.split('```')[1].split('```')[0]

        result_data = json.loads(json_str.strip())

        # Extract claims
        claims = []
        for claim_data in result_data.get('claims', []):
            status_str = claim_data.get('status', 'UNSUPPORTED').upper()
            try:
                status = ClaimStatus(status_str.lower())
            except ValueError:
                status = ClaimStatus.UNSUPPORTED

            claims.append(VerifiedClaim(
                claim=claim_data.get('claim', ''),
                status=status,
                evidence=claim_data.get('evidence'),
                reasoning=claim_data.get('reasoning', '')
            ))

        # Calculate confidence if not provided by LLM
        confidence = result_data.get('overall_confidence', 0.0)
        if not confidence and claims:
            confidence = calculate_confidence(claims)

        verification = VerificationResult(
            confidence=confidence,
            is_grounded=result_data.get('is_grounded', confidence >= 0.7),
            claims=claims,
            suggested_corrections=result_data.get('suggested_corrections', [])
        )

        logger.info(f"Verification complete: confidence={verification.confidence:.2f}, "
                   f"grounded={verification.is_grounded}, claims={len(claims)}")

        return verification

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"Failed to parse verification response: {e}")
        return None


def calculate_confidence(claims: List[VerifiedClaim]) -> float:
    """
    Calculate confidence score from claim verification results.

    Formula:
    - Base: (supported_claims + 0.5 * partial_claims) / total_claims
    - Penalty: -0.1 for each UNSUPPORTED claim
    - Bonus: +0.1 if no UNSUPPORTED claims

    Args:
        claims: List of verified claims

    Returns:
        Confidence score from 0.0 to 1.0
    """
    if not claims:
        return 1.0  # No claims to verify = assume grounded

    supported = sum(1 for c in claims if c.status == ClaimStatus.SUPPORTED)
    partial = sum(1 for c in claims if c.status == ClaimStatus.PARTIALLY_SUPPORTED)
    unsupported = sum(1 for c in claims if c.status == ClaimStatus.UNSUPPORTED)
    total = len(claims)

    # Base score
    base = (supported + 0.5 * partial) / total

    # Apply penalty for unsupported claims
    penalty = 0.1 * unsupported

    # Bonus if all claims are supported
    bonus = 0.1 if unsupported == 0 else 0

    confidence = base - penalty + bonus

    # Clamp to [0.0, 1.0]
    return max(0.0, min(1.0, confidence))


def generate_crag_queries(
    query: str,
    unsupported_claims: List[VerifiedClaim],
    user_model: Optional[str] = None,
    workspace: Optional[str] = None
) -> List[str]:
    """
    Generate targeted search queries to find evidence for unsupported claims.

    Args:
        query: Original user query
        unsupported_claims: Claims that need additional evidence
        user_model: User's selected model
        workspace: Workspace name

    Returns:
        List of targeted search queries
    """
    if not unsupported_claims:
        return []

    # Format claims for the prompt
    claims_text = "\n".join([f"- {c.claim}" for c in unsupported_claims])

    prompt = CRAG_QUERY_PROMPT.format(
        query=query,
        claims=claims_text
    )

    response = _call_fast_llm(prompt, user_model, workspace)

    if response:
        try:
            json_str = response
            if '```json' in json_str:
                json_str = json_str.split('```json')[1].split('```')[0]
            elif '```' in json_str:
                json_str = json_str.split('```')[1].split('```')[0]

            result_data = json.loads(json_str.strip())
            queries = result_data.get('queries', [])
            logger.info(f"Generated {len(queries)} CRAG queries")
            return queries

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse CRAG query response: {e}")

    # Fallback: extract key terms from unsupported claims
    fallback_queries = []
    for claim in unsupported_claims[:3]:
        # Simple extraction of key terms
        words = claim.claim.split()
        if len(words) >= 3:
            fallback_queries.append(' '.join(words[:5]))

    return fallback_queries


def corrective_rag_loop(
    query: str,
    original_response: str,
    verification: VerificationResult,
    context: str,
    workspace_name: str,
    user_model: Optional[str] = None,
    max_attempts: int = 2,
    confidence_threshold: float = 0.7,
    regenerate_callback=None
) -> CRAGResult:
    """
    Attempt to correct a poorly grounded response through additional retrieval.

    This implements CRAG (Corrective RAG):
    1. Identify unsupported claims from verification
    2. Generate targeted queries to find supporting evidence
    3. Retrieve additional context
    4. Regenerate response with enhanced context
    5. Re-verify (up to max_attempts)

    Args:
        query: Original user query
        original_response: The response that failed verification
        verification: Initial verification result
        context: Original retrieved context
        workspace_name: Workspace for additional retrieval
        user_model: User's selected model
        max_attempts: Maximum correction attempts (default: 2)
        confidence_threshold: Target confidence to stop CRAG (default: 0.7)
        regenerate_callback: Optional callback to regenerate response
            Signature: (query: str, context: str) -> str
            If not provided, returns improved context without regeneration

    Returns:
        CRAGResult with final response and verification history
    """
    verification_history = [verification]
    current_response = original_response
    current_context = context
    additional_context_used = False

    for attempt in range(max_attempts):
        logger.info(f"CRAG attempt {attempt + 1}/{max_attempts}")

        # Find unsupported claims
        unsupported = [c for c in verification.claims
                      if c.status == ClaimStatus.UNSUPPORTED]

        if not unsupported:
            logger.info("No unsupported claims found, stopping CRAG")
            break

        # Generate targeted queries
        crag_queries = generate_crag_queries(
            query, unsupported, user_model, workspace_name
        )

        if not crag_queries:
            logger.info("No CRAG queries generated, stopping")
            break

        # Retrieve additional context for each query
        new_context_parts = []
        for crag_query in crag_queries[:3]:  # Limit to 3 queries
            results = hybrid_search(
                workspace_name, crag_query, limit=5,
                use_bm25=True, use_rrf=True
            )

            for result in results[:3]:  # Top 3 per query
                text = result.get('text', '')
                if text and text not in current_context:
                    source = result.get('metadata', {}).get('filename', 'unknown')
                    new_context_parts.append(f"[Additional: {source}]\n{text}")
                    additional_context_used = True

        if not new_context_parts:
            logger.info("No additional context found, stopping CRAG")
            break

        # Enhance context with new findings
        enhanced_context = current_context + "\n\n--- Additional Context ---\n" + \
                          "\n\n".join(new_context_parts)

        # Regenerate response if callback provided
        if regenerate_callback:
            try:
                current_response = regenerate_callback(query, enhanced_context)
                logger.info(f"Regenerated response ({len(current_response)} chars)")
            except Exception as e:
                logger.warning(f"Response regeneration failed: {e}")
                # Continue with original response but enhanced context
                pass

        current_context = enhanced_context

        # Re-verify the response
        new_verification = verify_response(
            query, current_response, enhanced_context,
            user_model, workspace_name
        )

        if new_verification:
            verification_history.append(new_verification)
            verification = new_verification

            if verification.confidence >= confidence_threshold:
                logger.info(f"CRAG success: confidence improved to {verification.confidence:.2f}")
                break
        else:
            logger.warning("Re-verification failed")
            break

    # Return final result
    return CRAGResult(
        final_response=current_response,
        confidence=verification.confidence,
        attempts=len(verification_history) - 1,  # -1 for initial verification
        verification_history=verification_history,
        additional_context_used=additional_context_used
    )


def verify_and_correct(
    query: str,
    response: str,
    context: str,
    workspace_name: str,
    user_model: Optional[str] = None,
    enable_verification: bool = True,
    enable_crag: bool = True,
    confidence_threshold: float = 0.7,
    regenerate_callback=None
) -> Dict[str, Any]:
    """
    Main entry point for Phase 4: verify a response and optionally correct it.

    This is the function to call after generating a response to check
    for hallucinations and potentially improve accuracy through CRAG.

    Args:
        query: Original user query
        response: Generated response to verify
        context: Retrieved context used for generation
        workspace_name: Workspace for additional retrieval
        user_model: User's selected model
        enable_verification: Enable verification (default: True)
        enable_crag: Enable CRAG correction loop (default: True)
        confidence_threshold: Trigger CRAG below this confidence (default: 0.7)
        regenerate_callback: Callback to regenerate response with new context

    Returns:
        Dict with:
        - response: Final response (may be corrected)
        - confidence: Confidence score (0.0-1.0)
        - is_grounded: Whether response is well-grounded
        - verification: Full verification details (if enabled)
        - crag_used: Whether CRAG was triggered
        - crag_attempts: Number of CRAG attempts
    """
    result = {
        'response': response,
        'confidence': 1.0,
        'is_grounded': True,
        'verification': None,
        'crag_used': False,
        'crag_attempts': 0
    }

    if not enable_verification:
        return result

    # Step 1: Verify the response
    verification = verify_response(
        query, response, context, user_model, workspace_name
    )

    if not verification:
        # Verification failed - return original response
        result['confidence'] = 0.5  # Unknown confidence
        return result

    result['verification'] = {
        'confidence': verification.confidence,
        'is_grounded': verification.is_grounded,
        'claims_checked': len(verification.claims),
        'claims_supported': sum(1 for c in verification.claims
                               if c.status == ClaimStatus.SUPPORTED),
        'claims_unsupported': sum(1 for c in verification.claims
                                  if c.status == ClaimStatus.UNSUPPORTED),
        'suggested_corrections': verification.suggested_corrections
    }
    result['confidence'] = verification.confidence
    result['is_grounded'] = verification.is_grounded

    # Step 2: CRAG if needed
    if enable_crag and verification.confidence < confidence_threshold:
        logger.info(f"Triggering CRAG: confidence {verification.confidence:.2f} < {confidence_threshold}")

        crag_result = corrective_rag_loop(
            query=query,
            original_response=response,
            verification=verification,
            context=context,
            workspace_name=workspace_name,
            user_model=user_model,
            max_attempts=2,
            confidence_threshold=confidence_threshold,
            regenerate_callback=regenerate_callback
        )

        result['response'] = crag_result.final_response
        result['confidence'] = crag_result.confidence
        result['crag_used'] = True
        result['crag_attempts'] = crag_result.attempts
        result['verification']['crag_improved'] = (
            crag_result.confidence > verification.confidence
        )

        # Update grounded status based on final confidence
        result['is_grounded'] = crag_result.confidence >= confidence_threshold

    return result


def clear_collection(workspace_name: str) -> bool:
    """
    Clear all indexed content for a workspace.

    Args:
        workspace_name: Name of the workspace

    Returns:
        True if successful
    """
    if not VECTOR_STORE_AVAILABLE:
        return False
    vs = get_vector_store()
    if vs is None:
        return False
    try:
        ok = vs.delete_collection(workspace_name)
        if ok:
            logger.info("Cleared collection: %s (backend=%s)", workspace_name, vs.backend_name)
        return ok
    except Exception as exc:
        logger.error("Failed to clear collection: %s", exc)
        return False
