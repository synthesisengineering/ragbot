"""Reranking orders results on one scale, end to end.

Retrieval scores differ in scale by source: vector hits carry cosine
similarity (0-1), full-text hits carry a rank that can exceed 8, and fused
hits carry an RRF score near 0.03. On 2026-11-06 the reranker scored the one
relevant chunk 10 and the rest 0, but mixing those judgments with raw scores
put keyword hits it scored 0 ahead of it, and the context labelled them with
higher "relevance". These tests pin that the LLM's judgment decides the
order, that unscored and unreranked results keep their retrieval order, and
that hybrid search gives every result an RRF score.
"""

from __future__ import annotations

import os
import re
import sys
from unittest.mock import patch

_SRC = os.path.join(os.path.dirname(__file__), '..', 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import rag  # noqa: E402
from rag import get_relevant_context, hybrid_search, rerank_with_llm  # noqa: E402


def _hit(name, rrf, raw, start=0):
    return {
        'text': f'{name} text',
        'score': raw,
        'rrf_score': rrf,
        'metadata': {'filename': f'{name}.md', 'char_start': start},
    }


def _live_shape():
    """The 2026-11-06 case: the relevant chunk leads on RRF but has a cosine
    score, while keyword hits behind it carry full-text ranks near 8."""
    return [
        _hit('answer', 0.0328, 0.69),
        _hit('kw-a', 0.0323, 8.1),
        _hit('kw-b', 0.0304, 8.8),
        _hit('kw-c', 0.0273, 7.9),
    ]


class TestRerankScale:
    def test_llm_judgment_beats_raw_keyword_scores(self):
        with patch('rag._call_fast_llm', return_value='{"scores": [10, 0, 0, 0]}'):
            ranked = rerank_with_llm('q', _live_shape())

        assert ranked[0]['metadata']['filename'] == 'answer.md'
        assert all(0 <= r['combined_score'] <= 1 for r in ranked)

    def test_unscored_tail_stays_behind_reranked_results(self):
        head = [_hit(f'h{i}', 0.03 - i * 0.0001, 0.5) for i in range(3)]
        tail = [_hit(f't{i}', 0.01, 50.0) for i in range(3)]  # huge raw scores
        with patch('rag._call_fast_llm', return_value='{"scores": [5, 5, 5]}'):
            ranked = rerank_with_llm('q', head + tail, top_k=3)

        assert [r['metadata']['filename'] for r in ranked[3:]] == ['t0.md', 't1.md', 't2.md']
        assert max(r['combined_score'] for r in ranked[3:]) < min(r['combined_score'] for r in ranked[:3])
        assert all(r['llm_score'] is None for r in ranked[3:])

    def test_skipped_reranking_keeps_retrieval_order_and_scale(self):
        results = _live_shape()
        with patch('rag._call_fast_llm', return_value=None):
            ranked = rerank_with_llm('q', results)

        assert [r['metadata']['filename'] for r in ranked] == ['answer.md', 'kw-a.md', 'kw-b.md', 'kw-c.md']
        combined = [r['combined_score'] for r in ranked]
        assert combined == sorted(combined, reverse=True)
        assert combined[0] == 1.0

    def test_unparseable_scores_fall_back_like_no_reply(self):
        with patch('rag._call_fast_llm', return_value='{"scores": ["high", "low"]}'):
            ranked = rerank_with_llm('q', _live_shape())

        assert [r['metadata']['filename'] for r in ranked][0] == 'answer.md'
        assert all(r['llm_score'] is None for r in ranked)


class TestContextOrderAndLabels:
    def test_context_leads_with_the_reranked_chunk_and_labels_it_highest(self):
        with patch('rag.hybrid_search', return_value=_live_shape()), \
             patch('rag._call_fast_llm', return_value='{"scores": [10, 0, 0, 0]}'):
            context = get_relevant_context(
                'personal', 'what did they advise', use_phase2=False, additional_workspaces=[],
            )

        headers = re.findall(r'\[\w+: (\S+) \(relevance: ([\d.]+)\)\]', context)
        assert headers[0][0] == 'answer.md'
        relevances = [float(r) for _, r in headers]
        assert relevances == sorted(relevances, reverse=True)
        assert max(relevances) <= 1.0


class _NoKeywordHits:
    """A store whose full-text search finds nothing for the query."""

    def keyword_search(self, *args, **kwargs):
        return []

    def scroll_documents(self, *args, **kwargs):
        return []


class TestHybridSearchFusion:
    def test_vector_only_hits_still_carry_rrf_scores(self):
        vector = [
            {'text': 'a', 'score': 0.9, 'metadata': {'filename': 'a.md', 'char_start': 0}},
            {'text': 'b', 'score': 0.7, 'metadata': {'filename': 'b.md', 'char_start': 0}},
        ]
        with patch('rag.search', return_value=vector), \
             patch('rag.get_vector_store', return_value=_NoKeywordHits()):
            results = hybrid_search('personal', 'query')

        assert [r['metadata']['filename'] for r in results] == ['a.md', 'b.md']
        assert all(0 < r['rrf_score'] < 0.1 for r in results)

    def test_keyword_search_failure_still_fuses_vector_hits(self):
        class _Broken(_NoKeywordHits):
            def keyword_search(self, *args, **kwargs):
                raise RuntimeError('full-text index unavailable')

        vector = [{'text': 'a', 'score': 0.9, 'metadata': {'filename': 'a.md', 'char_start': 0}}]
        with patch('rag.search', return_value=vector), \
             patch('rag.get_vector_store', return_value=_Broken()):
            results = hybrid_search('personal', 'query')

        assert results[0]['metadata']['filename'] == 'a.md'
        assert 'rrf_score' in results[0]
