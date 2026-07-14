"""Tests for RAG Phase 2 features.

Phase 2 adds query intelligence:
- Planner stage using provider's fast model
- Multi-query expansion for better recall
- HyDE (Hypothetical Document Embeddings)
- Provider-agnostic model selection

These tests verify both:
1. Unit functionality (parsing, fallbacks)
2. Provider-agnostic model selection
"""

import pytest
from unittest.mock import patch, MagicMock

# Import functions to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag import (
    _get_fast_model,
    plan_query,
    expand_query,
    generate_hyde_document,
    enhanced_preprocess_query,
    PLANNER_PROMPT,
    MULTI_QUERY_PROMPT,
    HYDE_PROMPT,
)


class TestGetFastModel:
    """Tests for _get_fast_model function."""

    def test_get_fast_model_for_anthropic(self):
        """Fast model for Anthropic should be Haiku."""
        fast = _get_fast_model('anthropic/claude-opus-4-5-20251101')
        assert fast == 'anthropic/claude-haiku-4-5-20251001'

    def test_get_fast_model_for_openai(self):
        """Fast model for OpenAI should be the small-category model from engines.yaml."""
        fast = _get_fast_model('openai/gpt-5.6-terra')
        assert fast == 'openai/gpt-5.6-luna'

    def test_get_fast_model_for_google(self):
        """Fast model for Google should be the small-category Gemini Flash Lite."""
        fast = _get_fast_model('gemini/gemini-3.1-pro-preview')
        assert fast == 'gemini/gemini-3.1-flash-lite-preview'

    def test_get_fast_model_falls_back_to_default(self):
        """When no model specified, use default provider's fast model."""
        fast = _get_fast_model(None)
        # Default is Anthropic, so should get Haiku
        assert fast is not None
        assert 'haiku' in fast.lower() or 'mini' in fast.lower() or 'flash' in fast.lower()

    def test_get_fast_model_with_sonnet(self):
        """Fast model for Sonnet should also be Haiku (same provider)."""
        fast = _get_fast_model('anthropic/claude-sonnet-4-20250514')
        assert fast == 'anthropic/claude-haiku-4-5-20251001'


class TestPlannerPrompt:
    """Tests for planner prompt template."""

    def test_planner_prompt_has_required_fields(self):
        """Planner prompt should include all required output fields."""
        assert 'query_type' in PLANNER_PROMPT
        assert 'retrieval_strategy' in PLANNER_PROMPT
        assert 'filename_hints' in PLANNER_PROMPT
        assert 'answer_style' in PLANNER_PROMPT
        assert 'complexity' in PLANNER_PROMPT

    def test_planner_prompt_has_query_placeholder(self):
        """Planner prompt should have {query} placeholder."""
        assert '{query}' in PLANNER_PROMPT

    def test_planner_prompt_defines_query_types(self):
        """Planner prompt should define all query types."""
        assert 'document_lookup' in PLANNER_PROMPT
        assert 'factual_qa' in PLANNER_PROMPT
        assert 'procedural' in PLANNER_PROMPT
        assert 'multi_step' in PLANNER_PROMPT


class TestPlanQuery:
    """Tests for plan_query function."""

    def test_plan_query_fallback_for_document_request(self):
        """Plan should detect document requests in fallback mode."""
        # Mock _call_fast_llm to return None (simulating no LLM available)
        with patch('rag._call_fast_llm', return_value=None):
            plan = plan_query("show me my biography")

        assert plan['query_type'] == 'document_lookup'
        assert plan['retrieval_strategy'] == 'full_document'
        assert plan['answer_style'] == 'return_content'
        assert plan['used_llm'] is False

    def test_plan_query_fallback_for_general_query(self):
        """Plan should detect general queries in fallback mode."""
        with patch('rag._call_fast_llm', return_value=None):
            plan = plan_query("What is machine learning?")

        assert plan['query_type'] == 'factual_qa'
        assert plan['retrieval_strategy'] == 'semantic_chunks'
        assert plan['answer_style'] == 'synthesize'
        assert plan['used_llm'] is False

    def test_plan_query_parses_llm_response(self):
        """Plan should parse valid LLM JSON response."""
        mock_response = '''{
            "query_type": "procedural",
            "retrieval_strategy": "hybrid",
            "filename_hints": ["guide", "howto"],
            "answer_style": "synthesize",
            "complexity": "moderate"
        }'''

        with patch('rag._call_fast_llm', return_value=mock_response):
            plan = plan_query("How do I write a blog post?")

        assert plan['query_type'] == 'procedural'
        assert plan['retrieval_strategy'] == 'hybrid'
        assert plan['filename_hints'] == ['guide', 'howto']
        assert plan['used_llm'] is True

    def test_plan_query_handles_markdown_wrapped_json(self):
        """Plan should handle JSON wrapped in markdown code blocks."""
        mock_response = '''```json
{
    "query_type": "document_lookup",
    "retrieval_strategy": "full_document",
    "filename_hints": ["biography"],
    "answer_style": "return_content",
    "complexity": "simple"
}
```'''

        with patch('rag._call_fast_llm', return_value=mock_response):
            plan = plan_query("show me my biography")

        assert plan['query_type'] == 'document_lookup'
        assert plan['used_llm'] is True

    def test_plan_query_falls_back_on_invalid_json(self):
        """Plan should fall back to heuristics on invalid JSON."""
        mock_response = 'This is not valid JSON'

        with patch('rag._call_fast_llm', return_value=mock_response):
            plan = plan_query("show me my biography")

        # Should fall back to Phase 1 heuristics
        assert plan['query_type'] == 'document_lookup'
        assert plan['used_llm'] is False


class TestExpandQuery:
    """Tests for expand_query function."""

    def test_expand_query_fallback_produces_variations(self):
        """Fallback expansion should produce at least 2 query variations."""
        with patch('rag._call_fast_llm', return_value=None):
            result = expand_query("what's in my biography")

        assert len(result['queries']) >= 2
        assert result['used_llm'] is False
        # Should include processed query
        assert any('biography' in q for q in result['queries'])

    def test_expand_query_includes_document_hint(self):
        """Fallback expansion should include document hint as a query."""
        with patch('rag._call_fast_llm', return_value=None):
            result = expand_query("show me my resume")

        # "resume" should be in the queries
        assert any('resume' in q.lower() for q in result['queries'])

    def test_expand_query_parses_llm_response(self):
        """Expansion should parse valid LLM JSON response."""
        mock_response = '''{
            "queries": ["biography document", "personal bio", "about me", "cv resume"],
            "key_entities": ["biography", "personal"],
            "filename_patterns": ["bio", "about"]
        }'''

        with patch('rag._call_fast_llm', return_value=mock_response):
            result = expand_query("show me my biography")

        assert len(result['queries']) == 4
        assert 'biography document' in result['queries']
        assert result['used_llm'] is True

    def test_expand_query_extracts_key_entities(self):
        """Expansion should extract key entities."""
        with patch('rag._call_fast_llm', return_value=None):
            result = expand_query("how do I configure authentication")

        # Should have extracted meaningful terms
        assert 'key_entities' in result
        assert len(result['key_entities']) > 0


class TestGenerateHydeDocument:
    """Tests for generate_hyde_document function."""

    def test_hyde_returns_none_when_llm_fails(self):
        """HyDE should return None when LLM is unavailable."""
        with patch('rag._call_fast_llm', return_value=None):
            result = generate_hyde_document("What is machine learning?")

        assert result is None

    def test_hyde_returns_document_from_llm(self):
        """HyDE should return the LLM-generated document."""
        mock_response = "Machine learning is a subset of artificial intelligence that enables systems to learn from data."

        with patch('rag._call_fast_llm', return_value=mock_response):
            result = generate_hyde_document("What is machine learning?")

        assert result == mock_response

    def test_hyde_prompt_has_query_placeholder(self):
        """HyDE prompt should have {query} placeholder."""
        assert '{query}' in HYDE_PROMPT


class TestEnhancedPreprocessQuery:
    """Tests for enhanced_preprocess_query function."""

    def test_enhanced_includes_phase1_fields(self):
        """Enhanced preprocessing should include all Phase 1 fields."""
        with patch('rag._call_fast_llm', return_value=None):
            result = enhanced_preprocess_query("show me my biography")

        # Phase 1 fields
        assert 'original_query' in result
        assert 'processed_query' in result
        assert 'is_document_request' in result
        assert 'document_hint' in result
        assert 'search_terms' in result

    def test_enhanced_includes_phase2_fields(self):
        """Enhanced preprocessing should include Phase 2 fields."""
        with patch('rag._call_fast_llm', return_value=None):
            result = enhanced_preprocess_query("what is machine learning")

        # Phase 2 fields
        assert 'plan' in result
        assert 'expanded_queries' in result
        assert 'hyde_document' in result
        assert 'phase2_enabled' in result

    def test_enhanced_skips_hyde_for_document_lookup(self):
        """HyDE should be skipped for document lookup queries."""
        with patch('rag._call_fast_llm', return_value=None):
            result = enhanced_preprocess_query("show me my biography")

        # Document request detected
        assert result['is_document_request'] is True
        # HyDE should be None for document lookups
        assert result['hyde_document'] is None

    def test_enhanced_can_disable_features(self):
        """Should be able to disable Phase 2 features."""
        with patch('rag._call_fast_llm', return_value=None):
            result = enhanced_preprocess_query(
                "what is machine learning",
                use_planner=False,
                use_multi_query=False,
                use_hyde=False
            )

        # Features should be disabled
        assert result['plan'] is None
        assert result['expanded_queries'] == [result['processed_query']]
        assert result['hyde_document'] is None
        assert result['phase2_enabled'] is False

    def test_enhanced_uses_planner_hints(self):
        """Enhanced preprocessing should use planner's filename hints."""
        mock_plan_response = '''{
            "query_type": "document_lookup",
            "retrieval_strategy": "full_document",
            "filename_hints": ["bio", "biography"],
            "answer_style": "return_content",
            "complexity": "simple"
        }'''

        with patch('rag._call_fast_llm', return_value=mock_plan_response):
            result = enhanced_preprocess_query("tell me about myself")

        # Should have picked up planner's hints
        assert 'bio' in result['search_terms'] or 'biography' in result['search_terms']
        assert result['is_document_request'] is True

    def test_enhanced_marks_phase2_enabled_when_llm_used(self):
        """phase2_enabled should be True when LLM features work."""
        mock_response = '''{
            "query_type": "factual_qa",
            "retrieval_strategy": "semantic_chunks",
            "filename_hints": [],
            "answer_style": "synthesize",
            "complexity": "simple"
        }'''

        with patch('rag._call_fast_llm', return_value=mock_response):
            result = enhanced_preprocess_query("what is the capital of France")

        assert result['phase2_enabled'] is True


class TestMultiQueryPrompt:
    """Tests for multi-query expansion prompt."""

    def test_multi_query_prompt_has_placeholders(self):
        """Multi-query prompt should have required placeholders."""
        assert '{query}' in MULTI_QUERY_PROMPT
        assert '{query_type}' in MULTI_QUERY_PROMPT

    def test_multi_query_prompt_requests_variations(self):
        """Multi-query prompt should request 5-7 variations."""
        assert '5-7' in MULTI_QUERY_PROMPT or ('5' in MULTI_QUERY_PROMPT and '7' in MULTI_QUERY_PROMPT)


class TestProviderAgnosticIntegration:
    """Integration tests for provider-agnostic model selection."""

    def test_different_providers_use_their_fast_models(self):
        """Each provider should use its own fast model, not hardcoded Haiku."""
        # Test that we're not hardcoding "haiku" but using categories
        fast_anthropic = _get_fast_model('anthropic/claude-sonnet-5')
        fast_openai = _get_fast_model('openai/gpt-5.6-terra')
        fast_google = _get_fast_model('gemini/gemini-3.1-pro-preview')

        # Each should be different (different providers)
        assert fast_anthropic != fast_openai
        assert fast_openai != fast_google
        assert fast_anthropic != fast_google

        # Each should be from the correct provider
        assert 'anthropic' in fast_anthropic
        assert 'openai' in fast_openai
        assert 'gemini' in fast_google

    def test_fast_model_uses_small_category(self):
        """Fast model should use 'small' category models."""
        from synthesis_engine.config import get_model_by_category

        # Verify we're using category-based selection
        anthropic_small = get_model_by_category('anthropic', 'small')
        openai_small = get_model_by_category('openai', 'small')
        google_small = get_model_by_category('google', 'small')

        # These should match _get_fast_model results
        assert anthropic_small == _get_fast_model('anthropic/claude-opus-4-8')
        assert openai_small == _get_fast_model('openai/gpt-5.6-terra')
        assert google_small == _get_fast_model('gemini/gemini-3.1-pro-preview')


class TestFallbackBehavior:
    """Tests for graceful fallback when LLM is unavailable."""

    def test_all_functions_handle_llm_failure(self):
        """All Phase 2 functions should handle LLM failures gracefully."""
        with patch('rag._call_fast_llm', return_value=None):
            # None of these should raise exceptions
            plan = plan_query("test query")
            assert plan is not None
            assert 'query_type' in plan

            expansion = expand_query("test query")
            assert expansion is not None
            assert 'queries' in expansion

            hyde = generate_hyde_document("test query")
            # HyDE returns None when LLM fails, which is valid
            assert hyde is None

            enhanced = enhanced_preprocess_query("test query")
            assert enhanced is not None
            assert 'processed_query' in enhanced

    def test_fallback_preserves_phase1_accuracy(self):
        """Fallback should produce results as good as Phase 1."""
        with patch('rag._call_fast_llm', return_value=None):
            # Document request detection should still work
            result = enhanced_preprocess_query("show me my biography")
            assert result['is_document_request'] is True
            assert result['document_hint'] == 'biography'

            # Contraction expansion should still work
            result = enhanced_preprocess_query("what's in my resume")
            assert 'what is' in result['processed_query']
