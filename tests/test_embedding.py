"""Tests for the embedding model contract.

The embedding model, the dimension the pgvector schema stores and the query
instruction have to agree. These tests pin that agreement:

  * engines.yaml's ``embedding:`` block is read and validated by
    ``get_embedding_config()``, which refuses a malformed block instead of
    falling back to some other model.
  * The last schema migration sets the vector columns to the dimension
    engines.yaml declares.
  * Queries are encoded with the configured retrieval instruction and
    documents without one, in batches.
  * Loading a model that contradicts engines.yaml is an error, not a silent
    degradation.
  * The pgvector backend refuses vectors whose dimension the schema does not
    store.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_SRC = os.path.join(os.path.dirname(__file__), '..', 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from synthesis_engine import config as engine_config  # noqa: E402
from synthesis_engine.config import EmbeddingConfig, get_embedding_config  # noqa: E402
from synthesis_engine.exceptions import ConfigurationError  # noqa: E402
from synthesis_engine.vectorstore.pgvector_backend import PgvectorBackend  # noqa: E402

_MIGRATIONS = Path(_SRC) / 'synthesis_engine' / 'vectorstore' / 'migrations'

_VALID_BLOCK = {
    'provider': 'sentence-transformers',
    'model': 'example/model',
    'revision': 'abc123',
    'dimensions': 8,
    'dtype': 'float32',
    'max_seq_length': 512,
    'batch_size': 4,
    'query_prompt_name': 'web_search_query',
}


def _config_with(block):
    return patch.object(engine_config, 'load_engines_config', return_value={'embedding': block})


class TestEnginesYamlEmbeddingBlock:
    def test_declares_a_pinned_instruction_tuned_model(self):
        config = get_embedding_config()
        assert config.provider == 'sentence-transformers'
        assert config.model == 'microsoft/harrier-oss-v1-0.6b'
        # A full commit hash, so the weights cannot change underneath the index.
        assert re.fullmatch(r'[0-9a-f]{40}', config.revision)
        assert config.dimensions == 1024
        assert config.query_prompt_name == 'web_search_query'

    def test_version_names_model_and_revision(self):
        config = get_embedding_config()
        assert config.version == f'{config.model}@{config.revision}'


class TestEmbeddingConfigValidation:
    def test_valid_block_parses(self):
        with _config_with(dict(_VALID_BLOCK)):
            config = get_embedding_config()
        assert config == EmbeddingConfig(**_VALID_BLOCK)

    def test_missing_block_is_an_error(self):
        with patch.object(engine_config, 'load_engines_config', return_value={}):
            with pytest.raises(ConfigurationError, match="no 'embedding' block"):
                get_embedding_config()

    @pytest.mark.parametrize('key', ['provider', 'model', 'revision', 'dimensions',
                                     'dtype', 'max_seq_length', 'batch_size'])
    def test_each_required_key_is_enforced(self, key):
        block = dict(_VALID_BLOCK)
        del block[key]
        with _config_with(block):
            with pytest.raises(ConfigurationError, match=key):
                get_embedding_config()

    def test_query_prompt_is_optional(self):
        block = dict(_VALID_BLOCK)
        del block['query_prompt_name']
        with _config_with(block):
            assert get_embedding_config().query_prompt_name is None

    def test_rejects_other_providers(self):
        with _config_with(dict(_VALID_BLOCK, provider='openai')):
            with pytest.raises(ConfigurationError, match='Unsupported embedding provider'):
                get_embedding_config()

    def test_rejects_unknown_dtype(self):
        with _config_with(dict(_VALID_BLOCK, dtype='int8')):
            with pytest.raises(ConfigurationError, match='dtype'):
                get_embedding_config()

    @pytest.mark.parametrize('key', ['dimensions', 'max_seq_length', 'batch_size'])
    @pytest.mark.parametrize('value', [0, -1, '1024', True])
    def test_sizes_must_be_positive_integers(self, key, value):
        with _config_with(dict(_VALID_BLOCK, **{key: value})):
            with pytest.raises(ConfigurationError, match=key):
                get_embedding_config()


class TestSchemaMatchesEmbeddingModel:
    """The newest migration that sets a vector column's dimension must set
    the dimension engines.yaml declares, for both vector columns."""

    @pytest.mark.parametrize('table', ['chunks', 'entities'])
    def test_latest_column_dimension_matches_config(self, table):
        declared = None
        for path in sorted(_MIGRATIONS.glob('*.sql')):
            sql = path.read_text()
            for match in re.finditer(
                rf'ALTER TABLE {table} ALTER COLUMN embedding TYPE VECTOR\((\d+)\)', sql,
            ):
                declared = int(match.group(1))
        assert declared == get_embedding_config().dimensions

    @pytest.mark.parametrize('table', ['chunks', 'entities'])
    def test_conversion_is_guarded_by_the_same_dimension(self, table):
        # The runner executes every migration on every start; the guard is
        # what keeps 0004 from clearing the index each time.
        sql = (_MIGRATIONS / '0004_embedding_dimensions.sql').read_text()
        dimension = get_embedding_config().dimensions
        guard = re.search(
            rf"attrelid = '{table}'::regclass AND attname = 'embedding'\) <> (\d+)", sql,
        )
        assert guard is not None
        assert int(guard.group(1)) == dimension


class TestEncoding:
    def test_queries_carry_the_configured_instruction(self):
        import rag

        model = MagicMock()
        model.encode_query.return_value.tolist.return_value = [0.5, 0.5]
        assert rag._encode_query(model, 'where is the runbook') == [0.5, 0.5]
        model.encode_query.assert_called_once_with(
            'where is the runbook', prompt_name='web_search_query',
        )

    def test_documents_are_batched_without_an_instruction(self):
        import rag

        model = MagicMock()
        vectors = [MagicMock(), MagicMock()]
        for i, vector in enumerate(vectors):
            vector.tolist.return_value = [float(i)]
        model.encode_document.return_value = vectors

        assert rag._encode_documents(model, ['a', 'b']) == [[0.0], [1.0]]
        model.encode_document.assert_called_once_with(
            ['a', 'b'], batch_size=get_embedding_config().batch_size, show_progress_bar=False,
        )

    def test_no_documents_means_no_model_call(self):
        import rag

        model = MagicMock()
        assert rag._encode_documents(model, []) == []
        model.encode_document.assert_not_called()

    def test_model_version_records_revision_and_library(self):
        import rag

        sentence_transformers = pytest.importorskip('sentence_transformers')
        version = rag._get_embedding_model_version()
        config = get_embedding_config()
        assert version == (
            f'{config.model}@{config.revision}'
            f'+sentence-transformers-{sentence_transformers.__version__}'
        )


class TestModelLoading:
    """``_get_embedding_model`` refuses a model that contradicts engines.yaml."""

    @pytest.fixture(autouse=True)
    def _fresh_model_cache(self):
        import rag

        rag._embedding_model = None
        yield
        rag._embedding_model = None

    def _load_with(self, dimension, prompts):
        import rag

        pytest.importorskip('sentence_transformers')
        fake = MagicMock()
        fake.get_embedding_dimension.return_value = dimension
        fake.prompts = prompts
        with patch('sentence_transformers.SentenceTransformer', return_value=fake) as loader:
            model = rag._get_embedding_model()
        return model, loader, fake

    def test_loads_pinned_revision_dtype_and_sequence_cap(self):
        import torch

        config = get_embedding_config()
        model, loader, fake = self._load_with(
            config.dimensions, {config.query_prompt_name: 'Instruct: ...'},
        )
        assert model is fake
        loader.assert_called_once_with(
            config.model,
            revision=config.revision,
            model_kwargs={'dtype': getattr(torch, config.dtype)},
        )
        assert fake.max_seq_length == config.max_seq_length

    def test_dimension_mismatch_is_an_error(self):
        config = get_embedding_config()
        with pytest.raises(ConfigurationError, match='dimension'):
            self._load_with(config.dimensions + 1, {config.query_prompt_name: ''})

    def test_missing_query_prompt_is_an_error(self):
        config = get_embedding_config()
        with pytest.raises(ConfigurationError, match=config.query_prompt_name):
            self._load_with(config.dimensions, {'query': '', 'document': ''})


class TestBackendDimensionGuard:
    def _backend(self, dimension):
        backend = PgvectorBackend(dsn='postgresql://fake/fake')
        backend._schema_dimension = dimension
        backend._migrated = True
        return backend

    def test_init_collection_refuses_a_different_dimension(self):
        backend = self._backend(1024)
        backend._connection = MagicMock()  # type: ignore[method-assign]
        assert backend.init_collection('ws', vector_size=384) is False
        backend._connection.assert_not_called()

    def test_upsert_refuses_wrong_dimension_vectors(self):
        from synthesis_engine.vectorstore import Point

        backend = self._backend(4)
        point = Point(chunk_uid='a', vector=[0.0] * 3, text='t', embedding_model='m')
        with pytest.raises(ValueError, match='3 dimensions; the pgvector schema stores 4'):
            backend.upsert_points('ws', [point])

    def test_search_refuses_wrong_dimension_query(self):
        backend = self._backend(4)
        with pytest.raises(ValueError, match='5 dimensions'):
            backend.search('ws', [0.0] * 5)
