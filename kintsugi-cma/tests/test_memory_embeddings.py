"""Tests for kintsugi.memory.embeddings module."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio

from kintsugi.memory.embeddings import (
    APIEmbeddingProvider,
    EmbeddingProvider,
    LocalEmbeddingProvider,
    get_embedding_provider,
    _LOCAL_DIM,
    _API_DIM,
)


# ---------------------------------------------------------------------------
# LocalEmbeddingProvider
# ---------------------------------------------------------------------------


class TestLocalEmbeddingProvider:
    def test_dimension(self):
        p = LocalEmbeddingProvider()
        assert p.dimension == _LOCAL_DIM == 768

    def test_custom_model_name(self):
        p = LocalEmbeddingProvider(model_name="custom/model", batch_size=32)
        assert p._model_name == "custom/model"
        assert p._batch_size == 32

    def test_lazy_load_not_called_on_init(self):
        p = LocalEmbeddingProvider()
        assert p._model is None

    @patch.dict(sys.modules, {"sentence_transformers": None})
    def test_import_error_when_missing(self):
        p = LocalEmbeddingProvider()
        with pytest.raises(ImportError, match="sentence-transformers is required"):
            p._load_model()

    def test_runtime_error_when_model_fails(self):
        mock_st = MagicMock()
        mock_st.SentenceTransformer.side_effect = RuntimeError("download failed")
        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            p = LocalEmbeddingProvider()
            with pytest.raises(RuntimeError, match="Failed to load model"):
                p._load_model()

    def test_load_model_caches(self):
        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_st.SentenceTransformer.return_value = mock_model
        with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
            p = LocalEmbeddingProvider()
            m1 = p._load_model()
            m2 = p._load_model()
            assert m1 is m2
            mock_st.SentenceTransformer.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed(self):
        fake_vec = np.random.randn(768).astype(np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_vec

        p = LocalEmbeddingProvider()
        p._model = mock_model

        result = await p.embed("hello")
        assert result.shape == (768,)
        assert result.dtype == np.float32
        mock_model.encode.assert_called_once_with(
            "hello", convert_to_numpy=True, normalize_embeddings=True
        )

    @pytest.mark.asyncio
    async def test_embed_batch(self):
        fake_vecs_1 = np.random.randn(2, 768).astype(np.float32)
        fake_vecs_2 = np.random.randn(1, 768).astype(np.float32)
        mock_model = MagicMock()
        mock_model.encode.side_effect = [fake_vecs_1, fake_vecs_2]

        p = LocalEmbeddingProvider(batch_size=2)
        p._model = mock_model

        texts = ["a", "b", "c"]
        result = await p.embed_batch(texts)
        # With batch_size=2, two calls: [a,b] and [c]
        assert mock_model.encode.call_count == 2
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_embed_batch_empty(self):
        mock_model = MagicMock()
        p = LocalEmbeddingProvider()
        p._model = mock_model
        result = await p.embed_batch([])
        assert result == []


# ---------------------------------------------------------------------------
# APIEmbeddingProvider
# ---------------------------------------------------------------------------


class TestAPIEmbeddingProvider:
    def test_dimension(self):
        p = APIEmbeddingProvider(api_key="sk-test")
        assert p.dimension == _API_DIM == 1536

    def test_missing_api_key(self):
        with pytest.raises(ValueError, match="api_key is required"):
            APIEmbeddingProvider(api_key="")

    @pytest.mark.asyncio
    async def test_embed(self):
        fake_embedding = [0.1] * 1536
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "data": [{"index": 0, "embedding": fake_embedding}]
        }

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("httpx.AsyncClient", return_value=mock_client):
            p = APIEmbeddingProvider(api_key="sk-test")
            result = await p.embed("hello")
            assert result.shape == (1536,)
            assert result.dtype == np.float32

    @pytest.mark.asyncio
    async def test_embed_batch(self):
        fake_data = [
            {"index": 1, "embedding": [0.2] * 1536},
            {"index": 0, "embedding": [0.1] * 1536},
        ]
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": fake_data}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("httpx.AsyncClient", return_value=mock_client):
            p = APIEmbeddingProvider(api_key="sk-test")
            result = await p.embed_batch(["a", "b"])
            assert len(result) == 2
            # Should be sorted by index: index 0 first
            np.testing.assert_array_almost_equal(result[0][0], 0.1)
            np.testing.assert_array_almost_equal(result[1][0], 0.2)

    @pytest.mark.asyncio
    async def test_embed_batch_chunking(self):
        """Batches larger than max_batch are chunked."""
        fake_data = [{"index": 0, "embedding": [0.1] * 1536}]
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": fake_data}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_resp)

        with patch("httpx.AsyncClient", return_value=mock_client):
            p = APIEmbeddingProvider(api_key="sk-test", max_batch=1)
            result = await p.embed_batch(["a", "b"])
            assert mock_client.post.call_count == 2
            assert len(result) == 2


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_local(self):
        p = get_embedding_provider("local")
        assert isinstance(p, LocalEmbeddingProvider)

    def test_api(self):
        p = get_embedding_provider("api", api_key="sk-test")
        assert isinstance(p, APIEmbeddingProvider)

    def test_unknown_mode(self):
        with pytest.raises(ValueError, match="Unknown embedding mode"):
            get_embedding_provider("gpu")

    def test_abc_not_instantiable(self):
        with pytest.raises(TypeError):
            EmbeddingProvider()
