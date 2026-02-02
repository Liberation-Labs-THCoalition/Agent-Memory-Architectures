"""Embedding generation providers for Kintsugi CMA.

Supports local sentence-transformers (all-mpnet-base-v2, 768D) and
OpenAI API (text-embedding-3-small, 1536D).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class EmbeddingProvider(ABC):
    """Abstract interface for embedding generation."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the output embedding vectors."""

    @abstractmethod
    async def embed(self, text: str) -> NDArray[np.float32]:
        """Embed a single text string and return a 1-D float32 array."""

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[NDArray[np.float32]]:
        """Embed a batch of texts. Returns one array per input text."""


# ---------------------------------------------------------------------------
# Local provider — sentence-transformers all-mpnet-base-v2
# ---------------------------------------------------------------------------

_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
_LOCAL_DIM = 768


class LocalEmbeddingProvider(EmbeddingProvider):
    """Generate embeddings locally via sentence-transformers.

    The model is lazy-loaded on first call to avoid import-time overhead.
    If sentence-transformers or the model weights are unavailable, a clear
    error is raised with installation instructions.
    """

    def __init__(self, model_name: str = _MODEL_NAME, batch_size: int = 64) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model: Any = None  # lazy

    @property
    def dimension(self) -> int:
        return _LOCAL_DIM

    # -- lazy loading -------------------------------------------------------

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            msg = (
                "sentence-transformers is required for LocalEmbeddingProvider. "
                "Install it with: pip install sentence-transformers"
            )
            logger.error(msg)
            raise ImportError(msg) from exc
        try:
            logger.info("Loading embedding model %s ...", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        except Exception as exc:
            msg = (
                f"Failed to load model '{self._model_name}'. Ensure model weights are "
                "downloaded. Run: python -c \"from sentence_transformers import "
                f"SentenceTransformer; SentenceTransformer('{self._model_name}')\""
            )
            logger.warning(msg)
            raise RuntimeError(msg) from exc
        return self._model

    # -- public API ---------------------------------------------------------

    async def embed(self, text: str) -> NDArray[np.float32]:
        model = self._load_model()
        vec: NDArray[np.float32] = model.encode(
            text, convert_to_numpy=True, normalize_embeddings=True
        )
        return vec.astype(np.float32)

    async def embed_batch(self, texts: list[str]) -> list[NDArray[np.float32]]:
        model = self._load_model()
        all_vecs: list[NDArray[np.float32]] = []
        for i in range(0, len(texts), self._batch_size):
            chunk = texts[i : i + self._batch_size]
            vecs: NDArray[np.float32] = model.encode(
                chunk, convert_to_numpy=True, normalize_embeddings=True, batch_size=self._batch_size
            )
            all_vecs.extend(v.astype(np.float32) for v in vecs)
        return all_vecs


# ---------------------------------------------------------------------------
# API provider — OpenAI text-embedding-3-small
# ---------------------------------------------------------------------------

_API_MODEL = "text-embedding-3-small"
_API_DIM = 1536
_API_URL = "https://api.openai.com/v1/embeddings"
_API_MAX_BATCH = 2048


class APIEmbeddingProvider(EmbeddingProvider):
    """Generate embeddings via the OpenAI embeddings API.

    Requires *httpx* (async) and a valid ``api_key``.
    """

    def __init__(
        self,
        api_key: str,
        model: str = _API_MODEL,
        base_url: str = _API_URL,
        max_batch: int = _API_MAX_BATCH,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required for APIEmbeddingProvider")
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._max_batch = max_batch

    @property
    def dimension(self) -> int:
        return _API_DIM

    async def _request(self, texts: list[str]) -> list[NDArray[np.float32]]:
        import httpx

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {"input": texts, "model": self._model}
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(self._base_url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()["data"]
        # API returns embeddings sorted by index
        sorted_data = sorted(data, key=lambda d: d["index"])
        return [np.array(d["embedding"], dtype=np.float32) for d in sorted_data]

    async def embed(self, text: str) -> NDArray[np.float32]:
        results = await self._request([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[NDArray[np.float32]]:
        all_vecs: list[NDArray[np.float32]] = []
        for i in range(0, len(texts), self._max_batch):
            chunk = texts[i : i + self._max_batch]
            all_vecs.extend(await self._request(chunk))
        return all_vecs


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_embedding_provider(mode: str = "local", **kwargs: Any) -> EmbeddingProvider:
    """Create an embedding provider.

    Args:
        mode: ``"local"`` for sentence-transformers, ``"api"`` for OpenAI.
        **kwargs: Forwarded to the provider constructor.

    Returns:
        An :class:`EmbeddingProvider` instance.
    """
    if mode == "local":
        return LocalEmbeddingProvider(**kwargs)
    if mode == "api":
        return APIEmbeddingProvider(**kwargs)
    raise ValueError(f"Unknown embedding mode: {mode!r}. Choose 'local' or 'api'.")
