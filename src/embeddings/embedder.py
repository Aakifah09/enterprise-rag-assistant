"""Embedding models behind a small interface so retrieval code is model-agnostic."""

from __future__ import annotations

import logging
import threading
import time
from typing import Protocol, runtime_checkable

import numpy as np

from src.config import query_prefix_for

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """The embedding model could not be loaded or failed to encode text."""


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into L2-normalised float32 vectors."""

    model_name: str

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Row-normalise so inner product equals cosine similarity."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover
        return "cpu"


class SentenceTransformerEmbedder:
    """Sentence Transformers model with optional query instruction prefix."""

    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 32) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.query_prefix = query_prefix_for(model_name)
        self.load_seconds = 0.0
        try:
            from sentence_transformers import SentenceTransformer

            start = time.perf_counter()
            try:
                # Use the local Hugging Face cache first: avoids slow online
                # update checks and makes the app work offline after one download.
                self._model = SentenceTransformer(model_name, device=self.device, local_files_only=True)
            except Exception:
                self._model = SentenceTransformer(model_name, device=self.device)
            self.load_seconds = time.perf_counter() - start
        except Exception as exc:
            raise EmbeddingError(
                f"Could not load embedding model '{model_name}'. If this is the first run, an internet "
                "connection is needed once to download it from Hugging Face."
            ) from exc
        getter = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self._dim = int(getter())
        self._lock = threading.Lock()  # Streamlit may call from several script threads
        logger.info("Loaded embedding model %s on %s (dim=%d)", model_name, self.device, self._dim)

    @property
    def dimension(self) -> int:
        return self._dim

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        try:
            with self._lock:
                vecs = self._model.encode(
                    texts,
                    batch_size=self.batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed: {exc}") from exc
        return l2_normalize(vecs)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.query_prefix + text])[0]


_CACHE: dict[tuple[str, str], SentenceTransformerEmbedder] = {}
_CACHE_LOCK = threading.Lock()


def get_embedder(model_name: str, device: str = "auto") -> SentenceTransformerEmbedder:
    """Process-wide cache so each model is loaded once."""
    key = (model_name, resolve_device(device))
    with _CACHE_LOCK:
        if key not in _CACHE:
            _CACHE[key] = SentenceTransformerEmbedder(model_name, device=device)
        return _CACHE[key]
