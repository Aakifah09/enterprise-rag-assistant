"""FAISS vector index over L2-normalised embeddings (inner product == cosine)."""

from __future__ import annotations

import faiss
import numpy as np


class VectorStore:
    """Exact (flat) inner-product index that maps FAISS row ids to chunk ids.

    A flat index is exact and fast enough for tens of thousands of chunks on a
    CPU; an IVF/HNSW index would only pay off at much larger scale.
    """

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunk_ids: list[str] = []
        self._vectors = np.zeros((0, dimension), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.chunk_ids)

    @property
    def is_empty(self) -> bool:
        return len(self.chunk_ids) == 0

    def add(self, chunk_ids: list[str], vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(f"expected vectors of shape (n, {self.dimension}), got {vectors.shape}")
        if len(chunk_ids) != vectors.shape[0]:
            raise ValueError("chunk_ids and vectors must have the same length")
        if len(set(chunk_ids)) != len(chunk_ids) or set(chunk_ids) & set(self.chunk_ids):
            raise ValueError("duplicate chunk ids")
        self.index.add(vectors)
        self.chunk_ids.extend(chunk_ids)
        self._vectors = np.vstack([self._vectors, vectors])

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Return up to ``k`` (chunk_id, cosine) pairs, best first."""
        if self.is_empty or k <= 0:
            return []
        q = np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != self.dimension:
            raise ValueError(f"query dimension {q.shape[1]} does not match index dimension {self.dimension}")
        scores, ids = self.index.search(q, min(k, len(self.chunk_ids)))
        return [(self.chunk_ids[i], float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]

    def similarities(self, query: np.ndarray) -> np.ndarray:
        """Cosine similarity of the query to every stored vector (row order)."""
        if self.is_empty:
            return np.zeros(0, dtype=np.float32)
        return self._vectors @ np.asarray(query, dtype=np.float32).reshape(-1)
