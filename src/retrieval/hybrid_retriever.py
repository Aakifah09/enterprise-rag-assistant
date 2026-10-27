"""Hybrid retrieval: fuse dense and BM25 rankings.

Two fusion methods are provided so they can be compared experimentally:

* ``hybrid`` (weighted score fusion). Take the union of the top
  ``candidate_pool`` chunks from each retriever. For every candidate compute
  both its cosine similarity and its BM25 score, min-max normalise each score
  over the candidate set to [0, 1], and combine
  ``fused = alpha * dense_norm + (1 - alpha) * bm25_norm``.
  Min-max is needed because cosine lives in [-1, 1] while BM25 is unbounded.
  The fused value is only meaningful *within* one query's candidate set.

* ``rrf`` (Reciprocal Rank Fusion, Cormack et al. 2009).
  ``score = sum_r 1 / (k + rank_r)`` with ``k = 60``, over the rankers in
  which the chunk appears. RRF ignores raw scores entirely, which makes it
  robust to scale differences but blind to how *much* better a hit is.
"""

from __future__ import annotations

import numpy as np

from src.models import RetrievedChunk
from src.retrieval.dense_retriever import to_retrieved
from src.retrieval.index import SearchIndex

RRF_K = 60


def minmax(values: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]; a constant vector maps to all-ones if positive, else zeros."""
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-12:
        return np.ones_like(values) if hi > 0 else np.zeros_like(values)
    return (values - lo) / (hi - lo)


def _candidates(index: SearchIndex, query: str, query_vec: np.ndarray, pool: int) -> tuple[list[int], list[int], np.ndarray, list[float]]:
    dense_rows = [index.row_of[cid] for cid, _ in index.store.search(query_vec, pool)]
    bm25_all = index.bm25.scores(query)
    kw_rows = sorted((r for r, s in enumerate(bm25_all) if s > 0), key=lambda r: (-bm25_all[r], r))[:pool]
    sims = index.store.similarities(query_vec)
    return dense_rows, kw_rows, sims, bm25_all


def hybrid_search(
    index: SearchIndex, query: str, query_vec: np.ndarray, top_k: int, alpha: float = 0.6, pool: int = 30
) -> list[RetrievedChunk]:
    if index.is_empty:
        return []
    dense_rows, kw_rows, sims, bm25_all = _candidates(index, query, query_vec, pool)
    rows = list(dict.fromkeys(dense_rows + kw_rows))
    d = np.array([sims[r] for r in rows])
    k = np.array([bm25_all[r] for r in rows])
    fused = alpha * minmax(d) + (1 - alpha) * minmax(k)
    order = sorted(range(len(rows)), key=lambda i: (-fused[i], -d[i], rows[i]))[:top_k]
    return [
        to_retrieved(index, rows[i], rank, fused[i], d[i], k[i], "hybrid") for rank, i in enumerate(order, start=1)
    ]


def rrf_search(index: SearchIndex, query: str, query_vec: np.ndarray, top_k: int, pool: int = 30) -> list[RetrievedChunk]:
    if index.is_empty:
        return []
    dense_rows, kw_rows, sims, bm25_all = _candidates(index, query, query_vec, pool)
    scores: dict[int, float] = {}
    for ranking in (dense_rows, kw_rows):
        for rank, row in enumerate(ranking, start=1):
            scores[row] = scores.get(row, 0.0) + 1.0 / (RRF_K + rank)
    order = sorted(scores, key=lambda r: (-scores[r], -sims[r], r))[:top_k]
    return [
        to_retrieved(index, row, rank, scores[row], sims[row], bm25_all[row], "rrf")
        for rank, row in enumerate(order, start=1)
    ]
