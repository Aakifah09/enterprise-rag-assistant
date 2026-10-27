"""Dense (embedding) retrieval and keyword-only retrieval over a :class:`SearchIndex`."""

from __future__ import annotations

import numpy as np

from src.models import RetrievedChunk
from src.retrieval.index import SearchIndex


def to_retrieved(
    index: SearchIndex, row: int, rank: int, score: float, dense: float, keyword: float, strategy: str
) -> RetrievedChunk:
    c = index.chunks[row]
    return RetrievedChunk(
        chunk_id=c.chunk_id,
        doc_id=c.doc_id,
        filename=index.filenames.get(c.doc_id, c.doc_id),
        text=c.text,
        rank=rank,
        score=float(score),
        dense_score=float(dense),
        keyword_score=float(keyword),
        page_number=c.page_number,
        section=c.section,
        strategy=strategy,
    )


def dense_search(index: SearchIndex, query_vec: np.ndarray, top_k: int) -> list[RetrievedChunk]:
    """Rank chunks by cosine similarity to the query embedding (FAISS inner product)."""
    hits = index.store.search(query_vec, top_k)
    return [
        to_retrieved(index, index.row_of[cid], rank, score, score, 0.0, "dense")
        for rank, (cid, score) in enumerate(hits, start=1)
    ]


def keyword_search(index: SearchIndex, query: str, query_vec: np.ndarray, top_k: int) -> list[RetrievedChunk]:
    """Rank chunks by BM25; the dense cosine is still reported for the evidence gate."""
    hits = index.bm25.top(query, top_k)
    sims = index.store.similarities(query_vec)
    return [
        to_retrieved(index, row, rank, score, float(sims[row]) if len(sims) else 0.0, score, "keyword")
        for rank, (row, score) in enumerate(hits, start=1)
    ]
