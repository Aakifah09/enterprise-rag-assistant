"""An in-memory search index for one (chunking config, embedding model) pair."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.models import Chunk
from src.retrieval.keyword_retriever import BM25Index
from src.retrieval.vector_store import VectorStore


def indexable_text(chunk: Chunk) -> str:
    """Text that is embedded / keyword-indexed for a chunk.

    The section heading is prepended so a chunk like "Requests are approved
    within 2 days" is still findable by the heading words ("Laptop requests").
    The stored chunk text, which is shown and cited, is unchanged.
    """
    return f"{chunk.section}\n{chunk.text}" if chunk.section else chunk.text


@dataclass
class SearchIndex:
    chunking_key: str
    model_name: str
    corpus_version: int
    chunks: list[Chunk]
    filenames: dict[str, str]  # doc_id -> filename
    store: VectorStore
    bm25: BM25Index
    row_of: dict[str, int] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        chunking_key: str,
        model_name: str,
        corpus_version: int,
        chunks: list[Chunk],
        vectors: np.ndarray,
        filenames: dict[str, str],
        dimension: int,
    ) -> "SearchIndex":
        store = VectorStore(dimension)
        if chunks:
            store.add([c.chunk_id for c in chunks], vectors)
        bm25 = BM25Index([indexable_text(c) for c in chunks])
        return cls(
            chunking_key=chunking_key,
            model_name=model_name,
            corpus_version=corpus_version,
            chunks=chunks,
            filenames=filenames,
            store=store,
            bm25=bm25,
            row_of={c.chunk_id: i for i, c in enumerate(chunks)},
        )

    @property
    def is_empty(self) -> bool:
        return not self.chunks
