"""Plain data types shared across the ingestion, retrieval and generation layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TextUnit:
    """A citable unit of extracted text: one PDF page or one Markdown/TXT section.

    ``page_number`` is 1-based for PDFs and ``None`` for formats without pages.
    """

    text: str
    page_number: int | None = None
    section: str | None = None


@dataclass
class ExtractedDocument:
    """Result of loading a single file, before chunking."""

    filename: str
    file_type: str
    units: list[TextUnit]
    page_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return sum(len(u.text) for u in self.units)

    @property
    def has_text(self) -> bool:
        return any(u.text.strip() for u in self.units)


@dataclass
class Chunk:
    """A retrievable piece of a document with full provenance."""

    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    page_number: int | None = None
    section: str | None = None
    char_start: int = 0
    char_end: int = 0


@dataclass
class RetrievedChunk:
    """A chunk returned by a retriever, with its scores and rank.

    ``score`` is the strategy's ranking score (cosine, BM25, fused or RRF) and is
    *not* a probability. ``dense_score`` is always the raw query/chunk cosine
    similarity so the evidence gate can use one consistent scale.
    """

    chunk_id: str
    doc_id: str
    filename: str
    text: str
    rank: int
    score: float
    dense_score: float
    keyword_score: float = 0.0
    page_number: int | None = None
    section: str | None = None
    strategy: str = "dense"

    def location(self) -> str:
        parts = [self.filename]
        if self.page_number is not None:
            parts.append(f"p. {self.page_number}")
        if self.section:
            parts.append(f"§ {self.section}")
        return " · ".join(parts)
