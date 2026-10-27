"""Sentence-aware chunking with configurable size and overlap.

Chunks never cross a :class:`TextUnit` boundary (a PDF page or Markdown
section), so every chunk maps to exactly one page/section for citations.

Within a unit, text is split into sentence/line segments which are packed
greedily into chunks of at most ``chunk_size`` characters. The next chunk
starts with the trailing segments of the previous one whose combined length
is at most ``chunk_overlap`` characters, i.e. overlap is aligned to sentence
boundaries rather than cutting words in half. Sentences longer than
``chunk_size`` are hard-split at whitespace.
"""

from __future__ import annotations

import re

from src.config import chunking_key
from src.models import Chunk, TextUnit

_BOUNDARY = re.compile(r"(?<=[.!?;:])\s+|\n+")


def _segments(text: str, max_len: int) -> list[tuple[int, int]]:
    """Return (start, end) spans of sentence-like segments, each <= max_len."""
    starts = [0] + [m.end() for m in _BOUNDARY.finditer(text)]
    spans: list[tuple[int, int]] = []
    for i, start in enumerate(starts):
        stop = starts[i + 1] if i + 1 < len(starts) else len(text)
        piece = text[start:stop].rstrip()
        if not piece:
            continue
        end = start + len(piece)
        if end - start <= max_len:
            spans.append((start, end))
            continue
        # Hard-split an over-long sentence at whitespace.
        pos = start
        while pos < end:
            limit = min(pos + max_len, end)
            if limit < end:
                cut = text.rfind(" ", pos + 1, limit)
                limit = cut if cut > pos else limit
            spans.append((pos, limit))
            pos = limit
            while pos < end and text[pos] == " ":
                pos += 1
    return spans


def chunk_spans(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[int, int]]:
    """Compute chunk character spans for one unit of text."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")
    segs = _segments(text, chunk_size)
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(segs):
        start = segs[i][0]
        j = i
        while j + 1 < len(segs) and segs[j + 1][1] - start <= chunk_size:
            j += 1
        end = segs[j][1]
        spans.append((start, end))
        if j + 1 >= len(segs):
            break
        k = j + 1
        while k - 1 > i and end - segs[k - 1][0] <= chunk_overlap:
            k -= 1
        i = k
    return spans


def make_chunk_id(doc_id: str, chunk_size: int, chunk_overlap: int, index: int) -> str:
    return f"{doc_id}-{chunking_key(chunk_size, chunk_overlap)}-{index:04d}"


def chunk_units(doc_id: str, units: list[TextUnit], chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Chunk all units of a document, numbering chunks document-wide."""
    chunks: list[Chunk] = []
    for unit in units:
        for start, end in chunk_spans(unit.text, chunk_size, chunk_overlap):
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(doc_id, chunk_size, chunk_overlap, index),
                    doc_id=doc_id,
                    chunk_index=index,
                    text=unit.text[start:end],
                    page_number=unit.page_number,
                    section=unit.section,
                    char_start=start,
                    char_end=end,
                )
            )
    return chunks
