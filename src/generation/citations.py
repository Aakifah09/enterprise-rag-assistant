"""Parse, resolve and validate citations in generated answers.

The model cites sources by label (``[S2]``). A label is *valid* only if it
maps to one of the chunks that were actually placed in the prompt **and**
that chunk still exists in the database. Invalid labels are reported, never
silently turned into citations.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from src.generation.prompt_templates import INSUFFICIENT_MARKER
from src.models import RetrievedChunk

_GROUP = re.compile(r"\[((?:\s*S\d+\s*[,;]?)+)\]", re.IGNORECASE)
_ONE = re.compile(r"S(\d+)", re.IGNORECASE)

_ABSTAIN_PATTERNS = [
    re.compile(re.escape(INSUFFICIENT_MARKER), re.IGNORECASE),
    re.compile(r"\b(?:cannot|can't|could not|couldn't|unable to)\s+(?:be\s+)?(?:find|found|locate|determine|answer)", re.I),
    re.compile(r"\b(?:does not|doesn't|do not|don't)\s+(?:contain|mention|provide|include|specify|say)", re.I),
    re.compile(r"\bno (?:information|mention|details?)\b.*\b(?:provided|context|sources?|documents?)\b", re.I),
    re.compile(r"\bnot (?:mentioned|specified|stated|provided|available) in the (?:provided )?(?:context|sources?|documents?)", re.I),
]


@dataclass
class Citation:
    label: str
    chunk_id: str
    doc_id: str
    filename: str
    page_number: int | None
    section: str | None
    excerpt: str

    def to_dict(self) -> dict:
        return asdict(self)


def extract_labels(answer: str) -> list[str]:
    """All cited labels in order of first appearance, normalised to 'S<n>'."""
    seen: dict[str, None] = {}
    for group in _GROUP.finditer(answer or ""):
        for num in _ONE.findall(group.group(1)):
            seen.setdefault(f"S{int(num)}", None)
    return list(seen)


def resolve_citations(
    answer: str, sources: list[RetrievedChunk], existing_chunk_ids: set[str] | None = None
) -> tuple[list[Citation], list[str]]:
    """Map labels to retrieved chunks. Returns (valid citations, invalid labels)."""
    valid: list[Citation] = []
    invalid: list[str] = []
    for label in extract_labels(answer):
        idx = int(label[1:]) - 1
        if not 0 <= idx < len(sources):
            invalid.append(label)
            continue
        chunk = sources[idx]
        if existing_chunk_ids is not None and chunk.chunk_id not in existing_chunk_ids:
            invalid.append(label)
            continue
        valid.append(
            Citation(
                label=label,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                filename=chunk.filename,
                page_number=chunk.page_number,
                section=chunk.section,
                excerpt=chunk.text,
            )
        )
    return valid, invalid


def is_abstention(answer: str) -> bool:
    """Heuristic: does the answer decline for lack of evidence?"""
    head = (answer or "").strip()[:400]
    return any(p.search(head) for p in _ABSTAIN_PATTERNS)
