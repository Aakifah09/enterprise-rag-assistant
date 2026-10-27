"""TXT and Markdown loading.

Plain text becomes a single unit. Markdown is split at headings so every chunk
can cite the section it came from (Markdown has no page numbers).
"""

from __future__ import annotations

import re

from src.ingestion.cleaning import clean_text
from src.ingestion.errors import ExtractionError
from src.models import ExtractedDocument, TextUnit

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_ENCODINGS = ("utf-8-sig", "utf-16", "cp1252", "latin-1")


def decode_text(filename: str, data: bytes) -> str:
    """Decode bytes as text, rejecting content that is clearly binary."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError as exc:
            raise ExtractionError(f"'{filename}' has an invalid UTF-16 encoding.") from exc
    sample = data[:4096]
    if b"\x00" in sample:
        raise ExtractionError(f"'{filename}' looks like a binary file, not text.")
    for encoding in _ENCODINGS:
        if encoding == "utf-16":
            continue
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ExtractionError(f"'{filename}' could not be decoded as text.")  # pragma: no cover


def load_text(filename: str, data: bytes) -> ExtractedDocument:
    text = clean_text(decode_text(filename, data))
    units = [TextUnit(text=text)] if text else []
    return ExtractedDocument(filename=filename, file_type="txt", units=units, page_count=1)


def split_markdown_sections(text: str) -> list[TextUnit]:
    """Split Markdown into sections at headings, tracking the heading path.

    Heading lines are kept in the section text so they remain searchable.
    """
    units: list[TextUnit] = []
    path: list[str] = []
    buffer: list[str] = []
    current: str | None = None
    in_code = False

    def flush() -> None:
        body = clean_text("\n".join(buffer))
        if body:
            units.append(TextUnit(text=body, section=current))

    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
        match = None if in_code else _HEADING.match(line)
        if match:
            flush()
            buffer = []
            level = len(match.group(1))
            title = match.group(2).strip()
            path = path[: level - 1] + [title]
            current = " > ".join(path)
        buffer.append(line)
    flush()
    return units


def load_markdown(filename: str, data: bytes) -> ExtractedDocument:
    raw = decode_text(filename, data)
    units = split_markdown_sections(raw.replace("\r\n", "\n"))
    meta = {}
    first_heading = next((u.section for u in units if u.section), None)
    if first_heading:
        meta["title"] = first_heading.split(" > ")[0]
    return ExtractedDocument(filename=filename, file_type="md", units=units, page_count=1, metadata=meta)
