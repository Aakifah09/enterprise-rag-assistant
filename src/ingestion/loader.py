"""Upload validation and format dispatch."""

from __future__ import annotations

import hashlib
import re
from pathlib import PurePath

from src.config import SUPPORTED_EXTENSIONS
from src.ingestion.errors import EmptyDocumentError, ExtractionError, FileTooLargeError, UnsupportedFormatError
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.text_loader import load_markdown, load_text
from src.models import ExtractedDocument

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]+")


def sanitize_filename(filename: str) -> str:
    """Keep only the base name and a conservative character set.

    Upload names can contain directory components; we never want to echo a
    client-side path back or use it to build a local path.
    """
    base = PurePath(filename.replace("\\", "/")).name.strip()
    base = _SAFE_NAME.sub("_", base)
    return base[:120] or "document"


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def doc_id_for(data: bytes) -> str:
    """Deterministic document id derived from content (first 12 hex chars of SHA-256)."""
    return content_hash(data)[:12]


def extract_document(filename: str, data: bytes, max_upload_mb: int = 20) -> ExtractedDocument:
    """Validate an upload and extract its text.

    Raises:
        UnsupportedFormatError, FileTooLargeError, ExtractionError, EmptyDocumentError
    """
    name = sanitize_filename(filename)
    ext = PurePath(name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"'{name}' has unsupported type '{ext or 'none'}'. Supported: PDF, TXT, Markdown."
        )
    if len(data) > max_upload_mb * 1024 * 1024:
        raise FileTooLargeError(f"'{name}' is {len(data) / 1e6:.1f} MB; the limit is {max_upload_mb} MB.")
    if not data:
        raise EmptyDocumentError(f"'{name}' is empty.")

    if ext == ".pdf":
        if not data.lstrip()[:5].startswith(b"%PDF"):
            raise ExtractionError(f"'{name}' does not have a PDF header; it may be corrupted or mislabelled.")
        doc = load_pdf(name, data)
    elif ext in (".md", ".markdown"):
        doc = load_markdown(name, data)
    else:
        doc = load_text(name, data)

    if not doc.has_text:
        hint = " It may be a scanned PDF; OCR is not performed." if ext == ".pdf" else ""
        raise EmptyDocumentError(f"'{name}' contains no extractable text.{hint}")
    return doc
