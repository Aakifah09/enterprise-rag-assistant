"""PDF text extraction with PyMuPDF, one :class:`TextUnit` per page."""

from __future__ import annotations

import fitz  # PyMuPDF

from src.ingestion.cleaning import clean_text
from src.ingestion.errors import ExtractionError
from src.models import ExtractedDocument, TextUnit

# A page with fewer extractable characters than this but with images is
# reported as "probably scanned".
_SCANNED_CHAR_THRESHOLD = 20


def load_pdf(filename: str, data: bytes) -> ExtractedDocument:
    """Extract text from PDF bytes held in memory (the file is never written to disk).

    Raises:
        ExtractionError: the bytes are not a readable PDF or are encrypted.
    """
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several exception types
        raise ExtractionError(f"'{filename}' is not a valid or readable PDF.") from exc

    try:
        if doc.needs_pass:
            raise ExtractionError(f"'{filename}' is password-protected; remove the password and re-upload.")

        units: list[TextUnit] = []
        warnings: list[str] = []
        scanned_pages: list[int] = []
        for index, page in enumerate(doc):
            page_no = index + 1
            try:
                raw = page.get_text("text")
            except Exception:
                warnings.append(f"Page {page_no}: text extraction failed and was skipped.")
                continue
            text = clean_text(raw)
            if len(text) < _SCANNED_CHAR_THRESHOLD and page.get_images(full=False):
                scanned_pages.append(page_no)
            if text:
                units.append(TextUnit(text=text, page_number=page_no))

        if scanned_pages:
            shown = ", ".join(map(str, scanned_pages[:10]))
            more = "…" if len(scanned_pages) > 10 else ""
            warnings.append(
                f"Pages {shown}{more} look like scanned images with no text layer. "
                "OCR is not performed, so their content is not searchable."
            )

        meta = {k: v for k, v in (doc.metadata or {}).items() if v and k in {"title", "author", "subject", "creationDate"}}
        return ExtractedDocument(
            filename=filename,
            file_type="pdf",
            units=units,
            page_count=doc.page_count,
            metadata=meta,
            warnings=warnings,
        )
    finally:
        doc.close()
