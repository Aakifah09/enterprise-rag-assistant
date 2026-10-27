import fitz
import pytest

from src.ingestion.cleaning import clean_text
from src.ingestion.errors import (
    DuplicateDocumentError,
    EmptyDocumentError,
    ExtractionError,
    FileTooLargeError,
    UnsupportedFormatError,
)
from src.ingestion.loader import doc_id_for, extract_document, sanitize_filename
from src.ingestion.text_loader import split_markdown_sections
from tests.conftest import IT_TXT, POLICY_MD, make_pdf


def test_pdf_pages_and_metadata_are_preserved():
    doc = extract_document("policy.pdf", make_pdf(["First page about leave.", "Second page about expenses."]))
    assert doc.file_type == "pdf"
    assert doc.page_count == 2
    assert [u.page_number for u in doc.units] == [1, 2]
    assert "expenses" in doc.units[1].text


def test_txt_is_loaded_as_single_unit_without_page_number():
    doc = extract_document("it.txt", IT_TXT)
    assert len(doc.units) == 1
    assert doc.units[0].page_number is None
    assert "15 minutes" in doc.units[0].text


def test_markdown_sections_track_heading_path():
    doc = extract_document("policy.md", POLICY_MD)
    sections = [u.section for u in doc.units]
    assert "Leave Policy > Annual leave" in sections
    assert "Leave Policy > Sick leave" in sections
    assert doc.metadata["title"] == "Leave Policy"


def test_markdown_heading_inside_code_block_is_not_a_section():
    units = split_markdown_sections("# Title\n```\n# not a heading\n```\ntext")
    assert [u.section for u in units] == ["Title"]


def test_unsupported_extension_rejected():
    with pytest.raises(UnsupportedFormatError):
        extract_document("sheet.xlsx", b"PK\x03\x04data")


def test_file_size_limit_enforced():
    with pytest.raises(FileTooLargeError):
        extract_document("big.txt", b"a" * (1024 * 1024 + 1), max_upload_mb=1)


def test_empty_file_rejected():
    with pytest.raises(EmptyDocumentError):
        extract_document("empty.txt", b"")


def test_whitespace_only_file_rejected():
    with pytest.raises(EmptyDocumentError):
        extract_document("blank.md", b"   \n\n\t  ")


def test_malformed_pdf_rejected():
    with pytest.raises(ExtractionError):
        extract_document("broken.pdf", b"%PDF-1.7 this is not really a pdf")


def test_mislabelled_pdf_rejected():
    with pytest.raises(ExtractionError):
        extract_document("fake.pdf", b"just some text")


def test_binary_content_in_txt_rejected():
    with pytest.raises(ExtractionError):
        extract_document("binary.txt", b"\x00\x01\x02\x03binary")


def test_scanned_pdf_without_text_layer_gives_clear_error():
    doc = fitz.open()
    page = doc.new_page()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 50, 50), False)
    pix.clear_with(200)
    page.insert_image(fitz.Rect(0, 0, 200, 200), pixmap=pix)
    data = doc.tobytes()
    with pytest.raises(EmptyDocumentError, match="scanned"):
        extract_document("scan.pdf", data)


def test_partially_scanned_pdf_warns_about_image_pages():
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((50, 50), "This page has a normal text layer with enough characters.")
    p2 = doc.new_page()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 20, 20), False)
    pix.clear_with(100)
    p2.insert_image(fitz.Rect(0, 0, 100, 100), pixmap=pix)
    result = extract_document("mixed.pdf", doc.tobytes())
    assert result.page_count == 2
    assert any("OCR is not performed" in w for w in result.warnings)


def test_encrypted_pdf_rejected():
    doc = fitz.open()
    doc.new_page().insert_text((50, 50), "secret")
    data = doc.tobytes(encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="owner")
    with pytest.raises(ExtractionError, match="password"):
        extract_document("locked.pdf", data)


def test_cp1252_text_is_decoded():
    doc = extract_document("legacy.txt", "Caf\xe9 policy \u2013 r\xe9sum\xe9".encode("cp1252"))
    assert "Café" in doc.units[0].text


def test_sanitize_filename_strips_paths_and_odd_characters():
    assert sanitize_filename("C:\\Users\\me\\secret\\report.pdf") == "report.pdf"
    assert sanitize_filename("../../etc/passwd.txt") == "passwd.txt"
    assert sanitize_filename("we<i>rd|name?.md") == "we_i_rd_name_.md"


def test_clean_text_normalises_whitespace_and_hyphenation():
    raw = "infor-\nmation   is\t\tkey\r\n\r\n\r\n\r\nNew ﬁle\x07"
    assert clean_text(raw) == "information is key\n\nNew file"


def test_doc_id_is_deterministic():
    assert doc_id_for(b"abc") == doc_id_for(b"abc")
    assert doc_id_for(b"abc") != doc_id_for(b"abd")


def test_duplicate_upload_detected(kb):
    kb.add_document("a.txt", IT_TXT)
    with pytest.raises(DuplicateDocumentError) as info:
        kb.add_document("copy_of_a.txt", IT_TXT)
    assert info.value.existing_doc_id == doc_id_for(IT_TXT)


def test_ingest_many_reports_errors_without_raising(kb):
    reports = kb.ingest_many([("ok.txt", IT_TXT), ("bad.docx", b"x"), ("ok_again.txt", IT_TXT), ("empty.md", b"")])
    assert [r.status for r in reports] == ["indexed", "error", "duplicate", "error"]
    assert reports[0].chunks > 0
