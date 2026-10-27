import pandas as pd
import streamlit as st

from src.services import SAMPLE_DOCS_DIR, SECURITY_DEMO_DIR
from src.ui.components import current_config, current_settings, get_services, location

services = get_services()
kb = services.kb
cfg = current_config()

st.title("Knowledge base", anchor=False)
st.caption(
    "Uploaded files are processed in memory. Only the extracted text, chunks and embeddings are stored "
    "(in a local SQLite database); the original files are not kept."
)

# ---------------------------------------------------------------- upload
with st.container(border=True):
    st.markdown("#### :material/upload_file: Add documents")
    with st.form("upload_form", clear_on_submit=True):
        files = st.file_uploader(
            "PDF, TXT or Markdown files",
            type=["pdf", "txt", "md", "markdown"],
            accept_multiple_files=True,
            help=f"Up to {cfg.max_upload_mb} MB per file. Scanned PDFs without a text layer are not OCR'd.",
        )
        submitted = st.form_submit_button("Extract and index", type="primary", icon=":material/bolt:")
    if submitted:
        if not files:
            st.warning("Choose at least one file first.")
        else:
            with st.spinner(f"Processing {len(files)} file(s)…"):
                st.session_state.kb_reports = kb.ingest_many([(f.name, f.getvalue()) for f in files])

    with st.container(horizontal=True):
        if st.button("Load sample documents", icon=":material/library_add:"):
            st.session_state.kb_reports = kb.load_sample_data(SAMPLE_DOCS_DIR)
        if st.button("Add prompt-injection test document", icon=":material/bug_report:",
                     help="A fictional vendor note containing a malicious instruction, to test injection resistance."):
            st.session_state.kb_reports = kb.load_sample_data(SECURITY_DEMO_DIR)

    for r in st.session_state.pop("kb_reports", []):
        if r.status == "indexed":
            st.success(f"**{r.filename}** — {r.pages} page(s), {r.chars:,} characters, {r.chunks} chunks.", icon=":material/check_circle:")
            for w in r.warnings:
                st.warning(f"{r.filename}: {w}", icon=":material/warning:")
        elif r.status == "duplicate":
            st.info(r.message, icon=":material/content_copy:")
        else:
            st.error(r.message, icon=":material/error:")

# ---------------------------------------------------------------- documents
docs = kb.list_documents()
counts = kb.chunks.counts_by_doc(cfg.chunking_key)
embedded = kb.chunks.embedding_counts().get(cfg.embedding_model, 0)

st.markdown("#### :material/description: Indexed documents")
if not docs:
    st.info("No documents yet. Upload files or load the sample documents above.", icon=":material/folder_off:")
    st.stop()

table = pd.DataFrame(
    [
        {
            "document": d.filename,
            "type": d.file_type.upper(),
            "pages": d.page_count,
            "characters": d.char_count,
            f"chunks ({cfg.chunking_key})": counts.get(d.doc_id, 0),
            "extraction": "warnings" if d.warnings else "ok",
            "sample": d.is_sample,
            "added": d.created_at[:19].replace("T", " "),
            "doc_id": d.doc_id,
        }
        for d in docs
    ]
)
event = st.dataframe(table, hide_index=True, on_select="rerun", selection_mode="single-row", key="doc_table")
total_chunks = sum(counts.values())
st.caption(
    f"{len(docs)} documents · {total_chunks} chunks for the current chunking config · "
    f"{embedded} chunk vectors cached for {cfg.embedding_model.split('/')[-1]} "
    "(embeddings are computed on first search). Select a row to preview or remove it."
)

with st.container(horizontal=True):
    if st.button("Rebuild index", icon=":material/refresh:",
                 help="Drop in-memory indexes and rebuild from the stored chunks for the current settings."):
        kb.clear_index_cache()
        with st.spinner("Embedding and indexing…"):
            index, secs = kb.get_index(current_settings())
        st.success(f"Index rebuilt: {len(index.chunks)} chunks in {secs:.1f} s.", icon=":material/check:")

selected = event.selection.rows
if selected:
    doc = docs[selected[0]]
    with st.container(border=True):
        st.markdown(f"#### :material/preview: {doc.filename}")
        cols = st.columns(4)
        cols[0].metric("Pages", doc.page_count)
        cols[1].metric("Characters", f"{doc.char_count:,}")
        cols[2].metric("Chunks", counts.get(doc.doc_id, 0))
        cols[3].metric("Document id", doc.doc_id)
        if doc.metadata:
            st.caption(" · ".join(f"{k}: {v}" for k, v in doc.metadata.items()))
        for w in doc.warnings:
            st.warning(w, icon=":material/warning:")
        units = kb.docs.units(doc.doc_id)
        tab_text, tab_chunks = st.tabs(["Extracted text", "Chunks"])
        with tab_text:
            for u in units[:30]:
                st.markdown(f"**{location(u.page_number, u.section).capitalize()}**")
                st.text(u.text[:3000] + ("…" if len(u.text) > 3000 else ""))
        with tab_chunks:
            chunks = kb.chunks.chunks_for_key(cfg.chunking_key, doc.doc_id)
            st.dataframe(
                pd.DataFrame([{"chunk_id": c.chunk_id, "page": c.page_number, "section": c.section,
                               "chars": len(c.text), "text": c.text} for c in chunks]),
                hide_index=True,
                column_config={"text": st.column_config.TextColumn(width="large")},
            )
        confirm = st.checkbox(f"I want to permanently remove {doc.filename}", key=f"confirm_{doc.doc_id}")
        if st.button("Remove document", type="primary", disabled=not confirm, icon=":material/delete:"):
            kb.remove_document(doc.doc_id)
            st.toast(f"Removed {doc.filename}; indexes will rebuild without it.", icon=":material/delete:")
            st.rerun()

# ---------------------------------------------------------------- chunk search
st.markdown("#### :material/manage_search: Search indexed chunks")
needle = st.text_input("Exact text search (case-insensitive)", placeholder="e.g. annual leave",
                       help="Literal substring search over stored chunks, useful for checking what was extracted.")
if needle.strip():
    hits = kb.chunks.search_text(cfg.chunking_key, needle.strip())
    names = {d.doc_id: d.filename for d in docs}
    st.caption(f"{len(hits)} matching chunk(s){' (showing first 50)' if len(hits) == 50 else ''}")
    if hits:
        st.dataframe(
            pd.DataFrame([{"document": names.get(c.doc_id), "page": c.page_number, "section": c.section,
                           "chunk_id": c.chunk_id, "text": c.text} for c in hits]),
            hide_index=True,
            column_config={"text": st.column_config.TextColumn(width="large")},
        )
