import pandas as pd
import streamlit as st

from src.services import SAMPLE_DOCS_DIR
from src.ui.components import (
    STRATEGY_LABELS,
    current_config,
    current_provider_status,
    get_services,
    model_short,
)

services = get_services()
cfg = current_config()
kb = services.kb
stats = kb.stats()
status = current_provider_status(cfg)

st.title("Enterprise knowledge assistant", anchor=False)
st.markdown(
    "Ask questions about company documents and get answers **grounded in cited evidence**. "
    "Documents are chunked, embedded and indexed (FAISS + BM25); every answer cites the exact chunk "
    "it came from, a relevance gate refuses when evidence is weak, and an evaluation lab measures "
    "retrieval and answer quality on a labelled benchmark."
)

if stats["documents"] == 0:
    with st.container(border=True):
        st.markdown("#### :material/rocket_launch: Get started")
        st.markdown(
            "The knowledge base is empty. Load the three **fictional demo documents** (onboarding policy, "
            "IT incident policy, product FAQ) to try the assistant and the sample benchmark immediately."
        )
        if st.button("Load sample documents", type="primary", icon=":material/download:"):
            with st.spinner("Extracting, chunking and indexing…"):
                reports = kb.load_sample_data(SAMPLE_DOCS_DIR)
            ok = [r for r in reports if r.status == "indexed"]
            st.toast(f"Indexed {len(ok)} sample documents", icon=":material/check:")
            st.rerun()

with st.container(horizontal=True):
    st.metric("Documents", stats["documents"], border=True)
    st.metric("Pages", stats["pages"], border=True, help="PDF pages; TXT/Markdown files count as one page.")
    st.metric("Chunks (current config)", kb.chunks.count(cfg.chunking_key), border=True,
              help=f"Chunking config {cfg.chunking_key}. Other configs are chunked on demand.")
    st.metric("Queries logged", services.query_log.count(), border=True)
    st.metric("Experiments", len(services.experiments.list()), border=True)

left, right = st.columns([3, 2])
with left:
    with st.container(border=True):
        st.markdown("#### :material/settings_suggest: Active configuration")
        st.table(
            pd.DataFrame(
                [
                    ("Embedding model", model_short(cfg.embedding_model)),
                    ("Retrieval strategy", STRATEGY_LABELS[cfg.retrieval_strategy]),
                    ("Top-k", str(cfg.top_k)),
                    ("Chunk size / overlap", f"{cfg.chunk_size} / {cfg.chunk_overlap} characters"),
                    ("Evidence floor (cosine)", f"{cfg.effective_min_relevance:.2f}"),
                    ("LLM provider", f"{cfg.llm_provider} · {cfg.llm_model}"),
                    ("Prompt variant", cfg.prompt_variant),
                ],
                columns=["Setting", "Value"],
            ).set_index("Setting")
        )
with right:
    with st.container(border=True):
        st.markdown("#### :material/monitor_heart: System readiness")
        checks = [
            (True, "SQLite metadata store", "connected"),
            (stats["documents"] > 0, "Knowledge base", f"{stats['documents']} documents indexed" if stats["documents"] else "no documents yet"),
            (bool(services.benchmarks.names()), "Benchmarks", ", ".join(services.benchmarks.names()) or "none"),
            (status.available, "LLM", status.message),
        ]
        for ok, name, detail in checks:
            icon = ":green[:material/check_circle:]" if ok else ":orange[:material/error:]"
            st.markdown(f"{icon} **{name}** — {detail}")
        embedded = services.kb.chunks.embedding_counts()
        st.markdown(
            ":green[:material/check_circle:] **Embedding cache** — "
            + (", ".join(f"{model_short(m)}: {n} vectors" for m, n in embedded.items()) or "empty (built on first query)")
        )
        if not status.available:
            st.caption("Retrieval, the playground and retrieval evaluation work without an LLM.")

c1, c2 = st.columns(2)
with c1:
    with st.container(border=True):
        st.markdown("#### :material/forum: Recent questions")
        recent = services.query_log.recent(6)
        if recent:
            st.dataframe(
                pd.DataFrame([{"time": r["created_at"][:19].replace("T", " "), "question": r["question"],
                               "status": r["status"], "model": r["model"]} for r in recent]),
                hide_index=True,
            )
        else:
            st.caption("No questions asked yet.")
with c2:
    with st.container(border=True):
        st.markdown("#### :material/fact_check: Recent experiments")
        exps = services.experiments.list()[:6]
        if exps:
            st.dataframe(
                pd.DataFrame([{"id": e["id"], "name": e["name"], "kind": e["kind"], "runs": e["run_count"],
                               "status": e["status"], "created": e["created_at"][:19].replace("T", " ")} for e in exps]),
                hide_index=True,
            )
        else:
            st.caption("No experiments yet. Run one in the Evaluation lab.")

with st.container(border=True):
    st.markdown("#### :material/account_tree: How it works")
    st.mermaid_chart(
        """
flowchart LR
    U[Upload PDF / TXT / MD] --> X[Extract + clean] --> C[Chunk with overlap] --> E[Embed]
    E --> F[(FAISS index)]
    C --> B[(BM25 index)]
    Q[Question] --> R{Retrieve: dense / BM25 / hybrid}
    F --> R
    B --> R
    R --> G{Evidence gate}
    G -- weak --> N[Refuse: insufficient evidence]
    G -- ok --> P[Grounded prompt] --> L[LLM] --> V[Validate citations] --> A[Answer + sources]
"""
    )
