import itertools

import pandas as pd
import streamlit as st

from src.evaluation.retrieval_metrics import chunk_relevance, score_retrieval
from src.knowledge_base import RetrievalSettings
from src.ui.components import (
    STRATEGY_LABELS,
    current_config,
    embedding_model_options,
    get_services,
    model_short,
    render_retrieved,
    retrieved_frame,
)

services = get_services()
kb = services.kb
cfg = current_config()

st.title("Retrieval playground", anchor=False)
st.caption("Run one question through several retrieval configurations side by side. No LLM is needed.")

if kb.stats()["documents"] == 0:
    st.info("Load documents first (Overview → Load sample documents).", icon=":material/folder_off:")
    st.stop()

# Benchmark questions come with relevance labels, so the playground can mark relevant chunks.
bench_items = {}
for name in services.benchmarks.names():
    try:
        for item in services.load_benchmark(name).items:
            bench_items[f"{item.qid} · {item.question}"] = item
    except Exception:
        continue

with st.form("playground"):
    source = st.segmented_control("Question source", ["Benchmark question", "Free text"], default="Benchmark question")
    picked = st.selectbox("Benchmark question (labelled)", list(bench_items), disabled=not bench_items)
    free = st.text_input("Free-text question", placeholder="e.g. What happens if I lose my laptop?")
    c1, c2, c3 = st.columns(3)
    strategies = c1.pills("Strategies", list(STRATEGY_LABELS), selection_mode="multi", default=["dense", "keyword", "hybrid"],
                          format_func=lambda s: s.upper() if s == "rrf" else s.capitalize())
    top_k = c2.slider("Top-k", 1, 15, cfg.top_k)
    alpha = c3.slider("Hybrid α (dense weight)", 0.0, 1.0, cfg.hybrid_alpha, 0.05)
    c4, c5 = st.columns(2)
    chunk_cfgs = c4.multiselect("Chunk size / overlap", ["400/60", "800/120", "1200/200", "800/0"], default=[f"{cfg.chunk_size}/{cfg.chunk_overlap}"])
    models = c5.multiselect("Embedding models", embedding_model_options(), default=[cfg.embedding_model], format_func=model_short)
    run = st.form_submit_button("Compare configurations", type="primary", icon=":material/play_arrow:")

if run:
    item = bench_items.get(picked) if source == "Benchmark question" else None
    question = item.question if item else free.strip()
    if not question:
        st.warning("Enter a question.")
        st.stop()
    if not strategies or not chunk_cfgs or not models:
        st.warning("Pick at least one strategy, chunk configuration and model.")
        st.stop()
    configs = []
    for model, cc, strat in itertools.product(models, chunk_cfgs, strategies):
        size, overlap = (int(x) for x in cc.split("/"))
        configs.append(RetrievalSettings(strat, top_k, size, overlap, model, alpha, max(30, top_k)))
    results = []
    progress = st.progress(0.0, "Retrieving…")
    for i, s in enumerate(configs):
        progress.progress(i / len(configs), f"{s.label()} (first use of a model/chunk config builds its index)")
        results.append(kb.retrieve(question, s))
    progress.empty()
    st.session_state.playground_result = {"question": question, "item": item, "results": results}

state = st.session_state.get("playground_result")
if not state:
    st.stop()

question, item, results = state["question"], state["item"], state["results"]
st.markdown(f"#### Results for: _{question}_")
if item is not None:
    st.caption("Chunks marked :green[:material/check:] contain a labelled evidence passage for this benchmark question.")

summary = []
for r in results:
    row = {
        "configuration": r.settings.label(),
        "strategy": r.settings.strategy,
        "best cosine": round(r.max_dense_score, 3),
        "latency ms": round(r.retrieval_s * 1000, 1),
        "top-1 document": r.chunks[0].filename if r.chunks else None,
    }
    if item is not None and item.answerable:
        sc = score_retrieval(item, r.chunks, r.settings.top_k)
        row.update({"precision@k": sc.precision, "recall@k": sc.recall, "RR": sc.reciprocal_rank})
    summary.append(row)
st.dataframe(pd.DataFrame(summary), hide_index=True)

# Overlap between result sets explains *how* configurations differ.
if len(results) > 1:
    sets = [{c.text[:80] for c in r.chunks} for r in results]
    labels = [r.settings.label() for r in results]
    overlap = pd.DataFrame(
        [[len(a & b) / len(a | b) if a | b else 1.0 for b in sets] for a in sets], index=labels, columns=labels
    )
    with st.expander("Result overlap (Jaccard similarity of retrieved passages)", icon=":material/join_inner:"):
        st.dataframe(overlap.style.format("{:.2f}").background_gradient(cmap="Blues", vmin=0, vmax=1))

with st.expander("Why do configurations return different chunks?", icon=":material/lightbulb:"):
    st.markdown(
        """
- **Dense** retrieval compares meaning: it finds paraphrases ("time off" ≈ "annual leave") but can miss exact
  identifiers such as *P1*, *SAML* or a product name, and can rank a topically similar but wrong passage first.
- **Keyword (BM25)** rewards exact, rare terms. It excels at codes, numbers and names, but fails when the question
  uses different words from the document.
- **Hybrid** fusion combines both signals. Weighted fusion keeps score magnitudes (after min-max scaling); RRF uses
  only ranks, so a chunk ranked well by both retrievers rises to the top.
- **Chunk size** trades precision for context: small chunks pin-point a sentence but may split an answer across
  chunks; large chunks keep context but dilute the embedding with unrelated text and use more prompt tokens.
- **Overlap** repeats trailing sentences so facts near a boundary appear whole in at least one chunk.
- **Embedding models** differ in training data and similarity scale, so cosine values are not comparable across models.
"""
    )

cols = st.columns(min(len(results), 3))
for i, r in enumerate(results):
    with cols[i % len(cols)]:
        with st.container(border=True):
            st.markdown(f"**{r.settings.label()}**")
            relevant = [chunk_relevance(c, item) for c in r.chunks] if item is not None and item.answerable else None
            render_retrieved(r.chunks, relevant, key=f"pg{i}")
            st.download_button("Download ranking (CSV)", retrieved_frame(r.chunks).to_csv(index=False),
                               file_name=f"ranking_{i + 1}.csv", key=f"dl{i}", icon=":material/download:")
