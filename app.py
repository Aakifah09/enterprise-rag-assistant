"""Enterprise Knowledge Assistant — Streamlit entry point.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generation.prompt_templates import PROMPT_DESCRIPTIONS, PROMPT_VARIANTS  # noqa: E402
from src.ui.components import (  # noqa: E402
    STRATEGY_LABELS,
    current_config,
    current_provider_status,
    embedding_model_options,
    get_services,
    init_session,
    model_short,
    ollama_models,
)

st.set_page_config(page_title="Enterprise Knowledge Assistant", page_icon=":material/hub:", layout="wide")

services = get_services()
init_session()

pages = {
    "": [
        st.Page("app_pages/overview.py", title="Overview", icon=":material/dashboard:", default=True),
        st.Page("app_pages/knowledge_base.py", title="Knowledge base", icon=":material/folder_open:"),
        st.Page("app_pages/ask.py", title="Ask your documents", icon=":material/forum:"),
    ],
    "Evaluate": [
        st.Page("app_pages/playground.py", title="Retrieval playground", icon=":material/science:"),
        st.Page("app_pages/evaluation.py", title="Evaluation lab", icon=":material/fact_check:"),
        st.Page("app_pages/history.py", title="Experiment history", icon=":material/history:"),
    ],
    "Learn": [
        st.Page("app_pages/architecture.py", title="Architecture & docs", icon=":material/account_tree:"),
    ],
}
page = st.navigation(pages, position="sidebar")

with st.sidebar:
    st.markdown("### :material/tune: Pipeline settings")
    st.caption("Applied to Ask, Playground defaults and evaluations in this session.")

    with st.expander("Retrieval", icon=":material/search:", expanded=True):
        st.selectbox("Embedding model", embedding_model_options(), key="set_embedding_model", format_func=model_short)
        st.selectbox("Strategy", list(STRATEGY_LABELS), key="set_strategy", format_func=STRATEGY_LABELS.get)
        st.slider("Top-k chunks", 1, 15, key="set_top_k")
        if st.session_state.set_strategy == "hybrid":
            st.slider("Dense weight α", 0.0, 1.0, step=0.05, key="set_alpha",
                      help="fused = α·dense + (1-α)·BM25 after min-max normalisation")
        st.selectbox("Chunk size (characters)", [400, 600, 800, 1200], key="set_chunk_size")
        st.selectbox("Chunk overlap (characters)", [0, 60, 120, 200], key="set_chunk_overlap")

    with st.expander("Evidence gate", icon=":material/shield:"):
        st.toggle("Use model default floor", key="set_use_model_floor")
        st.slider("Minimum cosine similarity", 0.0, 1.0, step=0.01, key="set_min_relevance",
                  disabled=st.session_state.set_use_model_floor,
                  help="If no retrieved chunk reaches this similarity, the assistant refuses without calling the LLM.")

    with st.expander("Generation", icon=":material/smart_toy:"):
        st.selectbox("LLM provider", ["ollama", "openai_compatible", "none"], key="set_llm_provider",
                     format_func={"ollama": "Ollama (local)", "openai_compatible": "OpenAI-compatible API",
                                  "none": "None (retrieval only)"}.get)
        if st.session_state.set_llm_provider == "ollama":
            installed = ollama_models(services.cfg.ollama_base_url)
            options = sorted(set(installed) | {st.session_state.set_ollama_model})
            st.selectbox("Ollama model", options, key="set_ollama_model")
        st.selectbox("Prompt", list(PROMPT_VARIANTS), key="set_prompt_variant",
                     help="\n\n".join(f"**{k}**: {v}" for k, v in PROMPT_DESCRIPTIONS.items()))

    cfg = current_config()
    status = current_provider_status(cfg)
    if status.available:
        st.success(status.message, icon=":material/check_circle:")
    else:
        st.warning(status.message, icon=":material/cloud_off:")
    st.caption(f"Evidence floor in use: **{cfg.effective_min_relevance:.2f}** cosine")

page.run()
