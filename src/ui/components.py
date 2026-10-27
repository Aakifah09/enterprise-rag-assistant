"""Shared Streamlit helpers: cached services, session settings and render widgets."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.config import EMBEDDING_MODELS, AppConfig, ConfigError
from src.generation.citations import Citation
from src.generation.llm_provider import ProviderStatus
from src.generation.ollama_provider import OllamaProvider
from src.generation.provider_factory import build_provider
from src.knowledge_base import RetrievalSettings
from src.models import RetrievedChunk
from src.services import Services, build_services

STRATEGY_LABELS = {
    "dense": "Dense (FAISS cosine)",
    "keyword": "Keyword (BM25)",
    "hybrid": "Hybrid (weighted fusion)",
    "rrf": "Hybrid (reciprocal rank fusion)",
}

STATUS_BADGES = {
    "answered": ":green-badge[:material/check_circle: Answered from sources]",
    "insufficient_evidence": ":orange-badge[:material/warning: Insufficient evidence]",
    "llm_abstained": ":orange-badge[:material/help: Model found no answer in sources]",
    "llm_unavailable": ":red-badge[:material/cloud_off: LLM unavailable]",
    "llm_error": ":red-badge[:material/error: LLM error]",
    "no_documents": ":gray-badge[:material/folder_off: Knowledge base empty]",
}


@st.cache_resource(show_spinner="Starting knowledge assistant…")
def get_services() -> Services:
    return build_services()


@st.cache_data(ttl=15, show_spinner=False)
def provider_status(provider: str, model: str, base_url: str) -> ProviderStatus:
    """Provider reachability, cached briefly so every rerun does not hit the network."""
    cfg = get_services().cfg.with_overrides(llm_provider=provider)
    if provider == "ollama":
        cfg = cfg.with_overrides(ollama_model=model, ollama_base_url=base_url)
    return build_provider(cfg).status()


@st.cache_data(ttl=30, show_spinner=False)
def ollama_models(base_url: str) -> list[str]:
    try:
        return OllamaProvider("probe", base_url).list_models()
    except Exception:
        return []


def init_session() -> None:
    base = get_services().cfg
    defaults = {
        "set_embedding_model": base.embedding_model,
        "set_strategy": base.retrieval_strategy,
        "set_top_k": base.top_k,
        "set_chunk_size": base.chunk_size,
        "set_chunk_overlap": base.chunk_overlap,
        "set_alpha": base.hybrid_alpha,
        "set_use_model_floor": base.min_relevance is None,
        "set_min_relevance": base.effective_min_relevance,
        "set_llm_provider": base.llm_provider,
        "set_ollama_model": base.ollama_model,
        "set_prompt_variant": base.prompt_variant,
        "chat_messages": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def current_config() -> AppConfig:
    """The base config with this session's sidebar overrides applied (validated)."""
    base = get_services().cfg
    s = st.session_state
    try:
        return base.with_overrides(
            embedding_model=s.set_embedding_model,
            retrieval_strategy=s.set_strategy,
            top_k=int(s.set_top_k),
            chunk_size=int(s.set_chunk_size),
            chunk_overlap=int(s.set_chunk_overlap),
            hybrid_alpha=float(s.set_alpha),
            min_relevance=None if s.set_use_model_floor else float(s.set_min_relevance),
            llm_provider=s.set_llm_provider,
            ollama_model=s.set_ollama_model,
            prompt_variant=s.set_prompt_variant,
        )
    except ConfigError as exc:
        st.error(f"Invalid settings: {exc}. Falling back to defaults.", icon=":material/error:")
        return base


def current_settings() -> RetrievalSettings:
    return RetrievalSettings.from_config(current_config())


def current_provider_status(cfg: AppConfig | None = None) -> ProviderStatus:
    cfg = cfg or current_config()
    return provider_status(cfg.llm_provider, cfg.ollama_model, cfg.ollama_base_url)


def model_short(name: str) -> str:
    return name.split("/")[-1]


def embedding_model_options() -> list[str]:
    return list(EMBEDDING_MODELS)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def location(page: int | None, section: str | None) -> str:
    parts = []
    if page is not None:
        parts.append(f"page {page}")
    if section:
        parts.append(f"section “{section}”")
    return ", ".join(parts) or "whole document"


def render_citations(citations: list[Citation], invalid: list[str]) -> None:
    if citations:
        st.markdown("**Citations**")
        for c in citations:
            with st.expander(f"[{c.label}] {c.filename} · {location(c.page_number, c.section)}", icon=":material/format_quote:"):
                st.caption(f"Chunk `{c.chunk_id}`")
                st.text(c.excerpt)
    if invalid:
        st.warning(
            f"The model cited {', '.join(invalid)}, which do not correspond to any retrieved source. "
            "These labels were discarded and are not shown as citations.",
            icon=":material/link_off:",
        )


def render_retrieved(chunks: list[RetrievedChunk], relevant: list[bool] | None = None, key: str = "") -> None:
    """Ranked chunk list with scores and expandable excerpts."""
    if not chunks:
        st.info("No chunks were retrieved.", icon=":material/search_off:")
        return
    for i, c in enumerate(chunks):
        mark = ""
        if relevant is not None:
            mark = " :green[:material/check:]" if relevant[i] else " :gray[:material/close:]"
        title = f"#{c.rank} · {c.filename} · {location(c.page_number, c.section)}{mark}"
        with st.expander(title):
            cols = st.columns(3)
            cols[0].metric("Rank score", fmt(c.score), help=score_help(c.strategy))
            cols[1].metric("Cosine similarity", fmt(c.dense_score), help="Query–chunk cosine similarity of the embeddings (-1…1). Not a probability.")
            cols[2].metric("BM25", fmt(c.keyword_score, 2), help="Keyword relevance (unbounded, only comparable within one query).")
            st.caption(f"Chunk `{c.chunk_id}`")
            st.text(c.text)


def score_help(strategy: str) -> str:
    return {
        "dense": "Cosine similarity between query and chunk embeddings.",
        "keyword": "Okapi BM25 score; larger is better, unbounded.",
        "hybrid": "α·minmax(cosine) + (1-α)·minmax(BM25) over this query's candidates, in [0, 1].",
        "rrf": "Σ 1/(60 + rank) across the dense and BM25 rankings.",
    }.get(strategy, "")


def retrieved_frame(chunks: list[RetrievedChunk]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rank": c.rank,
                "document": c.filename,
                "page": c.page_number,
                "section": c.section,
                "score": round(c.score, 4),
                "cosine": round(c.dense_score, 4),
                "bm25": round(c.keyword_score, 3),
                "chunk_id": c.chunk_id,
            }
            for c in chunks
        ]
    )


def metrics_frame(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        row = {"run": r["label"], **{k: v for k, v in r["config"].items() if k != "candidate_pool"}}
        row.update(r["metrics"])
        rows.append(row)
    return pd.DataFrame(rows)


def metric_bar_chart(df: pd.DataFrame, metric: str, color: str = "strategy", title: str | None = None):
    """Horizontal bar chart of one metric per run, coloured by a config field, with tooltips."""
    import altair as alt

    data = df.dropna(subset=[metric]).copy()
    if data.empty:
        return None
    color_enc = alt.Color(f"{color}:N", title=color) if color in data else alt.value("#3987e5")
    tooltip = [alt.Tooltip("run:N"), alt.Tooltip(f"{metric}:Q", format=".3f")]
    if color in data:
        tooltip.append(alt.Tooltip(f"{color}:N"))
    return (
        alt.Chart(data, title=title or metric)
        .mark_bar(cornerRadiusEnd=4, height={"band": 0.75})
        .encode(
            x=alt.X(f"{metric}:Q", title=metric, scale=alt.Scale(zero=True)),
            y=alt.Y("run:N", sort="-x", title=None, axis=alt.Axis(labelLimit=380)),
            color=color_enc,
            tooltip=tooltip,
        )
        .properties(height=max(120, 28 * len(data)))
    )
