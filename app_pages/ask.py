import pandas as pd
import streamlit as st

from src.generation.rag_pipeline import RAGResult
from src.ui.components import (
    STATUS_BADGES,
    STRATEGY_LABELS,
    current_config,
    current_provider_status,
    fmt,
    get_services,
    render_citations,
    render_retrieved,
)

services = get_services()
cfg = current_config()
status = current_provider_status(cfg)

SUGGESTIONS = {
    "Annual leave carry-over": ("beach_access", "Can I carry unused annual leave into next year, and is there a deadline?"),
    "P1 incident response": ("emergency", "What is the first response time and resolution target for a P1 critical incident?"),
    "Refunds": ("payments", "Can a customer get a refund on an annual Beacon subscription?"),
    "Working abroad": ("flight", "How many days per year can I work remotely from another country, and whose approval do I need?"),
    "Not in the documents": ("block", "How many weeks of paid parental leave does Halcyon Ridge offer?"),
}

st.title("Ask your documents", anchor=False)
st.caption(
    f"{STRATEGY_LABELS[cfg.retrieval_strategy]} · top-{cfg.top_k} · chunks {cfg.chunk_size}/{cfg.chunk_overlap} · "
    f"LLM: {cfg.llm_model if cfg.llm_provider != 'none' else 'none'} · prompt: {cfg.prompt_variant}"
)

if services.kb.stats()["documents"] == 0:
    st.info("The knowledge base is empty. Add documents on the Knowledge base page or load the sample data on Overview.",
            icon=":material/folder_off:")
if not status.available:
    st.warning(
        f"**LLM unavailable** — {status.message} Questions still run retrieval and show the evidence, "
        "but no answer is generated (the assistant never invents one).",
        icon=":material/cloud_off:",
    )


def render_result(result: RAGResult, show_answer: bool = True) -> None:
    st.markdown(STATUS_BADGES.get(result.status, result.status))
    if show_answer and result.answer:
        st.markdown(result.answer)
    if result.status != "answered":
        box = st.warning if result.status in ("insufficient_evidence", "llm_abstained") else st.error
        box(result.message, icon=":material/info:")
    render_citations(result.citations, result.invalid_citations)
    ev = result.evidence
    t = result.timings
    with st.expander(f"Evidence & retrieval details · {len(result.retrieved)} chunks · best cosine {ev.max_dense_score:.2f}",
                     icon=":material/insights:"):
        cols = st.columns(4)
        cols[0].metric("Best cosine", fmt(ev.max_dense_score, 2), help="Similarity, not a probability of correctness.")
        cols[1].metric("Evidence floor", fmt(ev.threshold, 2))
        cols[2].metric("Query-term coverage", f"{ev.term_coverage:.0%}", help="Share of question keywords present in retrieved chunks.")
        cols[3].metric("Gate", "pass" if ev.sufficient else "blocked")
        st.caption(ev.reason)
        lat = st.columns(4)
        lat[0].metric("Retrieval", f"{t['retrieval_s'] * 1000:.0f} ms")
        lat[1].metric("Generation", f"{t['generation_s']:.1f} s" if t["generation_s"] else "—")
        lat[2].metric("First token", f"{t['first_token_s']:.1f} s" if t.get("first_token_s") else "—")
        lat[3].metric("End-to-end", f"{t['total_s']:.1f} s")
        if result.usage.get("total_tokens") is not None:
            cost = result.usage.get("estimated_cost")
            st.caption(
                f"Tokens: {result.usage['prompt_tokens']} prompt + {result.usage['completion_tokens']} completion"
                + (f" · estimated cost {cost:.5f}" if cost is not None else " · cost not estimated (no pricing configured)")
            )
        render_retrieved(result.retrieved)


tab_chat, tab_history = st.tabs([":material/chat: Conversation", ":material/history: Query history"])

with tab_chat:
    messages = st.session_state.chat_messages
    if not messages:
        picked = st.pills("Try a sample question", list(SUGGESTIONS), key="ask_suggestion",
                         format_func=lambda k: f":material/{SUGGESTIONS[k][0]}: {k}")
    else:
        picked = None
    for msg in messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(msg["content"])
            else:
                render_result(msg["result"])

    prompt = st.chat_input("Ask a question about your documents", submit_mode="disable")
    question = prompt or (SUGGESTIONS[picked][1] if picked else None)

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            pipeline = services.pipeline(cfg)
            with st.status(":shimmer[Retrieving evidence]", type="compact") as stat:
                prep = pipeline.prepare(question)
                stat.update(
                    label=f"Retrieved {len(prep.retrieval.chunks)} chunks in {prep.retrieval.retrieval_s * 1000:.0f} ms"
                    + ("" if prep.evidence.sufficient else " · evidence too weak"),
                    state="complete",
                )
            if prep.blocked_status:
                list(pipeline.stream_answer(prep))
                render_result(prep.result)
            else:
                streamed = st.write_stream(pipeline.stream_answer(prep))
                render_result(prep.result, show_answer=not streamed)
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "result": prep.result})

    if messages and st.button("Clear conversation", icon=":material/delete_sweep:"):
        st.session_state.chat_messages = []
        st.rerun()

with tab_history:
    history = services.query_log.recent(100)
    if not history:
        st.caption("No questions logged yet.")
    else:
        df = pd.DataFrame(
            [
                {
                    "time": h["created_at"][:19].replace("T", " "),
                    "question": h["question"],
                    "status": h["status"],
                    "citations": len(h["citations"]),
                    "strategy": h["config"].get("strategy"),
                    "top_k": h["config"].get("top_k"),
                    "model": h["model"],
                    "end-to-end s": round(h["timings"].get("total_s", 0), 2),
                    "answer": h["answer"],
                }
                for h in history
            ]
        )
        st.dataframe(df, hide_index=True, column_config={"answer": st.column_config.TextColumn(width="large")})
        if st.button("Clear query history", icon=":material/delete:"):
            services.query_log.clear()
            st.rerun()
