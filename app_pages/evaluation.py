import json

import pandas as pd
import streamlit as st

from src.evaluation.benchmark import (
    BenchmarkError,
    benchmark_from_csv,
    benchmark_from_dict,
    benchmark_to_csv,
    parse_item,
    validate_items,
)
from src.evaluation.experiment_runner import build_grid
from src.generation.prompt_templates import PROMPT_VARIANTS
from src.ui.components import (
    STRATEGY_LABELS,
    current_config,
    current_provider_status,
    current_settings,
    embedding_model_options,
    get_services,
    metric_bar_chart,
    metrics_frame,
    model_short,
)

services = get_services()
cfg = current_config()

st.title("Evaluation lab", anchor=False)
st.caption(
    "All metrics below are computed from actual runs against a labelled benchmark and stored in SQLite. "
    "They describe *this* corpus and benchmark only; they are not a general accuracy figure for any model."
)

names = services.benchmarks.names()
if not names:
    st.warning("No benchmark available. Import one in the Benchmark tab.")
bench_name = st.selectbox("Benchmark", names, key="eval_bench") if names else None
bench = services.load_benchmark(bench_name) if bench_name else None
if bench is not None and services.benchmarks.get(bench_name)["is_sample"]:
    st.info(
        "**Sample benchmark** — 22 questions written for the fictional demo documents (18 answerable with labelled "
        "evidence passages, 4 unanswerable). Load the sample documents before evaluating.",
        icon=":material/science:",
    )

tab_bench, tab_ret, tab_ans = st.tabs(
    [":material/list_alt: Benchmark", ":material/search: Retrieval evaluation", ":material/rate_review: Answer evaluation"]
)

# =============================================================== benchmark
with tab_bench:
    if bench is not None:
        st.markdown(
            "Edit questions inline. List fields use `|` between values; **evidence** passages use `||`. "
            "A retrieved chunk counts as relevant when it contains one of the evidence passages (≥60% contiguous overlap)."
        )
        df = pd.DataFrame(
            [
                {
                    **i.to_dict(),
                    "relevant_documents": "|".join(i.relevant_documents),
                    "evidence": "||".join(i.evidence),
                    "relevant_chunk_ids": "|".join(i.relevant_chunk_ids),
                    "key_facts": "|".join(i.key_facts),
                }
                for i in bench.items
            ]
        )
        edited = st.data_editor(
            df,
            num_rows="dynamic",
            hide_index=True,
            key=f"editor_{bench_name}",
            column_config={
                "question": st.column_config.TextColumn(width="large", required=True),
                "qid": st.column_config.TextColumn(required=True),
                "answerable": st.column_config.CheckboxColumn(default=True),
                "evidence": st.column_config.TextColumn(width="large"),
            },
        )
        with st.container(horizontal=True):
            save_as = st.text_input("Save as", value=bench_name, label_visibility="collapsed")
            if st.button("Save benchmark", icon=":material/save:", type="primary"):
                try:
                    items = [parse_item(r, n + 1) for n, r in enumerate(edited.to_dict("records"))]
                    warnings = validate_items(items)
                    stored = services.benchmarks.get(save_as)
                    services.benchmarks.upsert(save_as, [i.to_dict() for i in items], bench.description,
                                               is_sample=bool(stored and stored["is_sample"] and save_as == bench_name))
                    st.success(f"Saved {len(items)} items to '{save_as}'.", icon=":material/check:")
                    for w in warnings:
                        st.warning(w)
                except BenchmarkError as exc:
                    st.error(str(exc))
        dl1, dl2 = st.columns(2)
        dl1.download_button("Export JSON", json.dumps(bench.to_dict(), indent=2), f"{bench_name}.json",
                            icon=":material/download:")
        dl2.download_button("Export CSV", benchmark_to_csv(bench), f"{bench_name}.csv", icon=":material/download:")

    with st.expander("Import a benchmark (JSON or CSV)", icon=":material/upload:"):
        up = st.file_uploader("Benchmark file", type=["json", "csv"], key="bench_upload")
        new_name = st.text_input("Name for the imported benchmark", key="bench_new_name")
        if st.button("Import", disabled=up is None):
            try:
                text = up.getvalue().decode("utf-8-sig")
                name = new_name.strip() or up.name.rsplit(".", 1)[0]
                b = benchmark_from_csv(text, name) if up.name.lower().endswith(".csv") else benchmark_from_dict(json.loads(text), name)
                b.name = name
                services.benchmarks.upsert(b.name, [i.to_dict() for i in b.items], b.description)
                st.success(f"Imported {len(b.items)} items as '{b.name}'.")
                for w in validate_items(b.items):
                    st.warning(w)
            except (BenchmarkError, ValueError, UnicodeDecodeError) as exc:
                st.error(f"Import failed: {exc}")


def show_experiment(exp_id: int, primary: list[str]) -> None:
    exp = services.experiments.get(exp_id)
    df = metrics_frame(exp["runs"])
    st.markdown(f"#### Results · experiment #{exp_id} · {exp['name']}")
    st.dataframe(df[["run"] + [c for c in primary if c in df]], hide_index=True,
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in primary})
    chart_cols = st.columns(2)
    for i, metric in enumerate([m for m in primary[:4] if m in df]):
        chart = metric_bar_chart(df, metric, color="strategy" if "prompt_variant" not in df else "prompt_variant")
        if chart is not None:
            chart_cols[i % 2].altair_chart(chart)
    csv_path, json_path = services.runner(cfg).export(exp_id)
    c1, c2 = st.columns(2)
    c1.download_button("Per-query results (CSV)", csv_path.read_bytes(), csv_path.name, icon=":material/download:",
                       key=f"csv{exp_id}")
    c2.download_button("Experiment metadata (JSON)", json_path.read_bytes(), json_path.name, icon=":material/download:",
                       key=f"json{exp_id}")
    st.caption("Exports are also written to outputs/experiments/.")


# =============================================================== retrieval
with tab_ret:
    st.markdown(
        "Evaluates retrieval only (no LLM). For each configuration every benchmark question is run and scored against "
        "its labels: **Precision@k, Recall@k, Hit rate@k, MRR, nDCG@k**, latency, and how the evidence gate would "
        "treat answerable vs. unanswerable questions."
    )
    with st.form("ret_grid"):
        c1, c2 = st.columns(2)
        strategies = c1.pills("Strategies", list(STRATEGY_LABELS), selection_mode="multi",
                              default=["dense", "keyword", "hybrid", "rrf"])
        top_ks = c2.pills("Top-k values", [1, 3, 5, 10], selection_mode="multi", default=[5])
        c3, c4 = st.columns(2)
        chunk_cfgs = c3.multiselect("Chunk size / overlap", ["400/60", "600/100", "800/0", "800/120", "1200/200"],
                                    default=[f"{cfg.chunk_size}/{cfg.chunk_overlap}"])
        models = c4.multiselect("Embedding models", embedding_model_options(), default=[cfg.embedding_model],
                                format_func=model_short)
        exp_name = st.text_input("Experiment name (optional)")
        go = st.form_submit_button("Run retrieval evaluation", type="primary", icon=":material/play_arrow:")
    if go:
        if bench is None or services.kb.stats()["documents"] == 0:
            st.error("Need a benchmark and at least one indexed document.")
        elif not (strategies and top_ks and chunk_cfgs and models):
            st.error("Select at least one value in every field.")
        else:
            grid = build_grid(strategies, sorted(top_ks), [tuple(int(x) for x in c.split("/")) for c in chunk_cfgs],
                              models, cfg.hybrid_alpha)
            bar = st.progress(0.0, "Starting…")
            try:
                exp_id = services.runner(cfg).run_retrieval_experiment(
                    bench, grid, exp_name, progress=lambda f, m: bar.progress(min(f, 1.0), m))
                st.session_state.last_ret_exp = exp_id
            except Exception as exc:
                st.error(f"Experiment failed: {exc}")
            bar.empty()
    if st.session_state.get("last_ret_exp"):
        show_experiment(st.session_state.last_ret_exp,
                        ["precision@k", "recall@k", "hit_rate@k", "mrr", "ndcg@k", "doc_recall@k",
                         "gate_pass_answerable", "gate_block_unanswerable", "retrieval_latency_ms_median", "n_chunks"])

# =============================================================== answers
with tab_ans:
    status = current_provider_status(cfg)
    st.markdown(
        "Runs the full RAG pipeline on each question with the **current sidebar retrieval settings** and scores the "
        "answers: citation validity and coverage, cited-evidence precision, key-fact recall, token F1, lexical "
        "faithfulness, abstention on unanswerable questions, latency and tokens."
    )
    if not status.available:
        st.warning(f"Answer evaluation needs an LLM. {status.message} Retrieval evaluation still works.",
                   icon=":material/cloud_off:")
    with st.form("ans_form"):
        variants = st.pills("Prompt variants", list(PROMPT_VARIANTS), selection_mode="multi", default=["strict"])
        limit = st.slider("Questions to evaluate", 1, len(bench.items) if bench else 1, len(bench.items) if bench else 1,
                          help="Local LLMs on CPU take roughly 5–30 s per answer.")
        use_judge = st.toggle(
            "Also run LLM-as-judge (same local model)",
            help="Asks the model to rate faithfulness and correctness 1–5. Model-graded scores are estimates, "
                 "and a judge grading its own answers is biased; treat them as a secondary signal.")
        exp_name = st.text_input("Experiment name (optional)", key="ans_name")
        st.caption(f"Retrieval: {current_settings().label()} · LLM: {cfg.llm_model} · temperature {cfg.temperature}")
        go_ans = st.form_submit_button("Run answer evaluation", type="primary", icon=":material/play_arrow:",
                                       disabled=not status.available)
    if go_ans and bench is not None:
        if not variants:
            st.error("Pick at least one prompt variant.")
        else:
            bar = st.progress(0.0, "Starting…")
            provider = services.provider(cfg)
            try:
                exp_id = services.runner(cfg).run_answer_experiment(
                    bench, provider, current_settings(), variants, judge=services.provider(cfg) if use_judge else None,
                    name=exp_name, progress=lambda f, m: bar.progress(min(f, 1.0), m), limit=limit)
                st.session_state.last_ans_exp = exp_id
            except Exception as exc:
                st.error(f"Experiment failed: {exc}")
            bar.empty()
    if st.session_state.get("last_ans_exp"):
        show_experiment(st.session_state.last_ans_exp,
                        ["answer_rate_answerable", "correct_abstention_unanswerable", "citation_validity",
                         "citation_coverage", "cited_evidence_precision", "key_fact_recall", "token_f1",
                         "faithfulness_lexical", "context_relevance", "judge_faithfulness", "judge_correctness",
                         "generation_s_median", "end_to_end_s_median", "prompt_tokens_mean", "completion_tokens_mean"])
