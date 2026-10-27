import pandas as pd
import streamlit as st

from src.ui.components import get_services, metric_bar_chart, metrics_frame

services = get_services()
repo = services.experiments

st.title("Experiment history", anchor=False)
st.caption("Every run is stored in SQLite with its configuration, environment and per-question results, so it can be reopened after a restart.")

exps = repo.list()
if not exps:
    st.info("No experiments yet. Run one in the Evaluation lab, or `python scripts/run_experiments.py`.", icon=":material/history:")
    st.stop()

overview = pd.DataFrame(
    [
        {
            "id": e["id"],
            "name": e["name"],
            "kind": e["kind"],
            "benchmark": e["benchmark"],
            "runs": e["run_count"],
            "status": e["status"],
            "created (UTC)": e["created_at"][:19].replace("T", " "),
            "LLM": e["env"].get("llm_model") or "—",
            "embedding models": ", ".join(m.split("/")[-1] for m in e["env"].get("embedding_models", [])) or "—",
        }
        for e in exps
    ]
)
event = st.dataframe(overview, hide_index=True, on_select="rerun", selection_mode="single-row", key="exp_table")

tab_detail, tab_compare = st.tabs([":material/visibility: Experiment details", ":material/compare: Compare all runs"])

with tab_detail:
    rows = event.selection.rows
    if not rows:
        st.caption("Select an experiment above to see its details.")
    else:
        exp = repo.get(int(overview.iloc[rows[0]]["id"]))
        st.markdown(f"#### #{exp['id']} · {exp['name']}")
        meta = st.columns(4)
        meta[0].metric("Kind", exp["kind"])
        meta[1].metric("Runs", len(exp["runs"]))
        meta[2].metric("Status", exp["status"])
        meta[3].metric("Finished (UTC)", (exp["finished_at"] or "—")[:19].replace("T", " "))
        if exp["notes"]:
            st.caption(exp["notes"])
        with st.expander("Environment & model versions", icon=":material/memory:"):
            st.json(exp["env"])
        df = metrics_frame(exp["runs"])
        st.dataframe(df, hide_index=True)
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ("top_k", "chunk_size", "chunk_overlap", "alpha", "k")]
        if numeric:
            metric = st.selectbox("Chart metric", numeric, key="hist_metric")
            color = "prompt_variant" if "prompt_variant" in df else "strategy"
            chart = metric_bar_chart(df, metric, color=color)
            if chart is not None:
                st.altair_chart(chart)
        run_label = st.selectbox("Per-question results for run", [r["label"] for r in exp["runs"]], key="hist_run")
        run = next(r for r in exp["runs"] if r["label"] == run_label)
        per_q = pd.DataFrame(repo.run_results(run["id"]))
        st.dataframe(per_q, hide_index=True)
        csv_path, json_path = services.runner().export(exp["id"])
        c1, c2, c3 = st.columns(3)
        c1.download_button("Results CSV", csv_path.read_bytes(), csv_path.name, icon=":material/download:")
        c2.download_button("Metadata JSON", json_path.read_bytes(), json_path.name, icon=":material/download:")
        with c3:
            if st.checkbox("Confirm delete", key=f"del_confirm_{exp['id']}"):
                if st.button("Delete experiment", icon=":material/delete:", type="primary"):
                    repo.delete(exp["id"])
                    st.rerun()

with tab_compare:
    runs = repo.all_runs()
    kind = st.segmented_control("Experiment kind", ["retrieval", "answer"], default="retrieval", key="cmp_kind")
    selected = [r for r in runs if r["kind"] == kind]
    if not selected:
        st.caption(f"No {kind} runs yet.")
    else:
        df = pd.DataFrame(
            [{"run": f"#{r['experiment_id']} {r['label']}", "experiment": r["experiment_id"],
              "created": r["created_at"][:19], **r["config"], **r["metrics"]} for r in selected]
        )
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                   and c not in ("experiment", "top_k", "chunk_size", "chunk_overlap", "alpha", "candidate_pool", "k")]
        default_metric = "mrr" if "mrr" in numeric else numeric[0]
        metric = st.selectbox("Metric", numeric, index=numeric.index(default_metric), key="cmp_metric")
        color = st.segmented_control("Colour by", [c for c in ("strategy", "embedding_model", "chunk_size", "prompt_variant") if c in df],
                                     default="strategy" if "strategy" in df else None, key="cmp_color")
        top = df.dropna(subset=[metric]).sort_values(metric, ascending=False).head(25)
        if color in top:
            top[color] = top[color].astype(str)
        chart = metric_bar_chart(top, metric, color=color or "strategy", title=f"Top runs by {metric}")
        if chart is not None:
            st.altair_chart(chart)
        st.dataframe(top, hide_index=True)
        st.download_button("Download comparison (CSV)", df.to_csv(index=False), f"{kind}_runs_comparison.csv",
                           icon=":material/download:")
