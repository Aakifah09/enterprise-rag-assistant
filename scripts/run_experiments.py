"""Run benchmark experiments from the command line and print a results table.

Results are stored in the app's SQLite database (visible in Experiment history)
and exported to outputs/experiments/.

    python scripts/run_experiments.py retrieval            # full retrieval grid (no LLM needed)
    python scripts/run_experiments.py answer --limit 22    # RAG answers with Ollama, strict vs baseline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252

from src.config import EMBEDDING_MODELS  # noqa: E402
from src.evaluation.experiment_runner import build_grid  # noqa: E402
from src.knowledge_base import RetrievalSettings  # noqa: E402
from src.services import SAMPLE_DOCS_DIR, build_services  # noqa: E402


def table(rows: list[dict], cols: list[str]) -> str:
    def f(v):
        return f"{v:.3f}" if isinstance(v, float) else ("—" if v is None else str(v))

    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(f(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["retrieval", "answer"])
    ap.add_argument("--benchmark", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--variants", default="strict,baseline")
    ap.add_argument("--strategy", default="hybrid")
    ap.add_argument("--judge", action="store_true", help="also run LLM-as-judge with the same model")
    args = ap.parse_args()

    s = build_services()
    if s.kb.stats()["documents"] == 0:
        print("Knowledge base empty -> loading sample documents")
        s.kb.load_sample_data(SAMPLE_DOCS_DIR)
    bench = s.load_benchmark(args.benchmark or s.benchmarks.names()[0])
    runner = s.runner()

    def progress(f: float, msg: str) -> None:
        print(f"  [{f:5.0%}] {msg}", flush=True)

    if args.kind == "retrieval":
        grid = build_grid(["dense", "keyword", "hybrid", "rrf"], [5], [(800, 120)], list(EMBEDDING_MODELS))
        grid += build_grid(["dense", "hybrid"], [5], [(400, 60), (800, 0), (1200, 200)], ["sentence-transformers/all-MiniLM-L6-v2"])
        grid += build_grid(["hybrid"], [1, 3, 10], [(800, 120)], ["sentence-transformers/all-MiniLM-L6-v2"])
        exp_id = runner.run_retrieval_experiment(bench, grid, "CLI retrieval grid", progress)
        cols = ["run", "precision@k", "recall@k", "hit_rate@k", "mrr", "ndcg@k", "gate_pass_answerable",
                "gate_block_unanswerable", "retrieval_latency_ms_median", "n_chunks"]
    else:
        provider = s.provider()
        st = provider.status()
        if not st.available:
            print(f"LLM unavailable: {st.message}")
            return 1
        settings = RetrievalSettings.from_config(s.cfg)
        settings = RetrievalSettings(args.strategy, settings.top_k, settings.chunk_size, settings.chunk_overlap,
                                     settings.embedding_model, settings.alpha, settings.candidate_pool)
        exp_id = runner.run_answer_experiment(
            bench, provider, settings, args.variants.split(","), judge=s.provider() if args.judge else None,
            name=f"CLI answer eval · {provider.model}", progress=progress, limit=args.limit)
        cols = ["run", "answer_rate_answerable", "correct_abstention_unanswerable", "citation_validity",
                "citation_coverage", "cited_evidence_precision", "key_fact_recall", "token_f1",
                "faithfulness_lexical", "judge_faithfulness", "judge_correctness", "generation_s_median", "end_to_end_s_median",
                "prompt_tokens_mean", "completion_tokens_mean"]

    exp = s.experiments.get(exp_id)
    rows = [{"run": r["label"], **r["metrics"]} for r in exp["runs"]]
    csv_path, json_path = runner.export(exp_id)
    print(f"\nExperiment #{exp_id} ({exp['benchmark']}, {len(bench.items)} items)\n")
    print(table(rows, cols))
    print(f"\nExported: {csv_path}\n          {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
