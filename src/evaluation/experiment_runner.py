"""Run benchmark experiments, persist every result in SQLite and export them.

Two experiment kinds:

* ``retrieval`` -- a grid of retrieval settings. Needs only the embedding
  model, so it always works without an LLM. Also measures how the
  evidence-sufficiency gate would behave on answerable vs. unanswerable items.
* ``answer`` -- full RAG over the benchmark for one or more prompt variants,
  with deterministic answer metrics and an optional LLM judge.

All numbers are computed from actual runs; nothing is hard-coded. Results are
specific to the benchmark and corpus used.
"""

from __future__ import annotations

import csv
import itertools
import json
import logging
import platform
import statistics
from dataclasses import replace
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

from src.config import AppConfig, EMBEDDING_MODELS
from src.embeddings.embedder import resolve_device
from src.evaluation.answer_metrics import (
    JUDGE_PROMPT_VERSION,
    cited_evidence_precision,
    context_relevance,
    key_fact_recall,
    lexical_faithfulness,
    llm_judge,
    token_f1,
)
from src.evaluation.benchmark import Benchmark
from src.evaluation.retrieval_metrics import score_retrieval
from src.generation.llm_provider import LLMProvider
from src.generation.rag_pipeline import RAGPipeline
from src.knowledge_base import KnowledgeBase, RetrievalSettings
from src.persistence.repositories import ExperimentRepository

logger = logging.getLogger(__name__)
Progress = Callable[[float, str], None]


def _noop(_: float, __: str) -> None:
    pass


def build_grid(
    strategies: list[str],
    top_ks: list[int],
    chunk_configs: list[tuple[int, int]],
    models: list[str],
    alpha: float = 0.6,
    candidate_pool: int = 30,
) -> list[RetrievalSettings]:
    return [
        RetrievalSettings(s, k, cs, ov, m, alpha, max(candidate_pool, k))
        for m, (cs, ov), s, k in itertools.product(models, chunk_configs, strategies, top_ks)
    ]


def environment_info(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    def ver(pkg: str) -> str:
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return "not installed"

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": resolve_device("auto"),
        "packages": {p: ver(p) for p in ("sentence-transformers", "faiss-cpu", "torch", "numpy", "streamlit", "PyMuPDF")},
    }
    if extra:
        info.update(extra)
    return info


def _mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return float(statistics.fmean(vals)) if vals else None


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, round(q * (len(vals) - 1))))
    return vals[idx]


def gate_floor(cfg: AppConfig, model: str) -> float:
    if cfg.min_relevance is not None:
        return cfg.min_relevance
    spec = EMBEDDING_MODELS.get(model)
    return spec.default_min_relevance if spec else 0.35


class ExperimentRunner:
    def __init__(self, kb: KnowledgeBase, cfg: AppConfig, repo: ExperimentRepository) -> None:
        self.kb = kb
        self.cfg = cfg
        self.repo = repo

    # ------------------------------------------------------------ retrieval
    def evaluate_retrieval(self, bench: Benchmark, settings: RetrievalSettings) -> tuple[dict, list[dict]]:
        k = settings.top_k
        floor = gate_floor(self.cfg, settings.embedding_model)
        _, build_s = self.kb.get_index(settings)
        rows: list[dict] = []
        for item in bench.items:
            res = self.kb.retrieve(item.question, settings)
            row: dict[str, Any] = {
                "qid": item.qid,
                "question": item.question,
                "answerable": item.answerable,
                "category": item.category,
                "retrieved_chunk_ids": [c.chunk_id for c in res.chunks],
                "retrieved_files": [c.filename for c in res.chunks],
                "max_dense_score": res.max_dense_score,
                "gate_pass": res.max_dense_score >= floor,
                "retrieval_latency_ms": res.retrieval_s * 1000,
                "query_embedding_ms": res.query_embedding_s * 1000,
                "n_retrieved": len(res.chunks),
            }
            if item.answerable and item.has_relevance_labels:
                scores = score_retrieval(item, res.chunks, k)
                row.update(
                    precision=scores.precision,
                    recall=scores.recall,
                    hit=scores.hit,
                    rr=scores.reciprocal_rank,
                    ndcg=scores.ndcg,
                    doc_recall=scores.doc_recall,
                    relevance=scores.relevance,
                    label_type=scores.label_type,
                )
            rows.append(row)

        scored = [r for r in rows if "precision" in r]
        answerable = [r for r in rows if r["answerable"]]
        unanswerable = [r for r in rows if not r["answerable"]]
        latencies = [r["retrieval_latency_ms"] for r in rows]
        metrics = {
            "k": k,
            "n_items": len(rows),
            "n_scored": len(scored),
            "precision@k": _mean([r["precision"] for r in scored]),
            "recall@k": _mean([r["recall"] for r in scored]),
            "hit_rate@k": _mean([r["hit"] for r in scored]),
            "mrr": _mean([r["rr"] for r in scored]),
            "ndcg@k": _mean([r["ndcg"] for r in scored]),
            "doc_recall@k": _mean([r["doc_recall"] for r in scored]),
            "gate_floor": floor,
            "gate_pass_answerable": _mean([float(r["gate_pass"]) for r in answerable]),
            "gate_block_unanswerable": _mean([float(not r["gate_pass"]) for r in unanswerable]),
            "retrieval_latency_ms_mean": _mean(latencies),
            "retrieval_latency_ms_median": float(statistics.median(latencies)) if latencies else None,
            "retrieval_latency_ms_p95": _pct(latencies, 0.95),
            "index_build_s": build_s,
            "n_chunks": len(self.kb.get_index(settings)[0].chunks),
        }
        return metrics, rows

    def run_retrieval_experiment(
        self, bench: Benchmark, grid: list[RetrievalSettings], name: str = "", progress: Progress = _noop, notes: str = ""
    ) -> int:
        if not grid:
            raise ValueError("The experiment grid is empty.")
        if self.kb.stats()["documents"] == 0:
            raise ValueError("The knowledge base is empty; load documents before running an experiment.")
        exp_id = self.repo.create(
            name or f"Retrieval grid ({len(grid)} configs)",
            "retrieval",
            bench.name,
            [s.__dict__ for s in grid],
            environment_info({"embedding_models": sorted({s.embedding_model for s in grid})}),
            notes,
        )
        try:
            for i, settings in enumerate(grid):
                progress(i / len(grid), f"Evaluating {settings.label()}")
                metrics, rows = self.evaluate_retrieval(bench, settings)
                self.repo.add_run(exp_id, settings.label(), settings.__dict__, metrics, rows)
            progress(1.0, "Done")
            self.repo.finish(exp_id, "completed")
        except Exception as exc:
            self.repo.finish(exp_id, "failed", notes=f"{notes} | failed: {exc}".strip(" |"))
            raise
        return exp_id

    # --------------------------------------------------------------- answers
    def run_answer_experiment(
        self,
        bench: Benchmark,
        provider: LLMProvider,
        settings: RetrievalSettings,
        prompt_variants: list[str],
        judge: LLMProvider | None = None,
        name: str = "",
        progress: Progress = _noop,
        notes: str = "",
        limit: int | None = None,
    ) -> int:
        status = provider.status()
        if not status.available:
            raise RuntimeError(f"Answer evaluation needs an available LLM: {status.message}")
        if judge is not None and not judge.status().available:
            judge = None
        items = bench.items[:limit] if limit else bench.items
        env = environment_info(
            {
                "llm_provider": provider.name,
                "llm_model": provider.model,
                "llm_model_digest": _model_digest(provider),
                "judge_model": judge.model if judge else None,
                "judge_prompt_version": JUDGE_PROMPT_VERSION if judge else None,
                "temperature": self.cfg.temperature,
                "evidence_floor": gate_floor(self.cfg, settings.embedding_model),
            }
        )
        grid = [{**settings.__dict__, "prompt_variant": v} for v in prompt_variants]
        exp_id = self.repo.create(name or f"Answer eval · {provider.model}", "answer", bench.name, grid, env, notes)
        total = len(items) * len(prompt_variants)
        done = 0
        try:
            for variant in prompt_variants:
                cfg = replace(
                    self.cfg,
                    prompt_variant=variant,
                    retrieval_strategy=settings.strategy,
                    top_k=settings.top_k,
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                    embedding_model=settings.embedding_model,
                    hybrid_alpha=settings.alpha,
                )
                pipeline = RAGPipeline(self.kb, provider, cfg)
                rows = []
                for item in items:
                    progress(done / total, f"[{variant}] {item.qid}: generating answer")
                    rows.append(self._answer_row(pipeline, item, settings, judge))
                    done += 1
                metrics = self._aggregate_answers(rows)
                label = f"{variant} · {provider.model} · {settings.label()}"
                self.repo.add_run(exp_id, label, {**settings.__dict__, "prompt_variant": variant,
                                                  "llm_model": provider.model}, metrics, rows)
            progress(1.0, "Done")
            self.repo.finish(exp_id, "completed")
        except Exception as exc:
            self.repo.finish(exp_id, "failed", notes=f"{notes} | failed: {exc}".strip(" |"))
            raise
        return exp_id

    def _answer_row(self, pipeline: RAGPipeline, item, settings: RetrievalSettings, judge: LLMProvider | None) -> dict:
        result = pipeline.answer(item.question, settings, log=False)
        answer = result.answer or ""
        abstained = result.status in ("insufficient_evidence", "llm_abstained")
        n_labels = len(result.citations) + len(result.invalid_citations)
        row: dict[str, Any] = {
            "qid": item.qid,
            "question": item.question,
            "answerable": item.answerable,
            "status": result.status,
            "answer": answer,
            "expected_answer": item.expected_answer,
            "abstained": abstained,
            "n_citations": len(result.citations),
            "n_invalid_citations": len(result.invalid_citations),
            "citation_validity": (len(result.citations) / n_labels) if n_labels else None,
            "has_valid_citation": bool(result.citations),
            "cited_chunk_ids": [c.chunk_id for c in result.citations],
            "context_relevance": context_relevance(result.retrieved),
            "retrieval_latency_ms": result.timings["retrieval_s"] * 1000,
            "generation_s": result.timings["generation_s"],
            "total_s": result.timings["total_s"],
            "prompt_tokens": result.usage.get("prompt_tokens"),
            "completion_tokens": result.usage.get("completion_tokens"),
            "estimated_cost": result.usage.get("estimated_cost"),
        }
        if result.status in ("llm_error", "llm_unavailable"):
            row["error"] = result.message
        if item.answerable and answer and not abstained:
            row.update(
                key_fact_recall=key_fact_recall(answer, item.key_facts),
                token_f1=token_f1(answer, item.expected_answer) if item.expected_answer else None,
                faithfulness_lexical=lexical_faithfulness(answer, result.retrieved),
                cited_evidence_precision=cited_evidence_precision(row["cited_chunk_ids"], result.retrieved, item),
            )
            if judge is not None:
                js = llm_judge(judge, item.question, item.expected_answer, answer, result.retrieved)
                row.update(judge_faithfulness=js.faithfulness, judge_correctness=js.correctness,
                           judge_rationale=js.rationale, judge_error=js.error)
        return row

    @staticmethod
    def _aggregate_answers(rows: list[dict]) -> dict:
        ans = [r for r in rows if r["answerable"]]
        unans = [r for r in rows if not r["answerable"]]
        produced = [r for r in rows if r["answer"] and not r["abstained"]]
        with_labels = [r for r in produced if r["citation_validity"] is not None]
        gens = [r["generation_s"] for r in rows if r["generation_s"]]
        totals = [r["total_s"] for r in rows]
        prompt_toks = [r["prompt_tokens"] for r in rows if r.get("prompt_tokens") is not None]
        comp_toks = [r["completion_tokens"] for r in rows if r.get("completion_tokens") is not None]
        costs = [r["estimated_cost"] for r in rows if r.get("estimated_cost") is not None]
        return {
            "n_items": len(rows),
            "n_answers": len(produced),
            "n_errors": sum(1 for r in rows if r.get("error")),
            "answer_rate_answerable": _mean([float(not r["abstained"] and bool(r["answer"])) for r in ans]),
            "correct_abstention_unanswerable": _mean([float(r["abstained"]) for r in unans]),
            "citation_validity": _mean([r["citation_validity"] for r in with_labels]),
            "citation_coverage": _mean([float(r["has_valid_citation"]) for r in produced]),
            "cited_evidence_precision": _mean([r.get("cited_evidence_precision") for r in produced]),
            "key_fact_recall": _mean([r.get("key_fact_recall") for r in produced]),
            "token_f1": _mean([r.get("token_f1") for r in produced]),
            "faithfulness_lexical": _mean([r.get("faithfulness_lexical") for r in produced]),
            "context_relevance": _mean([r["context_relevance"] for r in rows]),
            "judge_faithfulness": _mean([r.get("judge_faithfulness") for r in produced]),
            "judge_correctness": _mean([r.get("judge_correctness") for r in produced]),
            "generation_s_mean": _mean(gens),
            "generation_s_median": float(statistics.median(gens)) if gens else None,
            "end_to_end_s_mean": _mean(totals),
            "end_to_end_s_median": float(statistics.median(totals)) if totals else None,
            "end_to_end_s_p95": _pct(totals, 0.95),
            "prompt_tokens_mean": _mean(prompt_toks),
            "completion_tokens_mean": _mean(comp_toks),
            "estimated_cost_total": sum(costs) if costs else None,
        }

    # ---------------------------------------------------------------- export
    def export(self, experiment_id: int, out_dir: Path | None = None) -> tuple[Path, Path]:
        """Write per-query results (CSV) and experiment metadata + aggregates (JSON)."""
        exp = self.repo.get(experiment_id)
        if exp is None:
            raise ValueError(f"Experiment {experiment_id} does not exist.")
        out_dir = out_dir or (self.cfg.outputs_dir / "experiments")
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"experiment_{experiment_id}_results.csv"
        json_path = out_dir / f"experiment_{experiment_id}_metadata.json"
        rows = export_rows(self.repo, exp)
        fields: list[str] = []
        for r in rows:
            for key in r:
                if key not in fields:
                    fields.append(key)
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})
        json_path.write_text(json.dumps(exp, indent=2, default=str), encoding="utf-8")
        return csv_path, json_path


def export_rows(repo: ExperimentRepository, exp: dict) -> list[dict]:
    rows = []
    for run in exp["runs"]:
        for r in repo.run_results(run["id"]):
            rows.append({"experiment_id": exp["id"], "run_id": run["id"], "run_label": run["label"], **r})
    return rows


def _model_digest(provider: LLMProvider) -> str | None:
    if provider.name != "ollama":
        return None
    try:
        resp = provider.session.get(f"{provider.base_url}/api/tags", timeout=3)  # type: ignore[attr-defined]
        for m in resp.json().get("models", []):
            if m.get("name") == provider.model:
                return m.get("digest", "")[:12]
    except Exception:
        return None
    return None

