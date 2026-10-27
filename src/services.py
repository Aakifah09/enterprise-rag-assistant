"""Composition root: builds the object graph used by the UI and CLI scripts."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.config import PROJECT_ROOT, AppConfig
from src.evaluation.benchmark import Benchmark, load_benchmark_file
from src.evaluation.experiment_runner import ExperimentRunner
from src.generation.llm_provider import LLMProvider
from src.generation.provider_factory import build_provider
from src.generation.rag_pipeline import RAGPipeline
from src.knowledge_base import KnowledgeBase
from src.persistence.database import Database
from src.persistence.repositories import BenchmarkRepository, ExperimentRepository, QueryLogRepository
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)

SAMPLE_DOCS_DIR = PROJECT_ROOT / "sample_data" / "documents"
SAMPLE_BENCHMARK = PROJECT_ROOT / "sample_data" / "benchmark" / "sample_benchmark.json"
SECURITY_DEMO_DIR = PROJECT_ROOT / "sample_data" / "security_demo"


@dataclass
class Services:
    cfg: AppConfig
    db: Database
    kb: KnowledgeBase
    benchmarks: BenchmarkRepository
    experiments: ExperimentRepository
    query_log: QueryLogRepository

    def provider(self, cfg: AppConfig | None = None) -> LLMProvider:
        return build_provider(cfg or self.cfg)

    def pipeline(self, cfg: AppConfig | None = None, provider: LLMProvider | None = None) -> RAGPipeline:
        cfg = cfg or self.cfg
        return RAGPipeline(self.kb, provider or build_provider(cfg), cfg, self.query_log)

    def runner(self, cfg: AppConfig | None = None) -> ExperimentRunner:
        return ExperimentRunner(self.kb, cfg or self.cfg, self.experiments)

    def load_benchmark(self, name: str) -> Benchmark:
        from src.evaluation.benchmark import benchmark_from_dict

        stored = self.benchmarks.get(name)
        if stored is None:
            raise KeyError(f"Benchmark '{name}' not found")
        return benchmark_from_dict(stored, default_name=name)


def ensure_sample_benchmark(repo: BenchmarkRepository, path: Path = SAMPLE_BENCHMARK) -> None:
    """Register the bundled sample benchmark the first time the app starts."""
    if not path.exists():
        return
    bench = load_benchmark_file(path)
    if repo.get(bench.name) is None:
        repo.upsert(bench.name, [i.to_dict() for i in bench.items], bench.description, is_sample=True)


def build_services(cfg: AppConfig | None = None) -> Services:
    cfg = cfg or AppConfig.from_env()
    setup_logging(cfg.outputs_dir / "logs")
    db = Database(cfg.db_path)
    services = Services(
        cfg=cfg,
        db=db,
        kb=KnowledgeBase(cfg, db),
        benchmarks=BenchmarkRepository(db),
        experiments=ExperimentRepository(db),
        query_log=QueryLogRepository(db),
    )
    ensure_sample_benchmark(services.benchmarks)
    return services
