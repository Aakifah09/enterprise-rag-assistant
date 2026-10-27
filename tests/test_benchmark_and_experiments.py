import csv
import json

import pytest

from src.config import PROJECT_ROOT
from src.evaluation.benchmark import (
    Benchmark,
    BenchmarkError,
    BenchmarkItem,
    benchmark_from_csv,
    benchmark_from_dict,
    benchmark_to_csv,
    load_benchmark_file,
    validate_items,
)
from src.evaluation.experiment_runner import ExperimentRunner, build_grid
from src.knowledge_base import RetrievalSettings
from src.persistence.database import Database
from src.persistence.repositories import BenchmarkRepository, ExperimentRepository
from tests.conftest import MockProvider

SAMPLE = PROJECT_ROOT / "sample_data" / "benchmark" / "sample_benchmark.json"


def small_bench() -> Benchmark:
    return Benchmark(
        "unit",
        [
            BenchmarkItem("Q1", "How many days of annual leave do employees receive?", "18 days", True,
                          ["leave_policy.md"], ["18 days of paid annual leave"], [], ["18 days"]),
            BenchmarkItem("Q2", "When must the CTO be notified about P1 incidents?", "Within 30 minutes", True,
                          ["it_policy.txt"], ["The CTO is notified within 30 minutes"], [], ["30 minutes"]),
            BenchmarkItem("U1", "What is the parental leave policy?", answerable=False),
        ],
    )


# --- benchmark format ------------------------------------------------------------
def test_sample_benchmark_loads_and_validates():
    bench = load_benchmark_file(SAMPLE)
    assert bench.is_sample
    assert len(bench.items) >= 20
    assert any(not i.answerable for i in bench.items)
    assert validate_items(bench.items) == []


def test_sample_benchmark_evidence_exists_in_sample_documents():
    from src.evaluation.retrieval_metrics import evidence_in_text
    from src.ingestion.loader import extract_document

    docs = {}
    for p in (PROJECT_ROOT / "sample_data" / "documents").iterdir():
        docs[p.name] = " ".join(u.text for u in extract_document(p.name, p.read_bytes()).units)
    for item in load_benchmark_file(SAMPLE).items:
        for doc in item.relevant_documents:
            assert doc in docs, f"{item.qid} references unknown document {doc}"
        for ev in item.evidence:
            assert any(evidence_in_text(ev, docs[d], ratio=1.0) for d in item.relevant_documents), (item.qid, ev)


def test_duplicate_qids_rejected():
    with pytest.raises(BenchmarkError, match="Duplicate"):
        benchmark_from_dict({"items": [{"qid": "a", "question": "x"}, {"qid": "a", "question": "y"}]})


def test_missing_question_rejected():
    with pytest.raises(BenchmarkError):
        benchmark_from_dict({"items": [{"qid": "a"}]})


def test_bad_structure_rejected():
    with pytest.raises(BenchmarkError):
        benchmark_from_dict({"questions": []})
    with pytest.raises(BenchmarkError):
        benchmark_from_dict({"items": []})


def test_soft_warnings_for_missing_labels():
    warnings = validate_items([BenchmarkItem("a", "question?")])
    assert any("no relevance labels" in w for w in warnings)


def test_csv_round_trip():
    bench = small_bench()
    again = benchmark_from_csv(benchmark_to_csv(bench), "unit")
    assert [i.to_dict() for i in again.items] == [i.to_dict() for i in bench.items]


def test_csv_requires_columns():
    with pytest.raises(BenchmarkError):
        benchmark_from_csv("foo,bar\n1,2\n", "x")


def test_benchmark_repository_round_trip(tmp_path):
    repo = BenchmarkRepository(Database(tmp_path / "b.db"))
    bench = small_bench()
    repo.upsert(bench.name, [i.to_dict() for i in bench.items], "desc")
    stored = benchmark_from_dict(repo.get("unit"))
    assert len(stored.items) == 3 and stored.items[0].key_facts == ["18 days"]
    assert repo.names() == ["unit"]


# --- experiments ----------------------------------------------------------------
def test_build_grid_is_cartesian():
    grid = build_grid(["dense", "hybrid"], [3, 5], [(200, 40), (400, 0)], ["m"])
    assert len(grid) == 8
    assert {g.strategy for g in grid} == {"dense", "hybrid"}


def test_retrieval_experiment_persists_and_exports(loaded_kb, cfg, tmp_path):
    repo = ExperimentRepository(loaded_kb.db)
    runner = ExperimentRunner(loaded_kb, cfg, repo)
    grid = build_grid(["dense", "keyword", "hybrid"], [3], [(200, 40)], ["test/hashing"])
    exp_id = runner.run_retrieval_experiment(small_bench(), grid, name="unit test")

    exp = repo.get(exp_id)
    assert exp["status"] == "completed" and exp["kind"] == "retrieval"
    assert len(exp["runs"]) == 3
    m = exp["runs"][0]["metrics"]
    assert m["n_items"] == 3 and m["n_scored"] == 2
    for key in ("precision@k", "recall@k", "hit_rate@k", "mrr", "retrieval_latency_ms_mean"):
        assert m[key] is not None and m[key] >= 0
    assert 0 <= m["hit_rate@k"] <= 1
    assert "packages" in exp["env"]

    csv_path, json_path = runner.export(exp_id, tmp_path / "exports")
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert len(rows) == 9  # 3 runs x 3 questions
    assert {"qid", "precision", "rr", "retrieval_latency_ms", "run_label"} <= set(rows[0])
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    assert meta["id"] == exp_id and len(meta["runs"]) == 3


def test_experiments_survive_restart(loaded_kb, cfg):
    runner = ExperimentRunner(loaded_kb, cfg, ExperimentRepository(loaded_kb.db))
    exp_id = runner.run_retrieval_experiment(small_bench(), build_grid(["dense"], [3], [(200, 40)], ["test/hashing"]))
    reopened = ExperimentRepository(Database(cfg.db_path))  # fresh connection = app restart
    exp = reopened.get(exp_id)
    assert exp is not None and exp["runs"]
    assert len(reopened.run_results(exp["runs"][0]["id"])) == 3
    assert reopened.all_runs()[0]["experiment_id"] == exp_id


def test_retrieval_experiment_requires_documents(kb, cfg):
    runner = ExperimentRunner(kb, cfg, ExperimentRepository(kb.db))
    with pytest.raises(ValueError, match="empty"):
        runner.run_retrieval_experiment(small_bench(), build_grid(["dense"], [3], [(200, 40)], ["test/hashing"]))


def test_answer_experiment_with_mock_llm(loaded_kb, cfg):
    repo = ExperimentRepository(loaded_kb.db)
    runner = ExperimentRunner(loaded_kb, cfg, repo)
    provider = MockProvider("Employees receive 18 days of paid annual leave [S1].")
    settings = RetrievalSettings("hybrid", 3, 200, 40, "test/hashing")
    exp_id = runner.run_answer_experiment(small_bench(), provider, settings, ["strict", "baseline"])
    exp = repo.get(exp_id)
    assert len(exp["runs"]) == 2
    m = exp["runs"][0]["metrics"]
    assert m["citation_validity"] == 1.0
    assert m["citation_coverage"] == 1.0
    assert m["key_fact_recall"] is not None
    assert m["judge_faithfulness"] is None  # no judge configured -> not reported
    assert exp["env"]["llm_model"] == "mock-model"


def test_answer_experiment_refuses_without_llm(loaded_kb, cfg):
    runner = ExperimentRunner(loaded_kb, cfg, ExperimentRepository(loaded_kb.db))
    with pytest.raises(RuntimeError, match="needs an available LLM"):
        runner.run_answer_experiment(small_bench(), MockProvider(available=False),
                                     RetrievalSettings(embedding_model="test/hashing"), ["strict"])
