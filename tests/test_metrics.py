"""Metric calculations checked against hand-computed expected values."""

import math

import pytest

from src.evaluation.answer_metrics import (
    _scale,
    cited_evidence_precision,
    context_relevance,
    key_fact_recall,
    lexical_faithfulness,
    llm_judge,
    normalize_answer,
    token_f1,
)
from src.evaluation.benchmark import BenchmarkItem
from src.evaluation.retrieval_metrics import (
    chunk_relevance,
    evidence_in_text,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
    score_retrieval,
)
from src.models import RetrievedChunk
from tests.conftest import MockProvider


def rc(cid: str, text: str, filename: str = "a.md", dense: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(cid, "d", filename, text, 1, dense, dense)


# --- primitive retrieval metrics -------------------------------------------------
def test_precision_at_k():
    assert precision_at_k([True, False, True, False], 4) == 0.5
    assert precision_at_k([True], 5) == 0.2  # denominator is k, not len(results)


def test_hit_at_k():
    assert hit_at_k([False, False, True], 2) == 0.0
    assert hit_at_k([False, False, True], 3) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank([False, False, True]) == pytest.approx(1 / 3)
    assert reciprocal_rank([True, True]) == 1.0
    assert reciprocal_rank([False, False]) == 0.0


def test_ndcg_known_value():
    # relevant at ranks 1 and 3, two relevant items exist
    expected = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k([True, False, True], 3, n_relevant=2) == pytest.approx(expected)
    assert ndcg_at_k([True, True], 2, 2) == pytest.approx(1.0)
    assert ndcg_at_k([False, False], 2, 1) == 0.0


# --- evidence matching ---------------------------------------------------------
def test_evidence_substring_match_ignores_case_and_whitespace():
    assert evidence_in_text("Leave  accrues\nmonthly", "Policy: leave accrues monthly for all staff.")


def test_evidence_partial_match_across_chunk_boundary():
    evidence = "P1 incidents get a first response within fifteen minutes around the clock"
    chunk_with_most = "...P1 incidents get a first response within fifteen minutes"  # ~80% of the passage
    chunk_with_little = "around the clock"
    assert evidence_in_text(evidence, chunk_with_most)
    assert not evidence_in_text(evidence, chunk_with_little)


def test_chunk_relevance_label_precedence():
    c = rc("x-1", "Employees get 18 days of leave.", "policy.md")
    by_ids = BenchmarkItem("q", "?", relevant_chunk_ids=["other"], evidence=["18 days of leave"])
    assert not chunk_relevance(c, by_ids)  # chunk ids take precedence
    by_evidence = BenchmarkItem("q", "?", evidence=["18 days of leave"], relevant_documents=["zzz.md"])
    assert chunk_relevance(c, by_evidence)
    by_doc = BenchmarkItem("q", "?", relevant_documents=["policy.md"])
    assert chunk_relevance(c, by_doc)


def test_score_retrieval_with_evidence_labels():
    item = BenchmarkItem(
        "Q", "?", evidence=["probation lasts three months", "extended once by three months"],
        relevant_documents=["hr.md"],
    )
    retrieved = [
        rc("1", "unrelated text about laptops", "it.pdf"),
        rc("2", "Probation lasts three months for everyone.", "hr.md"),
        rc("3", "nothing here either", "it.pdf"),
        rc("4", "It can be extended once by three months.", "hr.md"),
    ]
    s = score_retrieval(item, retrieved, k=4)
    assert s.relevance == [False, True, False, True]
    assert s.precision == 0.5
    assert s.recall == 1.0
    assert s.reciprocal_rank == 0.5
    assert s.hit == 1.0
    assert s.doc_recall == 1.0
    s2 = score_retrieval(item, retrieved, k=2)
    assert s2.recall == 0.5 and s2.precision == 0.5


def test_score_retrieval_with_chunk_id_labels():
    item = BenchmarkItem("Q", "?", relevant_chunk_ids=["b", "z"])
    s = score_retrieval(item, [rc("a", "x"), rc("b", "y")], k=2)
    assert s.recall == 0.5 and s.precision == 0.5 and s.reciprocal_rank == 0.5 and s.label_type == "chunk_ids"


def test_score_retrieval_no_relevant_results():
    item = BenchmarkItem("Q", "?", evidence=["absent passage text"])
    s = score_retrieval(item, [rc("a", "x"), rc("b", "y")], k=2)
    assert (s.precision, s.recall, s.hit, s.reciprocal_rank, s.ndcg) == (0, 0, 0, 0, 0)


# --- answer metrics --------------------------------------------------------------
def test_normalize_answer():
    assert normalize_answer("The fee is USD 1,200.") == "fee is usd 1200"
    assert normalize_answer("Version 4.2 applies") == "version 4.2 applies"


def test_token_f1_known_value():
    # pred tokens: 18 days leave ; ref tokens: employees get 18 days
    assert token_f1("18 days leave [S1]", "employees get 18 days") == pytest.approx(2 * (2 / 3) * (2 / 4) / ((2 / 3) + (2 / 4)))
    assert token_f1("", "x") == 0.0
    assert token_f1("totally different", "nothing shared") == 0.0


def test_key_fact_recall():
    assert key_fact_recall("You get 18 days and must use them by 31 March [S1].", ["18 days", "31 March"]) == 1.0
    assert key_fact_recall("You get 18 days.", ["18 days", "31 March"]) == 0.5
    assert key_fact_recall("The budget is SGD 1200", ["SGD 1,200"]) == 1.0
    assert key_fact_recall("anything", []) is None


def test_lexical_faithfulness_supported_and_unsupported_sentences():
    sources = [rc("1", "Employees receive 18 days of paid annual leave per calendar year.")]
    answer = (
        "Employees receive 18 days of paid annual leave per year [S1]. "
        "They also receive a company car and free gym membership [S1]."
    )
    assert lexical_faithfulness(answer, sources) == 0.5
    assert lexical_faithfulness("Employees receive 18 days annual leave [S1].", sources) == 1.0
    assert lexical_faithfulness("ok", sources) is None
    assert lexical_faithfulness("anything", []) is None


def test_context_relevance_is_mean_cosine():
    assert context_relevance([rc("1", "a", dense=0.2), rc("2", "b", dense=0.6)]) == pytest.approx(0.4)
    assert context_relevance([]) is None


def test_cited_evidence_precision():
    item = BenchmarkItem("Q", "?", evidence=["18 days"])
    sources = [rc("1", "We give 18 days."), rc("2", "Unrelated.")]
    assert cited_evidence_precision(["1", "2"], sources, item) == 0.5
    assert cited_evidence_precision([], sources, item) is None


def test_llm_judge_parses_json_and_scales():
    judge = MockProvider('Sure: {"faithfulness": 5, "correctness": 3, "rationale": "ok"}')
    score = llm_judge(judge, "q", "ref", "ans", [rc("1", "ctx")])
    assert score.faithfulness == 1.0 and score.correctness == 0.5 and score.error is None
    assert score.judge_model == "mock-model"


def test_llm_judge_unparseable_output_is_reported_not_guessed():
    score = llm_judge(MockProvider("I think it's good"), "q", "ref", "ans", [rc("1", "ctx")])
    assert score.faithfulness is None and score.correctness is None and score.error


def test_judge_scale_rejects_out_of_range():
    assert _scale(7) is None and _scale("x") is None and _scale(1) == 0.0
