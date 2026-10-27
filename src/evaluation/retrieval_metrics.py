"""Retrieval metrics over labelled relevance.

Definitions (k = the retrieval cutoff, i.e. top_k):

* Precision@k  = (# relevant chunks in the top k) / k
* Recall@k     = share of the item's relevance targets found in the top k.
                 Targets are its evidence passages (a passage is found when
                 any top-k chunk contains it), else its labelled chunk ids,
                 else its relevant documents.
* Hit Rate@k   = 1 if at least one relevant chunk is in the top k, else 0
* RR           = 1 / rank of the first relevant chunk (0 if none); MRR is the
                 mean over queries
* nDCG@k       = binary-relevance DCG normalised by the ideal ordering
* Doc Recall@k = share of the relevant documents that appear in the top k

A chunk matches an evidence passage if, after normalisation, the passage is
a substring of the chunk, or the chunk shares a contiguous run of at least
``EVIDENCE_MATCH_RATIO`` (60%) of the passage's characters -- which handles
passages split across a chunk boundary.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from src.evaluation.benchmark import BenchmarkItem
from src.models import RetrievedChunk

EVIDENCE_MATCH_RATIO = 0.6
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _WS.sub(" ", text.lower()).strip()


def evidence_in_text(evidence: str, text: str, ratio: float = EVIDENCE_MATCH_RATIO) -> bool:
    ev, tx = normalize(evidence), normalize(text)
    if not ev:
        return False
    if ev in tx:
        return True
    match = SequenceMatcher(None, ev, tx, autojunk=False).find_longest_match(0, len(ev), 0, len(tx))
    return match.size >= ratio * len(ev)


def chunk_relevance(chunk: RetrievedChunk, item: BenchmarkItem) -> bool:
    """Is this retrieved chunk relevant to the benchmark item?"""
    if item.relevant_chunk_ids:
        return chunk.chunk_id in item.relevant_chunk_ids
    if item.evidence:
        return any(evidence_in_text(e, chunk.text) for e in item.evidence)
    if item.relevant_documents:
        return chunk.filename in item.relevant_documents
    return False


@dataclass
class RetrievalScores:
    precision: float
    recall: float
    hit: float
    reciprocal_rank: float
    ndcg: float
    doc_recall: float | None
    relevance: list[bool]
    label_type: str

    def as_dict(self, k: int) -> dict:
        return {
            f"precision@{k}": self.precision,
            f"recall@{k}": self.recall,
            f"hit@{k}": self.hit,
            "rr": self.reciprocal_rank,
            f"ndcg@{k}": self.ndcg,
            f"doc_recall@{k}": self.doc_recall,
        }


def label_type(item: BenchmarkItem) -> str:
    if item.relevant_chunk_ids:
        return "chunk_ids"
    if item.evidence:
        return "evidence"
    if item.relevant_documents:
        return "documents"
    return "none"


def precision_at_k(relevance: list[bool], k: int) -> float:
    return sum(relevance[:k]) / k if k > 0 else 0.0


def hit_at_k(relevance: list[bool], k: int) -> float:
    return 1.0 if any(relevance[:k]) else 0.0


def reciprocal_rank(relevance: list[bool]) -> float:
    for i, rel in enumerate(relevance, start=1):
        if rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(relevance: list[bool], k: int, n_relevant: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, rel in enumerate(relevance[:k]) if rel)
    ideal_hits = min(max(n_relevant, sum(relevance[:k])), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def score_retrieval(item: BenchmarkItem, retrieved: list[RetrievedChunk], k: int) -> RetrievalScores:
    top = retrieved[:k]
    relevance = [chunk_relevance(c, item) for c in top]
    kind = label_type(item)

    if kind == "chunk_ids":
        targets = len(item.relevant_chunk_ids)
        found = len({c.chunk_id for c in top} & set(item.relevant_chunk_ids))
    elif kind == "evidence":
        targets = len(item.evidence)
        found = sum(1 for e in item.evidence if any(evidence_in_text(e, c.text) for c in top))
    elif kind == "documents":
        targets = len(item.relevant_documents)
        found = len({c.filename for c in top} & set(item.relevant_documents))
    else:
        targets, found = 0, 0
    recall = found / targets if targets else 0.0

    doc_recall = None
    if item.relevant_documents:
        doc_recall = len({c.filename for c in top} & set(item.relevant_documents)) / len(item.relevant_documents)

    return RetrievalScores(
        precision=precision_at_k(relevance, k),
        recall=recall,
        hit=hit_at_k(relevance, k),
        reciprocal_rank=reciprocal_rank(relevance),
        ndcg=ndcg_at_k(relevance, k, targets),
        doc_recall=doc_recall,
        relevance=relevance,
        label_type=kind,
    )
