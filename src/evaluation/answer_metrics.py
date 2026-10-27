"""Answer-quality metrics.

Deterministic metrics (always available once an answer exists):

* **Citation validity** -- valid labels / all labels in the answer. A label is
  valid only if it resolves to a chunk that was in the prompt and still exists.
* **Citation coverage** -- (aggregate) share of non-abstaining answers with at
  least one valid citation.
* **Cited-evidence precision** -- share of cited chunks that are relevant
  according to the benchmark labels.
* **Key-fact recall** -- share of the item's ``key_facts`` that appear in the
  answer after normalisation (case, whitespace, thousands separators).
* **Token F1** -- SQuAD-style token overlap with the reference answer.
* **Lexical faithfulness** -- share of answer sentences whose content words
  are mostly (>= 60%) found in the cited chunks (all retrieved chunks if the
  sentence has no citation). A cheap, transparent proxy for "is every claim
  backed by a source".
* **Context relevance** -- mean query/chunk cosine of the retrieved context.

Optional model-graded metrics (:func:`llm_judge`) ask an LLM to rate
faithfulness and correctness 1-5. They are estimates from another model, not
ground truth, and the judge model and prompt version are recorded with the
results.
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from dataclasses import dataclass

from src.evaluation.retrieval_metrics import chunk_relevance
from src.evaluation.benchmark import BenchmarkItem
from src.generation.citations import extract_labels
from src.generation.llm_provider import ChatMessage, LLMError, LLMProvider
from src.models import RetrievedChunk
from src.retrieval.keyword_retriever import tokenize

FAITHFUL_SENTENCE_THRESHOLD = 0.6
_LABEL_GROUP = re.compile(r"\[(?:\s*S\d+\s*[,;]?)+\]", re.IGNORECASE)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation) - {"$", "%", "."}


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    text = "".join(" " if ch in _PUNCT else ch for ch in text)
    text = re.sub(r"\.(?!\d)", " ", text)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def token_f1(prediction: str, reference: str) -> float:
    pred = normalize_answer(_LABEL_GROUP.sub(" ", prediction)).split()
    ref = normalize_answer(reference).split()
    if not pred or not ref:
        return 0.0
    common = Counter(pred) & Counter(ref)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(pred), same / len(ref)
    return 2 * precision * recall / (precision + recall)


def key_fact_recall(answer: str, key_facts: list[str]) -> float | None:
    if not key_facts:
        return None
    norm = normalize_answer(answer)
    return sum(1 for f in key_facts if normalize_answer(f) in norm) / len(key_facts)


def split_sentences(answer: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(answer) if s and s.strip()]


def lexical_faithfulness(answer: str, sources: list[RetrievedChunk]) -> float | None:
    """Share of substantive sentences supported by their cited (or any retrieved) chunk."""
    if not sources:
        return None
    source_tokens = [set(tokenize(c.text)) for c in sources]
    scored = 0
    supported = 0
    for sentence in split_sentences(answer):
        labels = extract_labels(sentence)
        tokens = set(tokenize(_LABEL_GROUP.sub(" ", sentence)))
        tokens.discard("inference")
        if len(tokens) < 3:
            continue
        idxs = [int(l[1:]) - 1 for l in labels if 0 < int(l[1:]) <= len(sources)]
        pool = [source_tokens[i] for i in idxs] or source_tokens
        best = max(len(tokens & toks) / len(tokens) for toks in pool)
        scored += 1
        supported += best >= FAITHFUL_SENTENCE_THRESHOLD
    return supported / scored if scored else None


def context_relevance(sources: list[RetrievedChunk]) -> float | None:
    if not sources:
        return None
    return sum(c.dense_score for c in sources) / len(sources)


def cited_evidence_precision(cited_chunk_ids: list[str], sources: list[RetrievedChunk], item: BenchmarkItem) -> float | None:
    if not cited_chunk_ids or not item.has_relevance_labels:
        return None
    by_id = {c.chunk_id: c for c in sources}
    cited = [by_id[cid] for cid in cited_chunk_ids if cid in by_id]
    if not cited:
        return None
    return sum(chunk_relevance(c, item) for c in cited) / len(cited)


# ----------------------------------------------------------------- LLM judge
JUDGE_PROMPT_VERSION = "judge-v1"
JUDGE_SYSTEM = """You are a strict evaluator of answers produced by a document question-answering system.
You will receive a question, a reference answer, the retrieved source passages and the system's answer.
Rate two things on an integer scale from 1 (worst) to 5 (best):
- faithfulness: are all claims in the answer supported by the source passages? (5 = every claim supported, 1 = mostly unsupported or invented)
- correctness: does the answer agree with the reference answer? (5 = fully correct and complete, 1 = wrong)
Respond with JSON only, for example: {"faithfulness": 4, "correctness": 5, "rationale": "one sentence"}"""


@dataclass
class JudgeScore:
    faithfulness: float | None
    correctness: float | None
    rationale: str
    judge_model: str
    prompt_version: str = JUDGE_PROMPT_VERSION
    error: str | None = None


def _scale(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not 1 <= v <= 5:
        return None
    return (v - 1) / 4


def llm_judge(judge: LLMProvider, question: str, reference: str, answer: str, sources: list[RetrievedChunk]) -> JudgeScore:
    context = "\n\n".join(f"[S{i}] {c.text}" for i, c in enumerate(sources, start=1))
    user = (
        f"Question:\n{question}\n\nReference answer:\n{reference or '(none)'}\n\n"
        f"Source passages:\n{context}\n\nSystem answer:\n{answer}\n\nReturn JSON only."
    )
    try:
        resp = judge.generate([ChatMessage("system", JUDGE_SYSTEM), ChatMessage("user", user)], temperature=0.0, max_tokens=200)
    except LLMError as exc:
        return JudgeScore(None, None, "", judge.model, error=str(exc))
    match = re.search(r"\{.*\}", resp.text, re.S)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    f, c = _scale(data.get("faithfulness")), _scale(data.get("correctness"))
    err = None if (f is not None and c is not None) else "Judge output could not be parsed."
    return JudgeScore(f, c, str(data.get("rationale", ""))[:300], resp.model, error=err)
