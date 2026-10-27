"""Retrieval-augmented generation pipeline.

question -> retrieve -> evidence gate -> prompt -> LLM -> citation validation

The evidence gate runs *before* the LLM: if the best chunk's cosine
similarity is below the configured floor, the pipeline answers "insufficient
evidence" without calling the model. The LLM can additionally abstain on its
own. If no provider is available the pipeline returns the retrieved evidence
with an explicit status -- it never fabricates an answer.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from typing import Iterator

from src.config import AppConfig
from src.generation.citations import Citation, is_abstention, resolve_citations
from src.generation.llm_provider import ChatMessage, LLMError, LLMProvider, LLMResponse, estimate_cost
from src.generation.prompt_templates import INSUFFICIENT_MARKER, build_messages
from src.knowledge_base import KnowledgeBase, RetrievalResult, RetrievalSettings
from src.models import RetrievedChunk
from src.persistence.repositories import QueryLogRepository
from src.retrieval.keyword_retriever import tokenize

logger = logging.getLogger(__name__)

STATUS_MESSAGES = {
    "answered": "Answer generated from the retrieved sources.",
    "insufficient_evidence": "The knowledge base does not contain sufficiently relevant evidence for this question.",
    "llm_abstained": "The model reported that the retrieved sources do not answer the question.",
    "llm_unavailable": "No LLM is available, so no answer was generated. The retrieved evidence is shown below.",
    "llm_error": "The LLM call failed, so no answer was generated. The retrieved evidence is shown below.",
    "no_documents": "The knowledge base is empty. Upload documents or load the sample data first.",
}


@dataclass
class EvidenceCheck:
    sufficient: bool
    max_dense_score: float
    threshold: float
    term_coverage: float
    reason: str


def check_evidence(question: str, chunks: list[RetrievedChunk], threshold: float) -> EvidenceCheck:
    """Rule-based sufficiency check independent of the LLM.

    * ``max_dense_score``: highest query/chunk cosine among retrieved chunks.
    * ``term_coverage``: share of the question's content words that appear in
      at least one retrieved chunk (reported for transparency).
    The decision uses the cosine floor only, so it is easy to reason about.
    """
    if not chunks:
        return EvidenceCheck(False, 0.0, threshold, 0.0, "No chunks were retrieved.")
    best = max(c.dense_score for c in chunks)
    terms = set(tokenize(question))
    found = set()
    for c in chunks:
        found |= terms & set(tokenize(c.text))
    coverage = len(found) / len(terms) if terms else 0.0
    if best < threshold:
        reason = f"Best semantic match {best:.2f} is below the relevance floor {threshold:.2f}."
        return EvidenceCheck(False, best, threshold, coverage, reason)
    return EvidenceCheck(True, best, threshold, coverage, f"Best semantic match {best:.2f} ≥ floor {threshold:.2f}.")


@dataclass
class RAGResult:
    question: str
    status: str
    message: str
    answer: str | None
    retrieved: list[RetrievedChunk]
    citations: list[Citation]
    invalid_citations: list[str]
    evidence: EvidenceCheck
    timings: dict[str, float]
    usage: dict[str, int | float | None]
    provider: str
    model: str
    prompt_variant: str
    settings: RetrievalSettings

    @property
    def has_valid_citation(self) -> bool:
        return bool(self.citations)

    def to_log_entry(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "status": self.status,
            "provider": self.provider,
            "model": self.model,
            "config": {**asdict(self.settings), "prompt_variant": self.prompt_variant, "evidence_floor": self.evidence.threshold},
            "citations": [{k: v for k, v in c.to_dict().items() if k != "excerpt"} for c in self.citations],
            "retrieved": [
                {"rank": c.rank, "chunk_id": c.chunk_id, "filename": c.filename, "page": c.page_number,
                 "score": round(c.score, 4), "dense": round(c.dense_score, 4)}
                for c in self.retrieved
            ],
            "timings": self.timings,
            "usage": self.usage,
        }


@dataclass
class PreparedQuery:
    question: str
    retrieval: RetrievalResult
    evidence: EvidenceCheck
    messages: list[ChatMessage]
    started: float
    result: RAGResult | None = None
    blocked_status: str | None = None


class RAGPipeline:
    def __init__(
        self,
        kb: KnowledgeBase,
        provider: LLMProvider,
        cfg: AppConfig,
        query_log: QueryLogRepository | None = None,
    ) -> None:
        self.kb = kb
        self.provider = provider
        self.cfg = cfg
        self.query_log = query_log

    @property
    def settings(self) -> RetrievalSettings:
        return RetrievalSettings.from_config(self.cfg)

    # -------------------------------------------------------------- stages
    def prepare(self, question: str, settings: RetrievalSettings | None = None) -> PreparedQuery:
        started = time.perf_counter()
        settings = settings or self.settings
        retrieval = self.kb.retrieve(question, settings)
        evidence = check_evidence(question, retrieval.chunks, self.cfg.effective_min_relevance)
        messages = build_messages(question, retrieval.chunks, self.cfg.prompt_variant)
        prep = PreparedQuery(question, retrieval, evidence, messages, started)
        if self.kb.stats()["documents"] == 0:
            prep.blocked_status = "no_documents"
        elif not evidence.sufficient:
            prep.blocked_status = "insufficient_evidence"
        return prep

    def _finish(
        self, prep: PreparedQuery, status: str, answer: str | None, response: LLMResponse | None, message: str | None = None
    ) -> RAGResult:
        r = prep.retrieval
        citations, invalid = [], []
        if answer:
            existing = set(self.kb.chunks.get_many([c.chunk_id for c in r.chunks]))
            citations, invalid = resolve_citations(answer, r.chunks, existing)
            explicit = INSUFFICIENT_MARKER in answer.upper()
            if status == "answered" and (explicit or (is_abstention(answer) and not citations)):
                status = "llm_abstained"
        timings = {
            "index_build_s": r.index_build_s,
            "query_embedding_s": r.query_embedding_s,
            "search_s": r.search_s,
            "retrieval_s": r.retrieval_s,
            "generation_s": response.latency_s if response else 0.0,
            "first_token_s": response.first_token_s if response and response.first_token_s else None,
            "total_s": time.perf_counter() - prep.started,
        }
        usage: dict[str, int | float | None] = {}
        if response:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "estimated_cost": estimate_cost(response.usage, self.cfg.price_per_1k_input, self.cfg.price_per_1k_output),
            }
        result = RAGResult(
            question=prep.question,
            status=status,
            message=message or STATUS_MESSAGES[status],
            answer=answer,
            retrieved=r.chunks,
            citations=citations,
            invalid_citations=invalid,
            evidence=prep.evidence,
            timings=timings,
            usage=usage,
            provider=self.provider.name,
            model=response.model if response else self.provider.model,
            prompt_variant=self.cfg.prompt_variant,
            settings=r.settings,
        )
        prep.result = result
        return result

    def log(self, result: RAGResult) -> None:
        if self.query_log is None:
            return
        try:
            self.query_log.add(result.to_log_entry())
        except Exception:  # logging must never break answering
            logger.exception("Could not write query log entry")

    # --------------------------------------------------------- entrypoints
    def answer(self, question: str, settings: RetrievalSettings | None = None, log: bool = True) -> RAGResult:
        """Run the full pipeline synchronously."""
        prep = self.prepare(question, settings)
        if prep.blocked_status:
            result = self._finish(prep, prep.blocked_status, None, None)
        else:
            try:
                response = self.provider.generate(prep.messages, self.cfg.temperature, self.cfg.max_answer_tokens)
                result = self._finish(prep, "answered", response.text.strip(), response)
            except LLMError as exc:
                result = self._llm_failure(prep, exc)
        if log:
            self.log(result)
        return result

    def stream_answer(self, prep: PreparedQuery, log: bool = True) -> Iterator[str]:
        """Yield answer text as it is generated; ``prep.result`` is set when done.

        Call :meth:`prepare` first and check ``prep.blocked_status``.
        """
        if prep.blocked_status:
            result = self._finish(prep, prep.blocked_status, None, None)
            if log:
                self.log(result)
            return
        parts: list[str] = []
        try:
            for delta in self.provider.stream(prep.messages, self.cfg.temperature, self.cfg.max_answer_tokens):
                parts.append(delta)
                yield delta
            response = self.provider.last_response
            self._finish(prep, "answered", "".join(parts).strip(), response)
        except LLMError as exc:
            self._llm_failure(prep, exc, partial="".join(parts))
        if log and prep.result:
            self.log(prep.result)

    def _llm_failure(self, prep: PreparedQuery, exc: LLMError, partial: str = "") -> RAGResult:
        from src.generation.llm_provider import ProviderUnavailableError

        status = "llm_unavailable" if isinstance(exc, ProviderUnavailableError) else "llm_error"
        logger.warning("LLM failure (%s): %s", type(exc).__name__, exc)
        msg = f"{STATUS_MESSAGES[status]} Details: {exc}"
        if partial:
            msg += " A partial answer was discarded because it was incomplete."
        return self._finish(prep, status, None, None, message=msg)
