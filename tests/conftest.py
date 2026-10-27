"""Shared fixtures. Unit tests need no downloaded model, LLM or network."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Iterator

import fitz
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import AppConfig  # noqa: E402
from src.embeddings.embedder import l2_normalize  # noqa: E402
from src.generation.llm_provider import (  # noqa: E402
    ChatMessage,
    LLMProvider,
    LLMResponse,
    LLMUsage,
    ProviderStatus,
)
from src.knowledge_base import KnowledgeBase  # noqa: E402
from src.persistence.database import Database  # noqa: E402
from src.retrieval.keyword_retriever import tokenize  # noqa: E402


class HashingEmbedder:
    """Deterministic bag-of-words embedder: cosine reflects word overlap."""

    def __init__(self, model_name: str = "test/hashing", dim: int = 256) -> None:
        self.model_name = model_name
        self._dim = dim
        self.calls = 0

    @property
    def dimension(self) -> int:
        return self._dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self._dim, dtype=np.float32)
        for tok in tokenize(text):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self._dim] += 1.0
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return l2_normalize(np.vstack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return l2_normalize(self._vec(text))[0]


class MockProvider(LLMProvider):
    """Scriptable LLM: returns a fixed answer or raises a given exception."""

    name = "mock"

    def __init__(self, answer: str = "The answer is 18 days [S1].", error: Exception | None = None, available: bool = True):
        super().__init__(model="mock-model")
        self.answer = answer
        self.error = error
        self.available = available
        self.received: list[list[ChatMessage]] = []

    def status(self) -> ProviderStatus:
        return ProviderStatus(self.available, "mock ready" if self.available else "mock offline")

    def generate(self, messages, temperature=0.1, max_tokens=512) -> LLMResponse:
        self.received.append(messages)
        if self.error:
            raise self.error
        resp = LLMResponse(self.answer, self.model, self.name, 0.01, LLMUsage(100, 20))
        self.last_response = resp
        return resp

    def stream(self, messages, temperature=0.1, max_tokens=512) -> Iterator[str]:
        self.received.append(messages)
        if self.error:
            raise self.error
        for word in self.answer.split(" "):
            yield word + " "
        self.last_response = LLMResponse(self.answer, self.model, self.name, 0.01, LLMUsage(100, 20), first_token_s=0.001)


POLICY_MD = b"""# Leave Policy

## Annual leave
Full-time employees receive 18 days of paid annual leave per calendar year. Unused leave of up to 5 days can be carried over.

## Sick leave
Employees are entitled to 14 days of paid sick leave. A medical certificate is required for absences of more than two consecutive days.
"""

IT_TXT = b"""IT incident handling.
P1 incidents receive a first response within 15 minutes. The CTO is notified within 30 minutes.
Lost laptops must be reported within 2 hours. Passwords must be at least 14 characters long.
"""


def make_pdf(pages: list[str]) -> bytes:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(fitz.Rect(50, 50, 550, 800), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def cfg(tmp_path: Path) -> AppConfig:
    return AppConfig(
        data_dir=tmp_path / "data",
        outputs_dir=tmp_path / "outputs",
        embedding_model="test/hashing",
        chunk_size=200,
        chunk_overlap=40,
        top_k=3,
        min_relevance=0.2,
    )


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()


@pytest.fixture
def kb(cfg: AppConfig, embedder: HashingEmbedder) -> KnowledgeBase:
    return KnowledgeBase(cfg, Database(cfg.db_path), embedder_factory=lambda name: embedder)


@pytest.fixture
def loaded_kb(kb: KnowledgeBase) -> KnowledgeBase:
    kb.add_document("leave_policy.md", POLICY_MD)
    kb.add_document("it_policy.txt", IT_TXT)
    kb.add_document("handbook.pdf", make_pdf(["Expense claims must be submitted within 30 days.", "The learning budget is SGD 1,200 per year."]))
    return kb
