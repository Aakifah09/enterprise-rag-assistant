"""Benchmark format, validation and import/export.

A benchmark item::

    {
      "qid": "Q01",
      "question": "How many days of annual leave do new employees get?",
      "expected_answer": "18 days per year, accrued monthly.",
      "answerable": true,
      "relevant_documents": ["employee_onboarding_policy.md"],
      "evidence": ["Full-time employees receive 18 days of paid annual leave"],
      "relevant_chunk_ids": [],
      "key_facts": ["18 days"],
      "category": "policy"
    }

Relevance is defined by labels, not by similarity:

* ``evidence`` -- verbatim passages that answer the question. A retrieved
  chunk is relevant if it contains (most of) one of them. This survives
  re-chunking, so the same benchmark works for every chunk size.
* ``relevant_chunk_ids`` -- exact chunk ids; only valid for the chunking
  configuration they were labelled under, so they take precedence only when
  provided.
* ``relevant_documents`` -- used for document-level metrics and as a last
  resort when no finer label exists.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class BenchmarkError(ValueError):
    """A benchmark file or item is invalid."""


@dataclass
class BenchmarkItem:
    qid: str
    question: str
    expected_answer: str = ""
    answerable: bool = True
    relevant_documents: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    relevant_chunk_ids: list[str] = field(default_factory=list)
    key_facts: list[str] = field(default_factory=list)
    category: str = ""

    @property
    def has_relevance_labels(self) -> bool:
        return bool(self.evidence or self.relevant_chunk_ids or self.relevant_documents)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Benchmark:
    name: str
    items: list[BenchmarkItem]
    description: str = ""
    is_sample: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "is_sample": self.is_sample,
            "items": [i.to_dict() for i in self.items],
        }


def _as_list(value: Any, sep: str = "|") -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and value != value:  # NaN from pandas
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            return _as_list(json.loads(text), sep)
        except json.JSONDecodeError:
            pass
    return [p.strip() for p in text.split(sep) if p.strip()]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and value != value):
        return True
    return str(value).strip().lower() not in ("false", "0", "no", "n", "")


def parse_item(raw: dict[str, Any], position: int) -> BenchmarkItem:
    if not isinstance(raw, dict):
        raise BenchmarkError(f"Item {position} must be an object.")
    qid = str(raw.get("qid") or "").strip()
    question = str(raw.get("question") or "").strip()
    if not qid:
        raise BenchmarkError(f"Item {position} is missing 'qid'.")
    if not question:
        raise BenchmarkError(f"Item {qid} is missing 'question'.")
    expected = raw.get("expected_answer")
    return BenchmarkItem(
        qid=qid,
        question=question,
        expected_answer="" if expected is None or expected != expected else str(expected).strip(),
        answerable=_as_bool(raw.get("answerable", True)),
        relevant_documents=_as_list(raw.get("relevant_documents")),
        evidence=_as_list(raw.get("evidence"), sep="||"),
        relevant_chunk_ids=_as_list(raw.get("relevant_chunk_ids")),
        key_facts=_as_list(raw.get("key_facts")),
        category=str(raw.get("category") or "").strip() if raw.get("category") == raw.get("category") else "",
    )


def validate_items(items: list[BenchmarkItem]) -> list[str]:
    """Raise on hard errors; return a list of soft warnings."""
    if not items:
        raise BenchmarkError("A benchmark needs at least one item.")
    seen: set[str] = set()
    warnings: list[str] = []
    for item in items:
        if item.qid in seen:
            raise BenchmarkError(f"Duplicate qid '{item.qid}'.")
        seen.add(item.qid)
        if item.answerable and not item.has_relevance_labels:
            warnings.append(f"{item.qid}: answerable but has no relevance labels; it is skipped in retrieval metrics.")
        if item.answerable and not (item.expected_answer or item.key_facts):
            warnings.append(f"{item.qid}: no expected answer or key facts; answer correctness cannot be scored.")
        if not item.answerable and (item.evidence or item.relevant_chunk_ids):
            warnings.append(f"{item.qid}: marked unanswerable but has evidence labels.")
    return warnings


def benchmark_from_dict(data: dict[str, Any], default_name: str = "imported") -> Benchmark:
    if isinstance(data, list):
        data = {"items": data}
    if not isinstance(data, dict) or "items" not in data:
        raise BenchmarkError("Benchmark JSON must be an object with an 'items' list (or a bare list of items).")
    items = [parse_item(r, i + 1) for i, r in enumerate(data["items"])]
    validate_items(items)
    return Benchmark(
        name=str(data.get("name") or default_name).strip(),
        items=items,
        description=str(data.get("description") or ""),
        is_sample=bool(data.get("is_sample", False)),
    )


def load_benchmark_file(path: Path | str) -> Benchmark:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"{path.name} is not valid JSON: {exc}") from exc
    return benchmark_from_dict(data, default_name=path.stem)


def benchmark_from_csv(text: str, name: str) -> Benchmark:
    """CSV columns: qid, question, expected_answer, answerable, relevant_documents (|),
    evidence (||), relevant_chunk_ids (|), key_facts (|), category."""
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or not {"qid", "question"} <= set(reader.fieldnames):
        raise BenchmarkError("CSV must have at least 'qid' and 'question' columns.")
    items = [parse_item(row, i + 1) for i, row in enumerate(reader)]
    validate_items(items)
    return Benchmark(name=name, items=items)


def benchmark_to_csv(bench: Benchmark) -> str:
    buf = io.StringIO()
    fields = ["qid", "question", "expected_answer", "answerable", "relevant_documents", "evidence",
              "relevant_chunk_ids", "key_facts", "category"]
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for it in bench.items:
        writer.writerow({
            "qid": it.qid,
            "question": it.question,
            "expected_answer": it.expected_answer,
            "answerable": it.answerable,
            "relevant_documents": "|".join(it.relevant_documents),
            "evidence": "||".join(it.evidence),
            "relevant_chunk_ids": "|".join(it.relevant_chunk_ids),
            "key_facts": "|".join(it.key_facts),
            "category": it.category,
        })
    return buf.getvalue()
