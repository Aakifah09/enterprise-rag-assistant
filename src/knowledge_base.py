"""Knowledge base service: documents in, search indexes out.

This is the single entry point the UI, the RAG pipeline and the evaluation
runner use for documents and retrieval. It owns:

* ingestion (validation, extraction, duplicate detection, persistence),
* chunking per chunking configuration (chunks are cached in SQLite),
* embedding per model (vectors cached in SQLite, so switching configs or
  restarting the app does not recompute them),
* in-memory FAISS + BM25 indexes keyed by (chunking, model, corpus version).

Any document add/remove bumps the corpus version, which invalidates every
cached index, so deleted documents can never be retrieved from a stale index.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from src.config import AppConfig, chunking_key
from src.embeddings.embedder import Embedder, get_embedder
from src.ingestion.chunking import chunk_units
from src.ingestion.errors import DuplicateDocumentError
from src.ingestion.loader import content_hash, doc_id_for, extract_document
from src.models import RetrievedChunk
from src.persistence.database import Database
from src.persistence.repositories import ChunkRepository, DocumentRecord, DocumentRepository
from src.retrieval.dense_retriever import dense_search, keyword_search
from src.retrieval.hybrid_retriever import hybrid_search, rrf_search
from src.retrieval.index import SearchIndex, indexable_text

logger = logging.getLogger(__name__)

EmbedderFactory = Callable[[str], Embedder]


@dataclass(frozen=True)
class RetrievalSettings:
    strategy: str = "hybrid"
    top_k: int = 5
    chunk_size: int = 800
    chunk_overlap: int = 120
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    alpha: float = 0.6
    candidate_pool: int = 30

    @classmethod
    def from_config(cls, cfg: AppConfig) -> "RetrievalSettings":
        return cls(
            strategy=cfg.retrieval_strategy,
            top_k=cfg.top_k,
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            embedding_model=cfg.embedding_model,
            alpha=cfg.hybrid_alpha,
            candidate_pool=max(cfg.candidate_pool, cfg.top_k),
        )

    @property
    def chunking_key(self) -> str:
        return chunking_key(self.chunk_size, self.chunk_overlap)

    def label(self) -> str:
        model = self.embedding_model.split("/")[-1]
        extra = f" α={self.alpha}" if self.strategy == "hybrid" else ""
        return f"{self.strategy}{extra} · k={self.top_k} · {self.chunk_size}/{self.chunk_overlap} · {model}"


@dataclass
class IngestReport:
    filename: str
    status: str  # indexed | duplicate | error
    message: str = ""
    doc_id: str | None = None
    pages: int = 0
    chars: int = 0
    chunks: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    settings: RetrievalSettings
    query_embedding_s: float
    search_s: float
    index_build_s: float

    @property
    def retrieval_s(self) -> float:
        return self.query_embedding_s + self.search_s

    @property
    def max_dense_score(self) -> float:
        return max((c.dense_score for c in self.chunks), default=0.0)


class KnowledgeBase:
    def __init__(self, cfg: AppConfig, db: Database | None = None, embedder_factory: EmbedderFactory | None = None):
        self.cfg = cfg
        self.db = db or Database(cfg.db_path)
        self.docs = DocumentRepository(self.db)
        self.chunks = ChunkRepository(self.db)
        self._embedder_factory = embedder_factory or (lambda name: get_embedder(name, cfg.device))
        self._indexes: dict[tuple[str, str], SearchIndex] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ docs
    def embedder(self, model_name: str) -> Embedder:
        return self._embedder_factory(model_name)

    def add_document(self, filename: str, data: bytes, is_sample: bool = False) -> IngestReport:
        """Validate, extract, persist and chunk (default config) one upload.

        Raises IngestionError subclasses for invalid input; duplicates raise
        :class:`DuplicateDocumentError` carrying the existing document id.
        """
        doc = extract_document(filename, data, self.cfg.max_upload_mb)
        digest = content_hash(data)
        existing = self.docs.find_by_hash(digest)
        if existing:
            raise DuplicateDocumentError(
                f"'{doc.filename}' has identical content to already indexed '{existing.filename}'.", existing.doc_id
            )
        doc_id = doc_id_for(data)
        self.docs.add(doc_id, digest, len(data), doc, is_sample=is_sample)
        n_chunks = self._chunk_document(doc_id, self.cfg.chunk_size, self.cfg.chunk_overlap)
        logger.info("Indexed document %s (%d pages, %d chunks)", doc_id, doc.page_count, n_chunks)
        return IngestReport(
            filename=doc.filename,
            status="indexed",
            doc_id=doc_id,
            pages=doc.page_count,
            chars=doc.char_count,
            chunks=n_chunks,
            warnings=doc.warnings,
        )

    def ingest_many(self, files: list[tuple[str, bytes]], is_sample: bool = False) -> list[IngestReport]:
        """Ingest several uploads, turning every failure into a report instead of raising."""
        from src.ingestion.errors import IngestionError

        reports = []
        for name, data in files:
            try:
                reports.append(self.add_document(name, data, is_sample=is_sample))
            except DuplicateDocumentError as exc:
                reports.append(IngestReport(filename=name, status="duplicate", message=str(exc), doc_id=exc.existing_doc_id))
            except IngestionError as exc:
                reports.append(IngestReport(filename=name, status="error", message=str(exc)))
            except Exception as exc:  # unexpected parser failure: never crash the UI
                logger.exception("Unexpected ingestion failure for an upload")
                reports.append(IngestReport(filename=name, status="error", message=f"Unexpected error: {exc}"))
        return reports

    def load_sample_data(self, sample_dir: Path) -> list[IngestReport]:
        files = [
            (p.name, p.read_bytes())
            for p in sorted(sample_dir.iterdir())
            if p.is_file() and p.suffix.lower() in (".pdf", ".txt", ".md")
        ]
        return self.ingest_many(files, is_sample=True)

    def remove_document(self, doc_id: str) -> bool:
        with self._lock:
            removed = self.docs.delete(doc_id)
        self.clear_index_cache()
        return removed

    def clear_index_cache(self) -> None:
        """Drop in-memory indexes; the next search rebuilds from SQLite (embeddings stay cached)."""
        with self._lock:
            self._indexes.clear()

    def list_documents(self) -> list[DocumentRecord]:
        return self.docs.list()

    def corpus_version(self) -> int:
        return self.docs.corpus_version()

    # -------------------------------------------------------------- chunking
    def _chunk_document(self, doc_id: str, chunk_size: int, chunk_overlap: int) -> int:
        key = chunking_key(chunk_size, chunk_overlap)
        if self.chunks.has_chunks(doc_id, key):
            return self.chunks.counts_by_doc(key).get(doc_id, 0)
        chunks = chunk_units(doc_id, self.docs.units(doc_id), chunk_size, chunk_overlap)
        self.chunks.add_chunks(key, chunks)
        return len(chunks)

    def ensure_chunks(self, chunk_size: int, chunk_overlap: int) -> int:
        """Chunk every document for a configuration (idempotent); returns total chunks."""
        for rec in self.docs.list():
            self._chunk_document(rec.doc_id, chunk_size, chunk_overlap)
        return self.chunks.count(chunking_key(chunk_size, chunk_overlap))

    # ---------------------------------------------------------------- index
    def get_index(self, settings: RetrievalSettings) -> tuple[SearchIndex, float]:
        """Return a current index for the settings, building it if needed.

        Returns the index and the seconds spent building it (0 when cached).
        """
        key = (settings.chunking_key, settings.embedding_model)
        version = self.corpus_version()
        with self._lock:
            cached = self._indexes.get(key)
            if cached and cached.corpus_version == version:
                return cached, 0.0
            start = time.perf_counter()
            self.ensure_chunks(settings.chunk_size, settings.chunk_overlap)
            chunks = self.chunks.chunks_for_key(settings.chunking_key)
            embedder = self.embedder(settings.embedding_model)
            ids = [c.chunk_id for c in chunks]
            cached_vecs = self.chunks.embeddings(ids, settings.embedding_model)
            missing = [c for c in chunks if c.chunk_id not in cached_vecs or cached_vecs[c.chunk_id].shape[0] != embedder.dimension]
            if missing:
                new_vecs = embedder.embed_documents([indexable_text(c) for c in missing])
                self.chunks.save_embeddings(settings.embedding_model, [c.chunk_id for c in missing], new_vecs)
                cached_vecs.update({c.chunk_id: v for c, v in zip(missing, new_vecs)})
            vectors = (
                np.vstack([cached_vecs[i] for i in ids]).astype(np.float32)
                if ids
                else np.zeros((0, embedder.dimension), dtype=np.float32)
            )
            filenames = {r.doc_id: r.filename for r in self.docs.list()}
            index = SearchIndex.build(
                settings.chunking_key, settings.embedding_model, version, chunks, vectors, filenames, embedder.dimension
            )
            self._indexes[key] = index
            elapsed = time.perf_counter() - start
            logger.info(
                "Built index %s/%s: %d chunks (%d newly embedded) in %.2fs",
                settings.chunking_key, settings.embedding_model, len(chunks), len(missing), elapsed,
            )
            return index, elapsed

    def retrieve(self, query: str, settings: RetrievalSettings) -> RetrievalResult:
        """Run one retrieval strategy and return ranked chunks with timings."""
        if not query.strip():
            raise ValueError("Query must not be empty.")
        index, build_s = self.get_index(settings)
        if index.is_empty:
            return RetrievalResult([], settings, 0.0, 0.0, build_s)

        t0 = time.perf_counter()
        qvec = self.embedder(settings.embedding_model).embed_query(query)
        t1 = time.perf_counter()
        if settings.strategy == "dense":
            hits = dense_search(index, qvec, settings.top_k)
        elif settings.strategy == "keyword":
            hits = keyword_search(index, query, qvec, settings.top_k)
        elif settings.strategy == "hybrid":
            hits = hybrid_search(index, query, qvec, settings.top_k, settings.alpha, settings.candidate_pool)
        elif settings.strategy == "rrf":
            hits = rrf_search(index, query, qvec, settings.top_k, settings.candidate_pool)
        else:
            raise ValueError(f"Unknown retrieval strategy: {settings.strategy}")
        # Defence in depth: never return chunks whose document no longer exists.
        active = self.docs.active_ids()
        hits = [h for h in hits if h.doc_id in active]
        for rank, h in enumerate(hits, start=1):
            h.rank = rank
        t2 = time.perf_counter()
        return RetrievalResult(hits, settings, t1 - t0, t2 - t1, build_s)

    def stats(self) -> dict[str, int]:
        docs = self.docs.list()
        return {
            "documents": len(docs),
            "pages": sum(d.page_count for d in docs),
            "chunks": self.chunks.count(self.cfg.chunking_key),
            "corpus_version": self.corpus_version(),
        }
