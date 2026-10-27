"""Repository classes: all SQL lives here, the rest of the app uses Python objects."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from src.models import Chunk, ExtractedDocument, TextUnit
from src.persistence.database import Database


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class DocumentRecord:
    doc_id: str
    filename: str
    file_type: str
    content_hash: str
    size_bytes: int
    page_count: int
    char_count: int
    metadata: dict[str, Any]
    warnings: list[str]
    is_sample: bool
    created_at: str


class DocumentRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    @staticmethod
    def _row(row: Any) -> DocumentRecord:
        return DocumentRecord(
            doc_id=row["doc_id"],
            filename=row["filename"],
            file_type=row["file_type"],
            content_hash=row["content_hash"],
            size_bytes=row["size_bytes"],
            page_count=row["page_count"],
            char_count=row["char_count"],
            metadata=json.loads(row["metadata_json"]),
            warnings=json.loads(row["warnings_json"]),
            is_sample=bool(row["is_sample"]),
            created_at=row["created_at"],
        )

    def add(self, doc_id: str, digest: str, size: int, doc: ExtractedDocument, is_sample: bool = False) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                """INSERT INTO documents(doc_id, filename, file_type, content_hash, size_bytes, page_count,
                       char_count, metadata_json, warnings_json, is_sample, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    doc_id,
                    doc.filename,
                    doc.file_type,
                    digest,
                    size,
                    doc.page_count,
                    doc.char_count,
                    json.dumps(doc.metadata, default=str),
                    json.dumps(doc.warnings),
                    int(is_sample),
                    utc_now(),
                ),
            )
            conn.executemany(
                "INSERT INTO text_units(doc_id, unit_index, page_number, section, text) VALUES (?,?,?,?,?)",
                [(doc_id, i, u.page_number, u.section, u.text) for i, u in enumerate(doc.units)],
            )
            _bump_corpus_version(conn)

    def get(self, doc_id: str) -> DocumentRecord | None:
        with self.db.read() as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return self._row(row) if row else None

    def find_by_hash(self, digest: str) -> DocumentRecord | None:
        with self.db.read() as conn:
            row = conn.execute("SELECT * FROM documents WHERE content_hash = ?", (digest,)).fetchone()
        return self._row(row) if row else None

    def list(self) -> list[DocumentRecord]:
        with self.db.read() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY created_at, filename").fetchall()
        return [self._row(r) for r in rows]

    def active_ids(self) -> set[str]:
        with self.db.read() as conn:
            return {r[0] for r in conn.execute("SELECT doc_id FROM documents")}

    def units(self, doc_id: str) -> list[TextUnit]:
        with self.db.read() as conn:
            rows = conn.execute(
                "SELECT page_number, section, text FROM text_units WHERE doc_id = ? ORDER BY unit_index", (doc_id,)
            ).fetchall()
        return [TextUnit(text=r["text"], page_number=r["page_number"], section=r["section"]) for r in rows]

    def delete(self, doc_id: str) -> bool:
        """Delete a document; chunks, text and embeddings cascade."""
        with self.db.transaction() as conn:
            cur = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            if cur.rowcount:
                _bump_corpus_version(conn)
            return cur.rowcount > 0

    def corpus_version(self) -> int:
        with self.db.read() as conn:
            row = conn.execute("SELECT value FROM app_state WHERE key = 'corpus_version'").fetchone()
        return int(row[0]) if row else 0


def _bump_corpus_version(conn: Any) -> None:
    conn.execute("UPDATE app_state SET value = CAST(value AS INTEGER) + 1 WHERE key = 'corpus_version'")


class ChunkRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def has_chunks(self, doc_id: str, key: str) -> bool:
        with self.db.read() as conn:
            row = conn.execute(
                "SELECT 1 FROM chunks WHERE doc_id = ? AND chunking_key = ? LIMIT 1", (doc_id, key)
            ).fetchone()
        return row is not None

    def add_chunks(self, key: str, chunks: list[Chunk]) -> None:
        with self.db.transaction() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO chunks(chunk_id, doc_id, chunking_key, chunk_index, page_number, section,
                       char_start, char_end, text) VALUES (?,?,?,?,?,?,?,?,?)""",
                [
                    (c.chunk_id, c.doc_id, key, c.chunk_index, c.page_number, c.section, c.char_start, c.char_end, c.text)
                    for c in chunks
                ],
            )

    @staticmethod
    def _chunk(row: Any) -> Chunk:
        return Chunk(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            chunk_index=row["chunk_index"],
            text=row["text"],
            page_number=row["page_number"],
            section=row["section"],
            char_start=row["char_start"],
            char_end=row["char_end"],
        )

    def chunks_for_key(self, key: str, doc_id: str | None = None) -> list[Chunk]:
        sql = """SELECT c.* FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
                 WHERE c.chunking_key = ?"""
        params: list[Any] = [key]
        if doc_id:
            sql += " AND c.doc_id = ?"
            params.append(doc_id)
        sql += " ORDER BY d.created_at, c.doc_id, c.chunk_index"
        with self.db.read() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._chunk(r) for r in rows]

    def get_many(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        out: dict[str, Chunk] = {}
        with self.db.read() as conn:
            for start in range(0, len(chunk_ids), 500):
                batch = chunk_ids[start : start + 500]
                marks = ",".join("?" * len(batch))
                for row in conn.execute(f"SELECT * FROM chunks WHERE chunk_id IN ({marks})", batch):
                    out[row["chunk_id"]] = self._chunk(row)
        return out

    def count(self, key: str | None = None) -> int:
        with self.db.read() as conn:
            if key:
                return conn.execute("SELECT COUNT(*) FROM chunks WHERE chunking_key = ?", (key,)).fetchone()[0]
            return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def counts_by_doc(self, key: str) -> dict[str, int]:
        with self.db.read() as conn:
            rows = conn.execute(
                "SELECT doc_id, COUNT(*) FROM chunks WHERE chunking_key = ? GROUP BY doc_id", (key,)
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    def search_text(self, key: str, needle: str, limit: int = 50) -> list[Chunk]:
        """Case-insensitive substring search over chunk text (parameterised LIKE)."""
        escaped = needle.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with self.db.read() as conn:
            rows = conn.execute(
                """SELECT * FROM chunks WHERE chunking_key = ? AND text LIKE ? ESCAPE '\\'
                   ORDER BY doc_id, chunk_index LIMIT ?""",
                (key, f"%{escaped}%", limit),
            ).fetchall()
        return [self._chunk(r) for r in rows]

    # ---- embedding cache -------------------------------------------------
    def embeddings(self, chunk_ids: list[str], model: str) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        with self.db.read() as conn:
            for start in range(0, len(chunk_ids), 500):
                batch = chunk_ids[start : start + 500]
                marks = ",".join("?" * len(batch))
                rows = conn.execute(
                    f"SELECT chunk_id, dim, vector FROM embeddings WHERE model_name = ? AND chunk_id IN ({marks})",
                    [model, *batch],
                )
                for row in rows:
                    vec = np.frombuffer(row["vector"], dtype=np.float32)
                    if vec.shape[0] == row["dim"]:
                        out[row["chunk_id"]] = vec
        return out

    def save_embeddings(self, model: str, chunk_ids: list[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        with self.db.transaction() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings(chunk_id, model_name, dim, vector) VALUES (?,?,?,?)",
                [(cid, model, int(vec.shape[0]), vec.tobytes()) for cid, vec in zip(chunk_ids, vectors)],
            )

    def embedding_counts(self) -> dict[str, int]:
        with self.db.read() as conn:
            rows = conn.execute("SELECT model_name, COUNT(*) FROM embeddings GROUP BY model_name").fetchall()
        return {r[0]: r[1] for r in rows}


class QueryLogRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def add(self, entry: dict[str, Any]) -> int:
        with self.db.transaction() as conn:
            cur = conn.execute(
                """INSERT INTO query_log(created_at, question, answer, status, provider, model, config_json,
                       citations_json, retrieved_json, timings_json, usage_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    utc_now(),
                    entry["question"],
                    entry.get("answer"),
                    entry["status"],
                    entry.get("provider"),
                    entry.get("model"),
                    json.dumps(entry.get("config", {})),
                    json.dumps(entry.get("citations", [])),
                    json.dumps(entry.get("retrieved", [])),
                    json.dumps(entry.get("timings", {})),
                    json.dumps(entry.get("usage", {})),
                ),
            )
            return int(cur.lastrowid)

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.db.read() as conn:
            rows = conn.execute("SELECT * FROM query_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for col in ("config_json", "citations_json", "retrieved_json", "timings_json", "usage_json"):
                d[col.removesuffix("_json")] = json.loads(d.pop(col))
            out.append(d)
        return out

    def count(self) -> int:
        with self.db.read() as conn:
            return conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]

    def clear(self) -> None:
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM query_log")


class BenchmarkRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def upsert(self, name: str, items: list[dict[str, Any]], description: str = "", is_sample: bool = False) -> None:
        now = utc_now()
        with self.db.transaction() as conn:
            conn.execute(
                """INSERT INTO benchmarks(name, description, is_sample, items_json, created_at, updated_at)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(name) DO UPDATE SET description = excluded.description,
                       items_json = excluded.items_json, updated_at = excluded.updated_at,
                       is_sample = excluded.is_sample""",
                (name, description, int(is_sample), json.dumps(items), now, now),
            )

    def get(self, name: str) -> dict[str, Any] | None:
        with self.db.read() as conn:
            row = conn.execute("SELECT * FROM benchmarks WHERE name = ?", (name,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["items"] = json.loads(d.pop("items_json"))
        d["is_sample"] = bool(d["is_sample"])
        return d

    def names(self) -> list[str]:
        with self.db.read() as conn:
            return [r[0] for r in conn.execute("SELECT name FROM benchmarks ORDER BY is_sample DESC, name")]

    def delete(self, name: str) -> None:
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM benchmarks WHERE name = ?", (name,))


class ExperimentRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, name: str, kind: str, benchmark: str, grid: list[dict], env: dict, notes: str = "") -> int:
        with self.db.transaction() as conn:
            cur = conn.execute(
                """INSERT INTO experiments(name, kind, benchmark, created_at, status, grid_json, env_json, notes)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (name, kind, benchmark, utc_now(), "running", json.dumps(grid), json.dumps(env), notes),
            )
            return int(cur.lastrowid)

    def finish(self, experiment_id: int, status: str = "completed", notes: str | None = None) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                "UPDATE experiments SET status = ?, finished_at = ?, notes = COALESCE(?, notes) WHERE id = ?",
                (status, utc_now(), notes, experiment_id),
            )

    def add_run(self, experiment_id: int, label: str, config: dict, metrics: dict, per_query: list[dict]) -> int:
        with self.db.transaction() as conn:
            cur = conn.execute(
                "INSERT INTO experiment_runs(experiment_id, label, config_json, metrics_json, created_at) VALUES (?,?,?,?,?)",
                (experiment_id, label, json.dumps(config), json.dumps(metrics), utc_now()),
            )
            run_id = int(cur.lastrowid)
            conn.executemany(
                "INSERT INTO run_results(run_id, qid, question, result_json) VALUES (?,?,?,?)",
                [(run_id, r["qid"], r["question"], json.dumps(r, default=str)) for r in per_query],
            )
            return run_id

    def list(self) -> list[dict[str, Any]]:
        with self.db.read() as conn:
            rows = conn.execute(
                """SELECT e.*, (SELECT COUNT(*) FROM experiment_runs r WHERE r.experiment_id = e.id) AS run_count
                   FROM experiments e ORDER BY e.id DESC"""
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["grid"] = json.loads(d.pop("grid_json"))
            d["env"] = json.loads(d.pop("env_json"))
            out.append(d)
        return out

    def get(self, experiment_id: int) -> dict[str, Any] | None:
        with self.db.read() as conn:
            row = conn.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
            if not row:
                return None
            runs = conn.execute(
                "SELECT * FROM experiment_runs WHERE experiment_id = ? ORDER BY id", (experiment_id,)
            ).fetchall()
        d = dict(row)
        d["grid"] = json.loads(d.pop("grid_json"))
        d["env"] = json.loads(d.pop("env_json"))
        d["runs"] = [
            {
                "id": r["id"],
                "label": r["label"],
                "config": json.loads(r["config_json"]),
                "metrics": json.loads(r["metrics_json"]),
                "created_at": r["created_at"],
            }
            for r in runs
        ]
        return d

    def run_results(self, run_id: int) -> list[dict[str, Any]]:
        with self.db.read() as conn:
            rows = conn.execute("SELECT result_json FROM run_results WHERE run_id = ? ORDER BY id", (run_id,)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def all_runs(self) -> list[dict[str, Any]]:
        """Every run joined with its experiment, for cross-experiment comparison."""
        with self.db.read() as conn:
            rows = conn.execute(
                """SELECT r.id AS run_id, r.label, r.config_json, r.metrics_json, r.created_at,
                          e.id AS experiment_id, e.name AS experiment, e.kind, e.benchmark
                   FROM experiment_runs r JOIN experiments e ON e.id = r.experiment_id ORDER BY r.id"""
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["config"] = json.loads(d.pop("config_json"))
            d["metrics"] = json.loads(d.pop("metrics_json"))
            out.append(d)
        return out

    def delete(self, experiment_id: int) -> None:
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
