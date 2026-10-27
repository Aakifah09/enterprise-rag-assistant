"""SQLite connection management and schema.

The database stores extracted text (not the original uploaded files), chunk
mappings, an embedding cache, benchmarks, query history and experiments.
Embeddings are stored as raw float32 bytes and decoded with
``numpy.frombuffer`` -- no pickle is ever used.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id        TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    file_type     TEXT NOT NULL,
    content_hash  TEXT NOT NULL UNIQUE,
    size_bytes    INTEGER NOT NULL,
    page_count    INTEGER NOT NULL,
    char_count    INTEGER NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    warnings_json TEXT NOT NULL DEFAULT '[]',
    is_sample     INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS text_units (
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    unit_index  INTEGER NOT NULL,
    page_number INTEGER,
    section     TEXT,
    text        TEXT NOT NULL,
    PRIMARY KEY (doc_id, unit_index)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunking_key TEXT NOT NULL,
    chunk_index  INTEGER NOT NULL,
    page_number  INTEGER,
    section      TEXT,
    char_start   INTEGER NOT NULL,
    char_end     INTEGER NOT NULL,
    text         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_key ON chunks(chunking_key, doc_id);

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id   TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    model_name TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    vector     BLOB NOT NULL,
    PRIMARY KEY (chunk_id, model_name)
);

CREATE TABLE IF NOT EXISTS app_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS query_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at     TEXT NOT NULL,
    question       TEXT NOT NULL,
    answer         TEXT,
    status         TEXT NOT NULL,
    provider       TEXT,
    model          TEXT,
    config_json    TEXT NOT NULL,
    citations_json TEXT NOT NULL DEFAULT '[]',
    retrieved_json TEXT NOT NULL DEFAULT '[]',
    timings_json   TEXT NOT NULL DEFAULT '{}',
    usage_json     TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS benchmarks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    is_sample   INTEGER NOT NULL DEFAULT 0,
    items_json  TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL,
    kind         TEXT NOT NULL,
    benchmark    TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    finished_at  TEXT,
    status       TEXT NOT NULL,
    grid_json    TEXT NOT NULL,
    env_json     TEXT NOT NULL,
    notes        TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS experiment_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    label         TEXT NOT NULL,
    config_json   TEXT NOT NULL,
    metrics_json  TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_results (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER NOT NULL REFERENCES experiment_runs(id) ON DELETE CASCADE,
    qid          TEXT NOT NULL,
    question     TEXT NOT NULL,
    result_json  TEXT NOT NULL
);
"""


class Database:
    """Thin wrapper around a SQLite file with per-call connections.

    A new connection per operation keeps things safe across Streamlit's script
    threads; SQLite handles the file locking. Foreign keys are enabled on every
    connection so deleting a document cascades to its chunks and embeddings.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.RLock()
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._write_lock:
            conn = self.connect()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            yield conn
        finally:
            conn.close()

    def initialize(self) -> None:
        with self.transaction() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(SCHEMA)
            conn.execute(
                "INSERT OR IGNORE INTO app_state(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
            )
            conn.execute("INSERT OR IGNORE INTO app_state(key, value) VALUES ('corpus_version', '0')")
