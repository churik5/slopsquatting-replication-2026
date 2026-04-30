"""SQLite schema and resume logic for generations, heuristics, validations.

Tables
------
generations       one row per (prompt_id, model_key, n_sample)
heuristic_hits    package names extracted by H1/H2/H3 per generation
validations       pypi / npm membership results per (package_name, source)
api_calls         one row per LiteLLM call, for spend audit

WAL mode + UNIQUE index makes resume safe.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .config import DB_PATH

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS generations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_id       TEXT NOT NULL,        -- "<lang>:<dataset>:<row_index>"
    language        TEXT NOT NULL,
    dataset         TEXT NOT NULL,        -- SO_AT | SO_LY | LLM_AT | LLM_LY
    model_key       TEXT NOT NULL,
    n_sample        INTEGER NOT NULL DEFAULT 0,
    temperature     REAL    NOT NULL,
    top_p           REAL    NOT NULL,
    max_tokens      INTEGER NOT NULL,
    prompt_text     TEXT NOT NULL,
    generated_code  TEXT NOT NULL,        -- preserved verbatim, never overwritten
    finish_reason   TEXT,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    cost_usd        REAL,
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(prompt_id, model_key, n_sample)
);

CREATE TABLE IF NOT EXISTS heuristic_hits (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    generation_id   INTEGER NOT NULL REFERENCES generations(id) ON DELETE CASCADE,
    heuristic       TEXT NOT NULL,        -- H1 | H2 | H3
    package_raw     TEXT NOT NULL,
    package_norm    TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_heuristic_hits_gen ON heuristic_hits(generation_id);

CREATE TABLE IF NOT EXISTS validations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    package_norm    TEXT NOT NULL,
    registry        TEXT NOT NULL,        -- pypi | npm
    exists_in_master INTEGER NOT NULL,    -- 0/1 vs local master list
    live_status     INTEGER,              -- HTTP status from live HEAD, nullable
    false_positive  INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(package_norm, registry)
);

CREATE TABLE IF NOT EXISTS api_calls (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    generation_id   INTEGER REFERENCES generations(id) ON DELETE CASCADE,
    provider        TEXT NOT NULL,
    model_key       TEXT NOT NULL,
    endpoint        TEXT NOT NULL,        -- code_gen | h2 | h3
    input_tokens    INTEGER NOT NULL,
    output_tokens   INTEGER NOT NULL,
    batch           INTEGER NOT NULL DEFAULT 0,
    cost_usd        REAL NOT NULL,
    latency_ms      INTEGER,
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


@contextmanager
def cursor(path: Path = DB_PATH):
    conn = connect(path)
    try:
        yield conn.cursor()
    finally:
        conn.close()
