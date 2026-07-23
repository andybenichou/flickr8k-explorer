"""SQLite access layer.

The dataset is small (8k images, 40k captions) so a single SQLite file holds all
the structured data. Full-text search over captions uses the FTS5 extension that
ships with the stdlib build of SQLite, which keeps the dependency list short.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id          TEXT PRIMARY KEY,
    split       TEXT NOT NULL,
    row_index   INTEGER NOT NULL UNIQUE,  -- position in embeddings.npy
    width       INTEGER NOT NULL,
    height      INTEGER NOT NULL,
    file_size   INTEGER NOT NULL,
    umap_x      REAL,
    umap_y      REAL,
    cluster     INTEGER
);
CREATE INDEX IF NOT EXISTS idx_images_split ON images(split);

CREATE TABLE IF NOT EXISTS captions (
    image_id      TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    caption_index INTEGER NOT NULL,
    text          TEXT NOT NULL,
    n_words       INTEGER NOT NULL,
    PRIMARY KEY (image_id, caption_index)
);

CREATE VIRTUAL TABLE IF NOT EXISTS captions_fts USING fts5(
    text,
    image_id UNINDEXED,
    tokenize = 'porter unicode61'
);

CREATE TABLE IF NOT EXISTS clusters (
    id      INTEGER PRIMARY KEY,   -- KMeans label
    label   TEXT NOT NULL,         -- human-readable, from distinctive caption words
    size    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def connect(db_path: Path, *, read_only: bool = False) -> sqlite3.Connection:
    """Open a connection with sane defaults for this workload."""
    if read_only:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    else:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode = WAL")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, str(value)),
    )


def get_meta(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()
