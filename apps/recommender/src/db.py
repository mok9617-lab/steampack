from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS games (
  app_id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  release_date TEXT,
  genres TEXT,
  tags TEXT,
  positive_ratio REAL,
  review_count INTEGER,
  updated_at TEXT
);

CREATE TABLE IF NOT EXISTS reviews (
  review_id TEXT PRIMARY KEY,
  app_id INTEGER NOT NULL,
  language TEXT NOT NULL,
  review_text TEXT NOT NULL,
  cleaned_text TEXT,
  voted_up INTEGER NOT NULL,
  votes_up INTEGER,
  weighted_vote_score REAL,
  steam_purchase INTEGER,
  received_for_free INTEGER,
  playtime_forever INTEGER,
  playtime_at_review INTEGER,
  review_date TEXT,
  sentiment_score REAL,
  trust_label TEXT,
  trust_reasons TEXT,
  embedding BLOB,
  FOREIGN KEY (app_id) REFERENCES games(app_id)
);

CREATE TABLE IF NOT EXISTS game_profiles (
  app_id INTEGER PRIMARY KEY,
  profile_embedding BLOB,
  top_keywords TEXT,
  mood_tags TEXT,
  recent_review_count INTEGER,
  positive_ratio_1y REAL,
  median_playtime_1y REAL,
  updated_at TEXT,
  FOREIGN KEY (app_id) REFERENCES games(app_id)
);

CREATE TABLE IF NOT EXISTS eval_queries (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  query_text TEXT NOT NULL,
  expected_constraints TEXT,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS eval_results (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  eval_query_id INTEGER NOT NULL,
  app_id INTEGER NOT NULL,
  rank INTEGER NOT NULL,
  relevance_score INTEGER,
  reason_quality_score INTEGER,
  notes TEXT,
  created_at TEXT,
  FOREIGN KEY (eval_query_id) REFERENCES eval_queries(id),
  FOREIGN KEY (app_id) REFERENCES games(app_id)
);
"""


def get_connection(db_path: Path, readonly: bool = False) -> sqlite3.Connection:
    # Use read-only URI connection for query-only paths to reduce lock contention.
    if readonly:
        db_uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
        conn = sqlite3.connect(db_uri, uri=True, timeout=30.0)
    else:
        conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(db_path: Path) -> None:
    with get_connection(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.executescript(SCHEMA_SQL)
        conn.commit()


def reset_db_data(db_path: Path) -> None:
    with get_connection(db_path) as conn:
        conn.executescript(
            """
            DELETE FROM eval_results;
            DELETE FROM eval_queries;
            DELETE FROM game_profiles;
            DELETE FROM reviews;
            DELETE FROM games;
            """
        )
        conn.commit()
