"""
SQLite client - word_log, datamuse_cache tables.
"""
import sqlite3
import threading
from word_study.config import SQLITE_DB_PATH

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(SQLITE_DB_PATH)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
    return _local.conn


def init_db():
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS word_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            word        TEXT    NOT NULL,
            meaning_with_pos TEXT NOT NULL,
            source      TEXT    NOT NULL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS datamuse_cache (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            word         TEXT    NOT NULL,
            api_type     TEXT    NOT NULL,
            response_json TEXT   NOT NULL,
            cached_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(word, api_type)
        );
    """)
    conn.commit()


def insert_word_log(word: str, meaning_with_pos: str, source: str):
    conn = _get_conn()
    conn.execute(
        "INSERT INTO word_log (word, meaning_with_pos, source) VALUES (?, ?, ?)",
        (word, meaning_with_pos, source),
    )
    conn.commit()


def get_cache(word: str, api_type: str) -> str | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT response_json FROM datamuse_cache WHERE word = ? AND api_type = ?",
        (word, api_type),
    ).fetchone()
    return row["response_json"] if row else None


def set_cache(word: str, api_type: str, response_json: str):
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO datamuse_cache (word, api_type, response_json, cached_at) "
        "VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
        (word, api_type, response_json),
    )
    conn.commit()


def get_recent_logs(limit: int = 5) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT word, meaning_with_pos, source, created_at FROM word_log ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [
        {"word": row["word"], "meaning": row["meaning_with_pos"],
         "source": row["source"], "time": row["created_at"]}
        for row in rows
    ]


def _ensure_daily_words_table():
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_words (
            date       TEXT PRIMARY KEY,
            words_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def get_daily_words_today(date_str: str) -> list[dict] | None:
    _ensure_daily_words_table()
    conn = _get_conn()
    row = conn.execute(
        "SELECT words_json FROM daily_words WHERE date = ?", (date_str,)
    ).fetchone()
    if row:
        import json
        return json.loads(row["words_json"])
    return None


def save_daily_words(date_str: str, words: list[dict]):
    _ensure_daily_words_table()
    import json
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO daily_words (date, words_json) VALUES (?, ?)",
        (date_str, json.dumps(words, ensure_ascii=False)),
    )
    conn.commit()
