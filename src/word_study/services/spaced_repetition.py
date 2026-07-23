"""
SM-2 spaced repetition algorithm service.
"""
from __future__ import annotations

from datetime import date, timedelta
from word_study.utils.logging_utils import get_logger
from word_study.services.sqlite_client import _get_conn

_logger = get_logger(__name__)


def init_review_table() -> None:
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS review_schedule (
            word        TEXT PRIMARY KEY,
            ef          REAL NOT NULL DEFAULT 2.5,
            interval    REAL NOT NULL DEFAULT 1.0,
            reps        INTEGER NOT NULL DEFAULT 0,
            next_date   TEXT NOT NULL DEFAULT '',
            last_review TEXT NOT NULL DEFAULT ''
        )
    """)
    conn.commit()
    _logger.debug("review_schedule table ready")


def add_review(word: str, quality: int) -> dict:
    quality = max(0, min(5, quality))
    today = date.today()
    conn = _get_conn()
    init_review_table()

    row = conn.execute(
        "SELECT ef, interval, reps FROM review_schedule WHERE word = ?",
        (word,),
    ).fetchone()

    if row:
        ef, interval, reps = row["ef"], row["interval"], row["reps"]
    else:
        ef, interval, reps = 2.5, 0.0, 0

    if quality < 3:
        reps = 0
        interval = 1.0
    else:
        if reps == 0:
            interval = 1.0
        elif reps == 1:
            interval = 6.0
        else:
            interval = interval * ef
        reps += 1

    ef = ef + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    if ef < 1.3:
        ef = 1.3

    next_date = (today + timedelta(days=max(1, int(interval)))).isoformat()

    conn.execute("""
        INSERT OR REPLACE INTO review_schedule (word, ef, interval, reps, next_date, last_review)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (word, ef, interval, reps, next_date, today.isoformat()))
    conn.commit()

    _logger.info("SM-2 review | word=%s quality=%d ef=%.2f interval=%.0fd reps=%d next=%s",
        word, quality, ef, interval, reps, next_date)

    return {"word": word, "ef": ef, "interval": interval, "reps": reps, "next_date": next_date}


def get_due_words(limit: int = 20) -> list[dict]:
    today = date.today().isoformat()
    conn = _get_conn()
    init_review_table()

    rows = conn.execute("""
        SELECT word, ef, interval, reps
        FROM review_schedule
        WHERE next_date <= ? OR next_date = ''
        ORDER BY
            CASE WHEN reps = 0 THEN 0 ELSE 1 END,
            next_date ASC
        LIMIT ?
    """, (today, limit)).fetchall()

    return [
        {"word": r["word"], "ef": r["ef"], "interval": r["interval"], "reps": r["reps"]}
        for r in rows
    ]


def get_today_pending_count() -> int:
    today = date.today().isoformat()
    conn = _get_conn()
    init_review_table()
    row = conn.execute(
        "SELECT COUNT(*) AS cnt FROM review_schedule WHERE next_date <= ? OR next_date = ''",
        (today,),
    ).fetchone()
    return row["cnt"] if row else 0
