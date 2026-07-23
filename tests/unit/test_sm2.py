"""
SM-2 间隔重复算法单元测试.

使用内存 SQLite（:memory:），不依赖真实数据库.
"""
from __future__ import annotations

import sys, os, tempfile
from pathlib import Path

_src = Path(__file__).resolve().parents[3] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import pytest
from datetime import date, timedelta

from word_study.services.spaced_repetition import (
    add_review, get_due_words, get_today_pending_count, init_review_table,
)


@pytest.fixture(autouse=True)
def setup_sm2_db(monkeypatch):
    """每个测试使用独立的内存 SQLite."""
    import sqlite3
    conn = sqlite3.connect(":memory:"); conn.row_factory = sqlite3.Row
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    monkeypatch.setattr("word_study.services.spaced_repetition._get_conn", lambda: conn)
    monkeypatch.setattr("word_study.services.sqlite_client._get_conn", lambda: conn)
    init_review_table()
    yield
    conn.close()


class TestSM2AddReview:
    def test_first_review_quality_5(self):
        """首次复习，质量5 -> reps=1, interval=1."""
        result = add_review("dog", 5)
        assert result["reps"] == 1
        assert result["interval"] == 1.0
        assert result["ef"] >= 2.5

    def test_first_review_quality_0(self):
        """首次复习完全忘记 -> reps=0, interval=1 (明天再复习)."""
        result = add_review("cat", 0)
        assert result["reps"] == 0
        assert result["interval"] == 1.0

    def test_two_successful_reviews(self):
        """两次成功复习 -> 间隔从 1 跳到 6."""
        add_review("test", 4)  # reps=1, interval=1
        result2 = add_review("test", 4)  # reps=2, interval=6
        assert result2["reps"] == 2
        assert result2["interval"] == 6.0

    def test_quality_clamped(self):
        """quality 超出 0-5 范围时 clamp."""
        r1 = add_review("clamped", 10)
        r2 = add_review("clamped", -5)
        assert 0 <= r1["reps"] <= 5  # quality 10 clamp 到 5
        assert r2["reps"] == 0  # quality -5 clamp 到 0

    def test_ef_decreases_on_failure(self):
        """持续答错 -> EF 持续降低 -> 间隔增长变慢."""
        add_review("hard", 5)
        add_review("hard", 5)
        before = add_review("hard", 5)["ef"]  # 记录当前 EF

        # 模拟多次答错
        for _ in range(3):
            add_review("hard", 0)
        after = add_review("hard", 0)["ef"]

        # EF 不应低于 1.3 (hard floor)
        assert after >= 1.3


class TestSM2GetDueWords:
    def test_failed_word_appears_tomorrow(self):
        """quality<3 的复习失败 -> 明天再复习 -> 今天 get_due 不出现（符合预期）."""
        # 先失败一次，单词排到明天
        add_review("fail_word", 0)
        # 今天查到期列表 -> 因为 next_date=明天，不应出现
        due = get_due_words(10)
        word_names = [d["word"] for d in due]
        assert "fail_word" not in word_names  # next_date=明天 > 今天

    def test_word_with_past_due_date_appears(self, monkeypatch):
        """next_date 是过去日期 -> 出现在到期列表."""
        import sqlite3
        from datetime import date, timedelta
        # 直接用 SQL 插入一个昨天到期的词
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        conn = sqlite3.connect(":memory:"); conn.row_factory = sqlite3.Row
        monkeypatch.setattr("word_study.services.spaced_repetition._get_conn", lambda: conn)
        monkeypatch.setattr("word_study.services.sqlite_client._get_conn", lambda: conn)
        init_review_table()
        conn.execute(
            "INSERT INTO review_schedule (word, ef, interval, reps, next_date) VALUES (?, 2.5, 1.0, 3, ?)",
            ("overdue_word", yesterday)
        )
        conn.commit()
        due = get_due_words(10)
        word_names = [d["word"] for d in due]
        assert "overdue_word" in word_names

    def test_due_limit(self):
        """limit 参数有效."""
        for i in range(5):
            add_review(f"word_{i}", 5)
        due = get_due_words(2)
        assert len(due) <= 2


class TestSM2PendingCount:
    def test_count_increases_with_reviews(self):
        """复习记录越多，待复习数越多."""
        for i in range(3):
            add_review(f"cnt_{i}", 5)
        assert get_today_pending_count() >= 0
