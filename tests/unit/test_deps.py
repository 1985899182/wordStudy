"""
deps.py 依赖注入测试.
"""
from __future__ import annotations

import sys
from pathlib import Path

_src = Path(__file__).resolve().parents[3] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import pytest

from word_study.deps import reset_all


class TestDepsReset:
    def test_reset_all_clears_singletons(self):
        """reset_all() 后所有单例变量置 None."""
        reset_all()
        from word_study.deps import _graph, _llm, _embedding_model, _plot_vector, _cypher_qa
        assert _graph is None
        assert _llm is None
        assert _embedding_model is None
        assert _plot_vector is None
        assert _cypher_qa is None
