"""
MeaningPipeline 单元测试 —— 四级归并管线，每级可独立验证.

全部使用 mock 依赖，不依赖真实 Neo4j / LLM / Datamuse.
"""
from __future__ import annotations

import sys
from pathlib import Path

_src = Path(__file__).resolve().parents[3] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from word_study.services.meaning_pipeline import MeaningPipeline


@pytest.fixture
def mock_datamuse() -> AsyncMock:
    m = AsyncMock()
    m.fetch_definitions.return_value = ["a state of well-being", "feeling pleasure"]
    return m

@pytest.fixture
def mock_llm() -> AsyncMock:
    m = AsyncMock()
    m.select_best_definition.return_value = "feeling pleasure"
    m.judge_meaning_merge.return_value = (False, -1, "not similar")
    return m

@pytest.fixture
def mock_embedding() -> MagicMock:
    m = MagicMock()
    m.ensure_vector_index = MagicMock()
    m.embed_text = AsyncMock(return_value=[0.1] * 1536)
    m.search_similar_meanings.return_value = []
    return m

@pytest.fixture
def pipeline(mock_datamuse, mock_llm, mock_embedding):
    return MeaningPipeline(mock_datamuse, mock_llm, mock_embedding)


class TestL1ExactSameWord:
    @patch("word_study.services.meaning_pipeline.find_meaning_node")
    @patch("word_study.services.meaning_pipeline.update_mean_query_times")
    def test_hit_returns_meaning(self, mock_update, mock_find, pipeline):
        mock_find.return_value = {"node_id": "abc", "means": ["高兴的"]}
        result = pipeline._step_exact_same_word("happy", "Adjective", "高兴的")
        assert result == "高兴的"
        mock_update.assert_called_once_with("happy", "Adjective", "高兴的", 1)

    @patch("word_study.services.meaning_pipeline.find_meaning_node")
    def test_miss_returns_none(self, mock_find, pipeline):
        mock_find.return_value = None
        assert pipeline._step_exact_same_word("happy", "Adjective", "新释义") is None


class TestL2ExactCrossWord:
    @patch("word_study.services.meaning_pipeline.find_meaning_cross_word")
    @patch("word_study.services.meaning_pipeline.attach_word_to_meaning")
    @patch("word_study.services.meaning_pipeline.create_word_synonym_rel")
    def test_cross_word_merge(self, mock_syn, mock_attach, mock_find, pipeline):
        mock_find.return_value = [{"word_name": "cheerful", "node_id": "n456"}]
        result = pipeline._step_exact_cross_word("happy", "Adjective", "高兴的")
        assert result == "高兴的"
        mock_attach.assert_called_once()
        mock_syn.assert_called_once_with("happy", "cheerful")

    @patch("word_study.services.meaning_pipeline.find_meaning_cross_word")
    def test_same_word_skipped(self, mock_find, pipeline):
        mock_find.return_value = [{"word_name": "happy", "node_id": "n123"}]
        assert pipeline._step_exact_cross_word("happy", "Adjective", "高兴的") is None


class TestL3Semantic:
    @pytest.mark.asyncio
    async def test_no_definitions(self, pipeline, mock_datamuse):
        mock_datamuse.fetch_definitions.return_value = []
        assert await pipeline._step_semantic("xyz", "Noun", "x") is None

    @pytest.mark.asyncio
    async def test_no_similar_results(self, pipeline):
        assert await pipeline._step_semantic("happy", "Adjective", "高兴的") is None

    @pytest.mark.asyncio
    async def test_same_word_merge(self, pipeline, mock_embedding, mock_llm):
        mock_embedding.search_similar_meanings.return_value = [
            {"word": "happy", "node_id": "a", "means": ["快乐的"], "score": 0.90}
        ]
        with patch("word_study.services.meaning_pipeline.add_means_to_list"),              patch("word_study.services.meaning_pipeline.update_mean_query_times"):
            result = await pipeline._step_semantic("happy", "Adjective", "高兴的")
            assert result == "快乐的"

    @pytest.mark.asyncio
    async def test_cross_word_llm_confirmed(self, pipeline, mock_embedding, mock_llm):
        mock_embedding.search_similar_meanings.return_value = [
            {"word": "cheerful", "node_id": "x", "means": ["愉快的"], "score": 0.88}
        ]
        mock_llm.judge_meaning_merge.return_value = (True, 0, "similar")
        with patch("word_study.services.meaning_pipeline.attach_word_to_meaning"),              patch("word_study.services.meaning_pipeline.create_word_synonym_rel"):
            result = await pipeline._step_semantic("happy", "Adjective", "高兴的")
            assert result == "高兴的"

    @pytest.mark.asyncio
    async def test_llm_rejected(self, pipeline, mock_embedding, mock_llm):
        mock_embedding.search_similar_meanings.return_value = [
            {"word": "sad", "node_id": "y", "means": ["悲伤的"], "score": 0.87}
        ]
        mock_llm.judge_meaning_merge.return_value = (False, -1, "different")
        assert await pipeline._step_semantic("happy", "Adjective", "高兴的") is None


class TestL4Create:
    @patch("word_study.services.meaning_pipeline.create_meaning_node")
    def test_create_simple(self, mock_create, pipeline):
        assert pipeline._step_create("new", "Noun", "新词") == "新词"
        mock_create.assert_called_once_with("new", "Noun", "新词")

    @patch("word_study.services.meaning_pipeline.create_meaning_node")
    def test_create_with_plot(self, mock_create, pipeline):
        result = pipeline._step_create("happy", "Adjective", "高兴的", plot="feeling pleasure", embedding=[0.5]*10)
        assert result == "高兴的"
        mock_create.assert_called_once()


class TestEnsureMeaning:
    @pytest.mark.asyncio
    async def test_l1_shortcuts(self, pipeline):
        with patch.object(pipeline, "_step_exact_same_word", return_value="hit"),              patch.object(pipeline, "_step_exact_cross_word") as l2,              patch.object(pipeline, "_step_semantic") as l3,              patch.object(pipeline, "_step_create") as l4:
            assert await pipeline.ensure_meaning("x", "n", "Noun", "x") == "hit"
            l2.assert_not_called(); l3.assert_not_called(); l4.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_miss_to_l4(self, pipeline):
        with patch.object(pipeline, "_step_exact_same_word", return_value=None),              patch.object(pipeline, "_step_exact_cross_word", return_value=None),              patch.object(pipeline, "_step_semantic", return_value=None),              patch.object(pipeline, "_step_create", return_value="l4"):
            assert await pipeline.ensure_meaning("u", "n", "Noun", "u") == "l4"
