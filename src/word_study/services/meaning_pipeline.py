"""释义归并管线 - 四级去重策略的核心实现。

原嵌入在 word_router.py 的 _ensure_meaning_node 函数（约 120 行），
现在抽为独立的 MeaningPipeline 类，便于:
  - 单元测试每一级归并逻辑
  - 在任何调用方复用管线
  - 管线流程清晰: L1 精确 -> L2 跨词 -> L3 语义 -> L4 新建
"""
from __future__ import annotations

from typing import Protocol

from word_study.utils.logging_utils import get_logger
from word_study.utils.meaning_tools import (
    find_meaning_node, find_meaning_cross_word, attach_word_to_meaning,
    update_mean_query_times, add_means_to_list, create_meaning_node,
)
from word_study.utils.relation_tools import create_word_synonym_rel
from word_study.config import get_settings

_logger = get_logger(__name__)


class DatamuseProvider(Protocol):
    """Datamuse 英文释义获取器接口."""
    async def fetch_definitions(self, word: str) -> list[str]: ...

class LLMProvider(Protocol):
    """LLM 服务接口."""
    async def select_best_definition(self, word: str, chinese_meaning: str, english_defs: list[str]) -> str: ...
    async def judge_meaning_merge(self, existing: list[str], new: str) -> tuple: ...

class EmbeddingProvider(Protocol):
    """向量嵌入与搜索接口."""
    def ensure_vector_index(self) -> None: ...
    async def embed_text(self, text: str) -> list[float]: ...
    def search_similar_meanings(self, text: str, k: int = 5) -> list[dict]: ...


class MeaningPipeline:
    """四级释义归并管线.

    Usage:
        pipeline = create_meaning_pipeline()
        result = await pipeline.ensure_meaning("happy", "adjective", "Adjective", "高兴的")
    """

    def __init__(self, datamuse: DatamuseProvider, llm: LLMProvider, embedding: EmbeddingProvider) -> None:
        self._datamuse = datamuse
        self._llm = llm
        self._embedding = embedding
        self._threshold = get_settings().vector_similarity_threshold

    async def ensure_meaning(self, word: str, pos: str, pos_label: str, meaning: str) -> str:
        """四级管线入口: L1 -> L2 -> L3 -> L4, 每级命中则短路."""
        _logger.debug("MeaningPipeline | word=%s label=%s meaning=%s", word, pos_label, meaning)
        if r := self._step_exact_same_word(word, pos_label, meaning): return r
        if r := self._step_exact_cross_word(word, pos_label, meaning): return r
        if r := await self._step_semantic(word, pos_label, meaning): return r
        return self._step_create(word, pos_label, meaning)

    def _step_exact_same_word(self, word: str, pos_label: str, meaning: str) -> str | None:
        """L1: 同词同词性精确匹配, 命中则 query_times +1."""
        node = find_meaning_node(word, pos_label, meaning)
        if node:
            update_mean_query_times(word, pos_label, meaning, 1)
            _logger.debug("L1 hit | %s:%s", word, meaning)
            return meaning
        return None

    def _step_exact_cross_word(self, word: str, pos_label: str, meaning: str) -> str | None:
        """L2: 跨词精确匹配, 共享 Meaning 节点 + 建立 SYNONYM."""
        for match in find_meaning_cross_word(pos_label, meaning):
            mw = (match.get("word_name") or "").lower()
            nid = match.get("node_id", "")
            if mw != word.lower() and nid:
                attach_word_to_meaning(word, nid, meaning)
                create_word_synonym_rel(word, mw)
                _logger.info("L2 cross-word | %s <-> %s share=%s", word, mw, meaning)
                return meaning
        return None

    async def _step_semantic(self, word: str, pos_label: str, meaning: str) -> str | None:
        """L3: Datamuse -> LLM 选最佳 -> 向量嵌入 -> 相似度搜索 -> LLM 确认."""
        defs = await self._datamuse.fetch_definitions(word)
        if not defs: return None
        plot = await self._llm.select_best_definition(word, meaning, defs)
        if not plot: return None

        self._embedding.ensure_vector_index()
        try:
            sims = self._embedding.search_similar_meanings(plot, k=3)
        except Exception as exc:
            _logger.warning("L3 vector search failed | %s", exc)
            return None

        for s in sims:
            if s.get("score", 0) < self._threshold: continue
            sw = (s.get("word") or "").lower()
            snid = s.get("node_id", "")
            smeans = s.get("means", [])
            if not sw or not snid: continue

            if sw == word.lower():
                t = smeans[0] if smeans else ""
                if t:
                    add_means_to_list(word, pos_label, t, meaning)
                    update_mean_query_times(word, pos_label, t, 1)
                    _logger.info("L3 same-word | %s -> %s", meaning, t)
                    return t
                continue

            ok, _, reason = await self._llm.judge_meaning_merge(smeans, meaning)
            if ok:
                attach_word_to_meaning(word, snid, meaning)
                create_word_synonym_rel(word, sw)
                _logger.info("L3 merge | %s <-> %s score=%s", word, sw, s.get("score"))
                return meaning
            _logger.info("L3 rejected | reason=%s", reason)
        return None

    def _step_create(self, word: str, pos_label: str, meaning: str,
                     plot: str = "", embedding: list[float] | None = None) -> str:
        """L4: 创建新释义节点."""
        if plot and embedding:
            create_meaning_node(word, pos_label, meaning, plot=plot, plot_embedding=embedding)
        else:
            create_meaning_node(word, pos_label, meaning)
        _logger.info("L4 create | %s[%s]=%s", word, pos_label, meaning)
        return meaning


def create_meaning_pipeline() -> MeaningPipeline:
    """用真实依赖构建 MeaningPipeline.

    导入在函数体内延迟执行，避免模块顶层循环导入.
    底层依赖由 deps.py 统一管理，测试时可 override.
    """
    from word_study.services.datamuse_service import fetch_definitions
    from word_study.deps import (
        ensure_vector_index, embed_text, search_similar_meanings,
    )
    # LLM 方法仍从 llm_service 获取（含 retry 装饰器）
    from word_study.services.llm_service import select_best_definition, judge_meaning_merge

    class _DatamuseAdapter:
        async def fetch_definitions(self, w): return await fetch_definitions(w)
    class _LLMAdapter:
        async def select_best_definition(self, w, c, e): return await select_best_definition(w, c, e)
        async def judge_meaning_merge(self, e, n): return await judge_meaning_merge(e, n)
    class _EmbeddingAdapter:
        def ensure_vector_index(self): ensure_vector_index()
        async def embed_text(self, t): return await embed_text(t)
        def search_similar_meanings(self, t, k=5): return search_similar_meanings(t, k)

    return MeaningPipeline(_DatamuseAdapter(), _LLMAdapter(), _EmbeddingAdapter())
