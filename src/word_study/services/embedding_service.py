"""
向量嵌入服务 —— DashScopeEmbeddings + Neo4jVector 向量搜索。

v2: 底层状态托管给 deps.py，本模块作为兼容性 facade 重新导出.
"""
from __future__ import annotations

from word_study.deps import (
    get_embedding_model,
    ensure_vector_index,
    get_plot_vector,
    embed_text,
    search_similar_meanings,
    VECTOR_INDEX_NAME,
    VECTOR_DIMENSIONS,
)

__all__ = [
    "get_embedding_model", "ensure_vector_index", "get_plot_vector",
    "embed_text", "search_similar_meanings",
    "VECTOR_INDEX_NAME", "VECTOR_DIMENSIONS",
]
