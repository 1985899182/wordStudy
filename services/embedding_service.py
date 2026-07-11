"""
向量嵌入服务 —— DashScopeEmbeddings + Neo4jVector 向量搜索。

参照 genai-integration-langchain/vector_graph_retriever.py 的实现方式。
"""
from __future__ import annotations

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_neo4j import Neo4jVector

from config import DASHSCOPE_API_KEY
from models.neo4j_client import get_graph

# ── 单例 ─────────────────────────────────────────────────

_embedding_model: DashScopeEmbeddings | None = None
_plot_vector: Neo4jVector | None = None

VECTOR_INDEX_NAME = "meaning_plot_embedding"
VECTOR_DIMENSIONS = 1536  # DashScope text-embedding-v2 输出维度


def get_embedding_model() -> DashScopeEmbeddings:
    """获取 DashScopeEmbeddings 单例。"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = DashScopeEmbeddings(
            model="text-embedding-v2",
            dashscope_api_key=DASHSCOPE_API_KEY,
        )
    return _embedding_model


def ensure_vector_index() -> None:
    """确保 Neo4j 中存在向量索引（不存在则创建）。

    索引在 :Meaning 标签的 plotEmbedding 属性上。
    创建后重置 Neo4jVector 缓存，确保后续搜索使用新索引。
    """
    global _plot_vector
    graph = get_graph()
    cypher = f"""
    CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
    FOR (m:Meaning) ON (m.plotEmbedding)
    OPTIONS {{indexConfig: {{`vector.dimensions`: {VECTOR_DIMENSIONS}, `vector.similarity_function`: 'cosine'}}}}
    """
    try:
        graph.query(cypher)
        print(f"[INFO] 向量索引 {VECTOR_INDEX_NAME} 已就绪")
    except Exception as exc:
        print(f"[WARN] 向量索引创建异常: {exc}")
    # 重置缓存，下次 get_plot_vector() 会基于最新索引重建
    _plot_vector = None


def get_plot_vector() -> Neo4jVector:
    """获取 Neo4jVector 单例，用于向量相似度搜索。"""
    global _plot_vector
    if _plot_vector is None:
        graph = get_graph()
        emb = get_embedding_model()

        retrieval_query = """
        MATCH (node)<-[r:TRANSLATION_INTO]-(w:Word)
        RETURN
            "Word: " + w.name + ", Plot: " + node.plot AS text,
            score,
            {
                word: w.name,
                means: node.means,
                labels: labels(node),
                node_id: elementId(node)
            } AS metadata
        """

        _plot_vector = Neo4jVector.from_existing_index(
            emb,
            graph=graph,
            index_name=VECTOR_INDEX_NAME,
            embedding_node_property="plotEmbedding",
            text_node_property="plot",
            retrieval_query=retrieval_query,
        )
    return _plot_vector


async def embed_text(text: str) -> list[float]:
    """将文本转为向量嵌入。"""
    emb = get_embedding_model()
    result = await emb.aembed_query(text)
    return result


def search_similar_meanings(query_text: str, k: int = 5) -> list[dict]:
    """向量相似度搜索，返回与 query_text 最相似的释义节点列表。

    Returns: [{"word": str, "means": list[str], "labels": list[str], "node_id": str, "score": float}, ...]
    """
    vector = get_plot_vector()
    docs = vector.similarity_search_with_score(query_text, k=k)
    results: list[dict] = []
    for doc, score in docs:
        meta = doc.metadata or {}
        results.append({
            "word": meta.get("word", ""),
            "means": meta.get("means", []),
            "labels": meta.get("labels", []),
            "node_id": meta.get("node_id", ""),
            "score": score,
        })
    return results
