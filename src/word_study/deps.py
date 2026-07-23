"""
FastAPI 依赖注入层 —— 替代全局模块级单例。

Why Depends over global singletons:
  1. 测试时可以 app.dependency_overrides 一键换成 mock，不需要真实 Neo4j/LLM
  2. 资源生命周期可控：FastAPI lifespan 统一管理连接创建/销毁
  3. 并发安全：每个请求可以拿到独立的资源实例（如果需要）

Usage:
  from word_study.deps import get_graph, get_llm, get_embedding_model

  @router.post("/api/ai-query")
  async def ai_query(
      req: AIQueryRequest,
      graph: Neo4jGraph = Depends(get_graph),
      llm: BaseChatModel = Depends(get_llm),
  ):
      ...

Testing:
  from word_study.deps import override_graph, override_llm
  app.dependency_overrides[get_graph] = override_graph
"""
from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_neo4j import Neo4jVector
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache

from word_study.config import get_settings, BASE_DIR
from word_study.utils.logging_utils import get_logger

_logger = get_logger(__name__)

# ── 全局 LLM 缓存（应用级，非请求级）──
set_llm_cache(InMemoryCache())

# ══════════════════════════════════════════════════════════
# 内部状态 —— 惰性初始化，首次 Depends 调用时创建
# ══════════════════════════════════════════════════════════

_graph: Neo4jGraph | None = None
_llm = None  # BaseChatModel
_embedding_model: DashScopeEmbeddings | None = None
_plot_vector: Neo4jVector | None = None
_cypher_qa: GraphCypherQAChain | None = None

# 常量
VECTOR_INDEX_NAME = "meaning_plot_embedding"
VECTOR_DIMENSIONS = 1536


# ══════════════════════════════════════════════════════════
# Neo4j Graph
# ══════════════════════════════════════════════════════════

def get_graph() -> Neo4jGraph:
    """获取 Neo4jGraph 惰性单例.

    首次调用时连接 Neo4j，后续返回缓存实例.
    测试时通过 app.dependency_overrides 替换.
    """
    global _graph
    if _graph is None:
        s = get_settings()
        _logger.info("Neo4jGraph 初始化 | uri=%s db=%s", s.neo4j_uri, s.neo4j_database)
        _graph = Neo4jGraph(
            url=s.neo4j_uri,
            username=s.neo4j_username,
            password=s.neo4j_password,
            database=s.neo4j_database,
        )
    return _graph


# ══════════════════════════════════════════════════════════
# LLM
# ══════════════════════════════════════════════════════════

def get_llm():
    """获取 DeepSeek ChatModel 惰性单例."""
    global _llm
    if _llm is None:
        s = get_settings()
        _logger.info("DeepSeek ChatModel 初始化 | model=%s", s.deepseek_model)
        _llm = init_chat_model(s.deepseek_model, model_provider="deepseek")
    return _llm


def get_cypher_qa() -> GraphCypherQAChain:
    """获取 GraphCypherQAChain 惰性单例，接入只读 prompt."""
    global _cypher_qa
    if _cypher_qa is None:
        graph = get_graph()
        llm = get_llm()

        # 加载只读 prompt 模板
        prompt_path = BASE_DIR / "prompts" / "read_only_cypher.txt"
        cypher_prompt = None
        if prompt_path.exists():
            from langchain_core.prompts import PromptTemplate
            cypher_prompt = PromptTemplate(
                input_variables=["schema", "question"],
                template=prompt_path.read_text(encoding="utf-8"),
            )
            _logger.info("已接入只读 Cypher prompt")

        _logger.info("GraphCypherQAChain 初始化")
        _cypher_qa = GraphCypherQAChain.from_llm(
            graph=graph, llm=llm,
            allow_dangerous_requests=True, return_intermediate_steps=True, validate_cypher=True,
            cypher_prompt=cypher_prompt,
            verbose=True,
        )
    return _cypher_qa


# ══════════════════════════════════════════════════════════
# Embedding / Vector
# ══════════════════════════════════════════════════════════

def get_embedding_model() -> DashScopeEmbeddings:
    """获取 DashScopeEmbeddings 惰性单例."""
    global _embedding_model
    if _embedding_model is None:
        s = get_settings()
        _logger.info("DashScopeEmbeddings 初始化 | dim=%d", VECTOR_DIMENSIONS)
        _embedding_model = DashScopeEmbeddings(
            model="text-embedding-v2",
            dashscope_api_key=s.dashscope_api_key,
        )
    return _embedding_model


def ensure_vector_index() -> None:
    """确保 Neo4j 中存在向量索引（不存在则创建）.

    索引在 :Meaning 标签的 plotEmbedding 属性上.
    创建后重置 Neo4jVector 缓存.
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
        _logger.info("向量索引 %s 已就绪", VECTOR_INDEX_NAME)
    except Exception as exc:
        _logger.warning("向量索引创建异常 | error=%s", exc)
    _plot_vector = None


def get_plot_vector() -> Neo4jVector:
    """获取 Neo4jVector 惰性单例，用于向量相似度搜索."""
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
            emb, graph=graph,
            index_name=VECTOR_INDEX_NAME,
            embedding_node_property="plotEmbedding",
            text_node_property="plot",
            retrieval_query=retrieval_query,
        )
    return _plot_vector


async def embed_text(text: str) -> list[float]:
    """将文本转为向量嵌入."""
    emb = get_embedding_model()
    return await emb.aembed_query(text)


def search_similar_meanings(query_text: str, k: int = 5) -> list[dict]:
    """向量相似度搜索.

    Returns:
        [{"word": str, "means": list[str], "labels": list[str], "node_id": str, "score": float}, ...]
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


# ══════════════════════════════════════════════════════════
# 测试辅助: 重置全局状态
# ══════════════════════════════════════════════════════════

def reset_all() -> None:
    """重置所有惰性单例（仅供测试使用）."""
    global _graph, _llm, _embedding_model, _plot_vector, _cypher_qa
    _graph = None
    _llm = None
    _embedding_model = None
    _plot_vector = None
    _cypher_qa = None
    _logger.debug("所有依赖单例已重置")
