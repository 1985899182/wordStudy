"""
Neo4j 图数据库客户端 —— 单例模式，复用连接。
"""
from langchain_neo4j import Neo4jGraph
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE

_graph: Neo4jGraph | None = None


def get_graph() -> Neo4jGraph:
    """获取或创建 Neo4jGraph 单例。"""
    global _graph
    if _graph is None:
        _graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database=NEO4J_DATABASE,
        )
    return _graph
