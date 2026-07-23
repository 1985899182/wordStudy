"""
Neo4j 图数据库客户端 —— 单例模式，复用连接。

v2: 底层状态托管给 deps.py，本模块作为兼容性 facade 重新导出.
"""
from word_study.deps import get_graph, reset_all

# 保持原有导出 API 不变
__all__ = ["get_graph", "reset_all"]
