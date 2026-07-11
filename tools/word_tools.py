"""
Word 节点相关 Cypher 工具函数 —— 每个函数对应一种 Cypher 操作。
"""
from models.neo4j_client import get_graph
from services.log_service import log_cypher


def init_alphabet_nodes() -> list[dict]:
    """MERGE 创建 26 个 Alphabet 节点（a-z）。"""
    graph = get_graph()
    cypher = """
    UNWIND ['a','b','c','d','e','f','g','h','i','j','k','l','m',
            'n','o','p','q','r','s','t','u','v','w','x','y','z'] AS letter
    MERGE (a:Alphabet {name: letter})
    RETURN a.name AS name
    """
    log_cypher(cypher, {})
    return graph.query(cypher)


def search_word(name: str) -> list[dict]:
    """查询单词节点是否存在及其基本信息。"""
    graph = get_graph()
    cypher = """
    MATCH (w:Word {name: $name})
    OPTIONAL MATCH (w)-[t:TRANSLATION_INTO]->(m)
    RETURN w.name AS name,
           w.prefix AS prefix,
           w.word_query_times AS word_query_times,
           w.resource AS resource,
           labels(m) AS pos_labels,
           m.means AS means,
           t.query_times AS trans_query_times
    """
    log_cypher(cypher, {"name": name})
    return graph.query(cypher, params={"name": name})


def search_word_batch(names: list[str]) -> list[dict]:
    """批量查询多个单词。"""
    graph = get_graph()
    cypher = """
    UNWIND $names AS word_name
    OPTIONAL MATCH (w:Word {name: word_name})
    OPTIONAL MATCH (w)-[t:TRANSLATION_INTO]->(m)
    RETURN word_name AS name,
           w IS NOT NULL AS exists,
           w.word_query_times AS word_query_times,
           w.resource AS resource,
           labels(m) AS pos_labels,
           m.means AS means,
           t.query_times AS trans_query_times
    """
    log_cypher(cypher, {"names": names})
    return graph.query(cypher, params={"names": names})


def create_word_node(name: str, prefix: str, resource: str) -> list[dict]:
    """创建 Word 节点，并与 Alphabet 建立 PREFIX_TO 关系。"""
    graph = get_graph()
    cypher = """
    MERGE (w:Word {name: $name})
    ON CREATE SET w.prefix = $prefix,
                  w.word_query_times = 1,
                  w.resource = [$resource]
    ON MATCH SET w.word_query_times = w.word_query_times + 1
    WITH w
    MATCH (a:Alphabet {name: $prefix})
    MERGE (a)-[:PREFIX_TO]->(w)
    RETURN w.name AS name, w.word_query_times AS word_query_times, w.resource AS resource
    """
    log_cypher(cypher, {"name": name, "prefix": prefix, "resource": resource})
    return graph.query(cypher, params={"name": name, "prefix": prefix, "resource": resource})


def update_word_query_times(name: str, delta: int = 1) -> list[dict]:
    """单词查询次数 += delta（可为正/负）。"""
    graph = get_graph()
    cypher = """
    MATCH (w:Word {name: $name})
    SET w.word_query_times = w.word_query_times + $delta
    RETURN w.name AS name, w.word_query_times AS word_query_times
    """
    log_cypher(cypher, {"name": name, "delta": delta})
    return graph.query(cypher, params={"name": name, "delta": delta})


def append_word_resource(name: str, resource: str) -> list[dict]:
    """向 Word.resource 列表追加新来源（去重）。"""
    graph = get_graph()
    cypher = """
    MATCH (w:Word {name: $name})
    WITH w, CASE WHEN $resource IN w.resource THEN w.resource
                 ELSE w.resource + $resource END AS new_resource
    SET w.resource = new_resource
    RETURN w.name AS name, w.resource AS resource
    """
    log_cypher(cypher, {"name": name, "resource": resource})
    return graph.query(cypher, params={"name": name, "resource": resource})
