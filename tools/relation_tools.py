"""
SYNONYM / ANTONYM 关系相关 Cypher 工具函数。
所有关系函数均限定 Word 节点，防止不同单词的同名释义被错误关联。
"""
from models.neo4j_client import get_graph
from services.log_service import log_cypher


# ══════════════════════════════════════════════════════════
# Word 层级关系（Word 直连 SYNONYM/ANTONYM）
# ══════════════════════════════════════════════════════════

def create_word_synonym_rel(word1: str, word2: str) -> list[dict]:
    """在两个 Word 节点间建立 SYNONYM 关系。"""
    graph = get_graph()
    cypher = """
    MATCH (w1:Word {name: $word1})
    MATCH (w2:Word {name: $word2})
    MERGE (w1)-[r:SYNONYM]-(w2)
    RETURN w1.name AS a, w2.name AS b
    """
    log_cypher(cypher, {"word1": word1, "word2": word2})
    return graph.query(cypher, params={"word1": word1, "word2": word2})


def create_word_antonym_rel(word1: str, word2: str) -> list[dict]:
    """在两个 Word 节点间建立 ANTONYM 关系。"""
    graph = get_graph()
    cypher = """
    MATCH (w1:Word {name: $word1})
    MATCH (w2:Word {name: $word2})
    MERGE (w1)-[r:ANTONYM]-(w2)
    RETURN w1.name AS a, w2.name AS b
    """
    log_cypher(cypher, {"word1": word1, "word2": word2})
    return graph.query(cypher, params={"word1": word1, "word2": word2})


def delete_word_rel(word1: str, word2: str, rel_type: str) -> list[dict]:
    """删除两个 Word 节点间的 SYNONYM 或 ANTONYM 关系。"""
    graph = get_graph()
    cypher = f"""
    MATCH (w1:Word {{name: $word1}})-[r:{rel_type}]-(w2:Word {{name: $word2}})
    DELETE r
    RETURN count(r) AS deleted
    """
    log_cypher(cypher, {"word1": word1, "word2": word2})
    return graph.query(cypher, params={"word1": word1, "word2": word2})


# ══════════════════════════════════════════════════════════
# Meaning 层级关系（释义间 SYNONYM/ANTONYM，保留兼容）
# ══════════════════════════════════════════════════════════

def create_synonym_rel(
    word1: str, meaning1: str, pos1: str,
    word2: str, meaning2: str, pos2: str,
) -> list[dict]:
    """在两个单词的释义节点间创建 SYNONYM 关系（通过 word 限定匹配）。"""
    graph = get_graph()
    label1 = _to_label(pos1)
    label2 = _to_label(pos2)
    cypher = f"""
    MATCH (w1:Word {{name: $word1}})-[:TRANSLATION_INTO]->(m1:{label1})
    WHERE $meaning1 IN m1.means
    MATCH (w2:Word {{name: $word2}})-[:TRANSLATION_INTO]->(m2:{label2})
    WHERE $meaning2 IN m2.means
    MERGE (m1)-[r:SYNONYM]-(m2)
    ON CREATE SET r.query_times = 1
    ON MATCH SET r.query_times = r.query_times + 1
    RETURN m1.means AS means1, m2.means AS means2, r.query_times AS query_times
    """
    log_cypher(cypher, {"word1": word1, "meaning1": meaning1, "word2": word2, "meaning2": meaning2})
    return graph.query(cypher, params={
        "word1": word1, "meaning1": meaning1,
        "word2": word2, "meaning2": meaning2,
    })


def create_antonym_rel(
    word1: str, meaning1: str, pos1: str,
    word2: str, meaning2: str, pos2: str,
) -> list[dict]:
    """在两个单词的释义节点间创建 ANTONYM 关系（通过 word 限定匹配）。"""
    graph = get_graph()
    label1 = _to_label(pos1)
    label2 = _to_label(pos2)
    cypher = f"""
    MATCH (w1:Word {{name: $word1}})-[:TRANSLATION_INTO]->(m1:{label1})
    WHERE $meaning1 IN m1.means
    MATCH (w2:Word {{name: $word2}})-[:TRANSLATION_INTO]->(m2:{label2})
    WHERE $meaning2 IN m2.means
    MERGE (m1)-[r:ANTONYM]-(m2)
    ON CREATE SET r.query_times = 1
    ON MATCH SET r.query_times = r.query_times + 1
    RETURN m1.means AS means1, m2.means AS means2, r.query_times AS query_times
    """
    log_cypher(cypher, {"word1": word1, "meaning1": meaning1, "word2": word2, "meaning2": meaning2})
    return graph.query(cypher, params={
        "word1": word1, "meaning1": meaning1,
        "word2": word2, "meaning2": meaning2,
    })


def match_synonyms_of_word(word_name: str) -> list[dict]:
    """查询某个单词的近义词。"""
    graph = get_graph()
    cypher = """
    MATCH (w:Word {name: $word_name})-[:TRANSLATION_INTO]->(m)
    MATCH (m)-[r:SYNONYM]-(m2)
    OPTIONAL MATCH (w2:Word)-[:TRANSLATION_INTO]->(m2)
    RETURN DISTINCT m.means AS source_meaning,
                    labels(m) AS source_pos,
                    m2.means AS synonym_meaning,
                    labels(m2) AS synonym_pos,
                    w2.name AS synonym_word,
                    r.query_times AS query_times
    """
    log_cypher(cypher, {"word_name": word_name})
    return graph.query(cypher, params={"word_name": word_name})


def match_antonyms_of_word(word_name: str) -> list[dict]:
    """查询某个单词的反义词。"""
    graph = get_graph()
    cypher = """
    MATCH (w:Word {name: $word_name})-[:TRANSLATION_INTO]->(m)
    MATCH (m)-[r:ANTONYM]-(m2)
    OPTIONAL MATCH (w2:Word)-[:TRANSLATION_INTO]->(m2)
    RETURN DISTINCT m.means AS source_meaning,
                    labels(m) AS source_pos,
                    m2.means AS antonym_meaning,
                    labels(m2) AS antonym_pos,
                    w2.name AS antonym_word,
                    r.query_times AS query_times
    """
    log_cypher(cypher, {"word_name": word_name})
    return graph.query(cypher, params={"word_name": word_name})


def _to_label(pos: str) -> str:
    mapping = {"noun": "Noun", "verb": "Verb", "adjective": "Adjective", "adverb": "Adverb"}
    return mapping.get(pos, pos)
