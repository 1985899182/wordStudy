"""
SYNONYM / ANTONYM 关系相关 Cypher 工具函数。
所有关系函数均限定 Word 节点，防止不同单词的同名释义被错误关联。
"""
from word_study.services.neo4j_client import get_graph
from word_study.services.log_service import log_cypher


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
    """删除两个 Word 节点间的 SYNONYM 或 ANTONYM 关系。

    rel_type 会拼入 Cypher，必须白名单校验（防御性，路由层已校验一次）。
    """
    if rel_type not in ("SYNONYM", "ANTONYM"):
        raise ValueError(f"不允许的关系类型: {rel_type}")
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


def get_synonym_clusters() -> list[dict]:
    """获取所有 Word 层级的近义词簇（通过 SYNONYM 关系连接）。

    使用 BFS/DFS 将连通分量分组为簇。
    Returns: [{words: [{word, meanings: {pos_abbr: [meaning, ...]}}], pairs: [[a,b],...]}, ...]
    """
    return _get_word_rel_clusters("SYNONYM")


def get_antonym_clusters() -> list[dict]:
    """获取所有 Word 层级的反义词簇（通过 ANTONYM 关系连接）。"""
    return _get_word_rel_clusters("ANTONYM")


def _get_word_rel_clusters(rel_type: str) -> list[dict]:
    """通用：获取 Word 层级关系簇。"""
    graph = get_graph()
    cypher = f"""
    MATCH (w1:Word)-[r:{rel_type}]-(w2:Word)
    OPTIONAL MATCH (w1)-[:TRANSLATION_INTO]->(m1)
    WHERE m1:Noun OR m1:Verb OR m1:Adjective OR m1:Adverb
    OPTIONAL MATCH (w2)-[:TRANSLATION_INTO]->(m2)
    WHERE m2:Noun OR m2:Verb OR m2:Adjective OR m2:Adverb
    RETURN w1.name AS word1, w2.name AS word2,
           labels(m1) AS pos1, m1.means AS means1,
           labels(m2) AS pos2, m2.means AS means2
    """
    log_cypher(cypher, {})
    rows = graph.query(cypher)

    # 构建邻接表和无向图中所有节点
    adj: dict[str, set[str]] = {}
    word_meanings: dict[str, dict[str, list[str]]] = {}
    pos_map = {"Noun": "n.", "Verb": "v.", "Adjective": "adj.", "Adverb": "adv."}

    for row in rows:
        w1 = row.get("word1", "")
        w2 = row.get("word2", "")
        if not w1 or not w2 or w1 == w2:
            continue

        adj.setdefault(w1, set()).add(w2)
        adj.setdefault(w2, set()).add(w1)

        # 收集每个单词的释义
        for w, pos_labels_raw, means_raw in [(w1, row.get("pos1"), row.get("means1")),
                                               (w2, row.get("pos2"), row.get("means2"))]:
            if w not in word_meanings:
                word_meanings[w] = {}
            pos_labels = pos_labels_raw or []
            means = means_raw or []
            for label in pos_labels:
                abbr = pos_map.get(label)
                if not abbr:
                    continue
                word_meanings[w].setdefault(abbr, [])
                for m in means:
                    if m not in word_meanings[w][abbr]:
                        word_meanings[w][abbr].append(m)

    # DFS 找连通分量
    visited: set[str] = set()
    clusters: list[dict] = []

    for node in adj:
        if node in visited:
            continue
        # BFS 收集簇
        component: list[str] = []
        stack = [node]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            component.append(cur)
            for nb in adj.get(cur, []):
                if nb not in visited:
                    stack.append(nb)

        # 收集簇内单词及其释义
        words_data = []
        for w in sorted(component):
            words_data.append({
                "word": w,
                "meanings": word_meanings.get(w, {}),
            })

        # 收集簇内所有边对
        pairs = []
        seen_pairs = set()
        for w in component:
            for nb in adj.get(w, []):
                pair_key = tuple(sorted([w, nb]))
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    pairs.append([w, nb])

        if words_data:
            clusters.append({"words": words_data, "pairs": pairs})

    return clusters
