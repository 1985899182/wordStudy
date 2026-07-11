"""
释义节点相关 Cypher 工具函数 —— 每个函数对应一种 Cypher 操作。
"""
from models.neo4j_client import get_graph
from services.log_service import log_cypher

POS_LABEL_MAP = {
    "noun": "Noun", "verb": "Verb",
    "adjective": "Adjective", "adverb": "Adverb",
}


def create_meaning_node(word_name: str, pos_label: str, meaning: str,
                        plot: str = "", plot_embedding: list[float] | None = None) -> list[dict]:
    """创建释义节点 + TRANSLATION_INTO 关系，mean_query_times 初始 1。

    节点同时带有 :Meaning 标签 + plot / plotEmbedding 用于向量搜索。
    """
    graph = get_graph()
    label = POS_LABEL_MAP.get(pos_label, pos_label)
    if plot_embedding is None:
        plot_embedding = []
    cypher = f"""
    MERGE (w:Word {{name: $word_name}})
    CREATE (m:{label}:Meaning {{means: [$meaning], plot: $plot, plotEmbedding: $plot_embedding}})
    CREATE (w)-[t:TRANSLATION_INTO]->(m)
    SET t.mean_query_times = 1
    RETURN m.means AS means, t.mean_query_times AS mean_query_times, labels(m) AS pos
    """
    params = {
        "word_name": word_name, "meaning": meaning,
        "plot": plot, "plot_embedding": plot_embedding,
    }
    log_cypher(cypher, params)
    return graph.query(cypher, params=params)


def add_meaning_to_means(word_name: str, pos_label: str, existing_meaning: str, new_meaning: str) -> list[dict]:
    """将新释义追加到指定释义节点的 means 列表。"""
    graph = get_graph()
    label = POS_LABEL_MAP.get(pos_label, pos_label)
    cypher = f"""
    MATCH (w:Word {{name: $word_name}})-[:TRANSLATION_INTO]->(m:{label})
    WHERE $existing_meaning IN m.means
    SET m.means = m.means + $new_meaning
    RETURN m.means AS means, labels(m) AS pos
    """
    log_cypher(cypher, {"word_name": word_name, "existing_meaning": existing_meaning, "new_meaning": new_meaning})
    return graph.query(cypher, params={"word_name": word_name, "existing_meaning": existing_meaning, "new_meaning": new_meaning})


def update_mean_query_times(word_name: str, pos_label: str, meaning: str, delta: int) -> list[dict]:
    """TRANSLATION_INTO.mean_query_times += delta（可为正/负）。"""
    graph = get_graph()
    label = POS_LABEL_MAP.get(pos_label, pos_label)
    cypher = f"""
    MATCH (w:Word {{name: $word_name}})-[t:TRANSLATION_INTO]->(m:{label})
    WHERE $meaning IN m.means
    SET t.mean_query_times = t.mean_query_times + $delta
    RETURN m.means AS means, t.mean_query_times AS mean_query_times
    """
    log_cypher(cypher, {"word_name": word_name, "meaning": meaning, "delta": delta})
    return graph.query(cypher, params={"word_name": word_name, "meaning": meaning, "delta": delta})


def find_all_meanings(word_name: str) -> list[dict]:
    """查询单词的所有释义节点。"""
    graph = get_graph()
    cypher = """
    MATCH (w:Word {name: $word_name})-[t:TRANSLATION_INTO]->(m)
    RETURN labels(m) AS pos, m.means AS means, t.mean_query_times AS mean_query_times
    """
    log_cypher(cypher, {"word_name": word_name})
    return graph.query(cypher, params={"word_name": word_name})


def find_meaning_by_text(pos_label: str, meaning_text: str) -> list[dict]:
    """按释义文本查找释义节点（不限单词）。"""
    graph = get_graph()
    label = POS_LABEL_MAP.get(pos_label, pos_label)
    cypher = f"""
    MATCH (m:{label})
    WHERE $meaning_text IN m.means
    RETURN m.means AS means, labels(m) AS pos
    """
    log_cypher(cypher, {"meaning_text": meaning_text})
    return graph.query(cypher, params={"meaning_text": meaning_text})


def get_word_meanings(word_name: str) -> dict[str, list[tuple[str, int]]]:
    """获取单词所有释义: {pos: [(meaning, mean_query_times), ...]}。"""
    graph = get_graph()
    cypher = """
    MATCH (w:Word {name: $word_name})-[t:TRANSLATION_INTO]->(m)
    RETURN labels(m) AS pos, m.means AS means, t.mean_query_times AS mean_query_times
    """
    log_cypher(cypher, {"word_name": word_name})
    results = graph.query(cypher, params={"word_name": word_name})
    out: dict[str, list[tuple[str, int]]] = {}
    VALID_POS = {"noun", "verb", "adjective", "adverb"}
    for row in results:
        pos_list = row.get("pos", [])
        for p_label in pos_list:
            key = p_label.lower()
            if key not in VALID_POS:
                continue
            out.setdefault(key, [])
            for m in row.get("means", []):
                out[key].append((m, row.get("mean_query_times", 1)))
    return out


def add_means_to_list(word_name: str, pos_label: str, target_meaning: str, new_meaning: str) -> list[dict]:
    """向单词指定词性下、指定释义节点的 means 列表追加新释义。"""
    graph = get_graph()
    label = POS_LABEL_MAP.get(pos_label, pos_label)
    cypher = f"""
    MATCH (w:Word {{name: $word_name}})-[:TRANSLATION_INTO]->(m:{label})
    WHERE $target_meaning IN m.means
    SET m.means = m.means + [$new_meaning]
    RETURN m.means AS means, labels(m) AS pos
    """
    log_cypher(cypher, {"word_name": word_name, "target_meaning": target_meaning, "new_meaning": new_meaning})
    return graph.query(cypher, params={"word_name": word_name, "target_meaning": target_meaning, "new_meaning": new_meaning})


def find_meaning_node(word_name: str, pos_label: str, meaning: str) -> dict | None:
    """查找单词的特定释义节点，返回 elementId。"""
    graph = get_graph()
    label = POS_LABEL_MAP.get(pos_label, pos_label)
    cypher = f"""
    MATCH (w:Word {{name: $word_name}})-[:TRANSLATION_INTO]->(m:{label})
    WHERE $meaning IN m.means
    RETURN elementId(m) AS node_id, m.means AS means
    """
    log_cypher(cypher, {"word_name": word_name, "meaning": meaning})
    results = graph.query(cypher, params={"word_name": word_name, "meaning": meaning})
    return results[0] if results else None


# ══════════════════════════════════════════════════════════
# 跨单词释义操作（释义节点去重 + 关系建立）
# ══════════════════════════════════════════════════════════

def find_meaning_cross_word(pos_label: str, meaning_text: str) -> list[dict]:
    """跨所有单词，精确匹配释义文本，返回匹配的释义节点信息。

    Returns: [{"node_id": str, "means": list[str], "word_name": str, "pos": list[str]}, ...]
    """
    graph = get_graph()
    label = POS_LABEL_MAP.get(pos_label, pos_label)
    cypher = f"""
    MATCH (w:Word)-[:TRANSLATION_INTO]->(m:{label})
    WHERE $meaning_text IN m.means
    RETURN elementId(m) AS node_id, m.means AS means, w.name AS word_name, labels(m) AS pos
    """
    params = {"meaning_text": meaning_text}
    log_cypher(cypher, params)
    return graph.query(cypher, params=params)


def attach_word_to_meaning(word_name: str, node_id: str, new_meaning: str) -> list[dict]:
    """将当前单词连接到已有的释义节点，同时追加新释义到 means 列表。

    执行：
    1. CREATE (word)-[:TRANSLATION_INTO]->(meaning_node)
    2. SET meaning_node.means += [new_meaning]
    """
    graph = get_graph()
    cypher = """
    MATCH (w:Word {name: $word_name})
    MATCH (m) WHERE elementId(m) = $node_id
    MERGE (w)-[t:TRANSLATION_INTO]->(m)
    ON CREATE SET t.mean_query_times = 1
    ON MATCH SET t.mean_query_times = t.mean_query_times + 1
    FOREACH (_ IN CASE WHEN $new_meaning IN m.means THEN [] ELSE [1] END |
        SET m.means = m.means + [$new_meaning]
    )
    RETURN m.means AS means, labels(m) AS pos, t.mean_query_times AS mean_query_times
    """
    params = {"word_name": word_name, "node_id": node_id, "new_meaning": new_meaning}
    log_cypher(cypher, params)
    return graph.query(cypher, params=params)


def get_words_of_meaning_node(node_id: str) -> list[str]:
    """获取连接到指定释义节点的所有单词名。"""
    graph = get_graph()
    cypher = """
    MATCH (w:Word)-[:TRANSLATION_INTO]->(m)
    WHERE elementId(m) = $node_id
    RETURN w.name AS word_name
    """
    params = {"node_id": node_id}
    log_cypher(cypher, params)
    results = graph.query(cypher, params=params)
    return [r.get("word_name", "") for r in results]


def get_all_meanings() -> list[dict]:
    """获取所有 Meaning 节点及其关联的单词列表。

    Returns: [{element_id, pos: str, means_text: str, words: [str]}, ...]
    """
    graph = get_graph()
    cypher = """
    MATCH (m)
    WHERE m:Noun OR m:Verb OR m:Adjective OR m:Adverb
    OPTIONAL MATCH (w:Word)-[:TRANSLATION_INTO]->(m)
    RETURN elementId(m) AS element_id, labels(m) AS pos_labels,
           m.means AS means, collect(DISTINCT w.name) AS words
    """
    log_cypher(cypher, {})
    rows = graph.query(cypher)

    pos_map = {"Noun": "n.", "Verb": "v.", "Adverb": "adv.", "Adjective": "adj."}
    result = []
    for row in rows:
        pos_labels = row.get("pos_labels") or []
        pos_abbr = ""
        for label in pos_labels:
            abbr = pos_map.get(label)
            if abbr:
                pos_abbr = abbr
                break
        means_list = row.get("means") or []
        words_list = row.get("words") or []
        # 过滤掉 null
        words_list = [w for w in words_list if w]
        means_text = "，".join(means_list)
        result.append({
            "element_id": row.get("element_id", ""),
            "pos": pos_abbr,
            "means_text": means_text,
            "words": sorted(words_list),
        })
    return result


def delete_meaning_relation(word_name: str, pos_label: str, meaning: str) -> dict:
    """删除单词的某条释义关系（参数化 Cypher，非 LLM 生成）。

    执行：
    1. 从释义节点的 means 列表中移除该释义文本
    2. 删除单词到这个释义节点的 TRANSLATION_INTO 关系
    3. 如果释义节点 means 变空且没有其他单词引用，则删除该节点

    Returns: {"deleted_node": bool, "means_updated": bool}
    """
    graph = get_graph()
    label = POS_LABEL_MAP.get(pos_label, pos_label)
    cypher = f"""
    MATCH (w:Word {{name: $word_name}})-[t:TRANSLATION_INTO]->(m:{label})
    WHERE $meaning IN m.means
    SET m.means = [x IN m.means WHERE x <> $meaning]
    DELETE t
    WITH m
    WHERE size(m.means) = 0
    OPTIONAL MATCH (m)<-[other:TRANSLATION_INTO]-()
    WITH m, count(other) AS ref_count
    WHERE ref_count = 0
    DETACH DELETE m
    RETURN true AS deleted_node
    """
    params = {"word_name": word_name, "meaning": meaning}
    log_cypher(cypher, params)
    results = graph.query(cypher, params=params)
    return {"deleted_node": len(results) > 0, "means_updated": True}
