"""
FastAPI 路由 —— 单词提取 / 释义保存 / 近义搜索 / AI 查询。
"""
from __future__ import annotations

import traceback

from fastapi import APIRouter, Query

from config import SOURCE_MANUAL
from models.schemas import (
    AIQueryRequest, AIQueryResponse,
    ExtractRequest, ExtractResponse,
    SaveRequest, SaveResponse, SaveResultItem,
    PosValidateRequest, PosValidateResponse,
    FetchSuggestionsResponse, ManualMeaning, ExistingMeaning, WordInfo,
    QueryTimesRequest, WordRelRequest, DeleteMeaningRequest,
    StatsResponse, RecentActivity,
)
from services.llm_service import (
    extract_words, judge_meaning_merge, validate_pos, get_cypher_qa,
    select_best_definition,
)
from services.datamuse_service import fetch_synonyms, fetch_antonyms, fetch_definitions
from services.embedding_service import ensure_vector_index, embed_text, search_similar_meanings
from services.log_service import log_cypher
from models.sqlite_client import insert_word_log, get_recent_logs
from tools.word_tools import (
    search_word, create_word_node, update_word_query_times, append_word_resource,
    get_all_words_grouped,
)
from tools.meaning_tools import (
    get_word_meanings, create_meaning_node, add_means_to_list,
    update_mean_query_times, find_meaning_node,
    find_meaning_cross_word, attach_word_to_meaning,
    delete_meaning_relation, get_all_meanings,
)
from tools.relation_tools import (
    create_word_synonym_rel, create_word_antonym_rel, delete_word_rel,
    get_synonym_clusters, get_antonym_clusters,
)
from models.neo4j_client import get_graph

router = APIRouter(prefix="/api")

# ── POS 标签映射 ─────────────────────────────────────────
_POS_LABEL = {"noun": "Noun", "verb": "Verb", "adjective": "Adjective", "adverb": "Adverb"}

# ── 向量相似度阈值 ────────────────────────────────────────
_VECTOR_SIMILARITY_THRESHOLD = 0.85


@router.get("/stats", response_model=StatsResponse)
def get_stats():
    """获取全局统计数据（单词/释义/关系数量 + 最近操作日志）。"""
    graph = get_graph()
    cypher = """
    CALL { MATCH (w:Word) RETURN count(w) AS total_words }
    CALL { MATCH (m) WHERE m:Noun OR m:Verb OR m:Adjective OR m:Adverb RETURN count(m) AS total_meanings }
    CALL { MATCH ()-[r:SYNONYM]-() RETURN count(r) AS total_synonyms }
    CALL { MATCH ()-[r:ANTONYM]-() RETURN count(r) AS total_antonyms }
    RETURN total_words, total_meanings, total_synonyms, total_antonyms
    """
    try:
        rows = graph.query(cypher)
        row = rows[0] if rows else {}
    except Exception:
        row = {}

    recent = get_recent_logs(8)

    return StatsResponse(
        total_words=row.get("total_words", 0),
        total_meanings=row.get("total_meanings", 0),
        total_synonyms=row.get("total_synonyms", 0),
        total_antonyms=row.get("total_antonyms", 0),
        recent_activity=[
            RecentActivity(word=r["word"], meaning=r["meaning"],
                           source=r["source"], time=r["time"])
            for r in recent
        ],
    )


async def _ensure_meaning_node(word: str, pos: str, pos_label: str, meaning: str) -> str:
    """确保某单词的某 POS 下存在该释义节点（必要时归并），返回可定位释义的文本。

    流程（按优先级）：
    1. 同单词精确匹配 → update mean_query_times +1
    2. 跨单词精确匹配 → attach 当前单词到已有节点 + SYNONYM 关系
    3. Datamuse 英文释义 → LLM 选最佳 → 向量嵌入 → 向量相似度搜索 → LLM确认 → 归并
    4. 以上都未命中 → 创建新释义节点（含 plot + plotEmbedding）
    """
    print(f"[DEBUG] _ensure_meaning_node: word={word}, pos={pos}, pos_label={pos_label}, meaning={meaning}")

    # ── Step 1: 同单词精确匹配 ──
    existing_node = find_meaning_node(word, pos_label, meaning)
    if existing_node:
        update_mean_query_times(word, pos_label, meaning, 1)
        print(f"[DEBUG]   Step1 同单词精确匹配 → mean_query_times +1")
        return meaning

    # ── Step 2: 跨单词精确匹配（参数化 Cypher，非 LLM 生成）──
    cross_matches = find_meaning_cross_word(pos_label, meaning)
    for match in cross_matches:
        match_word = (match.get("word_name") or "").lower()
        if match_word != word.lower():
            node_id = match.get("node_id", "")
            if node_id:
                attach_word_to_meaning(word, node_id, meaning)
                create_word_synonym_rel(word, match_word)
                print(f"[INFO] Step2 跨单词精确匹配: '{word}' ↔ '{match_word}' 共享释义 '{meaning}'")
                return meaning

    # ── Step 3: Datamuse + LLM + 向量相似度搜索 ──
    english_defs = await fetch_definitions(word)
    plot_text = ""
    embedding = None

    if english_defs:
        plot_text = await select_best_definition(word, meaning, english_defs)

    if plot_text:
        # 确保向量索引存在（首次创建释义节点时）
        ensure_vector_index()

        try:
            embedding = await embed_text(plot_text)
            similar_results = search_similar_meanings(plot_text, k=3)

            for sim in similar_results:
                sim_score = sim.get("score", 0)
                if sim_score < _VECTOR_SIMILARITY_THRESHOLD:
                    continue

                sim_word = (sim.get("word") or "").lower()
                sim_node_id = sim.get("node_id", "")
                sim_means = sim.get("means", [])

                if not sim_word or not sim_node_id:
                    continue

                # 同一单词的同词性相似释义 → 归并到已有节点
                if sim_word == word.lower():
                    target_meaning = sim_means[0] if sim_means else ""
                    if target_meaning:
                        add_means_to_list(word, pos_label, target_meaning, meaning)
                        update_mean_query_times(word, pos_label, target_meaning, 1)
                        print(f"[INFO] Step3 向量搜索 - 同单词归并: '{meaning}' → '{target_meaning}'")
                        return target_meaning
                    continue

                # 不同单词 → LLM 二次确认相似度
                should_merge, matched_idx, reason = await judge_meaning_merge(sim_means, meaning)
                if should_merge:
                    attach_word_to_meaning(word, sim_node_id, meaning)
                    create_word_synonym_rel(word, sim_word)
                    print(f"[INFO] Step3 向量搜索+LLM归并: '{word}' ↔ '{sim_word}' (score={sim_score:.3f}): {reason}")
                    return meaning
                else:
                    print(f"[INFO] Step3 向量搜索命中但LLM判定不归并: {reason}")
        except Exception as exc:
            print(f"[WARN] 向量搜索异常，回退到创建新节点: {exc}")

    # ── Step 4: 创建新释义节点 ──
    if plot_text and embedding:
        create_meaning_node(word, pos_label, meaning, plot=plot_text, plot_embedding=embedding)
    else:
        create_meaning_node(word, pos_label, meaning)

    print(f"[INFO] Step4 创建新释义节点: [{pos_label}] {meaning}" + (f" plot={plot_text[:50]}..." if plot_text else ""))
    return meaning


# ══════════════════════════════════════════════════════════
# Step 1: 单词提取
# ══════════════════════════════════════════════════════════

@router.post("/step1-extract", response_model=ExtractResponse)
async def step1_extract(req: ExtractRequest):
    extracted = await extract_words(req.text)
    words = [w.lower() for w in extracted]

    # 查询每个词是否已存在
    infos: list[WordInfo] = []
    for w in words:
        existing = search_word(w)
        if existing:
            means = get_word_meanings(w)
            em: dict[str, list[ExistingMeaning]] = {}
            for pos, meaning_list in means.items():
                em[pos] = [ExistingMeaning(meaning=m, query_times=t) for m, t in meaning_list]
            infos.append(WordInfo(
                word=w, exists=True,
                existing_meanings=em,
            ))
        else:
            infos.append(WordInfo(word=w, exists=False, existing_meanings={}))

    return ExtractResponse(words=infos, spell_warnings=[])


# ══════════════════════════════════════════════════════════
# POS 实时校验
# ══════════════════════════════════════════════════════════

@router.post("/validate-pos", response_model=PosValidateResponse)
async def validate_pos_endpoint(req: PosValidateRequest):
    # 获取该词该 POS 下的已有释义列表
    existing_means = get_word_meanings(req.word).get(req.pos, [])
    existing_list = [em[0] for em in existing_means]
    result = await validate_pos(req.word, req.pos, req.meaning, existing_list)
    return PosValidateResponse(
        valid=result["valid"],
        suggested_pos=result["suggested_pos"],
        merge_hint=result["merge_hint"],
        reason=result["reason"],
    )


# ══════════════════════════════════════════════════════════
# 单词/释义 选择/取消（query_times +1/-1）
# ══════════════════════════════════════════════════════════

@router.post("/update-query-times")
async def update_query_times(req: QueryTimesRequest):
    delta = -1 if req.action == "unselect" else 1

    if req.pos and req.meaning:
        # 释义操作
        pos_label = _POS_LABEL.get(req.pos, req.pos)
        update_mean_query_times(req.word, pos_label, req.meaning, delta)
        msg = f"释义 '{req.meaning}' mean_query_times {'-1' if delta < 0 else '+1'}"
    else:
        # 单词操作
        update_word_query_times(req.word, delta)
        msg = f"单词 '{req.word}' word_query_times {'-1' if delta < 0 else '+1'}"

    print(f"[INFO] {msg}")
    return {"ok": True, "message": msg}


# ══════════════════════════════════════════════════════════
# 近义/反义词搜索（Datamuse → 筛选已存在图库的）
# ══════════════════════════════════════════════════════════

@router.get("/fetch-suggestions", response_model=FetchSuggestionsResponse)
async def fetch_suggestions(word: str, type: str = "synonym"):
    """查 Datamuse，过滤出已在 Neo4j 中的词。"""
    fetch_fn = fetch_synonyms if type == "synonym" else fetch_antonyms
    try:
        all_words = await fetch_fn(word)
    except Exception as exc:
        print(f"[DEBUG] Datamuse {type} for '{word}' failed: {exc}")
        return FetchSuggestionsResponse(words=[])

    print(f"[DEBUG] Datamuse {type} for '{word}': {all_words}")

    # 过滤已存在图库中的
    existing_words: list[str] = []
    for w in all_words:
        if search_word(w):
            existing_words.append(w)

    return FetchSuggestionsResponse(words=existing_words)


# ══════════════════════════════════════════════════════════
# Word 直连 SYNONYM/ANTONYM
# ══════════════════════════════════════════════════════════

@router.post("/create-word-rel")
async def create_word_rel(req: WordRelRequest):
    if req.rel_type == "synonym":
        create_word_synonym_rel(req.word1, req.word2)
    else:
        create_word_antonym_rel(req.word1, req.word2)
    print(f"[INFO] Word 关系已建立: {req.word1} ↔ {req.word2} ({req.rel_type})")
    return {"ok": True}


@router.post("/delete-word-rel")
async def delete_word_rel_endpoint(req: WordRelRequest):
    delete_word_rel(req.word1, req.word2, req.rel_type)
    print(f"[INFO] Word 关系已删除: {req.word1} ↔ {req.word2} ({req.rel_type})")
    return {"ok": True}


# ══════════════════════════════════════════════════════════
# 删除释义
# ══════════════════════════════════════════════════════════

@router.post("/delete-meaning")
async def delete_meaning(req: DeleteMeaningRequest):
    """删除单词的某条释义（参数化 Cypher，非 LLM 生成）。

    移除 TRANSLATION_INTO 关系和 means 列表中的释义文本，
    如果释义节点变为空引用则自动清理。
    """
    pos_label = _POS_LABEL.get(req.pos)
    if not pos_label:
        return {"ok": False, "message": f"无效的词性: {req.pos}"}

    try:
        result = delete_meaning_relation(req.word, pos_label, req.meaning)
        if result.get("means_updated"):
            print(f"[INFO] 释义已删除: '{req.word}' [{req.pos}] '{req.meaning}'"
                  + (" (节点已清理)" if result.get("deleted_node") else ""))
            return {"ok": True, "message": "释义已删除"}
        else:
            return {"ok": False, "message": f"未找到释义 '{req.meaning}'"}
    except Exception as exc:
        print(f"[ERROR] 删除释义失败: {exc}")
        return {"ok": False, "message": str(exc)}


# ══════════════════════════════════════════════════════════
# Step 2: 保存单词 + 释义 + 关系
# ══════════════════════════════════════════════════════════

@router.post("/step2-save", response_model=SaveResponse)
async def step2_save(req: SaveRequest):
    results: list[SaveResultItem] = []

    for item in req.selections:
        word = item.word.lower()
        prefix = word[0]
        resource = item.resource or SOURCE_MANUAL

        try:
            # ── 创建 / 更新 Word 节点 ──
            existing = search_word(word)
            if existing:
                append_word_resource(word, resource)
                print(f"[INFO] 单词 '{word}' 已存在，追加 resource")
            else:
                create_word_node(word, prefix, resource)
                print(f"[INFO] 单词 '{word}' 节点已创建")

            # ── 处理释义（统一走 _ensure_meaning_node）──
            saved_meanings: list[dict] = []
            for mm in item.manual_meanings:
                pos_label = _POS_LABEL.get(mm.pos)
                if not pos_label:
                    continue

                await _ensure_meaning_node(word, mm.pos, pos_label, mm.meaning)

                saved_meanings.append({"pos": mm.pos, "meaning": mm.meaning})

                # ── SQLite 日志 ──
                pos_abbr = {"noun": "n.", "verb": "v.", "adjective": "adj.", "adverb": "adv."}
                pos_short = pos_abbr.get(mm.pos, mm.pos)
                insert_word_log(word, f"{pos_short}{mm.meaning}", resource)

            # ── 建立近义/反义关系（单词已确保存在后再执行）──
            for rel in item.relations:
                target = rel.target_word.lower()
                if target == word:
                    continue  # 跳过自己
                # 确保目标单词也存在
                if not search_word(target):
                    print(f"[WARN] 关系目标词 '{target}' 不在图库中，跳过")
                    continue
                if rel.rel_type == "synonym":
                    create_word_synonym_rel(word, target)
                else:
                    create_word_antonym_rel(word, target)
                print(f"[INFO] 单词关系已建立: {word} ↔ {target} ({rel.rel_type})")

            results.append(SaveResultItem(
                word=word, status="ok",
                message=f"已保存 {len(item.manual_meanings)} 条释义",
            ))

        except Exception as exc:
            print(f"[ERROR] 单词 '{word}' 保存异常:")
            traceback.print_exc()
            results.append(SaveResultItem(
                word=word, status="error",
                message=f"保存失败: {exc}",
            ))

    return SaveResponse(results=results)


# ══════════════════════════════════════════════════════════
# AI 自然语言查询
# ══════════════════════════════════════════════════════════

@router.post("/ai-query", response_model=AIQueryResponse)
async def ai_query(req: AIQueryRequest):
    try:
        qa = get_cypher_qa()
        raw_result = qa.invoke({"query": req.question})
        result_text = raw_result.get("result", "")

        log_cypher(req.question, result_text)

        return AIQueryResponse(
            question=req.question,
            result=str(result_text),
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return AIQueryResponse(
            question=req.question,
            result=f"[错误] 查询失败: {exc}",
        )


# ══════════════════════════════════════════════════════════
# 侧边栏搜索（参数化 Cypher）
# ══════════════════════════════════════════════════════════

@router.get("/search-words")
def search_words(q: str = Query(..., min_length=1, description="搜索前缀")):
    """按前缀搜索单词（参数化 Cypher，STARTS WITH），最多返回 10 条。"""
    graph = get_graph()
    # 参数化 Cypher — 非 LLM 生成，硬编码安全
    cypher = "MATCH (w:Word) WHERE w.name STARTS WITH $prefix RETURN w.name ORDER BY w.name LIMIT 10"
    try:
        rows = graph.query(cypher, params={"prefix": q.lower()})
        words = [row.get("w.name", "") for row in rows]
    except Exception:
        words = []
    return {"words": words}


@router.get("/word-meanings")
def word_meanings(word: str = Query(..., min_length=1, description="单词")):
    """获取单词的各词性释义列表。"""
    w = word.lower()
    # 更新 word_query_times
    try:
        update_word_query_times(w, 1)
    except Exception:
        pass

    # 获取释义
    try:
        means = get_word_meanings(w)
    except Exception:
        means = {}

    # 转换为前端友好格式: [{pos: "noun", meanings: ["硬的","坚固的"]}, ...]
    pos_order = {"noun": "n.", "verb": "v.", "adjective": "adj.", "adverb": "adv."}
    result = []
    for pos, meaning_list in means.items():
        abbr = pos_order.get(pos, pos)
        result.append({
            "pos": pos,
            "abbr": abbr,
            "meanings": [m for m, _ in meaning_list],
        })

    return {
        "word": w,
        "exists": bool(means),
        "meanings": result,
    }


# ══════════════════════════════════════════════════════════
#  每日背诵单词 API
# ══════════════════════════════════════════════════════════

from datetime import date as dt_date
from models.sqlite_client import get_daily_words_today, save_daily_words
import random


# ══════════════════════════════════════════════════════════
#  Browse 浏览页面 API（总单词 / 总释义 / 近义词 / 反义词）
# ══════════════════════════════════════════════════════════

@router.get("/browse/words")
def api_browse_words():
    """获取所有单词，按首字母分组，含各词性释义。"""
    try:
        grouped = get_all_words_grouped()
        return {"ok": True, "data": grouped}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@router.get("/browse/meanings")
def api_browse_meanings():
    """获取所有释义节点及其关联的单词。"""
    try:
        meanings = get_all_meanings()
        return {"ok": True, "data": meanings}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@router.get("/browse/synonyms")
def api_browse_synonyms():
    """获取近义词簇。"""
    try:
        clusters = get_synonym_clusters()
        return {"ok": True, "data": clusters}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@router.get("/browse/antonyms")
def api_browse_antonyms():
    """获取反义词簇。"""
    try:
        clusters = get_antonym_clusters()
        return {"ok": True, "data": clusters}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ══════════════════════════════════════════════════════════
#  每日背诵单词 API（续）
# ══════════════════════════════════════════════════════════


@router.get("/daily-words")
def api_get_daily_words():
    """获取今日的每日单词列表。"""
    today = dt_date.today().isoformat()
    words = get_daily_words_today(today)
    return {"date": today, "words": words or []}


@router.post("/generate-daily-words")
def api_generate_daily_words(req: dict):
    """全新生成今日每日单词。body: {count: int}"""
    today = dt_date.today().isoformat()
    existing = get_daily_words_today(today)
    if existing:
        return {"ok": False, "message": "今天已经生成过了，请使用追加功能添加更多单词"}

    count = min(max(int(req.get("count", 10)), 1), 100)
    graph = get_graph()
    all_rows = graph.query("MATCH (w:Word) RETURN w.name")
    all_words = [r["w.name"] for r in all_rows]
    if count > len(all_words):
        return {"ok": False, "message": f"数据库中只有 {len(all_words)} 个单词，无法生成 {count} 个"}

    selected = random.sample(all_words, count)
    result_words = []
    for w in selected:
        try:
            means = get_word_meanings(w)
        except Exception:
            means = {}
        abbreviations = []
        for pos, mlist in means.items():
            abbr = _pos_abbr(pos)
            for m_val, _ in mlist:
                abbreviations.append(f"{abbr} {m_val}")
        result_words.append({
            "word": w,
            "meanings": abbreviations,
            "meaning_text": "; ".join(abbreviations) if abbreviations else "暂无释义",
        })

    save_daily_words(today, result_words)
    return {"ok": True, "date": today, "words": result_words}


@router.post("/add-daily-words")
def api_add_daily_words(req: dict):
    """追加每日单词。body: {count: int}"""
    today = dt_date.today().isoformat()
    existing = get_daily_words_today(today) or []
    existing_names = {w["word"] for w in existing}

    count = min(max(int(req.get("count", 5)), 1), 50)
    graph = get_graph()
    all_rows = graph.query("MATCH (w:Word) RETURN w.name")
    all_words = [r["w.name"] for r in all_rows if r["w.name"] not in existing_names]

    if not all_words:
        return {"ok": False, "message": "没有更多新单词可供添加"}
    if count > len(all_words):
        count = len(all_words)

    selected = random.sample(all_words, count)
    new_words = []
    for w in selected:
        try:
            means = get_word_meanings(w)
        except Exception:
            means = {}
        abbreviations = []
        for pos, mlist in means.items():
            abbr = _pos_abbr(pos)
            for m_val, _ in mlist:
                abbreviations.append(f"{abbr} {m_val}")
        new_words.append({
            "word": w,
            "meanings": abbreviations,
            "meaning_text": "; ".join(abbreviations) if abbreviations else "暂无释义",
        })

    merged = existing + new_words
    save_daily_words(today, merged)
    return {"ok": True, "date": today, "words": merged, "added": len(new_words)}


def _pos_abbr(pos: str) -> str:
    m = {"noun": "n.", "verb": "v.", "adjective": "adj.", "adverb": "adv."}
    return m.get(pos, pos)
