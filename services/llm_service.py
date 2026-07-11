"""
LLM 服务 —— LangChain 图 QA、单词提取、释义归并、近义/反义词匹配。
使用 DeepSeek API（参照 genai-integration-langchain 代码格式）。
"""
import json
import re

from langchain.chat_models import init_chat_model
from langchain_neo4j import GraphCypherQAChain
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache

from config import DEEPSEEK_MODEL, BASE_DIR
from models.neo4j_client import get_graph
from services.cache_service import get_l1_cache, set_l1_cache

# ── 全局 LLM 缓存（相同 prompt 直接命中，节省 token）──
set_llm_cache(InMemoryCache())

# ── LLM 初始化 ──────────────────────────────────────────

_llm = init_chat_model(DEEPSEEK_MODEL, model_provider="deepseek")

_cypher_qa: GraphCypherQAChain | None = None


def get_cypher_qa() -> GraphCypherQAChain:
    """获取 GraphCypherQAChain 单例。使用默认 prompt，自动从图库提取 schema。"""
    global _cypher_qa
    if _cypher_qa is None:
        graph = get_graph()
        _cypher_qa = GraphCypherQAChain.from_llm(
            graph=graph,
            llm=_llm,
            allow_dangerous_requests=True,
            validate_cypher=True,
            verbose=True,  # 终端可看到实际生成的 Cypher
        )
    return _cypher_qa


# ── Prompt 文件加载 ─────────────────────────────────────

_PROMPTS_DIR = BASE_DIR / "prompts"


def _load_prompt(filename: str) -> str:
    path = _PROMPTS_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


_EXTRACT_PROMPT = _load_prompt("extract_words.txt")
_MERGE_PROMPT_TEMPLATE = _load_prompt("merge_meaning.txt")


# ── 单词提取 ────────────────────────────────────────────

async def extract_words(text: str) -> list[str]:
    """用 LLM 从输入文本中提取英文单词。带 L1 缓存。"""
    cache_key = f"extract:{text.strip().lower()}"
    cached = get_l1_cache(cache_key)
    if cached:
        try:
            return json.loads(cached)
        except json.JSONDecodeError:
            pass

    messages = [
        ("system", _EXTRACT_PROMPT),
        ("user", text),
    ]
    response = await _llm.ainvoke(messages)
    content = _extract_json_array(response.content)

    # 写入缓存
    try:
        set_l1_cache(cache_key, json.dumps(content))
    except Exception:
        pass

    return content


def _extract_json_array(raw: str | list) -> list[str]:
    """从 LLM 响应中提取 JSON 数组。"""
    if isinstance(raw, list):
        return [str(item).strip().lower() for item in raw if str(item).strip()]
    if isinstance(raw, str):
        match = re.search(r"\[(.*?)\]", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    return [str(item).strip().lower() for item in parsed if str(item).strip()]
            except json.JSONDecodeError:
                pass
        words = re.findall(r"[a-zA-Z]{2,}", raw)
        return list({w.lower() for w in words})
    return []


def _extract_json_object(raw: str | list) -> dict | None:
    """从 LLM 响应中提取 JSON 对象，处理 markdown 代码块包裹等常见噪音。

    支持格式:
        {"key": "value"}
        ```json\n{"key": "value"}\n```
        一些说明文字 \n{"key": "value"}\n 更多文字
    """
    text = raw if isinstance(raw, str) else str(raw)

    # 1. 去掉 markdown 代码块标记
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)

    # 2. 找到最外层 { } 之间的内容
    # 从第一个 { 到最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        return None

    json_candidate = text[start:end + 1]

    # 3. 尝试直接解析
    try:
        result = json.loads(json_candidate)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # 4. 尝试修复常见问题：LLM 可能把 true/false 写成 Python 风格的 True/False
    fixed = re.sub(r":\s*True\b", ": true", json_candidate)
    fixed = re.sub(r":\s*False\b", ": false", fixed)
    fixed = re.sub(r":\s*None\b", ": null", fixed)
    try:
        result = json.loads(fixed)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    return None


# ── 释义归并判断 ────────────────────────────────────────

async def judge_meaning_merge(existing_meanings: list[str], new_meaning: str) -> tuple[bool, int, str]:
    """让 LLM 判断新释义是否应归并到已有释义列表。"""
    prompt_text = _MERGE_PROMPT_TEMPLATE.format(
        existing_meanings=json.dumps(existing_meanings, ensure_ascii=False),
        new_meaning=new_meaning,
    )
    response = await _llm.ainvoke(prompt_text)

    # LangChain 可能返回 list 类型的 content
    raw_content = response.content
    if isinstance(raw_content, list):
        raw_content = " ".join(
            str(block.get("text", "") if isinstance(block, dict) else block)
            for block in raw_content
        )

    result = _extract_json_object(raw_content)

    if result:
        return (
            bool(result.get("should_merge", False)),
            int(result.get("matched_index", -1)),
            str(result.get("reason", "")),
        )

    # 回退时打印原始响应便于调试
    print(f"[WARN] LLM 释义归并 JSON 提取失败，原始响应:\n{raw_content[:500]}")

    if new_meaning in existing_meanings:
        return True, existing_meanings.index(new_meaning), "exact match fallback"
    return False, -1, "parse error fallback"


# ── 近义词中文释义推荐（top-3）───────────────────────────

async def generate_synonym_meanings(
    synonym_word: str,
    source_meanings: list[dict],        # [{"pos": "noun", "meaning": "中文"}, ...]
    synonym_existing: list[dict] | None = None,
) -> list[dict]:
    """LLM 为近义/反义词生成 3 个中文释义建议。

    优先从已有释义选，不足时 LLM 根据英文词翻译。
    Returns: [{"pos": "noun", "meaning": "中文"}, ...] 最多3个
    """
    if synonym_existing is None:
        synonym_existing = []

    src_text = "\n".join(
        f"- [{m.get('pos', '?')}] {m.get('meaning', '')}" for m in source_meanings
    ) if source_meanings else "（无）"
    existing_text = "\n".join(
        f"- [{m.get('pos', '?')}] {m.get('meaning', '')}" for m in synonym_existing
    ) if synonym_existing else "（无已有释义）"

    prompt = (
        f"源单词的释义（中文）：\n{src_text}\n\n"
        f"英文单词 '{synonym_word}' 在图库中的已有释义：\n{existing_text}\n\n"
        f"请为 '{synonym_word}' 推荐 1 到 3 个最准确的中文释义。"
        "优先从已有释义中选，如果没有合适的，请你根据英文单词的含义翻译。\n"
        '输出纯 JSON：{"meanings": [{"pos": "noun|verb|adjective|adverb", "meaning": "中文"}, ...]}\n'
    )

    response = await _llm.ainvoke(prompt)
    raw = response.content
    if isinstance(raw, list):
        raw = " ".join(str(b.get("text", "") if isinstance(b, dict) else b) for b in raw)

    result = _extract_json_object(raw)
    if result and "meanings" in result:
        meanings = result["meanings"]
        if isinstance(meanings, list):
            filtered = [
                m for m in meanings
                if isinstance(m, dict) and m.get("meaning") and m.get("pos") in ("noun", "verb", "adjective", "adverb")
            ][:3]
            if filtered:
                return filtered

    return synonym_existing[:3] if synonym_existing else []


# ── 英文释义选择（Datamuse → LLM 挑选最佳匹配）─────────

async def select_best_definition(word: str, chinese_meaning: str, english_definitions: list[str]) -> str:
    """LLM 从 Datamuse 英文释义列表中，选出最匹配用户中文释义的一条。

    重要约束：LLM 只能从给定列表中选择，不允许自己生成英文释义。
    返回选中的英文释义文本。如果列表为空或 LLM 失败，返回空字符串。
    """
    if not english_definitions:
        return ""

    numbered = "\n".join(f"{i}. {d}" for i, d in enumerate(english_definitions))
    prompt = (
        f"英文单词: {word}\n"
        f"用户提供的中文释义: {chinese_meaning}\n\n"
        f"以下是从词典中查询到的该单词的英文释义列表：\n{numbered}\n\n"
        f"请选出与中文释义「{chinese_meaning}」含义最匹配的一条英文释义。\n"
        f"必须从上述列表中选择，不允许自己生成或修改英文释义。\n\n"
        f'输出纯 JSON: {{"selected_index": 数字, "reason": "简短理由"}}'
    )

    response = await _llm.ainvoke(prompt)
    raw = response.content
    if isinstance(raw, list):
        raw = " ".join(str(b.get("text", "") if isinstance(b, dict) else b) for b in raw)

    result = _extract_json_object(raw)
    if result and "selected_index" in result:
        idx = int(result.get("selected_index", 0))
        if 0 <= idx < len(english_definitions):
            print(f"[INFO] LLM 选中英文释义 #{idx}: {english_definitions[idx][:80]}...")
            return english_definitions[idx]

    print(f"[WARN] LLM 英文释义选择失败，回退到第0条。原始响应:\n{raw[:300]}")
    return english_definitions[0] if english_definitions else ""


# ── 词性校验 ─────────────────────────────────────────────

async def validate_pos(word: str, pos: str, meaning: str, existing_meanings: list[str] | None = None) -> dict:
    """LLM 校验词性+语义+归并相似度。

    接收该单词该 POS 下已有释义列表（可选），用于归并判断。
    Returns: {"valid": bool, "suggested_pos": str, "merge_hint": str, "reason": str}
        valid=false → 词性或语义错误，阻止添加
        valid=true, merge_hint 为空 → 直接通过
        valid=true, merge_hint 非空 → 通过但提示与已有释义相似，将归并
    """
    if existing_meanings is None:
        existing_meanings = []

    pos_map = {
        "noun": "名词", "verb": "动词",
        "adjective": "形容词", "adverb": "副词",
    }
    existing_str = "、".join(f'"{m}"' for m in existing_meanings) if existing_meanings else "（无）"

    prompt = (
        f"判断以下英文单词的中文释义是否合理（三项检查：词性 + 语义 + 归并相似度）。\n\n"
        f"英文单词: {word}\n"
        f"用户选择的词性: {pos_map.get(pos, pos)} ({pos})\n"
        f"用户输入的中文释义: {meaning}\n"
        f"该词在该词性下已有释义: {existing_str}\n\n"
        "检查规则：\n"
        "1. 词性检查：「释义的中文意思」是否匹配「所选词性」。"
        "例如 'n.高兴的' 不合理（'高兴的'是形容词）。\n"
        "2. 语义检查：「释义的中文意思」是否是该英文单词的合理翻译。"
        "例如 'sad' + '开心的' 不合理。\n"
        "3. 归并检查：如果该释义与已有释义语义高度相似(>90%)，"
        "在 merge_hint 中提示如'与已有释义\"高兴的\"高度相似，将归并'。\n\n"
        "规则：词性或语义任一不通过 → valid=false。"
        "词性和语义都通过 → valid=true。"
        "valid=true 时如有归并 → merge_hint 建议归并到哪个已有释义。"
        "无已有释义或不需要归并 → merge_hint 为空字符串。\n\n"
        '输出纯 JSON: {"valid": true|false, "suggested_pos": "noun|verb|adjective|adverb", "merge_hint": "", "reason": "简短理由"}\n'
        "如果 valid 为 true，suggested_pos 和 reason 可为空字符串。"
    )

    response = await _llm.ainvoke(prompt)
    raw = response.content
    if isinstance(raw, list):
        raw = " ".join(str(b.get("text", "") if isinstance(b, dict) else b) for b in raw)

    result = _extract_json_object(raw)
    if result and "valid" in result:
        return {
            "valid": bool(result.get("valid", True)),
            "suggested_pos": str(result.get("suggested_pos", "")),
            "merge_hint": str(result.get("merge_hint", "")),
            "reason": str(result.get("reason", "")),
        }

    print(f"[WARN] LLM 词性校验 JSON 提取失败，原始响应:\n{raw[:300]}")
    return {"valid": True, "suggested_pos": "", "merge_hint": "", "reason": ""}

