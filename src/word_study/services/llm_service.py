"""LLM 服务 —— DeepSeek API 封装：提取、归并、校验、AI 查询。

v2 变更:
  1. _extract_json_array 兜底用 dict.fromkeys 保持原始顺序（修复 set 乱序）
  2. get_cypher_qa() 接入 read_only_cypher.txt 只读 prompt
  3. 所有 LLM 调用新增 @retry_llm 指数退避重试装饰器
  4. print() 替换为结构化 logger (logging_utils)
"""
from __future__ import annotations

import asyncio
import json
import re
from functools import wraps

from word_study.config import get_settings, BASE_DIR
from word_study.deps import get_llm, get_cypher_qa as _deps_cypher_qa
from word_study.services.cache_service import get_l1_cache, set_l1_cache
from word_study.utils.logging_utils import get_logger

_logger = get_logger(__name__)

# LLM 实例委托给 deps.py 统一管理（支持测试 dependency_overrides）
_llm = None  # 惰性获取


def _get_llm():
    """获取 LLM 实例（委托 deps.py 管理）."""
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm


def get_cypher_qa():
    """获取 GraphCypherQAChain（委托 deps.py 管理）."""
    return _deps_cypher_qa()

# Prompt 模板目录
_PROMPTS_DIR = BASE_DIR / "prompts"


def _load_prompt(filename: str) -> str:
    """从 prompts/ 目录加载文本模板。"""
    path = _PROMPTS_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    _logger.warning("prompt 文件不存在 | path=%s", path)
    return ""


# 预加载所有 prompt 模板
_EXTRACT_PROMPT = _load_prompt("extract_words.txt")
_MERGE_PROMPT_TEMPLATE = _load_prompt("merge_meaning.txt")
# read_only_cypher.txt 由 deps.py 直接加载


# ══════════════════════════════════════════════════════════
# LLM 调用重试装饰器
# ══════════════════════════════════════════════════════════

def retry_llm(func):
    """LLM 调用指数退避重试.

    当 LLM API 因网络波动/限流/临时错误失败时，
    按 2^n 秒递增等待后重试，最多重试 N 次（从 settings 读取）。
    所有重试均失败则重新抛出最后一次异常。
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        s = get_settings()
        max_retries = s.llm_retry_count
        backoff = s.llm_retry_backoff
        last_exc = None
        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    wait = backoff ** attempt
                    _logger.warning(
                        "LLM 调用重试 (%d/%d)，%.1fs 后重试 | func=%s error=%s",
                        attempt + 1, max_retries + 1, wait, func.__name__, exc
                    )
                    await asyncio.sleep(wait)
                else:
                    _logger.error(
                        "LLM 调用最终失败，已用尽 %d 次尝试 | func=%s error=%s",
                        max_retries + 1, func.__name__, exc
                    )
                    raise
        raise last_exc  # type: ignore[misc]
    return wrapper


# ══════════════════════════════════════════════════════════
# 单词提取
# ══════════════════════════════════════════════════════════

@retry_llm
async def extract_words(text: str) -> list[str]:
    """用 LLM 从输入文本中提取英文单词。带 L1 内存缓存."""
    cache_key = f"extract:{text.strip().lower()}"
    cached = get_l1_cache(cache_key)
    if cached:
        try:
            return json.loads(cached)
        except json.JSONDecodeError:
            _logger.debug("L1 缓存解码失败，重新提取 | key_hash=%s", cache_key[:60])

    messages = [("system", _EXTRACT_PROMPT), ("user", text)]
    response = await _get_llm().ainvoke(messages)
    content = _extract_json_array(response.content)

    try:
        set_l1_cache(cache_key, json.dumps(content))
    except Exception:
        pass

    return content


# ══════════════════════════════════════════════════════════
# JSON 解析辅助函数
# ══════════════════════════════════════════════════════════

def _extract_json_array(raw: str | list) -> list[str]:
    """从 LLM 响应中提取 JSON 数组，多重降级确保不崩溃.

    策略:
      1. 正则匹配 [...]
      2. 正则扫描所有 2+ 字母英文词 → 用 dict.fromkeys 保持顺序去重
         （修复: 原版用 set() 会打乱词序）
    """
    if isinstance(raw, list):
        return [str(item).strip().lower() for item in raw if str(item).strip()]
    if isinstance(raw, str):
        # 尝试 1: 正则匹配 JSON 数组
        match = re.search(r"\[(.*?)\]", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    return [str(i).strip().lower() for i in parsed if str(i).strip()]
            except json.JSONDecodeError:
                _logger.debug("JSON 数组正则匹配失败，回退到单词扫描")

        # 尝试 2: 扫描英文词，保持出现顺序
        words = re.findall(r"[a-zA-Z]{2,}", raw)
        # 用 dict.fromkeys 去重但保持顺序（修复原版 set() 乱序问题）
        ordered = list(dict.fromkeys(w.lower() for w in words))
        _logger.debug("回退单词扫描: %d 个不重复词（顺序已保持）", len(ordered))
        return ordered
    return []


def _extract_json_object(raw: str | list) -> dict | None:
    """从 LLM 响应中提取 JSON 对象，逐级降级.

    处理: markdown 代码块包裹、Python 风格 bool/None、多余说明文字.
    每个处理步骤失败都有下一步兜底.
    """
    text = raw if isinstance(raw, str) else str(raw)

    # 1. 去掉 markdown 代码块标记
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)

    # 2. 找到最外层 { }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        return None
    json_candidate = text[start:end + 1]

    # 3. 直接解析
    try:
        result = json.loads(json_candidate)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # 4. 修复 Python 风格 True/False/None → JSON
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


# ══════════════════════════════════════════════════════════
# 多释义拆分 + 相似度分组
# ══════════════════════════════════════════════════════════

_DELIMITER_RE = re.compile(r"[；，、;,/]+")


def split_raw_input(raw: str) -> list[str]:
    """按分隔符拆分用户输入的释义字符串.

    分隔符: 中文 ；，、 英文 ;,/ 
    去空白、去重、保持原始顺序.
    """
    parts = _DELIMITER_RE.split(raw)
    seen: set[str] = set()
    result: list[str] = []
    for p in parts:
        p = p.strip()
        if p and p not in seen:
            seen.add(p)
            result.append(p)
    return result


@retry_llm
async def judge_meaning_similarity(meanings: list[str]) -> list[list[str]]:
    """LLM 判断多条中文释义的语义相似度，分组返回.

    Args:
        meanings: 拆分后的释义列表, e.g. ["增加", "添加", "补充"]
    Returns:
        分组列表, e.g. [["增加", "添加"], ["补充"]]
    """
    if len(meanings) <= 1:
        return [meanings] if meanings else []

    meanings_str = "\n".join(f"{i}. {m}" for i, m in enumerate(meanings))
    prompt = (
        "以下是一组中文释义，请判断哪些释义含义极为相近（可共享同一个释义节点），"
        "将它们分到同一组。含义明显不同的分到不同组。\n\n"
        f"释义列表:\n{meanings_str}\n\n"
        "输出纯 JSON 格式（不要 markdown 包裹）:\n"
        '{"groups": [["增加","添加"], ["补充"]]}\n'
        "注意:\n1. 每个释义必须且只能出现在一个组中\n"
        "2. 组内顺序保持原输入顺序\n"
        "3. 极相近的标准: 核心语义相同，仅表述方式不同"
    )

    response = await _get_llm().ainvoke(prompt)
    raw = _flatten_content(response.content)
    result = _extract_json_object(raw)
    if result and "groups" in result:
        groups = result["groups"]
        if isinstance(groups, list) and all(isinstance(g, list) for g in groups):
            return [g for g in groups if g]

    _logger.debug("释义相似度分组 LLM 解析失败，回退为独立分组")
    return [[m] for m in meanings]


# ══════════════════════════════════════════════════════════
# 释义归并判断
# ══════════════════════════════════════════════════════════

@retry_llm
async def judge_meaning_merge(
    existing_meanings: list[str], new_meaning: str
) -> tuple[bool, int, str]:
    """LLM 判断新释义是否应归并到已有释义列表.

    Returns:
        (should_merge, matched_index, reason)
    """
    prompt_text = _MERGE_PROMPT_TEMPLATE.format(
        existing_meanings=json.dumps(existing_meanings, ensure_ascii=False),
        new_meaning=new_meaning,
    )
    response = await _get_llm().ainvoke(prompt_text)
    raw_content = _flatten_content(response.content)
    result = _extract_json_object(raw_content)

    if result:
        return (
            bool(result.get("should_merge", False)),
            int(result.get("matched_index", -1)),
            str(result.get("reason", "")),
        )

    _logger.warning("释义归并 JSON 提取失败 | existing=%s... new=%s",
                     str(existing_meanings)[:80], new_meaning)

    if new_meaning and new_meaning in existing_meanings:
        idx = existing_meanings.index(new_meaning)
        return True, idx, "exact match fallback (JSON parse failed)"
    return False, -1, "parse error fallback"


# ══════════════════════════════════════════════════════════
# 近义词中文释义推荐
# ══════════════════════════════════════════════════════════

async def generate_synonym_meanings(
    synonym_word: str,
    source_meanings: list[dict],
    synonym_existing: list[dict] | None = None,
) -> list[dict]:
    """LLM 为近义/反义词生成最多 3 个中文释义建议.

    Returns:
        [{"pos": "noun", "meaning": "中文"}, ...], 最多 3 个
    """
    if synonym_existing is None:
        synonym_existing = []

    src_text = "\n".join(
        f"- [{m.get('pos', '?')}] {m.get('meaning', '')}"
        for m in source_meanings
    ) if source_meanings else "（无）"
    existing_text = "\n".join(
        f"- [{m.get('pos', '?')}] {m.get('meaning', '')}"
        for m in synonym_existing
    ) if synonym_existing else "（无已有释义）"

    prompt = (
        f"源单词的释义（中文）:\n{src_text}\n\n"
        f"英文单词 '{synonym_word}' 在图库中的已有释义:\n{existing_text}\n\n"
        f"请为 '{synonym_word}' 推荐 1 到 3 个最准确的中文释义。"
        "优先从已有释义中选，如果没有合适的，请你根据英文单词的含义翻译。\n"
        '输出纯 JSON: {"meanings": [{"pos": "noun|verb|adjective|adverb", "meaning": "中文"}, ...]}\n'
    )

    response = await _get_llm().ainvoke(prompt)
    raw = _flatten_content(response.content)
    result = _extract_json_object(raw)
    if result and "meanings" in result:
        meanings = result["meanings"]
        if isinstance(meanings, list):
            filtered = [
                m for m in meanings
                if isinstance(m, dict)
                and m.get("meaning")
                and m.get("pos") in ("noun", "verb", "adjective", "adverb")
            ][:3]
            if filtered:
                return filtered

    return synonym_existing[:3] if synonym_existing else []


# ══════════════════════════════════════════════════════════
# 英文释义选择（Datamuse -> LLM 挑选最佳匹配）
# ══════════════════════════════════════════════════════════

@retry_llm
async def select_best_definition(
    word: str, chinese_meaning: str, english_definitions: list[str]
) -> str:
    """LLM 从 Datamuse 英文释义列表中选出最匹配中文释义的一条.

    核心约束: LLM 只能从给定列表中选择，不允许自己生成英文释义.
    返回选中的英文释义文本，列表为空或 LLM 失败返回空字符串.
    """
    if not english_definitions:
        return ""

    numbered = "\n".join(f"{i}. {d}" for i, d in enumerate(english_definitions))
    prompt = (
        f"英文单词: {word}\n"
        f"用户提供的中文释义: {chinese_meaning}\n\n"
        f"以下是从词典中查询到的该单词的英文释义列表:\n{numbered}\n\n"
        f"请选出与中文释义「{chinese_meaning}」含义最匹配的一条英文释义。\n"
        "必须从上述列表中选择，不允许自己生成或修改英文释义。\n\n"
        '输出纯 JSON: {"selected_index": 数字, "reason": "简短理由"}'
    )

    response = await _get_llm().ainvoke(prompt)
    raw = _flatten_content(response.content)
    result = _extract_json_object(raw)
    if result and "selected_index" in result:
        idx = int(result.get("selected_index", 0))
        if 0 <= idx < len(english_definitions):
            _logger.info("LLM 选中英文释义 #%d | word=%s def=%s...",
                         idx, word, english_definitions[idx][:60])
            return english_definitions[idx]

    _logger.warning("LLM 英文释义选择失败，回退到第 0 条 | word=%s", word)
    return english_definitions[0]


# ══════════════════════════════════════════════════════════
# 词性校验
# ══════════════════════════════════════════════════════════

async def validate_pos(
    word: str, pos: str, meaning: str,
    existing_meanings: list[str] | None = None,
) -> dict:
    """LLM 校验词性 + 语义 + 归并相似度（三合一）.

    Returns:
        {"valid": bool, "suggested_pos": str, "merge_hint": str, "reason": str}
        valid=false -> 词性或语义错误，阻止添加
        valid=true + merge_hint 为空 -> 直接通过
        valid=true + merge_hint 非空 -> 通过但提示与已有释义相似
    """
    if existing_meanings is None:
        existing_meanings = []

    pos_map = {"noun": "名词", "verb": "动词", "adjective": "形容词", "adverb": "副词"}
    existing_str = "、".join(f'"{m}"' for m in existing_meanings) if existing_meanings else "（无）"

    prompt = (
        f"判断以下英文单词的中文释义是否合理（三项检查: 词性 + 语义 + 归并相似度）。\n\n"
        f"英文单词: {word}\n"
        f"用户选择的词性: {pos_map.get(pos, pos)} ({pos})\n"
        f"用户输入的中文释义: {meaning}\n"
        f"该词在该词性下已有释义: {existing_str}\n\n"
        "检查规则:\n"
        "1. 词性检查: 「释义的中文意思」是否匹配「所选词性」。"
        "例如 'n.高兴的' 不合理（'高兴的'是形容词）。\n"
        "2. 语义检查: 「释义的中文意思」是否是该英文单词的合理翻译。"
        "例如 'sad' + '开心的' 不合理。\n"
        "3. 归并检查: 如果该释义与已有释义语义高度相似(>90%)，"
        "在 merge_hint 中提示。\n\n"
        "规则: 词性或语义任一不通过 -> valid=false。"
        "词性和语义都通过 -> valid=true。"
        "valid=true 时如有归并 -> merge_hint 建议归并到哪个已有释义。"
        "无已有释义或不需要归并 -> merge_hint 为空字符串。\n\n"
        "输出纯 JSON: "
        '{"valid": true|false, "suggested_pos": "noun|verb|adjective|adverb", '
        '"merge_hint": "", "reason": "简短理由"}\n'
        "如果 valid 为 true，suggested_pos 和 reason 可为空字符串。"
    )

    response = await _get_llm().ainvoke(prompt)
    raw = _flatten_content(response.content)
    result = _extract_json_object(raw)
    if result and "valid" in result:
        return {
            "valid": bool(result.get("valid", True)),
            "suggested_pos": str(result.get("suggested_pos", "")),
            "merge_hint": str(result.get("merge_hint", "")),
            "reason": str(result.get("reason", "")),
        }

    # 最坏情况: JSON 完全无法解析，放行避免阻塞用户
    _logger.warning("POS 校验 JSON 提取失败，跳过校验 | word=%s pos=%s", word, pos)
    return {"valid": True, "suggested_pos": "", "merge_hint": "", "reason": ""}


# ══════════════════════════════════════════════════════════
# 内部工具
# ══════════════════════════════════════════════════════════

def _flatten_content(content):
    """将 LangChain 可能的 list 型 content 展平为纯字符串."""
    if isinstance(content, list):
        return " ".join(
            str(block.get("text", "") if isinstance(block, dict) else block)
            for block in content
        )
    return str(content)
