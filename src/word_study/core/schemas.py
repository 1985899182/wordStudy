"""
Pydantic 数据模型 —— 前后端数据交换格式。

v2 新增: ErrorResponse 统一错误格式，所有 API 错误都走此模型返回。
"""
from __future__ import annotations

from pydantic import BaseModel


# ══════════════════════════════════════════════════════════
# 请求模型
# ══════════════════════════════════════════════════════════

class ExtractRequest(BaseModel):
    text: str


class ManualMeaning(BaseModel):
    pos: str       # "noun" | "verb" | "adjective" | "adverb"
    meaning: str   # 中文释义
    group_key: str = ""  # 同组标记，非空时同一 group_key 的释义共享一个 Meaning 节点


class WordRelItem(BaseModel):
    """保存单词时附带要建立的近义/反义关系。"""
    target_word: str
    rel_type: str  # "synonym" | "antonym"


class WordSaveItem(BaseModel):
    word: str
    resource: str
    manual_meanings: list[ManualMeaning]
    relations: list[WordRelItem] = []   # 保存单词后建立的近义/反义关系


class SaveRequest(BaseModel):
    selections: list[WordSaveItem]


class AIQueryRequest(BaseModel):
    question: str


class PosValidateRequest(BaseModel):
    word: str
    pos: str
    meaning: str


class SplitMeaningsRequest(BaseModel):
    """多释义拆分请求。"""
    word: str
    pos: str          # "noun" | "verb" | "adjective" | "adverb"
    raw_input: str    # 用户原始输入，可能含分隔符


class SplitMeaningsGroup(BaseModel):
    """一组释义（极相近的归入同一组，share_node=True 则共享节点）。"""
    meanings: list[str]
    share_node: bool


class SplitMeaningsWarning(BaseModel):
    """某条释义 POS 校验未通过。"""
    meaning: str
    valid: bool = False
    suggested_pos: str = ""
    reason: str = ""


class SplitMeaningsResponse(BaseModel):
    groups: list[SplitMeaningsGroup] = []
    warnings: list[SplitMeaningsWarning] = []


class QueryTimesRequest(BaseModel):
    """单词/释义选择/取消选择。"""
    word: str
    action: str = "select"
    pos: str = ""
    meaning: str = ""


class WordRelRequest(BaseModel):
    """Word 节点直连 SYNONYM/ANTONYM。"""
    word1: str
    word2: str
    rel_type: str  # "synonym" | "antonym"


class DeleteMeaningRequest(BaseModel):
    """删除单词的某条释义。"""
    word: str
    pos: str       # "noun" | "verb" | "adjective" | "adverb"
    meaning: str   # 要删除的中文释义


class RecentActivity(BaseModel):
    word: str
    meaning: str
    source: str
    time: str


class StatsResponse(BaseModel):
    total_words: int = 0
    total_meanings: int = 0
    total_synonyms: int = 0
    total_antonyms: int = 0
    recent_activity: list[RecentActivity] = []


# ══════════════════════════════════════════════════════════
# 响应模型
# ══════════════════════════════════════════════════════════

class ExistingMeaning(BaseModel):
    meaning: str
    query_times: int


class WordInfo(BaseModel):
    word: str
    exists: bool
    existing_meanings: dict[str, list[ExistingMeaning]]


class ExtractResponse(BaseModel):
    words: list[WordInfo]
    spell_warnings: list[dict]


class PosValidateResponse(BaseModel):
    valid: bool
    suggested_pos: str = ""
    merge_hint: str = ""
    reason: str = ""


class FetchSuggestionsResponse(BaseModel):
    words: list[str] = []


class SaveResultItem(BaseModel):
    word: str
    status: str
    message: str


class SaveResponse(BaseModel):
    results: list[SaveResultItem]


class AIQueryResponse(BaseModel):
    question: str
    cypher: str = ""
    result: list | str = []


# ══════════════════════════════════════════════════════════
# v2: 统一错误响应模型
# ══════════════════════════════════════════════════════════

class ErrorDetail(BaseModel):
    """单条错误详情，用于批量操作（如保存多个单词时部分失败）."""
    field: str = ""         # 出错字段名
    message: str            # 错误描述


class ErrorResponse(BaseModel):
    """API 统一错误响应.

    所有 API 端点在出错时均返回此模型，前端可统一处理.
    """
    ok: bool = False
    error_code: str = "UNKNOWN"     # 机器可读错误码: VALIDATION_ERROR, NOT_FOUND, LLM_ERROR..
    message: str                    # 人类可读错误描述
    detail: str | None = None       # 详细上下文（可选）
    errors: list[ErrorDetail] = []  # 多条错误时使用
