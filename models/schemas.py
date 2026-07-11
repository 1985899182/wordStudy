"""
Pydantic 数据模型 —— 前后端数据交换格式。
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
