"""
Datamuse API 服务 —— 近义词、反义词查询 + 英文释义获取。
文档: https://www.datamuse.com/api/
"""
import httpx
from word_study.config import DATAMUSE_BASE_URL
from word_study.services.cache_service import get_cached_or_none, set_cached


async def fetch_synonyms(word: str) -> list[str]:
    """获取单词的近义词列表（仅返回单词名）。"""
    results = await _fetch_with_cache(word, "synonyms")
    return [r.get("word", "") for r in results if r.get("word") != word]


async def fetch_antonyms(word: str) -> list[str]:
    """获取单词的反义词列表（仅返回单词名）。"""
    results = await _fetch_with_cache(word, "antonyms")
    return [r.get("word", "") for r in results if r.get("word") != word]


async def fetch_definitions(word: str) -> list[str]:
    """获取单词的英文释义定义列表（Datamuse md=d 端点）。

    调用 /words?sp={word}&md=d，解析 defs 字段，返回纯释义文本列表。
    """
    cached = get_cached_or_none(word, "definitions")
    if cached is not None:
        return [d if isinstance(d, str) else str(d) for d in cached]

    url = f"{DATAMUSE_BASE_URL}/words"
    params = {"sp": word, "md": "d", "max": 10}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data: list[dict] = resp.json()
    except Exception:
        data = []

    defs: list[str] = []
    if data:
        raw_defs = data[0].get("defs", []) if data else []
        for d in raw_defs:
            # 格式: "pos\tdefinition text"
            parts = d.split("\t", 1)
            if len(parts) == 2:
                defs.append(parts[1].strip())

    if defs:
        set_cached(word, "definitions", defs)

    return defs


_API_CONFIG = {
    "synonyms": {"params_key": "rel_syn", "extra": {"max": 10}},
    "antonyms": {"params_key": "rel_ant", "extra": {"max": 10}},
}


async def _fetch_with_cache(word: str, api_type: str) -> list[dict]:
    cached = get_cached_or_none(word, api_type)
    if cached is not None:
        return cached

    config = _API_CONFIG[api_type]
    params: dict = {config["params_key"]: word, **config["extra"]}
    url = f"{DATAMUSE_BASE_URL}/words"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data: list[dict] = resp.json()
    except Exception:
        data = []

    if data:
        set_cached(word, api_type, data)

    return data
