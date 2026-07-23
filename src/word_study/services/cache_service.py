"""
缓存服务 —— L1 内存缓存 + L2 SQLite 永久缓存。
L1: 内存 dict，进程重启清空，无伤大雅。
L2: SQLite 表 datamuse_cache，永久保留不清理。
"""
import json
from word_study.services.sqlite_client import get_cache, set_cache


# ── L1 内存缓存 ─────────────────────────────────────────

_l1_fallback: dict[str, str] = {}


def get_l1_cache(key: str) -> str | None:
    return _l1_fallback.get(key)


def set_l1_cache(key: str, value: str):
    _l1_fallback[key] = value


# ── L2 SQLite 缓存：Datamuse API 响应 ──────────────────

def get_datamuse_cache(word: str, api_type: str) -> dict | None:
    """从 L2 SQLite 缓存读取 Datamuse API 响应。"""
    raw = get_cache(word, api_type)
    if raw:
        return json.loads(raw)
    return None


def set_datamuse_cache(word: str, api_type: str, response: dict):
    """将 Datamuse API 响应写入 L2 SQLite 缓存。"""
    set_cache(word, api_type, json.dumps(response, ensure_ascii=False))


def get_cached_or_none(word: str, api_type: str) -> list[dict] | None:
    """组合查询 L1 + L2 缓存。返回 Datamuse 响应列表或 None。"""
    # 先查 L1
    l1_key = f"datamuse:{word}:{api_type}"
    l1_result = get_l1_cache(l1_key)
    if l1_result:
        return json.loads(l1_result)

    # 再查 L2
    l2_result = get_datamuse_cache(word, api_type)
    if l2_result is not None:
        # 回写到 L1
        set_l1_cache(l1_key, json.dumps(l2_result, ensure_ascii=False))
        return l2_result

    return None


def set_cached(word: str, api_type: str, response: list[dict]):
    """写入 L1 + L2 缓存。"""
    l1_key = f"datamuse:{word}:{api_type}"
    set_l1_cache(l1_key, json.dumps(response, ensure_ascii=False))
    set_datamuse_cache(word, api_type, response)
