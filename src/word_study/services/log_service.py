"""
Cypher 查询日志服务 —— 将每次执行的 Cypher 记录到文件。
"""
import json
from datetime import datetime, timezone, timedelta
from word_study.config import LOG_FILE_PATH

# 东八区时区
_TZ = timezone(timedelta(hours=8))


def log_cypher(cypher: str, params: dict, *, result: str = ""):
    """将 Cypher 语句及参数写入日志文件。"""
    entry = {
        "timestamp": datetime.now(_TZ).isoformat(),
        "cypher": cypher,
        "params": params,
        "result": result,
    }
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass  # 日志写入失败不应中断主流程
