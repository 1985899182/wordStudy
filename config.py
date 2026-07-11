"""
应用配置模块 —— 从 .env 加载所有环境变量，禁止硬编码。
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载项目根目录的 .env 文件
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


# ── DeepSeek ────────────────────────────────────────────
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# ── DashScope (向量嵌入) ────────────────────────────────
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

# ── Neo4j ───────────────────────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# ── SQLite ──────────────────────────────────────────────
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", str(BASE_DIR / "word_study.db"))

# ── 日志 ────────────────────────────────────────────────
LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", str(BASE_DIR / "cypher_query.log"))

# ── Datamuse API ────────────────────────────────────────
DATAMUSE_BASE_URL = os.getenv("DATAMUSE_BASE_URL", "https://api.datamuse.com")

# ── Source 来源常量 ──────────────────────────────────────
SOURCE_MANUAL = "手动录入"
SOURCE_DATAMUSE = "datamuse_api_v1_1"


def validate_config() -> list[str]:
    """校验必要配置项，返回缺失项列表。"""
    missing = []
    if not DEEPSEEK_API_KEY:
        missing.append("DEEPSEEK_API_KEY")
    if not NEO4J_PASSWORD:
        missing.append("NEO4J_PASSWORD")
    return missing
