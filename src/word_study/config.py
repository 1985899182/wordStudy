"""
应用配置模块 —— Pydantic Settings 驱动的类型安全配置。

设计思路:
  - 所有环境变量由 Settings 统一管理，禁止在业务代码中硬编码或 os.getenv
  - 必填项缺失时在应用启动阶段立即报错（fail-fast），而非运行时静默失效
  - 提供 settings 单例供全项目使用，路径常量保持独立

Why Pydantic Settings:
  - 自动从 .env 文件加载，无需手动 load_dotenv
  - 类型校验 + 字段级验证，杜绝 'true' vs True 这类低级错误
  - 支持 SecretStr 避免敏感值在日志/异常中泄露
"""
from __future__ import annotations

from pathlib import Path
from functools import lru_cache

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ══════════════════════════════════════════════════════════
# 路径常量（不依赖环境变量，直接计算）
# ══════════════════════════════════════════════════════════

PACKAGE_DIR = Path(__file__).resolve().parent          # → src/word_study/
PROJECT_ROOT = PACKAGE_DIR.parent.parent                # → 项目根目录
ENV_FILE = PROJECT_ROOT / ".env"                        # .env 文件路径

# 兼容旧代码的别名：BASE_DIR 指向包目录（static/templates/prompts 在此下）
BASE_DIR = PACKAGE_DIR

# 常量：词汇来源标签
SOURCE_MANUAL = "手动录入"
SOURCE_DATAMUSE = "datamuse_api_v1_1"


# ══════════════════════════════════════════════════════════
# Settings —— 类型安全的环境变量管理器
# ══════════════════════════════════════════════════════════

class Settings(BaseSettings):
    """
    应用全局配置，所有值从 .env 文件 + 环境变量中加载。

    使用方式:
        from word_study.config import get_settings
        s = get_settings()
        print(s.neo4j_uri)        # bolt://localhost:7687
        print(s.deepseek_api_key) # SecretStr —— 需 .get_secret_value() 取原文
    """

    # ── Pydantic Settings 元配置 ────────────────────────
    model_config = SettingsConfigDict(
        # 自动从项目根目录 .env 加载，优先级低于系统环境变量
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        # 对未知字段宽容（避免添加新 env 后启动报错）
        extra="ignore",
        case_sensitive=False,
    )

    # ── DeepSeek LLM ────────────────────────────────────
    deepseek_api_key: str = Field(
        default="",
        description="DeepSeek API Key（必填，否则 LLM 功能不可用）",
    )
    deepseek_model: str = Field(
        default="deepseek-chat",
        description="对话模型名",
    )

    # ── DashScope 向量嵌入 ─────────────────────────────
    dashscope_api_key: str = Field(
        default="",
        description="阿里云 DashScope API Key（用于 text-embedding-v2）",
    )

    # ── Neo4j 图数据库 ──────────────────────────────────
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        description="Neo4j Bolt 连接 URI",
    )
    neo4j_username: str = Field(
        default="neo4j",
        description="Neo4j 用户名",
    )
    neo4j_password: str = Field(
        default="",
        description="Neo4j 密码（必填，否则图数据库不可用）",
    )
    neo4j_database: str = Field(
        default="neo4j",
        description="Neo4j 数据库名",
    )

    # ── SQLite ──────────────────────────────────────────
    sqlite_db_path: str = Field(
        default="",
        description="SQLite 数据库文件路径（默认 = 项目根/word_study.db）",
    )

    # ── 日志 ────────────────────────────────────────────
    log_file_path: str = Field(
        default="",
        description="Cypher 操作日志文件路径",
    )
    log_level: str = Field(
        default="INFO",
        description="应用日志级别: DEBUG | INFO | WARNING | ERROR",
    )

    # ── Datamuse 外部词典 API ───────────────────────────
    datamuse_base_url: str = Field(
        default="https://api.datamuse.com",
        description="Datamuse API 基础地址",
    )

    # ── 向量相似度阈值 ─────────────────────────────────
    vector_similarity_threshold: float = Field(
        default=0.85,
        ge=0.0, le=1.0,
        description="释义归并时向量余弦相似度的最低阈值",
    )

    # ── LLM 调用重试 ────────────────────────────────────
    llm_retry_count: int = Field(
        default=3,
        ge=0, le=10,
        description="LLM 调用失败时的最大重试次数",
    )
    llm_retry_backoff: float = Field(
        default=2.0,
        ge=0.1,
        description="LLM 重试指数退避的底数",
    )
    llm_request_timeout: float = Field(
        default=60.0,
        ge=1.0,
        description="单次 LLM 请求超时秒数",
    )

    # ── 域校验器 ────────────────────────────────────────

    @field_validator("sqlite_db_path", mode="before")
    @classmethod
    def _default_sqlite_path(cls, v: str) -> str:
        """sqlite_db_path 未配置时回退到项目根目录下的默认路径。"""
        if not v:
            return str(PROJECT_ROOT / "word_study.db")
        return v

    @field_validator("log_file_path", mode="before")
    @classmethod
    def _default_log_path(cls, v: str) -> str:
        """log_file_path 未配置时回退到项目根目录下的默认路径。"""
        if not v:
            return str(PROJECT_ROOT / "cypher_query.log")
        return v

    # ── 便捷属性：兼容旧代码的大写属性名 ──────────────────

    @property
    def DEEPSEEK_API_KEY(self) -> str: return self.deepseek_api_key
    @property
    def DEEPSEEK_MODEL(self) -> str: return self.deepseek_model
    @property
    def DASHSCOPE_API_KEY(self) -> str: return self.dashscope_api_key
    @property
    def NEO4J_URI(self) -> str: return self.neo4j_uri
    @property
    def NEO4J_USERNAME(self) -> str: return self.neo4j_username
    @property
    def NEO4J_PASSWORD(self) -> str: return self.neo4j_password
    @property
    def NEO4J_DATABASE(self) -> str: return self.neo4j_database
    @property
    def SQLITE_DB_PATH(self) -> str: return self.sqlite_db_path
    @property
    def LOG_FILE_PATH(self) -> str: return self.log_file_path
    @property
    def DATAMUSE_BASE_URL(self) -> str: return self.datamuse_base_url


# ══════════════════════════════════════════════════════════
# 惰性单例 —— 首次访问时加载并校验，加载后不可变
# ══════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    获取 Settings 惰性单例。

    首次调用时从 .env 加载所有配置并做 Pydantic 校验；
    后续调用直接返回缓存的实例，全局唯一。
    """
    return Settings()


def validate_config() -> list[str]:
    """
    启动时校验关键配置项，返回缺失列表，兼容旧调用方。

    如果 Pydantic 校验本身失败（例如布尔字段填了 "maybe"），
    会在创建 Settings 实例时直接抛出 ValidationError，应用不应继续启动。
    """
    try:
        s = get_settings()
    except ValidationError as exc:
        # 解析 Pydantic 校验错误，提取用户可读的信息
        errors: list[str] = []
        for e in exc.errors():
            loc = ".".join(str(x) for x in e["loc"])
            errors.append(f"{loc}: {e['msg']}")
        # 打印后重新抛出，让调用方决定是否继续
        print(f"[FATAL] 配置校验失败:\n" + "\n".join(f"  - {err}" for err in errors))
        return errors

    missing: list[str] = []
    if not s.deepseek_api_key:
        missing.append("DEEPSEEK_API_KEY")
    if not s.neo4j_password:
        missing.append("NEO4J_PASSWORD")
    return missing


# ══════════════════════════════════════════════════════════
# 模块级大写快捷变量 —— 兼容旧代码 from config import XXX 的写法
# ══════════════════════════════════════════════════════════
# 策略: 导入时即调用 get_settings() 加载配置，与旧行为一致。
#       如果 .env 不存在/关键字段缺失，赋默认值，由 validate_config() 负责告警。

_settings: Settings
try:
    _settings = get_settings()
except ValidationError:
    # ── 最坏情况：.env 完全不存在或格式错乱，用全默认值 ─
    _settings = Settings()

DEEPSEEK_API_KEY   = _settings.deepseek_api_key
DEEPSEEK_MODEL      = _settings.deepseek_model
DASHSCOPE_API_KEY   = _settings.dashscope_api_key
NEO4J_URI           = _settings.neo4j_uri
NEO4J_USERNAME      = _settings.neo4j_username
NEO4J_PASSWORD      = _settings.neo4j_password
NEO4J_DATABASE      = _settings.neo4j_database
SQLITE_DB_PATH      = _settings.sqlite_db_path
LOG_FILE_PATH        = _settings.log_file_path
DATAMUSE_BASE_URL   = _settings.datamuse_base_url
