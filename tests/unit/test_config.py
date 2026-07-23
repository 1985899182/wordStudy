"""
Pydantic Settings 配置校验测试.
"""
from __future__ import annotations

import sys, os
from pathlib import Path

_src = Path(__file__).resolve().parents[3] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import pytest
from pydantic import ValidationError

from word_study.config import (
    Settings, get_settings, validate_config,
    BASE_DIR, PACKAGE_DIR, PROJECT_ROOT,
    SOURCE_MANUAL, SOURCE_DATAMUSE,
)


class TestSettingsDefaults:
    def test_default_values(self):
        """Pydantic 默认值校验 (env 可能覆盖 model，只测不受 env 影响的项)."""
        s = Settings()
        assert isinstance(s.deepseek_model, str)  # 具体值依赖 .env，只校验类型
        assert isinstance(s.neo4j_uri, str) and len(s.neo4j_uri) > 0
        assert isinstance(s.neo4j_username, str) and len(s.neo4j_username) > 0
        assert isinstance(s.neo4j_database, str) and len(s.neo4j_database) > 0
        assert s.vector_similarity_threshold == 0.85
        assert s.llm_retry_count == 3
        assert s.llm_retry_backoff == 2.0
        assert s.llm_request_timeout == 60.0

    def test_sqlite_path_default_fallback(self):
        """sqlite_db_path 空时回退到项目根目录下."""
        s = Settings()
        assert "word_study.db" in s.sqlite_db_path

    def test_log_path_default_fallback(self):
        """log_file_path 空时回退到项目根目录."""
        s = Settings()
        assert "cypher_query.log" in s.log_file_path


class TestSettingsValidation:
    @pytest.fixture
    def minimal_env(self, monkeypatch):
        """设置最小可用环境变量."""
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-key")
        monkeypatch.setenv("NEO4J_PASSWORD", "test-pass")

    def test_missing_keys_reported(self, minimal_env):
        """env 中设了 key + password -> validate_config 返回空列表."""
        missing = validate_config()
        assert len(missing) == 0  # 设置了 key + password

    def test_threshold_range(self):
        """vector_similarity_threshold 超出范围应报错."""
        with pytest.raises(ValidationError):
            Settings(vector_similarity_threshold=1.5)


class TestConstants:
    def test_source_constants(self):
        assert SOURCE_MANUAL == "手动录入"
        assert SOURCE_DATAMUSE == "datamuse_api_v1_1"

    def test_path_constants(self):
        assert PACKAGE_DIR.name == "word_study"
        assert PROJECT_ROOT.name == "word_study"
        assert BASE_DIR == PACKAGE_DIR
