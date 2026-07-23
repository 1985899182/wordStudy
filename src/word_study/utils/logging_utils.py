"""
结构化日志工具 —— 替代全局散落的 print() 调用。

设计思路:
  - 全项目使用单一的 `get_logger()` 工厂函数获取 Logger
  - Logger 输出格式: "[LEVEL] module.name: message | key=value ..."
  - 支持 DEBUG / INFO / WARNING / ERROR 四级
  - 所有外部调用都应 try/except 包裹，日志本身不抛异常
  - 生产环境可替换为 structlog / loguru 等，接口保持一致

Why not print():
  - print() 无级别概念，调试日志和生产日志混在一起
  - print() 无法重定向到文件、日志聚合系统
  - 结构化日志（key=value）方便 grep / ELK 查询
"""
from __future__ import annotations

import logging
import sys
from functools import lru_cache

from word_study.config import get_settings

# ══════════════════════════════════════════════════════════
# Logger 工厂
# ══════════════════════════════════════════════════════════

@lru_cache(maxsize=64)
def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的 Logger（按名称缓存，避免重复创建）。

    Args:
        name: 通常传入 __name__，使得日志前缀带模块路径。

    Returns:
        已配置 Handler 和 Formatter 的 Logger。
    """
    settings = get_settings()

    # ── 创建 Logger ──────────────────────────────────
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    # 避免重复添加 Handler（logger 可能已被父级配置过）
    logger.propagate = False

    if not logger.handlers:
        # ── 控制台 Handler ──────────────────────────
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)

        # ── 格式化: "[INFO] module.path: message | extra=info" ──
        formatter = logging.Formatter(
            fmt="[%(levelname)-5s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # ── 文件 Handler（写入 cypher_query.log）─────
        log_path = settings.log_file_path
        if log_path:
            try:
                file_handler = logging.FileHandler(log_path, encoding="utf-8")
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except OSError:
                # 文件写入失败不影响主流程
                pass

    return logger


# ══════════════════════════════════════════════════════════
# 便捷函数 —— 用于不需要创建模块级 logger 的简单场景
# ══════════════════════════════════════════════════════════

def log_info(module: str, msg: str, **extra) -> None:
    """快速 INFO 日志。extra 参数会被格式化为 key=value。"""
    extra_str = " | ".join(f"{k}={v}" for k, v in extra.items()) if extra else ""
    full_msg = f"{msg} | {extra_str}" if extra_str else msg
    get_logger(module).info(full_msg)


def log_warning(module: str, msg: str, **extra) -> None:
    """快速 WARNING 日志。"""
    extra_str = " | ".join(f"{k}={v}" for k, v in extra.items()) if extra else ""
    full_msg = f"{msg} | {extra_str}" if extra_str else msg
    get_logger(module).warning(full_msg)


def log_error(module: str, msg: str, **extra) -> None:
    """快速 ERROR 日志。"""
    extra_str = " | ".join(f"{k}={v}" for k, v in extra.items()) if extra else ""
    full_msg = f"{msg} | {extra_str}" if extra_str else msg
    get_logger(module).error(full_msg)


def log_debug(module: str, msg: str, **extra) -> None:
    """快速 DEBUG 日志。"""
    extra_str = " | ".join(f"{k}={v}" for k, v in extra.items()) if extra else ""
    full_msg = f"{msg} | {extra_str}" if extra_str else msg
    get_logger(module).debug(full_msg)
