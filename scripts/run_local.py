"""开发模式启动脚本 —— 等价于 uvicorn word_study.main:app --reload。

用法:
    python scripts/run_local.py
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("word_study.main:app", host="0.0.0.0", port=8000, reload=True)
