"""
WordStudy 英语单词学习系统 —— FastAPI 入口。

启动方式:
    wordstudy                       # pip install -e . 后直接运行
    uvicorn main:app --reload       # 开发模式
"""
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中（兼容 PyCharm 直接运行）
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from config import validate_config, BASE_DIR
from models.sqlite_client import init_db
from tools.word_tools import init_alphabet_nodes


# ── 启动时初始化 ────────────────────────────────────────

def _startup_init():
    """应用启动初始化：校验配置 → 初始化 SQLite → 初始化 Neo4j Alphabet 节点。"""
    missing = validate_config()
    if missing:
        print(f"[WARN] 缺少以下环境变量: {', '.join(missing)}")
        print("[WARN] 请检查 .env 文件（参考 .env.example）")

    init_db()
    print("[INFO] SQLite 数据库初始化完成")

    try:
        init_alphabet_nodes()
        print("[INFO] Neo4j Alphabet 节点初始化完成（26个字母）")
    except Exception as exc:
        print(f"[WARN] Neo4j Alphabet 初始化失败: {exc}")
        print("[WARN] 请确保 Neo4j 已启动且连接信息正确")


# ── FastAPI 应用 ─────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _startup_init()
    yield


app = FastAPI(
    title="WordStudy",
    description="英语单词学习系统 - Neo4j + LangChain + FastAPI",
    version="1.0.0",
    lifespan=lifespan,
)

# 静态文件
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = BASE_DIR / "templates" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# ── 占位子页面（稍后实现具体功能）──

@app.get("/words", response_class=HTMLResponse)
async def words_page():
    return HTMLResponse(content=_placeholder_page("总单词", "words"))

@app.get("/meanings", response_class=HTMLResponse)
async def meanings_page():
    return HTMLResponse(content=_placeholder_page("总释义", "meanings"))

@app.get("/synonyms-antonyms", response_class=HTMLResponse)
async def synonyms_page():
    return HTMLResponse(content=_placeholder_page("近义词 / 反义词", "synonyms-antonyms"))

def _placeholder_page(title: str, page_id: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="zh-CN" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} — WordStudy</title>
    <style>
        *,*::before,*::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
        :root {{ --bg-body: #F1F5F9; --text-primary: #0F172A; --text-secondary: #475569; --color-primary: #4F46E5; --color-accent: #F59E0B; --radius-lg: 20px; }}
        [data-theme="dark"] {{ --bg-body: #0B1121; --text-primary: #F1F5F9; --text-secondary: #94A3B8; }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-body); color: var(--text-primary);
            min-height: 100vh; display: flex; align-items: center; justify-content: center;
            transition: background 0.4s, color 0.4s;
        }}
        .placeholder-card {{
            text-align: center; padding: 48px 40px;
            background: rgba(255,255,255,0.72); backdrop-filter: blur(16px);
            border: 1px solid rgba(255,255,255,0.5); border-radius: var(--radius-lg);
            box-shadow: 0 8px 32px rgba(0,0,0,0.06); max-width: 480px; width: 90%;
        }}
        [data-theme="dark"] .placeholder-card {{ background: rgba(30,41,59,0.68); border-color: rgba(255,255,255,0.06); }}
        .placeholder-card .icon {{ font-size: 3rem; margin-bottom: 16px; }}
        .placeholder-card h1 {{
            font-size: 1.6rem; font-weight: 800;
            background: linear-gradient(135deg, var(--color-primary), var(--color-accent));
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; margin-bottom: 8px;
        }}
        .placeholder-card p {{ color: var(--text-secondary); font-size: 0.95rem; line-height: 1.7; margin-bottom: 24px; }}
        .back-btn {{
            display: inline-flex; align-items: center; gap: 6px;
            padding: 10px 22px; background: var(--color-primary); color: #fff;
            border: none; border-radius: 12px; font-size: 0.9rem; font-weight: 600;
            cursor: pointer; text-decoration: none; transition: all 0.22s cubic-bezier(0.25, 0.8, 0.25, 1);
        }}
        .back-btn:hover {{ background: #4338CA; box-shadow: 0 4px 16px rgba(79,70,229,0.3); transform: translateY(-2px); }}
    </style>
</head>
<body>
    <div class="placeholder-card">
        <div class="icon">🚧</div>
        <h1>{title}</h1>
        <p>此页面正在建设中，稍后将为你呈现完整功能。<br>请先返回主页使用现有功能。</p>
        <a href="/" class="back-btn">← 返回主页</a>
    </div>
    <script>
        document.documentElement.setAttribute('data-theme', localStorage.getItem('ws-theme') || 'light');
    </script>
</body>
</html>"""


# 路由注册
from routers.word_router import router as word_router

app.include_router(word_router)


# ── CLI 入口（供 pyproject.toml [project.scripts] 调用）──

def main():
    """生产模式入口（wordstudy 命令）。"""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# ── PyCharm / 直接运行入口（开发模式，reload=True 必须用字符串）──

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
