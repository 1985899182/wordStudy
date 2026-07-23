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


# ── 浏览子页面 ──

@app.get("/words", response_class=HTMLResponse)
async def words_page():
    return _browse_page("words")

@app.get("/meanings", response_class=HTMLResponse)
async def meanings_page():
    return _browse_page("meanings")

@app.get("/synonyms", response_class=HTMLResponse)
async def synonyms_page():
    return _browse_page("synonyms")

@app.get("/antonyms", response_class=HTMLResponse)
async def antonyms_page():
    return _browse_page("antonyms")

def _browse_page(page: str) -> str:
    template = BASE_DIR / "templates" / "browse.html"
    html = template.read_text(encoding="utf-8")
    title_map = {
        "words": "总单词",
        "meanings": "总释义",
        "synonyms": "近义词",
        "antonyms": "反义词",
    }
    title = title_map.get(page, page)
    # 替换页面标题和 data-page 属性
    html = html.replace("__PAGE_TITLE__", title)
    html = html.replace("__PAGE_ID__", page)
    return html


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
