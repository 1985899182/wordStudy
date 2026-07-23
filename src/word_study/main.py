"""
WordStudy -- FastAPI entry point.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from word_study.config import validate_config, BASE_DIR
from word_study.services.sqlite_client import init_db
from word_study.utils.word_tools import init_alphabet_nodes
from word_study.utils.logging_utils import get_logger
from word_study.routers.word_router import router as word_router

_logger = get_logger(__name__)


def _startup_init():
    missing = validate_config()
    if missing:
        _logger.warning("missing env vars: %s", ", ".join(missing))
    init_db()
    _logger.info("SQLite ready")
    try:
        init_alphabet_nodes()
        _logger.info("Neo4j Alphabet nodes ready (26 letters)")
    except Exception as exc:
        _logger.warning("Neo4j Alphabet init failed | error=%s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _startup_init()
    yield


app = FastAPI(
    title="WordStudy",
    description="word study system - Neo4j + LangChain + FastAPI",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = BASE_DIR / "templates" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


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
    title_map = {"words": "words", "meanings": "meanings", "synonyms": "synonyms", "antonyms": "antonyms"}
    title = title_map.get(page, page)
    html = html.replace("__PAGE_TITLE__", title)
    html = html.replace("__PAGE_ID__", page)
    return html


app.include_router(word_router)


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("word_study.main:app", host="0.0.0.0", port=8000, reload=True)
