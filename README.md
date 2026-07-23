# WordStudy 英语单词学习系统

基于 **Neo4j 知识图谱 + LangChain + DeepSeek + FastAPI** 的英语单词学习系统。
从英文文本中提取单词、录入中文释义，通过 LLM + 向量检索自动完成释义去重与归并，
把词汇知识沉淀为可查询、可浏览的知识图谱。

详细架构说明见 [docs/architecture.md](docs/architecture.md)。

## 功能

- **单词提取**：粘贴英文文本，DeepSeek 自动提取其中的英文单词，已入库的词自动标注
- **释义录入**：支持一次输入多条释义（；，、;,/ 分隔），LLM 自动分组、校验词性与语义
- **智能归并**：同词/跨词精确匹配 → Datamuse 英文释义 + 向量相似度（≥0.85）→ LLM 二次确认，四级管线避免释义重复
- **近义/反义词**：Datamuse 搜索 + 图库过滤，保存时自动建立词级 SYNONYM/ANTONYM 关系
- **AI 查询**：自然语言提问，GraphCypherQAChain 自动生成并执行 Cypher
- **浏览页**：总单词（按字母分组）、总释义、近义词簇、反义词簇（SVG 连线图）
- **每日背诵**：每天从图库随机抽样生成背诵列表，可追加
- **侧边栏搜索**：前缀搜索、查询记录、暗色/亮色双主题、快捷键

## 技术栈

FastAPI · Neo4j 5.x · SQLite · LangChain（langchain-deepseek / langchain-neo4j）·
DashScope text-embedding-v2 · Datamuse API · 原生 HTML/CSS/JS

## 快速开始

前置条件：Python ≥ 3.10，本地 Neo4j 已启动（bolt://localhost:7687）。

```bash
# 1. 安装依赖（推荐 uv，也可用 pip）
pip install -e .

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY、NEO4J_PASSWORD、DASHSCOPE_API_KEY

# 3. 启动（开发模式，热重载）
python scripts/run_local.py
# 或
uvicorn word_study.main:app --reload
```

打开 http://localhost:8000 即可使用。启动时会自动初始化 SQLite 表和
Neo4j 的 26 个 Alphabet 节点。

## 项目结构

```
src/word_study/
├── main.py            FastAPI 入口（lifespan 初始化、页面路由、静态文件）
├── config.py          .env 配置加载，禁止硬编码
├── core/schemas.py    Pydantic 请求/响应模型
├── routers/           API 端点（word_router.py）
├── services/          LLM / Datamuse / 向量嵌入 / 缓存 / 日志 / 数据库客户端
├── utils/             参数化 Cypher 工具（word / meaning / relation）
├── prompts/           LLM prompt 模板
├── templates/         index.html（主界面）、browse.html（浏览页）
└── static/            style.css（玻璃拟态双主题）、背景图
```

## 测试

```bash
pytest            # 单元测试；集成测试需要 Neo4j 环境，默认 skip
```
