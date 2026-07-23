# WordStudy 项目架构

英语单词知识图谱学习系统：从英文文本中提取单词，人工录入中文释义，
借助 LLM + 向量检索做释义去重/归并，把词汇知识沉淀到 Neo4j 图数据库中，
并支持自然语言查询、近义/反义词簇浏览、每日背诵。

## 技术栈

| 层 | 技术 |
|----|------|
| Web 框架 | FastAPI + Uvicorn |
| 图数据库 | Neo4j 5.x（langchain-neo4j 的 Neo4jGraph / Neo4jVector） |
| 本地存储 | SQLite（操作日志、API 缓存、每日单词） |
| LLM | DeepSeek（langchain-deepseek / init_chat_model） |
| 向量嵌入 | 阿里 DashScope text-embedding-v2（1536 维，cosine） |
| 外部词典 | Datamuse API（近义/反义词、英文释义） |
| 前端 | 原生 HTML/CSS/JS（无框架，玻璃拟态双主题） |

## 分层结构

```
templates/ static/            前端页面（index.html 主界面, browse.html 浏览页）
        │  fetch /api/*
routers/word_router.py        API 层：所有 HTTP 端点 + _ensure_meaning_node 归并管线
        │
services/                     服务层：llm_service / datamuse_service / embedding_service
                              cache_service / sqlite_client / neo4j_client / log_service
        │
utils/                        Cypher 工具层：word_tools / meaning_tools / relation_tools
                              （全部参数化 Cypher，每函数对应一种图操作）
        │
Neo4j (bolt) + SQLite (file)  数据层
```

依赖方向自上而下，路由不直接写 Cypher，全部走 utils 层的参数化函数。

## Neo4j 图模型

```
(:Alphabet {name: "a".."z"})
    └─[:PREFIX_TO]→ (:Word {name, prefix, word_query_times, resource: []})
                        └─[:TRANSLATION_INTO {mean_query_times}]→ (:Noun|:Verb|:Adjective|:Adverb :Meaning {means: [], plot, plotEmbedding})
(:Word) ─[:SYNONYM | :ANTONYM]─ (:Word)          词级关系（主用）
(Meaning) ─[:SYNONYM | :ANTONYM {query_times}]─ (Meaning)   释义级关系（兼容保留）
```

要点：
- 释义节点同时打 POS 标签（:Noun 等）和 :Meaning 标签；`means` 是中文释义列表（极相近的释义共享一个节点），`plot` 是 LLM 从 Datamuse 选出的英文释义，`plotEmbedding` 是其向量。
- 启动时 `init_alphabet_nodes()` MERGE 26 个字母节点，单词按首字母挂到字母下，便于分组浏览。
- `word_query_times` / `mean_query_times` 记录选中次数，前端勾选/取消勾选会 ±1。

## 核心流程

### 1. 单词提取（POST /api/step1-extract）
文本 → DeepSeek（extract_words.txt prompt，输出 JSON 数组）→ 逐词 `search_word`
查图 → 已存在的词带回各 POS 释义和查询次数。带 L1 内存缓存。

### 2. 释义录入（POST /api/split-meanings）
用户输入（可含 ；，、;,/ 分隔符）→ `split_raw_input` 拆分 → LLM 按语义相似度分组
（极相近的同组，保存时共享节点）→ 每组第一条过 `validate_pos`（词性+语义+归并三合一校验），
不通过的进 warnings、整组丢弃。

### 3. 保存（POST /api/step2-save）—— 全项目最核心的管线
每个单词：MERGE Word 节点 → 按 group_key 分组处理释义，每组第一条走
`_ensure_meaning_node` 四级归并：

1. **同词精确匹配** → `mean_query_times +1`，直接复用。
2. **跨词精确匹配**（别的词已有完全相同的中文释义）→ `attach_word_to_meaning`
   共享节点 + 自动建词级 SYNONYM 关系。
3. **语义归并**：Datamuse 取英文释义 → LLM 只能"选"不能"写"选出最佳 plot →
   DashScope 嵌入 → Neo4j 向量索引搜索（cosine ≥ 0.85）→ 同词的追加进已有
   means 列表；跨词的再经 LLM `judge_meaning_merge` 二次确认才共享节点 + 建 SYNONYM。
4. **都不命中** → CREATE 新释义节点（带 plot + plotEmbedding）。

同组其余释义直接 append 到首条所在节点。最后处理 relations 建词级 SYNONYM/ANTONYM，
并写 SQLite `word_log`。

### 4. AI 自然语言查询（POST /api/ai-query）
`GraphCypherQAChain`（LangChain）：自动读取图 schema → 生成 Cypher →
`validate_cypher` 校验 → 执行 → 把结果回喂 LLM 生成自然语言回答。
`allow_dangerous_requests=True`，见「风险与改进」。

### 5. 每日背诵（/api/daily-words 等）
SQLite `daily_words` 表按日期存 JSON；全新生成 = 全库 `random.sample`，
追加 = 排除已选后再抽样。前端按 Tab 页展示，悬停看释义。

### 6. 浏览页（/words /meanings /synonyms /antonyms）
`main.py` 用占位符替换（`__PAGE_TITLE__` / `__PAGE_ID__`）复用同一个
browse.html 渲染四种页面；近义/反义页后端用 DFS 求词级关系的连通分量组成"词簇"，
前端用 SVG 按 pill 坐标画连线。

## 缓存与单例

| 对象 | 策略 |
|------|------|
| LLM 调用 | LangChain 全局 `InMemoryCache`，相同 prompt 直接命中 |
| 单词提取 | L1 进程内存 dict |
| Datamuse API | L1 内存 + L2 SQLite `datamuse_cache` 永久缓存（UNIQUE(word, api_type)，L2 命中回写 L1） |
| Neo4jGraph / GraphCypherQAChain / DashScopeEmbeddings / Neo4jVector | 模块级懒加载单例 |
| SQLite 连接 | `threading.local()` 每线程一个连接 + WAL 模式 |

## 值得学习的设计

1. **幂等写入**：`MERGE ... ON CREATE SET ... ON MATCH SET ...` 一条语句同时处理
   新建与计数；`FOREACH (CASE WHEN ...)` 实现条件追加。
2. **LLM 输出鲁棒解析**：`_extract_json_object` 依次去 markdown 代码块 → 截取最外层
   `{}` → 修复 Python 风格 True/False/None → 兜底回退，每个 LLM 调用都有 fallback。
3. **LLM 只做选择不做生成**：英文释义必须从 Datamuse 列表中"选"，降低幻觉。
4. **多级去重**：精确匹配 → 向量相似 → LLM 确认，成本与准确率逐级平衡。
5. **日志不阻塞主流程**：`log_cypher` 吞掉所有异常；所有外部调用 try/except 兜底。
6. **前端**：事件委托绑动态元素、搜索防抖 + AbortController 取消竞态、toast 队列、
   CSS 变量双主题、IntersectionObserver 滚动入场。

## 风险与改进（学习中发现）

1. **`/api/delete-word-rel` 存在 Cypher 注入**：`rel_type` 直接 f-string 拼进
   `MATCH ...-[r:{rel_type}]-...`，未做白名单校验（对比 `/create-word-rel` 有校验）。
   应限定 `rel_type in {"SYNONYM", "ANTONYM"}`。
2. **AI 查询可写图**：`read_only_cypher.txt` 只读 prompt 已写好但没有接到
   `GraphCypherQAChain` 上，且开了 `allow_dangerous_requests=True`，
   自然语言可能诱导生成 CREATE/DELETE。应通过 `cypher_prompt` 接入只读 prompt。
3. `ai_query` 里 `log_cypher(req.question, result_text)` 参数错位：question 存进
   cypher 字段、result 文本存进 params 字段，日志语义混乱。
4. `create_meaning_node` 用 CREATE 而非 MERGE，重复调用会产生重复节点（当前由
   四级归并管线兜底，但绕过管线的调用方需注意）。
5. `_extract_json_array` 的兜底用 `list({...})` 集合去重，会打乱单词顺序。
6. 测试覆盖很薄：core 只有 schema 构造测试，集成测试整体 skip。
