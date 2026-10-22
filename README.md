# 企业知识库问答系统（Agentic RAG + LangGraph）

基于 **LangGraph 多节点编排**、**混合检索（稠密 + BM25 + RRF）**、**Cross-Encoder 重排** 的企业级知识库问答系统。
支持多格式文档解析、父子块索引、多租户 + 文档级权限隔离、引用溯源与效果量化评测。

> 一句话定位：让员工用自然语言查询公司内部文档，答案**带可验证的引用**，且**不同权限的人看到的内容不同** —— 查不到就说查不到，绝不编造。

---

## 一、快速开始

```bash
# 0. 前置：本地向量模型（Ollama）—— 零 API 成本，推荐
#    下载安装 https://ollama.com 后拉取模型：
ollama pull bge-m3

# 1. 准备配置
copy .env.example .env
#   唯一必填：LLM_API_KEY（DeepSeek，用于生成答案；也可换 Ollama 本地 LLM）

# 2. 安装依赖
uv sync
#    本地重排（推荐，有 GPU 才有好速度）：uv sync --extra local-rerank
#    首次会从 hf-mirror.com 下载 bge-reranker-base（约 1.1GB）

# 3. 灌入演示数据（3 份文档 / 16 个向量 / 3 个不同权限的账号）
.venv\Scripts\python.exe scripts\seed.py

# 4. 启动服务
.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000

# 5. 打开交互文档
#    http://127.0.0.1:8000/docs
```

### 演示账号（租户 = `acme`）

| 账号 | 密码 | 角色 | 部门 | 密级 | 可见文档 |
|---|---|---|---|---|---|
| `admin` | `admin123` | admin | it | 4（绝密） | 全部 3 份 |
| `zhangsan` | `pass1234` | member | rd | 2（内部） | 2 份（看不到财务文档） |
| `lisi` | `pass1234` | member | finance | 3（机密） | 2 份（看得到财务文档） |

---

## 二、架构

### 分层设计

```
┌─────────────────────────────────────────────────────────────────┐
│ 接入层   FastAPI + JWT + SSE流式 + trace_id中间件 + 全局异常处理      │
│          路由: /auth  /documents  /chat  /admin  /health /ready   │
├─────────────────────────────────────────────────────────────────┤
│ 编排层   LangGraph StateGraph（7 节点 + 条件路由 + 重试回环）          │
│   entry → query_understanding → retrieval_planner                 │
│         → hybrid_retriever → generator → critic                   │
│         → 条件分支: END / 回 planner 重检索 / 拒答兜底               │
├─────────────────────────────────────────────────────────────────┤
│ 检索层   稠密(Chroma) ‖ 稀疏(BM25) → RRF融合 → 权限过滤              │
│         → Cross-Encoder重排 → 阈值过滤 → MMR去冗余 → 父块回填        │
├─────────────────────────────────────────────────────────────────┤
│ 知识层   解析器工厂(PDF/Word/Excel/CSV/TXT/MD) → 统一DocumentBlock   │
│         → 章节感知切分 → 父块+子块 → Embedding → 双写               │
├─────────────────────────────────────────────────────────────────┤
│ 存储层   SQLite(元数据/权限/审计) + ChromaDB(向量) + BM25(关键词索引)  │
│         + 对象存储(原始文件)                                        │
└─────────────────────────────────────────────────────────────────┘
```

### 关键设计决策（每条都有代价分析）

| 决策 | 为什么这么做 | 代价 / 何时该改 |
|---|---|---|
| **父子块索引** | 小块召回准、大块上下文全，二者矛盾用两层结构化解 | 多存一份文本（可忽略）；父块过大时要按 token 预算截断 |
| **混合检索 + RRF** | 向量擅长语义、BM25 擅长编号与专有名词，互补；RRF 只用名次、免疫量纲差异 | 多维护一份关键词索引；语料过万后应换 Milvus 稀疏向量 |
| **Cross-Encoder 重排** | 双塔粗排精度有天花板，交叉编码器逐词交互精度更高 | 实测 +865ms 延迟；所以只对 Top-30 精排，不全库重排 |
| **权限条件下推** | 在向量库阶段就过滤掉无权限数据，避免"先取回再过滤"导致候选塌缩与越权风险 | 权限字段冗余在每个 chunk 上；改权限需同步向量库元数据 |
| **确定性 chunk_uid** | 幂等重建、评测集可跨版本复用 | 与"软删除保留历史块"冲突，重建时需物理删除旧块 |
| **真实向量模型 + API 重排** | 免去本地模型下载与 transformers 版本坑，开箱可用 | 依赖网络；离线场景可切 `EMBEDDING_PROVIDER=local` + `RERANK_PROVIDER=local` |

### 权限模型（四元组）

```
tenant_id       租户硬隔离：A 公司永远查不到 B 公司的数据
department      部门：只能看本部门 + 全公司公开文档
security_level  密级：1公开 2内部 3机密 4绝密，规则是"用户级别 ≥ 文档级别"
allowed_roles   角色白名单：额外授权给指定角色
```

**双重防护**：条件先下推到向量库（Chroma `where` / SQL `WHERE`），再在应用层做一次兜底校验。
即使向量库过滤条件写错，兜底也能拦住越权数据 —— 不信任单点。

---

## 三、实测效果（消融实验）

所有数字都是本地实测，可复现：
`.venv\Scripts\python.exe scripts\eval.py --dataset data\eval_from_corpus.json`

> **技术栈已全本地化**：Embedding 用 Ollama `bge-m3`，重排用本地 `bge-reranker-base`（RTX 4060，69ms/30 条），
> 向量库 Chroma、元数据 SQLite。**整条 RAG 链路零云依赖、可完全离线运行**，
> 只有答案生成调用 DeepSeek（可换 Ollama 本地 LLM）。

### 实验 A：大语料（309 篇文档 / 2604 个向量）+ 40 道可验证题

题目由脚本从语料自动生成：取文档里**全局唯一**的特征串（编号格式、金额、比例），
gold 块由该特征串唯一确定 —— 这是可验证、跨策略可比的 ground truth。
`scripts/gen_eval_from_corpus.py`

本组题目的特点是**极端难**：合成语料里有 300 多篇结构几乎相同的文档，
每篇都有"编号格式"条款，只差前缀（`FW-JX-` / `FL-` / `SR-` / `CSD-` / `RD-APPEAL-`…）。
这类"纯编码差异"正是稠密向量检索的盲区。

| 配置 | Recall@5 | MRR | NDCG@5 | HitRate@5 | 平均延迟 |
|---|---|---|---|---|---|
| ① 仅向量检索 | 0.5500 | 0.3254 | 0.3811 | 0.5500 | 151 ms |
| ② + BM25 + RRF 融合 | 0.5500 | **0.4279** | **0.4585** | 0.5500 | 276 ms |
| ③ 完整链路（+ 本地 Cross-Encoder 重排） | **0.6250** | **0.5667** | **0.5815** | **0.6250** | 913 ms |

**结论（分两段看，这是本实验最有价值的地方）：**

1. **BM25 在这里体现了真实价值**：Recall 不变（0.5500），但 **MRR 从 0.3254 提升到 0.4279（+31%）、
   NDCG 从 0.3811 到 0.4585（+20%）**。
   原因：向量检索分不清 `RD-APPEAL-年份-四位流水号` 和 `FL-年份-四位流水号` 的差异
   （语义上都是"某单据编号格式"），而 BM25 靠精确字面匹配把**前缀吻合**的那一条顶上来。
   这正是"编号/型号/错误码型查询"场景下稀疏检索不可替代的地方。

2. **重排继续加分**：在融合结果上，Recall 0.5500 → **0.6250（+7.5pt）**、NDCG 0.4585 → **0.5815**。
   代价是延迟 276ms → 913ms（GPU 上单次重排 30 条候选仅 69ms，其余是 Ollama 向量化）。

### 实验 B：小语料（16 个向量）+ 75 条手写评测题（早期基线）

| 配置 | Recall@5 | NDCG@5 | HitRate@5 | 延迟 |
|---|---|---|---|---|
| 仅向量 | 0.9867 | 0.9719 | 0.9867 | 178 ms |
| +BM25+RRF | 0.9067 | 0.9017 | 0.9067 | 179 ms |
| **完整链路** | **1.0000** | **0.9902** | **1.0000** | 683 ms |

> 小语料上"天花板效应"明显（16 个向量、DENSE_TOP_K=20，稠密一次就取回全部），
> 因此这个实验只适合验证"链路正确"，不适合下结论。**结论以大语料的实验 A 为准。**

### 实验 C：BM25 的增量价值验证（发生过一次"反转"的实证）

"混合检索"是 RAG 项目的标配说法，所以我专门做了三组对照实验来验证它到底贡献了多少
（脚本：`scripts/ablation_bm25.py`）：

| 实验 | 做法 | 结果 |
|---|---|---|
| C1 增量召回统计 | 比较"仅稠密候选池"与"稠密∪稀疏候选池"是否含 gold | 换 embedding 前：**净贡献仅 1/34**；换成 bge-m3 后：靠 MRR 体现价值 |
| C2 关掉 BM25 | 稀疏权重设 0，其余链路不变 | 旧模型下指标不变；bge-m3 下 MRR 明显下降（见实验 A） |
| C3 候选池截断 | 融合后截断到单路 top_k(=20) vs 保留并集(~40) | 截断 → 掉分；保留并集 → 正常 |

**这组实验发生过一次"反转"，值得完整讲：**

- **第一轮（embedding = 阿里云 text-embedding-v4）**：BM25 净贡献只有 1/34 道，
  关掉它指标几乎不变。当时的结论是"在这份语料上 BM25 没体现增量"。
- **第二轮（embedding 换成 Ollama bge-m3）**：同样 40 道题、同样语料，
  BM25 把 **MRR 从 0.3254 提到 0.4279、NDCG 从 0.3811 提到 0.4585**，价值明显。
- **为什么结论会变**：两个 embedding 模型对"纯编码差异"（`RD-APPEAL-` vs `FL-`）的
  区分能力不同。text-embedding-v4 在这类 token 上表现更好，
  所以稀疏检索没有可补的空间；bge-m3 更容易把它们混为一谈，BM25 的精确匹配就有了用武之地。

**这条经验比任何单一结论都重要**：混合检索的收益**不是常数**，
它取决于「语料特征 + embedding 模型 + 查询类型」三者的组合。
说"混合检索能提升 X%"是不严谨的；正确说法是"在这套语料和模型下，它把 MRR 提升了 Y%"。

### 补充：相关性阈值的数据标定

阈值不是拍脑袋定的，是用评测集两组分布（期望作答 / 期望拒答）标定出来的。
**同一套链路只换了 embedding 和重排模型，阈值就必须重新标定** —— 因为分数尺度变了：

| 阶段 | 模型 | 无答案题最高分 | 有答案题最低分 | 采用阈值 | 分开准确率 |
|---|---|---|---|---|---|
| 稠密检索 | text-embedding-v4 | 0.4505 | 0.3948 | 0.34 | 98.7% |
| 稠密检索 | bge-m3 (Ollama) | — | — | 0.34 | 沿用量级 |
| RRF 融合 | 任意 | 0.0164 | 0.0161 | ——（完全重叠，不可分） | 90.7% |
| 重排（API） | 阿里云 gte-rerank-v2 | 0.1143 | 0.1592 | **0.115** | **100%** |
| 重排（本地） | bge-reranker-base (sigmoid) | 0.7310 | 0.5113 | **0.0（关闭）** | **66.7%（不可分）** |

> 这张表支撑了两个硬结论：
> 1. **RRF 分数是相对名次，不携带绝对相关性信息** —— 所以判断"库里有没有答案"
>    必须依赖重排给出的绝对分数，这是"为什么一定要接重排"的真正理由。
> 2. **换模型必须重新标定阈值**：本地 bge-reranker 的分数挤在 0.5~0.73（因为候选
>    本身就是"编号格式"这类高度相似的文本），单阈值只能分开 66.7%。
>    所以当前配置把重排阈值设为 0（不做分数过滤，只用于排序）——
>    这是一个**有意识的选择**，而不是忘了配。

### 诚实的局限说明

1. **分数阈值必须分阶段标定**：曾用统一的 0.02 阈值过滤所有阶段，直接把 ② 档打到 0.158
   （RRF 分数天然只有 0.01~0.03）。
2. **重排没生效时不能用重排阈值**：重排关闭/降级时候选上还是 RRF 分数，
   套用重排阈值会把结果全部过滤掉 —— 这类"阈值用错场景"的 bug 很隐蔽，已加判断。
3. **本地重排在"同质语料"上分数不可分**：这是语料问题不是模型问题。
   真实企业语料话题分散，分数分布会拉开；而我的合成语料 300 篇同模板，
   重排模型对它们全体给高分是合理的。要验证阈值效果，需要更真实的语料。
4. **"差多少"这类需要算术的问题**：Critic 的数字一致性校验会正确判为"无依据"并拒答 ——
   这是设计选择（不允许模型算没有出处的数字），不是缺陷。
5. **评测集规模**：大语料 40 题、小语料 75 题，统计置信度仍有限；
   正确做法是扩到 200+ 并人工复核 gold。

---

### 附：从云端切到全本地，实际改了什么（面试常问）

这次改造只动了 **配置 + 一个适配器类**，业务代码零改动：

| 改动 | 文件 | 说明 |
|---|---|---|
| 新增 `OllamaEmbedder` | `app/rag/embedding.py` | 约 120 行，走 Ollama 原生 `/api/embed`，含批量/重试/维度探测 |
| 工厂改为 async | `app/rag/embedding.py` | 需要探测模型真实维度，避免 .env 里维度写错导致写入失败 |
| 重排器加 sigmoid | `app/rag/reranker.py` | bge-reranker 输出的是 logits，必须归一化到 0~1 才能比阈值 |
| 重排器改单例 | `app/rag/reranker.py` | **修复严重 bug**：原来每次检索都重新加载 1GB 模型，延迟 7.8 秒 |
| 启动预热重排 | `app/main.py` | 否则首个用户请求要等 37 秒（冷启动） |
| 强制 HF 离线 | `.env` + `reranker.py` | 否则每次加载都去 huggingface.co 做版本检查，国内不可达、卡十几秒 |
| 配置项 | `.env` / `config.py` | 改 3 行：provider、模型名、base_url |

**换 embedding 模型必须全量重建索引**：

```powershell
.venv\Scripts\python.exe scripts\rebuild_index.py
#  309 篇 / 2604 向量 / 107 秒 / 2.89 篇每秒
```

> 为什么必须全量重建：不同 embedding 模型把文本映射到**完全不同的向量空间**，
> 即使维度相同（都是 1024）也不能混用 —— 旧向量与新 query 算余弦相似度毫无意义。
> 这也是 chunk 表记录 `embedding_model` 字段的原因：能查出哪些块的向量是旧模型算的。

---

## 四、核心链路详解

### 离线入库

```
上传 → 预检(格式/大小) + SHA256 去重 → 落盘对象存储 → 建记录(PENDING)
     → 解析器路由 → 统一 DocumentBlock(带页码/章节/类型)
     → 章节感知切分 → 父块(900字) + 子块(240字, 句边界对齐)
     → 元数据标注(tenant_id / department / security_level / allowed_roles)
     → Embedding 批量向量化(1024维)
     → 先写关系库(事实来源) → 再写向量库 → 状态置 INDEXED
     → 失败则置 FAILED 并记录 error_message(可定位到具体阶段)
```

**状态机**：`PENDING → PARSING → INDEXING → INDEXED / FAILED`，每阶段耗时单独落库。

### 在线问答（Agentic）

```
用户提问(JWT 携带 tenant/role/dept/level)
  → ① 查询理解：多轮指代改写 → 意图分类(事实/对比/多跳/闲聊) → 是否需要检索
       └ 闲聊/超纲 → 受限兜底话术(不走通用大模型自由生成，避免合规风险) → END
  → ② 检索规划：单跳 / 多查询 / 查询分解；重试轮次强制换策略
  → ③ 混合检索(并行扇出)：
       Dense(Chroma, 权限下推) ‖ Sparse(BM25, SQL权限下推)
       → RRF 融合(权重可配) → 权限兜底过滤
  → ④ 重排：Cross-Encoder 精排(Top-30 → Top-5)
  → ⑤ 阈值过滤 → MMR 去冗余 → 父块回填
  → ⑥ 生成：强制基于证据 + 结构化引用锚点([1][2] → chunk_uid + 文档 + 页码)
  → ⑦ Critic 三重校验：
       规则1 引用编号是否真实存在（编造引用 = 幻觉的直接证据）
       规则2 答案中的数字能否在证据里找到（数字错误最致命）
       规则3 LLM 判定整体是否被支持（只在前两条通过后才调用，省成本）
       ├ 通过 → END(答案 + 引用 + 置信度)
       ├ 不足且 retry < 2 → 换策略/改写查询 → 回 ②
       └ 不足且 retry ≥ 2 → 明确拒答 + 转人工 → END
```

**为什么重试不空转**：改写查询只允许使用问题中已有的词，且明确禁止引入统计类词汇（标准差、差额等）。曾用 HyDE（让模型编假想答案再检索），实测它会编出"500元""董事长"这类原文没有的内容，把检索彻底带偏 —— 所以**在企业知识库场景主动放弃了 HyDE**。

---

## 五、接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/v1/auth/login` | 登录（需传 tenant） |
| GET | `/api/v1/auth/me` | 查看当前身份与权限 |
| POST | `/api/v1/documents/upload` | 上传并入库（支持密级/部门/角色元数据） |
| GET | `/api/v1/documents` | 文档列表（按权限过滤） |
| GET | `/api/v1/documents/{id}/chunks` | 查看切分结果（演示父子块效果） |
| DELETE | `/api/v1/documents/{id}` | 删除（软删 + 清理向量） |
| POST | `/api/v1/chat/ask` | 提问（`use_agent` 切换 Agentic / 朴素链路） |
| GET | `/api/v1/chat/sessions/{uid}/messages` | 会话记录（含引用与运行轨迹） |
| GET | `/api/v1/admin/stats` | 统计 |
| GET | `/api/v1/admin/health/components` | 各依赖组件健康检查 |
| POST | `/api/v1/admin/eval/run` | 运行消融实验 |
| GET | `/health` `/ready` | 存活 / 就绪探针 |

---

## 六、项目结构

```
app/
├── main.py                  FastAPI 入口（lifespan 预热、trace 中间件、全局异常）
├── config.py                配置中心（所有模型可热切换）
├── models.py                领域模型（9 张表，契约层）
├── schemas.py               API 契约（内部模型不外泄）
├── dependencies.py          依赖注入（JWT → PermissionContext）
├── core/
│   ├── permissions.py       权限上下文 + 过滤条件构造 + 兜底校验
│   ├── security.py          bcrypt + JWT
│   └── logging.py           trace_id 贯穿的日志
├── db/session.py            异步引擎（SQLite → PostgreSQL 只改 URL）
├── rag/
│   ├── parser.py            解析器工厂 → 统一 DocumentBlock
│   ├── chunking.py          章节感知切分 + 父子块 + 语义切分(可选)
│   ├── embedding.py         Embedding 抽象（dashscope / local 热切换）
│   ├── vectorstore/         向量库抽象（chroma / milvus 热切换）
│   ├── sparse.py            BM25 稀疏检索（中文 字+bigram 分词）
│   ├── fusion.py            中文分词 + RRF 融合
│   ├── reranker.py          重排抽象（api / local / none）
│   ├── hybrid.py            混合检索编排（7 步流水线）
│   ├── generator.py         生成 + 引用校验 + 数字校验 + 置信度
│   ├── agent.py             LangGraph 编排（含朴素 RAG 对照组）
│   ├── ingest.py            入库流水线（状态机 + 幂等去重）
│   └── llm.py               LLM 客户端（OpenAI 兼容协议）
├── api/v1/                  auth / documents / chat / admin / health
└── eval/harness.py          评测与消融实验（Recall/MRR/NDCG/HitRate）

scripts/
├── seed.py                  一键灌演示数据
├── eval.py                  消融实验（命令行）
├── reindex.py               重建索引（改切分/换模型后必跑）
└── smoke_test.py            端到端冒烟测试（9 组场景）
```

---

## 七、运维与演进

### 何时升级组件

| 组件 | 当前 | 升级触发条件 | 升级成本 |
|---|---|---|---|
| 关系库 | SQLite | 多 worker 部署 / 写并发上不去 | 改 `DATABASE_URL`（代码零改动） |
| 向量库 | ChromaDB | 数据量 > 百万级 / 需要分布式与原生稀疏向量 | 新增一个适配器（接口已抽象） |
| 检索 | 自建 BM25 | 语料过万后 BM25 每次重建语料变慢 | Milvus 稀疏向量合并为单 collection |
| 入库 | 同步 | 大文件上传超时 | 抽到 Celery/arq worker（流程已按阶段切分） |

### 常用命令

```bash
scripts/seed.py --reset          # 重建演示数据
scripts/reindex.py --dry-run     # 查看哪些文档会被重建
scripts/reindex.py               # 改切分策略/换模型后重建全部索引
scripts/eval.py --limit 6        # 快速验证评测链路
scripts/eval.py --verbose        # 全量评测 + 逐题明细（调优入口）
scripts/smoke_test.py            # 端到端冒烟测试
```

---

## 八、技术栈

| 类别 | 选型 |
|---|---|
| 生成模型 | DeepSeek `deepseek-chat`（OpenAI 兼容协议，可换通义 / 月之暗面 / 本地 Ollama） |
| 向量模型 | **Ollama `bge-m3`（本地，1024 维，零 API 成本）** —— 可一键切阿里云 text-embedding-v4 |
| 重排模型 | **本地 `bge-reranker-base`（Cross-Encoder，GPU 加速 69ms/30条）** —— 可一键切阿里云 gte-rerank-v2 |
| 编排 | LangGraph（StateGraph + 条件边 + reducer 累积状态） |
| Web | FastAPI + SQLModel + SQLAlchemy 2.0（异步） |
| 存储 | SQLite（可升级 PostgreSQL）+ ChromaDB（可升级 Milvus） |
| 检索 | 向量检索 + BM25(rank_bm25) + RRF + Cross-Encoder + MMR |
| 解析 | pypdf / python-docx / openpyxl |

---

## 九、已知限制与后续改进

1. **入库仍为同步**：大文件上传会占住请求，应改为任务队列 + 进度查询。
2. **无语义缓存**：高频重复提问仍会打满检索链路，可加 Redis 语义缓存（注意按租户 + 权限指纹做 key，防止跨租户串味）。
3. **评测集偏小（19 条）**：结论的统计置信度有限，应扩到 150~300 条并人工校验标注。
4. **父块回填按字符截断**：应按 token 预算控制，避免超出模型上下文。
5. **无流式输出**：长答案应加 SSE，改善首字延迟。
6. **Milvus 适配器未实现**：接口已抽象，补齐约 200 行，见 `app/vectorstore/milvus_store.py` 内注释。
