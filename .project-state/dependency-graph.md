# 依赖图 —— 整合三项目（LexiCare 为主干）

> 本轮目标：以 LexiCare 为生产主干，吸收 DocQA 的 RAG 检索能力 + 收编 chat-from-scratch 为教学 side-module。
> 设计原则：低耦合高内聚——"因同样原因变化的东西放一起，因不同原因变化的东西分开"。

## 1. 三项目现状与吸收映射

| 能力 | chat-from-scratch | DocQA | LexiCare（现状） | 吸收动作 |
|---|---|---|---|---|
| 从零预训练 | ✅ 100M Llama 式 Transformer | ❌ | ❌ | 收编为 `pretrain/` 教学 stage |
| BM25 | ❌ | ✅ | ✅ 字级 bigram | 保留 LexiCare（领域已调优） |
| Dense | ❌ | ✅ bge-small-zh | ✅ bge-small-zh-v1.5 | 相同，不动 |
| Hybrid RRF | ❌ | ✅ 复合键(chunk_id) | ⚠️ 内容键(content) | **改用稳定 chunk_id** |
| 重排 | ❌ | ✅ CrossEncoder v2-m3 | ⚠️ 启发式正则 | **吸收 cross-encoder 精排** |
| 查询改写 | ❌ | ✅ LLM QueryRewriter | ✅ 同义词映射 | 保留同义词（便宜）；LLM 改写记「不做」 |
| 多查询/HyDE | ❌ | ✅（可选） | ❌ | 不做（范围外） |

**核心结论（gap 只有一处是刚需）**：LexiCare 检索栈已覆盖 DocQA 的 80%。真正的刚需缺口是
① 精排从「启发式正则」升级为「cross-encoder 语义精排」。②「RRF 对齐改用稳定 chunk_id」是可选优化——法条场景 chunk 内容基本唯一，`content` 键已足够，本轮不做（避免重建向量库的摩擦）。

## 2. 模块划分（内聚）

```
lora-law-qa/
├── app/domain_config.py        # 领域配置（单一事实来源，RagSpec 增 rerank 字段）
├── app/rag_retriever.py        # 混合检索（吸收目标：chunk_id + cross-encoder）
├── app/data_quality.py         # split_articles（04_build_rag 依赖，不变）
├── app/{safety_guard,intent_router,hallucination_detector}.py  # 安全/路由/幻觉（不变）
├── app/{gradio_app,api}.py     # RAG 消费者（只读 retrieve/format_context，不变）
├── scripts/04_build_rag.py     # 索引构建（chunk 加稳定 id）
├── scripts/08_professional_eval.py + chat.py + quick_*.py  # 消费者（不变）
└── pretrain/                   # 新增：from-scratch 教学 stage（完全解耦）
    ├── src/  scripts/  configs/  tests/
    └── README.md               # 标注「教学用途，不接入生产模型」
```

## 3. 依赖关系（谁依赖谁）

```
domain_config(RagSpec)  ←── rag_retriever（读 config）
        ↑                        ↑
   [新增字段]              [chunk_id + cross-encoder]
        
04_build_rag ──(import data_quality.split_articles)──▶ 生产 chunk（含 id）
04_build_rag ──(import rag_retriever.HybridRetriever)──▶ 索引

rag_retriever.retrieve()/format_context()  ←── 被消费于：
   scripts/chat.py, 08_professional_eval.py, quick_diag_base.py, quick_rag_test.py,
   app/api.py, app/gradio_app.py

pretrain/  ──▶ 无任何 import 指向 app/（decoupled，仅共享顶层 requirements 概念）
```

## 4. 变更影响面（改 A 查 B 的依据）

### 变更 1（可选，本轮不做）：chunk 加稳定 `id`
- 说明：法条 chunk 内容基本唯一，`_rrf_fusion` 现按 `content` 对齐已足够；改 chunk_id 需重建向量库（ChromaDB ids + bm25_documents.jsonl），收益不抵摩擦。记录在此，未来多语料（PDF 文档）场景再启用。

### 变更 2：cross-encoder 精排（核心吸收，本轮做）
- **改**：`rag_retriever.py` 新增 `CrossEncoderReranker`（懒加载 `BAAI/bge-reranker-v2-m3`）；`_rerank` 先尝试 cross-encoder，失败（ImportError/OOM/模型缺失）回退现有启发式。
- **配置**：`domain_config.py#RagSpec` 增 `reranker_model` / `reranker_enabled`（默认 False，向后兼容；生产建议 True）。
- **下游影响**：`retrieve()` 调用方（6 处）无需改；仅精排质量变化。依赖 `sentence-transformers`（已在 requirements，无新增依赖）。

### 变更 3：新增 `pretrain/` 教学 stage（解耦）
- **改**：收编 chat-from-scratch 的 `src/ scripts/ configs/ tests/` 到 `pretrain/`，附 README 标注教学用途。
- **下游影响**：无（不 import app/，不接入 Qwen3-4B 生产链）。

## 5. 接口契约（不能破坏）

```python
# HybridRetriever.retrieve(query, top_k=5, force_rag=False) → list[dict]
#   dict 必有键：content, source, title, score, method  （新增 id 是加键，不删键）
# HybridRetriever.format_context(docs, max_chars=2000) → str
# get_retriever(persist_dir="./data/vector_db") → 全局单例
# RagSpec 是 frozen dataclass，新增字段必须有默认值（向后兼容）
```

## 6. 验证计划（四层）

- **L1 数据验证**：chunk 加 id 后，`bm25_documents.jsonl` 每行有唯一 `id`；rebuild 后 ChromaDB count 与 chunk 数一致。
- **L2 局部测试**：`rag_retriever.py` 冒烟（无 GPU、无 rerank 模型 → 回退启发式，不崩）；`retrieve()` 返回含 id。
- **L3 全量回归**：`04_build_rag.py --rebuild` + 5 条测试 query 命中。
- **L4 效果验收**：对照 anchor.md 完成定义（A1 rerank 可用 / A2 口语改写 / A3 103 题三项指标不退化 / B1 pretrain 可跑 / B2 解耦 / C1 三阶段入口 / C2 无重复）。

---

## 7. 前端重做（Chainlit，本轮 2026-08-17）

### 定位
用 Chainlit 统一现有 3 个割裂的 Gradio 界面（gradio_app / app / doc_qa_gradio）+ FastAPI，做成一个现代聊天应用。**前端是纯消费者，不改任何现有 `app/` 模块**——复用它们的既有接口即可。

### 新增模块（单文件，自包含）
```
app/chainlit_app.py    # 统一 Chainlit 前端：法律咨询（默认）+ 文档问答（上传文件触发）
                       # 内置模型加载（同 chat.py 的 4-bit + enable_thinking=False，Qwen3-4B + law-lora-r8）
```

### 依赖关系（谁依赖谁）
```
app/chainlit_app.py ──import──▶ app/domain_config      (get_domain: default_system_prompt)
                    ──import──▶ app/rag_retriever      (get_retriever: 法条 RAG)
                    ──import──▶ app/document_qa        (DocumentQA: 文档 RAG)
                    ──import──▶ app/safety_guard       (SafetyGuard: 输入护栏)
                    ──import──▶ transformers + peft    (模型加载，同 chat.py 链路)
                    ──新增依赖──▶ chainlit             (UI 框架)
```

### 变更影响面（改 A 查 B）
- **改**：仅新增 `app/chainlit_app.py` + `requirements.txt` 增 `chainlit` + 可选 `.chainlit/config.toml`（主题）。
- **不动**：`domain_config / rag_retriever / document_qa / safety_guard` 及 3 个旧 Gradio 文件、`api.py` 全部零改动。
- **风险点**：chainlit 与 gradio 4.44 共享 starlette/fastapi 依赖，需验证无版本冲突（`starlette<1.0` 约束下 chainlit 2.x 应兼容）。

### 接口契约（复用，不破坏）
```python
get_domain().default_system_prompt                # str
get_retriever().retrieve(q, top_k=3) -> list[dict]  # 法条：content/source/title/score/method
get_retriever().format_context(docs, max_chars)   # str
DocumentQA(doc_path).retrieve(q, top_k=3)         # 文档：同 schema
SafetyGuard().check_input(q) -> result(.safe/.category/.fallback_response)
```

### 验证计划
- **L1 数据/加载**：`import app.chainlit_app` 无异常（不加载模型）；`chainlit` 可 import。
- **L2 局部**：demo 模式（无模型）启动 UI → 法律咨询返回占位、文档上传触发 DocumentQA 分块（I.pdf）。
- **L3 全量**：加载模型后，法律咨询口语化查询命中法条、I.pdf 文档问答命中合同条款（违约金/争议解决）。
- **L4 验收**：对照 anchor.md 完成定义 F1–F6。
