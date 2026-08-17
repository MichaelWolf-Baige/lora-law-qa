# 锚点文档 —— LexiCare（RAG + 微调 整合项目）

> 范围收窄（2026-08-17）：本项目 = **RAG + 微调**（应用/工程方向，简历主打）。从零训练（chat-from-scratch）分离为独立项目，已移至 `D:\桌面\chat-from-scratch`，不再属于本项目。
>
> 本轮（2026-08-17）：**前端重做**。现状是 4 个割裂的 Python 界面（3×Gradio + FastAPI + Plotly），无统一入口、无聊天体验。决策（用户已拍板）：用 **Chainlit** 统一为一个现代聊天应用（法律咨询 + 文档问答一体），复用现有 `app/` 模块与 FastAPI，不重写后台。测试文档 = `D:\桌面\I.pdf`（29 页《团队境内旅游合同》示范文本，非加密）。

## 1. 目标

把 DocQA 的 RAG 能力整合进 LexiCare 的微调管线，成为一个完整的「法律 QA：微调 + RAG」项目。

## 2. 完成定义（原子标准，二元 pass/fail）

| # | 标准 | 现状 |
|---|---|---|
| R1 | cross-encoder 精排可用（RagSpec 开关 + 懒加载 + 优雅回退） | ✅ 已实现并验证 |
| R2 | 检索对比：口语化查询 top1 命中正确条文 | ✅ 已验证（被辞退→87条、仲裁时效→27条、定金→587条） |
| M1 | LoRA 微调管线可复现（04_train_lora.py + 数据 + 已训练 adapter） | ✅ 已存在 |
| E1 | 103 题评测：拒答率≤40%、引用密度≥0.6、幻觉率≤5% | ✅ 34.95% / ✅ 0.83 / ⚠️ 6.8%（需 DPO） |
| P1 | 项目干净：无 pretrain/ 残留，README 准确反映 RAG+微调 | ✅ 本轮完成 |
| F1 | 统一 Chainlit 入口可启动：`chainlit run app/chainlit_app.py`，法律咨询 + 文档问答两模式可用 | ✅ 已验证 |
| F2 | 法律咨询：口语化查询 → 检索正确法条 → 回答带引用（复用 chat.py 链路） | ✅ 检索/展示（模型拒答属已知过度拒答） |
| F3 | 文档问答：上传 I.pdf → 分块索引 → 针对合同提问（如违约金）→ 回答引用合同原文 | ✅ 已验证（47 块，命中违约金/争议条款） |
| F4 | 流式输出：逐 token 显示，非一次性全吐 | ✅ 已验证（49 chunk） |
| F5 | 回答附「检索到的来源」展示（法条名 / 合同片段） | ✅ 已验证（cl.Text 侧栏） |
| F6 | 输入安全护栏生效（违规输入被拦截） | ✅ 已验证（off_topic 拦截） |

## 3. 技术路线（已确认：以 LexiCare 为主干吸收 DocQA RAG）

- DocQA 的 **cross-encoder 精排** → 已吸收进 `app/rag_retriever.py`（核心吸收，唯一高价值项）。
- DocQA 其余特征（LLM query-rewrite / multi-query / HyDE / chunk-expansion）→ 本轮不做（法条场景边际价值低，加延迟+复杂度）。

## 4. 范围

做：cross-encoder 精排（已完成）、pretrain/ 分离（已完成）、README 打磨（本轮）、验收
不做：从零训练（已分离）、LLM query-rewrite 等高级 RAG、幻觉修复的 DPO 重训（独立任务）

## 5. 先例

- LexiCare 自身（微调 + RAG 成熟）
- DocQA（RAG 能力来源，已吸收 cross-encoder）
- 业界共识：微调改行为、RAG 给事实

## 6. 风险/未知

- 数据：法条库已就绪（60K chunk）
- 评测：103 题已跑，幻觉 6.8% 需 DPO（记已知问题）
- 算力：RTX 4060 8GB（已适配 QLoRA 4-bit）

## 7. 里程碑

分离（已完成）→ README 打磨（本轮）→ 验收
