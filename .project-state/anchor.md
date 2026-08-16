# 锚点文档 —— LexiCare（RAG + 微调 整合项目）

> 范围收窄（2026-08-17）：本项目 = **RAG + 微调**（应用/工程方向，简历主打）。从零训练（chat-from-scratch）分离为独立项目，已移至 `D:\桌面\chat-from-scratch`，不再属于本项目。

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
