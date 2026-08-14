# 当前进度指针

**阶段**：第二步「做」—— 推理侧修复 + 工程卫生 + 评测修复 **已完成并提交（abb8509）**。

**已完成**：
1. ✅ 修过度拒答（推理侧，不需重训）：软化 RAG 指令、空上下文兜底、关思考模式、剥离 `</think>`
2. ✅ 工程卫生：git init + requirements 锁版 + README/配置对齐
3. ✅ 修评测口径：NHSR 司法解释引用 + 最长匹配、拒答率/有效回答率指标、测试集重标 8 条

**待交用户跑（需 GPU/网络/时间，命令见 README 或下方）**：
- 重建 Dense 索引：`python scripts/04_build_rag.py`（去掉 `--no_dense`）
- RAFT 补「部分相关」中间态数据再重训（打破全有/全无两态）
- SafeDPO：`python scripts/03_build_dpo_pairs.py ...` → `python scripts/06_train_dpo_safedpo.py ...`

**回退规则**：出错回到最近 checkpoint（本目录文件），禁止重跑已完成阶段。
