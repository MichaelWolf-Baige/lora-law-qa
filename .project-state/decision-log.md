# 决策日志

> 每个决策门的记录。revert cost 高 → 已请用户拍板；低 → 自动决定并记理由。

## 2026-08-15 决策门：本轮「完善」方向

用户勾选 4 个方向，按优先级排序（均经用户确认）：

| # | 方向 | revert cost | 决策 |
|---|------|-------------|------|
| 1 | 修过度拒答 | 高（可能牵涉重训） | 先诊断根因，再对症；守住幻觉率不回弹 |
| 2 | 工程卫生 | 低 | 自动执行（git/锁版本/文档对齐） |
| 3 | 修评测口径 | 低 | 自动执行（测试集/引用抽取/指标拆分） |
| 4 | SafeDPO | 高（GPU 训练） | 准备脚本+数据+命令，交用户跑 |

## 关键结论（根因诊断，2026-08-15）

**过度拒答根因不是「拒答数据太多」，是「检索烂」**：

- 103 题专业评测：87 题（84.5%）`n_citations=0`，其中 83 题直接拒答。
- BM25 字级 bigram 对口语化查询返回不相关条文（「老板不发工资」→ 检索到「工资台账/代发工资」，唯独没有《劳动合同法》85 条）。
- Dense（bge 语义）实际关闭：`04_build_rag.py --no_dense`。
- 推理未关思考模式 → `</think>` 泄漏（5 题）。
- 训练数据本身健康：RAFT 拒答负例仅 21.7%（符合设计），train.jsonl 0 条拒答。

详见 anchor.md。

## 2026-08-15 决策门：本轮「整合三项目」方向

用户拍板 3 项（revert cost 均高，逐个确认）：

| # | 决策点 | 候选 | 用户选择 |
|---|--------|------|---------|
| 1 | 整合定位 | 教学全栈框架 / 真·单模型 / 生产法律QA / 教学+生产 | **生产法律 QA** |
| 2 | 技术路线 | 统一管线+模型抽象 / 以LexiCare为主干 / 编排串联 | **以 LexiCare 为主干吸收** |
| 3 | 落点 | 新建目录 / 就地扩展 lora-law-qa / C:\Users\Lenovo | **就地扩展 lora-law-qa** |

关键结论：
- 产品主线是法律 QA；from-scratch 是教学 side-module（不接入生产 Qwen3-4B）。
- DocQA 的 rerank / query-rewrite 移植进 LexiCare 的 `rag_retriever.py`（补 LexiCare 检索短板——上一轮"过度拒答"的根因正是检索质量）。
- 不做「真·单模型全链路」（100M 末端 toy，无生产价值）。

## 2026-08-15 验收：整合三项目

对照 anchor.md 完成定义，8/9 标准通过。分流决策：**交付** + 记录已知问题。

- 唯一超标项：幻觉率 6.8%（目标 ≤5%）。定性为「模型引用纪律」问题，非本次整合缺陷——上一轮已诊断，解法是 Divide-Then-Align DPO（重训，归用户 GPU）+ 运行时 citation 校验（纯代码，可 agent 做）。
- 关键正面结果：拒答率 84.5%→34.95%、引用密度 0.20→0.83、NHSR 91.04%、免责率 99.03%，均为检索修复直接贡献。

## 2026-08-17 决策门：范围收窄（RAG + 微调 = 一个项目）

用户基于简历策略（两分法：应用/算法分开）拍板收窄范围：

| # | 决策点 | 决策 |
|---|---|---|
| 1 | 整合范围 | 只整合 **RAG（DocQA）+ 微调（LexiCare）**，从零训练分离 |
| 2 | 从零训练去向 | chat-from-scratch 移出到 `D:\桌面\chat-from-scratch`（独立项目，保留修的 bug） |
| 3 | RAG 吸收深度 | cross-encoder 精排已够；LLM query-rewrite/multi-query/HyDE 本轮不做（边际价值低） |

关键结论：
- 简历上本项目（LexiCare）= 应用/工程主打项目（微调 + RAG + 安全 + 评测）；chat-from-scratch = 算法/原理项目（独立）。
- 后续做 LangChain 项目补「Agent/工具编排」维度。

## 2026-08-17 决策门：前端重做（技术选型）

用户拍板：

| # | 决策点 | 候选 | 用户选择 |
|---|--------|------|---------|
| 1 | 前端方案 | Chainlit / 打磨 Gradio5 / 自定义 React+Vue / 轻量 HTML+JS | **Chainlit** |
| 2 | 测试文档 | I.pdf 加密 → 换未加密版 / 换文档 / 跳过 | **I.pdf 实际非加密**（读取器误报，实为 29 页旅游合同示范文本） |

关键结论：
- Chainlit = 纯 Python LLM 聊天框架，开箱即用流式/气泡/文件上传，复用 `app/` 模块零重写。风险：原团队 2025-05 停更、社区维护（演示/简历场景可接受）。
- 复用推理链路：`Qwen3-4B` + `law-lora-r8-20260814-1732` + 4-bit + `enable_thinking=False`（同 chat.py / doc_qa.py）。
- 前端是「纯消费者」，不改任何现有 `app/` 模块。

## 2026-08-17 纠偏：前端无法上传文档（bug 修复）

**症状**：Chainlit 输入框没有上传按钮，无法上传 PDF/txt。

**根因**：Chainlit 2.x 的「自发文件上传」功能（`features.spontaneous_file_upload`）默认 `enabled=None`（关闭）。我写的 config.toml 没包含该段，走 dataclass 默认 None，server 端 `validate_file_upload` 直接拒绝 → 输入框不显示上传入口。

**修复**：config.toml 增 `[features.spontaneous_file_upload] enabled=true`，`accept=["application/pdf","text/plain","text/markdown","text/x-markdown"]`、`max_files=1`、`max_size_mb=20`。已验证 `chainlit.config.config.features.spontaneous_file_upload.enabled == True`。

**handler 逻辑无需改**：`message.elements → cl.File.path` 读取路径正确（`upload_file → session.persist_file` 会落本地并返回 path）。

## 2026-08-17 完整 e2e 测试（Playwright 驱动浏览器），发现并修复 3 个 bug

用 Playwright 真实驱动浏览器走「上传 I.pdf → 摄入 → 提问 → 回答」全流程，发现并修复：

| # | bug | 根因 | 修复 |
|---|---|---|---|
| 1 | `No module named 'app.domain_config'; 'app' is not a package` | Chainlit `load_module` 把 `app/` 插到 `sys.path[0]` 执行模块后 `pop(0)`；我的 `sys.path.insert(0, 项目根)` 被它误 pop，`app/` 留在 sys.path[0]，`import app` 解析成 `app/app.py`（同名冲突）而非 `app` 包 | `insert(0)` → `append`，让项目根存活、`app/` 被正确清掉 |
| 2 | `</think>` 思考内容泄漏到回答 | 手写 prompt 未加空 think 块，`generation_config.enable_thinking=False` 对 Qwen3 手写格式不生效 | prompt 末尾补 `<think>\n\n</think>\n\n`（对应 chat_template `enable_thinking=false` 分支） |
| 3 | 文档问答误用法律安全护栏 | `SafetyGuard.check_input` 无条件套在 doc 模式，问「行程集合点」等非法律问题会被误判 off_topic | 护栏移到 legal 分支内，doc 模式不走法律护栏 |

**验证结果（两种模式均端到端通过）**：
- 文档问答：上传 I.pdf → 47 块 → 「行程开始前3天退团」→ 正确答「15% 违约金」+ 引用来源，无 `</think>`
- 法律咨询：「被公司辞退赔偿」→ 正确引用《劳动合同法》46条 + 实施条例25条 + 免责声明 + 来源，无 `</think>`



