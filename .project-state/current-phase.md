# 当前进度指针

**阶段**：第二步「做」—— 通用文档问答模式已加，进入收尾

**本轮目标**：RAG + 微调 = 一个项目（LexiCare），并完善到可上传 GitHub——见 anchor.md

**已完成**：
1. ✅ 范围收窄 + pretrain/ 分离（→ `D:\桌面\chat-from-scratch`）
2. ✅ 通用文档问答模式（吸收 DocQA 实时摄入）：
   - `app/document_ingestion.py`（PDF/txt 解析 + 句子边界分块）
   - `app/document_qa.py`（DocumentQA + 通用领域配置，复用 HybridRetriever）
   - `scripts/doc_qa.py`（CLI）
   - 验证：ingestion 分块 ✅、Dense+BM25 索引 ✅、cross-encoder 加载 ✅、检索相关性 ✅（fitz 已装）
3. ✅ README 文档化（核心能力 + 快速开始 + 目录结构）
4. ✅ requirements.txt 补 sentence-transformers/pymupdf；.gitignore 补训练数据

**待做（完善 review，待用户拍板）**：
- git 操作：`git rm --cached` 训练数据（55MB 已 tracked）+ 提交全部改动
- （可选）Gradio 上传文档 UI（让文档问答可演示）
- （独立任务）幻觉率 6.8% 修复（DPO 重训）

**回退规则**：出错回到最近 checkpoint（本目录文件），禁止重跑已完成阶段。
