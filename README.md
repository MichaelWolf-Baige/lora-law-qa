# LexiCare — 全法律领域本地咨询 LLM

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Model: Qwen3-4B](https://img.shields.io/badge/Model-Qwen3--4B-green.svg)](https://huggingface.co/Qwen/Qwen3-4B)
[![Hardware: RTX4060-8GB](https://img.shields.io/badge/Hardware-RTX4060-8GB-orange.svg)]()

> **从通用大模型到法律专家——本地全法律领域 LLM 微调全链路实践**
>
> 使用 SFT + SafeDPO + RAG 技术栈，将 Qwen3-4B 微调为全法律咨询助手。
> 在单张 RTX 4060（8GB 显存）上完成全部训练、评估和部署。
> 同一套技术流可复现到任何"确定资料"领域（财税/政务/说明书）。

> ⚠️ **当前状态**：已跑通「数据 → 训练 → RAG 推理 → 评测」全链路。
> - **模型**：Qwen3-4B + LoRA(r8) 已训练（`outputs/lora_weights/law-lora-r8-20260814-1732/`）
> - **数据**：法条库（1,479 部干净有效）+ RAG 检索库（60,311 chunk，BM25 + bge 中文 Dense）+ RAFT grounded 训练数据（1,324 条）
> - **评测**（103 题专业指标）：引用密度 0.20、引用可溯源率(NHSR) 90.6%、引用幻觉率 1.9%、免责声明率 95.2%
> - **对话**：`python scripts/chat.py`（RAG + 模型交互问答）
>
> ⚠️ **关键经验**：纯 DISC 问答微调会灾难性遗忘基座模型的「引用」能力，必须用 RAFT grounded 数据（蒸馏 QA + 检索上下文）重训补回。

> 📌 **项目定位（重要）**：本项目的目标是「**法条检索 + 法律知识问答 + 精准引用**」助手（帮用户查法条、答法律知识、准确引用条文），**不是**「律师级办案工具」（给完整案情出法律意见）。后者受限于 8B 模型能力与数据覆盖（缺部门规章/指导案例），见「已知局限性」。

---

## 项目定位

LexiCare 是一个**本地全法律领域的咨询助手**，覆盖劳动、合同、婚姻家事、刑事、公司商事、知识产权、行政等子领域。

**核心思路**：Qwen3-4B 的权重装不下全法律，因此采用「宽口径 SFT（教法律推理/引用/格式）+ 全法 RAG（知识放检索库）」——广度由检索库承载，模型专注「读法条 → 推理 → 准确引用 → 免责声明」。这样"最全知"由**系统（模型 + 检索）**达成。

### 核心能力

| 能力 | 说明 | 示例 |
|------|------|------|
| **劳动争议** | 辞退赔偿（N/N+1/2N）、竞业限制、加班费、社保工伤、仲裁时效 | "被公司辞退能拿多少赔偿" |
| **合同纠纷** | 合同解除/违约/定金、借款借贷、租赁 | "交了定金不买了能退吗" |
| **婚姻家事** | 离婚财产分割、抚养权、继承遗嘱 | "遗嘱没公证还有效吗" |
| **刑事** | 罪名量刑、取保候审（引导委托律师） | "盗窃罪量刑标准是什么" |
| **公司商事** | 股权、公司治理、破产清算 | "股东可以退股吗" |
| **知识产权** | 商标、专利、著作权侵权 | "商标被抢注了怎么办" |
| **行政** | 行政处罚、行政复议、行政诉讼 | "不服行政处罚怎么办" |
| **安全护栏** | 强制引用法条 + 免责声明，拒绝承诺胜诉/冒充律师 | 越界引导咨询执业律师 |
| **通用文档问答** | 上传任意 PDF/txt，实时解析分块 + RAG 问答（吸收 DocQA） | "上传产品手册，问具体参数" |

---

## 技术架构

```
┌──────────────────────────────────────────────────────────┐
│                      用户界面                             │
│              Gradio Demo / FastAPI                        │
└────────────────────────┬─────────────────────────────────┘
                         │
┌────────────────────────┴─────────────────────────────────┐
│                   安全护栏 (Safety Guard)                  │
│         输入过滤 → 意图路由 → 输出审核 → 免责声明           │
└────────────────────────┬─────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
   ┌────────────┐ ┌────────────┐ ┌────────────┐
   │  RAG 检索   │ │ 微调模型    │ │ 幻觉检测    │
   │ (ChromaDB) │ │ (LoRA+DPO) │ │ (规则+检索) │
   │ 法条/司法解释│ │ Qwen3      │ │ 编造法条检测 │
   └────────────┘ └────────────┘ └────────────┘
```

### RAG 检索能力（吸收自 DocQA）

RAG 检索吸收自 DocQA 的 **cross-encoder 语义精排**（`BAAI/bge-reranker-v2-m3`）：混合检索（BM25 + Dense）粗排后，用 cross-encoder 对候选条文做语义精排，显著提升口语化查询的命中率（「被辞退能拿多少赔偿」「劳动仲裁时效多久」等题 top1 命中正确条文）。

在 `app/domain_config.py#RagSpec` 里开关（`reranker_enabled` / `reranker_model` / `reranker_device`）。模型未下载或显存不足时自动回退启发式重排，不崩。

### 低耦合设计（换领域的关键）

所有领域相关内容（意图、提示词、安全规则、幻觉规则、RAG 同义词、预设问题）收敛在**单一文件 `app/domain_config.py`**。代码层是领域无关的"管道"，通过 `DomainConfig` 注入领域规则：

- 换领域（法律 → 财税/政务）= 新增一个 `DomainConfig` 实例，零改动逻辑代码。
- 当前提供 `LEGAL_DOMAIN`，通过 `get_domain()` 访问。

### 核心技术栈

| 技术 | 用途 | 说明 |
|------|------|------|
| **QLoRA (4-bit)** | 8GB 上训练 4B | bitsandbytes NF4 量化 |
| **rsLoRA** | 防止过拟合 | `alpha/sqrt(r)` 缩放 |
| **DoRA** | 提升微调质量（8GB 上 4-bit 会 OOM，本项目禁用） | 权重分解 LoRA |
| **SafeDPO** | 安全对齐（可选） | 拒绝编造法条、强制免责声明 |
| **RAG** | 知识保鲜 | 混合检索（BM25 + Dense），突破知识截止 |
| **LLM-as-Judge** | 自动评估 | DeepSeek 做法条引用准确性评分 |
| **蒸馏** | 数据构造 | DeepSeek 从法条生成高质量 QA |

---

## 快速开始

### 1. 环境

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

**已验证环境**：Python 3.12, torch 2.5.1+cu121, transformers 4.51.0（Qwen3 需 ≥4.51）, trl 0.15.0, peft 0.13.2, datasets 3.1.0, RTX 4060 Laptop 8GB。
> ⚠️ 不能用 unsloth（Windows 上会装成 CPU torch）；Qwen3 需要 transformers ≥4.51；datasets 4.x 会与 torch 2.5.1 组合 segfault。

### 2. 数据准备

**阶段 A：法条库（已完成 ✅，产物在 `data/`）**

```bash
# A1. 全量下载：从官方数据库镜像拿 22,552 部法条（宪法/法律/行政法规/司法解释/地方法规）
python scripts/01c_download_full_corpus.py

# A2. 预处理：清洗格式 + 剔除历史版本/地方法规 → 1,479 部干净有效法条
python scripts/01d_preprocess_laws.py            # --scope all 保留地方法规

# A3. 质量审计：完整性/格式/条文解析/核心法存在性 + 对比官方库定位遗漏
python scripts/audit_data_quality.py

# A4.（可选）官方直连：权威刷新指定法条（官方搜索枚举有验证码，按需单部拉取）
python scripts/01b_download_official_laws.py

# A5. 构建 RAG 检索库：按「条」分块，BM25 索引（60,311 chunk）
python scripts/04_build_rag.py --no_dense
```

**阶段 B：训练数据构造（需 DeepSeek API key）**

```bash
# B1. 两步法蒸馏：法条 → 带引用 QA（条文合成 + NHSR 三要素校验，防编造法条）
set DEEPSEEK_API_KEY=sk-xxx
python scripts/14_distill_guidelines.py --dry_run   # 先自测三闸门
python scripts/14_distill_guidelines.py             # 正式蒸馏

# B2. 清洗 / 三连去重（精确+MinHash+语义）/ 难度分层采样 / 切分
python scripts/02_curate_data.py

# B3. 质量过滤（规则 + NHSR 引用审计 + 可选 LLM-judge）
python scripts/15_quality_filter.py --law_file data/raw/laws_clean.jsonl

# B4. RAFT 式 RAG grounded 数据（含「检索不到→拒答」负例）
python scripts/04b_build_raft_data.py

# B5. DPO 偏好对（扰动法真负例 + SafeDPO 标签/重排）
python scripts/03_build_dpo_pairs.py --law_file data/raw/laws_clean.jsonl --sft_data data/processed/train.jsonl
```

> 全链路冒烟测试（合成数据，无需联网/API）：`python scripts/smoke_test_data_pipeline.py`

### 3. 训练

```bash
# SFT 训练（Qwen3-4B, QLoRA 4-bit, 标准 LoRA r=8, lr 3e-5, 1 epoch）
# 数据 = 蒸馏 QA + RAFT grounded + BELLE 通用（防遗忘）
python scripts/04_train_lora.py --no_wandb

# SafeDPO 对齐（可选，拒绝编造法条、强制免责声明）
python scripts/06_train_dpo_safedpo.py --sft_adapter outputs/lora_weights/lexicare-sft-XXX
```

### 4. 评估

```bash
# 内置自动评估（ROUGE-L + 法条召回率 + 幻觉标记）
python scripts/07_evaluate.py --no_judge

# LLM-Judge 评估（DeepSeek）
python scripts/07_evaluate.py --judge_model deepseek-chat

# 标准基准 LawBench（推荐主评测）
git clone https://github.com/open-compass/LawBench
# 用自己的模型在 /data/*.json 上生成预测，然后：
# cd LawBench/evaluation && python main.py -i ../predictions/zero_shot -o results.csv
```

### 5. 推理与 Demo

```bash
python app/gradio_app.py        # Gradio Demo
python -m uvicorn app.api:app   # FastAPI 服务
```

### 6. 通用文档问答（实时摄入任意文档）

```bash
# 上传任意 PDF/txt，实时解析分块 + RAG 问答（复用混合检索 + cross-encoder 精排）
python scripts/doc_qa.py --doc 你的文件.pdf
python scripts/doc_qa.py --doc 你的文件.txt --base_only   # 不用法律 LoRA，用基座模型
python scripts/doc_qa.py --doc 你的文件.pdf --no_dense    # 只 BM25（更快）
python app/doc_qa_gradio.py                               # Gradio UI（拖拽上传文档）
```

---

## 数据策略

### 数据来源（法条原文不受著作权保护，《著作权法》第五条，可商用）

| 来源 | 类型 | 说明 |
|------|------|------|
| **国家法律法规数据库 flk.npc.gov.cn** | 法条原文 | **主数据源**。官方权威，`01b` 直连拉全文（搜索枚举有验证码） |
| **官方完整镜像**（twang2218/law-datasets） | 法条原文 | 全量 22,552 部（2023-09 快照），`01c` 下载 |
| **DISC-Law-SFT**（可选） | SFT 问答 | Apache-2.0，93K 法律问答（不想自蒸馏时用） |
| **自蒸馏带引用 QA** | SFT 问答 | DeepSeek 从法条生成，NHSR 校验可溯源 |

**数据规模**（`data/`）：

| 产物 | 规模 |
|------|------|
| `laws.jsonl` | 22,552 部全量法条（87.7 万条条文） |
| `laws_clean.jsonl` | 1,479 部干净有效（法律 310 + 行政法规 599 + 司法解释 538 + …） |
| `vector_db/bm25_documents.jsonl` | RAG 检索库 60,311 chunk |

> ⚠️ **数据边界（诚实声明）**：已覆盖「法律 + 行政法规 + 司法解释 + 地方性法规」。
> **未覆盖**：部门规章（各部委规章）、指导性案例、裁判文书——这三块是「律师级办案」依赖的，本仓库当前不包含，属已知局限。

### 推荐混配比例（4B LoRA，约 3000 条）

```
~65% 法律 in-domain QA（DISC 精选 + 自蒸馏 grounded）
~10% RAG grounded（RAFT 式，含「检索不到→拒答」负例）
~15% 通用对话/客观题（JEC-QA + Alpaca-GPT4，防遗忘）
~10% CoT/复杂推理（教师生成真实推理链，非模板填充）
```

> 调研实证（LIMA/DEITA/AlpaGasus/InsTag）：3–6K 精选样本可匹敌 50–300K 全量，
> 瓶颈在「去重 + 质量门槛 + 引用 grounding + 难度分层」而非数量。

### 数据构造方法（两步法蒸馏 + NHSR）

```
法条原文 → 教师模型合成 QA（条文合成）→ NHSR 三要素校验（名称/条号/内容全对）
        → 质量过滤（规则 + 引用审计 + LLM-judge）→ 三连去重 → 难度分层采样
```

**关键校验规则**（防"编造法条"，已实现于 `app/data_quality.py`）：
- **NHSR 三要素**：回答中的每个《法名》第X条 必须能溯源到法条库原文，任一不匹配判编造
- 必须包含"本回答不构成法律意见，建议咨询专业律师"免责声明
- 不能给出确定的胜诉承诺（"肯定能赢""胜诉率X%"）
- DPO 负例用**扰动法**（编造条号/张冠李戴/去免责+绝对化），不用"截断答案"

---

## 训练配置

### SFT 阶段（Qwen3-4B, QLoRA 4-bit）

```yaml
model: Qwen/Qwen3-4B
use_4bit: true            # QLoRA NF4 —— 8B 在 8GB 卡跑不动，实测 4B 约 12 秒/步
use_dora: false           # DoRA 在 4-bit 上初始化会反量化爆显存，禁用
lora_r: 8
lora_alpha: 8             # rsLoRA: alpha=r
use_rslora: true
lora_dropout: 0.05        # 防遗忘正则化

epochs: 1                 # 1 epoch 防过拟合
effective_batch_size: 16  # 1 × 16 grad_accum
learning_rate: 3.0e-5     # 温和 lr，防遗忘
max_seq_length: 2048
gradient_checkpointing: true
fp16: true                # RTX4060 上 bf16 更慢
```

> **思考模式**：Qwen3 混合思考仅用部署开关控制（推理时 `enable_thinking=True/False`），训练数据不强制 `/think` 结构。

### SafeDPO 阶段（可选）

```yaml
base_model: Qwen/Qwen3-4B
sft_adapter: outputs/lora_weights/lexicare-sft-XXX
lora_r: 64
lora_alpha: 64
beta: 0.1
safety_margin: 0.1
learning_rate: 5.0e-7
epochs: 1
```

---

## 目录结构

```
lora-law-qa/
├── app/
│   ├── domain_config.py        # 领域配置层（低耦合的核心）
│   ├── data_quality.py         # 数据质量共享模块（NHSR/去重/引用提取/中文数字）
│   ├── safety_guard.py         # 四层安全护栏
│   ├── intent_router.py        # 意图分类路由
│   ├── rag_retriever.py        # 混合 RAG 检索（BM25 + Dense + cross-encoder 精排）
│   ├── document_ingestion.py   # 通用文档摄入（PDF/txt 解析 + 分块，吸收 DocQA）
│   ├── document_qa.py          # 通用文档问答（实时摄入 + 复用检索）
│   ├── hallucination_detector.py # 幻觉检测
│   ├── gradio_app.py / api.py  # Demo / API
│   └── ...
├── scripts/
│   ├── 00_inspect_disc_schema.py    # 抽样确认 DISC 字段 schema
│   ├── 01_download_data.py          # 下载 DISC-Law-SFT（可选）
│   ├── 01b_download_official_laws.py# 官方 flk.npc.gov.cn 直连（权威刷新）
│   ├── 01c_download_full_corpus.py  # 全量法条下载（22K 部）
│   ├── 01d_preprocess_laws.py       # 预处理（清洗/去重/过滤有效）
│   ├── audit_data_quality.py        # 数据质量审计
│   ├── 02_curate_data.py            # 清洗/三连去重/分层采样
│   ├── 03_build_dpo_pairs.py        # DPO 偏好对（扰动法+SafeDPO）
│   ├── 03b_build_cot_data.py        # CoT 训练数据（可选）
│   ├── 04_build_rag.py              # RAG 检索库（按条分块）
│   ├── 04b_build_raft_data.py       # RAFT 式 grounded 数据
│   ├── 05_train_sft_unsloth.py      # SFT 训练
│   ├── 06_train_dpo_safedpo.py      # SafeDPO 对齐
│   ├── 07_evaluate.py               # 评估
│   ├── 13_train_grpo.py             # GRPO（可选）
│   ├── 14_distill_guidelines.py     # 法条→QA 蒸馏（两步法+NHSR）
│   ├── 15_quality_filter.py         # 质量过滤（NHSR 审计+LLM-judge）
│   └── smoke_test_data_pipeline.py  # 全链路冒烟测试
├── configs/                    # Qwen3-8B QLoRA 配置
├── data/
│   ├── raw/                    # laws.jsonl(22K全量) + laws_clean.jsonl(1479干净有效)
│   ├── vector_db/              # RAG 检索库（BM25 文档）
│   ├── processed/              # 训练数据（蒸馏/清洗后）
│   └── test_cases/             # 评测用例
└── outputs/                    # 训练产物（checkpoints/lora_weights）
```

---

## 已知局限性

1. **不构成法律意见**：模型输出仅为法律知识参考，不能替代执业律师的专业意见
2. **定位边界**：本项目是「法条检索 + 法律知识问答 + 精准引用」助手，**不是律师级办案工具**（不能对完整案情给出确定性法律意见）
3. **数据覆盖缺口**：已覆盖法律/行政法规/司法解释/地方法规；**未覆盖部门规章、指导性案例、裁判文书**（这三块是律师实务依赖的，需另找数据源）
4. **法条时效性**：法条库为 2023-09 快照，此后新增/修订的法规（全国性约 33 部）未包含；官方 API 可按需刷新
5. **地域差异**：仲裁实践各地存在差异（如北京/上海/广东），需本地化适配
6. **模型规模限制**：8B（Q4 量化）是 8GB 卡上的质量上限；复杂法律推理有天花板，更小体量需重度依赖 RAG
7. **⚠️ 免责声明**：本项目仅供技术学习和产品原型展示，**不构成任何法律意见**

---

## 参考资源

### 数据源
- [国家法律法规数据库 flk.npc.gov.cn](https://flk.npc.gov.cn/) — **主数据源**，官方权威法条原文
- [twang2218/law-datasets](https://github.com/twang2218/law-datasets) — 官方完整镜像（22,552 部，2023 快照）
- [ZongziForu/cn-law-hub](https://github.com/ZongziForu/cn-law-hub) — 官方 API 接口文档
- [DISC-Law-SFT](https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT) — 法律 SFT 数据集（Apache-2.0，可选）

### 参考项目
- [DISC-LawLLM](https://github.com/FudanDISC/DISC-LawLLM) — 复旦法律大模型（DISC-Law-SFT 数据集来源）
- [InternLM-Law](https://github.com/InternLM/InternLM-Law) — 最强开源法律模型（SFT-only 超 GPT-4）
- [Awesome-LegalAI-Resources](https://github.com/CSHaitao/Awesome-LegalAI-Resources) — 法律 AI 资源合集

### 评测基准
- [LawBench](https://github.com/open-compass/LawBench) — 主评测（1 万题 / 20 任务，规则打分）
- [LexEval](https://github.com/CSHaitao/LexEval) — 次评测（1.4 万题 / 23 任务，MIT）

### 技术参考
- [Unsloth LoRA Hyperparameters Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)
- [DoRA Paper](https://arxiv.org/abs/2402.09353)
- [rsLoRA](https://arxiv.org/abs/2312.03732)
- [SafeDPO](https://arxiv.org/abs/2505.20065)
- [DPO Paper](https://arxiv.org/abs/2305.18290)

---

## License

MIT License — 详见 [LICENSE](LICENSE) 文件。
