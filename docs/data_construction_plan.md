# LexiCare 高质量法律 QA 数据构造方案

> 基于 4 路并行调研（法律 SFT 数据集方法论 / 通用 SFT 最佳实践 / 法律幻觉与引用 / DPO+RAG 数据构造）与本项目代码逐脚本诊断的对照结论。
> 目标：Qwen3-8B · QLoRA · RTX 4060 8GB · 约 4000–5000 条 · SFT + SafeDPO + RAG。

---

## 0. 一句话结论

**4–5K 条对 8B QLoRA 是"偏宽裕"的规模，数量不是瓶颈，胜负手在于：严格去重 → LLM-judge 质量门槛 → 引用 grounding 强约束 → 复杂度/难度分层 → 标签级多样性采样 → 少量合成增强 → 保留 20–30% 通用数据防遗忘。**（LIMA / DEITA / AlpaGasus / InsTag 一致结论：3–6K 精选样本可匹敌 50–300K 全量。）

法律 QA 的第一性问题是**"引用可溯源、不编法条"**，而非"答得漂亮"。纯 SFT 只能把幻觉从 81% 降到 ~15%，仍会编条号；**"只允许引用检索白名单法条"的约束才使幻觉归零**（Transformer Lab 实测，见 §4）。

---

## 1. 现状诊断：逐脚本硬伤对照

| 脚本 | 当前问题 | 调研依据 |
|------|---------|---------|
| `01_download_data.py` | 只下 DISC + 法条；**未确认 DISC Triplet 的 reference 字段是否含条号**（决定能否直接做引用溯源）；无 JEC-QA/CAIL/LexEval 补充 | DISC schema 未公开，落地前必须抽样 |
| `14_distill_guidelines.py` | `verify_fact_in_law` 只做**字符串子串匹配**（法条号是否出现在 chunk 里），不校验"条号↔内容"是否真实对应；漏中文数字；跨 chunk 引用误判 | LawyerLlaMA 实测 ChatGPT 生成 1/4 错误且编法条；需两步法纠错 |
| `02_curate_data.py` | 去重只有精确 hash + 问题前 50 字规范化；**无 MinHash/语义去重**；downsample 是**均匀随机**，没按复杂度/质量/多样性排序 | FineWeb/RefinedWeb/Dolma 三连去重；DEITA 排序采样 |
| `15_quality_filter.py` | 只有长度/中文字符/重复率 + 5 条正则；**无 LLM 质量打分、无引用审计** | AlpaGasus τ=4.5 门槛；引用审计四步 |
| `03_build_dpo_pairs.py` | 手写仅 15 对；`generate_from_sft` 的 rejected 是**"截断答案"**（把"简洁"当"坏"，偏好信号弱甚至有害）；无教师模型生成的真负例 | Finding the Sweet Spot：rejected 取 μ−2σ；扰动法造编造法条负例 |
| `03_prepare_sft.py` | ✅ 正确（assistant-only loss masking） | 业界标准做法 |
| `03b_build_cot_data.py` | CoT 是**模板填充**（机械），非真实推理链 | 用教师模型生成真实推理链 |
| `04_build_rag.py` | 只建了检索库；**缺 RAG grounded 训练数据**（模型没学过"读检索法条→准确引用"） | RAFT 范式 |
| `13_train_grpo.py` | 复杂度词表残留"药物/禁忌/副作用"，注释还是 chronic disease；reward 无"法条号白名单"硬约束 | GRPO 用 rule-based reward |

---

## 2. 数据来源与许可（可商用清单）

| 来源 | 用途 | 许可 | 说明 |
|------|------|------|------|
| **DISC-Law-SFT** (Pair-QA 93K / Triplet-QA 23K) | SFT 基座 | Apache-2.0 | 主数据源，需抽样确认 reference schema |
| **法条原文**（国家法律法规数据库 flk.npc.gov.cn / chinese-law-and-regulations） | 蒸馏种子 + RAG 语料 | 法条本身不受著作权保护 | 蒸馏 answer 直接用法条原文=零幻觉零版权风险 |
| **JEC-QA**（2.6 万法考题） | 客观题/防遗忘 | — | 司法考试问答，保客观准确率 |
| **CAIL2018**（thunlp，19.6 万刑事文书，含 relevant_articles 法条编号） | 法条预测 gold 标注 | 未明确（商用前确认） | 法条编号可直接做引用 label |
| **LexEval**（23 任务 14,150 题，6250 题人工标注） | 评测 + 可回灌数据 | MIT | 少数可商用评测集 |

**避开**（非商用/未校验）：LexiLaw、LaWGPT、HanFei、LawGPT_zh、chinese-legal-sft（CC-BY-NC）。

> ⚠️ 关键未知数：**DISC-Law-SFT Triplet-QA 的 `reference` 字段是否含具体条号**，官方未公开 schema。落地第一步就是抽样 100 条确认——这决定"引用溯源"能否直接在其上改造，还是必须自建字段。

---

## 3. SFT 数据构造（核心）

### 3.1 去重三连（目标剔除 10–30%）

```
① 精确去重：规范化（小写/去空格标点）后按 (question, answer) 哈希
② MinHash：5-gram、~9000 哈希、相似度阈值 ~0.8 判近重复
③ 语义去重：bge 法律 embedding 算余弦，≥0.9 判重，每簇留最优一条
```

依据：FineWeb（MinHash 5-gram 判重）、RefinedWeb（suffix-array 精确子串）、DEITA（embedding τ=0.9 贪心保多样性）、SemDeDup（删 50% 几乎无损）。

### 3.2 硬过滤规则（RefinedWeb/FineWeb 迁移）

- question 30–500 字符、answer 100–3000 字符（超界剔除）
- 行/句/n-gram 重复率过高即弃
- 垃圾特征：URL 残留、乱码、模板占位符、emoji 过多
- **answer 必须同时含《 与 法（引用符号）**（InternLM 的硬过滤：600 万咨询 → 100 万就是靠这一刀）
- 去 PII（姓名/身份证/手机号）

### 3.3 LLM-judge 质量门槛（AlpaGasus τ=4.5）

用强模型（GPT-4/Claude/DeepSeek）对每条 (instruction, answer) 打 0–5 分，**保留 ≥4 分**（预期保留 30–50%）。维度：准确性（法条正确）、相关性、完整性、格式、无幻觉。**只打分，不必逐条人工。**

### 3.4 复杂度/难度分层（InsTag/DEITA/Cherry）

- 用 IFD 分数（Cherry LLM）或 LLM 打复杂度 1–3 星，分**简单/中等/困难**三档，按 **3:4:3** 配比
- 保证 20–30% 是"多步推理/长链条"难题
- ⚠️ **不要用困惑度选数据**（DEITA 证明 ppl 比随机还差）

### 3.5 多样性采样（DEITA 排序采样）

综合分数 `s = 复杂度 × 质量` 排序 → embedding 贪心扩标签覆盖（保证劳动/合同/婚姻/刑事/公司/知产/行政 7 类均衡）。

### 3.6 蒸馏改造：两步法（取代当前朴素字符串校验）

当前 `14_distill` 只做子串匹配，必须改成 **2501.06521 的两步法**：

```
Step 1 纠错替换（消除 LLM 自带编造法条）：
  LLM 起草答案 → 正则/NER 抽取被引法条
  → 检索真实法条库中最相似条文 → 替换回答案

Step 2 条文合成（从真实法条出发，天然 grounded）：
  给定真实法条 → LLM 生成"该法条可回答的问题"
  → 生成以该法条为依据、含精确条号的答案
```

**关键校验：NHSR 三要素**（Non-Hallucinated Statute Rate）——被引法条的**名称、条号、内容三者全对**才算过，任一不匹配判编造、过滤或重采样。

### 3.7 引用 grounding 强约束字段

新数据统一为四元组 `{instruction, reference(法条原文+条号), output(必须引用《XX法》第X条), 溯源标记}`。**answer 引用条号必须能在 reference 或法条库中溯源**，否则丢弃。这直接为 SafeDPO 提供"正确引用 vs 幻觉引用"的对齐信号。

### 3.8 混配比例（取代 README 的 40/10/40/10，落地到代码）

| 成分 | 比例 | 来源 |
|------|------|------|
| 法律 in-domain QA（精选） | ~65% | DISC 精选 + 自蒸馏 grounded |
| RAG grounded 数据（RAFT 式） | ~10% | 见 §5 |
| 通用对话/客观题（防遗忘） | ~15% | JEC-QA + Alpaca-GPT4 中文 |
| CoT/复杂推理 | ~10% | 教师生成真实推理链 |

总计 4000–5000 条。依据 LIMIT（70–80% in-domain + 20–30% 通用）与 Tulu V2 混配思路。

---

## 4. 引用训练数据（抗幻觉核心，Transformer Lab 实证）

**最可复制的基线**（Qwen2.5-7B LoRA + BM25，引用精确匹配 0.48、幻觉 0%）：

| 配置 | 引用精确匹配 | 幻觉率 |
|------|------------|--------|
| 零样本 | 0.00 | 81% |
| 仅 SFT | 0.148 | ~15% |
| 仅 RAG (k=5) | 0.44 | 0% |
| **混合 SFT+RAG (k=10)** | **0.481** | **0%** |

三条硬结论：
1. **检索带来主要精度跃升，微调负责"学会引用格式/行为"**；纯 SFT 仍会编条号。
2. **幻觉归零靠设计**：检索臂只能引用白名单条文。
3. **数据质量/格式 > 数据量**（2645 对 vs 更小集无提升）。

落地：解析法条为"条文单元清单"（白名单）→ 合成 Q→citation 对（1–3K 即可）→ 训练时把检索条文拼进 prompt，约束"只引检索到的法条"。

---

## 5. RAG grounded 训练数据（RAFT 范式）

每条数据 = `问题 + 文档集（1 个 oracle 法条 chunk + 2–3 个干扰法条 chunk）+ 答案（含引用 + 免责声明）`。

- **P=0.8**：80% 含 oracle，20% 只有干扰/空检索 → 训练**"检索不到就拒答"**（否则模型会硬编法条）。
- 引用标记：从上下文抄录用 `##begin_quote## … ##end_quote##` 包裹（防幻觉）。
- 数据量：2000–5000 条 grounded SFT + 10–20% 拒答负例。

格式：

```json
{
  "instruction": "根据以下检索到的法条回答问题，必须引用真实条文（《法律名》第X条），检索不到时明确说明无法回答。",
  "context": "[文档1]《劳动合同法》第八十五条：……\n[文档2](干扰)《公司法》第二十一条：……",
  "question": "员工被拖欠工资三个月，可以主张哪些权利？",
  "answer": "##begin_quote##……##end_quote##\n依据《劳动合同法》第八十五条，可主张……以上仅供参考。",
  "has_oracle": true
}
```

---

## 6. SafeDPO 偏好对（取代截断负例）

### 6.1 构造原则（Finding the Sweet Spot，ACL 2025）

- **rejected 不能取最差样本**（导致 shortcut learning）；rejected 取 reward 分布 **μ−2σ ≈ 倒数第 2–5 名**。
- **chosen 取最高分**；避免小间隔对。

### 6.2 四类差异轴 + 扰动法

| 差异轴 | chosen | rejected |
|--------|--------|----------|
| 正确性 | 引用真实法条号+内容 | 编造/张冠李戴法条号 |
| 完整性 | 完整分析+结论 | 只有结论无推理 |
| 安全 | 含免责声明+引导咨询 | 无免责、绝对化断言 |
| 引用 grounding | 引用检索 chunk 原文 | 脱离检索凭空发挥 |

**扰动法（成本低）**：取 SFT 正确引用的回答作 chosen；用规则/LLM 把法条号替换为**不存在的号**或张冠李戴作 rejected；去掉免责声明 + 改成"你一定能赢"作安全负例。

### 6.3 SafeDPO 省力技巧（ICLR 2026）

无需手写安全偏好对——只需对候选回复打**二值安全标签**（编造法条/无免责/绝对化 = 1，否则 = 0），用安全标签**重排偏好对**（原偏好但越红线的换到拒绝侧）+ 偏移项 Δ∈[2,10]。一条数据 = `(问题, 安全回答, 越界回答, 安全标签对)`。规模 **500–2000 条**高质偏好对即可。

数据格式（兼容 TRL）：

```json
{
  "prompt": [{"role":"user","content":"员工被拖欠工资三个月，可以主张哪些权利？"}],
  "chosen": "依据《劳动合同法》第八十五条……（真实法条）……以上仅供参考。",
  "rejected": "依据《劳动法》第二百条，你一定能拿到三倍赔偿……",
  "chosen_safety": 0,
  "rejected_safety": 1
}
```

---

## 7. GRPO（可选第三阶段）

- 数据 **prompt-only**，额外列 `gold_statutes`（从 CAIL 法条预测或人工标注）+ `statute_lookup`（全库合法法条号集合）。
- reward 组合（**完全 rule-based，规避 reward hacking**）：`1.5 × 法条召回率 − 1.0 × 编造数 + 0.5 × 免责声明 + 0.3 × 格式`。
- 法条号是否在库是硬事实，规则 reward 无钻空子空间。
- **清理 `13_train_grpo.py` 的医疗残留**（"药物/禁忌/副作用"词表、chronic disease 注释）。

---

## 8. 时效性 / 版本化（进阶，非首版必需）

- **语料版本化**：存每个条文全版本史 + `date_debut/date_fin` + `etat`（仿 FiscalQA Pro），检索时先解析 query 的"as-of 日期"再过滤——这是把时效 RAG 从 2.7% 拉到 98.3% 的那一步。
- **时效 QA 三式**：Post-Cutoff / Pre-Amendment / 多条文 Pre-Amendment，事实日期刻意错位，制造"用错版本即答错"的题。
- 时效敏感题**优先用确定性打分**（regex 条文号 + 数值容差），不用 LLM-judge（judge 有 recency bias）。

---

## 9. 落地优先级（按性价比）

1. ✅ **条文白名单 + "只引检索到的法条"约束**（幻觉归零的设计保证，成本最低）
2. ✅ **去重三连 + 硬过滤规则**（改 `02_curate`）
3. ✅ **两步法蒸馏 + NHSR 三要素校验**（改 `14_distill`）
4. ✅ **LLM-judge 质量门槛 + 复杂度分层 + 多样性采样**（改 `02_curate` / `15_quality`）
5. ✅ **扰动法 + SafeDPO 重排构造偏好对**（改 `03_build_dpo`）
6. ✅ **RAFT 式 RAG grounded 数据 + 拒答负例**（新增脚本）
7. ⬜ 版本化语料 + 日期过滤（进阶，视时效需求）

---

## 10. 建议训练顺序

```
SFT（DISC 精选 + 自蒸馏 grounded + RAG grounded + 15% 通用）
  → SafeDPO（500–2000 偏好对，β=0.1, Δ=5–10）
    → GRPO（prompt-only，rule reward，group size 8，可选）
```

三步都用 LoRA，共享同一法条库与安全标签体系。

---

## 11. 验收指标

| 指标 | 定义 | 目标 |
|------|------|------|
| **NHSR**（非幻觉法条率） | 被引法条名称/条号/内容三者全对的比例 | 越高越好，> 90% |
| 引用精确匹配 | 法条+条+款全对（款错半分） | > 0.4（SFT+RAG 混合） |
| 幻觉率 | incorrect 或 misgrounded 的回答占比 | 越接近 0 越好（RAG 白名单下可设计性归零） |
| 免责声明存在率 | chosen 回答含免责声明 | 100% |
| advice-leakage rate | 越界给具体法律意见/承诺胜诉的比例 | < 1% |
| LawBench / LexEval | 标准基准 | 主评测 |

> ⚠️ 免责声明**必要但不充分**——不能靠声明掩盖本不该给的越界建议。拒绝数据要配"下一步动作"（预约律师/升级人工），避免学会干巴巴拒答。

---

## 主要参考来源

- **法律数据集**：DISC-LawLLM (arXiv:2309.11325)、InternLM-Law (arXiv:2406.14887)、ChatLaw (arXiv:2306.16092)、LawyerLlaMA (arXiv:2305.15062)
- **质量/精选**：LIMA (arXiv:2305.11206)、DEITA (arXiv:2312.15685)、AlpaGasus (arXiv:2307.08701)、InsTag (arXiv:2308.07074)、Cherry LLM (arXiv:2308.12032)、LIMIT (Databricks)
- **去重/过滤**：FineWeb、RefinedWeb (arXiv:2306.01116)、Dolma (arXiv:2402.00159)、SemDeDup (arXiv:2303.09540)
- **合成/增强**：Evol-Instruct (arXiv:2304.12244)、Magpie (arXiv:2406.08464)、Instruction Backtranslation (arXiv:2308.06259)
- **引用/抗幻觉**：Transformer Lab citation training、ALCE (arXiv:2305.14627)、抗幻觉 SFT+HIPO (arXiv:2501.06521)、Stanford 法律 RAG 幻觉研究 (arXiv:2405.20362)
- **安全/偏好**：SafeDPO (arXiv:2505.20065)、Finding the Sweet Spot (ACL 2025)、UltraFeedback、HelpSteer
- **RAG 训练**：RAFT (arXiv:2403.10131)
- **时效**：版本化语料基准 (arXiv:2608.09393)、法律时效失败模式 (arXiv:2605.23497)
