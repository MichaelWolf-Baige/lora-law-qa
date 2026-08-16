# 验证证据包

> 本轮「完善」的验证证据。评测因生成慢（~40s/题 = 300 token 自回归、RTX 4060 笔记本硬件天花板）只跑通 11 题代表样本，但信号足够强。全量 103 题需在用户机器上自行跑（`python scripts/08_professional_eval.py`，约 1 小时）。

## 1. 过度拒答修复的前后对照（11 题样本，2026-08-15）

| 指标 | 修复前 | 修复后 | 目标 | 判定 |
|------|--------|--------|------|------|
| 拒答率 | 73% | **18%** | ≤ 40% | ✅ |
| 引用密度(条/答) | 0.45 | **1.09** | ≥ 0.6 | ✅ |
| 幻觉(编造引用) | 1 例(产假 NHSR=0) | **0 例** | ≤ 5% | ✅（样本小，待全量复核） |

- 7 题「拒答→回答」：工作失职赔偿、合同违约、小微企业纳税、双倍工资时效、老板不发工资、家属放弃抢救工伤等。
- 1 题「编造→正确」：女职工产假，从引《劳动保险条例》第16条（已废止）→ 正确引用。
- 0 回退。

## 2. 仍拒答的 2 题（模型层残留拒答，需重训）

- 「有抵押的汽车,买卖的效力问题?」——检索已对，模型仍拒
- 「怎么领失业补偿金」——检索到《失业保险条例》第14条，模型仍拒

根因：RAFT 教的「不确定就拒」+ 基座 Qwen3-4B 安全先验 + 系统提示保守，**非检索问题**。
解法（已上网搜索确认）：*Divide-Then-Align*（ACL 2025）四象限 DPO —— ✓✗/✗✓ 象限构造「正确 > IDK > 错误」偏好对，复用已建的 SafeDPO 基建。

## 3. 检索质量证据

修 RRF 融合 bug + 补口语→法言同义词后，5/6 口语化问题从「检索到不相关条文」变「检索到正确法条」。
例：「老板不发工资」修复前检索到「工资台账/代发工资」，修复后检索到《劳动法》91条、《劳动合同法》30条。

## 4. 冒烟测试证据

「老板不发工资怎么解决」从直接拒答 → 正确引用《劳动法》第91条 + 《劳动合同法》第30条（支付令）。

## 5. 评测耗时根因（已定位）

生成 ~40s/题（300 token 自回归、300 次串行 forward）。检索已上 GPU（`bge` device=cuda，每次 query <1s）。非检索、非融合 bug。

## 6. 本轮「整合三项目」验证证据（2026-08-15）

### 6.1 L1/L2：cross-encoder 精排 + 接口契约

隔离冒烟测试（临时目录，未碰真实 60K 法条索引）：

| 验证项 | 结果 |
|---|---|
| `py_compile` rag_retriever.py + domain_config.py | ✅ 语法通过 |
| `RagSpec` 新字段（reranker_model/enabled/device） | ✅ 生效（v2-m3 / True / cpu） |
| `CrossEncoderReranker` 类 | ✅ 存在 |
| 懒加载失败降级（不存在模型名） | ✅ `_load()` 返回 None，不崩 |
| 本机真实加载 v2-m3 | ✅ 已缓存，加载跑通（非回退） |
| `retrieve()` 基础检索 | ✅ 命中 |
| complex 查询触发 `_rerank` | ✅ 不崩 |
| 接口契约 content/source/title/score/method | ✅ 齐全 |

### 6.2 pretrain/ 收编

- chat-from-scratch 核心（src/scripts/configs/tests + 小样本 + tokenizer）已迁入 `pretrain/`（5.3M）。
- 数据样本为 `{"text": ...}` 格式，与 `train_single.py` 的 `load_texts` 匹配，可直接训练。
- README 已标注「教学用途，不接入生产模型」。

### 6.3 已由 agent 自跑的验证（2026-08-15，用户确认测试可由 agent 跑）

- **cross-encoder 精排 before/after（真实 60K 法条库，8 个口语化查询）**：
  - 「被辞退能拿多少赔偿」：top1 从 46条(经济补偿) → **87条(违法解除二倍赔偿金)** ✅ 正是答案
  - 「劳动仲裁时效多久」：top1 从 30条(受理,无关) → **27条(仲裁时效一年)** ✅ 正是答案
  - 「交了定金不买能退吗」：top1 从 商品房解释4条 → **民法典587条(定金规则)** ✅ 正是答案
  - 其余（不发工资/失业金/竞业/抵押车）保持相关，不退化
- **pretrain 单元测试**：`pytest tests/` **38 passed**（ModelConfig/RMSNorm/SwiGLUFFN/RoPE/Attention/TransformerBlock/Transformer/EdgeCases/Dataset/Dataloader）
- **from-scratch 模型**：`phase4_100m()` 98.6M 参数，前向 logits `(2,32,8192)` 通过

### 6.4 端到端 103 题评测（后台运行中）

- 命令：`python scripts/08_professional_eval.py`（Qwen3-4B 4bit + LoRA r8 + RAG + cross-encoder）
- 已确认首题正常：`[1/103] labo 引用=1 NHSR=1.0 免责=✓`
- 结果落盘 `outputs/eval_results/professional_metrics.json`，完成后对照 A3（拒答率≤40%、引用密度≥0.6、幻觉率≤5%）

### 6.5 仍需用户操作（训练，非测试）

- pretrain 最小规模训练（B1 的「实际跑通」）：`train_single.py -d template_5k.jsonl -e 1 --max_docs 2000`

### 6.6 端到端 103 题评测结果（2026-08-15，全量跑完）

| 指标 | 修复前(103题) | 本轮(103题) | 目标 | 判定 |
|---|---|---|---|---|
| 拒答率 | 84.5% | **34.95%** | ≤40% | ✅ |
| 引用密度(条/答) | 0.20 | **0.83** | ≥0.6 | ✅ |
| NHSR(可溯源引用) | 90.6% | **91.04%** | ≥90% | ✅ |
| 免责声明率 | 95.2% | **99.03%** | ≥98% | ✅ |
| 承诺胜诉率 | — | **0.0** | <1% | ✅ |
| 幻觉率(含编造引用) | 1.9% | **6.8%** | ≤5% | ⚠️ 轻微超标 |

**幻觉率解读（诚实口径）**：修复前 1.9% 是「84.5% 拒答」下的条件数字（几乎不答所以几乎不编）。本轮检索修复后模型从「答 16 题」跃到「答 67 题」，绝对幻觉从 ~2 条升到 7 条——**按作答量算，幻觉/回答比从 12.5% 降到 10.4%，反而略降**。残留的 7 条幻觉是「模型引用纪律」问题（引了检索上下文外的条号/内容），上一轮已诊断、解法是 Divide-Then-Align DPO（重训，归用户 GPU）+ 运行时 citation 校验（纯代码，可 agent 做）。

**幻觉样本（7 条）**：借条有效期(自造"三年")、抚养费(引错条)、猥亵自诉、帮信罪数额(自造阈值)、公司合并(条号错)、商标续展、抵押优先受偿(引错条)。均为「模型凭记忆引，未从检索上下文抄」。

### 6.7 pretrain 训练跑通（agent 自跑，2026-08-15）

**结果**：300 docs smoke 训练完整跑通——loss 9.14→6.23（ppl 9355→507），checkpoint 中间(step20/40/60/80/100)+best+最终 全部正常保存（含损坏校验），VAL PPL 491，0 次 NaN，EXIT=0。

**修复了 2 个原仓库 bug**（否则训练无法跑通）：
1. **tokenizer 路径**：`train_single.py` 硬编码 `tokenizers/phase1_8k_real/tokenizer.json`，实际提交在 `saved_models/tokenizers/phase1_8k_real_tokenizer.json` → 复制到脚本期望路径。
2. **`save_ckpt` 优化器状态 bug**（关键）：`pstate[k] = pstate[k].cpu()` 就地篡改 `opt.state_dict()` 的活引用，把 Adam 的 exp_avg/exp_avg_sq 移到 CPU 且没移回，下一个 `opt.step()` 必崩 "cuda:0 and cpu"。已改 `.cpu().clone()` 建副本。

**性能发现（影响真实训练）**：`train_single.py` 设 `cudnn.deterministic=True` + transformer 显式构建因果 mask → SDPA 退化为物化注意力。bs=8×sl=1024 下显存 7.9GB/8GB 几乎爆 + 极慢（~167 tok/s）；sl=256 快（3K tok/s）。真实训练建议小 bs/sl，或后续优化（去 deterministic / 用 is_causal）。

**已知 gap**：`generate.py` 自动检测 `configs/train/phase1.yaml`（14M 配置），与 `train_single.py` 的 100M 配置对不上，100M checkpoint 无法直接用 generate.py 生成。

### 6.8 看效果 + 补两个 gap（2026-08-16）

**训练效果**（用户正式训练 5000 docs / 2 epoch 产物）：
- `model_best.pt` val **loss**=0.887（step500，字段名是原代码 bug，存的是 loss 不是 ppl）
- `model.pt` 最终 val **PPL**=2.17（step836）
- 判断：PPL 2.17 对 100M 低得可疑，是**过拟合 template_5k 公式化模板数据**（原模型 87K 蒸馏数据才 PPL 5）。生成验证：输出"天文学家/老师"角色扮演+抽象模板句，确认是模板数据格式记忆，非真实能力（好数据 distill_merged.jsonl 在远程服务器）。

**gap ①（generate.py 100M 配置）——已修**：
- 新增 `configs/model/phase4_100m.yaml`（与 train_single.py 一致）
- generate.py 加 `--model_config` 直连参数 + 修 tokenizer 默认路径 + 修 `steps` 键名
- 验证：`generate.py --model_config phase4_100m.yaml` 成功生成

**gap ②（注意力慢）——根因是内存压力，已修**：
- 诊断：单层 forward 仅 17.7ms，但完整 forward 11.9s、fwd+bwd 直接 OOM（需 ~14GB/8GB）。根因是 **fp32 + 无梯度检查点**，激活内存随 bs×sl×层数爆，近 OOM 导致严重变慢。
- 修复 3 处：① transformer 用 `is_causal=True`（不建显式 mask）；② 去掉 `cudnn.deterministic=True`；③ **加梯度检查点**（`use_gradient_checkpointing`，重算激活省显存）。
- 效果：bs=8×sl=1024 从 42s/步 → **1.6s/步（26×）**，显存 7.9GB → **2.74GB**，不再 OOM，4-5K tok/s。
- 完整 smoke 验证通过（10 步 loss 正常下降、checkpoint 正常保存、0 NaN）。


