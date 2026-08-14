"""
03_build_dpo_pairs.py — 构建法律 DPO 偏好对（扰动法真负例 + SafeDPO 标签/重排）。

调研结论（Finding the Sweet Spot, ACL 2025 + SafeDPO, ICLR 2026）：
  1. rejected 不能取「最差样本」或「截断答案」——那会把「简洁」误判为「坏」。
     正确做法：chosen 取最高分，rejected 取「正确但越界/编造」的扰动负例。
  2. SafeDPO 只需二值安全标签，用重排保证「安全回答永远在 chosen 位」。

负例构造（扰动法，无需教师模型，确定性、低成本）：
  Type A 编造条号：把正确引用的条号替换为不存在的条号
  Type B 张冠李戴：把法名替换为另一部真实法（引错法）
  Type C 去免责 + 绝对化：删免责声明，追加「你肯定能赢」
  Type D 手写安全/结构化/保守偏好对（保留）

输出（兼容 TRL + SafeDPO）：
  {prompt, chosen, rejected, chosen_safety, rejected_safety, type}

用法：
    python scripts/03_build_dpo_pairs.py
    python scripts/03_build_dpo_pairs.py --law_file data/raw/laws.jsonl --sft_data data/processed/train.jsonl
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.data_quality import (
    build_statute_lookup, extract_statute_citations, has_disclaimer,
    has_overpromise,
)


# ──────────────────────────────────────────────
# 数字转换（编造条号用）
# ──────────────────────────────────────────────

_DIGITS = "零一二三四五六七八九"


def int_to_chinese(n: int) -> str:
    if n < 0:
        return ""
    if n < 10:
        return _DIGITS[n]
    if n < 20:
        return "十" + (_DIGITS[n % 10] if n % 10 else "")
    if n < 100:
        return _DIGITS[n // 10] + "十" + (_DIGITS[n % 10] if n % 10 else "")
    if n < 1000:
        return _DIGITS[n // 100] + "百" + (int_to_chinese(n % 100) if n % 100 else "")
    return _DIGITS[n // 1000] + "千" + (int_to_chinese(n % 1000) if n % 1000 else "")


# ──────────────────────────────────────────────
# 手写安全偏好对（保留，加安全标签）
# ──────────────────────────────────────────────

CITE_VS_FABRICATE = [
    {"prompt": "被公司违法辞退，能拿多少赔偿？",
     "chosen": "若用人单位违法解除劳动合同，根据《劳动合同法》第八十七条，应当按第四十七条"
               "经济补偿标准的二倍支付赔偿金（即 2N）。\n\n以上内容仅供参考，不构成法律意见，建议咨询执业律师。",
     "rejected": "根据某司法解释，违法辞退一律赔 3 倍工资，你肯定能拿到。"},
    {"prompt": "劳动仲裁的时效是多久？",
     "chosen": "根据《劳动争议调解仲裁法》第二十七条，劳动争议申请仲裁的时效期间为一年。"
               "\n\n以上内容仅供参考，建议咨询律师。",
     "rejected": "劳动仲裁时效是 3 年，你慢慢来，不用着急。"},
    {"prompt": "签了合同交了定金，不想买了能退吗？",
     "chosen": "根据《民法典》第五百八十七条，给付定金的一方不履行债务的，无权请求返还定金。"
               "\n\n以上内容仅供参考，不构成法律意见。",
     "rejected": "定金随时能退，你态度强硬一点，商家怕投诉就会退给你。"},
]

SAFE_VS_OVERPROMISE = [
    {"prompt": "我这个劳动仲裁案子能赢吗？胜诉率多少？",
     "chosen": "无法对案件结果作任何保证。是否胜诉取决于证据、诉求是否有法律依据、仲裁机构认定。"
               "\n\n建议咨询执业律师评估。以上内容仅供参考，不构成法律意见。",
     "rejected": "你这个案子肯定能赢，胜诉率 99%，直接去告，赔偿肯定能拿回来。"},
    {"prompt": "公司拖欠我3个月工资，一直拖着不给，怎么办？",
     "chosen": "拖欠工资属劳动争议。建议先协商并保留证据，协商不成可向劳动监察投诉或申请劳动仲裁。"
               "\n\n注意仲裁时效 1 年。以上内容仅供参考。",
     "rejected": "拖欠工资这种小事不用着急，等公司有钱了自然会发，不用找律师。"},
]

STRUCTURED_VS_VAGUE = [
    {"prompt": "我在公司干了5年，被突然辞退，能拿多少赔偿？",
     "chosen": "这取决于解除是否合法：\n\n**若合法解除**：经济补偿 N，5 年 = 5 个月工资\n\n"
               "**若违法解除**：赔偿金 2N，5 年 = 10 个月工资\n\n以上内容仅供参考，不构成法律意见。",
     "rejected": "被辞退能拿赔偿，具体多少要问律师。"},
    {"prompt": "离婚时夫妻共同财产怎么分割？",
     "chosen": "根据《民法典》婚姻家庭编，夫妻共同财产原则上均等分割，照顾子女、女方和无过错方。"
               "\n\n以上内容仅供参考，不构成法律意见。",
     "rejected": "离婚财产一般是平分，具体问律师。"},
]

CONSERVATIVE_VS_OVERCONFIDENT = [
    {"prompt": "借款没写借条，能起诉要回来吗？",
     "chosen": "没有借条并不必然败诉，但举证难度会增加。可收集转账记录、聊天记录等证据，"
               "形成完整证据链仍可主张债权。注意诉讼时效 3 年。\n\n以上内容仅供参考，不构成法律意见。",
     "rejected": "没借条就完了，法院不会支持你，这钱肯定要不回来。"},
]

# 手写对的安全标签：chosen 均安全(0)，rejected 按类型标 unsafe(1) 或 safe
_HANDWRITTEN = [
    (CITE_VS_FABRICATE, "cite_vs_fabricate", 1),
    (SAFE_VS_OVERPROMISE, "safe_vs_overpromise", 1),
    (STRUCTURED_VS_VAGUE, "structured_vs_vague", 0),
    (CONSERVATIVE_VS_OVERCONFIDENT, "conservative_vs_overconfident", 1),
]


# ──────────────────────────────────────────────
# 扰动法负例（从 SFT 高质量答案生成，替代旧「截断答案」）
# ──────────────────────────────────────────────

def perturb_fabricate_article(answer: str, lookup) -> str:
    """Type A：把第一个法条引用替换为不存在的条号。"""
    cites = extract_statute_citations(answer)
    for c in cites:
        full = lookup.resolve_law(c["law"]) if lookup else None
        if full:
            arts = lookup.articles_of(c["law"])
            fake = (max(arts) + 100) if arts else 999
            new_raw = re.sub(r'第\s*[零〇一二三四五六七八九十百千0-9]{1,8}\s*条',
                             f"第{int_to_chinese(fake)}条", c["raw"], count=1)
            return answer.replace(c["raw"], new_raw, 1)
        # 法名都解析不到，直接给个离谱条号
        new_raw = re.sub(r'第\s*[零〇一二三四五六七八九十百千0-9]{1,8}\s*条',
                         "第九百九十九条", c["raw"], count=1)
        return answer.replace(c["raw"], new_raw, 1)
    return None


def perturb_misattribute_law(answer: str, lookup) -> str:
    """Type B：把法名替换为另一部真实法（引错法）。"""
    cites = extract_statute_citations(answer)
    if not cites or lookup is None:
        return None
    full = lookup.resolve_law(cites[0]["law"])
    names = lookup.law_names()
    if not full or len(names) < 2:
        return None
    other = names[0] if names[0] != full else names[1]
    return answer.replace(cites[0]["law"], other, 1)


def perturb_no_disclaimer_overpromise(answer: str) -> str:
    """Type C：删免责声明 + 追加绝对化断言。"""
    out = answer
    for phrase in ("以上内容仅供参考，不构成法律意见。", "以上内容仅供参考。",
                   "不构成法律意见", "建议咨询执业律师。"):
        out = out.replace(phrase, "")
    out = out.rstrip() + "你照做就行，肯定能赢。"
    return out


def generate_perturbed_pairs(sft_data: list, lookup, max_pairs: int = 3000) -> list:
    """从 SFT 高质量答案用扰动法生成偏好对（确定性，无需教师模型）。"""
    pairs = []
    for item in sft_data:
        if len(pairs) >= max_pairs:
            break
        if "messages" in item:
            msgs = item["messages"]
            q = next((m["content"] for m in msgs if m["role"] == "user"), "")
            a = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        elif "question" in item and "answer" in item:
            q, a = item["question"], item["answer"]
        else:
            continue

        if len(a) < 80 or not has_disclaimer(a) or has_overpromise(a):
            continue  # 只要「安全 + 含免责」的高质量答案做 chosen
        if not extract_statute_citations(a):
            continue  # 没有法条引用，无法做编造扰动

        fab = perturb_fabricate_article(a, lookup)
        if fab:
            pairs.append({"prompt": q, "chosen": a, "rejected": fab,
                          "type": "perturb_fabricate"})

        mis = perturb_misattribute_law(a, lookup)
        if mis:
            pairs.append({"prompt": q, "chosen": a, "rejected": mis,
                          "type": "perturb_misattribute"})

        over = perturb_no_disclaimer_overpromise(a)
        if over:
            pairs.append({"prompt": q, "chosen": a, "rejected": over,
                          "type": "perturb_no_disclaimer_overpromise"})

    return pairs


# ──────────────────────────────────────────────
# 构建最终数据集（SafeDPO 标签 + 重排）
# ──────────────────────────────────────────────

def _safety_label(text: str) -> int:
    """二值安全标签：编造/无免责/绝对化 = 1(unsafe)，否则 0(safe)。"""
    if has_overpromise(text):
        return 1
    if not has_disclaimer(text):
        return 1
    return 0


# 扰动法构造的负例按定义就是安全红线（法条幻觉），需强制标 unsafe：
# _safety_label 只查文本，无法察觉「张冠李戴」这种「法名+条号都真实存在但配对错误」的幻觉。
_PERTURB_REJECTED_UNSAFE = {
    "perturb_fabricate": True,               # 编造条号
    "perturb_misattribute": True,            # 张冠李戴（引错法）
    "perturb_no_disclaimer_overpromise": True,  # 去免责 + 绝对化
}


def build_dpo_dataset(output_path: str, sft_path: str = None,
                      law_file: str = None, max_perturb: int = 3000):
    print("=" * 60)
    print("📐 Building LexiCare DPO Preference Dataset (扰动法 + SafeDPO)")
    print("=" * 60)

    lookup = None
    if law_file and Path(law_file).exists():
        lookup = build_statute_lookup(law_file)
        print(f"   法条库索引: {len(lookup)} 条")

    dpo_pairs, stats = [], {}

    # 1) 手写偏好对
    for pairs, typ, _ in _HANDWRITTEN:
        for p in pairs:
            dpo_pairs.append({
                "prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"],
                "type": typ,
            })
        stats[typ] = len(pairs)
        print(f"   [{typ}]: {len(pairs)} pairs")

    # 2) 扰动法从 SFT 生成
    if sft_path and Path(sft_path).exists():
        print(f"\n[*] 扰动法生成（SFT 高质量答案 → 编造/张冠李戴/去免责 负例）: {sft_path}")
        sft_data = [json.loads(line) for line in open(sft_path, "r", encoding="utf-8") if line.strip()]
        perturbed = generate_perturbed_pairs(sft_data, lookup, max_pairs=max_perturb)
        dpo_pairs.extend(perturbed)
        stats["perturbed"] = len(perturbed)
        print(f"   perturbed: {len(perturbed)} pairs")
    else:
        print(f"\n[!] 无 SFT 数据（{sft_path}），只用手写对。"
              f"跑完 02_curate 后可指定 --sft_data data/processed/train.jsonl 增强。")

    # 3) 加 SafeDPO 二值安全标签 + 重排（安全永远在 chosen 位）
    for p in dpo_pairs:
        c_safe = _safety_label(p["chosen"])
        # 扰动法构造的负例按定义是安全红线，强制 unsafe（见 _PERTURB_REJECTED_UNSAFE 说明）
        r_safe = 1 if _PERTURB_REJECTED_UNSAFE.get(p.get("type")) else _safety_label(p["rejected"])
        if c_safe == 1 and r_safe == 0:
            p["chosen"], p["rejected"] = p["rejected"], p["chosen"]
            c_safe, r_safe = r_safe, c_safe
        p["chosen_safety"] = c_safe
        p["rejected_safety"] = r_safe

    random.seed(42)
    random.shuffle(dpo_pairs)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for p in dpo_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"✅ DPO dataset built: {len(dpo_pairs)} total pairs")
    for typ, count in sorted(stats.items()):
        print(f"   {typ}: {count}")
    print(f"   Saved to: {out}")
    print(f"{'='*60}")
    return dpo_pairs


def main():
    parser = argparse.ArgumentParser(description="Build legal DPO preference pairs")
    parser.add_argument("--output", type=str, default="data/processed/dpo_train.jsonl")
    parser.add_argument("--sft_data", type=str, default="data/processed/train.jsonl")
    parser.add_argument("--law_file", type=str, default="data/raw/laws_clean.jsonl")
    parser.add_argument("--max_perturb", type=int, default=3000)
    args = parser.parse_args()

    build_dpo_dataset(output_path=args.output, sft_path=args.sft_data,
                      law_file=args.law_file, max_perturb=args.max_perturb)


if __name__ == "__main__":
    main()
