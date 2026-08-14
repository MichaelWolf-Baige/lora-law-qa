"""
15_quality_filter.py — 法律数据质量过滤 + 法条引用审计（NHSR）+ 可选 LLM-judge。

三层过滤（对齐调研结论）：
  L1 规则：承诺胜诉 / 伪造胜诉率 / 冒充律师 / 编造司法解释 / 免责声明缺失 / 非法律
  L2 引用审计（NHSR）：回答中每个《法名》第X条 必须能在法条库溯源，否则判编造
  L3 LLM-judge（可选）：对 (问题,回答) 打 0–5 分，保留 ≥ 阈值

用法：
    python scripts/15_quality_filter.py --input data/processed/train.jsonl
    python scripts/15_quality_filter.py --input data/processed/train.jsonl --law_file data/raw/laws.jsonl
    python scripts/15_quality_filter.py --input data/processed/train.jsonl --judge --judge_min_score 4.0
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.domain_config import get_domain
from app.data_quality import (
    build_statute_lookup, verify_nhsr, has_disclaimer, has_overpromise, LLMClient,
)


# ──────────────────────────────────────────────
# L1 规则
# ──────────────────────────────────────────────

FACT_RULES = [
    {
        "name": "承诺胜诉",
        "pattern": r"肯定能赢|一定胜诉|保证胜诉|包赢|必胜|稳赢",
        "severity": "error",
        "explanation": "法律结果无法被保证，禁止承诺胜诉/确定结果",
    },
    {
        "name": "伪造胜诉率",
        "pattern": r"胜诉率.{0,6}\d{1,3}%|胜算.{0,6}\d{1,3}%",
        "severity": "error",
        "explanation": "无法验证的胜诉率数字，属编造",
    },
    {
        "name": "冒充律师",
        "pattern": r"我(?:是|作为|以).{0,6}(?:律师|执业律师)|本律师",
        "severity": "error",
        "explanation": "AI 不得冒充执业律师身份",
    },
    {
        "name": "编造司法解释/判例",
        "pattern": r"根据.{0,6}(?:某|相关).{0,4}(?:司法解释|规定|判例).{0,10}(?:显示|表明)",
        "severity": "warning",
        "explanation": "引用未指明具体名称的司法解释/判例，需核实",
    },
]


def check_facts(question: str, answer: str) -> list:
    issues = []
    text = question + " " + answer
    for rule in FACT_RULES:
        if re.search(rule["pattern"], text):
            issues.append({"rule": rule["name"], "severity": rule["severity"],
                           "explanation": rule["explanation"]})
    return issues


def is_legal_related(text: str) -> bool:
    return any(kw in text for kw in get_domain().safety.in_scope_keywords)


def extract_qa(item: dict) -> tuple:
    if "messages" in item:
        q = next((m["content"] for m in item["messages"] if m["role"] == "user"), "")
        a = next((m["content"] for m in item["messages"] if m["role"] == "assistant"), "")
        return q, a
    if "question" in item and "answer" in item:
        return item["question"], item["answer"]
    return "", ""


# ──────────────────────────────────────────────
# L2 引用审计（NHSR）
# ──────────────────────────────────────────────

def audit_citations(answer: str, lookup) -> dict:
    """NHSR 三要素校验，返回 {nhsr, invalid, total}。lookup 为 None 时跳过。"""
    if lookup is None:
        return {"nhsr": None, "invalid": 0, "total": 0, "skipped": True}
    r = verify_nhsr(answer, lookup)
    r["skipped"] = False
    return r


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def filter_dataset(input_path: str, output_path: str = None,
                   min_answer_len: int = 30, max_answer_len: int = 1500,
                   lookup=None, judge_client=None, judge_min_score: float = 4.0,
                   require_disclaimer: bool = True):
    print("=" * 60)
    print("🔍 法律数据质量过滤 + 法条引用审计")
    print("=" * 60)

    with open(input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    print(f"\n[1] 原始样本: {len(data)}")
    stats = Counter()
    kept, filtered, flagged = [], [], []

    for item in data:
        q, a = extract_qa(item)
        if not q or not a:
            stats["无 question/answer 字段"] += 1
            continue

        # L1-1 法律相关度
        if not is_legal_related(q + a):
            stats["非法律相关"] += 1
            filtered.append({"question": q, "reason": "非法律相关"})
            continue

        # L1-2 长度
        if len(a) < min_answer_len:
            stats["回答过短"] += 1
            filtered.append({"question": q, "reason": "回答过短"})
            continue
        if len(a) > max_answer_len:
            stats["回答过长"] += 1
            flagged.append({"question": q, "reason": "回答过长"})

        # L1-3 规则事实校验
        fact_issues = check_facts(q, a)
        if any(i["severity"] == "error" for i in fact_issues):
            stats["法律事实错误"] += 1
            filtered.append({"question": q, "answer": a[:100],
                             "reason": "法律事实错误", "issues": fact_issues})
            continue

        # L1-4 免责声明
        if require_disclaimer and not has_disclaimer(a):
            stats["缺免责声明"] += 1
            flagged.append({"question": q, "reason": "缺免责声明"})

        # L2 引用审计（NHSR）
        audit = audit_citations(a, lookup)
        if not audit.get("skipped"):
            if audit["invalid"] > 0:
                stats["编造法条(NHSR)"] += 1
                filtered.append({"question": q, "answer": a[:100],
                                 "reason": "编造法条", "nhsr": audit["nhsr"]})
                continue

        # L3 LLM-judge（可选）
        if judge_client is not None:
            judge = judge_client.judge_quality(q, a)
            score = float(judge.get("score", 0))
            if score < judge_min_score:
                stats[f"LLM-judge <{judge_min_score}"] += 1
                filtered.append({"question": q, "reason": f"LLM-judge 低分({score})"})
                continue

        kept.append(item)

    print(f"\n[2] 过滤结果:")
    print(f"    保留: {len(kept)} ({100*len(kept)/max(len(data),1):.1f}%)")
    for reason, count in stats.most_common():
        print(f"    过滤-{reason}: {count}")

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for item in kept:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"\n[3] 干净数据保存到: {out}")

        filtered_path = out.parent / "filtered_out.jsonl"
        with open(filtered_path, "w", encoding="utf-8") as f:
            for item in filtered:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"    被过滤样本保存到: {filtered_path}（供人工审核）")

    if filtered:
        print(f"\n[4] 被过滤样本示例（前3条）:")
        for item in filtered[:3]:
            q = item.get("question", "")
            print(f"  ❌ [{item.get('reason','')}] {q[:60]}...")

    return kept, filtered


def main():
    parser = argparse.ArgumentParser(description="法律数据质量过滤 + NHSR 引用审计")
    parser.add_argument("--input", type=str, default="data/processed/train.jsonl")
    parser.add_argument("--output", type=str, default="data/processed/train_clean.jsonl")
    parser.add_argument("--law_file", type=str, default="data/raw/laws_clean.jsonl",
                        help="法条库（用于 NHSR 引用审计），不存在则跳过")
    parser.add_argument("--judge", action="store_true", help="启用 LLM-judge 打分")
    parser.add_argument("--judge_min_score", type=float, default=4.0)
    parser.add_argument("--no_require_disclaimer", action="store_true")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"⚠ 输入文件不存在: {args.input}")
        return

    lookup = None
    if Path(args.law_file).exists():
        lookup = build_statute_lookup(args.law_file)
        print(f"法条库索引: {len(lookup)} 条")
    else:
        print(f"⚠ 法条库不存在，跳过 NHSR 引用审计: {args.law_file}")

    judge_client = None
    if args.judge:
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            print("⚠ 启用 --judge 但未设置 DEEPSEEK_API_KEY，跳过 LLM-judge")
        else:
            judge_client = LLMClient(api_key=key)

    filter_dataset(args.input, args.output, lookup=lookup,
                   judge_client=judge_client, judge_min_score=args.judge_min_score,
                   require_disclaimer=not args.no_require_disclaimer)


if __name__ == "__main__":
    main()
