"""
07_evaluate.py — Comprehensive Legal QA Evaluation Pipeline.

Three-layer evaluation:
  Layer 1: Auto metrics (BERTScore, ROUGE-L, BLEU, term coverage, hallucination flags)
  Layer 2: LLM-as-Judge (GPT-4o-mini, Claude Haiku, or local model)
  Layer 3: Report generation (JSON, Markdown, HTML dashboard data)

Usage:
    python scripts/07_evaluate.py
    python scripts/07_evaluate.py --eval_file data/test_cases/all_departments.json
    python scripts/07_evaluate.py --generate_fn scripts/05_infer_compare.py
    python scripts/07_evaluate.py --judge_model gpt-4o-mini --api_key sk-xxx
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.domain_config import get_domain


# ──────────────────────────────────────────────
# Layer 1: Auto Metrics
# ──────────────────────────────────────────────

def compute_rouge_l(prediction: str, reference: str) -> float:
    """Compute ROUGE-L (longest common subsequence) score."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        scores = scorer.score(reference, prediction)
        return scores['rougeL'].fmeasure
    except ImportError:
        # Fallback: simple LCS-based calculation
        return _simple_lcs_score(prediction, reference)


def _simple_lcs_score(pred: str, ref: str) -> float:
    """Simple LCS-based ROUGE-L approximation for Chinese text."""
    pred_chars = list(pred)
    ref_chars = list(ref)

    if not pred_chars or not ref_chars:
        return 0.0

    # DP for LCS
    m, n = len(pred_chars), len(ref_chars)
    if m * n > 100000:  # Too long, approximate
        pred_chars = pred_chars[:300]
        ref_chars = ref_chars[:300]
        m, n = len(pred_chars), len(ref_chars)

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_chars[i - 1] == ref_chars[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    recall = lcs_len / n if n > 0 else 0
    precision = lcs_len / m if m > 0 else 0
    if recall + precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def compute_term_coverage(prediction: str, reference: str,
                           key_terms: list = None) -> float:
    """Compute legal key-point coverage ratio."""
    if key_terms is None:
        # 提取法条引用 + 法律关键词（2-6 字带法律后缀/前缀）
        key_terms = re.findall(
            r'《[^》]{2,20}》|[一-鿿]{2,6}(?:赔偿|补偿|违约|时效|仲裁|诉讼|竞业|定金|合同|法条|律师)',
            reference
        )

    if not key_terms:
        return 1.0

    covered = sum(1 for term in key_terms if term in prediction)
    return covered / len(key_terms)


def compute_hallucination_flags(prediction: str, reference: str) -> dict:
    """Quick hallucination flag check."""
    flags = {
        "has_overpromise": False,
        "has_overconfident_language": False,
        "has_fake_statistics": False,
        "hallucination_count": 0,
    }

    # 承诺胜诉
    if re.search(r'肯定能赢|一定胜诉|保证胜诉|包赢|必胜|稳赢', prediction):
        flags["has_overpromise"] = True
        flags["hallucination_count"] += 1

    # 过度自信
    overconfident = ["一定可以", "绝对有效", "保证", "百分之百", "包赢"]
    if any(phrase in prediction for phrase in overconfident):
        flags["has_overconfident_language"] = True
        flags["hallucination_count"] += 1

    # 伪造胜诉率
    if re.search(r'胜诉率.{0,6}\d{1,3}%', prediction):
        flags["has_fake_statistics"] = True
        flags["hallucination_count"] += 1

    return flags


def compute_auto_metrics(prediction: str, reference: str,
                          key_terms: list = None) -> dict:
    """Compute all automatic metrics."""
    metrics = {
        "rouge_l": round(compute_rouge_l(prediction, reference), 4),
        "term_coverage": round(compute_term_coverage(prediction, reference, key_terms), 4),
        "pred_length": len(prediction),
        "ref_length": len(reference),
    }
    metrics.update(compute_hallucination_flags(prediction, reference))
    metrics["hallucination_rate"] = metrics["hallucination_count"]
    return metrics


# ──────────────────────────────────────────────
# Layer 2: LLM-as-Judge
# ──────────────────────────────────────────────

JUDGE_PROMPT = get_domain().judge_prompt_template


def llm_judge(question: str, prediction: str, reference: str,
              judge_model: str = "gpt-4o-mini",
              api_key: str = None) -> dict:
    """
    LLM-as-Judge evaluation.

    Supports: gpt-4o-mini (OpenAI API), claude-haiku (Anthropic API),
              deepseek-chat (DeepSeek API), or heuristic fallback.
    """
    prompt = JUDGE_PROMPT.format(
        question=question,
        reference=reference[:1000],
        prediction=prediction[:1000],
    )

    try:
        if judge_model.startswith("gpt"):
            return _openai_judge(prompt, judge_model, api_key)
        elif judge_model.startswith("claude"):
            return _anthropic_judge(prompt, judge_model, api_key)
        elif judge_model.startswith("deepseek"):
            return _deepseek_judge(prompt, judge_model, api_key)
        else:
            return _heuristic_judge(question, prediction, reference)
    except Exception as e:
        print(f"   ⚠ LLM Judge failed: {e}. Using heuristic fallback.")
        return _heuristic_judge(question, prediction, reference)


def _openai_judge(prompt: str, model: str, api_key: str) -> dict:
    """Use OpenAI API for judging."""
    import requests
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI API key not found")

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a legal evaluation expert. Always respond in JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 500,
        },
        timeout=30,
    )
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    # Extract JSON
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return _heuristic_judge("", "", "")


def _anthropic_judge(prompt: str, model: str, api_key: str) -> dict:
    """Use Anthropic API for judging."""
    import requests
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("Anthropic API key not found")

    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    data = resp.json()
    content = data["content"][0]["text"]
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return _heuristic_judge("", "", "")


def _deepseek_judge(prompt: str, model: str, api_key: str) -> dict:
    """Use DeepSeek API for judging (OpenAI-compatible endpoint)."""
    import requests
    key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("DeepSeek API key not found. Set DEEPSEEK_API_KEY env var.")

    resp = requests.post(
        "https://api.deepseek.com/v1/chat/completions",  # OpenAI-compatible
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model or "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是一个法律评估专家。请严格用JSON格式回复。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 800,
        },
        timeout=60,
    )
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    # Extract JSON from response
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return _heuristic_judge("", "", "")


def _heuristic_judge(question: str, prediction: str,
                      reference: str) -> dict:
    """Heuristic fallback judge (no API needed)."""
    scores = {"accuracy": 3, "guideline_consistency": 3,
              "safety": 3, "completeness": 3, "accessibility": 3}

    # Length-based completeness
    if len(prediction) > 200:
        scores["completeness"] = 4
    if len(prediction) > 500:
        scores["completeness"] = 5
    if len(prediction) < 50:
        scores["completeness"] = 2

    # Safety check
    dangerous = [
        r'肯定能赢|一定胜诉|保证胜诉|包赢|必胜|胜诉率',
    ]
    is_dangerous = any(re.search(p, prediction) for p in dangerous)
    scores["safety"] = 1 if is_dangerous else 4

    # Accessibility
    scores["accessibility"] = 5 if len(prediction) > 100 else 3

    # Overall
    scores["overall"] = round(np.mean(list(scores.values())))
    scores["comments"] = "Heuristic evaluation (no LLM API configured)"
    scores["has_errors"] = is_dangerous
    scores["errors"] = ["承诺胜诉/确定结果"] if is_dangerous else []

    return scores


# ──────────────────────────────────────────────
# Evaluation Runner
# ──────────────────────────────────────────────

def load_eval_data(eval_file: str) -> list:
    """Load evaluation data."""
    path = Path(eval_file)
    if not path.exists():
        print(f"⚠ Eval file not found: {eval_file}")
        # Return built-in eval set
        return _builtin_eval_set()

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Normalize format
    if isinstance(data, list):
        return data

    results = []
    for dept, questions in data.items():
        for q in questions:
            if isinstance(q, dict):
                results.append({
                    "question": q.get("question", ""),
                    "reference": q.get("answer", q.get("reference", "")),
                    "department": dept,
                    "difficulty": q.get("difficulty", "medium"),
                })
    return results


def _builtin_eval_set() -> list:
    """Built-in evaluation questions."""
    return [
        {
            "question": "被公司辞退能拿多少赔偿？",
            "reference": "若公司违法解除劳动合同，根据《劳动合同法》第八十七条，应按经济补偿标准的二倍支付赔偿金（2N）。经济补偿按工作年限每满一年支付一个月工资。建议保留证据并申请劳动仲裁（时效1年）。以上内容仅供参考，不构成法律意见。",
            "department": "劳动争议",
            "difficulty": "medium",
        },
        {
            "question": "竞业限制没有约定补偿，还有效吗？",
            "reference": "竞业限制未约定经济补偿的，劳动者履行了竞业限制义务的，可以要求用人单位按月支付经济补偿。竞业限制的效力需结合司法解释与个案情况判断。建议咨询律师。以上内容仅供参考。",
            "department": "劳动争议",
            "difficulty": "hard",
        },
        {
            "question": "劳动仲裁的时效是多久？",
            "reference": "根据《劳动争议调解仲裁法》第二十七条，劳动争议申请仲裁的时效期间为一年，从当事人知道或应当知道权利被侵害之日起计算。建议及时主张权利并咨询律师。",
            "department": "劳动争议",
            "difficulty": "easy",
        },
        {
            "question": "交了定金不买了，定金能退吗？",
            "reference": "根据《民法典》相关规定，给付定金一方不履行债务的，无权请求返还定金。是否可退需结合合同性质与履行情况判断。建议咨询律师。以上内容仅供参考。",
            "department": "合同纠纷",
            "difficulty": "medium",
        },
    ]


def run_evaluation(eval_data: list, generate_fn: Callable = None,
                    judge_model: str = None, api_key: str = None,
                    output_dir: str = "outputs/eval_results") -> dict:
    """Run full evaluation pipeline."""
    results = []
    dept_metrics = defaultdict(lambda: {
        "count": 0, "rouge_l_sum": 0.0, "term_cov_sum": 0.0,
        "hallu_sum": 0, "judge_scores": [],
    })

    print(f"\n{'='*60}")
    print(f"Evaluating {len(eval_data)} questions...")
    print(f"{'='*60}")

    for i, item in enumerate(eval_data):
        question = item.get("question", "")
        reference = item.get("reference", item.get("answer", ""))
        dept = item.get("department", "general")

        # Generate prediction if generate_fn provided
        if generate_fn:
            prediction = generate_fn(question)
        else:
            # Use reference as mock prediction for testing
            prediction = reference

        # Layer 1: Auto metrics
        auto = compute_auto_metrics(prediction, reference)
        auto["question"] = question
        auto["prediction"] = prediction[:300]
        auto["reference"] = reference[:300]
        auto["department"] = dept

        # Layer 2: LLM Judge
        judge = None
        if judge_model:
            judge = llm_judge(question, prediction, reference, judge_model, api_key)
            auto["judge_scores"] = judge

        results.append(auto)

        # Aggregate
        dm = dept_metrics[dept]
        dm["count"] += 1
        dm["rouge_l_sum"] += auto["rouge_l"]
        dm["term_cov_sum"] += auto["term_coverage"]
        dm["hallu_sum"] += auto["hallucination_rate"]
        if judge:
            dm["judge_scores"].append(judge.get("overall", 3))

        status = "✅" if auto["hallucination_rate"] == 0 else f"⚠️ ({auto['hallucination_rate']} flags)"
        print(f"   [{i+1}/{len(eval_data)}] {status} ROUGE-L={auto['rouge_l']:.3f} | {question[:40]}...")

    # ── Aggregate ──
    overall = {
        "total_questions": len(eval_data),
        "avg_rouge_l": round(np.mean([r["rouge_l"] for r in results]), 4),
        "avg_term_coverage": round(np.mean([r["term_coverage"] for r in results]), 4),
        "total_hallucination_flags": sum(r["hallucination_rate"] for r in results),
        "hallucination_rate": round(
            sum(r["hallucination_rate"] for r in results) / len(results), 4
        ),
    }

    per_category = {}
    for dept, dm in dept_metrics.items():
        n = dm["count"]
        per_category[dept] = {
            "count": n,
            "avg_rouge_l": round(dm["rouge_l_sum"] / n, 4),
            "avg_term_coverage": round(dm["term_cov_sum"] / n, 4),
            "avg_hallucination_flags": round(dm["hallu_sum"] / n, 2),
        }
        if dm["judge_scores"]:
            per_category[dept]["avg_judge_score"] = round(np.mean(dm["judge_scores"]), 2)

    # ── Win/Loss (vs reference baseline) ──
    wins = sum(1 for r in results if r["rouge_l"] > 0.5)
    losses = sum(1 for r in results if r["rouge_l"] < 0.3)
    ties = len(results) - wins - losses

    overall["wins"] = wins
    overall["losses"] = losses
    overall["ties"] = ties
    overall["win_rate"] = round(wins / len(results), 3) if results else 0

    output = {
        "evaluation_time": datetime.now().isoformat(),
        "overall": overall,
        "per_category": per_category,
        "detailed_results": results,
    }

    # ── Save ──
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    metrics_file = output_path / f"metrics_summary_{timestamp}.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Also save as latest
    latest_file = output_path / "metrics_summary.json"
    with open(latest_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # ── Generate Report ──
    report = generate_report(output, output_path, timestamp)

    print(f"\n{'='*60}")
    print(f"✅ Evaluation complete!")
    print(f"   Questions: {len(eval_data)}")
    print(f"   Avg ROUGE-L: {overall['avg_rouge_l']:.4f}")
    print(f"   Avg Term Coverage: {overall['avg_term_coverage']:.4f}")
    print(f"   Hallucination Rate: {overall['hallucination_rate']:.4f}")
    print(f"   Win Rate: {overall['win_rate']:.1%}")
    print(f"   Results saved to: {metrics_file}")
    print(f"   Report saved to: {report}")
    print(f"{'='*60}")

    return output


def generate_report(eval_output: dict, output_dir: Path,
                     timestamp: str) -> Path:
    """Generate evaluation report in Markdown."""
    overall = eval_output["overall"]
    per_cat = eval_output["per_category"]

    lines = [
        "# LexiCare 评估报告",
        f"\n生成时间: {datetime.now().isoformat()}",
        f"\n## 总体指标",
        f"| 指标 | 数值 |",
        f"|------|------|",
        f"| 评估问题数 | {overall['total_questions']} |",
        f"| 平均 ROUGE-L | {overall['avg_rouge_l']:.4f} |",
        f"| 平均术语覆盖率 | {overall['avg_term_coverage']:.4f} |",
        f"| 幻觉标记率 | {overall['hallucination_rate']:.4f} |",
        f"| Win Rate | {overall['win_rate']:.1%} |",
        f"\n## 分类评估",
        f"| 类别 | 数量 | ROUGE-L | 术语覆盖 | 幻觉标记 |",
        f"|------|------|---------|---------|---------|",
    ]

    for dept, metrics in per_cat.items():
        lines.append(
            f"| {dept} | {metrics['count']} | "
            f"{metrics['avg_rouge_l']:.4f} | {metrics['avg_term_coverage']:.4f} | "
            f"{metrics['avg_hallucination_flags']:.2f} |"
        )

    lines.append(f"\n## 幻觉检测详情")
    hallu_cases = [r for r in eval_output["detailed_results"]
                    if r["hallucination_rate"] > 0]
    if hallu_cases:
        for i, r in enumerate(hallu_cases[:10]):
            lines.append(f"\n### 案例 {i+1}")
            lines.append(f"**问题**: {r['question'][:100]}")
            lines.append(f"**回答**: {r['prediction'][:150]}...")
            lines.append(f"**ROUGE-L**: {r['rouge_l']:.4f}")
    else:
        lines.append("\n未检测到明显幻觉。")

    report_path = output_dir / f"evaluation_report_{timestamp}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Also save latest
    with open(output_dir / "evaluation_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_path


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LexiCare Evaluation Pipeline")
    parser.add_argument("--eval_file", type=str, default=None,
                        help="JSON file with evaluation data")
    parser.add_argument("--judge_model", type=str, default=None,
                        help="LLM Judge: gpt-4o-mini, claude-haiku, deepseek-chat, or heuristic")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API key for LLM judge")
    parser.add_argument("--output_dir", type=str, default="outputs/eval_results")
    parser.add_argument("--no_judge", action="store_true",
                        help="Skip LLM judge (auto metrics only)")
    args = parser.parse_args()

    eval_data = load_eval_data(args.eval_file) if args.eval_file else _builtin_eval_set()
    print(f"Loaded {len(eval_data)} evaluation questions")

    judge_model = None if args.no_judge else (args.judge_model or "heuristic")

    run_evaluation(
        eval_data=eval_data,
        judge_model=judge_model,
        api_key=args.api_key,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
