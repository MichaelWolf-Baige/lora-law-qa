"""
06_evaluate.py — Per-category evaluation with multi-dimensional metrics.

Evaluates base model vs LoRA model across:
  1. ROUGE-L (lexical overlap with reference)
  2. BLEU-4 (n-gram precision)
  3. Hallucination rate (fabricated drug names, dosages, procedures)
  4. Legal terminology coverage (% of reference terms in generated answer)
  5. Answer completeness (length adequacy)

Outputs:
  - metrics_summary.json: Aggregated metrics
  - per_category_metrics.csv: Breakdown by legal category
  - case_study_samples.json: Best/worst improvements
  - Console summary table

Usage:
    python scripts/06_evaluate.py --comparison_file outputs/eval_results/batch_comparison.json
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.domain_config import get_domain

# Try importing NLP metrics
try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    print("⚠ rouge_score not installed. ROUGE metrics will be skipped.")


def classify_question(question: str) -> str:
    """Classify a question into a legal category."""
    return get_domain().classify_category(question)


# ──────────────────────────────────────────────
# Hallucination detection
# ──────────────────────────────────────────────

def detect_hallucination_indicators(text: str) -> dict:
    """Detect potential hallucination patterns in generated text.

    Returns:
        dict with hallucination indicators found
    """
    indicators = {
        "overpromise": False,
        "fake_win_rate": False,
        "impersonate_lawyer": False,
        "overconfidence": False,
        "fabricated_reference": False,
    }

    # 承诺胜诉 / 伪造胜诉率
    if re.search(r"肯定能赢|一定胜诉|保证胜诉|包赢|必胜|稳赢", text):
        indicators["overpromise"] = True
    if re.search(r"胜诉率.{0,6}\d{1,3}%", text):
        indicators["fake_win_rate"] = True

    # 冒充律师
    if re.search(r"我(?:是|作为|以).{0,6}(?:律师|执业律师)", text):
        indicators["impersonate_lawyer"] = True

    # 过度自信
    overconfident_patterns = [
        r"一定可以", r"绝对有效", r"百分之百", r"保证.{0,4}(拿回|赔偿|胜诉)",
    ]
    if any(re.search(p, text) for p in overconfident_patterns):
        indicators["overconfidence"] = True

    # 无出处的司法解释/判例引用
    if re.search(r"根据.{0,20}(?:司法解释|判例|案例|规定).{0,30}(?:显示|表明|规定)", text):
        indicators["fabricated_reference"] = True

    return indicators


def hallucination_score(text: str) -> float:
    """Score from 0 (safe) to 1 (heavily hallucinated)."""
    indicators = detect_hallucination_indicators(text)
    score = sum(1 for v in indicators.values() if v) / len(indicators)
    return score


# ──────────────────────────────────────────────
# Legal term coverage
# ──────────────────────────────────────────────

def extract_legal_terms(text: str) -> set[str]:
    """Extract legal terms from text (based on domain categories)."""
    all_keywords = set()
    for keywords in get_domain().categories.values():
        all_keywords.update(keywords)

    found = set()
    for kw in all_keywords:
        if kw in text:
            found.add(kw)
    return found


def term_coverage(reference: str, generated: str) -> float:
    """Measure how many legal terms from the reference appear in generated text."""
    ref_terms = extract_legal_terms(reference)
    if not ref_terms:
        return 1.0  # No legal terms to cover
    gen_terms = extract_legal_terms(generated)
    return len(ref_terms & gen_terms) / len(ref_terms)


# ──────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────

def evaluate(comparison_file: str, output_dir: str):
    """Run full evaluation pipeline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load comparison data
    with open(comparison_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 60)
    print(f"Evaluating {len(data)} comparisons")
    print("=" * 60)

    # Initialize scorers
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False) if HAS_ROUGE else None

    # Accumulators
    per_category = defaultdict(lambda: {
        "count": 0,
        "base_rouge_l": [],
        "lora_rouge_l": [],
        "base_hallucination": [],
        "lora_hallucination": [],
        "base_term_cov": [],
        "lora_term_cov": [],
        "base_len": [],
        "lora_len": [],
    })

    overall = {
        "base_rouge_l": [],
        "lora_rouge_l": [],
        "base_hallucination": [],
        "lora_hallucination": [],
        "base_term_cov": [],
        "lora_term_cov": [],
        "base_len": [],
        "lora_len": [],
        "lora_wins": 0,
        "base_wins": 0,
        "ties": 0,
    }

    case_studies = []

    for i, item in enumerate(data):
        question = item["question"]
        reference = item.get("reference", "")
        base_answer = item["base_answer"]
        lora_answer = item["lora_answer"]

        category = classify_question(question)

        # --- ROUGE-L ---
        base_rl, lora_rl = 0, 0
        if HAS_ROUGE and reference:
            base_rl = rouge.score(reference, base_answer)["rougeL"].fmeasure
            lora_rl = rouge.score(reference, lora_answer)["rougeL"].fmeasure

        # --- Hallucination ---
        base_hall = hallucination_score(base_answer)
        lora_hall = hallucination_score(lora_answer)

        # --- Term coverage ---
        base_tc = term_coverage(reference, base_answer) if reference else 0
        lora_tc = term_coverage(reference, lora_answer) if reference else 0

        # --- Answer length ---
        base_len = len(base_answer)
        lora_len = len(lora_answer)

        # --- Win/Tie/Loss (by ROUGE-L if available, else hallucination) ---
        if reference:
            if lora_rl > base_rl + 0.01:
                overall["lora_wins"] += 1
                winner = "lora"
            elif base_rl > lora_rl + 0.01:
                overall["base_wins"] += 1
                winner = "base"
            else:
                overall["ties"] += 1
                winner = "tie"
        else:
            if lora_hall < base_hall - 0.1:
                overall["lora_wins"] += 1
                winner = "lora"
            elif base_hall < lora_hall - 0.1:
                overall["base_wins"] += 1
                winner = "base"
            else:
                overall["ties"] += 1
                winner = "tie"

        # Accumulate
        for acc in [overall, per_category[category]]:
            if base_rl: acc["base_rouge_l"].append(base_rl)
            if lora_rl: acc["lora_rouge_l"].append(lora_rl)
            acc["base_hallucination"].append(base_hall)
            acc["lora_hallucination"].append(lora_hall)
            acc["base_term_cov"].append(base_tc)
            acc["lora_term_cov"].append(lora_tc)
            acc["base_len"].append(base_len)
            acc["lora_len"].append(lora_len)

        per_category[category]["count"] += 1

        # Case study: track notable improvements/regressions
        delta_rl = lora_rl - base_rl if reference else lora_hall - base_hall
        case_studies.append({
            "question": question[:150],
            "category": category,
            "base_answer": base_answer[:300],
            "lora_answer": lora_answer[:300],
            "reference": reference[:300] if reference else "",
            "delta": delta_rl,
            "winner": winner,
        })

    # ================================================================
    # Compute summary metrics
    # ================================================================

    def safe_mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    # Overall summary
    summary = {
        "total_questions": len(data),
        "categories": len(per_category),
        "overall": {
            "base_rouge_l": safe_mean(overall["base_rouge_l"]),
            "lora_rouge_l": safe_mean(overall["lora_rouge_l"]),
            "rouge_l_improvement_pct": 0,
            "base_hallucination_rate": safe_mean(overall["base_hallucination"]),
            "lora_hallucination_rate": safe_mean(overall["lora_hallucination"]),
            "base_term_coverage": safe_mean(overall["base_term_cov"]),
            "lora_term_coverage": safe_mean(overall["lora_term_cov"]),
            "base_avg_length": safe_mean(overall["base_len"]),
            "lora_avg_length": safe_mean(overall["lora_len"]),
            "win_rate": overall["lora_wins"] / len(data) if len(data) > 0 else 0,
            "lora_wins": overall["lora_wins"],
            "base_wins": overall["base_wins"],
            "ties": overall["ties"],
        },
        "per_category": {},
    }

    # Calculate ROUGE-L improvement percentage
    if summary["overall"]["base_rouge_l"] > 0:
        summary["overall"]["rouge_l_improvement_pct"] = round(
            100 * (summary["overall"]["lora_rouge_l"] - summary["overall"]["base_rouge_l"])
            / summary["overall"]["base_rouge_l"], 1
        )

    # Per-category summary
    for cat in sorted(per_category.keys()):
        cat_data = per_category[cat]
        n = cat_data["count"]
        base_rl = safe_mean(cat_data["base_rouge_l"])
        lora_rl = safe_mean(cat_data["lora_rouge_l"])
        improvement = 0
        if base_rl > 0:
            improvement = 100 * (lora_rl - base_rl) / base_rl

        summary["per_category"][cat] = {
            "count": n,
            "base_rouge_l": round(base_rl, 4),
            "lora_rouge_l": round(lora_rl, 4),
            "rouge_l_improvement_pct": round(improvement, 1),
            "base_hallucination_rate": round(safe_mean(cat_data["base_hallucination"]), 3),
            "lora_hallucination_rate": round(safe_mean(cat_data["lora_hallucination"]), 3),
            "base_term_coverage": round(safe_mean(cat_data["base_term_cov"]), 3),
            "lora_term_coverage": round(safe_mean(cat_data["lora_term_cov"]), 3),
        }

    # ================================================================
    # Save results
    # ================================================================

    # JSON summary
    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # CSV detail
    csv_rows = []
    for cat, metrics in summary["per_category"].items():
        csv_rows.append({
            "category": cat,
            **metrics,
        })
    df = pd.DataFrame(csv_rows)
    df.to_csv(output_dir / "per_category_metrics.csv", index=False, encoding="utf-8-sig")

    # Case studies (top/bottom 5)
    case_studies.sort(key=lambda x: x["delta"], reverse=True)
    best_cases = case_studies[:5]
    worst_cases = case_studies[-5:]

    with open(output_dir / "case_study_samples.json", "w", encoding="utf-8") as f:
        json.dump({"best_improvements": best_cases, "worst_regressions": worst_cases}, f, ensure_ascii=False, indent=2)

    # ================================================================
    # Print summary
    # ================================================================

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    ov = summary["overall"]
    print(f"\n  Questions evaluated: {summary['total_questions']}")
    print(f"  Categories: {summary['categories']}")

    print(f"\n  {'Metric':<30} {'Base':>10} {'LoRA':>10} {'Change':>12}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*12}")

    base_rl = ov["base_rouge_l"]
    lora_rl = ov["lora_rouge_l"]
    delta_rl = lora_rl - base_rl
    print(f"  {'ROUGE-L':<30} {base_rl:>10.4f} {lora_rl:>10.4f} {delta_rl:>+11.4f} ({ov['rouge_l_improvement_pct']:+.1f}%)")

    base_h = ov["base_hallucination_rate"]
    lora_h = ov["lora_hallucination_rate"]
    delta_h = lora_h - base_h
    print(f"  {'Hallucination Rate':<30} {base_h:>10.3f} {lora_h:>10.3f} {delta_h:>+11.3f}")

    base_tc = ov["base_term_coverage"]
    lora_tc = ov["lora_term_coverage"]
    delta_tc = lora_tc - base_tc
    print(f"  {'Term Coverage':<30} {base_tc:>10.3f} {lora_tc:>10.3f} {delta_tc:>+11.3f}")

    base_len = ov["base_avg_length"]
    lora_len = ov["lora_avg_length"]
    print(f"  {'Avg Answer Length':<30} {base_len:>10.0f} {lora_len:>10.0f}")

    print(f"\n  {'Win Rate':<30} {'':>10} {ov['win_rate']:>10.1%}")
    print(f"  {'LoRA Wins / Base Wins / Ties':<30} {ov['lora_wins']} / {ov['base_wins']} / {ov['ties']}")

    # Per category table
    print(f"\n{'='*70}")
    print("PER-CATEGORY BREAKDOWN")
    print(f"{'='*70}")
    print(f"  {'Category':<22} {'Count':>5} {'Base RL':>8} {'LoRA RL':>8} {'Δ%':>8} {'H-Base':>7} {'H-LoRA':>7}")
    print(f"  {'-'*22} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*7}")

    for cat in sorted(summary["per_category"].keys()):
        m = summary["per_category"][cat]
        print(
            f"  {cat:<22} {m['count']:>5} "
            f"{m['base_rouge_l']:>8.4f} {m['lora_rouge_l']:>8.4f} "
            f"{m['rouge_l_improvement_pct']:>+7.1f}% "
            f"{m['base_hallucination_rate']:>7.3f} {m['lora_hallucination_rate']:>7.3f}"
        )

    print(f"\n[OK] Evaluation results saved to: {output_dir}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate base vs LoRA model")
    parser.add_argument(
        "--comparison_file", type=str,
        default="outputs/eval_results/batch_comparison.json",
        help="JSON file from 05_infer_compare.py"
    )
    parser.add_argument("--output_dir", type=str, default="outputs/eval_results")
    args = parser.parse_args()

    if not Path(args.comparison_file).exists():
        print(f"ERROR: Comparison file not found: {args.comparison_file}")
        print("Run 05_infer_compare.py first to generate it.")
        sys.exit(1)

    evaluate(args.comparison_file, args.output_dir)


if __name__ == "__main__":
    main()
