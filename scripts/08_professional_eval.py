"""08_professional_eval.py — 专业法律指标评测（真实模型 + RAG）。

核心指标（对齐 docs/data_construction_plan.md §11 验收指标）：
  1. NHSR（非幻觉法条率）：回答中每个《法名》第X条 能否在法条库溯源（名称/条号/内容）
  2. 引用幻觉率：含 ≥1 处无法溯源引用的回答占比（越低越好）
  3. 免责声明率：含免责声明的回答占比（目标 100%）
  4. 承诺胜诉率（advice-leakage）：绝对化/承诺胜诉的回答占比（目标 <1%）
  5. ROUGE-L：与参考答案的文本重叠（通用质量）
  6. 术语覆盖：法律要点覆盖

用法：
    python scripts/08_professional_eval.py [--max_cases 50] [--no_rag]
"""
import sys, json, torch, time, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from app.domain_config import get_domain
from app.rag_retriever import get_retriever
from app.data_quality import build_statute_lookup, verify_nhsr, has_disclaimer, has_overpromise

ADAPTER = "outputs/lora_weights/law-lora-r8-20260814-1732"
LAW_FILE = "data/raw/laws_clean.jsonl"
RAG_INSTRUCTION = "根据以下检索到的法条回答问题。若检索结果与问题相关，引用其中真实存在的条文（《法名》第X条）；若检索结果不足以直接回答，给出一般性法律说明、说明依据不足，并建议核实或咨询执业律师；不得编造法条。"


def _lcs_rouge(pred: str, ref: str) -> float:
    p, r = list(pred), list(ref)
    if not p or not r:
        return 0.0
    m, n = len(p), len(r)
    if m * n > 200000:
        p, r = p[:400], r[:400]; m, n = len(p), len(r)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if p[i - 1] == r[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    rec = lcs / n if n else 0
    prec = lcs / m if m else 0
    return 2 * rec * prec / (rec + prec) if rec + prec else 0.0


def term_coverage(pred: str, ref: str) -> float:
    terms = set()
    import re
    for m in re.finditer(r'《[^》]{2,20}》|[一-鿿]{2,6}(?:赔偿|补偿|违约|时效|仲裁|诉讼|竞业|定金|合同|律师)', ref):
        terms.add(m.group())
    if not terms:
        return 1.0
    return sum(1 for t in terms if t in pred) / len(terms)


def _is_refusal(ans: str, n_citations: int) -> bool:
    """拒答判定：零引用 + 含拒答标记（该答没答）。"""
    return n_citations == 0 and any(k in ans for k in ("抱歉", "无法", "不能", "无法回答", "无法提供"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_cases", type=int, default=None)
    ap.add_argument("--no_rag", action="store_true")
    args = ap.parse_args()

    print("加载模型 + 检索器 + 法条库...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True,
                                                 device_map="auto", quantization_config=bnb)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(model, ADAPTER)
    if hasattr(model.generation_config, "enable_thinking"):
        model.generation_config.enable_thinking = False   # 关闭 Qwen3 思考模式
    model.eval()
    retriever = None if args.no_rag else get_retriever()
    lookup = build_statute_lookup(LAW_FILE)
    sys_prompt = get_domain().default_system_prompt

    # 加载测试集
    tc_file = Path("data/test_cases/all_departments.json")
    cases = json.load(open(tc_file, encoding="utf-8")) if tc_file.exists() else []
    if args.max_cases:
        cases = cases[:args.max_cases]
    print(f"评测 {len(cases)} 道题\n")

    results = []
    for i, c in enumerate(cases):
        q = c["question"]
        ref = c.get("answer", "")
        dept = c.get("department", "general")

        # RAG 检索 + 构造 prompt（空上下文也注入指令，避免裸问题触发拒答）
        context = ""
        if retriever:
            docs = retriever.retrieve(q, top_k=3)
            context = retriever.format_context(docs)
        ctx_block = context if context else "【参考法律法规】未检索到直接相关条文。"
        user = f"{RAG_INSTRUCTION}\n{ctx_block}\n\n问题：{q}"
        prompt = (f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
                  f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=300, do_sample=False)
        ans = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        # 兜底剥离思考标记（若仍残留思考模式，恢复 </think> 之后的真实回答）
        if "</think>" in ans:
            ans = ans.split("</think>")[-1].strip()

        # 专业指标
        nhsr = verify_nhsr(ans, lookup, content_check="lexical")
        r = {
            "question": q, "department": dept, "answer": ans[:200],
            "n_citations": nhsr["total"],
            "nhsr": nhsr["nhsr"],  # None 表示无引用
            "has_invalid_citation": nhsr["invalid"] > 0,
            "has_disclaimer": has_disclaimer(ans),
            "has_overpromise": has_overpromise(ans),
            "rouge_l": round(_lcs_rouge(ans, ref), 4),
            "term_cov": round(term_coverage(ans, ref), 4),
        }
        results.append(r)
        print(f"  [{i+1}/{len(cases)}] {dept[:4]} 引用={r['n_citations']} NHSR={r['nhsr']} "
              f"免责={'✓' if r['has_disclaimer'] else '✗'} | {q[:30]}")

    # 聚合
    with_cite = [r for r in results if r["n_citations"] > 0]
    nhsr_vals = [r["nhsr"] for r in with_cite if r["nhsr"] is not None]
    n = len(results)
    overall = {
        "总题数": n,
        "拒答率(零引用且拒答)": round(np.mean([_is_refusal(r["answer"], r["n_citations"]) for r in results]), 4),
        "有效回答率(引用≥1)": round(np.mean([1 if r["n_citations"] > 0 else 0 for r in results]), 4),
        "引用密度(条/答)": round(np.mean([r["n_citations"] for r in results]), 2),
        "NHSR(可溯源引用占比)": round(np.mean(nhsr_vals), 4) if nhsr_vals else None,
        "引用幻觉率(含编造引用)": round(np.mean([r["has_invalid_citation"] for r in results]), 4),
        "免责声明率": round(np.mean([r["has_disclaimer"] for r in results]), 4),
        "承诺胜诉率": round(np.mean([r["has_overpromise"] for r in results]), 4),
        "ROUGE-L": round(np.mean([r["rouge_l"] for r in results]), 4),
        "术语覆盖率": round(np.mean([r["term_cov"] for r in results]), 4),
    }

    # 按类别
    by_dept = defaultdict(list)
    for r in results:
        by_dept[r["department"]].append(r)
    per_cat = {}
    for d, rs in by_dept.items():
        per_cat[d] = {
            "count": len(rs),
            "NHSR": round(np.mean([x["nhsr"] for x in rs if x["nhsr"] is not None]), 3) if any(x["nhsr"] is not None for x in rs) else None,
            "幻觉率": round(np.mean([x["has_invalid_citation"] for x in rs]), 3),
            "免责率": round(np.mean([x["has_disclaimer"] for x in rs]), 3),
            "ROUGE-L": round(np.mean([x["rouge_l"] for x in rs]), 3),
        }

    print("\n" + "=" * 60)
    print("专业指标汇总")
    print("=" * 60)
    for k, v in overall.items():
        print(f"  {k}: {v}")
    print("\n按类别:")
    for d, m in per_cat.items():
        print(f"  {d}: {m}")

    out = {"overall": overall, "per_category": per_cat, "detailed": results}
    Path("outputs/eval_results").mkdir(parents=True, exist_ok=True)
    with open("outputs/eval_results/professional_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n已保存 → outputs/eval_results/professional_metrics.json")


if __name__ == "__main__":
    main()
