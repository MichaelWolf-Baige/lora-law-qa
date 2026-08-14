"""
14_distill_guidelines.py — 法条 → 带引用 QA 蒸馏（两步法 + NHSR 三要素校验）。

取代旧的「字符串子串匹配」校验：旧版只检查法条号是否出现在 chunk 里，
不校验「条号↔内容」是否真实对应，也漏中文数字、跨 chunk 引用。

新方法（对齐 InternLM-Law / 2501.06521 实证）：
  Step 1 条文合成（默认，天然 grounded）：
         给定真实法条片段 → 教师模型生成「咨询问题 + 含精确条号引用的回答」
  Step 2 NHSR 三要素校验（名称/条号/内容 全对）：
         回答中的每个《法名》第X条 必须能在全局法条库中溯源
  Step 3 纠错替换（可选 --attempt_correction）：
         NHSR 不通过的，用教师模型按「真实法条库」改写修正；
         仍不过的丢弃（默认直接丢弃，安全优先）

核心：只保留 nhsr==1.0 且含免责声明、无承诺胜诉的回答。

用法：
    set DEEPSEEK_API_KEY=sk-xxx
    python scripts/14_distill_guidelines.py                     # 默认蒸馏 laws_clean.jsonl（1,479 部干净有效法条）
    python scripts/14_distill_guidelines.py --dry_run          # 只测 1 条
    python scripts/14_distill_guidelines.py --attempt_correction
    python scripts/14_distill_guidelines.py --law_file data/raw/laws.jsonl   # 如需全量（含历史版本，慎用）

注意：默认使用 laws_clean.jsonl（已剔除历史版本/失效条文/地方法规），避免从过期法条蒸馏；
laws.jsonl（22,552 部全量）含历史版本，仅在明确需要时显式传入。
"""

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.domain_config import get_domain
from app.data_quality import (
    LLMClient, build_statute_lookup, verify_nhsr,
    extract_statute_citations, has_disclaimer, has_overpromise,
)


# ──────────────────────────────────────────────
# 教师模型 Prompt
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """你是法律领域的数据标注专家，负责从法条/司法解释中生成高质量的法律问答对。

生成规则（严格遵守）：
1. 问题必须是真实咨询者口吻（"我..."、"公司..."、"被辞退..."等）
2. 回答必须 100% 基于给出的法条原文，**只能引用法条片段中真实存在的法条号**
3. 引用格式必须是《法名》第X条，法名与条号都必须准确（不得编造）
4. 关键数字（N/N+1/2N、时效年限、违约金比例）必须与法条原文一致
5. 每个回答必须包含"以上内容仅供参考，不构成法律意见"类免责声明
6. 禁止承诺胜诉、禁止给出确定结果（"肯定能赢"、"胜诉率X%"等）

输出格式（纯 JSON 数组，不要输出其他内容）：
[{"question": "咨询者问题", "answer": "基于法条、含准确引用与免责声明的回答"}]"""


QA_TEMPLATE = """请阅读以下法条片段，生成 {n} 个不同类型的法律问答对：

【法条片段】
{law_text}

【要求】覆盖以下类型（尽量均衡）：
- 赔偿计算类："被辞退能拿多少赔偿""经济补偿怎么算"
- 竞业限制类："竞业限制没补偿有效吗"
- 时效类："仲裁/诉讼时效是多久"
- 合同类："定金能退吗""违约金怎么算"
- 劳动争议类："拖欠工资怎么办""违法解除怎么认定"

只能引用上面法条片段中真实存在的法条号，直接输出 JSON 数组。"""


CORRECT_PROMPT = """以下回答引用了不存在或张冠李戴的法条号，请改写修正，使每个引用都指向真实法条。

【原始回答】
{answer}

【问题】
{question}

【真实法条库中可用的引用】（法名 + 条号）
{available}

要求：
1. 只把错误引用替换为上面列出的真实法条号（或删除无法修正的引用）
2. 法律结论必须与真实法条内容一致，不得改错
3. 保留免责声明
4. 直接输出修正后的回答全文，不要输出其他内容。"""


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def load_law_chunks(law_file: str, chunk_size: int = 1500,
                    focus_types: tuple = None) -> list:
    """加载并切分法条文本。默认聚焦「法律 + 司法解释 + 法律解释 + 行政法规」。"""
    if focus_types is None:
        focus_types = ("法律", "司法解释", "法律解释", "行政法规", "宪法", "监察法规")

    chunks = []
    law_path = Path(law_file)
    if not law_path.exists():
        print(f"  ⚠ 法条文件不存在: {law_file}（先运行 01_download_data.py）")
        return chunks

    with open(law_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            typ = item.get("type", "")
            if focus_types and typ not in focus_types:
                continue
            title = item.get("title", "")
            content = item.get("content", "")
            if not content:
                continue
            paragraphs = content.split("\n")
            current = ""
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                if len(current) + len(para) > chunk_size and current:
                    chunks.append({"source": f"{title}（{typ}）", "title": title,
                                   "text": current})
                    current = para
                else:
                    current += para + "\n"
            if current.strip():
                chunks.append({"source": f"{title}（{typ}）", "title": title,
                               "text": current.strip()})
    return chunks


def list_available_citations(chunk_text: str, lookup) -> list:
    """列出法条片段中可引用的真实法条号（供纠错 prompt 用）。"""
    av = set()
    for c in extract_statute_citations(chunk_text):
        full = lookup.resolve_law(c["law"])
        if full:
            av.add(f"{full} 第{c['article']}条")
        else:
            av.add(f"{c['law']} 第{c['article']}条")
    return sorted(av)


def gate_answer(question: str, answer: str, lookup, require_disclaimer: bool = True):
    """
    质量闸门：NHSR 全对 + 免责声明 + 无承诺胜诉。

    返回 (keep: bool, reason: str, info: dict)。
    """
    nhsr = verify_nhsr(answer, lookup, content_check="lexical")
    if nhsr["invalid"] > 0:
        if nhsr["content_invalid"] > 0:
            return False, (f"编造法条 {nhsr['invalid']} 处"
                           f"（含内容不符 {nhsr['content_invalid']} 处）"), nhsr
        return False, f"编造法条 {nhsr['invalid']} 处", nhsr
    if require_disclaimer and not has_disclaimer(answer):
        return False, "缺免责声明", nhsr
    if has_overpromise(question + " " + answer):
        return False, "承诺胜诉/绝对化断言", nhsr
    return True, "ok", nhsr


def synthesize_chunk(chunk: dict, lookup, client: LLMClient, n_qa: int = 3,
                     attempt_correction: bool = False):
    """从单个法条片段蒸馏 QA 对（合成 + NHSR 闸门 + 可选纠错）。"""
    prompt = QA_TEMPLATE.format(n=n_qa, law_text=chunk["text"][:2000])
    raw = client.chat(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": prompt}],
        temperature=0.3, max_tokens=2000,
    )

    json_match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not json_match:
        return []
    try:
        qa_pairs = json.loads(json_match.group())
    except json.JSONDecodeError:
        return []

    validated = []
    for pair in qa_pairs:
        q = (pair.get("question") or "").strip()
        a = (pair.get("answer") or "").strip()
        if not q or not a:
            continue
        keep, reason, nhsr = gate_answer(q, a, lookup)

        # 可选：纠错替换
        if not keep and attempt_correction and nhsr["invalid"] > 0:
            a = correct_answer(q, a, chunk, lookup, client)
            keep, reason, nhsr = gate_answer(q, a, lookup)

        validated.append({
            "question": q, "answer": a, "source": chunk["source"],
            "fact_verified": keep, "reject_reason": reason if not keep else "",
            "nhsr": nhsr["nhsr"], "n_citations": nhsr["total"],
            "generated_by": "deepseek-distillation-v2",
        })
    return validated


def correct_answer(question: str, answer: str, chunk: dict, lookup,
                   client: LLMClient) -> str:
    """纠错替换：让教师模型按真实法条库改写，修正编造/张冠李戴的引用。"""
    available = list_available_citations(chunk["text"], lookup)
    prompt = CORRECT_PROMPT.format(
        answer=answer, question=question,
        available="、".join(available) if available else "（无可用引用，请删除错误引用）",
    )
    try:
        return client.chat(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=2000,
        ).strip()
    except Exception:
        return answer


def demo_gate():
    """不调 API 的闸门自测：验证 NHSR / 免责声明 / 承诺胜诉 三个闸门。"""
    from app.data_quality import build_statute_lookup
    laws = [{
        "title": "中华人民共和国劳动合同法",
        "content": "第八十七条：用人单位违反本法规定解除或者终止劳动合同的，"
                   "应当依照本法第四十七条规定的经济补偿标准的二倍向劳动者支付赔偿金。",
    }]
    lookup = build_statute_lookup(laws=laws)
    cases = [
        ("通过-正确引用",
         "被公司违法辞退能拿多少赔偿？",
         "根据《劳动合同法》第八十七条，违法解除应支付二倍赔偿金（2N）。"
         "以上内容仅供参考，不构成法律意见。"),
        ("拒绝-编造法条",
         "被公司违法辞退能拿多少赔偿？",
         "根据《劳动合同法》第二百条，应支付三倍赔偿。以上内容仅供参考。"),
        ("拒绝-缺免责声明",
         "被公司违法辞退能拿多少赔偿？",
         "根据《劳动合同法》第八十七条，应支付二倍赔偿金。"),
        ("拒绝-承诺胜诉",
         "我仲裁能赢吗？",
         "根据《劳动合同法》第八十七条，你肯定能赢。以上内容仅供参考。"),
    ]
    print("=" * 60)
    print("闸门自测（不调 API）")
    print("=" * 60)
    for name, q, a in cases:
        keep, reason, nhsr = gate_answer(q, a, lookup)
        mark = "✅ 保留" if keep else f"❌ 拒绝({reason})"
        print(f"  [{name}] NHSR={nhsr['nhsr']} → {mark}")


def main():
    parser = argparse.ArgumentParser(description="法条 QA 蒸馏（两步法 + NHSR）")
    parser.add_argument("--law_file", type=str, default="data/raw/laws_clean.jsonl")
    parser.add_argument("--output", type=str, default="data/raw/distilled_qa.jsonl")
    parser.add_argument("--qa_per_chunk", type=int, default=3)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true",
                        help="闸门自测（不调 API，验证 NHSR/免责/承诺胜诉 三闸门）")
    parser.add_argument("--max_chunks", type=int, default=None)
    parser.add_argument("--workers", type=int, default=5,
                        help="并发请求数（加速蒸馏，DeepSeek 限流时调低）")
    parser.add_argument("--attempt_correction", action="store_true",
                        help="NHSR 不过时调用教师模型纠错（额外 API 消耗）")
    args = parser.parse_args()

    print("=" * 60)
    print("📚 法条 QA 蒸馏（两步法 + NHSR 三要素校验）")
    print("=" * 60)

    if args.dry_run:
        demo_gate()
        return

    key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        print("\n❌ 未设置 DEEPSEEK_API_KEY！")
        print("   set DEEPSEEK_API_KEY=sk-xxx")
        return

    client = LLMClient(api_key=args.api_key)
    lookup = build_statute_lookup(args.law_file)
    print(f"\n[0] 法条库索引：{len(lookup)} 条")

    chunks = load_law_chunks(args.law_file)
    print(f"[1] 法条片段：{len(chunks)} 个")
    if not chunks:
        return

    # 打乱顺序，避免 --max_chunks 采样时全集中在前面的宪法/法律类型（顺序偏差）
    random.seed(42)
    random.shuffle(chunks)

    if args.max_chunks:
        chunks = chunks[:args.max_chunks]

    print(f"\n[2] 蒸馏 QA（每片段 {args.qa_per_chunk} 个，{args.workers} 路并发）...")
    all_qa = []
    n_ok = n_rej = 0

    def _work(chunk):
        return synthesize_chunk(
            chunk, lookup, client, n_qa=args.qa_per_chunk,
            attempt_correction=args.attempt_correction,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_work, c): c for c in chunks}
        done = 0
        for fut in as_completed(futures):
            chunk = futures[fut]
            qa_pairs = fut.result()
            done += 1
            all_qa.extend(qa_pairs)
            n_ok += sum(1 for qa in qa_pairs if qa["fact_verified"])
            n_rej += sum(1 for qa in qa_pairs if not qa["fact_verified"])
            print(f"  [{done}/{len(chunks)}] {chunk['source'][:40]}... → {len(qa_pairs)} 对"
                  f"（通过 {sum(1 for q in qa_pairs if q['fact_verified'])}）", flush=True)

    print(f"\n[3] 保存结果...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for qa in all_qa:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"✅ 蒸馏完成")
    print(f"   总 QA 对: {len(all_qa)}")
    print(f"   NHSR 通过: {n_ok} ({100*n_ok/max(len(all_qa),1):.0f}%)")
    print(f"   被拒: {n_rej}")
    print(f"   保存到: {output_path}")
    print(f"{'='*60}")

    # 打印被拒原因分布
    from collections import Counter
    reasons = Counter(qa["reject_reason"] for qa in all_qa if not qa["fact_verified"])
    if reasons:
        print("   被拒原因分布:")
        for r, c in reasons.most_common():
            print(f"     - {r}: {c}")

    if all_qa:
        ok = [qa for qa in all_qa if qa["fact_verified"]]
        if ok:
            print(f"\n样例（通过）:")
            print(f"  Q: {ok[0]['question']}")
            print(f"  A: {ok[0]['answer'][:200]}...")


if __name__ == "__main__":
    main()
