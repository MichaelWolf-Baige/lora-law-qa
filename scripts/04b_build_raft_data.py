"""
04b_build_raft_data.py — RAFT 式 RAG grounded 训练数据。

依据 RAFT（arXiv:2403.10131）：每条数据 = 问题 + 文档集（1 个 oracle 法条 chunk
+ 2–3 个干扰法条 chunk）+ 答案（引用真实条文）。P=0.8 含 oracle，20% 只有干扰
（或无检索）→ 训练「检索不到就拒答」，防止模型硬编法条。

输入：
  - data/raw/distilled_qa.jsonl   （14_distill 产出，已过 NHSR）
  - data/raw/laws.jsonl           （法条库，供切 chunk + 找 oracle + 抽干扰）

输出（RAFT 格式）：
  data/processed/raft_train.jsonl
  每条 {instruction, context, question, answer, has_oracle, oracle_law, oracle_article}

用法：
    python scripts/04b_build_raft_data.py
    python scripts/04b_build_raft_data.py --p_oracle 0.8 --n_distractors 3
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.data_quality import (
    StatuteLookup, build_statute_lookup, extract_statute_citations, split_articles,
)

RAFT_INSTRUCTION = (
    "你是一名中国法律咨询助手。根据下面检索到的法条内容回答用户问题：\n"
    "1. 必须引用检索结果中真实存在的条文（《法名》第X条），不得引用未提供的法条；\n"
    "2. 检索结果可能包含无关法条，请只引用与问题相关的条文；\n"
    "3. 若检索结果中没有相关法条，明确说明无法回答，并引导补充信息；\n"
    "4. 结尾附免责声明。"
)

REFUSAL_ANSWER = (
    "抱歉，当前检索结果中没有与该问题直接相关的法条，我无法给出准确的法律意见。\n"
    "建议您补充更多细节（如时间、主体、金额、是否已签订书面文件），或咨询执业律师。\n"
    "以上内容仅供参考，不构成法律意见。"
)


# ──────────────────────────────────────────────
# 法条库 → chunk（保留条号，供 oracle 匹配）
# ──────────────────────────────────────────────

def chunk_laws(law_file: str, chunk_size: int = 400,
               focus_types: tuple = None) -> list:
    """把法条库切成 chunk，每个 chunk 带其包含的条号集合。"""
    if focus_types is None:
        focus_types = ("法律", "司法解释", "法律解释", "行政法规", "宪法", "监察法规")
    chunks = []
    law_path = Path(law_file)
    if not law_path.exists():
        print(f"  ⚠ 法条文件不存在: {law_file}")
        return chunks
    with open(law_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if focus_types and item.get("type", "") not in focus_types:
                continue
            title = (item.get("title") or "").strip()
            content = item.get("content") or ""
            if not title or not content:
                continue
            # 按条切，再按长度聚合
            articles = split_articles(content)
            if not articles:
                continue
            current_text, current_arts = "", set()
            for num, text in articles:
                if current_text and len(current_text) + len(text) > chunk_size:
                    chunks.append({"title": title, "text": current_text.strip(),
                                   "articles": current_arts})
                    current_text, current_arts = text, {num}
                else:
                    current_text += text + "\n"
                    current_arts.add(num)
            if current_text.strip():
                chunks.append({"title": title, "text": current_text.strip(),
                               "articles": current_arts})
    return chunks


def match_oracle_chunks(qa: dict, chunks: list, lookup: StatuteLookup) -> list:
    """找出 QA 回答所引法条对应的 chunk（oracle）。"""
    cites = extract_statute_citations(qa.get("answer", ""))
    matched = []
    seen = set()
    for c in cites:
        full = lookup.resolve_law(c["law"])
        if not full:
            continue
        for i, ch in enumerate(chunks):
            if ch["title"] == full and c["article"] in ch["articles"]:
                if i not in seen:
                    matched.append(i)
                    seen.add(i)
                break
    return matched


# ──────────────────────────────────────────────
# 构建 RAFT 数据
# ──────────────────────────────────────────────

def format_context(chunks: list, max_chars: int = 1200) -> str:
    parts = ["【检索到的法条】"]
    total = 0
    for k, ch in enumerate(chunks, 1):
        header = f"[文档{k}] {ch['title']}：\n"
        body = ch["text"]
        if total + len(header) + len(body) > max_chars:
            remaining = max_chars - total - len(header)
            if remaining > 80:
                body = body[:remaining] + "..."
            else:
                break
        parts.append(header + body)
        total += len(header) + len(body)
    return "\n\n".join(parts)


def build_raft_records(distilled_qa: list, chunks: list, lookup: StatuteLookup,
                       p_oracle: float = 0.8, n_distractors: int = 2) -> list:
    records = []
    qa_pool = [q for q in distilled_qa if q.get("fact_verified") and q.get("answer")]

    for qa in qa_pool:
        oracle_idx = match_oracle_chunks(qa, chunks, lookup)
        if not oracle_idx:
            continue  # 无法定位 oracle 的丢弃
        oracle_i = oracle_idx[0]

        # 抽干扰 chunk（不同法的随机 chunk，教模型区分相关/无关法条）
        others = [i for i in range(len(chunks)) if chunks[i]["title"] != chunks[oracle_i]["title"]]
        distractor_idx = random.sample(others, min(n_distractors, len(others)))

        if random.random() < p_oracle:
            ctx_chunks = [chunks[oracle_i]] + [chunks[i] for i in distractor_idx]
            random.shuffle(ctx_chunks)  # oracle 位置随机，避免位置捷径
            records.append({
                "instruction": RAFT_INSTRUCTION,
                "context": format_context(ctx_chunks),
                "question": qa["question"],
                "answer": qa["answer"],
                "has_oracle": True,
                "oracle_law": chunks[oracle_i]["title"],
                "oracle_article": sorted(chunks[oracle_i]["articles"])[:5],
            })
        else:
            # 拒答负例：只给干扰/无相关法条
            records.append({
                "instruction": RAFT_INSTRUCTION,
                "context": format_context([chunks[i] for i in distractor_idx]),
                "question": qa["question"],
                "answer": REFUSAL_ANSWER,
                "has_oracle": False,
                "oracle_law": chunks[oracle_i]["title"],
            })
    return records


def main():
    parser = argparse.ArgumentParser(description="构建 RAFT RAG grounded 数据")
    parser.add_argument("--distilled", type=str, default="data/raw/distilled_qa.jsonl")
    parser.add_argument("--law_file", type=str, default="data/raw/laws_clean.jsonl")
    parser.add_argument("--output", type=str, default="data/processed/raft_train.jsonl")
    parser.add_argument("--p_oracle", type=float, default=0.8)
    parser.add_argument("--n_distractors", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 60)
    print("📐 构建 RAFT RAG grounded 训练数据")
    print("=" * 60)

    if not Path(args.distilled).exists():
        print(f"  ⚠ 蒸馏 QA 不存在: {args.distilled}")
        print("    先运行 python scripts/14_distill_guidelines.py")
        return
    if not Path(args.law_file).exists():
        print(f"  ⚠ 法条库不存在: {args.law_file}")
        return

    print("[1] 加载法条库并切 chunk...")
    chunks = chunk_laws(args.law_file)
    print(f"    {len(chunks)} 个 chunk")
    lookup = build_statute_lookup(args.law_file)

    print("[2] 加载蒸馏 QA...")
    with open(args.distilled, "r", encoding="utf-8") as f:
        distilled = [json.loads(line) for line in f if line.strip()]
    print(f"    {len(distilled)} 条（其中通过 {sum(1 for q in distilled if q.get('fact_verified'))} 条）")

    print(f"[3] 构建 RAFT 记录（P_oracle={args.p_oracle}，干扰 {args.n_distractors} 个）...")
    records = build_raft_records(distilled, chunks, lookup,
                                 p_oracle=args.p_oracle, n_distractors=args.n_distractors)
    n_oracle = sum(1 for r in records if r["has_oracle"])
    n_refusal = len(records) - n_oracle
    print(f"    {len(records)} 条（含 oracle {n_oracle} / 拒答 {n_refusal}）")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[4] 保存 → {out}")

    if records:
        r = records[0]
        print(f"\n样例（has_oracle={r['has_oracle']}）:")
        print(f"  instruction: {r['instruction'][:50]}...")
        print(f"  context: {r['context'][:120]}...")
        print(f"  question: {r['question'][:60]}")
        print(f"  answer: {r['answer'][:100]}...")

    print("\n✅ RAFT 数据构建完成")


if __name__ == "__main__":
    main()
