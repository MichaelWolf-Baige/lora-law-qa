"""
02_curate_data.py — 法律数据清洗、去重、混配、切分。

Pipeline:
  1. 读取 data/raw/ 下的 DISC Pair-QA / Triplet-QA / 蒸馏 QA
  2. 去重（精确 + 近重复）
  3. 质量过滤（长度、中文字符、法条相关度）
  4. 按法律类别分类（劳动/合同/婚姻/刑事/公司/知产/行政）
  5. 分层切分为 train/eval/test，输出 ChatML（messages）格式

用法：
    python scripts/02_curate_data.py [--max_samples 4000]
"""

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.domain_config import get_domain
from app.data_quality import (
    exact_dedup, minhash_dedup, semantic_dedup,
    normalize_text, text_hash, estimate_difficulty, difficulty_bucket,
    extract_statute_citations,
)

# ──────────────────────────────────────────────
# 字段提取（兼容多种来源格式）
# ──────────────────────────────────────────────

def extract_text_fields(sample: dict) -> tuple[str, str, list]:
    """提取 (question, answer, reference) 从不同格式。"""
    q = a = ""
    ref = []

    if "messages" in sample:
        for m in sample["messages"]:
            if m.get("role") == "user":
                q = m.get("content", "")
            elif m.get("role") == "assistant":
                a = m.get("content", "")
    elif "instruction" in sample and "output" in sample:
        q = sample["instruction"]
        a = sample["output"]
        if sample.get("input"):
            q = q + " " + sample["input"]
    elif "question" in sample and "answer" in sample:
        q = sample["question"]
        a = sample["answer"]
    elif "query" in sample and "response" in sample:
        q = sample["query"]
        a = sample["response"]
    elif "input" in sample and "output" in sample:
        q = sample["input"]
        a = sample["output"]

    r = sample.get("reference", [])
    ref = r if isinstance(r, list) else ([r] if r else [])

    return (q or "").strip(), (a or "").strip(), ref


def compute_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_embed_fn():
    """语义去重的 embedding 函数（可选）。

    默认返回 None → 跳过语义去重（MinHash 已覆盖近重复，足够 4–5K 规模）。
    要启用语义去重：pip install sentence-transformers 后，把此函数改为返回
    `SentenceTransformer("BAAI/bge-small-zh-v1.5").encode`（会联网下载权重）。
    """
    return None


def deduplicate(samples: list[dict]) -> list[dict]:
    """
    三连去重：精确规范化 → MinHash 近重复 → 语义去重（可选 embedding）。

    先转成统一 {question, answer, reference, source} 结构再按 (question+answer) 去重。
    """
    normalized = []
    for s in samples:
        q, a, ref = extract_text_fields(s)
        if not q or not a:
            continue
        normalized.append({
            "question": q, "answer": a,
            "reference": ref,
            "source": s.get("source", ""),
        })

    def _key(x):
        return x["question"] + " " + x["answer"]

    # 1) 精确去重（规范化后哈希）
    u1 = exact_dedup(normalized, _key)
    # 2) MinHash 近重复去重（5-gram，相似度 ~0.8）
    u2 = minhash_dedup(u1, _key, threshold=0.8)
    # 3) 语义去重（可选，无 embedding 则跳过）
    u3 = semantic_dedup(u2, _key, embed_fn=get_embed_fn())
    return u3


def quality_filter(samples: list[dict], min_answer_len: int = 20, max_answer_len: int = 1500) -> list[dict]:
    """质量过滤。"""
    filtered = []
    for s in samples:
        q, a = s["question"], s["answer"]
        if len(q) < 4 or len(q) > 500:
            continue
        if len(a) < min_answer_len or len(a) > max_answer_len:
            continue
        if len(re.findall(r"[一-鿿]", a)) < 10:
            continue
        if len(re.findall(r"[一-鿿]", q)) < 2:
            continue
        if len(set(a)) < len(a) * 0.3:  # 过多重复字符 → 噪声
            continue
        filtered.append(s)
    return filtered


def classify_category(question: str, answer: str) -> str:
    """按法律类别分类（复用领域配置）。"""
    return get_domain().classify_category(question + " " + answer)


def diversity_sample(samples: list[dict], max_samples: int, seed: int = 42) -> list[dict]:
    """
    多样性采样（取代均匀随机 downsample）：
      1. 按类别均衡分配配额（保底 + 按类别大小比例）
      2. 类别内按难度分层（easy/medium/hard ≈ 3:4:3）
    """
    rng = np.random.RandomState(seed)
    for s in samples:
        s["category"] = classify_category(s["question"], s["answer"])
        s["difficulty"] = estimate_difficulty(s["question"], s["answer"])
        s["diff_bucket"] = difficulty_bucket(s["difficulty"])

    groups: dict[str, list] = {}
    for s in samples:
        groups.setdefault(s["category"], []).append(s)

    if len(samples) <= max_samples:
        return samples

    # 每类保底 + 按类别大小比例分配剩余
    min_per_cat = max(1, max_samples // (len(groups) * 4))
    alloc: dict[str, int] = {cat: min_per_cat for cat in groups}
    remaining = max_samples - sum(alloc.values())
    total = sum(len(g) for g in groups.values())
    for cat, g in sorted(groups.items(), key=lambda x: -len(x[1])):
        if remaining <= 0:
            break
        extra = int(remaining * len(g) / total)
        alloc[cat] += extra
        remaining -= extra

    selected: list[dict] = []
    for cat, g in groups.items():
        n = min(alloc.get(cat, 0), len(g))
        buckets = {"easy": [], "medium": [], "hard": []}
        for s in g:
            buckets[s["diff_bucket"]].append(s)
        # 3:4:3 难度分层
        target = {"easy": int(n * 0.3), "medium": int(n * 0.4), "hard": int(n * 0.3)}
        for b, arr in buckets.items():
            rng.shuffle(arr)
            selected.extend(arr[:target[b]])

    # 不足则从剩余补，超出则截断
    if len(selected) < max_samples:
        used = {id(s) for s in selected}
        pool = [s for s in samples if id(s) not in used]
        rng.shuffle(pool)
        selected.extend(pool[:max_samples - len(selected)])
    selected = selected[:max_samples]
    rng.shuffle(selected)
    return selected


def stratified_split(samples: list[dict], train_ratio: float = 0.7, eval_ratio: float = 0.15) -> dict:
    """按类别分层切分。"""
    for s in samples:
        s["category"] = classify_category(s["question"], s["answer"])
        s["difficulty"] = estimate_difficulty(s["question"], s["answer"])

    groups = {}
    for s in samples:
        groups.setdefault(s["category"], []).append(s)

    train, eval_set, test = [], [], []
    for cat, group in groups.items():
        group.sort(key=lambda x: x["difficulty"])
        n = len(group)
        n_train = int(n * train_ratio)
        n_eval = int(n * eval_ratio)
        train.extend(group[:n_train])
        eval_set.extend(group[n_train:n_train + n_eval])
        test.extend(group[n_train + n_eval:])

    rng = np.random.RandomState(42)
    for split in [train, eval_set, test]:
        rng.shuffle(split)

    return {"train": train, "eval": eval_set, "test": test}


def to_messages(sample: dict) -> dict:
    """转成 ChatML（messages）格式，嵌入法律 system prompt。"""
    sys_prompt = get_domain().default_system_prompt
    return {
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]},
        ],
        "metadata": {
            "source": sample.get("source", ""),
            "category": sample.get("category", "general"),
            "reference": sample.get("reference", []),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Curate legal QA dataset")
    parser.add_argument("--max_samples", type=int, default=4000, help="Max total samples to keep")
    parser.add_argument("--min_answer_len", type=int, default=20)
    parser.add_argument("--max_answer_len", type=int, default=1500)
    args = parser.parse_args()

    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    test_cases_dir = Path("data/test_cases")
    processed_dir.mkdir(parents=True, exist_ok=True)
    test_cases_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 1: 读取原始数据")
    print("=" * 60)
    raw_samples = []
    for jsonl_file in sorted(raw_dir.glob("*.jsonl")):
        # laws.jsonl / laws_clean.jsonl 是 RAG 法条语料，不是 SFT 数据
        if jsonl_file.name in ("laws.jsonl", "laws_clean.jsonl"):
            continue
        # disc_law_triplet_qa.jsonl 的 question 字段是「法条原文上下文」（非自然语言问题），
        # 它是 RAFT 式 grounded 数据，应走 04b_build_raft_data.py 或单独转换，不能当普通 QA 混入 SFT。
        if jsonl_file.name == "disc_law_triplet_qa.jsonl":
            print(f"  Skip {jsonl_file.name}（RAFT grounded 数据，非普通 QA）")
            continue
        print(f"  Loading {jsonl_file.name}...")
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    print(f"  Total raw samples: {len(raw_samples)}")

    print("\nSTEP 2: 去重")
    samples = deduplicate(raw_samples)
    print(f"  After dedup: {len(samples)} (removed {len(raw_samples) - len(samples)})")

    print("\nSTEP 3: 质量过滤")
    samples = quality_filter(samples, args.min_answer_len, args.max_answer_len)
    print(f"  After quality filter: {len(samples)}")

    if len(samples) > args.max_samples:
        samples = diversity_sample(samples, args.max_samples)
        print(f"  多样性采样到: {len(samples)}（类别均衡 + 难度 3:4:3）")

    print("\nSTEP 4: 分层切分")
    splits = stratified_split(samples)
    print(f"  Train: {len(splits['train'])} | Eval: {len(splits['eval'])} | Test: {len(splits['test'])}")

    for split_name in ["train", "eval", "test"]:
        cat_counts = Counter(s["category"] for s in splits[split_name])
        print(f"\n  {split_name} 类别分布:")
        for cat, count in cat_counts.most_common():
            print(f"    {cat}: {count} ({100*count/len(splits[split_name]):.0f}%)")

    print("\nSTEP 5: 保存 messages 格式（ChatML）")
    for split_name, split_data in splits.items():
        out = processed_dir / f"{split_name}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for s in split_data:
                f.write(json.dumps(to_messages(s), ensure_ascii=False) + "\n")
        print(f"  Saved {len(split_data)} → {out}")

    print("\nSTEP 6: 生成分类测试用例")
    all_test = []
    for cat in get_domain().categories:
        cat_samples = [s for s in splits["test"] if s.get("category") == cat]
        if not cat_samples:
            cat_samples = [s for s in samples if s.get("category") == cat][:10]
        records = [{
            "question": s["question"],
            "answer": s["answer"],
            "department": s.get("category", cat),
            "difficulty": s.get("difficulty", 0.5),
        } for s in cat_samples[:15]]
        with open(test_cases_dir / f"{cat}.json", "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        all_test.extend(records)
        print(f"  {cat}: {len(records)} test cases")

    with open(test_cases_dir / "all_departments.json", "w", encoding="utf-8") as f:
        json.dump(all_test, f, ensure_ascii=False, indent=2)

    print(f"\n  总测试用例: {len(all_test)}")
    print("\n✅ 数据整理完成！")
    print("   下一步: python scripts/15_quality_filter.py（法条真实性 + 免责声明校验）")


if __name__ == "__main__":
    main()
