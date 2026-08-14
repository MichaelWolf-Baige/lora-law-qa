"""10_assemble_train_data.py — 组装重训数据（弃 DISC，用蒸馏+RAFT+BELLE）。

组合：
  1. 蒸馏 QA（fact_verified，1623）— 教「引用真实法条」+ 领域
  2. RAFT grounded（1324）— 教「读检索法条→引用」+ 拒答
  3. BELLE 通用（600）— 防遗忘

输出 messages 格式到 data/processed/{train,eval,test}.jsonl。
"""
import sys, json, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.domain_config import get_domain
from app.data_quality import text_hash, normalize_text

SYS = get_domain().default_system_prompt


def load_distilled(path):
    """蒸馏 QA → messages（只取 fact_verified）。"""
    out = []
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        if not d.get("fact_verified"):
            continue
        out.append({
            "messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": d["question"]},
                {"role": "assistant", "content": d["answer"]},
            ],
            "metadata": {"source": "distilled", "category": "legal"},
        })
    return out


def load_raft(path):
    """RAFT → messages（instruction + context + question）。"""
    out = []
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        user = f"{d['instruction']}\n{d['context']}\n\n问题：{d['question']}"
        out.append({
            "messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": user},
                {"role": "assistant", "content": d["answer"]},
            ],
            "metadata": {"source": "raft", "category": "legal-grounded"},
        })
    return out


def load_belle(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def dedup(samples):
    seen, kept = set(), []
    for s in samples:
        q = next((m["content"] for m in s["messages"] if m["role"] == "user"), "")
        a = next((m["content"] for m in s["messages"] if m["role"] == "assistant"), "")
        h = text_hash(normalize_text(q + a))
        if h in seen:
            continue
        seen.add(h)
        kept.append(s)
    return kept


def main():
    distilled = load_distilled("data/raw/distilled_qa.jsonl")
    raft = load_raft("data/processed/raft_train.jsonl")
    belle = load_belle("data/raw/belle_general.jsonl")
    print(f"蒸馏 {len(distilled)} + RAFT {len(raft)} + BELLE {len(belle)}")

    all_s = dedup(distilled + raft + belle)
    print(f"去重后: {len(all_s)}")

    random.seed(42)
    random.shuffle(all_s)
    n = len(all_s)
    n_train = int(n * 0.8)
    n_eval = int(n * 0.1)
    splits = {
        "train": all_s[:n_train],
        "eval": all_s[n_train:n_train + n_eval],
        "test": all_s[n_train + n_eval:],
    }
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        with open(f"data/processed/{name}.jsonl", "w", encoding="utf-8") as f:
            for s in data:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(data)}")
    print("✅ 重训数据组装完成")


if __name__ == "__main__":
    main()
