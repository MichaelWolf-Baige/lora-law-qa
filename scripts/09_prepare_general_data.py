"""09_prepare_general_data.py — 采样通用中文指令数据（BELLE）用于防遗忘数据混合。

BELLE train_0.5M_CN（51.9 万条中文指令）→ 采样 ~600 条多样本，
转成 messages 格式（system/user/assistant），供重训时与 RAFT/DISC 混合。

用法：
    python scripts/09_prepare_general_data.py [--n 600]
"""
import sys, json, random, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

GENERAL_SYSTEM = "你是一个乐于助人、知识渊博的中文助手。请直接、准确地回答用户的问题或完成用户的任务。"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--output", type=str, default="data/raw/belle_general.jsonl")
    args = ap.parse_args()

    from datasets import load_dataset
    print("加载 BELLE train_0.5M_CN...")
    ds = load_dataset("BelleGroup/train_0.5M_CN", split="train")

    # 过滤：output 长度适中、去掉空的
    def keep(x):
        out = (x.get("output") or "").strip()
        instr = (x.get("instruction") or "").strip()
        return 20 <= len(out) <= 1500 and len(instr) >= 3

    ds = ds.filter(keep)
    print(f"过滤后: {len(ds)} 条")

    # 采样（固定 seed 可复现）
    random.seed(42)
    idxs = sorted(random.sample(range(len(ds)), min(args.n, len(ds))))
    sampled = ds.select(idxs)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for x in sampled:
            instr = (x.get("instruction") or "").strip()
            inp = (x.get("input") or "").strip()
            out = (x.get("output") or "").strip()
            user = instr + ("\n" + inp if inp else "")
            f.write(json.dumps({
                "messages": [
                    {"role": "system", "content": GENERAL_SYSTEM},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": out},
                ],
                "metadata": {"source": "belle-general", "category": "general"},
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"✅ 保存 {n} 条通用指令 → {out_path}")


if __name__ == "__main__":
    main()
