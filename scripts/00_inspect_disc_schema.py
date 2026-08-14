"""
00_inspect_disc_schema.py — 抽样确认 DISC-Law-SFT 字段结构。

落地第一步（调研确认的最大未知数）：确认 DISC Triplet-QA 的 reference 字段
是否含具体法条号，这决定「引用溯源」能否直接在其上改造，还是必须自建字段。

用法：
    python scripts/00_inspect_disc_schema.py
    python scripts/00_inspect_disc_schema.py --repo ShengbinYue/DISC-Law-SFT
    HF_ENDPOINT=https://hf-mirror.com python scripts/00_inspect_disc_schema.py
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.data_quality import extract_statute_citations


def inspect_item(item: dict, label: str):
    print(f"\n--- {label} ---")
    print(f"  keys: {sorted(item.keys())}")
    for k, v in item.items():
        if isinstance(v, str):
            preview = v[:200].replace("\n", " ")
            print(f"  [{k}] ({len(v)} chars) = {preview}")
        elif isinstance(v, list):
            print(f"  [{k}] (list, {len(v)} items)")
            for i, e in enumerate(v[:3]):
                if isinstance(e, dict):
                    print(f"      [{i}] keys={sorted(e.keys())}")
                    for kk, vv in list(e.items())[:5]:
                        s = str(vv)[:120].replace("\n", " ")
                        print(f"          {kk}: {s}")
                else:
                    print(f"      [{i}] {str(e)[:120]}")
        else:
            print(f"  [{k}] = {v}")
    # 是否含法条引用
    text = json.dumps(item, ensure_ascii=False)
    cites = extract_statute_citations(text)
    print(f"  法条引用: {len(cites)} 处")
    for c in cites[:5]:
        print(f"    - {c['law']} 第{c['article']}条")


def main():
    parser = argparse.ArgumentParser(description="抽样检查 DISC-Law-SFT schema")
    parser.add_argument("--repo", type=str, default="ShengbinYue/DISC-Law-SFT")
    parser.add_argument("--n", type=int, default=3, help="每个文件抽样条数")
    args = parser.parse_args()

    from datasets import load_dataset

    files = {
        "Pair-QA": "DISC-Law-SFT-Pair-QA-released.jsonl",
        "Triplet-QA": "DISC-Law-SFT-Triplet-QA-released.jsonl",
    }

    for label, fname in files.items():
        print("=" * 60)
        print(f"检查 {label}: {fname}")
        print("=" * 60)
        try:
            ds = load_dataset(args.repo, data_files=fname, split="train")
            print(f"  总条数: {len(ds)}")
            for i in range(min(args.n, len(ds))):
                inspect_item(ds[i], f"sample {i}")
        except Exception as e:
            print(f"  ⚠ 加载失败: {e}")
            print("    （可能是 HF 被墙，试 HF_ENDPOINT=https://hf-mirror.com）")


if __name__ == "__main__":
    main()
