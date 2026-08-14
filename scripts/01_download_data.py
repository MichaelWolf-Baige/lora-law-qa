"""
01_download_data.py — 下载法律 SFT 数据 + 法条语料（商用安全版）。

下载内容（全部 Apache-2.0 或官方文件，可商用）：
  1. DISC-Law-SFT Pair-QA    —— 法律问答（8 子领域广度），~93K 条
  2. DISC-Law-SFT Triplet-QA —— 带法条引用的法律问答，~23K 条（教「法条→回答」）
  3. chinese-law-and-regulations —— 法条 + 司法解释语料（RAG 用），~22.5K 条

关键坑（已处理）：
  - DISC-Law-SFT 不能整体 load_dataset（Pair/Triplet 列不同会报 schema 错），
    必须用 data_files= 按单文件加载。
  - DISC 无 ModelScope 镜像，HuggingFace 被墙时设置环境变量 HF_ENDPOINT=https://hf-mirror.com

用法：
    python scripts/01_download_data.py
    python scripts/01_download_data.py --max_pair 2500 --max_triplet 500 --skip_laws
    HF_ENDPOINT=https://hf-mirror.com python scripts/01_download_data.py
"""

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

RAW_DIR = Path("data/raw")

# DISC-Law-SFT 仓库（Apache-2.0）
DISC_REPO = "ShengbinYue/DISC-Law-SFT"
DISC_PAIR_QA_FILE = "DISC-Law-SFT-Pair-QA-released.jsonl"       # 法律问答 ~93K
DISC_TRIPLET_QA_FILE = "DISC-Law-SFT-Triplet-QA-released.jsonl"  # 带法条引用 QA ~23K

# 法条语料（Apache-2.0，含 788 司法解释 + 26 法律解释 + 429 法律 + 693 行政法规）
LAWS_REPO = "twang2218/chinese-law-and-regulations"


def _extract_disc_qa(item: dict, with_reference: bool) -> dict:
    """DISC 条目 → 统一 {'question','answer',['reference'],'source'} 格式。"""
    q = (item.get("input") or "").strip()
    a = (item.get("output") or "").strip()
    if not q or not a:
        return None
    record = {"question": q, "answer": a, "source": "disc-triplet-qa" if with_reference else "disc-pair-qa"}
    if with_reference:
        ref = item.get("reference")
        record["reference"] = ref if isinstance(ref, list) else ([ref] if ref else [])
    return record


def download_disc(raw_dir: Path, max_pair: int, max_triplet: int) -> None:
    """下载 DISC-Law-SFT 两个 QA 文件。"""
    print("=" * 60)
    print("下载 DISC-Law-SFT（Apache-2.0）")
    print("=" * 60)

    if max_pair > 0:
        print(f"\n[1/2] Pair-QA（法律问答，目标 {max_pair} 条）...")
        ds = load_dataset(DISC_REPO, data_files=DISC_PAIR_QA_FILE, split="train")
        out = raw_dir / "disc_law_pair_qa.jsonl"
        count = 0
        with open(out, "w", encoding="utf-8") as f:
            for item in tqdm(ds, total=min(max_pair, len(ds)), desc="  pair-qa"):
                rec = _extract_disc_qa(item, with_reference=False)
                if rec is None:
                    continue
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
                if count >= max_pair:
                    break
        print(f"  ✅ 保存 {count} 条 → {out}")

    if max_triplet > 0:
        print(f"\n[2/2] Triplet-QA（带法条引用，目标 {max_triplet} 条）...")
        ds = load_dataset(DISC_REPO, data_files=DISC_TRIPLET_QA_FILE, split="train")
        out = raw_dir / "disc_law_triplet_qa.jsonl"
        count = 0
        with open(out, "w", encoding="utf-8") as f:
            for item in tqdm(ds, total=min(max_triplet, len(ds)), desc="  triplet-qa"):
                rec = _extract_disc_qa(item, with_reference=True)
                if rec is None:
                    continue
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
                if count >= max_triplet:
                    break
        print(f"  ✅ 保存 {count} 条 → {out}")


def download_laws(raw_dir: Path) -> None:
    """下载法条 + 司法解释语料（RAG 用）。"""
    print("\n" + "=" * 60)
    print("下载法条语料 chinese-law-and-regulations（Apache-2.0）")
    print("=" * 60)

    ds = load_dataset(LAWS_REPO, split="train")
    out = raw_dir / "laws.jsonl"
    count = 0
    with open(out, "w", encoding="utf-8") as f:
        for item in tqdm(ds, desc="  laws"):
            record = {
                "title": item.get("title", ""),
                "type": item.get("type", ""),
                "publish_date": item.get("publish_date", ""),
                "effective_date": item.get("effective_date", ""),
                "status": item.get("status", ""),
                "content": item.get("content", ""),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(f"  ✅ 保存 {count} 条 → {out}")


def main():
    parser = argparse.ArgumentParser(description="下载法律 SFT 数据 + 法条语料")
    parser.add_argument("--max_pair", type=int, default=2500, help="DISC Pair-QA 下载条数（0 跳过）")
    parser.add_argument("--max_triplet", type=int, default=500, help="DISC Triplet-QA 下载条数（0 跳过）")
    parser.add_argument("--skip_laws", action="store_true", help="跳过法条语料下载")
    args = parser.parse_args()

    raw_dir = RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    mirror = os.environ.get("HF_ENDPOINT", "")
    if mirror:
        print(f"⚠ 使用 HuggingFace 镜像: {mirror}")

    download_disc(raw_dir, args.max_pair, args.max_triplet)
    if not args.skip_laws:
        download_laws(raw_dir)

    print("\n[OK] 数据下载完成")
    print(f"   Raw 目录: {raw_dir.resolve()}")
    print("   下一步: python scripts/14_distill_guidelines.py（法条→带引用QA 蒸馏，可选）")
    print("   然后:   python scripts/02_curate_data.py（去重/清洗/切分）")


if __name__ == "__main__":
    main()
