"""
01c_download_full_corpus.py — 下载全量法律条文库（22K+ 部）。

「绝对齐全」数据源：官方数据库的完整镜像（twang2218/law-datasets，GitHub LFS）。
该镜像从国家法律法规数据库（flk.npc.gov.cn）爬取，含宪法/法律/行政法规/监察法规/
司法解释/地方法规 全部条文正文，共 ~22.5K 部（2023-09 快照）。

为什么用镜像而非直连官方：官方搜索枚举接口已加验证码 + 旧 /api/ 已下线，
批量枚举被反爬封死；镜像是一次性抓全的完整快照。权威刷新可用
01b_download_official_laws.py 直连官方 flfgDetails 按需更新。

流程：下载 laws.json.zip（102MB，GitHub LFS）→ 解压 laws.json → 转 laws.jsonl
      （title/type/content/日期/状态）→ 按 type 统计分布。

用法：
    python scripts/01c_download_full_corpus.py                    # 全量
    python scripts/01c_download_full_corpus.py --skip_local_types # 剔除地方法规
    python scripts/01c_download_full_corpus.py --url <镜像URL>    # 自定义镜像
"""

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).parent.parent))

DEFAULT_URL = ("https://media.githubusercontent.com/media/twang2218/"
               "law-datasets/main/law-and-regulations/laws.json.zip")
ZIP_PATH = "data/raw/laws_full.zip"
OUT_PATH = "data/raw/laws.jsonl"

# 地方法规类目（全国性法律咨询通常可剔除，但「绝对齐全」默认保留）
LOCAL_TYPES = ("地方法规", "地方性法规", "地方政府规章", "自治条例", "单行条例")


def download(url: str, dest: str) -> None:
    print(f"下载 {url[:80]}...")
    with urlopen(url, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100 // total
                    print(f"\r  {done // (1024*1024)}/{total // (1024*1024)} MB ({pct}%)",
                          end="", flush=True)
        print()
    print(f"  完成 → {dest}")


def extract_json(zip_path: str, dest_json: str) -> None:
    print(f"解压 {zip_path}...")
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        print(f"  zip 内文件: {names[:5]}")
        target = next((n for n in names if n.endswith(".json")), names[0])
        data = z.read(target)
    with open(dest_json, "wb") as f:
        f.write(data)
    print(f"  完成 → {dest_json}（{len(data)//(1024*1024)} MB）")


def convert(src_json: str, dest_jsonl: str, skip_local: bool) -> None:
    print("转换 laws.json → laws.jsonl...")
    with open(src_json, "r", encoding="utf-8") as f:
        laws = json.load(f)
    print(f"  总条数: {len(laws)}")

    from collections import Counter
    type_counter = Counter()
    out = Path(dest_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for item in laws:
            typ = (item.get("type") or "").strip()
            type_counter[typ] += 1
            if skip_local and typ in LOCAL_TYPES:
                continue
            # 字段映射（镜像字段 → 标准 schema）
            rec = {
                "title": (item.get("title") or "").strip(),
                "type": typ,
                "publish_date": item.get("publish", "") or "",
                "effective_date": item.get("effective", item.get("sxrq", "")) or "",
                "status": item.get("status", "") or "",
                "content": item.get("content", "") or "",
                "source": "flk.npc.gov.cn (mirror)",
                "source_id": item.get("id", ""),
            }
            if not rec["title"] or not rec["content"]:
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"  写入 {n} 条 → {dest_jsonl}")
    print("\n类型分布（前 15）:")
    for typ, cnt in type_counter.most_common(15):
        print(f"  {typ or '(空)'}: {cnt}")


def main():
    parser = argparse.ArgumentParser(description="下载全量法律条文库")
    parser.add_argument("--url", type=str, default=DEFAULT_URL)
    parser.add_argument("--skip_local", action="store_true",
                        help="剔除地方法规/规章（全国性咨询用）")
    parser.add_argument("--zip_path", type=str, default=ZIP_PATH)
    parser.add_argument("--output", type=str, default=OUT_PATH)
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    src_json = zip_path.with_suffix("") / "laws.json"
    src_json.parent.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        download(args.url, str(zip_path))
    else:
        print(f"已存在 {zip_path}，跳过下载")

    # 解压到临时 json
    import tempfile, os
    tmpdir = tempfile.mkdtemp()
    tmp_json = os.path.join(tmpdir, "laws.json")
    extract_json(str(zip_path), tmp_json)

    convert(tmp_json, args.output, args.skip_local)
    print("\n✅ 全量法律条文库构建完成")
    print(f"   下一步: python scripts/14_distill_guidelines.py（蒸馏）或 04_build_rag.py（RAG 入库）")


if __name__ == "__main__":
    main()
