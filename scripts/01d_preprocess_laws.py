"""
01d_preprocess_laws.py — 法律条文库预处理（清洗 + 去重 + 过滤）。

审计发现的 5 类问题，一次清干净：
  1. 历史版本（已修改/已废止/status异常）→ 只保留「有效」
  2. 同名多版本 → 同名保留最新公布日期的有效版
  3. 0 条文（修改决定/附件/决定类）→ 过滤
  4. Markdown 格式（blockquote > / 目录 / 全角空格）→ 清理为纯文本
  5. 乱码 → 过滤

scope 控制范围：
  --scope national  默认：只保留全国性法规（宪法/法律/行政法规/司法解释/法律解释/监察法规）
  --scope all       保留全部（含地方法规）

输出：data/raw/laws_clean.jsonl（干净、去重、有效、纯文本）

用法：
    python scripts/01d_preprocess_laws.py
    python scripts/01d_preprocess_laws.py --scope all
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.data_quality import split_articles

NATIONAL_TYPES = ("宪法", "法律", "行政法规", "司法解释", "法律解释", "监察法规")


def clean_content(content: str) -> str:
    """Markdown → 纯文本：去 blockquote、去目录、规范化空白。"""
    lines = []
    in_toc = False
    for raw in content.splitlines():
        line = raw.strip()
        # 去 markdown blockquote 前缀（修订说明）
        line = re.sub(r"^>\s*", "", line)
        # 去 markdown 链接/图片
        line = re.sub(r"!\[.*?\]\(.*?\)", "", line)
        line = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", line)
        # 目录段：从「目录」到第一个「第X章」之间的行跳过
        if "目" in line and "录" in line and len(line) <= 4:
            in_toc = True
            continue
        if in_toc and re.match(r"^\s*第[一二三四五六七八九十百千0-9]+章", line):
            in_toc = False
        if in_toc:
            continue
        # 全角空格 → 半角空格（保留缩进语义但统一）
        line = line.replace("　", " ")
        # pandoc 转义反转义：\< → 〈（单书名号）等
        line = line.replace("\\<", "〈").replace("\\>", "〉")
        line = line.replace("\\[", "[").replace("\\]", "]")
        line = line.replace("\\*", "*").replace("\\_", "_")
        if line:
            lines.append(line)
    return "\n".join(lines)


def preprocess(law_file: str, out_file: str, scope: str = "national"):
    laws = [json.loads(l) for l in open(law_file, encoding="utf-8") if l.strip()]
    print("=" * 60)
    print("法律条文库预处理")
    print("=" * 60)
    print(f"[0] 原始: {len(laws)} 部")

    # 1. scope 过滤
    if scope == "national":
        laws = [l for l in laws if l.get("type", "") in NATIONAL_TYPES]
        print(f"[1] 范围(national): {len(laws)} 部（剔除地方法规）")

    # 2. 只保留「有效」
    laws = [l for l in laws if l.get("status") == "有效"]
    print(f"[2] 只保留有效: {len(laws)} 部")

    # 3. 同名去重，保留最新公布日期
    by_title = defaultdict(list)
    for l in laws:
        by_title[l["title"]].append(l)
    deduped = []
    for title, versions in by_title.items():
        versions.sort(key=lambda x: x.get("publish_date", ""), reverse=True)
        deduped.append(versions[0])
    print(f"[3] 同名去重(保留最新): {len(deduped)} 部（去除 {len(laws)-len(deduped)} 个历史版本）")

    # 4. 清理格式 + 过滤空/乱码（不按「有无第X条」过滤——
    #    批复/规定/决定类司法解释用「一二三」或散文结构，仍是有价值内容）
    cleaned = []
    dropped_empty = 0
    dropped_mojibake = 0
    for l in deduped:
        content = clean_content(l.get("content", ""))
        if "�" in content:
            dropped_mojibake += 1
            continue
        if len(content) < 30:  # 真正空/极短才过滤
            dropped_empty += 1
            continue
        l["content"] = content
        cleaned.append(l)
    print(f"[4] 清理格式 + 过滤空/乱码: {len(cleaned)} 部"
          f"（空 {dropped_empty}，乱码 {dropped_mojibake}）")

    # 5. 输出
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for l in cleaned:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")

    from collections import Counter
    tc = Counter(l["type"] for l in cleaned)
    total_articles = sum(len(split_articles(l["content"])) for l in cleaned)
    print(f"\n{'='*60}")
    print(f"✅ 预处理完成: {len(cleaned)} 部 → {out}")
    print(f"   总条文: {total_articles}")
    print("   类型分布:")
    for t, c in tc.most_common():
        print(f"     {t}: {c}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="法律条文库预处理")
    parser.add_argument("--input", type=str, default="data/raw/laws.jsonl")
    parser.add_argument("--output", type=str, default="data/raw/laws_clean.jsonl")
    parser.add_argument("--scope", type=str, default="national", choices=["national", "all"])
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"⚠ 输入不存在: {args.input}（先跑 01c_download_full_corpus.py）")
        return
    preprocess(args.input, args.output, args.scope)


if __name__ == "__main__":
    main()
