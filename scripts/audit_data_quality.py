"""
audit_data_quality.py — 法律条文库质量审计。

对 data/raw/laws.jsonl 做多维质量检查：
  完整性（空值/短文本/重复）、内容格式（HTML/乱码）、条文解析、
  类型分布、时效性、核心法存在性，并对比官方库实时统计定位遗漏。

用法：
    python scripts/audit_data_quality.py
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.data_quality import split_articles

LAW_FILE = "data/raw/laws.jsonl"

# 官方库 flk.npc.gov.cn aggregateData 的实时统计（手动更新）
OFFICIAL_COUNTS = {"宪法": 1, "法律": 310, "行政法规": 608,
                   "监察法规": 2, "地方法规": 15843, "司法解释": 561}

CORE_LAWS = ["中华人民共和国民法典", "中华人民共和国刑法", "中华人民共和国劳动合同法",
             "中华人民共和国劳动法", "中华人民共和国公司法", "中华人民共和国商标法",
             "中华人民共和国专利法", "中华人民共和国著作权法", "中华人民共和国行政处罚法",
             "中华人民共和国行政复议法", "中华人民共和国行政诉讼法",
             "中华人民共和国刑事诉讼法", "中华人民共和国治安管理处罚法"]


def load():
    return [json.loads(l) for l in open(LAW_FILE, encoding="utf-8") if l.strip()]


def audit():
    laws = load()
    n = len(laws)
    print("=" * 60)
    print("法律条文库质量审计报告")
    print("=" * 60)

    # 1. 完整性
    empty_title = sum(1 for l in laws if not (l.get("title") or "").strip())
    empty_content = sum(1 for l in laws if not (l.get("content") or "").strip())
    short = sum(1 for l in laws if len(l.get("content", "")) < 50)
    titles = [l["title"].strip() for l in laws]
    dup_title = n - len(set(titles))
    print(f"\n[1 完整性] 总记录 {n}")
    print(f"   空标题 {empty_title} | 空正文 {empty_content} | 正文<50字 {short}")
    print(f"   同名多版本法律 {len(set(t for t,c in Counter(titles).items() if c>1))} 部"
          f"（历史版本，共 {dup_title} 条冗余）")

    # 2. 格式
    html = sum(1 for l in laws if re.search(r"<[a-z]+[ >]", l["content"]))
    moji = sum(1 for l in laws if "�" in l["content"])
    print(f"\n[2 格式] HTML残留 {html} | 乱码 {moji}")

    # 3. 条文解析
    arts = [split_articles(l["content"]) for l in laws]
    total_articles = sum(len(a) for a in arts)
    zero = sum(1 for a in arts if len(a) == 0)
    print(f"\n[3 条文] 总条文 {total_articles} | 平均每部 {total_articles/n:.1f}")
    print(f"   0条文 {zero} 部（多为修改决定/附件，本身无『第X条』结构）")

    # 4. 状态
    print(f"\n[4 状态]")
    for s, c in Counter(l.get("status", "") for l in laws).most_common():
        print(f"   {s or '(空)'}: {c}")

    # 5. 类型
    print(f"\n[5 类型]")
    for t, c in Counter(l.get("type", "") for l in laws).most_common():
        print(f"   {t or '(空)'}: {c}")

    # 6. 时效
    ds = [l["publish_date"][:10] for l in laws if re.match(r"^\d{4}", l.get("publish_date", ""))]
    print(f"\n[6 时效] 有日期 {len(ds)}/{n} | 最早 {min(ds)} | 最新 {max(ds)}")

    # 7. 核心法存在性
    ts = set(titles)
    missing_core = [c for c in CORE_LAWS if c not in ts]
    print(f"\n[7 核心法] 缺失 {len(missing_core)}/{len(CORE_LAWS)}")
    for c in missing_core:
        print(f"   ❌ {c}")

    # 8. 有效子集 vs 官方实时
    valid = [l for l in laws if l.get("status") == "有效"]
    print(f"\n[8 有效子集 vs 官方实时统计]")
    vc = Counter(l["type"] for l in valid)
    print(f"   {'类型':<8}{'镜像有效':>8}{'官方实时':>8}{'遗漏':>8}")
    for t, official in [("法律", 310), ("行政法规", 608), ("司法解释", 561),
                        ("监察法规", 2), ("地方性法规", 15843)]:
        mirror = vc.get(t, 0)
        gap = official - mirror
        flag = "✅" if gap <= 0 else f"缺{gap}"
        print(f"   {t:<8}{mirror:>8}{official:>8}{flag:>8}")

    return laws


if __name__ == "__main__":
    audit()
