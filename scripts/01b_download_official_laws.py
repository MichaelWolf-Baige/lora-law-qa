"""
01b_download_official_laws.py — 从国家法律法规数据库（flk.npc.gov.cn）官方 API 下载法条全文。

「最权威」数据源：国家法律法规数据库（全国人大常委会/司法部建设），比 HF 镜像更权威、更新。

已实测可用的官方接口：
  - GET /law-search/index/aggregateData    → 统计 + 热门/新法（含 bbbs ID）
  - GET /law-search/search/enumData        → 分类树（宪法/法律/行政法规/司法解释 codeId）
  - GET /law-search/search/flfgDetails?bbbs={id}  → 法规元数据 + 目录树
  - GET /law-search/download/pc?format=docx&bbbs={id} → 签名 OSS 直链（下载 docx 全文）
  - POST /law-search/search/list           → 搜索列表（有验证码/防爬，自动化枚举受限）

流程：bbbs ID → flfgDetails 拿元数据 → download/pc 拿签名直链 → 下载 docx → 提取全文
      → 存 data/raw/laws.jsonl（title/type/publish_date/effective_date/status/content）

用法：
    python scripts/01b_download_official_laws.py                    # 下载内置 5 部核心法
    python scripts/01b_download_official_laws.py --law_ids "id1,id2"  # 追加指定 ID
    python scripts/01b_download_official_laws.py --discover          # 用 recommend 发现更多法

⚠️ 合规：法条原文不受著作权保护（《著作权法》第五条）；批量采集请遵守官方服务条款、
    控制频率，学习研究用。搜索枚举接口有验证码，本脚本用「精选核心法 bbbs」直连详情接口，
    避免高频抓取列表。
"""

import argparse
import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE = "https://flk.npc.gov.cn"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# 内置核心法（bbbs 从 aggregateData.popularSearch 获取，覆盖 7 子域）
CORE_LAWS = {
    # bbbs ID                               名称
    "ff808081729d1efe01729d50b5c500bf": "中华人民共和国民法典",
    "ff808181796a636a0179822a19640c92": "中华人民共和国刑法",
    "ff8081818a21dc13018b425303b7086d": "中华人民共和国民事诉讼法",
    "2c909fdd678bf17901678bf74d7106b3": "中华人民共和国劳动合同法",
    "ff80818197af9ccc0197b159c38a0408": "中华人民共和国治安管理处罚法",
}

SXX_MAP = {1: "已废止", 2: "已修改", 3: "有效", 4: "尚未生效"}


def get_json(endpoint, params=None):
    resp = requests.get(f"{BASE}{endpoint}", params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_details(bbbs: str) -> dict:
    """拿法规元数据 + ossFile + 目录树。"""
    return get_json("/law-search/search/flfgDetails", {"bbbs": bbbs})["data"]


def get_download_url(bbbs: str, fmt: str = "docx") -> str:
    """拿签名 OSS 直链（download/pc 返回的 data.url）。"""
    data = get_json("/law-search/download/pc", {"format": fmt, "bbbs": bbbs})["data"]
    return data["url"]


def docx_to_text(content: bytes) -> str:
    """docx → 纯文本（按段落，一行一条）。"""
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    paras = re.split(r"</w:p>", xml)
    out = []
    for p in paras:
        # 精确匹配 <w:t> 或 <w:t xml:space="preserve">（避开 <w:topLinePunct> 等）
        texts = re.findall(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", p)
        line = "".join(texts).strip()
        if line:
            out.append(line)
    return "\n".join(out)


def fetch_law(bbbs: str, delay: float = 0.5) -> dict:
    """下载一部法，返回 {title, type, publish_date, effective_date, status, content}。"""
    d = get_details(bbbs)
    url = get_download_url(bbbs)
    docx = requests.get(url, headers=HEADERS, timeout=60).content
    content = docx_to_text(docx)
    time.sleep(delay)
    return {
        "title": d.get("title", ""),
        "type": d.get("flxz", ""),
        "publish_date": d.get("gbrq", ""),
        "effective_date": d.get("sxrq", ""),
        "status": SXX_MAP.get(d.get("sxx"), str(d.get("sxx", ""))),
        "content": content,
        "source": "flk.npc.gov.cn",
        "bbbs": bbbs,
    }


def discover_more(seed_bbbs: str, n: int = 10) -> list:
    """用 recommend 接口发现相关法（返回 [{bbbs, title}]）。"""
    try:
        data = get_json("/law-search/search/recommend", {"bbbs": seed_bbbs})
        items = data.get("data", []) or []
        return [{"bbbs": x.get("bbbs"), "title": x.get("title")} for x in items[:n]
                if x.get("bbbs")]
    except Exception as e:
        print(f"  ⚠ recommend 失败: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="从国家法律法规数据库下载法条全文")
    parser.add_argument("--law_ids", type=str, default="", help="逗号分隔的 bbbs ID")
    parser.add_argument("--output", type=str, default="data/raw/laws.jsonl")
    parser.add_argument("--discover", action="store_true", help="用 recommend 发现更多法")
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    laws = dict(CORE_LAWS)
    if args.law_ids:
        for i in args.law_ids.split(","):
            i = i.strip()
            if i:
                laws[i] = f"bbbs:{i}"  # 名称待 fetch 后回填

    if args.discover:
        seeds = list(CORE_LAWS.keys())
        for s in seeds:
            for item in discover_more(s):
                if item["bbbs"] not in laws:
                    laws[item["bbbs"]] = item["title"]
        print(f"[discover] 当前共 {len(laws)} 部法")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("📚 从国家法律法规数据库下载法条全文")
    print(f"   共 {len(laws)} 部法 → {out}")
    print("=" * 60)

    ok = fail = 0
    with open(out, "w", encoding="utf-8") as f:
        for i, (bbbs, name) in enumerate(laws.items(), 1):
            print(f"  [{i}/{len(laws)}] {name[:30]}...", end="", flush=True)
            try:
                rec = fetch_law(bbbs, args.delay)
                rec["title"] = rec["title"] or name
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f" → {len(rec['content'])} 字")
                ok += 1
            except Exception as e:
                print(f" → 失败: {e}")
                fail += 1

    print(f"\n✅ 完成：成功 {ok} / 失败 {fail}")
    print(f"   保存到: {out.resolve()}")
    print(f"   下一步: python scripts/14_distill_guidelines.py（法条→QA 蒸馏）")


if __name__ == "__main__":
    main()
