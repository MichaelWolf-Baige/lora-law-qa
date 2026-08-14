"""
data_quality.py — 法律数据质量共享工具（去重 / 引用校验 / 质量打分）。

单一事实来源：蒸馏、清洗、过滤、DPO 构造共用的数据质量原语。
刻意保持轻依赖（仅 stdlib + numpy），避免引入 sentence-transformers 等重库。

核心能力：
  1. 中文数字转换（第八十七条 → 87）
  2. 法条引用提取（《法名》第X条 第X款/项，含中文数字）
  3. 法条库 lookup + NHSR 校验（名称/条号，可选轻量内容比对）
  4. 去重：精确 + MinHash（纯 Python LSH）+ 语义（可选 embedding）
  5. 免责声明 / 承诺胜诉 / 难度估计
  6. OpenAI 兼容 LLM client（DeepSeek 等，用于质量打分与蒸馏）

用法：
    from app.data_quality import (
        chinese_numeral_to_int, extract_statute_citations, StatuteLookup,
        build_statute_lookup, verify_nhsr, minhash_dedup, normalize_text,
        has_disclaimer, estimate_difficulty, LLMClient,
    )
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# ──────────────────────────────────────────────
# 1. 中文数字转换
# ──────────────────────────────────────────────

_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}


def chinese_numeral_to_int(s: str) -> Optional[int]:
    """中文数字 → int。如 '八十七'→87、'一百二十三'→123、'十'→10。非法返回 None。"""
    s = s.strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    section, num = 0, 0
    for ch in s:
        if ch in _CN_DIGITS:
            num = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            unit = _CN_UNITS[ch]
            if num == 0:
                num = 1
            section += num * unit
            num = 0
        else:
            return None
    return section + num


# ──────────────────────────────────────────────
# 2. 法条引用提取
# ──────────────────────────────────────────────

# 匹配《法名》第X条 [第X款/项]，X 支持中文数字 + 阿拉伯数字
STATUTE_CITE_RE = re.compile(
    r'《(?P<law>[^》]{2,30}?)》\s*'
    r'第\s*(?P<article>[零〇一二三四五六七八九十百千0-9]{1,8})\s*条'
    r'(?:\s*第\s*(?P<para>[零〇一二三四五六七八九十百千0-9]{1,4})\s*[款项])?'
)

# 裸条号（无书名号）：第X条
BARE_ARTICLE_RE = re.compile(
    r'(?:^|[^《])第\s*([零〇一二三四五六七八九十百千0-9]{1,8})\s*条'
)

# 司法解释式引用：最高人民法院…解释（X）第Y条。法名本身含嵌套《》（如
# 「最高人民法院关于适用《中华人民共和国民法典》婚姻家庭编的解释（一）」），
# 标准《法名》第X条 正则抓不到，需单独匹配「法名 + 第X条」尾缀。
JUDICIAL_CITE_RE = re.compile(
    r'(?P<law>最高人民法院[^第。；，]{1,60}?)'
    r'第\s*(?P<article>[零〇一二三四五六七八九十百千0-9]{1,8})\s*条'
)


def extract_statute_citations(text: str) -> List[dict]:
    """提取文本中的法条引用，返回 [{law, article(int), para(int|None), raw}]。"""
    citations = []
    seen = set()
    for m in STATUTE_CITE_RE.finditer(text):
        article = chinese_numeral_to_int(m.group("article"))
        if article is None:
            continue
        para = None
        if m.group("para"):
            para = chinese_numeral_to_int(m.group("para"))
        key = (m.group("law").strip(), article, para)
        if key in seen:
            continue
        seen.add(key)
        citations.append({
            "law": m.group("law").strip(),
            "article": article,
            "para": para,
            "raw": m.group(0),
        })
    # 司法解释式引用（法名含嵌套《》，标准正则抓不到）
    for m in JUDICIAL_CITE_RE.finditer(text):
        article = chinese_numeral_to_int(m.group("article"))
        if article is None:
            continue
        law = m.group("law").strip()
        key = (law, article, None)
        if key in seen:
            continue
        seen.add(key)
        citations.append({"law": law, "article": article, "para": None, "raw": m.group(0)})
    return citations


# ──────────────────────────────────────────────
# 3. 法条库 lookup + NHSR 校验
# ──────────────────────────────────────────────

def _short_name(title: str) -> str:
    """法名简称：去掉 '中华人民共和国' 前缀。"""
    return title.replace("中华人民共和国", "").strip()


def split_articles(content: str) -> List[Tuple[int, str]]:
    """
    把法条全文按「行首的第X条」切分成 [(条号, 条文文本), ...]。

    关键：只在**行首**匹配「第X条」作为新条目边界；正文里的交叉引用
    （如"依照本法第四十七条规定的…"）出现在行中间，不会被误判为新条目；
    续行（如"本条所称月工资…"）并入上一条。
    """
    head_re = re.compile(r'^\s*第\s*([零〇一二三四五六七八九十百千0-9]{1,8})\s*条')
    articles: List[Tuple[int, str]] = []
    current_num = None
    current_parts: List[str] = []

    def _flush():
        nonlocal current_num, current_parts
        if current_num is not None:
            articles.append((current_num, "\n".join(current_parts).strip()))
        current_num = None
        current_parts = []

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        m = head_re.match(line)
        if m:
            num = chinese_numeral_to_int(m.group(1))
            if num is not None:
                _flush()
                current_num = num
                current_parts = [line]
                continue
        # 非新条 → 续行（正文可能分多段）
        if current_num is not None:
            current_parts.append(line)
    _flush()
    return articles


class StatuteLookup:
    """
    法条库查询器。支持法名简称/全称匹配 + 条号 → 条文原文。

    index: {full_title: {article_num: article_text}}
    """

    def __init__(self, laws: Iterable[dict] = None):
        self._by_full = {}      # full_title -> {article: text}
        self._alias = {}        # alias(short/substr) -> full_title
        if laws:
            self.index(laws)

    def index(self, laws: Iterable[dict]):
        for item in laws:
            title = (item.get("title") or "").strip()
            content = item.get("content") or ""
            if not title or not content:
                continue
            arts = split_articles(content)
            if not arts:
                continue
            self._by_full[title] = dict(arts)
            # 建别名：简称 + 常见截断
            aliases = {title, _short_name(title)}
            for a in aliases:
                if a and a not in self._alias:
                    self._alias[a] = title

    def resolve_law(self, law: str) -> Optional[str]:
        """法名（可能简称）→ 全称，匹配不到返回 None。"""
        law = (law or "").strip()
        if not law:
            return None
        if law in self._alias:
            return self._alias[law]
        # 子串匹配：优先最长匹配，避免「…婚姻家庭编的解释（一）」被短标题「民法典」抢先命中
        best = None
        for full in self._by_full:
            if law in full or full in law:
                if best is None or len(full) > len(best):
                    best = full
        if best:
            return best
        # 去前缀后再试
        short = _short_name(law)
        if short in self._alias:
            return self._alias[short]
        return None

    def get_article(self, law: str, article: int) -> Optional[str]:
        full = self.resolve_law(law)
        if not full:
            return None
        return self._by_full[full].get(article)

    def article_exists(self, law: str, article: int) -> bool:
        return self.get_article(law, article) is not None

    def articles_of(self, law: str) -> List[int]:
        """某部法当前索引到的全部条号（升序）。"""
        full = self.resolve_law(law)
        if not full:
            return []
        return sorted(self._by_full[full].keys())

    def law_names(self) -> List[str]:
        """已索引的全部法名（全称）。"""
        return sorted(self._by_full.keys())

    def nearest_article(self, law: str, article: int) -> Optional[int]:
        """某部法中离 article 最近的合法条号。"""
        arts = self.articles_of(law)
        if not arts:
            return None
        return min(arts, key=lambda x: abs(x - article))

    def __len__(self):
        return sum(len(v) for v in self._by_full.values())


def build_statute_lookup(laws_jsonl: Optional[str] = None,
                         laws: Optional[Iterable[dict]] = None) -> StatuteLookup:
    """从 laws.jsonl（或 laws 列表）构建 lookup。"""
    lookup = StatuteLookup()
    if laws is not None:
        lookup.index(laws)
        return lookup
    if laws_jsonl and Path(laws_jsonl).exists():
        items = []
        with open(laws_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        lookup.index(items)
    return lookup


# 数量表达：数字 + 内容单位（倍/年/月/日/天/元/万/%/小时/周）。
# 注意：刻意排除「条/款/项」——它们是结构性引用单位，已由「名称+条号」检查覆盖，
# 若纳入会把「第X条」这个引用本身当成数量，稀释内容比对信号。
# 用于「内容一致性」轻量比对——回答里引用的关键数量是否在条文原文中出现。
_QUANT_RE = re.compile(
    r'[零〇一二两三四五六七八九十百千万0-9]{1,6}'
    r'(?:倍|年|个月|月|日|天|元|万|%|％|小时|周)'
)


def _lexical_content_score(answer: str, article_text: str) -> float:
    """
    轻量内容一致性信号（0~1）：回答中「数量表达」有多少能在条文原文中溯源。

    这是一个廉价启发式（triage），不是语义证明：
    - 抓「三倍」「一年」「一个月」「20%」这类带单位的数字，检查是否出现在条文里；
    - 回答没有数量表达时返回 1.0（无可比对，宽松放行）。

    它擅长抓「条号存在但内容张冠李戴」的粗错（如条文说二倍、回答说三倍），
    但**无法**验证非数字的语义转述是否准确——那需要 embedding 语义比对或 LLM 复核。
    """
    spans = set(_QUANT_RE.findall(answer or ""))
    if not spans:
        return 1.0
    article = article_text or ""
    hit = sum(1 for s in spans if s in article)
    return hit / len(spans)


def verify_nhsr(answer: str, lookup: StatuteLookup,
                content_check: Optional[str] = None,
                embed_fn=None, content_threshold: float = 0.5) -> dict:
    """
    NHSR（非幻觉法条率）校验：名称 / 条号 / （可选）内容。

    content_check 三档（默认 None，只查名称+条号，向后兼容）：
      - None       ：只查法名 + 条号是否存在（原行为）
      - "lexical"  ：轻量数量表达比对（_lexical_content_score，抓粗错）
      - "embed"    ：embedding 语义相似度（需传 embed_fn(text)->vector）

    返回 {total, valid, invalid, content_invalid, nhsr, citations}，
    其中 citations 每条含 {law, article, raw, name_matched, article_exists,
    content_score, content_matched, article_text, valid}。
    """
    citations = extract_statute_citations(answer)
    valid, invalid, content_invalid = 0, 0, 0
    checked = []
    for c in citations:
        full = lookup.resolve_law(c["law"])
        name_matched = full is not None
        text = lookup.get_article(c["law"], c["article"]) if name_matched else None
        article_exists = text is not None

        content_score = None
        content_matched = True
        if content_check and text is not None:
            if content_check == "lexical":
                content_score = _lexical_content_score(answer, text)
            elif content_check == "embed" and embed_fn is not None:
                import numpy as np
                a = embed_fn(answer)
                b = embed_fn(text)
                content_score = float(
                    np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
            if content_score is not None:
                content_matched = content_score >= content_threshold

        ok = name_matched and article_exists and content_matched
        if ok:
            valid += 1
        else:
            invalid += 1
            if name_matched and article_exists and not content_matched:
                content_invalid += 1
        checked.append({
            "law": c["law"], "article": c["article"], "para": c["para"],
            "raw": c["raw"], "resolved_law": full,
            "name_matched": name_matched, "article_exists": article_exists,
            "content_score": content_score, "content_matched": content_matched,
            "article_text": text, "valid": ok,
        })
    return {
        "total": len(citations), "valid": valid, "invalid": invalid,
        "content_invalid": content_invalid,
        "nhsr": (valid / len(citations)) if citations else None,
        "citations": checked,
    }


# ──────────────────────────────────────────────
# 4. 去重（精确 + MinHash LSH + 语义）
# ──────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """去重规范化：小写、去空白与标点。"""
    return re.sub(r"[\s\W_]+", "", (text or "").lower())


def text_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _char_ngrams(text: str, n: int = 5):
    """字符 n-gram（中文按字，英文按词）。"""
    t = normalize_text(text)
    return [t[i:i + n] for i in range(max(len(t) - n + 1, 0))]


def _hash_fn(seed: int):
    def h(x: str) -> int:
        return int(hashlib.md5(f"{seed}:{x}".encode("utf-8")).hexdigest()[:8], 16)
    return h


class MinHasher:
    """纯 Python MinHash 签名（字符 n-gram）。"""

    def __init__(self, num_perm: int = 128, ngram: int = 5):
        self.num_perm = num_perm
        self.ngram = ngram
        self.hash_fns = [_hash_fn(i) for i in range(num_perm)]

    def signature(self, text: str) -> List[int]:
        grams = _char_ngrams(text, self.ngram)
        if not grams:
            return [0] * self.num_perm
        sig = []
        for h in self.hash_fns:
            m = min(h(g) for g in grams)
            sig.append(m)
        return sig


def minhash_dedup(samples: List[dict], key_fn, threshold: float = 0.8,
                  num_perm: int = 128, bands: int = 16) -> List[dict]:
    """
    MinHash LSH 近重复去重。

    bands=16, rows=8 → 相似度阈值约 0.7~0.8。返回去重后的 samples（保留先出现的）。
    """
    hasher = MinHasher(num_perm=num_perm)
    rows = num_perm // bands
    buckets: Dict[tuple, int] = {}   # band-hash -> first sample index
    kept: List[dict] = []
    for sample in samples:
        sig = hasher.signature(key_fn(sample))
        dup = False
        for b in range(bands):
            band = tuple(sig[b * rows:(b + 1) * rows])
            bucket_key = (b, band)
            if bucket_key in buckets:
                dup = True
                break
        if dup:
            continue
        for b in range(bands):
            band = tuple(sig[b * rows:(b + 1) * rows])
            buckets[(b, band)] = len(kept)
        kept.append(sample)
    return kept


def semantic_dedup(samples: List[dict], key_fn, embed_fn, threshold: float = 0.9,
                   top_check: int = 20) -> List[dict]:
    """
    语义去重（可选，需外部 embedding 函数 embed_fn(text)->vector）。
    近似：仅与已保留的最近 top_check 个样本比余弦，避免 O(n²) 精确全比。
    若无 embed_fn，退回不做任何事（返回原列表）。
    """
    if embed_fn is None:
        return samples
    import numpy as np

    def _cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    kept, kept_vecs = [], []
    for sample in samples:
        vec = embed_fn(key_fn(sample))
        dup = False
        for v in kept_vecs[-top_check:]:
            if _cos(vec, v) >= threshold:
                dup = True
                break
        if dup:
            continue
        kept.append(sample)
        kept_vecs.append(vec)
    return kept


def exact_dedup(samples: List[dict], key_fn) -> List[dict]:
    """精确去重（规范化后哈希）。"""
    seen = set()
    kept = []
    for s in samples:
        h = text_hash(normalize_text(key_fn(s)))
        if h in seen:
            continue
        seen.add(h)
        kept.append(s)
    return kept


# ──────────────────────────────────────────────
# 5. 免责声明 / 安全 / 难度
# ──────────────────────────────────────────────

DISCLAIMER_PHRASES = (
    "不构成法律意见", "仅供参考", "建议咨询律师", "请咨询执业律师",
    "咨询专业律师", "以专业律师意见为准", "不建立律师",
)


def has_disclaimer(text: str) -> bool:
    return any(p in text for p in DISCLAIMER_PHRASES)


OVERPROMISE_RE = re.compile(
    r"肯定能赢|一定胜诉|保证胜诉|包赢|必胜|稳赢|百分百.{0,3}(赢|胜)|"
    r"胜诉率.{0,6}\d{1,3}%|胜算.{0,6}\d{1,3}%"
)


def has_overpromise(text: str) -> bool:
    return bool(OVERPROMISE_RE.search(text))


def impersonate_lawyer_re() -> bool:
    return False  # 占位，实际检测在 domain_config 的 forbidden_patterns 里


def estimate_difficulty(question: str, answer: str) -> float:
    """
    难度估计（0~1）：长度 + 法条引用密度 + 数字特异性 + 复杂词。
    调研结论：不要用困惑度，用结构信号更稳（DEITA）。
    """
    q, a = question or "", answer or ""
    score = 0.0
    score += min(len(q) / 100, 1.0) * 0.20
    score += min(len(a) / 600, 1.0) * 0.20
    cites = len(extract_statute_citations(a))
    score += min(cites / 4, 1.0) * 0.25
    nums = len(re.findall(r"\d+\.?\d*\s*(?:年|个月|倍|%|元|天|日)", q + a))
    score += min(nums / 5, 1.0) * 0.20
    complex_words = ("区别", "对比", "如何", "怎么办", "同时", "竞业", "违法解除",
                     "举证责任", "时效中断", "优先受偿", "善意取得", "溯及")
    score += min(sum(1 for w in complex_words if w in q + a) / 4, 1.0) * 0.15
    return round(min(score, 1.0), 3)


def difficulty_bucket(score: float) -> str:
    if score >= 0.6:
        return "hard"
    if score >= 0.3:
        return "medium"
    return "easy"


# ──────────────────────────────────────────────
# 6. LLM Client（OpenAI 兼容）
# ──────────────────────────────────────────────

class LLMClient:
    """OpenAI 兼容端点客户端（DeepSeek / 通用）。"""

    def __init__(self, api_key: Optional[str] = None,
                 base_url: str = "https://api.deepseek.com/v1",
                 model: str = "deepseek-chat"):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.base_url = base_url
        self.model = model

    def chat(self, messages: List[dict], temperature: float = 0.3,
             max_tokens: int = 1500) -> str:
        import requests
        if not self.api_key:
            raise ValueError("未设置 DEEPSEEK_API_KEY（或 api_key）")
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json={"model": self.model, "messages": messages,
                  "temperature": temperature, "max_tokens": max_tokens},
            timeout=90,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def judge_quality(self, question: str, answer: str,
                      temperature: float = 0.1) -> dict:
        """
        AlpaGasus 式质量打分（0–5）。返回 dict 含 score 与 reason。
        维度：法律准确性 / 相关性 / 完整性 / 格式 / 无幻觉。
        """
        prompt = (
            "你是法律数据质量评审专家。请对以下 (问题, 回答) 打分，0–5 分，"
            "只输出 JSON：\n"
            "{\n"
            '  "accuracy": <法条结论是否正确 0-5>,\n'
            '  "relevance": <是否紧扣问题 0-5>,\n'
            '  "completeness": <是否覆盖关键要点 0-5>,\n'
            '  "format": <是否引用法条+结构清晰 0-5>,\n'
            '  "no_hallucination": <无法条编造/无过度承诺 0-5>,\n'
            '  "score": <平均分>,\n'
            '  "reason": "<一句话>"\n'
            "}\n\n"
            f"【问题】{question}\n【回答】{answer[:2000]}"
        )
        raw = self.chat(
            [{"role": "system", "content": "你是法律数据评审专家，严格输出 JSON。"},
             {"role": "user", "content": prompt}],
            temperature=temperature, max_tokens=500,
        )
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return {"score": 0.0, "reason": "JSON 解析失败", "raw": raw[:200]}
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            return {"score": 0.0, "reason": "JSON 解析失败", "raw": raw[:200]}


if __name__ == "__main__":
    # 自检
    assert chinese_numeral_to_int("八十七") == 87
    assert chinese_numeral_to_int("一百二十三") == 123
    assert chinese_numeral_to_int("十") == 10
    assert chinese_numeral_to_int("二十") == 20

    cites = extract_statute_citations(
        "根据《中华人民共和国劳动合同法》第八十七条、第47条，"
        "以及《民法典》第五百八十七条第二款。"
    )
    assert any(c["law"] == "中华人民共和国劳动合同法" and c["article"] == 87 for c in cites), cites
    assert any(c["law"] == "民法典" and c["article"] == 587 and c["para"] == 2 for c in cites), cites

    laws = [{
        "title": "中华人民共和国劳动合同法",
        "content": "第八十七条：用人单位违反本法规定解除或者终止劳动合同的，"
                   "应当依照本法第四十七条规定的经济补偿标准的二倍向劳动者支付赔偿金。",
    }]
    lookup = build_statute_lookup(laws=laws)
    r = verify_nhsr("依据《劳动合同法》第八十七条，应支付二倍赔偿金。", lookup)
    assert r["nhsr"] == 1.0, r
    r2 = verify_nhsr("依据《劳动合同法》第二百条。", lookup)
    assert r2["nhsr"] == 0.0, r2

    dup_in = [{"q": "被公司辞退能拿多少赔偿？"}, {"q": "被公司辞退能拿多少赔偿？"},
              {"q": "公司违法辞退我能拿多少赔偿金？"}]
    assert len(exact_dedup(dup_in, lambda s: s["q"])) == 2
    print("✅ data_quality.py 自检通过")
