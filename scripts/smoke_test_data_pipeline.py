"""
smoke_test_data_pipeline.py — 数据管道冒烟测试（合成数据，无需联网/API）。

覆盖新增/改动的代码路径：
  1. 共享模块：中文数字 / 法条提取 / NHSR / MinHash 去重
  2. DPO 扰动法：编造条号 / 张冠李戴 / 去免责+绝对化
  3. RAFT：oracle 匹配 + 拒答负例
  4. 02_curate：三连去重 + 多样性采样
  5. 15_quality：NHSR 引用审计

用法：
    PYTHONIOENCODING=utf-8 python scripts/smoke_test_data_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.data_quality import (
    chinese_numeral_to_int, extract_statute_citations, build_statute_lookup,
    verify_nhsr, minhash_dedup, exact_dedup, estimate_difficulty,
    difficulty_bucket, has_disclaimer, has_overpromise,
)


def _laws():
    return [
        {"title": "中华人民共和国劳动合同法", "type": "法律",
         "content": "第四十七条：经济补偿按劳动者在本单位工作的年限，每满一年支付一个月工资。\n"
                    "第八十七条：用人单位违反本法规定解除或者终止劳动合同的，应当依照本法第四十七条"
                    "规定的经济补偿标准的二倍向劳动者支付赔偿金。"},
        {"title": "中华人民共和国民法典", "type": "法律",
         "content": "第五百八十七条：给付定金的一方不履行债务的，无权请求返还定金；"
                    "收受定金的一方不履行债务的，应当双倍返还定金。"},
        {"title": "中华人民共和国劳动争议调解仲裁法", "type": "法律",
         "content": "第二十七条：劳动争议申请仲裁的时效期间为一年。"},
    ]


def test_core():
    print("== 1. 共享模块 ==")
    assert chinese_numeral_to_int("八十七") == 87
    assert chinese_numeral_to_int("一百二十三") == 123
    cites = extract_statute_citations("《劳动合同法》第八十七条、《民法典》第五百八十七条第二款")
    assert cites[0]["article"] == 87 and cites[1]["article"] == 587 and cites[1]["para"] == 2

    lookup = build_statute_lookup(laws=_laws())
    r = verify_nhsr("依据《劳动合同法》第八十七条，应支付二倍赔偿金。", lookup)
    assert r["nhsr"] == 1.0, r
    r2 = verify_nhsr("依据《劳动合同法》第二百条。", lookup)
    assert r2["nhsr"] == 0.0

    # MinHash 去重：近似重复应被合并
    samples = [
        {"q": "被公司辞退能拿多少赔偿？", "a": "根据《劳动合同法》第八十七条，应支付二倍赔偿金。" * 3},
        {"q": "被公司辞退能拿多少赔偿？", "a": "根据《劳动合同法》第八十七条，应支付二倍赔偿金。" * 3},
        {"q": "离婚财产怎么分割？", "a": "夫妻共同财产原则上均等分割。" * 3},
    ]
    u = exact_dedup(samples, lambda s: s["q"] + s["a"])
    assert len(u) == 2, f"exact dedup failed: {len(u)}"
    u2 = minhash_dedup(samples, lambda s: s["q"] + s["a"])
    assert len(u2) == 2, f"minhash dedup failed: {len(u2)}"
    print("   共享模块 OK")


def test_dpo_perturb():
    print("\n== 2. DPO 扰动法 ==")
    from scripts03_build_dpo_pairs import (
        perturb_fabricate_article, perturb_misattribute_law,
        perturb_no_disclaimer_overpromise, int_to_chinese,
    )
    lookup = build_statute_lookup(laws=_laws())
    good = ("根据《劳动合同法》第八十七条，违法解除应支付二倍赔偿金。"
            "以上内容仅供参考，不构成法律意见。")

    fab = perturb_fabricate_article(good, lookup)
    assert "第八十七条" not in fab, fab
    r = verify_nhsr(fab, lookup)
    assert r["nhsr"] == 0.0, f"fabricate 应导致 NHSR=0: {r}"

    mis = perturb_misattribute_law(good, lookup)
    assert "劳动合同法" not in mis, mis

    over = perturb_no_disclaimer_overpromise(good)
    assert not has_disclaimer(over) and has_overpromise(over), over

    assert int_to_chinese(87) == "八十七"
    assert int_to_chinese(187) == "一百八十七"
    print("   DPO 扰动法 OK")


def test_raft():
    print("\n== 3. RAFT ==")
    from scripts04b_build_raft_data import chunk_laws, build_raft_records
    import tempfile, json, os

    tmp = Path(tempfile.mkdtemp())
    laws_file = tmp / "laws.jsonl"
    with open(laws_file, "w", encoding="utf-8") as f:
        for l in _laws():
            f.write(json.dumps(l, ensure_ascii=False) + "\n")

    chunks = chunk_laws(str(laws_file))
    assert len(chunks) >= 3, f"chunk 数不对: {len(chunks)}"
    lookup = build_statute_lookup(str(laws_file))

    distilled = [
        {"question": "被公司违法辞退能拿多少赔偿？",
         "answer": "根据《劳动合同法》第八十七条，应支付二倍赔偿金。以上内容仅供参考，不构成法律意见。",
         "fact_verified": True},
    ]
    records = build_raft_records(distilled, chunks, lookup, p_oracle=1.0, n_distractors=2)
    assert len(records) == 1, records
    r = records[0]
    assert r["has_oracle"] is True
    assert "第八十七条" in r["context"], r["context"]
    assert "【检索到的法条】" in r["context"]
    print(f"   RAFT oracle 匹配 OK（context 含 {r['oracle_law']}）")
    # 拒答负例
    records2 = build_raft_records(distilled, chunks, lookup, p_oracle=0.0, n_distractors=2)
    assert records2[0]["has_oracle"] is False and "无法给出准确" in records2[0]["answer"]
    print("   RAFT 拒答负例 OK")


def test_curate_sampling():
    print("\n== 4. 02_curate 采样 ==")
    from scripts02_curate_data import diversity_sample, quality_filter
    samples = []
    for i in range(200):
        samples.append({"question": f"劳动问题{i} 辞退赔偿 加班费 竞业限制",
                        "answer": f"根据《劳动合同法》第八十七条 赔偿 {i}。" * 5,
                        "reference": [], "source": "disc"})
    for i in range(200):
        samples.append({"question": f"离婚财产分割问题{i}",
                        "answer": f"夫妻共同财产均等分割 {i}。" * 5,
                        "reference": [], "source": "disc"})
    picked = diversity_sample(samples, 100)
    assert len(picked) == 100, len(picked)
    print(f"   多样性采样 OK（{len(samples)} → {len(picked)}）")


def test_quality_filter():
    print("\n== 5. 15_quality 引用审计 ==")
    from scripts15_quality_filter import filter_dataset, audit_citations
    import tempfile, json

    lookup = build_statute_lookup(laws=_laws())
    tmp = Path(tempfile.mkdtemp())
    inp = tmp / "in.jsonl"
    data = [
        {"question": "被公司辞退能拿多少赔偿",
         "answer": "根据《劳动合同法》第八十七条，应支付二倍赔偿金。以上内容仅供参考，不构成法律意见。"},
        {"question": "被公司辞退能拿多少赔偿",
         "answer": "根据《劳动合同法》第二百条，应支付三倍赔偿。以上内容仅供参考。"},
    ]
    with open(inp, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    kept, filtered = filter_dataset(str(inp), str(tmp / "out.jsonl"), lookup=lookup)
    assert len(kept) == 1 and len(filtered) == 1, (len(kept), len(filtered))
    assert filtered[0]["reason"] == "编造法条"
    print("   15_quality 引用审计 OK（正确保留 / 编造法条过滤）")


if __name__ == "__main__":
    # 允许以 import 方式导入子模块（脚本文件名以数字开头，不能直接 import）
    import importlib.util

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules[name] = mod
        return mod

    _load("scripts03_build_dpo_pairs", Path("scripts/03_build_dpo_pairs.py"))
    _load("scripts04b_build_raft_data", Path("scripts/04b_build_raft_data.py"))
    _load("scripts02_curate_data", Path("scripts/02_curate_data.py"))
    _load("scripts15_quality_filter", Path("scripts/15_quality_filter.py"))

    test_core()
    test_dpo_perturb()
    test_raft()
    test_curate_sampling()
    test_quality_filter()
    print("\n✅ 全部冒烟测试通过")
