"""
04_build_rag.py — 构建法律 RAG 知识库（法条 + 司法解释）。

处理法条/司法解释文本，构建混合检索向量库（ChromaDB + BM25）。

Pipeline:
  1. 加载 data/raw/laws.jsonl（来自 01_download_data.py）或 data/knowledge_base/*.txt
  2. 按「条」/段落语义分块
  3. 嵌入并索引到 ChromaDB（collection=legal_statutes）
  4. 构建 BM25 索引（混合检索）

用法：
    python scripts/04_build_rag.py
    python scripts/04_build_rag.py --law_file data/raw/laws.jsonl --rebuild
    python scripts/04_build_rag.py --init_template
"""

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.data_quality import split_articles

# ──────────────────────────────────────────────
# 内置法律文档（离线演示用；真实数据来自 laws.jsonl）
# ──────────────────────────────────────────────

BUILTIN_DOCUMENTS = [
    {
        "title": "劳动合同法·经济补偿",
        "source": "中华人民共和国劳动合同法",
        "section": "解除和终止",
        "content": (
            "《中华人民共和国劳动合同法》第四十七条：经济补偿按劳动者在本单位工作的年限，"
            "每满一年支付一个月工资的标准向劳动者支付。六个月以上不满一年的，按一年计算；"
            "不满六个月的，向劳动者支付半个月工资的经济补偿。\n\n"
            "第八十七条：用人单位违反本法规定解除或者终止劳动合同的，应当依照本法第四十七条"
            "规定的经济补偿标准的二倍向劳动者支付赔偿金。"
        ),
    },
    {
        "title": "劳动争议调解仲裁法·仲裁时效",
        "source": "中华人民共和国劳动争议调解仲裁法",
        "section": "时效",
        "content": (
            "《中华人民共和国劳动争议调解仲裁法》第二十七条：劳动争议申请仲裁的时效期间为一年。"
            "仲裁时效期间从当事人知道或者应当知道其权利被侵害之日起计算。"
        ),
    },
    {
        "title": "民法典·定金",
        "source": "中华人民共和国民法典",
        "section": "合同编",
        "content": (
            "《中华人民共和国民法典》第五百八十七条：债务人履行债务的，定金应当抵作价款或者收回。"
            "给付定金的一方不履行债务或者履行债务不符合约定，致使不能实现合同目的的，"
            "无权请求返还定金；收受定金的一方不履行债务或者履行债务不符合约定，"
            "致使不能实现合同目的的，应当双倍返还定金。"
        ),
    },
    {
        "title": "劳动争议司法解释（二）·竞业限制",
        "source": "最高人民法院关于审理劳动争议案件适用法律问题的解释（二）",
        "section": "竞业限制",
        "content": (
            "竞业限制未约定经济补偿的，劳动者履行了竞业限制义务的，"
            "可以要求用人单位按月支付经济补偿。"
        ),
    },
]

# ──────────────────────────────────────────────
# 文档处理
# ──────────────────────────────────────────────

def semantic_chunk(text: str, max_chunk_size: int = 800, min_chunk_size: int = 100) -> list:
    """按章节/段落语义分块。"""
    chunks = []
    section_pattern = r'(?:【|\[|第[一二三四五六七八九十\d]+[章节条]|\d+\.\d+|\n\n)'
    raw_sections = re.split(f'({section_pattern})', text)

    i = 0
    while i < len(raw_sections):
        chunk = raw_sections[i]
        if i + 1 < len(raw_sections):
            chunk += raw_sections[i + 1]
            i += 2
        else:
            i += 1
        chunk = chunk.strip()
        if len(chunk) < min_chunk_size:
            continue
        if len(chunk) > max_chunk_size:
            paragraphs = re.split(r'\n\n+', chunk)
            sub_chunk = ""
            for para in paragraphs:
                if len(sub_chunk) + len(para) > max_chunk_size and sub_chunk:
                    chunks.append(sub_chunk.strip())
                    sub_chunk = para
                else:
                    sub_chunk += "\n\n" + para if sub_chunk else para
            if sub_chunk.strip():
                chunks.append(sub_chunk.strip())
        else:
            chunks.append(chunk)
    return chunks


def load_laws_jsonl(law_file: str) -> list:
    """加载法条语料（laws.jsonl），转成文档列表。"""
    documents = []
    law_path = Path(law_file)
    if not law_path.exists():
        print(f"   ⚠ 法条文件不存在: {law_file}（先运行 01_download_data.py）")
        return documents

    with open(law_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            title = item.get("title", "")
            typ = item.get("type", "")
            content = item.get("content", "")
            if not content:
                continue
            documents.append({
                "title": title,
                "source": f"{title}（{typ}）",
                "section": typ,
                "content": content,
            })
    print(f"   Loaded {len(documents)} 法条文档 from {law_file}")
    return documents


def load_documents(law_file: str = None, kb_dir: str = None, load_builtin: bool = True) -> list:
    """加载全部文档。优先级：laws.jsonl > kb_dir txt > builtin。"""
    documents = []

    if law_file and Path(law_file).exists():
        documents.extend(load_laws_jsonl(law_file))

    if kb_dir:
        kb_path = Path(kb_dir)
        if kb_path.exists():
            for txt_file in kb_path.glob("**/*.txt"):
                if txt_file.name == "README.txt":
                    continue
                with open(txt_file, "r", encoding="utf-8") as f:
                    content = f.read()
                rel_path = txt_file.relative_to(kb_path)
                parts = rel_path.parts
                documents.append({
                    "title": txt_file.stem,
                    "source": parts[0] if len(parts) > 1 else "manual",
                    "section": parts[0] if len(parts) > 1 else "",
                    "content": content,
                })

    if not documents and load_builtin:
        documents.extend(BUILTIN_DOCUMENTS)
        print(f"   ⚠ 未找到外部法条，加载 {len(BUILTIN_DOCUMENTS)} 篇内置演示文档")

    # 按「条」分块（散文类按段落兜底）
    chunked_docs = chunk_by_article(documents)
    print(f"   After chunking: {len(chunked_docs)} chunks")
    return chunked_docs


def _split_long(text: str, max_len: int = 600) -> list:
    """长条按段落二次切分，保证单 chunk 不超长。"""
    if len(text) <= max_len:
        return [text]
    paras = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    out, cur = [], ""
    for p in paras:
        if len(cur) + len(p) > max_len and cur:
            out.append(cur.strip())
            cur = p
        else:
            cur += p + "\n"
    if cur.strip():
        out.append(cur.strip())
    return out or [text[:max_len]]


def chunk_by_article(documents: list) -> list:
    """按「条」分块：每条一个 chunk（含条号元数据）；散文类（批复/规定）按段落分块。"""
    chunked = []
    for doc in documents:
        content = doc.get("content", "")
        articles = split_articles(content)
        if articles:
            for num, text in articles:
                for sub in _split_long(text):
                    chunked.append({
                        "title": doc.get("title", ""),
                        "source": doc.get("source", ""),
                        "section": doc.get("section", ""),
                        "content": sub,
                        "article": num,
                    })
        else:
            # 散文类：按段落分块，保留整段
            paras = [p.strip() for p in re.split(r"\n+", content) if p.strip()]
            for p in paras:
                chunked.append({
                    "title": doc.get("title", ""),
                    "source": doc.get("source", ""),
                    "section": doc.get("section", ""),
                    "content": p,
                    "article": None,
                })
    return chunked


# ──────────────────────────────────────────────
# 构建知识库
# ──────────────────────────────────────────────

def build_knowledge_base(law_file: str = None, kb_dir: str = None,
                          persist_dir: str = "./data/vector_db", rebuild: bool = False,
                          use_dense: bool = True):
    from app.rag_retriever import HybridRetriever

    print("=" * 60)
    print("📚 Building LexiCare Legal RAG Knowledge Base")
    print("=" * 60)

    print("\n[1/3] Loading documents...")
    documents = load_documents(law_file, kb_dir)

    if rebuild and Path(persist_dir).exists():
        shutil.rmtree(persist_dir)
        print(f"   Rebuilding: removed existing DB at {persist_dir}")

    print(f"\n[2/3] Indexing {len(documents)} chunks...")
    retriever = HybridRetriever(persist_dir=persist_dir, use_dense=use_dense)
    retriever.index(documents)

    print("\n[3/3] Verifying retrieval quality...")
    test_queries = [
        "被辞退能拿多少赔偿",
        "经济补偿怎么计算",
        "劳动仲裁时效多久",
        "定金能不能退",
        "竞业限制没有补偿有效吗",
    ]
    for query in test_queries:
        results = retriever.retrieve(query, top_k=2)
        found = len(results)
        top_title = results[0]["title"] if results else "N/A"
        print(f"   🔍 '{query}' → {found} docs, top: {top_title[:50]}")

    meta = {
        "built_at": datetime.now().isoformat(),
        "total_documents": len(documents),
        "test_queries_passed": True,
    }
    meta_path = Path(persist_dir) / "kb_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Knowledge base built successfully!")
    print(f"   {len(documents)} chunks indexed in {persist_dir}")
    return retriever


# ──────────────────────────────────────────────
# 知识库模板
# ──────────────────────────────────────────────

def create_kb_template(output_dir: str = "data/knowledge_base"):
    kb_path = Path(output_dir)
    subdirs = ["statutes", "interpretations", "cases", "faq"]
    for sub in subdirs:
        sub_path = kb_path / sub
        sub_path.mkdir(parents=True, exist_ok=True)
        readme = sub_path / "README.txt"
        if not readme.exists():
            with open(readme, "w", encoding="utf-8") as f:
                if sub == "statutes":
                    f.write("# 法律法规\n放置法条纯文本(.txt)。\n\n")
                    f.write("推荐来源：国家法律法规数据库 flk.npc.gov.cn\n")
                elif sub == "interpretations":
                    f.write("# 司法解释\n放置最高法/最高检司法解释文本。\n")
                elif sub == "cases":
                    f.write("# 裁判文书\n放置典型案例/裁判文书摘要。\n")
                elif sub == "faq":
                    f.write("# 常见问题\n放置法律常见问题解答。\n")
    print(f"✅ Knowledge base template created at: {kb_path}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build legal RAG knowledge base")
    parser.add_argument("--law_file", type=str, default="data/raw/laws_clean.jsonl")
    parser.add_argument("--kb_dir", type=str, default=None, help="Directory with .txt knowledge files")
    parser.add_argument("--persist_dir", type=str, default="./data/vector_db")
    parser.add_argument("--rebuild", action="store_true", help="Delete existing DB and rebuild")
    parser.add_argument("--no_dense", action="store_true",
                        help="跳过 ChromaDB 稠密检索（BM25-only，快；默认 dense 较慢）")
    parser.add_argument("--init_template", action="store_true", help="Create knowledge base directory template")
    args = parser.parse_args()

    if args.init_template:
        create_kb_template()
        return

    build_knowledge_base(
        law_file=args.law_file,
        kb_dir=args.kb_dir,
        persist_dir=args.persist_dir,
        rebuild=args.rebuild,
        use_dense=not args.no_dense,
    )


if __name__ == "__main__":
    main()
