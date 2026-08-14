"""
rag_retriever.py — 混合 RAG 检索（领域无关，参数来自 DomainConfig）。

实现 2025-2026 检索最佳实践：
  1. 混合检索：BM25（关键词）+ Dense（语义）集成
  2. 检索门控：简单问题跳过 RAG，降低延迟
  3. 查询改写：领域同义词/全称扩展（config.rag.synonyms）
  4. 重排：复杂问题做启发式/交叉编码器重排

用法：
    retriever = HybridRetriever()     # 默认 get_domain()，collection=legal_statutes
    docs = retriever.retrieve("被辞退能拿多少赔偿", top_k=5)
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np

from app.domain_config import DomainConfig, get_domain


# ──────────────────────────────────────────────
# 查询处理（领域无关，数据来自 config）
# ──────────────────────────────────────────────

def expand_query(query: str, config: DomainConfig) -> str:
    """领域同义词/全称扩展。"""
    expanded = query
    for term, expansion in config.rag.synonyms.items():
        if term in query:
            expanded = expanded.replace(term, f"{term} ({expansion})")
    return expanded


def classify_query_complexity(query: str, config: DomainConfig) -> str:
    """检索门控：simple（跳过）/ medium（推荐）/ complex（必须）。"""
    for pattern in config.rag.simple_patterns:
        if re.search(pattern, query):
            return "simple"
    for pattern in config.rag.complex_patterns:
        if re.search(pattern, query):
            return "complex"
    return "medium"


# ──────────────────────────────────────────────
# BM25 稀疏检索（领域无关）
# ──────────────────────────────────────────────

class BM25Retriever:
    """
    BM25 关键词检索，擅长精确匹配（法条名、专有名词）。

    中文：字级 bigram + 词级混合分词。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_texts = []
        self.avgdl = 0
        self.idf = {}
        self.term_freqs = []

    def index(self, documents: list):
        self.documents = documents
        self.doc_texts = [d.get("content", "") for d in documents]
        tokenized = [self._tokenize(t) for t in self.doc_texts]
        doc_lengths = [len(t) for t in tokenized]
        self.avgdl = np.mean(doc_lengths) if doc_lengths else 1

        N = len(tokenized)
        term_doc_count = {}
        for tokens in tokenized:
            for term in set(tokens):
                term_doc_count[term] = term_doc_count.get(term, 0) + 1
        self.idf = {
            term: np.log((N - count + 0.5) / (count + 0.5) + 1)
            for term, count in term_doc_count.items()
        }
        self.term_freqs = tokenized

    def _tokenize(self, text: str) -> list:
        tokens = []
        chinese_chars = re.findall(r'[一-鿿]', text)
        for i in range(len(chinese_chars) - 1):
            tokens.append(chinese_chars[i] + chinese_chars[i + 1])
        words = re.findall(r'[一-鿿]{2,6}', text)
        tokens.extend(words)
        en_terms = re.findall(r'[a-zA-Z0-9]+(?:\.[a-zA-Z0-9]+)*', text)
        tokens.extend([t.lower() for t in en_terms])
        return tokens

    def search(self, query: str, top_k: int = 5) -> list:
        if not self.documents:
            return []
        query_tokens = self._tokenize(query)
        scores = []
        for i, doc_tokens in enumerate(self.term_freqs):
            score = 0
            doc_len = len(doc_tokens)
            tf = {}
            for t in doc_tokens:
                tf[t] = tf.get(t, 0) + 1
            for term in query_tokens:
                if term not in self.idf:
                    continue
                term_tf = tf.get(term, 0)
                idf = self.idf[term]
                numerator = term_tf * (self.k1 + 1)
                denominator = term_tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += idf * numerator / denominator
            scores.append((i, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ──────────────────────────────────────────────
# 混合检索器
# ──────────────────────────────────────────────

class HybridRetriever:
    """
    混合检索：BM25 + Dense，带门控与重排。
    """

    def __init__(self, config: DomainConfig = None, persist_dir: str = "./data/vector_db",
                 use_reranker: bool = True, use_dense: bool = True):
        self.config = config or get_domain()
        self.persist_dir = Path(persist_dir)
        self.bm25 = BM25Retriever()
        self.documents = []
        self.embedding_fn = None
        self.use_reranker = use_reranker
        self.use_dense = use_dense
        self._init_vector_db()
        self._load_bm25_from_disk()

    def _load_bm25_from_disk(self):
        """从磁盘重载 BM25 文档索引（BM25 是内存索引，需持久化文档再重建）。"""
        doc_file = self.persist_dir / "bm25_documents.jsonl"
        if doc_file.exists():
            try:
                docs = [json.loads(l) for l in open(doc_file, encoding="utf-8") if l.strip()]
                if docs:
                    self.documents = docs
                    self.bm25.index(docs)
                    print(f"   Loaded {len(docs)} BM25 documents from disk")
            except Exception as e:
                print(f"   ⚠ BM25 重载失败: {e}")

    def _init_vector_db(self):
        if not self.use_dense:
            print("   ⚠ Dense 检索关闭（BM25-only 模式）")
            self.chroma_client = None
            self.collection = None
            self.has_vector_db = False
            return
        try:
            import chromadb
            from chromadb.config import Settings
            from chromadb.utils import embedding_functions
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(anonymized_telemetry=False),
            )
            # 中文 embedding（bge-small-zh-v1.5），替代 ChromaDB 默认英文 MiniLM
            try:
                self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="BAAI/bge-small-zh-v1.5",
                )
                print("   中文 embedding: BAAI/bge-small-zh-v1.5")
            except Exception as e:
                print(f"   ⚠ 中文 embedding 初始化失败: {e}，回退 ChromaDB 默认")
                self.embedding_fn = None
            try:
                self.collection = self.chroma_client.get_collection(
                    self.config.rag.collection_name,
                    embedding_function=self.embedding_fn,
                )
                print(f"   Loaded existing ChromaDB collection: {self.collection.count()} documents")
            except Exception:
                self.collection = self.chroma_client.create_collection(
                    self.config.rag.collection_name,
                    metadata={"description": f"LexiCare {self.config.name} knowledge base"},
                    embedding_function=self.embedding_fn,
                )
                print("   Created new ChromaDB collection")
            self.has_vector_db = True
        except ImportError:
            print("   ⚠ ChromaDB not installed. Using BM25-only mode.")
            self.chroma_client = None
            self.collection = None
            self.has_vector_db = False
        except Exception as e:
            print(f"   ⚠ Vector DB init failed: {e}. Using BM25-only mode.")
            self.chroma_client = None
            self.collection = None
            self.has_vector_db = False

    def index(self, documents: list):
        self.documents = documents
        print(f"   Indexing {len(documents)} documents with BM25...")
        self.bm25.index(documents)

        # 持久化 BM25 文档，供后续实例重载（BM25 是内存索引）
        doc_file = self.persist_dir / "bm25_documents.jsonl"
        doc_file.parent.mkdir(parents=True, exist_ok=True)
        with open(doc_file, "w", encoding="utf-8") as f:
            for d in documents:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

        if self.has_vector_db and self.collection is not None:
            print(f"   Indexing {len(documents)} documents with ChromaDB (batch)...")
            batch = 500
            for start in range(0, len(documents), batch):
                chunk = documents[start:start + batch]
                self.collection.add(
                    ids=[f"doc_{start + j}" for j in range(len(chunk))],
                    documents=[d.get("content", "") for d in chunk],
                    metadatas=[{
                        "source": d.get("source", "unknown"),
                        "title": d.get("title", ""),
                        "section": d.get("section", ""),
                        "article": str(d.get("article") or ""),
                    } for d in chunk],
                )
                print(f"     {min(start + batch, len(documents))}/{len(documents)}", end="\r", flush=True)
            print()

    def retrieve(self, query: str, top_k: int = 5, force_rag: bool = False) -> list:
        """混合检索（带门控）。"""
        if not force_rag:
            if classify_query_complexity(query, self.config) == "simple":
                return []

        expanded_query = expand_query(query, self.config)

        bm25_results = self.bm25.search(expanded_query, top_k=top_k * 2)

        dense_results = []
        if self.has_vector_db and self.collection is not None:
            try:
                chroma_results = self.collection.query(
                    query_texts=[expanded_query],
                    n_results=top_k * 2,
                )
                for i, (doc_id, content, metadata, distance) in enumerate(zip(
                    chroma_results["ids"][0],
                    chroma_results["documents"][0],
                    chroma_results["metadatas"][0],
                    chroma_results.get("distances", [[0]] * top_k * 2)[0],
                )):
                    dense_results.append({
                        "content": content,
                        "source": metadata.get("source", "unknown"),
                        "title": metadata.get("title", ""),
                        "score": 1.0 / (1.0 + distance),
                        "method": "dense",
                    })
            except Exception as e:
                print(f"   ⚠ Dense retrieval error: {e}")

        fused = self._rrf_fusion(bm25_results, dense_results, k=60)

        if self.use_reranker and classify_query_complexity(query, self.config) == "complex":
            fused = self._rerank(query, fused, top_k)

        return fused[:top_k]

    def _rrf_fusion(self, bm25_results: list, dense_results: list, k: int = 60) -> list:
        """加权 RRF 融合：按「内容」对齐 BM25 与 Dense 命中（同一 chunk 两边都中则分数相加）。

        实测（口语化法律查询）：BM25 字级 bigram 会返回大量不相关条文并占据 top 位，
        Dense（bge 语义）才找得到真正的法条。故：
        - 按 content 对齐（旧实现用 bm25_{idx}/dense_{rank} 两套独立 id，永远合并不了）；
        - Dense 权重 > BM25（口语→法言 靠语义，不靠字面）。
        """
        scores: dict = {}   # content -> rrf 分数
        infos: dict = {}    # content -> 文档信息
        bm25_w, dense_w = 0.5, 1.0

        for rank, (doc_idx, _bm25_score) in enumerate(bm25_results):
            if doc_idx >= len(self.documents):
                continue
            doc = self.documents[doc_idx]
            content = doc.get("content", "")
            if not content:
                continue
            scores[content] = scores.get(content, 0.0) + bm25_w / (k + rank + 1)
            infos[content] = {
                "content": content,
                "source": doc.get("source", "unknown"),
                "title": doc.get("title", ""),
                "method": "bm25",
            }

        for rank, result in enumerate(dense_results):
            content = result.get("content", "")
            if not content:
                continue
            scores[content] = scores.get(content, 0.0) + dense_w / (k + rank + 1)
            if content in infos:
                infos[content]["method"] = "hybrid"
            else:
                infos[content] = result

        fused = []
        for content, score in sorted(scores.items(), key=lambda kv: -kv[1]):
            info = dict(infos[content])
            info["score"] = score
            fused.append(info)
        return fused

    def _rerank(self, query: str, documents: list, top_k: int) -> list:
        """启发式重排：关键词覆盖 + 法条引用/时效加成。"""
        for doc in documents:
            content = doc.get("content", "")
            # 查询词覆盖加成
            key_terms = re.findall(r'[一-鿿]{2,}', query)
            term_matches = sum(1 for term in key_terms if term in content)
            # 法条/司法解释引用加成
            legal_boost = 1.5 if re.search(r'(?:法|条例|司法解释|第.{0,4}条)', content) else 1.0
            # 时效加成
            recency_boost = 1.0
            year_match = re.search(r'(20[0-9]{2})', content)
            if year_match:
                recency_boost = 1.1
            doc["score"] = doc.get("score", 0) * (1 + 0.2 * term_matches) * legal_boost * recency_boost

        documents.sort(key=lambda x: x.get("score", 0), reverse=True)
        return documents[:top_k]

    def search(self, query: str, top_k: int = 5) -> list:
        return self.retrieve(query, top_k)

    def format_context(self, documents: list, max_chars: int = 2000) -> str:
        """把检索到的文档格式化为 prompt 上下文。"""
        if not documents:
            return ""
        parts = ["【参考法律法规】"]
        total_chars = 0
        for doc in documents:
            source = doc.get("source", "法规")
            title = doc.get("title", "")
            content = doc.get("content", "")
            header = f"\n--- {title or source} ---\n" if title else f"\n--- {source} ---\n"
            body = content
            if total_chars + len(header) + len(body) > max_chars:
                remaining = max_chars - total_chars - len(header)
                if remaining > 100:
                    body = body[:remaining] + "..."
                else:
                    break
            parts.append(header + body)
            total_chars += len(header) + len(body)
        return "\n".join(parts)


# ──────────────────────────────────────────────
# 全局实例
# ──────────────────────────────────────────────

_retriever = None


def get_retriever(persist_dir: str = "./data/vector_db") -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(persist_dir=persist_dir)
    return _retriever


# ──────────────────────────────────────────────
# CLI Demo
# ──────────────────────────────────────────────

if __name__ == "__main__":
    sample_docs = [
        {
            "content": "《中华人民共和国劳动合同法》第四十七条：经济补偿按劳动者在本单位工作的年限，每满一年支付一个月工资的标准向劳动者支付。六个月以上不满一年的，按一年计算；不满六个月的，向劳动者支付半个月工资的经济补偿。",
            "source": "中华人民共和国劳动合同法",
            "title": "经济补偿标准",
            "section": "解除和终止",
        },
        {
            "content": "《中华人民共和国劳动合同法》第八十七条：用人单位违反本法规定解除或者终止劳动合同的，应当依照本法第四十七条规定的经济补偿标准的二倍向劳动者支付赔偿金。",
            "source": "中华人民共和国劳动合同法",
            "title": "违法解除赔偿金",
            "section": "法律责任",
        },
        {
            "content": "《最高人民法院关于审理劳动争议案件适用法律问题的解释（二）》（2025）：竞业限制未约定经济补偿的，劳动者履行了竞业限制义务的，可以要求用人单位按月支付经济补偿。",
            "source": "劳动争议司法解释（二）",
            "title": "竞业限制补偿",
            "section": "竞业限制",
        },
        {
            "content": "《中华人民共和国劳动争议调解仲裁法》第二十七条：劳动争议申请仲裁的时效期间为一年。仲裁时效期间从当事人知道或者应当知道其权利被侵害之日起计算。",
            "source": "劳动争议调解仲裁法",
            "title": "仲裁时效",
            "section": "时效",
        },
    ]

    retriever = HybridRetriever()
    retriever.index(sample_docs)

    test_queries = [
        "被辞退能拿多少赔偿",
        "竞业限制没有补偿有效吗",
        "劳动仲裁时效多久",
        "什么是法律常识",  # simple → gated
    ]

    print("\n" + "=" * 60)
    print("Hybrid RAG Retrieval Demo")
    print("=" * 60)

    for query in test_queries:
        complexity = classify_query_complexity(query, retriever.config)
        print(f"\n🔍 Query: {query}")
        print(f"   Complexity: {complexity}")
        results = retriever.retrieve(query, top_k=3)
        if not results:
            print(f"   → GATED: Simple question, no RAG needed")
        else:
            for i, doc in enumerate(results):
                print(f"   [{i+1}] {doc['title']}: {doc['content'][:60]}... (score={doc.get('score', 0):.3f})")
