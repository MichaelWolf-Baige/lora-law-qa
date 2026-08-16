"""document_qa.py — 通用文档问答（吸收自 DocQA 的实时摄入 + 复用 LexiCare 检索）。

流程：上传任意 PDF/txt → 实时解析分块 → 混合检索（BM25 + Dense + cross-encoder 精排）→ 生成。

与法条库 RAG 共用同一套 HybridRetriever，只是换领域配置 + 临时索引目录，不污染法条库。
"""

import tempfile
from dataclasses import replace

from app.document_ingestion import load_document
from app.domain_config import get_domain
from app.rag_retriever import HybridRetriever


def general_doc_domain():
    """通用文档检索的领域配置：复用 legal 的 reranker 参数，但清空法律同义词/门控，
    任意查询都触发 cross-encoder 精排。"""
    legal = get_domain()
    rag = replace(
        legal.rag,
        collection_name="doc_qa",        # 独立 collection，不污染法条库
        synonyms={},                      # 通用文档不需要法律同义词
        simple_patterns=(),               # 不跳过检索（简单问题也走 RAG）
        complex_patterns=(r".*",),        # 任意查询都触发 cross-encoder 精排
    )
    return replace(legal, rag=rag)


class DocumentQA:
    """通用文档问答器：摄入任意文档，复用 HybridRetriever 检索。"""

    def __init__(self, doc_path, persist_dir=None, use_dense=True):
        chunks = load_document(doc_path)
        if not chunks:
            raise ValueError(f"文档为空或无法解析: {doc_path}")
        self.doc_path = str(doc_path)
        self.n_chunks = len(chunks)
        # 临时索引目录（默认 tempdir，用完丢弃；不污染法条库 data/vector_db）
        self.persist_dir = persist_dir or tempfile.mkdtemp(prefix="docqa_")
        self.retriever = HybridRetriever(
            config=general_doc_domain(),
            persist_dir=self.persist_dir,
            use_dense=use_dense,
            use_reranker=True,
        )
        self.retriever.index(chunks)

    def retrieve(self, question, top_k=3):
        return self.retriever.retrieve(question, top_k=top_k)

    def format_context(self, docs, max_chars=2000):
        return self.retriever.format_context(docs, max_chars=max_chars)
