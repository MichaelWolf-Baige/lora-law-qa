"""document_ingestion.py — 通用文档摄入（吸收自 DocQA 的实时解析能力）。

支持任意 PDF / txt / md 文档：解析 → 句子边界分块 → 产出与法条库同 schema 的 chunk。

与法条库（预分割固定语料）的区别：这里是「实时摄入」——用户上传任意文档即席解析分块。
"""

import re
from pathlib import Path


def parse_pdf(path):
    """PyMuPDF 解析 PDF，返回每页文本列表 [{'page': n, 'text': str}]。"""
    import fitz  # PyMuPDF
    doc = fitz.open(str(path))
    pages = [{"page": i + 1, "text": page.get_text()} for i, page in enumerate(doc)]
    doc.close()
    return pages


def parse_text(path):
    """解析纯文本 / markdown，按整篇作为一页。"""
    return [{"page": 1, "text": Path(path).read_text(encoding="utf-8")}]


def _split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[。！？；\.\!\?\;\n])', text) if s.strip()]


def chunk_text(text, chunk_size=512, overlap=128, source="", title="", section=""):
    """句子边界分块，带 overlap。返回法条同 schema 的 chunk dict 列表。"""
    sentences = _split_sentences(text)
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) > chunk_size and current:
            chunks.append({"content": current.strip(), "source": source,
                           "title": title, "section": section, "article": None})
            # overlap：保留末尾几个句子，避免语义截断
            tail = ""
            for s in reversed(sentences[:sentences.index(sent)]):
                if len(tail) + len(s) <= overlap:
                    tail = s + tail
                else:
                    break
            current = tail
        current += sent
    if current.strip():
        chunks.append({"content": current.strip(), "source": source,
                       "title": title, "section": section, "article": None})
    return chunks


def load_document(path, chunk_size=512, overlap=128):
    """加载任意文档，解析 + 分块，返回 chunk dict 列表（法条同 schema）。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"文档不存在: {path}")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pages = parse_pdf(path)
    elif suffix in (".txt", ".md"):
        pages = parse_text(path)
    else:
        raise ValueError(f"不支持的文件类型 {suffix}（仅支持 .pdf/.txt/.md）")

    title = path.stem
    chunks = []
    for p in pages:
        chunks.extend(chunk_text(p["text"], chunk_size, overlap,
                                 source=str(path), title=title))
    return chunks
