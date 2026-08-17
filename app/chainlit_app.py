"""
chainlit_app.py — LexiCare 统一 Chainlit 前端（法律咨询 + 文档问答）。

两种模式：
  1. 法律咨询（默认）：法条 RAG（HybridRetriever）+ Qwen3-4B + law-lora-r8，流式输出 + 检索来源。
  2. 文档问答：在输入框附带上传 PDF/txt/md → 实时分块索引（DocumentQA）→ 针对文档问答。

复用 app/ 模块（domain_config / rag_retriever / document_qa / safety_guard），
模型加载同 chat.py（4-bit + enable_thinking=False）。

用法（在项目根目录）：
    chainlit run app/chainlit_app.py -w
"""

import asyncio
import os
import sys
import threading
from pathlib import Path

# 注意：必须用 append 而非 insert(0)。Chainlit 的 load_module 会先把目标文件所在目录
# （app/）插到 sys.path[0]，执行完模块后再 sys.path.pop(0) 清理。若这里 insert(0) 项目根，
# 会被 Chainlit 的 pop(0) 误删，导致 app/ 留在 sys.path[0]，`import app` 解析成 app/app.py
# （同名冲突）而非 app 包。用 append 则项目根存活、app/ 被正确清掉。
sys.path.append(str(Path(__file__).parent.parent))

import chainlit as cl
import torch

# ──────────────────────────────────────────────
# 配置（可用环境变量覆盖）
# ──────────────────────────────────────────────

BASE_MODEL = os.environ.get("LEXICARE_MODEL", "Qwen/Qwen3-4B")
ADAPTER = os.environ.get("LEXICARE_ADAPTER", "outputs/lora_weights/law-lora-r8-20260814-1732")

RAG_INSTRUCTION = (
    "根据以下检索到的法条回答问题。若检索结果与问题相关，引用其中真实存在的条文（《法名》第X条）；"
    "若检索结果不足以直接回答，给出一般性法律说明、说明依据不足，并建议核实或咨询执业律师；不得编造法条。"
)

GENERIC_SYSTEM = "你是一个文档问答助手，根据用户提供的文档内容回答问题，引用具体信息，用中文回答。"
GENERIC_RAG = (
    "根据以下从文档中检索到的内容回答问题。优先引用检索内容作答；"
    "若检索内容不足以回答，明确说明「文档中没有足够信息」，不要编造。"
)

DOC_SUFFIXES = (".pdf", ".txt", ".md")


# ──────────────────────────────────────────────
# 模型（全局单例，懒加载，线程安全）
# ──────────────────────────────────────────────

_model = None
_tokenizer = None
_model_lock = threading.Lock()


def _load_model():
    """懒加载 Qwen3-4B + law-lora-r8（4-bit，关闭思考模式）。"""
    global _model, _tokenizer
    with _model_lock:
        if _model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        )
        _model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, trust_remote_code=True, device_map="auto", quantization_config=bnb
        )
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        if Path(ADAPTER).exists():
            _model = PeftModel.from_pretrained(_model, ADAPTER)
        if hasattr(_model.generation_config, "enable_thinking"):
            _model.generation_config.enable_thinking = False
        _model.eval()


async def _astream(streamer):
    """把 TextIteratorStreamer 的同步队列转成 async 迭代器（不阻塞事件循环）。"""
    while True:
        value = await asyncio.to_thread(streamer.text_queue.get)
        if value == streamer.stop_signal:  # None 哨兵
            break
        yield value


async def stream_answer(msg: cl.Message, system_prompt: str, user_prompt: str,
                        max_new_tokens: int = 512):
    """流式生成并把 token 写入 msg；防御性处理 Qwen3 思考标记泄漏。"""
    from transformers import TextIteratorStreamer
    _load_model()

    formatted = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        # 关闭 Qwen3 思考模式：chat_template 里 enable_thinking=False 靠这个空 think 块，
        # 手写 prompt 必须显式补上，否则 generation_config.enable_thinking=False 不生效、会漏 </think>
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    inputs = _tokenizer(formatted, return_tensors="pt").to(_model.device)
    streamer = TextIteratorStreamer(_tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.15,
        pad_token_id=_tokenizer.pad_token_id,
        eos_token_id=_tokenizer.eos_token_id,
    )
    thread = threading.Thread(target=_model.generate, kwargs=gen_kwargs)
    thread.start()

    buf = ""
    started = False
    async for token in _astream(streamer):
        if started:
            await msg.stream_token(token)
            continue
        buf += token
        if "</think>" in buf:
            # 思考标记泄漏：丢弃其之前内容，输出其后残留
            tail = buf.split("</think>", 1)[1]
            buf = ""
            started = True
            if tail:
                await msg.stream_token(tail)
        elif len(buf) > 64:
            # 无思考模式（enable_thinking=False 已保证），正常输出
            started = True
            await msg.stream_token(buf)
            buf = ""
    if not started and buf:
        await msg.stream_token(buf.strip())
    thread.join()


# ──────────────────────────────────────────────
# 来源元素（检索到的法条 / 文档片段，展示在消息侧栏）
# ──────────────────────────────────────────────

def source_elements(docs: list, kind: str) -> list:
    elements = []
    for i, d in enumerate(docs or [], 1):
        title = (d.get("title") or d.get("source") or "来源")[:24]
        content = (d.get("content") or "").strip()[:600]
        elements.append(cl.Text(name=f"{kind} {i} · {title}", content=content, display="side"))
    return elements


# ──────────────────────────────────────────────
# 回答：法律咨询 / 文档问答
# ──────────────────────────────────────────────

async def answer_legal(query: str):
    from app.rag_retriever import get_retriever
    from app.domain_config import get_domain

    retriever = get_retriever()
    docs = retriever.retrieve(query, top_k=3)
    context = retriever.format_context(docs)
    ctx_block = context if context else "【参考法律法规】未检索到直接相关条文。"

    msg = cl.Message(content="", elements=source_elements(docs, "法条"))
    await stream_answer(
        msg,
        system_prompt=get_domain().default_system_prompt,
        user_prompt=f"{RAG_INSTRUCTION}\n{ctx_block}\n\n问题：{query}",
    )
    await msg.send()


async def answer_doc(query: str, docqa):
    docs = docqa.retrieve(query, top_k=3)
    context = docqa.format_context(docs)
    ctx = context if context else "【文档内容】未检索到相关内容。"

    msg = cl.Message(content="", elements=source_elements(docs, "文档"))
    await stream_answer(
        msg,
        system_prompt=GENERIC_SYSTEM,
        user_prompt=f"{GENERIC_RAG}\n{ctx}\n\n问题：{query}",
    )
    await msg.send()


# ──────────────────────────────────────────────
# Chainlit 生命周期
# ──────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("docqa", None)  # None = 法律咨询模式
    await cl.Message(
        content=(
            "你好，我是 **LexiCare 法律咨询助手** ⚖️\n\n"
            "**两种用法：**\n"
            "1. 直接输入法律问题（劳动 / 合同 / 婚姻家事 / 刑事 / 公司 / 知产 / 行政）\n"
            "2. 点击输入框的 📎 上传一份 PDF / txt / md 文档（如合同），再针对文档提问\n\n"
            "> 首次提问会自动加载模型（约 1 分钟）。回答末尾附「检索来源」。\n"
            "> 输入 `/reset` 可从文档问答回到法律咨询模式。"
        )
    ).send()


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(label="被辞退能拿多少赔偿", message="被公司辞退能拿多少赔偿？"),
        cl.Starter(label="劳动仲裁时效多久", message="劳动仲裁的时效是多久？"),
        cl.Starter(label="交了定金不买能退吗", message="交了定金不买了能退吗？"),
        cl.Starter(label="遗嘱没公证还有效吗", message="遗嘱没有公证还有效吗？"),
    ]


@cl.on_message
async def on_message(message: cl.Message):
    query = (message.content or "").strip()
    docqa = cl.user_session.get("docqa")

    # 命令：/reset 回到法律咨询
    if query.startswith("/reset"):
        cl.user_session.set("docqa", None)
        await cl.Message(content="✅ 已回到法律咨询模式。重新上传文档可再切换回文档问答。").send()
        return

    # 检测上传的文件 → 切换文档问答
    attached = None
    for el in (message.elements or []):
        if isinstance(el, cl.File):
            attached = el
            break

    if attached is not None:
        path = getattr(attached, "path", None)
        if not path or not Path(path).exists() or Path(path).suffix.lower() not in DOC_SUFFIXES:
            await cl.Message(content="⚠️ 仅支持 PDF / txt / md 文档。").send()
            return
        ingest = cl.Message(content=f"🔍 正在解析文档 `{Path(path).name}` …")
        await ingest.send()
        try:
            from app.document_qa import DocumentQA
            docqa = DocumentQA(path)
            cl.user_session.set("docqa", docqa)
            await cl.Message(
                content=f"✅ 文档已摄入：**{docqa.n_chunks}** 个分块。现在可以针对这份文档提问。"
            ).send()
        except Exception as e:
            await cl.Message(content=f"❌ 文档摄入失败：{e}").send()
            return
        if not query:  # 只上传、没提问
            return

    if not query:
        await cl.Message(content="请输入问题，或上传一份文档。").send()
        return

    # 回答
    if docqa is not None:
        # 文档问答：不走法律领域护栏（问的是文档内容，可能是合同/行程等非法律问题）
        await answer_doc(query, docqa)
    else:
        # 输入安全护栏（仅法律咨询模式）
        from app.safety_guard import SafetyGuard
        guard = SafetyGuard()
        input_result = guard.check_input(query)
        if not input_result.safe:
            await cl.Message(content=input_result.fallback_response).send()
            return
        await answer_legal(query)


if __name__ == "__main__":
    print("请用 `chainlit run app/chainlit_app.py -w` 启动（在项目根目录）。")
