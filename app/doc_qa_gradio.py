"""doc_qa_gradio.py — 通用文档问答 Gradio UI（上传任意 PDF/txt 实时问答）。

用法：
    python app/doc_qa_gradio.py
    python app/doc_qa_gradio.py --port 7861 --share
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
import torch

from app.document_qa import DocumentQA

BASE_MODEL = "Qwen/Qwen3-4B"
ADAPTER = "outputs/lora_weights/law-lora-r8-20260814-1732"
GENERIC_SYSTEM = "你是一个文档问答助手，根据用户提供的文档内容回答问题，引用具体信息，用中文回答。"
GENERIC_RAG = ("根据以下从文档中检索到的内容回答问题。优先引用检索内容作答；"
               "若检索内容不足以回答，明确说明「文档中没有足够信息」，不要编造。")


class DocQAPipeline:
    """文档问答流水线：摄入文档 → 检索 → 生成（模型懒加载）。"""

    def __init__(self, base_only: bool = True):
        self.base_only = base_only
        self.model = None
        self.tokenizer = None
        self.docqa = None

    def _load_model(self):
        if self.model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        self.model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, trust_remote_code=True,
                                                          device_map="auto", quantization_config=bnb)
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if not self.base_only and Path(ADAPTER).exists():
            self.model = PeftModel.from_pretrained(self.model, ADAPTER)
        if hasattr(self.model.generation_config, "enable_thinking"):
            self.model.generation_config.enable_thinking = False
        self.model.eval()

    def ingest(self, file_obj):
        """摄入上传的文档。Gradio File 可能返回 str 路径或 dict。"""
        if isinstance(file_obj, dict):
            path = file_obj.get("name") or file_obj.get("path")
        elif isinstance(file_obj, str):
            path = file_obj
        else:
            path = getattr(file_obj, "name", None)
        if not path:
            return "❌ 无法读取文件，请重新上传"
        try:
            self.docqa = DocumentQA(path)
            return f"✅ 文档已摄入：{self.docqa.n_chunks} 个分块\n📄 {Path(path).name}"
        except Exception as e:
            return f"❌ 摄入失败：{e}"

    def ask(self, question):
        if not question or not question.strip():
            return "请输入问题", ""
        if self.docqa is None:
            return "请先上传文档", ""
        self._load_model()
        docs = self.docqa.retrieve(question, top_k=3)
        context = self.docqa.format_context(docs)
        ctx = context if context else "【文档内容】未检索到相关内容。"
        prompt = (f"<|im_start|>system\n{GENERIC_SYSTEM}<|im_end|>\n"
                  f"<|im_start|>user\n{GENERIC_RAG}\n{ctx}\n\n问题：{question}<|im_end|>\n"
                  f"<|im_start|>assistant\n")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=400, do_sample=True,
                                      temperature=0.7, top_p=0.9, repetition_penalty=1.15)
        ans = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if "</think>" in ans:
            ans = ans.split("</think>")[-1].strip()
        sources = "\n\n".join(f"[{i+1}] {d['content'][:120]}" for i, d in enumerate(docs))
        return ans, sources


def create_ui():
    pipe = DocQAPipeline(base_only=True)
    with gr.Blocks(title="通用文档问答") as demo:
        gr.Markdown("## 📄 通用文档问答\n上传任意 PDF / txt，实时解析分块 + RAG 问答（BM25 + Dense + cross-encoder 精排）")
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(label="上传文档", file_types=[".pdf", ".txt", ".md"])
                ingest_status = gr.Textbox(label="摄入状态", lines=2)
                question = gr.Textbox(label="问题", lines=2, placeholder="针对这份文档提问...")
                submit_btn = gr.Button("提问", variant="primary", size="lg")
            with gr.Column(scale=2):
                answer = gr.Textbox(label="回答", lines=16)
                sources = gr.Textbox(label="检索到的内容片段", lines=8)
        file_input.change(pipe.ingest, file_input, ingest_status)
        submit_btn.click(pipe.ask, question, [answer, sources])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    create_ui().launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
