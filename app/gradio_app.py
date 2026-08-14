"""
gradio_app.py — LexiCare 统一 Demo 应用。

整合全部模块到一个 Gradio 界面：
  - Safety Guard（输入过滤 + 输出审核）
  - Intent Router（查询分类 + prompt 选择）
  - RAG Retriever（法条检索）
  - Hallucination Detector（输出安全复核）
  - 模型推理（base / sft / sft+dpo）

用法：
    python app/gradio_app.py
    python app/gradio_app.py --model_path Qwen/Qwen3-8B
    python app/gradio_app.py --lora_sft outputs/lora_weights/lexicare-sft-XXX
"""

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.domain_config import get_domain
from app.safety_guard import SafetyGuard
from app.intent_router import IntentRouter, ConversationContext
from app.hallucination_detector import HallucinationDetector, quick_hallucination_check

# ──────────────────────────────────────────────
# 领域配置
# ──────────────────────────────────────────────

SYSTEM_PROMPT = get_domain().default_system_prompt
PRESET_QUESTIONS = {k: list(v) for k, v in get_domain().preset_questions.items()}


# ──────────────────────────────────────────────
# Model Manager
# ──────────────────────────────────────────────

class ModelManager:
    """管理多个 LoRA 变体的加载/卸载。"""

    def __init__(self, base_model: str, lora_paths: dict = None):
        self.base_model_name = base_model
        self.lora_paths = lora_paths or {}
        self.tokenizer = None
        self.base_model = None
        self.lora_models = {}

    def load_tokenizer(self):
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_model(self, variant: str = "base"):
        if variant == "base":
            return self._load_base()
        return self._load_lora(variant)

    def _load_base(self):
        if self.base_model is not None:
            return self.base_model
        from transformers import AutoModelForCausalLM
        self._clear_gpu()
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.base_model.eval()
        return self.base_model

    def _load_lora(self, name: str):
        if name in self.lora_models:
            return self.lora_models[name]
        path = self.lora_paths.get(name)
        if not path:
            return self._load_base()

        from transformers import AutoModelForCausalLM
        from peft import PeftModel
        self._clear_gpu()
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, path)
        model = model.merge_and_unload()
        model.eval()
        self.lora_models[name] = model
        return model

    def _clear_gpu(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(self, query: str, variant: str = "base",
                 system_prompt: str = None, max_tokens: int = 400,
                 temperature: float = 0.7) -> tuple:
        self.load_tokenizer()
        prompt_text = system_prompt or SYSTEM_PROMPT
        formatted = (
            f"<|im_start|>system\n{prompt_text}<|im_end|>\n"
            f"<|im_start|>user\n{query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        model = self.get_model(variant)
        inputs = self.tokenizer(formatted, return_tensors="pt").to(model.device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_tokens,
                temperature=temperature, top_p=0.9, do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip(), elapsed


# ──────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────

CSS = """
.gradio-container { max-width: 1400px !important; }
.header { text-align: center; margin-bottom: 20px; border-bottom: 2px solid #1a73e8; padding-bottom: 16px; }
.header h1 { color: #1a73e8; font-size: 2em; margin: 0; }
.header p { color: #666; font-size: 1.1em; margin-top: 4px; }
.safety-badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: bold; }
.safety-safe { background: #d4edda; color: #155724; }
.safety-warn { background: #fff3cd; color: #856404; }
.safety-danger { background: #f8d7da; color: #721c24; }
.intent-tag { font-size: 0.85em; padding: 3px 10px; border-radius: 15px; background: #e8f0fe; color: #1a73e8; }
.rag-tag { font-size: 0.8em; padding: 2px 8px; border-radius: 10px; }
.rag-on { background: #d4edda; color: #155724; }
.rag-off { background: #f0f0f0; color: #666; }
"""


def create_ui(model_manager: ModelManager = None):
    """构建统一 Gradio 界面。"""
    guard = SafetyGuard()
    router = IntentRouter()
    conv_ctx = ConversationContext()

    with gr.Blocks(css=CSS, title="LexiCare 法律咨询助手") as demo:
        gr.HTML("""
        <div class="header">
            <h1>⚖️ LexiCare — 法律咨询智能助手</h1>
            <p>劳动争议 · 合同 · 婚姻家事 · 刑事 · 公司 · 知识产权 · 行政</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📝 输入问题")
                question_input = gr.Textbox(
                    label="输入您的法律问题",
                    placeholder="例如：被公司辞退能拿多少赔偿？",
                    lines=3,
                )
                with gr.Row():
                    submit_btn = gr.Button("🔍 咨询", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ 清空", size="lg")

                gr.Markdown("### 📋 预设测试题")
                dept_dropdown = gr.Dropdown(
                    label="问题类别",
                    choices=list(PRESET_QUESTIONS.keys()),
                    value=list(PRESET_QUESTIONS.keys())[0],
                )
                question_dropdown = gr.Dropdown(
                    label="选择预设问题",
                    choices=PRESET_QUESTIONS[list(PRESET_QUESTIONS.keys())[0]],
                )
                random_btn = gr.Button("🎲 随机出题", size="sm")

                gr.Markdown("### ⚙️ 模型选择")
                model_variant = gr.Radio(
                    choices=["base", "sft", "sft+dpo"],
                    value="base",
                    label="模型版本",
                )

                with gr.Accordion("🔧 高级选项", open=False):
                    enable_rag = gr.Checkbox(label="启用 RAG 检索", value=True)
                    enable_safety = gr.Checkbox(label="启用安全护栏", value=True)
                    temperature_slider = gr.Slider(
                        label="Temperature", minimum=0.1, maximum=1.5, value=0.7, step=0.1
                    )
                    max_tokens_slider = gr.Slider(
                        label="最大输出长度", minimum=100, maximum=800, value=400, step=50
                    )

            with gr.Column(scale=2):
                with gr.Row():
                    intent_badge = gr.HTML("")
                    safety_badge = gr.HTML("")
                    rag_status = gr.HTML("")
                    time_display = gr.Textbox(label="⏱️ 生成耗时", value="--", scale=1)

                gr.Markdown("### 💬 助手回答")
                response_output = gr.Textbox(
                    label="", lines=16, max_lines=25, placeholder="回答将在此显示...",
                )

                with gr.Accordion("📊 详细分析", open=False):
                    with gr.Row():
                        hallu_result = gr.Textbox(label="幻觉检测", lines=4)
                        confidence_score = gr.Textbox(label="置信度评估", lines=2, scale=1)
                    with gr.Row():
                        intent_info = gr.Textbox(label="意图分析", lines=3)
                        rag_info = gr.Textbox(label="RAG 检索结果", lines=4)

        # ── Event Handlers ──

        def update_questions(department):
            return gr.update(choices=PRESET_QUESTIONS.get(department, []))

        dept_dropdown.change(update_questions, dept_dropdown, question_dropdown)

        def random_question():
            import random
            all_qs = []
            for qs in PRESET_QUESTIONS.values():
                all_qs.extend(qs)
            return random.choice(all_qs)

        random_btn.click(random_question, None, question_input)

        def set_preset(q):
            return q

        question_dropdown.change(set_preset, question_dropdown, question_input)

        def process_query(query, variant, use_rag, use_safety, temperature, max_tokens):
            if not query or not query.strip():
                return "", "", "", "", "", "", "", "", "⏱️ --"

            t_start = time.time()

            if use_safety:
                input_result = guard.check_input(query)
                if not input_result.safe:
                    time_elapsed = f"⏱️ {time.time() - t_start:.1f}s"
                    return (
                        input_result.fallback_response,
                        f"<span class='intent-tag'>🚫 {input_result.category}</span>",
                        "<span class='safety-badge safety-safe'>已拦截</span>",
                        "<span class='rag-tag rag-off'>RAG 未触发</span>",
                        time_elapsed,
                        "N/A (输入已拦截)",
                        f"拒绝原因: {input_result.category}",
                        "N/A",
                        "N/A",
                    )
                intent = input_result.category
            else:
                intent = "general"

            decision = router.route(query)
            system_prompt = decision.system_prompt_override or SYSTEM_PROMPT

            rag_docs = []
            rag_summary = "未启用 RAG 或不需要"
            if use_rag and decision.needs_rag:
                try:
                    from app.rag_retriever import get_retriever
                    retriever = get_retriever()
                    rag_query = decision.rag_query if decision.rag_query else query
                    rag_docs = retriever.retrieve(rag_query, top_k=3)
                    rag_summary = f"检索到 {len(rag_docs)} 篇相关文档" if rag_docs else "未找到相关文档"
                except Exception as e:
                    rag_summary = f"RAG 检索异常: {e}"

            if model_manager is not None:
                response, gen_time = model_manager.generate(
                    query, variant=variant,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            else:
                response = _demo_response(query, intent, rag_docs)
                gen_time = 0.01

            time_elapsed = f"⏱️ {gen_time:.1f}s"
            hallu_report = "N/A"
            confidence = 0.8

            if use_safety:
                output_result = guard.check_output(
                    query, response, category=intent, retrieved_docs=rag_docs,
                )
                response = output_result.response
                confidence = output_result.confidence
                try:
                    detector = HallucinationDetector()
                    hallu = detector.check(response, rag_docs, question=query)
                    hallu_report = hallu.summary()
                except Exception:
                    hallu_report = quick_hallucination_check(response)

            intent_label = router.intent_label(intent) if intent != "general" else "📋 通用"
            intent_html = f"<span class='intent-tag'>{intent_label}</span>"

            if use_safety:
                if confidence >= 0.7:
                    safety_html = "<span class='safety-badge safety-safe'>✅ 安全</span>"
                elif confidence >= 0.5:
                    safety_html = "<span class='safety-badge safety-warn'>⚠️ 注意</span>"
                else:
                    safety_html = "<span class='safety-badge safety-danger'>🔴 低置信度</span>"
            else:
                safety_html = "<span class='safety-badge' style='background:#f0f0f0'>⚪ 未启用</span>"

            if rag_docs:
                rag_html = "<span class='rag-tag rag-on'>📚 RAG ({})</span>".format(len(rag_docs))
            else:
                rag_html = "<span class='rag-tag rag-off'>RAG 未使用</span>"

            intent_detail = (
                f"意图: {decision.intent} (置信度 {decision.confidence:.0%})\n"
                f"系统提示: {system_prompt[:100]}..."
            )
            rag_detail = rag_summary
            if rag_docs:
                for i, doc in enumerate(rag_docs):
                    rag_detail += f"\n  [{i+1}] {doc.get('title', '?')[:40]}"

            confidence_detail = f"置信度: {confidence:.0%} | 安全状态: 已审核"

            return (
                response, intent_html, safety_html, rag_html, time_elapsed,
                hallu_report, intent_detail, rag_detail, confidence_detail,
            )

        submit_btn.click(
            process_query,
            [question_input, model_variant, enable_rag, enable_safety,
             temperature_slider, max_tokens_slider],
            [response_output, intent_badge, safety_badge, rag_status,
             time_display, hallu_result, intent_info, rag_info, confidence_score],
        )

        clear_btn.click(
            lambda: ("", "", "", "", "⏱️ --", "", "", "", ""),
            None,
            [question_input, response_output, intent_badge, safety_badge,
             rag_status, time_display, hallu_result, intent_info, rag_info, confidence_score],
        )

        gr.Markdown("""
        ---
        ⚠️ **免责声明**：LexiCare 是技术学习与研究项目，所有回答仅供参考，
        不构成法律意见。如有具体案件，请咨询执业律师。
        """)

    return demo


def _demo_response(query: str, intent: str, rag_docs: list = None) -> str:
    """无模型加载时的演示响应。"""
    return (
        "您好！我是 LexiCare 法律咨询助手。\n\n"
        "我可以帮您了解：\n"
        "- 📋 劳动争议（辞退赔偿、竞业限制、加班费、社保工伤）\n"
        "- 📄 合同纠纷（违约、借款、租赁）\n"
        "- 👨‍👩‍👧 婚姻家事与继承\n"
        "- ⚖️ 刑事、公司、知识产权、行政等一般法律知识\n\n"
        "（这是一条演示回复。加载模型后可获得真实的法律分析。）"
    )


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LexiCare Gradio Demo")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--lora_sft", type=str, default=None)
    parser.add_argument("--lora_dpo", type=str, default=None)
    parser.add_argument("--demo_mode", action="store_true",
                        help="Run without loading model (UI test only)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    model_manager = None
    if not args.demo_mode:
        lora_paths = {}
        if args.lora_sft:
            lora_paths["sft"] = args.lora_sft
        else:
            sft_dirs = sorted(Path("outputs/lora_weights").glob("lexicare-sft-*"))
            if sft_dirs:
                lora_paths["sft"] = str(sft_dirs[-1])
                print(f"Auto-detected SFT adapter: {lora_paths['sft']}")

        if args.lora_dpo:
            lora_paths["sft+dpo"] = args.lora_dpo
        else:
            dpo_dirs = sorted(Path("outputs/lora_weights").glob("lexicare-safedpo-*"))
            if dpo_dirs:
                lora_paths["sft+dpo"] = str(dpo_dirs[-1])
                print(f"Auto-detected DPO adapter: {lora_paths['sft+dpo']}")

        try:
            print(f"Loading model: {args.model_path}")
            model_manager = ModelManager(args.model_path, lora_paths)
            model_manager.load_tokenizer()
            print("Model loaded successfully")
        except Exception as e:
            print(f"⚠ Failed to load model: {e}")
            print("Falling back to demo mode")
    else:
        print("Demo mode: no model loaded")

    demo = create_ui(model_manager)
    demo.queue(default_concurrency_limit=3)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
