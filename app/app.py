"""
app.py — Gradio interactive comparison UI for LoRA Legal QA.

Features:
  - Side-by-side base vs LoRA model comparison
  - Preset legal questions organized by category
  - Real-time hallucination detection indicators
  - Batch comparison mode (run all presets at once)
  - Export comparison report

Usage:
    python app.py --base_model Qwen/Qwen3-8B --lora_path outputs/lora_weights/XXX
    python app.py --lora_only  # Only show LoRA model (for simpler UI)

Memory management:
    Only ONE model is loaded at a time. The base model runs first,
    then is unloaded before the LoRA model loads. This keeps VRAM
    usage under 4GB for a 1.5B model.
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from app.domain_config import get_domain

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

SYSTEM_PROMPT = get_domain().default_system_prompt


def format_prompt(question: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ──────────────────────────────────────────────
# Model loading (with memory management)
# ──────────────────────────────────────────────

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class ModelManager:
    """Manages loading/unloading of models to stay within 8GB VRAM."""

    def __init__(self, base_model_name: str, lora_path: str = None):
        self.base_model_name = base_model_name
        self.lora_path = lora_path
        self.tokenizer = None
        self.base_model = None
        self.lora_model = None
        self._load_tokenizer()

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_base(self):
        if self.base_model is not None:
            return
        clear_gpu()
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.base_model.eval()

    def _load_lora(self):
        if self.lora_model is not None:
            return
        clear_gpu()
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        if self.lora_path:
            model = PeftModel.from_pretrained(model, self.lora_path)
            model = model.merge_and_unload()
        model.eval()
        self.lora_model = model

    def generate_base(self, question: str, max_tokens: int = 300) -> str:
        self._load_base()
        # Free lora to save memory
        if self.lora_model is not None:
            del self.lora_model
            self.lora_model = None
            clear_gpu()

        prompt = format_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.base_model.device)
        with torch.no_grad():
            outputs = self.base_model.generate(
                **inputs, max_new_tokens=max_tokens,
                temperature=0.7, top_p=0.9, do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def generate_lora(self, question: str, max_tokens: int = 300) -> str:
        self._load_lora()
        # Free base to save memory
        if self.base_model is not None:
            del self.base_model
            self.base_model = None
            clear_gpu()

        prompt = format_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.lora_model.device)
        with torch.no_grad():
            outputs = self.lora_model.generate(
                **inputs, max_new_tokens=max_tokens,
                temperature=0.7, top_p=0.9, do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


# ──────────────────────────────────────────────
# Preset questions (organized by department)
# ──────────────────────────────────────────────

PRESET_QUESTIONS = {k: list(v) for k, v in get_domain().preset_questions.items()}


def load_test_cases(test_cases_dir: str = "data/test_cases") -> list:
    """Load test cases from JSON files, or use presets as fallback."""
    all_questions = []
    tc_dir = Path(test_cases_dir)
    if tc_dir.exists():
        dept_files = sorted(tc_dir.glob("*.json"))
        for df in dept_files:
            if df.name == "all_departments.json":
                continue
            with open(df, "r", encoding="utf-8") as f:
                cases = json.load(f)
            for c in cases:
                all_questions.append({
                    "question": c["question"],
                    "department": c.get("department", df.stem),
                    "reference": c.get("answer", ""),
                })

    # Fall back to presets
    if not all_questions:
        for dept, questions in PRESET_QUESTIONS.items():
            for q in questions:
                all_questions.append({
                    "question": q,
                    "department": dept,
                    "reference": "",
                })

    return all_questions


# ──────────────────────────────────────────────
# Gradio Interface
# ──────────────────────────────────────────────

CSS = """
.gradio-container { max-width: 1400px !important; }
.header { text-align: center; margin-bottom: 20px; }
.header h1 { color: #1a73e8; font-size: 2em; }
.base-panel { border: 2px solid #ff6b6b; border-radius: 12px; padding: 16px; }
.lora-panel { border: 2px solid #51cf66; border-radius: 12px; padding: 16px; }
.base-label { color: #ff6b6b; font-weight: bold; font-size: 1.2em; }
.lora-label { color: #51cf66; font-weight: bold; font-size: 1.2em; }
.hallucination-warn { background: #fff3cd; padding: 8px; border-radius: 8px; font-size: 0.85em; }
.hallucination-ok { background: #d4edda; padding: 8px; border-radius: 8px; font-size: 0.85em; }
"""


from app.hallucination_detector import quick_hallucination_check


def create_ui(model_manager: ModelManager, test_cases: list):
    """Build the Gradio interface."""

    with gr.Blocks(css=CSS, title="LoRA 法律问答微调效果对比") as demo:
        gr.HTML("""
        <div class="header">
            <h1>⚖️ LoRA 法律问答微调效果对比</h1>
            <p style="font-size: 1.1em; color: #666;">
                基座模型 Qwen3-8B vs LoRA 微调模型 — 并排对比
            </p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                # Question input area
                gr.Markdown("### 📝 输入问题")

                question_input = gr.Textbox(
                    label="输入你的法律问题",
                    placeholder="例如：被公司辞退能拿多少赔偿？",
                    lines=3,
                )

                with gr.Row():
                    submit_btn = gr.Button("🔍 对比两个模型", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ 清空", size="lg")

                # Presets
                gr.Markdown("### 📋 预设测试题")

                # Build department dropdown choices
                dept_choices = list(PRESET_QUESTIONS.keys())
                dept_dropdown = gr.Dropdown(
                    label="按科室筛选",
                    choices=dept_choices,
                    value=dept_choices[0],
                )

                question_dropdown = gr.Dropdown(
                    label="选择预设问题",
                    choices=PRESET_QUESTIONS[dept_choices[0]],
                )

                random_btn = gr.Button("🎲 随机出题", size="sm")

                # Batch mode
                gr.Markdown("---")
                batch_btn = gr.Button("📊 批量对比全部预设题 (30题)", variant="secondary", size="sm")
                batch_status = gr.Markdown("")

            with gr.Column(scale=2):
                # Result panels
                with gr.Row():
                    with gr.Column():
                        gr.HTML('<div class="base-label">🔴 基座模型 (Qwen3-8B)</div>')
                        base_output = gr.Textbox(
                            label="基座模型回答",
                            lines=12,
                            max_lines=20,
                            elem_classes=["base-panel"],
                        )
                        base_hall = gr.Textbox(
                            label="幻觉检测",
                            lines=2,
                            elem_classes=["hallucination-warn"],
                        )

                    with gr.Column():
                        gr.HTML('<div class="lora-label">🟢 LoRA 微调模型</div>')
                        lora_output = gr.Textbox(
                            label="LoRA 模型回答",
                            lines=12,
                            max_lines=20,
                            elem_classes=["lora-panel"],
                        )
                        lora_hall = gr.Textbox(
                            label="幻觉检测",
                            lines=2,
                            elem_classes=["hallucination-ok"],
                        )

                # Quick metrics
                gr.Markdown("### 📊 快速对比")
                with gr.Row():
                    base_len = gr.Textbox(label="基座回答长度", scale=1)
                    lora_len = gr.Textbox(label="LoRA 回答长度", scale=1)
                    base_time = gr.Textbox(label="基座生成耗时", scale=1)
                    lora_time = gr.Textbox(label="LoRA 生成耗时", scale=1)

        # ---- Event handlers ----

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

        def compare(question):
            if not question or not question.strip():
                return "", "", "", "", "", "", ""

            # Base model
            t0 = time.time()
            base_ans = model_manager.generate_base(question)
            base_elapsed = f"{time.time() - t0:.1f}s"

            # LoRA model
            t0 = time.time()
            lora_ans = model_manager.generate_lora(question)
            lora_elapsed = f"{time.time() - t0:.1f}s"

            base_hall_check = quick_hallucination_check(base_ans)
            lora_hall_check = quick_hallucination_check(lora_ans)

            return (
                base_ans, lora_ans,
                base_hall_check, lora_hall_check,
                f"{len(base_ans)} 字", f"{len(lora_ans)} 字",
                base_elapsed, lora_elapsed,
            )

        submit_btn.click(
            compare, question_input,
            [base_output, lora_output, base_hall, lora_hall, base_len, lora_len, base_time, lora_time],
        )

        clear_btn.click(
            lambda: ("", "", "", "", "", "", "", ""),
            None,
            [question_input, base_output, lora_output, base_hall, lora_hall, base_len, lora_len, base_time, lora_time],
        )

        def batch_compare():
            all_questions = []
            for qs in PRESET_QUESTIONS.values():
                all_questions.extend(qs)

            results = []
            for i, q in enumerate(all_questions[:30]):
                base_ans = model_manager.generate_base(q)
                lora_ans = model_manager.generate_lora(q)
                results.append({
                    "question": q,
                    "base_answer": base_ans[:200],
                    "lora_answer": lora_ans[:200],
                })

            # Generate markdown table
            table = "| # | 问题 | 基座回答 (摘要) | LoRA回答 (摘要) |\n"
            table += "|---|------|----------------|----------------|\n"
            for i, r in enumerate(results):
                q_short = r["question"][:30]
                b_short = r["base_answer"][:60].replace("\n", " ")
                l_short = r["lora_answer"][:60].replace("\n", " ")
                table += f"| {i+1} | {q_short} | {b_short} | {l_short} |\n"

            # Save
            output_path = Path("outputs/eval_results")
            output_path.mkdir(parents=True, exist_ok=True)
            with open(output_path / "batch_quick_compare.json", "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            return f"✅ 完成 {len(results)} 道题的批量对比！\n结果已保存至 outputs/eval_results/batch_quick_compare.json\n\n{table}"

        batch_btn.click(batch_compare, None, batch_status)

    return demo


def main():
    parser = argparse.ArgumentParser(description="Gradio LoRA Legal QA comparison UI")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--lora_only", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    # Auto-find latest LoRA weights
    if args.lora_path is None and not args.lora_only:
        lora_dirs = sorted(Path("outputs/lora_weights").glob("law-lora-*"))
        if lora_dirs:
            args.lora_path = str(lora_dirs[-1])
            print(f"Using latest LoRA adapter: {args.lora_path}")
        else:
            print("⚠ No LoRA adapter found. Run training first.")
            print("  Starting in base-model-only mode.")
            args.lora_only = True

    print("Loading models...")
    model_manager = ModelManager(args.base_model, args.lora_path)

    print("Loading test cases...")
    test_cases = load_test_cases()

    print(f"Starting Gradio on port {args.port}...")
    demo = create_ui(model_manager, test_cases)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
