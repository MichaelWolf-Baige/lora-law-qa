"""
05_infer_compare.py — Side-by-side inference comparison (base vs LoRA).

CRITICAL for 8GB VRAM: Only ONE model is loaded at a time.
Between models, we explicitly free GPU memory with del + gc.collect() + empty_cache().

Usage:
    python scripts/05_infer_compare.py --lora_path outputs/lora_weights/law-lora-r16-XXX
    python scripts/05_infer_compare.py --interactive  # Interactive mode
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.domain_config import get_domain


# ──────────────────────────────────────────────
# System prompt (must match training)
# ──────────────────────────────────────────────

SYSTEM_PROMPT = get_domain().default_system_prompt


def format_chatml(question: str) -> str:
    """Format a question as ChatML prompt (no assistant response)."""
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def clear_gpu():
    """Aggressively free GPU memory — essential for 8GB VRAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_model_and_tokenizer(
    base_model_name: str,
    lora_path: str = None,
    use_4bit: bool = False,
):
    """Load model, optionally with LoRA adapter.

    Always loads base model first, then optionally merges LoRA.
    This avoids OOM from loading two separate models.
    """
    print(f"  Loading base model: {base_model_name}")
    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs = {"quantization_config": bnb_config}
    else:
        model_kwargs = {"torch_dtype": torch.float16}

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name if lora_path is None else lora_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        device_map="auto",
        **model_kwargs,
    )

    if lora_path:
        print(f"  Loading LoRA adapter from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()
        print("  LoRA merged and unloaded")

    model.eval()
    return model, tokenizer


def generate(
    model,
    tokenizer,
    question: str,
    max_new_tokens: int = 300,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate answer for a legal question."""
    prompt = format_chatml(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part (remove prompt tokens)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True)
    return answer.strip()


def compare_single(
    base_model,
    base_tokenizer,
    lora_model,
    lora_tokenizer,
    question: str,
    reference: str = None,
) -> dict:
    """Run inference with both base and LoRA models, return comparison."""
    print(f"\n{'='*60}")
    print(f"Question: {question[:100]}...")
    print(f"{'='*60}")

    # Base model
    t0 = time.time()
    base_answer = generate(base_model, base_tokenizer, question)
    base_time = time.time() - t0
    print(f"\n[Base Model] ({base_time:.1f}s):")
    print(f"  {base_answer[:200]}...")

    # LoRA model
    t0 = time.time()
    lora_answer = generate(lora_model, lora_tokenizer, question)
    lora_time = time.time() - t0
    print(f"\n[LoRA Model] ({lora_time:.1f}s):")
    print(f"  {lora_answer[:200]}...")

    ref_str = reference[:100] + "..." if reference else "N/A"
    print(f"\n[Reference]: {ref_str}")

    return {
        "question": question,
        "reference": reference,
        "base_answer": base_answer,
        "lora_answer": lora_answer,
        "base_time": base_time,
        "lora_time": lora_time,
    }


def batch_compare(
    base_model_name: str,
    lora_path: str,
    questions_file: str,
    output_file: str,
    use_4bit: bool = False,
    max_questions: int = None,
):
    """Run batch comparison on a set of legal questions."""
    print("=" * 60)
    print("BATCH COMPARISON MODE")
    print("=" * 60)

    # Load questions
    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if max_questions:
        questions = questions[:max_questions]

    print(f"  Total questions: {len(questions)}")

    # --- BASE MODEL ---
    print("\n" + "-" * 40)
    print("PHASE 1: Base model inference")
    print("-" * 40)
    base_model, base_tokenizer = load_model_and_tokenizer(base_model_name, use_4bit=use_4bit)

    base_results = []
    for i, item in enumerate(questions):
        q = item["question"] if isinstance(item, dict) else item
        ref = item.get("answer") if isinstance(item, dict) else None
        print(f"\n[{i+1}/{len(questions)}] {q[:80]}...")
        t0 = time.time()
        answer = generate(base_model, base_tokenizer, q)
        elapsed = time.time() - t0
        base_results.append({"question": q, "answer": answer, "time": elapsed, "reference": ref})

    # Free base model memory
    del base_model, base_tokenizer
    clear_gpu()
    print("\n  Base model unloaded, GPU cleared.")

    # --- LORA MODEL ---
    print("\n" + "-" * 40)
    print("PHASE 2: LoRA model inference")
    print("-" * 40)
    lora_model, lora_tokenizer = load_model_and_tokenizer(
        base_model_name, lora_path=lora_path, use_4bit=use_4bit
    )

    lora_results = []
    for i, item in enumerate(questions):
        q = item["question"] if isinstance(item, dict) else item
        print(f"\n[{i+1}/{len(questions)}] {q[:80]}...")
        t0 = time.time()
        answer = generate(lora_model, lora_tokenizer, q)
        elapsed = time.time() - t0
        lora_results.append({"answer": answer, "time": elapsed})

    # Combine results
    combined = []
    for base_r, lora_r in zip(base_results, lora_results):
        combined.append({
            "question": base_r["question"],
            "reference": base_r["reference"],
            "base_answer": base_r["answer"],
            "base_time": base_r["time"],
            "lora_answer": lora_r["answer"],
            "lora_time": lora_r["time"],
        })

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Batch comparison saved to: {output_path}")
    print(f"   Total comparisons: {len(combined)}")

    # Quick stats
    avg_base_time = sum(r["base_time"] for r in combined) / len(combined)
    avg_lora_time = sum(r["lora_time"] for r in combined) / len(combined)
    print(f"   Avg base time: {avg_base_time:.1f}s")
    print(f"   Avg lora time: {avg_lora_time:.1f}s")


def interactive_compare(base_model_name: str, lora_path: str, use_4bit: bool = False):
    """Interactive mode: ask questions and compare answers in real-time."""
    print("=" * 60)
    print("INTERACTIVE COMPARISON MODE")
    print("=" * 60)

    # Load both models
    print("Loading base model...")
    base_model, base_tokenizer = load_model_and_tokenizer(base_model_name, use_4bit=use_4bit)

    print("Loading LoRA model...")
    # For interactive mode: load base again, apply LoRA, merge
    lora_model, lora_tokenizer = load_model_and_tokenizer(
        base_model_name, lora_path=lora_path, use_4bit=use_4bit
    )

    print("\n" + "=" * 60)
    print("Ready! Type a legal question (or 'quit' to exit)")
    print("=" * 60)

    while True:
        try:
            question = input("\n❓ Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue

        result = compare_single(
            base_model, base_tokenizer,
            lora_model, lora_tokenizer,
            question,
        )


def main():
    parser = argparse.ArgumentParser(description="Compare base vs LoRA model inference")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--questions", type=str, default="data/test_cases/all_departments.json")
    parser.add_argument("--output", type=str, default="outputs/eval_results/batch_comparison.json")
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--interactive", action="store_true")

    args = parser.parse_args()

    if args.interactive:
        if not args.lora_path:
            print("ERROR: --lora_path required for interactive mode")
            sys.exit(1)
        interactive_compare(args.base_model, args.lora_path, args.use_4bit)
    else:
        # Find latest LoRA weights if not specified
        if args.lora_path is None:
            lora_dirs = sorted(Path("outputs/lora_weights").glob("law-lora-*"))
            if lora_dirs:
                args.lora_path = str(lora_dirs[-1])
                print(f"Using latest LoRA adapter: {args.lora_path}")
            else:
                print("ERROR: No LoRA adapter found. Run training first or specify --lora_path")
                sys.exit(1)

        questions_file = args.questions
        if not Path(questions_file).exists():
            print(f"ERROR: Questions file not found: {questions_file}")
            sys.exit(1)

        batch_compare(
            args.base_model,
            args.lora_path,
            questions_file,
            args.output,
            args.use_4bit,
            args.max_questions,
        )


if __name__ == "__main__":
    main()
