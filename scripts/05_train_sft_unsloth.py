"""
05_train_sft_unsloth.py — Accelerated SFT Training with Unsloth.

Replaces the standard HuggingFace Trainer with Unsloth's optimized backend:
  - 2-5x faster training via custom Triton kernels
  - ~70% less VRAM usage
  - Native rsLoRA + DoRA support
  - Smart gradient checkpointing ("unsloth" mode saves 30% more VRAM)

Usage:
    python scripts/05_train_sft_unsloth.py
    python scripts/05_train_sft_unsloth.py --config configs/lora_config.yaml
    python scripts/05_train_sft_unsloth.py --model Qwen/Qwen3-4B --qlora  # Qwen3 with 4-bit

Expected training time (RTX 4060, ~4000 samples, 1-2 epochs):
    - Qwen3-8B (4-bit QLoRA): ~60-120 min (feasible on 8GB with Unsloth)
"""

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_dataset_for_sft(data_path: str, tokenizer, max_seq_length: int):
    """
    Load and format the SFT dataset for Unsloth training.
    Expects JSONL with "messages" field (ChatML format) or "text" field.
    """
    from datasets import load_dataset
    from unsloth.chat_templates import standardize_sharegpt

    dataset = load_dataset("json", data_files=data_path, split="train")

    # If dataset has "messages" (ChatML format), convert to ShareGPT then standardize
    if "messages" in dataset.column_names:

        def convert_to_sharegpt(example):
            messages = example["messages"]
            # Extract system, user, assistant turns
            conv = []
            for msg in messages:
                if msg["role"] == "system":
                    conv.append({"from": "system", "value": msg["content"]})
                elif msg["role"] == "user":
                    conv.append({"from": "human", "value": msg["content"]})
                elif msg["role"] == "assistant":
                    conv.append({"from": "gpt", "value": msg["content"]})
            return {"conversations": conv}

        dataset = dataset.map(convert_to_sharegpt, remove_columns=["messages"])

    # Standardize to Unsloth format
    dataset = standardize_sharegpt(dataset)

    # Apply chat template and tokenize
    def tokenize_function(examples):
        texts = []
        for conv in examples["conversations"]:
            text = tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}

    dataset = dataset.map(tokenize_function, batched=True)
    return dataset


def train(config: dict):
    """Main training function."""
    from unsloth import FastLanguageModel, is_training_supported
    from unsloth.chat_templates import get_chat_template
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import load_dataset

    # ── Config ──
    model_name = config.get("model_name", "Qwen/Qwen3-8B")
    use_4bit = config.get("use_4bit", True)
    use_dora = config.get("use_dora", True)
    lora_r = config.get("lora_r", 16)
    lora_alpha = config.get("lora_alpha", 16)
    lora_dropout = config.get("lora_dropout", 0.05)
    use_rslora = config.get("use_rslora", True)
    target_modules = config.get("target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    per_device_batch = config.get("per_device_train_batch_size", 2)
    grad_accum = config.get("gradient_accumulation_steps", 4)
    learning_rate = config.get("learning_rate", 2.0e-4)
    num_epochs = config.get("num_train_epochs", 3)
    warmup_ratio = config.get("warmup_ratio", 0.1)
    max_seq_length = config.get("max_seq_length", 1024)
    lr_scheduler = config.get("lr_scheduler_type", "cosine")
    logging_steps = config.get("logging_steps", 10)
    save_steps = config.get("save_steps", 200)
    eval_steps = config.get("eval_steps", 200)
    save_total_limit = config.get("save_total_limit", 3)
    fp16 = config.get("fp16", True)

    dataset_name = config.get("dataset_name", None)  # HF fallback dataset; 数据管道待后续研究
    max_train_samples = config.get("max_train_samples", 3000)
    max_eval_samples = config.get("max_eval_samples", 500)

    # Output dir
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    method = "qlora" if use_4bit else "lora"
    dora_str = "-dora" if use_dora else ""
    rs_str = "-rs" if use_rslora else ""
    output_dir = f"outputs/checkpoints/law-{method}-r{lora_r}{dora_str}{rs_str}-{timestamp}"

    print("=" * 60)
    print(f"🚀 LexiCare SFT Training (Unsloth Accelerated)")
    print(f"   Model: {model_name}")
    print(f"   Method: {'QLoRA 4-bit' if use_4bit else 'LoRA fp16'} + {'DoRA' if use_dora else 'Standard LoRA'} + {'rsLoRA' if use_rslora else 'Standard Scaling'}")
    print(f"   Rank: r={lora_r}, alpha={lora_alpha}")
    print(f"   Batch: {per_device_batch} × {grad_accum} = eff {per_device_batch * grad_accum}")
    print(f"   LR: {learning_rate}, Epochs: {num_epochs}")
    print(f"   Max Seq Length: {max_seq_length}")
    print(f"   Output: {output_dir}")
    print("=" * 60)

    # ── Load Model & Tokenizer ──
    print("\n[1/4] Loading model and tokenizer...")
    t0 = time.time()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=use_4bit,
        fast_inference=True,  # Enable Unsloth's fast inference kernels
        max_lora_rank=128 if use_rslora else 64,
        gpu_memory_utilization=0.85,  # Leave some VRAM for batch processing
    )

    print(f"   Base model loaded in {time.time() - t0:.0f}s")

    # Apply chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",  # Qwen uses ChatML format
    )

    # ── Apply LoRA ──
    print("\n[2/4] Applying LoRA adapters...")

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        use_rslora=use_rslora,
        use_dora=use_dora,
        use_gradient_checkpointing="unsloth",  # 30% less VRAM vs standard
        random_state=42,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── Load Data ──
    print("\n[3/4] Loading dataset...")

    # Try processed data first, fall back to HuggingFace dataset
    processed_path = Path("data/processed/train.jsonl")
    if processed_path.exists():
        print(f"   Using processed data: {processed_path}")

        def load_jsonl(path, max_samples=None):
            data = []
            with open(path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if max_samples and i >= max_samples:
                        break
                    data.append(json.loads(line))
            return data

        train_data = load_jsonl(processed_path, max_train_samples)
        # Convert to ShareGPT format for Unsloth
        from datasets import Dataset

        def to_sharegpt(item):
            messages = item.get("messages", [])
            conv = []
            for msg in messages:
                role_map = {"system": "system", "user": "human", "assistant": "gpt"}
                conv.append({
                    "from": role_map.get(msg["role"], msg["role"]),
                    "value": msg["content"],
                })
            return {"conversations": conv}

        train_dataset = Dataset.from_list([to_sharegpt(d) for d in train_data])

        # Load eval
        eval_path = Path("data/processed/eval.jsonl")
        if eval_path.exists():
            eval_data = load_jsonl(eval_path, max_eval_samples)
            eval_dataset = Dataset.from_list([to_sharegpt(d) for d in eval_data])
        else:
            # Split train
            split = train_dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset = split["train"]
            eval_dataset = split["test"]
    else:
        print(f"   Using HuggingFace dataset: {dataset_name}")
        from datasets import load_dataset

        dataset = load_dataset(dataset_name, split="train")
        if max_train_samples:
            dataset = dataset.select(range(min(max_train_samples, len(dataset))))

        # Format for Unsloth — wrap in chat format
        def format_as_chat(example):
            # Handle different dataset formats
            if "input" in example and "output" in example:
                conv = [
                    {"from": "human", "value": example["input"]},
                    {"from": "gpt", "value": example["output"]},
                ]
            elif "instruction" in example and "output" in example:
                inp = example.get("input", "")
                user_msg = f"{example['instruction']}\n{inp}" if inp else example["instruction"]
                conv = [
                    {"from": "human", "value": user_msg},
                    {"from": "gpt", "value": example["output"]},
                ]
            else:
                # Unknown format — skip
                return {"conversations": []}
            return {"conversations": conv}

        dataset = dataset.map(format_as_chat)
        dataset = dataset.filter(lambda x: len(x["conversations"]) > 0)
        split = dataset.train_test_split(test_size=0.15, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]

    from unsloth.chat_templates import standardize_sharegpt
    train_dataset = standardize_sharegpt(train_dataset)
    eval_dataset = standardize_sharegpt(eval_dataset)

    print(f"   Train: {len(train_dataset)} samples, Eval: {len(eval_dataset)} samples")

    # ── Train ──
    print("\n[4/4] Training...")
    print(f"   Effective batch size: {per_device_batch * grad_accum}")
    print(f"   Expected steps/epoch: ~{len(train_dataset) // (per_device_batch * grad_accum)}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch,
        per_device_eval_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler,
        fp16=fp16,
        bf16=False,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        evaluation_strategy="steps",
        save_strategy="steps",
        optim="adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=1.0,
        seed=42,
        report_to="wandb" if config.get("use_wandb", False) else "none",
        run_name=f"lexicare-sft-{method}-r{lora_r}-{timestamp}",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        logging_dir=f"{output_dir}/logs",
    )

    from trl import SFTTrainer

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        packing=False,  # Not recommended for accuracy-sensitive legal data
    )

    t_train_start = time.time()
    trainer.train()
    train_time = time.time() - t_train_start

    print(f"\n✅ Training completed in {train_time/60:.1f} minutes")

    # ── Save ──
    lora_save_path = f"outputs/lora_weights/lexicare-sft-r{lora_r}-{timestamp}"
    print(f"\n💾 Saving LoRA adapter to: {lora_save_path}")
    model.save_pretrained(lora_save_path)
    tokenizer.save_pretrained(lora_save_path)

    # Save training summary
    summary = {
        "model": model_name,
        "method": f"{'QLoRA' if use_4bit else 'LoRA'}+{'DoRA' if use_dora else 'Standard'}+{'rsLoRA' if use_rslora else 'Standard'}",
        "config": {
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "use_dora": use_dora,
            "use_rslora": use_rslora,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "max_seq_length": max_seq_length,
            "effective_batch_size": per_device_batch * grad_accum,
        },
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "train_time_minutes": round(train_time / 60, 1),
        "timestamp": timestamp,
        "framework": "Unsloth (accelerated)",
    }
    summary_path = Path(lora_save_path) / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ All done! LoRA adapter saved to: {lora_save_path}")
    print(f"   Adapter size: ~{Path(lora_save_path).stat().st_size / 1024 / 1024:.1f} MB (approx)")
    print(f"   Training time: {train_time/60:.1f} min")
    print(f"{'='*60}")

    return lora_save_path


def main():
    parser = argparse.ArgumentParser(description="LexiCare SFT Training with Unsloth")
    parser.add_argument("--config", type=str, default="configs/lora_config.yaml")
    parser.add_argument("--model", type=str, default=None,
                        help="Override model name (e.g., Qwen/Qwen3-4B)")
    parser.add_argument("--qlora", action="store_true",
                        help="Use 4-bit QLoRA (for 3B+ models on 8GB VRAM)")
    parser.add_argument("--data", type=str, default=None,
                        help="Override training data path")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--no_wandb", action="store_true", default=True,
                        help="Disable wandb logging")
    args = parser.parse_args()

    config = load_config(args.config)

    # CLI overrides
    if args.model:
        config["model_name"] = args.model
    if args.qlora:
        config["use_4bit"] = True
    if args.epochs:
        config["num_train_epochs"] = args.epochs
    if args.lr:
        config["learning_rate"] = args.lr
    config["use_wandb"] = not args.no_wandb

    train(config)


if __name__ == "__main__":
    main()
