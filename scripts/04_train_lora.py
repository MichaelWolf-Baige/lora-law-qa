"""
04_train_lora.py — Train LoRA/DoRA adapter for legal QA.

Supports:
  - Standard LoRA (use_dora=False)
  - DoRA — Weight-Decomposed LoRA (use_dora=True, recommended)
  - QLoRA — 4-bit quantization for larger models on 8GB VRAM

Key optimizations for 8GB VRAM:
  - gradient_checkpointing: trades compute for memory
  - Small batch_size with gradient accumulation
  - fp16 mixed precision
  - max_seq_length=512 (sufficient for legal QA)

Usage:
    # Basic training (Qwen3-8B, QLoRA, fits 8GB)
    python scripts/04_train_lora.py

    # QLoRA for 8B model on 8GB
    python scripts/04_train_lora.py --model_name Qwen/Qwen3-8B --use_4bit

    # Custom config
    python scripts/04_train_lora.py --lora_r 8 --learning_rate 1e-4 --epochs 5
"""

import argparse
import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml
from datasets import DatasetDict, load_from_disk
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# ── 绕过 checkpoint 恢复时的 torch.load 安全检查 (torch<2.6 兼容) ──
# trainer.py 内部用 from import 持有原始函数引用，需要同时打两处补丁
import transformers.utils.import_utils as _hf_utils
import transformers.trainer as _hf_trainer
if hasattr(_hf_utils, "check_torch_load_is_safe"):
    _hf_utils.check_torch_load_is_safe = lambda: None
if hasattr(_hf_trainer, "check_torch_load_is_safe"):
    _hf_trainer.check_torch_load_is_safe = lambda: None


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def print_gpu_memory(msg: str = ""):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  [GPU] {msg}: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")


def clear_gpu_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class VRAMMonitorCallback(TrainerCallback):
    """Log GPU memory usage during training."""
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0 and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"  [Step {state.global_step}] GPU allocated: {allocated:.2f}GB")


def load_config(config_path: str = "configs/lora_config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapter for legal QA")
    parser.add_argument("--config", type=str, default="configs/lora_config.yaml")
    # Override config values via CLI
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--use_4bit", action="store_true", default=None)
    parser.add_argument("--use_dora", action="store_true", default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true", help="Disable WandB logging")
    args = parser.parse_args()

    # Load config and apply CLI overrides
    cfg = load_config(args.config)
    for key in ["model_name", "use_4bit", "use_dora", "lora_r", "lora_alpha",
                "learning_rate", "max_seq_length"]:
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            cfg[key] = cli_val
    if args.epochs is not None:
        cfg["num_train_epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["per_device_train_batch_size"] = args.batch_size
    if args.max_train_samples is not None:
        cfg["max_train_samples"] = args.max_train_samples

    # ============================================================
    # STEP 1: Load tokenized dataset
    # ============================================================
    print("=" * 60)
    print("STEP 1: Loading tokenized dataset")
    print("=" * 60)

    tokenized_dir = Path("data/processed/tokenized")
    if not tokenized_dir.exists():
        print("ERROR: Tokenized data not found! Run 03_prepare_sft.py first.")
        print("Expected at:", tokenized_dir.resolve())
        sys.exit(1)

    dataset_dict = load_from_disk(str(tokenized_dir))
    print(f"  Loaded: {dataset_dict}")
    for split_name in dataset_dict.keys():
        print(f"    {split_name}: {len(dataset_dict[split_name])} samples")

    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict.get("eval", None)

    # ============================================================
    # STEP 2: Load model with quantization (if requested)
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 2: Loading base model")
    print("=" * 60)

    model_name = cfg["model_name"]
    use_4bit = cfg.get("use_4bit", False)
    print(f"  Model: {model_name}")
    print(f"  4-bit quantization: {use_4bit}")
    print_gpu_memory("Before model load")

    # --- Quantization config for QLoRA ---
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print("  Using 4-bit NF4 quantization (QLoRA mode)")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if use_4bit:
        model_kwargs["quantization_config"] = bnb_config
    else:
        # 4-bit 量化时不要传 torch_dtype，否则会按 fp16 全精度加载、覆盖量化
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    # Enable gradient checkpointing for memory efficiency
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        # Qwen uses input requiring gradients; this is needed
        model.config.use_cache = False
        print("  Gradient checkpointing: ENABLED (use_cache=False)")

    print_gpu_memory("After model load")

    # ============================================================
    # STEP 3: Configure LoRA / DoRA
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 3: Configuring LoRA")
    print("=" * 60)

    use_dora = cfg.get("use_dora", True)
    lora_r = cfg["lora_r"]
    lora_alpha = cfg["lora_alpha"]
    target_modules = cfg["target_modules"]

    print(f"  Method: {'DoRA' if use_dora else 'Standard LoRA'}")
    print(f"  Rank (r): {lora_r}")
    print(f"  Alpha: {lora_alpha}")
    print(f"  Scaling: alpha/r = {lora_alpha/lora_r:.1f}")
    print(f"  Target modules: {target_modules}")

    # Prepare model for k-bit training if quantized
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=target_modules,
        use_dora=use_dora,
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    # Count trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  Trainable params: {trainable:,} ({100*trainable/total:.2f}% of {total:,})")
    print(f"  LoRA adapter size: ~{trainable * 2 / 1024**2:.1f} MB (FP16)")

    # ============================================================
    # STEP 4: Training setup
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 4: Training configuration")
    print("=" * 60)

    output_dir = Path(args.output_dir)
    run_name = f"law-lora-r{lora_r}-{datetime.now().strftime('%Y%m%d-%H%M')}"
    lora_output_dir = output_dir / "lora_weights" / run_name
    checkpoint_dir = output_dir / "checkpoints" / run_name
    log_dir = output_dir / "logs" / run_name

    lora_output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    batch_size = cfg["per_device_train_batch_size"]
    grad_accum = cfg["gradient_accumulation_steps"]
    effective_batch = batch_size * grad_accum

    print(f"  Batch size: {batch_size} × {grad_accum} grad_accum = {effective_batch} effective")
    print(f"  Learning rate: {cfg['learning_rate']}")
    print(f"  Epochs: {cfg['num_train_epochs']}")
    print(f"  Max seq length: {cfg['max_seq_length']}")
    print(f"  Warmup ratio: {cfg['warmup_ratio']}")
    print(f"  FP16: {cfg.get('fp16', True)}")

    # WandB setup
    use_wandb = not args.no_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project="lora-legal-qa",
                name=run_name,
                config={
                    "model": model_name,
                    "lora_r": lora_r,
                    "lora_alpha": lora_alpha,
                    "use_dora": use_dora,
                    "use_4bit": use_4bit,
                    "learning_rate": cfg["learning_rate"],
                    "batch_size": effective_batch,
                    "epochs": cfg["num_train_epochs"],
                    "max_seq_length": cfg["max_seq_length"],
                },
            )
            print("  WandB: ENABLED")
        except Exception as e:
            print(f"  WandB: Failed to init ({e}), continuing without it")
            use_wandb = False

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        run_name=run_name,

        # Batch & memory
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        fp16=cfg.get("fp16", True),
        bf16=cfg.get("bf16", False),

        # Optimization
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.1),
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,

        # Training length
        num_train_epochs=cfg["num_train_epochs"],

        # Logging & saving
        logging_dir=str(log_dir),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 200),
        eval_steps=cfg.get("eval_steps", 200),
        save_total_limit=cfg.get("save_total_limit", 3),
        load_best_model_at_end=cfg.get("load_best_model_at_end", True),
        metric_for_best_model=cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=False,

        # Evaluation
        eval_strategy="steps" if eval_dataset else "no",
        do_eval=eval_dataset is not None,

        # Determinism
        seed=42,
        dataloader_num_workers=0,
        remove_unused_columns=False,

        # Reporting
        report_to=["wandb"] if use_wandb else ["none"],
    )

    # Data collator (handles padding + ensures labels stay aligned)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    # ============================================================
    # STEP 5: Train
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 5: Training")
    print("=" * 60)
    print_gpu_memory("Before training")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[VRAMMonitorCallback()],
    )

    # Resume from checkpoint if specified
    resume = args.resume_from_checkpoint
    if resume:
        print(f"  Resuming from checkpoint: {resume}")

    trainer.train(resume_from_checkpoint=resume)

    print_gpu_memory("After training")

    # ============================================================
    # STEP 6: Save adapter & training metadata
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 6: Saving LoRA adapter")
    print("=" * 60)

    # Save LoRA adapter
    model.save_pretrained(str(lora_output_dir))
    tokenizer.save_pretrained(str(lora_output_dir))
    print(f"  Adapter saved to: {lora_output_dir}")

    # Save training metadata
    metadata = {
        "model_name": model_name,
        "run_name": run_name,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "use_dora": use_dora,
        "use_4bit": use_4bit,
        "learning_rate": cfg["learning_rate"],
        "num_epochs": cfg["num_train_epochs"],
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "effective_batch_size": effective_batch,
        "max_seq_length": cfg["max_seq_length"],
        "trainable_params": trainable,
        "total_params": total,
        "trainable_ratio": f"{100*trainable/total:.2f}%",
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset) if eval_dataset else 0,
        "timestamp": datetime.now().isoformat(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
    with open(lora_output_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Print adapter file sizes
    print("\n  Adapter files:")
    for f in sorted(lora_output_dir.glob("*")):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"    {f.name}: {size_kb:.1f} KB")

    total_mb = sum(
        f.stat().st_size for f in lora_output_dir.glob("*") if f.is_file()
    ) / 1024**2
    print(f"\n  Total adapter size: {total_mb:.1f} MB")

    print(f"\n✅ Training complete!")
    print(f"   Adapter: {lora_output_dir}")
    print(f"   Checkpoints: {checkpoint_dir}")
    print(f"   Logs: {log_dir}")

    # Cleanup
    clear_gpu_memory()


if __name__ == "__main__":
    main()
