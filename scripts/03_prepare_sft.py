"""
03_prepare_sft.py — Format data for SFT training with assistant-only loss masking.

Converts processed JSONL into ChatML format:
  <|im_start|>system
  {system_prompt}<|im_end|>
  <|im_start|>user
  {question}<|im_end|>
  <|im_start|>assistant
  {answer}<|im_end|>

IMPORTANT: During tokenization, prompt tokens (system+user) are masked with -100
so loss is ONLY computed on the assistant's response. This is the critical
"assistant-only loss masking" technique used by all top LoRA projects.

Usage:
    python scripts/03_prepare_sft.py [--model_name Qwen/Qwen3-8B]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.domain_config import get_domain
from app.data_quality import has_disclaimer


# ──────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────

SYSTEM_PROMPT = get_domain().default_system_prompt


# ──────────────────────────────────────────────
# ChatML formatting
# ──────────────────────────────────────────────

DISCLAIMER = "以上内容仅供参考，不构成法律意见，建议咨询执业律师。"


def ensure_disclaimer(answer: str) -> str:
    """答案缺免责声明时补上（对齐「强制免责声明」目标）。

    DISC 数据 92% 答案不带免责声明，直接 SFT 会教模型「不加免责声明」，
    因此在 tokenize 前统一补齐。已含免责声明的（蒸馏/RAFT 数据）不动。
    """
    if has_disclaimer(answer):
        return answer
    return answer.rstrip() + "\n\n" + DISCLAIMER


def _extract_qa(sample: dict) -> dict:
    """从统一样本提取 (question, answer)，兼容两种格式，并补齐免责声明。

    02_curate 输出 ChatML messages 格式（{messages:[...], metadata:{...}}）；
    旧格式是 {question, answer}。这里统一转成 {question, answer}。
    """
    if "messages" in sample:
        q = a = ""
        for m in sample["messages"]:
            if m.get("role") == "user":
                q = m.get("content", "")
            elif m.get("role") == "assistant":
                a = m.get("content", "")
        return {"question": q, "answer": ensure_disclaimer(a)}
    return {
        "question": sample.get("question", ""),
        "answer": ensure_disclaimer(sample.get("answer", "")),
    }


def format_chatml(question: str, answer: str = None) -> str:
    """Format a QA pair into ChatML format.

    When answer is None, format as an inference prompt (no assistant response).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})

    # Build ChatML string manually to have full control
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    return "\n".join(parts)


def tokenize_with_mask(
    examples: dict[str, list],
    tokenizer: AutoTokenizer,
    max_length: int = 512,
) -> dict[str, torch.Tensor]:
    """
    Tokenize conversations with assistant-only loss masking.

    This is the KEY technique for SFT quality:
    - System and user tokens are labeled -100 (ignored in loss)
    - Only assistant tokens contribute to the loss

    The approach:
    1. Tokenize the full conversation (system + user + assistant)
    2. Create a labels copy of input_ids
    3. Find where the assistant response starts
    4. Mask everything before that with -100
    """

    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    for question, answer in zip(examples["question"], examples["answer"]):
        # Build conversation
        conversation = format_chatml(question, answer)

        # Tokenize full conversation
        tokenized = tokenizer(
            conversation,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # ---- ASSISTANT-ONLY LOSS MASKING ----
        # We need to identify which tokens belong to the assistant response.
        # Strategy: tokenize only the prompt (system + user), find its length,
        # then mask all tokens before and including the prompt.

        prompt_only = format_chatml(question, answer=None)  # No assistant response
        prompt_tokens = tokenizer(
            prompt_only,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )["input_ids"]

        prompt_len = len(prompt_tokens)

        # Create labels: -100 for prompt, real token ids for assistant
        labels = [-100] * min(prompt_len, len(input_ids))
        labels += input_ids[prompt_len:]

        # Pad labels to match input_ids length (should already match)
        if len(labels) < len(input_ids):
            labels += input_ids[len(labels):]
        elif len(labels) > len(input_ids):
            labels = labels[:len(input_ids)]

        # Edge case: if the entire sequence got truncated to only prompt,
        # mask everything (no assistant tokens to learn from)
        if prompt_len >= len(input_ids):
            labels = [-100] * len(input_ids)

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(labels)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT dataset")
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen3-8B",
        help="Tokenizer to use"
    )
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_train_samples", type=int, default=3000)
    parser.add_argument("--max_eval_samples", type=int, default=500)
    args = parser.parse_args()

    processed_dir = Path("data/processed")
    print("=" * 60)
    print("STEP 1: Load processed data")
    print("=" * 60)

    splits = {}
    for split_name in ["train", "eval", "test"]:
        file_path = processed_dir / f"{split_name}.jsonl"
        # train 优先用 15_quality_filter 产出的 train_clean.jsonl（已剔除编造法条/非法律）
        if split_name == "train" and (processed_dir / "train_clean.jsonl").exists():
            file_path = processed_dir / "train_clean.jsonl"
        if not file_path.exists():
            print(f"  ⚠ {file_path.name} not found, skipping")
            continue
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(_extract_qa(json.loads(line)))
        splits[split_name] = data
        print(f"  {split_name}: {len(data)} samples")

    if "train" not in splits:
        print("ERROR: No training data found! Run 02_curate_data.py first.")
        return

    # Limit samples
    if args.max_train_samples and len(splits["train"]) > args.max_train_samples:
        splits["train"] = splits["train"][:args.max_train_samples]
    if "eval" in splits and args.max_eval_samples and len(splits["eval"]) > args.max_eval_samples:
        splits["eval"] = splits["eval"][:args.max_eval_samples]

    print("\n" + "=" * 60)
    print("STEP 2: Load tokenizer")
    print("=" * 60)
    print(f"  Model: {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="right",
    )

    # Qwen tokenizer may not have pad_token set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("  Set pad_token = eos_token")

    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Pad token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")

    print("\n" + "=" * 60)
    print("STEP 3: Tokenize with assistant-only loss masking")
    print("=" * 60)

    # Demonstrate the masking on a sample
    sample_q = splits["train"][0]["question"]
    sample_a = splits["train"][0]["answer"]

    print("\n  --- Sample ChatML Format ---")
    print(format_chatml(sample_q, sample_a)[:500] + "...")

    print("\n  --- Tokenization with Mask ---")
    sample_result = tokenize_with_mask(
        {"question": [sample_q], "answer": [sample_a]},
        tokenizer,
        max_length=args.max_length,
    )

    masked_count = sum(1 for lbl in sample_result["labels"][0] if lbl == -100)
    learning_count = sum(1 for lbl in sample_result["labels"][0] if lbl != -100)
    print(f"  Total tokens: {len(sample_result['input_ids'][0])}")
    print(f"  Masked tokens (prompt, loss ignored): {masked_count}")
    print(f"  Learning tokens (assistant, loss computed): {learning_count}")
    print(f"  Loss computed on {100*learning_count/len(sample_result['input_ids'][0]):.1f}% of tokens ✅")

    print("\n" + "=" * 60)
    print("STEP 4: Process all splits")
    print("=" * 60)

    hf_datasets = {}
    for split_name, data in splits.items():
        dataset = Dataset.from_list(data)
        dataset = dataset.map(
            lambda x: tokenize_with_mask(x, tokenizer, max_length=args.max_length),
            batched=True,
            remove_columns=dataset.column_names,
            desc=f"Tokenizing {split_name}",
        )
        hf_datasets[split_name] = dataset
        print(f"  {split_name}: {len(dataset)} tokenized samples")

    # Save as HuggingFace DatasetDict
    output_dir = processed_dir / "tokenized"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dict = DatasetDict(hf_datasets)
    dataset_dict.save_to_disk(str(output_dir))

    print(f"\n✅ Tokenized datasets saved to: {output_dir}")
    print(f"\nDataset summary:")
    for split_name, ds in hf_datasets.items():
        total_tokens = sum(len(ids) for ids in ds["input_ids"])
        avg_len = total_tokens / len(ds)
        print(f"  {split_name}: {len(ds)} samples, avg length: {avg_len:.0f} tokens")

    # Save tokenizer alongside
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))
    print(f"\n✅ Tokenizer saved to: {output_dir / 'tokenizer'}")


if __name__ == "__main__":
    main()
