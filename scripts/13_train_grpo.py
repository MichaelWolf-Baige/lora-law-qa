"""
13_train_grpo.py — GRPO Phase 3 Reinforcement Learning Training.

Group Relative Policy Optimization (GRPO) for legal reasoning enhancement.
Based on DeepSeek's GRPO algorithm, adapted for chronic disease management.

Pipeline:  SFT → SafeDPO → GRPO (this script)
           知识    安全边界   推理+结构化输出

Key features:
  - Reward functions: structure, safety, legal grounding, conciseness
  - Group sampling: G completions per prompt, relative advantage
  - No critic/value network needed (unlike PPO)
  - Works with small datasets (50-100 examples effective)

Reference:
  - GRPO Paper: arxiv.org/abs/2402.03300 (DeepSeek)
  - GRPO Paper: arxiv.org/abs/2402.03300 (DeepSeek)
  - TRL GRPOTrainer: huggingface.co/docs/trl/grpo_trainer

Usage:
    python scripts/13_train_grpo.py
    python scripts/13_train_grpo.py --adapter outputs/lora_weights/lexicare-safedpo-XXX
    python scripts/13_train_grpo.py --num_generations 8 --learning_rate 1e-6
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))


# ──────────────────────────────────────────────
# Reward Functions
# ──────────────────────────────────────────────

def structure_reward(completion: str) -> float:
    """
    Reward structured output with <分析> and <建议> tags.

    Returns 0.0-1.0 based on structural completeness.
    """
    score = 0.0

    if "<分析>" in completion and "</分析>" in completion:
        score += 0.3
        analysis = completion.split("<分析>")[1].split("</分析>")[0]
        if "核心问题" in analysis:
            score += 0.1
        if "法律依据" in analysis or "法条" in analysis:
            score += 0.1
        if "风险" in analysis or "时效" in analysis:
            score += 0.1

    if "<建议>" in completion and "</建议>" in completion:
        score += 0.2
        recs = completion.split("<建议>")[1].split("</建议>")[0]
        if "证据" in recs or "保全" in recs:
            score += 0.05
        if "仲裁" in recs or "起诉" in recs or "诉讼" in recs:
            score += 0.05
        if "咨询律师" in recs or "委托" in recs:
            score += 0.05
        if "⚠️" in recs or "免责" in recs:
            score += 0.05

    return min(score, 1.0)


def safety_reward(completion: str, question: str = "") -> float:
    """
    Reward safe responses, penalize dangerous ones.

    Returns 0.0-1.0. Returns 0.0 if dangerous content detected.
    """
    dangerous = [
        r"(?:肯定能赢|一定胜诉|保证胜诉|包赢|必胜|稳赢)",
        r"胜诉率.{0,6}\d{1,3}%",
        r"我(?:是|作为|以).{0,6}(?:律师|执业律师)",
        r"(?:不用|不需要|没必要).{0,5}(?:找律师|请律师)",
    ]
    for pattern in dangerous:
        if re.search(pattern, completion):
            match = re.search(pattern, completion)
            start = max(0, match.start() - 10)
            prefix = completion[start:match.start()]
            if not any(neg in prefix for neg in ["请勿", "不要", "不可", "禁止", "避免", "并非"]):
                return 0.0

    # Reward safety markers
    score = 0.5
    if "咨询律师" in completion or "委托律师" in completion or "找律师" in completion:
        score += 0.15
    if "不构成法律意见" in completion or "仅供参考" in completion:
        score += 0.15
    if "证据" in completion and ("保留" in completion or "固定" in completion):
        score += 0.1
    if "时效" in completion:
        score += 0.1

    return min(score, 1.0)


def legal_basis_reward(completion: str) -> float:
    """
    Reward responses grounded in legal provisions.

    Returns 0.0-1.0 based on statute citation and legal-concept correctness.
    """
    score = 0.0

    # 法条引用加成
    statute_cites = re.findall(r'《[^》]{2,20}》\s*第\s*[0-9一二三四五六七八九十百零〇]+\s*条', completion)
    if statute_cites:
        score += 0.4

    # 正确法律概念
    legal_markers = [
        "经济补偿", "赔偿金", "违约金", "仲裁", "诉讼时效", "仲裁时效",
        "劳动合同", "竞业限制", "定金", "违约金", "过错", "举证", "证据",
        "违法解除", "拖欠工资", "工伤", "不构成法律意见",
    ]
    matches = sum(1 for m in legal_markers if m in completion)
    score += min(matches * 0.08, 0.4)

    # 关键法律原则
    principles = {
        "不构成法律意见": 0.1,
        "仲裁时效": 0.05,
        "诉讼时效": 0.05,
        "二倍": 0.05,
        "N+1": 0.05,
        "2N": 0.05,
    }
    for p, r in principles.items():
        if p in completion:
            score += r

    return min(score, 1.0)


def conciseness_reward(completion: str) -> float:
    """
    Reward appropriately detailed responses. Penalize too short or too long.

    Optimal range: 200-800 characters for legal advice.
    """
    length = len(completion)
    if length < 50:
        return 0.0   # Too short
    elif length < 200:
        return 0.5   # A bit short
    elif length <= 800:
        return 1.0   # Optimal
    elif length <= 1500:
        return 0.7   # A bit long
    else:
        return 0.4   # Too long


def hallucination_penalty(completion: str, retrieved_docs: list = None) -> float:
    """
    Penalize potential hallucinations. Returns 0.0-1.0 (higher = more hallucinated).
    """
    penalty = 0.0

    # Fake statistics
    if re.search(r'\d{2,3}\.\d+%.{0,20}(?:胜诉|获赔|支持|劳动者)', completion):
        penalty += 0.3

    # Vague references without specifics
    if re.search(r'根据.{0,20}(?:司法解释|判例|研究).{0,20}(?:显示|表明)', completion):
        if "《" not in completion and "第" not in completion:
            penalty += 0.2

    # Overconfident language
    overconfident = ["一定可以", "肯定能赢", "绝对有效", "保证拿回", "包赢", "胜诉率"]
    for phrase in overconfident:
        if phrase in completion:
            penalty += 0.2
            break

    return min(penalty, 1.0)


def combined_reward(completions: list, questions: list = None,
                    **kwargs) -> list:
    """
    Combined reward function for GRPO.

    Args:
        completions: List of generated completions
        questions: List of corresponding prompts

    Returns:
        List of reward scores (each 0.0-1.0)
    """
    rewards = []
    for i, completion in enumerate(completions):
        question = questions[i] if questions else ""

        r_structure = structure_reward(completion)
        r_safety = safety_reward(completion, question)
        r_legal = legal_basis_reward(completion)
        r_conciseness = conciseness_reward(completion)
        r_hallucination = hallucination_penalty(completion)

        # Weighted combination
        total = (
            0.25 * r_structure +
            0.30 * r_safety +        # Safety is highest weight
            0.20 * r_legal +
            0.10 * r_conciseness +
            0.15 * (1.0 - r_hallucination)  # Invert: low hallucination = high reward
        )

        rewards.append(total)

    return rewards


# ──────────────────────────────────────────────
# GRPO Data Preparation
# ──────────────────────────────────────────────

def prepare_grpo_dataset(data_path: str, max_samples: int = 200) -> list:
    """
    Prepare dataset for GRPO training.

    GRPO needs diverse prompts that benefit from reasoning.
    We select: emergency cases, drug interactions, complex indicator interpretation.
    """
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))

    # Prioritize complex cases for GRPO
    grpo_prompts = []
    for item in data:
        if "messages" in item:
            user_msg = next((m["content"] for m in item["messages"] if m["role"] == "user"), "")
        elif "question" in item:
            user_msg = item["question"]
        else:
            continue

        # Select complex questions that benefit from reasoning
        complexity_score = 0
        complexity_terms = [
            "为什么", "怎么办", "如何", "区别", "对比",
            "合并", "同时", "相互作用", "禁忌", "副作用",
            "赔偿", "补偿", "违约", "竞业", "时效", "证据", "仲裁", "诉讼",
        ]
        for term in complexity_terms:
            if term in user_msg:
                complexity_score += 1

        if complexity_score >= 2:  # At least 2 complexity indicators
            grpo_prompts.append(user_msg)

    # Sample diverse prompts
    if len(grpo_prompts) > max_samples:
        import random
        random.seed(42)
        grpo_prompts = random.sample(grpo_prompts, max_samples)

    print(f"   Selected {len(grpo_prompts)} prompts for GRPO training")
    return grpo_prompts


# ──────────────────────────────────────────────
# Main GRPO Training
# ──────────────────────────────────────────────

def train_grpo(config: dict):
    """Main GRPO training function."""
    from trl import GRPOConfig, GRPOTrainer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, LoraConfig, get_peft_model
    from datasets import Dataset

    # ── Config ──
    base_model = config.get("base_model", "Qwen/Qwen3-8B")
    adapter_path = config.get("adapter_path", None)
    cot_data_path = config.get("cot_data", "data/processed/train_cot.jsonl")

    num_generations = config.get("num_generations", 4)  # Group size G
    learning_rate = config.get("grpo_lr", 1.0e-6)
    max_prompt_length = config.get("max_prompt_length", 512)
    max_completion_length = config.get("max_completion_length", 512)
    beta = config.get("grpo_beta", 0.04)  # KL penalty coefficient
    max_steps = config.get("max_steps", 200)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    output_dir = f"outputs/checkpoints/grpo-g{num_generations}-{timestamp}"

    print("=" * 60)
    print(f"🧠 LexiCare GRPO Training (Phase 3)")
    print(f"   Base Model: {base_model}")
    print(f"   Adapter: {adapter_path}")
    print(f"   Group Size G: {num_generations}")
    print(f"   Learning Rate: {learning_rate}")
    print(f"   KL Beta: {beta}")
    print(f"   Max Steps: {max_steps}")
    print(f"   Output: {output_dir}")
    print("=" * 60)

    # ── Load Model ──
    print("\n[1/4] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load SFT+DPO adapter if available
    if adapter_path and Path(adapter_path).exists():
        print(f"   Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print("   Adapter merged into base model")

    # Apply light LoRA for GRPO (small rank, only attention)
    print("\n[2/4] Applying LoRA for GRPO...")
    grpo_lora = LoraConfig(
        r=8,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=True,
    )
    model = get_peft_model(model, grpo_lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable GRPO params: {trainable:,}")

    # ── Prepare Data ──
    print("\n[3/4] Preparing GRPO training data...")
    grpo_prompts = prepare_grpo_dataset(cot_data_path)
    dataset = Dataset.from_list([{"prompt": p} for p in grpo_prompts])

    # ── Train ──
    print("\n[4/4] Training GRPO...")

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        num_train_epochs=1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        fp16=True,
        bf16=False,
        logging_steps=10,
        save_steps=50,
        save_total_limit=3,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        num_generations=num_generations,
        beta=beta,
        temperature=0.9,
        max_steps=max_steps,
        reward_weights={
            "structure": 0.25,
            "safety": 0.30,
            "legal_basis": 0.20,
            "conciseness": 0.10,
            "hallucination_penalty": 0.15,
        },
        report_to="none",
        seed=42,
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=grpo_config,
        train_dataset=dataset,
        reward_functions=[combined_reward],
    )

    print(f"\n   Starting GRPO training with {len(dataset)} prompts...")
    print(f"   Generating G={num_generations} completions per prompt")
    t0 = time.time()

    try:
        trainer.train()
        train_time = time.time() - t0
        print(f"\n✅ GRPO training completed in {train_time/60:.1f} minutes")
    except Exception as e:
        print(f"\n⚠ GRPO training encountered an error: {e}")
        print("   This is expected if TRL version doesn't fully support GRPO yet.")
        print("   The script structure is ready for when GRPOTrainer is stable.")
        return None

    # ── Save ──
    lora_save_path = f"outputs/lora_weights/lexicare-grpo-g{num_generations}-{timestamp}"
    print(f"\n💾 Saving GRPO adapter: {lora_save_path}")
    model.save_pretrained(lora_save_path)
    tokenizer.save_pretrained(lora_save_path)

    summary = {
        "method": "GRPO (Phase 3)",
        "base_model": base_model,
        "adapter_path": adapter_path,
        "config": {
            "num_generations": num_generations,
            "learning_rate": learning_rate,
            "beta": beta,
            "max_steps": max_steps,
            "reward_weights": {
                "structure": 0.25, "safety": 0.30, "legal_basis": 0.20,
                "conciseness": 0.10, "hallucination_penalty": 0.15,
            },
        },
        "train_prompts": len(dataset),
        "train_time_minutes": round(time.time() - t0, 1) if 't0' in dir() else None,
        "timestamp": timestamp,
    }
    with open(Path(lora_save_path) / "grpo_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ GRPO Phase 3 complete! Adapter saved to: {lora_save_path}")
    print(f"{'='*60}")

    return lora_save_path


# ──────────────────────────────────────────────
# Reward Function Testing
# ──────────────────────────────────────────────

def test_rewards():
    """Test reward functions with example completions."""
    test_cases = [
        {
            "name": "Perfect CoT answer",
            "text": (
                "<分析>\n"
                "- 核心问题: 公司违法解除劳动合同的赔偿标准\n"
                "- 法律依据: 《劳动合同法》第八十七条\n"
                "- 风险: 仲裁时效 1 年\n"
                "</分析>\n"
                "<建议>\n"
                "🔴 证据: 保留解除通知、劳动合同、工资流水\n"
                "🟡 途径: 申请劳动仲裁\n"
                "🟢 咨询律师: 建议委托执业律师\n"
                "⚠️ 免责声明: 以上内容仅供参考，不构成法律意见\n"
                "</建议>"
            ),
        },
        {
            "name": "Overpromise / win-rate claim",
            "text": "你这个案子肯定能赢，胜诉率 99%，直接去起诉，赔偿肯定能拿回来。",
        },
        {
            "name": "Safe refusal (off-topic)",
            "text": "您的问题不在法律咨询范围内。建议您咨询专业律师获取帮助。",
        },
        {
            "name": "Too short answer",
            "text": "可以起诉。",
        },
    ]

    print("\n" + "=" * 60)
    print("Reward Function Test")
    print("=" * 60)

    for tc in test_cases:
        r_struct = structure_reward(tc["text"])
        r_safety = safety_reward(tc["text"])
        r_guide = legal_basis_reward(tc["text"])
        r_concise = conciseness_reward(tc["text"])
        r_hallu = hallucination_penalty(tc["text"])
        total = 0.25*r_struct + 0.30*r_safety + 0.20*r_guide + 0.10*r_concise + 0.15*(1-r_hallu)

        print(f"\n📝 {tc['name']}")
        print(f"   Structure: {r_struct:.2f} | Safety: {r_safety:.2f} | Guideline: {r_guide:.2f}")
        print(f"   Conciseness: {r_concise:.2f} | Hallu Penalty: {r_hallu:.2f}")
        print(f"   TOTAL: {total:.2f}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LexiCare GRPO Phase 3 Training")
    parser.add_argument("--config", type=str, default="configs/lora_config.yaml")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to SFT+DPO adapter")
    parser.add_argument("--cot_data", type=str, default="data/processed/train_cot.jsonl")
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1.0e-6)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--test_rewards", action="store_true",
                        help="Test reward functions and exit")
    args = parser.parse_args()

    if args.test_rewards:
        test_rewards()
        return

    config = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    # Auto-detect adapter
    adapter_path = args.adapter
    if not adapter_path:
        # Try SafeDPO first, then SFT
        for pattern in ["lexicare-safedpo-*", "lexicare-sft-*"]:
            candidates = sorted(Path("outputs/lora_weights").glob(pattern))
            if candidates:
                adapter_path = str(candidates[-1])
                print(f"Auto-detected adapter: {adapter_path}")
                break

    grpo_config = {
        "base_model": config.get("model_name", "Qwen/Qwen3-8B"),
        "adapter_path": adapter_path,
        "cot_data": args.cot_data,
        "num_generations": args.num_generations,
        "grpo_lr": args.learning_rate,
        "grpo_beta": 0.04,
        "max_prompt_length": 512,
        "max_completion_length": 512,
        "max_steps": args.max_steps,
    }

    train_grpo(grpo_config)


if __name__ == "__main__":
    main()
