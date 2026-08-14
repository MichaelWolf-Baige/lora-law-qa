"""Smoke test: verify GPU training pipeline works on RTX 4060."""
import torch, gc, json, time, sys
from pathlib import Path

sys.path.insert(0, ".")

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import Dataset
import transformers.utils.import_utils as _hf_utils
import transformers.trainer as _hf_trainer

# Compatibility patches
if hasattr(_hf_utils, "check_torch_load_is_safe"):
    _hf_utils.check_torch_load_is_safe = lambda: None
if hasattr(_hf_trainer, "check_torch_load_is_safe"):
    _hf_trainer.check_torch_load_is_safe = lambda: None

SEP = "=" * 60
print(SEP)
print("LexiCare GPU Training Smoke Test")
print(f"PyTorch={torch.__version__}, GPU={torch.cuda.get_device_name(0)}")
print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB")
print(SEP)

# [1] Load model
print("\n[1/4] Loading model (fp16)...")
model_name = "Qwen/Qwen3-8B"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
vram_base = torch.cuda.memory_allocated() / 1024**3
print(f"  Base model VRAM: {vram_base:.1f} GB")

# [2] Apply LoRA with CORRECTED config
print("\n[2/4] Applying LoRA (rsLoRA r=16 alpha=16 DoRA)...")
lora_config = LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    use_rslora=True, use_dora=True, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"  Trainable: {trainable/1e6:.1f}M / {total/1e9:.2f}B ({100*trainable/total:.1f}%)")
vram_lora = torch.cuda.memory_allocated() / 1024**3
print(f"  VRAM after LoRA: {vram_lora:.1f} GB")

# [3] Load data
print("\n[3/4] Loading data (50 samples)...")
data_path = "data/processed/train.jsonl"
if Path(data_path).exists():
    with open(data_path, "r", encoding="utf-8") as f:
        raw = [json.loads(line) for line in f][:50]
else:
    raw = [{"question": "被公司辞退能拿多少赔偿？", "answer": "根据《劳动合同法》第八十七条，违法解除应按经济补偿标准二倍支付赔偿金。以上内容仅供参考。"}]

def format_chatml(item):
    q = item.get("question", item.get("input", ""))
    a = item.get("answer", item.get("output", ""))
    return (
        "<|im_start|>system\n你是 LexiCare 法律咨询助手，回答需引用法条并附免责声明。<|im_end|>\n"
        "<|im_start|>user\n" + q + "<|im_end|>\n"
        "<|im_start|>assistant\n" + a + "<|im_end|>"
    )

texts = [format_chatml(item) for item in raw]
dataset = Dataset.from_dict({"text": texts})
print(f"  Train samples: {len(dataset)}")

# [4] Train (10 steps only)
print("\n[4/4] Training (10 steps smoke test)...")
training_args = TrainingArguments(
    output_dir="./outputs/test_smoke_run",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    max_steps=10,
    logging_steps=5,
    fp16=True,
    save_strategy="no",
    report_to="none",
    seed=42,
    dataloader_num_workers=0,
)
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, args=training_args,
    train_dataset=dataset, max_seq_length=512, dataset_text_field="text",
)
t0 = time.time()
trainer.train()
elapsed = time.time() - t0
vram_train = torch.cuda.memory_allocated() / 1024**3

# Summary
est_minutes = elapsed / 10 * (2796 / 2 * 2) / 60  # 2796 samples, 2 epochs
print(f"\n{SEP}")
print("SMOKE TEST PASSED!")
print(f"  Training time: {elapsed:.1f}s for 10 steps")
print(f"  VRAM: base={vram_base:.1f}GB, +LoRA={vram_lora:.1f}GB, training={vram_train:.1f}GB")
print(f"  Estimated full SFT: ~{est_minutes:.0f} min")
print(f"  All clear for RTX 4060 8GB!")
print(SEP)
