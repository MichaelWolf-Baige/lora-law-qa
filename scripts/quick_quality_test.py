"""Quick quality test: compare base model vs LoRA checkpoint on legal questions."""
import torch, time, sys
from pathlib import Path
sys.path.insert(0, ".")

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SEP = "=" * 70

TEST_QUESTIONS = [
    "被公司辞退能拿多少赔偿？",
    "竞业限制没有给补偿，还有效吗？",
    "劳动仲裁的时效是多久？",
    "交了定金不买了，定金能退吗？",
]

SYSTEM_PROMPT = (
    "你是 LexiCare 法律咨询助手，依据法律法规与司法解释回答问题，"
    "引用具体法条，结尾附免责声明。你不提供正式法律意见，不对案件结果作承诺。"
)

def generate(model, tokenizer, question, max_tokens=300):
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens,
            temperature=0.7, top_p=0.9, do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def main():
    model_name = "Qwen/Qwen3-8B"
    lora_path = "outputs/lora_weights/law-lora-r16-XXXX"  # 替换为实际 LoRA 路径

    print(SEP)
    print("LexiCare 训练质量快速测试")
    print(SEP)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print(f"\n[1] 加载基座模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    print(f"    基座模型 VRAM: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

    # Test base model
    print(f"\n[2] 测试基座模型...")
    for q in TEST_QUESTIONS[:2]:
        t0 = time.time()
        ans = generate(base_model, tokenizer, q)
        print(f"\n  Q: {q}")
        print(f"  基座: {ans[:150]}...")

    # Free base model
    del base_model
    torch.cuda.empty_cache()

    # Load LoRA model
    print(f"\n[3] 加载 LoRA checkpoint-200...")
    lora_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    lora_model = PeftModel.from_pretrained(lora_model, lora_path)
    lora_model = lora_model.merge_and_unload()
    lora_model = lora_model.to(torch.float16)  # Fix dtype mismatch after DoRA merge
    lora_model.eval()
    print(f"    LoRA模型 VRAM: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

    # Test LoRA model
    print(f"\n[4] 测试微调模型 (checkpoint-200)...")
    for q in TEST_QUESTIONS:
        t0 = time.time()
        ans = generate(lora_model, tokenizer, q)
        print(f"\n  Q: {q}")
        print(f"  微调: {ans[:250]}...")
        print(f"  ({time.time()-t0:.1f}s)")

    print(f"\n{SEP}")
    print("测试完成！对比上面基座 vs 微调的回答质量")
    print(SEP)


if __name__ == "__main__":
    main()
