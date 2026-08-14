"""quick_infer_test.py — 快速加载 LoRA 适配器，测试法律问答效果。"""
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from app.domain_config import get_domain

ADAPTER = "outputs/lora_weights/law-lora-r16-20260814-1138"

def main():
    print("加载 Qwen3-4B (4-bit) + LoRA 适配器...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B", trust_remote_code=True,
        device_map="auto", quantization_config=bnb,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    print(f"适配器加载完成: {ADAPTER}\n")

    sys_prompt = get_domain().default_system_prompt

    questions = [
        "被公司违法辞退能拿多少赔偿？",
        "签了合同交了定金，不想买了能退吗？",
        "劳动仲裁的时效是多久？",
        "离婚时夫妻共同财产怎么分割？",
    ]

    for q in questions:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": q},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=300,
                do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.15,
            )
        answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"Q: {q}")
        print(f"A: {answer}\n")
        print("=" * 70)

if __name__ == "__main__":
    main()
