"""quick_rag_test.py — 测试 RAG 检索 + 模型生成（验证检索白名单防编造法条）。"""
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from app.domain_config import get_domain
from app.rag_retriever import get_retriever

ADAPTER = "outputs/lora_weights/law-lora-r16-20260814-1138"
RAG_INSTRUCTION = "根据以下检索到的法条回答问题，必须引用检索结果中真实存在的条文（《法名》第X条），不得引用未提供的法条。"

def main():
    print("加载 RAG 检索器...")
    retriever = get_retriever()

    print("加载 Qwen3-4B (4-bit) + LoRA...")
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

    sys_prompt = get_domain().default_system_prompt
    questions = ["被公司违法辞退能拿多少赔偿？", "劳动仲裁的时效是多久？"]

    for q in questions:
        docs = retriever.retrieve(q, top_k=3)
        print(f"\n{'='*70}\nQ: {q}\n检索到 {len(docs)} 篇相关法条:")
        for d in docs:
            print(f"  · {d.get('title','')}: {d.get('content','')[:70]}...")
        context = retriever.format_context(docs)

        prompt = (
            f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{RAG_INSTRUCTION}\n{context}\n\n问题：{q}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=300,
                do_sample=True, temperature=0.7, top_p=0.9, repetition_penalty=1.15,
            )
        answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"A: {answer}")

if __name__ == "__main__":
    main()
