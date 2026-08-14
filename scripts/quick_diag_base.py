"""quick_diag_base.py — 诊断：基座 Qwen3-4B（无 LoRA）的 RAG 引用能力。"""
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from app.domain_config import get_domain
from app.rag_retriever import get_retriever

RAG_INSTRUCTION = "根据以下检索到的法条回答问题，必须引用检索结果中真实存在的条文（《法名》第X条），不得引用未提供的法条。"

def main():
    print("加载 RAG 检索器 + 基座 Qwen3-4B（无 LoRA）...")
    retriever = get_retriever()
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True,
                                                 device_map="auto", quantization_config=bnb)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    sys_prompt = get_domain().default_system_prompt
    questions = ["被公司违法辞退能拿多少赔偿？", "劳动仲裁的时效是多久？"]

    for q in questions:
        docs = retriever.retrieve(q, top_k=3)
        context = retriever.format_context(docs)
        print(f"\n{'='*70}\nQ: {q}\n检索到: {[d.get('title','')[:20] for d in docs]}")
        prompt = (f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
                  f"<|im_start|>user\n{RAG_INSTRUCTION}\n{context}\n\n问题：{q}<|im_end|>\n<|im_start|>assistant\n")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=300, do_sample=True,
                                 temperature=0.7, top_p=0.9, repetition_penalty=1.15)
        ans = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        cited = sum(ans.count(f"《{d.get('title','')[:6]}") for d in docs)
        print(f"A: {ans}\n   [引用检索法条的次数约: {cited}]")

if __name__ == "__main__":
    main()
