"""chat.py — 与训练好的法律模型交互对话（RAG + LoRA）。

加载 Qwen3-4B + r8 LoRA 适配器 + RAG 检索，交互式问答。
输入 exit / q 退出。

用法：
    python scripts/chat.py
"""
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from app.domain_config import get_domain
from app.rag_retriever import get_retriever

ADAPTER = "outputs/lora_weights/law-lora-r8-20260814-1732"
RAG_INSTRUCTION = "根据以下检索到的法条回答问题。若检索结果与问题相关，引用其中真实存在的条文（《法名》第X条）；若检索结果不足以直接回答，给出一般性法律说明、说明依据不足，并建议核实或咨询执业律师；不得编造法条。"


def main():
    print("加载模型 + RAG 检索器（约 1 分钟）...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True,
                                                 device_map="auto", quantization_config=bnb)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    retriever = get_retriever()
    sys_prompt = get_domain().default_system_prompt
    print(f"就绪！输入法律问题开始（输入 q 退出）\n")

    while True:
        q = input("你：").strip()
        if q.lower() in ("q", "exit", "quit", "退出"):
            print("再见！")
            break
        if not q:
            continue

        # RAG 检索
        docs = retriever.retrieve(q, top_k=3)
        context = retriever.format_context(docs)
        ctx_block = context if context else "【参考法律法规】未检索到直接相关条文。"

        prompt = (f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
                  f"<|im_start|>user\n{RAG_INSTRUCTION}\n{ctx_block}\n\n问题：{q}<|im_end|>\n"
                  f"<|im_start|>assistant\n")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=400, do_sample=True,
                                 temperature=0.7, top_p=0.9, repetition_penalty=1.15,
                                 enable_thinking=False)
        ans = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if "</think>" in ans:
            ans = ans.split("</think>")[-1].strip()

        print(f"\n助手：{ans}\n")
        if docs:
            print(f"  📚 检索到：{'、'.join(d.get('title','')[:16] for d in docs)}\n")


if __name__ == "__main__":
    main()
