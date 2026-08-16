"""doc_qa.py — 通用文档问答 CLI（实时摄入任意 PDF/txt）。

用法：
    python scripts/doc_qa.py --doc 你的文件.pdf
    python scripts/doc_qa.py --doc 你的文件.txt --base_only   # 不用法律 LoRA，用基座模型
    python scripts/doc_qa.py --doc 你的文件.pdf --no_dense    # 只 BM25（更快，Dense 需建索引）
"""
import sys, argparse, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from app.document_qa import DocumentQA

ADAPTER = "outputs/lora_weights/law-lora-r8-20260814-1732"
BASE_MODEL = "Qwen/Qwen3-4B"
GENERIC_SYSTEM = "你是一个文档问答助手，根据用户提供的文档内容回答问题，引用具体信息，用中文回答。"
GENERIC_RAG = ("根据以下从文档中检索到的内容回答问题。优先引用检索内容作答；"
               "若检索内容不足以回答，明确说明「文档中没有足够信息」，不要编造。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", required=True, help="PDF/txt 文档路径")
    ap.add_argument("--base_only", action="store_true", help="不用法律 LoRA，直接用基座模型")
    ap.add_argument("--no_dense", action="store_true", help="只 BM25，不建 Dense 索引（更快）")
    args = ap.parse_args()

    print(f"摄入文档: {args.doc}")
    docqa = DocumentQA(args.doc, use_dense=not args.no_dense)
    print(f"  分块 {docqa.n_chunks} 个，索引完成")

    print("加载模型（约 1 分钟）...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, trust_remote_code=True,
                                                 device_map="auto", quantization_config=bnb)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not args.base_only and Path(ADAPTER).exists():
        model = PeftModel.from_pretrained(model, ADAPTER)
        print("  已加载法律 LoRA 适配器")
    if hasattr(model.generation_config, "enable_thinking"):
        model.generation_config.enable_thinking = False   # 关闭 Qwen3 思考模式
    model.eval()

    print(f"就绪！针对文档提问（输入 q 退出）\n")
    while True:
        q = input("你：").strip()
        if q.lower() in ("q", "exit", "退出"):
            print("再见！")
            break
        if not q:
            continue

        docs = docqa.retrieve(q, top_k=3)
        context = docqa.format_context(docs)
        ctx = context if context else "【文档内容】未检索到相关内容。"
        prompt = (f"<|im_start|>system\n{GENERIC_SYSTEM}<|im_end|>\n"
                  f"<|im_start|>user\n{GENERIC_RAG}\n{ctx}\n\n问题：{q}<|im_end|>\n"
                  f"<|im_start|>assistant\n")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=400, do_sample=True,
                                 temperature=0.7, top_p=0.9, repetition_penalty=1.15)
        ans = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        if "</think>" in ans:
            ans = ans.split("</think>")[-1].strip()

        print(f"\n助手：{ans}\n")
        if docs:
            print(f"  📄 检索到 {len(docs)} 段相关文本\n")


if __name__ == "__main__":
    main()
