"""
03b_build_cot_data.py — 构建法律 Chain-of-Thought 训练数据（可选）。

将法律 SFT 数据转换为结构化推理格式：
  <分析> 法律事实与依据 </分析>
  <建议> 结构化建议 + 免责声明 </建议>

用途：
  1. 训练模型「先分析后回答」（引用法条）
  2. 为 GRPO 奖励函数提供结构化输出
  3. 生产环境可解析输出

注：本项目思考模式仅用部署开关（enable_thinking），此 CoT 数据为可选增强。

用法：
    python scripts/03b_build_cot_data.py
    python scripts/03b_build_cot_data.py --input data/processed/train.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.domain_config import get_domain


COT_SYSTEM_PROMPT = (
    "你是 LexiCare 法律咨询助手。在回答每个问题前，先进行结构化分析：\n"
    "1. 识别用户的核心法律问题\n"
    "2. 引用相关法条作为法律依据\n"
    "3. 给出分层建议（证据→途径→咨询律师）\n"
    "4. 结尾附免责声明\n\n"
    "输出格式：\n"
    "<分析>\n"
    "- 核心问题: ...\n"
    "- 法律依据: 《XX法》第X条\n"
    "- 风险/时效: ...\n"
    "</分析>\n"
    "<建议>\n"
    "- 证据: ...\n"
    "- 途径: ...\n"
    "- 咨询律师: ...\n"
    "- 免责声明: 以上内容仅供参考，不构成法律意见\n"
    "</建议>"
)


def classify_question_type(question: str, answer: str) -> str:
    """按法律子领域分类。"""
    return get_domain().classify_category(question + " " + answer)


def extract_key_info(question: str, answer: str, q_type: str) -> dict:
    """提取关键信息填充模板。"""
    info = {
        "question_summary": question[:50],
        "answer": answer,
        "core_issue": question[:60],
    }
    # 法条引用
    cites = re.findall(r'《[^》]{2,20}》\s*第\s*[0-9一二三四五六七八九十百零〇]+\s*条', answer)
    info["legal_basis"] = "、".join(cites[:2]) if cites else "请引用相关法条"
    # 时效/风险
    if "时效" in answer:
        m = re.search(r'[^。；]*时效[^。；]*', answer)
        info["risk"] = m.group()[:60] if m else "注意相关时效"
    else:
        info["risk"] = "请关注相关时效与举证责任"
    return info


def convert_to_cot(item: dict) -> dict:
    """将单条数据转为 CoT 格式。"""
    if "messages" in item:
        msgs = item["messages"]
        user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
        assistant_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
    elif "question" in item and "answer" in item:
        user_msg, assistant_msg = item["question"], item["answer"]
    else:
        return item

    if not user_msg or not assistant_msg:
        return item

    q_type = classify_question_type(user_msg, assistant_msg)
    info = extract_key_info(user_msg, assistant_msg, q_type)

    cot_answer = (
        "<分析>\n"
        f"- 核心问题: {info['core_issue']}\n"
        f"- 法律依据: {info['legal_basis']}\n"
        f"- 风险/时效: {info['risk']}\n"
        "</分析>\n"
        "<建议>\n"
        f"{assistant_msg}\n"
        "</建议>"
    )

    return {
        "messages": [
            {"role": "system", "content": COT_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": cot_answer},
        ]
    }


def build_cot_dataset(input_path: str, output_path: str, max_samples: int = None):
    print(f"Loading data from: {input_path}")
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"  Total samples: {len(data)}")
    if max_samples:
        data = data[:max_samples]
        print(f"  Using: {len(data)} samples")

    print("\nConverting to CoT format...")
    cot_data = []
    skipped = 0
    for item in data:
        try:
            cot_data.append(convert_to_cot(item))
        except Exception:
            skipped += 1
    if skipped:
        print(f"  ⚠ Skipped {skipped} items due to errors")

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in cot_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ Saved {len(cot_data)} CoT samples to: {output_path}")

    if cot_data:
        print("\n样例：")
        for msg in cot_data[0].get("messages", []):
            print(f"[{msg['role']}]", msg["content"][:200])
    return cot_data


def main():
    parser = argparse.ArgumentParser(description="Build CoT training data for LexiCare")
    parser.add_argument("--input", type=str, default="data/processed/train.jsonl")
    parser.add_argument("--output", type=str, default="data/processed/train_cot.jsonl")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"⚠ Input file not found: {args.input}")
        return

    build_cot_dataset(args.input, args.output, args.max_samples)


if __name__ == "__main__":
    main()
