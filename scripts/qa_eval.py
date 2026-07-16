"""会议问答离线评估脚本（RAG 设计 §6.2）。

用法：
    uv run python scripts/qa_eval.py --dataset eval/qa_dataset.json
    uv run python scripts/qa_eval.py --dataset eval/qa_dataset.json --rrf-k 10,30,60
    uv run python scripts/qa_eval.py --dataset eval/qa_dataset.json --with-answers

- 默认只评检索（零 LLM 消耗）；--with-answers 会真实调用当前配置的
  LLM provider 生成回答（注意额度），额外产出引用准确率/拒答率/置信度分布
- --rrf-k 接受逗号分隔的多个值，逐 k 输出一份报告（评审 §8.3 对比 10/30/60）
- 数据集格式见 eval/qa_dataset.example.json
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="会议问答离线评估")
    parser.add_argument("--dataset", required=True, help="数据集 JSON 路径")
    parser.add_argument(
        "--rrf-k", default=None,
        help="RRF k 值，逗号分隔可多组对比（如 10,30,60）；缺省用当前配置",
    )
    parser.add_argument(
        "--with-answers", action="store_true",
        help="完整生成回答并评引用/拒答（真实消耗 LLM 额度）",
    )
    args = parser.parse_args()

    from app.services.qa_eval import evaluate, load_dataset

    items = load_dataset(args.dataset)
    ks = (
        [int(k.strip()) for k in args.rrf_k.split(",") if k.strip()]
        if args.rrf_k
        else [None]
    )
    reports = [
        asyncio.run(evaluate(items, rrf_k=k, with_answers=args.with_answers))
        for k in ks
    ]
    print(json.dumps(reports if len(reports) > 1 else reports[0],
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
