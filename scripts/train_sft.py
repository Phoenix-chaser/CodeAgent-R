# -*- coding: utf-8 -*-
"""Reproducible Qwen QLoRA SFT entry point."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agentlab.training.hf_sft import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SFT：Qwen + 单卡 QLoRA，严格 assistant-only mask")
    parser.add_argument("--data", required=True, help="JSONL train split；strict_pass=true")
    parser.add_argument("--eval-data", default=None, help="独立 JSONL dev split；真实训练必需")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--revision", default=None, help="基座模型 commit/revision")
    parser.add_argument("--output", default="checkpoints/sft-v2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="只审计数据，不加载模型")
    parser.add_argument("--save-every-epoch", action=argparse.BooleanOptionalAction,
                        default=True, help="保留各 epoch adapter，供独立 dev 执行评测选择（默认开启）")
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false", default=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    train(args)
    if args.dry_run:
        print("Dry run 完成；真实训练必须提供 --eval-data。")
    else:
        print(f"训练完成：dev token loss 选择的 adapter 已保存至 {args.output}/adapter")
        if args.save_every_epoch:
            print(f"各 epoch adapter 索引：{args.output}/checkpoint-index.json；请在 dev 执行评测后选择")
        print(f"完整 manifest：{args.output}/training-manifest.json")


if __name__ == "__main__":
    main()
