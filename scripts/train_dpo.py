# -*- coding: utf-8 -*-
"""DPO entry point: refine an SFT adapter on teacher-vs-student preferences."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agentlab.training.hf_dpo import train_dpo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DPO：SFT adapter 偏好精炼")
    parser.add_argument("--data", required=True, help="Preference pair JSONL")
    parser.add_argument("--sft-adapter", required=True,
                        help="SFT adapter directory (policy init AND frozen reference)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every-epoch", dest="save_every_epoch",
                        action="store_true", default=True)
    parser.add_argument("--no-save-every-epoch", dest="save_every_epoch",
                        action="store_false")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest = train_dpo(args)
    print(f"DPO 完成：{manifest.get('optimizer_steps')} 步，"
          f"输出 {args.output}")


if __name__ == "__main__":
    main()
