"""模型冒烟测试：加载 → 挂 LoRA → 一步训练 → 一次生成。

在付费采集和长时间训练之前运行。首次运行会下载权重，之后一两分钟内给出结果；
任何一项检查失败都以非零状态退出，并说明失败在哪一步。走的是与正式训练、评测
相同的代码路径和参数。

    python scripts/smoke_model.py                               # 默认模型，NF4，GPU
    python scripts/smoke_model.py --quantization bf16           # 不量化（9B 约需 18GB）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agentlab.benchmark.builder import SPECS
from agentlab.benchmark.protocol import build_messages
from agentlab.models import DEFAULT_MODEL
from agentlab.training.hf_runtime import (
    QUANTIZATIONS, assert_model_supported, load_causal_lm, load_tokenizer, resolve_revision,
)
from agentlab.training.hf_smoke import smoke_test
from _shared_extract import extract_code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=None,
                        help="基座模型 commit/revision；默认用该模型已核对的固定版本")
    parser.add_argument("--quantization", choices=QUANTIZATIONS, default="nf4")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", help="把完整报告写成 JSON")
    parser.add_argument("--cpu", action="store_true",
                        help="仅供开发自测：在 CPU 上跑（只能配合 --quantization bf16）")
    args = parser.parse_args()
    if args.cpu and args.quantization != "bf16":
        parser.error("CPU self-test needs --quantization bf16: NF4 requires CUDA")

    args.revision = resolve_revision(args.model, args.revision)
    assert_model_supported(args.model, args.revision)
    print(f"加载 {args.model} @ {args.revision or 'main'}（{args.quantization}）；"
          "首次运行需要下载权重，之后一两分钟内给出结果。", flush=True)
    tokenizer = load_tokenizer(args.model, args.revision, padding_side="right")
    model = load_causal_lm(args.model, args.revision, args.quantization,
                           device="cpu" if args.cpu else None)
    print(f"已加载 {type(model).__name__}，4bit={getattr(model, 'is_loaded_in_4bit', False)}",
          flush=True)

    spec = SPECS[0]                           # 一道内置的简单修复题
    report = smoke_test(model, tokenizer, build_messages(spec),
                        {"role": "assistant", "content": spec["fixed"]},
                        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                        max_length=args.max_length, max_new_tokens=args.max_new_tokens,
                        extractor=extract_code)
    report.update(model=args.model, revision=args.revision, quantization=args.quantization,
                  task_id=spec["task_id"])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
    if not all(report["fast_kernels"].values()):
        print("提示：未安装线性注意力的加速内核，Qwen3.5 会走慢速的参考实现（结果正确）。"
              "需要提速可执行：python -m pip install causal-conv1d flash-linear-attention", flush=True)
    failed = [name for name, passed in report["checks"].items() if not passed]
    print(("冒烟测试通过。" if report["passed"] else "冒烟测试失败，未通过的检查: " + ", ".join(failed)),
          flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
