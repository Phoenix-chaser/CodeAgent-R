"""Generate RFT (Rejection Sampling Fine-Tuning) training data with
FailForge-style failure recovery.

References:
- SWE-Gym: Training Software Engineering Agents and Verifiers
  (https://arxiv.org/abs/2412.21139, ICML 2025) — execution-feedback agent
  training paradigm
- FailForge: Distilling Procedural Competence from Rejection Sampling
  Fine-Tuning (https://arxiv.org/abs/2608.08570, 2026) — failure recovery
  via diagnostic hints
- Rejection Sampling Fine-Tuning for mathematical reasoning
  (https://arxiv.org/abs/2308.01825) — foundational RFT method

Phase 1 (Standard RFT, SWE-Gym ICML 2025):
    For each train task, sample N times from the current policy.
    Keep strict_pass outputs as self-generated SFT training data.

Phase 2 (FailForge, arXiv 2608.08570):
    For tasks where ALL samples failed (unsolvable), construct a
    diagnostic hint from grader_details, re-sample with the hint,
    and keep successful retries — stripping the hint at training time.

Output: SFT-format JSONL (messages + target) compatible with train_sft.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agentlab.benchmark.builder import build_all
from agentlab.benchmark.protocol import build_messages, load_specs, protocol_hash
from evaluate import evaluate_candidate


def make_hint(spec: dict, failures: list[dict]) -> str:
    """Construct a FailForge-style diagnostic hint from grader details."""
    parts = []
    types = Counter(f.get("failure_type") for f in failures)
    if types.get("private_test_failure"):
        parts.append("你的代码通过了公开测试但未通过隐藏的边界测试。"
                     "请仔细检查边界条件：空输入、极端值、特殊字符、"
                     "返回类型是否符合文档要求。")
    if types.get("public_test_failure"):
        parts.append("你的代码连公开测试都未通过，说明核心逻辑有误。"
                     "请重新阅读需求，确保基本功能正确。")
    if types.get("syntax_error"):
        parts.append("你输出的代码存在语法错误，请确保输出完整的合法 Python。")
    if types.get("invalid_patch"):
        parts.append("你的修改可能触及了不允许修改的文件。只修改主文件。")
    # 从判分器详情提取具体断言失败信息
    for f in failures[:2]:
        details = f.get("grader_details", {}).get("layers", {})
        functional = details.get("functional", {})
        tail = functional.get("private_tail") or functional.get("public_tail") or ""
        if tail:
            # 提取最后几行有意义的错误信息
            lines = [l.strip() for l in tail.splitlines() if l.strip()]
            if lines:
                parts.append("最后一次失败详情：" + " | ".join(lines[-2:]))
            break
    return "\n".join(parts) if parts else "请重新分析需求并修复代码。"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-file", required=True)
    parser.add_argument("--split", default="train", choices=["train", "dev"])
    parser.add_argument("--adapter", default=None,
                        help="SFT adapter path; omit for base model")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--samples", type=int, default=8,
                        help="Phase 1: samples per task for standard RFT")
    parser.add_argument("--retry-samples", type=int, default=4,
                        help="Phase 2: FailForge retry samples for unsolvable tasks")
    parser.add_argument("--max-per-task", type=int, default=4,
                        help="Max training samples per task (dedup by AST)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--quantization", choices=["nf4", "bf16"], default="nf4")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    specs = load_specs(args.tasks_file, args.split)
    output = Path(args.output)
    if output.exists():
        parser.error("output exists; use a new path")
    output.parent.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, set_seed)

    model_kwargs = dict(revision=args.revision, device_map={"": 0},
                        torch_dtype=torch.bfloat16, trust_remote_code=False,
                        attn_implementation="sdpa")
    if args.quantization == "nf4":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter or args.model, revision=None if args.adapter else args.revision,
        trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    model.config.use_cache = True

    def generate(messages, temp, seed_offset):
        set_seed(args.seed + seed_offset)
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt").to(model.device)
        gen = dict(max_new_tokens=args.max_new_tokens,
                   do_sample=temp > 0, pad_token_id=tokenizer.pad_token_id)
        if temp > 0:
            gen.update(temperature=temp, top_p=1.0)
        with torch.inference_mode():
            out = model.generate(inputs, **gen)
        return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)

    samples = []
    stats = Counter()
    started = time.time()

    with tempfile.TemporaryDirectory(prefix="rft-") as temporary:
        work = Path(temporary)
        build_all(work / "tasks", specs)
        seen_ast = {}  # task_id -> set of AST fingerprints

        # ── Phase 1: Standard RFT ──
        print("── Phase 1: Standard RFT ──", flush=True)
        for pos, spec in enumerate(specs, 1):
            messages = build_messages(spec)
            task_pass = []
            task_fail = []
            for si in range(args.samples):
                raw = generate(messages, args.temperature, pos * 1000 + si)
                grade_dir = work / f"g1-{spec['task_id']}-{si}"
                grade_dir.mkdir()
                rec = evaluate_candidate(spec, raw, work / "tasks", grade_dir,
                                         dict(phase=1, sample=si))
                if rec["strict_pass"]:
                    task_pass.append(rec)
                else:
                    task_fail.append(rec)
                stats[f"p1_{rec['verdict']}"] += 1
            for rec in task_pass:
                fp = _ast_fp(rec["generated_code"])
                bucket = seen_ast.setdefault(spec["task_id"], set())
                samples_with_fp = sum(1 for s in samples
                                      if s["task_id"] == spec["task_id"])
                if fp not in bucket and samples_with_fp < args.max_per_task:
                    bucket.add(fp)
                    samples.append(_make_sample(spec, rec, "rft_pass"))
            if task_pass:
                stats["task_solvable"] += 1
            else:
                stats["task_unsolvable"] += 1
            print(f"[{pos}/{len(specs)}] {spec['task_id']} "
                  f"pass={len(task_pass)}/{args.samples}", flush=True)
            # 保存失败详情供 Phase 2 用
            if not task_pass:
                spec["_failures"] = task_fail

        # ── Phase 2: FailForge recovery ──
        unsolvable = [s for s in specs if s.get("_failures")]
        print(f"\n── Phase 2: FailForge ({len(unsolvable)} unsolvable) ──",
              flush=True)
        for pos, spec in enumerate(unsolvable, 1):
            hint = make_hint(spec, spec["_failures"])
            base_messages = build_messages(spec)
            hint_messages = [
                base_messages[0],
                {"role": "user",
                 "content": base_messages[1]["content"] +
                 f"\n\n【诊断提示】\n{hint}\n请根据以上提示修复代码。"},
            ]
            recovered = 0
            for si in range(args.retry_samples):
                raw = generate(hint_messages, args.temperature,
                               90000 + pos * 100 + si)
                grade_dir = work / f"g2-{spec['task_id']}-{si}"
                grade_dir.mkdir()
                rec = evaluate_candidate(spec, raw, work / "tasks", grade_dir,
                                         dict(phase=2, sample=si, hint=hint))
                stats[f"p2_{rec['verdict']}"] += 1
                if rec["strict_pass"]:
                    fp = _ast_fp(rec["generated_code"])
                    bucket = seen_ast.setdefault(spec["task_id"], set())
                    if fp not in bucket:
                        bucket.add(fp)
                        # 训练时用原始 prompt（不含提示），target 为修复代码
                        samples.append(_make_sample(spec, rec, "failforge"))
                        recovered += 1
                        break  # 恢复一题即停
            stats["failforge_recovered" if recovered else "failforge_failed"] += 1
            print(f"  [{pos}/{len(unsolvable)}] {spec['task_id']} "
                  f"{'✓ recovered' if recovered else '✗ not recovered'}",
                  flush=True)

    # ── 写输出 ──
    with output.open("x", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": 1,
        "protocol": {"tasks_hash": protocol_hash(specs), "split": args.split,
                     "model": args.model, "adapter": args.adapter,
                     "samples": args.samples, "retry_samples": args.retry_samples,
                     "max_per_task": args.max_per_task,
                     "temperature": args.temperature, "seed": args.seed},
        "total_samples": len(samples),
        "tasks_covered": len({s["task_id"] for s in samples}),
        "by_source": dict(Counter(s["provenance"]["source"] for s in samples)),
        "stats": dict(stats),
        "elapsed_s": round(time.time() - started, 1),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRFT 数据：{len(samples)} 条 / {manifest['tasks_covered']} 题")
    print(f"来源分布：{manifest['by_source']}")
    print(f"耗时：{manifest['elapsed_s']}s")


def _ast_fp(code: str) -> str:
    try:
        import ast as _ast
        return hashlib.md5(
            _ast.dump(_ast.parse(code), include_attributes=False).encode()).hexdigest()
    except (SyntaxError, ValueError):
        return f"syn:{hashlib.md5(code.encode()).hexdigest()}"


def _make_sample(spec: dict, rec: dict, source: str) -> dict:
    return {
        "sample_id": f"rft-{spec['task_id']}-{source}-{rec.get('sample_index', 0):03d}",
        "task_id": spec["task_id"],
        "source_group": spec["source_group"],
        "split": spec["split"],
        "messages": build_messages(spec),  # 原始 prompt（无提示）
        "target": {"role": "assistant", "content": rec["generated_code"]},
        "provenance": {
            "source": source,
            "strict_pass": True,
            "generator": "self",
            "sample_index": rec.get("sample_index"),
            "revalidated": True,
        },
    }


if __name__ == "__main__":
    main()
