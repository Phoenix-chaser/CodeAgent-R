"""Build DPO preference pairs: teacher success (chosen) vs student failure.

Runs an SFT adapter (or the base model) on frozen train tasks, grades every
sample with the three-layer grader, and pairs each strict-fail generation with
one strict-pass teacher answer for the same task. The shared public prompt is
identical by construction, so the P06/P08 common-state contract holds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from agentlab.benchmark.task_store import TaskStore
from evaluate import evaluate_candidate
from collect_data import grade_candidate


def teacher_solutions(candidates_file: str, specs: list[dict]) -> dict[str, list[str]]:
    """One or more strict-pass teacher targets per task (canonical prompt)."""
    by_task: dict[str, list[str]] = {}
    for line in Path(candidates_file).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("provenance", {}).get("strict_pass") is not True:
            continue
        if record.get("messages") != build_messages(
                next(s for s in specs if s["task_id"] == record["task_id"])):
            continue
        by_task.setdefault(record["task_id"], []).append(record["target"]["content"])
    return by_task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-file", required=True, help="Frozen task JSONL")
    parser.add_argument("--split", default="train", choices=["train", "dev"])
    parser.add_argument("--candidates-file", required=True,
                        help="Teacher candidates JSONL for the same split")
    parser.add_argument("--adapter", default=None,
                        help="SFT adapter path; omit to sample the base model")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--samples", type=int, default=4,
                        help="Student generations per task")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--quantization", choices=["nf4", "bf16"], default="nf4")
    parser.add_argument("--min-failures-per-task", type=int, default=1,
                        help="Pair at most this many failures per task with the teacher")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.samples < 1 or args.temperature < 0:
        parser.error("samples must be >= 1 and temperature >= 0")

    specs = load_specs(args.tasks_file, args.split)
    teachers = teacher_solutions(args.candidates_file, specs)
    missing = [s["task_id"] for s in specs if s["task_id"] not in teachers]
    if missing:
        parser.error(f"{len(missing)} tasks lack strict-pass teacher answers: "
                     f"{missing[:5]}")

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, set_seed)

    output = Path(args.output)
    if output.exists():
        parser.error("output exists; use a new path")
    output.parent.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema_version": 1, "tasks_hash": protocol_hash(specs),
        "split": args.split, "model": args.model, "adapter": args.adapter,
        "revision": args.revision, "samples": args.samples,
        "temperature": args.temperature, "seed": args.seed,
        "max_new_tokens": args.max_new_tokens, "quantization": args.quantization,
        "teacher_candidates": str(Path(args.candidates_file).resolve()),
    }
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
    resolved = getattr(model.config, "_commit_hash", None) or args.revision
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    model.config.use_cache = True

    pairs = []
    stats = Counter()
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="dpo-pref-") as temporary:
        work = Path(temporary)
        build_all(work / "tasks", specs)
        store = TaskStore(work / "tasks")
        for position, spec in enumerate(specs, 1):
            messages = build_messages(spec)
            for sample_index in range(args.samples):
                set_seed(args.seed + position * 1000 + sample_index)
                inputs = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_tensors="pt").to(model.device)
                generation = dict(max_new_tokens=args.max_new_tokens,
                                  do_sample=args.temperature > 0,
                                  pad_token_id=tokenizer.pad_token_id)
                if args.temperature > 0:
                    generation.update(temperature=args.temperature, top_p=1.0)
                with torch.inference_mode():
                    out = model.generate(inputs, **generation)
                raw = tokenizer.decode(out[0][inputs.shape[1]:],
                                       skip_special_tokens=True)
                grade_dir = work / f"grade-{spec['task_id']}-{sample_index}"
                grade_dir.mkdir()
                record = evaluate_candidate(spec, raw, work / "tasks", grade_dir,
                                            dict(sample_index=sample_index))
                verdict = record["verdict"]
                stats[f"student_{verdict}"] += 1
                if record["strict_pass"]:
                    continue
                failures = [p for p in pairs if p["task_id"] == spec["task_id"]]
                if len(failures) >= args.min_failures_per_task:
                    stats["cap_reached"] += 1
                    continue
                chosen = teachers[spec["task_id"]][len(failures) % len(teachers[spec["task_id"]])]
                pairs.append({
                    "pair_id": f"dpo-{spec['task_id']}-{len(failures):02d}",
                    "task_id": spec["task_id"],
                    "source_group": spec["source_group"],
                    "split": args.split,
                    "messages": messages,
                    "chosen": {"role": "assistant", "content": chosen},
                    "rejected": {"role": "assistant", "content": record["generated_code"]},
                    "provenance": {
                        "chosen_source": "teacher_strict_pass",
                        "rejected_source": "student_" + str(args.adapter or "base"),
                        "rejected_verdict": verdict,
                        "rejected_failure_type": record.get("failure_type"),
                        "rejected_sample_index": sample_index,
                        "student_model": args.model,
                        "student_adapter": args.adapter,
                        "resolved_revision": resolved,
                        "temperature": args.temperature,
                    },
                })
                print(f"[{position}/{len(specs)}] {spec['task_id']} "
                      f"pair+={len(failures) + 1} ({verdict})", flush=True)
    with output.open("x", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    tasks_covered = len({p["task_id"] for p in pairs})
    manifest = {
        "schema_version": 1, "protocol": protocol,
        "pairs": len(pairs), "tasks": len(specs),
        "tasks_with_pairs": tasks_covered,
        "stats": dict(stats), "elapsed_s": round(time.time() - started, 1),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in
                      ("pairs", "tasks", "tasks_with_pairs", "stats")}),
          ensure_ascii=False, flush=True)


if __name__ == "__main__":
    main()
