"""Reproducible code-repair evaluation with complete generation evidence.

Best-of-N strategy inspired by:
- PerfCodeGen (https://arxiv.org/abs/2412.03578)
- Self-Execution Simulation (https://arxiv.org/abs/2604.03253)
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agentlab.benchmark.builder import build_all
from agentlab.benchmark.grader import grade
from agentlab.benchmark.protocol import (
    build_messages, load_specs, prompt_fingerprint, protocol_hash,
    spec_fingerprint,
)
from agentlab.benchmark.task_store import TaskStore
from _shared_extract import extract_code


SCHEMA_VERSION = "codeagent-eval-v2"


def content_hash(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_patch(module_path, buggy, fixed):
    return "".join(difflib.unified_diff(
        buggy.splitlines(keepends=True), fixed.splitlines(keepends=True),
        fromfile=f"a/{module_path}", tofile=f"b/{module_path}"))


def grade_one(spec, fixed_code, task_dir, grade_dir):
    patch_file = grade_dir / f"eval-{spec['task_id']}.diff"
    patch_file.write_text(make_patch(spec["module"], spec["buggy"], fixed_code),
                          encoding="utf-8")
    task = TaskStore(task_dir).load(spec["task_id"])
    result, details = grade(task, patch_file, grade_dir / "ws")
    return result.verdict, result.strict_pass, details


def summarize_results(results: list[dict]) -> dict:
    grouped = defaultdict(list)
    for result in results:
        grouped[result["task_id"]].append(result)
    task_rates = {
        task_id: sum(bool(item.get("strict_pass")) for item in attempts) / len(attempts)
        for task_id, attempts in grouped.items()
    }
    total_passes = sum(bool(item.get("strict_pass")) for item in results)
    any_pass = sum(any(item.get("strict_pass", False) for item in attempts)
                   for attempts in grouped.values())
    infra_errors = sum(item.get("failure_type") == "infrastructure_error"
                       for item in results)
    return {
        "n_tasks": len(grouped), "n_attempts": len(results),
        "n_passed_attempts": total_passes,
        "pass_at_1": sum(task_rates.values()) / len(task_rates) if task_rates else 0.0,
        "any_pass_rate": any_pass / len(grouped) if grouped else 0.0,
        "n_tasks_any_pass": any_pass, "infrastructure_errors": infra_errors,
        "valid": infra_errors == 0,
        "task_pass_rates": task_rates,
        "failure_counts": dict(Counter(item.get("failure_type") for item in results
                                       if not item.get("strict_pass"))),
    }


def training_overlap(specs: list[dict], path: str | Path | None,
                     include_selection: bool = False) -> dict:
    if path is None:
        return {"status": "not_checked", "source": None, "task_ids": [],
                "source_groups": [], "prompt_hashes": []}
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    task_ids, source_groups, prompt_hashes = set(), set(), set()
    if source.suffix.lower() == ".jsonl":
        for line in text.splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("task_id"):
                task_ids.add(record["task_id"])
            if record.get("source_group"):
                source_groups.add(record["source_group"])
            if record.get("messages"):
                prompt_hashes.add(content_hash(record["messages"]))
    else:
        manifest = json.loads(text)
        data = manifest.get("data", manifest.get("train", manifest))
        task_ids.update(data.get("task_ids", []))
        source_groups.update(data.get("source_groups", []))
        prompt_hashes.update(data.get("prompt_hashes", []))
        if include_selection:
            selection_data = manifest.get("eval_data") or {}
            task_ids.update(selection_data.get("task_ids", []))
            source_groups.update(selection_data.get("source_groups", []))
            prompt_hashes.update(selection_data.get("prompt_hashes", []))
    if not any((task_ids, source_groups, prompt_hashes)):
        raise ValueError("Training data has no task/source/prompt identities")
    overlaps = {
        "task_ids": sorted(task_ids & {spec["task_id"] for spec in specs}),
        "source_groups": sorted(source_groups & {spec["source_group"] for spec in specs}),
        "prompt_hashes": sorted(prompt_hashes & {content_hash(build_messages(spec))
                                                 for spec in specs}),
    }
    return {"status": "contaminated" if any(overlaps.values()) else "clean",
            "source": str(source.resolve()), "sha256": hashlib.sha256(text.encode()).hexdigest(),
            **overlaps}


def failure_type(code: str, verdict: str, details: dict) -> str | None:
    if verdict == "strict_pass":
        return None
    if not code.strip():
        return "empty_response"
    try:
        ast.parse(code)
    except SyntaxError:
        return "syntax_error"
    if verdict == "invalid":
        return "invalid_patch"
    if verdict == "partial":
        return "private_test_failure"
    return "public_test_failure"


def evaluate_candidate(spec, raw, task_dir, grade_dir, metadata=None):
    fixed_code = extract_code(raw)
    record = dict(task_id=spec["task_id"], source_group=spec["source_group"],
                  split=spec["split"], difficulty=spec["difficulty"],
                  prompt_hash=prompt_fingerprint(spec), raw_response=raw,
                  generated_code=fixed_code, **(metadata or {}))
    try:
        verdict, strict_pass, details = grade_one(spec, fixed_code, task_dir, grade_dir)
        record.update(verdict=verdict, strict_pass=strict_pass, grader_details=details,
                      failure_type=failure_type(fixed_code, verdict, details))
    except subprocess.TimeoutExpired as error:
        record.update(verdict="timeout", strict_pass=False,
                      failure_type="execution_timeout", error=str(error), grader_details={})
    except Exception as error:
        record.update(verdict="error", strict_pass=False,
                      failure_type="infrastructure_error", error=f"{type(error).__name__}: {error}",
                      grader_details={})
    return record


def passes_public_test(spec, code, task_dir, grade_dir):
    """Quick check: does this code pass the public test only? (for best-of-n)"""
    try:
        patch_file = grade_dir / f"bon-{spec['task_id']}.diff"
        patch_file.write_text(make_patch(spec["module"], spec["buggy"], code),
                              encoding="utf-8")
        task = TaskStore(task_dir).load(spec["task_id"])
        result, _ = grade(task, patch_file, grade_dir / "ws")
        details = _.get("layers", {}).get("functional", {})
        return details.get("public_pass") is True
    except Exception:
        return False


def generate_best_of_n(model, tokenizer, spec, args, device, task_dir, grade_dir):
    """Best-of-N sampling: try N samples, return first that passes public test,
    fallback to greedy if none pass. Ref: PerfCodeGen / Self-Execution Simulation."""
    import torch
    from _shared_extract import extract_code as _extract
    messages = build_messages(spec)

    # Phase 1: greedy baseline (always generated as fallback)
    inputs, attention = tokenize_prompt(tokenizer, messages, device)
    set_seed(args.seed)
    with torch.inference_mode():
        output = model.generate(inputs, attention_mask=attention,
                                max_new_tokens=args.max_new_tokens,
                                do_sample=False,
                                pad_token_id=tokenizer.pad_token_id)
    greedy_raw = tokenizer.decode(output[0][inputs.shape[1]:],
                                  skip_special_tokens=True)
    greedy_code = _extract(greedy_raw)

    # Phase 2: if greedy passes public test, return it (saves sampling time)
    greedy_dir = grade_dir / f"greedy-{spec['task_id']}"
    greedy_dir.mkdir(exist_ok=True)
    if passes_public_test(spec, greedy_code, task_dir, greedy_dir):
        return greedy_raw, "greedy_hit", 1

    # Phase 3: sample up to N times, return first public-test pass
    for trial in range(args.best_of_n):
        set_seed(args.seed + 10000 + trial)
        inputs, attention = tokenize_prompt(tokenizer, messages, device)
        with torch.inference_mode():
            output = model.generate(inputs, attention_mask=attention,
                                    max_new_tokens=args.max_new_tokens,
                                    do_sample=True,
                                    temperature=args.sample_temperature,
                                    top_p=1.0,
                                    pad_token_id=tokenizer.pad_token_id)
        raw = tokenizer.decode(output[0][inputs.shape[1]:],
                               skip_special_tokens=True)
        code = _extract(raw)
        trial_dir = grade_dir / f"bon-{trial}-{spec['task_id']}"
        trial_dir.mkdir(exist_ok=True)
        if passes_public_test(spec, code, task_dir, trial_dir):
            return raw, f"sample_{trial + 1}_hit", trial + 2  # +1 greedy +1 indexing

    # Fallback: no sample passed public test, use greedy
    return greedy_raw, "greedy_fallback", args.best_of_n + 1


def error_record(spec, repeat_index, seed, error):
    return dict(task_id=spec["task_id"], source_group=spec["source_group"],
                split=spec["split"], difficulty=spec["difficulty"], repeat_index=repeat_index,
                seed=seed, prompt_hash=prompt_fingerprint(spec), verdict="error",
                strict_pass=False, failure_type="infrastructure_error", raw_response="",
                generated_code="", grader_details={}, error=f"{type(error).__name__}: {error}")


def tokenize_prompt(tokenizer, messages, device):
    """Normalize legacy tensor and current BatchEncoding chat-template outputs."""
    import torch
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    if isinstance(encoded, Mapping):
        inputs = encoded["input_ids"].to(device)
        attention = encoded.get("attention_mask")
        attention = attention.to(device) if attention is not None else torch.ones_like(inputs)
    else:
        inputs = encoded.to(device)
        attention = torch.ones_like(inputs)
    return inputs, attention


def evaluate_gpu(args, specs, task_dir, grade_dir):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

    if args.adapter and not Path(args.adapter).is_dir():
        raise FileNotFoundError(
            f"--adapter 目录不存在：{args.adapter}（相对路径以当前工作目录为准；"
            "请先确认训练已产出该 checkpoint）")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This bf16/NF4 protocol requires a GPU with bf16 support")
    training_manifest = None
    if args.adapter:
        manifest_path = Path(args.adapter).parent / "training-manifest.json"
        if manifest_path.exists():
            training_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if training_manifest.get("model") != args.model:
                raise ValueError("Adapter was trained against a different base model")
    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter or args.model, revision=None if args.adapter else args.revision,
        trust_remote_code=False)
    if training_manifest:
        template_hash = hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest()
        if template_hash != training_manifest.get("chat_template_sha256"):
            raise ValueError("Adapter tokenizer chat template differs from training")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = dict(revision=args.revision, device_map={"": 0},
                        torch_dtype=torch.bfloat16, trust_remote_code=False,
                        attn_implementation="sdpa")
    if args.quantization == "nf4":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    revision = getattr(model.config, "_commit_hash", None) or args.revision
    if training_manifest and training_manifest.get("resolved_revision") != revision:
        raise ValueError("Evaluation base revision differs from adapter training revision")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    model.config.use_cache = True
    results = []
    total = args.repeats * len(specs)
    done = 0
    run_started = time.perf_counter()
    for repeat_index in range(args.repeats):
        for spec in specs:
            seed = args.seed + repeat_index
            set_seed(seed)
            try:
                started = time.perf_counter()
                if getattr(args, "strategy", "greedy") == "best-of-n":
                    raw, bon_source, bon_trials = generate_best_of_n(
                        model, tokenizer, spec, args, model.device,
                        task_dir, grade_dir)
                    record = evaluate_candidate(spec, raw, task_dir, grade_dir, dict(
                        repeat_index=repeat_index, seed=seed,
                        strategy="best-of-n", bon_source=bon_source,
                        bon_trials=bon_trials,
                        generation_time_s=round(time.perf_counter() - started, 4))
                    )
                else:
                    inputs, attention = tokenize_prompt(
                        tokenizer, build_messages(spec), model.device)
                    generation = dict(max_new_tokens=args.max_new_tokens,
                                      do_sample=args.temperature > 0,
                                      pad_token_id=tokenizer.pad_token_id)
                    if args.temperature > 0:
                        generation.update(temperature=args.temperature, top_p=1.0, top_k=0)
                    with torch.inference_mode():
                        output = model.generate(inputs, attention_mask=attention,
                                                **generation)
                    torch.cuda.synchronize()
                    generated = output[0][inputs.shape[1]:]
                    raw = tokenizer.decode(generated, skip_special_tokens=True)
                    record = evaluate_candidate(spec, raw, task_dir, grade_dir, dict(
                        repeat_index=repeat_index, seed=seed,
                        generation_time_s=round(time.perf_counter() - started, 4),
                        input_tokens=inputs.shape[1], output_tokens=len(generated),
                        hit_token_limit=len(generated) >= args.max_new_tokens))
            except Exception as error:
                record = error_record(spec, repeat_index, seed, error)
            results.append(record)
            done += 1
            elapsed = time.perf_counter() - run_started
            eta = elapsed / done * (total - done)
            filled = 24 * done // total
            bar = '#' * filled + '-' * (24 - filled)
            print(f"[{bar}] {done}/{total} ({done * 100 // total}%) "
                  f"(r{repeat_index + 1}) {spec['task_id']}: {record['verdict']} · "
                  f"已用 {elapsed / 60:.1f}m · 剩余≈{eta / 60:.1f}m", flush=True)
    return results, revision, args.model, {
        "chat_template_sha256": hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
        "tokenizer_vocabulary_sha256": content_hash(tokenizer.get_vocab()),
        "attention_implementation": "sdpa",
    }


def evaluate_api(args, specs, task_dir, grade_dir):
    from openai import OpenAI

    if args.api == "primary":
        base_url = os.environ.get("TEACHER_BASE_URL", "https://teacher-api.example/api/paas/v4")
        api_key = os.environ.get("TEACHER_API_KEY", "")
        model_name = os.environ.get("TEACHER_MODEL", "teacher-api-model")
    else:
        base_url = "https://secondary-api.example/v1"
        api_key = os.environ.get("SECONDARY_API_KEY", "")
        model_name = os.environ.get("SECONDARY_MODEL", "secondary-model")
    if not api_key:
        raise ValueError("The selected API key environment variable is missing")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120, max_retries=0)
    results = []
    total = args.repeats * len(specs)
    done = 0
    run_started = time.perf_counter()
    for repeat_index in range(args.repeats):
        for spec in specs:
            seed = args.seed + repeat_index
            try:
                started = time.perf_counter()
                response = client.chat.completions.create(
                    model=model_name, messages=build_messages(spec),
                    max_tokens=args.max_new_tokens, temperature=args.temperature, seed=seed)
                choice = response.choices[0]
                raw = choice.message.content or ""
                usage = response.usage
                record = evaluate_candidate(spec, raw, task_dir, grade_dir, dict(
                    repeat_index=repeat_index, seed=seed,
                    generation_time_s=round(time.perf_counter() - started, 4),
                    input_tokens=usage.prompt_tokens if usage else None,
                    output_tokens=usage.completion_tokens if usage else None,
                    finish_reason=choice.finish_reason,
                    hit_token_limit=choice.finish_reason == "length"))
            except Exception as error:
                record = error_record(spec, repeat_index, seed, error)
            results.append(record)
            done += 1
            elapsed = time.perf_counter() - run_started
            eta = elapsed / done * (total - done)
            filled = 24 * done // total
            bar = '#' * filled + '-' * (24 - filled)
            print(f"[{bar}] {done}/{total} ({done * 100 // total}%) "
                  f"(r{repeat_index + 1}) {spec['task_id']}: {record['verdict']} · "
                  f"已用 {elapsed / 60:.1f}m · 剩余≈{eta / 60:.1f}m", flush=True)
    client.close()
    return results, args.revision, model_name, {"attention_implementation": "provider_managed"}


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model")
    mode.add_argument("--api", choices=["primary", "secondary"])
    parser.add_argument("--adapter")
    parser.add_argument("--revision", help="Pinned Hugging Face model commit")
    parser.add_argument("--tasks-file", help="External JSONL task specs")
    parser.add_argument("--split", choices=["smoke", "train", "dev", "test", "ood"], default="smoke")
    parser.add_argument("--training-data", help="Training JSONL or training-manifest.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--quantization", choices=["nf4", "bf16"], default="nf4")
    parser.add_argument("--strategy", choices=["greedy", "best-of-n"], default="greedy",
                        help="greedy: single decode; best-of-n: sample N times, "
                             "return the first that passes public tests")
    parser.add_argument("--best-of-n", type=int, default=4,
                        help="Number of samples for --strategy best-of-n")
    parser.add_argument("--sample-temperature", type=float, default=0.7,
                        help="Temperature for best-of-n sampling phase")
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    if args.repeats < 1 or args.max_new_tokens < 1 or args.temperature < 0:
        parser.error("repeats/max-new-tokens must be positive; temperature cannot be negative")
    if args.api and args.adapter:
        parser.error("API evaluation cannot load a local adapter")
    specs = load_specs(args.tasks_file, split=args.split)
    if not specs:
        parser.error(f"No tasks in split {args.split!r}")
    training_data = args.training_data
    if not training_data and args.adapter:
        candidate = Path(args.adapter).parent / "training-manifest.json"
        if candidate.exists():
            training_data = str(candidate)
    overlap = training_overlap(specs, training_data)
    selection_overlap = None
    if args.adapter and args.split in {"test", "ood"}:
        candidate = Path(args.adapter).parent / "training-manifest.json"
        if candidate.exists():
            selection_overlap = training_overlap(specs, candidate, include_selection=True)
            if selection_overlap["status"] == "contaminated":
                parser.error("Test tasks overlap adapter training or checkpoint-selection data")
    if args.split in {"dev", "test", "ood"}:
        if overlap["status"] == "contaminated":
            parser.error(f"Evaluation overlaps training identities: {json.dumps(overlap, ensure_ascii=False)}")
        if args.adapter and overlap["status"] == "not_checked":
            parser.error("Held-out adapter evaluation requires --training-data or training-manifest.json")
    generation = dict(repeats=args.repeats, seed=args.seed, temperature=args.temperature,
                      max_new_tokens=args.max_new_tokens,
                      quantization="api" if args.api else args.quantization,
                      decoding="greedy" if args.temperature == 0 else "sampling",
                      backend="api" if args.api else "transformers")
    benchmark_hash = protocol_hash(specs)
    prompts = {spec["task_id"]: prompt_fingerprint(spec) for spec in specs}
    protocol = dict(schema_version=SCHEMA_VERSION, benchmark_hash=benchmark_hash,
                    prompt_hashes=prompts, generation=generation, split=args.split)
    with tempfile.TemporaryDirectory(prefix="codeagent-eval-") as directory:
        work = Path(directory)
        task_dir, grade_dir = work / "tasks", work / "grade"
        grade_dir.mkdir()
        build_all(task_dir, specs=specs)
        runner = evaluate_api if args.api else evaluate_gpu
        results, resolved_revision, model_name, tokenizer_protocol = runner(args, specs, task_dir, grade_dir)
    protocol["tokenization"] = tokenizer_protocol
    packages = {}
    for package in ("torch", "transformers", "peft", "bitsandbytes", "openai"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    metrics = summarize_results(results)
    summary = dict(schema_version=SCHEMA_VERSION, model=model_name,
                   requested_revision=args.revision, resolved_revision=resolved_revision,
                   adapter=args.adapter, tasks_file=args.tasks_file, split=args.split,
                   task_ids=sorted(spec["task_id"] for spec in specs),
                   task_hashes={spec["task_id"]: spec_fingerprint(spec) for spec in specs},
                   tasks_hash=benchmark_hash, protocol=protocol,
                   protocol_hash=content_hash(protocol), contamination=overlap,
                   selection_contamination=selection_overlap,
                   generation=generation, repeats=args.repeats, metrics=metrics,
                   environment={"python": sys.version, "packages": packages},
                   n_tasks=metrics["n_tasks"], success_rate=metrics["pass_at_1"],
                   results=results, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   interpretation="diagnostic_only" if args.split in {"smoke", "train"}
                   else "held_out")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"pass@1={metrics['pass_at_1']:.2%}; any-pass={metrics['any_pass_rate']:.2%}; "
          f"infrastructure errors={metrics['infrastructure_errors']}; output={output}")
    if not metrics["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
