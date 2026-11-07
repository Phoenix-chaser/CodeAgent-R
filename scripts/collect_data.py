"""Collect bounded teacher candidates against frozen train/dev task splits."""
import argparse
import difflib
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agentlab.benchmark.builder import build_all
from agentlab.benchmark.grader import grade
from agentlab.benchmark.protocol import build_messages, load_specs, protocol_hash, spec_fingerprint
from agentlab.benchmark.task_store import TaskStore
from agentlab.training.preparation import reject_legacy_tasks
from _shared_extract import extract_code


def collection_status(output, specs, candidates):
    expected = {(spec["task_id"], index) for spec in specs for index in range(candidates)}
    latest = {}
    path = Path(output)
    hashes = {spec["task_id"]: spec_fingerprint(spec) for spec in specs}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = (record["task_id"], record["candidate_idx"])
            if key not in expected:
                continue
            if record.get("task_hash") != hashes[key[0]]:
                raise ValueError("Existing collection task hash differs from the frozen tasks")
            latest[key] = record
    retryable = {key for key, record in latest.items()
                 if record.get("provenance", {}).get("verdict") in
                 {"collection_error", "incomplete_response"}}
    return latest, expected - latest.keys(), retryable


def make_patch(module_path: str, buggy: str, fixed: str) -> str:
    return "".join(difflib.unified_diff(
        buggy.splitlines(keepends=True), fixed.splitlines(keepends=True),
        fromfile=f"a/{module_path}", tofile=f"b/{module_path}"))


def code_hash(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


def grade_candidate(spec: dict, fixed_code: str, task_dir: Path, grade_dir: Path) -> tuple:
    grade_dir.mkdir(parents=True, exist_ok=True)
    patch = make_patch(spec["module"], spec["buggy"], fixed_code)
    patch_file = grade_dir / f"cand-{spec['task_id']}.diff"
    patch_file.write_text(patch, encoding="utf-8")
    task = TaskStore(task_dir).load(spec["task_id"])
    result, details = grade(task, patch_file, grade_dir / "ws")
    return result.verdict, result.strict_pass, details


def call_teacher(spec, client, model, max_tokens, temperature):
    response = client.chat.completions.create(
        model=model, messages=build_messages(spec),
        max_tokens=max_tokens, temperature=temperature)
    choice = response.choices[0] if response.choices else None
    content = choice.message.content if choice else None
    usage = {
        "input_tokens": response.usage.prompt_tokens if response.usage else None,
        "output_tokens": response.usage.completion_tokens if response.usage else None,
    }
    return {
        "raw_response": content or "",
        "finish_reason": choice.finish_reason if choice else None,
        "response_id": response.id,
        "usage": usage,
    }


def collect_one(spec, candidate_idx, client, args, task_dir, grade_dir):
    temperature = args.temperature if candidate_idx % 2 == 0 else args.exploration_temperature
    response = call_teacher(spec, client, args.model, args.max_tokens, temperature)
    base = {
        "record_type": "candidate", "sample_id": f"sft-{spec['task_id']}-{candidate_idx:03d}",
        "task_id": spec["task_id"], "task_hash": spec_fingerprint(spec),
        "candidate_idx": candidate_idx, "source_group": spec["source_group"],
        "split": spec["split"], "mode": "S1_final_patch", "temperature": temperature,
        "messages": build_messages(spec), **response,
    }
    provenance = {
        "generator": args.model, "grader_version": "three_layer_v1",
        "strict_pass": False, "usage": response["usage"],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if response["finish_reason"] != "stop" or not response["raw_response"].strip():
        return {**base, "record_type": "attempt",
                "provenance": {**provenance, "verdict": "incomplete_response"}}
    fixed = extract_code(response["raw_response"])
    verdict, strict_pass, details = grade_candidate(spec, fixed, task_dir, grade_dir)
    return {
        **base, "code_hash": code_hash(fixed),
        "target": {"role": "assistant", "content": fixed}, "grader_details": details,
        "provenance": {**provenance, "verdict": verdict, "strict_pass": strict_pass},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-file", required=True, help="Frozen external task JSONL from prepare_data.py")
    parser.add_argument("--split", choices=["train", "dev"], default="train")
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=os.environ.get("TEACHER_MODEL", "teacher-api-model"))
    parser.add_argument("--base-url", default=os.environ.get("TEACHER_BASE_URL", "https://teacher-api.example/api/paas/v4"))
    parser.add_argument("--api-key-env", default="TEACHER_API_KEY")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--exploration-temperature", type=float, default=0.7)
    parser.add_argument("--request-delay", type=float, default=1.0)
    parser.add_argument("--include-failures", action="store_true",
                        help="Retain full failed candidates for separate preference experiments")
    parser.add_argument("--status-only", action="store_true",
                        help="No requests: exit 0 complete, 3 incomplete/retryable, 2 invalid")
    parser.add_argument("--retry-errors", action="store_true",
                        help="Explicitly retry previously incomplete/API-error slots once")
    args = parser.parse_args()
    if args.candidates < 1 or args.max_tokens < 1 or args.request_delay < 0:
        parser.error("candidates/max-tokens must be positive; request-delay must be nonnegative")
    specs = load_specs(args.tasks_file, args.split)
    reject_legacy_tasks(specs)
    protocol = {
        "schema_version": 2, "tasks_hash": protocol_hash(specs), "split": args.split,
        "model": args.model, "base_url": args.base_url,
        "max_tokens": args.max_tokens, "temperature": args.temperature,
        "exploration_temperature": args.exploration_temperature,
        "include_failures": args.include_failures,
    }
    output = Path(args.output)
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists():
        if not manifest_path.exists():
            parser.error("existing output has no protocol manifest; use a new output path")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["protocol"] != protocol:
            parser.error("collection protocol changed; use a new output path")
    try:
        latest, missing, retryable = collection_status(output, specs, args.candidates)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    if args.status_only:
        print(json.dumps({"slots": len(specs) * args.candidates, "recorded": len(latest),
                          "missing": len(missing), "retryable": len(retryable)}))
        raise SystemExit(3 if missing or retryable else 0)
    if retryable and not args.retry_errors:
        parser.error(f"{len(retryable)} previous errors/incomplete responses; use --retry-errors to retry explicitly")
    pending = missing | (retryable if args.retry_errors else set())
    if not pending:
        print("Collection already complete; no API requests.")
        return
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        parser.error(f"missing API key environment variable: {args.api_key_env}")
    existing = set(latest) - pending
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"protocol": protocol}, ensure_ascii=False, indent=2), encoding="utf-8")
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=120, max_retries=0)
    request_limit = len(specs) * args.candidates
    print(f"Tasks={len(specs)}; candidate slots={request_limit}; completed={len(existing)}",
          flush=True)
    print("Every attempt is persisted; resume skips recorded slots.", flush=True)
    started = time.monotonic()
    processed = 0
    try:
        with tempfile.TemporaryDirectory(prefix="codeagent-collect-") as temporary:
            work = Path(temporary)
            build_all(work / "tasks", specs)
            for spec in specs:
                for candidate_idx in range(args.candidates):
                    if (spec["task_id"], candidate_idx) in existing:
                        continue
                    try:
                        with tempfile.TemporaryDirectory(dir=work, prefix="candidate-") as candidate_dir:
                            record = collect_one(spec, candidate_idx, client, args, work / "tasks", Path(candidate_dir))
                    except Exception as error:
                        record = {
                            "record_type": "error", "task_id": spec["task_id"],
                            "candidate_idx": candidate_idx, "task_hash": spec_fingerprint(spec),
                            "error_type": type(error).__name__,
                            "provenance": {"strict_pass": False, "verdict": "collection_error"},
                        }
                    processed += 1
                    verdict = record["provenance"]["verdict"]
                    done_total = len(existing) + processed
                    elapsed = time.monotonic() - started
                    eta = elapsed / processed * (request_limit - done_total)
                    filled = 24 * done_total // request_limit
                    bar = '#' * filled + '-' * (24 - filled)
                    print(f"[{bar}] {done_total}/{request_limit} "
                          f"({done_total * 100 // request_limit}%) "
                          f"{spec['task_id']} [{candidate_idx}] {verdict} · "
                          f"已用 {elapsed / 60:.0f}m · 剩余≈{eta / 60:.0f}m", flush=True)
                    if not args.include_failures and not record["provenance"]["strict_pass"]:
                        record = {key: value for key, value in record.items()
                                  if key not in {"raw_response", "target", "messages", "grader_details"}}
                        record["record_type"] = "attempt"
                    with output.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        handle.flush()
                    time.sleep(args.request_delay)
    finally:
        client.close()
    print(f"Saved {output}; prepare_data.py will replay accepted solutions and enforce task caps.")
    _, remaining, retryable = collection_status(output, specs, args.candidates)
    if remaining or retryable:
        print(f"Collection incomplete: missing={len(remaining)}, retryable={len(retryable)}", file=sys.stderr)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
