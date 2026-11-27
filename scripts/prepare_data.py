"""Freeze independent task splits, then replay and deduplicate SFT candidates."""
import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agentlab.benchmark.builder import build_all
from agentlab.benchmark import grader
from agentlab.benchmark.protocol import load_specs, spec_fingerprint
from agentlab.benchmark.task_store import TaskStore
from agentlab.training.preparation import content_hash, freeze_splits, prepare_records
from _shared_jsonl import jsonl_line
from collect_data import grade_candidate


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path, records):
    """一条记录一行，行尾只认 "\n"：记录里的 U+2028、U+2029、U+0085 写成转义（_shared_jsonl.jsonl_line），
    和 sample_rft.py、rft_control.py 写出来的一样——按行切的脚本（训练入口、编排脚本里的片段）切不开它。"""
    path.write_text("".join(jsonl_line(record) + "\n" for record in records), encoding="utf-8")


def load_task_ids(path) -> set[str]:
    """读 --drop-tasks-file：每行一个题号，或者一个题号的 JSON 数组；空文件、空行都不要紧。"""
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith(("[", "{")):
        # 题号里不会出现这两个字符，所以开头是它们就按 JSON 读；读不出题号就报错，
        # 免得把别的文件（例如一份 JSONL）当成题号，结果一条也没去掉却没人发现
        try:
            items = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}: --drop-tasks-file 里要么每行一个题号，要么是一个题号的 JSON 数组") from error
        if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
            raise ValueError(f"{path}: --drop-tasks-file 的 JSON 数组里只能放题号")
        return set(items)
    return {line.strip() for line in text.splitlines() if line.strip()}


def drop_records(records: list[dict], path) -> tuple[list[dict], set[str]]:
    """把 --drop-tasks-file 里这些题的记录挡在外面。返回剩下的记录，以及实际有记录被去掉的题号。"""
    dropped = load_task_ids(path)
    removed = [record for record in records if record.get("task_id") in dropped]
    return ([record for record in records if record.get("task_id") not in dropped],
            {record["task_id"] for record in removed})


def audit_tasks(specs: list[dict], timeout: int) -> dict:
    """Require valid references and hidden tests that reproduce each bug."""
    if timeout < 1:
        raise ValueError("audit timeout must be positive")
    rows = []
    previous_timeout = grader.PYTEST_TIMEOUT
    try:
        grader.PYTEST_TIMEOUT = timeout
        with tempfile.TemporaryDirectory(prefix="codeagent-audit-") as temporary:
            root = Path(temporary)
            build_all(root / "tasks", specs)
            store = TaskStore(root / "tasks")
            empty_patch = root / "no-change.diff"
            empty_patch.write_text("", encoding="utf-8")
            for spec in specs:
                start = time.monotonic()
                task = store.load(spec["task_id"])
                reference, reference_details = grader.grade(
                    task, task.reference_patch, root / "reference-grades")
                if not reference.strict_pass:
                    raise ValueError(f"{spec['task_id']}: reference patch failed "
                                     f"({reference.verdict}): {reference_details}")
                try:
                    buggy, buggy_details = grader.grade(task, empty_patch, root / "buggy-grades")
                except subprocess.TimeoutExpired as error:
                    command = error.cmd if isinstance(error.cmd, list) else [str(error.cmd)]
                    if "tests_private" not in command:
                        raise ValueError(
                            f"{spec['task_id']}: buggy public test timed out; hidden "
                            "bug reproduction could not be verified") from error
                    buggy_verdict = "private_timeout"
                    private_pass = False
                else:
                    private_pass = buggy_details.get("layers", {}).get("functional", {}).get("private_pass")
                    buggy_verdict = buggy.verdict
                    if private_pass is not False:
                        raise ValueError(f"{spec['task_id']}: buggy code does not fail "
                                         "hidden tests; strengthen the trusted test suite")
                rows.append({
                    "task_id": spec["task_id"], "task_sha256": spec_fingerprint(spec),
                    "reference_verdict": reference.verdict,
                    "buggy_verdict": buggy_verdict, "buggy_private_pass": private_pass,
                    "elapsed_s": round(time.monotonic() - start, 3),
                })
                print(f"audit {spec['task_id']}: reference=pass, buggy={buggy_verdict}", flush=True)
    finally:
        grader.PYTEST_TIMEOUT = previous_timeout
    return {"schema_version": 1, "tasks_sha256": content_hash(specs),
            "pytest_timeout_seconds": timeout, "tasks": rows,
            "reference_passes": len(rows), "bugs_reproduced": len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-file", required=True)
    parser.add_argument("--split-tasks", action="store_true")
    parser.add_argument("--input", nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--stratify", action="store_true",
                        help="Give source groups that declare a 'stratum' field (the hard batch) their "
                             "proportional share of dev and test instead of leaving it to one shuffle")
    parser.add_argument("--max-per-task", type=int, default=1,
                        help="Keep at most this many verified solutions per independent task")
    parser.add_argument("--revalidate-stale", action="store_true",
                        help="Replay candidates after private-test/reference updates; public prompts "
                             "and frozen split metadata must still match exactly")
    parser.add_argument("--audit-timeout", type=int, default=10,
                        help="Reference/bug reproduction pytest timeout per test suite")
    parser.add_argument("--drop-tasks-file", default=None,
                        help="把这些题的记录挡在训练数据之外：每行一个题号，或一个题号的 JSON 数组"
                             "（例如 rft_control.py 写出的 unmatched-tasks.txt）")
    args = parser.parse_args()
    if args.split_tasks == bool(args.input):
        parser.error("choose --split-tasks or --input, but not both")
    if args.split_tasks and args.revalidate_stale:
        parser.error("--revalidate-stale applies only to --input candidate preparation")
    if args.split_tasks and args.drop_tasks_file:
        parser.error("--drop-tasks-file 只用于 --input 的候选准备：划分出来的题集里没有记录可去")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        parser.error("output directory must be empty to protect frozen experiments")
    specs = load_specs(args.tasks_file)
    if args.split_tasks:
        frozen, manifest = freeze_splits(specs, args.seed, args.dev_ratio, args.test_ratio,
                                         stratify=args.stratify)
        audit = audit_tasks(frozen, args.audit_timeout)
        output.mkdir(parents=True, exist_ok=True)
        write_jsonl(output / "tasks.jsonl", frozen)
        write_json(output / "split-manifest.json", manifest)
        write_json(output / "audit-manifest.json", audit)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    records = []
    for name in args.input:
        with open(name, encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    dropped_tasks = set()
    if args.drop_tasks_file:
        # 在重放验证之前过滤：被去掉的题不再占判分的时间
        kept, dropped_tasks = drop_records(records, args.drop_tasks_file)
        print(f"按 --drop-tasks-file 去掉 {len(records) - len(kept)} 条记录（{len(dropped_tasks)} 道题）")
        records = kept
    with tempfile.TemporaryDirectory(prefix="codeagent-verify-") as temporary:
        root = Path(temporary)
        selected = [spec for spec in specs if spec["split"] in {"train", "dev"}]
        if args.revalidate_stale:
            audit = audit_tasks(selected, args.audit_timeout)
        build_all(root / "tasks", selected)

        def replay(spec, code):
            print(f"重放验证候选: {spec['task_id']}", flush=True)
            return grade_candidate(spec, code, root / "tasks", root / "grade")

        outputs, manifest = prepare_records(records, specs, args.max_per_task,
                                            replay=replay, revalidate_stale=args.revalidate_stale)
    manifest["replayed_samples"] = sum(len(samples) for samples in outputs.values())
    if args.drop_tasks_file:
        # 这些题在 missing_task_ids 里也看得出来；单独列一遍，说明它们是被主动去掉的，不是没采到
        manifest["dropped_task_ids"] = sorted(dropped_tasks)
    output.mkdir(parents=True, exist_ok=True)
    for split, samples in outputs.items():
        write_jsonl(output / f"{split}.jsonl", samples)
    write_json(output / "dataset-manifest.json", manifest)
    write_jsonl(output / "rejected-samples.jsonl", manifest["replay_rejections"])
    if args.revalidate_stale:
        write_json(output / "audit-manifest.json", audit)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
