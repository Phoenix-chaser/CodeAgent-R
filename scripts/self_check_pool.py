#!/usr/bin/env python3
"""Self-check one task list without registering it as a pool.

Same checks as ``validate_pool`` in export_tasks_v2.py: schema, collisions with
the built-in specs and the default pools, duplicates inside the list, reference
strict pass, and the unchanged program rejected by the private tests.  Two
extras help when authoring harder tasks:

* ``MUTANTS`` in the same module, ``{task_id: [(old, new), ...]}``.  Each pair
  replaces ``old`` with ``new`` in the reference (``old`` must occur exactly
  once) and the result is graded.  Every mutant must fail: a mutant that
  passes means the private tests do not cover that requirement on its own.
  Mutants are graded with a limit close to the real one, and a run that times
  out or errors does not count as rejected — a test has to fail the mutant.
* a rough token estimate of prompt and answer.  Training samples longer than
  the configured maximum are skipped, not truncated, so a task whose estimate
  is far above the default 2048 would silently drop out of SFT and RFT.

Usage:
    python3 scripts/self_check_pool.py --module agentlab.benchmark.hard_tasks_v3_exc
    python3 scripts/self_check_pool.py --module ... --only h3_a_001,h3_b_001
    python3 scripts/self_check_pool.py --module ... --no-grade
"""
from __future__ import annotations

import argparse
import importlib
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import export_tasks_v2 as pools  # noqa: E402
from agentlab.benchmark.protocol import build_messages  # noqa: E402
from agentlab.benchmark.task_store import TaskStore  # noqa: E402

# 训练默认长度是 2048 个 token；估算偏保守，超过这个数就该把题压缩
LONG_ESTIMATE = 1900
# 参考解和原程序沿用导出审计的 5 秒上限。变体用接近正式判分（60 秒）的上限：
# 只是让测试变慢的变体，在正式判分里会通过，不能算成"被拦住"
AUDIT_TIMEOUT, MUTANT_TIMEOUT = 5, 30


def rough_tokens(text: str) -> int:
    """Rough estimate for a byte-level tokenizer: about one token per CJK character,
    about one per 3.3 ASCII characters.  Only meant to flag tasks that are too long."""
    return round(sum(1.05 if ord(char) > 0x2E7F else 0.3 for char in text))


def prompt_estimate(spec: dict) -> int:
    return sum(rough_tokens(message["content"]) for message in build_messages(spec))


def grade_all(specs: list[dict], mutants: dict, work: Path) -> list[dict]:
    grader = importlib.import_module("agentlab.benchmark.grader")
    previous_timeout = grader.PYTEST_TIMEOUT
    grader.PYTEST_TIMEOUT = AUDIT_TIMEOUT
    rows = []
    try:
        task_dir = work / "tasks"
        pools.build_all(task_dir, specs=specs)
        store = TaskStore(task_dir)

        def run(task, spec, source, label, timeout):
            grader.PYTEST_TIMEOUT = timeout
            try:
                return pools.grade_program(task, spec, source, work / label)
            except Exception as error:  # timeouts and sandbox failures are findings here
                return dict(verdict=f"error: {type(error).__name__}: {error}"[:160],
                            strict_pass=False, public_pass=None, private_pass=None)

        for spec in specs:
            task = store.load(spec["task_id"])
            row = dict(task_id=spec["task_id"], group=spec["source_group"],
                       reference=run(task, spec, spec["fixed"], "reference", AUDIT_TIMEOUT),
                       original=run(task, spec, spec["buggy"], "unchanged", AUDIT_TIMEOUT),
                       prompt=prompt_estimate(spec), answer=rough_tokens(spec["fixed"]),
                       mutants=[])
            for old, new in mutants.get(spec["task_id"], []):
                count = spec["fixed"].count(old)
                if count != 1:
                    raise SystemExit(f"{spec['task_id']}: mutant anchor occurs {count} times "
                                     f"(must be exactly once): {old[:60]!r}")
                row["mutants"].append(
                    run(task, spec, spec["fixed"].replace(old, new), "mutant", MUTANT_TIMEOUT))
            rows.append(row)
    finally:
        grader.PYTEST_TIMEOUT = previous_timeout
    return rows


def show(result: dict) -> str:
    if result["strict_pass"]:
        return "strict_pass"
    if result["verdict"].startswith("error"):
        return result["verdict"]
    return (f"{result['verdict']}(public={result['public_pass']},"
            f"private={result['private_pass']})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--module", required=True, help="Dotted module that defines the task list")
    parser.add_argument("--attr", default="SPECS", help="Attribute holding the task list")
    parser.add_argument("--only", default="", help="Comma separated task ids to check")
    parser.add_argument("--no-grade", action="store_true",
                        help="Schema and collisions only; do not run any tests")
    args = parser.parse_args()

    module = importlib.import_module(args.module)
    specs = list(getattr(module, args.attr))
    if args.only:
        wanted = {name.strip() for name in args.only.split(",") if name.strip()}
        specs = [spec for spec in specs if spec["task_id"] in wanted]
    others = [("builtin", pools.SPECS), ("pilot", pools.PILOT_SPECS)]
    others += [(name, pools.load_pool(name)) for name in pools.DEFAULT_POOLS]
    pools.check_pool_shape(specs, others)
    print(f"{args.module}.{args.attr}: {len(specs)} tasks, schema and collisions OK")
    if args.no_grade:
        for spec in specs:
            prompt, answer = prompt_estimate(spec), rough_tokens(spec["fixed"])
            flag = "LONG" if prompt + answer > LONG_ESTIMATE else ""
            print(f"{spec['task_id']:<34} tokens~{prompt}+{answer}={prompt + answer}  {flag}")
        return 0

    # mutants live in ``MUTANTS`` next to ``SPECS``, or in ``<PREFIX>_MUTANTS`` next to
    # ``<PREFIX>_SPECS`` (the aggregated pool uses HARD3_SPECS / HARD3_MUTANTS)
    declared = getattr(module, args.attr.removesuffix("_SPECS") + "_MUTANTS", None)
    if declared is None:
        declared = getattr(module, "MUTANTS", {})
    # with --only, mutants of the tasks left out are simply not graded
    selected = {spec["task_id"] for spec in specs}
    mutants = {key: value for key, value in declared.items() if key in selected}
    unknown = [] if args.only else sorted(set(declared) - selected)
    if unknown:
        raise SystemExit(f"MUTANTS names tasks outside the list: {unknown}")
    with tempfile.TemporaryDirectory(prefix="codeagent-selfcheck-") as directory:
        rows = grade_all(specs, mutants, Path(directory))

    problems = 0
    for row in rows:
        reference, original = row["reference"], row["original"]
        escaped = [index + 1 for index, mutant in enumerate(row["mutants"])
                   if mutant["strict_pass"]]
        # a timeout or a sandbox failure is not a rejection by the tests
        not_rejected = [index + 1 for index, mutant in enumerate(row["mutants"])
                        if mutant["verdict"].startswith("error")]
        caught = len(row["mutants"]) - len(escaped) - len(not_rejected)
        total = row["prompt"] + row["answer"]
        flags = []
        if not reference["strict_pass"]:
            flags.append("REFERENCE-FAILS")
        if original["strict_pass"] or original["private_pass"] is not False:
            flags.append("ORIGINAL-NOT-CAUGHT")
        if escaped:
            flags.append(f"MUTANTS-ESCAPE={escaped}")
        if not_rejected:
            flags.append(f"MUTANTS-NOT-REJECTED-BY-A-TEST={not_rejected}")
        if total > LONG_ESTIMATE:
            flags.append("LONG")
        problems += len(flags)
        print(f"{row['task_id']:<34} reference={show(reference):<12} original={show(original):<44} "
              f"mutants={caught}/{len(row['mutants'])} caught  "
              f"tokens~{row['prompt']}+{row['answer']}={total}  {' '.join(flags)}")
    print(f"checked {len(rows)} tasks, {problems} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
