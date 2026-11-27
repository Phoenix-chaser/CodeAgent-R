"""Export and self-check v2 scale-up task packs (batch pools v2 / v2b)."""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import importlib
import json
import os
import tempfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agentlab.benchmark.builder import SPECS, build_all
from agentlab.benchmark.pilot_tasks import PILOT_SPECS
from agentlab.benchmark.protocol import protocol_hash, validate_specs
from agentlab.benchmark.task_store import TaskStore

POOLS = {
    "v2": ("agentlab.benchmark.pilot_tasks_v2", "V2_SPECS", "batch 1"),
    "v2b": ("agentlab.benchmark.pilot_tasks_v2b", "V2B_SPECS", "batch 2"),
    "v2c": ("agentlab.benchmark.pilot_tasks_v2c", "V2C_SPECS", "batch 3"),
    "v2d": ("agentlab.benchmark.pilot_tasks_v2d", "V2D_SPECS", "batch 4"),
    # 多缺陷的难题（跨文件、状态一致性、异常安全、并发）。不进默认题池，
    # 要用 EXTRA_POOLS=h3 显式启用，已冻结的实验目录因此不受影响
    "h3": ("agentlab.benchmark.hard_tasks_v3", "HARD3_SPECS", "hard batch"),
    # 难题的单缺陷变体：题面、参考解、测试都不变，程序里只留一处缺陷。
    # 与母题同来源组，所以永远和母题落在同一个划分
    "h3s": ("agentlab.benchmark.hard_tasks_v3_steps", "STEP_SPECS", "hard steps"),
}
# 默认基准是 pilot 加这四个池，共 104 题
DEFAULT_POOLS = ("v2", "v2b", "v2c", "v2d")


def extra_pools() -> list[str]:
    """EXTRA_POOLS 里列出的可选池（空格分隔），保持给定顺序。"""
    names = [name for name in os.environ.get("EXTRA_POOLS", "").split() if name]
    invalid = [name for name in names if name not in POOLS or name in DEFAULT_POOLS]
    if invalid or len(set(names)) != len(names):
        raise ValueError(f"EXTRA_POOLS must list opt-in pools once each: {names}")
    return names


def active_pools() -> list[str]:
    return [*DEFAULT_POOLS, *extra_pools()]


def load_pool(name: str) -> list[dict]:
    module_name, attr, _ = POOLS[name]
    return list(getattr(importlib.import_module(module_name), attr))


def prior_pools(exclude: str) -> list[tuple[str, list[dict]]]:
    others = [("builtin", SPECS), ("pilot", PILOT_SPECS)]
    for name in POOLS:
        if name != exclude:
            others.append((name, load_pool(name)))
    return others


def make_patch(module_path: str, buggy: str, fixed: str) -> str:
    return "".join(difflib.unified_diff(
        buggy.splitlines(keepends=True), fixed.splitlines(keepends=True),
        fromfile=f"a/{module_path}", tofile=f"b/{module_path}"))


def _same_family(spec: dict, other_specs: list[dict]) -> bool:
    """spec 与另一个题池里的某道题同来源组、同文件，并且两者之间有"变体"声明。"""
    for other in other_specs:
        if other["source_group"] != spec["source_group"] or other["module"] != spec["module"]:
            continue
        if other["task_id"] == spec.get("variant_of") or other.get("variant_of") == spec["task_id"]:
            return True
    return False


def check_pool_shape(specs: list[dict],
                     others: list[tuple[str, list[dict]]]) -> None:
    """Validate the schema, reject collisions with other pools and duplicates inside.

    A single-defect variant (it declares ``variant_of``) shares the source group and
    the file of its parent on purpose.  Everything else has to stay disjoint from
    every other pool, and no two tasks may carry the same buggy program."""
    validate_specs(specs)
    for other_name, other_specs in others:
        ids = {spec["task_id"] for spec in other_specs}
        groups = {spec["source_group"] for spec in other_specs}
        modules = {spec["module"] for spec in other_specs}
        asts = {ast.dump(ast.parse(spec["buggy"])) for spec in other_specs}
        for spec in specs:
            program = ast.dump(ast.parse(spec["buggy"]))
            shared_family = _same_family(spec, other_specs)
            if (spec["task_id"] in ids or program in asts
                    or (not shared_family
                        and (spec["source_group"] in groups or spec["module"] in modules))):
                raise ValueError(
                    f"{spec['task_id']} collides with {other_name} pool")
    seen_ast, seen_groups = set(), {}
    for spec in specs:
        candidate_ast = ast.dump(ast.parse(spec["buggy"]))
        ast.parse(spec["fixed"])
        if candidate_ast in seen_ast:
            raise ValueError(f"Duplicate buggy program: {spec['task_id']}")
        # 同一来源组的变体共用一个文件名；不同来源组不得共用
        if seen_groups.setdefault(spec["module"], spec["source_group"]) != spec["source_group"]:
            raise ValueError(f"Duplicate module: {spec['module']}")
        seen_ast.add(candidate_ast)


def grade_program(task, spec: dict, source: str, work: Path) -> dict:
    """Grade one candidate program of a task (the main file replaced by ``source``)."""
    grader_module = importlib.import_module("agentlab.benchmark.grader")
    work.mkdir(parents=True, exist_ok=True)
    patch = work / "candidate.diff"
    patch.write_text(make_patch(spec["module"], spec["buggy"], source), encoding="utf-8")
    result, details = grader_module.grade(task, patch, work / "ws")
    functional = details.get("layers", {}).get("functional", {})
    return dict(verdict=result.verdict, strict_pass=result.strict_pass,
                public_pass=functional.get("public_pass"),
                private_pass=functional.get("private_pass"))


def grade_self_check(specs: list[dict], timeout: int = 5) -> list[dict]:
    """Reference must pass every layer; the unchanged program must fail the private tests."""
    grader_module = importlib.import_module("agentlab.benchmark.grader")
    previous_timeout = grader_module.PYTEST_TIMEOUT
    grader_module.PYTEST_TIMEOUT = timeout
    results = []
    try:
        with tempfile.TemporaryDirectory(prefix="codeagent-v2-audit-") as directory:
            root = Path(directory)
            task_dir = root / "tasks"
            build_all(task_dir, specs=specs)
            store = TaskStore(task_dir)
            for spec in specs:
                task = store.load(spec["task_id"])
                reference = grade_program(task, spec, spec["fixed"], root / "reference")
                unchanged = grade_program(task, spec, spec["buggy"], root / "unchanged")
                record = dict(task_id=spec["task_id"],
                              source_group=spec["source_group"],
                              reference_verdict=reference["verdict"],
                              original_verdict=unchanged["verdict"],
                              original_public_pass=unchanged["public_pass"],
                              original_private_pass=unchanged["private_pass"])
                results.append(record)
                print(f"{spec['task_id']}: reference={reference['verdict']}, "
                      f"original={unchanged['verdict']} "
                      f"(public={unchanged['public_pass']}, "
                      f"private={unchanged['private_pass']})", flush=True)
                if (not reference["strict_pass"] or unchanged["strict_pass"]
                        or unchanged["private_pass"] is not False):
                    raise ValueError(f"self-check failed: "
                                     f"{json.dumps(record, ensure_ascii=False)}")
    finally:
        grader_module.PYTEST_TIMEOUT = previous_timeout
    return results


def validate_pool(specs: list[dict], exclude: str) -> list[dict]:
    check_pool_shape(specs, prior_pools(exclude))
    return grade_self_check(specs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", choices=sorted(POOLS), default="v2")
    parser.add_argument("--list-pools", action="store_true",
                        help="Print the pools of the current benchmark and exit")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.list_pools:
        print(" ".join(active_pools()))
        return
    specs = load_pool(args.pool)
    batch = POOLS[args.pool][2]
    output = Path(args.output or f"data/task-pool-{args.pool}.jsonl")
    audit_path = output.with_suffix(".audit.json")
    if output.exists() or audit_path.exists():
        parser.error("Refusing to overwrite an existing task pool or audit report")
    print(f"{len(specs)} self-authored tasks / "
          f"{len({spec['source_group'] for spec in specs})} source groups: {batch}.")
    print("Freeze group splits before collection; still not generalization evidence alone.")
    try:
        checks = validate_pool(specs, exclude=args.pool)
    except (ValueError, SyntaxError) as error:
        parser.error(str(error))
    encoded = "".join(json.dumps(spec, ensure_ascii=False, sort_keys=True) + "\n"
                      for spec in specs)
    audit = dict(
        purpose=f"scaleup_{batch}_pipeline_tasks",
        n_tasks=len(specs),
        n_source_groups=len({spec["source_group"] for spec in specs}),
        split="train_unfrozen", pytest_timeout_seconds=5,
        reference_passed=sum(c["reference_verdict"] == "strict_pass" for c in checks),
        original_failed=sum(c["original_verdict"] != "strict_pass" for c in checks),
        original_private_failed=sum(c["original_private_pass"] is False for c in checks),
        original_public_passed=sum(c["original_public_pass"] is True for c in checks),
        sha256=hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        protocol_hash=protocol_hash(specs), checks=checks,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
    with audit_path.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    print(f"Exported {output}; audit={audit_path}")


if __name__ == "__main__":
    main()
