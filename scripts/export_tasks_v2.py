"""Export and self-check v2 scale-up task packs (batch pools v2 / v2b)."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
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
}


def load_pool(name: str) -> list[dict]:
    module_name, attr, _ = POOLS[name]
    return list(getattr(importlib.import_module(module_name), attr))


def prior_pools(exclude: str) -> list[tuple[str, list[dict]]]:
    others = [("builtin", SPECS), ("pilot", PILOT_SPECS)]
    for name in POOLS:
        if name != exclude:
            others.append((name, load_pool(name)))
    return others


def validate_pool(specs: list[dict], exclude: str) -> list[dict]:
    validate_specs(specs)
    for other_name, other_specs in prior_pools(exclude):
        ids = {spec["task_id"] for spec in other_specs}
        groups = {spec["source_group"] for spec in other_specs}
        modules = {spec["module"] for spec in other_specs}
        asts = {ast.dump(ast.parse(spec["buggy"])) for spec in other_specs}
        for spec in specs:
            if (spec["task_id"] in ids or spec["source_group"] in groups
                    or spec["module"] in modules
                    or ast.dump(ast.parse(spec["buggy"])) in asts):
                raise ValueError(
                    f"{spec['task_id']} collides with {other_name} pool")
    seen_ast, seen_modules = set(), set()
    for spec in specs:
        candidate_ast = ast.dump(ast.parse(spec["buggy"]))
        ast.parse(spec["fixed"])
        if candidate_ast in seen_ast:
            raise ValueError(f"Duplicate buggy program: {spec['task_id']}")
        if spec["module"] in seen_modules:
            raise ValueError(f"Duplicate module: {spec['module']}")
        seen_ast.add(candidate_ast)
        seen_modules.add(spec["module"])
    grader_module = importlib.import_module("agentlab.benchmark.grader")
    previous_timeout = grader_module.PYTEST_TIMEOUT
    grader_module.PYTEST_TIMEOUT = 5
    results = []
    try:
        with tempfile.TemporaryDirectory(prefix="codeagent-v2-audit-") as directory:
            root = Path(directory)
            task_dir = root / "tasks"
            build_all(task_dir, specs=specs)
            store = TaskStore(task_dir)
            empty = root / "unchanged.diff"
            empty.write_text("", encoding="utf-8")
            for spec in specs:
                task = store.load(spec["task_id"])
                reference, _ = grader_module.grade(task, task.reference_patch,
                                                   root / "reference")
                unchanged, unchanged_details = grader_module.grade(
                    task, empty, root / "unchanged")
                functional = unchanged_details.get("layers", {}).get("functional", {})
                record = dict(task_id=spec["task_id"],
                              source_group=spec["source_group"],
                              reference_verdict=reference.verdict,
                              original_verdict=unchanged.verdict,
                              original_public_pass=functional.get("public_pass"),
                              original_private_pass=functional.get("private_pass"))
                results.append(record)
                print(f"{spec['task_id']}: reference={reference.verdict}, "
                      f"original={unchanged.verdict} "
                      f"(public={functional.get('public_pass')}, "
                      f"private={functional.get('private_pass')})", flush=True)
                if (not reference.strict_pass or unchanged.strict_pass
                        or functional.get("private_pass") is not False):
                    raise ValueError(f"self-check failed: "
                                     f"{json.dumps(record, ensure_ascii=False)}")
    finally:
        grader_module.PYTEST_TIMEOUT = previous_timeout
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", choices=sorted(POOLS), default="v2")
    parser.add_argument("--list-pools", action="store_true",
                        help="Print registered pool names and exit")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.list_pools:
        print(" ".join(POOLS))
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
