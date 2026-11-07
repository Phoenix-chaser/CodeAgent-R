"""Export and self-check 24 small pilot tasks; not generalization evidence."""
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


def validate_pilot_tasks() -> list[dict]:
    validate_specs(PILOT_SPECS)
    original_ast = {ast.dump(ast.parse(spec["buggy"])) for spec in SPECS}
    seen_ast, seen_modules = set(), set()
    for spec in PILOT_SPECS:
        candidate_ast = ast.dump(ast.parse(spec["buggy"]))
        ast.parse(spec["fixed"])
        if candidate_ast in original_ast or candidate_ast in seen_ast:
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
        with tempfile.TemporaryDirectory(prefix="codeagent-pilot-audit-") as directory:
            root = Path(directory)
            task_dir = root / "tasks"
            build_all(task_dir, specs=PILOT_SPECS)
            store = TaskStore(task_dir)
            empty = root / "unchanged.diff"
            empty.write_text("", encoding="utf-8")
            for spec in PILOT_SPECS:
                task = store.load(spec["task_id"])
                reference, reference_details = grader_module.grade(task, task.reference_patch,
                                                                   root / "reference")
                unchanged, unchanged_details = grader_module.grade(task, empty, root / "unchanged")
                record = dict(task_id=spec["task_id"], source_group=spec["source_group"],
                              reference_verdict=reference.verdict,
                              original_verdict=unchanged.verdict,
                              original_private_pass=unchanged_details.get("layers", {}).get(
                                  "functional", {}).get("private_pass"),
                              reference_details=reference_details,
                              original_details=unchanged_details)
                results.append(record)
                print(f"{spec['task_id']}: reference={reference.verdict}, "
                      f"original={unchanged.verdict}", flush=True)
                if (not reference.strict_pass or unchanged.strict_pass
                        or record["original_private_pass"] is not False):
                    raise ValueError(f"Pilot self-check failed: {json.dumps(record, ensure_ascii=False)}")
    finally:
        grader_module.PYTEST_TIMEOUT = previous_timeout
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/pilot-pool.jsonl")
    args = parser.parse_args()
    output = Path(args.output)
    audit_path = output.with_suffix(".audit.json")
    if output.exists() or audit_path.exists():
        parser.error("Refusing to overwrite an existing task pool or audit report")
    print("24 self-authored tasks / 12 source groups: pipeline smoke/pilot only.")
    print("This pack does not establish job-ready repair generalization. Freeze group splits before collection.")
    try:
        checks = validate_pilot_tasks()
    except (ValueError, SyntaxError) as error:
        parser.error(str(error))
    encoded = "".join(json.dumps(spec, ensure_ascii=False, sort_keys=True) + "\n"
                      for spec in PILOT_SPECS)
    audit = dict(
        purpose="pipeline_smoke_pilot_only_not_generalization_evidence",
        n_tasks=len(PILOT_SPECS),
        n_source_groups=len({spec["source_group"] for spec in PILOT_SPECS}),
        split="train_unfrozen", pytest_timeout_seconds=5,
        reference_passed=sum(check["reference_verdict"] == "strict_pass" for check in checks),
        original_failed=sum(check["original_verdict"] != "strict_pass" for check in checks),
        original_private_failed=sum(check["original_private_pass"] is False for check in checks),
        sha256=hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        protocol_hash=protocol_hash(PILOT_SPECS), checks=checks,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
    with audit_path.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2)
    print(f"Exported {output}; audit={audit_path}; references=24/24; originals rejected=24/24.")


if __name__ == "__main__":
    main()
