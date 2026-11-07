"""Refresh training-only private contracts while preserving a frozen experiment."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agentlab.benchmark.pilot_tasks import PILOT_SPECS
from agentlab.benchmark.pilot_tasks_v2 import V2_SPECS
from agentlab.benchmark.pilot_tasks_v2b import V2B_SPECS
from agentlab.benchmark.pilot_tasks_v2c import V2C_SPECS
from agentlab.benchmark.protocol import load_specs, spec_fingerprint, validate_specs
from agentlab.training.preparation import content_hash
from prepare_data import audit_tasks, write_json, write_jsonl


def refresh_train_specs(specs: list[dict], references: list[dict]) -> tuple[list[dict], dict]:
    validate_specs(specs)
    validate_specs(references)
    current = {spec["task_id"]: spec for spec in references}
    refreshed = []
    changes = []
    hashes = []
    unmatched = []
    mutable_fields = {"fixed", "private_tests"}
    for original in specs:
        updated = copy.deepcopy(original)
        if original["split"] == "train":
            reference = current.get(original["task_id"])
            if reference is None:
                unmatched.append(original["task_id"])
            else:
                frozen_fields = (set(original) | set(reference)) - mutable_fields - {"split"}
                different = sorted(field for field in frozen_fields
                                   if original.get(field) != reference.get(field))
                if different:
                    raise ValueError(f"{original['task_id']}: frozen public metadata changed: {different}")
                changed = sorted(field for field in mutable_fields
                                 if original[field] != reference[field])
                if changed:
                    updated.update({field: copy.deepcopy(reference[field]) for field in changed})
                    changes.append({"task_id": original["task_id"], "fields": changed})
        refreshed.append(updated)
        hashes.append({"task_id": original["task_id"], "split": original["split"],
                       "old_task_hash": spec_fingerprint(original),
                       "new_task_hash": spec_fingerprint(updated)})
    validate_specs(refreshed)
    manifest = {
        "schema_version": 1, "policy": "training_fixed_and_private_tests_only",
        "source_tasks_sha256": content_hash(specs), "tasks_sha256": content_hash(refreshed),
        "changed_tasks": changes, "task_hashes": hashes,
        "unmatched_train_task_ids": sorted(unmatched),
        "task_counts": dict(Counter(spec["split"] for spec in refreshed)),
        "source_groups": {spec["source_group"]: spec["split"] for spec in refreshed},
        "held_out_specs_unchanged": all(original == updated
                                        for original, updated in zip(specs, refreshed)
                                        if original["split"] != "train"),
        "limitation": "This preserves the original split; previously inspected test results "
                      "do not become a new blind evaluation by refreshing training labels.",
    }
    return refreshed, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-file", required=True, help="Existing frozen task JSONL")
    parser.add_argument("--reference-tasks-file",
                        help="Updated trusted specs; default: current repository pilot/v2 pools")
    parser.add_argument("--output-dir", required=True, help="New directory; never overwritten")
    parser.add_argument("--audit-timeout", type=int, default=10)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        parser.error("output directory must not already exist")
    specs = load_specs(args.tasks_file)
    references = (load_specs(args.reference_tasks_file) if args.reference_tasks_file
                  else PILOT_SPECS + V2_SPECS + V2B_SPECS + V2C_SPECS)
    try:
        refreshed, manifest = refresh_train_specs(specs, references)
        changed_ids = {row["task_id"] for row in manifest["changed_tasks"]}
        changed = [spec for spec in refreshed if spec["task_id"] in changed_ids]
        audit = audit_tasks(changed, args.audit_timeout) if changed else None
    except ValueError as error:
        parser.error(str(error))
    manifest["source_file"] = str(Path(args.tasks_file).resolve())
    manifest["reference_source"] = (str(Path(args.reference_tasks_file).resolve())
                                    if args.reference_tasks_file else "repository_pilot_v2_pools")
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "tasks.jsonl", refreshed)
    write_json(output / "migration-manifest.json", manifest)
    if audit is not None:
        write_json(output / "migration-audit.json", audit)
    print(json.dumps({"output": str(output), "task_counts": manifest["task_counts"],
                      "changed_tasks": manifest["changed_tasks"],
                      "held_out_specs_unchanged": manifest["held_out_specs_unchanged"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
