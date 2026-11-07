"""Migrate verified teacher candidates into a newly frozen experiment split.

Reads old candidate records, keeps those whose task_hash matches the new
frozen spec fingerprint (and whose prompt still matches the canonical public
context), and writes them plus a collect-compatible protocol manifest so
``collect_data.py`` only fills the remaining slots instead of re-paying for
everything.

Usage:
  python scripts/migrate_candidates.py \
    --tasks-file data/experiment-v6/tasks.jsonl --split train \
    --old-candidates data/train-candidates-v3.jsonl \
    --output data/train-candidates-v4.jsonl --candidates 12
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agentlab.benchmark.protocol import (build_messages, load_specs,
                                         protocol_hash, spec_fingerprint)

CARRIER_FIELDS = ("model", "base_url", "max_tokens", "temperature",
                  "exploration_temperature", "include_failures")


def migrate(tasks_file: str, split: str, old_candidates: str,
            output: str, candidates: int, old_manifest: str | None = None,
            model: str | None = None, base_url: str | None = None) -> dict:
    if candidates < 1:
        raise ValueError("candidates must be positive")
    specs = {spec["task_id"]: spec for spec in load_specs(tasks_file, split)}
    if not specs:
        raise ValueError(f"no tasks with split={split!r} in {tasks_file}")

    old_manifest_path = Path(old_manifest) if old_manifest else \
        Path(str(old_candidates).removesuffix(".jsonl") + ".manifest.json")
    if not old_manifest_path.exists():
        raise ValueError(f"old manifest not found: {old_manifest_path}")
    previous = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    carriers = {field: previous["protocol"][field] for field in CARRIER_FIELDS}
    if model:
        carriers["model"] = model
    if base_url:
        carriers["base_url"] = base_url
    protocol = {"schema_version": 2, "tasks_hash": protocol_hash(list(specs.values())),
                "split": split, **carriers}

    output_path = Path(output)
    manifest_path = output_path.with_suffix(".manifest.json")
    existing_slots: set[tuple[str, int]] = set()
    if output_path.exists():
        if not manifest_path.exists():
            raise ValueError(f"existing output has no protocol manifest: {output_path}")
        written = json.loads(manifest_path.read_text(encoding="utf-8"))
        if written.get("protocol") != protocol:
            raise ValueError("existing output protocol differs; use a new output path")
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                existing_slots.add((record["task_id"], record["candidate_idx"]))

    skipped: Counter[str] = Counter()
    migrated: Counter[str] = Counter()
    kept_records = []
    old_path = Path(old_candidates)
    if not old_path.exists():
        raise ValueError(f"old candidates file not found: {old_path}")
    for line in old_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        task_id = record.get("task_id")
        slot = (task_id, record.get("candidate_idx"))
        if slot in existing_slots:
            skipped["already_present"] += 1
            continue
        if not record.get("provenance", {}).get("strict_pass"):
            skipped["not_strict_pass"] += 1
            continue
        if record.get("split") != split:
            skipped["split_mismatch"] += 1
            continue
        spec = specs.get(task_id)
        if spec is None:
            skipped["task_not_in_split"] += 1
            continue
        if record.get("task_hash") != spec_fingerprint(spec):
            skipped["task_hash_mismatch"] += 1
            continue
        if record.get("messages") != build_messages(spec):
            skipped["prompt_mismatch"] += 1
            continue
        if not isinstance(record.get("candidate_idx"), int) or \
                not 0 <= record["candidate_idx"] < candidates:
            skipped["candidate_idx_out_of_range"] += 1
            continue
        existing_slots.add(slot)
        migrated[task_id] += 1
        kept_records.append(line)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for line in kept_records:
            handle.write(line + "\n")
    manifest_path.write_text(json.dumps({"protocol": protocol},
                                        ensure_ascii=False, indent=2),
                             encoding="utf-8")
    total_slots = len(specs) * candidates
    report = {
        "schema_version": 1, "split": split, "output": str(output_path),
        "protocol": protocol,
        "tasks": len(specs), "slots": total_slots,
        "migrated_records": sum(migrated.values()),
        "migrated_tasks": len(migrated),
        "tasks_needing_collection": len(specs) - len(migrated),
        "remaining_slots": total_slots - len(existing_slots),
        "skipped": dict(skipped),
        "per_task_migrated": dict(sorted(migrated.items())),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-file", required=True, help="New frozen task JSONL")
    parser.add_argument("--split", choices=["train", "dev"], required=True)
    parser.add_argument("--old-candidates", required=True, help="Old candidate JSONL")
    parser.add_argument("--old-manifest", default=None,
                        help="Defaults to <old-candidates>.manifest.json")
    parser.add_argument("--output", required=True, help="New candidate JSONL")
    parser.add_argument("--candidates", type=int, default=None,
                        help="Slots per task for reporting (default: 12 train / 4 dev)")
    parser.add_argument("--model", default=None, help="Override teacher model name")
    parser.add_argument("--base-url", default=None, help="Override teacher base URL")
    args = parser.parse_args()
    slots = args.candidates if args.candidates is not None else \
        (12 if args.split == "train" else 4)
    try:
        report = migrate(args.tasks_file, args.split, args.old_candidates,
                         args.output, slots, args.old_manifest,
                         args.model, args.base_url)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
