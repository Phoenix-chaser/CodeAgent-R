"""Select an adapter using fixed dev execution tests, including the base model."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agentlab.benchmark.protocol import load_specs, protocol_hash
from compare import validate_comparison
from evaluate import content_hash, training_overlap


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verified_adapter(root, adapter, epoch, expected_hashes=None):
    adapter = adapter.resolve()
    if not adapter.is_relative_to(root) or not (adapter / "adapter_model.safetensors").is_file():
        raise ValueError(f"Missing or out-of-root checkpoint: {adapter}")
    manifest_path = adapter.parent / "training-manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Checkpoint is missing its training manifest: {adapter}")
    manifest = read_json(manifest_path)
    recorded_epoch = manifest.get("best_epoch") if adapter.parent == root else manifest.get("checkpoint_epoch")
    if recorded_epoch != epoch:
        raise ValueError(f"Checkpoint epoch differs from manifest: {adapter}")
    declared_hashes = manifest.get("adapter_files_sha256")
    if not isinstance(declared_hashes, dict) or "adapter_model.safetensors" not in declared_hashes:
        raise ValueError(f"Checkpoint manifest lacks adapter hashes: {adapter}")
    if expected_hashes is not None and expected_hashes != declared_hashes:
        raise ValueError(f"Checkpoint index hashes differ from manifest: {adapter}")
    actual_hashes = {
        file.name: hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted(adapter.iterdir()) if file.is_file()
    }
    if actual_hashes != declared_hashes:
        raise ValueError(f"Checkpoint files differ from recorded hashes: {adapter}")
    return {"epoch": epoch, "adapter": str(adapter),
            "adapter_files_sha256": actual_hashes,
            "adapter_sha256": actual_hashes["adapter_model.safetensors"]}


def checkpoint_candidates(root):
    root = Path(root).expanduser().resolve()
    manifest = read_json(root / "training-manifest.json")
    candidates = [{"epoch": 0, "adapter": None}]
    index_path = root / "checkpoint-index.json"
    if index_path.exists():
        for entry in read_json(index_path)["checkpoints"]:
            if not entry.get("adapter"):
                continue
            adapter = (root / entry["adapter"]).resolve()
            if not adapter.is_relative_to(root):
                raise ValueError(f"Missing or out-of-root checkpoint: {entry['adapter']}")
            if not isinstance(entry.get("adapter_files_sha256"), dict):
                raise ValueError(f"Checkpoint index lacks adapter hashes: {entry['adapter']}")
            candidates.append(verified_adapter(root, adapter, entry["epoch"],
                                               entry["adapter_files_sha256"]))
    if len(candidates) == 1:
        if manifest.get("best_epoch") == 0:
            return manifest, candidates
        adapter = root / "adapter"
        candidates.append(verified_adapter(root, adapter, manifest.get("best_epoch", 1)))
    if len({entry["epoch"] for entry in candidates}) != len(candidates):
        raise ValueError("Duplicate checkpoint epochs; no trained epoch available")
    return manifest, sorted(candidates, key=lambda entry: entry["epoch"])


def choose_checkpoint(candidates, reports):
    if len(candidates) != len(reports) or not candidates or candidates[0]["adapter"] is not None:
        raise ValueError("Selection requires a base report followed by matching checkpoints")
    baseline = reports[0]
    rows = []
    for candidate, report in zip(candidates, reports):
        if report.get("split") != "dev":
            raise ValueError("Checkpoint selection only accepts dev reports")
        if report.get("contamination", {}).get("status") != "clean":
            raise ValueError("Dev report lacks a clean training-overlap audit")
        if report.get("schema_version") != "codeagent-eval-v2":
            raise ValueError("Checkpoint selection requires a verified modern report")
        if report.get("adapter") != candidate["adapter"]:
            raise ValueError("Evaluation adapter identity differs from checkpoint")
        base_tasks, candidate_tasks, warnings = validate_comparison(baseline, report)
        if warnings:
            raise ValueError("Cannot select from unverified reports: " + "; ".join(warnings))
        base_rates = {task: sum(attempts.values()) / len(attempts) for task, attempts in base_tasks.items()}
        rates = {task: sum(attempts.values()) / len(attempts) for task, attempts in candidate_tasks.items()}
        rows.append({
            **candidate, "pass_at_1": sum(rates.values()) / len(rates),
            "improved_tasks": sum(rates[task] > base_rates[task] for task in rates),
            "regressed_tasks": sum(rates[task] < base_rates[task] for task in rates),
            "protocol_hash": report["protocol_hash"],
        })
    selected = max(rows, key=lambda row: (
        row["pass_at_1"], -row["regressed_tasks"], -row["epoch"]))
    return {
        "schema_version": 1, "selection_metric": "dev_execution_pass_at_1",
        "tie_break": "fewest regressions versus base, then earliest epoch (base wins ties)",
        "selected_epoch": selected["epoch"], "selected_adapter": selected["adapter"],
        "base_pass_at_1": rows[0]["pass_at_1"], "selected_pass_at_1": selected["pass_at_1"],
        "improvement_over_base": selected["pass_at_1"] - rows[0]["pass_at_1"],
        "decision": "adapter_selected" if selected["adapter"] else "keep_base_no_dev_gain",
        "candidates": rows, "scope": "Dev model selection only; no test-generalization claim",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--tasks-file", required=True)
    parser.add_argument("--output", required=True, help="New directory for all dev reports and selection.json")
    parser.add_argument("--quantization", choices=["nf4", "bf16"], default="nf4")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(args.checkpoint_root).expanduser().resolve()
    tasks_file = Path(args.tasks_file).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Selection output must be empty; prior evidence must not be overwritten")
    manifest, candidates = checkpoint_candidates(root)
    specs = load_specs(tasks_file, "dev")
    if training_overlap(specs, root / "training-manifest.json")["status"] != "clean":
        parser.error("Dev tasks overlap training identities")
    revision = manifest.get("resolved_revision") or manifest.get("revision")
    if not revision:
        parser.error("Training manifest must pin an exact base revision")
    output.mkdir(parents=True, exist_ok=True)
    configuration = {"checkpoint_root": str(root), "tasks_hash": protocol_hash(specs),
                     "candidates": candidates, "model": manifest["model"], "revision": revision,
                     "quantization": args.quantization, "seed": args.seed,
                     "max_new_tokens": args.max_new_tokens}
    (output / "selection-config.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2), encoding="utf-8")
    reports = []
    for candidate in candidates:
        report_path = output / f"epoch-{candidate['epoch']:03d}-dev.json"
        command = [
            sys.executable, str(ROOT / "scripts/evaluate.py"), "--model", manifest["model"],
            "--revision", revision, "--tasks-file", str(tasks_file),
            "--split", "dev", "--training-data", str(root / "training-manifest.json"),
            "--quantization", args.quantization, "--temperature", "0",
            "--seed", str(args.seed), "--max-new-tokens", str(args.max_new_tokens),
            "--output", str(report_path),
        ]
        if candidate["adapter"]:
            command.extend(["--adapter", candidate["adapter"]])
        subprocess.run(command, check=True)
        reports.append(read_json(report_path))
    result = choose_checkpoint(candidates, reports)
    result.update(configuration=configuration, configuration_sha256=content_hash(configuration))
    (output / "selection.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
