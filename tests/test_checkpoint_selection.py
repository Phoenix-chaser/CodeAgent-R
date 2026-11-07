"""Execution checkpoint selection must use dev, fixed budgets, and real reports."""
import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate import content_hash
import select_checkpoint
from select_checkpoint import checkpoint_candidates, choose_checkpoint
from collect_data import collection_status
from agentlab.benchmark.builder import SPECS
from agentlab.benchmark.protocol import spec_fingerprint


def report(adapter=None, passes=(True, False), split="dev"):
    generation = {"repeats": 1, "seed": 42}
    protocol = {"generation": generation, "split": split, "benchmark_hash": "tasks",
                "prompt_hashes": {"first": "prompt-first", "second": "prompt-second"}}
    return {
        "schema_version": "codeagent-eval-v2", "model": "qwen", "resolved_revision": "commit",
        "adapter": adapter, "split": split, "tasks_hash": "tasks", "protocol": protocol,
        "protocol_hash": content_hash(protocol), "task_ids": ["first", "second"],
        "task_hashes": {"first": "hash-first", "second": "hash-second"},
        "generation": generation, "contamination": {"status": "clean"},
        "results": [
            {"task_id": task_id, "repeat_index": 0, "strict_pass": passed,
             "seed": 42, "prompt_hash": f"prompt-{task_id}",
             "verdict": "strict_pass" if passed else "strict_fail"}
            for task_id, passed in zip(("first", "second"), passes)
        ],
    }


def test_execution_selection_keeps_base_on_tie_or_regression():
    candidates = [{"epoch": 0, "adapter": None}, {"epoch": 3, "adapter": "epoch3"}]
    for scores in [(True, False), (False, False), (False, True)]:
        result = choose_checkpoint(candidates, [report(), report("epoch3", scores)])
        assert result["selected_adapter"] is None
        assert result["decision"] == "keep_base_no_dev_gain"


def test_execution_selection_can_choose_earlier_epoch_over_lower_loss():
    candidates = [{"epoch": 0, "adapter": None}, {"epoch": 1, "adapter": "epoch1"},
                  {"epoch": 2, "adapter": "epoch2"}]
    reports = [report(), report("epoch1", (True, True)), report("epoch2", (True, False))]
    result = choose_checkpoint(candidates, reports)
    assert result["selected_epoch"] == 1
    assert result["improvement_over_base"] == 0.5


@pytest.mark.parametrize("mutation", ["test", "dirty", "infra", "identity"])
def test_selection_rejects_noncomparable_reports(mutation):
    candidates = [{"epoch": 0, "adapter": None}, {"epoch": 1, "adapter": "epoch1"}]
    candidate = report("epoch1")
    if mutation == "test":
        candidate = report("epoch1", split="test")
    elif mutation == "dirty":
        candidate["contamination"]["status"] = "contaminated"
    elif mutation == "infra":
        candidate["results"][0]["failure_type"] = "infrastructure_error"
    else:
        candidate["adapter"] = "different"
    with pytest.raises(ValueError):
        choose_checkpoint(candidates, [report(), candidate])


def test_checkpoint_index_resolves_paths_and_zero_adapter(tmp_path):
    (tmp_path / "training-manifest.json").write_text(json.dumps({"best_epoch": 0}))
    _, candidates = checkpoint_candidates(tmp_path)
    assert candidates == [{"epoch": 0, "adapter": None}]
    (tmp_path / "checkpoint-index.json").write_text(json.dumps({
        "checkpoints": [{"epoch": 1, "adapter": "../outside/adapter"}]}))
    with pytest.raises(ValueError, match="out-of-root"):
        checkpoint_candidates(tmp_path)


def write_adapter(root, epoch=1, indexed=True):
    adapter = root / (f"epoch-{epoch:03d}/adapter" if indexed else "adapter")
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"verified-weights")
    (adapter / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    hashes = {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
              for file in adapter.iterdir()}
    manifest = {"best_epoch": epoch, "checkpoint_epoch": epoch, "adapter_files_sha256": hashes}
    (root / "training-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (adapter.parent / "training-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if indexed:
        (root / "checkpoint-index.json").write_text(json.dumps({"checkpoints": [
            {"epoch": 0, "adapter": None},
            {"epoch": epoch, "adapter": str(adapter.relative_to(root)),
             "adapter_files_sha256": hashes},
        ]}), encoding="utf-8")
    return adapter, hashes


@pytest.mark.parametrize("indexed", [True, False])
def test_checkpoint_hashes_are_verified_and_preserved_in_configuration(tmp_path, indexed):
    adapter, hashes = write_adapter(tmp_path, indexed=indexed)
    _, candidates = checkpoint_candidates(tmp_path)
    assert candidates[1]["adapter"] == str(adapter)
    assert candidates[1]["adapter_files_sha256"] == hashes
    assert candidates[1]["adapter_sha256"] == hashes["adapter_model.safetensors"]
    (adapter / "adapter_model.safetensors").write_bytes(b"different-epoch-weights")
    with pytest.raises(ValueError, match="recorded hashes"):
        checkpoint_candidates(tmp_path)


def test_checkpoint_candidates_expand_home_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    adapter, _ = write_adapter(tmp_path / "old-project" / "checkpoint", indexed=False)
    _, candidates = checkpoint_candidates("~/old-project/checkpoint")
    assert candidates[1]["adapter"] == str(adapter)


def test_selection_cli_expands_quoted_home_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "training-manifest.json").write_text(json.dumps({
        "best_epoch": 0, "model": "qwen", "resolved_revision": "commit",
        "data": {"task_ids": ["separate-training-task"]},
    }))
    tasks_file = tmp_path / "tasks.jsonl"
    tasks_file.write_text(json.dumps({**SPECS[0], "split": "dev"}) + "\n")
    monkeypatch.setattr(sys, "argv", ["select_checkpoint.py",
                                      "--checkpoint-root", "~/checkpoint",
                                      "--tasks-file", "~/tasks.jsonl",
                                      "--output", "~/selection"])

    def evaluate(command, check):
        assert check is True
        assert command[command.index("--tasks-file") + 1] == str(tasks_file)
        assert command[command.index("--training-data") + 1] == str(checkpoint / "training-manifest.json")
        output = Path(command[command.index("--output") + 1])
        assert output.parent == tmp_path / "selection"
        output.write_text(json.dumps(report()))

    monkeypatch.setattr(select_checkpoint.subprocess, "run", evaluate)
    select_checkpoint.main()
    selected = json.loads((tmp_path / "selection/selection.json").read_text())
    assert selected["selected_adapter"] is None
    assert selected["configuration"]["checkpoint_root"] == str(checkpoint)


@pytest.mark.parametrize("mutation", ["tokenizer", "missing_file", "missing_manifest",
                                      "epoch", "index_hash", "missing_index_hash",
                                      "missing_manifest_hash"])
def test_checkpoint_selection_rejects_untraceable_artifacts(tmp_path, mutation):
    adapter, _ = write_adapter(tmp_path)
    manifest_path = adapter.parent / "training-manifest.json"
    index_path = tmp_path / "checkpoint-index.json"
    if mutation == "tokenizer":
        (adapter / "tokenizer_config.json").write_text('{"different": true}')
    elif mutation == "missing_file":
        (adapter / "tokenizer_config.json").unlink()
    elif mutation == "missing_manifest":
        manifest_path.unlink()
    elif mutation in {"epoch", "missing_manifest_hash"}:
        manifest = json.loads(manifest_path.read_text())
        if mutation == "epoch":
            manifest["checkpoint_epoch"] = 2
        else:
            manifest.pop("adapter_files_sha256")
        manifest_path.write_text(json.dumps(manifest))
    else:
        index = json.loads(index_path.read_text())
        if mutation == "index_hash":
            index["checkpoints"][1]["adapter_files_sha256"]["adapter_model.safetensors"] = "incorrect"
        else:
            index["checkpoints"][1].pop("adapter_files_sha256")
        index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError):
        checkpoint_candidates(tmp_path)


def test_no_retention_index_uses_selected_epoch_from_root_manifest(tmp_path):
    _, hashes = write_adapter(tmp_path, epoch=2, indexed=False)
    (tmp_path / "checkpoint-index.json").write_text(json.dumps({"checkpoints": [
        {"epoch": 0, "adapter": None},
        {"epoch": 2, "adapter": "adapter", "adapter_files_sha256": hashes},
    ]}))
    _, candidates = checkpoint_candidates(tmp_path)
    assert candidates[1]["epoch"] == 2


def test_collection_status_tracks_latest_attempt_not_line_count(tmp_path):
    spec = SPECS[0]
    path = tmp_path / "candidates.jsonl"
    failed = {"task_id": spec["task_id"], "task_hash": spec_fingerprint(spec),
              "candidate_idx": 0, "provenance": {"verdict": "collection_error"}}
    path.write_text(json.dumps(failed) + "\n" + json.dumps(failed) + "\n")
    latest, missing, retryable = collection_status(path, [spec], 2)
    assert len(latest) == 1
    assert missing == {(spec["task_id"], 1)}
    assert retryable == {(spec["task_id"], 0)}
    repaired = copy.deepcopy(failed)
    repaired["provenance"]["verdict"] = "strict_pass"
    with path.open("a") as handle:
        handle.write(json.dumps(repaired) + "\n")
    assert not collection_status(path, [spec], 2)[2]
