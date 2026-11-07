"""Checkpoint artifacts must remain evaluable after later training failures."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from agentlab.training.hf_sft import save_epoch_checkpoint, write_checkpoint_index


class AdapterModel:
    def __init__(self, weights=b"epoch-one-weights"):
        self.weights = weights

    def save_pretrained(self, path, safe_serialization):
        assert safe_serialization is True
        (Path(path) / "adapter_model.safetensors").write_bytes(self.weights)


class AdapterTokenizer:
    def save_pretrained(self, path):
        (Path(path) / "tokenizer_config.json").write_text("{}", encoding="utf-8")


def training_manifest():
    return {
        "model": "example/model", "revision": "pinned-revision",
        "resolved_revision": "pinned-revision", "chat_template_sha256": "template-hash",
        "data": {"task_ids": ["train-task"], "source_groups": ["train-group"]},
        "eval_data": {"task_ids": ["dev-task"], "source_groups": ["dev-group"]},
        "best_epoch": 1, "history": [{"epoch": 1, "dev_loss": 0.5}],
        "selection_metric": "dev_token_loss",
    }


def test_epoch_artifact_keeps_model_and_split_provenance(tmp_path):
    manifest = training_manifest()
    entry = save_epoch_checkpoint(AdapterModel(), AdapterTokenizer(), tmp_path,
                                  manifest, epoch=1, optimizer_steps=9, dev_loss=0.5)
    assert entry["adapter"] == "epoch-001/adapter"
    assert entry["training_manifest"] == "epoch-001/training-manifest.json"
    snapshot_path = tmp_path / entry["training_manifest"]
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot_path.parent == (tmp_path / entry["adapter"]).parent
    assert snapshot["status"] == "checkpoint_complete"
    assert snapshot["checkpoint_epoch"] == 1
    assert snapshot["optimizer_steps"] == 9
    for key in ("model", "revision", "resolved_revision", "chat_template_sha256",
                "data", "eval_data", "selection_metric"):
        assert snapshot[key] == manifest[key]
    weights = tmp_path / entry["adapter"] / "adapter_model.safetensors"
    assert entry["adapter_files_sha256"][weights.name] == hashlib.sha256(weights.read_bytes()).hexdigest()
    manifest["history"].append({"epoch": 2, "dev_loss": 0.7})
    assert len(json.loads(snapshot_path.read_text())["history"]) == 1


def test_epoch_manifest_supports_existing_contamination_checks(tmp_path):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    from evaluate import training_overlap

    entry = save_epoch_checkpoint(AdapterModel(), AdapterTokenizer(), tmp_path,
                                  training_manifest(), 1, 9, 0.5)
    spec = {
        "task_id": "dev-task", "source_group": "dev-group", "module": "repair.py",
        "instruction": "Repair the function", "buggy": "def value(): return None\n",
        "public_test": "def test_public(): pass\n", "context_files": {},
    }
    path = tmp_path / entry["training_manifest"]
    assert training_overlap([spec], path)["status"] == "clean"
    assert training_overlap([spec], path, include_selection=True)["status"] == "contaminated"


def test_checkpoint_index_retains_base_and_all_epochs(tmp_path):
    manifest = training_manifest()
    entries = [{"epoch": 0, "optimizer_steps": 0, "dev_loss": 0.6,
                "adapter": None, "training_manifest": "training-manifest.json",
                "adapter_files_sha256": {}}]
    for epoch, dev_loss in ((1, 0.5), (2, 0.7)):
        entries.append(save_epoch_checkpoint(AdapterModel(str(epoch).encode()),
                                            AdapterTokenizer(), tmp_path, manifest,
                                            epoch, epoch * 9, dev_loss))
    write_checkpoint_index(tmp_path, manifest, entries)
    index = json.loads((tmp_path / "checkpoint-index.json").read_text())
    assert index["selection_metric"] == "dev_token_loss"
    assert index["selected_epoch"] == 1
    assert [entry["epoch"] for entry in index["checkpoints"]] == [0, 1, 2]
    assert index["checkpoints"][0]["adapter"] is None
    assert not list(tmp_path.glob("*.tmp"))


def test_later_save_failure_does_not_destroy_completed_epoch(tmp_path):
    manifest = training_manifest()
    entry = save_epoch_checkpoint(AdapterModel(), AdapterTokenizer(), tmp_path,
                                  manifest, 1, 9, 0.5)
    write_checkpoint_index(tmp_path, manifest, [entry])
    index_before = (tmp_path / "checkpoint-index.json").read_bytes()
    manifest_before = (tmp_path / entry["training_manifest"]).read_bytes()

    class FailedModel:
        def save_pretrained(self, path, safe_serialization):
            raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        save_epoch_checkpoint(FailedModel(), AdapterTokenizer(), tmp_path,
                              manifest, 2, 18, 0.4)
    assert (tmp_path / "checkpoint-index.json").read_bytes() == index_before
    assert (tmp_path / entry["training_manifest"]).read_bytes() == manifest_before
    with pytest.raises(ValueError, match="already exists"):
        save_epoch_checkpoint(AdapterModel(b"replacement"), AdapterTokenizer(),
                              tmp_path, manifest, 1, 9, 0.4)
    weights = tmp_path / entry["adapter"] / "adapter_model.safetensors"
    assert weights.read_bytes() == b"epoch-one-weights"


def test_checkpoint_retention_cli_defaults_and_opt_out():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_sft.py"
    spec = importlib.util.spec_from_file_location("train_checkpoint_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = module.build_parser()
    assert parser.parse_args(["--data", "train.jsonl"]).save_every_epoch is True
    assert parser.parse_args(["--data", "train.jsonl", "--no-save-every-epoch"]).save_every_epoch is False
