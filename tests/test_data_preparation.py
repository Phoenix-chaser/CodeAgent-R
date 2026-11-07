"""Regressions for held-out data contracts and verified teacher collection."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentlab.benchmark.builder import SPECS, build_all
from agentlab.benchmark.protocol import build_messages, load_specs, protocol_hash, spec_fingerprint
from agentlab.benchmark.task_store import TaskStore
from agentlab.training.preparation import freeze_splits, prepare_records

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from collect_data import collect_one


def independent_spec(index):
    return {
        "task_id": f"independent_{index}", "source_group": f"repository_{index}",
        "split": "train", "bug_family": "rounding", "difficulty": "L1",
        "instruction": f'Return the rounded value with offset "{index}".',
        "module": "rounding.py", "buggy": f"def rounded(value):\n    return int(value) + {index}\n",
        "fixed": f"def rounded(value):\n    return round(value) + {index}\n",
        "public_test": f"from rounding import rounded\ndef test_rounding():\n    assert rounded(1.9) == {index + 2}\n",
        "private_tests": {"test_negative.py": f"from rounding import rounded\ndef test_negative():\n    assert rounded(-1.9) == {index - 2}\n"},
    }


def sample(spec):
    return {
        "task_id": spec["task_id"], "source_group": spec["source_group"],
        "split": spec["split"], "task_hash": spec_fingerprint(spec),
        "messages": build_messages(spec), "target": {"role": "assistant", "content": spec["fixed"]},
        "provenance": {"strict_pass": True},
    }


def test_jsonl_loader_preserves_frozen_split_and_valid_yaml(tmp_path):
    specs = [independent_spec(index) for index in range(3)]
    specs[1]["split"] = "dev"
    path = tmp_path / "tasks.jsonl"
    path.write_text("\n".join(json.dumps(spec) for spec in specs), encoding="utf-8")
    assert load_specs(path, "dev") == [specs[1]]
    build_all(tmp_path / "built", specs)
    assert TaskStore(tmp_path / "built").load(specs[0]["task_id"]).public.instruction == specs[0]["instruction"]


def test_legacy_tasks_are_smoke_and_not_training_material():
    assert len(load_specs(split="smoke")) == 14
    with pytest.raises(ValueError, match="legacy"):
        freeze_splits(SPECS)


def test_public_prompt_contains_dependencies_but_no_private_material():
    spec = copy.deepcopy(SPECS[6])
    spec["private_tests"] = {"test_secret.py": "PRIVATE_TEST_MARKER"}
    spec["fixed"] = "REFERENCE_SOLUTION_MARKER"
    prompt = json.dumps(build_messages(spec), ensure_ascii=False)
    assert spec["extra_files"]["data_store.py"] in build_messages(spec)[1]["content"]
    assert spec["public_test"] in build_messages(spec)[1]["content"]
    assert "PRIVATE_TEST_MARKER" not in prompt
    assert "REFERENCE_SOLUTION_MARKER" not in prompt


def test_frozen_split_reproducible_grouped_and_duplicate_safe():
    specs = [independent_spec(index) for index in range(10)]
    frozen, manifest = freeze_splits(specs)
    assert freeze_splits(specs) == (frozen, manifest)
    assert set(manifest["task_counts"]) == {"train", "dev", "test"}
    assert protocol_hash(frozen) == protocol_hash(list(reversed(frozen)))
    specs[1]["buggy"] = specs[0]["buggy"]
    with pytest.raises(ValueError, match="duplicate task code"):
        freeze_splits(specs)


def test_preparation_deduplicates_ast_and_rejects_stale_or_heldout_data():
    specs = [independent_spec(index) for index in range(3)]
    specs[1]["split"] = "dev"
    specs[2]["split"] = "test"
    original = sample(specs[0])
    duplicate = copy.deepcopy(original)
    duplicate["target"]["content"] += "\n"
    outputs, manifest = prepare_records([original, duplicate, sample(specs[1])], specs)
    assert len(outputs["train"]) == 1
    assert manifest["rejected"]["duplicate_ast"] == 1
    with pytest.raises(ValueError, match="test/smoke"):
        prepare_records([original, sample(specs[2])], specs)
    original["task_hash"] = "stale"
    with pytest.raises(ValueError, match="stale"):
        prepare_records([original], specs)


def test_prepare_rejects_wrong_prompt_and_boolean_string_label():
    specs = [independent_spec(index) for index in range(2)]
    specs[1]["split"] = "dev"
    malformed = sample(specs[0])
    malformed["messages"][1]["content"] += "\nleaked answer"
    with pytest.raises(ValueError, match="canonical"):
        prepare_records([malformed, sample(specs[1])], specs)
    malformed = sample(specs[0])
    malformed["provenance"]["strict_pass"] = "true"
    with pytest.raises(ValueError, match="no accepted"):
        prepare_records([malformed, sample(specs[1])], specs)


def test_preparation_prefers_verified_minimal_edits_independent_of_input_order():
    specs = [independent_spec(index) for index in range(2)]
    specs[1]["split"] = "dev"
    minimal = sample(specs[0])
    rewritten = copy.deepcopy(minimal)
    rewritten["target"]["content"] = (
        "def rounded(value):\n    result = round(value)\n    return result + 0\n")
    records = [rewritten, minimal, sample(specs[1])]
    outputs, manifest = prepare_records(records, specs, replay=lambda spec, code: ("strict_pass", True, {}))
    reversed_outputs, _ = prepare_records(list(reversed(records)), specs,
                                          replay=lambda spec, code: ("strict_pass", True, {}))
    assert outputs == reversed_outputs
    assert outputs["train"][0]["target"] == minimal["target"]
    assert manifest["rejected"]["task_cap"] == 1
    assert manifest["max_per_task"] == 1
    assert manifest["splits"]["train"]["samples_per_task"] == {specs[0]["task_id"]: 1}


def test_preparation_replays_before_cap_and_reports_missing_tasks():
    specs = [independent_spec(index) for index in range(3)]
    specs[1]["split"] = "dev"
    unchanged = sample(specs[0])
    unchanged["target"]["content"] = specs[0]["buggy"]

    def replay(spec, code):
        passed = code == spec["fixed"]
        return "strict_pass" if passed else "partial", passed, {"checked": True}

    outputs, manifest = prepare_records([unchanged, sample(specs[0]), sample(specs[1])],
                                        specs, replay=replay)
    assert outputs["train"][0]["target"]["content"] == specs[0]["fixed"]
    assert manifest["rejected"]["replay_failed"] == 1
    assert manifest["replayed_candidates"] == 3
    assert manifest["replay_rejections"][0]["verdict"] == "partial"
    assert manifest["splits"]["train"]["missing_task_ids"] == [specs[2]["task_id"]]


def test_stale_revalidation_requires_replay_and_identical_public_context():
    specs = [independent_spec(index) for index in range(2)]
    specs[1]["split"] = "dev"
    original = sample(specs[0])
    prior_hash = original["task_hash"]
    specs[0]["private_tests"]["test_extra.py"] = "def test_extra():\n    assert True\n"
    records = [original, sample(specs[1])]
    with pytest.raises(ValueError, match="requires grader replay"):
        prepare_records(records, specs, revalidate_stale=True)
    outputs, manifest = prepare_records(records, specs, revalidate_stale=True,
                                        replay=lambda spec, code: ("strict_pass", True, {}))
    accepted = outputs["train"][0]
    assert accepted["task_hash"] == spec_fingerprint(specs[0])
    assert accepted["provenance"]["revalidated_from_task_hash"] == prior_hash
    assert manifest["revalidated_stale_samples"] == 1
    specs[0]["instruction"] += " New public contract."
    with pytest.raises(ValueError, match="canonical"):
        prepare_records(records, specs, revalidate_stale=True,
                        replay=lambda spec, code: ("strict_pass", True, {}))


def test_path_escape_and_cross_group_splits_rejected(tmp_path):
    specs = [independent_spec(index) for index in range(2)]
    specs[0]["module"] = "../escape.py"
    with pytest.raises(ValueError, match="file names"):
        build_all(tmp_path, specs)
    specs[0]["module"] = "rounding.py"
    specs[1].update(source_group=specs[0]["source_group"], split="dev")
    with pytest.raises(ValueError, match="crosses splits"):
        build_all(tmp_path, specs)


def test_incomplete_teacher_response_keeps_usage_without_grading(tmp_path):
    class Completions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="def broken("), finish_reason="length")],
                usage=SimpleNamespace(prompt_tokens=20, completion_tokens=40), id="fake")
    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    args = SimpleNamespace(temperature=0.2, exploration_temperature=0.7, model="fake", max_tokens=40)
    result = collect_one(independent_spec(0), 0, client, args, tmp_path, tmp_path)
    assert result["provenance"]["strict_pass"] is False
    assert result["provenance"]["verdict"] == "incomplete_response"
    assert result["usage"]["output_tokens"] == 40
    assert "target" not in result
