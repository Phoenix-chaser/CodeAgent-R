"""Migration preserves frozen task identity and all held-out evidence."""
import copy
import sys
from pathlib import Path

import pytest

from agentlab.benchmark.pilot_tasks import PILOT_SPECS
from agentlab.benchmark.protocol import spec_fingerprint

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from refresh_train_specs import refresh_train_specs


def frozen_specs():
    specs = copy.deepcopy([PILOT_SPECS[index] for index in (0, 2, 4)])
    specs[1]["split"] = "dev"
    specs[2]["split"] = "test"
    return specs


def test_refresh_changes_only_training_private_material_and_records_hashes():
    specs = frozen_specs()
    before = copy.deepcopy(specs)
    references = copy.deepcopy(specs)
    for reference in references:
        reference["fixed"] += "\n"
        reference["private_tests"]["test_extra.py"] = "def test_extra():\n    assert True\n"
    refreshed, manifest = refresh_train_specs(specs, references)
    assert specs == before
    assert refreshed[0]["fixed"] == references[0]["fixed"]
    assert refreshed[1:] == before[1:]
    assert manifest["held_out_specs_unchanged"] is True
    assert manifest["changed_tasks"] == [
        {"task_id": specs[0]["task_id"], "fields": ["fixed", "private_tests"]}]
    assert manifest["task_hashes"][0]["old_task_hash"] == spec_fingerprint(before[0])
    assert manifest["task_hashes"][0]["new_task_hash"] == spec_fingerprint(refreshed[0])
    assert manifest["task_counts"] == {"train": 1, "dev": 1, "test": 1}


@pytest.mark.parametrize("field,value", [
    ("instruction", "New prompt"), ("buggy", "def changed():\n    return None\n"),
    ("source_group", "new-source"), ("difficulty", "L4"),
])
def test_refresh_refuses_changed_public_fields(field, value):
    specs = frozen_specs()
    references = copy.deepcopy(specs)
    references[0][field] = value
    with pytest.raises(ValueError, match="frozen public metadata changed"):
        refresh_train_specs(specs, references)


def test_refresh_preserves_unknown_training_tasks_and_split_assignments():
    specs = frozen_specs()
    references = copy.deepcopy(specs[1:])
    for reference in references:
        reference["split"] = "train"
    refreshed, manifest = refresh_train_specs(specs, references)
    assert refreshed == specs
    assert manifest["changed_tasks"] == []
    assert manifest["unmatched_train_task_ids"] == [specs[0]["task_id"]]
