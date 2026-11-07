# -*- coding: utf-8 -*-
"""migrate_candidates: hash-matched candidate reuse into a new frozen split."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentlab.benchmark.pilot_tasks_v2d import V2D_SPECS
from agentlab.benchmark.protocol import build_messages, protocol_hash, spec_fingerprint
from migrate_candidates import migrate


@pytest.fixture()
def frozen(tmp_path):
    specs = []
    for spec in V2D_SPECS[:2]:
        specs.append({**spec, "split": "train"})
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in specs),
                     encoding="utf-8")
    return tmp_path, tasks, specs


def _record(spec, index, split="train", strict=True, hash_override=None):
    return {
        "record_type": "candidate",
        "sample_id": f"sft-{spec['task_id']}-{index:03d}",
        "task_id": spec["task_id"],
        "task_hash": hash_override or spec_fingerprint(spec),
        "candidate_idx": index, "source_group": spec["source_group"],
        "split": split, "messages": build_messages(spec),
        "target": {"role": "assistant", "content": "pass"},
        "provenance": {"strict_pass": strict, "verdict": "strict_pass" if strict else "strict_fail"},
    }


def _old_files(tmp_path, records):
    old = tmp_path / "old.jsonl"
    old.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                   encoding="utf-8")
    protocol = {"schema_version": 2, "tasks_hash": "old", "split": "train",
                "model": "teacher-api-model", "base_url": "https://teacher-api.example/api/paas/v4",
                "max_tokens": 4096, "temperature": 0.2,
                "exploration_temperature": 0.7, "include_failures": False}
    manifest = tmp_path / "old.manifest.json"
    manifest.write_text(json.dumps({"protocol": protocol}), encoding="utf-8")
    return old


def test_migrates_only_hash_matching_records(frozen):
    tmp_path, tasks, specs = frozen
    stale = dict(specs[0])
    stale["instruction"] = stale["instruction"] + "（旧版）"
    records = [
        _record(specs[0], 0),
        _record(specs[0], 1, strict=False),
        _record(specs[0], 2, hash_override=spec_fingerprint(stale)),
        _record(specs[1], 0, split="dev"),
        _record(specs[1], 1),
    ]
    old = _old_files(tmp_path, records)
    report = migrate(str(tasks), "train", str(old), str(tmp_path / "new.jsonl"), 12)
    assert report["migrated_records"] == 2
    assert report["migrated_tasks"] == 2
    assert report["remaining_slots"] == 2 * 12 - 2
    assert report["skipped"] == {"not_strict_pass": 1, "task_hash_mismatch": 1,
                                 "split_mismatch": 1}
    lines = [json.loads(l) for l in (tmp_path / "new.jsonl").read_text().splitlines()]
    assert {r["candidate_idx"] for r in lines} == {0, 1}
    manifest = json.loads((tmp_path / "new.manifest.json").read_text())
    assert manifest["protocol"]["split"] == "train"
    assert manifest["protocol"]["model"] == "teacher-api-model"
    assert manifest["protocol"]["tasks_hash"] == protocol_hash(specs)


def test_rerun_is_idempotent(frozen):
    tmp_path, tasks, specs = frozen
    old = _old_files(tmp_path, [_record(specs[0], 0), _record(specs[1], 0)])
    first = migrate(str(tasks), "train", str(old), str(tmp_path / "new.jsonl"), 12)
    second = migrate(str(tasks), "train", str(old), str(tmp_path / "new.jsonl"), 12)
    assert first["migrated_records"] == 2
    assert second["migrated_records"] == 0
    assert second["skipped"]["already_present"] == 2
    assert len((tmp_path / "new.jsonl").read_text().splitlines()) == 2


def test_conflicting_protocol_rejected(frozen):
    tmp_path, tasks, specs = frozen
    old = _old_files(tmp_path, [_record(specs[0], 0)])
    migrate(str(tasks), "train", str(old), str(tmp_path / "new.jsonl"), 12)
    other = [_record(specs[0], 5)]
    other_old = tmp_path / "other.jsonl"
    other_old.write_text("".join(json.dumps(r) + "\n" for r in other), encoding="utf-8")
    protocol = {"schema_version": 2, "tasks_hash": "x", "split": "train",
                "model": "other-model", "base_url": "u", "max_tokens": 1,
                "temperature": 0.1, "exploration_temperature": 0.2,
                "include_failures": False}
    (tmp_path / "other.manifest.json").write_text(json.dumps({"protocol": protocol}))
    with pytest.raises(ValueError, match="protocol differs"):
        migrate(str(tasks), "train", str(other_old), str(tmp_path / "new.jsonl"), 12)


def test_missing_old_manifest_rejected(frozen):
    tmp_path, tasks, specs = frozen
    old = tmp_path / "lonely.jsonl"
    old.write_text(json.dumps(_record(specs[0], 0)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="old manifest"):
        migrate(str(tasks), "train", str(old), str(tmp_path / "new.jsonl"), 12)
