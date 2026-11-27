"""Freeze task-level splits and prepare verified repair demonstrations."""
from __future__ import annotations

import ast
import difflib
import hashlib
import io
import json
import random
import subprocess
import tokenize
from collections import Counter

from agentlab.benchmark.builder import SPECS
from agentlab.benchmark.protocol import build_messages, spec_fingerprint, validate_specs


def content_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def code_fingerprint(code: str) -> str:
    return content_hash(ast.dump(ast.parse(code), include_attributes=False))


def reject_legacy_tasks(specs: list[dict]) -> None:
    legacy_ids = {spec["task_id"] for spec in SPECS}
    legacy_groups = {spec["source_group"] for spec in SPECS}
    legacy_code = {code_fingerprint(spec["buggy"]) for spec in SPECS}
    for spec in specs:
        if (spec["task_id"] in legacy_ids or spec["source_group"] in legacy_groups
                or code_fingerprint(spec["buggy"]) in legacy_code):
            raise ValueError(f"{spec['task_id']}: legacy benchmark is diagnostic only; supply independent tasks")


def _stratum_of(spec: dict) -> str:
    """任务用 ``stratum`` 字段声明自己属于哪一层（例如难题池）；没有声明的都在默认层。"""
    return spec.get("stratum") or "default"


def _stratified_mapping(groups: list[str], strata: dict[str, str], seed: int,
                        dev_count: int, test_count: int) -> tuple[dict[str, str], dict]:
    """每个层拿到与它的大小成正比的 dev / test 组数，然后各层分别打乱。

    比例的整数部分先分下去，余数按小数部分从大到小补；每层至少留一组给 train。
    一层只有一组时无法再分，这时总数会略少于请求的组数，manifest 里如实记录。
    """
    sizes = Counter(strata[group] for group in groups)
    counts = {name: [size, 0, 0] for name, size in sizes.items()}     # 组数、dev 组数、test 组数
    for slot, total in ((1, dev_count), (2, test_count)):
        exact = {name: size * total / len(groups) for name, size in sizes.items()}
        for name in counts:
            counts[name][slot] = int(exact[name])
        left = total - sum(counts[name][slot] for name in counts)
        order = sorted(counts, key=lambda name: (exact[name] - int(exact[name]), name), reverse=True)
        while left > 0:
            ready = [name for name in order if counts[name][0] - sum(counts[name][1:]) >= 2]
            if not ready:
                break
            counts[ready[0]][slot] += 1
            order.remove(ready[0])
            left -= 1
    mapping = {}
    for name in sorted(counts):
        members = sorted(group for group in groups if strata[group] == name)
        random.Random(f"{seed}:{name}").shuffle(members)
        _, dev, test = counts[name]
        for index, group in enumerate(members):
            mapping[group] = "dev" if index < dev else "test" if index < dev + test else "train"
    summary = {name: {"groups": size, "train": size - dev - test, "dev": dev, "test": test}
               for name, (size, dev, test) in counts.items()}
    return mapping, summary


def freeze_splits(specs: list[dict], seed: int = 42,
                  dev_ratio: float = 0.15, test_ratio: float = 0.15, *,
                  stratify: bool = False) -> tuple[list[dict], dict]:
    """按来源组冻结划分。``stratify`` 让带 ``stratum`` 字段的来源组（难题池）按比例进入三个集合，
    而不是全部听凭一次打乱；只有一层时结果与不分层的划分完全相同。"""
    validate_specs(specs)
    reject_legacy_tasks(specs)
    if not 0 < dev_ratio < 1 or not 0 < test_ratio < 1 or dev_ratio + test_ratio >= 1:
        raise ValueError("dev/test ratios must be positive and sum to less than one")
    groups = sorted({spec["source_group"] for spec in specs})
    if len(groups) < 3:
        raise ValueError("at least three independent source groups are required")
    dev_count = max(1, int(len(groups) * dev_ratio))
    test_count = max(1, int(len(groups) * test_ratio))
    if dev_count + test_count >= len(groups):
        raise ValueError("not enough source groups for a nonempty train split")
    # 同一来源组的题必须在同一层：层和划分一样以来源组为单位
    strata = {}
    for spec in specs:
        stratum = _stratum_of(spec)
        group = spec["source_group"]
        if strata.setdefault(group, stratum) != stratum:
            raise ValueError(f"{spec['task_id']}: source group crosses strata: {group}")
    summary = None
    if stratify and len(set(strata.values())) > 1:
        mapping, summary = _stratified_mapping(groups, strata, seed, dev_count, test_count)
    else:
        random.Random(seed).shuffle(groups)
        mapping = {group: ("dev" if index < dev_count else
                           "test" if index < dev_count + test_count else "train")
                   for index, group in enumerate(groups)}
    frozen = [{**spec, "split": mapping[spec["source_group"]]} for spec in specs]
    hashes = {}
    for spec in frozen:
        fingerprint = code_fingerprint(spec["buggy"])
        if fingerprint in hashes and hashes[fingerprint] != spec["source_group"]:
            raise ValueError("duplicate task code assigned to different source groups")
        hashes[fingerprint] = spec["source_group"]
    manifest = {
        "schema_version": 1, "seed": seed, "dev_ratio": dev_ratio, "test_ratio": test_ratio,
        "source_groups": mapping, "tasks_sha256": content_hash(frozen),
        "task_counts": dict(Counter(spec["split"] for spec in frozen)),
        "task_ids": {split: sorted(spec["task_id"] for spec in frozen if spec["split"] == split)
                     for split in ("train", "dev", "test")},
        "limitation": "Source groups must track original repositories/templates. "
                      "AST matching does not detect all semantic duplicates.",
    }
    if summary:
        manifest["stratified"] = summary
    return frozen, manifest


def repair_edit_score(buggy: str, candidate: str) -> tuple[int, int]:
    def tokens(code):
        ignored = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE,
                   tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}
        return [token.string for token in tokenize.generate_tokens(io.StringIO(code).readline)
                if token.type not in ignored]

    before, after = tokens(buggy), tokens(candidate)
    edits = sum(max(before_end - before_start, after_end - after_start)
                for operation, before_start, before_end, after_start, after_end
                in difflib.SequenceMatcher(None, before, after, autojunk=False).get_opcodes()
                if operation != "equal")
    return edits, len(after)


def prepare_records(records: list[dict], specs: list[dict],
                    max_per_task: int = 1, *, replay=None,
                    revalidate_stale: bool = False) -> tuple[dict[str, list[dict]], dict]:
    validate_specs(specs)
    reject_legacy_tasks(specs)
    if max_per_task < 1:
        raise ValueError("max_per_task must be positive")
    if revalidate_stale and replay is None:
        raise ValueError("revalidate_stale requires grader replay")
    tasks = {spec["task_id"]: spec for spec in specs}
    outputs = {"train": [], "dev": []}
    seen = set()
    counts = Counter()
    rejected = Counter()
    prompt_splits = {}
    candidates = []
    candidate_counts = Counter()
    for record in records:
        if record.get("record_type") in {"error", "attempt"}:
            rejected["attempt_or_error"] += 1
            continue
        if record.get("provenance", {}).get("strict_pass") is not True:
            rejected["not_strict_pass"] += 1
            continue
        task_id = record.get("task_id")
        if task_id not in tasks:
            raise ValueError(f"unknown task in samples: {task_id}")
        spec = tasks[task_id]
        split = spec["split"]
        if split not in outputs:
            raise ValueError(f"{task_id}: test/smoke samples cannot enter SFT preparation")
        if record.get("split") != split or record.get("source_group") != spec["source_group"]:
            raise ValueError(f"{task_id}: sample metadata does not match frozen task split")
        stale = record.get("task_hash") != spec_fingerprint(spec)
        if stale and (not revalidate_stale or not record.get("task_hash")):
            raise ValueError(f"{task_id}: task hash is missing or stale; recollect with the frozen spec")
        messages = record.get("messages")
        if messages != build_messages(spec):
            raise ValueError(f"{task_id}: prompt differs from canonical public context")
        target = record.get("target", {})
        if target.get("role") != "assistant" or not isinstance(target.get("content"), str):
            raise ValueError(f"{task_id}: invalid assistant target")
        try:
            fingerprint = code_fingerprint(target["content"])
        except (SyntaxError, ValueError):
            rejected["invalid_python"] += 1
            continue
        if not ast.parse(target["content"]).body:
            rejected["empty_code"] += 1
            continue
        prompt_hash = content_hash(messages)
        if prompt_hash in prompt_splits and prompt_splits[prompt_hash] != split:
            raise ValueError("duplicate prompt crosses data splits")
        prompt_splits[prompt_hash] = split
        key = (task_id, fingerprint)
        if key in seen:
            rejected["duplicate_ast"] += 1
            continue
        seen.add(key)
        candidate_counts[task_id] += 1
        candidates.append((task_id, repair_edit_score(spec["buggy"], target["content"]),
                           fingerprint, record, stale))
    replayed = 0
    replay_rejections = []
    for task_id, edit_score, fingerprint, record, stale in sorted(candidates, key=lambda row: row[:3]):
        if counts[task_id] >= max_per_task:
            rejected["task_cap"] += 1
            continue
        spec = tasks[task_id]
        split = spec["split"]
        provenance = dict(record["provenance"])
        if replay is not None:
            replayed += 1
            try:
                verdict, passed, details = replay(spec, record["target"]["content"])
            except subprocess.TimeoutExpired:
                verdict, passed, details = "timeout", False, {}
            if not passed:
                rejected["replay_failed"] += 1
                replay_rejections.append({"task_id": task_id, "target_sha256": fingerprint,
                                          "verdict": verdict, "details": details})
                continue
            provenance["revalidated"] = True
            if stale:
                provenance["revalidated_from_task_hash"] = record["task_hash"]
        counts[task_id] += 1
        outputs[split].append({**record, "task_hash": spec_fingerprint(spec),
                               "provenance": provenance,
                               "sample_id": f"{task_id}-{fingerprint[:16]}",
                               "selection": {"policy": "minimum_token_edits_then_length",
                                             "edited_tokens": edit_score[0],
                                             "target_tokens": edit_score[1]}})
    for split, samples in outputs.items():
        if not samples:
            raise ValueError(f"{split}: no accepted samples; collect both train and dev")
    manifest = {
        "schema_version": 2, "tasks_sha256": content_hash(specs),
        "max_per_task": max_per_task, "rejected": dict(rejected),
        "selection_policy": "minimum_token_edits_then_length",
        "replayed_candidates": replayed, "replay_rejections": replay_rejections,
        "revalidated_stale_samples": sum(
            bool(sample["provenance"].get("revalidated_from_task_hash"))
            for samples in outputs.values() for sample in samples),
        "splits": {split: {
            "samples": len(samples), "sha256": content_hash(samples),
            "available_tasks": sum(spec["split"] == split for spec in specs),
            "accepted_tasks": len({sample["task_id"] for sample in samples}),
            "missing_task_ids": sorted(spec["task_id"] for spec in specs
                                       if spec["split"] == split and not counts[spec["task_id"]]),
            "samples_per_task": {spec["task_id"]: counts[spec["task_id"]]
                                 for spec in specs if spec["split"] == split},
            "unique_candidates_per_task": {spec["task_id"]: candidate_counts[spec["task_id"]]
                                           for spec in specs if spec["split"] == split},
            "task_ids": sorted({sample["task_id"] for sample in samples}),
            "source_groups": sorted({sample["source_group"] for sample in samples}),
            "by_bug_family": dict(Counter(tasks[sample["task_id"]]["bug_family"] for sample in samples)),
        } for split, samples in outputs.items()},
        "label_validation": ("all_exported_samples_replayed_strict_pass" if replay is not None
                             else "metadata_only_until_grader_replay"),
        "limitation": "Multiple solutions to one task do not increase independent task coverage. "
                      "Minimal edit ranking is a preservation heuristic, not a correctness proof.",
    }
    return outputs, manifest
