# -*- coding: utf-8 -*-
"""Shared, auditable benchmark/prompt protocol.

The model sees only public task material.  Private tests and reference patches
are used by the grader but are deliberately excluded from the prompt.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from agentlab.benchmark.builder import SPECS


def build_messages(spec: dict[str, Any], *, include_public_test: bool = True) -> list[dict[str, str]]:
    """Build the canonical prompt for collection, training and evaluation."""
    public_files = [
        f"当前有缺陷的文件 {spec['module']}：\n\n{spec['buggy']}"
    ]
    for name, content in sorted((spec.get("extra_files") or {}).items()):
        public_files.append(f"相关依赖文件 {name}：\n\n{content}")
    if include_public_test and spec.get("public_test"):
        public_files.append(f"公开测试（仅用于理解接口）：\n\n{spec['public_test']}")
    user = f"{spec['instruction']}\n\n" + "\n\n".join(public_files)
    return [
        {"role": "system", "content": (
            "你是代码修复专家。只输出需要修改的主文件的完整 Python 代码，"
            "不要解释、不要 markdown 代码块标记、不要注释说明。"
        )},
        {"role": "user", "content": user},
    ]


def spec_fingerprint(spec: dict[str, Any]) -> str:
    payload = json.dumps(spec, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def prompt_fingerprint(spec: dict[str, Any]) -> str:
    payload = json.dumps(build_messages(spec), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def protocol_hash(specs: list[dict[str, Any]]) -> str:
    payload = [{"task_id": spec["task_id"], "spec": spec_fingerprint(spec),
                "prompt": prompt_fingerprint(spec)}
               for spec in sorted(specs, key=lambda item: item["task_id"])]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_specs(tasks_file: str | Path | None = None,
               split: str | None = None) -> list[dict[str, Any]]:
    """Load task specs from JSON/JSONL or the repository's built-in specs."""
    if tasks_file is None:
        specs = [{**spec, "split": "smoke"} for spec in SPECS]
    else:
        path = Path(tasks_file)
        if not path.exists():
            raise FileNotFoundError(path)
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            specs = []
        else:
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                specs = [json.loads(line) for line in text.splitlines() if line.strip()]
            else:
                specs = (data.get("tasks", [data]) if isinstance(data, dict)
                         else data)
    validate_specs(specs)
    if split and split != "all":
        specs = [s for s in specs if s.get("split") == split]
    if not specs:
        raise ValueError(f"no task specs found for split={split!r}")
    return specs


def validate_specs(specs: list[dict[str, Any]]) -> None:
    """Validate the trusted benchmark input before creating files."""
    if not isinstance(specs, list) or not specs:
        raise ValueError("task specs must be a nonempty list")
    required = {"task_id", "source_group", "split", "bug_family", "difficulty",
                "instruction", "module", "buggy", "fixed", "public_test",
                "private_tests"}
    seen = set()
    groups = {}
    for spec in specs:
        if not isinstance(spec, dict) or required - spec.keys():
            raise ValueError(f"task spec missing required fields: {required}")
        task_id = spec["task_id"]
        if not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", task_id):
            raise ValueError("task_id must be a safe directory name")
        if task_id in seen:
            raise ValueError(f"duplicate task_id: {task_id}")
        seen.add(task_id)
        if spec["split"] not in {"train", "dev", "test", "ood", "smoke"}:
            raise ValueError(f"invalid split for {task_id}")
        group = spec["source_group"]
        if not isinstance(group, str) or not group.strip():
            raise ValueError(f"missing source_group for {task_id}")
        if group in groups and groups[group] != spec["split"]:
            raise ValueError(f"source_group crosses splits: {group}")
        groups[group] = spec["split"]
        extra_files = spec.get("extra_files", {})
        private_tests = spec["private_tests"]
        if not isinstance(extra_files, dict) or not isinstance(private_tests, dict) or not private_tests:
            raise ValueError(f"files/private_tests must be mappings: {task_id}")
        paths = [spec["module"], *extra_files, *private_tests]
        if any(not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_]+\.py", name)
               for name in paths):
            raise ValueError(f"only flat Python file names supported: {task_id}")
        if spec["module"] in extra_files:
            raise ValueError(f"extra_files overwrites target module: {task_id}")
        if not all(name.startswith("test_") for name in private_tests):
            raise ValueError(f"private test files must start with test_: {task_id}")
        for field in ("instruction", "buggy", "fixed", "public_test", "bug_family", "difficulty"):
            if not isinstance(spec[field], str) or not spec[field].strip():
                raise ValueError(f"empty {field}: {task_id}")
        if not all(isinstance(content, str) and content.strip()
                   for content in [*extra_files.values(), *private_tests.values()]):
            raise ValueError(f"empty file contents: {task_id}")
