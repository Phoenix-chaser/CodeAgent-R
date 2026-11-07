# -*- coding: utf-8 -*-
"""任务库：加载、公开/私有分离校验、split 完整性。

计划书 §4.3：observations/ 与 labels/ 物理分离——本模块强制：
- public.yaml + fixture.tar 在任务根（公开侧）；
- private/ 目录含私有测试/参考补丁/canary（标签侧）；
- canary 与任何私有材料不得出现在 fixture 内容中。
"""
from __future__ import annotations

import tarfile
from dataclasses import dataclass
from pathlib import Path

import yaml

from agentlab.contracts import ContractError, PublicTask, to_public
from agentlab.contracts.task import PrivateTask


@dataclass
class LoadedTask:
    public: PublicTask
    private: PrivateTask
    task_dir: Path
    fixture_tar: Path
    reference_patch: Path
    private_tests_dir: Path


class TaskStore:
    def __init__(self, root: Path):
        self.root = root
        self._cache: dict[str, LoadedTask] = {}

    def task_ids(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir()
                      if (p / "public.yaml").exists())

    def load(self, task_id: str) -> LoadedTask:
        if task_id in self._cache:
            return self._cache[task_id]
        td = self.root / task_id
        pub_yaml = td / "public.yaml"
        priv_yaml = td / "private" / "private.yaml"
        for p in (pub_yaml, priv_yaml):
            if not p.exists():
                raise ContractError(f"task manifest missing: {p}")
        pub = yaml.safe_load(pub_yaml.read_text(encoding="utf-8"))
        priv = yaml.safe_load(priv_yaml.read_text(encoding="utf-8"))
        # 显式映射构造（yaml 中的 grader_profile/strict_pass 等流程字段
        # 不属于任务契约，留在文件里供 grader/审计读取，不进模型）
        merged = dict(pub)
        merged.update(
            repository_id=priv.get("repository_id", f"selfbuilt:{pub['source_group']}"),
            repository_revision=priv.get("repository_revision", "fixture-v1"),
            fixture_artifact=str(td / "fixture.tar"),
            private_grader_id=priv.get("grader_profile", "default"),
            reference_patch_id=priv.get("reference_patch_id"),
            generator_family=priv.get("generator_family"),
            canary=priv["canary"],
        )
        try:
            private = PrivateTask.model_validate(merged)
        except Exception as e:
            raise ContractError(f"task {task_id}: invalid manifests: {e}") from e
        # 泄漏检查：canary 与私有文件内容不得出现在 fixture 内
        self._assert_no_private_leak(task_id, td, private.canary)
        loaded = LoadedTask(
            public=to_public(private),
            private=private,
            task_dir=td,
            fixture_tar=td / "fixture.tar",
            reference_patch=td / "private" / "reference.patch",
            private_tests_dir=td / "private" / "tests",
        )
        self._cache[task_id] = loaded
        return loaded

    def _assert_no_private_leak(self, task_id: str, td: Path, canary: str) -> None:
        fixture = td / "fixture.tar"
        if not fixture.exists():
            raise ContractError(f"fixture missing: {fixture}")
        with tarfile.open(fixture) as tar:
            for member in tar.getmembers():
                if member.name.startswith("private/") or "private" in member.name:
                    raise ContractError(
                        f"task {task_id}: private material inside fixture: {member.name}")
                if member.size and member.isfile():
                    content = tar.extractfile(member).read().decode("utf-8", "replace")
                    if canary in content:
                        raise ContractError(
                            f"task {task_id}: canary leaked into fixture member {member.name}")

    def split_integrity(self) -> dict[str, set[str]]:
        """source_group 不得跨 split（计划书 §5.5）。返回 split→groups 供审计。"""
        by_split: dict[str, set[str]] = {}
        for tid in self.task_ids():
            t = self.load(tid)
            by_split.setdefault(t.public.split, set()).add(t.public.source_group)
        overlap_violations = []
        splits = list(by_split.items())
        for i, (s1, g1) in enumerate(splits):
            for s2, g2 in splits[i + 1:]:
                overlap = g1 & g2
                if overlap:
                    overlap_violations.append(f"{sorted(overlap)} in both {s1} and {s2}")
        if overlap_violations:
            raise ContractError("split integrity violated: " + "; ".join(overlap_violations))
        return by_split
