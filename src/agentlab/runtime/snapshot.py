# -*- coding: utf-8 -*-
"""快照与分支恢复（文件状态快照，非进程检查点）。

契约（execution-state.md §3）：
- 记录 fixture_hash、workspace_manifest_hash、公开历史实际内容与哈希、
  环境 digest、已用/剩余预算、边界说明、父快照。
- 原子发布：先写临时目录再 rename，半写入快照不可见。
- 恢复前校验存储内容 hash 与记录一致，不一致拒绝。
- 分支 = 从同一快照恢复出的独立工作区，互不传播写入。
"""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path

from agentlab.contracts import ContractError
from agentlab.runtime.workspace import Workspace


def environment_digest() -> str:
    blob = f"{platform.platform()}|python={sys.version.split()[0]}"
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class Snapshot:
    def __init__(self, meta: dict, store_path: Path):
        self.meta = meta
        self.store_path = store_path        # 快照树所在目录

    @property
    def snapshot_id(self) -> str:
        return self.meta["snapshot_id"]

    @classmethod
    def create(cls, workspace: Workspace, task_id: str,
               public_prefix: list[dict], consumed_budget: dict,
               parent: "Snapshot | None" = None,
               store_dir: Path | None = None) -> "Snapshot":
        store = store_dir or workspace.root.parent / "snapshots"
        store.mkdir(parents=True, exist_ok=True)
        manifest = workspace.manifest_hash()
        prefix_blob = json.dumps(public_prefix, ensure_ascii=False,
                                 sort_keys=True).encode()
        prefix_hash = hashlib.sha256(prefix_blob).hexdigest()
        sid = hashlib.sha256(
            f"{manifest}|{prefix_hash}|{parent.snapshot_id if parent else ''}"
            .encode()).hexdigest()[:16]
        dest = store / sid
        if dest.exists():
            meta = json.loads((dest / "snapshot.json").read_text(encoding="utf-8"))
            if meta["workspace_manifest_hash"] != manifest:
                raise ContractError(f"snapshot id collision or corruption: {sid}")
            return cls(meta, dest)
        tmp = store / f".tmp-{sid}-{time.time_ns()}"
        tmp.mkdir()
        try:
            for item in workspace.root.iterdir():
                if item.name == ".pristine":
                    continue
                if item.is_dir():
                    shutil.copytree(item, tmp / item.name)
                else:
                    shutil.copy2(item, tmp / item.name)
            meta = {
                "schema_version": 1,
                "snapshot_id": sid,
                "parent_snapshot_id": parent.snapshot_id if parent else None,
                "task_id": task_id,
                "fixture_hash": Workspace(workspace.pristine).manifest_hash(),
                "workspace_manifest_hash": manifest,
                "public_prefix_ref": {"content": public_prefix,
                                      "sha256": prefix_hash},
                "environment_digest": environment_digest(),
                "consumed_budget": consumed_budget,
                "snapshot_boundary": "下一次 assistant 决策前，工具调用已全部完成，"
                                     "无后台写入者（文件快照，不含运行中进程）",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            (tmp / "snapshot.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.rename(dest)                  # 原子发布
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
        return cls(meta, dest)

    @staticmethod
    def _load_existing(dest: Path, sid: str, manifest: str,
                       prefix_hash: str) -> "Snapshot":
        meta = json.loads((dest / "snapshot.json").read_text(encoding="utf-8"))
        if meta["workspace_manifest_hash"] != manifest:
            raise ContractError(
                f"snapshot id collision or corruption: {sid}")
        return cls(meta, dest)

    def restore(self, new_root: Path) -> Workspace:
        """恢复到新工作区（分支）；恢复前校验存储树与记录哈希一致。

        哈希口径：快照存储树（排除 snapshot.json 元数据）必须等于
        快照创建时的工作区 manifest_hash——同一函数同一口径，无歧义。
        """
        if new_root.exists():
            raise ContractError(f"restore target exists: {new_root}")
        recorded = self.meta["workspace_manifest_hash"]
        actual = Workspace(self.store_path).manifest_hash(
            extra_exclude=("snapshot.json",))
        if actual != recorded:
            raise ContractError(
                f"snapshot content hash mismatch: recorded={recorded} actual={actual}")
        new_root.mkdir(parents=True)
        for item in self.store_path.iterdir():
            if item.name == "snapshot.json":
                continue
            if item.is_dir():
                shutil.copytree(item, new_root / item.name)
            else:
                shutil.copy2(item, new_root / item.name)
        return Workspace(new_root)
