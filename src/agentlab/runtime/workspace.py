# -*- coding: utf-8 -*-
"""工作区：每个 episode 从不可变 fixture 创建的独立目录。

- 解包安全：拒绝绝对路径、`..` 逃逸、符号链接成员（fixture 攻击面）。
- 保留 .pristine 原始副本：同 episode 内合法修改保留，候选冻结时用于 diff。
- manifest_hash：确定性文件树哈希（排序路径 + 内容 sha256），快照/恢复校验用。
"""
from __future__ import annotations

import hashlib
import shutil
import tarfile
from pathlib import Path

from agentlab.contracts import ContractError


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


class Workspace:
    def __init__(self, root: Path):
        self.root = root
        self.pristine = root / ".pristine"

    @classmethod
    def create_from_fixture(cls, episode_id: str, fixture_tar: Path,
                            base_dir: Path) -> "Workspace":
        if not fixture_tar.exists():
            raise ContractError(f"fixture not found: {fixture_tar}")
        root = base_dir / f"ws-{episode_id}"
        if root.exists():
            raise ContractError(f"workspace already exists: {root}")
        root.mkdir(parents=True)
        cls._safe_extract(fixture_tar, root)
        # 原始副本（同一 episode 内的重试不重建工作区）
        pristine = root / ".pristine"
        pristine.mkdir()
        for item in root.iterdir():
            if item.name == ".pristine":
                continue
            shutil.copytree(item, pristine / item.name,
                            dirs_exist_ok=True) if item.is_dir() else \
                shutil.copy2(item, pristine / item.name)
        return cls(root)

    @staticmethod
    def _safe_extract(fixture_tar: Path, dest: Path) -> None:
        dest_resolved = dest.resolve()
        with tarfile.open(fixture_tar) as tar:
            for member in tar.getmembers():
                target = (dest / member.name).resolve()
                if not str(target).startswith(str(dest_resolved)):
                    raise ContractError(
                        f"unsafe fixture member escapes workspace: {member.name}")
                if member.issym() or member.islnk():
                    raise ContractError(
                        f"symlink members not allowed in fixture: {member.name}")
            tar.extractall(dest, filter="data")

    def manifest_hash(self, exclude_pristine: bool = True,
                      extra_exclude: tuple[str, ...] = ()) -> str:
        skip = ({".pristine", *extra_exclude} if exclude_pristine
                else set(extra_exclude))
        entries = []
        for p in sorted(self.root.rglob("*")):
            if any(part in skip for part in p.parts):
                continue
            rel = str(p.relative_to(self.root))
            if p.is_dir():
                entries.append(f"D {rel}")
            else:
                entries.append(f"F {rel} {_sha256_file(p)}")
        blob = "\n".join(entries).encode()
        return hashlib.sha256(blob).hexdigest()

    def write(self, rel_path: str, content: str) -> Path:
        target = (self.root / rel_path).resolve()
        if not str(target).startswith(str(self.root.resolve())):
            raise ContractError(f"write outside workspace: {rel_path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def read(self, rel_path: str) -> str:
        return (self.root / rel_path).read_text(encoding="utf-8")

    def diff_against_pristine(self, out_path: Path) -> Path:
        """候选冻结：标准 a/ b/ 可移植补丁（可在全新环境用 git apply 复应用）。

        实现：临时目录内 a/=原始副本、b/=当前工作区（均排除 .pristine），
        diff -ruN a b → 标签即为相对路径。不再依赖绝对路径标签。
        """
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            layout = Path(td)
            a, b = layout / "a", layout / "b"
            a.mkdir(); b.mkdir()
            for item in self.root.iterdir():
                if item.name == ".pristine":
                    continue
                if item.is_dir():
                    shutil.copytree(item, b / item.name)
                else:
                    shutil.copy2(item, b / item.name)
            for item in self.pristine.iterdir():
                if item.is_dir():
                    shutil.copytree(item, a / item.name)
                else:
                    shutil.copy2(item, a / item.name)
            proc = subprocess.run(["diff", "-ruN", "a", "b"], cwd=layout,
                                  capture_output=True, text=True, timeout=60)
            out_path.write_text(proc.stdout, encoding="utf-8")
        return out_path
