# -*- coding: utf-8 -*-
"""沙箱后端注册表：能力显式声明，不静默回退。

后端状态只报告实际情况：

- ``landlock``：内核 Landlock（runtime/isolation.py）。可用时工作区只读、
  真实 HOME 不可读、TCP 全部拒绝；不可用时判分退化为 ``process`` 并在
  详情里如实记录（AGENTLAB_SANDBOX=required 时改为拒绝执行）。
- ``process``：始终可用。清洁环境 + rlimit + 进程组超时清理；没有 Landlock
  时文件系统与网络不受限制，同用户的 /proc/<pid>/environ 仍可读。
- ``container``：尚未实现 docker/podman 执行后端。即使本机装了容器运行时，
  这里也报告不可用——请求它时抛 UnsupportedSandbox，绝不静默回退。
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass


class UnsupportedSandbox(RuntimeError):
    """请求的沙箱后端不可用；调用方必须显式处理，不存在静默回退。"""


@dataclass(frozen=True)
class SandboxBackend:
    name: str
    available: bool
    evidence: str


def detect_backends() -> dict[str, SandboxBackend]:
    from agentlab.runtime.isolation import probe_landlock

    container_cmd = next((c for c in ("docker", "podman") if shutil.which(c)), None)
    landlock = probe_landlock()
    if landlock.available:
        landlock_evidence = (
            f"kernel Landlock ABI {landlock.landlock_abi}: workspace read-only, "
            "HOME and /proc unreachable"
            + (", all TCP connect/bind denied" if landlock.blocks_tcp
               else "; ABI < 4 does not restrict TCP"))
    else:
        landlock_evidence = f"unavailable: {landlock.reason}"
    return {
        "landlock": SandboxBackend(
            name="landlock", available=landlock.available, evidence=landlock_evidence),
        "process": SandboxBackend(
            name="process", available=True,
            evidence="host process isolation only: clean environment + rlimits + "
                     "process-group timeout kill; without Landlock the filesystem and "
                     "network are not restricted"),
        "container": SandboxBackend(
            name="container", available=False,
            evidence=f"not implemented (container runtime probe: {container_cmd or 'not found'})"),
    }


def require_backend(name: str) -> SandboxBackend:
    backend = detect_backends().get(name)
    if backend is None:
        raise UnsupportedSandbox(f"unknown sandbox backend: {name}")
    if not backend.available:
        raise UnsupportedSandbox(
            f"sandbox backend '{name}' unavailable: {backend.evidence}")
    return backend
