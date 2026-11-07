# -*- coding: utf-8 -*-
"""沙箱后端注册表：能力显式声明，不静默回退。

本机无 docker/podman（见 docs/p00-doctor.txt）→ container 后端
available=False；请求 container 执行时抛 UnsupportedSandbox，
绝不静默切到宿主进程执行（计划书：沙箱不可用时保留明确 unsupported）。
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
    container_cmd = next((c for c in ("docker", "podman") if shutil.which(c)), None)
    return {
        "process": SandboxBackend(
            name="process", available=True,
            evidence="宿主进程级隔离：独立工作区 + 子进程参数列表 + 进程组超时清理；"
                     "非容器隔离，边界弱于计划书目标，容器可用后升级"),
        "container": SandboxBackend(
            name="container", available=container_cmd is not None,
            evidence=f"container runtime probe: {container_cmd or 'docker/podman not found'}"
                     ""),
    }


def require_backend(name: str) -> SandboxBackend:
    backend = detect_backends().get(name)
    if backend is None:
        raise UnsupportedSandbox(f"unknown sandbox backend: {name}")
    if not backend.available:
        raise UnsupportedSandbox(
            f"sandbox backend '{name}' unavailable: {backend.evidence}")
    return backend
