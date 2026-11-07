# -*- coding: utf-8 -*-
"""最小 CLI。

P00 只提供两个真实可用的命令：
  agentlab --help     帮助
  agentlab doctor     环境探测：只报告实际观察，未具备/未探测的能力保留 unknown

不注册尚未实现的 run/train/benchmark 命令（后续工作包按验收逐个开放）。
"""
from __future__ import annotations

import argparse
import importlib.util
import platform
import shutil
import subprocess
import sys

from agentlab import __version__


def _probe_optional_module(name: str) -> str:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return "not installed"
    try:
        mod = importlib.import_module(name)
        return getattr(mod, "__version__", "installed")
    except Exception as e:  # 导入失败也如实报告
        return f"import error: {type(e).__name__}"


def _probe_command(cmd: str) -> str:
    path = shutil.which(cmd)
    return path if path else "not found"


def _probe_container_runtime() -> str:
    for cmd in ("docker", "podman"):
        path = shutil.which(cmd)
        if path:
            try:
                out = subprocess.run([cmd, "--version"], capture_output=True,
                                     text=True, timeout=10)
                return f"{cmd}: {out.stdout.strip() or 'unknown version'}"
            except Exception:
                return f"{cmd}: found, version unknown"
    return "not found"


def _probe_gpu() -> str:
    nvidia = shutil.which("nvidia-smi")
    if not nvidia:
        return "none (no nvidia-smi)"
    try:
        out = subprocess.run([nvidia], capture_output=True, text=True, timeout=10)
        first = next((ln for ln in out.stdout.splitlines() if "NVIDIA" in ln), None)
        return first.strip() if first else "nvidia-smi present, model unknown"
    except Exception:
        return "nvidia-smi present, probe failed"


def doctor() -> int:
    """环境探测：只输出实际观察，不做任何能力假设。"""
    rows = [
        ("python", sys.version.split()[0]),
        ("platform", platform.platform()),
        ("agentlab", __version__),
        ("pydantic", _probe_optional_module("pydantic")),
        ("yaml", _probe_optional_module("yaml")),
        ("pytest", _probe_optional_module("pytest")),
        ("torch", _probe_optional_module("torch")),
        ("transformers", _probe_optional_module("transformers")),
        ("container_runtime", _probe_container_runtime()),
        ("gpu", _probe_gpu()),
        ("pi", _probe_command("pi")),
        ("node", _probe_command("node")),
        ("git", _probe_command("git")),
        # 以下能力在本环境无法探测，保留 unknown，不猜测
        ("model_revision", "unknown"),
        ("api_credentials", "unknown (not probed)"),
    ]
    width = max(len(k) for k, _ in rows)
    for key, val in rows:
        print(f"{key:<{width}}  {val}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agentlab",
        description="CodeAgent-R 评测与后训练基础设施（离线验证阶段）。"
                    "当前仅提供 doctor；训练与评测命令待资源解锁后逐个开放。",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("doctor", help="环境探测（只报告实际观察，未知保留 unknown）")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return doctor()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
