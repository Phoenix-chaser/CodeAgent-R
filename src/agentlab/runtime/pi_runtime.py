# -*- coding: utf-8 -*-
"""PiRuntime：仅通过已核实的 subprocess/JSON 协议外驱 pi。

能力核实（runtime-protocol.md §2）：capability registry 逐项
verified / unsupported / unknown + 证据；不猜测旗标与能力。
本机 pi 未安装 → 全部 unknown（evidence: pi not found），
PiRuntime.run 拒绝执行（不猜协议、不静默换实现）。
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from agentlab.runtime.protocol import StreamAssembler, StreamResult


@dataclass
class Capability:
    name: str
    status: str            # verified | unsupported | unknown
    evidence: str


def probe_pi_capabilities(pi_path: str | None = None) -> dict[str, Capability]:
    found = pi_path or shutil.which("pi")
    if not found:
        note = "pi executable not found"
        return {cap: Capability(cap, "unknown", note) for cap in (
            "subprocess_json", "model_selection", "usage_reporting",
            "cancellation", "session_resume", "arbitrary_pause")}
    caps: dict[str, Capability] = {}
    try:
        proc = subprocess.run([found, "--version"], capture_output=True,
                              text=True, timeout=15)
        version = (proc.stdout or proc.stderr).strip()[:120]
        caps["subprocess_json"] = Capability(
            "subprocess_json", "unknown",
            f"pi found ({version}); JSON 旗标尚未对照该版本文档核实")
    except Exception as e:
        caps["subprocess_json"] = Capability(
            "subprocess_json", "unknown", f"probe failed: {e}")
    for cap in ("model_selection", "usage_reporting", "cancellation",
                "session_resume", "arbitrary_pause"):
        caps[cap] = Capability(cap, "unknown",
                               "pending verification against installed pi docs")
    return caps


class PiUnsupported(RuntimeError):
    """所需能力未核实（unknown/unsupported）——拒绝执行，不猜协议。"""


class PiRuntime:
    name = "pi"

    def __init__(self, pi_path: str | None = None):
        self.capabilities = probe_pi_capabilities(pi_path)

    def require(self, *cap_names: str) -> None:
        for name in cap_names:
            cap = self.capabilities.get(name)
            if cap is None or cap.status != "verified":
                status = cap.status if cap else "missing"
                raise PiUnsupported(
                    f"pi capability '{name}' is {status}: "
                    f"{cap.evidence if cap else 'not probed'}")

    def run(self, episode_id: str, argv: list[str], cwd: Path,
            timeout: float) -> StreamResult:
        """外驱 pi：需 subprocess_json 已 verified 才执行。"""
        self.require("subprocess_json")
        proc = subprocess.run(argv, cwd=str(cwd), capture_output=True,
                              text=True, timeout=timeout)
        asm = StreamAssembler(episode_id)
        for line in (proc.stdout or "").splitlines():
            asm.feed(line)
        exit_status = ("ok" if proc.returncode == 0 else "nonzero_exit")
        return asm.finish(exit_status, proc.stderr or "")
