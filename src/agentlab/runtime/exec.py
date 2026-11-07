# -*- coding: utf-8 -*-
"""受控执行：参数列表子进程（无 shell）+ 进程组超时清理。

超时语义：kill 整个进程组（start_new_session）并等待退出，
保证超时后没有写入者继续改工作区文件。
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

MAX_OUTPUT_BYTES = 1 << 20  # 1MiB 截断上限


@dataclass
class ExecOutcome:
    exit_status: str            # ok | nonzero_exit | timeout | spawn_error
    returncode: int | None
    stdout: str
    stderr: str
    duration_ms: int
    stdout_truncated: bool = False


def run_command(argv: list[str], cwd: Path, timeout: float,
                env: dict | None = None) -> ExecOutcome:
    if not argv:
        raise ValueError("argv must be non-empty")
    start = time.monotonic()
    try:
        proc = subprocess.Popen(
            argv, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=env, start_new_session=True)
    except OSError as e:
        return ExecOutcome(exit_status="spawn_error", returncode=None,
                           stdout="", stderr=str(e),
                           duration_ms=int((time.monotonic() - start) * 1000))
    try:
        out, err = proc.communicate(timeout=timeout)
        truncated = False
        if len(out.encode()) > MAX_OUTPUT_BYTES:
            out = out[:MAX_OUTPUT_BYTES // 2]
            truncated = True
        return ExecOutcome(
            exit_status="ok" if proc.returncode == 0 else "nonzero_exit",
            returncode=proc.returncode, stdout=out, stderr=err,
            duration_ms=int((time.monotonic() - start) * 1000),
            stdout_truncated=truncated)
    except subprocess.TimeoutExpired:
        # 杀整个进程组并等待——不留后台写入者
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        out, err = proc.communicate()
        return ExecOutcome(
            exit_status="timeout", returncode=proc.returncode,
            stdout=(out or "")[:MAX_OUTPUT_BYTES // 2],
            stderr=(err or "")[:MAX_OUTPUT_BYTES // 2],
            duration_ms=int((time.monotonic() - start) * 1000))
