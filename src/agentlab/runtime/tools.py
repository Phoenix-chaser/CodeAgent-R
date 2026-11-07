# -*- coding: utf-8 -*-
"""统一工具语义：read / search / patch / command。

约束：相对路径且不得逃逸工作区；输出截断带标记；错误类型显式分类
（invalid_arguments / path_escape / timeout / nonzero_exit）。
诊断命令非零退出 ≠ 模型动作错误（调用方按 error_type 区分）。
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from agentlab.runtime.workspace import Workspace

MAX_TOOL_OUTPUT = 64_000


class ToolErrorType:
    INVALID_ARGUMENTS = "invalid_arguments"
    PATH_ESCAPE = "path_escape"
    TIMEOUT = "timeout"
    NONZERO_EXIT = "nonzero_exit"
    APPLY_FAILED = "apply_failed"


@dataclass
class ToolResult:
    ok: bool
    output: str = ""
    error_type: str | None = None
    truncated: bool = False


def _clip(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_TOOL_OUTPUT:
        return text, False
    return text[:MAX_TOOL_OUTPUT] + "\n[truncated]", True


def _safe_rel(ws: Workspace, rel: str) -> Path:
    target = (ws.root / rel).resolve()
    if not str(target).startswith(str(ws.root.resolve())):
        raise PermissionError(ToolErrorType.PATH_ESCAPE)
    return target


def tool_read(ws: Workspace, rel_path: str) -> ToolResult:
    if not rel_path or rel_path.startswith("/"):
        return ToolResult(False, error_type=ToolErrorType.INVALID_ARGUMENTS)
    try:
        target = _safe_rel(ws, rel_path)
    except PermissionError:
        return ToolResult(False, error_type=ToolErrorType.PATH_ESCAPE)
    if not target.is_file():
        return ToolResult(False, output="", error_type=ToolErrorType.NONZERO_EXIT)
    text, trunc = _clip(target.read_text(encoding="utf-8", errors="replace"))
    return ToolResult(True, output=text, truncated=trunc)


def tool_search(ws: Workspace, pattern: str) -> ToolResult:
    if not pattern:
        return ToolResult(False, error_type=ToolErrorType.INVALID_ARGUMENTS)
    proc = subprocess.run(["grep", "-rn", "--", pattern, "."],
                          cwd=ws.root, capture_output=True, text=True, timeout=30)
    text, trunc = _clip(proc.stdout)
    # grep 无匹配 exit 1：属诊断语义，不算工具错误
    ok = proc.returncode in (0, 1)
    return ToolResult(ok, output=text, truncated=trunc,
                      error_type=None if ok else ToolErrorType.NONZERO_EXIT)


def tool_patch(ws: Workspace, diff_text: str) -> ToolResult:
    if not diff_text.strip():
        return ToolResult(False, error_type=ToolErrorType.INVALID_ARGUMENTS)
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False) as f:
        f.write(diff_text)
        diff_path = f.name
    proc = subprocess.run(["git", "apply", "--whitespace=nowarn", diff_path],
                          cwd=ws.root, capture_output=True, text=True, timeout=30)
    ok = proc.returncode == 0
    return ToolResult(ok, output=proc.stderr[-500:] if not ok else "applied",
                      error_type=None if ok else ToolErrorType.APPLY_FAILED)


def tool_command(ws: Workspace, argv: list[str], timeout: float = 60.0) -> ToolResult:
    if not argv or not isinstance(argv, list):
        return ToolResult(False, error_type=ToolErrorType.INVALID_ARGUMENTS)
    try:
        proc = subprocess.run(argv, cwd=ws.root, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return ToolResult(False, error_type=ToolErrorType.TIMEOUT)
    text = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
    text, trunc = _clip(text)
    ok = proc.returncode == 0
    return ToolResult(ok, output=text, truncated=trunc,
                      error_type=None if ok else ToolErrorType.NONZERO_EXIT)
