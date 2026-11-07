# -*- coding: utf-8 -*-
"""prompt 构造：运行时类型门保证只接受 PublicTask。"""
from __future__ import annotations

from agentlab.contracts.task import PublicTask
from agentlab.contracts.versioning import ContractError


def build_prompt(task: PublicTask, extra_context: str | None = None) -> str:
    if not isinstance(task, PublicTask):
        # PrivateTask 故意不继承 PublicTask；duck-typing 也进不来
        raise ContractError(
            f"build_prompt only accepts PublicTask, got {type(task).__name__}; "
            "use to_public() projection first")
    lines = [
        f"# Task {task.task_id} (v{task.task_version})",
        "",
        task.instruction,
        "",
        f"允许修改范围: {', '.join(task.allowed_paths) or '（见任务说明）'}",
        f"不可修改: {', '.join(task.protected_paths) or '无'}",
        f"预算: {task.budgets.max_model_requests or '∞'} 次模型请求, "
        f"{task.budgets.max_tool_calls or '∞'} 次工具调用, "
        f"{task.budgets.max_wall_seconds or '∞'} 秒",
    ]
    if task.public_check_profile:
        lines.append(f"公开检查: {task.public_check_profile}")
    if extra_context:
        lines += ["", "## 附加上下文", extra_context]
    return "\n".join(lines)
