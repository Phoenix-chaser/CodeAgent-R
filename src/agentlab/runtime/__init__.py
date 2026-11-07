# -*- coding: utf-8 -*-
"""P02 运行时层：沙箱注册、工作区、受控执行、预算准入、快照、Mock 运行器。"""
from agentlab.runtime.budget_guard import BudgetExceeded, BudgetGuard
from agentlab.runtime.exec import ExecOutcome, run_command
from agentlab.runtime.mock_runtime import MockRuntime, MockRunOutput, MockStep
from agentlab.runtime.sandbox import (
    SandboxBackend, UnsupportedSandbox, detect_backends, require_backend,
)
from agentlab.runtime.snapshot import Snapshot, environment_digest
from agentlab.runtime.workspace import Workspace

__all__ = [
    "SandboxBackend", "UnsupportedSandbox", "detect_backends", "require_backend",
    "Workspace", "run_command", "ExecOutcome",
    "BudgetGuard", "BudgetExceeded",
    "Snapshot", "environment_digest",
    "MockRuntime", "MockStep", "MockRunOutput",
]
