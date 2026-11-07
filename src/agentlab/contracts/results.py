# -*- coding: utf-8 -*-
"""运行结果与判分契约。

核心原则（P01 验收）：**运行器退出成功 ≠ 评分通过**。
RunResult 根本不含任何 pass/评分字段——评分只能由 grader 产出
GradeResult；GradeResult.strict_pass 默认 False，显式判 strict_pass
必须携带 grader_id 与一致 verdict。
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import field_validator, model_validator

from agentlab.contracts.events import Usage
from agentlab.contracts.versioning import ContractError, VersionedModel

ExitStatus = Literal["ok", "error", "timeout", "budget_exceeded", "aborted"]
Verdict = Literal["strict_pass", "partial", "strict_fail", "invalid"]


class CostAmount(VersionedModel):
    """费用；usd=None 表示未知（unknown）。任何聚合不得把 None 当 0。"""
    usd: float | None = None

    @classmethod
    def unknown(cls) -> "CostAmount":
        return cls(usd=None)


class ArtifactRef(VersionedModel):
    path: str
    kind: Literal["workspace_diff", "log", "report", "other"] = "other"
    sha256: str | None = None


class RunResult(VersionedModel):
    """一次 episode 的执行侧结果。注意：没有 pass 字段——判分在 GradeResult。"""
    episode_id: str
    task_id: str
    exit_status: ExitStatus
    usage: Usage | None = None
    cost: CostAmount = CostAmount.unknown()
    artifacts: list[ArtifactRef] = []
    events_summary: dict[str, Any] = {}
    started_at: str | None = None
    finished_at: str | None = None


class GradeResult(VersionedModel):
    """grader 产出的判分。运行退出成功不会被自动转成 strict_pass。"""
    episode_id: str
    grader_id: str
    graded_at: str
    verdict: Verdict
    strict_pass: bool = False
    details: dict[str, Any] = {}

    @model_validator(mode="after")
    def _verdict_consistency(self):
        if self.strict_pass != (self.verdict == "strict_pass"):
            raise ContractError(
                "strict_pass must equal (verdict == 'strict_pass')")
        if self.strict_pass and not self.grader_id:
            raise ContractError("strict_pass requires a grader_id")
        return self
