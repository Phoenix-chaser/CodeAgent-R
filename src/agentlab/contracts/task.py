# -*- coding: utf-8 -*-
"""任务契约：PublicTask（模型可见）与 PrivateTask（可信端私有）。

计划书 §4.3：observations/ 与 labels/ 物理分离。PublicTask 只含模型
可见信息；fixture、私有判分器、参考补丁、canary 只存在于 PrivateTask，
经 to_public() 投影后才可进入 prompt 构造（prompt.py 有运行时类型门）。
"""
from __future__ import annotations

from typing import Literal

from pydantic import Field

from agentlab.contracts.versioning import ContractError, VersionedModel

Split = Literal["train", "dev", "test", "ood", "smoke"]


class Budgets(VersionedModel):
    """任务预算上限；None 表示该维度不限（区别于 0=禁止）。"""
    max_model_requests: int | None = None
    max_tool_calls: int | None = None
    max_wall_seconds: int | None = None


class PublicTask(VersionedModel):
    """模型可见的任务信息——prompt 构造的唯一合法输入类型。"""
    task_id: str
    task_version: int = 1
    source_group: str
    split: Split
    bug_family: str
    difficulty: str                      # L1–L4（§5.3），字符串以兼容扩展
    instruction: str
    allowed_paths: list[str] = Field(default_factory=list)
    protected_paths: list[str] = Field(default_factory=list)
    public_check_profile: str | None = None
    budgets: Budgets = Field(default_factory=Budgets)
    metadata: dict[str, str] = Field(default_factory=dict)


class PrivateTask(VersionedModel):
    """可信端完整任务定义。含 fixture、私有判分与 canary（防泄漏探针）。

    注意：故意不继承 PublicTask——类型隔离保证它无法被直接传给
    build_prompt（运行时 isinstance 门会拒绝）。
    """
    # 公开部分（与 PublicTask 字段一一对应）
    task_id: str
    task_version: int = 1
    source_group: str
    split: Split
    bug_family: str
    difficulty: str
    instruction: str
    allowed_paths: list[str] = Field(default_factory=list)
    protected_paths: list[str] = Field(default_factory=list)
    public_check_profile: str | None = None
    budgets: Budgets = Field(default_factory=Budgets)
    metadata: dict[str, str] = Field(default_factory=dict)
    # 私有部分（绝不进入 prompt）
    repository_id: str
    repository_revision: str
    fixture_artifact: str
    environment_digest: str | None = None
    private_grader_id: str
    reference_patch_id: str | None = None
    generator_family: str | None = None
    canary: str = Field(description="防泄漏探针：随机串，仅存在于私有侧；"
                                    "若出现在任何 prompt/observation 即为泄漏事故")


def to_public(private: PrivateTask) -> PublicTask:
    """私有任务 → 公开投影：只保留 PublicTask 字段。"""
    data = private.to_contract_dict()
    public_fields = set(PublicTask.model_fields.keys())
    projected = {k: v for k, v in data.items() if k in public_fields}
    try:
        return PublicTask.from_contract_dict(projected)
    except ContractError as e:
        raise ContractError(f"to_public projection failed: {e}") from e
