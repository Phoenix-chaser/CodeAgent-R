# -*- coding: utf-8 -*-
"""EpisodeConfig：一次独立完整运行的冻结配置（计划书 §4.2）。

episode = 模型在任务上的一次独立完整运行，从原始 fixture 开始；
重复 episode 必须重建环境；attempt 递增不清零账本。
"""
from __future__ import annotations

from typing import Literal

from agentlab.contracts.task import Budgets
from agentlab.contracts.versioning import VersionedModel

RuntimeKind = Literal["mock", "controlled_model", "pi"]


class EpisodeConfig(VersionedModel):
    episode_id: str
    task_id: str
    task_version: int = 1
    runtime: RuntimeKind = "mock"
    attempt: int = 1                 # 外层重试编号；预算不随 attempt 清零
    seed: int | None = None
    budgets: Budgets                 # 运行时实际生效的预算快照（冻结于创建时）
