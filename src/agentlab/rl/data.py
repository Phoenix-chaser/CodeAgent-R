# -*- coding: utf-8 -*-
"""在线 RL rollout 数据契约与分组。

规则（implementation.md §2）：
- 每个组绑定同一 task/checkpoint、相同环境/可见历史/剩余预算/policy version；
- 每条轨迹独立可写工作区；
- 记录 policy/reference/tokenizer 版本、逐 token 动作 mask、old-policy logp；
- 不能把不同题或不同恢复状态凑成组来制造 reward 方差。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from agentlab.contracts import Usage

POLICY_VERSION = "policy_v0"


@dataclass
class RolloutTrajectory:
    """一条完整的多轮轨迹（一个 rollout）。"""
    traj_id: str
    group_id: str
    task_id: str
    source_group: str
    split: str
    episode_id: str
    checkpoint_snapshot_id: str | None      # None = 从头开始
    policy_version: str = POLICY_VERSION
    reference_version: str = "ref_v0"
    messages: list[dict] = field(default_factory=list)
    # 逐 token 动作 mask：仅本次生成的 assistant token 为 1
    action_masks: list[list[int]] = field(default_factory=list)
    # 逐 token old-policy logp（采样时保存）
    old_logps: list[list[float]] = field(default_factory=list)
    new_logps: list[list[float]] | None = None
    ref_logps: list[list[float]] | None = None
    # 终局
    termination_reason: str = ""             # done|truncated|infra_error|budget_exceeded
    reward: float | None = None              # 终局严格成功 reward（1/0），None=未评分
    candidate_sha256: str = ""
    grader_ref: str = ""
    # 成本
    usage: Usage | None = None
    cost_usd: float | None = None
    # 时间戳
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S%z"))

    def validate_logp_shapes(self) -> None:
        if (self.new_logps is None) != (self.ref_logps is None):
            raise ValueError("new_logps and ref_logps must be supplied together")
        for name in ("old_logps", "new_logps", "ref_logps"):
            turns = getattr(self, name)
            if turns is None:
                continue
            if len(turns) != len(self.action_masks):
                raise ValueError(f"{name} turn count does not match action_masks")
            for turn_index, (values, mask) in enumerate(zip(turns, self.action_masks)):
                if len(values) != len(mask):
                    raise ValueError(
                        f"{name} token count does not match action_masks at turn {turn_index}")


@dataclass
class RolloutGroup:
    """同任务/同状态的多条轨迹（GRPO 分组）。"""
    group_id: str
    task_id: str
    source_group: str
    split: str
    checkpoint_snapshot_id: str | None
    policy_version: str
    trajectories: list[RolloutTrajectory] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.trajectories = list(self.trajectories)
        self.validate()

    def validate(self) -> None:
        for trajectory in self.trajectories:
            self._validate_trajectory(trajectory)

    def _validate_trajectory(self, trajectory: RolloutTrajectory) -> None:
        for name in ("task_id", "source_group", "split", "policy_version",
                     "checkpoint_snapshot_id"):
            expected = getattr(self, name)
            actual = getattr(trajectory, name)
            if actual != expected:
                raise ValueError(f"{name} mismatch: group={expected!r} traj={actual!r}")

    def add(self, traj: RolloutTrajectory) -> None:
        self._validate_trajectory(traj)
        self.trajectories.append(traj)

    @property
    def rewards(self) -> list[float | None]:
        return [t.reward for t in self.trajectories]

    @property
    def n_valid(self) -> int:
        """排除 infra_error 后的有效轨迹数。"""
        return sum(1 for t in self.trajectories
                   if t.termination_reason != "infra_error")
