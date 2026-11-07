# -*- coding: utf-8 -*-
"""在线 RL 的 rollout → reward → advantage → loss 数学检查骨架。

离线模式：Mock 环境验证状态机、奖励路由、分组、mask、概率语义。
当前不执行 policy forward/backward、optimizer step 或权重同步。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from agentlab.rl.advantage import compute_group_advantage
from agentlab.rl.data import RolloutGroup, RolloutTrajectory
from agentlab.rl.loss import grpo_step


@dataclass
class RLConfig:
    group_size: int = 4
    clip_eps: float = 0.2
    kl_coef: float = 0.05
    max_steps: int = 10
    policy_version: str = "policy_v0"
    reference_version: str = "ref_v0"


@dataclass
class RLEpisodeLog:
    step: int
    group_id: str
    n_traj: int
    n_valid: int
    advantages: list[float]
    losses: list[float]
    total_cost: float | None = None
    known_cost_subtotal: float = 0.0
    unknown_cost_count: int = 0
    loss_trajectory_ids: list[str] = field(default_factory=list)
    logp_modes: list[str] = field(default_factory=list)
    n_action_tokens: int = 0
    zero_variance_group: bool = False
    execution_mode: str = field(default="math_check", init=False)
    update_applied: bool = field(default=False, init=False)


class RLRunner:
    """GRPO 风格采样与数学检查；train_step 保留名称兼容已有调用。"""

    def __init__(self, config: RLConfig,
                 rollout_fn: Callable[[str, int], list[RolloutTrajectory]],
                 reward_fn: Callable[[RolloutTrajectory], float | None]):
        """rollout_fn(task_id, group_size) -> 轨迹列表；reward_fn(traj) -> 终局 reward。"""
        self.config = config
        self.rollout_fn = rollout_fn
        self.reward_fn = reward_fn
        self.logs: list[RLEpisodeLog] = []

    def train_step(self, task_id: str, source_group: str, split: str,
                   checkpoint_id: str | None = None) -> RLEpisodeLog:
        """采样、验证来源、评分并计算所有轮次的 loss，不更新模型。"""
        group_id = f"grp-{task_id}-{len(self.logs)}"
        trajectories = self.rollout_fn(task_id, self.config.group_size)
        group = RolloutGroup(
            group_id=group_id, task_id=task_id, source_group=source_group,
            split=split, checkpoint_snapshot_id=checkpoint_id,
            policy_version=self.config.policy_version,
            trajectories=trajectories)
        for trajectory in group.trajectories:
            if trajectory.reference_version != self.config.reference_version:
                raise ValueError("reference_version mismatch")
            if trajectory.termination_reason != "infra_error":
                trajectory.validate_logp_shapes()
        for trajectory in group.trajectories:
            if trajectory.termination_reason != "infra_error":
                trajectory.reward = self.reward_fn(trajectory)
        adv = compute_group_advantage(group)
        losses = []
        loss_trajectory_ids = []
        logp_modes = []
        n_action_tokens = 0
        for index, trajectory in enumerate(group.trajectories):
            if trajectory.termination_reason == "infra_error" or trajectory.reward is None:
                continue
            identity_check = trajectory.new_logps is None
            new_logps = trajectory.old_logps if identity_check else trajectory.new_logps
            ref_logps = trajectory.old_logps if identity_check else trajectory.ref_logps
            result = grpo_step(
                new_logps=[value for turn in new_logps for value in turn],
                old_logps=[value for turn in trajectory.old_logps for value in turn],
                ref_logps=[value for turn in ref_logps for value in turn],
                advantage=adv.advantages[index],
                action_mask=[value for turn in trajectory.action_masks for value in turn],
                clip_eps=self.config.clip_eps,
                kl_coef=self.config.kl_coef,
                zero_variance_group=adv.zero_variance)
            losses.append(result.loss)
            loss_trajectory_ids.append(trajectory.traj_id)
            logp_modes.append("identity_logps" if identity_check else "provided_logps")
            n_action_tokens += result.n_action_tokens
        known_cost_subtotal = sum(
            trajectory.cost_usd for trajectory in trajectories
            if trajectory.cost_usd is not None)
        unknown_cost_count = sum(trajectory.cost_usd is None for trajectory in trajectories)
        log = RLEpisodeLog(
            step=len(self.logs), group_id=group_id,
            n_traj=len(trajectories), n_valid=adv.n_valid,
            advantages=adv.advantages, losses=losses,
            total_cost=None if unknown_cost_count else known_cost_subtotal,
            known_cost_subtotal=known_cost_subtotal,
            unknown_cost_count=unknown_cost_count,
            loss_trajectory_ids=loss_trajectory_ids,
            logp_modes=logp_modes,
            n_action_tokens=n_action_tokens,
            zero_variance_group=adv.zero_variance)
        self.logs.append(log)
        return log
