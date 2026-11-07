# -*- coding: utf-8 -*-
"""GRPO 风格组内 advantage 与零方差诊断（implementation.md §6）。"""
from __future__ import annotations

import math
from dataclasses import dataclass

from agentlab.rl.data import RolloutGroup


@dataclass
class AdvantageResult:
    advantages: list[float]
    group_mean: float
    group_std: float
    zero_variance: bool
    n_valid: int
    n_total: int
    n_infra: int
    note: str = ""


def compute_group_advantage(group: RolloutGroup,
                            eps: float = 1e-8) -> AdvantageResult:
    """组内 centering + std 归一化（GRPO 风格）。

    全组同 reward → advantage 全零（零方差组）——记录比例，不冒称有效学习。
    infra_error 轨迹排除在分母外。
    """
    group.validate()
    valid = [t for t in group.trajectories if t.termination_reason != "infra_error"]
    rewards = [t.reward for t in valid if t.reward is not None]
    n_infra = len(group.trajectories) - len(valid)
    if not rewards:
        return AdvantageResult(
            advantages=[0.0] * len(group.trajectories), group_mean=0.0, group_std=0.0,
            zero_variance=True, n_valid=0,
            n_total=len(group.trajectories), n_infra=n_infra,
            note="no scored trajectories")
    if any(not math.isfinite(reward) for reward in rewards):
        raise ValueError("scored rewards must be finite")
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    zero_var = std < eps
    if zero_var:
        adv = [0.0] * len(rewards)
        note = f"zero variance: all {len(rewards)} rewards = {mean}"
    else:
        adv = [(r - mean) / (std + eps) for r in rewards]
        note = ""
    # 把 advantage 映射回原始索引（含 infra 的位为 0）
    full_adv = [0.0] * len(group.trajectories)
    vi = 0
    for i, t in enumerate(group.trajectories):
        if t.termination_reason != "infra_error" and t.reward is not None:
            full_adv[i] = adv[vi]
            vi += 1
    return AdvantageResult(
        advantages=full_adv, group_mean=mean, group_std=std,
        zero_variance=zero_var, n_valid=len(rewards),
        n_total=len(group.trajectories), n_infra=n_infra, note=note)
