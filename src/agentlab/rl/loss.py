# -*- coding: utf-8 -*-
"""GRPO 风格浮点数学检查，不执行梯度计算或参数更新。

公式：
  ratio = exp(new_logp - old_logp)          （仅动作 mask=1 的 token）
  clip_ratio = clamp(ratio, 1-ε, 1+ε)
  sampled_kl = exp(ref_logp - new_logp) - 1 - (ref_logp - new_logp)
  loss = -min(ratio * adv, clip_ratio * adv) + β_KL * sampled_kl

sampled_kl 的期望仅在 token 来自 new policy 时等于 KL(new || ref)。
旧策略样本上这里只报告非负采样惩罚，不宣称无偏 KL 或真实训练。

- 仅本次 rollout 新生成的 assistant token 参与（mask=1）；
- 恢复前历史 / 工具返回 / padding 全部 mask=0 不参与；
- reference 冻结（ref_logp 作为常数传入）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class GRPOStepResult:
    loss: float
    mean_ratio: float
    clip_fraction: float
    mean_kl: float
    n_action_tokens: int
    zero_variance_group: bool


def grpo_step(new_logps: list[float], old_logps: list[float],
              ref_logps: list[float], advantage: float,
              action_mask: list[int],
              clip_eps: float = 0.2, kl_coef: float = 0.05,
              zero_variance_group: bool = False) -> GRPOStepResult:
    """单轨迹的一步 GRPO loss（多 token 平均）。

    new/old/ref_logps 长度 == action_mask 长度；mask=0 的位置不参与。
    """
    if not (len(new_logps) == len(old_logps) == len(ref_logps) == len(action_mask)):
        raise ValueError("new/old/ref logps and action_mask must have equal lengths")
    if any(value not in (0, 1) for value in action_mask):
        raise ValueError("action_mask must contain only 0 or 1")
    if not math.isfinite(advantage):
        raise ValueError("advantage must be finite")
    if not math.isfinite(clip_eps) or not 0 <= clip_eps < 1:
        raise ValueError("clip_eps must be finite and in [0, 1)")
    if not math.isfinite(kl_coef) or kl_coef < 0:
        raise ValueError("kl_coef must be finite and nonnegative")
    n_action_tokens = sum(action_mask)
    if n_action_tokens == 0:
        return GRPOStepResult(loss=0.0, mean_ratio=1.0, clip_fraction=0.0,
                              mean_kl=0.0, n_action_tokens=0,
                              zero_variance_group=zero_variance_group)
    ratios, kls = [], []
    clipped = 0
    for new, old, ref, active in zip(new_logps, old_logps, ref_logps, action_mask):
        if active == 0:
            continue
        if not all(math.isfinite(value) for value in (new, old, ref)):
            raise ValueError("action log probabilities must be finite")
        try:
            ratio = math.exp(new - old)
            log_ratio = ref - new
            sampled_kl = max(0.0, math.expm1(log_ratio) - log_ratio)
        except OverflowError as error:
            raise ValueError("log probability difference overflows the math check") from error
        ratios.append(ratio)
        kls.append(sampled_kl)
        if ratio > 1 + clip_eps or ratio < 1 - clip_eps:
            clipped += 1
    # policy loss（裁剪）
    policy_losses = []
    for ratio in ratios:
        unclipped = ratio * advantage
        clipped_r = max(1 - clip_eps, min(1 + clip_eps, ratio)) * advantage
        policy_losses.append(-min(unclipped, clipped_r))
    mean_policy_loss = sum(policy_losses) / len(policy_losses)
    mean_kl = sum(kls) / len(kls)
    loss = mean_policy_loss + kl_coef * mean_kl
    return GRPOStepResult(
        loss=loss,
        mean_ratio=sum(ratios) / len(ratios),
        clip_fraction=clipped / len(ratios),
        mean_kl=mean_kl,
        n_action_tokens=n_action_tokens,
        zero_variance_group=zero_variance_group)


def micro_grpo_update(groups: list[dict],
                      clip_eps: float = 0.2, kl_coef: float = 0.05
                      ) -> list[GRPOStepResult]:
    """兼容入口：仅计算多组浮点 loss，不执行可微更新。

    groups: [{advantage, trajectories: [{new_logps, old_logps, ref_logps, action_mask}]}]
    """
    results = []
    for group in groups:
        for trajectory in group["trajectories"]:
            result = grpo_step(trajectory["new_logps"], trajectory["old_logps"],
                               trajectory["ref_logps"], group["advantage"],
                               trajectory["action_mask"], clip_eps, kl_coef)
            results.append(result)
    return results
