# -*- coding: utf-8 -*-
"""标准动作级 DPO 损失（implementation.md §4 公式实现）。

loss = -logsigmoid(beta * ((logp_c - ref_c) - (logp_r - ref_r)))

- logp 为完整候选 assistant token 的条件 log prob 之和（标准求和，不做长度归一化）；
- 共同 prefix / padding / 工具 observation 全部 mask（-100）；
- reference 冻结（不更新参数）；
- 长度归一化变体单独命名，不与标准版混淆。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sequence_logp(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """完整序列中监督 token 的条件 log prob 之和（labels=-100 跳过）。

    返回 0-dim tensor（保持梯度流，不调 .item()）。
    """
    shift_logits = logits[:-1]
    shift_labels = labels[1:]
    mask = shift_labels != -100
    if mask.sum() == 0:
        return logits.sum() * 0.0
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logp = log_probs.gather(1, shift_labels.clamp(min=0).unsqueeze(1)).squeeze(1)
    return token_logp[mask].sum()


def dpo_loss(logp_chosen, logp_rejected,
             ref_logp_chosen, ref_logp_rejected,
             beta: float = 0.1) -> dict:
    """标准 DPO loss（支持 float 或 torch.Tensor 输入）。

    loss = -log σ(β * ((logp_c - ref_c) - (logp_r - ref_r)))
    tensor 输入时保留 policy 梯度，reference log prob 始终作为常数。
    """
    if isinstance(ref_logp_chosen, torch.Tensor):
        ref_logp_chosen = ref_logp_chosen.detach()
    if isinstance(ref_logp_rejected, torch.Tensor):
        ref_logp_rejected = ref_logp_rejected.detach()
    if isinstance(logp_chosen, torch.Tensor):
        chosen_margin = logp_chosen - ref_logp_chosen
        rejected_margin = logp_rejected - ref_logp_rejected
        delta = beta * (chosen_margin - rejected_margin)
        loss = -F.logsigmoid(delta)
        return dict(loss=loss,
                    chosen_margin=chosen_margin.detach(),
                    rejected_margin=rejected_margin.detach(),
                    reward_margin=delta.detach(),
                    sigmoid_delta=torch.sigmoid(delta).detach())
    else:
        import math
        chosen_margin = logp_chosen - ref_logp_chosen
        rejected_margin = logp_rejected - ref_logp_rejected
        delta = beta * (chosen_margin - rejected_margin)
        sig = 1.0 / (1.0 + math.exp(-delta)) if abs(delta) < 700 else \
            (1.0 if delta > 0 else 0.0)
        loss = max(-delta, 0.0) + math.log1p(math.exp(-abs(delta)))
        return dict(loss=loss, chosen_margin=chosen_margin,
                    rejected_margin=rejected_margin,
                    reward_margin=delta, sigmoid_delta=sig)


def dpo_loss_length_normalized(logp_chosen, logp_rejected,
                               ref_chosen, ref_rejected,
                               n_chosen_tokens: int, n_rejected_tokens: int,
                               beta: float = 0.1) -> dict:
    """长度归一化变体——单独命名，不与标准版混淆。"""
    cn = max(n_chosen_tokens, 1)
    rn = max(n_rejected_tokens, 1)
    result = dpo_loss(logp_chosen / cn, logp_rejected / rn,
                      ref_chosen / cn, ref_rejected / rn, beta)
    return dict(loss=result["loss"], reward_margin=result["reward_margin"],
                variant="length_normalized_v1")


def micro_dpo_step(pairs: list[dict], beta: float = 0.1,
                   lr: float = 0.01) -> dict:
    """微型可微 DPO 更新（CPU, toy 模型）：含真实 optimizer step。

    pairs: [{input_ids, labels, ref_logp}]
    验证 loss 梯度与参数更新；reference 为外部提供的固定 log prob，
    此微型步骤不包含 reference 模型或真实语言模型训练。
    """
    if len(pairs) < 2:
        raise ValueError("DPO step requires >= 2 sequences")
    chosen, rejected = pairs[0], pairs[1]
    vocab = max(max(chosen["input_ids"]), max(rejected["input_ids"])) + 8
    emb = torch.nn.Embedding(vocab, 16)
    head = torch.nn.Linear(16, vocab)
    params_before = emb.weight.data.clone()

    optimizer = torch.optim.AdamW(list(emb.parameters()) + list(head.parameters()),
                                  lr=lr)

    def logp(seq):
        ids = torch.tensor(seq["input_ids"])
        labels = torch.tensor(seq["labels"])
        logits = head(emb(ids))
        return sequence_logp(logits, labels)     # tensor，保持梯度

    logp_c = logp(chosen)
    logp_r = logp(rejected)
    reference_logps = [
        sequence["ref_logp"].detach()
        if isinstance(sequence["ref_logp"], torch.Tensor)
        else sequence["ref_logp"]
        for sequence in (chosen, rejected)
    ]
    result = dpo_loss(logp_c, logp_r, *reference_logps, beta=beta)
    # loss 是 tensor → backward
    loss_tensor = result["loss"]
    optimizer.zero_grad()
    loss_tensor.backward()
    result["grad_norm"] = sum(p.grad.norm().item() for p in emb.parameters()
                              if p.grad is not None)
    optimizer.step()
    params_after = emb.weight.data
    result["params_changed"] = not torch.equal(params_before, params_after)
    result["reference_frozen"] = all(
        not isinstance(value, torch.Tensor) or not value.requires_grad
        for value in reference_logps
    )
    result["loss_value"] = loss_tensor.item()
    result["policy_logp"] = dict(chosen=logp_c.item(), rejected=logp_r.item())
    return result
