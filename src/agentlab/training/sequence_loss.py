# -*- coding: utf-8 -*-
"""分段算因果语言模型的损失：不在显存里一次性放下整段序列的全部 logits。

transformers 的做法是先算出 [序列长度, 词表] 的全部 logits，再转成 fp32 做交叉熵，
反向传播时还有同样大小的梯度。词表 24.8 万、序列 2048 时，每份拷贝约 2 GiB，几份同时
存在，24 GB 的卡在长样本上就会显存不足。这里主干只过一次，lm_head 和交叉熵只对被监督的
位置分段计算，每一段用 gradient checkpointing 包起来（反向时重算），所以任何时刻显存里
只有一段 logits。结果与模型自带的损失相同，差别在浮点误差以内；``compare_losses`` 在一段
输入上量出这个差，训练开始前会先核对。
"""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

IGNORE_INDEX = -100
CHUNK = 256


def split_lm(model: nn.Module) -> tuple[nn.Module, nn.Module]:
    """取出 (主干, lm_head)。LoRA 包装过的模型先解到底层模型。"""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    backbone, lm_head = getattr(base, "model", None), getattr(base, "lm_head", None)
    if not isinstance(backbone, nn.Module) or not isinstance(lm_head, nn.Module):
        raise ValueError("这个模型没有 model（主干）和 lm_head 两段结构，没法分段算损失；"
                         "请改用 --no-chunked-loss")
    return backbone, lm_head


def _piece(lm_head: nn.Module, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = lm_head(hidden).float()                     # 与 transformers 一样，在 fp32 里算交叉熵
    return nn.functional.cross_entropy(logits, targets, reduction="sum")


def sequence_loss(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor | None,
                  labels: torch.Tensor, *, chunk: int = CHUNK,
                  recompute: bool = True) -> torch.Tensor:
    """与 ``model(..., labels=labels).loss`` 相同：对被监督的 token 求平均的交叉熵。

    第 p 个位置的隐状态预测第 p+1 个 token。``recompute`` 为真时每一段在反向时重算，
    前向不保留 logits；只做推断（no_grad）时用不着。
    """
    if chunk < 1:
        raise ValueError("chunk must be positive")
    backbone, lm_head = split_lm(model)
    hidden = backbone(input_ids=input_ids, attention_mask=attention_mask)[0]
    supervised = labels[:, 1:] != IGNORE_INDEX
    predictors = hidden[:, :-1][supervised]              # 只留下一个 token 被监督的位置
    targets = labels[:, 1:][supervised]
    if not targets.numel():
        return hidden.sum() * 0.0
    total = None
    for start in range(0, targets.shape[0], chunk):
        piece = (predictors[start:start + chunk], targets[start:start + chunk])
        if recompute and torch.is_grad_enabled() and predictors.requires_grad:
            term = checkpoint(_piece, lm_head, *piece, use_reentrant=False)
        else:
            term = _piece(lm_head, *piece)
        total = term if total is None else total + term
    return total / targets.numel()


def compare_losses(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor | None,
                   labels: torch.Tensor) -> dict:
    """在一段输入上比较模型自带的损失和分段损失。输入要短：自带的做法会放下全部 logits。"""
    reference = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
    chunked = sequence_loss(model, input_ids, attention_mask, labels, recompute=False)
    return {"reference": float(reference.detach()), "chunked": float(chunked.detach()),
            "difference": abs(float(reference.detach()) - float(chunked.detach()))}


def consistent(check: dict) -> bool:
    """两者相差不到千分之一（或绝对值不到 0.001）就算一致。错位、漏算位置带来的差远大于此。"""
    return check["difference"] <= max(1e-3, 1e-3 * abs(check["reference"]))
