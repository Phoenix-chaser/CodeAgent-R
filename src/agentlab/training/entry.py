# -*- coding: utf-8 -*-
"""SFT 离线验证入口。

offline：数据→collate→mask 检查→微型 forward/backward（torch CPU）→
         导出 manifest；不下载模型、不产生能力结论。
真实模型加载、tokenizer 适配和 QLoRA 训练循环尚未实现。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from agentlab.training.collator import SFTCollator
from agentlab.training.data import SFTSample, export_manifest


def micro_forward_backward(batch) -> dict:
    """微型可微检查：验证损失只来自监督 token、分母正确、梯度可回传。"""
    import torch
    import torch.nn.functional as F
    all_ids, all_labels = [], []
    for s in batch.samples:
        all_ids += s.input_ids
        all_labels += s.labels
    ids = torch.tensor(all_ids)
    labels = torch.tensor(all_labels)
    vocab = int(ids.max().item()) + 8
    emb = torch.nn.Embedding(vocab, 16)
    head = torch.nn.Linear(16, vocab)
    logits = head(emb(ids))
    # causal shift：预测下一个 token
    shift_logits = logits[:-1]
    shift_labels = labels[1:]
    loss = F.cross_entropy(shift_logits, shift_labels,
                           ignore_index=-100, reduction="sum")
    n_target = (shift_labels != -100).sum().item()
    mean_loss = (loss / max(n_target, 1)).item()
    loss.backward()
    grad_ok = emb.weight.grad is not None and emb.weight.grad.abs().sum().item() > 0
    return dict(n_target_tokens=int(n_target), mean_loss=mean_loss,
                grad_flow=grad_ok)


def export_checkpoint_manifest(out_dir: Path, batch, micro: dict,
                               provenance: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        schema_version=1,
        level="offline_micro_validated",       # 不冒充真实训练
        samples=[s.sample_id for s in batch.samples],
        rejected=batch.rejected,
        total_tokens=batch.total_tokens,
        supervised_tokens=batch.supervised_tokens,
        supervision_ratio=(batch.supervised_tokens / batch.total_tokens
                           if batch.total_tokens else 0),
        micro_check=micro,
        provenance=provenance,
        content_hash=hashlib.sha256(
            json.dumps([s.sample_id for s in batch.samples]).encode()
        ).hexdigest()[:16])
    path = out_dir / "checkpoint-manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return path


def run_offline_validation(samples: list[SFTSample], out_dir: Path,
                           collator: SFTCollator,
                           provenance: dict) -> dict:
    batch = collator.collate(samples)
    micro = micro_forward_backward(batch) if batch.samples else {}
    manifest_path = export_checkpoint_manifest(out_dir, batch, micro,
                                               provenance)
    export_manifest(samples, out_dir / "dataset-manifest.json")
    return dict(batch=batch, micro=micro, manifest=manifest_path)
