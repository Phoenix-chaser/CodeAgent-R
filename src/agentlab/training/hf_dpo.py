# -*- coding: utf-8 -*-
"""Real single-GPU DPO trainer refining an existing SFT adapter.

Design decisions (mirroring hf_sft.py where possible):
- policy  = base model (NF4) + SFT adapter loaded with is_trainable=True;
- reference = the Base model by default (standard DPO), or the adapter named by
  --reference-adapter; either way it is frozen and eval-only;
- per-pair β chosen from the sampling group's outcome (pair_objectives) is this
  project's own heuristic. It is not a method from any cited paper;
- preference targets use the exact chat-template prefix subtraction, so
  chosen and rejected share a byte-identical supervised prefix boundary;
- per-epoch immutable checkpoints reuse save_epoch_checkpoint / manifests.

Related work: DPO (https://arxiv.org/abs/2305.18290). FEST
(https://arxiv.org/abs/2605.15012) is few-shot-guided RLVR; it also groups
tasks by whether the model can solve them, which is where the group_solvable
label comes from, but it does not propose a per-group β.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from agentlab.dpo.loss import dpo_loss, sequence_logp
from agentlab.training.hf_runtime import (
    apply_chat_template, load_causal_lm, load_tokenizer, resolve_revision,
)
from agentlab.training.hf_sft import (
    DatasetAudit, _as_list, environment_manifest, parameter_delta, set_seed,
)


def _target_only(tokenizer: Any, messages: list[dict], target: dict) -> tuple[list[int], list[int]]:
    """Tokenize prompt+target once; labels supervise only the target tokens."""
    prefix = _as_list(apply_chat_template(tokenizer, messages, add_generation_prompt=True))
    full = _as_list(apply_chat_template(
        tokenizer, messages + [target], add_generation_prompt=False))
    if len(full) < len(prefix) or full[:len(prefix)] != prefix:
        raise ValueError("chat template prefix mismatch for preference target")
    return full, [-100] * len(prefix) + full[len(prefix):]


class DPODataset(Dataset):
    """Preference JSONL: shared messages + chosen/rejected assistant targets."""

    def __init__(self, data_path: str | Path, tokenizer: Any,
                 max_length: int = 2048, split: str = "train"):
        self.data_path = str(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.samples: list[dict[str, Any]] = []
        self.stats = DatasetAudit(self.data_path, split)
        self.stats.file_sha256 = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
        self._load()

    def _load(self) -> None:
        for line_no, line in enumerate(open(self.data_path, encoding="utf-8"), 1):
            if not line.strip():
                continue
            self.stats.total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                self.stats.skipped_json += 1
                continue
            messages, chosen, rejected = (record.get("messages"),
                                          record.get("chosen"),
                                          record.get("rejected"))
            ok = (isinstance(messages, list) and messages
                  and isinstance(chosen, dict) and isinstance(rejected, dict)
                  and chosen.get("role") == "assistant"
                  and rejected.get("role") == "assistant")
            if not ok or record.get("split") != self.split:
                self.stats.skipped_invalid += 1
                continue
            record_hash = hashlib.sha256(json.dumps(
                {"m": messages, "c": chosen, "r": rejected},
                ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            if record_hash in self.stats.record_hashes:
                self.stats.skipped_duplicate += 1
                continue
            try:
                ids_c, labels_c = _target_only(self.tokenizer, messages, chosen)
                ids_r, labels_r = _target_only(self.tokenizer, messages, rejected)
            except (TypeError, ValueError):
                self.stats.skipped_invalid += 1
                continue
            if max(len(ids_c), len(ids_r)) > self.max_length:
                self.stats.skipped_long += 1
                continue
            if sum(label != -100 for label in labels_c) == 0 or \
                    sum(label != -100 for label in labels_r) == 0:
                self.stats.skipped_invalid += 1
                continue
            self.stats.record_hashes.add(record_hash)
            self.stats.task_ids.add(str(record.get("task_id", "")))
            self.stats.source_groups.add(str(record.get("source_group", "")))
            self.stats.loaded += 1
            self.stats.total_tokens += len(ids_c) + len(ids_r)
            self.stats.supervised_tokens += (sum(label != -100 for label in labels_c)
                                             + sum(label != -100 for label in labels_r))
            self.samples.append({
                "input_ids_chosen": ids_c, "labels_chosen": labels_c,
                "input_ids_rejected": ids_r, "labels_rejected": labels_r,
                "pair_id": record.get("pair_id", f"line-{line_no}"),
                "group_solvable": record.get("group_solvable"),
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]

    def report(self) -> str:
        s = self.stats
        return (f"偏好对统计({self.split}): 总行数={s.total}, 加载={s.loaded}, "
                f"重复={s.skipped_duplicate}, 超长={s.skipped_long}, "
                f"非法/非本split={s.skipped_invalid}, "
                f"监督比例={s.supervised_ratio:.1%}")


class DPOCollator:
    """Pad chosen/rejected jointly; both keep exact -100 masks."""

    def __init__(self, pad_token_id: int):
        self.pad_id = int(pad_token_id)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(max(len(item["input_ids_chosen"]),
                          len(item["input_ids_rejected"])) for item in batch)

        def pad_rows(ids_key: str, labels_key: str):
            inputs, labels, attention = [], [], []
            for item in batch:
                n = len(item[ids_key])
                pad = max_len - n
                inputs.append(item[ids_key] + [self.pad_id] * pad)
                labels.append(item[labels_key] + [-100] * pad)
                attention.append([1] * n + [0] * pad)
            return (torch.tensor(inputs, dtype=torch.long),
                    torch.tensor(labels, dtype=torch.long),
                    torch.tensor(attention, dtype=torch.long))

        ids_c, labels_c, att_c = pad_rows("input_ids_chosen", "labels_chosen")
        ids_r, labels_r, att_r = pad_rows("input_ids_rejected", "labels_rejected")
        # group_solvable → 自适应 β 标记（None=旧格式，默认用 solvable β）
        solvable = torch.tensor(
            [0 if item.get("group_solvable") is False or
             str(item.get("group_solvable")).lower() == "false" else 1
             for item in batch], dtype=torch.long)
        return {"input_ids_chosen": ids_c, "labels_chosen": labels_c,
                "attention_chosen": att_c, "input_ids_rejected": ids_r,
                "labels_rejected": labels_r, "attention_rejected": att_r,
                "group_solvable": solvable}


def _logp_batch(model: torch.nn.Module, input_ids: torch.Tensor,
                labels: torch.Tensor, attention: torch.Tensor,
                device: torch.device, autocast: bool) -> torch.Tensor:
    """Sum supervised-token log prob per row of the batch."""
    # autocast 参数目前不起作用：原实现创建了 autocast 上下文却没有进入，等于从未
    # 开启。为不改变已记录的 DPO 轮次的数值，这里保持该行为（bnb 量化层自带计算
    # 精度，其余层按权重 dtype 运行）。
    del autocast
    outputs = model(input_ids=input_ids.to(device),
                    attention_mask=attention.to(device))
    logits = outputs.logits.float()
    logps = []
    for row in range(logits.shape[0]):
        logps.append(sequence_logp(logits[row], labels[row].to(device)))
    return torch.stack(logps)


def _capture_lora_state(model: Any) -> dict:
    return {name: parameter.detach().clone()
            for name, parameter in model.named_parameters() if parameter.requires_grad}


def pair_objectives(logp_chosen: torch.Tensor, logp_rejected: torch.Tensor,
                    ref_chosen: torch.Tensor, ref_rejected: torch.Tensor,
                    betas: list[float]) -> dict[str, torch.Tensor]:
    """One DPO loss per preference pair (each with its own β), averaged over the batch."""
    if not (len(betas) == logp_chosen.shape[0] == logp_rejected.shape[0]
            == ref_chosen.shape[0] == ref_rejected.shape[0]):
        raise ValueError("one β and one log-probability per preference pair are required")
    terms = [dpo_loss(logp_chosen[i:i + 1], logp_rejected[i:i + 1],
                      ref_chosen[i:i + 1], ref_rejected[i:i + 1], beta=beta)
             for i, beta in enumerate(betas)]
    return {
        "loss": torch.stack([term["loss"] for term in terms]).mean(),
        "preference_accuracy": torch.stack(
            [term["sigmoid_delta"].reshape(()) for term in terms]).gt(0.5).float().mean(),
        "reward_margin": torch.stack(
            [term["reward_margin"].reshape(()) for term in terms]).mean(),
    }


def dpo_step(policy: torch.nn.Module, reference: torch.nn.Module,
             batch: dict[str, torch.Tensor], betas: list[float], *, optimizer,
             device: torch.device, max_grad_norm: float = 1.0,
             autocast: bool = True) -> dict[str, float]:
    """在一个 batch 的偏好对上做一次 optimizer step，返回写日志用的指标。

    reference 只参与前向，不产生梯度。训练循环本身需要 CUDA，这里把每一步
    单独拆出来，是为了能在 CPU 上验证。
    """
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        ref_c = _logp_batch(reference, batch["input_ids_chosen"],
                            batch["labels_chosen"], batch["attention_chosen"],
                            device, autocast)
        ref_r = _logp_batch(reference, batch["input_ids_rejected"],
                            batch["labels_rejected"], batch["attention_rejected"],
                            device, autocast)
    pol_c = _logp_batch(policy, batch["input_ids_chosen"], batch["labels_chosen"],
                        batch["attention_chosen"], device, autocast)
    pol_r = _logp_batch(policy, batch["input_ids_rejected"], batch["labels_rejected"],
                        batch["attention_rejected"], device, autocast)
    metrics = pair_objectives(pol_c, pol_r, ref_c, ref_r, betas)
    loss = metrics["loss"]
    if not torch.isfinite(loss):
        raise FloatingPointError("DPO loss is not finite")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in policy.parameters() if p.requires_grad],
        max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {"loss": float(loss.detach()), "reward_margin": float(metrics["reward_margin"]),
            "preference_accuracy": float(metrics["preference_accuracy"])}


def train_dpo(args: Any) -> dict[str, Any]:
    """Run DPO refinement of an SFT adapter. Requires CUDA + bf16."""
    for name in ("beta", "lr"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be > 0")
    for name in ("epochs", "batch_size", "max_length"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be >= 1")
    if not getattr(args, "sft_adapter", None):
        raise ValueError("DPO refines an SFT adapter: --sft-adapter is required")
    set_seed(args.seed)

    revision = resolve_revision(args.model, args.revision)
    tokenizer = load_tokenizer(args.sft_adapter, padding_side="right")
    data = DPODataset(args.data, tokenizer, args.max_length, split="train")
    if not len(data):
        raise ValueError(f"preference set is empty: {data.report()}")
    print(data.report(), flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("DPO training requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("this configuration needs bf16 support")
    from peft import PeftModel

    device = torch.device("cuda:0")
    output_dir = Path(args.output)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output directory must be empty; use a new --output")
    output_dir.mkdir(parents=True, exist_ok=True)

    def load_frozen_reference() -> torch.nn.Module:
        """Reference = Base model (no adapter), per standard DPO.

        Using Base as reference creates real gradient signal: the SFT
        policy has shifted teacher-answer probabilities up relative to
        Base, while failure probabilities shifted less → margin > 0.
        Using SFT as reference (policy ≡ reference at init) yields
        σ→0 vanishing gradients when rejected comes from the same SFT.
        """
        model = load_causal_lm(args.model, revision)
        if args.reference_adapter:
            model = PeftModel.from_pretrained(
                model, args.reference_adapter, is_trainable=False)
        model.eval()
        model.config.use_cache = True
        return model

    reference = load_frozen_reference()
    policy_base = load_causal_lm(args.model, revision)
    from peft import prepare_model_for_kbit_training
    policy_base = prepare_model_for_kbit_training(
        policy_base, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    policy_base.config.use_cache = False
    policy = PeftModel.from_pretrained(policy_base, args.sft_adapter,
                                       is_trainable=True)
    if any(p.requires_grad and "lora_" not in n
           for n, p in policy.named_parameters()):
        raise RuntimeError("base weights are not frozen")
    before = _capture_lora_state(policy)
    policy.print_trainable_parameters()

    manifest = json.loads((Path(args.sft_adapter).parent / "training-manifest.json")
                          .read_text(encoding="utf-8")) \
        if (Path(args.sft_adapter).parent / "training-manifest.json").exists() else {}
    if manifest.get("model") not in (None, args.model):
        raise ValueError("SFT adapter was trained on a different base model")

    collator = DPOCollator(tokenizer.pad_token_id)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True,
                        generator=generator, collate_fn=collator, num_workers=0)
    steps_per_epoch = math.ceil(len(loader))
    total_steps = steps_per_epoch * args.epochs
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.0)

    initial = {
        "schema_version": 1, "status": "initialized",
        "kind": "dpo", "model": args.model, "revision": revision,
        "resolved_revision": getattr(
            policy_base.config, "_commit_hash", None),
        "best_epoch": args.epochs,
        "chat_template_sha256": hashlib.sha256(
            str(tokenizer.chat_template).encode()).hexdigest(),
        "configuration": vars(args),
        "data": data.stats.as_dict(),
        "sft_adapter": str(args.sft_adapter),
        "environment": environment_manifest(),
    }
    (output_dir / "run-config.json").write_text(
        json.dumps(initial, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"reference: NF4 Base{' + ' + args.reference_adapter if args.reference_adapter else ''}"
          "（冻结）; policy: NF4 Base + SFT adapter（可训练）", flush=True)

    from agentlab.training.hf_sft import (save_epoch_checkpoint,
                                           write_checkpoint_index)
    checkpoints: list[dict] = []
    history: list[dict] = []
    global_step = 0
    started = time.time()
    log_path = output_dir / "training-log.jsonl"
    policy.train()
    with log_path.open("w", encoding="utf-8") as log_file:
        for epoch in range(1, args.epochs + 1):
            loss_sum = acc_sum = margin_sum = 0.0
            batches = 0
            for batch in loader:
                # 按组自适应 β（本项目启发式）：整组采样都失败的题用 β_unsolvable，
                # 学生能解但这次失败的题用 β_solvable
                solvable = batch["group_solvable"].tolist()
                betas = ([args.beta_solvable if flag else args.beta_unsolvable
                          for flag in solvable]
                         if hasattr(args, "beta_solvable") else [args.beta] * len(solvable))
                step = dpo_step(policy, reference, batch, betas, optimizer=optimizer,
                                device=device, max_grad_norm=args.max_grad_norm)
                global_step += 1
                batches += 1
                loss_sum += step["loss"]
                acc_sum += step["preference_accuracy"]
                margin_sum += step["reward_margin"]
                entry = {
                    "step": global_step, "epoch": epoch, **step,
                    "beta_mean": sum(betas) / len(betas),
                    "beta_unsolvable": args.beta_unsolvable if hasattr(args, "beta_unsolvable") else args.beta,
                    "beta_solvable": args.beta_solvable if hasattr(args, "beta_solvable") else args.beta,
                    "elapsed_s": time.time() - started,
                }
                log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                log_file.flush()
                print(f"[dpo] step {global_step}/{total_steps} "
                      f"loss={entry['loss']:.4f} acc={entry['preference_accuracy']:.2f} "
                      f"margin={entry['reward_margin']:.3f}", flush=True)
            history.append({"epoch": epoch, "loss": loss_sum / max(batches, 1),
                            "preference_accuracy": acc_sum / max(batches, 1),
                            "reward_margin": margin_sum / max(batches, 1)})
            progress = {**initial, "status": "checkpoint_complete",
                        "history": history, "optimizer_steps": global_step,
                        "elapsed_s": time.time() - started}
            (output_dir / "training-manifest.json").write_text(
                json.dumps(progress, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8")
            if args.save_every_epoch:
                snapshot = {**progress, "checkpoint_epoch": epoch,
                            "checkpoint_dev_loss": None}
                checkpoints.append(save_epoch_checkpoint(
                    policy, tokenizer, output_dir, snapshot, epoch,
                    global_step, float("nan")))
                write_checkpoint_index(output_dir, progress, checkpoints)
    delta = parameter_delta(before, policy)
    if delta["changed_parameter_tensors"] == 0:
        print("警告：DPO 结束但没有可训练参数发生变化，请检查梯度或学习率", flush=True)
    final = {**initial, "status": "complete", "history": history,
             "optimizer_steps": global_step, "elapsed_s": time.time() - started,
             "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
             "parameter_delta": delta}
    (output_dir / "training-manifest.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    if not args.save_every_epoch:
        checkpoints.append(save_epoch_checkpoint(
            policy, tokenizer, output_dir, {**final, "checkpoint_epoch": args.epochs},
            args.epochs, global_step, float("nan")))
        write_checkpoint_index(output_dir, final, checkpoints)
    print(f"DPO 完成：{output_dir}（manifest 已写入）", flush=True)
    return final


__all__ = ["DPODataset", "DPOCollator", "train_dpo", "_target_only", "pair_objectives",
           "dpo_step"]
