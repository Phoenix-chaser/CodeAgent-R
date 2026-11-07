# -*- coding: utf-8 -*-
"""Real single-GPU DPO trainer refining an existing SFT adapter.

Design decisions (mirroring hf_sft.py where possible):
- policy  = base model (NF4) + SFT adapter loaded with is_trainable=True;
- reference = the SAME SFT adapter, frozen and eval-only — DPO refines the
  SFT checkpoint rather than restarting from base;
- preference targets use the exact chat-template prefix subtraction, so
  chosen and rejected share a byte-identical supervised prefix boundary;
- per-epoch immutable checkpoints reuse save_epoch_checkpoint / manifests.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from agentlab.dpo.loss import dpo_loss, sequence_logp
from agentlab.training.hf_sft import (
    DatasetAudit, _as_list, environment_manifest, set_seed,
)


def _target_only(tokenizer: Any, messages: list[dict], target: dict) -> tuple[list[int], list[int]]:
    """Tokenize prompt+target once; labels supervise only the target tokens."""
    prefix = _as_list(tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors=None))
    full = _as_list(tokenizer.apply_chat_template(
        messages + [target], tokenize=True, add_generation_prompt=False,
        return_tensors=None))
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
            if sum(l != -100 for l in labels_c) == 0 or \
                    sum(l != -100 for l in labels_r) == 0:
                self.stats.skipped_invalid += 1
                continue
            self.stats.record_hashes.add(record_hash)
            self.stats.task_ids.add(str(record.get("task_id", "")))
            self.stats.source_groups.add(str(record.get("source_group", "")))
            self.stats.loaded += 1
            self.stats.total_tokens += len(ids_c) + len(ids_r)
            self.stats.supervised_tokens += (sum(l != -100 for l in labels_c)
                                             + sum(l != -100 for l in labels_r))
            self.samples.append({
                "input_ids_chosen": ids_c, "labels_chosen": labels_c,
                "input_ids_rejected": ids_r, "labels_rejected": labels_r,
                "pair_id": record.get("pair_id", f"line-{line_no}"),
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
        return {"input_ids_chosen": ids_c, "labels_chosen": labels_c,
                "attention_chosen": att_c, "input_ids_rejected": ids_r,
                "labels_rejected": labels_r, "attention_rejected": att_r}


def _logp_batch(model: torch.nn.Module, input_ids: torch.Tensor,
                labels: torch.Tensor, attention: torch.Tensor,
                device: torch.device, autocast: bool) -> torch.Tensor:
    """Sum supervised-token log prob per row of the batch."""
    context = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
               if autocast else torch.enable_grad())
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
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    revision = args.revision or None
    tokenizer = AutoTokenizer.from_pretrained(
        args.sft_adapter, trust_remote_code=False, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    def load_frozen_reference() -> torch.nn.Module:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=revision, quantization_config=bnb,
            device_map={"": 0}, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa", trust_remote_code=False)
        reference = PeftModel.from_pretrained(model, args.sft_adapter,
                                              is_trainable=False)
        reference.eval()
        reference.config.use_cache = True
        return reference

    reference = load_frozen_reference()
    policy_base = AutoModelForCausalLM.from_pretrained(
        args.model, revision=revision, quantization_config=bnb,
        device_map={"": 0}, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", trust_remote_code=False)
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
        "chat_template_sha256": hashlib.sha256(
            str(tokenizer.chat_template).encode()).hexdigest(),
        "configuration": vars(args),
        "data": data.stats.as_dict(),
        "sft_adapter": str(args.sft_adapter),
        "environment": environment_manifest(),
    }
    (output_dir / "run-config.json").write_text(
        json.dumps(initial, ensure_ascii=False, indent=2), encoding="utf-8")
    print("reference/policy 均为 NF4 + SFT adapter（policy 可训练）", flush=True)

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
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    ref_c = _logp_batch(reference, batch["input_ids_chosen"],
                                        batch["labels_chosen"],
                                        batch["attention_chosen"], device, True)
                    ref_r = _logp_batch(reference, batch["input_ids_rejected"],
                                        batch["labels_rejected"],
                                        batch["attention_rejected"], device, True)
                pol_c = _logp_batch(policy, batch["input_ids_chosen"],
                                    batch["labels_chosen"],
                                    batch["attention_chosen"], device, True)
                pol_r = _logp_batch(policy, batch["input_ids_rejected"],
                                    batch["labels_rejected"],
                                    batch["attention_rejected"], device, True)
                metrics = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=args.beta)
                loss = metrics["loss"].mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("DPO loss is not finite")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in policy.parameters() if p.requires_grad],
                    args.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                batches += 1
                loss_sum += float(loss)
                acc_sum += float((metrics["sigmoid_delta"] > 0.5).float().mean())
                margin_sum += float(metrics["reward_margin"].mean())
                entry = {
                    "step": global_step, "epoch": epoch,
                    "loss": float(loss), "beta": args.beta,
                    "reward_margin": float(metrics["reward_margin"].mean()),
                    "chosen_margin": float(metrics["chosen_margin"].mean()),
                    "rejected_margin": float(metrics["rejected_margin"].mean()),
                    "preference_accuracy": float(
                        (metrics["sigmoid_delta"] > 0.5).float().mean()),
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
                from copy import deepcopy
                snapshot = {**progress, "checkpoint_epoch": epoch,
                            "checkpoint_dev_loss": None}
                checkpoints.append(save_epoch_checkpoint(
                    policy, tokenizer, output_dir, snapshot, epoch,
                    global_step, float("nan")))
                write_checkpoint_index(output_dir, progress, checkpoints)
    final = {**initial, "status": "complete", "history": history,
             "optimizer_steps": global_step, "elapsed_s": time.time() - started,
             "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30}
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


__all__ = ["DPODataset", "DPOCollator", "train_dpo", "_target_only"]
