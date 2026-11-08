# -*- coding: utf-8 -*-
"""Small, auditable Hugging Face QLoRA SFT pipeline.

The project originally had a one-off training loop in ``scripts/train_sft.py``.
This module keeps the GPU-specific bits thin and puts the decisions that affect
the result (data filtering, chat-template boundaries, weighted loss and split
checks) in testable helpers.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import random
import shutil
import sys
import time
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch
from torch.utils.data import DataLoader, Dataset


def _as_list(ids: Any) -> list[int]:
    """Convert tokenizer output (list/tensor/nested list) into token ids."""
    if isinstance(ids, Mapping):
        ids = ids["input_ids"]
    if hasattr(ids, "detach"):
        ids = ids.detach().cpu().tolist()
    while isinstance(ids, (list, tuple)) and len(ids) == 1 and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return list(ids)


def _hash_record(record: dict) -> str:
    payload = {
        "messages": record.get("messages", []),
        "target": record.get("target", {}),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class DatasetAudit:
    path: str
    expected_split: str
    total: int = 0
    loaded: int = 0
    skipped_json: int = 0
    skipped_split: int = 0
    skipped_fail: int = 0
    skipped_duplicate: int = 0
    skipped_long: int = 0
    skipped_invalid: int = 0
    total_tokens: int = 0
    supervised_tokens: int = 0
    task_ids: set[str] | None = None
    source_groups: set[str] | None = None
    record_hashes: set[str] | None = None
    prompt_hashes: set[str] | None = None
    file_sha256: str = ""

    def __post_init__(self) -> None:
        self.task_ids = set() if self.task_ids is None else self.task_ids
        self.source_groups = set() if self.source_groups is None else self.source_groups
        self.record_hashes = set() if self.record_hashes is None else self.record_hashes
        self.prompt_hashes = set() if self.prompt_hashes is None else self.prompt_hashes

    @property
    def supervised_ratio(self) -> float:
        return self.supervised_tokens / self.total_tokens if self.total_tokens else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "expected_split": self.expected_split,
            "file_sha256": self.file_sha256,
            "total": self.total,
            "loaded": self.loaded,
            "skipped_json": self.skipped_json,
            "skipped_split": self.skipped_split,
            "skipped_fail": self.skipped_fail,
            "skipped_duplicate": self.skipped_duplicate,
            "skipped_long": self.skipped_long,
            "skipped_invalid": self.skipped_invalid,
            "total_tokens": self.total_tokens,
            "supervised_tokens": self.supervised_tokens,
            "supervised_ratio": round(self.supervised_ratio, 6),
            "task_ids": sorted(self.task_ids or set()),
            "source_groups": sorted(self.source_groups or set()),
            "prompt_hashes": sorted(self.prompt_hashes or set()),
            "content_hash": hashlib.sha256(
                "\n".join(sorted(self.record_hashes or set())).encode()
            ).hexdigest()[:16],
        }


class SFTDataset(Dataset):
    """JSONL dataset with exact chat-template assistant-only labels.

    A response is obtained by tokenizing the *complete* conversation and
    subtracting the tokenizer's own ``add_generation_prompt`` prefix.  This
    means the end-of-turn token is whatever the model template emits; no model
    specific token id is hard-coded here.
    """

    def __init__(self, data_path: str | Path, tokenizer: Any,
                 max_length: int = 2048, split: str = "train",
                 strict_pass_only: bool = True):
        self.data_path = str(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.samples: list[dict[str, Any]] = []
        self.stats = DatasetAudit(self.data_path, split)
        self.stats.file_sha256 = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
        self._load(strict_pass_only)

    @staticmethod
    def _messages_and_target(record: dict) -> tuple[list[dict], dict]:
        messages = record.get("messages")
        target = record.get("target")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        if any(not isinstance(message, dict) or message.get("role") not in
               {"system", "user", "assistant", "tool"} for message in messages):
            raise ValueError("messages contain an invalid role")
        if messages[-1].get("role") not in {"user", "tool"}:
            raise ValueError("prompt must end in user or tool")
        if not isinstance(target, dict) or target.get("role") != "assistant":
            raise ValueError("target must be an assistant message")
        if not target.get("content") and not target.get("tool_calls"):
            raise ValueError("target has no content or tool calls")
        return messages, target

    def _tokenize(self, messages: list[dict], target: dict) -> tuple[list[int], list[int]]:
        prefix = _as_list(self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors=None))
        full = _as_list(self.tokenizer.apply_chat_template(
            messages + [target], tokenize=True, add_generation_prompt=False,
            return_tensors=None))
        if len(full) < len(prefix) or full[:len(prefix)] != prefix:
            raise ValueError(
                "chat template prefix mismatch: complete and generation "
                "tokenization are not aligned")
        response = full[len(prefix):]
        if not response:
            raise ValueError("assistant target produced zero tokens")
        input_ids = full
        labels = [-100] * len(prefix) + response
        return input_ids, labels

    def _load(self, strict_pass_only: bool) -> None:
        with open(self.data_path, encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                self.stats.total += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    self.stats.skipped_json += 1
                    continue
                if not isinstance(record, dict):
                    self.stats.skipped_invalid += 1
                    continue
                if record.get("split") != self.split:
                    raise ValueError(f"{self.data_path}:{line_no}: expected split={self.split}, "
                                     f"got {record.get('split')!r}")
                provenance = record.get("provenance")
                if strict_pass_only and (not isinstance(provenance, dict) or
                                         provenance.get("strict_pass") is not True):
                    self.stats.skipped_fail += 1
                    continue
                if not all(isinstance(record.get(key), str) and record[key].strip()
                           for key in ("task_id", "source_group")):
                    self.stats.skipped_invalid += 1
                    continue
                record_hash = _hash_record(record)
                if record_hash in self.stats.record_hashes:
                    self.stats.skipped_duplicate += 1
                    continue
                try:
                    messages, target = self._messages_and_target(record)
                except (TypeError, ValueError, KeyError):
                    self.stats.skipped_invalid += 1
                    continue
                input_ids, labels = self._tokenize(messages, target)
                if len(input_ids) > self.max_length:
                    self.stats.skipped_long += 1
                    continue
                n_supervised = sum(label != -100 for label in labels)
                if not n_supervised:
                    self.stats.skipped_invalid += 1
                    continue
                self.stats.record_hashes.add(record_hash)
                self.stats.prompt_hashes.add(hashlib.sha256(json.dumps(
                    messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest())
                self.stats.loaded += 1
                self.stats.total_tokens += len(input_ids)
                self.stats.supervised_tokens += n_supervised
                self.stats.task_ids.add(str(record.get("task_id", "")))
                self.stats.source_groups.add(str(record.get("source_group", "")))
                self.samples.append({
                    "input_ids": input_ids,
                    "labels": labels,
                    "sample_id": record.get("sample_id", f"line-{line_no}"),
                    "task_id": record.get("task_id", ""),
                    "source_group": record.get("source_group", ""),
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]

    def report(self) -> str:
        s = self.stats
        return (f"数据统计({self.split}): 总行数={s.total}, 加载={s.loaded}, "
                f"非目标 split={s.skipped_split}, 未通过={s.skipped_fail}, "
                f"重复={s.skipped_duplicate}, 超长={s.skipped_long}, "
                f"非法={s.skipped_invalid}, 监督比例={s.supervised_ratio:.1%}")


class Collator:
    """Right-padding collator retaining the exact ``-100`` mask."""

    def __init__(self, pad_token_id: int):
        self.pad_id = int(pad_token_id)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids, labels, attention = [], [], []
        for item in batch:
            n = len(item["input_ids"])
            pad = max_len - n
            input_ids.append(item["input_ids"] + [self.pad_id] * pad)
            labels.append(item["labels"] + [-100] * pad)
            attention.append([1] * n + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }


def validate_split_isolation(train: DatasetAudit, dev: DatasetAudit) -> None:
    """Reject task/source-group overlap before loading a model."""
    task_overlap = sorted((train.task_ids or set()) & (dev.task_ids or set()))
    group_overlap = sorted((train.source_groups or set()) & (dev.source_groups or set()))
    prompt_overlap = (train.prompt_hashes or set()) & (dev.prompt_hashes or set())
    if task_overlap or group_overlap or prompt_overlap:
        details = []
        if task_overlap:
            details.append(f"task_id={task_overlap[:8]}")
        if group_overlap:
            details.append(f"source_group={group_overlap[:8]}")
        if prompt_overlap:
            details.append(f"identical_prompts={len(prompt_overlap)}")
        raise ValueError("train/dev 泄漏：" + "; ".join(details))


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def token_count(labels: torch.Tensor) -> int:
    return int((labels[..., 1:] != -100).sum().item())


def weighted_accumulation_windows(batches: Iterable[dict[str, torch.Tensor]],
                                  grad_accum: int) -> Iterator[tuple[list[dict], int]]:
    """Group micro-batches and return their exact target-token denominators."""
    if grad_accum < 1:
        raise ValueError("grad_accum must be >= 1")
    group = []
    for batch in batches:
        group.append(batch)
        if len(group) == grad_accum:
            denominator = sum(token_count(item["labels"]) for item in group)
            if denominator:
                yield group, denominator
            group = []
    if group:
        denominator = sum(token_count(item["labels"]) for item in group)
        if denominator:
            yield group, denominator


def validate_arguments(args: Any) -> None:
    for name in ("epochs", "batch_size", "grad_accum", "max_length", "lora_r",
                 "lora_alpha", "patience", "log_interval"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be >= 1")
    for name in ("lr", "max_grad_norm"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be > 0")
    for name in ("weight_decay", "min_delta"):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be >= 0")
    for name in ("warmup_ratio", "lora_dropout"):
        if not 0 <= getattr(args, name) < 1:
            raise ValueError(f"{name} must be in [0, 1)")


def environment_manifest() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "transformers", "peft", "bitsandbytes", "accelerate"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    payload = {
        "python": sys.version,
        "platform": sys.platform,
        "packages": packages,
        "torch_cuda": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    payload["environment_hash"] = hashlib.sha256(encoded).hexdigest()[:20]
    return payload


def parameter_delta(before: dict[str, torch.Tensor], model: torch.nn.Module) -> dict[str, Any]:
    changed = 0
    changed_values = 0
    max_abs = 0.0
    l2_sq = 0.0
    total = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or name not in before:
            continue
        current = parameter.detach().float().cpu()
        delta = current - before[name]
        total += delta.numel()
        if torch.any(delta != 0):
            changed += 1
        changed_values += int((delta != 0).sum().item())
        max_abs = max(max_abs, float(delta.abs().max().item())) if delta.numel() else max_abs
        l2_sq += float((delta * delta).sum().item())
    return {
        "changed_parameter_tensors": changed,
        "trainable_parameter_tensors": len(before),
        "changed_parameter_values": changed_values,
        "trainable_parameter_values": total,
        "max_abs_delta": max_abs,
        "l2_delta": math.sqrt(l2_sq),
    }


def evaluate_loss(model: torch.nn.Module, dataloader: Iterable[dict[str, torch.Tensor]],
                  device: torch.device) -> dict[str, float | int]:
    """Return token-weighted causal LM loss; no per-batch mean bias."""
    was_training = model.training
    model.eval()
    weighted_sum = 0.0
    n_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            labels = batch["labels"]
            n = token_count(labels)
            if not n:
                continue
            context = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                       if device.type == "cuda" else nullcontext())
            with context:
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=labels.to(device))
            if not torch.isfinite(outputs.loss):
                raise FloatingPointError("validation loss is not finite")
            weighted_sum += float(outputs.loss.detach().item()) * n
            n_tokens += n
    if was_training:
        model.train()
    return {"loss": weighted_sum / n_tokens if n_tokens else float("inf"),
            "tokens": n_tokens}


def _capture_trainable(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: p.detach().float().cpu().clone()
            for name, p in model.named_parameters() if p.requires_grad}


def _save_adapter(model: Any, tokenizer: Any, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path), safe_serialization=True)
    tokenizer.save_pretrained(str(path))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                         encoding="utf-8")
    temporary.replace(path)


def _adapter_file_hashes(path: Path) -> dict[str, str]:
    return {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
            for file in sorted(path.iterdir()) if file.is_file()}


def save_epoch_checkpoint(model: Any, tokenizer: Any, output_dir: Path,
                          manifest: dict[str, Any], epoch: int,
                          optimizer_steps: int, dev_loss: float) -> dict[str, Any]:
    """Save an immutable adapter with provenance beside it for evaluation."""
    checkpoint_dir = output_dir / f"epoch-{epoch:03d}"
    if checkpoint_dir.exists():
        raise ValueError(f"epoch checkpoint already exists: {checkpoint_dir}")
    adapter_dir = checkpoint_dir / "adapter"
    _save_adapter(model, tokenizer, adapter_dir)
    hashes = _adapter_file_hashes(adapter_dir)
    snapshot = {
        **manifest, "status": "checkpoint_complete", "checkpoint_epoch": epoch,
        "checkpoint_dev_loss": dev_loss, "optimizer_steps": optimizer_steps,
        "adapter": str(adapter_dir), "adapter_files_sha256": hashes,
    }
    _write_json_atomic(checkpoint_dir / "training-manifest.json", snapshot)
    return {
        "epoch": epoch, "optimizer_steps": optimizer_steps, "dev_loss": dev_loss,
        "adapter": f"{checkpoint_dir.name}/adapter",
        "training_manifest": f"{checkpoint_dir.name}/training-manifest.json",
        "adapter_files_sha256": hashes,
    }


def write_checkpoint_index(output_dir: Path, manifest: dict[str, Any],
                           checkpoints: list[dict[str, Any]]) -> None:
    _write_json_atomic(output_dir / "checkpoint-index.json", {
        "schema_version": 1, "model": manifest["model"],
        "revision": manifest.get("revision"),
        "resolved_revision": manifest.get("resolved_revision"),
        "selection_metric": manifest.get("selection_metric", "dev_token_loss"),
        "selected_epoch": manifest.get("best_epoch"),
        "checkpoints": checkpoints,
    })


def train(args: Any) -> dict[str, Any]:
    """Run QLoRA SFT.  Imports HF dependencies only when actually training."""
    validate_arguments(args)
    if not args.dry_run and not args.eval_data:
        raise ValueError("真实训练必须提供 --eval-data（独立 dev split）")
    set_seed(args.seed)
    from transformers import AutoTokenizer

    revision = args.revision or None
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=revision, trust_remote_code=False, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_data = SFTDataset(args.data, tokenizer, args.max_length, split="train")
    if not len(train_data):
        raise ValueError(f"训练集为空：{train_data.report()}")
    dev_data = None
    if args.eval_data:
        dev_data = SFTDataset(args.eval_data, tokenizer, args.max_length, split="dev")
        if not len(dev_data):
            raise ValueError(f"dev 集为空：{dev_data.report()}")
        validate_split_isolation(train_data.stats, dev_data.stats)
    print(train_data.report(), flush=True)
    if dev_data:
        print(dev_data.report(), flush=True)
    if args.dry_run:
        return {"train": train_data.stats.as_dict(),
                "dev": dev_data.stats.as_dict() if dev_data else None}
    if not torch.cuda.is_available():
        raise RuntimeError("真实 QLoRA 训练需要 CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("此训练配置需要支持 BF16 的 GPU，例如 24GB 单卡 GPU")
    from transformers import (AutoModelForCausalLM, BitsAndBytesConfig,
                              get_linear_schedule_with_warmup)
    from peft import (LoraConfig, get_peft_model, prepare_model_for_kbit_training)
    from peft.utils.save_and_load import (get_peft_model_state_dict, load_peft_weights,
                                         set_peft_model_state_dict)

    device = torch.device("cuda:0")
    output_dir = Path(args.output)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("输出目录非空，请使用新的 --output 目录保留实验记录")
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_manifest = {
        "schema_version": 2, "status": "initialized", "configuration": vars(args),
        "model": args.model, "revision": revision,
        "selection_metric": "dev_token_loss",
        "data": train_data.stats.as_dict(),
        "eval_data": dev_data.stats.as_dict() if dev_data else None,
        "environment": environment_manifest(),
    }
    (output_dir / "run-config.json").write_text(
        json.dumps(initial_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_json_atomic(output_dir / "training-manifest.json", initial_manifest)
    if torch.cuda.device_count() > 1:
        print("提示：显式使用 cuda:0，忽略其它可见 GPU", flush=True)
    torch.cuda.reset_peak_memory_stats()
    print("加载基座模型（首次运行需下载约 15GB；NF4 量化加载约 1-3 分钟，此段无输出属正常）…",
          flush=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=revision, quantization_config=bnb,
        device_map={"": 0}, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", trust_remote_code=False)
    resolved_revision = getattr(model.config, "_commit_hash", None)
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    if any(p.requires_grad and "lora_" not in name
           for name, p in model.named_parameters()):
        raise RuntimeError("base weights are not frozen")
    before = _capture_trainable(model)

    collator = Collator(tokenizer.pad_token_id)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              generator=generator, collate_fn=collator, num_workers=0)
    dev_loader = DataLoader(dev_data, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collator, num_workers=0) if dev_data else None
    total_steps = math.ceil(len(train_loader) / args.grad_accum) * args.epochs
    if total_steps < 1:
        raise ValueError("训练集没有可用于优化的监督 token")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.999), eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=min(total_steps - 1, math.ceil(total_steps * args.warmup_ratio)),
        num_training_steps=total_steps)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training-log.jsonl"
    best_dir = output_dir / "best_adapter"
    print("模型加载完成，计算未训练基线的 dev loss…", flush=True)
    baseline_dev = evaluate_loss(model, dev_loader, device) if dev_loader else {"loss": float("inf"), "tokens": 0}
    best_loss = baseline_dev["loss"]
    best_epoch = 0
    no_improve = 0
    history = []
    checkpoints = [{
        "epoch": 0, "optimizer_steps": 0, "dev_loss": baseline_dev["loss"],
        "adapter": None, "training_manifest": "training-manifest.json",
        "adapter_files_sha256": {},
    }]
    global_step = 0
    train_start = time.time()
    _save_adapter(model, tokenizer, best_dir)
    progress_manifest = {
        **initial_manifest, "status": "training", "resolved_revision": resolved_revision,
        "tokenizer_revision": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        "chat_template_sha256": hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
        "baseline_dev": baseline_dev, "history": history, "best_epoch": 0,
        "best_dev_loss": best_loss, "optimizer_steps": 0,
        "checkpoint_index": "checkpoint-index.json",
        "save_every_epoch": getattr(args, "save_every_epoch", True),
    }
    _write_json_atomic(output_dir / "training-manifest.json", progress_manifest)
    write_checkpoint_index(output_dir, progress_manifest, checkpoints)
    with log_path.open("w", encoding="utf-8") as log_file:
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_loss_sum, epoch_tokens = 0.0, 0
            windows = weighted_accumulation_windows(train_loader, args.grad_accum)
            for group, denominator in windows:
                optimizer.zero_grad(set_to_none=True)
                step_loss_sum = 0.0
                for batch in group:
                    n = token_count(batch["labels"])
                    if not n:
                        continue
                    try:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            outputs = model(
                                input_ids=batch["input_ids"].to(device),
                                attention_mask=batch["attention_mask"].to(device),
                                labels=batch["labels"].to(device))
                            loss = outputs.loss
                        if not torch.isfinite(loss):
                            raise FloatingPointError("training loss is not finite")
                        (loss * (n / denominator)).backward()
                        value = float(loss.detach().item())
                        step_loss_sum += value * n
                        epoch_loss_sum += value * n
                        epoch_tokens += n
                    except torch.cuda.OutOfMemoryError as exc:
                        optimizer.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        progress_manifest.update({
                            "status": "interrupted", "failure_type": "cuda_out_of_memory",
                            "failed_epoch": epoch, "optimizer_steps": global_step,
                        })
                        _write_json_atomic(output_dir / "training-manifest.json", progress_manifest)
                        raise RuntimeError(
                            "CUDA OOM；已 fail-fast，请降低 --max-length 或 --batch-size") from exc
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                entry = {
                    "step": global_step, "epoch": epoch,
                    "loss": step_loss_sum / denominator,
                    "lr": scheduler.get_last_lr()[0], "target_tokens": denominator,
                    "elapsed_s": time.time() - train_start,
                }
                log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                log_file.flush()
                if global_step % args.log_interval == 0:
                    elapsed = entry["elapsed_s"]
                    remaining = elapsed / global_step * (total_steps - global_step)
                    filled = 24 * global_step // total_steps
                    bar = '#' * filled + '-' * (24 - filled)
                    print(f"[{bar}] step {global_step}/{total_steps} "
                          f"({global_step * 100 // total_steps}%) "
                          f"loss={entry['loss']:.4f} lr={entry['lr']:.2e} · "
                          f"已用 {elapsed / 60:.1f}m · 剩余≈{remaining / 60:.1f}m", flush=True)
            dev_metrics = evaluate_loss(model, dev_loader, device) if dev_loader else {"loss": float("inf"), "tokens": 0}
            epoch_entry = {
                "epoch": epoch, "train_loss": epoch_loss_sum / epoch_tokens if epoch_tokens else float("inf"),
                "dev_loss": dev_metrics["loss"], "dev_tokens": dev_metrics["tokens"],
            }
            history.append(epoch_entry)
            print(f"epoch {epoch}/{args.epochs} train_loss={epoch_entry['train_loss']:.4f} "
                  f"dev_loss={epoch_entry['dev_loss']:.4f}", flush=True)
            if dev_metrics["loss"] < best_loss - args.min_delta:
                best_loss = dev_metrics["loss"]
                best_epoch = epoch
                no_improve = 0
                _save_adapter(model, tokenizer, best_dir)
            else:
                no_improve += 1
            progress_manifest.update({
                "best_epoch": best_epoch, "best_dev_loss": best_loss,
                "optimizer_steps": global_step, "elapsed_s": time.time() - train_start,
                "last_completed_epoch": epoch,
            })
            if getattr(args, "save_every_epoch", True):
                checkpoints.append(save_epoch_checkpoint(
                    model, tokenizer, output_dir, progress_manifest,
                    epoch, global_step, dev_metrics["loss"]))
            _write_json_atomic(output_dir / "training-manifest.json", progress_manifest)
            write_checkpoint_index(output_dir, progress_manifest, checkpoints)
            if no_improve >= args.patience:
                print(f"early stop: {args.patience} 个 epoch 无 dev 改善")
                break
    final_delta = parameter_delta(before, model)
    if final_delta["changed_parameter_tensors"] == 0:
        raise RuntimeError("训练结束但没有可训练参数发生变化，请检查梯度或学习率")
    best_weights = load_peft_weights(str(best_dir), device="cpu")
    set_peft_model_state_dict(model, best_weights)
    reloaded_weights = get_peft_model_state_dict(model)
    reload_verified = all(
        name in reloaded_weights and torch.equal(
            tensor.cpu(), reloaded_weights[name].detach().cpu())
        for name, tensor in best_weights.items())
    if not reload_verified:
        raise RuntimeError("saved adapter reload did not reproduce its weights")
    selected_delta = parameter_delta(before, model)
    adapter_dir = output_dir / "adapter"
    shutil.copytree(best_dir, adapter_dir)
    if not getattr(args, "save_every_epoch", True) and best_epoch:
        checkpoints.append({
            "epoch": best_epoch,
            "optimizer_steps": math.ceil(len(train_loader) / args.grad_accum) * best_epoch,
            "dev_loss": best_loss, "adapter": "adapter",
            "training_manifest": "training-manifest.json",
            "adapter_files_sha256": _adapter_file_hashes(adapter_dir),
        })
        write_checkpoint_index(output_dir, progress_manifest, checkpoints)
    peak = torch.cuda.max_memory_allocated() / 2**30
    config = vars(args).copy()
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    adapter_files = _adapter_file_hashes(adapter_dir)
    manifest = {
        "schema_version": 2, "status": "complete", "model": args.model, "revision": revision,
        "resolved_revision": resolved_revision,
        "tokenizer_revision": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        "chat_template_sha256": hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
        "configuration": config, "configuration_sha256": config_hash,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": args.seed, "data": train_data.stats.as_dict(),
        "eval_data": dev_data.stats.as_dict() if dev_data else None,
        "train_config": {k: getattr(args, k) for k in (
            "epochs", "batch_size", "grad_accum", "lr", "weight_decay",
            "warmup_ratio", "max_length", "lora_r", "lora_alpha",
            "lora_dropout", "max_grad_norm", "patience", "min_delta")},
        "baseline_dev": baseline_dev, "history": history,
        "best_epoch": best_epoch, "best_dev_loss": best_loss,
        "optimizer_steps": global_step, "elapsed_s": time.time() - train_start,
        "peak_vram_gb": peak, "parameter_delta": selected_delta,
        "final_parameter_delta": final_delta,
        "adapter_reload_weights_verified": reload_verified,
        "adapter_files_sha256": adapter_files,
        "selection_reason": "dev_loss_improved" if best_epoch else "no_dev_improvement_keep_base",
        "selection_metric": "dev_token_loss", "checkpoint_index": "checkpoint-index.json",
        "save_every_epoch": getattr(args, "save_every_epoch", True),
        "last_completed_epoch": history[-1]["epoch"],
        "environment": environment_manifest(),
        "adapter": str(adapter_dir),
    }
    _write_json_atomic(output_dir / "training-manifest.json", manifest)
    return manifest


__all__ = [
    "SFTDataset", "Collator", "DatasetAudit", "validate_split_isolation",
    "set_seed", "token_count", "weighted_accumulation_windows",
    "evaluate_loss", "parameter_delta", "environment_manifest", "train",
]
