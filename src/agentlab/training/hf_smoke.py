# -*- coding: utf-8 -*-
"""模型冒烟测试：在已加载的模型上把 SFT 的每一步走一遍。

目的是在付费采集和长时间训练之前，把只能在目标机器上验证的环节暴露出来：
bitsandbytes 的 NF4 量化、显存占用、LoRA 能否挂到这种架构上并得到梯度、
关闭思考模式后的输出是否是干净的代码。

走的是与 ``scripts/train_sft.py`` 相同的路径和参数：
chat template 与 assistant-only 掩码 → 冻结基座 → 挂 LoRA → bf16 autocast 下
带 labels 的一步训练 → 贪心解码。每一步的结果都写进返回的报告。
"""
from __future__ import annotations

import ast
import importlib.util
import json
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

SMOKE_LR = 2e-5          # 与 train_sft.py 的默认学习率相同


def _target_of(parameter_name: str, targets: list[str]) -> str | None:
    """可训练参数属于哪个 LoRA 目标模块。"""
    parts = parameter_name.split(".")
    return next((target for target in targets if target in parts), None)


def smoke_test(model: Any, tokenizer: Any, messages: list[dict], target: dict, *,
               lora_r: int = 16, lora_alpha: int = 32, max_length: int = 2048,
               max_new_tokens: int = 256,
               extractor: Callable[[str], str] | None = None) -> dict:
    """返回报告；``passed`` 为真表示每一项检查都通过。

    ``extractor`` 给出时，还会检查生成内容能提取出可解析的 Python。
    """
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    from agentlab.training.hf_runtime import (
        generate_text, lora_targets_for, prompt_inputs, thinking_capable,
    )
    from agentlab.training.hf_sft import SFTDataset
    from agentlab.training.sequence_loss import compare_losses, sequence_loss

    device = next(model.parameters()).device
    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    checks: dict[str, bool] = {}
    report: dict[str, Any] = {
        "model_class": type(model).__name__, "device": str(device),
        "quantized": bool(getattr(model, "is_loaded_in_4bit", False)
                          or getattr(model, "is_loaded_in_8bit", False)),
        "thinking_switch_in_template": thinking_capable(tokenizer),
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        report["weight_memory_gib"] = round(torch.cuda.memory_allocated() / 2**30, 2)
    # 线性注意力层的加速内核。没有它们 transformers 会退回参考实现：结果正确，但慢很多。
    report["fast_kernels"] = {
        name: importlib.util.find_spec(module) is not None
        for name, module in (("causal_conv1d", "causal_conv1d"),
                             ("flash_linear_attention", "fla"))}

    # 1. 样本：训练时的同一套 chat template 与掩码
    with tempfile.TemporaryDirectory(prefix="codeagent-smoke-") as temporary:
        path = Path(temporary) / "smoke.jsonl"
        path.write_text(json.dumps({
            "sample_id": "smoke", "task_id": "smoke", "source_group": "smoke",
            "split": "train", "messages": messages, "target": target,
            "provenance": {"strict_pass": True}}, ensure_ascii=False) + "\n",
            encoding="utf-8")
        try:
            dataset = SFTDataset(path, tokenizer, max_length)
        except Exception as error:           # 掩码或模板问题：到这里就够了
            checks["sample_built"] = False
            report["error"] = f"building the training sample: {type(error).__name__}: {error}"
            report.update(checks=checks, passed=False)
            return report
    checks["sample_built"] = len(dataset) == 1
    sample = dataset[0]
    report["sample_tokens"] = len(sample["input_ids"])
    report["supervised_tokens"] = sum(label != -100 for label in sample["labels"])

    # 2. LoRA：与 hf_sft.train 相同的冻结、梯度检查点和目标选择
    targets = lora_targets_for(model)
    matched = Counter(name.rsplit(".", 1)[-1] for name, _ in model.named_modules()
                      if name.rsplit(".", 1)[-1] in set(targets))
    report["lora"] = {"r": lora_r, "alpha": lora_alpha, "targets": targets,
                      "modules_per_target": dict(matched)}
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05,
        target_modules=targets, task_type="CAUSAL_LM"))
    trainable = {name: parameter for name, parameter in model.named_parameters()
                 if parameter.requires_grad}
    checks["base_frozen"] = bool(trainable) and all("lora_" in name for name in trainable)
    report["trainable_parameters"] = sum(p.numel() for p in trainable.values())

    # 3. 一步训练：hf_sft.train 循环体在 microbatch 1 上的等价形式
    input_ids = torch.tensor([sample["input_ids"]], device=device)
    labels = torch.tensor([sample["labels"]], device=device)
    attention = torch.ones_like(input_ids)
    before = {name: parameter.detach().clone() for name, parameter in trainable.items()}
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=SMOKE_LR, weight_decay=0.01)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    with torch.no_grad(), autocast:                       # 与训练开始前的核对相同
        loss_check = compare_losses(model, input_ids, attention, labels)
    with autocast:
        loss = sequence_loss(model, input_ids, attention, labels)
    checks["loss_finite"] = bool(torch.isfinite(loss))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1.0, error_if_nonfinite=True)
    optimizer.step()
    report["train_step"] = {
        "loss": float(loss.detach()), "seconds": round(time.perf_counter() - started, 2),
        "loss_implementation": "chunked_lm_head", "loss_check": loss_check}
    with_grad, updated = Counter(), Counter()
    for name, parameter in trainable.items():
        owner = _target_of(name, targets)
        if parameter.grad is not None:
            with_grad[owner] += 1
        if not torch.equal(before[name], parameter.detach()):
            updated[owner] += 1
    report["train_step"].update(tensors_with_grad=dict(with_grad), tensors_updated=dict(updated))
    # lora_B 初始为 0，所以 lora_A 在第一步的梯度为 0；只要求每个目标都有参数被更新
    checks["every_target_updated"] = bool(targets) and all(updated[target] > 0 for target in targets)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # 4. 生成：评测时的同一条路径
    model.eval()
    inputs, attention = prompt_inputs(tokenizer, messages, device)
    started = time.perf_counter()
    raw, n_tokens = generate_text(model, tokenizer, inputs, attention,
                                  max_new_tokens=max_new_tokens, do_sample=False)
    checks["generated"] = n_tokens > 0 and bool(raw.strip())
    checks["no_thinking"] = "<think>" not in raw and "</think>" not in raw
    if extractor is not None:
        code = extractor(raw)
        checks["code_extracted"] = bool(code.strip())
        try:
            ast.parse(code)                # 空字符串也能解析，所以先看是否提取到了代码
            checks["code_parses"] = checks["code_extracted"]
        except SyntaxError:
            checks["code_parses"] = False
    seconds = time.perf_counter() - started
    report["generation"] = {
        "seconds": round(seconds, 2), "output_tokens": n_tokens,
        "tokens_per_second": round(n_tokens / seconds, 1) if seconds else None,
        "hit_token_limit": n_tokens >= max_new_tokens, "output_head": raw[:300]}
    if device.type == "cuda":
        report["peak_memory_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    report.update(checks=checks, passed=all(checks.values()))
    return report
