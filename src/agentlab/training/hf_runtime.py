# -*- coding: utf-8 -*-
"""本地模型的统一入口：加载、chat template、生成。评测与数据脚本共用。

此前 evaluate.py、generate_rft_data.py、generate_preferences.py 各自复制了一份
量化与加载代码，模型家族的差异没有地方集中处理。现在差异都在这里：

- **思考模式**：Qwen3 起的 chat template 默认在生成提示后追加 ``<think>``，
  模型会先输出思考再给答案。训练时前缀相减会对不上，``</think>`` 也不是特殊
  token、不会被过滤，思考内容可能占满生成额度。所以这里默认关闭；评测可以
  显式打开（``enable_thinking=True``），再用 ``split_thinking`` 取出 ``</think>``
  之后的回答。对不含该变量的模板（如 Qwen2.5）不传任何额外参数，行为逐 token 不变。
- **LoRA 目标**：按模型里真实存在的模块选择。Qwen3.5 每 4 层有 3 层线性
  注意力，投影名是 ``in_proj_qkv`` / ``in_proj_z`` / ``out_proj``，Qwen2.5 的
  列表一个都命不中。
- **transformers 版本**：Qwen3.5 需要 ≥ 5.2.0，加载前给出可操作的报错。

本模块导入时不依赖 torch / transformers，相关导入都在函数内部。
"""
from __future__ import annotations

from typing import Any

from agentlab.models import default_revision, minimum_transformers

# Qwen2.5 的全部线性层；与改动前的列表逐项相同
BASE_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj")
# 线性注意力层（Gated DeltaNet）的混合投影，Qwen2.5 没有这些模块。in_proj_a /
# in_proj_b 的输出维度只有头数，不挂 LoRA。
HYBRID_LORA_TARGETS = ("in_proj_qkv", "in_proj_z", "out_proj")
QUANTIZATIONS = ("nf4", "bf16")


def thinking_capable(tokenizer: Any) -> bool:
    """模板里是否有 enable_thinking 开关。"""
    return "enable_thinking" in str(getattr(tokenizer, "chat_template", "") or "")


def apply_chat_template(tokenizer: Any, messages: list[dict], *,
                        add_generation_prompt: bool, return_tensors: Any = None,
                        enable_thinking: bool = False) -> Any:
    """项目内唯一的 chat template 调用入口：思考模式默认关闭，评测显式要求时才打开。

    打开后提示以 ``<think>`` 结尾，模型先输出推理、再输出 ``</think>``、最后才是回答。
    """
    switch = {"enable_thinking": enable_thinking} if thinking_capable(tokenizer) else {}
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt,
        return_tensors=return_tensors, **switch)


def split_thinking(text: str) -> tuple[str, str, bool]:
    """把思考模式的输出拆成（推理内容，正式回答，推理是否结束）。

    回答取第一个 ``</think>`` 之后的部分，与 Qwen3.5 的对话模板一致；
    没有出现 ``</think>`` 说明生成额度用完时推理还没结束，没有可判分的回答。
    """
    if "</think>" not in text:
        return text, "", False
    reasoning, answer = text.split("</think>", 1)
    return reasoning, answer, True


def lora_targets_for(model: Any) -> list[str]:
    """只保留模型里真实存在的目标名。Qwen2.5 下与原列表完全相同。"""
    names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    return [target for target in (*BASE_LORA_TARGETS, *HYBRID_LORA_TARGETS)
            if target in names]


def _version_tuple(version: str) -> tuple[int, ...]:
    parts = []
    for piece in version.split(".")[:3]:
        digits = "".join(character for character in piece if character.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def check_version(model_type: str, transformers_version: str) -> None:
    """架构比已安装的 transformers 新时，给出版本要求而不是模糊的报错。"""
    minimum = minimum_transformers(model_type)
    if minimum and _version_tuple(transformers_version) < minimum:
        required = ".".join(str(part) for part in minimum)
        raise RuntimeError(
            f"model architecture {model_type!r} needs transformers >= {required}, "
            f"but {transformers_version} is installed")


def assert_model_supported(model: str, revision: str | None = None) -> None:
    """加载权重之前先读配置：架构不认识或版本不够时尽早失败。"""
    import transformers
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(model, revision=revision, trust_remote_code=False)
    except ValueError as error:
        if "does not recognize" in str(error):
            raise RuntimeError(
                f"transformers {transformers.__version__} does not know the architecture of "
                f"{model}. Update transformers in this environment; Qwen3.5 needs 5.2.0 or "
                "later. A fresh environment gets the right versions from '.[dev,train,api]'."
            ) from error
        raise
    check_version(config.model_type, transformers.__version__)


def resolve_revision(model: str, revision: str | None) -> str | None:
    """未指定 revision 时用该模型已核对的固定版本。"""
    return revision or default_revision(model)


def load_tokenizer(source: str, revision: str | None = None, *,
                   padding_side: str | None = None) -> Any:
    from transformers import AutoTokenizer

    options = {"padding_side": padding_side} if padding_side else {}
    tokenizer = AutoTokenizer.from_pretrained(
        source, revision=revision, trust_remote_code=False, **options)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def quantization_config(quantization: str):
    """NF4 双量化；'bf16' 表示不量化。"""
    import torch
    from transformers import BitsAndBytesConfig

    if quantization == "bf16":
        return None
    if quantization != "nf4":
        raise ValueError(f"unknown quantization {quantization!r}; expected one of {QUANTIZATIONS}")
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)


def load_causal_lm(model: str, revision: str | None = None,
                   quantization: str = "nf4", device: str | None = None) -> Any:
    """bf16 权重放在 cuda:0，按需 NF4 量化，sdpa 注意力。

    ``device`` 只给冒烟测试在 CPU 上自测用（例如 ``"cpu"``）；bitsandbytes 的
    量化需要 CUDA，CPU 上只能用 ``'bf16'``。
    """
    import torch
    from transformers import AutoModelForCausalLM

    assert_model_supported(model, revision)
    if device is None:
        placement = {"": 0}
    elif device.startswith("cuda"):
        placement = {"": int(device.split(":")[1] or 0)}
    else:
        placement = device
    options = dict(revision=revision, device_map=placement, torch_dtype=torch.bfloat16,
                   trust_remote_code=False, attn_implementation="sdpa")
    bnb = quantization_config(quantization)
    if bnb is not None:
        options["quantization_config"] = bnb
    return AutoModelForCausalLM.from_pretrained(model, **options)


def prompt_inputs(tokenizer: Any, messages: list[dict], device, *,
                  enable_thinking: bool = False) -> tuple[Any, Any]:
    """把提示转成模型输入。兼容返回张量和 BatchEncoding 的 transformers 版本。"""
    from collections.abc import Mapping

    import torch

    encoded = apply_chat_template(tokenizer, messages, add_generation_prompt=True,
                                  return_tensors="pt", enable_thinking=enable_thinking)
    if isinstance(encoded, Mapping):
        inputs = encoded["input_ids"].to(device)
        attention = encoded.get("attention_mask")
        attention = attention.to(device) if attention is not None else torch.ones_like(inputs)
    else:
        inputs = encoded.to(device)
        attention = torch.ones_like(inputs)
    return inputs, attention


def generate_text(model: Any, tokenizer: Any, inputs, attention, *, max_new_tokens: int,
                  do_sample: bool, temperature: float | None = None,
                  top_k: int | None = None, top_p: float | None = None) -> tuple[str, int]:
    """单条序列生成，返回 (去掉特殊 token 的文本, 生成的 token 数)。"""
    import torch

    generation = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
                      pad_token_id=tokenizer.pad_token_id)
    if do_sample:
        generation.update(temperature=temperature, top_p=1.0 if top_p is None else top_p)
        if top_k is not None:
            generation["top_k"] = top_k
    with torch.inference_mode():
        output = model.generate(inputs, attention_mask=attention, **generation)
    generated = output[0][inputs.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True), len(generated)
