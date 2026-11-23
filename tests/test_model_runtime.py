# -*- coding: utf-8 -*-
"""统一模型入口的行为：思考模式、LoRA 目标、版本门禁、加载与生成参数。

这些测试全部离线：用假 tokenizer / 假模型复现各模型家族的关键行为，
CI 不需要 GPU、权重或网络。
"""
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentlab.models import (  # noqa: E402
    DEFAULT_MODEL, default_revision, install_extra, needs_new_transformers,
)
from agentlab.training import hf_runtime  # noqa: E402
from agentlab.training.hf_runtime import (  # noqa: E402
    BASE_LORA_TARGETS, HYBRID_LORA_TARGETS, apply_chat_template, check_version,
    generate_text, lora_targets_for, prompt_inputs, resolve_revision,
)
from agentlab.training.hf_sft import SFTDataset  # noqa: E402

try:
    import transformers  # noqa: E402
except ImportError:  # pragma: no cover - the plain dev extra has no transformers
    transformers = None
needs_transformers = pytest.mark.skipif(transformers is None, reason="transformers not installed")


def _tokens(text):
    """连续相同字符合并成一个 token，模拟 BPE：让“换行 + 换行”变成另一个 token。"""
    tokens, index = [], 0
    while index < len(text):
        end = index
        while end < len(text) and text[end] == text[index]:
            end += 1
        tokens.append(ord(text[index]) << 8 | (end - index))
        index = end
    return tokens


def _text(tokens):
    return "".join(chr(token >> 8) * (token & 0xFF) for token in tokens)


class ThinkingTokenizer:
    """复现 Qwen3 起的 chat template：默认在生成提示后打开 <think>。

    训练代码用“完整对话的 token 减去生成提示的 token”得到只覆盖回答的标签。
    思考模式打开时，生成提示以单独一个换行结尾，完整对话里同一个位置是
    “换行换行”，token 不同，前缀对不上——与真实 Qwen3.5 tokenizer 的表现一致。
    """
    chat_template = "{% if enable_thinking is defined and enable_thinking is false %}"
    pad_token = "<pad>"
    pad_token_id = 0
    CLOSED = "<think>\n\n</think>\n\n"

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True,
                            return_tensors=None, **kwargs):
        assert tokenize
        thinking = kwargs.get("enable_thinking", True)
        context = "".join(f"<{message['role']}>{message['content']}<end>\n"
                          for message in messages if message["role"] != "assistant")
        if add_generation_prompt:
            tail = "<assistant><think>\n" if thinking else "<assistant>" + self.CLOSED
        else:
            tail = "<assistant>" + self.CLOSED + messages[-1]["content"] + "<end>\n"
        return _tokens(context + tail)


class PlainTokenizer:
    """Qwen2.5 一类：没有 enable_thinking 变量，也不接受额外参数。"""
    chat_template = "<im_start>{{ message['content'] }}<im_end>"

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True,
                            return_tensors=None):
        assert tokenize
        context = "".join(f"<{message['role']}>{message['content']}<end>\n"
                          for message in messages if message["role"] != "assistant")
        tail = ("<assistant>" if add_generation_prompt
                else "<assistant>" + messages[-1]["content"] + "<end>\n")
        return _tokens(context + tail)


def record():
    return {"sample_id": "s1", "task_id": "t1", "source_group": "g1", "split": "train",
            "messages": [{"role": "user", "content": "fix"}],
            "target": {"role": "assistant", "content": "return 7"},
            "provenance": {"strict_pass": True}}


def dataset(tmp_path, tokenizer):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(record()) + "\n", encoding="utf-8")
    return SFTDataset(path, tokenizer)


def parts(sample):
    prefix = _text([token for token, label in zip(sample["input_ids"], sample["labels"])
                    if label == -100])
    supervised = _text([token for token, label in zip(sample["input_ids"], sample["labels"])
                        if label != -100])
    return prefix, supervised


def test_thinking_templates_are_used_with_thinking_disabled(tmp_path):
    prefix, supervised = parts(dataset(tmp_path, ThinkingTokenizer())[0])
    assert prefix == "<user>fix<end>\n<assistant><think>\n\n</think>\n\n"
    assert supervised == "return 7<end>\n"          # 不带思考块


def test_thinking_templates_fail_closed_when_thinking_stays_on(tmp_path):
    class NeverDisables(ThinkingTokenizer):
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True,
                                return_tensors=None, **kwargs):
            kwargs["enable_thinking"] = True       # 相当于改动前：没人关闭思考模式
            return super().apply_chat_template(messages, tokenize, add_generation_prompt,
                                               return_tensors, **kwargs)

    with pytest.raises(ValueError, match="prefix mismatch"):
        dataset(tmp_path, NeverDisables())


def test_plain_templates_receive_no_extra_arguments():
    seen = []

    class Recording(PlainTokenizer):
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True,
                                return_tensors=None, **rest):
            seen.append(rest)
            return super().apply_chat_template(messages, tokenize, add_generation_prompt,
                                               return_tensors)

    messages = record()["messages"]
    apply_chat_template(Recording(), messages, add_generation_prompt=True)
    apply_chat_template(Recording(), messages, add_generation_prompt=False)
    assert seen == [{}, {}]                         # Qwen2.5：行为逐 token 不变


def test_plain_template_mask_is_unchanged(tmp_path):
    prefix, supervised = parts(dataset(tmp_path, PlainTokenizer())[0])
    assert prefix == "<user>fix<end>\n<assistant>"
    assert supervised == "return 7<end>\n"


def test_prompt_inputs_handle_tensor_and_mapping_outputs():
    class TensorTokenizer(PlainTokenizer):
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True,
                                return_tensors=None):
            return torch.tensor([[1, 2, 3]])

    class MappingTokenizer(TensorTokenizer):
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True,
                                return_tensors=None):
            return {"input_ids": torch.tensor([[1, 2, 3]]),
                    "attention_mask": torch.tensor([[0, 1, 1]])}

    inputs, attention = prompt_inputs(TensorTokenizer(), [], torch.device("cpu"))
    assert inputs.tolist() == [[1, 2, 3]] and attention.tolist() == [[1, 1, 1]]
    inputs, attention = prompt_inputs(MappingTokenizer(), [], torch.device("cpu"))
    assert inputs.tolist() == [[1, 2, 3]] and attention.tolist() == [[0, 1, 1]]


def module_with(qualified_names):
    """两层解码器，每层带给定的 `<block>.<module>` 线性层。"""
    model = torch.nn.Module()
    for index in range(2):
        layer = torch.nn.Module()
        for qualified in qualified_names:
            block, name = qualified.split(".")
            if not hasattr(layer, block):
                setattr(layer, block, torch.nn.Module())
            setattr(getattr(layer, block), name, torch.nn.Linear(4, 4))
        model.add_module(f"layer_{index}", layer)
    return model


def test_lora_targets_follow_the_modules_that_exist():
    dense = module_with([f"self_attn.{name}" for name in BASE_LORA_TARGETS[:4]]
                        + [f"mlp.{name}" for name in BASE_LORA_TARGETS[4:]])
    assert lora_targets_for(dense) == list(BASE_LORA_TARGETS)      # Qwen2.5：与原列表相同
    hybrid = module_with([f"linear_attn.{name}" for name in
                          ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")]
                         + [f"mlp.{name}" for name in BASE_LORA_TARGETS[4:]])
    # 线性注意力层没有 q/k/v/o 投影，改挂它自己的三个混合投影
    assert lora_targets_for(hybrid) == [*BASE_LORA_TARGETS[4:], *HYBRID_LORA_TARGETS]
    assert "in_proj_a" not in lora_targets_for(hybrid)
    assert lora_targets_for(torch.nn.Module()) == []


def test_version_gate_and_model_table():
    check_version("qwen2", "4.46.0")
    check_version("qwen3.5", "5.2.0")
    for old in ("4.57.6", "5.1.0"):
        with pytest.raises(RuntimeError, match="transformers >= 5.2.0"):
            check_version("qwen3.5", old)
    assert default_revision("Qwen/Qwen3.5-9B") == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert default_revision(None) == default_revision(DEFAULT_MODEL)
    assert default_revision("some-org/model") is None
    assert resolve_revision("some-org/model", "abc") == "abc"
    assert resolve_revision("some-org/model", None) is None
    assert resolve_revision("Qwen/Qwen3.5-9B", None) == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    assert needs_new_transformers("Qwen/Qwen3.5-9B") and not needs_new_transformers(DEFAULT_MODEL)
    assert install_extra("Qwen/Qwen3.5-9B") == "train-qwen35"
    assert install_extra(DEFAULT_MODEL) == "train"


@needs_transformers
def test_unsupported_architecture_names_the_fix(monkeypatch):
    def unknown(*args, **kwargs):
        raise ValueError("model type 'qwen3_5' ... Transformers does not recognize this architecture")

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", unknown)
    with pytest.raises(RuntimeError, match="train-qwen35"):
        hf_runtime.assert_model_supported("Qwen/Qwen3.5-9B", "abc")


@needs_transformers
def test_model_loading_arguments_are_unchanged(monkeypatch):
    """与改动前三份脚本各自的 from_pretrained 参数逐项相同。"""
    calls = []

    def fake_from_pretrained(model, **kwargs):
        calls.append((model, kwargs))
        return object()

    monkeypatch.setattr(hf_runtime, "assert_model_supported", lambda *args, **kwargs: None)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained)
    hf_runtime.load_causal_lm("org/model", "rev", "nf4")
    hf_runtime.load_causal_lm("org/model", None, "bf16")
    common = dict(device_map={"": 0}, torch_dtype=torch.bfloat16, trust_remote_code=False,
                  attn_implementation="sdpa")
    assert calls[0][0] == calls[1][0] == "org/model"
    quantized, unquantized = calls[0][1], calls[1][1]
    assert {key: value for key, value in quantized.items() if key != "quantization_config"} == {
        **common, "revision": "rev"}
    assert quantized["quantization_config"].bnb_4bit_quant_type == "nf4"
    assert quantized["quantization_config"].bnb_4bit_use_double_quant is True
    assert unquantized == {**common, "revision": None}
    with pytest.raises(ValueError, match="quantization"):
        hf_runtime.load_causal_lm("org/model", None, "int8")


@needs_transformers
def test_generation_arguments_match_the_previous_call_sites():
    calls = []

    class Model:
        def generate(self, inputs, **kwargs):
            calls.append(kwargs)
            return torch.cat([inputs, torch.tensor([[7, 8, 9]])], dim=1)

    class Tokenizer:
        pad_token_id = 0

        def decode(self, tokens, skip_special_tokens):
            assert skip_special_tokens
            return "decoded"

    inputs = torch.tensor([[1, 2, 3]])
    attention = torch.ones(1, 3, dtype=torch.long)
    model, tokenizer = Model(), Tokenizer()
    assert generate_text(model, tokenizer, inputs, attention, max_new_tokens=5,
                         do_sample=False) == ("decoded", 3)
    generate_text(model, tokenizer, inputs, attention, max_new_tokens=5, do_sample=True,
                  temperature=0.7, top_k=0)              # evaluate.py 的采样路径
    generate_text(model, tokenizer, inputs, attention, max_new_tokens=5, do_sample=True,
                  temperature=0.7)                       # RFT / 偏好数据采样路径
    assert calls[0] == dict(max_new_tokens=5, do_sample=False, pad_token_id=0,
                            attention_mask=attention)
    assert calls[1] == dict(max_new_tokens=5, do_sample=True, pad_token_id=0,
                            attention_mask=attention, temperature=0.7, top_p=1.0, top_k=0)
    assert calls[2] == dict(max_new_tokens=5, do_sample=True, pad_token_id=0,
                            attention_mask=attention, temperature=0.7, top_p=1.0)
