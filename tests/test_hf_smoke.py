# -*- coding: utf-8 -*-
"""冒烟测试的逻辑在 CPU 上用微型的真实架构跑一遍。

覆盖的是架构和代码路径：LoRA 能否挂到这种架构的模块上、带 labels 的前向和反向、
梯度检查点、贪心解码。bitsandbytes 的 NF4 量化需要 CUDA，由服务器上的
``smoke`` 阶段验证。未安装 peft、或 transformers 太旧而没有 Qwen3.5 时跳过。
"""
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3_5ForCausalLM"):
    pytest.skip("this transformers has no Qwen3.5", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentlab.training.hf_runtime import (  # noqa: E402
    BASE_LORA_TARGETS, HYBRID_LORA_TARGETS,
)
from agentlab.training.hf_smoke import smoke_test  # noqa: E402
from tiny_models import tiny_qwen25, tiny_qwen35  # noqa: E402

TRAINING_CHECKS = ("sample_built", "base_frozen", "loss_finite", "every_target_updated",
                   "generated", "no_thinking")
MESSAGES = [{"role": "system", "content": "Output Python."},
            {"role": "user", "content": "Fix even_numbers(n)."}]
TARGET = {"role": "assistant", "content": "def even_numbers(n):\n    return [i for i in range(2, n + 1, 2)]\n"}


class TinyTokenizer:
    """和 Qwen3 起的模板一样：默认在生成提示后打开 <think>，除非显式关闭。"""
    chat_template = "{% if enable_thinking is defined and enable_thinking is false %}"
    pad_token = "<pad>"
    pad_token_id = 0
    CLOSED = "<think>\n\n</think>\n\n"

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True,
                            return_tensors=None, **kwargs):
        assert tokenize
        thinking = kwargs.get("enable_thinking", True)
        text = "".join(f"<{message['role']}>{message['content']}<end>\n"
                       for message in messages if message["role"] != "assistant")
        if add_generation_prompt:
            text += "<assistant><think>\n" if thinking else "<assistant>" + self.CLOSED
        else:
            text += "<assistant>" + self.CLOSED + messages[-1]["content"] + "<end>\n"
        ids = [ord(character) % 250 + 2 for character in text]
        return torch.tensor([ids]) if return_tensors == "pt" else ids

    def decode(self, tokens, skip_special_tokens=True):
        # 只含小写字母：随机权重下的输出不可能碰巧组成 <think>
        return "".join(chr(97 + int(token) % 26) for token in tokens)


@pytest.mark.parametrize("build,targets", [
    (tiny_qwen35, [*BASE_LORA_TARGETS, *HYBRID_LORA_TARGETS]),
    (tiny_qwen25, list(BASE_LORA_TARGETS)),
], ids=["qwen3.5", "qwen2.5"])
def test_smoke_test_runs_the_sft_path_on_a_tiny_model(build, targets):
    report = smoke_test(build(), TinyTokenizer(), MESSAGES, TARGET, lora_r=4, lora_alpha=8,
                        max_new_tokens=16)
    assert report["passed"], report
    assert all(report["checks"][name] for name in TRAINING_CHECKS)
    assert report["device"] == "cpu" and report["quantized"] is False
    assert report["lora"]["targets"] == targets
    # Qwen3.5：4 层里 3 层是线性注意力，所以混合投影各 3 个，q/k/v/o 各 1 个
    assert report["lora"]["modules_per_target"].get("in_proj_qkv", 0) == \
        (3 if build is tiny_qwen35 else 0)
    assert report["lora"]["modules_per_target"]["q_proj"] == \
        (1 if build is tiny_qwen35 else 2)
    # 训练用的是分段算的损失；冒烟时已经和模型自带的损失核对过
    assert report["train_step"]["loss_implementation"] == "chunked_lm_head"
    assert report["train_step"]["loss_check"]["difference"] <= max(1e-3, 1e-3 * abs(
        report["train_step"]["loss_check"]["reference"]))
    assert set(report["train_step"]["tensors_updated"]) == set(targets)
    assert all(count > 0 for count in report["train_step"]["tensors_updated"].values())
    assert report["generation"]["output_tokens"] > 0


def test_smoke_test_reports_a_sample_that_cannot_be_built():
    """chat template 的前缀对不上时，报告说明失败在构造样本，后面的步骤不再执行。"""
    class IncompatibleTokenizer(TinyTokenizer):
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True,
                                return_tensors=None, **kwargs):
            ids = super().apply_chat_template(messages, tokenize, add_generation_prompt,
                                              return_tensors, **kwargs)
            return [999] + ids if add_generation_prompt else ids

    report = smoke_test(tiny_qwen25(), IncompatibleTokenizer(), MESSAGES, TARGET)
    assert report["passed"] is False
    assert report["checks"] == {"sample_built": False}
    assert "prefix mismatch" in report["error"]


def test_smoke_test_adds_code_checks_only_when_given_an_extractor():
    model, tokenizer = tiny_qwen25(), TinyTokenizer()
    without = smoke_test(model, tokenizer, MESSAGES, TARGET, max_new_tokens=8)
    assert "code_parses" not in without["checks"]
    report = smoke_test(model, tokenizer, MESSAGES, TARGET, max_new_tokens=8,
                        extractor=lambda raw: "x = 1\n")
    assert report["checks"]["code_extracted"] and report["checks"]["code_parses"]
    assert report["passed"]
    # 没有提取到代码时不能因为“空字符串可解析”而算通过
    report = smoke_test(model, tokenizer, MESSAGES, TARGET, max_new_tokens=8,
                        extractor=lambda raw: "")
    assert not report["checks"]["code_extracted"] and not report["checks"]["code_parses"]
    assert report["passed"] is False
