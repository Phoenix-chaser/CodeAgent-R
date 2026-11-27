# -*- coding: utf-8 -*-
"""分段算损失：在 CPU 上用微型的真实架构，和模型自带的损失、梯度逐项比对。

显存上的效果（长样本不再放得下全部 logits）只能在 GPU 上量；这里能证明的是两件事：
分段的算法与自带损失等价，以及 lm_head 只处理被监督的位置、每段不超过 chunk。
"""
import sys
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3_5ForCausalLM"):
    pytest.skip("this transformers has no Qwen3.5", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentlab.training.hf_sft import evaluate_loss  # noqa: E402
from agentlab.training import sequence_loss  # noqa: E402
from tiny_models import tiny_qwen25, tiny_qwen35  # noqa: E402

BUILDS = [(tiny_qwen35, "qwen3.5"), (tiny_qwen25, "qwen2.5")]


def example(length=300, supervised_from=200, batch=1, seed=1):
    torch.manual_seed(seed)
    ids = torch.randint(0, 500, (batch, length))
    labels = ids.clone()
    labels[:, :supervised_from] = sequence_loss.IGNORE_INDEX
    return ids, labels


def reference_gradients(model, ids, labels):
    model.zero_grad(set_to_none=True)
    loss = model(input_ids=ids, attention_mask=None, labels=labels).loss
    loss.backward()
    return float(loss), {name: parameter.grad.detach().clone()
                         for name, parameter in model.named_parameters() if parameter.grad is not None}


@pytest.mark.parametrize("build, name", BUILDS, ids=[name for _, name in BUILDS])
def test_chunked_loss_matches_the_model_loss_and_its_gradients(build, name):
    model = build(torch.float32).train()
    ids, labels = example()
    check = sequence_loss.compare_losses(model, ids, None, labels)
    assert check["difference"] < 1e-6 and sequence_loss.consistent(check)
    loss, gradients = reference_gradients(model, ids, labels)
    model.zero_grad(set_to_none=True)
    chunked = sequence_loss.sequence_loss(model, ids, None, labels)      # 反向时逐段重算
    chunked.backward()
    assert abs(float(chunked) - loss) < 1e-6
    seen = 0
    for parameter_name, parameter in model.named_parameters():
        if parameter_name not in gradients:
            continue
        seen += 1
        assert torch.allclose(parameter.grad, gradients[parameter_name], rtol=1e-3, atol=1e-7), parameter_name
    assert seen == len(gradients) > 10                         # 每个拿到梯度的张量都比对过


@pytest.mark.parametrize("chunk", [1, 7, 10**6])
def test_chunk_size_does_not_change_the_loss(chunk):
    model = tiny_qwen25(torch.float32)
    ids, labels = example()
    plain = float(sequence_loss.sequence_loss(model, ids, None, labels, recompute=False))
    assert abs(float(sequence_loss.sequence_loss(model, ids, None, labels, chunk=chunk)) - plain) < 1e-6
    with pytest.raises(ValueError, match="positive"):
        sequence_loss.sequence_loss(model, ids, None, labels, chunk=0)


def test_only_supervised_positions_reach_the_lm_head_in_bounded_pieces():
    model = tiny_qwen25(torch.float32)
    calls = []
    original = model.lm_head

    class Recorder(torch.nn.Module):
        def forward(self, hidden):
            calls.append(hidden.shape[0])
            return original(hidden)

    model.lm_head = Recorder()
    ids, labels = example(length=300, supervised_from=200)
    sequence_loss.sequence_loss(model, ids, None, labels, chunk=64)
    assert calls == [64, 36]                                     # 100 个被监督的位置，不是 300 个
    assert sum(calls) == 300 - 200


def test_prompt_only_labels_give_a_differentiable_zero():
    model = tiny_qwen25(torch.float32)
    ids, labels = example(length=40, supervised_from=40)
    loss = sequence_loss.sequence_loss(model, ids, None, labels)
    assert float(loss) == 0 and loss.requires_grad
    loss.backward()                                             # 没有被监督的 token 也不报错


def test_padding_is_ignored_exactly_as_in_the_model_loss():
    model = tiny_qwen25(torch.float32)
    ids, labels = example(length=120, supervised_from=80, batch=2)
    labels[1, 100:] = sequence_loss.IGNORE_INDEX
    ids[1, 100:] = 0
    mask = torch.ones_like(ids)
    mask[1, 100:] = 0
    check = sequence_loss.compare_losses(model, ids, mask, labels)
    assert check["difference"] < 1e-6


def test_bf16_autocast_stays_within_float_noise():
    model = tiny_qwen35(torch.float32).train()
    ids, labels = example(length=120, supervised_from=80)
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        check = sequence_loss.compare_losses(model, ids, None, labels)
    assert check["reference"] > 1 and sequence_loss.consistent(check)


def test_lora_wrapped_models_are_unwrapped_to_the_base_model():
    model = tiny_qwen25(torch.float32)

    class Wrapper(torch.nn.Module):
        def get_base_model(self):
            return model

    ids, labels = example()
    outer = Wrapper()
    assert sequence_loss.split_lm(outer) == (model.model, model.lm_head)
    assert float(sequence_loss.sequence_loss(outer, ids, None, labels)) == \
        float(sequence_loss.sequence_loss(model, ids, None, labels))

    class Plain(torch.nn.Module):
        pass

    with pytest.raises(ValueError, match="--no-chunked-loss"):
        sequence_loss.split_lm(Plain())


def test_consistency_threshold_separates_float_noise_from_real_mistakes():
    assert sequence_loss.consistent({"reference": 6.2, "chunked": 6.2006, "difference": 0.0006})
    assert not sequence_loss.consistent({"reference": 0.03, "chunked": 0.05, "difference": 0.02})


def test_validation_loss_is_the_same_whichever_way_it_is_computed():
    model = tiny_qwen25(torch.float32).train()
    ids, labels = example(length=90, supervised_from=60, batch=3)
    batches = [{"input_ids": ids[:1], "attention_mask": torch.ones_like(ids[:1]), "labels": labels[:1]},
               {"input_ids": ids[1:], "attention_mask": torch.ones_like(ids[1:]), "labels": labels[1:]}]
    plain = evaluate_loss(model, batches, torch.device("cpu"))
    chunked = evaluate_loss(model, batches, torch.device("cpu"), chunked=True)
    assert abs(plain["loss"] - chunked["loss"]) < 1e-6 and plain["tokens"] == chunked["tokens"]
    assert model.training                                           # 评完恢复训练状态
