# -*- coding: utf-8 -*-
"""P08 离线验收：DPO 数据契约 / 共同状态校验 / token 概率 / reference 冻结 / loss 符号。"""
import math

import pytest
import torch

from agentlab.dpo import (
    DPOCollator, PreferencePair, check_pair_state_consistency, dpo_loss,
    dpo_loss_length_normalized, make_pair, micro_dpo_step, sequence_logp,
    verify_no_future_info_in_x, verify_no_tool_observation_in_y,
)
from agentlab.training.stub_tokenizer import StubTokenizer
from agentlab.training.template import SimpleToolTemplate


def make_valid_pair() -> PreferencePair:
    shared = dict(
        task_id="t1", source_group="sg1", split="train",
        public_prefix_hash="abc123", snapshot_id="snap-001",
        workspace_hash="ws-hash-1", environment_digest="env-1",
        tool_schema_hash="ts-1", remaining_budget={"model_requests": 28},
        continuation_policy_version="det_v1")
    return make_pair(
        x_messages=[{"role": "user", "content": "修复 bug"},
                    {"role": "assistant", "content": "试过错的"},
                    {"role": "tool", "call_id": "c0", "observation": "exit 1"}],
        y_chosen={"role": "assistant", "content": "改对了",
                  "tool_calls": [{"id": "g1", "name": "patch",
                                  "args": {"diff": "+correct"}}]},
        y_rejected={"role": "assistant", "content": "没修",
                    "tool_calls": [{"id": "b1", "name": "read",
                                    "args": {"rel_path": "x.py"}}]},
        shared_keys=shared, chosen_verdict="strict_pass",
        rejected_verdict="strict_fail", preference_reason="success_vs_failure")


# ── 1. 数据契约 ──────────────────────────────────────────
def test_valid_pair_constructs():
    p = make_valid_pair()
    assert p.chosen_verdict == "strict_pass"


def test_missing_shared_keys_rejected():
    shared = dict(task_id="t1")     # 缺 9 个键
    with pytest.raises(ValueError, match="missing"):
        make_pair(x_messages=[], y_chosen={}, y_rejected={},
                  shared_keys=shared, chosen_verdict="strict_pass",
                  rejected_verdict="strict_fail")


def test_failure_as_chosen_rejected():
    p = make_valid_pair()
    with pytest.raises(ValueError, match="strict_pass"):
        PreferencePair(
            pair_id="x", x_messages=p.x_messages,
            y_chosen=p.y_chosen, y_rejected=p.y_rejected,
            shared_keys=p.shared_keys,
            chosen_verdict="strict_fail",       # 不能失败当正样本
            rejected_verdict="strict_fail")


def test_cost_only_preference_rejected():
    p = make_valid_pair()
    with pytest.raises(ValueError, match="cost-only"):
        PreferencePair(
            pair_id="x", x_messages=p.x_messages,
            y_chosen=p.y_chosen, y_rejected=p.y_rejected,
            shared_keys=p.shared_keys, chosen_verdict="strict_pass",
            rejected_verdict="strict_pass",
            preference_reason="cost_only")    # 成功表现未匹配就比费用 → 拒


def test_state_consistency_check():
    p = make_valid_pair()
    same = dict(p.shared_keys)
    assert check_pair_state_consistency(p, same) == []
    diff = {**p.shared_keys, "snapshot_id": "snap-OTHER"}
    diffs = check_pair_state_consistency(p, diff)
    assert "snapshot_id" in diffs


# ── 2. y 纯净性 ──────────────────────────────────────────
def test_y_no_tool_observation():
    p = make_valid_pair()
    assert verify_no_tool_observation_in_y(p.y_chosen)
    assert verify_no_tool_observation_in_y(p.y_rejected)
    dirty = {**p.y_chosen, "observation": "tool returned x"}
    assert not verify_no_tool_observation_in_y(dirty)


def test_x_no_future_info():
    p = make_valid_pair()
    assert verify_no_future_info_in_x(p.x_messages)
    future = p.x_messages + [{"role": "user", "content": "strict_pass!"}]
    assert not verify_no_future_info_in_x(future)


# ── 3. collator：x/y 分离、候选 token mask ───────────────
def test_dpo_collator():
    p = make_valid_pair()
    col = DPOCollator(StubTokenizer())
    chosen, rejected = col.collate_pair(p)
    # x 部分全 -100
    x_len = len(chosen.input_ids) - chosen.n_target_tokens
    assert all(l == -100 for l in chosen.labels[:x_len])
    # y 部分有监督 token
    assert chosen.n_target_tokens > 0
    assert rejected.n_target_tokens > 0
    # chosen 与 rejected 的 x 前缀完全一致（同条件）
    assert chosen.input_ids[:x_len] == rejected.input_ids[:x_len]
    # chosen/rejected 的 y 部分不同
    assert chosen.input_ids[x_len:] != rejected.input_ids[x_len:]


def test_dpo_collator_matches_complete_template():
    pair = make_valid_pair()
    tokenizer = StubTokenizer()
    template = SimpleToolTemplate()
    sequences = DPOCollator(tokenizer, template).collate_pair(pair)
    for sequence, candidate in zip(sequences, (pair.y_chosen, pair.y_rejected)):
        messages = [
            {**message, "message_id": f"x-{index}"}
            for index, message in enumerate(pair.x_messages)
        ] + [{**candidate, "message_id": "y-target"}]
        rendered = template.render(messages, {"y-target"})
        full_text = "".join(segment.text for segment in rendered)
        assert sequence.input_ids == tokenizer.tokenize(full_text)
        assert sequence.n_target_tokens == sum(
            len(tokenizer.tokenize(segment.text))
            for segment in rendered if segment.supervise
        )


# ── 4. DPO loss：符号正确 / reference 冻结 ────────────────
def test_dpo_loss_signs():
    # case 1: chosen 比 rejected 好（margin 正） → loss 小
    good = dpo_loss(logp_chosen=-10, logp_rejected=-20,
                    ref_logp_chosen=-10, ref_logp_rejected=-10, beta=0.1)
    assert good["reward_margin"] > 0
    assert good["loss"] < 0.7       # -log σ(正) < -log(0.5) = 0.693
    # case 2: rejected 更好 → loss 大
    bad = dpo_loss(logp_chosen=-20, logp_rejected=-10,
                   ref_logp_chosen=-10, ref_logp_rejected=-10, beta=0.1)
    assert bad["reward_margin"] < 0
    assert bad["loss"] > 0.7
    # case 3: 完全相同 → loss = -log(0.5) ≈ 0.693
    tie = dpo_loss(-10, -10, -10, -10, beta=0.1)
    assert abs(tie["loss"] - math.log(2)) < 1e-4


@pytest.mark.parametrize("delta", [-1000.0, -100.0, 100.0, 1000.0])
def test_dpo_extreme_margins_have_finite_loss_and_gradients(delta):
    chosen = torch.tensor(delta, requires_grad=True)
    rejected = torch.tensor(0.0, requires_grad=True)
    result = dpo_loss(chosen, rejected, 0.0, 0.0, beta=1.0)
    result["loss"].backward()
    expected = max(-delta, 0.0) + math.log1p(math.exp(-abs(delta)))
    assert result["loss"].item() == pytest.approx(expected, abs=1e-6)
    assert dpo_loss(delta, 0.0, 0.0, 0.0, beta=1.0)["loss"] == pytest.approx(expected)
    assert torch.isfinite(chosen.grad)
    assert torch.isfinite(rejected.grad)
    if delta < 0:
        assert chosen.grad.item() == pytest.approx(-1.0)
        assert rejected.grad.item() == pytest.approx(1.0)


def test_dpo_reference_logps_do_not_receive_gradients():
    chosen = torch.tensor(-10.0, requires_grad=True)
    rejected = torch.tensor(-12.0, requires_grad=True)
    reference_chosen = torch.tensor(-10.0, requires_grad=True)
    reference_rejected = torch.tensor(-10.0, requires_grad=True)
    result = dpo_loss(chosen, rejected, reference_chosen, reference_rejected)
    result["loss"].backward()
    assert chosen.grad.abs().item() > 0
    assert rejected.grad.abs().item() > 0
    assert reference_chosen.grad is None
    assert reference_rejected.grad is None


def test_length_normalized_separate():
    r = dpo_loss_length_normalized(-100, -10, -100, -10, 20, 5)
    assert r["variant"] == "length_normalized_v1"   # 与标准版分开命名


def test_sequence_logp_masks():
    logits = torch.randn(10, 50, requires_grad=True)
    labels = torch.tensor([-100, -100, 3, 7, -100, -100, 12, -100, -100, -100])
    lp = sequence_logp(logits, labels)
    # 返回 tensor 且只有 3 个监督 token 参与
    assert isinstance(lp, torch.Tensor)
    expected = (
        logits[1].log_softmax(-1)[3]
        + logits[2].log_softmax(-1)[7]
        + logits[5].log_softmax(-1)[12]
    )
    torch.testing.assert_close(lp, expected)
    lp.backward()
    assert logits.grad[[1, 2, 5]].abs().sum().item() > 0
    assert logits.grad[[0, 3, 4, 6, 7, 8, 9]].count_nonzero().item() == 0


def test_sequence_logp_all_masked_preserves_device_dtype_and_graph():
    logits = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
    labels = torch.full((4,), -100, dtype=torch.long, device=logits.device)
    result = sequence_logp(logits, labels)
    assert result.item() == 0.0
    assert result.dtype == logits.dtype
    assert result.device == logits.device
    result.backward()
    assert logits.grad is not None
    assert logits.grad.count_nonzero().item() == 0


def test_micro_dpo_step():
    torch.manual_seed(42)
    chosen = dict(input_ids=[10, 20, 30, 40, 50],
                  labels=[-100, -100, 30, 40, 50], ref_logp=-15.0)
    rejected = dict(input_ids=[10, 20, 35, 45, 55],
                    labels=[-100, -100, 35, 45, 55], ref_logp=-15.0)
    result = micro_dpo_step([chosen, rejected])
    assert result["loss"].requires_grad
    assert math.isfinite(result["loss_value"])
    assert math.isfinite(result["grad_norm"])
    assert result["grad_norm"] > 0
    assert result["params_changed"] is True
    assert result["reference_frozen"] is True
    assert result["policy_logp"]["chosen"] != result["policy_logp"]["rejected"]


def test_micro_dpo_step_keeps_reference_tensors_fixed():
    reference_chosen = torch.tensor(-10.0, requires_grad=True)
    reference_rejected = torch.tensor(-12.0, requires_grad=True)
    chosen = dict(input_ids=[1, 2, 3], labels=[-100, 2, 3],
                  ref_logp=reference_chosen)
    rejected = dict(input_ids=[1, 4, 5], labels=[-100, 4, 5],
                    ref_logp=reference_rejected)
    result = micro_dpo_step([chosen, rejected])
    assert result["reference_frozen"] is True
    assert reference_chosen.item() == -10.0
    assert reference_rejected.item() == -12.0
    assert reference_chosen.grad is None
    assert reference_rejected.grad is None


def test_micro_dpo_step_needs_two():
    with pytest.raises(ValueError, match=">= 2"):
        micro_dpo_step([dict(input_ids=[1], labels=[1], ref_logp=0)])
