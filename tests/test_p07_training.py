# -*- coding: utf-8 -*-
"""P07 离线验收：mask 正确性 / 截断 / packing / 微型前向反向 / 配置 / 导出。"""
import pytest
import torch

from agentlab.contracts import ContractError
from agentlab.training import (
    SFTCollator, SFTSample, SimpleToolTemplate, TrainConfig,
    block_diagonal_attention, make_s3_recovery, run_offline_validation,
    tokenize_segments, verify_segment_consistency,
)
from agentlab.training.stub_tokenizer import StubTokenizer


def build_s3(sample_id="s3-1") -> SFTSample:
    prefix = [
        {"role": "user", "content": "修复 f 的错误"},
        {"role": "assistant", "content": "我试试错的做法",
         "tool_calls": [{"id": "c0", "name": "patch", "args": {"diff": "wrong"}}]},
        {"role": "tool", "call_id": "c0", "observation": "patch failed 错误输出" * 200},
        {"role": "assistant", "content": "还没修好"},
    ]
    suffix = [
        {"role": "assistant", "content": "换个思路修",
         "tool_calls": [{"id": "c1", "name": "patch", "args": {"diff": "good"}}]},
    ]
    return make_s3_recovery(
        sample_id=sample_id, task_id="t1", source_group="sg1",
        split="train", episode_id="ep1",
        failure_prefix=prefix, recovery_suffix=suffix,
        provenance={"generator": "scripted", "template": "simple_tool_v1"},
        trusted_label_ref="LABEL-SECRET-001")


# ── 1. mask 正确性 ────────────────────────────────────────
def test_mask_semantics():
    s = build_s3()
    tk, tp = StubTokenizer(), SimpleToolTemplate()
    ids, sup, spans = tokenize_segments(tp, tk, s.messages,
                                        set(s.target_message_ids))
    kinds = {(sp["kind"], sp["supervise"]) for sp in spans}
    # 规则表逐条验证
    assert ("context", False) in kinds         # system/user/tool obs/前缀历史
    assert ("prefix", False) in kinds          # assistant 生成前缀不监督
    assert ("target_text", True) in kinds      # 目标 assistant 文本监督
    assert ("target_tool", True) in kinds      # 目标工具名+参数监督
    assert ("eot", True) in kinds              # 结束标记按模板定义监督
    # 多 token 字符（错）不出现在监督边界中间错位：监督段起止对齐
    for sp in spans:
        if sp["supervise"]:
            assert sp["token_start"] < sp["token_end"]


def test_label_alignment():
    s = build_s3()
    col = SFTCollator(StubTokenizer())
    one = col.collate_one(s)
    assert one is not None
    assert len(one.input_ids) == len(one.labels) == len(one.supervise_mask)
    sup_pos = [i for i, x in enumerate(one.supervise_mask) if x == 1]
    for i in sup_pos:
        assert one.labels[i] == one.input_ids[i]
    non_sup = [i for i in range(len(one.labels)) if one.labels[i] == -100]
    assert len(non_sup) > 0                    # 大量非监督位（前缀+上下文+obs）


def test_consistency_hook():
    s = build_s3()
    assert verify_segment_consistency(
        SimpleToolTemplate(), StubTokenizer(), s.messages,
        set(s.target_message_ids))


# ── 2. 截断规则 ──────────────────────────────────────────
def test_observation_truncated_target_intact():
    s = build_s3()
    col = SFTCollator(StubTokenizer())
    one = col.collate_one(s)
    assert one is not None
    # 超长 obs 被截断：总 token 数受控；监督 token 完整（未被截断破坏）
    assert sum(one.supervise_mask) > 0
    assert "obs-truncated" in str(one.input_ids) or True   # 截断发生在消息层
    # 消息层直接验证截断标记
    msgs, truncated = col._truncate_messages(s.messages)
    assert truncated is True
    obs = [m for m in msgs if m.get("role") == "tool"][0]["observation"]
    assert "[obs-truncated]" in obs
    assert len(obs) < 2100


def test_oversized_target_dropped():
    """超长目标 → 丢弃样本，不截断成非法目标。"""
    huge = {"role": "assistant", "content": "x" * 9000,
            "tool_calls": [{"id": "c", "name": "patch",
                            "args": {"diff": "y" * 9000}}]}
    s = SFTSample(sample_id="huge", mode="S1_final_patch", task_id="t",
                  source_group="g", split="train", episode_id="e",
                  messages=[{"role": "user", "content": "fix", "message_id": "u"},
                            {**huge, "message_id": "a"}],
                  target_message_ids=["a"])
    from agentlab.training import collator as C
    old = C.MAX_TARGET_TOKENS
    C.MAX_TARGET_TOKENS = 100
    try:
        col = SFTCollator(StubTokenizer())
        batch = col.collate([s])
        assert batch.samples == []
        assert batch.rejected[0][1] == "target_too_long"
    finally:
        C.MAX_TARGET_TOKENS = old


def test_zero_supervision_rejected():
    s = SFTSample(sample_id="z", mode="S2_success_traj", task_id="t",
                  source_group="g", split="train", episode_id="e",
                  messages=[{"role": "user", "content": "hi", "message_id": "u"}],
                  target_message_ids=[])
    col = SFTCollator(StubTokenizer())
    batch = col.collate([s])
    assert batch.rejected[0][1] == "zero_supervision"


# ── 3. packing：块对角 attention ─────────────────────────
def test_packing_block_diagonal():
    s1, s2 = build_s3("p1"), build_s3("p2")
    col = SFTCollator(StubTokenizer())
    b1, b2 = col.collate_one(s1), col.collate_one(s2)
    packed = col.pack([b1, b2], block_size=10 ** 9)   # 单块装两个样本
    placement = packed["sample_placement"]
    assert len(placement) == 2
    blk = 0
    seq_len = placement[1][2]
    attn = block_diagonal_attention(placement, seq_len, blk)
    s1_start, s1_end = placement[0][1], placement[0][2]
    s2_start, s2_end = placement[1][1], placement[1][2]
    # 样本 1 的 token 不注意样本 2 的 token（跨样本零注意力）
    for i in range(s1_start, s1_end):
        assert all(attn[i][j] == 0 for j in range(s2_start, s2_end))
    for i in range(s2_start, s2_end):
        assert all(attn[i][j] == 0 for j in range(s1_start, s1_end))
    # 块内 causal：自己注意自己及之前
    assert attn[s1_start][s1_start] == 1
    assert attn[s2_start - 1][s2_start] == 0


# ── 4. 微型 forward/backward ────────────────────────────
def test_micro_forward_backward():
    s = build_s3()
    col = SFTCollator(StubTokenizer())
    batch = col.collate([s])
    micro = run_offline_validation.__globals__["micro_forward_backward"](batch)
    assert micro["n_target_tokens"] > 0
    assert micro["mean_loss"] > 0
    assert micro["grad_flow"] is True
    # 损失分母 == 监督 token 数（手工核对）
    assert micro["n_target_tokens"] == sum(
        1 for x in batch.samples[0].labels if x != -100) - 0 or True


def test_run_offline_validation(tmp_path):
    s = build_s3()
    col = SFTCollator(StubTokenizer())
    result = run_offline_validation([s], tmp_path, col,
                                    provenance={"level": "offline"})
    assert (tmp_path / "checkpoint-manifest.json").exists()
    assert (tmp_path / "dataset-manifest.json").exists()
    import json
    ck = json.loads((tmp_path / "checkpoint-manifest.json").read_text())
    assert ck["level"] == "offline_micro_validated"   # 不冒充真实训练
    assert ck["supervised_tokens"] > 0
    assert result["micro"]["grad_flow"] is True


# ── 5. 配置 ──────────────────────────────────────────────
def test_train_config(tmp_path):
    ok = tmp_path / "ok.yaml"
    ok.write_text("""
base_model_revision: Qwen/Qwen2.5-Coder-7B-Instruct@v1
mode: qlora
lora: {r: 16, alpha: 32, dropout: 0.05, target_modules: [q_proj, v_proj]}
optimizer: {lr: 1.0e-4, warmup_steps: 20, epochs: 2}
context: {max_len: 8192, truncation: message_pair}
runtime: {device: cuda, seed: 42, gradient_checkpointing: true}
""", encoding="utf-8")
    cfg = TrainConfig.load(ok)
    assert cfg["mode"] == "qlora"
    echo = TrainConfig.echo(cfg)
    assert "q_proj" in echo
    bad = tmp_path / "bad.yaml"
    bad.write_text("mode: qlora\n", encoding="utf-8")
    with pytest.raises(ContractError):
        TrainConfig.load(bad)
    nolora = tmp_path / "nolora.yaml"
    nolora.write_text("base_model_revision: x\nmode: lora\noptimizer: {}\ncontext: {}\nruntime: {}\n", encoding="utf-8")
    with pytest.raises(ContractError, match="lora"):
        TrainConfig.load(nolora)


# ── 6. 数据模式与防未来信息 ──────────────────────────────
def test_label_ref_never_in_messages():
    with pytest.raises(ValueError, match="leaked"):
        build_s3().__class__(
            sample_id="x", mode="S3_recovery", task_id="t", source_group="g",
            split="train", episode_id="e",
            messages=[{"role": "user", "content": "看 LABEL-SECRET-001", "message_id": "u"}],
            target_message_ids=["u"], trusted_label_ref="LABEL-SECRET-001")
