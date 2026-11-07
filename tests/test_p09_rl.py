# -*- coding: utf-8 -*-
"""P09 离线验收：分组 / advantage / mask / GRPO loss / 零方差 / infra 排除 / 状态机。"""
import math

import pytest

from agentlab.rl import (
    AdvantageResult, GRPOStepResult, RLConfig, RLRunner,
    RolloutGroup, RolloutTrajectory, compute_group_advantage, grpo_step,
)


def make_traj(traj_id, group_id, task_id="t1", reward=1.0,
              term="done", policy="policy_v0", checkpoint=None) -> RolloutTrajectory:
    return RolloutTrajectory(
        traj_id=traj_id, group_id=group_id, task_id=task_id,
        source_group="sg1", split="train", episode_id=f"ep-{traj_id}",
        checkpoint_snapshot_id=checkpoint, policy_version=policy,
        reward=reward, termination_reason=term)


# ── 1. 分组契约 ──────────────────────────────────────────
def test_group_same_task_only():
    g = RolloutGroup(group_id="g1", task_id="t1", source_group="sg1",
                     split="train", checkpoint_snapshot_id=None,
                     policy_version="policy_v0")
    g.add(make_traj("a", "g1", "t1"))
    g.add(make_traj("b", "g1", "t1"))
    assert len(g.trajectories) == 2

    with pytest.raises(ValueError, match="task"):
        g.add(make_traj("c", "g1", "t2"))          # 不同任务 → 拒

    with pytest.raises(ValueError, match="policy"):
        g.add(make_traj("d", "g1", "t1", policy="policy_v1"))  # 跨版本 → 拒

    ckpt_g = RolloutGroup(group_id="g2", task_id="t1", source_group="sg1",
                          split="train", checkpoint_snapshot_id="snap-1",
                          policy_version="policy_v0")
    with pytest.raises(ValueError, match="checkpoint"):
        ckpt_g.add(make_traj("e", "g2", "t1", checkpoint=None))  # 不同恢复状态 → 拒


# ── 2. 组内 advantage ────────────────────────────────────
def test_advantage_mixed_rewards():
    g = RolloutGroup(group_id="g1", task_id="t1", source_group="sg1",
                     split="train", checkpoint_snapshot_id=None,
                     policy_version="policy_v0")
    g.add(make_traj("a", "g1", "t1", reward=1.0))
    g.add(make_traj("b", "g1", "t1", reward=0.0))
    g.add(make_traj("c", "g1", "t1", reward=1.0))
    g.add(make_traj("d", "g1", "t1", reward=0.0))
    adv = compute_group_advantage(g)
    assert adv.n_valid == 4
    assert not adv.zero_variance
    # mean=0.5, 1→正 adv, 0→负 adv
    assert adv.advantages[0] > 0 and adv.advantages[1] < 0
    # centering: 正负对称
    assert abs(sum(adv.advantages)) < 1e-6

def test_advantage_zero_variance():
    g = RolloutGroup(group_id="g1", task_id="t1", source_group="sg1",
                     split="train", checkpoint_snapshot_id=None,
                     policy_version="policy_v0")
    for i in range(4):
        g.add(make_traj(f"a{i}", "g1", "t1", reward=1.0))
    adv = compute_group_advantage(g)
    assert adv.zero_variance
    assert all(a == 0.0 for a in adv.advantages)
    assert "zero variance" in adv.note

def test_advantage_infra_excluded():
    g = RolloutGroup(group_id="g1", task_id="t1", source_group="sg1",
                     split="train", checkpoint_snapshot_id=None,
                     policy_version="policy_v0")
    g.add(make_traj("a", "g1", "t1", reward=1.0))
    g.add(make_traj("b", "g1", "t1", reward=0.0))
    g.add(make_traj("c", "g1", "t1", term="infra_error", reward=None))
    adv = compute_group_advantage(g)
    assert adv.n_valid == 2
    assert adv.n_infra == 1
    assert adv.n_total == 3


# ── 3. GRPO loss 与 mask ────────────────────────────────
def test_grpo_loss_basic():
    # 4 token，2 个动作 token（mask=1），advantage=1.0
    new = [-1.0, -2.0, -1.5, -0.5]
    old = [-1.0, -2.0, -1.5, -0.5]
    ref = [-1.0, -2.0, -1.5, -0.5]
    mask = [0, 0, 1, 1]     # 前两个是历史（mask=0），后两个是本次生成
    r = grpo_step(new, old, ref, advantage=1.0, action_mask=mask)
    assert r.n_action_tokens == 2
    assert r.mean_ratio == 1.0        # new==old → ratio=1
    assert r.clip_fraction == 0.0     # ratio=1 不触发裁剪
    assert r.mean_kl == 0.0           # new==ref → KL=0

def test_grpo_loss_sign():
    # advantage > 0：loss 应为负（鼓励增大概率）
    r_pos = grpo_step([-1.0], [-1.0], [-1.0], advantage=1.0, action_mask=[1])
    assert r_pos.loss < 0
    # advantage < 0：loss 应为正
    r_neg = grpo_step([-1.0], [-1.0], [-1.0], advantage=-1.0, action_mask=[1])
    assert r_neg.loss > 0

def test_grpo_mask_excludes_history():
    """历史 token（mask=0）不影响 loss——把历史 logp 改成任何值结果不变。"""
    mask = [0, 0, 0, 1, 1, 1]
    old = [-1, -2, -3, -1.5, -1.5, -1.5]
    ref = [-1, -2, -3, -1.5, -1.5, -1.5]
    new_a = [-99, -99, -99, -1.5, -1.5, -1.5]   # 历史 logp 乱改
    new_b = [-1, -1, -1, -1.5, -1.5, -1.5]       # 历史 logp 正常
    ra = grpo_step(new_a, old, ref, 1.0, mask)
    rb = grpo_step(new_b, old, ref, 1.0, mask)
    assert ra.loss == rb.loss              # loss 完全相同 → mask 有效

def test_grpo_zero_mask_returns_zero():
    r = grpo_step([-1], [-1], [-1], 1.0, [0])
    assert r.loss == 0.0 and r.n_action_tokens == 0

def test_grpo_clipping():
    """ratio 超出 [1-ε, 1+ε] → 裁剪生效。"""
    new = [-0.5]       # new >> old → ratio >> 1
    old = [-5.0]
    ref = [-5.0]
    r = grpo_step(new, old, ref, advantage=1.0, action_mask=[1], clip_eps=0.2)
    assert r.mean_ratio > 1.2
    assert r.clip_fraction == 1.0


# ── 4. 训练循环骨架 ──────────────────────────────────────
def test_rl_runner_step():
    """完整状态机：rollout→reward→advantage→update。"""
    def mock_rollout(task_id, n):
        trajs = []
        for i in range(n):
            t = make_traj(f"r{i}", "unused", task_id,
                          reward=1.0 if i < n // 2 else 0.0)
            t.old_logps = [[-1.0, -1.5, -2.0]]
            t.action_masks = [[0, 1, 1]]
            trajs.append(t)
        return trajs

    def mock_reward(t):
        return t.reward

    config = RLConfig(group_size=4)
    runner = RLRunner(config, mock_rollout, mock_reward)
    log = runner.train_step("t1", "sg1", "train")
    assert log.n_traj == 4
    assert log.n_valid == 4
    assert len(log.advantages) == 4
    assert len(log.losses) == 4
    assert any(a != 0 for a in log.advantages)    # 有 reward 方差

def test_rl_runner_all_same_reward():
    def mock_rollout(task_id, n):
        return [make_traj(f"r{i}", "g", task_id, reward=1.0) for i in range(n)]
    def mock_reward(t):
        return 1.0
    runner = RLRunner(RLConfig(group_size=4), mock_rollout, mock_reward)
    log = runner.train_step("t1", "sg1", "train")
    assert all(a == 0 for a in log.advantages)    # 零方差组 → 全零 adv


@pytest.mark.parametrize("field,value", [
    ("task_id", "other-task"),
    ("source_group", "other-source"),
    ("split", "test"),
    ("policy_version", "stale-policy"),
    ("checkpoint_snapshot_id", "other-checkpoint"),
])
@pytest.mark.parametrize("construction", ["constructor", "add"])
def test_group_rejects_mismatched_provenance(field, value, construction):
    trajectory = make_traj("mismatch", "g1")
    setattr(trajectory, field, value)
    arguments = dict(group_id="g1", task_id="t1", source_group="sg1",
                     split="train", checkpoint_snapshot_id=None,
                     policy_version="policy_v0")
    with pytest.raises(ValueError, match=field):
        if construction == "constructor":
            RolloutGroup(**arguments, trajectories=[trajectory])
        else:
            RolloutGroup(**arguments).add(trajectory)
    assert getattr(trajectory, field) == value


@pytest.mark.parametrize("field,value", [
    ("task_id", "other-task"),
    ("source_group", "other-source"),
    ("split", "test"),
    ("policy_version", "stale-policy"),
    ("checkpoint_snapshot_id", "other-checkpoint"),
    ("reference_version", "other-reference"),
])
def test_runner_preserves_and_rejects_mismatched_provenance(field, value):
    trajectory = make_traj("mismatch", "producer-group")
    setattr(trajectory, field, value)
    calls = []
    runner = RLRunner(RLConfig(group_size=1),
                      lambda task, count: [trajectory],
                      lambda sample: calls.append(sample) or 1.0)
    with pytest.raises(ValueError, match=field):
        runner.train_step("t1", "sg1", "train")
    assert getattr(trajectory, field) == value
    assert trajectory.group_id == "producer-group"
    assert calls == []
    assert runner.logs == []


def test_runner_math_check_modes_are_explicit():
    identity = make_traj("identity", "producer", reward=1.0)
    identity.old_logps = [[-1.0]]
    identity.action_masks = [[1]]
    provided = make_traj("provided", "producer", reward=0.0)
    provided.old_logps = [[-1.0]]
    provided.new_logps = [[-0.9]]
    provided.ref_logps = [[-1.1]]
    provided.action_masks = [[1]]
    runner = RLRunner(RLConfig(group_size=2),
                      lambda task, count: [identity, provided],
                      lambda sample: sample.reward)
    log = runner.train_step("t1", "sg1", "train")
    assert log.execution_mode == "math_check"
    assert log.update_applied is False
    assert log.logp_modes == ["identity_logps", "provided_logps"]
    assert log.loss_trajectory_ids == ["identity", "provided"]
    assert log.n_action_tokens == 2
    assert identity.new_logps is None
    assert identity.ref_logps is None


def test_runner_includes_all_turns_with_token_weighting():
    trajectory = make_traj("multi", "producer", reward=1.0)
    trajectory.old_logps = [[-99.0, -1.0], [-1.0, -1.0]]
    trajectory.new_logps = [[-5.0, -1.0], [-0.5, -2.0]]
    trajectory.ref_logps = [[-77.0, -1.0], [-1.0, -1.0]]
    trajectory.action_masks = [[0, 1], [1, 1]]
    failed = make_traj("failed", "producer", reward=0.0)
    runner = RLRunner(RLConfig(group_size=2),
                      lambda task, count: [trajectory, failed],
                      lambda sample: sample.reward)
    log = runner.train_step("t1", "sg1", "train")
    expected = grpo_step(
        [-5.0, -1.0, -0.5, -2.0], [-99.0, -1.0, -1.0, -1.0],
        [-77.0, -1.0, -1.0, -1.0], log.advantages[0], [0, 1, 1, 1])
    assert log.losses[0] == pytest.approx(expected.loss)
    assert log.n_action_tokens == 3
    trajectory.new_logps[1] = [-1.0, -1.0]
    changed = runner.train_step("t1", "sg1", "train")
    assert changed.losses[0] != pytest.approx(log.losses[0])


@pytest.mark.parametrize("field,value", [
    ("old_logps", []),
    ("old_logps", [[-1.0, -2.0]]),
    ("new_logps", [[-1.0]]),
    ("ref_logps", [[-1.0]]),
])
def test_runner_rejects_incomplete_or_misaligned_logps(field, value):
    trajectory = make_traj("shape", "producer")
    trajectory.old_logps = [[-1.0]]
    trajectory.action_masks = [[1]]
    setattr(trajectory, field, value)
    runner = RLRunner(RLConfig(group_size=1),
                      lambda task, count: [trajectory],
                      lambda sample: sample.reward)
    with pytest.raises(ValueError, match="logps"):
        runner.train_step("t1", "sg1", "train")


def test_runner_rejects_equal_flat_lengths_with_different_turn_boundaries():
    trajectory = make_traj("boundaries", "producer")
    trajectory.old_logps = [[-1.0], [-1.0, -1.0]]
    trajectory.action_masks = [[1, 1], [1]]
    runner = RLRunner(RLConfig(group_size=1),
                      lambda task, count: [trajectory],
                      lambda sample: sample.reward)
    with pytest.raises(ValueError, match="token count"):
        runner.train_step("t1", "sg1", "train")


@pytest.mark.parametrize("new,old,reference,mask", [
    ([-1.0], [], [-1.0], [1]),
    ([], [], [], [0]),
    ([-1.0], [-1.0], [-1.0], [2]),
    ([float("nan")], [-1.0], [-1.0], [1]),
])
def test_grpo_rejects_invalid_shapes_masks_and_probabilities(new, old, reference, mask):
    with pytest.raises(ValueError):
        grpo_step(new, old, reference, 1.0, mask)


@pytest.mark.parametrize("new,reference", [
    (-2.0, -1.0),
    (-1.0, -2.0),
    (-1.0, -1.0),
    (-1.0, -1.0 + 1e-8),
])
def test_grpo_sampled_kl_is_nonnegative(new, reference):
    result = grpo_step([new], [new], [reference], 0.0, [1])
    difference = reference - new
    assert result.mean_kl == pytest.approx(math.expm1(difference) - difference)
    assert result.mean_kl >= 0.0
    assert result.loss >= 0.0


def test_runner_keeps_unknown_costs_and_known_subtotal():
    known = make_traj("known", "producer")
    known.cost_usd = 0.25
    unknown = make_traj("unknown", "producer", term="infra_error")
    runner = RLRunner(RLConfig(group_size=2),
                      lambda task, count: [known, unknown],
                      lambda sample: sample.reward)
    mixed = runner.train_step("t1", "sg1", "train")
    assert mixed.total_cost is None
    assert mixed.known_cost_subtotal == 0.25
    assert mixed.unknown_cost_count == 1
    unknown.cost_usd = 0.1
    complete = runner.train_step("t1", "sg1", "train")
    assert complete.total_cost == pytest.approx(0.35)
    assert complete.unknown_cost_count == 0


def test_runner_handles_unscored_and_infrastructure_trajectories():
    pending = make_traj("pending", "producer", reward=None)
    infrastructure = make_traj("infra", "producer", term="infra_error", reward=None)
    scored_ids = []
    runner = RLRunner(RLConfig(group_size=2),
                      lambda task, count: [pending, infrastructure],
                      lambda sample: scored_ids.append(sample.traj_id))
    log = runner.train_step("t1", "sg1", "train")
    assert scored_ids == ["pending"]
    assert log.advantages == [0.0, 0.0]
    assert log.n_valid == 0
    assert log.losses == []
    assert log.zero_variance_group
    assert log.update_applied is False


def test_advantage_revalidates_mutated_group_members():
    trajectory = make_traj("a", "g1")
    group = RolloutGroup(
        group_id="g1", task_id="t1", source_group="sg1", split="train",
        checkpoint_snapshot_id=None, policy_version="policy_v0",
        trajectories=[trajectory])
    trajectory.policy_version = "stale-policy"
    with pytest.raises(ValueError, match="policy_version"):
        compute_group_advantage(group)


def test_advantage_revalidates_directly_appended_members():
    group = RolloutGroup(
        group_id="g1", task_id="t1", source_group="sg1", split="train",
        checkpoint_snapshot_id=None, policy_version="policy_v0")
    group.trajectories.append(make_traj("other", "g1", task_id="other-task"))
    with pytest.raises(ValueError, match="task_id"):
        compute_group_advantage(group)


def test_sampled_kl_expectation_matches_current_policy_kl():
    policy_probabilities = [0.7, 0.3]
    reference_probabilities = [0.4, 0.6]
    expected = 0.0
    sampled = 0.0
    for policy_probability, reference_probability in zip(
            policy_probabilities, reference_probabilities):
        policy_logp = math.log(policy_probability)
        reference_logp = math.log(reference_probability)
        result = grpo_step(
            [policy_logp], [policy_logp], [reference_logp], 0.0, [1])
        expected += policy_probability * (policy_logp - reference_logp)
        sampled += policy_probability * result.mean_kl
    assert sampled == pytest.approx(expected)
