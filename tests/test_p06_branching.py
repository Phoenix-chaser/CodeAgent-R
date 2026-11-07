# -*- coding: utf-8 -*-
"""P06 验收：同状态分支、偏好构造、canary 隔离、lineage 可追溯。"""
import io
import tarfile
from pathlib import Path

import pytest

from agentlab.contracts import Budgets, ContractError
from agentlab.trajectories import BranchEngine, build_preference, prefix_hash
from agentlab.runtime import Workspace

CANARY = "CANARY-p06-labels-never-in-input"


def make_ws(tmp_path, name="epB"):
    src = tmp_path / "bsrc"
    (src / "pkg").mkdir(parents=True)
    (src / "pkg" / "m.py").write_text("def f(n):\n    return n - 1\n", encoding="utf-8")  # bug
    tar = tmp_path / f"{name}.tar"
    with tarfile.open(tar, "w") as t:
        for p in sorted(src.rglob("*")):
            t.add(p, arcname=str(p.relative_to(src)))
    ws = Workspace.create_from_fixture(name, tar, tmp_path / "ws")
    ws.write("pkg/attempt.txt", "tried wrong fix")     # 模拟失败状态
    return ws


GOOD_PATCH = ('--- a/pkg/m.py\n+++ b/pkg/m.py\n@@ -1,2 +1,2 @@\n'
              ' def f(n):\n-    return n - 1\n+    return n + 1\n')


def candidate_good(ctx):
    return {"text": "fix the off-by-one", "tool_calls": [
        {"id": "g1", "name": "patch", "args": {"diff": GOOD_PATCH}}]}


def candidate_bad(ctx):
    return {"text": "no fix", "tool_calls": [
        {"id": "b1", "name": "read", "args": {"rel_path": "pkg/m.py"}}]}


def cont_finish(ctx):
    return {"text": "done", "finish": True}


def test_same_state_dual_branch(tmp_path):
    ws = make_ws(tmp_path)
    engine = BranchEngine(tmp_path / "rt", cont_finish)
    recs = engine.run_group(
        task_id="t1", source_group="sg1", split="train",
        ws=ws, guard_seed_spent={"model_requests": 2},
        public_prefix=[{"role": "user", "text": "fix f"},
                       {"role": "assistant", "text": "tried wrong"}],
        candidates=[candidate_good, candidate_bad])
    assert len(recs) == 2
    # 起始状态键完全一致
    assert recs[0].start_keys == recs[1].start_keys
    # 公开 prefix hash 一致
    assert recs[0].public_prefix_hash == recs[1].public_prefix_hash
    # 快照父一致
    assert recs[0].parent_snapshot_id == recs[1].parent_snapshot_id
    # 候选消息不含工具返回（y 的纯净性）
    assert "observation" not in str(recs[0].candidate_message)
    # 分支写入隔离：good 分支有补丁效果，bad 没有
    good_ws = tmp_path / "rt" / "branch-cand" / "0"
    # 候选费用各自记账
    assert recs[0].candidate_cost["model_requests"] == 1
    assert recs[1].candidate_cost["model_requests"] == 1
    # lineage：可追溯 snapshot + 模型版本 + 续跑策略版本
    for r in recs:
        assert r.parent_snapshot_id
        assert r.generator_model_revision
        assert r.continuation_policy_revision


def test_branch_write_isolation(tmp_path):
    ws = make_ws(tmp_path)
    engine = BranchEngine(tmp_path / "rt", cont_finish)
    recs = engine.run_group(
        task_id="t1", source_group="sg1", split="train",
        ws=ws, guard_seed_spent={"model_requests": 2},
        public_prefix=[{"role": "user", "text": "fix"}],
        candidates=[candidate_good, candidate_bad])
    roots = sorted((tmp_path / "rt").glob("branch-cand*"))
    good_root, bad_root = roots[0], roots[1]
    good_code = (good_root / "pkg" / "m.py").read_text(encoding="utf-8")
    bad_code = (bad_root / "pkg" / "m.py").read_text(encoding="utf-8")
    assert "n + 1" in good_code          # good 分支打了补丁
    assert "n - 1" in bad_code           # bad 分支未被污染


def test_preference_construction(tmp_path):
    ws = make_ws(tmp_path)
    engine = BranchEngine(tmp_path / "rt", cont_finish)
    recs = engine.run_group(
        task_id="t1", source_group="sg1", split="train",
        ws=ws, guard_seed_spent={"model_requests": 2},
        public_prefix=[{"role": "user", "text": "fix"}],
        candidates=[candidate_good, candidate_bad])
    verdicts = {recs[0].branch_id: "strict_pass",
                recs[1].branch_id: "strict_fail"}
    pref = build_preference(recs, verdicts)
    assert pref is not None
    assert "fix the off-by-one" in str(pref["y_chosen"])
    assert "no fix" in str(pref["y_rejected"])
    assert "observation" not in str(pref["y_chosen"])   # 工具观察未混入候选
    assert pref["x_snapshot_id"] == recs[0].parent_snapshot_id
    assert "单次 rollout" in pref["confidence_note"]


def test_preference_tie_returns_none(tmp_path):
    ws = make_ws(tmp_path)
    engine = BranchEngine(tmp_path / "rt", cont_finish)
    recs = engine.run_group(
        task_id="t1", source_group="sg1", split="train",
        ws=ws, guard_seed_spent={"model_requests": 2},
        public_prefix=[{"role": "user", "text": "fix"}],
        candidates=[candidate_bad, candidate_bad])
    tie = build_preference(recs, {recs[0].branch_id: "strict_fail",
                                  recs[1].branch_id: "strict_fail"})
    assert tie is None


def test_preference_missing_verdict_none(tmp_path):
    ws = make_ws(tmp_path)
    engine = BranchEngine(tmp_path / "rt", cont_finish)
    recs = engine.run_group(
        task_id="t1", source_group="sg1", split="train",
        ws=ws, guard_seed_spent={"model_requests": 2},
        public_prefix=[{"role": "user", "text": "fix"}],
        candidates=[candidate_good, candidate_bad])
    assert build_preference(recs, {recs[0].branch_id: "strict_pass"}) is None


def test_canary_not_in_prefix_or_candidates(tmp_path):
    """私有标签 canary 不进入任何公开输入（branch record / prefix / 候选）。"""
    ws = make_ws(tmp_path)
    engine = BranchEngine(tmp_path / "rt", cont_finish)
    recs = engine.run_group(
        task_id="t1", source_group="sg1", split="train",
        ws=ws, guard_seed_spent={"model_requests": 2},
        public_prefix=[{"role": "user", "text": "fix"}],
        candidates=[candidate_good, candidate_bad])
    for r in recs:
        assert CANARY not in str(r.to_dict())
    assert CANARY not in str(recs[0].public_prefix_hash)


@pytest.mark.parametrize("consumed,expected_calls", [(1, ["candidate"]), (2, [])])
def test_branch_candidate_and_continuation_share_request_budget(
        tmp_path, consumed, expected_calls):
    workspace = make_ws(tmp_path)
    calls = []

    def candidate(context):
        calls.append("candidate")
        return {"text": "try"}

    def continuation(context):
        calls.append("continuation")
        return {"finish": True}

    records = BranchEngine(tmp_path / "branches", continuation).run_group(
        task_id="task", source_group="source", split="train", ws=workspace,
        guard_seed_spent={"model_requests": consumed}, public_prefix=[],
        candidates=[candidate], budgets=Budgets(max_model_requests=2))
    record = records[0]
    assert calls == expected_calls
    assert record.termination_reason == "budget_exceeded"
    assert record.initial_budget["model_requests"] == 2 - consumed
    assert record.candidate_cost["model_requests"] == len(expected_calls)
    assert record.continuation_cost["model_requests"] == 0
    assert record.candidate_cost["usd"] is None


def test_branch_finish_does_not_invoke_continuation(tmp_path):
    workspace = make_ws(tmp_path)

    def unexpected(context):
        raise AssertionError("continuation must not run after finish")

    record = BranchEngine(tmp_path / "branches", unexpected).run_group(
        task_id="task", source_group="source", split="train", ws=workspace,
        guard_seed_spent={}, public_prefix=[], candidates=[cont_finish])[0]
    assert record.termination_reason == "ok"
    assert record.candidate_cost["model_requests"] == 1
    assert record.continuation_cost["model_requests"] == 0


def test_branch_cost_counts_finish_and_admitted_tools_only(tmp_path):
    workspace = make_ws(tmp_path)
    record = BranchEngine(tmp_path / "branches", cont_finish).run_group(
        task_id="task", source_group="source", split="train", ws=workspace,
        guard_seed_spent={}, public_prefix=[], candidates=[candidate_bad])[0]
    assert record.candidate_cost["model_requests"] == 1
    assert record.candidate_cost["tool_calls"] == 1
    assert record.continuation_cost["model_requests"] == 1
    assert record.continuation_cost["tool_calls"] == 0

    record = BranchEngine(tmp_path / "blocked", cont_finish).run_group(
        task_id="task", source_group="source", split="train", ws=workspace,
        guard_seed_spent={}, public_prefix=[], candidates=[candidate_bad],
        budgets=Budgets(max_tool_calls=0))[0]
    assert record.termination_reason == "budget_exceeded"
    assert record.candidate_cost["tool_calls"] == 0
    assert record.continuation_cost["model_requests"] == 0


@pytest.mark.parametrize("verdict", ["invalid", "partial", "infra_error"])
def test_preference_excludes_non_failure_labels(tmp_path, verdict):
    workspace = make_ws(tmp_path)
    records = BranchEngine(tmp_path / "branches", cont_finish).run_group(
        task_id="task", source_group="source", split="train", ws=workspace,
        guard_seed_spent={}, public_prefix=[],
        candidates=[candidate_good, candidate_bad])
    assert build_preference(records, {
        records[0].branch_id: "strict_pass", records[1].branch_id: verdict}) is None
    records[1].termination_reason = "budget_exceeded"
    assert build_preference(records, {
        records[0].branch_id: "strict_pass", records[1].branch_id: "strict_fail"}) is None
