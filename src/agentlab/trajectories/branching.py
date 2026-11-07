# -*- coding: utf-8 -*-
"""同失败状态分支采样引擎（branching.md 协议实现）。

核心语义：
- 候选组所有起始状态键一致（task/prefix/workspace/env/剩余预算）；
- 候选 y = 下一条完整 assistant 消息（含工具调用 JSON），不含工具返回；
- 执行候选后由冻结 continuation policy 续跑到终止；
- 分支互不污染（P02 快照恢复）、费用各自记账；
- branch record 保存全部 lineage 字段。
"""
from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from uuid import uuid4

from agentlab.contracts import Budgets, EpisodeConfig, Usage
from agentlab.runtime.budget_guard import BudgetExceeded, BudgetGuard
from agentlab.runtime.controlled import ControlledModelRuntime
from agentlab.runtime.snapshot import Snapshot
from agentlab.runtime.workspace import Workspace

CONTINUATION_POLICY_VERSION = "deterministic_v1"


def prefix_hash(public_prefix: list[dict]) -> str:
    blob = json.dumps(public_prefix, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


@dataclass
class BranchRecord:
    branch_id: str
    candidate_group_id: str
    parent_snapshot_id: str
    candidate_message: dict                       # 完整 assistant 消息（候选 y）
    generator_model_revision: str
    sampling_config: dict = field(default_factory=dict)
    continuation_policy_revision: str = CONTINUATION_POLICY_VERSION
    rollout_repetition: int = 1
    initial_budget: dict = field(default_factory=dict)
    candidate_cost: dict = field(default_factory=dict)
    continuation_cost: dict = field(default_factory=dict)
    termination_reason: str = ""
    final_verdict: str = ""                       # grader 结果（labels 侧引用）
    public_prefix_hash: str = ""
    workspace_hash: str = ""
    start_keys: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S%z"))

    def to_dict(self) -> dict:
        return dict(branch_id=self.branch_id,
                    candidate_group_id=self.candidate_group_id,
                    parent_snapshot_id=self.parent_snapshot_id,
                    candidate_message=self.candidate_message,
                    generator_model_revision=self.generator_model_revision,
                    sampling_config=self.sampling_config,
                    continuation_policy_revision=self.continuation_policy_revision,
                    rollout_repetition=self.rollout_repetition,
                    initial_budget=self.initial_budget,
                    candidate_cost=self.candidate_cost,
                    continuation_cost=self.continuation_cost,
                    termination_reason=self.termination_reason,
                    final_verdict=self.final_verdict,
                    public_prefix_hash=self.public_prefix_hash,
                    workspace_hash=self.workspace_hash,
                    start_keys=self.start_keys,
                    created_at=self.created_at)


class BranchEngine:
    """同一失败快照上执行多个候选分支。"""

    def __init__(self, work_root: Path, continuation_fn: Callable[[dict], dict]):
        """continuation_fn(ctx) -> assistant_message|finish：冻结续跑策略。"""
        self.work_root = work_root
        self.continuation_fn = continuation_fn

    def _start_keys(self, task_id: str, source_group: str, split: str,
                    snap: Snapshot, prefix: list[dict],
                    guard: BudgetGuard, budgets: Budgets) -> dict:
        """共同状态键：含真正计算的 remaining_budget（≠ consumed_budget）。"""
        consumed = snap.meta.get("consumed_budget", {})
        remaining = {}
        for dim, limit in (("model_requests", budgets.max_model_requests),
                           ("tool_calls", budgets.max_tool_calls),
                           ("wall_seconds", budgets.max_wall_seconds)):
            remaining[dim] = max(0, limit - consumed.get(dim, 0)) if limit is not None else None
        return dict(
            task_id=task_id, task_version=1,
            source_group=source_group, split=split,
            snapshot_id=snap.snapshot_id,
            public_prefix_hash=prefix_hash(prefix),
            workspace_hash=snap.meta["workspace_manifest_hash"],
            environment_digest=snap.meta["environment_digest"],
            remaining_budget=remaining,
            consumed_budget=consumed,
            continuation_policy_version=CONTINUATION_POLICY_VERSION)

    def run_group(self, task_id: str, source_group: str, split: str,
                  ws: Workspace, guard_seed_spent: dict,
                  public_prefix: list[dict],
                  candidates: list[Callable[[dict], dict]],
                  generator_revision: str = "scripted",
                  max_steps: int = 10,
                  budgets: Budgets | None = None) -> list[BranchRecord]:
        """在共同失败状态上执行候选组。

        ws 当前状态即失败状态；先做快照，然后每个候选从快照恢复独立分支。
        """
        budgets = budgets or Budgets(max_model_requests=30, max_tool_calls=60,
                                     max_wall_seconds=600)
        snap = Snapshot.create(
            ws, task_id=task_id, public_prefix=public_prefix,
            consumed_budget=guard_seed_spent,
            store_dir=self.work_root / "snapshots")
        group_id = f"cand-{snap.snapshot_id[:8]}-{uuid4().hex}"
        records = []
        for idx, cand_fn in enumerate(candidates):
            branch_ws_root = self.work_root / f"branch-{group_id}-{idx}"
            branch_ws = snap.restore(branch_ws_root)
            ep = EpisodeConfig(episode_id=f"br-{group_id}-{idx}",
                               task_id=task_id, attempt=1,
                               budgets=budgets)
            guard = BudgetGuard(ep.episode_id, budgets)
            for dim, amt in guard_seed_spent.items():
                guard.settle(f"seed-{dim}", dim, amt)
            start_keys = self._start_keys(task_id, source_group, split,
                                         snap, public_prefix, guard, budgets)
            seed_spent = dict(guard.ledger.spent)
            rt = ControlledModelRuntime(cand_fn, max_steps=1)
            msg = {}
            history = None
            termination_reason = "ok"
            try:
                msg = rt.generate(ep, guard, public_prefix)
                if not msg.get("finish"):
                    outcome = rt.step(ep, branch_ws, guard, 0, msg)
                    history = deepcopy(public_prefix) + [{
                        "role": "assistant", "content": msg.get("text", ""),
                        "tool_calls": msg.get("tool_calls", [])}]
                    history.extend({
                        "role": "tool", "observation": observation["output"],
                        "call_id": observation["call_id"]}
                        for observation in outcome.observations)
                    termination_reason = outcome.exit_status
            except BudgetExceeded:
                termination_reason = "budget_exceeded"
            after_candidate = dict(guard.ledger.spent)
            if history is not None and termination_reason == "ok":
                cont_rt = ControlledModelRuntime(self.continuation_fn,
                                                max_steps=max_steps)
                cont_result = cont_rt.run(ep, branch_ws, guard, history, start_step=1)
                termination_reason = cont_result["exit_status"]
            candidate_cost = {
                dim: after_candidate.get(dim, 0) - seed_spent.get(dim, 0)
                for dim in ("model_requests", "tool_calls")}
            continuation_cost = {
                dim: guard.ledger.spent.get(dim, 0) - after_candidate.get(dim, 0)
                for dim in ("model_requests", "tool_calls")}
            candidate_cost["usd"] = None
            continuation_cost["usd"] = None
            rec = BranchRecord(
                branch_id=f"br-{group_id}-{idx}",
                candidate_group_id=group_id,
                parent_snapshot_id=snap.snapshot_id,
                candidate_message=msg,
                generator_model_revision=generator_revision,
                initial_budget=dict(start_keys["remaining_budget"]),
                candidate_cost=candidate_cost,
                continuation_cost=continuation_cost,
                termination_reason=termination_reason,
                public_prefix_hash=prefix_hash(public_prefix),
                workspace_hash=snap.meta["workspace_manifest_hash"],
                start_keys=start_keys)
            records.append(rec)
        return records


def build_preference(records: list[BranchRecord],
                     verdicts: dict[str, str]) -> dict | None:
    """状态对齐动作偏好：x=共同公开历史，y±=候选消息（无工具返回）。

    verdicts: branch_id -> strict_pass|strict_fail（来自 grader labels）。
    平局/缺标签/低证据差异 → 返回 None（不硬造偏好）。
    """
    if len(records) < 2:
        return None
    keys0 = records[0].start_keys
    for r in records[1:]:
        if r.start_keys != keys0:
            return None              # 起始状态不一致 → 不构造
    v = [verdicts.get(r.branch_id) for r in records]
    if any(x is None for x in v):
        return None
    eligible = {"ok", "max_steps"}
    passes = [record for record, verdict in zip(records, v)
              if verdict == "strict_pass" and record.termination_reason in eligible]
    fails = [record for record, verdict in zip(records, v)
             if verdict == "strict_fail" and record.termination_reason in eligible]
    if not passes or not fails:
        return None                  # 全过/全挂 → 平局不硬造
    chosen = passes[0]
    rejected = fails[0]
    return dict(
        x_prefix_hash=records[0].public_prefix_hash,
        x_snapshot_id=records[0].parent_snapshot_id,
        y_chosen=chosen.candidate_message,
        y_rejected=rejected.candidate_message,
        chosen_branch=chosen.branch_id,
        rejected_branch=rejected.branch_id,
        rollout_repetitions=1,
        confidence_note="单次 rollout：标签有随机噪声，不作'稳定成功'表述",
        preference_reason="合法成功 vs 合法失败（同状态键）")
