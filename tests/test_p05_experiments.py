# -*- coding: utf-8 -*-
"""P05 验收：Mock 记录驱动的报表管线口径（对照 devkit VALIDATION 前向试用数值）。"""
import json
from pathlib import Path

import pytest

from agentlab.contracts import ContractError
from agentlab.experiments import (
    ExperimentContract, load_records, render_markdown, summarize,
)


@pytest.fixture
def contract() -> ExperimentContract:
    return ExperimentContract.load("contracts/experiments/pilot-baseline.v1.yaml")


def test_contract_loads_in_draft(contract):
    assert contract.status == "draft"
    assert contract.max_real_episodes == 0


def test_contract_validation(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: 1\nexperiment_id: x\n", encoding="utf-8")
    with pytest.raises(ContractError):
        ExperimentContract.load(bad)


def _mock_records() -> list[dict]:
    """4 任务 × 2 来源组 × 3 次重复 = 12 条 Mock（含 1 infra 失败、1 未知费用）。

    期望（对照 devkit 前向试用口径）：
      宏成功率 50%（2/4）；排除 infra 后 2/4+1=… 按实现口径计算；
      已知费用 0.11；未知 1 条；无对照无提升。
    """
    rows = []
    def add(task, group, rep, verdict, cost, exit_status="ok"):
        rows.append(dict(task_id=task, source_group=group, repeat=rep,
                         origin="mock", verdict=verdict, cost_usd=cost,
                         exit_status=exit_status))
    # task1：3 次全过（宏成功）
    for r in (1, 2, 3):
        add("t1", "g1", r, "strict_pass", 0.01)
    # task2：全失败
    for r in (1, 2, 3):
        add("t2", "g1", r, "strict_fail", 0.02)
    # task3：两次失败 + 一次 infra（task 不因 infra 移出分母，因为非全部 infra）
    add("t3", "g2", 1, "strict_fail", 0.03)
    add("t3", "g2", 2, "strict_fail", 0.04)
    add("t3", "g2", 3, "infra_placeholder", None, exit_status="timeout")
    # task4：一次过 + 一次未知费用失败 + 一次 infra
    add("t4", "g2", 1, "strict_pass", 0.05)
    add("t4", "g2", 2, "strict_fail", None)
    add("t4", "g2", 3, "x", None, exit_status="aborted")
    return rows


def test_reporting_rules(tmp_path):
    logs = tmp_path / "runs.jsonl"
    logs.write_text("\n".join(json.dumps(r) for r in _mock_records()),
                    encoding="utf-8")
    records = load_records(logs)
    assert len(records) == 12
    reports = summarize(records, contract_max_real_episodes=0,
                        has_control=False)
    rep = reports["mock"]
    # 宏成功：t1、t4 → 2/4 = 50%
    assert rep.n_tasks == 4
    assert rep.macro_success == 0.5
    # 排除 infra：t3 有非 infra run，仍在分母 → 仍 2/4
    assert rep.macro_success_excl_infra == 0.5
    # pooled 排除 infra：pass 3 / 非 infra 10
    assert rep.pooled_excl_infra == "4/10"  # t1×3 + t4×1 = 4 过，非 infra 10 次
    # 费用：已知 0.01*3+0.02*3+0.03+0.04+0.05 = 0.21；未知 3 条（1 infra+1 unknown+1 aborted）
    assert abs(rep.known_cost_total_usd - 0.21) < 1e-9
    assert rep.unknown_cost_count == 3
    assert rep.infra_failures == 2           # timeout + aborted
    # 无对照 → 无 lift；来源组 2 < 5 → 声明不稳定
    assert rep.lift is None
    assert any("无对照组" in n for n in rep.notes)
    assert any("来源组仅 2" in n for n in rep.notes)
    md = render_markdown(reports)
    assert "MOCK" in md and "不代表模型能力" in md


def test_unauthorized_real_records_rejected(tmp_path):
    rows = _mock_records()[:1]
    rows[0]["origin"] = "real"
    logs = tmp_path / "real.jsonl"
    logs.write_text(json.dumps(rows[0]), encoding="utf-8")
    with pytest.raises(ContractError, match="authorization"):
        summarize(load_records(logs), contract_max_real_episodes=0)
