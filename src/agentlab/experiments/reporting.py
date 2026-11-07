# -*- coding: utf-8 -*-
"""原始运行记录 → 可信汇总报表（evidence-rules 口径）。

硬规则：
- Mock/真实/回放必须分开，报表显式标注 origin，Mock 不产生模型能力结论；
- 基础设施失败（infra_error）从任务宏成功率分母中排除，但单独列出计数；
- 宏平均（每任务一票）与 pooled（每次运行一票）分开报告；
- unknown 费用保持 unknown（None），绝不填 0，也不计入已知费用小计；
- 没有对照组 → 不生成任何提升（lift）字段；
- 少数来源组 → 不给置信区间，只给描述统计并声明不稳定；
- 授权为 0 时出现真实 episode 记录 → 拒绝出报表（ContractError）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from agentlab.contracts import ContractError

INFRA_FAILURE_STATUSES = {"timeout", "aborted", "budget_exceeded"}
ORIGIN_MOCK = "mock"
ORIGIN_REAL = "real"


@dataclass
class RunRecord:
    task_id: str
    source_group: str
    repeat: int
    origin: str                 # mock | real | replay
    verdict: str                # strict_pass/partial/strict_fail/invalid/infra_error
    cost_usd: float | None = None
    exit_status: str = "ok"

    @classmethod
    def from_dict(cls, d: dict) -> "RunRecord":
        required = ("task_id", "source_group", "repeat", "origin", "verdict")
        missing = [k for k in required if k not in d]
        if missing:
            raise ValueError(f"run record missing fields: {missing}")
        return cls(task_id=d["task_id"], source_group=d["source_group"],
                   repeat=d["repeat"], origin=d["origin"], verdict=d["verdict"],
                   cost_usd=d.get("cost_usd"), exit_status=d.get("exit_status", "ok"))


def load_records(path: Path) -> list[RunRecord]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(RunRecord.from_dict(json.loads(line)))
    return records


@dataclass
class Report:
    origin: str
    n_runs: int = 0
    n_tasks: int = 0
    macro_success: float | None = None
    macro_success_excl_infra: float | None = None
    pooled_success: float | None = None
    pooled_excl_infra: str | None = None
    known_cost_total_usd: float | None = None
    unknown_cost_count: int = 0
    infra_failures: int = 0
    n_source_groups: int = 0
    lift: dict | None = None
    notes: list[str] = field(default_factory=list)


def summarize(records: list[RunRecord], contract_max_real_episodes: int = 0,
              has_control: bool = False) -> dict[str, Report]:
    """按 origin 分组汇总；真实记录需授权，否则拒绝。"""
    by_origin: dict[str, list[RunRecord]] = {}
    for r in records:
        by_origin.setdefault(r.origin, []).append(r)
    reports: dict[str, Report] = {}
    for origin, rs in by_origin.items():
        if origin == ORIGIN_REAL and contract_max_real_episodes <= 0:
            raise ContractError(
                "real-episode records present but authorization.max_real_episodes=0; "
                "refusing to report unauthorized data")
        rep = Report(origin=origin, n_runs=len(rs))
        tasks = {}
        for r in rs:
            tasks.setdefault(r.task_id, []).append(r)
        rep.n_tasks = len(tasks)
        # 基础设施失败：exit_status 属 infra 类且 verdict 非 strict_pass
        def is_infra(r: RunRecord) -> bool:
            return r.exit_status in INFRA_FAILURE_STATUSES and \
                r.verdict != "strict_pass"
        # 宏平均（每任务一票：任一次 strict_pass 记成功）
        task_ok = [any(r.verdict == "strict_pass" for r in runs)
                   for runs in tasks.values()]
        rep.macro_success = (sum(task_ok) / len(task_ok)) if task_ok else None
        # 排除 infra 后的宏平均：整任务所有 run 都是 infra 才移出分母
        eff_tasks = [runs for runs in tasks.values()
                     if not all(is_infra(r) for r in runs)]
        eff_ok = [any(r.verdict == "strict_pass" for r in runs)
                  for runs in eff_tasks]
        rep.infra_failures = sum(1 for r in rs if is_infra(r))
        rep.macro_success_excl_infra = (
            sum(eff_ok) / len(eff_ok)) if eff_ok else None
        # pooled（每次运行一票）
        rep.pooled_success = (sum(1 for r in rs if r.verdict == "strict_pass")
                              / len(rs)) if rs else None
        ok_runs = [r for r in rs if not is_infra(r)]
        n_pass = sum(1 for r in ok_runs if r.verdict == "strict_pass")
        rep.pooled_excl_infra = f"{n_pass}/{len(ok_runs)}" if ok_runs else None
        # 费用：unknown 独立计数，已知单独小计
        known = [r.cost_usd for r in rs if r.cost_usd is not None]
        rep.known_cost_total_usd = sum(known) if known else 0.0
        rep.unknown_cost_count = sum(1 for r in rs if r.cost_usd is None)
        rep.n_source_groups = len({r.source_group for r in rs})
        # 注记规则
        if origin == ORIGIN_MOCK:
            rep.notes.append("MOCK 数据：不代表模型能力或真实计费，仅验证报表口径")
        if not has_control:
            rep.notes.append("无对照组：不生成提升幅度")
        if rep.n_source_groups < 5:
            rep.notes.append(f"来源组仅 {rep.n_source_groups} 个：置信区间不稳定，只报描述统计")
        reports[origin] = rep
    return reports


def render_markdown(reports: dict[str, Report]) -> str:
    lines = ["# 实验汇总报表", ""]
    for origin, rep in sorted(reports.items()):
        tag = "（MOCK）" if origin == ORIGIN_MOCK else ""
        lines += [f"## {origin}{tag}", ""]
        if rep.macro_success is not None:
            lines.append(f"- 端到端任务宏成功率：{rep.macro_success:.2%}")
        if rep.macro_success_excl_infra is not None:
            lines.append(f"- 排除基础设施失败后的任务宏平均：{rep.macro_success_excl_infra:.2%}")
        if rep.pooled_excl_infra:
            lines.append(f"- 对应 pooled 比例（排除 infra）：{rep.pooled_excl_infra}（与宏平均分开）")
        lines.append(f"- 已知费用小计：USD {rep.known_cost_total_usd:.2f}"
                     if rep.known_cost_total_usd else "- 已知费用小计：USD 0.00")
        lines.append(f"- 未知费用：{rep.unknown_cost_count} 条（保持 unknown，不计入小计）")
        lines.append(f"- 基础设施失败：{rep.infra_failures} 条")
        for n in rep.notes:
            lines.append(f"- ⚠ {n}")
        lines.append("")
    return "\n".join(lines)
