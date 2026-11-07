"""Compare matched evaluation runs without conflating pass@1 and any-pass."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path


def load_results(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def task_attempts(data: dict) -> dict[str, dict[int, bool]]:
    grouped: dict[str, dict[int, bool]] = {}
    for record in data.get("results", []):
        task_id = record.get("task_id")
        if not task_id:
            raise ValueError("Result row is missing task_id")
        attempts = grouped.setdefault(task_id, {})
        index = record.get("repeat_index", len(attempts))
        if index in attempts:
            raise ValueError(f"Duplicate task/repeat: {task_id}/{index}")
        passed = record.get("strict_pass", False)
        if not isinstance(passed, bool):
            raise ValueError(f"strict_pass is not a boolean: {task_id}/{index}")
        attempts[index] = passed
    if not grouped:
        raise ValueError("No task results")
    return grouped


def validate_report_protocol(report: dict, tasks: dict) -> None:
    protocol = report.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("Modern report is missing protocol")
    digest = hashlib.sha256(json.dumps(protocol, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    if digest != report.get("protocol_hash"):
        raise ValueError("Stored protocol_hash does not match protocol")
    if set(report.get("task_ids", [])) != set(tasks) or set(report["task_hashes"]) != set(tasks):
        raise ValueError("Declared task identities do not match result rows")
    if protocol.get("generation") != report["generation"] or protocol.get("split") != report.get("split"):
        raise ValueError("Protocol generation/split metadata differs from report")
    if protocol.get("benchmark_hash") != report["tasks_hash"]:
        raise ValueError("Protocol benchmark hash differs from report")
    prompts = protocol.get("prompt_hashes", {})
    if set(prompts) != set(tasks):
        raise ValueError("Prompt identities do not match task set")
    repeats = report["generation"]["repeats"]
    for task_id, attempts in tasks.items():
        if set(attempts) != set(range(repeats)):
            raise ValueError(f"Missing or invalid repeat indices for {task_id}")
    for record in report["results"]:
        if record.get("prompt_hash") != prompts[record["task_id"]]:
            raise ValueError("Result prompt hash differs from frozen protocol")
        if record.get("seed") != report["generation"]["seed"] + record["repeat_index"]:
            raise ValueError("Result seed differs from frozen protocol")


def validate_comparison(base: dict, treat: dict) -> tuple[dict, dict, list[str]]:
    base_tasks, treat_tasks = task_attempts(base), task_attempts(treat)
    if set(base_tasks) != set(treat_tasks):
        raise ValueError("Task sets differ; rerun both models on the same complete task set")
    for task_id in base_tasks:
        if set(base_tasks[task_id]) != set(treat_tasks[task_id]):
            raise ValueError(f"Repeat budgets differ for {task_id}")
    repeat_counts = {len(attempts) for attempts in base_tasks.values()}
    if len(repeat_counts) != 1:
        raise ValueError("Unequal repeat budgets between tasks")
    warnings = []
    modern = base.get("schema_version") == treat.get("schema_version") == "codeagent-eval-v2"
    if not modern:
        warnings.append("unverified：旧报告缺少冻结协议，无法核实 prompt、量化、采样与任务版本一致。")
    else:
        required = ("protocol_hash", "tasks_hash", "generation", "task_hashes")
        for key in required:
            if not base.get(key) or not treat.get(key):
                raise ValueError(f"Modern report is missing {key}")
            if base[key] != treat[key]:
                raise ValueError(f"Evaluation {key} differs; comparisons require the same protocol")
        validate_report_protocol(base, base_tasks)
        validate_report_protocol(treat, treat_tasks)
        if base.get("model") != treat.get("model"):
            raise ValueError("Base model identities differ")
        if base.get("resolved_revision") != treat.get("resolved_revision"):
            raise ValueError("Resolved base-model revisions differ")
        if not base.get("resolved_revision"):
            warnings.append("unverified：模型精确 revision 未固定，不能核实权重版本一致。")
        expected = base["generation"].get("repeats")
        if repeat_counts != {expected}:
            raise ValueError("Stored attempts do not match the declared repeat budget")
        for name, report in (("基线", base), ("训练后", treat)):
            if any(item.get("failure_type") == "infrastructure_error"
                   for item in report["results"]):
                raise ValueError(f"{name} contains infrastructure errors; repair and rerun")
            contamination = report.get("contamination", {}).get("status")
            if contamination == "contaminated":
                warnings.append(f"{name}与训练数据重合；此结果仅用于训练内诊断。")
        if base.get("split") in {"smoke", "train"}:
            warnings.append("当前是 smoke/train 诊断集，不能据此声称对未见任务泛化提升。")
        if treat.get("split") in {"dev", "test", "ood"}:
            if treat.get("contamination", {}).get("status") != "clean":
                warnings.append("unverified：训练后模型缺少训练污染审计。")
    if any(record.get("verdict") == "error" for report in (base, treat)
           for record in report.get("results", [])):
        warnings.append("报告含 error 记录，失败可能来自运行基础设施。")
    return base_tasks, treat_tasks, warnings


def wilson_interval(passed: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    z_value = 1.959963984540054
    rate = passed / total
    denominator = 1 + z_value ** 2 / total
    center = (rate + z_value ** 2 / (2 * total)) / denominator
    margin = z_value * math.sqrt(rate * (1 - rate) / total
                                + z_value ** 2 / (4 * total ** 2)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def paired_bootstrap(base_rates: list[float], treatment_rates: list[float],
                     samples: int = 10000, seed: int = 42) -> tuple[float, float]:
    if len(base_rates) != len(treatment_rates) or not base_rates:
        raise ValueError("Bootstrap requires nonempty paired task rates")
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    differences = [treatment - baseline
                   for baseline, treatment in zip(base_rates, treatment_rates)]
    rng = random.Random(seed)
    count = len(differences)
    estimates = sorted(sum(rng.choice(differences) for _ in differences) / count
                       for _ in range(samples))
    return estimates[int(0.025 * (samples - 1))], estimates[int(0.975 * (samples - 1))]


def format_interval(interval, percentage_points=False):
    if percentage_points:
        return f"[{interval[0] * 100:+.2f}, {interval[1] * 100:+.2f}] 个百分点"
    return f"[{interval[0]:.2%}, {interval[1]:.2%}]"


def make_report(base: dict, treat: dict, bootstrap_samples=10000, seed=42) -> str:
    base_tasks, treat_tasks, warnings = validate_comparison(base, treat)
    base_selected = (base.get("schema_version") == treat.get("schema_version")
                     == "codeagent-eval-v2" and "adapter" in base and "adapter" in treat
                     and base["adapter"] is None and treat["adapter"] is None)
    treatment_label = "最终选择（Base）" if base_selected else "训练后"
    task_ids = sorted(base_tasks)
    base_rates = [sum(base_tasks[task_id].values()) / len(base_tasks[task_id])
                  for task_id in task_ids]
    treat_rates = [sum(treat_tasks[task_id].values()) / len(treat_tasks[task_id])
                   for task_id in task_ids]
    base_rate, treat_rate = sum(base_rates) / len(task_ids), sum(treat_rates) / len(task_ids)
    delta = treat_rate - base_rate
    delta_ci = paired_bootstrap(base_rates, treat_rates, bootstrap_samples, seed)
    repeats = len(base_tasks[task_ids[0]])
    base_any = sum(any(attempts.values()) for attempts in base_tasks.values())
    treat_any = sum(any(attempts.values()) for attempts in treat_tasks.values())
    lines = [f"# 评估对比：基线 vs {treatment_label}", "",
             f"- 基线模型: {base.get('model', '?')}",
             f"- {treatment_label}模型: {treat.get('model', '?')}",
             f"- Adapter: {treat.get('adapter') or ('未采用，保留 Base' if base_selected else '无')}",
             f"- 任务数: {len(task_ids)}；每任务重复: {repeats}",
             f"- 协议验证: {'unverified / diagnostic' if warnings else 'verified'}", ""]
    if base_selected:
        lines.append("> 本表对比 Base 与 Base，没有比较任何训练后的 adapter。"
                     "checkpoint 选择过程请查看 selection.json 及各 epoch 的评测报告；"
                     "不能由本表推断各 epoch 的生成输出相同。")
    for warning in warnings:
        lines.append(f"> {warning}")
    lines.extend(["", f"| 任务 | 基线 pass@1 | {treatment_label} pass@1 | 变化（百分点） |",
                  "|---|---|---|---|"])
    for task_id, baseline, treatment in zip(task_ids, base_rates, treat_rates):
        lines.append(f"| {task_id} | {baseline:.2%} | {treatment:.2%} | "
                     f"{(treatment - baseline) * 100:+.2f} |")
    lines.extend(["", "## 汇总", "",
                  f"| 指标 | 基线 | {treatment_label} |", "|---|---|---|",
                  f"| 平均 pass@1 | {base_rate:.2%} | {treat_rate:.2%} |",
                  f"| 至少一次通过（any-pass） | {base_any}/{len(task_ids)} "
                  f"({base_any / len(task_ids):.2%}) | {treat_any}/{len(task_ids)} "
                  f"({treat_any / len(task_ids):.2%}) |",
                  "| any-pass Wilson 95% 区间 | "
                  f"{format_interval(wilson_interval(base_any, len(task_ids)))} | "
                  f"{format_interval(wilson_interval(treat_any, len(task_ids)))} |",
                  "", f"平均 pass@1 变化: **{delta * 100:+.2f} 个百分点**。",
                  f"按任务配对 bootstrap 95% 区间: {format_interval(delta_ci, True)} "
                  f"（{bootstrap_samples} 次重采样，seed={seed}）。", "",
                  "pass@1 是每任务各次独立生成通过率的平均值；any-pass 是在全部重复中至少通过一次的任务比例，二者不能混用。",
                  "多重复按 task_id/repeat_index 配对；bootstrap 以任务为抽样单位，保留同一任务重复间的关联。"])
    if repeats == 1:
        lines.append("本次每任务只有一次生成，因此 any-pass 与 pass@1 相同，Wilson 区间也适用于本次 pass@1。")
    if base_rates == treat_rates:
        lines.append("各任务观测通过率的配对差值均为 0，因此 bootstrap 区间退化为 [0, 0]；"
                     "这不证明总体能力等效，也不表示生成代码相同。")
    if len(task_ids) < 100:
        lines.append(f"当前只有 {len(task_ids)} 个任务，区间仅反映这组任务的重采样波动；请扩大独立留出集并检查失败案例，不能由单个任务变化断言训练有效或无效。")
    if delta_ci[0] <= 0 <= delta_ci[1]:
        lines.append("差值区间包含 0，当前结果不足以证明稳定提升。")
    else:
        lines.append("差值区间不包含 0，但泛化结论仍依赖独立留出任务、无污染审计与相同计算预算。")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--output")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        report = make_report(load_results(args.baseline), load_results(args.treatment),
                             args.bootstrap_samples, args.seed)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(report)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
