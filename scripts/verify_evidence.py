"""Recompute the conclusions of a published evidence bundle.

每个 docs/evidence/<label>/ 由 export_evidence.py 导出。这个脚本不信任写在
README 或 claims.json 里的数字，而是从逐题结果重新算一遍，任何一处对不上就
以非零状态退出。CI 会对全部证据包运行它。

逐项检查：
1. 完整性：每个文件与 index.json 记录的 sha256 一致，没有未登记的文件；
2. 每份报告：由逐题结果重算的 pass@1、任务数与文件里存的指标一致；
3. 选型：按文档写明的规则从各 epoch 的 dev 报告重新推导，与 selection.json 一致；
4. 终评对比：通过数、平均变化与按来源组聚类的 bootstrap 区间，与 comparison.md 一致；
5. 跨文件：模型、revision 与任务哈希一致；
6. claims.json（可选）：每个声明的数字与重算值一致。

usage: python scripts/verify_evidence.py [BUNDLE_DIR ...]    默认检查 docs/evidence/ 下全部
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from compare import paired_bootstrap, report_source_groups, task_attempts, validate_comparison  # noqa: E402
from select_checkpoint import choose_checkpoint  # noqa: E402

_UNLISTED_ALLOWED = {"index.json", "claims.json"}
_DELTA = re.compile(r"平均 pass@1 变化: \*\*([+-]?\d+\.\d+) 个百分点\*\*")
_INTERVAL = re.compile(r"95% 区间: \[([+-]?\d+\.\d+), ([+-]?\d+\.\d+)\] 个百分点")
_RESAMPLING = re.compile(r"（(\d+) 次重采样，seed=(\d+)")


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def is_report(payload) -> bool:
    return isinstance(payload, dict) and isinstance(payload.get("results"), list) \
        and isinstance(payload.get("task_ids"), list)


def rates(report: dict) -> tuple[list[str], list[float]]:
    tasks = task_attempts(report)
    task_ids = sorted(tasks)
    return task_ids, [sum(tasks[task_id].values()) / len(tasks[task_id]) for task_id in task_ids]


def passed_tasks(report: dict) -> int:
    """全部重复都通过的任务数；单次生成时就是“修好了几道题”。"""
    return sum(all(attempts.values()) for attempts in task_attempts(report).values())


def any_passed_tasks(report: dict) -> int:
    return sum(any(attempts.values()) for attempts in task_attempts(report).values())


def _close(expected, actual, tolerance=1e-6) -> bool:
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(expected, actual, abs_tol=tolerance)
    return expected == actual


def verify_bundle(directory: Path) -> tuple[dict, list[str]]:
    index_path = directory / "index.json"
    if not index_path.is_file():
        return {}, [f"not an evidence bundle (no index.json): {directory}"]
    problems: list[str] = []
    index = _load(index_path)

    # 1. 完整性
    for name, meta in index["files"].items():
        path = directory / name
        if not path.is_file():
            problems.append(f"missing file: {name}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
            problems.append(f"sha256 differs from index.json: {name}")
    listed = set(index["files"]) | _UNLISTED_ALLOWED
    unlisted = sorted({str(path.relative_to(directory)) for path in directory.rglob("*")
                       if path.is_file()} - listed)
    if unlisted:
        problems.append(f"files not listed in index.json: {unlisted}")

    reports = {}
    for name in index["files"]:
        path = directory / name
        if name.endswith(".json") and path.is_file():
            payload = _load(path)
            if is_report(payload):
                reports[name] = payload
    summary: dict = {"label": index["label"], "models": index["models"]}

    # 2. 每份报告的指标能由逐题结果重算
    for name, report in sorted(reports.items()):
        task_ids, task_rates = rates(report)
        stored = report.get("metrics", {})
        recomputed = {"n_tasks": len(task_ids), "pass_at_1": sum(task_rates) / len(task_rates),
                      "n_tasks_any_pass": any_passed_tasks(report)}
        for key, value in recomputed.items():
            if key in stored and not _close(stored[key], value):
                problems.append(f"{name}: metrics.{key} is {stored[key]}, recomputed {value}")
        if "n_tasks" in report and report["n_tasks"] != recomputed["n_tasks"]:
            problems.append(f"{name}: n_tasks is {report['n_tasks']}, "
                            f"recomputed {recomputed['n_tasks']}")
        if "success_rate" in report and not _close(report["success_rate"], recomputed["pass_at_1"]):
            problems.append(f"{name}: success_rate is {report['success_rate']}, "
                            f"recomputed {recomputed['pass_at_1']}")
        if report.get("infrastructure_errors") or stored.get("infrastructure_errors"):
            problems.append(f"{name}: contains infrastructure errors; not valid evidence")
        for record in report["results"]:
            if record.get("failure_type") == "infrastructure_error":
                problems.append(f"{name}: infrastructure_error on {record.get('task_id')}")

    # 5. 跨文件一致：模型、revision、同一 split 的任务哈希
    for field in ("model", "resolved_revision"):
        values = {report[field] for report in reports.values() if report.get(field)}
        if len(values) > 1:
            problems.append(f"{field} differs across reports: {sorted(values)}")
    for split in ("dev", "test"):
        hashes = {report["tasks_hash"] for name, report in reports.items()
                  if report.get("split") == split and report.get("tasks_hash")}
        if len(hashes) > 1:
            problems.append(f"tasks_hash differs across {split} reports")
        split_reports = [report for report in reports.values() if report.get("split") == split]
        if split_reports:
            summary[f"n_{split}_tasks"] = len(split_reports[0]["task_ids"])
    training = directory / "training.json"
    if training.is_file():
        manifest = _load(training)
        if reports and manifest.get("model") and manifest["model"] not in {
                report.get("model") for report in reports.values()}:
            problems.append("training.json records a different base model than the reports")
        summary["optimizer_steps"] = manifest.get("optimizer_steps")

    # 3. 选型可由 dev 报告重新推导
    selection_path = directory / "selection.json"
    dev_reports = sorted((name for name in reports if re.fullmatch(r"dev/epoch-\d+-dev\.json", name)))
    if selection_path.is_file():
        selection = _load(selection_path)
        candidates = [{"epoch": row["epoch"], "adapter": row["adapter"]}
                      for row in selection.get("candidates", [])]
        epochs = [int(re.search(r"epoch-(\d+)", name).group(1)) for name in dev_reports]
        if sorted(epochs) != sorted(row["epoch"] for row in candidates):
            problems.append("selection.json candidates do not match the exported dev reports")
        else:
            ordered = [reports[name] for name in sorted(
                dev_reports, key=lambda name: int(re.search(r"epoch-(\d+)", name).group(1)))]
            try:
                again = choose_checkpoint(candidates, ordered)
            except ValueError as error:
                problems.append(f"selection cannot be re-derived: {error}")
            else:
                for key in ("selected_epoch", "selected_adapter", "decision",
                            "base_pass_at_1", "selected_pass_at_1"):
                    if not _close(selection.get(key), again[key]):
                        problems.append(f"selection.json {key} is {selection.get(key)}, "
                                        f"re-derived {again[key]}")
                summary["selected_epoch"] = again["selected_epoch"]
                summary["selected_decision"] = again["decision"]
                summary["base_dev_passed"] = round(again["base_pass_at_1"] * len(ordered[0]["task_ids"]))
                summary["selected_dev_passed"] = round(again["selected_pass_at_1"] * len(ordered[0]["task_ids"]))

    # 4. 终评对比：通过数、平均变化、聚类 bootstrap 区间
    if all(name in reports for name in ("test/base.json", "test/selected.json")):
        base, selected = reports["test/base.json"], reports["test/selected.json"]
        try:
            base_tasks, selected_tasks, _ = validate_comparison(base, selected)
        except ValueError as error:
            problems.append(f"test reports cannot be compared: {error}")
        else:
            task_ids = sorted(base_tasks)
            base_rates = [sum(base_tasks[t].values()) / len(base_tasks[t]) for t in task_ids]
            selected_rates = [sum(selected_tasks[t].values()) / len(selected_tasks[t])
                              for t in task_ids]
            delta = (sum(selected_rates) - sum(base_rates)) / len(task_ids) * 100
            summary.update(base_test_passed=passed_tasks(base),
                           selected_test_passed=passed_tasks(selected),
                           test_delta_pp=delta)
            markdown = directory / "test/comparison.md"
            if markdown.is_file():
                text = markdown.read_text(encoding="utf-8")
                stated = _DELTA.search(text)
                if not stated or not _close(float(stated.group(1)), delta, 0.01):
                    problems.append(f"comparison.md states {stated and stated.group(1)} pp, "
                                    f"recomputed {delta:+.2f} pp")
                resampling = _RESAMPLING.search(text)
                interval = _INTERVAL.search(text)
                if resampling and interval:
                    groups = report_source_groups(base, task_ids)
                    if report_source_groups(selected, task_ids) == groups:
                        ci = paired_bootstrap(base_rates, selected_rates,
                                              int(resampling.group(1)), int(resampling.group(2)),
                                              groups)
                        if not all(_close(float(bound), value * 100, 0.01)
                                   for bound, value in zip(interval.groups(), ci)):
                            problems.append(
                                f"comparison.md states the interval {interval.groups()}, "
                                f"recomputed {[round(value * 100, 2) for value in ci]}")
                        summary["test_ci_pp"] = [round(value * 100, 2) for value in ci]
                    else:
                        problems.append("source_group differs between base and selected reports")

    # 6. claims.json：README 等处引用的数字
    claims_path = directory / "claims.json"
    if claims_path.is_file():
        for claim, expected in _load(claims_path).items():
            actual = summary.get(claim)
            if isinstance(expected, list) and isinstance(actual, list):
                agrees = len(expected) == len(actual) and all(
                    _close(float(a), float(b), 0.01) for a, b in zip(expected, actual))
            else:
                agrees = actual is not None and _close(float(expected), float(actual), 0.01)
            if not agrees:
                problems.append(f"claims.json: {claim} claims {expected!r}, evidence gives {actual!r}")
    return summary, problems


def main() -> int:
    directories = [Path(argument) for argument in sys.argv[1:]] or sorted(
        path for path in (ROOT / "docs" / "evidence").glob("*") if path.is_dir())
    if not directories:
        print("没有证据包。用 scripts/export_evidence.py 从本地产物导出。")
        return 0
    failures = 0
    for directory in directories:
        summary, problems = verify_bundle(directory)
        print(f"── {directory.name}")
        for key, value in summary.items():
            if key != "label":
                print(f"   {key}: {value}")
        if problems:
            failures += 1
            for problem in problems:
                print(f"   ✗ {problem}")
        else:
            print("   ✓ 结论可由证据包重算")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
