"""Aggregate final-eval Base/SFT pairs across training seeds.

Each --pair is `LABEL BASE_JSON SFT_JSON` from one seed's frozen-test
evaluation.  Reports per-seed deltas, mean ± std across seeds, and a
bootstrap interval over the mean-across-seeds delta.  Tasks are resampled
by source group (cluster bootstrap), matching the grouped split and the
experiment contract; --cluster-by task reproduces the historical interval.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from compare import (cluster_bootstrap_mean, report_source_groups,  # noqa: E402
                    validate_comparison)


def load_pair(label: str, base_path: str, sft_path: str) -> dict:
    base = json.loads(Path(base_path).read_text(encoding="utf-8"))
    sft = json.loads(Path(sft_path).read_text(encoding="utf-8"))
    base_tasks, sft_tasks, warnings = validate_comparison(base, sft)
    task_ids = sorted(base_tasks)
    base_rates = [sum(base_tasks[t].values()) / len(base_tasks[t]) for t in task_ids]
    sft_rates = [sum(sft_tasks[t].values()) / len(sft_tasks[t]) for t in task_ids]
    improved = sum(1 for b, s in zip(base_rates, sft_rates) if s > b)
    regressed = sum(1 for b, s in zip(base_rates, sft_rates) if s < b)
    return {
        "label": label,
        "n_tasks": len(task_ids),
        "base_pass_at_1": sum(base_rates) / len(task_ids),
        "sft_pass_at_1": sum(sft_rates) / len(task_ids),
        "delta_pp": (sum(sft_rates) - sum(base_rates)) / len(task_ids) * 100,
        "improved": improved,
        "regressed": regressed,
        "task_deltas": [s - b for b, s in zip(base_rates, sft_rates)],
        "source_groups": report_source_groups(base, task_ids),
        "warnings": warnings,
    }


def aggregate(pairs: list[dict], samples: int, seed: int,
              cluster_by="source_group") -> dict:
    task_sets = {p["n_tasks"] for p in pairs}
    deltas = [p["delta_pp"] for p in pairs]
    mean_delta = sum(deltas) / len(deltas)
    variance = sum((d - mean_delta) ** 2 for d in deltas) / len(deltas)
    per_task_mean = [
        sum(p["task_deltas"][i] for p in pairs) / len(pairs)
        for i in range(pairs[0]["n_tasks"])
    ]
    groups = None
    if cluster_by == "source_group":
        first = pairs[0]["source_groups"]
        if first is None or any(p["source_groups"] != first for p in pairs):
            raise ValueError("cluster bootstrap needs the same source_group on every task in every pair")
        groups = first
    elif cluster_by != "task":
        raise ValueError("cluster_by must be 'source_group' or 'task'")
    # 对「各 seed 平均后的逐任务 delta」重采样，得到平均提升的中心区间。
    # 给出来源组时整簇抽取，保留同组两题的相关性。
    lower, upper = cluster_bootstrap_mean(per_task_mean, groups, samples, seed)
    return {
        "schema_version": 1,
        "seeds": len(pairs),
        "cluster_by": cluster_by,
        "source_groups": len(set(groups)) if groups else None,
        "per_seed": [{k: v for k, v in p.items() if k not in ("task_deltas", "source_groups")}
                     for p in pairs],
        "mean_delta_pp": mean_delta,
        "std_delta_pp": math.sqrt(variance),
        "min_delta_pp": min(deltas),
        "max_delta_pp": max(deltas),
        "mean_delta_ci_pp": [lower * 100, upper * 100],
        "consistent_task_counts": len(task_sets) == 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=3, action="append", required=True,
                        metavar=("LABEL", "BASE_JSON", "SFT_JSON"),
                        help="One seed's final-eval result pair; repeatable")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cluster-by", choices=["source_group", "task"], default="source_group",
                        help="bootstrap 抽样单位；task 仅用于复现按题重采样的历史区间")
    parser.add_argument("--output")
    args = parser.parse_args()
    pairs = []
    problems = []
    for label, base_path, sft_path in args.pair:
        try:
            pair = load_pair(label, base_path, sft_path)
        except (ValueError, OSError, json.JSONDecodeError) as error:
            problems.append(f"{label}: {error}")
            continue
        pairs.append(pair)
        print(f"seed {label}: base {pair['base_pass_at_1']:.2%} → "
              f"sft {pair['sft_pass_at_1']:.2%} "
              f"({pair['delta_pp']:+.2f}pp, +{pair['improved']}/-{pair['regressed']})")
        for warning in pair["warnings"]:
            print(f"  ! {warning}")
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
    if len(pairs) < 1:
        parser.error("no valid pairs")
    try:
        report = aggregate(pairs, args.bootstrap_samples, args.seed, args.cluster_by)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
