"""Aggregate final-eval Base/SFT pairs across training seeds.

Each --pair is `LABEL BASE_JSON SFT_JSON` from one seed's frozen-test
evaluation.  Reports per-seed deltas, mean ± std across seeds, and a
task-level paired bootstrap over the mean-across-seeds delta.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from compare import paired_bootstrap, task_attempts, validate_comparison  # noqa: E402


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
        "warnings": warnings,
    }


def aggregate(pairs: list[dict], samples: int, seed: int) -> dict:
    task_sets = {p["n_tasks"] for p in pairs}
    deltas = [p["delta_pp"] for p in pairs]
    mean_delta = sum(deltas) / len(deltas)
    variance = sum((d - mean_delta) ** 2 for d in deltas) / len(deltas)
    per_task_mean = [
        sum(p["task_deltas"][i] for p in pairs) / len(pairs)
        for i in range(pairs[0]["n_tasks"])
    ]
    # 单样本 bootstrap：对「各 seed 平均后的逐任务 delta」重采样任务，
    # 得到平均提升的中心区间（不是配对差分——那会把同一列表相减得零）。
    rng = random.Random(seed)
    count = len(per_task_mean)
    estimates = sorted(
        sum(rng.choice(per_task_mean) for _ in range(count)) / count
        for _ in range(samples)
    )
    ci = [estimates[int(0.025 * (samples - 1))] * 100,
          estimates[int(0.975 * (samples - 1))] * 100]
    return {
        "schema_version": 1,
        "seeds": len(pairs),
        "per_seed": [{k: v for k, v in p.items() if k != "task_deltas"}
                     for p in pairs],
        "mean_delta_pp": mean_delta,
        "std_delta_pp": math.sqrt(variance),
        "min_delta_pp": min(deltas),
        "max_delta_pp": max(deltas),
        "mean_delta_ci_pp": ci,
        "consistent_task_counts": len(task_sets) == 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=3, action="append", required=True,
                        metavar=("LABEL", "BASE_JSON", "SFT_JSON"),
                        help="One seed's final-eval result pair; repeatable")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
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
    report = aggregate(pairs, args.bootstrap_samples, args.seed)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
