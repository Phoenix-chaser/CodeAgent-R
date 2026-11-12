"""Merge teacher SFT data with RFT self-generated data for hybrid training.

References:
- SWE-Gym (https://arxiv.org/abs/2412.21139, ICML 2025) — hybrid data paradigm
- GiFT (https://arxiv.org/abs/2502.11466) — self-generated data improves fine-tuning

Combines the distribution alignment of self-generated correct answers
(RFT, same output style as the model) with the broader knowledge coverage
of teacher answers (from API, potentially higher quality edge cases).

Deduplicates by AST fingerprint within each task; teacher samples take
priority (higher confidence), RFT samples fill remaining slots.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _ast_fp(code: str) -> str:
    try:
        return hashlib.md5(
            ast.dump(ast.parse(code), include_attributes=False).encode()).hexdigest()
    except (SyntaxError, ValueError):
        return f"syn:{hashlib.md5(code.encode()).hexdigest()}"


def load_sft(path: str) -> list[dict]:
    """Load SFT-format JSONL (has messages + target fields)."""
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-data", required=True,
                        help="Teacher SFT JSONL (from prepare_data.py)")
    parser.add_argument("--rft-data", required=True,
                        help="RFT self-generated JSONL (from generate_rft_data.py)")
    parser.add_argument("--max-per-task", type=int, default=4)
    parser.add_argument("--teacher-ratio", type=float, default=0.5,
                        help="Target fraction of teacher samples per task")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0 < args.teacher_ratio < 1:
        parser.error("--teacher-ratio must be in (0, 1)")

    teacher = load_sft(args.teacher_data)
    rft = load_sft(args.rft_data)
    output = Path(args.output)
    if output.exists():
        parser.error("output exists; use a new path")

    by_task_teacher: dict[str, list[dict]] = {}
    by_task_rft: dict[str, list[dict]] = {}
    for rec in teacher:
        by_task_teacher.setdefault(rec.get("task_id", ""), []).append(rec)
    for rec in rft:
        by_task_rft.setdefault(rec.get("task_id", ""), []).append(rec)

    merged = []
    stats = Counter()
    all_tasks = sorted(set(by_task_teacher) | set(by_task_rft))
    for task_id in all_tasks:
        seen = set()
        t_list = by_task_teacher.get(task_id, [])
        r_list = by_task_rft.get(task_id, [])
        # 教师优先（高置信度），RFT 填充剩余
        teacher_quota = max(1, int(args.max_per_task * args.teacher_ratio))
        count = 0
        for rec in t_list:
            if count >= teacher_quota:
                break
            fp = _ast_fp(rec["target"]["content"])
            if fp in seen:
                stats["teacher_dup"] += 1
                continue
            seen.add(fp)
            merged.append(rec)
            count += 1
            stats["teacher_kept"] += 1
        for rec in r_list:
            if count >= args.max_per_task:
                break
            fp = _ast_fp(rec["target"]["content"])
            if fp in seen:
                stats["rft_dup"] += 1
                continue
            seen.add(fp)
            merged.append({**rec, "provenance": {**rec.get("provenance", {}),
                                                 "hybrid_source": "rft"}})
            count += 1
            stats["rft_kept"] += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        for rec in merged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    manifest = {
        "schema_version": 1,
        "teacher_file": str(Path(args.teacher_data).resolve()),
        "rft_file": str(Path(args.rft_data).resolve()),
        "max_per_task": args.max_per_task,
        "teacher_ratio": args.teacher_ratio,
        "total_samples": len(merged),
        "tasks_covered": len({r["task_id"] for r in merged}),
        "stats": dict(stats),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"合并：{len(merged)} 条 / {manifest['tasks_covered']} 题")
    print(f"统计：{dict(stats)}")


if __name__ == "__main__":
    main()
