"""按 原有题 / 难题 / 单缺陷变体 汇总评测报告或选型目录。

用法：python scripts/tier_report.py 标签=报告.json [标签=报告.json …]
      python scripts/tier_report.py 选型目录 [选型目录 …]

报告是 evaluate.py 的输出；选型目录是 select_checkpoint.py 的输出（会展开成基座和各个
epoch，并标出被选中的那个）。第一行当作基座，后面的行给出相对它变好、变差的题数。

一次给多个选型目录时，每组的行都叫 base / epoch N，不带组名分不清是哪一组的：这时行名的后面带上
目录名（也就是组名），写成 "epoch 1（选中） · D-prefix"；行名照样以 base / epoch 开头，便于 grep。
只给一个目录时和原来一样不带。

表格和"各层变好/变差"列出哪些层由第一行（基座）决定：原有三层照旧；后来加的层（合成难题、合成变体）
只在基座这一行里有题时才列，只评了原有三层的题时输出和以前一字不差。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agentlab.benchmark.tiers import LABELS, TIERS, tier_of

# 最早的三层：表里永远列出，哪怕这一行里一层题都没有
BUILT_IN = ("default", "hard", "variant")


def display_width(text: str) -> int:
    return sum(2 if ord(char) > 0x2E7F else 1 for char in text)


def cell(text, size: int) -> str:
    text = str(text)
    return text + " " * (size - display_width(text))


def tier_counts(report: dict) -> dict:
    """每层的 (通过数, 题数)。层的清单取自 tiers.py 的 TIERS：每层都有一项，没有题的是 [0, 0]。"""
    counts = {tier: [0, 0] for tier in TIERS}
    for record in report["results"]:
        row = counts[tier_of(record["task_id"])]
        row[0] += bool(record["strict_pass"])
        row[1] += 1
    return counts


def shown_tiers(counts: dict) -> tuple:
    """表里和"各层变好/变差"里列出哪些层，由第一行（基座）的计数决定。

    原有三层照旧列出（哪怕题数是 0）；别的层要有题才列——只评了原有三层的题时，这里就只有三层。
    """
    return tuple(tier for tier in TIERS if tier in BUILT_IN or counts.get(tier, [0, 0])[1] > 0)


def transitions(base: dict, other: dict) -> dict:
    """相对基座：每层变好、变差的题数，以及总体。层同 shown_tiers：基座这一行里有哪些层。"""
    before = {record["task_id"]: bool(record["strict_pass"]) for record in base["results"]}
    after = {record["task_id"]: bool(record["strict_pass"]) for record in other["results"]}
    if set(before) != set(after):
        raise ValueError("两份报告的题目不一样，不能逐题比较")
    rows = {tier: [0, 0] for tier in shown_tiers(tier_counts(base))}
    total = [0, 0]
    for task, passed in after.items():
        row = rows[tier_of(task)]
        row[0] += passed and not before[task]
        row[1] += before[task] and not passed
        total[0] += passed and not before[task]
        total[1] += before[task] and not passed
    return {"tiers": rows, "total": total}


def table(rows: list[tuple[str, dict]]) -> str:
    """rows: (标签, 每层计数)。输出一张对齐的表；列哪些层由第一行（基座）决定。"""
    shown = shown_tiers(rows[0][1])
    width = max(display_width(label) for label, _ in rows)
    lines = [cell("模型", width + 2) + "".join(cell(title, 9) for title in [*map(LABELS.get, shown), "合计"])]
    for label, counts in rows:
        lines.append(cell(label, width + 2)
                     + "".join(cell(f"{counts[tier][0]}/{counts[tier][1]}", 9) for tier in shown)
                     + f"{sum(row[0] for row in counts.values())}/{sum(row[1] for row in counts.values())}")
    return "\n".join(lines)


def load_rows(arguments: list[str]) -> list[tuple[str, dict]]:
    """把命令行参数展开成 (标签, 报告)。多个选型目录一起给时，选型目录的行名后面带上目录名（组名）。"""
    named = sum("=" not in argument for argument in arguments) > 1
    rows = []
    for argument in arguments:
        if "=" in argument:
            label, path = argument.split("=", 1)
            rows.append((label, json.loads(Path(path).read_text(encoding="utf-8"))))
            continue
        directory = Path(argument)
        selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
        group = f" · {directory.name}" if named else ""
        for candidate in selection["candidates"]:
            epoch = candidate["epoch"]
            label = "base" if epoch == 0 else f"epoch {epoch}"
            if selection["selected_epoch"] == epoch:
                label += "（保留基座）" if epoch == 0 else "（选中）"
            rows.append((label + group, json.loads(
                (directory / f"epoch-{epoch:03d}-dev.json").read_text(encoding="utf-8"))))
    return rows


def main() -> None:
    rows = load_rows(sys.argv[1:])
    if not rows:
        raise SystemExit(__doc__)
    print(table([(label, tier_counts(report)) for label, report in rows]))
    for label, report in rows[1:]:
        change = transitions(rows[0][1], report)
        per_tier = "、".join(f"{LABELS[tier]} +{row[0]}/-{row[1]}" for tier, row in change["tiers"].items())
        print(f"{label} 相对 {rows[0][0]}：变好 {change['total'][0]} 道、变差 {change['total'][1]} 道"
              f"（各层变好/变差：{per_tier}）")


if __name__ == "__main__":
    main()
