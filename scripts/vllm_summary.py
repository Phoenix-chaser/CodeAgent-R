"""打印一轮 RFT 的 vLLM 评测（辅助指标）和族内探针的汇总表。

用法：python scripts/vllm_summary.py --run 运行目录 --arms 组名… [--probe-ids-file 题号文件]

读 <运行目录>/vllm/<标签>/ 下 scripts/eval_vllm.py 写出的 summary、报告和自检结果（标签是 base 或某个组名；
自检没过，也就是适配器的输出和基座完全一样时，eval_vllm.py 不写 summary 和报告，只写 <标签>-probe-sanity.json，
表二里这一组的原因从这里来；自检没能和基座比较的组在表二里注明没能确认适配器是否生效），
再读 <运行目录>/sft/<组名>/train.jsonl 里出现过的题号，按固定版式打印四段：
  一、vLLM 评测：每个标签在 dev 上的贪心、采样通过率和 pass@k，下一行是各层的贪心结果（只列有题的层）；
  二、族内探针：训练过的题和没训练过的同族题各通过多少，括号里是基座的通过数；
  三、每张表后面一行以"结果："开头的结论（进度面板靠它认，也可以 grep 取）；
  四、怎么读。
这些数字来自 vLLM（bf16 权重），只是辅助指标；主结果仍是 evaluate.py 在 HF 上的贪心评测，两边不能混着比。
缺的报告跳过并写明原因，不抛异常。只用标准库：服务器上训练环境装好之前也能跑。
层的清单、中文名和"题号属于哪一层"取自 src/agentlab/benchmark/tiers.py，同 eval_vllm.py 写 by_tier 用的那份：
题数为 0 的层不列（表里不会出现 0/0），只有原有三层的题时和三层的时候一模一样，加了新层不用改这里。
"""
from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

BASE = "base"
DEV_HEADING = "vLLM 评测（bf16 权重，辅助指标；主结果是上面 HF 贪心那张表）"
PROBE_HEADING = "族内探针（上一轮零通过的题，贪心）"
VOID_TEXT = "（评测作废或没做）"
HOW_TO_READ = (
    "D 组明显高于 F 组才说明难题的正确答案有用；",
    "采样通过率比贪心灵敏，只作旁证；",
    "族内探针分清是\"训练信号太弱\"还是\"跨不了族\"；",
    "test 这一轮不评。",
)
TRAINED, UNTRAINED = "训练过的题", "没训练过的同族题"
PARTS = ((TRAINED, "trained"), (UNTRAINED, "untrained"))     # 表二里一个组的两项
TIERS_FILE = Path(__file__).resolve().parent.parent / "src" / "agentlab" / "benchmark" / "tiers.py"
BUILT_IN = {"default": "原有题", "hard": "难题", "variant": "变体"}     # tiers.py 读不到时用的三层


def builtin_tier_of(task_id: str) -> str:
    """tiers.py 读不到的时候按题号前缀判断：h3v_ 开头是单缺陷变体，h3_ 开头是难题，其余是原有题。"""
    if task_id.startswith("h3v_"):
        return "variant"
    if task_id.startswith("h3_"):
        return "hard"
    return "default"


def load_tiers(path=TIERS_FILE) -> tuple:
    """层的定义：(各层中文名的先后, {层名或中文名: 中文名}, 按题号判断层别的函数, 说明)。

    前三项取自 tiers.py 的 TIERS、LABELS、tier_of。按文件路径执行 tiers.py，不 import agentlab.benchmark：包的
    __init__ 会把构建任务、判分那些模块一起拉进来，这个脚本就不再只靠标准库了。读不到、执行不了、缺了哪一项时
    退回内置的三层，说明里写原因（否则是 None）：这时只认得出原有题、难题、变体，别的层照 summary 里的键名原样列出。
    """
    try:
        namespace = runpy.run_path(str(path))
        order, labels, tell = tuple(namespace["TIERS"]), dict(namespace["LABELS"]), namespace["tier_of"]
        names = tuple(labels[tier] for tier in order)
    except Exception as error:          # 文件不在、语法错、import 不了、缺名字：都是"读不到层的定义"，一并退回
        # 系统调用的报错（文件不在、没有权限）只取原因：它自带的文件名和前面重复
        reason = error.strerror or str(error) if isinstance(error, OSError) else f"{type(error).__name__}: {error}"
        note = (f"层的定义读不到：{path}（{reason}）"
                "：层名按内置的三层（原有题、难题、变体）来写，别的层照 summary 里的键名原样列出。把脚本放在仓库里运行")
        return tuple(BUILT_IN.values()), BUILT_IN, builtin_tier_of, note
    return names, labels, tell, None


TIER_ORDER, TIER_KEYS, tier_of, FALLBACK_NOTE = load_tiers()


# ---- 读文件 ----


def read_json(path):
    """读一份 JSON：返回 (内容, None)。文件不存在是 (None, None)，读不出来是 (None, 原因)。"""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, None
    except (OSError, ValueError) as error:
        return None, f"{Path(path).name} 读不出来（{error}）"


def load_part(run, label: str, stem: str) -> dict:
    """某个标签在 dev 或 probe 上的 summary、贪心报告和自检结果。没有的项是 None；读不出来的原因放进 problems。

    ``sanity`` 是自检结果（pass、total，以及和基座比较的 changed、compared 等）：先取 summary 的 sanity 字段；
    没有 summary（自检没过时 eval_vllm.py 以 4 退出、不往后评）或者 summary 里没有，就读 <label>-<stem>-sanity.json
    ——eval_vllm.py 自检没过时只留下这一个文件，里面同样有这些项。
    """
    directory = Path(run) / "vllm" / label
    summary_path = directory / f"{label}-{stem}-summary.json"
    report_path = directory / f"{label}-{stem}-greedy.json"
    sanity_path = directory / f"{label}-{stem}-sanity.json"
    summary, problem = read_json(summary_path)
    report, report_problem = read_json(report_path)
    sanity, sanity_problem = read_json(sanity_path)
    problems = [item for item in (problem, report_problem) if item]
    if summary is not None and not isinstance(summary, dict):
        summary, problems = None, [f"{summary_path.name} 不是一份汇总"]
    if report is not None and not (isinstance(report, dict) and isinstance(report.get("results"), list)):
        report = None
    if not sanity_numbers(sanity):
        sanity = None
    if not sanity_numbers(summary.get("sanity") if isinstance(summary, dict) else None):
        # 这一组靠 -sanity.json：它读不出来、或者里面没有 pass / total，才值得说一句
        if sanity is None and sanity_path.exists():
            problems.append(sanity_problem or f"{sanity_path.name} 里没有自检的 pass / total")
    return {"summary": summary, "report": report, "sanity": sanity, "problems": problems,
            "attempted": summary_path.exists() or report_path.exists()}


def trained_task_ids(path):
    """某一组训练数据（train.jsonl）里出现过的题号。文件不存在或读不出来时返回 None：
    这时分不出哪些探针题训练过，不能当成"一道都没训练过"。"""
    try:
        lines = Path(path).read_text(encoding="utf-8").split("\n")     # 训练数据一条记录一行，行尾只认 "\n"
    except (OSError, ValueError):
        return None
    task_ids = set()
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue                                  # 写到一半的坏行跳过
        if isinstance(record, dict) and record.get("task_id"):
            task_ids.add(str(record["task_id"]))
    return task_ids


def read_probe_ids(path):
    """探针题号文件：一行一个 task_id，或者一行一道题的 JSONL。按出现的顺序去重。"""
    task_ids = []
    for line in Path(path).read_text(encoding="utf-8").split("\n"):
        text = line.strip()
        if not text:
            continue
        try:
            record = json.loads(text)
        except ValueError:
            record = None
        task = record.get("task_id") if isinstance(record, dict) else None
        task = str(task) if task else text
        if task not in task_ids:
            task_ids.append(task)
    return task_ids


# ---- 纯函数：把读进来的报告算成表里的数 ----


def split_by_trained(probe_ids, trained_ids):
    """探针题拆成（训练过的题, 没训练过的题），保持原来的顺序。

    trained_ids 是 None 表示找不到训练数据，这时第一项也是 None：分不出，不能当成都没训练过。
    """
    if trained_ids is None:
        return None, list(probe_ids)
    return ([task for task in probe_ids if task in trained_ids],
            [task for task in probe_ids if task not in trained_ids])


def count_passed(report, task_ids):
    """这些题里通过的道数和题数；报告里没有的题不计入。report 是 None 时返回 None。"""
    if report is None:
        return None
    passed = {str(item.get("task_id")): bool(item.get("strict_pass")) for item in report["results"]}
    present = [task for task in task_ids if task in passed]
    return sum(passed[task] for task in present), len(present)


def overall(report, probe_ids):
    """一份探针报告在全部探针题上的 (通过, 题数)；报告是 None 时返回 None。
    没有题号文件时，报告里出现的题就当作全部探针题。"""
    if report is None:
        return None
    if probe_ids is None:
        results = report["results"]
        return sum(bool(item.get("strict_pass")) for item in results), len(results)
    return count_passed(report, probe_ids)


def tier_rows(by_tier, report):
    """各层的 (层, 通过, 题数)，只留有题的层，按 tiers.py 里 TIERS 的先后排（原有题 / 难题 / 变体，以后加的层在后面）。

    summary 的 by_tier 每一项是 [通过, 题数]，键可以是层名（default、hard、variant），也可以是层的中文名；
    summary 里没有 by_tier 时从贪心报告里按题号前缀数（tiers.py 的 tier_of）。题数为 0 的层不列，免得表里出现 0/0。
    认不出的层名排在后面、照原样列出。
    """
    counts = {}
    if isinstance(by_tier, dict):
        for key, value in by_tier.items():
            name = TIER_KEYS.get(key, key)
            if isinstance(value, dict):
                value = [value.get("pass"), value.get("total")]
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                continue
            try:
                counts[name] = (int(value[0]), int(value[1]))
            except (TypeError, ValueError):
                continue
    elif report is not None:
        for item in report["results"]:
            key = tier_of(str(item.get("task_id")))
            name = TIER_KEYS.get(key, key)
            passed, total = counts.get(name, (0, 0))
            counts[name] = (passed + bool(item.get("strict_pass")), total + 1)
    known = [(name, *counts[name]) for name in TIER_ORDER if counts.get(name, (0, 0))[1]]
    extra = [(name, *value) for name, value in counts.items() if name not in TIER_ORDER and value[1]]
    return known + extra


def sampled_rate(sampled):
    """采样通过率（通过的尝试数 / 尝试数）；没有采样或数不全时返回 None。"""
    if not isinstance(sampled, dict):
        return None
    attempts, passed = sampled.get("attempts"), sampled.get("passed_attempts")
    if isinstance(attempts, int) and isinstance(passed, int) and attempts > 0:
        return passed / attempts
    rate = sampled.get("pass_at_1")
    return rate if isinstance(rate, (int, float)) else None


def sanity_numbers(sanity):
    """自检结果里的 (严格通过的道数, 自检的题数)；没有自检结果（或者里面没有这两项）时返回 None。"""
    if isinstance(sanity, dict) and isinstance(sanity.get("pass"), int) and isinstance(sanity.get("total"), int):
        return sanity["pass"], sanity["total"]
    return None


def dev_row(label: str, part: dict) -> dict:
    """表一的一行：greedy 是 (通过, 题数)，tiers 是 [(层, 通过, 题数)]，sampled 是 summary 里的采样统计。
    缺的项是 None 或空表，原因放进 notes。"""
    row = {"label": label, "greedy": None, "tiers": [], "sampled": None, "notes": list(part["problems"])}
    summary, report = part["summary"], part["report"]
    if summary is None:
        row["notes"].append("没做完" if part["attempted"] else "没做")
        return row
    greedy = summary.get("greedy")
    greedy = greedy if isinstance(greedy, dict) else {}
    passed, total = greedy.get("pass"), greedy.get("total")
    if isinstance(passed, int) and isinstance(total, int):
        row["greedy"] = (passed, total)
    elif report is not None:
        results = report["results"]
        row["greedy"] = (sum(bool(item.get("strict_pass")) for item in results), len(results))
    else:
        row["notes"].append("summary 里没有贪心的通过数")
    row["tiers"] = tier_rows(greedy.get("by_tier"), report)
    if isinstance(summary.get("sampled"), dict):
        row["sampled"] = summary["sampled"]
    return row


def probe_group(label: str, part: dict, probe_ids, trained_ids, base_part) -> dict:
    """表二里某一组的数：训练过的题和没训练过的同族题各自的通过数，以及基座在同一批题上的通过数。

    probe_ids 是 None 时用这份报告里出现的题。没有贪心报告时 state 是 "void"：自检没过（适配器的输出和
    基座完全一样）时 eval_vllm.py 不往后评，或者这一组根本没评——版式上统一写"评测作废或没做"；前一种
    情况带着自检结果（summary 里的，或者自检没过时留下的 -sanity.json 里的），写在后面。
    """
    row = {"label": label, "state": "void", "trained": None, "untrained": None, "all": None,
           "sanity": None, "problems": list(part["problems"]), "note": ""}
    summary, report = part["summary"], part["report"]
    sanity = summary.get("sanity") if isinstance(summary, dict) else None
    if not isinstance(sanity, dict):
        sanity = part["sanity"]
    if isinstance(sanity, dict):
        row["sanity"] = sanity
    if report is None:
        return row
    row["state"] = "ok"
    tasks = probe_ids if probe_ids is not None else [str(item.get("task_id")) for item in report["results"]]
    base_report = base_part["report"] if base_part else None

    def block(task_ids):
        return {"own": count_passed(report, task_ids), "base": count_passed(base_report, task_ids)}

    row["all"] = block(tasks)
    trained, untrained = split_by_trained(tasks, trained_ids)
    if trained is None:
        row["note"] = "分不出哪些题训练过（找不到或读不出这一组的 train.jsonl），只能看全部"
        return row
    row["trained"], row["untrained"] = block(trained), block(untrained)
    return row


# ---- 打印 ----


def display_width(text: str) -> int:
    return sum(2 if ord(char) > 0x2E7F else 1 for char in text)


def cell(text, size: int) -> str:
    text = str(text)
    return text + " " * max(0, size - display_width(text))


def ratio(counts) -> str:
    return "-" if counts is None else f"{counts[0]}/{counts[1]}"


def sanity_text(sanity, prefix: str) -> str:
    """表二里跟在每组后面的"自检：…"；没有自检结果时返回空串。

    作数的是和基座的比较："适配器的输出在 N 道题里有 C 道和基座不同（门槛 M）"，不到门槛时写明适配器在 vLLM
    里没有生效（eval_vllm.py 这时以 4 退出，这一组作废）；训练过的题严格通过几道只作参考。没能比较时写明原因，
    并注明没能确认适配器是否生效——那不是适配器没生效，这一组也没有作废。没能比较的原因取自 sanity 的
    not_compared_reason（eval_vllm.py 写的）；没有这一项的是旧版脚本写的自检结果：题数是 0 就是要评的题里没有
    这一组训练过的题，否则那时还没有和基座比较这一项。
    """
    numbers = sanity_numbers(sanity)
    if numbers is None:
        return ""
    passed, total = numbers
    changed, compared = sanity.get("changed"), sanity.get("compared")
    if compared is not True or not isinstance(changed, int):
        why = sanity.get("not_compared_reason") or (
            "要评的题里没有这一组训练过的题" if not total
            else "这份自检结果是旧版脚本写的，那时还没有和基座比较这一项")
        return f"{prefix}没能和基座比较（{why}），没能确认适配器是否生效"
    count = sanity.get("compared_tasks")
    text = f"{prefix}适配器的输出在 {count if isinstance(count, int) else total} 道题里有 {changed} 道和基座不同"
    minimum = sanity.get("min")
    if isinstance(minimum, int):
        text += f"（门槛 {minimum}）"
        if changed < minimum:
            text += "，适配器在 vLLM 里没有生效" if not changed else f"，不到门槛 {minimum}，按设置算作没有生效"
    return text + f"；训练过的题严格通过 {passed}/{total}（仅供参考）"


def render_dev_table(rows) -> list[str]:
    width = max([display_width(row["label"]) for row in rows] + [display_width("标签")]) + 2
    lines = [cell("标签", width) + cell("dev 贪心", 12) + cell("采样通过率", 28) + "pass@k"]
    for row in rows:
        label = cell(row["label"], width)
        if row["greedy"] is None:
            lines.append(label + "没有结果（" + "、".join(dict.fromkeys(row["notes"])) + "）")
            continue
        sampled = row["sampled"]
        rate = sampled_rate(sampled)
        if rate is None:
            rate_text, pass_k = "没采样" if sampled is None else "采样结果里没有通过率", "-"
        else:
            attempts, hits = sampled.get("attempts"), sampled.get("passed_attempts")
            counts = f"（{hits}/{attempts} 次尝试）" if isinstance(attempts, int) and isinstance(hits, int) else ""
            rate_text = f"{rate:.1%}{counts}"
            any_pass, k = sampled.get("any_pass_rate"), sampled.get("samples_per_task")
            pass_k = f"pass@{k} {any_pass:.1%}" if isinstance(any_pass, (int, float)) and k else "-"
        lines.append(label + cell(ratio(row["greedy"]), 12) + cell(rate_text, 28) + pass_k)
        if row["tiers"]:
            lines.append(" " * width + "、".join(f"{name} {hit}/{total}" for name, hit, total in row["tiers"]))
    return lines


def block_text(name: str, block: dict) -> str:
    """表二里一项：题数 a/n（基座 b/n）。这一项没有题时写明；没有基座报告时省略括号。"""
    own = block["own"]
    if own[1] == 0:
        return f"{name}：没有这样的题"
    text = f"{name} {ratio(own)}"
    if block["base"] is not None:
        text += f"（基座 {ratio(block['base'])}）"
    return text


def compact_piece(block: dict) -> str:
    """结论行里同一项的紧凑写法：a/n（基座 b/n）；没有这样的题写"无"。"""
    own = block["own"]
    if own[1] == 0:
        return "无"
    text = ratio(own)
    if block["base"] is not None:
        text += f"（{ratio(block['base'])}）"
    return text


def render_probe_table(base_row: dict, rows) -> list[str]:
    width = max([display_width(row["label"]) for row in [base_row, *rows]] + [display_width("标签")]) + 2
    lines = [cell("标签", width) + "族内探针（贪心）"]
    if base_row["all"] is None:
        lines.append(cell(BASE, width) + "没有探针报告（没评基座或没评完），下面各行的括号里不写基座的通过数")
    else:
        lines.append(cell(BASE, width) + f"全部 {ratio(base_row['all'])}")
    for row in rows:
        label = cell(row["label"], width)
        if row["state"] != "ok":
            text = VOID_TEXT + sanity_text(row["sanity"], "；自检：")
        else:
            if row["trained"] is None:
                text = f"{row['note']}；全部 {ratio(row['all']['own'])}"
                if row["all"]["base"] is not None:
                    text += f"（基座 {ratio(row['all']['base'])}）"
            else:
                text = "、".join(block_text(name, row[key]) for name, key in PARTS)
            text += sanity_text(row["sanity"], "；自检：")
        lines.append(label + "；".join([text, *row["problems"]]))
    return lines


def dev_verdict(rows) -> str:
    """表一后面的结论行：各标签的贪心、采样通过率，以及相对基座多了几道。"""
    have = [row for row in rows if row["greedy"] is not None]
    if not have:
        return "结果：vLLM 评测还没有可用的结果（summary 和报告没做完）"
    text = "结果：dev 贪心 " + "、".join(f"{row['label']} {ratio(row['greedy'])}" for row in have)
    sampled = [row for row in have if sampled_rate(row["sampled"]) is not None]
    if sampled:
        text += "；采样通过率 " + "、".join(f"{row['label']} {sampled_rate(row['sampled']):.1%}" for row in sampled)
    base = next((row for row in have if row["label"] == BASE), None)
    changes = []
    if base is not None:
        for row in have:
            delta = row["greedy"][0] - base["greedy"][0]
            if row["label"] != BASE:
                changes.append(f"{row['label']} 持平" if delta == 0 else f"{row['label']} {delta:+d} 道")
    if changes:
        text += "；相对基座 " + "、".join(changes)
    return text


def probe_verdict(base_row: dict, rows) -> str:
    """表二后面的结论行。每个组依次写训练过的题、没训练过的同族题，括号里是基座。"""
    if base_row["all"] is None and not any(row["state"] == "ok" for row in rows):
        return "结果：族内探针没有可用的结果（没做或没评完）"
    text = "结果：族内探针（贪心；每个组依次是训练过的题、没训练过的同族题，括号里是基座）："
    text += f"基座 {ratio(base_row['all'])}" if base_row["all"] is not None else "没有基座的探针报告"
    for row in rows:
        if row["state"] != "ok":
            text += f"；{row['label']} 评测作废或没做"
        elif row["trained"] is None:
            text += f"；{row['label']} 全部 {ratio(row['all']['own'])}（分不出训练过的题）"
        else:
            pieces = [compact_piece(row[key]) for _, key in PARTS]
            text += f"；{row['label']} " + "、".join(pieces)
    return text


def report(run, arms, probe_ids_file=None) -> str:
    """整个汇总的文本。arms 是各个组的名字，顺序照给；标签 base 的行只在评过基座时出现在表一。"""
    base_dev = load_part(run, BASE, "dev")
    labels = list(dict.fromkeys(([BASE] if base_dev["attempted"] else []) + [arm for arm in arms if arm != BASE]))
    dev_rows = [dev_row(label, base_dev if label == BASE else load_part(run, label, "dev")) for label in labels]

    probe_ids, probe_note = None, ""
    if probe_ids_file:
        try:
            probe_ids = read_probe_ids(probe_ids_file)
        except (OSError, ValueError) as error:
            probe_note = f"探针题号文件 {probe_ids_file} 读不出来（{error}），改用探针报告里出现的题"
    base_probe = load_part(run, BASE, "probe")
    base_row = {"label": BASE, "all": overall(base_probe["report"], probe_ids)}
    group_rows = [probe_group(arm, load_part(run, arm, "probe"), probe_ids,
                              trained_task_ids(Path(run) / "sft" / arm / "train.jsonl"), base_probe)
                  for arm in dict.fromkeys(arms) if arm != BASE]

    lines = [f"── {DEV_HEADING} ──", *render_dev_table(dev_rows), dev_verdict(dev_rows), "",
             f"── {PROBE_HEADING} ──"]
    if probe_note:
        lines.append(probe_note)
    lines += [*render_probe_table(base_row, group_rows), probe_verdict(base_row, group_rows), "", "── 怎么读 ──"]
    lines += [f"  - {item}" for item in HOW_TO_READ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="运行目录；vllm/<标签>/ 里是 eval_vllm.py 的输出")
    parser.add_argument("--arms", required=True, nargs="+", help="各个组的名字，照这个顺序打印")
    parser.add_argument("--probe-ids-file", default=None,
                        help="探针题号文件（run_rft_guided.sh 写的 probe-ids.txt）；不给就用探针报告里出现的题")
    args = parser.parse_args(argv)
    if FALLBACK_NOTE:
        print(FALLBACK_NOTE, file=sys.stderr)
    print(report(args.run, args.arms, args.probe_ids_file))
    return 0


if __name__ == "__main__":
    sys.exit(main())
