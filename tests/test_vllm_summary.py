# -*- coding: utf-8 -*-
"""vllm_summary.py：用合成的 eval_vllm.py 产物核对两张表、两行结论和"怎么读"，以及缺文件时怎么交代。"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import vllm_summary  # noqa: E402

PROBE = [f"h3_t{index}" for index in range(8)]
DEV_HEADING = "── vLLM 评测（bf16 权重，辅助指标；主结果是上面 HF 贪心那张表） ──"
PROBE_HEADING = "── 族内探针（上一轮零通过的题，贪心） ──"
HOW_HEADING = "── 怎么读 ──"
HOW_TO_READ = [
    "  - D 组明显高于 F 组才说明难题的正确答案有用；",
    "  - 采样通过率比贪心灵敏，只作旁证；",
    '  - 族内探针分清是"训练信号太弱"还是"跨不了族"；',
    "  - test 这一轮不评。",
]
# 自检作数的是"输出和基座不同的题够不够门槛"；训练过的题严格通过几道只作参考
CHECKED = r"自检：适配器的输出在 6 道题里有 4 道和基座不同（门槛 1）；训练过的题严格通过 4/6（仅供参考）"
SPLIT = r"训练过的题 2/4（基座 1/4）、没训练过的同族题 0/4（基座 0/4）；" + CHECKED
SPLIT_NO_BASE = r"训练过的题 2/4、没训练过的同族题 0/4；" + CHECKED
LEGACY = "这份自检结果是旧版脚本写的，那时还没有和基座比较这一项"      # 没有 compared / changed 的自检结果


def sanity_result(passed=4, total=6, changed=4, minimum=1, compared=True, reason=None):
    """eval_vllm.py 写进 summary 的自检结果（新判据）：compared 为假时没能比较，reason 是没能比较的原因。"""
    return {"pass": passed, "total": total, "min": minimum, "changed": changed if compared else 0,
            "compared": compared, "compared_tasks": total if compared else 0, "not_compared_reason": reason}
NO_BASE = r"没有探针报告（没评基座或没评完），下面各行的括号里不写基座的通过数"
VOID = r"（评测作废或没做）"
CANNOT_SPLIT = "分不出哪些题训练过（找不到或读不出这一组的 train.jsonl），只能看全部；"


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def evaluation(run, label, stem, rows, sampled=None, sanity=None, report=True, tiers=None):
    """照 eval_vllm.py 的写法造一份评测产物：贪心报告加 summary。rows 是 [(题号, 通过没有)]。tiers 是
    (各层的键, {键: 中文名}, tier_of)，不给就是 tiers.py 现在的定义。"""
    folder = run / "vllm" / label
    results = [dict(task_id=task, strict_pass=passed, verdict="strict_pass" if passed else "partial")
               for task, passed in rows]
    if report:
        write(folder / f"{label}-{stem}-greedy.json", {"schema_version": "codeagent-eval-v2", "results": results})
    keys, _, tell = tiers or (tuple(vllm_summary.TIER_KEYS), vllm_summary.TIER_KEYS, vllm_summary.tier_of)
    by_tier = {tier: [0, 0] for tier in keys}      # 每一层都写，没有题的层是 [0, 0]（eval_vllm.py 的 by_tier 就是这样）
    for task, passed in rows:
        by_tier[tell(task)][0] += bool(passed)
        by_tier[tell(task)][1] += 1
    write(folder / f"{label}-{stem}-summary.json", {
        "label": label, "model": label, "split": "train" if stem == "probe" else "dev", "tag": stem,
        "greedy": {"pass": sum(bool(passed) for _, passed in rows), "total": len(rows), "by_tier": by_tier},
        "sampled": sampled, "sanity": sanity})


def sampled(samples, passed, solved, tasks=16):
    return {"samples_per_task": samples, "attempts": samples * tasks, "passed_attempts": passed,
            "pass_at_1": passed / (samples * tasks), "any_pass_rate": solved / tasks, "by_tier": {}}


def dev_rows(fixed=(), broke=()):
    """16 道 dev 题：原有题 4、难题 6、变体 6，base 通过 8 道；fixed 里的题改成通过，broke 里的改成不通过。"""
    rows = [(f"v2_d{index}", index % 4 != 0) for index in range(4)]
    rows += [(f"h3_d{index}", index % 3 == 0) for index in range(6)]
    rows += [(f"h3v_d{index}", index % 2 == 0) for index in range(6)]
    return [(task, (passed or task in fixed) and task not in broke) for task, passed in rows]


def stage(tmp_path, base=True, ids_file=True):
    """一次完整的演练数据：base 评了 dev（贪心加采样）和探针；D-prefix 评了探针（带自检）和 dev（只贪心）；
    F-control 只有训练数据，探针没做。返回 (运行目录, 探针题号文件或 None)。"""
    run = tmp_path / "run"
    if base:
        evaluation(run, "base", "dev", dev_rows(), sampled=sampled(4, 11, 6))
        evaluation(run, "base", "probe", [(task, index == 0) for index, task in enumerate(PROBE)])
    evaluation(run, "D-prefix", "dev", dev_rows(fixed={"h3_d1"}))
    evaluation(run, "D-prefix", "probe", [(task, index in (1, 2)) for index, task in enumerate(PROBE)],
               sanity=sanity_result())
    write(run / "sft/D-prefix/train.jsonl", "".join(
        json.dumps({"task_id": f"h3_t{index}", "messages": []}) + "\n" for index in range(4)))
    write(run / "sft/F-control/train.jsonl", json.dumps({"task_id": "h3_t4"}) + "\n")
    if ids_file:
        ids = run / "probe-ids.txt"
        write(ids, "".join(task + "\n" for task in PROBE) + "\n")
        return run, ids
    return run, None


def sections(text):
    """整份输出按三个标题切成（vLLM 评测，族内探针，怎么读）。"""
    lines = text.splitlines()
    probe, how = lines.index(PROBE_HEADING), lines.index(HOW_HEADING)
    return lines[:probe], lines[probe:how], lines[how:]


def row(section, label):
    """这一段里以标签开头的那一行；必须恰好一行。"""
    matching = [line for line in section if line.startswith(label)]
    assert len(matching) == 1, section
    return matching[0]


def expect(section, label, pattern):
    """以标签开头的那一行：标签后面（隔一段空白）恰好是 pattern。"""
    line = row(section, label)
    assert re.fullmatch(re.escape(label) + r"\s+" + pattern, line), line
    return line


def verdict(section, prefix):
    matching = [line for line in section if line.startswith(prefix)]
    assert len(matching) == 1, section
    return matching[0]


def test_tables_result_lines_and_how_to_read(tmp_path):
    run, ids = stage(tmp_path)
    dev, probe, how = sections(vllm_summary.report(run, ["D-prefix", "F-control"], ids))
    assert dev[0] == DEV_HEADING and re.fullmatch(r"标签\s+dev 贪心\s+采样通过率\s+pass@k", dev[1])
    base = expect(dev, "base", r"8/16\s+17\.2%（11/64 次尝试）\s+pass@4 37\.5%")
    # 各层写在下一行、缩进；只写有题的层（summary 里三层都写，原有题在这批题里是 4 道）
    assert re.fullmatch(r"\s+原有题 3/4、难题 2/6、变体 3/6", dev[dev.index(base) + 1])
    d_prefix = expect(dev, "D-prefix", r"9/16\s+没采样\s+-")                         # 只做了贪心
    assert re.fullmatch(r"\s+原有题 3/4、难题 3/6、变体 3/6", dev[dev.index(d_prefix) + 1])
    expect(dev, "F-control", r"没有结果（没做）")                                     # dev 没评
    assert verdict(dev, "结果：") == (
        "结果：dev 贪心 base 8/16、D-prefix 9/16；采样通过率 base 17.2%；相对基座 D-prefix +1 道")

    assert probe[0] == PROBE_HEADING
    expect(probe, "base", r"全部 1/8")
    expect(probe, "D-prefix", SPLIT)
    expect(probe, "F-control", r"（评测作废或没做）")
    assert verdict(probe, "结果：") == (
        "结果：族内探针（贪心；每个组依次是训练过的题、没训练过的同族题，括号里是基座）：基座 1/8；"
        "D-prefix 2/4（1/4）、0/4（0/4）；F-control 评测作废或没做")

    assert how == [HOW_HEADING, *HOW_TO_READ]
    assert sum(line.startswith("结果：") for line in dev + probe) == 2               # 每张表后面一行，便于 grep


def test_only_the_tiers_that_have_tasks_are_listed(tmp_path):
    run, ids = stage(tmp_path, base=False)
    evaluation(run, "D-skill", "dev", [("h3_x1", True), ("h3_x2", False), ("h3v_x3", True)])
    dev, _, _ = sections(vllm_summary.report(run, ["D-prefix", "D-skill"], ids))
    skill = expect(dev, "D-skill", r"2/3\s+没采样\s+-")
    assert re.fullmatch(r"\s+难题 1/2、变体 1/1", dev[dev.index(skill) + 1])         # 原有题一道都没有，不写


def test_a_failing_sanity_check_is_shown_next_to_the_void_group(tmp_path):
    """自检没过（适配器的输出和基座完全一样）时没有探针报告：表里这组是"评测作废或没做"，附上和基座比的
    结果；结论行里也有交代。自检过了就不加那句原因——训练过的题复现了几道不作数。"""
    run, ids = stage(tmp_path)
    evaluation(run, "D-skill", "probe", [], sanity=sanity_result(passed=1, changed=0), report=False)
    _, probe, _ = sections(vllm_summary.report(run, ["D-skill"], ids))
    expect(probe, "D-skill", VOID + r"；自检：适配器的输出在 6 道题里有 0 道和基座不同（门槛 1），适配器在 vLLM 里"
                       r"没有生效；训练过的题严格通过 1/6（仅供参考）")
    assert verdict(probe, "结果：").endswith("；D-skill 评测作废或没做")
    # 复现得少、输出却和基座不同：不算没生效（这一组没有 train.jsonl，所以分不出训练过的题）
    evaluation(run, "D-skill", "probe", [(task, False) for task in PROBE], sanity=sanity_result(passed=0, changed=5))
    _, probe, _ = sections(vllm_summary.report(run, ["D-skill"], ids))
    expect(probe, "D-skill", CANNOT_SPLIT + r"全部 0/8（基座 1/8）；自检：适配器的输出在 6 道题里有 5 道和基座"
                       r"不同（门槛 1）；训练过的题严格通过 0/6（仅供参考）")
    # 门槛调高到 3 而只有 1 道不同：同样作废、没有探针报告；说成不到门槛，不说"完全一样"
    (run / "vllm/D-skill/D-skill-probe-greedy.json").unlink()           # 上一步留下的报告不算
    evaluation(run, "D-skill", "probe", [], sanity=sanity_result(passed=6, changed=1, minimum=3), report=False)
    _, probe, _ = sections(vllm_summary.report(run, ["D-skill"], ids))
    expect(probe, "D-skill", VOID + r"；自检：适配器的输出在 6 道题里有 1 道和基座不同（门槛 3），不到门槛 3，"
                       r"按设置算作没有生效；训练过的题严格通过 6/6（仅供参考）")


def sanity_file(run, label, stem, passed=1, total=6, changed=0, minimum=1, compared=True, reason=None, task_ids=None):
    """eval_vllm.py 自检没过时留下的那一个文件 <label>-<stem>-sanity.json（照它实际写出的样子）。"""
    task_ids = task_ids if task_ids is not None else PROBE[:total]
    write(run / "vllm" / label / f"{label}-{stem}-sanity.json", {
        "label": label, "model": label, "adapter": label, "split": "train", "tag": stem,
        "pass": passed, "total": total, "min": minimum, "source": str(run / "sft" / label / "train.jsonl"),
        "task_ids": task_ids, "generation": {"decoding": "greedy", "repeats": 1},
        "results": [{"task_id": task, "strict_pass": index < passed} for index, task in enumerate(task_ids)],
        "changed": changed, "changed_tasks": task_ids[:changed], "compared": compared,
        "compared_tasks": total if compared else 0, "not_compared_reason": reason,
        "base_results": [], "base_unfinished": [],
        "unfinished": [], "timestamp": "2026-11-26T10:00:00+0800"})


def test_a_failed_sanity_check_that_left_only_its_sanity_file_is_still_shown(tmp_path):
    """eval_vllm.py 自检没过时只写 -sanity.json 就以 4 退出，不写 summary 和报告（真脚本串起来跑就是这样的产物）：
    汇总要从这个文件里读出自检结果，写在这一组的后面。"""
    run, ids = stage(tmp_path)
    sanity_file(run, "D-skill", "probe")
    _, probe, _ = sections(vllm_summary.report(run, ["D-skill"], ids))
    expect(probe, "D-skill", VOID + r"；自检：适配器的输出在 6 道题里有 0 道和基座不同（门槛 1），适配器在 vLLM 里"
                       r"没有生效；训练过的题严格通过 1/6（仅供参考）")
    assert verdict(probe, "结果：").endswith("；D-skill 评测作废或没做")     # 结论行照旧简短，原因在表里

    # 探针题里没有这一组训练过的题：没能比较。写明原因和"没能确认"，不说成适配器没生效
    reason = "本次要评的题里没有出现在 --sanity-tasks-file 里的，没有可比较的题"
    sanity_file(run, "D-skill", "probe", passed=0, total=0, compared=False, reason=reason)
    _, probe, _ = sections(vllm_summary.report(run, ["D-skill"], ids))
    expect(probe, "D-skill", VOID + "；自检：没能和基座比较（" + reason + "），没能确认适配器是否生效")

    # 拿不到基座名：同样没能比较，但这一组照常评完了——表里照常有数，后面注明没能确认，结论行里不作废
    reason = "服务的模型列表里分不出 D-skill 挂在哪个基座上，拿不到基座名"
    evaluation(run, "D-skill", "probe", [(task, index < 2) for index, task in enumerate(PROBE)],
               sanity=sanity_result(passed=4, compared=False, reason=reason))
    _, probe, _ = sections(vllm_summary.report(run, ["D-skill"], ids))
    expect(probe, "D-skill", CANNOT_SPLIT + r"全部 2/8（基座 1/8）；自检：没能和基座比较（" + reason
                       + r"），没能确认适配器是否生效")   # 它没有 train.jsonl
    assert not verdict(probe, "结果：").endswith("评测作废或没做")
    # 旧版脚本写的自检结果（只有 pass、total、min，没有和基座比过）：同样说没能确认，不冒充比较过
    evaluation(run, "D-skill", "probe", [(task, index < 2) for index, task in enumerate(PROBE)],
               sanity={"pass": 4, "total": 6, "min": 3})
    _, probe, _ = sections(vllm_summary.report(run, ["D-skill"], ids))
    expect(probe, "D-skill", CANNOT_SPLIT + r"全部 2/8（基座 1/8）；自检：没能和基座比较（" + LEGACY
                       + r"），没能确认适配器是否生效")

    # summary 里的 sanity 和 -sanity.json 都在时以 summary 为准；-sanity.json 读不出来才说一句
    evaluation(run, "D-skill", "probe", [(task, index < 2) for index, task in enumerate(PROBE)],
               sanity=sanity_result())
    _, probe, _ = sections(vllm_summary.report(run, ["D-skill"], ids))
    expect(probe, "D-skill", CANNOT_SPLIT + r"全部 2/8（基座 1/8）；" + CHECKED)   # 它没有 train.jsonl
    folder, part = run / "vllm/D-skill", vllm_summary.load_part
    write(folder / "D-skill-probe-sanity.json", "{not json")
    assert part(run, "D-skill", "probe")["problems"] == []             # 用不着 -sanity.json，就不抱怨
    (folder / "D-skill-probe-greedy.json").unlink()                    # 没有报告了：自检结果只能靠 -sanity.json
    write(folder / "D-skill-probe-summary.json", "{not json")
    assert "D-skill-probe-sanity.json 读不出来" in " ".join(part(run, "D-skill", "probe")["problems"])
    write(folder / "D-skill-probe-sanity.json", {"pass": "many"})
    checked = part(run, "D-skill", "probe")
    assert checked["sanity"] is None and "D-skill-probe-sanity.json 里没有自检的 pass / total" in checked["problems"]


def test_without_a_base_report_the_base_columns_are_left_out(tmp_path):
    run, ids = stage(tmp_path, base=False)
    dev, probe, _ = sections(vllm_summary.report(run, ["D-prefix", "F-control"], ids))
    assert not any(line.startswith("base") for line in dev)                          # 表一没有基座的行
    assert verdict(dev, "结果：") == "结果：dev 贪心 D-prefix 9/16"
    expect(probe, "base", NO_BASE)
    expect(probe, "D-prefix", SPLIT_NO_BASE)
    assert verdict(probe, "结果：") == (
        "结果：族内探针（贪心；每个组依次是训练过的题、没训练过的同族题，括号里是基座）：没有基座的探针报告；"
        "D-prefix 2/4、0/4；F-control 评测作废或没做")


def test_without_an_ids_file_the_probe_tasks_are_the_ones_in_the_reports(tmp_path):
    run, _ = stage(tmp_path, ids_file=False)
    _, probe, _ = sections(vllm_summary.report(run, ["D-prefix"], None))
    expect(probe, "base", r"全部 1/8")
    expect(probe, "D-prefix", SPLIT)
    broken = run / "broken-ids.txt"                                                  # 不是题号：照原样当题号
    write(broken, "{not json\n")
    _, probe, _ = sections(vllm_summary.report(run, ["D-prefix"], broken))
    expect(probe, "base", r"全部 0/0")                                               # 报告里没有的题不计入
    expect(probe, "D-prefix", r"训练过的题：没有这样的题、没训练过的同族题：没有这样的题；" + CHECKED)
    _, probe, _ = sections(vllm_summary.report(run, ["D-prefix"], run / "missing-ids.txt"))
    assert "探针题号文件" in probe[1] and "改用探针报告里出现的题" in probe[1]         # 题号文件读不出：退回报告里的题
    expect(probe, "base", r"全部 1/8")


def test_probe_ids_file_accepts_task_ids_and_task_lines(tmp_path):
    path = tmp_path / "ids.txt"
    write(path, 'h3_a\n\n{"task_id": "h3_b", "split": "train"}\nh3_a\n{"messages": []}\n')
    assert vllm_summary.read_probe_ids(path) == ["h3_a", "h3_b", '{"messages": []}']


def test_missing_reports_are_explained_and_nothing_is_raised(tmp_path):
    run = tmp_path / "nowhere"
    dev, probe, how = sections(vllm_summary.report(run, ["D-prefix", "F-control"], run / "ids.txt"))
    expect(dev, "D-prefix", r"没有结果（没做）")
    expect(dev, "F-control", r"没有结果（没做）")
    assert not any(line.startswith("base") for line in dev)                          # 基座什么都没评：表一不列
    expect(probe, "base", NO_BASE)
    expect(probe, "D-prefix", r"（评测作废或没做）")
    assert verdict(dev, "结果：") == "结果：vLLM 评测还没有可用的结果（summary 和报告没做完）"
    assert verdict(probe, "结果：") == "结果：族内探针没有可用的结果（没做或没评完）"
    assert "探针题号文件" in probe[1]                                                # 题号文件也不在，照样交代
    assert how == [HOW_HEADING, *HOW_TO_READ]

    run, ids = stage(tmp_path)
    write(run / "vllm/base/base-dev-summary.json", "{not json")                      # 写坏了的 summary
    dev, _, _ = sections(vllm_summary.report(run, ["D-prefix"], ids))
    base = row(dev, "base")
    assert "读不出来" in base and "没做完" in base
    assert "D-prefix" in verdict(dev, "结果：")                                      # 别的标签照常


def test_missing_training_data_is_not_the_same_as_no_trained_tasks(tmp_path):
    run, ids = stage(tmp_path)
    (run / "sft/D-prefix/train.jsonl").unlink()
    _, probe, _ = sections(vllm_summary.report(run, ["D-prefix", "F-control"], ids))
    expect(probe, "D-prefix", CANNOT_SPLIT + r"全部 2/8（基座 1/8）；" + CHECKED)
    ending = "；D-prefix 全部 2/8（分不出训练过的题）；F-control 评测作废或没做"
    assert verdict(probe, "结果：").endswith(ending)

    # 训练数据里一道探针题都没有：写明"没有这样的题"，不能写成 0/0
    evaluation(run, "G-other", "probe", [(task, index == 0) for index, task in enumerate(PROBE)])
    write(run / "sft/G-other/train.jsonl", json.dumps({"task_id": "h3_zz"}) + "\n")
    _, probe, _ = sections(vllm_summary.report(run, ["G-other"], ids))
    expect(probe, "G-other", r"训练过的题：没有这样的题、没训练过的同族题 1/8（基座 1/8）")
    assert verdict(probe, "结果：").endswith("；G-other 无、1/8（1/8）")


def test_an_arm_that_dropped_below_the_base_is_reported_as_fewer_tasks(tmp_path):
    run, ids = stage(tmp_path)
    evaluation(run, "F-control", "dev", dev_rows(broke={"h3_d0", "h3_d3"}), sampled=sampled(4, 9, 5))
    dev, _, _ = sections(vllm_summary.report(run, ["F-control"], ids))
    expect(dev, "F-control", r"6/16\s+14\.1%（9/64 次尝试）\s+pass@4 31\.2%")
    assert verdict(dev, "结果：") == (
        "结果：dev 贪心 base 8/16、F-control 6/16；采样通过率 base 17.2%、F-control 14.1%；相对基座 F-control -2 道")


def test_split_by_trained_keeps_the_order_and_tells_when_it_cannot_tell():
    assert vllm_summary.split_by_trained(["a", "b", "c", "d"], {"c", "a", "zzz"}) == (["a", "c"], ["b", "d"])
    assert vllm_summary.split_by_trained(["a"], set()) == ([], ["a"])             # 训练数据在，但一道都没训练过
    assert vllm_summary.split_by_trained(["a", "b"], None) == (None, ["a", "b"])  # 找不到训练数据：分不出
    assert vllm_summary.split_by_trained([], {"a"}) == ([], [])


def test_trained_task_ids_reads_jsonl_and_reports_a_missing_file(tmp_path):
    path = tmp_path / "train.jsonl"
    write(path, '{"task_id": "h3_a"}\n{"task_id": "h3_b"}\n{"messages": []}\n{not json\n\n{"task_id": "h3_a"}\n')
    assert vllm_summary.trained_task_ids(path) == {"h3_a", "h3_b"}
    assert vllm_summary.trained_task_ids(tmp_path / "missing.jsonl") is None
    (tmp_path / "bytes.jsonl").write_bytes(b"\xff\xfe\x00")                         # 不是 UTF-8
    assert vllm_summary.trained_task_ids(tmp_path / "bytes.jsonl") is None
    write(path, '{"task_id": "h3_a", "note": "说明\u2028第二行\u0085"}\n{"task_id": "h3v_c"}\n')   # 原样写着的分隔符
    assert vllm_summary.trained_task_ids(path) == {"h3_a", "h3v_c"}           # 训练数据一条记录一行，行尾只认 "\n"


def test_tier_rows_accepts_both_key_styles_and_counts_from_the_report_when_they_are_missing():
    assert vllm_summary.tier_rows({"default": [0, 0], "hard": [2, 6], "variant": [3, 6]}, None) == [
        ("难题", 2, 6), ("变体", 3, 6)]                                             # 没有题的层不写
    assert vllm_summary.tier_rows({"原有题": {"pass": 1, "total": 4}, "其他": [1, 2]}, None) == [
        ("原有题", 1, 4), ("其他", 1, 2)]                                           # 认不出的层排在后面
    report = {"results": [{"task_id": "h3_a", "strict_pass": True}, {"task_id": "h3_a", "strict_pass": False},
                          {"task_id": "h3v_b", "strict_pass": False}, {"task_id": "v2_c", "strict_pass": True}]}
    assert vllm_summary.tier_rows(None, report) == [("原有题", 1, 1), ("难题", 1, 2), ("变体", 0, 1)]
    assert vllm_summary.tier_rows({"hard": "bad"}, report) == []                    # 写坏了的 by_tier 不致崩


def test_counts_come_from_the_report_when_the_summary_lacks_them(tmp_path):
    run, _ = stage(tmp_path, base=False)
    path = run / "vllm/D-prefix/D-prefix-dev-summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    del summary["greedy"]                                                           # summary 里没有这些字段
    write(path, summary)
    built = vllm_summary.dev_row("D-prefix", vllm_summary.load_part(run, "D-prefix", "dev"))
    assert built["greedy"] == (9, 16) and [tier[0] for tier in built["tiers"]] == ["原有题", "难题", "变体"]


@pytest.mark.parametrize("counts, expected", [(None, "-"), ((2, 5), "2/5"), ((0, 3), "0/3")])
def test_ratio(counts, expected):
    assert vllm_summary.ratio(counts) == expected


def test_command_line_prints_the_same_text_and_deduplicates_arms(tmp_path):
    run, ids = stage(tmp_path)
    command = [sys.executable, str(ROOT / "scripts/vllm_summary.py"), "--run", str(run)]
    result = subprocess.run([*command, "--arms", "D-prefix", "D-prefix", "F-control", "base",
                             "--probe-ids-file", str(ids)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout == vllm_summary.report(run, ["D-prefix", "F-control"], ids) + "\n"
    assert result.stdout.count("2/4（") == 2                                         # 表里一次、结论行一次
    assert result.stdout.count("训练过的题 ") == 1 and result.stderr == ""
    elsewhere = [sys.executable, str(ROOT / "scripts/vllm_summary.py"), "--run", str(tmp_path / "x")]
    nowhere = subprocess.run([*elsewhere, "--arms", "D-prefix"], capture_output=True, text=True, timeout=60)
    assert nowhere.returncode == 0 and "Traceback" not in nowhere.stderr and "没有结果（没做）" in nowhere.stdout
    missing = subprocess.run([*command], capture_output=True, text=True, timeout=60)
    assert missing.returncode == 2 and "--arms" in missing.stderr                    # 一个组名都不给是用法错误


# ---- 层的清单取自 tiers.py：题数为 0 的层不列 ----

FIVE_TIERS = '''"""tiers.py 多了两层（合成难题、合成变体）以后的样子：仓库里那一份由别的作者去改，这里照着演。"""
TIERS = ("default", "hard", "variant", "synthetic_hard", "synthetic_variant")
LABELS = {"default": "原有题", "hard": "难题", "variant": "变体",
          "synthetic_hard": "合成难题", "synthetic_variant": "合成变体"}


def tier_of(task_id: str) -> str:
    if task_id.startswith("s3v_"):
        return "synthetic_variant"
    if task_id.startswith("s3_"):
        return "synthetic_hard"
    if task_id.startswith("h3v_"):
        return "variant"
    if task_id.startswith("h3_"):
        return "hard"
    return "default"
'''


def summary_from(repo, *arguments):
    """把 vllm_summary.py 单独拷进一份临时仓库再运行：repo 里可以带着另一份 tiers.py，也可以什么都不带。"""
    (repo / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "scripts/vllm_summary.py", repo / "scripts/vllm_summary.py")
    return subprocess.run([sys.executable, str(repo / "scripts/vllm_summary.py"), *arguments],
                          capture_output=True, text=True, timeout=60)


def tier_line(section, label):
    """表一里某个标签下一行的各层结果。"""
    return section[section.index(row(section, label)) + 1]


def test_the_tier_definitions_come_from_tiers_py():
    """层名、先后和"题号属于哪一层"都读仓库里的 tiers.py，不在这里另抄一份（以前抄的是三层，加了新层就对不上）。
    读回来的和 agentlab.benchmark.tiers 里的一样——eval_vllm.py 写 by_tier 用的就是它。"""
    from agentlab.benchmark import tiers

    order, labels, tell, note = vllm_summary.load_tiers()
    assert note is None
    assert list(labels) == list(tiers.TIERS) and labels == tiers.LABELS
    assert order == tuple(tiers.LABELS[tier] for tier in tiers.TIERS)
    tasks = ("h3_a", "h3v_b", "v2_c", "s3_a", "s3v_b")
    assert [tell(task) for task in tasks] == [tiers.tier_of(task) for task in tasks]
    assert (vllm_summary.TIER_ORDER, vllm_summary.TIER_KEYS, vllm_summary.FALLBACK_NOTE) == (order, labels, None)
    assert vllm_summary.tier_of is not vllm_summary.builtin_tier_of and vllm_summary.tier_of("h3v_b") == "variant"


def test_new_tiers_in_tiers_py_are_left_out_while_they_have_no_tasks(tmp_path):
    """tiers.py 加了两层，但要评的题只有原有三层：输出和三层的时候一个字都不差——新层没有题就不列，表里没有 0/0。"""
    run, ids = stage(tmp_path)
    now = vllm_summary.report(run, ["D-prefix", "F-control"], ids)         # 仓库里现在的 tiers.py：三层
    repo = tmp_path / "repo-five"
    write(repo / "src/agentlab/benchmark/tiers.py", FIVE_TIERS)
    shown = summary_from(repo, "--run", str(run), "--arms", "D-prefix", "F-control", "--probe-ids-file", str(ids))
    assert shown.returncode == 0 and shown.stderr == ""
    assert shown.stdout == now + "\n"
    dev, _, _ = sections(now)
    assert re.fullmatch(r"\s+原有题 3/4、难题 2/6、变体 3/6", tier_line(dev, "base"))
    assert re.fullmatch(r"\s+原有题 3/4、难题 3/6、变体 3/6", tier_line(dev, "D-prefix"))
    assert "合成" not in shown.stdout and "0/0" not in shown.stdout


def test_new_tiers_are_listed_under_their_own_names_when_they_have_tasks(tmp_path, monkeypatch):
    """有合成题时：按 tiers.py 里的中文名、排在原有三层后面；这一批里没有题的层照样不列。summary 里没有 by_tier、
    要从报告里按题号数的时候也分得清合成题（以前题号前缀是写死的三层，合成题会被算进原有题）。"""
    run, ids = stage(tmp_path / "data", base=False)
    five = tmp_path / "tiers.py"
    write(five, FIVE_TIERS)
    order, labels, tell, note = vllm_summary.load_tiers(five)
    assert note is None and order == ("原有题", "难题", "变体", "合成难题", "合成变体")
    keys = tuple(labels)
    evaluation(run, "D-skill", "dev", [("s3_x1", True), ("s3_x2", False), ("s3v_x3", True), ("h3_x4", False)],
               tiers=(keys, labels, tell))
    for name, value in (("TIER_ORDER", order), ("TIER_KEYS", labels), ("tier_of", tell)):
        monkeypatch.setattr(vllm_summary, name, value)
    dev, _, _ = sections(vllm_summary.report(run, ["D-prefix", "D-skill"], ids))
    expect(dev, "D-skill", r"2/4\s+没采样\s+-")
    assert re.fullmatch(r"\s+难题 0/1、合成难题 1/2、合成变体 1/1", tier_line(dev, "D-skill"))     # 原有题、变体没有题
    expect(dev, "D-prefix", r"9/16\s+没采样\s+-")
    assert re.fullmatch(r"\s+原有题 3/4、难题 3/6、变体 3/6", tier_line(dev, "D-prefix"))        # 合成题一道都没有
    # summary 里没有 by_tier：从贪心报告里按题号数，用的也是 tiers.py 的 tier_of
    path = run / "vllm/D-skill/D-skill-dev-summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    del summary["greedy"]
    write(path, summary)
    built = vllm_summary.dev_row("D-skill", vllm_summary.load_part(run, "D-skill", "dev"))
    assert built["greedy"] == (2, 4) and built["tiers"] == [("难题", 0, 1), ("合成难题", 1, 2), ("合成变体", 1, 1)]
    # 对照：按写死的三层去数，合成题就被算成了原有题
    monkeypatch.setattr(vllm_summary, "tier_of", vllm_summary.builtin_tier_of)
    assert vllm_summary.tier_rows(None, {"results": [dict(task_id=task, strict_pass=passed) for task, passed in
                                                     (("s3_x1", True), ("s3v_x3", True), ("v2_c", False))]}) == [
        ("原有题", 2, 3)]
    # 命令行：脚本旁边放着五层的 tiers.py
    repo = tmp_path / "repo-five"
    write(repo / "src/agentlab/benchmark/tiers.py", FIVE_TIERS)
    shown = summary_from(repo, "--run", str(run), "--arms", "D-prefix", "D-skill", "--probe-ids-file", str(ids))
    assert shown.returncode == 0 and shown.stderr == ""
    assert re.search(r"^\s+难题 0/1、合成难题 1/2、合成变体 1/1$", shown.stdout, re.M)
    assert "合成" not in tier_line(*sections(shown.stdout)[:1], "D-prefix")


def test_without_a_readable_tiers_py_the_three_builtin_tiers_are_used_and_it_is_said(tmp_path):
    """tiers.py 读不到（脚本被单独拷走、文件坏了）：照常出表，层名按内置的三层来写，标准错误里说一声；有题的陌生层
    照 summary 里的键名原样列出，没有题的层（不管认不认得）都不列。"""
    missing, broken = tmp_path / "nowhere/tiers.py", tmp_path / "broken.py"
    write(broken, "TIERS = ('default',)\n")                               # 缺 LABELS 和 tier_of
    tail = "：层名按内置的三层（原有题、难题、变体）来写，别的层照 summary 里的键名原样列出。把脚本放在仓库里运行"
    for path in (missing, broken):
        order, labels, tell, note = vllm_summary.load_tiers(path)
        assert order == ("原有题", "难题", "变体") and labels == vllm_summary.BUILT_IN
        assert tell is vllm_summary.builtin_tier_of and tell("h3v_b") == "variant"
        assert note.startswith(f"层的定义读不到：{path}（") and note.endswith(tail)
    assert vllm_summary.load_tiers(broken)[3] == f"层的定义读不到：{broken}（KeyError: 'LABELS'）" + tail
    assert vllm_summary.tier_rows(
        {"default": [0, 0], "hard": [0, 0], "variant": [0, 0], "synthetic_hard": [1, 2], "synthetic_variant": [0, 0]},
        None) == [("synthetic_hard", 1, 2)]

    run, ids = stage(tmp_path / "data")
    shown = summary_from(tmp_path / "repo-bare", "--run", str(run), "--arms", "D-prefix", "F-control",
                         "--probe-ids-file", str(ids))
    assert shown.returncode == 0 and "Traceback" not in shown.stderr
    assert shown.stderr.startswith("层的定义读不到：") and shown.stderr.count("\n") == 1
    assert shown.stdout == vllm_summary.report(run, ["D-prefix", "F-control"], ids) + "\n"     # 三层的题：表和平时一样
