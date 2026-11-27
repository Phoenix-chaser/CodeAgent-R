# -*- coding: utf-8 -*-
"""任务分层多了两层（合成难题、合成变体）以后：tier_of 认新的题号前缀，tier_report 只列基座这一行里有的层。

只评了原有三层的题时，scripts/tier_report.py 的输出要和只有三层的时候一字不差：下面存着的几段原文，
是 tiers.py 加层之前它打印出来的。
"""
import ast
import json
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import tier_report  # noqa: E402
from agentlab.benchmark import tiers  # noqa: E402

BASE = [("v2_a", True), ("v2_b", False), ("h3_x_001", False), ("h3v_y_1", False), ("h3v_y_2", True)]
TUNED = [("v2_a", True), ("v2_b", True), ("h3_x_001", False), ("h3v_y_1", True), ("h3v_y_2", False)]

# 加层之前的原文：两份报告、只有一行、一个选型目录、直接调 table()
BEFORE_TWO_REPORTS = (
    "模型   原有题   难题     变体     合计     \n"
    "base   1/2      0/1      1/2      2/5\n"
    "tuned  2/2      0/1      1/2      3/5\n"
    "tuned 相对 base：变好 2 道、变差 1 道（各层变好/变差：原有题 +1/-0、难题 +0/-0、变体 +1/-1）\n")
BEFORE_ONE_ROW = "模型  原有题   难题     变体     合计     \nbase  0/1      0/0      0/0      0/1\n"
BEFORE_SELECTION = (
    "模型             原有题   难题     变体     合计     \n"
    "base             1/1      0/1      0/1      1/3\n"
    "epoch 1（选中）  1/1      1/1      0/1      2/3\n"
    "epoch 1（选中） 相对 base：变好 1 道、变差 0 道（各层变好/变差：原有题 +0/-0、难题 +1/-0、变体 +0/-0）\n")
BEFORE_TABLE = (
    "模型                        原有题   难题     变体     合计     \n"
    "base                        1/2      0/1      1/2      2/5\n"
    "tuned                       2/2      0/1      1/2      3/5\n"
    "epoch 3（选中） · D-prefix  0/1      0/0      0/0      0/1")


def evaluation(tasks):
    """一份合成的评测报告：tasks 是 (题号, 是否通过)。"""
    return {"results": [{"task_id": task, "strict_pass": passed} for task, passed in tasks]}


def report(capsys, monkeypatch, arguments):
    """照命令行的样子跑一次 tier_report.main()，返回它打印的内容。"""
    monkeypatch.setattr(sys, "argv", ["tier_report.py", *arguments])
    capsys.readouterr()
    tier_report.main()
    return capsys.readouterr().out


def test_new_task_ids_belong_to_the_new_tiers():
    assert tiers.TIERS == ("default", "hard", "variant", "synth_hard", "synth_variant")
    assert tiers.LABELS == {"default": "原有题", "hard": "难题", "variant": "变体",
                            "synth_hard": "合成难题", "synth_variant": "合成变体"}
    assert list(tiers.LABELS) == list(tiers.TIERS)                  # 每一层都有中文名，先后一致
    for task, tier in (("v2_a_001", "default"), ("h3_state_001", "hard"), ("h3v_state_1", "variant"),
                       ("s3_state_inventory_001", "synth_hard"), ("s3v_state_inventory_1", "synth_variant"),
                       ("s3v_state_inventory_12", "synth_variant"),
                       ("s3", "default"), ("s3v", "default"), ("xs3_a_001", "default"), ("S3_a_001", "default"),
                       ("s3a_b_001", "default")):
        assert tiers.tier_of(task) == tier, task


def test_tiers_py_stands_alone():
    """scripts/vllm_summary.py 按文件路径单独执行 tiers.py 来读层的定义：它只能依赖标准库。"""
    source = (ROOT / "src/agentlab/benchmark/tiers.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = [alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names]
    imported += [node.module for node in tree.body if isinstance(node, ast.ImportFrom)]
    assert imported == ["__future__"]
    namespace = runpy.run_path(str(ROOT / "src/agentlab/benchmark/tiers.py"))
    assert namespace["TIERS"] == tiers.TIERS and namespace["LABELS"] == tiers.LABELS
    assert [namespace["tier_of"](task) for task in ("s3_a_001", "s3v_a_1", "h3_a_001", "v2_a")] == [
        "synth_hard", "synth_variant", "hard", "default"]


def test_tier_counts_has_an_entry_for_every_tier():
    assert tier_report.tier_counts(evaluation(BASE)) == {
        "default": [1, 2], "hard": [0, 1], "variant": [1, 2], "synth_hard": [0, 0], "synth_variant": [0, 0]}
    assert tier_report.shown_tiers(tier_report.tier_counts(evaluation(BASE))) == ("default", "hard", "variant")
    # 没有某一层的计数（旧版本的报告）也当作 0
    assert tier_report.shown_tiers({"default": [1, 1]}) == ("default", "hard", "variant")


def test_the_output_is_unchanged_while_the_reports_only_have_the_three_original_tiers(tmp_path, capsys, monkeypatch):
    base, tuned, bare = tmp_path / "base.json", tmp_path / "tuned.json", tmp_path / "bare.json"
    base.write_text(json.dumps(evaluation(BASE)), encoding="utf-8")
    tuned.write_text(json.dumps(evaluation(TUNED)), encoding="utf-8")
    bare.write_text(json.dumps(evaluation([("v2_a", False)])), encoding="utf-8")
    assert report(capsys, monkeypatch, [f"base={base}", f"tuned={tuned}"]) == BEFORE_TWO_REPORTS
    assert report(capsys, monkeypatch, [f"base={bare}"]) == BEFORE_ONE_ROW
    selection = tmp_path / "results/A-verified"
    selection.mkdir(parents=True)
    for epoch, tasks in ((0, [("v2_a", True), ("h3_x_001", False), ("h3v_y_1", False)]),
                         (1, [("v2_a", True), ("h3_x_001", True), ("h3v_y_1", False)])):
        (selection / f"epoch-{epoch:03d}-dev.json").write_text(json.dumps(evaluation(tasks)), encoding="utf-8")
    (selection / "selection.json").write_text(json.dumps(
        {"selected_epoch": 1, "candidates": [{"epoch": 0}, {"epoch": 1}]}), encoding="utf-8")
    assert report(capsys, monkeypatch, [str(selection)]) == BEFORE_SELECTION
    rows = [("base", tier_report.tier_counts(evaluation(BASE))), ("tuned", tier_report.tier_counts(evaluation(TUNED))),
            ("epoch 3（选中） · D-prefix", tier_report.tier_counts(evaluation([("v2_a", False)])))]
    assert tier_report.table(rows) == BEFORE_TABLE
    change = tier_report.transitions(evaluation(BASE), evaluation(TUNED))
    assert change["tiers"] == {"default": [1, 0], "hard": [0, 0], "variant": [1, 1]} and change["total"] == [2, 1]


def test_synthetic_tasks_in_the_base_row_add_their_own_columns(tmp_path, capsys, monkeypatch):
    plain = [("v2_a", False), ("h3_a_001", True)]
    synthetic = [("s3_b_001", False), ("s3v_b_1", False)]
    better = [("v2_a", True), ("h3_a_001", True), ("s3_b_001", False), ("s3v_b_1", True)]
    base, tuned = tmp_path / "base.json", tmp_path / "tuned.json"
    base.write_text(json.dumps(evaluation([*plain, *synthetic])), encoding="utf-8")
    tuned.write_text(json.dumps(evaluation(better)), encoding="utf-8")
    # 合成题的两栏排在原有三层后面；每一栏 9 列宽，所以合成难题后面只空一格
    assert report(capsys, monkeypatch, [f"base={base}", f"tuned={tuned}"]) == (
        "模型   原有题   难题     变体     合成难题 合成变体 合计     \n"
        "base   0/1      1/1      0/0      0/1      0/1      1/4\n"
        "tuned  1/1      1/1      0/0      0/1      1/1      3/4\n"
        "tuned 相对 base：变好 2 道、变差 0 道"
        "（各层变好/变差：原有题 +1/-0、难题 +0/-0、变体 +0/-0、合成难题 +0/-0、合成变体 +1/-0）\n")
    change = tier_report.transitions(evaluation([*plain, *synthetic]), evaluation(better))
    assert change["tiers"] == {"default": [1, 0], "hard": [0, 0], "variant": [0, 0],
                               "synth_hard": [0, 0], "synth_variant": [1, 0]}
    assert change["total"] == [2, 0]
    # 基座里只有合成难题：合成变体那一栏不列（变体照旧，哪怕题数是 0）
    base.write_text(json.dumps(evaluation([*plain, ("s3_b_001", False)])), encoding="utf-8")
    fixed_up = [("v2_a", True), ("h3_a_001", True), ("s3_b_001", True)]
    (tmp_path / "later.json").write_text(json.dumps(evaluation(fixed_up)), encoding="utf-8")
    lines = report(capsys, monkeypatch, [f"base={base}", f"later={tmp_path / 'later.json'}"]).splitlines()
    assert lines == [
        "模型   原有题   难题     变体     合成难题 合计     ",
        "base   0/1      1/1      0/0      0/1      1/3",
        "later  1/1      1/1      0/0      1/1      3/3",
        "later 相对 base：变好 2 道、变差 0 道"
        "（各层变好/变差：原有题 +1/-0、难题 +0/-0、变体 +0/-0、合成难题 +1/-0）"]


def test_tasks_of_a_tier_the_base_row_does_not_have_are_counted_only_in_the_total():
    """列哪些层只看第一行：后面几行里多出来的层不另起一栏，它们的题只算进合计。"""
    three = tier_report.tier_counts(evaluation(BASE))
    five = tier_report.tier_counts(evaluation([("v2_a", True), ("s3_b_001", True), ("s3v_b_1", False)]))
    shown = tier_report.table([("base", three), ("late", five)]).splitlines()
    assert [line.split() for line in shown] == [
        ["模型", "原有题", "难题", "变体", "合计"], ["base", "1/2", "0/1", "1/2", "2/5"],
        ["late", "1/1", "0/0", "0/0", "2/3"]]


def test_two_reports_with_different_tasks_still_cannot_be_compared():
    with pytest.raises(ValueError, match="题目不一样"):
        tier_report.transitions(evaluation(BASE), evaluation([("v2_a", True)]))


def test_the_sampling_script_accepts_the_new_tiers_on_its_own():
    """sample_rft.py 的 --tiers 拿 tiers.py 的 TIERS 当合法取值：新层不用改它就能点名。这里只看它的检查：
    导入它会把判分、建任务的模块一起拉进来，所以放到最后、只做一次。"""
    import sample_rft

    assert sample_rft.TIERS is tiers.TIERS and sample_rft.tier_of is tiers.tier_of
    stats = sample_rft.summarise(
        [{"task_id": "s3_b_001", "provenance": {"strict_pass": True}},
         {"task_id": "s3v_b_1", "provenance": {"strict_pass": False}}],
        [{"task_id": "s3_b_001"}, {"task_id": "s3v_b_1"}], samples=1)
    assert [stats["tiers"][tier]["tasks"] for tier in tiers.TIERS] == [0, 0, 0, 1, 1]
    assert stats["tiers"]["synth_variant"]["pass_at_1"] == 0.0 and stats["tiers"]["synth_hard"]["solved_tasks"] == 1
