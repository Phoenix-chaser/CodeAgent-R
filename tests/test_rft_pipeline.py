# -*- coding: utf-8 -*-
"""RFT 第 1 轮用到的几段胶水：不验证的对照组数据、分层汇总、选型时复用已有的 dev 报告。"""
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import rft_control
import tier_report
from agentlab.training.hf_sft import SFTDataset
from test_sft_pipeline import TemplateTokenizer


def graded(task, index, passed, code="value = 1\n"):
    return {"sample_id": f"rft-{task}-{index:03d}", "task_id": task, "source_group": f"group-{task}",
            "split": "train", "task_hash": "hash", "candidate_idx": index,
            "messages": [{"role": "user", "content": f"fix {task}"}],
            "target": {"role": "assistant", "content": code},
            "provenance": {"strict_pass": passed, "verdict": "strict_pass" if passed else "partial"}}


SAMPLES = [
    graded("h3v_a_001", 0, True),
    graded("h3v_a_001", 1, False, "value = 2\n"),
    graded("h3v_a_001", 2, False, "抱歉，我不确定怎么改。"),      # 不是代码：不算格式完整
    graded("h3_b_001", 0, False, "value = 3\n"),
    graded("h3_b_001", 1, True),
    graded("h3v_c_001", 0, True),                                 # 不在验证组训练集里的题
]
VERIFIED_TRAIN = [{"task_id": "h3v_a_001"}, {"task_id": "h3_b_001"}]


def test_control_picks_one_well_formed_sample_per_solved_task_regardless_of_correctness():
    samples, stats = rft_control.build_control(SAMPLES, VERIFIED_TRAIN, seed=42)
    assert [sample["task_id"] for sample in samples] == ["h3_b_001", "h3v_a_001"]      # 不含 h3v_c_001
    assert all(sample["selection"] == {"policy": "control_unverified_random", "pool": 2} for sample in samples)
    assert all(sample["provenance"]["control"] == "unverified_random_sample" for sample in samples)
    assert samples == rft_control.build_control(SAMPLES, VERIFIED_TRAIN, seed=42)[0]   # 同一种子可复现
    assert stats["actually_passing"] == sum(sample["provenance"]["strict_pass"] for sample in samples)
    assert stats["label_validation"] == "none_by_design_control"
    assert stats["by_tier"] == {"变体": 1, "难题": 1}
    picked = {(sample["task_id"], sample["candidate_idx"])
              for seed in range(40) for sample in rft_control.build_control(SAMPLES, VERIFIED_TRAIN, seed)[0]}
    assert picked == {("h3v_a_001", 0), ("h3v_a_001", 1), ("h3_b_001", 0), ("h3_b_001", 1)}   # 那句道歉从未入选
    with pytest.raises(ValueError, match="没有格式完整"):
        rft_control.build_control([graded("h3_b_001", 0, True)], [{"task_id": "h3v_a_001"}], seed=1)


def test_control_cli_writes_a_dataset_that_only_loads_in_full_with_allow_unverified(tmp_path):
    verified = tmp_path / "verified"
    verified.mkdir()
    (verified / "train.jsonl").write_text("".join(json.dumps(item) + "\n" for item in VERIFIED_TRAIN),
                                          encoding="utf-8")
    (verified / "dev.jsonl").write_text(json.dumps({"task_id": "dev-task", "split": "dev"}) + "\n",
                                        encoding="utf-8")
    (tmp_path / "graded.jsonl").write_text("".join(json.dumps(item) + "\n" for item in SAMPLES),
                                           encoding="utf-8")
    command = [sys.executable, str(ROOT / "scripts/rft_control.py"), "--graded", str(tmp_path / "graded.jsonl"),
               "--verified-dir", str(verified), "--output-dir", str(tmp_path / "control"), "--seed", "7"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0 and "对照组：2 条 / 2 题" in result.stdout, result.stderr
    output = tmp_path / "control"
    assert (output / "dev.jsonl").read_bytes() == (verified / "dev.jsonl").read_bytes()
    records = [json.loads(line) for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    passing = sum(item["provenance"]["strict_pass"] for item in records)
    assert len(records) == 2 and all(item["provenance"]["control"] == "unverified_random_sample"
                                     for item in records)
    # 默认的训练入口会把没通过的样本筛掉；加 --allow-unverified 才整份加载
    strict = SFTDataset(output / "train.jsonl", TemplateTokenizer(), split="train")
    lax = SFTDataset(output / "train.jsonl", TemplateTokenizer(), split="train", strict_pass_only=False)
    assert len(lax) == 2 and len(strict) == passing and strict.stats.skipped_fail == 2 - passing
    again = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert again.returncode == 2 and "must be empty" in again.stderr       # 不覆盖已有的数据
    spec = importlib.util.spec_from_file_location("rft_train_sft", ROOT / "scripts/train_sft.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.build_parser().parse_args(["--data", "x"]).allow_unverified is False
    assert module.build_parser().parse_args(["--data", "x", "--allow-unverified"]).allow_unverified is True


def evaluation(tasks):
    """一份合成的评测报告：tasks 是 (题号, 是否通过)。"""
    return {"results": [{"task_id": task, "strict_pass": passed} for task, passed in tasks]}


BASE = [("v2_a", True), ("v2_b", False), ("h3_x_001", False), ("h3v_y_1", False), ("h3v_y_2", True)]
TUNED = [("v2_a", True), ("v2_b", True), ("h3_x_001", False), ("h3v_y_1", True), ("h3v_y_2", False)]


def test_tier_report_splits_counts_by_tier_and_compares_to_the_first_row(tmp_path, capsys, monkeypatch):
    base, tuned = tmp_path / "base.json", tmp_path / "tuned.json"
    base.write_text(json.dumps(evaluation(BASE)), encoding="utf-8")
    tuned.write_text(json.dumps(evaluation(TUNED)), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["tier_report.py", f"base={base}", f"tuned={tuned}"])
    tier_report.main()
    out = capsys.readouterr().out
    assert re.search(r"模型\s+原有题\s+难题\s+变体\s+合计", out)
    assert re.search(r"base\s+1/2\s+0/1\s+1/2\s+2/5", out)
    assert re.search(r"tuned\s+2/2\s+0/1\s+1/2\s+3/5", out)
    assert "tuned 相对 base：变好 2 道、变差 1 道（各层变好/变差：原有题 +1/-0、难题 +0/-0、变体 +1/-1）" in out
    assert tier_report.transitions(evaluation(BASE), evaluation(TUNED))["tiers"] == {
        "default": [1, 0], "hard": [0, 0], "variant": [1, 1]}
    with pytest.raises(ValueError, match="题目不一样"):
        tier_report.transitions(evaluation(BASE), evaluation([("v2_a", True)]))


def test_tier_report_expands_selection_directories_and_marks_the_chosen_epoch(tmp_path, capsys, monkeypatch):
    (tmp_path / "epoch-000-dev.json").write_text(
        json.dumps(evaluation([("h3v_y_1", False), ("v2_a", True)])), encoding="utf-8")
    (tmp_path / "epoch-001-dev.json").write_text(
        json.dumps(evaluation([("h3v_y_1", True), ("v2_a", True)])), encoding="utf-8")
    (tmp_path / "selection.json").write_text(json.dumps({
        "selected_epoch": 1, "candidates": [{"epoch": 0}, {"epoch": 1}]}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["tier_report.py", str(tmp_path)])
    tier_report.main()
    out = capsys.readouterr().out
    assert re.search(r"base\s+1/1\s+0/0\s+0/1\s+1/2", out)
    assert re.search(r"epoch 1（选中）\s+1/1\s+0/0\s+1/1\s+2/2", out)
    assert "epoch 1（选中） 相对 base：变好 1 道、变差 0 道" in out
    (tmp_path / "selection.json").write_text(json.dumps({
        "selected_epoch": 0, "candidates": [{"epoch": 0}]}), encoding="utf-8")
    tier_report.main()
    assert re.search(r"base（保留基座）\s+1/1\s+0/0\s+0/1\s+1/2", capsys.readouterr().out)


def test_tier_report_names_the_group_of_each_row_when_several_directories_are_given(tmp_path, capsys, monkeypatch):
    """多个选型目录一起汇总：各组的行都叫 base / epoch N，不带组名分不清。行名后面带上目录名，
    开头照旧是 base / epoch（便于 grep）。"""
    for group, chosen in (("A-verified", 1), ("C-control", 0)):
        folder = tmp_path / "results" / group
        folder.mkdir(parents=True)
        for epoch, passed in ((0, False), (1, True)):
            (folder / f"epoch-{epoch:03d}-dev.json").write_text(
                json.dumps(evaluation([("h3v_y_1", passed), ("v2_a", True)])), encoding="utf-8")
        (folder / "selection.json").write_text(json.dumps({
            "selected_epoch": chosen, "candidates": [{"epoch": 0}, {"epoch": 1}]}), encoding="utf-8")
    directories = [str(tmp_path / "results" / group) for group in ("A-verified", "C-control")]
    monkeypatch.setattr(sys, "argv", ["tier_report.py", *directories])
    tier_report.main()
    out = capsys.readouterr().out
    rows = [line.split() for line in out.splitlines() if "相对" not in line]
    assert rows == [
        ["模型", "原有题", "难题", "变体", "合计"],
        ["base", "·", "A-verified", "1/1", "0/0", "0/1", "1/2"],
        ["epoch", "1（选中）", "·", "A-verified", "1/1", "0/0", "1/1", "2/2"],
        ["base（保留基座）", "·", "C-control", "1/1", "0/0", "0/1", "1/2"],
        ["epoch", "1", "·", "C-control", "1/1", "0/0", "1/1", "2/2"]]
    assert [line for line in out.splitlines() if "相对" in line] == [
        "epoch 1（选中） · A-verified 相对 base · A-verified：变好 1 道、变差 0 道"
        "（各层变好/变差：原有题 +0/-0、难题 +0/-0、变体 +1/-0）",
        "base（保留基座） · C-control 相对 base · A-verified：变好 0 道、变差 0 道"
        "（各层变好/变差：原有题 +0/-0、难题 +0/-0、变体 +0/-0）",
        "epoch 1 · C-control 相对 base · A-verified：变好 1 道、变差 0 道"
        "（各层变好/变差：原有题 +0/-0、难题 +0/-0、变体 +1/-0）"]
    # 只给一个目录：行名不带组名，和原来一样
    monkeypatch.setattr(sys, "argv", ["tier_report.py", directories[0]])
    tier_report.main()
    assert [line.split() for line in capsys.readouterr().out.splitlines() if "相对" not in line][1:] == [
        ["base", "1/1", "0/0", "0/1", "1/2"], ["epoch", "1（选中）", "1/1", "0/0", "1/1", "2/2"]]


# ───── 引导采样那一轮：按题配对的对照组，以及把配不上的题从引导组里去掉 ─────
import random

import prepare_data
from agentlab.benchmark.protocol import build_messages, spec_fingerprint
from test_rft_sampling import FakeVLLM, lines


def control_cli(*arguments, cwd=None):
    """跑一遍 rft_control.py；参数里的 Path 原样传过去。cwd 不给时在仓库根运行。"""
    return subprocess.run([sys.executable, str(ROOT / "scripts/rft_control.py"), *map(str, arguments)],
                          capture_output=True, text=True, timeout=60, cwd=cwd)


def dataset(directory, train, dev):
    """一个 prepare_data.py 风格的数据目录：只有 train.jsonl 和 dev.jsonl。"""
    directory.mkdir(parents=True)
    (directory / "train.jsonl").write_text("".join(json.dumps(item) + "\n" for item in train), encoding="utf-8")
    (directory / "dev.jsonl").write_text("".join(json.dumps(item) + "\n" for item in dev), encoding="utf-8")
    return directory


# 配对对照的池子：h3v_a 有一份、h3_b 有两份"是代码、但没通过"的采样；ELIGIBLE 是手写的预期池子
MATCHED_GRADED = [
    graded("h3v_a_001", 0, True),
    graded("h3v_a_001", 1, False, "value = 2\n"),
    graded("h3v_a_001", 2, False, "抱歉，我不确定怎么改。"),      # 不是代码：不算格式完整
    graded("h3_b_001", 0, False, "value = 3\n"),
    graded("h3_b_001", 1, False, "value = 4\n"),
    graded("h3_b_001", 2, True),
    graded("h3v_c_001", 0, True),                                 # 只有通过的：配不上
    graded("h3v_d_001", 0, False, "我说不好。"),                   # 没通过的都不是代码：配不上
    graded("h3v_e_001", 0, False, "value = 5\n"),                  # 不是引导题：不取
]
MATCHED_TASKS = ["h3v_c_001", "h3v_a_001", "h3v_d_001", "h3_b_001"]
ELIGIBLE = {"h3v_a_001": [MATCHED_GRADED[1]], "h3_b_001": [MATCHED_GRADED[3], MATCHED_GRADED[4]]}


def test_matched_control_pairs_each_guided_task_with_one_failing_well_formed_sample():
    samples, stats = rft_control.build_matched_control(MATCHED_GRADED, MATCHED_TASKS, seed=42)
    assert [sample["task_id"] for sample in samples] == ["h3_b_001", "h3v_a_001"]     # 排过序；配不上的不占位
    for sample in samples:
        assert sample["provenance"]["strict_pass"] is False and rft_control.well_formed(sample)
        assert sample["provenance"]["control"] == "failing_random_sample"
        assert sample["selection"] == {"policy": "control_failing_random",
                                       "pool": len(ELIGIBLE[sample["task_id"]])}     # 通过的和不是代码的不进池子
        original = next(item for item in ELIGIBLE[sample["task_id"]]
                        if item["candidate_idx"] == sample["candidate_idx"])
        assert sample["sample_id"] == original["sample_id"]           # 修饰的只有 sample_id、selection、provenance
        assert {key: sample[key] for key in original if key != "provenance"} == {
            key: value for key, value in original.items() if key != "provenance"}
        assert sample["provenance"] == {**original["provenance"], "control": "failing_random_sample"}
    assert stats["unmatched"] == ["h3v_c_001", "h3v_d_001"]          # 不抛异常，如实报告
    assert (stats["samples"], stats["tasks"]) == (2, 4)
    assert stats["by_tier"] == {"难题": 1, "变体": 1}
    assert stats["label_validation"] == "none_by_design_control" and stats["limitation"]
    assert rft_control.build_matched_control(MATCHED_GRADED, MATCHED_TASKS, seed=42) == (samples, stats)   # 可复现
    # 题号的顺序和重复都不影响结果：种子按题分开，池子按记录在 graded 里出现的先后
    shuffled = ["h3_b_001", "h3v_d_001", "h3v_a_001", "h3v_c_001", "h3v_a_001"]
    assert rft_control.build_matched_control(MATCHED_GRADED, shuffled, seed=42) == (samples, stats)
    picked = set()
    for seed in range(60):
        for sample in rft_control.build_matched_control(MATCHED_GRADED, MATCHED_TASKS, seed)[0]:
            task = sample["task_id"]
            assert sample["candidate_idx"] == random.Random(f"{seed}:{task}:failing").choice(
                ELIGIBLE[task])["candidate_idx"]
            picked.add((task, sample["candidate_idx"]))
    assert picked == {("h3v_a_001", 1), ("h3_b_001", 0), ("h3_b_001", 1)}    # 通过的和不是代码的从来没入选
    anonymous = {key: value for key, value in MATCHED_GRADED[1].items() if key != "sample_id"}
    (only,), _ = rft_control.build_matched_control([anonymous], ["h3v_a_001"], seed=1)
    assert only["sample_id"] == "h3v_a_001-control" and only["target"] == anonymous["target"]
    empty, empty_stats = rft_control.build_matched_control(MATCHED_GRADED, [], seed=42)
    assert empty == [] and empty_stats["unmatched"] == [] and (empty_stats["samples"], empty_stats["tasks"]) == (0, 0)


def test_matched_control_cli_copies_the_base_then_adds_one_failing_answer_per_guided_task(tmp_path):
    base_records = [graded("v2_a", 0, True), graded("v2_b", 1, True, "value = 2\n")]
    dev_records = [{**graded("v2_dev", 0, True), "split": "dev"}]
    base = dataset(tmp_path / "verified", base_records, dev_records)
    solved = [graded("h3v_a_001", 0, True, "def solve():\n    return 1\n"),
              graded("h3v_c_001", 0, True, "def solve():\n    return 3\n"),
              graded("h3_b_001", 0, True, "def solve():\n    return 2\n")]
    prefix = dataset(tmp_path / "D-prefix", [*base_records, *solved[:2]], [])
    skill = dataset(tmp_path / "D-skill", [*base_records, solved[2]], [])
    graded_file = tmp_path / "train-graded.jsonl"
    graded_file.write_text("".join(json.dumps(item) + "\n" for item in MATCHED_GRADED), encoding="utf-8")
    common = ["--graded", graded_file, "--base-dir", base, "--seed", "7"]
    output = tmp_path / "F-control"

    result = control_cli(*common, "--guided-from", prefix, skill, "--output-dir", output)
    assert result.returncode == 0, result.stderr
    # 引导题是两组训练集的并集减去基础集：h3v_a（prefix）、h3_b（skill）配上，h3v_c（prefix）配不上
    assert "结果：配对对照 = 基础 2 条 + 引导题 2 条（来自 2 组）；配不上的题 1 道 ['h3v_c_001']" in result.stdout
    assert str(output / "unmatched-tasks.txt") in result.stdout and "--drop-tasks-file" in result.stdout
    # 开头就是基础集的训练集，一个字节都不差；后面每道配上的引导题恰有一条
    assert (output / "train.jsonl").read_bytes().startswith((base / "train.jsonl").read_bytes())
    records = [json.loads(line) for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert records[:2] == base_records
    control = records[2:]
    assert [item["task_id"] for item in control] == ["h3_b_001", "h3v_a_001"]
    assert all(item["provenance"]["strict_pass"] is False
               and item["selection"]["policy"] == "control_failing_random"
               and item["provenance"]["control"] == "failing_random_sample" for item in control)
    for item in control:                     # 抽到的是这道题自己的、没通过的那几份之一
        assert item["target"]["content"] in [record["target"]["content"] for record in ELIGIBLE[item["task_id"]]]
    assert (output / "dev.jsonl").read_bytes() == (base / "dev.jsonl").read_bytes()
    assert (output / "unmatched-tasks.txt").read_text(encoding="utf-8") == "h3v_c_001\n"
    manifest = json.loads((output / "dataset-manifest.json").read_text(encoding="utf-8"))
    assert manifest["base_dir"] == str(base) and manifest["guided_from"] == [str(prefix), str(skill)]
    assert (manifest["samples"], manifest["tasks"], manifest["base_samples"]) == (2, 3, 2)
    assert manifest["unmatched"] == ["h3v_c_001"] and manifest["label_validation"] == "none_by_design_control"
    # 和上一轮的对照组一样：不加 --allow-unverified 时，训练入口只加载判分通过的记录
    strict = SFTDataset(output / "train.jsonl", TemplateTokenizer(), split="train")
    lax = SFTDataset(output / "train.jsonl", TemplateTokenizer(), split="train", strict_pass_only=False)
    assert len(strict) == 2 and strict.stats.skipped_fail == 2 and len(lax) == 4

    # 引导组可以重复写同一个参数；两种写法的结果完全一样
    other = tmp_path / "F-control-again"
    repeated = control_cli(*common, "--guided-from", prefix, "--guided-from", skill, "--output-dir", other)
    assert repeated.returncode == 0, repeated.stderr
    for name in ("train.jsonl", "dev.jsonl", "unmatched-tasks.txt", "dataset-manifest.json"):
        assert (other / name).read_bytes() == (output / name).read_bytes(), name
    # 不覆盖已有的数据
    again = control_cli(*common, "--guided-from", prefix, skill, "--output-dir", output)
    assert again.returncode == 2 and "must be empty" in again.stderr


def test_matched_control_cli_lists_every_unmatched_task_and_writes_an_empty_file_when_all_pair_up(tmp_path):
    base_records = [graded("v2_a", 0, True)]
    base = dataset(tmp_path / "verified", base_records, [])
    graded_file = tmp_path / "train-graded.jsonl"
    graded_file.write_text(json.dumps(graded("h3v_a_001", 1, False, "value = 2\n")) + "\n", encoding="utf-8")
    # 十道题在 graded 里一条记录都没有，另有一道只有没通过的采样
    ten = [graded(f"h3v_x_{index:03d}", 0, True) for index in range(10)]
    both = dataset(tmp_path / "D", [*base_records, graded("h3v_a_001", 0, True), *ten], [])
    output = tmp_path / "F-control"
    result = control_cli("--graded", graded_file, "--base-dir", base, "--guided-from", both,
                         "--output-dir", output)
    assert result.returncode == 0, result.stderr
    assert ("结果：配对对照 = 基础 1 条 + 引导题 1 条（来自 1 组）；配不上的题 10 道 "
            "['h3v_x_000', 'h3v_x_001', 'h3v_x_002', 'h3v_x_003', 'h3v_x_004', 'h3v_x_005', 'h3v_x_006', "
            "'h3v_x_007']…") in result.stdout                      # 屏幕上只列前 8 个
    assert (output / "unmatched-tasks.txt").read_text(encoding="utf-8") == "".join(
        f"h3v_x_{index:03d}\n" for index in range(10))
    assert [item["task_id"] for item in lines(output / "train.jsonl")] == ["v2_a", "h3v_a_001"]

    paired = dataset(tmp_path / "D-paired", [*base_records, graded("h3v_a_001", 0, True)], [])
    clean = tmp_path / "F-clean"
    result = control_cli("--graded", graded_file, "--base-dir", base, "--guided-from", paired,
                         "--output-dir", clean)
    assert result.returncode == 0 and "配不上的题 0 道 []" in result.stdout, result.stderr
    assert (clean / "unmatched-tasks.txt").exists() and (clean / "unmatched-tasks.txt").read_text() == ""


def test_control_cli_rejects_mixing_the_two_modes_or_giving_neither(tmp_path):
    verified = dataset(tmp_path / "verified", [graded("v2_a", 0, True)], [])
    graded_file = tmp_path / "train-graded.jsonl"
    graded_file.write_text("".join(json.dumps(item) + "\n" for item in MATCHED_GRADED), encoding="utf-8")
    output = tmp_path / "out"
    cases = [
        ([], "二选一"),                                                               # 两种模式都没给
        (["--verified-dir", verified, "--base-dir", verified], "二选一"),             # 两种模式混用
        (["--base-dir", verified], "至少要配一个 --guided-from"),
        (["--verified-dir", verified, "--guided-from", verified], "不能和 --verified-dir 混用"),
    ]
    for extra, message in cases:
        result = control_cli("--graded", graded_file, "--output-dir", output, *extra)
        assert result.returncode == 2 and message in result.stderr, result.stderr
    assert not output.exists()
    # 配对对照模式的输入不全、或者引导组里没有基础集之外的题：说清楚是哪一个，不留下输出目录
    empty = tmp_path / "D-empty"
    empty.mkdir()
    result = control_cli("--graded", graded_file, "--base-dir", verified, "--guided-from", empty,
                         "--output-dir", output)
    assert result.returncode == 2 and f"找不到 {empty / 'train.jsonl'}" in result.stderr
    result = control_cli("--graded", graded_file, "--base-dir", verified, "--guided-from", verified,
                         "--output-dir", output)
    assert result.returncode == 2 and "没有基础集之外的题" in result.stderr
    assert not output.exists()


def test_matched_control_leaves_out_unfinished_samples_even_when_their_code_parses(tmp_path):
    # 写到一半被截断：代码能解析，但从没判过分，而且差不多用满了采样的长度上限
    truncated = {**graded("h3v_t_001", 0, False, "value = 1\n" + "# 解释" * 1400 + "\n"),
                 "finish_reason": "length", "usage": {"input_tokens": 300, "output_tokens": 2048},
                 "provenance": {"strict_pass": False, "verdict": "incomplete_response"}}
    timed_out = {**graded("h3v_t_001", 1, False, "value = 2\n"), "finish_reason": "stop",
                 "usage": {"input_tokens": 300, "output_tokens": 20}}     # 写完了、判分超时：算没通过的答案
    base = [graded("v2_a", 0, True), graded("v2_b", 1, True)]
    path = tmp_path / "train.jsonl"

    def loaded(records):
        path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
        return SFTDataset(path, TemplateTokenizer(), max_length=2000, strict_pass_only=False)

    # 前提：这一份是格式完整的代码，但真实的训练入口会把它记成"超长"丢掉
    assert rft_control.well_formed(truncated) and truncated["provenance"]["verdict"] == "incomplete_response"
    assert loaded([*base, truncated]).stats.skipped_long == 1
    rows = [truncated, timed_out, graded("h3v_t_001", 2, True)]
    for seed in range(40):
        samples, stats = rft_control.build_matched_control(rows, ["h3v_t_001"], seed)
        assert [sample["candidate_idx"] for sample in samples] == [1] and stats["unmatched"] == []
        data = loaded([*base, *samples])          # 对照组和基础集的记录一条不少：两组仍然逐题对齐
        assert len(data) == 3 and data.stats.skipped_long == 0
    samples, stats = rft_control.build_matched_control(rows, ["h3v_t_001"], seed=3)
    assert stats["chosen"] == [{"task_id": "h3v_t_001", "sample_id": "rft-h3v_t_001-001", "verdict": "partial",
                                "finish_reason": "stop", "input_tokens": 300, "output_tokens": 20}]
    # 一道题只有没写完的采样：记成配不上，不抛异常
    samples, stats = rft_control.build_matched_control([truncated, rows[2]], ["h3v_t_001"], seed=1)
    assert samples == [] and stats["unmatched"] == ["h3v_t_001"] and stats["chosen"] == []
    # 没有 finish_reason 字段的记录按写完了算
    (only,), _ = rft_control.build_matched_control([graded("h3v_t_001", 5, False)], ["h3v_t_001"], seed=1)
    assert only["candidate_idx"] == 5
    # 写完了但很长的照样能抽到（这里没有分词器，判断不了长度）：manifest 记下 token 数，留给"加载"条数去核对
    rambled = {**graded("h3v_t_001", 3, False, "value = " + "1 + " * 900 + "1\n"), "finish_reason": "stop",
               "usage": {"input_tokens": 1500, "output_tokens": 900}}
    seen = set()
    for seed in range(40):
        (only,), stats = rft_control.build_matched_control([timed_out, rambled], ["h3v_t_001"], seed)
        seen.add((only["candidate_idx"], stats["chosen"][0]["input_tokens"] + stats["chosen"][0]["output_tokens"]))
    assert seen == {(1, 320), (3, 2400)}


def test_matched_control_cli_counts_a_guided_directory_once_however_it_is_spelled(tmp_path):
    dataset(tmp_path / "verified", [graded("v2_a", 0, True)], [])         # 命令行里用的是相对路径 verified
    guided = dataset(tmp_path / "D", [graded("v2_a", 0, True), graded("h3v_a_001", 0, True)], [])
    (tmp_path / "train-graded.jsonl").write_text(json.dumps(graded("h3v_a_001", 1, False, "value = 2\n")) + "\n",
                                                 encoding="utf-8")
    common = ["--graded", "train-graded.jsonl", "--base-dir", "verified", "--seed", "7"]
    # 同一个目录：绝对路径、相对路径、带 ./ 的相对路径，再重复一次
    spelled = control_cli(*common, "--guided-from", guided, "D", "--guided-from", "./D", "D",
                          "--output-dir", "F-spelled", cwd=tmp_path)
    plain = control_cli(*common, "--guided-from", "D", "--output-dir", "F-plain", cwd=tmp_path)
    assert spelled.returncode == 0 and plain.returncode == 0, spelled.stderr + plain.stderr
    assert "结果：配对对照 = 基础 1 条 + 引导题 1 条（来自 1 组）；配不上的题 0 道 []" in spelled.stdout
    for name in ("train.jsonl", "dev.jsonl", "unmatched-tasks.txt"):
        assert (tmp_path / "F-spelled" / name).read_bytes() == (tmp_path / "F-plain" / name).read_bytes(), name
    manifests = [json.loads((tmp_path / name / "dataset-manifest.json").read_text(encoding="utf-8"))
                 for name in ("F-spelled", "F-plain")]
    assert [manifest["guided_from"] for manifest in manifests] == [[str(guided)], ["D"]]   # 只记一次：第一次的写法
    assert {key: value for key, value in manifests[0].items() if key != "guided_from"} == {
        key: value for key, value in manifests[1].items() if key != "guided_from"}


def test_matched_control_cli_refuses_to_write_a_control_that_pairs_no_task(tmp_path):
    base = dataset(tmp_path / "verified", [graded("v2_a", 0, True)], [])
    guided = dataset(tmp_path / "D-prefix", [graded("v2_a", 0, True), graded("h3v_a_001", 0, True)], [])
    output = tmp_path / "F-control"
    absent = "1 道里 1 道在 --graded 里一条记录都没有"
    present = "1 道里 0 道在 --graded 里一条记录都没有"
    check_graded = "先核对 --graded 是不是这次运行里 train 的 graded 文件"
    nothing_usable = "这些题的采样里没有一份写完了、格式完整、又没判分通过的答案"
    cases = [
        ("empty.jsonl", "", absent, check_graded),                            # --graded 给错了文件
        ("dev-graded.jsonl", json.dumps({**graded("v2_dev", 0, False), "split": "dev"}) + "\n",
         absent, check_graded),
        ("truncated.jsonl", json.dumps({**graded("h3v_a_001", 0, False), "finish_reason": "length"}) + "\n",
         present, nothing_usable),                                           # 文件是对的，但没一份能用的
        ("passing.jsonl", json.dumps(graded("h3v_a_001", 0, True)) + "\n", present, nothing_usable),
    ]
    for name, content, count, hint in cases:
        (tmp_path / name).write_text(content, encoding="utf-8")
        result = control_cli("--graded", tmp_path / name, "--base-dir", base, "--guided-from", guided,
                             "--output-dir", output)
        assert result.returncode == 2, (name, result.stdout)
        assert "引导题一道也配不上" in result.stderr and count in result.stderr and hint in result.stderr, result.stderr
    assert not output.exists()            # 不留一份和基础集一模一样的"配对对照"


def rounding_task(index, split="train"):
    """一道独立的小题：rounded 没有四舍五入，公开测试过、隐藏测试挂。"""
    return {
        "task_id": f"paired_{index}", "source_group": f"repository_{index}",
        "split": split, "bug_family": "rounding", "difficulty": "L1",
        "instruction": f'Return the rounded value with offset "{index}".',
        "module": "rounding.py", "buggy": f"def rounded(value):\n    return int(value) + {index}\n",
        "fixed": f"def rounded(value):\n    return round(value) + {index}\n",
        "public_test": f"from rounding import rounded\n\ndef test_rounding():\n"
                       f"    assert rounded(1.9) == {index + 2}\n",
        "private_tests": {"test_negative.py": "from rounding import rounded\n\n"
                          f"def test_negative():\n    assert rounded(-1.9) == {index - 2}\n"},
    }


def passing_sample(spec):
    """教师采集那种格式的一条记录：参考解，判分通过。"""
    return {"task_id": spec["task_id"], "source_group": spec["source_group"], "split": spec["split"],
            "task_hash": spec_fingerprint(spec), "messages": build_messages(spec),
            "target": {"role": "assistant", "content": spec["fixed"]}, "provenance": {"strict_pass": True}}


def test_drop_tasks_file_reads_lines_or_a_json_array_and_drops_whole_tasks(tmp_path, capsys):
    lines_file = tmp_path / "unmatched-tasks.txt"
    lines_file.write_text("paired_1\n\n  paired_2  \n\n", encoding="utf-8")
    assert prepare_data.load_task_ids(lines_file) == {"paired_1", "paired_2"}
    array = tmp_path / "drop.json"
    array.write_text('["paired_1", "paired_2"]', encoding="utf-8")
    assert prepare_data.load_task_ids(array) == {"paired_1", "paired_2"}
    lines_file.write_text("", encoding="utf-8")           # rft_control.py 没有配不上的题时写的是空文件
    assert prepare_data.load_task_ids(lines_file) == set()
    for bad in ("[1, 2]", '{"task_id": "paired_1"}', '["paired_1", '):
        array.write_text(bad, encoding="utf-8")
        with pytest.raises(ValueError, match="--drop-tasks-file"):
            prepare_data.load_task_ids(array)
    records = [{"task_id": "paired_1", "n": 0}, {"task_id": "paired_0", "n": 1},
               {"task_id": "paired_1", "n": 2}, {"task_id": "paired_2", "n": 3}]
    array.write_text('["paired_1", "paired_9"]', encoding="utf-8")   # 没有记录的题号不要紧
    assert prepare_data.drop_records(records, array) == (
        [{"task_id": "paired_0", "n": 1}, {"task_id": "paired_2", "n": 3}], {"paired_1"})
    assert capsys.readouterr().out == ""                   # 纯函数：那一行由 main 打印


def test_prepare_data_main_keeps_the_dropped_tasks_out_of_the_dataset(tmp_path):
    specs = [rounding_task(index) for index in range(2)] + [rounding_task(2, "dev")]
    tasks_file = tmp_path / "tasks.jsonl"
    tasks_file.write_text("".join(json.dumps(spec, ensure_ascii=False) + "\n" for spec in specs),
                          encoding="utf-8")
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text("".join(json.dumps(passing_sample(spec)) + "\n" for spec in specs), encoding="utf-8")
    dropped = tmp_path / "unmatched-tasks.txt"
    dropped.write_text("paired_1\n\nnot_a_task\n", encoding="utf-8")     # 空行、没有记录的题号都不要紧

    def run(output, *extra):
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts/prepare_data.py"), "--tasks-file", str(tasks_file),
             "--input", str(candidates), "--output-dir", str(output), *extra],
            capture_output=True, text=True, timeout=300)

    output = tmp_path / "sft"
    result = run(output, "--drop-tasks-file", dropped)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "按 --drop-tasks-file 去掉 1 条记录（1 道题）" in result.stdout
    assert "重放验证候选: paired_0" in result.stdout and "重放验证候选: paired_1" not in result.stdout
    assert [item["task_id"] for item in lines(output / "train.jsonl")] == ["paired_0"]
    assert [item["task_id"] for item in lines(output / "dev.jsonl")] == ["paired_2"]
    manifest = json.loads((output / "dataset-manifest.json").read_text(encoding="utf-8"))
    assert manifest["splits"]["train"]["missing_task_ids"] == ["paired_1"]
    assert manifest["dropped_task_ids"] == ["paired_1"]

    # JSON 数组也认；不给这个参数时 manifest 里没有 dropped_task_ids
    array = tmp_path / "drop.json"
    array.write_text('["paired_1"]', encoding="utf-8")
    other = tmp_path / "sft-array"
    result = run(other, "--drop-tasks-file", array)
    assert result.returncode == 0 and "按 --drop-tasks-file 去掉 1 条记录（1 道题）" in result.stdout, result.stderr
    assert (other / "train.jsonl").read_bytes() == (output / "train.jsonl").read_bytes()
    # 不给这个参数时一切照旧：没有那一行，manifest 里也没有 dropped_task_ids
    plain = tmp_path / "sft-plain"
    result = run(plain)
    assert result.returncode == 0 and "按 --drop-tasks-file" not in result.stdout, result.stderr
    assert [item["task_id"] for item in lines(plain / "train.jsonl")] == ["paired_0", "paired_1"]
    assert "dropped_task_ids" not in json.loads((plain / "dataset-manifest.json").read_text(encoding="utf-8"))

    # 把 dev 的题全去掉就没有 dev 集了：报错，不留下写了一半的目录
    array.write_text('["paired_1", "paired_2"]', encoding="utf-8")
    broken = tmp_path / "sft-broken"
    result = run(broken, "--drop-tasks-file", array)
    assert result.returncode != 0 and "no accepted samples" in result.stderr and not broken.exists()
    # 划分模式下没有记录可去，给了也不悄悄忽略
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/prepare_data.py"), "--tasks-file", str(tasks_file),
         "--split-tasks", "--drop-tasks-file", str(dropped), "--output-dir", str(tmp_path / "split")],
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 2 and "--drop-tasks-file 只用于 --input" in result.stderr


def clamp_task(task_id, split="train"):
    """一道独立的小题：clamp 没做截断，公开测试过、隐藏测试挂。模块名跟着题号走，假服务才分得清是哪道题。"""
    module = f"{task_id}.py"
    return {
        "task_id": task_id, "source_group": f"group-{task_id}", "split": split,
        "bug_family": "clamping", "difficulty": "L1",
        "instruction": "实现 clamp(values, low, high)：把每个数限制在 [low, high] 内，返回新列表。",
        "module": module,
        "buggy": "def clamp(values, low, high):\n    return list(values)\n",
        "fixed": "def clamp(values, low, high):\n"
                 "    return [min(max(v, low), high) for v in values]\n",
        "public_test": f"from {module[:-3]} import clamp\n\n"
                       "def test_public():\n    assert clamp([1, 5], 0, 10) == [1, 5]\n",
        "private_tests": {"test_hidden.py": f"from {module[:-3]} import clamp\n\n"
                          "def test_hidden():\n    assert clamp([-3, 3, 13], 0, 10) == [0, 3, 10]\n"},
    }


def hinted_answers(specs):
    """假模型：h3_base 和 dev 题不引导时偶数号答对；h3_zero 不引导时总是照抄有缺陷的代码（是代码，但判分
    不过），h3v_trunc 不引导时总是写到一半被截断（代码能解析，但没判过分）。后两道题只要带着参考解的
    开头就答对。"""
    by_module = {spec["module"]: spec for spec in specs}

    def answer(payload):
        user = payload["messages"][-1]["content"]
        spec = next((item for module, item in by_module.items() if f"文件 {module}：" in user), None)
        if spec is None:                       # 探测参数用的请求（只生成 1 个 token）
            return 200, {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}
        if "到第" in user:                     # 带着参考解开头的提示
            code, finish = spec["fixed"], "stop"
        elif spec["task_id"] == "h3v_trunc_001":
            code = spec["buggy"] + "# 这样改是因为原来的实现没有"      # 解释写到一半被截断
            finish = "length"
        elif spec["task_id"] == "h3_zero_001" or payload["seed"] % 2:
            code, finish = spec["buggy"], "stop"
        else:
            code, finish = spec["fixed"], "stop"
        return 200, {"choices": [{"message": {"content": code}, "finish_reason": finish}],
                     "usage": {"prompt_tokens": 9, "completion_tokens": 9}}

    return answer


def test_guided_sampling_end_to_end_pairs_both_arms_task_by_task(tmp_path, dead_proxy):
    """真的采样、判分、准备数据：引导组多出来的题在对照组里各有一份没通过的答案，配不上的题两组都不留。"""
    specs = [clamp_task(name) for name in ("h3_base_001", "h3_zero_001", "h3v_trunc_001")]
    specs.append(clamp_task("h3_dev_001", "dev"))
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("".join(json.dumps(spec, ensure_ascii=False) + "\n" for spec in specs), encoding="utf-8")
    sampling = tmp_path / "sampling"
    fake = FakeVLLM(hinted_answers(specs))

    def sample(split, *extra):
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts/sample_rft.py"), "--tasks-file", str(tasks), "--split", split,
             "--samples", "2", "--hint-fractions", "0.5", "--vllm-url", fake.url, "--output-dir", str(sampling),
             "--concurrency", "4", "--grade-workers", "2", "--model", "org/model", *extra],
            capture_output=True, text=True, timeout=600)

    try:
        unguided = sample("train")
        dev = sample("dev")
        guided = sample("train", "--hint-kind", "prefix",
                        "--only-zero-pass", str(sampling / "train-manifest.json"), "--tag", "guided-prefix")
    finally:
        fake.close()
    for step in (unguided, dev, guided):
        assert step.returncode == 0, step.stderr[-2000:]
    sft = tmp_path / "sft"
    base_inputs = [sampling / "train-candidates.jsonl", sampling / "dev-candidates.jsonl"]
    guided_inputs = [sampling / "train-candidates.jsonl", sampling / "train-guided-prefix-candidates.jsonl",
                     sampling / "dev-candidates.jsonl"]

    def prepare(name, inputs, *extra):
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts/prepare_data.py"), "--tasks-file", str(tasks),
             "--input", *map(str, inputs), "--output-dir", str(sft / name), *extra],
            capture_output=True, text=True, timeout=600)

    # 不引导时一次都没通过的是两道题（--only-zero-pass 读的就是这个清单）
    manifest = json.loads((sampling / "train-manifest.json").read_text(encoding="utf-8"))
    assert manifest["zero_pass_tasks"] == ["h3_zero_001", "h3v_trunc_001"]
    # 基础集是上一轮的验证组；引导组 = 基础集 + 带着提示才解出来的两道题
    verified = prepare("verified", base_inputs)
    prefix = prepare("D-prefix", guided_inputs)
    assert verified.returncode == 0 and prefix.returncode == 0, verified.stderr[-2000:] + prefix.stderr[-2000:]
    assert [item["task_id"] for item in lines(sft / "verified/train.jsonl")] == ["h3_base_001"]
    assert [item["task_id"] for item in lines(sft / "D-prefix/train.jsonl")] == [
        "h3_base_001", "h3_zero_001", "h3v_trunc_001"]

    # 配对对照：h3_zero 有写完了、没通过的采样，配上；h3v_trunc 不引导时没写完过，配不上
    result = control_cli("--graded", sampling / "train-graded.jsonl", "--base-dir", sft / "verified",
                         "--guided-from", sft / "D-prefix", "--output-dir", sft / "F-control", "--seed", 42)
    assert result.returncode == 0, result.stderr
    assert ("结果：配对对照 = 基础 1 条 + 引导题 1 条（来自 1 组）；"
            "配不上的题 1 道 ['h3v_trunc_001']") in result.stdout
    truncated = [record for record in lines(sampling / "train-graded.jsonl")
                 if record["task_id"] == "h3v_trunc_001"]
    assert [(record["finish_reason"], record["provenance"]["verdict"]) for record in truncated] == [
        ("length", "incomplete_response")] * 2
    assert all(rft_control.well_formed(record) for record in truncated)   # 代码能解析也没用：没写完的不进池子
    assert (sft / "F-control/train.jsonl").read_bytes().startswith((sft / "verified/train.jsonl").read_bytes())
    assert (sft / "F-control/dev.jsonl").read_bytes() == (sft / "verified/dev.jsonl").read_bytes()
    assert (sft / "F-control/unmatched-tasks.txt").read_text(encoding="utf-8") == "h3v_trunc_001\n"
    # 真实的训练入口把对照组整份加载进来，一条不少
    paired = SFTDataset(sft / "F-control/train.jsonl", TemplateTokenizer(), max_length=4000, strict_pass_only=False)
    assert len(paired) == 2 and paired.stats.skipped_long == 0

    # 配不上的题从引导组里也去掉：两组在引导题上逐题对齐
    paired = prepare("D-prefix-paired", guided_inputs,
                     "--drop-tasks-file", sft / "F-control/unmatched-tasks.txt")
    assert paired.returncode == 0, paired.stderr[-2000:]
    assert "按 --drop-tasks-file 去掉 2 条记录（1 道题）" in paired.stdout
    assert [item["task_id"] for item in lines(sft / "D-prefix-paired/train.jsonl")] == [
        "h3_base_001", "h3_zero_001"]

    def guided_tasks(directory):
        base = {item["task_id"] for item in lines(sft / "verified/train.jsonl")}
        return [item["task_id"] for item in lines(directory / "train.jsonl") if item["task_id"] not in base]

    assert guided_tasks(sft / "D-prefix-paired") == guided_tasks(sft / "F-control") == ["h3_zero_001"]
    (control,) = [item for item in lines(sft / "F-control/train.jsonl") if item["task_id"] == "h3_zero_001"]
    (correct,) = [item for item in lines(sft / "D-prefix-paired/train.jsonl") if item["task_id"] == "h3_zero_001"]
    assert control["provenance"]["strict_pass"] is False and control["record_type"] == "attempt"
    assert control["selection"] == {"policy": "control_failing_random", "pool": 2}
    assert control["target"]["content"] == specs[1]["buggy"].strip() + "\n"
    assert correct["provenance"]["strict_pass"] is True and correct["provenance"]["guided"]["fraction"] == 0.5
    assert control["messages"] == correct["messages"] == build_messages(specs[1])    # 同一道题、同一个提示
    # manifest 记下这道题抽到的是哪一份，连同采样时数出的 token 数（训练前核对"加载"条数时用）
    (chosen,) = json.loads((sft / "F-control/dataset-manifest.json").read_text(encoding="utf-8"))["chosen"]
    assert (chosen["task_id"], chosen["finish_reason"], chosen["input_tokens"], chosen["output_tokens"]) == (
        "h3_zero_001", "stop", 9, 9)
    assert chosen["sample_id"] == control["sample_id"]


SEPARATORS = "\u2028\u0085\u2029"       # str.splitlines() 把它们当成换行，JSON 的字符串里却可以原样出现


def test_a_record_with_unicode_line_separators_stays_on_one_line_through_the_control(tmp_path):
    """采样记录里带着 U+2028 这类字符（模型回答、判分输出里偶尔会有）：rft_control.py 读 --graded 时不能把它
    当成换行，抽中的那份写进训练集后也还是一行一条记录、读回来还是原来的字符——按行切的脚本（训练入口、
    编排脚本里的片段）才切不开它；prepare_data.py 写出的是同一种写法。"""
    noisy = {"raw_response": f"说明：{SEPARATORS}。", "grader_details": {"stdout": f"a{SEPARATORS}b"}}
    base_records = [graded("v2_a", 0, True, "value = 1\n"), {**graded("v2_b", 1, True), **noisy}]
    base = tmp_path / "verified"
    base.mkdir()
    prepare_data.write_jsonl(base / "train.jsonl", base_records)        # 和真的一样：基础集是 prepare_data.py 写的
    (base / "dev.jsonl").write_text("", encoding="utf-8")
    prefix = dataset(tmp_path / "D-prefix", [*base_records, graded("h3v_a_001", 0, True),
                                             graded("h3v_d_001", 0, True)], [])
    graded_file = tmp_path / "train-graded.jsonl"
    # 旧版本的采样脚本把它们原样写出来（json.dumps(ensure_ascii=False)）：这样的文件现在也读得回来
    graded_file.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in [
        *MATCHED_GRADED, {**graded("h3v_d_001", 1, False, "value = 6\n"), **noisy}]), encoding="utf-8")
    read = [item for item in rft_control.read_jsonl(graded_file) if item["task_id"] == "h3v_d_001"]
    assert len(read) == 2 and read[1]["raw_response"] == noisy["raw_response"]   # 没有被切成几条坏行
    output = tmp_path / "F-control"
    result = control_cli("--graded", graded_file, "--base-dir", base, "--guided-from", prefix,
                         "--output-dir", output, "--seed", "42")
    assert result.returncode == 0, result.stderr
    assert "基础 2 条 + 引导题 2 条（来自 1 组）；配不上的题 0 道 []" in result.stdout    # h3v_d 这回配上了
    text = (output / "train.jsonl").read_text(encoding="utf-8")
    assert text.count("\n") == len(text.splitlines()) == 4             # 基础 2 条 + 引导题 2 条，怎么按行读都是四行
    assert not set(SEPARATORS) & set(text)                             # 写出来的是转义
    assert text.startswith((base / "train.jsonl").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in text.splitlines()]
    assert records[:2] == base_records                                 # 读回来还是原来的字符
    control = [item for item in records if item["task_id"] == "h3v_d_001"]
    assert len(control) == 1 and control[0]["target"]["content"] == "value = 6\n"
    assert control[0]["selection"] == {"policy": "control_failing_random", "pool": 1}
    assert control[0]["raw_response"] == noisy["raw_response"]
    prepared = tmp_path / "prepared.jsonl"
    prepare_data.write_jsonl(prepared, [*records, {"task_id": f"x{SEPARATORS}y"}])
    text = prepared.read_text(encoding="utf-8")
    assert text.count("\n") == len(text.splitlines()) == 5 and not set(SEPARATORS) & set(text)
    assert [json.loads(line).get("raw_response") for line in text.splitlines()[:4]] == [
        item.get("raw_response") for item in records]
    assert json.loads(text.splitlines()[4]) == {"task_id": f"x{SEPARATORS}y"}
