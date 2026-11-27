# -*- coding: utf-8 -*-
"""选型时复用已有的 dev 报告：基座不重复评，中断后续跑时不重评已经评完的 epoch。"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate import content_hash
import select_checkpoint
from test_checkpoint_selection import report, write_adapter
from agentlab.benchmark.builder import SPECS
from agentlab.benchmark.protocol import protocol_hash

GREEDY = {"repeats": 1, "seed": 42, "temperature": 0.0, "max_new_tokens": 2048,
          "quantization": "nf4", "decoding": "greedy", "backend": "transformers"}


def modern_report(adapter, tasks_hash, passes=(True, False)):
    """与 select_checkpoint.py 调 evaluate.py 时得到的一致：贪心、单次、本地模型、固定版本。"""
    item = report(adapter, passes)
    item.update(generation=dict(GREEDY), requested_revision="commit", tasks_hash=tasks_hash)
    item["protocol"] = {**item["protocol"], "generation": dict(GREEDY), "benchmark_hash": tasks_hash}
    item["protocol_hash"] = content_hash(item["protocol"])
    return item


def trained_checkpoint(tmp_path):
    """一个带 epoch 1 adapter 的 checkpoint 目录，加上它的 dev 任务文件。"""
    adapter, _ = write_adapter(tmp_path / "checkpoint")
    manifest_path = tmp_path / "checkpoint" / "training-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(model="qwen", resolved_revision="commit",
                    data={"task_ids": ["separate-training-task"]})
    manifest_path.write_text(json.dumps(manifest))
    specs = [{**SPECS[0], "split": "dev"}]
    (tmp_path / "tasks.jsonl").write_text(json.dumps(specs[0]) + "\n")
    return adapter, protocol_hash(specs)


def fake_evaluator(calls, tasks_hash):
    def evaluate(command, check):
        output = Path(command[command.index("--output") + 1])
        adapter = command[command.index("--adapter") + 1] if "--adapter" in command else None
        calls.append(adapter)
        output.write_text(json.dumps(modern_report(adapter, tasks_hash, (True, True) if adapter else (True, False))))
    return evaluate


def selection_arguments(tmp_path, *extra):
    return ["--checkpoint-root", str(tmp_path / "checkpoint"), "--tasks-file", str(tmp_path / "tasks.jsonl"),
            "--output", str(tmp_path / "selection"), *extra]


def test_selection_reuses_a_matching_base_report_instead_of_re_evaluating(tmp_path, monkeypatch):
    adapter, tasks_hash = trained_checkpoint(tmp_path)
    prior = modern_report(None, tasks_hash)
    (tmp_path / "prior-base.json").write_text(json.dumps(prior))
    calls = []
    monkeypatch.setattr(select_checkpoint.subprocess, "run", fake_evaluator(calls, tasks_hash))
    monkeypatch.setattr(sys, "argv", ["select_checkpoint.py", *selection_arguments(
        tmp_path, "--base-report", str(tmp_path / "prior-base.json"))])
    select_checkpoint.main()
    assert calls == [str(adapter)]                               # 基座没有重评
    assert json.loads((tmp_path / "selection/epoch-000-dev.json").read_text()) == prior
    selected = json.loads((tmp_path / "selection/selection.json").read_text())
    assert selected["selected_epoch"] == 1
    assert selected["configuration"]["base_report"] == str(tmp_path / "prior-base.json")


@pytest.mark.parametrize("change, reason", [
    (lambda item: {**item, "generation": {**GREEDY, "temperature": 1.0}}, "生成设置不同"),
    (lambda item: {**item, "adapter": "some/adapter"}, "带着 adapter"),
    (lambda item: {**item, "tasks_hash": "another-benchmark"}, "题目不同"),
    (lambda item: {**item, "requested_revision": "another-commit"}, "版本不同"),
    (lambda item: {**item, "contamination": {"status": "not_checked"}}, "重叠审计"),
    (lambda item: {**item, "split": "test"}, "不是 dev 报告"),
])
def test_a_base_report_that_does_not_match_the_selection_is_refused(tmp_path, change, reason):
    _, tasks_hash = trained_checkpoint(tmp_path)
    (tmp_path / "prior-base.json").write_text(json.dumps(change(modern_report(None, tasks_hash))))
    specs = [{**SPECS[0], "split": "dev"}]
    problems = select_checkpoint.problems_with_report(
        tmp_path / "prior-base.json", specs, "qwen", "commit", dict(GREEDY), None)
    assert any(reason in problem for problem in problems)


def test_an_unreadable_report_is_never_reused(tmp_path):
    (tmp_path / "half.json").write_text('{"results": [')          # 写到一半就断了
    for name in ("half.json", "missing.json"):
        problems = select_checkpoint.problems_with_report(
            tmp_path / name, [], "qwen", "commit", GREEDY, None)
        assert len(problems) == 1 and problems[0].startswith("读不出来")


def test_resume_keeps_finished_reports_and_redoes_damaged_ones(tmp_path, monkeypatch):
    adapter, tasks_hash = trained_checkpoint(tmp_path)
    selection = tmp_path / "selection"
    calls = []
    monkeypatch.setattr(select_checkpoint.subprocess, "run", fake_evaluator(calls, tasks_hash))
    selection.mkdir()
    (selection / "leftover.txt").write_text("x")
    monkeypatch.setattr(sys, "argv", ["select_checkpoint.py", *selection_arguments(tmp_path)])
    with pytest.raises(SystemExit) as refused:                    # 不声明续跑就不许写进非空目录
        select_checkpoint.main()
    assert refused.value.code == 2 and calls == []

    finished = json.dumps(modern_report(None, tasks_hash))
    (selection / "epoch-000-dev.json").write_text(finished)       # 上一次评完基座就中断了
    monkeypatch.setattr(sys, "argv", ["select_checkpoint.py", *selection_arguments(tmp_path, "--resume")])
    select_checkpoint.main()
    assert calls == [str(adapter)]                                # 基座那份原样保留
    assert (selection / "epoch-000-dev.json").read_text() == finished

    calls.clear()
    (selection / "epoch-000-dev.json").write_text('{"results": [')     # 写到一半就断了
    select_checkpoint.main()
    assert calls == [None]                                       # 坏的那份重评，epoch 那份不动
    assert len(json.loads((selection / "epoch-000-dev.json").read_text())["results"]) == 2
    assert json.loads((selection / "selection.json").read_text())["selected_epoch"] == 1


def test_expected_generation_follows_the_selection_flags():
    plain = SimpleNamespace(seed=42, max_new_tokens=2048, quantization="nf4", thinking=False)
    assert select_checkpoint.expected_generation(plain) == GREEDY
    thinking = SimpleNamespace(seed=42, max_new_tokens=8192, quantization="bf16", thinking=True)
    assert select_checkpoint.expected_generation(thinking) == {
        **GREEDY, "max_new_tokens": 8192, "quantization": "bf16", "thinking": True}
