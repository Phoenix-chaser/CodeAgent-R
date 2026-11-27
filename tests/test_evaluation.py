"""Regression coverage for repaired extraction and honest evaluation metrics."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from _shared_extract import extract_code
from compare import make_report, paired_bootstrap, validate_comparison, wilson_interval
from evaluate import (content_hash, evaluate_candidate, generation_protocol, make_parser,
                      summarize_results, training_overlap)
from agentlab.benchmark.builder import SPECS, build_all
from agentlab.benchmark.protocol import build_messages


@pytest.mark.parametrize("spec", SPECS, ids=lambda spec: spec["task_id"])
def test_reference_code_survives_extraction(spec):
    assert extract_code(spec["fixed"]) == spec["fixed"]


@pytest.mark.parametrize("code", [
    'STATES = ["PENDING", "DONE"]\n\ndef state():\n    return STATES[0]\n',
    'MESSAGE = "错误：请重试。"\n',
    'def message():\n    return "错误：请重试。"\n',
])
def test_extraction_preserves_constants_and_unicode(code):
    assert extract_code(code) == code
    assert extract_code("```python\n" + code + "```\n说明。") == code


def test_extraction_handles_inline_fence_and_empty_prose():
    assert extract_code("```python VALUE = 3\n```\n") == "VALUE = 3\n"
    assert extract_code("修复后的代码：\nVALUE = 3\n") == "VALUE = 3\n"
    assert extract_code("这个函数没有问题，不需要修复。") == ""


def test_extraction_does_not_treat_string_contents_as_markdown():
    code = 'FENCE = "```python"\n\ndef marker():\n    return FENCE\n'
    assert extract_code(code) == code


def test_repeats_report_mean_pass_at_one_not_any_pass():
    summary = summarize_results([
        dict(task_id="first", strict_pass=True),
        dict(task_id="first", strict_pass=False, failure_type="public_test_failure"),
        dict(task_id="second", strict_pass=False, failure_type="public_test_failure"),
        dict(task_id="second", strict_pass=False, failure_type="public_test_failure"),
    ])
    assert summary["pass_at_1"] == 0.25
    assert summary["any_pass_rate"] == 0.5
    assert summary["n_passed_attempts"] == 1
    assert summary["valid"]


def test_infrastructure_error_invalidates_aggregate():
    summary = summarize_results([dict(task_id="one", strict_pass=False,
                                     failure_type="infrastructure_error")])
    assert summary["infrastructure_errors"] == 1
    assert not summary["valid"]


def report_fixture(passes=(True, False)):
    generation = {"repeats": 1, "seed": 42}
    protocol = {"generation": generation, "split": "test", "benchmark_hash": "tasks",
                "prompt_hashes": {"one": "prompt-one", "two": "prompt-two"}}
    return dict(schema_version="codeagent-eval-v2", model="model", resolved_revision="sha",
                adapter=None, split="test", tasks_hash="tasks", protocol_hash=content_hash(protocol),
                protocol=protocol, task_ids=["one", "two"],
                task_hashes={"one": "one-hash", "two": "two-hash"},
                contamination={"status": "clean"}, generation=generation,
                results=[dict(task_id=task_id, repeat_index=0, strict_pass=passed, seed=42,
                              prompt_hash=f"prompt-{task_id}",
                              verdict="strict_pass" if passed else "strict_fail")
                         for task_id, passed in zip(("one", "two"), passes)])


@pytest.mark.parametrize("field,value", [("seed", 999), ("prompt_hash", "changed")])
def test_comparison_rejects_tampered_attempt_metadata(field, value):
    baseline, treatment = report_fixture(), report_fixture()
    treatment["results"][0][field] = value
    with pytest.raises(ValueError):
        validate_comparison(baseline, treatment)


def test_modern_reports_validate_and_empty_training_audit_is_rejected(tmp_path):
    validate_comparison(report_fixture(), report_fixture())
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="identities"):
        training_overlap([SPECS[0]], empty)


@pytest.mark.parametrize("key,value", [
    ("tasks_hash", "different"), ("protocol_hash", "different"),
    ("generation", {"repeats": 2}), ("model", "different"),
    ("resolved_revision", "different"),
])
def test_comparison_rejects_mismatched_protocol(key, value):
    baseline, treatment = report_fixture(), report_fixture()
    treatment[key] = value
    with pytest.raises(ValueError):
        validate_comparison(baseline, treatment)


def test_comparison_rejects_missing_tasks_and_repeat_budgets():
    baseline, treatment = report_fixture(), report_fixture()
    treatment["results"].pop()
    with pytest.raises(ValueError, match="Task sets differ"):
        validate_comparison(baseline, treatment)
    treatment = report_fixture()
    treatment["results"].append(dict(task_id="one", repeat_index=1, strict_pass=False))
    with pytest.raises(ValueError, match="Repeat budgets differ"):
        validate_comparison(baseline, treatment)


def test_legacy_report_is_explicitly_unverified_and_uses_task_means():
    baseline = dict(model="model", results=[
        dict(task_id="one", strict_pass=True), dict(task_id="one", strict_pass=False),
        dict(task_id="two", strict_pass=False), dict(task_id="two", strict_pass=False)])
    treatment = copy.deepcopy(baseline)
    report = make_report(baseline, treatment, bootstrap_samples=100)
    assert "unverified" in report
    assert "| 平均 pass@1 | 25.00% | 25.00% |" in report
    assert "+0.00 个百分点" in report
    assert "两组均未通过" not in report
    assert "最终选择（Base）" not in report


def test_base_fallback_report_identifies_no_adapter_comparison():
    baseline = report_fixture()
    report = make_report(baseline, copy.deepcopy(baseline), bootstrap_samples=100)
    assert "# 评估对比：基线 vs 最终选择（Base）" in report
    assert "Adapter: 未采用，保留 Base" in report
    assert "| 指标 | 基线 | 最终选择（Base） |" in report
    assert "本表对比 Base 与 Base" in report
    assert "不能由本表推断各 epoch 的生成输出相同" in report
    assert "这不证明总体能力等效" in report


def test_adapter_comparison_keeps_training_label_and_explains_zero_interval():
    baseline, treatment = report_fixture(), report_fixture()
    treatment["adapter"] = "checkpoints/epoch-001/adapter"
    baseline["results"][0]["generated_code"] = "return 1"
    treatment["results"][0]["generated_code"] = "return 2 - 1"
    report = make_report(baseline, treatment, bootstrap_samples=100)
    assert "# 评估对比：基线 vs 训练后" in report
    assert "Adapter: checkpoints/epoch-001/adapter" in report
    assert "| 指标 | 基线 | 训练后 |" in report
    assert "本表对比 Base 与 Base" not in report
    assert "bootstrap 区间退化为 [0, 0]" in report
    assert "也不表示生成代码相同" in report


def test_paired_intervals_are_reproducible_and_respect_pairing():
    assert paired_bootstrap([0, 1], [0, 1], samples=100) == (0, 0)
    assert paired_bootstrap([0, 0], [1, 1], samples=100) == (1, 1)
    assert paired_bootstrap([0, 1, 0], [1, 1, 0], samples=100, seed=5) == \
        paired_bootstrap([0, 1, 0], [1, 1, 0], samples=100, seed=5)
    lower, upper = wilson_interval(7, 14)
    assert lower < 0.5 < upper
    assert upper - lower > 0.4


def test_paired_bootstrap_resamples_whole_source_groups():
    # 四个来源组、每组两题，组内两题同涨同跌
    base = [0, 0, 1, 1, 0, 0, 1, 1]
    treat = [0.5] * 8
    groups = ["g1", "g1", "g2", "g2", "g3", "g3", "g4", "g4"]
    clustered = paired_bootstrap(base, treat, samples=4000, seed=42, groups=groups)
    by_task = paired_bootstrap(base, treat, samples=4000, seed=42)
    assert clustered == (-0.5, 0.5)        # 整簇抽取：组内相关性被保留
    assert by_task == (-0.375, 0.375)      # 按题重采样把同组两题当成独立观测，区间偏窄
    with pytest.raises(ValueError):
        paired_bootstrap(base, treat, groups=groups[:-1])


def test_report_bootstraps_by_source_group_when_recorded():
    baseline, treatment = report_fixture(), report_fixture()
    treatment["adapter"] = "checkpoints/epoch-001/adapter"
    treatment["results"][1]["strict_pass"] = True
    for report in (baseline, treatment):
        for record in report["results"]:
            record["source_group"] = "same-group"
    report = make_report(baseline, treatment, bootstrap_samples=400)
    assert "按来源组聚类重采样的配对 bootstrap 95% 区间: [+50.00, +50.00] 个百分点" in report
    assert "1 个来源组）" in report                 # 同组的两题总是一起被抽中
    assert "bootstrap 以来源组为抽样单位" in report
    historical = make_report(baseline, treatment, bootstrap_samples=400, cluster_by="task")
    assert "按任务配对 bootstrap 95% 区间: [+0.00, +100.00] 个百分点" in historical


def test_report_falls_back_to_tasks_when_source_groups_are_missing():
    baseline, treatment = report_fixture(), report_fixture()
    treatment["adapter"] = "checkpoints/epoch-001/adapter"
    for record in baseline["results"]:
        record["source_group"] = "g"
    report = make_report(baseline, treatment, bootstrap_samples=100)
    assert "缺少一致的 source_group，按任务重采样" in report
    assert "按任务配对 bootstrap 95% 区间" in report
    with pytest.raises(ValueError):
        make_report(baseline, treatment, cluster_by="repository")


def test_training_overlap_detects_source_groups_and_manifest(tmp_path):
    spec = SPECS[0]
    training = tmp_path / "train.jsonl"
    training.write_text(json.dumps(dict(task_id="renamed-task",
                                       source_group=spec["source_group"],
                                       messages=build_messages(spec))), encoding="utf-8")
    overlap = training_overlap([spec], training)
    assert overlap["status"] == "contaminated"
    assert overlap["source_groups"] == [spec["source_group"]]
    assert overlap["prompt_hashes"]
    manifest = tmp_path / "training-manifest.json"
    manifest.write_text(json.dumps({"data": {"task_ids": [spec["task_id"]],
                                             "source_groups": [], "prompt_hashes": []}}),
                        encoding="utf-8")
    assert training_overlap([spec], manifest)["task_ids"] == [spec["task_id"]]
    assert training_overlap([spec], None)["status"] == "not_checked"


def test_final_test_rejects_checkpoint_selection_data(tmp_path):
    spec = SPECS[0]
    manifest = tmp_path / "training-manifest.json"
    manifest.write_text(json.dumps({
        "data": {"task_ids": ["independent"], "source_groups": ["independent"]},
        "eval_data": {"task_ids": [spec["task_id"]], "source_groups": [spec["source_group"]]},
    }), encoding="utf-8")
    assert training_overlap([spec], manifest)["status"] == "clean"
    assert training_overlap([spec], manifest, include_selection=True)["status"] == "contaminated"


def test_state_machine_reference_survives_full_evaluation(tmp_path):
    spec = next(spec for spec in SPECS if spec["task_id"] == "hard_state_machine_001")
    task_dir, grade_dir = tmp_path / "tasks", tmp_path / "grade"
    grade_dir.mkdir()
    build_all(task_dir, specs=[spec])
    record = evaluate_candidate(spec, spec["fixed"], task_dir, grade_dir)
    assert record["strict_pass"], record
    assert record["generated_code"] == spec["fixed"]
    assert record["raw_response"] == spec["fixed"]
    assert record["grader_details"]["layers"]["functional"]["private_pass"]


def test_evaluation_retains_long_failed_response(tmp_path, monkeypatch):
    response = "VALUE = 1\n" * 100
    monkeypatch.setattr("evaluate.grade_one", lambda *args: (
        "strict_fail", False, {"layers": {"functional": {"public_pass": False}}}))
    record = evaluate_candidate(SPECS[0], response, tmp_path, tmp_path)
    assert len(record["generated_code"]) > 200
    assert record["raw_response"] == response
    assert record["failure_type"] == "public_test_failure"


def test_thinking_mode_grades_only_the_text_after_the_think_block(tmp_path):
    """思考模式下回答在 </think> 之后；推理随完整回答留作证据，但不进判分器。"""
    spec = SPECS[0]
    task_dir, grade_dir = tmp_path / "tasks", tmp_path / "grade"
    grade_dir.mkdir()
    build_all(task_dir, specs=[spec])
    reasoning = "The loop stops one element early.\n"
    raw = reasoning + "</think>\n\n" + spec["fixed"]
    record = evaluate_candidate(spec, raw, task_dir, grade_dir, thinking=True)
    assert record["strict_pass"], record
    assert record["generated_code"] == spec["fixed"]
    assert record["raw_response"] == raw
    assert record["reasoning_chars"] == len(reasoning.strip())
    assert evaluate_candidate(spec, spec["fixed"], task_dir, grade_dir)["reasoning_chars"] == 0


def test_unfinished_thinking_is_reported_without_reaching_the_grader(tmp_path, monkeypatch):
    """额度用完时推理还没结束：记为“推理未结束”，不算代码错误，也不调判分器。"""
    monkeypatch.setattr("evaluate.grade_one",
                        lambda *args: pytest.fail("an unfinished answer has nothing to grade"))
    record = evaluate_candidate(SPECS[0], "still reasoning about the loop", tmp_path, tmp_path,
                               thinking=True)
    assert record["verdict"] == "incomplete_thinking" and record["strict_pass"] is False
    assert record["failure_type"] == "thinking_unfinished"
    assert record["raw_response"] == "still reasoning about the loop"
    metrics = summarize_results([record])
    assert metrics["n_unfinished_thinking"] == 1
    assert metrics["failure_counts"] == {"thinking_unfinished": 1}


def test_thinking_flag_is_opt_in_and_rejects_the_sampling_strategies(tmp_path):
    parser = make_parser()
    base = ["--model", "Qwen/Qwen3.5-9B", "--output", "out.json"]
    api = ["--api", "primary", "--output", "out.json"]      # 不带 --model，否则是"两者互斥"在报错
    assert parser.parse_args(base).thinking is False            # 默认关闭，已有协议不变
    assert parser.parse_args([*base, "--thinking"]).thinking is True
    assert parser.parse_args([*base, "--thinking", "--no-thinking"]).thinking is False
    for arguments in ([*base, "--thinking", "--strategy", "best-of-n"], [*api, "--thinking"]):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/evaluate.py"), *arguments],
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 2 and "--thinking" in result.stderr, result.stdout + result.stderr


def test_default_protocol_is_unchanged_and_sampling_settings_are_recorded():
    """top_p / top_k 取默认值时不进协议，旧报告的协议哈希不变；采样用了它们才记下来。"""
    parser = make_parser()
    base = ["--model", "Qwen/Qwen3.5-9B", "--output", "out.json"]
    api = ["--api", "primary", "--output", "out.json"]      # 不带 --model，否则是"两者互斥"在报错
    legacy = dict(repeats=1, seed=42, temperature=0.0, max_new_tokens=2048, quantization="nf4",
                  decoding="greedy", backend="transformers")
    assert generation_protocol(parser.parse_args(base)) == legacy
    assert generation_protocol(parser.parse_args([*base, "--temperature", "1"])) == {
        **legacy, "temperature": 1.0, "decoding": "sampling"}          # 以往的采样评测也不变
    # Qwen3.5 给思考模式的推荐采样：temperature 0.6、top_p 0.95、top_k 20
    official = parser.parse_args([*base, "--thinking", "--temperature", "0.6",
                                  "--top-p", "0.95", "--top-k", "20"])
    assert generation_protocol(official) == {**legacy, "temperature": 0.6, "decoding": "sampling",
                                             "thinking": True, "top_p": 0.95, "top_k": 20}
    for arguments, reason in (
            ([*base, "--top-p", "0.9"], "temperature"),                 # 贪心解码时采样参数没有意义
            ([*base, "--temperature", "0.6", "--top-k", "-1"], "top-k"),
            ([*base, "--temperature", "0.6", "--top-p", "0"], "top-p"),
            ([*api, "--temperature", "0.6", "--top-p", "0.9"], "local model")):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/evaluate.py"), *arguments],
                                capture_output=True, text=True, timeout=60)
        assert result.returncode == 2 and reason in result.stderr, result.stdout + result.stderr
