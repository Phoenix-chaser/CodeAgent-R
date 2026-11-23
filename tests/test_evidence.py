# -*- coding: utf-8 -*-
"""证据包：导出时不带走不该公开的内容，结论能从逐题结果重算，篡改会被发现。

用合成的实验产物走完整条链路（导出 → 脱敏 → 复核），不依赖真实实验目录。
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import export_evidence  # noqa: E402
import verify_evidence  # noqa: E402
from compare import make_report  # noqa: E402
from evaluate import content_hash  # noqa: E402
from select_checkpoint import choose_checkpoint  # noqa: E402

DEV = ["dev_a1", "dev_a2", "dev_b1", "dev_b2", "dev_c1", "dev_c2"]
TEST = ["test_a1", "test_a2", "test_b1", "test_b2", "test_c1", "test_c2"]
ADAPTERS = {0: None, 1: "/home/u/codeagent/checkpoints/sft-run/epoch-001/adapter",
            2: "/home/u/codeagent/checkpoints/sft-run/epoch-002/adapter"}
MODEL = dict(model="Qwen/Qwen2.5-Coder-7B-Instruct",
             resolved_revision="c03e6d358207e414f1eca0bb1891e29f1db0e242")
# 运行时拼出一段会触发“硬件型号”规则的文本；本文件源码里不能出现它本身
HARDWARE_MODEL = "GTX" + " 1234"


def report(split, tasks, passing, adapter=None):
    generation = {"repeats": 1, "seed": 42}
    protocol = {"generation": generation, "split": split, "benchmark_hash": f"{split}-tasks",
                "prompt_hashes": {task: f"prompt-{task}" for task in tasks}}
    rate = len(passing) / len(tasks)
    results = []
    for task in tasks:
        ok = task in passing
        layer = {"passed": ok, "exit_status": "ok", "expected_tests": 2,
                 "executed_tests": 2 if ok else 1, "summary": {"passed": 2 if ok else 1},
                 "missing_tests": [] if ok else ["tests_private.test_hidden::test_secret_case"],
                 "public_tail": "FAILED tests_private/test_hidden.py::test_secret_case"}
        results.append(dict(
            task_id=task, source_group=task.split("_")[0], split=split, difficulty="L1",
            prompt_hash=f"prompt-{task}", repeat_index=0, seed=42,
            raw_response="def solution(): return PRIVATE_ANSWER", generated_code="x",
            verdict="strict_pass" if ok else "strict_fail", strict_pass=ok,
            failure_type=None if ok else "public_test_failure",
            grader_details={"layers": {
                "legality": {"valid": True},
                "execution": {"isolation": "landlock", "expected_tests": {"tests_public": 2}},
                "functional": {"public_pass": ok, "private_pass": ok,
                               "public": layer, "private": layer, "public_tail": "secret"}}}))
    return dict(schema_version="codeagent-eval-v2", **MODEL, adapter=adapter,
                tasks_file="/home/u/codeagent/data/repair-104/frozen/tasks.jsonl",
                split=split, task_ids=list(tasks), task_hashes={t: f"{t}-hash" for t in tasks},
                tasks_hash=f"{split}-tasks", protocol=protocol,
                protocol_hash=content_hash(protocol),
                contamination={"status": "clean", "source": "/home/u/train.jsonl"},
                generation=generation, repeats=1,
                metrics={"n_tasks": len(tasks), "n_attempts": len(tasks),
                         "n_passed_attempts": len(passing), "pass_at_1": rate,
                         "any_pass_rate": rate, "n_tasks_any_pass": len(passing),
                         "infrastructure_errors": 0, "valid": True},
                n_tasks=len(tasks), success_rate=rate, results=results)


def write(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def artifacts(tmp_path):
    results = tmp_path / "results"
    passing = {0: {"dev_a1"}, 1: {"dev_a1", "dev_b1", "dev_c1"}, 2: {"dev_a1", "dev_b1"}}
    dev_reports = [report("dev", DEV, passing[epoch], ADAPTERS[epoch]) for epoch in (0, 1, 2)]
    for epoch, data in zip((0, 1, 2), dev_reports):
        write(results / f"selection/epoch-{epoch:03d}-dev.json", data)
    selection = choose_checkpoint(
        [{"epoch": epoch, "adapter": ADAPTERS[epoch]} for epoch in (0, 1, 2)], dev_reports)
    write(results / "selection/selection.json", selection)
    base = report("test", TEST, {"test_a1", "test_b1", "test_c1"})
    selected = report("test", TEST, {"test_a1", "test_a2", "test_b1", "test_c1"}, ADAPTERS[1])
    write(results / "test/base.json", base)
    write(results / "test/selected.json", selected)
    (results / "test/comparison.md").write_text(make_report(base, selected, 1000, 42),
                                                encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    write(checkpoint / "training-manifest.json", {
        "status": "complete", **MODEL,
        "optimizer_steps": 78, "peak_vram_gb": 10.3,
        "data": {"path": "/home/u/codeagent/data/sft/train.jsonl", "loaded": 202},
        "environment": {"python": "3.11", "gpu": HARDWARE_MODEL,
                        "packages": {"torch": "2.5.1"}},
        "history": [{"epoch": 1, "train_loss": 0.4, "dev_loss": 0.3}]})
    (checkpoint / "training-log.jsonl").write_text(
        '{"step": 1, "loss": 0.5}\n{"step": 2, "loss": 0.4}\n', encoding="utf-8")
    candidates = write(tmp_path / "train-candidates.manifest.json", {
        "protocol": {"model": "teacher-model-x", "base_url": "https://api.example-teacher.com/v1",
                     "max_tokens": 4096}})
    return SimpleNamespace(results=results, checkpoint=checkpoint, candidates=candidates,
                           base=base, selected=selected)


def export_bundle(artifacts, tmp_path, label="seed42"):
    files = export_evidence.collect(artifacts.results, artifacts.checkpoint, artifacts.candidates)
    assert export_evidence.export(label, files, tmp_path / "evidence" / label) == []
    return tmp_path / "evidence" / label


def bundle_text(bundle: Path) -> str:
    return "\n".join(path.read_text(encoding="utf-8")
                     for path in sorted(bundle.rglob("*")) if path.is_file())


def restamp(bundle: Path, name: str) -> None:
    """把被改文件的哈希同步进 index.json，模拟一份“自洽”的伪造证据。"""
    index_path = bundle / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["files"][name]["sha256"] = hashlib.sha256((bundle / name).read_bytes()).hexdigest()
    index_path.write_text(json.dumps(index), encoding="utf-8")


def test_exported_bundle_recomputes_to_the_same_conclusions(artifacts, tmp_path):
    bundle = export_bundle(artifacts, tmp_path)
    summary, problems = verify_evidence.verify_bundle(bundle)
    assert problems == []
    assert (summary["n_test_tasks"], summary["base_test_passed"],
            summary["selected_test_passed"]) == (6, 3, 4)
    assert summary["selected_epoch"] == 1 and summary["selected_decision"] == "adapter_selected"
    assert (summary["base_dev_passed"], summary["selected_dev_passed"]) == (1, 3)
    assert summary["test_delta_pp"] == pytest.approx(100 / 6)
    assert summary["optimizer_steps"] == 78
    assert summary["models"] == [f"{MODEL['model']}@{MODEL['resolved_revision']}"]


def test_bundle_carries_verdicts_but_not_private_content(artifacts, tmp_path):
    bundle = export_bundle(artifacts, tmp_path)
    text = bundle_text(bundle)
    for needle in ("def solution(): return PRIVATE_ANSWER", "test_secret_case", "/home/",
                   HARDWARE_MODEL, "example-teacher.com", "frozen/tasks.jsonl"):
        assert needle not in text, needle
    teacher = json.loads((bundle / "teacher.json").read_text(encoding="utf-8"))
    assert teacher["protocol"]["model"] == "teacher-model-x"      # 教师模型名公开
    assert teacher["protocol"]["base_url"] == "<url>"             # 服务地址不公开
    training = json.loads((bundle / "training.json").read_text(encoding="utf-8"))
    assert "gpu" not in training["environment"] and training["peak_vram_gb"] == 10.3
    selection = (bundle / "selection.json").read_text(encoding="utf-8")
    assert "<path>/epoch-001/adapter" in selection
    row = json.loads((bundle / "test/base.json").read_text(encoding="utf-8"))["results"][1]
    assert "raw_response" not in row and "grader_details" not in row
    assert row["grader"]["public"]["executed_tests"] == 1        # 判分证据保留计数
    assert "missing_tests" not in row["grader"]["public"]        # 不带私有测试名
    assert (bundle / "training-log.json").is_file() and not (bundle / "training-log.jsonl").exists()


def test_privacy_gate_blocks_the_whole_export(tmp_path):
    files = {"a.json": (json.dumps({"note": "CANARY-deadbeefcafe"}).encode(), None),
             "b.json": (json.dumps({"label": HARDWARE_MODEL}).encode(), None),
             "c.json": (json.dumps({"key": "sk-" + "a" * 24}).encode(), None)}
    violations = export_evidence.export("leaky", files, tmp_path / "evidence" / "leaky")
    assert sorted(item.split("]")[0] for item in violations) == [
        "[API密钥格式", "[硬件型号", "[私有 canary"]
    assert not (tmp_path / "evidence").exists()


def test_scrubbing_keeps_public_urls_and_shortens_paths():
    assert export_evidence.scrub_text(
        "see https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct and https://x.example/v1") == \
        "see https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct and <url>"
    assert export_evidence.scrub_text(
        "Adapter: /home/u/codeagent/checkpoints/sft-run/epoch-003/adapter") == \
        "Adapter: <path>/epoch-003/adapter"
    assert export_evidence.scrub_text("train/dev 和 2026/11/23 不受影响") == "train/dev 和 2026/11/23 不受影响"


def test_stale_hash_is_detected(artifacts, tmp_path):
    bundle = export_bundle(artifacts, tmp_path)
    path = bundle / "test/base.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["results"][0]["strict_pass"] = False
    write(path, payload)
    _, problems = verify_evidence.verify_bundle(bundle)
    assert any("sha256 differs" in problem for problem in problems)


def test_consistently_restamped_tampering_is_detected(artifacts, tmp_path):
    bundle = export_bundle(artifacts, tmp_path)
    path = bundle / "test/base.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["results"][1]["strict_pass"] = True          # 多算一道题，但没改指标
    write(path, payload)
    restamp(bundle, "test/base.json")
    _, problems = verify_evidence.verify_bundle(bundle)
    assert not any("sha256" in problem for problem in problems)
    assert any("recomputed" in problem for problem in problems)


def test_restamped_comparison_and_claims_are_recomputed(artifacts, tmp_path):
    bundle = export_bundle(artifacts, tmp_path)
    comparison = bundle / "test/comparison.md"
    comparison.write_text(comparison.read_text(encoding="utf-8").replace(
        "平均 pass@1 变化: **+16.67 个百分点**", "平均 pass@1 变化: **+33.33 个百分点**"),
        encoding="utf-8")
    restamp(bundle, "test/comparison.md")
    write(bundle / "claims.json", {"base_test_passed": 3, "selected_test_passed": 5})
    _, problems = verify_evidence.verify_bundle(bundle)
    assert any("comparison.md states" in problem for problem in problems)
    assert any("selected_test_passed claims 5" in problem for problem in problems)
    write(bundle / "claims.json", {"base_test_passed": 3, "selected_test_passed": 4,
                                   "selected_epoch": 1})
    _, problems = verify_evidence.verify_bundle(bundle)
    assert not any("claims.json" in problem for problem in problems)


def test_unlisted_files_are_rejected(artifacts, tmp_path):
    bundle = export_bundle(artifacts, tmp_path)
    (bundle / "notes.txt").write_text("added later", encoding="utf-8")
    _, problems = verify_evidence.verify_bundle(bundle)
    assert any("not listed" in problem for problem in problems)


def test_evidence_files_are_not_gitignored():
    result = subprocess.run(
        ["git", "check-ignore", "docs/evidence/seed42/test/base.json",
         "docs/evidence/seed42/training-log.json"], cwd=ROOT)
    assert result.returncode == 1                   # 1 = 没有任何路径被忽略
