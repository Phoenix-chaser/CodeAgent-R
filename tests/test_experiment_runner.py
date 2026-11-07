import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from agentlab.experiments.run_state import check_receipt, mark_receipt, validate_split
from agentlab.training.preparation import content_hash


ROOT = Path(__file__).resolve().parents[1]


def test_receipt_detects_input_output_and_adapter_changes(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text("first")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"weights")
    receipt = tmp_path / "done.json"
    mark_receipt(receipt, [source], [adapter])
    assert check_receipt(receipt, [source], [adapter])[0]
    source.write_text("changed")
    assert not check_receipt(receipt, [source], [adapter])[0]
    source.write_text("first")
    weights.write_bytes(b"other weights")
    assert not check_receipt(receipt, [source], [adapter])[0]
    weights.unlink()
    assert not check_receipt(receipt, [source], [adapter])[0]


def test_frozen_split_cannot_reuse_changed_pool(tmp_path):
    tasks = [{"task_id": "case", "split": "train", "buggy": "old", "description": "边界"}]
    (tmp_path / "tasks.jsonl").write_text(json.dumps(tasks[0]), encoding="utf-8")
    (tmp_path / "split-manifest.json").write_text(json.dumps({"tasks_sha256": content_hash(tasks)}))
    pool = tmp_path / "pool.jsonl"
    pool.write_text(json.dumps({**tasks[0], "split": "unfrozen"}))
    validate_split(tmp_path, pool)
    pool.write_text(json.dumps({**tasks[0], "buggy": "new"}))
    with pytest.raises(ValueError, match="Task pool changed"):
        validate_split(tmp_path, pool)
    (tmp_path / "tasks.jsonl").write_text(json.dumps({**tasks[0], "buggy": "new"}))
    with pytest.raises(ValueError, match="Frozen tasks differ"):
        validate_split(tmp_path, None)


def test_frozen_split_must_match_requested_seed(tmp_path):
    tasks = [{"task_id": "case", "split": "train"}]
    (tmp_path / "tasks.jsonl").write_text(json.dumps(tasks[0]))
    manifest = {"tasks_sha256": content_hash(tasks), "seed": 42,
                "dev_ratio": 0.15, "test_ratio": 0.15}
    (tmp_path / "split-manifest.json").write_text(json.dumps(manifest))
    validate_split(tmp_path, None, 42)
    with pytest.raises(ValueError, match="split configuration differs"):
        validate_split(tmp_path, None, 43)


def test_incomplete_frozen_audit_is_rejected(tmp_path):
    tasks = [{"task_id": "case", "split": "train"}]
    (tmp_path / "tasks.jsonl").write_text(json.dumps(tasks[0]))
    (tmp_path / "split-manifest.json").write_text(json.dumps({"tasks_sha256": content_hash(tasks)}))
    audit = {"tasks_sha256": content_hash(tasks), "reference_passes": 1, "bugs_reproduced": 0}
    (tmp_path / "audit-manifest.json").write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="audit does not cover"):
        validate_split(tmp_path, None)


@pytest.fixture
def runner(tmp_path):
    if os.name == "nt" or not shutil.which("bash"):
        pytest.skip("The GPU runner targets Linux Bash")
    for relative in ("scripts/run_experiment_v2.sh", "src/agentlab/experiments/run_state.py"):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    for relative in ("scripts/train_sft.py", "src/agentlab/training/hf_sft.py",
                     "scripts/collect_data.py", "src/agentlab/benchmark/protocol.py"):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("stub")
    data = tmp_path / "sft"
    data.mkdir()
    (data / "train.jsonl").write_text("train")
    (data / "dev.jsonl").write_text("dev")
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    (experiment / "tasks.jsonl").write_text("tasks")
    binary = tmp_path / "bin"
    binary.mkdir()
    shim = binary / "python"
    shim.write_text(f"#!{sys.executable}\n" + '''
import json, os, pathlib, sys
args = sys.argv[1:]
if args[0].endswith('run_state.py') or args[0] == '-':
    os.execv(sys.executable, [sys.executable, *args])
with open(os.environ['RUNNER_TRACE'], 'a') as trace:
    trace.write(json.dumps(args) + '\\n')
def argument(name):
    return args[args.index(name) + 1]
if args[0] == 'scripts/train_sft.py':
    if os.environ.get('FAKE_TRAIN_FAILURE'):
        raise SystemExit(17)
    if '--dry-run' not in args:
        output = pathlib.Path(argument('--output'))
        (output / 'adapter').mkdir(parents=True)
        for name in ('training-manifest.json', 'training-log.jsonl', 'checkpoint-index.json',
                     'adapter/adapter_model.safetensors'):
            (output / name).write_text('{}')
elif args[0] == 'scripts/collect_data.py':
    output = pathlib.Path(argument('--output'))
    if '--status-only' in args:
        raise SystemExit(int(os.environ.get('FAKE_COLLECT_STATUS', '0' if output.exists() else '3')))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('{}\\n')
    output.with_suffix('.manifest.json').write_text('{}')
else:
    raise SystemExit('unexpected fake invocation: ' + str(args))
''', encoding="utf-8")
    shim.chmod(0o755)
    environment = {key: value for key, value in os.environ.items() if key in {
        "PATH", "HOME", "SYSTEMROOT", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL"}}
    environment.update(PATH=str(binary) + os.pathsep + environment.get("PATH", ""),
                       EXP_DIR=str(experiment), SFT_DIR=str(data), RUNNER_TRACE=str(tmp_path / "trace"))

    def invoke(stage, **settings):
        return subprocess.run(["bash", "scripts/run_experiment_v2.sh", stage],
                              cwd=tmp_path, env={**environment, **settings},
                              capture_output=True, text=True, timeout=30)

    def trace():
        path = tmp_path / "trace"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    return tmp_path, environment, invoke, trace


def test_training_failure_has_no_receipt_and_does_not_run_after_dry_run(runner):
    root, _, invoke, trace = runner
    result = invoke("train", FAKE_TRAIN_FAILURE="1")
    assert result.returncode == 17, result.stdout + result.stderr
    assert len(trace()) == 1
    assert "--dry-run" in trace()[0]
    assert not list(root.glob(".state/experiments/*/train.json"))


def test_same_training_resumes_but_different_checkpoint_does_not_skip(runner):
    root, _, invoke, trace = runner
    assert invoke("train").returncode == 0
    assert len(trace()) == 2
    assert invoke("train").returncode == 0
    assert len(trace()) == 2
    result = invoke("train", CKPT_DIR=str(root / "second-checkpoint"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(trace()) == 4
    assert (root / "second-checkpoint/adapter/adapter_model.safetensors").is_file()
    assert len(list(root.glob(".state/experiments/*/train.json"))) == 2


@pytest.mark.parametrize("setting", [{"LR": "1e-5"}, {"MODEL_REVISION": "different"}, {"MAX_PER_TASK": "2"}])
def test_configuration_change_gets_separate_state_and_checkpoint(runner, setting):
    root, _, invoke, trace = runner
    assert invoke("train").returncode == 0
    assert invoke("train", **setting).returncode == 0
    assert len(trace()) == 4
    assert len(list(root.glob("checkpoints/*/adapter"))) == 2


def test_changed_training_data_cannot_silently_reuse_existing_adapter(runner):
    root, _, invoke, trace = runner
    assert invoke("train").returncode == 0
    (root / "sft/train.jsonl").write_text("new data")
    result = invoke("train")
    assert result.returncode != 0
    assert "CKPT_DIR" in result.stdout + result.stderr
    assert len(trace()) == 2


def test_completed_collection_needs_no_key_or_teacher_calls(runner):
    root, _, invoke, trace = runner
    directory = root / "experiment/candidates"
    directory.mkdir()
    for split in ("train", "dev"):
        (directory / f"{split}.jsonl").write_text("{}\n")
        (directory / f"{split}.manifest.json").write_text("{}")
    result = invoke("collect")
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(trace()) == 2
    assert all("--status-only" in call for call in trace())
    assert len(list(root.glob(".state/experiments/*/collect.json"))) == 1


def test_incomplete_collection_cannot_create_done_receipt(runner):
    root, _, invoke, trace = runner
    result = invoke("collect", FAKE_COLLECT_STATUS="3", TEACHER_API_KEY="fixture-key",
                    ASSUME_YES="1", RETRY_ERRORS="1")
    assert result.returncode == 3, result.stdout + result.stderr
    assert not list(root.glob(".state/experiments/*/collect.json"))
    requests = [call for call in trace() if "--status-only" not in call]
    assert len(requests) == 2
    assert all("--retry-errors" in call for call in requests)


def test_test_split_is_not_allowed_in_routine_eval(runner):
    _, _, invoke, trace = runner
    result = invoke("eval", EVAL_SPLIT="test")
    assert result.returncode == 2
    assert "final-eval" in result.stdout + result.stderr
    assert trace() == []


@pytest.mark.parametrize("verdict,expected_code", [
    ("strict_pass", 0), ("collection_error", 3), ("incomplete_response", 3),
])
def test_collection_status_cli_requires_no_api_key(tmp_path, verdict, expected_code):
    from agentlab.benchmark.pilot_tasks import PILOT_SPECS
    from agentlab.benchmark.protocol import protocol_hash, spec_fingerprint

    spec = {**PILOT_SPECS[0], "split": "train"}
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps(spec) + "\n", encoding="utf-8")
    output = tmp_path / "candidates.jsonl"
    record = {"task_id": spec["task_id"], "candidate_idx": 0,
              "task_hash": spec_fingerprint(spec), "provenance": {"verdict": verdict}}
    output.write_text(json.dumps(record) + "\n", encoding="utf-8")
    protocol = {"schema_version": 2, "tasks_hash": protocol_hash([spec]), "split": "train",
                "model": "fixture-teacher", "base_url": "http://invalid.invalid", "max_tokens": 4096,
                "temperature": 0.2, "exploration_temperature": 0.7, "include_failures": False}
    output.with_suffix(".manifest.json").write_text(json.dumps({"protocol": protocol}))
    environment = dict(os.environ)
    environment.pop("RUNNER_FIXTURE_API_KEY", None)
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/collect_data.py"), "--tasks-file", str(tasks),
        "--output", str(output), "--candidates", "1", "--status-only", "--model", "fixture-teacher",
        "--base-url", "http://invalid.invalid", "--api-key-env", "RUNNER_FIXTURE_API_KEY",
    ], capture_output=True, text=True, timeout=30, env=environment)
    assert result.returncode == expected_code, result.stdout + result.stderr
    status = json.loads(result.stdout)
    assert status["slots"] == status["recorded"] == 1
    assert status["retryable"] == int(expected_code == 3)
