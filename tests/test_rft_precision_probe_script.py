"""scripts/rft_precision_probe.sh：用做完的一轮 RFT 的训练数据，在另一种精度下重训各组，再交给 vLLM 评测。

端到端演练：把脚本放进临时仓库，PATH 最前面放一个假的 python（拦下 train_sft.py，记参数、造产物；run_state.py
给模型名；其余交给真解释器），vLLM 评测的包装脚本换成会记录参数的假脚本。rft_lib.sh 是真的。
不加载模型，不启动 vLLM。
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "rft_precision_probe.sh"

STUB = r'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path

args = sys.argv[1:]
if args[0].endswith("train_sft.py"):
    out = Path(args[args.index("--output") + 1])
    with open(os.environ["STUB_LOG"], "a", encoding="utf-8") as handle:
        handle.write("train\t" + " ".join(args) + "\n")
    if "--dry-run" in args:
        sys.exit(0)
    if os.environ.get("STUB_OOM"):
        print("torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB")
        sys.exit(1)
    (out / "epoch-001" / "adapter").mkdir(parents=True)
    (out / "epoch-001" / "adapter" / "adapter_config.json").write_text("{}")
    (out / "training-manifest.json").write_text(json.dumps({
        "schema_version": 2, "status": "complete", "model": "Qwen/Qwen3.5-9B", "revision": "abc123",
        "resolved_revision": "abc123", "train_config": {"lr": 1e-4, "lora_r": 32}}))
    print("fake train done")
elif args[0].endswith("run_state.py"):
    print("Qwen/Qwen3.5-9B")
else:
    os.execv(sys.executable, [sys.executable, *args])
'''

FAKE_EVAL = """#!/usr/bin/env bash
ARGS="INCLUDE_BASE=$INCLUDE_BASE EVAL_SAMPLES=$EVAL_SAMPLES TASKS_FILE=$TASKS_FILE RUN=$1 ARMS=${*:2}"
echo "eval\t$ARGS" >> "$STUB_LOG"
"""


@pytest.fixture
def stage(tmp_path):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "src/agentlab/experiments").mkdir(parents=True)
    (repo / "bin").mkdir()
    for name in ("rft_precision_probe.sh", "rft_lib.sh"):
        (repo / "scripts" / name).write_text((ROOT / "scripts" / name).read_text(encoding="utf-8"))
    (repo / "scripts" / "rft_vllm_eval.sh").write_text(FAKE_EVAL)
    (repo / "scripts" / "rft_vllm_eval.sh").chmod(0o755)
    (repo / "bin" / "python").write_text(STUB)
    (repo / "bin" / "python").chmod(0o755)
    source = tmp_path / "source"
    for arm in ("verified", "control"):
        for split in ("train", "dev"):
            path = source / "sft" / arm / f"{split}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n")
    tasks = tmp_path / "ladder.jsonl"
    tasks.write_text("{}\n")
    return {"repo": repo, "source": source, "tasks": tasks, "run": tmp_path / "run", "log": tmp_path / "calls.log",
            "home": tmp_path / "home"}


def run(stage, **environment):
    env = {**os.environ, "PATH": f"{stage['repo'] / 'bin'}{os.pathsep}{os.environ['PATH']}",
           "HOME": str(stage["home"]), "STUB_LOG": str(stage["log"]), "SOURCE_DIR": str(stage["source"]),
           "RUN_DIR": str(stage["run"]), "TASKS_FILE": str(stage["tasks"]), "VLLM_BIN": "/usr/bin/env",
           "PYTHONDONTWRITEBYTECODE": "1", **environment}
    return subprocess.run(["bash", "scripts/rft_precision_probe.sh"], cwd=stage["repo"], env=env,
                          capture_output=True, text=True, timeout=120)


def calls(stage, kind):
    path = stage["log"]
    if not path.exists():
        return []
    return [line.split("\t", 1)[1] for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith(kind + "\t")]


def test_a_fresh_run_trains_each_arm_in_the_requested_precision_and_hands_over_to_the_vllm_eval(stage):
    result = run(stage, QUANT="bf16")
    assert result.returncode == 0, result.stdout + result.stderr
    for step in ("1/5 检查前提", "2/5 A-verified：在 bf16 下训练", "3/5 C-control：在 bf16 下训练",
                 "4/5 写各组的选型结果", "5/5 经 vLLM（bf16）评基座和各组"):
        assert f"===== {step}" in result.stdout, step
    assert result.stdout.rstrip().splitlines()[-1] == "===== 全部结束 ====="
    trains = calls(stage, "train")
    # 每组请求两次：先带 --dry-run 查数据，再真训练；除此之外参数相同
    assert len(trains) == 4 and all(call.endswith(" --dry-run") for call in (trains[0], trains[2]))
    verified, control = trains[1], trains[3]
    assert trains[0] == verified + " --dry-run" and trains[2] == control + " --dry-run"
    assert "--quantization bf16" in verified and "--quantization bf16" in control
    assert "/sft/verified/train.jsonl" in verified and "--allow-unverified" not in verified
    assert "/sft/control/train.jsonl" in control and "--allow-unverified" in control       # 对照组的数据是没通过的答案
    for flag, value in (("--epochs", "1"), ("--lr", "1e-4"), ("--lora-r", "32"), ("--lora-alpha", "64"),
                        ("--grad-accum", "8"), ("--max-length", "2048"), ("--seed", "42")):
        assert f" {flag} {value} " in f" {verified} ", flag
    for arm in ("A-verified", "C-control"):
        selection = json.loads((stage["run"] / "results" / arm / "selection.json").read_text(encoding="utf-8"))
        adapter = stage["run"] / "checkpoints" / arm / "epoch-001" / "adapter"
        # rft_vllm_eval.sh 只认这几项：适配器的路径，以及训练时的基座和版本
        assert selection["selected_adapter"] == str(adapter) and selection["selected_epoch"] == 1
        assert selection["configuration"] == {"model": "Qwen/Qwen3.5-9B", "revision": "abc123", "quantization": "bf16",
                                              "lr": 1e-4, "lora_r": 32, "epochs": 1}
    (evaluated,) = calls(stage, "eval")
    assert evaluated.startswith("INCLUDE_BASE=1 EVAL_SAMPLES=4 TASKS_FILE=" + str(stage["tasks"]))
    assert evaluated.endswith(f"RUN={stage['run']} ARMS=A-verified C-control")


def test_a_rerun_skips_what_is_finished(stage):
    assert run(stage).returncode == 0
    again = run(stage)
    assert again.returncode == 0, again.stdout + again.stderr
    assert again.stdout.count("训练已经完成过，跳过训练") == 2
    assert again.stdout.count("选型结果已经有了，跳过") == 2
    assert len(calls(stage, "train")) == 4                         # 重跑没有再训练
    assert again.stdout.rstrip().splitlines()[-1] == "===== 全部结束 ====="


def test_an_out_of_memory_training_says_to_try_int8_and_stops(stage):
    result = run(stage, QUANT="bf16", STUB_OOM="1")
    assert result.returncode != 0
    assert "显存不够" in result.stdout and "QUANT=int8" in result.stdout
    assert "C-control：在" not in result.stdout and "===== 全部结束 =====" not in result.stdout
    assert calls(stage, "eval") == []


def test_a_quantization_name_it_does_not_know_stops_before_anything_runs(stage):
    result = run(stage, QUANT="fp8")
    assert result.returncode != 0 and "QUANT 只认 bf16、int8、nf4" in result.stdout
    assert calls(stage, "train") == []


def test_the_precision_switch_is_available_on_the_training_entry_point():
    sys.path.insert(0, str(ROOT / "scripts"))
    pytest.importorskip("transformers")
    import train_sft
    from agentlab.training import hf_runtime

    parser = train_sft.build_parser()
    assert parser.parse_args(["--data", "x.jsonl"]).quantization == "nf4"        # 默认还是 NF4，老行为不变
    assert parser.parse_args(["--data", "x.jsonl", "--quantization", "int8"]).quantization == "int8"
    with pytest.raises(SystemExit):
        parser.parse_args(["--data", "x.jsonl", "--quantization", "fp8"])
    assert hf_runtime.QUANTIZATIONS == ("nf4", "int8", "bf16")
    assert hf_runtime.quantization_config("bf16") is None
    assert hf_runtime.quantization_config("int8").load_in_8bit is True
    nf4 = hf_runtime.quantization_config("nf4")
    assert nf4.load_in_4bit and nf4.bnb_4bit_quant_type == "nf4"
    with pytest.raises(ValueError):
        hf_runtime.quantization_config("fp8")
