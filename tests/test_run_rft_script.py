# -*- coding: utf-8 -*-
"""run_rft.sh 的端到端演练：假的 python 给每个子脚本写出像样的产物，假的 vllm 服务记下自己的启动参数。

演的是编排本身——步骤顺序、每组的超参、基座报告的传递、中断后怎么接着跑、服务起不来时怎么停，
以及三个开关（STAGES=sampling、SKIP_ARMS、VLLM_EVAL=1）各自带来什么。rft_vllm_eval.sh、rft_lib.sh、
vllm_summary.py 用真的：VLLM_EVAL=1 那一步是靠它们做的。全程不碰真的显卡和服务。
"""
import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not shutil.which("bash"), reason="needs Bash")
RUN = "data/rft-round1"
# run_rft.sh 原有的两行标记（第 2、7 步）右边"====="前面没有空格，新的都有：两种都认
STEP = re.compile(r"^===== (.*?) *=====$", re.M)
HEREDOC = re.compile(r"python - (.*?) <<'PY'(?:[^\n]*\\\n[^\n]*)*[^\n]*\n(.*?)\nPY\n", re.S)
SWITCHES = ("MODEL", "MODEL_REVISION", "TASKS_FILE", "RUN_DIR", "RFT_TIERS", "RFT_SAMPLES", "RFT_EPOCHS", "RFT_LR",
            "RFT_LORA_R", "RFT_LORA_ALPHA", "OLD_LR", "OLD_LORA_R", "OLD_LORA_ALPHA", "GRAD_ACCUM", "MAX_LENGTH",
            "VLLM_BIN", "VLLM_PORT", "RFT_CONCURRENCY", "RFT_GRADE_WORKERS", "RFT_POLL_SECONDS", "BASE_REPORT",
            "STAGES", "SKIP_ARMS", "VLLM_EVAL", "EXPECT_REPRODUCE", "PROBE_IDS_FILE", "INCLUDE_BASE", "EVAL_SAMPLES")
STUB = '''#!/usr/bin/env python3
"""假的 python：编排脚本调到的每个脚本都给出像样的产物；其余命令交给真解释器。"""
import json, os, shutil, signal, sys
from pathlib import Path
args = sys.argv[1:]


def setting(name):
    return os.environ.get("STUB_" + name, "")

def value(flag, default=None):
    return args[args.index(flag) + 1] if flag in args else default

def log(name):
    with open(os.environ["STUB_LOG"], "a", encoding="utf-8") as handle:
        handle.write(name + " " + " ".join(args) + "\\n")

def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

if args[:1] == ["src/agentlab/experiments/run_state.py"]:
    print("Qwen/Qwen3.5-9B" if args[1] == "default-model" else "rev-123")
elif args[0].endswith("sample_rft.py"):
    log("sample")
    split, out = value("--split"), Path(value("--output-dir"))
    # STUB_KILL_SERVER="train=次数"：前几次采这个 split 时把假 vllm 关掉（模拟它中途退出）
    kills = Path(os.environ["STUB_LOG"] + ".kills-" + split)
    seen = int(kills.read_text()) if kills.exists() else 0
    kills.write_text(str(seen + 1))
    if seen < int(dict(item.split("=") for item in setting("KILL_SERVER").split(",") if item).get(split, 0)):
        pid_file = Path(os.environ["FAKE_VLLM_PID_FILE"])
        for pid in pid_file.read_text().split():
            try:
                os.kill(int(pid), signal.SIGKILL)
            except OSError:
                pass
        print("有 3 个槽位没能生成，原样重跑会只补这些（假的：vLLM 不在了）", file=sys.stderr)
        sys.exit(3)
    # STUB_SAMPLE_EXIT="dev=2"：采这个 split 时以这个退出码结束（2 是参数或采样协议不对）
    code = int(dict(item.split("=") for item in setting("SAMPLE_EXIT").split(",") if item).get(split, 0))
    if code:
        print("sample_rft.py: error: 采样协议与这个目录里已有的不一致（假的）", file=sys.stderr)
        sys.exit(code)
    # 和真的一样：清单里记着槽数、读得回的采样数、判分数（编排脚本 sampling_complete 核对这三个数）
    for name in (f"{split}-candidates.jsonl", f"{split}-graded.jsonl"):
        write(out / name, "{}\\n")
    write(out / f"{split}-manifest.json", json.dumps({"slots": 3, "raw_records": 3, "graded_records": 3}))
    print(f"结果：{split} 难题 解出 2/3（通过样本 5/24）")
elif args[0].endswith("prepare_data.py"):
    log("prepare")
    out = Path(value("--output-dir"))
    write(out / "train.jsonl", '{"task_id": "h3_a"}\\n')
    write(out / "dev.jsonl", '{"task_id": "h3_d"}\\n')
    write(out / "dataset-manifest.json", "{}")
elif args[0].endswith("rft_control.py"):
    log("control")
    out = Path(value("--output-dir"))
    write(out / "train.jsonl", '{"task_id": "h3_a"}\\n')
    write(out / "dev.jsonl", '{"task_id": "h3_d"}\\n')
    write(out / "dataset-manifest.json", "{}")
elif args[0].endswith("train_sft.py"):
    log("train")
    if "--dry-run" in args:
        print("Dry run 完成")
    else:
        write(Path(value("--output")) / "training-manifest.json", '{"status": "complete"}')
        print("训练完成")
elif args[0].endswith("select_checkpoint.py"):
    log("select")
    out = Path(value("--output"))
    out.mkdir(parents=True, exist_ok=True)
    # 选中的是 epoch-001 的适配器：把它也做出来，rft_vllm_eval.sh 要拿它去登记 LoRA
    adapter = Path(value("--checkpoint-root")) / "epoch-001/adapter"
    write(adapter / "adapter_config.json", "{}")
    if value("--base-report"):
        shutil.copyfile(value("--base-report"), out / "epoch-000-dev.json")
    else:
        write(out / "epoch-000-dev.json", '{"adapter": null}')
    if not (out / "epoch-001-dev.json").exists():
        write(out / "epoch-001-dev.json", json.dumps({"adapter": str(adapter)}))
    write(out / "selection.json", json.dumps({
        "selected_epoch": 1, "selected_adapter": str(adapter), "base_pass_at_1": 0.439,
        "configuration": {"model": value("--model"), "revision": value("--revision")},
        "candidates": [
            {"epoch": 0, "adapter": None, "pass_at_1": 0.439, "improved_tasks": 0, "regressed_tasks": 0},
            {"epoch": 1, "adapter": str(adapter), "pass_at_1": 0.5, "improved_tasks": 5, "regressed_tasks": 1}]}))
elif args[0].endswith("tier_report.py"):
    log("tiers")
    print("结果：（假的分层表）")
elif args[0].endswith("eval_vllm.py"):
    log("eval")
    label, stem = value("--label"), value("--tag") or value("--split")
    out, samples = Path(value("--output-dir")), int(value("--samples", "8"))
    results = [dict(task_id=spec["task_id"], strict_pass=len(spec["task_id"]) % 2 == 0)
               for spec in rows(value("--tasks-file")) if spec["split"] == value("--split")]
    write(out / f"{label}-{stem}-greedy.json", json.dumps({"results": results}))
    summary = {"label": label, "model": value("--model"), "split": value("--split"), "tag": stem, "sanity": None,
               "greedy": {"pass": sum(row["strict_pass"] for row in results), "total": len(results), "by_tier": {}},
               "sampled": None}
    if samples:
        summary["sampled"] = {"samples_per_task": samples, "attempts": samples * len(results),
                              "passed_attempts": 1, "pass_at_1": 1 / (samples * len(results)) if results else 0,
                              "any_pass_rate": 0.5}
        write(out / f"{label}-{stem}-sampled-k{samples}.json", json.dumps({"results": []}))
    write(out / f"{label}-{stem}-summary.json", json.dumps(summary, ensure_ascii=False))
    print(f"结果：{label} {stem} 贪心 {summary['greedy']['pass']}/{len(results)}（假的）")
elif args[0].endswith("vllm_summary.py"):
    log("summary")
    os.execv(sys.executable, [sys.executable, *args])        # 真的 vllm_summary.py 读上面写出的产物
else:
    os.execv(sys.executable, [sys.executable, *args])        # 内嵌的 Python 片段：真解释器从 stdin 读
'''

FAKE_VLLM = '''#!/usr/bin/env bash
# 假的 vllm serve：把自己的启动参数和 PID 记进文件。FAKE_VLLM_FAIL=1 时像显存不足那样失败；
# 否则报告启动完成，然后常驻（命令行里一直带着 "vllm serve"，关服务时按这个名字找得到它）
printf '%s\\n' "$*" >> "$FAKE_VLLM_LOG"
echo $$ >> "$FAKE_VLLM_PID_FILE"
if [[ "${FAKE_VLLM_FAIL:-0}" == 1 ]]; then
  echo "ERROR [core.py] torch.OutOfMemoryError: CUDA out of memory."
  echo "RuntimeError: Engine core initialization failed. See root cause above."
  exit 1
fi
echo "INFO:     Application startup complete."
while true; do sleep 1; done
'''

NVIDIA_SMI = '''#!/usr/bin/env bash
# 假的 nvidia-smi：第一块显卡已用的显存取自 FAKE_GPU_USED（MiB），没设就是查不到
case "$*" in *--query-gpu=memory.used*) echo "${FAKE_GPU_USED:-}" ;; esac
exit 0
'''

PKILL = '''#!/usr/bin/env bash
# 假的 pkill -f：照正则匹配整条命令行，但只对命令行里带着这次演练目录（STAGE）的进程动手，自己除外。
# 用真的 pkill 会连这台机器上别的 vllm serve 一起关掉
[[ "$1" == -f && $# == 2 ]] || { echo "假的 pkill 只会 pkill -f 模式" >&2; exit 2; }
status=1
for entry in /proc/[0-9]*; do
  [[ -r "$entry/cmdline" && "$entry" != "/proc/$$" ]] || continue
  mapfile -d '' -t argv < "$entry/cmdline"    # 命令行里各参数之间是 NUL，像真的 pkill 那样拼成一整行再匹配
  line="${argv[*]}"
  [[ "$line" == *"$STAGE"* && "$line" =~ $2 ]] || continue
  kill "${entry#/proc/}" 2>/dev/null && status=0
done
exit $status
'''

SLEEP = '''#!/usr/bin/env bash
# 演练用的 sleep：关掉旧服务后固定等的那 3 秒不必真等，其余照常
if [[ "$*" == 3 ]]; then exec @REAL@ 0.05; fi
exec @REAL@ "$@"
'''


def install(home, folder, name, text):
    program = home / folder / name
    program.parent.mkdir(parents=True, exist_ok=True)
    program.write_text(text, encoding="utf-8")
    program.chmod(0o755)


@pytest.fixture
def stage(tmp_path):
    """一份临时的仓库（编排脚本和它用到的库）、假的 python、vllm、nvidia-smi、pkill、sleep；HOME 也指到这里。"""
    home = tmp_path / "home"
    (home / "repo/scripts").mkdir(parents=True)
    for name in ("run_rft.sh", "rft_lib.sh", "rft_vllm_eval.sh", "vllm_summary.py"):
        shutil.copy(ROOT / "scripts" / name, home / "repo/scripts" / name)
    install(home, "bin", "python", STUB)
    install(home, "bin", "nvidia-smi", NVIDIA_SMI)
    install(home, "bin", "pkill", PKILL)
    install(home, "bin", "sleep", SLEEP.replace("@REAL@", shutil.which("sleep")))
    install(home, "vllm-qwen/bin", "vllm", FAKE_VLLM)
    (home / "repo/data").mkdir()
    (home / "repo/data/frozen-ladder.jsonl").write_text("".join(
        json.dumps({"task_id": f"h3_t{index}", "split": "train"}) + "\n" for index in range(3))
        + json.dumps({"task_id": "h3_d1", "split": "dev"}) + "\n", encoding="utf-8")
    yield home
    for pid in processes(home):                     # 断言失败时也不把假的 vllm 留在机器上
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)


def processes(stage):
    """这次演练里假 vllm 记下的 PID。"""
    pid_file = stage / "vllm.pid"
    return [int(pid) for pid in pid_file.read_text(encoding="utf-8").split()] if pid_file.exists() else []


def running(pid):
    """这个 PID 还在跑吗（已经退出、还没被收回的僵尸不算）。"""
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return False
    return state != "Z"


def alive(stage):
    return any(running(pid) for pid in processes(stage))


def eventually(condition, seconds=5.0):
    """等 condition 成立（假的 vllm 被关掉要一点时间）；到时间还不成立就返回它最后的值。"""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


def run(stage, log_path, **extra):
    """跑一遍 run_rft.sh。返回 (结果, 假 python 这次记下的调用，按先后的脚本名)。"""
    for name in (log_path, stage / "vllm.log"):      # 每次只看这一次的调用和启动
        name.unlink(missing_ok=True)
    clean = {name: value for name, value in os.environ.items()
             if name not in SWITCHES and not name.startswith(("STUB_", "FAKE_"))}
    environment = {**clean, "HOME": str(stage), "STAGE": str(stage), "STUB_LOG": str(log_path),
                   "FAKE_VLLM_LOG": str(stage / "vllm.log"), "FAKE_VLLM_PID_FILE": str(stage / "vllm.pid"),
                   "RFT_POLL_SECONDS": "1", "PATH": f"{stage / 'bin'}{os.pathsep}{os.environ['PATH']}", **extra}
    result = subprocess.run(["bash", "scripts/run_rft.sh"], cwd=stage / "repo", env=environment,
                            capture_output=True, text=True, timeout=180)
    # 有 Traceback 就是有什么地方崩了而编排没有接住：哪怕这次演练预期它以非 0 退出，也不该是这样退出的
    assert "Traceback" not in result.stderr, result.stderr[-2000:]
    calls = [line.split(" ", 1)[0] for line in log_path.read_text(encoding="utf-8").splitlines()] \
        if log_path.exists() else []
    return result, calls


def calls_matching(log_path, name):
    return [line[len(name) + 1:] for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.startswith(name + " ")]


def flag(args, name):
    found = re.search(r"--" + name + r" (\S+)", args)
    return found.group(1) if found else None


def starts(stage):
    """假 vllm 每次被启动时收到的参数。"""
    log = stage / "vllm.log"
    return log.read_text(encoding="utf-8").splitlines() if log.exists() else []


def test_a_fresh_run_samples_then_trains_three_arms_in_order(stage):
    log = stage / "calls.log"
    result, calls = run(stage, log)
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
    for marker in ("1/7 检查前提", "2/7 启动 vLLM 并采样", "3/7 准备训练数据", "4/7 A-verified",
                   "5/7 C-control", "6/7 B-oldhp", "7/7 汇总", "===== 全部结束 ====="):
        assert marker in result.stdout, marker
    assert calls == ["sample", "sample", "prepare", "control",
                     "train", "train", "select", "tiers",
                     "train", "train", "select", "tiers",
                     "train", "train", "select", "tiers", "tiers"]
    assert "--split train" in calls_matching(log, "sample")[0]
    assert "--split dev --tiers hard,variant --samples 8" in calls_matching(log, "sample")[1]
    trains = [call for call in calls_matching(log, "train") if "--dry-run" not in call]
    assert "--lr 1e-4 --lora-r 32 --lora-alpha 64" in trains[0] and "--allow-unverified" not in trains[0]
    assert "--lr 1e-4 --lora-r 32 --lora-alpha 64" in trains[1] and "--allow-unverified" in trains[1]
    assert "--lr 1e-5 --lora-r 16 --lora-alpha 32" in trains[2] and "--allow-unverified" not in trains[2]
    assert "/sft/verified/train.jsonl" in trains[0] and "/sft/control/train.jsonl" in trains[1]
    assert "/sft/verified/train.jsonl" in trains[2] and "--epochs 1" in trains[0]
    selects = calls_matching(log, "select")
    base = "data/rft-round1/results/A-verified/epoch-000-dev.json"       # 路径相对仓库根目录
    assert "--base-report" not in selects[0] and "--resume" in selects[0]      # 第一组自己评基座
    assert selects[1].endswith(f"--resume --base-report {base}")              # 后面两组复用基座的报告
    assert selects[2].endswith(f"--resume --base-report {base}")
    assert result.stdout.count("结果：") >= 6                                   # 每组都有一行结论
    assert "test 这一轮不评" in result.stdout
    assert calls_matching(log, "tiers")[-1] == ("scripts/tier_report.py data/rft-round1/results/A-verified "
                                                "data/rft-round1/results/C-control data/rft-round1/results/B-oldhp")
    assert "任务文件：4 题，train 3 / dev 1 / test 0；其中难题来源族 train 3 / dev 1" in result.stdout
    assert "其中合成题" not in result.stdout                                 # 任务文件里没有 s3_ 开头的题
    assert starts(stage) == ["serve Qwen/Qwen3.5-9B --revision rev-123 --language-model-only "
                             "--max-model-len 8192 --port 8000"]               # 只有采样那一次；VLLM_EVAL 默认关
    assert eventually(lambda: not alive(stage))                                # 采完把服务关掉


def test_rerunning_skips_everything_that_is_already_done(stage):
    log = stage / "calls.log"
    assert run(stage, log)[0].returncode == 0
    result, calls = run(stage, log)
    assert result.returncode == 0, result.stderr[-1500:]
    assert "采样已完成，跳过" in result.stdout and result.stdout.count("已经选过型，跳过") == 3
    assert calls == ["tiers", "tiers", "tiers", "tiers"]                        # 只剩下打印结论


def test_an_interrupted_selection_resumes_without_retraining_or_re_evaluating_the_base(stage):
    log = stage / "calls.log"
    assert run(stage, log)[0].returncode == 0
    shutil.rmtree(stage / "repo/data/rft-round1/results/C-control")            # 选到一半的产物不留
    (stage / "repo/data/rft-round1/results/B-oldhp/selection.json").unlink()
    result, calls = run(stage, log)
    assert result.returncode == 0, result.stderr[-1500:]
    assert calls == ["tiers", "select", "tiers", "select", "tiers", "tiers"]
    assert "训练已经完成过，跳过训练" in result.stdout
    for call in calls_matching(log, "select"):
        assert "--base-report" in call and "A-verified/epoch-000-dev.json" in call


def test_an_incomplete_training_directory_is_set_aside_and_redone(stage):
    log = stage / "calls.log"
    assert run(stage, log)[0].returncode == 0
    root = stage / "repo/data/rft-round1"
    shutil.rmtree(root / "results/A-verified")
    (root / "checkpoints/A-verified/training-manifest.json").write_text('{"status": "training"}')
    result, calls = run(stage, log)
    assert result.returncode == 0, result.stderr[-1500:]
    assert "重新训练" in result.stdout
    assert [name for name in (root / "checkpoints").iterdir()
            if name.name.startswith("A-verified.incomplete-")]                    # 没完成的那份留着
    assert calls[:3] == ["train", "train", "select"] and "--base-report" not in calls_matching(log, "select")[0]


def test_when_the_server_cannot_start_the_run_stops_before_sampling(stage):
    log = stage / "calls.log"
    result, calls = run(stage, log, FAKE_VLLM_FAIL="1")
    assert result.returncode == 1 and calls == []
    assert "vLLM 没起来" in result.stdout and "torch.OutOfMemoryError" in result.stdout
    assert not (stage / "repo/data/rft-round1/sampling").exists()


def test_stages_sampling_stops_after_sampling_and_the_rest_runs_another_night(stage):
    log = stage / "calls.log"
    result, calls = run(stage, log, STAGES="sampling")
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
    assert calls == ["sample", "sample"]                                     # 采样做完就停，不碰训练
    assert "结果：采样阶段结束" in result.stdout and "训练时去掉这个变量再运行一次" in result.stdout
    assert "===== 2/7 启动 vLLM 并采样" in result.stdout and "===== 3/7" not in result.stdout
    assert "===== 全部结束 =====" not in result.stdout                        # 这一轮还没做完
    assert not (stage / "repo/data/rft-round1/sft").exists()
    assert eventually(lambda: not alive(stage))                               # 停之前把服务关掉

    result, calls = run(stage, log, STAGES="sampling")                        # 采样已经做完：还是停在这里
    assert result.returncode == 0 and calls == []
    assert "采样已完成，跳过" in result.stdout and "结果：采样阶段结束" in result.stdout

    result, calls = run(stage, log)                                           # 第二晚：从准备数据接着做
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
    assert calls == ["prepare", "control", "train", "train", "select", "tiers",
                     "train", "train", "select", "tiers", "train", "train", "select", "tiers", "tiers"]
    assert "===== 2/7 采样已完成，跳过 =====" in result.stdout
    assert result.stdout.rstrip().endswith("===== 全部结束 =====")

    result, calls = run(stage, log, STAGES="train")                           # 不认识的值不能悄悄当成整轮
    assert result.returncode == 1 and calls == []
    assert "STAGES 只认 all（默认，整轮做完）或 sampling（采样做完就停），现在是 train" in result.stdout


def test_skipped_arms_are_not_trained_and_are_left_out_of_the_summary(stage):
    log = stage / "calls.log"
    result, calls = run(stage, log, SKIP_ARMS="B-oldhp")
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
    assert "===== 6/7 B-oldhp：列在 SKIP_ARMS 里，这一组不训，跳过 =====" in result.stdout
    assert calls == ["sample", "sample", "prepare", "control",
                     "train", "train", "select", "tiers", "train", "train", "select", "tiers", "tiers"]
    assert all("--lr 1e-5" not in call for call in calls_matching(log, "train"))  # 旧超参那一组没有训
    assert calls_matching(log, "tiers")[-1] == ("scripts/tier_report.py data/rft-round1/results/A-verified "
                                                "data/rft-round1/results/C-control")
    assert result.stdout.count("dev 0.439 → 0.500") == 2                      # 两组各有一行结论
    assert [marker for marker in ("4/7 A-verified", "5/7 C-control", "6/7 B-oldhp", "7/7 汇总")
            if marker in result.stdout] == ["4/7 A-verified", "5/7 C-control", "6/7 B-oldhp", "7/7 汇总"]

    result, calls = run(stage, log, SKIP_ARMS="B-oldhp")                      # 重跑时它照样跳过
    assert result.returncode == 0 and calls == ["tiers", "tiers", "tiers"]
    assert result.stdout.count("已经选过型，跳过") == 2 and "这一组不训，跳过" in result.stdout


def test_skipping_the_first_arm_leaves_the_base_to_be_evaluated_by_the_next_one(stage):
    log = stage / "calls.log"
    result, calls = run(stage, log, SKIP_ARMS="A-verified B-oldhp")
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
    assert calls == ["sample", "sample", "prepare", "control", "train", "train", "select", "tiers", "tiers"]
    select, = calls_matching(log, "select")
    assert "--base-report" not in select and "results/C-control" in select    # 它成了第一组：基座由它自己评
    assert calls_matching(log, "tiers")[-1] == "scripts/tier_report.py data/rft-round1/results/C-control"


def test_vllm_eval_rates_the_kept_arms_after_the_summary(stage):
    log = stage / "calls.log"
    # 外面留着上一轮引导的变量也不要紧：这一轮没有族内探针，也不对哪一组自检
    result, calls = run(stage, log, VLLM_EVAL="1", EXPECT_REPRODUCE="A-verified", PROBE_IDS_FILE="data/nowhere.txt")
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-1000:]
    assert calls == ["sample", "sample", "prepare", "control",
                     "train", "train", "select", "tiers", "train", "train", "select", "tiers",
                     "train", "train", "select", "tiers", "tiers", "eval", "eval", "eval", "summary"]
    # 步骤标记还是 1/7 到 7/7；vLLM 评测排在第 7 步汇总之后、"全部结束"之前（它自己的步骤是 x/8）
    own = [marker.split(" ")[0] for marker in STEP.findall(result.stdout) if "/7 " in marker]
    assert own == [f"{number}/7" for number in range(1, 8)]
    assert result.stdout.index("7/7 汇总") < result.stdout.index("1/8 读取各组的选型结果")
    assert "----- 经本机 vLLM 服务评没跳过的组选中的适配器（辅助指标）-----" in result.stdout
    assert result.stdout.rstrip().endswith("===== 全部结束 =====")
    assert "找不到探针题号文件" not in result.stdout and "EXPECT_REPRODUCE 里列了组" not in result.stdout
    assert "5/8 评基座：INCLUDE_BASE 不是 1，跳过" in result.stdout           # 只评三组，不评基座
    evaluations = calls_matching(log, "eval")
    assert [(flag(call, "label"), flag(call, "split"), flag(call, "samples")) for call in evaluations] == [
        ("A-verified", "dev", "4"), ("C-control", "dev", "4"), ("B-oldhp", "dev", "4")]
    assert all("--ids-file" not in call and "--sanity-tasks-file" not in call for call in evaluations)
    adapters = f"{stage}/repo/data/rft-round1/checkpoints"
    assert starts(stage)[1:] == ["serve Qwen/Qwen3.5-9B --revision rev-123 --language-model-only "
                                 "--max-model-len 8192 --port 8000 --enable-lora --max-lora-rank 64 --lora-modules "
                                 + " ".join(f"{group}={adapters}/{group}/epoch-001/adapter"
                                            for group in ("A-verified", "C-control", "B-oldhp"))]
    assert calls_matching(log, "summary") == ["scripts/vllm_summary.py --run data/rft-round1 "
                                              "--arms A-verified C-control B-oldhp"]
    assert "── vLLM 评测（bf16 权重，辅助指标；主结果是上面 HF 贪心那张表） ──" in result.stdout

    # 没跳过的组只有两个时只评这两个；INCLUDE_BASE 是 rft_vllm_eval.sh 自己的开关，照常从环境里读
    shutil.rmtree(stage / "repo/data/rft-round1/vllm")
    result, calls = run(stage, log, VLLM_EVAL="1", SKIP_ARMS="B-oldhp", INCLUDE_BASE="1")
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1000:]
    assert [flag(call, "label") for call in calls_matching(log, "eval")] == ["base", "A-verified", "C-control"]
    assert "===== 5/8 评基座 =====" in result.stdout and calls[-1] == "summary"
    assert "B-oldhp=" not in starts(stage)[-1]


def panel_state(stage, log_text):
    """把一份日志交给进度面板（scripts/show_progress.sh），返回它显示的状态那一行。"""
    log = stage / "panel.log"
    log.write_text(log_text, encoding="utf-8")
    clean = {name: value for name, value in os.environ.items()
             if name not in SWITCHES and not name.startswith(("STUB_", "FAKE_"))}
    shown = subprocess.run(["bash", str(ROOT / "scripts/show_progress.sh"), str(log), "", ""], cwd=stage / "repo",
                           env={**clean, "HOME": str(stage)}, capture_output=True, text=True, timeout=60)
    assert shown.returncode == 0, shown.stderr
    return next(line for line in shown.stdout.splitlines() if line.startswith("状态:"))


def test_a_vllm_eval_that_does_not_finish_is_said_so_at_the_end_and_is_done_on_a_rerun(stage):
    """vLLM 评测是辅助指标，没评完不影响 dev 上的选型结果。但它没评完就是没做完：不能照样打印"全部结束"再以 0 退出
    ——进度面板见到那一行就显示"已完成"。原样重跑只补没评完的。"""
    log = stage / "calls.log"
    shutil.copy(ROOT / "scripts/rft_vllm_eval.sh", stage / "kept-eval.sh")
    (stage / "repo/scripts/rft_vllm_eval.sh").write_text("echo '假的 rft_vllm_eval.sh：vLLM 没起来'\nexit 1\n",
                                                          encoding="utf-8")
    result, calls = run(stage, log, VLLM_EVAL="1")
    assert result.returncode == 1, result.stdout[-1500:] + result.stderr[-1000:]
    assert "假的 rft_vllm_eval.sh：vLLM 没起来" in result.stdout
    assert "===== 7/7 汇总（都在 dev 上；test 这一轮不评）=====" in result.stdout      # 主结果已经在前面打出来了
    assert result.stdout.rstrip().endswith(
        "===== 没有全部完成：vLLM 评测没有全部完成（rft_vllm_eval.sh 以退出码 1 结束）；"
        "上面的 dev 选型结果不受影响，重新运行同一条命令只补没做完的 =====")
    assert "===== 全部结束 =====" not in result.stdout
    assert calls[-1] == "tiers" and len(starts(stage)) == 1                   # 主结果做完以后才轮到它；评测没开服务
    assert panel_state(stage, result.stdout).startswith("状态: 已结束，但没有全部完成")

    shutil.copy(stage / "kept-eval.sh", stage / "repo/scripts/rft_vllm_eval.sh")
    result, calls = run(stage, log, VLLM_EVAL="1")                            # 重跑：训练都跳过，只把评测补上
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1000:]
    assert calls == ["tiers"] * 4 + ["eval"] * 3 + ["summary"]
    assert result.stdout.count("已经选过型，跳过") == 3
    assert result.stdout.rstrip().endswith("===== 全部结束 =====")
    assert panel_state(stage, result.stdout) == "状态: 已完成"


def test_a_sampling_manifest_with_records_missing_is_not_taken_as_done(stage):
    """清单在不等于采样做完了：旧版本的 sample_rft.py 读不回某些采样时照样写清单（那几道题就被记成"一次都没通过"）。
    槽数 ≠ 记录数的清单不算数——重新调用它，只补缺的槽位；另一个 split 做完了的不重做。"""
    log = stage / "calls.log"
    assert run(stage, log)[0].returncode == 0
    path = stage / "repo/data/rft-round1/sampling/dev-manifest.json"
    path.write_text(json.dumps({"slots": 3, "raw_records": 2, "graded_records": 2}), encoding="utf-8")
    result, calls = run(stage, log)
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
    assert calls == ["sample", "tiers", "tiers", "tiers", "tiers"]           # 只重做 dev 那一份采样
    assert "===== 2/7 启动 vLLM 并采样（难题和变体，每题 8 次）=====" in result.stdout
    assert "train 的采样已经做完了，跳过（只补另一个 split 缺的槽位）" in result.stdout
    assert "采样已完成，跳过" not in result.stdout and "===== 3/7" in result.stdout
    assert len(starts(stage)) == 1 and eventually(lambda: not alive(stage))
    assert json.loads(path.read_text(encoding="utf-8")) == {"slots": 3, "raw_records": 3, "graded_records": 3}
    assert result.stdout.rstrip().endswith("===== 全部结束 =====")


def test_a_server_that_quits_during_sampling_is_restarted_and_the_sampling_goes_on(stage):
    """采样中途 vLLM 退出（显存不足、被系统杀掉）：对着一个不在的服务重试三次没有用，夜里就停在那里了。脚本把服务
    重新启动一次，接着只补没完成的槽位。"""
    log = stage / "calls.log"
    result, calls = run(stage, log, STUB_KILL_SERVER="train=1")
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
    assert calls[:3] == ["sample", "sample", "sample"]                       # train 两次（重启后补的算一次）、dev 一次
    assert "采样 train 没跑完的时候 vLLM 已经不在了。vLLM 日志最后 15 行：" in result.stdout
    assert "INFO:     Application startup complete." in result.stdout        # 服务日志的最后几行原样给用户看
    assert "重新启动 vLLM 后再试（只补没完成的槽位）" in result.stdout
    assert "5 秒后重试" not in result.stdout                                 # 不对着不在的服务干等
    assert len(starts(stage)) == 2 and starts(stage)[0] == starts(stage)[1]
    assert "===== 全部结束 =====" in result.stdout and eventually(lambda: not alive(stage))


def test_a_server_that_quits_again_during_sampling_stops_with_the_log_tail(stage):
    """重启过一次 vLLM 又退出：不再重启（免得整夜反复加载），停下时带上服务日志的末尾。"""
    log = stage / "calls.log"
    result, calls = run(stage, log, STUB_KILL_SERVER="train=2")
    assert result.returncode == 1, result.stdout[-1500:] + result.stderr[-1000:]
    assert calls == ["sample", "sample"] and len(starts(stage)) == 2
    assert "采样 train 没跑完，vLLM 又不在了（已经重新启动过一次），停止。vLLM 日志最后 15 行：" in result.stdout
    assert "===== 3/7" not in result.stdout and "===== 全部结束 =====" not in result.stdout
    assert not (stage / "repo/data/rft-round1/sft").exists() and eventually(lambda: not alive(stage))


def test_a_sampling_call_refused_with_exit_code_2_is_not_retried(stage):
    """退出码 2 是参数或采样协议不对（两次运行之间改了采样次数）：重试没有用，也不该拿 vLLM 日志的末尾当原因。"""
    log = stage / "calls.log"
    result, calls = run(stage, log, STUB_SAMPLE_EXIT="dev=2")
    assert result.returncode == 1 and calls == ["sample", "sample"]         # dev 只调了一次
    assert "采样 dev 没有做成：sample_rft.py 以退出码 2 结束，是参数或采样协议不对，重试没有用。把上面的报错贴给我" \
        in result.stdout
    assert "vLLM 日志最后 15 行" not in result.stdout and "5 秒后重试" not in result.stdout
    assert not (stage / "repo/data/rft-round1/sft").exists() and eventually(lambda: not alive(stage))


def test_when_every_arm_is_skipped_there_is_nothing_to_summarise_or_rate(stage):
    log = stage / "calls.log"
    result, calls = run(stage, log, SKIP_ARMS="A-verified C-control B-oldhp", VLLM_EVAL="1")
    assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1000:]
    assert calls == ["sample", "sample", "prepare", "control"]
    assert result.stdout.count("这一组不训，跳过") == 3
    assert "没有已经选完的组，汇总跳过。" in result.stdout
    assert "三组都列在 SKIP_ARMS 里，没有要评的组，vLLM 评测跳过。" in result.stdout
    assert len(starts(stage)) == 1 and "-----" not in result.stdout           # 只有采样开过服务

    result, calls = run(stage, log, VLLM_EVAL="2")
    assert result.returncode == 1 and calls == [] and "VLLM_EVAL 只能是 0 或 1（现在是 2）" in result.stdout


def test_the_task_count_snippet_also_counts_synthetic_tasks_when_there_are_any(tmp_path):
    """第 1 步内嵌的 Python 用真解释器跑：有 s3_、s3v_ 开头的合成题时多数一行，没有就不提。"""
    snippet, = [found for found in HEREDOC.findall((ROOT / "scripts/run_rft.sh").read_text(encoding="utf-8"))
                if '"$TASKS"' in found[0]]
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text('{"task_id": "h3_t1", "split": "train"}\n{"task_id": "h3v_d1", "split": "dev"}\n'
                     '{"task_id": "v2_t1", "split": "train"}\n', encoding="utf-8")

    def count():
        return subprocess.run([sys.executable, "-", str(tasks)], input=snippet[1], text=True, capture_output=True,
                              timeout=60)

    result = count()
    assert result.returncode == 0, result.stderr
    assert result.stdout == "任务文件：3 题，train 2 / dev 1 / test 0；其中难题来源族 train 1 / dev 1\n"
    with tasks.open("a", encoding="utf-8") as handle:
        handle.write('{"task_id": "s3_t1", "split": "train"}\n{"task_id": "s3v_d1", "split": "dev"}\n'
                     '{"task_id": "s3_x1", "split": "test"}\n')
    assert count().stdout == ("任务文件：6 题，train 3 / dev 2 / test 1；其中难题来源族 train 1 / dev 1\n"
                              "其中合成题（s3_、s3v_ 开头）train 1 / dev 1 / test 1\n")


def test_the_sampling_complete_snippet_wants_as_many_records_as_slots(tmp_path):
    """内嵌的 sampling_complete 片段用真解释器跑：清单在不够，还要 槽数 = 采样数 = 判分数（都大于 0）。"""
    snippet, = [found for found in HEREDOC.findall((ROOT / "scripts/run_rft.sh").read_text(encoding="utf-8"))
                if '"$RUN/sampling/$1-manifest.json"' in found[0]]
    whole = {"tasks": 3, "slots": 6, "raw_records": 6, "graded_records": 6, "zero_pass_tasks": []}

    def done(manifest):
        (tmp_path / "train-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return subprocess.run([sys.executable, "-", str(tmp_path / "train-manifest.json")], input=snippet[1],
                              text=True, capture_output=True, timeout=60).returncode

    assert done(whole) == 0
    for broken in ({**whole, "raw_records": 5}, {**whole, "graded_records": 5}, {**whole, "slots": 7},
                   {**whole, "slots": 0, "raw_records": 0, "graded_records": 0}, {}, [1], "{not json"):
        assert done(broken) == 1, broken
