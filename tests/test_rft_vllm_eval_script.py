# -*- coding: utf-8 -*-
"""rft_lib.sh 和 rft_vllm_eval.sh 的演练。

假的 python 给 eval_vllm.py 写出像样的产物（文件名和 summary 照它的约定；vllm_summary.py 用真的去读），
假的 vllm 服务把自己的启动参数记下来，假的 nvidia-smi 按环境变量报显存，假的 pkill 只管这次演练自己的进程。
演的是编排本身——先探针后 dev、自检没过（适配器的输出和基座完全一样）就作废、自检没能比较的不作废只注明、
重跑时跳过、服务起不来或显卡被占时怎么停；
库函数挨个调一遍。最后一个用例不用假的 python：库函数真的去调 eval_vllm.py（服务是本机的假服务，
判分是真的），核对两边的文件名和字段对得上。全程不碰真的显卡和服务。
"""
import contextlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not shutil.which("bash"), reason="needs Bash")
RUN = "data/rft-guided"
GROUPS = ("D-prefix", "D-skill", "F-control")
PROBE = f"{RUN}/probe-ids.txt"
TRAINED = {"D-prefix": 2, "D-skill": 3, "F-control": 2}     # 各组的训练数据里有几道探针题

PYTHON = r'''#!/usr/bin/env python3
"""假的 python：编排脚本调到的每个脚本都写出像样的产物，并把参数记进 STUB_LOG；
内嵌的 Python 片段和 vllm_summary.py 交给真解释器。"""
import json, os, shutil, signal, sys
from pathlib import Path
args = sys.argv[1:]


def value(flag, default=None):
    return args[args.index(flag) + 1] if flag in args else default


def log(name):
    with open(os.environ["STUB_LOG"], "a", encoding="utf-8") as handle:
        handle.write(name + " " + " ".join(args) + "\n")


def write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def counts(rows):
    tiers = {tier: [0, 0] for tier in ("default", "hard", "variant")}      # 三层都写，和 eval_vllm.py 一样
    for task, passed in rows:
        tier = "variant" if task.startswith("h3v_") else "hard" if task.startswith("h3_") else "default"
        tiers[tier][0] += bool(passed)
        tiers[tier][1] += 1
    return {"pass": sum(bool(passed) for _, passed in rows), "total": len(rows), "by_tier": tiers}


def kill_server(tag):
    """STUB_KILL_SERVER="标签/段=次数,…"：前几次评这一部分时把假 vllm 关掉（模拟它中途退出）。"""
    wanted = dict(item.split("=") for item in os.environ.get("STUB_KILL_SERVER", "").split(",") if item)
    counter = Path(os.environ["FAKE_VLLM_PID_FILE"] + ".kills-" + tag.replace("/", "-"))
    seen = int(counter.read_text()) if counter.exists() else 0
    counter.write_text(str(seen + 1))
    if seen >= int(wanted.get(tag, 0)):
        return False
    pid_file = Path(os.environ["FAKE_VLLM_PID_FILE"])
    for pid in pid_file.read_text().split() if pid_file.exists() else []:
        try:
            os.kill(int(pid), signal.SIGKILL)
        except OSError:
            pass
    return True


if args[:1] == ["src/agentlab/experiments/run_state.py"]:
    print("org/model" if args[1] == "default-model" else os.environ.get("STUB_REVISION", "rev-123"))
elif args[0].endswith("eval_vllm.py"):
    log("eval")
    label, stem = value("--label"), value("--tag") or value("--split")
    if kill_server(label + "/" + stem):                   # vLLM 在评这一部分时退出了
        print("假的 eval_vllm.py：" + label + " " + stem + " 评到一半 vLLM 不在了，以退出码 3 结束")
        sys.exit(3)
    out, model, samples = Path(value("--output-dir")), value("--model"), int(value("--samples", "8"))
    rows = [(spec["task_id"], (len(model) + len(spec["task_id"])) % 2 == 0)
            for spec in lines(value("--tasks-file")) if spec["split"] == value("--split")]
    if value("--ids-file"):
        wanted = {line.strip() for line in Path(value("--ids-file")).read_text(encoding="utf-8").splitlines()}
        rows = [row for row in rows if row[0] in wanted]
    failed = dict(item.split("=") for item in os.environ.get("STUB_EVAL_FAIL", "").split())
    code = int(failed.get(label + "/" + stem, 0))
    if code:
        print("假的 eval_vllm.py：" + label + " " + stem + " 没评完，以退出码 " + str(code) + " 结束")
        sys.exit(code)                                 # 没评完：什么都不写
    sanity = None
    if value("--sanity-tasks-file"):
        # 和真的一样（新判据）：自检拿训练过的题比较适配器和基座的输出，写出 <标签>-<stem>-sanity.json。
        # STUB_SANITY="组名=几道和基座不同"：一道都不同（0）就以退出码 4 结束，不写报告和 summary；
        # STUB_UNCOMPARED="组名"：演出"拿不到基座名"——没能比较，不作废，照常往下评
        forced = dict(item.split("=") for item in os.environ.get("STUB_SANITY", "").split(",") if item)
        changed = int(forced.get(label, 4))
        unpaired = label in os.environ.get("STUB_UNCOMPARED", "").split()
        minimum = int(value("--sanity-min", "1"))
        sanity = {"pass": max(0, changed - 1), "total": 6, "min": minimum, "changed": 0 if unpaired else changed,
                  "compared": not unpaired, "compared_tasks": 0 if unpaired else 6, "not_compared_reason": None}
        if unpaired:
            sanity["not_compared_reason"] = "服务的模型列表里分不出 " + label + " 挂在哪个基座上，拿不到基座名"
        write(out / (label + "-" + stem + "-sanity.json"), json.dumps(sanity, ensure_ascii=False))
        print("自检：训练过的题严格通过 " + str(sanity["pass"]) + "/6（仅供参考；训练强度小时复现不了是正常的）")
        if unpaired:
            print("自检：没能和基座比较（" + sanity["not_compared_reason"]
                  + "），没能确认适配器是否生效；不作废，接着往后评")
        else:
            print("自检：适配器的输出在 6 道题里有 " + str(changed) + " 道和基座不同（门槛 " + str(minimum) + "）")
            if changed < minimum:                      # 自检没过：和真的一样不写报告和 summary
                print("适配器在这几道题上的输出和基座完全一样，适配器在 vLLM 里没有生效")
                sys.exit(4)
    prefix = out / (label + "-" + stem)
    write(str(prefix) + "-greedy.json", json.dumps({"results": [
        dict(task_id=task, strict_pass=passed) for task, passed in rows]}))
    summary = {"label": label, "model": model, "split": value("--split"), "tag": stem, "sanity": sanity,
               "greedy": counts(rows), "sampled": None}
    if samples:
        summary["sampled"] = {"samples_per_task": samples, "attempts": samples * len(rows), "passed_attempts": 1,
                              "pass_at_1": 1 / (samples * len(rows)) if rows else 0, "any_pass_rate": 0.5}
        write(str(prefix) + "-sampled-k" + str(samples) + ".json", json.dumps({"results": []}))
    write(str(prefix) + "-summary.json", json.dumps(summary, ensure_ascii=False))
    print("结果：" + label + " " + stem + " 贪心 " + str(summary["greedy"]["pass"]) + "/" + str(len(rows)))
elif args[0].endswith("vllm_summary.py"):
    log("summary")
    if os.environ.get("STUB_SUMMARY_FAIL"):
        print("汇总出错了（假的）", file=sys.stderr)
        sys.exit(1)
    os.execv(sys.executable, [sys.executable, *args])            # 真的 vllm_summary.py 读上面写出的产物
elif args[0].endswith("train_sft.py"):
    log("train")
    if "--dry-run" in args:
        if os.environ.get("STUB_TRAIN_FAIL") == "dry":
            sys.exit("训练数据没通过检查（假的）")
        print("Dry run 完成")
    elif os.environ.get("STUB_TRAIN_FAIL") == "real":
        sys.exit("训练出错（假的）")
    else:
        write(Path(value("--output")) / "training-manifest.json", '{"status": "complete"}')
        print("训练完成")
elif args[0].endswith("select_checkpoint.py"):
    log("select")
    if os.environ.get("STUB_SELECT_FAIL"):
        sys.exit("选型出错（假的）")
    out = Path(value("--output"))
    out.mkdir(parents=True, exist_ok=True)
    if value("--base-report"):
        shutil.copyfile(value("--base-report"), out / "epoch-000-dev.json")
    else:
        write(out / "epoch-000-dev.json", '{"adapter": null}')
    write(out / "epoch-001-dev.json", '{"adapter": "epoch-1"}')
    write(out / "selection.json", json.dumps({
        "selected_epoch": 1, "base_pass_at_1": 0.439, "candidates": [
            {"epoch": 0, "pass_at_1": 0.439, "improved_tasks": 0, "regressed_tasks": 0},
            {"epoch": 1, "pass_at_1": 0.5, "improved_tasks": 5, "regressed_tasks": 1}]}))
elif args[0].endswith("tier_report.py"):
    log("tiers")
    print("结果：（假的分层表）")
else:
    os.execv(sys.executable, [sys.executable, *args])            # 内嵌的 Python 片段：真解释器从 stdin 读
'''

VLLM = r'''#!/usr/bin/env bash
# 假的 vllm serve：把启动参数和自己的 PID 记进文件。FAKE_VLLM_MODE=ready（默认）报告启动完成后常驻，
# =oom 像显存不足那样失败，=vanish 什么都不打印就退出
printf '%s\n' "$*" >> "$FAKE_VLLM_LOG"
echo $$ >> "$FAKE_VLLM_PID_FILE"
case "${FAKE_VLLM_MODE:-ready}" in
  oom) echo "ERROR [core.py] torch.OutOfMemoryError: CUDA out of memory."
       echo "RuntimeError: Engine core initialization failed. See root cause above."
       exit 1 ;;
  vanish) exit 1 ;;
esac
echo "INFO:     Application startup complete."
while true; do sleep 1; done    # 常驻；命令行里一直带着 "vllm serve"，关服务时按这个名字找得到它
'''

NVIDIA_SMI = r'''#!/usr/bin/env bash
# 假的 nvidia-smi：已用显存取自 FAKE_GPU_USED（MiB），没设就是查不到；
# 设了 FAKE_GPU_FREE_AFTER=N 就先报 N 次 FAKE_GPU_USED，之后降回 500
count="${FAKE_SMI_COUNT:-/dev/null}"
if [[ -n "${FAKE_GPU_FREE_AFTER:-}" ]]; then
  echo $(( $(cat "$count" 2>/dev/null || echo 0) + 1 )) > "$count"
fi
used="${FAKE_GPU_USED:-}"
if [[ -n "${FAKE_GPU_FREE_AFTER:-}" && "$(cat "$count" 2>/dev/null || echo 0)" -gt "$FAKE_GPU_FREE_AFTER" ]]; then
  used=500
fi
case "$*" in
  *--query-gpu=memory.used*) echo "$used" ;;
  *--query-compute-apps*) [[ "$used" =~ ^[0-9]+$ ]] && (( used > 2000 )) && echo "12345, 18000 MiB" ;;
esac
exit 0
'''

OFFLINE_SMI = r'''#!/usr/bin/env bash
# nvidia-smi 自己出错（驱动没起来）：只往 stderr 报错
echo "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver." >&2
exit 9
'''

PKILL = r'''#!/usr/bin/env bash
# 假的 pkill -f：照正则匹配整条命令行，但只对命令行里带着这次演练目录（STAGE）的进程动手。
# 用真的 pkill 会连这台机器上别的 vllm serve（包括真在跑的那个）一起关掉
[[ "$1" == -f && $# == 2 ]] || { echo "假的 pkill 只会 pkill -f 模式" >&2; exit 2; }
status=1
for entry in /proc/[0-9]*; do
  [[ -r "$entry/cmdline" ]] || continue
  mapfile -d '' -t argv < "$entry/cmdline"    # 命令行里各参数之间是 NUL，像真的 pkill 那样拼成一整行再匹配
  line="${argv[*]}"
  [[ "$line" == *"$STAGE"* && "$line" =~ $2 ]] || continue
  kill "${entry#/proc/}" 2>/dev/null && status=0
done
exit $status
'''

SLEEP = r'''#!/usr/bin/env bash
# 演练用的 sleep：rft_start_vllm 启动前固定等的那 3 秒（等残留的旧服务退出）不必真等，其余照常
if [[ "$*" == 3 ]]; then exec @REAL@ 0.1; fi
exec @REAL@ "$@"
'''

NO_SLEEP = r'''#!/usr/bin/env bash
# 不真的等：演练"等显存释放"的两分钟循环时放在 PATH 最前面
exit 0
'''

HARNESS = r'''#!/usr/bin/env bash
# 测试用：设好变量，按名字调用 scripts/rft_lib.sh 里的一个函数，最后一行打印它的返回值
cd "$(dirname "$0")/.."
source scripts/rft_lib.sh
MODEL="${MODEL:-org/model}" REVISION="${REVISION:-rev-123}" RUN="${RUN:-data/run}"
TASKS="${TASKS:-data/frozen-ladder.jsonl}" EPOCHS="${EPOCHS:-1}" GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_LENGTH="${MAX_LENGTH:-2048}" GRADE_WORKERS="${GRADE_WORKERS:-4}" VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_BIN="${VLLM_BIN:-$HOME/vllm-qwen/bin/vllm}" BASE_REPORT="${BASE_REPORT:-}" TOTAL="${TOTAL:-7}"
STARTED=$SECONDS
run() { local rc=0; "$@" || rc=$?; echo "rc=$rc"; }
mode="$1"; shift
case "$mode" in
  progress) SECONDS=125 STARTED=65 run rft_progress "$@" ;;
  gpu-used) run rft_gpu_used ;;
  check-gpu) run rft_check_gpu_free ;;
  wait-gpu) run rft_wait_gpu_free ;;
  stop-vllm) run rft_stop_vllm ;;
  start-stop-vllm)
    rc=0; rft_start_vllm "$@" || rc=$?
    pid="${RFT_VLLM_PID:-}"
    rft_stop_vllm
    alive=0
    for i in 1 2 3 4 5 6 7 8 9 10; do
      [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null || break
      alive=1; sleep 0.2
    done
    echo "rc=$rc alive_after_stop=$alive" ;;
  training-complete) run rft_training_complete "$@" ;;
  arm) run rft_arm "$@" ;;
  arms) rft_arm "${@:1:6}"; shift 6; run rft_arm "$@" ;;
  eval) run rft_eval_vllm "$@" ;;
  eval-probe) run rft_eval_probe "$@" ;;
  eval-done) run rft_eval_done "$@" ;;
  eval-unconfirmed) run rft_eval_unconfirmed "$@" ;;
  sampling-complete) run rft_sampling_complete "$@" ;;
  vllm-running) run rft_vllm_running ;;
  *) echo "不认识的函数：$mode" >&2; exit 2 ;;
esac
'''

STEP = re.compile(r"^===== (.+?) =====$", re.M)
STEP_ONE = re.compile(r"python - \"\$RUN\" \"\$MODEL\" \"\$REVISION\" \"\$@\" <<'PY'\n(.*?)\nPY\n", re.S)


def install(home, folder, name, text):
    program = home / folder / name
    program.parent.mkdir(parents=True, exist_ok=True)
    program.write_text(text, encoding="utf-8")
    program.chmod(0o755)


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def select(repo, group, adapter, model="org/model", revision="rev-123"):
    """写一份选型结果。adapter 是 None 表示 dev 上保留了基座；model 是 None 表示选型结果里没记基座。"""
    selection = {"selected_epoch": 0 if adapter is None else 1, "selected_adapter": adapter, "base_pass_at_1": 0.439,
                 "candidates": [{"epoch": 0, "pass_at_1": 0.439, "improved_tasks": 0, "regressed_tasks": 0}]}
    if model:
        selection["configuration"] = {"model": model, "revision": revision}
    write(repo / RUN / "results" / group / "selection.json", selection)


def selected(repo, *groups, model="org/model", revision="rev-123"):
    """几个组都选了 epoch-001 的适配器，训练数据里各有 TRAINED 那么多道探针题（第一组的路径故意写成相对的）。"""
    for index, group in enumerate(groups):
        adapter = repo / RUN / "checkpoints" / group / "epoch-001/adapter"
        write(adapter / "adapter_config.json", "{}\n")
        tasks = "".join(json.dumps({"task_id": f"h3_t{number}"}) + "\n" for number in range(1, TRAINED[group] + 1))
        write(repo / RUN / "sft" / group / "train.jsonl", tasks)
        select(repo, group, f"{RUN}/checkpoints/{group}/epoch-001/adapter" if index == 0 else str(adapter),
               model=model, revision=revision)


def kept_base(repo, *groups):
    for group in groups:
        select(repo, group, None)


def write_frozen_split(repo):
    """4 道 train 题就是探针题，dev 里三层各一道。"""
    rows = [{"task_id": f"h3_t{index}", "split": "train"} for index in range(1, 5)]
    rows += [{"task_id": task, "split": "dev"} for task in ("h3_d1", "h3v_d1", "v2_d1")]
    write(repo / "data/frozen-ladder.jsonl", "".join(json.dumps(row) + "\n" for row in rows))
    write(repo / PROBE, "".join(f"h3_t{index}\n" for index in range(1, 5)))


def processes(stage):
    """这次演练里假 vllm 记下的 PID。"""
    pid_file = stage / "vllm.pid"
    return [int(pid) for pid in pid_file.read_text(encoding="utf-8").split()] if pid_file.exists() else []


def running(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def alive(stage):
    """这次演练启动的假 vllm 还有没有活着的。"""
    return any(running(pid) for pid in processes(stage))


def eventually(condition, seconds=5.0):
    """等 condition 成立（假的 vllm 起来、被关掉都要一点时间）；到时间还不成立就返回它最后的值。"""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


@pytest.fixture
def stage(tmp_path):
    """一份临时的仓库（三个脚本和调用库函数的小程序）、假的 python、vllm、nvidia-smi、pkill、sleep。"""
    home = tmp_path / "home"
    repo = home / "repo"
    (repo / "scripts").mkdir(parents=True)
    for name in ("rft_lib.sh", "rft_vllm_eval.sh", "vllm_summary.py"):
        shutil.copy(ROOT / "scripts" / name, repo / "scripts" / name)
    install(home, "repo/scripts", "lib-harness.sh", HARNESS)
    install(home, "bin", "python", PYTHON)
    install(home, "bin", "nvidia-smi", NVIDIA_SMI)
    install(home, "offline", "nvidia-smi", OFFLINE_SMI)
    install(home, "bin", "pkill", PKILL)
    install(home, "bin", "sleep", SLEEP.replace("@REAL@", shutil.which("sleep")))
    install(home, "nosleep", "sleep", NO_SLEEP)
    install(home, "vllm-qwen/bin", "vllm", VLLM)
    write_frozen_split(repo)
    yield home
    for pid in processes(home):                     # 断言失败时也不把假的 vllm 留在机器上
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)


def environment(stage, first=(), **extra):
    return {**os.environ, "HOME": str(stage), "STAGE": str(stage), "STUB_LOG": str(stage / "calls.log"),
            "FAKE_VLLM_LOG": str(stage / "vllm.log"), "FAKE_VLLM_PID_FILE": str(stage / "vllm.pid"),
            "FAKE_SMI_COUNT": str(stage / "smi.count"), "RFT_POLL_SECONDS": "1", **extra,
            "PATH": os.pathsep.join([*(str(item) for item in first), str(stage / "bin"), os.environ["PATH"]])}


def bash(stage, *command, **extra):
    for name in ("calls.log", "vllm.log"):          # 每次只看这一次的调用
        (stage / name).unlink(missing_ok=True)
    return subprocess.run(["bash", *command], cwd=stage / "repo", env=environment(stage, **extra),
                          capture_output=True, text=True, timeout=180)


def calls_of(stage):
    """假 python 在刚才那次运行里记下的调用：[(脚本名, 参数)]。"""
    log = stage / "calls.log"
    if not log.exists():
        return []
    return [(parts[0], parts[1] if len(parts) > 1 else "")
            for parts in (line.split(" ", 1) for line in log.read_text(encoding="utf-8").splitlines())]


def run(stage, *groups, **extra):
    """跑一遍 rft_vllm_eval.sh（运行目录 data/rft-guided）。返回 (结果, 假 python 记下的调用)。"""
    result = bash(stage, "scripts/rft_vllm_eval.sh", RUN, *groups, **extra)
    return result, calls_of(stage)


def harness(stage, mode, *arguments, **extra):
    """调用库里的一个函数，返回整个结果（最后一行是 rc=…）。"""
    return bash(stage, "scripts/lib-harness.sh", mode, *arguments, **extra)


def last(result):
    return result.stdout.splitlines()[-1] if result.stdout.splitlines() else ""


def starts(stage):
    """假 vllm 每次被启动时收到的参数。"""
    log = stage / "vllm.log"
    return log.read_text(encoding="utf-8").splitlines() if log.exists() else []


def flag(args, name):
    found = re.search(r"--" + name + r" (\S+)", args)
    return found.group(1) if found else None


def evaluations(calls):
    """每次 eval_vllm.py 调用的要点：(标签, stem, 采样次数, 自检用的训练数据)。"""
    rows = []
    for name, args in calls:
        if name == "eval":
            rows.append((flag(args, "label"), flag(args, "tag") or flag(args, "split"), flag(args, "samples"),
                         flag(args, "sanity-tasks-file")))
    return rows


def vllm_output(repo):
    return repo / RUN / "vllm"


def steps(result):
    return STEP.findall(result.stdout)


# ---- rft_vllm_eval.sh 端到端 ----


@pytest.mark.parametrize("name", ["rft_lib.sh", "rft_vllm_eval.sh"])
def test_shell_scripts_parse(name):
    assert subprocess.run(["bash", "-n", str(ROOT / "scripts" / name)], capture_output=True).returncode == 0


def test_a_fresh_run_evaluates_the_base_then_each_adapter_with_the_probe_first(stage):
    repo = stage / "repo"
    selected(repo, *GROUPS)
    switches = dict(INCLUDE_BASE="1", PROBE_IDS_FILE=PROBE, EXPECT_REPRODUCE="D-prefix D-skill")
    result, calls = run(stage, *GROUPS, **switches, STUB_SANITY="D-skill=0")
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-2000:]
    assert steps(result) == [
        "1/8 读取各组的选型结果", "2/8 确定要评的对象", "3/8 检查显卡", "4/8 启动 vLLM", "5/8 评基座",
        "6/8 D-prefix：先探针后 dev", "6/8 D-skill：先探针后 dev", "6/8 F-control：先探针后 dev",
        "7/8 关闭 vLLM", "8/8 汇总", "全部结束"]

    # 服务带着三组的 LoRA 启动：名字是组名，路径是绝对的（D-prefix 在选型结果里写的是相对路径）
    adapters = repo / RUN / "checkpoints"
    assert starts(stage) == [
        "serve org/model --revision rev-123 --language-model-only --max-model-len 8192 --port 8000 --enable-lora "
        "--max-lora-rank 64 --lora-modules "
        + " ".join(f"{group}={adapters}/{group}/epoch-001/adapter" for group in GROUPS)]

    # 基座先评；每组先探针后 dev；EXPECT_REPRODUCE 里的两组带自检用的训练数据；自检没过的那组不评 dev
    assert evaluations(calls) == [
        ("base", "dev", "4", None), ("base", "probe", "0", None),
        ("D-prefix", "probe", "0", f"{RUN}/sft/D-prefix/train.jsonl"), ("D-prefix", "dev", "4", None),
        ("D-skill", "probe", "0", f"{RUN}/sft/D-skill/train.jsonl"),
        ("F-control", "probe", "0", None), ("F-control", "dev", "4", None)]
    assert [name for name, _ in calls] == ["eval"] * 7 + ["summary"]
    for _, args in calls[:-1]:
        assert "--tasks-file data/frozen-ladder.jsonl --vllm-url http://127.0.0.1:8000 " in args
        assert "--grade-workers 4" in args and "--label " in args
    for _, args in [call for call in calls if "--tag probe" in call[1]]:
        assert f"--split train --ids-file {PROBE} --tag probe --samples 0" in args
    assert calls[-1] == ("summary", f"scripts/vllm_summary.py --run {RUN} --arms {' '.join(GROUPS)} "
                                   f"--probe-ids-file {PROBE}")
    void = "D-skill：适配器在这几道题上的输出和基座完全一样，适配器在 vLLM 里没有生效，这组的 vLLM 评测作废"
    assert void in result.stdout
    assert re.search(r"复现 \d+/\d+", result.stdout) is None         # "复现 x/N"这个数字不再出现
    assert f"结果：vLLM 评测做完了；作废的组：D-skill（适配器的输出和基座完全一样）；报告在 {RUN}/vllm" in result.stdout

    # 汇总是真的 vllm_summary.py 读上面写出的产物：训练数据里的 2 道和其余 2 道分开数，自检结果跟在后面
    assert "── vLLM 评测（bf16 权重，辅助指标；主结果是上面 HF 贪心那张表） ──" in result.stdout
    # 自检没过的组：和基座比的结果写在这一行的后面——eval_vllm.py 只留下 -sanity.json，汇总也读得到。
    # 训练过的题严格通过几道只作参考，跟在后面
    void_with_sanity = (r"^D-skill\s+（评测作废或没做）；自检：适配器的输出在 6 道题里有 0 道和基座不同（门槛 1），"
                        r"适配器在 vLLM 里没有生效；训练过的题严格通过 0/6（仅供参考）$")
    assert re.search(void_with_sanity, result.stdout, re.M)
    split = r"(\d+)/2（基座 \d+/2）"
    assert re.search(rf"^D-prefix\s+训练过的题 {split}、没训练过的同族题 {split}；"
                     r"自检：适配器的输出在 6 道题里有 4 道和基座不同（门槛 1）；"
                     r"训练过的题严格通过 3/6（仅供参考）$", result.stdout, re.M)
    assert re.search(rf"^F-control\s+训练过的题 {split}、没训练过的同族题 {split}$", result.stdout, re.M)
    for name in ("base/base-dev-summary.json", "base/base-probe-greedy.json", "D-prefix/D-prefix-dev-sampled-k4.json",
                 "D-skill/D-skill-probe-sanity.json", "F-control/F-control-probe-summary.json"):
        assert (vllm_output(repo) / name).is_file(), name
    assert not list((vllm_output(repo) / "D-skill").glob("*-dev-*"))         # 作废的组没有评 dev
    assert not alive(stage)                                               # 用完把服务关掉


def test_a_group_whose_sanity_check_cannot_compare_with_the_base_is_evaluated_and_said_to_be_unconfirmed(stage):
    """自检拿不到基座名（或者探针题里没有这一组训练过的题）时没法比较：这不是"输出和基座完全一样"，不作废，
    照常评 dev；结束时说明这一组没能确认适配器是否生效，汇总表里这一行也注明。"""
    selected(stage / "repo", *GROUPS)
    reason = "服务的模型列表里分不出 D-skill 挂在哪个基座上，拿不到基座名"
    result, calls = run(stage, *GROUPS, PROBE_IDS_FILE=PROBE, EXPECT_REPRODUCE="D-prefix D-skill",
                        STUB_UNCOMPARED="D-skill")
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-1000:]
    evaluated = [(label, stem) for label, stem, *_ in evaluations(calls)]
    assert evaluated == [
        ("D-prefix", "probe"), ("D-prefix", "dev"), ("D-skill", "probe"), ("D-skill", "dev"),
        ("F-control", "probe"), ("F-control", "dev")]                     # 没能比较的组照常评完
    assert f"D-skill：没能确认适配器是否生效（{reason}），不作废，照常评" in result.stdout
    assert "评测作废" not in result.stdout and "没有生效" not in result.stdout   # "不作废"里也有"作废"两个字
    assert (f"结果：vLLM 评测做完了；没能确认适配器是否生效的组：D-skill（{reason}）；"
            f"报告在 {RUN}/vllm") in result.stdout
    # 汇总表里这一行照常有数，后面注明没能比较、没能确认（读的是 eval_vllm.py 写的 -sanity.json）
    assert re.search(rf"^D-skill\s+训练过的题 \d+/3、没训练过的同族题 \d+/1；"
                      rf"自检：没能和基座比较（{reason}），没能确认适配器是否生效$", result.stdout, re.M)
    assert steps(result)[-1] == "全部结束" and not alive(stage)


def test_a_void_group_is_checked_again_on_a_rerun_and_everything_else_is_skipped(stage):
    selected(stage / "repo", *GROUPS)
    switches = dict(INCLUDE_BASE="1", PROBE_IDS_FILE=PROBE, EXPECT_REPRODUCE="D-prefix D-skill")
    assert run(stage, *GROUPS, **switches, STUB_SANITY="D-skill=0")[0].returncode == 0
    result, calls = run(stage, *GROUPS, **switches, STUB_SANITY="D-skill=0")
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-1000:]
    # 自检查的是服务现在的状态，没有报告留下来：重跑时只重新自检作废的那一组，其余都跳过
    assert evaluations(calls) == [("D-skill", "probe", "0", f"{RUN}/sft/D-skill/train.jsonl")]
    assert [name for name, _ in calls] == ["eval", "summary"] and len(starts(stage)) == 1
    assert result.stdout.count("已经评过，跳过") == 3                          # 基座、D-prefix、F-control
    assert "D-skill：适配器在这几道题上的输出和基座完全一样，适配器在 vLLM 里没有生效，这组的 vLLM 评测作废" \
        in result.stdout
    assert steps(result)[-1] == "全部结束" and not alive(stage)


def test_rerunning_skips_what_is_done_and_only_redoes_the_missing_part(stage):
    repo = stage / "repo"
    selected(repo, *GROUPS)
    switches = dict(PROBE_IDS_FILE=PROBE, EXPECT_REPRODUCE="D-prefix D-skill")
    first, calls = run(stage, *GROUPS, **switches)
    assert first.returncode == 0, first.stdout[-2000:] + first.stderr[-1000:]
    assert evaluations(calls) == [
        ("D-prefix", "probe", "0", f"{RUN}/sft/D-prefix/train.jsonl"), ("D-prefix", "dev", "4", None),
        ("D-skill", "probe", "0", f"{RUN}/sft/D-skill/train.jsonl"), ("D-skill", "dev", "4", None),
        ("F-control", "probe", "0", None), ("F-control", "dev", "4", None)]
    assert len(starts(stage)) == 1 and "===== 5/8 评基座：INCLUDE_BASE 不是 1，跳过 =====" in first.stdout

    again, calls = run(stage, *GROUPS, **switches)                            # 全都评过了：服务都不用开
    assert again.returncode == 0, again.stderr[-1000:]
    assert calls == [("summary", f"scripts/vllm_summary.py --run {RUN} --arms {' '.join(GROUPS)} "
                                 f"--probe-ids-file {PROBE}")]
    assert starts(stage) == [] and "── 怎么读 ──" in again.stdout
    assert "===== 3/8 检查显卡：都评过了，不用启动 vLLM，跳过 =====" in again.stdout
    assert "===== 7/8 关闭 vLLM：这次没有启动，跳过 =====" in again.stdout
    assert again.stdout.count("已经评过，跳过") == 3 and steps(again)[-1] == "全部结束"

    (vllm_output(repo) / "F-control/F-control-dev-summary.json").unlink()     # dev 评到一半断了
    resumed, calls = run(stage, *GROUPS, **switches)
    assert resumed.returncode == 0, resumed.stderr[-1000:]
    assert evaluations(calls) == [("F-control", "dev", "4", None)]            # 只补它；服务照旧带着三组的 LoRA
    assert len(starts(stage)) == 1 and starts(stage)[0].count("epoch-001/adapter") == 3
    assert resumed.stdout.count("已经评过，跳过") == 2 and steps(resumed)[-1] == "全部结束"
    assert not alive(stage)


def test_without_adapters_and_without_the_base_there_is_nothing_to_do(stage):
    kept_base(stage / "repo", "D-prefix", "F-control")
    result, calls = run(stage, "D-prefix", "F-control")
    assert result.returncode == 0, result.stderr[-1000:]
    assert calls == [] and starts(stage) == []
    assert "D-prefix：dev 上保留了基座（没有适配器），不用评" in result.stdout
    assert "各组都没有适配器，也没有要评基座：没有可做的 vLLM 评测。" in result.stdout
    assert "结果：没有适配器，vLLM 评测跳过" in result.stdout
    assert steps(result) == ["1/8 读取各组的选型结果", "2/8 确定要评的对象", "全部结束"]


def test_without_adapters_the_base_alone_is_evaluated_when_asked(stage):
    repo = stage / "repo"
    kept_base(repo, "D-prefix")
    result, calls = run(stage, "D-prefix", INCLUDE_BASE="1")
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-1000:]
    assert evaluations(calls) == [("base", "dev", "4", None)]
    assert starts(stage) == ["serve org/model --revision rev-123 --language-model-only --max-model-len 8192 "
                             "--port 8000"]                                   # 没有适配器：不带 LoRA 的参数
    assert "要评的模型：基座" in result.stdout and (vllm_output(repo) / "base/base-dev-summary.json").is_file()
    assert steps(result)[-1] == "全部结束" and not alive(stage)


@pytest.mark.parametrize("mode, reason", [("oom", "引擎初始化失败"), ("vanish", "进程在就绪之前就退出了")])
def test_a_server_that_cannot_start_stops_before_evaluating(stage, mode, reason):
    selected(stage / "repo", *GROUPS)
    result, calls = run(stage, *GROUPS, INCLUDE_BASE="1", FAKE_VLLM_MODE=mode)
    assert result.returncode == 1, result.stdout[-1000:] + result.stderr[-1000:]
    assert calls == [] and starts(stage) and not vllm_output(stage / "repo").exists()
    assert f"vLLM 没起来（{reason}），日志最后 15 行：" in result.stdout
    assert "===== 5/8" not in result.stdout and "===== 全部结束 =====" not in result.stdout
    assert not alive(stage)
    if mode == "oom":                                   # 日志的最后几行原样给用户看
        assert "torch.OutOfMemoryError: CUDA out of memory." in result.stdout
        assert "Engine core initialization failed" in result.stdout


def test_a_busy_gpu_stops_the_run_before_starting_the_server(stage):
    selected(stage / "repo", *GROUPS)
    result, calls = run(stage, *GROUPS, FAKE_GPU_USED="18000")
    assert result.returncode == 1, result.stderr[-1000:]
    assert calls == [] and starts(stage) == []
    assert "显卡被占用（已用 18000 MiB）" in result.stdout and "12345, 18000 MiB" in result.stdout
    assert "===== 4/8" not in result.stdout and "===== 全部结束 =====" not in result.stdout


def test_a_group_that_does_not_finish_does_not_stop_the_others_and_a_rerun_finishes_it(stage):
    repo = stage / "repo"
    selected(repo, *GROUPS)
    result, calls = run(stage, *GROUPS, INCLUDE_BASE="1", STUB_EVAL_FAIL="F-control/dev=3")
    assert result.returncode == 1, result.stdout[-1000:] + result.stderr[-1000:]
    assert "F-control：vLLM 评测没完成（退出码 3），重新运行同一条命令会只补没完成的" in result.stdout
    assert [(label, stem) for label, stem, _, _ in evaluations(calls)] == [
        ("base", "dev"), ("D-prefix", "dev"), ("D-skill", "dev"), ("F-control", "dev")]
    assert "── 怎么读 ──" in result.stdout                                     # 没评完也把能汇总的汇总出来
    assert "结果：vLLM 评测没有全部完成，重新运行同一条命令会只补没完成的" in result.stdout
    assert "===== 8/8 汇总 =====" in result.stdout and "===== 全部结束 =====" not in result.stdout
    assert not alive(stage)                                                   # 没评完也不把服务留在显卡上
    assert not (vllm_output(repo) / "F-control/F-control-dev-summary.json").exists()

    result, calls = run(stage, *GROUPS, INCLUDE_BASE="1")
    assert result.returncode == 0, result.stderr[-1000:]
    assert evaluations(calls) == [("F-control", "dev", "4", None)]
    assert result.stdout.count("已经评过，跳过") == 3 and "===== 全部结束 =====" in result.stdout


def test_a_probe_that_does_not_finish_does_not_void_the_group_and_a_broken_summary_is_only_noted(stage):
    selected(stage / "repo", *GROUPS)
    result, calls = run(stage, "D-prefix", PROBE_IDS_FILE=PROBE, STUB_EVAL_FAIL="D-prefix/probe=3",
                        STUB_SUMMARY_FAIL="1")
    assert result.returncode == 1, result.stdout[-1000:] + result.stderr[-1000:]
    assert "D-prefix 的探针：vLLM 评测没完成（退出码 3），重新运行同一条命令会只补没完成的" in result.stdout
    assert [(label, stem) for label, stem, _, _ in evaluations(calls)] == [("D-prefix", "probe"), ("D-prefix", "dev")]
    assert "作废" not in result.stdout                                          # 退出码不是 4：不算作废，dev 照评
    assert "汇总没打印出来（把上面的报错贴给我）；已经评完的报告都在 " + RUN + "/vllm" in result.stdout
    assert "===== 8/8 汇总 =====" in result.stdout and "===== 全部结束 =====" not in result.stdout
    assert not alive(stage)


def test_a_group_to_be_checked_that_has_no_training_data_is_evaluated_and_said_to_be_unconfirmed(stage):
    """列在 EXPECT_REPRODUCE 里、却找不到训练数据：挑不出训练过的题，没法比较。作废只有一种原因（输出和基座
    完全一样），所以不作废：不带自检照常评，结束时说明这一组没能确认适配器是否生效。"""
    repo = stage / "repo"
    selected(repo, *GROUPS)
    (repo / RUN / "sft/D-skill/train.jsonl").unlink()
    result, calls = run(stage, *GROUPS, PROBE_IDS_FILE=PROBE, EXPECT_REPRODUCE="D-prefix D-skill")
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-1000:]
    message = (f"D-skill：找不到自检用的训练数据 {RUN}/sft/D-skill/train.jsonl，"
               "没法比较适配器和基座的输出；不作废，照常评")
    assert message in result.stdout
    assert "评测作废" not in result.stdout
    assert ("结果：vLLM 评测做完了；没能确认适配器是否生效的组：D-skill（找不到自检用的训练数据）；"
            f"报告在 {RUN}/vllm") in result.stdout
    assert [(label, stem) for label, stem, *_ in evaluations(calls)] == [
        ("D-prefix", "probe"), ("D-prefix", "dev"), ("D-skill", "probe"), ("D-skill", "dev"),
        ("F-control", "probe"), ("F-control", "dev")]
    assert "--sanity-tasks-file" not in calls[2][1]           # D-skill 的探针不带自检，其余照常
    assert "===== 6/8 D-skill：先探针后 dev =====" in result.stdout and "===== 全部结束 =====" in result.stdout


def test_a_group_whose_adapter_is_gone_is_skipped_and_the_others_are_still_evaluated(stage):
    repo = stage / "repo"
    selected(repo, *GROUPS)
    shutil.rmtree(repo / RUN / "checkpoints/D-skill")
    result, calls = run(stage, *GROUPS)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-1000:]
    adapter = f"{stage}/repo/{RUN}/checkpoints/D-skill/epoch-001/adapter"
    assert f"D-skill：找不到适配器 {adapter}（里面没有 adapter_config.json），这组不评" in result.stdout
    assert {label for label, *_ in evaluations(calls)} == {"D-prefix", "F-control"}
    start, = starts(stage)
    assert "D-prefix=" in start and "F-control=" in start and "D-skill=" not in start
    assert steps(result)[-1] == "全部结束"


def test_a_misspelled_group_stops_before_touching_the_gpu(stage):
    repo = stage / "repo"
    selected(repo, "D-prefix")
    result, calls = run(stage, "D-prefix", "no-such-group")
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-1000:]   # 读不出的组跳过，别的组照评
    assert "no-such-group：读不出选型结果" in result.stderr
    assert {label for label, *_ in evaluations(calls)} == {"D-prefix"}
    result, calls = run(stage, "no-such-group")                              # 一个组的选型结果都读不到
    assert result.returncode == 1 and calls == [] and starts(stage) == []
    assert "这些组一个都还没有选型结果：no-such-group" in result.stderr
    assert "读各组的选型结果时出错，把上面的报错贴给我" in result.stdout and "===== 2/8" not in result.stdout
    result, _ = run(stage, "D.prefix")                                        # 组名要是合法的 LoRA 名字
    assert result.returncode == 1 and "组名只能由字母、数字、- 和 _ 组成：D.prefix" in result.stderr
    result, _ = run(stage)                                                    # 一个组名都不给
    assert result.returncode == 1 and "至少给一个组名" in result.stderr
    result, _ = run(stage, "D-prefix", PROBE_IDS_FILE="data/nowhere.txt")
    assert result.returncode == 1 and "找不到探针题号文件 data/nowhere.txt" in result.stdout
    result, _ = run(stage, "D-prefix", TASKS_FILE="data/nowhere.jsonl")
    assert result.returncode == 1 and "缺少任务文件 data/nowhere.jsonl" in result.stdout
    assert starts(stage) == [] and "===== 2/8" not in result.stdout


def test_the_base_recorded_with_the_adapters_is_what_vllm_serves(stage):
    repo = stage / "repo"
    selected(repo, "D-prefix", model="org/other", revision="sha-abc")
    select(repo, "F-control", None)
    result, _ = run(stage, "D-prefix", "F-control", INCLUDE_BASE="1")
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-1000:]
    assert starts(stage)[0].startswith("serve org/other --revision sha-abc --language-model-only")
    assert ("各组的适配器是在 org/other @ sha-abc 上训练的，vLLM 按这个基座启动（环境里的是 org/model @ rev-123）"
            in result.stdout)

    selected(repo, "D-prefix", model="org/one", revision="sha-1")
    selected(repo, "F-control", model="org/two", revision="sha-2")
    result, calls = run(stage, "D-prefix", "F-control")                      # 两个基座，一个服务装不下
    assert result.returncode == 1 and calls == [] and starts(stage) == []
    assert "org/one @ sha-1：D-prefix" in result.stderr and "org/two @ sha-2：F-control" in result.stderr

    shutil.rmtree(vllm_output(repo))
    select(repo, "D-prefix", str(repo / RUN / "checkpoints/D-prefix/epoch-001/adapter"), model=None)    # 没记基座
    result, _ = run(stage, "D-prefix")                                       # 用默认的模型和固定版本
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-1000:]
    assert starts(stage)[0].startswith("serve org/model --revision rev-123") and "按这个基座启动" not in result.stdout

    shutil.rmtree(vllm_output(repo))
    result, _ = run(stage, "D-prefix", MODEL="org/explicit", MODEL_REVISION="sha-x")    # 环境里给的就照用
    assert result.returncode == 0 and starts(stage)[0].startswith("serve org/explicit --revision sha-x")


def test_switches_that_do_not_go_together_are_pointed_out(stage):
    repo = stage / "repo"
    selected(repo, "D-prefix")
    result, calls = run(stage, "D-prefix", EXPECT_REPRODUCE="D-prefix")
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-1000:]
    assert "EXPECT_REPRODUCE 里列了组，但没有给 PROBE_IDS_FILE：自检只在评探针时做，这次不会自检" in result.stdout
    assert evaluations(calls) == [("D-prefix", "dev", "4", None)]            # 照常评，只是没法自检

    shutil.rmtree(vllm_output(repo))
    result, calls = run(stage, "D-prefix", PROBE_IDS_FILE=PROBE, EVAL_SAMPLES="0")
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-1000:]
    assert "（还没有基座的探针报告：汇总里各组的括号不写基座的通过数；要写的话加 INCLUDE_BASE=1）" in result.stdout
    assert evaluations(calls) == [("D-prefix", "probe", "0", None), ("D-prefix", "dev", "0", None)]
    assert not list((vllm_output(repo) / "D-prefix").glob("*sampled*"))      # EVAL_SAMPLES=0：只做贪心


def test_an_unknown_revision_is_not_replaced_by_a_group_name(stage):
    """基座的固定版本查不到（空）而各组都保留了基座：空的版本号也占一个字段，不能被吞掉、让后面的组名顶上来
    当成版本号交给 vllm serve。"""
    kept_base(stage / "repo", *GROUPS)
    result, calls = run(stage, *GROUPS, INCLUDE_BASE="1", STUB_REVISION="")
    assert result.returncode == 1 and calls == [] and starts(stage) == []
    assert "查不到默认模型或固定版本（org/model / ），把这一行贴给我" in result.stdout
    assert "===== 2/8" not in result.stdout and "===== 全部结束 =====" not in result.stdout


def test_a_server_that_quits_in_the_middle_is_restarted_and_the_interrupted_part_is_finished(stage):
    """评 D-prefix 的 dev 时 vLLM 退出：对着一个不在的服务重试没有用。脚本把服务重新开一次，把这一组没评完的
    部分补上，后面的组照常评——夜里不必等有人来重新运行。"""
    selected(stage / "repo", *GROUPS)
    switches = dict(PROBE_IDS_FILE=PROBE, EXPECT_REPRODUCE="D-prefix D-skill")
    result, calls = run(stage, *GROUPS, **switches, STUB_KILL_SERVER="D-prefix/dev=1")
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-2000:]
    assert "D-prefix：vLLM 评测没完成（退出码 3），重新运行同一条命令会只补没完成的" in result.stdout
    assert "D-prefix：没评完的时候 vLLM 已经不在了" in result.stdout
    assert "vLLM 已经不在了（多半是中途退出），日志最后 15 行：" in result.stdout
    assert "INFO:     Application startup complete." in result.stdout      # 服务日志的最后几行原样给用户看
    assert "重新启动 vLLM，接着评剩下的部分" in result.stdout
    assert "===== 6/8 D-prefix：vLLM 中途退出，重新启动后再评一次 =====" in result.stdout
    trained = {group: f"{RUN}/sft/{group}/train.jsonl" if group != "F-control" else None for group in GROUPS}
    assert evaluations(calls) == [
        ("D-prefix", "probe", "0", trained["D-prefix"]), ("D-prefix", "dev", "4", None),
        ("D-prefix", "dev", "4", None),                                 # 重启后只补这一组没评完的 dev
        ("D-skill", "probe", "0", trained["D-skill"]), ("D-skill", "dev", "4", None),
        ("F-control", "probe", "0", None), ("F-control", "dev", "4", None)]
    assert len(starts(stage)) == 2 and starts(stage)[0] == starts(stage)[1]   # 重启时带着同样的 LoRA
    assert f"结果：vLLM 评测做完了；报告在 {RUN}/vllm" in result.stdout
    assert steps(result).count("全部结束") == 1 and steps(result)[-1] == "全部结束"
    assert not alive(stage)


def test_a_server_that_quits_again_is_not_restarted_twice_and_the_rest_is_left_to_the_next_run(stage):
    selected(stage / "repo", *GROUPS)
    switches = dict(PROBE_IDS_FILE=PROBE, EXPECT_REPRODUCE="D-prefix D-skill")
    result, calls = run(stage, *GROUPS, **switches, STUB_KILL_SERVER="D-prefix/dev=2")
    assert result.returncode == 1, result.stdout[-2000:] + result.stderr[-1000:]
    assert len(starts(stage)) == 2                                       # 只重启一次
    assert result.stdout.count("D-prefix：vLLM 评测没完成（退出码 3）") == 2
    assert "vLLM 重新启动过一次，又不在了：不再重启" in result.stdout
    assert "D-skill：vLLM 不在了，没有评；重新运行同一条命令会只补没评完的" in result.stdout
    assert "F-control：vLLM 不在了，没有评；重新运行同一条命令会只补没评完的" in result.stdout
    assert [(label, stem) for label, stem, *_ in evaluations(calls)] == [
        ("D-prefix", "probe"), ("D-prefix", "dev"), ("D-prefix", "dev")]  # 后面两组没有对着不在的服务发请求
    assert "结果：vLLM 评测没有全部完成，重新运行同一条命令会只补没完成的" in result.stdout
    assert "===== 全部结束 =====" not in result.stdout and not alive(stage)
    result, calls = run(stage, *GROUPS, **switches)                      # 下一次运行把没评的补上
    assert result.returncode == 0, result.stderr[-1000:]
    assert [(label, stem) for label, stem, *_ in evaluations(calls)] == [
        ("D-prefix", "dev"), ("D-skill", "probe"), ("D-skill", "dev"), ("F-control", "probe"), ("F-control", "dev")]
    assert len(starts(stage)) == 1 and steps(result)[-1] == "全部结束"


def test_step_one_reads_the_selections_and_puts_only_the_last_line_on_stdout(tmp_path):
    """第 1 步内嵌的 Python 用真解释器跑：最后一行是 vLLM 该用的基座和每组的适配器，诊断只进 stderr。"""
    snippet, = STEP_ONE.findall((ROOT / "scripts/rft_vllm_eval.sh").read_text(encoding="utf-8"))
    run_dir = tmp_path / "run"
    write(run_dir / "results/A/selection.json", {"selected_adapter": "/abs/a", "configuration": {
        "model": "org/trained", "revision": "sha-trained"}})
    write(run_dir / "results/B/selection.json", {"selected_adapter": None})
    write(run_dir / "results/C/selection.json", {"selected_adapter": "rel/c"})
    write(run_dir / "results/Broken/selection.json", "{not json")

    def snippet_with(*names):
        return subprocess.run([sys.executable, "-", str(run_dir), "org/model", "rev-123", *names], input=snippet,
                              text=True, capture_output=True, timeout=60)

    unit = "\x1f"                                     # 字段之间是 ASCII 的单元分隔符：模型名、版本号、路径里都不会有它
    result = snippet_with("A", "B", "C", "Nope", "Broken")
    assert result.returncode == 0, result.stderr
    assert result.stdout == unit.join(["org/trained", "sha-trained", "A", "/abs/a", "B", "-", "C", "rel/c", "Nope",
                                       "-", "Broken", "-"]) + "\n"
    assert len(result.stderr.splitlines()) == 2                                # 读不出来的两组各说一句，不进 stdout
    assert "Nope：读不出选型结果" in result.stderr and "Broken：读不出选型结果" in result.stderr
    # 版本号查不到（空）：这一项照样占一个字段，不能被吞掉让后面的组名顶上来
    empty = subprocess.run([sys.executable, "-", str(run_dir), "org/model", "", "B"], input=snippet, text=True,
                           capture_output=True, timeout=60)
    assert empty.returncode == 0 and empty.stdout == unit.join(["org/model", "", "B", "-"]) + "\n"

    only_missing = snippet_with("Nope")
    assert only_missing.returncode == 1 and only_missing.stdout == ""
    assert "这些组一个都还没有选型结果：Nope" in only_missing.stderr


# ---- rft_lib.sh 里的函数 ----


def test_the_progress_bar_has_the_format_the_progress_panel_recognises(stage):
    result = harness(stage, "progress", "5", "20", "等待 vLLM 加载")
    assert result.stdout == "[######------------------] 5/20 (25%) 等待 vLLM 加载 · 已用 1m\nrc=0\n"    # 24 * 5 / 20 格
    full = "[########################] 8/8 (100%) x · 已用 1m"
    assert harness(stage, "progress", "8", "8", "x").stdout == full + "\nrc=0\n"
    assert harness(stage, "progress", "0", "8", "x").stdout == "[------------------------] 0/8 (0%) x · 已用 1m\nrc=0\n"
    assert harness(stage, "progress", "3", "0", "x").stdout == "rc=0\n"      # 总数是 0：不打印，也不做除法


def test_gpu_used_is_a_bare_number_or_nothing_at_all(stage):
    assert harness(stage, "gpu-used", FAKE_GPU_USED=" 1234 ").stdout == "1234\nrc=0\n"
    assert harness(stage, "gpu-used").stdout == "rc=0\n"                     # nvidia-smi 什么都没报
    assert harness(stage, "gpu-used", FAKE_GPU_USED="[N/A]").stdout == "rc=0\n"    # 驱动异常时的输出不能拿去比大小
    offline = harness(stage, "gpu-used", first=[stage / "offline"])          # nvidia-smi 自己出错
    assert offline.stdout == "rc=0\n" and offline.returncode == 0


def test_gpu_check_and_wait_say_what_holds_the_card(stage):
    result = harness(stage, "check-gpu")
    assert result.returncode == 0 and result.stdout == "rc=0\n"               # 显卡空着：什么都不用说
    result = harness(stage, "check-gpu", FAKE_GPU_USED="18000")
    assert "显卡被占用（已用 18000 MiB）" in result.stdout and "12345, 18000 MiB" in result.stdout
    assert result.stdout.endswith("rc=1\n")
    result = harness(stage, "wait-gpu", FAKE_GPU_USED="18000", first=[stage / "nosleep"])
    assert "vLLM 关掉两分钟后显卡仍被占用（已用 18000 MiB）" in result.stdout     # 两分钟还不放：先停在这里
    assert "12345, 18000 MiB" in result.stdout and result.stdout.endswith("rc=1\n")
    (stage / "smi.count").unlink(missing_ok=True)
    result = harness(stage, "wait-gpu", FAKE_GPU_USED="18000", FAKE_GPU_FREE_AFTER="7", first=[stage / "nosleep"])
    assert "[######------------------] 6/24 (25%) 等待显存释放 · 已用 0m" in result.stdout
    assert result.stdout.endswith("rc=0\n")                                   # 显存降下来了就接着往下走
    assert harness(stage, "wait-gpu").stdout == "rc=0\n"                     # 一开始就是空的


def test_start_vllm_waits_for_ready_and_says_why_it_could_not(stage):
    result = harness(stage, "start-stop-vllm", "data/run/vllm.log", "--enable-lora")
    assert last(result) == "rc=0 alive_after_stop=0", result.stdout + result.stderr
    assert starts(stage) == ["serve org/model --revision rev-123 --language-model-only --max-model-len 8192 "
                             "--port 8000 --enable-lora"]
    # 每 1 秒查一次日志、总共最多 15 分钟：900 次都算上，就绪时进度条是满的
    assert "[########################] 900/900 (100%) vLLM 已就绪 · 已用 0m" in result.stdout
    assert "INFO:     Application startup complete." in (stage / "repo/data/run/vllm.log").read_text(encoding="utf-8")
    for mode, reason in (("oom", "引擎初始化失败"), ("vanish", "进程在就绪之前就退出了")):
        result = harness(stage, "start-stop-vllm", "data/run/vllm.log", FAKE_VLLM_MODE=mode)
        assert last(result) == "rc=1 alive_after_stop=0", result.stdout
        assert f"vLLM 没起来（{reason}），日志最后 15 行：" in result.stdout and not alive(stage)
        if mode == "oom":                                   # 日志的最后几行原样给用户看
            assert "torch.OutOfMemoryError: CUDA out of memory." in result.stdout
    result = harness(stage, "start-stop-vllm", "data/run/vllm.log", VLLM_BIN="/nowhere/bin/vllm")
    assert last(result) == "rc=1 alive_after_stop=0" and starts(stage) == []
    assert "找不到 vLLM：/nowhere/bin/vllm（用 VLLM_BIN 指定它的路径）" in result.stdout


def test_stop_vllm_also_clears_a_server_left_behind(stage):
    """上一轮脚本中途断掉时留下的服务不是这次启动的：按进程名找得到也要关掉。"""
    stray = subprocess.Popen([str(stage / "vllm-qwen/bin/vllm"), "serve", "org/model"], start_new_session=True,
                             env=environment(stage), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert eventually(lambda: processes(stage)) and stray.poll() is None
        result = harness(stage, "stop-vllm")
        assert last(result) == "rc=0" and eventually(lambda: stray.poll() is not None), result.stdout
    finally:
        if stray.poll() is None:
            stray.kill()
        stray.wait()


def test_training_complete_reads_the_manifest(stage):
    checkpoint = stage / "repo/data/run/checkpoints/A"
    status = lambda: last(harness(stage, "training-complete", str(checkpoint)))    # noqa: E731
    assert status() == "rc=1"                                                   # 目录还没有
    write(checkpoint / "training-manifest.json", '{"status": "training"}')
    assert status() == "rc=1"                                                   # 训练做到一半
    write(checkpoint / "training-manifest.json", "{not json}")
    assert status() == "rc=1"                                                   # 写坏了
    write(checkpoint / "training-manifest.json", '{"status": "complete"}')
    assert status() == "rc=0"


ARMS = ("A-verified", "data/run/sft/A-verified", "1e-4", "32", "64", "4",
        "C-control", "data/run/sft/C-control", "1e-4", "32", "64", "5", "--allow-unverified")


def test_arm_trains_selects_and_passes_the_base_report_on_to_the_next_group(stage):
    repo = stage / "repo"
    for name in ("A-verified", "C-control"):
        write(repo / f"data/run/sft/{name}/train.jsonl", "{}\n")
    result = harness(stage, "arms", *ARMS)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = calls_of(stage)
    assert [name for name, _ in calls] == ["train", "train", "select", "tiers", "train", "train", "select", "tiers"]
    trains = [args for name, args in calls if name == "train"]
    assert "--dry-run" in trains[0] and "--dry-run" not in trains[1]             # 先 --dry-run 再真训
    assert "--epochs 1 --lr 1e-4 --lora-r 32 --lora-alpha 64 --batch-size 1 --grad-accum 8 --max-length 2048" \
        in trains[1] and "--output data/run/checkpoints/A-verified" in trains[1]
    assert "--allow-unverified" in trains[2] and "--allow-unverified" in trains[3]     # 只给对照组
    assert all("--allow-unverified" not in args for args in trains[:2])
    selects = [args for name, args in calls if name == "select"]
    base = "data/run/results/A-verified/epoch-000-dev.json"
    assert selects[0].endswith("--seed 42 --resume")                            # 第一组自己评基座
    assert selects[1].endswith(f"--seed 42 --resume --base-report {base}")      # 第二组复用第一组评出的基座报告
    assert "===== 4/7 A-verified：训练（lr 1e-4、秩 32）并在 dev 上选型 =====" in result.stdout
    assert "===== 5/7 C-control：训练（lr 1e-4、秩 32）并在 dev 上选型 =====" in result.stdout
    assert result.stdout.count("结果：A-verified dev 0.439 → 0.500（选中 epoch 1；变好 5 道、变差 1 道）") == 1
    assert "结果：C-control dev 0.439 → 0.500（选中 epoch 1；变好 5 道、变差 1 道）" in result.stdout

    result = harness(stage, "arms", *ARMS)                                      # 都选过型了：只重新打印结论
    assert result.returncode == 0 and [name for name, _ in calls_of(stage)] == ["tiers", "tiers"]
    assert "===== 4/7 A-verified：已经选过型，跳过 =====" in result.stdout
    assert "===== 5/7 C-control：已经选过型，跳过 =====" in result.stdout
    assert result.stdout.count("dev 0.439 → 0.500（选中 epoch 1；变好 5 道、变差 1 道）") == 2    # 每组一行结论


def test_arm_redoes_an_interrupted_training_and_skips_a_finished_one(stage):
    repo = stage / "repo"
    write(repo / "data/run/sft/A-verified/train.jsonl", "{}\n")
    write(repo / "data/run/checkpoints/A-verified/training-manifest.json", '{"status": "training"}')
    write(repo / "data/run/checkpoints/A-verified/leftover.bin", "x")
    result = harness(stage, "arm", *ARMS[:6])
    assert result.returncode == 0, result.stdout + result.stderr
    moved, = [item for item in (repo / "data/run/checkpoints").iterdir()
              if item.name.startswith("A-verified.incomplete-")]
    assert (moved / "leftover.bin").is_file()                                   # 没训完的那份挪开留着，不删
    assert f"上次的训练没有完成，目录移到 data/run/checkpoints/{moved.name}，重新训练" in result.stdout
    assert [name for name, _ in calls_of(stage)] == ["train", "train", "select", "tiers"]

    shutil.rmtree(repo / "data/run/results/A-verified")                        # 训练做完了、选型做到一半断了
    write(repo / "data/run/checkpoints/A-verified/training-manifest.json", '{"status": "complete"}')
    result = harness(stage, "arm", *ARMS[:6])
    assert result.returncode == 0 and "训练已经完成过，跳过训练" in result.stdout
    assert [name for name, _ in calls_of(stage)] == ["select", "tiers"]


def test_arm_stops_at_the_first_error_like_run_rft_did(stage):
    repo = stage / "repo"
    write(repo / "data/run/sft/A-verified/train.jsonl", "{}\n")
    failures = (({"STUB_TRAIN_FAIL": "dry"}, "训练数据没通过检查，把上面的报错贴给我"),
                ({"STUB_TRAIN_FAIL": "real"}, "训练出错，把上面的报错贴给我"),
                ({"STUB_SELECT_FAIL": "1"}, "选型出错，把上面的报错贴给我"))
    for extra, message in failures:
        result = harness(stage, "arm", *ARMS[:6], **extra)
        assert result.returncode == 1 and message in result.stdout, result.stdout + result.stderr
    assert not (repo / "data/run/results/A-verified/selection.json").exists()


def test_arm_reuses_a_base_report_given_by_the_caller(stage):
    repo = stage / "repo"
    write(repo / "data/run/sft/A-verified/train.jsonl", "{}\n")
    write(repo / "data/run/base.json", "{}")
    result = harness(stage, "arm", *ARMS[:6], BASE_REPORT="data/run/base.json")
    assert result.returncode == 0, result.stdout + result.stderr
    select_call, = [args for name, args in calls_of(stage) if name == "select"]
    assert select_call.endswith("--resume --base-report data/run/base.json")


def test_eval_helpers_skip_what_is_done_and_return_the_largest_exit_code(stage):
    repo = stage / "repo"
    ids, out = "data/run/probe-ids.txt", "data/run/vllm/base"
    write(repo / ids, "h3_t1\n")

    def labels():
        return [(flag(args, "label"), flag(args, "tag") or flag(args, "split")) for _, args in calls_of(stage)]

    result = harness(stage, "eval", "base", "base", out, "4", ids)           # 先 dev（贪心加采样）再探针（只贪心）
    assert last(result) == "rc=0" and labels() == [("base", "dev"), ("base", "probe")]
    dev, probe = [args for _, args in calls_of(stage)]
    assert dev.endswith("--split dev --samples 4") and probe.endswith("--ids-file data/run/probe-ids.txt --tag probe "
                                                                        "--samples 0")
    result = harness(stage, "eval", "base", "base", out, "4", ids)           # 都评过了：不再调 eval_vllm.py
    assert last(result) == "rc=0" and calls_of(stage) == []
    assert "base 的 dev 评测已经有了，跳过" in result.stdout and "base 的探针评测已经有了，跳过" in result.stdout

    (repo / out / "base-dev-sampled-k4.json").unlink()                        # summary 在、采样报告没了：不算评过
    result = harness(stage, "eval", "base", "base", out, "4", ids)
    assert last(result) == "rc=0" and labels() == [("base", "dev")]

    (repo / out / "base-probe-summary.json").unlink()                        # 探针的上次结果不留，这次又没评完
    result = harness(stage, "eval", "base", "base", out, "4", ids, STUB_EVAL_FAIL="base/probe=3")
    assert last(result) == "rc=3" and labels() == [("base", "probe")]        # 探针没评完：退出码原样带出去

    write(repo / "data/run/sft/D/train.jsonl", '{"task_id": "h3_t1"}\n')
    result = harness(stage, "eval", "D", "D", "data/run/vllm/D", "0", ids, "data/run/sft/D/train.jsonl",
                     STUB_SANITY="D=0")
    assert last(result) == "rc=4" and labels() == [("D", "dev"), ("D", "probe")]    # 自检没过：4，dev 的 0 不盖住它
    assert calls_of(stage)[1][1].endswith("--sanity-tasks-file data/run/sft/D/train.jsonl")
    assert calls_of(stage)[0][1].endswith("--split dev --samples 0")
    # 门槛就是 eval_vllm.py 的默认值 1：探针题里只有一道训练过的题时也不再按题数改 --sanity-min
    assert all("--sanity-min" not in args for _, args in calls_of(stage))

    # rft_eval_unconfirmed：自检没能和基座比较时打印原因、返回 0；比较成了、没有自检结果、读不出来都返回 1
    sanity = repo / "data/run/vllm/U/U-probe-sanity.json"
    write(sanity, {"pass": 1, "total": 6, "compared": False, "not_compared_reason": "拿不到基座名"})
    assert harness(stage, "eval-unconfirmed", "data/run/vllm/U", "U").stdout == "拿不到基座名\nrc=0\n"
    write(sanity, {"pass": 3, "total": 6, "compared": True, "changed": 3})
    assert harness(stage, "eval-unconfirmed", "data/run/vllm/U", "U").stdout == "rc=1\n"      # 比较成了
    assert last(harness(stage, "eval-unconfirmed", "data/run/vllm/U", "nobody")) == "rc=1"      # 没有自检结果
    write(sanity, {"pass": 3, "total": 6})                    # 旧版脚本写的：那时没有和基座比过
    assert harness(stage, "eval-unconfirmed", "data/run/vllm/U", "U").stdout == "没能和基座比较\nrc=0\n"

    result = harness(stage, "eval-probe", "D", "D", "data/run/vllm/D", "", "")    # 没给探针题号文件：不评探针
    assert last(result) == "rc=0" and calls_of(stage) == []

    for label, stem, expected in (("base", "dev", 0), ("base", "probe", 1), ("nobody", "dev", 1)):
        result = harness(stage, "eval-done", out, label, stem)               # 评完了 0，否则 1
        assert last(result) == f"rc={expected}", result.stdout
    assert last(harness(stage, "eval-done", "data/nowhere", "x", "dev")) == "rc=1"


def test_the_library_tells_a_finished_sampling_apart_from_a_manifest_with_slots_missing(stage):
    """清单在不等于采样做完了：槽数、读得回的采样数、判分数要一样多（旧版本的 sample_rft.py 读不回某些采样时
    照样写一份少的清单）。"""
    sampling = stage / "repo/data/run/sampling"
    done = lambda: last(harness(stage, "sampling-complete", str(sampling), "train-manifest.json"))   # noqa: E731
    assert done() == "rc=1"                                              # 清单还没有
    whole = {"tasks": 3, "slots": 6, "raw_records": 6, "graded_records": 6, "zero_pass_tasks": []}
    write(sampling / "train-manifest.json", whole)
    assert done() == "rc=0"
    for missing in ({"raw_records": 5}, {"graded_records": 5}, {"slots": 7}, {"raw_records": 0, "graded_records": 0},
                    {"slots": 0, "raw_records": 0, "graded_records": 0}):
        write(sampling / "train-manifest.json", {**whole, **missing})
        assert done() == "rc=1", missing
    for broken in ({}, {"tasks": 3}, [1, 2], "{not json"):
        write(sampling / "train-manifest.json", broken)
        assert done() == "rc=1", broken
    (sampling / "train-guided-prefix-manifest.json").write_text(json.dumps(whole), encoding="utf-8")
    assert last(harness(stage, "sampling-complete", str(sampling), "train-guided-prefix-manifest.json")) == "rc=0"
    assert last(harness(stage, "sampling-complete", str(sampling), "train-guided-skill-manifest.json")) == "rc=1"


def test_the_library_sees_a_vllm_that_has_quit_or_become_a_zombie_as_gone(stage):
    """rft_vllm_running 看的是 rft_start_vllm 记下的那个进程：没启动过、已经退出、退出后还没被收回，都算不在。"""
    assert last(harness(stage, "vllm-running")) == "rc=1"              # 没有记下的 PID
    alive_process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    zombie = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        gone.wait()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:                             # 等它退出，但不收回：留一个僵尸
            if Path(f"/proc/{zombie.pid}/stat").read_text().rsplit(")", 1)[1].startswith("Z"):
                break
            time.sleep(0.02)
        for pid, expected in ((alive_process.pid, "rc=0"), (gone.pid, "rc=1"), (zombie.pid, "rc=1"),
                              (2 ** 22 + 7, "rc=1")):
            assert last(harness(stage, "vllm-running", RFT_VLLM_PID=str(pid))) == expected, pid
    finally:
        alive_process.kill()
        alive_process.wait()
        zombie.wait()


def tiny(task_id, split):
    """一道独立的小题：第一个正数没找对，公开测试过、隐藏测试挂。"""
    module = f"{task_id}.py"
    return {
        "task_id": task_id, "source_group": f"group-{task_id}", "split": split, "bug_family": "first_positive",
        "difficulty": "L1", "module": module,
        "instruction": "实现 positive(values)：返回列表里第一个正数，没有正数时返回 None。",
        "buggy": "def positive(values):\n    return values[0] if values else None\n",
        "fixed": "def positive(values):\n    return next((v for v in values if v > 0), None)\n",
        "public_test": f"from {task_id} import positive\n\n\ndef test_public():\n    assert positive([1, 5]) == 1\n",
        "private_tests": {"test_hidden.py": f"from {task_id} import positive\n\n\ndef test_hidden():\n"
                          "    assert positive([-3, 3, 13]) == 3\n    assert positive([-1]) is None\n"},
    }


def test_the_library_drives_the_real_eval_vllm_and_the_summary_reads_what_it_writes(tmp_path, dead_proxy):
    """rft_eval_vllm 真的去调 scripts/eval_vllm.py（服务是本机的假服务、判分是真的），再让真的 vllm_summary.py
    读它写出的文件：库函数认的文件名、汇总读的字段，都得以 eval_vllm.py 实际写出的为准。"""
    sys.path.insert(0, str(ROOT / "tests"))
    from test_rft_sampling import FakeVLLM

    specs = [tiny(f"h3_p{index}", "train") for index in range(1, 4)] + [tiny("h3_d1", "dev"), tiny("h3v_d1", "dev")]
    run, tasks = tmp_path / "run", tmp_path / "tasks.jsonl"
    write(tasks, "".join(json.dumps(spec, ensure_ascii=False) + "\n" for spec in specs))
    write(run / "sft/D-prefix/train.jsonl", "".join(json.dumps({"task_id": f"h3_p{index}"}) + "\n"
                                                   for index in range(1, 4)))
    write(run / "sft/F-control/train.jsonl", '{"task_id": "h3_p1"}\n')
    write(run / "probe-ids.txt", "h3_p1\nh3_p2\nh3_p3\n")
    install(tmp_path, "bin", "python", f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')      # 脚本里写的是 python
    # 基座：train 题答错、dev 题答对；适配器（生效时）都答对。wrong["train"] 为真时适配器对 train 题也答得
    # 和基座一模一样：模拟适配器在服务里没有生效（新判据下这正是作废的原因）
    wrong = {"train": False}

    def answer(payload):
        if payload["max_tokens"] == 1:             # 探测参数用的请求（只生成 1 个 token）
            return 200, {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}
        user = payload["messages"][-1]["content"]
        spec = next(item for item in specs if f"文件 {item['module']}：" in user)
        if payload["model"] == "org/model" or (wrong["train"] and spec["split"] == "train"):
            code = spec["buggy"] if spec["split"] == "train" else spec["fixed"]
        else:
            code = spec["fixed"]
        return 200, {"choices": [{"message": {"content": f"```python\n{code}```"}, "finish_reason": "stop"}]}

    fake = FakeVLLM(answer)
    try:
        def library(*statements):
            settings = (f"RUN={shlex.quote(str(run))} TASKS={shlex.quote(str(tasks))} GRADE_WORKERS=2"
                        f" VLLM_PORT={fake.url.rsplit(':', 1)[1]}")
            environment = {**os.environ, "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}"}
            environment.pop("AGENTLAB_SANDBOX", None)
            program = f"source scripts/rft_lib.sh; {settings}; " + " ".join(statements)
            return subprocess.run(["bash", "-c", program], cwd=ROOT, env=environment, capture_output=True, text=True,
                                  timeout=600)

        quoted = {name: shlex.quote(str(run / path)) for name, path in (
            ("prefix", "vllm/D-prefix"), ("control", "vllm/F-control"), ("skill", "vllm/D-skill"),
            ("ids", "probe-ids.txt"), ("trained", "sft/D-prefix/train.jsonl"))}
        result = library(f"rft_eval_vllm D-prefix D-prefix {quoted['prefix']} 1 {quoted['ids']} {quoted['trained']};"
                         " echo rc=$?")
        assert result.stdout.strip().endswith("rc=0"), result.stdout[-2000:] + result.stderr[-2000:]
        assert "----- D-prefix：dev 上评（贪心，每题再采样 1 次）-----" in result.stdout
        assert "----- D-prefix：族内探针（贪心，先自检）-----" in result.stdout
        folder = run / "vllm/D-prefix"
        assert {item.name for item in folder.iterdir()} == {           # 和接口约定里的文件名逐个对上
            "D-prefix-dev-greedy.json", "D-prefix-dev-records.jsonl", "D-prefix-dev-sampled-k1.json",
            "D-prefix-dev-summary.json", "D-prefix-probe-greedy.json", "D-prefix-probe-records.jsonl",
            "D-prefix-probe-sanity.json", "D-prefix-probe-summary.json"}
        probe = json.loads((folder / "D-prefix-probe-summary.json").read_text(encoding="utf-8"))
        assert probe["sanity"] == {"pass": 3, "total": 3, "min": 1, "changed": 3, "compared": True,
                                    "compared_tasks": 3, "not_compared_reason": None}
        assert probe["sampled"] is None
        dev = json.loads((folder / "D-prefix-dev-summary.json").read_text(encoding="utf-8"))
        assert (dev["greedy"]["pass"], dev["greedy"]["total"]) == (2, 2)
        assert dev["sampled"]["samples_per_task"] == 1 and dev["sampled"]["passed_attempts"] == 2

        sent = len(fake.requests)
        again = library(f"rft_eval_vllm D-prefix D-prefix {quoted['prefix']} 1 {quoted['ids']} {quoted['trained']};"
                        " echo rc=$?")
        assert again.stdout.strip().endswith("rc=0") and len(fake.requests) == sent      # 评过了：不再发请求
        assert again.stdout.count("已经有了，跳过") == 2

        result = library(f"rft_eval_probe F-control F-control {quoted['control']} {quoted['ids']} ''; echo rc=$?")
        assert result.stdout.strip().endswith("rc=0"), result.stdout[-2000:] + result.stderr[-2000:]

        wrong["train"] = True                        # 适配器的输出和基座一模一样：eval_vllm.py 以 4 退出
        result = library(f"rft_eval_vllm D-skill D-skill {quoted['skill']} 1 {quoted['ids']} {quoted['trained']};"
                         " echo rc=$?")
        assert result.stdout.strip().endswith("rc=4"), result.stdout[-2000:] + result.stderr[-2000:]
        skill = run / "vllm/D-skill"
        assert (skill / "D-skill-dev-summary.json").is_file()            # dev 排在探针前面，已经评完
        assert (skill / "D-skill-probe-sanity.json").is_file()           # 自检没过：不写探针的报告和 summary
        assert not [name for name in {item.name for item in skill.iterdir()}
                    if name.startswith("D-skill-probe-") and name.endswith(("-greedy.json", "-summary.json"))]
        checked = library(f"rft_eval_done {quoted['skill']} D-skill probe; echo done=$?;"
                          f" rft_eval_unconfirmed {quoted['skill']} D-skill; echo unconfirmed_rc=$?")
        assert "done=1" in checked.stdout and "unconfirmed_rc=1" in checked.stdout    # 比较成了，不是"没能确认"
        refused = json.loads((skill / "D-skill-probe-sanity.json").read_text(encoding="utf-8"))
        assert (refused["changed"], refused["compared"], refused["min"], refused["total"]) == (0, True, 1, 3)

        summary = subprocess.run(
            [sys.executable, str(ROOT / "scripts/vllm_summary.py"), "--run", str(run), "--arms", "D-prefix",
             "F-control", "D-skill", "--probe-ids-file", str(run / "probe-ids.txt")],
            capture_output=True, text=True, timeout=60)
        assert summary.returncode == 0, summary.stderr
        def row(label, rest):
            return re.search("^" + re.escape(label) + r"[ \t]+" + rest + "$", summary.stdout, re.M)

        assert row("D-prefix", r"2/2[ \t]+100\.0%（2/2 次尝试）[ \t]+pass@1 100\.0%")
        assert re.search(r"^\s+难题 1/1、变体 1/1$", summary.stdout, re.M)
        assert row("D-prefix", r"训练过的题 3/3、没训练过的同族题：没有这样的题；"
                    r"自检：适配器的输出在 3 道题里有 3 道和基座不同（门槛 1）；训练过的题严格通过 3/3（仅供参考）")
        assert row("F-control", r"训练过的题 1/1、没训练过的同族题 2/2")      # 它的训练数据里只有 1 道探针题
        assert row("D-skill", r"（评测作废或没做）；自检：适配器的输出在 3 道题里有 0 道和基座不同（门槛 1），"
                    r"适配器在 vLLM 里没有生效；训练过的题严格通过 0/3（仅供参考）")
    finally:
        fake.close()
