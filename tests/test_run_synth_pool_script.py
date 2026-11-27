# -*- coding: utf-8 -*-
"""run_synth_pool.sh 的端到端演练：PATH 最前面的假 python 代替 scripts/synth_pool.py 的四个子命令。

演的是编排本身——前提检查、五个步骤的先后和传下去的参数、重跑时已经跑完的那一批被跳过、run 的退出码被传递、
后面三步出错只提示；脚本被 kill 或者按 Ctrl-C 时它启动的 python 跟着停。文件头里给用户粘贴的那一行启动命令
也真的启动一遍。脚本里没有内嵌的 Python 片段（它只调用 scripts/synth_pool.py），所以没有可以抽出来单独执行的片段。

末尾一个用例把真的链路整个跑一遍（真的脚本、真的 synth_pool.py、沙箱里真的判分），只有教师接口——openai SDK——
是假的：两批接着导出到同一个阶梯文件、回复全被截断的一批不算跑完。教师接口的节流参数（并发、请求间隔、输出上限）
由环境变量给、默认值保守，也在这里核对传下去的值。
"""
import contextlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from agentlab.benchmark.protocol import load_specs, validate_specs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
pytestmark = pytest.mark.skipif(not shutil.which("bash"), reason="needs Bash")
SCRIPT = ROOT / "scripts/run_synth_pool.sh"
KEY = "test-key-not-a-real-one"

STUB = r'''#!/usr/bin/env python3
"""假的 python：按脚本名分发。synth_pool.py 的四个子命令记录参数和环境，打印像样的输出，写出像样的产物；
环境变量 STUB_* 让某一步失败、占着时间（STUB_SLEEP），或者把自己的进程号和"跑完了"写到文件里
（STUB_PIDFILE、STUB_DONE）。其余的命令交给真解释器。"""
import json, os, sys, time
from pathlib import Path

args = sys.argv[1:]


def log(name):
    seen = {"sandbox": os.environ.get("AGENTLAB_SANDBOX"), "hf": os.environ.get("HF_ENDPOINT"),
            "path": os.environ.get("PYTHONPATH"), "key": os.environ.get("TEACHER_API_KEY"), "cwd": os.getcwd(),
            "watch_parent": os.environ.get("SYNTH_POOL_STOP_WITH_PARENT"),
            "max_tokens": os.environ.get("TEACHER_MAX_TOKENS")}
    with open(os.environ["STUB_LOG"], "a", encoding="utf-8") as handle:
        handle.write(name + "\t" + " ".join(args[1:]) + "\t" + json.dumps(seen) + "\n")   # 不记脚本自己的路径


def value(flag):
    return args[args.index(flag) + 1]


if not args[0].endswith("synth_pool.py"):
    os.execv(sys.executable, [sys.executable, *args])
elif args[1] == "run":
    log("run")
    if os.environ.get("STUB_PIDFILE"):
        Path(os.environ["STUB_PIDFILE"]).write_text(str(os.getpid()))
    pool = Path(value("--output-dir"))
    (pool / "families").mkdir(parents=True, exist_ok=True)
    batch, families = value("--batch"), int(value("--families"))
    for number in range(families):               # 学真的 run：已经有记录的题族不再请求
        path = pool / "families" / ("theme_%d.json" % number)
        if path.exists() and not os.environ.get("STUB_RUN_UNSETTLED"):
            continue
        path.write_text(json.dumps({"slug": "theme_%d" % number, "status": "accepted" if number % 3 else "rejected"}))
    print("第 %s 批：向教师要 %d 个题族（全部方向）；每个最多 2 轮，同时 2 个、请求至少隔 1 秒；"
          "目录里已有定论的题族 0 个" % (batch, families))
    print("[########################] %d/%d (100%%) theme_0: 接受（第 1 轮，变体 4 个）"
          " · 已用 0m" % (families, families))
    print("结果：第 %s 批 接受 2 族（母题 2 道、变体 8 道）、拒绝 1 族（原因：解析失败 1）、教师调用 3 次、用时 0 分钟"
          % batch)
    if os.environ.get("STUB_RUN_UNSETTLED"):
        print("其中 1 族没有定论，不算用掉。只补它们、不要新主题：FAMILIES=1 BATCH=%s bash scripts/run_synth_pool.sh"
              % batch)
    print("下一批这样跑：BATCH=%d bash scripts/run_synth_pool.sh" % (int(batch) + 1))
    if os.environ.get("STUB_SLEEP"):
        time.sleep(float(os.environ["STUB_SLEEP"]))       # 让这一步占着时间，好演"正在运行"
    if os.environ.get("STUB_DONE"):
        Path(os.environ["STUB_DONE"]).write_text("跑完了")  # 没被停掉才走得到这里
    if os.environ.get("STUB_RUN_FAIL"):
        print("教师调用全部出错：检查 TEACHER_API_KEY", file=sys.stderr)
        sys.exit(int(os.environ["STUB_RUN_FAIL"]))
elif args[1] == "status":
    log("status")
    if os.environ.get("STUB_STATUS_FAIL"):
        sys.exit("统计出错（假的）")
    print("结果：共接受 2 族（母题 2、变体 8）、拒绝 1 族；各方向 状态一致性 2/3")
elif args[1] == "show":
    log("show")
    if os.environ.get("STUB_SHOW_FAIL"):
        sys.exit("抽查出错（假的）")
    print("结果：抽查 3 个已接受的题族（一共 2 个）")
elif args[1] == "export":
    log("export")
    if os.environ.get("STUB_EXPORT_FAIL"):
        sys.exit("导出没有完成：阶梯文件已经有了别的内容（假的）")
    records = sorted((Path(value("--output-dir")) / "families").glob("*.json"))
    tasks = "".join(json.dumps({"task_id": "s3_%s_001" % path.stem, "split": "train"}) + "\n" for path in records)
    Path(value("--tasks-out")).write_text(tasks)
    if "--ladder" in args:
        Path(value("--ladder-out")).write_text(Path(value("--ladder")).read_text() + tasks)
    print("结果：导出 %d 道合成题（母题 3 道、变体 0 道）" % len(tasks.splitlines()))
'''

STEPS = ["===== 1/5 检查前提 =====", "===== 2/5 向教师要题族并逐个审计（每族最多 2 轮）=====",
         "===== 3/5 汇总这一池 =====", "===== 4/5 抽查 3 个已接受的题族 =====", "===== 5/5 并入阶梯 =====",
         "===== 全部结束 ====="]
# 并发、请求间隔、输出上限由脚本交给 run：取自环境变量 TEACHER_CONCURRENCY、TEACHER_MIN_INTERVAL、TEACHER_MAX_TOKENS，
# 没给就是脚本自己的默认值（2、1 秒、12000）。这一批的"跑完"记录不看它们
THROTTLE = " --concurrency 2 --min-interval 1 --max-tokens 12000"
RUN = "run --output-dir data/synth-pool --batch 1 --families 50 --directions all" + THROTTLE
STATUS = "status --output-dir data/synth-pool"
SHOW = "show --output-dir data/synth-pool --sample 3"
EXPORT = ("export --output-dir data/synth-pool --ladder data/frozen-ladder.jsonl "
          "--tasks-out data/synth-pool/synth-pool.jsonl --ladder-out data/frozen-ladder-v2.jsonl")
LADDER = '{"task_id": "demo_001", "split": "train"}\n'
ALL = ["run", "status", "show", "export"]
REST = ["status", "show", "export"]
SKIPPED = "这一批用同样的参数正常跑完过，跳过；再出一批用 BATCH=2。"
# 这个脚本自己的环境变量：外面设着的不能漏进演练里
OWNED = ("TEACHER_", "BATCH", "FAMILIES", "DIRECTIONS", "SYNTH_", "LADDER_", "STUB_", "AGENTLAB_", "HF_")
POOL = Path("data/synth-pool")


@pytest.fixture
def stage(tmp_path):
    """一份临时仓库：只放着这个脚本；PATH 最前面是假的 python。返回仓库根目录。"""
    home = tmp_path / "home"
    (home / "repo/scripts").mkdir(parents=True)
    (home / "bin").mkdir()
    shutil.copy(SCRIPT, home / "repo/scripts/run_synth_pool.sh")
    (home / "bin/python").write_text(STUB, encoding="utf-8")
    (home / "bin/python").chmod(0o755)
    (home / "repo/data").mkdir()
    (home / "repo/data/frozen-ladder.jsonl").write_text(LADDER, encoding="utf-8")
    return home / "repo"


def dot_env(repo, text):
    (repo / ".env").write_text(text, encoding="utf-8")


def environment(repo, **extra):
    """干净的环境：去掉外面设着的同类变量和这个脚本的开关，PATH 最前面是假的 python，HOME 在临时目录里
    （启动行上的锁在 $HOME 里）。沙箱的开关故意设成 required：脚本要把它取消掉。"""
    kept = {name: value for name, value in os.environ.items() if not name.startswith(OWNED)}
    kept.update({"HOME": str(repo.parent), "STUB_LOG": str(repo.parent / "calls.log"),
                 "PATH": f"{repo.parent / 'bin'}{os.pathsep}{os.environ['PATH']}", "AGENTLAB_SANDBOX": "required"})
    kept.update(extra)
    return kept


def calls_of(repo):
    """假 python 到此刻为止记下的调用：[(子命令, 参数, 当时的环境)]。"""
    log = repo.parent / "calls.log"
    if not log.exists():
        return []
    return [tuple(line.split("\t")) for line in log.read_text(encoding="utf-8").splitlines()]


def run(repo, **settings):
    """在临时仓库里跑一遍脚本（传进来的设置成了环境变量；key=None 表示没有密钥）。返回 (进程结果, 调用)。"""
    (repo.parent / "calls.log").unlink(missing_ok=True)
    key = settings.pop("key", KEY)
    settings["TEACHER_API_KEY"] = "" if key is None else key
    result = subprocess.run(["bash", "scripts/run_synth_pool.sh"], cwd=repo, env=environment(repo, **settings),
                            capture_output=True, text=True, timeout=120)
    return result, calls_of(repo)


def start(repo, **settings):
    """在后台起一份脚本（不等着它跑完）：演"正在运行的时候停掉它"。"""
    settings["TEACHER_API_KEY"] = KEY
    return subprocess.Popen(["bash", "scripts/run_synth_pool.sh"], cwd=repo, env=environment(repo, **settings),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def names(calls):
    return [name for name, _, _ in calls]


def stamps(repo, directory=POOL):
    """batches.done 里记着的各行（正常跑完过的批次，一行一个）。"""
    path = repo / directory / "batches.done"
    return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


def stamp(repo, batch=1, directory=POOL):
    """这一批最先记下的那次是用什么参数跑完的；没跑完过就是 None。"""
    lines = [line for line in stamps(repo, directory) if line.startswith(f"batch={batch} ")]
    return lines[0][len(f"batch={batch} "):] if lines else None


def families(repo, directory=POOL):
    return sorted(path.read_bytes() for path in (repo / directory / "families").glob("*.json"))


def eventually(condition, seconds=20):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return bool(condition())


def alive(pid):
    """这个进程还在不在（僵尸算不在）。"""
    try:
        state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()[0]
    except OSError:
        return False
    return state != "Z"


def test_a_fresh_run_goes_through_the_five_steps_and_hands_over_the_arguments(stage):
    dot_env(stage, "TEACHER_API_KEY=from-dot-env\nAGENTLAB_SANDBOX=required\n")
    result, calls = run(stage)
    assert result.returncode == 0, result.stdout + result.stderr
    assert [line for line in result.stdout.splitlines() if line.startswith("=====")] == STEPS
    assert result.stdout.rstrip("\n").splitlines()[-1] == "===== 全部结束 ====="
    assert [(name, arguments) for name, arguments, _ in calls] == [
        ("run", RUN), ("status", STATUS), ("show", SHOW), ("export", EXPORT)]
    # 环境照其它脚本：回到仓库根、PYTHONPATH 指向 src、沙箱的开关被取消（.env 里写了也不算数）、
    # HF 走镜像、.env 里的密钥传给了子进程；run 另外被告知盯着父进程
    assert json.loads(calls[0][2]) == {"sandbox": None, "hf": "https://hf-mirror.com", "path": f"{stage}/src",
                                       "key": "from-dot-env", "cwd": str(stage), "watch_parent": "1",
                                       "max_tokens": None}
    # 环境里没给 TEACHER_MAX_TOKENS：输出上限是脚本自己的默认值，作为 --max-tokens 交给 run
    # run 的输出原样透出来（同时存进这一批自己的日志），后面三步的结论行也在
    assert "第 1 批：向教师要 50 个题族（全部方向）；每个最多 2 轮，同时 2 个、请求至少隔 1 秒；" in result.stdout
    assert "[########################] 50/50 (100%) theme_0: 接受（第 1 轮，变体 4 个） · 已用 0m" in result.stdout
    assert ("结果：第 1 批 接受 2 族（母题 2 道、变体 8 道）、拒绝 1 族（原因：解析失败 1）、"
            "教师调用 3 次") in result.stdout
    assert "下一批这样跑：BATCH=2 bash scripts/run_synth_pool.sh" in result.stdout
    log = (stage / POOL / "batch-1.log").read_text(encoding="utf-8")
    assert log.startswith("第 1 批：向教师要 50 个题族") and log.rstrip("\n").endswith(
        "下一批这样跑：BATCH=2 bash scripts/run_synth_pool.sh")
    for piece in ("结果：共接受 2 族", "结果：抽查 3 个已接受的题族", "结果：导出 50 道合成题"):
        assert piece in result.stdout, piece
    assert "环境里没有 TEACHER_API_KEY" not in result.stdout and "跳过导出" not in result.stdout
    # 这一批正常跑完的记录；导出的两个文件也写出来了
    assert stamps(stage) == ["batch=1 families=50 directions=all"]
    assert (stage / "data/frozen-ladder-v2.jsonl").read_text(encoding="utf-8") == LADDER + (
        stage / POOL / "synth-pool.jsonl").read_text(encoding="utf-8")
    # 文件头写清了用法、怎么停、看进度的办法和防重复启动的那一行；脚本里没有内嵌的 Python 片段
    text = SCRIPT.read_text(encoding="utf-8")
    header = text.split("\ncd ", 1)[0]
    for piece in ("bash scripts/run_synth_pool.sh", "BATCH=2 bash scripts/run_synth_pool.sh",
                  "FAMILIES=3 BATCH=1 bash scripts/run_synth_pool.sh", "TEACHER_API_KEY", "TEACHER_MAX_TOKENS",
                  "怎么停", "pkill -f run_synth_pool.sh", "scripts/show_progress.sh synth-batch1.log",
                  'flock -n -o "$HOME/synth-pool.lock"', "batches.done",
                  # 教师接口的节流参数和它们的默认值，写在环境变量那一段里
                  "TEACHER_CONCURRENCY=2", "TEACHER_MIN_INTERVAL=1", "TEACHER_MAX_TOKENS=12000"):
        assert piece in header, piece
    assert "<<" not in text


def test_environment_variables_choose_the_batch_and_the_files(stage):
    result, calls = run(stage, BATCH="2", FAMILIES="5", DIRECTIONS="lock,cond", SYNTH_DIR="data/other-pool",
                        LADDER_FILE="data/frozen-ladder.jsonl", LADDER_OUT="data/ladder-next.jsonl")
    assert result.returncode == 0, result.stdout + result.stderr
    assert [(name, arguments) for name, arguments, _ in calls] == [
        ("run", "run --output-dir data/other-pool --batch 2 --families 5 --directions lock,cond" + THROTTLE),
        ("status", "status --output-dir data/other-pool"), ("show", "show --output-dir data/other-pool --sample 3"),
        ("export", "export --output-dir data/other-pool --ladder data/frozen-ladder.jsonl "
                   "--tasks-out data/other-pool/synth-pool.jsonl --ladder-out data/ladder-next.jsonl")]
    assert stamps(stage, "data/other-pool") == ["batch=2 families=5 directions=lock,cond"]
    assert "第 2 批：向教师要 5 个题族（方向 lock,cond）" in result.stdout
    assert "下一批这样跑：BATCH=3 bash scripts/run_synth_pool.sh" in result.stdout


def test_a_rerun_skips_a_batch_that_finished_and_only_redoes_what_is_left(stage):
    result, calls = run(stage)
    assert result.returncode == 0 and names(calls) == ALL
    before = families(stage)
    # 原样重跑：第 2 步跳过（再要就是下一批的主题，价钱照算），后面三步照做
    result, calls = run(stage)
    assert result.returncode == 0, result.stdout + result.stderr
    assert names(calls) == REST and SKIPPED in result.stdout
    assert result.stdout.rstrip("\n").splitlines()[-1] == "===== 全部结束 ====="
    assert families(stage) == before
    # 这一批的参数变了才算没跑过：重新请求，记录里另起一行，原来那一行还在
    result, calls = run(stage, FAMILIES="20")
    assert result.returncode == 0 and names(calls) == ALL
    assert calls[0][1] == "run --output-dir data/synth-pool --batch 1 --families 20 --directions all" + THROTTLE
    assert stamps(stage) == ["batch=1 families=50 directions=all", "batch=1 families=20 directions=all"]
    # 默认参数的那一次照样算跑完：原样重跑不会再请求（补做过一次不能把这一保护弄丢）
    result, calls = run(stage)
    assert result.returncode == 0 and names(calls) == REST
    # 下一批有自己的记录，照常请求
    result, calls = run(stage, BATCH="2")
    assert result.returncode == 0 and names(calls) == ALL
    assert calls[0][1] == "run --output-dir data/synth-pool --batch 2 --families 50 --directions all" + THROTTLE
    assert stamps(stage)[2:] == ["batch=2 families=50 directions=all"]


def test_families_without_a_verdict_do_not_keep_the_batch_from_counting_as_done(stage):
    # run 正常跑完（退出码 0），只是有一族没有定论：这一批照样记成跑完，原样重跑不会再请求
    result, calls = run(stage, STUB_RUN_UNSETTLED="1")
    assert result.returncode == 0 and names(calls) == ALL
    assert ("其中 1 族没有定论，不算用掉。只补它们、不要新主题："
            "FAMILIES=1 BATCH=1 bash scripts/run_synth_pool.sh") in result.stdout
    assert stamps(stage) == ["batch=1 families=50 directions=all"]
    result, calls = run(stage)
    assert result.returncode == 0 and names(calls) == REST
    # 照提示只补它们（FAMILIES 写成它们的个数）：这是另一种参数，另记一行；默认参数的记录还在
    result, calls = run(stage, FAMILIES="1")
    assert result.returncode == 0 and names(calls) == ALL
    assert calls[0][1] == "run --output-dir data/synth-pool --batch 1 --families 1 --directions all" + THROTTLE
    assert stamps(stage) == ["batch=1 families=50 directions=all", "batch=1 families=1 directions=all"]
    result, calls = run(stage, FAMILIES="1")
    assert result.returncode == 0 and names(calls) == REST              # 补做过一次的也不再重复
    result, calls = run(stage)
    assert result.returncode == 0 and names(calls) == REST


def test_without_a_key_it_stops_and_says_where_to_put_the_env_file(stage):
    result, calls = run(stage, key=None)
    assert result.returncode == 1 and calls == []
    assert "环境里没有 TEACHER_API_KEY：把 .env 放到项目目录（仓库根，和 scripts 并列）" in result.stdout
    assert [line for line in result.stdout.splitlines() if line.startswith("=====")] == STEPS[:1]
    assert "全部结束" not in result.stdout and not (stage / POOL / "families").exists()
    # 密钥直接放在环境变量里也行
    result, calls = run(stage, key="from-environment")
    assert result.returncode == 0 and names(calls) == ALL
    # 当前目录有 .env 时先载入它：里面的值盖过外面同名的环境变量
    dot_env(stage, "TEACHER_API_KEY=from-dot-env\nTEACHER_MAX_TOKENS=9000\n")
    result, calls = run(stage, key="from-environment")
    assert result.returncode == 0 and json.loads(calls[0][2])["key"] == "from-dot-env"
    assert json.loads(calls[0][2])["max_tokens"] == "9000"         # 教师接口的参数原样传给 run


def test_the_teacher_throttling_follows_the_environment_variables(stage):
    """并发、请求间隔、输出上限由环境变量给（默认值保守：2、1 秒、12000）；.env 里写的也算。"""
    result, calls = run(stage, TEACHER_CONCURRENCY="3", TEACHER_MIN_INTERVAL="0.5", TEACHER_MAX_TOKENS="9000")
    assert result.returncode == 0, result.stdout + result.stderr
    assert calls[0][1] == ("run --output-dir data/synth-pool --batch 1 --families 50 --directions all"
                           " --concurrency 3 --min-interval 0.5 --max-tokens 9000")
    dot_env(stage, f"TEACHER_API_KEY={KEY}\nTEACHER_CONCURRENCY=1\nTEACHER_MIN_INTERVAL=2\n")
    result, calls = run(stage, BATCH="2", TEACHER_CONCURRENCY="5")     # 外面给了也没用：.env 里的盖过它
    assert result.returncode == 0, result.stdout + result.stderr
    assert calls[0][1].endswith("--batch 2 --families 50 --directions all"
                                " --concurrency 1 --min-interval 2 --max-tokens 12000")
    assert json.loads(calls[0][2])["max_tokens"] is None               # .env 里没写：脚本自己的默认值


def test_a_missing_ladder_file_skips_the_export(stage):
    (stage / "data/frozen-ladder.jsonl").unlink()
    result, calls = run(stage)
    assert result.returncode == 0, result.stdout + result.stderr
    assert names(calls) == ["run", "status", "show"]
    assert [line for line in result.stdout.splitlines() if line.startswith("=====")] == STEPS
    assert ("阶梯文件 data/frozen-ladder.jsonl 不存在：跳过导出。接受的题族都在 data/synth-pool/families，"
            "要并入阶梯时把它拷到这个路径，再原样重跑（第 1 批不会重新请求）。") in result.stdout.replace("\n", "")
    assert not (stage / "data/frozen-ladder-v2.jsonl").exists() and stamp(stage) is not None
    # 把阶梯拷进来再原样重跑：第 2 步跳过，只做导出
    (stage / "data/frozen-ladder.jsonl").write_text(LADDER, encoding="utf-8")
    result, calls = run(stage)
    assert result.returncode == 0 and names(calls) == REST
    assert (stage / "data/frozen-ladder-v2.jsonl").exists()


@pytest.mark.parametrize("code", ["3", "1"])
def test_the_exit_code_of_run_is_passed_through_and_nothing_else_runs(stage, code):
    result, calls = run(stage, STUB_RUN_FAIL=code)
    assert result.returncode == int(code) and names(calls) == ["run"]
    # run 自己的报错（标准错误）也透出来了；脚本说明退出码，然后停下，不写"全部结束"
    assert "教师调用全部出错：检查 TEACHER_API_KEY" in result.stdout
    assert result.stdout.rstrip("\n").splitlines()[-1] == (
        f"第 1 批没有跑完（退出码 {code}）：原因看上面的说明。修好以后原样重跑——已经有定论的题族不会再请求。")
    assert "===== 3/5" not in result.stdout and not (stage / POOL / "synth-pool.jsonl").exists()
    assert stamp(stage) is None                               # 没跑完：不留记录
    # 修好以后原样重跑：run 被再次调用，这一回走到了最后
    result, calls = run(stage)
    assert result.returncode == 0 and names(calls) == ALL and stamp(stage) is not None


def test_failures_in_status_show_and_export_only_leave_a_note(stage):
    result, calls = run(stage, STUB_STATUS_FAIL="1", STUB_SHOW_FAIL="1", STUB_EXPORT_FAIL="1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert names(calls) == ALL
    assert [line for line in result.stdout.splitlines() if line.startswith("=====")] == STEPS
    # 这三步自己的报错留在标准错误里；脚本在标准输出里各补一句提示
    assert result.stderr.splitlines() == [
        "统计出错（假的）", "抽查出错（假的）", "导出没有完成：阶梯文件已经有了别的内容（假的）"]
    for piece in ("统计没打印出来（不影响后面的步骤；把上面的报错贴给我）",
                  "抽查没打印出来（不影响导出；把上面的报错贴给我）",
                  "导出没有完成（原因见上一行）。处理以后原样重跑，第 1 批不会重新请求。"):
        assert piece in result.stdout, piece
    assert stamp(stage) is not None                           # 这一批本身跑完了：重跑时不再请求
    result, calls = run(stage)
    assert result.returncode == 0 and names(calls) == REST
    assert (stage / "data/frozen-ladder-v2.jsonl").exists()


def test_killing_the_script_stops_the_python_it_started(stage):
    """只 kill 这个脚本（按名字 pkill 打到的就是它）：它启动的 python 跟着停，不会成了孤儿把这一批要完。"""
    if not Path("/proc/self/stat").exists():
        pytest.skip("要看 /proc 才知道那个不归测试管的进程停没停")
    pid_file, done = stage.parent / "stub.pid", stage.parent / "stub.done"
    process = start(stage, STUB_SLEEP="8", STUB_PIDFILE=str(pid_file), STUB_DONE=str(done))
    try:
        assert eventually(pid_file.exists)
        victim = int(pid_file.read_text(encoding="utf-8"))
        process.send_signal(signal.SIGTERM)
        out, _ = process.communicate(timeout=20)
        assert process.returncode == 143
        assert eventually(lambda: not alive(victim), seconds=5)    # python 跟着停了
        time.sleep(1.0)
        assert not done.exists()                                  # 没有把这一步跑完
        assert "===== 全部结束 =====" not in out and stamps(stage) == []
    finally:
        stop(process)


def test_the_first_ctrl_c_waits_for_the_run_to_finish_and_the_second_stops_it(stage):
    """Ctrl-C 是终端同时发给脚本和 python 的：第一次 python 自己会停掉排着的、等正在做的做完，脚本只提醒一句、
    接着等它；再按一次，脚本把 python 立刻停掉。这里只给脚本发信号，所以假 python 第一次不受影响。"""
    if not Path("/proc/self/stat").exists():
        pytest.skip("要看 /proc 才知道那个不归测试管的进程停没停")
    pid_file, done = stage.parent / "stub.pid", stage.parent / "stub.done"
    warning = ("正在停：排着的主题不再请求，正在进行的题族做完就退出；再按一次 Ctrl-C 立刻停"
               "（它们这一次的回答不保存）。")
    process = start(stage, STUB_SLEEP="3", STUB_PIDFILE=str(pid_file), STUB_DONE=str(done))
    try:
        assert eventually(pid_file.exists)
        process.send_signal(signal.SIGINT)
        out, _ = process.communicate(timeout=30)
        assert warning in out and process.returncode == 0       # 假 python 没收到信号，照常跑完，脚本走到最后
        assert done.exists() and "===== 全部结束 =====" in out and stamp(stage) is not None
    finally:
        stop(process)
    done.unlink()
    pid_file.unlink()
    process = start(stage, BATCH="2", STUB_SLEEP="8", STUB_PIDFILE=str(pid_file), STUB_DONE=str(done))
    try:
        assert eventually(pid_file.exists)
        victim = int(pid_file.read_text(encoding="utf-8"))
        process.send_signal(signal.SIGINT)
        time.sleep(0.3)
        process.send_signal(signal.SIGINT)
        out, _ = process.communicate(timeout=20)
        assert warning in out and process.returncode == 130
        assert eventually(lambda: not alive(victim), seconds=5) and not done.exists()
        assert "===== 全部结束 =====" not in out and stamp(stage, 2) is None
    finally:
        stop(process)


def stop(process):
    if process.poll() is None:
        process.kill()
        process.wait(timeout=10)


# ── 文件头里给用户粘贴的那一行启动命令 ──────────────────────────────────

STARTED = "已启动：在后台运行，日志在 synth-pool.log"
REFUSED = "已经在运行，不重复启动"
DONE = "已经跑完。synth-pool.log 的最后一行：===== 全部结束 ====="


def launch_line():
    """文件头里给用户粘贴的那一行启动命令。"""
    header = SCRIPT.read_text(encoding="utf-8")
    line, = [row[len("#   "):] for row in header.splitlines() if row.startswith("#   setsid ")]
    return line


def need_launch_tools():
    for program in ("flock", "setsid", "tail"):
        if not shutil.which(program):
            pytest.skip(f"需要 {program}")


def paste(repo, out, prefix="", **settings):
    """像用户那样把启动行交给一个 bash（prefix 是写在最前面的环境变量，例如 "BATCH=3 "）。后台那一串进程
    继承着标准输出，用管道去接要等它们全部退出，所以让它们写进文件。"""
    settings["TEACHER_API_KEY"] = KEY
    command = "{ " + prefix + launch_line() + " } > " + shlex.quote(str(out)) + " 2>&1"
    result = subprocess.run(["bash", "-c", command], cwd=repo, env=environment(repo, **settings),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def terminal(out):
    """终端里这一次粘贴打出来的各行：等到有一行、而且短时间内不再增加为止。"""
    deadline, lines = time.monotonic() + 30, []
    while time.monotonic() < deadline:
        current = out.read_text(encoding="utf-8").splitlines() if out.exists() else []
        if len(current) > len(lines):
            lines = current
        elif lines:
            return lines
        time.sleep(0.2)
    return lines


def log_shows(repo, *pieces, seconds=60):
    """等 synth-pool.log 里同时出现 pieces（启动时日志被重新写过，所以不会读到上一次的），返回整份日志。"""
    path, deadline = repo / "synth-pool.log", time.monotonic() + seconds
    while time.monotonic() < deadline:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        if all(piece in text for piece in pieces):
            return text
        time.sleep(0.2)
    return path.read_text(encoding="utf-8") if path.exists() else ""


def test_the_launch_line_runs_in_the_background_and_a_second_paste_starts_nothing_else(stage):
    need_launch_tools()
    (stage.parent / "calls.log").unlink(missing_ok=True)
    first = stage.parent / "first.out"
    paste(stage, first, STUB_SLEEP="4")
    assert terminal(first)[:1] == [STARTED]                 # 拿到了锁：脚本在后台跑起来了
    # 还在跑的时候再粘贴一次：锁被占着，只打一句话，没有动正在跑的那份日志
    second = stage.parent / "second.out"
    paste(stage, second, STUB_SLEEP="4")
    assert terminal(second) == [REFUSED]
    text = log_shows(stage, "===== 全部结束 =====")
    assert text.count("===== 全部结束 =====") == 1 and text.startswith("===== 1/5 检查前提 =====")
    assert names(calls_of(stage)) == ALL and stamp(stage) is not None
    assert terminal(first)[-1] == DONE
    # 跑完以后再粘贴：锁已经放开，这就是一次重跑——第 2 步跳过，只补后面三步
    third = stage.parent / "third.out"
    paste(stage, third)
    text = log_shows(stage, "正常跑完过，跳过", "===== 全部结束 =====")
    assert text.count("===== 全部结束 =====") == 1 and "===== 2/5" in text
    assert names(calls_of(stage)) == [*ALL, *REST]
    assert terminal(third)[-1] == DONE
    # 照文件头说的把 BATCH=3 写在最前面：这是第 3 批，照常向教师要题
    fourth = stage.parent / "fourth.out"
    paste(stage, fourth, prefix="BATCH=3 ")
    text = log_shows(stage, "第 3 批：向教师要 50 个题族", "===== 全部结束 =====")
    assert text.count("===== 全部结束 =====") == 1 and "跳过" not in text
    assert names(calls_of(stage)) == [*ALL, *REST, *ALL]
    assert calls_of(stage)[7][1] == RUN.replace("--batch 1", "--batch 3")
    assert stamp(stage, 3) == "families=50 directions=all" and terminal(fourth)[-1] == DONE


# ── 真的流水线端到端：真的脚本、真的 synth_pool.py、真的判分，只有教师接口（openai SDK）是假的 ──────
# 前面的演练只看编排。这里把整条链真跑一遍：两批接着导出、回复全被截断的一批、修好以后原样重跑。

FAKE_OPENAI = r'''"""假的 openai SDK：只有 make_client 用到的 OpenAI 类。教师的回复由环境变量 FAKE_TEACHER_MODE 决定：
不给就是按题族的模块名拼一份合格的题族（手工构造的点名册小程序）；truncated 是写到一半被截断的回复。"""
import json, os, re, sys, time, types


class Completions:
    def create(self, model, messages, max_tokens, temperature):
        user = messages[1]["content"]
        with open(os.environ["FAKE_TEACHER_CALLS"], "a", encoding="utf-8") as handle:
            handle.write(re.search(r"s3_[a-z0-9_]+(?=\.py)", user).group(0) + "\n")
        time.sleep(float(os.environ.get("FAKE_TEACHER_SLEEP", "0")))     # 演示"正在进行的请求"要花时间
        if os.environ.get("FAKE_TEACHER_MODE") == "truncated":
            content, finish = '```json\n{"theme": "点名册", "instruction": "修复 Tally：要求：\\n1）到场', "length"
        else:
            sys.path.insert(0, os.environ["FAKE_TESTS_DIR"])
            import test_synth_pool
            content = json.dumps(test_synth_pool.family(re.search(r"s3_([a-z0-9_]+)(?=\.py)", user).group(1)),
                                 ensure_ascii=False)
            finish = "stop"
        message = types.SimpleNamespace(content=content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message, finish_reason=finish)],
                                     usage=types.SimpleNamespace(prompt_tokens=11, completion_tokens=7))


class OpenAI:
    def __init__(self, *arguments, **keywords):
        self.chat = types.SimpleNamespace(completions=Completions())
'''

BOOT = r'''"""PATH 最前面的 python：真的解释器，只是先把假的 openai SDK 放进搜索路径，再照原样运行脚本。"""
import runpy, sys
sys.path.insert(0, sys.argv[1])                   # 假的 openai 包在这里
sys.argv = sys.argv[2:]
runpy.run_path(sys.argv[0], run_name="__main__")
'''


def tiny_task(name, split="train"):
    """阶梯里原有的一道小题（和合成题无关）。"""
    return {"task_id": name, "source_group": name, "split": split, "bug_family": "clamp", "difficulty": "L1",
            "instruction": "实现 clamp(values, low, high)：把每个数限制在 [low, high] 内，返回新列表。",
            "module": f"{name}.py",
            "buggy": "def clamp(values, low, high):\n    return list(values)\n",
            "fixed": "def clamp(values, low, high):\n    return [min(max(v, low), high) for v in values]\n",
            "public_test": f"from {name} import clamp\n\ndef test_public():\n"
                           "    assert clamp([1, 5], 0, 10) == [1, 5]\n",
            "private_tests": {"test_hidden.py": f"from {name} import clamp\n\ndef test_hidden():\n"
                                                "    assert clamp([-3, 13], 0, 10) == [0, 10]\n"}}


@pytest.fixture
def real_stage(tmp_path):
    """一份临时仓库：scripts 和 src 指向真的仓库；PATH 最前面是一个 python，它把假的 openai SDK 交给真的解释器。"""
    fake, repo = tmp_path / "fake", tmp_path / "home/repo"
    (fake / "openai").mkdir(parents=True)
    (repo / "data").mkdir(parents=True)
    (fake / "openai/__init__.py").write_text(FAKE_OPENAI, encoding="utf-8")
    (fake / "boot.py").write_text(BOOT, encoding="utf-8")
    python = fake / "python"
    python.write_text(f'#!/usr/bin/env bash\nexec "{sys.executable}" "{fake / "boot.py"}" "{fake}" "$@"\n',
                      encoding="utf-8")
    python.chmod(0o755)
    os.symlink(ROOT / "scripts", repo / "scripts")
    os.symlink(ROOT / "src", repo / "src")
    (repo / "data/frozen-ladder.jsonl").write_text("".join(
        json.dumps(tiny_task(name), sort_keys=True) + "\n" for name in ("demo_a", "h3_c")), encoding="utf-8")
    return repo



def real_environment(repo, calls, **settings):
    """跑真的脚本用的环境。HOME 不换：解释器的用户级 site-packages（pydantic 在里面）是按 HOME 找的。"""
    kept = {name: value for name, value in os.environ.items() if not name.startswith((*OWNED, "FAKE_TEACHER_"))}
    kept.update({"PATH": f"{repo.parent.parent / 'fake'}{os.pathsep}{os.environ['PATH']}",
                 "AGENTLAB_SANDBOX": "required", "TEACHER_API_KEY": KEY, "FAKE_TESTS_DIR": str(ROOT / "tests"),
                 "FAKE_TEACHER_CALLS": str(calls), **settings})
    return kept


def real_run(repo, **settings):
    """在临时仓库里把真的脚本整个跑一遍。返回 (进程结果, 教师被请求的模块名，按先后)。"""
    calls = repo.parent / "teacher-calls.log"
    calls.unlink(missing_ok=True)
    result = subprocess.run(["bash", "scripts/run_synth_pool.sh"], cwd=repo,
                            env=real_environment(repo, calls, **settings), capture_output=True, text=True, timeout=900)
    return result, (calls.read_text(encoding="utf-8").splitlines() if calls.exists() else [])


def teacher_calls(calls):
    return len(calls.read_text(encoding="utf-8").splitlines()) if calls.exists() else 0


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def test_killing_the_real_script_with_kill_9_stops_the_run_too(real_stage):
    """kill -9 打到脚本自己：trap 也执行不了。python 发现父进程没了（run_synth_pool.sh 调它时交代过要盯着），
    自己停下：正在进行的几次调用跟着断，排着的主题不再请求。"""
    if not Path("/proc/self/stat").exists():
        pytest.skip("要看 /proc 才知道那些不归测试管的进程停没停")
    calls = real_stage.parent / "teacher-calls.log"
    calls.unlink(missing_ok=True)
    log = open(real_stage.parent / "run.log", "w", encoding="utf-8")     # noqa: SIM115   子进程的输出都进这个文件
    process = subprocess.Popen(["bash", "scripts/run_synth_pool.sh"], cwd=real_stage,
                               env=real_environment(real_stage, calls, FAMILIES="8", FAKE_TEACHER_SLEEP="5"),
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    children = []
    try:
        assert eventually(lambda: teacher_calls(calls) >= 2)
        children = [int(pid) for pid in
                    Path(f"/proc/{process.pid}/task/{process.pid}/children").read_text(encoding="utf-8").split()]
        process.kill()                                   # kill -9：只打到 bash
        process.wait(timeout=10)
        assert eventually(lambda: not any(alive(pid) for pid in children), seconds=10)   # python（和 tee）都退了
        time.sleep(4.0)                                  # 正在进行的调用各要 5 秒：python 还活着的话这时还会有调用
        assert teacher_calls(calls) <= 4                 # 同时最多 2 个：排着的六个主题没有请求
        assert not (real_stage / POOL / "batches.done").exists()
    finally:
        log.close()
        stop(process)
        for pid in children:
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)


def records(pool):
    """families/ 里的题族记录，按 slug 排序。"""
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted((pool / "families").glob("*.json"))]


def task_count(record):
    return len(record["family"]["variants"]) + 1          # 母题加变体


def test_two_batches_export_into_the_same_ladder_file_and_a_hopless_batch_does_not_count_as_done(real_stage):
    pool, ladder, ladder_out = (real_stage / "data/synth-pool", real_stage / "data/frozen-ladder.jsonl",
                                real_stage / "data/frozen-ladder-v2.jsonl")
    # 第 1 批：默认的 LADDER_OUT 还没有，写出来
    result, asked = real_run(real_stage, FAMILIES="3")
    assert result.returncode == 0, result.stdout + result.stderr
    assert [line for line in result.stdout.splitlines() if line.startswith("=====")] == STEPS
    kept = [record for record in records(pool) if record["status"] == "accepted"]
    assert len(asked) == 3 and sorted(f"s3_{record['slug']}" for record in kept) == sorted(asked)   # 一族一次
    assert "并入阶梯：" in result.stdout and len(jsonl(pool / "synth-pool.jsonl")) == sum(map(task_count, kept))
    assert jsonl(ladder_out) == [*jsonl(ladder), *jsonl(pool / "synth-pool.jsonl")]
    assert (pool / "batches.done").read_text(encoding="utf-8").splitlines() == ["batch=1 families=3 directions=all"]
    # 第 2 批：照 run 的提示原样跑下一批。LADDER_OUT 已经在了（第 1 批导出的），这一回把它更新成阶梯 + 全部新题
    result, asked = real_run(real_stage, BATCH="2", FAMILIES="3")
    assert result.returncode == 0, result.stdout + result.stderr
    kept = [record for record in records(pool) if record["status"] == "accepted"]
    assert len(asked) == 3 and len(kept) == 6
    assert "是以前导出的（只有前几批的题），更新成现在的内容" in result.stdout
    assert "导出没有完成" not in result.stdout and len(jsonl(pool / "synth-pool.jsonl")) == sum(map(task_count, kept))
    assert jsonl(ladder_out) == [*jsonl(ladder), *jsonl(pool / "synth-pool.jsonl")]
    validate_specs(load_specs(ladder_out))
    assert (pool / "batches.done").read_text(encoding="utf-8").splitlines() == [
        "batch=1 families=3 directions=all", "batch=2 families=3 directions=all"]
    # 第 2 批原样重跑：第 2 步跳过；阶梯已经是这个内容，说明后跳过
    result, asked = real_run(real_stage, BATCH="2", FAMILIES="3")
    assert result.returncode == 0 and asked == []                       # 没有再向教师要
    assert "这一批用同样的参数正常跑完过，跳过；再出一批用 BATCH=3。" in result.stdout
    assert "阶梯文件 data/frozen-ladder-v2.jsonl 已经是这个内容，跳过" in result.stdout
    # 第 3 批：教师的回复全被截断（输出上限太小）。一族都没接受、都不是题族的事：退出码 3，这一批不算跑完
    result, asked = real_run(real_stage, BATCH="3", FAMILIES="2", FAKE_TEACHER_MODE="truncated")
    assert result.returncode == 3 and len(asked) == 4                # 两族各两轮
    assert "回复被截断 2 族）。回复被截断就把输出上限调大：环境变量 TEACHER_MAX_TOKENS（现在 12000）" in result.stdout
    assert result.stdout.rstrip("\n").splitlines()[-1] == (
        "第 3 批没有跑完（退出码 3）：原因看上面的说明。修好以后原样重跑——已经有定论的题族不会再请求。")
    assert "===== 3/5" not in result.stdout
    third = [record for record in records(pool) if record["batch"] == 3]
    assert {record["reason"] for record in third} == {"回复被截断"} and len(third) == 2
    assert "batch=3" not in (pool / "batches.done").read_text(encoding="utf-8")
    # 修好以后原样重跑：就是把这两族再要一次，不会要新主题；然后照常导出
    result, asked = real_run(real_stage, BATCH="3", FAMILIES="2")
    assert result.returncode == 0, result.stdout + result.stderr
    assert sorted(asked) == sorted(f"s3_{record['slug']}" for record in third)
    assert "都是以前要过、还没有定论的（补做）" in result.stdout     # 就是把这两族再要一次，没有别的
    kept = [record for record in records(pool) if record["status"] == "accepted"]
    assert len(kept) == 8 and len(jsonl(pool / "synth-pool.jsonl")) == sum(map(task_count, kept))
    assert jsonl(ladder_out) == [*jsonl(ladder), *jsonl(pool / "synth-pool.jsonl")]
    assert (pool / "batches.done").read_text(encoding="utf-8").splitlines()[-1] == "batch=3 families=2 directions=all"
    validate_specs(load_specs(ladder_out))
