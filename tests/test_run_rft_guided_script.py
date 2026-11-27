# -*- coding: utf-8 -*-
"""run_rft_guided.sh 的端到端演练。

假的 python 给编排脚本调到的每个脚本写出像样的产物，并把参数记进日志（内嵌的 Python 片段和 vllm_summary.py
交给真解释器）；假的 vllm 服务记下自己的启动参数，假的 nvidia-smi 按环境变量报显存，假的 pkill 只管这次演练
自己的进程。rft_lib.sh、rft_vllm_eval.sh、vllm_summary.py 用真的。演的是编排本身——步骤的顺序、三组的超参、
配不上时怎么重做、两次运行之间 SKILL 变了怎么重配、重跑时跳过什么、各种失败时停在哪里；内嵌的 Python 片段
另外抽出来在合成数据上跑。文件头里那一行启动命令也原样粘出来演练（重复粘贴、kill -9、关掉终端、脚本以非 0
结束、锁本身加不上）。另有一个用例不用假的 python：rft_eval_probe 真的去调 eval_vllm.py（服务是本机的假服务、
判分是真的）。
全程不碰真的显卡和服务。
"""
import contextlib
import hashlib
import json
import os
import pty
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
RUN, ROUND1, TASKS = "data/rft-guided", "data/rft-round1", "data/frozen-ladder.jsonl"
ZERO_PASS = ["h3_t1", "h3_t2", "h3_t3", "h3v_t1", "h3v_t2", "v2_t1"]   # 上一轮零通过的题；v2_t1 是原有题
BASE = ["h3_t9", "v2_t2"]                # 第 1 轮验证组的 train：一道难题、一道原有题
DEV = ["h3_d1", "h3v_d1", "v2_d1"]
PROBE = [task for task in ZERO_PASS if task != "v2_t1"]          # 默认的层（难题、变体）里的探针题
UNMATCHABLE = "h3v_t2"                   # 上一轮的采样里它没有一份写完了的答案时配不上（见 make_unmatchable）

PYTHON = r'''#!/usr/bin/env python3
"""假的 python：run_rft_guided.sh 调到的每个脚本都写出像样的产物，并把参数记进 STUB_LOG；
内嵌的 Python 片段和 vllm_summary.py 交给真解释器。"""
import hashlib, json, os, random, shutil, signal, sys, time
from pathlib import Path

args = sys.argv[1:]


def value(flag, default=None):
    return args[args.index(flag) + 1] if flag in args else default


def values(flag):
    """能带多个值的参数（--input 甲 乙 丙）的各个值：到下一个参数或结尾为止。"""
    items, take = [], False
    for item in args:
        take = item == flag or (take and not item.startswith("--"))
        if take and item != flag:
            items.append(item)
    return items


def setting(name, default=None):
    """演练里控制假脚本的开关：环境变量 STUB_<name>。"""
    return os.environ.get("STUB_" + name, default)


def log(name):
    with open(os.environ["STUB_LOG"], "a", encoding="utf-8") as handle:
        handle.write(name + " " + " ".join(args) + "\n")


def write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def dump(path, records):
    write(path, "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))


def rows(path):
    """JSONL 每行一条记录；没写完的坏行、截在汉字中间的坏字节都跳过（同真的脚本）。"""
    path = Path(path)
    if not path.is_file():
        return []
    found = []
    for line in path.read_text(encoding="utf-8", errors="replace").split("\n"):
        if not line.strip():
            continue
        try:
            found.append(json.loads(line))
        except ValueError:
            pass
    return found


def tier_of(task):
    return "variant" if task.startswith("h3v_") else "hard" if task.startswith("h3_") else "default"


def refuse_nonempty(path):
    if path.exists() and any(path.iterdir()):     # 和真的一样：只往空目录里写
        sys.exit("output directory must be empty to protect frozen experiments")


def attempts(name):
    """这个假脚本是第几次被调用，记在 STUB_STATE 里。"""
    counter = Path(os.environ["STUB_STATE"]) / name
    seen = int(counter.read_text()) if counter.exists() else 0
    counter.parent.mkdir(parents=True, exist_ok=True)
    counter.write_text(str(seen + 1))
    return seen + 1


def failure(name):
    """STUB_FAIL="名字=退出码,…"里给这次调用安排的退出码（0 表示照常）。名字可以写成"名字.N"：只在第 N 次
    调用时生效；"名字.<N"：前 N-1 次调用都生效。"""
    count = attempts(name)
    for item in setting("FAIL", "").split(","):
        target, _, code = item.partition("=")
        if not code or not target.startswith(name):
            continue
        rest = target[len(name):]
        if rest == "" or (rest.startswith(".") and rest[1:].isdigit() and int(rest[1:]) == count) \
                or (rest.startswith(".<") and count < int(rest[2:])):
            return int(code)
    return 0


def kill_server(tag):
    """STUB_KILL_SERVER="名字=次数,…"：前几次调到这个假脚本时把假 vllm 关掉（模拟它中途退出）。"""
    wanted = int(dict(item.split("=") for item in setting("KILL_SERVER", "").split(",") if item).get(tag, 0))
    counter = Path(os.environ["STUB_STATE"]) / ("kills-" + tag)
    seen = int(counter.read_text()) if counter.exists() else 0
    counter.parent.mkdir(parents=True, exist_ok=True)
    counter.write_text(str(seen + 1))
    if seen >= wanted:
        return False
    pid_file = Path(os.environ["FAKE_VLLM_PID_FILE"])
    for pid in pid_file.read_text().split() if pid_file.exists() else []:
        try:
            os.kill(int(pid), signal.SIGKILL)
        except OSError:
            pass                                   # 以前几次演练留下的 PID，早就不在了
    return True


def note_server(name):
    """记下这个假脚本被调用时有没有假 vllm 在跑：编排该开服务的时候开了、该关的时候关了。"""
    pid_file = Path(os.environ["FAKE_VLLM_PID_FILE"])
    running = []
    for pid in pid_file.read_text().split() if pid_file.exists() else []:
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except OSError:
            continue
        if state != "Z":
            running.append(pid)
    counter = Path(os.environ["STUB_STATE"]) / "servers"
    counter.parent.mkdir(parents=True, exist_ok=True)
    with counter.open("a", encoding="utf-8") as handle:
        handle.write(f"{name}:{' '.join(running)}\n")


if args[:1] == ["src/agentlab/experiments/run_state.py"]:
    print("org/model" if args[1] == "default-model" else "rev-123")
elif args[0].endswith("write_skills.py"):
    log("skills")
    note_server("skills")
    zero_pass = json.loads(Path(value("--zero-pass-from")).read_text(encoding="utf-8"))["zero_pass_tasks"]
    tiers = set(value("--tiers").split(","))
    output = Path(value("--output"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.touch()                                           # 和真的一样：文件先建出来（可以是空的）
    if setting("SKILLS_EXIT") == "2":                        # 环境不对（没有密钥、没装 SDK）：读不了题文件那一类，
        print("write_skills.py: error: 环境变量 TEACHER_API_KEY 没有设置（假的）", file=sys.stderr)
        sys.exit(2)                                          # 还没整理已有的输出就以 2 退出
    settled = {row["task_id"] for row in rows(output) if row.get("status") in ("ok", "leaky")}
    ok = int(setting("SKILLS_OK", "99"))
    written = 0
    with output.open("a", encoding="utf-8") as handle:       # 追加写：有定论的题不再写
        for index, task in enumerate(task for task in zero_pass if tier_of(task) in tiers):
            if task in settled:
                continue
            written += 1
            status = "ok" if index < ok else ("error" if setting("SKILLS_EXIT") else "leaky")
            row = {"task_id": task, "status": status, "rounds": 1}
            if status == "ok":
                row["text"] = f"技能 {index}：先核对输入的边界情况，再比较改动前后的行为。"
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"写技能：待写 {written} 道（假的）")
    print(f"结果：技能提示 可用 {ok} 道（假的）")
    sys.exit(int(setting("SKILLS_EXIT", "0")))
elif args[0].endswith("sample_rft.py"):
    tag = value("--tag")
    log("sample")
    note_server("sample." + tag)
    if setting("HOLD"):                        # STUB_HOLD="秒"：这次采样停一会儿（演练中途 kill 掉脚本）
        Path(os.environ["STUB_STATE"], "script.pid").write_text(str(os.getppid()))
        time.sleep(float(setting("HOLD")))
        if os.getppid() == 1:                  # 启动它的脚本已经不在了：不往它的目录里写东西
            sys.exit(1)
    out = Path(value("--output-dir"))
    # 和真的一样：采样协议先写下，续跑时核对，对不上就以 2 退出（技能引导的协议里有技能文件的摘要）
    protocol = {"kind": value("--hint-kind"), "samples": int(value("--samples"))}
    if protocol["kind"] == "prefix":
        protocol["fractions"] = value("--hint-fractions")
    else:
        protocol["skills_sha256"] = hashlib.sha256(Path(value("--skills-file")).read_bytes()).hexdigest()
    protocol_path = out / f"train-{tag}-protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        print("sample_rft.py: error: 采样协议与这个目录里已有的不一致；换一个新的 --output-dir", file=sys.stderr)
        sys.exit(2)
    write(protocol_path, json.dumps(protocol))
    if kill_server("sample." + tag):                # vLLM 在采样中途退出了
        print("有 3 个槽位没能生成，原样重跑会只补这些（假的：vLLM 不在了）", file=sys.stderr)
        sys.exit(3)
    code = failure("sample." + tag)
    if code:
        print("有 3 个槽位没能生成，原样重跑会只补这些（假的）", file=sys.stderr)
        sys.exit(code)                                       # 没采完：和真的一样不写 manifest
    zero_pass = json.loads(Path(value("--only-zero-pass")).read_text(encoding="utf-8"))["zero_pass_tasks"]
    tasks = [task for task in zero_pass if tier_of(task) in value("--tiers").split(",")]
    per_task = int(value("--samples"))
    if value("--skills-file"):
        usable = {row["task_id"] for row in rows(value("--skills-file"))
                  if row.get("status") == "ok" and row.get("text")}
        tasks = [task for task in tasks if task in usable]
    else:
        per_task *= len(value("--hint-fractions").split(","))
    solved = {"guided-prefix": ["h3_t1", "h3_t2", "h3v_t2"], "guided-skill": ["h3_t3", "h3v_t1", "h3v_t2"]}[tag]
    if setting("SOLVED") is not None:                        # STUB_SOLVED=""：演"引导也没解出新题"
        solved = []
    if tag == "guided-skill" and setting("SOLVED_SKILL") is not None:
        solved = setting("SOLVED_SKILL").split(",")          # STUB_SOLVED_SKILL="题号,题号"：技能引导只解出这几道
    stem = f"train-{tag}"
    graded = []
    for task in tasks:
        for index in range(per_task):
            passed = task in solved and index == 0
            graded.append({
                "record_type": "candidate" if passed else "attempt", "sample_id": f"rft-{task}-{index:03d}",
                "task_id": task, "candidate_idx": index, "source_group": f"group-{task}", "split": "train",
                "messages": [{"role": "user", "content": f"题目 {task}"}], "finish_reason": "stop",
                "target": {"role": "assistant",
                           "content": "def fixed():\n    pass\n" if passed else "def wrong():\n    return 1\n"},
                "provenance": {"generator": "self", "backend": "vllm",
                               "verdict": "pass" if passed else "private_test_failure", "strict_pass": passed,
                               "guided": {"kind": value("--hint-kind")}}})
    dump(out / f"{stem}-raw.jsonl", graded)
    dump(out / f"{stem}-graded.jsonl", graded)
    dump(out / f"{stem}-candidates.jsonl", [row for row in graded if row["provenance"]["strict_pass"]])
    # 和真的一样：清单里记着槽数、读得回的采样数、判分数（调用方 rft_sampling_complete 核对这三个数）
    write(out / f"{stem}-manifest.json", json.dumps({
        "tasks": len(tasks), "slots": len(graded), "raw_records": len(graded), "graded_records": len(graded),
        "candidates": sum(row["provenance"]["strict_pass"] for row in graded),
        "zero_pass_tasks": sorted(set(tasks) - set(solved))}))
    print(f"结果：{stem} 引导 解出 {len(set(tasks) & set(solved))}/{len(tasks)} 题（假的）")
elif args[0].endswith("prepare_data.py"):
    log("prepare")
    code = failure("prepare")
    if code:
        sys.exit("假的 prepare_data.py：重放验证出错")
    out = Path(value("--output-dir"))
    refuse_nonempty(out)
    specs = {spec["task_id"]: spec for spec in rows(value("--tasks-file"))}
    drop = value("--drop-tasks-file")
    dropped = {line.strip() for line in Path(drop).read_text(encoding="utf-8").splitlines() if line.strip()} if drop \
        else set()
    kept, seen, removed = {"train": [], "dev": []}, set(), set()
    for name in values("--input"):
        for row in rows(name):
            failing = row.get("record_type") in ("error", "attempt") \
                or row.get("provenance", {}).get("strict_pass") is not True
            if failing:
                continue
            task = row["task_id"]
            if task in dropped:                               # 按 --drop-tasks-file 挡在外面的题
                removed.add(task)
            if task in dropped or task in seen or task not in specs:   # 每题只留一条
                continue
            seen.add(task)
            kept[specs[task]["split"]].append(row)
    for split, records in kept.items():
        dump(out / f"{split}.jsonl", records)
    write(out / "dataset-manifest.json", json.dumps(
        {"splits": {split: {"samples": len(records)} for split, records in kept.items()},
         "dropped_task_ids": sorted(removed)}))
    print(f"结果：重放验证后 train {len(kept['train'])} 条、dev {len(kept['dev'])} 条（假的）")
elif args[0].endswith("rft_control.py"):
    log("control")
    code = failure("control")
    if code:
        sys.exit(code)
    out = Path(value("--output-dir"))
    refuse_nonempty(out)
    base = Path(value("--base-dir"))
    base_train = rows(base / "train.jsonl")
    base_tasks = {row["task_id"] for row in base_train}
    guided = set()
    for directory in values("--guided-from"):
        guided |= {row["task_id"] for row in rows(Path(directory) / "train.jsonl")} - base_tasks
    if not guided:
        sys.exit("rft_control.py: error: --guided-from 的训练集里没有基础集之外的题")
    pool = {}
    for row in rows(value("--graded")):                      # 能当对照答案的：写完了、格式完整、没判分通过
        if row["task_id"] in guided and row.get("finish_reason", "stop") == "stop" and row["target"]["content"] \
                and row.get("provenance", {}).get("strict_pass") is not True:
            pool.setdefault(row["task_id"], []).append(row)
    chosen = [random.Random(f"42:{task}:failing").choice(pool[task]) for task in sorted(guided) if pool.get(task)]
    if not chosen:
        sys.exit("rft_control.py: error: 引导题一道也配不上")
    unmatched = sorted(guided - {row["task_id"] for row in chosen})
    dump(out / "train.jsonl", base_train + chosen)
    shutil.copyfile(base / "dev.jsonl", out / "dev.jsonl")
    write(out / "unmatched-tasks.txt", "".join(f"{task}\n" for task in unmatched))
    write(out / "dataset-manifest.json", json.dumps({
        "samples": len(chosen), "tasks": len(guided), "unmatched": unmatched, "base_samples": len(base_train),
        "guided_from": values("--guided-from")}))            # 和真的一样：记下是按哪几个引导组配的
    print(f"结果：配对对照 = 基础 {len(base_train)} 条 + 引导题 {len(chosen)} 条；"
          f"配不上的题 {len(unmatched)} 道（假的）")
elif args[0].endswith("train_sft.py"):
    log("train")
    if "--dry-run" in args:
        print("Dry run 完成")
    else:
        if failure("train"):                    # STUB_FAIL="train.1=1"：第一次真训练出错，什么也不写
            sys.exit("训练出错（假的）")
        note_server("train." + Path(value("--output")).name)
        tasks = [row["task_id"] for row in rows(value("--data"))]
        # STUB_DROP_LONG="组名=题号 …"：这一组里这道题的记录超过长度上限，被训练入口丢掉
        dropped = dict(item.split("=") for item in setting("DROP_LONG", "").split() if item)
        lost = dropped.get(Path(value("--output")).name)
        loaded = sorted({task for task in tasks if task != lost})
        write(Path(value("--output")) / "training-manifest.json", json.dumps({
            "status": "complete", "model": value("--model"), "revision": value("--revision"),
            "data": {"total": len(tasks), "loaded": len(loaded), "task_ids": loaded}}))
        print("训练完成")
elif args[0].endswith("select_checkpoint.py"):
    log("select")
    out = Path(value("--output"))
    out.mkdir(parents=True, exist_ok=True)
    adapter = Path(value("--checkpoint-root")) / "epoch-001/adapter"
    write(adapter / "adapter_config.json", "{}")
    if value("--base-report"):
        shutil.copyfile(value("--base-report"), out / "epoch-000-dev.json")
    else:
        write(out / "epoch-000-dev.json", '{"adapter": null}')
    write(out / "epoch-001-dev.json", json.dumps({"adapter": str(adapter)}))
    write(out / "selection.json", json.dumps({
        "selected_epoch": 1, "selected_adapter": str(adapter), "base_pass_at_1": 0.439,
        "configuration": {"model": value("--model"), "revision": value("--revision")},
        "candidates": [{"epoch": 0, "adapter": None, "pass_at_1": 0.439, "improved_tasks": 0, "regressed_tasks": 0},
                       {"epoch": 1, "adapter": str(adapter), "pass_at_1": 0.5, "improved_tasks": 5,
                        "regressed_tasks": 1}]}))
elif args[0].endswith("tier_report.py"):
    log("tiers")
    print("结果：（假的分层表）")
elif args[0].endswith("eval_vllm.py"):
    log("eval")
    note_server("eval")
    label, stem = value("--label"), value("--tag") or value("--split")
    out, samples = Path(value("--output-dir")), int(value("--samples", "8"))
    code = int(dict(item.split("=") for item in setting("EVAL_FAIL", "").split(",") if item).get(f"{label}/{stem}", 0))
    if code:
        print(f"假的 eval_vllm.py：{label} {stem} 没评完，以退出码 {code} 结束")
        sys.exit(code)                                       # 没评完：什么都不写
    sanity = None
    if value("--sanity-tasks-file"):
        # 和真的一样（新判据）：自检拿"探针题里这一组训练过的题"（最多 6 道）比较适配器和基座的输出，不同的不到
        # --sanity-min（默认 1）就以 4 退出。STUB_SANITY="组名=几道和基座不同" 可以把这个数压低；不给就是都不同。
        # 训练过的题严格通过几道只作参考（这里一律记 0）
        probe = {line.strip() for line in Path(value("--ids-file")).read_text(encoding="utf-8").splitlines()}
        trained = {row["task_id"] for row in rows(value("--sanity-tasks-file"))}
        count = min(6, len(probe & trained))
        changed = int(dict(item.split("=") for item in setting("SANITY", "").split(",") if item).get(label, count))
        sanity = {"pass": 0, "total": count, "min": int(value("--sanity-min", "1")), "changed": changed,
                  "compared": True, "compared_tasks": count, "not_compared_reason": None}
        write(out / f"{label}-{stem}-sanity.json", json.dumps(sanity))
        if changed < sanity["min"]:
            print("适配器在这几道题上的输出和基座完全一样，适配器在 vLLM 里没有生效")
            sys.exit(4)                                      # 自检没过：和真的一样不写报告和 summary
    tasks = [(spec["task_id"], (len(spec["task_id"]) + len(stem)) % 2 == 0)
             for spec in rows(value("--tasks-file")) if spec["split"] == value("--split")]
    if value("--ids-file"):
        wanted = {line.strip() for line in Path(value("--ids-file")).read_text(encoding="utf-8").splitlines()}
        tasks = [row for row in tasks if row[0] in wanted]
    prefix = out / f"{label}-{stem}"
    write(str(prefix) + "-greedy.json", json.dumps(
        {"results": [dict(task_id=task, strict_pass=passed) for task, passed in tasks]}))
    tiers = {tier: [0, 0] for tier in ("default", "hard", "variant")}
    for task, passed in tasks:
        tiers[tier_of(task)][0] += bool(passed)
        tiers[tier_of(task)][1] += 1
    summary = {"label": label, "model": value("--model"), "split": value("--split"), "tag": stem, "sanity": sanity,
               "greedy": {"pass": sum(row[0] for row in tiers.values()), "total": len(tasks), "by_tier": tiers},
               "sampled": None}
    if samples:
        summary["sampled"] = {"samples_per_task": samples, "attempts": samples * len(tasks),
                              "passed_attempts": 1 if tasks else 0,
                              "pass_at_1": 1 / (samples * len(tasks)) if tasks else 0, "any_pass_rate": 0.5}
        write(str(prefix) + f"-sampled-k{samples}.json", json.dumps({"results": []}))
    write(str(prefix) + "-summary.json", json.dumps(summary, ensure_ascii=False))
    print(f"结果：{label} {stem} 贪心 {summary['greedy']['pass']}/{len(tasks)}（假的）")
elif args[0].endswith("vllm_summary.py"):
    log("summary")
    os.execv(sys.executable, [sys.executable, *args])        # 真的 vllm_summary.py 读上面写出的产物
else:
    os.execv(sys.executable, [sys.executable, *args])        # 内嵌的 Python 片段：真解释器从 stdin 读
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
# 假的 nvidia-smi：第一块显卡已用的显存取自 FAKE_GPU_USED（MiB），没设就是查不到；
# 显存超过 2000 MiB 时列出占着显卡的进程
used="${FAKE_GPU_USED:-}"
case "$*" in
  *--query-gpu=memory.used*) echo "$used" ;;
  *--query-compute-apps*) if [[ "$used" =~ ^[0-9]+$ ]] && (( used > 2000 )); then echo "12345, ${used} MiB"; fi ;;
esac
exit 0
'''

PKILL = r'''#!/usr/bin/env bash
# 假的 pkill -f：照正则匹配整条命令行，但只对命令行里带着这次演练目录（STAGE）的进程动手，
# 自己除外（真的 pkill 也不会关掉自己）。用真的 pkill 会连这台机器上别的 vllm serve 一起关掉
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

SLEEP = r'''#!/usr/bin/env bash
# 演练用的 sleep：启动 vLLM 前固定等的那 3 秒、采样重试前的 5 秒、查日志的间隔 1 秒都不必真等，其余照常
case "$*" in 1|3) exec @REAL@ 0.05 ;; 5) exec @REAL@ 0.2 ;; esac
exec @REAL@ "$@"
'''

SWITCHES = ("MODEL", "MODEL_REVISION", "TASKS_FILE", "RUN_DIR", "ROUND1_DIR", "ROUND1_PACKAGE", "RFT_TIERS",
            "HINT_FRACTIONS", "RFT_SAMPLES", "SKILL", "SKILL_SAMPLES", "MIN_SKILL_TASKS", "VLLM_EVAL", "EVAL_SAMPLES",
            "RFT_LR", "RFT_LORA_R", "RFT_LORA_ALPHA", "RFT_EPOCHS", "GRAD_ACCUM", "MAX_LENGTH", "VLLM_BIN", "VLLM_PORT",
            "RFT_CONCURRENCY", "RFT_GRADE_WORKERS", "RFT_POLL_SECONDS", "BASE_REPORT", "TEACHER_API_KEY",
            "EXPECT_REPRODUCE", "PROBE_IDS_FILE", "INCLUDE_BASE", "TEACHER_CONCURRENCY", "TEACHER_MIN_INTERVAL",
            "TEACHER_MAX_TOKENS", "SKILL_JUDGE")
STEP = re.compile(r"^===== (.*?) *=====$", re.M)      # 右边"====="前面有没有空格都认（嵌在里面的脚本两种都有）
HEREDOC = re.compile(r"python - (.*?) <<'PY'(?:[^\n]*\\\n[^\n]*)*[^\n]*\n(.*?)\nPY\n", re.S)
SAMPLING_START = "serve org/model --revision rev-123 --language-model-only --max-model-len 8192 --port 8000"
SUMMARY_HEAD = "scripts/tier_report.py data/rft-round1/results/A-verified data/rft-round1/results/C-control "
NO_PROBE = ("上一轮一次都没通过的 1 道题里没有属于 hard、variant 层的：没有可引导的题，这一轮做不了。"
            "RFT_TIERS 写的是哪些层？")
MISMATCH = "结果：训练时实际加载的题 D-prefix、D-skill 合起来和 F-control 不一样：只在 D-prefix、D-skill 里的 "
LOST_TAIL = "多半是有记录超过 --max-length 被训练入口丢掉了，比较 D 组和 F-control 时要看住这几道题"
ALL_STEPS = [
    "1/13 检查前提",
    "2/13 写探针题号文件（上一轮零通过的题，之后经 vLLM 评适配器时拿它们做族内探针）",
    "3/13 启动 vLLM 并先评基座",
    "4/13 教师给零通过的题写技能提示",
    "5/13 前缀引导采样（参考解的开头按 0.25,0.5,0.75 各给一部分，每种 4 次）",
    "6/13 技能引导采样（每题 8 次）",
    "7/13 关闭 vLLM，等显存释放",
    "8/13 准备训练数据（第 1 轮的验证组 + 引导出的正确答案，以及逐题配对的对照）",
    "9/13 D-prefix：训练（lr 1e-4、秩 32）并在 dev 上选型",
    "10/13 D-skill：训练（lr 1e-4、秩 32）并在 dev 上选型",
    "11/13 F-control：训练（lr 1e-4、秩 32）并在 dev 上选型",
    "12/13 汇总主结果（都在 dev 上；test 这一轮不评）",
    "13/13 经 vLLM 评各组选中的适配器（辅助指标；主结果是上面 HF 贪心那张表）",
]


def install(home, folder, name, text):
    program = home / folder / name
    program.parent.mkdir(parents=True, exist_ok=True)
    program.write_text(text, encoding="utf-8")
    program.chmod(0o755)


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def split_of(task):
    return "dev" if task in DEV else "train"


def solved(task):
    """一条准备好的训练记录（第 1 轮验证组里的样子）。"""
    return {"record_type": "candidate", "task_id": task, "source_group": f"group-{task}", "split": split_of(task),
            "messages": [{"role": "user", "content": f"题目 {task}"}],
            "target": {"role": "assistant", "content": "def fixed():\n    pass\n"},
            "provenance": {"strict_pass": True, "verdict": "pass"}}


def write_round1(repo):
    """第 1 轮留下来的东西：任务文件、采样的清单和判分记录、验证组的训练集、A-verified 和 C-control 的选型结果。"""
    train = [*ZERO_PASS, *BASE]
    where = [(task, "train") for task in train] + [(task, "dev") for task in DEV]
    write(repo / TASKS, "".join(json.dumps({"task_id": task, "split": split}) + "\n" for task, split in where))
    sampling = repo / ROUND1 / "sampling"
    write(sampling / "train-manifest.json", {"zero_pass_tasks": sorted(ZERO_PASS)})
    write(sampling / "train-graded.jsonl", "".join(json.dumps({
        "record_type": "attempt", "task_id": task, "source_group": f"group-{task}", "split": "train",
        "finish_reason": "stop",
        "target": {"role": "assistant", "content": "def wrong():\n    return 1\n"},
        "provenance": {"strict_pass": False, "verdict": "private_test_failure"}}) + "\n"
        for task in ZERO_PASS for _ in range(8)))
    verified = repo / ROUND1 / "sft/verified"
    write(verified / "train.jsonl", "".join(json.dumps(solved(task)) + "\n" for task in BASE))
    write(verified / "dev.jsonl", "".join(json.dumps(solved(task)) + "\n" for task in DEV))
    for group in ("A-verified", "C-control"):
        results = repo / ROUND1 / "results" / group
        write(results / "selection.json", {"selected_epoch": 1, "candidates": [{"epoch": 0}, {"epoch": 1}]})
        write(results / "epoch-000-dev.json", {"results": []})
        write(results / "epoch-001-dev.json", {"results": []})


def processes(stage):
    """这次演练里假 vllm 每次启动时的 PID。"""
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
    """等 condition 成立（假的 vllm 起来、被关掉都要一点时间）；到时间还不成立就返回它最后的值。"""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


@pytest.fixture
def stage(tmp_path):
    """一份临时的仓库（编排脚本和它用到的库）、第 1 轮的产物、假的 python、vllm、nvidia-smi、pkill、sleep。"""
    home = tmp_path / "home"
    (home / "repo/scripts").mkdir(parents=True)
    for name in ("run_rft_guided.sh", "rft_lib.sh", "rft_vllm_eval.sh", "vllm_summary.py"):
        shutil.copy(ROOT / "scripts" / name, home / "repo/scripts" / name)
    install(home, "bin", "python", PYTHON)
    install(home, "bin", "nvidia-smi", NVIDIA_SMI)
    install(home, "bin", "pkill", PKILL)
    install(home, "bin", "sleep", SLEEP.replace("@REAL@", shutil.which("sleep")))
    install(home, "vllm-qwen/bin", "vllm", VLLM)
    write_round1(home / "repo")
    yield home
    for pid in processes(home):                     # 断言失败时也不把假的 vllm 留在机器上
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, signal.SIGKILL)


def environment(stage, **extra):
    """外面环境里跟这次演练有关的变量一律不带进来，只用 extra 里给的。"""
    clean = {name: value for name, value in os.environ.items()
             if name not in SWITCHES and not name.startswith(("STUB_", "FAKE_"))}
    return {**clean, "HOME": str(stage), "STAGE": str(stage), "STUB_LOG": str(stage / "calls.log"),
            "STUB_STATE": str(stage / "state"), "FAKE_VLLM_LOG": str(stage / "vllm.log"),
            "FAKE_VLLM_PID_FILE": str(stage / "vllm.pid"), "RFT_POLL_SECONDS": "1",
            "PATH": f"{stage / 'bin'}{os.pathsep}{os.environ['PATH']}", **extra}


def calls_in(stage):
    """假 python 在 calls.log 里记下的调用：[(脚本名, 参数)]。"""
    log = stage / "calls.log"
    return [(parts[0], parts[1] if len(parts) > 1 else "")
            for parts in (line.split(" ", 1) for line in log.read_text(encoding="utf-8").splitlines())] \
        if log.exists() else []


def run(stage, **extra):
    """跑一遍 run_rft_guided.sh。返回 (结果, 假 python 记下的调用 [(脚本名, 参数)])。"""
    for name in ("calls.log", "vllm.log"):                   # 每次只看这一次的调用和启动
        (stage / name).unlink(missing_ok=True)
    result = subprocess.run(["bash", "scripts/run_rft_guided.sh"], cwd=stage / "repo", env=environment(stage, **extra),
                            capture_output=True, text=True, timeout=300)
    # 有 Traceback 就是有什么地方崩了而编排没有接住：哪怕这次演练预期它以非 0 退出，也不该是这样退出的
    assert "Traceback" not in result.stderr, result.stderr[-2000:]
    return result, calls_in(stage)


def calls_of(calls, name):
    """名字是这个的各次调用的参数。"""
    return [args for called, args in calls if called == name]


def flag(args, name):
    found = re.search(r"--" + name + r" (\S+)", args)
    return found.group(1) if found else None


def steps(result):
    return STEP.findall(result.stdout)


def own_steps(result, total):
    """编排脚本自己的步骤（总步数是 total 的那些；嵌在里面那次 vLLM 评测的总步数是 8）。"""
    return [step for step in steps(result) if re.match(rf"\d+/{total} ", step)]


def starts(stage):
    """假 vllm 每次被启动时收到的参数。"""
    log = stage / "vllm.log"
    return log.read_text(encoding="utf-8").splitlines() if log.exists() else []


def servers(stage):
    """[(哪个假脚本被调用, 当时在跑的假 vllm 的 PID)]。"""
    path = stage / "state/servers"
    return [line.split(":") for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def evaluations(calls):
    """每次 eval_vllm.py 调用的要点：(标签, stem, 采样次数, 自检用的训练数据)。"""
    return [(flag(args, "label"), flag(args, "tag") or flag(args, "split"), flag(args, "samples"),
             flag(args, "sanity-tasks-file")) for args in calls_of(calls, "eval")]


def task_ids(path):
    """一个 JSONL 里的题号，按出现的先后。"""
    return [json.loads(line)["task_id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if path.exists() else None


def tasks_of(repo, group):
    return task_ids(repo / RUN / "sft" / group / "train.jsonl")


def assert_clean_shutdown(stage):
    assert eventually(lambda: not alive(stage)), "假的 vllm 还在跑：用完没有关掉"
    for name, pids in servers(stage):
        kind = name.split(".")[0]
        if kind in ("sample", "eval"):
            assert pids, name                        # 采样和评测时服务开着
        elif kind == "train":
            assert not pids, name                    # 训练时服务已经关掉（写技能走的是教师接口，不用管服务）


def test_shell_scripts_parse():
    for name in ("run_rft_guided.sh", "run_rft.sh"):
        assert subprocess.run(["bash", "-n", str(ROOT / "scripts" / name)], capture_output=True).returncode == 0


def test_a_fresh_run_does_every_step_in_order(stage):
    repo = stage / "repo"
    result, calls = run(stage)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-2000:]
    assert own_steps(result, 13) == ALL_STEPS
    assert steps(result).count("全部结束") == 2 and result.stdout.rstrip().endswith("===== 全部结束 =====")

    # 探针：上一轮零通过的题里属于难题、变体两层的（原有题 v2_t1 不算），一行一个
    assert (repo / RUN / "probe-ids.txt").read_text(encoding="utf-8") == "".join(f"{task}\n" for task in PROBE)
    assert "探针题：上一轮零通过的 train 题里属于 hard、variant 层的共 5 道，写在 data/rft-guided/probe-ids.txt" \
        in result.stdout

    assert [name for name, _ in calls] == [
        "eval", "eval", "skills", "sample", "sample", "prepare", "prepare", "control",
        "train", "train", "select", "tiers", "train", "train", "select", "tiers", "train", "train", "select", "tiers",
        "tiers", "eval", "eval", "eval", "eval", "eval", "eval", "summary"]
    skills, = calls_of(calls, "skills")
    assert skills == ("scripts/write_skills.py --tasks-file data/frozen-ladder.jsonl "
                      "--zero-pass-from data/rft-round1/sampling/train-manifest.json "
                      "--split train --tiers hard,variant "
                      "--graded data/rft-round1/sampling/train-graded.jsonl --output data/rft-guided/skills.jsonl "
                      "--concurrency 2 --min-interval 1 --max-tokens 4096 --judge")     # 默认值保守：限额按账户算
    sampling = ("scripts/sample_rft.py --tasks-file data/frozen-ladder.jsonl --split train --tiers hard,variant "
                "--vllm-url http://127.0.0.1:8000 --model org/model --concurrency 24 --grade-workers 4 "
                "--output-dir data/rft-guided/sampling ")
    zero_pass = "--only-zero-pass data/rft-round1/sampling/train-manifest.json"
    assert calls_of(calls, "sample") == [
        sampling + "--hint-kind prefix --hint-fractions 0.25,0.5,0.75 --samples 4 " + zero_pass
        + " --tag guided-prefix",
        sampling + "--hint-kind skill --samples 8 --skills-file data/rft-guided/skills.jsonl " + zero_pass
        + " --tag guided-skill"]
    verified = "data/rft-round1/sft/verified"
    prepare = ("scripts/prepare_data.py --tasks-file data/frozen-ladder.jsonl "
               f"--input {verified}/train.jsonl {verified}/dev.jsonl ")
    assert calls_of(calls, "prepare") == [
        prepare + "data/rft-guided/sampling/train-guided-prefix-candidates.jsonl "
                  "--output-dir data/rft-guided/sft/D-prefix --max-per-task 1",
        prepare + "data/rft-guided/sampling/train-guided-skill-candidates.jsonl "
                  "--output-dir data/rft-guided/sft/D-skill --max-per-task 1"]
    control, = calls_of(calls, "control")
    assert control == ("scripts/rft_control.py --graded data/rft-round1/sampling/train-graded.jsonl "
                       "--base-dir data/rft-round1/sft/verified --guided-from data/rft-guided/sft/D-prefix "
                       "data/rft-guided/sft/D-skill --output-dir data/rft-guided/sft/F-control --seed 42")

    # 三组的顺序和超参一样；--allow-unverified 只给了对照组；基座的 dev 报告用第 1 轮 A-verified 评出的那份
    trains = [args for args in calls_of(calls, "train") if "--dry-run" not in args]
    assert [flag(args, "output") for args in trains] == [f"data/rft-guided/checkpoints/{group}"
                                                       for group in ("D-prefix", "D-skill", "F-control")]
    assert [flag(args, "data") for args in trains] == [f"data/rft-guided/sft/{group}/train.jsonl"
                                                     for group in ("D-prefix", "D-skill", "F-control")]
    for args in trains:
        assert "--model org/model --revision rev-123" in args
        assert ("--epochs 1 --lr 1e-4 --lora-r 32 --lora-alpha 64 --batch-size 1 --grad-accum 8 --max-length 2048 "
                "--seed 42") in args
    assert ["--allow-unverified" in args for args in trains] == [False, False, True]
    selects = calls_of(calls, "select")
    assert [flag(args, "output") for args in selects] == [f"data/rft-guided/results/{group}"
                                                        for group in ("D-prefix", "D-skill", "F-control")]
    assert all(args.endswith("--seed 42 --resume --base-report data/rft-round1/results/A-verified/epoch-000-dev.json")
               for args in selects)
    assert "===== 9/13 D-prefix：训练（lr 1e-4、秩 32）并在 dev 上选型 =====" in result.stdout

    # 每份数据一行结论；三组各一行
    for line in (
            "结果：D-prefix 训练集 5 条 / 5 题，其中引导题 3 条（难题 2、变体 1）；dev 3 条",
            "结果：D-skill 训练集 5 条 / 5 题，其中引导题 3 条（难题 1、变体 2）；dev 3 条",
            "结果：F-control 训练集 7 条 / 7 题，其中引导题 5 条（难题 3、变体 2）；dev 3 条",
            "结果：D-prefix dev 0.439 → 0.500（选中 epoch 1；变好 5 道、变差 1 道）",
            "结果：D-skill dev 0.439 → 0.500（选中 epoch 1；变好 5 道、变差 1 道）",
            "结果：F-control dev 0.439 → 0.500（选中 epoch 1；变好 5 道、变差 1 道）",
            "结果：训练时实际加载的题 D-prefix、D-skill 和 F-control 逐题相同（7 道）"):
        assert line in result.stdout, line
    assert result.stdout.count("结果：") >= 6
    assert tasks_of(repo, "F-control") == [*BASE, *PROBE]

    # 汇总：第 1 轮的两组在前，后面是这一轮选完的三组
    assert calls_of(calls, "tiers")[-1] == SUMMARY_HEAD + (
        "data/rft-guided/results/D-prefix data/rft-guided/results/D-skill data/rft-guided/results/F-control")

    # 服务开了两次：采样时只有基座；评测时带着三组的 LoRA
    adapters = f"{repo}/data/rft-guided/checkpoints"
    assert starts(stage) == [
        SAMPLING_START,
        SAMPLING_START + " --enable-lora --max-lora-rank 64 --lora-modules "
        + " ".join(f"{group}={adapters}/{group}/epoch-001/adapter" for group in ("D-prefix", "D-skill", "F-control"))]
    # 先评基座（dev 加探针）；再逐组先探针后 dev。EXPECT_REPRODUCE 是两个 D 组：只有它们带自检用的训练数据
    assert evaluations(calls) == [
        ("base", "dev", "4", None), ("base", "probe", "0", None),
        ("D-prefix", "probe", "0", f"{RUN}/sft/D-prefix/train.jsonl"), ("D-prefix", "dev", "4", None),
        ("D-skill", "probe", "0", f"{RUN}/sft/D-skill/train.jsonl"), ("D-skill", "dev", "4", None),
        ("F-control", "probe", "0", None), ("F-control", "dev", "4", None)]
    for args in calls_of(calls, "eval"):
        assert "--tasks-file data/frozen-ladder.jsonl --vllm-url http://127.0.0.1:8000 " in args
        assert "--grade-workers 4" in args and "--label " in args
    assert f"族内探针：{RUN}/probe-ids.txt；评探针前要自检的组：D-prefix D-skill" in result.stdout
    assert calls_of(calls, "summary") == [f"scripts/vllm_summary.py --run {RUN} --arms D-prefix D-skill F-control "
                                          f"--probe-ids-file {RUN}/probe-ids.txt"]
    assert "── vLLM 评测（bf16 权重，辅助指标；主结果是上面 HF 贪心那张表） ──" in result.stdout   # 汇总是真的
    assert_clean_shutdown(stage)


def test_rerunning_skips_everything_that_is_done(stage):
    first, calls = run(stage)
    assert first.returncode == 0, first.stdout[-2000:] + first.stderr[-1000:]
    assert starts(stage)                                   # 第一次开了服务（采样一次、评测一次）
    skills = (stage / "repo" / RUN / "skills.jsonl").read_bytes()
    again, calls = run(stage)
    assert again.returncode == 0, again.stdout[-2000:] + again.stderr[-1000:]
    # 只剩下打印结论。技能引导采样已经开始过，技能文件就不再动：连 write_skills.py 都不调了
    assert [name for name, _ in calls] == ["tiers", "tiers", "tiers", "tiers", "summary"]
    assert "技能引导采样已经开始过：技能文件保持原样，这次不再调用教师" in again.stdout
    assert (stage / "repo" / RUN / "skills.jsonl").read_bytes() == skills
    assert starts(stage) == [] and "===== 3/8 检查显卡：都评过了，不用启动 vLLM，跳过 =====" in again.stdout
    skipped = [step for step in own_steps(again, 13) if "已完成，跳过" in step]
    assert skipped == ["3/13 启动 vLLM：已完成，跳过", "5/13 前缀引导采样：已完成，跳过",
                       "6/13 技能引导采样：已完成，跳过", "7/13 关闭 vLLM：已完成，跳过"]
    assert again.stdout.count("已经选过型，跳过") == 3
    for line in ("D-prefix 的训练数据已经有了，跳过准备", "D-skill 的训练数据已经有了，跳过准备",
                 "F-control 的训练数据已经有了，跳过准备"):
        assert line in again.stdout, line
    assert again.stdout.count("结果：D-prefix 训练集") == 1              # 跳过了也把结论重新打印一遍
    assert steps(again)[-1] == "全部结束"


def test_a_sampling_manifest_whose_records_do_not_add_up_is_not_taken_as_done(stage):
    """清单在不等于采样做完了：旧版本的 sample_rft.py 读不回某些采样时照样写清单（那几道题就被记成"一次都没通过"）。
    槽数 ≠ 记录数的清单不算数——重新调用它，只补缺的槽位。"""
    repo = stage / "repo"
    assert run(stage)[0].returncode == 0
    path = repo / RUN / "sampling/train-guided-prefix-manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["slots"] == manifest["raw_records"] == manifest["graded_records"]
    write(path, {**manifest, "raw_records": manifest["slots"] - 2, "graded_records": manifest["slots"] - 2})
    result, calls = run(stage)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-1000:]
    assert [flag(args, "tag") for args in calls_of(calls, "sample")] == ["guided-prefix"]   # 只重做这一份
    assert "上次留下的前缀引导采样清单里槽数和记录数对不上" in result.stdout
    assert "5/13 前缀引导采样：已完成，跳过" not in result.stdout and "6/13 技能引导采样：已完成，跳过" in result.stdout
    assert "===== 3/13 启动 vLLM 并先评基座 =====" in result.stdout     # 前缀采样要补：第 3 步就把服务开起来
    assert len(starts(stage)) == 1 and "7/13 关闭 vLLM，等显存释放 =====" in result.stdout   # 两次采样共用这一次
    again = json.loads(path.read_text(encoding="utf-8"))
    assert again["slots"] == again["raw_records"] == again["graded_records"] == manifest["slots"]
    assert steps(result)[-1] == "全部结束" and assert_clean_shutdown(stage) is None


def test_the_previous_round_sampling_manifest_is_checked_for_lost_slots(stage):
    """引导哪些题、探针清单都从上一轮的清单来：它少记了采样（旧版本的 sample_rft.py 会这样）时要说一声、给出补救的办法，
    不然那几道题就一直被当成"一次都没通过"。清单对得上就什么都不说。"""
    repo = stage / "repo"
    first, _ = run(stage)
    assert first.returncode == 0 and "上一轮的采样清单对不上" not in first.stdout
    path = repo / ROUND1 / "sampling/train-manifest.json"
    write(path, {"zero_pass_tasks": sorted(ZERO_PASS), "slots": 48, "raw_records": 45, "graded_records": 45})
    second, calls = run(stage)
    assert second.returncode == 0, second.stdout[-2000:] + second.stderr[-1000:]
    assert ("注意：上一轮的采样清单对不上——3 个槽位的采样或判分记录读不回来（旧版本的 sample_rft.py 会这样），\n"
            "      那几道题可能被记成「一次都没通过」。重新运行 bash scripts/run_rft.sh 会只补这些槽位，"
            "做完再运行本脚本" in second.stdout)
    assert steps(second)[-1] == "全部结束"                              # 只是提醒：这一轮照常做完


def test_a_server_that_quits_during_sampling_is_restarted_and_the_sampling_goes_on(stage):
    """采样中途 vLLM 退出（显存不足、被系统杀掉）：对着一个不在的服务重试三次没有用，夜里就停在那里了。脚本把服务
    重新启动一次，接着只补没完成的槽位。"""
    result, calls = run(stage, SKILL="0", VLLM_EVAL="0", STUB_KILL_SERVER="sample.guided-prefix=1")
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-1000:]
    assert [flag(args, "tag") for args in calls_of(calls, "sample")] == ["guided-prefix"] * 2
    assert "前缀引导采样没跑完的时候 vLLM 已经不在了。vLLM 日志最后 15 行：" in result.stdout
    assert "INFO:     Application startup complete." in result.stdout      # 服务日志的最后几行原样给用户看
    assert "重新启动 vLLM 后再试（只补没完成的槽位）" in result.stdout
    assert "5 秒后重试" not in result.stdout                             # 不对着不在的服务干等
    assert len(starts(stage)) == 2 and starts(stage)[0] == starts(stage)[1]
    assert "===== 3/9 启动 vLLM 并先评基座 =====" in result.stdout      # 中途退出、重启都记在同一步里
    assert steps(result)[-1] == "全部结束"
    assert_clean_shutdown(stage)


def test_a_server_that_quits_again_during_sampling_stops_with_the_log_tail(stage):
    """重启过一次 vLLM 又退出：不再重启（免得整夜反复加载），停下时带上服务日志的末尾。"""
    result, calls = run(stage, SKILL="0", VLLM_EVAL="0", STUB_KILL_SERVER="sample.guided-prefix=2")
    assert result.returncode == 1, result.stdout[-2000:] + result.stderr[-1000:]
    assert [flag(args, "tag") for args in calls_of(calls, "sample")] == ["guided-prefix"] * 2
    assert len(starts(stage)) == 2
    assert "前缀引导采样没跑完，vLLM 又不在了（已经重新启动过一次），停止。vLLM 日志最后 15 行：" in result.stdout
    assert "===== 5/10" not in result.stdout and "全部结束" not in result.stdout
    assert not (stage / "repo" / RUN / "sft").exists()
    assert_clean_shutdown(stage)


def test_a_rerun_with_a_new_guided_group_drops_the_newly_unmatchable_tasks_without_stopping(stage):
    """上一跑只有 D-prefix，做到"配不上的题"那一步时断了，留下按 D-prefix 配出的清单；这一跑多出 D-skill，它带进一道
    新的配不上的题。旧清单里没有这道题：照它准备 D-skill、重新配对，必然还有配不上的——以前脚本在这里以 1 停下，
    没人重跑就停到天亮，再跑一次才自愈。现在把新出现的题并进清单再来一轮，一次跑完。"""
    repo = stage / "repo"
    make_unmatchable(repo)
    make_unmatchable(repo, "h3_t3")                             # 只有 D-skill 解得出的一道题，也配不上
    first, calls = run(stage, SKILL="0", STUB_FAIL="prepare.2=1")   # D-prefix：对照组配出了配不上的题，重新准备时断了
    assert first.returncode == 1 and "准备 D-prefix 的训练数据失败" in first.stdout
    assert (repo / RUN / "sft/F-control/unmatched-tasks.txt").read_text(encoding="utf-8") == f"{UNMATCHABLE}\n"
    second, calls = run(stage)
    assert second.returncode == 0, second.stdout[-3000:] + second.stderr[-1000:]
    assert [name for name, _ in calls][:9] == ["skills", "sample", "prepare", "prepare", "control",
                                              "prepare", "control", "train", "train"]
    drops = [flag(args, "drop-tasks-file") for args in calls_of(calls, "prepare")]
    assert drops == [f"{RUN}/sft/F-control/unmatched-tasks.txt"] * 3
    assert [flag(args, "output-dir") for args in calls_of(calls, "prepare")] == [
        f"data/rft-guided/sft/{group}" for group in ("D-prefix", "D-skill", "D-skill")]
    assert "重新配对后又出现配不上的题" in second.stdout and "重新配对后新配不上的题：h3_t3 " in second.stdout
    assert "去掉之后还是有配不上的题" not in second.stdout
    # 两道配不上的题两组都不再有，对照组里也没有
    assert tasks_of(repo, "D-prefix") == [*BASE, "h3_t1", "h3_t2"]
    assert tasks_of(repo, "D-skill") == [*BASE, "h3v_t1"]
    assert tasks_of(repo, "F-control") == [*BASE, "h3_t1", "h3_t2", "h3v_t1"]
    assert (repo / RUN / "sft/F-control/unmatched-tasks.txt").read_text(encoding="utf-8") == ""
    assert "结果：训练时实际加载的题 D-prefix、D-skill 和 F-control 逐题相同（5 道）" in second.stdout
    aside = {path.name.split(".unmatched-")[0]: path for path in (repo / RUN / "sft").glob("*.unmatched-*")}
    assert sorted(aside) == ["D-prefix", "D-skill"]              # D-prefix 是上一跑挪开的，D-skill 是第二轮挪开的
    assert aside["D-skill"].name.endswith("-2")                   # 第二轮挪开的目录名带着轮次，不和第一轮的撞
    assert task_ids(aside["D-skill"] / "train.jsonl") == [*BASE, "h3_t3", "h3v_t1"]
    assert steps(second)[-1] == "全部结束"
    third, calls = run(stage)                                  # 再往后重跑：数据都对上了，什么都不重做
    assert third.returncode == 0 and not calls_of(calls, "prepare") and not calls_of(calls, "control")


def test_a_group_of_the_previous_round_without_a_selection_is_named_when_it_is_left_out(stage):
    """汇总只列已经选完的组：没列的要说一声，不能悄悄跳过——按行数着看表的人会把组认错。"""
    repo = stage / "repo"
    (repo / ROUND1 / "results/C-control/selection.json").unlink()
    result, calls = run(stage)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-1000:]
    assert "data/rft-round1/results/C-control：还没有选型结果，不列进汇总" in result.stdout
    assert calls_of(calls, "tiers")[-1] == ("scripts/tier_report.py data/rft-round1/results/A-verified "
                                            "data/rft-guided/results/D-prefix data/rft-guided/results/D-skill "
                                            "data/rft-guided/results/F-control")


def test_an_interrupted_run_picks_up_where_it_stopped(stage):
    broken, calls = run(stage, STUB_FAIL="sample.guided-skill=3")
    assert broken.returncode == 1, broken.stdout[-2000:] + broken.stderr[-1000:]
    assert [name for name, _ in calls] == ["eval", "eval", "skills", "sample", "sample", "sample", "sample"]
    assert "技能引导采样三次都没完成，停止。vLLM 日志最后 15 行：" in broken.stdout
    assert "===== 7/13" not in broken.stdout and "===== 全部结束 =====" not in broken.stdout
    assert_clean_shutdown(stage)                           # 停下时自己启动的服务也关掉了
    skills = (stage / "repo" / RUN / "skills.jsonl").read_bytes()

    result, calls = run(stage)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-1000:]
    # 技能文件不再动、基座不用重评、前缀那一次采样不用重做；服务开一次给剩下的技能引导采样（到那一步才开），
    # 评测时再开一次
    assert [name for name, _ in calls][:6] == ["sample", "prepare", "prepare", "control", "train", "train"]
    assert calls_of(calls, "sample") and "--tag guided-skill" in calls_of(calls, "sample")[0]
    assert "技能引导采样已经开始过：技能文件保持原样，这次不再调用教师" in result.stdout
    assert (stage / "repo" / RUN / "skills.jsonl").read_bytes() == skills
    assert "5/13 前缀引导采样：已完成，跳过" in result.stdout
    assert "3/13 启动 vLLM：已完成，跳过" in result.stdout and "（为这次采样启动 vLLM）" in result.stdout
    assert len(starts(stage)) == 2                           # 这次只为剩下的技能引导采样开一次，评测时再开一次
    assert steps(result)[-1] == "全部结束"
    assert_clean_shutdown(stage)


def test_sampling_is_retried_and_then_goes_on(stage):
    result, calls = run(stage, STUB_FAIL="sample.guided-prefix.<3=3")
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-1000:]
    assert [flag(args, "tag") for args in calls_of(calls, "sample")] == ["guided-prefix"] * 3 + ["guided-skill"]
    assert "前缀引导采样第 1 次没跑完，5 秒后重试（只补没完成的槽位）" in result.stdout   # 前两次没跑完，第三次补齐
    assert "前缀引导采样第 2 次没跑完，5 秒后重试（只补没完成的槽位）" in result.stdout
    assert "三次都没完成" not in result.stdout and steps(result)[-1] == "全部结束"
    assert_clean_shutdown(stage)


def test_sampling_that_never_finishes_stops_and_shuts_the_server_down(stage):
    result, calls = run(stage, STUB_FAIL="sample.guided-prefix=3")
    assert result.returncode == 1, result.stdout[-2000:] + result.stderr[-1000:]
    assert [name for name, _ in calls] == ["eval", "eval", "skills", "sample", "sample", "sample"]
    assert "前缀引导采样三次都没完成，停止。vLLM 日志最后 15 行：" in result.stdout
    assert "INFO:     Application startup complete." in result.stdout       # 服务日志的最后几行原样给用户看
    assert not (stage / "repo" / RUN / "sft").exists()
    assert "===== 6/13" not in result.stdout and "===== 全部结束 =====" not in result.stdout
    assert_clean_shutdown(stage)


def test_without_skills_there_is_no_skill_sampling_and_no_d_skill_arm(stage):
    repo = stage / "repo"
    result, calls = run(stage, SKILL="0")
    assert result.returncode == 0, result.stdout[-2500:] + result.stderr[-1000:]
    assert own_steps(result, 10) == [
        "1/10 检查前提", "2/10 写探针题号文件（上一轮零通过的题，之后经 vLLM 评适配器时拿它们做族内探针）",
        "3/10 启动 vLLM 并先评基座", "4/10 前缀引导采样（参考解的开头按 0.25,0.5,0.75 各给一部分，每种 4 次）",
        "5/10 关闭 vLLM，等显存释放",
        "6/10 准备训练数据（第 1 轮的验证组 + 引导出的正确答案，以及逐题配对的对照）",
        "7/10 D-prefix：训练（lr 1e-4、秩 32）并在 dev 上选型",
        "8/10 F-control：训练（lr 1e-4、秩 32）并在 dev 上选型",
        "9/10 汇总主结果（都在 dev 上；test 这一轮不评）",
        "10/10 经 vLLM 评各组选中的适配器（辅助指标；主结果是上面 HF 贪心那张表）"]
    assert not calls_of(calls, "skills") and len(calls_of(calls, "sample")) == 1
    control, = calls_of(calls, "control")
    assert "--guided-from data/rft-guided/sft/D-prefix --output-dir" in control and "D-skill" not in control
    assert [flag(args, "output") for args in calls_of(calls, "train") if "--dry-run" not in args] == [
        "data/rft-guided/checkpoints/D-prefix", "data/rft-guided/checkpoints/F-control"]
    assert not (repo / RUN / "sft/D-skill").exists()
    assert calls_of(calls, "tiers")[-1] == SUMMARY_HEAD + (
        "data/rft-guided/results/D-prefix data/rft-guided/results/F-control")
    assert evaluations(calls)[2:] == [
        ("D-prefix", "probe", "0", f"{RUN}/sft/D-prefix/train.jsonl"), ("D-prefix", "dev", "4", None),
        ("F-control", "probe", "0", None), ("F-control", "dev", "4", None)]
    assert "评探针前要自检的组：D-prefix" in result.stdout          # EXPECT_REPRODUCE 只有 D-prefix
    assert calls_of(calls, "summary") == [f"scripts/vllm_summary.py --run {RUN} --arms D-prefix F-control "
                                          f"--probe-ids-file {RUN}/probe-ids.txt"]
    assert "结果：F-control 训练集 5 条 / 5 题，其中引导题 3 条（难题 2、变体 1）；dev 3 条" in result.stdout
    assert "结果：训练时实际加载的题 D-prefix 和 F-control 逐题相同（5 道）" in result.stdout
    assert steps(result)[-1] == "全部结束"
    assert_clean_shutdown(stage)


def test_without_vllm_eval_nothing_is_evaluated_through_the_service(stage):
    result, calls = run(stage, VLLM_EVAL="0")
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-1000:]
    assert not calls_of(calls, "eval") and not calls_of(calls, "summary")
    assert not (stage / "repo" / RUN / "vllm").exists()
    assert starts(stage) == [SAMPLING_START]                  # 采样还是要开服务；评完就不再开了
    assert own_steps(result, 12) == [step.replace("/13 ", "/12 ") for step in ALL_STEPS if not step.startswith("13/")]
    assert own_steps(result, 12)[-1] == "12/12 汇总主结果（都在 dev 上；test 这一轮不评）"
    assert_clean_shutdown(stage)


def test_too_few_usable_skills_drop_the_skill_sampling_and_the_d_skill_arm(stage):
    result, calls = run(stage, STUB_SKILLS_OK="3")
    assert result.returncode == 0, result.stdout[-2500:] + result.stderr[-1000:]
    assert ("可用的技能只有 3 条，少于 MIN_SKILL_TASKS=5：技能引导的题太少，\n不做技能引导采样，也不训 D-skill 组"
            in result.stdout)
    assert len(calls_of(calls, "sample")) == 1 and "D-skill" not in " ".join(calls_of(calls, "control"))
    assert [step.split(" ")[0] for step in own_steps(result, 13)] == ["1/13", "2/13", "3/13", "4/13"]
    assert [step.split(" ")[0] for step in own_steps(result, 11)] == [f"{number}/11" for number in range(5, 12)]
    assert "评探针前要自检的组：D-prefix" in result.stdout
    assert steps(result)[-1] == "全部结束"
    assert_clean_shutdown(stage)


def test_skills_that_did_not_all_get_written_do_not_stop_the_run(stage):
    result, calls = run(stage, STUB_SKILLS_OK="5", STUB_SKILLS_EXIT="3")
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-1000:]
    assert "技能提示没全部写成，继续（没写成的题不做技能引导）" in result.stdout
    assert len(calls_of(calls, "sample")) == 2 and "D-skill" in calls_of(calls, "control")[0]
    assert steps(result)[-1] == "全部结束"


def test_unmatched_guided_tasks_are_dropped_from_both_groups_and_the_control_is_rebuilt(stage):
    repo = stage / "repo"
    make_unmatchable(repo)
    result, calls = run(stage)
    assert result.returncode == 0, result.stdout[-2500:] + result.stderr[-1000:]
    assert [name for name, _ in calls][:11] == [
        "eval", "eval", "skills", "sample", "sample", "prepare", "prepare", "control",
        "prepare", "prepare", "control"]                        # 重新准备两组，再重新配对
    first, second = calls_of(calls, "prepare")[:2], calls_of(calls, "prepare")[2:4]
    assert all("--drop-tasks-file" not in args for args in first)
    assert [flag(args, "output-dir") for args in second] == [f"data/rft-guided/sft/{group}"
                                                           for group in ("D-prefix", "D-skill")]
    assert all(flag(args, "drop-tasks-file") == f"{RUN}/sft/F-control/unmatched-tasks.txt" for args in second)
    assert (f"有引导题配不上没通过的答案（题号在 {RUN}/sft/F-control/unmatched-tasks.txt）"
            "：把它们从各个引导组里也去掉，两组才逐题对得上。" in result.stdout)
    # 两组都不再有这道题，对照组里也没有；去掉之前的目录挪到一边留着（同一次运行里挪开的用同一个时间戳）
    assert tasks_of(repo, "D-prefix") == [*BASE, "h3_t1", "h3_t2"]
    assert tasks_of(repo, "D-skill") == [*BASE, "h3_t3", "h3v_t1"]
    assert tasks_of(repo, "F-control") == [*BASE, "h3_t1", "h3_t2", "h3_t3", "h3v_t1"]
    assert (repo / RUN / "sft/F-control/unmatched-tasks.txt").read_text(encoding="utf-8") == ""
    set_aside = sorted((repo / RUN / "sft").glob("*.unmatched-*"))
    assert [path.name.split(".")[0] for path in set_aside] == ["D-prefix", "D-skill"]
    assert len({path.name.split(".unmatched-")[1] for path in set_aside}) == 1
    assert [task_ids(path / "train.jsonl") for path in set_aside] == [[*BASE, "h3_t1", "h3_t2", UNMATCHABLE],
                                                                        [*BASE, "h3_t3", "h3v_t1", UNMATCHABLE]]
    for line in ("结果：D-prefix 训练集 4 条 / 4 题，其中引导题 2 条（难题 2）；dev 3 条",
                 "结果：D-skill 训练集 4 条 / 4 题，其中引导题 2 条（难题 1、变体 1）；dev 3 条",
                 "结果：F-control 训练集 6 条 / 6 题，其中引导题 4 条（难题 3、变体 1）；dev 3 条"):
        assert line in result.stdout, line
    assert "结果：训练时实际加载的题 D-prefix、D-skill 和 F-control 逐题相同（6 道）" in result.stdout
    # 每个引导组各剩两道探针题：门槛就是 1，不按题数改 --sanity-min；没有被作废
    assert flag(probe_call(calls, "D-prefix"), "sanity-min") is None
    assert flag(probe_call(calls, "D-skill"), "sanity-min") is None
    assert "作废" not in result.stdout
    assert steps(result)[-1] == "全部结束"
    # 重跑：数据都对上了，什么都不重做
    again, calls = run(stage)
    assert again.returncode == 0, again.stdout[-2000:] + again.stderr[-1000:]
    assert not calls_of(calls, "prepare") and not calls_of(calls, "control") and "去掉" not in again.stdout
    assert len(list((repo / RUN / "sft").glob("*.unmatched-*"))) == 2


def make_unmatchable(repo, task=UNMATCHABLE):
    """让这道题（默认 h3v_t2）在上一轮的采样里全是写到一半被截断的输出：这样的题配不上一份没通过的答案。"""
    path = repo / ROUND1 / "sampling/train-graded.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row["task_id"] == task:
            row["finish_reason"] = "length"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_a_rerun_inside_the_dropping_flow_finishes_it_without_repeating_work(stage):
    repo = stage / "repo"
    make_unmatchable(repo)
    broken, calls = run(stage, STUB_FAIL="prepare.3=1")       # D-prefix 挪开之后、重新准备之前断了
    assert broken.returncode == 1, broken.stdout[-2000:] + broken.stderr[-1000:]
    assert "准备 D-prefix 的训练数据失败" in broken.stdout
    assert len(calls_of(calls, "prepare")) == 3 and not (repo / RUN / "sft/D-prefix").exists()
    assert (repo / RUN / "sft/F-control/unmatched-tasks.txt").read_text(encoding="utf-8") == f"{UNMATCHABLE}\n"

    result, calls = run(stage)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-1000:]
    # 接着做：两组带 --drop-tasks-file 重新准备（D-skill 还没去过掉，这次才挪开），对照删掉重配
    assert [name for name, _ in calls][:3] == ["prepare", "prepare", "control"]
    assert [flag(args, "output-dir") for args in calls_of(calls, "prepare")] == [
        "data/rft-guided/sft/D-prefix", "data/rft-guided/sft/D-skill"]
    assert tasks_of(repo, "D-prefix") == [*BASE, "h3_t1", "h3_t2"] and tasks_of(repo, "F-control") == [
        *BASE, "h3_t1", "h3_t2", "h3_t3", "h3v_t1"]
    assert len(list((repo / RUN / "sft").glob("*.unmatched-*"))) == 2
    assert steps(result)[-1] == "全部结束"


def test_a_rerun_after_the_control_was_rebuilt_only_redoes_the_control(stage):
    repo = stage / "repo"
    make_unmatchable(repo)
    broken, _ = run(stage, STUB_FAIL="control.2=2")           # 两组已经去掉配不上的题，对照还没重建
    assert broken.returncode == 1 and "配对对照没做成，停止" in broken.stdout
    assert tasks_of(repo, "D-prefix") == [*BASE, "h3_t1", "h3_t2"] and not (repo / RUN / "sft/F-control").exists()
    result, calls = run(stage)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-1000:]
    assert [name for name, _ in calls][:6] == ["control", "train", "train", "select", "tiers", "train"]
    assert tasks_of(repo, "F-control") == [*BASE, "h3_t1", "h3_t2", "h3_t3", "h3v_t1"]
    assert len(list((repo / RUN / "sft").glob("*.unmatched-*"))) == 2      # 去过掉的两组原样复用，没有再挪一次
    assert "D-prefix 的训练数据已经有了，跳过准备" in result.stdout


def test_when_guided_sampling_solves_nothing_new_the_run_stops_before_training(stage):
    result, calls = run(stage, STUB_SOLVED="")
    assert result.returncode == 1, result.stdout[-2000:] + result.stderr[-1000:]
    assert [name for name, _ in calls] == ["eval", "eval", "skills", "sample", "sample",
                                           "prepare", "prepare", "control"]
    assert "配对对照没做成，停止。引导采样没有解出第 1 轮验证组之外的新题时就会这样" in result.stdout
    assert "===== 9/13" not in result.stdout and "===== 全部结束 =====" not in result.stdout
    assert_clean_shutdown(stage)


def test_records_dropped_by_the_trainer_are_pointed_out(stage):
    result, _ = run(stage, STUB_DROP_LONG="F-control=h3_t2")
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-1000:]
    assert (MISMATCH + "['h3_t2']，只在 F-control 里的 []。" + LOST_TAIL) in result.stdout
    assert "逐题相同" not in result.stdout


@pytest.mark.parametrize("missing", [
    "sampling/train-manifest.json", "sampling/train-graded.jsonl",
    "sft/verified/train.jsonl", "sft/verified/dev.jsonl"])
def test_missing_round1_artifacts_stop_at_step_one(stage, missing):
    (stage / "repo" / ROUND1 / missing).unlink()
    result, calls = run(stage)
    assert result.returncode == 1 and calls == [] and starts(stage) == []
    assert f"缺少第 1 轮的产物（这一轮是在它的基础上做的）：\n  {ROUND1}/{missing}" in result.stdout
    assert "ln -s ../CodeAgent-R-v19/data data" in result.stdout
    assert "===== 2/13" not in result.stdout and "===== 全部结束 =====" not in result.stdout


def test_a_manifest_without_any_task_in_the_chosen_tiers_stops_before_the_gpu_is_touched(stage):
    path = stage / "repo" / ROUND1 / "sampling/train-manifest.json"
    write(path, {"zero_pass_tasks": ["v2_t1"]})
    result, calls = run(stage)
    assert result.returncode == 1 and calls == [] and starts(stage) == []
    assert NO_PROBE in result.stdout
    assert "探针题号文件没写成，停止" in result.stdout and "===== 3/13" not in result.stdout


@pytest.mark.parametrize("mode, reason", [("oom", "引擎初始化失败"), ("vanish", "进程在就绪之前就退出了")])
def test_a_server_that_cannot_start_stops_before_sampling(stage, mode, reason):
    result, calls = run(stage, FAKE_VLLM_MODE=mode)
    assert result.returncode == 1, result.stdout[-1500:] + result.stderr[-1000:]
    assert calls == [] and starts(stage) and not (stage / "repo" / RUN / "sampling").exists()
    assert f"vLLM 没起来（{reason}），日志最后 15 行：" in result.stdout
    assert "===== 4/13" not in result.stdout and "===== 全部结束 =====" not in result.stdout
    assert eventually(lambda: not alive(stage))


def test_a_busy_gpu_stops_the_run_at_step_one(stage):
    result, calls = run(stage, FAKE_GPU_USED="18000")
    assert result.returncode == 1, result.stderr[-1000:]
    assert calls == [] and starts(stage) == []
    assert "显卡被占用（已用 18000 MiB）" in result.stdout and "12345, 18000 MiB" in result.stdout
    assert "===== 2/13" not in result.stdout and "===== 全部结束 =====" not in result.stdout


def panel_state(stage, log_text):
    """把一份日志交给进度面板（scripts/show_progress.sh），返回它显示的状态那一行。"""
    log = stage / "panel.log"
    log.write_text(log_text, encoding="utf-8")
    shown = subprocess.run(["bash", str(ROOT / "scripts/show_progress.sh"), str(log), "", ""], cwd=stage / "repo",
                           env=environment(stage), capture_output=True, text=True, timeout=60)
    assert shown.returncode == 0, shown.stderr
    return next(line for line in shown.stdout.splitlines() if line.startswith("状态:"))


INCOMPLETE_END = "；上面的 dev 选型结果不受影响，重新运行同一条命令只补没做完的 ====="
SUMMARY_STEP = "汇总主结果（都在 dev 上；test 这一轮不评）"       # 主结果那张表打印的那一步


def test_a_base_evaluation_that_does_not_finish_goes_on_but_the_run_does_not_end_as_complete(stage):
    """基座的 vLLM 评测只是给各组作对照：没评完不耽误后面的采样和训练。但这一轮要做的的事少了一件——结束时不打
    "全部结束"、退出码是 1，进度面板上看得出没做完；原样重跑只补基座没评完的那部分。"""
    result, calls = run(stage, STUB_EVAL_FAIL="base/dev=3")
    assert result.returncode == 1, result.stdout[-2000:] + result.stderr[-1000:]
    assert ("基座的 vLLM 评测没完成（退出码 3），不影响后面的采样和训练；这一轮结束时不算是全部完成，"
            "重新运行本脚本时会只补没完成的") in result.stdout
    assert [label for label, *_ in evaluations(calls)][:2] == ["base", "base"]     # dev 没评完，探针照评
    assert SUMMARY_STEP in result.stdout                               # 主结果照常出了
    assert result.stdout.rstrip().endswith("===== 没有全部完成：基座的 vLLM 评测没评完（dev）" + INCOMPLETE_END)
    assert steps(result).count("全部结束") == 1                 # 只有里面那次 vLLM 评测自己的
    assert_clean_shutdown(stage)
    assert panel_state(stage, result.stdout) .startswith("状态: 已结束，但没有全部完成")
    again, calls = run(stage)                             # 重跑只补基座的 dev，其余都评过了
    assert again.returncode == 0, again.stderr[-1000:]
    assert [(label, stem) for label, stem, *_ in evaluations(calls)] == [("base", "dev")]
    assert steps(again)[-1] == "全部结束" and panel_state(stage, again.stdout) == "状态: 已完成"


def test_a_final_evaluation_that_does_not_finish_is_said_so_at_the_end_and_is_done_on_a_rerun(stage):
    """各组选中的适配器经 vLLM 评是辅助指标：没评完不影响上面的 dev 选型结果。但它没评完就是没做完——rft_vllm_eval.sh
    自己不打印"全部结束"、以 1 退出，外层不能替它打印"全部结束"再以 0 退出（那样进度面板会显示"已完成"）。"""
    result, calls = run(stage, STUB_EVAL_FAIL="F-control/dev=3")
    assert result.returncode == 1, result.stdout[-2000:] + result.stderr[-1000:]
    assert "F-control：vLLM 评测没完成（退出码 3），重新运行同一条命令会只补没完成的" in result.stdout
    assert ("结果：vLLM 评测没有全部完成，重新运行同一条命令会只补没完成的；已经评完的报告在 " + RUN + "/vllm"
            in result.stdout)                                    # rft_vllm_eval.sh 自己的说法
    assert SUMMARY_STEP in result.stdout                               # 主结果已经在前面打出来了
    assert result.stdout.rstrip().endswith(
        "===== 没有全部完成：vLLM 评测没有全部完成（rft_vllm_eval.sh 以退出码 1 结束）" + INCOMPLETE_END)
    assert steps(result).count("全部结束") == 0 and not result.stdout.rstrip().endswith("===== 全部结束 =====")
    assert_clean_shutdown(stage)
    assert panel_state(stage, result.stdout).startswith("状态: 已结束，但没有全部完成")
    again, calls = run(stage)
    assert again.returncode == 0, again.stderr[-1000:]
    assert [(label, stem) for label, stem, *_ in evaluations(calls)] == [("F-control", "dev")]   # 只补没评完的
    assert again.stdout.rstrip().endswith("===== 全部结束 =====") and panel_state(stage, again.stdout) == "状态: 已完成"


def test_switches_reach_the_scripts_that_use_them(stage):
    result, calls = run(stage, RFT_TIERS="hard", RFT_SAMPLES="2", SKILL_SAMPLES="3", HINT_FRACTIONS="0.5",
                        EVAL_SAMPLES="2", MIN_SKILL_TASKS="3", RFT_LR="5e-5", RFT_LORA_R="8", RFT_LORA_ALPHA="16",
                        RFT_EPOCHS="2",
                        GRAD_ACCUM="4", MAX_LENGTH="1024", RFT_CONCURRENCY="7", RFT_GRADE_WORKERS="2")
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-1000:]
    assert (stage / "repo" / RUN / "probe-ids.txt").read_text(encoding="utf-8") == "h3_t1\nh3_t2\nh3_t3\n"
    assert "--tiers hard " in calls_of(calls, "skills")[0]
    prefix, skill = calls_of(calls, "sample")
    assert "--tiers hard --vllm-url http://127.0.0.1:8000 --model org/model --concurrency 7 " in prefix
    assert "--grade-workers 2 --output-dir" in prefix
    assert "--hint-kind prefix --hint-fractions 0.5 --samples 2 --only-zero-pass" in prefix
    assert "--hint-kind skill --samples 3 --skills-file" in skill
    train = [args for args in calls_of(calls, "train") if "--dry-run" not in args][0]
    assert ("--epochs 2 --lr 5e-5 --lora-r 8 --lora-alpha 16 --batch-size 1 --grad-accum 4 --max-length 1024 --seed 42"
            in train)
    assert {samples for _, _, samples, _ in evaluations(calls)} == {"2", "0"}
    assert "===== 9/13 D-prefix：训练（lr 5e-5、秩 8）并在 dev 上选型 =====" in result.stdout


def test_the_base_report_is_taken_from_round_one_unless_one_is_given(stage):
    repo = stage / "repo"
    result, calls = run(stage)
    assert result.returncode == 0, result.stderr[-1000:]
    assert all(flag(args, "base-report") == "data/rft-round1/results/A-verified/epoch-000-dev.json"
               for args in calls_of(calls, "select"))
    shutil.rmtree(repo / RUN)                              # 换一个运行目录重来；第 1 轮的基座报告没有了
    shutil.rmtree(repo / ROUND1 / "results/A-verified")
    write(repo / "data/given-base.json", "{}")
    result, calls = run(stage, BASE_REPORT="data/given-base.json")     # 外面给了一份：三组都用它
    assert result.returncode == 0, result.stderr[-1000:]
    assert {flag(args, "base-report") for args in calls_of(calls, "select")} == {"data/given-base.json"}
    shutil.rmtree(repo / RUN)
    result, calls = run(stage)                             # 哪儿都没有：第一组自己评基座，后面两组复用它的报告
    assert result.returncode == 0, result.stderr[-1000:]
    base = "data/rft-guided/results/D-prefix/epoch-000-dev.json"
    assert [flag(args, "base-report") for args in calls_of(calls, "select")] == [None, base, base]

    write_round1(repo)                                     # 第 1 轮的基座报告有了，但明白地说了不用它
    shutil.rmtree(repo / RUN)
    result, calls = run(stage, BASE_REPORT="")
    assert result.returncode == 0, result.stderr[-1000:]
    assert [flag(args, "base-report") for args in calls_of(calls, "select")] == [None, base, base]


NOT_DONE = "===== 没有全部完成："
NO_KEY = "写技能没有做成：教师接口的参数或环境不对（write_skills.py 退出码 2），没有 D-skill 组"
NO_KEY_END = NOT_DONE + NO_KEY + INCOMPLETE_END
NO_KEY_NOTE = ("提醒：环境里没有 TEACHER_API_KEY，这次写不了技能提示（write_skills.py 会以退出码 2 结束）：\n"
               "      上一跑没有留下够用的技能时，这一轮就没有技能引导采样、也没有 D-skill 组，结束时不算是全部完成\n")


def test_without_a_teacher_key_the_run_goes_on_but_does_not_end_as_if_everything_was_done(stage):
    """没有 TEACHER_API_KEY（例如换了新的仓库包、.env 没跟过来）：write_skills.py 以 2 退出，是环境不对。这一轮照常做
    前缀引导和两组对照——夜里的工夫不白费——但没有 D-skill 组：结束时不打"全部结束"、退出码是 1，进度面板上看得出
    没做完（以前只打印一句提醒，最后照样"全部结束"、退出码 0）。"""
    result, calls = run(stage, STUB_SKILLS_OK="0", STUB_SKILLS_EXIT="2")
    assert result.returncode == 1, result.stdout[-2000:] + result.stderr[-1000:]
    assert NO_KEY_NOTE in result.stdout
    assert "从上一包拷过来" in result.stdout                          # 新的仓库包里没有 .env：说清楚该怎么做
    assert ("写技能没有做成（write_skills.py 以退出码 2 结束，是参数或环境不对，见上面的报错）。\n"
            "上一跑留下的可用技能不够时，这一次就没有技能引导采样、也不训 D-skill 组；处理好了原样重跑会把它补上") \
        in result.stdout
    assert "技能提示没全部写成，继续" not in result.stdout             # 那是"有几道题没写成"的说法，不是这种情况
    assert len(calls_of(calls, "sample")) == 1 and "评探针前要自检的组：D-prefix" in result.stdout
    assert SUMMARY_STEP in result.stdout                               # 两组的选型照常做了
    assert result.stdout.rstrip().endswith(NO_KEY_END)
    assert steps(result).count("全部结束") == 1 and not result.stdout.rstrip().endswith("===== 全部结束 =====")
    assert_clean_shutdown(stage)
    assert panel_state(stage, result.stdout).startswith("状态: 已结束，但没有全部完成")
    # 密钥配上以后原样重跑：D-skill 组补上，F-control 按两个引导组重新配（另一个用例演的就是这个）
    assert not (stage / "repo" / RUN / "sft/D-skill").exists()
    # 有密钥、写出来的技能全都泄露被拦：这是实验结果，不是故障，照旧算做完
    fixed, calls = run(stage, TEACHER_API_KEY="teacher-key-for-tests", STUB_SKILLS_OK="0")
    assert fixed.returncode == 0 and "提醒：环境里没有 TEACHER_API_KEY" not in fixed.stdout
    assert fixed.stdout.rstrip().endswith("===== 全部结束 =====")


def test_skills_that_all_errored_leave_the_run_incomplete_while_leaky_skills_alone_do_not(stage):
    """教师接口对每道题都出错（write_skills.py 以 3 退出）：可用的技能是 0，没有 D-skill 组。少了一个组是因为出了错，
    这一轮不算是全部完成；技能全都泄露被拦（leaky）则是实验结果，照旧算做完。"""
    result, calls = run(stage, STUB_SKILLS_OK="0", STUB_SKILLS_EXIT="3")
    assert result.returncode == 1, result.stdout[-2000:] + result.stderr[-1000:]
    assert "技能提示没全部写成，继续（没写成的题不做技能引导）" in result.stdout
    assert "可用的技能只有 0 条，少于 MIN_SKILL_TASKS=5" in result.stdout
    assert len(calls_of(calls, "sample")) == 1
    assert result.stdout.rstrip().endswith(
        NOT_DONE + "技能引导没有做成：可用的技能只有 0 条、少于 MIN_SKILL_TASKS=5，其中 5 道出错，没有 D-skill 组"
        + INCOMPLETE_END)
    assert panel_state(stage, result.stdout).startswith("状态: 已结束，但没有全部完成")
    # 原样重跑：出错的那几道重写（泄露被拦，用不上）——没有出错的题了，就只是技能不够，算做完
    again, calls = run(stage, STUB_SKILLS_OK="0")
    assert again.returncode == 0, again.stdout[-2000:] + again.stderr[-1000:]
    assert "可用的技能只有 0 条，少于 MIN_SKILL_TASKS=5" in again.stdout and NOT_DONE not in again.stdout
    assert again.stdout.rstrip().endswith("===== 全部结束 =====")


CRASHED_SKILLS = "写技能没有做成：write_skills.py 以退出码 1 结束（不是约定的 0 或 3），没有 D-skill 组"


def test_a_write_skills_that_crashes_leaves_the_run_incomplete_when_it_cost_the_skill_arm(stage):
    """write_skills.py 以约定之外的退出码结束（自己崩了）：和"有几道题没写成"不是一回事。可用的技能因此不够、
    没有 D-skill 组时，这一轮不算是全部完成。"""
    result, calls = run(stage, STUB_SKILLS_OK="0", STUB_SKILLS_EXIT="1")
    assert result.returncode == 1, result.stdout[-2000:] + result.stderr[-1000:]
    assert "写技能没有做成（write_skills.py 以退出码 1 结束，不是约定的 0 或 3，见上面的报错）。" in result.stdout
    assert "技能提示没全部写成，继续" not in result.stdout
    assert len(calls_of(calls, "sample")) == 1
    assert result.stdout.rstrip().endswith(NOT_DONE + CRASHED_SKILLS + INCOMPLETE_END)
    assert panel_state(stage, result.stdout).startswith("状态: 已结束，但没有全部完成")


def test_the_env_file_is_loaded_only_when_skills_are_on(stage):
    repo = stage / "repo"
    write(repo / ".env", "TEACHER_API_KEY=teacher-key-for-tests\nMODEL=org/from-env-file\n")
    result, calls = run(stage)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-1000:]
    assert "提醒：环境里没有 TEACHER_API_KEY" not in result.stdout
    assert "--model org/from-env-file --revision rev-123" in calls_of(calls, "train")[0]
    serving = SAMPLING_START.replace("org/model", "org/from-env-file")
    assert starts(stage)[0] == serving and starts(stage)[1].startswith(serving + " --enable-lora")   # 开服务用的也是它
    shutil.rmtree(repo / RUN)
    result, calls = run(stage, SKILL="0")
    assert result.returncode == 0, result.stderr[-1000:]
    assert "--model org/model --revision rev-123" in calls_of(calls, "train")[0]


def test_switches_that_are_not_0_or_1_stop_before_anything_runs(stage):
    for name in ("SKILL", "VLLM_EVAL", "SKILL_JUDGE"):
        result, calls = run(stage, **{name: "yes"})
        assert result.returncode == 1 and calls == []
        assert f"{name} 只能是 0 或 1（现在是 yes）" in result.stdout


def test_the_teacher_switches_reach_write_skills(stage):
    """教师接口的节流和输出上限由环境变量给（默认值保守：并发 2、请求至少隔 1 秒、每次回复 4096 个 token——服务器上
    并发 4 时成片 429，900 个 token 时成片报"没有 JSON 对象"），自判这一步可以用 SKILL_JUDGE=0 关掉。"""
    header = (ROOT / "scripts/run_rft_guided.sh").read_text(encoding="utf-8").split("\ncd ", 1)[0]
    for piece in ("TEACHER_CONCURRENCY=2", "TEACHER_MIN_INTERVAL=1", "SKILL_JUDGE=1", "TEACHER_MAX_TOKENS=4096"):
        assert piece in header, piece                              # 都写进了文件头的环境变量说明
    result, calls = run(stage, TEACHER_CONCURRENCY="3", TEACHER_MIN_INTERVAL="0.5", TEACHER_MAX_TOKENS="9000",
                        SKILL_JUDGE="0")
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-1000:]
    skills, = calls_of(calls, "skills")
    assert skills.endswith("--output data/rft-guided/skills.jsonl "
                           "--concurrency 3 --min-interval 0.5 --max-tokens 9000 --no-judge")
    assert steps(result)[-1] == "全部结束"                       # 这几个值不碍着整轮从头走到尾


def test_the_env_file_sets_the_teacher_switches(stage):
    """.env 里写的并发盖过外面同名的环境变量（和密钥一样：脚本先载入 .env，然后才取这几个默认值）。"""
    (stage / "repo/.env").write_text("TEACHER_API_KEY=teacher-key-for-tests\nTEACHER_CONCURRENCY=1\n", encoding="utf-8")
    result, calls = run(stage, TEACHER_CONCURRENCY="4")
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-1000:]
    skills, = calls_of(calls, "skills")
    assert skills.endswith("--concurrency 1 --min-interval 1 --max-tokens 4096 --judge")


def probe_call(calls, label):
    """某个组那一次探针评测（eval_vllm.py --tag probe）的参数。"""
    return next(args for args in calls_of(calls, "eval") if f"--label {label} " in args and "--tag probe" in args)


def test_a_rerun_after_a_run_without_skills_prepares_the_skill_arm_and_repairs_the_control(stage):
    """上一跑没有可用的技能（例如还没配教师的密钥），整轮做完了；密钥配好之后再跑一次：D-skill 这一组要做出来，
    F-control 也得按"两个引导组"重新配、重新训——照用旧的，D 组和 F 组的题目就对不上。"""
    repo = stage / "repo"
    first, calls = run(stage, STUB_SKILLS_OK="0", STUB_SKILLS_EXIT="2")       # 没有密钥：写技能以 2 退出
    assert first.returncode == 1 and first.stdout.rstrip().endswith(NO_KEY_END)
    assert not (repo / RUN / "sft/D-skill").exists()
    assert tasks_of(repo, "F-control") == [*BASE, "h3_t1", "h3_t2", UNMATCHABLE]

    second, calls = run(stage, STUB_SKILLS_OK="99", STUB_SKILLS_EXIT="0")
    assert second.returncode == 0, second.stdout[-3000:] + second.stderr[-1000:]
    assert [name for name, _ in calls] == [
        "skills", "sample", "prepare", "control",
        "tiers",                                              # D-prefix 选过型了，照旧
        "train", "train", "select", "tiers",                  # D-skill：新训的一组
        "train", "train", "select", "tiers",                  # F-control：按两个引导组重新配过，重训
        "tiers", "eval", "eval", "eval", "eval", "summary"]
    assert [flag(args, "tag") for args in calls_of(calls, "sample")] == ["guided-skill"]
    assert "===== 3/13 启动 vLLM：已完成，跳过 =====" in second.stdout
    assert "（为这次采样启动 vLLM）" in second.stdout          # 第 3 步没有开服务：技能引导采样要用，到那一步才开
    assert ("F-control 和现在的引导组对不上（这一次的引导组是 D-prefix D-skill；原来的 F-control 是按 D-prefix 配的）"
            in second.stdout)
    for folder in ("sft", "checkpoints", "results", "vllm"):  # 旧的 F-control 没有删，挪到一边留着
        moved, = list((repo / RUN / folder).glob("F-control.unpaired-*"))
        assert moved.is_dir()
    assert tasks_of(repo, "D-skill") == [*BASE, "h3_t3", "h3v_t1", UNMATCHABLE]
    assert tasks_of(repo, "F-control") == [*BASE, *PROBE]
    assert (repo / RUN / "vllm/F-control/F-control-dev-summary.json").is_file()     # 新的 F-control 重新评过
    assert "逐题相同（7 道）" in second.stdout and len(starts(stage)) == 2

    third, calls = run(stage)                                 # 再往后重跑就是全部跳过
    assert third.returncode == 0 and [name for name, _ in calls] == ["tiers"] * 4 + ["summary"]
    assert starts(stage) == [] and "对不上" not in third.stdout


def test_switching_skills_off_repair_the_control_to_the_remaining_group(stage):
    """上一跑的数据是按 D-prefix、D-skill 两组备好的，还没训完就断了；这一次设了 SKILL=0：只训 D-prefix 的话，
    F-control 也得只按 D-prefix 配。"""
    repo = stage / "repo"
    broken, calls = run(stage, STUB_FAIL="train.1=1")         # 第一次真训练就出错：一组都没训完
    assert broken.returncode == 1 and "训练出错，把上面的报错贴给我" in broken.stdout
    assert (repo / RUN / "sft/D-skill/train.jsonl").is_file() and not (repo / RUN / "results").exists()
    assert tasks_of(repo, "F-control") == [*BASE, *PROBE]

    result, calls = run(stage, SKILL="0")
    assert result.returncode == 0, result.stdout[-2500:] + result.stderr[-1000:]
    assert [name for name, _ in calls] == [
        "control", "train", "train", "select", "tiers", "train", "train", "select", "tiers", "tiers",
        "eval", "eval", "eval", "eval", "summary"]
    assert ("F-control 和现在的引导组对不上（这一次的引导组是 D-prefix；原来的 F-control 是按 D-prefix、D-skill 配的）"
            in result.stdout)
    assert sorted(path.parent.name for path in (repo / RUN).glob("*/F-control.unpaired-*")) == ["sft"]
    assert tasks_of(repo, "F-control") == tasks_of(repo, "D-prefix") == [*BASE, "h3_t1", "h3_t2", UNMATCHABLE]
    assert "结果：训练时实际加载的题 D-prefix 和 F-control 逐题相同（5 道）" in result.stdout
    assert "作废" not in result.stdout and steps(result)[-1] == "全部结束"

    result, calls = run(stage)                                # SKILL 再打开：D-skill 的数据还在，这一组只管训
    assert result.returncode == 0, result.stdout[-2500:] + result.stderr[-1000:]
    assert [name for name, _ in calls] == [
        "control", "tiers", "train", "train", "select", "tiers", "train", "train", "select", "tiers", "tiers",
        "eval", "eval", "eval", "eval", "summary"]
    moved = sorted(path.parent.name for path in (repo / RUN).glob("*/F-control.unpaired-*"))
    assert moved == ["checkpoints", "results", "sft", "sft", "vllm"]   # 训过、评过的 F-control 也是挪开，不删
    assert tasks_of(repo, "F-control") == [*BASE, *PROBE]
    assert "逐题相同（7 道）" in result.stdout


def test_skills_are_frozen_once_the_skill_sampling_has_started(stage):
    """上一跑写技能有 error、技能引导采样又没做完：重跑时不能再调教师。它会把出错的题重写、追加进技能文件，
    采样协议里记着的摘要就对不上，技能引导采样就再也续不上了。"""
    repo = stage / "repo"
    broken, calls = run(stage, STUB_SKILLS_OK="4", STUB_SKILLS_EXIT="3", MIN_SKILL_TASKS="3",
                        STUB_FAIL="sample.guided-skill=3")
    assert broken.returncode == 1, broken.stdout[-2000:] + broken.stderr[-1000:]
    assert "技能提示没全部写成，继续（没写成的题不做技能引导）" in broken.stdout
    assert "技能引导采样三次都没完成，停止。" in broken.stdout
    skills = (repo / RUN / "skills.jsonl").read_bytes()
    protocol = json.loads((repo / RUN / "sampling/train-guided-skill-protocol.json").read_text(encoding="utf-8"))
    assert protocol["skills_sha256"] == hashlib.sha256(skills).hexdigest()
    assert not (repo / RUN / "sampling/train-guided-skill-manifest.json").exists()

    result, calls = run(stage, STUB_SKILLS_OK="99", STUB_SKILLS_EXIT="0", MIN_SKILL_TASKS="3")
    assert result.returncode == 0, result.stdout[-2500:] + result.stderr[-1000:]
    assert "skills" not in [name for name, _ in calls]       # 出错的题要是重写了，技能文件的摘要就变了
    assert (repo / RUN / "skills.jsonl").read_bytes() == skills
    assert "技能引导采样已经开始过：技能文件保持原样，这次不再调用教师" in result.stdout
    assert [flag(args, "tag") for args in calls_of(calls, "sample")] == ["guided-skill"]
    assert steps(result)[-1] == "全部结束"
    assert_clean_shutdown(stage)


def test_a_sampling_call_refused_with_exit_code_2_is_not_retried(stage):
    """退出码 2 是参数或采样协议不对（这里是两次运行之间改了采样次数）：重试没有用，也不该拿 vLLM 日志的末尾当原因。"""
    broken, calls = run(stage, STUB_FAIL="sample.guided-prefix=3")
    assert broken.returncode == 1 and len(calls_of(calls, "sample")) == 3
    result, calls = run(stage, RFT_SAMPLES="9")
    assert result.returncode == 1, result.stdout[-2000:] + result.stderr[-1000:]
    assert len(calls_of(calls, "sample")) == 1               # 只调了一次
    assert "采样协议与这个目录里已有的不一致" in result.stderr   # sample_rft.py 的原话（假脚本照抄）
    assert ("前缀引导采样没有做成：sample_rft.py 以退出码 2 结束，是参数或采样协议不对，重试没有用。把上面的报错贴给我"
            in result.stdout)
    assert "vLLM 日志最后 15 行" not in result.stdout and "5 秒后重试" not in result.stdout
    assert "===== 全部结束 =====" not in result.stdout
    assert_clean_shutdown(stage)


def test_a_rerun_in_a_dir_whose_skill_arm_was_dropped_does_not_start_the_server(stage):
    """技能太少而降级过的运行目录：重跑时技能还是不够，就不要再白开一次 vLLM（加载几分钟、占着显卡）。"""
    assert run(stage, STUB_SKILLS_OK="0")[0].returncode == 0
    result, calls = run(stage, STUB_SKILLS_OK="0")
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-1000:]
    assert starts(stage) == []                               # 没有开服务；采样和评测都跳过
    assert [name for name, _ in calls] == ["skills", "tiers", "tiers", "tiers", "summary"]
    assert "===== 3/13 启动 vLLM：已完成，跳过 =====" in result.stdout
    assert "可用的技能只有 0 条，少于 MIN_SKILL_TASKS=5" in result.stdout
    assert "===== 5/11 前缀引导采样：已完成，跳过 =====" in result.stdout


@pytest.mark.parametrize("changed, voided", [(None, False), ("0", True)], ids=["differs", "identical"])
def test_a_group_with_a_single_trained_probe_task_is_still_checked_at_the_default_minimum(stage, changed, voided):
    """技能引导只解出一道探针题：自检照样做，门槛就是 eval_vllm.py 的默认值 1，不按题数改；
    这唯一的题都和基座不同不了时照样作废。"""
    result, calls = run(stage, STUB_SOLVED_SKILL="h3_t3",
                        **({} if changed is None else {"STUB_SANITY": f"D-skill={changed}"}))
    assert result.returncode == 0, result.stdout[-2500:] + result.stderr[-1000:]
    assert flag(probe_call(calls, "D-skill"), "sanity-min") is None
    assert flag(probe_call(calls, "D-prefix"), "sanity-min") is None      # 三道都训练过：门槛也是默认的 1
    sanity = json.loads((stage / "repo" / RUN / "vllm/D-skill/D-skill-probe-sanity.json").read_text(encoding="utf-8"))
    assert (sanity["total"], sanity["min"], sanity["changed"]) == (1, 1, 0 if voided else 1)
    evaluated = [(label, stem) for label, stem, *_ in evaluations(calls)]
    if voided:
        assert "D-skill：适配器在这几道题上的输出和基座完全一样，适配器在 vLLM 里没有生效，这组的 vLLM 评测作废" \
            in result.stdout
        assert ("D-skill", "dev") not in evaluated                       # 自检没过，不往下评 dev
    else:
        assert "作废" not in result.stdout and ("D-skill", "dev") in evaluated


def tiny(task_id):
    """一道独立的小题：第一个正数没找对，公开测试过、隐藏测试挂。"""
    module = f"{task_id}.py"
    return {
        "task_id": task_id, "source_group": f"group-{task_id}", "split": "train", "bug_family": "first_positive",
        "difficulty": "L1", "module": module,
        "instruction": "实现 positive(values)：返回列表里第一个正数，没有正数时返回 None。",
        "buggy": "def positive(values):\n    return values[0] if values else None\n",
        "fixed": "def positive(values):\n    return next((v for v in values if v > 0), None)\n",
        "public_test": f"from {task_id} import positive\n\n\ndef test_public():\n    assert positive([1, 5]) == 1\n",
        "private_tests": {"test_hidden.py": f"from {task_id} import positive\n\n\ndef test_hidden():\n"
                          "    assert positive([-3, 3, 13]) == 3\n    assert positive([-1]) is None\n"},
    }


def test_the_probe_helper_checks_the_adapter_against_the_base_at_the_default_minimum(tmp_path, dead_proxy):
    """不用假的 python：rft_eval_probe 真的去调 scripts/eval_vllm.py（服务是本机的假服务、判分是真的）。
    训练数据里只有两道探针题：rft_eval_probe 不再按题数传 --sanity-min，门槛就是 eval_vllm.py 的默认值 1。
    训练过的题复现不了不再作废（旧判据下两道全复现也过不了"至少 3 道"）；作废只看输出是不是和基座完全一样。"""
    sys.path.insert(0, str(ROOT / "tests"))
    from test_rft_sampling import FakeVLLM

    specs = [tiny(f"h3_p{index}") for index in range(1, 5)]
    run, tasks = tmp_path / "run", tmp_path / "tasks.jsonl"
    write(tasks, "".join(json.dumps(spec, ensure_ascii=False) + "\n" for spec in specs))
    write(run / "probe-ids.txt", "".join(f"{spec['task_id']}\n" for spec in specs))
    install(tmp_path, "bin", "python", f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')      # 脚本里写的是 python
    # 适配器答得和基座一样的题：some 时只有 h3_p1（这一道复现不了），all 时全部（适配器在服务里没有生效）
    alike = {"some": False, "all": False}

    def answer(payload):
        if payload["max_tokens"] == 1:           # 探测参数用的请求（只生成 1 个 token）
            return 200, {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}
        user = payload["messages"][-1]["content"]
        spec = next(item for item in specs if f"文件 {item['module']}：" in user)
        if payload["model"] == "org/model" or alike["all"] or (alike["some"] and spec["task_id"] == "h3_p1"):
            code = spec["buggy"]                 # 基座全答错
        else:
            code = spec["fixed"]
        return 200, {"choices": [{"message": {"content": f"```python\n{code}```"}, "finish_reason": "stop"}]}

    fake = FakeVLLM(answer)
    try:
        def probe(*trained):
            write(run / "sft/D/train.jsonl", "".join(json.dumps({"task_id": task}) + "\n" for task in trained))
            shutil.rmtree(run / "vllm/D", ignore_errors=True)
            quote = lambda path: shlex.quote(str(path))      # noqa: E731
            settings = f"RUN={quote(run)} TASKS={quote(tasks)} GRADE_WORKERS=2 VLLM_PORT={fake.url.rsplit(':', 1)[1]}"
            environment = {**os.environ, "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}"}
            environment.pop("AGENTLAB_SANDBOX", None)
            program = (f"source scripts/rft_lib.sh; {settings}; rft_eval_probe D D {quote(run / 'vllm/D')} "
                       f"{quote(run / 'probe-ids.txt')} {quote(run / 'sft/D/train.jsonl')}; echo rc=$?")
            result = subprocess.run(["bash", "-c", program], cwd=ROOT, env=environment, capture_output=True, text=True,
                                    timeout=600)
            folder = run / "vllm/D"
            report = folder / "D-probe-sanity.json"
            return (result.stdout.strip().splitlines()[-1], result.stdout,
                    {item.name for item in folder.iterdir()} if folder.exists() else set(),
                    json.loads(report.read_text(encoding="utf-8")) if report.exists() else None)

        done = {"D-probe-greedy.json", "D-probe-records.jsonl", "D-probe-sanity.json", "D-probe-summary.json"}
        rc, out, files, sanity = probe("h3_p1", "h3_p2")
        assert rc == "rc=0" and "自检：训练过的题严格通过 2/2（仅供参考" in out, out[-2000:]
        assert "结果：D probe 贪心 4/4" in out and "自检：适配器的输出在 2 道题里有 2 道和基座不同（门槛 1）" in out
        assert files == done
        assert (sanity["pass"], sanity["total"], sanity["min"], sanity["changed"],
                sanity["compared"]) == (2, 2, 1, 2, True)
        alike["some"] = True
        rc, out, files, sanity = probe("h3_p1", "h3_p2")      # 有一道复现不了：输出还是和基座不同，不作废
        assert rc == "rc=0" and "自检：训练过的题严格通过 1/2（仅供参考" in out, out[-2000:]
        assert "结果：D probe 贪心 3/4" in out
        assert "自检：适配器的输出在 2 道题里有 1 道和基座不同（门槛 1）" in out and files == done
        assert (sanity["pass"], sanity["changed"], sanity["changed_tasks"]) == (1, 1, ["h3_p2"])
        alike.update(some=False, all=True)
        rc, out, files, sanity = probe("h3_p1", "h3_p2")      # 输出和基座完全一样：以 4 结束，不写探针的报告和 summary
        assert rc == "rc=4" and files == {"D-probe-sanity.json"}, out[-2000:]
        assert "自检：训练过的题严格通过 0/2（仅供参考" in out
        assert "自检：适配器的输出在 2 道题里有 0 道和基座不同（门槛 1）" in out
        # 最后一行是 rc=4，它前面那句是 eval_vllm.py 的结论
        assert out.rstrip().splitlines()[-2] == "适配器在这几道题上的输出和基座完全一样，适配器在 vLLM 里没有生效"
        assert (sanity["changed"], sanity["compared"], sanity["min"]) == (0, True, 1)
        alike["all"] = False
        rc, out, files, sanity = probe("v2_t9")               # 探针题一道都没训练过：没能比较，不作废，照常评
        assert rc == "rc=0" and "没能确认适配器是否生效；不作废，接着往后评" in out, out[-2000:]
        assert "结果：D probe 贪心 4/4" in out
        assert "自检：没能和基座比较（本次要评的题里没有出现在 --sanity-tasks-file 里的，没有可比较的题）" in out
        assert (sanity["total"], sanity["min"], sanity["compared"], sanity["changed"]) == (0, 1, False, 0)
        assert files == done                                   # 旧版这时以 4 结束、不写探针报告；现在照常评完
    finally:
        fake.close()


def launch_line():
    """文件头里给用户粘贴的那一行启动命令。"""
    header = (ROOT / "scripts/run_rft_guided.sh").read_text(encoding="utf-8")
    line, = [row[len("#   "):] for row in header.splitlines() if row.startswith("#   setsid ")]
    return line


def need_launch_tools():
    for program in ("flock", "setsid", "tail"):
        if not shutil.which(program):
            pytest.skip(f"需要 {program}")


def paste(stage, **extra):
    """像用户那样把启动行交给一个 bash，返回它们的输出写进的文件。后台那一串进程继承着标准输出：用管道去接的话
    要等它们全部退出才读得到，所以写进文件；被拒绝时的那句话是它们过一会儿才打印的。每次粘贴一个新文件：上一次
    粘贴的那串进程还开着它的文件，可能几个小时后才写最后一句。"""
    out = stage / f"paste-{len(list(stage.glob('paste-*.out'))) + 1}.out"
    assert not out.exists()
    command = "{ " + launch_line() + " } > " + shlex.quote(str(out)) + " 2>&1"
    result = subprocess.run(["bash", "-c", command], cwd=stage / "repo", env=environment(stage, **extra),
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert out.exists()
    return out


# 启动行在终端里打印的两种情况：拿到锁时马上打"已启动"；锁被占时只有"已经在运行，不重复启动"这一句
STARTED = "已启动：在后台运行，日志在 rft-guided.log"
REFUSED = "已经在运行，不重复启动"
# 终端里的每一句都是事件一发生就打的（拿到锁、被拒、脚本停下），本来不用久等，这 30 秒只是机器有负载时的余量。
# 要跑完整条流水线才出现的（日志里停下的那一行、最后的「全部结束」）不去等终端，去等日志：上限同 finished（120 秒）。
# 不能拿终端的这几秒去等流水线：假脚本下跑一遍要四五秒，机器稍一负载就超时（log_shows、finished）
TERMINAL_SECONDS = 30


def waited_on_terminal(condition):
    return eventually(condition, TERMINAL_SECONDS)


def terminal(out):
    """这一次粘贴到现在为止在终端里打出来的各行。"""
    return out.read_text(encoding="utf-8").splitlines()


def refused_by(out):
    return waited_on_terminal(lambda: REFUSED in terminal(out))


def started_by(out):
    """拿到了锁：马上打印"已启动"。"""
    return waited_on_terminal(lambda: STARTED in terminal(out))


def finished(stage):
    """等 rft-guided.log 里那一次跑到最后一行，返回整份日志；到时间还没完就交回它到目前为止的内容。"""
    output = stage / "repo/rft-guided.log"
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        text = output.read_text(encoding="utf-8") if output.exists() else ""
        if text.rstrip().endswith("===== 全部结束 ====="):
            # 嵌在第 13 步里面的 rft_vllm_eval.sh 自己也打这一行：等它稳住，确认后面没有外层的收尾了
            time.sleep(0.4)
            settled = output.read_text(encoding="utf-8") if output.exists() else ""
            if settled == text:
                return text
        time.sleep(0.1)
    return output.read_text(encoding="utf-8") if output.exists() else ""


def last_line(stage):
    """这一轮的日志 rft-guided.log 的最后一行。"""
    return (stage / "repo/rft-guided.log").read_text(encoding="utf-8").rstrip("\n").splitlines()[-1]


def log_shows(stage, text, seconds=120):
    """等 rft-guided.log 里出现 text：整条流水线在假脚本下跑一遍要四五秒，机器有负载时更久。到时间还没出现是 False。"""
    output = stage / "repo/rft-guided.log"
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if output.exists() and text in output.read_text(encoding="utf-8"):
            return True
        time.sleep(0.1)
    return False


def terminal_line(out, head):
    """终端里以 head 开头的那一行；到时间还没打出来返回 None。"""
    if not waited_on_terminal(lambda: any(line.startswith(head) for line in terminal(out))):
        return None
    return next(line for line in terminal(out) if line.startswith(head))


def stopped_with(stage, out, code, sign=None):
    """脚本以 code 结束以后终端里多的那一行：退出码，后面跟着日志的最后一行；没等到返回 None。

    sign 是它在日志里停下的那一行（被 kill 的那次没有，就不等日志）：先等日志，再等终端——那一句是脚本一停就打的。
    """
    if sign is not None and not log_shows(stage, sign):
        return None
    return terminal_line(out, f"脚本已经以退出码 {code} 结束，现在不在运行。rft-guided.log 的最后一行：")


def finished_with(stage, out):
    """那一次以 0 结束：等日志跑到「===== 全部结束 =====」（同 finished），再看终端里的两行是不是"已启动"和"已经跑完"
    加日志的最后一行。"""
    if not finished(stage).rstrip().endswith("===== 全部结束 ====="):
        return False
    wanted = [STARTED, "已经跑完。rft-guided.log 的最后一行：===== 全部结束 ====="]
    return waited_on_terminal(lambda: terminal(out) == wanted)


def test_the_launch_line_in_the_header_starts_one_run_and_refuses_a_second(stage):
    """文件头里给用户粘贴的那一行：真的粘一次，趁它还在跑再粘一次，跑完以后再粘一次。三种情况分得清：拿到了锁是
    "已启动"（跑完再打一句"已经跑完"和日志的最后一行）；锁被占只有"已经在运行，不重复启动"这一句。"""
    need_launch_tools()
    output = stage / "repo/rft-guided.log"
    (stage / "calls.log").unlink(missing_ok=True)
    first = paste(stage, STUB_HOLD="2")
    assert started_by(first)
    assert log_shows(stage, "===== 1/13")                       # 那一次起来了（日志是脚本打的：等日志，上限 120 秒）
    before = output.read_text(encoding="utf-8")
    second = paste(stage)                                    # 第一次还在跑：不再起一个
    assert refused_by(second)
    time.sleep(0.3)
    assert output.read_text(encoding="utf-8").startswith(before)       # 正在跑的那份日志没有被截断
    assert terminal(second) == [REFUSED]                      # 只有这一句：没说"已启动"
    assert "13/13 经 vLLM 评各组选中的适配器" in finished(stage)
    assert finished_with(stage, first)
    assert len(calls_in(stage)) == 28 and len(starts(stage)) == 2      # 只跑了一遍，和全新跑一遍一样
    output.unlink()
    again = paste(stage)                                     # 跑完以后锁自动放开：再粘贴就是重跑，重跑全都跳过
    assert started_by(again)
    assert "===== 3/13 启动 vLLM：已完成，跳过 =====" in finished(stage) and len(starts(stage)) == 2
    assert finished_with(stage, again)
    assert_clean_shutdown(stage)


def test_the_launch_line_lets_a_run_killed_with_sigkill_be_restarted_while_its_vllm_is_alive(stage):
    """脚本被 kill -9 之后它启动的 vLLM 还活着：锁不能被它攥着，否则之后再怎么粘贴都只会说"已经在运行"。终端里说的
    也是"已经以退出码 137 结束、现在不在运行"，不是"已经在运行"。"""
    need_launch_tools()
    output, script = stage / "repo/rft-guided.log", stage / "state/script.pid"
    # 前缀采样停 10 秒：用不了这么久，采样脚本一启动（PID 文件一出现）就 kill；留出余量给机器有负载的时候
    first = paste(stage, STUB_HOLD="10")
    assert started_by(first)
    assert eventually(lambda: script.exists(), 120) and alive(stage)     # 走到采样这一步：等的上限同 finished
    pid = int(script.read_text())
    os.kill(pid, signal.SIGKILL)
    assert eventually(lambda: not running(pid))
    stopped = stopped_with(stage, first, 137)
    assert stopped == f"脚本已经以退出码 137 结束，现在不在运行。rft-guided.log 的最后一行：{last_line(stage)}"
    assert alive(stage)                                      # 服务成了没人管的，但它不攥着锁
    assert REFUSED not in terminal(first)
    output.rename(stage / "killed.log")
    again = paste(stage)
    assert started_by(again)                                 # 新的一次起来了：锁已经放开
    assert finished_with(stage, again)
    assert REFUSED not in terminal(again)
    assert_clean_shutdown(stage)


@pytest.mark.parametrize("failure, sign", [
    ({"STUB_EVAL_FAIL": "F-control/dev=3"}, "===== 没有全部完成："),
    ({"STUB_FAIL": "sample.guided-prefix=3"}, "前缀引导采样三次都没完成，停止。")],
    ids=["not-all-done", "stopped-by-an-error"])
def test_the_launch_line_says_the_script_stopped_instead_of_claiming_it_is_still_running(stage, failure, sign):
    """脚本以非 0 结束：走到最后但没全部做完（它自己以 1 退出），或者中途出错停下。以前这两种情况终端里打的都是
    "已经在运行，不重复启动"——其实那一个已经停了。现在打的是"脚本已经以退出码 1 结束，现在不在运行"和日志的最后
    一行；之后再粘贴是重跑，不是"不重复启动"。"""
    need_launch_tools()
    output = stage / "repo/rft-guided.log"
    first = paste(stage, **failure)
    assert started_by(first)
    stopped = stopped_with(stage, first, 1, sign)             # 等日志里出现它停下的那一行，再看终端
    assert sign in output.read_text(encoding="utf-8")
    assert stopped == f"脚本已经以退出码 1 结束，现在不在运行。rft-guided.log 的最后一行：{last_line(stage)}"
    assert terminal(first) == [STARTED, stopped]              # 没有"已经在运行"：它已经不在了
    again = paste(stage)                                     # 那一个已经停了、锁放开了：再粘贴就是重跑
    assert started_by(again) and REFUSED not in terminal(again)
    assert finished_with(stage, again)                       # 只补没做完的，这一轮做完
    assert_clean_shutdown(stage)


def test_the_launch_line_says_so_when_the_lock_itself_cannot_be_taken(stage):
    """flock 自己没成功（这里让锁文件成为一个指向自己的符号链接，谁的权限都打不开它）：既没有启动，也不是"已经在
    运行"——打的是"没有启动"，flock 自己的报错就在这一行的上面。"""
    need_launch_tools()
    (stage / "rft-guided.lock").symlink_to("rft-guided.lock")
    out = paste(stage)
    assert waited_on_terminal(lambda: terminal(out)[-1:] == ["没有启动：加锁没有成功，看上面 flock 的报错"])
    assert len(terminal(out)) == 2 and "rft-guided.lock" in terminal(out)[0]     # 上面一行是 flock 自己的报错
    assert STARTED not in terminal(out) and REFUSED not in terminal(out)
    assert not (stage / "repo/rft-guided.log").exists() and calls_in(stage) == [] and starts(stage) == []


def test_the_launch_line_keeps_running_and_keeps_the_lock_after_the_terminal_is_closed(stage):
    """在伪终端里开一个交互式 bash、粘贴这一行，然后关掉终端（用户关掉网页）：跑着的那一次不受影响，锁也还在。
    脚本停下以后往已经关掉的终端里打的那几句打不出去，也不碍事。"""
    need_launch_tools()
    pid, master = pty.fork()
    if pid == 0:                                             # 子进程：伪终端里的交互式 bash，工作目录是临时仓库
        try:
            os.chdir(stage / "repo")
            os.execvpe("bash", ["bash", "--norc", "-i"], environment(stage, STUB_HOLD="2"))
        except BaseException:
            os._exit(127)
    os.write(master, (launch_line() + "\n").encode("utf-8"))
    assert log_shows(stage, "===== 1/13")                       # 那一次起来了（日志是脚本打的：等日志，上限 120 秒）

    def gone():
        try:
            return os.waitpid(pid, os.WNOHANG)[0] != 0       # 收回过一次之后再问会报错：那也是已经退出了
        except ChildProcessError:
            return True

    os.close(master)                                         # 挂断：终端里的 bash 收到 SIGHUP 退出
    assert eventually(gone)
    assert refused_by(paste(stage))                          # 那一次还在跑：锁还在
    assert finished(stage).rstrip().endswith("===== 全部结束 =====")     # 关掉终端也照常跑完


# ---- 内嵌的 Python 片段：抽出来用真解释器在合成数据上跑 ----


def snippets():
    """run_rft_guided.sh 里内嵌的 Python 片段，按出现的先后：(调用那一行, 片段正文)。"""
    text = (ROOT / "scripts/run_rft_guided.sh").read_text(encoding="utf-8")
    return [(call.strip(), body) for call, body in HEREDOC.findall(text)]


def run_snippet(index, *arguments, cwd=None):
    call, body = snippets()[index]
    return subprocess.run([sys.executable, "-", *arguments], input=body, text=True, capture_output=True, timeout=60,
                          cwd=cwd)


def test_the_embedded_snippets_are_all_picked_up():
    calls = [call for call, _ in snippets()]
    assert len(calls) == 8                                   # 多了或少了片段，下面按次序取的就要跟着改
    assert ['"$TASKS"' in calls[0], '"$TIERS"' in calls[1], '"$RUN/skills.jsonl"' in calls[2], '"$1" "$2"' in calls[3],
            '"$SFT/F-control/train.jsonl" "$ROUND1_SFT/train.jsonl" "$@"' in calls[4],
            '"$SFT/F-control/dataset-manifest.json"' in calls[5], '"$ROUND1_SFT/train.jsonl"' in calls[6],
            'F-control "${guided_groups[@]}"' in calls[7]] == [True] * 8


def test_the_probe_snippet_keeps_only_the_zero_pass_tasks_in_the_chosen_tiers(tmp_path):
    manifest, output = tmp_path / "manifest.json", tmp_path / "deep/probe-ids.txt"
    write(manifest, {"zero_pass_tasks": ["h3v_t2", "h3_t1", "s3_t1", "v2_t1", "h3_t10", "h3v_t1"]})
    result = run_snippet(1, str(manifest), "variant,hard", str(output))
    assert result.returncode == 0, result.stderr
    assert output.read_text(encoding="utf-8") == "h3_t1\nh3_t10\nh3v_t1\nh3v_t2\n"       # 一行一个，按题号排好
    assert result.stdout == ("探针题：上一轮零通过的 train 题里属于 hard、variant 层的共 4 道，写在 "
                             + str(output) + "\n")

    run_snippet(1, str(manifest), "default", str(output))      # 只要原有题那一层（合成题各有自己的层）
    assert output.read_text(encoding="utf-8") == "v2_t1\n"
    run_snippet(1, str(manifest), "synth_hard", str(output))   # 分层同 tiers.py
    assert output.read_text(encoding="utf-8") == "s3_t1\n"
    run_snippet(1, str(manifest), "hard", str(output))
    assert output.read_text(encoding="utf-8") == "h3_t1\nh3_t10\n"

    write(manifest, {"zero_pass_tasks": ["v2_t1"]})
    result = run_snippet(1, str(manifest), "hard,variant", str(output))
    assert result.returncode == 1 and result.stdout == NO_PROBE + "\n"
    assert output.read_text(encoding="utf-8") == "h3_t1\nh3_t10\n"      # 没有可引导的题：旧文件原样留着

    write(manifest, {"tiers": {}})
    assert run_snippet(1, str(manifest), "hard", str(output)).returncode == 1   # 清单里没有 zero_pass_tasks
    assert run_snippet(1, str(tmp_path / "nowhere.json"), "hard", str(output)).returncode == 1


def test_the_skills_snippet_counts_what_the_sampler_would_actually_use(tmp_path):
    path = tmp_path / "skills.jsonl"
    call, _ = snippets()[2]
    assert '"$RUN/skills.jsonl"' in call
    assert run_snippet(2, str(path)).stdout == "0 0\n"        # 文件还没有：一条也用不上；出错的 0 道
    write(path, '{"task_id": "h3_a", "status": "ok", "text": "先核对边界。"}\n'
                '{"task_id": "h3_b", "status": "leaky"}\n'
                '{"task_id": "h3_c", "status": "error"}\n'
                '{"task_id": "h3_d", "status": "ok"}\n'                       # 没有正文：用不上
                '{"task_id": "h3_e", "status": "ok", "text": "比较改动前后。"}\n'
                '{"task_id": "h3_f", "status": "ok", "text": "先写检查点。"}\n'
                '{"task_id": "h3_g", "status": "ok", "text": "写到一半"\n'    # 上次没写完的坏行
                '{"task_id": "h3_c", "status": "ok", "text": "重写好了。"}\n'  # 同一道题以最后一行为准：不再是出错
                '{"task_id": "h3_a", "status": "leaky"}\n'
                '[1, 2]\n')
    assert run_snippet(2, str(path)).stdout == "3 0\n"        # h3_c、h3_e、h3_f
    with path.open("ab") as handle:                           # 最后一行截在某个汉字中间（写技能写到一半被打断）
        handle.write('{"task_id": "h3_h", "status": "ok", "text": "先看边界情况'.encode("utf-8")[:-1])
    counted = run_snippet(2, str(path))
    assert counted.returncode == 0 and counted.stdout == "3 0\n"      # 坏尾跳过：原来这里是 UnicodeDecodeError
    assert "Traceback" not in counted.stderr
    with path.open("ab") as handle:
        handle.write('\n{"task_id": "h3_i", "status": "error"}\n'.encode("utf-8"))
    assert run_snippet(2, str(path)).stdout == "3 1\n"        # 出错的题另数一个：D-skill 组因它没做成时要看得出来
    write(path, '{"task_id": "h3_a\u2028第二行", "status": "ok", "text": "正文里也有\u0085一个"}\n')
    assert run_snippet(2, str(path)).stdout == "1 0\n"        # 一条记录一行，行尾只认 "\n"


def test_a_skill_count_that_cannot_be_made_stops_the_run_instead_of_switching_skills_off(stage):
    """数可用技能的片段以非 0 结束，或者打出来的不是两个非负整数：不能把空串当成 0 条、悄悄不做技能引导——停在这里。"""
    script = stage / "repo/scripts/run_rft_guided.sh"
    head, separator, tail = script.read_text(encoding="utf-8").partition(
        '  counted="$(python - "$RUN/skills.jsonl" <<\'PY\'\n')
    assert separator
    for body in ("import sys\nsys.exit(3)\n", "print(\"数不出来\")\n", "print(\"5 -1\")\n"):
        script.write_text(head + separator + body + "PY\n)" + tail, encoding="utf-8")
        result, calls = run(stage)
        assert result.returncode == 1, result.stdout[-2000:] + result.stderr[-1000:]
        assert ("数不出可用的技能（data/rft-guided/skills.jsonl 读不了，或者格式不对）：把上面的报错贴给我"
                in result.stdout)
        assert not calls_of(calls, "sample") and "可用的技能只有" not in result.stdout
        assert "===== 5/13" not in result.stdout and "===== 全部结束" not in result.stdout
        assert_clean_shutdown(stage)


def test_a_skills_file_with_a_cut_tail_from_the_last_run_still_counts_when_the_teacher_cannot_be_reached(stage):
    """上一跑把技能写好了、采样还没开始；技能文件后来多出半截坏尾（截在汉字中间），这一跑又没有密钥：写技能以 2 退出，
    连坏尾都没来得及整理。数可用技能时坏尾跳过、上一跑那 5 条照常用——以前片段在这里崩掉（UnicodeDecodeError），
    空串被当成 0 条，技能引导和 D-skill 组就被悄悄关掉了。"""
    repo = stage / "repo"
    broken, _ = run(stage, STUB_FAIL="sample.guided-prefix=3")        # 上一跑：技能写好了，前缀采样没做完
    assert broken.returncode == 1 and "前缀引导采样三次都没完成，停止。" in broken.stdout
    skills = repo / RUN / "skills.jsonl"
    assert len(skills.read_text(encoding="utf-8").splitlines()) == 5
    with skills.open("ab") as handle:
        handle.write('{"task_id": "h3_t9", "status": "ok", "text": "先看边界情况'.encode("utf-8")[:-1])
    result, calls = run(stage, STUB_SKILLS_EXIT="2")                     # 这一跑：写技能没碰这个文件就退出了
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-1000:]
    assert "写技能没有做成（write_skills.py 以退出码 2 结束" in result.stdout
    assert "可用的技能只有" not in result.stdout and NOT_DONE not in result.stdout   # 技能够用：这一轮照样做完
    assert [flag(args, "tag") for args in calls_of(calls, "sample")] == ["guided-prefix", "guided-skill"]
    assert "D-skill" in calls_of(calls, "control")[0]
    assert tasks_of(repo, "D-skill") == [*BASE, "h3_t3", "h3v_t1", UNMATCHABLE]
    assert result.stdout.rstrip().endswith("===== 全部结束 =====")
    assert_clean_shutdown(stage)


def test_the_dataset_snippet_counts_guided_records_against_the_base_set(tmp_path):
    base, data = tmp_path / "round1/sft/verified/train.jsonl", tmp_path / "sft/D-prefix"
    write(base, "".join(json.dumps({"task_id": task}) + "\n" for task in ("h3_t9", "v2_t2")))
    call, _ = snippets()[6]
    assert '"$ROUND1_SFT/train.jsonl"' in call                 # 引导题是相对第 1 轮的验证组说的
    write(data / "train.jsonl", "".join(json.dumps({"task_id": task}) + "\n"
                                       for task in ("h3_t9", "h3_t1", "h3v_t1", "v2_t2", "v2_t5")))
    write(data / "dev.jsonl", '{"task_id": "h3_d1"}\n\n')
    result = run_snippet(6, str(data), "D-prefix", str(base))
    assert result.returncode == 0, result.stderr
    assert result.stdout == ("结果：D-prefix 训练集 5 条 / 5 题，其中引导题 3 条"
                             "（原有题 1、难题 1、变体 1）；dev 1 条\n")
    write(data / "train.jsonl", '{"task_id": "h3_t9"}\n{"task_id": "h3_t9"}\n')    # 没有引导题，还有重复的题
    assert run_snippet(6, str(data), "D-prefix", str(base)).stdout == \
        "结果：D-prefix 训练集 2 条 / 1 题，其中引导题 0 条；dev 1 条\n"
    shutil.rmtree(data)                                        # 数据目录还没准备好：报错退出，由调用方停下
    assert run_snippet(6, str(data), "D-prefix", str(base)).returncode == 1


def test_the_drop_snippet_tells_whether_a_dataset_still_contains_the_dropped_tasks(tmp_path):
    data, drop = tmp_path / "sft/D-prefix", tmp_path / "unmatched-tasks.txt"
    call, _ = snippets()[3]
    assert '"$1" "$2"' in call
    write(data / "train.jsonl", '{"task_id": "h3_a"}\n{"task_id": "h3v_b"}\n')
    write(drop, "h3v_b\n\nh3_t9\n")
    assert run_snippet(3, str(data), str(drop)).returncode == 1          # h3v_b 还在：没有去过掉
    write(data / "train.jsonl", '{"task_id": "h3_a"}\n')
    assert run_snippet(3, str(data), str(drop)).returncode == 0          # 题号文件里的题一道都不在了
    write(drop, "")
    assert run_snippet(3, str(data), str(drop)).returncode == 0
    (data / "train.jsonl").unlink()
    assert run_snippet(3, str(data), str(drop)).returncode == 1          # 读不出来就当作没去过掉
    assert run_snippet(3, str(data), str(tmp_path / "nowhere.txt")).returncode == 1


def test_the_pairing_snippet_tells_whether_the_control_matches_the_current_guided_groups(tmp_path):
    """两次运行之间 SKILL 变了：F-control 的题目必须正好是"基础集 ∪ 现在这几个引导组的引导题"。"""
    base, control = tmp_path / "round1/train.jsonl", tmp_path / "sft/F-control/train.jsonl"
    prefix, skill = tmp_path / "sft/D-prefix/train.jsonl", tmp_path / "sft/D-skill/train.jsonl"
    for path, tasks in ((base, "ab"), (prefix, "abc"), (skill, "abd"), (control, "abcd")):
        write(path, "".join(json.dumps({"task_id": task}) + "\n" for task in tasks))
    call, _ = snippets()[4]
    assert '"$SFT/F-control/train.jsonl" "$ROUND1_SFT/train.jsonl" "$@"' in call
    directories = [str(path.parent) for path in (prefix, skill)]       # 调用方传的是各引导组的数据目录
    assert run_snippet(4, str(control), str(base), *directories).returncode == 0      # 按两组配的
    assert run_snippet(4, str(control), str(base), str(prefix.parent)).returncode == 1   # 只剩一组：对照组多出了 d
    write(control, "".join(json.dumps({"task_id": task}) + "\n" for task in "abc"))
    assert run_snippet(4, str(control), str(base), str(prefix.parent)).returncode == 0   # 按这一组配的
    skill.unlink()
    assert run_snippet(4, str(control), str(base), *directories).returncode == 1      # 有一组还没有数据
    control.unlink()
    assert run_snippet(4, str(control), str(base), str(prefix.parent)).returncode == 1   # 对照本身读不出来


def test_the_snippet_that_names_the_groups_of_the_old_control_reads_its_manifest(tmp_path):
    manifest = tmp_path / "sft/F-control/dataset-manifest.json"
    call, _ = snippets()[5]
    assert '"$SFT/F-control/dataset-manifest.json"' in call
    unknown = "是按哪几个引导组配的看不出来（它的 manifest 里没记）\n"
    assert run_snippet(5, str(manifest)).returncode == 0 and run_snippet(5, str(manifest)).stdout == unknown   # 还没有
    write(manifest, {"guided_from": ["data/rft-guided/sft/D-prefix"]})
    assert run_snippet(5, str(manifest)).stdout == "是按 D-prefix 配的\n"
    write(manifest, {"guided_from": ["x/D-skill", "x/D-prefix"]})
    assert run_snippet(5, str(manifest)).stdout == "是按 D-skill、D-prefix 配的\n"
    write(manifest, {"samples": 3})                               # 没记引导组
    assert run_snippet(5, str(manifest)).stdout == unknown
    write(manifest, "{not json")                                  # 读不出来也不碍事：它只用来拼一句话
    assert run_snippet(5, str(manifest)).returncode == 0 and run_snippet(5, str(manifest)).stdout == unknown


def test_the_alignment_snippet_compares_the_union_of_the_d_groups_with_the_control(tmp_path):
    root = tmp_path / "checkpoints"
    call, _ = snippets()[7]
    assert 'F-control "${guided_groups[@]}"' in call          # 各个 D 组都拿来和对照组比

    def trained(group, tasks):
        write(root / group / "training-manifest.json", {"data": {"task_ids": tasks}})

    trained("D-prefix", ["a", "b"])
    trained("D-skill", ["c", "d"])
    trained("F-control", ["a", "c"])                          # b、d 的对照答案被训练入口丢掉了
    result = run_snippet(7, str(root), "F-control", "D-prefix", "D-skill")
    assert result.returncode == 0, result.stderr
    assert result.stdout == MISMATCH + "['b', 'd']，只在 F-control 里的 []。" + LOST_TAIL + "\n"
    trained("F-control", ["a", "b", "c", "d"])                # 每个 D 组各自只有一部分，合起来和对照组一样
    assert run_snippet(7, str(root), "F-control", "D-prefix", "D-skill").stdout == \
        "结果：训练时实际加载的题 D-prefix、D-skill 和 F-control 逐题相同（4 道）\n"
    result = run_snippet(7, str(root), "F-control", "D-prefix")     # 只拿一组来比：对照组多出来的也算不一样
    assert "只在 D-prefix 里的 []，只在 F-control 里的 ['c', 'd']" in result.stdout
    trained("F-control", ["a", "b", "e"])
    assert "只在 F-control 里的 ['e']" in run_snippet(7, str(root), "F-control", "D-prefix").stdout
    write(root / "D-skill/training-manifest.json", '{"status": "complete"}')      # 旧的清单里没记题号：没什么可核对的
    assert run_snippet(7, str(root), "F-control", "D-skill").stdout == ""
    assert run_snippet(7, str(root), "F-control", "nobody").stdout == ""         # 这一组还没训过


def test_the_task_count_snippet_also_counts_synthetic_tasks_when_there_are_any(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    where = [("h3_t1", "train"), ("h3v_t1", "train"), ("h3_d1", "dev"),
             ("v2_t1", "train"), ("v2_d1", "dev"), ("v2_x1", "test")]
    write(tasks, "".join(json.dumps({"task_id": task, "split": split}) + "\n" for task, split in where))
    result = run_snippet(0, str(tasks))
    assert result.returncode == 0, result.stderr
    assert result.stdout == "任务文件：6 题，train 3 / dev 2 / test 1；其中难题来源族 train 2 / dev 1\n"
    with tasks.open("a", encoding="utf-8") as handle:
        handle.write('{"task_id": "s3_t1", "split": "train"}\n{"task_id": "s3v_d1", "split": "dev"}\n')
    assert run_snippet(0, str(tasks)).stdout == (
        "任务文件：8 题，train 4 / dev 3 / test 1；其中难题来源族 train 2 / dev 1\n"
        "其中合成题（s3_、s3v_ 开头）train 1 / dev 1 / test 0\n")
