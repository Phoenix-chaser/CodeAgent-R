"""RFT 的采样阶段：让基座模型（经本机 vLLM 服务）对冻结任务各采样若干次，逐个判分。

通过的样本与教师采集（collect_data.py）是同一种记录格式，所以 prepare_data.py 可以直接
重放验证、去重并限制每题条数。此外还保留全部样本和每题的通过次数：

- ``<split>-raw.jsonl``        模型的原始输出，一条一次采样
- ``<split>-graded.jsonl``     全部样本及判分结果（对照组和后续诊断用）
- ``<split>-candidates.jsonl`` 其中判分通过的，交给 prepare_data.py
- ``<split>-protocol.json``    采样协议；续跑时核对
- ``<split>-manifest.json``    每层每题的通过次数、pass@k、失败类型

每一步都能续跑：已经生成的样本不重新生成，已经判分的不重新判分。退出码 3 表示还有
槽位没完成（服务中断、判分本身出错，或者记录写进文件后读不回来），原样重跑会只补这些
槽位；这时不写 manifest（上一次留下的也去掉）。manifest 存在就表示
``slots = raw_records = graded_records``：每个槽位都有一条读得回来的采样和判分记录，
调用方（run_rft.sh、run_rft_guided.sh）也是按这一点认"采样做完了"的。

JSONL 里一条记录一行，行尾只认 ``"\n"``：模型回答里的 U+2028、U+2029、U+0085 写成转义
（``_shared_jsonl.jsonl_line``），读的时候不用 ``str.splitlines()``（它把这三个字符也当成换行）——
prepare_data.py、rft_control.py 读写这些记录用的是同一个模块。

引导采样（``--hint-kind prefix|skill``）：只对指定的一批题（通常是上一轮一次都没通过的题，
用 ``--only-zero-pass`` 给上一轮的 manifest）发**带提示**的消息，采出判分通过的答案。
记录里的 ``messages`` 是不带提示的标准提示，提示的信息记在 ``provenance.guided`` 里，
所以训练时模型看不到提示。``--tag`` 给输出文件名加标记（例如 ``train-guided-prefix-*.jsonl``）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
from typing import NoReturn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agentlab.benchmark.builder import build_all
from agentlab.benchmark.protocol import build_messages, load_specs, protocol_hash, spec_fingerprint
from agentlab.benchmark.tiers import LABELS, TIERS, tier_of
from agentlab.sampling import vllm_client
from agentlab.sampling.hints import KINDS, prefix_hint, restated, skill_hint, with_hint
from _shared_extract import extract_code
from _shared_jsonl import cut_partial_tail, jsonl_line, read_jsonl
from collect_data import grade_candidate
from evaluate import failure_type


def code_hash(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


def pass_at_k(n: int, c: int, k: int) -> float:
    """n 次采样里通过 c 次，任取 k 次至少通过一次的概率（无偏估计）。"""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def slot_label(slot: dict) -> str:
    return f"{slot['task_id']} [{slot['candidate_idx']}]"


def unfinished(message: str, manifest: Path) -> NoReturn:
    """还有槽位没完成：说明原因，以 3 退出。上一次（或者旧版本）留下的 manifest 也去掉——它说这份采样是齐的，
    现在不是了，不能让调用方照着它往下走。"""
    print(message, file=sys.stderr)
    if manifest.exists():
        manifest.unlink()
        print(f"去掉了 {manifest.name}：采样没有凑齐，不能留一份说它是齐的清单", file=sys.stderr)
    raise SystemExit(3)


def slot_key(record: dict) -> tuple:
    return record["task_id"], record["candidate_idx"]


def load_skills(path: Path) -> dict[str, str]:
    """write_skills.py 的输出里可用的技能：{task_id: 技能正文}。"""
    return {record["task_id"]: record["text"] for record in read_jsonl(path)
            if record.get("status") == "ok" and record.get("text")}


def hint_plan(spec: dict, args, skills: dict) -> list:
    """一道题要用的提示，每种各采 ``args.samples`` 次；不引导时只有"没有提示"一种。"""
    if args.hint_kind == "none":
        return [None]
    if args.hint_kind == "prefix":
        return [prefix_hint(spec["fixed"], fraction) for fraction in args.hint_fractions]
    return [skill_hint(skills[spec["task_id"]])]


def samples_per_task(args) -> int:
    if args.hint_kind == "prefix":
        return args.samples * len(args.hint_fractions)
    return args.samples


def make_slots(specs: list[dict], args, skills: dict | None = None) -> list[dict]:
    """每个 (题, 提示, 第几次采样) 一个槽位。偶数号用常规温度，奇数号用探索温度。"""
    slots = []
    for position, spec in enumerate(specs):
        canonical = build_messages(spec)
        index = 0
        for hint in hint_plan(spec, args, skills or {}):
            for repeat in range(args.samples):
                slot = {
                    "task_id": spec["task_id"], "candidate_idx": index,
                    "temperature": args.temperature if repeat % 2 == 0 else args.exploration_temperature,
                    "seed": args.seed + position * 1000 + index,
                    "messages": canonical,
                }
                if hint is not None:
                    # 发出去的带提示；grade_sample 写记录时另取不带提示的标准提示
                    slot["messages"] = with_hint(canonical, hint)
                    slot["hint"] = {key: hint[key] for key in ("kind", "fraction", "lines_shown", "lines_total")
                                    if key in hint}
                slots.append(slot)
                index += 1
    return slots


def grade_sample(spec: dict, raw: dict, model: str, task_dir: Path, grade_dir: Path) -> dict:
    """给一次采样判分，返回与教师采集同格式的记录。没通过的记录标为 attempt。"""
    fixed = extract_code(raw["text"])
    usage = {"input_tokens": raw.get("prompt_tokens"), "output_tokens": raw.get("completion_tokens")}
    base = {
        "record_type": "candidate",
        "sample_id": f"rft-{spec['task_id']}-{raw['candidate_idx']:03d}",
        "task_id": spec["task_id"], "task_hash": spec_fingerprint(spec),
        "candidate_idx": raw["candidate_idx"], "source_group": spec["source_group"],
        "split": spec["split"], "mode": "S1_final_patch", "temperature": raw["temperature"],
        "seed": raw["seed"], "messages": build_messages(spec),
        "raw_response": raw["text"], "finish_reason": raw["finish_reason"], "usage": usage,
    }
    provenance = {"generator": "self", "backend": "vllm", "model": model,
                  "grader_version": "three_layer_v1", "usage": usage,
                  "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if raw.get("hint"):
        guided = dict(raw["hint"])
        if guided["kind"] == "prefix":          # 有没有照抄提示里的开头
            guided["restated"] = restated(raw["text"], prefix_hint(spec["fixed"], guided["fraction"]))
        provenance["guided"] = guided
    target = {"role": "assistant", "content": fixed}
    if raw["finish_reason"] != "stop" or not fixed.strip():
        kind = "incomplete_response" if raw["finish_reason"] != "stop" else "no_code"
        return {**base, "record_type": "attempt", "target": target,
                "provenance": {**provenance, "verdict": kind, "strict_pass": False,
                               "failure_type": kind}}
    try:
        verdict, strict_pass, details = grade_candidate(spec, fixed, task_dir, grade_dir)
        # 失败类型的划分与评测报告一致（public_test_failure / private_test_failure / …）
        failure = None if strict_pass else failure_type(fixed, verdict, details)
    except subprocess.TimeoutExpired:
        verdict, strict_pass, details, failure = "timeout", False, {}, "execution_timeout"
    return {**base, "record_type": "candidate" if strict_pass else "attempt",
            "code_hash": code_hash(fixed), "target": target, "grader_details": details,
            "provenance": {**provenance, "verdict": verdict, "strict_pass": strict_pass,
                           "failure_type": failure}}


def bar_line(done: int, total: int, started: float, label: str) -> str:
    elapsed = time.monotonic() - started
    eta = elapsed / done * (total - done) if done else 0
    filled = 24 * done // total
    return (f"[{'#' * filled}{'-' * (24 - filled)}] {done}/{total} ({done * 100 // total}%) "
            f"{label} · 已用 {elapsed / 60:.1f}m · 剩余≈{eta / 60:.1f}m")


def summarise(graded: list[dict], specs: list[dict], samples: int, pass_k: int | None = None) -> dict:
    """每层每题的通过次数。``samples`` 是每题应有的样本数；``pass_k`` 是算 pass@k 的 k，
    0 表示不算（引导采样时每题有多种提示，pass@k 没有意义）。"""
    k = samples if pass_k is None else pass_k
    per_task = defaultdict(lambda: [0, 0])                    # 通过数、已判分数
    for record in graded:
        per_task[record["task_id"]][0] += bool(record["provenance"]["strict_pass"])
        per_task[record["task_id"]][1] += 1
    tiers = {}
    for tier in TIERS:
        ids = [spec["task_id"] for spec in specs if tier_of(spec["task_id"]) == tier]
        passes = sum(per_task[task][0] for task in ids)
        graded_count = sum(per_task[task][1] for task in ids)
        complete = [task for task in ids if per_task[task][1] == samples]
        tiers[tier] = {
            "tasks": len(ids), "graded_samples": graded_count, "passing_samples": passes,
            "solved_tasks": sum(per_task[task][0] > 0 for task in ids),
            "pass_at_1": passes / graded_count if graded_count else None,
            "pass_at_k": (sum(pass_at_k(k, per_task[task][0], k) for task in complete)
                          / len(complete)) if complete and k else None,
            "fully_graded_tasks": len(complete),
        }
    failures = Counter(record["provenance"].get("failure_type") for record in graded
                       if not record["provenance"]["strict_pass"])
    return {
        "tiers": tiers,
        "failure_types": dict(failures),
        "task_pass_counts": {task: counts[0] for task, counts in sorted(per_task.items())},
        "zero_pass_tasks": sorted(task for task, counts in per_task.items() if counts[0] == 0),
    }


def lowest_fraction_only(records: list[dict]) -> list[dict]:
    """前缀引导：每道题只留"最短的、成功过的比例"的样本（POPE 的取法）。比例越短，
    模型自己写的部分越多；更长的比例采到的答案留在 graded 文件里供分析。"""
    lowest: dict[str, float] = {}
    for record in records:
        fraction = record["provenance"]["guided"]["fraction"]
        lowest[record["task_id"]] = min(lowest.get(record["task_id"], fraction), fraction)
    return [record for record in records
            if record["provenance"]["guided"]["fraction"] == lowest[record["task_id"]]]


def hint_summary(graded: list[dict]) -> list[dict]:
    """每种提示一行：槽位数、通过样本、解出的题；再看其中有多少照抄了提示里的开头。"""
    groups = defaultdict(list)
    for record in graded:
        guided = record["provenance"].get("guided")
        if guided:
            groups[(guided["kind"], guided.get("fraction"))].append(record)
    rows = []
    for (kind, fraction), records in sorted(groups.items(), key=lambda item: item[0][1] or 0):
        passed = [record for record in records if record["provenance"]["strict_pass"]]
        restated = [record for record in records if record["provenance"]["guided"].get("restated")]
        rows.append({
            "kind": kind, "fraction": fraction, "slots": len(records),
            "passing_samples": len(passed),
            "tasks": len({record["task_id"] for record in records}),
            "solved_tasks": len({record["task_id"] for record in passed}),
            "restated": len(restated),
            "passing_restated": sum(record["provenance"]["strict_pass"] for record in restated),
        })
    return rows


def hint_label(row: dict) -> str:
    return "技能提示" if row["kind"] == "skill" else f"前缀 {row['fraction']:g}"


def display_width(text: str) -> int:
    return sum(2 if ord(char) > 0x2E7F else 1 for char in text)


def cell(text: str, size: int) -> str:
    return str(text) + " " * (size - display_width(str(text)))


def print_summary(split: str, stats: dict, samples: int) -> None:
    print(f"── 采样汇总（{split}）──")
    print(cell("层", 10) + cell("题数", 6) + cell("通过/已判", 12) + cell("解出的题", 10)
          + cell("pass@1", 9) + f"pass@{samples}")

    def percent(value):
        return "-" if value is None else f"{value:.1%}"
    for tier in TIERS:
        row = stats["tiers"][tier]
        if not row["tasks"]:
            continue
        print(cell(LABELS[tier], 10) + cell(row["tasks"], 6)
              + cell(f"{row['passing_samples']}/{row['graded_samples']}", 12)
              + cell(f"{row['solved_tasks']}/{row['tasks']}", 10)
              + cell(percent(row["pass_at_1"]), 9) + percent(row["pass_at_k"]))
    print(f"结果：{split} " + "、".join(
        f"{LABELS[tier]} 解出 {stats['tiers'][tier]['solved_tasks']}/{stats['tiers'][tier]['tasks']}"
        f"（通过样本 {stats['tiers'][tier]['passing_samples']}/{stats['tiers'][tier]['graded_samples']}）"
        for tier in TIERS if stats["tiers"][tier]["tasks"]))
    print(f"失败类型：{stats['failure_types']}")
    print(f"一次都没通过的题：{len(stats['zero_pass_tasks'])} 道 {stats['zero_pass_tasks'][:8]}")


def file_digest(path: str | None) -> str | None:
    """协议里记下输入文件的内容摘要，换了清单或技能文件就不能混进同一个目录。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks-file", required=True, help="冻结的任务文件（带 split）")
    parser.add_argument("--split", required=True, choices=["train", "dev"])
    parser.add_argument("--tiers", default="default,hard,variant",
                        help="采样的层，逗号分隔：default（原有题）/ hard（难题）/ variant（单缺陷变体）")
    parser.add_argument("--samples", type=int, default=8, help="每题采样次数（引导采样时是每种提示各几次）")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--exploration-temperature", type=float, default=1.0,
                        help="奇数号采样用的温度")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20, help="服务不支持这个参数时不发")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vllm-url", default=vllm_client.DEFAULT_URL)
    parser.add_argument("--model", default=None, help="服务上的模型名；不填时用服务报告的唯一模型")
    parser.add_argument("--concurrency", type=int, default=24, help="同时发出的请求数")
    parser.add_argument("--grade-workers", type=int, default=4, help="同时判分的进程数")
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hint-kind", choices=["none", *KINDS], default="none",
                        help="引导采样：prefix 把参考解的开头放进提示，skill 放教师写的通用技能")
    parser.add_argument("--hint-fractions", default="0.25,0.5,0.75",
                        help="前缀提示各给参考解的多大比例（逗号分隔）")
    parser.add_argument("--only-zero-pass", default=None,
                        help="只采这个清单里的题：上一轮 <split>-manifest.json（读它的 zero_pass_tasks）")
    parser.add_argument("--skills-file", default=None, help="write_skills.py 的输出（每行一道题的技能）")
    parser.add_argument("--tag", default="", help="输出文件名里的标记，例如 guided-prefix")
    args = parser.parse_args()
    tiers = [tier.strip() for tier in args.tiers.split(",") if tier.strip()]
    if args.samples < 1 or args.concurrency < 1 or args.grade_workers < 1:
        parser.error("samples / concurrency / grade-workers must be positive")
    if not 0 < args.top_p <= 1 or args.temperature <= 0 or args.exploration_temperature <= 0:
        parser.error("temperatures must be positive and top-p must lie in (0, 1]")
    if not set(tiers) <= set(TIERS):
        parser.error(f"unknown tier in --tiers; choose from {TIERS}")
    try:
        args.hint_fractions = sorted({float(piece) for piece in args.hint_fractions.split(",") if piece.strip()})
    except ValueError:
        parser.error("--hint-fractions must be numbers, e.g. 0.25,0.5,0.75")
    if args.hint_kind == "prefix" and (not args.hint_fractions
                                       or not all(0 < fraction < 1 for fraction in args.hint_fractions)):
        parser.error("--hint-fractions must lie strictly between 0 and 1")
    if args.hint_kind == "skill" and not args.skills_file:
        parser.error("--hint-kind skill needs --skills-file (the output of write_skills.py)")
    if args.tag and not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        parser.error("--tag may only contain letters, digits, '-' and '_'")

    specs = load_specs(args.tasks_file, args.split)
    note = ""
    if args.only_zero_pass:
        wanted = set(json.loads(Path(args.only_zero_pass).read_text(encoding="utf-8"))["zero_pass_tasks"])
        specs = [spec for spec in specs if spec["task_id"] in wanted]
        note += f"；--only-zero-pass 指定 {len(wanted)} 道，其中 {len(specs)} 道在 {args.split} 里"
    skills = {}
    if args.hint_kind == "skill":
        skills = load_skills(Path(args.skills_file))
        specs = [spec for spec in specs if spec["task_id"] in skills]
        note += f"；有可用技能的 {len(specs)} 道"
    specs = [spec for spec in specs if tier_of(spec["task_id"]) in tiers]
    if not specs:
        parser.error(f"no {args.split} tasks in tiers {tiers}"
                     + (" after the --only-zero-pass / --skills-file filter" if note else ""))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{args.split}-{args.tag}" if args.tag else args.split
    model = args.model
    if model is None:
        try:
            served = vllm_client.server_models(args.vllm_url)
        except vllm_client.VLLMError as error:
            parser.error(f"连不上 vLLM 服务：{error}")
        if len(served) != 1:
            parser.error(f"服务上有多个模型 {served}；用 --model 指定一个")
        model = served[0]
    protocol = {
        "schema_version": 1, "tasks_hash": protocol_hash(specs), "split": args.split, "tiers": tiers,
        "model": model, "backend": "vllm", "vllm_url": args.vllm_url, "samples": args.samples,
        "temperature": args.temperature, "exploration_temperature": args.exploration_temperature,
        "top_p": args.top_p, "top_k": args.top_k, "max_tokens": args.max_tokens, "seed": args.seed,
    }
    if args.hint_kind != "none":
        # 不引导时协议原样不变，上一轮的采样目录照常续跑
        protocol["hint"] = {
            "kind": args.hint_kind, "samples_per_hint": args.samples,
            "fractions": args.hint_fractions if args.hint_kind == "prefix" else [],
            "zero_pass_from": file_digest(args.only_zero_pass), "skills_sha256": file_digest(args.skills_file),
        }
    protocol_path = output / f"{stem}-protocol.json"
    manifest_path = output / f"{stem}-manifest.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        options = previous.pop("sampling_options")
        if previous != protocol:
            parser.error("采样协议与这个目录里已有的不一致；换一个新的 --output-dir")
    else:
        options = vllm_client.sampling_options(args.vllm_url, model)
        protocol_path.write_text(json.dumps({**protocol, "sampling_options": options}, indent=2),
                                 encoding="utf-8")
    kwargs = dict(top_p=args.top_p, max_tokens=args.max_tokens, timeout=args.request_timeout,
                  top_k=args.top_k if options["top_k"] else None)

    expected = samples_per_task(args)
    slots = make_slots(specs, args, skills)
    raw_path = output / f"{stem}-raw.jsonl"
    graded_path = output / f"{stem}-graded.jsonl"
    for path in (raw_path, graded_path):
        lost = cut_partial_tail(path)
        if lost:
            print(f"{path.name} 的最后一行没写完（{lost} 个字节）：去掉它，这个槽位重做", file=sys.stderr)
    done_raw = {slot_key(record) for record in read_jsonl(raw_path)}
    pending = [slot for slot in slots if (slot["task_id"], slot["candidate_idx"]) not in done_raw]
    print(f"采样 {stem}：{len(specs)} 道题 × {expected} 次 = {len(slots)} 个槽位，"
          f"已有 {len(slots) - len(pending)} 个，待生成 {len(pending)} 个{note}", flush=True)
    lock = threading.Lock()
    started = time.monotonic()
    count = len(slots) - len(pending)
    failed = []

    def generate(slot):
        try:
            result = vllm_client.chat(
                args.vllm_url, model, slot["messages"], temperature=slot["temperature"],
                seed=slot["seed"] if options["seed"] else None, **kwargs)
            return None, {**slot, **result}
        except Exception as error:                          # 记下来，不中断其它槽位
            return f"{slot['task_id']} [{slot['candidate_idx']}]: {type(error).__name__}: {error}", slot

    for _, (error, record) in vllm_client.map_concurrent(generate, pending, args.concurrency):
        if error:
            failed.append(error)
            continue
        with lock:
            with raw_path.open("a", encoding="utf-8") as handle:
                handle.write(jsonl_line(record) + "\n")
            count += 1
            print(bar_line(count, len(slots), started,
                           f"采样 {stem} {record['task_id']} [{record['candidate_idx']}]"
                           f" {record['finish_reason']}"), flush=True)
    if failed:
        unfinished(f"有 {len(failed)} 个槽位没能生成，原样重跑会只补这些：{failed[:5]}", manifest_path)

    # 以落盘的记录为准。槽位都报了完成，却还有读不出记录的（那一行没写完，或者写坏了）：同样算没完成，
    # 不开始判分——带着少了样本的清单往下走，少判的那几道题会被记成"一次都没通过"
    raw_records = read_jsonl(raw_path)
    unreadable = [slot_label(slot) for slot in slots if slot_key(slot) not in {slot_key(row) for row in raw_records}]
    if unreadable:
        unfinished(f"有 {len(unreadable)} 条采样写进文件后读不出来，原样重跑会重做：{unreadable[:5]}", manifest_path)
    specs_by_id = {spec["task_id"]: spec for spec in specs}
    done_graded = {slot_key(record) for record in read_jsonl(graded_path)}
    to_grade = [record for record in raw_records if slot_key(record) not in done_graded]
    print(f"判分 {stem}：待判 {len(to_grade)} 个（同时 {args.grade_workers} 个）", flush=True)
    started = time.monotonic()
    count = len(raw_records) - len(to_grade)
    broken = []

    def grade(raw):
        try:
            return None, grade_sample(specs_by_id[raw["task_id"]], raw, model, work / "tasks",
                                      work / "grade" / f"{raw['task_id']}-{raw['candidate_idx']}")
        except Exception as error:                          # 判分本身出错：下次重做这个槽位
            return f"{raw['task_id']} [{raw['candidate_idx']}]: {type(error).__name__}: {error}", raw

    with tempfile.TemporaryDirectory(prefix="rft-sample-") as temporary:
        work = Path(temporary)
        build_all(work / "tasks", specs)
        for _, (error, record) in vllm_client.map_concurrent(grade, to_grade, args.grade_workers):
            if error:
                broken.append(error)
                continue
            with lock:
                with graded_path.open("a", encoding="utf-8") as handle:
                    handle.write(jsonl_line(record) + "\n")
                count += 1
                print(bar_line(count, len(raw_records), started,
                               f"判分 {stem} {record['task_id']} [{record['candidate_idx']}] "
                               f"{record['provenance']['verdict']}"), flush=True)
    if broken:
        unfinished(f"有 {len(broken)} 个样本判分时出错，原样重跑会只补这些：{broken[:5]}", manifest_path)

    # 以落盘的记录为准：每个槽位都要有一条读得回来的判分记录，少一条就不写候选和 manifest（退出码 3，
    # 原样重跑只补这一条）。有了 manifest，调用方就当这份采样是齐的
    graded_by_key = {slot_key(record): record for record in read_jsonl(graded_path)}
    unreadable = [slot_label(slot) for slot in slots if slot_key(slot) not in graded_by_key]
    if unreadable:
        unfinished(f"有 {len(unreadable)} 条判分记录写进文件后读不出来，原样重跑会重做：{unreadable[:5]}",
                   manifest_path)
    # 按采样的先后排列；到这里每个槽位都已判分
    graded = [graded_by_key[slot_key(record)] for record in raw_records]
    assert len(graded) == len(slots) == len(raw_records)
    candidates = [record for record in graded if record["provenance"]["strict_pass"]]
    # 前缀引导时每题只把"最短的、成功过的比例"交给训练；不引导时 candidates 原样就是全部
    keep = lowest_fraction_only(candidates) if args.hint_kind == "prefix" else candidates
    (output / f"{stem}-candidates.jsonl").write_text(
        "".join(jsonl_line(record) + "\n" for record in keep), encoding="utf-8")
    stats = summarise(graded, specs, expected,
                      pass_k=0 if args.hint_kind != "none" else None)
    manifest = {
        "schema_version": 1, "protocol": {**protocol, "sampling_options": options},
        "tasks": len(specs), "slots": len(slots), "raw_records": len(raw_records),
        "graded_records": len(graded), "candidates": len(candidates),
        "candidates_kept_for_training": len(keep),
        **stats,
    }
    if args.hint_kind != "none":
        manifest["hint"] = hint_summary(graded)
    (output / f"{stem}-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(stem, stats, expected)
    for row in manifest.get("hint", []):
        line = (f"结果：{stem} 引导（{hint_label(row)}）解出 {row['solved_tasks']}/{row['tasks']} 题"
                f"（通过样本 {row['passing_samples']}/{row['slots']}）")
        if row["kind"] == "prefix":
            line += f"、照抄了开头的 {row['restated']}/{row['slots']}（其中通过 {row['passing_restated']}）"
        print(line)
    if args.hint_kind != "none":
        print(f"交给训练的候选：{len(keep)} 条"
              + ("（每题只取最短的、成功过的比例）" if args.hint_kind == "prefix" else ""))
    print(f"输出目录：{output}")


if __name__ == "__main__":
    main()
