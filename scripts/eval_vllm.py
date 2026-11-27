"""经本机 vLLM 服务评测一批题：每题贪心一次，再采样 K 次，逐条判分（辅助指标）。

主协议是 evaluate.py（transformers + NF4 + 贪心）：慢，一批几十道题要半小时上下。这里改走本机的
vLLM 服务（bf16 权重），用在训练过程中看趋势、给几组适配器做初筛。两个协议的数字不能混用，所以
报告里 generation.backend 写 "vllm"、quantization 写 "bf16"。

只看贪心，对"这一轮训练有没有起作用"常常不灵敏：一道题采样 8 次对了 3 次，贪心没对就仍然算
错。所以每题另外采样 K 次，另外报告每次采样的通过率（pass@1）和至少通过一次的题占多少（pass@K）。

``--model`` 写基座名就是评基座；写启动服务时 ``--lora-modules`` 登记的名字就是评那个适配器（服务
要以 ``--enable-lora`` 启动）。适配器没在 vLLM 里生效时，评出来的其实是基座，数字看上去也没有
异常。所以评适配器时加 ``--sanity-tasks-file 某一组的 train.jsonl``：先从本次要评的题里取几道
训练过的题，适配器和同一条服务上的基座各贪心生成一次，逐题比较两边的输出。输出和基座不同的题不到
``--sanity-min`` 道（默认 1，也就是一道都不同）就以 4 退出，不往后评。训练过的题严格通过几道只作
参考、不作门槛：训练强度小时（一个 epoch、几十条数据）贪心解码复现不了训练答案是正常的，和适配器
有没有生效无关。自检每次都重做，因为它查的是服务现在的样子。拿不到基座名、或者要评的题里一道训练
过的都没有时没法比较：不作废，照常往后评，自检结果里记下"没能比较"和原因。

输出都在 ``--output-dir`` 里，``<stem>`` 是 ``--tag``（没给就是 split 名）：

- ``<label>-<stem>-records.jsonl``      每条样本一行，追加写
- ``<label>-<stem>-greedy.json``         贪心报告，每题一条（``--no-greedy`` 时没有）
- ``<label>-<stem>-sampled-k<K>.json``   采样报告，每题 K 条（``--samples 0`` 时没有）
- ``<label>-<stem>-sanity.json``         自检的逐题结果（给了 ``--sanity-tasks-file`` 时）
- ``<label>-<stem>-summary.json``        汇总

报告的结构与 evaluate.py 的输出相同（schema_version ``codeagent-eval-v2``），tier_report.py 能把它当
普通报告读。每条样本是一个槽位，由 (题, 解码方式, 第几条) 区分；记录里带着当时的整份设置，原样
重跑只补没完成的槽位（记录齐全时连服务都不用，关了服务也能把报告重新出一遍）。records 文件是追加写的，
读写都用 ``_shared_jsonl``（同 sample_rft.py）：上一次运行写到一半留下半行时（磁盘写满、配额超限、被 kill，
截在某个汉字中间也算），续跑前先去掉它，那个槽位重做；读回来的记录里缺了哪个槽位就补哪个，补不上就以 3
结束。同一个 (label, tag) 在一个目录里只能有一套设置：设置变了就换一个新的 ``--tag`` 或
``--output-dir``，否则拒绝续跑。

退出码：0 完成；3 还有槽位没完成（或者自检没能比完，下不了结论）；4 自检没过（适配器的输出和基座
完全一样）。

用法：python scripts/eval_vllm.py --tasks-file 题文件 --split train --model 服务上的模型名
        --label 标签 --output-dir 目录 [--vllm-url 地址] [--tag 段] [--samples 8] …
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agentlab.benchmark.builder import build_all
from agentlab.benchmark.protocol import (
    build_messages, load_specs, prompt_fingerprint, protocol_hash, spec_fingerprint,
)
from agentlab.benchmark.tiers import LABELS, TIERS, tier_of
from agentlab.sampling import vllm_client
from _shared_jsonl import cut_partial_tail, jsonl_line, read_lines, read_jsonl
from evaluate import SCHEMA_VERSION, content_hash, evaluate_candidate, summarize_results, training_overlap

GREEDY, SAMPLING = "greedy", "sampling"
KIND = {GREEDY: "贪心", SAMPLING: "采样"}
# 连续这么多个槽位因为连接问题失败、其间没有一次成功，就认定服务已经不可用，剩下的槽位不再发。
# 服务卡住而端口还在时，每个槽位都要把重试的几次超时等满才放弃，几百个槽位会白等几个钟头
OUTAGE_AFTER = 8
# 续跑被拒绝时给用户的出路，几处报错共用
ANOTHER = "；换一个新的 --tag 或 --output-dir"
# 自检结果里进 summary.json 的那几项（vllm_summary.py 打表二用）；逐题记录只留在 -sanity.json 里
SANITY_NUMBERS = ("pass", "total", "min", "changed", "compared", "compared_tasks", "not_compared_reason")


class SlotFailed(RuntimeError):
    """这个槽位没能完成：生成时服务出错，或者判分环境自己出了问题。不写记录，重跑时重做。"""


def generation_settings(decoding: str, args) -> dict:
    """报告里的生成设置，键与 evaluate.py 的 generation_protocol 相同。

    ``decoding`` 是 greedy 或 sampling。采样时的 top_p、top_k 记的是 ``--top-p``、``--top-k`` 的值；
    服务不接受的扩展参数实际没有发，这件事记在报告的 protocol.sampling_options 里。贪心请求不带
    seed，记录里的 seed 是协议这一层的值（与 evaluate.py 的贪心报告同一种写法）。
    """
    greedy = decoding == GREEDY
    generation = dict(repeats=1 if greedy else args.samples, seed=args.seed,
                      temperature=0.0 if greedy else args.temperature,
                      max_new_tokens=args.max_tokens, quantization="bf16", decoding=decoding,
                      backend="vllm")
    if not greedy:
        generation.update(top_p=args.top_p, top_k=args.top_k)
    return generation


def make_slots(specs: list[dict], args, options: dict, *, greedy: bool, samples: int) -> list[dict]:
    """每道题的槽位：``greedy`` 为真时贪心一条，再采样 ``samples`` 条（0 条就是不采样）。

    主评测按命令行参数给（``greedy=not args.no_greedy``、``samples=args.samples``）。自检永远是
    ``greedy=True, samples=0``：它的结论只能来自真的发了的贪心请求，不能跟着 ``--no-greedy`` 变空。

    贪心请求 temperature 0、top_p 1，不带 top_k 和 seed。采样的 seed 是 ``--seed`` 加上这条的序号，
    与 evaluate.py 的 ``seed = args.seed + repeat_index`` 一致。服务不接受的扩展参数（``options`` 是
    探测结果）不发；``request`` 记下实际发出去的采样参数。两种解码各有第 0 条，所以槽位由
    (题, 解码方式, 第几条) 区分。
    """
    sampling = dict(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
    if args.top_k and options["top_k"]:
        sampling["top_k"] = args.top_k
    slots = []
    for spec in specs:
        task_id = spec["task_id"]
        if greedy:
            slots.append(dict(task_id=task_id, decoding=GREEDY, repeat_index=0, seed=args.seed,
                              request=dict(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)))
        for repeat in range(samples):
            request = {**sampling, "seed": args.seed + repeat} if options["seed"] else dict(sampling)
            slots.append(dict(task_id=task_id, decoding=SAMPLING, repeat_index=repeat,
                              seed=args.seed + repeat, request=request))
    return slots


def slot_key(item: dict) -> tuple:
    return item.get("task_id"), item.get("decoding"), item.get("repeat_index")


def slot_label(slot: dict) -> str:
    task_id, decoding, repeat = slot_key(slot)
    return f"{task_id} [{decoding} {repeat}]"


def report_record(record: dict) -> dict:
    """进报告的记录：``setup`` 是续跑时核对用的，报告顶层已经有同样的信息。"""
    return {name: value for name, value in record.items() if name != "setup"}


def failures(unfinished: list[dict]) -> list[str]:
    """没完成的槽位，写成给用户看的一行一个（最多 5 个）。"""
    return [f"{item['slot']}: {item['error']}" for item in unfinished[:5]]


def read_ids(path: Path) -> list[str]:
    """``--ids-file`` 里的题号：每行一个 task_id，或者每行一道题（一条训练记录也行）的 JSONL。"""
    ids = []
    for line in read_lines(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            record = json.loads(line)
            task_id = record.get("task_id")
            if not task_id:
                raise ValueError(f"{path.name} 里有一行没有 task_id：{line[:60]}")
        else:
            task_id = line
        ids.append(task_id)
    return list(dict.fromkeys(ids))


def identify_model(url: str, name: str) -> tuple[str | None, str | None]:
    """``--model`` 评的是基座还是适配器：返回 (基座名, 适配器名；评基座时是 None)。

    新版本的服务把适配器也列在 /v1/models 里，条目上的 parent 指向它的基座；旧版本只列基座，
    适配器的名字不在列表里。所以名字在列表里就看它的 parent；不在列表里就是适配器，基座取列表里
    唯一的那个（列了不止一个时不知道是哪个，如实记 None）。名字写错了要等真发请求时才报错。
    """
    entries = {entry.get("id"): entry for entry in vllm_client._request(f"{url}/v1/models", None, 10.0)["data"]}
    parent = (entries.get(name) or {}).get("parent")
    if name in entries:
        return (parent, name) if parent else (name, None)
    bases = [entry_id for entry_id, entry in entries.items() if not entry.get("parent")]
    return (bases[0] if len(bases) == 1 else None), name


def check_model(url: str, model: str, timeout: float) -> None:
    """先用一个不带扩展参数的请求确认服务认这个名字，不认就抛 VLLMError。

    不能只靠 sampling_options：它把一切 4xx 都当成"这个参数不接受"，模型名写错时探测也会
    "通过"，要到后面每个槽位才逐个失败，看上去像评测没跑完。
    """
    vllm_client.chat(url, model, [{"role": "user", "content": "1"}], temperature=1, max_tokens=1, timeout=timeout)


def ask_server(url: str, model: str, timeout: float) -> tuple[str | None, str | None, dict]:
    """问服务三件事：这个名字是基座还是适配器、它认不认这个名字、接不接受 seed 和 top_k。"""
    base, adapter = identify_model(url, model)
    check_model(url, model, timeout)
    return base, adapter, vllm_client.sampling_options(url, model)


def service_line(url: str, model: str, base: str | None, adapter: str | None, options: dict) -> str:
    if adapter:
        kind = f"挂在基座 {base} 上的适配器" if base else "适配器（模型列表里分不出它挂在哪个基座上）"
    else:
        kind = "基座"
    message = f"服务 {url} 上的 {model} 是{kind}"
    rejected = [name for name, accepted in options.items() if not accepted]
    if rejected:
        message += "；服务不接受 " + "、".join(rejected) + "：采样时不发这个参数"
    return message + "。"


def connect_service(args, parser: argparse.ArgumentParser) -> tuple[str | None, str | None, dict]:
    """问服务现在的样子并打印一行；连不上、或者它不认这个名字，就报错退出（退出码 2）。"""
    try:
        found = ask_server(args.vllm_url, args.model, args.request_timeout)
    except vllm_client.VLLMError as error:
        if error.transient:
            parser.error(f"连不上 vLLM 服务：{error}")
        parser.error(f"服务不接受模型 {args.model} 的请求：{error}")
    print(service_line(args.vllm_url, args.model, *found), flush=True)
    return found


def changed(previous: dict, wanted: dict) -> str | None:
    """``wanted`` 里第一项与 ``previous`` 不同的设置，写成一句话；都一样返回 None。"""
    for name, value in wanted.items():
        if previous.get(name) != value:
            return f"{name}：记录里是 {previous.get(name)!r}，这次是 {value!r}"
    return None


def setup_of(records: list[dict]) -> dict | None:
    """第一条带着完整设置（基座、适配器、探测结果都在）的记录的设置；没有就返回 None，这时去问服务。"""
    for record in records:
        setup = record.get("setup")
        if isinstance(setup, dict) and all(name in setup for name in ("base", "adapter", "options")):
            return setup
    return None


def done_records(records: list[dict], slots: list[dict], setup: dict, name: str) -> dict[tuple, dict]:
    """records 文件里这次能直接用的记录：{槽位键: 记录}。

    记录里的设置必须与这次完全一致，文件里也不能有这次不评的槽位，否则拒绝续跑：把两次不同设置
    的样本混进同一份报告，看上去与正常结果没有区别。设置先比：设置变了是原因，多出来的槽位只是结果。
    """
    expected = {slot_key(slot) for slot in slots}
    done = {}
    for record in records:
        difference = changed(record.get("setup") or {}, setup)
        if difference:
            raise ValueError(f"{name} 里的记录与这次的设置不一致（{difference}）{ANOTHER}")
        if slot_key(record) not in expected:
            raise ValueError(f"{name} 里有这次不评的槽位（{slot_label(record)}）{ANOTHER}")
        if record.get("failure_type") != "infrastructure_error":
            done[slot_key(record)] = record
    return done


class Outage:
    """连续失败的计数。连接问题（连不上、超时、5xx）连续出现 ``OUTAGE_AFTER`` 次、其间没有一次
    成功，就认定服务已经不可用。请求本身被拒（400）说明是这个请求有问题，不计入。"""

    def __init__(self):
        self.stopped = threading.Event()
        self._failures = 0
        self._lock = threading.Lock()

    def record(self, transport_failure: bool) -> None:
        with self._lock:
            self._failures = self._failures + 1 if transport_failure else 0
            if self._failures >= OUTAGE_AFTER:
                self.stopped.set()


class Runner:
    """并发地生成并判分一批槽位：每条样本一个线程先后做这两件事，所以同时发出的请求不超过
    ``concurrency``；判分另有信号量限住，同时跑的测试进程不超过 ``grade_workers`` 组。"""

    def __init__(self, url: str, specs_by_id: dict, timeout: float, concurrency: int, grade_workers: int,
                 setup: dict):
        self.url, self.specs_by_id, self.timeout, self.setup = url, specs_by_id, timeout, setup
        self.concurrency = concurrency
        self.grading = threading.BoundedSemaphore(grade_workers)
        self.outage = Outage()

    def evaluate(self, slot: dict, task_dir: Path, grade_root: Path, model: str | None = None) -> dict:
        """生成并判分一条样本，返回能直接进报告的记录。没完成时抛 SlotFailed。

        ``model`` 是请求里写的模型名；不给就是 ``--model``。自检拿基座作对照时给基座名。
        """
        spec = self.specs_by_id[slot["task_id"]]
        if self.outage.stopped.is_set():
            raise SlotFailed("服务已经不可用，没有发")
        started = time.perf_counter()
        try:
            raw = vllm_client.chat(self.url, model or self.setup["served_model"], build_messages(spec),
                                   timeout=self.timeout, **slot["request"])
        except vllm_client.VLLMError as error:           # 记下来，不中断其它槽位
            self.outage.record(error.transient)          # 请求被拒说明服务还在，不算连接问题
            raise SlotFailed(f"{type(error).__name__}: {error}") from error
        except Exception as error:                       # 回答的格式不对等：不计入连接问题
            raise SlotFailed(f"{type(error).__name__}: {error}") from error
        self.outage.record(False)
        seconds = round(time.perf_counter() - started, 4)
        with self.grading:
            grade_dir = grade_root / f"{spec['task_id']}-{slot['decoding']}-{slot['repeat_index']}"
            grade_dir.mkdir(parents=True, exist_ok=True)    # evaluate_candidate 往这里写补丁，目录要先有
            record = evaluate_candidate(spec, raw["text"], task_dir, grade_dir, {
                "decoding": slot["decoding"], "repeat_index": slot["repeat_index"], "seed": slot["seed"],
                "request": slot["request"], "setup": self.setup, "generation_time_s": seconds,
                "input_tokens": raw.get("prompt_tokens"), "output_tokens": raw.get("completion_tokens"),
                "finish_reason": raw.get("finish_reason"),
                "hit_token_limit": raw.get("finish_reason") == "length"})
        if record["failure_type"] == "infrastructure_error":
            # 判分环境自己出了问题（沙箱建不起来等）：这不是这条样本的结论，不写记录，重跑时重做
            raise SlotFailed(record.get("error", "infrastructure_error"))
        return record

    def run(self, slots: list[dict], task_dir: Path, grade_root: Path, on_record,
            *, model: str | None = None) -> list[dict]:
        """跑完 ``slots``：每判完一条就调用一次 ``on_record(记录)``。返回没完成的槽位 [{slot, error}, …]。"""
        unfinished = []
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {pool.submit(self.evaluate, slot, task_dir, grade_root, model): slot for slot in slots}
            try:
                for future in as_completed(futures):
                    slot = futures[future]
                    try:
                        record = future.result()
                    except SlotFailed as error:
                        unfinished.append({"slot": slot_label(slot), "error": str(error)})
                    else:
                        on_record(record)
            except BaseException:
                # 中途出错或者按了 Ctrl-C：还没开始的槽位不再发。不做这一步，线程池退出前会把它们全跑完
                for future in futures:
                    future.cancel()
                raise
        return unfinished


def by_tier(specs: list[dict], results: list[dict]) -> dict:
    """每层的 [通过的题数, 题数]。"通过"指这道题至少有一条样本判分通过；贪心只有一条，就是它。"""
    solved = defaultdict(bool)
    for record in results:
        solved[record["task_id"]] |= bool(record["strict_pass"])
    counts = {tier: [0, 0] for tier in TIERS}
    for spec in specs:
        row = counts[tier_of(spec["task_id"])]
        row[1] += 1
        row[0] += solved[spec["task_id"]]
    return counts


def environment() -> dict:
    """判分这边的环境。生成在服务那边做，本机装没装 torch、transformers 不影响结果，所以不记它们。"""
    packages = {}
    for package in ("pytest", "pydantic", "PyYAML"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {"python": sys.version, "packages": packages}


def build_report(specs: list[dict], results: list[dict], generation: dict, *, served: str, base: str | None,
                 adapter: str | None, tasks_file: str, split: str, overlap: dict,
                 options: dict | None = None) -> dict:
    """一份报告：键与 evaluate.py 的 main 写出的 summary 相同，每道题 ``generation["repeats"]`` 条。"""
    benchmark_hash = protocol_hash(specs)
    protocol = dict(
        schema_version=SCHEMA_VERSION, benchmark_hash=benchmark_hash,
        prompt_hashes={spec["task_id"]: prompt_fingerprint(spec) for spec in specs},
        generation=generation, split=split,
        # 提示由服务端按模型自带的 chat template 渲染，这边没有 tokenizer 的信息可记
        tokenization={"attention_implementation": "vllm", "served_model": served})
    if options is not None:
        # seed、top_k 是扩展参数，服务不接受时不发；generation 里记的是设置，这里记实际发没发
        protocol["sampling_options"] = options
    metrics = summarize_results(results)
    return dict(
        schema_version=SCHEMA_VERSION, model=base,
        # 服务加载的是哪个 revision，这边看不到，如实留空
        requested_revision=None, resolved_revision=None,
        adapter=adapter, tasks_file=tasks_file, split=split,
        task_ids=sorted(spec["task_id"] for spec in specs),
        task_hashes={spec["task_id"]: spec_fingerprint(spec) for spec in specs},
        tasks_hash=benchmark_hash, protocol=protocol, protocol_hash=content_hash(protocol),
        contamination=overlap, selection_contamination=None,
        generation=generation, repeats=generation["repeats"], metrics=metrics,
        environment=environment(), n_tasks=metrics["n_tasks"], success_rate=metrics["pass_at_1"],
        results=[report_record(record) for record in results],
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        interpretation="diagnostic_only" if split in {"train", "all"} else "held_out")


def build_summary(label: str, served: str, split: str, tag: str, specs: list[dict], greedy: list[dict] | None,
                  sampled: list[dict] | None, samples: int, sanity: dict | None) -> dict:
    """summary.json：给 shell 脚本读的汇总。``greedy``、``sampled`` 是两份报告的结果行，没做的那部分是 None。

    ``by_tier`` 的键是层名，值是 [通过, 题数]；采样的"通过"指这道题至少有一次采样通过。
    """
    summary = dict(label=label, model=served, split=split, tag=tag, greedy=None, sampled=None, sanity=None)
    if greedy is not None:
        summary["greedy"] = {"pass": sum(bool(record["strict_pass"]) for record in greedy), "total": len(specs),
                             "by_tier": by_tier(specs, greedy)}
    if sampled is not None:
        metrics = summarize_results(sampled)
        summary["sampled"] = {
            "samples_per_task": samples, "attempts": metrics["n_attempts"],
            "passed_attempts": metrics["n_passed_attempts"], "pass_at_1": metrics["pass_at_1"],
            "any_pass_rate": metrics["any_pass_rate"], "by_tier": by_tier(specs, sampled)}
    if sanity is not None:
        summary["sanity"] = {name: sanity.get(name) for name in SANITY_NUMBERS}
    return summary


def result_line(label: str, stem: str, summary: dict) -> str:
    """最后一行的结论。没做贪心就省略贪心那半句，没采样就省略采样的那半句。"""
    parts = []
    if summary["greedy"] is not None:
        parts.append(f"贪心 {summary['greedy']['pass']}/{summary['greedy']['total']}")
    if summary["sampled"] is not None:
        sampled = summary["sampled"]
        parts.append(f"采样通过率 {sampled['pass_at_1']:.0%}（{sampled['passed_attempts']}/"
                     f"{sampled['attempts']} 次尝试）、pass@{sampled['samples_per_task']} "
                     f"{sampled['any_pass_rate']:.0%}")
    return f"结果：{label} {stem} " + "、".join(parts)


def display_width(text: str) -> int:
    return sum(2 if ord(char) > 0x2E7F else 1 for char in text)


def cell(text, size: int) -> str:
    return str(text) + " " * (size - display_width(str(text)))


def tier_table(specs: list[dict], greedy: list[dict] | None, sampled: list[dict] | None) -> str:
    """分层汇总，每层一行：题数、贪心通过的题、采样解出的题（至少通过一次）、通过的样本。

    只列这次有题的层，按 TIERS 的先后：题数为 0 的层不打印，表里不会多出一行 0/0。层清单和层名都取自 tiers.py
    （TIERS、LABELS），加了新层不用改这里；层名一栏的宽度跟着要打印的层名走，层名再长也对得齐。
    """
    passes, solved = by_tier(specs, greedy or []), by_tier(specs, sampled or [])
    attempts, passed = Counter(), Counter()
    for record in sampled or []:
        attempts[tier_of(record["task_id"])] += 1
        passed[tier_of(record["task_id"])] += bool(record["strict_pass"])
    shown = [tier for tier in TIERS if passes[tier][1]]
    width = max([10, *(display_width(LABELS[tier]) + 2 for tier in shown)])    # 最宽的层名后面留两格
    lines = [cell("层", width) + cell("题数", 6) + cell("贪心", 8) + cell("解出的题", 10) + cell("通过样本", 10)]
    for tier in shown:
        tasks = passes[tier][1]
        lines.append(
            cell(LABELS[tier], width) + cell(tasks, 6)
            + (cell(f"{passes[tier][0]}/{tasks}", 8) if greedy is not None else cell("-", 8))
            + (cell(f"{solved[tier][0]}/{tasks}", 10) if sampled is not None else cell("-", 10))
            + (cell(f"{passed[tier]}/{attempts[tier]}", 10) if sampled is not None else ""))
    return "\n".join(lines)


def bar_line(done: int, total: int, started: float, label: str) -> str:
    """进度行，与 run_rft.sh 的 progress() 同一种格式（show_progress.sh 靠它认进度）。"""
    elapsed = time.monotonic() - started
    eta = elapsed / done * (total - done) if done else 0
    filled = 24 * done // total
    return (f"[{'#' * filled}{'-' * (24 - filled)}] {done}/{total} ({done * 100 // total}%) "
            f"{label} · 已用 {elapsed / 60:.1f}m · 剩余≈{eta / 60:.1f}m")


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sanity_specs(specs: list[dict], path: Path, limit: int) -> list[dict]:
    """自检用的题：本次要评的题里出现在训练数据 ``path`` 里的，按 task_id 排序，最多 ``limit`` 道。"""
    trained = set(training_overlap(specs, path)["task_ids"])
    return sorted((spec for spec in specs if spec["task_id"] in trained),
                  key=lambda spec: spec["task_id"])[:limit]


def run_sanity(args, specs: list[dict], options: dict, runner: Runner, work: Path) -> dict:
    """自检：取几道训练过的题，适配器各贪心评一次（严格通过几道只作参考），再让基座各贪心评一次，
    逐题比较两边的输出。每次都重做，因为它查的是服务现在的状态，代价只有十来次生成。

    自检不经过 records 文件：没过的时候主评测不往下做，这时留下的记录会在修好之后被当成适配器的
    结果接着用。返回的字典原样写进 ``<label>-<stem>-sanity.json``：``pass``、``total``、``results`` 是
    适配器在训练过的题上的严格通过数、题数和逐题记录；``changed``、``compared``、``base_results`` 是
    和基座的比较，即输出和基座不同的题数、有没有比较成、基座在同样这几道题上的记录（字段同
    ``results``）。没能比较时 ``compared`` 是 False，原因写在 ``not_compared_reason``。
    """
    check = sanity_specs(specs, Path(args.sanity_tasks_file), args.sanity_tasks)
    base, adapter = runner.setup["base"], runner.setup["adapter"]
    sanity = {
        "label": args.label, "model": args.model, "adapter": adapter, "base": base,
        "split": args.split, "tag": args.tag or args.split,
        "pass": 0, "total": len(check), "min": args.sanity_min,
        "changed": 0, "changed_tasks": [], "compared": False, "compared_tasks": 0,
        "base_results": [], "base_unfinished": [],
        "source": str(Path(args.sanity_tasks_file).resolve()),
        "task_ids": [spec["task_id"] for spec in check],
        "generation": runner.setup["generations"][GREEDY],
        "results": [], "unfinished": [], "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    blocked = why_not_compared(adapter, base, len(check))
    if blocked:
        sanity["not_compared_reason"] = blocked
    if not check:
        return sanity
    print(f"自检：{len(check)} 道训练过的题，适配器各贪心评一次", flush=True)
    task_dir = work / "tasks"
    build_all(task_dir, check)

    def note(record):
        sanity["results"].append(report_record(record))
        print(f"  自检 {record['task_id']}：{record['verdict']}", flush=True)

    # 自检永远用贪心，与 --no-greedy 无关：比较的是贪心输出，结论只能来自真的发了的贪心请求
    slots = make_slots(check, args, options, greedy=True, samples=0)
    sanity["unfinished"] = runner.run(slots, task_dir, work / "grade", note)
    # 按题号排好再落盘：先完成哪个不确定，文件的内容不该跟着变
    sanity["results"].sort(key=lambda record: record["task_id"])
    sanity["pass"] = sum(bool(record["strict_pass"]) for record in sanity["results"])
    if blocked:
        return sanity

    print(f"自检：基座 {base} 在这几道题上也各贪心评一次，逐题比较两边的输出", flush=True)
    mine = {record["task_id"]: record["raw_response"].strip() for record in sanity["results"]}
    base_records = []

    def note_base(record):
        # 适配器那边先跑完，这里分得清是"输出一样"、"输出不同"，还是适配器那一条没生成出来
        other = mine.get(record["task_id"])
        if other is None:
            relation = "比不了（适配器那一条没生成出来）"
        else:
            relation = "输出一样" if other == record["raw_response"].strip() else "输出不同"
        base_records.append(record)
        print(f"  自检 {record['task_id']}：基座 {record['verdict']}，{relation}", flush=True)

    # 基座的判分另放一个目录：同一道题两边各判一次，不共用对方留下的文件。没完成的槽位前面写上"基座"，
    # 报错时和适配器那边的分得开
    sanity["base_unfinished"] = [{**item, "slot": f"基座 {item['slot']}"} for item in
                                  runner.run(slots, task_dir, work / "grade-base", note_base, model=base)]
    concluded = len(base_records) + len(sanity["base_unfinished"])
    if concluded != len(check):
        # 和适配器那边一样：每道选中的题必须要么有基座的结论、要么记为没完成，少了一条就不能据此下结论
        raise RuntimeError(f"基座在选中的 {len(check)} 道题上只有 {concluded} 道有下文，不能据此下结论")
    sanity["base_results"] = sorted((report_record(record) for record in base_records),
                                    key=lambda record: record["task_id"])
    compare_with_base(sanity)
    return sanity


def why_not_compared(adapter: str | None, base: str | None, tasks: int) -> str | None:
    """没法拿基座作比较的原因；能比较时返回 None。

    评的是基座（没有适配器）、服务的模型列表里分不出适配器挂在哪个基座上（或者把基座报成了适配器
    自己）、要评的题里没有训练过的题：这几种情况下一道题都比不了，自检只能统计严格通过几道。
    """
    if adapter is None:
        return "这次评的是基座，没有适配器可比"
    if not base:
        return f"服务的模型列表里分不出 {adapter} 挂在哪个基座上，拿不到基座名"
    if base == adapter:
        return f"服务把基座报成了 {base}，基座名就是适配器名，没有另一个基座可以比"
    if not tasks:
        return "本次要评的题里没有出现在 --sanity-tasks-file 里的，没有可比较的题"
    return None


def compare_with_base(sanity: dict) -> None:
    """逐题比较适配器和基座的输出（去掉首尾空白后），记进 sanity：

    - ``compared_tasks``、``changed_tasks``、``changed``：两边都生成出来的题有几道、其中输出不同的
      是哪几道、有几道；
    - ``compared``：有没有比较成。一道都比不了（有一边全没生成出来）时是 False，原因写进
      ``not_compared_reason``。
    """
    mine = {record["task_id"]: record["raw_response"] for record in sanity["results"]}
    theirs = {record["task_id"]: record["raw_response"] for record in sanity["base_results"]}
    compared = [task for task in sanity["task_ids"] if task in mine and task in theirs]
    sanity["compared_tasks"] = len(compared)
    sanity["changed_tasks"] = [task for task in compared if mine[task].strip() != theirs[task].strip()]
    sanity["changed"] = len(sanity["changed_tasks"])
    sanity["compared"] = bool(compared)
    if not compared:
        sanity["not_compared_reason"] = "适配器和基座没有一道题都生成出来的（见 unfinished 和 base_unfinished）"


def stop_for_sanity(args, sanity: dict) -> None:
    """自检的结论：打印两行，决定往不往下评。

    严格通过几道只作参考，不作门槛。作数的是和基座的比较：输出不同的题不到 ``--sanity-min`` 道就以 4
    退出；还有题没能比上、下不了结论的以 3 退出，原样重跑会重做自检。没能比较（没有可比的基座、
    没有训练过的题）时不作废：接着往后评，由汇总注明这一组没能确认适配器是否生效。
    """
    concluded = len(sanity["results"]) + len(sanity["unfinished"])
    if concluded != sanity["total"]:
        # 每道选中的题必须要么有结论、要么记为没完成。少了一条，"严格通过 x/N"和比较的结论就都对不上
        raise RuntimeError(f"自检选了 {sanity['total']} 道题，只有 {concluded} 道有下文，不能据此下结论")
    if sanity["total"]:
        print(f"自检：训练过的题严格通过 {sanity['pass']}/{sanity['total']}"
              f"（仅供参考；训练强度小时复现不了是正常的）", flush=True)
    if not sanity["compared"]:
        print(f"自检：没能和基座比较（{sanity['not_compared_reason']}），没能确认适配器是否生效；"
              f"不作废，接着往后评", flush=True)
        return
    outstanding = len(sanity["unfinished"]) + len(sanity["base_unfinished"])
    note = f"；另有 {outstanding} 道没能比上，没算进来" if outstanding else ""
    print(f"自检：适配器的输出在 {sanity['compared_tasks']} 道题里有 {sanity['changed']} 道和基座不同"
          f"（门槛 {args.sanity_min}{note}）", flush=True)
    if sanity["changed"] >= args.sanity_min:
        return
    if outstanding:
        # 没能比上的那几道里只要有不同的，结论就可能反过来：不能说它没生效，也不能就这么放过它
        print(f"自检有 {outstanding} 道题没能和基座比上，不能据此下结论，原样重跑会重做自检："
              f"{failures([*sanity['unfinished'], *sanity['base_unfinished']])}", file=sys.stderr)
        raise SystemExit(3)
    if sanity["changed"]:
        print(f"适配器的输出只有 {sanity['changed']} 道题和基座不同，不到 --sanity-min {args.sanity_min}："
              f"按这个门槛算作没有生效")
    else:
        print("适配器在这几道题上的输出和基座完全一样，适配器在 vLLM 里没有生效")
    raise SystemExit(4)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks-file", required=True, help="冻结的任务文件（带 split）")
    parser.add_argument("--split", required=True, choices=["train", "dev", "test", "all"])
    parser.add_argument("--vllm-url", default=vllm_client.DEFAULT_URL, help="本机 vLLM 服务的地址")
    parser.add_argument("--model", required=True,
                        help="服务上的模型名：写基座名评基座，写 --lora-modules 登记的名字评那个适配器")
    parser.add_argument("--label", required=True, help="输出文件名的开头，例如 base、A-verified")
    parser.add_argument("--tag", default="", help="输出文件名里的一段；不给时用 split 名")
    parser.add_argument("--ids-file", help="只评这些题：每行一个 task_id，或每行一道题的 JSONL；"
                                           "只用其中出现在这个 split 里的题")
    parser.add_argument("--samples", type=int, default=8, help="每题采样次数；0 表示只做贪心")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20,
                        help="采样时的 top-k，0 表示不做 top-k 截断；服务不接受这个参数时不发")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-greedy", action="store_true", help="不做贪心评测，只采样")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--concurrency", type=int, default=24, help="同时发出的请求数")
    parser.add_argument("--grade-workers", type=int, default=4, help="同时判分的进程数")
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--training-data", help="训练数据（某一组的 train.jsonl）：查它与此批评测题的重合")
    parser.add_argument("--sanity-tasks-file",
                        help="某一组的训练数据 train.jsonl：先拿其中本次要评的题比较适配器和基座的输出，"
                             "检验适配器在 vLLM 里是否生效")
    parser.add_argument("--sanity-tasks", type=int, default=6, help="自检最多用几道题")
    parser.add_argument("--sanity-min", type=int, default=1, help="适配器的输出至少要在几道题上和基座不同")
    return parser


def check_arguments(args, parser: argparse.ArgumentParser) -> None:
    if args.samples < 0 or args.concurrency < 1 or args.grade_workers < 1:
        parser.error("samples 不能是负数；concurrency / grade-workers 必须是正数")
    if args.max_tokens < 1 or args.request_timeout <= 0:
        parser.error("max-tokens / request-timeout 必须是正数")
    if args.top_k < 0:
        parser.error("top-k 不能是负数（0 表示不做 top-k 截断）")
    if args.samples and (args.temperature <= 0 or not 0 < args.top_p <= 1):
        parser.error("采样需要正的 --temperature，top-p 要在 (0, 1] 里")
    if not args.samples and args.no_greedy:
        parser.error("--samples 0 加 --no-greedy 就没有可评的了")
    if args.sanity_tasks < 0 or args.sanity_min < 0 or args.sanity_min > args.sanity_tasks:
        parser.error("sanity-tasks / sanity-min 不能是负数，sanity-min 也不能大于 sanity-tasks")
    for name in ("label", "tag"):
        if getattr(args, name) and not re.fullmatch(r"[A-Za-z0-9_-]+", getattr(args, name)):
            parser.error(f"--{name} 只能由字母、数字、'-' 和 '_' 组成")


def load_tasks(args, parser: argparse.ArgumentParser) -> tuple[list[dict], str]:
    """读任务文件，按 ``--ids-file`` 筛。返回 (题, 给用户看的一句说明)。"""
    if not Path(args.tasks_file).exists():
        parser.error(f"缺少任务文件 {args.tasks_file}")
    try:
        specs = load_specs(args.tasks_file, split=args.split)
        listed = set(read_ids(Path(args.ids_file))) if args.ids_file else set()
    except (OSError, ValueError) as error:
        parser.error(str(error))
    note = ""
    if args.ids_file:
        specs = [spec for spec in specs if spec["task_id"] in listed]
        note = f"；--ids-file 指定 {len(listed)} 道，其中 {len(specs)} 道在 {args.split} 里"
    if not specs:
        parser.error(f"{args.split} 里没有可评的题{note}")
    return specs, note


def resolve_settings(args, parser: argparse.ArgumentParser, records_path: Path, records: list[dict],
                     settings: dict) -> tuple[dict, bool]:
    """这一次的整份设置。``settings`` 是不依赖服务的部分；基座、适配器、探测结果来自旧记录或者服务。

    返回 (设置, 是不是沿用的旧记录)。旧记录与这次的设置对不上就拒绝；对得上就先不连服务，
    等真要发请求时再确认服务还是记录里的那个样子。
    """
    stored = setup_of(records)
    if stored is None:
        base, adapter, options = connect_service(args, parser)
        return {**settings, "base": base, "adapter": adapter, "options": options}, False
    difference = changed(stored, settings)
    if difference:
        parser.error(f"{records_path.name} 里的记录与这次的设置不一致（{difference}）{ANOTHER}")
    return {**settings, "base": stored["base"], "adapter": stored["adapter"], "options": stored["options"]}, True


def write_report(path: Path, specs: list[dict], results: list[dict], generation: dict, args, setup: dict,
                 overlap: dict, options: dict | None = None) -> None:
    write_json(path, build_report(specs, results, generation, served=args.model, base=setup["base"],
                                  adapter=setup["adapter"], tasks_file=args.tasks_file, split=args.split,
                                  overlap=overlap, options=options))


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    check_arguments(args, parser)
    specs, note = load_tasks(args, parser)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = args.tag or args.split
    prefix = f"{args.label}-{stem}"
    records_path = output / f"{prefix}-records.jsonl"
    generations = {decoding: generation_settings(decoding, args) for decoding in (GREEDY, SAMPLING)}
    settings = {"served_model": args.model, "split": args.split, "tasks_hash": protocol_hash(specs),
                "greedy": not args.no_greedy, "samples": args.samples, "generations": generations}
    # 上一次运行写到一半被打断，文件末尾会留下半行。不去掉，这次追加的记录会接在它后面粘成一条坏行，那个槽位
    # 这一回白做。截在某个汉字中间也没关系：读的时候坏字节换成替换字符，那一行因解析不了被跳过
    lost = cut_partial_tail(records_path)
    if lost:
        print(f"{records_path.name} 的最后一行没写完（{lost} 个字节）：去掉它，这个槽位重做", file=sys.stderr)
    records = read_jsonl(records_path)
    setup, resumed = resolve_settings(args, parser, records_path, records, settings)
    options = setup["options"]
    slots = make_slots(specs, args, options, greedy=not args.no_greedy, samples=args.samples)
    try:
        # train 和 all 本来就含训练过的题，查重合没有意义；其余 split 给了 --training-data 才查
        overlap = training_overlap(specs, None if args.split in {"train", "all"} else args.training_data)
        done = done_records(records, slots, setup, records_path.name)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if overlap["status"] == "contaminated":
        print(f"警告：这批题与训练数据重合（题 {overlap['task_ids']}，来源组 {overlap['source_groups']}），"
              "结果只能当训练内的诊断看", file=sys.stderr)
    pending = [slot for slot in slots if slot_key(slot) not in done]
    each = " + ".join(part for part, count in (("贪心 1", not args.no_greedy), (f"采样 {args.samples}", args.samples))
                      if count)
    print(f"评测 {args.label} {stem}：{len(specs)} 道题（每题 {each}）= {len(slots)} 个槽位，"
          f"已有 {len(slots) - len(pending)} 个，待完成 {len(pending)} 个{note}", flush=True)
    if resumed and (pending or args.sanity_tasks_file):
        # 用的是旧记录，又要向服务发请求：先确认服务还是记录里的那个样子
        for name, now in zip(("base", "adapter", "options"), connect_service(args, parser)):
            if now != setup[name]:
                parser.error(f"服务与记录里的设置不一致（{name}：记录里是 {setup[name]!r}，现在是 {now!r}）"
                             f"{ANOTHER}")
    elif resumed:
        print("记录齐全，不用连服务，直接出报告", flush=True)

    specs_by_id = {spec["task_id"]: spec for spec in specs}
    sanity = None
    with tempfile.TemporaryDirectory(prefix="codeagent-eval-vllm-") as directory:
        work = Path(directory)
        runner = Runner(args.vllm_url, specs_by_id, args.request_timeout, args.concurrency,
                        args.grade_workers, setup)
        if args.sanity_tasks_file:
            try:
                sanity = run_sanity(args, specs, options, runner, work / "sanity")
            except (OSError, ValueError) as error:
                parser.error(str(error))
            write_json(output / f"{prefix}-sanity.json", sanity)
            stop_for_sanity(args, sanity)

        lock = threading.Lock()
        count = len(slots) - len(pending)
        started = time.monotonic()

        def save(record):
            nonlocal count
            with lock:
                with records_path.open("a", encoding="utf-8") as handle:
                    handle.write(jsonl_line(record) + "\n")
                count += 1
                print(bar_line(count, len(slots), started,
                               f"{args.label} {stem} {KIND[record['decoding']]} {record['task_id']} "
                               f"[{record['repeat_index']}] {record['verdict']}"), flush=True)

        if pending:
            task_dir = work / "tasks"
            build_all(task_dir, [specs_by_id[task_id] for task_id in sorted({slot["task_id"] for slot in pending})])
        unfinished = runner.run(pending, work / "tasks", work / "grade", save)
    if unfinished:
        print(f"有 {len(unfinished)} 个槽位没能完成，原样重跑会只补这些：{failures(unfinished)}", file=sys.stderr)
        if runner.outage.stopped.is_set():
            print(f"连续 {OUTAGE_AFTER} 个槽位因为连接问题失败：服务可能已经退出或者卡住，"
                  "剩下的槽位没有再发。恢复服务后原样重跑", file=sys.stderr)
        raise SystemExit(3)

    # 以落盘的记录为准。槽位都报了完成，却还有读不出记录的（那一行没写完，或者被判分环境错误
    # 作废）：同样算没完成，不出报告，原样重跑只补这些
    finished = done_records(read_jsonl(records_path), slots, setup, records_path.name)
    missing = [slot_label(slot) for slot in slots if slot_key(slot) not in finished]
    if missing:
        print(f"有 {len(missing)} 条记录写进文件后读不出来，原样重跑会重做：{missing[:5]}", file=sys.stderr)
        raise SystemExit(3)
    greedy = sampled = None
    if not args.no_greedy:
        greedy = [finished[(spec["task_id"], GREEDY, 0)] for spec in specs]
        write_report(output / f"{prefix}-greedy.json", specs, greedy, generations[GREEDY], args, setup, overlap)
    if args.samples:
        sampled = [finished[(spec["task_id"], SAMPLING, repeat)]
                   for spec in specs for repeat in range(args.samples)]
        write_report(output / f"{prefix}-sampled-k{args.samples}.json", specs, sampled, generations[SAMPLING],
                     args, setup, overlap, options)
    summary = build_summary(args.label, args.model, args.split, stem, specs, greedy, sampled, args.samples, sanity)
    write_json(output / f"{prefix}-summary.json", summary)
    print(f"── 分层（{args.label} {stem}）──")
    print(tier_table(specs, greedy, sampled))
    print(result_line(args.label, stem, summary))


if __name__ == "__main__":
    main()
