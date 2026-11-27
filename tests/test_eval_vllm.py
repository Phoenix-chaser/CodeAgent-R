# -*- coding: utf-8 -*-
"""经本机 vLLM 服务的评测脚本（辅助指标）：服务是本机上的假服务，判分是真的（沙箱里跑 pytest）。"""
import hashlib
import json
import re
import socket
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_vllm
import tier_report
from agentlab.benchmark.builder import build_all
from agentlab.benchmark.tiers import TIERS
from agentlab.sampling import vllm_client
from compare import task_attempts, validate_report_protocol
from test_rft_sampling import FakeVLLM, lines

TRAIN = ("h3_demo_001", "h3v_demo_001", "demo_001")       # 难题、变体、原有题各一道
DEV = ("h3_dev_001", "demo_dev_001")
# evaluate.py 的 main 写进报告的键：这里的报告要与之同构
REPORT_KEYS = {"schema_version", "model", "requested_revision", "resolved_revision", "adapter", "tasks_file",
               "split", "task_ids", "task_hashes", "tasks_hash", "protocol", "protocol_hash", "contamination",
               "selection_contamination", "generation", "repeats", "metrics", "environment", "n_tasks",
               "success_rate", "results", "timestamp", "interpretation"}
GREEDY_GENERATION = {"repeats": 1, "seed": 42, "temperature": 0.0, "max_new_tokens": 2048, "quantization": "bf16",
                     "decoding": "greedy", "backend": "vllm"}
SAMPLING_GENERATION = {"repeats": 2, "seed": 42, "temperature": 0.7, "max_new_tokens": 2048, "quantization": "bf16",
                       "decoding": "sampling", "backend": "vllm", "top_p": 0.95, "top_k": 20}
NOT_CHECKED = {"status": "not_checked", "source": None, "task_ids": [], "source_groups": [], "prompt_hashes": []}
BY_TIER = {"default": [0, 1], "hard": [1, 1], "variant": [0, 1]}       # 原有三层各一道题
# tiers.py 以后加的层在这批题里没有题：summary 的 by_tier 里是 [0, 0]（打印分层表时才不列）
EMPTY_TIERS = {tier: [0, 0] for tier in TIERS if tier not in BY_TIER}
# json.dumps(ensure_ascii=False) 原样写出、str.splitlines() 却当成换行的三个字符
SEPARATORS = "第一行\u2028第二行\u0085第三行\u2029末行"
REFERENCE = "（仅供参考；训练强度小时复现不了是正常的）"     # "训练过的题严格通过 x/N"只作参考
IDENTICAL = "适配器在这几道题上的输出和基座完全一样，适配器在 vLLM 里没有生效\n"     # 自检没过（退出码 4）的最后一句


def tiny_task(task_id, split):
    """一道独立的小题：clamp 没做截断，公开测试过、隐藏测试挂。"""
    module = f"{task_id}.py"
    return {
        "task_id": task_id, "source_group": f"group-{task_id}", "split": split,
        "bug_family": "clamping", "difficulty": "L1",
        "instruction": "实现 clamp(values, low, high)：把每个数限制在 [low, high] 内，返回新列表。",
        "module": module,
        "buggy": "def clamp(values, low, high):\n    return list(values)\n",
        "fixed": "def clamp(values, low, high):\n"
                 "    return [min(max(v, low), high) for v in values]\n",
        "public_test": f"from {module[:-3]} import clamp\n\n"
                       "def test_public():\n    assert clamp([1, 5], 0, 10) == [1, 5]\n",
        "private_tests": {"test_hidden.py": f"from {module[:-3]} import clamp\n\n"
                          "def test_hidden():\n    assert clamp([-3, 3, 13], 0, 10) == [0, 3, 10]\n"},
    }


def write_tasks(path):
    """训练三道（三层各一）、dev 两道（难题、原有题）。"""
    specs = [tiny_task(name, "dev" if name in DEV else "train") for name in (*TRAIN, *DEV)]
    path.write_text("".join(json.dumps(spec, ensure_ascii=False) + "\n" for spec in specs), encoding="utf-8")
    return specs


def completion(text, finish="stop"):
    return {"choices": [{"message": {"content": text}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2}}


def model_answers(specs):
    """假模型。贪心时只有难题答对；采样时难题的种子是偶数才对，变体题的种子是 4 的倍数才对，原有题
    总答错（其中一条写到一半被截断）。探测参数用的请求（只生成 1 个 token）认不出题目，随便答。"""
    by_module = {spec["module"]: spec for spec in specs}

    def answer(payload):
        user = payload["messages"][-1]["content"]
        spec = next((item for module, item in by_module.items() if f"文件 {module}：" in user), None)
        if spec is None:
            return 200, completion("")
        name, seed = spec["task_id"], payload.get("seed", 0)
        if payload["temperature"] == 0:
            passes = name.startswith("h3_")
        elif name.startswith("h3v_"):
            passes = seed % 4 == 0
        else:
            passes = name.startswith("h3_") and seed % 2 == 0
        if name == "demo_001" and seed == 43:
            return 200, completion(spec["buggy"], finish="length")
        return 200, completion(spec["fixed"] if passes else spec["buggy"])

    return answer


def adapter_answers(specs):
    """假适配器：训练过的三道题都答对，dev 的原有题照旧答错。"""
    by_module = {spec["module"]: spec for spec in specs}

    def answer(payload):
        user = payload["messages"][-1]["content"]
        spec = next((item for module, item in by_module.items() if f"文件 {module}：" in user), None)
        if spec is None:
            return 200, completion("")
        return 200, completion(spec["fixed"] if spec["task_id"] in TRAIN else spec["buggy"])

    return answer


def by_model(answers):
    """按请求里的 model 分派的假服务：answers 是 {模型名: answer(payload)}，没登记的模型名一律 404。
    适配器和基座对同一道题答得一样还是不一样，由这里决定。"""
    def answer(payload):
        handler = answers.get(payload["model"])
        if handler is None:
            return 404, {"error": {"message": f"The model `{payload['model']}` does not exist."}}
        return handler(payload)

    return answer


class FakeLoRAVLLM(FakeVLLM):
    """服务上除了基座还登记了一个适配器：/v1/models 里适配器那条的 parent 指向基座。
    listing 换成别的，就能演出"模型列表里分不出基座"的服务。"""

    def __init__(self, answer, base="org/model", adapter="lora-a", listing=None):
        super().__init__(answer)
        listing = listing if listing is not None else [{"id": base, "parent": None}, {"id": adapter, "parent": base}]
        base_handler = self.server.RequestHandlerClass

        class Handler(base_handler):
            def do_GET(self):
                self._reply(200, {"data": listing})

        self.server.RequestHandlerClass = Handler


def trained_file(path, *names):
    """某一组的训练数据 train.jsonl：每行一条训练样本，这里只用到 task_id。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"task_id": name, "source_group": "unrelated",
                                        "messages": [{"role": "user", "content": "提示"}]}) + "\n"
                            for name in names), encoding="utf-8")
    return path


def closed_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def run_eval(fake, tmp_path, *extra, output=None, split="train", label="base", url=None):
    command = [sys.executable, str(ROOT / "scripts/eval_vllm.py"), "--tasks-file", str(tmp_path / "tasks.jsonl"),
               "--split", split, "--vllm-url", url or fake.url, "--model", "org/model", "--label", label,
               "--output-dir", str(output or tmp_path / "eval"), "--concurrency", "4", "--grade-workers", "2", *extra]
    return subprocess.run(command, capture_output=True, text=True, timeout=300)


def names(output):
    return sorted(path.name for path in output.iterdir())


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def generated(requests):
    """假服务收到的生成请求（探测参数用的请求只生成 1 个 token，不算）。"""
    return [request for request in requests if request["max_tokens"] == 2048]


def parsed(*extra):
    return eval_vllm.make_parser().parse_args(
        ["--tasks-file", "tasks.jsonl", "--split", "train", "--model", "org/model", "--label", "base",
         "--output-dir", "out", *extra])


# ── 纯逻辑 ──

def test_slots_and_generation_settings_follow_the_sampling_protocol():
    specs = [tiny_task(TRAIN[0], "train")]
    options = {"seed": True, "top_k": True}
    greedy = {"task_id": "h3_demo_001", "decoding": "greedy", "repeat_index": 0, "seed": 42,
              "request": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 2048}}

    def sampled(repeat):
        return {"task_id": "h3_demo_001", "decoding": "sampling", "repeat_index": repeat, "seed": 42 + repeat,
                "request": {"temperature": 0.7, "top_p": 0.95, "max_tokens": 2048, "top_k": 20, "seed": 42 + repeat}}

    assert eval_vllm.make_slots(specs, parsed("--samples", "2"), options, greedy=True, samples=2) == [
        greedy, sampled(0), sampled(1)]
    assert eval_vllm.generation_settings("greedy", parsed()) == GREEDY_GENERATION
    assert eval_vllm.generation_settings("sampling", parsed("--samples", "2")) == SAMPLING_GENERATION
    assert eval_vllm.generation_settings("sampling", parsed("--samples", "4", "--top-p", "0.9", "--top-k", "0")) == {
        "repeats": 4, "seed": 42, "temperature": 0.7, "max_new_tokens": 2048, "quantization": "bf16",
        "decoding": "sampling", "backend": "vllm", "top_p": 0.9, "top_k": 0}

    # 服务不接受 seed 和 top_k 时不发；记录里的 seed 仍是协议里的值
    refused = eval_vllm.make_slots(specs, parsed(), {"seed": False, "top_k": False}, greedy=True, samples=1)
    assert [slot["request"] for slot in refused] == [
        {"temperature": 0.0, "top_p": 1.0, "max_tokens": 2048},
        {"temperature": 0.7, "top_p": 0.95, "max_tokens": 2048}]
    assert [slot["seed"] for slot in refused] == [42, 42]
    # 做不做贪心、采样几条由调用方明说；--top-k 0 不做 top-k 截断
    only = eval_vllm.make_slots(specs, parsed(), options, greedy=False, samples=8)
    assert len(only) == 8 and {slot["decoding"] for slot in only} == {"sampling"}
    off = eval_vllm.make_slots(specs, parsed("--top-k", "0"), options, greedy=True, samples=3)
    assert [slot["seed"] for slot in off] == [42, 42, 43, 44] and all("top_k" not in s["request"] for s in off)
    assert eval_vllm.make_slots(specs, parsed(), options, greedy=False, samples=0) == []
    # 自检用的槽位：每道题一条贪心
    assert eval_vllm.make_slots(specs, parsed("--no-greedy", "--samples", "4"), options,
                                greedy=True, samples=0) == [greedy]


def test_tiers_summary_line_and_progress_bar():
    specs = [tiny_task(name, "train") for name in (*TRAIN, *DEV)]
    results = [dict(task_id="h3_demo_001", strict_pass=False), dict(task_id="h3_demo_001", strict_pass=True),
               dict(task_id="demo_dev_001", strict_pass=True)]
    # "通过"指至少有一条样本通过；没出结果的题也计入题数（合成两层在这批题里没有题：[0, 0]，同下面的 by_tier）
    assert eval_vllm.by_tier(specs, results) == {"default": [1, 2], "hard": [1, 2], "variant": [0, 1]} | EMPTY_TIERS
    assert eval_vllm.result_line("A", "dev", {"greedy": {"pass": 2, "total": 5}, "sampled": None}) == \
        "结果：A dev 贪心 2/5"
    sampled = {"pass_at_1": 0.25, "passed_attempts": 3, "attempts": 12, "samples_per_task": 4, "any_pass_rate": 2 / 3}
    assert eval_vllm.result_line("A", "dev", {"greedy": None, "sampled": sampled}) == \
        "结果：A dev 采样通过率 25%（3/12 次尝试）、pass@4 67%"
    line = eval_vllm.bar_line(6, 12, time.monotonic() - 60, "等待 vLLM 加载")
    assert line == "[############------------] 6/12 (50%) 等待 vLLM 加载 · 已用 1.0m · 剩余≈1.0m"
    assert re.search(r"\] [0-9]+/[0-9]+ \([0-9]+%\)", line)       # show_progress.sh 认进度用的样子


def test_ids_file_takes_task_ids_or_task_lines(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_text(f" demo_001 \n{json.dumps(tiny_task('h3_dev_001', 'dev'))}\n\n demo_001 \nghost\n",
                    encoding="utf-8")
    assert eval_vllm.read_ids(path) == ["demo_001", "h3_dev_001", "ghost"]
    path.write_text('{"source_group": "x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="没有 task_id"):
        eval_vllm.read_ids(path)


def test_records_and_id_files_are_split_on_newlines_only(tmp_path, capsys):
    whole = {"raw_response": SEPARATORS + "\n", "note": "中文保留原样"}
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(whole, ensure_ascii=False) + "\n" + '{"task_id": "t", "decod', encoding="utf-8")
    assert eval_vllm.read_jsonl(path) == [whole]                # 原样写下的 U+2028 不是换行，记录读得回来
    assert "跳过没写完的一行" in capsys.readouterr().err
    # 现在写出的那一行把这三个字符转义掉：用 splitlines() 读的工具也不会把一条记录切成几条
    line = eval_vllm.jsonl_line(whole)
    assert not set("\u2028\u0085\u2029") & set(line) and "中文保留原样" in line
    assert line.splitlines() == [line] and json.loads(line) == whole
    tricky = {"code": "反斜杠\\\u2028之后" + "\\\u2029" * 2}          # 转义出来的 \u2028 前面正好有个反斜杠
    assert json.loads(eval_vllm.jsonl_line(tricky)) == tricky
    ids = tmp_path / "ids.jsonl"
    spec = tiny_task("h3_dev_001", "dev")
    spec["instruction"] = "说明\u2028第二行"
    ids.write_text(json.dumps(spec, ensure_ascii=False) + "\r\n", encoding="utf-8")
    assert eval_vllm.read_ids(ids) == ["h3_dev_001"]


def test_done_records_skip_broken_lines_and_grading_environment_errors(tmp_path, capsys):
    setup = {"served_model": "org/model"}
    slot = {"task_id": "t", "decoding": "greedy", "repeat_index": 0}
    good = {**slot, "setup": setup}
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps({**good, "failure_type": "infrastructure_error"}) + "\n"
                    + '{"task_id": "t", "decod\n' + json.dumps(good) + "\n", encoding="utf-8")
    rows = eval_vllm.read_jsonl(path)
    assert "跳过没写完的一行" in capsys.readouterr().err
    assert eval_vllm.done_records(rows, [slot], setup, path.name) == {("t", "greedy", 0): good}
    # 设置不一致、多出槽位：都拒绝，并说清楚差在哪
    path.write_text(json.dumps({**good, "setup": {"served_model": "other/model"}}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="设置不一致（served_model：记录里是 'other/model'，这次是 'org/model'）"):
        eval_vllm.done_records(eval_vllm.read_jsonl(path), [slot], setup, path.name)
    path.write_text(json.dumps({"task_id": "t", "decoding": "sampling", "repeat_index": 0, "setup": setup}) + "\n",
                    encoding="utf-8")
    with pytest.raises(ValueError, match="不评的槽位（t \\[sampling 0\\]）"):
        eval_vllm.done_records(eval_vllm.read_jsonl(path), [slot], setup, path.name)
    # 记录里带着完整设置（基座、适配器、探测结果）时，续跑沿用它，不用再问服务
    complete = {"base": "org/model", "adapter": None, "options": {"seed": True, "top_k": True}}
    assert eval_vllm.setup_of(rows + [{**good, "setup": complete}]) == complete
    assert eval_vllm.setup_of(rows) is None and eval_vllm.setup_of([]) is None


def test_the_served_model_is_told_apart_from_its_adapters():
    fake = FakeVLLM()
    try:
        assert eval_vllm.identify_model(fake.url, "org/model") == ("org/model", None)
        # 旧版本的服务不列适配器：名字不在列表里就是适配器，基座是列表里唯一的那个
        assert eval_vllm.identify_model(fake.url, "lora-a") == ("org/model", "lora-a")
    finally:
        fake.close()
    fake = FakeLoRAVLLM(None)
    try:
        assert eval_vllm.identify_model(fake.url, "org/model") == ("org/model", None)
        assert eval_vllm.identify_model(fake.url, "lora-a") == ("org/model", "lora-a")
    finally:
        fake.close()


def test_outage_counter_only_counts_consecutive_transport_failures():
    outage = eval_vllm.Outage()
    for _ in range(eval_vllm.OUTAGE_AFTER - 1):
        outage.record(True)
    outage.record(False)                                       # 中间成功一次就重新数
    for _ in range(eval_vllm.OUTAGE_AFTER - 1):
        outage.record(True)
    assert not outage.stopped.is_set()
    outage.record(True)
    assert outage.stopped.is_set()


def test_sanity_verdict_needs_an_outcome_for_every_selected_task(capsys):
    """选了 3 道题却只有 2 道有下文：不能拿"严格通过 0/3"和"0 道不同"去冤枉适配器，也不能就这么放过它。"""
    silent = {"pass": 0, "total": 3, "results": [], "unfinished": []}
    with pytest.raises(RuntimeError, match="选了 3 道题，只有 0 道有下文"):
        eval_vllm.stop_for_sanity(parsed(), silent)
    assert capsys.readouterr().out == ""                                   # 数目对不上就什么结论都不打印


def test_the_sanity_verdict_hinges_on_how_many_outputs_differ_from_the_base(capsys):
    """自检的结论只看"输出和基座不同的题够不够门槛"（默认 1）。严格通过几道只作参考：0/3 也照样往下评。"""
    assert parsed().sanity_min == 1
    assert "适配器的输出至少要在几道题上和基座不同" in eval_vllm.make_parser().format_help()

    def conclude(changed, *, minimum=1, compared=3, passed=0, unfinished=0, base_unfinished=0, total=3,
                reason="拿不到基座名"):
        """跑一遍结论。返回 (退出码, stdout, stderr)；退出码是 None 表示接着往后评。"""
        sanity = {"pass": passed, "total": total, "min": minimum, "changed": changed, "compared_tasks": compared,
                  "compared": compared > 0, "not_compared_reason": reason,
                  "results": [{"task_id": f"t{index}", "raw_response": "x"} for index in range(total - unfinished)],
                  "unfinished": [{"slot": f"t{index} [greedy 0]", "error": "VLLMError: 拒了"}
                                 for index in range(unfinished)],
                  "base_unfinished": [{"slot": f"基座 t{index} [greedy 0]", "error": "VLLMError: 拒了"}
                                      for index in range(base_unfinished)]}
        capsys.readouterr()
        code = None                                   # 没有以退出码结束就是接着往后评
        try:
            eval_vllm.stop_for_sanity(parsed("--sanity-min", str(minimum)), sanity)
        except SystemExit as stopped:
            code = stopped.code
        shown = capsys.readouterr()
        return code, shown.out, shown.err

    reference = "自检：训练过的题严格通过 0/3" + REFERENCE + "\n"
    # 门槛是默认的 1：有一道题的输出和基座不同就算生效，训练过的题一道都没严格通过也往下评
    assert conclude(3) == (None, reference + "自检：适配器的输出在 3 道题里有 3 道和基座不同（门槛 1）\n", "")
    assert conclude(1)[0] is None
    assert conclude(1, passed=3)[1].startswith("自检：训练过的题严格通过 3/3" + REFERENCE)
    # 一道都不同：以 4 结束
    assert conclude(0) == (4, reference + "自检：适配器的输出在 3 道题里有 0 道和基座不同（门槛 1）\n" + IDENTICAL, "")
    # 门槛调到 2 而只有 1 道不同：同样以 4 结束，但说清楚是不到门槛，不是"完全一样"
    code, out, _ = conclude(1, minimum=2)
    assert code == 4 and out.endswith(
        "自检：适配器的输出在 3 道题里有 1 道和基座不同（门槛 2）\n"
        "适配器的输出只有 1 道题和基座不同，不到 --sanity-min 2：按这个门槛算作没有生效\n")
    # 有题没能比上（适配器或者基座那一条没生成出来），不同的题又不到门槛：下不了结论，以 3 结束
    code, out, err = conclude(0, compared=2, unfinished=1, base_unfinished=1)
    assert code == 3 and out == reference + (
        "自检：适配器的输出在 2 道题里有 0 道和基座不同（门槛 1；另有 2 道没能比上，没算进来）\n")
    assert "自检有 2 道题没能和基座比上，不能据此下结论，原样重跑会重做自检" in err
    assert "t0 [greedy 0]: VLLMError" in err and "基座 t0 [greedy 0]: VLLMError" in err     # 两边的分得开
    # 不同的题已经够门槛：没比上的那几道改变不了结论，照常往下评
    assert conclude(2, minimum=2, compared=2, unfinished=1)[0] is None
    # 没能比较（拿不到基座名、没有训练过的题）：不是适配器没生效，不作废
    unchecked = "自检：没能和基座比较（拿不到基座名），没能确认适配器是否生效；不作废，接着往后评\n"
    assert conclude(0, compared=0) == (None, reference + unchecked, "")
    assert conclude(0, compared=0, total=0, reason="没有可比较的题")[1] == \
        "自检：没能和基座比较（没有可比较的题），没能确认适配器是否生效；不作废，接着往后评\n"


def test_what_blocks_the_comparison_and_how_outputs_are_compared():
    """哪些情况下没法拿基座作比较，以及"输出不同"怎么算（去掉首尾空白后逐题比）。"""
    assert eval_vllm.why_not_compared("lora-a", "org/model", 3) is None
    assert eval_vllm.why_not_compared(None, "org/model", 3) == "这次评的是基座，没有适配器可比"
    assert eval_vllm.why_not_compared("lora-a", None, 3) == "服务的模型列表里分不出 lora-a 挂在哪个基座上，拿不到基座名"
    assert eval_vllm.why_not_compared("lora-a", "lora-a", 3) == (
        "服务把基座报成了 lora-a，基座名就是适配器名，没有另一个基座可以比")
    assert eval_vllm.why_not_compared("lora-a", "org/model", 0) == (
        "本次要评的题里没有出现在 --sanity-tasks-file 里的，没有可比较的题")
    sanity = {"task_ids": ["a", "b", "c"],
              "results": [{"task_id": "b", "raw_response": " 修好了()\n"}, {"task_id": "a", "raw_response": "一样()"}],
              "base_results": [{"task_id": "a", "raw_response": "一样()\n"},
                               {"task_id": "b", "raw_response": "没修()"}]}
    eval_vllm.compare_with_base(sanity)
    assert (sanity["compared"], sanity["compared_tasks"], sanity["changed"],
            sanity["changed_tasks"]) == (True, 2, 1, ["b"])
    # 基座一道都没生成出来：没能比较，写明原因
    sanity["base_results"] = []
    eval_vllm.compare_with_base(sanity)
    assert sanity["compared"] is False and "没有一道题都生成出来" in sanity["not_compared_reason"]


@pytest.mark.parametrize("extra, reason", [
    (["--samples", "-1"], "samples 不能是负数"),
    (["--samples", "0", "--no-greedy"], "没有可评的了"),
    (["--samples", "2", "--temperature", "0"], "采样需要正的 --temperature"),
    (["--samples", "2", "--top-p", "0"], "top-p 要在"),
    (["--top-k", "-1"], "top-k 不能是负数"),
    (["--concurrency", "0"], "concurrency / grade-workers 必须是正数"),
    (["--max-tokens", "0"], "max-tokens / request-timeout 必须是正数"),
    (["--sanity-min", "7"], "sanity-min 也不能大于 sanity-tasks"),
    (["--tag", "a/b"], "--tag 只能由字母"),
    (["--label", "a b"], "--label 只能由字母"),
])
def test_impossible_arguments_are_refused(extra, reason, capsys):
    with pytest.raises(SystemExit) as stopped:
        eval_vllm.check_arguments(parsed(*extra), eval_vllm.make_parser())
    assert stopped.value.code == 2 and reason in capsys.readouterr().err


# ── 端到端：假服务 + 真判分 ──

def test_reports_match_the_main_protocol_and_a_finished_run_sends_no_requests(tmp_path, dead_proxy):
    """环境里设着指向空端口的代理变量，发给本机服务的请求不能被它截走。"""
    specs = write_tasks(tmp_path / "tasks.jsonl")
    fake = FakeVLLM(model_answers(specs))
    try:
        first = run_eval(fake, tmp_path, "--samples", "2")
        assert first.returncode == 0, first.stderr[-3000:]
        sent = len(fake.requests)
        again = run_eval(fake, tmp_path, "--samples", "2")
    finally:
        fake.close()
    assert again.returncode == 0, again.stderr[-3000:]
    assert "结果：base train 贪心 1/3、采样通过率 17%（1/6 次尝试）、pass@2 33%" in first.stdout
    assert [line.split() for line in first.stdout.splitlines() if line.startswith(("原有题", "难题", "变体"))] == [
        ["原有题", "1", "0/1", "0/1", "0/2"], ["难题", "1", "1/1", "1/1", "1/2"], ["变体", "1", "0/1", "0/1", "0/2"]]

    # 先确认服务认这个模型名（不带扩展参数），再探测一次参数（seed 和 top_k 各试一次），然后生成 9 条
    requests = fake.requests[:sent]
    assert len(requests) == 2 + 9 and [request["max_tokens"] for request in requests[:2]] == [1, 1]
    assert {key: requests[0].get(key) for key in ("seed", "top_k")} == {"seed": None, "top_k": None}
    assert {key: requests[1][key] for key in ("seed", "top_k")} == {"seed": 1, "top_k": 20}
    # 贪心不带 top_k 和 seed；采样带着参数值，种子是 --seed 加序号
    generations = generated(requests)
    greedy = [request for request in generations if request["temperature"] == 0]
    assert len(greedy) == 3 and all("seed" not in r and "top_k" not in r and r["top_p"] == 1 for r in greedy)
    sampled = [request for request in generations if request["temperature"] == 0.7]
    assert sorted(request["seed"] for request in sampled) == [42, 42, 42, 43, 43, 43]
    assert all(request["top_k"] == 20 and request["top_p"] == 0.95 for request in sampled)
    assert all(request["chat_template_kwargs"] == {"enable_thinking": False} for request in generations)
    # 续跑：什么都不缺，就不再生成，也不连服务
    assert "已有 9 个，待完成 0 个" in again.stdout and "记录齐全，不用连服务，直接出报告" in again.stdout
    assert fake.requests[sent:] == []

    out = tmp_path / "eval"
    assert names(out) == ["base-train-greedy.json", "base-train-records.jsonl", "base-train-sampled-k2.json",
                          "base-train-summary.json"]
    greedy_report = read(out / "base-train-greedy.json")
    sampled_report = read(out / "base-train-sampled-k2.json")
    assert set(greedy_report) == REPORT_KEYS == set(sampled_report)

    assert greedy_report["generation"] == GREEDY_GENERATION and greedy_report["repeats"] == 1
    assert greedy_report["protocol"]["generation"] == greedy_report["generation"]
    assert greedy_report["metrics"]["pass_at_1"] == pytest.approx(1 / 3)
    assert greedy_report["metrics"]["any_pass_rate"] == pytest.approx(1 / 3)
    assert (greedy_report["metrics"]["n_attempts"], greedy_report["metrics"]["n_passed_attempts"]) == (3, 1)
    assert greedy_report["metrics"]["valid"] and greedy_report["metrics"]["failure_counts"] == {
        "private_test_failure": 2}
    assert greedy_report["adapter"] is None and greedy_report["model"] == "org/model"
    assert greedy_report["protocol"]["tokenization"] == {"attention_implementation": "vllm",
                                                         "served_model": "org/model"}
    assert greedy_report["interpretation"] == "diagnostic_only" and greedy_report["contamination"] == NOT_CHECKED
    assert "sampling_options" not in greedy_report["protocol"]

    assert sampled_report["generation"] == SAMPLING_GENERATION and sampled_report["repeats"] == 2
    assert sampled_report["metrics"]["n_attempts"] == 6 and sampled_report["metrics"]["n_passed_attempts"] == 1
    assert sampled_report["metrics"]["pass_at_1"] == pytest.approx(1 / 6)
    assert sampled_report["metrics"]["any_pass_rate"] == pytest.approx(1 / 3)
    assert sampled_report["metrics"]["task_pass_rates"] == {"h3_demo_001": 0.5, "h3v_demo_001": 0.0, "demo_001": 0.0}
    assert sampled_report["protocol"]["sampling_options"] == {"seed": True, "top_k": True}
    for report in (greedy_report, sampled_report):
        validate_report_protocol(report, task_attempts(report))         # compare.py 的协议校验也过得去
        assert report["task_ids"] == sorted(TRAIN) and report["split"] == "train"

    # 判分是真的：通过的样本隐藏测试也过；没通过的样本公开测试过、隐藏测试挂
    assert [(row["task_id"], row["repeat_index"], row["seed"], row["strict_pass"])
            for row in sampled_report["results"]] == [
        ("h3_demo_001", 0, 42, True), ("h3_demo_001", 1, 43, False),
        ("h3v_demo_001", 0, 42, False), ("h3v_demo_001", 1, 43, False),
        ("demo_001", 0, 42, False), ("demo_001", 1, 43, False)]
    passing = sampled_report["results"][0]
    assert passing["generated_code"] == specs[0]["fixed"] and passing["verdict"] == "strict_pass"
    assert passing["grader_details"]["layers"]["functional"]["private_pass"]
    failing = greedy_report["results"][2]
    assert failing["generated_code"] == specs[2]["buggy"] and failing["failure_type"] == "private_test_failure"
    layers = failing["grader_details"]["layers"]["functional"]
    assert layers["public_pass"] and not layers["private_pass"]
    assert sampled_report["results"][5]["finish_reason"] == "length"                 # 写到一半被截断
    assert sampled_report["results"][5]["hit_token_limit"] is True
    assert [row["request"] for row in greedy_report["results"]] == [
        {"temperature": 0.0, "top_p": 1.0, "max_tokens": 2048}] * 3
    assert all("setup" not in row for row in greedy_report["results"] + sampled_report["results"])

    records = lines(out / "base-train-records.jsonl")
    assert len(records) == 9 and all(record["setup"]["served_model"] == "org/model" for record in records)
    by_key = {(record["task_id"], record["decoding"], record["repeat_index"]): record for record in records}
    assert eval_vllm.report_record(by_key[("h3_demo_001", "sampling", 0)]) == passing

    assert read(out / "base-train-summary.json") == {
        "label": "base", "model": "org/model", "split": "train", "tag": "train",
        "greedy": {"pass": 1, "total": 3, "by_tier": BY_TIER | EMPTY_TIERS},
        "sampled": {"samples_per_task": 2, "attempts": 6, "passed_attempts": 1, "pass_at_1": pytest.approx(1 / 6),
                    "any_pass_rate": pytest.approx(1 / 3), "by_tier": BY_TIER | EMPTY_TIERS},
        "sanity": None}


def test_greedy_report_reads_as_an_ordinary_report_in_tier_report(tmp_path):
    specs = write_tasks(tmp_path / "tasks.jsonl")
    fake = FakeVLLM(model_answers(specs))
    try:
        result = run_eval(fake, tmp_path, "--samples", "0")
    finally:
        fake.close()
    assert result.returncode == 0, result.stderr[-3000:]
    report_path = tmp_path / "eval/base-train-greedy.json"
    assert tier_report.load_rows([f"base={report_path}"]) == [("base", read(report_path))]
    assert tier_report.tier_counts(read(report_path)) == BY_TIER | EMPTY_TIERS
    shown = subprocess.run([sys.executable, str(ROOT / "scripts/tier_report.py"), f"base={report_path}"],
                           capture_output=True, text=True, timeout=60)
    assert shown.returncode == 0, shown.stderr[-2000:]
    assert [line.split() for line in shown.stdout.splitlines()] == [
        ["模型", "原有题", "难题", "变体", "合计"], ["base", "0/1", "1/1", "0/1", "1/3"]]


def test_ids_file_selects_tasks_and_tag_names_the_files(tmp_path):
    specs = write_tasks(tmp_path / "tasks.jsonl")
    wanted = tmp_path / "wanted.txt"
    wanted.write_text(f"demo_001\n{json.dumps(specs[0])}\n\nghost_001\n", encoding="utf-8")   # 题号与整道题混着
    fake = FakeVLLM(model_answers(specs))
    try:
        result = run_eval(fake, tmp_path, "--samples", "0", "--ids-file", str(wanted), "--tag", "picked")
        wanted.write_text("ghost_001\n", encoding="utf-8")
        empty = run_eval(fake, tmp_path, "--ids-file", str(wanted), "--tag", "picked")
    finally:
        fake.close()
    assert result.returncode == 0, result.stderr[-3000:]
    assert "--ids-file 指定 3 道，其中 2 道在 train 里" in result.stdout
    assert "评测 base picked：2 道题（每题 贪心 1）= 2 个槽位" in result.stdout
    out = tmp_path / "eval"
    assert names(out) == ["base-picked-greedy.json", "base-picked-records.jsonl", "base-picked-summary.json"]
    report = read(out / "base-picked-greedy.json")
    assert report["task_ids"] == ["demo_001", "h3_demo_001"]                     # 不在这个 split 里的不理
    assert [row["task_id"] for row in report["results"]] == ["h3_demo_001", "demo_001"]     # 按任务文件里的顺序
    summary = read(out / "base-picked-summary.json")
    assert summary["tag"] == "picked" and summary["greedy"] == {
        "pass": 1, "total": 2, "by_tier": {"default": [0, 1], "hard": [1, 1], "variant": [0, 0]} | EMPTY_TIERS}
    assert "结果：base picked 贪心 1/2" in result.stdout
    asked = [request["messages"][-1]["content"] for request in generated(fake.requests)]
    assert len(asked) == 2
    assert {spec["module"] for spec in specs if any(f"文件 {spec['module']}：" in user for user in asked)} == {
        "h3_demo_001.py", "demo_001.py"}
    # 一道都对不上就没有可评的题
    assert empty.returncode == 2 and "train 里没有可评的题" in empty.stderr
    assert "--ids-file 指定 1 道，其中 0 道在 train 里" in empty.stderr


def test_samples_zero_keeps_only_the_greedy_report_and_no_greedy_only_the_sampled_one(tmp_path):
    specs = write_tasks(tmp_path / "tasks.jsonl")
    fake = FakeVLLM(model_answers(specs))
    try:
        greedy_only = run_eval(fake, tmp_path, "--samples", "0")
        sampled_only = run_eval(fake, tmp_path, "--no-greedy", "--samples", "2", "--tag", "so")
    finally:
        fake.close()
    assert greedy_only.returncode == 0, greedy_only.stderr[-3000:]
    assert sampled_only.returncode == 0, sampled_only.stderr[-3000:]
    out = tmp_path / "eval"
    assert names(out) == ["base-so-records.jsonl", "base-so-sampled-k2.json", "base-so-summary.json",
                          "base-train-greedy.json", "base-train-records.jsonl", "base-train-summary.json"]
    greedy_summary = read(out / "base-train-summary.json")
    assert greedy_summary["sampled"] is None and greedy_summary["tag"] == "train"
    assert len(lines(out / "base-train-records.jsonl")) == 3
    assert "结果：base train 贪心 1/3" in greedy_only.stdout and "采样通过率" not in greedy_only.stdout
    sampled_summary = read(out / "base-so-summary.json")
    assert sampled_summary["greedy"] is None and sampled_summary["tag"] == "so"
    assert sampled_summary["sampled"]["attempts"] == 6
    assert sampled_summary["sampled"]["by_tier"] == BY_TIER | EMPTY_TIERS
    assert len(lines(out / "base-so-records.jsonl")) == 6
    assert sampled_only.stdout.endswith("结果：base so 采样通过率 17%（1/6 次尝试）、pass@2 33%\n")


def test_training_data_is_written_into_contamination_for_held_out_splits_only(tmp_path):
    specs = write_tasks(tmp_path / "tasks.jsonl")
    training = trained_file(tmp_path / "sft/train.jsonl", "demo_dev_001", TRAIN[2])
    fake = FakeVLLM(model_answers(specs))
    try:
        audited = run_eval(fake, tmp_path, "--samples", "0", "--split", "dev", "--tag", "audit",
                           "--training-data", str(training))
        bare = run_eval(fake, tmp_path, "--samples", "0", "--split", "dev", "--tag", "bare")
        trained = run_eval(fake, tmp_path, "--samples", "0", "--training-data", str(training))
        everything = run_eval(fake, tmp_path, "--samples", "0", "--split", "all", "--tag", "everything",
                              "--training-data", str(training))
    finally:
        fake.close()
    assert [run.returncode for run in (audited, bare, trained, everything)] == [0, 0, 0, 0]
    audited_report = read(tmp_path / "eval/base-audit-greedy.json")
    assert audited_report["contamination"] == {
        "status": "contaminated", "source": str(training.resolve()),
        "sha256": hashlib.sha256(training.read_bytes()).hexdigest(),
        "task_ids": ["demo_dev_001"], "source_groups": [], "prompt_hashes": []}
    assert audited_report["interpretation"] == "held_out" and audited_report["task_ids"] == sorted(DEV)
    assert "警告：这批题与训练数据重合（题 ['demo_dev_001']，来源组 []）" in audited.stderr
    assert "警告" not in bare.stderr + trained.stderr
    # 没给 --training-data 就不查；train 和 all 里本来就有训练过的题，给了也不查
    bare_report = read(tmp_path / "eval/base-bare-greedy.json")
    assert bare_report["contamination"] == NOT_CHECKED and bare_report["interpretation"] == "held_out"
    assert read(tmp_path / "eval/base-train-greedy.json")["contamination"] == NOT_CHECKED
    whole = read(tmp_path / "eval/base-everything-greedy.json")
    assert whole["split"] == "all" and whole["task_ids"] == sorted(TRAIN + DEV)
    assert whole["contamination"] == NOT_CHECKED and whole["interpretation"] == "diagnostic_only"
    assert "结果：base everything 贪心 2/5" in everything.stdout


def test_the_sanity_check_passes_fails_or_stays_inconclusive(tmp_path):
    """自检的几种结局。适配器生效时训练过的题都答对；"weak" 时改了几行但没改对（训练强度小的样子：一道都
    不严格通过，输出却和基座不同）；"alike" 时和基座答得一模一样（没有生效）。自检的题按 task_id 取前 2 道：
    demo_001（基座答错）和 h3_demo_001（基座贪心本来就对）。
      没能比完以 3 结束，原样重跑重做自检；输出和基座不同的题不到门槛以 4 结束，不往后评；够了门槛就往下评，
      哪怕训练过的题一道都没严格通过；要评的题里没有训练过的题时没法比较，不算数、照常评。"""
    specs = write_tasks(tmp_path / "tasks.jsonl")
    trained = trained_file(tmp_path / "sft-a/train.jsonl", "h3v_demo_001", "demo_001", "h3_demo_001", "ghost_001")
    state = {"broken": False, "alike": False, "weak": False}
    healthy = model_answers(specs)

    def answer(payload):
        user = payload["messages"][-1]["content"]
        spec = next((item for item in specs if f"文件 {item['module']}：" in user), None)
        # 出故障时 demo_001 的贪心请求被拒（适配器、基座两边都拒）；探测参数的请求认不出题目，不受影响
        if state["broken"] and payload["temperature"] == 0 and f"文件 {specs[2]['module']}：" in user:
            return 400, {"error": {"message": "simulated rejection"}}
        mine = payload["model"] == "lora-a" and not state["alike"]
        if mine and state["weak"] and spec is not None and payload["max_tokens"] == 2048 and spec["task_id"] in TRAIN:
            return 200, completion(spec["buggy"] + "\n# 改了一行，但没改对\n")
        if mine and spec is not None and payload["max_tokens"] == 2048 and spec["task_id"] in TRAIN:
            return 200, completion(spec["fixed"])
        return healthy(payload)

    fake = FakeLoRAVLLM(answer)
    common = ["--samples", "0", "--model", "lora-a", "--sanity-tasks-file", str(trained), "--sanity-tasks", "2",
              "--tag", "check"]
    out = tmp_path / "eval"
    waves, snapshots = {}, {}

    def attempt(name, *extra):
        """跑一次，记下这一回新生成的请求（模型名、温度）和跑完后输出目录里的文件。"""
        sent = len(fake.requests)
        result = run_eval(fake, tmp_path, *extra, output=out)
        waves[name] = [(request["model"], request["temperature"]) for request in generated(fake.requests[sent:])]
        snapshots[name] = {path.name: path.read_bytes() for path in out.iterdir()}
        return result

    try:
        state["broken"] = True
        inconclusive = attempt("inconclusive", *common, "--sanity-min", "2")
        state["broken"] = False
        failed = attempt("failed", *common, "--sanity-min", "2")
        state["alike"] = True
        identical = attempt("identical", *common)
        state.update(alike=False, weak=True)
        weak = attempt("weak", *common)
        again = attempt("again", *common)
        state["weak"] = False
        nothing = attempt("nothing", "--samples", "0", "--model", "lora-a", "--tag", "none",
                          "--sanity-tasks-file", str(trained_file(tmp_path / "sft-b/train.jsonl", "ghost_001")))
    finally:
        fake.close()
    # 自检只用贪心，适配器 2 条、基座 2 条；没过或没评完时主评测一条都不发；记录齐全的那一回只重做了自检
    adapter_pair, base_pair = [("lora-a", 0)] * 2, [("org/model", 0)] * 2
    assert waves == {"inconclusive": adapter_pair + base_pair, "failed": adapter_pair + base_pair,
                     "identical": adapter_pair + base_pair, "weak": adapter_pair + base_pair + [("lora-a", 0)] * 3,
                     "again": adapter_pair + base_pair, "nothing": [("lora-a", 0)] * 3}
    reference = lambda passed: f"自检：训练过的题严格通过 {passed}/2{REFERENCE}\n"      # noqa: E731

    # 没能比完：3。往下不做，也不写主评测的记录
    assert inconclusive.returncode == 3 and "自检有 2 道题没能和基座比上" in inconclusive.stderr
    assert "demo_001 [greedy 0]: VLLMError" in inconclusive.stderr
    assert "基座 demo_001 [greedy 0]: VLLMError" in inconclusive.stderr
    assert inconclusive.stdout.endswith(reference(1) + "自检：适配器的输出在 1 道题里有 0 道和基座不同"
                                                 "（门槛 2；另有 2 道没能比上，没算进来）\n")
    assert "没有生效" not in inconclusive.stdout and list(snapshots["inconclusive"]) == ["base-check-sanity.json"]
    inconclusive_sanity = json.loads(snapshots["inconclusive"]["base-check-sanity.json"])
    assert (len(inconclusive_sanity["unfinished"]), len(inconclusive_sanity["base_unfinished"]),
            inconclusive_sanity["compared"], inconclusive_sanity["compared_tasks"],
            inconclusive_sanity["changed"]) == (1, 1, True, 1, 0)
    # 输出不同的题不到门槛（门槛 2，只有 1 道）：4。不往后评，也不写贪心报告
    assert failed.returncode == 4
    assert failed.stdout.endswith(reference(2) + "自检：适配器的输出在 2 道题里有 1 道和基座不同（门槛 2）\n"
                                    "适配器的输出只有 1 道题和基座不同，不到 --sanity-min 2：按这个门槛算作没有生效\n")
    assert list(snapshots["failed"]) == ["base-check-sanity.json"]
    sanity = json.loads(snapshots["failed"]["base-check-sanity.json"])
    assert {key: sanity[key] for key in ("label", "model", "adapter", "base", "split", "pass", "total", "min", "source",
                                         "task_ids", "generation", "unfinished", "base_unfinished", "changed",
                                         "changed_tasks",
                                         "compared", "compared_tasks")} == {
        "label": "base", "model": "lora-a", "adapter": "lora-a", "base": "org/model", "split": "train",
        "pass": 2, "total": 2, "min": 2, "source": str(trained.resolve()),
        "task_ids": ["demo_001", "h3_demo_001"],                            # 按 task_id 排序取前 2 道
        "generation": GREEDY_GENERATION, "unfinished": [], "base_unfinished": [],
        "changed": 1, "changed_tasks": ["demo_001"], "compared": True, "compared_tasks": 2}
    assert "not_compared_reason" not in sanity
    assert [(row["task_id"], row["verdict"], row["decoding"], row["strict_pass"]) for row in sanity["results"]] == [
        ("demo_001", "strict_pass", "greedy", True), ("h3_demo_001", "strict_pass", "greedy", True)]
    assert [(row["task_id"], row["verdict"], row["raw_response"]) for row in sanity["base_results"]] == [
        ("demo_001", "partial", specs[2]["buggy"]), ("h3_demo_001", "strict_pass", specs[0]["fixed"])]
    assert set(sanity["base_results"][0]) == set(sanity["results"][0])       # 字段同 results
    assert all("setup" not in row for row in sanity["results"] + sanity["base_results"])
    assert "  自检 demo_001：基座 partial，输出不同" in failed.stdout
    assert "  自检 h3_demo_001：基座 strict_pass，输出一样" in failed.stdout
    # 适配器和基座答得一模一样：4。changed 记成 0
    assert identical.returncode == 4 and json.loads(snapshots["identical"]["base-check-sanity.json"])["changed"] == 0
    assert identical.stdout.endswith(reference(1) + "自检：适配器的输出在 2 道题里有 0 道和基座不同（门槛 1）\n"
                                   + IDENTICAL)
    assert list(snapshots["identical"]) == ["base-check-sanity.json"]
    # 训练过的题一道都没严格通过、输出却和基座不同：过了，接着评（严格通过几道只作参考）
    assert weak.returncode == 0, weak.stderr[-3000:]
    assert weak.stdout.endswith("结果：base check 贪心 0/3\n")                  # 往下评了
    assert reference(0) + "自检：适配器的输出在 2 道题里有 2 道和基座不同（门槛 1）\n" in weak.stdout
    assert "── 分层（base check）──" in weak.stdout
    assert sorted(snapshots["weak"]) == ["base-check-greedy.json", "base-check-records.jsonl",
                                         "base-check-sanity.json", "base-check-summary.json"]
    weak_sanity = json.loads(snapshots["weak"]["base-check-sanity.json"])
    assert (weak_sanity["pass"], weak_sanity["changed"], weak_sanity["compared"]) == (0, 2, True)
    summary = json.loads(snapshots["weak"]["base-check-summary.json"])
    assert summary["sanity"] == {"pass": 0, "total": 2, "min": 1, "changed": 2, "compared": True, "compared_tasks": 2,
                                 "not_compared_reason": None} and summary["greedy"]["pass"] == 0
    # 记录齐全之后再跑一遍：自检照样重做（它查的是服务现在的样子），主评测不再生成
    assert again.returncode == 0, again.stderr[-3000:]
    assert "已有 3 个，待完成 0 个" in again.stdout and reference(0) in again.stdout
    assert snapshots["again"]["base-check-records.jsonl"] == snapshots["weak"]["base-check-records.jsonl"]
    # 本次要评的题里没有训练过的题：没法比较。不算适配器没生效，不作废，照常评
    assert nothing.returncode == 0, nothing.stderr[-3000:]
    assert "自检：没能和基座比较（本次要评的题里没有出现在 --sanity-tasks-file 里的，没有可比较的题），" \
        "没能确认适配器是否生效；不作废，接着往后评" in nothing.stdout
    assert "结果：base none 贪心 3/3" in nothing.stdout
    nothing_sanity = json.loads(snapshots["nothing"]["base-none-sanity.json"])
    assert (nothing_sanity["pass"], nothing_sanity["total"], nothing_sanity["min"]) == (0, 0, 1)
    assert (nothing_sanity["compared"], nothing_sanity["changed"], nothing_sanity["base_results"]) == (False, 0, [])
    assert nothing_sanity["task_ids"] == [] and nothing_sanity["results"] == []
    assert json.loads(snapshots["nothing"]["base-none-summary.json"])["sanity"] == {
        "pass": 0, "total": 0, "min": 1, "changed": 0, "compared": False, "compared_tasks": 0,
        "not_compared_reason": "本次要评的题里没有出现在 --sanity-tasks-file 里的，没有可比较的题"}


def test_evaluating_the_base_with_a_sanity_file_only_counts_and_never_stops(tmp_path):
    """评的就是基座：没有适配器可以和基座比。训练过的题严格通过几道照旧统计（只作参考），不比较、不设门槛，
    不因为严格通过得少就以 4 退出。"""
    specs = write_tasks(tmp_path / "tasks.jsonl")
    trained = trained_file(tmp_path / "sft/train.jsonl", *TRAIN)
    fake = FakeVLLM(model_answers(specs))
    try:
        result = run_eval(fake, tmp_path, "--samples", "0", "--sanity-tasks-file", str(trained), "--sanity-min", "3")
    finally:
        fake.close()
    assert result.returncode == 0, result.stderr[-3000:]
    assert result.stdout.endswith("结果：base train 贪心 1/3\n")              # 不设门槛：照常评完
    assert (f"自检：训练过的题严格通过 1/3{REFERENCE}\n"
            "自检：没能和基座比较（这次评的是基座，没有适配器可比），没能确认适配器是否生效；不作废，接着往后评\n"
            ) in result.stdout and "── 分层（base train）──" in result.stdout
    assert {request["model"] for request in generated(fake.requests)} == {"org/model"}
    sanity = read(tmp_path / "eval/base-train-sanity.json")
    assert (sanity["adapter"], sanity["compared"], sanity["changed"],
            sanity["pass"], sanity["total"], sanity["min"]) == (None, False, 0, 1, 3, 3)
    assert sanity["not_compared_reason"] == "这次评的是基座，没有适配器可比" and sanity["base_results"] == []
    assert (tmp_path / "eval/base-train-greedy.json").is_file()


def test_an_adapter_is_recognized_and_its_lora_name_is_recorded(tmp_path):
    specs = write_tasks(tmp_path / "tasks.jsonl")
    trained = trained_file(tmp_path / "sft/train.jsonl", *TRAIN)
    # 基座贪心时只有难题答对，适配器训练过的三道都答对：难题那道两边的输出一样，另外两道不同
    fake = FakeLoRAVLLM(by_model({"org/model": model_answers(specs), "lora-a": adapter_answers(specs)}))
    try:
        result = run_eval(fake, tmp_path, "--samples", "0", "--model", "lora-a", "--tag", "epoch-001",
                          "--sanity-tasks-file", str(trained))
    finally:
        fake.close()
    assert result.returncode == 0, result.stderr[-3000:]
    assert f"服务 {fake.url} 上的 lora-a 是挂在基座 org/model 上的适配器。" in result.stdout
    assert f"自检：训练过的题严格通过 3/3{REFERENCE}" in result.stdout
    assert "结果：base epoch-001 贪心 3/3" in result.stdout
    assert "自检：适配器的输出在 3 道题里有 2 道和基座不同（门槛 1）" in result.stdout
    out = tmp_path / "eval"
    report = read(out / "base-epoch-001-greedy.json")
    assert report["model"] == "org/model" and report["adapter"] == "lora-a"
    assert report["protocol"]["tokenization"] == {"attention_implementation": "vllm", "served_model": "lora-a"}
    # 主评测的请求里 model 就是适配器名；自检另外用基座名发了 3 条（同一条服务上的基座作对照）
    assert Counter(request["model"] for request in fake.requests) == {"lora-a": 2 + 3 + 3, "org/model": 3}
    assert read(out / "base-epoch-001-summary.json")["model"] == "lora-a"
    sanity = read(out / "base-epoch-001-sanity.json")
    assert (sanity["adapter"], sanity["base"], sanity["changed"], sanity["compared"], sanity["changed_tasks"]) == (
        "lora-a", "org/model", 2, True, ["demo_001", "h3v_demo_001"])       # 按题号排好
    assert [(row["task_id"], row["verdict"]) for row in sanity["base_results"]] == [
        ("demo_001", "partial"), ("h3_demo_001", "strict_pass"), ("h3v_demo_001", "partial")]


def test_an_adapter_whose_output_differs_from_the_base_passes_the_sanity_check(tmp_path):
    """适配器和基座的输出在几道题上不同：生效，往下评。训练过的题严格通过几道不作数——这里适配器改了几行却
    一道都没改对（只训一个 epoch 的样子），照样不作废。"""
    specs = write_tasks(tmp_path / "tasks.jsonl")
    trained = trained_file(tmp_path / "sft/train.jsonl", *TRAIN)

    def tweaked(payload):                      # 适配器：训练过的题都改了，但没改对
        user = payload["messages"][-1]["content"]
        spec = next((item for item in specs if f"文件 {item['module']}：" in user), None)
        if spec is not None and payload["max_tokens"] == 2048 and spec["task_id"] in TRAIN:
            return 200, completion(spec["buggy"] + "\n# 改了一行，但没改对\n")
        return model_answers(specs)(payload)

    fake = FakeLoRAVLLM(by_model({"org/model": model_answers(specs), "lora-a": tweaked}))
    try:
        result = run_eval(fake, tmp_path, "--samples", "0", "--model", "lora-a", "--sanity-tasks-file", str(trained))
    finally:
        fake.close()
    assert result.returncode == 0, result.stderr[-3000:]
    assert (f"自检：训练过的题严格通过 0/3{REFERENCE}\n"
            "自检：适配器的输出在 3 道题里有 3 道和基座不同（门槛 1）\n") in result.stdout
    assert "没有生效" not in result.stdout and "结果：base train 贪心 0/3" in result.stdout
    sanity = read(tmp_path / "eval/base-train-sanity.json")
    assert (sanity["pass"], sanity["total"], sanity["changed"], sanity["compared"], sanity["min"]) == (0, 3, 3, True, 1)


def test_an_adapter_whose_output_equals_the_base_ends_with_4(tmp_path):
    """适配器和基座的输出完全相同（它在 vLLM 里没有生效）：以 4 结束、不往后评，sanity.json 里 changed 是 0。"""
    specs = write_tasks(tmp_path / "tasks.jsonl")
    trained = trained_file(tmp_path / "sft/train.jsonl", *TRAIN)
    fake = FakeLoRAVLLM(model_answers(specs))       # 不看请求里的 model：适配器和基座答得一模一样
    try:
        result = run_eval(fake, tmp_path, "--samples", "0", "--model", "lora-a", "--sanity-tasks-file", str(trained))
    finally:
        fake.close()
    assert result.returncode == 4 and "Traceback" not in result.stderr
    assert result.stdout.endswith(f"自检：训练过的题严格通过 1/3{REFERENCE}\n"
                                  "自检：适配器的输出在 3 道题里有 0 道和基座不同（门槛 1）\n" + IDENTICAL)
    assert names(tmp_path / "eval") == ["base-train-sanity.json"]
    sanity = read(tmp_path / "eval/base-train-sanity.json")
    assert (sanity["changed"], sanity["compared"], sanity["changed_tasks"], sanity["pass"], sanity["total"]) == (
        0, True, [], 1, 3)
    assert [row["raw_response"] for row in sanity["base_results"]] == [row["raw_response"] for row in sanity["results"]]
    # 两边各发了三条贪心请求：适配器名、基座名各三条
    assert Counter((request["model"], request["temperature"]) for request in generated(fake.requests)) == {
        ("lora-a", 0): 3, ("org/model", 0): 3}


def test_without_a_base_name_the_sanity_check_is_not_compared_and_the_evaluation_goes_on(tmp_path):
    """拿不到基座名：服务的模型列表里列了两个基座、没有适配器那条，分不出它挂在哪一个上。没法比较就不是"输出
    完全一样"：不作废，照常往下评；sanity.json 里写明没能比较和原因。"""
    specs = write_tasks(tmp_path / "tasks.jsonl")
    trained = trained_file(tmp_path / "sft/train.jsonl", *TRAIN)
    listing = [{"id": "org/one", "parent": None}, {"id": "org/two", "parent": None}]
    fake = FakeLoRAVLLM(adapter_answers(specs), listing=listing)
    try:
        result = run_eval(fake, tmp_path, "--samples", "0", "--model", "lora-a", "--sanity-tasks-file", str(trained))
    finally:
        fake.close()
    assert result.returncode == 0, result.stderr[-3000:]
    assert f"服务 {fake.url} 上的 lora-a 是适配器（模型列表里分不出它挂在哪个基座上）。" in result.stdout
    reason = "服务的模型列表里分不出 lora-a 挂在哪个基座上，拿不到基座名"
    assert (f"自检：训练过的题严格通过 3/3{REFERENCE}\n"
            f"自检：没能和基座比较（{reason}），没能确认适配器是否生效；不作废，接着往后评\n") in result.stdout
    assert "没有生效" not in result.stdout and "结果：base train 贪心 3/3" in result.stdout
    assert {request["model"] for request in fake.requests} == {"lora-a"}      # 没有基座名，不向基座发请求
    sanity = read(tmp_path / "eval/base-train-sanity.json")
    assert (sanity["compared"], sanity["changed"], sanity["base"], sanity["base_results"],
            sanity["base_unfinished"]) == (False, 0, None, [], [])
    assert sanity["not_compared_reason"] == reason and len(sanity["results"]) == 3
    assert read(tmp_path / "eval/base-train-summary.json")["sanity"] == {
        "pass": 3, "total": 3, "min": 1, "changed": 0, "compared": False, "compared_tasks": 0,
        "not_compared_reason": reason}
    # 基座名就是适配器名：同样没有另一个基座可以比
    other = FakeLoRAVLLM(adapter_answers(specs), listing=[{"id": "lora-a", "parent": "lora-a"}])
    try:
        again = run_eval(other, tmp_path, "--samples", "0", "--model", "lora-a", "--tag", "self",
                         "--sanity-tasks-file", str(trained))
    finally:
        other.close()
    assert again.returncode == 0, again.stderr[-3000:]
    reason = "服务把基座报成了 lora-a，基座名就是适配器名，没有另一个基座可以比"
    assert f"自检：没能和基座比较（{reason}），没能确认适配器是否生效；不作废，接着往后评" in again.stdout
    assert read(tmp_path / "eval/base-self-sanity.json")["not_compared_reason"] == reason


def test_sanity_check_sends_greedy_requests_even_with_no_greedy(tmp_path):
    """自检的结论只能来自真的发了的贪心请求，不能跟着 --no-greedy 变成"一条都没评"。"""
    specs = write_tasks(tmp_path / "tasks.jsonl")
    trained = trained_file(tmp_path / "sft/train.jsonl", *TRAIN)
    common = ("--no-greedy", "--samples", "1", "--model", "lora-a", "--sanity-tasks-file", str(trained), "--tag", "ng")
    good = FakeLoRAVLLM(by_model({"org/model": model_answers(specs), "lora-a": adapter_answers(specs)}))
    bad = FakeLoRAVLLM(model_answers(specs))
    try:
        passed = run_eval(good, tmp_path, *common, output=tmp_path / "good")
        failed = run_eval(bad, tmp_path, *common, output=tmp_path / "bad")
    finally:
        good.close()
        bad.close()
    # 适配器生效：自检先发六条贪心请求（适配器三条、基座三条；自检跑完才开始主评测，所以前面的就是自检的）
    sanity_requests, main_requests = generated(good.requests)[:6], generated(good.requests)[6:]
    assert passed.returncode == 0, passed.stderr[-3000:]
    assert [request["model"] for request in sanity_requests] == ["lora-a"] * 3 + ["org/model"] * 3
    assert all(request["temperature"] == 0 and "seed" not in request for request in sanity_requests)
    assert len(main_requests) == 3 and all(request["temperature"] == 0.7 and request["model"] == "lora-a"
                                           for request in main_requests)
    assert f"自检：训练过的题严格通过 3/3{REFERENCE}" in passed.stdout
    assert "自检：适配器的输出在 3 道题里有 2 道和基座不同（门槛 1）" in passed.stdout
    assert "没有生效" not in passed.stdout
    assert "结果：base ng 采样通过率 100%（3/3 次尝试）、pass@1 100%" in passed.stdout
    out = tmp_path / "good"
    assert names(out) == ["base-ng-records.jsonl", "base-ng-sampled-k1.json", "base-ng-sanity.json",
                          "base-ng-summary.json"]                              # 只采样，没有贪心报告
    sanity = read(out / "base-ng-sanity.json")
    assert (sanity["pass"], sanity["total"], sanity["min"], sanity["changed"]) == (3, 3, 1, 2)
    assert [(row["task_id"], row["decoding"], row["request"]["temperature"]) for row in sanity["results"]] == [
        (TRAIN[2], "greedy", 0.0), (TRAIN[0], "greedy", 0.0), (TRAIN[1], "greedy", 0.0)]
    summary = read(out / "base-ng-summary.json")
    assert summary["greedy"] is None and summary["sampled"]["attempts"] == 3
    assert summary["sanity"] == {"pass": 3, "total": 3, "min": 1, "changed": 2, "compared": True, "compared_tasks": 3,
                                 "not_compared_reason": None}
    # 适配器没生效（和基座答得一模一样）：同样各发三条贪心请求，输出一道都不同不了就以 4 退出，不往后评
    assert failed.returncode == 4
    assert len(generated(bad.requests)) == 6 and all(request["temperature"] == 0 for request in generated(bad.requests))
    assert failed.stdout.endswith(f"自检：训练过的题严格通过 1/3{REFERENCE}\n"
                                  "自检：适配器的输出在 3 道题里有 0 道和基座不同（门槛 1）\n" + IDENTICAL)
    assert names(tmp_path / "bad") == ["base-ng-sanity.json"]


def test_a_rejected_request_exits_3_and_the_rerun_fills_only_that_slot(tmp_path):
    specs = write_tasks(tmp_path / "tasks.jsonl")
    healthy = model_answers(specs)
    state = {"broken": True, "no_top_k": False}

    def answer(payload):
        if state["no_top_k"] and "top_k" in payload:
            return 400, {"error": {"message": "extra fields not permitted"}}
        user = payload["messages"][-1]["content"]
        if state["broken"] and payload.get("seed") == 43 and f"文件 {specs[0]['module']}：" in user:
            return 400, {"error": {"message": "simulated rejection"}}
        return healthy(payload)

    fake = FakeVLLM(answer)
    out = tmp_path / "eval"
    marks = {}

    def attempt(name):
        sent = len(fake.requests)
        result = run_eval(fake, tmp_path, "--samples", "2", output=out)
        marks[name] = (sent, len(fake.requests))
        return result

    try:
        broken = attempt("broken")
        after_break = names(out)
        broken_lines = len(lines(out / "base-train-records.jsonl"))
        state.update(broken=False, no_top_k=True)              # 中途换了一个不接受 top_k 的服务
        moved = attempt("moved")
        state["no_top_k"] = False
        fixed = attempt("fixed")
    finally:
        fake.close()
    assert broken.returncode == 3, broken.stderr[-3000:]
    assert "有 1 个槽位没能完成，原样重跑会只补这些" in broken.stderr
    assert "h3_demo_001 [sampling 1]: VLLMError" in broken.stderr and "simulated rejection" in broken.stderr
    assert "结果：" not in broken.stdout                                   # 没凑齐就不出报告
    assert after_break == ["base-train-records.jsonl"] and broken_lines == 8
    # 旧记录是带着 top_k 采的，换了服务就不能接着往里补；那一回只确认了模型名和参数，没有生成
    start, stop = marks["moved"]
    assert moved.returncode == 2 and "换一个新的 --tag 或 --output-dir" in moved.stderr
    assert ("服务与记录里的设置不一致（options：记录里是 {'seed': True, 'top_k': True}，"
            "现在是 {'seed': True, 'top_k': False}）") in moved.stderr
    assert generated(fake.requests[start:stop]) == []
    assert fixed.returncode == 0, fixed.stderr[-3000:]
    assert "已有 8 个，待完成 1 个" in fixed.stdout
    start, stop = marks["fixed"]
    assert len(generated(fake.requests[start:stop])) == 1
    assert names(out) == ["base-train-greedy.json", "base-train-records.jsonl", "base-train-sampled-k2.json",
                          "base-train-summary.json"]
    assert len(lines(out / "base-train-records.jsonl")) == 9
    assert "结果：base train 贪心 1/3、采样通过率 17%（1/6 次尝试）、pass@2 33%" in fixed.stdout


def test_records_from_another_setup_are_refused(tmp_path):
    specs = write_tasks(tmp_path / "tasks.jsonl")
    (tmp_path / "one.txt").write_text("demo_001\n", encoding="utf-8")
    fake = FakeVLLM(model_answers(specs))
    try:
        assert run_eval(fake, tmp_path, "--samples", "2").returncode == 0
        sent = len(fake.requests)
        for extra, reason in ((["--samples", "3"], "设置不一致（samples：记录里是 2，这次是 3"),
                              (["--samples", "0"], "设置不一致（samples：记录里是 2，这次是 0"),
                              (["--no-greedy"], "设置不一致（greedy：记录里是 True，这次是 False"),
                              (["--samples", "2", "--temperature", "0.9"], "设置不一致（generations"),
                              (["--samples", "2", "--model", "other/model"], "设置不一致（served_model"),
                              (["--ids-file", str(tmp_path / "one.txt")], "设置不一致（tasks_hash")):
            again = run_eval(fake, tmp_path, *extra)
            assert again.returncode == 2 and reason in again.stderr, again.stderr[-2000:]
            assert "换一个新的 --tag 或 --output-dir" in again.stderr
    finally:
        fake.close()
    # 设置对不上就不用连服务：这六次一个请求都没发；原来的报告没有被改动
    assert fake.requests[sent:] == []
    assert names(tmp_path / "eval") == ["base-train-greedy.json", "base-train-records.jsonl",
                                        "base-train-sampled-k2.json", "base-train-summary.json"]


def test_extensions_the_server_rejects_are_not_sent_but_the_settings_stay_visible(tmp_path):
    specs = write_tasks(tmp_path / "tasks.jsonl")

    def answer(payload):
        if "top_k" in payload or "seed" in payload:
            return 400, {"error": {"message": "extra fields not permitted"}}
        return model_answers(specs)(payload)

    fake = FakeVLLM(answer)
    try:
        result = run_eval(fake, tmp_path, "--samples", "1", "--no-greedy", "--tag", "plain")
    finally:
        fake.close()
    assert result.returncode == 0, result.stderr[-3000:]
    assert "是基座；服务不接受 seed、top_k：采样时不发这个参数。" in result.stdout
    requests = generated(fake.requests)
    assert len(requests) == 3 and all("seed" not in r and "top_k" not in r for r in requests)
    report = read(tmp_path / "eval/base-plain-sampled-k1.json")
    assert report["generation"] == SAMPLING_GENERATION | {"repeats": 1}        # 设置还是参数值
    assert report["protocol"]["sampling_options"] == {"seed": False, "top_k": False}       # 实际没有发
    assert [row["request"] for row in report["results"]] == [
        {"temperature": 0.7, "top_p": 0.95, "max_tokens": 2048}] * 3
    assert [row["seed"] for row in report["results"]] == [42, 42, 42]                       # 协议里的值


def test_a_missing_server_or_an_unknown_model_stops_before_anything_is_evaluated(tmp_path):
    specs = write_tasks(tmp_path / "tasks.jsonl")

    def answer(payload):
        if payload["model"] != "org/model":
            return 404, {"error": {"message": f"The model `{payload['model']}` does not exist."}}
        return model_answers(specs)(payload)

    fake = FakeVLLM(answer)
    try:
        unknown = run_eval(fake, tmp_path, "--samples", "0", "--model", "lora-x")
        offline = run_eval(fake, tmp_path, "--samples", "0", url=f"http://127.0.0.1:{closed_port()}")
    finally:
        fake.close()
    assert unknown.returncode == 2 and "服务不接受模型 lora-x 的请求" in unknown.stderr
    assert "does not exist" in unknown.stderr and len(fake.requests) == 1
    assert offline.returncode == 2 and "连不上 vLLM 服务" in offline.stderr
    assert "Traceback" not in offline.stderr
    assert not (tmp_path / "eval/base-train-records.jsonl").exists()


def test_reports_are_rebuilt_from_complete_records_even_when_the_service_is_down(tmp_path):
    """记录齐全时不出报告的原因可以是服务已经关了：这时原样重跑不该卡在连服务上。"""
    specs = write_tasks(tmp_path / "tasks.jsonl")
    fake = FakeVLLM(model_answers(specs))
    out = tmp_path / "eval"
    try:
        assert run_eval(fake, tmp_path, "--samples", "0", output=out).returncode == 0
    finally:
        fake.close()
    reports = names(out)
    (out / "base-train-greedy.json").unlink()
    (out / "base-train-summary.json").unlink()
    offline = f"http://127.0.0.1:{closed_port()}"
    again = run_eval(fake, tmp_path, "--samples", "0", output=out, url=offline)
    assert again.returncode == 0, again.stderr[-3000:]
    assert "记录齐全，不用连服务，直接出报告" in again.stdout and "结果：base train 贪心 1/3" in again.stdout
    assert names(out) == reports


def test_runner_stops_sending_after_consecutive_transport_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(vllm_client.time, "sleep", lambda seconds: None)      # 重试之间的退避不真的等
    specs = [tiny_task(name, "train") for name in TRAIN]
    by_id = {spec["task_id"]: spec for spec in specs}
    slots = eval_vllm.make_slots(specs, parsed(), {"seed": True, "top_k": True}, greedy=True, samples=3)
    assert len(slots) == 12
    busy = FakeVLLM(lambda payload: (503, {"error": "busy"}))
    refused = FakeVLLM(lambda payload: (400, {"error": {"message": "unknown field"}}))
    records = []
    try:
        runner = eval_vllm.Runner(busy.url, by_id, 5.0, 4, 2, {"served_model": "org/model"})
        outage = runner.run(slots, tmp_path, tmp_path / "grade", records.append)
        strict = eval_vllm.Runner(refused.url, by_id, 5.0, 3, 2, {"served_model": "org/model"})
        rejected = strict.run(
            eval_vllm.make_slots(specs, parsed(), {"seed": True, "top_k": True}, greedy=True, samples=1),
            tmp_path, tmp_path / "grade", records.append)
    finally:
        busy.close()
        refused.close()
    assert not records
    # 每个发出去的槽位重试 4 次。连续 OUTAGE_AFTER 个失败之后剩下的不再发；断路之前已经有线程接走了
    # 下一个槽位，所以多发的不会超过同时并发的线程数
    sent = len(busy.requests) // vllm_client.RETRIES
    assert len(busy.requests) % vllm_client.RETRIES == 0
    assert eval_vllm.OUTAGE_AFTER <= sent <= eval_vllm.OUTAGE_AFTER + 4 - 1
    assert len(outage) == 12 and runner.outage.stopped.is_set()
    assert Counter("VLLMError" in item["error"] for item in outage) == {True: sent, False: 12 - sent}
    assert {item["error"] for item in outage if "VLLMError" not in item["error"]} == {"服务已经不可用，没有发"}
    # 请求本身被拒不说明服务不可用：不重试，也不掐断后面的槽位
    assert len(rejected) == 6 and not strict.outage.stopped.is_set() and len(refused.requests) == 6


def test_interrupting_a_run_leaves_the_slots_that_have_not_started_untouched(tmp_path):
    """按了 Ctrl-C：已经判完的记录都交了出去，还没开始的槽位不再发（原样重跑会补上）。"""
    specs = [tiny_task(name, "train") for name in TRAIN]
    build_all(tmp_path / "tasks", specs)
    slots = eval_vllm.make_slots(specs, parsed(), {"seed": True, "top_k": True}, greedy=True, samples=3)
    fake = FakeVLLM(model_answers(specs))
    kept = []

    def interrupt(record):
        kept.append(record)
        raise KeyboardInterrupt

    try:
        runner = eval_vllm.Runner(fake.url, {spec["task_id"]: spec for spec in specs}, 5.0, 2, 2,
                                  {"served_model": "org/model"})
        with pytest.raises(KeyboardInterrupt):
            runner.run(slots, tmp_path / "tasks", tmp_path / "grade", interrupt)
    finally:
        fake.close()
    assert len(kept) == 1 and kept[0]["task_id"] in TRAIN           # 先判完的那条交给了调用方
    assert len(generated(fake.requests)) < len(slots)


def test_line_separators_in_a_model_answer_do_not_split_the_record(tmp_path):
    """回答里的 U+2028、U+0085、U+2029 在 JSON 里可以原样出现，但 str.splitlines() 把它们当成换行：
    一条样本不能因此被切成几条坏行、读不回来，最后取报告时崩掉。"""
    specs = write_tasks(tmp_path / "tasks.jsonl")

    def answer(payload):
        user = payload["messages"][-1]["content"]
        spec = next((item for item in specs if f"文件 {item['module']}：" in user), None)
        if spec is None:
            return 200, completion("")
        return 200, completion(f"说明：{SEPARATORS}。\n```python\n{spec['fixed']}```\n")

    fake = FakeVLLM(answer)
    out = tmp_path / "eval"
    try:
        first = run_eval(fake, tmp_path, "--samples", "0", output=out)
        sent = len(fake.requests)
        again = run_eval(fake, tmp_path, "--samples", "0", output=out)
    finally:
        fake.close()
    assert first.returncode == 0, first.stderr[-3000:]
    assert again.returncode == 0, again.stderr[-3000:]
    assert "Traceback" not in first.stderr + again.stderr
    assert names(out) == ["base-train-greedy.json", "base-train-records.jsonl", "base-train-summary.json"]
    raw = (out / "base-train-records.jsonl").read_text(encoding="utf-8")
    assert raw.count("\n") == 3 and len(raw.splitlines()) == 3          # 一条样本一行，怎么按行读都是三行
    assert not set("\u2028\u0085\u2029") & set(raw)
    assert "结果：base train 贪心 3/3" in first.stdout
    for row, spec in zip(read(out / "base-train-greedy.json")["results"], specs[:3]):
        assert row["strict_pass"] and row["raw_response"] == f"说明：{SEPARATORS}。\n```python\n{spec['fixed']}```\n"
    assert generated(fake.requests[sent:]) == [] and "已有 3 个，待完成 0 个" in again.stdout


def test_a_record_that_cannot_be_read_back_is_redone_instead_of_crashing(tmp_path, monkeypatch, capsys):
    """槽位都跑完了、收尾时却读不出某条的记录（比如那一行没写完）：按"没完成"处理——退出码 3、
    不出报告、原样重跑只补这一条，而不是带着 Traceback 崩掉。"""
    specs = write_tasks(tmp_path / "tasks.jsonl")
    fake = FakeVLLM(model_answers(specs))
    out = tmp_path / "eval"
    monkeypatch.setattr(sys, "argv", [
        str(ROOT / "scripts/eval_vllm.py"), "--tasks-file", str(tmp_path / "tasks.jsonl"), "--split", "train",
        "--vllm-url", fake.url, "--model", "org/model", "--label", "base", "--output-dir", str(out),
        "--samples", "0", "--concurrency", "2", "--grade-workers", "2"])
    honest = eval_vllm.jsonl_line
    try:
        monkeypatch.setattr(eval_vllm, "jsonl_line",
                            lambda record: honest(record)[:40] if record["task_id"] == "demo_001" else honest(record))
        with pytest.raises(SystemExit) as stopped:
            eval_vllm.main()
        assert stopped.value.code == 3
        broken = capsys.readouterr()
        assert "有 1 条记录写进文件后读不出来" in broken.err and "Traceback" not in broken.err
        assert names(out) == ["base-train-records.jsonl"]               # 没凑齐就不出报告
        monkeypatch.setattr(eval_vllm, "jsonl_line", honest)
        eval_vllm.main()
    finally:
        fake.close()
    final = capsys.readouterr()
    assert names(out) == ["base-train-greedy.json", "base-train-records.jsonl", "base-train-summary.json"]
    assert "已有 2 个，待完成 1 个" in final.out and "Traceback" not in final.err
    assert len(generated(fake.requests)) == 4                           # 第一回三条，第二回只补一条
    assert "结果：base train 贪心 1/3" in final.out
    assert len(eval_vllm.read_jsonl(out / "base-train-records.jsonl")) == 3


def chatty_answers(specs):
    """同 model_answers，只是回答前面多一句中文说明：记录的 raw_response 里就有汉字，可以切在汉字中间。"""
    healthy = model_answers(specs)

    def answer(payload):
        status, body = healthy(payload)
        content = body["choices"][0]["message"]["content"]
        if status == 200 and content:
            body["choices"][0]["message"]["content"] = "说明：这道题要修的是截断。\n" + content
        return status, body

    return answer


def cut_last_line(path, kept):
    """上一次运行写最后一条记录时被打断：最后一行从"截"这个汉字处截断，后面（连同行尾的换行）都没写出来。
    kept 是"截"的三个字节里写出来的个数（3 就是切在两个字符之间）。返回 (留下的半行有多少字节, 这道题的题号,
    它前面各条完整记录的字节)。"""
    raw = path.read_bytes()
    start = raw.rindex(b"\n", 0, len(raw) - 1) + 1            # 最后一行的开头：它前面各条记录都是完整的
    line = raw[start:-1]
    character = line.index("截".encode("utf-8"))
    path.write_bytes(raw[:start + character + kept])
    return character + kept, json.loads(line)["task_id"], raw[:start]


@pytest.mark.parametrize("kept", [1, 2, 3], ids=["one-byte", "two-bytes", "whole-character"])
def test_a_records_file_whose_last_line_is_cut_inside_a_character_is_resumed_not_crashed(tmp_path, kept):
    """上一跑写最后一条记录时被打断（磁盘写满、配额超限、I/O 出错），最后一行截在"截"这个汉字的中间。以前整份文件
    按 UTF-8 严格解码：这里抛 UnicodeDecodeError、带着 Traceback 崩掉，一份报告都不写，原样重跑每次都停在这里。
    现在去掉没写完的那一行，只补它那个槽位；切在两个字符之间时也是一次补齐，不多做、不粘坏。"""
    specs = write_tasks(tmp_path / "tasks.jsonl")
    fake = FakeVLLM(chatty_answers(specs))
    out, records = tmp_path / "eval", tmp_path / "eval/base-train-records.jsonl"
    # 并发 1：记录按题的先后写，最后一行是确定的
    common = ["--samples", "0", "--concurrency", "1", "--grade-workers", "1"]
    try:
        first = run_eval(fake, tmp_path, *common, output=out)
        sent = len(fake.requests)
        whole = read(out / "base-train-summary.json")         # 那一跑跑完时的汇总
        partial, task, head = cut_last_line(records, kept)
        for name in names(out):                                # 那一跑没走到出报告
            if name != records.name:
                (out / name).unlink()
        assert not records.read_bytes().endswith(b"\n")
        if kept < 3:
            with pytest.raises(UnicodeDecodeError):
                records.read_text(encoding="utf-8")            # 对照：严格的 UTF-8 解码到这里确实解不开
        again = run_eval(fake, tmp_path, *common, output=out)
        stop = len(fake.requests)
        third = run_eval(fake, tmp_path, *common, output=out)
    finally:
        fake.close()
    assert first.returncode == 0, first.stderr[-3000:]
    assert again.returncode == 0, again.stdout[-3000:] + again.stderr[-3000:]
    assert "Traceback" not in again.stderr and third.returncode == 0, third.stderr[-3000:]
    assert again.stderr.strip() == (f"base-train-records.jsonl 的最后一行没写完（{partial} 个字节）："
                                    "去掉它，这个槽位重做")
    assert "已有 2 个，待完成 1 个" in again.stdout and "结果：base train 贪心 1/3" in again.stdout
    # 只补缺的那一个槽位：一条生成请求，正是被切掉的那道题
    sent_again = generated(fake.requests[sent:stop])
    assert len(sent_again) == 1
    module = next(spec["module"] for spec in specs if spec["task_id"] == task)
    assert f"文件 {module}：" in sent_again[0]["messages"][-1]["content"]
    # 记录文件是干净的三行：半行去掉了，新记录没有接在它后面粘成坏行
    raw = records.read_bytes()
    assert raw.endswith(b"\n") and raw.count(b"\n") == 3 and "\ufffd".encode("utf-8") not in raw
    assert len(lines(records)) == 3 and {record["task_id"] for record in lines(records)} == set(TRAIN)
    assert raw.startswith(head)                                     # 前两条原样没动
    assert names(out) == ["base-train-greedy.json", "base-train-records.jsonl", "base-train-summary.json"]
    # 补出来的 summary 和那一跑跑到最后时的一模一样（里面没有时间戳，可以整体比）
    assert read(out / "base-train-summary.json") == whole
    assert (whole["greedy"]["pass"], whole["greedy"]["total"]) == (1, 3)
    assert "已有 3 个，待完成 0 个" in third.stdout and generated(fake.requests[stop:]) == []


def test_the_slot_of_the_cut_tail_exits_3_while_it_cannot_be_filled_and_a_rerun_fills_it(tmp_path):
    """半行去掉以后缺的那个槽位补不上（服务拒了这条请求）：和别的没完成的槽位一样——以 3 结束、不出报告，原样重跑
    只补它；以落盘的记录为准，缺一条就不出报告。"""
    specs = write_tasks(tmp_path / "tasks.jsonl")
    healthy = chatty_answers(specs)
    state = {"broken": False}

    def answer(payload):
        user = payload["messages"][-1]["content"]
        if state["broken"] and payload["temperature"] == 0 and f"文件 {specs[2]['module']}：" in user:
            return 400, {"error": {"message": "simulated rejection"}}
        return healthy(payload)

    common = ["--samples", "0", "--concurrency", "1", "--grade-workers", "1"]
    fake = FakeVLLM(answer)
    out, records = tmp_path / "eval", tmp_path / "eval/base-train-records.jsonl"
    try:
        assert run_eval(fake, tmp_path, *common, output=out).returncode == 0
        cut_last_line(records, 2)
        for name in names(out):                                # 那一跑没走到出报告
            if name != records.name:
                (out / name).unlink()
        state["broken"] = True
        sent = len(fake.requests)
        failed = run_eval(fake, tmp_path, *common, output=out)
        after_failure, kept_records = names(out), lines(records)     # 这一跑跑完时的样子
        state["broken"] = False
        stop = len(fake.requests)
        fixed = run_eval(fake, tmp_path, *common, output=out)
    finally:
        fake.close()
    assert failed.returncode == 3, failed.stderr[-3000:]
    assert "有 1 个槽位没能完成，原样重跑会只补这些" in failed.stderr
    assert "demo_001 [greedy 0]: VLLMError" in failed.stderr        # 缺的正是被切掉那道题的
    assert "Traceback" not in failed.stderr and "结果：" not in failed.stdout
    assert after_failure == [records.name] and len(kept_records) == 2      # 没凑齐就不出报告；半行没有留下来
    assert len(generated(fake.requests[sent:stop])) == 1
    assert fixed.returncode == 0, fixed.stdout[-3000:] + fixed.stderr[-3000:]
    assert "已有 2 个，待完成 1 个" in fixed.stdout and len(generated(fake.requests[stop:])) == 1
    assert len(lines(records)) == 3 and names(out) == [
        "base-train-greedy.json", "base-train-records.jsonl", "base-train-summary.json"]
    assert "结果：base train 贪心 1/3" in fixed.stdout


# tiers.py 会多两层（合成难题、合成变体）。eval_vllm.py 从那里只导入 TIERS、LABELS、tier_of 三个名字：测试里把这三个
# 名字换成想要的那一份，就等于 tiers.py 变成了那样。这里从"原有三层"拼起，不看仓库里的 tiers.py 现在有几层
NEW_TIERS = ("synthetic_hard", "synthetic_variant")
NEW_LABELS = {"synthetic_hard": "合成难题", "synthetic_variant": "合成变体"}


def use_tiers(monkeypatch, extra=()):
    """eval_vllm.py 眼里的 tiers.py：原有三层，再加上 extra 这几层（合成题按题号前缀 s3_、s3v_ 分到这两层）。"""
    def tier_of(task_id):
        if task_id.startswith("s3v_"):
            return "synthetic_variant"
        if task_id.startswith("s3_"):
            return "synthetic_hard"
        return "hard" if task_id.startswith("h3_") else "variant" if task_id.startswith("h3v_") else "default"

    monkeypatch.setattr(eval_vllm, "TIERS", ("default", "hard", "variant", *extra))
    monkeypatch.setattr(eval_vllm, "LABELS", {"default": "原有题", "hard": "难题", "variant": "变体",
                                              **{tier: NEW_LABELS[tier] for tier in extra}})
    monkeypatch.setattr(eval_vllm, "tier_of", tier_of)


def test_the_tier_table_lists_only_tiers_that_have_tasks_whatever_tiers_py_contains(monkeypatch):
    """按"题数为 0 的层不打印"写，层的清单取自 tiers.py 的 TIERS：没有某一层的题时表里没有它那一行，也不会出现 0/0。
    只有原有三层的题时，加了新层的 TIERS 和现在的三层打印出来一字不差；有合成题时排在原有三层后面、用它自己的
    中文名，层名一栏跟着加宽。"""
    specs = [tiny_task(name, "train") for name in TRAIN]
    greedy = [dict(task_id="h3_demo_001", strict_pass=True), dict(task_id="h3v_demo_001", strict_pass=False),
              dict(task_id="demo_001", strict_pass=True)]
    sampled = [*greedy, dict(task_id="h3_demo_001", strict_pass=False), dict(task_id="h3v_demo_001", strict_pass=False)]
    use_tiers(monkeypatch)                                                # tiers.py 现在的样子：原有三层
    now = eval_vllm.tier_table(specs, greedy, sampled)
    assert [line.split() for line in now.splitlines()] == [
        ["层", "题数", "贪心", "解出的题", "通过样本"],
        ["原有题", "1", "1/1", "1/1", "1/1"], ["难题", "1", "1/1", "1/1", "1/2"], ["变体", "1", "0/1", "0/1", "0/2"]]
    # 层名一栏 10 列宽（原有三层的层名都只有 6 列）：和以前一样
    assert now.splitlines()[0] == "层" + " " * 8 + "题数  " + "贪心    " + "解出的题  " + "通过样本  "
    use_tiers(monkeypatch, NEW_TIERS)                                    # 加了两层
    assert eval_vllm.tier_table(specs, greedy, sampled) == now          # 只有原有三层的题：一字不差
    # summary 的 by_tier 照旧每层一项，没有题的层是 [0, 0]（打印时才筛掉）
    assert eval_vllm.by_tier(specs, greedy) == {"default": [1, 1], "hard": [1, 1], "variant": [0, 1],
                                                **{tier: [0, 0] for tier in NEW_TIERS}}
    # 合成题评进来了：排在原有三层后面，各列照常；层名比 10 列宽时"层"一栏跟着加宽，每行一样宽
    monkeypatch.setattr(eval_vllm, "LABELS", {**eval_vllm.LABELS, "synthetic_variant": "合成单缺陷变体"})
    more = [*specs, tiny_task("s3_demo_001", "train"), tiny_task("s3v_demo_001", "train")]
    rows = [*greedy, dict(task_id="s3_demo_001", strict_pass=True), dict(task_id="s3v_demo_001", strict_pass=False)]
    table = eval_vllm.tier_table(more, rows, [*sampled, dict(task_id="s3_demo_001", strict_pass=True),
                                              dict(task_id="s3v_demo_001", strict_pass=False)])
    assert [line.split() for line in table.splitlines()] == [
        ["层", "题数", "贪心", "解出的题", "通过样本"],
        ["原有题", "1", "1/1", "1/1", "1/1"], ["难题", "1", "1/1", "1/1", "1/2"], ["变体", "1", "0/1", "0/1", "0/2"],
        ["合成难题", "1", "1/1", "1/1", "1/1"], ["合成单缺陷变体", "1", "0/1", "0/1", "0/1"]]
    assert len({eval_vllm.display_width(line) for line in table.splitlines()}) == 1
    assert table.splitlines()[0].startswith("层" + " " * 14 + "题数")        # 14 列宽，后面照例空两格
    # 只有合成变体的题、只做贪心：原有三层和合成难题一行都不印（表头照旧是五栏，只做贪心时后两栏写 "-"）
    only = eval_vllm.tier_table(more[4:], rows[3:], None)
    assert [line.split() for line in only.splitlines()] == [
        ["层", "题数", "贪心", "解出的题", "通过样本"], ["合成单缺陷变体", "1", "0/1", "-"]]


def test_a_run_with_synthetic_tasks_prints_every_tier_that_has_tasks(tmp_path, monkeypatch, capsys):
    """端到端：TIERS 有了合成两层，这批题里有一道合成难题。脚本从头跑到尾：分层表列出合成难题，没有题的合成变体
    不列，summary 的 by_tier 照旧每层一项。"""
    use_tiers(monkeypatch, NEW_TIERS)
    specs = [*write_tasks(tmp_path / "tasks.jsonl"), tiny_task("s3_demo_001", "train")]
    (tmp_path / "tasks.jsonl").write_text("".join(json.dumps(spec, ensure_ascii=False) + "\n" for spec in specs),
                                          encoding="utf-8")
    out = tmp_path / "eval"
    fake = FakeVLLM(model_answers(specs))
    monkeypatch.setattr(sys, "argv", [
        str(ROOT / "scripts/eval_vllm.py"), "--tasks-file", str(tmp_path / "tasks.jsonl"), "--split", "train",
        "--vllm-url", fake.url, "--model", "org/model", "--label", "base", "--output-dir", str(out),
        "--samples", "0", "--concurrency", "2", "--grade-workers", "2"])
    try:
        eval_vllm.main()
    finally:
        fake.close()
    shown = capsys.readouterr()
    assert shown.err == ""
    # 只做贪心：后两列写 "-"。合成变体没有题，不列
    assert [line.split() for line in shown.out.splitlines()
            if line.startswith(("原有题", "难题", "变体", "合成"))] == [
        ["原有题", "1", "0/1", "-"], ["难题", "1", "1/1", "-"], ["变体", "1", "0/1", "-"],
        ["合成难题", "1", "0/1", "-"]]
    assert shown.out.rstrip().splitlines()[-1] == "结果：base train 贪心 1/4"
    assert read(out / "base-train-summary.json")["greedy"] == {
        "pass": 1, "total": 4,
        "by_tier": {"default": [0, 1], "hard": [1, 1], "variant": [0, 1], "synthetic_hard": [0, 1],
                    "synthetic_variant": [0, 0]}}
