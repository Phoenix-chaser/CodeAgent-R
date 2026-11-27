# -*- coding: utf-8 -*-
"""引导采样这一轮（run_rft_guided.sh）的集成测试：把真实的脚本按它的调用顺序串起来跑。

服务是本机的假服务（FakeVLLM），教师是注入的假客户端，判分是真的（沙箱里跑 pytest）。每一步读的都是上一步
的脚本真的写出来的文件；命令行的参数照 run_rft.sh、run_rft_guided.sh、rft_lib.sh 里传的写。训练做不了，它的
位置（HF 贪心那张表要的选型目录）用 tests/test_checkpoint_selection.py 的辅助函数造出最小的来顶替。

  第 1 轮   sample_rft.py（不引导）→ prepare_data.py                   train-manifest、train-graded、验证组
  第 2 轮   write_skills.py → sample_rft.py（前缀、技能）
            prepare_data.py（D-prefix、D-skill）→ rft_control.py（F-control）
            tier_report.py（HF 贪心那张表）→ eval_vllm.py（dev、族内探针、自检）→ vllm_summary.py

用例：
- 全新跑一遍上面的整条链，最后原样重跑（齐全的步骤都不重做）。跑两遍：一遍照常，一遍模型的每个回答里都带着
  U+2028 这类会被当成换行的字符——这样的采样以前写进文件就读不回来，槽位被悄悄丢掉；
- 有一道引导题在上一轮没有一份写完了、又没通过的采样：配对对照配不上它，照脚本第 8 步的做法把它从两组里去掉；
- 自检没过（适配器的输出和基座完全一样）的组在汇总里带着自检的结果（eval_vllm.py 只留下 -sanity.json，汇总读它）；
- 自检没能和基座比较的组（探针题里没有这一组训练过的题）不作废，汇总里注明没能确认适配器是否生效；
- 采样文件的末行截在半截时原样重跑：那个槽位重做，不把新记录粘在半行后面；
- 真的选型目录交给真的 scripts/rft_vllm_eval.sh（第 13 步）：它读选型结果、起（假的）vLLM、逐组先探针后 dev、
  打印汇总；自检没过的那组修好以后原样重跑只补它。
"""
import ast
import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import vllm_summary
import write_skills
from agentlab.benchmark.protocol import build_messages
from agentlab.benchmark.tiers import tier_of
from agentlab.sampling.hints import SKILL_HINT
from select_checkpoint import checkpoint_candidates, choose_checkpoint
from test_checkpoint_selection import report, write_adapter
from test_rft_sampling import FakeVLLM, lines

BASE = "org/model"                  # 服务上的基座名；适配器按组名登记（vLLM 的 --lora-modules 组名=路径）
ARMS = ("D-prefix", "D-skill", "F-control")
UNTRAINED = "D-empty"               # 一个引导组：一道新题都没引导出来，训练集和基础集一样
TRAIN = ("h3_solved_001", "h3_zero_001", "h3v_zero_002", "h3v_zero_003")
ZERO_PASS = TRAIN[1:]               # 上一轮一次都没通过的三道：难题一道、变体两道
DEV = ("h3_dev_001", "demo_dev_001")
TIERS = "hard,variant"              # run_rft.sh / run_rft_guided.sh 的 RFT_TIERS
FRACTIONS = "0.25,0.5,0.75"         # HINT_FRACTIONS
SAMPLES = "2"                       # 每种采样各几次（RFT_SAMPLES、SKILL_SAMPLES、EVAL_SAMPLES）
# scripts/evaluate.py 的 main 写进报告的键：eval_vllm.py 的报告要与之同构
REPORT_KEYS = {"schema_version", "model", "requested_revision", "resolved_revision", "adapter", "tasks_file",
               "split", "task_ids", "task_hashes", "tasks_hash", "protocol", "protocol_hash", "contamination",
               "selection_contamination", "generation", "repeats", "metrics", "environment", "n_tasks",
               "success_rate", "results", "timestamp", "interpretation"}


def by_tier(default, hard, variant):
    """eval_vllm.py 写的 by_tier：tiers.py 里每层一项。这一轮的题只有原有三层，合成两层没有题，都是 [0, 0]。"""
    return {"default": default, "hard": hard, "variant": variant, "synth_hard": [0, 0], "synth_variant": [0, 0]}


BUGGY = "def clamp(values, low, high):\n    return list(values)\n"        # 公开测试过、隐藏测试挂：是代码，但判分不过
WRONG = "def clamp(values, low, high):\n    return tuple(values)\n"       # 公开测试就过不了
# 另一种对的写法：不带着参考解的前几行
SHORT = "def clamp(values, low, high):\n    return [min(max(item, low), high) for item in values]\n"
# 解释写到一半：接在有缺陷的代码后面，代码能解析，但输出没有写完
CUT_OFF = "# 修复思路：先逐个看元素有没有越界，越界的换成对应的边界，还要注意"
# 参考解 10 行：前缀提示按 0.25 / 0.5 / 0.75 给前 3 / 5 / 8 行，三档各不相同
FIXED = """def clamp(values, low, high):
    clamped = []
    for item in values:
        if item < low:
            clamped.append(low)
        elif item > high:
            clamped.append(high)
        else:
            clamped.append(item)
    return clamped
"""

CLEAN = {"applicability": "要把一批数限制在上下界之间时。", "principles": "低于下界和高于上界是两种情形，分别处理。",
         "pitfalls": "原样返回，或只处理了一侧。", "approach": "先列出越界的情形逐个处理，最后拿边界值核对。"}
LEAKY = {**CLEAN, "pitfalls": "把 values 原样返回。"}      # 带着参考修复里的参数名：静态检查拦得下
SECTIONS = ("适用条件", "要点", "常见错误", "推荐做法")      # write_skills.py 给四段加的小标题，按这个顺序
SKILL = "\n".join(f"【{title}】{body}" for title, body in zip(SECTIONS, CLEAN.values()))
JUDGE_MARK = "重建参考修复"                                 # 自判请求的系统提示里有、写技能的里没有
VOID = r"（评测作废或没做）"                                # vllm_summary.py 写没有探针报告的组的那一句
PAIRED_3 = r"训练过的题 3/3（基座 0/3）、没训练过的同族题：没有这样的题"     # 三道探针题都训练过的组
# 自检作数的是"输出和基座不同的题够不够门槛"（默认 1）；训练过的题严格通过几道只作参考
CHECKED = r"；自检：适配器的输出在 3 道题里有 3 道和基座不同（门槛 1）；训练过的题严格通过 3/3（仅供参考）"
# 自检没过（输出和基座完全一样）的组：eval_vllm.py 以 4 退出、只留下 -sanity.json；汇总从它读出自检结果，跟在"作废"后面
FAILED_SANITY = VOID + (r"；自检：适配器的输出在 3 道题里有 0 道和基座不同（门槛 1），适配器在 vLLM 里没有生效；"
                        r"训练过的题严格通过 0/3（仅供参考）")
NO_TRAINED = "本次要评的题里没有出现在 --sanity-tasks-file 里的，没有可比较的题"      # 探针题里一道训练过的都没有
SEPARATORS = "\u2028\u0085\u2029"       # str.splitlines() 把它们当成换行，JSON 的字符串里却可以原样出现


def tiny_task(task_id, split):
    """一道独立的小题：clamp 没做截断，公开测试过、隐藏测试挂。模块名跟着题号走，假服务才分得清是哪道题。"""
    module = f"{task_id}.py"
    return {
        "task_id": task_id, "source_group": f"group-{task_id}", "split": split,
        "bug_family": "clamping", "difficulty": "L1",
        "instruction": "实现 clamp(values, low, high)：把每个数限制在 [low, high] 内，返回新列表。",
        "module": module, "buggy": BUGGY, "fixed": FIXED,
        "public_test": f"from {module[:-3]} import clamp\n\n"
                       "def test_public():\n    assert clamp([1, 5], 0, 10) == [1, 5]\n",
        "private_tests": {"test_hidden.py": f"from {module[:-3]} import clamp\n\n"
                          "def test_hidden():\n    assert clamp([-3, 3, 13], 0, 10) == [0, 3, 10]\n"},
    }


def write_tasks(path):
    """冻结的题文件：train 四道（三道上一轮零通过）、dev 两道（难题一道、原有题一道）。"""
    specs = [tiny_task(name, "dev" if name in DEV else "train") for name in (*TRAIN, *DEV)]
    path.write_text("".join(json.dumps(spec, ensure_ascii=False) + "\n" for spec in specs), encoding="utf-8")
    return specs


def completion(text, finish="stop"):
    return {"choices": [{"message": {"content": text}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 5}}


class FakeModel:
    """假服务背后的"模型"：回答由题目、提示、模型名、解码方式和种子决定。

    基座、不带提示（第 1 轮和评测时的样子）见 unguided。带着参考解的开头：h3_zero_001 全对；h3v_zero_002
    至少给到第 5 行才对，给到第 8 行时换一种对的写法；h3v_zero_003 偶数号才对。带着技能：前两道全对，
    h3v_zero_003 偶数号才对。
    适配器（模型名是组名）：生效、而且训练数据里有这道题时，照训练数据里的答案回答——D 组学到的是正确答案，
    F-control 学到的是没通过的答案。dev 上的题谁都没训练过，各组比基座多对几道见 DEV_SOLID、DEV_SHAKY。
    effective 里记成 False 的组和基座一样：模拟适配器在服务里没有生效。
    """
    DEV_SOLID = {"D-prefix": {"h3_dev_001"}, "D-skill": {"h3_dev_001", "demo_dev_001"}}
    DEV_SHAKY = {"D-prefix": {"demo_dev_001"}}       # 采样时偶数号才对，贪心不对

    def __init__(self, specs, sft):
        self.specs = {spec["module"]: spec for spec in specs}
        self.sft = Path(sft)
        self.never_finishes = set()                  # 这些题不带提示时从来不写完
        self.effective = {}                          # 记成 False 的组：适配器在服务里没有生效

    def trained(self, arm):
        """这一组的训练数据（prepare_data.py / rft_control.py 真的写出来的 train.jsonl）里有哪几道题。"""
        path = self.sft / arm / "train.jsonl"
        return {record["task_id"] for record in lines(path)} if path.exists() else set()

    def unguided(self, name, greedy, even):
        """不带提示时的 (回答, 是否写完)。贪心请求不带种子，按偶数算。"""
        if name in self.never_finishes:
            return BUGGY + CUT_OFF, "length"
        if name == "h3v_zero_002":
            return (WRONG, "stop") if even else (BUGGY + CUT_OFF, "length")
        if name == "h3v_zero_003" and even:
            return BUGGY + CUT_OFF, "length"
        if name in {"h3_solved_001", "h3_dev_001"} and even and not greedy:
            return FIXED, "stop"
        return BUGGY, "stop"

    def shown_lines(self, spec, hint):
        """前缀提示给了参考解的前几行；不是前缀提示时是 None。"""
        code = spec["fixed"].splitlines()
        for count in range(len(code) - 1, 0, -1):
            if hint.endswith("\n".join(code[:count])):
                return count
        return None

    def answer(self, payload):
        user = payload["messages"][-1]["content"]
        spec = next((item for module, item in self.specs.items() if f"文件 {module}：" in user), None)
        if spec is None:                 # 探测参数、确认模型名的请求（只生成 1 个 token）：认不出题目
            return 200, completion("")
        name, arm = spec["task_id"], payload["model"]
        greedy = payload["temperature"] == 0
        even = payload.get("seed", 0) % 2 == 0
        canonical = build_messages(spec)[1]["content"]
        hint = user[len(canonical):] if user.startswith(canonical) else None      # 提示追加在标准提示的后面
        shown = self.shown_lines(spec, hint or "")
        if arm != BASE:
            code, finish = self.as_adapter(arm, name, greedy, even)
        elif shown:
            code, finish = self.with_prefix(name, shown, even)
        elif hint:
            code, finish = self.with_skill(name, even)
        else:
            code, finish = self.unguided(name, greedy, even)
        return 200, completion(code, finish)

    def with_prefix(self, name, shown, even):
        if name == "h3v_zero_002":
            if shown >= 8:
                return SHORT, "stop"
            return (FIXED, "stop") if shown >= 5 else (BUGGY, "stop")
        if name == "h3v_zero_003" and not even:
            return BUGGY, "stop"
        return FIXED, "stop"

    def with_skill(self, name, even):
        return (BUGGY, "stop") if name == "h3v_zero_003" and not even else (FIXED, "stop")

    def as_adapter(self, arm, name, greedy, even):
        """适配器的回答：生效、而且训练数据里有这道题时照训练数据里的答案；其余和基座一样。"""
        if not self.effective.get(arm, True):
            return self.unguided(name, greedy, even)
        if name in self.trained(arm):            # D 组的训练数据里是正确答案，F-control 里是没通过的
            return (BUGGY if arm == "F-control" else FIXED), "stop"
        shaky = name in self.DEV_SHAKY.get(arm, ()) and even and not greedy
        if name in self.DEV_SOLID.get(arm, ()) or shaky:
            return FIXED, "stop"
        return self.unguided(name, greedy, even)


class FakeServing(FakeVLLM):
    """假的 vLLM 服务：除了基座，还按组名登记着各组的适配器（/v1/models 里它们的 parent 指向基座）。"""

    def __init__(self, answer, adapters):
        super().__init__(answer)
        listing = [{"id": BASE, "parent": None}, *({"id": name, "parent": BASE} for name in adapters)]
        base_handler = self.server.RequestHandlerClass

        class Handler(base_handler):
            def do_GET(self):
                self._reply(200, {"data": listing})

        self.server.RequestHandlerClass = Handler


class FakeTeacher:
    """注入给 write_skills.main 的假教师客户端：接口同 openai SDK（chat.completions.create）。

    写技能的回复是 CLEAN；leaky_first 里的题第一版带着参考修复里的参数名，被静态检查拦下后才给干净的。
    自判一律说不能重建。记下每次请求。
    """

    def __init__(self, specs, leaky_first=()):
        self.specs, self.requests = specs, []
        self.leaky_first, self.diagnoses = set(leaky_first), Counter()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        messages = kwargs["messages"]
        self.requests.append(kwargs)
        # 写技能的消息里是"## 有缺陷的文件 <模块名>"，自判的消息里是模型看到的"当前有缺陷的文件 <模块名>："
        spec = next(item for item in self.specs if f"文件 {item['module']}" in messages[1]["content"])
        if JUDGE_MARK in messages[0]["content"]:                  # 自判
            text = json.dumps({"reconstructable": False, "reason": "只是指出了方向"})
        else:
            first = self.diagnoses[spec["task_id"]] == 0
            self.diagnoses[spec["task_id"]] += 1
            leaky = first and spec["task_id"] in self.leaky_first
            text = json.dumps(LEAKY if leaky else CLEAN, ensure_ascii=False)
        choice = SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=text))
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=120, completion_tokens=40), choices=[choice])


# ---- 跑脚本、读产物 ----


def script(name, *arguments):
    """跑仓库里的一个脚本；Path 原样传。shell 脚本都是先回到仓库根再调它们，这里也一样。"""
    return subprocess.run([sys.executable, str(ROOT / "scripts" / name), *map(str, arguments)],
                          capture_output=True, text=True, timeout=600, cwd=ROOT)


# 进度面板（scripts/show_progress.sh）认的进度行：同 run_rft.sh 的 progress()。它见到以 "=====" 开头的行就当成
# 步骤标记，见到 "===== 全部结束 =====" 就判整轮已完成——这些只能由外层的 shell 脚本打印
BAR = re.compile(r"^\[[#-]{24}\] \d+/\d+ \(\d+%\) .+ · 已用 \d+\.\dm · 剩余≈\d+\.\dm$")


def panel_friendly(text):
    """子脚本的输出：不打步骤标记；进度行都是面板认的格式。"""
    rows = text.splitlines()
    assert not any(line.startswith("=====") for line in rows), text[-3000:]
    bars = [line for line in rows if line.startswith(("[#", "[-"))]
    assert all(BAR.match(line) for line in bars), [line for line in bars if not BAR.match(line)][:3]


def ok(result):
    """退出码是 0 才往下走；失败时带上输出的末尾。不许有 Traceback，输出也要是进度面板认得的样子。"""
    assert result.returncode == 0, f"退出码 {result.returncode}\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
    assert "Traceback" not in result.stderr, result.stderr[-3000:]
    panel_friendly(result.stdout)
    return result


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def names(directory):
    return sorted(path.name for path in Path(directory).iterdir())


def task_ids(path):
    """一份训练数据里的题号，按文件里的顺序，重复的照列。"""
    return [record["task_id"] for record in lines(path)]


def finished_code(record):
    """这条采样模型写完了（finish_reason 是 stop），而且代码是完整的 Python。"""
    try:
        return record["finish_reason"] == "stop" and bool(ast.parse(record["target"]["content"]).body)
    except SyntaxError:
        return False


def evaluated(directory, label, stem):
    """rft_eval_done 认"评过"的条件：summary 在，它提到的报告也都在。"""
    summary_path = Path(directory) / f"{label}-{stem}-summary.json"
    if not summary_path.is_file():
        return False
    summary = read(summary_path)
    wanted = [f"{label}-{stem}-greedy.json"]
    if isinstance(summary.get("sampled"), dict) and summary["sampled"].get("samples_per_task"):
        k = summary["sampled"]["samples_per_task"]
        wanted.append(f"{label}-{stem}-sampled-k{k}.json")
    return all((Path(directory) / name).is_file() for name in wanted)


def summary_sections(text):
    """vllm_summary.py 的输出按三个标题切成（vLLM 评测，族内探针，怎么读）。"""
    rows = text.splitlines()
    probe = next(index for index, line in enumerate(rows) if line.startswith("── 族内探针"))
    how = rows.index("── 怎么读 ──")
    return rows[:probe], rows[probe:how], rows[how:]


def line_starting(section, label):
    """这一段里以 label 开头的那一行；必须恰好一行。"""
    matching = [line for line in section if line.startswith(label)]
    assert len(matching) == 1, section
    return matching[0]


def expect(section, label, pattern):
    """以 label 开头的那一行（必须恰好一行）：label 后面隔一段空白，恰好是 pattern。"""
    line = line_starting(section, label)
    assert re.fullmatch(re.escape(label) + r"\s+" + pattern, line), line
    return line


def verdict(section):
    matching = [line for line in section if line.startswith("结果：")]
    assert len(matching) == 1, section
    return matching[0]


def with_line_separators(answer):
    """把假服务的每个回答前面加一句带着 U+2028、U+0085、U+2029 的说明。代码在它后面，extract_code 取的还是代码。"""
    def decorated(payload):
        status, body = answer(payload)
        for choice in body["choices"]:
            if choice["message"]["content"]:
                choice["message"]["content"] = f"说明：{SEPARATORS}。\n" + choice["message"]["content"]
        return status, body
    return decorated


def cut_last_line(path, keep=20):
    """把文件最后一行截掉后半截：像上一次运行写到一半被打断。"""
    whole = path.read_bytes()[:-1].split(b"\n")
    path.write_bytes(b"\n".join([*whole[:-1], whole[-1][:keep]]))


# scripts/rft_vllm_eval.sh 从环境里读的开关：外面留着的一律不带进来
SHELL_SWITCHES = ("MODEL", "MODEL_REVISION", "TASKS_FILE", "EVAL_SAMPLES", "EXPECT_REPRODUCE", "PROBE_IDS_FILE",
                  "INCLUDE_BASE", "AGENTLAB_SANDBOX")
# 第 13 步要用的几个程序换成假的（放在 PATH 最前面）：不碰真的显卡、真的 vllm，也不关别的进程
FAKE_TOOLS = {
    "python": f'#!/bin/sh\nexec "{sys.executable}" "$@"\n',                       # 脚本里写的是 python
    "vllm": ('#!/usr/bin/env bash\n'                                           # 记下启动参数，报告就绪后常驻
             'printf \'%s\\n\' "$*" >> "$FAKE_VLLM_ARGS"\necho $$ >> "$FAKE_VLLM_PID_FILE"\n'
             'echo "INFO:     Application startup complete."\nwhile true; do sleep 1; done\n'),
    # 只管命令行里带着这次演练目录（STAGE）的进程；不知道演练目录就什么都不关
    "pkill": ('#!/usr/bin/env bash\n'
              '[[ "$1" == -f && $# == 2 && -n "$STAGE" ]] || exit 2\nstatus=1\n'
              'for entry in /proc/[0-9]*; do\n'
              '  [[ -r "$entry/cmdline" && "$entry" != "/proc/$$" ]] || continue\n'
              '  line="$(tr \'\\0\' \' \' < "$entry/cmdline")"\n'
              '  [[ "$line" == *"$STAGE"* && "$line" =~ $2 ]] || continue\n'
              '  kill "${entry#/proc/}" 2>/dev/null && status=0\ndone\nexit $status\n'),
    "nvidia-smi": "#!/usr/bin/env bash\nexit 0\n",                              # 什么都查不到：当作显卡空着
    # 启动 vLLM 前固定等的那 3 秒不必真等
    "sleep": (f'#!/usr/bin/env bash\ncase "$*" in 3) exec {shutil.which("sleep")} 0.1 ;; esac\n'
              f'exec {shutil.which("sleep")} "$@"\n'),
}


class Pipeline:
    """一次演练：题文件、假服务、假教师，和各个脚本的真实调用（参数照 shell 脚本里传的写）。

    poisoned 为真时假服务的每个回答里都带着 U+2028 这类字符。
    """

    def __init__(self, tmp_path, poisoned=False):
        self.tmp = tmp_path
        self.tasks = tmp_path / "tasks.jsonl"
        self.specs = write_tasks(self.tasks)
        self.round1, self.run = tmp_path / "round1", tmp_path / "run"
        self.sampling, self.sft, self.vllm = self.run / "sampling", self.run / "sft", self.run / "vllm"
        self.model = FakeModel(self.specs, self.sft)
        answer = with_line_separators(self.model.answer) if poisoned else self.model.answer
        self.serving = FakeServing(answer, (*ARMS, UNTRAINED))
        self.teacher = FakeTeacher(self.specs, leaky_first=("h3v_zero_002",))
        self.poisoned = poisoned

    def close(self):
        self.serving.close()

    def generation_requests(self, since=0):
        """假服务收到的生成请求（since 是这次运行之前已收到的请求数）；探测参数用的请求只生成 1 个 token，不算。"""
        return [request for request in self.serving.requests[since:] if request["max_tokens"] == 2048]

    def sample(self, split, output, *extra):
        """run_rft.sh 第 2 步、run_rft_guided.sh 的 guided_sample 传给 sample_rft.py 的参数。"""
        return script("sample_rft.py", "--tasks-file", self.tasks, "--split", split, "--tiers", TIERS,
                      "--vllm-url", self.serving.url, "--model", BASE, "--concurrency", "4",
                      "--grade-workers", "2", "--output-dir", output, *extra)

    def prepare(self, output, *inputs, extra=()):
        """prepare_data.py：inputs 是候选文件；extra 是 --drop-tasks-file 这样的额外参数。"""
        return script("prepare_data.py", "--tasks-file", self.tasks, "--input", *inputs,
                      "--output-dir", output, "--max-per-task", "1", *extra)

    def first_round(self):
        """第 1 轮（run_rft.sh 第 2、3 步）：不引导的采样，再准备验证组。返回 train 那次采样的输出。"""
        outputs = {split: ok(self.sample(split, self.round1 / "sampling", "--samples", SAMPLES))
                   for split in ("train", "dev")}
        ok(self.prepare(self.round1 / "sft/verified", self.round1 / "sampling/train-candidates.jsonl",
                        self.round1 / "sampling/dev-candidates.jsonl"))
        return outputs["train"]

    def write_skills(self):
        """run_rft_guided.sh 第 4 步的参数，教师换成注入的假客户端。返回 (退出码, 标准输出, 标准错误)。"""
        command = ["--tasks-file", str(self.tasks),
                   "--zero-pass-from", str(self.round1 / "sampling/train-manifest.json"),
                   "--split", "train", "--tiers", TIERS,
                   "--graded", str(self.round1 / "sampling/train-graded.jsonl"),
                   "--output", str(self.run / "skills.jsonl"), "--concurrency", "2", "--model", "teacher-test"]
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = write_skills.main(command, client=self.teacher)
        return code, out.getvalue(), err.getvalue()

    def guided(self, tag, *extra):
        """run_rft_guided.sh 的 guided_sample：只采上一轮零通过的题，输出文件名里带 tag。"""
        return self.sample("train", self.sampling, *extra,
                           "--only-zero-pass", self.round1 / "sampling/train-manifest.json", "--tag", tag)

    def guided_prefix(self):
        return self.guided("guided-prefix", "--hint-kind", "prefix", "--hint-fractions", FRACTIONS,
                           "--samples", SAMPLES)

    def guided_skill(self):
        return self.guided("guided-skill", "--hint-kind", "skill", "--samples", SAMPLES,
                           "--skills-file", self.run / "skills.jsonl")

    def prepare_group(self, name, tag, *extra):
        """run_rft_guided.sh 的 prepare_guided：第 1 轮的验证组加上这种引导采出的候选。"""
        return self.prepare(self.sft / name, self.round1 / "sft/verified/train.jsonl",
                            self.round1 / "sft/verified/dev.jsonl", self.sampling / f"train-{tag}-candidates.jsonl",
                            extra=extra)

    def build_control(self, *groups):
        """run_rft_guided.sh 的 build_control：基础记录加上每道引导题一份没通过的采样。"""
        return script("rft_control.py", "--graded", self.round1 / "sampling/train-graded.jsonl",
                      "--base-dir", self.round1 / "sft/verified", "--guided-from", *groups,
                      "--output-dir", self.sft / "F-control", "--seed", "42")

    def write_probe_ids(self):
        """run_rft_guided.sh 第 2 步：探针 = 上一轮零通过的题里属于 TIERS 层的，按题号排序。"""
        manifest = read(self.round1 / "sampling/train-manifest.json")
        probe = sorted(task for task in manifest["zero_pass_tasks"] if tier_of(task) in TIERS.split(","))
        (self.run / "probe-ids.txt").write_text("".join(f"{task}\n" for task in probe), encoding="utf-8")
        return probe

    def evaluate(self, label, *extra):
        """rft_eval_dev / rft_eval_probe 传给 eval_vllm.py 的参数。评基座时模型名是基座名，评某一组时是组名。"""
        return script("eval_vllm.py", "--tasks-file", self.tasks, "--vllm-url", self.serving.url,
                      "--model", BASE if label == "base" else label, "--label", label,
                      "--output-dir", self.vllm / label, "--grade-workers", "2", *extra)

    def eval_dev(self, label):
        return self.evaluate(label, "--split", "dev", "--samples", SAMPLES)

    def eval_probe(self, label, training=None):
        """族内探针（只贪心）。给了训练数据就先自检（同 rft_eval_probe：不传 --sanity-min，门槛就是默认的 1）。"""
        extra = ["--split", "train", "--ids-file", self.run / "probe-ids.txt", "--tag", "probe", "--samples", "0"]
        if training is not None:
            extra += ["--sanity-tasks-file", training]
        return self.evaluate(label, *extra)

    def summary(self, *arms):
        return script("vllm_summary.py", "--run", self.run, "--arms", *arms,
                      "--probe-ids-file", self.run / "probe-ids.txt")

    def rate(self, *arms):
        """run_rft_guided.sh 第 13 步：真的 scripts/rft_vllm_eval.sh 读真的选型目录（rft_lib.sh 也是真的），起一个
        假的 vLLM（只记下启动参数），逐组调真的 eval_vllm.py 评 FakeServing，最后打印真的汇总。EXPECT_REPRODUCE
        是这几组：评探针前先自检。不检查输出是不是进度面板认得的样子——步骤标记本来就是这个脚本打的。"""
        tools = self.tmp / "bin"
        for name, body in FAKE_TOOLS.items():
            path = tools / name
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(body, encoding="utf-8")
                path.chmod(0o755)
        outer = {name: value for name, value in os.environ.items()
                 if name not in SHELL_SWITCHES and not name.startswith(("RFT_", "STUB_", "FAKE_", "VLLM_"))}
        # HOME 不改：项目要的包可能装在用户目录下
        environment = {**outer, "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}",
                       "STAGE": str(self.tmp), "FAKE_VLLM_ARGS": str(self.tmp / "vllm-args.log"),
                       "FAKE_VLLM_PID_FILE": str(self.tmp / "vllm.pid"), "MODEL": BASE, "MODEL_REVISION": "rev-1",
                       "TASKS_FILE": str(self.tasks), "VLLM_BIN": str(tools / "vllm"),
                       "VLLM_PORT": self.serving.url.rsplit(":", 1)[1], "RFT_GRADE_WORKERS": "2",
                       "RFT_POLL_SECONDS": "1", "EVAL_SAMPLES": SAMPLES,
                       # 评探针前要自检的是 D 组（同 run_rft_guided.sh）；对照组不在这份名单里
                       "EXPECT_REPRODUCE": " ".join(arm for arm in arms if arm.startswith("D-")),
                       "PROBE_IDS_FILE": str(self.run / "probe-ids.txt")}
        result = subprocess.run(["bash", "scripts/rft_vllm_eval.sh", str(self.run), *arms], cwd=ROOT, env=environment,
                                capture_output=True, text=True, timeout=600)
        assert "Traceback" not in result.stderr, result.stderr[-3000:]
        assert eventually(lambda: not self.fake_vllm_alive()), "假的 vllm 还在跑：用完没有关掉"
        return result

    def fake_vllm_alive(self):
        """假的 vllm（FAKE_VLLM_PID_FILE 里记下的那些）还有活着的没有；已经退出、还没收回的僵尸不算。"""
        pid_file = self.tmp / "vllm.pid"
        for pid in pid_file.read_text(encoding="utf-8").split() if pid_file.exists() else []:
            try:
                if Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z":
                    return True
            except OSError:
                continue
        return False

    def vllm_starts(self):
        """假的 vllm 每次被启动时收到的参数。"""
        log = self.tmp / "vllm-args.log"
        return log.read_text(encoding="utf-8").splitlines() if log.exists() else []

    def select(self, root, arm, passes):
        """给一组造一个最小的选型目录（select_checkpoint.py 的输出），HF 那张表和 vLLM 评测都读它。

        检查点目录用 tests/test_checkpoint_selection.py 的 write_adapter 造（基座和一个适配器两个 epoch），
        候选和选型结论走真的 checkpoint_candidates / choose_checkpoint，dev 报告是那里的 report()：
        passes 是适配器在两道题上过没过，和基座打平时按规则保留基座。
        """
        checkpoints, results = root / "checkpoints" / arm, root / "results" / arm
        write_adapter(checkpoints)
        _, candidates = checkpoint_candidates(checkpoints)
        chosen = choose_checkpoint(candidates, [report(), report(candidates[1]["adapter"], passes)])
        chosen["configuration"] = {"checkpoint_root": str(checkpoints), "candidates": candidates,
                                   "model": BASE, "revision": "rev-1"}
        results.mkdir(parents=True)
        for epoch, dev_report in ((0, report()), (1, report(candidates[1]["adapter"], passes))):
            (results / f"epoch-{epoch:03d}-dev.json").write_text(json.dumps(dev_report), encoding="utf-8")
        (results / "selection.json").write_text(json.dumps(chosen, ensure_ascii=False, indent=2), encoding="utf-8")
        return chosen


def eventually(condition, seconds=5.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


@pytest.fixture
def pipeline(tmp_path, request):
    pipe = Pipeline(tmp_path, poisoned=getattr(request, "param", None) == "poisoned")
    yield pipe
    pipe.close()


def task_of(pipe, content):
    """一段提示说的是哪道题（提示里写着"当前有缺陷的文件 <模块名>："）。"""
    return next(spec for spec in pipe.specs if f"文件 {spec['module']}：" in content)


def hinted(pipe, requests):
    """把生成请求逐条拆成 (题, 提示)：提示是追加在不带提示的标准提示后面的那一段。"""
    rows = []
    for request in requests:
        spec = task_of(pipe, request["messages"][-1]["content"])
        user = request["messages"][-1]["content"]
        canonical = build_messages(spec)[1]["content"]
        assert user.startswith(canonical + "\n\n"), user[:300]
        rows.append((spec["task_id"], user[len(canonical) + 2:]))
    return rows


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spec_of(pipe, task_id):
    return next(spec for spec in pipe.specs if spec["task_id"] == task_id)


def check_first_round(pipe, sampled, finished, failures):
    """第 1 轮的产物：清单里零通过的是那三道；每道两份没通过的采样，其中写完了、格式完整的各有 finished[task]
    份；验证组里只有答对的那道（第 1 轮只采属于 TIERS 层的题，dev 的原有题不在里面）。"""
    round1 = pipe.round1
    assert "结果：train 难题 解出 1/2（通过样本 1/4）、变体 解出 0/2（通过样本 0/4）" in sampled.stdout
    assert f"一次都没通过的题：3 道 {list(ZERO_PASS)}" in sampled.stdout
    manifest = read(round1 / "sampling/train-manifest.json")
    assert manifest["zero_pass_tasks"] == list(ZERO_PASS) and manifest["failure_types"] == failures
    assert {tier: (row["tasks"], row["graded_samples"], row["passing_samples"], row["solved_tasks"])
            for tier, row in manifest["tiers"].items()} == {
        "hard": (2, 4, 1, 1), "variant": (2, 4, 0, 0), "default": (0, 0, 0, 0),
        "synth_hard": (0, 0, 0, 0), "synth_variant": (0, 0, 0, 0)}      # 清单里每层一项；合成两层没有题
    graded = lines(round1 / "sampling/train-graded.jsonl")
    assert Counter(record["task_id"] for record in graded) == {task: 2 for task in TRAIN}
    for task in ZERO_PASS:
        rows = [record for record in graded if record["task_id"] == task]
        assert not any(record["provenance"]["strict_pass"] for record in rows)      # 零通过的题：两份都没过
        assert sum(finished_code(record) for record in rows) == finished[task]
    assert task_ids(round1 / "sft/verified/train.jsonl") == ["h3_solved_001"]
    assert task_ids(round1 / "sft/verified/dev.jsonl") == ["h3_dev_001"]
    dev = read(round1 / "sft/verified/dataset-manifest.json")["splits"]["dev"]
    assert dev["task_ids"] == ["h3_dev_001"] and dev["missing_task_ids"] == ["demo_dev_001"]


def check_skills(pipe):
    """教师写技能（run_rft_guided.sh 第 4 步）：三道题都拿到可用的技能。返回 {题号: 技能正文}。"""
    skills_path = pipe.run / "skills.jsonl"
    code, out, err = pipe.write_skills()
    assert (code, err) == (0, "")
    assert out.rstrip().splitlines()[-1] == (f"结果：技能提示 可用 3/3 道，泄露被拦 0 道，出错 0 道；"
                                             f"教师调用 8 次，输出 {skills_path}")
    assert "写技能 h3v_zero_002 ok（2 轮）" in out           # 泄露被拦下后重写的那一道
    panel_friendly(out)
    skills = {record["task_id"]: record for record in lines(skills_path)}
    assert {task: record["status"] for task, record in skills.items()} == {task: "ok" for task in ZERO_PASS}
    assert [skills[task]["rounds"] for task in ZERO_PASS] == [1, 2, 1]      # h3v_zero_002 第一版带着名字，重写才过
    assert skills["h3v_zero_002"]["checks"] == [
        {"round": 1, "leaks": ["values"], "reconstructable": False, "reason": "只是指出了方向"},
        {"round": 2, "leaks": [], "reconstructable": False, "reason": "只是指出了方向"}]
    assert {record["text"] for record in skills.values()} == {SKILL}
    assert {record["model"] for record in skills.values()} == {"teacher-test"}
    return {task: record["text"] for task, record in skills.items()}


def diagnoses_for(pipe, task):
    """教师为这道题收到的写技能的请求（用户消息），按先后。"""
    return [request["messages"][1]["content"] for request in pipe.teacher.requests
            if JUDGE_MARK not in request["messages"][0]["content"]
            and f"## 有缺陷的文件 {spec_of(pipe, task)['module']}" in request["messages"][1]["content"]]


def check_teacher_sees_the_real_failures(pipe):
    """教师读到的是上一轮真的判过分的失败尝试：失败类型、没通过的隐藏测试、代码，以及这道题不许出现的名字。"""
    requests = pipe.teacher.requests
    assert len(requests) == 8                                   # a 写+判；b 两轮各写+判；c 写+判
    assert {(r["model"], r["max_tokens"], r["temperature"]) for r in requests} == {("teacher-test", 4096, 0.3)}
    diagnoses = [r for r in requests if JUDGE_MARK not in r["messages"][0]["content"]]
    judgements = [r for r in requests if JUDGE_MARK in r["messages"][0]["content"]]
    assert len(diagnoses) == 4 and len(judgements) == 4
    asked = {task: diagnoses_for(pipe, task) for task in ZERO_PASS}
    assert [len(asked[task]) for task in ZERO_PASS] == [1, 2, 1]
    same_code = asked["h3_zero_001"][0]
    assert ("### 失败尝试 1\n失败类型：private_test_failure（公开测试通过，隐藏测试没通过）\n"
            "没通过的隐藏测试：test_hidden.py::test_hidden\n```python\n" + BUGGY.strip() + "\n```") in same_code
    assert "失败尝试 2" not in same_code           # 两份采样代码相同：只给教师看一份
    assert (same_code.index("## 参考修复") < same_code.index("## 模型的失败尝试")
            < same_code.index("## 技能里不许出现的名字"))
    forbidden = same_code[same_code.index("## 技能里不许出现的名字"):]
    # 这道题的名字：函数名、参数名（短的 low 不要）、测试名、题号；按字典序列给教师
    assert forbidden.endswith("\nclamp、h3_zero_001、high、test_hidden、test_public、values")
    two_kinds = asked["h3v_zero_002"][0]
    assert two_kinds.count("### 失败尝试 ") == 2   # 两份代码不同；都不是"离做对最近的"，先后看判分完成的先后
    # 公开测试没过的那一份：没通过的隐藏测试（也跑了、也没过）和没通过的公开测试都列出来
    assert ("失败类型：public_test_failure（公开测试没通过）\n没通过的隐藏测试：test_hidden.py::test_hidden\n"
            "没通过的公开测试：test_public.py::test_public\n```python\n" + WRONG.strip() + "\n```") in two_kinds
    assert two_kinds.count("没通过的公开测试：") == 1              # 公开层过了的那一份、没写完的那一份没有这一行
    # 没写完的不判分，也列给教师
    assert ("失败类型：incomplete_response（输出没写完）\n没通过的隐藏测试：（没有记录）\n"
            f"```python\n{(BUGGY + CUT_OFF).strip()}\n```") in two_kinds
    leaky_skill = "\n".join(f"【{title}】{body}" for title, body in zip(SECTIONS, LEAKY.values()))
    second = asked["h3v_zero_002"][1]      # 第二轮带着上一版的原文和意见：接口不带上下文，教师不记得上一版
    assert ("## 上一版的问题\n\n上一版技能：\n\n" + leaky_skill
            + "\n上一版的问题：\n- 出现了这道题里的具体名字或字面量：values。") in second
    assert second.endswith("请重新写一条，五条硬性要求都要满足。")
    near_miss_first = asked["h3v_zero_003"][0]
    assert (near_miss_first.index("失败类型：private_test_failure（公开测试通过，隐藏测试没通过）")
            < near_miss_first.index("失败类型：incomplete_response（输出没写完）"))  # 离做对最近的排在前面
    assert Counter((task_of(pipe, r["messages"][1]["content"])["task_id"],
                    r["messages"][1]["content"].split("## 技能\n\n")[1].rsplit("\n\n## 参考修复", 1)[0])
                   for r in judgements) == Counter({
        ("h3_zero_001", SKILL): 1, ("h3v_zero_002", leaky_skill): 1, ("h3v_zero_002", SKILL): 1,
        ("h3v_zero_003", SKILL): 1})                 # 自判看的是那一轮真的写出来的技能


# ---- 第 2 轮的各个阶段：每一段先跑真脚本，再对着它们写出来的文件核对 ----


def check_prefix_sampling(pipe):
    """前缀引导采样（run_rft_guided.sh 第 5 步）：参考解的开头按三种比例各给一部分，每种采两次。"""
    sampling = pipe.sampling
    mark = len(pipe.serving.requests)
    prefix = ok(pipe.guided_prefix())
    assert ("采样 train-guided-prefix：3 道题 × 6 次 = 18 个槽位，已有 0 个，待生成 18 个；"
            "--only-zero-pass 指定 3 道，其中 3 道在 train 里") in prefix.stdout
    assert [line for line in prefix.stdout.splitlines() if line.startswith("结果：")] == [
        "结果：train-guided-prefix 难题 解出 1/1（通过样本 6/6）、变体 解出 2/2（通过样本 7/12）",
        "结果：train-guided-prefix 引导（前缀 0.25）解出 2/3 题（通过样本 3/6）、照抄了开头的 3/6（其中通过 3）",
        "结果：train-guided-prefix 引导（前缀 0.5）解出 3/3 题（通过样本 5/6）、照抄了开头的 5/6（其中通过 5）",
        "结果：train-guided-prefix 引导（前缀 0.75）解出 3/3 题（通过样本 5/6）、照抄了开头的 3/6（其中通过 3）"]
    assert "交给训练的候选：5 条（每题只取最短的、成功过的比例）" in prefix.stdout
    manifest = read(sampling / "train-guided-prefix-manifest.json")
    assert (manifest["tasks"], manifest["slots"], manifest["graded_records"]) == (3, 18, 18)
    assert (manifest["candidates"], manifest["candidates_kept_for_training"]) == (13, 5)
    assert manifest["zero_pass_tasks"] == []                        # 带着提示，三道都解出来了
    # 每种提示一行：(比例, 通过样本, 解出的题, 照抄了开头的, 其中通过)。
    # 0.75 那一档 h3v_zero_002 答对的两份换了写法，没有照抄开头
    assert [(row["fraction"], row["passing_samples"], row["solved_tasks"], row["restated"], row["passing_restated"])
            for row in manifest["hint"]] == [(0.25, 3, 2, 3, 3), (0.5, 5, 3, 5, 5), (0.75, 5, 3, 3, 3)]
    protocol = read(sampling / "train-guided-prefix-protocol.json")
    assert protocol["hint"] == {"kind": "prefix", "samples_per_hint": 2, "fractions": [0.25, 0.5, 0.75],
                                "zero_pass_from": digest(pipe.round1 / "sampling/train-manifest.json"),
                                "skills_sha256": None}
    guided = lines(sampling / "train-guided-prefix-graded.jsonl")
    assert Counter((r["task_id"], r["provenance"]["guided"]["fraction"], r["provenance"]["strict_pass"])
                   for r in guided) == Counter({
        ("h3_zero_001", 0.25, True): 2, ("h3_zero_001", 0.5, True): 2, ("h3_zero_001", 0.75, True): 2,
        ("h3v_zero_002", 0.25, False): 2, ("h3v_zero_002", 0.5, True): 2, ("h3v_zero_002", 0.75, True): 2,
        **{("h3v_zero_003", fraction, passed): 1 for fraction in (0.25, 0.5, 0.75) for passed in (True, False)}})
    assert sum(r["target"]["content"] == SHORT for r in guided) == 2          # 给到第 8 行时换了一种对的写法
    kept = lines(sampling / "train-guided-prefix-candidates.jsonl")
    assert Counter((r["task_id"], r["provenance"]["guided"]["fraction"]) for r in kept) == {
        ("h3_zero_001", 0.25): 2, ("h3v_zero_002", 0.5): 2, ("h3v_zero_003", 0.25): 1}    # 每题只留最短的、成功过的比例
    assert {r["target"]["content"] for r in kept} == {FIXED}                 # 换了写法的那两份只留在 graded 里
    sent = pipe.generation_requests(mark)
    assert Counter((task, pipe.model.shown_lines(spec_of(pipe, task), hint)) for task, hint in hinted(pipe, sent)) == {
        (task, shown): 2 for task in ZERO_PASS for shown in (3, 5, 8)}
    assert sorted(r["seed"] for r in sent) == sorted(42 + 1000 * position + index
                                                     for position in range(3) for index in range(6))
    assert Counter(r["temperature"] for r in sent) == {0.7: 9, 1.0: 9}      # 奇数号用探索温度
    assert {r["model"] for r in sent} == {BASE}


def check_skill_sampling(pipe, skills):
    """技能引导采样（run_rft_guided.sh 第 6 步）：每题带着教师写的技能采两次。"""
    sampling = pipe.sampling
    mark = len(pipe.serving.requests)
    skill = ok(pipe.guided_skill())
    assert [line for line in skill.stdout.splitlines() if line.startswith("结果：")] == [
        "结果：train-guided-skill 难题 解出 1/1（通过样本 2/2）、变体 解出 2/2（通过样本 3/4）",
        "结果：train-guided-skill 引导（技能提示）解出 3/3 题（通过样本 5/6）"]
    assert ("采样 train-guided-skill：3 道题 × 2 次 = 6 个槽位，已有 0 个，待生成 6 个；"
            "--only-zero-pass 指定 3 道，其中 3 道在 train 里；有可用技能的 3 道") in skill.stdout
    manifest = read(sampling / "train-guided-skill-manifest.json")
    assert (manifest["slots"], manifest["graded_records"], manifest["candidates"],
            manifest["candidates_kept_for_training"]) == (6, 6, 5, 5)
    assert manifest["hint"] == [{"kind": "skill", "fraction": None, "slots": 6, "passing_samples": 5, "tasks": 3,
                                 "solved_tasks": 3, "restated": 0, "passing_restated": 0}]
    protocol = read(sampling / "train-guided-skill-protocol.json")
    assert protocol["hint"] == {"kind": "skill", "samples_per_hint": 2, "fractions": [],
                                "zero_pass_from": digest(pipe.round1 / "sampling/train-manifest.json"),
                                "skills_sha256": digest(pipe.run / "skills.jsonl")}
    assert Counter(hinted(pipe, pipe.generation_requests(mark))) == Counter({
        (task, SKILL_HINT + "\n\n" + skills[task]): 2 for task in ZERO_PASS})    # 发出去的消息带着教师写的技能
    guided = lines(sampling / "train-guided-skill-graded.jsonl")
    assert Counter((r["task_id"], r["provenance"]["strict_pass"]) for r in guided) == Counter({
        ("h3_zero_001", True): 2, ("h3v_zero_002", True): 2, ("h3v_zero_003", True): 1, ("h3v_zero_003", False): 1})
    for record in guided:
        assert record["messages"] == build_messages(spec_of(pipe, record["task_id"]))   # 训练记录里是标准提示
        assert record["provenance"]["guided"] == {"kind": "skill"}
        assert SKILL not in json.dumps(record, ensure_ascii=False)            # 技能只进采样的提示


def check_datasets(pipe):
    """三组训练数据（run_rft_guided.sh 第 8 步）。D-prefix、D-skill 是第 1 轮的验证组加上引导出的正确答案；
    F-control 是基础记录原样，加上每道引导题一份没通过的采样。三份训练集逐题相同、每题恰有一条；同一道题
    D 组是正确答案、F 组是上一轮真的没通过的采样之一，两边的提示完全一样。"""
    round1, sft = pipe.round1, pipe.sft
    graded_path = round1 / "sampling/train-graded.jsonl"
    for name, tag, key, expected in (("D-prefix", "guided-prefix", "fraction",
                                      {"h3_zero_001": 0.25, "h3v_zero_002": 0.5, "h3v_zero_003": 0.25}),
                                     ("D-skill", "guided-skill", "kind", {task: "skill" for task in ZERO_PASS})):
        prepared = ok(pipe.prepare_group(name, tag))
        assert "重放验证候选: h3_solved_001" in prepared.stdout             # 基础集的记录也重新判一遍
        assert task_ids(sft / name / "train.jsonl") == list(TRAIN)
        assert (sft / name / "dev.jsonl").read_bytes() == (round1 / "sft/verified/dev.jsonl").read_bytes()
        for record in lines(sft / name / "train.jsonl"):
            if record["task_id"] in ZERO_PASS:                              # 引导题：带着提示才采到的正确答案
                provenance = record["provenance"]
                assert provenance["strict_pass"] is True and provenance["revalidated"] is True
                assert provenance["guided"][key] == expected[record["task_id"]]
    built = ok(pipe.build_control(sft / "D-prefix", sft / "D-skill"))
    assert "结果：配对对照 = 基础 1 条 + 引导题 3 条（来自 2 组）；配不上的题 0 道 []" in built.stdout
    control = sft / "F-control"
    assert names(control) == ["dataset-manifest.json", "dev.jsonl", "train.jsonl", "unmatched-tasks.txt"]
    assert (control / "unmatched-tasks.txt").read_text(encoding="utf-8") == ""
    assert (control / "train.jsonl").read_bytes().startswith((round1 / "sft/verified/train.jsonl").read_bytes())
    assert (control / "dev.jsonl").read_bytes() == (round1 / "sft/verified/dev.jsonl").read_bytes()
    sets = {name: {record["task_id"]: record for record in lines(sft / name / "train.jsonl")} for name in ARMS}
    assert [list(records) for records in sets.values()] == [list(TRAIN)] * 3     # 三份训练集逐题相同，每题恰有一条
    pool = {"h3_zero_001": 2, "h3v_zero_002": 1, "h3v_zero_003": 1}             # 每道题写完了、又没通过的采样各有几份
    for task in ZERO_PASS:
        paired = sets["F-control"][task]
        assert (paired["record_type"], paired["provenance"]["strict_pass"], paired["finish_reason"]) == (
            "attempt", False, "stop")
        assert finished_code(paired) and paired["provenance"]["control"] == "failing_random_sample"
        assert paired["selection"] == {"policy": "control_failing_random", "pool": pool[task]}
        failed = {r["target"]["content"] for r in lines(graded_path)
                  if r["task_id"] == task and finished_code(r) and not r["provenance"]["strict_pass"]}
        assert paired["target"]["content"] in failed                          # 答案是上一轮真的没通过的采样之一
        for name in ("D-prefix", "D-skill"):
            assert sets[name][task]["provenance"]["strict_pass"] is True
            assert sets[name][task]["target"] != paired["target"]              # 同一道题：D 组对，F 组不对
            assert sets[name][task]["messages"] == paired["messages"] == build_messages(spec_of(pipe, task))
    stats = read(control / "dataset-manifest.json")
    assert (stats["samples"], stats["tasks"], stats["base_samples"], stats["unmatched"]) == (3, 3, 1, [])
    assert stats["base_dir"] == str(round1 / "sft/verified")
    assert stats["guided_from"] == [str(sft / "D-prefix"), str(sft / "D-skill")]
    assert [(chosen["task_id"], chosen["verdict"]) for chosen in stats["chosen"]] == [
        ("h3_zero_001", "partial"), ("h3v_zero_002", "strict_fail"), ("h3v_zero_003", "partial")]   # 都没通过
    assert {chosen["finish_reason"] for chosen in stats["chosen"]} == {"stop"}


def check_hf_table(pipe):
    """HF 贪心那张表：tier_report.py 读各组的选型目录（run_rft_guided.sh 第 12 步）。训练做不了，选型目录是照
    最小的样子造的：A-verified 和三个引导组都选了适配器，第 1 轮的对照组和基座打平、保留了基座。"""
    round1, run = pipe.round1, pipe.run
    for root, arm, passes in ((round1, "A-verified", (True, True)), (round1, "C-control", (True, False)),
                              (run, "D-prefix", (True, True)), (run, "D-skill", (True, True)),
                              (run, "F-control", (True, True))):
        selection = pipe.select(root, arm, passes)
        assert selection["selected_epoch"] == (0 if arm == "C-control" else 1)
        assert (selection["selected_adapter"] is None) == (arm == "C-control")
    table = ok(script("tier_report.py", round1 / "results/A-verified", round1 / "results/C-control",
                      *(run / "results" / arm for arm in ARMS)))
    # 多个选型目录一起汇总时行名后面带着组名（选型目录的名字）：不然五组的行都叫 "base"、"epoch 1（选中）"，
    # 按行数着看会把组认错。行名照样以 base / epoch 开头，便于 grep
    def row(label, group, cells="1/2 0/0 0/0 1/2"):
        return [*label.split(" "), "·", group, *cells.split(" ")]

    chosen = "2/2 0/0 0/0 2/2"
    assert [line.split() for line in table.stdout.splitlines() if "相对" not in line] == [
        ["模型", "原有题", "难题", "变体", "合计"],
        row("base", "A-verified"), row("epoch 1（选中）", "A-verified", chosen),
        row("base（保留基座）", "C-control"), row("epoch 1", "C-control"),
        row("base", "D-prefix"), row("epoch 1（选中）", "D-prefix", chosen),
        row("base", "D-skill"), row("epoch 1（选中）", "D-skill", chosen),
        row("base", "F-control"), row("epoch 1（选中）", "F-control", chosen)]
    # 变好、变差的题数都相对整张表的第一行（第 1 轮 A-verified 的基座）
    changes = "变好 1 道、变差 0 道（各层变好/变差：原有题 +1/-0、难题 +0/-0、变体 +0/-0）"
    same = "变好 0 道、变差 0 道（各层变好/变差：原有题 +0/-0、难题 +0/-0、变体 +0/-0）"
    first = "base · A-verified"

    def moved(label, change):
        return f"{label} 相对 {first}：{change}"

    assert [line for line in table.stdout.splitlines() if "相对" in line] == [
        moved("epoch 1（选中） · A-verified", changes),
        moved("base（保留基座） · C-control", same), moved("epoch 1 · C-control", same),
        *(line for arm in ("D-prefix", "D-skill", "F-control")
          for line in (moved(f"base · {arm}", same), moved(f"epoch 1（选中） · {arm}", changes)))]


def check_base_evaluation(pipe):
    """基座（run_rft_guided.sh 第 3 步）：dev 上贪心加采样，再评探针。"""
    run = pipe.run
    mark = len(pipe.serving.requests)
    base = ok(pipe.eval_dev("base"))
    assert f"服务 {pipe.serving.url} 上的 {BASE} 是基座。" in base.stdout
    assert "评测 base dev：2 道题（每题 贪心 1 + 采样 2）= 6 个槽位，已有 0 个，待完成 6 个" in base.stdout
    assert "结果：base dev 贪心 0/2、采样通过率 25%（1/4 次尝试）、pass@2 50%" in base.stdout
    assert {r["model"] for r in pipe.generation_requests(mark)} == {BASE}
    probed = ok(pipe.eval_probe("base"))
    assert "结果：base probe 贪心 0/3" in probed.stdout and "自检" not in probed.stdout
    out = run / "vllm/base"
    assert names(out) == ["base-dev-greedy.json", "base-dev-records.jsonl", "base-dev-sampled-k2.json",
                          "base-dev-summary.json", "base-probe-greedy.json", "base-probe-records.jsonl",
                          "base-probe-summary.json"]
    assert evaluated(out, "base", "dev") and evaluated(out, "base", "probe")     # rft_eval_done 会认它们评过了
    assert read(out / "base-dev-summary.json") == {
        "label": "base", "model": BASE, "split": "dev", "tag": "dev",
        "greedy": {"pass": 0, "total": 2, "by_tier": by_tier([0, 1], [0, 1], [0, 0])},
        "sampled": {"samples_per_task": 2, "attempts": 4, "passed_attempts": 1, "pass_at_1": 0.25,
                    "any_pass_rate": 0.5, "by_tier": by_tier([0, 1], [1, 1], [0, 0])},
        "sanity": None}
    # 探针是上一轮零通过的题：难题一道、变体两道，基座一道都做不对
    assert read(out / "base-probe-summary.json") == {
        "label": "base", "model": BASE, "split": "train", "tag": "probe",
        "greedy": {"pass": 0, "total": 3, "by_tier": by_tier([0, 0], [0, 1], [0, 2])},
        "sampled": None, "sanity": None}
    records = lines(out / "base-dev-records.jsonl")
    assert Counter((r["task_id"], r["decoding"], r["repeat_index"]) for r in records) == Counter(
        {(task, "greedy", 0) for task in DEV} | {(task, "sampling", repeat) for task in DEV for repeat in range(2)})
    assert all(r["setup"]["adapter"] is None and r["setup"]["base"] == BASE for r in records)
    greedy_report, sampled_report = read(out / "base-dev-greedy.json"), read(out / "base-dev-sampled-k2.json")
    assert set(greedy_report) == REPORT_KEYS == set(sampled_report)             # 结构同 evaluate.py 的报告
    # 判分是真的：贪心两条都没过；采样里 h3_dev_001 第 0 条过了，隐藏测试也过
    assert [(row["task_id"], row["strict_pass"]) for row in greedy_report["results"]] == [
        ("h3_dev_001", False), ("demo_dev_001", False)]
    passing = sampled_report["results"][0]
    assert (passing["task_id"], passing["strict_pass"], passing["verdict"]) == ("h3_dev_001", True, "strict_pass")
    assert passing["grader_details"]["layers"]["functional"]["private_pass"] and passing["generated_code"] == FIXED
    shown = ok(script("tier_report.py", f"base={out / 'base-dev-greedy.json'}"))      # 当普通报告读得进去
    assert [row.split() for row in shown.stdout.splitlines()] == [
        ["模型", "原有题", "难题", "变体", "合计"], ["base", "0/1", "0/1", "0/0", "0/2"]]


def check_each_arm(pipe):
    """各个组（rft_vllm_eval.sh 第 6 步）：先评探针；列在 EXPECT_REPRODUCE 里的组先自检——适配器和基座在训练过的题上
    各贪心一次、逐题比输出——过了再评 dev。严格通过几道只作参考。"""
    run, sft = pipe.run, pipe.sft
    # D-prefix：适配器生效，训练过的题都复现，输出和基座都不同
    mark = len(pipe.serving.requests)
    probed = ok(pipe.eval_probe("D-prefix", sft / "D-prefix/train.jsonl"))
    assert f"服务 {pipe.serving.url} 上的 D-prefix 是挂在基座 {BASE} 上的适配器。" in probed.stdout
    assert "自检：3 道训练过的题，适配器各贪心评一次" in probed.stdout
    assert "自检：训练过的题严格通过 3/3（仅供参考；训练强度小时复现不了是正常的）" in probed.stdout
    assert "自检：适配器的输出在 3 道题里有 3 道和基座不同（门槛 1）" in probed.stdout
    assert "结果：D-prefix probe 贪心 3/3" in probed.stdout
    # 自检是适配器 3 条、基座 3 条贪心请求（请求里的 model 分别是组名和基座名），探针又是适配器 3 条
    assert Counter((r["model"], r["temperature"]) for r in pipe.generation_requests(mark)) == Counter(
        {("D-prefix", 0): 6, (BASE, 0): 3})
    dev = ok(pipe.eval_dev("D-prefix"))
    assert "结果：D-prefix dev 贪心 1/2、采样通过率 75%（3/4 次尝试）、pass@2 100%" in dev.stdout
    folder = run / "vllm/D-prefix"
    assert names(folder) == ["D-prefix-dev-greedy.json", "D-prefix-dev-records.jsonl",
                             "D-prefix-dev-sampled-k2.json", "D-prefix-dev-summary.json",
                             "D-prefix-probe-greedy.json", "D-prefix-probe-records.jsonl",
                             "D-prefix-probe-sanity.json", "D-prefix-probe-summary.json"]
    assert evaluated(folder, "D-prefix", "dev") and evaluated(folder, "D-prefix", "probe")
    sanity = read(folder / "D-prefix-probe-sanity.json")
    assert (sanity["pass"], sanity["total"], sanity["min"], sanity["task_ids"], sanity["adapter"]) == (
        3, 3, 1, list(ZERO_PASS), "D-prefix")
    assert (sanity["base"], sanity["changed"], sanity["changed_tasks"],
            sanity["compared"], sanity["compared_tasks"]) == (BASE, 3, list(ZERO_PASS), True, 3)
    # 基座在同样这三道题上的记录：字段同 results，一道都没通过
    assert [row["task_id"] for row in sanity["base_results"]] == list(ZERO_PASS)
    assert all(set(row) == set(sanity["results"][0]) and not row["strict_pass"] for row in sanity["base_results"])
    assert read(folder / "D-prefix-probe-summary.json") == {
        "label": "D-prefix", "model": "D-prefix", "split": "train", "tag": "probe",
        "greedy": {"pass": 3, "total": 3, "by_tier": by_tier([0, 0], [1, 1], [2, 2])},
        "sampled": None,
        "sanity": {"pass": 3, "total": 3, "min": 1, "changed": 3, "compared": True, "compared_tasks": 3,
                   "not_compared_reason": None}}
    assert read(folder / "D-prefix-dev-summary.json") == {
        "label": "D-prefix", "model": "D-prefix", "split": "dev", "tag": "dev",
        "greedy": {"pass": 1, "total": 2, "by_tier": by_tier([0, 1], [1, 1], [0, 0])},
        "sampled": {"samples_per_task": 2, "attempts": 4, "passed_attempts": 3, "pass_at_1": 0.75,
                    "any_pass_rate": 1.0, "by_tier": by_tier([1, 1], [1, 1], [0, 0])},
        "sanity": None}
    greedy_report = read(folder / "D-prefix-dev-greedy.json")
    assert (greedy_report["model"], greedy_report["adapter"], greedy_report["split"], greedy_report["task_ids"],
            greedy_report["interpretation"]) == (BASE, "D-prefix", "dev", sorted(DEV), "held_out")
    assert greedy_report["protocol"]["tokenization"] == {"attention_implementation": "vllm",
                                                         "served_model": "D-prefix"}
    assert [(row["task_id"], row["strict_pass"]) for row in greedy_report["results"]] == [
        ("h3_dev_001", True), ("demo_dev_001", False)]

    # D-skill：适配器在服务里没有生效（答得和基座一模一样），输出 0/3 不同，以 4 退出；不写报告和 summary，不往下评 dev
    pipe.model.effective["D-skill"] = False
    mark = len(pipe.serving.requests)
    refused = pipe.eval_probe("D-skill", sft / "D-skill/train.jsonl")
    assert refused.returncode == 4 and "Traceback" not in refused.stderr
    assert refused.stdout.endswith("自检：训练过的题严格通过 0/3（仅供参考；训练强度小时复现不了是正常的）\n"
                                   "自检：适配器的输出在 3 道题里有 0 道和基座不同（门槛 1）\n"
                                   "适配器在这几道题上的输出和基座完全一样，适配器在 vLLM 里没有生效\n")
    assert "结果：" not in refused.stdout
    # 自检是适配器 3 条、基座 3 条；没过就不往下评，探针那 3 条没有发
    assert sorted((task_of(pipe, r["messages"][-1]["content"])["task_id"], r["temperature"], r["model"])
                  for r in pipe.generation_requests(mark)) == sorted(
        (task, 0, model) for model in ("D-skill", BASE) for task in ZERO_PASS)
    folder = run / "vllm/D-skill"
    assert names(folder) == ["D-skill-probe-sanity.json"]
    sanity = read(folder / "D-skill-probe-sanity.json")
    assert (sanity["pass"], sanity["total"], sanity["min"], sanity["task_ids"]) == (0, 3, 1, list(ZERO_PASS))
    assert (sanity["changed"], sanity["changed_tasks"], sanity["compared"],
            sanity["compared_tasks"]) == (0, [], True, 3)
    assert [(row["task_id"], row["strict_pass"]) for row in sanity["results"]] == [(task, False) for task in ZERO_PASS]
    assert [row["raw_response"] for row in sanity["base_results"]] == [row["raw_response"] for row in sanity["results"]]
    middle_dev, middle_probe, _ = summary_sections(ok(pipe.summary(*ARMS, UNTRAINED)).stdout)
    expect(middle_dev, "base", r"0/2\s+25\.0%（1/4 次尝试）\s+pass@2 50\.0%")
    expect(middle_dev, "D-prefix", r"1/2\s+75\.0%（3/4 次尝试）\s+pass@2 100\.0%")
    for label in ("D-skill", "F-control", UNTRAINED):
        expect(middle_dev, label, r"没有结果（没做）")
    assert verdict(middle_dev) == ("结果：dev 贪心 base 0/2、D-prefix 1/2；采样通过率 base 25.0%、D-prefix 75.0%；"
                                   "相对基座 D-prefix +1 道")
    expect(middle_probe, "base", r"全部 0/3")
    expect(middle_probe, "D-prefix", PAIRED_3 + CHECKED)
    # 自检没过的组：eval_vllm.py 只留下 -sanity.json，汇总从它读出和基座比的结果，跟在"作废"后面
    expect(middle_probe, "D-skill", FAILED_SANITY)
    expect(middle_probe, "F-control", VOID)
    expect(middle_probe, UNTRAINED, VOID)                      # 这一组还没评过：只有"作废"，没有自检结果可带
    assert verdict(middle_probe) == (
        "结果：族内探针（贪心；每个组依次是训练过的题、没训练过的同族题，括号里是基座）：基座 0/3；"
        f"D-prefix 3/3（0/3）、无；D-skill 评测作废或没做；F-control 评测作废或没做；{UNTRAINED} 评测作废或没做")

    # 修好了原样重跑：自检每次都重做（它查的是服务现在的样子），过了才往下评
    pipe.model.effective["D-skill"] = True
    mark = len(pipe.serving.requests)
    probed = ok(pipe.eval_probe("D-skill", sft / "D-skill/train.jsonl"))
    assert "自检：训练过的题严格通过 3/3（仅供参考" in probed.stdout
    assert "自检：适配器的输出在 3 道题里有 3 道和基座不同（门槛 1）" in probed.stdout
    assert "结果：D-skill probe 贪心 3/3" in probed.stdout
    # 自检的 6 条重做（适配器 3、基座 3），再加上探针的 3 条
    assert len(pipe.generation_requests(mark)) == 9
    dev = ok(pipe.eval_dev("D-skill"))
    assert "结果：D-skill dev 贪心 2/2、采样通过率 100%（4/4 次尝试）、pass@2 100%" in dev.stdout
    # F-control 不在 EXPECT_REPRODUCE 里：不自检
    probed = ok(pipe.eval_probe("F-control"))
    assert "自检" not in probed.stdout and "结果：F-control probe 贪心 0/3" in probed.stdout
    dev = ok(pipe.eval_dev("F-control"))
    assert "结果：F-control dev 贪心 0/2、采样通过率 25%（1/4 次尝试）、pass@2 50%" in dev.stdout
    for arm in ("D-skill", "F-control"):
        assert evaluated(run / "vllm" / arm, arm, "dev") and evaluated(run / "vllm" / arm, arm, "probe")
    # 探针题里没有这一组训练过的题：没法比较。这不是"输出和基座完全一样"：不作废，照常评完探针，
    # 自检结果和汇总里都写明没能比较、没能确认适配器是否生效
    (sft / UNTRAINED).mkdir()
    for split in ("train", "dev"):
        shutil.copyfile(pipe.round1 / f"sft/verified/{split}.jsonl", sft / UNTRAINED / f"{split}.jsonl")
    mark = len(pipe.serving.requests)
    blind = ok(pipe.eval_probe(UNTRAINED, sft / UNTRAINED / "train.jsonl"))
    assert (f"自检：没能和基座比较（{NO_TRAINED}），没能确认适配器是否生效；不作废，接着往后评\n"
            in blind.stdout)
    assert f"结果：{UNTRAINED} probe 贪心 0/3" in blind.stdout
    assert {r["model"] for r in pipe.generation_requests(mark)} == {UNTRAINED}   # 没有可比较的题：不向基座发请求
    folder = run / "vllm" / UNTRAINED
    assert evaluated(folder, UNTRAINED, "probe")                  # 旧版这时以 4 退出、没有探针报告
    sanity = read(folder / f"{UNTRAINED}-probe-sanity.json")
    assert (sanity["total"], sanity["min"], sanity["compared"], sanity["changed"]) == (0, 1, False, 0)
    assert sanity["not_compared_reason"] == NO_TRAINED and sanity["base_results"] == []
    _, probe, _ = summary_sections(ok(pipe.summary(UNTRAINED)).stdout)
    expect(probe, UNTRAINED, r"训练过的题：没有这样的题、没训练过的同族题 0/3（基座 0/3）；"
               r"自检：没能和基座比较（" + NO_TRAINED + r"），没能确认适配器是否生效")


def check_summary(pipe):
    """汇总（rft_vllm_eval.sh 第 8 步）：两张表、两行结论、怎么读；表上的数是 eval_vllm.py 写出来的那些。"""
    run = pipe.run
    final = ok(pipe.summary(*ARMS, UNTRAINED))
    assert final.stderr == ""
    dev, probe, how = summary_sections(final.stdout)
    assert dev[0] == "── vLLM 评测（bf16 权重，辅助指标；主结果是上面 HF 贪心那张表） ──"
    assert re.fullmatch(r"标签\s+dev 贪心\s+采样通过率\s+pass@k", dev[1])
    expect(dev, "base", r"0/2\s+25\.0%（1/4 次尝试）\s+pass@2 50\.0%")
    expect(dev, "D-prefix", r"1/2\s+75\.0%（3/4 次尝试）\s+pass@2 100\.0%")
    expect(dev, "D-skill", r"2/2\s+100\.0%（4/4 次尝试）\s+pass@2 100\.0%")
    expect(dev, "F-control", r"0/2\s+25\.0%（1/4 次尝试）\s+pass@2 50\.0%")
    expect(dev, UNTRAINED, r"没有结果（没做）")
    assert [line.strip() for line in dev if line.startswith(" ")] == [   # 各层写在下一行
        "原有题 0/1、难题 0/1", "原有题 0/1、难题 1/1", "原有题 1/1、难题 1/1", "原有题 0/1、难题 0/1"]
    assert verdict(dev) == ("结果：dev 贪心 base 0/2、D-prefix 1/2、D-skill 2/2、F-control 0/2；"
                            "采样通过率 base 25.0%、D-prefix 75.0%、D-skill 100.0%、F-control 25.0%；"
                            "相对基座 D-prefix +1 道、D-skill +2 道、F-control 持平")
    assert probe[0] == "── 族内探针（上一轮零通过的题，贪心） ──"
    expect(probe, "base", r"全部 0/3")
    for arm in ("D-prefix", "D-skill"):
        expect(probe, arm, PAIRED_3 + CHECKED)
    expect(probe, "F-control", r"训练过的题 0/3（基座 0/3）、没训练过的同族题：没有这样的题")
    # 没能比较的组（探针题里没有它训练过的题）：这一组照常评了，表里照常有数，后面注明没能确认适配器是否生效
    expect(probe, UNTRAINED, r"训练过的题：没有这样的题、没训练过的同族题 0/3（基座 0/3）；"
               r"自检：没能和基座比较（" + NO_TRAINED + r"），没能确认适配器是否生效")
    assert verdict(probe) == (
        "结果：族内探针（贪心；每个组依次是训练过的题、没训练过的同族题，括号里是基座）：基座 0/3；"
        f"D-prefix 3/3（0/3）、无；D-skill 3/3（0/3）、无；F-control 0/3（0/3）、无；{UNTRAINED} 无、0/3（0/3）")
    assert how == ["── 怎么读 ──", "  - D 组明显高于 F 组才说明难题的正确答案有用；",
                   "  - 采样通过率比贪心灵敏，只作旁证；", '  - 族内探针分清是"训练信号太弱"还是"跨不了族"；',
                   "  - test 这一轮不评。"]
    assert sum(line.startswith("结果：") for line in final.stdout.splitlines()) == 2     # 每张表后面一行，便于 grep
    for arm, greedy_pass, passed in (("D-prefix", 1, 3), ("D-skill", 2, 4), ("F-control", 0, 1)):   # 表一的数
        summary = read(run / "vllm" / arm / f"{arm}-dev-summary.json")
        assert (summary["greedy"]["pass"], summary["sampled"]["passed_attempts"]) == (greedy_pass, passed)
        assert f"{arm} {greedy_pass}/2" in verdict(dev) and f"{arm} {passed / 4:.1%}" in verdict(dev)
    for arm in ARMS:                                   # 表二的数：训练过的题里通过的道数
        report = read(run / "vllm" / arm / f"{arm}-probe-greedy.json")
        trained = set(task_ids(pipe.sft / arm / "train.jsonl"))
        assert Counter(bool(row["strict_pass"]) == (arm != "F-control") for row in report["results"]
                       if row["task_id"] in trained) == {True: 3}


def check_rerun(pipe):
    """原样重跑：有定论的不再请求教师，齐全的不再生成、不再判分、不再连服务。"""
    run, round1 = pipe.run, pipe.round1
    before = (run / "skills.jsonl").read_bytes()
    code, out, err = pipe.write_skills()
    assert (code, err) == (0, "") and "没有要写的题：3 道都已有定论，不发请求。" in out
    assert len(pipe.teacher.requests) == 8 and (run / "skills.jsonl").read_bytes() == before   # 协议里记着它的摘要
    mark = len(pipe.serving.requests)
    again = [pipe.sample("train", round1 / "sampling", "--samples", SAMPLES), pipe.guided_prefix(),
             pipe.guided_skill()]
    for result in again:
        assert "待生成 0 个" in ok(result).stdout and "待判 0 个" in result.stdout
    summaries = {label: (run / "vllm" / label / f"{label}-dev-summary.json").read_bytes()
                 for label in ("base", *ARMS)}
    for label in ("base", *ARMS):
        assert "记录齐全，不用连服务，直接出报告" in ok(pipe.eval_dev(label)).stdout
        assert (run / "vllm" / label / f"{label}-dev-summary.json").read_bytes() == summaries[label]
    assert len(pipe.serving.requests) == mark                         # 这几步一个请求都没发


# ---- 用例 ----


@pytest.mark.parametrize("pipeline", ["plain", "poisoned"], indirect=True)
def test_the_whole_guided_round_runs_on_the_real_scripts_and_the_reports_agree(pipeline, dead_proxy):
    """全新跑一遍：第 1 轮 → 写技能 → 两种引导采样 → 三组训练数据 → HF 表 → vLLM 评测 → 汇总，最后原样重跑。

    跑两遍。第二遍里模型的每个回答都带着 U+2028、U+0085、U+2029：str.splitlines() 把它们当成换行，这样的采样
    以前写进文件就读不回来——槽位被悄悄丢掉、清单照写、退出码 0，答对的题记成"一次都没通过"。两遍的断言
    完全一样：这样的回答不能让任何一个数字变掉。"""
    # 1、2. 题文件和第 1 轮：不引导的采样，再准备验证组
    check_first_round(pipeline, ok(pipeline.first_round()),
                      finished={"h3_zero_001": 2, "h3v_zero_002": 1, "h3v_zero_003": 1},
                      failures={"private_test_failure": 4, "public_test_failure": 1, "incomplete_response": 2})
    # 3. 教师给零通过的题写技能；它读到的是上一轮真的判过分的失败尝试
    skills = check_skills(pipeline)
    check_teacher_sees_the_real_failures(pipeline)
    # 4. 两种引导采样
    check_prefix_sampling(pipeline)
    check_skill_sampling(pipeline, skills)
    # 5. 三组训练数据
    check_datasets(pipeline)
    # 训练做不了：HF 贪心那张表要的选型目录用最小的样子顶上
    check_hf_table(pipeline)
    # 6. vLLM 评测：探针清单是上一轮零通过的题
    assert pipeline.write_probe_ids() == list(ZERO_PASS)
    check_base_evaluation(pipeline)
    check_each_arm(pipeline)
    # 7. 汇总
    check_summary(pipeline)
    # 原样重跑
    check_rerun(pipeline)
    check_no_record_is_cut_by_a_line_reader(pipeline)


def test_a_guided_task_with_no_finished_failure_is_dropped_from_both_arms(pipeline, dead_proxy):
    """上一轮的采样里有一道题一份"写完了、格式完整、又没通过"的答案都没有（全是被截断的）：配对对照配不上它。
    照 run_rft_guided.sh 第 8 步的做法：引导组带着 --drop-tasks-file 重新准备，对照组删掉重新配，三组才逐题
    对得上；探针清单里它还在，汇总里它算"没训练过的同族题"。"""
    pipe, round1, run = pipeline, pipeline.round1, pipeline.run
    sft = run / "sft"
    dropped = "h3v_zero_003"
    kept = [task for task in TRAIN if task != dropped]
    pipe.model.never_finishes.add(dropped)
    graded_path = round1 / "sampling/train-graded.jsonl"

    # 第 1 轮：这道题的两份采样都没写完（代码能解析，但没判过分）
    check_first_round(pipe, ok(pipe.first_round()), finished={"h3_zero_001": 2, "h3v_zero_002": 1, dropped: 0},
                      failures={"private_test_failure": 3, "public_test_failure": 1, "incomplete_response": 3})
    rows = [record for record in lines(graded_path) if record["task_id"] == dropped]
    assert [(r["finish_reason"], r["provenance"]["verdict"]) for r in rows] == [("length", "incomplete_response")] * 2
    for record in rows:
        assert ast.parse(record["target"]["content"]).body and not finished_code(record)

    # 第 2 轮：两种引导都把它解出来了，两组的训练集里都有它。教师看到的是它两份没写完的采样（代码相同，只给一份）
    skills = check_skills(pipe)
    (shown,) = diagnoses_for(pipe, dropped)
    assert shown.count("### 失败尝试 ") == 1 and "失败类型：incomplete_response（输出没写完）" in shown
    check_prefix_sampling(pipe)
    check_skill_sampling(pipe, skills)
    for name, tag in (("D-prefix", "guided-prefix"), ("D-skill", "guided-skill")):
        ok(pipe.prepare_group(name, tag))
        assert task_ids(sft / name / "train.jsonl") == list(TRAIN)

    # 配对对照：它配不上，题号记进 unmatched-tasks.txt；另外两道照常配上
    built = ok(pipe.build_control(sft / "D-prefix", sft / "D-skill"))
    assert f"结果：配对对照 = 基础 1 条 + 引导题 2 条（来自 2 组）；配不上的题 1 道 ['{dropped}']" in built.stdout
    unmatched = sft / "F-control/unmatched-tasks.txt"
    assert str(unmatched) in built.stdout and "--drop-tasks-file" in built.stdout
    assert unmatched.read_text(encoding="utf-8") == f"{dropped}\n"
    assert task_ids(sft / "F-control/train.jsonl") == kept
    stats = read(sft / "F-control/dataset-manifest.json")
    assert (stats["samples"], stats["tasks"], stats["unmatched"]) == (2, 3, [dropped])
    assert [chosen["task_id"] for chosen in stats["chosen"]] == ["h3_zero_001", "h3v_zero_002"]
    # 这时两组的题目对不上：各 D 组合起来比对照组多一道
    together = set(task_ids(sft / "D-prefix/train.jsonl")) | set(task_ids(sft / "D-skill/train.jsonl"))
    assert together - set(task_ids(sft / "F-control/train.jsonl")) == {dropped}

    # 把它从各个引导组里也去掉：脚本把去掉之前的目录挪到一边，往原来的名字里重新准备
    for name, tag in (("D-prefix", "guided-prefix"), ("D-skill", "guided-skill")):
        moved = sft / f"{name}.unmatched-stamp"
        shutil.move(str(sft / name), str(moved))
        prepared = ok(pipe.prepare_group(name, tag, "--drop-tasks-file", unmatched))
        assert "按 --drop-tasks-file 去掉 1 条记录（1 道题）" in prepared.stdout
        assert f"重放验证候选: {dropped}" not in prepared.stdout      # 在重放验证之前就挡掉，不占判分的时间
        assert task_ids(sft / name / "train.jsonl") == kept
        assert (sft / name / "dev.jsonl").read_bytes() == (moved / "dev.jsonl").read_bytes()
        manifest = read(sft / name / "dataset-manifest.json")
        assert manifest["dropped_task_ids"] == [dropped]
        assert manifest["splits"]["train"]["missing_task_ids"] == [dropped]
        assert "dropped_task_ids" not in read(moved / "dataset-manifest.json")
    # 对照组是按去掉之前的引导组配的：删掉，按新的引导组重新配
    shutil.rmtree(sft / "F-control")
    rebuilt = ok(pipe.build_control(sft / "D-prefix", sft / "D-skill"))
    assert "结果：配对对照 = 基础 1 条 + 引导题 2 条（来自 2 组）；配不上的题 0 道 []" in rebuilt.stdout
    assert unmatched.read_text(encoding="utf-8") == ""                  # 没有配不上的题时是空文件
    assert task_ids(sft / "F-control/train.jsonl") == kept
    paired = [set(task_ids(sft / name / "train.jsonl")) for name in ARMS]
    assert paired[0] == paired[1] == paired[2] == set(kept)            # data_paired：三组逐题相同
    for record in lines(sft / "F-control/train.jsonl"):
        if record["task_id"] != TRAIN[0]:
            assert record["provenance"]["strict_pass"] is False and finished_code(record)

    # 探针清单还是第 1 轮那三道（去掉的那道也在里面）：对各个组来说它是"没训练过的同族题"
    assert pipe.write_probe_ids() == list(ZERO_PASS)
    ok(pipe.eval_probe("base"))
    # 自检只能用探针题里这一组训练过的题（两道）。门槛就是默认的 1，不按题数改：旧判据（至少复现 3 道）下
    # 适配器两道全复现也会被作废；现在输出和基座不同就过，严格通过几道只作参考
    probed = ok(pipe.eval_probe("D-prefix", sft / "D-prefix/train.jsonl"))
    assert "自检：训练过的题严格通过 2/2（仅供参考" in probed.stdout
    assert "结果：D-prefix probe 贪心 2/3" in probed.stdout
    sanity = read(run / "vllm/D-prefix/D-prefix-probe-sanity.json")
    assert (sanity["min"], sanity["total"], sanity["changed"], sanity["compared_tasks"]) == (1, 2, 2, 2)
    ok(pipe.eval_probe("D-skill", sft / "D-skill/train.jsonl"))
    ok(pipe.eval_probe("F-control"))
    _, probe, _ = summary_sections(ok(pipe.summary(*ARMS)).stdout)
    expect(probe, "base", r"全部 0/3")
    for arm in ("D-prefix", "D-skill"):
        expect(probe, arm, r"训练过的题 2/2（基座 0/2）、没训练过的同族题 0/1（基座 0/1）；"
                   r"自检：适配器的输出在 2 道题里有 2 道和基座不同（门槛 1）；训练过的题严格通过 2/2（仅供参考）")
    expect(probe, "F-control", r"训练过的题 0/2（基座 0/2）、没训练过的同族题 0/1（基座 0/1）")
    assert verdict(probe) == (
        "结果：族内探针（贪心；每个组依次是训练过的题、没训练过的同族题，括号里是基座）：基座 0/3；"
        "D-prefix 2/2（0/2）、0/1（0/1）；D-skill 2/2（0/2）、0/1（0/1）；F-control 0/2（0/2）、0/1（0/1）")


def check_no_record_is_cut_by_a_line_reader(pipe):
    """这条链写出的每一份 JSONL 都是一条记录一行：按 "\n" 和按 splitlines() 读出来的行数一样，文件里没有原样的
    U+2028 这类字符。带着它们的回答走完整条链也照样如此（pipe.poisoned）：按行切的脚本——训练入口、编排脚本里的
    片段——才切不开它。"""
    written = sorted(pipe.round1.rglob("*.jsonl")) + sorted(pipe.run.rglob("*.jsonl"))
    assert {path.name for path in written} >= {
        "train-raw.jsonl", "train-graded.jsonl", "train-candidates.jsonl", "train-guided-prefix-raw.jsonl",
        "train-guided-skill-graded.jsonl", "skills.jsonl", "rejected-samples.jsonl", "D-prefix-dev-records.jsonl"}
    for path in written:
        raw = path.read_bytes().decode("utf-8")
        assert not set(SEPARATORS) & set(raw), path.relative_to(pipe.tmp)
        assert raw.count("\n") == len(raw.splitlines()) == len(lines(path)), path.relative_to(pipe.tmp)
    if pipe.poisoned:
        # 回答里确实带着它们：采样的原始输出读回来还是原来的字符，而采样的槽数一个不少（上面的断言都对过）
        first = lines(pipe.round1 / "sampling/train-raw.jsonl")
        assert len(first) == 8 and all(record["text"].startswith(f"说明：{SEPARATORS}。") for record in first)


def test_the_summary_names_the_failed_sanity_check_next_to_the_void_group(pipeline, dead_proxy):
    """自检没过（适配器的输出和基座完全一样）的组：eval_vllm.py 只写 <标签>-probe-sanity.json 就以 4 退出，没有
    summary 和报告。汇总要从那个文件读出自检结果，写在这一组的后面——只写"评测作废或没做"看不出是适配器没生效，
    还是这一组根本没评。"""
    pipe, run = pipeline, pipeline.run
    training = run / "sft/D-prefix/train.jsonl"                  # 这一组训练过全部三道探针题
    training.parent.mkdir(parents=True)
    training.write_text("".join(json.dumps({"task_id": task}) + "\n" for task in ZERO_PASS), encoding="utf-8")
    (run / "probe-ids.txt").write_text("".join(f"{task}\n" for task in ZERO_PASS), encoding="utf-8")
    pipe.model.effective["D-prefix"] = False                     # 适配器在服务里没有生效
    refused = pipe.eval_probe("D-prefix", training)
    assert refused.returncode == 4 and "自检：适配器的输出在 3 道题里有 0 道和基座不同（门槛 1）" in refused.stdout
    assert names(run / "vllm/D-prefix") == ["D-prefix-probe-sanity.json"]     # 自检的结果只写在这里，没有 summary
    assert read(run / "vllm/D-prefix/D-prefix-probe-sanity.json")["min"] == 1
    _, probe, _ = summary_sections(ok(pipe.summary("D-prefix")).stdout)
    expect(probe, "D-prefix", FAILED_SANITY)
    assert verdict(probe) == "结果：族内探针没有可用的结果（没做或没评完）"   # 结论行照旧简短，原因在表里


def test_a_sampling_file_cut_mid_line_is_repaired_on_a_rerun(pipeline, dead_proxy):
    """采样文件的最后一行只有半截（上一次运行写到一半被打断）：原样重跑时那一行去掉、那个槽位重做，新记录接在完整的
    行后面。以前 raw 的半行让那个槽位读不回来却不重做，清单里就少一份；graded 的半行让新记录粘在它后面，
    收尾时 KeyError（退出码 1）。"""
    pipe, sampling = pipeline, pipeline.sampling
    check_first_round(pipe, ok(pipe.first_round()),
                      finished={"h3_zero_001": 2, "h3v_zero_002": 1, "h3v_zero_003": 1},
                      failures={"private_test_failure": 4, "public_test_failure": 1, "incomplete_response": 2})
    skills = check_skills(pipe)
    check_prefix_sampling(pipe)
    manifest_path = sampling / "train-guided-prefix-manifest.json"
    whole = read(manifest_path)
    assert (whole["slots"], whole["raw_records"], whole["graded_records"]) == (18, 18, 18)
    mark = len(pipe.serving.requests)
    cut_last_line(sampling / "train-guided-prefix-raw.jsonl")
    repaired = ok(pipe.guided_prefix())
    assert "待生成 1 个" in repaired.stdout and "待判 0 个" in repaired.stdout
    assert "train-guided-prefix-raw.jsonl 的最后一行没写完" in repaired.stderr
    assert "Traceback" not in repaired.stderr and "跳过没写完的一行" not in repaired.stderr
    sent = pipe.generation_requests(mark)
    assert len(sent) == 1                                       # 只重新生成被截掉的那一个槽位
    assert task_of(pipe, sent[0]["messages"][-1]["content"])["task_id"] in ZERO_PASS
    cut_last_line(sampling / "train-guided-prefix-graded.jsonl")
    regraded = ok(pipe.guided_prefix())
    assert "待生成 0 个" in regraded.stdout and "待判 1 个" in regraded.stdout
    assert "Traceback" not in regraded.stderr and "跳过没写完的一行" not in regraded.stderr
    for name in ("train-guided-prefix-raw.jsonl", "train-guided-prefix-graded.jsonl",
                 "train-guided-prefix-candidates.jsonl"):
        text = (sampling / name).read_text(encoding="utf-8")
        assert text.count("\n") == len(lines(sampling / name))   # 每一行都是完整的记录，没有半行粘着新记录
    assert (sampling / "train-guided-prefix-raw.jsonl").read_text(encoding="utf-8").count("\n") == 18
    assert (sampling / "train-guided-prefix-graded.jsonl").read_text(encoding="utf-8").count("\n") == 18
    assert read(manifest_path) == whole                           # 清单和没截过的时候一模一样
    again = ok(pipe.guided_prefix())                              # 再跑：什么都不缺
    assert "待生成 0 个" in again.stdout and "待判 0 个" in again.stdout and read(manifest_path) == whole
    check_skill_sampling(pipe, skills)


@pytest.mark.skipif(not shutil.which("bash") or not Path("/proc/self/stat").exists(), reason="needs Bash and /proc")
def test_the_real_selection_directories_feed_the_vllm_evaluation_script(pipeline, dead_proxy):
    """run_rft_guided.sh 第 13 步：真的选型目录（choose_checkpoint 算出来的，适配器目录里有 adapter_config.json）交给
    真的 scripts/rft_vllm_eval.sh。它读选型结果，带着三组的 LoRA 启动（假的）vLLM，逐组先探针后 dev，最后打印汇总。
    第一次 D-skill 的适配器在服务里没有生效（输出和基座完全一样，0/3 不同）：这一组作废，汇总里写明原因；
    修好以后原样重跑只补它。"""
    pipe, run = pipeline, pipeline.run
    check_first_round(pipe, ok(pipe.first_round()),
                      finished={"h3_zero_001": 2, "h3v_zero_002": 1, "h3v_zero_003": 1},
                      failures={"private_test_failure": 4, "public_test_failure": 1, "incomplete_response": 2})
    skills = check_skills(pipe)
    check_prefix_sampling(pipe)
    check_skill_sampling(pipe, skills)
    check_datasets(pipe)
    check_hf_table(pipe)
    assert pipe.write_probe_ids() == list(ZERO_PASS)

    pipe.model.effective["D-skill"] = False                       # D-skill 的适配器在服务里没有生效
    first = pipe.rate(*ARMS)
    assert first.returncode == 0, first.stdout[-3000:] + first.stderr[-3000:]
    assert re.findall(r"^===== (.+?) =====$", first.stdout, re.M) == [
        "1/8 读取各组的选型结果", "2/8 确定要评的对象", "3/8 检查显卡", "4/8 启动 vLLM",
        "5/8 评基座：INCLUDE_BASE 不是 1，跳过", "6/8 D-prefix：先探针后 dev", "6/8 D-skill：先探针后 dev",
        "6/8 F-control：先探针后 dev", "7/8 关闭 vLLM", "8/8 汇总", "全部结束"]
    assert ("D-skill：适配器在这几道题上的输出和基座完全一样，适配器在 vLLM 里没有生效，这组的 vLLM 评测作废"
            in first.stdout)
    assert f"结果：vLLM 评测做完了；作废的组：D-skill（适配器的输出和基座完全一样）；报告在 {run}/vllm" in first.stdout
    # vLLM 按选型结果里记的基座启动，带着三组的 LoRA：名字是组名，路径是适配器的绝对路径
    port = pipe.serving.url.rsplit(":", 1)[1]
    assert pipe.vllm_starts() == [
        f"serve {BASE} --revision rev-1 --language-model-only --max-model-len 8192 --port {port} "
        "--enable-lora --max-lora-rank 64 --lora-modules "
        + " ".join(f"{arm}={run}/checkpoints/{arm}/epoch-001/adapter" for arm in ARMS)]
    # 汇总是真的 vllm_summary.py 读上面写出的产物：自检没过的那一组带着原因（读的是 -sanity.json）
    summary = vllm_summary.report(run, ARMS, run / "probe-ids.txt")
    assert summary in first.stdout and "── 族内探针（上一轮零通过的题，贪心） ──" in summary
    _, probe, _ = summary_sections(summary)
    expect(probe, "base", r"没有探针报告（没评基座或没评完），下面各行的括号里不写基座的通过数")
    expect(probe, "D-prefix", r"训练过的题 3/3、没训练过的同族题：没有这样的题" + CHECKED)
    expect(probe, "D-skill", FAILED_SANITY)
    expect(probe, "F-control", r"训练过的题 0/3、没训练过的同族题：没有这样的题")
    assert names(run / "vllm/D-skill") == ["D-skill-probe-sanity.json"]      # 作废的组没有评 dev
    for arm in ("D-prefix", "F-control"):                                   # 别的组照常评完了探针和 dev
        assert evaluated(run / "vllm" / arm, arm, "probe") and evaluated(run / "vllm" / arm, arm, "dev")

    pipe.model.effective["D-skill"] = True                         # 修好了原样重跑：只补这一组
    mark = len(pipe.serving.requests)
    second = pipe.rate(*ARMS)
    assert second.returncode == 0, second.stdout[-3000:] + second.stderr[-3000:]
    assert "===== 6/8 D-skill：先探针后 dev =====" in second.stdout
    assert "D-prefix：已经评过，跳过" in second.stdout and "F-control：已经评过，跳过" in second.stdout
    assert f"结果：vLLM 评测做完了；报告在 {run}/vllm" in second.stdout and "作废" not in second.stdout
    assert re.findall(r"^===== (.+?) =====$", second.stdout, re.M) == [
        "1/8 读取各组的选型结果", "2/8 确定要评的对象", "3/8 检查显卡", "4/8 启动 vLLM",
        "5/8 评基座：INCLUDE_BASE 不是 1，跳过", "6/8 D-prefix：已经评过，跳过", "6/8 D-skill：先探针后 dev",
        "6/8 F-control：已经评过，跳过", "7/8 关闭 vLLM", "8/8 汇总", "全部结束"]
    assert len(pipe.vllm_starts()) == 2 and pipe.vllm_starts()[1] == pipe.vllm_starts()[0]
    sent = pipe.generation_requests(mark)
    # 自检 6（适配器 3、基座 3）、探针 3、dev 2×3；基座那 3 条的 model 是基座名，其余是组名
    assert Counter(request["model"] for request in sent) == Counter({BASE: 3, "D-skill": 12}) and len(sent) == 15
    assert vllm_summary.report(run, ARMS, run / "probe-ids.txt") in second.stdout
    _, probe, _ = summary_sections(vllm_summary.report(run, ARMS, run / "probe-ids.txt"))
    for arm in ("D-prefix", "D-skill"):
        expect(probe, arm, r"训练过的题 3/3、没训练过的同族题：没有这样的题" + CHECKED)
    for stem in ("probe", "dev"):
        assert evaluated(run / "vllm/D-skill", "D-skill", stem)
