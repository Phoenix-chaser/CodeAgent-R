# -*- coding: utf-8 -*-
"""教师写技能：标识符检查、提示内容、重写轮次、挑选失败尝试、教师接口的重试、命令行的续跑和退出码。

教师是注入的假对象（假的请求函数、假的客户端；子进程里用一份假的 openai SDK 顶替），判分是真的。

_shared_teacher.py（所有用教师接口的脚本共用的那一部分）也在这里测：429、5xx 长时间重试，进程内的节流，
TEACHER_EXTRA_JSON。重试和计时的用例把 _shared_teacher 里的 time 换成假时钟（Clock），不用真等；节流的用例
用真线程、很短的间隔。
"""
import difflib
import json
import os
import re
import subprocess
import datetime
import email.utils
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import _shared_teacher as shared
import sample_rft
import write_skills
from agentlab.benchmark.builder import build_all
from agentlab.benchmark.protocol import build_messages
from agentlab.sampling import hints

BUGGY = "def clamp(values, low, high):\n    return list(values)\n"


@pytest.fixture(autouse=True)
def requests_follow_each_other_at_once(monkeypatch):
    """这些用例看的不是节流：两次请求之间不隔（默认隔 1 秒，真等的话每个用例要多等好几秒），进程里共用的那一份
    节流器也不限在飞的个数。脚本在 main 里按参数改它，所以每个用例结束后都把它恢复成原来的设置。专门测节流的
    两个用例自己设自己的。"""
    monkeypatch.setenv("TEACHER_MIN_INTERVAL", "0")
    monkeypatch.setattr(shared.THROTTLE, "max_in_flight", 64)
    monkeypatch.setattr(shared.THROTTLE, "min_interval", 0.0)
    monkeypatch.setattr(shared.THROTTLE, "_last_start", None)      # 同理：上一次请求的时刻不留到下一个用例
FIXED = '''"""Clamp every value."""

DEFAULT_HIGH = 10

if DEFAULT_HIGH > 0:
    LIMITS = {DEFAULT_HIGH}


def clamp(values, low, high, *, ceiling=DEFAULT_HIGH):
    if high > ceiling:
        raise ValueError("ceiling exceeded")
    return [min(max(item, low), high) for item in values]


class Range:
    def __init__(self, low, high):
        self.low, self.high = low, high

    def clip(self, items, strict=False):
        seen = self.high
        return clamp(items, self.low, self.high)
'''
WRONG = "def clamp(values, low, high):\n    return tuple(values)\n"        # 公开测试就过不了
BROKEN = "def clamp(values, low, high):\n    return list(values) + [\n"     # 解析不了
HIDDEN = ["test_hidden", "test_hidden_ceiling", "test_hidden_keyword", "test_hidden_range"]
# 判分器（grader.parse_junit）记用例的写法，以及 write_skills 给教师看的写法
HIDDEN_IDS = [f"tests_private.test_hidden::{name}" for name in HIDDEN]
HIDDEN_SHOWN = [f"test_hidden.py::{name}" for name in HIDDEN]
WHOLE_FILE = "::tests_private.test_hidden"        # 整个隐藏测试文件导入失败时只有这一条
CLEAN = {"applicability": "函数要把每个数限制在一个范围内时。", "principles": "上下两个边界要分别处理。",
         "pitfalls": "原样返回输入，或只处理了一侧边界。", "approach": "先列出越界的情形，逐个处理，再用边界值核对。"}
LEAKY = {**CLEAN, "pitfalls": "把 values 原样返回。"}                  # 带着参考修复里的名字：静态检查能拦
SPECIFIC = {**CLEAN, "approach": "在返回之前，先判断上界是否超过预设的上限，超过就抛出带说明的异常。"}   # 只能靠自判
JUDGE = "重建参考修复"             # 自判的系统提示里有、写技能的里没有：假教师靠它分辨两种请求

FAKE_SDK = '''"""假的 openai SDK（测试用）：诊断回复从 FAKE_TEACHER_SCRIPT（题号 → 回复列表）里按顺序取；
自判一律说不能重建。"""
import json
import os
from types import SimpleNamespace


class OpenAI:
    def __init__(self, **kwargs):
        self.note({"client": kwargs})
        with open(os.environ["FAKE_TEACHER_SCRIPT"], encoding="utf-8") as handle:
            self.scripts = json.load(handle)                  # 读一次：按顺序取过的回复不会再来
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def note(self, entry):
        with open(os.environ["FAKE_TEACHER_LOG"], "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\\n")

    def create(self, model, messages, max_tokens, temperature, extra_body=None):
        # 和真的 SDK 一样只认这几个参数：接口特有的参数要从 extra_body 进来（SDK 会把它合并进请求体）。
        # 多出来的关键字参数，真的 SDK 会报 unexpected keyword argument；这里让 Python 自己报同样的 TypeError
        self.note({"model": model, "messages": messages, "max_tokens": max_tokens,
                   "temperature": temperature, **(extra_body or {})})
        if "重建参考修复" in messages[0]["content"]:
            text = json.dumps({"reconstructable": False, "reason": "只是指出了方向"})
        else:
            task_id = next(name for name in self.scripts if f"文件 {name}.py" in messages[1]["content"])
            script = self.scripts[task_id]
            text = script.pop(0) if len(script) > 1 else script[0]
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50)
        return SimpleNamespace(usage=usage, choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content=text))])
'''


def task(task_id, split="train"):
    """一道能判分的小题：截断函数漏了截断。有缺陷的版本公开测试过、隐藏测试挂。"""
    module = f"{task_id}.py"
    return {
        "task_id": task_id, "source_group": f"group-{task_id}", "split": split,
        "bug_family": "clamping", "difficulty": "L1",
        "instruction": "实现 clamp(values, low, high)：把每个数限制在 [low, high] 内，返回新列表。",
        "module": module, "buggy": BUGGY, "fixed": FIXED,
        "public_test": f"from {module[:-3]} import clamp\n\n"
                       "def test_public():\n    assert clamp([1, 5], 0, 10) == [1, 5]\n",
        "private_tests": {"test_hidden.py": (
            f"from {module[:-3]} import Range, clamp\n\n\n"
            "def test_hidden():\n    assert clamp([-3, 3, 13], 0, 10) == [0, 3, 10]\n\n\n"
            "def test_hidden_keyword():\n    assert clamp([5], 0, 40, ceiling=50) == [5]\n\n\n"
            "def test_hidden_range():\n    assert Range(0, 4).clip([-1, 2, 9]) == [0, 2, 4]\n\n\n"
            "def test_hidden_ceiling():\n    try:\n        clamp([1], 0, 99)\n    except ValueError:\n        return\n"
            "    raise AssertionError('expected ValueError')\n")},
    }


def render(parts):
    return write_skills.render_skill(parts)


def skill(parts, wrap="plain"):
    text = json.dumps(parts, ensure_ascii=False)
    return {"plain": text, "fence": f"```json\n{text}\n```", "prose": f"好的，技能如下：\n{text}\n希望有帮助。"}[wrap]


def verdict(reconstructable, reason="只是指出了方向"):
    return json.dumps({"reconstructable": reconstructable, "reason": reason}, ensure_ascii=False)


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, records):
    Path(path).write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                          encoding="utf-8")


def write_manifest(path, zero_pass):
    Path(path).write_text(json.dumps({"zero_pass_tasks": zero_pass}), encoding="utf-8")


def graded(code, kind, failing, strict=False, missing=()):
    """一条 graded 记录（同 sample_rft.grade_sample 的字段）：代码、失败类型、没通过和没能执行的隐藏测试。"""
    return {"task_id": "h3_demo_001", "target": {"role": "assistant", "content": code},
            "grader_details": {"layers": {"functional": {"private": {
                "non_passing_tests": failing, "missing_tests": missing}}}},
            "provenance": {"strict_pass": strict, "failure_type": kind}}


def hidden_tests(record):
    return record["grader_details"]["layers"]["functional"]["private"]["non_passing_tests"]


def next_script(script, fallback):
    """按顺序取下一条回复；用完后重复最后一条（模拟教师每次都这么回答）。"""
    return script.pop(0) if len(script) > 1 else (script[0] if script else fallback)


def response(text, usage=(100, 50)):
    return SimpleNamespace(
        choices=[] if text is None else [SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content=text))],
        usage=None if usage is None else SimpleNamespace(prompt_tokens=usage[0], completion_tokens=usage[1]))


class FakeAsk:
    """注入给 write_skill 的请求函数：记下每次请求，按顺序给出诊断回复和自判回复。"""

    def __init__(self, diagnoses, judges=()):
        self.diagnoses, self.judges, self.calls = list(diagnoses), list(judges), []

    def __call__(self, messages, max_tokens, temperature):
        self.calls.append((messages, max_tokens, temperature))
        script = self.judges if JUDGE in messages[0]["content"] else self.diagnoses
        text = next_script(script, verdict(False) if script is self.judges else "")
        return {"text": text, "finish_reason": "stop", "usage": {"input_tokens": 120, "output_tokens": 40}}


class ScriptedTeacher:
    """假的教师客户端：chat.completions.create 按题、按请求种类给出预设的回复，记下每次请求。"""

    def __init__(self, specs):
        self.specs, self.requests = specs, []
        self.diagnoses, self.judges = defaultdict(list), defaultdict(list)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        messages = kwargs["messages"]
        self.requests.append(kwargs)
        spec = next(item for item in self.specs if f"文件 {item['module']}" in messages[1]["content"])
        if JUDGE in messages[0]["content"]:                  # 没有预设的自判一律说"不能重建"
            return response(next_script(self.judges[spec["task_id"]], verdict(False)))
        return response(next_script(self.diagnoses[spec["task_id"]], skill(CLEAN)))


class Flaky:
    """假的客户端：依次抛出 errors 里的错误，用完后回答 text；接口不报用量。"""

    def __init__(self, errors, text="ok"):
        self.errors, self.text, self.calls = list(errors), text, 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return response(self.text, usage=None)




def test_identifiers_come_from_the_reference_the_tests_and_the_file_names():
    names = write_skills.code_identifiers(task("h3_demo_001"))
    assert names == {
        "clamp", "clip", "__init__", "Range",                                        # 函数名、类名
        "values", "high", "ceiling", "items", "strict",                              # 函数参数名
        "DEFAULT_HIGH", "LIMITS",                                                    # 模块级赋值名（含顶层 if 里的）
        "Clamp every value.", "ceiling exceeded",                                    # 长度不小于 4 的字符串字面量
        "test_public", *HIDDEN,                                                      # 公开测试和隐藏测试里的测试函数名
        "h3_demo_001", "test_hidden",                                                # 主文件名、隐藏测试的文件名
    }
    assert not {"low", "item", "seen", "self"} & names       # 太短的名字、self、函数里的局部名不算

    # 解析不了的源码跳过，其余的照常
    broken = {**task("h3_b_001"), "public_test": "def broken(:", "private_tests": {"test_x.py": "not python ("}}
    assert write_skills.code_identifiers(broken) == (names - {"h3_demo_001", "test_public", *HIDDEN}) | {
        "h3_b_001", "test_x"}


def test_static_leaks_flag_task_specific_names_and_spare_ordinary_text():
    names = write_skills.code_identifiers(task("h3_demo_001"))
    assert write_skills.static_leaks("先比较上下界，再逐个截断，最后用边界值核对。", names) == []
    assert write_skills.static_leaks("检查 low，再看 high", names) == ["high"]           # low 太短，不拦
    assert write_skills.static_leaks("调用clamp，把values逐个处理", names) == ["clamp", "values"]   # 紧贴汉字也算
    assert write_skills.static_leaks("越界时抛出 ceiling exceeded", names) == ["ceiling", "ceiling exceeded"]
    assert write_skills.static_leaks("目标是让 test_hidden 通过", names) == ["test_hidden"]
    assert write_skills.static_leaks("这道题在 h3_demo_001 里", names) == ["h3_demo_001"]
    assert write_skills.static_leaks("任何文字", set()) == []
    # 整词匹配：higher、strictly、arranged 只是包含了某个名字的一段字母，不算泄露
    assert write_skills.static_leaks("higher、strictly、arranged 都只是包含了一段字母", names) == []


def test_static_leaks_do_not_mistake_a_longer_english_word_for_a_name():
    names = write_skills.code_identifiers({**task("h3_sync_001"), "fixed": "def poll(self, sync=True, cond=None):\n"
                                                                        "    return sync, cond\n"})
    assert names == {"sync", "cond", "poll", "test_public", *HIDDEN, "h3_sync_001", "test_hidden"}
    assert write_skills.static_leaks("async 操作完成前可能被别的协程改写（race condition）。", names) == []
    assert write_skills.static_leaks("先 sync，再检查 cond", names) == ["cond", "sync"]
    assert write_skills.static_leaks('状态存到"data.tmp"里', {".tmp"}) == [".tmp"]        # 以点开头的字面量
    assert write_skills.static_leaks("data_template", {".tmp"}) == []


def test_render_skill_joins_four_titled_sections_and_refuses_an_empty_one():
    assert render(CLEAN) == ("【适用条件】函数要把每个数限制在一个范围内时。\n"
                             "【要点】上下两个边界要分别处理。\n"
                             "【常见错误】原样返回输入，或只处理了一侧边界。\n"
                             "【推荐做法】先列出越界的情形，逐个处理，再用边界值核对。")
    for bad in ({"principles": "  "}, {"approach": ""}, {"pitfalls": None}):
        with pytest.raises(ValueError, match="空的"):
            write_skills.render_skill({**CLEAN, **bad})
    with pytest.raises(ValueError, match="要点（principles）"):
        write_skills.render_skill({key: value for key, value in CLEAN.items() if key != "principles"})


@pytest.mark.parametrize("text", [
    skill(CLEAN),
    skill(CLEAN, "fence"),
    skill(CLEAN, "prose"),
    "格式是 {键: 值}。这是技能：\n" + skill(CLEAN) + "\n另附一个说明：" + json.dumps({"note": "x"}),
    json.dumps([CLEAN]),                                         # 对象包在数组里也取得到
])
def test_parse_skill_json_takes_the_first_object_even_inside_a_fence_or_prose(text):
    assert write_skills.parse_skill_json(text) == CLEAN


def test_parse_skill_json_accepts_a_line_break_inside_a_string():
    reply = '{"applicability": "第一行\n第二行", "principles": "b", "pitfalls": "c", "approach": "d"}'
    assert write_skills.parse_skill_json(reply)["applicability"] == "第一行\n第二行"


def laid_out(parts, brk="\u2028"):
    """教师拿 brk 当排版换行排出来的 JSON 对象：花括号后面、键与键之间都是它（字符串里面没有）。"""
    return "{" + ",".join(f'{brk}"{key}": {json.dumps(value, ensure_ascii=False)}' for key, value in parts.items()) \
        + brk + "}"


@pytest.mark.parametrize("brk", ["\u2028", "\u2029", "\u0085"], ids=["U+2028", "U+2029", "U+0085"])
def test_first_json_object_takes_layout_breaks_outside_strings_as_whitespace(brk):
    """教师把这三个字符当排版换行用在 JSON 字符串的外面。JSON 只把空格、制表符和 \\n、\\r 当空白，原样解不出来——
    以前写技能和自判两步的回复都解不出，这道题白白记成 error。原样解不出时，把字符串外面的它们换成换行再试一次。"""
    assert json.loads(laid_out(CLEAN, brk).replace(brk, "")) == CLEAN     # 对照：去掉它们就是普通的 JSON
    reply = laid_out(CLEAN, brk)
    assert write_skills.first_json_object(reply) == CLEAN
    assert write_skills.parse_skill_json(reply) == CLEAN
    assert write_skills.first_json_object(f"```json\n{reply}\n```") == CLEAN              # 围栏里、紧贴着花括号也行
    # 三种混着用，中间夹着普通的空白
    mixed = "{\u2028" + '\u0085"applicability"\u2029: "a",\n  "principles": "b"\t\u2028}'
    assert write_skills.first_json_object(mixed) == {"applicability": "a", "principles": "b"}
    assert write_skills.parse_judge_json(laid_out({"reconstructable": True, "reason": "照着就能写"}, brk)) == (
        True, "照着就能写")


def test_layout_break_retry_changes_only_what_is_outside_strings():
    """只动字符串外面的。字符串里面的三个字符照旧：技能的各段取出来以后换成换行（plain_breaks），自判的理由原样
    留着；写成 \\u2028 转义的更不能动，它是 JSON 的一部分。对象前面的说明文字里带着不成对的引号也不碍事。"""
    assert write_skills.parse_skill_json(laid_out({**CLEAN, "pitfalls": "一\u2028二\u0085三"}, "\u2029")) == {
        **CLEAN, "pitfalls": "一\n二\n三"}
    assert write_skills.parse_judge_json("{\u2028" + '"reconstructable": false,\u2029"reason": "一\u2028二"}') == (
        False, "一\u2028二")
    assert write_skills.first_json_object('{\u2028"note": "\\u2028\\u0085"}') == {"note": "\u2028\u0085"}
    assert write_skills.first_json_object('他说："好。\n' + "{\u2028" + '"applicability": "a"}') == {
        "applicability": "a"}
    # 回复本身不是 JSON 对象（说明文字、写到一半被截断）：换了换行也一样解不出，照旧报"没有 JSON 对象"
    for reply in ("抱歉，我做不到。", "{\u2028" + '"applicability": "a", "principles": "b"'):
        with pytest.raises(ValueError, match="回复里没有 JSON 对象"):
            write_skills.first_json_object(reply)


def test_a_teacher_that_lays_out_its_json_with_line_separators_is_not_recorded_as_an_error(tmp_path, capsys):
    """教师的诊断回复和自判回复都拿这三个字符当换行排版：两步都解得出来，这道题才是 ok。自判说"接近答案"的那一条
    也要解得出来，重写才发生。"""
    spec = task("h3_a_001")
    asked = {**CLEAN, "pitfalls": "一\u2028二"}                 # 字符串里面的照旧：取出各段以后换成换行
    fake = FakeAsk([laid_out(LEAKY, "\u2028"), "```json\n" + laid_out(asked, "\u0085") + "\n```"],
                   [laid_out({"reconstructable": True, "reason": "只是指出了\u2029方向"}, "\u2029"), verdict(False)])
    record = write_skills.write_skill(spec, [], fake)
    assert record["status"] == "ok" and record["rounds"] == 2
    assert record["parts"] == {**CLEAN, "pitfalls": "一\n二"} and "一\n二" in record["text"]
    # 自判的理由里字符串里面的那个字符原样留着（没被换成换行）
    assert record["checks"][0] == {"round": 1, "leaks": ["values"], "reconstructable": True,
                                   "reason": "只是指出了\u2029方向"}
    assert len(fake.calls) == 4                                 # 两轮各写一次、判一次：没有哪一步因为解析不了重来

    write_jsonl(tmp_path / "tasks.jsonl", [spec])
    write_manifest(tmp_path / "manifest.json", ["h3_a_001"])
    teacher = ScriptedTeacher([spec])
    teacher.diagnoses["h3_a_001"] = [laid_out(CLEAN, "\u2029")]
    teacher.judges["h3_a_001"] = [laid_out({"reconstructable": False, "reason": "只是指出了方向"}, "\u0085")]
    output = tmp_path / "run/skills.jsonl"
    assert write_skills.main(cli_arguments(tmp_path), client=teacher) == 0
    out, err = capsys.readouterr()
    # 以前这里是 stderr 里的一行"h3_a_001 出错：ValueError: 回复里没有 JSON 对象"
    assert err == "" and "出错（原样重跑会重做）" not in out
    assert out.rstrip().splitlines()[-1] == (f"结果：技能提示 可用 1/1 道，泄露被拦 0 道，出错 0 道；"
                                             f"教师调用 2 次，输出 {output}")
    written, = lines(output)
    assert written["status"] == "ok" and written["rounds"] == 1 and sample_rft.load_skills(output) == {
        "h3_a_001": render(CLEAN)}


@pytest.mark.parametrize("reply, problem", [
    ('{"applicability": "a", "principles": "b", "pitfalls": "c"}', "没有 approach（推荐做法）这一项"),
    ('{"applicability": "a", "principles": "  ", "pitfalls": "c", "approach": "d"}', "principles（要点）是空的"),
    ('{"applicability": ["a"], "principles": "b", "pitfalls": "c", "approach": "d"}',
     "applicability（适用条件）不是字符串"),
    ("抱歉，我做不到。", "回复里没有 JSON 对象"),
    ('["applicability", "principles"]', "回复里没有 JSON 对象"),
    ('{"applicability": "a", "principles": "b", "pitfalls": "c", "approach": "d"', "回复里没有 JSON 对象"),   # 截断了
])
def test_parse_skill_json_says_which_part_is_wrong(reply, problem):
    with pytest.raises(ValueError, match=re.escape(problem)):
        write_skills.parse_skill_json(reply)


def test_parse_judge_json_accepts_booleans_and_words():
    assert write_skills.parse_judge_json(verdict(True, "照着就能写")) == (True, "照着就能写")
    assert write_skills.parse_judge_json('```json\n{"reconstructable": "FALSE"}\n```') == (False, "")
    with pytest.raises(ValueError, match="reconstructable"):
        write_skills.parse_judge_json('{"answer": "yes"}')
    with pytest.raises(ValueError, match="没有 JSON 对象"):
        write_skills.parse_judge_json("不能重建。")


def test_diagnosis_prompt_carries_the_reference_diff_and_the_failed_attempts():
    spec = task("h3_demo_001")
    long_code = "def clamp(values, low, high):\n" + "".join(f"    line_{index} = {index}\n" for index in range(150))
    attempts = [graded(BUGGY, "private_test_failure", ["tests_private.test_hidden::test_hidden",
                                                       "tests_private.test_hidden::test_hidden_range"]),
                graded(WRONG, "private_test_failure", [WHOLE_FILE], missing=HIDDEN_IDS),
                graded(long_code, "public_test_failure", [])]
    system, user = write_skills.diagnosis_messages(spec, attempts)
    assert [message["role"] for message in (system, user)] == ["system", "user"]
    for requirement in ('"applicability"', '"principles"', '"pitfalls"', '"approach"', "1 到 3 句",
                        "只输出一个 JSON 对象", "不出现任何具体的名字", '"该操作"'):
        assert requirement in system["content"]

    diff = "".join(difflib.unified_diff(
        BUGGY.splitlines(keepends=True), FIXED.splitlines(keepends=True),
        fromfile="a/h3_demo_001.py", tofile="b/h3_demo_001.py"))
    assert f"```diff\n{diff}```" in user["content"]          # 同 collect_data.make_patch 的 diff
    assert "-    return list(values)\n" in diff
    assert "+    return [min(max(item, low), high) for item in values]\n" in diff
    assert user["content"].startswith("## 题目要求\n\n" + spec["instruction"] + "\n\n"
                                      f"## 有缺陷的文件 h3_demo_001.py\n\n```python\n{BUGGY.strip()}\n```")
    assert "只供诊断，不要写进技能" in user["content"]
    assert user["content"].index("## 参考修复") < user["content"].index("## 模型的失败尝试")
    assert ("离做对最近的排在前面）\n\n### 失败尝试 1\n"
            "失败类型：private_test_failure（公开测试通过，隐藏测试没通过）\n"
            "没通过的隐藏测试：test_hidden.py::test_hidden、test_hidden.py::test_hidden_range\n"
            f"```python\n{BUGGY.strip()}\n```") in user["content"]
    # 整个隐藏测试文件导入失败：说明是整个文件，并列出因此一个都没执行的用例
    assert ("### 失败尝试 2\n失败类型：private_test_failure（公开测试通过，隐藏测试没通过）\n"
            "没通过的隐藏测试：test_hidden.py（整个文件没能导入）\n"
            f"没能执行的隐藏测试：{'、'.join(HIDDEN_SHOWN)}\n```python\n{WRONG.strip()}\n```") in user["content"]
    limit = write_skills.ATTEMPT_LINES
    assert ("### 失败尝试 3\n失败类型：public_test_failure（公开测试没通过）\n"
            "没通过的隐藏测试：（没有记录）\n```python\n"
            # 代码超过 120 行：只给前 120 行，并说明截断了
            f"{chr(10).join(long_code.splitlines()[:limit])}\n```\n（代码超过 {limit} 行，只给出前 {limit} 行）"
            ) in user["content"]
    assert "line_118 = 118" in user["content"] and "line_119" not in user["content"]
    assert user["content"].count("没能执行的隐藏测试：") == 1

    # 不许出现的名字在第一轮就列给教师，排在失败尝试之后、上一版的问题之前
    forbidden = ("## 技能里不许出现的名字\n\n下面是这道题里的名字和字面量，技能里一个都不要出现——"
                 "其中不少看起来只是普通的英文单词，也一样不要用。需要指代时用\"该操作\"\"这条要求\"一类的说法：\n"
                 + "、".join(sorted(write_skills.code_identifiers(spec))))
    assert user["content"].endswith("\n\n" + forbidden)
    assert user["content"].index("## 模型的失败尝试") < user["content"].index("## 技能里不许出现的名字")

    assert "这一题没有附上失败尝试" in write_skills.diagnosis_messages(spec, [])[1]["content"]
    assert "上一版" not in user["content"]
    with_feedback = write_skills.diagnosis_messages(spec, attempts, "出现了 clamp")[1]["content"]
    assert with_feedback == user["content"] + "\n\n## 上一版的问题\n\n出现了 clamp"


def test_judge_prompt_shows_the_problem_the_skill_and_the_reference():
    spec = task("h3_demo_001")
    messages = write_skills.judge_messages("【要点】先看边界。", spec)
    assert [message["role"] for message in messages] == ["system", "user"]
    system, user = messages[0]["content"], messages[1]["content"]
    assert "能否基本重建参考修复里的具体代码" in system
    assert '{"reconstructable": true 或 false, "reason": "一句话说明理由"}' in system
    problem = build_messages(spec)[1]["content"]              # 模型看到的题目：要求、有缺陷的文件、公开测试
    assert user.startswith(f"## 题目\n\n{problem}\n\n## 技能\n\n【要点】先看边界。\n\n## 参考修复\n\n")
    assert user.endswith(f"```diff\n{write_skills.reference_diff(spec)}```")
    assert "+    return [min(max(item, low), high) for item in values]" in user


@pytest.mark.parametrize("wrap", ["plain", "fence", "prose"])
def test_write_skill_accepts_a_clean_skill_the_first_time(wrap):
    ask = FakeAsk([skill(CLEAN, wrap)], [verdict(False)])
    record = write_skills.write_skill(task("h3_demo_001"), [], ask)
    assert record == {
        "task_id": "h3_demo_001", "status": "ok", "text": render(CLEAN), "parts": CLEAN, "rounds": 1,
        "checks": [{"round": 1, "leaks": [], "reconstructable": False, "reason": "只是指出了方向"}],
        "usage": {"input_tokens": 240, "output_tokens": 80},      # 写技能和自判两次请求的用量合计
        "reminders": 0}                                            # 没有哪一步需要带着提醒重问
    assert [call[1:] for call in ask.calls] == [(4096, 0.3), (4096, 0.3)]     # 默认的输出上限是 4096


def test_write_skill_passes_the_token_limit_and_temperature_to_every_request():
    ask = FakeAsk([skill(CLEAN)], [verdict(False)])
    write_skills.write_skill(task("h3_demo_001"), [], ask, max_tokens=500, temperature=0.9)
    assert [call[1:] for call in ask.calls] == [(500, 0.9), (500, 0.9)]


def test_write_skill_rewrites_after_a_leaked_identifier():
    ask = FakeAsk([skill(LEAKY), skill(CLEAN)], [verdict(True, "写出了具体改法"), verdict(False)])
    record = write_skills.write_skill(task("h3_demo_001"), [], ask)
    assert record["status"] == "ok" and record["rounds"] == 2 and record["text"] == render(CLEAN)
    assert record["checks"] == [
        {"round": 1, "leaks": ["values"], "reconstructable": True, "reason": "写出了具体改法"},
        {"round": 2, "leaks": [], "reconstructable": False, "reason": "只是指出了方向"}]
    assert len(ask.calls) == 4
    second = ask.calls[2][0][1]["content"]                      # 第二轮的诊断请求
    assert second.endswith("## 上一版的问题\n\n"
                           + write_skills.revision_feedback(render(LEAKY), record["checks"][0]))
    feedback = second[second.index("## 上一版的问题"):]
    assert render(LEAKY) in feedback and "values" in feedback and "写出了具体改法" in feedback


def test_write_skill_rewrites_after_the_judge_says_the_answer_can_be_rebuilt():
    ask = FakeAsk([skill(SPECIFIC), skill(CLEAN)], [verdict(True, "照着就能写出那行检查"), verdict(False)])
    record = write_skills.write_skill(task("h3_demo_001"), [], ask)
    assert record["status"] == "ok" and record["rounds"] == 2 and record["text"] == render(CLEAN)
    assert [check["leaks"] for check in record["checks"]] == [[], []]      # 这种泄露静态检查拦不住
    assert [check["reconstructable"] for check in record["checks"]] == [True, False]
    assert "照着就能写出那行检查" in ask.calls[2][0][1]["content"]
    assert render(SPECIFIC) in ask.calls[2][0][1]["content"]


def test_write_skill_gives_up_as_leaky_after_the_last_round():
    ask = FakeAsk([skill(LEAKY)], [verdict(False)])
    record = write_skills.write_skill(task("h3_demo_001"), [], ask, max_rounds=3)
    assert record["status"] == "leaky" and record["rounds"] == 3
    assert "text" not in record and "parts" not in record and record["rejected_text"] == render(LEAKY)
    assert [check["leaks"] for check in record["checks"]] == [["values"]] * 3
    assert len(ask.calls) == 6                                  # 三轮，每轮写一次、自判一次
    assert all("上一版的问题" in call[0][1]["content"] for call in ask.calls[2::2])
    without_judge = FakeAsk([skill(LEAKY)])
    record = write_skills.write_skill(task("h3_demo_001"), [], without_judge, judge=False)
    assert record["status"] == "leaky" and record["rounds"] == 2 and len(without_judge.calls) == 2
    assert record["checks"][0]["reconstructable"] is None and "rejected_text" in record


def scripted(*replies):
    """假的 ask：按先后给出 replies（用完以后重复最后一个），记下每次的消息。每条回复是字典（同 chat_text
    的返回值）。"""
    calls, left = [], [reply if isinstance(reply, dict) else {"text": reply} for reply in replies]

    def ask(messages, max_tokens, temperature):
        calls.append(messages)
        reply = left.pop(0) if len(left) > 1 else left[0]
        return {**{"finish_reason": "stop", "usage": {"input_tokens": 120, "output_tokens": 40},
                   "other_text": {}}, **reply}

    return ask, calls


CUT_ADVICE = ("把上限调大（--max-tokens，现在是 4096；经 run_rft_guided.sh 跑时设环境变量 TEACHER_MAX_TOKENS），"
              "调大后原样重跑只重做出错的题")
# 正文有、可是带着提醒重问一次还是没有可用的 JSON：最后记 error 时的话
UNUSABLE = "ValueError: 写技能的回复里还是没有可用的 JSON（已经带着提醒重问过一次）：回复里没有 JSON 对象"


def test_a_reply_cut_off_at_the_limit_or_empty_is_reported_as_cut_off_not_as_missing_json():
    """教师会先写一大段推理再给 JSON：回复没写完就到了输出上限，或者（推理占满了输出）正文干脆是空的。以前这两种
    都报成"回复里没有 JSON 对象"，出错记录里又没留回复，事后查不出原因。现在先判断是不是没写完：报的就是
    "到了输出上限、把上限调大"，带上该调的环境变量，不带着提醒重问（再问一次还是截断）。"""
    spec = task("h3_demo_001")
    cut = skill(CLEAN)[:-6]
    ask, calls = scripted({"text": cut, "finish_reason": "length", "other_text": {"reasoning_content": 3000}})
    record = write_skills.write_skill(spec, [], ask)
    assert (record["status"], record["rounds"], record["reminders"], len(calls)) == ("error", 1, 0, 1)
    assert record["error"] == ("ValueError: 写技能的回复在写完之前到了输出上限（finish_reason=length，正文 "
                               f"{len(cut)} 个字符，推理等其它字段：reasoning_content 3000 个字符），{CUT_ADVICE}")
    assert "没有 JSON" not in record["error"] and record["checks"] == []
    assert record["reply"] == {"text": cut, "chars": len(cut), "head_only": False, "finish_reason": "length",
                               "usage": {"input_tokens": 120, "output_tokens": 40},
                               "other_text": {"reasoning_content": 3000}}
    # 正文是空的、推理很长：同一种事。只记推理有多长，不记内容
    ask, calls = scripted({"text": "", "finish_reason": "length", "other_text": {"reasoning_content": 4000}})
    record = write_skills.write_skill(spec, [], ask)
    assert len(calls) == 1 and record["error"] == (
        "ValueError: 写技能的回复在写完之前到了输出上限（finish_reason=length，正文 0 个字符，"
        f"推理等其它字段：reasoning_content 4000 个字符），{CUT_ADVICE}")
    # 空回复、结束原因是 stop：同样按没写完报
    ask, calls = scripted({"text": "  \n"})
    record = write_skills.write_skill(spec, [], ask)
    assert len(calls) == 1 and record["error"] == (
        f"ValueError: 写技能的回复是空的（finish_reason=stop，正文 0 个字符），多半同样是没有写完，{CUT_ADVICE}")
    assert record["reply"]["chars"] == 3 and record["reply"]["text"] == "  \n"
    # 自判那一步被截断：说的是自判的回复
    cut = '{"reconstructable": fal'
    ask, calls = scripted(skill(CLEAN), {"text": cut, "finish_reason": "length"})
    record = write_skills.write_skill(spec, [], ask)
    assert record["status"] == "error" and len(calls) == 2
    assert record["error"] == (f"ValueError: 自判的回复在写完之前到了输出上限（finish_reason=length，正文 "
                               f"{len(cut)} 个字符），{CUT_ADVICE}")


def test_a_reply_without_usable_json_is_asked_again_with_a_reminder():
    """正文有、可是没有可解析的 JSON 对象（夹了一段说明、JSON 缺了后半截却没报 length……）：不直接记 error，在原
    消息末尾加一句提醒重问一次。重问的那次给对了，这道题照常是 ok；重问不算一轮。"""
    spec = task("h3_demo_001")
    ask, calls = scripted("好的，技能如下：", skill(CLEAN), verdict(False))
    record = write_skills.write_skill(spec, [], ask)
    assert record["status"] == "ok" and record["reminders"] == 1 and record["rounds"] == 1
    assert record["text"] == render(CLEAN) and len(calls) == 3             # 写（不成）、带着提醒重写、自判
    assert calls[1][:-1] == calls[0][:-1] and calls[1][-1]["role"] == "user"
    assert calls[1][-1]["content"] == calls[0][-1]["content"] + "\n\n" + write_skills.REMINDER
    assert write_skills.REMINDER == "上一次的回复里没有可解析的 JSON 对象，请只输出一个 JSON 对象，不要任何别的文字。"
    assert "上一次的回复里没有" not in calls[0][-1]["content"]             # 第一次问的时候没有这句
    assert "上一次的回复里没有" not in calls[2][-1]["content"]                # 自判那一步的消息没有被牵连
    # 自判的回复解析不了：同样重问一次；重问的那次说能重建，结论照常记下（只写一轮，所以最后是 leaky）
    ask, calls = scripted(skill(CLEAN), "不能重建。", verdict(True, "照着就能写"))
    record = write_skills.write_skill(spec, [], ask, max_rounds=1)
    assert record["status"] == "leaky" and record["reminders"] == 1 and len(calls) == 3
    assert record["checks"][0]["reconstructable"] is True and calls[2][-1]["content"].endswith(write_skills.REMINDER)
    # 只重问一次：第二次还不行才往下报
    ask, calls = scripted("好的。", "还是不成。", skill(CLEAN))
    record = write_skills.write_skill(spec, [], ask)
    assert record["status"] == "error" and record["reminders"] == 1 and len(calls) == 2
    assert "text" not in record


def test_the_record_of_a_failure_keeps_the_raw_reply_and_nothing_of_the_task():
    """最终失败的记录里留下原始回复的前 800 个字符、结束原因、用量和推理字段的长度；不把整道题写进去。"""
    spec = task("h3_demo_001")
    prose = "我先想一想：" + "思路" * 1000                     # 比 800 个字符长：只留前面一段
    ask, calls = scripted(prose, prose, prose)
    record = write_skills.write_skill(spec, [], ask)
    assert record["status"] == "error" and record["reminders"] == 1 and len(calls) == 2
    assert record["error"] == ("ValueError: 写技能的回复里还是没有可用的 JSON（已经带着提醒重问过一次）："
                              "回复里没有 JSON 对象")
    assert record["reply"] == {"text": prose[:800], "chars": len(prose), "head_only": True, "finish_reason": "stop",
                               "usage": {"input_tokens": 120, "output_tokens": 40}, "other_text": {}}
    assert record["usage"] == {"input_tokens": 240, "output_tokens": 80}    # 重问那一次的用量也记着
    assert write_skills.dump_line(record).count("\n") == 1                # 还是一行一条记录
    dumped = json.dumps(record, ensure_ascii=False)
    assert spec["buggy"] not in dumped and spec["fixed"] not in dumped and "instruction" not in record
    assert calls[1][-1]["content"].endswith(write_skills.REMINDER)
    # 请求本身出错（重试用完）：没有拿到回复，记录里就没有 reply 这一项，错误照旧记着
    def offline(messages, max_tokens, temperature):
        raise ConnectionError("接口连不上")

    record = write_skills.write_skill(spec, [], offline)
    assert (record["status"], record["error"], record["rounds"]) == ("error", "ConnectionError: 接口连不上", 1)
    assert "reply" not in record and record["reminders"] == 0
    # 被截断的回复也留着诊断：看得出它停在哪
    cut = skill(CLEAN)[:-6]
    ask, _ = scripted({"text": cut, "finish_reason": "length", "usage": {"input_tokens": 9, "output_tokens": 7}})
    record = write_skills.write_skill(spec, [], ask)
    assert record["status"] == "error" and record["usage"] == {"input_tokens": 9, "output_tokens": 7}
    assert record["reply"]["text"] == cut and record["reply"]["finish_reason"] == "length"


def test_the_key_never_reaches_an_error_record(tmp_path, capsys, monkeypatch):
    """出错记录会落盘、会跟着日志贴给别人：异常的消息里万一带着密钥，不能原样记进去；教师的回复里带着也一样。"""
    secret = "offline-key-for-tests-1234567890"
    monkeypatch.setenv("TEACHER_API_KEY", secret)

    def leaking(messages, max_tokens, temperature):
        raise RuntimeError(f"request rejected for {secret}")

    record = write_skills.write_skill(task("h3_demo_001"), [], leaking, secret=secret)
    assert record["error"] == "RuntimeError: request rejected for ***" and secret not in json.dumps(record)
    ask, _ = scripted(f"（调试信息：密钥是 {secret}）这不是 JSON", f"（调试信息：密钥是 {secret}）还不是 JSON")
    record = write_skills.write_skill(task("h3_demo_001"), [], ask, secret=secret)
    assert record["status"] == "error" and record["reply"]["text"] == "（调试信息：密钥是 ***）还不是 JSON"
    assert secret not in json.dumps(record, ensure_ascii=False)
    # 经命令行跑：密钥从 --api-key-env 指的环境变量里读出来去抹；太短的值不抹（到处都碰得到）
    write_jsonl(tmp_path / "tasks.jsonl", [task("h3_a_001")])
    write_manifest(tmp_path / "manifest.json", ["h3_a_001"])
    teacher = ScriptedTeacher([task("h3_a_001")])
    teacher.diagnoses["h3_a_001"] = [f"密钥是 {secret}"]
    assert write_skills.main([*cli_arguments(tmp_path), "--no-judge"], client=teacher) == 3
    capsys.readouterr()
    stored = (tmp_path / "run/skills.jsonl").read_text(encoding="utf-8")
    assert secret not in stored and "密钥是 ***" in stored
    monkeypatch.setenv("TEACHER_API_KEY", "k1")
    assert write_skills.redact("k1 和 k1", "k1") == "k1 和 k1"


def test_write_skill_names_the_step_whose_reply_was_cut_off_at_the_token_limit():
    def asking(cut):
        def ask(messages, max_tokens, temperature):
            judging = JUDGE in messages[0]["content"]
            text = verdict(False) if judging else skill(CLEAN)
            return {"text": text[:-6] if judging == cut else text,       # 被截断的那条去掉结尾
                    "finish_reason": "length" if judging == cut else "stop",
                    "usage": {"input_tokens": 1, "output_tokens": 1}}
        return ask

    # 经编排脚本跑时只能靠环境变量调：话里点名是哪一步的回复、现在上限是多少
    record = write_skills.write_skill(task("h3_demo_001"), [], asking(cut=False), max_tokens=700)
    assert record["error"] == ("ValueError: 写技能的回复在写完之前到了输出上限"
                               f"（finish_reason=length，正文 {len(skill(CLEAN)) - 6} 个字符），"
                               "把上限调大（--max-tokens，现在是 700；经 run_rft_guided.sh 跑时设环境变量"
                               " TEACHER_MAX_TOKENS），调大后原样重跑只重做出错的题")
    # 截断了：不重问、不自判
    assert record["rounds"] == 1 and record["usage"] == {"input_tokens": 1, "output_tokens": 1}
    assert "没有 JSON" not in record["error"] and record["reminders"] == 0

    record = write_skills.write_skill(task("h3_demo_001"), [], asking(cut=True), max_tokens=700)   # 只有自判被截断
    assert record["status"] == "error" and "text" not in record and record["rounds"] == 1
    assert record["error"] == ("ValueError: 自判的回复在写完之前到了输出上限（finish_reason=length，正文 "
                               f"{len(verdict(False)) - 6} 个字符），把上限调大（--max-tokens，现在是 700；"
                               "经 run_rft_guided.sh 跑时设环境变量 TEACHER_MAX_TOKENS），调大后原样重跑只重做出错的题")
    assert record["usage"] == {"input_tokens": 2, "output_tokens": 2}   # 写技能、自判各一次，没有重问
    assert record["checks"] == [{"round": 1, "leaks": [], "reconstructable": None, "reason": None}]

    record = write_skills.write_skill(task("h3_demo_001"), [], asking(cut=None))   # 都没截断
    assert record["status"] == "ok" and record["text"] == render(CLEAN)


def test_write_skill_also_accepts_plain_string_replies():
    def ask(messages, max_tokens, temperature):
        return skill(CLEAN) if JUDGE not in messages[0]["content"] else verdict(False)

    record = write_skills.write_skill(task("h3_demo_001"), [], ask)
    assert record["status"] == "ok" and record["text"] == render(CLEAN)
    assert record["usage"] == {"input_tokens": 0, "output_tokens": 0}


def test_write_skill_never_raises_so_one_bad_task_cannot_stop_the_batch():
    def explode(messages, max_tokens, temperature):
        raise RuntimeError("boom")

    record = write_skills.write_skill(task("h3_demo_001"), [], explode)
    assert record["status"] == "error" and record["error"] == "RuntimeError: boom"
    for broken in ({}, None):
        record = write_skills.write_skill(broken, [], FakeAsk([skill(CLEAN)]))
        assert record["status"] == "error" and record["task_id"] is None and "text" not in record


def test_pick_attempts_prefers_near_misses_deduplicates_and_skips_unusable_code():
    records = [
        graded(WRONG, "public_test_failure", []),
        graded(BROKEN, "syntax_error", []),                         # 代码解析不了
        graded("", "no_code", []),
        graded(BUGGY, "private_test_failure", HIDDEN_IDS),
        graded(FIXED, None, [], strict=True),                       # 通过的不算失败尝试
        graded(BUGGY, "private_test_failure", HIDDEN_IDS),              # 代码相同，只留一份
        graded("def clamp(values, low, high):\n    return sorted(values)\n", "private_test_failure", HIDDEN_IDS),
        graded("def clamp(values, low, high):\n    return list(values) + []\n", "execution_timeout", []),
    ]
    chosen = write_skills.pick_attempts(records)
    assert [record["target"]["content"] for record in chosen] == [BUGGY, records[6]["target"]["content"]]
    assert write_skills.pick_attempts(records, limit=3) == [records[3], records[6], records[0]]
    assert write_skills.pick_attempts(records, limit=1) == [records[3]]
    assert write_skills.pick_attempts([records[1], records[2]]) == []
    assert write_skills.pick_attempts(records, limit=0) == []


class HttpError(Exception):
    def __init__(self, status_code, retry_after=None):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        if retry_after is not None:              # 同真的 SDK：头在异常的 response.headers 上，名字不分大小写
            self.response = SimpleNamespace(headers={"Retry-After": retry_after})


class Named(Exception):
    pass


class Clock:
    """假的时间：sleep 把时钟往前拨，monotonic 读它。替换掉 _shared_teacher 里的 time 以后，重试和计时的
    用例不用真等。"""

    def __init__(self, start=1000.0):
        self.now, self.sleeps = start, []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.mark.parametrize("error, expected", [
    (HttpError(429), True), (HttpError(500), True), (HttpError(503), True),
    (HttpError(408), False), (HttpError(404), False), (HttpError(401), False), (HttpError(400), False),
    (type("APIConnectionError", (Exception,), {})("连接被重置"), True),
    (type("APITimeoutError", (Exception,), {})("超时"), True),
    (type("APIConnectionError", (Named,), {})("子类也算"), True),
    (ConnectionResetError("reset"), True), (TimeoutError("timed out"), True),
    (ValueError("bad"), False), (TypeError("bad"), False), (KeyError("bad"), False),
    (type("RateLimitError", (HttpError,), {})(400), False),        # 类名像限流，状态码却不是：不重试
])
def test_transient_separates_retryable_failures_from_the_rest(error, expected):
    assert shared.transient(error) is expected


OFF = lambda: shared.Throttle(min_interval=0)       # noqa: E731   测重试的用例不看节流


def ask(client, **options):
    return shared.chat_text(client, "teacher", [{"role": "user", "content": "hi"}], max_tokens=32, temperature=0.2,
                            throttle=OFF(), **options)


def test_transient_and_chat_text_work_with_the_real_sdk_errors_when_installed(capsys, monkeypatch):
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    request = httpx.Request("POST", "https://teacher-api.example/x")
    limited = openai.RateLimitError("您的账户已达到速率限制", body=None,
                                    response=httpx.Response(429, headers={"Retry-After": "7"}, request=request))
    errors = [openai.APITimeoutError(request=request), limited, openai.APIConnectionError(request=request)]
    assert all(shared.transient(error) for error in errors) and shared.limited(limited)
    assert shared.retry_after(limited) == 7.0                      # Retry-After 头：SDK 把它放在异常的 response 上
    clock = Clock()
    monkeypatch.setattr(shared, "time", clock)
    client = Flaky(errors)
    result = ask(client)
    assert result["text"] == "ok" and client.calls == 4
    assert clock.sleeps == [2.0, 7.0, 4.0]            # 超时 2 秒；限流按 Retry-After 等 7 秒；连不上再等 4 秒
    assert capsys.readouterr().err.splitlines() == ["教师接口限速（429），等 7 秒后重试（第 1 次）"]
    refused = openai.AuthenticationError("bad key", response=httpx.Response(401, request=request), body=None)
    client = Flaky([refused])
    with pytest.raises(openai.AuthenticationError):
        ask(client)
    assert client.calls == 1 and clock.sleeps == [2.0, 7.0, 4.0]      # 鉴权失败：不重试，也不多等
    assert capsys.readouterr().err == ""


def test_a_rate_limited_call_is_retried_for_a_long_time_and_says_so_on_stderr(capsys, monkeypatch):
    """429 不再是"一共请求 4 次、退避 2、4、8 秒就放弃"（服务器上并发 4 时成片 429，72 道题里大半记成了 error）：
    第一次等 15 秒，之后每次翻倍、最长 120 秒，一直等到成功。每等一次往标准错误打一行，看日志的人知道是在等。"""
    clock = Clock()
    monkeypatch.setattr(shared, "time", clock)
    client = Flaky([HttpError(429)] * 5, text="好了")
    result = ask(client)
    assert result["text"] == "好了" and client.calls == 6
    assert clock.sleeps == [15.0, 30.0, 60.0, 120.0, 120.0]
    assert capsys.readouterr().err.splitlines() == [
        "教师接口限速（429），等 15 秒后重试（第 1 次）", "教师接口限速（429），等 30 秒后重试（第 2 次）",
        "教师接口限速（429），等 60 秒后重试（第 3 次）", "教师接口限速（429），等 120 秒后重试（第 4 次）",
        "教师接口限速（429），等 120 秒后重试（第 5 次）"]
    # 服务端错误（5xx）是同一种重法：过一阵多半能自己恢复
    client = Flaky([HttpError(503)] * 2, text="好了")
    assert ask(client)["text"] == "好了" and clock.sleeps[5:] == [15.0, 30.0]
    assert capsys.readouterr().err.splitlines() == ["教师接口出错（503），等 15 秒后重试（第 1 次）",
                                                     "教师接口出错（503），等 30 秒后重试（第 2 次）"]


def test_retry_after_in_the_response_decides_how_long_to_wait(capsys, monkeypatch):
    """响应头里有 Retry-After 就按它等（秒数，或者一个 HTTP 日期）；读不出来才按退避算。"""
    clock = Clock()
    monkeypatch.setattr(shared, "time", clock)
    soon = email.utils.format_datetime(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=40),
                                        usegmt=True)
    client = Flaky([HttpError(429, "7"), HttpError(429, "0.5"), HttpError(503, soon), HttpError(429, "soon")],
                   text="好了")
    assert ask(client)["text"] == "好了"
    assert clock.sleeps[:2] == [7.0, 0.5] and 38 <= clock.sleeps[2] <= 40.0 and clock.sleeps[3] == 120.0
    shown = capsys.readouterr().err.splitlines()
    assert shown[:2] == ["教师接口限速（429），等 7 秒后重试（第 1 次）",
                         "教师接口限速（429），等 0.5 秒后重试（第 2 次）"]
    assert re.fullmatch(r"教师接口出错（503），等 (39|40)(\.\d)? 秒后重试（第 3 次）", shown[2])
    # 头里写的不是数也不是日期：当没有这个头，第 4 次按退避等 120 秒
    assert shown[3] == "教师接口限速（429），等 120 秒后重试（第 4 次）"
    # 没有这个头、负数、异常对象上连 response 都没有：都当没有，按退避算
    for odd in (HttpError(429), HttpError(429, "-3"), SimpleNamespace(status_code=429),
                SimpleNamespace(status_code=429, response=None)):
        assert shared.retry_after(odd) is None
    assert shared.retry_after(HttpError(429, "0")) == 0.0      # 写着 0：不用等

def test_gives_up_only_when_the_time_since_the_first_failure_has_run_out(capsys, monkeypatch):
    """从第一次失败算起等满总时长（默认 30 分钟）才放弃；最后一次等待不会超过上限。放弃前打一行，然后抛最后那个错误。"""
    clock = Clock()
    monkeypatch.setattr(shared, "time", clock)
    client = Flaky([HttpError(429)] * 10, text="不该走到这里")
    with pytest.raises(HttpError, match="HTTP 429"):
        ask(client, retry_seconds=300)
    # 15 + 30 + 60 + 120 = 225 秒；再等 120 就超过 300：最后一次只等剩下的 75 秒，正好到上限，那一次请求还是 429 才放弃
    assert clock.sleeps == [15.0, 30.0, 60.0, 120.0, 75.0] and client.calls == 6
    assert clock.now == 1300.0
    assert capsys.readouterr().err.splitlines() == [
        "教师接口限速（429），等 15 秒后重试（第 1 次）", "教师接口限速（429），等 30 秒后重试（第 2 次）",
        "教师接口限速（429），等 60 秒后重试（第 3 次）", "教师接口限速（429），等 120 秒后重试（第 4 次）",
        "教师接口限速（429），等 75 秒后重试（第 5 次）",
        "教师接口限速（429）：从第一次失败起已经等了 5 分钟，重试了 6 次，不再重试"]
    assert shared.RETRY_SECONDS == 30 * 60 and shared.FIRST_WAIT == 15.0 and shared.MAX_WAIT == 120.0
    # 重试的请求自己花掉的时间也算在总时长里：第二次请求（从第一次失败算起）就耗光了上限，它再失败就不等了
    class Slow(Flaky):
        def create(self, **kwargs):
            if self.calls:                       # 第一次请求之后，每次都要花 290 秒
                clock.now += 290
            return super().create(**kwargs)

    client = Slow([HttpError(503)] * 3)
    with pytest.raises(HttpError, match="HTTP 503"):
        ask(client, retry_seconds=300)
    assert client.calls == 2 and clock.sleeps[5:] == [15.0]
    assert capsys.readouterr().err.splitlines()[-1] == (
        "教师接口出错（503）：从第一次失败起已经等了 5 分钟，重试了 2 次，不再重试")


def test_connection_errors_keep_the_short_counted_retry(capsys, monkeypatch):
    """连不上、超时还是老样子：一共最多请求 4 次（``retries``），间隔 2、4、8 秒，不为它们长时间等。
    429、5xx 不占这个次数。"""
    clock = Clock()
    monkeypatch.setattr(shared, "time", clock)
    client = Flaky([HttpError(503), ConnectionResetError("reset"), HttpError(429)])
    result = ask(client)
    assert result == {"text": "ok", "finish_reason": "stop",      # 接口没报用量；没有别的文字字段
                      "usage": {"input_tokens": None, "output_tokens": None}, "other_text": {}}
    assert clock.sleeps == [15.0, 2.0, 30.0] and client.calls == 4
    assert capsys.readouterr().err.splitlines() == ["教师接口出错（503），等 15 秒后重试（第 1 次）",
                                                     "教师接口限速（429），等 30 秒后重试（第 2 次）"]
    clock.sleeps.clear()
    capsys.readouterr()
    client = Flaky([HttpError(400)])
    with pytest.raises(HttpError):
        ask(client)
    assert client.calls == 1 and clock.sleeps == [] and capsys.readouterr().err == ""   # 不是瞬时错误：立刻抛，也不多等
    client = Flaky([ConnectionResetError("reset")] * 3)
    with pytest.raises(ConnectionResetError):
        ask(client, retries=2)                                      # 连不上两次（retries=2）就放弃，中间只等了一次
    assert client.calls == 2 and clock.sleeps == [2.0]
    client = Flaky([HttpError(500)] * 6, text=None)
    assert ask(client, retries=2)["text"] == ""                     # 5xx 不看这个次数：等到成功（第 7 次）
    assert client.calls == 7 and clock.sleeps[1:] == [15.0, 30.0, 60.0, 120.0, 120.0, 120.0]
    with pytest.raises(ValueError, match="positive"):
        ask(client, retries=0)


# ---- _shared_teacher：进程内的节流


def test_the_throttle_limits_requests_in_flight_and_spaces_out_their_starts():
    """同一时刻在飞的请求不超过 max_in_flight 个，相邻两次请求的开始至少隔 min_interval 秒。真线程，间隔取得很短。"""
    throttle = shared.Throttle(max_in_flight=3, min_interval=0.02)
    starts, state, guard = [], {"now": 0, "peak": 0}, threading.Lock()
    gate = threading.Barrier(10)

    def worker():
        gate.wait()                                   # 十个线程一起挤到节流器前
        with throttle.request() as started:
            starts.append(started)
            with guard:
                state["now"] += 1
                state["peak"] = max(state["peak"], state["now"])
            time.sleep(0.12)                          # 一次请求要 0.12 秒：是间隔的六倍，不限的话会有七个同时在飞
            with guard:
                state["now"] -= 1

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert len(starts) == 10 and state == {"now": 0, "peak": 3}
    ordered = sorted(starts)
    assert all(later - earlier >= 0.02 - 1e-6 for earlier, later in zip(ordered, ordered[1:]))
    assert throttle.state() == {"in_flight": 0, "last_start": ordered[-1]}
    # 请求抛了异常，名额也放回去
    with pytest.raises(RuntimeError):
        with throttle.request():
            raise RuntimeError("boom")
    assert throttle.state()["in_flight"] == 0
    # 改设置马上生效；不合理的值报错
    throttle.configure(max_in_flight=5, min_interval=0)
    assert (throttle.max_in_flight, throttle.min_interval) == (5, 0)
    for bad in ({"max_in_flight": 0}, {"min_interval": -1}):
        with pytest.raises(ValueError):
            throttle.configure(**bad)
    assert (throttle.max_in_flight, throttle.min_interval) == (5, 0)


def test_every_call_in_the_process_goes_through_the_same_throttle(monkeypatch):
    """限额按账户算，不能每个调用点各算各的：不另外给 throttle 的 chat_text 都走进程里共用的那一份（THROTTLE），
    configure_throttle 改的也是它。两个调用点、两个客户端，请求照样隔开。"""
    seen, lock = [], threading.Lock()

    class Client:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            with lock:
                seen.append((time.monotonic(), len(seen)))
            return response("ok")

    shared.configure_throttle(max_in_flight=1, min_interval=0.2)
    assert (shared.THROTTLE.max_in_flight, shared.THROTTLE.min_interval) == (1, 0.2)
    assert shared.DEFAULT_IN_FLIGHT == 2 and shared.DEFAULT_INTERVAL == 1.0
    threads = [threading.Thread(target=shared.chat_text, args=(Client(), "teacher", []),
                                kwargs={"max_tokens": 8, "temperature": 0}) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert len(seen) == 2 and abs(seen[1][0] - seen[0][0]) >= 0.2 - 0.01      # 两个请求的开始至少隔 0.2 秒
    assert shared.THROTTLE.state()["in_flight"] == 0


# ---- _shared_teacher：TEACHER_EXTRA_JSON 和推理字段的长度


class Recorder:
    """假的客户端：记下每次请求的参数。"""

    def __init__(self, text="ok"):
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.text = text

    def create(self, model, messages, max_tokens, temperature, extra_body=None):
        # 签名同真的 SDK：只认这几个参数，多出来的会像真 SDK 那样报 TypeError；记下的是最终发出去的请求体
        self.requests.append({"model": model, "messages": messages, "max_tokens": max_tokens,
                              "temperature": temperature, **(extra_body or {})})
        return response(self.text)


def test_extra_json_from_the_environment_is_merged_into_every_request(monkeypatch):
    """TEACHER_EXTRA_JSON 是一个 JSON 对象时原样合并进每次请求的请求体（同名以它为准）；没设就什么都不加。"""
    monkeypatch.delenv("TEACHER_EXTRA_JSON", raising=False)
    client = Recorder()
    ask(client)
    ask(client)
    body = {"model": "teacher", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 32, "temperature": 0.2}
    assert client.requests == [body, body]                       # 没设：请求体和原来一模一样
    monkeypatch.setenv("TEACHER_EXTRA_JSON", '  {"sample_flag": true, "nested": {"limit": [1, 2]}} ')
    ask(client)
    assert client.requests[-1] == {**body, "sample_flag": True, "nested": {"limit": [1, 2]}}
    monkeypatch.setenv("TEACHER_EXTRA_JSON", '{"temperature": 0.9}')
    ask(client)
    assert client.requests[-1] == {**body, "temperature": 0.9}
    monkeypatch.setenv("TEACHER_EXTRA_JSON", "")
    ask(client, extra={"width": 3})                              # 调用点自己给了：不读环境变量
    assert client.requests[-1] == {**body, "width": 3}
    # 设了却不是 JSON 对象：报清楚，一个请求都不发
    for value, piece in (("not json", "现在不是合法的 JSON"), ("[1, 2]", "现在解析出来是list"),
                         ('"text"', "现在解析出来是str")):
        monkeypatch.setenv("TEACHER_EXTRA_JSON", value)
        with pytest.raises(ValueError, match="环境变量 TEACHER_EXTRA_JSON 要是一个 JSON 对象") as caught:
            ask(client)
        assert piece in str(caught.value) and piece.split("是")[-1] in str(caught.value)
    assert len(client.requests) == 5                                  # 那几次都停在发请求之前
    monkeypatch.delenv("TEACHER_EXTRA_JSON", raising=False)
    assert shared.extra_json() == {} and shared.extra_json("  ") == {} and shared.extra_json('{"a": 1}') == {"a": 1}


def test_the_length_of_the_reasoning_field_is_reported_but_never_its_content():
    """思考类模型把推理放在单独的字段里（字段名各家不一样，消息上有什么就记什么）。只记长度，不记内容。"""
    thinking = "想" * 1200

    class Message:                                    # 像 SDK 的消息对象：不认识的字段放在 model_extra 里
        role = "assistant"
        content = None
        model_extra = {"reasoning_content": thinking, "detail": {"kind": "thin"}}

    assert shared.other_text(Message()) == {"reasoning_content": 1200}
    assert shared.other_text(SimpleNamespace(content="正文", role="assistant", refusal="",
                                             notes="备注" * 5)) == {"notes": 10}
    assert shared.other_text(SimpleNamespace(content="正文")) == {}
    for broken in (None, 7, object()):
        assert shared.other_text(broken) == {}
    client = Recorder("")
    client.chat.completions.create = lambda **kwargs: SimpleNamespace(
        usage=None, choices=[SimpleNamespace(finish_reason="length", message=Message())])
    result = ask(client)
    assert result == {"text": "", "finish_reason": "length", "usage": {"input_tokens": None, "output_tokens": None},
                      "other_text": {"reasoning_content": 1200}}
    assert "想" not in json.dumps(result, ensure_ascii=False)


def test_make_client_reads_the_key_and_the_address_from_the_environment(monkeypatch):
    created = []

    class OpenAI:
        def __init__(self, **kwargs):
            created.append(kwargs)
            self.options = kwargs

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=OpenAI))
    for name in ("TEACHER_API_KEY", "TEACHER_API_KEY_2", "TEACHER_BASE_URL", "TEACHER_MODEL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match=r"TEACHER_API_KEY.*\.env"):
        shared.make_client()
    assert created == []                                        # 没有密钥就不建客户端
    monkeypatch.setenv("TEACHER_API_KEY", "offline-key")
    assert shared.make_client(timeout=30).options == {
        "api_key": "offline-key", "base_url": "https://teacher-api.example/api/paas/v4",
        "timeout": 30, "max_retries": 0}
    monkeypatch.setenv("TEACHER_BASE_URL", "https://teacher.example/v1")
    shared.make_client()
    assert created[-1]["base_url"] == "https://teacher.example/v1"
    with pytest.raises(RuntimeError, match="TEACHER_API_KEY_2"):
        shared.make_client("TEACHER_API_KEY_2")
    monkeypatch.setenv("TEACHER_API_KEY_2", "second-key")
    shared.make_client("TEACHER_API_KEY_2", "https://teacher.example/other", 5)
    assert created[-1] == {"api_key": "second-key", "base_url": "https://teacher.example/other",
                           "timeout": 5, "max_retries": 0}
    assert shared.teacher_model() == shared.DEFAULT_MODEL
    monkeypatch.setenv("TEACHER_MODEL", "teacher-x")
    assert shared.teacher_model() == "teacher-x"


def test_make_client_treats_a_missing_sdk_as_an_import_error(monkeypatch):
    monkeypatch.setenv("TEACHER_API_KEY", "offline-key")
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(ImportError):
        shared.make_client()


def cli_arguments(tmp_path, output="run/skills.jsonl"):
    return ["--tasks-file", str(tmp_path / "tasks.jsonl"), "--zero-pass-from", str(tmp_path / "manifest.json"),
            "--graded", str(tmp_path / "train-graded.jsonl"), "--output", str(tmp_path / output)]


def test_cli_writes_skills_resumes_and_reports_errors(tmp_path, capsys):
    specs = [task(name, "dev" if name == "h3_e_001" else "train")
             for name in ("h3_a_001", "h3v_b_001", "h3_c_001", "demo_d_001", "h3_e_001")]
    write_jsonl(tmp_path / "tasks.jsonl", specs)
    write_manifest(tmp_path / "manifest.json", [*sorted(spec["task_id"] for spec in specs), "h3_ghost_001"])
    teacher = ScriptedTeacher(specs)
    teacher.diagnoses["h3_a_001"] = [skill(CLEAN, "fence")]
    teacher.diagnoses["h3v_b_001"] = [skill(LEAKY), skill(CLEAN)]
    teacher.diagnoses["h3_c_001"] = ["服务繁忙，请稍后再试。"]          # 每次都回一句说明文字，不给 JSON
    output = tmp_path / "run/skills.jsonl"
    command = [*cli_arguments(tmp_path), "--concurrency", "1", "--model", "teacher-x",
               "--max-tokens", "600", "--temperature", "0.5"]

    # 第一次：一道一次就过，一道泄露后重写通过，一道回复不是 JSON。原有题和 dev 的题不在选题里
    assert write_skills.main(command, client=teacher) == 3
    out, err = capsys.readouterr()
    assert "没找到判分文件" in out and "不带失败尝试，继续。" in out
    assert ("写技能：3 道题（train，hard,variant 层，上一轮零通过），已有定论 0 道，待写 3 道，"
            "其中 3 道没有可用的失败尝试。") in out
    assert ("教师：teacher-x；每题最多 2 轮，写完自判；每次回复最多 600 个 token，温度 0.5，单次请求超时 120 秒"
            "（调法见本脚本开头的说明）；同时最多 1 个请求，两次请求至少隔 0 秒。") in out
    progress = [line for line in out.splitlines() if line.startswith("[")]
    assert [re.sub(r"已用.*", "", line) for line in progress] == [
        "[########----------------] 1/3 (33%) 写技能 h3_a_001 ok（1 轮） · ",
        "[################--------] 2/3 (66%) 写技能 h3v_b_001 ok（2 轮） · ",
        "[########################] 3/3 (100%) 写技能 h3_c_001 error（1 轮） · "]
    assert all(re.search(r"已用 \d+\.\dm · 剩余≈\d+\.\dm$", line) for line in progress)
    assert f"h3_c_001 出错：{UNUSABLE}" in err
    # 出错的那道：问一次、带着提醒重问一次，所以一共 8 次
    assert out.rstrip().splitlines()[-2:] == ["出错（原样重跑会重做）：h3_c_001",
                                              "结果：技能提示 可用 2/3 道，泄露被拦 0 道，出错 1 道；"
                                              f"教师调用 8 次，输出 {output}"]
    records = {record["task_id"]: record for record in lines(output)}
    assert {name: record["status"] for name, record in records.items()} == {
        "h3_a_001": "ok", "h3v_b_001": "ok", "h3_c_001": "error"}
    assert records["h3_a_001"]["text"] == render(CLEAN) and records["h3_a_001"]["parts"] == CLEAN
    assert records["h3_a_001"]["model"] == "teacher-x" and "created_at" in records["h3_a_001"]
    assert records["h3v_b_001"]["rounds"] == 2 and records["h3v_b_001"]["checks"][0]["leaks"] == ["values"]
    assert "error" in records["h3_c_001"] and "text" not in records["h3_c_001"]
    assert records["h3_c_001"]["reminders"] == 1 and records["h3_c_001"]["reply"]["text"] == "服务繁忙，请稍后再试。"
    assert len(teacher.requests) == 8                          # a：写+判；b：两轮各写+判；c：写、带着提醒重问（都失败）
    reminder = teacher.requests[7]["messages"][-1]["content"]     # 重问的那一次：原消息末尾多了一句提醒
    assert reminder == teacher.requests[6]["messages"][-1]["content"] + "\n\n" + write_skills.REMINDER
    assert {(request["model"], request["max_tokens"], request["temperature"]) for request in teacher.requests} == {
        ("teacher-x", 600, 0.5)}
    assert all("这一题没有附上失败尝试" in request["messages"][1]["content"]
               for request in teacher.requests if JUDGE not in request["messages"][0]["content"])
    assert sample_rft.load_skills(output) == {"h3_a_001": render(CLEAN), "h3v_b_001": render(CLEAN)}

    # 第二次：已有定论的不再请求，只重做出错的那道；上次写了一半的坏行被去掉，文件重写成干净的。
    # 教师这时肯给 JSON 了
    teacher.diagnoses["h3_c_001"] = [skill(CLEAN)]
    torn = '{"task_id": "h3_a_001", "status": "ok", "te'
    with output.open("a", encoding="utf-8") as handle:
        handle.write(torn)
    assert write_skills.main(command, client=teacher) == 0
    out, err = capsys.readouterr()
    assert "skills.jsonl 里有 1 行没写完：去掉它们，重写这个文件" in err
    assert "已有定论 2 道，待写 1 道" in out
    assert out.rstrip().splitlines()[-1] == ("结果：技能提示 可用 3/3 道，泄露被拦 0 道，出错 0 道；"
                                             f"教师调用 2 次，输出 {output}")
    assert len(teacher.requests) == 10
    asked = {spec["task_id"] for request in teacher.requests[8:] for spec in specs
             if f"文件 {spec['module']}" in request["messages"][1]["content"]}
    assert asked == {"h3_c_001"}
    written = output.read_text(encoding="utf-8").splitlines()
    records = lines(output)                                     # 每一行都是完整的记录，后面读它的脚本不会报错
    assert [(record["task_id"], record["status"]) for record in records] == [
        ("h3_a_001", "ok"), ("h3v_b_001", "ok"), ("h3_c_001", "error"), ("h3_c_001", "ok")]
    assert not output.with_name("skills.jsonl.tmp").exists()
    assert sample_rft.load_skills(output) == {name: render(CLEAN) for name in ("h3_a_001", "h3v_b_001", "h3_c_001")}

    # 第三次：没有要写的题，不发请求
    assert write_skills.main(command, client=teacher) == 0
    out, _ = capsys.readouterr()
    assert "没有要写的题：3 道都已有定论，不发请求。" in out and len(teacher.requests) == 10
    assert out.rstrip().splitlines()[-1] == ("结果：技能提示 可用 3/3 道，泄露被拦 0 道，出错 0 道；"
                                             f"教师调用 0 次，输出 {output}")
    assert output.read_text(encoding="utf-8").splitlines() == written


def test_cli_resumes_past_a_line_cut_in_the_middle_of_a_character(tmp_path, capsys):
    specs = [task("h3_a_001"), task("h3v_b_001")]
    write_jsonl(tmp_path / "tasks.jsonl", specs)
    write_manifest(tmp_path / "manifest.json", ["h3_a_001", "h3v_b_001"])
    teacher = ScriptedTeacher(specs)
    teacher.diagnoses["h3v_b_001"] = [skill(CLEAN)]
    output = tmp_path / "run/skills.jsonl"
    output.parent.mkdir()
    settled = json.dumps({"task_id": "h3_a_001", "status": "ok", "text": "技能"}, ensure_ascii=False) + "\n"
    torn = '{"task_id": "h3v_b_001", "status": "ok", "text": "'.encode("utf-8") + "技".encode("utf-8")[:2]
    output.write_bytes(settled.encode("utf-8") + torn)           # 上次写到"技"的三个字节中间被中断

    assert write_skills.main([*cli_arguments(tmp_path), "--no-judge"], client=teacher) == 0
    out, err = capsys.readouterr()
    assert "skills.jsonl 里有 1 行没写完" in err and "已有定论 1 道，待写 1 道" in out
    assert out.rstrip().splitlines()[-1] == ("结果：技能提示 可用 2/2 道，泄露被拦 0 道，出错 0 道；"
                                             f"教师调用 1 次，输出 {output}")
    assert [record["task_id"] for record in lines(output)] == ["h3_a_001", "h3v_b_001"]   # 文件已重写成干净的
    assert sample_rft.load_skills(output) == {"h3_a_001": "技能", "h3v_b_001": render(CLEAN)}


def test_cli_keeps_leaky_skills_out_and_still_exits_zero(tmp_path, capsys):
    specs = [task("h3_a_001"), task("h3v_b_001")]
    write_jsonl(tmp_path / "tasks.jsonl", specs)
    write_manifest(tmp_path / "manifest.json", ["h3_a_001", "h3v_b_001"])
    teacher = ScriptedTeacher(specs)
    teacher.diagnoses["h3_a_001"] = [skill(CLEAN)]
    teacher.judges["h3_a_001"] = [verdict(True, "太接近答案")]      # a 每一轮都被自判拦下
    teacher.diagnoses["h3v_b_001"] = [skill({**CLEAN, "approach": "在 clamp 里比较上下界。"})]   # b 每轮都带着名字
    output = tmp_path / "run/skills.jsonl"
    assert write_skills.main([*cli_arguments(tmp_path), "--concurrency", "2"], client=teacher) == 0
    out, err = capsys.readouterr()
    assert out.rstrip().splitlines()[-2:] == ["泄露被拦（不用）：h3_a_001、h3v_b_001",
                                              "结果：技能提示 可用 0/2 道，泄露被拦 2 道，出错 0 道；"
                                              f"教师调用 8 次，输出 {output}"]
    assert err == ""
    records = {record["task_id"]: record for record in lines(output)}
    assert {name: record["status"] for name, record in records.items()} == {"h3_a_001": "leaky", "h3v_b_001": "leaky"}
    assert all("text" not in record and record["rounds"] == 2 for record in records.values())
    assert records["h3_a_001"]["checks"][0]["reconstructable"] is True
    assert records["h3v_b_001"]["checks"][0]["leaks"] == ["clamp"]
    assert sample_rft.load_skills(output) == {}


def test_cli_exits_zero_when_nothing_is_selected(tmp_path, capsys):
    specs = [task("h3_a_001", "dev"), task("demo_d_001")]       # 一道不在 train，一道属于原有题
    write_jsonl(tmp_path / "tasks.jsonl", specs)
    write_manifest(tmp_path / "manifest.json", ["h3_a_001", "demo_d_001"])
    teacher = ScriptedTeacher(specs)
    assert write_skills.main(cli_arguments(tmp_path), client=teacher) == 0
    out, _ = capsys.readouterr()
    output = tmp_path / "run/skills.jsonl"
    assert out.splitlines() == [
        "没有要写的题：上一轮零通过的题里没有属于 train 的 hard,variant 层的题，不发请求。",
        "（清单里属于这些层的题号：['h3_a_001']，它们不在 train 里）",
        f"结果：技能提示 可用 0/0 道，泄露被拦 0 道，出错 0 道；教师调用 0 次，输出 {output}"]
    assert output.exists() and output.read_text(encoding="utf-8") == ""    # 后面的步骤可以直接读
    assert teacher.requests == []


SEPARATORS = "\u2028\u0085\u2029"        # splitlines() 当成换行、json.dumps 却原样写出的字符


def test_parse_skill_json_turns_unicode_line_separators_into_plain_newlines():
    asked = {**CLEAN, "pitfalls": "一\u2028二\u0085三\u2029四"}
    plain = {**CLEAN, "pitfalls": "一\n二\n三\n四"}
    assert write_skills.parse_skill_json(json.dumps(asked, ensure_ascii=False)) == plain
    assert write_skills.render_skill(asked) == render(plain)        # 没经过 parse 的各段也一样
    with pytest.raises(ValueError, match="pitfalls（常见错误）是空的"):          # 只有分隔符的一段等于空的
        write_skills.parse_skill_json(json.dumps({**CLEAN, "pitfalls": " \u2028\u2029 "}, ensure_ascii=False))


def test_a_skill_carries_nothing_that_cuts_a_prompt_record_into_pieces(tmp_path):
    """sample_rft.py 把带提示的消息原样存进它的 raw 文件，判分前再读回来。技能里只要留着 U+2028 这类字符，
    按 splitlines() 读的工具就把那几条采样切成几段（旧版本的 sample_rft.py 读不回来，也就不判分）。所以
    写进记录的技能里不能有它们：不能指望下游的脚本都按 "\\n" 切行。"""
    asked = {**CLEAN, "pitfalls": "一\u2028二\u0085三\u2029四"}
    plain = {**CLEAN, "pitfalls": "一\n二\n三\n四"}
    fake = FakeAsk([json.dumps(asked, ensure_ascii=False)], [verdict(False)])
    record = write_skills.write_skill(task("h3_demo_001"), [], fake)
    assert record["status"] == "ok" and record["parts"] == plain
    assert "【常见错误】一\n二\n三\n四\n【推荐做法】" in record["text"]
    assert not set(SEPARATORS) & set(json.dumps(record, ensure_ascii=False))      # 整条记录（连同各段）里都没有

    spec = task("h3_demo_001")
    messages = hints.with_hint(build_messages(spec), hints.skill_hint(record["text"]))    # 采样时真正发出去的消息
    raw = tmp_path / "train-guided-skill-raw.jsonl"
    raw.write_text(json.dumps({"task_id": spec["task_id"], "candidate_idx": 0, "messages": messages},
                              ensure_ascii=False) + "\n", encoding="utf-8")     # sample_rft.py 存 raw 文件的写法
    assert len(sample_rft.read_jsonl(raw)) == 1                             # 它读 raw 文件的做法
    poison = tmp_path / "poison.jsonl"                                     # 对照：原样写着的这种字符确实会坏事
    poison.write_text(json.dumps({"messages": messages, "note": SEPARATORS}, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    # 按 splitlines() 读的工具（旧版本的 sample_rft.py 就是）把它切成四段；现在的 sample_rft.py 按 "\n" 切行，
    # 读得回来——两头都堵上了，技能里照样不留这种字符
    assert len(poison.read_text(encoding="utf-8").splitlines()) == 4
    assert len(sample_rft.read_jsonl(poison)) == 1


def test_dump_line_keeps_a_record_on_one_line_whatever_line_breaks_it_contains():
    breaks = "\n\r\r\n\v\f\x1c\x1d\x1e\x85\u2028\u2029"       # str.splitlines() 认的全部换行
    record = {"task_id": "h3_a_001", "status": "ok", "reason": "第一行" + breaks + "第二行", "rounds": 1}
    line = write_skills.dump_line(record)
    assert line.endswith("\n") and line.count("\n") == 1 and len(line.splitlines()) == 1
    assert json.loads(line) == record                             # 读回来还是原来的字符
    assert write_skills.dump_line({"note": '中文和"引号"'}) == '{"note": "中文和\\"引号\\""}\n'      # 其余原样可读


def test_cli_keeps_a_skill_with_unicode_line_separators_readable_by_other_scripts(tmp_path, capsys):
    specs = [task("h3_a_001")]
    write_jsonl(tmp_path / "tasks.jsonl", specs)
    write_manifest(tmp_path / "manifest.json", ["h3_a_001"])
    asked = {**CLEAN, "applicability": "第一行\u2028第二行",                 # 教师拿这三个字符当换行
             "principles": "上下\u0085两个边界要分别处理。", "pitfalls": "原样返回输入，\u2029或只处理了一侧边界。"}
    plain = {**CLEAN, "applicability": "第一行\n第二行",                   # 写进记录的技能：分隔符换成了换行
             "principles": "上下\n两个边界要分别处理。", "pitfalls": "原样返回输入，\n或只处理了一侧边界。"}
    teacher = ScriptedTeacher(specs)
    teacher.diagnoses["h3_a_001"] = [json.dumps(asked, ensure_ascii=False)]
    output = tmp_path / "run/skills.jsonl"

    assert write_skills.main(cli_arguments(tmp_path), client=teacher) == 0
    out, err = capsys.readouterr()
    assert out.rstrip().splitlines()[-1] == (f"结果：技能提示 可用 1/1 道，泄露被拦 0 道，出错 0 道；"
                                             f"教师调用 2 次，输出 {output}")
    assert err == ""
    raw = output.read_text(encoding="utf-8")
    assert raw.count("\n") == 1 and len(raw.splitlines()) == 1    # 一条记录就是一行，按哪种切法都一样
    assert not set(SEPARATORS) & set(raw)                          # 正文和各段里都没有这三个字符，用不着转义
    record, = lines(output)
    assert record["status"] == "ok" and record["text"] == render(plain) and record["parts"] == plain
    assert sample_rft.load_skills(output) == {"h3_a_001": render(plain)}     # 按行切的消费方读得到

    assert write_skills.main(cli_arguments(tmp_path), client=teacher) == 0   # 续跑：已有定论，文件不动
    out, err = capsys.readouterr()
    assert "没有要写的题：1 道都已有定论，不发请求。" in out and err == "" and len(teacher.requests) == 2
    assert output.read_text(encoding="utf-8") == raw


@pytest.mark.parametrize("left_behind, kept", [
    ('{"task_id": "h3_a_001", "status": "ok", "text": "旧技能"}', "旧技能"),                 # 末行没有换行
    (json.dumps({"task_id": "h3_a_001", "status": "ok", "text": "旧\u2028技能"}, ensure_ascii=False) + "\n",
     "旧\n技能"),                                                          # 旧写法：技能里原样带着 U+2028
    (json.dumps({"task_id": "h3_a_001", "status": "ok", "text": "旧技能"}, ensure_ascii=False) + "\r\n",
     "旧技能"),                                                           # CRLF 行尾
], ids=["no-final-newline", "raw-separator", "crlf"])
def test_cli_tidies_an_output_file_left_by_other_tools_before_appending(tmp_path, capsys, left_behind, kept):
    specs = [task("h3_a_001"), task("h3v_b_001")]
    write_jsonl(tmp_path / "tasks.jsonl", specs)
    write_manifest(tmp_path / "manifest.json", ["h3_a_001", "h3v_b_001"])
    output = tmp_path / "run/skills.jsonl"
    output.parent.mkdir()
    output.write_bytes(left_behind.encode("utf-8"))                # 另一道题已有定论，文件不是这个脚本写的标准格式
    teacher = ScriptedTeacher(specs)
    teacher.diagnoses["h3v_b_001"] = [skill(CLEAN)]

    assert write_skills.main(cli_arguments(tmp_path), client=teacher) == 0
    out, err = capsys.readouterr()
    assert err == ("skills.jsonl 的格式不标准（行尾缺换行，或内容里带着会被当成换行的字符）："
                   "整理成每行一条记录，重写这个文件\n")
    assert "已有定论 1 道，待写 1 道" in out
    assert out.rstrip().splitlines()[-1] == (f"结果：技能提示 可用 2/2 道，泄露被拦 0 道，出错 0 道；"
                                             f"教师调用 2 次，输出 {output}")
    asked = {spec["task_id"] for request in teacher.requests for spec in specs
             if f"文件 {spec['module']}" in request["messages"][1]["content"]}
    assert asked == {"h3v_b_001"} and len(teacher.requests) == 2     # 已有定论的那道不请求
    raw = output.read_text(encoding="utf-8")
    assert raw.endswith("\n") and "\r" not in raw and not set(SEPARATORS) & set(raw)
    assert [record["task_id"] for record in lines(output)] == ["h3_a_001", "h3v_b_001"]   # 两条记录各占一行
    assert sample_rft.load_skills(output) == {"h3_a_001": kept, "h3v_b_001": render(CLEAN)}
    assert not output.with_name("skills.jsonl.tmp").exists()

    assert write_skills.main(cli_arguments(tmp_path), client=teacher) == 0   # 再跑：文件已经标准，不动它
    out, err = capsys.readouterr()
    assert err == "" and "2 道都已有定论" in out and len(teacher.requests) == 2
    assert output.read_text(encoding="utf-8") == raw


def test_cli_reads_a_graded_file_whose_records_contain_unicode_line_separators(tmp_path, capsys):
    spec = task("h3_demo_001")
    write_jsonl(tmp_path / "tasks.jsonl", [spec])
    write_manifest(tmp_path / "manifest.json", ["h3_demo_001"])
    # sample_rft.py 的写法：模型的原样回答里带着 U+2028，整条记录原样写出
    record = {**graded(BUGGY, "private_test_failure", HIDDEN_IDS), "raw_response": "说明\u2028" + BUGGY}
    (tmp_path / "train-graded.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    teacher = ScriptedTeacher([spec])
    teacher.diagnoses["h3_demo_001"] = [skill(CLEAN)]

    assert write_skills.main([*cli_arguments(tmp_path), "--no-judge"], client=teacher) == 0
    out, err = capsys.readouterr()
    assert "已有定论 0 道，待写 1 道，其中 0 道没有可用的失败尝试。" in out and "跳过没写完" not in err
    request, = teacher.requests
    assert "失败类型：private_test_failure（公开测试通过，隐藏测试没通过）" in request["messages"][1]["content"]
    assert f"```python\n{BUGGY.strip()}\n```" in request["messages"][1]["content"]


def test_cli_feeds_real_failures_to_the_teacher(tmp_path, capsys):
    spec = task("h3_demo_001")
    write_jsonl(tmp_path / "tasks.jsonl", [spec])
    write_manifest(tmp_path / "manifest.json", ["h3_demo_001"])
    build_all(tmp_path / "tasks", [spec])
    teacher = ScriptedTeacher([spec])
    teacher.diagnoses["h3_demo_001"] = [skill(CLEAN)]
    near = FIXED.replace('    if high > ceiling:\n        raise ValueError("ceiling exceeded")\n', "")
    records = []
    for index, code in enumerate((WRONG, BUGGY, near, BROKEN, FIXED)):   # 真的判分，得到真的 graded 记录
        raw = {"text": code, "candidate_idx": index, "temperature": 0.7, "seed": index, "finish_reason": "stop"}
        records.append(sample_rft.grade_sample(spec, raw, "org/model", tmp_path / "tasks", tmp_path / "grade"))
    write_jsonl(tmp_path / "train-graded.jsonl", records)
    assert [record["provenance"]["failure_type"] for record in records] == [
        "public_test_failure", "private_test_failure", "private_test_failure", "syntax_error", None]
    # 有缺陷的版本缺一个被隐藏测试导入的名字：整个文件导入失败；near 只挂一个用例
    assert hidden_tests(records[1]) == [WHOLE_FILE]
    assert records[1]["grader_details"]["layers"]["functional"]["private"]["missing_tests"] == HIDDEN_IDS
    assert hidden_tests(records[2]) == ["tests_private.test_hidden::test_hidden_ceiling"]

    assert write_skills.main([*cli_arguments(tmp_path), "--no-judge"], client=teacher) == 0
    out, err = capsys.readouterr()
    assert "已有定论 0 道，待写 1 道，其中 0 道没有可用的失败尝试。" in out and err == ""
    assert re.search(r"^\[########################\] 1/1 \(100%\) 写技能 h3_demo_001 ok（1 轮）"
                     r" · 已用 \d+\.\dm · 剩余≈\d+\.\dm$", out, re.M)
    request, = teacher.requests
    user = request["messages"][1]["content"]
    assert ("离做对最近的排在前面）\n\n### 失败尝试 1\n"
            "失败类型：private_test_failure（公开测试通过，隐藏测试没通过）\n"
            "没通过的隐藏测试：test_hidden.py（整个文件没能导入）\n"
            f"没能执行的隐藏测试：{'、'.join(HIDDEN_SHOWN)}\n```python\n{BUGGY.strip()}\n```") in user
    assert ("### 失败尝试 2\n失败类型：private_test_failure（公开测试通过，隐藏测试没通过）\n"
            "没通过的隐藏测试：test_hidden.py::test_hidden_ceiling\n"
            f"```python\n{near.strip()}\n```") in user
    assert user.count("没能执行的隐藏测试：") == 1 and "失败尝试 3" not in user   # 公开测试没过的排不上
    assert FIXED.strip() not in user.replace(write_skills.reference_diff(spec), "")   # 通过的不是失败尝试


@pytest.mark.parametrize("arguments, complaint", [
    (["--tiers", "hard,bogus"], "unknown tier"),
    (["--tiers", " "], "unknown tier"),
    (["--concurrency", "0"], "must be positive"),
    (["--max-rounds", "0"], "must be positive"),
    (["--min-interval", "-0.5"], "min-interval cannot be negative"),
    (["--split", "test"], "invalid choice"),
])
def test_cli_rejects_impossible_settings_before_touching_the_teacher(tmp_path, capsys, arguments, complaint):
    write_jsonl(tmp_path / "tasks.jsonl", [task("h3_a_001")])
    write_manifest(tmp_path / "manifest.json", ["h3_a_001"])
    teacher = ScriptedTeacher([task("h3_a_001")])
    with pytest.raises(SystemExit) as caught:
        write_skills.main([*cli_arguments(tmp_path), *arguments], client=teacher)
    assert caught.value.code == 2 and complaint in capsys.readouterr().err
    assert teacher.requests == []


TEACHER_SETTINGS = {"TEACHER_MAX_ROUNDS": "3", "TEACHER_MAX_TOKENS": "4096", "TEACHER_TEMPERATURE": "0.7",
                    "TEACHER_MIN_INTERVAL": "0.25", "TEACHER_TIMEOUT": "30"}


def test_the_teacher_settings_come_from_the_environment_unless_the_command_line_gives_them(tmp_path, capsys,
                                                                                          monkeypatch):
    """--max-rounds、--max-tokens、--min-interval、--temperature、--timeout 没给时取环境变量 TEACHER_MAX_ROUNDS 等
    （同 TEACHER_MODEL；经 run_rft_guided.sh 跑时没有命令行可加，只能这样调，写在 .env 里也行）。命令行给了的
    以命令行为准。"""
    specs = [task("h3_a_001")]
    write_jsonl(tmp_path / "tasks.jsonl", specs)
    write_manifest(tmp_path / "manifest.json", ["h3_a_001"])
    for name, value in {**TEACHER_SETTINGS, "TEACHER_MODEL": "teacher-from-env"}.items():
        monkeypatch.setenv(name, value)

    def run(output, *arguments):
        teacher = ScriptedTeacher(specs)
        code = write_skills.main([*cli_arguments(tmp_path, output), "--concurrency", "1", *arguments], client=teacher)
        return code, capsys.readouterr().out, teacher

    code, out, teacher = run("run/from-env.jsonl")
    assert code == 0 and len(teacher.requests) == 2                       # 一轮：写、判
    assert ("教师：teacher-from-env；每题最多 3 轮，写完自判；每次回复最多 4096 个 token，温度 0.7，"
            "单次请求超时 30 秒") in out                                   # 实际用的是哪一组参数，日志里看得到
    assert all(request["max_tokens"] == 4096 and request["temperature"] == 0.7 for request in teacher.requests)
    assert "同时最多 1 个请求，两次请求至少隔 0.25 秒。" in out

    code, _, teacher = run("run/from-both.jsonl", "--max-tokens", "1500", "--temperature", "0.2", "--max-rounds", "1")
    assert code == 0
    assert {(request["max_tokens"], request["temperature"]) for request in teacher.requests} == {(1500, 0.2)}
    code, out, teacher = run("run/defaults.jsonl", *sum((["--" + name[8:].lower().replace("_", "-"), value]
                                                        for name, value in TEACHER_SETTINGS.items()), []))
    assert code == 0 and "单次请求超时 30 秒" in out                      # 都在命令行上给了：和取环境变量一样
    assert "两次请求至少隔 0.25 秒。" in out

    for name in TEACHER_SETTINGS:
        monkeypatch.setenv(name, "")                                    # 环境变量是空的：用内置的默认值
    code, out, teacher = run("run/built-in.jsonl")
    assert code == 0 and "每题最多 2 轮，写完自判；每次回复最多 4096 个 token，温度 0.3，单次请求超时 120 秒" in out
    assert all(request["max_tokens"] == 4096 and request["temperature"] == 0.3 for request in teacher.requests)


@pytest.mark.parametrize("name, value, complaint", [
    ("TEACHER_MAX_TOKENS", "many", "环境变量 TEACHER_MAX_TOKENS 要是正整数，现在是 'many'"),
    ("TEACHER_MIN_INTERVAL", "soon", "环境变量 TEACHER_MIN_INTERVAL 要是非负数，现在是 'soon'"),
    ("TEACHER_MIN_INTERVAL", "-1", "min-interval cannot be negative"),
    ("TEACHER_MAX_ROUNDS", "0", "must be positive"),
    ("TEACHER_TEMPERATURE", "warm", "环境变量 TEACHER_TEMPERATURE 要是数字，现在是 'warm'"),
    ("TEACHER_TIMEOUT", "-1", "must be positive"),
    ("TEACHER_TIMEOUT", "soon", "环境变量 TEACHER_TIMEOUT 要是数字，现在是 'soon'"),
])
def test_a_teacher_setting_in_the_environment_that_is_not_a_number_stops_before_the_teacher_is_asked(
        tmp_path, capsys, monkeypatch, name, value, complaint):
    write_jsonl(tmp_path / "tasks.jsonl", [task("h3_a_001")])
    write_manifest(tmp_path / "manifest.json", ["h3_a_001"])
    monkeypatch.setenv(name, value)
    teacher = ScriptedTeacher([task("h3_a_001")])
    with pytest.raises(SystemExit) as caught:
        write_skills.main(cli_arguments(tmp_path), client=teacher)
    assert caught.value.code == 2 and complaint in capsys.readouterr().err       # 话里点名是哪个环境变量
    assert teacher.requests == []


def test_the_cut_off_reply_says_what_the_limit_is_and_how_to_raise_it_when_run_through_the_orchestrator():
    asked = json.dumps({**CLEAN, "approach": "一" * 5000}, ensure_ascii=False)

    def ask(messages, max_tokens, temperature):
        return {"text": asked[:-6], "finish_reason": "length", "usage": {"input_tokens": 1, "output_tokens": 1}}

    record = write_skills.write_skill(task("h3_demo_001"), [], ask)
    assert record["status"] == "error" and record["error"].endswith(
        "写技能的回复在写完之前到了输出上限（finish_reason=length，正文 5111 个字符），"
        "把上限调大（--max-tokens，现在是 4096；经 run_rft_guided.sh 跑时设环境变量 TEACHER_MAX_TOKENS），"
        "调大后原样重跑只重做出错的题")
    assert record["reply"]["head_only"] is True and record["reply"]["text"] == asked[:-6][:800]


def test_the_teacher_is_told_which_public_tests_failed(tmp_path):
    """公开测试没过的那份失败尝试：以前只说"公开测试没通过"，列出来的是也跑了的隐藏测试，没通过的公开测试本身
    没有列给教师。"""
    public = {"non_passing_tests": ["tests_public.test_public::test_public", "tests_public.test_public::test_edges"]}
    attempt = graded(WRONG, "public_test_failure", ["tests_private.test_hidden::test_hidden"])
    attempt["grader_details"]["layers"]["functional"]["public"] = public
    brief = write_skills.attempt_brief(attempt)
    assert brief["failing_public"] == ["test_public.py::test_public", "test_public.py::test_edges"]
    user = write_skills.diagnosis_messages(task("h3_demo_001"), [attempt])[1]["content"]
    assert ("失败类型：public_test_failure（公开测试没通过）\n"
            "没通过的隐藏测试：test_hidden.py::test_hidden\n"
            "没通过的公开测试：test_public.py::test_public、test_public.py::test_edges\n"
            f"```python\n{WRONG.strip()}\n```") in user
    # 公开层过了的：没有这一行；判分记录里没记公开层（旧版本、或者超时）的：也没有
    passing = graded(BUGGY, "private_test_failure", ["tests_private.test_hidden::test_hidden"])
    passing["grader_details"]["layers"]["functional"]["public"] = {"non_passing_tests": []}
    assert "没通过的公开测试" not in write_skills.diagnosis_messages(task("h3_demo_001"), [passing])[1]["content"]
    assert write_skills.attempt_brief(graded(WRONG, "public_test_failure", []))["failing_public"] == []


def run_cli(environment, *arguments):
    return subprocess.run([sys.executable, str(ROOT / "scripts/write_skills.py"), *arguments],
                          capture_output=True, text=True, env=environment, timeout=180)


def fake_sdk(tmp_path):
    """把一份假的 openai SDK 放到 PYTHONPATH 最前面；返回给脚本用的环境变量、回复脚本和请求日志的路径。"""
    (tmp_path / "sdk").mkdir(exist_ok=True)
    (tmp_path / "sdk/openai.py").write_text(FAKE_SDK, encoding="utf-8")
    script, log = tmp_path / "teacher.json", tmp_path / "teacher.log"
    return {"TEACHER_API_KEY": "offline-key", "TEACHER_BASE_URL": "https://teacher.example/v1",
            "TEACHER_MODEL": "teacher-from-env", "PYTHONPATH": str(tmp_path / "sdk"),
            "FAKE_TEACHER_SCRIPT": str(script), "FAKE_TEACHER_LOG": str(log)}, script, log


def test_command_line_runs_through_the_real_entry_point_and_resumes(tmp_path):
    specs = [task(name) for name in ("h3_a_001", "h3v_b_001", "h3_c_001")]
    write_jsonl(tmp_path / "tasks.jsonl", specs)
    write_manifest(tmp_path / "manifest.json", [spec["task_id"] for spec in specs])
    extra, script, log = fake_sdk(tmp_path)
    environment = {**os.environ, **extra}
    script.write_text(json.dumps({"h3_a_001": [skill(CLEAN)], "h3v_b_001": [skill(LEAKY), skill(CLEAN)],
                                  "h3_c_001": ["服务繁忙，请稍后再试。"]}, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "skills.jsonl"
    arguments = ["--tasks-file", str(tmp_path / "tasks.jsonl"), "--zero-pass-from", str(tmp_path / "manifest.json"),
                 "--output", str(output), "--concurrency", "3", "--timeout", "45"]

    broken = run_cli(environment, *arguments)                  # 有一道题的回复不是 JSON：退出码 3
    assert broken.returncode == 3, broken.stderr
    assert "Traceback" not in broken.stderr
    assert f"h3_c_001 出错：{UNUSABLE}" in broken.stderr
    assert broken.stdout.rstrip().splitlines()[-1] == ("结果：技能提示 可用 2/3 道，泄露被拦 0 道，出错 1 道；"
                                                       f"教师调用 8 次，输出 {output}")
    entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert entries[0] == {"client": {"api_key": "offline-key", "base_url": "https://teacher.example/v1",
                                     "timeout": 45.0, "max_retries": 0}}
    assert len(entries) == 9 and {entry["model"] for entry in entries[1:]} == {"teacher-from-env"}
    assert all(entry["max_tokens"] == 4096 for entry in entries[1:])     # 默认的输出上限
    failing = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
               if json.loads(line)["task_id"] == "h3_c_001"][0]
    assert failing["reminders"] == 1 and failing["reply"]["text"] == "服务繁忙，请稍后再试。"

    script.write_text(json.dumps({name: [skill(CLEAN)] for name in ("h3_a_001", "h3v_b_001", "h3_c_001")},
                                 ensure_ascii=False), encoding="utf-8")          # 教师恢复正常
    log.unlink()
    resumed = run_cli(environment, *arguments)                 # 只重做出错的那道：退出码 0
    assert resumed.returncode == 0, resumed.stderr
    assert resumed.stdout.rstrip().splitlines()[-1] == ("结果：技能提示 可用 3/3 道，泄露被拦 0 道，出错 0 道；"
                                                        f"教师调用 2 次，输出 {output}")
    assert len(log.read_text(encoding="utf-8").splitlines()) == 3   # 建客户端一行，加上那道题的写和判
    asked = {spec["task_id"] for entry in map(json.loads, log.read_text(encoding="utf-8").splitlines()[1:])
             for spec in specs if f"文件 {spec['module']}" in entry["messages"][1]["content"]}
    assert asked == {"h3_c_001"}
    assert sample_rft.load_skills(output) == {spec["task_id"]: render(CLEAN) for spec in specs}


def test_command_line_refuses_to_start_without_a_key_or_without_the_sdk(tmp_path):
    specs = [task("h3_a_001")]
    write_jsonl(tmp_path / "tasks.jsonl", specs)
    write_manifest(tmp_path / "manifest.json", ["h3_a_001"])
    extra, _, log = fake_sdk(tmp_path)
    arguments = ["--tasks-file", str(tmp_path / "tasks.jsonl"), "--zero-pass-from", str(tmp_path / "manifest.json"),
                 "--output", str(tmp_path / "skills.jsonl")]
    without_key = {key: value for key, value in {**os.environ, **extra}.items() if key != "TEACHER_API_KEY"}
    refused = run_cli(without_key, *arguments)
    assert refused.returncode == 2 and "TEACHER_API_KEY" in refused.stderr and ".env" in refused.stderr
    assert "Traceback" not in refused.stderr and not log.exists()

    (tmp_path / "sdk/openai.py").write_text("raise ImportError('simulated: openai is not installed')\n",
                                            encoding="utf-8")
    absent = run_cli({**os.environ, **extra}, *arguments)
    assert absent.returncode == 2 and "openai SDK" in absent.stderr and "Traceback" not in absent.stderr
    output = tmp_path / "skills.jsonl"
    assert output.exists() and output.read_text(encoding="utf-8") == ""    # 没有请求，也没有记录


def test_help_needs_neither_the_key_nor_the_sdk():
    without_key = {key: value for key, value in os.environ.items() if key != "TEACHER_API_KEY"}
    helped = run_cli(without_key, "--help")
    assert helped.returncode == 0, helped.stderr
    assert "让教师给" in helped.stdout and "--no-judge" in helped.stdout and "--zero-pass-from" in helped.stdout
    # 默认值写得保守：并发 2、输出上限 4096、请求之间隔 1 秒
    assert "（默认 2）" in helped.stdout and "默认 4096" in helped.stdout
    assert "默认 1；环境变量 TEACHER_MIN_INTERVAL" in helped.stdout


def test_the_defaults_are_conservative_and_the_shared_throttle_follows_the_command_line(
        tmp_path, capsys, monkeypatch):
    """服务器上实测：并发 4 时成片 429；900 个 token 时成片报"没有 JSON 对象"，多半是没写完。所以默认并发 2、
    输出上限 4096、两次请求之间隔 1 秒；跑起来以后，进程里共用的那份节流按这一次的 --concurrency、
    --min-interval 来定。"""
    for name in ("TEACHER_MAX_TOKENS", "TEACHER_MIN_INTERVAL"):
        monkeypatch.setenv(name, "")                     # 空的：用内置的默认值
    write_jsonl(tmp_path / "tasks.jsonl", [task("h3_a_001")])
    write_manifest(tmp_path / "manifest.json", ["h3_a_001"])
    teacher = ScriptedTeacher([task("h3_a_001")])
    teacher.diagnoses["h3_a_001"] = [skill(CLEAN)]
    assert write_skills.main(cli_arguments(tmp_path), client=teacher) == 0     # 什么都不给
    assert (shared.THROTTLE.max_in_flight, shared.THROTTLE.min_interval) == (2, 1.0)
    assert "同时最多 2 个请求，两次请求至少隔 1 秒。" in capsys.readouterr().out
    assert {request["max_tokens"] for request in teacher.requests} == {4096}      # 没给 --max-tokens：新的默认值
    # 给了就以给的为准
    assert write_skills.main([*cli_arguments(tmp_path, "run/again.jsonl"), "--concurrency", "3",
                              "--min-interval", "0.5"], client=teacher) == 0
    assert (shared.THROTTLE.max_in_flight, shared.THROTTLE.min_interval) == (3, 0.5)
    assert "同时最多 3 个请求，两次请求至少隔 0.5 秒。" in capsys.readouterr().out


def test_an_extra_json_that_is_not_an_object_stops_before_the_teacher_is_asked(tmp_path, capsys, monkeypatch):
    write_jsonl(tmp_path / "tasks.jsonl", [task("h3_a_001")])
    write_manifest(tmp_path / "manifest.json", ["h3_a_001"])
    teacher = ScriptedTeacher([task("h3_a_001")])
    for value, piece in (("{bad", "现在不是合法的 JSON"), ("[1, 2]", "现在解析出来是list")):
        monkeypatch.setenv("TEACHER_EXTRA_JSON", value)
        with pytest.raises(SystemExit) as caught:
            write_skills.main(cli_arguments(tmp_path), client=teacher)
        shown = capsys.readouterr().err
        assert caught.value.code == 2 and "环境变量 TEACHER_EXTRA_JSON 要是一个 JSON 对象" in shown and piece in shown
    assert teacher.requests == [] and not (tmp_path / "run/skills.jsonl").exists()


def test_extra_json_reaches_the_teacher_through_the_real_entry_point(tmp_path):
    specs = [task("h3_a_001")]
    write_jsonl(tmp_path / "tasks.jsonl", specs)
    write_manifest(tmp_path / "manifest.json", ["h3_a_001"])
    extra, script, log = fake_sdk(tmp_path)
    script.write_text(json.dumps({"h3_a_001": [skill(CLEAN)]}), encoding="utf-8")
    environment = {**os.environ, **extra, "TEACHER_EXTRA_JSON": '{"sample_flag": true, "width": 3}'}
    run_cli(environment, *cli_arguments(tmp_path), "--no-judge")
    entry, = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()][1:]
    assert (entry["sample_flag"], entry["width"], entry["max_tokens"]) == (True, 3, 4096)
    assert {"model", "messages", "temperature"} <= set(entry)
