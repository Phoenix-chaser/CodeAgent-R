"""让教师给"基座一次都没做对的题"各写一条通用的修复技能，给 sample_rft.py --hint-kind skill 当提示。

用法：
    python scripts/write_skills.py --tasks-file 题文件 --zero-pass-from 上一轮的 train-manifest.json \\
        --split train --tiers hard,variant --graded 上一轮的 train-graded.jsonl --output skills.jsonl

选题是三个集合的交集：这个 split、这些层、上一轮一次都没通过的题（清单里的 zero_pass_tasks）。教师看
题目要求、有缺陷的文件、参考修复的 diff 和最多两份基座的失败尝试（失败类型、没通过的公开测试和隐藏测试、
代码），写一条四段的技能：适用条件、要点、常见错误、推荐做法。

技能必须是通用的方法性说明。依据 FailForge（arXiv:2608.08570）：泄露了答案的技能，训出来的模型更差。
所以每写一版都要过两道检查，没过就带着意见让教师重写：
- 静态检查：技能里不能出现这道题的名字（整词匹配）——参考修复里的函数名、类名、参数名、模块级赋值名和
  长度不小于 4 的字符串字面量，测试函数名，文件名。这份名字的清单第一轮就给教师，省得泄露一次才知道；
- 自判（--no-judge 关掉）：把技能、题目和参考修复交给教师，问"只凭这条技能加上题目，能否基本重建
  参考修复里的具体代码"。

输出每行一道题，追加写，同一道题以最后一行为准（续跑时先去掉上次没写完的坏行、把不标准的行尾整理好）。
技能里带的 U+2028 这类会被当成换行的字符写成转义，读回来不变——这样按行切的脚本（sample_rft.py 的
--skills-file）不会把一条记录切成几段。status：ok 技能可用；leaky 重写到上限仍没通过检查，弃用；
error 请求或解析出错，原样重跑会重做。
退出码：0 全部有定论，3 还有 error，2 参数或环境不对。

教师接口在服务器上实测出的两个毛病，这里都防着（重试和节流本身在 _shared_teacher.py）：
- 限额按账户算，并发 4 时成片 429。所以 --concurrency 默认 2（也是同一时刻在飞的请求上限），两次请求之间
  至少隔 --min-interval 秒（默认 1）；429 和 5xx 长时间重试，不急着记成 error；
- 教师先写一大段推理再给 JSON。每次回复 900 个 token 时成片报"回复里没有 JSON 对象"（出错的记录没留回复，
  原因无法确证，最可能是没写完就到了上限）。所以 --max-tokens 默认 4096。
回复不成 JSON 时：被截断的（finish_reason 是 length，或者正文是空的）直接报"回复在写完之前到了输出上限"和
该调大的环境变量，不报成"没有 JSON 对象"；正文有、可是没有可解析的 JSON 对象的，在原消息末尾加一句提醒自动
重问一次。最后还是不行的，出错的记录里留下原始回复的前 800 个字符、结束原因、用量和推理字段的长度（不记
推理内容，也不记题目），事后查得到原因。

--max-rounds、--max-tokens、--min-interval、--temperature、--timeout 没给时，依次取环境变量 TEACHER_MAX_ROUNDS、
TEACHER_MAX_TOKENS、TEACHER_MIN_INTERVAL、TEACHER_TEMPERATURE、TEACHER_TIMEOUT（同 TEACHER_MODEL、
TEACHER_BASE_URL；没有就是内置的默认值 2、4096、1、0.3、120）。经 run_rft_guided.sh 跑时只能靠这几个环境
变量调：调大以后原样重跑只重做出错的题。接口特有的参数写进环境变量 TEACHER_EXTRA_JSON（一个 JSON 对象），
原样合并进每次请求。sample_rft.py --skills-file 只用 status 为 ok 的行；技能只进采样的提示，训练记录里不含技能。
"""
from __future__ import annotations

import argparse
import ast
import difflib
import json
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agentlab.benchmark.protocol import build_messages, load_specs
from agentlab.benchmark.tiers import TIERS, tier_of
from agentlab.sampling import vllm_client
from _shared_teacher import KEY_ENV, chat_text, configure_throttle, extra_json, make_client, teacher_model

# 四段的键名、小标题和"这一段写什么"（写进给教师的系统提示）
SECTIONS = (
    ("applicability", "适用条件", "遇到什么现象、什么类型的缺陷时该用这条技能"),
    ("principles", "要点", "解决这类问题要抓住的原理、不变量或性质"),
    ("pitfalls", "常见错误", "模型在这类问题上容易犯的错，尤其是上面失败尝试里暴露出来的"),
    ("approach", "推荐做法", "按步骤给出的做法，每一步带一个可以自己核对的检查点"),
)
MIN_IDENTIFIER = 4            # 更短的名字（low、run）太普通，拦了只会误伤，不算泄露
ATTEMPT_LINES = 120           # 给教师看的每份失败尝试最多多少行
SETTLED = ("ok", "leaky")     # 有了这两种定论就不再重写；error 原样重跑时重做
NEAR_MISS = "private_test_failure"     # 公开测试过了、隐藏测试没过：离做对最近，优先给教师看
FAILURE_NOTES = {
    "private_test_failure": "公开测试通过，隐藏测试没通过", "public_test_failure": "公开测试没通过",
    "invalid_patch": "补丁不合法", "incomplete_response": "输出没写完", "execution_timeout": "执行超时",
    "thinking_unfinished": "推理没写完",
}
# Python 把这三个字符也当成换行（str.splitlines() 在它们处切行），json.dumps(ensure_ascii=False) 却把它们
# 原样写出来。技能里带着一个，存提示或存记录的 JSONL 文件就会被按行切的脚本切坏，所以技能正文里不能有
# 它们（plain_breaks），写记录时也要转义（dump_line）。
LINE_SEPARATORS = ("\u2028", "\u2029", "\u0085")
DEFAULT_MAX_TOKENS = 4096      # 教师先写一大段推理再给 JSON：上限 900 时成片的回复解析不了，多半是没写完就到了上限
REPLY_EXCERPT = 800            # 出错的记录里留原始回复的前多少个字符：看得出它卡在哪，又不把文件撑大
# 正文有、可是没有可解析的 JSON 对象时，重问那一次在原消息末尾加的一句
REMINDER = "上一次的回复里没有可解析的 JSON 对象，请只输出一个 JSON 对象，不要任何别的文字。"

DIAGNOSIS_SYSTEM = """你是一位代码修复的导师。用户会给你一道代码修复题：题目要求、有缺陷的代码、参考修复，\
以及模型之前的几份失败尝试。请从这道题归纳出一条通用的修复技能，让模型以后遇到同类问题时用得上。

技能会放进提示里，让模型带着它重新解这道题。它必须是通用的方法性说明：把答案写进技能看起来更有用，\
实际上泄露了答案的技能训出来的模型更差（FailForge，arXiv:2608.08570）。因此有五条硬性要求：
1. 讲方法和检查点：从哪里入手、按什么顺序分析、要抓住什么性质、每一步怎样确认做对了。
2. 不出现任何具体的名字：函数名、类名、参数名、变量名、测试名、文件名，以及代码里的字面量（数字、字符串）。\
用户会列出这道题的这份清单，其中不少看起来只是普通的英文单词，也一样不要用；需要指代时用"该操作""这条要求"\
"这个返回值"一类的说法，整段用中文写。
3. 不出现代码，也不要写成"把某一处改成什么"。只说做什么、为什么这样做；改哪里、怎么写留给模型自己分析。
4. 四段都用中文，每段 1 到 3 句，合起来是一份不依赖这道题也能读懂的方法说明。
5. 只输出一个 JSON 对象，不要解释，不要 markdown 代码块标记。键名和每段写什么如下：
""" + "\n".join(f'   "{key}"（{title}）：{hint}。' for key, title, hint in SECTIONS) + """
请只输出这个 JSON 对象。"""

JUDGE_SYSTEM = """你在审核一条给代码修复模型用的技能。合格的技能讲的是通用的方法和检查点：\
模型读了以后仍然要自己分析题目、自己写代码。不合格的技能实际上是这道题的答案：\
把具体改哪里、改成什么结构、用哪些具体的量都写出来了。

你会看到题目（模型看到的内容）、这条技能，以及参考修复（从有缺陷的代码到修复后的 unified diff）。
请回答一个问题：只凭这条技能加上题目，能否基本重建参考修复里的具体代码——也就是说，模型不必自己分析，
照着技能就能写出那段代码？

- 技能只是指出了方向、顺序和要检查的性质：即使它让这道题变容易了，也算不能重建，reconstructable 为 false。
- 技能把这道题的具体改法写得足够具体，照着就能拼出参考修复里的代码：reconstructable 为 true。

只输出一个 JSON 对象，不要输出其它文字：{"reconstructable": true 或 false, "reason": "一句话说明理由"}"""


def parses(code: str) -> bool:
    """代码是完整的、能解析的 Python。"""
    try:
        return bool(code) and bool(ast.parse(code).body)
    except (SyntaxError, ValueError):
        return False


def _module_level_assignments(tree: ast.Module) -> set[str]:
    """参考修复里模块层的赋值名。顶层的 if、try、循环在导入时就执行，里面的赋值也算；函数和类里的不算。"""
    names = set()
    pending = list(tree.body)
    while pending:
        statement = pending.pop()
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(statement, ast.Assign):
            targets = statement.targets
        elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
            targets = [statement.target]
        else:
            targets = []
        names.update(node.id for target in targets for node in ast.walk(target) if isinstance(node, ast.Name))
        for field in ("body", "orelse", "finalbody"):
            pending.extend(getattr(statement, field, None) or [])
        for handler in getattr(statement, "handlers", None) or ():
            pending.extend(handler.body)
        for case in getattr(statement, "cases", None) or ():
            pending.extend(case.body)
    return names


def _reference_identifiers(source: str) -> set[str]:
    """参考修复里的名字：函数名、类名、函数参数名、模块级赋值名、字符串字面量。"""
    names = set()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            arguments = node.args
            names.update(argument.arg for argument in
                         (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs))
            names.update(argument.arg for argument in (arguments.vararg, arguments.kwarg) if argument is not None)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.add(node.value)
    names.update(_module_level_assignments(tree))
    return names


def _test_identifiers(source: str) -> set[str]:
    """测试代码里的测试函数名和测试类名（pytest 默认收集 test_* 和 Test*）。夹具和辅助函数不是题目的一部分，不算。"""
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            names.add(node.name)
    return names


def code_identifiers(spec: dict) -> set[str]:
    """这道题的技能里不许出现的名字。

    来源：参考修复里的函数名、类名、函数参数名、模块级赋值名、长度不小于 4 的字符串字面量；公开测试和
    隐藏测试里的测试函数名；主文件名（去掉 .py）；隐藏测试的文件名（去掉 .py）。长度小于 4 的和
    self、cls 不要。哪份源码解析失败就跳过它：宁可少拦一项，也不能因为一道题的格式问题整轮跑不下去。
    """
    sources = [(spec.get("fixed"), _reference_identifiers), (spec.get("public_test"), _test_identifiers)]
    sources += [(content, _test_identifiers) for content in (spec.get("private_tests") or {}).values()]
    names = set()
    for source, collect in sources:
        try:
            names.update(collect(source))
        except (SyntaxError, ValueError, TypeError):
            continue
    names.add(str(spec.get("module", "")).removesuffix(".py"))
    names.update(str(name).removesuffix(".py") for name in (spec.get("private_tests") or {}))
    return {name for name in names
            if len(name) >= MIN_IDENTIFIER and name.strip() and name not in {"self", "cls"}}


_WORD = re.compile(r"[A-Za-z0-9_]")


def _appears(text: str, name: str) -> bool:
    """``name`` 有没有作为一个完整的名字出现在 ``text`` 里。"""
    pattern = re.escape(name)
    # 整词匹配：名字的两边不能再紧挨字母、数字或下划线。不能用 \b：在它眼里汉字也是单词字符，
    # "调用clamp函数"里的 clamp 就找不到边界。以点、引号等开头的字面量不加左边这条要求。
    if _WORD.match(name[0]):
        pattern = r"(?<![A-Za-z0-9_])" + pattern
    if _WORD.fullmatch(name[-1]):
        pattern += r"(?![A-Za-z0-9_])"
    return re.search(pattern, text) is not None


def static_leaks(text: str, identifiers) -> list[str]:
    """技能里出现了哪些不许出现的名字（按字典序，便于核对）。

    整词匹配。参考修复的参数名常常就是普通英文单词（timeout、data、sync）：按子串找，async、database、
    condition 这些只是"包含"了名字的词也会被当成泄露；而名字紧贴着汉字写（"调用clamp函数"）照样找得到。
    """
    return sorted(name for name in identifiers if name and _appears(text, name))


# JSON 字符串（两个没被反斜杠转义的引号之间），或者被当成排版换行的三个字符里的一个
_STRING_OR_BREAK = re.compile(r'"(?:[^"\\]|\\.)*"|[\u2028\u2029\u0085]', re.S)


def _layout_breaks(text: str) -> str:
    """把 JSON 字符串外面的 U+2028、U+2029、U+0085 换成换行；字符串里面的不动。

    ``text`` 要从 JSON 对象的 "{" 起：从那里扫起，才知道哪些引号是字符串的开头和结尾（对象前面的说明文字里
    可能带着不成对的引号）。
    """
    return _STRING_OR_BREAK.sub(lambda match: "\n" if match.group()[0] != '"' else match.group(), text)


def first_json_object(text: str) -> dict:
    """回复里的第一个 JSON 对象。外面包着说明文字或 markdown 围栏都行。

    从每个 "{" 起试着解码，解不出（说明文字里的花括号、写坏的 JSON）就看下一个。字符串里带换行也接受。

    原样解不出时，把 JSON 字符串外面的 U+2028、U+2029、U+0085 换成换行再试一次。教师有时把它们当排版换行
    用在花括号后面、键与键之间，而 JSON 只把空格、制表符和 "\\n"、"\\r" 当空白：原样解不出来，这道题就被
    记成 error，重跑多半还是这样。字符串里面的不动，取出来以后各段照旧过 plain_breaks（自判的理由不过，原样留着）。
    """
    decoder = json.JSONDecoder(strict=False)
    position = text.find("{")
    while position != -1:
        for source in (text, text[:position] + _layout_breaks(text[position:])):
            try:
                value, _ = decoder.raw_decode(source, position)
            except (ValueError, RecursionError):
                continue
            if isinstance(value, dict):
                return value
        position = text.find("{", position + 1)
    raise ValueError("回复里没有 JSON 对象")


def plain_breaks(text: str) -> str:
    """把 U+2028、U+2029、U+0085 这三个分隔符换成换行，其余不动。

    技能会进提示，sample_rft.py 把带提示的消息原样存进它自己的 raw 文件，读回来再判分。按 splitlines()
    读的脚本会把带着这种字符的一条记录切成几段（旧版本的 sample_rft.py 就是这样，那几条采样读不回来
    也不判分，退出码还是 0；它现在按 "\\n" 切行、写的时候转义，别的按行切的工具还是会切坏）。技能里不留
    它们最稳：它们本来就是换行的意思，换成换行对模型读起来没有区别。
    """
    for char in LINE_SEPARATORS:
        text = text.replace(char, "\n")
    return text


def parse_skill_json(text: str) -> dict:
    """从教师的回复里取出四段技能。四个键都必须是非空字符串，否则 ValueError，并说明是哪一项不对。

    各段里的 U+2028 这类分隔符换成换行（plain_breaks），所以返回的各段里不会有它们。
    """
    value = first_json_object(text)
    for key, title, _ in SECTIONS:
        if key not in value:
            raise ValueError(f"JSON 里没有 {key}（{title}）这一项")
        if not isinstance(value[key], str):
            raise ValueError(f"{key}（{title}）不是字符串，而是 {type(value[key]).__name__}")
        value[key] = plain_breaks(value[key])
        if not value[key].strip():
            raise ValueError(f"{key}（{title}）是空的")
    return value


def parse_judge_json(text: str) -> tuple[bool, str]:
    """自判的回复：(能不能重建参考修复, 理由)。"""
    value = first_json_object(text)
    verdict = value.get("reconstructable")
    if isinstance(verdict, str):
        verdict = {"true": True, "false": False}.get(verdict.strip().lower())
    if not isinstance(verdict, bool):
        raise ValueError("自判的回复里没有 reconstructable（true 或 false）")
    reason = value.get("reason")
    return verdict, "" if reason is None else str(reason)


def render_skill(parts: dict) -> str:
    """四段拼成给模型看的文本，每段带【】小标题。缺任何一段都不能用：残缺的提示比没有提示更误导。

    各段先过一遍 plain_breaks：不管各段是从哪儿来的，给模型的正文里都不会有 U+2028 这类分隔符。
    """
    pieces = []
    for key, title, _ in SECTIONS:
        body = plain_breaks(parts.get(key) or "").strip()
        if not body:
            raise ValueError(f"技能的 {title}（{key}）是空的")
        pieces.append(f"【{title}】{body}")
    return "\n".join(pieces)


def reference_diff(spec: dict) -> str:
    """从有缺陷的文件到参考修复的 unified diff。"""
    old = [line + "\n" for line in spec["buggy"].splitlines()]
    new = [line + "\n" for line in spec["fixed"].splitlines()]
    return "".join(difflib.unified_diff(old, new, fromfile=f"a/{spec['module']}", tofile=f"b/{spec['module']}"))


def _test_names(layer: dict, field: str, whole_file: str) -> list[str]:
    """把判分器记的用例名换成教师读得懂的写法。

    判分器（grader.parse_junit）记的是 junit 的 <classname>::<name>，classname 是点号隔的内部路径，例如
    tests_private.test_hidden::test_hidden。整个文件导入失败时只记一条 "::tests_private.test_hidden"，
    一个用例都没执行——这时去掉内部路径、说明是整个文件，教师才知道该往"缺了被导入的名字"上想。
    """
    names = []
    for node in layer.get(field) or []:
        classname, separator, name = str(node).rpartition("::")
        if not separator:
            names.append(str(node))
        elif classname:
            names.append(f"{classname.rsplit('.', 1)[-1]}.py::{name}")
        else:
            names.append(whole_file.format(file=name.rsplit(".", 1)[-1]))
    return names


def attempt_brief(record: dict) -> dict:
    """一份失败尝试里给教师看的东西（``record`` 是 sample_rft.py 的 graded 记录）。

    kind 失败类型；failing 没通过的隐藏测试；missing 没能执行的隐藏测试（通常是整个文件导入失败）；
    failing_public 没通过的公开测试（公开层过了就是空的）；code 代码，超过 ATTEMPT_LINES 行时只给前面的，
    这时 truncated 为真。
    """
    functional = ((record.get("grader_details") or {}).get("layers") or {}).get("functional") or {}
    private, public = functional.get("private") or {}, functional.get("public") or {}
    lines = ((record.get("target") or {}).get("content") or "").splitlines()
    return {
        "kind": (record.get("provenance") or {}).get("failure_type"),
        "failing": _test_names(private, "non_passing_tests", "{file}.py（整个文件没能导入）"),
        "missing": _test_names(private, "missing_tests", "{file}.py"),
        "failing_public": _test_names(public, "non_passing_tests", "{file}.py（整个文件没能导入）"),
        "code": "\n".join(lines[:ATTEMPT_LINES]), "truncated": len(lines) > ATTEMPT_LINES,
    }


def _attempts_text(attempts) -> str:
    if not attempts:
        return "## 模型的失败尝试\n\n这一题没有附上失败尝试。"
    pieces = []
    for index, record in enumerate(attempts, 1):
        brief = attempt_brief(record)
        note = f"（{FAILURE_NOTES[brief['kind']]}）" if brief["kind"] in FAILURE_NOTES else ""
        lines = [f"### 失败尝试 {index}", f"失败类型：{brief['kind'] or '未知'}{note}",
                 f"没通过的隐藏测试：{'、'.join(brief['failing']) or '（没有记录）'}"]
        if brief["failing_public"]:      # 公开测试没过的那几条：只说"公开测试没通过"，教师不知道是哪一条
            lines.append(f"没通过的公开测试：{'、'.join(brief['failing_public'])}")
        if brief["missing"]:
            lines.append(f"没能执行的隐藏测试：{'、'.join(brief['missing'])}")
        lines.append(f"```python\n{brief['code']}\n```")
        if brief["truncated"]:
            lines.append(f"（代码超过 {ATTEMPT_LINES} 行，只给出前 {ATTEMPT_LINES} 行）")
        pieces.append("\n".join(lines))
    return "## 模型的失败尝试（离做对最近的排在前面）\n\n" + "\n\n".join(pieces)


def _forbidden_text(spec: dict) -> str:
    """把不许出现的名字直接列给教师。第一轮就给：不必等泄露了一次，教师才知道哪些词不能用。"""
    names = sorted(code_identifiers(spec))
    if not names:
        return ""
    return ("\n\n## 技能里不许出现的名字\n\n下面是这道题里的名字和字面量，技能里一个都不要出现——"
            "其中不少看起来只是普通的英文单词，也一样不要用。需要指代时用\"该操作\"\"这条要求\"一类的说法：\n"
            + "、".join(names))


def diagnosis_messages(spec: dict, attempts, feedback: str | None = None) -> list[dict]:
    """请教师写技能的消息。``attempts`` 是 pick_attempts 挑出的 graded 记录；``feedback`` 非空时附在最后。"""
    user = f"## 题目要求\n\n{spec['instruction']}\n\n## 有缺陷的文件 {spec['module']}\n\n"
    user += f"```python\n{spec['buggy'].rstrip()}\n```\n\n"
    user += ("## 参考修复\n\n从有缺陷的文件到修复后的 unified diff。只供诊断，不要写进技能。\n\n"
             f"```diff\n{reference_diff(spec)}```\n\n")
    user += _attempts_text(attempts)
    user += _forbidden_text(spec)
    if feedback:
        user += f"\n\n## 上一版的问题\n\n{feedback}"
    return [{"role": "system", "content": DIAGNOSIS_SYSTEM}, {"role": "user", "content": user}]


def judge_messages(skill_text: str, spec: dict) -> list[dict]:
    """请教师自判的消息：题目（模型看到的内容）、技能、参考修复。"""
    user = (f"## 题目\n\n{build_messages(spec)[1]['content']}\n\n## 技能\n\n{skill_text}\n\n"
            f"## 参考修复\n\n```diff\n{reference_diff(spec)}```")
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def revision_feedback(text: str, check: dict) -> str:
    """下一轮附在诊断后面的"上一版的问题"：上一版的原文，加上每道检查的意见。

    接口不带上下文，教师并不记得上一版，所以原文要一起带上。
    """
    lines = [f"上一版技能：\n\n{text}", "上一版的问题："]
    if check["leaks"]:
        lines.append(f"- 出现了这道题里的具体名字或字面量：{'、'.join(check['leaks'])}。技能里不能出现任何函数名、"
                     "类名、参数名、变量名、测试名、文件名和字面量；需要指代时用\"该操作\"\"这条要求\"一类的说法。")
    if check["reconstructable"]:
        lines.append("- 它已经接近答案" + (f"（{check['reason']}）" if check["reason"] else "")
                     + "。请上升一步：先说这一类问题的判断依据，再给通用的步骤和检查点，不要落到这道题的具体改法上。")
    lines.append("请重新写一条，五条硬性要求都要满足。")
    return "\n".join(lines)


def redact(text: str, secret: str | None) -> str:
    """把 ``text`` 里出现的密钥换成 ***。出错信息和教师的回复都要记进文件：万一异常的消息里带着密钥，不能落盘。
    密钥不到 8 个字符就不替换：那多半是测试用的占位值，太短的话到处都碰得到。"""
    return text.replace(secret, "***") if secret and len(secret) >= 8 else text


def with_reminder(messages: list[dict], reminder: str = REMINDER) -> list[dict]:
    """原消息不动，在最后一条的末尾加一句提醒：重问的时候用。

    加在最后一条（用户的那条）里面，不另起一条用户消息：消息的结构照旧（系统提示一条、用户消息一条），
    不用赌接口认不认同一个角色连着两条。
    """
    last = messages[-1]
    return [*messages[:-1], {**last, "content": f"{last['content']}\n\n{reminder}"}]


class Reply:
    """教师的一条回复：正文、结束原因、用量，以及正文以外的文字字段（推理）各有多长。注入的 ask 给字符串也行。"""

    def __init__(self, text: str = "", finish_reason=None, usage=None, other=None):
        self.text = text or ""
        self.finish_reason, self.usage, self.other = finish_reason, usage, other or {}

    @classmethod
    def of(cls, reply) -> "Reply":
        """ask 的返回值：chat_text 那样的字典，或者就是正文的字符串。"""
        if isinstance(reply, dict):
            return cls(reply.get("text"), reply.get("finish_reason"), reply.get("usage"), reply.get("other_text"))
        return cls(str(reply))

    def cut_short(self) -> bool:
        """没写完就没有了：到了输出上限，或者正文是空的（思考类模型把输出全花在推理上时就是这样）。
        再问一次多半还是这样，要把上限调大——所以这种不重问，直接报出来。"""
        return self.finish_reason == "length" or not self.text.strip()

    def side_note(self) -> str:
        """正文以外的文字字段有多长，写进给用户看的话里："，推理等其它字段：reasoning_content 1234 个字符"。"""
        if not self.other:
            return ""
        return "，推理等其它字段：" + "、".join(f"{name} {chars} 个字符" for name, chars in self.other.items())

    def cut_short_error(self, what: str, max_tokens: int) -> ValueError:
        """被截断的回复的报错：说清是到了输出上限还是正文是空的、正文有多长，以及上限到哪去调。

        不带上解析器那句"没有 JSON 对象"：那只是结果，不是原因。
        """
        advice = (f"把上限调大（--max-tokens，现在是 {max_tokens}；经 run_rft_guided.sh 跑时设环境变量 "
                  "TEACHER_MAX_TOKENS），调大后原样重跑只重做出错的题")
        if self.finish_reason == "length":
            head = (f"{what}在写完之前到了输出上限（finish_reason=length，正文 {len(self.text)} 个字符"
                    f"{self.side_note()}）")
        else:
            head = (f"{what}是空的（finish_reason={self.finish_reason}，正文 0 个字符{self.side_note()}），"
                    "多半同样是没有写完")
        return ValueError(f"{head}，{advice}")

    def diagnosis(self, secret: str | None = None) -> dict:
        """出了事以后留给人事后看的：原始回复的前 REPLY_EXCERPT 个字符、正文一共多长、结束原因、用量、推理等
        其它字段的长度。只留回复，不留题目。"""
        return {"text": redact(self.text[:REPLY_EXCERPT], secret), "chars": len(self.text),
                "head_only": len(self.text) > REPLY_EXCERPT, "finish_reason": self.finish_reason,
                "usage": self.usage, "other_text": dict(self.other)}


def write_skill(spec: dict, attempts, ask, *, max_rounds: int = 2, judge: bool = True,
                max_tokens: int = DEFAULT_MAX_TOKENS, temperature: float = 0.3, secret: str | None = None) -> dict:
    """让教师给一道题写技能：写一版、查一版，有问题带着意见重写，最多 ``max_rounds`` 轮。

    ``ask(messages, max_tokens, temperature)`` 是发请求的函数（测试里注入假的）。它的返回值可以是回复的
    字符串，也可以是 ``chat_text`` 那样的字典（带 ``usage``、``finish_reason`` 和 ``other_text``）。
    ``secret`` 是教师的密钥，拿得到就给：记进记录的文字里出现了它就换成 ***。

    返回的记录：task_id、status、rounds（实际写了几轮）、checks（每轮的静态命中、自判结果和理由）、
    usage（这道题的用量合计）、reminders（带着提醒重问了几次）；ok 时有 text 和 parts，leaky 时有
    rejected_text，error 时有 error，拿到过回复的话还有 reply（最后那条回复的诊断，见 Reply.diagnosis）。
    """
    record = {"task_id": None, "rounds": 0, "checks": [], "usage": {"input_tokens": 0, "output_tokens": 0},
              "reminders": 0}
    last = None                       # 最后拿到的那条回复：出错时留作诊断

    def finish(status: str, **extra) -> dict:
        return {"task_id": record["task_id"], "status": status, **extra, "rounds": record["rounds"],
                "checks": record["checks"], "usage": record["usage"], "reminders": record["reminders"]}

    def request(messages) -> Reply:
        nonlocal last
        reply = Reply.of(ask(messages, max_tokens, temperature))
        last = reply
        if reply.usage:
            record["usage"]["input_tokens"] += reply.usage.get("input_tokens") or 0
            record["usage"]["output_tokens"] += reply.usage.get("output_tokens") or 0
        return reply

    def ask_json(parse, messages, what):
        """请教师回复，解析成 ``parse`` 要的东西；``what`` 写明是哪一步的回复。

        回复被截断或是空的：直接报"到了输出上限、把上限调大"。正文有、可是解析不了的：在原消息末尾加一句
        提醒重问一次；还是没有才报错。以前解析不了就记 error，而出错的记录里不留回复，事后查不出原因。
        """
        first = request(messages)
        if first.cut_short():
            raise first.cut_short_error(what, max_tokens)
        try:
            return parse(first.text)
        except ValueError:
            pass
        record["reminders"] += 1
        again = request(with_reminder(messages))
        if again.cut_short():
            raise again.cut_short_error(what, max_tokens)
        try:
            return parse(again.text)
        except ValueError as error:
            raise ValueError(f"{what}里还是没有可用的 JSON（已经带着提醒重问过一次）：{error}") from None

    feedback, text = "", ""
    try:                                    # 这里面出的一切错都记成 error：一道题坏了不停掉整批
        record["task_id"] = spec["task_id"]
        identifiers = code_identifiers(spec)
        for round_index in range(1, max_rounds + 1):
            record["rounds"] = round_index
            parts = ask_json(parse_skill_json, diagnosis_messages(spec, attempts, feedback), "写技能的回复")
            text = render_skill(parts)
            check = {"round": round_index, "leaks": static_leaks(text, identifiers),
                     "reconstructable": None, "reason": None}
            record["checks"].append(check)
            # 两道检查每轮都做：静态检查没过也照样问自判，下一轮才能把两条意见一起带给教师
            if judge:
                answer = ask_json(parse_judge_json, judge_messages(text, spec), "自判的回复")
                check["reconstructable"], check["reason"] = answer
            if not check["leaks"] and not check["reconstructable"]:
                return finish("ok", text=text, parts=parts)
            feedback = revision_feedback(text, check)
    except Exception as error:          # 请求出错、回复读不出来：记下来，别的题照常进行
        kept = {"reply": last.diagnosis(secret)} if last is not None else {}
        return finish("error", error=redact(f"{type(error).__name__}: {error}", secret), **kept)
    return finish("leaky", rejected_text=text)


def pick_attempts(records, limit: int = 2) -> list:
    """挑给教师看的失败尝试：没通过、代码能解析的 graded 记录。

    公开测试过了而隐藏测试没过的排前面（离做对最近），代码相同的只留一份，最多 ``limit`` 份。
    """
    usable = []
    if limit < 1:
        return usable
    for record in records:
        if (record.get("provenance") or {}).get("strict_pass"):
            continue
        code = (record.get("target") or {}).get("content") or ""
        if parses(code):
            usable.append(record)
    chosen, seen = [], set()
    # sorted 是稳定的：同一优先级里保持原来的先后
    for record in sorted(usable, key=lambda item: (item.get("provenance") or {}).get("failure_type") != NEAR_MISS):
        code = record["target"]["content"]
        if code in seen:
            continue
        seen.add(code)
        chosen.append(record)
        if len(chosen) == limit:
            break
    return chosen


def bar_line(done: int, total: int, started: float, label: str) -> str:
    """进度行。格式同 sample_rft.py 和 run_rft.sh 的 progress()：进度面板靠它识别。"""
    elapsed = time.monotonic() - started
    eta = elapsed / done * (total - done) if done else 0
    filled = 24 * done // total
    return (f"[{'#' * filled}{'-' * (24 - filled)}] {done}/{total} ({done * 100 // total}%) "
            f"{label} · 已用 {elapsed / 60:.1f}m · 剩余≈{eta / 60:.1f}m")


# 记录里别的字段（自判的理由、出错信息）仍可能带着 LINE_SEPARATORS 里的字符，而不少读 JSONL 的脚本按
# splitlines() 切行（旧版本的 sample_rft.py 就是）：不转义，这一行会被切成几段、当成坏行丢掉。


def dump_line(record) -> str:
    """一条记录写成一行，末尾带换行；按 splitlines() 或按 "\\n" 切行的脚本都不会把它切开。

    中文原样写出，便于人读；只有 LINE_SEPARATORS 里的字符写成 \\uXXXX 转义——在 JSON 里完全等价，
    读回来还是原来的字符（其余的控制字符 json.dumps 本来就会转义）。
    """
    text = json.dumps(record, ensure_ascii=False)
    for char in LINE_SEPARATORS:
        text = text.replace(char, f"\\u{ord(char):04x}")
    return text + "\n"


def _read_text(path: Path) -> str:
    """按文件里本来的样子读出文字，不做行尾转换（newline=""：\r\n 还是 \r\n），截在汉字中间的坏字节换成
    替换字符——那一行下一刻就因解析不了被跳过。"""
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        return handle.read()


def _split_lines(raw: str) -> list[str]:
    """按行尾切出各行。只认 "\n" 和 "\r"：合法的 JSON 记录里不会有原样的 "\r"，而 U+2028 这类字符
    splitlines() 也当成换行，会把一条完整的记录切成几段。"""
    return raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def read_jsonl(path: Path) -> list:
    """读一个 JSONL 输入文件，每行解析成对象；写到一半被中断的坏行跳过。只读，不改文件。"""
    records = []
    if not path.exists():
        return records
    for line in _split_lines(_read_text(path)):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"跳过没写完的一行：{path.name}", file=sys.stderr)
    return records


def _plain_record(record):
    """旧记录的技能正文和各段里若还带着 U+2028 这类分隔符，换成换行（同 plain_breaks）；其余字段原样。"""
    if not isinstance(record, dict):
        return record
    cleaned = dict(record)
    for field in ("text", "rejected_text"):
        if isinstance(cleaned.get(field), str):
            cleaned[field] = plain_breaks(cleaned[field])
    if isinstance(cleaned.get("parts"), dict):
        cleaned["parts"] = {key: plain_breaks(value) if isinstance(value, str) else value
                            for key, value in cleaned["parts"].items()}
    return cleaned


def load_output(path: Path) -> dict:
    """读已有的输出，返回 {题号: 这道题的最后一条记录}；顺手把文件整理成"每行一条记录、行尾带换行"。

    不整理的话，后面读这个文件的脚本会读错，新记录也会接坏。要整理的有三种情况：
    - 写到一半被中断的行（没写完，甚至截在某个汉字中间）：去掉。去掉的那道题没有定论，这次重写；
    - 技能正文或别的字段里原样带着 U+2028 这类字符（旧版本或别的工具写出来的）：正文里换成换行，别的
      字段改成转义写法；
    - 行尾不标准（最后一行没有换行，或用的是 \\r\\n）：补上、换成 \\n。不然追加的新记录会接在最后一行
      后面，两条粘成一行坏行。
    """
    if not path.exists():
        return {}
    raw = _read_text(path)
    kept, broken = [], 0
    for line in _split_lines(raw):
        if not line.strip():
            continue
        try:
            kept.append(json.loads(line))
        except json.JSONDecodeError:
            broken += 1
    untidy = (bool(raw) and not raw.endswith("\n")) or any(char in raw for char in ("\r", *LINE_SEPARATORS))
    if broken:
        print(f"{path.name} 里有 {broken} 行没写完：去掉它们，重写这个文件", file=sys.stderr)
    elif untidy:
        print(f"{path.name} 的格式不标准（行尾缺换行，或内容里带着会被当成换行的字符）："
              "整理成每行一条记录，重写这个文件", file=sys.stderr)
    if broken or untidy:
        kept = [_plain_record(record) for record in kept]
        temporary = path.with_name(path.name + ".tmp")     # 先写旁边再换过去：重写到一半被打断也不丢已有的记录
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write("".join(dump_line(record) for record in kept))
        temporary.replace(path)
    return {record["task_id"]: record for record in kept if isinstance(record, dict) and "task_id" in record}


def _run(pending, attempts, args, client, model, output: Path, latest: dict, total: int) -> int:
    """并发地给 ``pending`` 里的题写技能，写完一道追加一道；返回这次一共请求了教师多少次。"""
    lock = threading.Lock()
    calls = 0

    def ask(messages, tokens, heat):
        nonlocal calls
        with lock:
            calls += 1
        return chat_text(client, model, messages, max_tokens=tokens, temperature=heat)

    def work(spec):
        record = write_skill(spec, attempts[spec["task_id"]], ask, max_rounds=args.max_rounds, judge=args.judge,
                             max_tokens=args.max_tokens, temperature=args.temperature,
                             secret=os.environ.get(args.api_key_env))
        return {**record, "model": model, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    started, done = time.monotonic(), total - len(pending)
    for _, record in vllm_client.map_concurrent(work, pending, args.concurrency):
        with lock:
            with output.open("a", encoding="utf-8", newline="") as handle:     # 写出去的就是 "\n"，不做行尾转换
                handle.write(dump_line(record))
                handle.flush()
            latest[record["task_id"]] = record
            done += 1
            print(bar_line(done, total, started,
                           f"写技能 {record['task_id']} {record['status']}（{record['rounds']} 轮）"), flush=True)
        if record["status"] == "error":
            print(f"{record['task_id']} 出错：{record.get('error')}", file=sys.stderr, flush=True)
    return calls


# 教师接口的参数：命令行没给时取环境变量 TEACHER_<名字>（同 TEACHER_MODEL、TEACHER_BASE_URL）。经
# run_rft_guided.sh 跑时没有命令行可加，只能靠它们调（.env 里写上就行）
SETTINGS = (("max_rounds", int, 2, "正整数"), ("max_tokens", int, DEFAULT_MAX_TOKENS, "正整数"),
            ("min_interval", float, 1.0, "非负数"), ("temperature", float, 0.3, "数字"),
            ("timeout", float, 120.0, "数字"))


def teacher_setting(name: str, kind: type, fallback, what: str):
    """环境变量 TEACHER_<NAME> 给的默认值；没设（或者是空的）就是 ``fallback``。

    写得不是数时抛 ValueError，话里带上 ``what``（"正整数" 或 "数字"）：经 run_rft_guided.sh 跑时这个报错是
    用户看出 .env 里写错了的唯一线索。
    """
    raw = os.environ.get(f"TEACHER_{name.upper()}")
    if raw is None or not raw.strip():
        return fallback
    try:
        return kind(raw.strip())
    except ValueError:
        raise ValueError(f"环境变量 TEACHER_{name.upper()} 要是{what}，现在是 {raw!r}") from None


def main(argv=None, client=None) -> int:
    """命令行入口。``client`` 给了就不再自己建（测试里注入假的教师客户端）；返回退出码。"""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks-file", required=True, help="冻结的任务文件（带 split）")
    parser.add_argument("--zero-pass-from", required=True,
                        help="上一轮的 <split>-manifest.json（sample_rft.py 的输出，读它的 zero_pass_tasks）")
    parser.add_argument("--split", choices=["train", "dev"], default="train")
    parser.add_argument("--tiers", default="hard,variant",
                        help="给哪些层写技能，逗号分隔：default（原有题）/ hard（难题）/ variant（单缺陷变体）")
    parser.add_argument("--graded", default=None,
                        help="上一轮的 <split>-graded.jsonl；不给或文件不存在时不带失败尝试")
    parser.add_argument("--output", required=True, help="输出文件（skills.jsonl），每行一道题")
    parser.add_argument("--concurrency", type=int, default=2,
                        help="同时写几道题，也就是同一时刻最多有几个教师请求在飞（默认 2）")
    parser.add_argument("--max-rounds", type=int, default=None,
                        help="每道题最多让教师写几轮（默认 2；环境变量 TEACHER_MAX_ROUNDS）")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help=f"教师每次回复的输出上限（默认 {DEFAULT_MAX_TOKENS}；环境变量 TEACHER_MAX_TOKENS）")
    parser.add_argument("--min-interval", type=float, default=None,
                        help="两次教师请求之间至少隔几秒，全进程共用（默认 1；环境变量 TEACHER_MIN_INTERVAL）")
    parser.add_argument("--temperature", type=float, default=None,
                        help="教师接口的温度（默认 0.3；环境变量 TEACHER_TEMPERATURE）")
    parser.add_argument("--judge", action=argparse.BooleanOptionalAction, default=True,
                        help="写完让教师自判技能是否接近答案（默认开，--no-judge 关掉）")
    parser.add_argument("--model", default=None, help="教师模型名；不填时用环境变量 TEACHER_MODEL")
    parser.add_argument("--base-url", default=None,
                        help="教师接口地址；不填时用环境变量 TEACHER_BASE_URL，再没有就用内置的占位地址")
    parser.add_argument("--api-key-env", default=KEY_ENV, help="存密钥的环境变量名")
    parser.add_argument("--timeout", type=float, default=None,
                        help="单次请求的超时（秒）（默认 120；环境变量 TEACHER_TIMEOUT）")
    args = parser.parse_args(argv)
    for name, kind, fallback, what in SETTINGS:
        if getattr(args, name) is None:       # 命令行没给：取环境变量，再没有就是内置的默认值
            try:
                setattr(args, name, teacher_setting(name, kind, fallback, what))
            except ValueError as error:
                parser.error(str(error))
    try:
        extra_json()                          # 设了 TEACHER_EXTRA_JSON 就得是 JSON 对象：不对就在这儿报，别等请求的时候
    except ValueError as error:
        parser.error(str(error))
    tiers = [tier.strip() for tier in args.tiers.split(",") if tier.strip()]
    if not tiers or not set(tiers) <= set(TIERS):
        parser.error(f"unknown tier in --tiers; choose from {', '.join(TIERS)}")
    if min(args.concurrency, args.max_rounds, args.max_tokens) < 1 or args.timeout <= 0:
        parser.error("concurrency / max-rounds / max-tokens / timeout must be positive")
    if args.min_interval < 0:
        parser.error("min-interval cannot be negative")
    try:
        specs = load_specs(args.tasks_file, args.split)
        manifest = json.loads(Path(args.zero_pass_from).read_text(encoding="utf-8"))
        zero_pass = set(manifest["zero_pass_tasks"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(f"读不了任务文件或上一轮的清单：{error}")

    selected = [spec for spec in specs if spec["task_id"] in zero_pass and tier_of(spec["task_id"]) in tiers]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # 文件先建出来（可以是空的）：后面的步骤直接读它来数可用的技能，不用再判断文件在不在
    if not output.exists():
        output.touch()
    latest = load_output(output)
    pending = [spec for spec in selected if latest.get(spec["task_id"], {}).get("status") not in SETTLED]

    by_task = defaultdict(list)
    if args.graded and Path(args.graded).exists():
        for record in read_jsonl(Path(args.graded)):
            if isinstance(record, dict) and record.get("task_id"):
                by_task[record["task_id"]].append(record)
    elif pending:
        missing = f"没找到判分文件 {args.graded}" if args.graded else "没有给 --graded"
        print(f"{missing}：不带失败尝试，继续。", flush=True)
    attempts = {spec["task_id"]: pick_attempts(by_task[spec["task_id"]]) for spec in pending}

    calls = 0
    if pending:
        model = args.model or teacher_model()
        print(f"写技能：{len(selected)} 道题（{args.split}，{args.tiers} 层，上一轮零通过），"
              f"已有定论 {len(selected) - len(pending)} 道，待写 {len(pending)} 道，"
              f"其中 {sum(not picks for picks in attempts.values())} 道没有可用的失败尝试。", flush=True)
        print(f"教师：{model}；每题最多 {args.max_rounds} 轮，{'写完自判' if args.judge else '不自判'}；"
              f"每次回复最多 {args.max_tokens} 个 token，温度 {args.temperature:g}，单次请求超时 {args.timeout:g} 秒"
              "（调法见本脚本开头的说明）；"
              f"同时最多 {args.concurrency} 个请求，两次请求至少隔 {args.min_interval:g} 秒。", flush=True)
        if client is None:
            try:
                client = make_client(args.api_key_env, args.base_url, args.timeout)
            except RuntimeError as error:
                parser.error(str(error))
            except ImportError as error:
                parser.error(f"没装 openai SDK：{error}（pip install 'openai>=1.0'）")
        # 限额按账户算：在飞的请求数和间隔按这一次的参数来定，之后所有的请求（两步、所有线程）都走这一份
        configure_throttle(max_in_flight=args.concurrency, min_interval=args.min_interval)
        calls = _run(pending, attempts, args, client, model, output, latest, len(selected))
    elif selected:
        print(f"没有要写的题：{len(selected)} 道都已有定论，不发请求。")
    else:
        print(f"没有要写的题：上一轮零通过的题里没有属于 {args.split} 的 {args.tiers} 层的题，不发请求。")
        elsewhere = sorted(task for task in zero_pass if tier_of(task) in tiers)
        if elsewhere:
            print(f"（清单里属于这些层的题号：{elsewhere[:8]}，它们不在 {args.split} 里）")

    final = [latest[spec["task_id"]] for spec in selected if spec["task_id"] in latest]
    counts = Counter(record.get("status") for record in final)
    ok, leaky, errors = (counts[status] for status in ("ok", "leaky", "error"))
    for status, label in (("leaky", "泄露被拦（不用）"), ("error", "出错（原样重跑会重做）")):
        tasks = [record["task_id"] for record in final if record.get("status") == status]
        if tasks:
            print(f"{label}：{'、'.join(tasks[:8])}{'…' if len(tasks) > 8 else ''}")
    print(f"结果：技能提示 可用 {ok}/{len(selected)} 道，泄露被拦 {leaky} 道，出错 {errors} 道；"
          f"教师调用 {calls} 次，输出 {output}", flush=True)
    return 3 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
