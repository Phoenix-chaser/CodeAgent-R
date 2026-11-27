# -*- coding: utf-8 -*-
"""合成题族的执行审计：教师生成的题族逐项过检，通过的才能并入题池。

教师给的题族（parse_family 的返回值）不能直接进题池：测试可能接受错误答案，参考解可能
过不了自己的测试，单个缺陷可能没被任何测试拦住，代码里可能藏着起子进程、读写文件的写法。
audit_family 在沙箱里真跑每一道题，八项都查完才给结论；发现的问题写成可以直接动手改的
中文说明，原样进返修提示：

1. 结构：规格合法；task_id、source_group、module、原程序不和现有题池撞；参考解不和已知
   的参考解重复（撞了只能换 slug，返修解决不了）；
2. 参考解：公开测试和隐藏测试全部通过，重复 repeats 次结果一致；
3. 每个单缺陷变体：被隐藏测试的断言拦住，公开测试仍然通过；
4. 母题：被隐藏测试的断言拦住，公开测试通过，重复 repeats 次一致；
5. 隐藏测试能区分各条要求（不同变体挂掉的测试不全一样）；
6. 长度：题面加参考解的估算不超过 1900 token，参考解 30 到 70 行；
7. 安全：没有起子进程、联网、读写文件、等待的写法；
8. 没有提示缺陷在哪的注释标记。

“被拦住”只认隐藏测试里的断言失败：全部通过、报错、进程异常退出、超时、沙箱出错都不算。
第 5、8 项的问题只记发现，不影响是否接受（见 audit_family）。审计会临时调小判分器的 pytest 超时
（进程内的全局变量），不要在同一个进程里并发判分别的题。

发现项的 code，按项分：
- 结构：bad_slug、invalid_family、syntax_error、no_test_functions、mutant_anchor、mutant_syntax、
  mutant_duplicate、mutant_overlap、invalid_spec、collision；
- 参考解：reference_fails、reference_timeout、reference_no_results、grading_error、unstable_tests；
- 变体：mutant_escapes（全部测试仍然通过）、mutant_breaks_public、mutant_not_failed（报错，或者没有
  断言失败）、mutant_timeout、mutant_ungraded（沙箱出错，没判完）；
- 母题：parent_escapes、parent_breaks_public、parent_not_failed、parent_timeout、parent_ungraded
  （原因同上），另有 too_few_mutants、parent_broken；
- 第 5 到 8 项：indistinct_tests、fixed_lines、too_long_tokens、unsafe_code、hint_comment。
不可返修的只有 bad_slug、collision、grading_error，以及 slug 不合规时的 invalid_spec。参考解判不完的时候
会用一道完好的小题自检判分环境（_environment_ok）：环境是好的就记成可返修的 reference_no_results，
环境自检也没过才是 grading_error。

返回之前，findings 和 stats 里每一段文本都去掉绝对路径的目录部分（见 without_paths）：判分的输出和
异常的文本里有工作区、解释器的路径，工作区放在个人目录下时那就是个人路径。family 是判过分的内容，不动。
题族的问题都变成发现项，audit_family 只为调用方传错参数抛 TypeError / ValueError。判分环境本身出了
问题（比如工作区建不出来）记成不可返修的 grading_error；审计程序自己出了错也只记成 grading_error
（消息里说明是程序的问题）。两种情况都不打断整批审计。
"""
from __future__ import annotations

import ast
import difflib
import itertools
import json
import re
import subprocess
import time
from pathlib import Path

from agentlab.benchmark import grader
from agentlab.benchmark.builder import SPECS, build_all
from agentlab.benchmark.grader import GradeInfrastructureError
from agentlab.benchmark.hard_tasks_v3 import HARD3_SPECS
from agentlab.benchmark.hard_tasks_v3_steps import STEP_SPECS
from agentlab.benchmark.pilot_tasks import PILOT_SPECS
from agentlab.benchmark.pilot_tasks_v2 import V2_SPECS
from agentlab.benchmark.pilot_tasks_v2b import V2B_SPECS
from agentlab.benchmark.pilot_tasks_v2c import V2C_SPECS
from agentlab.benchmark.pilot_tasks_v2d import V2D_SPECS
from agentlab.benchmark.protocol import validate_specs
from agentlab.benchmark.task_store import TaskStore
from agentlab.runtime.isolation import SandboxError, resolve_mode
from agentlab.synth.specs import (
    MAX_SLUG, SLUG_PATTERN, FamilyError, build_family, build_parent, build_variant, family_specs, fixed_hash,
    mutant_name, prompt_estimate, rough_tokens, shorten,
)

# 训练默认长度是 2048 个 token；估算偏保守，超过这个数就该把题压缩（同 scripts/self_check_pool.py）
LONG_ESTIMATE = 1900
FIXED_LINES = (30, 70)           # 参考解的行数范围，照现有难题的量级
MIN_MUTANTS = 3                  # 可用的 mutant 少于这个数，母题就不成立
# 参考解和母题沿用导出审计的 5 秒上限。单缺陷变体用接近正式判分（60 秒）的上限：
# 只是让测试变慢的变体在正式判分里会通过，不能算成“被拦住”
REFERENCE_TIMEOUT, MUTANT_TIMEOUT = 5, 30

BANNED_IMPORTS = ("subprocess", "socket", "http", "urllib", "ctypes", "multiprocessing", "shutil",
                  "asyncio", "pathlib")
BANNED_OS_CALLS = ("system", "popen", "remove", "unlink", "rmdir", "rename", "replace", "kill", "fork")
BANNED_BUILTINS = ("eval", "exec", "compile", "__import__", "input")
SLEEP_LIMIT = 0.2
# 题面不能被代码里的注释剧透（同 tests/test_tasks_h3.py；只看注释）
HINT = re.compile(r"#.*\b(BUG|FIXME|TODO|XXX|HACK)\b", re.IGNORECASE)

CHECKS = ("structure", "reference", "variants", "parent", "tests_distinguish", "length", "safety", "no_hints")
# 一个带缺陷的程序为什么不可用，几个字的说法（详细说明见各自的发现项）
BRIEF = {"timeout": "测试超时", "ungraded": "没能判分", "public": "公开测试没通过",
         "escaped": "全部测试仍然通过", "errored": "隐藏测试报错", "unfailed": "没有断言失败"}
UNSAFE_ADVICE = "题目代码和测试不能起子进程、联网、读写文件或等待，请改成纯计算的写法"
MUTANT_PUBLIC_ADVICE = ("单缺陷变体必须仍然通过公开测试：请把这处疏忽挪到公开测试覆盖不到的行为上，"
                        "或者把公开测试收缩成最基本的接口演示。")
PARENT_PUBLIC_ADVICE = ("母题的原程序必须通过公开测试：各处缺陷单独出现时还没破坏公开测试，合在一起却破坏了，"
                        "请换一个不冲突的 mutant，或者把公开测试收缩成最基本的接口演示。")


def existing_pools(extra_specs=()) -> list[tuple[str, list[dict]]]:
    """合成题不能撞上的全部题目：内置题池，加上调用方额外给出的（例如本轮已经接受、
    还没写进题池的合成题）。"""
    return [
        ("builtin", SPECS), ("pilot", PILOT_SPECS), ("v2", V2_SPECS), ("v2b", V2B_SPECS),
        ("v2c", V2C_SPECS), ("v2d", V2D_SPECS), ("h3", HARD3_SPECS), ("h3s", STEP_SPECS),
        ("extra", list(extra_specs)),
    ]


# ── 静态检查（第 7、8 项）：只看语法树和注释 ─────────────────────────

def _called(func: ast.expr, modules: dict[str, str],
            names: dict[str, tuple[str, str]]) -> tuple[str, str] | None:
    """一次调用写成“哪个模块的哪个名字”；直接调内建函数时第一项是空串。"""
    if isinstance(func, ast.Name):
        if func.id == "open" or func.id in BANNED_BUILTINS:
            return "", func.id
        return names.get(func.id)
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        module = modules.get(func.value.id)
        if module in ("os", "time", "io", "builtins"):
            return module, func.attr
    return None


def _sleep_is_short(call: ast.Call) -> bool:
    argument = call.args[0] if call.args else next(
        (keyword.value for keyword in call.keywords if keyword.arg in (None, "seconds")), None)
    # bool 是 int 的子类：time.sleep(True) 不是“不超过 0.2 秒”的字面量
    return (isinstance(argument, ast.Constant) and type(argument.value) in (int, float)
            and 0 <= argument.value <= SLEEP_LIMIT)


def unsafe_uses(tree: ast.AST, offset: int = 0) -> list[str]:
    """源码里不允许的写法，每处一条“第 N 行 …”；offset 是不计入行号的文件头行数。

    只认写得出来的直接用法：import、from … import、对模块别名的属性调用、由 from … import
    绑定下来的名字。getattr 之类的间接调用查不出来，那是对抗性代码的事，交给沙箱。
    """
    modules: dict[str, str] = {}                   # import os as o → o → os
    names: dict[str, tuple[str, str]] = {}         # from os import system → system → (os, system)
    uses: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                modules[alias.asname or top] = top
                if top in BANNED_IMPORTS:
                    uses.append((node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""             # 相对导入（from . import x）的 module 是 None
            for alias in node.names:
                names[alias.asname or alias.name] = (module.split(".")[0], alias.name)
                if module.split(".")[0] in BANNED_IMPORTS:
                    uses.append((node.lineno, f"from {module} import {alias.name}"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        used = _called(node.func, modules, names)
        if used is None:
            continue
        module, attribute = used
        if attribute == "open" and module in ("", "io", "builtins"):
            uses.append((node.lineno, "open() 的调用"))
        elif attribute in BANNED_BUILTINS and module in ("", "builtins"):
            uses.append((node.lineno, f"{attribute}() 的调用"))
        elif module == "os" and attribute in BANNED_OS_CALLS:
            uses.append((node.lineno, f"os.{attribute}() 的调用"))
        elif module == "time" and attribute == "sleep" and not _sleep_is_short(node):
            shown = ast.unparse(node.args[0]) if node.args else ""
            uses.append((node.lineno, f"time.sleep({shown})：等待时间必须是不超过 {SLEEP_LIMIT} 秒的字面量"))
    return [f"第 {line - offset} 行 {what}" for line, what in sorted(uses)]


def hint_markers(text: str, offset: int = 0) -> list[tuple[int, str, str]]:
    """注释里的提示性标记：(行号, 标记, 那一行去掉首尾空白)。offset 是不计入行号的文件头行数。"""
    found = []
    for number, line in enumerate(text.splitlines(), 1):
        match = HINT.search(line)
        if match:
            found.append((number - offset, match.group(1).upper(), line.strip()))
    return found


def collectable_tests(tree: ast.Module) -> list[str]:
    """pytest 默认规则下收集得到的测试函数名：模块层 test 开头的函数，以及 Test 开头、没有 __init__ 的类里
    test 开头的方法。只认写得出来的 def，动态生成的测试认不出来。"""
    names = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            names.append(node.name)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test") and not any(
                isinstance(item, ast.FunctionDef) and item.name == "__init__" for item in node.body):
            names.extend(item.name for item in node.body
                         if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test"))
    return names


def _declared(tree: ast.Module) -> str:
    """测试文件的顶层定义了哪些名字：给“没有测试函数”的发现项当证据。"""
    names = [node.name for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    return f"现在定义的只有 {'、'.join(names[:6])}" if names else "现在一个函数都没有定义"


def _slug_problem(slug) -> str | None:
    """slug 能不能拿来出题；不能时说明原因（没有问题时返回 None）。"""
    if not isinstance(slug, str) or not SLUG_PATTERN.fullmatch(slug):
        return "只能用小写字母、数字、下划线。"
    if len(slug) > MAX_SLUG:
        return f"最多 {MAX_SLUG} 个字符（现在 {len(slug)} 个），它要进文件名和 task_id。"
    if "private" in slug:
        # 任务库的泄漏检查把 fixture 里名字带 private 的文件当成私有材料，这道题就读不出来
        return "不能带 private，任务库会把这样的文件名当成私有材料、拒绝读出这道题。"
    return None


def _where(error: Exception, offset: int = 0) -> str:
    """解析错误出在哪一行；offset 是不计入行号的文件头行数。拿不到行号时只说解析时出了错。"""
    line = getattr(error, "lineno", None)
    return f"第 {(line or 0) - offset} 行" if line is not None else "解析时出错"


def _reason(error: Exception) -> str:
    return getattr(error, "msg", None) or str(error)


def _mutant_problem(error: FamilyError) -> tuple[str, str]:
    """构造变体时题族不对：（几个字的原因，给教师的改法）。"""
    text = str(error)
    if "写不进文件" in text:
        return "new 片段里有写不进文件的字符", "请把 new 片段里孤立的代理项换成正常的字符。"
    if "空字节" in text:
        return "new 片段里有空字节", "请删掉 new 片段里的空字节（NUL）。"
    if "行末尾有控制字符" in text:
        return "new 片段的行末尾有控制字符", "请删掉 new 片段行末尾的控制字符（回车、垂直制表符、换页符）。"
    if "old 片段" in text and "出现" in text:
        return "old 片段在参考解里不是恰好出现一次", "请把 old 改成能唯一定位的那几行，new 写成改坏后的样子。"
    return "old / new 片段不合格", ("请照题族的格式写全 old 和 new：old 是参考解里原样照抄的一段，"
                                    "new 是把它改坏之后的样子。")


# 判分的输出和异常的文本里会带着绝对路径：工作区的（可能在一个人的目录下）、解释器的。
# 前面的 (?<![\w/.\-]) 是为了不碰 1/2 这样的算式；测试的编号（tests_private/test_x.py）原样保留。
_TEST_DIRS = re.compile(r"\S*/(tests_(?:public|private)/)")
_LOCATION = re.compile(r"(?<!\S)(?!tests_(?:public|private)/)\S*/(?=\S+\.py:\d)")
_POSIX_DIRS = re.compile(r"(?<![\w/.\-])(?:/[\w.\-@+%]+)+/")
_POSIX_PATH = re.compile(r"(?<![\w/.\-])(?:/[\w.\-@+%]+)+/([\w.\-@+%]+)")
_WINDOWS_DIRS = re.compile(r"\b[A-Za-z]:(?:\\+[\w.\- ]+)+\\+")
_WINDOWS_PATH = re.compile(r"\b[A-Za-z]:(?:\\+[\w.\- ]+)+\\+([\w.\- ]+)")


def without_paths(text: str, roots=()) -> str:
    """去掉文本里绝对路径的目录部分，只留文件名；测试文件保留 tests_private/、tests_public/ 这一层。

    判分的输出和异常的文本会进返修提示和报告。里面的工作区路径、解释器路径对返修没有用；工作区
    放在个人目录下时，原样带出去就是个人路径。roots 是已知的工作区目录，先原样去掉：目录名里可能
    带着空格这类正则认不了的字符。相对路径和算式（1/2、n/a）不动。
    """
    for root in roots:
        if root:
            text = text.replace(root + "/", "").replace(root, "")
    text = _TEST_DIRS.sub(r"\1", text)
    text = _LOCATION.sub("", text)
    text = _POSIX_DIRS.sub("", text)
    text = _POSIX_PATH.sub(r"\1", text)
    return _WINDOWS_PATH.sub(r"\1", _WINDOWS_DIRS.sub("", text))


def _scrub(value, roots=()):
    """把结果里的一段段文本都过一遍 without_paths。用于 findings 和 stats；family 是判过分的内容，不动。"""
    if isinstance(value, str):
        return without_paths(value, roots)
    if isinstance(value, list):
        return [_scrub(item, roots) for item in value]
    if isinstance(value, dict):
        return {key: _scrub(item, roots) for key, item in value.items()}
    return value


# ── 判分：做法同 scripts/export_tasks_v2.py ───────────────────────────

def make_patch(module_path: str, buggy: str, fixed: str) -> str:
    return "".join(difflib.unified_diff(
        buggy.splitlines(keepends=True), fixed.splitlines(keepends=True),
        fromfile=f"a/{module_path}", tofile=f"b/{module_path}"))


def grade_program(task, spec: dict, source: str, work: Path, timeout: float | None = None) -> dict:
    """给一道题的一个候选程序判分（主文件整个换成 source），做法同 scripts/export_tasks_v2.py。

    返回 verdict / strict_pass / public_pass / private_pass，另外带上两层各自的执行记录
    （含没通过的测试名）、pytest 输出的末尾，以及判分没走到功能层时的原因。timeout 是每层
    pytest 的上限（秒），不给就用判分器当时的设置；用完恢复原设置。
    """
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    patch = work / "candidate.diff"
    patch.write_text(make_patch(spec["module"], spec["buggy"], source), encoding="utf-8")
    previous = grader.PYTEST_TIMEOUT
    if timeout is not None:
        grader.PYTEST_TIMEOUT = timeout
    try:
        result, details = grader.grade(task, patch, work / "ws")
    finally:
        grader.PYTEST_TIMEOUT = previous
    layers = details.get("layers", {})
    functional = layers.get("functional", {})
    return dict(
        verdict=result.verdict, strict_pass=result.strict_pass,
        public_pass=functional.get("public_pass"), private_pass=functional.get("private_pass"),
        public=functional.get("public"), private=functional.get("private"),
        public_tail=(functional.get("public_tail") or "")[-400:],
        private_tail=(functional.get("private_tail") or "")[-400:],
        isolation=(layers.get("execution") or {}).get("isolation"),
        blocked={name: value for name, value in layers.items()
                 if name in ("patch", "security") or value.get("valid") is False},
    )


def _error_result(error: Exception) -> dict:
    """判分没跑完时的结果：超时、沙箱出错都不算候选被测试拦住，也不打断审计。"""
    if isinstance(error, subprocess.TimeoutExpired):
        # 异常的文本里是整条命令（解释器的绝对路径、沙箱的源码），对说明问题没有用
        kind, detail = "timeout", f"测试在 {getattr(error, 'timeout', '?')} 秒内没有结束"
    else:
        kind = "sandbox" if isinstance(error, (GradeInfrastructureError, SandboxError)) else "error"
        detail = f"{type(error).__name__}: {error}"
    return dict(verdict=f"error: {detail}"[:600], strict_pass=False,
                public_pass=None, private_pass=None, public=None, private=None, public_tail="",
                private_tail="", isolation=None, blocked={}, error_kind=kind)


# 判分环境自检用的小题：和题族无关，判分环境没问题时它的参考解一定通过
_PROBE = {
    "task_id": "synth_audit_probe", "source_group": "synth_audit_probe", "split": "smoke",
    "bug_family": "probe", "difficulty": "L1",
    "instruction": "让 double(x) 返回 x 的两倍。",
    "module": "synth_audit_probe.py",
    "buggy": "def double(x):\n    return x\n",
    "fixed": "def double(x):\n    return 2 * x\n",
    "public_test": "from synth_audit_probe import double\n\ndef test_public_double():\n    assert double(2) == 4\n",
    "private_tests": {"test_probe.py": "from synth_audit_probe import double\n\n"
                                       "def test_hidden_double():\n    assert double(-3) == -6\n"},
}
_PROBE_PASSED: set[str] = set()        # 自检通过的隔离模式：同一种模式下每个进程只自检一次


def _environment_ok(work_dir: Path) -> tuple[bool, str]:
    """判分环境自己能不能用：拿一道已知完好的小题（_PROBE）走一遍判分。返回（好不好，判分的结论）。

    参考解判不完有两种可能：判分环境坏了（重试、修环境就好），或者参考解、测试让 pytest 中途退出
    （要返修）。只有遇到这种情况时才自检；通过了的结果按隔离模式记住，后面再遇到就不用重跑。
    """
    try:
        mode = resolve_mode()
    except Exception as error:          # 比如要求必须有 Landlock 而内核不支持
        return False, f"{type(error).__name__}: {error}"
    if mode in _PROBE_PASSED:
        return True, "之前已经通过"
    root = work_dir / "environment-check"
    try:
        build_all(root / "tasks", specs=[_PROBE])
        outcome = grade_program(TaskStore(root / "tasks").load(_PROBE["task_id"]), _PROBE, _PROBE["fixed"],
                                root / "grade")
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"
    if outcome["strict_pass"]:
        _PROBE_PASSED.add(mode)
        return True, outcome["verdict"]
    return False, outcome["verdict"]


def _rejection(result: dict) -> tuple[bool, str, list[str]]:
    """一个带缺陷的程序被隐藏测试拦住了吗：(拦住, 没拦住的原因, 没通过的隐藏测试名)。"""
    if result.get("error_kind"):
        return False, result["error_kind"], []
    if result["strict_pass"]:
        return False, "escaped", []
    if result.get("blocked"):
        return False, "ungraded", []           # 判分器在测试之前就拒绝了，根本没跑到隐藏测试
    private = result.get("private") or {}
    if result["private_pass"] is False:
        # 有测试失败时 pytest 本来就以非 0 退出，所以不看退出状态，只看汇总行里的计数
        summary = private.get("summary") or {}
        names = private.get("non_passing_tests") or []
        if summary.get("failed") and len(names) == summary["failed"]:
            return True, "", sorted(names)        # 没通过的正好是那几个断言失败，没有被跳过或报错的
        return False, "errored" if summary.get("errors") else "unfailed", []
    return False, "escaped", []


def _diagnosis(result: dict) -> str:
    """为什么不算“公开测试通过、被隐藏测试的断言拦住”，按判分的先后取第一个原因。"""
    if result.get("error_kind") == "timeout":
        return "timeout"
    if result.get("error_kind") or result.get("blocked"):
        return "ungraded"
    if result["public_pass"] is not True:
        return "public"
    return _rejection(result)[1]


def _tail(result: dict) -> str:
    """pytest 输出的末尾，每行先去掉绝对路径的目录部分再截短：路径对说明问题没有用，只会把行撑得很长。
    公开测试没过时给公开层的输出（要先修它），否则给隐藏层的。"""
    shown = result["public_tail"] if result.get("public_pass") is not True else result["private_tail"]
    lines = [without_paths(line)[:200]
             for line in (shown or result["private_tail"] or result["public_tail"]).splitlines() if line.strip()]
    return "\n".join(lines[-4:])


def _summary(result: dict) -> dict:
    """一次判分写进 stats 的部分：结论、没通过的测试、pytest 输出的末尾。"""
    return dict(verdict=result["verdict"], strict_pass=result["strict_pass"],
                public_pass=result["public_pass"], private_pass=result["private_pass"],
                public_failed=(result.get("public") or {}).get("non_passing_tests") or [],
                private_failed=(result.get("private") or {}).get("non_passing_tests") or [],
                error=result.get("error_kind"), tail=_tail(result))


def _signature(result: dict) -> tuple:
    """两次判分算不算同一个结论：只看结论和没通过的测试，不看耗时。"""
    summary = _summary(result)
    return (summary["verdict"], summary["strict_pass"], summary["public_pass"], summary["private_pass"],
            summary["error"], tuple(summary["public_failed"]), tuple(summary["private_failed"]))


def _outcome(result: dict) -> str:
    """重复判分里的一次：通过，或者哪里没通过。"""
    if result.get("error_kind"):
        return "测试超时" if result["error_kind"] == "timeout" else "判分出错"
    if result["strict_pass"]:
        return "通过"
    failed = [*_summary(result)["public_failed"], *_summary(result)["private_failed"]]
    return "没通过" + (f"（没通过的测试：{'、'.join(failed)}）" if failed else "")


def _blocked_reason(result: dict) -> tuple[str, str]:
    """判分没走到功能层的原因：(是哪一层、判分器说了什么，给教师的提示)。没有这种原因时两项都是空串。"""
    if not result["blocked"]:
        return "", ""
    layer, evidence = next(iter(result["blocked"].items()))
    advice = {
        # 行末尾的换页符这类控制字符会让补丁变成坏补丁；结构那一项一般已经按行报过
        "patch": "多半是程序的某一行末尾带着换页符这类控制字符，补丁因此成了坏补丁；"
                 "请检查参考解和 mutant 的 new 片段。",
        "security": "公开测试运行期间工作区里的文件被改动了：请去掉参考解或测试里读写文件的写法。",
    }.get(layer, "请只修改主文件，不要动公开测试和受保护的文件。")
    return f"{layer}：{json.dumps(evidence, ensure_ascii=False)[:200]}", advice


def _failure(who: str, result: dict) -> str:
    """把一次没通过的判分写清楚：哪一层、哪些测试、pytest 说了什么。"""
    where, advice = _blocked_reason(result)
    if where:
        return f"{who}没走到功能层就被判分器拒绝（{where}）。{advice}"
    parts = []
    for label, layer in (("公开测试", "public"), ("隐藏测试", "private")):
        if result[f"{layer}_pass"] is True:
            continue
        record = result.get(layer)
        if not record:
            parts.append(f"{label}没有运行")
            continue
        names = record.get("non_passing_tests") or []
        state = "报错" if (record.get("summary") or {}).get("errors") else "没有通过"
        parts.append(f"{label}{state}" + (f"（没通过：{'、'.join(names)}）" if names else ""))
    tail = _tail(result)
    if tail:
        parts.append(f"pytest 输出的末尾：\n{tail}")
    parts.append("请让参考解满足题面的每一条要求，并核对测试断言的期望值")
    return f"{who}没有通过全部测试：" + "。".join(parts) + "。"


def _rejection_finding(kind: str, who: str, result: dict, change: str, limit: int,
                       public_advice: str, escape_advice: str) -> tuple[str, str, bool]:
    """一个带缺陷的程序不可用时，写清楚是哪一种不行、该怎么改（kind 是 mutant 或 parent）。"""
    diagnosis = _diagnosis(result)
    did = f"{change}之后" if change else ""
    if diagnosis == "timeout":
        return (f"{kind}_timeout",
                f"{who}让测试超时（上限 {limit} 秒），没有任何断言失败：{did}测试跑不完。只是让程序变慢的"
                f"缺陷在正式判分里会被判为通过，请改成能被断言抓住的疏忽。", True)
    if diagnosis == "ungraded":
        where, advice = _blocked_reason(result)
        detail = f"判分器在 {where} 就拒绝了，测试没有运行" if where else result["verdict"]
        return f"{kind}_ungraded", f"{who}没能判完（{detail}），不能当作被拦住。{advice}", True
    if diagnosis == "public":
        record = result.get("public") or {}
        names = record.get("non_passing_tests") or []
        state = "报错" if (record.get("summary") or {}).get("errors") else "失败"
        detail = f"（没通过的公开测试：{'、'.join(names)}）" if names else ""
        return (f"{kind}_breaks_public",
                f"{who}让公开测试{state}{detail}：{did}公开测试没能全部通过。{public_advice}", True)
    if diagnosis == "escaped":
        return (f"{kind}_escapes",
                f"{who}没有被任何隐藏测试拦住：{did}公开测试和隐藏测试全部通过，{escape_advice}", True)
    if diagnosis == "errored":
        errors = ((result.get("private") or {}).get("summary") or {}).get("errors")
        return (f"{kind}_not_failed",
                f"{who}没有被断言拦住：{did}隐藏测试报错（{errors} 个 error），没有断言失败。多半是改坏后模块"
                f"导入或测试准备阶段就抛了异常，请让这处疏忽表现为错误的结果，并被断言抓住。", True)
    private = result.get("private") or {}
    return (f"{kind}_not_failed",
            f"{who}没有被断言拦住：{did}隐藏测试没有通过，但不是干净的断言失败（pytest 退出状态 "
            f"{private.get('exit_status')}，汇总 {private.get('summary')}）：有测试被跳过、报错或没跑完。"
            f"请让这处疏忽表现为错误的结果，并被断言抓住。", True)


# ── 一次审计 ──────────────────────────────────────────────────────────

class _Audit:
    """一次审计的中间状态：每一项检查是一个方法，发现的问题记进 findings。"""

    def __init__(self, parsed: dict, slug: str, direction: str, work_dir: Path, repeats: int,
                 known: set[str], pools: list[tuple[str, list[dict]]]):
        self.parsed, self.slug, self.direction = parsed, slug, direction
        self.work_dir, self.repeats, self.known, self.pools = work_dir, repeats, known, pools
        self.findings: list[dict] = []
        self.checks = dict.fromkeys(CHECKS, False)
        self.stats: dict = {}
        self.base = None            # 只有参考解和测试的题目（母题去掉全部缺陷）
        self.family = None          # 结构上没问题的 mutant 构造出的题族
        self.parent_spec = None     # 母题判分和长度估算用的题目
        self.parent_task = None
        self.records: list[dict] = []       # 每个单缺陷变体的判分结论
        self.dropped: dict[int, str] = {}  # 结构上就被排除的 mutant：下标 → 几个字的原因
        self.usable: list[int] = []         # 可用的 mutant 在 parsed["mutants"] 里的下标
        self.final = None           # 只用可用的 mutant 构造的题族；接受时返回它
        self.fixed_tree = ""
        self.slug_ok = self.specs_valid = self.tests_ok = False
        self.gradable = True

    def add(self, code: str, message: str, repairable: bool = True) -> None:
        self.findings.append({"code": code, "message": message, "repairable": repairable})

    def run(self) -> dict:
        self.structure()
        if self.base is not None:   # 题族连题目都构造不出来时，能报的只有结构那一项的发现
            if self.specs_valid and self.tests_ok:
                self.reference()
                if self.gradable:
                    self.variants()
                    self.parent()
            self.length()
            self.safety()
            self.hints()
        self.distinguish()
        return self.result()

    def _texts(self, spec: dict) -> list[tuple[str, str, int]]:
        """一份题目里要扫的三类文本：(位置, 文本, 不计入行号的文件头行数)。"""
        items = [("参考解", spec["fixed"], 0), ("公开测试", spec["public_test"], 2)]
        return items + [(f"隐藏测试 {name}", content, 3)
                        for name, content in spec["private_tests"].items()]

    def repeat(self, task, spec: dict, code: str, label: str, timeout: int) -> list[dict]:
        return [self._assess(task, spec, code, self.work_dir / label, timeout) for _ in range(self.repeats)]

    def _assess(self, task, spec: dict, code: str, work: Path, timeout: int) -> dict:
        try:
            return grade_program(task, spec, code, work, timeout)
        except Exception as error:     # 超时、沙箱出错也是审计的结论，不让它打断剩下的检查
            return _error_result(error)

    def unstable(self, who: str, runs: list[dict]) -> None:
        outcomes = "；".join(f"第 {number} 次{_outcome(run)}" for number, run in enumerate(runs, 1))
        self.add("unstable_tests",
                 f"{who}重复 {self.repeats} 次结果不一致（{outcomes}）：测试不稳定，可能依赖时间、随机数或执行"
                 f"顺序，请改成确定性的实现和测试。", True)

    # 1 结构
    def structure(self) -> None:
        parsed, slug, direction = self.parsed, self.slug, self.direction
        problem = _slug_problem(slug)
        self.slug_ok = problem is None
        if problem:
            self.add("bad_slug", f"slug {shorten(str(slug), 40)!r} 不合规：{problem}换一个 slug 再审计。", False)
        try:
            self.base = self.parent_spec = build_parent(parsed, [], slug=slug, direction=direction)
        except FamilyError as error:
            self.add("invalid_family", f"{error}。", True)
            return
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as error:
            self.add("invalid_family", "题族的格式不对，构造不出题目"
                                     f"（{type(error).__name__}: {error}）。请按题族的格式重新给出。", True)
            return
        mutants, tests = parsed.get("mutants"), parsed.get("tests")
        if not isinstance(mutants, list) or not isinstance(tests, dict):
            self.add("invalid_family",
                     "题族必须有 mutant 列表和隐藏测试字典 tests（{标签: {requirement, body}}）。", True)
            return
        try:
            # ValueError：源码里有空字节、孤立的代理项这类编译器不肯读的东西
            self.fixed_tree = ast.dump(ast.parse(self.base["fixed"]))
        except (SyntaxError, ValueError) as error:
            self.add("syntax_error", f"参考解不是合法的 Python（{_where(error)}：{_reason(error)}），"
                                     f"请改正后重新给出 fixed。", True)
            return          # 参考解解析不了，锚点、变体、判分都无从谈起
        self.tests_ok = True
        for location, code, offset in self._texts(self.base)[1:]:
            try:
                tree = ast.parse(code)
            except (SyntaxError, ValueError) as error:
                self.tests_ok = False
                self.add("syntax_error", f"{location}：不是合法的 Python（{_where(error, offset)}："
                                         f"{_reason(error)}），请改正。", True)
                continue
            if not collectable_tests(tree):
                # 一个测试都收集不到时判分会报“没有测试结果”，看起来像判分环境出了问题；其实改一下测试函数的名字就好
                self.tests_ok = False
                self.add("no_test_functions", f"{location}里没有 pytest 收得到的测试函数（{_declared(tree)}）。"
                                              f"pytest 只收集名字以 test 开头的函数，写在模块层，或者写在 Test "
                                              f"开头、没有 __init__ 的类里；只定义 fixture 是收集不到的。请把测试"
                                              f"函数改成这样的写法。", True)
        if not mutants:
            self.add("invalid_family", "题族里没有任何 mutant：每条要求至少要有一个把参考解改坏的 mutant。", True)
        if not tests:
            self.add("invalid_family", "题族里没有任何隐藏测试（tests 是空的）：每条要求至少要有一个隐藏测试。", True)

        # 每个 mutant 单独看：锚点能定位，改坏后的程序合法，而且和参考解、别的变体不是同一个程序
        candidates, programs = [], {}
        for index, mutant in enumerate(mutants):
            try:
                variant = build_variant(parsed, index, slug=slug, direction=direction)
                program = ast.dump(ast.parse(variant["buggy"]))
            except FamilyError as error:
                brief, advice = _mutant_problem(error)
                self.dropped[index] = brief
                self.add("mutant_anchor", f"{error}。{advice}", True)
                continue
            except (SyntaxError, ValueError) as error:
                self.dropped[index] = "改坏后不是合法的 Python"
                self.add("mutant_syntax", f"{mutant_name(mutant, index)}把代码改坏之后不是合法的 Python"
                                          f"（{_where(error)}：{_reason(error)}）：new 片段是 "
                                          f"{shorten(mutant['new'])!r}。改坏后的程序必须还能运行，只是行为不对。", True)
                continue
            same = programs.get(program)
            if program == self.fixed_tree:
                self.dropped[index] = "没有引入缺陷"
                self.add("mutant_duplicate", f"{mutant_name(mutant, index)}改坏后的程序和参考解一模一样，没有引入"
                                             f"任何缺陷：请让 new 和 old 有行为上的差别。", True)
                continue
            if same is not None:
                self.dropped[index] = "和别的 mutant 改坏后的程序相同"
                self.add("mutant_duplicate", f"{mutant_name(mutant, index)}改坏后的程序和第 {same + 1} 个 mutant 的"
                                             f"一模一样，两道变体会是同一道题：请换一种不同的疏忽。", True)
                continue
            programs[program] = index
            candidates.append(index)

        # 再依次放进母题：前一个改动不能让后一个的锚点失灵
        applied = []
        for index in candidates:
            try:
                build_parent(parsed, [*applied, index], slug=slug, direction=direction)
            except FamilyError as error:
                self.dropped[index] = "和别的 mutant 改动重叠"
                self.add("mutant_overlap", f"{error}。它和前面的 mutant 改动了重叠的位置，不能同时放进母题；"
                                           f"请让各个 mutant 改动互不重叠的地方。", True)
                continue
            applied.append(index)
        try:
            self.family = build_family(parsed, slug=slug, direction=direction, indices=applied)
        except FamilyError as error:      # 上面的循环已经逐步验证过，正常到不了这里；真到了也不能丢掉已有的发现
            self.add("invalid_family", f"{error}。", True)
            return
        self.parent_spec = self.family["parent"]
        if not applied:
            # 一个能用的 mutant 都没有时，母题的原程序就是参考解本身：参考补丁是空的，git 不肯应用，
            # 判分器会把它当成环境错误。在原程序末尾多留一个空行，补丁就不是空的，参考解照样判得出来
            self.parent_spec["buggy"] += "\n"
        try:
            validate_specs(family_specs(self.family))
            self.specs_valid = True
        except ValueError as error:
            # slug 不合规时规格可能不合法：那是 slug 的问题，返修题族内容解决不了
            self.add("invalid_spec", f"题目的规格不合法：{error}。", self.slug_ok)
        self.collisions()
        # 到这里 findings 都来自这一项：slug 不合规、撞车都是不可返修的，也算结构不过关
        self.checks["structure"] = self.specs_valid and all(item["repairable"] for item in self.findings)

    def collisions(self) -> None:
        """和现有题池比对：撞了只能换 slug，所以都是不可返修的。"""
        parent = self.family["parent"]
        ids, groups, modules, programs = {}, {}, {}, {}
        for name, pool in self.pools:
            for other in pool:
                label = f"{name} 题池的 {other.get('task_id', '?')}"
                ids.setdefault(other.get("task_id"), label)
                groups.setdefault(other.get("source_group"), label)
                modules.setdefault(other.get("module"), label)
                try:
                    programs.setdefault(ast.dump(ast.parse(other["buggy"])), label)
                except (KeyError, TypeError, ValueError, SyntaxError):
                    continue        # 调用方给的额外题目坏了就跳过：它不会和一个合法的程序相同
        specs = family_specs(self.family)
        for spec in specs:
            if spec["task_id"] in ids:
                who = "母题" if spec is parent else "变体"
                self.add("collision", f"{who}的 task_id {spec['task_id']} 和 {ids[spec['task_id']]} 重复："
                                      f"换一个 slug 才行。", False)
        for field, value, known in (("source_group", parent["source_group"], groups),
                                    ("module", parent["module"], modules)):
            if value in known:
                self.add("collision", f"{field} {value} 和 {known[value]} 重复：换一个 slug 才行。", False)
        for spec in specs:
            program = ast.dump(ast.parse(spec["buggy"]))
            if program in programs:
                original = "母题的原程序" if spec is parent else f"变体 {spec['task_id']} 的原程序"
                self.add("collision", f"{original}和 {programs[program]} 的原程序语法树相同：这不是一道新题，"
                                      f"请换一个程序。", False)
        digest = fixed_hash(parent)
        self.stats["fixed_hash"] = digest
        if digest in self.known:
            self.add("collision", f"参考解和已知题池里的一道题语法树相同（fixed_hash={digest[:12]}…）："
                                  f"请换一个程序，而不只是换写法。", False)

    def _build(self, specs: list[dict]):
        """把题目建成任务目录并逐个读出来（含任务库的泄漏检查）。建不出来就记一条发现，返回 None。"""
        tasks = self.work_dir / "tasks"
        try:
            build_all(tasks, specs=specs)
            store = TaskStore(tasks)
            for spec in specs:
                store.load(spec["task_id"])
            return store
        except Exception as error:     # 不能让一个题族打断整批审计：原因写进发现项
            self.add("grading_error", f"建不出判分用的任务目录，没法审计这个题族：{type(error).__name__}: {error}。"
                                      f"多半是题族里有任务库不接受的地方（文件名、模块名），请对照题族的格式检查；"
                                      f"换一个 slug 也不行，才是判分环境的问题。", False)
            return None

    # 2 参考解
    def reference(self) -> None:
        store = self._build(family_specs(self.family))
        if store is None:
            self.gradable = False      # 任务都建不出来，后面的变体和母题都判不了
            return
        parent = self.family["parent"]
        self.parent_task = store.load(parent["task_id"])
        runs = self.repeat(self.parent_task, parent, parent["fixed"], "reference", REFERENCE_TIMEOUT)
        self.stats["reference"] = [_summary(run) for run in runs]
        self.stats["isolation"] = runs[0]["isolation"]
        if len({_signature(run) for run in runs}) > 1:
            self.unstable("参考解", runs)
        if all(run["strict_pass"] for run in runs):
            self.checks["reference"] = True
            return
        broken = next(run for run in runs if not run["strict_pass"])
        kind = broken.get("error_kind")
        healthy, detail = (False, "")
        if kind and kind != "timeout":
            healthy, detail = _environment_ok(self.work_dir)      # 环境好不好，决定这是不是题族的问题
        if kind == "timeout":
            self.add("reference_timeout", f"参考解在 {REFERENCE_TIMEOUT} 秒内没跑完测试：请检查参考解和测试里"
                                          f"有没有死循环或过长的等待。", True)
        elif kind == "sandbox" and healthy:
            # 判分环境刚用一道完好的小题自检过，是好的：那就是参考解或测试让 pytest 中途退出了
            self.add("reference_no_results",
                     f"参考解的测试没有留下任何结果：{broken['verdict']}。判分环境刚用一道完好的小题自检过，"
                     f"是好的；所以是参考解或测试让 pytest 中途退出了（比如导入时就调用 os._exit、sys.exit）。"
                     f"请去掉这类写法，让 pytest 把测试跑完。", True)
        elif kind:
            cause = (f"判分环境是好的（自检小题 {detail}），问题出在这个题族上，或者出在判分它的过程中"
                     if healthy else f"判分环境自检也没通过（{detail}），请先检查沙箱环境（agentlab doctor），"
                     f"修好以后原样重试")
            self.add("grading_error", f"判分没能完成，没法审计这个题族：{broken['verdict']}。{cause}。", False)
        else:
            self.add("reference_fails", _failure("参考解", broken), True)
        if kind:
            self.gradable = False      # 参考解都判不完，后面的变体和母题只会一个一个地同样失败

    # 3 单缺陷变体
    def variants(self) -> None:
        parent = self.family["parent"]
        for number, index in enumerate(self.family["mutants"]):
            variant = self.family["variants"][number]
            # 变体和母题的题面、测试完全相同：在母题的任务上判它，参考解不必每个变体重跑一遍
            run = self._assess(self.parent_task, parent, variant["buggy"],
                               self.work_dir / f"variant-{number + 1}", MUTANT_TIMEOUT)
            caught, _, failed = _rejection(run)
            mutant = self.parsed["mutants"][index]
            usable = caught and run["public_pass"] is True
            self.records.append(dict(
                index=index, requirement=mutant.get("requirement"), usable=usable, caught=caught,
                why=_diagnosis(run), failed_tests=failed, verdict=run["verdict"]))
            if usable:
                continue
            change = f"把 {shorten(mutant['old'])} 改成 {shorten(mutant['new'])} "
            self.add(*_rejection_finding("mutant", mutant_name(mutant, index), run, change, MUTANT_TIMEOUT,
                                         MUTANT_PUBLIC_ADVICE, "请给这条要求补一个能暴露它的隐藏测试。"))
        self.checks["variants"] = all(record["usable"] for record in self.records)

    # 4 母题
    def parent(self) -> None:
        self.usable = [record["index"] for record in self.records if record["usable"]]
        if len(self.usable) < MIN_MUTANTS:
            # 不可用的有两类：结构上就被排除的（dropped）和判了分不合格的（records）
            detail = ""
            if self.records or self.dropped:
                kept = "、".join(f"第 {index + 1} 个" for index in self.usable) or "一个都没有"
                lost = {**self.dropped, **{record["index"]: BRIEF[record["why"]]
                                           for record in self.records if not record["usable"]}}
                listed = "；".join(f"第 {index + 1} 个（{brief}）" for index, brief in sorted(lost.items()))
                detail = f"（可用的是{kept}；不可用的是{listed}）"
            self.add("too_few_mutants", f"可用的 mutant 只有 {len(self.usable)} 个，不到 {MIN_MUTANTS} 个，母题不成立"
                                        f"{detail}。请先按上面的说明修好不可用的 mutant，或者补新的。", True)
            return
        try:
            self.final = build_family(self.parsed, slug=self.slug, direction=self.direction,
                                      indices=self.usable)
            parent = self.parent_spec = self.final["parent"]
            ast.parse(parent["buggy"])
        except (FamilyError, SyntaxError, ValueError) as error:
            self.add("parent_broken", f"只用可用的 mutant 构造母题时出错：{error}。请让各个 mutant 改动互不重叠的"
                                      f"地方，改坏后的代码仍然是合法的 Python。", True)
            return
        task = self.parent_task
        if parent["buggy"] != self.family["parent"]["buggy"]:
            store = self._build([parent])
            if store is None:
                return                  # 建不出母题的任务目录：发现项已经记下，母题这一项不过关
            task = store.load(parent["task_id"])
        runs = self.repeat(task, parent, parent["buggy"], "parent", REFERENCE_TIMEOUT)
        self.stats["parent"] = [_summary(run) for run in runs]
        stable = len({_signature(run) for run in runs}) == 1
        if not stable:
            self.unstable("母题", runs)
        caught = all(run["public_pass"] is True and _rejection(run)[0] for run in runs)
        if not caught:
            broken = next(run for run in runs if not (run["public_pass"] is True and _rejection(run)[0]))
            who = f"母题（同时带着 {len(self.usable)} 处缺陷）"
            self.add(*_rejection_finding("parent", who, broken, "", REFERENCE_TIMEOUT, PARENT_PUBLIC_ADVICE,
                                         "请给每条要求都补上能暴露它的隐藏测试。"))
        # 每次都被隐藏测试的断言拦住、公开测试通过，并且几次的结论一致，才算过关
        self.checks["parent"] = stable and caught

    # 5 测试要能区分要求
    def distinguish(self) -> None:
        if not self.records:         # 没判过分（参考解判不完、规格不合法）：这一项无从谈起
            return
        usable = [record for record in self.records if record["usable"]]
        sets = [frozenset(record["failed_tests"]) for record in usable]
        pairs = list(itertools.combinations(range(len(usable)), 2))
        same = [(a, b) for a, b in pairs if sets[a] == sets[b]]
        distinct = len(pairs) - len(same)
        self.stats.update(pairs=len(pairs), distinct_pairs=distinct)
        if not pairs or distinct * 2 > len(pairs):
            # 不足两个可用变体时无从比较，不算通过（可用数另有门槛）
            self.checks["tests_distinguish"] = bool(pairs)
            return
        shared = "、".join(sorted(sets[same[0][0]]))
        if len(same) == len(pairs):
            self.add("indistinct_tests", f"隐藏测试区分不了各条要求：{len(usable)} 个可用的变体没通过的隐藏测试"
                                         f"完全一样（{shared}），每个 mutant 都让同一批测试失败。请给每条要求写"
                                         f"一个专门的测试函数，只断言那条要求的行为，让改坏不同要求的变体挂不同"
                                         f"的测试。", True)
            return
        couples = "、".join(f"第 {usable[a]['index'] + 1} 个和第 {usable[b]['index'] + 1} 个" for a, b in same)
        self.add("indistinct_tests", f"隐藏测试区分不了各条要求：{len(usable)} 个可用的变体里有 {len(same)} 对"
                                     f"（{couples}）没通过的隐藏测试完全一样，不同的只占少数。请给这些要求各补"
                                     f"一个专门的测试函数，只断言那条要求的行为。", True)

    # 6 长度
    def length(self) -> None:
        spec = self.parent_spec
        lines = len(spec["fixed"].splitlines())
        prompt, answer = prompt_estimate(spec), rough_tokens(spec["fixed"])
        self.stats.update(fixed_lines=lines, prompt_tokens=prompt, answer_tokens=answer,
                          total_tokens=prompt + answer)
        low, high = FIXED_LINES
        fits = True
        if not low <= lines <= high:
            fits = False
            advice = ("请拆出一部分逻辑，或者删掉与要求无关的内容" if lines > high
                      else "请把它写成一个完整的小程序（把要求里的行为都实现出来），不要只留几行")
            self.add("fixed_lines", f"参考解有 {lines} 行，应在 {low} 到 {high} 行之间：{advice}。", True)
        if prompt + answer > LONG_ESTIMATE:
            fits = False
            self.add("too_long_tokens", f"题面加参考解的估算长度是 {prompt + answer} 个 token（题面 {prompt} + 参考解 "
                                        f"{answer}），超过上限 {LONG_ESTIMATE}：训练样本超长会被整条丢弃。请精简"
                                        f"题面、缩短参考解或公开测试。", True)
        self.checks["length"] = fits

    # 7 安全
    def safety(self) -> None:
        clean = True
        for location, code, offset in self._texts(self.base):
            try:
                tree = ast.parse(code)
            except (SyntaxError, ValueError):
                continue            # 解析不了的情况已经在结构那一项报过
            uses = unsafe_uses(tree, offset)
            if uses:
                clean = False
                self.add("unsafe_code", f"{location}：有 {len(uses)} 处不允许的写法：{'；'.join(uses)}。"
                                        f"{UNSAFE_ADVICE}。", True)
        self.checks["safety"] = clean

    # 8 提示性注释
    def hints(self) -> None:
        inherited = {(text, marker) for _, marker, text in hint_markers(self.base["fixed"])}
        items = [(location, hint_markers(code, offset)) for location, code, offset in self._texts(self.base)]
        # 变体的原程序是参考解加上 mutant 的 new 片段：只报片段新带进来的标记，参考解里已有的上面报过
        for number, variant in enumerate(self.family["variants"] if self.family else [], 1):
            items.append((f"变体 {number} 的原程序", [
                (line, marker, text) for line, marker, text in hint_markers(variant["buggy"])
                if (text, marker) not in inherited]))
        clean = True
        for location, marks in items:
            if marks:
                clean = False
                listed = "、".join(f"第 {line} 行 {marker}" for line, marker, _ in marks)
                self.add("hint_comment", f"{location}：注释里有提示缺陷的标记：{listed}。这些标记会告诉读者"
                                         f"缺陷在哪，请删掉或改写。", True)
        self.checks["no_hints"] = clean

    def result(self) -> dict:
        # 接受的条件：没有不可返修的发现，参考解稳定通过，可用 mutant 不少于 3 个，
        # 母题每次都被拦住且公开测试通过、几次的结论一致，长度和安全合格。第 5、8 项只记发现，不单独卡接受。
        accepted = (all(finding["repairable"] for finding in self.findings)
                    and self.checks["reference"] and self.checks["parent"]
                    and self.checks["length"] and self.checks["safety"]
                    and len(self.usable) >= MIN_MUTANTS)
        mutants = self.parsed.get("mutants")
        self.stats.update(mutants=len(mutants) if isinstance(mutants, list) else 0, usable=len(self.usable),
                          variants=self.records, checks=dict(self.checks))
        return {"accepted": accepted, "family": self.final if accepted else None,
                "findings": self.findings, "usable": list(self.usable), "stats": self.stats}


def audit_family(parsed, *, slug, direction, work_dir, repeats=2, known_fixed_hashes=(), extra_specs=()) -> dict:
    """审计一个题族，八项都查完才返回，不因为第一个问题就停。

    parsed 是 parse_family 返回的题族；slug 决定 task_id、source_group 和文件名（s3_<slug>…），
    direction 是方向短名（进 bug_family、direction 和 stratum）。题目和判分的工作区写在 work_dir 下
    （审计过程的临时产物，要留档请导出返回的 family）。known_fixed_hashes 是已知参考解的
    fixed_hash；extra_specs 是内置题池之外还要避开的题目（至少带 task_id、source_group、module、
    buggy）。每层 pytest 的上限：参考解和母题 REFERENCE_TIMEOUT 秒，变体 MUTANT_TIMEOUT 秒；
    机器很慢时先调大这两个常量再审计。

    返回的 dict：
    - accepted：没有不可返修的发现，参考解 repeats 次都通过且结论一致，可用的 mutant 不少于 3 个，
      母题每次都被隐藏测试的断言拦住、公开测试通过、几次的结论一致，长度和安全都合格。第 5 项（测试
      区分不了要求）和第 8 项（提示性注释）只记发现，不影响 accepted；只想要零发现的题族，调用方自己
      再看 findings 是否为空；
    - family：接受时的题族 ``{"parent", "variants", "mutants"}``，只用可用的 mutant 构造；否则 None；
    - findings：每项 ``{"code", "message", "repairable"}``，按审计的先后排列。message 写成可以直接
      动手改的中文说明；repairable 为假的问题（slug 不合规、撞车、判分没能完成）返修题族内容解决不了；
    - usable：可用 mutant 的下标（在 parsed["mutants"] 里从 0 起），不管接不接受都给出；
    - stats：fixed_lines、prompt_tokens、answer_tokens、total_tokens、fixed_hash、isolation（实际生效的
      隔离）、reference 和 parent（每次判分的结论）、variants（每个变体是否可用、没通过的隐藏测试名）、
      pairs / distinct_pairs、checks（八项各自的通过情况）、duration_ms。

    题族的问题、判分环境的问题、审计程序自己的错都记成发现项；只有调用方传错参数才抛 TypeError / ValueError。
    findings 和 stats 里的文本不带绝对路径的目录部分（见 without_paths），family 原样不动。
    """
    if not isinstance(parsed, dict):
        raise TypeError("parsed must be the dict returned by parse_family")
    if not isinstance(direction, str) or not direction.strip():
        raise ValueError("direction must be a non-empty string")
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    extra_specs = list(extra_specs)          # 先落成列表：生成器只能读一遍，后面还要拿它比对
    if any(not isinstance(spec, dict) for spec in extra_specs):
        raise TypeError("extra_specs must be task specs (dicts)")
    started = time.monotonic()
    work_dir = Path(work_dir)
    try:
        result = _Audit(parsed, slug, direction, work_dir, repeats, set(known_fixed_hashes),
                        existing_pools(extra_specs)).run()
    except Exception as error:      # 最后一道兜底：审计程序自己的错，也不能让整批审计断在一个题族上
        result = {"accepted": False, "family": None, "usable": [],
                  "findings": [{"code": "grading_error",
                                "message": f"审计这个题族时程序出了错，没能得出结论：{type(error).__name__}: "
                                           f"{error}。这不是题族内容的问题：请把这个题族和这条信息报告给维护者。",
                                "repairable": False}],
                  "stats": {"checks": dict.fromkeys(CHECKS, False)}}
    result["stats"]["duration_ms"] = int((time.monotonic() - started) * 1000)
    # 判分的输出和异常的文本里带着工作区、解释器的绝对路径：交给调用方之前统一去掉。family 不动
    roots = tuple(str(path) for path in {work_dir, work_dir.resolve()} if path.is_absolute())
    result["findings"] = _scrub(result["findings"], roots)
    result["stats"] = _scrub(result["stats"], roots)
    return result
