"""把教师的一次回复解析成一个题族，结构不合格时说清楚哪里不对。

一次回复 = 一个题族 = 一个程序 = 一个来源族：theme、instruction、fixed、public_test、tests、mutants
（格式见 prompts.format_spec）。这里只做不用运行代码就能做的检查——结构、类型、锚点、语法树、要求条数；
要跑测试才知道的事（参考解是不是真的全过、mutant 是不是只被隐藏测试拦住、长度有没有超限）留给审计。

检查不过抛 ValueError，消息里逐条写明哪一项不对、内容是什么：这段话原样进返修提示，所以写成给出题人看的
话。检查通过时返回规范化后的题族：源码都化成规范形式，requirement 化成整数，多余的字段丢掉。
"""
from __future__ import annotations

import ast
import json
import re
from collections import Counter
from textwrap import dedent

from agentlab.synth.themes import MAX_REQUIREMENTS, MIN_REQUIREMENTS, SLUG_PATTERN, requirement_count

# 题族的六个部件；多余的字段不报错，规范化时丢掉
REQUIRED_KEYS = ("theme", "instruction", "fixed", "public_test", "tests", "mutants")
# 隐藏测试的标签是文件名 test_<标签>.py 的一部分
LABEL_PATTERN = re.compile(r"[A-Za-z0-9_]+")
# mutant 的锚点至少这么长：太短的片段在参考解里到处都是，起不到定位的作用
MIN_OLD_CHARS = 12
# 题面里编号的行为要求："1）" 或 "1)" 写在行首，或者紧跟在"要求："后面。行中间的"（容量默认 16）"
# "空库存时返回 0）"不是编号
_REQUIREMENT = re.compile(r"(?m)(?:^[ \t]*|[：:][ \t]*)(\d{1,2})[ \t]*[）)]")
# 写在行中间的编号：只用来发现"各条要求挤在同一行"这种写法
_INLINE = re.compile(r"(?<![A-Za-z0-9_.(（])(\d{1,2})[ \t]*[）)]")
# 有的回复把源码包在 markdown 围栏里再放进 JSON 字符串
_FENCED = re.compile(r"\A\s*```[A-Za-z0-9_+-]*[ \t]*\n(.*?)\n[ \t]*```[ \t]*\s*\Z", re.DOTALL)
# 解释器自己给出的"这是哪个版本才有的语法"
_NEWER_SYNTAX = re.compile(r"only supported in Python 3\.(\d+)")
# 锚点每一行最多认多少格缩进的出入（手写 JSON 时片段跟着排版缩进）
_MAX_SHIFT = 24


def requirement_numbers(instruction: str) -> list[int]:
    """题面里编号的行为要求的序号（升序、去重）：只认行首的"1）"，或者紧跟在"要求："后面的。"""
    return sorted({int(match.group(1)) for match in _REQUIREMENT.finditer(instruction)})


def _inline_numbers(instruction: str) -> set[int]:
    return {int(match.group(1)) for match in _INLINE.finditer(instruction)}


def parse_family(text: str, slug=None) -> dict:
    """教师的一次回复 → 规范化的题族；不合格抛 ValueError（消息可以原样给回教师）。

    返回的题族是 ``{"theme", "instruction", "fixed", "public_test", "tests", "mutants"}``：tests 是
    ``{标签: {"requirement": 序号, "body": 测试}}``，mutants 是 ``[{"requirement", "old", "new", "idea"}]``
    （没有给 idea 时是空串）。源码都是规范形式（见 _source），锚点在这个形式里能找到。

    要求的条数总要查：题面里编号的要求必须在 4 到 6 条之间。给了 ``slug`` 时还必须正好是
    requirement_count(slug) 条——条数由 slug 决定，审计不查这一项，流水线解析时把 slug 传进来才算把关。
    """
    expected = None
    if slug is not None:
        if not isinstance(slug, str) or not SLUG_PATTERN.fullmatch(slug):
            raise ValueError(f"slug 只能用小写字母、数字、下划线：{slug!r}")
        expected = requirement_count(slug)
    family = _first_json_object(text)
    problems, normalized = _check(family, expected)
    if problems:
        listed = "\n".join(f"- {problem}" for problem in problems)
        raise ValueError(f"回复不是合格的题族，共 {len(problems)} 处问题：\n{listed}")
    return normalized


def _first_json_object(text) -> dict:
    """回复里的第一个 JSON 对象，允许前后有别的话、允许包在代码围栏里。

    取的是第一个带有题族字段的对象：源码里散落的 ``{}``、tests 里的条目能解析成 JSON 对象，但不是题族，
    跳过它们。回复被截断（或者多了个尾逗号）时外层对象解析不出来，这时要如实报"不完整"，不能拿里面的
    某个小对象去报"键名不对"。
    """
    if not isinstance(text, str):
        raise ValueError(f"回复不是字符串，而是{_kind(text)}")
    decoder = json.JSONDecoder(strict=False)      # 字符串里有没转义的换行、制表符也认
    broken, seen, seen_start = None, None, None
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError as error:
            if broken is None or error.pos > broken[0].pos:     # 记下走得最远的那次失败：它就是没写完的外层对象
                broken = (error, match.start())
            continue
        if not isinstance(value, dict):
            continue
        if set(REQUIRED_KEYS).intersection(value):
            return value
        if value and seen is None:                # 记下第一个像样的对象：能解析、但一个题族的字段都没有
            seen, seen_start = value, match.start()
    # 能解析的小对象落在那个解析不出来的大对象的范围里：它是大对象的一部分，问题是回复没写完
    if broken is not None and (seen is None or broken[1] < seen_start < broken[0].pos):
        raise ValueError(_incomplete_reply(text, broken[0]))
    if seen is None:
        raise ValueError("回复里找不到题族的 JSON 对象：只输出一个 JSON 对象，可以放在 json 代码围栏里，"
                         f"不要别的内容。回复的开头是：{_show(text, 80)!r}")
    raise ValueError("回复里的 JSON 对象不是题族：缺少全部必需的键（" + "、".join(REQUIRED_KEYS)
                     + f"），实际的键是 {list(seen)!r}")


def _incomplete_reply(text: str, error: json.JSONDecodeError) -> str:
    """回复没写完或者 JSON 有笔误时的话：说清在哪一步解析不下去、多半是什么原因。"""
    place = f"第 {error.lineno} 行第 {error.colno} 列"
    if "Unterminated string" in error.msg:
        where, why = f"一个字符串从{place}开始，直到回复的结尾都没有收尾", "多半是回复被截断了"
    elif error.pos >= len(text.rstrip()):
        where, why = f"解析到{place}时就没有内容了", "回复多半是被截断了，没有写完"
    elif "Expecting property name" in error.msg:
        where, why = f"解析到{place}时出错", "多半是对象的最后一项后面多了一个逗号"
    else:
        where, why = f"解析到{place}时出错（{error.msg}）", "可能是没写完，也可能多写了或少写了逗号、引号"
    return (f"回复里的 JSON 对象不完整或者格式有错：{where}，{why}。"
            "请重新输出完整的 JSON 对象（可以放在 json 代码围栏里），不要别的内容。")


def _check(family: dict, expected) -> tuple[list[str], dict]:
    """逐项检查，返回（问题列表，规范化后的题族）。没有问题时后者的每一项都已就绪。"""
    problems = []
    missing = [key for key in REQUIRED_KEYS if key not in family]
    if missing:
        problems.append(f"缺少必填键：{'、'.join(missing)}")

    out = {}
    for key in ("theme", "instruction"):
        value = family.get(key)
        if key in family and not _is_text(value):
            problems.append(f"{key} 应是非空的字符串，现在是{_kind(value)}")
        # 题面的各行都带着同样的缩进，只会是手写 JSON 时的排版缩进：去掉
        out[key] = dedent(_lines(value)).strip() if isinstance(value, str) else ""

    fixed, margin, fixed_ok = "", "", False
    if "fixed" in family:
        value = family["fixed"]
        if not _is_text(value):
            problems.append(f"fixed 应是非空的字符串，现在是{_kind(value)}")
        else:
            fixed, margin = _source(value)
            error = _syntax_error(fixed)
            if error:
                problems.append(f"fixed 不是合法的 Python：{error}")
            else:
                fixed_ok = True
    out["fixed"] = fixed

    out["public_test"] = ""
    if "public_test" in family:
        value = family["public_test"]
        if not _is_text(value):
            problems.append(f"public_test 应是非空的字符串，现在是{_kind(value)}")
        else:
            out["public_test"], _ = _source(value)
            error = _syntax_error(out["public_test"])
            if error:
                problems.append(f"public_test 不是合法的 Python：{error}")

    tests, tests_ok, found = _check_tests(family)
    problems += found
    mutants, mutants_ok, found = _check_mutants(family, fixed, margin, fixed_ok)
    problems += found
    # 题面、tests、mutants 成形的才拿来对；缺了哪一个，上面已经报过，不换个说法再来一遍
    if out["instruction"] and (tests_ok or mutants_ok):
        problems += _coverage_problems(out["instruction"], tests if tests_ok else None,
                                       mutants if mutants_ok else None, expected)
    return problems, {**out, "tests": tests, "mutants": mutants}


def _check_tests(family: dict) -> tuple[dict, bool, list[str]]:
    """tests 的每一项：标签合法、requirement 是整数、body 是合法的 Python。第二个返回值说 tests 是不是非空的对象。"""
    if "tests" not in family:
        return {}, False, []
    value = family["tests"]
    if not isinstance(value, dict):
        return {}, False, [f"tests 应是对象（标签 → {{\"requirement\": 序号, \"body\": 测试}}），"
                           f"现在是{_kind(value)}"]
    if not value:
        return {}, False, ["tests 是空的：每条要求至少要有一个测试"]
    problems, tests = [], {}
    for label, entry in value.items():
        if not LABEL_PATTERN.fullmatch(label):
            problems.append(f"tests 的标签 {_show(label)!r} 不合法：只能用字母、数字、下划线"
                            "（它要当文件名 test_<标签>.py 的一部分）")
        where = f"tests[{_show(label)!r}]"
        if not isinstance(entry, dict):
            problems.append(f"{where}应是对象（requirement 和 body），现在是{_kind(entry)}")
            continue
        requirement = _requirement_number(entry.get("requirement"))
        if requirement is None:
            problems.append(f"{where}的 requirement 应是一个整数（对应的要求序号），现在是"
                            f"{_kind(entry.get('requirement'))}：一条测试只对应一条要求，覆盖几条就拆成几个标签")
        body = ""
        if not _is_text(entry.get("body")):
            problems.append(f"{where}的 body 应是非空的字符串，现在是{_kind(entry.get('body'))}")
        else:
            body, _ = _source(entry["body"])
            error = _syntax_error(body)
            if error:
                problems.append(f"{where}的 body 不是合法的 Python：{error}")
        tests[label] = {"requirement": requirement, "body": body}
    for label, count in Counter(label.lower() for label in value).items():
        if count > 1:
            problems.append(f"tests 的标签只差大小写：{label!r} 出现 {count} 次，文件名会撞在一起")
    return tests, True, problems


def _check_mutants(family: dict, fixed: str, margin: str, fixed_ok: bool) -> tuple[list, bool, list[str]]:
    """mutants 的每一项：三个必需字段的类型；参考解可用时再查锚点（old 不等于 new、够长、在参考解里找得到）。

    改坏之后是不是合法的 Python、两个 mutant 是不是同一处缺陷，要构造出变体才看得清，留给审计。
    """
    if "mutants" not in family:
        return [], False, []
    value = family["mutants"]
    if not isinstance(value, list):
        return [], False, ["mutants 应是数组，每一项是 {\"requirement\", \"old\", \"new\", \"idea\"}，"
                           f"现在是{_kind(value)}"]
    if not value:
        return [], False, ["mutants 是空的：每条要求至少要有一个 mutant"]
    problems, mutants = [], []
    for index, entry in enumerate(value, 1):
        where = f"mutants 第 {index} 项"
        if not isinstance(entry, dict):
            problems.append(f"{where}应是对象（requirement、old、new、idea），现在是{_kind(entry)}")
            continue
        requirement = _requirement_number(entry.get("requirement"))
        if requirement is None:
            problems.append(f"{where}的 requirement 应是一个整数（对应的要求序号），现在是"
                            f"{_kind(entry.get('requirement'))}")
        else:
            where += f"（对应第 {requirement} 条要求）"
        for name in ("old", "new"):
            if not isinstance(entry.get(name), str):
                problems.append(f"{where}的 {name} 应是字符串，现在是{_kind(entry.get(name))}")
        idea = entry.get("idea", "")
        if not isinstance(idea, str):
            problems.append(f"{where}的 idea 应是一句话的字符串，现在是{_kind(idea)}")
            idea = ""
        old = _lines(entry["old"]) if isinstance(entry.get("old"), str) else ""
        new = _lines(entry["new"]) if isinstance(entry.get("new"), str) else ""
        if fixed_ok and isinstance(entry.get("old"), str) and isinstance(entry.get("new"), str):
            old, new = _locate(fixed, old, new, margin)
            problem = _anchor_problem(where, old, new, fixed, margin)
            if problem:
                problems.append(problem)
        mutants.append({"requirement": requirement, "old": old, "new": new, "idea": idea})
    return mutants, True, problems


def _anchor_problem(where, old, new, fixed, margin="") -> str | None:
    """锚点本身的问题，没有就返回 None。"""
    if old == new:
        return f"{where}的 old 和 new 相同，替换之后程序没有变化：{_show(old)!r}"
    if len(old) < MIN_OLD_CHARS:
        return (f"{where}的 old 太短：只有 {len(old)} 个字符，至少 {MIN_OLD_CHARS} 个。"
                f"old 要带上足够的上下文，保证它在 fixed 里只出现一次：{_show(old)!r}")
    if fixed.count(old) == 0:
        return (f"{where}的 old 在 fixed 里找不到：{_show(old, 120)!r}。"
                f"old 要原样照抄 fixed 里的片段，包括缩进和换行{_near_hint(fixed, old, new, margin)}")
    return None


def _near_hint(fixed, old, new, margin) -> str:
    """锚点只是抄法不一致、_locate 又修不了的时候，说清楚差在哪、改了之后能找到几处。

    找得到但没被认，只有三种可能：改了之后在 fixed 里不止一处（教师要把 old 写长一点才能定位）；改了之后
    old 不到 MIN_OLD_CHARS；照这个修法改完 new、换上去之后程序不再是合法的 Python。修法按教师看得懂的程度排：
    先说首尾的空白，再说缩进。
    """
    plans = []
    for tidy, description in _TIDYINGS:
        anchor, replacement = tidy(old), tidy(new)
        if anchor != old:
            plans.append((description, anchor, replacement))
        plans += _reindents(fixed, anchor, replacement, margin)
    for description, moved, replaced in plans:
        count = fixed.count(moved)
        if not count:
            continue
        if count > 1:
            why = "，但那样就不唯一了：请把 old 写得长一点，带上只属于这一处的上下文"
        elif len(moved) < min(len(old), MIN_OLD_CHARS):
            why = f"，但那样 old 不到 {MIN_OLD_CHARS} 个字符：old 要带上足够的上下文"
        elif not _safe(fixed, moved, replaced):
            why = "，但照这个修法改完 new 再换上去，程序就不是合法的 Python：old 和 new 每一行的缩进要照 fixed 里写"
        else:
            continue                    # 这样的修法 _locate 已经认了，走不到这里
        return f"（{description}能在 fixed 里找到 {count} 处{why}）"
    return ""


def _coverage_problems(instruction, tests, mutants, expected) -> list[str]:
    """题面里的每条要求都要有自己的测试和 mutant，条数要对；tests、mutants 也不能指向题面里没有的要求。

    ``tests``、``mutants`` 是成形的那一部分，不成形的那一边传 None、不查。``expected`` 是这个主题要求的条数
    （由 slug 决定），不知道时传 None。
    """
    numbers, inline = requirement_numbers(instruction), _inline_numbers(instruction)
    if not numbers:
        if inline:
            return ["题面里的每条要求要单独占一行，序号（1）2）3）…）写在行首：现在各条要求挤在同一行里，"
                    "分不出是哪几条"]
        return ["题面 instruction 里找不到编号的行为要求：先用一句话说明要修复什么，"
                "再换行用 1）2）3）… 逐条编号"]
    problems = []
    rest = sorted(inline - set(numbers))
    if len(rest) >= 2 and rest[:2] == [numbers[-1] + 1, numbers[-1] + 2]:
        return [f"题面里的每条要求要单独占一行，序号写在行首：第 {rest[0]} 条起的序号写在了行中间"]
    if numbers != list(range(1, len(numbers) + 1)):
        problems.append(f"题面里的要求序号是 {numbers}，应从 1 起连续编号（写成 1）2）3）…）")
    if not MIN_REQUIREMENTS <= len(numbers) <= MAX_REQUIREMENTS:
        problems.append(f"题面里编号的行为要求有 {len(numbers)} 条，应在 {MIN_REQUIREMENTS} 到 "
                        f"{MAX_REQUIREMENTS} 条之间")
    elif expected is not None and len(numbers) != expected:
        if len(numbers) > expected:
            advice = f"请删掉或合并 {len(numbers) - expected} 条，tests、mutants 里对应的也去掉"
        else:
            advice = f"请补上 {expected - len(numbers)} 条，每条都要有自己的测试和 mutant"
        problems.append(f"题面里编号的行为要求有 {len(numbers)} 条，这道题要正好 {expected} 条"
                        f"（条数由模块名决定）：{advice}")
    known, total = set(numbers), f"（题面共 {len(numbers)} 条要求）"
    for label, item in (tests or {}).items():
        if item["requirement"] is not None and item["requirement"] not in known:
            problems.append(f"tests[{_show(label)!r}]的 requirement 是 {item['requirement']}，"
                            f"题面里没有这一条要求{total}")
    for index, item in enumerate(mutants or [], 1):
        if item["requirement"] is not None and item["requirement"] not in known:
            problems.append(f"mutants 第 {index} 项的 requirement 是 {item['requirement']}，"
                            f"题面里没有这一条要求{total}")
    tested = {item["requirement"] for item in (tests or {}).values()}
    broken = {item["requirement"] for item in (mutants or [])}
    for number in numbers:
        if tests is not None and number not in tested:
            problems.append(f"第 {number} 条要求在 tests 里没有对应的测试：每条要求至少要有一个，"
                            "并用 requirement 注明序号")
        if mutants is not None and number not in broken:
            problems.append(f"第 {number} 条要求在 mutants 里没有对应的 mutant：每条要求至少要有一个")
    return problems


def _source(text: str) -> tuple[str, str]:
    """一份源码的规范形式，以及它去掉的公共缩进。

    规范形式同 hard_tasks_v3_common._task：去掉 markdown 围栏和公共缩进、首尾空白，以一个换行结束。
    整个文件带一层公共缩进时教师照抄的锚点也带着它，所以把去掉的那一段缩进一并返回。
    """
    code = _unfence(_lines(text))              # 先统一换行：围栏的正则按 \n 写
    canonical = dedent(code).strip() + "\n"
    first = next((line for line in code.splitlines() if line.strip()), "")
    kept = next((line for line in canonical.splitlines() if line.strip()), "")
    margin = first[:len(first) - len(kept)] if kept and first.endswith(kept) else ""
    return canonical, margin


def _unfence(text: str) -> str:
    match = _FENCED.match(text)
    return match.group(1) if match else text


def _lines(text: str) -> str:
    """换行一律写成 \\n：带 \\r 的源码进补丁以后，每一行都会多出一个看不见的字符。"""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _locate(fixed: str, old: str, new: str, margin: str) -> tuple[str, str]:
    """锚点和替换片段在规范形式的参考解里的写法。

    抄得和参考解一模一样的原样返回。抄法上的不一致也认，不然教师明明是照抄的片段却要返修一轮：
      * 只是首尾的空白（或者只有空白的行）抄得不一样；
      * 整个文件带一层公共缩进时，教师照抄的片段也带着它；
      * 片段的每一行都比参考解里多或者少同样的空格（手写 JSON 时跟着排版缩进）。
    后几种可以叠在一起。不管怎么修，都要同时满足三条才认（改动最少的修法优先）：修完之后恰好对上一处；
    修完之后锚点不会变得太短；把 new 照同样的办法修好、换上去之后程序还是合法的 Python。哪一条不满足，
    留给 _anchor_problem 说清楚。"""
    if old in fixed:
        return old, new
    for tidy, _ in _TIDYINGS:
        anchor, replacement = tidy(old), tidy(new)
        if _unique(fixed, anchor, len(old)) and _safe(fixed, anchor, replacement):
            return anchor, replacement
        for _, moved, replaced in _reindents(fixed, anchor, replacement, margin):
            if _unique(fixed, moved, len(old)) and _safe(fixed, moved, replaced):
                return moved, replaced
    return old, new


def _unique(fixed, text, original) -> bool:
    """text 在 fixed 里恰好出现一次，而且（除非原来就短）不比 MIN_OLD_CHARS 短。"""
    return fixed.count(text) == 1 and len(text) >= min(original, MIN_OLD_CHARS)


def _safe(fixed, old, new) -> bool:
    """自己修出来的锚点不能把参考解改坏：换上去之后，程序还得是合法的 Python。"""
    return _syntax_error(fixed.replace(old, new)) is None


def _reindents(fixed, old, new, margin):
    """缩进上的修法，按可能的程度排，每种是 (说明, 修过之后的 old, 修过之后的 new)：
    去掉整个文件的那层公共缩进；给每一行加上或者去掉同样多的前导空格（取改动最少的那一种）。"""
    plans = []
    if margin:
        moved = _unindent(old, margin)
        plans.append(("去掉整个文件的那层公共缩进之后", moved, _follow(old, moved, new)))
    shift = _shift(fixed, old)
    if shift is not None:
        moved = _move(old, shift)
        plans.append((f"给每一行{'加上' if shift > 0 else '去掉'} {abs(shift)} 个前导空格之后",
                      moved, _follow(old, moved, new)))
    return plans


def _follow(old: str, moved: str, new: str) -> str:
    """old 修掉缩进上的出入之后是 moved；new 跟不跟着修。

    照同一种写法抄的 new 要跟着修。但 new 本来的缩进就已经等于修好之后的 old 时（它写对了，只有 old 抄多了
    或者抄少了缩进），不能再动它，不然替换进去的代码会丢掉缩进。
    """
    if not new.strip() or (_indent(new) == _indent(moved) != _indent(old)):
        return new
    return _move(new, _indent(moved) - _indent(old))


def _indent(text: str) -> int:
    line = next((line for line in text.split("\n") if line.strip()), "")
    return len(line) - len(line.lstrip(" "))


def _shift(fixed: str, text: str) -> int | None:
    """让 text 在 fixed 里出现，要给它的每一行加（正数）或者去掉（负数）多少个前导空格：取改动最少的那一种。

    修完之后锚点不能比 MIN_OLD_CHARS 还短（除非它本来就短）。对不上返回 None。
    """
    if not text:
        return None
    shortest = min(len(text), MIN_OLD_CHARS)
    for size in range(1, _MAX_SHIFT + 1):
        for shift in (-size, size):
            moved = _move(text, shift)
            if moved in fixed and len(moved) >= shortest:
                return shift
    return None


def _move(text: str, shift: int) -> str:
    """每一行都加上（shift 为正）或者去掉（shift 为负）|shift| 个前导空格；空行不动。"""
    lines = text.split("\n")
    if shift > 0:
        return "\n".join(" " * shift + line if line.strip() else line for line in lines)
    return "\n".join(line[min(-shift, len(line) - len(line.lstrip(" "))):] if line.strip() else line
                     for line in lines)


def _unindent(text: str, margin: str) -> str:
    return "\n".join(line[len(margin):] if line.startswith(margin) else line for line in text.split("\n"))


def _tidy_anchor(text: str) -> str:
    """锚点的干净写法：去掉开头和结尾只有空白的行，去掉最后一行行尾的空白。

    手写 JSON 时，字符串的第一行常常是排版换行、最后一行是收尾引号前的缩进。结尾那个空白行前面的换行是
    内容的一部分，留着：``"\\n    a\\n  "`` 化成 ``"    a\\n"``，``"\\n    a  "`` 化成 ``"    a"``。
    第一行开头的空白不动：那是它的缩进。
    """
    lines = text.split("\n")
    ends_blank = bool(lines) and not lines[-1].strip()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if lines:
        lines[-1] = lines[-1].rstrip()
    return "\n".join(lines) + ("\n" if lines and ends_blank else "")


# 抄法不一致时先收拾锚点首尾的空白（按可能的程度排），每一种再试一遍缩进上的修法。只收拾一头的排在后面：
# 它们会把第一行的缩进也抹掉，对上之后锚点就不再是原样
_TIDYINGS = ((_tidy_anchor, "去掉首尾只有空白的行和行尾的空白之后"), (str.strip, "去掉首尾的空白之后"),
             (str.lstrip, "去掉开头的空白之后"))


def _syntax_error(code: str) -> str | None:
    """按 Python 3.10 的语法判：解释器比 3.10 新时，也不能放行 3.10 上跑不了的程序。"""
    try:
        ast.parse(code, feature_version=(3, 10))
    except SyntaxError as error:
        newer = _NEWER_SYNTAX.search(error.msg)
        hint = f"；这是 Python 3.{newer.group(1)} 才有的语法，程序要能在 3.10 上运行" if newer else ""
        return f"{error.msg}（第 {error.lineno} 行）{hint}"
    except ValueError as error:                  # 例如源码里有空字节
        return str(error)
    return None


def _requirement_number(value) -> int | None:
    """requirement 字段的整数：教师常把它写成字符串 "2"。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdecimal():
        return int(value)
    return None


def _is_text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _kind(value) -> str:
    """消息里说明一个值是什么：类型，外加简短的内容。直接接在"现在是"后面，所以英文前面留一个空格。"""
    if isinstance(value, str):
        return "空字符串" if not value.strip() else f"字符串 {_show(value, 40)!r}"
    if isinstance(value, dict):
        return f"对象，键是 {list(value)!r}"
    if isinstance(value, list):
        return f"长度 {len(value)} 的数组"
    return f" {type(value).__name__} {value!r}" if value is not None else " null"


def _show(text, limit: int = 70) -> str:
    """消息里引用一段内容：压掉换行，太长就截断。"""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[:limit] + "…"


__all__ = ["LABEL_PATTERN", "MIN_OLD_CHARS", "REQUIRED_KEYS", "parse_family", "requirement_numbers"]
