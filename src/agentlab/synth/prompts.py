"""向教师要一个题族的提示：首轮怎么出题，返修时怎么把上一版和问题一起给回去。

这里只拼消息（[system, user]），不发请求：教师接口的读法同 scripts/collect_data.py，由调用方负责。
每一轮都是单独的一次对话，教师看不到上一轮的 system，所以返修时把硬性要求原样再给一遍。

提示里写死的数字（锚点最短长度）取自 parse.py 的常量，要求条数的范围取自 themes.py；标签能用哪些字符，
提示里的说法和 parse 的 LABEL_PATTERN 一致。提示和解析不会各说各的。
"""
from __future__ import annotations

import ast
import importlib
import re
from textwrap import dedent

from agentlab.synth import parse
from agentlab.synth.themes import (
    MAX_REQUIREMENTS, MIN_REQUIREMENTS, SLUG_PATTERN, Domain, Direction, requirement_count, resolve_direction,
    resolve_theme, stable_hash,
)

# 合成题的量级照现有难题定：参考解 30 到 70 行（现有的是 16 到 71 行、中位 52），4 到 6 条编号要求
FIXED_LINES = (30, 70)
# 整道题粗估不超过 1900 个 token（审计按 scripts/self_check_pool.py 的算法判：汉字约 1.05 个、其余字符约
# 0.3 个一个）。合成题的原程序带着全部缺陷，长度同参考解，不像现有题的原程序短一大截（现有的题按这个算法
# 有 7/40 会超），所以除了行数还有下面两个上限。三个上限都写满、题面取现有题里最密的那种、母题比参考解长
# 一成时，粗估约 1670；题面全是汉字也有余量（tests 里有测试盯着）
MAX_FIXED_CHARS = 1500
MAX_INSTRUCTION_CHARS = 560
MAX_PUBLIC_TEST_LINES = 12
# 风格参照里的参考解最多给这么多行
EXAMPLE_LINES = 150

_ROLE = (
    "你是一位资深 Python 工程师，也是一位严谨的出题人。你出的题用来训练和考察另一个模型："
    "它拿到有缺陷的程序和题面，要把程序修到通过隐藏测试。所以程序要像真实项目里的代码，"
    "缺陷要像人真的会犯的疏忽，题面、测试和缺陷要互相对得上、没有歧义。"
)

# 六条硬性要求，一轮一条。首轮和返修都用它，所以单独放在一起；里面不提风格参照（返修时未必带）
HARD_REQUIREMENTS = (
    "程序是一个自包含的单文件模块，只用标准库，不使用 Python 3.10 之后才有的语法和标准库；threading 可以用，"
    "锁序和条件变量方向的题必须用到它。不联网、不起子进程、不读写文件、不用 eval、exec、compile、__import__ "
    "和 input；真实系统里靠文件、网络、数据库完成的事，在这里用内存里的结构或者调用方传入的回调代替。"
    "凡是与时间有关的行为都要可以注入：时钟通过参数或属性传入（例如 clock 回调或 now 参数），"
    "测试不依赖真实时间；等待只允许 time.sleep 加一个不超过 0.2 秒的字面量参数，能不用就不用。",
    f"fixed 是完全正确的参考解，{FIXED_LINES[0]} 到 {FIXED_LINES[1]} 行、全文不超过 {MAX_FIXED_CHARS} 个字符，"
    "顶格写，不带 markdown 围栏。可以带简短的中文注释，但不要出现 BUG、FIXME、TODO、XXX、HACK 之类的标记。",
    "instruction 是题面：先用一句话说明这个程序做什么、要修复什么，再换行用 1）2）3）… 列出编号的行为要求，"
    f"每条单独占一行，从 1 起连续编号，条数等于下面指定的数目，整段不超过 {MAX_INSTRUCTION_CHARS} 个字符。"
    "每条要求都要有一个针对它的测试函数（只断言这条要求的行为），并用 requirement 注明对应的要求序号。",
    "mutants 是这个程序容易出的疏忽，每条要求至少对应一个。old 是 fixed 里恰好出现一次的片段，"
    f"要带上足够的上下文（完整的一行或几行），至少 {parse.MIN_OLD_CHARS} 个字符；new 是把它改坏后的片段，"
    "要像真实的疏忽：漏了一步、顺序颠倒、条件写反、用错变量、忘了恢复状态。new 不能造成语法错误，"
    "也不能让公开测试失败；不同的 mutant 改坏之后得到的程序不能相同。",
    "public_test 只展示接口和最基本的行为：1 到 2 个测试函数，只用 assert，"
    f"不超过 {MAX_PUBLIC_TEST_LINES} 行。每个 mutant 单独出现时（也就是 fixed 只替换这一处）公开测试仍然要通过，"
    "缺陷只能被隐藏测试暴露。",
    "只输出一个 JSON 对象，可以放在 json 代码围栏里；不要输出任何解释。",
)
REQUIREMENTS = "硬性要求（每一条都必须满足）：\n" + "\n".join(
    f"{number}. {text}" for number, text in enumerate(HARD_REQUIREMENTS, 1))

# 题面的写法：有风格参照时说"照它"，返修时说"沿用上一版"
_INSTRUCTION_STYLE = "写清方法名、参数、返回值，以及抛出的异常类型"

_SELF_CHECK = (
    "输出之前逐条自查：fixed 能通过 public_test 和 tests 里的全部测试；"
    "把任何一个 mutant 的 old 换成 new 之后，对应那条要求的测试会失败，而 public_test 仍然通过；"
    "每条要求都至少有一个测试和一个 mutant；程序不碰文件、网络、子进程和真实时间。"
)

_REPAIR_HEAD = (
    "你是一位资深 Python 工程师，也是一位严谨的出题人。你上一版出的题没有通过检查，请按列出的问题把它改好。"
    "改好之后的题仍然要满足下面全部硬性要求：\n"
)


def module_name(slug) -> str:
    """模块名（不带 .py）：s3_<slug>。slug 不合法时抛 ValueError，它要进文件名和 task_id。"""
    if not isinstance(slug, str) or not SLUG_PATTERN.fullmatch(slug):
        raise ValueError(f"slug 只能用小写字母、数字、下划线：{slug!r}")
    return f"s3_{slug}"


def pick_example(direction, slug, pool=None) -> dict:
    """从现有难题里挑一道同方向的题做风格参照。

    给题面全文、参考解（超过 EXAMPLE_LINES 行只给前 EXAMPLE_LINES 行）和一个隐藏测试文件的正文。挑哪一道、
    给哪个测试文件都由 slug 决定：同一个 slug 永远挑到同一道。

    现有难题里有不少不合这份提示的硬性要求：带只读依赖文件、参考解太长、题面太长、读写文件。风格参照是
    让教师照着写详细程度的，所以先在完全合规的题里挑；一个方向里没有完全合规的题时，挑不合规处最少的，
    把这些地方列在 ``caveats`` 里，由提示明说"这几处不要学"。隐藏测试文件也从不带这些写法的里面挑。

    ``pool`` 是 {方向短名: [题目规格]}，不给就从 themes.DIRECTIONS 指向的现有模块里读。
    """
    direction = resolve_direction(direction)
    if pool is None:
        specs = _existing_specs(direction)
    else:
        specs = list(pool.get(direction.name, ()))
    if not specs:
        raise ValueError(f"方向 {direction.name} 没有可供参照的现有题")
    options = []
    for spec in specs:
        caveats, clean = _deviations(spec)
        options.append((len(caveats), spec, caveats, clean))
    fewest = min(count for count, _, _, _ in options)
    best = [option for option in options if option[0] == fewest]
    _, spec, caveats, clean = best[stable_hash(f"example:{slug}") % len(best)]
    test_file = clean[stable_hash(f"test:{slug}") % len(clean)]
    lines = spec["fixed"].splitlines()
    return {
        "task_id": spec["task_id"],
        "instruction": spec["instruction"],
        "fixed": "\n".join(lines[:EXAMPLE_LINES]) + "\n",
        "test_file": test_file,
        "test_body": spec["private_tests"][test_file],
        "truncated": len(lines) > EXAMPLE_LINES,
        "caveats": caveats,
    }


def generation_messages(theme, example, slug) -> list[dict]:
    """向教师要一个题族的消息：``[system, user]``。

    ``theme`` 是 plan_themes 给的 (方向, 领域)，``example`` 是 pick_example 的结果，``slug`` 决定模块名和
    要求条数。
    """
    direction, domain = resolve_theme(theme)
    module = module_name(slug)
    sections = [f"请按下面的主题出一道「{direction.label}」方向的代码修复题。", _theme_section(direction, domain),
                _count_section(slug)]
    if example:
        sections.append(_example_section(example))
    sections += [format_spec(module), _SELF_CHECK]
    return [{"role": "system", "content": f"{_ROLE}\n\n{REQUIREMENTS}"},
            {"role": "user", "content": "\n\n".join(sections)}]


def repair_messages(previous_raw, findings, theme, slug, example=None) -> list[dict]:
    """返修的消息：把上一版原样给出，逐条列出问题，要求只输出修改后的完整 JSON 对象。

    ``findings`` 是问题的列表：字符串、异常、审计的一条发现（带 message 的字典）都行，单独传一个也行，
    每一条在提示里占一行。``theme``、``slug`` 同 generation_messages。

    上一版是空的时这等于重新出题：附上风格参照和自查清单（``example`` 不给就按方向自己挑）。上一版不是空的
    时候默认不带风格参照——上一版本身就是写法的样本；调用方给了 ``example`` 才附上。
    """
    direction, domain = resolve_theme(theme)
    module = module_name(slug)
    blank = not isinstance(previous_raw, str) or not previous_raw.strip()
    if example is None and blank:
        example = pick_example(direction, slug)
    opening = "请按下面的主题出一道代码修复题。" if blank else "请修改下面这道没有通过检查的题。"
    sections = [opening, _theme_section(direction, domain), _count_section(slug)]
    if example:
        sections.append(_example_section(example))
    sections += [_previous_section(previous_raw), _findings_section(findings), format_spec(module)]
    if example:
        sections.append(_SELF_CHECK)
    style = "题面照风格参照的写法：" if example else "题面的写法沿用上一版："
    tail = (f"\n{style}{_INSTRUCTION_STYLE}。只改与问题有关的地方，没有问题的部分原样保留；"
            "改完之后整道题仍然要自洽（题面、测试、mutants 互相对得上）。只输出修改后的完整 JSON 对象，"
            "不要只给差异，也不要任何解释。")
    return [{"role": "system", "content": f"{_REPAIR_HEAD}{REQUIREMENTS}{tail}"},
            {"role": "user", "content": "\n\n".join(sections)}]


def format_skeleton() -> str:
    """题族的 JSON 骨架：本身是合法的 JSON，值的位置写的是对值的要求。"""
    return dedent('''\
        {
          "theme": "一句话：这个程序是做什么的",
          "instruction": "题面：先用一句话说明这个程序做什么、要修复什么，再换行用 1）2）3）… 列出编号的行为要求",
          "fixed": "完全正确的主文件源码",
          "public_test": "公开测试的函数体",
          "tests": {
            "<标签>": {"requirement": 1, "body": "对应第 1 条要求的测试函数，可以有几个"}
          },
          "mutants": [
            {"requirement": 1, "old": "在 fixed 里恰好出现一次的片段", "new": "把它改坏的片段",
             "idea": "这处缺陷是什么疏忽（一句话）"}
          ]
        }''')


def format_notes(module: str) -> str:
    """骨架后面跟着的注解：标签的字符、自动加上的 import、多线程测试的写法。"""
    return "\n".join([
        "说明：",
        "- tests 的标签只用字母、数字和下划线，建议用小写英文单词（例如 rename、bulk_update）；"
        "它是隐藏测试文件名 test_<标签>.py 的一部分，一条要求可以拆成几个标签。",
        "- 每个要求序号都要在 tests 和 mutants 里出现，requirement 写序号的数字。",
        f"- 模块名是 {module}.py，程序就写在这一个文件里。隐藏测试文件会自动在开头加上 import pytest 和"
        f" from {module} import *，body 里不要再写这两行；用到 threading 等其他标准库时，在 body 的最前面"
        "自己 import。公开测试文件只会自动加上这个 from 行，所以 public_test 里只用 assert，不要用 pytest。"
        "测试只通过模块的公开接口检查行为；等待都要有上限（带 timeout 的 wait、join），"
        "在有缺陷的程序上可能卡住的线程要设成 daemon。",
    ])


def format_spec(module: str) -> str:
    """输出格式的说明：JSON 骨架加注解。提示里原样给出；parse.py 按同样的规则检查教师的回复。"""
    return ("【输出格式】只输出下面这一个 JSON 对象，键名和结构完全一致（值的位置写的是要求，不是要照抄的内容）：\n"
            f"{_fence(format_skeleton(), 'json')}\n{format_notes(module)}")


def _existing_specs(direction: Direction) -> list:
    module = importlib.import_module(direction.module)
    return list(getattr(module, direction.specs))


# 提示里禁用的写法里，靠 import 和内建函数名就看得出来的这些
_BANNED_MODULES = ("os", "pathlib", "subprocess", "shutil", "io", "socket", "http", "urllib", "tempfile",
                   "ctypes", "multiprocessing", "asyncio")
_BANNED_CALLS = ("open", "eval", "exec", "compile", "input", "__import__")


def _banned_io(code: str) -> bool:
    """源码里有没有 import 上面这些模块，或者直接调用 open() 这类函数。解析不了的源码也当作有。"""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(alias.name.split(".")[0] in _BANNED_MODULES for alias in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in _BANNED_MODULES:
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _BANNED_CALLS:
            return True
    return False


def _deviations(spec: dict) -> tuple[list[str], list[str]]:
    """一道现有题和这份提示的硬性要求不一致的地方（每处一句话），以及它哪些隐藏测试文件里没有被禁的写法。

    测试文件里没有干净的就全都算不干净。
    """
    caveats, fixed, instruction = [], spec["fixed"], spec["instruction"]
    lines = fixed.count("\n")
    if spec.get("extra_files"):
        caveats.append("它另带只读依赖文件 " + "、".join(sorted(spec["extra_files"])) + "，参考解和测试里 import 了它。"
                       "你的题只能有一个文件：被操作的库、后端这类东西写进同一个文件，或者改成由调用方传入的对象，"
                       "测试里也不要 import 别的文件")
    if not FIXED_LINES[0] <= lines <= FIXED_LINES[1]:
        caveats.append(f"它的参考解有 {lines} 行；你的要在 {FIXED_LINES[0]} 到 {FIXED_LINES[1]} 行之间")
    if len(fixed) > MAX_FIXED_CHARS:
        caveats.append(f"它的参考解有 {len(fixed)} 个字符；你的不超过 {MAX_FIXED_CHARS} 个")
    if len(instruction) > MAX_INSTRUCTION_CHARS:
        caveats.append(f"它的题面有 {len(instruction)} 个字符；你的不超过 {MAX_INSTRUCTION_CHARS} 个")
    count = len(parse.requirement_numbers(instruction))
    if not MIN_REQUIREMENTS <= count <= MAX_REQUIREMENTS:
        caveats.append(f"它的题面有 {count} 条编号的要求；你的题的条数见下面【要求条数】一栏")
    if _banned_io(fixed):
        caveats.append("它的参考解用 open()、os.replace() 这类写法读写文件；你的题不能读写文件")
    names = sorted(spec["private_tests"])
    clean = [name for name in names if not _banned_io(spec["private_tests"][name])]
    if not clean:
        caveats.append("它的隐藏测试读写文件或者 import 了依赖文件；你的测试里不能有这些写法")
    return caveats, clean or names


def _theme_section(direction: Direction, domain: Domain) -> str:
    slips = "\n".join(f"- {slip}" for slip in direction.slips)
    hint = f"：{domain.hint}" if domain.hint else ""
    return (f"【方向】{direction.label}\n"
            f"{direction.summary}\n"
            "这类题常见的疏忽（新的 mutants 要像这些一样真实，但不要逐条照搬）：\n"
            f"{slips}\n\n"
            f"【领域】{domain.name}{hint}。\n"
            "程序要围绕这个领域的数据和操作来写，名字和术语也要符合它。")


def _count_section(slug) -> str:
    count = requirement_count(slug)
    return (f"【要求条数】{count} 条：instruction 里从 1）编到 {count}）。\n\n"
            "整道题要精炼：题面、程序和公开测试合在一起有长度上限，写长了会被打回。")


def _example_section(example: dict) -> str:
    """风格参照那一段：先说它合不合硬性要求（不合的几处点名不要学），再给题面、参考解、一个隐藏测试文件。"""
    caveats = example.get("caveats") or []
    intro = ["下面是同一个方向的一道现有题，只用来看题面的写法、参考解的详细程度和测试的风格；"
             "不要照抄它的主题、命名和逻辑。"]
    if caveats:
        intro.append("这道参照题有下面几处不符合上面的硬性要求，这几处不要学：")
        intro += [f"- {caveat}。" for caveat in caveats]
    else:
        intro.append("这道题本身满足上面的全部硬性要求，题面、参考解和测试的详细程度照它来。")
    intro.append("题面照它的写法：先用一句话说明程序做什么，再逐条写清方法名、参数、返回值，以及抛出的异常类型。")
    note = f"（超过 {EXAMPLE_LINES} 行，只给前 {EXAMPLE_LINES} 行）" if example.get("truncated") else ""
    return ("【风格参照】\n" + "\n".join(intro) + "\n\n"
            f"题面：\n{example['instruction']}\n\n"
            f"参考解{note}：\n{_fence(example['fixed'], 'python')}\n"
            f"隐藏测试文件 {example['test_file']}（开头两行 import 是自动加的）：\n"
            f"{_fence(example['test_body'], 'python')}")


def _previous_section(previous_raw) -> str:
    if not isinstance(previous_raw, str) or not previous_raw.strip():
        return "【上一版】上一版回复是空的，没有可改的东西：请按上面的要求重新出一道题。"
    return f"【上一版】你上一版给的题原样如下（检查的就是它）：\n{_fence(previous_raw)}"


def _findings_section(findings) -> str:
    items = _finding_list(findings)
    if not items:
        return "【检查发现的问题】检查没有给出具体问题：请对照硬性要求逐条自查，把不满足的地方改掉。"
    listed = "\n".join(f"{index}. {item}" for index, item in enumerate(items, 1))
    return f"【检查发现的问题】共 {len(items)} 条，每一条都要修掉：\n{listed}"


def _finding_list(findings) -> list[str]:
    """问题列表：调用方给一串字符串、一个字符串、一个异常或者审计的一条发现（字典）都行。"""
    if findings is None:
        return []
    if isinstance(findings, (str, BaseException, dict)):      # 单独一条：字典不能当序列迭代，否则列出的是键名
        findings = [findings]
    return [text.strip() for text in (_finding_text(item) for item in findings) if text.strip()]


def _finding_text(item) -> str:
    """一条问题的文字。审计（audit.py）的发现是 {"code", "message", "repairable"}：取它的 message；
    别处给的字典按 ``键：值`` 列出来，不让一段 JSON 原样挡在提示里。"""
    if isinstance(item, (str, BaseException)):
        return str(item)
    if isinstance(item, dict):
        if isinstance(item.get("message"), str) and item["message"].strip():
            return item["message"]
        return "；".join(f"{key}：{value}" for key, value in item.items())
    if isinstance(item, (list, tuple)):
        return "；".join(_finding_text(part) for part in item)
    return str(item)


def _fence(text: str, language: str = "") -> str:
    """把一段内容原样放进代码围栏；内容里有反引号时把围栏加长，免得提前收尾。"""
    body = text.rstrip("\n")
    width = max([3, *(len(run) + 1 for run in re.findall(r"`+", body))])
    mark = "`" * width
    return f"{mark}{language}\n{body}\n{mark}"


__all__ = ["EXAMPLE_LINES", "FIXED_LINES", "HARD_REQUIREMENTS", "MAX_FIXED_CHARS", "MAX_INSTRUCTION_CHARS",
           "MAX_PUBLIC_TEST_LINES", "MAX_REQUIREMENTS", "MIN_REQUIREMENTS", "REQUIREMENTS", "format_notes",
           "format_skeleton", "format_spec", "generation_messages", "module_name", "pick_example", "repair_messages",
           "requirement_count"]
