# -*- coding: utf-8 -*-
"""合成题族的题目规格：教师的一次回复是一个题族（一个程序），这里把它变成母题和单缺陷变体。

教师给的是一份完全正确的主文件 fixed、题面、公开测试、按标签组织的隐藏测试，以及若干
mutant：每个 mutant 把 fixed 里的一处改坏，对应题面里的一条要求。由此构造：

- 母题 ``s3_<slug>_001``：在参考解上依次应用全部可用的 mutant，程序里同时带着多处缺陷；
- 单缺陷变体 ``s3v_<slug>_<k>``：参考解上只应用一个 mutant，题面、参考解、测试都沿用
  母题（做法同 hard_tasks_v3_steps 的 _step）。它是比母题低一档的台阶。

源码和测试都 dedent().strip() 后补一个换行，测试文件的开头同 hard_tasks_v3_common._task；
题面去掉公共缩进和首尾空白。mutant 的 old 片段在规范形式（dedent 之后）的参考解里定位。
本模块只做文本和语法树层面的构造，不执行任何代码。构造时顺手拦下几类到写文件、做补丁时才会出错的
问题，都抛 FamilyError：标签不合规或太长（它要当文件名），文字里有空字节或写不进文件的字符（孤立的
代理项），源码的行末尾有控制字符（回车、垂直制表符、换页符）。
"""
from __future__ import annotations

import ast
import hashlib
import re
from textwrap import dedent

from agentlab.benchmark.protocol import build_messages

# slug 进了 task_id、source_group 和文件名，只能用这些字符，也不能太长
SLUG_PATTERN = re.compile(r"[a-z0-9_]+")
MAX_SLUG = 100
# 隐藏测试的标签是文件名 test_<标签>.py 的一部分
LABEL_PATTERN = re.compile(r"[A-Za-z0-9_]+")
MAX_LABEL = 100
# 行末尾的这几个字符 git 不认：参考解和原程序之间的补丁会被当成坏补丁，判分走不到测试那一步
LINE_END_CONTROLS = {"\r": "回车", "\x0b": "垂直制表符", "\x0c": "换页符"}


class FamilyError(ValueError):
    """题族本身有问题（锚点不唯一、标签不合规……）；消息里写明是哪一处、该怎么改。"""


def source(text: str) -> str:
    """一份源码的规范形式：去掉公共缩进和首尾空白，以一个换行结束（同 _task）。"""
    return dedent(text).strip() + "\n"


def shorten(text: str, limit: int = 60) -> str:
    """给消息用的一行片段：压掉换行和多余空白，太长就截断。"""
    clipped = " ".join(text.split())
    return clipped if len(clipped) <= limit else clipped[:limit] + "…"


def _writable(text: str, what: str, *, patched: bool = True) -> str:
    """源码都要写进文件、再做成判分用的补丁。有几类写法要到那时才出错，构造题目时就报出来：
    空字节、孤立的代理项（JSON 里的 \\ud800 这类转义）写不进像样的文件；行末尾的回车、垂直制表符、
    换页符让 git 把补丁当成坏补丁。patched 为假表示这段文字不进补丁（题面），只查前两类。
    """
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as error:
        line = text.count("\n", 0, error.start) + 1
        raise FamilyError(f"{what}里有写不进文件的字符（孤立的代理项，第 {line} 行）：多半是 JSON 转义写错了，"
                          f"请换成正常的字符") from error
    if "\x00" in text:
        line = text.count("\n", 0, text.index("\x00")) + 1
        raise FamilyError(f"{what}里有空字节（NUL，第 {line} 行）：这多半是生成时出的错，请删掉它")
    if patched:
        for number, line in enumerate(text.split("\n"), 1):
            if line and line[-1] in LINE_END_CONTROLS:
                raise FamilyError(f"{what}的第 {number} 行末尾有控制字符（{LINE_END_CONTROLS[line[-1]]}）：判分"
                                  f"用的补丁会因此变成坏补丁，请删掉它")
    return text


def _mutant(parsed: dict, index: int) -> dict:
    mutants = parsed["mutants"]
    if not 0 <= index < len(mutants):
        raise FamilyError(f"没有第 {index + 1} 个 mutant：题族里一共 {len(mutants)} 个")
    mutant = mutants[index]
    if not isinstance(mutant, dict):
        raise FamilyError(f"第 {index + 1} 个 mutant 不是一个带 old / new 的对象")
    if not isinstance(mutant.get("old"), str) or not mutant["old"]:
        raise FamilyError(f"第 {index + 1} 个 mutant 没有给出要替换的 old 片段")
    if not isinstance(mutant.get("new"), str):
        raise FamilyError(f"第 {index + 1} 个 mutant 的 new 不是字符串")
    return mutant


def mutant_name(mutant: dict, index: int) -> str:
    """给人和教师看的称呼：优先带上它对应的要求序号。"""
    requirement = mutant.get("requirement")
    if isinstance(requirement, int) and not isinstance(requirement, bool):
        return f"第 {requirement} 条要求的 mutant（第 {index + 1} 个）"
    return f"第 {index + 1} 个 mutant"


def _apply(text: str, old: str, new: str, name: str, where: str) -> str:
    """把 text 里的 old 换成 new；old 必须恰好出现一次，否则说明是哪一处。"""
    _writable(new, f"{name}的 new 片段")
    count = text.count(old)
    if count != 1:
        raise FamilyError(f"{name}的 old 片段在{where}出现 {count} 次（必须恰好 1 次）：{shorten(old)!r}")
    return text.replace(old, new)


def _spec(parsed: dict, buggy: str, task_id: str, slug: str, direction: str) -> dict:
    module = f"s3_{slug}"
    for label in parsed["tests"]:
        if not isinstance(label, str) or not LABEL_PATTERN.fullmatch(label):
            raise FamilyError(f"隐藏测试的标签 {shorten(str(label), 40)!r} 只能用字母、数字、下划线"
                              f"（它要当文件名 test_<标签>.py）")
        if len(label) > MAX_LABEL:
            raise FamilyError(f"隐藏测试的标签 {label[:20]}… 太长（{len(label)} 个字符，最多 {MAX_LABEL} 个）："
                              f"它要当文件名 test_<标签>.py，请改短")
    return {
        "task_id": task_id, "source_group": module, "split": "train",
        "bug_family": direction, "difficulty": "L3",
        "instruction": _writable(dedent(parsed["instruction"]).strip(), "题面", patched=False),
        "module": f"{module}.py",
        "buggy": buggy, "fixed": _writable(source(parsed["fixed"]), "参考解"),
        "public_test": f"from {module} import *\n\n" + _writable(source(parsed["public_test"]), "公开测试"),
        "private_tests": {f"test_{label}.py": f"import pytest\nfrom {module} import *\n\n"
                                              + _writable(source(item["body"]), f"隐藏测试 {label}")
                          for label, item in parsed["tests"].items()},
        # 冻结划分时按方向分层（preparation.freeze_splits 读 stratum）；母题和变体永远同层
        "stratum": f"synth-{direction}",
        "origin": "synth", "direction": direction,
    }


def _variant(parsed: dict, index: int, number: int, slug: str, direction: str) -> dict:
    mutant = _mutant(parsed, index)
    name = mutant_name(mutant, index)
    buggy = _apply(source(parsed["fixed"]), mutant["old"], mutant["new"], name, "参考解里")
    return {**_spec(parsed, buggy, f"s3v_{slug}_{number}", slug, direction), "variant_of": f"s3_{slug}_001"}


def build_variant(parsed: dict, index: int, *, slug: str, direction: str) -> dict:
    """单缺陷变体：参考解上只应用第 index 个 mutant（从 0 起），编号取 index + 1。"""
    return _variant(parsed, index, index + 1, slug, direction)


def build_parent(parsed: dict, indices, *, slug: str, direction: str) -> dict:
    """母题：在参考解上依次应用 indices 里的 mutant。

    每一步的 old 都必须在当时的代码里恰好出现一次；前面的改动让后面的锚点失灵时抛
    FamilyError，说明是第几步、哪个 mutant。
    """
    text = source(parsed["fixed"])
    for step, index in enumerate(indices, 1):
        mutant = _mutant(parsed, index)
        name = f"第 {step} 步、{mutant_name(mutant, index)}"
        text = _apply(text, mutant["old"], mutant["new"], name, "前面的 mutant 应用之后的代码里")
    return _spec(parsed, text, f"s3_{slug}_001", slug, direction)


def build_family(parsed: dict, *, slug: str, direction: str, indices=None) -> dict:
    """一个题族的全部题目：``{"parent", "variants", "mutants"}``，mutants 是用到的 mutant 序号。

    indices 是要用的 mutant 序号（在 parsed["mutants"] 里从 0 起），按给出的顺序应用；
    不给就用全部。变体按这个顺序从 1 起编号，所以只传可用的 mutant 时编号是连着的。
    """
    indices = list(range(len(parsed["mutants"]))) if indices is None else list(indices)
    variants = [_variant(parsed, index, number, slug, direction) for number, index in enumerate(indices, 1)]
    parent = build_parent(parsed, indices, slug=slug, direction=direction)
    return {"parent": parent, "variants": variants, "mutants": indices}


def family_specs(family: dict) -> list[dict]:
    """题族里的全部题目，母题在前、变体按编号在后。"""
    return [family["parent"], *family["variants"]]


def _dump(tree: ast.AST) -> str:
    # Python 3.13 起 ast.dump 默认省略空字段；显式打开，新旧版本算出的哈希才一致
    try:
        return ast.dump(tree, show_empty=True)
    except TypeError:      # 3.12 及以前没有这个参数，本来就会输出全部字段
        return ast.dump(tree)


def fixed_hash(spec: dict) -> str:
    """参考解的语法树哈希：只差注释或排版的两个参考解得到同一个值。"""
    return hashlib.sha256(_dump(ast.parse(spec["fixed"])).encode("utf-8")).hexdigest()


def rough_tokens(text: str) -> int:
    """粗估 token 数（按字节级分词器）：CJK 字符约 1.05 个，其余约每 3.3 个字符 1 个。

    算法同 scripts/self_check_pool.py，只用来发现太长的题。
    """
    return round(sum(1.05 if ord(char) > 0x2E7F else 0.3 for char in text))


def prompt_estimate(spec: dict) -> int:
    """一道题的提示（系统提示加用户消息）的粗估 token 数。"""
    return sum(rough_tokens(message["content"]) for message in build_messages(spec))
