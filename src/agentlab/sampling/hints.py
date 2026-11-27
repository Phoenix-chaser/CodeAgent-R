"""引导采样用的提示：让基座模型带着"参考解的开头"或"教师写的通用技能"去采样。

采样时发出去的消息带提示；训练记录里的 ``messages`` 仍然是不带提示的标准提示，所以训练时
模型看不到提示。两种提示各有依据：

- ``prefix``：把参考解的前若干行放进用户消息，让模型接着写完（POPE, arXiv:2601.18779）。
  提示里明确要求"给出的开头原样保留"：POPE 的消融里，禁止复述会降低不带提示的通过率。
- ``skill``：一条不涉及具体题目的程序性技能（FailForge, arXiv:2608.08570）。
"""
from __future__ import annotations

import math

PREFIX_HINT = (
    "下面是参考修复后的主文件的开头部分（到第 {shown} 行为止，参考解共 {total} 行），仅供参考。\n"
    "请从这部分接着写完：给出的开头代码原样保留，输出修复后主文件的完整代码。")

SKILL_HINT = "下面是一条通用的修复技能，仅供参考，不要在输出里提到它。请输出修复后主文件的完整代码。"

KINDS = ("prefix", "skill")


def prefix_lines(fixed: str, fraction: float) -> int:
    """参考解里给模型看的行数：至少 1 行，并且至少留 1 行让它自己写。"""
    total = len(fixed.splitlines())
    return max(1, min(math.ceil(total * fraction), total - 1))


def prefix_hint(fixed: str, fraction: float) -> dict:
    lines = fixed.splitlines()
    shown = prefix_lines(fixed, fraction)
    return {
        "kind": "prefix", "fraction": fraction,
        "lines_shown": shown, "lines_total": len(lines),
        "text": "\n".join(lines[:shown]),
        "message": PREFIX_HINT.format(shown=shown, total=len(lines)) + "\n\n" + "\n".join(lines[:shown]),
    }


def skill_hint(skill: str) -> dict:
    text = skill.strip()
    if not text:
        raise ValueError("empty skill")
    return {"kind": "skill", "text": text, "message": SKILL_HINT + "\n\n" + text}


def with_hint(messages: list[dict], hint: dict) -> list[dict]:
    """带提示的消息：提示追加在最后一条用户消息的末尾，其余原样保留。"""
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("a hint needs a final user message")
    return [*messages[:-1],
            {**messages[-1], "content": messages[-1]["content"] + "\n\n" + hint["message"]}]


def restated(output: str, hint: dict) -> bool | None:
    """模型有没有照抄提示里的代码。前缀提示看开头的第一行和最后一行是否都出现在输出里
    （先后不变）；技能提示没有可对照的代码，返回 None。"""
    if hint["kind"] != "prefix":
        return None
    lines = [line.strip() for line in hint["text"].splitlines() if line.strip()]
    if not lines:
        return False
    positions = [output.find(line) for line in (lines[0], lines[-1])]
    if min(positions) < 0:
        return False
    return positions[0] <= positions[1]
