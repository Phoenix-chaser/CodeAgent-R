# -*- coding: utf-8 -*-
"""合成出题的流水线（scripts/synth_pool.py）。

不访问教师接口：假的 ask 按调用的先后给出准备好的回复。解析是真的；审计默认也是真的——在沙箱里跑 pytest，
所以用到它的几个用例比别的慢。只看编排的用例（并发、停下这一批、出错以后的去向）换成假的审计。
题族是手工构造的点名册小程序：六条要求、六个 mutant，每个 mutant 只让自己那条要求的隐藏测试失败。

怎么停（Ctrl-C、kill、启动它的脚本没了）要在真的进程里才演得出来：末尾那几个用例起一个子进程跑真的 main，
教师接口换成假的，给子进程发真的信号。

教师接口出毛病的那几种情况（服务器上实测）：成片 429 时 _shared_teacher.chat_text 长时间重试，重试用完才算
这一轮出错；回复被截断（或者正文是空的、推理很长）算一轮返修，不直接判这一族拒绝；默认值取得保守（并发 2、
请求之间隔 1 秒、输出上限 12000）。这里的用例不看节流，两次请求之间不隔（fixture no_spacing）。
"""
import json
import os
import random
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import types
from collections import Counter
from pathlib import Path
from textwrap import dedent

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import _shared_teacher  # noqa: E402
import privacy_check  # noqa: E402
import synth_pool  # noqa: E402
from agentlab.benchmark import grader  # noqa: E402
from agentlab.benchmark.protocol import load_specs, validate_specs  # noqa: E402
from agentlab.sampling import vllm_client  # noqa: E402
from agentlab.synth import audit, parse, prompts, specs, themes  # noqa: E402
from agentlab.training.preparation import freeze_splits  # noqa: E402

# 点名册，43 行。各个题族的程序只有类名不同（都一样的话，冻结划分时会被判成重复题）
TALLY = '''\
class Tally:
    """点名册：记下到场的人、迟到分钟数，并按组统计在场人数。"""

    def __init__(self):
        self._present = {}      # 名字 → 所在的组
        self._late = {}         # 名字 → 迟到的分钟数

    def arrive(self, name, group):
        """到场：记下所在的组；已经到场过的名字再到场一次抛 ValueError。"""
        if name in self._present:
            raise ValueError('already arrived')
        self._present[name] = group

    def leave(self, name):
        """离场：从在场名单里去掉；不在场的名字抛 KeyError。"""
        if name not in self._present:
            raise KeyError(name)
        del self._present[name]

    def mark_late(self, name, minutes):
        """迟到：分钟数必须是正数；多次迟到记下最大的那次。"""
        if minutes <= 0:
            raise ValueError('minutes must be positive')
        self._late[name] = max(minutes, self._late.get(name, 0))

    def late_minutes(self, name):
        """这个人的迟到分钟数；没迟到过是 0。"""
        return self._late.get(name, 0)

    def count(self, group=None):
        """在场人数；给了组就只数这个组的人。"""
        if group is None:
            return len(self._present)
        return sum(item == group for item in self._present.values())

    def groups(self):
        """到场过的组，按字母顺序。"""
        return sorted(set(self._present.values()))

    def clear(self):
        """清空全部记录。"""
        self._present.clear()
        self._late.clear()
'''

# 公开测试只碰最基本的接口：每个 mutant 单独出现时它都照过，缺陷只能被隐藏测试暴露
PUBLIC = '''\
def test_public_tally():
    tally = Tally()
    tally.arrive('ann', 'red')
    tally.arrive('bob', 'red')
    assert tally.count() == 2
    assert tally.count('red') == 2
    assert tally.groups() == ['red']
    assert tally.late_minutes('ann') == 0
'''

# (标签, 题面里的那条要求, 隐藏测试, old, new, 这处缺陷是什么疏忽)。
# 每个 mutant 单独出现时公开测试照过、自己那条要求的隐藏测试断言失败；六个一起（母题）也是公开过、隐藏挂
REQUIREMENTS = (
    ("arrive",
     "arrive(name, group)：登记到场；已经到场过的名字再到场一次抛 ValueError，原来的登记不变。",
     '''
def test_arrive_twice_is_rejected():
    tally = Tally()
    tally.arrive('ann', 'red')
    with pytest.raises(ValueError):
        tally.arrive('ann', 'blue')
    assert tally.count('red') == 1
    assert tally.count('blue') == 0
''',
     "        if name in self._present:\n            raise ValueError('already arrived')\n", "",
     "到场时忘了拦住重复登记"),
    ("leave",
     "leave(name)：离场之后这个名字不再计入在场人数；不在场的名字抛 KeyError。",
     '''
def test_leave_removes_the_name():
    tally = Tally()
    tally.arrive('ann', 'red')
    tally.arrive('bob', 'red')
    tally.leave('bob')
    assert tally.count() == 1
    assert tally.count('red') == 1
    with pytest.raises(KeyError):
        tally.leave('bob')
''',
     "        del self._present[name]\n", "        pass\n", "离场时忘了把名字从在场名单里去掉"),
    ("late_positive",
     "mark_late(name, minutes)：minutes 不是正数时抛 ValueError，不记下任何东西。",
     '''
def test_late_minutes_must_be_positive():
    tally = Tally()
    with pytest.raises(ValueError):
        tally.mark_late('ann', 0)
    with pytest.raises(ValueError):
        tally.mark_late('ann', -5)
    assert tally.late_minutes('ann') == 0
''',
     "        if minutes <= 0:\n", "        if minutes < 0:\n", "边界写反了：0 也被当成合法的分钟数"),
    ("late_max",
     "mark_late(name, minutes)：多次迟到记下最大的那次。",
     '''
def test_late_minutes_keep_the_largest():
    tally = Tally()
    tally.mark_late('ann', 9)
    tally.mark_late('ann', 2)
    assert tally.late_minutes('ann') == 9
    tally.mark_late('ann', 5)
    assert tally.late_minutes('ann') == 9
''',
     "        self._late[name] = max(minutes, self._late.get(name, 0))\n", "        self._late[name] = minutes\n",
     "后一次迟到把前一次的记录盖掉了"),
    ("group_count",
     "count(group)：只数这个组的在场人数；不给组就数全部。",
     '''
def test_count_only_counts_the_given_group():
    tally = Tally()
    tally.arrive('ann', 'red')
    tally.arrive('bob', 'blue')
    tally.arrive('cat', 'red')
    assert tally.count('red') == 2
    assert tally.count('blue') == 1
    assert tally.count() == 3
''',
     "        return sum(item == group for item in self._present.values())\n", "        return len(self._present)\n",
     "按组统计时忘了过滤，数成了全部"),
    ("clear",
     "clear()：清空之后在场人数和迟到分钟数都回到 0。",
     '''
def test_clear_forgets_everything():
    tally = Tally()
    tally.arrive('ann', 'red')
    tally.mark_late('ann', 3)
    tally.clear()
    assert tally.count() == 0
    assert tally.late_minutes('ann') == 0
''',
     "        self._present.clear()\n        self._late.clear()\n", "        self._present.clear()\n",
     "清空时漏掉了迟到记录"),
)

HEAD = "点名册：登记到场、迟到分钟数，按组统计在场人数"
# 写到一半被截断的回复
TRUNCATED = '```json\n{"theme": "点名册", "instruction": "修复 Tally：要求：\\n1）到场'
# 判分环境没问题时参考解的测试没过，审计给的 code
REFERENCE_FAILS = "参考解不通过自己的测试"


# ── 手工构造的小题 ──────────────────────────────────────────────────────

def slugs(count, batch=1):
    """第 batch 批的前 count 个主题的 slug（themes.plan_themes 给的先后）。"""
    return [themes.slug_of(*item) for item in themes.plan_themes(batch, count)]


def theme(slug):
    """slug 对应的主题 (方向, 领域)。"""
    direction, code = slug.split("_", 1)
    return themes.DIRECTIONS[direction], themes.resolve_domain(code)


def class_name(slug):
    """题族自己的类名：方向和领域的短名拼成驼峰，每个题族都不一样（同一个领域会在几个方向下各出一道题）。"""
    direction, domain = theme(slug)
    return "Tally" + direction.name.capitalize() + "".join(part.capitalize() for part in domain.code.split("_"))


def count_of(slug):
    """这个 slug 要求几条行为要求（由 themes.requirement_count 决定，4 到 6）。"""
    return themes.requirement_count(slug)


def family(slug):
    """一份合格的题族，已经是规范形式：前 requirement_count(slug) 条要求。"""
    name, items = class_name(slug), REQUIREMENTS[:count_of(slug)]
    listed = "\n".join(f"{number}）{text}" for number, (_, text, _, _, _, _) in enumerate(items, 1))
    return {
        "theme": HEAD,
        "instruction": f"修复 {name}：点名册登记到场的人、迟到分钟数，并按组统计在场人数，现在的实现有几处疏忽。\n"
                       f"要求：\n{listed}",
        "fixed": TALLY.replace("Tally", name),
        "public_test": PUBLIC.replace("Tally", name),
        "tests": {label: {"requirement": number, "body": dedent(body.replace("Tally", name)).strip() + "\n"}
                  for number, (label, _, body, _, _, _) in enumerate(items, 1)},
        "mutants": [{"requirement": number, "old": old, "new": new, "idea": idea}
                    for number, (_, _, _, old, new, idea) in enumerate(items, 1)],
    }


def reply(item) -> str:
    """教师的一次回复：题族包在 json 围栏里。"""
    return "```json\n" + json.dumps(item, ensure_ascii=False) + "\n```"


def broken(slug):
    """第一轮的回复：程序没问题，但第 4 条要求的隐藏测试把期望值写错了，参考解过不了它。"""
    item = family(slug)
    item["tests"]["late_max"]["body"] = item["tests"]["late_max"]["body"].replace("== 9", "== 8")
    return item


def failed_tests(requirement):
    """第 requirement 条要求的隐藏测试里第一个测试函数的 pytest 编号。"""
    label, _, body, _, _, _ = REQUIREMENTS[requirement - 1]
    first = re.search(r"def (test_\w+)", body).group(1)
    return [f"tests_private.test_{label}.py::{first}"]


def parsed_of(item, slug):
    return parse.parse_family(reply(item), slug)


def accepted(slug, batch=1, calls=1):
    """一份已接受的记录：题族由手工构造的小题拼出来，不经过审计。calls 是它请求了几轮才被接受。"""
    item, direction = family(slug), theme(slug)[0].name
    parsed = parsed_of(item, slug)
    built = specs.build_family(parsed, slug=slug, direction=direction)
    rounds = [{"round": number, "kind": "generation" if number == 1 else "repair",
               "raw": reply(item) if number == calls else "上一版的回复", "finish_reason": "stop",
               "usage": {"input_tokens": 100 * number, "output_tokens": 10 * number}, "parse_error": None,
               "findings": [] if number == calls else [
                   {"code": "reference_fails", "message": "参考解没有通过全部测试", "repairable": True}],
               "stats": None, "error": None} for number in range(1, calls + 1)]
    parent = built["parent"]
    stats = {
        "fixed_lines": len(parsed["fixed"].splitlines()), "fixed_hash": specs.fixed_hash(parent),
        "prompt_tokens": specs.prompt_estimate(parent), "answer_tokens": specs.rough_tokens(parsed["fixed"]),
        "isolation": "process", "checks": dict.fromkeys(audit.CHECKS, True),
        "reference": [{"verdict": "strict_pass", "strict_pass": True}],
        "parent": [{"verdict": "private_test_failure", "strict_pass": False}],
        "variants": [{"index": index, "requirement": parsed["mutants"][index]["requirement"], "usable": True,
                      "caught": True, "why": "", "failed_tests": failed_tests(parsed["mutants"][index]["requirement"])}
                     for index in built["mutants"]],
    }
    stats["total_tokens"] = stats["prompt_tokens"] + stats["answer_tokens"]
    rounds[-1]["stats"] = stats
    return {"slug": slug, "direction": direction, "domain": theme(slug)[1].name, "batch": batch, "status": "accepted",
            "reason": None, "rounds": rounds, "family": built,
            "mutants": [parsed["mutants"][index] for index in built["mutants"]], "fixed_hash": stats["fixed_hash"],
            "created_at": "2026-11-01T10:00:00+0800"}


def rejected(slug, reason, calls=1):
    """一份被拒绝的记录。"""
    record = accepted(slug, calls=calls)
    record.update(status="rejected", reason=reason, family=None, mutants=[], fixed_hash=None)
    for entry in record["rounds"]:
        entry.update(findings=[], stats=None)
    return record


def write_record(pool, record):
    synth_pool.families_dir(pool).mkdir(parents=True, exist_ok=True)
    synth_pool.write_record(pool, record)


def old_task(name, split):
    """阶梯里原有的一道题（和合成题无关的小题）。"""
    return {
        "task_id": name, "source_group": f"group-{name}", "split": split, "bug_family": "clamping", "difficulty": "L1",
        "instruction": "实现 clamp(values, low, high)：把每个数限制在 [low, high] 内，返回新列表。",
        "module": f"{name}.py",
        "buggy": "def clamp(values, low, high):\n    return list(values)\n",
        "fixed": "def clamp(values, low, high):\n    return [min(max(v, low), high) for v in values]\n",
        "public_test": f"from {name} import clamp\n\ndef test_public():\n    assert clamp([1, 5], 0, 10) == [1, 5]\n",
        "private_tests": {"test_hidden.py": f"from {name} import clamp\n\n"
                                             "def test_hidden():\n    assert clamp([-3, 3, 13], 0, 10) "
                                             "== [0, 3, 10]\n"},
    }


# ── 假的教师接口、假的审计 ───────────────────────────────────────────────


@pytest.fixture(autouse=True)
def no_spacing(monkeypatch):
    """这些用例看的不是节流：走真的 chat_text 的请求之间不隔（默认隔 1 秒），进程里共用的那份节流器也不限在飞
    的个数，用完恢复成原来的设置。专测节流的参数怎么传下去的用例自己设。"""
    monkeypatch.setenv("TEACHER_MIN_INTERVAL", "0")
    monkeypatch.setattr(_shared_teacher.THROTTLE, "max_in_flight", 64)
    monkeypatch.setattr(_shared_teacher.THROTTLE, "min_interval", 0.0)
    # 上一次请求开始的时刻也清掉、用完恢复：把 time 换成假时钟的用例里它会被记成"假"的时刻，留着的话下一个
    # 用例（真时钟）要等到那个时刻才发得出请求
    monkeypatch.setattr(_shared_teacher.THROTTLE, "_last_start", None)

class Teacher:
    """假的教师接口：按调用的先后给出准备好的结果，字符串就是回复，异常就在那一次调用抛出来。

    给了 {"slug": [结果…]} 就按题族分发：哪个 slug 的模块名出现在消息里，就取它那一份的下一个。
    每次调用的消息和参数都记在 calls 里。
    """

    def __init__(self, *outcomes, by_slug=None):
        self.outcomes, self.by_slug, self.calls = list(outcomes), by_slug, []

    def __call__(self, messages, max_tokens, temperature):
        self.calls.append((messages, max_tokens, temperature))
        if self.by_slug is None:
            assert self.outcomes, "教师被调用的次数比准备好的回复多"
            outcome = self.outcomes.pop(0)
        else:
            text = messages[1]["content"]
            slug = next((name for name in self.by_slug if f"s3_{name}." in text), None)
            assert slug is not None, "发出去的消息里认不出是哪个题族"
            assert self.by_slug[slug], f"{slug} 的回复已经用完了"
            outcome = self.by_slug[slug].pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def accepting(parsed, **options):
    """假的审计：接受，题族照 specs 构造。"""
    built = specs.build_family(parsed, slug=options["slug"], direction=options["direction"])
    return {"accepted": True, "family": built, "findings": [], "usable": list(built["mutants"]),
            "stats": {"fixed_hash": specs.fixed_hash(built["parent"])}}


def refusing(code, repairable=True):
    """假的审计：因为一条发现拒绝。"""

    def refuse(parsed, **options):
        return {"accepted": False, "family": None, "usable": [], "stats": {},
                "findings": [{"code": code, "message": f"{code} 的说明", "repairable": repairable}]}

    return refuse


def run(pool, teacher, wanted, **options):
    """对 wanted 这些 slug 的主题跑一批（第 1 批）。解析是真的；审计不给就是真的（沙箱里判分，每层只判一遍）。"""
    lines = []
    options.setdefault("concurrency", 1)
    options.setdefault("repeats", 1)
    summary = synth_pool.run_batch([theme(slug) for slug in wanted], teacher, pool, batch=options.pop("batch", 1),
                                   out=lines.append, warn=lines.append, examples={},
                                   work_dir=pool.parent / f"work-{pool.name}", **options)
    return summary, lines, synth_pool.read_records(pool)


# ── 手工构造的小题本身是合格的 ──────────────────────────────────────────

def test_the_hand_made_family_fits_the_format_and_the_size_limits():
    for slug in slugs(6):
        item = family(slug)
        assert len(item["mutants"]) == len(item["tests"]) == len(parse.requirement_numbers(item["instruction"]))
        assert 4 <= count_of(slug) <= 6 and 30 <= len(item["fixed"].splitlines()) <= 70
        assert parse.parse_family(reply(item), slug) == item                 # 已经是规范形式：解析不改任何东西
        built = specs.build_family(parsed_of(item, slug), slug=slug, direction=theme(slug)[0].name)
        # 母题的原程序带着全部缺陷：和参考解只差各处 mutant 的那几行
        delta = sum(len(new.splitlines()) - len(old.splitlines())
                    for _, _, _, old, new, _ in REQUIREMENTS[:count_of(slug)])
        assert len(built["parent"]["buggy"].splitlines()) == len(item["fixed"].splitlines()) + delta
        assert specs.prompt_estimate(built["parent"]) + specs.rough_tokens(item["fixed"]) <= 1900


# ── run：走真实的 run_batch ─────────────────────────────────────────────

def test_a_sound_family_is_accepted_in_the_first_round(tmp_path):
    slug, wanted = slugs(1)[0], count_of(slugs(1)[0])
    teacher = Teacher(reply(family(slug)))
    summary, lines, records = run(tmp_path / "pool", teacher, [slug])
    assert (summary["accepted"], summary["rejected"], summary["calls"], summary["skipped"]) == (1, 0, 1, 0)
    assert not summary["stopped"] and summary["unsettled"] == 0 and summary["reasons"] == {}
    record, = records
    assert record["status"] == "accepted" and record["reason"] is None and record["batch"] == 1
    assert (record["direction"], record["domain"]) == (theme(slug)[0].name, theme(slug)[1].name)
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d{4}", record["created_at"])
    built = specs.build_family(parsed_of(family(slug), slug), slug=slug, direction=theme(slug)[0].name)
    assert record["family"] == built
    assert [spec["task_id"] for spec in synth_pool.family_specs(record["family"])] == [
        f"s3_{slug}_001", *(f"s3v_{slug}_{number}" for number in range(1, wanted + 1))]
    validate_specs(synth_pool.family_specs(record["family"]))
    assert record["fixed_hash"] == specs.fixed_hash(record["family"]["parent"])
    assert [mutant["idea"] for mutant in record["mutants"]] == [item[5] for item in REQUIREMENTS[:wanted]]
    # 这一轮：首轮出题、原始回复原样留着、没有解析报错、审计八项全过、没有发现
    entry, = record["rounds"]
    assert entry["kind"] == "generation" and entry["raw"] == reply(family(slug))
    assert entry["parse_error"] is None and entry["error"] is None and entry["findings"] == []
    assert entry["stats"]["checks"] == dict.fromkeys(audit.CHECKS, True)
    assert entry["stats"]["usable"] == wanted and 30 <= entry["stats"]["fixed_lines"] <= 70
    assert all(row["usable"] and row["failed_tests"] for row in entry["stats"]["variants"])
    assert entry["usage"] is None and entry["finish_reason"] is None     # 假的 ask 给的是普通字符串
    # 发出去的就是首轮的出题提示
    assert teacher.calls == [(prompts.generation_messages(theme(slug), None, slug), synth_pool.MAX_TOKENS, 0.8)]
    # 教师先写一大段推理再给题族：给推理留足余量
    assert synth_pool.MAX_TOKENS == 12000
    # 每一族结束打一行进度，结束打结果和下一批
    assert lines == [
        f"[########################] 1/1 (100%) {slug}: 接受（第 1 轮，变体 {wanted} 个） · 已用 0m",
        f"结果：第 1 批 接受 1 族（母题 1 道、变体 {wanted} 道）、拒绝 0 族（原因：无）、教师调用 1 次、用时 0 分钟",
        "下一批这样跑：BATCH=2 bash scripts/run_synth_pool.sh"]
    # 记录落盘了，读回来一模一样
    assert json.loads(synth_pool.family_path(tmp_path / "pool", slug).read_text(encoding="utf-8")) == record


def test_the_findings_go_back_to_the_teacher_and_the_repaired_family_is_accepted(tmp_path):
    slug = slugs(2 + 1)[2]
    teacher = Teacher(reply(broken(slug)), reply(family(slug)))
    summary, lines, records = run(tmp_path / "pool", teacher, [slug])
    record, = records
    assert summary["accepted"] == 1 and summary["calls"] == 2 and record["status"] == "accepted"
    first, second = record["rounds"]
    # 第一轮：参考解过不了自己的隐藏测试，这是可以返修的发现
    assert first["kind"] == "generation" and first["raw"] == reply(broken(slug))
    assert [finding["code"] for finding in first["findings"]] == ["reference_fails"]
    assert first["findings"][0]["repairable"] is True
    assert "参考解没有通过全部测试" in first["findings"][0]["message"]
    assert first["stats"]["checks"]["reference"] is False
    # 第二轮是返修：带着上一版的原样和审计的全部发现
    assert second["kind"] == "repair" and second["raw"] == reply(family(slug))
    assert second["findings"] == [] and second["stats"]["checks"]["reference"] is True
    (messages, max_tokens, temperature), = teacher.calls[1:]
    assert messages == prompts.repair_messages(reply(broken(slug)), first["findings"], theme(slug), slug)
    system, user = [message["content"] for message in messages]
    assert first["findings"][0]["message"] in user and reply(broken(slug)).rstrip("\n") in user
    assert "请修改下面这道没有通过检查的题" in user and "只输出修改后的完整 JSON 对象" in system
    assert (max_tokens, temperature) == (12000, 0.8)
    assert len(lines) == 3                                    # 中间那一轮不打结果行，一族只打一行进度
    assert lines == [
        f"[########################] 1/1 (100%) {slug}: 接受（第 2 轮，变体 {count_of(slug)} 个） · 已用 0m",
        f"结果：第 1 批 接受 1 族（母题 1 道、变体 {count_of(slug)} 道）、拒绝 0 族（原因：无）、"
        "教师调用 2 次、用时 0 分钟",
        "下一批这样跑：BATCH=2 bash scripts/run_synth_pool.sh"]


def test_a_family_is_rejected_when_the_rounds_run_out(tmp_path):
    slug = slugs(1 + 1)[1]
    teacher = Teacher(reply(broken(slug)), reply(broken(slug)), reply(family(slug)))
    summary, lines, records = run(tmp_path / "pool", teacher, [slug], rounds=2)
    record, = records
    assert (summary["accepted"], summary["rejected"], summary["calls"]) == (0, 1, 2)
    assert summary["reasons"] == {REFERENCE_FAILS: 1} and summary["unsettled"] == 0
    assert record["status"] == "rejected" and record["reason"] == REFERENCE_FAILS
    assert record["family"] is None and record["fixed_hash"] is None and record["mutants"] == []
    assert [entry["kind"] for entry in record["rounds"]] == ["generation", "repair"]
    assert all(entry["findings"][0]["code"] == "reference_fails" for entry in record["rounds"])
    assert len(teacher.calls) == 2                                   # 轮次用完，第三份回复没有被用上
    assert lines == [
        f"[########################] 1/1 (100%) {slug}: 拒绝（{REFERENCE_FAILS}） · 已用 0m",
        f"结果：第 1 批 接受 0 族（母题 0 道、变体 0 道）、拒绝 1 族（原因：{REFERENCE_FAILS} 1）、"
        "教师调用 2 次、用时 0 分钟",
        "下一批这样跑：BATCH=2 bash scripts/run_synth_pool.sh"]
    assert synth_pool.settled(record) is True                        # 因为题族本身被拒绝：定论了


def test_a_reply_that_does_not_parse_costs_a_round_and_the_error_goes_back_to_the_teacher(tmp_path):
    slug = slugs(3 + 1)[3]
    teacher = Teacher(TRUNCATED, reply(family(slug)), reply(family(slug)))
    summary, lines, records = run(tmp_path / "pool", teacher, [slug], rounds=2)
    record, = records
    first, second = record["rounds"]
    assert summary["accepted"] == 1 and summary["calls"] == 2 and len(teacher.calls) == 2
    assert first["kind"] == "generation" and first["raw"] == TRUNCATED and first["findings"] is None
    assert "不完整或者格式有错" in first["parse_error"] and "截断" in first["parse_error"]
    assert second["kind"] == "repair" and second["findings"] == [] and second["raw"] == reply(family(slug))
    assert teacher.calls[1][0] == prompts.repair_messages(TRUNCATED, first["parse_error"], theme(slug), slug)
    assert first["parse_error"] in teacher.calls[1][0][1]["content"]
    assert lines == [
        f"[########################] 1/1 (100%) {slug}: 接受（第 2 轮，变体 {count_of(slug)} 个） · 已用 0m",
        f"结果：第 1 批 接受 1 族（母题 1 道、变体 {count_of(slug)} 道）、拒绝 0 族（原因：无）、"
        "教师调用 2 次、用时 0 分钟",
        "下一批这样跑：BATCH=2 bash scripts/run_synth_pool.sh"]



def replied(text, finish="stop", reasoning=None):
    """一条带结束原因和用量的回复（同 make_ask 给的 Reply）。reasoning 是推理字段有几个字符。"""
    return synth_pool.Reply(text, {"input_tokens": 11, "output_tokens": 7}, finish,
                            {"reasoning_content": reasoning} if reasoning else None)


def test_a_reply_cut_off_at_the_limit_costs_a_repair_round_that_says_it_was_cut_off(tmp_path):
    """回复在写完之前到了输出上限（finish_reason 是 length）：不成 JSON 不是"格式不对"，所以不去解析它，记下的
    也不是"找不到 JSON 对象"。它算一轮——还有轮次的话下一轮是返修，带着"上次被截断"请教师重新输出，不直接
    判这一族拒绝。推理字段只记它有多长。"""
    slug = slugs(1)[0]
    audited = []
    teacher = Teacher(replied(TRUNCATED, finish="length", reasoning=3000), reply(family(slug)), reply(family(slug)))
    summary, lines, records = run(
        tmp_path / "pool", teacher, [slug], rounds=2,
        auditor=lambda parsed, **options: audited.append(parsed) or accepting(parsed, **options))
    record, = records
    first, second = record["rounds"]
    assert summary["accepted"] == 1 and summary["calls"] == 2 and len(teacher.calls) == 2
    assert first["kind"] == "generation" and first["raw"] == TRUNCATED and first["truncated"] is True
    assert first["findings"] is None and first["error"] is None and len(audited) == 1   # 没写完的回复不拿去审计
    assert (first["finish_reason"], first["usage"], first["other_text"]) == (
        "length", {"input_tokens": 11, "output_tokens": 7}, {"reasoning_content": 3000})
    assert first["parse_error"] == (
        f"上次被截断：回复在写完之前到了输出上限（finish_reason=length，正文 {len(TRUNCATED)} 个字符，"
        "推理等其它字段：reasoning_content 3000 个字符），JSON 没有写完。请重新输出完整的 JSON 对象，写紧凑一些，"
        "不要别的内容")
    assert second["kind"] == "repair" and second["findings"] == [] and second["truncated"] is False
    assert teacher.calls[1][0] == prompts.repair_messages(TRUNCATED, first["parse_error"], theme(slug), slug)
    system, user = [message["content"] for message in teacher.calls[1][0]]
    assert "上次被截断" in user and "到了输出上限" in user and "只输出修改后的完整 JSON 对象" in system
    assert "想" not in record["rounds"][0]["other_text"]              # 只记长度：dict 里只有数字


def test_an_empty_reply_is_treated_as_cut_off_too_and_the_next_round_says_so(tmp_path):
    """正文是空的（思考类模型把输出全花在推理上）：同"被截断"。以前空的回复下一轮闷头重新出题、不说为什么。"""
    slug = slugs(2)[0]
    teacher = Teacher(replied("", finish="length", reasoning=4000), reply(family(slug)), reply(family(slug)))
    summary, _, records = run(tmp_path / "pool", teacher, [slug], rounds=2, auditor=accepting)
    first, second = records[0]["rounds"]
    assert summary["accepted"] == 1 and first["truncated"] is True and second["kind"] == "repair"
    assert first["other_text"] == {"reasoning_content": 4000}
    assert first["parse_error"] == (
        "上次被截断：回复在写完之前到了输出上限（finish_reason=length，正文 0 个字符，"
        "推理等其它字段：reasoning_content 4000 个字符），JSON 没有写完。请重新输出完整的 JSON 对象，写紧凑一些，"
        "不要别的内容")
    teacher = Teacher(replied("  \n", reasoning=900), reply(family(slug)), reply(family(slug)))   # 结束原因是 stop
    summary, _, records = run(tmp_path / "other", teacher, [slug], rounds=2, auditor=accepting)
    first, second = records[0]["rounds"]
    assert summary["accepted"] == 1 and first["truncated"] is True and second["kind"] == "repair"
    assert first["parse_error"] == (
        "上次被截断：上一次的回复是空的（finish_reason=stop，正文 0 个字符，"
        "推理等其它字段：reasoning_content 900 个字符），没有可用的内容。请输出完整的 JSON 对象")
    assert teacher.calls[1][0] == prompts.repair_messages("  \n", first["parse_error"], theme(slug), slug)
    user = teacher.calls[1][0][1]["content"]
    assert "上一版回复是空的" in user and "上次被截断" in user
    # 轮次用完还是空的：拒绝的原因是"回复被截断"，不是"解析失败"；没有定论，重跑会再要一次
    teacher = Teacher(replied("", reasoning=100), replied("", reasoning=100))
    summary, lines, records = run(tmp_path / "third", teacher, [slug], rounds=2)
    assert records[0]["reason"] == "回复被截断" and not synth_pool.settled(records[0])
    assert summary["reasons"] == {"回复被截断": 1}
    assert lines[0] == f"[########################] 1/1 (100%) {slug}: 拒绝（回复被截断） · 已用 0m"
    # 教师调用出错（没有回复）照旧：下一轮重新出题
    teacher = Teacher(ConnectionError("连不上"), reply(family(slug)), reply(family(slug)))
    summary, _, records = run(tmp_path / "fourth", teacher, [slug], rounds=2, auditor=accepting)
    assert [entry["kind"] for entry in records[0]["rounds"]] == ["generation", "generation"]


def test_a_reply_that_is_not_json_costs_a_repair_round_that_says_so(tmp_path):
    """正文有、可是不是合格的题族 JSON：带着"上次不是合法的 JSON"和解析器的说明返修一轮，不直接拒绝。"""
    slug = slugs(3)[0]
    teacher = Teacher("这不是 JSON", reply(family(slug)), reply(family(slug)))
    summary, _, records = run(tmp_path / "pool", teacher, [slug], rounds=2, auditor=accepting)
    record, = records
    first, second = record["rounds"]
    assert record["status"] == "accepted" and first["truncated"] is False and second["kind"] == "repair"
    assert first["parse_error"].startswith("上次不是合法的 JSON：回复里找不到题族的 JSON 对象")
    assert teacher.calls[1][0] == prompts.repair_messages("这不是 JSON", first["parse_error"], theme(slug), slug)
    assert "上次不是合法的 JSON" in teacher.calls[1][0][1]["content"]


def test_a_parse_error_in_every_round_rejects_the_family_without_auditing(tmp_path):
    slug = slugs(5)[4]
    teacher = Teacher("这不是 JSON", TRUNCATED)
    audited = []
    summary, lines, records = run(tmp_path / "pool", teacher, [slug], rounds=2,
                                  auditor=lambda parsed, **options: audited.append(parsed))
    record, = records
    assert audited == [] and (summary["accepted"], summary["rejected"]) == (0, 1)
    assert record["status"] == "rejected" and record["reason"] == "解析失败"
    assert [entry["parse_error"] is not None for entry in record["rounds"]] == [True, True]
    assert "找不到题族的 JSON 对象" in record["rounds"][0]["parse_error"]
    assert lines[0] == f"[########################] 1/1 (100%) {slug}: 拒绝（解析失败） · 已用 0m"
    # 回复没法用说的不是题族的事：这一族没有定论，不算用掉，重跑时会再要一次
    assert synth_pool.settled(record) is False and synth_pool.used_slugs(tmp_path / "pool") == set()
    assert lines[2] == ("其中 1 族没有定论，不算用掉。只补它们、不要新主题："
                        "FAMILIES=1 BATCH=1 bash scripts/run_synth_pool.sh")


def test_a_finding_that_cannot_be_repaired_ends_the_family_without_another_request(tmp_path):
    slug = slugs(5 + 1)[5]
    teacher = Teacher(reply(family(slug)), reply(family(slug)))
    summary, lines, records = run(tmp_path / "pool", teacher, [slug], rounds=3, auditor=refusing("collision", False))
    record, = records
    assert len(teacher.calls) == 1 and record["status"] == "rejected" and record["reason"] == "和现有题池撞了"
    assert len(record["rounds"]) == 1 and record["rounds"][0]["kind"] == "generation"
    assert lines[0] == f"[########################] 1/1 (100%) {slug}: 拒绝（和现有题池撞了） · 已用 0m"
    assert synth_pool.settled(record) is True



def test_a_teacher_error_is_a_round_and_leaves_the_family_without_a_verdict(tmp_path):
    pool, slug = tmp_path / "pool", slugs(7)[6]
    teacher = Teacher(ConnectionError("连不上教师接口"), ConnectionError("连不上教师接口"))
    summary, lines, records = run(pool, teacher, [slug], rounds=2)
    record, = records
    assert (summary["accepted"], summary["rejected"], summary["calls"]) == (0, 1, 2)
    assert (summary["errors"], summary["unsettled"]) == (2, 1)
    assert record["status"] == "rejected" and record["reason"] == "教师调用出错"
    assert [entry["kind"] for entry in record["rounds"]] == ["generation", "generation"]    # 没有可改的上一版：重新出题
    assert all(entry["error"] == "ConnectionError: 连不上教师接口" and entry["raw"] is None
               for entry in record["rounds"])
    assert lines == [
        f"[########################] 1/1 (100%) {slug}: 拒绝（教师调用出错） · 已用 0m",
        "结果：第 1 批 接受 0 族（母题 0 道、变体 0 道）、拒绝 1 族（原因：教师调用出错 1）、教师调用 2 次、"
        "用时 0 分钟",
        "其中 1 族没有定论，不算用掉。只补它们、不要新主题：FAMILIES=1 BATCH=1 bash scripts/run_synth_pool.sh",
        "下一批这样跑：BATCH=2 bash scripts/run_synth_pool.sh"]
    # 没有定论：不算"用过"，重跑同一批时会先再要它一次
    assert synth_pool.settled(record) is False and synth_pool.used_slugs(pool) == set()
    summary, lines, records = run(pool, Teacher(reply(family(slug))), [slug], auditor=accepting)
    assert summary["accepted"] == 1 and synth_pool.used_slugs(pool) == {slug}
    assert lines == [
        f"[########################] 1/1 (100%) {slug}: 接受（第 1 轮，变体 {count_of(slug)} 个） · 已用 0m",
        f"结果：第 1 批 接受 1 族（母题 1 道、变体 {count_of(slug)} 道）、拒绝 0 族（原因：无）、教师调用 1 次、"
        "用时 0 分钟",
        "下一批这样跑：BATCH=2 bash scripts/run_synth_pool.sh"]



def test_a_crash_in_the_pipeline_is_recorded_as_a_rejected_family_and_the_batch_goes_on(tmp_path, monkeypatch):
    real = parse.parse_family

    def explode(text, slug):
        real(text, slug)                                 # 解析不了的照常报错；解析得出一道完整的题时才出错
        raise RuntimeError("流水线自己出了错")

    monkeypatch.setattr(parse, "parse_family", explode)
    wanted = slugs(2)
    teacher = Teacher(reply(family(wanted[0])), "不是 JSON", TRUNCATED)
    summary, lines, records = run(tmp_path / "pool", teacher, wanted, rounds=2)
    assert [record["slug"] for record in records] == sorted(wanted)
    crashed, other = (records[0], records[1]) if records[0]["slug"] == wanted[0] else (records[1], records[0])
    assert crashed["status"] == "rejected" and crashed["reason"] == "处理时程序出错"
    assert crashed["rounds"] == [] and crashed["error"] == "RuntimeError: 流水线自己出了错"
    assert other["reason"] == "解析失败"
    assert summary["reasons"] == {"处理时程序出错": 1, "解析失败": 1} and summary["unsettled"] == 2
    assert f"处理 {wanted[0]} 时流水线出了错（已记成拒绝，重跑会再要一次）：RuntimeError: 流水线自己出了错" in lines
    assert synth_pool.used_slugs(tmp_path / "pool") == set()     # 都没有定论：重跑会再要一次



def test_families_with_a_verdict_are_not_requested_again_when_the_batch_is_rerun(tmp_path):
    pool, wanted = tmp_path / "pool", slugs(6)
    write_record(pool, accepted(wanted[0]))
    write_record(pool, rejected(wanted[1], REFERENCE_FAILS))           # 因为题族本身被拒绝：定论了
    write_record(pool, rejected(wanted[2], "教师调用出错"))            # 没有定论：重跑时再要一次
    # 排主题时跳过有定论的 slug：没定论的那个排在最前，后面接着序列里没用过的，仍然凑满要的个数
    every = synth_pool.parse_directions("all")
    plan = synth_pool.planned_themes(1, 4, every, seed=42, used_slugs=synth_pool.used_slugs(pool))
    assert [themes.slug_of(*item) for item in plan] == [wanted[2], *wanted[3:6]]
    before = [synth_pool.family_path(pool, slug).read_bytes() for slug in wanted[:2]]
    teacher = Teacher(*[TRUNCATED, "不是 JSON"] * 4)
    summary, _, records = run(pool, teacher, [themes.slug_of(*item) for item in plan], rounds=2)
    assert len(teacher.calls) == 8 and (summary["accepted"], summary["rejected"]) == (0, 4)
    assert {record["slug"] for record in records} == set(wanted)      # 原来那两份记录原封没动
    assert [synth_pool.family_path(pool, slug).read_bytes() for slug in wanted[:2]] == before
    assert [record["reason"] for record in records if record["slug"] == wanted[2]] == ["解析失败"]
    # 再跑一遍：后做的四族都是回复没法用，还是没定论，照样排在最前；有定论的两族不再请求
    again = synth_pool.planned_themes(1, 4, every, seed=42, used_slugs=synth_pool.used_slugs(pool))
    assert again == plan and synth_pool.used_slugs(pool) == set(wanted[:2])


def test_a_record_that_cannot_be_read_back_is_skipped_and_its_family_will_be_requested_again(tmp_path, capsys):
    pool, slug = tmp_path / "pool", slugs(1)[0]
    write_record(pool, accepted(slug))
    synth_pool.family_path(pool, slug).write_text("{写到一半", encoding="utf-8")
    assert synth_pool.read_records(pool) == [] and synth_pool.used_slugs(pool) == set()
    assert f"{slug}.json 读不出来，跳过" in capsys.readouterr().err


def test_a_later_family_is_audited_against_the_ones_already_accepted(tmp_path):
    wanted, seen = slugs(2), {}

    def spy(parsed, **options):
        seen[options["slug"]] = options
        return accepting(parsed, **options)

    run(tmp_path / "first", Teacher(reply(family(wanted[0])), reply(family(wanted[1]))), wanted, auditor=spy)
    earlier = accepted(wanted[0])["family"]
    assert seen[wanted[0]]["extra_specs"] == [] and seen[wanted[0]]["known_fixed_hashes"] == []
    assert seen[wanted[1]]["extra_specs"] == synth_pool.family_specs(earlier)
    assert seen[wanted[1]]["known_fixed_hashes"] == [specs.fixed_hash(earlier["parent"])]
    # 目录里原先就接受的题族也在比对的范围里
    seen.clear()
    write_record(tmp_path / "second", accepted(wanted[0]))
    run(tmp_path / "second", Teacher(reply(family(wanted[1]))), [wanted[1]], auditor=spy)
    assert seen[wanted[1]]["extra_specs"] == synth_pool.family_specs(earlier)
    assert seen[wanted[1]]["known_fixed_hashes"] == [specs.fixed_hash(earlier["parent"])]


def test_audits_never_overlap_even_when_four_families_are_in_flight(tmp_path):
    wanted, state, guard = slugs(4), {"inside": 0, "peak": 0}, threading.Lock()
    original = grader.PYTEST_TIMEOUT

    def slow_audit(parsed, **options):
        # 学真审计的样子：临时改判分器的全局超时。两路同时进来就会互相盖掉，退出去以后全局的值也不对
        with guard:
            state["inside"] += 1
            state["peak"] = max(state["peak"], state["inside"])
        before = grader.PYTEST_TIMEOUT
        grader.PYTEST_TIMEOUT = 5
        try:
            time.sleep(0.05)
            return accepting(parsed, **options)
        finally:
            grader.PYTEST_TIMEOUT = before
            with guard:
                state["inside"] -= 1

    teacher = Teacher(by_slug={slug: [reply(family(slug))] for slug in wanted})
    summary, lines, records = run(tmp_path / "pool", teacher, wanted, concurrency=4, auditor=slow_audit)
    assert state["peak"] == 1 and grader.PYTEST_TIMEOUT == original
    assert (summary["accepted"], summary["calls"], len(records)) == (4, 4, 4)
    progress = [line for line in lines if line.startswith("[")]
    assert len(lines) == 6                                      # 4 行进度、结果、下一批
    # 进度按 1、2、3、4 的先后打印，进度条一格 1/24；完成的先后不定，所以只核对这一批里谁被接受了
    assert [line.split("] ")[0] + "]" for line in progress] == [
        "[" + "#" * (6 * number) + "-" * (24 - 6 * number) + "]" for number in range(1, 5)]
    assert [re.search(r"\] (\d)/4 \((\d+)%\)", line).groups() for line in progress] == [
        ("1", "25"), ("2", "50"), ("3", "75"), ("4", "100")]
    assert sorted(re.search(r" (\S+): 接受（第 1 轮，变体 (\d+) 个） · 已用 0m$", line).groups() for line in progress) \
        == sorted((slug, str(count_of(slug))) for slug in wanted)


def test_the_batch_stops_when_grading_keeps_failing(tmp_path):
    wanted = slugs(6)
    teacher = Teacher(*[reply(family(slug)) for slug in wanted])

    def slow_refusal(parsed, **options):
        time.sleep(0.2)                # 审计要花一点时间，停下来的消息才来得及传到后面排着的题族
        return refusing("grading_error", False)(parsed, **options)

    summary, lines, records = run(tmp_path / "pool", teacher, wanted, auditor=slow_refusal)
    # 连续 3 个都因为判分没能完成被拒：停下。停下的时候工作线程里已经在做第 4 族，它照常做完；
    # 后面 2 个主题不再请求（留给重跑）
    assert summary["stopped"] and len(records) == 4 and summary["skipped"] == 2
    assert (summary["rejected"], summary["unsettled"], summary["calls"]) == (4, 4, 4)
    assert len(teacher.calls) == 4 and summary["reasons"] == {"判分没能完成": 4}
    assert {record["slug"] for record in records} == set(wanted[:4])
    assert all(record["reason"] == "判分没能完成" and not synth_pool.settled(record) for record in records)
    assert "连续 3 个题族都因为判分没能完成或者程序出错被拒，这一批停下了" in lines[-3]
    assert "判分环境多半有问题" in lines[-3]
    assert lines[-2] == ("修好以后原样重跑这一批（没做完的排在最前，再要满这一批的 6 个）："
                         "BATCH=1 bash scripts/run_synth_pool.sh")
    assert lines[-1] == "下一批这样跑：BATCH=2 bash scripts/run_synth_pool.sh"
    assert synth_pool.used_slugs(tmp_path / "pool") == set()         # 都没有定论：重跑会先补上它们


def test_the_batch_goes_on_when_only_some_families_cannot_be_graded(tmp_path):
    wanted = slugs(6)
    outcomes = {slug: [reply(family(slug))] for slug in wanted}
    results = {wanted[1]: refusing("grading_error", False), wanted[4]: refusing("collision", False)}

    def mixed(parsed, **options):
        return results.get(options["slug"], accepting)(parsed, **options)

    summary, lines, records = run(tmp_path / "pool", Teacher(by_slug=outcomes), wanted, auditor=mixed)
    assert not summary["stopped"] and summary["skipped"] == 0 and len(records) == 6
    assert summary["accepted"] == 4 and summary["reasons"] == {"判分没能完成": 1, "和现有题池撞了": 1}
    assert "这一批停下了" not in "\n".join(lines)



def test_the_summary_line_counts_what_happened_in_the_batch():
    summary = {"batch": 3, "accepted": 2, "variants": 9, "rejected": 3, "calls": 7, "seconds": 125,
               "reasons": {"解析失败": 2, REFERENCE_FAILS: 1}}
    assert synth_pool.result_line(summary) == (
        f"结果：第 3 批 接受 2 族（母题 2 道、变体 9 道）、拒绝 3 族（原因：解析失败 2、{REFERENCE_FAILS} 1）、"
        "教师调用 7 次、用时 2 分钟")
    base = {**summary, "planned": 6, "unsettled": 2, "stopped": False, "unwritten": []}
    # 跑完了、有没定论的：只补它们的办法写明白（FAMILIES 写成它们的个数），不说"先重跑这一批"
    assert synth_pool.next_batch_hints(base) == [
        "其中 2 族没有定论，不算用掉。只补它们、不要新主题：FAMILIES=2 BATCH=3 bash scripts/run_synth_pool.sh",
        "下一批这样跑：BATCH=4 bash scripts/run_synth_pool.sh"]
    assert synth_pool.next_batch_hints({**base, "unsettled": 0}) == [
        "下一批这样跑：BATCH=4 bash scripts/run_synth_pool.sh"]
    # 没跑完（判分环境坏了、写不进盘）：原样重跑会先补没做完的，再要满这一批
    assert synth_pool.next_batch_hints({**base, "stopped": True}) == [
        "修好以后原样重跑这一批（没做完的排在最前，再要满这一批的 6 个）：BATCH=3 bash scripts/run_synth_pool.sh",
        "下一批这样跑：BATCH=4 bash scripts/run_synth_pool.sh"]
    assert synth_pool.next_batch_hints({**base, "unwritten": ["s3_a（磁盘满）"]})[0].startswith(
        "修好以后原样重跑这一批")
    # 进度条的样子同 scripts/run_rft.sh 的 progress()：进度面板靠 "] k/n (p%)" 认它
    assert synth_pool.progress_line(6, 24, "等待教师", time.monotonic() - 60) == (
        "[######------------------] 6/24 (25%) 等待教师 · 已用 1m")
    assert synth_pool.progress_line(24, 24, "完成", time.monotonic() - 125) == (
        "[########################] 24/24 (100%) 完成 · 已用 2m")


# ── 主题 ────────────────────────────────────────────────────────────────

def test_a_batch_takes_fresh_themes_and_skips_the_slugs_already_used():
    every = synth_pool.parse_directions("all")
    plan = synth_pool.planned_themes(2, 5, every, seed=42, used_slugs=slugs(7))
    # 第 2 批（每批 5 个）从序列的第 5 个起：头两个已经用过，跳过；取位置 5 到 9 里剩下的 3 个，
    # 再往后补 2 个，凑满 5 个（每批 1 个的第 11、12 批就是位置 10、11）
    assert [themes.slug_of(*item) for item in plan] == [
        *slugs(5, batch=2)[2:], *(themes.slug_of(*themes.plan_themes(number, 1)[0]) for number in (11, 12))]
    assert synth_pool.planned_themes(2, 5, every, seed=42) == themes.plan_themes(2, 5, seed=42)


def test_only_the_chosen_directions_are_requested_but_still_a_full_batch():
    lock = synth_pool.parse_directions("lock")

    def locks(names):
        return [slug for slug in names if slug.startswith("lock_")]

    # 第 1 批的窗口里有 10 个 lock 方向的主题：往后接着找，凑满 12 个；路过的别的方向不算用过
    every = locks(slugs(545))                      # 序列里全部 lock 方向的主题，按先后
    assert every[:10] == locks(slugs(50))
    picked = synth_pool.planned_themes(1, 12, lock, seed=42)
    assert [themes.slug_of(*item) for item in picked] == every[:12]
    assert all(item[0].name == "lock" for item in picked)
    # 用过的跳过，仍然凑满
    again = synth_pool.planned_themes(1, 12, lock, seed=42, used_slugs=every[:12])
    assert [themes.slug_of(*item) for item in again] == every[12:24]
    # 几个方向一起要：头 10 个里 lock 和 cond 各 5 个
    both = synth_pool.parse_directions("lock,cond")
    names = [item[0].name for item in synth_pool.planned_themes(1, 10, both, seed=42)]
    assert Counter(names) == {"lock": 5, "cond": 5}
    # 序列走到头还凑不够就说清楚：第 11 批（每批 50 个）的窗口里只剩 9 个 lock 方向的主题
    with pytest.raises(ValueError, match="第 11 批凑不够 50 个 lock 方向的主题：序列里只剩 9 个"):
        synth_pool.planned_themes(11, 50, lock, seed=42)
    early = {slug for batch in range(1, 11) for slug in slugs(50, batch)}      # 前 500 个位置
    tail = [slug for slug in locks(slugs(545)) if slug not in early]
    assert len(tail) == 9
    with pytest.raises(ValueError, match="序列里只剩 6 个"):
        synth_pool.planned_themes(11, 50, lock, seed=42, used_slugs=tail[:3])
    with pytest.raises(ValueError, match="序列里只剩 0 个"):                  # 全用过了：一开始就走到头
        synth_pool.planned_themes(11, 50, lock, seed=42, used_slugs=tail)




def test_directions_must_be_all_or_known_short_names():
    assert synth_pool.parse_directions("all") == frozenset(themes.DIRECTION_NAMES)
    assert synth_pool.parse_directions(" lock,cond ,") == {"lock", "cond"}
    for value, piece in (("", "要写 all"), ("lock,bogus", "未知的方向 bogus"), ("状态一致性", "未知的方向 状态一致性")):
        with pytest.raises(ValueError, match=piece):
            synth_pool.parse_directions(value)


# ── 命令行：run ─────────────────────────────────────────────────────────

class FakeClient:
    """假的教师客户端，读法同 chat_text（client.chat.completions.create）。"""

    def __init__(self, *outcomes):
        self.outcomes, self.requests = list(outcomes), []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    delay = 0.0

    def create(self, model, messages, max_tokens, temperature):
        self.requests.append((model, max_tokens, temperature, messages))
        assert self.outcomes, "教师被调用的次数比准备好的回复多"
        time.sleep(self.delay)                        # 演示"正在进行的请求"要花时间
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def response(text, finish="stop"):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=text), finish_reason=finish)],
        usage=types.SimpleNamespace(prompt_tokens=11, completion_tokens=7))


@pytest.fixture
def reject_on_content(monkeypatch):
    """审计换成假的：每个题族都因为题族本身的问题被拒（不可返修，一轮就完）。不用判分，也不花第二轮。"""
    monkeypatch.setattr(audit, "audit_family", refusing("collision", False))


@pytest.fixture
def teacher_api(monkeypatch):
    """把教师接口换成假的客户端。用它给一串回复（或者异常），返回 (客户端, make_client 收到的参数)。"""
    made = []

    def install(*outcomes):
        client = FakeClient(*outcomes)
        monkeypatch.setenv("TEACHER_API_KEY", "test-key-not-a-real-one")
        monkeypatch.setenv("TEACHER_MODEL", "teacher-test")
        monkeypatch.setattr(_shared_teacher, "make_client", lambda *args: made.append(args) or client)
        return client, made

    return install


def asked_slugs(client, candidates):
    """客户端收到的请求里问到过哪些题族（按 candidates 的先后）。"""
    return [slug for slug in candidates
            if any(f"s3_{slug}." in messages[1]["content"] for _, _, _, messages in client.requests)]



def test_main_run_asks_the_teacher_through_the_shared_client_and_keeps_the_usage(
        tmp_path, teacher_api, capsys, reject_on_content):
    pool, wanted = tmp_path / "pool", slugs(3)
    client, made = teacher_api(*(response(reply(family(slug))) for slug in wanted))
    code = synth_pool.main(["run", "--output-dir", str(pool), "--families", "3", "--rounds", "2", "--concurrency", "1",
                            "--max-tokens", "1234", "--temperature", "0.5", "--timeout", "77", "--model", "teacher-x"])
    shown = capsys.readouterr()
    assert code == 0 and shown.err == ""
    assert made == [("TEACHER_API_KEY", None, 77)]                  # 密钥的环境变量、地址、超时原样交给 make_client
    assert {(model, tokens, heat) for model, tokens, heat, _ in client.requests} == {("teacher-x", 1234, 0.5)}
    assert len(client.requests) == 3 and asked_slugs(client, wanted) == wanted
    # 发出去的提示带着这个题族的风格参照（从同方向的现有难题里挑的）
    slug = wanted[0]
    assert client.requests[0][3] == prompts.generation_messages(
        theme(slug), prompts.pick_example(theme(slug)[0], slug), slug)
    record = next(item for item in synth_pool.read_records(pool) if item["slug"] == slug)
    assert record["status"] == "rejected" and record["reason"] == "和现有题池撞了"
    # 真的 ask 带回了用量和结束原因，记在每一轮里
    assert [(entry["usage"], entry["finish_reason"], entry["raw"]) for entry in record["rounds"]] == [
        ({"input_tokens": 11, "output_tokens": 7}, "stop", reply(family(slug)))]
    assert shown.out.splitlines() == [
        "第 1 批：向教师要 3 个题族（全部方向）；每个最多 2 轮，同时 1 个、请求至少隔 0 秒；"
        "目录里已有定论的题族 0 个",
        *(f"[{'#' * (24 * number // 3)}{'-' * (24 - 24 * number // 3)}] {number}/3 ({number * 100 // 3}%) "
          f"{name}: 拒绝（和现有题池撞了） · 已用 0m" for number, name in enumerate(wanted, 1)),
        "结果：第 1 批 接受 0 族（母题 0 道、变体 0 道）、拒绝 3 族（原因：和现有题池撞了 3）、教师调用 3 次、"
        "用时 0 分钟",
        "下一批这样跑：BATCH=2 bash scripts/run_synth_pool.sh"]
    assert (pool / ".run.lock").exists()                           # 同一个目录同一时间只跑一个 run 的那把锁



def test_main_run_exits_3_when_nothing_is_accepted_and_every_call_fails(tmp_path, teacher_api, capsys):
    pool = tmp_path / "pool"
    client, _ = teacher_api(*[ValueError("密钥不对")] * 6)
    code = synth_pool.main(["run", "--output-dir", str(pool), "--families", "3", "--concurrency", "1"])
    shown = capsys.readouterr()
    assert code == 3 and len(client.requests) == 6
    assert ("教师调用全部出错：检查 TEACHER_API_KEY、TEACHER_BASE_URL 和网络。这一批的主题都还没有定论：修好以后"
            "原样重跑，就是把它们再要一次，不会要新主题。") in shown.err
    assert "结果：第 1 批 接受 0 族（母题 0 道、变体 0 道）、拒绝 3 族（原因：教师调用出错 3）" in shown.out
    assert "其中 3 族没有定论，不算用掉。只补它们、不要新主题：FAMILIES=3 BATCH=1 bash scripts/run_synth_pool.sh" \
        in shown.out
    assert all(not synth_pool.settled(record) for record in synth_pool.read_records(pool))


def test_main_run_exits_3_when_every_reply_is_unusable_and_says_what_to_do(tmp_path, teacher_api, capsys):
    pool, wanted = tmp_path / "pool", slugs(2)
    client, _ = teacher_api(*(response(TRUNCATED, finish="length") for _ in wanted))
    code = synth_pool.main(["run", "--output-dir", str(pool), "--families", "2", "--rounds", "1", "--concurrency", "1",
                            "--max-tokens", "99"])
    shown = capsys.readouterr()
    assert code == 3 and len(client.requests) == 2                   # 一族都没接受：这一批没有跑完
    assert "回复被截断 2 族）。回复被截断就把输出上限调大：环境变量 TEACHER_MAX_TOKENS（现在 99）" in shown.err
    assert "就是把它们再要一次，不会要新主题" in shown.err
    records = synth_pool.read_records(pool)
    assert [record["reason"] for record in records] == ["回复被截断"] * 2
    assert [record["rounds"][0]["finish_reason"] for record in records] == ["length", "length"]
    assert synth_pool.used_slugs(pool) == set()                      # 没有定论：修好以后原样重跑就是再要它们
    # 没被截断、可是没法解析的回复也一样：不算这一批跑完了
    client, _ = teacher_api(response("这不是 JSON"), response("这也不是"))
    code = synth_pool.main(["run", "--output-dir", str(pool), "--families", "2", "--rounds", "1", "--concurrency", "1"])
    shown = capsys.readouterr()
    assert code == 3 and "教师有回复，但没有一族能用（解析失败 2 族）。" in shown.err
    assert "换一个 TEACHER_MODEL" in shown.err and synth_pool.used_slugs(pool) == set()


def test_main_run_stops_before_calling_the_teacher_without_a_key(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("TEACHER_API_KEY", raising=False)
    for extra in ([], ["--api-key-env", "OTHER_KEY"]):
        code = synth_pool.main(["run", "--output-dir", str(tmp_path / "pool"), "--families", "1", *extra])
        shown = capsys.readouterr()
        assert code == 1 and "结果：" not in shown.out
        assert f"环境变量 {extra[-1] if extra else 'TEACHER_API_KEY'} 没有设置" in shown.err
    assert synth_pool.families_dir(tmp_path / "pool").exists() and synth_pool.read_records(tmp_path / "pool") == []


def test_main_run_refuses_to_run_twice_on_the_same_directory(tmp_path, teacher_api, capsys, reject_on_content):
    pool = tmp_path / "pool"
    pool.mkdir()
    client, _ = teacher_api(response(reply(family(slugs(1)[0]))))
    arguments = ["run", "--output-dir", str(pool), "--families", "1", "--concurrency", "1"]
    with synth_pool.one_at_a_time(pool):
        code = synth_pool.main(arguments)
    assert code == 1 and client.requests == [] and "另一个 synth_pool.py run 正在" in capsys.readouterr().err
    assert synth_pool.main(arguments) == 0 and len(client.requests) == 1     # 上一把锁放开了就能跑



def test_main_run_leaves_families_with_a_verdict_alone_and_retries_the_rest(
        tmp_path, teacher_api, capsys, reject_on_content):
    pool, wanted = tmp_path / "pool", slugs(6)
    write_record(pool, accepted(wanted[0]))
    write_record(pool, rejected(wanted[1], REFERENCE_FAILS))
    write_record(pool, rejected(wanted[2], "教师调用出错"))
    before = synth_pool.family_path(pool, wanted[0]).read_bytes()
    client, _ = teacher_api(*(response(reply(family(slug))) for slug in wanted[2:5]))
    code = synth_pool.main(["run", "--output-dir", str(pool), "--families", "3", "--concurrency", "1"])
    shown = capsys.readouterr()
    assert code == 0 and len(client.requests) == 3
    # 有定论的两族不再请求；没定论的那一族排在最前重试，后面接着没有要过的两个。开头把这件事说清楚：
    # 原样重跑会把这一批补满，只补那一族就把 FAMILIES 写成 1
    assert asked_slugs(client, wanted) == wanted[2:5]
    assert shown.out.splitlines()[0] == (
        "第 1 批：向教师要 3 个题族（全部方向）；其中 1 个是以前要过、还没有定论的，2 个是没有要过的"
        "（只补这 1 个、不要新主题：FAMILIES=1 BATCH=1 bash scripts/run_synth_pool.sh）；每个最多 2 轮，"
        "同时 1 个、请求至少隔 0 秒；目录里已有定论的题族 2 个")
    assert synth_pool.family_path(pool, wanted[0]).read_bytes() == before
    records = {record["slug"]: record for record in synth_pool.read_records(pool)}
    assert records[wanted[0]]["status"] == "accepted" and records[wanted[1]]["reason"] == REFERENCE_FAILS
    assert [records[slug]["reason"] for slug in wanted[2:5]] == ["和现有题池撞了"] * 3
    assert synth_pool.used_slugs(pool) == set(wanted[:5])


def test_the_header_says_when_a_rerun_only_redoes_what_was_asked_before(
        tmp_path, teacher_api, capsys, reject_on_content):
    pool, wanted = tmp_path / "pool", slugs(4)
    for slug in wanted:
        write_record(pool, rejected(slug, "停下时还没做完"))          # 上一次中途被打断：这一批一个都没做完
    client, _ = teacher_api(*(response(reply(family(slug))) for slug in wanted))
    assert synth_pool.main(["run", "--output-dir", str(pool), "--families", "4", "--concurrency", "1"]) == 0
    line = capsys.readouterr().out.splitlines()[0]
    assert "向教师要 4 个题族（全部方向）；都是以前要过、还没有定论的（补做）；" in line
    assert asked_slugs(client, wanted) == wanted


def test_the_header_does_not_offer_a_families_hint_when_it_would_not_pick_the_right_themes(
        tmp_path, teacher_api, capsys, reject_on_content):
    """以前要过的那一族前面还有没要过的：FAMILIES 写几都取不到单取它，这时只说清来路，不给补做的写法。"""
    pool, wanted = tmp_path / "pool", slugs(4)
    write_record(pool, rejected(wanted[1], "教师调用出错"))          # 排在第二个位置上
    client, _ = teacher_api(*(response(reply(family(slug))) for slug in wanted))
    assert synth_pool.main(["run", "--output-dir", str(pool), "--families", "4", "--concurrency", "1"]) == 0
    line = capsys.readouterr().out.splitlines()[0]
    assert line.startswith("第 1 批：向教师要 4 个题族（全部方向）；其中 1 个是以前要过、还没有定论的，"
                           "3 个是没有要过的；")
    assert "FAMILIES=" not in line and asked_slugs(client, wanted) == wanted


def test_environment_variables_set_the_teacher_defaults(tmp_path, teacher_api, capsys, reject_on_content, monkeypatch):
    monkeypatch.setenv("TEACHER_MAX_TOKENS", "1234")
    monkeypatch.setenv("TEACHER_TEMPERATURE", "0.5")
    monkeypatch.setenv("TEACHER_TIMEOUT", "77")
    monkeypatch.setenv("TEACHER_MIN_INTERVAL", "0.25")
    parsed = synth_pool.build_parser().parse_args(["run", "--output-dir", "d"])
    assert (parsed.max_tokens, parsed.temperature, parsed.timeout, parsed.min_interval) == (1234, 0.5, 77.0, 0.25)
    slug = slugs(1)[0]
    client, made = teacher_api(response(reply(family(slug))))
    code = synth_pool.main(["run", "--output-dir", str(tmp_path / "pool"), "--families", "1", "--concurrency", "1"])
    assert code == 0 and made == [("TEACHER_API_KEY", None, 77.0)]
    assert [(model, tokens, heat) for model, tokens, heat, _ in client.requests] == [("teacher-test", 1234, 0.5)]



class Limited(Exception):
    """限流或者服务端错误：同真的 SDK，状态码在 status_code 上，Retry-After 头在 response.headers 上。"""

    def __init__(self, status, retry_after=None):
        super().__init__(f"HTTP {status}")
        self.status_code = status
        if retry_after is not None:
            self.response = types.SimpleNamespace(headers={"Retry-After": retry_after})


class Clock:
    """假的时间：sleep 把时钟往前拨，monotonic 读它。替换掉 _shared_teacher 里的 time，重试不用真等。"""

    def __init__(self, start=1000.0):
        self.now, self.sleeps = start, []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def test_a_rate_limited_teacher_is_retried_for_a_long_time_and_does_not_cost_the_family_a_round(
        tmp_path, teacher_api, capsys, monkeypatch):
    """服务器上实测：并发 4 时成片 429。以前 4 次就放弃，这一轮记成"教师调用出错"；现在长时间重试（第一次等 15 秒，
    之后翻倍，Retry-After 优先），等到了就照常往下走：这一族一轮就被接受。重试用完才算这一轮出错。"""
    slug = slugs(1)[0]
    client, _ = teacher_api(*[Limited(429, "3")] * 3, response(reply(family(slug))))
    clock = Clock()
    monkeypatch.setattr(_shared_teacher, "time", clock)
    code = synth_pool.main(["run", "--output-dir", str(tmp_path / "pool"), "--families", "1", "--concurrency", "1"])
    shown = capsys.readouterr()
    record, = synth_pool.read_records(tmp_path / "pool")
    assert code == 0 and record["status"] == "accepted" and len(record["rounds"]) == 1
    assert len(client.requests) == 4 and clock.sleeps == [3.0, 3.0, 3.0]
    assert shown.err.splitlines() == ["教师接口限速（429），等 3 秒后重试（第 1 次）",
                                      "教师接口限速（429），等 3 秒后重试（第 2 次）",
                                      "教师接口限速（429），等 3 秒后重试（第 3 次）"]
    # 一直限速：等到总时长（30 分钟）用完才放弃，这一轮才记成教师调用出错；这一族没有定论，重跑会再要一次
    client, _ = teacher_api(*[Limited(429)] * 40)
    code = synth_pool.main(["run", "--output-dir", str(tmp_path / "again"), "--families", "1", "--rounds", "1",
                            "--concurrency", "1"])
    shown = capsys.readouterr()
    record, = synth_pool.read_records(tmp_path / "again")
    # 15 + 30 + 60 + 120×14 = 1785 秒，最后一次只等剩下的 15 秒，正好 30 分钟；那一次请求还是 429 才放弃
    assert code == 3 and clock.sleeps[3:] == [15.0, 30.0, 60.0] + [120.0] * 14 + [15.0]
    assert len(client.requests) == 19 and clock.now == 1000.0 + 9.0 + 1800.0
    assert record["reason"] == "教师调用出错" and record["rounds"][0]["error"] == "Limited: HTTP 429"
    assert record["rounds"][0]["raw"] is None and not synth_pool.settled(record)
    told = shown.err.splitlines()                                  # 18 行等待、一行放弃、一行这一批的结论
    assert len(told) == 20 and told[0].endswith("等 15 秒后重试（第 1 次）")
    assert told[16].endswith("等 120 秒后重试（第 17 次）") and told[17].endswith("等 15 秒后重试（第 18 次）")
    assert told[18] == "教师接口限速（429）：从第一次失败起已经等了 30 分钟，重试了 19 次，不再重试"
    assert "教师调用全部出错：检查 TEACHER_API_KEY、TEACHER_BASE_URL 和网络" in told[19]


def test_run_sets_the_shared_throttle_from_its_arguments(tmp_path, teacher_api, capsys, reject_on_content, monkeypatch):
    """限额按账户算：在飞的请求数和间隔按 --concurrency、--min-interval 来定，之后这一批所有的请求都走进程里的
    同一份节流。默认值保守：并发 2、请求至少隔 1 秒。"""
    monkeypatch.delenv("TEACHER_MIN_INTERVAL", raising=False)
    arguments = ["run", "--output-dir", str(tmp_path / "pool"), "--families", "1"]
    client, _ = teacher_api(response(reply(family(slugs(1)[0]))))
    assert synth_pool.main([*arguments, "--concurrency", "3", "--min-interval", "0.5"]) == 0
    assert (_shared_teacher.THROTTLE.max_in_flight, _shared_teacher.THROTTLE.min_interval) == (3, 0.5)
    assert "每个最多 2 轮，同时 3 个、请求至少隔 0.5 秒；目录里已有定论的题族 0 个" in capsys.readouterr().out
    client, _ = teacher_api(response(reply(family(slugs(1)[0]))))
    assert synth_pool.main([*arguments, "--output-dir", str(tmp_path / "defaults")]) == 0     # 什么都不给
    assert (_shared_teacher.THROTTLE.max_in_flight, _shared_teacher.THROTTLE.min_interval) == (2, 1.0)
    assert [(model, tokens) for model, tokens, _, _ in client.requests] == [("teacher-test", 12000)]
    assert synth_pool.MIN_INTERVAL == 1.0 and synth_pool.CONCURRENCY == 2


def test_an_extra_json_that_is_not_an_object_stops_the_run_and_a_good_one_reaches_the_teacher(
        tmp_path, teacher_api, capsys, monkeypatch):
    pool, slug = tmp_path / "pool", slugs(1)[0]
    client, _ = teacher_api(response(reply(family(slug))))
    monkeypatch.setenv("TEACHER_EXTRA_JSON", "[1, 2]")
    with pytest.raises(SystemExit) as caught:
        synth_pool.main(["run", "--output-dir", str(pool), "--families", "1", "--concurrency", "1"])
    shown = capsys.readouterr().err
    assert caught.value.code == 2 and "环境变量 TEACHER_EXTRA_JSON 要是一个 JSON 对象" in shown
    assert "现在解析出来是list" in shown and client.requests == [] and not pool.exists()
    # 是 JSON 对象：原样合并进每一次请求
    seen = []

    def create(model, messages, max_tokens, temperature, extra_body=None):     # 签名同真的 SDK：只认这几个参数
        seen.append({"model": model, "messages": messages, "max_tokens": max_tokens,
                     "temperature": temperature, **(extra_body or {})})
        return response(reply(family(slug)))

    client.chat.completions.create = create
    monkeypatch.setenv("TEACHER_EXTRA_JSON", '{"sample_flag": true}')
    assert synth_pool.main(["run", "--output-dir", str(pool), "--families", "1", "--concurrency", "1"]) == 0
    assert (seen[0]["sample_flag"], seen[0]["max_tokens"], seen[0]["model"]) == (True, 12000, "teacher-test")


def test_run_rejects_bad_arguments_and_a_batch_that_cannot_be_planned(tmp_path, capsys, monkeypatch):
    pool = tmp_path / "pool"
    for extra in (["--families", "0"], ["--directions", "bogus"], ["--temperature", "0"], ["--rounds", "x"],
                  ["--batch", "-1"], ["--concurrency", "0"], ["--min-interval", "-1"]):
        with pytest.raises(SystemExit) as caught:
            synth_pool.main(["run", "--output-dir", str(pool), *extra])
        assert caught.value.code == 2
    assert not pool.exists()
    capsys.readouterr()
    # 环境变量里的教师参数写得不对：同样是用法错误，话里说清是哪一个
    for name, value, piece in (("TEACHER_MAX_TOKENS", "warm", "TEACHER_MAX_TOKENS 要是整数，现在是 'warm'"),
                               ("TEACHER_TEMPERATURE", "hot", "TEACHER_TEMPERATURE 要是数字，现在是 'hot'"),
                               ("TEACHER_MAX_TOKENS", "0", "都必须是正数"),
                               ("TEACHER_MIN_INTERVAL", "soon", "TEACHER_MIN_INTERVAL 要是非负数，现在是 'soon'"),
                               ("TEACHER_MIN_INTERVAL", "-1", "不能是负数")):
        monkeypatch.setenv(name, value)
        with pytest.raises(SystemExit) as caught:
            synth_pool.build_parser().parse_args(["run", "--output-dir", "d"])
        assert caught.value.code == 2 and piece in capsys.readouterr().err
        monkeypatch.delenv(name)
    assert synth_pool.main(["run", "--output-dir", str(pool), "--batch", "11", "--families", "50"]) == 1
    assert "排不出第 11 批的主题：第 11 批凑不够 50 个主题" in capsys.readouterr().err



def test_the_command_line_defaults_are_what_the_one_step_script_relies_on(monkeypatch):
    for name in ("TEACHER_MAX_TOKENS", "TEACHER_TEMPERATURE", "TEACHER_TIMEOUT", "TEACHER_MIN_INTERVAL"):
        monkeypatch.delenv(name, raising=False)
    parser = synth_pool.build_parser()
    run = parser.parse_args(["run", "--output-dir", "data/synth-pool"])
    assert (run.batch, run.families, run.directions, run.rounds, run.concurrency) == (1, 50, "all", 2, 2)
    assert (run.max_tokens, run.temperature, run.seed, run.timeout) == (12000, 0.8, 42, 300)
    assert run.min_interval == 1.0                                     # 默认值取得保守：要慢，不要成片出错
    assert (run.model, run.base_url, run.api_key_env) == (None, None, None)
    show = parser.parse_args(["show", "--output-dir", "data/synth-pool"])
    assert (show.sample, show.seed) == (3, 42)
    export = parser.parse_args(["export", "--output-dir", "data/synth-pool", "--tasks-out", "t.jsonl"])
    assert (export.ladder, export.ladder_out, export.dev_ratio, export.test_ratio, export.seed) == (
        None, None, 0.2, 0.2, 42)


# ── 命令行：status、show ────────────────────────────────────────────────

def test_status_counts_families_per_direction_and_lists_why_they_were_rejected(tmp_path, capsys):
    pool, wanted = tmp_path / "pool", slugs(10)
    for slug in wanted[:3]:
        write_record(pool, accepted(slug))
    write_record(pool, rejected(wanted[3], REFERENCE_FAILS))
    write_record(pool, rejected(wanted[4], "解析失败"))
    write_record(pool, rejected(wanted[5], "解析失败"))
    write_record(pool, rejected(wanted[6], "教师调用出错"))
    assert synth_pool.main(["status", "--output-dir", str(pool)]) == 0
    out = capsys.readouterr().out
    total = Counter(theme(slug)[0].name for slug in wanted[:7])
    good = Counter(theme(slug)[0].name for slug in wanted[:3])
    per_direction = "、".join(f"{themes.DIRECTIONS[name].label} {good[name]}/{total[name]}"
                             for name in themes.DIRECTION_NAMES if name in total)
    # 各有 1 族的两个原因：记录靠前的（slug 的顺序）排在前面
    singles = sorted([(wanted[3], REFERENCE_FAILS), (wanted[6], "教师调用出错")])
    variants = sum(count_of(slug) for slug in wanted[:3])
    assert out.splitlines() == [
        f"结果：共接受 3 族（母题 3、变体 {variants}）、拒绝 4 族；各方向 {per_direction}",
        f"拒绝原因：解析失败 2、{singles[0][1]} 1、{singles[1][1]} 1",
        f"其中 3 族没有定论（教师调用出错、回复被截断或没法解析、没能判分），不是题族的问题、也不算用掉："
        f"重跑它们所在的那一批时排在最前（{'、'.join(sorted(wanted[4:7]))}）"]


def test_status_and_show_say_so_when_there_is_nothing(tmp_path, capsys):
    pool = tmp_path / "pool"
    assert synth_pool.main(["status", "--output-dir", str(pool)]) == 0
    assert synth_pool.main(["show", "--output-dir", str(pool)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"结果：还没有题族记录（{pool / 'families'} 里没有文件）：先跑一批，bash scripts/run_synth_pool.sh",
        f"结果：{pool / 'families'} 里还没有已接受的题族，没有可看的"]
    write_record(pool, rejected(slugs(1)[0], "解析失败"))
    for arguments in ([], ["--sample", "0"]):
        assert synth_pool.main(["show", "--output-dir", str(pool), *arguments]) == 0
        assert capsys.readouterr().out.splitlines() == [f"结果：{pool / 'families'} 里还没有已接受的题族，没有可看的"]


def block(number, record):
    """show 打出来的一个题族那一段：题面、隐藏测试文件名、母题和一个变体各自的 diff。"""
    family_built, parent = record["family"], record["family"]["parent"]
    variant = family_built["variants"][0]
    return "\n".join([
        f"── {number}/2 {record['slug']}（{themes.DIRECTIONS[record['direction']].label} · {record['domain']}；"
        f"第 {len(record['rounds'])} 轮接受，母题 1 道、变体 {len(family_built['variants'])} 道）──",
        "题面：", parent["instruction"],
        "隐藏测试文件：" + "、".join(sorted(parent["private_tests"])),
        f"母题 {parent['task_id']} 的原程序 → 参考解：", synth_pool.unified_diff(parent).rstrip("\n"),
        f"变体 {variant['task_id']} 的原程序 → 参考解（这处缺陷：{record['mutants'][0]['idea']}）：",
        synth_pool.unified_diff(variant).rstrip("\n")])


def test_show_prints_the_instruction_the_test_files_and_the_diffs(tmp_path, capsys):
    pool, wanted = tmp_path / "pool", slugs(6)
    for slug in wanted[:4]:
        write_record(pool, accepted(slug, calls=2 if slug == wanted[1] else 1))
    arguments = ["show", "--output-dir", str(pool), "--sample", "2", "--seed", "7"]
    assert synth_pool.main(arguments) == 0
    out = capsys.readouterr().out
    picked = sorted(random.Random(7).sample(synth_pool.accepted_records(pool), 2), key=lambda record: record["slug"])
    assert len(picked) == 2 and out == "\n".join([
        block(1, picked[0]), block(2, picked[1]),
        "结果：抽查 2 个已接受的题族（一共 4 个；换个样本：--seed 另给一个数）"]) + "\n"
    # 换个种子抽到的不一样；要得比有的多就全看
    assert synth_pool.main([*arguments[:-1], "8"]) == 0
    other = capsys.readouterr().out
    assert other != out and other.endswith("结果：抽查 2 个已接受的题族（一共 4 个；换个样本：--seed 另给一个数）\n")
    assert synth_pool.main(["show", "--output-dir", str(pool), "--sample", "9"]) == 0
    everything = capsys.readouterr().out
    assert everything.count("\n── ") == 3 and everything.startswith("── ")
    assert everything.rstrip("\n").splitlines()[-1] == (
        "结果：抽查 4 个已接受的题族（一共 4 个；换个样本：--seed 另给一个数）")
    # diff 是真的 difflib 输出，方向是原程序 → 参考解：变体的那一份去掉的正是第 1 个 mutant 的 new、
    # 加上的是它的 old（这处缺陷的 new 是空的，所以只有加的行）
    slug, variant = picked[0]["slug"], picked[0]["family"]["variants"][0]
    diff = synth_pool.unified_diff(variant).splitlines()
    assert diff[:2] == [f"--- a/s3_{slug}.py", f"+++ b/s3_{slug}.py"] and diff[2].startswith("@@ ")
    assert [line[1:] for line in diff[2:] if line.startswith("+")] == REQUIREMENTS[0][3].splitlines()
    assert [line[1:] for line in diff[2:] if line.startswith("-")] == REQUIREMENTS[0][4].splitlines() == []
    # 母题的那一份：每处缺陷的 old 都被加回来过，new 里不是空的行都被去掉过
    mother = synth_pool.unified_diff(picked[0]["family"]["parent"]).splitlines()[2:]
    added = [line[1:] for line in mother if line.startswith("+")]
    removed = [line[1:] for line in mother if line.startswith("-")]
    for _, _, _, old, new, _ in REQUIREMENTS[:count_of(slug)]:
        assert all(line in added for line in old.splitlines())
        assert all(line in removed for line in new.splitlines())


# ── export ──────────────────────────────────────────────────────────────

def export(capsys, pool, *arguments):
    """跑一次 export，返回 (退出码, 标准输出, 标准错误)。"""
    capsys.readouterr()
    code = synth_pool.main(["export", "--output-dir", str(pool), *arguments])
    shown = capsys.readouterr()
    return code, shown.out, shown.err


def splits_of(specs):
    return {spec["source_group"]: spec["split"] for spec in specs}


def test_export_freezes_only_the_new_families_and_keeps_earlier_splits(tmp_path, capsys):
    pool, tasks_out = tmp_path / "pool", tmp_path / "synth-pool.jsonl"
    first, second = slugs(20)[:8], slugs(20)[8:12]
    for slug in first:
        write_record(pool, accepted(slug, batch=1))
    code, _, err = export(capsys, pool, "--tasks-out", str(tasks_out))
    assert code == 0 and err == ""
    document = json.loads((pool / "splits.json").read_text(encoding="utf-8"))
    before = document["source_groups"]
    tasks = load_specs(tasks_out)
    validate_specs(tasks)
    assert set(before) == {f"s3_{slug}" for slug in first}
    assert len(tasks) == sum(count_of(slug) + 1 for slug in first) and splits_of(tasks) == before
    # 只把这 8 族的新题交给 freeze_splits（按 stratum 分层）：划分和直接算出来的一样
    fresh = [spec for slug in first for spec in synth_pool.family_specs(accepted(slug)["family"])]
    manifest = freeze_splits(fresh, 42, 0.2, 0.2, stratify=True)[1]
    assert manifest["source_groups"] == before and "stratified" in manifest
    assert set(manifest["stratified"]) == {f"synth-{theme(slug)[0].name}" for slug in first}
    assert [(entry["families"], entry["seed"], entry["dev_ratio"], entry["test_ratio"])
            for entry in document["history"]] == [(8, 42, 0.2, 0.2)]
    # 第二批：旧族的划分一个都不变，新族单独冻结
    for slug in second:
        write_record(pool, accepted(slug, batch=2))
    code, out, err = export(capsys, pool, "--tasks-out", str(tasks_out))
    assert code == 0 and err == ""
    after = json.loads((pool / "splits.json").read_text(encoding="utf-8"))["source_groups"]
    assert {group: after[group] for group in before} == before
    assert set(after) == {f"s3_{slug}" for slug in first + second}
    newer = [spec for slug in second for spec in synth_pool.family_specs(accepted(slug)["family"])]
    later = freeze_splits(newer, 42, 0.2, 0.2, stratify=True)[1]["source_groups"]
    assert len(later) == 4 and {group: after[group] for group in later} == later
    tasks = load_specs(tasks_out)
    validate_specs(tasks)
    assert splits_of(tasks) == after and len(tasks) == sum(count_of(slug) + 1 for slug in first + second)
    assert synth_pool.groups_crossing_splits(tasks) == []
    assert len(json.loads((pool / "splits.json").read_text(encoding="utf-8"))["history"]) == 2
    counted = Counter(spec["split"] for spec in tasks)
    groups = Counter(after.values())
    assert sum(counted.values()) == len(tasks) and sum(groups.values()) == 12
    reminder = synth_pool.split_reminder(groups, 0.2, 0.2)      # 划分离 0.2/0.2 太远时才有的一行
    assert out.splitlines() == [
        f"结果：导出 {len(tasks)} 道合成题（母题 12 道、变体 {len(tasks) - 12} 道；这次新冻结 4 族）："
        f"train {counted['train']} 道、dev {counted['dev']} 道、test {counted['test']} 道；"
        "读回来核对过，没有来源组跨划分",
        f"新题文件：{tasks_out}；每道题的审计结论：synth-pool.audit.json；划分：{pool / 'splits.json'}"
        f"（12 个来源族：train {groups['train']}、dev {groups['dev']}、test {groups['test']}）",
        *([reminder] if reminder else [])]
    # 审计结论文件：每族的轮次和教师调用次数，每道题的审计结论
    report = json.loads((tmp_path / "synth-pool.audit.json").read_text(encoding="utf-8"))
    assert set(report["families"]) == set(first + second)
    assert set(report["tasks"]) == {spec["task_id"] for spec in tasks}
    for slug in first + second:
        row, parent_id = report["families"][slug], f"s3_{slug}_001"
        assert (row["rounds"], row["teacher_calls"], row["accepted_in_round"]) == (1, 1, 1)
        assert row["tokens"] == {"input_tokens": 100, "output_tokens": 10} and row["findings"] == []
        assert row["checks"] == dict.fromkeys(audit.CHECKS, True)
        assert (report["tasks"][parent_id]["kind"], report["tasks"][parent_id]["split"]) == (
            "parent", after[f"s3_{slug}"])
        for number in range(1, count_of(slug) + 1):
            task = report["tasks"][f"s3v_{slug}_{number}"]
            assert (task["kind"], task["variant_of"], task["mutant"], task["split"]) == (
                "variant", parent_id, number, after[f"s3_{slug}"])
            assert task["caught_by"] == failed_tests(number)
    # 再导出一次：没有新族要冻结，写出来的文件一个字节都没变
    snapshot = tasks_out.read_bytes(), (pool / "splits.json").read_bytes()
    code, out, err = export(capsys, pool, "--tasks-out", str(tasks_out))
    assert code == 0 and err == "" and "这次新冻结 0 族" in out
    assert (tasks_out.read_bytes(), (pool / "splits.json").read_bytes()) == snapshot


def test_export_writes_one_task_per_line_sorted_and_ascii(tmp_path, capsys):
    pool, tasks_out = tmp_path / "pool", tmp_path / "pool.jsonl"
    wanted = slugs(12)[:4]
    for slug in wanted:
        write_record(pool, accepted(slug))
    code, _, err = export(capsys, pool, "--tasks-out", str(tasks_out))
    assert code == 0 and err == ""
    text = tasks_out.read_text(encoding="utf-8")
    assert text.isascii() and text.endswith("\n")                       # 非 ASCII 都写成转义，一行只有一种换行
    lines = text.split("\n")[:-1]
    tasks = [json.loads(line) for line in lines]
    ordered = sorted(wanted, key=lambda slug: (themes.DIRECTION_NAMES.index(theme(slug)[0].name), slug))
    assert [spec["task_id"] for spec in tasks] == [
        task_id for slug in ordered
        for task_id in (f"s3_{slug}_001", *(f"s3v_{slug}_{number}" for number in range(1, count_of(slug) + 1)))]
    assert lines == [json.dumps(spec, sort_keys=True) for spec in tasks]
    assert (tmp_path / "pool.audit.json").exists()                      # <tasks-out 去掉 .jsonl>.audit.json
    assert len({spec["source_group"] for spec in tasks}) == 4 and synth_pool.groups_crossing_splits(tasks) == []
    # 字段同现有题：母题没有 variant_of，变体指向母题；同一族的题在同一个划分里
    for slug in ordered:
        family_tasks = [spec for spec in tasks if spec["source_group"] == f"s3_{slug}"]
        assert "variant_of" not in family_tasks[0] and all(
            spec["variant_of"] == f"s3_{slug}_001" for spec in family_tasks[1:])
        assert (family_tasks[0]["module"], family_tasks[0]["bug_family"], family_tasks[0]["difficulty"]) == (
            f"s3_{slug}.py", theme(slug)[0].name, "L3")
        assert all(spec["stratum"] == f"synth-{theme(slug)[0].name}" and spec["origin"] == "synth"
                   and spec["direction"] == theme(slug)[0].name for spec in family_tasks)
        assert len({spec["split"] for spec in family_tasks}) == 1



def export_lines(capsys, pool, *arguments):
    """跑一次 export，返回 (退出码, 标准输出, 标准错误)。"""
    return export(capsys, pool, *arguments)


def ladder_text(path, specs):
    """阶梯输出的内容：原有的行原样，后面每行一道新题（键排过序）。"""
    return Path(path).read_text(encoding="utf-8") + "".join(json.dumps(spec, sort_keys=True) + "\n" for spec in specs)


def test_export_merges_the_ladder_and_refuses_to_overwrite_a_different_file(tmp_path, capsys):
    pool, ladder, merged = tmp_path / "pool", tmp_path / "ladder.jsonl", tmp_path / "ladder-v2.jsonl"
    for slug in slugs(12)[:4]:
        write_record(pool, accepted(slug))
    # 原有的行键的顺序不是排过序的：也要原样留着
    original = "".join(json.dumps(spec) + "\n" for spec in (
        old_task("demo_a", "train"), old_task("h3_b", "dev"), old_task("h3v_b", "test")))
    ladder.write_text(original, encoding="utf-8")
    arguments = ["--ladder", str(ladder), "--tasks-out", str(tmp_path / "pool.jsonl"), "--ladder-out", str(merged)]
    code, out, err = export_lines(capsys, pool, *arguments)
    tasks = load_specs(tmp_path / "pool.jsonl")
    assert code == 0 and err == ""
    assert f"并入阶梯：{merged} = {ladder}（原有的 3 道原样保留）+ 新题 {len(tasks)} 道" in out
    assert merged.read_text(encoding="utf-8") == original + "".join(
        json.dumps(spec, sort_keys=True) + "\n" for spec in tasks)
    read_back = load_specs(merged)
    validate_specs(read_back)
    assert [spec["task_id"] for spec in read_back] == ["demo_a", "h3_b", "h3v_b", *[spec["task_id"] for spec in tasks]]
    # 再导出一次：内容一样，说明后跳过
    snapshot = merged.read_bytes()
    code, out, err = export_lines(capsys, pool, *arguments)
    assert code == 0 and err == "" and "已经是这个内容，跳过" in out and merged.read_bytes() == snapshot
    # 有人在里面加了自己的题：这不是我们以前导出的版本，覆盖就会把这一行弄丢——拒绝，退出 1，文件没有变
    extra = json.dumps(old_task("extra", "train"), sort_keys=True) + "\n"
    changed = snapshot.decode("utf-8") + extra
    merged.write_text(changed, encoding="utf-8")
    code, _, err = export_lines(capsys, pool, *arguments)
    assert code == 1 and f"{merged} 已经存在，内容和将要写的不一样，也不是以前导出的版本，不覆盖" in err
    assert "换一个还没有的文件名再导出（LADDER_OUT=…）" in err and merged.read_text(encoding="utf-8") == changed
    # 阶梯自己就是输出文件：不往阶梯里写
    code, _, err = export_lines(capsys, pool, "--ladder", str(ladder), "--tasks-out", str(tmp_path / "pool.jsonl"),
                                "--ladder-out", str(ladder))
    assert code == 1 and "阶梯文件和输出是同一个文件" in err and ladder.read_text(encoding="utf-8") == original


def test_a_ladder_from_an_earlier_export_is_updated_with_the_new_batches(tmp_path, capsys):
    """默认的 LADDER_OUT 是固定的一个文件：第 2 批、第 3 批导出时它已经在了。里面是这个脚本以前导出的
    （阶梯原来的行加前几批的题）：更新它不丢任何一行，所以原样跑下一批就能接着并进去。"""
    pool, ladder, merged = tmp_path / "pool", tmp_path / "ladder.jsonl", tmp_path / "ladder-v2.jsonl"
    ladder.write_text("".join(json.dumps(old_task(name, "train"), sort_keys=True) + "\n"
                              for name in ("demo_a", "h3_c")), encoding="utf-8")
    original = ladder.read_text(encoding="utf-8")
    tasks_out = tmp_path / "pool.jsonl"
    arguments = ["--ladder", str(ladder), "--tasks-out", str(tasks_out), "--ladder-out", str(merged)]
    for number, batch in enumerate((slugs(20)[:6], slugs(20)[6:12], slugs(20)[12:18]), 1):
        for slug in batch:
            write_record(pool, accepted(slug, batch=number))
        code, out, err = export_lines(capsys, pool, *arguments)       # 参数不变：默认的 LADDER_OUT 没有换
        tasks = load_specs(tasks_out)
        assert code == 0 and err == ""
        assert merged.read_text(encoding="utf-8") == ladder_text(ladder, tasks)
        note = (f"并入阶梯：{merged} = {ladder}（原有的 2 道原样保留）+ 新题 {len(tasks)} 道" if number == 1 else
                f"阶梯文件 {merged} 是以前导出的（只有前几批的题），更新成现在的内容：{ladder} 原有的 2 道"
                f"原样保留，加新题 {len(tasks)} 道")
        assert note in out
        assert ladder.read_text(encoding="utf-8") == original
    read_back = load_specs(merged)
    validate_specs(read_back)
    assert [spec["task_id"] for spec in read_back][:2] == ["demo_a", "h3_c"] and len(read_back) == 2 + len(tasks)


def test_export_reminds_when_the_frozen_split_is_far_off(tmp_path, capsys):
    pool, tasks_out = tmp_path / "pool", tmp_path / "pool.jsonl"
    for slug in slugs(12)[:4]:
        write_record(pool, accepted(slug))
    code, out, err = export_lines(capsys, pool, "--tasks-out", str(tasks_out))
    assert code == 0 and err == "" and "（4 个来源族：train 4、dev 0、test 0）" in out
    # 头一回导出时接受的族太少，一个方向只有一族，分不出 dev 和 test；冻结过的族以后也不动
    assert ("提醒：4 个来源族里 dev 占 0%、test 占 0%，比要的 20%、20% 少得多。冻结过的族划分不再改（加了批次，"
            "旧族也不动），早先导出的那几族留在 train 里，比例偏了以后只能靠新批次慢慢往回拉。想一次分得准，"
            "等接受的族多一些（每个方向五六族以上）再导出") in out
    lopsided = synth_pool.split_reminder(Counter({"train": 9, "dev": 2, "test": 1}), 0.2, 0.2)
    assert lopsided.startswith("提醒：12 个来源族里 dev 占 17%、test 占 8%，比要的 20%、20% 少得多。")
    assert synth_pool.split_reminder(Counter({"train": 16, "dev": 4, "test": 4}), 0.2, 0.2) is None
    assert synth_pool.split_reminder(Counter({"train": 3, "dev": 5, "test": 2}), 0.5, 0.2) is None      # 要 50%/20%
    assert synth_pool.split_reminder(Counter(), 0.2, 0.2) is None


def test_export_chains_a_new_ladder_from_the_previous_one(tmp_path, capsys):
    pool, tasks_out = tmp_path / "pool", tmp_path / "synth-pool.jsonl"
    first, second = slugs(12)[:4], slugs(12)[4:8]
    ladder, second_ladder, third_ladder = (tmp_path / name for name in ("ladder.jsonl", "v2.jsonl", "v3.jsonl"))
    ladder.write_text("".join(json.dumps(old_task(name, "train"), sort_keys=True) + "\n"
                              for name in ("demo_a", "h3_c")), encoding="utf-8")
    for slug in first:
        write_record(pool, accepted(slug))
    assert export(capsys, pool, "--ladder", str(ladder), "--tasks-out", str(tasks_out),
                  "--ladder-out", str(second_ladder))[0] == 0
    earlier = load_specs(tasks_out)
    for slug in second:
        write_record(pool, accepted(slug, batch=2))
    code, out, err = export(capsys, pool, "--ladder", str(second_ladder), "--tasks-out", str(tasks_out),
                            "--ladder-out", str(third_ladder))
    assert code == 0 and err == "" and f"阶梯里已经有 {len(earlier)} 道合成题，不再加一遍" in out
    merged = load_specs(third_ladder)
    validate_specs(merged)
    groups = {f"s3_{slug}" for slug in second}
    assert [spec["task_id"] for spec in merged] == [
        "demo_a", "h3_c", *[spec["task_id"] for spec in earlier],
        *[spec["task_id"] for spec in load_specs(tasks_out) if spec["source_group"] in groups]]


def test_export_without_a_ladder_file_only_exports_the_new_tasks(tmp_path, capsys):
    pool, tasks_out = tmp_path / "pool", tmp_path / "pool.jsonl"
    ladder_out, missing = tmp_path / "nowhere-v2.jsonl", tmp_path / "nowhere.jsonl"
    for slug in slugs(4)[:3]:
        write_record(pool, accepted(slug))
    code, out, err = export(capsys, pool, "--ladder", str(missing), "--tasks-out", str(tasks_out),
                            "--ladder-out", str(ladder_out))
    assert code == 0 and err == "" and tasks_out.exists() and not ladder_out.exists()
    assert f"阶梯文件 {missing} 不存在：只导出新题，没有写 {ladder_out}" in out
    # 阶梯在、却没说写到哪：不做，也没有写新题文件
    ladder, other_out = tmp_path / "ladder.jsonl", tmp_path / "other.jsonl"
    ladder.write_text(json.dumps(old_task("demo_a", "train"), sort_keys=True) + "\n", encoding="utf-8")
    code, _, err = export(capsys, pool, "--ladder", str(ladder), "--tasks-out", str(other_out))
    assert code == 1 and "就要给 --ladder-out" in err and not other_out.exists()




def test_fewer_than_three_new_families_are_exported_into_train(tmp_path, capsys):
    """freeze_splits 至少要三个来源族才肯分。只接受了两族的时候照样导出：这么几族本来就分不出 dev 和 test，
    都记在 train 里，以后也不再改；不让整次导出失败（那两族的题照样能用）。"""
    pool, tasks_out = tmp_path / "pool", tmp_path / "pool.jsonl"
    first, third = slugs(4)[:2], slugs(4)[2]
    for slug in first:
        write_record(pool, accepted(slug))
    code, out, err = export_lines(capsys, pool, "--tasks-out", str(tasks_out))
    tasks = load_specs(tasks_out)
    validate_specs(tasks)
    assert code == 0 and err == ""
    assert len(tasks) == sum(count_of(slug) + 1 for slug in first) and {task["split"] for task in tasks} == {"train"}
    assert ("新接受的题族只有 2 个：freeze_splits 至少要 3 个来源族才分得开，这么几族也分不出 dev 和 test，"
            "都记在 train 里（以后不再改）") in out
    document = json.loads((pool / "splits.json").read_text(encoding="utf-8"))
    groups = [f"s3_{slug}" for slug in first]
    assert [document["source_groups"][group] for group in groups] == ["train", "train"]
    assert [(entry["families"], entry["how"]) for entry in document["history"]] == [(2, "train_only")]
    # 再来一族：只冻结还没冻结过的，新的只有这一族，照样记在 train 里；头两族的划分不变
    write_record(pool, accepted(third))
    code, out, err = export_lines(capsys, pool, "--tasks-out", str(tasks_out))
    document = json.loads((pool / "splits.json").read_text(encoding="utf-8"))
    assert code == 0 and err == "" and "新接受的题族只有 1 个" in out
    assert [document["source_groups"][group] for group in groups] == ["train", "train"]
    assert [document["source_groups"][f"s3_{third}"]] == ["train"]
    how = [(entry["families"], entry["how"]) for entry in document["history"]]
    assert how == [(2, "train_only"), (1, "train_only")]
    # 一次新来三族：交给 freeze_splits；已经冻结过的那三族不动
    later = slugs(9)[3:6]
    for slug in later:
        write_record(pool, accepted(slug, batch=2))
    code, out, err = export_lines(capsys, pool, "--tasks-out", str(tasks_out))
    document = json.loads((pool / "splits.json").read_text(encoding="utf-8"))
    assert code == 0 and err == "" and "新接受的题族只有" not in out
    assert [document["source_groups"][group] for group in [*groups, f"s3_{third}"]] == ["train"] * 3
    assert [(entry["families"], entry["how"]) for entry in document["history"]] == [
        (2, "train_only"), (1, "train_only"), (3, "freeze_splits")]
    tasks = load_specs(tasks_out)
    assert synth_pool.groups_crossing_splits(tasks) == [] and len(tasks) == sum(
        count_of(slug) + 1 for slug in (*first, third, *later))
    # 比例写得不对：freeze_splits 会拒的，这里同样拒，话里说清是哪一项
    for dev, test in ((0.8, 0.8), (0, 0.2), (1, 0.1), (0.2, 1.5)):
        with pytest.raises(synth_pool.ExportError, match="dev 和 test 的比例"):
            synth_pool.freeze_new([], [], 42, dev, test)
    # 一个已接受的都没有：说清楚目录里是什么、接着怎么办，不再叫人"原样重跑"
    empty = tmp_path / "empty"
    synth_pool.families_dir(empty).mkdir(parents=True)
    tasks_out.unlink()
    code, _, err = export_lines(capsys, empty, "--tasks-out", str(tasks_out))
    assert code == 1 and "里还没有题族记录，没有可导出的题：先跑一批（bash scripts/run_synth_pool.sh）" in err
    assert not tasks_out.exists()
    lost = tmp_path / "lost"
    for slug in first:
        write_record(lost, rejected(slug, REFERENCE_FAILS))
    code, _, err = export_lines(capsys, lost, "--tasks-out", str(tasks_out))
    assert code == 1 and (f"{lost} 里没有已接受的题族（2 份记录都被拒绝了），没有可导出的题。各是为什么看 "
                          f"python scripts/synth_pool.py status --output-dir {lost}；都是因为题族本身被拒绝的，"
                          f"换一批主题再跑（BATCH 写成下一批的批号）") in err
    assert not tasks_out.exists()
    write_record(lost, rejected(slugs(5)[4], "教师调用出错"))
    code, _, err = export_lines(capsys, lost, "--tasks-out", str(tasks_out))
    assert code == 1 and ("；其中 1 族没有定论（不是题族的问题），照 run 上次打印的提示把它们补做"
                         "（FAMILIES 写成那一批里这样的族数）再导出") in err


def test_export_refuses_a_ladder_that_would_break_the_pool(tmp_path, capsys):
    pool, tasks_out, ladder, ladder_out = (tmp_path / name for name in (
        "pool", "pool.jsonl", "ladder.jsonl", "v2.jsonl"))
    wanted = slugs(4)
    for slug in wanted:
        write_record(pool, accepted(slug))
    slug = wanted[0]
    assert export(capsys, pool, "--tasks-out", str(tasks_out))[0] == 0          # 先冻结，拿到各族的划分
    document = json.loads((pool / "splits.json").read_text(encoding="utf-8"))
    split = document["source_groups"][f"s3_{slug}"]
    variant = accepted(slug)["family"]["variants"][0]
    arguments = ["--ladder", str(ladder), "--tasks-out", str(tasks_out), "--ladder-out", str(ladder_out)]
    # 阶梯里有同一个题号的题，但内容不一样
    ladder.write_text(json.dumps({**variant, "fixed": variant["fixed"] + "# 多一行\n"}, sort_keys=True) + "\n",
                      encoding="utf-8")
    code, _, err = export(capsys, pool, *arguments)
    assert code == 1 and f"阶梯文件 {ladder} 里已经有 {variant['task_id']}，但和要加的这道题不一样" in err
    assert not ladder_out.exists() and document == json.loads((pool / "splits.json").read_text(encoding="utf-8"))
    # 阶梯里有同一来源组、却在另一个划分里的题：并进去就有来源组跨划分
    moved = {**variant, "task_id": f"s3v_{slug}_77", "split": "dev" if split != "dev" else "train"}
    ladder.write_text(json.dumps(moved, sort_keys=True) + "\n", encoding="utf-8")
    code, _, err = export(capsys, pool, *arguments)
    assert code == 1 and "以后不合法" in err and "source_group crosses splits" in err and not ladder_out.exists()


def test_export_stops_when_the_frozen_splits_cannot_be_read(tmp_path, capsys):
    pool, tasks_out = tmp_path / "pool", tmp_path / "pool.jsonl"
    for slug in slugs(4)[:3]:
        write_record(pool, accepted(slug))
    for text in ("{坏掉的", json.dumps({"source_groups": {"s3_x": "yesterday"}})):
        (pool / "splits.json").write_text(text, encoding="utf-8")
        code, _, err = export(capsys, pool, "--tasks-out", str(tasks_out))
        assert code == 1 and "splits.json 读不出来" in err and "已经冻结的划分" in err
        assert not tasks_out.exists()


# ── 停下：写不进盘、Ctrl-C、kill、启动它的脚本没了 ──────────────────────────

def test_a_record_that_cannot_be_written_stops_the_batch_before_more_teacher_calls(tmp_path, monkeypatch):
    wanted = slugs(6)
    real = synth_pool.write_record
    failed = []

    def write(pool, record):
        if failed or record["slug"] != wanted[0]:
            failed.append(record["slug"])
            raise OSError(28, "No space left on device")            # 磁盘满了
        real(pool, record)

    monkeypatch.setattr(synth_pool, "write_record", write)
    teacher = Teacher(*[TRUNCATED] * 12)
    summary, lines, records = run(tmp_path / "pool", teacher, wanted, rounds=2)
    # 第一族写成了；第二族的回答要回来了却写不进盘：马上停下，后面四族不再向教师要
    assert failed == [wanted[1]] and len(teacher.calls) == 4
    assert [record["slug"] for record in records] == [wanted[0]]
    assert (summary["skipped"], summary["stopped"], summary["accepted"], summary["calls"]) == (4, True, 0, 4)
    assert summary["unwritten"] == [f"{wanted[1]}（[Errno 28] No space left on device）"]
    assert summary["unsettled"] == 2
    message = (f"有 1 族的记录写不进 {synth_pool.families_dir(tmp_path / 'pool')}：{wanted[1]}"
               "（[Errno 28] No space left on device）。磁盘满了，还是目录只读？这一批停下了，不再调用教师。"
               "腾出空间或者改好权限以后原样重跑——没写进盘的题族会再要一次。")
    assert message in lines
    assert lines[-2] == ("修好以后原样重跑这一批（没做完的排在最前，再要满这一批的 6 个）："
                         "BATCH=1 bash scripts/run_synth_pool.sh")


def test_main_run_exits_1_when_records_cannot_be_written(tmp_path, teacher_api, capsys, monkeypatch):
    pool = tmp_path / "pool"
    client, _ = teacher_api(*[response(TRUNCATED)] * 12)

    def cannot_write(pool, record):
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(synth_pool, "write_record", cannot_write)
    code = synth_pool.main(["run", "--output-dir", str(pool), "--families", "6", "--concurrency", "1"])
    assert code == 1 and len(client.requests) == 2      # 第一族就写不进：它做完（两轮）就停，后面五族没有请求
    assert "有 1 族的记录写不进" in capsys.readouterr().err


def test_a_family_in_flight_when_the_batch_stops_is_left_for_the_rerun(tmp_path):
    slug, stop = slugs(2)[0], threading.Event()
    options = dict(batch=1, rounds=3, auditor=refusing("reference_fails"), work_dir=tmp_path / "work",
                   accepted=synth_pool.Accepted([]), stop=stop)

    def interrupted(messages, max_tokens, temperature):
        stop.set()                                       # 这一批在手上这一轮还没做完时被停下了
        return reply(broken(slug))

    record = synth_pool.process_family(theme(slug), interrupted, **options)
    assert len(record["rounds"]) == 1 and record["reason"] == "停下时还没做完"
    assert synth_pool.settled(record) is False            # 不算定论：重跑时会再要
    stop.clear()
    calls = []

    def counting(messages, max_tokens, temperature):
        calls.append(messages)
        return reply(broken(slug))

    record = synth_pool.process_family(theme(slug), counting, **options)
    assert len(calls) == 3 and record["reason"] == REFERENCE_FAILS     # 没停下的话：返修到轮次用完


def test_the_first_ctrl_c_finishes_the_families_in_flight_saves_them_and_exits_130(
        tmp_path, teacher_api, capsys, reject_on_content, monkeypatch):
    pool, wanted = tmp_path / "pool", slugs(6)
    client, _ = teacher_api(*(response(reply(family(slug))) for slug in wanted))
    client.delay = 0.5
    real = vllm_client.map_concurrent
    pressed = []

    def interrupt_once(fn, items, workers):
        for index, result in real(fn, items, workers):
            yield index, result
            if not pressed:
                pressed.append(index)
                # 同一次 Ctrl-C：处理器先把开关置上，再打断主线程
                signal.getsignal(signal.SIGINT)(signal.SIGINT, None)

    monkeypatch.setattr(vllm_client, "map_concurrent", interrupt_once)
    handlers = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    code = synth_pool.main(["run", "--output-dir", str(pool), "--families", "6", "--concurrency", "1"])
    shown = capsys.readouterr()
    assert pressed == [0] and code == 130
    assert {number: signal.getsignal(number) for number in handlers} == handlers      # 原来的处理器装回去了
    # 正在做的第二族做完、记录落盘；排着的四族没有请求
    assert len(client.requests) == 2
    assert [record["slug"] for record in synth_pool.read_records(pool)] == sorted(wanted[:2])
    assert ("收到 Ctrl-C，这一批停下了：6 个主题里 2 个已经有了定论，其余的没有做完。原样重跑会接着做（没做完的"
            "排在最前，再要满这一批）；只补它们、不要新主题：FAMILIES=4 BATCH=1 bash scripts/run_synth_pool.sh") \
        in shown.err


DRIVER = r'''
"""代替命令行跑 synth_pool.main 的 run：教师接口换成假的（每次调用睡一会儿、回一段没法解析的话），
每次调用往 CALLS 里记一行。"""
import os, sys, time, types
sys.path[:0] = [sys.argv[1], sys.argv[2]]                  # src、scripts
import _shared_teacher, synth_pool


class Client:
    def __init__(self):
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def create(self, model, messages, max_tokens, temperature):
        with open(os.environ["CALLS"], "a", encoding="utf-8") as handle:
            handle.write("call\n")
        time.sleep(float(os.environ.get("SECONDS", "3")))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content="这不是 JSON"), finish_reason="stop")], usage=None)


_shared_teacher.make_client = lambda *args: Client()
os.environ["TEACHER_API_KEY"] = "test-key-not-a-real-one"
sys.exit(synth_pool.main(sys.argv[3:]))
'''


def start_run(tmp_path, *arguments, seconds, output=None, extra=None):
    """起一个子进程，照命令行跑真的 run（教师是假的；请求之间不隔，这里看的是信号）。返回 (进程, 记调用的
    文件, 它的输出)。"""
    calls, driver = tmp_path / "calls.log", tmp_path / "driver.py"
    driver.write_text(DRIVER, encoding="utf-8")
    environment = {name: value for name, value in os.environ.items()
                   if not name.startswith(("TEACHER_", "SYNTH_POOL_", "CALLS", "SECONDS"))}
    environment.update(CALLS=str(calls), SECONDS=str(seconds), **(extra or {}))
    command = [sys.executable, str(driver), str(ROOT / "src"), str(ROOT / "scripts"), "run",
               "--output-dir", str(output or tmp_path / "pool"), "--min-interval", "0", *arguments]
    log = open(tmp_path / "run.log", "w", encoding="utf-8")       # noqa: SIM115   进程的输出都进这个文件
    process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT, cwd=tmp_path)
    return process, calls, log


def wait_calls(calls, count, seconds=30):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if calls.exists() and len(calls.read_text(encoding="utf-8").splitlines()) >= count:
            return True
        time.sleep(0.05)
    return False


def finish(process, log, seconds=30):
    """等子进程退出：返回 (退出码, 从调用它到退出过了几秒, 它打印的全部内容)。"""
    started = time.monotonic()
    code = process.wait(timeout=seconds)
    log.close()
    return code, time.monotonic() - started, (log.name and Path(log.name).read_text(encoding="utf-8"))


def work_dirs():
    return {path.name for path in Path(tempfile.gettempdir()).glob("synth-pool-*")}


def test_sigterm_stops_the_run_at_once(tmp_path):
    before = work_dirs()
    process, calls, log = start_run(tmp_path, "--families", "6", "--concurrency", "2", seconds=6)
    assert wait_calls(calls, 2)
    process.send_signal(signal.SIGTERM)
    code, elapsed, shown = finish(process, log)
    assert code == 143 and elapsed < 4                  # 正在进行的两次调用各要 6 秒：没有等它们
    assert calls.read_text(encoding="utf-8").splitlines() == ["call", "call"]     # 排着的四个主题没有请求
    assert synth_pool.read_records(tmp_path / "pool") == []       # 正在做的两族这一次的回答没有保存
    assert "收到 kill 信号：立刻停下。排着的主题没有请求；正在进行的几个题族这一次的回答没有保存，重跑会再要。" \
        in shown
    assert work_dirs() - before == set()                # 审计用的临时目录删掉了


def test_the_first_sigint_lets_the_families_in_flight_finish_and_exits_130(tmp_path):
    process, calls, log = start_run(tmp_path, "--families", "6", "--concurrency", "2", "--rounds", "1", seconds=3)
    assert wait_calls(calls, 2)
    process.send_signal(signal.SIGINT)
    code, elapsed, shown = finish(process, log)
    assert code == 130 and 2 < elapsed < 20             # 没有立刻退：等正在做的两族做完
    assert calls.read_text(encoding="utf-8").splitlines() == ["call", "call"]     # 排着的四个主题没有请求
    records = synth_pool.read_records(tmp_path / "pool")
    assert len(records) == 2 and all(record["reason"] == "解析失败" for record in records)
    assert "收到 Ctrl-C，这一批停下了：6 个主题里 0 个已经有了定论，其余的没有做完。" in shown
    assert "只补它们、不要新主题：FAMILIES=6 BATCH=1 bash scripts/run_synth_pool.sh" in shown


def test_a_second_sigint_leaves_at_once(tmp_path):
    process, calls, log = start_run(tmp_path, "--families", "6", "--concurrency", "2", "--rounds", "1", seconds=6)
    assert wait_calls(calls, 2)
    process.send_signal(signal.SIGINT)
    time.sleep(0.3)
    process.send_signal(signal.SIGINT)
    code, elapsed, shown = finish(process, log)
    assert code == 130 and elapsed < 4 and calls.read_text(encoding="utf-8").splitlines() == ["call", "call"]
    assert synth_pool.read_records(tmp_path / "pool") == []


def test_the_run_stops_when_the_shell_that_started_it_is_killed(tmp_path):
    """run_synth_pool.sh 调 run 时设了 SYNTH_POOL_STOP_WITH_PARENT=1：脚本被 kill -9 时 trap 也执行不了，
    python 发现父进程没了，就照 kill 处理。"""
    if not Path("/proc/self/stat").exists():
        pytest.skip("要看 /proc 才知道一个不归自己管的进程退没退")
    calls, driver = tmp_path / "calls.log", tmp_path / "driver.py"
    driver.write_text(DRIVER, encoding="utf-8")
    pool, pid_file = tmp_path / "pool", tmp_path / "python.pid"
    environment = {name: value for name, value in os.environ.items()
                   if not name.startswith(("TEACHER_", "SYNTH_POOL_", "CALLS", "SECONDS"))}
    environment.update(CALLS=str(calls), SECONDS="6", SYNTH_POOL_STOP_WITH_PARENT="1")
    command = (f'"{sys.executable}" "{driver}" "{ROOT / "src"}" "{ROOT / "scripts"}" run --output-dir "{pool}" '
               f'--families 6 --concurrency 2 --rounds 1 --min-interval 0 & echo $! > "{pid_file}"; wait')
    log = open(tmp_path / "run.log", "w", encoding="utf-8")       # noqa: SIM115   子进程的输出都进这个文件
    shell = subprocess.Popen(["bash", "-c", command], env=environment, stdout=log, stderr=subprocess.STDOUT,
                             cwd=tmp_path, start_new_session=True)
    try:
        assert wait_calls(calls, 2)
        shell.kill()                                    # kill -9：只打到 bash，python 成了孤儿
        shell.wait(timeout=10)
        orphan = int(pid_file.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and alive(orphan):
            time.sleep(0.1)
        assert not alive(orphan)                        # 盯着父进程的那个线程两秒查一次：发现就退出
        assert calls.read_text(encoding="utf-8").splitlines() == ["call", "call"]
        log.flush()
        assert "启动这个进程的脚本已经没了（被强行结束）" in Path(log.name).read_text(encoding="utf-8")
    finally:
        log.close()
        if shell.poll() is None:
            shell.kill()


def alive(pid):
    """这个进程还在不在（僵尸算不在）。"""
    try:
        state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()[0]
    except OSError:
        return False
    return state != "Z"


# ── 归类和加锁 ───────────────────────────────────────────────────────────

def test_every_finding_the_audit_can_report_has_a_reason_label():
    """审计报出来的每一个 code 都有给人看的拒绝原因：流水线按 code 归类，漏了就会把英文 code 原样打给用户。"""
    source = Path(audit.__file__).read_text(encoding="utf-8")
    literals = set(re.findall(r'self\.add\(\s*"([a-z_]+)"', source))
    suffixes = set(re.findall(r'f"\{kind\}_([a-z_]+)"', source))
    emitted = literals | {prefix + suffix for prefix in ("mutant_", "parent_") for suffix in suffixes}
    assert len(emitted) == 32 and emitted == set(synth_pool.REASONS)
    assert set(synth_pool.DECISIVE) <= emitted
    # 这些原因说的不是题族的事：没有定论，重跑时会再要一次
    assert synth_pool.NOT_ABOUT_CONTENT == {
        "教师调用出错", synth_pool.REASONS["grading_error"], "处理时程序出错", "解析失败", "回复被截断",
        "停下时还没做完"}
    assert set(synth_pool.NOT_ABOUT_CONTENT_ORDER) == synth_pool.NOT_ABOUT_CONTENT


def test_the_reason_of_a_rejected_family_is_what_blocked_it():
    def entry(**fields):
        return {"error": None, "parse_error": None, "findings": None, **fields}

    def finding(code, repairable=True):
        return {"code": code, "message": "说明", "repairable": repairable}

    blocked = synth_pool.reason_of(entry(findings=[finding("mutant_escapes"), finding("too_few_mutants")]))
    assert blocked == "可用的 mutant 太少"                     # 各个 mutant 自己的发现不是决定性的那一条
    worst = synth_pool.reason_of(entry(findings=[finding("reference_fails"), finding("collision", False)]))
    assert worst == "和现有题池撞了"                             # 不可返修的那一条先算
    softest = synth_pool.reason_of(entry(findings=[finding("hint_comment"), finding("indistinct_tests")]))
    assert softest == "注释里有提示缺陷的标记"                   # 都不是决定性的：取头一条的
    assert synth_pool.reason_of(entry(parse_error="回复不是合格的题族")) == "解析失败"
    assert synth_pool.reason_of(entry(error="TimeoutError: 超时")) == "教师调用出错"
    cut = entry(parse_error="回复里的 JSON 对象不完整", finish_reason="length")
    assert synth_pool.reason_of(cut) == "回复被截断"        # 在输出上限处被截断的，单独说：要调大 --max-tokens
    # 正文是空的、结束原因却不是 length（推理占满了输出）：同一种。旧记录没有 truncated 这一项，照旧看 finish_reason
    assert synth_pool.reason_of(entry(parse_error="x", finish_reason="stop", truncated=True)) == "回复被截断"
    assert synth_pool.reason_of(entry(parse_error="x", finish_reason="stop")) == "解析失败"
    assert synth_pool.reason_of(entry(parse_error="回复里的 JSON 对象不完整")) == "解析失败"
    assert synth_pool.reason_of(entry()) is None and synth_pool.reason_of(entry(findings=[])) is None
    record = {"rounds": [entry(error="x"), entry(findings=[finding("reference_fails")]), entry(findings=[])]}
    assert synth_pool.final_reason(record) == REFERENCE_FAILS   # 最后一个说明了问题的轮次
    assert synth_pool.final_reason({"rounds": []}) == "审计没有接受，也没有说明原因"


def test_one_run_at_a_time_per_directory(tmp_path):
    with pytest.raises(RuntimeError, match="另一个 synth_pool.py run 正在"):
        with synth_pool.one_at_a_time(tmp_path):
            with synth_pool.one_at_a_time(tmp_path):
                pass
    (tmp_path / "other").mkdir()
    with synth_pool.one_at_a_time(tmp_path):                    # 上一把锁放开了就能再进；别的目录互不影响
        with synth_pool.one_at_a_time(tmp_path / "other"):
            pass


def test_the_pool_sources_carry_no_private_details():
    """这几个文件会进公开仓库：显卡型号、个人路径、服务商名和商用模型名、密钥一个都不能有。规则直接取
    scripts/privacy_check.py 的 CONTENT_PATTERNS（推送前的隐私审查用的就是它）。"""
    patterns = [re.compile(pattern, re.IGNORECASE) for _, pattern in privacy_check.CONTENT_PATTERNS]
    for name in ("scripts/synth_pool.py", "scripts/run_synth_pool.sh", "scripts/tier_report.py",
                 "src/agentlab/benchmark/tiers.py", "tests/test_synth_pool.py", "tests/test_synth_tiers.py",
                 "tests/test_run_synth_pool_script.py"):
        text = (ROOT / name).read_text(encoding="utf-8")
        for pattern in patterns:
            assert not pattern.search(text), (name, pattern.pattern)
