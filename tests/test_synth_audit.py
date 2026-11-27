# -*- coding: utf-8 -*-
"""合成题族的规格构造和执行审计。

题族是手工构造的：一个 41 行的购物车程序（Cart）、三条要求、三个 mutant，每个 mutant 只挂自己
那条要求的隐藏测试。判分是真的：审计在沙箱里跑 pytest，所以这个文件比别的测试慢。
"""
import ast
import copy
import itertools
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import privacy_check  # noqa: E402
from agentlab.benchmark.grader import GradeInfrastructureError
from agentlab.benchmark.protocol import validate_specs
from agentlab.synth import audit, specs

SLUG, DIRECTION = "cart", "cart_rules"

CART = '''
class Cart:
    """购物车：按商品名累计件数，支持优惠码，运费按件数计算。"""

    def __init__(self):
        self._items = {}
        self._discount = 0

    def add(self, name, price, count=1):
        # 同名商品累加件数，单价以第一次加入时为准
        if name not in self._items:
            self._items[name] = {'price': price, 'count': 0}
        self._items[name]['count'] += count

    def remove(self, name, count=1):
        item = self._items[name]
        item['count'] -= count
        if item['count'] <= 0:
            del self._items[name]

    def apply_code(self, code, percent):
        # 折扣百分比必须在 0 到 100 之间
        if not 0 <= percent <= 100:
            raise ValueError('percent out of range')
        self._discount = percent

    def subtotal(self):
        return sum(item['price'] * item['count'] for item in self._items.values())

    def shipping(self):
        # 商品金额满 100 免运费，否则每件 5
        if self.subtotal() >= 100:
            return 0
        return 5 * sum(item['count'] for item in self._items.values())

    def total(self):
        # 优惠码只折扣商品金额，运费不打折
        discounted = self.subtotal() * (100 - self._discount) // 100
        return discounted + self.shipping()

    def summary(self):
        return ', '.join(f'{name} x{item["count"]}' for name, item in sorted(self._items.items()))
'''

INSTRUCTION = (
    "修复 Cart：购物车按商品名累计件数、支持优惠码、计算运费和总价，现在的实现有几处疏忽。要求：\n"
    "1）add(name, price, count=1)：同名商品累加件数，单价以第一次加入时为准。\n"
    "2）remove(name, count=1)：减少件数；件数减到 0 或更少时该商品从购物车里消失，再次加入时件数从头算。\n"
    "3）total()：优惠码只折扣商品金额，运费不打折；shipping() 在商品金额满 100 时免运费，否则每件 5。")

PUBLIC = '''
def test_public_cart():
    cart = Cart()
    cart.add('apple', 10)
    cart.add('pear', 20, 2)
    assert cart.subtotal() == 50
    assert cart.total() == 65
    assert cart.summary() == 'apple x1, pear x2'
'''

TEST_COUNT = '''
def test_count():
    cart = Cart()
    cart.add('pear', 20)
    cart.add('pear', 99, 2)
    assert cart.subtotal() == 60          # 三件，单价仍是第一次加入时的 20
    assert cart.summary() == 'pear x3'
'''

TEST_REMOVE = '''
def test_remove():
    cart = Cart()
    cart.add('apple', 10, 2)
    cart.remove('apple', 2)
    with pytest.raises(KeyError):
        cart.remove('apple')
    assert cart.summary() == ''
    cart.add('apple', 10)
    assert cart.summary() == 'apple x1'
'''

TEST_TOTAL = '''
def test_discount():
    cart = Cart()
    cart.add('pear', 20, 2)
    cart.apply_code('NEW', 50)
    assert cart.shipping() == 10
    assert cart.total() == 30             # 20 是折扣后的商品金额，运费 10 不打折


def test_free_shipping():
    cart = Cart()
    cart.add('plum', 30, 4)
    assert cart.shipping() == 0
    assert cart.total() == 120
'''

TEST_CODE = '''
@pytest.mark.parametrize('percent', Cart.PERCENTS)
def test_code_accepts_listed_percent(percent):
    cart = Cart()
    cart.apply_code('X', percent)
    cart.add('pear', 20)
    assert cart.total() == 20 * (100 - percent) // 100 + 5


def test_code_rejects_other_percent():
    cart = Cart()
    with pytest.raises(ValueError):
        cart.apply_code('X', 101)
    with pytest.raises(ValueError):
        cart.apply_code('Y', -1)
'''

# 三个测试文件里的测试都把三条要求走一遍：每个 mutant 都让同一批测试失败
FLOW = '''
def test_cart_flow():
    cart = Cart()
    cart.add('pear', 20)
    cart.add('pear', 20)                   # 要求 1：件数累加
    assert cart.subtotal() == 40
    cart.remove('pear', 2)                 # 要求 2：减到 0 后条目消失
    assert cart.summary() == ''
    cart.add('plum', 30, 2)
    cart.apply_code('HALF', 50)            # 要求 3：运费不打折
    assert cart.total() == 30 + 10
'''

TEST_SHIPPING = '''
def test_shipping_per_piece():
    cart = Cart()
    cart.add('pear', 20, 2)
    cart.add('plum', 5)
    assert cart.shipping() == 15           # 三件
'''

PUBLIC_ADDS_TWICE = '''
def test_public_cart():
    cart = Cart()
    cart.add('pear', 20)
    cart.add('pear', 20)
    assert cart.subtotal() == 40
    assert cart.total() == 50              # 40 加上两件的运费 10
'''

PUBLIC_AFTER_REMOVE = '''
def test_public_cart():
    cart = Cart()
    cart.add('apple', 10)
    cart.remove('apple')
    cart.add('pear', 30)
    cart.add('plum', 40)
    assert cart.subtotal() == 70
    assert cart.total() == 80              # 70 加上两件商品的运费 10
'''

UNSAFE_HELPERS = '''

def run_case(name):
    return eval(name)                      # FIXME 这里只是演示


def sync_clock():
    os.system('true')
'''

COUNT, REMOVE = "tests_private.test_count::test_count", "tests_private.test_remove::test_remove"
DISCOUNT = "tests_private.test_total::test_discount"
PUBLIC_TEST = "tests_public.test_public::test_public_cart"


def cart_family():
    """完好的小题族：三条要求、三个 mutant。每个 mutant 让自己那条要求的隐藏测试失败，公开测试照常通过。"""
    return {
        "theme": "购物车：累计件数、优惠码、运费和总价",
        "instruction": INSTRUCTION, "fixed": CART, "public_test": PUBLIC,
        "tests": {"count": {"requirement": 1, "body": TEST_COUNT},
                  "remove": {"requirement": 2, "body": TEST_REMOVE},
                  "total": {"requirement": 3, "body": TEST_TOTAL}},
        "mutants": [
            {"requirement": 1, "old": "self._items[name]['count'] += count",
             "new": "self._items[name]['count'] = count", "idea": "覆盖了件数而不是累加"},
            {"requirement": 2, "old": "if item['count'] <= 0:", "new": "if item['count'] < 0:",
             "idea": "刚好减到 0 时没有删掉条目"},
            {"requirement": 3, "old": "return discounted + self.shipping()",
             "new": "return (self.subtotal() + self.shipping()) * (100 - self._discount) // 100",
             "idea": "折扣也算到了运费上"},
        ],
    }


def with_code_rule(parsed):
    """加上第 4 条要求（优惠码的取值范围）：题族变成四条要求、四个 mutant。"""
    parsed = copy.deepcopy(parsed)
    docstring = '    """购物车：按商品名累计件数，支持优惠码，运费按件数计算。"""\n'
    parsed["fixed"] = parsed["fixed"].replace(docstring, docstring + "\n    PERCENTS = (0, 10, 50)\n")
    parsed["instruction"] += ("\n4）apply_code(code, percent)：percent 是 Cart.PERCENTS 里的档位时正常生效；"
                              "不在 0 到 100 之间抛 ValueError。")
    parsed["tests"]["code"] = {"requirement": 4, "body": TEST_CODE}
    parsed["mutants"].append(
        {"requirement": 4, "old": "if not 0 <= percent <= 100:", "new": "if percent < 0:", "idea": "漏了上限"})
    return parsed


TEST_COUNT_WEAK = '''
def test_count():
    cart = Cart()
    cart.add('pear', 20)
    assert cart.subtotal() == 20
'''


def without_accumulation_test(parsed):
    """第 1 条要求的测试只加一次商品：第 1 个 mutant 改坏之后没有任何测试发现。"""
    parsed = copy.deepcopy(parsed)
    parsed["tests"]["count"]["body"] = TEST_COUNT_WEAK
    return parsed


def with_unsafe_code(parsed):
    """参考解里 import 了 subprocess、读了文件、睡了一秒；隐藏测试里有 eval 和 os.system。它们都没有被
    执行，所以判分照常通过：这样的写法只能靠静态扫描发现。"""
    parsed = copy.deepcopy(parsed)
    parsed["fixed"] = parsed["fixed"].replace("class Cart:\n", "import subprocess\nimport time\n\n\nclass Cart:\n")
    parsed["fixed"] = parsed["fixed"].replace(
        "    def summary(self):\n",
        "    def snapshot(self, path):\n"
        "        with open(path) as handle:\n"
        "            text = handle.read()\n"
        "        time.sleep(1)\n"
        "        return text\n"
        "\n"
        "    def summary(self):\n")
    parsed["fixed"] = parsed["fixed"].replace("        # 折扣百分比必须在 0 到 100 之间\n",
                                              "        # TODO：折扣百分比必须在 0 到 100 之间\n")
    parsed["tests"]["count"]["body"] += "\nimport os\n" + UNSAFE_HELPERS
    return parsed


def run_audit(parsed, tmp_path, **options):
    """在真沙箱里审计一次。默认只判一遍（判分慢），要查稳定性的测试自己传 repeats。"""
    return audit.audit_family(parsed, slug=SLUG, direction=DIRECTION,
                              work_dir=tmp_path / "audit", repeats=1, **options)


def codes(result):
    return [finding["code"] for finding in result["findings"]]


def finding(result, code):
    """code 那一类发现恰好一条时返回它。"""
    found = [item for item in result["findings"] if item["code"] == code]
    assert len(found) == 1, (code, result["findings"])
    return found[0]


def line_of(text, snippet):
    """snippet 第一次出现的那一行是第几行（从 1 起）。"""
    return next(number for number, line in enumerate(text.splitlines(), 1) if snippet in line)


# ── 规格构造（不判分）──────────────────────────────────────────────────

def test_specs_build_a_parent_and_one_variant_per_mutant():
    parsed = cart_family()
    family = specs.build_family(parsed, slug=SLUG, direction=DIRECTION)
    parent, first, second, third = specs.family_specs(family)
    assert family["mutants"] == [0, 1, 2]
    assert [spec["task_id"] for spec in (parent, first, second, third)] == [
        "s3_cart_001", "s3v_cart_1", "s3v_cart_2", "s3v_cart_3"]
    for spec in (parent, first, second, third):
        assert spec["source_group"] == "s3_cart" and spec["module"] == "s3_cart.py"
        assert spec["split"] == "train" and spec["difficulty"] == "L3"
        assert spec["bug_family"] == DIRECTION and spec["direction"] == DIRECTION
        assert spec["stratum"] == "synth-cart_rules" and spec["origin"] == "synth"
        assert spec["instruction"] == INSTRUCTION and spec["fixed"] == CART.strip() + "\n"
        for field in ("instruction", "fixed", "public_test", "private_tests", "stratum"):
            assert spec[field] == parent[field]
    assert "variant_of" not in parent
    assert [spec["variant_of"] for spec in (first, second, third)] == ["s3_cart_001"] * 3
    # 测试文件的开头同 hard_tasks_v3_common._task
    assert parent["public_test"] == "from s3_cart import *\n\n" + PUBLIC.strip() + "\n"
    assert parent["private_tests"] == {
        "test_count.py": "import pytest\nfrom s3_cart import *\n\n" + TEST_COUNT.strip() + "\n",
        "test_remove.py": "import pytest\nfrom s3_cart import *\n\n" + TEST_REMOVE.strip() + "\n",
        "test_total.py": "import pytest\nfrom s3_cart import *\n\n" + TEST_TOTAL.strip() + "\n"}
    # 变体 = 参考解上只换一处；母题 = 依次换完
    for spec, mutant in zip((first, second, third), parsed["mutants"]):
        assert spec["buggy"] == spec["fixed"].replace(mutant["old"], mutant["new"])
    expected = parent["fixed"]
    for mutant in parsed["mutants"]:
        expected = expected.replace(mutant["old"], mutant["new"])
    assert parent["buggy"] == expected
    assert specs.build_variant(parsed, 1, slug=SLUG, direction=DIRECTION) == second
    assert specs.build_parent(parsed, [0, 1, 2], slug=SLUG, direction=DIRECTION) == parent
    validate_specs(specs.family_specs(family))


def test_only_the_given_mutants_are_used_and_variants_are_numbered_from_one():
    parsed = cart_family()
    family = specs.build_family(parsed, slug=SLUG, direction=DIRECTION, indices=[2, 0])
    assert family["mutants"] == [2, 0]
    assert [spec["task_id"] for spec in family["variants"]] == ["s3v_cart_1", "s3v_cart_2"]
    assert family["variants"][0]["buggy"] == family["variants"][0]["fixed"].replace(
        parsed["mutants"][2]["old"], parsed["mutants"][2]["new"])
    assert specs.build_family(parsed, slug=SLUG, direction=DIRECTION)["mutants"] == [0, 1, 2]


def test_a_variant_whose_anchor_is_not_unique_is_a_family_error():
    parsed = cart_family()
    parsed["mutants"][1]["old"] = "self._items[name]"        # 在参考解里出现 4 次
    with pytest.raises(specs.FamilyError, match=r"第 2 条要求的 mutant（第 2 个）的 old 片段在参考解里出现 4 次"):
        specs.build_family(parsed, slug=SLUG, direction=DIRECTION)
    assert specs.build_family(parsed, slug=SLUG, direction=DIRECTION, indices=[0, 2])["mutants"] == [0, 2]
    with pytest.raises(specs.FamilyError, match="没有第 5 个 mutant：题族里一共 3 个"):
        specs.build_variant(parsed, 4, slug=SLUG, direction=DIRECTION)


def test_the_parent_names_the_step_whose_anchor_stopped_being_unique():
    """两个 mutant 单独都能定位；前一个的改动让后一个的锚点多出来一份，这时要说明是第几步。"""
    parsed = cart_family()
    per_piece = "5 * sum(item['count'] for item in self._items.values())"
    parsed["mutants"] = [
        {"requirement": 3, "old": "discounted = self.subtotal() * (100 - self._discount) // 100",
         "new": "discounted = self.subtotal() * (100 - self._discount) // 100 or " + per_piece, "idea": "x"},
        {"requirement": 3, "old": per_piece, "new": "5 * len(self._items)", "idea": "y"},
    ]
    for index in (0, 1):
        specs.build_variant(parsed, index, slug=SLUG, direction=DIRECTION)
    with pytest.raises(specs.FamilyError, match="第 2 步、第 3 条要求的 mutant（第 2 个）的 old 片段在前面的 "
                                                "mutant 应用之后的代码里出现 2 次"):
        specs.build_parent(parsed, [0, 1], slug=SLUG, direction=DIRECTION)
    specs.build_parent(parsed, [1, 0], slug=SLUG, direction=DIRECTION)      # 换个顺序就放得下


def test_a_label_that_cannot_be_a_file_name_is_rejected():
    parsed = cart_family()
    parsed["tests"]["count extra"] = parsed["tests"].pop("count")
    with pytest.raises(specs.FamilyError, match="隐藏测试的标签 'count extra' 只能用字母、数字、下划线"):
        specs.build_family(parsed, slug=SLUG, direction=DIRECTION)


def test_fixed_hash_follows_the_syntax_tree_not_the_layout():
    parent = specs.build_family(cart_family(), slug=SLUG, direction=DIRECTION)["parent"]
    reworded = {**parent, "fixed": "# 同一个程序，换一种排版\n\n\n" + parent["fixed"].replace(
        "self._discount = 0", "self._discount =  0")}
    different = {**parent, "fixed": parent["fixed"].replace("self._discount = 0", "self._discount = 1")}
    assert specs.fixed_hash(reworded) == specs.fixed_hash(parent) != specs.fixed_hash(different)


def test_token_estimates_match_self_check_pool():
    import self_check_pool

    parent = specs.build_family(cart_family(), slug=SLUG, direction=DIRECTION)["parent"]
    for text in ("修复 Cart: add(name, price) 12345", INSTRUCTION, CART):
        assert specs.rough_tokens(text) == self_check_pool.rough_tokens(text)
    assert specs.prompt_estimate(parent) == self_check_pool.prompt_estimate(parent)


def test_every_built_in_pool_is_checked_for_collisions():
    pools = dict(audit.existing_pools([{"task_id": "extra_001"}]))
    assert list(pools) == ["builtin", "pilot", "v2", "v2b", "v2c", "v2d", "h3", "h3s", "extra"]
    assert sum(len(pool) for name, pool in pools.items() if name != "extra") == 341
    assert len(pools["extra"]) == 1 and pools["h3s"][0]["task_id"].startswith("h3v_")


# ── 静态检查和判读（不判分）────────────────────────────────────────────

@pytest.mark.parametrize("code, expected", [
    ("import subprocess as sp\n", ["第 1 行 import subprocess"]),
    ("import urllib.request\n", ["第 1 行 import urllib.request"]),
    ("from pathlib import Path\n", ["第 1 行 from pathlib import Path"]),
    ("import os as o\no.system('ls')\n", ["第 2 行 os.system() 的调用"]),
    ("from os import remove as rm\nrm('x')\n", ["第 2 行 os.remove() 的调用"]),
    ("import builtins\nbuiltins.eval('1')\n", ["第 2 行 eval() 的调用"]),
    ("value = compile('x', 'f', 'exec')\n", ["第 1 行 compile() 的调用"]),
    ("module = __import__('os')\nname = input()\n", ["第 1 行 __import__() 的调用", "第 2 行 input() 的调用"]),
    ("data = open('x')\n", ["第 1 行 open() 的调用"]),
    ("import io\nio.open('x')\n", ["第 2 行 open() 的调用"]),
    ("import time\ntime.sleep(1)\n", ["第 2 行 time.sleep(1)：等待时间必须是不超过 0.2 秒的字面量"]),
    ("import time\nfrom time import sleep\nsleep(delay)\n",
     ["第 3 行 time.sleep(delay)：等待时间必须是不超过 0.2 秒的字面量"]),
    ("import time\ntime.sleep(0.2 + extra)\n",
     ["第 2 行 time.sleep(0.2 + extra)：等待时间必须是不超过 0.2 秒的字面量"]),
])
def test_unsafe_uses_names_each_banned_construct_and_its_line(code, expected):
    assert audit.unsafe_uses(ast.parse(code)) == expected


def test_ordinary_code_passes_the_safety_scan():
    assert audit.unsafe_uses(ast.parse(
        "import os\n"
        "import time\n"
        "\n"
        "\n"
        "def wait():\n"
        "    time.sleep(0.2)\n"
        "\n"
        "\n"
        "def join(parts):\n"
        "    return os.path.join(*parts)\n"
        "\n"
        "\n"
        "class Drawer:\n"
        "    def open(self):\n"
        "        return self.handle.open()\n")) == []
    # offset 是测试文件开头那几行 import：报出来的行号要让教师对得上自己写的 body
    assert audit.unsafe_uses(ast.parse("import pytest\nfrom m import *\n\nimport os\nos.fork()\n"), 3) == [
        "第 2 行 os.fork() 的调用"]


def test_hint_markers_only_look_at_comments():
    # 规则同 tests/test_tasks_h3.py：只看井号后面，不区分大小写；变量名里有 todo 不算
    body = "# todo: fix later\ncode = 1  # XXX\ntodo_count = 0  # 处理待办项\n# a hack day\nassert todo_count == 0\n"
    assert audit.hint_markers("import pytest\nfrom m import *\n\n" + body, 3) == [
        (1, "TODO", "# todo: fix later"), (2, "XXX", "code = 1  # XXX"), (4, "HACK", "# a hack day")]
    assert audit.hint_markers("todo_count = 1\nvalue = 'TODO'\n") == []


def graded(**changes):
    """一份判分结果的最小形状（同 grade_program 的返回）。"""
    return {"verdict": "partial", "strict_pass": False, "public_pass": True, "private_pass": False,
            "public": {"summary": {"passed": 1}, "non_passing_tests": [], "exit_status": "ok"},
            "private": {"summary": {"failed": 1, "passed": 3}, "non_passing_tests": ["t::a"],
                        "exit_status": "nonzero_exit"},
            **changes}


PASSED = {"summary": {"passed": 4}, "non_passing_tests": [], "exit_status": "ok"}


# (判分结果, _rejection 的结论, _diagnosis 取到的原因)
CASES = [
    (graded(strict_pass=True, verdict="strict_pass", private_pass=True, private=PASSED),
     (False, "escaped", []), "escaped"),
    (graded(), (True, "", ["t::a"]), ""),                  # 有测试失败时 pytest 本来就以非 0 退出
    (graded(private={"summary": {"errors": 1}, "non_passing_tests": ["tests_private.test_x"], "exit_status": "ok"}),
     (False, "errored", []), "errored"),
    (graded(private={"summary": {"skipped": 2, "passed": 2}, "non_passing_tests": ["t::b"], "exit_status": "ok"}),
     (False, "unfailed", []), "unfailed"),
    (graded(private={"summary": {"failed": 1, "skipped": 1, "passed": 2}, "non_passing_tests": ["t::a", "t::b"],
                     "exit_status": "nonzero_exit"}), (False, "unfailed", []), "unfailed"),   # 失败里掺着跳过
    (graded(private=None), (False, "unfailed", []), "unfailed"),
    (graded(public_pass=False, private_pass=None, private=None), (False, "escaped", []), "public"),
    (graded(public_pass=None, private_pass=None, private=None, error_kind="timeout"),
     (False, "timeout", []), "timeout"),
    (graded(public_pass=None, private_pass=None, private=None, error_kind="sandbox"),
     (False, "sandbox", []), "ungraded"),
    (graded(verdict="invalid", public_pass=None, private_pass=None, public=None, private=None,
            blocked={"patch": {"apply_failed": True}}), (False, "ungraded", []), "ungraded"),   # 测试之前就被拒绝
]


@pytest.mark.parametrize("result, expected, diagnosis", CASES)
def test_only_a_failing_hidden_assertion_counts_as_caught(result, expected, diagnosis):
    assert audit._rejection(result) == expected
    assert audit._diagnosis(result) == diagnosis


def test_bad_arguments_are_rejected_up_front(tmp_path):
    work_dir = tmp_path / "audit"
    with pytest.raises(ValueError, match="repeats"):
        audit.audit_family(cart_family(), slug=SLUG, direction=DIRECTION, work_dir=work_dir, repeats=0)
    with pytest.raises(ValueError, match="direction"):
        audit.audit_family(cart_family(), slug=SLUG, direction=" ", work_dir=work_dir)
    with pytest.raises(TypeError, match="parse_family"):
        audit.audit_family([], slug=SLUG, direction=DIRECTION, work_dir=work_dir)
    with pytest.raises(TypeError, match="extra_specs"):
        audit.audit_family(cart_family(), slug=SLUG, direction=DIRECTION, work_dir=work_dir, extra_specs=["x"])
    assert not work_dir.exists()                            # 参数不对就不建工作区


def test_extra_specs_can_be_any_iterable_and_are_compared_in_full(tmp_path):
    """extra_specs 可以是生成器：读一遍就落成列表，撞车的比对要拿它跑完整个结构检查。"""
    parsed = cart_family()
    parsed["public_test"] = "def broken(:\n"                 # 只看结构，不去判分
    variant = specs.build_variant(parsed, 0, slug=SLUG, direction=DIRECTION)
    result = run_audit(parsed, tmp_path, extra_specs=iter([variant]))
    assert codes(result).count("collision") == 4            # task_id、source_group、module、原程序
    assert result["stats"]["checks"]["structure"] is False and not (tmp_path / "audit" / "tasks").exists()


@pytest.mark.parametrize("break_it, code, piece", [
    (lambda parsed: parsed["tests"]["count"].pop("body"), "invalid_family",
     "题族的格式不对，构造不出题目（KeyError: 'body'）"),
    (lambda parsed: parsed.update(mutants={"count": {}}), "invalid_family", "题族必须有 mutant 列表"),
    (lambda parsed: parsed["tests"].clear(), "invalid_family", "题族里没有任何隐藏测试"),
    (lambda parsed: parsed.update(fixed="def broken(:\n"), "syntax_error", "参考解不是合法的 Python（第 1 行"),
    (lambda parsed: parsed["tests"]["count"].update(body="def test_count(:\n"), "syntax_error",
                                                     "隐藏测试 test_count.py：不是合法的 Python（第 1 行"),
    (lambda parsed: parsed.update(public_test="    def indented():\n        pass\n\n    x = (\n"), "syntax_error",
     "公开测试：不是合法的 Python（第 4 行"),               # 行号按去掉公共缩进之后的文本算
])
def test_a_family_that_cannot_be_built_is_a_finding_not_a_crash(tmp_path, break_it, code, piece):
    parsed = cart_family()
    break_it(parsed)
    result = run_audit(parsed, tmp_path)
    assert result["accepted"] is False and result["family"] is None and result["usable"] == []
    assert piece in finding(result, code)["message"] and finding(result, code)["repairable"] is True
    assert not (tmp_path / "audit" / "tasks").exists()        # 题目都构造不出来，不会去判分


def test_a_bad_slug_cannot_be_repaired_by_rewriting_the_family(tmp_path):
    """大写的 slug 仍能通过 validate_specs（它只要求是安全的目录名），所以这一条只能靠审计自己来查。"""
    result = audit.audit_family(cart_family(), slug="Cart", direction=DIRECTION, work_dir=tmp_path / "audit")
    assert codes(result) == ["bad_slug"] and result["accepted"] is False and result["family"] is None
    slug = finding(result, "bad_slug")
    assert slug["repairable"] is False
    assert slug["message"] == "slug 'Cart' 不合规：只能用小写字母、数字、下划线。换一个 slug 再审计。"
    assert result["stats"]["checks"]["structure"] is False and result["usable"] == [0, 1, 2]   # 其余各项照常查完


# ── 执行审计：在真沙箱里判分 ───────────────────────────────────────────

def test_a_sound_family_is_accepted(tmp_path):
    result = run_audit(cart_family(), tmp_path)
    assert result["accepted"] is True, result["findings"]
    assert result["findings"] == [] and result["usable"] == [0, 1, 2]
    family = result["family"]
    assert family == specs.build_family(cart_family(), slug=SLUG, direction=DIRECTION)
    parent, *variants = specs.family_specs(family)
    assert [spec["task_id"] for spec in (parent, *variants)] == [
        "s3_cart_001", "s3v_cart_1", "s3v_cart_2", "s3v_cart_3"]
    assert all(spec["variant_of"] == parent["task_id"] for spec in variants)
    assert (tmp_path / "audit" / "tasks" / "s3_cart_001" / "fixture.tar").exists()
    stats = result["stats"]
    assert stats["mutants"] == stats["usable"] == 3
    assert stats["checks"] == dict.fromkeys(audit.CHECKS, True)
    # 每个变体只挂自己那条要求的测试；母题三处缺陷都暴露出来
    assert [(record["index"], record["failed_tests"]) for record in stats["variants"]] == [
        (0, [COUNT]), (1, [REMOVE]), (2, [DISCOUNT])]
    assert stats["parent"][0]["private_failed"] == sorted([COUNT, DISCOUNT, REMOVE])
    assert stats["parent"][0]["public_pass"] is True and stats["reference"][0]["strict_pass"] is True
    assert stats["pairs"] == 3 and stats["distinct_pairs"] == 3
    assert 30 <= stats["fixed_lines"] <= 70 and stats["total_tokens"] <= audit.LONG_ESTIMATE
    assert stats["isolation"] in ("landlock", "process") and stats["fixed_hash"] == specs.fixed_hash(parent)
    assert json.loads(json.dumps(result, ensure_ascii=False)) == result      # 报告里原样存得下


def test_a_mutant_that_no_hidden_test_catches_drops_out_but_the_family_can_still_pass(tmp_path):
    """四个 mutant 里有一个没被拦住：可用的还够 3 个，题族被接受，但带着一条可返修的发现。"""
    parsed = without_accumulation_test(with_code_rule(cart_family()))
    result = run_audit(parsed, tmp_path)
    assert result["accepted"] is True, result["findings"]
    assert result["usable"] == [1, 2, 3] and codes(result) == ["mutant_escapes"]
    escaped = finding(result, "mutant_escapes")
    assert escaped["repairable"] is True
    assert escaped["message"] == (
        "第 1 条要求的 mutant（第 1 个）没有被任何隐藏测试拦住：把 self._items[name]['count'] += count 改成 "
        "self._items[name]['count'] = count 之后公开测试和隐藏测试全部通过，请给这条要求补一个能暴露它的隐藏测试。")
    assert result["family"] == specs.build_family(parsed, slug=SLUG, direction=DIRECTION, indices=[1, 2, 3])
    parent, first, *_ = specs.family_specs(result["family"])
    # 题族只用可用的三个 mutant 构造：母题里没有第 1 条要求的缺陷，变体从 1 起连着编号
    assert parent["task_id"] == "s3_cart_001" and first["task_id"] == "s3v_cart_1"
    assert first["buggy"] == first["fixed"].replace(parsed["mutants"][1]["old"], parsed["mutants"][1]["new"])
    assert result["stats"]["checks"]["variants"] is False and result["stats"]["usable"] == 3


def test_too_few_usable_mutants_means_no_parent(tmp_path):
    """一个 mutant 逃过全部测试，另一个让隐藏测试在收集阶段报错：都不算被拦住，可用的不够 3 个。"""
    parsed = without_accumulation_test(with_code_rule(cart_family()))
    parsed["mutants"][3] = {"requirement": 4, "old": "PERCENTS = (0, 10, 50)", "new": "PERCENT = (0, 10, 50)",
                            "idea": "属性名写错"}
    result = run_audit(parsed, tmp_path)
    assert result["accepted"] is False and result["family"] is None and result["usable"] == [1, 2]
    errored = finding(result, "mutant_not_failed")
    assert re.fullmatch(
        r"第 4 条要求的 mutant（第 4 个）没有被断言拦住：把 PERCENTS = \(0, 10, 50\) 改成 PERCENT = \(0, 10, 50\) "
        r"之后隐藏测试报错（\d+ 个 error），没有断言失败。多半是改坏后模块导入或测试准备阶段就抛了异常，"
        r"请让这处疏忽表现为错误的结果，并被断言抓住。", errored["message"]), errored["message"]
    too_few = finding(result, "too_few_mutants")
    assert too_few["message"] == (
        "可用的 mutant 只有 2 个，不到 3 个，母题不成立（可用的是第 2 个、第 3 个；不可用的是第 1 个（全部测试"
        "仍然通过）；第 4 个（隐藏测试报错））。请先按上面的说明修好不可用的 mutant，或者补新的。")
    assert codes(result) == ["mutant_escapes", "mutant_not_failed", "too_few_mutants"]
    assert "parent" not in result["stats"] and result["stats"]["checks"]["parent"] is False


def test_a_variant_that_breaks_the_public_test_is_unusable(tmp_path):
    parsed = cart_family()
    parsed["public_test"] = PUBLIC_ADDS_TWICE
    result = run_audit(parsed, tmp_path)
    assert result["accepted"] is False and result["usable"] == [1, 2]
    breaks = finding(result, "mutant_breaks_public")
    assert breaks["message"] == (
        f"第 1 条要求的 mutant（第 1 个）让公开测试失败（没通过的公开测试：{PUBLIC_TEST}）：把 "
        "self._items[name]['count'] += count 改成 self._items[name]['count'] = count 之后公开测试没能全部通过。"
        + audit.MUTANT_PUBLIC_ADVICE)
    assert codes(result) == ["mutant_breaks_public", "too_few_mutants"]


def test_a_parent_that_breaks_the_public_test_is_rejected(tmp_path):
    """各处缺陷单独出现时都通过公开测试，合在一起却破坏了它：remove 留下 0 件的条目，运费按条目数收。"""
    parsed = cart_family()
    parsed["instruction"] = "".join(line + "\n" for line in INSTRUCTION.splitlines() if not line.startswith("3）")) + (
        "3）shipping() 按件数收运费：每件 5，商品金额满 100 免运费；total() = 商品金额加运费。")
    parsed["public_test"] = PUBLIC_AFTER_REMOVE
    parsed["tests"]["total"] = {"requirement": 3, "body": TEST_SHIPPING}
    parsed["mutants"][2] = {"requirement": 3, "old": "return 5 * sum(item['count'] for item in self._items.values())",
                            "new": "return 5 * len(self._items)", "idea": "按条目数而不是件数收运费"}
    result = run_audit(parsed, tmp_path)
    assert result["accepted"] is False and result["family"] is None
    assert result["usable"] == [0, 1, 2]                       # 三个变体单看都没问题
    breaks = finding(result, "parent_breaks_public")
    assert breaks["message"] == (
        f"母题（同时带着 3 处缺陷）让公开测试失败（没通过的公开测试：{PUBLIC_TEST}）：公开测试没能全部通过。"
        f"{audit.PARENT_PUBLIC_ADVICE}")
    assert codes(result) == ["parent_breaks_public"]
    assert result["stats"]["checks"]["parent"] is False and result["stats"]["checks"]["variants"] is True


def test_a_reference_that_fails_its_own_tests_is_spelled_out_and_the_rest_is_still_checked(tmp_path):
    parsed = cart_family()
    parsed["tests"]["count"]["body"] = TEST_COUNT.replace("== 60", "== 61")
    result = run_audit(parsed, tmp_path)
    fails = finding(result, "reference_fails")
    assert fails["message"].startswith(
        f"参考解没有通过全部测试：隐藏测试没有通过（没通过：{COUNT}）。pytest 输出的末尾：\n")
    assert "assert 60 == 61" in fails["message"] and "tests_private/test_count.py:" in fails["message"]
    assert str(tmp_path) not in fails["message"]              # 工作区路径对返修没有用
    assert fails["message"].endswith("请让参考解满足题面的每一条要求，并核对测试断言的期望值。")
    assert result["accepted"] is False and result["stats"]["checks"]["reference"] is False
    assert [record["index"] for record in result["stats"]["variants"]] == [0, 1, 2]   # 后面的检查照常做完
    assert "parent" in result["stats"]


def test_a_reference_with_flaky_tests_is_rejected(tmp_path, monkeypatch):
    real, calls = audit.grade_program, []

    def flaky(task, spec, source, work, timeout=None):
        result = real(task, spec, source, work, timeout)
        calls.append(source)
        if len(calls) == 2:              # 第二次判参考解时，一个隐藏测试偶尔失败
            return {**result, "verdict": "partial", "strict_pass": False, "private_pass": False,
                    "private": {**result["private"], "passed": False, "summary": {"passed": 4, "failed": 1},
                                "non_passing_tests": [DISCOUNT]}}
        return result

    monkeypatch.setattr(audit, "grade_program", flaky)
    result = audit.audit_family(cart_family(), slug=SLUG, direction=DIRECTION,
                                work_dir=tmp_path / "audit", repeats=2)
    unstable = finding(result, "unstable_tests")
    assert unstable["message"] == (
        f"参考解重复 2 次结果不一致（第 1 次通过；第 2 次没通过（没通过的测试：{DISCOUNT}））：测试不稳定，"
        "可能依赖时间、随机数或执行顺序，请改成确定性的实现和测试。")
    assert "参考解没有通过全部测试" in finding(result, "reference_fails")["message"]
    assert result["accepted"] is False and result["family"] is None
    assert result["stats"]["checks"]["reference"] is False and result["usable"] == [0, 1, 2]
    assert len(calls) == 2 + 3 + 2        # 参考解两遍、三个变体、母题两遍：发现不稳定后没有就此停手


def test_a_mutant_whose_anchor_is_not_unique_is_left_out(tmp_path):
    parsed = cart_family()
    parsed["mutants"][1]["old"] = "self._items[name]"
    result = run_audit(parsed, tmp_path)
    anchor = finding(result, "mutant_anchor")
    assert anchor["repairable"] is True
    assert anchor["message"] == (
        "第 2 条要求的 mutant（第 2 个）的 old 片段在参考解里出现 4 次（必须恰好 1 次）：'self._items[name]'。"
        "请把 old 改成能唯一定位的那几行，new 写成改坏后的样子。")
    assert result["usable"] == [0, 2] and result["accepted"] is False
    assert "可用的 mutant 只有 2 个" in finding(result, "too_few_mutants")["message"]


def test_a_mutant_that_breaks_the_syntax_is_left_out(tmp_path):
    parsed = cart_family()
    parsed["mutants"][1]["new"] = "if item['count'] < 0"      # 少了冒号
    result = run_audit(parsed, tmp_path)
    message = finding(result, "mutant_syntax")["message"]
    # 语法错误的措辞和字符串的引号随 Python 版本变化，这里只认行号和片段
    assert re.fullmatch(
        rf"第 2 条要求的 mutant（第 2 个）把代码改坏之后不是合法的 Python（第 17 行：[^）]*）："
        rf"new 片段是 {re.escape(repr(parsed['mutants'][1]['new']))}。改坏后的程序必须还能运行，只是行为不对。",
        message), message
    assert result["usable"] == [0, 2] and result["accepted"] is False


def test_unsafe_constructs_and_hint_comments_are_reported_with_line_numbers(tmp_path):
    parsed = with_unsafe_code(cart_family())
    parsed["mutants"][1]["new"] += "  # BUG 先这样"           # mutant 的 new 片段带进来的标记会出现在变体里
    result = run_audit(parsed, tmp_path)
    fixed, body = specs.source(parsed["fixed"]), specs.source(parsed["tests"]["count"]["body"])
    unsafe = [item["message"] for item in result["findings"] if item["code"] == "unsafe_code"]
    assert unsafe == [
        f"参考解：有 3 处不允许的写法：第 {line_of(fixed, 'import subprocess')} 行 import subprocess；"
        f"第 {line_of(fixed, 'open(path)')} 行 open() 的调用；"
        f"第 {line_of(fixed, 'time.sleep(1)')} 行 time.sleep(1)：等待时间必须是不超过 0.2 秒的字面量。"
        f"{audit.UNSAFE_ADVICE}。",
        f"隐藏测试 test_count.py：有 2 处不允许的写法：第 {line_of(body, 'eval(name)')} 行 eval() 的调用；"
        f"第 {line_of(body, 'os.system')} 行 os.system() 的调用。{audit.UNSAFE_ADVICE}。"]
    hints = [item["message"] for item in result["findings"] if item["code"] == "hint_comment"]
    assert hints == [
        f"参考解：注释里有提示缺陷的标记：第 {line_of(fixed, 'TODO')} 行 TODO。"
        "这些标记会告诉读者缺陷在哪，请删掉或改写。",
        f"隐藏测试 test_count.py：注释里有提示缺陷的标记：第 {line_of(body, 'FIXME')} 行 FIXME。"
        "这些标记会告诉读者缺陷在哪，请删掉或改写。",
        # 参考解里的 TODO 也会出现在每个变体的原程序里，但不重复报；只报 mutant 自己带进来的
        f"变体 2 的原程序：注释里有提示缺陷的标记：第 {line_of(fixed, 'if item')} 行 BUG。"
        "这些标记会告诉读者缺陷在哪，请删掉或改写。"]
    assert codes(result) == ["unsafe_code", "unsafe_code", "hint_comment", "hint_comment", "hint_comment"]
    assert result["accepted"] is False
    assert result["stats"]["checks"]["safety"] is False and result["stats"]["checks"]["no_hints"] is False
    assert result["stats"]["checks"]["reference"] is True and result["usable"] == [0, 1, 2]   # 判分照常通过


def test_a_family_that_is_too_long_is_flagged(tmp_path):
    parsed = cart_family()
    padding = "\n".join(f"# 备用说明第 {number} 行" for number in range(40))
    parsed["fixed"] = parsed["fixed"].replace("class Cart:", padding + "\n\n\nclass Cart:")
    parsed["instruction"] += "\n背景：" + "这是一段用来把题面撑长的说明文字。" * 40
    result = run_audit(parsed, tmp_path)
    stats = result["stats"]
    assert stats["fixed_lines"] == len(specs.source(parsed["fixed"]).splitlines()) > 70
    assert finding(result, "fixed_lines")["message"] == (
        f"参考解有 {stats['fixed_lines']} 行，应在 30 到 70 行之间：请拆出一部分逻辑，或者删掉与要求无关的内容。")
    assert stats["total_tokens"] > audit.LONG_ESTIMATE
    assert finding(result, "too_long_tokens")["message"] == (
        f"题面加参考解的估算长度是 {stats['total_tokens']} 个 token（题面 {stats['prompt_tokens']} + 参考解 "
        f"{stats['answer_tokens']}），超过上限 {audit.LONG_ESTIMATE}：训练样本超长会被整条丢弃。请精简题面、"
        f"缩短参考解或公开测试。")
    assert result["accepted"] is False and result["stats"]["checks"]["length"] is False
    assert result["usable"] == [0, 1, 2] and result["stats"]["checks"]["reference"] is True


def test_hidden_tests_that_cannot_tell_requirements_apart_are_a_repairable_finding(tmp_path):
    parsed = cart_family()
    parsed["tests"] = {f"flow_{letter}": {"requirement": number, "body": FLOW.replace("flow", f"flow_{letter}")}
                       for number, letter in enumerate("abc", 1)}
    result = run_audit(parsed, tmp_path)
    shared = [f"tests_private.test_flow_{letter}::test_cart_flow_{letter}" for letter in "abc"]
    assert [record["failed_tests"] for record in result["stats"]["variants"]] == [shared] * 3
    indistinct = finding(result, "indistinct_tests")
    assert indistinct["repairable"] is True
    assert indistinct["message"] == (
        f"隐藏测试区分不了各条要求：3 个可用的变体没通过的隐藏测试完全一样（{'、'.join(shared)}），每个 mutant "
        "都让同一批测试失败。请给每条要求写一个专门的测试函数，只断言那条要求的行为，让改坏不同要求的变体挂"
        "不同的测试。")
    # 每个变体都被拦住、母题也成立：按接受的条件这个题族仍然接受，这一项只记发现
    assert result["accepted"] is True and result["usable"] == [0, 1, 2] and codes(result) == ["indistinct_tests"]
    assert result["stats"]["checks"]["tests_distinguish"] is False and result["stats"]["distinct_pairs"] == 0


def test_tests_that_only_partly_tell_requirements_apart_are_flagged_when_alike_pairs_are_not_the_minority(tmp_path):
    """四个可用的变体：前三个挂同一批测试，第四个挂另一个。六对里有三对相同，不同的正好一半，不算占多数。"""
    parsed = with_code_rule(cart_family())
    parsed["tests"] = {
        **{f"flow_{letter}": {"requirement": number, "body": FLOW.replace("flow", f"flow_{letter}")}
           for number, letter in enumerate("abc", 1)},
        # 第 4 条要求的测试改用空购物车（折扣和运费都不起作用），第 3 个 mutant 在那里就看不出来
        "code": {"requirement": 4, "body": TEST_CODE.replace("    cart.add('pear', 20)\n", "").replace(
            "    assert cart.total() == 20 * (100 - percent) // 100 + 5\n", "    assert cart.total() == 0\n")}}
    result = run_audit(parsed, tmp_path)
    assert result["usable"] == [0, 1, 2, 3]
    assert result["stats"]["pairs"] == 6 and result["stats"]["distinct_pairs"] == 3
    assert finding(result, "indistinct_tests")["message"] == (
        "隐藏测试区分不了各条要求：4 个可用的变体里有 3 对（第 1 个和第 2 个、第 1 个和第 3 个、第 2 个和第 3 个）"
        "没通过的隐藏测试完全一样，不同的只占少数。请给这些要求各补一个专门的测试函数，只断言那条要求的行为。")
    assert result["accepted"] is True and result["stats"]["checks"]["tests_distinguish"] is False


def test_collisions_with_other_pools_cannot_be_repaired_by_rewriting_the_family(tmp_path):
    family = specs.build_family(cart_family(), slug=SLUG, direction=DIRECTION)
    parent, first, *_ = specs.family_specs(family)
    digest = specs.fixed_hash(parent)
    fresh = {"module": "old_mod.py", "buggy": "value = 1\n", "fixed": "value = 2\n"}
    result = run_audit(cart_family(), tmp_path, known_fixed_hashes=[digest], extra_specs=[
        {**parent, "source_group": "old_group", **fresh},                         # 只撞 task_id
        {**parent, "task_id": "old_mod_001", "source_group": "old_group", "buggy": fresh["buggy"],
         "fixed": fresh["fixed"]},                                                # 只撞 module
        {**parent, "task_id": "old_group_001", **fresh},                          # 只撞 source_group
        {**first, "task_id": "old_prog_001", "source_group": "old_group2", "module": "old_mod3.py"},  # 撞原程序
    ])
    collisions = [item for item in result["findings"] if item["code"] == "collision"]
    assert [item["message"] for item in collisions] == [
        "母题的 task_id s3_cart_001 和 extra 题池的 s3_cart_001 重复：换一个 slug 才行。",
        "source_group s3_cart 和 extra 题池的 old_group_001 重复：换一个 slug 才行。",
        "module s3_cart.py 和 extra 题池的 old_mod_001 重复：换一个 slug 才行。",
        "变体 s3v_cart_1 的原程序和 extra 题池的 old_prog_001 的原程序语法树相同：这不是一道新题，请换一个程序。",
        f"参考解和已知题池里的一道题语法树相同（fixed_hash={digest[:12]}…）：请换一个程序，而不只是换写法。",
    ]
    assert all(item["repairable"] is False for item in collisions)
    assert result["accepted"] is False and result["family"] is None and result["usable"] == [0, 1, 2]
    assert result["stats"]["checks"]["structure"] is False and result["stats"]["fixed_hash"] == digest


def test_a_mutant_that_hangs_is_not_counted_as_caught(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "MUTANT_TIMEOUT", 2)
    parsed = cart_family()
    parsed["mutants"][2] = {"requirement": 3, "old": "return discounted + self.shipping()",
                            "new": "while True:\n            pass", "idea": "死循环"}
    result = run_audit(parsed, tmp_path)
    hangs = finding(result, "mutant_timeout")
    assert hangs["message"] == (
        "第 3 条要求的 mutant（第 3 个）让测试超时（上限 2 秒），没有任何断言失败：把 return discounted + "
        "self.shipping() 改成 while True: pass 之后测试跑不完。只是让程序变慢的缺陷在正式判分里会被判为通过，"
        "请改成能被断言抓住的疏忽。")
    assert result["usable"] == [0, 1] and result["accepted"] is False
    assert result["stats"]["variants"][2]["why"] == "timeout"
    assert "第 3 个（测试超时）" in finding(result, "too_few_mutants")["message"]


def test_duplicate_and_overlapping_mutants_are_left_out(tmp_path):
    """第 2 个 mutant 和第 1 个改坏后的程序完全相同；第 4 个的锚点会被第 3 个的改动破坏。都进不了题族。"""
    parsed = with_code_rule(cart_family())
    parsed["mutants"][1] = {**parsed["mutants"][0], "requirement": 2}
    per_piece = "5 * sum(item['count'] for item in self._items.values())"
    # 第 3 个 mutant 自己是合格的：不打折时照常，打了折折扣翻倍，结果归零之后落到 or 后面那一项
    parsed["mutants"][2] = {"requirement": 3, "old": "discounted = self.subtotal() * (100 - self._discount) // 100",
                            "new": "discounted = self.subtotal() * (100 - self._discount * 2) // 100 or " + per_piece,
                            "idea": "折扣翻倍"}
    parsed["mutants"][3] = {"requirement": 4, "old": per_piece, "new": "5 * len(self._items)", "idea": "按条目数收费"}
    result = run_audit(parsed, tmp_path)
    assert finding(result, "mutant_duplicate")["message"] == (
        "第 2 条要求的 mutant（第 2 个）改坏后的程序和第 1 个 mutant 的一模一样，两道变体会是同一道题："
        "请换一种不同的疏忽。")
    assert finding(result, "mutant_overlap")["message"] == (
        f"第 3 步、第 4 条要求的 mutant（第 4 个）的 old 片段在前面的 mutant 应用之后的代码里出现 2 次"
        f"（必须恰好 1 次）：{repr(per_piece)}。它和前面的 mutant 改动了重叠的位置，不能同时放进母题；"
        "请让各个 mutant 改动互不重叠的地方。")
    assert result["usable"] == [0, 2] and result["accepted"] is False
    assert codes(result) == ["mutant_duplicate", "mutant_overlap", "too_few_mutants"]
    # 可用的不够时，结构上排除的和判分不合格的一起列出来
    assert finding(result, "too_few_mutants")["message"] == (
        "可用的 mutant 只有 2 个，不到 3 个，母题不成立（可用的是第 1 个、第 3 个；不可用的是第 2 个（和别的 "
        "mutant 改坏后的程序相同）；第 4 个（和别的 mutant 改动重叠））。请先按上面的说明修好不可用的 mutant，"
        "或者补新的。")


def test_a_parent_whose_defects_cancel_out_is_not_caught(tmp_path):
    """四处缺陷两两抵消：单独改坏任何一处，求和的隐藏测试都会失败；四处一起改坏，两个和又都对上了。
    每个变体都可用，母题却畅通无阻：这样的题族不能要。"""
    parsed = {
        "theme": "预算拆分",
        "instruction": "修复 split(budget)：把预算分成食宿两项，再把其中一项分成出行和储蓄。要求：\n"
                       "1）food 与 rent 之和等于 budget。\n"
                       "2）travel 与 savings 之和等于 rent。",
        "fixed": '''
def split(budget):
    food = budget // 2
    rent = budget - budget // 2
    travel = rent // 4
    savings = rent - rent // 4
    return {'food': food, 'rent': rent, 'travel': travel, 'savings': savings}
''',
        "public_test": ("def test_public_split():\n"
                        "    assert sorted(split(100)) == ['food', 'rent', 'savings', 'travel']\n"),
        "tests": {
            "staple": {"requirement": 1, "body": "def test_staple():\n    parts = split(100)\n"
                                                 "    assert parts['food'] + parts['rent'] == 100\n"},
            "trip": {"requirement": 2, "body": "def test_trip():\n    parts = split(100)\n"
                                               "    assert parts['travel'] + parts['savings'] == parts['rent']\n"}},
        "mutants": [
            {"requirement": 1, "old": "food = budget // 2", "new": "food = budget // 2 + 1", "idea": "多算了一块"},
            {"requirement": 1, "old": "rent = budget - budget // 2", "new": "rent = budget - budget // 2 - 1",
             "idea": "少算了一块"},
            {"requirement": 2, "old": "travel = rent // 4", "new": "travel = rent // 4 + 1", "idea": "多算了一块"},
            {"requirement": 2, "old": "savings = rent - rent // 4", "new": "savings = rent - rent // 4 - 1",
             "idea": "少算了一块"},
        ]}
    result = run_audit(parsed, tmp_path)
    assert result["usable"] == [0, 1, 2, 3] and result["accepted"] is False and result["family"] is None
    assert finding(result, "parent_escapes")["message"] == (
        "母题（同时带着 4 处缺陷）没有被任何隐藏测试拦住：公开测试和隐藏测试全部通过，"
        "请给每条要求都补上能暴露它的隐藏测试。")
    assert result["stats"]["parent"][0]["strict_pass"] is True          # 四处一起改坏，测试全都过
    assert result["stats"]["checks"]["parent"] is False and result["stats"]["checks"]["variants"] is True


@pytest.mark.parametrize("error, code, repairable, piece, gradings", [
    (subprocess.TimeoutExpired("pytest", 5), "reference_timeout", True, "参考解在 5 秒内没跑完测试", 1),
    (GradeInfrastructureError("reference solution produced no test results in tests_public"), "grading_error", False,
     "error: GradeInfrastructureError: reference solution produced no test results in tests_public", 2),
])
def test_a_reference_that_cannot_be_graded_stops_grading_but_not_the_audit(
        tmp_path, monkeypatch, error, code, repairable, piece, gradings):
    """判分环境自己就坏着（什么题都判不完）：参考解判不完之后拿一道完好的小题自检一次，自检也不过才是
    不可返修的 grading_error；变体和母题不再一个一个地去撞，不靠判分的几项照常查完。"""
    calls = []
    monkeypatch.setattr(audit, "_PROBE_PASSED", set())       # 不受别的测试留下的自检结果影响

    def broken(task, spec, source, work, timeout=None):
        calls.append(source)
        raise error

    monkeypatch.setattr(audit, "grade_program", broken)
    result = run_audit(cart_family(), tmp_path)
    assert len(calls) == gradings           # 超时：只有参考解那一次；判不完：再加一次环境自检
    report = finding(result, code)
    assert report["repairable"] is repairable and piece in report["message"]
    if code == "grading_error":
        assert report["message"].endswith(
            "。判分环境自检也没通过（GradeInfrastructureError: reference solution produced no test results in "
            "tests_public），请先检查沙箱环境（agentlab doctor），修好以后原样重试。")
    assert result["accepted"] is False and result["family"] is None and result["usable"] == []
    checks = result["stats"]["checks"]
    assert checks["reference"] is False and checks["variants"] is False and checks["parent"] is False
    assert checks["structure"] and checks["length"] and checks["safety"] and checks["no_hints"]   # 不用判分的照常查完
    assert result["stats"]["variants"] == [] and result["stats"]["isolation"] is None


# ── 评审后补的场景 ─────────────────────────────────────────────────────

def test_a_parent_whose_grades_disagree_is_rejected(tmp_path, monkeypatch):
    """母题两次都被拦住、公开测试都过，只是挂掉的隐藏测试不一样：测试不稳定，这样的题族不能接受。"""
    real, graded = audit.grade_program, []

    def flaky(task, spec, source, work, timeout=None):
        result = real(task, spec, source, work, timeout)
        if source == spec["buggy"]:                      # 带着缺陷判分的只有母题
            graded.append(source)
            if len(graded) == 2:                         # 第二次判母题时，少挂了一个隐藏测试
                failed = [name for name in result["private"]["non_passing_tests"] if name != REMOVE]
                result = {**result, "private": {**result["private"], "non_passing_tests": failed,
                                                "summary": {"failed": len(failed), "passed": 5}}}
        return result

    monkeypatch.setattr(audit, "grade_program", flaky)
    result = audit.audit_family(cart_family(), slug=SLUG, direction=DIRECTION,
                                work_dir=tmp_path / "audit", repeats=2)
    assert result["accepted"] is False and result["family"] is None and result["usable"] == [0, 1, 2]
    assert codes(result) == ["unstable_tests"]
    unstable = finding(result, "unstable_tests")
    assert unstable["repairable"] is True
    assert unstable["message"] == (
        f"母题重复 2 次结果不一致（第 1 次没通过（没通过的测试：{COUNT}、{REMOVE}、{DISCOUNT}）；第 2 次没通过"
        f"（没通过的测试：{COUNT}、{DISCOUNT}））：测试不稳定，可能依赖时间、随机数或执行顺序，请改成确定性的"
        f"实现和测试。")
    assert result["stats"]["checks"]["parent"] is False and result["stats"]["checks"]["reference"] is True
    assert [run["private_failed"] for run in result["stats"]["parent"]] == [
        [COUNT, REMOVE, DISCOUNT], [COUNT, DISCOUNT]]


PERSONAL_PATH = next(pattern for label, pattern in privacy_check.CONTENT_PATTERNS if label == "个人路径")


def personal(*parts):
    """运行时拼出一个像个人目录的绝对路径：这种路径不能当字面量写进源码。"""
    return "/".join(["", "home", "user" + "1", *parts])


def test_without_paths_strips_directories_but_keeps_test_ids():
    windows = "C:" + "\\" * 2 + "Users" + "\\" * 2 + "ann" + "\\" * 2 + "x" + "\\" * 2 + "cart.py"
    module = personal("audit", "reference", "ws", "grade-s3_cart_001-7", "ws-grade-s3_cart_001", "s3_cart.py")
    test = personal("audit", "reference", "ws", "grade-s3_cart_001-7", "ws-grade-s3_cart_001", "tests_private",
                    "test_count.py")
    shown = audit.without_paths("\n".join([
        module + ":41: FileNotFoundError: [Errno 2] No such file or directory: 'missing.txt'",
        test + ":9: assert 60 == 61",
        "FAILED tests_private/test_count.py::test_count - assert 60 == 61",
        "rootdir: " + personal("audit", "reference", "ws"),
        "IsADirectoryError: [Errno 21] Is a directory: '" + personal("audit") + "'",
        "ImportError: no module named " + windows,
        "assert 1/2 == 0.5 and n/a and http://127.0.0.1:9/v1/models and /usr/bin/env",
    ]))
    assert shown.splitlines() == [
        "s3_cart.py:41: FileNotFoundError: [Errno 2] No such file or directory: 'missing.txt'",
        "tests_private/test_count.py:9: assert 60 == 61",
        "FAILED tests_private/test_count.py::test_count - assert 60 == 61",
        "rootdir: ws",
        "IsADirectoryError: [Errno 21] Is a directory: 'audit'",
        "ImportError: no module named cart.py",
        "assert 1/2 == 0.5 and n/a and http://127.0.0.1:9/v1/models and env",
    ]
    # 目录名里有空格时正则只能一段一段地认：已知的工作区目录要先原样去掉
    spaces = "/".join(["", "tmp", "my dir", "audit"])
    assert audit.without_paths("no space left on " + spaces, roots=[spaces]) == "no space left on "


def test_paths_in_pytest_output_do_not_reach_the_result(tmp_path):
    """参考解的代码里抛异常时，pytest 那一行带的是主模块的绝对路径。工作区放在一个像个人目录的地方
    （home 目录下再套一层带数字的用户名，见 personal()）：结果里的每一段文字都不能把它带出去，文件名和
    行号要留着。"""
    parsed = cart_family()
    parsed["fixed"] = parsed["fixed"].replace(
        "    def summary(self):\n",
        "    def load(self, path):\n"
        "        with open(path) as handle:\n"
        "            return handle.read()\n"
        "\n"
        "    def summary(self):\n")
    parsed["tests"]["count"]["body"] += ("\n\ndef test_load():\n    cart = Cart()\n"
                                         "    assert cart.load('missing.txt') == ''\n")
    work_dir = tmp_path.joinpath("home", "user" + "1", "audit")
    result = audit.audit_family(parsed, slug=SLUG, direction=DIRECTION, work_dir=work_dir, repeats=1)
    line = line_of(specs.source(parsed["fixed"]), "with open(path)")
    fails = finding(result, "reference_fails")
    assert "隐藏测试没有通过（没通过：tests_private.test_count::test_load）" in fails["message"]
    assert (f"\ns3_cart.py:{line}: FileNotFoundError: [Errno 2] No such file or directory: 'missing.txt'"
            in fails["message"])
    report = json.dumps({"findings": result["findings"], "stats": result["stats"]}, ensure_ascii=False)
    assert re.search(PERSONAL_PATH, str(work_dir))                  # 这个工作区确实长得像个人路径
    assert not re.search(PERSONAL_PATH, report) and str(work_dir) not in report
    assert f"s3_cart.py:{line}:" in result["stats"]["reference"][0]["tail"]


def test_grading_errors_do_not_carry_paths_or_the_whole_command(tmp_path, monkeypatch):
    def broken(task, spec, source, work, timeout=None):
        raise GradeInfrastructureError("reference solution produced no test results in tests_public; if pytest could "
                                       f"not start: python: can't open file '{personal('venv', 'bin', 'pytest')}'")

    monkeypatch.setattr(audit, "grade_program", broken)
    report = finding(run_audit(cart_family(), tmp_path), "grading_error")
    assert report["repairable"] is False
    assert "produced no test results in tests_public" in report["message"]
    assert "can't open file 'pytest'" in report["message"] and personal("venv") not in report["message"]
    # 超时的异常文本是整条命令（解释器的绝对路径、沙箱的源码）：不进结果，只说多少秒没结束
    timed = audit._error_result(subprocess.TimeoutExpired([personal("venv", "bin", "python3"), "-c", "bootstrap"], 5))
    assert timed["verdict"] == "error: 测试在 5 秒内没有结束" and timed["error_kind"] == "timeout"


def test_collectable_tests_follows_pytest_defaults():
    """同 pytest 的默认规则（tests/test_synth_audit.py 之外用 pytest --collect-only 核对过）：模块层 test 开头的
    函数，Test 开头、没有 __init__ 的类里 test 开头的方法；x_test、普通类里的方法、带 __init__ 的类都不收。"""
    tree = ast.parse(textwrap.dedent('''
        import pytest


        @pytest.fixture
        def thing():
            return 1


        def helper(thing):
            return thing


        def x_test():
            pass


        def test_plain():
            pass


        class Helper:
            def test_inside(self):
                pass


        class TestGroup:
            def test_method(self):
                pass


        class TestWithInit:
            def __init__(self):
                pass

            def test_method(self):
                pass
    '''))
    assert audit.collectable_tests(tree) == ["test_plain", "test_method"]


ONLY_HELPER = "def check_cart():\n    cart = Cart()\n    assert cart.subtotal() == 0\n"
ONLY_FIXTURE = "import pytest\n\n\n@pytest.fixture\ndef cart():\n    return Cart()\n"


@pytest.mark.parametrize("where, body, declared", [
    ("public_test", ONLY_HELPER, "现在定义的只有 check_cart"),
    ("public_test", ONLY_FIXTURE, "现在定义的只有 cart"),
    ("hidden", ONLY_FIXTURE, "现在定义的只有 cart"),
])
def test_a_test_file_without_collectable_tests_is_a_repairable_finding(tmp_path, where, body, declared):
    """测试函数没写成 test 开头（或者只写了 fixture）时 pytest 一个测试都收集不到。这是题族内容的问题，
    要给教师返修的机会，不能说成判分环境的问题。"""
    parsed = cart_family()
    location = "公开测试"
    if where == "public_test":
        parsed["public_test"] = body
    else:
        parsed["tests"]["remove"]["body"] = body
        location = "隐藏测试 test_remove.py"
    result = run_audit(parsed, tmp_path)
    report = finding(result, "no_test_functions")
    assert report["repairable"] is True
    assert report["message"] == (
        f"{location}里没有 pytest 收得到的测试函数（{declared}）。pytest 只收集名字以 test 开头的函数，写在模块层，"
        "或者写在 Test 开头、没有 __init__ 的类里；只定义 fixture 是收集不到的。请把测试函数改成这样的写法。")
    assert result["accepted"] is False and result["family"] is None and result["usable"] == []
    assert "grading_error" not in codes(result)
    assert not (tmp_path / "audit" / "tasks").exists()          # 一个测试都收集不到，不去判分
    assert result["stats"]["checks"]["reference"] is False


@pytest.mark.parametrize("slug, problem", [
    ("Cart", "只能用小写字母、数字、下划线。"),
    ("", "只能用小写字母、数字、下划线。"),
    (None, "只能用小写字母、数字、下划线。"),
    ("s" * 101, "最多 100 个字符（现在 101 个），它要进文件名和 task_id。"),
    ("my_private_cart", "不能带 private，任务库会把这样的文件名当成私有材料、拒绝读出这道题。"),
    ("s" * 100, None),
    ("state_shopping_cart", None),
])
def test_slug_problems_are_named(slug, problem):
    assert audit._slug_problem(slug) == problem


def test_a_family_that_makes_pytest_exit_is_repairable_when_the_grading_environment_is_fine(tmp_path, monkeypatch):
    """参考解在导入时就退出，pytest 一个测试结果都没留下。判分环境是好的（拿一道完好的小题自检），所以这是
    题族内容的问题，要给教师返修的机会。"""
    monkeypatch.setattr(audit, "_PROBE_PASSED", set())
    parsed = cart_family()
    parsed["fixed"] = "import os\nos._exit(3)\n\n\n" + parsed["fixed"]
    result = run_audit(parsed, tmp_path)
    report = finding(result, "reference_no_results")
    assert report["repairable"] is True
    assert report["message"].startswith(
        "参考解的测试没有留下任何结果：error: GradeInfrastructureError: reference solution produced no test "
        "results in tests_public")
    assert report["message"].endswith(
        "。判分环境刚用一道完好的小题自检过，是好的；所以是参考解或测试让 pytest 中途退出了（比如导入时就调用 "
        "os._exit、sys.exit）。请去掉这类写法，让 pytest 把测试跑完。")
    assert result["accepted"] is False and result["family"] is None and result["usable"] == []
    assert "grading_error" not in codes(result) and result["stats"]["checks"]["reference"] is False
    assert (tmp_path / "audit" / "environment-check" / "tasks" / "synth_audit_probe").is_dir()   # 自检是真跑的


def test_the_environment_self_check_is_real_and_runs_once_per_mode(tmp_path, monkeypatch):
    real, probed = audit.grade_program, []
    monkeypatch.setattr(audit, "_PROBE_PASSED", set())

    def broken_for_the_family(task, spec, source, work, timeout=None):
        if spec["module"].startswith("s3_"):        # 题族的题判不完；自检小题照常真判
            raise GradeInfrastructureError("reference solution produced no test results in tests_public")
        probed.append(spec["task_id"])
        return real(task, spec, source, work, timeout)

    monkeypatch.setattr(audit, "grade_program", broken_for_the_family)
    for result in (run_audit(cart_family(), tmp_path), run_audit(cart_family(), tmp_path)):
        report = finding(result, "reference_no_results")
        assert report["repairable"] is True
        assert report["message"] == (
            "参考解的测试没有留下任何结果：error: GradeInfrastructureError: reference solution produced no test "
            "results in tests_public。判分环境刚用一道完好的小题自检过，是好的；所以是参考解或测试让 pytest "
            "中途退出了（比如导入时就调用 os._exit、sys.exit）。请去掉这类写法，让 pytest 把测试跑完。")
        assert "grading_error" not in codes(result) and result["accepted"] is False
    assert probed == ["synth_audit_probe"]          # 两次审计只自检了一次：通过的结果按隔离模式记住

    # 认不出来的异常仍然记成不可返修的 grading_error，但要说清楚环境是好的
    def crash(task, spec, source, work, timeout=None):
        if spec["module"].startswith("s3_"):
            raise RuntimeError("something unexpected while grading")
        return real(task, spec, source, work, timeout)

    monkeypatch.setattr(audit, "grade_program", crash)
    report = finding(run_audit(cart_family(), tmp_path), "grading_error")
    assert report["repairable"] is False
    assert re.fullmatch(
        r"判分没能完成，没法审计这个题族：error: RuntimeError: something unexpected while grading。"
        r"判分环境是好的（自检小题 [^）]*），问题出在这个题族上，或者出在判分它的过程中。", report["message"])


def test_a_label_that_is_too_long_to_be_a_file_name_is_a_finding(tmp_path):
    """parse_family 不限制标签的长度；以前要到建任务目录时才以 OSError 中断整批审计。"""
    parsed = cart_family()
    parsed["tests"]["x" * 300] = parsed["tests"].pop("count")
    result = run_audit(parsed, tmp_path)
    report = finding(result, "invalid_family")
    assert report["repairable"] is True
    assert report["message"] == ("隐藏测试的标签 xxxxxxxxxxxxxxxxxxxx… 太长（300 个字符，最多 100 个）：它要当文件名 "
                                 "test_<标签>.py，请改短。")
    assert result["accepted"] is False and result["family"] is None and not (tmp_path / "audit" / "tasks").exists()


def test_a_slug_that_is_too_long_is_reported_without_grading(tmp_path):
    parsed = cart_family()
    parsed["public_test"] = "def check_cart():\n    assert True\n"      # 不去判分的题族：这一条单独看 slug
    result = audit.audit_family(parsed, slug="s" * 101, direction=DIRECTION, work_dir=tmp_path / "audit")
    slug = finding(result, "bad_slug")
    assert slug["repairable"] is False
    assert slug["message"] == (f"slug '{'s' * 40}…' 不合规：最多 100 个字符（现在 101 个），它要进文件名和 task_id。"
                              "换一个 slug 再审计。")
    assert result["accepted"] is False and not (tmp_path / "audit" / "tasks").exists()


def test_a_slug_with_private_in_it_is_caught_before_the_task_store_refuses_it(tmp_path):
    """任务库的泄漏检查把 fixture 里名字带 private 的文件当成私有材料：这样的 slug 出的题读不出来。"""
    result = audit.audit_family(cart_family(), slug="my_private_cart", direction=DIRECTION, work_dir=tmp_path / "audit")
    slug = finding(result, "bad_slug")
    assert slug["repairable"] is False
    assert slug["message"] == ("slug 'my_private_cart' 不合规：不能带 private，任务库会把这样的文件名当成私有材料、"
                              "拒绝读出这道题。换一个 slug 再审计。")
    assert codes(result) == ["bad_slug", "grading_error"] and result["accepted"] is False
    assert "private material inside fixture" in finding(result, "grading_error")["message"]
    assert result["stats"]["checks"]["structure"] is False and result["usable"] == []


def test_text_that_cannot_be_written_to_a_file_is_a_finding(tmp_path):
    """JSON 转义写错时会留下孤立的代理项：写不进任何文件。以前要到写文件时才以异常中断。"""
    lone = chr(0xD800)
    parsed = cart_family()
    parsed["mutants"][0]["new"] += "  # " + lone
    result = run_audit(parsed, tmp_path)
    report = finding(result, "mutant_anchor")
    assert report["repairable"] is True
    assert report["message"] == (
        "第 1 条要求的 mutant（第 1 个）的 new 片段里有写不进文件的字符（孤立的代理项，第 1 行）：多半是 JSON "
        "转义写错了，请换成正常的字符。请把 new 片段里孤立的代理项换成正常的字符。")
    assert result["usable"] == [1, 2] and result["accepted"] is False
    assert "第 1 个（new 片段里有写不进文件的字符）" in finding(result, "too_few_mutants")["message"]
    for where, mutate in (("题面", lambda family: family.update(instruction=family["instruction"] + lone)),
                          ("参考解", lambda family: family.update(fixed=family["fixed"] + "# " + lone)),
                          ("公开测试", lambda family: family.update(public_test=family["public_test"] + "# " + lone)),
                          ("隐藏测试 count", lambda family: family["tests"]["count"].update(
                              body=family["tests"]["count"]["body"] + "# " + lone))):
        parsed = cart_family()
        mutate(parsed)
        result = run_audit(parsed, tmp_path)
        assert codes(result) == ["invalid_family"], (where, result["findings"])
        assert f"{where}里有写不进文件的字符" in finding(result, "invalid_family")["message"]


def test_the_reference_is_still_graded_when_no_mutant_can_be_used(tmp_path):
    """一个能用的 mutant 都没有时，原程序就是参考解本身，参考补丁是空的。参考解照样要判：它的结论要进返修
    提示，也不能被说成判分环境的问题。"""
    parsed = cart_family()
    parsed["tests"]["count"]["body"] = TEST_COUNT.replace("== 60", "== 61")
    parsed["mutants"] = []
    result = run_audit(parsed, tmp_path)
    assert codes(result) == ["invalid_family", "reference_fails", "too_few_mutants"]
    assert f"隐藏测试没有通过（没通过：{COUNT}）" in finding(result, "reference_fails")["message"]
    assert result["stats"]["checks"]["reference"] is False and result["usable"] == []
    assert result["accepted"] is False and result["family"] is None
    # 所有 mutant 都在结构上被排除时同理
    parsed = cart_family()
    for mutant in parsed["mutants"]:
        mutant["old"] = "no such anchor in the reference"
    result = run_audit(parsed, tmp_path)
    assert "grading_error" not in codes(result) and result["stats"]["checks"]["reference"] is True
    assert codes(result).count("mutant_anchor") == 3 and "too_few_mutants" in codes(result)


def test_a_null_byte_anywhere_in_the_family_is_a_finding(tmp_path):
    """空字节（JSON 里的 \\u0000）在源码里是语法错误，在题面里是垃圾：都提前报出来，不说成别的问题。"""
    parsed = cart_family()
    parsed["mutants"][0]["new"] += "\x00"
    result = run_audit(parsed, tmp_path)
    assert finding(result, "mutant_anchor")["message"] == (
        "第 1 条要求的 mutant（第 1 个）的 new 片段里有空字节（NUL，第 1 行）：这多半是生成时出的错，请删掉它。"
        "请删掉 new 片段里的空字节（NUL）。")
    assert result["usable"] == [1, 2]
    assert "第 1 个（new 片段里有空字节）" in finding(result, "too_few_mutants")["message"]
    parsed = cart_family()
    parsed["instruction"] += "\x00"
    result = audit.audit_family(parsed, slug=SLUG, direction=DIRECTION, work_dir=tmp_path / "instruction")
    assert codes(result) == ["invalid_family"]
    assert finding(result, "invalid_family")["message"] == (
        "题面里有空字节（NUL，第 4 行）：这多半是生成时出的错，请删掉它。")
    assert not (tmp_path / "instruction").exists()            # 题目都构造不出来，没有工作区、不判分


def test_failing_to_build_the_task_directory_is_a_finding_not_a_crash(tmp_path, monkeypatch):
    """建任务目录时出的一切错都记成发现项：一个题族不该打断整批审计。"""
    def explode(output_dir, specs=None):
        raise OSError(36, "File name too long", personal("audit", "tasks", "s3_cart_001"))

    monkeypatch.setattr(audit, "build_all", explode)
    result = run_audit(cart_family(), tmp_path)
    report = finding(result, "grading_error")
    assert report["repairable"] is False
    assert report["message"].startswith("建不出判分用的任务目录，没法审计这个题族："
                                        "OSError: [Errno 36] File name too long: ")
    assert "s3_cart_001" in report["message"] and personal("audit") not in report["message"]
    assert report["message"].endswith("换一个 slug 也不行，才是判分环境的问题。")
    assert result["accepted"] is False and result["family"] is None and result["usable"] == []
    checks = result["stats"]["checks"]
    assert checks["reference"] is False and checks["variants"] is False and checks["parent"] is False
    assert checks["structure"] and checks["length"] and checks["safety"] and checks["no_hints"]   # 不用判分的照常查完


def test_a_bug_in_the_audit_itself_is_a_finding_not_a_broken_batch(tmp_path, monkeypatch):
    """最后一道兜底：审计程序自己的错也不该让整批审计断在一个题族上，但消息里要说明是程序的问题。"""
    def crash(self):
        raise RuntimeError("bug in the audit")

    monkeypatch.setattr(audit._Audit, "run", crash)
    result = run_audit(cart_family(), tmp_path)
    assert result["accepted"] is False and result["family"] is None and result["usable"] == []
    assert codes(result) == ["grading_error"]
    assert finding(result, "grading_error")["message"] == (
        "审计这个题族时程序出了错，没能得出结论：RuntimeError: bug in the audit。这不是题族内容的问题："
        "请把这个题族和这条信息报告给维护者。")
    assert result["stats"]["checks"] == dict.fromkeys(audit.CHECKS, False)


@pytest.mark.parametrize("control, name", [("\x0c", "换页符"), ("\x0b", "垂直制表符"), ("\r", "回车")])
def test_a_control_character_at_the_end_of_a_source_line_is_reported_with_its_line(control, name):
    """git 把这样的补丁当成坏补丁，判分走不到测试那一步。构造题目时就按行报出来。"""
    parsed = cart_family()
    parsed["mutants"][0]["new"] += control
    with pytest.raises(specs.FamilyError, match="new 片段的第 1 行末尾有控制字符"):
        specs.build_family(parsed, slug=SLUG, direction=DIRECTION)
    parsed = cart_family()
    parsed["fixed"] = parsed["fixed"].replace("        self._discount = 0\n",
                                              "        self._discount = 0" + control + "\n")
    with pytest.raises(specs.FamilyError, match=f"参考解的第 6 行末尾有控制字符（{name}）"):
        specs.build_family(parsed, slug=SLUG, direction=DIRECTION)
    # 行中间的不碍事；题面不进补丁，行尾的回车也不碍事
    parsed = cart_family()
    parsed["fixed"] = parsed["fixed"].replace("        # 同名商品累加件数，", "        # 同名商品累加件数，\x0c")
    parsed["instruction"] = parsed["instruction"].replace("\n", "\r\n")
    parent = specs.build_family(parsed, slug=SLUG, direction=DIRECTION)["parent"]
    assert "\x0c" in parent["fixed"] and parent["instruction"].count("\r") == INSTRUCTION.count("\n")


def test_a_mutant_that_would_break_the_grading_patch_is_left_out(tmp_path):
    """以前这个 mutant 会留到判分：判分器拒绝应用补丁，发现项却说成“公开测试失败”。"""
    parsed = cart_family()
    parsed["mutants"][0]["new"] += "\x0c"
    result = run_audit(parsed, tmp_path)
    report = finding(result, "mutant_anchor")
    assert report["repairable"] is True
    assert report["message"] == (
        "第 1 条要求的 mutant（第 1 个）的 new 片段的第 1 行末尾有控制字符（换页符）：判分用的补丁会因此变成"
        "坏补丁，请删掉它。请删掉 new 片段行末尾的控制字符（回车、垂直制表符、换页符）。")
    assert result["usable"] == [1, 2] and result["accepted"] is False
    assert "第 1 个（new 片段的行末尾有控制字符）" in finding(result, "too_few_mutants")["message"]


def test_a_patch_the_grader_refuses_is_explained(tmp_path, monkeypatch):
    """判分器在应用补丁时就拒绝：测试根本没跑，不能说成公开测试失败，也不能说成改动了受保护的文件。"""
    evidence = {"apply_failed": True, "detail": "error: corrupt patch at line 18"}
    shown = '{"apply_failed": true, "detail": "error: corrupt patch at line 18"}'
    advice = ("多半是程序的某一行末尾带着换页符这类控制字符，补丁因此成了坏补丁；"
              "请检查参考解和 mutant 的 new 片段。")

    def blocked(task, spec, source, work, timeout=None):
        return {"verdict": "invalid", "strict_pass": False, "public_pass": None, "private_pass": None,
                "public": None, "private": None, "public_tail": "", "private_tail": "", "isolation": "landlock",
                "blocked": {"patch": evidence}}

    monkeypatch.setattr(audit, "grade_program", blocked)
    result = run_audit(cart_family(), tmp_path)
    assert finding(result, "reference_fails")["message"] == (
        f"参考解没走到功能层就被判分器拒绝（patch：{shown}）。{advice}")
    ungraded = [item["message"] for item in result["findings"] if item["code"] == "mutant_ungraded"]
    assert ungraded == [
        f"第 {number} 条要求的 mutant（第 {number} 个）没能判完（判分器在 patch：{shown} 就拒绝了，测试没有运行），"
        f"不能当作被拦住。{advice}" for number in (1, 2, 3)]
    assert codes(result) == ["reference_fails", "mutant_ungraded", "mutant_ungraded", "mutant_ungraded",
                             "too_few_mutants"]
    assert result["accepted"] is False and result["usable"] == []


def test_the_pytest_tail_shows_the_layer_that_failed_first():
    """公开测试没过时给公开层的输出（要先修它），不能拿隐藏层的“4 passed”来充数。"""
    result = graded(public_pass=False, public_tail="E   boom in the public test\n1 failed in 0.01s\n",
                    private_tail="....\n4 passed in 0.02s\n")
    assert audit._tail(result).splitlines() == ["E   boom in the public test", "1 failed in 0.01s"]
    assert audit._tail(graded(private_tail="FAILED tests_private/test_count.py::test_count\n")).splitlines() == [
        "FAILED tests_private/test_count.py::test_count"]


def test_every_finding_code_is_listed_in_the_module_docstring():
    """调用方按 code 分门别类地处理发现项（比如遇到超时原样重试）：模块说明里列的 code 要和代码里真正
    报出来的一一对应，不能漏，也不能列了不存在的。"""
    source = Path(audit.__file__).read_text(encoding="utf-8")
    literals = set(re.findall(r'self\.add\(\s*"([a-z_]+)"', source))
    suffixes = set(re.findall(r'f"\{kind\}_([a-z_]+)"', source))
    emitted = literals | {prefix + suffix for prefix in ("mutant_", "parent_") for suffix in suffixes}
    # 说明里“发现项的 code”下面的几行只列 code；别的段落还提到 parse_family 这样的名字
    section = audit.__doc__.split("发现项的 code，按项分：", 1)[1].splitlines()[1:]      # 去掉标题后面的空行
    bullets = "".join(itertools.takewhile(lambda line: line[:1] in ("-", " "), section))
    documented = set(re.findall(r"\b[a-z]+(?:_[a-z]+)*\b", bullets))
    assert len(emitted) == 32 and emitted == documented, (sorted(emitted - documented), sorted(documented - emitted))
