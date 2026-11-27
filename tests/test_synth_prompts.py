# -*- coding: utf-8 -*-
"""合成出题的主题排布、提示和解析：不访问教师接口；解析是真的，按格式拼出来的题也在沙箱里真判分。"""
import ast
import copy
import importlib
import json
import re
import sys
import tempfile
import textwrap
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agentlab.benchmark.builder import build_all
from agentlab.benchmark.hard_tasks_v3 import HARD3_SPECS
from agentlab.benchmark.task_store import TaskStore
from agentlab.synth import parse, prompts, themes

import export_tasks_v2 as pools
import privacy_check
from self_check_pool import prompt_estimate, rough_tokens

# 一道照着提示里的格式写的小题族：带缓存的库存。六处缺陷分属四条要求，改动的行互不重叠
STOCK_FIXED = '''\
class Stock:
    """库存：商品名 → 数量。total() 的结果缓存在 _total 里。"""

    def __init__(self):
        self._items = {}
        self._total = None

    def _refresh(self):
        self._total = None

    def add(self, name, qty):
        if qty <= 0:
            raise ValueError('qty must be positive')
        self._items[name] = self._items.get(name, 0) + qty
        self._refresh()

    def take(self, name, qty):
        if name not in self._items:
            raise KeyError(name)
        if qty <= 0 or qty > self._items[name]:
            raise ValueError('not enough')
        self._items[name] -= qty
        self._refresh()

    def remove(self, name):
        if name not in self._items:
            raise KeyError(name)
        qty = self._items.pop(name)
        self._refresh()
        return qty

    def total(self):
        if self._total is None:
            self._total = sum(self._items.values())
        return self._total
'''
STOCK_INSTRUCTION = (
    "Stock 管理库存（商品名 → 数量），并缓存 total() 的结果。要求：\n"
    "1）add(name, qty)：qty ≤ 0 抛 ValueError 且不改变任何状态；否则把数量累加到该商品上。\n"
    "2）take(name, qty)：name 不存在抛 KeyError；qty ≤ 0 或超过现有数量抛 ValueError，都不改变任何状态；"
    "否则扣减数量。\n"
    "3）remove(name) 删除该商品并返回它的数量，name 不存在抛 KeyError。\n"
    "4）任何修改之后（包括抛出异常的操作），total() 都等于按当前各项数量重新计算的结果；空库存时为 0。"
)
# 标签 → (对应的要求, 测试函数)
STOCK_TESTS = {
    "add": (1, '''
def test_add_rejects_bad_quantity_and_keeps_state():
    stock = Stock()
    stock.add('pen', 3)
    with pytest.raises(ValueError):
        stock.add('ink', 0)
    with pytest.raises(ValueError):
        stock.add('ink', -1)
    stock.add('pen', 2)
    assert stock.total() == 5
'''),
    "take": (2, '''
def test_take_checks_before_changing():
    stock = Stock()
    stock.add('pen', 5)
    with pytest.raises(KeyError):
        stock.take('ink', 1)
    with pytest.raises(ValueError):
        stock.take('pen', 0)
    with pytest.raises(ValueError):
        stock.take('pen', 6)
    stock.take('pen', 2)
    assert stock.total() == 3
'''),
    "remove": (3, '''
def test_remove_returns_the_removed_quantity():
    stock = Stock()
    stock.add('pen', 5)
    stock.add('ink', 1)
    assert stock.remove('ink') == 1
    with pytest.raises(KeyError):
        stock.remove('ink')
    assert stock.total() == 5
'''),
    "total": (4, '''
def test_total_follows_every_change():
    stock = Stock()
    stock.add('pen', 5)
    assert stock.total() == 5
    stock.add('ink', 1)
    assert stock.total() == 6
    stock.take('pen', 2)
    assert stock.total() == 4
    stock.remove('ink')
    assert stock.total() == 3
'''),
}
# (对应的要求, old, new, 缺陷)
STOCK_MUTANTS = [
    (1, "        if qty <= 0:\n            raise ValueError('qty must be positive')\n", "", "add 忘了校验数量"),
    (2, "        if qty <= 0 or qty > self._items[name]:\n", "        if qty > self._items[name]:\n",
     "take 忘了拦住非正数"),
    (3, "        return qty\n", "", "remove 忘了返回数量"),
    (4, "        self._items[name] = self._items.get(name, 0) + qty\n        self._refresh()\n",
     "        self._items[name] = self._items.get(name, 0) + qty\n", "add 之后忘了让缓存作废"),
    (4, "        self._items[name] -= qty\n        self._refresh()\n", "        self._items[name] -= qty\n",
     "take 之后忘了让缓存作废"),
    (4, "        qty = self._items.pop(name)\n        self._refresh()\n", "        qty = self._items.pop(name)\n",
     "remove 之后忘了让缓存作废"),
]
STOCK_PUBLIC = ("def test_public_add_and_total():\n    stock = Stock()\n    stock.add('pen', 3)\n"
                 "    stock.add('ink', 2)\n    assert stock.total() == 5\n")


def family():
    """一份合格的题族，已经是规范形式。每次给一份新的，测试里随便改。"""
    return {
        "theme": "带缓存的库存：增减之后总数要对",
        "instruction": STOCK_INSTRUCTION,
        "fixed": STOCK_FIXED,
        "public_test": STOCK_PUBLIC,
        "tests": {label: {"requirement": number, "body": textwrap.dedent(body).strip() + "\n"}
                  for label, (number, body) in STOCK_TESTS.items()},
        "mutants": [{"requirement": number, "old": old, "new": new, "idea": idea}
                    for number, old, new, idea in STOCK_MUTANTS],
    }


def reply(item) -> str:
    return json.dumps(item, ensure_ascii=False)


def edit(change) -> str:
    item = family()
    change(item)
    return reply(item)


def first_theme(direction):
    """某个方向在本批里的第一个主题和它的 slug。"""
    plan = [theme for theme in themes.plan_themes(1, 50) if theme[0].name == direction]
    return plan[0], themes.slug_of(*plan[0])


def slugs(plan):
    return [themes.slug_of(direction, domain) for direction, domain in plan]


# ── 主题：方向、领域、排批次 ──


def test_directions_point_at_real_tasks_and_summarize_real_slips():
    """五个方向的导入路径和列表名都要真的能读到现有难题；常见疏忽每个方向三五句，一句一条。"""
    assert list(themes.DIRECTIONS) == ["state", "tx", "exc", "lock", "cond"]
    assert [direction.label for direction in themes.DIRECTIONS.values()] == [
        "状态一致性", "事务回滚", "异常安全", "锁序", "条件变量"]
    seen = set()
    for direction in themes.DIRECTIONS.values():
        specs = getattr(importlib.import_module(direction.module), direction.specs)
        assert len(specs) >= 2 and {spec["task_id"] for spec in specs}.isdisjoint(seen)
        seen.update(spec["task_id"] for spec in specs)
        assert direction.summary.endswith("。")
        assert 3 <= len(direction.slips) <= 5
        assert all(slip.endswith("。") and slip.count("。") == 1 for slip in direction.slips)


def test_domains_are_specific_and_distinct():
    assert len(themes.DOMAINS) >= 60
    codes = Counter(domain.code for domain in themes.DOMAINS)
    names = Counter(domain.name for domain in themes.DOMAINS)
    assert set(codes.values()) == set(names.values()) == {1}
    for domain in themes.DOMAINS:
        assert re.fullmatch(r"[a-z][a-z0-9_]*", domain.code) and len(domain.name) >= 2
        # 点得出它管哪几样数据和操作，才算具体的领域，而不是"数据管理"这种泛泛的词
        assert len(domain.hint) >= 8 and re.search("[、，]", domain.hint)


@pytest.mark.parametrize("batch, families, seed", [(1, 50, 42), (3, 50, 42), (2, 7, 42), (1, 50, 7), (9, 50, 42)])
def test_plan_is_deterministic_and_slugs_are_legal(batch, families, seed):
    first = themes.plan_themes(batch, families, seed=seed)
    second = themes.plan_themes(batch, families, seed=seed)
    assert first == second and len(first) == families
    for direction, domain in first:
        slug = themes.slug_of(direction, domain)
        assert re.fullmatch(r"[a-z0-9_]+", slug)
        assert slug == f"{direction.name}_{domain.code}"
        assert prompts.module_name(slug) == f"s3_{slug}"
    assert len(set(slugs(first))) == families


def test_seed_changes_the_order_but_not_what_is_available():
    everything = set(slugs(themes.plan_themes(1, 545)))
    default = slugs(themes.plan_themes(1, 50))
    other = slugs(themes.plan_themes(1, 50, seed=7))
    assert len(everything) == 5 * len(themes.DOMAINS)
    assert set(default) <= everything and set(other) <= everything and other != default


@pytest.mark.parametrize("batch", range(1, 7))
def test_directions_take_turns_evenly_inside_every_batch(batch):
    """一批 50 个主题里五个方向各 10 个：先跑哪一批，五个方向的题都一样多。"""
    plan = themes.plan_themes(batch, 50)
    assert Counter(direction.name for direction, _ in plan) == dict.fromkeys(themes.DIRECTION_NAMES, 10)
    counts = Counter(direction.name for direction, _ in themes.plan_themes(batch, 7)).values()
    assert max(counts) - min(counts) <= 1
    # 不按批切开的序列里也是每 5 个一轮，方向不扎堆（前 500 个主题够取 100 个）
    whole = themes.plan_themes(batch, 100 if batch <= 5 else 45)
    for start in range(0, len(whole) - 4, 5):
        assert {direction.name for direction, _ in whole[start:start + 5]} == set(themes.DIRECTION_NAMES)


def test_domains_do_not_repeat_inside_a_direction():
    """一个方向在一个领域上只出一道题；换一个方向用同一个领域，是另一道题。"""
    plans = [themes.plan_themes(batch, 50) for batch in (1, 2, 3, 4)]
    for plan in plans:
        for direction in themes.DIRECTION_NAMES:
            picked = [domain.code for item, domain in plan if item.name == direction]
            assert len(picked) == len(set(picked)) == 10
        assert len(set(slugs(plan))) == 50
    four_batches = [theme for plan in plans for theme in plan]
    assert len(set((item.name, domain.code) for item, domain in four_batches)) == 200
    assert len({domain.code for item, domain in four_batches if item.name == "lock"}) == 40


def test_batches_do_not_overlap_and_six_batches_of_fifty_fit():
    plans = [slugs(themes.plan_themes(batch, 50)) for batch in range(1, 8)]
    assert all(len(plan) == 50 for plan in plans)
    assert sum(map(len, plans)) == len(set(sum(plans, []))) == 350
    # 5 个方向 × 全部领域 = 545 个主题：每批 50 个时第 11 批只剩 45 个；每批 5 个时第 109 批正好是最后 5 个
    assert len(themes.plan_themes(109, 5)) == 5
    with pytest.raises(ValueError, match="凑不够"):
        themes.plan_themes(11, 50)
    with pytest.raises(ValueError, match="凑不够"):
        themes.plan_themes(110, 5)


def test_used_slugs_are_skipped_and_the_batch_is_still_filled():
    plain = themes.plan_themes(2, 50)
    used = slugs(plain)[:7]
    again = themes.plan_themes(2, 50, used_slugs=used)
    assert len(again) == 50 and set(slugs(again)).isdisjoint(used)
    assert again[:43] == plain[7:]                 # 没用过的照原顺序，缺的从序列后面补
    assert again[43:] == themes.plan_themes(3, 50)[:7]
    # 生成器也行。第 1 批因为有跳过而伸进了第 2 批的窗口：把第 1 批出过题的 slug 也记进去，两批就不撞
    first_batch = themes.plan_themes(1, 50, used_slugs=(slug for slug in slugs(themes.plan_themes(1, 50))[:7]))
    second_batch = themes.plan_themes(2, 50, used_slugs=[*slugs(first_batch)])
    assert len(second_batch) == 50 and set(slugs(second_batch)).isdisjoint(slugs(first_batch))


@pytest.mark.parametrize("batch, families", [(0, 50), (-1, 50), (1, 0), (1, -3), ("2", 50), (2, 4.0), (True, 50)])
def test_plan_rejects_bad_arguments(batch, families):
    with pytest.raises(ValueError):
        themes.plan_themes(batch, families)


def test_slug_of_takes_objects_or_names():
    state, stock = themes.DIRECTIONS["state"], themes.DOMAINS[0]
    assert (stock.name, stock.code) == ("库存", "inventory")
    assert {themes.slug_of(state, stock), themes.slug_of("state", stock), themes.slug_of("state", "inventory"),
            themes.slug_of("状态一致性", "库存")} == {"state_inventory"}
    with pytest.raises(ValueError, match="未知的方向"):
        themes.slug_of("state2", stock)
    with pytest.raises(ValueError, match="未知的领域"):
        themes.slug_of(state, "仓储")
    with pytest.raises(ValueError, match="二元组"):
        themes.resolve_theme(("state", "库存", "多余"))


def test_every_theme_fits_the_module_name_rule():
    """slug 进文件名和 task_id：全部主题的模块名都合乎题池对文件名的要求（只认字母、数字、下划线）。"""
    for direction, domain in themes.plan_themes(1, 545):
        slug = themes.slug_of(direction, domain)
        assert re.fullmatch(r"s3_[a-z0-9_]+\.py", f"{prompts.module_name(slug)}.py")


# ── 风格参照、要求条数 ──


def existing_ids():
    return {name: {spec["task_id"] for spec in getattr(importlib.import_module(item.module), item.specs)}
            for name, item in themes.DIRECTIONS.items()}


_BANNED_IO = re.compile(r"\bopen\(|\bos\.|\bimport os\b|\bpathlib\b|\bsubprocess\b|\bshutil\b")


def broken_rules(spec):
    """这道现有题违反了提示里的哪几条硬性要求——测试里自己按提示的规则判一遍，不借 pick_example 的判断。

    读写文件这类写法只做粗略的文字检查：够用来认出现有题里的 open()、os.replace()、import os。
    """
    fixed, instruction = spec["fixed"], spec["instruction"]
    count = len(parse.requirement_numbers(instruction))
    rules = [
        ("single file", not spec.get("extra_files")),
        ("lines", prompts.FIXED_LINES[0] <= fixed.count("\n") <= prompts.FIXED_LINES[1]),
        ("fixed chars", len(fixed) <= prompts.MAX_FIXED_CHARS),
        ("instruction chars", len(instruction) <= prompts.MAX_INSTRUCTION_CHARS),
        ("requirement count", prompts.MIN_REQUIREMENTS <= count <= prompts.MAX_REQUIREMENTS),
        ("no file io", not _BANNED_IO.search(fixed)),
    ]
    return [name for name, kept in rules if not kept]


@pytest.mark.parametrize("direction", themes.DIRECTION_NAMES)
def test_pick_example_chooses_a_same_direction_task_and_always_the_same_one(direction):
    """每个方向都挑得到同方向的题，而且只从本方向挑；挑哪道、给哪个测试文件由 slug 决定。"""
    ids = existing_ids()
    others = set().union(*(tasks for name, tasks in ids.items() if name != direction))
    specs = {spec["task_id"]: spec for spec in existing_specs(direction)}
    slugs_here = [themes.slug_of(item, domain) for item, domain in themes.plan_themes(1, 545) if item.name == direction]
    chosen = set()
    for slug in slugs_here:
        example = prompts.pick_example(direction, slug)
        assert example == prompts.pick_example(themes.DIRECTIONS[direction], slug)
        assert example["task_id"] in ids[direction] and example["task_id"] not in others
        spec = specs[example["task_id"]]
        chosen.add(example["task_id"])
        assert example["instruction"] == spec["instruction"] and example["fixed"] == spec["fixed"]
        assert example["test_file"] in spec["private_tests"]
        assert example["test_body"] == spec["private_tests"][example["test_file"]]
        assert example["test_body"].startswith("import pytest\nfrom h3_")
        assert example["truncated"] is False
    assert chosen                              # 这个方向的主题都拿到了参照


@pytest.mark.parametrize("direction", themes.DIRECTION_NAMES)
def test_pick_example_prefers_references_that_obey_the_prompt(direction):
    """风格参照是让教师照着写详细程度的：先在完全合规的现有题里挑；没有合规的（tx 的 8 道全带只读依赖文件）
    才挑违规最少的，并把违规处列进 caveats，让提示明说"这几处不要学"。"""
    specs = {spec["task_id"]: spec for spec in existing_specs(direction)}
    lawful = {task_id for task_id, spec in specs.items() if not broken_rules(spec)}
    fewest = min(len(broken_rules(spec)) for spec in specs.values())
    seen = {}
    for item, domain in themes.plan_themes(1, 545):
        if item.name != direction:
            continue
        example = prompts.pick_example(item, themes.slug_of(item, domain))
        seen.setdefault(example["task_id"], example)
        assert len(example["caveats"]) == len(broken_rules(specs[example["task_id"]]))
        assert len(example["caveats"]) == (0 if lawful else fewest)
        if lawful:
            assert example["task_id"] in lawful
            assert not _BANNED_IO.search(example["test_body"]) and "import os" not in example["test_body"]
    assert set(seen) == (lawful if lawful else {task_id for task_id, spec in specs.items()
                                                 if len(broken_rules(spec)) == fewest})


def test_generation_prompt_spells_out_where_the_reference_breaks_the_rules():
    """参照题不合规的几处，提示里要点名"不要学"；合规的就明说照它来。"""
    for direction, expects_caveats in (("tx", True), ("lock", False)):
        (item, domain), slug = first_theme(direction)
        example = prompts.pick_example(item, slug)
        content = prompts.generation_messages((item, domain), example, slug)[1]["content"]
        assert bool(example["caveats"]) is expects_caveats
        if expects_caveats:
            assert "这道参照题有下面几处不符合上面的硬性要求，这几处不要学：" in content
            for caveat in example["caveats"]:
                assert f"- {caveat}。" in content
            assert any("只读依赖文件" in caveat and "你的题只能有一个文件" in caveat for caveat in example["caveats"])
            assert "这道题本身满足上面的全部硬性要求" not in content
        else:
            assert "这道题本身满足上面的全部硬性要求，题面、参考解和测试的详细程度照它来。" in content
            assert "这几处不要学" not in content
        assert "题面照它的写法" in content and "抛出的异常类型" in content


def _reference(task_id, **overrides):
    """一道拿来当现有题的规格：默认完全合规，改哪个字段就哪里不合规。"""
    spec = {"task_id": task_id, "instruction": STOCK_INSTRUCTION, "fixed": STOCK_FIXED, "public_test": STOCK_PUBLIC,
            "private_tests": {"test_add.py": textwrap.dedent(STOCK_TESTS["add"][1]).strip() + "\n",
                              "test_remove.py": textwrap.dedent(STOCK_TESTS["remove"][1]).strip() + "\n"}}
    spec.update(overrides)
    return spec


def test_pick_example_ranks_references_by_how_many_rules_they_break():
    with_dependency = _reference("h3_dep_001", extra_files={"store.py": "x = 1\n"},
                                 fixed="from store import x\n" + STOCK_FIXED)
    file_user = _reference("h3_file_001", fixed="import os\n" + STOCK_FIXED)
    long_one = _reference("h3_long_001", extra_files={"store.py": "x = 1\n"},
                          fixed=STOCK_FIXED + "pad = '" + "x" * 2000 + "'\n")
    clean = _reference("h3_clean_001")
    slugs_here = [f"state_{'y' * length}" for length in range(1, 30)]
    for slug in slugs_here:
        example = prompts.pick_example("state", slug, pool={"state": [with_dependency, long_one, file_user, clean]})
        assert example["task_id"] == "h3_clean_001" and example["caveats"] == []
        # 没有完全合规的：挑违规最少的两道，违规处照实列出来
        example = prompts.pick_example("state", slug, pool={"state": [with_dependency, long_one, file_user]})
        assert example["task_id"] in {"h3_dep_001", "h3_file_001"} and len(example["caveats"]) == 1
    example = prompts.pick_example("state", "state_y", pool={"state": [long_one]})
    assert len(example["caveats"]) == 2
    assert any("只读依赖文件 store.py" in caveat for caveat in example["caveats"])
    assert any(f"它的参考解有 {len(long_one['fixed'])} 个字符；你的不超过 {prompts.MAX_FIXED_CHARS} 个"
               == caveat for caveat in example["caveats"])
    # 隐藏测试文件从不带这些写法的里面挑
    mixed = _reference("h3_mixed_001", private_tests={
        "test_io.py": "import os\n\ndef test_io(tmp_path):\n    assert True\n",
        "test_ok.py": "def test_ok():\n    assert True\n"})
    assert prompts.pick_example("state", "state_mixed", pool={"state": [mixed]})["test_file"] == "test_ok.py"
    only_io = _reference("h3_only_io_001", private_tests={"test_io.py": "import os\n\ndef test_io():\n    pass\n"})
    example = prompts.pick_example("state", "state_only_io", pool={"state": [only_io]})
    assert example["test_file"] == "test_io.py"
    assert any("它的隐藏测试读写文件" in caveat for caveat in example["caveats"])


def existing_specs(direction):
    module = importlib.import_module(themes.DIRECTIONS[direction].module)
    return list(getattr(module, themes.DIRECTIONS[direction].specs))


def test_pick_example_cuts_a_long_reference_solution_at_150_lines():
    long_fixed = STOCK_FIXED + "pad = [\n" + "    0,\n" * 200 + "]\n"
    spec = {"task_id": "h3_pad_001", "instruction": "题面", "fixed": long_fixed,
            "private_tests": {"test_a.py": "body a", "test_z.py": "body z"}}
    example = prompts.pick_example("state", "state_pad", pool={"state": [spec]})
    assert prompts.EXAMPLE_LINES == 150
    assert example["fixed"].count("\n") == 150
    assert example["fixed"] == "\n".join(long_fixed.splitlines()[:150]) + "\n"
    assert example["truncated"] is True and example["test_body"] in ("body a", "body z")
    short = prompts.pick_example("state", "state_pad", pool={"state": [{**spec, "fixed": STOCK_FIXED}]})
    assert short["fixed"] == STOCK_FIXED and short["truncated"] is False


def test_pick_example_needs_something_to_choose_from():
    with pytest.raises(ValueError, match="没有可供参照"):
        prompts.pick_example("state", "state_x", pool={"state": []})
    with pytest.raises(ValueError, match="没有可供参照"):
        prompts.pick_example("lock", "lock_x", pool={"state": []})
    with pytest.raises(ValueError, match="未知的方向"):
        prompts.pick_example("locking", "lock_x")


def test_requirement_count_follows_the_slug():
    assert prompts.MIN_REQUIREMENTS == 4 and prompts.MAX_REQUIREMENTS == 6
    seen = Counter()
    for slug in slugs(themes.plan_themes(1, 545)):
        count = prompts.requirement_count(slug)
        assert prompts.MIN_REQUIREMENTS <= count <= prompts.MAX_REQUIREMENTS
        assert count == prompts.requirement_count(slug)
        seen[count] += 1
    assert set(seen) == {4, 5, 6}                   # 三个档位都用得上，不是一个常数


# ── 生成提示 ──


@pytest.mark.parametrize("direction", themes.DIRECTION_NAMES)
def test_generation_prompt_carries_theme_example_and_format(direction):
    (item, domain), slug = first_theme(direction)
    example = prompts.pick_example(item, slug)
    system, user = prompts.generation_messages((item, domain), example, slug)
    assert [message["role"] for message in (system, user)] == ["system", "user"]
    # 角色 + 六条硬性要求，每条原样出现一次
    assert "资深 Python 工程师" in system["content"] and "出题人" in system["content"]
    assert re.findall(r"(?m)^(\d)\. ", system["content"]) == [str(number) for number in range(1, 7)]
    for text in prompts.HARD_REQUIREMENTS:
        assert system["content"].count(text) == 1
    # 方向：名称、说明、常见疏忽
    assert item.label in user["content"] and item.summary in user["content"]
    for slip in item.slips:
        assert slip in user["content"]
    # 领域、模块名、要求条数
    assert f"【领域】{domain.name}：{domain.hint}。" in user["content"]
    assert f"s3_{slug}.py" in user["content"] and f"from s3_{slug} import *" in user["content"]
    count = prompts.requirement_count(slug)
    assert f"【要求条数】{count} 条：instruction 里从 1）编到 {count}）。" in user["content"]
    # 风格参照：题面全文、参考解、一个隐藏测试文件
    assert "不要照抄它的主题、命名和逻辑" in user["content"]
    assert "风格参照" not in system["content"]            # 硬性要求里不提它：返修提示未必带参照
    assert example["instruction"] in user["content"]
    assert f"```python\n{example['fixed']}```" in user["content"]
    assert f"隐藏测试文件 {example['test_file']}" in user["content"]
    assert f"```python\n{example['test_body']}```" in user["content"]
    # 格式说明是提示里唯一的 json 围栏，骨架能解析、键同 parse
    blocks = re.findall(r"```json\n(.*?)\n```", user["content"], re.DOTALL)
    assert blocks == [prompts.format_skeleton()]
    assert list(json.loads(blocks[0])) == list(parse.REQUIRED_KEYS)


def test_hard_requirements_state_what_the_pipeline_needs():
    """六条硬性要求各管一件事；其中的数字就是 parse 和审计用的数字。"""
    one, two, three, four, five, six = prompts.HARD_REQUIREMENTS
    for needle in ("标准库", "threading", "不联网", "不起子进程", "不读写文件", "eval", "exec", "input",
                   "时钟", "测试不依赖真实时间", "0.2 秒"):
        assert needle in one
    assert f"{prompts.FIXED_LINES[0]} 到 {prompts.FIXED_LINES[1]} 行" in two and "BUG、FIXME、TODO" in two
    assert "1）2）3）" in three and "requirement" in three and "每条单独占一行" in three
    assert f"不超过 {prompts.MAX_INSTRUCTION_CHARS} 个字符" in three
    assert "恰好出现一次" in four and f"至少 {parse.MIN_OLD_CHARS} 个字符" in four
    assert "语法错误" in four and "公开测试失败" in four
    assert "公开测试" in five and f"不超过 {prompts.MAX_PUBLIC_TEST_LINES} 行" in five
    assert "只输出一个 JSON 对象" in six and "json" in six


def test_format_skeleton_is_the_shape_parse_accepts():
    skeleton = json.loads(prompts.format_skeleton())
    item = family()
    assert list(skeleton) == list(parse.REQUIRED_KEYS) == list(item)
    assert list(skeleton["tests"]["<标签>"]) == ["requirement", "body"] == list(item["tests"]["add"])
    assert list(skeleton["mutants"][0]) == ["requirement", "old", "new", "idea"] == list(item["mutants"][0])
    assert parse.parse_family(reply(item)) == item


def test_everything_named_in_the_skeleton_is_required_by_parse():
    def drop(path):
        def change(item):
            target = item
            for key in path[:-1]:
                target = target[key]
            del target[path[-1]]
        return change

    for path in [("theme",), ("instruction",), ("fixed",), ("public_test",), ("tests",), ("mutants",),
                 ("tests", "add", "requirement"), ("tests", "add", "body"),
                 ("mutants", 0, "requirement"), ("mutants", 0, "old"), ("mutants", 0, "new")]:
        with pytest.raises(ValueError, match="合格的题族"):
            parse.parse_family(edit(drop(path)))
    # 只有 idea 是可选的
    parsed = parse.parse_family(edit(drop(("mutants", 0, "idea"))))
    assert parsed["mutants"][0]["idea"] == ""


def relabel(item, label):
    item["tests"][label] = item["tests"].pop("add")


@pytest.mark.parametrize("label", ["rename", "bulk_update", "A1", "_", "Abc019_Z", "x" * 60])
def test_labels_made_of_letters_digits_and_underscores_are_accepted(label):
    assert label in parse.parse_family(edit(lambda item: relabel(item, label)))["tests"]


@pytest.mark.parametrize("label", ["a-b", "a b", "a.b", "改名", "a/b", "aé", "a\nb", ""])
def test_labels_with_any_other_character_are_rejected(label):
    with pytest.raises(ValueError, match="标签"):
        parse.parse_family(edit(lambda item: relabel(item, label)))


def test_labels_may_only_differ_by_more_than_case():
    def change(item):
        item["tests"]["ADD"] = copy.deepcopy(item["tests"]["add"])
    with pytest.raises(ValueError, match="只差大小写"):
        parse.parse_family(edit(change))


def test_generation_prompt_rejects_a_slug_that_cannot_be_a_module_name():
    (item, domain), _ = first_theme("state")
    example = prompts.pick_example(item, "state_inventory")
    for slug in ("State-Inventory", "state inventory", "", "s3/state", None):
        with pytest.raises(ValueError, match="slug"):
            prompts.generation_messages((item, domain), example, slug)


def test_theme_may_be_given_as_names_and_the_example_is_optional():
    slug = "state_inventory"
    theme = (themes.DIRECTIONS["state"], themes.DOMAINS[0])
    with_example = prompts.generation_messages(theme, prompts.pick_example("state", slug), slug)
    without = prompts.generation_messages(("state", "库存"), None, slug)
    assert without[0] == with_example[0]
    full = with_example[1]["content"]
    start, end = full.index("【风格参照】"), full.index("【输出格式】")
    assert full[:start] + full[end:] == without[1]["content"]
    with pytest.raises(ValueError, match="未知的方向"):
        prompts.generation_messages(("nope", "库存"), None, slug)


def test_the_stated_size_limits_together_stay_inside_the_token_budget():
    """三个上限要自洽：全都写到上限的题，按题池的粗估（审计用的是同一个算法）不能超过 1900 个 token。

    往宽处算：母题的原程序带着全部缺陷，按比参考解长一成算；公开测试每行 45 个字符写满；题面算两遍——
    现有难题里最"密"的那一篇（每个字符折合的 token 最多）循环到上限，以及整篇全是汉字。
    """
    module = "s3_state_inventory"
    fixed = (STOCK_FIXED * 20)[:prompts.MAX_FIXED_CHARS]
    mother = (STOCK_FIXED * 30)[:prompts.MAX_FIXED_CHARS * 11 // 10]
    public = "    assert ledger.balance('acct-0001') == 12345\n" * prompts.MAX_PUBLIC_TEST_LINES
    densest = max((spec["instruction"] for spec in HARD3_SPECS), key=lambda text: rough_tokens(text) / len(text))
    assert len(fixed) == prompts.MAX_FIXED_CHARS and len(mother) > len(fixed)
    assert public.count("\n") == prompts.MAX_PUBLIC_TEST_LINES
    totals = {}
    for name, instruction in (("最密的现有题面", (densest * 10)[:prompts.MAX_INSTRUCTION_CHARS]),
                              ("全是汉字", "要" * prompts.MAX_INSTRUCTION_CHARS)):
        assert len(instruction) == prompts.MAX_INSTRUCTION_CHARS
        spec = {"instruction": instruction, "module": f"{module}.py", "buggy": mother, "fixed": fixed,
                "public_test": f"from {module} import *\n\n" + public}
        totals[name] = prompt_estimate(spec) + rough_tokens(fixed)
    assert max(totals.values()) <= 1900, totals


# ── 返修提示 ──


def test_repair_prompt_quotes_the_previous_version_and_lists_every_finding():
    (item, domain), slug = first_theme("tx")
    # 上一版自己就带着 json 围栏：包它的围栏要更长，才不会在半截收尾
    previous = "题族如下：\n```json\n" + reply(family()) + "\n```\n完。"
    findings = ["参考解没通过隐藏测试 test_take。", "第 2 个 mutant 没有被任何隐藏测试拦住。"]
    system, user = prompts.repair_messages(previous, findings, (item, domain), slug)
    assert [message["role"] for message in (system, user)] == ["system", "user"]
    assert "没有通过检查" in system["content"] and "只输出修改后的完整 JSON 对象" in system["content"]
    assert prompts.REQUIREMENTS in system["content"]              # 每一轮都是新的对话，硬性要求要再给一遍
    content = user["content"]
    assert f"````\n{previous}\n````" in content                 # 上一版原样
    assert content.index("【上一版】") < content.index("【检查发现的问题】") < content.index("【输出格式】")
    listed = content.split("【检查发现的问题】")[1].split("【输出格式】")[0]
    assert "共 2 条" in listed
    assert "1. 参考解没通过隐藏测试 test_take。" in listed
    assert "2. 第 2 个 mutant 没有被任何隐藏测试拦住。" in listed
    assert "【风格参照】" not in content and "风格参照" not in system["content"]
    assert "题面的写法沿用上一版：写清方法名、参数、返回值，以及抛出的异常类型。" in system["content"]
    assert content.split("【输出格式】")[1].count("```json") == 1
    assert f"from s3_{slug} import *" in content and f"【要求条数】{prompts.requirement_count(slug)} 条" in content


def _drop_first_mutant_and_tests(item):
    item["mutants"].pop(0)
    item["tests"] = {}


def test_repair_prompt_takes_audit_findings_or_a_parse_error_as_findings():
    (item, domain), slug = first_theme("exc")
    audit_finding = {"code": "reference_fails", "message": "参考解过不了隐藏测试 test_take：断言失败。",
                     "repairable": True}
    findings = [audit_finding, "第 3 个 mutant 让公开测试挂了。", ValueError("题面里没有这一条要求")]
    content = prompts.repair_messages(reply(family()), findings, (item, domain), slug)[1]["content"]
    assert "1. 参考解过不了隐藏测试 test_take：断言失败。" in content
    assert "2. 第 3 个 mutant 让公开测试挂了。" in content and "3. 题面里没有这一条要求" in content
    assert "repairable" not in content and "code" not in content
    # 解析失败的原话进返修提示：一条不少
    broken = edit(_drop_first_mutant_and_tests)
    with pytest.raises(ValueError) as caught:
        parse.parse_family(broken)
    content = prompts.repair_messages(broken, caught.value, (item, domain), slug)[1]["content"]
    for line in str(caught.value).splitlines():
        assert line in content


def test_repair_prompt_for_an_empty_previous_reply_carries_a_style_reference():
    """上一版是空的，这一轮其实就是重新出题：自动带上风格参照和自查清单，写法也说"照参照"。"""
    (item, domain), slug = first_theme("cond")
    system, user = prompts.repair_messages("  \n", ["上一版不是 JSON。"], (item, domain), slug)
    content = user["content"]
    assert "上一版回复是空的" in content and "1. 上一版不是 JSON。" in content
    assert "【风格参照】" in content and "不要照抄它的主题、命名和逻辑" in content
    assert "题面照风格参照的写法：" in system["content"]
    assert content.endswith("程序不碰文件、网络、子进程和真实时间。")
    # 调用方给了参照也一并附上
    example = prompts.pick_example(item, slug)
    given = prompts.repair_messages(reply(family()), ["问题。"], (item, domain), slug, example=example)
    assert example["instruction"] in given[1]["content"] and "题面照风格参照的写法：" in given[0]["content"]


def test_repair_prompt_treats_a_single_finding_dict_as_one_finding():
    """审计的一条发现（字典）直接传进来，是一条问题；不能把它的三个键名当成三条。"""
    theme, slug = ("state", "库存"), "state_inventory"
    findings = {"code": "reference_fails", "message": "参考解过不了隐藏测试 test_take：断言失败。", "repairable": True}
    content = prompts.repair_messages("x", findings, theme, slug)[1]["content"]
    listed = content.split("【检查发现的问题】")[1].split("【输出格式】")[0]
    assert "共 1 条" in listed and "1. 参考解过不了隐藏测试 test_take：断言失败。" in listed
    assert "code" not in listed and "repairable" not in listed
    content = prompts.repair_messages("x", {"code": "c1"}, theme, slug)[1]["content"]
    assert "1. code：c1" in content.split("【检查发现的问题】")[1].split("【输出格式】")[0]


def test_repair_prompt_without_findings_says_so():
    (item, domain), slug = first_theme("cond")
    quiet = prompts.repair_messages(reply(family()), [], (item, domain), slug)[1]["content"]
    assert "没有给出具体问题" in quiet


# ── 解析 ──


@pytest.mark.parametrize("wrap", [
    lambda text: text,
    lambda text: f"```json\n{text}\n```",
    lambda text: "好的，题族如下：\n\n```json\n" + textwrap.indent(text, "    ") + "\n```\n\n希望合用。",
    lambda text: f"题族：\n{text}\n以上。",
], ids=["plain", "fenced", "prose-and-fenced", "prose"])
def test_parse_family_accepts_a_good_reply_however_it_is_wrapped(wrap):
    assert parse.parse_family(wrap(reply(family()))) == family()


def test_parse_family_normalizes_the_usual_small_deviations():
    """整个文件多一层缩进、源码带着围栏和 CRLF、序号写成字符串、多余的字段：认出来，化成规范形式。"""
    item = family()

    def fenced(code):
        return ("```python\n" + textwrap.indent(code, "    ") + "```").replace("\n", "\r\n")

    item["fixed"] = fenced(item["fixed"])
    item["public_test"] = fenced(item["public_test"])
    for test in item["tests"].values():
        test["requirement"], test["body"] = str(test["requirement"]), fenced(test["body"])
    for mutant in item["mutants"]:
        mutant["old"] = textwrap.indent(mutant["old"], "    ").replace("\n", "\r\n")
        mutant["new"] = textwrap.indent(mutant["new"], "    ").replace("\n", "\r\n")
    item.update(notes="多余的字段", extra=[1, 2])
    item["tests"]["add"]["label"] = "多余"
    item["mutants"][0]["note"] = "多余"
    parsed = parse.parse_family(reply(item))
    assert parsed == family()
    assert all(mutant["old"] in parsed["fixed"] for mutant in parsed["mutants"])


def test_parse_family_accepts_unescaped_newlines_inside_strings():
    text = reply(family()).replace("\\n", "\n")            # 字符串里直接换行：不合 JSON 的规矩，但意思清楚
    assert parse.parse_family(text) == family()


def test_parse_family_drops_extra_fields_and_keeps_idea():
    item = family()
    item["mutants"][1].pop("idea")
    item["tests"]["add"]["extra"] = 1
    parsed = parse.parse_family(reply({**item, "note": "x"}))
    assert set(parsed) == set(parse.REQUIRED_KEYS)
    assert parsed["mutants"][1] == {**item["mutants"][1], "idea": ""}
    assert parsed["tests"]["add"] == {"requirement": 1, "body": item["tests"]["add"]["body"]}


def test_an_anchor_with_a_stray_newline_is_accepted_and_restored():
    item = family()
    item["mutants"][0]["old"] += "\n"             # 多抄了一个换行；去掉之后在参考解里恰好只有一处
    assert parse.parse_family(reply(item)) == family()


def _laid_out_like_hand_written_json(item):
    """手写 JSON 的排版：多行字符串从换行开始，每一行按它在 JSON 里的列缩进，收尾引号前还留着缩进。"""
    def lay_out(value, depth):
        if isinstance(value, dict):
            return {key: lay_out(inner, depth + 1) for key, inner in value.items()}
        if isinstance(value, list):
            return [lay_out(inner, depth + 1) for inner in value]
        if isinstance(value, str) and "\n" in value:
            pad = " " * (2 * depth + 2)
            return "\n" + textwrap.indent(value, pad) + " " * (len(pad) - 2)
        return value
    return json.dumps(lay_out(item, 0), ensure_ascii=False).replace("\\n", "\n")


def _rstrip_anchors(item):
    item = copy.deepcopy(item)
    for mutant in item["mutants"]:
        mutant["old"], mutant["new"] = mutant["old"].rstrip(), mutant["new"].rstrip()
    return item


def test_layout_indentation_and_edge_whitespace_on_anchors_are_fixed_together():
    """整个文件带着排版缩进、锚点末行又留着收尾引号前的缩进：两种不一致叠在一起也认得出来。
    不能把照抄的锚点报成"找不到"，更不能把它缩短之后报"太短"。"""
    laid_out = json.loads(_laid_out_like_hand_written_json(family()), strict=False)
    assert parse.parse_family(json.dumps(laid_out, ensure_ascii=False).replace("\\n", "\n")) == family()
    # 排版之后再手工去掉锚点末尾的空白：多行锚点照样对得上，末尾照样没有换行
    after_layout = _rstrip_anchors(laid_out)
    restored = _rstrip_anchors(family())
    assert parse.parse_family(json.dumps(after_layout, ensure_ascii=False).replace("\\n", "\n")) == restored
    # 先去掉末尾的换行再排版：收尾的缩进接在最后一行后面，要既去掉行尾空白、又去掉多出来的缩进才对得上
    trimmed_first = _laid_out_like_hand_written_json(_rstrip_anchors(family()))
    assert parse.parse_family(trimmed_first) == restored


def _over_indent(item, index, old_by, new_by):
    """把第 index 个 mutant 的 old、new 各多缩进几格。"""
    mutant = item["mutants"][index]
    mutant["old"] = textwrap.indent(mutant["old"], " " * old_by)
    mutant["new"] = textwrap.indent(mutant["new"], " " * new_by) if mutant["new"].strip() else mutant["new"]


def test_correcting_the_anchor_leaves_a_replacement_that_was_already_right_alone():
    """old 抄多了缩进、new 的缩进本来就对：修 old 可以，new 不能跟着去掉缩进——不然替换进去的那一行
    会丢掉缩进，变体程序直接成了语法错误。"""
    def change(item):
        _over_indent(item, 3, old_by=8, new_by=0)
    parsed = parse.parse_family(edit(change))
    wanted = family()["mutants"][3]
    assert parsed["mutants"][3] == wanted
    ast.parse(parsed["fixed"].replace(wanted["old"], wanted["new"]))


def test_a_correction_that_would_break_the_program_is_refused():
    """old 每一行都多抄了 4 格缩进，new 的第二行却抄错了相对缩进：把 old 修对、new 原样换上去，程序就成了
    语法错误。这种修正不认，并且说清楚原因（别的修法也都对不上）。"""
    wrong_new = "        self._items[name] -= qty\n                self._refresh()\n"

    def change(item):
        item["mutants"][4].update(old="            self._items[name] -= qty\n            self._refresh()\n",
                                  new=wrong_new)
    with pytest.raises(SyntaxError):                 # 确实会改坏：old 修对之后原样换上去就是语法错误
        ast.parse(STOCK_FIXED.replace("        self._items[name] -= qty\n        self._refresh()\n", wrong_new))
    with pytest.raises(ValueError) as caught:
        parse.parse_family(edit(change))
    message = str(caught.value)
    assert "old 在 fixed 里找不到" in message
    assert ("去掉 4 个前导空格之后能在 fixed 里找到 1 处，但照这个修法改完 new 再换上去，程序就不是合法的 Python"
            in message)


def _drop_tests(item):
    del item["tests"]


def _tests_as_list(item):
    item["tests"] = list(item["tests"].values())


def _old_missing(item):
    item["mutants"][0].update(old="        if qty < 0:\n", new="")


def _no_test_for(item):
    del item["tests"]["total"]


def _no_mutant_for(item):
    item["mutants"] = item["mutants"][3:]


def _fixed_syntax(item):
    item["fixed"] = item["fixed"].replace("        return qty\n", "        return qty(\n")


def _body_syntax(item):
    item["tests"]["add"]["body"] = "def test_add(:\n"


def _bad_label(item):
    relabel(item, "a-b")


def _unnumbered(item):
    for digit, word in zip("1234", "一二三四"):
        item["instruction"] = item["instruction"].replace(f"{digit}）", f"{word}、")


def _numbering_gap(item):
    item["instruction"] = item["instruction"].replace("2）take", "5）take")


def _requirement_is_a_list(item):
    item["tests"]["add"]["requirement"] = [1, 2]


def _ambiguous_anchor(item):
    # 原样找不到；去掉尾部空白后是 add、take、remove 里各一句的 self._refresh()，分不清指哪一处
    item["mutants"][0].update(old="        self._refresh()\n\n\n", new="")


def _rename_test_requirement(item):
    item["tests"]["add"].update(requirement=9)


def _mutant_requirement_zero(item):
    item["mutants"][0].update(requirement=0)


def _old_is_null(item):
    item["mutants"][0].update(old=None)


def _mutant_not_dict(item):
    item["mutants"][0] = "old -> new"


def _mutant_new_is_number(item):
    item["mutants"][0].update(new=3)


def _idea_is_list(item):
    item["mutants"][0].update(idea=["a"])


def _public_syntax(item):
    item.update(public_test="def test_public(:\n")


def _blank_theme(item):
    item["theme"] = "  "


def _blank_body(item):
    item["tests"]["add"].update(body="")


def _tests_are_empty(item):
    item["tests"] = {}


def _mutants_are_empty(item):
    item["mutants"] = []


def _mutants_are_object(item):
    item["mutants"] = {"1": item["mutants"][0]}


def _same_old_and_new(item):
    item["mutants"][2].update(new=item["mutants"][2]["old"])


def _old_too_short(item):
    item["mutants"][2].update(old="return qty\n", new="")


@pytest.mark.parametrize("change, expected", [
    (_drop_tests, ["缺少必填键：tests"]),
    (lambda item: item.pop("mutants"), ["缺少必填键：mutants"]),
    (_tests_as_list, ["tests 应是对象"]),
    (_tests_are_empty, ["tests 是空的"]),
    (_mutants_are_empty, ["mutants 是空的"]),
    (_mutants_are_object, ["mutants 应是数组"]),
    (_old_missing, ["mutants 第 1 项（对应第 1 条要求）的 old 在 fixed 里找不到", "'if qty < 0:'"]),
    (_same_old_and_new, ["mutants 第 3 项（对应第 3 条要求）的 old 和 new 相同", "'return qty'"]),
    (_old_too_short, ["old 太短：只有 11 个字符，至少 12 个", "'return qty'"]),
    (_ambiguous_anchor, ["old 在 fixed 里找不到", "能在 fixed 里找到 3 处，但那样就不唯一了"]),
    (_old_is_null, ["的 old 应是字符串，现在是 null"]),
    (_mutant_new_is_number, ["的 new 应是字符串，现在是 int 3"]),
    (_idea_is_list, ["的 idea 应是一句话的字符串，现在是长度 1 的数组"]),
    (_no_test_for, ["第 4 条要求在 tests 里没有对应的测试"]),
    (_no_mutant_for, ["第 1 条要求在 mutants 里没有对应的 mutant", "第 2 条要求在 mutants 里没有对应的 mutant",
                      "第 3 条要求在 mutants 里没有对应的 mutant"]),
    (_rename_test_requirement, ["tests['add']的 requirement 是 9，题面里没有这一条要求（题面共 4 条要求）",
                                "第 1 条要求在 tests 里没有对应的测试"]),
    (_mutant_requirement_zero, ["mutants 第 1 项的 requirement 是 0，题面里没有这一条要求"]),
    (_requirement_is_a_list, ["tests['add']的 requirement 应是一个整数", "覆盖几条就拆成几个标签"]),
    (_fixed_syntax, ["fixed 不是合法的 Python："]),
    (_public_syntax, ["public_test 不是合法的 Python："]),
    (_body_syntax, ["tests['add']的 body 不是合法的 Python："]),
    (_bad_label, ["tests 的标签 'a-b' 不合法：只能用字母、数字、下划线"]),
    (_unnumbered, ["题面 instruction 里找不到编号的行为要求"]),
    (_numbering_gap, ["题面里的要求序号是 [1, 3, 4, 5]，应从 1 起连续编号"]),
    (_blank_theme, ["theme 应是非空的字符串，现在是空字符串"]),
    (lambda item: item.update(theme=[]), ["theme 应是非空的字符串，现在是长度 0 的数组"]),
    (_mutant_not_dict, ["mutants 第 1 项应是对象（requirement、old、new、idea）"]),
    (_blank_body, ["tests['add']的 body 应是非空的字符串，现在是空字符串"]),
], ids=lambda value: getattr(value, "__name__", "lambda"))
def test_parse_family_rejects_a_broken_family_and_names_what_is_wrong(change, expected):
    with pytest.raises(ValueError) as caught:
        parse.parse_family(edit(change))
    lines = str(caught.value).splitlines()
    assert lines[0].startswith("回复不是合格的题族，共 ") and lines[0].endswith(" 处问题：")
    assert len(lines) == int(lines[0].split("共 ")[1].split(" 处")[0]) + 1
    assert all(line.startswith("- ") for line in lines[1:])
    for needle in expected:
        assert needle in str(caught.value)


def test_parse_family_reports_independent_problems_together():
    """互不相干的问题一次报完，返修一轮就能全改掉。"""
    def change(item):
        item.update(tests={"add": item["tests"]["add"]}, theme=" ")
    with pytest.raises(ValueError) as caught:
        parse.parse_family(edit(change))
    lines = sorted(str(caught.value).splitlines()[1:])
    assert [line.split("：")[0] for line in lines] == [
        "- theme 应是非空的字符串，现在是空字符串",
        "- 第 2 条要求在 tests 里没有对应的测试",
        "- 第 3 条要求在 tests 里没有对应的测试",
        "- 第 4 条要求在 tests 里没有对应的测试",
    ]
    assert all("requirement 注明序号" in line for line in lines[1:])


@pytest.mark.parametrize("text, expected", [
    ("抱歉，我现在没法出题。", "找不到题族的 JSON 对象"),
    ("[1, 2, 3]", "找不到题族的 JSON 对象"),
    ('{"theme": "x"', "不完整或者格式有错"),
    ("好的。\n\n", "找不到题族的 JSON 对象"),
    ('{"note": "只有说明"}', "不是题族"),
    ('{"family": {"note": "还是没有题族的字段"}}', "不是题族"),
    (None, "回复不是字符串"),
    (b"{}", "回复不是字符串"),
], ids=["prose", "array", "truncated", "blank", "not-a-family", "nested-not-a-family", "none", "bytes"])
def test_parse_family_rejects_a_reply_that_is_not_a_family(text, expected):
    with pytest.raises(ValueError, match=expected):
        parse.parse_family(text)


def test_a_truncated_reply_whose_code_contains_empty_dicts_is_reported_as_incomplete():
    """参考解里有 ``self._items = {}``：回复被截断时它自己能解析成一个 JSON 对象，但不能把它当题族。"""
    text = reply(family())
    with pytest.raises(ValueError, match="不完整或者格式有错.*被截断了"):
        parse.parse_family(text[:text.index('"tests"')])


def test_every_truncated_prefix_of_a_good_reply_is_rejected_as_incomplete():
    """回复写到一半被截断：逐个字符截，没有一个前缀被放行；诊断一律是"不完整、多半被截断"，
    不能拿里面已经写完的小对象（tests 的条目）去报"键名不对"。"""
    text = reply(family())
    kinds = Counter()
    for cut in range(1, len(text)):
        with pytest.raises(ValueError) as caught:
            parse.parse_family(text[:cut], slug="state_inventory")
        message = str(caught.value)
        assert "不是题族" not in message and "找不到题族的 JSON 对象" not in message
        assert "不完整或者格式有错" in message and "截断" in message
        kinds[message.split("：")[0]] += 1
    assert sum(kinds.values()) == len(text) - 1


def test_a_trailing_comma_is_reported_as_a_comma_not_as_wrong_keys():
    text = reply(family()).replace('}, "take":', '},, "take":', 1)
    with pytest.raises(ValueError) as caught:
        parse.parse_family(text)
    message = str(caught.value)
    assert "不完整或者格式有错" in message and "多了一个逗号" in message
    column = text.index("},, ") + 3                       # 多出来的那个逗号在回复里的列
    assert f"解析到第 1 行第 {column} 列时出错，多半是对象的最后一项后面多了一个逗号" in message
    assert "不是题族" not in message and "实际的键是" not in message


def test_a_prose_brace_before_a_valid_json_object_does_not_confuse_the_diagnosis():
    """对象前面的话里带着花括号：能解析的题族照常取到；解析不了的那次不抢着下结论。"""
    assert parse.parse_family("题族{见下}\n" + reply(family())) == family()
    with pytest.raises(ValueError, match="不是题族"):
        parse.parse_family('{说明: 见下} 以及 {"note": "x"}')
    with pytest.raises(ValueError, match="不完整或者格式有错"):
        parse.parse_family("题族{见下}\n" + reply(family())[:-40])


def test_a_family_wrapped_in_an_extra_object_is_unwrapped():
    text = json.dumps({"family": family()}, ensure_ascii=False)
    assert parse.parse_family(text) == family()


def test_requirement_numbers_only_count_numbering_at_the_start_of_a_line():
    assert parse.requirement_numbers(STOCK_INSTRUCTION) == [1, 2, 3, 4]
    numbered = "X 管理缓存（容量默认 16）。\n1）a\n  2) b\n要求：3）c\n  4）d（细节见 7））\n    按 5）处理"
    assert parse.requirement_numbers(numbered) == [1, 2, 3, 4]


def test_numbers_inside_parentheses_in_the_instruction_are_not_requirements():
    """"（容量默认 16）""空库存时返回 0）"是题面里的普通数字，不是第 16 条、第 0 条要求。"""
    for extra in ("（容量默认 16）", "（空库存时返回 0）"):
        item = family()
        item["instruction"] = item["instruction"].replace("并缓存 total() 的结果。", f"并缓存 total() 的结果{extra}。")
        assert parse.parse_family(reply(item)) == item


def test_every_existing_hard_task_instruction_still_counts_as_numbered():
    """序号只认行首之后，现有 40 篇题面数出来的条数不能变：都是从 1 起连续，而且不少于 4 条。"""
    for spec in HARD3_SPECS:
        numbers = parse.requirement_numbers(spec["instruction"])
        assert numbers == list(range(1, len(numbers) + 1)) and len(numbers) >= 4, spec["task_id"]


def test_requirements_crammed_onto_one_line_are_asked_to_be_split():
    item = family()
    item["instruction"] = " ".join(item["instruction"].split("\n")[1:])       # 四条要求挤在同一行
    with pytest.raises(ValueError) as caught:
        parse.parse_family(reply(item))
    assert "每条要求要单独占一行，序号写在行首：第 2 条起的序号写在了行中间" in str(caught.value)
    item["instruction"] = "Stock 管理库存。要求 " + item["instruction"]
    with pytest.raises(ValueError, match="各条要求挤在同一行里，分不出是哪几条"):
        parse.parse_family(reply(item))


# ── 要求的条数（发现 1） ──


def _three_requirements(item):
    item["instruction"] = "\n".join(line for line in item["instruction"].split("\n") if not line.startswith("4）"))
    del item["tests"]["total"]
    item["mutants"] = item["mutants"][:3]


def _five_requirements(item):
    item["instruction"] += "\n5）total() 的结果不小于 0。"
    item["tests"]["nonneg"] = {"requirement": 5, "body": "def test_nonneg():\n    assert Stock().total() == 0\n"}
    item["mutants"].append({"requirement": 5, "old": "            self._total = sum(self._items.values())\n",
                            "new": "            self._total = -sum(self._items.values())\n", "idea": "符号写反"})


def test_a_family_with_too_few_or_too_many_requirements_is_rejected():
    """条数得有人把关：教师给出 3 条或 7 条要求的题族，不能带着零发现进题池（审计不查这一项）。"""
    with pytest.raises(ValueError) as caught:
        parse.parse_family(edit(_three_requirements))
    assert str(caught.value) == "回复不是合格的题族，共 1 处问题：\n- 题面里编号的行为要求有 3 条，应在 4 到 6 条之间"
    seven = family()
    seven["instruction"] += "\n5）多余的一条。\n6）再多一条。\n7）还有一条。"
    with pytest.raises(ValueError, match="题面里编号的行为要求有 7 条，应在 4 到 6 条之间"):
        parse.parse_family(reply(seven))
    five = parse.parse_family(edit(_five_requirements))
    assert parse.requirement_numbers(five["instruction"]) == [1, 2, 3, 4, 5]        # 5 条在范围内，放行


def test_parse_family_checks_the_count_the_slug_calls_for_when_it_is_given_the_slug():
    """条数由 slug 决定：把 slug 传给 parse_family，就要求正好 requirement_count(slug) 条；不传只查 4 到 6。"""
    assert themes.requirement_count == prompts.requirement_count
    assert themes.requirement_count("state_inventory") == 4
    assert parse.parse_family(reply(family()), slug="state_inventory") == family()
    assert parse.parse_family(reply(family())) == family()
    wanted = themes.requirement_count("state_calendar")
    assert wanted == 5
    with pytest.raises(ValueError) as caught:
        parse.parse_family(reply(family()), slug="state_calendar")
    assert str(caught.value) == ("回复不是合格的题族，共 1 处问题：\n- 题面里编号的行为要求有 4 条，这道题要正好 5 条"
                                 "（条数由模块名决定）：请补上 1 条，每条都要有自己的测试和 mutant")
    assert parse.parse_family(edit(_five_requirements), slug="state_calendar")
    with pytest.raises(ValueError) as caught:
        parse.parse_family(edit(_five_requirements), slug="state_inventory")
    assert str(caught.value) == ("回复不是合格的题族，共 1 处问题：\n- 题面里编号的行为要求有 5 条，这道题要正好 4 条"
                                 "（条数由模块名决定）：请删掉或合并 1 条，tests、mutants 里对应的也去掉")
    for slug in ("State-Inventory", "", "s3/x"):
        with pytest.raises(ValueError, match="slug 只能用小写字母"):
            parse.parse_family(reply(family()), slug=slug)


# ── 语法关按 Python 3.10 判（发现 7） ──


@pytest.mark.parametrize("code, newer", [
    ("type Alias = int\n", "3.12"),
    ("try:\n    pass\nexcept* ValueError:\n    pass\n", "3.11"),
    ("def f[T](x): return x\n", "3.12"),
], ids=["type-statement", "except-star", "generic-def"])
def test_syntax_that_python_3_10_cannot_run_is_rejected(code, newer):
    """服务器上的解释器比 3.10 新时，也要拦住 3.10 跑不了的程序（解释器自己报的版本号因版本而异，
    所以只在它报了的时候才要求提示里带上）。"""
    def change(item):
        item["fixed"] = STOCK_FIXED + code
    with pytest.raises(ValueError) as caught:
        parse.parse_family(edit(change))
    message = str(caught.value)
    with pytest.raises(SyntaxError) as syntax:
        ast.parse(code, feature_version=(3, 10))
    # 行号按整个文件数（这段代码接在参考解后面）
    assert "fixed 不是合法的 Python：" in message
    assert f"（第 {STOCK_FIXED.count(chr(10)) + syntax.value.lineno} 行）" in message
    if "only supported in Python" in message:
        assert f"这是 Python {newer} 才有的语法，程序要能在 3.10 上运行" in message


@pytest.mark.parametrize("code", [
    "def f(x):\n    return (n := x) + 1\n",
    "def f(x):\n    match x:\n        case 1:\n            return 1\n",
    "async def f():\n    async with lock:\n        pass\n",
    "x = f'{1=}'\n",
])
def test_syntax_that_python_3_10_can_run_is_not_rejected(code):
    assert parse._syntax_error(code) is None


def test_the_3_10_gate_still_accepts_every_existing_hard_task():
    for spec in HARD3_SPECS:
        for code in (spec["fixed"], spec["public_test"], *spec["private_tests"].values()):
            assert parse._syntax_error(code) is None, spec["task_id"]


# ── 按格式拼出来的题真的能判分 ──


def assemble(parsed, slug, buggy):
    """照提示里说的拼一道题：隐藏测试文件开头自动加 import pytest 和 from <module> import *，公开测试只加后者。"""
    module = f"s3_{slug}"
    return {
        "task_id": f"s3_{slug}_001", "source_group": module, "split": "train", "bug_family": "state",
        "difficulty": "L3", "instruction": parsed["instruction"], "module": f"{module}.py", "buggy": buggy,
        "fixed": parsed["fixed"], "public_test": f"from {module} import *\n\n{parsed['public_test']}",
        "private_tests": {f"test_{label}.py": f"import pytest\nfrom {module} import *\n\n{test['body']}"
                          for label, test in parsed["tests"].items()},
        "stratum": "synth-state", "origin": "synth", "direction": "state",
    }


def grade_sources(spec, sources):
    """在沙箱里真判分：sources 是 {名字: 主文件源码}，都按同一道题判，返回各自的结论。"""
    with tempfile.TemporaryDirectory(prefix="synth-prompts-") as directory:
        root = Path(directory)
        build_all(root / "tasks", specs=[spec])
        task = TaskStore(root / "tasks").load(spec["task_id"])
        return {name: pools.grade_program(task, spec, source, root / f"grade-{index}")
                for index, (name, source) in enumerate(sources.items())}


def test_a_family_written_per_the_format_grades_in_the_sandbox():
    """提示承诺的文件拼法（自动加的两行 import、不写 import 的测试）在真的判分器里成立：
    参考解全过；每个 mutant 单独出现时公开测试照过、隐藏测试挂；全部缺陷一起的母题也是公开过、隐藏挂。"""
    parsed = parse.parse_family("```json\n" + reply(family()) + "\n```")
    fixed = parsed["fixed"]
    assert all("import" not in test["body"] for test in parsed["tests"].values())
    assert "import" not in parsed["public_test"]
    variants = {f"mutant-{index}": fixed.replace(mutant["old"], mutant["new"])
                for index, mutant in enumerate(parsed["mutants"], 1)}
    mother = fixed
    for mutant in parsed["mutants"]:
        assert mother.count(mutant["old"]) == 1         # 各处缺陷改动的行互不重叠，顺序随便
        mother = mother.replace(mutant["old"], mutant["new"])
    spec = assemble(parsed, "state_inventory", mother)
    assert prompt_estimate(spec) + rough_tokens(fixed) <= 1900
    results = grade_sources(spec, {"reference": fixed, "mother": mother, **variants})
    assert results["reference"]["strict_pass"] is True, results["reference"]
    for name in ["mother", *variants]:
        assert results[name]["public_pass"] is True, (name, results[name])
        assert results[name]["private_pass"] is False, (name, results[name])


# ── 隐私 ──


def test_prompts_and_prompt_sources_carry_no_private_details():
    """提示会发给教师、源码会进公开仓库：显卡型号、个人路径、服务商名、密钥一个都不能有。

    规则直接取 scripts/privacy_check.py 里的 CONTENT_PATTERNS：推送前的隐私审查用的就是它。
    """
    patterns = [re.compile(pattern, re.IGNORECASE) for _, pattern in privacy_check.CONTENT_PATTERNS]
    texts = []
    for direction in themes.DIRECTION_NAMES:
        for domain in themes.DOMAINS[:3]:
            slug = themes.slug_of(direction, domain)
            theme = (themes.DIRECTIONS[direction], domain)
            example = prompts.pick_example(direction, slug)
            texts += [message["content"] for message in prompts.generation_messages(theme, example, slug)]
            texts += [message["content"] for message in prompts.repair_messages(
                reply(family()), ["问题一。", {"code": "x", "message": "问题二。"}], theme, slug)]
    for name in ("themes", "prompts", "parse", "__init__"):
        texts.append((Path(themes.__file__).parent / f"{name}.py").read_text(encoding="utf-8"))
    texts.append(Path(__file__).read_text(encoding="utf-8"))
    assert len(texts) > 30
    for text in texts:
        for pattern in patterns:
            assert not pattern.search(text), pattern.pattern
