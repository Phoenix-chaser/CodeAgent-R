# -*- coding: utf-8 -*-
"""Hard batch h3 (opt-in): schema, shape, dedup and one graded task per part."""
import ast
import importlib
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

from agentlab.benchmark.builder import SPECS, build_all
from agentlab.benchmark.grader import grade
from agentlab.benchmark.hard_tasks_v3 import HARD3_MUTANTS, HARD3_SPECS
from agentlab.benchmark.pilot_tasks import PILOT_SPECS
from agentlab.benchmark.pilot_tasks_v2 import V2_SPECS
from agentlab.benchmark.pilot_tasks_v2b import V2B_SPECS
from agentlab.benchmark.pilot_tasks_v2c import V2C_SPECS
from agentlab.benchmark.pilot_tasks_v2d import V2D_SPECS
from agentlab.benchmark.protocol import validate_specs
from agentlab.benchmark.task_store import TaskStore

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import export_tasks_v2 as pools  # noqa: E402

PARTS = ("state", "tx", "exc", "lock", "cond")
# 题面不能被代码里的注释剧透：有问题代码和依赖文件的注释里不许出现这些标记
# （只看注释；todo 之类的变量名是正常写法）
HINT = re.compile(r"#.*\b(BUG|FIXME|TODO|XXX|HACK)\b", re.IGNORECASE)


def _part(name):
    return importlib.import_module(f"agentlab.benchmark.hard_tasks_v3_{name}").SPECS


def test_h3_is_the_five_parts_in_order():
    assert HARD3_SPECS == [spec for part in PARTS for spec in _part(part)]


def test_h3_has_two_variants_per_family():
    validate_specs(HARD3_SPECS)
    counts = Counter(spec["source_group"] for spec in HARD3_SPECS)
    assert len(counts) == 20 and set(counts.values()) == {2}
    assert all(spec["task_id"].startswith("h3_") for spec in HARD3_SPECS)
    assert all(spec["difficulty"] in {"L3", "L4"} for spec in HARD3_SPECS)
    for spec in HARD3_SPECS:
        functions = re.findall(r"^def test_", "\n".join(spec["private_tests"].values()), re.MULTILINE)
        assert len(functions) >= 3, spec["task_id"]


def test_h3_is_disjoint_from_the_default_pools():
    default = SPECS + PILOT_SPECS + V2_SPECS + V2B_SPECS + V2C_SPECS + V2D_SPECS
    ids = {spec["task_id"] for spec in default}
    groups = {spec["source_group"] for spec in default}
    modules = {spec["module"] for spec in default}
    asts = {ast.dump(ast.parse(spec["buggy"])) for spec in default}
    for spec in HARD3_SPECS:
        assert spec["task_id"] not in ids and spec["source_group"] not in groups
        assert spec["module"] not in modules
        assert ast.dump(ast.parse(spec["buggy"])) not in asts


@pytest.mark.parametrize("spec", HARD3_SPECS, ids=lambda spec: spec["task_id"])
def test_h3_is_valid_python_without_hints(spec):
    ast.parse(spec["fixed"])
    ast.parse(spec["public_test"])
    for content in [*spec["private_tests"].values(), *spec.get("extra_files", {}).values()]:
        ast.parse(content)
    for text in [spec["buggy"], *spec.get("extra_files", {}).values()]:
        assert not HINT.search(text), spec["task_id"]


def test_h3_mutants_anchor_the_reference():
    assert set(HARD3_MUTANTS) == {spec["task_id"] for spec in HARD3_SPECS}
    for spec in HARD3_SPECS:
        anchors = HARD3_MUTANTS[spec["task_id"]]
        assert len(anchors) >= 3
        for old, _ in anchors:
            assert spec["fixed"].count(old) == 1, spec["task_id"]


def _steps():
    return importlib.import_module("agentlab.benchmark.hard_tasks_v3_steps").STEP_SPECS


def test_h3_steps_are_single_defect_variants_of_their_parents():
    """难度阶梯：每个变体只比参考解多一处缺陷，题面、参考解、测试都沿用母题。"""
    steps = _steps()
    parents = {spec["task_id"]: spec for spec in HARD3_SPECS}
    assert len(steps) >= 120 and len(parents) == 40
    seen = set()
    for step in steps:
        parent = parents[step["variant_of"]]
        assert step["source_group"] == parent["source_group"] and step["module"] == parent["module"]
        for field in ("instruction", "fixed", "public_test", "private_tests", "stratum"):
            assert step[field] == parent[field]
        program = ast.dump(ast.parse(step["buggy"]))
        assert program not in seen and program != ast.dump(ast.parse(parent["buggy"]))
        seen.add(program)
    # 母题和它的变体同族同文件是有意为之；除此之外各题池仍须互不相交
    pools.check_pool_shape(steps, [("h3", HARD3_SPECS)])
    pools.check_pool_shape(HARD3_SPECS, [("h3s", steps)])
    pools.check_pool_shape(steps, [("builtin", SPECS)])


@pytest.mark.parametrize("part", (*PARTS, "step"))
def test_h3_sample_task_grades_as_designed(part, tmp_path):
    """每个部分抽第一道题：参考解全过，原程序过公开测试、挂隐藏测试。"""
    spec = _steps()[0] if part == "step" else _part(part)[0]
    task_dir, grade_dir = tmp_path / "tasks", tmp_path / "grade"
    grade_dir.mkdir()
    build_all(task_dir, specs=[spec])
    task = TaskStore(task_dir).load(spec["task_id"])
    reference, _ = grade(task, task.reference_patch, grade_dir / "reference")
    assert reference.strict_pass
    empty = grade_dir / "unchanged.diff"
    empty.write_text("", encoding="utf-8")
    unchanged, details = grade(task, empty, grade_dir / "unchanged")
    assert not unchanged.strict_pass
    assert details["layers"]["functional"]["private_pass"] is False
