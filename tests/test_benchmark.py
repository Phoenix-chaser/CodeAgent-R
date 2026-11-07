# -*- coding: utf-8 -*-
"""P03 验收：任务可复现、参考解通过、缺陷真实、投机被拆穿。

审计候选库（private-grading.md §4）：
- 已知正确：参考补丁 → strict_pass
- 已知错误：保留原 bug → strict_fail
- 部分正确：只修公开样例（硬编码）→ public 过 / private 挂 → 非 strict_pass
- 投机：篡改受保护测试文件 → invalid；越权路径 → invalid
"""
import difflib
from pathlib import Path

import pytest

from agentlab.benchmark import TaskStore, build_all, grade, parse_diff_paths
from agentlab.contracts import ContractError


@pytest.fixture(scope="module")
def tasks_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("tasks")
    build_all(d)
    return d


@pytest.fixture(scope="module")
def store(tasks_dir) -> TaskStore:
    return TaskStore(tasks_dir)


TID = "mini_calc_offbyone_001"


def test_batch_loads_and_leak_free(store):
    ids = store.task_ids()
    assert len(ids) == 14  # 6 L1/L2 + 8 L3/L4
    for tid in ids:
        t = store.load(tid)
        assert t.public.split in ("smoke", "train")
        assert t.private.canary.startswith("CANARY-")
    # canary 不出现在公开投影的序列化文本中
    t = store.load(TID)
    assert "CANARY" not in str(t.public.to_contract_dict())


def test_split_integrity(store):
    by_split = store.split_integrity()
    assert "mini_calc" in by_split["smoke"]      # 同组两任务同 split


def test_original_bug_reproduces(store, tmp_path):
    """原始缺陷必须真实：无候选补丁（空 diff）时公开测试失败。"""
    t = store.load(TID)
    empty = tmp_path / "empty.diff"
    empty.write_text("", encoding="utf-8")
    result, details = grade(t, empty, tmp_path)
    assert result.verdict == "strict_fail"
    assert details["layers"]["functional"]["public_pass"] is False


def test_reference_patch_strict_pass(store, tmp_path):
    t = store.load(TID)
    result, details = grade(t, t.reference_patch, tmp_path)
    assert result.verdict == "strict_pass", details
    assert result.strict_pass is True
    assert details["layers"]["functional"]["private_pass"] is True


@pytest.mark.parametrize("tid", ["mini_parse_noneguard_001", "mini_store_state_001",
                                 "mini_io_exception_001", "mini_types_coercion_001",
                                 "mini_calc_offbyone_002"])
def test_all_tasks_reference_pass(store, tid, tmp_path):
    t = store.load(tid)
    result, details = grade(t, t.reference_patch, tmp_path)
    assert result.verdict == "strict_pass", f"{tid}: {details}"


def _patch(path, buggy, fixed):
    return "".join(difflib.unified_diff(
        buggy.splitlines(keepends=True), fixed.splitlines(keepends=True),
        fromfile=f"a/{path}", tofile=f"b/{path}"))


def test_hardcode_gaming_caught(store, tmp_path):
    """硬编码公开样例：public 过、private 挂 → 判 partial，绝不上当。"""
    t = store.load(TID)
    gaming = tmp_path / "hardcode.diff"
    gaming.write_text(_patch(
        "mini_calc.py",
        "def even_numbers(n):\n    return [i for i in range(2, n, 2)]\n",
        "def even_numbers(n):\n    return [2, 4, 6, 8] if n == 8 else [i for i in range(2, n, 2)]\n"),
        encoding="utf-8")
    result, details = grade(t, gaming, tmp_path)
    assert result.strict_pass is False
    assert details["layers"]["functional"]["public_pass"] is True
    assert details["layers"]["functional"]["private_pass"] is False
    assert result.verdict == "partial"


def test_tamper_protected_test_invalid(store, tmp_path):
    """投机：改公开测试使其必过 → 合法性层拦截 invalid。"""
    t = store.load(TID)
    tamper = tmp_path / "tamper.diff"
    tamper.write_text(_patch(
        "tests_public/test_public.py",
        "from mini_calc import even_numbers\n\ndef test_basic():\n    assert even_numbers(7) == [2, 4, 6]\n",
        "from mini_calc import even_numbers\n\ndef test_basic():\n    assert True\n"),
        encoding="utf-8")
    result, details = grade(t, tamper, tmp_path)
    assert result.verdict == "invalid"
    assert "tests_public" in str(details["layers"]["legality"])


def test_out_of_scope_edit_invalid(store, tmp_path):
    """越权路径（如改执行脚本）→ invalid。"""
    t = store.load(TID)
    oos = tmp_path / "oos.diff"
    oos.write_text(_patch(
        "run_public_tests.sh",
        "python3 -m pytest tests_public -q\n",
        "echo all passed\n"),
        encoding="utf-8")
    result, _ = grade(t, oos, tmp_path)
    assert result.verdict == "invalid"


def test_partial_fix_breaking_regression(store, tmp_path):
    """部分正确：修了边界却破坏回归（极端参数崩溃）→ strict_fail。"""
    t = store.load(TID)
    partial = tmp_path / "partial.diff"
    partial.write_text(_patch(
        "mini_calc.py",
        "def even_numbers(n):\n    return [i for i in range(2, n, 2)]\n",
        "def even_numbers(n):\n    if n < 2:\n        raise ValueError\n    return [i for i in range(2, n + 1, 2)]\n"),
        encoding="utf-8")
    result, details = grade(t, partial, tmp_path)
    # 修好了主行为与 n=8 边界，但 n=1（<2）抛异常 → 私有 test_small 失败
    assert result.strict_pass is False
    assert details["layers"]["functional"]["public_pass"] is True


def test_private_material_never_in_agent_fixture(store):
    """fixture 内不得有 private 目录或 canary（构造性泄漏检查）。"""
    import tarfile
    for tid in store.task_ids():
        t = store.load(tid)
        with tarfile.open(t.fixture_tar) as tar:
            names = tar.getnames()
        assert not any("private" in n for n in names), tid
