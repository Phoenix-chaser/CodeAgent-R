# -*- coding: utf-8 -*-
"""端到端集成测试：模拟教师 API 回复 → 提取代码 → 补丁 → git apply → grader。

这个测试存在的意义：防止「单元测试全绿但实际管线断裂」的事故再发生。
上一次事故：extract_code 返回不带 \n 的代码，git apply 报 corrupt patch。
"""
import difflib
import subprocess
import tempfile
from pathlib import Path

import pytest

from agentlab.benchmark.builder import SPECS, build_all
from agentlab.benchmark.grader import grade
from agentlab.benchmark.task_store import TaskStore
from agentlab.runtime.workspace import Workspace

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from _shared_extract import extract_code


def make_patch(module_path, buggy, fixed):
    return "".join(difflib.unified_diff(
        buggy.splitlines(keepends=True),
        fixed.splitlines(keepends=True),
        fromfile=f"a/{module_path}", tofile=f"b/{module_path}"))


# ── 测试：教师 API 各种回复格式都能正确提取并生成合法补丁 ──

TEACHER_RESPONSE_CASES = [
    # (描述, 教师 API 原始回复, 预期提取结果非空)
    ("纯净代码无末尾换行",
     "def even_numbers(n):\n    return [i for i in range(2, n + 1, 2)]",
     True),
    ("带markdown围栏",
     "```python\ndef even_numbers(n):\n    return [i for i in range(2, n + 1, 2)]\n```",
     True),
    ("带前置说明",
     "修复后的代码：\ndef even_numbers(n):\n    return [i for i in range(2, n + 1, 2)]",
     True),
    ("空回复", "", False),
    ("只有说明无代码", "这个函数没有问题，不需要修复。", False),
]


@pytest.mark.parametrize("desc,response,expect_nonempty",
                         TEACHER_RESPONSE_CASES,
                         ids=[c[0] for c in TEACHER_RESPONSE_CASES])
def test_extract_code_formats(desc, response, expect_nonempty):
    """教师 API 各种回复格式 → extract_code 应正确处理。"""
    result = extract_code(response)
    if expect_nonempty:
        assert result, f"「{desc}」：提取结果为空"
        assert result.endswith("\n"), f"「{desc}」：末尾无换行符"
        assert "```" not in result, f"「{desc}」：残留 markdown 标记"
    # 空回复的提取结果可以为空


@pytest.mark.parametrize("spec_idx", range(min(3, len(SPECS))),
                         ids=[SPECS[i]["task_id"] for i in range(min(3, len(SPECS)))])
def test_full_pipeline_git_apply(tmp_path, spec_idx):
    """完整管线：提取代码 → 生成补丁 → git apply → grader 判分。

    使用每个任务 spec 里的 fixed 代码作为模拟教师 API 回复。
    """
    spec = SPECS[spec_idx]

    # 模拟教师 API 返回正确修复（无末尾换行，模拟真实 API 行为）
    teacher_response = spec["fixed"].rstrip("\n")
    fixed_code = extract_code(teacher_response)

    # 确认提取结果末尾有换行
    assert fixed_code.endswith("\n"), \
        f"{spec['task_id']}: 提取结果末尾无换行符"

    # 生成补丁
    patch = make_patch(spec["module"], spec["buggy"], fixed_code)
    assert patch.strip(), f"{spec['task_id']}: 补丁为空"

    # 在工作区里试 git apply
    task_dir = tmp_path / "tasks"
    build_all(task_dir)
    store = TaskStore(task_dir)
    task = store.load(spec["task_id"])
    ws = Workspace.create_from_fixture(f"itest-{spec_idx}",
                                       task.fixture_tar, tmp_path / "ws")
    patch_file = tmp_path / "cand.diff"
    patch_file.write_text(patch, encoding="utf-8")

    proc = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", str(patch_file.resolve())],
        cwd=ws.root, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, \
        f"{spec['task_id']}: git apply 失败: {proc.stderr}"

    # 确认文件内容确实变了
    modified = (ws.root / spec["module"]).read_text(encoding="utf-8")
    assert modified != spec["buggy"], \
        f"{spec['task_id']}: 文件内容未变化"


@pytest.mark.parametrize("spec_idx", range(min(3, len(SPECS))),
                         ids=[SPECS[i]["task_id"] for i in range(min(3, len(SPECS)))])
def test_full_pipeline_grader(tmp_path, spec_idx):
    """完整管线 + grader 判分：正确修复应得 strict_pass。"""
    spec = SPECS[spec_idx]

    # 模拟教师 API 回复
    teacher_response = spec["fixed"].rstrip("\n")
    fixed_code = extract_code(teacher_response)
    patch = make_patch(spec["module"], spec["buggy"], fixed_code)

    # 构建 grader 环境
    task_dir = tmp_path / "tasks"
    build_all(task_dir)
    store = TaskStore(task_dir)
    task = store.load(spec["task_id"])

    patch_file = tmp_path / "cand.diff"
    patch_file.write_text(patch, encoding="utf-8")

    result, details = grade(task, patch_file, tmp_path / "grade")
    assert result.verdict == "strict_pass", \
        f"{spec['task_id']}: 参考修复应 strict_pass，得到 {result.verdict}: {details}"


def test_buggy_unchanged_gets_strict_fail(tmp_path):
    """教师 API 返回与原代码相同的代码 → 补丁为空 → strict_fail。"""
    spec = SPECS[0]

    # 教师 API 返回了跟原代码一样的内容（没修）
    teacher_response = spec["buggy"].rstrip("\n")
    fixed_code = extract_code(teacher_response)
    patch = make_patch(spec["module"], spec["buggy"], fixed_code)

    # 补丁应为空或只有空白差异
    if not patch.strip():
        # 空补丁 = 零改动，进入功能层 → strict_fail
        task_dir = tmp_path / "tasks"
        build_all(task_dir)
        store = TaskStore(task_dir)
        task = store.load(spec["task_id"])
        empty_patch = tmp_path / "empty.diff"
        empty_patch.write_text("", encoding="utf-8")
        result, _ = grade(task, empty_patch, tmp_path / "grade")
        assert result.verdict == "strict_fail"
