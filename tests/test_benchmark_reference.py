"""Every built-in reference patch must pass the real public/private grader."""
import pytest

from agentlab.benchmark.builder import SPECS, build_all
from agentlab.benchmark.grader import grade
from agentlab.benchmark.task_store import TaskStore


@pytest.mark.parametrize("spec", SPECS, ids=[spec["task_id"] for spec in SPECS])
def test_builtin_reference_passes_real_grader(tmp_path, spec):
    build_all(tmp_path / "tasks", [spec])
    task = TaskStore(tmp_path / "tasks").load(spec["task_id"])
    result, details = grade(task, task.reference_patch, tmp_path / "grades")
    assert result.strict_pass, details
