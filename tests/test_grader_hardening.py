# -*- coding: utf-8 -*-
"""判分加固回归：通过必须意味着“预期的那批测试真的跑了，而且都过了”。

评审指出的漏洞：旧 grader 只看 pytest 退出码，候选模块在顶层 os._exit(0)
就能让公开层和私有层同时“通过”。这里逐项锁住修复，并验证隔离实际生效。
"""
import difflib
import os
import subprocess
import time

import pytest

from agentlab.benchmark import TaskStore, build_all, grade
from agentlab.benchmark import grader
from agentlab.benchmark.task_store import LoadedTask
from agentlab.runtime import isolation

TID = "mini_calc_offbyone_001"
MODULE = "mini_calc.py"
BUGGY = "def even_numbers(n):\n    return [i for i in range(2, n, 2)]\n"
FIXED = "def even_numbers(n):\n    return [i for i in range(2, n + 1, 2)]\n"

# 测试的参数化列表来自被测模块：候选把它改短，收集到的用例就会变少
SHRINK = dict(
    task_id="probe_shrink_001", source_group="probe_shrink", split="smoke",
    bug_family="probe", difficulty="L1",
    instruction="让 solve(x) 对 CASES 里的每个 x 返回 x 的两倍。",
    module="probe_shrink.py",
    buggy="CASES = [1, 2, 3]\n\n\ndef solve(x):\n    return x\n",
    fixed="CASES = [1, 2, 3]\n\n\ndef solve(x):\n    return x * 2\n",
    public_test=("import pytest\n\nimport probe_shrink\n\n\n"
                 "@pytest.mark.parametrize('x', probe_shrink.CASES)\n"
                 "def test_solve(x):\n    assert probe_shrink.solve(x) == x * 2\n"),
    private_tests={"test_more.py": (
        "import pytest\n\nimport probe_shrink\n\n\n"
        "@pytest.mark.parametrize('x', [4, 5])\n"
        "def test_more(x):\n    assert probe_shrink.solve(x) == x * 2\n")},
)


def _patch(path, old, new):
    return "".join(difflib.unified_diff(
        old.splitlines(keepends=True), new.splitlines(keepends=True),
        fromfile=f"a/{path}", tofile=f"b/{path}"))


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> TaskStore:
    tasks = tmp_path_factory.mktemp("hardening-tasks")
    build_all(tasks)
    return TaskStore(tasks)


def _grade(tmp_path, task: LoadedTask, diff_text: str, name="candidate.diff"):
    diff = tmp_path / name
    diff.write_text(diff_text, encoding="utf-8")
    return grade(task, diff, tmp_path / "work")


def _probes(tmp_path, filename) -> list[str]:
    return sorted(path.read_text() for path in
                  (tmp_path / "work").glob(f"grade-*/scratch/tmp/{filename}"))


# ── 汇总行与 junit 的判定逻辑（不起子进程） ─────────────────

@pytest.mark.parametrize("stdout, expected", [
    ("..\n2 passed in 0.01s\n", {"passed": 2}),
    ("2 passed, 1 warning in 0.02s", {"passed": 2, "warnings": 1}),
    ("1 passed, 1 skipped, 2 deselected, 1 xfailed in 0.1s",
     {"passed": 1, "skipped": 1, "deselected": 2, "xfailed": 1}),
    ("3 failed, 1 error in 0.1s", {"failed": 3, "errors": 1}),
    ("no tests ran in 0.01s", {}),
    ("collected 0 items / 1 error", None),
    ("", None),
])
def test_parse_pytest_summary(stdout, expected):
    assert grader.parse_pytest_summary(stdout) == expected


def test_parse_junit_requires_real_cases(tmp_path):
    path = tmp_path / "junit.xml"
    assert grader.parse_junit(path) is None                 # 进程被杀，文件没写出来
    path.write_text("<testsuites><testsuite>"
                    '<testcase classname="tests_public.test_public" name="test_a"/>'
                    '<testcase classname="t" name="test_b"><skipped message="x"/></testcase>'
                    '<testcase classname="t" name="test_c"><failure>x</failure></testcase>'
                    "</testsuite></testsuites>", encoding="utf-8")
    assert grader.parse_junit(path) == {
        "tests_public.test_public::test_a": "passed", "t::test_b": "skipped",
        "t::test_c": "failure"}
    path.write_text("<broken", encoding="utf-8")
    assert grader.parse_junit(path) is None                 # 进程中途被杀，XML 只写了一半


def _layer(**changes):
    base = dict(expected=frozenset({"a::t1", "a::t2"}), exit_status="ok", returncode=0,
                duration_ms=1, summary={"passed": 2},
                cases={"a::t1": "passed", "a::t2": "passed"}, tail="")
    return grader.LayerRun(**{**base, **changes})


def test_layer_passes_only_when_every_signal_agrees():
    assert _layer().passed
    assert not _layer(summary=None).passed                  # 退出码 0，但没有汇总行
    assert not _layer(cases=None).passed                    # junit 没有写出来
    assert not _layer(summary={"passed": 1}).passed         # 收集到的用例变少
    assert not _layer(summary={"passed": 2, "skipped": 1}).passed
    assert not _layer(summary={"passed": 2, "errors": 1}).passed
    assert not _layer(cases={"a::t1": "passed", "a::t2": "skipped"}).passed
    assert not _layer(cases={"a::t1": "passed", "a::t2": "failure"}).passed
    assert not _layer(cases={"a::t1": "passed", "a::other": "passed"}).passed
    assert not _layer(exit_status="nonzero_exit", returncode=1).passed
    assert not _layer(exit_status="timeout").passed


# ── 真实判分：每种投机方式都必须拿不到 strict_pass ─────────────

def test_reference_fix_still_passes_and_reports_evidence(tmp_path, store):
    result, details = _grade(tmp_path, store.load(TID), _patch(MODULE, BUGGY, FIXED))
    execution = details["layers"]["execution"]
    landlock = execution["isolation"] == "landlock"
    assert result.strict_pass, details
    assert landlock is isolation.probe_landlock().available   # 如实记录，而不是声明
    assert execution["workspace_readonly"] is landlock
    assert execution["inherited_environment"] is False
    assert execution["reference_passed"] == {"tests_public": True, "tests_private": True}
    assert details["layers"]["functional"]["public"]["executed_tests"] == \
        execution["expected_tests"]["tests_public"] > 0


def test_module_early_exit_is_not_a_pass(tmp_path, store):
    result, details = _grade(tmp_path, store.load(TID), _patch(
        MODULE, BUGGY, "import os\nos._exit(0)  # pytest 以 0 退出，但一个测试都没跑\n"))
    public = details["layers"]["functional"]["public"]
    assert result.strict_pass is False
    assert public["exit_status"] == "ok"                     # 退出码确实骗过了旧判定
    assert public["summary"] is None and public["executed_tests"] == 0
    assert public["expected_tests"] > 0 and public["missing_tests"]


def test_skipped_tests_are_not_a_pass(tmp_path, store):
    """被测函数在被调用时调用 pytest.skip：退出码 0、汇总行 1 skipped。"""
    result, details = _grade(tmp_path, store.load(TID), _patch(
        MODULE, BUGGY,
        "import pytest\n\n\ndef even_numbers(n):\n    pytest.skip('cannot decide')\n"))
    public = details["layers"]["functional"]["public"]
    assert result.strict_pass is False
    assert public["exit_status"] == "ok"
    assert public["summary"]["skipped"] > 0 and public["non_passing_tests"]


def test_shrunk_parametrization_is_not_a_pass(tmp_path):
    build_all(tmp_path / "tasks", [SHRINK])
    task = TaskStore(tmp_path / "tasks").load(SHRINK["task_id"])
    result, details = _grade(tmp_path, task, _patch(
        SHRINK["module"], SHRINK["buggy"],
        "CASES = [2]\n\n\ndef solve(x):\n    return {2: 4}.get(x, x)\n"))
    public = details["layers"]["functional"]["public"]
    assert result.strict_pass is False
    assert public["exit_status"] == "ok"
    assert (public["executed_tests"], public["expected_tests"]) == (1, 3)
    assert len(public["missing_tests"]) == 2


@pytest.mark.parametrize("name", ["conftest.py", "pytest.ini", "sitecustomize.py", "helper.py"])
def test_only_the_target_module_is_editable(tmp_path, store, name):
    payloads = {
        "conftest.py": ("import pytest\n\n\ndef pytest_collection_modifyitems(config, items):\n"
                        "    for item in items:\n        item.add_marker(pytest.mark.skip)\n"),
        "pytest.ini": "[pytest]\naddopts = -p no:skipping\n",
        "sitecustomize.py": "import os\nos._exit(0)\n",
        "helper.py": "x = 1\n",
    }
    result, details = _grade(tmp_path, store.load(TID), _patch(name, "", payloads[name]))
    assert result.verdict == "invalid"
    assert details["layers"]["legality"]["illegal"] == [name]


def test_rename_only_patch_is_rejected(tmp_path, store):
    diff = f"diff --git a/{MODULE} b/{MODULE}\nrename from {MODULE}\nrename to gone.py\n"
    result, details = _grade(tmp_path, store.load(TID), diff)
    assert result.verdict == "invalid"
    assert details["layers"]["legality"]["unsupported_patch_lines"]


def test_workspace_mutation_during_public_run_is_invalid(tmp_path, store, monkeypatch):
    monkeypatch.setenv("AGENTLAB_SANDBOX", "off")           # 没有 Landlock 时写入才会成功
    result, details = _grade(tmp_path, store.load(TID), _patch(
        MODULE, BUGGY,
        "from pathlib import Path\n"
        "Path('tests_public/test_public.py').write_text('def test_basic():\\n    assert True\\n')\n"
        + FIXED))
    assert result.verdict == "invalid"
    assert details["layers"]["security"]["workspace_mutated"] == ["tests_public/test_public.py"]


# ── 隔离在真实判分路径里生效 ─────────────────────────────────

def test_private_tests_are_absent_during_public_run(tmp_path, store):
    result, details = _grade(tmp_path, store.load(TID), _patch(
        MODULE, BUGGY,
        "import os\n\n"
        "with open(os.path.join(os.environ['TMPDIR'], 'probe.txt'), 'a') as handle:\n"
        "    handle.write('private\\n' if os.path.exists('tests_private') else 'public\\n')\n"
        + FIXED))
    assert result.strict_pass, details
    assert _probes(tmp_path, "probe.txt") == ["public\nprivate\n"]


def test_secrets_do_not_reach_the_candidate(tmp_path, store, monkeypatch):
    monkeypatch.setenv("TEACHER_API_KEY", "should-not-leak")
    result, details = _grade(tmp_path, store.load(TID), _patch(
        MODULE, BUGGY,
        "import os\n\n"
        "try:\n"
        "    open('/proc/self/environ', 'rb').read()\n"
        "    proc = 'allowed'\n"
        "except OSError as error:\n"
        "    proc = f'denied:{error.errno}'\n"
        "with open(os.path.join(os.environ['TMPDIR'], 'env.txt'), 'w') as handle:\n"
        "    handle.write(f\"env={os.environ.get('TEACHER_API_KEY')}|proc={proc}\")\n"
        + FIXED))
    assert result.strict_pass, details
    probe = _probes(tmp_path, "env.txt")[0]
    assert probe.startswith("env=None|")
    if details["layers"]["execution"]["isolation"] == "landlock":
        assert probe.endswith("proc=denied:13")             # /proc 整体不可读


def test_sandbox_startup_failure_is_an_infrastructure_error_with_the_cause(
        tmp_path, store, monkeypatch):
    """沙箱里的解释器没起来时，报错里要带它的 stderr，而不是笼统的“没有测试结果”。"""
    from agentlab.runtime.exec import ExecOutcome

    monkeypatch.setattr(grader, "_REFERENCE_CACHE", {})     # 参考解必须重新跑，才会经过替换掉的函数
    monkeypatch.setattr(grader, "run_sandboxed", lambda policy, command, cwd, timeout: ExecOutcome(
        exit_status="nonzero_exit", returncode=127, stdout="", duration_ms=3,
        stderr="python: error while loading shared libraries: libc.so.6: cannot open shared "
               "object file"))
    with pytest.raises(grader.GradeInfrastructureError, match="libc.so.6"):
        _grade(tmp_path, store.load(TID), _patch(MODULE, BUGGY, FIXED))


def test_timeout_kills_the_whole_process_group(tmp_path, store, monkeypatch):
    task = store.load(TID)
    _grade(tmp_path, task, _patch(MODULE, BUGGY, FIXED), name="warmup.diff")
    expression = "os.path.join(os.environ['TMPDIR'], 'heartbeat.log')"
    writer = (f"import os, time\nwhile True:\n    open({expression}, 'a').write('x')\n"
              "    time.sleep(0.02)\n")
    monkeypatch.setattr(grader, "PYTEST_TIMEOUT", 1.5)
    with pytest.raises(subprocess.TimeoutExpired):
        _grade(tmp_path, task, _patch(
            MODULE, BUGGY,
            "import subprocess, sys, time\n"
            f"subprocess.Popen([sys.executable, '-c', {writer!r}])\n"
            "time.sleep(30)\n"))
    beats = [path for path in (tmp_path / "work").glob("grade-*/scratch/tmp/heartbeat.log")]
    assert beats, "the background writer never started"
    size = beats[-1].stat().st_size
    time.sleep(0.5)
    assert beats[-1].stat().st_size == size                  # 没有残留的写入者


def test_required_mode_refuses_without_landlock(tmp_path, store, monkeypatch):
    monkeypatch.setenv("AGENTLAB_SANDBOX", "required")
    monkeypatch.setattr(isolation, "probe_landlock",
                        lambda: isolation.SandboxProbe(0, "simulated: old kernel"))
    with pytest.raises(isolation.UnsupportedSandbox, match="simulated"):
        _grade(tmp_path, store.load(TID), _patch(MODULE, BUGGY, FIXED))


@pytest.mark.skipif(not hasattr(os, "geteuid"), reason="POSIX only")
def test_root_is_refused_unless_explicitly_allowed(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    with pytest.raises(isolation.SandboxError, match="as root"):
        isolation.build_policy(tmp_path, tmp_path / "scratch")
    monkeypatch.setenv("AGENTLAB_ALLOW_ROOT", "1")
    assert isolation.build_policy(tmp_path, tmp_path / "scratch", mode="process").mode \
        == "process"
