# -*- coding: utf-8 -*-
"""三层判分 grader：冻结候选 → 全新环境 → 合法性/功能/鲁棒性。

可信路径（private-grading.md §1）：
    冻结候选 diff（P02 产物，a/ b/ 可移植补丁）
    → 合法性：解析 diff 触达路径 ⊆ allowed_paths；受保护文件哈希不变
    → 全新工作区 = fixture + git apply 候选 + 注入私有测试
    → 功能：pytest(公开) / pytest(私有) 子进程黑盒执行
    → 鲁棒性：私有测试中的边界/回归子集
Agent 永远不接触 private/；评分不读取 Agent 活动工作区。

通过判定不只看 pytest 退出码。参考解先在同一隔离环境里跑一遍，记下每一层
应当执行的测试集合；候选必须同时满足：

- 退出码为 0，且输出里有 pytest 的汇总行；
- 汇总行的 passed 数等于预期，没有 skipped / deselected / xfailed / error；
- junit XML 里的用例集合与预期完全一致，且全部通过。

提前 os._exit(0)、让收集缩水、把用例标成 skip，都会判为不通过。
被测代码在 runtime.isolation 的隔离环境里运行；私有测试在公开层跑完后才
注入，公开层执行期间磁盘上没有私有测试。
"""
from __future__ import annotations

import fnmatch
import hashlib
import importlib.metadata
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, replace
from pathlib import Path

from agentlab.benchmark.task_store import LoadedTask
from agentlab.contracts import GradeResult
from agentlab.runtime.isolation import (
    SandboxError, SandboxPolicy, build_policy, resolve_mode, run_sandboxed,
)
from agentlab.runtime.workspace import Workspace

PYTEST_TIMEOUT = 60
PUBLIC, PRIVATE = "tests_public", "tests_private"
PROTECTED = ("tests_public/test_public.py", "run_public_tests.sh")
# git 专有补丁特性：parse_diff_paths 只读 +++ 行，识别不到重命名、二进制与模式变更
_GIT_ONLY = re.compile(
    r"^(?:diff --git |GIT binary patch|(?:rename|copy|new file|deleted file|old mode|"
    r"new mode|similarity|dissimilarity) |index )")
_SUMMARY = re.compile(r"^(?P<body>.+?) in [0-9.]+s(?: \([^)]*\))?$")
_COUNTERS = re.compile(r"(\d+) (passed|failed|errors?|skipped|deselected|xfailed|xpassed|warnings?)")
_NOT_PASSING = ("failed", "errors", "skipped", "deselected", "xfailed", "xpassed")


@dataclass
class GraderError:
    kind: str            # patch_apply_failed | pytest_spawn_error | ...
    detail: str


class GradeInfrastructureError(RuntimeError):
    """判分环境自身出了问题（隔离建不起来、参考解无法收集）。不是候选的失败。"""


@dataclass
class LayerRun:
    """一层 pytest 的执行证据。passed 只在证据齐全且与预期一致时为真。"""
    expected: frozenset[str]
    exit_status: str
    returncode: int | None
    duration_ms: int
    summary: dict[str, int] | None
    cases: dict[str, str] | None     # 用例 → passed | failure | error | skipped
    tail: str
    stderr: str = ""                 # 沙箱里的进程没起来时，原因在这里

    @property
    def passed(self) -> bool:
        return (self.exit_status == "ok" and self.summary is not None and self.cases is not None
                and self.summary.get("passed", 0) == len(self.expected)
                and all(self.summary.get(name, 0) == 0 for name in _NOT_PASSING)
                and frozenset(self.cases) == self.expected
                and all(status == "passed" for status in self.cases.values()))

    def as_dict(self) -> dict:
        cases = self.cases or {}
        return {
            "passed": self.passed, "exit_status": self.exit_status,
            "returncode": self.returncode, "duration_ms": self.duration_ms,
            "summary": self.summary, "expected_tests": len(self.expected),
            "executed_tests": sum(status == "passed" for status in cases.values()),
            "missing_tests": sorted(self.expected - set(cases)),
            "unexpected_tests": sorted(set(cases) - self.expected),
            "non_passing_tests": sorted(name for name, status in cases.items()
                                        if status != "passed"),
            "stderr_tail": self.stderr[-300:],
        }


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def parse_diff_paths(diff_text: str) -> list[str]:
    """从 unified diff 提取被修改文件（b/ 侧相对路径）。"""
    paths = []
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            rel = line[len("+++ b/"):].strip()
            if rel and rel != "/dev/null":
                paths.append(rel)
    return paths


def parse_pytest_summary(stdout: str) -> dict[str, int] | None:
    """解析 `-q` 输出最后一行的汇总；不是汇总行时返回 None。"""
    line = next((item.strip() for item in reversed(stdout.splitlines()) if item.strip()), "")
    match = _SUMMARY.match(line)
    if not match:
        return None
    body = match.group("body")
    if body == "no tests ran":
        return {}
    counts: dict[str, int] = {}
    for value, name in _COUNTERS.findall(body):
        key = {"error": "errors", "warning": "warnings"}.get(name, name)
        counts[key] = counts.get(key, 0) + int(value)
    return counts or None


def parse_junit(path: Path) -> dict[str, str] | None:
    """junit XML 里的用例与结果；文件缺失、损坏或没有用例时返回 None。"""
    try:
        root = ElementTree.fromstring(path.read_bytes())
    except (OSError, ElementTree.ParseError):
        return None
    cases = {}
    for case in root.iter("testcase"):
        status = next((child.tag for child in case
                       if child.tag in {"failure", "error", "skipped"}), "passed")
        cases[f"{case.get('classname', '')}::{case.get('name', '')}"] = status
    return cases or None


def _inject_private_tests(task: LoadedTask, root: Path) -> None:
    target = root / PRIVATE
    target.mkdir(exist_ok=True)
    for source in sorted(task.private_tests_dir.rglob("test_*.py")):
        (target / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _git_apply(diff: Path, ws: Workspace, home: Path) -> None:
    """在全新工作区应用补丁。不读取使用者的 git 配置，也不受外层仓库影响。"""
    env = {"PATH": "/usr/bin:/bin", "HOME": str(home),
           "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
           "GIT_CEILING_DIRECTORIES": str(ws.root.parent)}
    proc = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", str(diff.resolve())],
        cwd=ws.root, capture_output=True, text=True, timeout=30, env=env)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip()[:300] or f"git apply exited {proc.returncode}")


def _snapshot(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): _sha(path) for path in sorted(root.rglob("*"))
            if path.is_file() and ".pristine" not in path.relative_to(root).parts}


def _run_layer(policy: SandboxPolicy, root: Path, layer: str,
               expected: frozenset[str] | None = None) -> LayerRun:
    scratch = policy.scratch / layer
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "pytest.ini").touch()   # 空配置：工作区里的 pytest.ini / pyproject 不生效
    junit = scratch / "junit.xml"
    command = [sys.executable, "-B", "-m", "pytest", layer, "-q", "--tb=line",
               "-p", "no:cacheprovider", "--noconftest", "-o", "addopts=",
               "-c", str(scratch / "pytest.ini"), "--rootdir", str(root),
               "--basetemp", str(scratch / "pytest-tmp"), "--junitxml", str(junit)]
    try:
        outcome = run_sandboxed(policy, command, root, PYTEST_TIMEOUT)
    except SandboxError as error:
        raise GradeInfrastructureError(str(error)) from error
    if outcome.exit_status == "timeout":
        raise subprocess.TimeoutExpired(command, PYTEST_TIMEOUT,
                                        output=outcome.stdout, stderr=outcome.stderr)
    if outcome.exit_status == "spawn_error":
        raise GradeInfrastructureError(f"pytest could not start: {outcome.stderr[-300:]}")
    return LayerRun(
        expected=expected or frozenset(), exit_status=outcome.exit_status,
        returncode=outcome.returncode, duration_ms=outcome.duration_ms,
        summary=parse_pytest_summary(outcome.stdout), cases=parse_junit(junit),
        tail=outcome.stdout[-500:], stderr=outcome.stderr[-500:])


_REFERENCE_CACHE: dict[tuple, dict[str, LayerRun]] = {}


def reference_runs(task: LoadedTask, work_base: Path, mode: str) -> dict[str, LayerRun]:
    """参考解在隔离环境里的执行记录。它收集到的用例就是每层“必须跑完的测试”。

    按任务内容缓存：同一进程内多次判分同一道题只跑一遍参考解。
    """
    try:
        pytest_version = importlib.metadata.version("pytest")
    except importlib.metadata.PackageNotFoundError:
        pytest_version = "unknown"
    sources = [task.fixture_tar, task.reference_patch,
               *sorted(task.private_tests_dir.glob("test_*.py"))]
    key = (tuple(_sha(path) for path in sources), pytest_version, sys.version,
           sys.executable, mode)
    if key not in _REFERENCE_CACHE:
        env_root = work_base / f"reference-{task.public.task_id}-{time.time_ns()}"
        env_root.mkdir(parents=True)
        ws = Workspace.create_from_fixture(f"reference-{task.public.task_id}",
                                           task.fixture_tar, env_root)
        policy = build_policy(ws.root, env_root / "scratch", mode=mode)
        try:
            _git_apply(task.reference_patch, ws, policy.scratch / "home")
        except RuntimeError as error:
            raise GradeInfrastructureError(
                f"reference patch does not apply: {error}") from error
        _inject_private_tests(task, ws.root)
        runs = {}
        for layer in (PUBLIC, PRIVATE):
            run = _run_layer(policy, ws.root, layer)
            if run.cases is None:
                raise GradeInfrastructureError(
                    f"reference solution produced no test results in {layer}; if pytest "
                    f"could not start, the sandbox does not fit this interpreter "
                    f"(see `agentlab doctor`): {run.stderr.strip()[-300:]} "
                    f"{run.tail.strip()[-200:]}")
            # 参考解自己收集到的用例就是它的预期集合，passed 才有意义
            runs[layer] = replace(run, expected=frozenset(run.cases))
        _REFERENCE_CACHE[key] = runs
    return _REFERENCE_CACHE[key]


def grade(task: LoadedTask, candidate_diff: Path,
          work_base: Path) -> tuple[GradeResult, dict]:
    """判分主入口。返回 (GradeResult, details)。"""
    graded_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    details: dict = {"layers": {}}
    diff_text = candidate_diff.read_text(encoding="utf-8")

    def invalid(layer: str, evidence: dict) -> tuple[GradeResult, dict]:
        details["layers"][layer] = dict(valid=False, **evidence)
        return GradeResult(
            episode_id="graded-offline", grader_id="three_layer_v1",
            graded_at=graded_at, verdict="invalid", strict_pass=False,
            details=details), details

    # ── 第一层：合法性 ──
    touched = parse_diff_paths(diff_text)
    illegal = [path for path in touched
               if not any(fnmatch.fnmatch(path, pattern) for pattern in task.public.allowed_paths)]
    # 受保护文件：候选不得触达（diff 路径级判断即可覆盖篡改测试的场景）
    protected_hit = [path for path in touched
                     if any(fnmatch.fnmatch(path, pattern)
                            for pattern in task.public.protected_paths)]
    git_only = [line for line in diff_text.splitlines() if _GIT_ONLY.match(line)]
    if illegal or protected_hit or git_only:
        return invalid("legality", dict(
            touched=touched, illegal=illegal, protected_hit=protected_hit,
            unsupported_patch_lines=git_only[:5]))

    # ── 全新评分环境：fixture + 候选补丁 ──
    env_root = work_base / f"grade-{task.public.task_id}-{time.time_ns()}"
    env_root.mkdir(parents=True)
    ws = Workspace.create_from_fixture(f"grade-{task.public.task_id}",
                                       task.fixture_tar, env_root)
    mode = resolve_mode()
    policy = build_policy(ws.root, env_root / "scratch", mode=mode)
    if diff_text.strip():
        try:
            _git_apply(candidate_diff, ws, policy.scratch / "home")
        except RuntimeError as error:
            return invalid("patch", dict(apply_failed=True, detail=str(error)))
    # 受保护文件哈希复核：缺失与改动都算篡改
    protected_changed = [
        rel for rel in PROTECTED
        if not (ws.root / rel).is_file() or _sha(ws.root / rel) != _sha(ws.pristine / rel)]
    if protected_changed:
        return invalid("legality", dict(
            valid=False, touched=touched, illegal=illegal,
            protected_hit=protected_hit, protected_hash_changed=protected_changed))
    details["layers"]["legality"] = dict(valid=True, touched=touched)

    # ── 隔离环境与预期测试集合 ──
    reference = reference_runs(task, work_base, mode)
    expected = {layer: run.expected for layer, run in reference.items()}
    details["layers"]["execution"] = {
        **policy.document(),
        "expected_tests": {layer: len(expected[layer]) for layer in (PUBLIC, PRIVATE)},
        # 参考解在同一环境下的结果：它不通过说明隔离破坏了正常测试，而不是候选的问题
        "reference_passed": {layer: run.passed for layer, run in reference.items()}}

    # ── 功能层：先公开，后私有 ──
    before = _snapshot(ws.root)
    public = _run_layer(policy, ws.root, PUBLIC, expected[PUBLIC])
    mutated = sorted(rel for rel, digest in before.items()
                     if not (ws.root / rel).is_file() or _sha(ws.root / rel) != digest)
    if mutated:
        return invalid("security", dict(workspace_mutated=mutated))

    # 私有测试此刻才落盘：公开层执行期间磁盘上没有它们
    _inject_private_tests(task, ws.root)
    private_sources = sorted(task.private_tests_dir.glob("test_*.py"))
    if any(_sha(ws.root / PRIVATE / source.name) != _sha(source) for source in private_sources):
        return invalid("security", dict(private_test_tampered=[s.name for s in private_sources]))
    private = _run_layer(policy, ws.root, PRIVATE, expected[PRIVATE])
    tampered = [source.name for source in private_sources
                if _sha(ws.root / PRIVATE / source.name) != _sha(source)]
    if tampered:
        return invalid("security", dict(private_test_tampered=tampered))

    details["layers"]["functional"] = dict(
        public_pass=public.passed, private_pass=private.passed,
        public=public.as_dict(), private=private.as_dict(),
        public_tail=public.tail[-500:], private_tail=private.tail[-500:])

    strict = public.passed and private.passed
    verdict = "strict_pass" if strict else (
        "partial" if public.passed else "strict_fail")
    return GradeResult(
        episode_id="graded-offline", grader_id="three_layer_v1",
        graded_at=graded_at, verdict=verdict, strict_pass=strict,
        details=details), details
