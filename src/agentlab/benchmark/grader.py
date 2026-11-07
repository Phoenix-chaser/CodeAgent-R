# -*- coding: utf-8 -*-
"""三层判分 grader：冻结候选 → 全新环境 → 合法性/功能/鲁棒性。

可信路径（private-grading.md §1）：
    冻结候选 diff（P02 产物，a/ b/ 可移植补丁）
    → 合法性：解析 diff 触达路径 ⊆ allowed_paths；受保护文件哈希不变
    → 全新工作区 = fixture + git apply 候选 + 注入私有测试
    → 功能：pytest(公开) / pytest(私有) 子进程黑盒执行
    → 鲁棒性：私有测试中的边界/回归子集
Agent 永远不接触 private/；评分不读取 Agent 活动工作区。
"""
from __future__ import annotations

import hashlib
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from agentlab.benchmark.task_store import LoadedTask
from agentlab.contracts import GradeResult
from agentlab.runtime.workspace import Workspace

PYTEST_TIMEOUT = 60


@dataclass
class GraderError:
    kind: str            # patch_apply_failed | pytest_spawn_error | ...
    detail: str


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


def grade(task: LoadedTask, candidate_diff: Path,
          work_base: Path) -> tuple[GradeResult, dict]:
    """判分主入口。返回 (GradeResult, details)。"""
    graded_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    details: dict = {"layers": {}}
    diff_text = candidate_diff.read_text(encoding="utf-8")

    # ── 第一层：合法性 ──
    touched = parse_diff_paths(diff_text)
    allowed = set(task.public.allowed_paths)
    import fnmatch
    illegal = [p for p in touched
               if not any(fnmatch.fnmatch(p, pat) for pat in allowed)]
    # 受保护文件：候选不得触达（diff 路径级判断即可覆盖篡改测试的场景）
    protected_hit = [p for p in touched
                     if any(fnmatch.fnmatch(p, pat)
                            for pat in task.public.protected_paths)]
    if illegal or protected_hit:
        details["layers"]["legality"] = dict(
            valid=False, touched=touched, illegal=illegal,
            protected_hit=protected_hit)
        return GradeResult(
            episode_id="graded-offline", grader_id="three_layer_v1",
            graded_at=graded_at, verdict="invalid", strict_pass=False,
            details=details), details
    details["layers"]["legality"] = dict(valid=True, touched=touched)

    # ── 全新评分环境：fixture + 候选补丁 + 私有测试注入 ──
    import tempfile
    env_root = work_base / f"grade-{task.public.task_id}-{time.time_ns()}"
    env_root.mkdir(parents=True)
    ws = Workspace.create_from_fixture(f"grade-{task.public.task_id}",
                                       task.fixture_tar, env_root)
    if diff_text.strip():
        patch_proc = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", str(candidate_diff.resolve())],
            cwd=ws.root, capture_output=True, text=True, timeout=30)
    else:
        patch_proc = None   # 空候选 = 零改动，属合法候选，进入功能层
    if patch_proc is not None and patch_proc.returncode != 0:
        result = GradeResult(
            episode_id="graded-offline", grader_id="three_layer_v1",
            graded_at=graded_at, verdict="invalid", strict_pass=False,
            details={"layers": {"patch": dict(
                apply_failed=True, stderr=patch_proc.stderr[:300])}})
        return result, result.details

    # 受保护文件哈希复核（防 diff 解析遗漏的间接篡改）
    for rel in ("tests_public/test_public.py", "run_public_tests.sh"):
        f_ws, f_fix = ws.root / rel, ws.pristine / rel
        if f_ws.exists() and _sha(f_ws) != _sha(f_fix):
            result = GradeResult(
                episode_id="graded-offline", grader_id="three_layer_v1",
                graded_at=graded_at, verdict="invalid", strict_pass=False,
                details={"layers": {"legality": dict(
                    valid=False, protected_hash_changed=[rel])}})
            return result, result.details

    # 私有测试注入评分环境（不在 Agent 环境中存在）
    priv_dir = ws.root / "tests_private"
    priv_dir.mkdir()
    for pf in sorted(task.private_tests_dir.rglob("test_*.py")):
        (priv_dir / pf.name).write_text(pf.read_text(encoding="utf-8"),
                                        encoding="utf-8")

    def run_pytest(subdir: str) -> tuple[bool, str]:
        # 安全：清洗环境变量，防止模型代码读取 API key
        import os as _os
        clean_env = {k: v for k, v in _os.environ.items()
                     if not any(x in k.upper() for x in
                                ("API_KEY", "SECRET", "TOKEN", "PASSWORD"))}
        proc = subprocess.run(
            ["python3", "-m", "pytest", subdir, "-q", "--tb=line", "-p", "no:cacheprovider"],
            cwd=ws.root, capture_output=True, text=True,
            timeout=PYTEST_TIMEOUT, env=clean_env)
        return proc.returncode == 0, (proc.stdout or "")[-500:]

    public_ok, public_out = run_pytest("tests_public")
    # 安全：公开测试跑完后，校验私有测试是否被篡改
    for pf in sorted(task.private_tests_dir.rglob("test_*.py")):
        injected = ws.root / "tests_private" / pf.name
        if injected.exists() and _sha(injected) != _sha(pf):
            result = GradeResult(
                episode_id="graded-offline", grader_id="three_layer_v1",
                graded_at=graded_at, verdict="invalid", strict_pass=False,
                details={"layers": {"security": dict(
                    private_test_tampered=[pf.name])}})
            return result, result.details
    private_ok, private_out = run_pytest("tests_private")
    details["layers"]["functional"] = dict(
        public_pass=public_ok, private_pass=private_ok,
        public_tail=public_out[-200:], private_tail=private_out[-200:])

    strict = public_ok and private_ok
    verdict = "strict_pass" if strict else (
        "partial" if public_ok and not private_ok else "strict_fail")
    return GradeResult(
        episode_id="graded-offline", grader_id="three_layer_v1",
        graded_at=graded_at, verdict=verdict, strict_pass=strict,
        details=details), details
