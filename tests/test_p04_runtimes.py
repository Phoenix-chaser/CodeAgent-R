# -*- coding: utf-8 -*-
"""P04 验收：fake executable 契约全套 + 受控运行器分支一致性 + pi 拒绝未核实能力。"""
import subprocess
import sys
import time
from pathlib import Path

import pytest

from agentlab.contracts import Budgets, EpisodeConfig, Usage
from agentlab.runtime import Workspace
from agentlab.runtime.budget_guard import BudgetExceeded, BudgetGuard
from agentlab.runtime.controlled import ControlledModelRuntime
from agentlab.runtime.fake_exec import write_fake_exec
from agentlab.runtime.pi_runtime import PiRuntime, PiUnsupported
from agentlab.runtime.protocol import StreamAssembler


def run_fake(fake: Path, timeout: float = 10.0):
    proc = subprocess.run([sys.executable, str(fake)], capture_output=True,
                          text=True, timeout=timeout)
    asm = StreamAssembler("ep-fake")
    for line in (proc.stdout or "").splitlines(keepends=True):
        asm.feed(line)
    exit_status = "ok" if proc.returncode == 0 else "nonzero_exit"
    return asm.finish(exit_status, proc.stderr or "")


def test_legal_stream_complete(tmp_path):
    fake = write_fake_exec(tmp_path / "ok.py", {
        "model_events": True, "usage": {"input_tokens": 100, "output_tokens": 20}})
    r = run_fake(fake)
    assert r.stream_status == "complete"
    assert r.usage_total.input_tokens == 100 and r.usage_total.output_tokens == 20
    assert r.events[-1].event_type == "runtime_termination"


def test_missing_usage_is_unknown(tmp_path):
    fake = write_fake_exec(tmp_path / "nouse.py", {"model_events": True})
    r = run_fake(fake)
    # 有 usage 键但字段缺 → unknown=None，绝不落 0
    assert r.usage_total is not None
    assert r.usage_total.input_tokens is None


def test_nonzero_exit_kept(tmp_path):
    fake = write_fake_exec(tmp_path / "err.py", {"exit_code": 3})
    r = run_fake(fake)
    assert r.exit_status == "nonzero_exit"
    assert len(r.events) >= 2          # 终止前事件保留


def test_stderr_not_in_events(tmp_path):
    fake = write_fake_exec(tmp_path / "noise.py", {"stderr_noise": True})
    r = run_fake(fake)
    assert "noise" in r.stderr_tail
    assert not any("noise" in str(e.payload) for e in r.events)


def test_duplicate_event_dropped(tmp_path):
    fake = write_fake_exec(tmp_path / "dup.py", {"duplicate_times": 3})
    r = run_fake(fake)
    tool_results = [e for e in r.events if e.event_type == "tool_result"]
    assert len(tool_results) == 1      # 3 次发送 → 1 次入账
    assert r.dropped_duplicates == 2


def test_truncated_last_line(tmp_path):
    fake = write_fake_exec(tmp_path / "trunc.py", {"truncate_last": True})
    r = run_fake(fake)
    assert r.stream_status == "truncated"
    assert not any(e.event_type == "tool_resu" for e in r.events)  # 半行不成事件


def test_missing_termination_not_success(tmp_path):
    """exit 0 但缺终止事件 → 不声明完整成功。"""
    fake = write_fake_exec(tmp_path / "noterm.py", {"termination": False})
    r = run_fake(fake)
    assert r.exit_status == "ok"
    assert r.stream_status == "missing_termination"


def test_timeout_cancel(tmp_path):
    fake = write_fake_exec(tmp_path / "slow.py", {"sleep": 30})
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run([sys.executable, str(fake)], capture_output=True,
                       timeout=1.0)     # 外层超时取消语义（kill 由调用方进程管理）


# ── 统一工具语义 ──────────────────────────────────────────
def make_ws(tmp_path):
    import tarfile, io
    src = tmp_path / "fsrc"; (src / "pkg").mkdir(parents=True)
    (src / "pkg" / "m.py").write_text("def f(n):\n    return n + 1\n", encoding="utf-8")
    tar = tmp_path / "f.tar"
    with tarfile.open(tar, "w") as t:
        for p in sorted(src.rglob("*")):
            t.add(p, arcname=str(p.relative_to(src)))
    return Workspace.create_from_fixture("epT", tar, tmp_path / "ws")


def test_tool_path_escape_rejected(tmp_path):
    from agentlab.runtime.tools import tool_read
    ws = make_ws(tmp_path)
    r = tool_read(ws, "../../etc/passwd")
    assert not r.ok and r.error_type == "path_escape"


def test_tool_read_and_diagnostic_nonzero(tmp_path):
    from agentlab.runtime.tools import tool_read, tool_command
    ws = make_ws(tmp_path)
    r = tool_read(ws, "pkg/m.py")
    assert r.ok and "def f" in r.output
    # 诊断命令非零退出：工具如实报告 error_type，但不冒充"模型动作错误"
    c = tool_command(ws, ["ls", "/nonexistent-xyz"])
    assert not c.ok and c.error_type == "nonzero_exit"


# ── ControlledModelRuntime：同状态双候选一致性 ───────────
def test_controlled_two_candidates_same_state(tmp_path):
    """核心语义：同一失败状态 → 两个候选分支互不污染、同剩余预算起点。"""
    import tarfile
    from agentlab.runtime.snapshot import Snapshot
    src = tmp_path / "bsrc"
    (src / "pkg").mkdir(parents=True)
    (src / "pkg" / "m.py").write_text("x = 1\n", encoding="utf-8")
    tar = tmp_path / "b.tar"
    with tarfile.open(tar, "w") as t:
        for p in sorted(src.rglob("*")):
            t.add(p, arcname=str(p.relative_to(src)))
    ws = Workspace.create_from_fixture("epC", tar, tmp_path / "ws")
    # 模拟失败状态并快照（分支共同起点）
    ws.write("pkg/state.txt", "failed-state")
    snap = Snapshot.create(ws, task_id="t", public_prefix=[{"h": 1}],
                           consumed_budget={"model_requests": 2},
                           store_dir=tmp_path / "snaps")

    def candidate_a(ctx):
        return {"text": "try read", "tool_calls": [
            {"id": "a1", "name": "read", "args": {"rel_path": "pkg/m.py"}},
            {"id": "a2", "name": "patch", "args": {"diff":
             "--- a/pkg/m.py\n+++ b/pkg/m.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"}}]}

    def candidate_b(ctx):
        return {"text": "just write", "finish": True}

    outcomes = {}
    for name, fn in (("A", candidate_a), ("B", candidate_b)):
        branch_ws = snap.restore(tmp_path / f"ws" / f"branch{name}")
        ep = EpisodeConfig(episode_id="epC", task_id="t",
                           budgets=Budgets(max_model_requests=10))
        guard = BudgetGuard("epC", ep.budgets)
        guard.settle("seed", "model_requests", 2)   # 与快照 consumed 一致
        rt = ControlledModelRuntime(fn, max_steps=4)
        outcomes[name] = rt.run(ep, branch_ws, guard, [{"h": 1}])

    assert outcomes["A"]["exit_status"] == "max_steps" or outcomes["A"]["exit_status"] == "ok"
    assert outcomes["B"]["exit_status"] == "ok"
    # A 读了文件、打了补丁；B 什么都没动 → 分支互不污染
    b_ws = tmp_path / "ws" / "branchB"
    assert (b_ws / "pkg" / "m.py").read_text(encoding="utf-8") == "x = 1\n"
    a_ws = tmp_path / "ws" / "branchA"
    assert "x = 2" in (a_ws / "pkg" / "m.py").read_text(encoding="utf-8")


# ── pi：未核实能力拒绝执行 ───────────────────────────────
def test_pi_refuses_unverified(tmp_path):
    rt = PiRuntime()
    with pytest.raises(PiUnsupported, match="subprocess_json"):
        rt.run("ep-pi", ["pi", "-p", "hi"], cwd=tmp_path, timeout=5)


def test_controlled_finish_is_counted_and_next_run_cannot_reuse_admission(tmp_path):
    workspace = make_ws(tmp_path)
    episode = EpisodeConfig(episode_id="finish", task_id="task",
                            budgets=Budgets(max_model_requests=1))
    guard = BudgetGuard(episode.episode_id, episode.budgets)
    calls = []

    def finish(context):
        calls.append(context)
        return {"text": "done", "finish": True}

    runtime = ControlledModelRuntime(finish)
    first = runtime.run(episode, workspace, guard, [])
    second = runtime.run(episode, workspace, guard, [])
    assert first["steps"] == 1
    assert first["public_history"][-1]["content"] == "done"
    assert second["exit_status"] == "budget_exceeded"
    assert second["steps"] == 0
    assert len(calls) == 1
    assert guard.ledger.spent["model_requests"] == 1
    assert guard.ledger.reserved["model_requests"] == 0


def test_controlled_model_exception_still_settles_request():
    episode = EpisodeConfig(episode_id="failed-request", task_id="task",
                            budgets=Budgets(max_model_requests=1))
    guard = BudgetGuard(episode.episode_id, episode.budgets)

    def fail(context):
        raise RuntimeError("model unavailable")

    runtime = ControlledModelRuntime(fail)
    with pytest.raises(RuntimeError, match="unavailable"):
        runtime.generate(episode, guard, [])
    assert guard.ledger.spent["model_requests"] == 1
    assert guard.ledger.reserved["model_requests"] == 0
    with pytest.raises(BudgetExceeded):
        runtime.generate(episode, guard, [])


def test_duplicate_tool_ids_do_not_bypass_tool_budget(tmp_path):
    workspace = make_ws(tmp_path)
    episode = EpisodeConfig(episode_id="duplicate-tool", task_id="task",
                            budgets=Budgets(max_tool_calls=1))
    guard = BudgetGuard(episode.episode_id, episode.budgets)
    runtime = ControlledModelRuntime(lambda context: {})
    call = {"id": "same-id", "name": "read", "args": {"rel_path": "pkg/m.py"}}
    outcome = runtime.step(episode, workspace, guard, 0, {"tool_calls": [call, call]})
    assert outcome.exit_status == "budget_exceeded"
    assert len(outcome.observations) == 1
    assert "def f" in outcome.observations[0]["output"]
    assert guard.ledger.spent["tool_calls"] == 1
    assert guard.ledger.reserved["tool_calls"] == 0


def test_controlled_model_cannot_mutate_caller_history():
    episode = EpisodeConfig(episode_id="history", task_id="task", budgets=Budgets())
    guard = BudgetGuard(episode.episode_id, episode.budgets)
    history = [{"role": "user", "content": "original"}]

    def mutate(context):
        context["history"][0]["content"] = "changed"
        return {"finish": True}

    ControlledModelRuntime(mutate).generate(episode, guard, history)
    assert history[0]["content"] == "original"


def test_controlled_keeps_executed_observations_when_later_tool_is_rejected(tmp_path):
    workspace = make_ws(tmp_path)
    episode = EpisodeConfig(episode_id="partial-tools", task_id="task",
                            budgets=Budgets(max_tool_calls=1, max_model_requests=1))
    guard = BudgetGuard(episode.episode_id, episode.budgets)
    calls = [
        {"id": call_id, "name": "read", "args": {"rel_path": "pkg/m.py"}}
        for call_id in ("first", "second")]
    runtime = ControlledModelRuntime(lambda context: {"text": "inspect", "tool_calls": calls})
    result = runtime.run(episode, workspace, guard, [])
    assert result["exit_status"] == "budget_exceeded"
    assert [message["role"] for message in result["public_history"]] == ["assistant", "tool"]
    assert result["public_history"][1]["call_id"] == "first"
    assert "def f" in result["public_history"][1]["observation"]
