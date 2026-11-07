# -*- coding: utf-8 -*-
"""P02 验收测试：六条硬标准 + 安全/幂等/恢复补充项。

1. 两个任务不能看到对方写入
2. 超时后子进程不继续修改文件
3. 分支恢复 hash 一致
4. 并发准入不重复占用同一剩余额度
5. 重复 usage 不重复记账
6. 恢复（新 attempt）不重复提交旧结果，且预算不清零
"""
import io
import tarfile
import threading
import time
from pathlib import Path

import pytest

from agentlab.contracts import (
    Budgets, ContractError, EpisodeConfig, PublicTask, Usage,
)
from agentlab.runtime import (
    BudgetExceeded, BudgetGuard, MockRuntime, MockStep, Snapshot,
    UnsupportedSandbox, Workspace, detect_backends, require_backend,
)


# ── 公共夹具 ───────────────────────────────────────────────
def make_fixture(tmp_path: Path) -> Path:
    src = tmp_path / "fixture-src"
    (src / "src").mkdir(parents=True)
    (src / "src" / "app.py").write_text(
        "def sum_to(n):\n    if n < 0:\n        raise ValueError\n    return n * (n + 1) // 2\n",
        encoding="utf-8")
    (src / "tests").mkdir()
    (src / "tests" / "test_app.py").write_text(
        "from src.app import sum_to\n\ndef test_zero():\n    assert sum_to(0) == 0\n",
        encoding="utf-8")
    tar_path = tmp_path / "fixture.tar"
    with tarfile.open(tar_path, "w") as tar:
        for p in sorted(src.rglob("*")):
            tar.add(p, arcname=str(p.relative_to(src)))
    return tar_path


def make_task() -> PublicTask:
    return PublicTask(
        task_id="mini_alpha_bug_001", source_group="sg_001", split="smoke",
        bug_family="off_by_one", difficulty="L1",
        instruction="修复 sum_to 的 n=0 行为。",
        allowed_paths=["src/**"], protected_paths=["tests/**"],
        budgets=Budgets(max_model_requests=30, max_tool_calls=60,
                        max_wall_seconds=600))


def make_episode(eid: str = "ep-1", attempt: int = 1) -> EpisodeConfig:
    return EpisodeConfig(episode_id=eid, task_id="mini_alpha_bug_001",
                         attempt=attempt, runtime="mock",
                         budgets=Budgets(max_model_requests=30,
                                         max_tool_calls=60, max_wall_seconds=600))


# ── 1. 工作区隔离 ─────────────────────────────────────────
def test_workspaces_isolated(tmp_path):
    fixture = make_fixture(tmp_path)
    ws_a = Workspace.create_from_fixture("epA", fixture, tmp_path / "ws")
    ws_b = Workspace.create_from_fixture("epB", fixture, tmp_path / "ws")
    ws_a.write("src/leak.txt", "A was here")
    assert not (ws_b.root / "src" / "leak.txt").exists()
    assert ws_a.read("src/app.py") == ws_b.read("src/app.py")

def test_fixture_escape_rejected(tmp_path):
    bad = tmp_path / "bad.tar"
    with tarfile.open(bad, "w") as tar:
        info = tarfile.TarInfo("../evil.txt")
        data = b"x"
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    with pytest.raises(ContractError, match="escapes workspace"):
        Workspace.create_from_fixture("epE", bad, tmp_path / "ws")


# ── 2. 超时后无残留写入者 ─────────────────────────────────
def test_timeout_kills_process_group(tmp_path):
    fixture = make_fixture(tmp_path)
    ws = Workspace.create_from_fixture("epT", fixture, tmp_path / "ws")
    marker = ws.root / "grow.log"
    from agentlab.runtime import run_command
    out = run_command(
        ["bash", "-c",
         f"while true; do echo x >> {marker}; sleep 0.05; done"],
        cwd=ws.root, timeout=0.6)
    assert out.exit_status == "timeout"
    size_after_kill = marker.stat().st_size
    time.sleep(0.5)                           # 若有残留写入者，文件会继续增长
    assert marker.stat().st_size == size_after_kill


# ── 3. 快照分支：恢复 hash 一致 + 互不传播 ───────────────
def test_snapshot_restore_hash_and_branch_isolation(tmp_path):
    fixture = make_fixture(tmp_path)
    base_dir = tmp_path / "ws"
    ws = Workspace.create_from_fixture("epS", fixture, base_dir)
    ws.write("src/mod.py", "branch-base")
    snap = Snapshot.create(ws, task_id="t1", public_prefix=[{"role": "user", "text": "fix it"}],
                           consumed_budget={"model_requests": 3},
                           store_dir=tmp_path / "snaps")
    # 分支1：恢复后修改
    b1 = snap.restore(base_dir / "branch1")
    b1.write("src/branch1.txt", "one")
    # 分支2：恢复 → hash 与记录一致，且看不到分支1 的写入
    b2 = snap.restore(base_dir / "branch2")
    assert not (b2.root / "src" / "branch1.txt").exists()
    assert b2.read("src/mod.py") == "branch-base"
    # 篡改快照存储 → 恢复被拒
    (snap.store_path / "src" / "mod.py").write_text("tampered", encoding="utf-8")
    with pytest.raises(ContractError, match="hash mismatch"):
        snap.restore(base_dir / "branch3")


# ── 4. 并发准入原子性 ─────────────────────────────────────
def test_concurrent_admission_no_double_spend(tmp_path):
    budgets = Budgets(max_model_requests=10)
    guard = BudgetGuard("epC", budgets)
    guard.settle("seed-op", "model_requests", 4)      # 已花 4，剩 6
    results = []
    def worker(n):
        results.append(guard.try_reserve(f"r{n}", "model_requests", 4))
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    [t.start() for t in threads]; [t.join() for t in threads]
    # 4(已结) + 4(预留) + 4(新) = 12 > 10 → 恰好一个成功
    assert sorted(results) == [False, True]
    assert guard.ledger.reserved["model_requests"] == 4
    assert guard.ledger.spent["model_requests"] == 4


# ── 5. 幂等结算：重复 usage 不重复入账 ────────────────────
def test_idempotent_settle_and_replay(tmp_path):
    guard = BudgetGuard("epI", Budgets(max_model_requests=30))
    guard.try_reserve("op1", "model_requests", 1)
    guard.settle("op1", "model_requests", 1)
    guard.settle("op1", "model_requests", 1)          # 重放
    guard.settle("op1", "model_requests", 1)          # 再重放
    assert guard.ledger.spent["model_requests"] == 1
    # unknown 结算不落 0
    guard.try_reserve("op2", "usd", 1)
    guard.settle("op2", "usd", None)
    assert guard.ledger.total_usd() is None
    assert guard.ledger.unknown_spent_dims == ["usd"]


# ── 6. Mock 全链路 + 恢复不清零 + 旧结果不重提 ───────────
def test_mock_vertical_chain(tmp_path):
    fixture = make_fixture(tmp_path)
    task = make_task()
    ws = Workspace.create_from_fixture("epM", fixture, tmp_path / "ws")
    guard = BudgetGuard("epM", task.budgets)
    rt = MockRuntime()
    out = rt.run(task, make_episode("epM"), ws, guard, steps=[
        MockStep(kind="model_request", usage=Usage(input_tokens=100, output_tokens=20)),
        MockStep(kind="tool_exec", argv=["python3", "-c", "print('ok')"]),
        MockStep(kind="write_file", rel_path="src/note.txt", content="done"),
    ])
    r = out.result
    assert r.exit_status == "ok"
    assert r.usage.input_tokens == 100 and r.usage.output_tokens == 20
    assert r.cost.usd is None                          # unknown ≠ 0
    assert r.artifacts[0].kind == "workspace_diff" and r.artifacts[0].sha256
    types = [e.event_type for e in out.events]
    assert types[0] == "episode_start" and types[-1] == "episode_end"
    assert "candidate_frozen" in types
    assert guard.ledger.spent["model_requests"] == 1
    assert guard.ledger.spent["tool_calls"] == 1

def test_budget_exceeded_and_timeout_paths(tmp_path):
    fixture = make_fixture(tmp_path)
    task = make_task()
    ws = Workspace.create_from_fixture("epB", fixture, tmp_path / "ws")
    guard = BudgetGuard("epB", Budgets(max_model_requests=1))
    rt = MockRuntime()
    out = rt.run(task, make_episode("epB"), ws, guard, steps=[
        MockStep(kind="model_request"),     # 用掉唯一额度
        MockStep(kind="model_request"),     # 超限
    ])
    assert out.result.exit_status == "budget_exceeded"

    ws2 = Workspace.create_from_fixture("epT2", fixture, tmp_path / "ws")
    out2 = rt.run(task, make_episode("epT2"), ws2,
                  BudgetGuard("epT2", task.budgets), steps=[
        MockStep(kind="tool_exec", argv=["bash", "-c", "sleep 5"],
                 timeout=0.5),
    ])
    assert out2.result.exit_status == "timeout"

def test_attempt_restart_keeps_budget_no_result_resubmit(tmp_path):
    """恢复语义：attempt+1 复用同一 guard（预算不清零）；新结果只含新 attempt 事件。"""
    fixture = make_fixture(tmp_path)
    task = make_task()
    ws = Workspace.create_from_fixture("epR", fixture, tmp_path / "ws")
    guard = BudgetGuard("epR", task.budgets)
    rt = MockRuntime()
    # attempt 1：花 1 次模型请求
    out1 = rt.run(task, make_episode("epR", attempt=1), ws, guard,
                  steps=[MockStep(kind="model_request")])
    assert guard.ledger.spent["model_requests"] == 1
    # attempt 2：同一 guard 继续（不清零）
    ws2 = Workspace.create_from_fixture("epR-a2", fixture, tmp_path / "ws")
    out2 = rt.run(task, make_episode("epR", attempt=2), ws2, guard,
                  steps=[MockStep(kind="model_request")])
    assert guard.ledger.spent["model_requests"] == 2      # 累计
    assert all(e.attempt == 2 for e in out2.events)       # 无 attempt-1 旧结果混入
    assert out2.result.usage.input_tokens == 0 or out2.result.usage.input_tokens is None


# ── 沙箱显式 unsupported ─────────────────────────────────
def test_container_unsupported_explicit():
    backends = detect_backends()
    if not backends["container"].available:
        with pytest.raises(UnsupportedSandbox):
            require_backend("container")               # 不静默回退
    assert backends["process"].available


def test_settlement_releases_original_reservation():
    guard = BudgetGuard("settlement", Budgets(max_model_requests=20))
    assert guard.try_reserve("first", "model_requests", 10)
    assert guard.try_reserve("second", "model_requests", 5)
    guard.settle("first", "model_requests", 3)
    assert guard.ledger.spent["model_requests"] == 3
    assert guard.ledger.reserved["model_requests"] == 5
    guard.settle("second", "model_requests", 8)
    assert guard.ledger.spent["model_requests"] == 11
    assert guard.ledger.reserved["model_requests"] == 0


def test_seed_and_unknown_settlement_replay_is_idempotent():
    guard = BudgetGuard("replay", Budgets(max_model_requests=10))
    guard.settle("seed", "model_requests", 3)
    guard.settle("seed", "model_requests", 3)
    assert guard.ledger.spent["model_requests"] == 3
    guard.try_reserve("unknown", "usd", 2)
    guard.settle("unknown", "usd", None)
    guard.settle("unknown", "usd", None)
    guard.settle("unknown", "usd", 1)
    assert guard.ledger.reserved["usd"] == 2
    assert guard.ledger.total_usd() is None
    assert guard.ledger.spent.get("usd", 0) == 0


@pytest.mark.parametrize("amount", [-1, float("inf"), float("nan")])
def test_invalid_budget_amount_rejected(amount):
    guard = BudgetGuard("invalid", Budgets())
    with pytest.raises(ValueError, match="finite and nonnegative"):
        guard.try_reserve("reserve", "model_requests", amount)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        guard.settle("settle", "model_requests", amount)
    assert guard.ledger.spent == {}
    assert guard.ledger.reserved == {}


def test_operation_identity_cannot_change_dimension_or_reservation():
    guard = BudgetGuard("identity", Budgets())
    assert guard.try_reserve("operation", "model_requests", 1)
    with pytest.raises(ValueError):
        guard.try_reserve("operation", "model_requests", 2)
    with pytest.raises(ValueError):
        guard.settle("operation", "tool_calls", 1)
    guard.settle("operation", "model_requests", 1)
    assert guard.try_reserve("operation", "model_requests", 1)
    assert guard.ledger.spent["model_requests"] == 1
    assert guard.ledger.reserved["model_requests"] == 0
