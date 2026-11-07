# -*- coding: utf-8 -*-
"""MockRuntime：确定脚本的垂直链路（验证平台，不产生模型能力结论）。

公开任务 → 预算准入 → 工具执行/模型步骤 → 事件流 → 候选冻结 → RunResult。
费用恒为 unknown（mock 无真实计费，unknown ≠ 0）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from agentlab.contracts import (
    ArtifactRef, CostAmount, EpisodeConfig, PublicTask, RunResult,
    RuntimeEvent, Usage,
)
from agentlab.runtime.budget_guard import BudgetExceeded, BudgetGuard
from agentlab.runtime.exec import run_command
from agentlab.runtime.workspace import Workspace


@dataclass
class MockStep:
    kind: str                                  # model_request | tool_exec | write_file
    argv: list[str] | None = None
    rel_path: str | None = None
    content: str | None = None
    usage: Usage | None = None
    timeout: float = 30.0


@dataclass
class MockRunOutput:
    result: RunResult
    events: list[RuntimeEvent]


class MockRuntime:
    name = "mock"

    def run(self, task: PublicTask, episode: EpisodeConfig,
            workspace: Workspace, guard: BudgetGuard,
            steps: list[MockStep]) -> MockRunOutput:
        events: list[RuntimeEvent] = []
        seq = 0
        usage_total = Usage(input_tokens=0, output_tokens=0)

        def emit(source, event_type, step_idx, payload, usage=None):
            nonlocal seq
            seq += 1
            events.append(RuntimeEvent(
                episode_id=episode.episode_id, attempt=episode.attempt,
                step=step_idx, source=source, event_type=event_type,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                model_revision="mock@0.1.0", payload=payload, usage=usage,
                seq=seq))

        emit("runtime", "episode_start", None,
             {"task_id": task.task_id, "runtime": self.name})
        exit_status: str = "ok"
        error_detail: dict = {}
        op = lambda i: f"{episode.episode_id}:{episode.attempt}:{i}"

        for i, st in enumerate(steps):
            try:
                if st.kind == "model_request":
                    guard.admit_or_raise(op(f"req{i}"), "model_requests", 1)
                    emit("model", "model_request", i, {"index": i})
                    emit("model", "model_response", i, {"index": i},
                         usage=st.usage)
                    guard.settle(op(f"req{i}"), "model_requests", 1)
                    if st.usage is not None:
                        for dim, val in (("input_tokens", st.usage.input_tokens),
                                         ("output_tokens", st.usage.output_tokens)):
                            if val is not None:
                                guard.settle(op(f"req{i}:{dim}"), dim, val)
                                setattr(usage_total, dim,
                                        (getattr(usage_total, dim) or 0) + val)
                elif st.kind == "tool_exec":
                    guard.admit_or_raise(op(f"tool{i}"), "tool_calls", 1)
                    emit("tool", "tool_call", i, {"argv": st.argv})
                    out = run_command(st.argv or [], workspace.root, st.timeout)
                    emit("tool", "tool_result", i, {
                        "exit_status": out.exit_status,
                        "returncode": out.returncode,
                        "duration_ms": out.duration_ms,
                        "truncated": out.stdout_truncated})
                    guard.settle(op(f"tool{i}"), "tool_calls", 1)
                    if out.exit_status == "timeout":
                        exit_status, error_detail = "timeout", {"step": i}
                        break
                    if out.exit_status != "ok":
                        exit_status, error_detail = "error", {
                            "step": i, "stderr": out.stderr[:200]}
                        break
                elif st.kind == "write_file":
                    workspace.write(st.rel_path or "out.txt",
                                    st.content or "")
            except BudgetExceeded as e:
                exit_status, error_detail = "budget_exceeded", {"detail": str(e)}
                break

        # 候选冻结：diff vs pristine（无论成败，已发生的修改如实冻结）
        diff_path = workspace.root.parent / f"candidate-{episode.episode_id}.diff"
        workspace.diff_against_pristine(diff_path)
        import hashlib
        artifacts = [ArtifactRef(
            path=str(diff_path), kind="workspace_diff",
            sha256=hashlib.sha256(diff_path.read_bytes()).hexdigest())]
        emit("runtime", "candidate_frozen", None,
             {"diff_sha256": artifacts[0].sha256})
        emit("runtime", "episode_end", None, {"exit_status": exit_status})

        result = RunResult(
            episode_id=episode.episode_id, task_id=task.task_id,
            exit_status=exit_status, usage=usage_total,
            cost=CostAmount.unknown(),           # mock 无计费 → unknown ≠ 0
            artifacts=artifacts,
            events_summary={"n_events": len(events), "error": error_detail},
            started_at=events[0].timestamp, finished_at=events[-1].timestamp)
        return MockRunOutput(result=result, events=events)
