# -*- coding: utf-8 -*-
"""ControlledModelRuntime：决策边界受控的模型运行器。

这是主模型比较与分支采样的执行核心：
- 「模型」以 candidate_fn 注入（离线=脚本函数；真实=模型服务适配器），
  运行器控制 assistant 决策边界、工具 schema 与历史构造。
- step()：在决策边界接收一条 assistant 消息 → 执行其工具调用（统一工具层）
  → 返回观测与下一边界（快照就绪点）。
- 候选 y = 下一条完整 assistant 消息；工具返回由环境生成，绝不拼入候选。
"""
from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable
from uuid import uuid4

from agentlab.contracts import EpisodeConfig, RuntimeEvent, Usage
from agentlab.runtime.budget_guard import BudgetExceeded, BudgetGuard
from agentlab.runtime.tools import ToolErrorType, tool_command, tool_patch, tool_read, tool_search
from agentlab.runtime.workspace import Workspace

TOOL_IMPLS = {"read": tool_read, "search": tool_search,
              "patch": tool_patch, "command": tool_command}

AssistantMessage = dict          # {text, tool_calls: [{id, name, args}]}
CandidateFn = Callable[[dict], AssistantMessage]   # 输入公开历史 → 候选消息


@dataclass
class StepOutcome:
    observations: list[dict] = field(default_factory=list)   # 环境生成的工具返回
    events: list[RuntimeEvent] = field(default_factory=list)
    usage: Usage | None = None
    exit_status: str = "ok"


class ControlledModelRuntime:
    name = "controlled_model"

    def __init__(self, candidate_fn: CandidateFn, max_steps: int = 20):
        if max_steps < 0:
            raise ValueError("max_steps must be nonnegative")
        self.candidate_fn = candidate_fn
        self.max_steps = max_steps

    def generate(self, episode: EpisodeConfig, guard: BudgetGuard,
                 public_history: list[dict]) -> AssistantMessage:
        """每次真实调用单独准入、结算；异常也保留已发生的请求次数。"""
        op_id = f"{episode.episode_id}:{episode.attempt}:req:{uuid4().hex}"
        guard.admit_or_raise(op_id, "model_requests", 1)
        try:
            return self.candidate_fn({"history": deepcopy(public_history)})
        finally:
            guard.settle(op_id, "model_requests", 1)

    def run(self, episode: EpisodeConfig, ws: Workspace, guard: BudgetGuard,
            public_history: list[dict],
            emit: Callable[[RuntimeEvent], None] | None = None,
            start_step: int = 0) -> dict:
        """受控循环：candidate_fn(公开历史) → 执行工具 → 观测入历史 → 下一边界。

        返回 {exit_status, steps, public_history}。公开历史只含公开观测
        （工具返回由环境生成），不混入任何私有内容。
        """
        history = deepcopy(public_history)
        requests = 0
        for step_idx in range(self.max_steps):
            try:
                msg = self.generate(episode, guard, history)
                requests += 1
                if msg.get("finish"):
                    history.append({"role": "assistant", "content": msg.get("text", "")})
                    return dict(exit_status="ok", steps=requests,
                                public_history=history)
                outcome = self.step(episode, ws, guard, start_step + step_idx, msg,
                                    emit=emit)
            except BudgetExceeded:
                return dict(exit_status="budget_exceeded", steps=requests,
                            public_history=history)
            history.append({"role": "assistant", "content": msg.get("text", ""),
                            "tool_calls": msg.get("tool_calls", [])})
            for obs in outcome.observations:
                history.append({"role": "tool", "observation": obs["output"],
                                "call_id": obs["call_id"]})
            if outcome.exit_status != "ok":
                return dict(exit_status=outcome.exit_status, steps=requests,
                            public_history=history)
        return dict(exit_status="max_steps", steps=requests,
                    public_history=history)

    def step(self, episode: EpisodeConfig, ws: Workspace, guard: BudgetGuard,
             step_idx: int, msg: AssistantMessage,
             emit=None) -> StepOutcome:
        outcome = StepOutcome()
        for call in msg.get("tool_calls", []):
            name, args, cid = call.get("name"), call.get("args", {}), call.get("id")
            op_id = f"{episode.episode_id}:{episode.attempt}:tool:{uuid4().hex}"
            try:
                guard.admit_or_raise(op_id, "tool_calls", 1)
            except BudgetExceeded:
                outcome.exit_status = "budget_exceeded"
                return outcome
            try:
                if name not in TOOL_IMPLS:
                    result = type("R", (), {"ok": False, "output": f"unknown tool {name}",
                                            "error_type": ToolErrorType.INVALID_ARGUMENTS,
                                            "truncated": False})()
                else:
                    impl = TOOL_IMPLS[name]
                    if name in ("read", "search"):
                        key = "rel_path" if name == "read" else "pattern"
                        result = impl(ws, args.get(key, ""))
                    elif name == "patch":
                        result = impl(ws, args.get("diff", ""))
                    else:
                        result = impl(ws, args.get("argv", []),
                                      timeout=float(args.get("timeout", 60)))
            finally:
                guard.settle(op_id, "tool_calls", 1)
            outcome.observations.append({
                "call_id": cid, "tool": name, "ok": result.ok,
                "output": result.output, "error_type": result.error_type,
                "truncated": result.truncated})
            if emit:
                emit(RuntimeEvent(
                    episode_id=episode.episode_id, attempt=episode.attempt,
                    step=step_idx, source="tool", event_type="tool_result",
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    payload={"call_id": cid, "tool": name, "ok": result.ok,
                             "error_type": result.error_type}, seq=0))
        return outcome
