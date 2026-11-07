# -*- coding: utf-8 -*-
"""RuntimeEvent：执行事件契约。

去重与唯一性规则（P01 冻结，防臆造跨重试唯一性）：
- dedup_key = (episode_id, attempt, seq)。
- seq 由采集端按到达顺序在 (episode, attempt) 内自增分配；
  上游重放会产生重复，靠 dedup_key 去重。
- event_id 仅当来源原生提供时填充（透传，不合成）；
  未提供 event_id 的流不虚构全局唯一 ID。
- 重试是新的 attempt：跨 attempt 的事件永不等同。
"""
from __future__ import annotations

from typing import Any, Literal

from agentlab.contracts.versioning import VersionedModel

EventSource = Literal["runtime", "model", "tool", "grader"]


class Usage(VersionedModel):
    """用量；None = 未知（unknown ≠ 0）。"""
    input_tokens: int | None = None
    output_tokens: int | None = None


class RuntimeEvent(VersionedModel):
    episode_id: str
    attempt: int = 1
    step: int | None = None            # 所属 step；runtime 级事件可为 None
    source: EventSource
    event_type: str                    # 如 model_request / tool_call / tool_result / episode_end
    timestamp: str                     # ISO-8601；来源无时钟时由采集端打点并注明
    model_revision: str | None = None
    payload: dict[str, Any] = {}
    usage: Usage | None = None
    event_id: str | None = None  # 仅透传来源原生 ID，不合成
    seq: int = 0                  # 采集端在 (episode, attempt) 内按到达顺序分配
