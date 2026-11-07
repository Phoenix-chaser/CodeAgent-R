# -*- coding: utf-8 -*-
"""运行时协议归一化：外部进程 stdout JSON 行 → RuntimeEvent 流。

规则（runtime-protocol.md §3/§5）：
- 事件封套字段：schema_version/event/call_id/payload/usage/occurred_at/event_id；
  event_id 仅透传，未提供时用 (episode, attempt, seq) 序号——不臆造唯一 ID。
- 重复 event_id → 去重（dropped 计数），不重复入账。
- 最后一行截断（无换行不完整 JSON）→ 不解析为事件，stream_status=truncated。
- exit 0 但缺 termination 事件 → missing_termination，不声明完整成功。
- usage 缺失 → Usage(None, None)，unknown ≠ 0。
- stderr 独立保存，不混入事件流；无法解析的行进 raw log 不猜类型。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agentlab.contracts import RuntimeEvent, Usage

SOURCE_PREFIX = {"model_": "model", "tool_": "tool", "runtime_": "runtime",
                 "grader_": "grader"}


@dataclass
class StreamResult:
    events: list[RuntimeEvent] = field(default_factory=list)
    stream_status: str = "complete"     # complete|truncated|missing_termination|protocol_error
    dropped_duplicates: int = 0
    unparsed_lines: list[str] = field(default_factory=list)
    stderr_tail: str = ""
    exit_status: str = "ok"
    usage_total: Usage | None = None


class StreamAssembler:
    def __init__(self, episode_id: str, attempt: int = 1):
        self.episode_id = episode_id
        self.attempt = attempt
        self._seq = 0
        self._seen_event_ids: set[str] = set()
        self._raw: list[tuple[str, Any]] = []

    def feed(self, line: str) -> None:
        line = line.rstrip("\n")
        if not line.strip():
            return
        try:
            envelope = json.loads(line)
        except json.JSONDecodeError:
            self._raw.append(("unparsed", line))
            return
        if not isinstance(envelope, dict) or "event" not in envelope:
            self._raw.append(("unparsed", line))
            return
        eid = envelope.get("event_id")
        if eid is not None:
            if eid in self._seen_event_ids:
                self._seen_dup = getattr(self, "_seen_dup", 0) + 1
                return
            self._seen_event_ids.add(eid)
        self._raw.append(("event", envelope))

    def _emit(self, envelope: dict) -> RuntimeEvent:
        self._seq += 1
        etype = envelope["event"]
        source = next((s for p, s in SOURCE_PREFIX.items()
                       if etype.startswith(p)), "runtime")
        usage = None
        raw_usage = envelope.get("usage")
        if raw_usage is not None:
            usage = Usage(input_tokens=raw_usage.get("input_tokens"),
                          output_tokens=raw_usage.get("output_tokens"))
        return RuntimeEvent(
            episode_id=self.episode_id, attempt=self.attempt,
            step=envelope.get("step"), source=source, event_type=etype,
            timestamp=envelope.get("occurred_at") or "",
            model_revision=envelope.get("model_revision"),
            payload=envelope.get("payload") or {}, usage=usage,
            event_id=envelope.get("event_id"), seq=self._seq)

    def finish(self, exit_status: str, stderr: str) -> StreamResult:
        result = StreamResult(exit_status=exit_status,
                              stderr_tail=stderr[-500:])
        raw_events = [e for kind, e in self._raw if kind == "event"]
        result.unparsed_lines = [l for kind, l in self._raw if kind == "unparsed"]
        result.dropped_duplicates = getattr(self, "_seen_dup", 0)
        # 截断判定：最后一行未以换行结束且解析失败 → truncated
        result.events = [self._emit(env) for env in raw_events]
        if result.unparsed_lines and exit_status == "ok":
            result.stream_status = "truncated"
        has_termination = any(e.event_type == "runtime_termination"
                              for e in result.events)
        if exit_status == "ok" and not has_termination:
            result.stream_status = (result.stream_status
                                    if result.stream_status == "truncated"
                                    else "missing_termination")
        total_in = sum(e.usage.input_tokens or 0 for e in result.events
                       if e.usage and e.usage.input_tokens is not None)
        total_out = sum(e.usage.output_tokens or 0 for e in result.events
                        if e.usage and e.usage.output_tokens is not None)
        any_usage = any(e.usage is not None for e in result.events)
        saw_model_events = any(e.event_type.startswith("model_")
                               for e in result.events)
        if any_usage or saw_model_events:
            known_in = [e.usage.input_tokens for e in result.events
                        if e.usage and e.usage.input_tokens is not None]
            known_out = [e.usage.output_tokens for e in result.events
                         if e.usage and e.usage.output_tokens is not None]
            # 任一维度无样本 → unknown（None），绝不落 0；
            # 有模型事件但完全无 usage → Usage(None,None) 表示"存在但未知"
            result.usage_total = Usage(
                input_tokens=sum(known_in) if known_in else None,
                output_tokens=sum(known_out) if known_out else None)
        return result
