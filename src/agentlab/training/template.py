# -*- coding: utf-8 -*-
"""chat 模板协议与确定性工具模板（离线可测）。

关键约束（implementation.md §3）：token 级 mask 必须基于模型实际序列化
结果，不能假设逐条消息独立 tokenize 拼接 == 完整模板 tokenize。
本模块以「分段渲染 + 分段 tokenize + 边界记录」实现，并提供
verify_segment_consistency() 钩子——真实 tokenizer 适配器必须通过该
一致性检查后才允许进入训练。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

TEMPLATE_VERSION = "simple_tool_v1"


@dataclass
class Segment:
    """模板渲染的一段文本及其监督属性。"""
    text: str
    message_id: str | None      # None = 模板插入的结构 token
    supervise: bool             # 是否参与目标监督
    kind: str                   # context|prefix|target_text|target_tool|eot|pad


class SupportsSegmentTokenize(Protocol):
    def tokenize(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


class SimpleToolTemplate:
    """确定性模板：角色标记成对、assistant 生成前缀、工具调用结构化。"""

    version = TEMPLATE_VERSION

    def render(self, messages: list[dict],
               target_message_ids: set[str]) -> list[Segment]:
        segs: list[Segment] = []
        segs.append(Segment("<|sys|>", None, False, "context"))
        for m in messages:
            mid = m.get("message_id", "")
            role = m["role"]
            if role == "system":
                segs += [Segment(m["content"], mid, False, "context"),
                         Segment("<|/sys|>", None, False, "context")]
            elif role == "user":
                segs.append(Segment(f"<|user|>{m['content']}<|/user|>",
                                    mid, False, "context"))
            elif role == "assistant":
                is_target = mid in target_message_ids
                segs.append(Segment("<|assistant|>", None, False, "prefix"))
                if is_target:
                    segs.append(Segment(m["content"], mid, True, "target_text"))
                    for call in m.get("tool_calls", []):
                        args = str(call.get("args", {}))
                        segs.append(Segment(
                            f"<|tool_call|>{call.get('name','')}|{args}",
                            mid, True, "target_tool"))
                    segs.append(Segment("<|eot|>", None, True, "eot"))
                else:
                    # 恢复 checkpoint 之前的 assistant 历史：仅上下文，不监督
                    segs.append(Segment(m["content"], mid, False, "context"))
                    for call in m.get("tool_calls", []):
                        segs.append(Segment(
                            f"<|tool_call|>{call.get('name','')}|{call.get('args','')}",
                            mid, False, "context"))
                    segs.append(Segment("<|eot|>", None, False, "context"))
            elif role == "tool":
                segs.append(Segment(
                    f"<|tool_result|>{m.get('call_id','')}|{m.get('observation','')}"
                    f"<|/tool_result|>", mid, False, "context"))
        return segs


def tokenize_segments(template, tokenizer: SupportsSegmentTokenize,
                      messages: list[dict], target_message_ids: set[str]):
    """分段 tokenize 并保留 token 级监督边界。"""
    segs = template.render(messages, target_message_ids)
    token_ids: list[int] = []
    labels: list[int] = []          # 与 token 对齐的监督标记（0/1）
    spans: list[dict] = []
    for seg in segs:
        ids = tokenizer.tokenize(seg.text)
        start = len(token_ids)
        token_ids += ids
        labels += [1 if seg.supervise else 0] * len(ids)
        spans.append(dict(kind=seg.kind, message_id=seg.message_id,
                          token_start=start, token_end=start + len(ids),
                          supervise=seg.supervise))
    return token_ids, labels, spans


def verify_segment_consistency(template, tokenizer, messages,
                               target_message_ids) -> bool:
    """分段拼接 tokenize == 整体 tokenize 的一致性检查钩子。

    直接对比 token 序列（不经 decode 往返）。真实 BPE tokenizer 可能
    不满足此性质（跨段合并）；不满足时适配器必须改用 offset-mapping
    重建 mask，禁止静默沿用分段 mask。
    """
    segs = template.render(messages, target_message_ids)
    concat_ids: list[int] = []
    for s in segs:
        concat_ids += tokenizer.tokenize(s.text)
    full_text = "".join(s.text for s in segs)
    return concat_ids == tokenizer.tokenize(full_text)
