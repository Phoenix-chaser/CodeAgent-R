# -*- coding: utf-8 -*-
"""DPO collator：偏好对 → tokenize → x/y 分离 → 候选概率 mask。

x（共同历史）与 y±（候选）分别构造：
- x 部分 labels 全 -100（不参与候选概率）；
- y± 部分仅 assistant 目标 token 有 labels（工具返回/前缀不参与）；
- chosen 与 rejected 用相同 x 前缀（保证条件一致）。
"""
from __future__ import annotations

from dataclasses import dataclass

from agentlab.dpo.data import PreferencePair, verify_no_tool_observation_in_y
from agentlab.training.template import (
    SimpleToolTemplate, SupportsSegmentTokenize, tokenize_segments,
)


@dataclass
class DPOSequence:
    input_ids: list[int]
    labels: list[int]
    pair_id: str
    side: str               # chosen | rejected
    n_target_tokens: int


class DPOCollator:
    def __init__(self, tokenizer: SupportsSegmentTokenize,
                 template: SimpleToolTemplate | None = None):
        self.tokenizer = tokenizer
        self.template = template or SimpleToolTemplate()

    def collate_pair(self, pair: PreferencePair) -> tuple[DPOSequence, DPOSequence]:
        """x + y_chosen 与 x + y_rejected 两条序列。"""
        if not verify_no_tool_observation_in_y(pair.y_chosen):
            raise ValueError("y_chosen contains tool observation")
        if not verify_no_tool_observation_in_y(pair.y_rejected):
            raise ValueError("y_rejected contains tool observation")
        sequences = []
        for side, y in (("chosen", pair.y_chosen), ("rejected", pair.y_rejected)):
            # x 部分的消息（全不监督）
            x_msgs = []
            for i, m in enumerate(pair.x_messages):
                x_msgs.append({**m, "message_id": f"x-{i}"})
            # y 部分（只监督 y）
            y_msg = {**y, "message_id": "y-target"}
            all_msgs = x_msgs + [y_msg]
            input_ids, supervise_mask, _ = tokenize_segments(
                self.template, self.tokenizer, all_msgs, {"y-target"})
            labels = [
                token_id if supervise else -100
                for token_id, supervise in zip(input_ids, supervise_mask)]
            sequences.append(DPOSequence(
                input_ids=input_ids, labels=labels, pair_id=pair.pair_id,
                side=side, n_target_tokens=sum(supervise_mask)))
        return sequences[0], sequences[1]
