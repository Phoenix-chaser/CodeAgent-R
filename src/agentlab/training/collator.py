# -*- coding: utf-8 -*-
"""SFT collator：tokenize → 监督 mask → 截断 → padding / packing。

规则（implementation.md §3/§5）：
- labels 非 target 一律 -100（含 system/user/tool observation/前缀 assistant 历史/padding）；
- 截断在消息与 tool-call/observation 配对层完成，超长 observation 按冻结规则截断并标记；
  超长目标 JSON/patch 丢弃样本（不截断成非法目标）；
- packing：块对角 attention——样本之间不互相注意；
- 全 -100 样本拒绝并计数（防损失分母为零）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agentlab.training.data import SFTSample
from agentlab.training.template import (
    SimpleToolTemplate, SupportsSegmentTokenize, tokenize_segments,
    verify_segment_consistency,
)

MAX_OBS_CHARS = 2000
MAX_TARGET_TOKENS = 4096


@dataclass
class CollatedSample:
    input_ids: list[int]
    labels: list[int]              # -100 or token id（监督位）
    supervise_mask: list[int]      # 0/1 对齐
    sample_id: str


@dataclass
class CollatedBatch:
    samples: list[CollatedSample] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)  # (id, reason)
    total_tokens: int = 0
    supervised_tokens: int = 0


class SFTCollator:
    def __init__(self, tokenizer: SupportsSegmentTokenize,
                 template: SimpleToolTemplate | None = None,
                 verify_consistency: bool = True):
        self.tokenizer = tokenizer
        self.template = template or SimpleToolTemplate()
        self.verify = verify_consistency
        self._verified = False

    def _truncate_messages(self, msgs: list[dict]) -> tuple[list[dict], bool]:
        """observation 级截断（配对完整，标记 truncated）。"""
        out, truncated = [], False
        for m in msgs:
            if m.get("role") == "tool":
                obs = m.get("observation", "")
                if len(obs) > MAX_OBS_CHARS:
                    m = {**m, "observation": obs[:MAX_OBS_CHARS] + "\n[obs-truncated]"}
                    truncated = True
            out.append(m)
        return out, truncated

    def collate_one(self, sample: SFTSample) -> CollatedSample | None:
        msgs, _ = self._truncate_messages(sample.messages)
        target_ids = set(sample.target_message_ids)
        if self.verify and not self._verified:
            if not verify_segment_consistency(self.template, self.tokenizer,
                                              msgs, target_ids):
                raise RuntimeError(
                    "segment/full tokenization mismatch: adapter must use "
                    "offset-mapping mask rebuild (see template.py)")
            self._verified = True
        ids, supervise, spans = tokenize_segments(
            self.template, self.tokenizer, msgs, target_ids)
        # 目标 token 超限 → 丢弃（不截断成非法目标）
        n_sup = sum(supervise)
        if n_sup == 0:
            return None
        if n_sup > MAX_TARGET_TOKENS:
            return None
        labels = [i if sup == 1 else -100 for i, sup in zip(ids, supervise)]
        return CollatedSample(input_ids=ids, labels=labels,
                              supervise_mask=supervise,
                              sample_id=sample.sample_id)

    def collate(self, samples: list[SFTSample]) -> CollatedBatch:
        batch = CollatedBatch()
        for s in samples:
            one = self.collate_one(s)
            if one is None:
                n_sup_guess = len(s.target_message_ids)
                batch.rejected.append((s.sample_id,
                                       "zero_supervision" if n_sup_guess == 0
                                       else "target_too_long"))
                continue
            batch.samples.append(one)
            batch.total_tokens += len(one.input_ids)
            batch.supervised_tokens += sum(one.supervise_mask)
        return batch

    def pack(self, samples: list[CollatedSample],
             block_size: int) -> dict[str, list[list[int]]]:
        """装箱 + 块对角 attention：样本间不互相注意。

        返回 {blocks: [[input_ids...]], attention: [[0/1...]],
              position_of_sample: [(block_idx, start, end, sample_id)]}
        """
        blocks, attn, placement = [], [], []
        cur_ids, cur_attn, cur_len = [], [], 0
        for s in samples:
            if cur_len + len(s.input_ids) > block_size and cur_ids:
                blocks.append(cur_ids)
                attn.append(cur_attn)
                cur_ids, cur_attn, cur_len = [], [], 0
            start = cur_len
            cur_ids += s.input_ids
            cur_attn += [1] * len(s.input_ids)
            placement.append((len(blocks), start, start + len(s.input_ids),
                              s.sample_id))
            cur_len += len(s.input_ids)
        if cur_ids:
            blocks.append(cur_ids)
            attn.append(cur_attn)
        # 块内 attention 矩阵语义由消费方按 placement 构造块对角
        return dict(blocks=blocks, attention=attn,
                    sample_placement=placement)


def block_diagonal_attention(placement: list[tuple], seq_len: int,
                             block_idx: int) -> list[list[int]]:
    """给定一个 block 的样本布局，返回该 block 的块对角 attention 矩阵。"""
    attn = [[0] * seq_len for _ in range(seq_len)]
    for b, start, end, _sid in placement:
        if b != block_idx:
            continue
        for i in range(start, end):
            for j in range(start, i + 1):    # causal + 块内
                attn[i][j] = 1
    return attn
