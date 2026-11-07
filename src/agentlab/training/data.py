# -*- coding: utf-8 -*-
"""SFT 数据四种模式（S1–S4）与 provenance。"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field

MODES = {"S1_final_patch", "S2_success_traj", "S3_recovery", "S4_mixed"}


@dataclass
class SFTSample:
    sample_id: str
    mode: str
    task_id: str
    source_group: str
    split: str
    episode_id: str
    messages: list[dict]
    target_message_ids: list[str]
    provenance: dict = field(default_factory=dict)
    trusted_label_ref: str = ""       # 永不进入模型输入
    filter_status: str = "kept"       # kept|dropped|pending
    drop_reason: str = ""

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown mode: {self.mode}")
        ids = {m.get("message_id") for m in self.messages}
        missing = set(self.target_message_ids) - ids
        if missing:
            raise ValueError(f"target ids not in messages: {missing}")
        # 结构性防未来信息：label ref 不得出现在任何消息内容里
        blob = json.dumps(self.messages, ensure_ascii=False)
        if self.trusted_label_ref and self.trusted_label_ref in blob:
            raise ValueError("trusted label ref leaked into messages")


def make_s3_recovery(sample_id: str, task_id: str, source_group: str,
                     split: str, episode_id: str,
                     failure_prefix: list[dict],
                     recovery_suffix: list[dict],
                     provenance: dict, trusted_label_ref: str) -> SFTSample:
    """失败前缀（不监督）+ 验证过的恢复后缀（监督目标）。"""
    msgs = []
    for i, m in enumerate(failure_prefix):
        msgs.append({**m, "message_id": f"prefix-{i}"})
    targets = []
    for i, m in enumerate(recovery_suffix):
        mid = f"recovery-{i}"
        msgs.append({**m, "message_id": mid})
        targets.append(mid)
    s = SFTSample(sample_id=sample_id, mode="S3_recovery", task_id=task_id,
                  source_group=source_group, split=split, episode_id=episode_id,
                  messages=msgs, target_message_ids=targets,
                  provenance=provenance, trusted_label_ref=trusted_label_ref)
    s.validate()
    return s


def export_manifest(samples: list[SFTSample], path) -> dict:
    stats = dict(total=len(samples), kept=sum(s.filter_status == "kept" for s in samples),
                 dropped=sum(s.filter_status == "dropped" for s in samples),
                 by_mode={}, supervision_tokens=0)
    for s in samples:
        stats["by_mode"][s.mode] = stats["by_mode"].get(s.mode, 0) + 1
    manifest = dict(schema_version=1, stats=stats,
                    created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    content_hash=hashlib.sha256(
                        json.dumps([s.sample_id for s in samples]).encode()
                    ).hexdigest()[:16])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest
