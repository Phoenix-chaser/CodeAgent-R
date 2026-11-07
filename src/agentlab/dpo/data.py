# -*- coding: utf-8 -*-
"""DPO 偏好对数据契约与共同状态校验。

规则（implementation.md §2/§3）：
- chosen/rejected 必须共享 task/prefix_hash/snapshot/工具 schema/环境/剩余预算；
- y 是下一条完整 assistant 消息（可含 tool-call），不含工具返回与后续；
- 偏好依据：合法且严格成功为主；费用比较仅在成功表现相近时按冻结规则；
- 平局/缺标签 → 不硬造偏好；
- 长度与费用记录用于分析，不默认短输出正确。
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field

REQUIRED_SHARED_KEYS = (
    "task_id", "source_group", "split",
    "public_prefix_hash", "snapshot_id",
    "workspace_hash", "environment_digest",
    "tool_schema_hash", "remaining_budget",
    "continuation_policy_version",
)


@dataclass
class PreferencePair:
    pair_id: str
    x_messages: list[dict]                 # 共同可见历史（不含工具返回后的 y 时代内容）
    y_chosen: dict                         # 完整 assistant 消息
    y_rejected: dict
    shared_keys: dict                      # 共同状态键
    chosen_cost: dict = field(default_factory=dict)
    rejected_cost: dict = field(default_factory=dict)
    chosen_verdict: str = ""
    rejected_verdict: str = ""
    preference_reason: str = ""
    chosen_token_len: int | None = None
    rejected_token_len: int | None = None
    confidence: str = "single_rollout"     # single_rollout | repeated_verified
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S%z"))

    def validate(self) -> None:
        missing = [k for k in REQUIRED_SHARED_KEYS if k not in self.shared_keys]
        if missing:
            raise ValueError(f"shared_keys missing: {missing}")
        if self.chosen_verdict != "strict_pass":
            raise ValueError(
                f"chosen must be strict_pass (got {self.chosen_verdict}); "
                "cannot use failure as positive")
        if self.preference_reason == "cost_only":
            raise ValueError(
                "cost-only preference requires matched success; "
                "use success_vs_failure or set reason explicitly")

    def __post_init__(self):
        self.validate()


def make_pair(x_messages: list[dict], y_chosen: dict, y_rejected: dict,
              shared_keys: dict, chosen_verdict: str, rejected_verdict: str,
              preference_reason: str = "success_vs_failure",
              **kwargs) -> PreferencePair:
    pair = PreferencePair(
        pair_id=kwargs.pop("pair_id", f"pref-{int(time.time()*1000)}"),
        x_messages=x_messages, y_chosen=y_chosen, y_rejected=y_rejected,
        shared_keys=shared_keys,
        chosen_verdict=chosen_verdict, rejected_verdict=rejected_verdict,
        preference_reason=preference_reason, **kwargs)
    return pair


def check_pair_state_consistency(pair: PreferencePair,
                                 other_shared: dict) -> list[str]:
    """两个来源的 shared_keys 必须一致；不一致返回差异字段列表。"""
    diffs = []
    for k in REQUIRED_SHARED_KEYS:
        if pair.shared_keys.get(k) != other_shared.get(k):
            diffs.append(k)
    return diffs


def verify_no_tool_observation_in_y(y: dict) -> bool:
    """y 不含工具返回（observation / tool_result 字段）。"""
    blob = json.dumps(y, ensure_ascii=False)
    return ("observation" not in blob and "tool_result" not in blob)


def verify_no_future_info_in_x(x: list[dict]) -> bool:
    """x 不含终局标签、隐藏评分、参考补丁摘要。"""
    blob = json.dumps(x, ensure_ascii=False)
    banned = ("strict_pass", "strict_fail", "reference_patch", "hidden_test",
              "private_grader", "CANARY")
    return not any(b in blob for b in banned)


def pair_hash(pair: PreferencePair) -> str:
    blob = json.dumps(dict(x=pair.x_messages, yc=pair.y_chosen,
                           yr=pair.y_rejected,
                           shared=pair.shared_keys), ensure_ascii=False,
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
