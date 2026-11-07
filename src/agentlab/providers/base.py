# -*- coding: utf-8 -*-
"""API Provider 统一接口（P4 审查项）。

设计约束：
- API key 只从环境变量读取，不写入日志/轨迹/Git；
- 每次调用记录 token 用量、延迟、重试和估算费用；
- unknown 费用保持 unknown，不填零；
- 错误分类（rate_limit/network/auth/server），不静默吞掉。
"""
from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from agentlab.contracts import Usage


@dataclass
class APIRequest:
    model: str
    messages: list[dict]
    tools: list[dict] | None = None
    max_tokens: int = 4096
    temperature: float = 0.1
    request_id: str = ""


@dataclass
class CostEstimate:
    usd: float | None = None     # None = unknown，不填零
    source: str = "unknown"
    note: str = ""


@dataclass(frozen=True)
class ModelPricing:
    model: str
    source: str
    input_price_per_m: float
    output_price_per_m: float

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("pricing model must be a nonempty string")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("pricing source/version must be a nonempty string")
        if self.source.strip().lower() == "unknown":
            raise ValueError("configured pricing requires a known source/version")
        for name in ("input_price_per_m", "output_price_per_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite nonnegative number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")


@dataclass
class APIResponse:
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    stop_reason: str = ""
    usage: Usage | None = None
    cost: CostEstimate = field(default_factory=CostEstimate)
    latency_ms: int = 0
    model: str = ""
    raw_response_id: str = ""


@dataclass
class APIError(Exception):
    error_type: str       # rate_limit|network|auth|server|unknown
    message: str
    retryable: bool = False
    status_code: int | None = None

    def __str__(self):
        return f"APIError({self.error_type}): {self.message}"


class APIProvider(ABC):
    """统一商业模型接口。子类实现 call()，继承 get_provider_info()。"""

    name: str = "base"
    env_key_name: str = ""       # 如 "DEEPSEEK_API_KEY"

    @abstractmethod
    def call(self, request: APIRequest) -> APIResponse:
        ...

    def get_api_key(self) -> str:
        key = os.environ.get(self.env_key_name, "")
        if not key:
            raise APIError(error_type="auth",
                          message=f"环境变量 {self.env_key_name} 未设置",
                          retryable=False)
        return key

    def get_provider_info(self) -> dict[str, Any]:
        return dict(
            provider=self.name,
            env_key=self.env_key_name,
            has_key=bool(os.environ.get(self.env_key_name, "")),
        )
