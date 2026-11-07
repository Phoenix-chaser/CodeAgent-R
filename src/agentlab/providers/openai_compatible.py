# -*- coding: utf-8 -*-
"""OpenAI 兼容请求适配器；离线测试通过注入客户端完成。

真实调用需要 API key、openai SDK 和网络。一次 call 仅发送一次 SDK 请求，
SDK 自动重试关闭；费用必须使用调用方提供的模型级来源价表。
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable

from agentlab.providers.base import (
    APIError, APIProvider, APIRequest, APIResponse, CostEstimate, ModelPricing,
)
from agentlab.contracts import Usage

PROVIDERS = {
    "deepseek": {
        "env_key": "DEEPSEEK_API_KEY",
        "base_url": "https://secondary-api.example/v1",
        "default_model": "secondary-model",
    },
     "primary": {
        "env_key": "TEACHER_API_KEY",
        "base_url": "https://teacher-api.example/api/paas/v4",
        "default_model": "teacher-api-model",
    },
     "tertiary": {
        "env_key": "MOONSHOT_API_KEY",
        "base_url": "https://api.moonshot.cn/v1",
        "default_model": "moonshot-v1-32k",
    },
}


class OpenAICompatibleProvider(APIProvider):
    def __init__(self, provider_name: str, model: str | None = None,
                 pricing: ModelPricing | None = None,
                 client_factory: Callable[..., Any] | None = None,
                 timeout: float = 60.0):
        if provider_name not in PROVIDERS:
            raise ValueError(f"unknown provider: {provider_name}")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) \
                or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a finite positive number")
        if pricing is not None and not isinstance(pricing, ModelPricing):
            raise ValueError("pricing must be ModelPricing or None")
        config = PROVIDERS[provider_name]
        self.name = provider_name
        self.env_key_name = config["env_key"]
        self.base_url = config["base_url"]
        self.model = model or config["default_model"]
        self.pricing = pricing
        self.client_factory = client_factory
        self.timeout = timeout

    def estimate_cost(self, usage: Usage | None,
                      model: str | None = None) -> CostEstimate:
        if usage is None or usage.input_tokens is None or usage.output_tokens is None:
            return CostEstimate(note="usage incomplete")
        if usage.input_tokens < 0 or usage.output_tokens < 0:
            return CostEstimate(note="usage invalid")
        effective_model = model if model is not None else self.model
        if self.pricing is None:
            return CostEstimate(note="model pricing not configured")
        if self.pricing.model != effective_model:
            return CostEstimate(note="pricing model mismatch")
        cost = (usage.input_tokens * self.pricing.input_price_per_m
                + usage.output_tokens * self.pricing.output_price_per_m) / 1e6
        if not math.isfinite(cost):
            return CostEstimate(note="cost overflow")
        return CostEstimate(usd=cost, source=self.pricing.source)

    def _api_error(self, error: Exception) -> APIError:
        status = getattr(error, "status_code", None)
        if status is None:
            status = getattr(getattr(error, "response", None), "status_code", None)
        if not isinstance(status, int) or isinstance(status, bool):
            status = None
        exception_names = {kind.__name__ for kind in type(error).__mro__}
        network_error = isinstance(error, (ConnectionError, TimeoutError)) or bool(
            exception_names & {"APIConnectionError", "APITimeoutError", "NetworkError",
                               "ConnectError", "ConnectTimeout", "ReadTimeout",
                               "WriteTimeout", "PoolTimeout"})
        if status == 429:
            category, retryable = "rate_limit", True
        elif status in (401, 403):
            category, retryable = "auth", False
        elif status == 408 or network_error:
            category, retryable = "network", True
        elif status is not None and 500 <= status < 600:
            category, retryable = "server", True
        else:
            category, retryable = "unknown", False
        return APIError(
            error_type=category,
            message=f"{self.name} request failed ({category}, status={status})",
            retryable=retryable, status_code=status)

    def call(self, request: APIRequest) -> APIResponse:
        """一次 SDK 请求，不自动重试；供应商异常文本不进入可记录错误。"""
        api_key = self.get_api_key()
        factory = self.client_factory
        if factory is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise APIError("unknown", "openai SDK is not installed") from None
            factory = OpenAI
        requested_model = request.model or self.model
        start = time.monotonic()
        client = None
        try:
            client = factory(api_key=api_key, base_url=self.base_url,
                             timeout=self.timeout, max_retries=0)
            response = client.chat.completions.create(
                model=requested_model, messages=request.messages,
                max_tokens=request.max_tokens, temperature=request.temperature,
                tools=request.tools or None)
        except Exception as error:
            raise self._api_error(error) from None
        finally:
            if client is not None:
                close = getattr(client, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
        latency = int((time.monotonic() - start) * 1000)
        if not getattr(response, "choices", None):
            raise APIError("server", "provider returned no completion choices",
                           retryable=False)
        try:
            choice = response.choices[0]
            message = choice.message
            raw_usage = getattr(response, "usage", None)
            usage = Usage(
                input_tokens=getattr(raw_usage, "prompt_tokens", None),
                output_tokens=getattr(raw_usage, "completion_tokens", None))
            response_model = getattr(response, "model", None) or requested_model
            cost = self.estimate_cost(usage, requested_model)
            if response_model != requested_model:
                cost = CostEstimate(note="response model differs from requested model")
            return APIResponse(
                content=message.content or "",
                tool_calls=[dict(id=call.id, name=call.function.name,
                                 arguments=call.function.arguments)
                            for call in (message.tool_calls or [])],
                stop_reason=choice.finish_reason or "",
                usage=usage, cost=cost, latency_ms=latency,
                model=response_model,
                raw_response_id=getattr(response, "id", None) or "")
        except (AttributeError, IndexError, TypeError, ValueError):
            raise APIError("server", "provider returned an invalid response",
                           retryable=False) from None
