"""访问 vLLM 服务的最小客户端：只用标准库，并且固定不走代理。

只依赖标准库，是因为采样要在服务器现成的环境里运行；不走代理，是因为环境里的
``http_proxy`` 会让默认的 opener 把发给 127.0.0.1 的请求转给代理，得到“连接被拒”。
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

DEFAULT_URL = "http://127.0.0.1:8000"
RETRIES = 4


class VLLMError(RuntimeError):
    """服务返回了错误，或者连不上。``transient`` 为真时重试可能成功。"""

    def __init__(self, message: str, *, transient: bool = False):
        super().__init__(message)
        self.transient = transient


def _request(url: str, payload: dict | None, timeout: float) -> dict:
    request = urllib.request.Request(
        url, data=None if payload is None else json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        # 每次请求都新建 opener：代理设置不会被缓存，环境变量改了也立刻生效
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
                request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read(500).decode("utf-8", "replace").strip()
        raise VLLMError(f"{url} 返回 {error.code}：{body}", transient=error.code >= 500) from error
    except urllib.error.URLError as error:
        raise VLLMError(f"连不上 {url}（{error.reason}）：服务没起来，或者已经退出",
                        transient=True) from error
    except (TimeoutError, OSError) as error:            # 读超时、连接被重置
        raise VLLMError(f"{url} 请求失败：{error}", transient=True) from error


def _with_retries(call: Callable[[], dict]) -> dict:
    delay = 2.0
    for attempt in range(1, RETRIES + 1):
        try:
            return call()
        except VLLMError as error:
            if attempt == RETRIES or not error.transient:
                raise
            time.sleep(delay)
            delay *= 2
    raise AssertionError("unreachable")


def server_models(url: str = DEFAULT_URL, timeout: float = 10.0) -> list[str]:
    """服务上加载的模型名。"""
    return [entry["id"] for entry in _request(f"{url}/v1/models", None, timeout)["data"]]


def wait_until_ready(url: str = DEFAULT_URL, timeout: float = 900.0, interval: float = 5.0) -> list[str]:
    """等到服务能回答 ``/v1/models``。超时时抛出最后一次的错误。"""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return server_models(url)
        except VLLMError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(interval)


def chat(url: str, model: str, messages: list[dict], *, temperature: float = 0.7,
         top_p: float = 1.0, top_k: int | None = None, max_tokens: int = 2048,
         seed: int | None = None, enable_thinking: bool = False,
         timeout: float = 600.0) -> dict:
    """一次对话补全，返回 {text, finish_reason, prompt_tokens, completion_tokens}。

    提示由服务端按模型自带的 chat template 渲染；``enable_thinking`` 与训练、评测
    统一使用的设置一致（默认关闭）。
    """
    payload = {
        "model": model, "messages": messages, "temperature": temperature, "top_p": top_p,
        "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if seed is not None:
        payload["seed"] = seed
    if top_k is not None:                  # vLLM 的扩展参数；-1 表示不做 top-k 截断
        payload["top_k"] = top_k
    data = _with_retries(lambda: _request(f"{url}/v1/chat/completions", payload, timeout))
    choice = data["choices"][0]
    usage = data.get("usage") or {}
    return {"text": (choice["message"].get("content") or ""),
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens")}


def sampling_options(url: str, model: str) -> dict[str, bool]:
    """探测服务是否接受 ``seed`` 和 ``top_k`` 这两个扩展参数。

    各个版本对扩展参数的接受程度不同，不接受的发过去会得到 400。用一次只生成
    1 个 token 的请求来试；结果记进采样协议，不支持的参数就不发。
    """
    accepted = {"seed": False, "top_k": False}
    messages = [{"role": "user", "content": "1"}]
    for names in (("seed", "top_k"), ("seed",), ()):
        extra = {name: 1 if name == "seed" else 20 for name in names}
        try:
            _request(f"{url}/v1/chat/completions", {
                "model": model, "messages": messages, "max_tokens": 1, "temperature": 1,
                **extra}, timeout=120)
        except VLLMError as error:
            if error.transient:                # 连不上、超时、5xx：不是参数的问题
                raise
            continue
        for name in names:
            accepted[name] = True
        break
    return accepted


def map_concurrent(fn: Callable[[Any], Any], items: Sequence[Any],
                   workers: int) -> Iterator[tuple[int, Any]]:
    """并发地对每个元素调用 ``fn``，按完成的先后产出 (下标, 结果)；异常在产出时抛出。

    调用方想让个别失败不中断整批时，在 ``fn`` 里捕获异常并作为结果返回。
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    items = list(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, item): index for index, item in enumerate(items)}
        for future in as_completed(futures):
            yield futures[future], future.result()
