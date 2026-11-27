#!/usr/bin/env python3
"""测 vLLM 服务的生成速度：单路顺序 3 次，再 16 路并发一次。用法：python scripts/vllm_speed.py [端口]"""
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

address = f"http://127.0.0.1:{sys.argv[1] if len(sys.argv) > 1 else '8000'}"
# 本机地址不走代理：环境里若设了 http_proxy 之类的变量，默认的 opener 会把请求转给代理
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
body = json.dumps({
    "model": "Qwen/Qwen3.5-9B",
    "messages": [{"role": "user", "content":
                  "写一个 Python 函数 merge_intervals(intervals)：合并所有重叠的闭区间，"
                  "返回按起点排序的列表，空列表返回空列表。只输出代码。"}],
    "max_tokens": 256, "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
}).encode("utf-8")


def call(_):
    """一次请求：返回 (生成的 token 数, 用时秒)。"""
    request = urllib.request.Request(f"{address}/v1/chat/completions", data=body,
                                    headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with opener.open(request, timeout=900) as response:
        payload = json.load(response)
    return payload["usage"]["completion_tokens"], time.perf_counter() - started


try:
    singles = [call(None) for _ in range(3)]
    print(f"单路：平均 {sum(t for t, _ in singles) / sum(s for _, s in singles):.0f} token/s")
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=16) as pool:
        batch = list(pool.map(call, range(16)))
    elapsed = time.perf_counter() - started
    print(f"16 路并发：共 {sum(t for t, _ in batch)} 个 token，用时 {elapsed:.1f}s，"
          f"合计 {sum(t for t, _ in batch) / elapsed:.0f} token/s")
except urllib.error.HTTPError as error:          # 先于 URLError：它是 URLError 的子类
    sys.exit(f"{address} 返回 {error.code}：{error.read(300).decode('utf-8', 'replace')}")
except urllib.error.URLError as error:
    sys.exit(f"连不上 {address}（{error.reason}）：服务没起来、已经退出，或者端口不对。")
