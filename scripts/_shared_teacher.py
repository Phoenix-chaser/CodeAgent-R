"""访问教师接口的公共部分：建客户端、分辨瞬时错误、带退避的重试、进程内的节流。

读法同 collect_data.py：用 openai SDK，密钥、地址、模型名从环境变量来（TEACHER_API_KEY、
TEACHER_BASE_URL、TEACHER_MODEL）。SDK 自带的重试关掉（max_retries=0），改由 ``chat_text``
重试：参数不对、密钥不对这类错误重试多少次都一样，无人值守时只会白白多等，立刻抛。

瞬时错误分两种，重法也不一样：
- 限流（429）和服务端错误（5xx）：限额按账户算，成片出现时过一阵能自己恢复，所以长时间重试。第一次等
  15 秒，之后每次翻倍，最长 120 秒；响应头里有 Retry-After 就按它等；从第一次这样的失败算起等满
  30 分钟（``retry_seconds``）才放弃。每等一次往标准错误打一行，看日志就知道是在等、不是卡死了；
- 连不上、超时：一共最多请求 4 次（``retries``），间隔 2、4、8 秒。

进程内的节流（``Throttle``）：限额按账户算，同一个进程里所有的调用点共用一份（``THROTTLE``）——同一
时刻在飞的请求不超过 2 个（``max_in_flight``），相邻两次请求的开始至少隔 1 秒（``min_interval``）。两个
数都可以用 ``configure_throttle`` 改；脚本按命令行参数在 main 里改一次。

TEACHER_EXTRA_JSON：环境变量里是一个 JSON 对象时，原样合并进每次请求的请求体（同名以它为准），不用改
代码就能传接口特有的参数。没设就是空的；不是 JSON 对象时 ``extra_json`` 抛 ValueError，话里说清楚。

openai 只在函数里导入：本机和 CI 可能没装它，单元测试里注入假对象。
"""
from __future__ import annotations

import contextlib
import datetime
import email.utils
import json
import os
import sys
import threading
import time

DEFAULT_BASE_URL = "https://teacher-api.example/api/paas/v4"
DEFAULT_MODEL = "teacher-api-model"
KEY_ENV = "TEACHER_API_KEY"
EXTRA_ENV = "TEACHER_EXTRA_JSON"
# 瞬时错误的类名。按类名（连同基类）判断而不导入 openai：没装 openai 也能用，标准库的
# ConnectionError、TimeoutError 也认得。
TRANSIENT_CLASS_NAMES = frozenset({"APIConnectionError", "APITimeoutError", "TimeoutError", "ConnectionError"})
BACKOFF_SECONDS = (2.0, 4.0, 8.0)          # 连不上、超时：第 n 次失败后等第 n 个，用完最后一个
# 限流（429）、服务端错误（5xx）：第一次等这么久，之后每次翻倍，最长等这么久；从第一次失败算起等满这么久才放弃
FIRST_WAIT, MAX_WAIT, RETRY_SECONDS = 15.0, 120.0, 30 * 60
DEFAULT_IN_FLIGHT, DEFAULT_INTERVAL = 2, 1.0     # 同一时刻在飞的请求上限、相邻两次请求的最小间隔（秒）


def _status_code(error: BaseException) -> int | None:
    """HTTP 状态码；连接根本没建立时没有，返回 None。"""
    try:
        status = getattr(error, "status_code", None)
        if status is None:
            status = getattr(getattr(error, "response", None), "status_code", None)
    except Exception:                     # 个别异常类取属性时自己会抛：当作没有状态码
        return None
    return status if isinstance(status, int) and not isinstance(status, bool) else None


def transient(error: BaseException) -> bool:
    """这次失败重试一次有没有可能成功：限流（429）、服务端错误（5xx）、连不上或超时。

    两样东西满足一样就算：异常上的 ``status_code``；异常的类名（含基类）是不是连接或超时那一类。
    """
    status = _status_code(error)
    by_status = status is not None and (status == 429 or status >= 500)
    by_name = bool(TRANSIENT_CLASS_NAMES.intersection(cls.__name__ for cls in type(error).__mro__))
    return by_status or by_name


def limited(error: BaseException) -> bool:
    """限流（429）或者服务端错误（5xx）：这两种要长时间重试，连接、超时那一种不在此列。"""
    status = _status_code(error)
    return status is not None and (status == 429 or status >= 500)


def retry_after(error: BaseException) -> float | None:
    """响应头 Retry-After 要求等多久（秒）；没有这个头，或者头里写的东西读不出来，返回 None（按退避等）。

    头里可以写秒数，也可以写一个 HTTP 日期（等到那个时刻）。接口最清楚限额什么时候恢复，所以写了就照它等。
    """
    try:
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
        if headers is None:
            headers = getattr(error, "headers", None)
        pairs = headers.items() if hasattr(headers, "items") else ()
        raw = next((value for key, value in pairs if str(key).lower() == "retry-after"), None)
        if raw is None:
            return None
        try:
            seconds = float(raw)
        except (TypeError, ValueError):
            when = email.utils.parsedate_to_datetime(str(raw))
            if when is None:              # 既不是秒数也不是日期：当没有这个头，按退避等
                return None
            if when.tzinfo is None:
                when = when.replace(tzinfo=datetime.timezone.utc)
            return max((when - datetime.datetime.now(datetime.timezone.utc)).total_seconds(), 0.0)
        return seconds if seconds >= 0 else None      # 负数、不是数：同上，不照它等
    except Exception:                     # 异常对象长得不一样：当作没有这个头
        return None


def extra_json(value: str | None = None) -> dict:
    """合并进每次请求请求体的那个 JSON 对象：环境变量 TEACHER_EXTRA_JSON，没设（或者是空白）就是空字典。

    ``value`` 给了就解析它（测试用）。用来在不改代码的前提下传接口特有的参数。不是 JSON，或者不是对象
    （是数组、字符串……）时抛 ValueError：悄悄不发、或者发出去被接口拒掉，都不如在这儿说清楚。
    """
    raw = os.environ.get(EXTRA_ENV, "") if value is None else value
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError as error:
        raise ValueError(f"环境变量 {EXTRA_ENV} 要是一个 JSON 对象（例如 {{\"a\": 1}}），现在不是合法的 JSON："
                         f"{error}") from None
    if not isinstance(parsed, dict):
        raise ValueError(f"环境变量 {EXTRA_ENV} 要是一个 JSON 对象（例如 {{\"a\": 1}}），现在解析出来是"
                         f"{type(parsed).__name__}")
    return parsed


def other_text(message) -> dict:
    """消息上正文以外的文字字段各有多长：{字段名: 字符数}。只看长度，不存内容。

    思考类模型把推理放在单独的字段里（字段名各家不一样，所以不写死名字，消息上有什么就记什么）：正文是空的
    而它很长，说明输出到上限的时候还停在推理里、正文一个字没写。读不出来就是空字典。
    """
    try:
        fields = {}
        extra = getattr(message, "model_extra", None)        # SDK 的消息对象把不认识的字段放在这里
        if isinstance(extra, dict):
            fields.update(extra)
        if hasattr(message, "__dict__"):
            fields.update(vars(message))
        return {name: len(value) for name, value in sorted(fields.items())
                if isinstance(value, str) and value and not name.startswith("_") and name not in ("role", "content")}
    except Exception:
        return {}


def _seconds(amount: float) -> str:
    """给用户看的秒数：整数就不带小数点。"""
    return str(int(amount)) if amount == int(amount) else f"{amount:.1f}"


class Throttle:
    """进程内的节流：同一时刻在飞的请求不超过 ``max_in_flight`` 个，相邻两次请求的开始至少隔 ``min_interval`` 秒。

    限额按账户算，所以同一个进程里所有的调用点共用一份（模块级的 THROTTLE），不能各算各的。一把锁
    （Condition）加两个数：在飞的请求数、上一次请求开始的时刻。等在里面的请求不占名额，轮到了才占。
    """

    def __init__(self, max_in_flight: int = DEFAULT_IN_FLIGHT, min_interval: float = DEFAULT_INTERVAL):
        self._condition = threading.Condition()
        self._in_flight = 0
        self._last_start = None
        self.max_in_flight, self.min_interval = max_in_flight, min_interval

    def configure(self, max_in_flight: int | None = None, min_interval: float | None = None) -> None:
        """改设置，马上生效（已经排着的请求按新设置放行）。不给的项不动。"""
        with self._condition:
            if max_in_flight is not None:
                if max_in_flight < 1:
                    raise ValueError("max_in_flight must be at least 1")
                self.max_in_flight = max_in_flight
            if min_interval is not None:
                if min_interval < 0:
                    raise ValueError("min_interval cannot be negative")
                self.min_interval = min_interval
            self._condition.notify_all()

    def acquire(self) -> float:
        """排到可以发请求为止，返回这次请求开始的时刻（已经占了一个名额）。"""
        with self._condition:
            while True:
                now = time.monotonic()
                earliest = 0.0 if self._last_start is None else self._last_start + self.min_interval
                if self._in_flight < self.max_in_flight and now >= earliest:
                    self._in_flight += 1
                    self._last_start = now
                    return now
                # 名额已经满了就等到有人放出来（不定时）；只差时间就等到那个时候
                self._condition.wait(max(earliest - now, 0.001) if self._in_flight < self.max_in_flight else None)

    def release(self) -> None:
        with self._condition:
            self._in_flight -= 1
            self._condition.notify_all()

    @contextlib.contextmanager
    def request(self):
        """``with throttle.request() as started:`` 期间在飞，进出都记在账上；请求抛了异常名额也放回去。"""
        started = self.acquire()
        try:
            yield started
        finally:
            self.release()

    def state(self) -> dict:
        """现在的样子：在飞几个、上一次请求是什么时候开始的（还没有请求过就是 None）。"""
        with self._condition:
            return {"in_flight": self._in_flight, "last_start": self._last_start}


THROTTLE = Throttle()


def configure_throttle(max_in_flight: int | None = None, min_interval: float | None = None) -> None:
    """改进程里共用的那一份节流的设置。脚本在 main 里按命令行参数调一次，之后所有的请求都按它来。"""
    THROTTLE.configure(max_in_flight, min_interval)


def chat_text(client, model: str, messages: list[dict], *, max_tokens: int, temperature: float, retries: int = 4,
              retry_seconds: float = RETRY_SECONDS, first_wait: float = FIRST_WAIT, max_wait: float = MAX_WAIT,
              throttle: Throttle | None = None, extra: dict | None = None) -> dict:
    """一次对话补全，返回 {"text", "finish_reason", "usage": {"input_tokens", "output_tokens"}, "other_text"}。

    ``other_text`` 是正文以外的文字字段各有多长（见 other_text）：只记长度。接口没报用量时 ``usage`` 里是两个 None。

    瞬时错误怎么重试见模块说明：限流（429）、服务端错误（5xx）长时间重试——第一次等 ``first_wait`` 秒，之后
    每次翻倍，最长 ``max_wait``，响应头 Retry-After 优先，从第一次这样的失败算起满 ``retry_seconds`` 才放弃
    （放弃前抛最后那个错误，等待的这满一段时间不会超过上限）；连不上、超时一共最多请求 ``retries`` 次。
    其余异常直接抛。

    ``throttle`` 不给就用进程里共用的那一份；``extra`` 不给就读环境变量 TEACHER_EXTRA_JSON。每次请求
    （包括重试的）都先过节流：等待之后紧接着再发，也一样要隔够。
    """
    if retries < 1:
        raise ValueError("retries must be positive")
    extra = extra_json() if extra is None else extra
    gate = THROTTLE if throttle is None else throttle
    connection_failures = 0            # 连不上、超时的次数：有次数上限
    limited_failures, limited_since = 0, None      # 限流、服务端错误的次数，和第一次出现它的时刻：有总时长上限
    while True:
        try:
            with gate.request():
                body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
                merged = {**body, **extra}         # 同名以 TEACHER_EXTRA_JSON 里的为准
                # OpenAI 的 SDK 只认它签名里的那几个参数：接口特有的参数（关掉推理之类）必须放进 extra_body，
                # SDK 会把它合并进请求体。直接当关键字参数传进去，SDK 报 unexpected keyword argument，请求发不出去
                kwargs = {name: value for name, value in merged.items() if name in body}
                outside = {name: value for name, value in merged.items() if name not in body}
                if outside:
                    kwargs["extra_body"] = outside
                response = client.chat.completions.create(**kwargs)
        except Exception as error:
            if not transient(error):
                raise
            if limited(error):
                status = _status_code(error)
                what = "限速" if status == 429 else "出错"
                if limited_since is None:            # 总时长从第一次失败算起
                    limited_since = time.monotonic()
                limited_failures += 1
                wait = retry_after(error)
                if wait is None:                     # 指数最多算到 2 的 30 次方：再往上已经超过上限了
                    wait = min(first_wait * 2 ** min(limited_failures - 1, 30), max_wait)
                left = retry_seconds - (time.monotonic() - limited_since)
                if left <= 0:
                    print(f"教师接口{what}（{status}）：从第一次失败起已经等了 {int(retry_seconds // 60)} 分钟，"
                          f"重试了 {limited_failures} 次，不再重试", file=sys.stderr, flush=True)
                    raise
                wait = min(wait, left)               # 最后一次等待不超过总时长上限
                print(f"教师接口{what}（{status}），等 {_seconds(wait)} 秒后重试（第 {limited_failures} 次）",
                      file=sys.stderr, flush=True)
            else:
                connection_failures += 1
                if connection_failures >= retries:
                    raise
                wait = BACKOFF_SECONDS[min(connection_failures - 1, len(BACKOFF_SECONDS) - 1)]
            time.sleep(wait)
            continue
        choice = response.choices[0] if response.choices else None
        usage = response.usage
        return {
            "text": (choice.message.content or "") if choice else "",
            "finish_reason": choice.finish_reason if choice else None,
            "usage": {
                "input_tokens": usage.prompt_tokens if usage else None,
                "output_tokens": usage.completion_tokens if usage else None,
            },
            "other_text": other_text(choice.message) if choice else {},
        }


def teacher_model(default: str = DEFAULT_MODEL) -> str:
    """教师模型名：环境变量 TEACHER_MODEL，没有就用 ``default``。"""
    return os.environ.get("TEACHER_MODEL") or default


def make_client(api_key_env: str = KEY_ENV, base_url: str | None = None, timeout: float = 120.0):
    """建教师的客户端。密钥取环境变量 ``api_key_env``，没有就抛 RuntimeError（提示先载入 .env）。

    地址依次取：参数、环境变量 TEACHER_BASE_URL、内置的占位地址。
    """
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"环境变量 {api_key_env} 没有设置：先载入密钥（set -a; . ./.env; set +a）再运行")
    from openai import OpenAI           # 延迟导入：本机和 CI 可能没装 openai
    return OpenAI(api_key=api_key, base_url=base_url or os.environ.get("TEACHER_BASE_URL") or DEFAULT_BASE_URL,
                  timeout=timeout, max_retries=0)
