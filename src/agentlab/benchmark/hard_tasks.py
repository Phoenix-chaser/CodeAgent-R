# -*- coding: utf-8 -*-
"""L3/L4 级别任务构建器：跨文件 bug、状态一致性、多步推理。

设计原则：
- 每个任务需要理解多个文件之间的交互才能修复
- 基线（未训练 7B）预期通过率 < 50%
- 公开测试只给微弱提示（不可能仅靠公开测试通过推断出修复方案）
- 私有测试覆盖边界条件与回归
"""
from __future__ import annotations

HARD_SPECS = [
    # ── L3: 跨文件状态一致性 ──
    dict(
        task_id="hard_cache_invalidation_001", source_group="hard_cache",
        split="train", bug_family="state_consistency", difficulty="L3",
        instruction=(
            "DataCache 类缓存了 compute() 的结果。当 underlying_data 被外部修改后，"
            "缓存不会失效，导致 stale data 被返回。修复：实现基于版本的缓存失效机制。"
        ),
        module="cache.py",
        buggy="""from data_store import DataStore

class DataCache:
    def __init__(self, store: DataStore):
        self._store = store
        self._cache = {}
        self._cached_keys = set()

    def get(self, key):
        if key not in self._cache:
            self._cache[key] = self._store.compute(key)
            self._cached_keys.add(key)
        return self._cache[key]

    def invalidate(self, key):
        self._cache.pop(key, None)
        self._cached_keys.discard(key)

    def invalidate_all(self):
        self._cache.clear()
        self._cached_keys.clear()
""",
        fixed="""from data_store import DataStore

class DataCache:
    def __init__(self, store: DataStore):
        self._store = store
        self._cache = {}
        self._cached_keys = set()
        self._last_store_version = store.version

    def _check_version(self):
        if self._store.version != self._last_store_version:
            self._cache.clear()
            self._cached_keys.clear()
            self._last_store_version = self._store.version

    def get(self, key):
        self._check_version()
        if key not in self._cache:
            self._cache[key] = self._store.compute(key)
            self._cached_keys.add(key)
        return self._cache[key]

    def invalidate(self, key):
        self._cache.pop(key, None)
        self._cached_keys.discard(key)

    def invalidate_all(self):
        self._cache.clear()
        self._cached_keys.clear()
        self._last_store_version = self._store.version
""",
        extra_files={
            "data_store.py": """class DataStore:
    def __init__(self):
        self._data = {}
        self.version = 0

    def set(self, key, value):
        self._data[key] = value
        self.version += 1

    def compute(self, key):
        return self._data.get(key, 0) * 2
""",
        },
        public_test="""from cache import DataCache
from data_store import DataStore

def test_basic_cache():
    store = DataStore()
    store.set("a", 5)
    cache = DataCache(store)
    assert cache.get("a") == 10
""",
        private_tests={
            "test_invalidation.py": """from cache import DataCache
from data_store import DataStore

def test_stale_after_update():
    store = DataStore()
    store.set("a", 5)
    cache = DataCache(store)
    assert cache.get("a") == 10
    store.set("a", 7)  # underlying data changed
    assert cache.get("a") == 14  # must NOT return stale 10

def test_multiple_updates():
    store = DataStore()
    store.set("x", 1)
    cache = DataCache(store)
    assert cache.get("x") == 2
    store.set("x", 2)
    assert cache.get("x") == 4
    store.set("x", 3)
    assert cache.get("x") == 6

def test_partial_invalidation():
    store = DataStore()
    store.set("a", 1)
    store.set("b", 2)
    cache = DataCache(store)
    assert cache.get("a") == 2
    assert cache.get("b") == 4
    store.set("a", 10)  # only "a" changed
    assert cache.get("a") == 20
    assert cache.get("b") == 4  # "b" still cached correctly
""",
        },
    ),

    # ── L3: 异常传播与资源清理 ──
    dict(
        task_id="hard_resource_cleanup_001", source_group="hard_resource",
        split="train", bug_family="exception_safety", difficulty="L3",
        instruction=(
            "FileManager 管理文件句柄。当 process() 抛出异常时，文件不会被关闭，"
            "导致资源泄漏。修复：使用上下文管理器或 try/finally 确保文件始终被关闭。"
            "同时修复：读取后没有 seek(0)，导致第二次读取返回空内容。"
        ),
        module="file_manager.py",
        buggy="""class FileManager:
    def __init__(self, path):
        self.path = path
        self._file = None

    def open(self):
        self._file = open(self.path, 'w+')
        return self

    def close(self):
        if self._file:
            self._file.close()
            self._file = None

    def write(self, data):
        self._file.write(data)

    def read_all(self):
        return self._file.read()

    def process(self, data, transform):
        self.open()
        self.write(data)
        result = transform(self.read_all())
        self.close()
        return result
""",
        fixed="""class FileManager:
    def __init__(self, path):
        self.path = path
        self._file = None

    def open(self):
        self._file = open(self.path, 'w+')
        return self

    def close(self):
        if self._file:
            self._file.close()
            self._file = None

    def write(self, data):
        self._file.write(data)

    def read_all(self):
        self._file.seek(0)
        return self._file.read()

    def process(self, data, transform):
        self.open()
        try:
            self.write(data)
            result = transform(self.read_all())
        finally:
            self.close()
        return result
""",
        extra_files={},
        public_test="""from file_manager import FileManager

def test_basic_process(tmp_path):
    fm = FileManager(str(tmp_path / "test.txt"))
    result = fm.process("hello", lambda s: s.upper())
    assert result == "HELLO"
""",
        private_tests={
            "test_cleanup.py": """import pytest
from file_manager import FileManager

def test_file_closed_after_exception(tmp_path):
    fm = FileManager(str(tmp_path / "err.txt"))
    with pytest.raises(ValueError):
        fm.process("data", lambda s: (_ for _ in ()).throw(ValueError("boom")))
    assert fm._file is None  # must be closed even after exception

def test_read_after_write(tmp_path):
    fm = FileManager(str(tmp_path / "rw.txt"))
    fm.open()
    fm.write("line1\\nline2")
    assert fm.read_all() == "line1\\nline2"  # seek(0) before read
    assert fm.read_all() == "line1\\nline2"  # second read also works
    fm.close()

def test_multiple_process_calls(tmp_path):
    fm = FileManager(str(tmp_path / "multi.txt"))
    assert fm.process("abc", lambda s: len(s)) == 3
    assert fm.process("defgh", lambda s: len(s)) == 5
    assert fm._file is None  # closed after each process call
""",
        },
    ),

    # ── L3: 多文件接口契约 ──
    dict(
        task_id="hard_api_contract_001", source_group="hard_api",
        split="train", bug_family="interface_mismatch", difficulty="L3",
        instruction=(
            "ApiClient 调用 retry_wrapper 来处理网络重试。但 retry_wrapper 返回的格式"
            "与 ApiClient 期望的不一致：retry_wrapper 返回 (success, data) 元组，"
            "而 ApiClient 直接把返回值当 data 用。retry_wrapper 已实现指数退避。"
            "只修复 api_client.py：成功时返回 data，重试耗尽时抛 ConnectionError，"
            "每次 fetch_data 仍只增加一次 _call_count。"
        ),
        module="api_client.py",
        buggy="""from retry_wrapper import retry

class ApiClient:
    def __init__(self):
        self._call_count = 0

    def fetch_data(self, url):
        self._call_count += 1
        return retry(lambda: self._do_fetch(url))

    def _do_fetch(self, url):
        if "bad" in url:
            raise ConnectionError(f"Failed: {url}")
        return {"url": url, "status": 200}
""",
        fixed="""from retry_wrapper import retry

class ApiClient:
    def __init__(self):
        self._call_count = 0

    def fetch_data(self, url):
        self._call_count += 1
        success, data = retry(lambda: self._do_fetch(url))
        if not success:
            raise ConnectionError(f"Retry exhausted for {url}")
        return data

    def _do_fetch(self, url):
        if "bad" in url:
            raise ConnectionError(f"Failed: {url}")
        return {"url": url, "status": 200}
""",
        extra_files={
            "retry_wrapper.py": """import time

def retry(fn, max_retries=3, base_delay=0.01):
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            result = fn()
            return True, result
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(base_delay * (2 ** attempt))
    return False, str(last_error)
""",
        },
        public_test="""from api_client import ApiClient

def test_success():
    client = ApiClient()
    result = client.fetch_data("https://good.example.com")
    assert result["status"] == 200
""",
        private_tests={
            "test_retry.py": """import pytest
from api_client import ApiClient

def test_failure_raises():
    client = ApiClient()
    with pytest.raises(ConnectionError):
        client.fetch_data("https://bad.example.com")

def test_return_format():
    client = ApiClient()
    result = client.fetch_data("https://api.example.com/data")
    assert isinstance(result, dict)
    assert "url" in result
    assert "status" in result

def test_call_count():
    client = ApiClient()
    client.fetch_data("https://a.example.com")
    client.fetch_data("https://b.example.com")
    assert client._call_count == 2
""",
        },
    ),

    # ── L3: 状态机转换 ──
    dict(
        task_id="hard_state_machine_001", source_group="hard_state",
        split="train", bug_family="state_machine", difficulty="L3",
        instruction=(
            "OrderStateMachine 管理订单状态。CANCELLED 状态的订单不应允许再次转换，"
            "但当前转换矩阵允许取消后回到 PENDING。同时 SHIPPED 只能转到 DELIVERED，"
            "不能取消。保留 PENDING→CONFIRMED→SHIPPED→DELIVERED 的流程，"
            "PENDING 和 CONFIRMED 可以取消，CANCELLED 和 DELIVERED 都是终态。"
        ),
        module="order_state.py",
        buggy="""STATES = ["PENDING", "CONFIRMED", "SHIPPED", "CANCELLED", "DELIVERED"]

VALID_TRANSITIONS = {
    "PENDING": ["CONFIRMED", "CANCELLED"],
    "CONFIRMED": ["SHIPPED", "CANCELLED"],
    "SHIPPED": ["DELIVERED", "CANCELLED"],
    "DELIVERED": [],
    "CANCELLED": ["PENDING"],  # BUG: cancelled orders shouldn't transition
}

class OrderStateMachine:
    def __init__(self):
        self.state = "PENDING"
        self.history = ["PENDING"]

    def transition(self, new_state):
        if new_state not in STATES:
            raise ValueError(f"Unknown state: {new_state}")
        allowed = VALID_TRANSITIONS.get(self.state, [])
        if new_state not in allowed:
            raise ValueError(f"Cannot go from {self.state} to {new_state}")
        self.state = new_state
        self.history.append(new_state)
        return True

    def can_transition(self, new_state):
        return new_state in VALID_TRANSITIONS.get(self.state, [])
""",
        fixed="""STATES = ["PENDING", "CONFIRMED", "SHIPPED", "CANCELLED", "DELIVERED"]

VALID_TRANSITIONS = {
    "PENDING": ["CONFIRMED", "CANCELLED"],
    "CONFIRMED": ["SHIPPED", "CANCELLED"],
    "SHIPPED": ["DELIVERED"],  # shipped orders cannot be cancelled
    "DELIVERED": [],
    "CANCELLED": [],  # terminal state: no transitions allowed
}

class OrderStateMachine:
    def __init__(self):
        self.state = "PENDING"
        self.history = ["PENDING"]

    def transition(self, new_state):
        if new_state not in STATES:
            raise ValueError(f"Unknown state: {new_state}")
        allowed = VALID_TRANSITIONS.get(self.state, [])
        if new_state not in allowed:
            raise ValueError(f"Cannot go from {self.state} to {new_state}")
        self.state = new_state
        self.history.append(new_state)
        return True

    def can_transition(self, new_state):
        return new_state in VALID_TRANSITIONS.get(self.state, [])
""",
        extra_files={},
        public_test="""from order_state import OrderStateMachine

def test_valid_transition():
    sm = OrderStateMachine()
    sm.transition("CONFIRMED")
    assert sm.state == "CONFIRMED"
""",
        private_tests={
            "test_terminal.py": """import pytest
from order_state import OrderStateMachine

def test_cancelled_is_terminal():
    sm = OrderStateMachine()
    sm.transition("CANCELLED")
    with pytest.raises(ValueError):
        sm.transition("PENDING")
    with pytest.raises(ValueError):
        sm.transition("CONFIRMED")

def test_delivered_is_terminal():
    sm = OrderStateMachine()
    sm.transition("CONFIRMED")
    sm.transition("SHIPPED")
    sm.transition("DELIVERED")
    with pytest.raises(ValueError):
        sm.transition("CANCELLED")

def test_shipped_cannot_cancel():
    sm = OrderStateMachine()
    sm.transition("CONFIRMED")
    sm.transition("SHIPPED")
    with pytest.raises(ValueError):
        sm.transition("CANCELLED")

def test_full_lifecycle():
    sm = OrderStateMachine()
    sm.transition("CONFIRMED")
    sm.transition("SHIPPED")
    sm.transition("DELIVERED")
    assert sm.history == ["PENDING", "CONFIRMED", "SHIPPED", "DELIVERED"]

def test_can_transition():
    sm = OrderStateMachine()
    assert sm.can_transition("CONFIRMED")
    assert sm.can_transition("CANCELLED")
    assert not sm.can_transition("SHIPPED")  # PENDING -> SHIPPED not allowed
""",
        },
    ),

    # ── L4: 多处耦合缺陷 ──
    dict(
        task_id="hard_pipeline_race_001", source_group="hard_pipeline",
        split="train", bug_family="race_condition", difficulty="L4",
        instruction=(
            "Pipeline 类按序执行多个 stage。存在两个 bug：(1) stage 失败后 pipeline "
            "不会停止，后续 stage 继续执行并可能产生错误结果；(2) stage 的执行结果"
            "没有传递给下一个 stage（每个 stage 收到的是原始输入而不是上一个 stage "
            "的输出）。修复使 pipeline 正确串行传递数据并在失败时停止。"
        ),
        module="pipeline.py",
        buggy="""class Pipeline:
    def __init__(self):
        self.stages = []
        self.results = []
        self.errors = []

    def add_stage(self, name, fn):
        self.stages.append((name, fn))
        return self

    def execute(self, input_data):
        self.results = []
        self.errors = []
        for name, fn in self.stages:
            try:
                result = fn(input_data)  # BUG: should be fn(current_data)
                self.results.append((name, result))
            except Exception as e:
                self.errors.append((name, str(e)))
                # BUG: should break here
        return self.results

    @property
    def last_result(self):
        return self.results[-1][1] if self.results else None

    @property
    def failed(self):
        return len(self.errors) > 0
""",
        fixed="""class Pipeline:
    def __init__(self):
        self.stages = []
        self.results = []
        self.errors = []

    def add_stage(self, name, fn):
        self.stages.append((name, fn))
        return self

    def execute(self, input_data):
        self.results = []
        self.errors = []
        current_data = input_data
        for name, fn in self.stages:
            try:
                current_data = fn(current_data)
                self.results.append((name, current_data))
            except Exception as e:
                self.errors.append((name, str(e)))
                break  # stop on first failure
        return self.results

    @property
    def last_result(self):
        return self.results[-1][1] if self.results else None

    @property
    def failed(self):
        return len(self.errors) > 0
""",
        extra_files={},
        public_test="""from pipeline import Pipeline

def test_simple_pipeline():
    p = Pipeline()
    p.add_stage("double", lambda x: x * 2)
    p.add_stage("add1", lambda x: x + 1)
    results = p.execute(5)
    assert len(results) == 2
""",
        private_tests={
            "test_chaining.py": """from pipeline import Pipeline

def test_data_chaining():
    p = Pipeline()
    p.add_stage("step1", lambda x: x + 10)
    p.add_stage("step2", lambda x: x * 2)
    p.add_stage("step3", lambda x: x - 5)
    p.execute(1)
    # (1+10)*2-5 = 17
    assert p.last_result == 17

def test_failure_stops():
    p = Pipeline()
    p.add_stage("ok1", lambda x: x + 1)
    p.add_stage("fail", lambda x: (_ for _ in ()).throw(ValueError("boom")))
    p.add_stage("ok2", lambda x: x * 100)
    p.execute(0)
    assert p.failed
    assert len(p.results) == 1  # only "ok1" succeeded
    assert len(p.errors) == 1   # "fail" recorded
    # "ok2" should NOT have executed
    assert p.last_result == 1  # result from "ok1", not "ok2"

def test_string_pipeline():
    p = Pipeline()
    p.add_stage("upper", str.upper)
    p.add_stage("reverse", lambda s: s[::-1])
    p.add_stage("strip_vowels", lambda s: ''.join(c for c in s if c not in 'aeiouAEIOU'))
    p.execute("hello world")
    # "hello world" -> "HELLO WORLD" -> "DLROW OLLEH" -> "DLRW LLH"
    assert p.last_result == "DLRW LLH"
""",
        },
    ),

    # ── L4: 复杂依赖追踪 ──
    dict(
        task_id="hard_dependency_tracker_001", source_group="hard_dep",
        split="train", bug_family="dependency_tracking", difficulty="L4",
        instruction=(
            "DependencyTracker 追踪任务间的依赖关系。存在两个 bug：(1) 循环依赖检测"
            "只检查了直接依赖，没有检查间接依赖（A→B→C→A 不被检测）；(2) resolve "
            "方法没有处理依赖不存在的情况（依赖了未注册的任务时抛出 KeyError 而不是"
            "抛出 ValueError）。resolve 的起始任务不存在时消息须包含 'Unknown task'，"
            "依赖不存在时须包含 'Unknown dependency'；无环时保持依赖在前且每任务只输出一次。"
        ),
        module="dependency.py",
        buggy="""class DependencyTracker:
    def __init__(self):
        self._tasks = {}

    def register(self, name, deps=None):
        self._tasks[name] = list(deps or [])

    def has_circular(self, name):
        if name not in self._tasks:
            return False
        return name in self._tasks[name]  # only checks direct

    def resolve(self, name):
        order = []
        visited = set()
        def visit(task):
            if task in visited:
                return
            visited.add(task)
            for dep in self._tasks[task]:  # KeyError if dep not registered
                visit(dep)
            order.append(task)
        visit(name)
        return order
""",
        fixed="""class DependencyTracker:
    def __init__(self):
        self._tasks = {}

    def register(self, name, deps=None):
        self._tasks[name] = list(deps or [])

    def has_circular(self, name, _visited=None, _stack=None):
        if name not in self._tasks:
            return False
        _visited = _visited or set()
        _stack = _stack or set()
        if name in _stack:
            return True
        if name in _visited:
            return False
        _stack.add(name)
        for dep in self._tasks[name]:
            if self.has_circular(dep, _visited, _stack):
                return True
        _stack.discard(name)
        _visited.add(name)
        return False

    def resolve(self, name):
        if name not in self._tasks:
            raise ValueError(f"Unknown task: {name}")
        order = []
        visited = set()
        def visit(task):
            if task in visited:
                return
            if task not in self._tasks:
                raise ValueError(f"Unknown dependency: {task} (required by a task in the chain)")
            visited.add(task)
            for dep in self._tasks[task]:
                visit(dep)
            order.append(task)
        visit(name)
        return order
""",
        extra_files={},
        public_test="""from dependency import DependencyTracker

def test_simple_deps():
    dt = DependencyTracker()
    dt.register("c", deps=["b"])
    dt.register("b", deps=["a"])
    dt.register("a")
    order = dt.resolve("c")
    assert order == ["a", "b", "c"]
""",
        private_tests={
            "test_circular.py": """import pytest
from dependency import DependencyTracker

def test_indirect_circular():
    dt = DependencyTracker()
    dt.register("a", deps=["c"])
    dt.register("b", deps=["a"])
    dt.register("c", deps=["b"])
    assert dt.has_circular("a")  # a -> c -> b -> a

def test_no_circular():
    dt = DependencyTracker()
    dt.register("a")
    dt.register("b", deps=["a"])
    dt.register("c", deps=["b"])
    assert not dt.has_circular("c")

def test_unknown_task():
    dt = DependencyTracker()
    dt.register("a")
    with pytest.raises(ValueError, match="Unknown task"):
        dt.resolve("nonexistent")

def test_unknown_dependency():
    dt = DependencyTracker()
    dt.register("a", deps=["ghost"])
    with pytest.raises(ValueError, match="Unknown dependency"):
        dt.resolve("a")

def test_diamond_deps():
    dt = DependencyTracker()
    dt.register("d", deps=["b", "c"])
    dt.register("b", deps=["a"])
    dt.register("c", deps=["a"])
    dt.register("a")
    order = dt.resolve("d")
    assert order[0] == "a"
    assert order[-1] == "d"
    assert set(order) == {"a", "b", "c", "d"}
""",
        },
    ),

    # ── L3: 并发安全的计数器 ──
    dict(
        task_id="hard_thread_counter_001", source_group="hard_thread",
        split="train", bug_family="thread_safety", difficulty="L3",
        instruction=(
            "SafeCounter 是一个多线程安全的计数器。bug：increment 和 get_value "
            "方法没有加锁，在多线程环境下会出现竞态条件导致计数不准确。"
            "修复：使用 threading.Lock 保护共享状态。同时 fix reset 方法"
            "避免嵌套获取非重入锁。reset 和 set_value 都应原子地修改值并返回旧值。"
        ),
        module="safe_counter.py",
        buggy="""import threading

class SafeCounter:
    def __init__(self, initial=0):
        self._value = initial
        self._lock = threading.Lock()

    def increment(self, amount=1):
        self._value += amount  # not thread-safe

    def decrement(self, amount=1):
        self._value -= amount  # not thread-safe

    def get_value(self):
        return self._value  # not thread-safe

    def reset(self):
        with self._lock:
            self.set_value(0)  # deadlock: set_value also acquires lock

    def set_value(self, new_value):
        with self._lock:
            current = self.get_value()  # deadlock if called from locked context
            self._value = new_value
            return current
""",
        fixed="""import threading

class SafeCounter:
    def __init__(self, initial=0):
        self._value = initial
        self._lock = threading.Lock()

    def increment(self, amount=1):
        with self._lock:
            self._value += amount

    def decrement(self, amount=1):
        with self._lock:
            self._value -= amount

    def get_value(self):
        with self._lock:
            return self._value

    def reset(self):
        with self._lock:
            old = self._value
            self._value = 0
            return old

    def set_value(self, new_value):
        with self._lock:
            old = self._value
            self._value = new_value
            return old
""",
        extra_files={},
        public_test="""from safe_counter import SafeCounter

def test_basic():
    c = SafeCounter()
    c.increment()
    c.increment()
    assert c.get_value() == 2
""",
        private_tests={
            "test_thread_safety.py": """import threading
from safe_counter import SafeCounter

def test_concurrent_increment():
    c = SafeCounter()
    threads = []
    for _ in range(10):
        t = threading.Thread(target=lambda: [c.increment() for _ in range(100)])
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert c.get_value() == 1000

def test_reset_no_deadlock():
    c = SafeCounter(50)
    old = c.reset()
    assert old == 50
    assert c.get_value() == 0

def test_set_value_no_deadlock():
    c = SafeCounter(10)
    old = c.set_value(99)
    assert old == 10
    assert c.get_value() == 99

def test_concurrent_mixed_ops():
    c = SafeCounter(500)
    def inc():
        for _ in range(100):
            c.increment()
    def dec():
        for _ in range(100):
            c.decrement()
    threads = [threading.Thread(target=inc), threading.Thread(target=dec)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert c.get_value() == 500
""",
        },
    ),

    # ── L3: 配置优先级合并 ──
    dict(
        task_id="hard_config_merge_001", source_group="hard_config",
        split="train", bug_family="config_precedence", difficulty="L3",
        instruction=(
            "ConfigManager 按优先级合并多个配置源（低→高：defaults < file < env）。"
            "bug：嵌套字典没有被深度合并——如果 file 和 defaults 都有 'database' "
            "键（值是字典），file 的 'database' 会完全覆盖 defaults 的，丢失了 "
            "defaults 中 file 没有的子键。修复：实现递归深度合并。"
        ),
        module="config_manager.py",
        buggy="""import os

class ConfigManager:
    def __init__(self, defaults=None):
        self._sources = {"defaults": defaults or {}}

    def add_file_config(self, config):
        self._sources["file"] = config

    def add_env_config(self, env_prefix="APP_"):
        env_config = {}
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config_key = key[len(env_prefix):].lower()
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                env_config[config_key] = value
        self._sources["env"] = env_config

    def get(self, key, default=None):
        merged = self._merge()
        return merged.get(key, default)

    def _merge(self):
        result = {}
        for source_name in ["defaults", "file", "env"]:
            source = self._sources.get(source_name, {})
            result.update(source)  # shallow merge: nested dicts get replaced
        return result

    def get_all(self):
        return self._merge()
""",
        fixed="""import os
import copy

class ConfigManager:
    def __init__(self, defaults=None):
        self._sources = {"defaults": defaults or {}}

    def add_file_config(self, config):
        self._sources["file"] = config

    def add_env_config(self, env_prefix="APP_"):
        env_config = {}
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config_key = key[len(env_prefix):].lower()
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                env_config[config_key] = value
        self._sources["env"] = env_config

    def get(self, key, default=None):
        merged = self._merge()
        return merged.get(key, default)

    def _deep_merge(self, base, override):
        result = copy.deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    def _merge(self):
        result = {}
        for source_name in ["defaults", "file", "env"]:
            source = self._sources.get(source_name, {})
            result = self._deep_merge(result, source)
        return result

    def get_all(self):
        return self._merge()
""",
        extra_files={},
        public_test="""from config_manager import ConfigManager

def test_simple_override():
    cm = ConfigManager({"host": "localhost", "port": 8080})
    cm.add_file_config({"port": 9090})
    assert cm.get("port") == 9090
    assert cm.get("host") == "localhost"
""",
        private_tests={
            "test_deep_merge.py": """from config_manager import ConfigManager

def test_nested_merge():
    cm = ConfigManager({
        "database": {"host": "localhost", "port": 5432, "ssl": True}
    })
    cm.add_file_config({
        "database": {"host": "remote.example.com"}  # only override host
    })
    db = cm.get("database")
    assert db["host"] == "remote.example.com"  # overridden
    assert db["port"] == 5432  # preserved from defaults
    assert db["ssl"] is True   # preserved from defaults

def test_deep_nested():
    cm = ConfigManager({
        "a": {"b": {"c": 1, "d": 2}}
    })
    cm.add_file_config({
        "a": {"b": {"c": 99}}
    })
    result = cm.get("a")
    assert result["b"]["c"] == 99
    assert result["b"]["d"] == 2  # preserved

def test_env_overrides_all():
    import os
    os.environ["TEST_CFG_PORT"] = "7777"
    cm = ConfigManager({"port": 1000})
    cm.add_file_config({"port": 2000})
    cm.add_env_config("TEST_CFG_")
    assert cm.get("port") == 7777

def test_add_new_nested_key():
    cm = ConfigManager({"existing": "yes"})
    cm.add_file_config({"new_key": {"sub": "value"}})
    assert cm.get("existing") == "yes"
    assert cm.get("new_key") == {"sub": "value"}
""",
        },
    ),
]
