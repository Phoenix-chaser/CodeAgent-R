"""Lock-ordering tasks for the hard batch (h3).

Every program here is used from several threads at once.  The defects are the
ones a per-method lock does not fix: locks taken in call order, a compound
operation that validates after it mutates, a wait that ignores its timeout, a
method that takes no lock at all.  Every wait in the private tests is bounded,
and every thread that could block under the buggy program is a daemon thread.
"""
from __future__ import annotations

from agentlab.benchmark.hard_tasks_v3_common import _task

SPECS = [
    _task(
        "transfer_lock_bank", "transfer_lock",
        "Bank 管理多个账户，会被多个线程同时使用。open(name, amount) 开户，balance(name) 查余额。要求：\n"
        "1）transfer(src, dst, amount)：账户不存在抛 KeyError；amount ≤ 0 或余额不足抛 ValueError；"
        "失败时两个账户的余额都不变，扣款与入账整体原子。\n"
        "2）deposit / withdraw 与 transfer 互斥；withdraw 余额不足抛 ValueError 且余额不变；"
        "账户不存在同样抛 KeyError。\n"
        "3）方向相反的两笔转账同时执行时不能死锁。\n"
        "4）hold(name, ready, release)：持有该账户的锁，拿到锁后设置 ready，直到 release 被设置才释放；"
        "这期间不涉及该账户的转账仍能完成。\n"
        "5）transfer / deposit / withdraw / total / hold 都接受可选参数 timeout（秒，默认 None 表示一直等）："
        "等锁超时抛 TimeoutError，且不改变任何余额。\n"
        "6）total() 返回所有余额之和，必须等于某一时刻的真实总额。",
        '''
        import threading
        from contextlib import ExitStack


        class Bank:
            def __init__(self):
                self._accounts = {}
                self._locks = {}

            def open(self, name, amount):
                self._accounts[name] = amount
                self._locks[name] = threading.Lock()

            def _guard(self, names, timeout):
                stack = ExitStack()
                for name in names:
                    lock = self._locks[name]
                    lock.acquire(True, -1 if timeout is None else timeout)
                    stack.callback(lock.release)
                return stack

            def transfer(self, src, dst, amount, timeout=None):
                if src not in self._accounts or dst not in self._accounts:
                    raise KeyError(src if src not in self._accounts else dst)
                if amount <= 0:
                    raise ValueError('amount must be positive')
                with self._guard([src, dst], timeout):
                    self._accounts[src] -= amount
                    if self._accounts[src] < 0:
                        raise ValueError('insufficient funds')
                    self._accounts[dst] += amount
                    return True

            def deposit(self, name, amount, timeout=None):
                if amount <= 0:
                    raise ValueError('amount must be positive')
                with self._guard([name], timeout):
                    self._accounts[name] += amount
                    return self._accounts[name]

            def withdraw(self, name, amount, timeout=None):
                if amount <= 0:
                    raise ValueError('amount must be positive')
                if self._accounts[name] < amount:
                    raise ValueError('insufficient funds')
                self._accounts[name] -= amount
                return self._accounts[name]

            def balance(self, name):
                return self._accounts[name]

            def total(self, timeout=None):
                return sum(self._accounts.values())

            def hold(self, name, ready, release, timeout=None):
                with self._guard([name], timeout):
                    ready.set()
                    release.wait()
        ''',
        '''
        import threading
        from contextlib import ExitStack


        class Bank:
            def __init__(self):
                self._accounts = {}
                self._locks = {}

            def open(self, name, amount):
                self._accounts[name] = amount
                self._locks[name] = threading.Lock()

            def _guard(self, names, timeout):
                stack = ExitStack()
                try:
                    for name in sorted(set(names)):
                        lock = self._locks[name]
                        if not lock.acquire(True, -1 if timeout is None else timeout):
                            raise TimeoutError(name)
                        stack.callback(lock.release)
                except BaseException:
                    stack.close()
                    raise
                return stack

            def transfer(self, src, dst, amount, timeout=None):
                if src not in self._accounts or dst not in self._accounts:
                    raise KeyError(src if src not in self._accounts else dst)
                if amount <= 0:
                    raise ValueError('amount must be positive')
                with self._guard([src, dst], timeout):
                    if self._accounts[src] < amount:
                        raise ValueError('insufficient funds')
                    self._accounts[src] -= amount
                    self._accounts[dst] += amount
                    return True

            def deposit(self, name, amount, timeout=None):
                if amount <= 0:
                    raise ValueError('amount must be positive')
                with self._guard([name], timeout):
                    self._accounts[name] += amount
                    return self._accounts[name]

            def withdraw(self, name, amount, timeout=None):
                if amount <= 0:
                    raise ValueError('amount must be positive')
                with self._guard([name], timeout):
                    if self._accounts[name] < amount:
                        raise ValueError('insufficient funds')
                    self._accounts[name] -= amount
                    return self._accounts[name]

            def balance(self, name):
                return self._accounts[name]

            def total(self, timeout=None):
                with self._guard(list(self._accounts), timeout):
                    return sum(self._accounts.values())

            def hold(self, name, ready, release, timeout=None):
                with self._guard([name], timeout):
                    ready.set()
                    release.wait()
        ''',
        '''
        def test_single_threaded_operations():
            bank = Bank()
            bank.open('a', 100)
            bank.open('b', 50)
            assert bank.transfer('a', 'b', 30)
            assert bank.balance('a') == 70 and bank.balance('b') == 80
            assert bank.deposit('a', 5) == 75
            assert bank.withdraw('b', 80) == 0
            assert bank.total() == 75
        ''',
        {
            "state": '''
                def test_failures_leave_balances_unchanged():
                    bank = Bank()
                    for name in 'abc':
                        bank.open(name, 100)
                    with pytest.raises(ValueError):
                        bank.transfer('a', 'b', 500)
                    assert bank.balance('a') == 100 and bank.balance('b') == 100
                    with pytest.raises(ValueError):
                        bank.transfer('a', 'b', 0)
                    with pytest.raises(ValueError):
                        bank.withdraw('a', 500)
                    with pytest.raises(ValueError):
                        bank.deposit('a', -1)
                    assert bank.balance('a') == 100
                    with pytest.raises(KeyError):
                        bank.transfer('a', 'zz', 1)
                    with pytest.raises(KeyError):
                        bank.deposit('zz', 1)
                    assert bank.transfer('a', 'b', 1, timeout=1)
                    assert bank.balance('a') == 99
            ''',
            "threads": '''
                import threading


                def _bank():
                    bank = Bank()
                    for name in 'abc':
                        bank.open(name, 100)
                    return bank


                def _start(call):
                    thread = threading.Thread(target=call, daemon=True)
                    thread.start()
                    return thread


                def test_opposite_transfers_do_not_deadlock():
                    bank = _bank()
                    held, release = threading.Event(), threading.Event()
                    blocker = _start(lambda: bank.hold('a', held, release))
                    assert held.wait(1)
                    first = _start(lambda: bank.transfer('a', 'b', 30))
                    second = _start(lambda: bank.transfer('b', 'a', 20))
                    first.join(0.05)
                    second.join(0.05)
                    assert first.is_alive() and second.is_alive()
                    gate = threading.Event()
                    gate.set()
                    # a transfer waiting for 'a' must not be holding 'b'
                    bank.hold('b', threading.Event(), gate, timeout=0.05)
                    release.set()
                    for thread in (blocker, first, second):
                        thread.join(1)
                    assert all(not thread.is_alive() for thread in (blocker, first, second))
                    assert bank.total(timeout=1) == 300
                    assert bank.balance('a') == 90 and bank.balance('b') == 110


                def test_held_account_blocks_only_its_own_operations():
                    bank = _bank()
                    ready, release = threading.Event(), threading.Event()
                    holder = threading.Thread(target=bank.hold, args=('a', ready, release), daemon=True)
                    holder.start()
                    assert ready.wait(1)
                    for call in (lambda t=None: bank.transfer('a', 'b', 1, t),
                                 lambda t=None: bank.transfer('b', 'a', 1, t),
                                 lambda t=None: bank.withdraw('a', 1, t),
                                 lambda t=None: bank.deposit('a', 1, t), bank.total):
                        with pytest.raises(TimeoutError):
                            call(0.2)
                    assert bank.transfer('b', 'c', 5, timeout=1)
                    assert bank.balance('b') == 95 and bank.balance('c') == 105
                    assert bank.balance('a') == 100
                    release.set()
                    holder.join(1)
                    assert not holder.is_alive()
                    assert bank.transfer('a', 'b', 1, timeout=1)
            ''',
        },
    ),
    _task(
        "transfer_lock_limit", "transfer_lock",
        "Ledger 管理多个账户，会被多个线程同时使用。open(name, balance, limit) 开户，limit 是允许的透支额度；"
        "balance(name) 查余额。要求：\n"
        "1）transfer(src, dst, amount) 按固定顺序校验：账户不存在抛 KeyError → amount ≤ 0 抛 ValueError → "
        "源账户已冻结抛 PermissionError → 转出后余额低于 -limit 抛 ValueError；任何校验失败时两个账户都不变。\n"
        "2）transfer_all(src, dst)：把 src 的全部余额转入 dst 并返回 src 的新余额；余额不为正数抛 ValueError，"
        "账户不存在抛 KeyError。\n"
        "3）freeze / unfreeze 与该账户上的其他操作互斥，立即生效。\n"
        "4）方向相反的两笔转账不能死锁；被占用或被冻结的账户不阻塞其他账户之间的操作。\n"
        "5）transfer / transfer_all / total / freeze / unfreeze / pin 都接受可选 timeout"
        "（默认 None 表示一直等）：等锁超时抛 TimeoutError，且不改变任何余额。\n"
        "6）total() 等于某一时刻的真实总额。\n"
        "7）pin(name, ready, gate)：持有该账户的锁，拿到锁后设置 ready，直到 gate 被设置才释放。",
        '''
        import threading
        from contextlib import contextmanager


        class Ledger:
            def __init__(self):
                self._accounts = {}
                self._locks = {}

            def open(self, name, balance, limit=0):
                self._accounts[name] = [balance, limit, False]
                self._locks[name] = threading.Lock()

            @contextmanager
            def _guard(self, names, timeout):
                taken = []
                try:
                    for name in names:
                        lock = self._locks[name]
                        if not lock.acquire(True, -1 if timeout is None else timeout):
                            raise TimeoutError(name)
                        taken.append(lock)
                    yield
                finally:
                    for lock in reversed(taken):
                        lock.release()

            def _take(self, src, dst, amount):
                source, target = self._accounts[src], self._accounts[dst]
                source[0] -= amount
                if source[2]:
                    raise PermissionError('frozen')
                if source[0] < -source[1]:
                    raise ValueError('over limit')
                target[0] += amount
                return source[0]

            def transfer(self, src, dst, amount, timeout=None):
                if amount <= 0:
                    raise ValueError('amount must be positive')
                with self._guard([src, dst], timeout):
                    return self._take(src, dst, amount)

            def transfer_all(self, src, dst, timeout=None):
                with self._guard([src, dst], timeout):
                    return self.transfer(src, dst, self.balance(src), timeout)

            def freeze(self, name, timeout=None):
                self._accounts[name][2] = True

            def unfreeze(self, name, timeout=None):
                self._accounts[name][2] = False

            def balance(self, name):
                return self._accounts[name][0]

            def total(self, timeout=None):
                with self._guard(list(self._accounts), timeout):
                    return sum(entry[0] for entry in self._accounts.values())

            def pin(self, name, ready, gate, timeout=None):
                with self._guard([name], timeout):
                    ready.set()
                    gate.wait()
        ''',
        '''
        import threading
        from contextlib import contextmanager


        class Ledger:
            def __init__(self):
                self._accounts = {}
                self._locks = {}

            def open(self, name, balance, limit=0):
                self._accounts[name] = [balance, limit, False]
                self._locks[name] = threading.Lock()

            @contextmanager
            def _guard(self, names, timeout):
                taken = []
                try:
                    for name in sorted(set(names)):
                        lock = self._locks[name]
                        if not lock.acquire(True, -1 if timeout is None else timeout):
                            raise TimeoutError(name)
                        taken.append(lock)
                    yield
                finally:
                    for lock in reversed(taken):
                        lock.release()

            def _take(self, src, dst, amount):
                source, target = self._accounts[src], self._accounts[dst]
                if source[2]:
                    raise PermissionError('frozen')
                if source[0] - amount < -source[1]:
                    raise ValueError('over limit')
                source[0] -= amount
                target[0] += amount
                return source[0]

            def transfer(self, src, dst, amount, timeout=None):
                if src not in self._accounts or dst not in self._accounts:
                    raise KeyError(src)
                if amount <= 0:
                    raise ValueError('amount must be positive')
                with self._guard([src, dst], timeout):
                    return self._take(src, dst, amount)

            def transfer_all(self, src, dst, timeout=None):
                with self._guard([src, dst], timeout):
                    amount = self._accounts[src][0]
                    if amount <= 0:
                        raise ValueError('nothing to transfer')
                    return self._take(src, dst, amount)

            def freeze(self, name, timeout=None):
                with self._guard([name], timeout):
                    self._accounts[name][2] = True

            def unfreeze(self, name, timeout=None):
                with self._guard([name], timeout):
                    self._accounts[name][2] = False

            def balance(self, name):
                return self._accounts[name][0]

            def total(self, timeout=None):
                with self._guard(list(self._accounts), timeout):
                    return sum(entry[0] for entry in self._accounts.values())

            def pin(self, name, ready, gate, timeout=None):
                with self._guard([name], timeout):
                    ready.set()
                    gate.wait()
        ''',
        '''
        def test_single_threaded_operations():
            ledger = Ledger()
            ledger.open('a', 10, 5)
            ledger.open('b', 0)
            assert ledger.transfer('a', 'b', 12) == -2
            assert ledger.balance('b') == 12
            assert ledger.total() == 10
            ledger.freeze('a')
            ledger.unfreeze('a')
            assert ledger.transfer('a', 'b', 2) == -4
        ''',
        {
            "state": '''
                def test_validation_order_and_atomic_failure():
                    ledger = Ledger()
                    ledger.open('a', 10, 5)
                    ledger.open('b', 0)
                    with pytest.raises(KeyError):
                        ledger.transfer('zz', 'a', 0)
                    with pytest.raises(ValueError):
                        ledger.transfer('a', 'b', 0)
                    with pytest.raises(ValueError):
                        ledger.transfer('a', 'b', 16)
                    assert ledger.balance('a') == 10 and ledger.balance('b') == 0
                    assert ledger.transfer('a', 'b', 15) == -5
                    ledger.freeze('a')
                    with pytest.raises(PermissionError):
                        ledger.transfer('a', 'b', 1)
                    assert ledger.balance('a') == -5 and ledger.balance('b') == 15
                    ledger.unfreeze('a')
                    assert ledger.transfer_all('b', 'a', timeout=0.3) == 0
                    assert ledger.balance('a') == 10 and ledger.balance('b') == 0
                    with pytest.raises(ValueError):
                        ledger.transfer_all('b', 'a', timeout=0.3)
                    with pytest.raises(KeyError):
                        ledger.transfer_all('b', 'zz', timeout=0.3)
                    assert ledger.balance('b') == 0
            ''',
            "threads": '''
                import threading


                def _ledger():
                    ledger = Ledger()
                    for name in 'abc':
                        ledger.open(name, 100, 50)
                    return ledger


                def _start(call):
                    thread = threading.Thread(target=call, daemon=True)
                    thread.start()
                    return thread


                def test_opposite_transfers_do_not_deadlock():
                    ledger = _ledger()
                    held, gate = threading.Event(), threading.Event()
                    blocker = _start(lambda: ledger.pin('a', held, gate))
                    assert held.wait(1)
                    first = _start(lambda: ledger.transfer('a', 'b', 30))
                    second = _start(lambda: ledger.transfer('b', 'a', 70))
                    first.join(0.05)
                    second.join(0.05)
                    assert first.is_alive() and second.is_alive()
                    free = threading.Event()
                    free.set()
                    # a transfer waiting for 'a' must not be holding 'b'
                    ledger.pin('b', threading.Event(), free, timeout=0.05)
                    gate.set()
                    for thread in (blocker, first, second):
                        thread.join(1)
                    assert all(not thread.is_alive() for thread in (blocker, first, second))
                    assert ledger.total(timeout=1) == 300
                    assert ledger.balance('a') == 140 and ledger.balance('b') == 60


                def test_pinned_account_blocks_only_its_own_operations():
                    ledger = _ledger()
                    ready, gate = threading.Event(), threading.Event()
                    blocker = _start(lambda: ledger.pin('a', ready, gate))
                    assert ready.wait(1)
                    for call in (lambda t=None: ledger.freeze('a', t),
                                 lambda t=None: ledger.transfer('a', 'b', 1, t), ledger.total):
                        with pytest.raises(TimeoutError):
                            call(0.05)
                    assert ledger.transfer('b', 'c', 5, timeout=1)
                    assert ledger.balance('b') == 95 and ledger.balance('c') == 105
                    gate.set()
                    blocker.join(1)
                    assert not blocker.is_alive()
                    ledger.freeze('a', timeout=1)
                    with pytest.raises(PermissionError):
                        ledger.transfer('a', 'b', 1, timeout=1)
                    assert ledger.balance('a') == 100
            ''',
        },
    ),
    _task(
        "single_flight_cache", "single_flight",
        "Cache 是一个多线程使用的缓存：get(key, factory) 在没有缓存时调用 factory() 生成并缓存。要求：\n"
        "1）同一个 key 并发调用时 factory 至多执行一次，所有调用者拿到同一个返回值。\n"
        "2）不同 key 之间互不阻塞。\n"
        "3）等待同一次计算超过 timeout 秒抛 TimeoutError（timeout 默认 None 表示一直等）。\n"
        "4）factory 抛出异常：结果不缓存，同一个异常对象原样抛给执行者和所有等待者，"
        "之后的调用会重新执行 factory。\n"
        "5）invalidate(key)：丢弃已缓存的结果并返回 True；key 不存在或仍在计算中返回 False。\n"
        "6）size() 返回当前保存的 key 个数（含仍在计算中的）。",
        '''
        import threading


        class Cache:
            def __init__(self):
                self._guard = threading.Lock()
                self._slots = {}

            def get(self, key, factory, timeout=None):
                with self._guard:
                    if key in self._slots:
                        value, error = self._slots[key]
                        if error is not None:
                            raise error
                        return value
                    try:
                        value = factory()
                    except Exception as failure:
                        self._slots[key] = [None, failure]
                        raise failure
                    self._slots[key] = [value, None]
                    return value

            def invalidate(self, key):
                with self._guard:
                    return self._slots.pop(key, None) is not None

            def size(self):
                with self._guard:
                    return len(self._slots)
        ''',
        '''
        import threading


        class Cache:
            def __init__(self):
                self._guard = threading.Lock()
                self._slots = {}

            def get(self, key, factory, timeout=None):
                with self._guard:
                    owner = key not in self._slots
                    if owner:
                        self._slots[key] = [threading.Event(), None, None]
                    slot = self._slots[key]
                if not owner:
                    if not slot[0].wait(timeout):
                        raise TimeoutError(key)
                    if slot[2] is not None:
                        raise slot[2]
                    return slot[1]
                try:
                    slot[1] = factory()
                    return slot[1]
                except BaseException as error:
                    slot[2] = error
                    with self._guard:
                        self._slots.pop(key, None)
                    raise
                finally:
                    slot[0].set()

            def invalidate(self, key):
                with self._guard:
                    slot = self._slots.get(key)
                    if slot is None or not slot[0].is_set():
                        return False
                    del self._slots[key]
                    return True

            def size(self):
                with self._guard:
                    return len(self._slots)
        ''',
        '''
        def test_caches_the_factory_result():
            cache, calls = Cache(), []

            def factory():
                calls.append(1)
                return 'value'

            assert cache.get('k', factory) == 'value'
            assert cache.get('k', factory) == 'value'
            assert calls == [1]
            assert cache.size() == 1
            assert cache.invalidate('k') and not cache.invalidate('k')
            assert cache.size() == 0
        ''',
        '''
        import threading


        def _start(call):
            box = []

            def run():
                try:
                    box.append(('ok', call()))
                except BaseException as error:
                    box.append(('error', error))

            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            return thread, box


        def test_concurrent_calls_share_one_factory_run():
            cache, entered, release = Cache(), threading.Event(), threading.Event()
            runs = []

            def factory():
                runs.append(1)
                entered.set()
                release.wait(2)
                return 42

            first = _start(lambda: cache.get('k', factory))
            assert entered.wait(1)
            second = _start(lambda: cache.get('k', factory))
            probe = _start(lambda: (cache.invalidate('k'), cache.size()))
            second[0].join(0.05)
            assert second[0].is_alive()
            release.set()
            for thread, box in (first, second):
                thread.join(2)
                assert box == [('ok', 42)]
            probe[0].join(2)
            assert probe[1] == [('ok', (False, 1))]
            assert runs == [1]


        def test_other_keys_do_not_wait_and_waiters_can_time_out():
            cache, entered, release = Cache(), threading.Event(), threading.Event()

            def slow():
                entered.set()
                release.wait(2)
                return 'slow'

            owner = _start(lambda: cache.get('a', slow))
            assert entered.wait(1)
            other = _start(lambda: cache.get('b', lambda: 'fast', timeout=0.5))
            other[0].join(0.3)
            assert other[1] == [('ok', 'fast')]
            waiter = _start(lambda: cache.get('a', lambda: 'wrong', timeout=0.1))
            waiter[0].join(1)
            assert waiter[1][0][0] == 'error' and isinstance(waiter[1][0][1], TimeoutError)
            release.set()
            owner[0].join(2)
            assert owner[1] == [('ok', 'slow')]


        def test_errors_are_not_cached_and_reach_every_caller():
            cache, boom = Cache(), ValueError('boom')
            entered, release = threading.Event(), threading.Event()

            def failing():
                entered.set()
                release.wait(2)
                raise boom

            owner = _start(lambda: cache.get('k', failing))
            assert entered.wait(1)
            waiter = _start(lambda: cache.get('k', failing))
            probe = _start(cache.size)
            probe[0].join(0.05)
            assert probe[1] == [('ok', 1)]
            release.set()
            for thread, box in (owner, waiter):
                thread.join(2)
                assert box == [('error', boom)]
            runs = []
            assert cache.get('k', lambda: runs.append(1) or 'fresh') == 'fresh'
            assert runs == [1]
        ''',
    ),
    _task(
        "single_flight_ttl", "single_flight",
        "TtlCache 是多线程使用的缓存，条目在写入 ttl 秒后过期（ttl 在构造时给出）。"
        "get(key, factory, now) 中的 now 是调用方给出的秒数，不读系统时间。要求：\n"
        "1）条目未过期（now - 写入时间 < ttl）：直接返回缓存值，不调用 factory。\n"
        "2）条目不存在或已过期：执行 factory，把返回值和 now 作为新的条目保存；同一 key 并发时 factory "
        "只执行一次，其余调用者等待并得到同一个值；等待超过 timeout 抛 TimeoutError（默认 None 表示一直等）。\n"
        "3）factory 抛出异常：条目保持原样（原来的值和写入时间都不变，不存在时仍不存在），"
        "同一个异常对象抛给执行者和所有等待者。\n"
        "4）size() 返回保存的 key 个数（含已过期的）。",
        '''
        import threading


        class TtlCache:
            def __init__(self, ttl):
                self._ttl, self._guard, self._slots = ttl, threading.Lock(), {}

            def get(self, key, factory, now, timeout=None):
                with self._guard:
                    slot = self._slots.get(key)
                    if slot is not None and now - slot[0] <= self._ttl:
                        return slot[1]
                try:
                    value = factory()
                except Exception:
                    value = None
                with self._guard:
                    self._slots[key] = [now, value, None]
                return value

            def size(self):
                with self._guard:
                    return len(self._slots)
        ''',
        '''
        import threading


        class TtlCache:
            def __init__(self, ttl):
                self._ttl, self._guard, self._slots = ttl, threading.Lock(), {}

            def get(self, key, factory, now, timeout=None):
                with self._guard:
                    slot = self._slots.get(key)
                    if slot is not None and now - slot[0] < self._ttl:
                        return slot[1]
                    if slot is None:
                        slot = self._slots[key] = [None, None, None]
                    ticket, owner = slot[2], slot[2] is None
                    if owner:
                        ticket = slot[2] = [threading.Event(), None, None]
                if not owner:
                    if not ticket[0].wait(timeout):
                        raise TimeoutError(key)
                    if ticket[2] is not None:
                        raise ticket[2]
                    with self._guard:
                        return slot[1]
                try:
                    value = factory()
                except BaseException as error:
                    ticket[2] = error
                    with self._guard:
                        slot[2] = None
                        if slot[0] is None:
                            self._slots.pop(key, None)
                    ticket[0].set()
                    raise
                with self._guard:
                    slot[0], slot[1], slot[2] = now, value, None
                ticket[0].set()
                return value

            def size(self):
                with self._guard:
                    return len(self._slots)
        ''',
        '''
        def test_fresh_entries_are_reused():
            cache, calls = TtlCache(10), []

            def factory():
                calls.append(1)
                return 'value'

            assert cache.get('k', factory, 0) == 'value'
            assert cache.get('k', factory, 9) == 'value'
            assert calls == [1]
            assert cache.size() == 1
        ''',
        '''
        import threading


        def _start(call):
            box = []

            def run():
                try:
                    box.append(('ok', call()))
                except BaseException as error:
                    box.append(('error', error))

            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            return thread, box


        def test_boundary_and_reload():
            cache, calls = TtlCache(10), []

            def factory():
                calls.append(1)
                return 'new'

            assert cache.get('k', lambda: 'old', 0) == 'old'
            assert cache.get('k', factory, 10) == 'new'
            assert cache.get('k', factory, 19) == 'new'
            assert cache.get('k', lambda: 'later', 20) == 'later'
            assert calls == [1] and cache.size() == 1


        def test_expired_entry_is_reloaded_once_for_concurrent_callers():
            cache, entered, release = TtlCache(5), threading.Event(), threading.Event()

            def slow():
                entered.set()
                release.wait(2)
                return 'fresh'

            assert cache.get('k', lambda: 'stale', 0) == 'stale'
            owner = _start(lambda: cache.get('k', slow, 100))
            assert entered.wait(1)
            waiter = _start(lambda: cache.get('k', slow, 100, timeout=0.5))
            impatient = _start(lambda: cache.get('k', slow, 100, timeout=0.05))
            impatient[0].join(1)
            assert impatient[1][0][0] == 'error' and isinstance(impatient[1][0][1], TimeoutError)
            release.set()
            for thread, box in (owner, waiter):
                thread.join(2)
                assert box == [('ok', 'fresh')]
            assert cache.get('k', lambda: 'again', 104) == 'fresh'


        def test_failed_reload_keeps_the_stale_entry():
            cache, boom = TtlCache(5), ValueError('boom')
            entered, release = threading.Event(), threading.Event()

            def failing():
                entered.set()
                release.wait(2)
                raise boom

            assert cache.get('k', lambda: 'stale', 0) == 'stale'
            owner = _start(lambda: cache.get('k', failing, 100))
            assert entered.wait(1)
            waiter = _start(lambda: cache.get('k', failing, 100))
            release.set()
            for thread, box in (owner, waiter):
                thread.join(2)
                assert box == [('error', boom)]
            assert cache.get('k', lambda: 'wrong', 4) == 'stale'
            assert cache.get('k', lambda: 'reborn', 100) == 'reborn'
            assert cache.size() == 1


        def test_failure_on_a_missing_key_stores_nothing():
            cache, boom = TtlCache(5), KeyError('missing')

            def failing():
                raise boom

            with pytest.raises(KeyError) as caught:
                cache.get('k', failing, 0)
            assert caught.value is boom and cache.size() == 0
            assert cache.get('k', lambda: 'first', 0) == 'first'
        ''',
    ),
    _task(
        "rw_lock_prefer", "rw_lock",
        "RWLock 是一个读写锁，会被多个线程使用，readers() 返回当前持有读锁的份数。要求：\n"
        "1）多个读者可以同时持锁；写锁独占：有读者持锁时 acquire_write 等待，"
        "有写者持锁时 acquire_read 和 acquire_write 都等待。\n"
        "2）写者优先：只要有写者在等待，新来的读者就等待，直到该写者拿到并释放写锁；"
        "已经在持锁的读者不受影响。\n"
        "3）acquire_read / acquire_write 都接受可选 timeout（默认 None 表示一直等）：等锁超时抛 "
        "TimeoutError，并且不留任何等待记录（之后的读者不能被它挡住）。\n"
        "4）release_read / release_write 只由持锁线程调用；写者持锁期间 readers() 为 0。",
        '''
        import threading


        class RWLock:
            def __init__(self):
                self._cond = threading.Condition()
                self._readers, self._writers, self._waiting = 0, 0, 0

            def acquire_read(self, timeout=None):
                with self._cond:
                    self._cond.wait_for(lambda: self._writers == 0)
                    self._readers += 1

            def release_read(self):
                with self._cond:
                    self._readers -= 1

            def acquire_write(self, timeout=None):
                with self._cond:
                    self._waiting += 1
                    if not self._cond.wait_for(lambda: self._writers == 0 and self._readers == 0, timeout):
                        raise TimeoutError('write lock')
                    self._writers = 1
                    self._waiting -= 1

            def release_write(self):
                with self._cond:
                    self._writers = 0
                    self._cond.notify_all()

            def readers(self):
                return self._readers
        ''',
        '''
        import threading


        class RWLock:
            def __init__(self):
                self._cond = threading.Condition()
                self._readers, self._writers, self._waiting = 0, 0, 0

            def acquire_read(self, timeout=None):
                with self._cond:
                    if not self._cond.wait_for(
                            lambda: self._writers == 0 and self._waiting == 0, timeout):
                        raise TimeoutError('read lock')
                    self._readers += 1

            def release_read(self):
                with self._cond:
                    self._readers -= 1
                    if self._readers == 0:
                        self._cond.notify_all()

            def acquire_write(self, timeout=None):
                with self._cond:
                    self._waiting += 1
                    try:
                        if not self._cond.wait_for(
                                lambda: self._writers == 0 and self._readers == 0, timeout):
                            raise TimeoutError('write lock')
                        self._writers = 1
                    finally:
                        self._waiting -= 1
                        self._cond.notify_all()

            def release_write(self):
                with self._cond:
                    self._writers = 0
                    self._cond.notify_all()

            def readers(self):
                with self._cond:
                    return self._readers
        ''',
        '''
        def test_single_threaded_lock():
            lock = RWLock()
            lock.acquire_read()
            lock.acquire_read()
            assert lock.readers() == 2
            lock.release_read()
            lock.release_read()
            lock.acquire_write()
            assert lock.readers() == 0
            lock.release_write()
            lock.acquire_read()
            lock.release_read()
        ''',
        '''
        import threading


        def _start(call):
            box = []

            def run():
                try:
                    box.append(('ok', call()))
                except BaseException as error:
                    box.append(('error', error))

            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            return thread, box


        def _timed_out(result):
            thread, box = result
            thread.join(1)
            return box and box[0][0] == 'error' and isinstance(box[0][1], TimeoutError)


        def _hold_read(lock, done):
            def reader():
                lock.acquire_read()
                done.wait(2)
                lock.release_read()

            holder = threading.Thread(target=reader, daemon=True)
            holder.start()
            for _ in range(100):
                if lock.readers() == 1:
                    break
                holder.join(0.01)
            return holder


        def test_writer_waits_for_readers():
            lock, done = RWLock(), threading.Event()
            holder = _hold_read(lock, done)
            assert lock.readers() == 1
            assert _timed_out(_start(lambda: lock.acquire_write(timeout=0.05)))
            assert lock.readers() == 1
            done.set()
            holder.join(1)
            lock.acquire_write(timeout=0.5)
            assert lock.readers() == 0
            lock.release_write()


        def test_waiting_writer_blocks_new_readers():
            lock, done = RWLock(), threading.Event()
            holder = _hold_read(lock, done)
            writer = _start(lambda: (lock.acquire_write(), lock.release_write()))
            writer[0].join(0.05)
            assert writer[0].is_alive()
            assert _timed_out(_start(lambda: lock.acquire_read(timeout=0.05)))
            done.set()
            writer[0].join(1)
            assert writer[1] == [('ok', (None, None))]
            lock.acquire_read(timeout=0.5)
            lock.release_read()


        def test_timed_out_writer_leaves_no_trace():
            lock, done = RWLock(), threading.Event()
            holder = _hold_read(lock, done)
            assert _timed_out(_start(lambda: lock.acquire_write(timeout=0.05)))
            done.set()
            holder.join(1)
            after = _start(lambda: lock.acquire_read(timeout=0.3))
            after[0].join(1)
            assert after[1] == [('ok', None)]


        def test_writer_is_exclusive():
            lock = RWLock()
            lock.acquire_write()
            assert _timed_out(_start(lambda: lock.acquire_read(timeout=0.05)))
            assert _timed_out(_start(lambda: lock.acquire_write(timeout=0.05)))
            assert lock.readers() == 0
            lock.release_write()
            lock.acquire_read(timeout=0.5)
            assert lock.readers() == 1
            lock.release_read()
        ''',
    ),
    _task(
        "rw_lock_upgrade", "rw_lock",
        "RWLock 是一个读写锁，会被多个线程使用，readers() 返回当前读锁的份数。要求：\n"
        "1）多个读者可以同时持锁；写锁独占：有读者或写者持锁时 acquire_write 等待；"
        "有写者持锁时 acquire_read 等待。\n"
        "2）upgrade(timeout=None)：由持有一份读锁的线程调用，升级为写锁：只有当它是唯一的读者时才能升级，"
        "否则等待；升级成功后它不再持有读锁，之后用 release_write() 释放。\n"
        "3）升级超时抛 TimeoutError，并且仍持有原来的读锁。\n"
        "4）等待升级的线程算作等待中的写者：新来的读者要等它升级成功并 release_write 之后才能读。\n"
        "5）acquire_read / acquire_write / upgrade 都接受可选 timeout（默认 None 表示一直等）："
        "超时抛 TimeoutError，并且不留任何等待记录。",
        '''
        import threading


        class RWLock:
            def __init__(self):
                self._cond = threading.Condition()
                self._readers, self._writers, self._waiting = 0, 0, 0

            def acquire_read(self, timeout=None):
                with self._cond:
                    self._cond.wait_for(lambda: self._writers == 0 and self._waiting == 0)
                    self._readers += 1

            def release_read(self):
                with self._cond:
                    self._readers -= 1
                    if self._readers == 0:
                        self._cond.notify_all()

            def upgrade(self, timeout=None):
                with self._cond:
                    self._readers -= 1
                    if not self._cond.wait_for(lambda: self._readers == 0 and self._writers == 0, timeout):
                        raise TimeoutError('upgrade')
                    self._writers = 1

            def acquire_write(self, timeout=None):
                with self._cond:
                    self._waiting += 1
                    try:
                        if not self._cond.wait_for(
                                lambda: self._writers == 0 and self._readers == 0, timeout):
                            raise TimeoutError('write lock')
                        self._writers = 1
                    finally:
                        self._waiting -= 1
                        self._cond.notify_all()

            def release_write(self):
                with self._cond:
                    self._writers = 0
                    self._cond.notify_all()

            def readers(self):
                with self._cond:
                    return self._readers
        ''',
        '''
        import threading


        class RWLock:
            def __init__(self):
                self._cond = threading.Condition()
                self._readers, self._writers, self._waiting = 0, 0, 0

            def acquire_read(self, timeout=None):
                with self._cond:
                    if not self._cond.wait_for(
                            lambda: self._writers == 0 and self._waiting == 0, timeout):
                        raise TimeoutError('read lock')
                    self._readers += 1

            def release_read(self):
                with self._cond:
                    self._readers -= 1
                    self._cond.notify_all()

            def upgrade(self, timeout=None):
                with self._cond:
                    self._waiting += 1
                    try:
                        if not self._cond.wait_for(lambda: self._readers == 1 and self._writers == 0, timeout):
                            raise TimeoutError('upgrade')
                        self._readers, self._writers = 0, 1
                    finally:
                        self._waiting -= 1
                        self._cond.notify_all()

            def acquire_write(self, timeout=None):
                with self._cond:
                    self._waiting += 1
                    try:
                        if not self._cond.wait_for(
                                lambda: self._writers == 0 and self._readers == 0, timeout):
                            raise TimeoutError('write lock')
                        self._writers = 1
                    finally:
                        self._waiting -= 1
                        self._cond.notify_all()

            def release_write(self):
                with self._cond:
                    self._writers = 0
                    self._cond.notify_all()

            def readers(self):
                with self._cond:
                    return self._readers
        ''',
        '''
        def test_single_threaded_lock():
            lock = RWLock()
            lock.acquire_read()
            assert lock.readers() == 1
            lock.upgrade()
            assert lock.readers() == 0
            lock.release_write()
            lock.acquire_write()
            lock.release_write()
        ''',
        '''
        import threading


        def _start(call):
            box = []

            def run():
                try:
                    box.append(('ok', call()))
                except BaseException as error:
                    box.append(('error', error))

            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            return thread, box


        def _timed_out(result):
            thread, box = result
            thread.join(1)
            return bool(box) and box[0][0] == 'error' and isinstance(box[0][1], TimeoutError)


        def _hold_read(lock, done):
            def reader():
                lock.acquire_read()
                done.wait(2)
                lock.release_read()

            holder = threading.Thread(target=reader, daemon=True)
            holder.start()
            for _ in range(100):
                if lock.readers() == 1:
                    break
                holder.join(0.01)
            return holder


        def test_upgrade_with_a_single_reader():
            lock = RWLock()
            lock.acquire_read()
            lock.upgrade(timeout=0.5)
            assert lock.readers() == 0
            assert _timed_out(_start(lambda: lock.acquire_read(timeout=0.05)))
            lock.release_write()
            lock.acquire_read(timeout=0.5)
            assert lock.readers() == 1
            lock.release_read()


        def test_upgrade_waits_for_other_readers_and_keeps_the_read_lock_on_timeout():
            lock, done = RWLock(), threading.Event()
            holder = _hold_read(lock, done)
            assert lock.readers() == 1
            lock.acquire_read()
            with pytest.raises(TimeoutError):
                lock.upgrade(timeout=0.05)
            assert lock.readers() == 2
            done.set()
            holder.join(1)
            late = _start(lambda: lock.acquire_read(timeout=0.3))
            late[0].join(1)
            assert late[1] == [('ok', None)]
            assert lock.readers() == 2


        def test_waiting_upgrade_blocks_new_readers():
            lock, done, seen = RWLock(), threading.Event(), []
            holder = _hold_read(lock, done)
            ready = threading.Event()

            def upgrade_as_second_reader():
                lock.acquire_read()
                ready.set()
                lock.upgrade()
                seen.append(lock.readers())
                lock.release_write()

            upgrader = threading.Thread(target=upgrade_as_second_reader, daemon=True)
            upgrader.start()
            assert ready.wait(1)
            upgrader.join(0.05)
            assert upgrader.is_alive()
            assert _timed_out(_start(lambda: lock.acquire_read(timeout=0.05)))
            assert lock.readers() == 2
            done.set()
            upgrader.join(1)
            assert not upgrader.is_alive() and seen == [0]
            lock.acquire_read(timeout=0.5)
            lock.release_read()


        def test_writer_is_exclusive():
            lock = RWLock()
            lock.acquire_write()
            assert _timed_out(_start(lambda: lock.acquire_read(timeout=0.05)))
            assert _timed_out(_start(lambda: lock.acquire_write(timeout=0.05)))
            lock.release_write()
            lock.acquire_read(timeout=0.5)
            lock.release_read()
        ''',
    ),
    _task(
        "shard_counter_snapshot", "shard_counter",
        "ShardedCounter(shards) 把计数分到给定名字的若干分片上，每个分片一把锁，会被多个线程使用。要求：\n"
        "1）add(shard, amount) / value(shard)：只锁这一个分片；分片不存在抛 IndexError。\n"
        "2）move(src, dst, amount)：原子地把 amount 从 src 移到 dst 并返回 True；amount ≤ 0 抛 ValueError；"
        "分片不存在抛 IndexError；src 余额不足抛 ValueError，失败时两边都不变；"
        "src 与 dst 相同时不做任何事并返回 False。\n"
        "3）方向相反的两个 move 不能死锁。\n"
        "4）snapshot()：返回 {分片: 值} 的字典，等于某一时刻的状态；执行期间要同时拿到所有分片的锁。\n"
        "5）add / move / snapshot 都接受可选 timeout（默认 None 表示一直等）：等锁超时抛 TimeoutError，"
        "且不改变任何值。\n"
        "6）pin(shard, ready, gate)：持有该分片的锁，拿到锁后设置 ready，直到 gate 被设置才释放；"
        "这期间其他分片上的 add / move / value 照常完成。",
        '''
        import threading
        from contextlib import contextmanager


        class ShardedCounter:
            def __init__(self, shards):
                self._values = {shard: 0 for shard in shards}
                self._locks = {shard: threading.Lock() for shard in shards}

            @contextmanager
            def _guard(self, shards, timeout):
                taken = []
                try:
                    for shard in shards:
                        self._locks[shard].acquire(True, -1 if timeout is None else timeout)
                        taken.append(self._locks[shard])
                    yield
                finally:
                    for lock in reversed(taken):
                        lock.release()

            def add(self, shard, amount, timeout=None):
                with self._guard([shard], timeout):
                    self._values[shard] += amount
                    return self._values[shard]

            def value(self, shard):
                return self._values[shard]

            def move(self, src, dst, amount, timeout=None):
                if amount <= 0:
                    raise ValueError('amount must be positive')
                with self._guard([src, dst], timeout):
                    self._values[src] -= amount
                    if self._values[src] < 0:
                        raise ValueError('insufficient funds')
                    self._values[dst] += amount
                    return True

            def snapshot(self, timeout=None):
                return dict(self._values)

            def pin(self, shard, ready, gate, timeout=None):
                with self._guard([shard], timeout):
                    ready.set()
                    gate.wait()
        ''',
        '''
        import threading
        from contextlib import contextmanager


        class ShardedCounter:
            def __init__(self, shards):
                self._values = {shard: 0 for shard in shards}
                self._locks = {shard: threading.Lock() for shard in shards}

            @contextmanager
            def _guard(self, shards, timeout):
                taken = []
                try:
                    for shard in sorted(shards):
                        lock = self._locks.get(shard)
                        if lock is None:
                            raise IndexError(shard)
                        if not lock.acquire(True, -1 if timeout is None else timeout):
                            raise TimeoutError(shard)
                        taken.append(lock)
                    yield
                finally:
                    for lock in reversed(taken):
                        lock.release()

            def add(self, shard, amount, timeout=None):
                with self._guard([shard], timeout):
                    self._values[shard] += amount
                    return self._values[shard]

            def value(self, shard):
                with self._guard([shard], None):
                    return self._values[shard]

            def move(self, src, dst, amount, timeout=None):
                if amount <= 0:
                    raise ValueError('amount must be positive')
                if src not in self._locks or dst not in self._locks:
                    raise IndexError(src)
                if src == dst:
                    return False
                with self._guard([src, dst], timeout):
                    if self._values[src] < amount:
                        raise ValueError('insufficient funds')
                    self._values[src] -= amount
                    self._values[dst] += amount
                    return True

            def snapshot(self, timeout=None):
                with self._guard(self._values, timeout):
                    return dict(self._values)

            def pin(self, shard, ready, gate, timeout=None):
                with self._guard([shard], timeout):
                    ready.set()
                    gate.wait()
        ''',
        '''
        def test_single_threaded_counter():
            counter = ShardedCounter(['a', 'b'])
            assert counter.add('a', 5) == 5
            assert counter.value('a') == 5
            assert counter.move('a', 'b', 2)
            assert counter.snapshot() == {'a': 3, 'b': 2}
        ''',
        {
            "state": '''
                def test_validation_and_failed_moves():
                    counter = ShardedCounter(['a', 'b'])
                    counter.add('a', 1)
                    with pytest.raises(IndexError):
                        counter.add('zz', 1)
                    with pytest.raises(IndexError):
                        counter.value('zz')
                    with pytest.raises(ValueError):
                        counter.move('a', 'b', 0)
                    with pytest.raises(IndexError):
                        counter.move('a', 'zz', 1)
                    with pytest.raises(ValueError):
                        counter.move('a', 'b', 5)
                    assert counter.snapshot() == {'a': 1, 'b': 0}
                    assert counter.move('a', 'a', 1, timeout=0.3) is False
                    assert counter.snapshot() == {'a': 1, 'b': 0}
            ''',
            "threads": '''
                import threading


                def _start(call):
                    box = []

                    def run():
                        try:
                            box.append(('ok', call()))
                        except BaseException as error:
                            box.append(('error', error))

                    thread = threading.Thread(target=run, daemon=True)
                    thread.start()
                    return thread, box


                def _timed_out(result):
                    thread, box = result
                    thread.join(1)
                    return bool(box) and box[0][0] == 'error' and isinstance(box[0][1], TimeoutError)


                def test_pinned_shard_blocks_only_its_own_operations():
                    counter = ShardedCounter(['a', 'b', 'c'])
                    ready, gate = threading.Event(), threading.Event()
                    blocker = _start(lambda: counter.pin('a', ready, gate))
                    assert ready.wait(1)
                    assert _timed_out(_start(lambda: counter.add('a', 1, timeout=0.05)))
                    assert _timed_out(_start(lambda: counter.snapshot(timeout=0.05)))
                    assert _timed_out(_start(lambda: counter.move('b', 'a', 1, timeout=0.05)))
                    assert counter.add('b', 2, timeout=0.5) == 2
                    assert counter.move('b', 'c', 1, timeout=0.5) is True
                    assert counter.value('b') == 1
                    gate.set()
                    blocker[0].join(1)
                    assert blocker[1] == [('ok', None)]
                    assert counter.snapshot(timeout=0.5) == {'a': 0, 'b': 1, 'c': 1}


                def test_opposite_moves_do_not_deadlock():
                    counter = ShardedCounter(['a', 'b'])
                    counter.add('a', 10)
                    counter.add('b', 10)
                    ready, gate = threading.Event(), threading.Event()
                    blocker = _start(lambda: counter.pin('a', ready, gate))
                    assert ready.wait(1)
                    first = _start(lambda: counter.move('a', 'b', 3))
                    second = _start(lambda: counter.move('b', 'a', 4))
                    first[0].join(0.05)
                    second[0].join(0.05)
                    assert first[0].is_alive() and second[0].is_alive()
                    free = threading.Event()
                    free.set()
                    counter.pin('b', threading.Event(), free, timeout=0.05)
                    gate.set()
                    for result in (blocker, first, second):
                        result[0].join(1)
                    assert all(not result[0].is_alive() for result in (blocker, first, second))
                    assert counter.snapshot(timeout=0.5) == {'a': 11, 'b': 9}
            ''',
        },
    ),
    _task(
        "shard_counter_bulk", "shard_counter",
        "LimitedCounter(shards, cap) 把计数分到给定名字的若干分片上，每个分片的值必须始终在 0 到 cap 之间"
        "（含两端），每个分片一把锁，会被多个线程使用。要求：\n"
        "1）add(shard, amount)：结果越界抛 ValueError 且值不变；分片不存在抛 IndexError。\n"
        "2）bulk_add(pairs)：pairs 是 (分片, 增量) 的列表，整体原子——任何一片不存在或越界时全部不生效，"
        "并抛出对应的异常（不存在 IndexError，越界 ValueError）。\n"
        "3）bulk_add 的校验顺序：先查分片名重复（抛 ValueError），再查分片是否存在，"
        "最后按分片名排序检查越界，抛出排序后第一个越界分片的 ValueError（消息就是分片名）。\n"
        "4）move(src, dst, amount)：amount ≤ 0、src 不足或 dst 越界都抛 ValueError，失败时两边都不变。\n"
        "5）所有操作接受可选 timeout（默认 None 表示一直等）：等锁超时抛 TimeoutError，且不改变任何值。\n"
        "6）values() 返回 {分片: 值} 的字典，等于某一时刻的状态；方向相反的操作不能死锁。\n"
        "7）pin(shard, ready, gate)：持有该分片的锁，拿到锁后设置 ready，直到 gate 被设置才释放。",
        '''
        import threading
        from contextlib import contextmanager


        class LimitedCounter:
            def __init__(self, shards, cap):
                self._cap = cap
                self._values = {shard: 0 for shard in shards}
                self._locks = {shard: threading.Lock() for shard in shards}

            @contextmanager
            def _guard(self, shards, timeout):
                taken = []
                try:
                    for shard in shards:
                        self._locks[shard].acquire(True, -1 if timeout is None else timeout)
                        taken.append(self._locks[shard])
                    yield
                finally:
                    for lock in reversed(taken):
                        lock.release()

            def _check(self, shard, delta):
                self._values[shard] += delta
                if not 0 <= self._values[shard] <= self._cap:
                    raise ValueError(shard)
                return self._values[shard]

            def add(self, shard, amount, timeout=None):
                with self._guard([shard], timeout):
                    self._values[shard] = self._check(shard, amount)

            def bulk_add(self, pairs, timeout=None):
                with self._guard([shard for shard, _ in pairs], timeout):
                    for shard, amount in dict(pairs).items():
                        self._values[shard] = self._check(shard, amount)

            def move(self, src, dst, amount, timeout=None):
                if amount <= 0:
                    raise ValueError('amount must be positive')
                with self._guard([src, dst], timeout):
                    self._check(src, -amount)
                    self._check(dst, amount)
                    return True

            def values(self):
                return dict(self._values)

            def pin(self, shard, ready, gate, timeout=None):
                with self._guard([shard], timeout):
                    ready.set()
                    gate.wait()
        ''',
        '''
        import threading
        from contextlib import contextmanager


        class LimitedCounter:
            def __init__(self, shards, cap):
                self._cap = cap
                self._values = {shard: 0 for shard in shards}
                self._locks = {shard: threading.Lock() for shard in shards}

            @contextmanager
            def _guard(self, shards, timeout):
                taken = []
                try:
                    for shard in sorted(shards):
                        lock = self._locks.get(shard)
                        if lock is None:
                            raise IndexError(shard)
                        if not lock.acquire(True, -1 if timeout is None else timeout):
                            raise TimeoutError(shard)
                        taken.append(lock)
                    yield
                finally:
                    for lock in reversed(taken):
                        lock.release()

            def _check(self, shard, delta):
                value = self._values[shard] + delta
                if 0 <= value <= self._cap:
                    return value
                raise ValueError(shard)

            def add(self, shard, amount, timeout=None):
                with self._guard([shard], timeout):
                    self._values[shard] = self._check(shard, amount)

            def bulk_add(self, pairs, timeout=None):
                names = [shard for shard, _ in pairs]
                if len(set(names)) != len(names):
                    raise ValueError('duplicate shard')
                changes = dict(pairs)
                with self._guard(names, timeout):
                    for shard in sorted(changes):
                        self._check(shard, changes[shard])
                    for shard, amount in changes.items():
                        self._values[shard] += amount

            def move(self, src, dst, amount, timeout=None):
                if amount <= 0:
                    raise ValueError('amount must be positive')
                with self._guard([src, dst], timeout):
                    self._check(src, -amount)
                    self._check(dst, amount)
                    self._values[src] -= amount
                    self._values[dst] += amount
                    return True

            def values(self, timeout=None):
                with self._guard(self._values, timeout):
                    return dict(self._values)

            def pin(self, shard, ready, gate, timeout=None):
                with self._guard([shard], timeout):
                    ready.set()
                    gate.wait()
        ''',
        '''
        def test_single_threaded_counter():
            counter = LimitedCounter(['a', 'b'], 10)
            counter.add('a', 4)
            counter.bulk_add([('a', 1), ('b', 2)])
            assert counter.move('a', 'b', 2)
            assert counter.values() == {'a': 3, 'b': 4}
        ''',
        {
            "state": '''
                def test_out_of_range_changes_leave_values_untouched():
                    counter = LimitedCounter(['a', 'b'], 10)
                    counter.add('a', 8)
                    with pytest.raises(ValueError):
                        counter.add('a', 3)
                    with pytest.raises(ValueError):
                        counter.add('b', -1)
                    with pytest.raises(IndexError):
                        counter.add('zz', 1)
                    assert counter.values() == {'a': 8, 'b': 0}
                    with pytest.raises(ValueError):
                        counter.bulk_add([('a', -8), ('b', 11)])
                    with pytest.raises(IndexError):
                        counter.bulk_add([('a', 1), ('zz', 1)])
                    assert counter.values() == {'a': 8, 'b': 0}
                    counter.add('b', 9)
                    with pytest.raises(ValueError):
                        counter.move('a', 'b', 2)
                    with pytest.raises(ValueError):
                        counter.move('b', 'a', 10)
                    with pytest.raises(ValueError):
                        counter.move('a', 'b', 0)
                    assert counter.values() == {'a': 8, 'b': 9}


                def test_bulk_add_validation_order():
                    counter = LimitedCounter(['a', 'b'], 10)
                    with pytest.raises(ValueError, match='duplicate'):
                        counter.bulk_add([('zz', 1), ('zz', 1)])
                    with pytest.raises(IndexError):
                        counter.bulk_add([('zz', 1), ('b', 11)])
                    with pytest.raises(ValueError, match='^a$'):
                        counter.bulk_add([('b', 11), ('a', -1)])
                    assert counter.values() == {'a': 0, 'b': 0}
            ''',
            "threads": '''
                import threading


                def _start(call):
                    box = []

                    def run():
                        try:
                            box.append(('ok', call()))
                        except BaseException as error:
                            box.append(('error', error))

                    thread = threading.Thread(target=run, daemon=True)
                    thread.start()
                    return thread, box


                def _timed_out(result):
                    thread, box = result
                    thread.join(1)
                    return bool(box) and box[0][0] == 'error' and isinstance(box[0][1], TimeoutError)


                def test_pinned_shard_and_opposite_operations():
                    counter = LimitedCounter(['a', 'b'], 100)
                    counter.bulk_add([('a', 5), ('b', 5)])
                    ready, gate = threading.Event(), threading.Event()
                    blocker = _start(lambda: counter.pin('a', ready, gate))
                    assert ready.wait(1)
                    assert _timed_out(_start(lambda: counter.add('a', 1, timeout=0.05)))
                    assert _timed_out(_start(lambda: counter.bulk_add([('a', 1), ('b', 1)], timeout=0.05)))
                    assert _timed_out(_start(lambda: counter.values(timeout=0.05)))
                    assert counter.bulk_add([('b', 5)], timeout=0.5) is None
                    first = _start(lambda: counter.move('a', 'b', 1))
                    second = _start(lambda: counter.move('b', 'a', 1))
                    first[0].join(0.05)
                    second[0].join(0.05)
                    assert first[0].is_alive() and second[0].is_alive()
                    free = threading.Event()
                    free.set()
                    counter.pin('b', threading.Event(), free, timeout=0.05)
                    gate.set()
                    for result in (blocker, first, second):
                        result[0].join(1)
                    assert all(not result[0].is_alive() for result in (blocker, first, second))
                    assert counter.values(timeout=0.5) == {'a': 5, 'b': 10}
            ''',
        },
    ),
]

MUTANTS = {
    "h3_rw_lock_upgrade_001": [
        ("lambda: self._writers == 0 and self._waiting == 0, timeout",
         "lambda: self._writers == 0, timeout"),
        ("            self._readers -= 1\n"
         "            self._cond.notify_all()\n",
         "            self._readers -= 1\n"),
        ("                if not self._cond.wait_for(lambda: self._readers == 1 and self._writers == 0, timeout):\n"
         "                    raise TimeoutError('upgrade')\n",
         "                self._readers -= 1\n"
         "                if not self._cond.wait_for(lambda: self._readers == 1 and self._writers == 0, timeout):\n"
         "                    raise TimeoutError('upgrade')\n"),
        ("                    raise TimeoutError('upgrade')\n",
         "                    self._waiting += 1\n"
         "                    raise TimeoutError('upgrade')\n"),
    ],
    "h3_shard_counter_snapshot_001": [
        ("for shard in sorted(shards):", "for shard in shards:"),
        ("if not lock.acquire(True, -1 if timeout is None else timeout):",
         "if lock.acquire(True, -1 if timeout is None else timeout) and False:"),
        ("        with self._guard(self._values, timeout):\n"
         "            return dict(self._values)\n",
         "        if True:\n"
         "            return dict(self._values)\n"),
        ("        if src == dst:\n"
         "            return False\n", ""),
        ("            if self._values[src] < amount:\n"
         "                raise ValueError('insufficient funds')\n", ""),
    ],
    "h3_shard_counter_bulk_001": [
        ("for shard in sorted(shards):", "for shard in shards:"),
        ("        if len(set(names)) != len(names):\n"
         "            raise ValueError('duplicate shard')\n", ""),
        ("            for shard in sorted(changes):\n"
         "                self._check(shard, changes[shard])\n"
         "            for shard, amount in changes.items():\n"
         "                self._values[shard] += amount\n",
         "            for shard, amount in changes.items():\n"
         "                self._values[shard] = self._check(shard, amount)\n"),
        ("            for shard in sorted(changes):\n", "            for shard in changes:\n"),
        ("            self._check(src, -amount)\n"
         "            self._check(dst, amount)\n"
         "            self._values[src] -= amount\n",
         "            self._check(src, -amount)\n"
         "            self._values[src] -= amount\n"
         "            self._check(dst, amount)\n"),
    ],
    "h3_single_flight_ttl_001": [
        ("            if not ticket[0].wait(timeout):\n"
         "                raise TimeoutError(key)\n", ""),
        ("now - slot[0] < self._ttl", "now - slot[0] <= self._ttl"),
        ("                slot[2] = None\n"
         "                if slot[0] is None:\n"
         "                    self._slots.pop(key, None)\n",
         "                slot[2] = None\n"),
        ("            if ticket[2] is not None:\n"
         "                raise ticket[2]\n", ""),
        ("            ticket[2] = error\n"
         "            with self._guard:\n",
         "            ticket[2] = error\n"
         "            slot[0] = now\n"
         "            with self._guard:\n"),
    ],
    "h3_rw_lock_prefer_001": [
        ("lambda: self._writers == 0 and self._waiting == 0, timeout",
         "lambda: self._writers == 0, timeout"),
        ("            if self._readers == 0:\n"
         "                self._cond.notify_all()\n", ""),
        ("                self._writers = 1\n"
         "            finally:\n"
         "                self._waiting -= 1\n"
         "                self._cond.notify_all()\n",
         "                self._writers = 1\n"
         "                self._waiting -= 1\n"
         "                self._cond.notify_all()\n"),
        ("lambda: self._writers == 0 and self._waiting == 0, timeout",
         "lambda: self._writers == 0 and self._waiting == 0"),
    ],
    "h3_transfer_lock_limit_001": [
        ("for name in sorted(set(names)):", "for name in names:"),
        ("        if src not in self._accounts or dst not in self._accounts:\n"
         "            raise KeyError(src)\n"
         "        if amount <= 0:\n"
         "            raise ValueError('amount must be positive')\n",
         "        if amount <= 0:\n"
         "            raise ValueError('amount must be positive')\n"
         "        if src not in self._accounts or dst not in self._accounts:\n"
         "            raise KeyError(src)\n"),
        ("        if source[2]:\n"
         "            raise PermissionError('frozen')\n"
         "        if source[0] - amount < -source[1]:\n"
         "            raise ValueError('over limit')\n"
         "        source[0] -= amount\n",
         "        source[0] -= amount\n"
         "        if source[2]:\n"
         "            raise PermissionError('frozen')\n"
         "        if source[0] < -source[1]:\n"
         "            raise ValueError('over limit')\n"),
        ("                raise ValueError('nothing to transfer')\n"
         "            return self._take(src, dst, amount)",
         "                raise ValueError('nothing to transfer')\n"
         "            return self.transfer(src, dst, amount, timeout)"),
        ("        with self._guard([name], timeout):\n"
         "            self._accounts[name][2] = True",
         "        if True:\n"
         "            self._accounts[name][2] = True"),
    ],
    "h3_single_flight_cache_001": [
        ("            if not slot[0].wait(timeout):\n"
         "                raise TimeoutError(key)\n"
         "            if slot[2] is not None:\n"
         "                raise slot[2]\n"
         "            return slot[1]\n",
         "            return slot[1]\n"),
        ("            with self._guard:\n"
         "                self._slots.pop(key, None)\n"
         "            raise\n",
         "            raise\n"),
        ("if slot is None or not slot[0].is_set():", "if slot is None:"),
        ("        finally:\n"
         "            slot[0].set()",
         "        finally:\n"
         "            slot[0].set() if slot[2] is None else None"),
    ],
    "h3_transfer_lock_bank_001": [
        ("for name in sorted(set(names)):", "for name in names:"),
        ("            if self._accounts[src] < amount:\n"
         "                raise ValueError('insufficient funds')\n", ""),
        ("if not lock.acquire(True, -1 if timeout is None else timeout):",
         "if lock.acquire(True, -1 if timeout is None else timeout) and False:"),
        ("        with self._guard([name], timeout):\n"
         "            if self._accounts[name] < amount:",
         "        if True:\n"
         "            if self._accounts[name] < amount:"),
        ("with self._guard(list(self._accounts), timeout):", "if True:"),
    ],
}
