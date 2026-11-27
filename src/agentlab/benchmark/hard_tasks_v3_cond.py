"""Condition-variable and worker-lifecycle tasks for the hard batch (h3).

Every program here has threads that wait for a predicate or for work that
another thread produces.  The defects are the ones a per-method lock does not
fix: a wait that is never re-checked, a close that forgets some of its
waiters, a result recorded for the wrong caller, a stop that quietly changes
what happens to queued work.  Every wait in the private tests is bounded, and
every thread that could block under the buggy program is a daemon thread.
"""
from __future__ import annotations

from agentlab.benchmark.hard_tasks_v3_common import _task

SPECS = [
    _task(
        "bounded_channel", "bounded_channel",
        "Channel(capacity) 是多线程使用的有界通道，item 按放入顺序取出。要求：\n"
        "1）put(item, timeout=None)：有空位时放入并返回 True；已满则阻塞到有空位；"
        "等待超过 timeout 秒抛 TimeoutError，此时 item 不放入。\n"
        "2）take(timeout=None)：取出最早的 item；为空则阻塞到有 item；等待超时抛 TimeoutError。\n"
        "3）timeout 为 None 表示一直等待。\n"
        "4）close() 可重复调用。关闭后 put 立即抛 ChannelClosed（即使有空位），item 不放入。\n"
        "5）关闭后 take 仍按顺序取走剩余 item；关闭且已空时 take 立即抛 ChannelClosed，"
        "不阻塞（无论 timeout）。\n"
        "6）close() 唤醒所有等待中的线程：等待中的 put 抛 ChannelClosed，等待中的 take 按 5）处理。\n"
        "7）多个 take 同时等待时，一次 put 只交给其中一个，其余的继续等待，不抛异常。\n"
        "8）take 取走 item 后，因已满而等待的 put 必须随即完成。\n",
        '''
        import threading
        import time


        class ChannelClosed(Exception):
            pass


        class Channel:
            def __init__(self, capacity):
                self._items = []
                self._capacity = capacity
                self._closed = False
                self._lock = threading.Lock()
                self._room = threading.Condition(self._lock)
                self._data = threading.Condition(self._lock)

            def _await(self, cond, ready, timeout):
                deadline = None if timeout is None else time.monotonic() + timeout
                if not ready():
                    left = None if deadline is None else deadline - time.monotonic()
                    cond.wait(left)
                return ready()

            def _has_room(self):
                return len(self._items) < self._capacity or self._closed

            def _has_item(self):
                return bool(self._items) or self._closed

            def put(self, item, timeout=None):
                with self._room:
                    if len(self._items) >= self._capacity:
                        if not self._await(self._room, self._has_room, timeout):
                            raise TimeoutError('put timed out')
                        if self._closed:
                            raise ChannelClosed('channel is closed')
                    self._items.append(item)
                    self._data.notify_all()
                    return True

            def take(self, timeout=None):
                with self._data:
                    if not self._await(self._data, self._has_item, timeout):
                        raise TimeoutError('take timed out')
                    if not self._items:
                        raise ChannelClosed('channel is closed')
                    return self._items.pop(0)

            def close(self):
                with self._room:
                    self._closed = True
                    self._data.notify_all()
        ''',
        '''
        import threading
        import time


        class ChannelClosed(Exception):
            pass


        class Channel:
            def __init__(self, capacity):
                self._items = []
                self._capacity = capacity
                self._closed = False
                self._lock = threading.Lock()
                self._room = threading.Condition(self._lock)
                self._data = threading.Condition(self._lock)

            def _ensure_open(self):
                if self._closed:
                    raise ChannelClosed('channel is closed')

            def _await(self, cond, ready, timeout):
                deadline = None if timeout is None else time.monotonic() + timeout
                while not ready():
                    left = None if deadline is None else deadline - time.monotonic()
                    if left is not None and left <= 0:
                        return False
                    cond.wait(left)
                return True

            def _has_room(self):
                return len(self._items) < self._capacity or self._closed

            def _has_item(self):
                return bool(self._items) or self._closed

            def put(self, item, timeout=None):
                with self._room:
                    if not self._await(self._room, self._has_room, timeout):
                        raise TimeoutError('put timed out')
                    self._ensure_open()
                    self._items.append(item)
                    self._data.notify_all()
                    return True

            def take(self, timeout=None):
                with self._data:
                    if not self._await(self._data, self._has_item, timeout):
                        raise TimeoutError('take timed out')
                    if not self._items:
                        raise ChannelClosed('channel is closed')
                    item = self._items.pop(0)
                    self._room.notify_all()
                    return item

            def close(self):
                with self._room:
                    self._closed = True
                    self._room.notify_all()
                    self._data.notify_all()
        ''',
        '''
        import pytest


        def test_single_threaded_channel():
            channel = Channel(2)
            assert channel.put('a') and channel.put('b')
            assert channel.take() == 'a'
            channel.put('c', timeout=1)
            channel.close()
            assert channel.take() == 'b' and channel.take() == 'c'
            with pytest.raises(ChannelClosed):
                channel.take()
        ''',
        '''
        import threading


        def _start(call):
            out = []

            def run():
                try:
                    out.append(call())
                except Exception as error:
                    out.append(error)

            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            return thread, out


        def _blocked(*threads):
            for thread in threads:
                thread.join(0.1)
            assert all(thread.is_alive() for thread in threads)


        def test_timeouts_and_close_leave_only_the_stored_items():
            channel = Channel(1)
            with pytest.raises(TimeoutError):
                channel.take(timeout=0.05)
            channel.put('a', timeout=1)
            with pytest.raises(TimeoutError):
                channel.put('b', timeout=0.05)
            channel.close()
            channel.close()
            with pytest.raises(ChannelClosed):
                channel.put('c', timeout=1)
            assert channel.take(timeout=1) == 'a'
            with pytest.raises(ChannelClosed):
                channel.take(timeout=1)


        def test_close_releases_blocked_taker():
            channel = Channel(1)
            taker, out = _start(channel.take)
            _blocked(taker)
            channel.close()
            taker.join(0.5)
            assert not taker.is_alive() and isinstance(out[0], ChannelClosed)


        def test_close_releases_blocked_putter():
            channel = Channel(1)
            channel.put('a', timeout=1)
            putter, out = _start(lambda: channel.put('b', timeout=5))
            _blocked(putter)
            channel.close()
            putter.join(0.5)
            assert not putter.is_alive() and isinstance(out[0], ChannelClosed)
            assert channel.take(timeout=1) == 'a'
            with pytest.raises(ChannelClosed):
                channel.take(timeout=1)


        def test_take_releases_blocked_putter():
            channel = Channel(1)
            channel.put('a', timeout=1)
            putter, out = _start(lambda: channel.put('b', timeout=5))
            _blocked(putter)
            assert channel.take(timeout=1) == 'a'
            putter.join(0.5)
            assert out == [True] and channel.take(timeout=1) == 'b'


        def test_taker_that_finds_no_item_waits_again():
            channel = Channel(1)
            first, out_a = _start(channel.take)
            second, out_b = _start(channel.take)
            _blocked(first, second)
            channel.put('x', timeout=1)
            first.join(0.3)
            second.join(0.3)
            assert first.is_alive() != second.is_alive()
            waiting = first if first.is_alive() else second
            channel.put('y', timeout=1)
            waiting.join(0.5)
            assert not waiting.is_alive()
            assert sorted(out_a + out_b) == ['x', 'y']
        ''',
    ),
    _task(
        "bounded_channel_drain", "bounded_channel",
        "Channel(capacity) 是多线程使用的有界通道，item 按放入顺序取出。要求：\n"
        "1）put(item, timeout=None)：有空位时放入并返回 True；已满则阻塞到有空位；"
        "等待超过 timeout 秒抛 TimeoutError，此时 item 不放入（之后也取不到）。\n"
        "2）take(timeout=None)：取出最早的 item；为空则阻塞到有 item；等待超时抛 TimeoutError，"
        "不返回 None。\n"
        "3）timeout 为 None 表示一直等待。\n"
        "4）close() 可重复调用。关闭后 put 立即抛 ChannelClosed（即使有空位）。\n"
        "5）关闭后 take 仍按顺序取走关闭前剩余的 item，它们不会丢失；"
        "关闭且已空时 take 立即抛 ChannelClosed，不阻塞（无论 timeout）。\n"
        "6）close() 唤醒所有等待中的线程：等待中的 put 抛 ChannelClosed，item 不放入；"
        "等待中的 take 按 5）处理。\n"
        "7）put 之后，正在等待的 take 必须随即被唤醒并取到 item。\n"
        "8）多个 take 同时等待时，一次 put 只交给其中一个，其余的继续等待，不抛异常。\n",
        '''
        import threading


        class ChannelClosed(Exception):
            pass


        class Channel:
            def __init__(self, capacity):
                self._items = []
                self._capacity = capacity
                self._closed = False
                self._lock = threading.Lock()
                self._room = threading.Condition(self._lock)
                self._data = threading.Condition(self._lock)

            def put(self, item, timeout=None):
                with self._lock:
                    if self._closed:
                        raise ChannelClosed('channel is closed')
                    if len(self._items) >= self._capacity:
                        self._room.wait(timeout)
                    self._items.append(item)
                    self._room.notify_all()
                    return True

            def take(self, timeout=None):
                with self._lock:
                    if not self._items and not self._data.wait(timeout):
                        return None
                    if not self._items:
                        raise ChannelClosed('channel is closed')
                    item = self._items.pop(0)
                    self._room.notify_all()
                    return item

            def close(self):
                with self._lock:
                    self._closed = True
                    self._items.clear()
                    self._room.notify_all()
                    self._data.notify_all()
        ''',
        '''
        import threading


        class ChannelClosed(Exception):
            pass


        class Channel:
            def __init__(self, capacity):
                self._items = []
                self._capacity = capacity
                self._closed = False
                self._lock = threading.Lock()
                self._room = threading.Condition(self._lock)
                self._data = threading.Condition(self._lock)

            def _ensure_open(self):
                if self._closed:
                    raise ChannelClosed('channel is closed')

            def put(self, item, timeout=None):
                with self._lock:
                    if not self._room.wait_for(
                            lambda: len(self._items) < self._capacity or self._closed, timeout):
                        raise TimeoutError('put timed out')
                    self._ensure_open()
                    self._items.append(item)
                    self._data.notify_all()
                    return True

            def take(self, timeout=None):
                with self._lock:
                    if not self._data.wait_for(lambda: self._items or self._closed, timeout):
                        raise TimeoutError('take timed out')
                    if not self._items:
                        raise ChannelClosed('channel is closed')
                    item = self._items.pop(0)
                    self._room.notify_all()
                    return item

            def close(self):
                with self._lock:
                    self._closed = True
                    self._room.notify_all()
                    self._data.notify_all()
        ''',
        '''
        import pytest


        def test_single_threaded_channel():
            channel = Channel(2)
            assert channel.put('a') and channel.put('b')
            assert channel.take() == 'a' and channel.take() == 'b'
            channel.close()
            with pytest.raises(ChannelClosed):
                channel.put('c')
        ''',
        '''
        import threading


        def _start(call):
            out = []

            def run():
                try:
                    out.append(call())
                except Exception as error:
                    out.append(error)

            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            return thread, out


        def _blocked(*threads):
            for thread in threads:
                thread.join(0.1)
            assert all(thread.is_alive() for thread in threads)


        def test_close_drains_remaining_items():
            channel = Channel(2)
            channel.put('a', timeout=1)
            channel.put('b', timeout=1)
            channel.close()
            channel.close()
            with pytest.raises(ChannelClosed):
                channel.put('c', timeout=1)
            assert channel.take(timeout=1) == 'a'
            assert channel.take(timeout=1) == 'b'
            with pytest.raises(ChannelClosed):
                channel.take(timeout=1)


        def test_timeouts_raise_and_keep_the_item_out():
            channel = Channel(1)
            with pytest.raises(TimeoutError):
                channel.take(timeout=0.05)
            channel.put('a', timeout=1)
            with pytest.raises(TimeoutError):
                channel.put('b', timeout=0.05)
            assert channel.take(timeout=1) == 'a'
            with pytest.raises(TimeoutError):
                channel.take(timeout=0.05)


        def test_put_wakes_blocked_taker():
            channel = Channel(1)
            taker, out = _start(channel.take)
            _blocked(taker)
            channel.put('x', timeout=1)
            taker.join(0.5)
            assert out == ['x']


        def test_close_releases_blocked_putter():
            channel = Channel(1)
            channel.put('a', timeout=1)
            putter, out = _start(lambda: channel.put('b', timeout=5))
            _blocked(putter)
            channel.close()
            putter.join(0.5)
            assert not putter.is_alive() and isinstance(out[0], ChannelClosed)
            assert channel.take(timeout=1) == 'a'
            with pytest.raises(ChannelClosed):
                channel.take(timeout=1)


        def test_taker_that_finds_no_item_waits_again():
            channel = Channel(1)
            first, out_a = _start(channel.take)
            second, out_b = _start(channel.take)
            _blocked(first, second)
            channel.put('x', timeout=1)
            first.join(0.3)
            second.join(0.3)
            assert first.is_alive() != second.is_alive()
            waiting = first if first.is_alive() else second
            channel.put('y', timeout=1)
            waiting.join(0.5)
            assert not waiting.is_alive()
            assert sorted(out_a + out_b) == ['x', 'y']
        ''',
    ),
    _task(
        "worker_pool", "worker_pool",
        "WorkerPool(workers) 用 workers 个线程执行提交的任务，会被多个线程使用。要求：\n"
        "1）submit(fn, *args) 把任务加入队列，返回票据（从 0 起递增）；任务按提交顺序执行。\n"
        "2）result(ticket, timeout=None) 阻塞到该任务完成，返回其返回值；任务抛出异常时 result "
        "抛出同一个异常（类型和参数不变），只影响该票据；等待超过 timeout 秒抛 TimeoutError。\n"
        "3）timeout 为 None 表示一直等待；被其他任务的完成唤醒而自己的结果未就绪时继续等待，不能出错。\n"
        "4）stop(wait=True)：之后 submit 抛 PoolStopped；已排队的任务仍被执行完，"
        "stop 在它们全部完成后才返回。\n"
        "5）stop(wait=False)：队列中尚未开始的任务被丢弃，其 result 抛 PoolStopped，"
        "正在等待它的线程立即被唤醒；正在执行的任务照常完成。\n"
        "6）stop 可多次调用，效果相同。\n",
        '''
        import threading
        from collections import deque


        class PoolStopped(Exception):
            pass


        class WorkerPool:
            def __init__(self, workers):
                self._queue = deque()
                self._results = {}
                self._next = 0
                self._stopped = False
                self._cond = threading.Condition()
                self._workers = [threading.Thread(target=self._work, daemon=True) for _ in range(workers)]
                for worker in self._workers:
                    worker.start()

            def submit(self, fn, *args):
                with self._cond:
                    ticket = self._next
                    self._next += 1
                    self._queue.append((ticket, fn, args))
                    self._cond.notify()
                    return ticket

            def result(self, ticket, timeout=None):
                with self._cond:
                    if ticket not in self._results:
                        self._cond.wait(timeout)
                    value = self._results[ticket]
                    if isinstance(value, Exception):
                        raise value
                    return value

            def _work(self):
                while True:
                    with self._cond:
                        while not self._queue and not self._stopped:
                            self._cond.wait()
                        if self._stopped:
                            return
                        ticket, fn, args = self._queue.popleft()
                    value = fn(*args)
                    with self._cond:
                        self._results[ticket] = value
                        self._cond.notify_all()

            def stop(self, wait=True):
                with self._cond:
                    self._stopped = True
                    if wait:
                        self._cond.notify_all()
                    else:
                        while self._queue:
                            self._queue.pop()
                if wait:
                    for worker in self._workers:
                        worker.join()
        ''',
        '''
        import threading
        from collections import deque


        class PoolStopped(Exception):
            pass


        class WorkerPool:
            def __init__(self, workers):
                self._queue = deque()
                self._results = {}
                self._next = 0
                self._stopped = False
                self._cond = threading.Condition()
                self._workers = [threading.Thread(target=self._work, daemon=True) for _ in range(workers)]
                for worker in self._workers:
                    worker.start()

            def submit(self, fn, *args):
                with self._cond:
                    if self._stopped:
                        raise PoolStopped('pool is stopped')
                    ticket = self._next
                    self._next += 1
                    self._queue.append((ticket, fn, args))
                    self._cond.notify()
                    return ticket

            def result(self, ticket, timeout=None):
                with self._cond:
                    if not self._cond.wait_for(lambda: ticket in self._results, timeout):
                        raise TimeoutError('result timed out')
                    value = self._results[ticket]
                    if isinstance(value, Exception):
                        raise value
                    return value

            def _work(self):
                while True:
                    with self._cond:
                        while not self._queue and not self._stopped:
                            self._cond.wait()
                        if not self._queue:
                            return
                        ticket, fn, args = self._queue.popleft()
                    try:
                        value = fn(*args)
                    except Exception as error:
                        value = error
                    with self._cond:
                        self._results[ticket] = value
                        self._cond.notify_all()

            def stop(self, wait=True):
                with self._cond:
                    self._stopped = True
                    if not wait:
                        while self._queue:
                            ticket = self._queue.popleft()[0]
                            self._results[ticket] = PoolStopped('task was dropped')
                    self._cond.notify_all()
                if wait:
                    for worker in self._workers:
                        worker.join()
        ''',
        '''
        def test_submit_and_result():
            pool = WorkerPool(2)
            first = pool.submit(abs, -3)
            second = pool.submit(str.upper, 'x')
            assert pool.result(first, timeout=1) == 3
            assert pool.result(second, timeout=1) == 'X'
            pool.stop()
        ''',
        '''
        import threading


        def _start(call):
            out = []

            def run():
                try:
                    out.append(call())
                except Exception as error:
                    out.append(error)

            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            return thread, out


        def _stopped(pool, wait=True):
            thread, out = _start(lambda: pool.stop(wait))
            thread.join(1)
            assert not thread.is_alive()


        def _task(calls, name, started=None, *gates):
            calls.append(name)
            if started is not None:
                started.set()
            for gate in gates:
                gate.wait(5)
            if name == 'bad':
                raise ValueError('boom')
            return name


        def test_stop_drains_the_queue_and_each_result_stays_with_its_ticket():
            pool = WorkerPool(1)
            calls, started, gate = [], threading.Event(), threading.Event()
            pool.submit(_task, calls, 'a', started, gate)
            pool.submit(_task, calls, 'bad')
            third = pool.submit(_task, calls, 'c')
            assert started.wait(1)
            stopper, out = _start(pool.stop)
            stopper.join(0.2)
            assert stopper.is_alive()
            gate.set()
            stopper.join(1)
            assert not stopper.is_alive()
            assert pool.result(0, timeout=1) == 'a'
            with pytest.raises(ValueError) as info:
                pool.result(1, timeout=1)
            assert str(info.value) == 'boom' and pool.result(third, timeout=1) == 'c'
            assert calls == ['a', 'bad', 'c']
            with pytest.raises(PoolStopped):
                pool.submit(_task, calls, 'z')


        def test_stop_without_wait_drops_queued_tasks():
            pool = WorkerPool(1)
            calls, started, gate = [], threading.Event(), threading.Event()
            first = pool.submit(_task, calls, 'a', started, gate)
            second = pool.submit(_task, calls, 'b')
            assert started.wait(1)
            waiter, out = _start(lambda: pool.result(second))
            waiter.join(0.1)
            assert waiter.is_alive()
            _stopped(pool, wait=False)
            waiter.join(0.5)
            assert not waiter.is_alive() and isinstance(out[0], PoolStopped)
            gate.set()
            assert pool.result(first, timeout=1) == 'a' and calls == ['a']


        def test_waiter_keeps_waiting_for_its_own_ticket():
            pool = WorkerPool(1)
            calls, started, gates = [], threading.Event(), [threading.Event(), threading.Event()]
            pool.submit(_task, calls, 'a', started, gates[0])
            second = pool.submit(_task, calls, 'b', None, gates[1])
            assert started.wait(1)
            waiter, out = _start(lambda: pool.result(second))
            waiter.join(0.1)
            assert waiter.is_alive()
            gates[0].set()
            waiter.join(0.2)
            assert waiter.is_alive()
            gates[1].set()
            waiter.join(0.5)
            assert out == ['b']
            _stopped(pool)
        ''',
    ),
    _task(
        "worker_pool_take", "worker_pool",
        "WorkerPool(workers) 用 workers 个线程执行提交的任务，take 按提交顺序交付结果。要求：\n"
        "1）submit(fn, *args) 把任务排入队列并返回票据（从 0 起递增），任务按提交顺序开始执行。\n"
        "2）take(timeout=None) 返回 (ticket, fn 的返回值)：总是交付尚未交付的结果中票据最小的那个，"
        "即使更大票据的任务先完成也要等它；每个结果只交付一次，"
        "多个线程同时 take 时各拿到不同的结果。\n"
        "3）任务抛出异常时，对应该票据的 take 抛出同一个异常（类型和参数不变），"
        "该结果视为已交付，之后的票据照常交付。\n"
        "4）等待超过 timeout 秒抛 TimeoutError；timeout 为 None 表示一直等待；"
        "被别的结果唤醒而轮到的结果未就绪时继续等待，不能出错。\n"
        "5）stop()：之后 submit 抛 PoolStopped；已排队的任务仍按顺序执行完，stop 在它们全部完成后才返回。\n",
        '''
        import threading
        from collections import deque


        class PoolStopped(Exception):
            pass


        class WorkerPool:
            def __init__(self, workers):
                self._queue = deque()
                self._results = {}
                self._next = 0
                self._taken = 0
                self._stopped = False
                self._cond = threading.Condition()
                self._workers = [threading.Thread(target=self._work, daemon=True) for _ in range(workers)]
                for worker in self._workers:
                    worker.start()

            def submit(self, fn, *args):
                with self._cond:
                    if self._stopped:
                        raise PoolStopped('pool is stopped')
                    self._queue.append((self._next, fn, args))
                    self._next += 1
                    self._cond.notify()
                    return self._next - 1

            def take(self, timeout=None):
                ticket = self._taken
                with self._cond:
                    if ticket not in self._results:
                        self._cond.wait(timeout)
                    self._taken += 1
                    value = self._results[ticket]
                    if isinstance(value, Exception):
                        raise value
                    return ticket, value

            def _work(self):
                while True:
                    with self._cond:
                        while not self._queue and not self._stopped:
                            self._cond.wait()
                        if not self._queue:
                            return
                        ticket, fn, args = self._queue.pop()
                    try:
                        value = fn(*args)
                    except Exception:
                        value = RuntimeError('task failed')
                    with self._cond:
                        self._results[ticket] = value
                        self._cond.notify_all()

            def stop(self):
                with self._cond:
                    self._stopped = True
                    self._cond.notify_all()
        ''',
        '''
        import threading
        import time
        from collections import deque


        class PoolStopped(Exception):
            pass


        class WorkerPool:
            def __init__(self, workers):
                self._queue = deque()
                self._results = {}
                self._next = 0
                self._taken = 0
                self._stopped = False
                self._cond = threading.Condition()
                self._workers = [threading.Thread(target=self._work, daemon=True) for _ in range(workers)]
                for worker in self._workers:
                    worker.start()

            def submit(self, fn, *args):
                with self._cond:
                    if self._stopped:
                        raise PoolStopped('pool is stopped')
                    self._queue.append((self._next, fn, args))
                    self._next += 1
                    self._cond.notify()
                    return self._next - 1

            def take(self, timeout=None):
                with self._cond:
                    deadline = None if timeout is None else time.monotonic() + timeout
                    while self._taken not in self._results:
                        left = None if deadline is None else deadline - time.monotonic()
                        if left is not None and left <= 0:
                            raise TimeoutError('no result yet')
                        self._cond.wait(left)
                    value = self._results.pop(self._taken)
                    self._taken += 1
                    if isinstance(value, Exception):
                        raise value
                    return self._taken - 1, value

            def _work(self):
                while True:
                    with self._cond:
                        while not self._queue and not self._stopped:
                            self._cond.wait()
                        if not self._queue:
                            return
                        ticket, fn, args = self._queue.popleft()
                    try:
                        value = fn(*args)
                    except Exception as error:
                        value = error
                    with self._cond:
                        self._results[ticket] = value
                        self._cond.notify_all()

            def stop(self):
                with self._cond:
                    self._stopped = True
                    self._cond.notify_all()
                for worker in self._workers:
                    worker.join()
        ''',
        '''
        def test_submit_and_take():
            pool = WorkerPool(2)
            first = pool.submit(abs, -3)
            second = pool.submit(str.upper, 'x')
            assert pool.take(timeout=1) == (first, 3)
            assert pool.take(timeout=1) == (second, 'X')
            pool.stop()
        ''',
        '''
        import threading


        def _start(call):
            out = []

            def run():
                try:
                    out.append(call())
                except Exception as error:
                    out.append(error)

            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            return thread, out


        def _task(calls, name, started=None, *gates):
            calls.append(name)
            if started is not None:
                started.set()
            for gate in gates:
                gate.wait(5)
            if name == 'bad':
                raise ValueError('boom')
            return name


        def test_stop_drains_in_order_and_keeps_each_error_with_its_ticket():
            pool = WorkerPool(1)
            calls, started, gate = [], threading.Event(), threading.Event()
            pool.submit(_task, calls, 'a', started, gate)
            pool.submit(_task, calls, 'bad')
            pool.submit(_task, calls, 'c')
            assert started.wait(1)
            stopper, out = _start(pool.stop)
            stopper.join(0.2)
            assert stopper.is_alive()
            gate.set()
            stopper.join(1)
            assert not stopper.is_alive()
            assert pool.take(timeout=1) == (0, 'a')
            with pytest.raises(ValueError) as info:
                pool.take(timeout=1)
            assert str(info.value) == 'boom'
            assert pool.take(timeout=1) == (2, 'c') and calls == ['a', 'bad', 'c']
            with pytest.raises(PoolStopped):
                pool.submit(_task, calls, 'z')


        def test_each_taker_gets_a_different_result_in_order():
            pool = WorkerPool(2)
            first, out_a = _start(pool.take)
            second, out_b = _start(pool.take)
            first.join(0.1)
            second.join(0.1)
            assert first.is_alive() and second.is_alive()
            pool.submit(_task, [], 'a')
            pool.submit(_task, [], 'b')
            first.join(0.5)
            second.join(0.5)
            assert not first.is_alive() and not second.is_alive()
            assert sorted(out_a + out_b) == [(0, 'a'), (1, 'b')]


        def test_taker_waits_for_its_turn_even_if_a_later_task_finishes_first():
            pool = WorkerPool(2)
            with pytest.raises(TimeoutError):
                pool.take(timeout=0.05)
            started = [threading.Event() for _ in range(2)]
            gates = [threading.Event() for _ in range(2)]
            pool.submit(_task, [], 'a', started[0], gates[0])
            pool.submit(_task, [], 'b', started[1], gates[1])
            assert started[0].wait(1) and started[1].wait(1)
            taker, out = _start(pool.take)
            taker.join(0.1)
            assert taker.is_alive()
            gates[1].set()
            taker.join(0.2)
            assert taker.is_alive()
            gates[0].set()
            taker.join(0.5)
            assert out == [(0, 'a')] and pool.take(timeout=1) == (1, 'b')
        ''',
    ),
    _task(
        "delay_scheduler", "delay_scheduler",
        "Scheduler(clock) 管理延时任务，clock() 返回当前时间（秒，可注入）；各方法可从任意线程调用。要求：\n"
        "1）schedule(delay, fn)：delay < 0 抛 ValueError；到期时刻为登记时的 clock() + delay；"
        "返回票据（整数，从 0 起按登记顺序递增）。\n"
        "2）cancel(token)：取消尚未执行的任务返回 True；票据未知、已取消或已被 run_due 取走返回 False。\n"
        "3）run_due()：执行所有到期时刻 ≤ clock() 的任务，每个一次，按（到期时刻，登记顺序）排序；"
        "返回按执行顺序排列的各任务返回值。\n"
        "4）任务抛出异常时，该异常对象放在返回列表的对应位置，其余到期任务照常执行。\n"
        "5）run_due 先取走全部到期任务再逐个执行：任务内调用 schedule/cancel 不能死锁；"
        "其中新登记的任务不在同一轮执行；对同一轮尚未执行的任务 cancel 返回 False，该任务照常执行。\n",
        '''
        import itertools
        import threading


        class Scheduler:
            def __init__(self, clock):
                self._clock = clock
                self._tasks = {}
                self._tokens = itertools.count()
                self._lock = threading.Lock()

            def schedule(self, delay, fn):
                if delay < 0:
                    raise ValueError('delay must not be negative')
                with self._lock:
                    token = next(self._tokens)
                    self._tasks[token] = (delay, token, fn)
                    return token

            def cancel(self, token):
                with self._lock:
                    del self._tasks[token]
                    return True

            def run_due(self):
                results = []
                with self._lock:
                    now = self._clock()
                    for due, token, fn in sorted(self._tasks.values()):
                        if due <= now:
                            del self._tasks[token]
                            results.append(fn())
                return results
        ''',
        '''
        import itertools
        import threading


        class Scheduler:
            def __init__(self, clock):
                self._clock = clock
                self._tasks = {}
                self._tokens = itertools.count()
                self._lock = threading.Lock()

            def schedule(self, delay, fn):
                if delay < 0:
                    raise ValueError('delay must not be negative')
                with self._lock:
                    token = next(self._tokens)
                    self._tasks[token] = (self._clock() + delay, token, fn)
                    return token

            def cancel(self, token):
                with self._lock:
                    return self._tasks.pop(token, None) is not None

            def run_due(self):
                with self._lock:
                    now = self._clock()
                    due = sorted(task for task in self._tasks.values() if task[0] <= now)
                    for _, token, _ in due:
                        del self._tasks[token]
                results = []
                for _, _, fn in due:
                    try:
                        results.append(fn())
                    except Exception as error:
                        results.append(error)
                return results
        ''',
        '''
        def test_schedule_run_and_cancel():
            now = [0]
            scheduler = Scheduler(lambda: now[0])
            scheduler.schedule(5, lambda: 'later')
            skipped = scheduler.schedule(5, lambda: 'skipped')
            assert scheduler.run_due() == []
            now[0] = 5
            assert scheduler.cancel(skipped) is True
            assert scheduler.run_due() == ['later']
            assert scheduler.run_due() == []
        ''',
        '''
        import threading


        def _start(call):
            out = []

            def run():
                try:
                    out.append(call())
                except Exception as error:
                    out.append(error)

            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            return thread, out


        def test_due_tasks_run_once_in_time_then_registration_order():
            now = [0]
            scheduler = Scheduler(lambda: now[0])
            scheduler.schedule(10, lambda: 'a')
            scheduler.schedule(5, lambda: 'b')
            scheduler.schedule(6, lambda: 'c')
            scheduler.schedule(10, lambda: 'd')
            now[0] = 3
            scheduler.schedule(5, lambda: 'e')
            assert scheduler.run_due() == []
            now[0] = 5
            assert scheduler.run_due() == ['b']
            now[0] = 10
            assert scheduler.run_due() == ['c', 'e', 'a', 'd']
            assert scheduler.run_due() == []


        def test_cancel_only_removes_pending_tasks():
            now = [0]
            scheduler = Scheduler(lambda: now[0])
            token = scheduler.schedule(1, lambda: 'a')
            assert scheduler.cancel(token) is True
            assert scheduler.cancel(token) is False
            now[0] = 1
            assert scheduler.run_due() == []
            with pytest.raises(ValueError):
                scheduler.schedule(-1, int)


        def test_task_may_call_back_into_the_scheduler():
            now = [0]
            scheduler = Scheduler(lambda: now[0])
            later = scheduler.schedule(1, lambda: 'later')
            state = []

            def first():
                state.append(scheduler.cancel(later))
                state.append(scheduler.schedule(0, lambda: 'new'))
                return 'first'

            scheduler.schedule(0, first)
            now[0] = 1
            thread, out = _start(scheduler.run_due)
            thread.join(0.5)
            assert not thread.is_alive()
            assert out == [['first', 'later']] and state == [False, 2]
            assert scheduler.run_due() == ['new']


        def test_failing_task_keeps_its_slot_and_lets_the_others_run():
            def boom():
                raise ValueError('boom')

            scheduler = Scheduler(lambda: 0)
            scheduler.schedule(0, boom)
            scheduler.schedule(0, lambda: 'b')
            scheduler.schedule(1, lambda: 'c')
            thread, out = _start(scheduler.run_due)
            thread.join(0.5)
            assert not thread.is_alive()
            assert isinstance(out[0][0], ValueError) and out[0][1:] == ['b']
            assert scheduler.run_due() == []
        ''',
    ),
    _task(
        "delay_scheduler_wake", "delay_scheduler",
        "Scheduler(clock) 管理延时任务，clock() 返回当前时间（秒，可注入）；各方法可从任意线程调用。要求：\n"
        "1）schedule(delay, fn)：delay < 0 抛 ValueError；到期时刻为登记时的 clock() + delay；"
        "返回票据（整数，从 0 起按登记顺序递增）。\n"
        "2）cancel(token)：取消尚未执行的任务返回 True；票据未知、已取消或已被取走返回 False。\n"
        "3）next_due(timeout=None)：取走并执行到期时刻 ≤ clock() 的任务中（到期时刻，登记顺序）最小的一个，"
        "返回其返回值。\n"
        "4）没有到期任务时阻塞；等待超过 timeout 秒抛 TimeoutError；timeout 为 None 表示一直等待。\n"
        "5）每次被唤醒后重新检查：schedule 登记后会唤醒等待中的线程；多个线程等待时一个任务只交给其中一个，"
        "其余继续等待，不抛异常。\n"
        "6）任务在被取走时即从待执行集合移除，只执行一次。\n",
        '''
        import itertools
        import threading


        class Scheduler:
            def __init__(self, clock):
                self._clock = clock
                self._tasks = {}
                self._tokens = itertools.count()
                self._cond = threading.Condition()

            def schedule(self, delay, fn):
                if delay < 0:
                    raise ValueError('delay must not be negative')
                with self._cond:
                    token = next(self._tokens)
                    wake = not self._tasks
                    self._tasks[token] = (self._clock() + delay, token, fn)
                    if wake:
                        self._cond.notify_all()
                    return token

            def cancel(self, token):
                with self._cond:
                    return self._tasks.pop(token, None) is not None

            def next_due(self, timeout=None):
                with self._cond:
                    task = next(iter(self._tasks.values()), None)
                    if task is None:
                        self._cond.wait(timeout)
                        task = next(iter(self._tasks.values()), None)
                    if task is None:
                        raise TimeoutError('nothing is due')
                value = task[2]()
                with self._cond:
                    del self._tasks[task[1]]
                return value
        ''',
        '''
        import itertools
        import threading
        import time


        class Scheduler:
            def __init__(self, clock):
                self._clock = clock
                self._tasks = {}
                self._tokens = itertools.count()
                self._cond = threading.Condition()

            def schedule(self, delay, fn):
                if delay < 0:
                    raise ValueError('delay must not be negative')
                with self._cond:
                    token = next(self._tokens)
                    self._tasks[token] = (self._clock() + delay, token, fn)
                    self._cond.notify_all()
                    return token

            def cancel(self, token):
                with self._cond:
                    return self._tasks.pop(token, None) is not None

            def _due(self):
                return min((task for task in self._tasks.values() if task[0] <= self._clock()),
                           default=None)

            def next_due(self, timeout=None):
                with self._cond:
                    deadline = None if timeout is None else time.monotonic() + timeout
                    while (task := self._due()) is None:
                        left = None if deadline is None else deadline - time.monotonic()
                        if left is not None and left <= 0:
                            raise TimeoutError('nothing is due')
                        self._cond.wait(left)
                    del self._tasks[task[1]]
                return task[2]()
        ''',
        '''
        def test_schedule_and_next_due():
            scheduler = Scheduler(lambda: 0)
            skipped = scheduler.schedule(0, lambda: 'skipped')
            assert scheduler.cancel(skipped) is True
            scheduler.schedule(0, lambda: 'done')
            assert scheduler.next_due(timeout=1) == 'done'
        ''',
        '''
        import threading


        def _start(call):
            out = []

            def run():
                try:
                    out.append(call())
                except Exception as error:
                    out.append(error)

            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            return thread, out


        def test_next_due_runs_only_due_tasks_earliest_first():
            now = [0]
            scheduler = Scheduler(lambda: now[0])
            scheduler.schedule(10, lambda: 'a')
            skipped = scheduler.schedule(3, lambda: 'x')
            with pytest.raises(TimeoutError):
                scheduler.next_due(timeout=0.05)
            now[0] = 5
            scheduler.schedule(0, lambda: 'b')
            assert scheduler.cancel(skipped) is True and scheduler.cancel(skipped) is False
            now[0] = 10
            assert scheduler.next_due(timeout=1) == 'b'
            assert scheduler.next_due(timeout=1) == 'a'
            with pytest.raises(TimeoutError):
                scheduler.next_due(timeout=0.05)


        def test_dispatchers_wake_and_one_result_goes_to_one_of_them():
            now = [0]
            scheduler = Scheduler(lambda: now[0])
            scheduler.schedule(60, lambda: 'later')
            first, out_a = _start(scheduler.next_due)
            second, out_b = _start(scheduler.next_due)
            first.join(0.1)
            second.join(0.1)
            assert first.is_alive() and second.is_alive()
            scheduler.schedule(0, lambda: 'x')
            first.join(0.3)
            second.join(0.3)
            assert first.is_alive() != second.is_alive()
            waiting = first if first.is_alive() else second
            scheduler.schedule(0, lambda: 'y')
            waiting.join(0.5)
            assert not waiting.is_alive()
            assert sorted(out_a + out_b) == ['x', 'y']


        def test_a_task_taken_by_one_dispatcher_is_not_run_again():
            now = [0]
            scheduler = Scheduler(lambda: now[0])
            calls, started, gate = [], threading.Event(), threading.Event()

            def slow():
                calls.append('slow')
                started.set()
                gate.wait(5)
                return 'slow'

            token = scheduler.schedule(0, slow)
            first, out_a = _start(scheduler.next_due)
            assert started.wait(1)
            probe, out_c = _start(lambda: scheduler.cancel(token))
            probe.join(0.3)
            assert out_c == [False]
            now[0] = 1
            scheduler.schedule(0, lambda: 'next')
            second, out_b = _start(scheduler.next_due)
            second.join(0.5)
            assert out_b == ['next']
            gate.set()
            first.join(0.5)
            assert out_a == ['slow'] and calls == ['slow']
        ''',
    ),
    _task(
        "pipeline_stage", "pipeline_stage",
        "Pipeline(stage, capacity) 用一个后台线程对 push 进来的 item 依次调用 stage(item)，"
        "结果由 take 取走。要求：\n"
        "1）push(item, timeout=None)：未处理完的 item（含正在处理的）达到 capacity 个时阻塞；"
        "等待超过 timeout 秒抛 TimeoutError，item 不放入。\n"
        "2）take(timeout=None) 按 push 顺序返回 stage 的返回值，没有结果时阻塞；等待超时抛 TimeoutError。\n"
        "3）flush(timeout=None) 阻塞到所有已 push 的 item 都处理完才返回；等待超时抛 TimeoutError。\n"
        "4）stage 对某个 item 抛出异常时流水线停止：之后的 item 不再处理；flush 抛出该异常本身"
        "（类型和参数不变）；take 先返回它之前 item 的结果，取完后 take 抛出该异常；之后的 push 也抛出该异常。\n"
        "5）正因容量不足而阻塞的 push，在流水线停止时立即被唤醒并抛出该异常。\n"
        "6）timeout 为 None 表示一直等待。\n",
        '''
        import threading
        import time
        from collections import deque


        class Pipeline:
            def __init__(self, stage, capacity):
                self._stage, self._capacity = stage, capacity
                self._inbox, self._outbox, self._error = deque(), deque(), None
                self._cond = threading.Condition()
                threading.Thread(target=self._run, daemon=True).start()

            def _await(self, ready, timeout):
                deadline = None if timeout is None else time.monotonic() + timeout
                while not ready():
                    left = None if deadline is None else deadline - time.monotonic()
                    if left is not None and left <= 0:
                        return False
                    self._cond.wait(left)
                return True

            def push(self, item, timeout=None):
                with self._cond:
                    self._await(lambda: len(self._inbox) < self._capacity, timeout)
                    self._inbox.append(item)
                    self._cond.notify_all()

            def take(self, timeout=None):
                with self._cond:
                    if self._error is not None:
                        raise self._error
                    if not self._await(lambda: self._outbox, timeout):
                        raise TimeoutError('timed out')
                    return self._outbox.popleft()

            def flush(self, timeout=None):
                with self._cond:
                    if not self._await(lambda: not self._inbox, timeout):
                        raise TimeoutError('timed out')
                    if self._error is not None:
                        raise RuntimeError('stage failed')

            def _run(self):
                while True:
                    with self._cond:
                        self._await(lambda: self._inbox, None)
                        item = self._inbox.popleft()
                    try:
                        out = self._stage(item)
                    except Exception:
                        with self._cond:
                            self._error = RuntimeError('stage failed')
                            self._inbox.clear()
                            self._cond.notify_all()
                        return
                    with self._cond:
                        self._outbox.append(out)
                        self._cond.notify_all()
        ''',
        '''
        import threading
        import time
        from collections import deque


        class Pipeline:
            def __init__(self, stage, capacity):
                self._stage, self._capacity = stage, capacity
                self._inbox, self._outbox, self._error = deque(), deque(), None
                self._cond = threading.Condition()
                threading.Thread(target=self._run, daemon=True).start()

            def _await(self, ready, timeout):
                deadline = None if timeout is None else time.monotonic() + timeout
                while not ready():
                    left = None if deadline is None else deadline - time.monotonic()
                    if left is not None and left <= 0:
                        raise TimeoutError('timed out')
                    self._cond.wait(left)

            def push(self, item, timeout=None):
                with self._cond:
                    self._await(lambda: self._error is not None or len(self._inbox) < self._capacity, timeout)
                    if self._error is not None:
                        raise self._error
                    self._inbox.append(item)
                    self._cond.notify_all()

            def take(self, timeout=None):
                with self._cond:
                    self._await(lambda: self._error is not None or self._outbox, timeout)
                    if self._outbox:
                        return self._outbox.popleft()
                    raise self._error

            def flush(self, timeout=None):
                with self._cond:
                    self._await(lambda: self._error is not None or not self._inbox, timeout)
                    if self._error is not None:
                        raise self._error

            def _run(self):
                while True:
                    with self._cond:
                        self._await(lambda: self._inbox, None)
                        item = self._inbox[0]
                    try:
                        out = self._stage(item)
                    except Exception as error:
                        with self._cond:
                            self._error = error
                            self._inbox.clear()
                            self._cond.notify_all()
                        return
                    with self._cond:
                        self._inbox.popleft()
                        self._outbox.append(out)
                        self._cond.notify_all()
        ''',
        '''
        def test_push_take_flush():
            pipeline = Pipeline(str.upper, 2)
            pipeline.push('a')
            pipeline.push('b')
            pipeline.flush(timeout=1)
            assert pipeline.take(timeout=1) == 'A'
            assert pipeline.take(timeout=1) == 'B'
        ''',
        '''
        import threading


        def _start(call):
            out = []

            def run():
                try:
                    out.append(call())
                except Exception as error:
                    out.append(error)

            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            return thread, out


        def _stage(log, name, started=None, gate=None):
            log.append(name)
            if started is not None:
                started.set()
            if gate is not None:
                gate.wait(5)
            if name == 'bad':
                raise ValueError('boom')
            return name.upper()


        def _pipeline(log, capacity=1, **hooks):
            return Pipeline(lambda item: _stage(log, item, **hooks), capacity)


        def test_push_waits_for_room_and_times_out_without_adding():
            log, started, gate = [], threading.Event(), threading.Event()
            pipeline = _pipeline(log, 1, started=started, gate=gate)
            pipeline.push('a', timeout=1)
            assert started.wait(1)
            with pytest.raises(TimeoutError):
                pipeline.push('b', timeout=0.05)
            gate.set()
            assert pipeline.take(timeout=1) == 'A'
            pipeline.flush(timeout=1)
            assert log == ['a']


        def test_flush_returns_only_after_the_item_left_the_stage():
            log, started, gate = [], threading.Event(), threading.Event()
            pipeline = _pipeline(log, started=started, gate=gate)
            pipeline.push('a', timeout=1)
            assert started.wait(1)
            flusher, out = _start(pipeline.flush)
            flusher.join(0.2)
            assert flusher.is_alive()
            gate.set()
            flusher.join(0.5)
            assert not flusher.is_alive() and pipeline.take(timeout=1) == 'A'


        def test_stage_failure_keeps_earlier_results_and_surfaces_the_original_error():
            log, gate = [], threading.Event()
            pipeline = _pipeline(log, 3, gate=gate)
            for item in ('a', 'bad', 'c'):
                pipeline.push(item, timeout=1)
            gate.set()
            with pytest.raises(ValueError) as info:
                pipeline.flush(timeout=1)
            assert str(info.value) == 'boom'
            assert pipeline.take(timeout=1) == 'A'
            with pytest.raises(ValueError):
                pipeline.take(timeout=1)
            with pytest.raises(ValueError):
                pipeline.push('d', timeout=1)
            with pytest.raises(ValueError):
                pipeline.take(timeout=0.05)


        def test_failure_releases_a_blocked_push_with_the_stage_error():
            log, started, gate = [], threading.Event(), threading.Event()
            pipeline = _pipeline(log, 1, started=started, gate=gate)
            pipeline.push('bad', timeout=1)
            assert started.wait(1)
            pusher, out = _start(lambda: pipeline.push('b', timeout=5))
            pusher.join(0.1)
            assert pusher.is_alive()
            gate.set()
            pusher.join(0.5)
            assert not pusher.is_alive() and isinstance(out[0], ValueError)
            with pytest.raises(ValueError):
                pipeline.flush(timeout=1)
        ''',
    ),
    _task(
        "pipeline_stage_stop", "pipeline_stage",
        "Pipeline(stage, capacity) 用一个后台线程对 push 进来的 item 依次调用 stage(item)，"
        "结果由 take 取走。要求：\n"
        "1）push(item, timeout=None)：未处理完的 item（含正在处理的）达到 capacity 个时阻塞；"
        "等待超过 timeout 秒抛 TimeoutError，item 不放入。\n"
        "2）take(timeout=None) 按 push 顺序返回 stage 的返回值，没有结果时阻塞；等待超时抛 TimeoutError；"
        "timeout 为 None 表示一直等待。\n"
        "3）flush(timeout=None) 阻塞到所有已 push 的 item 都处理完才返回；等待超时抛 TimeoutError。\n"
        "4）stage 对某个 item 抛出异常时流水线停止：之后的 item 不再处理，之后的 push 抛出该异常；"
        "flush 抛出该异常本身（类型和参数不变）；take 先返回它之前 item 的结果，"
        "取完后 take 抛出该异常，且一直保持失败状态。\n"
        "5）多个线程同时 take 时，一个结果只交给其中一个，其余的继续等待，不抛异常。\n",
        '''
        import threading
        from collections import deque


        class Pipeline:
            def __init__(self, stage, capacity):
                self._stage, self._capacity = stage, capacity
                self._inbox, self._outbox = deque(), deque()
                self._cond = threading.Condition()
                threading.Thread(target=self._run, daemon=True).start()

            def push(self, item, timeout=None):
                with self._cond:
                    if len(self._inbox) >= self._capacity:
                        if not self._cond.wait(timeout):
                            raise TimeoutError('push timed out')
                    self._inbox.append(item)
                    self._cond.notify_all()

            def take(self, timeout=None):
                with self._cond:
                    if not self._outbox:
                        self._cond.wait(timeout)
                    out = self._outbox.pop()
                    if isinstance(out, Exception):
                        raise out
                    return out

            def flush(self, timeout=None):
                with self._cond:
                    if self._inbox:
                        self._cond.wait(timeout)

            def _run(self):
                while True:
                    with self._cond:
                        while not self._inbox:
                            self._cond.wait()
                        item = self._inbox.popleft()
                    try:
                        out = self._stage(item)
                    except Exception as error:
                        out = error
                    with self._cond:
                        self._outbox.append(out)
                        self._cond.notify_all()
        ''',
        '''
        import threading
        from collections import deque


        class Pipeline:
            def __init__(self, stage, capacity):
                self._stage, self._capacity = stage, capacity
                self._inbox, self._outbox, self._error = deque(), deque(), None
                self._cond = threading.Condition()
                threading.Thread(target=self._run, daemon=True).start()

            def push(self, item, timeout=None):
                with self._cond:
                    if not self._cond.wait_for(
                            lambda: self._error is not None or len(self._inbox) < self._capacity, timeout):
                        raise TimeoutError('push timed out')
                    if self._error is not None:
                        raise self._error
                    self._inbox.append(item)
                    self._cond.notify_all()

            def take(self, timeout=None):
                with self._cond:
                    if not self._cond.wait_for(lambda: self._error is not None or self._outbox, timeout):
                        raise TimeoutError('take timed out')
                    if self._outbox:
                        return self._outbox.popleft()
                    raise self._error

            def flush(self, timeout=None):
                with self._cond:
                    if not self._cond.wait_for(lambda: self._error is not None or not self._inbox, timeout):
                        raise TimeoutError('flush timed out')
                    if self._error is not None:
                        raise self._error

            def _run(self):
                while True:
                    with self._cond:
                        self._cond.wait_for(lambda: self._inbox)
                        item = self._inbox[0]
                    try:
                        out = self._stage(item)
                    except Exception as error:
                        with self._cond:
                            self._error = error
                            self._inbox.clear()
                            self._cond.notify_all()
                        return
                    with self._cond:
                        self._inbox.popleft()
                        self._outbox.append(out)
                        self._cond.notify_all()
        ''',
        '''
        def test_push_take_flush():
            pipeline = Pipeline(str.upper, 2)
            pipeline.push('a')
            pipeline.flush(timeout=1)
            assert pipeline.take(timeout=1) == 'A'
            pipeline.push('b')
            assert pipeline.take(timeout=1) == 'B'
        ''',
        '''
        import threading


        def _start(call):
            out = []

            def run():
                try:
                    out.append(call())
                except Exception as error:
                    out.append(error)

            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            return thread, out


        def _stage(log, name, started=None, gate=None):
            log.append(name)
            if started is not None:
                started.set()
            if gate is not None:
                gate.wait(5)
            if name == 'bad':
                raise ValueError('boom')
            return name.upper()


        def _pipeline(log, capacity=1, **hooks):
            return Pipeline(lambda item: _stage(log, item, **hooks), capacity)


        def test_take_times_out_and_results_keep_push_order():
            pipeline = Pipeline(str.upper, 3)
            with pytest.raises(TimeoutError):
                pipeline.take(timeout=0.05)
            for item in 'abc':
                pipeline.push(item, timeout=1)
            pipeline.flush(timeout=1)
            assert [pipeline.take(timeout=1) for _ in range(3)] == ['A', 'B', 'C']
            with pytest.raises(TimeoutError):
                pipeline.take(timeout=0.05)


        def test_push_blocks_while_capacity_items_are_unfinished():
            log, started, gate = [], threading.Event(), threading.Event()
            pipeline = _pipeline(log, 1, started=started, gate=gate)
            pipeline.push('a', timeout=1)
            assert started.wait(1)
            with pytest.raises(TimeoutError):
                pipeline.push('b', timeout=0.05)
            gate.set()
            assert pipeline.take(timeout=1) == 'A'
            pipeline.push('b', timeout=1)
            assert pipeline.take(timeout=1) == 'B' and log == ['a', 'b']


        def test_stage_failure_stops_the_pipeline_for_good():
            log, gate = [], threading.Event()
            pipeline = _pipeline(log, 3, gate=gate)
            for item in ('a', 'bad', 'c'):
                pipeline.push(item, timeout=1)
            gate.set()
            with pytest.raises(ValueError) as info:
                pipeline.flush(timeout=1)
            assert str(info.value) == 'boom'
            assert pipeline.take(timeout=1) == 'A'
            with pytest.raises(ValueError):
                pipeline.take(timeout=1)
            with pytest.raises(ValueError):
                pipeline.push('d', timeout=1)
            with pytest.raises(ValueError):
                pipeline.take(timeout=0.05)


        def test_consumer_that_finds_nothing_keeps_waiting():
            pipeline = Pipeline(str.upper, 3)
            first, out_a = _start(pipeline.take)
            second, out_b = _start(pipeline.take)
            first.join(0.1)
            second.join(0.1)
            assert first.is_alive() and second.is_alive()
            pipeline.push('a', timeout=1)
            first.join(0.3)
            second.join(0.3)
            assert first.is_alive() != second.is_alive()
            waiting = first if first.is_alive() else second
            pipeline.push('b', timeout=1)
            waiting.join(0.5)
            assert not waiting.is_alive()
            assert sorted(out_a + out_b) == ['A', 'B']
        ''',
    ),
]

MUTANTS = {
    "h3_pipeline_stage_stop_001": [
        ("                return self._outbox.popleft()\n", "                return self._outbox.pop()\n"),
        ("lambda: self._error is not None or len(self._inbox) < self._capacity, timeout):",
         "lambda: self._error is not None or len(self._inbox) < self._capacity + 1, timeout):"),
        ("            if not self._cond.wait_for(lambda: self._error is not None or not self._inbox, timeout):\n"
         "                raise TimeoutError('flush timed out')\n"
         "            if self._error is not None:\n                raise self._error\n",
         "            self._cond.wait_for(lambda: not self._inbox, timeout)\n"),
        ("                raise TimeoutError('push timed out')\n"
         "            if self._error is not None:\n                raise self._error\n",
         "                raise TimeoutError('push timed out')\n"),
        ("            if not self._cond.wait_for(lambda: self._error is not None or self._outbox, timeout):\n",
         "            if not (self._error is not None or self._outbox) and not self._cond.wait(timeout):\n"),
    ],
    "h3_pipeline_stage_001": [
        ("lambda: self._error is not None or len(self._inbox) < self._capacity, timeout)",
         "lambda: True, timeout)"),
        ("                    self._inbox.clear()\n                    self._cond.notify_all()\n",
         "                    self._inbox.clear()\n"),
        ("            if self._outbox:\n                return self._outbox.popleft()\n",
         "            if self._error is not None:\n                raise self._error\n"
         "            if self._outbox:\n                return self._outbox.popleft()\n"),
        ("lambda: self._error is not None or not self._inbox, timeout)",
         "lambda: self._error is not None, timeout)"),
        ("                    self._error = error\n",
         "                    self._error = RuntimeError('stage failed')\n"),
    ],
    "h3_delay_scheduler_wake_001": [
        ("        return min((task for task in self._tasks.values() if task[0] <= self._clock()),\n"
         "                   default=None)\n",
         "        return min(self._tasks.values(), default=None)\n"),
        ("            self._cond.notify_all()\n            return token\n", "            return token\n"),
        ("            while (task := self._due()) is None:\n",
         "            if (task := self._due()) is None:\n"),
        ("            del self._tasks[task[1]]\n        return task[2]()\n",
         "            value = task[2]()\n            with self._cond:\n"
         "                del self._tasks[task[1]]\n            return value\n"),
        ("            return self._tasks.pop(token, None) is not None\n",
         "            self._tasks.pop(token, None)\n            return True\n"),
    ],
    "h3_delay_scheduler_001": [
        ("            self._tasks[token] = (self._clock() + delay, token, fn)\n",
         "            self._tasks[token] = (delay, token, fn)\n"),
        ("            due = sorted(task for task in self._tasks.values() if task[0] <= now)\n",
         "            due = sorted((task for task in reversed(list(self._tasks.values())) if task[0] <= now),\n"
         "                         key=lambda task: task[0])\n"),
        ("            except Exception as error:\n                results.append(error)\n",
         "            except Exception:\n                break\n"),
        ("            return self._tasks.pop(token, None) is not None\n",
         "            del self._tasks[token]\n            return True\n"),
        ("            for _, token, _ in due:\n                del self._tasks[token]\n", "            pass\n"),
    ],
    "h3_worker_pool_take_001": [
        ("                ticket, fn, args = self._queue.popleft()\n",
         "                ticket, fn, args = self._queue.pop()\n"),
        ("            value = error\n", "            value = RuntimeError('task failed')\n"),
        ("            value = self._results.pop(self._taken)\n            self._taken += 1\n",
         "            value = self._results[self._taken]\n"),
        ("            while self._taken not in self._results:\n",
         "            if self._taken not in self._results:\n"),
        ("        for worker in self._workers:\n            worker.join()\n", ""),
    ],
    "h3_worker_pool_001": [
        ("            if self._stopped:\n                raise PoolStopped('pool is stopped')\n"
         "            ticket = self._next\n",
         "            ticket = self._next\n"),
        ("                if not self._queue:\n                    return\n",
         "                if self._stopped:\n                    return\n"),
        ("                value = error\n", "                value = RuntimeError('task failed')\n"),
        ("if not self._cond.wait_for(lambda: ticket in self._results, timeout):",
         "if ticket not in self._results and not self._cond.wait(timeout):"),
        ("                    self._results[ticket] = PoolStopped('task was dropped')\n"
         "            self._cond.notify_all()\n",
         "                    self._results[ticket] = PoolStopped('task was dropped')\n"),
    ],
    "h3_bounded_channel_drain_001": [
        ("            self._closed = True\n            self._room.notify_all()\n",
         "            self._closed = True\n            self._items.clear()\n"
         "            self._room.notify_all()\n"),
        ("                raise TimeoutError('take timed out')\n", "                return None\n"),
        ("            self._items.append(item)\n            self._data.notify_all()\n",
         "            self._items.append(item)\n            self._room.notify_all()\n"),
        ("                raise TimeoutError('put timed out')\n            self._ensure_open()\n", ""),
        ("if not self._data.wait_for(lambda: self._items or self._closed, timeout):",
         "if not self._items and not self._data.wait(timeout):"),
    ],
    "h3_bounded_channel_001": [
        ("        while not ready():\n", "        if not ready():\n"),
        ("                raise TimeoutError('put timed out')\n            self._ensure_open()\n",
         "                raise TimeoutError('put timed out')\n"),
        ("            self._closed = True\n            self._room.notify_all()\n",
         "            self._closed = True\n"),
        ("            item = self._items.pop(0)\n            self._room.notify_all()\n",
         "            item = self._items.pop(0)\n"),
        ("            self._room.notify_all()\n            self._data.notify_all()\n",
         "            self._data.notify_all()\n"),
    ],
}
