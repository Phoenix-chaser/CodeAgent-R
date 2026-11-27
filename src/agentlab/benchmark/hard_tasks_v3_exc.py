"""Exception-safety and cleanup tasks for the hard batch (h3).

Every program here cleans up or rolls back correctly on the happy path.  The
defects only show when something raises in the middle: a cleanup step that is
skipped or reordered, an original error that is masked by a later one, state
that survives a failed close, a rollback that runs at the wrong time.
"""
from __future__ import annotations

from agentlab.benchmark.hard_tasks_v3_common import _task

SPECS = [
    _task(
        "cleanup_scope", "resource_stack",
        "Scope 管理一组清理回调。push(callback) 登记回调并返回序号（从 1 开始）。要求：\n"
        "1）close(exc=None) 按登记的逆序调用尚未执行的回调，每个回调收到当时的 in-flight 异常："
        "初始为 exc，没有则为 None。\n"
        "2）回调返回 True 表示抑制 in-flight 异常：之后的回调收到 None，close 正常返回 None；"
        "没有回调抑制时 close 重新抛出同一个异常对象。\n"
        "3）回调抛异常不影响其余回调；close 结束时抛出第一个（按执行顺序）回调异常，"
        "并把当时尚未被抑制的 in-flight 异常设为它的 `__context__`（当时没有 in-flight 异常则不设置）。\n"
        "4）close 一开始 scope 即为已关闭：之后再 close（包括回调内部的重入调用）直接返回 None；"
        "再 push 抛 RuntimeError('scope is closed')。\n"
        "5）with scope: 正常退出时调用 close()；body 抛异常时用该异常执行同样的回调，"
        "只有它被回调抑制时 with 语句才不再抛出。",
        '''
        class Scope:
            def __init__(self):
                self._items = []

            def push(self, callback):
                self._items.append(callback)
                return len(self._items)

            def close(self, exc=None):
                while self._items:
                    if self._items[-1](exc):
                        return
                    self._items.pop()
                if exc is not None:
                    raise exc

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, tb):
                self.close(exc_value)
        ''',
        '''
        class Scope:
            def __init__(self):
                self._items = []
                self._closed = False

            def push(self, callback):
                if self._closed:
                    raise RuntimeError('scope is closed')
                self._items.append(callback)
                return len(self._items)

            def _drain(self, exc):
                self._closed = True
                items, self._items = self._items, []
                error = None
                for callback in reversed(items):
                    try:
                        if callback(exc):
                            exc = None
                    except Exception as caught:
                        if error is None:
                            error = caught
                            if exc is not None:
                                caught.__context__ = exc
                return error, exc

            def close(self, exc=None):
                if self._closed:
                    return None
                error, exc = self._drain(exc)
                if error is not None:
                    raise error
                if exc is not None:
                    raise exc
                return None

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, tb):
                error, exc = self._drain(exc_value)
                if error is not None:
                    raise error
                return exc is None
        ''',
        '''
        def test_callbacks_run_in_reverse_order():
            calls = []
            scope = Scope()
            scope.push(lambda exc: calls.append('first'))
            scope.push(lambda exc: calls.append('second'))
            assert scope.close() is None
            assert calls == ['second', 'first']
        ''',
        """
def _tracker(log, name, result=None):
    def callback(exc):
        log.append((name, exc))
        return result
    return callback


def _failer(error):
    def callback(exc):
        raise error
    return callback


def _scope(*callbacks):
    scope = Scope()
    for callback in callbacks:
        scope.push(callback)
    return scope


def test_callbacks_receive_the_exception_until_one_suppresses_it():
    log, exc = [], ValueError('boom')
    scope = Scope()
    for index, name in enumerate('abc', 1):
        assert scope.push(_tracker(log, name)) == index
    with pytest.raises(ValueError) as caught:
        scope.close(exc)
    assert caught.value is exc and log == [('c', exc), ('b', exc), ('a', exc)]
    assert scope.close() is None and len(log) == 3
    log, exc = [], KeyError('k')
    scope = _scope(_tracker(log, 'a'), _tracker(log, 'b', True), _tracker(log, 'c'))
    assert scope.close(exc) is None and log == [('c', exc), ('b', exc), ('a', None)]
    assert scope.close() is None and len(log) == 3


def test_failing_callback_does_not_stop_the_others_and_is_chained():
    log, exc, boom = [], ValueError('boom'), RuntimeError('cleanup')
    scope = _scope(_tracker(log, 'a'), _failer(boom), _tracker(log, 'c'))
    with pytest.raises(RuntimeError) as caught:
        scope.close(exc)
    assert caught.value is boom and boom.__context__ is exc and log == [('c', exc), ('a', exc)]
    assert scope.close() is None and len(log) == 2
    log, first, second, third = [], OSError('1'), OSError('2'), OSError('3')
    scope = _scope(_failer(third), _tracker(log, 'a', True), _failer(second), _failer(first))
    with pytest.raises(OSError) as caught:
        scope.close(exc)
    assert caught.value is first and first.__context__ is exc and log == [('a', exc)]
    log, boom, exc = [], OSError('late'), ValueError('boom')
    with pytest.raises(OSError) as caught:
        _scope(_failer(boom), _tracker(log, 'c', True)).close(exc)
    assert caught.value is boom and boom.__context__ is None and log == [('c', exc)]


def test_closed_scope_rejects_push_and_ignores_later_closes():
    log, exc, pending = [], ValueError('boom'), []
    scope = _scope(lambda e: log.append(scope.close(exc)),
                   lambda e: log.append(scope.push(pending.pop())) if pending else None)
    pending.append(_tracker(log, 'late'))
    with pytest.raises(RuntimeError) as caught:
        scope.close(exc)
    assert str(caught.value) == 'scope is closed' and log == [None]
    with pytest.raises(RuntimeError):
        scope.push(_tracker(log, 'x'))
    assert scope.close(exc) is None and log == [None]


def test_with_statement_follows_the_suppression_decision():
    log, exc, boom = [], ValueError('boom'), OSError('cleanup')
    with Scope() as scope:
        scope.push(_tracker(log, 'a'))
    assert log == [('a', None)]
    with _scope(_tracker(log, 'b', True)):
        raise exc
    assert log[-1] == ('b', exc)
    with pytest.raises(ValueError) as caught:
        with _scope(_tracker(log, 'c')):
            raise exc
    assert caught.value is exc and log[-1] == ('c', exc)
    with pytest.raises(OSError) as caught:
        with _scope(_failer(boom)):
            raise exc
    assert caught.value is boom and boom.__context__ is exc
        """,
    ),
    _task(
        "retry_step", "retry_rollback",
        "run(step, rollback, limit=3) 执行 step()，失败时按只读模块 client_errors 的 is_retryable(error) "
        "判断是否重试。要求：\n"
        "1）limit < 1 抛 ValueError('limit must be positive')，不调用 step 和 rollback。\n"
        "2）step 抛出可重试的异常且还有剩余次数：先调用 rollback()，再重新执行 step。"
        "rollback 自身抛异常则停止重试，抛出该异常，并把导致这次回滚的 step 异常设为它的 `__cause__`。\n"
        "3）异常不可重试，或已是最后一次尝试：直接抛出 step 的同一个异常对象，不调用 rollback。\n"
        "4）step 成功时返回它的返回值。",
        '''
        from client_errors import Busy, Unavailable


        def run(step, rollback, limit=3):
            for left in range(limit, 0, -1):
                try:
                    return step()
                except (Busy, Unavailable) as error:
                    rollback()
                    if left == 1:
                        raise error
        ''',
        '''
        from client_errors import is_retryable


        def run(step, rollback, limit=3):
            if limit < 1:
                raise ValueError('limit must be positive')
            for left in range(limit, 0, -1):
                try:
                    return step()
                except Exception as error:
                    if left == 1 or not is_retryable(error):
                        raise
                    try:
                        rollback()
                    except Exception as cleanup_error:
                        raise cleanup_error from error
        ''',
        '''
        import client_errors


        def test_retries_a_busy_client_until_it_succeeds():
            calls, failures = [], [client_errors.Busy('busy')] * 2

            def step():
                calls.append('step')
                if failures:
                    raise failures.pop()
                return 'ok'

            assert run(step, lambda: calls.append('rollback')) == 'ok'
            assert calls == ['step', 'rollback', 'step', 'rollback', 'step']
        ''',
        """
import client_errors as ce


def _step(log, outcomes):
    def step():
        log.append('step')
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome
    return step


def test_retries_only_retryable_errors_and_rolls_back_before_each_retry():
    log, errors = [], [ce.Busy('busy'), ce.Unavailable('later')]
    assert run(_step(log, errors + ['ok']), lambda: log.append('rollback')) == 'ok'
    assert log == ['step', 'rollback'] * 2 + ['step']


@pytest.mark.parametrize('error', [ce.Expired('x'), ce.Busy('terminal busy'), OSError('io')])
def test_non_retryable_errors_propagate_untouched_without_rollback(error):
    log = []
    with pytest.raises(type(error)) as caught:
        run(_step(log, [error]), lambda: log.append('rollback'))
    assert caught.value is error and log == ['step']


def test_last_error_is_raised_after_the_final_attempt():
    log, errors = [], [ce.Busy('1'), ce.Busy('2'), ce.Busy('3')]
    with pytest.raises(ce.Busy) as caught:
        run(_step(log, list(errors)), lambda: log.append('rollback'))
    assert caught.value is errors[-1] and log == ['step', 'rollback'] * 2 + ['step']
    log = []
    with pytest.raises(ce.Busy) as caught:
        run(_step(log, list(errors[:1])), lambda: log.append('rollback'), limit=1)
    assert caught.value is errors[0] and log == ['step']


def test_rollback_failure_stops_the_retries_and_is_chained():
    log, errors, broken = [], [ce.Busy('1'), ce.Busy('2')], OSError('undo')

    def rollback():
        log.append('rollback')
        if len(log) == 4:
            raise broken

    with pytest.raises(OSError) as caught:
        run(_step(log, list(errors)), rollback)
    assert caught.value is broken and broken.__cause__ is errors[1]
    assert log == ['step', 'rollback', 'step', 'rollback']


def test_non_retryable_error_on_a_later_attempt_and_bad_limit():
    log, errors = [], [ce.Busy('1'), ce.Expired('x')]
    with pytest.raises(ce.Expired) as caught:
        run(_step(log, list(errors)), lambda: log.append('rollback'))
    assert caught.value is errors[1] and log == ['step', 'rollback', 'step']
    log = []
    for limit in (0, -2):
        with pytest.raises(ValueError) as caught:
            run(_step(log, ['ok']), lambda: log.append('rollback'), limit)
        assert str(caught.value) == 'limit must be positive'
    assert log == []
        """,
        extra_files={
            "client_errors": '''
class ClientError(Exception):
    """Base class of the errors raised by the client."""

    retryable = False


class Busy(ClientError):
    """The service is busy; repeating the call may succeed."""

    retryable = True


class Expired(Busy):
    """The session expired; repeating the call can never succeed."""

    retryable = False


class Unavailable(ClientError):
    """No connection right now; repeating the call may succeed."""

    retryable = True


def is_retryable(error):
    """True when the call that raised ``error`` is worth repeating."""
    if not isinstance(error, ClientError) or not type(error).retryable:
        return False
    return not str(error).startswith('terminal ')
            ''',
        },
    ),
    _task(
        "atomic_write", "atomic_file",
        "write_atomic(path, data, sync, replace, unlink) 用“写临时文件再替换”的方式更新二进制文件 path，"
        "sync、replace、unlink 默认是 os.fsync、os.replace、os.unlink。要求：\n"
        "1）临时文件是同目录下的 <name>.tmp，顺序为："
        "写入 data、flush、对临时文件 fsync、关闭临时文件、replace 到 path；"
        "成功后目录里只剩 path，内容为 data。\n"
        "2）任一步抛异常（任何类型）：path 的原有内容不变，不留下临时文件，"
        "并抛出同一个异常对象；sync 失败时不再 replace。\n"
        "3）删除临时文件本身失败时不得掩盖原始异常：原始异常照常抛出，删除失败的异常作为它的 `__context__`"
        "（临时文件尚不存在时也一样）。\n"
        "4）无论成功还是失败，返回或抛出前临时文件都已关闭。",
        '''
        import os


        def write_atomic(path, data, sync=os.fsync, replace=os.replace, unlink=os.unlink):
            tmp = path.with_name(path.name + '.tmp')
            try:
                handle = open(tmp, 'wb')
                handle.write(data)
                sync(handle.fileno())
                handle.flush()
                handle.close()
                replace(tmp, path)
            except OSError:
                unlink(tmp)
                raise
        ''',
        '''
        import os


        def write_atomic(path, data, sync=os.fsync, replace=os.replace, unlink=os.unlink):
            tmp = path.with_name(path.name + '.tmp')
            try:
                with open(tmp, 'wb') as handle:
                    handle.write(data)
                    handle.flush()
                    sync(handle.fileno())
                replace(tmp, path)
            except BaseException as error:
                try:
                    unlink(tmp)
                except BaseException as cleanup_error:
                    error.__context__ = cleanup_error
                raise error
        ''',
        '''
        import os


        def test_replaces_the_file_content(tmp_path):
            target = tmp_path / 'settings.ini'
            target.write_bytes(b'old')
            write_atomic(target, b'new content')
            assert target.read_bytes() == b'new content'
            assert os.listdir(tmp_path) == ['settings.ini']
        ''',
        """
import os


def _raise(error):
    def call(*args):
        raise error
    return call


def _note(seen, tag):
    def call(*args):
        seen.append(tag)
    return call


def test_success_follows_the_order_and_leaves_only_the_target(tmp_path):
    target, seen, fds = tmp_path / 'conf.cfg', [], []
    target.write_bytes(b'old')

    def sync(fd):
        fds.append(fd)
        seen.append(os.fstat(fd).st_size)
        seen.append((tmp_path / 'conf.cfg.tmp').read_bytes())

    def replace(tmp, path):
        with pytest.raises(OSError):
            os.fstat(fds[0])
        seen.append(path.read_bytes())
        os.replace(tmp, path)

    write_atomic(target, b'new payload', sync=sync, replace=replace)
    assert seen == [11, b'new payload', b'old'] and target.read_bytes() == b'new payload'
    assert os.listdir(tmp_path) == ['conf.cfg']


@pytest.mark.parametrize('stage', ['sync', 'replace'])
def test_failure_keeps_the_old_file_and_removes_the_temp_file(tmp_path, stage):
    target, boom, seen = tmp_path / 'conf.cfg', ValueError('boom'), []
    target.write_bytes(b'old')
    calls = {stage: _raise(boom)}
    if stage == 'replace':
        calls['sync'] = _note(seen, 'sync')
    with pytest.raises(ValueError) as caught:
        write_atomic(target, b'new', **calls)
    assert caught.value is boom and target.read_bytes() == b'old'
    assert seen == ['sync'] * (stage == 'replace') and os.listdir(tmp_path) == ['conf.cfg']


def test_cleanup_failure_does_not_mask_the_original_error(tmp_path):
    target, original, cleanup = tmp_path / 'conf.cfg', OSError('disk full'), OSError('busy')
    target.write_bytes(b'old')
    calls = dict(sync=_note([], 'sync'), replace=_raise(original), unlink=_raise(cleanup))
    with pytest.raises(OSError) as caught:
        write_atomic(target, b'new', **calls)
    assert caught.value is original and original.__context__ is cleanup
    assert target.read_bytes() == b'old'


def test_missing_directory_and_unclosed_handle(tmp_path):
    target, cleanup = tmp_path / 'missing' / 'conf.cfg', OSError('busy')
    with pytest.raises(FileNotFoundError) as caught:
        write_atomic(target, b'new', unlink=_raise(cleanup))
    assert caught.value.__context__ is cleanup and not target.parent.exists()
    target, boom, fds = tmp_path / 'conf.cfg', ValueError('boom'), []
    target.write_bytes(b'old')

    def sync(fd):
        fds.append(fd)
        raise boom

    with pytest.raises(ValueError):
        write_atomic(target, b'new', sync=sync)
    with pytest.raises(OSError):
        os.fstat(fds[0])
    assert target.read_bytes() == b'old' and os.listdir(tmp_path) == ['conf.cfg']
        """,
    ),
    _task(
        "conn_session", "session_guard",
        "Session(conn) 用 with 语句管理一个连接，conn 有 open / close / commit / rollback 四个方法。要求：\n"
        "1）最外层进入时调用 conn.open()；嵌套进入什么都不做；"
        "只有最外层退出才收尾，内层退出什么都不做、也不吞异常。\n"
        "2）最外层正常退出：先 commit 再 close；commit 抛异常时改为 rollback，再 close，"
        "最后抛出 commit 的那个异常对象。\n"
        "3）body 抛异常而退出：rollback 后 close，然后 body 的异常继续传播（同一个对象）。\n"
        "4）rollback 抛异常也要 close，并抛出 rollback 的异常；close 抛异常则抛出 close 的异常。"
        "这两种情况下此前已有的异常（commit、rollback 或 body 的）都作为所抛异常的 `__context__`。"
        "但 commit 或 rollback 抛出的异常被只读模块 conn_driver 的 is_fatal(e) 判为致命时，"
        "不再调用 conn 的任何方法（不 rollback、不 close），直接抛出它。\n"
        "5）open() 抛异常时直接抛出、不调用 close，session 仍可再次使用；"
        "最外层退出（无论成败）之后都可以再次 with，会重新 open。",
        '''
        class Session:
            def __init__(self, conn):
                self.conn = conn
                self._depth = 0

            def __enter__(self):
                self._depth += 1
                if self._depth == 1:
                    self.conn.open()
                return self

            def __exit__(self, exc_type, exc_value, tb):
                if exc_value is None:
                    self.conn.commit()
                else:
                    self.conn.rollback()
                self._depth -= 1
                if self._depth == 0:
                    self.conn.close()
        ''',
        '''
        from conn_driver import is_fatal


        class Session:
            def __init__(self, conn):
                self.conn = conn
                self._depth = 0

            def __enter__(self):
                if self._depth == 0:
                    self.conn.open()
                self._depth += 1
                return self

            def _finish(self, error):
                fatal = False
                try:
                    if error is None:
                        try:
                            self.conn.commit()
                        except Exception as commit_error:
                            error, fatal = commit_error, is_fatal(commit_error)
                            if not fatal:
                                self.conn.rollback()
                    else:
                        self.conn.rollback()
                except Exception as rollback_error:
                    rollback_error.__context__ = error
                    error, fatal = rollback_error, is_fatal(rollback_error)
                if not fatal:
                    try:
                        self.conn.close()
                    except Exception as close_error:
                        close_error.__context__ = error
                        error = close_error
                if error is not None:
                    raise error

            def __exit__(self, exc_type, exc_value, tb):
                self._depth -= 1
                if self._depth == 0:
                    self._finish(exc_value)
                return False
        ''',
        '''
        def test_commits_and_closes_on_success():
            calls = []

            class Conn:
                pass

            for name in ('open', 'commit', 'rollback', 'close'):
                setattr(Conn, name, lambda self, name=name: calls.append(name))
            conn = Conn()
            with Session(conn):
                calls.append('body')
            assert calls == ['open', 'body', 'commit', 'close']
        ''',
        """
import conn_driver as cd


class _Conn:
    def __init__(self, **fails):
        self.calls, self.fails = [], fails

    def _call(self, name):
        self.calls.append(name)
        if name in self.fails:
            raise self.fails[name]

    open = lambda self: self._call('open')
    commit = lambda self: self._call('commit')
    rollback = lambda self: self._call('rollback')
    close = lambda self: self._call('close')


def test_nested_exit_does_nothing_and_the_outermost_exit_finishes():
    conn = _Conn()
    session = Session(conn)
    with session:
        with pytest.raises(KeyError):
            with session:
                raise KeyError('inner')
        conn.calls.append('after')
    assert conn.calls == ['open', 'after', 'commit', 'close']


@pytest.mark.parametrize('fails, calls, raised, chained', [
    (dict(commit=OSError('c')), ['open', 'commit', 'rollback', 'close'], 'commit', None),
    (dict(commit=OSError('c'), rollback=OSError('r')), ['open', 'commit', 'rollback', 'close'], 'rollback', 'commit'),
    (dict(close=OSError('x')), ['open', 'commit', 'close'], 'close', None),
    (dict(commit=OSError('c'), close=OSError('x')), ['open', 'commit', 'rollback', 'close'], 'close', 'commit'),
    (dict(commit=cd.Broken('b')), ['open', 'commit'], 'commit', None),
    (dict(commit=cd.Transient('t'), rollback=cd.Broken('b')), ['open', 'commit', 'rollback'], 'rollback', 'commit'),
    (dict(commit=cd.HalfOpen('h'), rollback=cd.Transient('t')), ['open', 'commit', 'rollback', 'close'], 'rollback',
     'commit'),
])
def test_normal_exit_finishes_step_by_step(fails, calls, raised, chained):
    conn = _Conn(**fails)
    with pytest.raises(Exception) as caught:
        with Session(conn):
            pass
    assert conn.calls == calls and caught.value is conn.fails[raised]
    assert caught.value.__context__ is (conn.fails[chained] if chained else None)


@pytest.mark.parametrize('fails, calls', [
    ({}, ['open', 'rollback', 'close']),
    (dict(rollback=OSError('r')), ['open', 'rollback', 'close']),
    (dict(close=OSError('x')), ['open', 'rollback', 'close']),
    (dict(rollback=cd.Broken('b')), ['open', 'rollback']),
])
def test_body_exception_finishes_step_by_step(fails, calls):
    conn, body = _Conn(**fails), ValueError('body')
    with pytest.raises(Exception) as caught:
        with Session(conn):
            raise body
    error = next(iter(conn.fails.values()), body)
    assert conn.calls == calls and caught.value is error
    assert error.__context__ is (None if error is body else body)


def test_open_failure_and_reuse():
    conn = _Conn(open=OSError('open'))
    session = Session(conn)
    with pytest.raises(OSError) as caught:
        with session:
            pass
    assert caught.value is conn.fails['open'] and conn.calls == ['open']
    del conn.fails['open']
    with session:
        pass
    assert conn.calls == ['open', 'open', 'commit', 'close']
    conn = _Conn(commit=OSError('c'))
    session = Session(conn)
    for _ in range(2):
        with pytest.raises(OSError):
            with session:
                pass
    assert conn.calls == ['open', 'commit', 'rollback', 'close'] * 2
        """,
        extra_files={
            "conn_driver": '''
class DriverError(Exception):
    """Base class of the errors the driver raises."""


class Transient(DriverError):
    """The call failed, but the connection is still usable afterwards."""


class Broken(Transient):
    """The connection is gone; every later call only raises again."""


class HalfOpen(Broken):
    """The connection is unusable, but close() still works and must be called."""


def is_fatal(error):
    """True when nothing may be sent to the connection after ``error``."""
    return isinstance(error, Broken) and not isinstance(error, HalfOpen)
            ''',
        },
    ),
    _task(
        "resource_pool", "resource_stack",
        "Pool 登记需要关闭的资源（都有 close() 方法）。要求：\n"
        "1）add(resource) 登记资源并返回句柄（从 1 起递增，释放后不复用）。\n"
        "2）release(handle)：先把该资源移出池，再调用它的 close()；句柄不在池中抛 KeyError(handle)；"
        "close() 抛异常时异常照常抛出（资源已移出）。\n"
        "3）close()：按登记逆序关闭池中剩余的全部资源；某个 close() 抛异常不影响其余资源，"
        "结束时抛出第一个（按执行顺序）close 异常，其余的丢弃。\n"
        "4）close() 一开始池即已关闭：之后 add 抛 RuntimeError('pool is closed')，release 抛 KeyError，"
        "再 close() 直接返回 None。",
        '''
        class Pool:
            def __init__(self):
                self._items = []
                self._next = 1

            def add(self, resource):
                handle = self._next
                self._next += 1
                self._items.append((handle, resource))
                return handle

            def release(self, handle):
                for index, (key, resource) in enumerate(self._items):
                    if key == handle:
                        resource.close()
                        del self._items[index]
                        return
                raise KeyError(handle)

            def close(self):
                while self._items:
                    self.release(self._items[0][0])
        ''',
        '''
        class Pool:
            def __init__(self):
                self._items = []
                self._next = 1
                self._closed = False

            def add(self, resource):
                if self._closed:
                    raise RuntimeError('pool is closed')
                self._items.append((self._next, resource))
                self._next += 1
                return self._next - 1

            def release(self, handle):
                for index, (key, resource) in enumerate(self._items):
                    if key == handle:
                        del self._items[index]
                        resource.close()
                        return None
                raise KeyError(handle)

            def close(self):
                if self._closed:
                    return None
                self._closed = True
                items, self._items = self._items, []
                error = None
                for _, resource in reversed(items):
                    try:
                        resource.close()
                    except Exception as caught:
                        error = error or caught
                if error is not None:
                    raise error
                return None
        ''',
        '''
        class Resource:
            def __init__(self, name, calls):
                self.name, self.calls = name, calls

            def close(self):
                self.calls.append(self.name)


        def test_close_closes_every_held_resource():
            calls, pool = [], Pool()
            pool.add(Resource('a', calls))
            pool.add(Resource('b', calls))
            assert pool.close() is None
            assert sorted(calls) == ['a', 'b']
        ''',
        """
class _Res:
    def __init__(self, log, name, error=None):
        def close():
            log.append(name)
            if error is not None:
                raise error
        self.close = close


def test_release_closes_only_that_resource_and_handles_are_not_reused():
    log, pool = [], Pool()
    first = pool.add(_Res(log, 'a'))
    second = pool.add(_Res(log, 'b'))
    pool.release(first)
    assert log == ['a'] and pool.add(_Res(log, 'c')) == 3
    with pytest.raises(KeyError) as caught:
        pool.release(first)
    assert caught.value.args == (first,)
    with pytest.raises(KeyError):
        pool.release(second + 5)
    assert pool.close() is None and log == ['a', 'c', 'b']


def test_failing_release_still_removes_the_resource():
    log, boom, pool = [], OSError('close'), Pool()
    handle = pool.add(_Res(log, 'a', boom))
    pool.add(_Res(log, 'b'))
    with pytest.raises(OSError) as caught:
        pool.release(handle)
    assert caught.value is boom and log == ['a']
    assert pool.close() is None and log == ['a', 'b']


def test_close_runs_in_reverse_and_survives_failures():
    log, first, second, pool = [], OSError('1'), OSError('2'), Pool()
    pool.add(_Res(log, 'a'))
    pool.add(_Res(log, 'b', second))
    pool.add(_Res(log, 'c'))
    pool.add(_Res(log, 'd', first))
    with pytest.raises(OSError) as caught:
        pool.close()
    assert caught.value is first and log == ['d', 'c', 'b', 'a']
    assert pool.close() is None and log == ['d', 'c', 'b', 'a']


def test_closed_pool_rejects_add_and_release():
    log, pool = [], Pool()
    pool.add(_Res(log, 'a'))
    assert pool.close() is None and log == ['a']
    with pytest.raises(RuntimeError) as caught:
        pool.add(_Res(log, 'b'))
    assert str(caught.value) == 'pool is closed'
    with pytest.raises(KeyError):
        pool.release(1)
    assert pool.close() is None and log == ['a']


def test_releasing_from_inside_close():
    log, pool, handle = [], Pool(), None

    class Selfish:
        def close(self):
            log.append('selfish')
            pool.release(handle)

    handle = pool.add(Selfish())
    with pytest.raises(KeyError) as caught:
        pool.release(handle)
    assert caught.value.args == (handle,) and log == ['selfish']
    log, pool = [], Pool()
    pool.add(_Res(log, 'a'))
    handle = pool.add(_Res(log, 'b'))

    class Neighbour:
        def close(self):
            log.append('c')
            pool.release(handle)

    pool.add(Neighbour())
    with pytest.raises(KeyError) as caught:
        pool.close()
    assert caught.value.args == (handle,) and log == ['c', 'b', 'a']
        """,
    ),
    _task(
        "retry_batch", "retry_rollback",
        "run_batch(jobs) 依次执行 jobs 中的 (action, undo)；action() 抛异常时，"
        "按只读模块 job_backend 的 worth_retrying(error, used) 决定是否重试（used 是该 job 已重试的次数）。要求：\n"
        "1）可重试：先调用该 job 的 undo()，再重试；不可重试：该 job 最终失败。\n"
        "2）最终失败时先回滚：调用该 job 的 undo()，再按相反顺序调用之前已完成的 job 的 undo()；"
        "然后抛出 action 的同一个异常对象。\n"
        "3）重试前的 undo() 抛异常：该 job 不再重试、也不再调用它自己的 undo，其余已完成的 job 照常回滚；"
        "抛出 undo 的异常，action 的异常作为它的 `__cause__`。\n"
        "4）回滚中某个 undo() 抛异常不影响其余 undo，也不掩盖要抛出的异常："
        "第一个（按执行顺序）undo 异常作为它的 `__context__`，其余丢弃。\n"
        "5）全部成功时返回执行的 job 数。",
        '''
        from job_backend import BackendFailure, worth_retrying


        def run_batch(jobs):
            done = []
            for action, undo in jobs:
                used = 0
                while True:
                    try:
                        action()
                        break
                    except BackendFailure as error:
                        if not worth_retrying(error, used):
                            for older_action, older_undo in done:
                                older_undo()
                            raise error
                        used += 1
                        undo()
                done.append((action, undo))
            return len(done)
        ''',
        '''
        from job_backend import worth_retrying


        def _roll_back(undos, error):
            first = None
            for undo in undos:
                try:
                    undo()
                except Exception as undo_error:
                    first = first or undo_error
            if first is not None:
                error.__context__ = first


        def run_batch(jobs):
            done = []
            for action, undo in jobs:
                used, failure = 0, None
                while True:
                    try:
                        action()
                        break
                    except Exception as error:
                        failure = error
                        if not worth_retrying(error, used):
                            _roll_back([undo] + [older for _, older in reversed(done)], error)
                            break
                        used += 1
                        try:
                            undo()
                            failure = None
                            continue
                        except Exception as undo_error:
                            undo_error.__cause__ = error
                            _roll_back([older for _, older in reversed(done)], undo_error)
                            failure = undo_error
                            break
                if failure is not None:
                    raise failure
                done.append((action, undo))
            return len(done)
        ''',
        '''
        import job_backend


        def test_runs_every_job_with_retries():
            calls, failures = [], [job_backend.BackendFailure('busy', 1)]

            def action():
                calls.append('action')
                if failures:
                    raise failures.pop()

            assert run_batch([(action, lambda: calls.append('undo')), (lambda: None, lambda: None)]) == 2
            assert calls == ['action', 'undo', 'action']
        ''',
        """
import job_backend as jb


def _job(log, name, errors=(), undo_error=None):
    def action():
        log.append(name)
        if errors:
            raise errors.pop(0)

    def undo():
        log.append('~' + name)
        if undo_error is not None:
            raise undo_error
    return action, undo


def test_retries_follow_the_budget_of_each_error():
    log, busy, later = [], jb.BackendFailure('busy', 1), jb.BackendFailure('later', 4)
    assert run_batch([_job(log, 'a', [busy]), _job(log, 'b', [later] * 4)]) == 2
    assert log == ['a', '~a', 'a'] + ['b', '~b'] * 4 + ['b']


def test_final_failure_rolls_back_in_reverse_order():
    log, busy, corrupt = [], jb.BackendFailure('busy', 1), jb.BackendFailure('corrupt', 3)
    foreign = ValueError('x')
    with pytest.raises(jb.BackendFailure) as caught:
        run_batch([_job(log, 'a'), _job(log, 'b', [busy, corrupt]), _job(log, 'z')])
    assert caught.value is corrupt and log == ['a', 'b', '~b', 'b', '~b', '~a']
    log = []
    with pytest.raises(ValueError) as caught:
        run_batch([_job(log, 'c'), _job(log, 'd', [foreign]), _job(log, 'z')])
    assert caught.value is foreign and log == ['c', 'd', '~d', '~c']


def test_undo_failure_before_a_retry_rolls_back_the_finished_jobs():
    log, busy, broken = [], jb.BackendFailure('busy', 3), OSError('undo c')
    with pytest.raises(OSError) as caught:
        run_batch([_job(log, 'a'), _job(log, 'b'), _job(log, 'c', [busy], broken)])
    assert caught.value is broken and broken.__cause__ is busy
    assert log == ['a', 'b', 'c', '~c', '~b', '~a']


def test_failures_inside_the_rollback_do_not_stop_or_mask():
    log, final, early, late = [], jb.BackendFailure('corrupt', 5), OSError('d'), OSError('b')
    with pytest.raises(jb.BackendFailure) as caught:
        run_batch([_job(log, 'a'), _job(log, 'b', (), late), _job(log, 'c'), _job(log, 'd', [final], early)])
    assert caught.value is final and final.__context__ is early
    assert log == ['a', 'b', 'c', 'd', '~d', '~c', '~b', '~a']
    log, busy, broken, late_b = [], jb.BackendFailure('busy', 2), OSError('c'), OSError('b')
    with pytest.raises(OSError) as caught:
        run_batch([_job(log, 'a'), _job(log, 'b', (), late_b), _job(log, 'c', [busy], broken)])
    assert caught.value is broken and broken.__cause__ is busy and broken.__context__ is late_b
    assert log == ['a', 'b', 'c', '~c', '~b', '~a']
        """,
        extra_files={
            "job_backend": '''
class BackendFailure(Exception):
    """A call the backend could not complete."""

    def __init__(self, reason, budget):
        super().__init__(reason)
        self.budget = budget


def worth_retrying(error, used):
    """True when the job that raised ``error`` may be sent again.

    ``used`` is how many times that job has already been repeated.  A corrupt
    job is never repeated; every other failure allows ``budget`` repeats.
    """
    if not isinstance(error, BackendFailure) or error.args[0] == 'corrupt':
        return False
    return used < error.budget
            ''',
        },
    ),
    _task(
        "staged_save", "atomic_file",
        "save(path, data, rename=os.rename, unlink=os.unlink) 分阶段更新文件。阶段：把 data 写入同目录的 <name>.new"
        "（flush、fsync、关闭）；path 已存在则先把 path 改名为 <name>.old；再把 <name>.new 改名为 path；"
        "最后删除 <name>.old，返回 True。要求：\n"
        "1）写 <name>.new 期间失败：删除它，path 不变，抛出同一个异常对象。\n"
        "2）把 path 改名为 <name>.old 失败：同上（删除 <name>.new，path 不变）。\n"
        "3）把 <name>.new 改名为 path 失败：先把 <name>.old 改回 path（没有备份则跳过这一步），再删除 <name>.new，"
        "抛出同一个异常对象。\n"
        "4）恢复或删除失败不得掩盖原始异常：原始异常照常抛出，清理中第一个（按执行顺序）异常作为它的 `__context__`；"
        "恢复失败时保留 <name>.old。\n"
        "5）最后删除 <name>.old 失败不算保存失败：不抛异常，返回 False。",
        '''
        import os


        def save(path, data, rename=os.rename, unlink=os.unlink):
            new, old = path.with_name(path.name + '.new'), path.with_name(path.name + '.old')
            with open(new, 'wb') as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            if path.exists():
                rename(path, old)
            try:
                rename(new, path)
            except OSError:
                rename(old, path)
                unlink(new)
                raise
            if old.exists():
                unlink(old)
            return True
        ''',
        '''
        import os


        def _cleanup(error, actions):
            first = None
            for action in actions:
                try:
                    action()
                except BaseException as caught:
                    first = first or caught
            if first is not None:
                error.__context__ = first


        def save(path, data, rename=os.rename, unlink=os.unlink):
            new, old = path.with_name(path.name + '.new'), path.with_name(path.name + '.old')
            backup = False
            try:
                with open(new, 'wb') as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                if path.exists():
                    rename(path, old)
                    backup = True
                rename(new, path)
            except BaseException as error:
                actions = ([lambda: rename(old, path)] if backup else []) + [lambda: unlink(new)]
                _cleanup(error, actions)
                raise error
            if backup:
                try:
                    unlink(old)
                except OSError:
                    return False
            return True
        ''',
        '''
        import os


        def test_replaces_an_existing_file(tmp_path):
            target = tmp_path / 'cfg.ini'
            target.write_bytes(b'old')
            assert save(target, b'new') is True
            assert target.read_bytes() == b'new'
            assert os.listdir(tmp_path) == ['cfg.ini']
        ''',
        """
import os


def _failing_rename(call, error, seen):
    def rename(src, dst):
        seen.append((src.name, dst.name))
        if len(seen) == call:
            raise error
        os.rename(src, dst)
    return rename


def test_success_with_and_without_a_backup(tmp_path):
    target, seen = tmp_path / 'cfg.ini', []
    target.write_bytes(b'old')
    rename = _failing_rename(0, None, seen)
    assert save(target, b'new', rename=rename) is True
    assert seen == [('cfg.ini', 'cfg.ini.old'), ('cfg.ini.new', 'cfg.ini')]
    assert target.read_bytes() == b'new' and os.listdir(tmp_path) == ['cfg.ini']
    assert save(tmp_path / 'other.ini', b'data', rename=rename) is True
    assert seen[-1] == ('other.ini.new', 'other.ini') and (tmp_path / 'other.ini').read_bytes() == b'data'


def test_write_and_backup_failures_remove_the_temp_file(tmp_path):
    target = tmp_path / 'cfg.ini'
    target.write_bytes(b'old')
    with pytest.raises(TypeError):
        save(target, 7)
    assert target.read_bytes() == b'old' and os.listdir(tmp_path) == ['cfg.ini']
    boom, seen = OSError('backup'), []
    with pytest.raises(OSError) as caught:
        save(target, b'new', rename=_failing_rename(1, boom, seen))
    assert caught.value is boom and target.read_bytes() == b'old'
    assert os.listdir(tmp_path) == ['cfg.ini']
    with pytest.raises(FileNotFoundError) as caught:
        save(tmp_path / 'missing' / 'cfg.ini', b'new')
    assert type(caught.value.__context__) is FileNotFoundError


def test_replace_failure_restores_the_backup(tmp_path):
    target, boom, seen = tmp_path / 'cfg.ini', OSError('replace'), []
    target.write_bytes(b'old')
    with pytest.raises(OSError) as caught:
        save(target, b'new', rename=_failing_rename(2, boom, seen))
    assert caught.value is boom and target.read_bytes() == b'old'
    assert seen[-1] == ('cfg.ini.old', 'cfg.ini') and os.listdir(tmp_path) == ['cfg.ini']
    with pytest.raises(OSError) as caught:
        save(tmp_path / 'fresh.ini', b'new', rename=_failing_rename(1, boom, []))
    assert caught.value is boom and os.listdir(tmp_path) == ['cfg.ini']


def test_restore_and_unlink_failures_do_not_mask(tmp_path):
    target, original, restore = tmp_path / 'cfg.ini', OSError('replace'), OSError('restore')
    target.write_bytes(b'old')

    def rename(src, dst):
        if dst == target:
            raise original if src.name.endswith('.new') else restore
        os.rename(src, dst)

    with pytest.raises(OSError) as caught:
        save(target, b'new', rename=rename)
    assert caught.value is original and original.__context__ is restore
    assert os.listdir(tmp_path) == ['cfg.ini.old']
    target.write_bytes(b'old')
    with pytest.raises(OSError) as caught:
        save(target, b'new', rename=rename, unlink=_raise(OSError('unlink')))
    assert caught.value is original and original.__context__ is restore
    assert sorted(os.listdir(tmp_path)) == ['cfg.ini.new', 'cfg.ini.old']


def test_backup_removal_failure_is_reported_not_raised(tmp_path):
    target = tmp_path / 'cfg.ini'
    target.write_bytes(b'old')
    assert save(target, b'new', unlink=_raise(OSError('unlink'))) is False
    assert target.read_bytes() == b'new' and sorted(os.listdir(tmp_path)) == ['cfg.ini', 'cfg.ini.old']


def _raise(error):
    def call(*args):
        raise error
    return call
        """,
    ),
    _task(
        "nested_tx", "session_guard",
        "Tx(conn) 用嵌套的 with 表示事务和保存点；conn 有 begin、commit、rollback、savepoint(name)、"
        "release(name)、rollback_to(name)。要求：\n"
        "1）最外层进入调用 conn.begin()；嵌套进入调用 conn.savepoint('sp<n>')，n 是它外面的层数（第一层嵌套是 sp1）。\n"
        "2）正常退出：最外层 conn.commit()，嵌套 conn.release('sp<n>')，返回 False。\n"
        "3）body 抛异常而退出：最外层 conn.rollback()，嵌套 conn.rollback_to('sp<n>')；"
        "若只读模块 tx_signals 的 undoes_only_the_block(exc) 为真则返回 True（吞掉它，with 之后的代码继续执行），"
        "否则返回 False，同一个异常继续传播。\n"
        "4）收尾调用抛异常：抛出该异常，body 的异常（若有）作为它的 `__cause__`。\n"
        "5）进入时 begin() / savepoint() 抛异常：直接抛出，层数不变；最外层退出（无论成败）之后 Tx 可以再次使用。",
        '''
        from tx_signals import Abort


        class Tx:
            def __init__(self, conn):
                self.conn = conn
                self._depth = 0

            def __enter__(self):
                self._depth += 1
                if self._depth == 1:
                    self.conn.begin()
                else:
                    self.conn.savepoint(f'sp{self._depth - 1}')
                return self

            def __exit__(self, exc_type, exc_value, tb):
                if exc_value is None:
                    if self._depth == 1:
                        self.conn.commit()
                    else:
                        self.conn.release(f'sp{self._depth - 1}')
                elif self._depth == 1:
                    self.conn.rollback()
                else:
                    self.conn.rollback_to(f'sp{self._depth - 1}')
                self._depth -= 1
                return isinstance(exc_value, Abort)
        ''',
        '''
        from tx_signals import undoes_only_the_block


        class Tx:
            def __init__(self, conn):
                self.conn = conn
                self._depth = 0

            def __enter__(self):
                if self._depth == 0:
                    self.conn.begin()
                else:
                    self.conn.savepoint(f'sp{self._depth}')
                self._depth += 1
                return self

            def __exit__(self, exc_type, exc_value, tb):
                self._depth -= 1
                name = f'sp{self._depth}'
                try:
                    if exc_value is None:
                        if self._depth == 0:
                            self.conn.commit()
                        else:
                            self.conn.release(name)
                    elif self._depth == 0:
                        self.conn.rollback()
                    else:
                        self.conn.rollback_to(name)
                except Exception as finish_error:
                    if exc_value is not None:
                        finish_error.__cause__ = exc_value
                    raise finish_error
                return undoes_only_the_block(exc_value)
        ''',
        '''
        def test_nested_blocks_commit_once():
            calls = []

            class Conn:
                pass

            for name in ('begin', 'commit', 'rollback', 'savepoint', 'release', 'rollback_to'):
                setattr(Conn, name, lambda self, *args, name=name: calls.append(name))
            tx = Tx(Conn())
            with tx:
                with tx:
                    calls.append('inner')
            assert calls == ['begin', 'savepoint', 'inner', 'release', 'commit']
        ''',
        """
import tx_signals as ts


class _Conn:
    def __init__(self, **fails):
        self.calls, self.fails = [], fails

    def _call(self, name, arg=None):
        key = name if arg is None else f'{name} {arg}'
        self.calls.append(key)
        if key in self.fails:
            raise self.fails[key]

    begin = lambda self: self._call('begin')
    commit = lambda self: self._call('commit')
    rollback = lambda self: self._call('rollback')
    savepoint = lambda self, name: self._call('savepoint', name)
    release = lambda self, name: self._call('release', name)
    rollback_to = lambda self, name: self._call('rollback_to', name)


def test_abort_undoes_only_its_own_block():
    tx = Tx(_Conn())
    with tx:
        with tx:
            with tx:
                raise ts.Abort('block')
            tx.conn.calls.append('after inner')
        tx.conn.calls.append('after middle')
    assert tx.conn.calls == ['begin', 'savepoint sp1', 'savepoint sp2', 'rollback_to sp2', 'after inner',
                             'release sp1', 'after middle', 'commit']
    with tx:
        raise ts.Abort('block')
    assert tx.conn.calls[-2:] == ['begin', 'rollback']


def test_other_failures_end_the_whole_transaction():
    fatal = ts.Abort('all')
    tx = Tx(_Conn())
    with pytest.raises(ts.Abort) as caught:
        with tx:
            with tx:
                raise fatal
    assert caught.value is fatal and tx.conn.calls == ['begin', 'savepoint sp1', 'rollback_to sp1', 'rollback']
    tx = Tx(_Conn())
    with pytest.raises(ValueError) as caught:
        with tx:
            raise ValueError('x')
    assert type(caught.value) is ValueError and tx.conn.calls == ['begin', 'rollback']


def test_failing_finish_call_is_chained_and_the_tx_stays_reusable():
    body, finish = ts.Abort('block'), OSError('finish')
    tx = Tx(_Conn(**{'rollback_to sp1': finish}))
    with pytest.raises(OSError) as caught:
        with tx:
            with tx:
                raise body
    assert caught.value is finish and finish.__cause__ is body
    assert tx.conn.calls == ['begin', 'savepoint sp1', 'rollback_to sp1', 'rollback']
    tx = Tx(_Conn(commit=OSError('commit')))
    with pytest.raises(OSError) as caught:
        with tx:
            pass
    assert caught.value is tx.conn.fails['commit'] and tx.conn.calls == ['begin', 'commit']
    del tx.conn.fails['commit']
    with tx:
        pass
    assert tx.conn.calls[-2:] == ['begin', 'commit']


def test_failing_enter_keeps_the_level_and_allows_reuse():
    tx = Tx(_Conn(**{'savepoint sp1': OSError('sp')}))
    with pytest.raises(OSError) as caught:
        with tx:
            with tx:
                tx.conn.calls.append('never')
    assert caught.value is tx.conn.fails['savepoint sp1']
    assert tx.conn.calls == ['begin', 'savepoint sp1', 'rollback']
    tx = Tx(_Conn(begin=OSError('begin')))
    with pytest.raises(OSError):
        with tx:
            pass
    del tx.conn.fails['begin']
    with tx:
        pass
    assert tx.conn.calls == ['begin', 'begin', 'commit']
        """,
        extra_files={
            "tx_signals": '''
class Abort(Exception):
    """Raised inside a transaction block that wants its work undone."""

    def __init__(self, scope='block'):
        super().__init__(scope)
        self.scope = scope


def undoes_only_the_block(error):
    """True when ``error`` undoes only the block that raised it.

    The caller then goes on after the ``with`` statement.  Every other
    failure, including Abort('all'), ends the whole transaction.
    """
    return isinstance(error, Abort) and error.scope == 'block'
            ''',
        },
    ),
]

MUTANTS = {
    "h3_cleanup_scope_001": [
        ("        if self._closed:\n"
         "            raise RuntimeError('scope is closed')\n", ""),
        ("        self._closed = True\n"
         "        items, self._items = self._items, []\n",
         "        items, self._items = self._items, []\n"),
        ("                if callback(exc):\n"
         "                    exc = None\n",
         "                if callback(exc):\n"
         "                    return None, None\n"),
        ("                    if exc is not None:\n"
         "                        caught.__context__ = exc\n", ""),
        ("        return exc is None\n", "        return True\n"),
    ],
    "h3_retry_step_001": [
        ("            if left == 1 or not is_retryable(error):\n",
         "            if left == 1:\n"),
        ("                raise cleanup_error from error\n", "                raise cleanup_error\n"),
        ("    if limit < 1:\n"
         "        raise ValueError('limit must be positive')\n", ""),
        ("    for left in range(limit, 0, -1):\n", "    for left in range(limit):\n"),
        ("raise ValueError('limit must be positive')", "raise ValueError('bad limit')"),
    ],
    "h3_atomic_write_001": [
        ("            handle.flush()\n"
         "            sync(handle.fileno())\n",
         "            sync(handle.fileno())\n"
         "            handle.flush()\n"),
        ("    except BaseException as error:\n", "    except OSError as error:\n"),
        ("            error.__context__ = cleanup_error\n", "            pass\n"),
        ("            unlink(tmp)\n", "            pass\n"),
        ("        with open(tmp, 'wb') as handle:\n"
         "            handle.write(data)\n"
         "            handle.flush()\n"
         "            sync(handle.fileno())\n"
         "        replace(tmp, path)\n",
         "        handle = open(tmp, 'wb')\n"
         "        handle.write(data)\n"
         "        handle.flush()\n"
         "        sync(handle.fileno())\n"
         "        handle.close()\n"
         "        replace(tmp, path)\n"),
    ],
    "h3_conn_session_001": [
        ("        self._depth -= 1\n"
         "        if self._depth == 0:\n"
         "            self._finish(exc_value)\n",
         "        self._depth -= 1\n"
         "        self._finish(exc_value)\n"),
        ("                    error, fatal = commit_error, is_fatal(commit_error)\n"
         "                    if not fatal:\n"
         "                        self.conn.rollback()\n",
         "                    error = commit_error\n"
         "                    self.conn.rollback()\n"),
        ("        if self._depth == 0:\n"
         "            self.conn.open()\n"
         "        self._depth += 1\n",
         "        self._depth += 1\n"
         "        if self._depth == 1:\n"
         "            self.conn.open()\n"),
        ("                    if not fatal:\n"
         "                        self.conn.rollback()\n",
         "                    self.conn.rollback()\n"),
        ("        if not fatal:\n"
         "            try:\n"
         "                self.conn.close()\n",
         "        if True:\n"
         "            try:\n"
         "                self.conn.close()\n"),
    ],
    "h3_resource_pool_001": [
        ("        for _, resource in reversed(items):\n", "        for _, resource in items:\n"),
        ("                del self._items[index]\n"
         "                resource.close()\n",
         "                resource.close()\n"
         "                del self._items[index]\n"),
        ("        if self._closed:\n"
         "            raise RuntimeError('pool is closed')\n", ""),
        ("                error = error or caught\n", "                error = caught\n"),
        ("            except Exception as caught:\n"
         "                error = error or caught\n",
         "            except Exception:\n"
         "                pass\n"),
    ],
    "h3_retry_batch_001": [
        ("                if not worth_retrying(error, used):\n"
         "                    _roll_back([undo] + [older for _, older in reversed(done)], error)\n",
         "                if not worth_retrying(error, used):\n"
         "                    _roll_back([older for _, older in done], error)\n"),
        ("                    undo_error.__cause__ = error\n"
         "                    _roll_back([older for _, older in reversed(done)], undo_error)\n",
         "                    _roll_back([older for _, older in reversed(done)], undo_error)\n"),
        ("        except Exception as undo_error:\n"
         "            first = first or undo_error\n",
         "        except Exception:\n"
         "            pass\n"),
        ("                    _roll_back([older for _, older in reversed(done)], undo_error)\n"
         "                    failure = undo_error\n",
         "                    failure = undo_error\n"),
        ("        if failure is not None:\n"
         "            raise failure\n",
         "        if failure is not None and not isinstance(failure, OSError):\n"
         "            raise failure\n"),
    ],
    "h3_staged_save_001": [
        ("            backup = True\n", "            pass\n"),
        ("    except BaseException as error:\n",
         "    except OSError as error:\n"),
        ("            first = first or caught\n",
         "            pass\n"),
        ("        except OSError:\n"
         "            return False\n",
         "        except OSError:\n"
         "            raise\n"),
        ("        actions = ([lambda: rename(old, path)] if backup else []) + [lambda: unlink(new)]\n",
         "        actions = ([] if backup else [lambda: rename(old, path)]) + [lambda: unlink(new)]\n"),
    ],
    "h3_nested_tx_001": [
        ("        return undoes_only_the_block(exc_value)\n",
         "        return exc_value is not None\n"),
        ("        if self._depth == 0:\n"
         "            self.conn.begin()\n"
         "        else:\n"
         "            self.conn.savepoint(f'sp{self._depth}')\n"
         "        self._depth += 1\n"
         "        return self\n",
         "        self._depth += 1\n"
         "        if self._depth == 1:\n"
         "            self.conn.begin()\n"
         "        else:\n"
         "            self.conn.savepoint(f'sp{self._depth - 1}')\n"
         "        return self\n"),
        ("            if exc_value is not None:\n"
         "                finish_error.__cause__ = exc_value\n"
         "            raise finish_error\n",
         "            raise finish_error\n"),
        ("        self._depth -= 1\n"
         "        name = f'sp{self._depth}'\n",
         "        name = f'sp{self._depth - 1}'\n"),
        ("        return undoes_only_the_block(exc_value)\n",
         "        self._depth -= 1\n"
         "        return undoes_only_the_block(exc_value)\n"),
    ],
}
