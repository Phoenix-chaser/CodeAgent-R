"""Transaction tasks for the hard batch (h3).

Four families, two variants each: a unit of work over a capacity-limited store
(``unit_of_work``), a batch applied through compensating undo (``batch_apply``),
an idempotent double-entry journal (``ledger_journal``) and the incremental
recomputation of a dependency graph (``graph_replan``).  The happy path always
works on the buggy program; the defects show up when a step in the middle fails:
earlier steps stay applied, the original exception is replaced or loses its
chain, or a closed or half-finished transaction silently accepts more work.
"""
from __future__ import annotations

from agentlab.benchmark.hard_tasks_v3_common import _task

_SLOT_STORE = '''
class StoreError(Exception):
    """args is (reason, key); reason is 'unknown key' or 'full'."""


class SlotStore:
    """put() and delete() apply one operation each."""

    def __init__(self, capacity=4, initial=None):
        self.capacity, self._data = capacity, dict(initial or {})

    def __contains__(self, key):
        return key in self._data

    def get(self, key):
        if key not in self._data:
            raise StoreError('unknown key', key)
        return self._data[key]

    def put(self, key, value):
        if key not in self._data and len(self._data) >= self.capacity:
            raise StoreError('full', key)
        self._data[key] = value

    def delete(self, key):
        if key not in self._data:
            raise StoreError('unknown key', key)
        del self._data[key]

    def items(self):
        return dict(sorted(self._data.items()))
'''

_BATCH_BACKEND = '''
class BackendError(Exception):
    """args is (code, key); code is 'apply failed', 'undo failed', 'exists', 'missing' or 'bad op'."""


class Backend:
    def __init__(self, initial=None, fail_apply=(), fail_undo=()):
        self.state, self.calls, self.journal = dict(initial or {}), [], []
        self.fail_apply, self.fail_undo = set(fail_apply), set(fail_undo)

    def _perform(self, name, op, failing):
        kind, key, value = op
        self.calls.append((name, op))
        if kind not in ('set', 'del'):
            raise BackendError('bad op', kind)
        if key in failing:
            raise BackendError(name + ' failed', key)
        adding = (kind == 'set') != (name == 'undo')
        if (key in self.state) == adding:
            raise BackendError('exists' if key in self.state else 'missing', key)
        if adding:
            self.state[key] = value
        else:
            del self.state[key]
        self.journal.append(op) if name == 'apply' else self.journal.remove(op)
        return 'created' if adding else 'deleted'

    def apply(self, op):
        return self._perform('apply', op, self.fail_apply)

    def undo(self, op):
        return self._perform('undo', op, self.fail_undo)
'''

_ACCOUNT_STORE = '''
class LedgerError(Exception):
    """args is (reason, name)."""


class AccountStore:
    """credit() and debit() each validate one operation: unknown account first,
    then the overdraft limit.  Every call is recorded in calls as (name, delta)."""

    def __init__(self, balances=None, limit=0):
        self.balance, self.limit, self.calls = dict(balances or {}), limit, []

    def _move(self, name, amount, sign):
        self.calls.append((name, sign * amount))
        if name not in self.balance:
            raise LedgerError('unknown account', name)
        if sign < 0 and self.balance[name] - amount < -self.limit:
            raise LedgerError('over limit', name)
        self.balance[name] += sign * amount
        return self.balance[name]

    def credit(self, name, amount):
        return self._move(name, amount, 1)

    def debit(self, name, amount):
        return self._move(name, amount, -1)
'''

_NODE_REGISTRY = '''
class GraphError(Exception):
    """args is (reason, detail)."""


class Registry:
    def __init__(self):
        self.inputs, self.users = {}, {}

    def define(self, name, inputs):
        if name in self.inputs:
            raise GraphError('duplicate', name)
        if name in inputs:
            raise GraphError('self', name)
        self.inputs[name] = list(inputs)
        for source in inputs:
            self.users.setdefault(source, []).append(name)
        return self
'''

SPECS = [
    _task(
        "work_commit", "unit_of_work",
        "只读依赖 slot_store 是容量固定的键值库 SlotStore：put / delete / get 每次只执行一个操作，失败抛 "
        "StoreError(reason, key)，多步之间不具备原子性。UnitOfWork 暂存修改，commit 时整体生效。要求：\n"
        "1）add(k, v)：k 在库中或已暂存则抛 KeyError(k)；update / delete：k 不在库中或已暂存则抛 KeyError(k)。"
        "三个方法都返回 self。\n"
        "2）commit()：先按暂存顺序删除，再按暂存顺序写入，返回执行的操作数。任一步失败：逆序撤回已执行的操作，"
        "库与提交前完全相同，原异常原样抛出（类型与 args 不变）；工作单元仍是打开状态，暂存保留，可再次提交。\n"
        "3）提交成功后关闭：此后 add / update / delete / commit / rollback 抛 RuntimeError('already committed')。\n"
        "4）rollback() 丢弃全部暂存并关闭，库不变；关闭后的这些调用抛 RuntimeError('rolled back')。\n"
        "5）pending() 返回 ('delete', k) 在前、('put', k, v) 在后的列表，各按暂存顺序；回滚后为 []。",
        '''
        from slot_store import StoreError


        class UnitOfWork:
            def __init__(self, store):
                self._store, self._state = store, 'open'
                self._puts, self._deletes = {}, {}

            def _guard(self):
                if self._state != 'open':
                    raise RuntimeError(self._state)

            def _stage(self, key, present):
                self._guard()
                busy = key in self._puts or key in self._deletes
                if busy or (key in self._store) != present:
                    raise KeyError(key)

            def add(self, key, value):
                self._stage(key, False)
                self._puts[key] = value
                return self

            def update(self, key, value):
                self._stage(key, True)
                self._puts[key] = value
                return self

            def delete(self, key):
                self._stage(key, True)
                self._deletes[key] = True
                return self

            def pending(self):
                puts = [('put', key, value) for key, value in self._puts.items()]
                return [('delete', key) for key in self._deletes] + puts

            def rollback(self):
                self._puts.clear()
                self._deletes.clear()

            def commit(self):
                self._guard()
                try:
                    for key, value in self._puts.items():
                        self._store.put(key, value)
                    for key in self._deletes:
                        self._store.delete(key)
                except StoreError as exc:
                    raise RuntimeError('commit failed') from exc
                return len(self._puts) + len(self._deletes)
        ''',
        '''
        class UnitOfWork:
            def __init__(self, store):
                self._store, self._state = store, 'open'
                self._puts, self._deletes = {}, {}

            def _guard(self):
                if self._state != 'open':
                    raise RuntimeError(self._state)

            def _stage(self, key, present):
                self._guard()
                busy = key in self._puts or key in self._deletes
                if busy or (key in self._store) != present:
                    raise KeyError(key)

            def add(self, key, value):
                self._stage(key, False)
                self._puts[key] = value
                return self

            def update(self, key, value):
                self._stage(key, True)
                self._puts[key] = value
                return self

            def delete(self, key):
                self._stage(key, True)
                self._deletes[key] = True
                return self

            def pending(self):
                puts = [('put', key, value) for key, value in self._puts.items()]
                return [('delete', key) for key in self._deletes] + puts

            def rollback(self):
                self._guard()
                self._puts.clear()
                self._deletes.clear()
                self._state = 'rolled back'

            def commit(self):
                self._guard()
                plan = [(key, None) for key in self._deletes] + list(self._puts.items())
                log = []
                try:
                    for key, value in plan:
                        old = self._store.get(key) if key in self._store else None
                        if key in self._deletes:
                            self._store.delete(key)
                        else:
                            self._store.put(key, value)
                        log.append((key, old))
                except Exception:
                    for key, old in reversed(log):
                        if old is None:
                            self._store.delete(key)
                        else:
                            self._store.put(key, old)
                    raise
                self._state = 'already committed'
                return len(plan)
        ''',
        '''
        from slot_store import SlotStore


        def test_public_interface():
            store = SlotStore(3, {'a': '1'})
            work = UnitOfWork(store)
            work.add('b', '2').update('a', '3')
            assert work.pending() == [('put', 'b', '2'), ('put', 'a', '3')]
            assert work.commit() == 2
            assert store.get('a') == '3' and store.get('b') == '2'
            assert UnitOfWork(store).add('c', '4').rollback() is None
            assert store.items() == {'a': '3', 'b': '2'}
        ''',
        {
            "commit": '''
                from slot_store import SlotStore, StoreError

                FULL = {'a': 'old a', 'x': 'old x', 'y': 'old y'}


                def test_failure_in_the_middle_then_retry():
                    store = SlotStore(3, FULL)
                    work = UnitOfWork(store).update('x', 'new x').delete('a').add('z', '1').add('b', '2')
                    with pytest.raises(StoreError) as caught:
                        work.commit()
                    assert type(caught.value) is StoreError and caught.value.args == ('full', 'b')
                    assert caught.value.__cause__ is None and store.items() == FULL
                    assert work.pending() == [('delete', 'a'), ('put', 'x', 'new x'), ('put', 'z', '1'),
                                              ('put', 'b', '2')]
                    work.delete('y')
                    assert work.commit() == 5 and store.items() == {'x': 'new x', 'z': '1', 'b': '2'}
                    for call in (work.commit, lambda: work.add('q', '1'), lambda: work.update('x', '9'),
                                 lambda: work.delete('x'), work.rollback):
                        with pytest.raises(RuntimeError, match='^already committed$'):
                            call()
                    assert store.items() == {'x': 'new x', 'z': '1', 'b': '2'}


                def test_key_removed_by_another_unit():
                    store = SlotStore(4, {'a': '1', 'b': '2', 'c': '3'})
                    work = UnitOfWork(store).delete('a').update('c', '30').delete('b')
                    assert UnitOfWork(store).delete('b').commit() == 1
                    with pytest.raises(StoreError) as caught:
                        work.commit()
                    assert caught.value.args == ('unknown key', 'b')
                    assert store.items() == {'a': '1', 'c': '3'}
                    assert work.rollback() is None
            ''',
            "stage": '''
                from slot_store import SlotStore


                def test_staging_rules():
                    store = SlotStore(4, {'a': '1', 'c': '3'})
                    work = UnitOfWork(store)
                    assert work.update('a', '2') is work and work.add('b', '4') is work
                    assert work.delete('c') is work
                    bad = (lambda: work.add('a', '9'), lambda: work.add('b', '9'), lambda: work.add('c', '9'),
                           lambda: work.update('a', '9'), lambda: work.update('d', '9'), lambda: work.delete('a'))
                    for call in bad:
                        with pytest.raises(KeyError):
                            call()
                    assert work.pending() == [('delete', 'c'), ('put', 'a', '2'), ('put', 'b', '4')]
                    assert store.items() == {'a': '1', 'c': '3'}
                    assert work.commit() == 3 and store.items() == {'a': '2', 'b': '4'}
            ''',
            "rollback": '''
                from slot_store import SlotStore


                def test_rollback_discards_and_closes():
                    store = SlotStore(4, {'a': '1'})
                    work = UnitOfWork(store).update('a', '2').add('b', '3')
                    assert work.rollback() is None and work.pending() == []
                    for call in (lambda: work.add('c', '4'), lambda: work.update('a', '5'), lambda: work.delete('a'),
                                 work.commit, work.rollback):
                        with pytest.raises(RuntimeError, match='^rolled back$'):
                            call()
                    assert store.items() == {'a': '1'}
            ''',
        },
        extra_files={"slot_store": _SLOT_STORE},
    ),
    _task(
        "work_abort", "unit_of_work",
        "只读依赖 slot_store 是容量固定的键值库 SlotStore：put / delete / get 每次只执行一个操作，失败抛 "
        "StoreError(reason, key)，多步之间不具备原子性。UnitOfWork 暂存修改，commit 时整体生效。要求：\n"
        "1）add(k, v)：k 在库中或已暂存则抛 KeyError(k)；update / delete：k 不在库中或已暂存则抛 KeyError(k)。"
        "三个方法都返回 self。\n"
        "2）commit()：先按暂存顺序删除，再按暂存顺序写入，返回执行的操作数。任一步失败：逆序撤回已执行的操作，"
        "库与提交前完全相同，原异常原样抛出（类型与 args 不变），并且该工作单元按已回滚处理：暂存丢弃并关闭。\n"
        "3）提交成功后关闭：此后 add / update / delete / commit / rollback 抛 RuntimeError('already committed')。\n"
        "4）rollback() 丢弃全部暂存并关闭，库不变。已回滚（含提交失败）后这些调用抛 RuntimeError('rolled back')。\n"
        "5）pending() 返回 ('delete', k) 在前、('put', k, v) 在后的列表，各按暂存顺序；回滚后为 []。",
        '''
        class UnitOfWork:
            def __init__(self, store):
                self._store, self._state = store, 'open'
                self._puts, self._deletes = {}, {}

            def _guard(self):
                if self._state != 'open':
                    raise RuntimeError(self._state)

            def _stage(self, key, present):
                busy = key in self._puts
                if busy or (key in self._store) != present:
                    raise KeyError(key)
                self._guard()

            def add(self, key, value):
                self._stage(key, False)
                self._puts[key] = value
                return self

            def update(self, key, value):
                self._stage(key, True)
                self._puts[key] = value
                return self

            def delete(self, key):
                self._stage(key, True)
                self._deletes[key] = True
                return self

            def pending(self):
                puts = [('put', key, value) for key, value in self._puts.items()]
                return [('delete', key) for key in self._deletes] + puts

            def _close(self):
                self._puts.clear()
                self._deletes.clear()
                self._state = 'rolled back'

            def rollback(self):
                self._close()

            def commit(self):
                self._guard()
                plan = [(key, None) for key in self._deletes] + list(self._puts.items())
                for key, value in plan:
                    if key in self._deletes:
                        self._store.delete(key)
                    else:
                        self._store.put(key, value)
                self._state = 'already committed'
                return len(plan)
        ''',
        '''
        class UnitOfWork:
            def __init__(self, store):
                self._store, self._state = store, 'open'
                self._puts, self._deletes = {}, {}

            def _guard(self):
                if self._state != 'open':
                    raise RuntimeError(self._state)

            def _stage(self, key, present):
                self._guard()
                busy = key in self._puts or key in self._deletes
                if busy or (key in self._store) != present:
                    raise KeyError(key)

            def add(self, key, value):
                self._stage(key, False)
                self._puts[key] = value
                return self

            def update(self, key, value):
                self._stage(key, True)
                self._puts[key] = value
                return self

            def delete(self, key):
                self._stage(key, True)
                self._deletes[key] = True
                return self

            def pending(self):
                puts = [('put', key, value) for key, value in self._puts.items()]
                return [('delete', key) for key in self._deletes] + puts

            def _close(self):
                self._puts.clear()
                self._deletes.clear()
                self._state = 'rolled back'

            def rollback(self):
                self._guard()
                self._close()

            def commit(self):
                self._guard()
                plan = [(key, None) for key in self._deletes] + list(self._puts.items())
                log = []
                try:
                    for key, value in plan:
                        old = self._store.get(key) if key in self._store else None
                        if key in self._deletes:
                            self._store.delete(key)
                        else:
                            self._store.put(key, value)
                        log.append((key, old))
                except Exception:
                    self._close()
                    for key, old in reversed(log):
                        if old is None:
                            self._store.delete(key)
                        else:
                            self._store.put(key, old)
                    raise
                self._state = 'already committed'
                return len(plan)
        ''',
        '''
        from slot_store import SlotStore


        def test_public_interface():
            store = SlotStore(3, {'a': '1', 'b': '2'})
            work = UnitOfWork(store).delete('b').update('a', '10').add('c', '3')
            assert work.commit() == 3
            assert store.items() == {'a': '10', 'c': '3'}
            work = UnitOfWork(store)
            work.update('a', '20')
            assert work.pending() == [('put', 'a', '20')]
            assert work.rollback() is None and store.items() == {'a': '10', 'c': '3'}
        ''',
        {
            "abort": '''
                from slot_store import SlotStore, StoreError

                FULL = {'a': 'old a', 'x': 'old x', 'y': 'old y'}


                def test_failed_commit_undoes_and_closes():
                    store = SlotStore(3, FULL)
                    work = UnitOfWork(store).delete('a').update('x', 'new x').add('z', '1').add('b', '2')
                    with pytest.raises(StoreError) as caught:
                        work.commit()
                    assert type(caught.value) is StoreError and caught.value.args == ('full', 'b')
                    assert caught.value.__cause__ is None and store.items() == FULL
                    assert work.pending() == []
                    for call in (work.commit, work.rollback, lambda: work.add('c', '1'), lambda: work.update('x', '2'),
                                 lambda: work.delete('x')):
                        with pytest.raises(RuntimeError, match='^rolled back$'):
                            call()
                    assert store.items() == FULL


                def test_key_removed_by_another_unit():
                    store = SlotStore(4, {'a': '1', 'b': '2', 'c': '3'})
                    work = UnitOfWork(store).delete('a').update('c', '30').delete('b')
                    assert UnitOfWork(store).delete('b').commit() == 1
                    with pytest.raises(StoreError) as caught:
                        work.commit()
                    assert caught.value.args == ('unknown key', 'b') and store.items() == {'a': '1', 'c': '3'}
                    with pytest.raises(RuntimeError, match='^rolled back$'):
                        work.commit()
            ''',
            "stage": '''
                from slot_store import SlotStore


                def test_staging_rules_and_rollback():
                    store = SlotStore(4, {'a': '1', 'c': '3'})
                    work = UnitOfWork(store)
                    work.delete('c').update('a', '2').add('b', '4')
                    bad = (lambda: work.update('c', '9'), lambda: work.delete('c'), lambda: work.add('c', '9'),
                           lambda: work.add('a', '9'), lambda: work.update('b', '9'), lambda: work.delete('b'))
                    for call in bad:
                        with pytest.raises(KeyError):
                            call()
                    assert work.pending() == [('delete', 'c'), ('put', 'a', '2'), ('put', 'b', '4')]
                    assert work.rollback() is None and store.items() == {'a': '1', 'c': '3'}
                    for call in (lambda: work.add('a', '9'), lambda: work.add('d', '9'), lambda: work.update('a', '9'),
                                 lambda: work.delete('a'), work.commit, work.rollback):
                        with pytest.raises(RuntimeError, match='^rolled back$'):
                            call()
                    assert work.pending() == [] and store.items() == {'a': '1', 'c': '3'}


                def test_committed_unit_rejects_rollback():
                    store = SlotStore(4, {'a': '1'})
                    work = UnitOfWork(store).update('a', '2')
                    assert work.commit() == 1 and store.items() == {'a': '2'}
                    for call in (work.commit, work.rollback, lambda: work.add('b', '3'), lambda: work.update('a', '3'),
                                 lambda: work.delete('a')):
                        with pytest.raises(RuntimeError, match='^already committed$'):
                            call()
                    assert store.items() == {'a': '2'}
            ''',
        },
        extra_files={"slot_store": _SLOT_STORE},
    ),
    _task(
        "undo_batch", "batch_apply",
        "只读依赖 batch_backend：Backend.apply(op) 执行一个操作 (kind, key, value)，kind 为 'set'（key 必须不存在）"
        "或 'del'（key 必须存在），undo(op) 执行其逆操作；失败抛 BackendError(code, key)，每次调用记入 calls。"
        "Batch 用它把多个操作作为整体执行。要求：\n"
        "1）apply_all(ops) 按顺序调用 backend.apply；全部成功时返回各结果组成的列表，remaining 为 []。\n"
        "2）某个 apply 抛异常时：对它之前已成功的操作按相反顺序调用 backend.undo，然后原样抛出该异常对象"
        "（类型与 args 不变）；失败的那个操作不做 undo。\n"
        "3）某个 undo 失败不能掩盖原异常：继续 undo 其余操作，最后仍抛出原异常对象，并把第一个（按 undo 执行顺序）"
        "失败的异常设为它的 __cause__；没有 undo 失败时 __cause__ 为 None。\n"
        "4）remaining 列出 undo 失败、仍然生效的操作，按执行顺序排列；每次 apply_all 开始时清空。\n"
        "5）Batch 不预先检查操作，合法性完全由后端判断。",
        '''
        class Batch:
            def __init__(self, backend):
                self.backend = backend
                self.done, self.remaining = [], []

            def apply_all(self, ops):
                results = []
                try:
                    for op in ops:
                        self.done.append(op)
                        results.append(self.backend.apply(op))
                except Exception:
                    for op in self.done:
                        self.backend.undo(op)
                    raise
                return results
        ''',
        '''
        class Batch:
            def __init__(self, backend):
                self.backend = backend
                self.remaining = []

            def apply_all(self, ops):
                done, results, self.remaining = [], [], []
                try:
                    for op in ops:
                        results.append(self.backend.apply(op))
                        done.append(op)
                except Exception as original:
                    failure = None
                    for op in reversed(done):
                        try:
                            self.backend.undo(op)
                        except Exception as exc:
                            self.remaining.insert(0, op)
                            failure = failure or exc
                    if failure is not None:
                        raise original from failure
                    raise
                return results
        ''',
        '''
        from batch_backend import Backend


        def test_public_interface():
            backend = Backend({'b': 2})
            batch = Batch(backend)
            ops = [('set', 'a', 1), ('del', 'b', 2), ('set', 'b', 3)]
            assert batch.apply_all(ops) == ['created', 'deleted', 'created']
            assert backend.state == {'a': 1, 'b': 3} and batch.remaining == []
        ''',
        {
            "clean": '''
                from batch_backend import Backend, BackendError


                def test_clean_undo_and_replay():
                    ops = [('set', 'a', 1), ('del', 'a', 1), ('set', 'b', 2), ('set', 'c', 3)]
                    backend = Backend(fail_apply={'c'})
                    batch = Batch(backend)
                    with pytest.raises(BackendError) as caught:
                        batch.apply_all(ops)
                    assert type(caught.value) is BackendError and caught.value.args == ('apply failed', 'c')
                    assert caught.value.__cause__ is None
                    assert backend.calls == [('apply', op) for op in ops] + [('undo', op) for op in reversed(ops[:3])]
                    assert backend.state == {} and batch.remaining == []
                    backend.fail_apply.clear()
                    assert batch.apply_all(ops) == ['created', 'deleted', 'created', 'created']
                    assert backend.state == {'b': 2, 'c': 3}
            ''',
            "partial": '''
                from batch_backend import Backend, BackendError


                def test_failing_undo_keeps_the_original_and_reports_leftovers():
                    ops = [('set', 'a', 1), ('set', 'b', 2), ('set', 'c', 3), ('set', 'd', 4), ('set', 'e', 5)]
                    backend = Backend(fail_apply={'e'}, fail_undo={'b', 'c'})
                    batch = Batch(backend)
                    with pytest.raises(BackendError) as caught:
                        batch.apply_all(ops)
                    original = caught.value
                    assert type(original) is BackendError and original.args == ('apply failed', 'e')
                    assert type(original.__cause__) is BackendError
                    assert original.__cause__.args == ('undo failed', 'c')
                    assert batch.remaining == ops[1:3] and backend.state == {'b': 2, 'c': 3}
                    assert backend.calls[5:] == [('undo', op) for op in reversed(ops[:4])]
                    assert batch.apply_all([('del', 'b', 2), ('del', 'c', 3)]) == ['deleted', 'deleted']
                    assert batch.remaining == [] and backend.state == {}
            ''',
            "backend": '''
                from batch_backend import Backend, BackendError


                def test_first_operation_fails_and_the_backend_decides():
                    backend = Backend({'b': 2}, fail_apply={'a'})
                    batch = Batch(backend)
                    with pytest.raises(BackendError) as caught:
                        batch.apply_all([('set', 'b', 9)])
                    assert caught.value.args == ('exists', 'b') and caught.value.__cause__ is None
                    assert backend.calls == [('apply', ('set', 'b', 9))] and backend.state == {'b': 2}
                    ops = [('del', 'b', 2), ('move', 'a', 1)]
                    with pytest.raises(BackendError) as caught:
                        batch.apply_all(ops)
                    assert caught.value.args == ('bad op', 'move') and caught.value.__cause__ is None
                    assert backend.calls[1:] == [('apply', op) for op in ops] + [('undo', ('del', 'b', 2))]
                    assert backend.state == {'b': 2} and batch.remaining == []
            ''',
        },
        extra_files={"batch_backend": _BATCH_BACKEND},
    ),
    _task(
        "recover_batch", "batch_apply",
        "只读依赖 batch_backend：Backend.apply(op) 执行一个操作 (kind, key, value)，kind 为 'set'（key 必须不存在）"
        "或 'del'（key 必须存在），undo(op) 执行其逆操作；成功的 apply 把 op 追加到 journal，成功的 undo 把它从 "
        "journal 移除；失败抛 BackendError(code, key)，每次调用记入 calls。Batch 用它把多个操作作为整体执行。要求：\n"
        "1）apply_all(ops) 先恢复：对 journal 中遗留的操作（此前失败且未能撤销的）按相反顺序调用 undo；若有 undo 失败，"
        "继续尝试其余的，最后抛出第一个失败的异常，并且不执行本次的 ops。\n"
        "2）然后按顺序调用 backend.apply；全部成功时清空 journal，返回各结果组成的列表。\n"
        "3）某个 apply 抛异常时：对 journal 中本次已执行的操作按相反顺序 undo；全部成功则原样抛出该异常对象"
        "（__cause__ 为 None）。\n"
        "4）有 undo 失败时不能掩盖原异常：继续 undo 其余操作，最后仍抛出原异常对象，并把第一个（按 undo 执行顺序）"
        "失败的异常设为它的 __cause__；撤销失败的操作留在 journal 中。",
        '''
        class Batch:
            def __init__(self, backend):
                self.backend = backend

            def apply_all(self, ops):
                results = []
                try:
                    for op in ops:
                        results.append(self.backend.apply(op))
                except Exception as original:
                    for op in list(self.backend.journal):
                        try:
                            self.backend.undo(op)
                        except Exception:
                            break
                    raise original
                return results
        ''',
        '''
        class Batch:
            def __init__(self, backend):
                self.backend = backend

            def _unwind(self, ops):
                failure = None
                for op in reversed(ops):
                    try:
                        self.backend.undo(op)
                    except Exception as exc:
                        failure = failure or exc
                return failure

            def apply_all(self, ops):
                failure = self._unwind(list(self.backend.journal))
                if failure is not None:
                    raise failure
                results = []
                try:
                    for op in ops:
                        results.append(self.backend.apply(op))
                except Exception as original:
                    raise original from self._unwind(list(self.backend.journal))
                self.backend.journal.clear()
                return results
        ''',
        '''
        from batch_backend import Backend


        def test_public_interface():
            backend = Backend({'b': 2})
            batch = Batch(backend)
            ops = [('set', 'a', 1), ('del', 'b', 2), ('set', 'b', 3)]
            assert batch.apply_all(ops) == ['created', 'deleted', 'created']
            assert backend.state == {'a': 1, 'b': 3}
        ''',
        {
            "recover": '''
                from batch_backend import Backend, BackendError

                OPS = [('set', 'a', 1), ('set', 'c', 3), ('del', 'x', 0)]


                def test_leftover_is_recovered_by_the_next_batch():
                    backend = Backend({'b': 2}, fail_undo={'a'})
                    with pytest.raises(BackendError) as caught:
                        Batch(backend).apply_all(OPS)
                    original = caught.value
                    assert type(original) is BackendError and original.args == ('missing', 'x')
                    assert type(original.__cause__) is BackendError
                    assert original.__cause__.args == ('undo failed', 'a')
                    assert backend.journal == OPS[:1] and backend.state == {'a': 1, 'b': 2}
                    assert backend.calls == [('apply', op) for op in OPS] + [('undo', OPS[1]), ('undo', OPS[0])]
                    with pytest.raises(BackendError) as caught:
                        Batch(backend).apply_all([('set', 'c', 3)])
                    assert caught.value.args == ('undo failed', 'a') and caught.value.__cause__ is None
                    assert backend.calls[5:] == [('undo', OPS[0])] and backend.journal == OPS[:1]
                    backend.fail_undo.clear()
                    assert Batch(backend).apply_all([('set', 'c', 3)]) == ['created']
                    assert backend.calls[6:] == [('undo', OPS[0]), ('apply', ('set', 'c', 3))]
                    assert backend.state == {'b': 2, 'c': 3} and backend.journal == []
            ''',
            "unwind": '''
                from batch_backend import Backend, BackendError


                def test_committed_operations_are_not_undone_by_a_later_failure():
                    backend = Backend()
                    batch = Batch(backend)
                    assert batch.apply_all([('set', 'a', 1)]) == ['created']
                    ops = [('set', 'b', 2), ('del', 'y', 0)]
                    with pytest.raises(BackendError) as caught:
                        batch.apply_all(ops)
                    assert caught.value.args == ('missing', 'y') and caught.value.__cause__ is None
                    head = [('apply', ('set', 'a', 1))] + [('apply', op) for op in ops]
                    assert backend.calls == head + [('undo', ops[0])]
                    assert backend.state == {'a': 1} and backend.journal == []


                def test_unwind_continues_and_reports_the_first_undo_failure():
                    ops = [('set', 'a', 1), ('set', 'b', 2), ('set', 'c', 3), ('del', 'x', 0)]
                    backend = Backend(fail_undo={'b', 'c'})
                    with pytest.raises(BackendError) as caught:
                        Batch(backend).apply_all(ops)
                    assert caught.value.args == ('missing', 'x')
                    assert caught.value.__cause__.args == ('undo failed', 'c')
                    assert backend.calls[4:] == [('undo', op) for op in reversed(ops[:3])]
                    assert backend.state == {'b': 2, 'c': 3} and backend.journal == ops[1:3]


                def test_unwind_runs_in_reverse_order():
                    ops = [('set', 'a', 1), ('del', 'a', 1), ('del', 'x', 0)]
                    backend = Backend()
                    with pytest.raises(BackendError) as caught:
                        Batch(backend).apply_all(ops)
                    assert caught.value.args == ('missing', 'x') and caught.value.__cause__ is None
                    assert backend.calls == [('apply', op) for op in ops] + [('undo', ops[1]), ('undo', ops[0])]
                    assert backend.state == {} and backend.journal == []
            ''',
        },
        extra_files={"batch_backend": _BATCH_BACKEND},
    ),
    _task(
        "ledger_post", "ledger_journal",
        "只读依赖 account_store：AccountStore 保存余额，credit(name, amount) / debit(name, amount) 每次只执行一个操作，"
        "按固定顺序校验：账户不存在抛 LedgerError('unknown account', name)；借方使余额低于 -limit 抛 "
        "LedgerError('over limit', name)。每次调用都记入 calls，格式为 (name, 变动量)。"
        "Ledger 在其上实现复式日记账，posted 是有序字典 key → 分录列表。要求：\n"
        "1）post(key, entries)：entries 为 (账户, 金额) 列表，正数入账、负数出账。校验顺序固定：key 已入账时，"
        "entries 相同则返回当初的序号且不做任何修改，不同则抛 LedgerError('key reused', key)；"
        "然后金额之和必须为 0，否则抛 LedgerError('unbalanced', key)。\n"
        "2）校验通过后按顺序执行分录；任一失败时对已执行的按相反顺序冲正，余额与 posted 都和 post 之前相同，"
        "原异常原样抛出（类型与 args 不变）。\n"
        "3）全部成功才把 key 记入 posted，返回序号（从 1 起）；失败的 post 不占序号，key 仍可使用。\n"
        "4）reverse(key, as_key)：key 不在 posted 抛 LedgerError('unknown key', key)；已冲正抛 "
        "LedgerError('already reversed', key)；否则以 as_key 为 key、把该条目各金额取反（顺序不变）按 post 入账，"
        "成功后把 key 加入 reversals，返回序号。",
        '''
        from account_store import LedgerError


        class Ledger:
            def __init__(self, store):
                self.store, self.posted, self.reversals = store, {}, set()

            def _move(self, name, amount):
                if amount > 0:
                    self.store.credit(name, amount)
                else:
                    self.store.debit(name, -amount)

            def post(self, key, lines):
                lines = [tuple(line) for line in lines]
                if sum(amount for _, amount in lines) != 0:
                    raise LedgerError('unbalanced', key)
                if key in self.posted:
                    if self.posted[key] != lines:
                        raise LedgerError('key reused', key)
                    return list(self.posted).index(key) + 1
                self.posted[key] = lines
                for name, amount in lines:
                    self._move(name, amount)
                return len(self.posted)

            def reverse(self, key, as_key):
                if key not in self.posted:
                    raise LedgerError('unknown key', key)
                if key in self.reversals:
                    raise LedgerError('already reversed', key)
                self.reversals.add(key)
                return self.post(as_key, [(name, -amount) for name, amount in reversed(self.posted[key])])
        ''',
        '''
        from account_store import LedgerError


        class Ledger:
            def __init__(self, store):
                self.store, self.posted, self.reversals = store, {}, set()

            def _move(self, name, amount):
                if amount > 0:
                    self.store.credit(name, amount)
                else:
                    self.store.debit(name, -amount)

            def post(self, key, lines):
                lines = [tuple(line) for line in lines]
                if key in self.posted:
                    if self.posted[key] != lines:
                        raise LedgerError('key reused', key)
                    return list(self.posted).index(key) + 1
                if sum(amount for _, amount in lines) != 0:
                    raise LedgerError('unbalanced', key)
                applied = []
                try:
                    for name, amount in lines:
                        self._move(name, amount)
                        applied.append((name, amount))
                except Exception:
                    for name, amount in reversed(applied):
                        self._move(name, -amount)
                    raise
                self.posted[key] = lines
                return len(self.posted)

            def reverse(self, key, as_key):
                if key not in self.posted:
                    raise LedgerError('unknown key', key)
                if key in self.reversals:
                    raise LedgerError('already reversed', key)
                seq = self.post(as_key, [(name, -amount) for name, amount in self.posted[key]])
                self.reversals.add(key)
                return seq
        ''',
        '''
        from account_store import AccountStore


        def test_public_interface():
            store = AccountStore({'cash': 100, 'bank': 50})
            ledger = Ledger(store)
            assert ledger.post('k1', [('cash', -40), ('bank', 40)]) == 1
            assert store.balance == {'cash': 60, 'bank': 90}
            assert ledger.reverse('k1', 'r1') == 2
            assert store.balance == {'cash': 100, 'bank': 50}
        ''',
        {
            "post": '''
                from account_store import AccountStore, LedgerError


                def test_failed_post_changes_nothing_and_the_key_stays_free():
                    store = AccountStore({'cash': 100, 'bank': 50})
                    ledger = Ledger(store)
                    lines = [('cash', 30), ('bank', 20), ('bank', -90), ('cash', 40)]
                    with pytest.raises(LedgerError) as caught:
                        ledger.post('k1', lines)
                    assert type(caught.value) is LedgerError and caught.value.args == ('over limit', 'bank')
                    assert store.balance == {'cash': 100, 'bank': 50} and ledger.posted == {}
                    assert store.calls == [('cash', 30), ('bank', 20), ('bank', -90), ('bank', -20), ('cash', -30)]
                    store.credit('bank', 100)
                    assert ledger.post('k1', [[name, amount] for name, amount in lines]) == 1
                    assert store.balance == {'cash': 170, 'bank': 80}
                    assert ledger.posted == {'k1': lines}


                def test_replay_is_idempotent_and_the_validation_order_is_fixed():
                    store = AccountStore({'cash': 100, 'bank': 50})
                    ledger = Ledger(store)
                    assert ledger.post('k1', [('cash', -40), ('bank', 40)]) == 1
                    assert ledger.post('k1', [('cash', -40), ('bank', 40)]) == 1
                    assert store.balance == {'cash': 60, 'bank': 90}
                    cases = ((('k1', [('cash', -40)]), 'key reused'),
                             (('k2', [('cash', -1), ('bank', 2)]), 'unbalanced'),
                             (('k3', [('cash', -10), ('ghost', 10)]), 'unknown account'))
                    for (key, lines), reason in cases:
                        with pytest.raises(LedgerError) as caught:
                            ledger.post(key, lines)
                        assert caught.value.args == (reason, 'ghost' if reason == 'unknown account' else key)
                    assert store.balance == {'cash': 60, 'bank': 90} and list(ledger.posted) == ['k1']
            ''',
            "reverse": '''
                from account_store import AccountStore, LedgerError


                def test_reversal_restores_balances_and_appends_to_the_audit_order():
                    store = AccountStore({'cash': 100, 'bank': 50})
                    ledger = Ledger(store)
                    ledger.post('k1', [('cash', -40), ('bank', 40)])
                    ledger.post('k2', [('bank', -10), ('cash', 10)])
                    assert ledger.reverse('k1', 'r1') == 3
                    assert store.balance == {'cash': 110, 'bank': 40}
                    assert list(ledger.posted.items()) == [
                        ('k1', [('cash', -40), ('bank', 40)]), ('k2', [('bank', -10), ('cash', 10)]),
                        ('r1', [('cash', 40), ('bank', -40)])]
                    for key, as_key, reason in (('k1', 'r9', 'already reversed'), ('zz', 'r9', 'unknown key')):
                        with pytest.raises(LedgerError) as caught:
                            ledger.reverse(key, as_key)
                        assert caught.value.args == (reason, key)
                    with pytest.raises(LedgerError) as caught:
                        ledger.reverse('k2', 'r1')
                    assert caught.value.args == ('key reused', 'r1') and 'k2' not in ledger.reversals
                    assert list(ledger.posted) == ['k1', 'k2', 'r1']
                    assert ledger.reverse('k2', 'r2') == 4 and store.balance == {'cash': 100, 'bank': 50}


                def test_failed_reversal_keeps_the_posting_reversible():
                    store = AccountStore({'cash': 10, 'bank': 0, 'vault': 0})
                    ledger = Ledger(store)
                    ledger.post('k1', [('cash', -10), ('bank', 10)])
                    ledger.post('k2', [('bank', -10), ('vault', 10)])
                    with pytest.raises(LedgerError) as caught:
                        ledger.reverse('k1', 'r1')
                    assert caught.value.args == ('over limit', 'bank')
                    assert store.balance == {'cash': 0, 'bank': 0, 'vault': 10}
                    assert list(ledger.posted) == ['k1', 'k2'] and ledger.reversals == set()
                    store.credit('bank', 10)
                    assert ledger.reverse('k1', 'r1') == 3
                    assert store.balance == {'cash': 10, 'bank': 0, 'vault': 10}
            ''',
        },
        extra_files={"account_store": _ACCOUNT_STORE},
    ),
    _task(
        "teller_transfer", "ledger_journal",
        "只读依赖 account_store：AccountStore 保存余额，credit(name, amount) / debit(name, amount) 每次只执行一个操作："
        "账户不存在抛 LedgerError('unknown account', name)，借方使余额低于 -limit 抛 LedgerError('over limit', name)，"
        "每次调用都记入 calls，格式为 (name, 变动量)。Teller 在其上实现带幂等键的转账，history 是已入账的 key 列表，"
        "receipts 是 key → 回执。要求：\n"
        "1）transfer(key, src, dst, amount) 按固定顺序校验：key 已入账时，(src, dst, amount) 与当初相同则返回当初的回执"
        "且不做任何修改，不同则抛 LedgerError('key reused', key)；amount 必须是正整数，否则抛 "
        "LedgerError('bad amount', amount)；src 与 dst 相同抛 LedgerError('same account', src)。\n"
        "2）然后先 debit(src, amount) 再 credit(dst, amount)；后一步失败时把 src 的钱退回，余额、history、receipts "
        "都与调用前相同，原异常原样抛出（类型与 args 不变）。\n"
        "3）全部成功才把 key 追加到 history，回执为 (src, dst, amount, 序号)，序号从 1 起。\n"
        "4）reverse(key, as_key)：key 不在 receipts 抛 LedgerError('unknown key', key)；已冲正抛 "
        "LedgerError('already reversed', key)；否则以 as_key 为 key 从 dst 转回 src 同样的金额，"
        "成功后把 key 加入 reversals，返回回执。",
        '''
        from account_store import LedgerError


        class Teller:
            def __init__(self, store):
                self.store, self.history, self.receipts, self.reversals = store, [], {}, set()

            def transfer(self, key, src, dst, amount):
                if src == dst:
                    raise LedgerError('same account', src)
                if key in self.receipts:
                    return self.receipts[key]
                if type(amount) is not int or amount <= 0:
                    raise LedgerError('bad amount', amount)
                self.history.append(key)
                self.receipts[key] = (src, dst, amount, len(self.history))
                self.store.credit(dst, amount)
                self.store.debit(src, amount)
                return self.receipts[key]

            def reverse(self, key, as_key):
                if key not in self.receipts:
                    raise LedgerError('unknown key', key)
                src, dst, amount, _ = self.receipts[key]
                receipt = self.transfer(as_key, dst, src, amount)
                self.reversals.add(key)
                return receipt
        ''',
        '''
        from account_store import LedgerError


        class Teller:
            def __init__(self, store):
                self.store, self.history, self.receipts, self.reversals = store, [], {}, set()

            def transfer(self, key, src, dst, amount):
                if key in self.receipts:
                    if self.receipts[key][:3] != (src, dst, amount):
                        raise LedgerError('key reused', key)
                    return self.receipts[key]
                if type(amount) is not int or amount <= 0:
                    raise LedgerError('bad amount', amount)
                if src == dst:
                    raise LedgerError('same account', src)
                self.store.debit(src, amount)
                try:
                    self.store.credit(dst, amount)
                except Exception:
                    self.store.credit(src, amount)
                    raise
                self.history.append(key)
                self.receipts[key] = (src, dst, amount, len(self.history))
                return self.receipts[key]

            def reverse(self, key, as_key):
                if key not in self.receipts:
                    raise LedgerError('unknown key', key)
                if key in self.reversals:
                    raise LedgerError('already reversed', key)
                src, dst, amount, _ = self.receipts[key]
                receipt = self.transfer(as_key, dst, src, amount)
                self.reversals.add(key)
                return receipt
        ''',
        '''
        from account_store import AccountStore


        def test_public_interface():
            store = AccountStore({'a': 100, 'b': 50})
            teller = Teller(store)
            assert teller.transfer('k1', 'a', 'b', 30) == ('a', 'b', 30, 1)
            assert store.balance == {'a': 70, 'b': 80}
            assert teller.reverse('k1', 'r1') == ('b', 'a', 30, 2)
            assert store.balance == {'a': 100, 'b': 50}
        ''',
        {
            "order": '''
                from account_store import AccountStore, LedgerError


                def test_replay_is_idempotent_and_the_validation_order_is_fixed():
                    store = AccountStore({'a': 100, 'b': 50})
                    teller = Teller(store)
                    assert teller.transfer('k1', 'a', 'b', 30) == ('a', 'b', 30, 1)
                    assert teller.transfer('k1', 'a', 'b', 30) == ('a', 'b', 30, 1)
                    assert store.balance == {'a': 70, 'b': 80} and teller.history == ['k1']
                    cases = ((('k1', 'a', 'c', 30), 'key reused', 'k1'), (('k2', 'b', 'b', 5), 'same account', 'b'),
                             (('k3', 'a', 'b', 0), 'bad amount', 0), (('k4', 'a', 'b', 'x'), 'bad amount', 'x'),
                             (('k5', 'b', 'b', 0), 'bad amount', 0), (('k1', 'a', 'a', 5), 'key reused', 'k1'))
                    for (key, src, dst, amount), reason, detail in cases:
                        with pytest.raises(LedgerError) as caught:
                            teller.transfer(key, src, dst, amount)
                        assert caught.value.args == (reason, detail)
                    assert store.balance == {'a': 70, 'b': 80} and teller.history == ['k1']
            ''',
            "refund": '''
                from account_store import AccountStore, LedgerError


                def test_failure_after_the_debit_refunds_and_frees_the_key():
                    store = AccountStore({'a': 100, 'b': 50})
                    teller = Teller(store)
                    with pytest.raises(LedgerError) as caught:
                        teller.transfer('k1', 'a', 'ghost', 30)
                    assert type(caught.value) is LedgerError and caught.value.args == ('unknown account', 'ghost')
                    assert store.balance == {'a': 100, 'b': 50} and teller.history == []
                    assert store.calls == [('a', -30), ('ghost', 30), ('a', 30)] and teller.receipts == {}
                    assert teller.transfer('k1', 'a', 'b', 30) == ('a', 'b', 30, 1)
                    assert store.balance == {'a': 70, 'b': 80}
                    with pytest.raises(LedgerError) as caught:
                        teller.transfer('k2', 'a', 'b', 500)
                    assert caught.value.args == ('over limit', 'a')
                    assert store.balance == {'a': 70, 'b': 80} and teller.history == ['k1']
            ''',
            "reverse": '''
                from account_store import AccountStore, LedgerError


                def test_failed_reversal_can_be_retried_but_not_repeated():
                    store = AccountStore({'a': 100, 'b': 50})
                    teller = Teller(store)
                    teller.transfer('k1', 'a', 'b', 30)
                    teller.transfer('k2', 'b', 'a', 80)
                    assert store.balance == {'a': 150, 'b': 0}
                    with pytest.raises(LedgerError) as caught:
                        teller.reverse('k1', 'r1')
                    assert caught.value.args == ('over limit', 'b')
                    assert store.balance == {'a': 150, 'b': 0} and teller.history == ['k1', 'k2']
                    with pytest.raises(LedgerError) as caught:
                        teller.reverse('zz', 'r9')
                    assert caught.value.args == ('unknown key', 'zz')
                    store.credit('b', 100)
                    assert teller.reverse('k1', 'r1') == ('b', 'a', 30, 3)
                    assert store.balance == {'a': 180, 'b': 70} and teller.history == ['k1', 'k2', 'r1']
                    with pytest.raises(LedgerError) as caught:
                        teller.reverse('k1', 'r9')
                    assert caught.value.args == ('already reversed', 'k1')
                    assert store.balance == {'a': 180, 'b': 70} and teller.history == ['k1', 'k2', 'r1']
            ''',
        },
        extra_files={"account_store": _ACCOUNT_STORE},
    ),
    _task(
        "replan_graph", "graph_replan",
        "只读依赖 node_registry：Registry.inputs[name] 是 name 的输入列表（可以是后定义的节点或外部输入），"
        "Registry.users[source] 是依赖 source 的节点列表；重名抛 GraphError('duplicate', name)，"
        "依赖自己抛 GraphError('self', name)；它不检查环。Engine 在其上做增量求值：values 是当前值，"
        "dirty 是待重算节点的集合。要求：\n"
        "1）define(name, inputs, fn)：fn 依次接收各输入的值；name 成为待重算节点。\n"
        "2）set(name, value)：name 若是已定义的节点则抛 GraphError('not an input', name)；否则写入该外部输入的值，"
        "并把直接或间接依赖 name 的节点全部标为待重算。\n"
        "3）recompute()：把待重算节点排成拓扑序（互不依赖的按名字字典序，与标记先后无关），依次用 fn 求值；"
        "全部成功后更新 values、清空 dirty，返回求值顺序列表。\n"
        "4）求值前先检查环：待重算节点之间有环则抛 GraphError('cycle', 描述)，不做任何求值，values 与 dirty 不变。"
        "描述为环上的名字从最小的开始、沿“被依赖 → 依赖者”方向用 '->' 连接，末尾再写一次起点，如 'b->c->b'。\n"
        "5）某个 fn 抛异常时 values 与 dirty 都与调用前完全相同，原异常原样抛出。",
        '''
        from node_registry import GraphError


        class Engine:
            def __init__(self, registry):
                self.graph, self.fns, self.values, self.dirty = registry, {}, {}, set()

            def define(self, name, inputs, fn):
                self.graph.define(name, inputs)
                self.fns[name] = fn
                self.dirty.add(name)

            def set(self, name, value):
                if name in self.fns:
                    raise GraphError('not an input', name)
                self.values[name] = value
                self.dirty.update(self.graph.users.get(name, []))

            def recompute(self):
                order, seen = [], set()

                def visit(name):
                    if name not in seen:
                        seen.add(name)
                        for source in self.graph.inputs[name]:
                            if source in self.dirty:
                                visit(source)
                        order.append(name)

                for name in sorted(self.dirty):
                    visit(name)
                for name in order:
                    self.values[name] = self.fns[name](*[self.values[s] for s in self.graph.inputs[name]])
                    self.dirty.discard(name)
                return order
        ''',
        '''
        from node_registry import GraphError


        class Engine:
            def __init__(self, registry):
                self.graph, self.fns, self.values, self.dirty = registry, {}, {}, set()

            def define(self, name, inputs, fn):
                self.graph.define(name, inputs)
                self.fns[name] = fn
                self._mark(name)

            def set(self, name, value):
                if name in self.fns:
                    raise GraphError('not an input', name)
                self.values[name] = value
                for user in self.graph.users.get(name, []):
                    self._mark(user)

            def _mark(self, name):
                pending = [name]
                while pending:
                    node = pending.pop()
                    if node not in self.dirty:
                        self.dirty.add(node)
                        pending.extend(self.graph.users.get(node, []))

            def _plan(self):
                need = {name: [s for s in self.graph.inputs[name] if s in self.dirty] for name in self.dirty}
                ready = sorted(name for name, sources in need.items() if not sources)
                order = []
                while ready:
                    name = ready.pop(0)
                    order.append(name)
                    for user in self.graph.users.get(name, []):
                        if user in need and name in need[user]:
                            need[user].remove(name)
                            if not need[user]:
                                ready.append(user)
                    ready.sort()
                stuck = {name for name, sources in need.items() if sources}
                if stuck:
                    path, node = [], min(stuck)
                    while node not in path:
                        path.append(node)
                        node = min(s for s in self.graph.inputs[node] if s in stuck)
                    loop = path[path.index(node):][::-1]
                    pivot = loop.index(min(loop))
                    raise GraphError('cycle', '->'.join(loop[pivot:] + loop[:pivot] + loop[pivot:pivot + 1]))
                return order

            def recompute(self):
                order = self._plan()
                fresh = dict(self.values)
                for name in order:
                    fresh[name] = self.fns[name](*[fresh[s] for s in self.graph.inputs[name]])
                self.values = fresh
                self.dirty.clear()
                return order
        ''',
        '''
        from node_registry import Registry


        def test_public_interface():
            engine = Engine(Registry())
            engine.define('b', ['a'], lambda a: a + 1)
            engine.define('c', ['b'], lambda b: b * 2)
            engine.set('a', 3)
            assert engine.recompute() == ['b', 'c']
            assert engine.values == {'a': 3, 'b': 4, 'c': 8}
            assert engine.dirty == set()
        ''',
        {
            "plan": '''
                from node_registry import Registry


                def build():
                    engine = Engine(Registry())
                    engine.set('x', 2)
                    engine.set('y', 3)
                    engine.define('sum', ['x', 'y'], lambda x, y: x + y)
                    engine.define('ratio', ['sum', 'x'], lambda s, x: s // (x - 2))
                    engine.define('aaa', ['ratio', 'sum'], lambda r, s: (r, s))
                    return engine


                def test_order_is_topological_ties_go_by_name_and_only_dirty_nodes_run():
                    engine = build()
                    engine.set('x', 4)
                    assert engine.recompute() == ['sum', 'ratio', 'aaa']
                    assert engine.values == {'x': 4, 'y': 3, 'sum': 7, 'ratio': 3, 'aaa': (3, 7)}
                    engine.set('y', 9)
                    engine.define('diff', ['sum', 'x'], lambda s, x: s - x)
                    assert sorted(engine.dirty) == ['aaa', 'diff', 'ratio', 'sum']
                    assert engine.recompute() == ['sum', 'diff', 'ratio', 'aaa']
                    assert engine.values['aaa'] == (6, 13) and engine.values['diff'] == 9
                    assert engine.recompute() == [] and engine.dirty == set()


                def test_failure_changes_nothing_and_the_retry_recomputes_everything():
                    engine = build()
                    with pytest.raises(ZeroDivisionError) as caught:
                        engine.recompute()
                    assert caught.value.__context__ is None and engine.values == {'x': 2, 'y': 3}
                    assert sorted(engine.dirty) == ['aaa', 'ratio', 'sum']
                    engine.set('x', 4)
                    assert engine.recompute() == ['sum', 'ratio', 'aaa']
                    assert engine.values == {'x': 4, 'y': 3, 'sum': 7, 'ratio': 3, 'aaa': (3, 7)}
                    assert engine.dirty == set()
            ''',
            "cycle": '''
                from node_registry import GraphError, Registry


                def test_cycle_is_named_and_nothing_is_evaluated():
                    engine = Engine(Registry())
                    engine.define('m', ['q'], lambda q: q)
                    engine.define('p', ['m'], lambda m: m)
                    engine.define('q', ['p'], lambda p: p)
                    with pytest.raises(GraphError) as caught:
                        engine.recompute()
                    assert type(caught.value) is GraphError and caught.value.args == ('cycle', 'm->p->q->m')
                    assert engine.values == {} and engine.dirty == {'m', 'p', 'q'}
                    engine = Engine(Registry())
                    engine.define('b', ['y'], lambda y: y)
                    engine.define('y', ['x'], lambda x: x)
                    engine.define('x', ['y'], lambda y: y)
                    with pytest.raises(GraphError) as caught:
                        engine.recompute()
                    assert caught.value.args == ('cycle', 'x->y->x')
                    assert engine.values == {} and engine.dirty == {'b', 'x', 'y'}


                def test_registry_rules_apply():
                    engine = Engine(Registry())
                    engine.set('x', 2)
                    engine.define('sum', ['x'], lambda x: x)
                    calls = (lambda: engine.define('sum', ['x'], lambda x: x),
                             lambda: engine.define('loop', ['loop'], lambda x: x), lambda: engine.set('sum', 1))
                    for call, args in zip(calls, (('duplicate', 'sum'), ('self', 'loop'), ('not an input', 'sum'))):
                        with pytest.raises(GraphError) as caught:
                            call()
                        assert caught.value.args == args
                    assert sorted(engine.dirty) == ['sum'] and engine.values == {'x': 2}
            ''',
        },
        extra_files={"node_registry": _NODE_REGISTRY},
    ),
    _task(
        "restat_graph", "graph_replan",
        "只读依赖 node_registry：Registry.inputs[name] 是 name 的输入列表（可为后定义的节点或外部输入），"
        "users[source] 是依赖 source 的节点；重名抛 GraphError('duplicate', name)，"
        "依赖自己抛 GraphError('self', name)；不检查环。Engine 增量求值，values 是当前值，dirty 是待重算节点的集合。"
        "要求：\n"
        "1）define(name, inputs, fn)：fn 依次接收各输入的值；name 及其直接、间接依赖者待重算，name 记为“已变化”。\n"
        "2）set(name, value)：value 与当前值相同则不修改，返回 False；否则写入新值，name 的直接、间接依赖者待重算，"
        "name 记为“已变化”，返回 True。\n"
        "3）recompute()：先反复把输入都已排好的待重算节点按名字排序接到末尾，得到求值顺序；只求值本身或某个输入"
        "“已变化”的节点（新值不等于旧值的节点也成为“已变化”），其余跳过。成功后更新 values，清空 dirty 和"
        "“已变化”记录，返回求值节点的列表。\n"
        "4）排序无法继续（剩余节点互相依赖成环）时抛 GraphError('cycle', 剩余节点中名字最小的节点)，不做任何求值，"
        "values 与 dirty 不变。\n"
        "5）fn 抛异常时 values、dirty 和“已变化”记录都与调用前相同，原异常原样抛出。",
        '''
        from node_registry import GraphError


        class Engine:
            def __init__(self, registry):
                self.graph, self.fns, self.values, self.dirty, self.stale = registry, {}, {}, set(), set()

            def define(self, name, inputs, fn):
                self.graph.define(name, inputs)
                self.fns[name] = fn
                self.stale.add(name)
                self._mark(name)

            def set(self, name, value):
                self.values[name] = value
                self.stale.add(name)
                self._mark(*self.graph.users.get(name, []))
                return True

            def _mark(self, *names):
                pending = [*names]
                while pending:
                    node = pending.pop()
                    if node not in self.dirty:
                        self.dirty.add(node)
                        pending.extend(self.graph.users.get(node, []))

            def _plan(self):
                order, left = [], set(self.dirty)
                while left:
                    free = sorted(name for name in left if not left.intersection(self.graph.inputs[name]))
                    if not free:
                        raise GraphError('cycle', max(left))
                    order += free
                    left.difference_update(free)
                return order

            def recompute(self):
                order = self._plan()
                for name in order:
                    self.values[name] = self.fns[name](*[self.values[s] for s in self.graph.inputs[name]])
                    self.dirty.discard(name)
                self.stale = set()
                return order
        ''',
        '''
        from node_registry import GraphError


        class Engine:
            def __init__(self, registry):
                self.graph, self.fns, self.values, self.dirty, self.stale = registry, {}, {}, set(), set()

            def define(self, name, inputs, fn):
                self.graph.define(name, inputs)
                self.fns[name] = fn
                self.stale.add(name)
                self._mark(name)

            def set(self, name, value):
                if self.values.get(name) == value:
                    return False
                self.values[name] = value
                self.stale.add(name)
                self._mark(*self.graph.users.get(name, []))
                return True

            def _mark(self, *names):
                pending = [*names]
                while pending:
                    node = pending.pop()
                    if node not in self.dirty:
                        self.dirty.add(node)
                        pending.extend(self.graph.users.get(node, []))

            def _plan(self):
                order, left = [], set(self.dirty)
                while left:
                    free = sorted(name for name in left if not left.intersection(self.graph.inputs[name]))
                    if not free:
                        raise GraphError('cycle', min(left))
                    order += free
                    left.difference_update(free)
                return order

            def recompute(self):
                order = self._plan()
                fresh, changed, done = dict(self.values), set(self.stale), []
                for name in order:
                    inputs = self.graph.inputs[name]
                    if name not in changed and not changed.intersection(inputs):
                        continue
                    fresh[name] = self.fns[name](*[fresh[s] for s in inputs])
                    done.append(name)
                    if name not in self.values or fresh[name] != self.values[name]:
                        changed.add(name)
                self.values, self.dirty, self.stale = fresh, set(), set()
                return done
        ''',
        '''
        from node_registry import Registry


        def test_public_interface():
            engine = Engine(Registry())
            engine.set('a', 3)
            engine.define('b', ['a'], lambda a: a + 1)
            engine.define('c', ['b', 'a'], lambda b, a: b * a)
            assert engine.recompute() == ['b', 'c'] and engine.values == {'a': 3, 'b': 4, 'c': 12}
            assert engine.set('a', 5) is True and engine.recompute() == ['b', 'c']
            assert engine.values == {'a': 5, 'b': 6, 'c': 30}
        ''',
        {
            "graph": '''
                from node_registry import GraphError, Registry


                def build():
                    engine = Engine(Registry())
                    engine.set('x', 2)
                    engine.set('y', 3)
                    engine.define('sum', ['x', 'y'], lambda x, y: x + y)
                    engine.define('sq', ['sum'], lambda s: s * s)
                    engine.define('half', ['sq'], lambda q: q // 2)
                    engine.define('tag', ['sum', 'half'], lambda s, h: (s, h))
                    return engine


                def test_unchanged_values_stop_the_recomputation():
                    engine = build()
                    assert engine.recompute() == ['sum', 'sq', 'half', 'tag']
                    assert engine.values == {'x': 2, 'y': 3, 'sum': 5, 'sq': 25, 'half': 12, 'tag': (5, 12)}
                    assert engine.set('x', 1) is True and engine.set('y', 4) is True
                    assert sorted(engine.dirty) == ['half', 'sq', 'sum', 'tag']
                    assert engine.recompute() == ['sum']
                    assert engine.values['tag'] == (5, 12) and engine.dirty == set()
                    assert engine.set('x', 3) is True
                    assert engine.recompute() == ['sum', 'sq', 'half', 'tag']
                    assert engine.values == {'x': 3, 'y': 4, 'sum': 7, 'sq': 49, 'half': 24, 'tag': (7, 24)}
                    assert engine.set('x', 3) is False and engine.dirty == set() and engine.recompute() == []


                def test_defined_nodes_run_even_when_nothing_changed():
                    engine = build()
                    engine.recompute()
                    engine.define('same', ['sq'], lambda q: q)
                    engine.define('user', ['later'], lambda v: v)
                    engine.define('later', ['sum'], lambda s: s)
                    assert engine.recompute() == ['later', 'same', 'user']
                    assert engine.values['user'] == 5 and engine.values['same'] == 25 and engine.dirty == set()


                def test_failure_changes_nothing_and_the_replay_runs_everything():
                    engine = build()
                    engine.recompute()
                    seen = []

                    def once(s, x):
                        seen.append(s)
                        if len(seen) == 1:
                            raise ValueError('first call fails')
                        return s * 10 + x

                    engine.define('ratio', ['sum', 'x'], once)
                    engine.set('y', 0)
                    with pytest.raises(ValueError) as caught:
                        engine.recompute()
                    assert str(caught.value) == 'first call fails' and caught.value.__context__ is None
                    assert engine.values == {'x': 2, 'y': 0, 'sum': 5, 'sq': 25, 'half': 12, 'tag': (5, 12)}
                    assert sorted(engine.dirty) == ['half', 'ratio', 'sq', 'sum', 'tag'] and seen == [2]
                    assert engine.recompute() == ['sum', 'ratio', 'sq', 'half', 'tag'] and seen == [2, 2]
                    assert engine.values == {'x': 2, 'y': 0, 'sum': 2, 'ratio': 22, 'sq': 4, 'half': 2, 'tag': (2, 2)}
                    assert engine.dirty == set()


                def test_cycle_is_reported_before_anything_is_evaluated():
                    engine = Engine(Registry())
                    engine.define('m', ['q'], lambda q: q)
                    engine.define('p', ['m'], lambda m: m)
                    engine.define('q', ['p'], lambda p: p)
                    with pytest.raises(GraphError) as caught:
                        engine.recompute()
                    assert type(caught.value) is GraphError and caught.value.args == ('cycle', 'm')
                    assert engine.values == {} and engine.dirty == {'m', 'p', 'q'}
            ''',
        },
        extra_files={"node_registry": _NODE_REGISTRY},
    ),
]

MUTANTS = {
    "h3_work_commit_001": [
        ("            for key, old in reversed(log):", "            for key, old in []:"),
        ("plan = [(key, None) for key in self._deletes] + list(self._puts.items())",
         "plan = list(self._puts.items()) + [(key, None) for key in self._deletes]"),
        ("            raise\n", "            raise RuntimeError('commit failed') from None\n"),
        ("        self._state = 'already committed'\n", ""),
        ("        self._state = 'rolled back'\n", ""),
    ],
    "h3_work_abort_001": [
        ("                    self._store.put(key, old)\n", "                    pass\n"),
        ("            for key, old in reversed(log):", "            for key, old in log:"),
        ("            self._close()\n            for key, old in reversed(log):",
         "            self._puts.clear()\n            self._deletes.clear()\n"
         "            for key, old in reversed(log):"),
        ("            self._close()\n            for key, old in reversed(log):",
         "            self._state = 'rolled back'\n            for key, old in reversed(log):"),
        ("        busy = key in self._puts or key in self._deletes", "        busy = key in self._puts"),
    ],
    "h3_undo_batch_001": [
        ("            for op in reversed(done):", "            for op in done:"),
        ("                results.append(self.backend.apply(op))\n                done.append(op)",
         "                done.append(op)\n                results.append(self.backend.apply(op))"),
        ("                    failure = failure or exc\n",
         "                    failure = failure or exc\n                    break\n"),
        ("                    failure = failure or exc\n", "                    failure = exc\n"),
        ("                    self.remaining.insert(0, op)\n", "                    self.remaining.append(op)\n"),
    ],
    "h3_recover_batch_001": [
        ("        failure = self._unwind(list(self.backend.journal))\n"
         "        if failure is not None:\n            raise failure\n", ""),
        ("            raise failure\n", "            pass\n"),
        ("                failure = failure or exc\n", "                failure = exc\n"),
        ("                failure = failure or exc\n",
         "                failure = failure or exc\n                break\n"),
        ("        self.backend.journal.clear()\n", ""),
    ],
    "h3_ledger_post_001": [
        ("            for name, amount in reversed(applied):\n                self._move(name, -amount)\n", ""),
        ("            for name, amount in reversed(applied):", "            for name, amount in applied:"),
        ("            return list(self.posted).index(key) + 1", "            pass"),
        ("        if key in self.reversals:\n            raise LedgerError('already reversed', key)\n", ""),
        ("        seq = self.post(as_key, [(name, -amount) for name, amount in self.posted[key]])\n"
         "        self.reversals.add(key)\n",
         "        self.reversals.add(key)\n"
         "        seq = self.post(as_key, [(name, -amount) for name, amount in self.posted[key]])\n"),
    ],
    "h3_teller_transfer_001": [
        ("            if self.receipts[key][:3] != (src, dst, amount):\n"
         "                raise LedgerError('key reused', key)\n", ""),
        ("            self.store.credit(src, amount)\n            raise\n", "            raise\n"),
        ("        receipt = self.transfer(as_key, dst, src, amount)\n        self.reversals.add(key)\n",
         "        self.reversals.add(key)\n        receipt = self.transfer(as_key, dst, src, amount)\n"),
        ("        if key in self.reversals:\n            raise LedgerError('already reversed', key)\n", ""),
        ("        if key in self.receipts:\n",
         "        if src == dst:\n            raise LedgerError('same account', src)\n"
         "        if key in self.receipts:\n"),
    ],
    "h3_replan_graph_001": [
        ("        fresh = dict(self.values)\n", "        fresh = self.values\n"),
        ("        order = self._plan()\n        fresh = dict(self.values)\n",
         "        order = self._plan()\n        self.dirty.clear()\n        fresh = dict(self.values)\n"),
        ("            ready.sort()\n", ""),
        ("                self.dirty.add(node)\n"
         "                pending.extend(self.graph.users.get(node, []))\n",
         "                self.dirty.add(node)\n"),
        ("            pivot = loop.index(min(loop))\n", "            pivot = 0\n"),
    ],
    "h3_restat_graph_001": [
        ("            if name not in changed and not changed.intersection(inputs):",
         "            if not changed.intersection(inputs):"),
        ("            if name not in self.values or fresh[name] != self.values[name]:\n", "            if True:\n"),
        ("        fresh, changed, done = dict(self.values), set(self.stale), []",
         "        fresh, changed, done = self.values, set(self.stale), []"),
        ("        self.values, self.dirty, self.stale = fresh, set(), set()",
         "        self.values, self.dirty = fresh, set()"),
        ("        if self.values.get(name) == value:\n            return False\n", ""),
    ],
}
