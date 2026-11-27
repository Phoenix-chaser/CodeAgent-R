"""State-consistency tasks for the hard batch (h3).

Every program here keeps an auxiliary structure next to its primary data: a
secondary index, a cached derived value, the reverse half of a bidirectional
map, the status and recency structures of a registry.  Some code paths update
both and others only the primary one, so repairing the obvious path leaves a
stale entry behind on another path (rename, pop, bulk update, clear, a failed
or expired entry).
"""
from __future__ import annotations

from agentlab.benchmark.hard_tasks_v3_common import _task

SPECS = [
    _task(
        "key_index_table", "index_sync",
        "Table 用递增 id 存储行，并维护索引 key → id 列表（id 按加入该 key 的先后顺序排列），"
        "任何操作之后索引都必须与行一致。要求：\n"
        "1）add(key, value) 新增一行并返回 id（从 1 起递增）；get(id) 返回行 dict 的副本，"
        "id 不存在抛 KeyError。\n"
        "2）ids(key) 返回该 key 下 id 列表的副本，key 不存在返回 []；keys() 按 key 首次出现的顺序"
        "返回仍有行的 key，某 key 下最后一行被删除后它即从 keys() 消失，再次出现时排在末尾。\n"
        "3）rename(id, new_key)：id 不存在抛 KeyError 且状态不变；new_key 等于原 key 时不做任何修改；"
        "否则该 id 移到 new_key 列表末尾。\n"
        "4）remove(id) 删除并返回该行；pop(key) 删除并返回该 key 下最早加入的行；"
        "id 或 key 不存在抛 KeyError。\n"
        "5）move(old_key, new_key) 把 old_key 下全部行按原顺序移到 new_key 末尾，返回被移动 id 的列表；"
        "old_key 不存在抛 KeyError 且状态不变。\n"
        "6）clear() 清空所有行与索引，id 计数器重置为 1。",
        '''
        class Table:
            def __init__(self):
                self._rows = {}
                self._index = {}
                self._next_id = 1

            def add(self, key, value):
                row_id = self._next_id
                self._next_id += 1
                self._rows[row_id] = {'id': row_id, 'key': key, 'value': value}
                self._index.setdefault(key, []).append(row_id)
                return row_id

            def get(self, row_id):
                return dict(self._rows[row_id])

            def ids(self, key):
                return list(self._index.get(key, ()))

            def keys(self):
                return list(self._index)

            def rename(self, row_id, new_key):
                row = self._rows[row_id]
                row['key'] = new_key
                self._index.setdefault(new_key, []).append(row_id)

            def remove(self, row_id):
                return self._rows.pop(row_id)

            def pop(self, key):
                return self.remove(self._index[key][0])

            def move(self, old_key, new_key):
                moved = self.ids(old_key)
                if not moved:
                    raise KeyError(old_key)
                for row_id in moved:
                    self.rename(row_id, new_key)
                return moved

            def clear(self):
                self._rows.clear()
        ''',
        '''
        class Table:
            def __init__(self):
                self._rows = {}
                self._index = {}
                self._next_id = 1

            def add(self, key, value):
                row_id = self._next_id
                self._next_id += 1
                self._rows[row_id] = {'id': row_id, 'key': key, 'value': value}
                self._index.setdefault(key, []).append(row_id)
                return row_id

            def get(self, row_id):
                return dict(self._rows[row_id])

            def ids(self, key):
                return list(self._index.get(key, ()))

            def keys(self):
                return list(self._index)

            def _unlink(self, row_id):
                key = self._rows[row_id]['key']
                bucket = self._index[key]
                bucket.remove(row_id)
                if not bucket:
                    del self._index[key]

            def rename(self, row_id, new_key):
                row = self._rows[row_id]
                if row['key'] != new_key:
                    self._unlink(row_id)
                    row['key'] = new_key
                    self._index.setdefault(new_key, []).append(row_id)

            def remove(self, row_id):
                self._unlink(row_id)
                return self._rows.pop(row_id)

            def pop(self, key):
                return self.remove(self._index[key][0])

            def move(self, old_key, new_key):
                moved = self.ids(old_key)
                if not moved:
                    raise KeyError(old_key)
                for row_id in moved:
                    self.rename(row_id, new_key)
                return moved

            def clear(self):
                self._rows.clear()
                self._index.clear()
                self._next_id = 1
        ''',
        '''
        def test_public_add_and_query():
            table = Table()
            first = table.add('red', 10)
            second = table.add('blue', 20)
            third = table.add('red', 30)
            assert [first, second, third] == [1, 2, 3]
            assert table.get(first) == {'id': 1, 'key': 'red', 'value': 10}
            assert table.ids('red') == [first, third]
            assert table.keys() == ['red', 'blue']
            assert table.ids('green') == []
        ''',
        {
            "rename": '''
                def test_rename_moves_row_between_keys():
                    table = Table()
                    first = table.add('a', 1)
                    second = table.add('b', 2)
                    third = table.add('a', 3)
                    table.rename(first, 'b')
                    assert table.ids('a') == [third] and table.ids('b') == [second, first]
                    table.rename(third, 'b')
                    table.rename(second, 'b')
                    assert table.ids('a') == [] and table.keys() == ['b']
                    assert table.ids('b') == [second, first, third]
                    row = table.get(first)
                    row['key'] = 'zz'
                    assert table.get(first)['key'] == 'b'
                    with pytest.raises(KeyError):
                        table.rename(99, 'c')
                    assert table.keys() == ['b'] and table.ids('c') == []
                    table.add('a', 4)
                    assert table.keys() == ['b', 'a'] and table.ids('a') == [4]
            ''',
            "remove": '''
                def test_remove_and_pop_keep_index_in_sync():
                    table = Table()
                    ids = [table.add('a', i) for i in range(3)]
                    last = table.add('b', 9)
                    assert table.pop('a')['value'] == 0
                    assert table.ids('a') == ids[1:] and table.keys() == ['a', 'b']
                    assert table.remove(last) == {'id': last, 'key': 'b', 'value': 9}
                    assert table.keys() == ['a'] and table.ids('b') == []
                    with pytest.raises(KeyError):
                        table.remove(last)
                    with pytest.raises(KeyError):
                        table.pop('b')
                    for row_id in table.ids('a'):
                        assert table.remove(row_id)['id'] == row_id
                    assert table.keys() == [] and table.ids('a') == []
                    with pytest.raises(KeyError):
                        table.pop('a')
            ''',
            "bulk": '''
                def test_move_keeps_order_and_clear_resets_everything():
                    table = Table()
                    moved = [table.add('x', i) for i in range(3)]
                    kept = table.add('y', 0)
                    assert table.move('x', 'y') == moved
                    assert table.ids('x') == [] and table.keys() == ['y']
                    assert table.ids('y') == [kept] + moved
                    assert table.move('y', 'y') == [kept] + moved
                    with pytest.raises(KeyError):
                        table.move('x', 'z')
                    assert table.keys() == ['y'] and table.ids('z') == []
                    table.clear()
                    assert table.keys() == []
                    with pytest.raises(KeyError):
                        table.get(kept)
                    with pytest.raises(KeyError):
                        table.pop('y')
                    assert table.add('z', 1) == 1 and table.keys() == ['z']
                    assert table.ids('z') == [1]
            ''',
        },
    ),
    _task(
        "norm_key_index", "index_sync",
        "Table 按 id 存储行，并维护索引 key → id 列表（id 按加入该 key 的先后顺序排列）。所有 key 都必须先经"
        "只读模块 key_rules 的 normalize 处理（去掉全部空白并合并大小写；不是字符串或结果为空时抛 InvalidKey），"
        "行里保存的也是规范化后的 key。要求：\n"
        "1）add(key, value) 新增一行并返回 id（从 1 起递增）；get(id) 返回行 dict 的副本，id 不存在抛 KeyError。\n"
        "2）extend(pairs) 批量新增，按给定顺序返回 id 列表；任一 key 无效抛 InvalidKey，且行、索引、计数器都不变。\n"
        "3）ids(key) 返回该 key 下 id 列表的副本，key 不存在返回 []；keys() 按首次出现顺序返回仍有行的 key。\n"
        "4）rename(id, new_key)：key 无效抛 InvalidKey，id 不存在抛 KeyError，都不改变任何状态；"
        "规范化后与原 key 相同时不做任何修改；否则把该 id 移到新 key 列表末尾。\n"
        "5）pop(key) 删除并返回该 key 下最早加入的行，key 不存在抛 KeyError。\n"
        "6）move(old_key, new_key) 把 old_key 下全部行按原顺序移到 new_key 末尾并返回其 id 列表；"
        "old_key 不存在抛 KeyError，任一 key 无效抛 InvalidKey，都不改变任何状态。\n"
        "7）clear() 清空全部行与索引，id 计数器重置为 1。",
        '''
        from key_rules import normalize


        class Table:
            def __init__(self):
                self._rows = {}
                self._index = {}
                self._next_id = 1

            def add(self, key, value):
                key = normalize(key)
                row_id = self._next_id
                self._next_id += 1
                self._rows[row_id] = {'id': row_id, 'key': key, 'value': value}
                self._index.setdefault(key, []).append(row_id)
                return row_id

            def extend(self, pairs):
                return [self.add(key, value) for key, value in pairs]

            def get(self, row_id):
                return dict(self._rows[row_id])

            def ids(self, key):
                return list(self._index.get(normalize(key), ()))

            def keys(self):
                return list(self._index)

            def _unlink(self, row_id):
                key = self._rows[row_id]['key']
                bucket = self._index[key]
                bucket.remove(row_id)
                if not bucket:
                    del self._index[key]

            def rename(self, row_id, new_key):
                row = self._rows[row_id]
                if row['key'] != new_key:
                    self._unlink(row_id)
                    row['key'] = normalize(new_key)
                    self._index.setdefault(row['key'], []).append(row_id)

            def pop(self, key):
                row_id = self._index[key][0]
                self._unlink(row_id)
                return self._rows.pop(row_id)

            def move(self, old_key, new_key):
                moved = self.ids(old_key)
                if not moved:
                    raise KeyError(old_key)
                for row_id in moved:
                    self.rename(row_id, new_key)
                return moved

            def clear(self):
                self._rows.clear()
                self._index.clear()
                self._next_id = 1
        ''',
        '''
        from key_rules import normalize


        class Table:
            def __init__(self):
                self._rows = {}
                self._index = {}
                self._next_id = 1

            def extend(self, pairs):
                items = [(normalize(key), value) for key, value in pairs]
                ids = list(range(self._next_id, self._next_id + len(items)))
                for row_id, (key, value) in zip(ids, items):
                    self._rows[row_id] = {'id': row_id, 'key': key, 'value': value}
                    self._index.setdefault(key, []).append(row_id)
                self._next_id += len(items)
                return ids

            def add(self, key, value):
                return self.extend([(key, value)])[0]

            def get(self, row_id):
                return dict(self._rows[row_id])

            def ids(self, key):
                return list(self._index.get(normalize(key), ()))

            def keys(self):
                return list(self._index)

            def _unlink(self, row_id):
                key = self._rows[row_id]['key']
                bucket = self._index[key]
                bucket.remove(row_id)
                if not bucket:
                    del self._index[key]

            def rename(self, row_id, new_key):
                new_key = normalize(new_key)
                row = self._rows[row_id]
                if row['key'] != new_key:
                    self._unlink(row_id)
                    row['key'] = new_key
                    self._index.setdefault(new_key, []).append(row_id)

            def pop(self, key):
                row_id = self._index[normalize(key)][0]
                self._unlink(row_id)
                return self._rows.pop(row_id)

            def move(self, old_key, new_key):
                moved = self.ids(old_key)
                new_key = normalize(new_key)
                if not moved:
                    raise KeyError(old_key)
                for row_id in moved:
                    self.rename(row_id, new_key)
                return moved

            def clear(self):
                self._rows.clear()
                self._index.clear()
                self._next_id = 1
        ''',
        '''
        import pytest
        from key_rules import InvalidKey


        def test_public_add_and_lookup():
            table = Table()
            assert [table.add('Red', 1), table.add(' red ', 2)] == [1, 2]
            with pytest.raises(InvalidKey):
                table.add('   ', 3)
            assert table.add('Blue', 3) == 3
            assert table.get(1) == {'id': 1, 'key': 'red', 'value': 1}
            assert table.ids('RED') == [1, 2]
            assert table.keys() == ['red', 'blue']
        ''',
        {
            "bulk": '''
                from key_rules import InvalidKey


                def test_failed_extend_leaves_table_untouched():
                    table = Table()
                    first = table.add('Red', 1)
                    assert table.extend([('red', 2), (' GREEN  TEA ', 3)]) == [2, 3]
                    assert table.keys() == ['red', 'greentea']
                    with pytest.raises(InvalidKey):
                        table.extend([('blue', 4), ('  ', 5), ('blue', 6)])
                    with pytest.raises(InvalidKey):
                        table.add(7, 8)
                    assert table.keys() == ['red', 'greentea'] and table.ids('blue') == []
                    assert table.extend([]) == []
                    assert table.add(' bLue ', 4) == 4
                    assert table.ids('BLUE') == [4]
                    assert table.keys() == ['red', 'greentea', 'blue']
                    assert table.get(3) == {'id': 3, 'key': 'greentea', 'value': 3}
                    table.clear()
                    assert table.keys() == [] and table.ids('red') == []
                    with pytest.raises(KeyError):
                        table.get(first)
                    assert table.add('red', 9) == 1 and table.ids(' RED ') == [1]
            ''',
            "rename": '''
                from key_rules import InvalidKey


                def test_rename_normalizes_before_comparing_and_fails_cleanly():
                    table = Table()
                    first = table.add('a', 1)
                    second = table.add('a', 2)
                    third = table.add(' A  b ', 3)
                    table.rename(first, ' A ')
                    assert table.ids('a') == [first, second]
                    table.rename(first, 'A  B')
                    assert table.ids('a') == [second] and table.ids('ab') == [third, first]
                    assert table.get(first)['key'] == 'ab'
                    with pytest.raises(InvalidKey):
                        table.rename(third, '')
                    with pytest.raises(InvalidKey):
                        table.move('a', ' ')
                    with pytest.raises(KeyError):
                        table.rename(99, 'c')
                    with pytest.raises(KeyError):
                        table.move('zz', 'c')
                    assert table.keys() == ['a', 'ab'] and table.ids('c') == []
                    assert table.ids('ab') == [third, first] and table.ids('a') == [second]
                    assert table.move('a', ' A ') == [second]
                    assert table.ids('a') == [second] and table.keys() == ['a', 'ab']
            ''',
            "pop": '''
                from key_rules import InvalidKey


                def test_pop_normalizes_lookup_key():
                    table = Table()
                    ids = table.extend([('Task', 0), ('task', 1), (' TASK ', 2)])
                    assert ids == [1, 2, 3]
                    assert table.pop(' tAsK ') == {'id': 1, 'key': 'task', 'value': 0}
                    assert table.ids('TASK') == [2, 3]
                    with pytest.raises(InvalidKey):
                        table.pop('   ')
                    with pytest.raises(InvalidKey):
                        table.ids(None)
                    assert table.ids('task') == [2, 3]
                    assert table.pop('task')['id'] == 2 and table.pop('task')['id'] == 3
                    assert table.keys() == [] and table.ids('task') == []
                    with pytest.raises(KeyError):
                        table.pop('task')
            ''',
        },
        extra_files={
            "key_rules": '''
                """Key rules shared by every table of the service."""


                class InvalidKey(ValueError):
                    """A key that is not text or is empty once folded."""


                def normalize(key):
                    if not isinstance(key, str):
                        raise InvalidKey('key must be text')
                    folded = ''.join(key.split()).casefold()
                    if not folded:
                        raise InvalidKey('key is empty')
                    return folded
            ''',
        },
    ),
    _task(
        "stock_totals", "derived_cache",
        "Stock 保存商品（name → price、qty），并缓存两个派生值：total() 为所有商品 price × qty 之和，"
        "count() 为所有 qty 之和。要求：\n"
        "1）put(name, price, qty)：price 或 qty 小于等于 0 抛 ValueError 且不改变任何状态；商品不存在则新增，"
        "已存在则数量累加并把价格改为新的 price。\n"
        "2）set_price(name, price)：name 不存在抛 KeyError，price 小于等于 0 抛 ValueError，都不改变任何状态。\n"
        "3）get(name) 返回条目 dict（price、qty）的副本，name 不存在抛 KeyError；"
        "remove(name) 删除并返回该条目，name 不存在抛 KeyError。\n"
        "4）apply(quantities) 批量设置数量：qty 为 0 的商品被删除，其余商品的数量改为给定值。\n"
        "5）clear() 删除全部商品。\n"
        "6）任何修改之后（包括上面抛出异常的操作），total() 与 count() 都必须等于按当前商品重新计算的结果；"
        "空库存时二者都是 0。",
        '''
        class Stock:
            def __init__(self):
                self._items = {}
                self._total = None
                self._count = None

            def put(self, name, price, qty):
                if price <= 0 or qty <= 0:
                    raise ValueError('price and qty must be positive')
                item = self._items.get(name)
                if item is None:
                    self._items[name] = {'price': price, 'qty': qty}
                    self._total = None
                    self._count = None
                else:
                    item['qty'] += qty

            def set_price(self, name, price):
                if price <= 0:
                    raise ValueError('price must be positive')
                self._items[name]['price'] = price
                self._total = None
                self._count = None

            def remove(self, name):
                return self._items.pop(name)

            def apply(self, quantities):
                for name, qty in quantities.items():
                    if qty:
                        self._items[name]['qty'] = qty
                    else:
                        del self._items[name]

            def clear(self):
                self._items.clear()

            def get(self, name):
                return dict(self._items[name])

            def total(self):
                if self._total is None:
                    self._total = sum(item['price'] * item['qty'] for item in self._items.values())
                return self._total

            def count(self):
                if self._count is None:
                    self._count = sum(item['qty'] for item in self._items.values())
                return self._count
        ''',
        '''
        class Stock:
            def __init__(self):
                self._items = {}
                self._total = None
                self._count = None

            def _invalidate(self):
                self._total = None
                self._count = None

            def put(self, name, price, qty):
                if price <= 0 or qty <= 0:
                    raise ValueError('price and qty must be positive')
                item = self._items.get(name)
                if item is None:
                    self._items[name] = {'price': price, 'qty': qty}
                else:
                    item['qty'] += qty
                    item['price'] = price
                self._invalidate()

            def set_price(self, name, price):
                if price <= 0:
                    raise ValueError('price must be positive')
                self._items[name]['price'] = price
                self._invalidate()

            def remove(self, name):
                item = self._items.pop(name)
                self._invalidate()
                return item

            def apply(self, quantities):
                self._invalidate()
                for name, qty in quantities.items():
                    if qty:
                        self._items[name]['qty'] = qty
                    else:
                        del self._items[name]

            def clear(self):
                self._items.clear()
                self._invalidate()

            def get(self, name):
                return dict(self._items[name])

            def total(self):
                if self._total is None:
                    self._total = sum(item['price'] * item['qty'] for item in self._items.values())
                return self._total

            def count(self):
                if self._count is None:
                    self._count = sum(item['qty'] for item in self._items.values())
                return self._count
        ''',
        '''
        def test_public_put_and_read():
            stock = Stock()
            stock.put('pen', 3, 2)
            stock.put('ink', 10, 1)
            stock.set_price('pen', 4)
            assert stock.get('pen') == {'price': 4, 'qty': 2}
            assert stock.total() == 18 and stock.count() == 3
            stock.apply({'ink': 5})
            stock.remove('pen')
            stock.clear()
        ''',
        {
            "inplace": '''
                def test_in_place_updates_refresh_caches():
                    stock = Stock()
                    stock.put('pen', 3, 2)
                    stock.put('ink', 10, 1)
                    assert (stock.total(), stock.count()) == (16, 3)
                    stock.put('pen', 4, 1)
                    assert (stock.total(), stock.count()) == (22, 4)
                    stock.set_price('ink', 5)
                    assert (stock.total(), stock.count()) == (17, 4)
                    stock.set_price('ink', 7)
                    assert (stock.total(), stock.count()) == (19, 4)
                    assert stock.get('ink') == {'price': 7, 'qty': 1}
            ''',
            "bulk": '''
                def test_bulk_update_and_remove_refresh_caches():
                    stock = Stock()
                    for name, price, qty in (('a', 2, 1), ('b', 3, 2), ('c', 5, 4)):
                        stock.put(name, price, qty)
                    assert (stock.total(), stock.count()) == (28, 7)
                    stock.apply({'a': 10, 'b': 0})
                    assert (stock.total(), stock.count()) == (40, 14)
                    with pytest.raises(KeyError):
                        stock.get('b')
                    assert stock.remove('c') == {'price': 5, 'qty': 4}
                    assert (stock.total(), stock.count()) == (20, 10)
                    row = stock.get('a')
                    row['qty'] = 99
                    stock.apply({})
                    assert (stock.total(), stock.count()) == (20, 10)
            ''',
            "failures": '''
                def test_failed_operations_and_clear_keep_caches_honest():
                    stock = Stock()
                    stock.put('a', 2, 1)
                    assert (stock.total(), stock.count()) == (2, 1)
                    with pytest.raises(ValueError):
                        stock.put('a', 0, 1)
                    with pytest.raises(ValueError):
                        stock.set_price('a', -1)
                    with pytest.raises(KeyError):
                        stock.set_price('zz', 1)
                    with pytest.raises(KeyError):
                        stock.remove('zz')
                    assert stock.get('a') == {'price': 2, 'qty': 1}
                    assert (stock.total(), stock.count()) == (2, 1)
                    stock.clear()
                    assert (stock.total(), stock.count()) == (0, 0)
                    stock.put('b', 7, 3)
                    assert (stock.total(), stock.count()) == (21, 3)
                    stock.clear()
                    stock.clear()
                    assert (stock.total(), stock.count()) == (0, 0)
            ''',
        },
    ),
    _task(
        "priced_bill", "derived_cache",
        "Bill 保存账单行（name → price、qty），并缓存 total()（各行合计之和）和 top()（合计最大的行名，"
        "相同时取 name 最小者，空账单为 None）。行合计必须用只读模块 pricing 计算：unit_price(price) "
        "校验单价（正整数，否则抛 BadPrice），line_total(price, qty) 在 qty ≥ 10 时打九折。要求：\n"
        "1）add(name, price, qty)：单价无效抛 BadPrice，qty 小于等于 0 抛 ValueError；新增或替换该行。\n"
        "2）merge(lines) 批量新增或替换行（name → (price, qty)）：任一行无效抛相应异常，且不做任何修改。\n"
        "3）transfer(src, dst, qty)：把 qty 从 src 行移到 dst 行；qty 小于等于 0 或超过 src 的数量抛 "
        "ValueError，行不存在抛 KeyError，都不改变任何状态；移空后 src 行被删除。\n"
        "4）remove(name) 删除并返回该行，不存在抛 KeyError；qty(name) 返回该行数量；clear() 删除全部行。\n"
        "5）任何修改之后（包括抛出异常的操作），total() 与 top() 都必须等于按当前各行重新计算的结果。",
        '''
        import pricing

        _STALE = object()


        class Bill:
            def __init__(self):
                self._lines = {}
                self._total = None
                self._top = _STALE

            def add(self, name, price, qty):
                price = pricing.unit_price(price)
                if qty <= 0:
                    raise ValueError('qty must be positive')
                self._lines[name] = {'price': price, 'qty': qty}
                self._total = None
                self._top = _STALE

            def merge(self, lines):
                for name, (price, qty) in lines.items():
                    self.add(name, price, qty)
                self._total = sum(item['price'] * item['qty'] for item in self._lines.values())

            def transfer(self, src, dst, qty):
                if qty <= 0:
                    raise ValueError('qty must be positive')
                self._lines[src]['qty'] -= qty
                if self._lines[src]['qty'] <= 0:
                    raise ValueError('not enough qty')
                self._lines[dst]['qty'] += qty
                self._total = None

            def remove(self, name):
                item = self._lines.pop(name)
                self._total = None
                return item

            def clear(self):
                self._lines.clear()
                self._total = None
                self._top = _STALE

            def qty(self, name):
                return self._lines[name]['qty']

            def total(self):
                if self._total is None:
                    self._total = sum(pricing.line_total(item['price'], item['qty'])
                                      for item in self._lines.values())
                return self._total

            def top(self):
                if self._top is _STALE:
                    totals = {name: pricing.line_total(item['price'], item['qty'])
                              for name, item in self._lines.items()}
                    self._top = min(totals, key=lambda name: (-totals[name], name), default=None)
                return self._top
        ''',
        '''
        import pricing

        _STALE = object()


        class Bill:
            def __init__(self):
                self._lines = {}
                self._total = None
                self._top = _STALE

            def _invalidate(self):
                self._total = None
                self._top = _STALE

            def add(self, name, price, qty):
                price = pricing.unit_price(price)
                if qty <= 0:
                    raise ValueError('qty must be positive')
                self._lines[name] = {'price': price, 'qty': qty}
                self._invalidate()

            def merge(self, lines):
                items = {name: {'price': pricing.unit_price(price), 'qty': qty}
                         for name, (price, qty) in lines.items()}
                if any(item['qty'] <= 0 for item in items.values()):
                    raise ValueError('qty must be positive')
                self._lines.update(items)
                self._invalidate()

            def transfer(self, src, dst, qty):
                if qty <= 0:
                    raise ValueError('qty must be positive')
                if src not in self._lines or dst not in self._lines:
                    raise KeyError(src if src not in self._lines else dst)
                if self._lines[src]['qty'] < qty:
                    raise ValueError('not enough qty')
                self._lines[src]['qty'] -= qty
                self._lines[dst]['qty'] += qty
                if not self._lines[src]['qty']:
                    del self._lines[src]
                self._invalidate()

            def remove(self, name):
                item = self._lines.pop(name)
                self._invalidate()
                return item

            def clear(self):
                self._lines.clear()
                self._invalidate()

            def qty(self, name):
                return self._lines[name]['qty']

            def total(self):
                if self._total is None:
                    self._total = sum(pricing.line_total(item['price'], item['qty'])
                                      for item in self._lines.values())
                return self._total

            def top(self):
                if self._top is _STALE:
                    totals = {name: pricing.line_total(item['price'], item['qty'])
                              for name, item in self._lines.items()}
                    self._top = min(totals, key=lambda name: (-totals[name], name), default=None)
                return self._top
        ''',
        '''
        import pytest
        from pricing import BadPrice


        def test_public_add_and_totals():
            bill = Bill()
            bill.add('pen', 3, 2)
            bill.merge({'ink': (5, 1), 'box': (2, 4)})
            assert bill.qty('box') == 4
            assert bill.total() == 19 and bill.top() == 'box'
            with pytest.raises(BadPrice):
                bill.add('glue', 0, 1)
        ''',
        {
            "transfer": '''
                def test_failed_transfer_changes_nothing_and_success_refreshes_caches():
                    bill = Bill()
                    bill.add('pen', 3, 8)
                    bill.add('ink', 5, 2)
                    assert (bill.total(), bill.top()) == (34, 'pen')
                    with pytest.raises(KeyError):
                        bill.transfer('ink', 'zz', 1)
                    with pytest.raises(ValueError):
                        bill.transfer('ink', 'pen', 0)
                    with pytest.raises(ValueError):
                        bill.transfer('ink', 'pen', 3)
                    assert (bill.qty('ink'), bill.qty('pen'), bill.total(), bill.top()) == (2, 8, 34, 'pen')
                    bill.transfer('ink', 'pen', 2)
                    with pytest.raises(KeyError):
                        bill.qty('ink')
                    assert (bill.total(), bill.top()) == (27, 'pen')
                    bill.add('box', 10, 2)
                    assert bill.top() == 'pen'
                    bill.transfer('pen', 'box', 2)
                    assert (bill.qty('pen'), bill.qty('box')) == (8, 4)
                    assert (bill.total(), bill.top()) == (64, 'box')
            ''',
            "bulk": '''
                from pricing import BadPrice


                def test_failed_merge_changes_nothing_and_merge_follows_pricing():
                    bill = Bill()
                    bill.merge({'a': (10, 2), 'b': (3, 8)})
                    assert (bill.total(), bill.top()) == (44, 'b')
                    with pytest.raises(BadPrice):
                        bill.merge({'c': (4, 1), 'd': (0, 5), 'e': (6, 1)})
                    with pytest.raises(ValueError):
                        bill.merge({'c': (4, 1), 'd': (5, 0)})
                    assert (bill.total(), bill.top()) == (44, 'b')
                    with pytest.raises(KeyError):
                        bill.qty('c')
                    bill.merge({'a': (2, 12), 'c': (4, 1)})
                    assert (bill.qty('a'), bill.total(), bill.top()) == (12, 49, 'b')
                    assert bill.remove('b')['qty'] == 8
                    assert (bill.total(), bill.top()) == (25, 'a')
                    bill.clear()
                    assert (bill.total(), bill.top()) == (0, None)
                    bill.merge({'z': (1, 10)})
                    assert (bill.total(), bill.top()) == (9, 'z')
            ''',
            "top": '''
                def test_top_breaks_ties_by_name_and_survives_remove():
                    bill = Bill()
                    assert bill.top() is None and bill.total() == 0
                    bill.add('y', 2, 5)
                    bill.add('x', 5, 2)
                    bill.add('w', 1, 9)
                    assert bill.top() == 'x'
                    bill.remove('x')
                    assert bill.top() == 'y'
                    bill.remove('y')
                    assert bill.top() == 'w'
                ''',
        },
        extra_files={
            "pricing": '''
                """Pricing rules shared by the billing service."""

                TIER_QTY = 10


                class BadPrice(ValueError):
                    """A unit price that is not a positive integer."""


                def unit_price(price):
                    if not isinstance(price, int) or isinstance(price, bool) or price <= 0:
                        raise BadPrice('unit price must be a positive integer')
                    return price


                def line_total(price, qty):
                    unit_price(price)
                    if qty >= TIER_QTY:
                        return price * qty * 9 // 10
                    return price * qty
            ''',
        },
    ),
    _task(
        "pair_bimap", "bimap_store",
        "BiMap 是一对一的双向映射（left ↔ right），配对按创建的先后排序。要求：\n"
        "1）put(left, right)：若二者已经配对则不做任何修改；否则删除 left 和 right 原有的配对（若有），"
        "并把 (left, right) 作为新配对加到末尾。\n"
        "2）rename(old, new)：把配对 (old, r) 改为 (new, r) 并保持原位置；old 不存在抛 KeyError，"
        "new 已与其他 right 配对抛 ValueError，都不改变任何状态；new 等于 old 时不做任何修改。\n"
        "3）pop_left(left) 删除该配对并返回 right，pop_right(right) 删除该配对并返回 left；不存在抛 KeyError。\n"
        "4）get_right(left)、get_left(right) 返回配对的另一侧，不存在抛 KeyError；len() 返回配对数；"
        "clear() 删除全部配对。\n"
        "5）lefts() 与 rights() 返回按配对顺序排列的列表且一一对应：rights() 的第 i 项是 lefts() 第 i 项的配对。\n"
        "6）任何操作之后映射都必须保持一对一：每个 left 只对应一个 right，反之亦然。",
        '''
        class BiMap:
            def __init__(self):
                self._left = {}
                self._right = {}

            def put(self, left, right):
                if left in self._left:
                    del self._right[self._left[left]]
                    del self._left[left]
                self._left[left] = right
                self._right[right] = left

            def rename(self, old, new):
                right = self._left[old]
                del self._left[old]
                self._left[new] = right
                self._right[right] = new

            def pop_left(self, left):
                right = self._left.pop(left)
                del self._right[right]
                return right

            def pop_right(self, right):
                left = self._right[right]
                del self._right[right]
                return left

            def get_left(self, right):
                return self._right[right]

            def get_right(self, left):
                return self._left[left]

            def lefts(self):
                return list(self._left)

            def rights(self):
                return list(self._right)

            def __len__(self):
                return len(self._left)

            def clear(self):
                self._left.clear()
        ''',
        '''
        class BiMap:
            def __init__(self):
                self._left = {}
                self._right = {}

            def put(self, left, right):
                if self._left.get(left) == right:
                    return
                if left in self._left:
                    del self._right[self._left[left]]
                    del self._left[left]
                if right in self._right:
                    del self._left[self._right[right]]
                    del self._right[right]
                self._left[left] = right
                self._right[right] = left

            def rename(self, old, new):
                right = self._left[old]
                if new != old and new in self._left:
                    raise ValueError('new left is already bound')
                self._left = {(new if key == old else key): value for key, value in self._left.items()}
                self._right[right] = new

            def pop_left(self, left):
                right = self._left.pop(left)
                del self._right[right]
                return right

            def pop_right(self, right):
                left = self._right.pop(right)
                del self._left[left]
                return left

            def get_left(self, right):
                return self._right[right]

            def get_right(self, left):
                return self._left[left]

            def lefts(self):
                return list(self._left)

            def rights(self):
                return list(self._right)

            def __len__(self):
                return len(self._left)

            def clear(self):
                self._left.clear()
                self._right.clear()
        ''',
        '''
        def test_public_put_and_lookup():
            pairs = BiMap()
            pairs.put('a', 'x')
            pairs.put('b', 'y')
            assert pairs.get_right('a') == 'x' and pairs.get_left('y') == 'b'
            assert pairs.lefts() == ['a', 'b'] and pairs.rights() == ['x', 'y']
            assert len(pairs) == 2
            pairs.rename('a', 'c')
            assert pairs.pop_left('c') == 'x'
        ''',
        {
            "put": '''
                def test_put_replaces_both_old_pairs_and_is_idempotent():
                    pairs = BiMap()
                    for left, right in (('a', 'x'), ('b', 'y'), ('c', 'z')):
                        pairs.put(left, right)
                    pairs.put('a', 'x')
                    assert pairs.lefts() == ['a', 'b', 'c'] and pairs.rights() == ['x', 'y', 'z']
                    pairs.put('b', 'x')
                    assert pairs.lefts() == ['c', 'b'] and pairs.rights() == ['z', 'x']
                    assert len(pairs) == 2
                    with pytest.raises(KeyError):
                        pairs.get_right('a')
                    with pytest.raises(KeyError):
                        pairs.get_left('y')
                    pairs.put('b', 'w')
                    assert pairs.lefts() == ['c', 'b'] and pairs.rights() == ['z', 'w']
                    assert pairs.get_left('w') == 'b' and pairs.get_right('c') == 'z'
                    pairs.put('q', 'z')
                    assert pairs.lefts() == ['b', 'q'] and pairs.rights() == ['w', 'z']
            ''',
            "rename": '''
                def test_rename_keeps_position_and_one_to_one():
                    pairs = BiMap()
                    for left, right in (('a', 'x'), ('b', 'y'), ('c', 'z')):
                        pairs.put(left, right)
                    pairs.rename('b', 'd')
                    assert pairs.lefts() == ['a', 'd', 'c'] and pairs.rights() == ['x', 'y', 'z']
                    assert pairs.get_left('y') == 'd' and pairs.get_right('d') == 'y'
                    pairs.rename('d', 'd')
                    assert pairs.lefts() == ['a', 'd', 'c']
                    with pytest.raises(ValueError):
                        pairs.rename('a', 'c')
                    with pytest.raises(KeyError):
                        pairs.rename('zz', 'e')
                    assert pairs.lefts() == ['a', 'd', 'c'] and pairs.rights() == ['x', 'y', 'z']
                    assert len(pairs) == 3
                    assert pairs.pop_right('y') == 'd'
                    assert pairs.lefts() == ['a', 'c'] and pairs.rights() == ['x', 'z']
                    with pytest.raises(KeyError):
                        pairs.get_right('d')
                    with pytest.raises(KeyError):
                        pairs.pop_left('d')
                    assert pairs.pop_left('c') == 'z'
                    assert len(pairs) == 1 and pairs.lefts() == ['a'] and pairs.rights() == ['x']
            ''',
            "pop": '''
                def test_pop_and_clear_remove_both_sides():
                    pairs = BiMap()
                    pairs.put('a', 'x')
                    pairs.put('b', 'y')
                    assert pairs.pop_right('x') == 'a'
                    assert pairs.lefts() == ['b'] and pairs.rights() == ['y'] and len(pairs) == 1
                    for call in (lambda: pairs.get_right('a'), lambda: pairs.get_left('x'),
                                 lambda: pairs.pop_right('x')):
                        with pytest.raises(KeyError):
                            call()
                    pairs.put('a', 'z')
                    assert pairs.lefts() == ['b', 'a'] and pairs.rights() == ['y', 'z']
                    assert pairs.pop_left('b') == 'y'
                    assert (len(pairs), pairs.lefts(), pairs.rights()) == (1, ['a'], ['z'])
                    with pytest.raises(KeyError):
                        pairs.pop_left('b')
                    pairs.put('c', 'z')
                    assert pairs.lefts() == ['c'] and pairs.rights() == ['z'] and len(pairs) == 1
                    pairs.clear()
                    assert (len(pairs), pairs.lefts(), pairs.rights()) == (0, [], [])
                    with pytest.raises(KeyError):
                        pairs.get_left('z')
                    pairs.put('d', 'x')
                    assert pairs.lefts() == ['d'] and pairs.rights() == ['x']
            ''',
        },
    ),
    _task(
        "policy_bimap", "bimap_store",
        "Pairs 是一对一的双向映射（left ↔ right），配对按最近使用时间从旧到新排列：新加入的配对排在最后，"
        "get_right / get_left 会使该配对变为最新。每一对 (left, right) 在 put 和 rebind 时都必须通过只读模块 "
        "limits 的 admits(left, right) 检查（任一侧是保留名 root、admin，或两侧忽略大小写后相同，抛 PolicyError）。"
        "要求：\n"
        "1）put(left, right)：不通过检查抛 PolicyError，任一侧已配对抛 ValueError，都不改变任何状态；"
        "否则作为最新配对加入。\n"
        "2）get_right(left)、get_left(right) 返回另一侧并把该配对移到最后，不存在抛 KeyError。\n"
        "3）rebind(left, new_right)：把 left 的配对改为 (left, new_right) 并保持原位置；left 不存在抛 KeyError，"
        "不通过检查抛 PolicyError，new_right 已配对抛 ValueError，都不改变任何状态；"
        "new_right 等于原 right 时不做任何修改。\n"
        "4）swap(a, b) 交换 a、b 各自的 right，两个 left 的位置不变；a 或 b 不存在抛 KeyError。\n"
        "5）pop_left(left)、pop_right(right) 删除该配对并返回另一侧，不存在抛 KeyError。\n"
        "6）lefts() 与 rights() 按上述顺序返回列表且一一对应；任何操作之后都必须保持一对一。",
        '''
        import limits


        class Pairs:
            def __init__(self):
                self._left = {}
                self._right = {}

            def put(self, left, right):
                limits.admits(left, right)
                if left in self._left:
                    raise ValueError('left side already paired')
                self._left[left] = right
                self._right[right] = left

            def _touch(self, left):
                right = self._left.pop(left)
                self._left[left] = right

            def get_right(self, left):
                right = self._left[left]
                self._touch(left)
                return right

            def get_left(self, right):
                left = self._right[right]
                self._touch(left)
                return left

            def rebind(self, left, new_right):
                old_right = self._left[left]
                if new_right in self._right:
                    raise ValueError('right side already paired')
                del self._right[old_right]
                self._left[left] = new_right
                self._right[new_right] = left

            def swap(self, first, second):
                self._left[first], self._left[second] = self._left[second], self._left[first]

            def pop_left(self, left):
                right = self._left.pop(left)
                del self._right[right]
                return right

            def pop_right(self, right):
                left = self._right.pop(right)
                del self._left[left]
                return left

            def lefts(self):
                return list(self._left)

            def rights(self):
                return list(self._right)
        ''',
        '''
        import limits


        def _rekey(mapping, old, new):
            return {new if key == old else key: value for key, value in mapping.items()}


        class Pairs:
            def __init__(self):
                self._left = {}
                self._right = {}

            def put(self, left, right):
                limits.admits(left, right)
                if left in self._left or right in self._right:
                    raise ValueError('side already paired')
                self._left[left] = right
                self._right[right] = left

            def _touch(self, left):
                right = self._left[left]
                del self._left[left]
                self._left[left] = right
                del self._right[right]
                self._right[right] = left

            def get_right(self, left):
                right = self._left[left]
                self._touch(left)
                return right

            def get_left(self, right):
                left = self._right[right]
                self._touch(left)
                return left

            def rebind(self, left, new_right):
                old_right = self._left[left]
                limits.admits(left, new_right)
                if new_right != old_right and new_right in self._right:
                    raise ValueError('right side already paired')
                self._left[left] = new_right
                self._right = _rekey(self._right, old_right, new_right)

            def swap(self, first, second):
                first_right = self._left[first]
                second_right = self._left[second]
                self._left[first], self._left[second] = second_right, first_right
                self._right = {right: left for left, right in self._left.items()}

            def pop_left(self, left):
                right = self._left.pop(left)
                del self._right[right]
                return right

            def pop_right(self, right):
                left = self._right.pop(right)
                del self._left[left]
                return left

            def lefts(self):
                return list(self._left)

            def rights(self):
                return list(self._right)
        ''',
        '''
        import pytest
        from limits import PolicyError


        def test_public_put_and_lookup():
            pairs = Pairs()
            pairs.put('a', 'x')
            pairs.put('b', 'y')
            assert pairs.get_right('a') == 'x' and pairs.get_left('y') == 'b'
            assert sorted(pairs.lefts()) == ['a', 'b'] and sorted(pairs.rights()) == ['x', 'y']
            assert pairs.pop_left('b') == 'y'
            with pytest.raises(PolicyError):
                pairs.put('c', 'root')
        ''',
        {
            "order": '''
                def test_lookups_move_pair_to_newest_on_both_sides():
                    pairs = Pairs()
                    for left, right in (('a', 'x'), ('b', 'y'), ('c', 'z')):
                        pairs.put(left, right)
                    assert pairs.get_right('a') == 'x'
                    assert (pairs.lefts(), pairs.rights()) == (['b', 'c', 'a'], ['y', 'z', 'x'])
                    assert pairs.get_left('z') == 'c'
                    assert (pairs.lefts(), pairs.rights()) == (['b', 'a', 'c'], ['y', 'x', 'z'])
                    for call in (lambda: pairs.get_right('zz'), lambda: pairs.get_left('zz')):
                        with pytest.raises(KeyError):
                            call()
                    assert (pairs.lefts(), pairs.rights()) == (['b', 'a', 'c'], ['y', 'x', 'z'])
                    assert pairs.pop_left('a') == 'x'
                    pairs.put('d', 'w')
                    assert (pairs.lefts(), pairs.rights()) == (['b', 'c', 'd'], ['y', 'z', 'w'])
                    assert pairs.pop_right('y') == 'b'
                    assert (pairs.lefts(), pairs.rights()) == (['c', 'd'], ['z', 'w'])
            ''',
            "rebind": '''
                from limits import PolicyError


                def test_rebind_checks_policy_and_keeps_position():
                    pairs = Pairs()
                    for left, right in (('a', 'x'), ('b', 'y'), ('c', 'z')):
                        pairs.put(left, right)
                    pairs.rebind('b', 'w')
                    assert (pairs.lefts(), pairs.rights()) == (['a', 'b', 'c'], ['x', 'w', 'z'])
                    assert pairs.get_right('b') == 'w'
                    assert (pairs.lefts(), pairs.rights()) == (['a', 'c', 'b'], ['x', 'z', 'w'])
                    pairs.rebind('b', 'w')
                    assert (pairs.lefts(), pairs.rights()) == (['a', 'c', 'b'], ['x', 'z', 'w'])
                    with pytest.raises(PolicyError):
                        pairs.rebind('b', 'root')
                    with pytest.raises(PolicyError):
                        pairs.rebind('a', 'A')
                    with pytest.raises(ValueError):
                        pairs.rebind('a', 'z')
                    with pytest.raises(KeyError):
                        pairs.rebind('zz', 'q')
                    assert (pairs.lefts(), pairs.rights()) == (['a', 'c', 'b'], ['x', 'z', 'w'])
                    assert pairs.get_left('x') == 'a' and pairs.get_right('c') == 'z'
            ''',
            "swap": '''
                from limits import PolicyError


                def test_swap_keeps_one_to_one_and_put_checks_both_sides():
                    pairs = Pairs()
                    for left, right in (('a', 'x'), ('b', 'y'), ('c', 'z')):
                        pairs.put(left, right)
                    pairs.swap('a', 'c')
                    assert (pairs.lefts(), pairs.rights()) == (['a', 'b', 'c'], ['z', 'y', 'x'])
                    assert pairs.get_left('x') == 'c' and pairs.get_right('a') == 'z'
                    assert (pairs.lefts(), pairs.rights()) == (['b', 'c', 'a'], ['y', 'x', 'z'])
                    with pytest.raises(KeyError):
                        pairs.swap('a', 'zz')
                    pairs.swap('b', 'b')
                    assert (pairs.lefts(), pairs.rights()) == (['b', 'c', 'a'], ['y', 'x', 'z'])
                    for call, error in ((lambda: pairs.put('d', 'x'), ValueError),
                                        (lambda: pairs.put('a', 'q'), ValueError),
                                        (lambda: pairs.put('d', 'admin'), PolicyError),
                                        (lambda: pairs.put('q', 'Q'), PolicyError)):
                        with pytest.raises(error):
                            call()
                    assert (pairs.lefts(), pairs.rights()) == (['b', 'c', 'a'], ['y', 'x', 'z'])
            ''',
        },
        extra_files={
            "limits": '''
                """Admission rules for the pairing service."""

                RESERVED = frozenset({'root', 'admin'})


                class PolicyError(ValueError):
                    """A pair that the policy never admits."""


                def admits(left, right):
                    if not isinstance(left, str) or not isinstance(right, str):
                        raise PolicyError('names must be text')
                    if left in RESERVED or right in RESERVED:
                        raise PolicyError('reserved names cannot be paired')
                    if left.casefold() == right.casefold():
                        raise PolicyError('a name cannot pair with itself')
                    return True
            ''',
        },
    ),
    _task(
        "session_lru", "session_registry",
        "Sessions 管理会话：每个会话有 user、到期时刻 expires 和状态（active / expired / revoked）；"
        "对 active 会话，now ≥ expires 即为到期。要求：\n"
        "1）open(user, ttl, now) 新建 active 会话并返回 id（从 1 起递增），expires = now + ttl；"
        "ttl 小于等于 0 抛 ValueError。\n"
        "2）get / touch / renew / revoke 遇到已到期但仍为 active 的会话时，先把它的状态改为 expired "
        "并移出 LRU，再按已过期的会话处理；id 不存在抛 KeyError。\n"
        "3）get(id, now) 返回 user 并把会话移到最近使用位置；会话已过期或已吊销返回 None。\n"
        "4）touch(id, now) 把 active 会话移到最近使用位置并返回 True；否则返回 False。\n"
        "5）renew(id, ttl, now) 把 active 会话的 expires 改为 now + ttl 并移到最近使用位置，"
        "返回新的 expires；ttl 小于等于 0 抛 ValueError；会话已过期或已吊销返回 None。\n"
        "6）revoke(id, now)：active 会话改为 revoked 并移出 LRU，返回 revoked；已吊销返回 revoked；"
        "已过期保持 expired 并返回 expired。\n"
        "7）sweep(now) 把所有到期的 active 会话改为 expired 并移出 LRU，返回其 id 列表（LRU 顺序）。"
        "lru() 返回 active 会话的 id（最久未用在前）；status(id) 返回状态。",
        '''
        class Sessions:
            def __init__(self):
                self._sessions = {}
                self._status = {}
                self._lru = []
                self._next_id = 1

            def open(self, user, ttl, now):
                if ttl <= 0:
                    raise ValueError('ttl must be positive')
                sid = self._next_id
                self._next_id += 1
                self._sessions[sid] = {'user': user, 'expires': now + ttl}
                self._status[sid] = 'active'
                self._lru.append(sid)
                return sid

            def get(self, sid, now):
                if sid not in self._sessions:
                    raise KeyError(sid)
                if self._status[sid] == 'active' and now >= self._sessions[sid]['expires']:
                    return None
                if self._status[sid] != 'active':
                    return None
                self._lru.remove(sid)
                self._lru.append(sid)
                return self._sessions[sid]['user']

            def touch(self, sid, now):
                if self._status[sid] != 'active':
                    return False
                self._lru.remove(sid)
                self._lru.append(sid)
                return True

            def renew(self, sid, ttl, now):
                if ttl <= 0:
                    raise ValueError('ttl must be positive')
                self._sessions[sid]['expires'] = now + ttl
                self._lru.remove(sid)
                self._lru.append(sid)
                return now + ttl

            def revoke(self, sid, now):
                if sid not in self._sessions:
                    raise KeyError(sid)
                self._status[sid] = 'revoked'

            def sweep(self, now):
                due = [sid for sid in self._lru if now >= self._sessions[sid]['expires']]
                for sid in due:
                    self._status[sid] = 'expired'
                return due

            def status(self, sid):
                return self._status[sid]

            def lru(self):
                return list(self._lru)
        ''',
        '''
        class Sessions:
            def __init__(self):
                self._sessions = {}
                self._status = {}
                self._lru = []
                self._next_id = 1

            def open(self, user, ttl, now):
                if ttl <= 0:
                    raise ValueError('ttl must be positive')
                sid = self._next_id
                self._next_id += 1
                self._sessions[sid] = {'user': user, 'expires': now + ttl}
                self._status[sid] = 'active'
                self._lru.append(sid)
                return sid

            def _lookup(self, sid, now):
                if sid not in self._sessions:
                    raise KeyError(sid)
                if self._status[sid] == 'active' and now >= self._sessions[sid]['expires']:
                    self._status[sid] = 'expired'
                    self._lru.remove(sid)
                return self._status[sid]

            def _bump(self, sid):
                self._lru.remove(sid)
                self._lru.append(sid)

            def get(self, sid, now):
                if self._lookup(sid, now) != 'active':
                    return None
                self._bump(sid)
                return self._sessions[sid]['user']

            def touch(self, sid, now):
                if self._lookup(sid, now) != 'active':
                    return False
                self._bump(sid)
                return True

            def renew(self, sid, ttl, now):
                if ttl <= 0:
                    raise ValueError('ttl must be positive')
                if self._lookup(sid, now) != 'active':
                    return None
                self._sessions[sid]['expires'] = now + ttl
                self._bump(sid)
                return now + ttl

            def revoke(self, sid, now):
                status = self._lookup(sid, now)
                if status == 'active':
                    self._status[sid] = 'revoked'
                    self._lru.remove(sid)
                return self._status[sid]

            def sweep(self, now):
                due = [sid for sid in self._lru if now >= self._sessions[sid]['expires']]
                for sid in due:
                    self._lookup(sid, now)
                return due

            def status(self, sid):
                return self._status[sid]

            def lru(self):
                return list(self._lru)
        ''',
        '''
        def test_public_open_and_get():
            sessions = Sessions()
            first = sessions.open('ann', 10, now=0)
            second = sessions.open('bob', 5, now=0)
            assert sessions.get(first, now=3) == 'ann'
            assert sessions.status(second) == 'active'
            assert sessions.lru() == [second, first]
            assert sessions.touch(second, now=4) is True
            assert sessions.lru() == [first, second]
        ''',
        {
            "expire": '''
                def test_due_session_is_finalized_by_every_lookup():
                    sessions = Sessions()
                    first = sessions.open('ann', 5, now=0)
                    second = sessions.open('bob', 5, now=0)
                    assert sessions.get(second, now=4) == 'bob' and sessions.lru() == [first, second]
                    assert sessions.get(first, now=5) is None
                    assert sessions.status(first) == 'expired' and sessions.lru() == [second]
                    assert sessions.get(first, now=6) is None and sessions.lru() == [second]
                    assert sessions.touch(second, now=5) is False
                    assert sessions.status(second) == 'expired' and sessions.lru() == []
                    assert sessions.renew(first, 5, now=6) is None
                    assert sessions.status(first) == 'expired'
                    for call in (lambda: sessions.get(99, now=1), lambda: sessions.touch(99, now=1),
                                 lambda: sessions.revoke(99, now=1)):
                        with pytest.raises(KeyError):
                            call()
            ''',
            "renew": '''
                def test_renew_and_revoke_only_act_on_active_sessions():
                    sessions = Sessions()
                    first = sessions.open('ann', 5, now=0)
                    second = sessions.open('bob', 5, now=0)
                    assert sessions.renew(first, 10, now=3) == 13
                    assert (sessions.lru(), sessions.get(first, now=12)) == ([second, first], 'ann')
                    with pytest.raises(ValueError):
                        sessions.renew(first, 0, now=12)
                    assert sessions.revoke(first, now=12) == 'revoked'
                    assert sessions.renew(first, 20, now=12) is None
                    assert sessions.get(first, now=12) is None and sessions.touch(first, now=12) is False
                    assert (sessions.status(first), sessions.lru()) == ('revoked', [second])
                    assert sessions.revoke(second, now=20) == 'expired'
                    assert (sessions.status(second), sessions.lru()) == ('expired', [])
                    assert sessions.sweep(now=20) == []
                    with pytest.raises(KeyError):
                        sessions.renew(99, 5, now=20)
                    with pytest.raises(KeyError):
                        sessions.revoke(99, now=20)
            ''',
            "sweep": '''
                def test_sweep_and_revoke_keep_lru_in_sync():
                    sessions = Sessions()
                    ids = [sessions.open(user, ttl, now=0) for user, ttl in (('a', 3), ('b', 9), ('c', 6), ('d', 20))]
                    assert sessions.touch(ids[2], now=1) is True
                    assert sessions.lru() == [ids[0], ids[1], ids[3], ids[2]]
                    assert sessions.sweep(now=6) == [ids[0], ids[2]]
                    assert sessions.lru() == [ids[1], ids[3]]
                    states = [sessions.status(sid) for sid in ids]
                    assert states == ['expired', 'active', 'expired', 'active']
                    assert sessions.sweep(now=6) == []
                    assert sessions.revoke(ids[1], now=7) == 'revoked'
                    assert sessions.revoke(ids[1], now=7) == 'revoked'
                    assert (sessions.status(ids[1]), sessions.lru()) == ('revoked', [ids[3]])
                    assert sessions.revoke(ids[0], now=7) == 'expired'
                    assert sessions.status(ids[0]) == 'expired' and sessions.lru() == [ids[3]]
                    assert sessions.sweep(now=25) == [ids[3]]
                    assert sessions.lru() == [] and sessions.status(ids[3]) == 'expired'
            ''',
        },
    ),
    _task(
        "clock_registry", "session_registry",
        "Registry 管理会话，当前时间只能从只读模块 clock 的 now() 获得（整数；测试用 clock.advance(n) 推进）。"
        "会话状态为 active / expired / revoked / evicted；对 active 会话，now() ≥ expires 即为到期。要求：\n"
        "1）open(user, ttl)：ttl 小于等于 0 抛 ValueError；否则先执行 sweep()，再新建 active 会话并返回 id"
        "（从 1 起递增），放在最近使用位置；若此时 active 会话数超过 capacity，把最久未用的 active 会话改为 "
        "evicted。\n"
        "2）get / renew / revoke 遇到已到期的 active 会话时，先把它改为 expired 并移出 LRU，再按已过期的会话处理；"
        "id 不存在抛 KeyError。\n"
        "3）get(id) 返回 user 并把会话移到最近使用位置；expired 抛 TimeoutError，revoked 或 evicted 抛 "
        "PermissionError。\n"
        "4）renew(id, ttl)：把 active 会话的 expires 改为 now() + ttl 并移到最近使用位置；ttl 小于等于 0 抛 "
        "ValueError；会话不可用时抛与 get 相同的异常，除此之外不做任何修改。\n"
        "5）revoke(id)：把 active 会话改为 revoked 并移出 LRU；已过期、已吊销、已淘汰的会话保持原状。\n"
        "6）sweep() 把所有到期的 active 会话改为 expired 并移出 LRU，返回它们的 id（LRU 顺序）。\n"
        "7）lru() 返回 active 会话的 id（最久未用在前）；status(id) 返回会话状态。",
        '''
        import clock


        class Registry:
            def __init__(self, capacity):
                self._sessions = {}
                self._status = {}
                self._lru = []
                self._capacity = capacity
                self._next_id = 1

            def status(self, sid):
                return self._status[sid]

            def _check(self, sid):
                if clock.now() >= self._sessions[sid]['expires']:
                    raise TimeoutError(sid)
                if self._status[sid] != 'active':
                    raise PermissionError(sid)

            def _leave(self, sid, status):
                self._status[sid] = status
                self._lru.remove(sid)

            def sweep(self):
                due = [sid for sid in self._lru if clock.now() >= self._sessions[sid]['expires']]
                for sid in due:
                    self._leave(sid, 'expired')
                return due

            def open(self, user, ttl):
                if ttl <= 0:
                    raise ValueError('ttl must be positive')
                sid = self._next_id
                self._next_id += 1
                self._sessions[sid] = {'user': user, 'expires': clock.now() + ttl}
                self._status[sid] = 'active'
                self._lru.append(sid)
                if len(self._lru) > self._capacity:
                    self._status[self._lru[0]] = 'evicted'
                return sid

            def get(self, sid):
                self._check(sid)
                self._lru.remove(sid)
                self._lru.append(sid)
                return self._sessions[sid]['user']

            def renew(self, sid, ttl):
                if ttl <= 0:
                    raise ValueError('ttl must be positive')
                self._check(sid)
                self._sessions[sid]['expires'] = clock.now() + ttl

            def revoke(self, sid):
                self._status[sid] = 'revoked'

            def lru(self):
                return list(self._lru)
        ''',
        '''
        import clock


        class Registry:
            def __init__(self, capacity):
                self._sessions = {}
                self._status = {}
                self._lru = []
                self._capacity = capacity
                self._next_id = 1

            def status(self, sid):
                return self._status[sid]

            def _check(self, sid):
                status = self._status[sid]
                if status == 'active' and clock.now() >= self._sessions[sid]['expires']:
                    self._leave(sid, 'expired')
                    raise TimeoutError(sid)
                if status != 'active':
                    raise (TimeoutError if status == 'expired' else PermissionError)(sid)

            def _leave(self, sid, status):
                self._status[sid] = status
                self._lru.remove(sid)

            def sweep(self):
                due = [sid for sid in self._lru if clock.now() >= self._sessions[sid]['expires']]
                for sid in due:
                    self._leave(sid, 'expired')
                return due

            def open(self, user, ttl):
                if ttl <= 0:
                    raise ValueError('ttl must be positive')
                self.sweep()
                sid = self._next_id
                self._next_id += 1
                self._sessions[sid] = {'user': user, 'expires': clock.now() + ttl}
                self._status[sid] = 'active'
                self._lru.append(sid)
                if len(self._lru) > self._capacity:
                    self._leave(self._lru[0], 'evicted')
                return sid

            def get(self, sid):
                self._check(sid)
                self._lru.remove(sid)
                self._lru.append(sid)
                return self._sessions[sid]['user']

            def renew(self, sid, ttl):
                if ttl <= 0:
                    raise ValueError('ttl must be positive')
                self._check(sid)
                self._sessions[sid]['expires'] = clock.now() + ttl
                self._lru.remove(sid)
                self._lru.append(sid)

            def revoke(self, sid):
                if self._status[sid] == 'active':
                    due = clock.now() >= self._sessions[sid]['expires']
                    self._leave(sid, 'expired' if due else 'revoked')

            def lru(self):
                return list(self._lru)
        ''',
        '''
        import clock


        def test_public_open_and_get():
            clock.reset()
            registry = Registry(capacity=3)
            first = registry.open('ann', ttl=10)
            second = registry.open('bob', ttl=10)
            assert registry.get(first) == 'ann'
            assert registry.lru() == [second, first]
            assert registry.status(first) == 'active'
            clock.advance(3)
            assert registry.sweep() == []
        ''',
        {
            "evict": '''
                import clock


                def test_open_sweeps_before_evicting_and_eviction_leaves_the_lru():
                    clock.reset()
                    registry = Registry(capacity=2)
                    first = registry.open('a', ttl=5)
                    second = registry.open('b', ttl=50)
                    clock.advance(6)
                    third = registry.open('c', ttl=50)
                    assert (registry.status(first), registry.status(third)) == ('expired', 'active')
                    assert registry.lru() == [second, third]
                    assert registry.get(third) == 'c'
                    fourth = registry.open('d', ttl=50)
                    assert (registry.status(second), registry.lru()) == ('evicted', [third, fourth])
                    with pytest.raises(PermissionError):
                        registry.get(second)
                    with pytest.raises(ValueError):
                        registry.open('e', ttl=0)
                    assert registry.lru() == [third, fourth]
            ''',
            "renew": '''
                import clock


                def test_renew_moves_to_newest_and_failed_calls_change_nothing():
                    clock.reset()
                    registry = Registry(capacity=5)
                    first = registry.open('a', ttl=20)
                    second = registry.open('b', ttl=30)
                    third = registry.open('c', ttl=30)
                    registry.renew(first, ttl=20)
                    assert registry.lru() == [second, third, first]
                    clock.advance(5)
                    assert registry.get(first) == 'a' and registry.lru() == [second, third, first]
                    clock.advance(15)
                    with pytest.raises(TimeoutError):
                        registry.get(first)
                    assert (registry.status(first), registry.lru()) == ('expired', [second, third])
                    with pytest.raises(TimeoutError):
                        registry.renew(first, ttl=5)
                    with pytest.raises(ValueError):
                        registry.renew(second, ttl=0)
                    registry.revoke(second)
                    clock.advance(11)
                    with pytest.raises(PermissionError):
                        registry.get(second)
                    with pytest.raises(PermissionError):
                        registry.renew(second, ttl=5)
                    assert (registry.status(second), registry.lru()) == ('revoked', [third])
                    with pytest.raises(KeyError):
                        registry.get(99)
            ''',
            "sweep": '''
                import clock


                def test_sweep_and_revoke_agree_on_expired_sessions():
                    clock.reset()
                    registry = Registry(capacity=5)
                    ids = [registry.open(user, ttl=ttl) for user, ttl in (('a', 3), ('b', 9), ('c', 6), ('d', 30))]
                    assert registry.get(ids[2]) == 'c'
                    clock.advance(6)
                    assert registry.sweep() == [ids[0], ids[2]]
                    assert registry.lru() == [ids[1], ids[3]]
                    assert [registry.status(sid) for sid in ids] == ['expired', 'active', 'expired', 'active']
                    assert registry.sweep() == []
                    registry.revoke(ids[0])
                    assert registry.status(ids[0]) == 'expired'
                    registry.revoke(ids[1])
                    assert (registry.status(ids[1]), registry.lru()) == ('revoked', [ids[3]])
                    clock.advance(30)
                    registry.revoke(ids[3])
                    assert (registry.status(ids[3]), registry.lru()) == ('expired', [])
                    with pytest.raises(KeyError):
                        registry.revoke(99)
            ''',
        },
        extra_files={
            "clock": '''
                """Wall clock of the session service."""

                _ticks = 0


                def now():
                    return _ticks


                def advance(seconds):
                    global _ticks
                    _ticks += seconds
                    return _ticks


                def reset():
                    global _ticks
                    _ticks = 0
            ''',
        },
    ),
]

MUTANTS = {
    "h3_key_index_table_001": [
        ("            self._unlink(row_id)\n            row['key'] = new_key\n",
         "            row['key'] = new_key\n"),
        ("        self._unlink(row_id)\n        return self._rows.pop(row_id)\n",
         "        return self._rows.pop(row_id)\n"),
        ("        self._rows.clear()\n        self._index.clear()\n", "        self._rows.clear()\n"),
        ("        self._index.clear()\n        self._next_id = 1\n", "        self._index.clear()\n"),
        ("return self.remove(self._index[key][0])", "return self.remove(self._index[key][-1])"),
    ],
    "h3_norm_key_index_001": [
        ("        self._next_id += len(items)\n", ""),
        ("        if row['key'] != new_key:\n", "        if True:\n"),
        ("        new_key = normalize(new_key)\n"
         "        row = self._rows[row_id]\n"
         "        if row['key'] != new_key:\n"
         "            self._unlink(row_id)\n"
         "            row['key'] = new_key\n",
         "        row = self._rows[row_id]\n"
         "        if row['key'] != new_key:\n"
         "            self._unlink(row_id)\n"
         "            row['key'] = normalize(new_key)\n"),
        ("row_id = self._index[normalize(key)][0]", "row_id = self._index[key][0]"),
        ("items = [(normalize(key), value) for key, value in pairs]",
         "items = [(key, value) for key, value in pairs]"),
    ],
    "h3_stock_totals_001": [
        ("            item['price'] = price\n        self._invalidate()\n",
         "            item['price'] = price\n"),
        ("        self._items[name]['price'] = price\n        self._invalidate()\n",
         "        self._items[name]['price'] = price\n"),
        ("        item = self._items.pop(name)\n        self._invalidate()\n",
         "        item = self._items.pop(name)\n"),
        ("    def apply(self, quantities):\n        self._invalidate()\n",
         "    def apply(self, quantities):\n"),
        ("        self._items.clear()\n        self._invalidate()\n", "        self._items.clear()\n"),
    ],
    "h3_priced_bill_001": [
        ("        if self._lines[src]['qty'] < qty:\n"
         "            raise ValueError('not enough qty')\n"
         "        self._lines[src]['qty'] -= qty\n",
         "        self._lines[src]['qty'] -= qty\n"
         "        if self._lines[src]['qty'] < 0:\n"
         "            raise ValueError('not enough qty')\n"),
        ("        if not self._lines[src]['qty']:\n"
         "            del self._lines[src]\n"
         "        self._invalidate()\n",
         "        if not self._lines[src]['qty']:\n"
         "            del self._lines[src]\n"
         "        self._total = None\n"),
        ("        items = {name: {'price': pricing.unit_price(price), 'qty': qty}\n"
         "                 for name, (price, qty) in lines.items()}\n"
         "        if any(item['qty'] <= 0 for item in items.values()):\n"
         "            raise ValueError('qty must be positive')\n"
         "        self._lines.update(items)\n",
         "        for name, (price, qty) in lines.items():\n"
         "            self.add(name, price, qty)\n"),
        ("        self._lines.update(items)\n        self._invalidate()\n",
         "        self._lines.update(items)\n        self._invalidate()\n"
         "        self._total = sum(item['price'] * item['qty'] for item in self._lines.values())\n"),
        ("        item = self._lines.pop(name)\n        self._invalidate()\n",
         "        item = self._lines.pop(name)\n        self._total = None\n"),
    ],
    "h3_pair_bimap_001": [
        ("        if right in self._right:\n"
         "            del self._left[self._right[right]]\n"
         "            del self._right[right]\n", ""),
        ("        if self._left.get(left) == right:\n            return\n", ""),
        ("        if new != old and new in self._left:\n"
         "            raise ValueError('new left is already bound')\n", ""),
        ("        self._left = {(new if key == old else key): value for key, value in self._left.items()}\n",
         "        self._left[new] = self._left.pop(old)\n"),
        ("        left = self._right.pop(right)\n        del self._left[left]\n",
         "        left = self._right.pop(right)\n"),
    ],
    "h3_policy_bimap_001": [
        ("        if left in self._left or right in self._right:\n", "        if left in self._left:\n"),
        ("        del self._right[right]\n        self._right[right] = left\n", ""),
        ("        limits.admits(left, new_right)\n", ""),
        ("        self._right = _rekey(self._right, old_right, new_right)\n",
         "        del self._right[old_right]\n        self._right[new_right] = left\n"),
        ("        self._right = {right: left for left, right in self._left.items()}\n",
         "        self._right[first_right], self._right[second_right] = second, first\n"),
    ],
    "h3_session_lru_001": [
        ("        if self._status[sid] == 'active' and now >= self._sessions[sid]['expires']:\n",
         "        if self._status[sid] == 'active' and now > self._sessions[sid]['expires']:\n"),
        ("            self._status[sid] = 'expired'\n            self._lru.remove(sid)\n",
         "            self._status[sid] = 'expired'\n"),
        ("            self._status[sid] = 'revoked'\n            self._lru.remove(sid)\n",
         "            self._status[sid] = 'revoked'\n"),
        ("        if status == 'active':\n", "        if status != 'revoked':\n"),
        ("        if self._lookup(sid, now) != 'active':\n            return None\n"
         "        self._sessions[sid]['expires'] = now + ttl\n",
         "        self._sessions[sid]['expires'] = now + ttl\n"),
    ],
    "h3_clock_registry_001": [
        ("        self.sweep()\n", ""),
        ("            self._leave(self._lru[0], 'evicted')\n",
         "            self._status[self._lru[0]] = 'evicted'\n"),
        ("            self._leave(sid, 'expired')\n            raise TimeoutError(sid)\n",
         "            raise TimeoutError(sid)\n"),
        ("            raise (TimeoutError if status == 'expired' else PermissionError)(sid)\n",
         "            raise TimeoutError(sid)\n"),
        ("        self._sessions[sid]['expires'] = clock.now() + ttl\n"
         "        self._lru.remove(sid)\n        self._lru.append(sid)\n",
         "        self._sessions[sid]['expires'] = clock.now() + ttl\n"),
    ],
}
