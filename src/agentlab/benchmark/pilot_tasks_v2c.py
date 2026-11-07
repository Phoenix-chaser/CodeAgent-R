"""Twenty v2c repair tasks in ten more source groups (scale-up batch 3).

New families distinct from all previous pools: graph adjacency, string
templates, queue semantics, coordinate moves, CSV splitting, range stepping,
banking ledger, ordinal indexing, inventory diff and slug generation.
Every buggy version passes its public tests but fails hidden private tests.
"""
from __future__ import annotations

from textwrap import dedent


def _task(name, group, instruction, buggy, fixed, public_tests, private_tests,
          difficulty="L2"):
    module = f"v2c_{name}"
    return {
        "task_id": f"v2c_{name}_001", "source_group": f"v2c_{group}",
        "split": "train", "bug_family": group, "difficulty": difficulty,
        "instruction": instruction, "module": f"{module}.py",
        "buggy": dedent(buggy).strip() + "\n",
        "fixed": dedent(fixed).strip() + "\n",
        "public_test": f"from {module} import *\n\n" + dedent(public_tests).strip() + "\n",
        "private_tests": {"test_regression.py":
                          f"import pytest\nfrom {module} import *\n\n"
                          + dedent(private_tests).strip() + "\n"},
    }


V2C_SPECS = [
    # ── 来源组 1：v2c_graph_adjacency ──────────────────────────────────
    _task("degree_list", "graph_adjacency",
          "修复 degree_list(n, edges)：n 个节点编号 0..n-1，edges 为无向边列表"
          "（每条边必须是恰好两个端点的 tuple）；返回每个节点的度数列表。"
          "自环 (u, u) 使该节点度数 +2。n 必须为正整数（bool 不视为整数）；"
          "端点越界、端点非 int（bool 不视为整数）或边格式非法一律 ValueError。",
          '''
          def degree_list(n, edges):
              degrees = [0] * n
              for u, v in edges:
                  degrees[u] += 1
                  degrees[v] += 1
              return degrees
          ''',
          '''
          def degree_list(n, edges):
              if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                  raise ValueError('n must be a positive integer')
              degrees = [0] * n
              for edge in edges:
                  if not isinstance(edge, tuple) or len(edge) != 2:
                      raise ValueError('edges must be (u, v) tuples')
                  u, v = edge
                  for node in (u, v):
                      if (isinstance(node, bool) or not isinstance(node, int)
                              or not 0 <= node < n):
                          raise ValueError('edge endpoint out of range')
                  if u == v:
                      degrees[u] += 2
                  else:
                      degrees[u] += 1
                      degrees[v] += 1
              return degrees
          ''',
          '''
          def test_simple_degrees():
              assert degree_list(3, [(0, 1), (1, 2)]) == [1, 2, 1]
              assert degree_list(2, [(0, 1)]) == [1, 1]
          ''',
          '''
          import pytest

          def test_self_loop_counts_twice():
              assert degree_list(2, [(1, 1)]) == [0, 2]

          def test_validation():
              with pytest.raises(ValueError):
                  degree_list(0, [])
              with pytest.raises(ValueError):
                  degree_list(3, [(0, 5)])
              with pytest.raises(ValueError):
                  degree_list(3, [(True, 0)])
              with pytest.raises(ValueError):
                  degree_list(3, [[0, 1]])
          '''),
    _task("isolated_nodes", "graph_adjacency",
          "修复 isolated_nodes(n, edges)：无向图（edges 每条为二元 tuple，"
          "端点须在 0..n-1 且为 int，bool 不视为整数；边格式非法 ValueError），"
          "返回度为 0 的节点升序列表；n 正整数（bool 不视为整数）否则 ValueError；"
          "两端都要计入度数，不能只看第一端。",
          '''
          def isolated_nodes(n, edges):
              connected = set()
              for u, v in edges:
                  connected.add(u)
              return [node for node in range(n) if node not in connected]
          ''',
          '''
          def isolated_nodes(n, edges):
              if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                  raise ValueError('n must be a positive integer')
              connected = set()
              for edge in edges:
                  if not isinstance(edge, tuple) or len(edge) != 2:
                      raise ValueError('edges must be (u, v) tuples')
                  u, v = edge
                  for node in (u, v):
                      if (isinstance(node, bool) or not isinstance(node, int)
                              or not 0 <= node < n):
                          raise ValueError('edge endpoint out of range')
                  connected.add(u)
                  connected.add(v)
              return [node for node in range(n) if node not in connected]
          ''',
          '''
          def test_basic_isolation():
              assert isolated_nodes(4, [(2, 2)]) == [0, 1, 3]
              assert isolated_nodes(2, []) == [0, 1]
          ''',
          '''
          import pytest

          def test_second_endpoint_counts():
              assert isolated_nodes(3, [(2, 0)]) == [1]
              assert isolated_nodes(4, [(0, 1)]) == [2, 3]

          def test_validation():
              with pytest.raises(ValueError):
                  isolated_nodes(0, [])
              with pytest.raises(ValueError):
                  isolated_nodes(2, [(0, 9)])
          '''),
    # ── 来源组 2：v2c_string_templates ─────────────────────────────────
    _task("render_template", "string_templates",
          "修复 render_template(template, values)：把 '{key}' 占位符替换为 "
          "str(values[key])，从左到右一次扫描，替换结果不再被扫描。"
          "'{{' 与 '}}' 是转义，分别输出字面 '{' 与 '}'；单独的 '}'（后面不是 '}'）"
          "与未闭合的 '{' 都是非法，抛 ValueError；占位符为空 '{}'"
          "或 key 不在 values（values 必须 dict）中抛 ValueError。"
          "key 可以是任意不含大括号的字符串（不要求是合法标识符）。",
          '''
          def render_template(template, values):
              return template.format(**values)
          ''',
          '''
          def render_template(template, values):
              if not isinstance(values, dict):
                  raise ValueError('values must be a dict')
              out = []
              i = 0
              n = len(template)
              while i < n:
                  ch = template[i]
                  if ch == '{':
                      if i + 1 < n and template[i + 1] == '{':
                          out.append('{')
                          i += 2
                          continue
                      end = template.find('}', i + 1)
                      if end == -1:
                          raise ValueError('unclosed placeholder')
                      key = template[i + 1:end]
                      if not key:
                          raise ValueError('empty placeholder')
                      if '{' in key:
                          raise ValueError('nested opening brace in placeholder')
                      if key not in values:
                          raise ValueError(f'unknown placeholder: {key}')
                      out.append(str(values[key]))
                      i = end + 1
                  elif ch == '}':
                      if i + 1 < n and template[i + 1] == '}':
                          out.append('}')
                          i += 2
                      else:
                          raise ValueError('unescaped closing brace')
                  else:
                      out.append(ch)
                      i += 1
              return ''.join(out)
          ''',
          '''
          def test_basic_placeholder_and_escape():
              assert render_template('Hello, {name}!', {'name': 'Ada'}) == 'Hello, Ada!'
              assert render_template('{{literal}}', {}) == '{literal}'
          ''',
          '''
          import pytest

          def test_missing_key_raises_value_error():
              with pytest.raises(ValueError):
                  render_template('{missing}', {})

          def test_non_identifier_keys_supported():
              assert render_template('{user name}', {'user name': 'x'}) == 'x'
              assert render_template('{0}', {'0': 'zero'}) == 'zero'

          def test_no_rescan_of_output():
              assert render_template('{a}', {'a': '{b}'}) == '{b}'

          def test_nested_brace_is_not_a_key():
              with pytest.raises(ValueError):
                  render_template('{a{b}', {'a{b': 'invalid'})
              assert render_template('{{{key}}}', {'key': 'ok'}) == '{ok}'

          def test_malformed_braces_rejected():
              with pytest.raises(ValueError):
                  render_template('}', {})
              with pytest.raises(ValueError):
                  render_template('{unclosed', {})
              with pytest.raises(ValueError):
                  render_template('{}', {})
              with pytest.raises(ValueError):
                  render_template('ok', 'notadict')
          ''', difficulty="L3"),
    _task("pluralize", "string_templates",
          "修复 pluralize(count, singular, plural=None)：plural 为 None 时默认为 "
          "singular + 's'；count == 1 返回 singular，其余一切 count（0、负数、2 及以上）"
          "返回 plural。count 必须为 int（bool 不视为整数）、singular 必须为非空 str，"
          "否则 ValueError。",
          '''
          def pluralize(count, singular, plural=None):
              if plural is None:
                  plural = singular + 's'
              return singular if count <= 1 else plural
          ''',
          '''
          def pluralize(count, singular, plural=None):
              if isinstance(count, bool) or not isinstance(count, int):
                  raise ValueError('count must be an integer')
              if not isinstance(singular, str) or not singular:
                  raise ValueError('singular must be a nonempty string')
              if plural is None:
                  plural = singular + 's'
              return singular if count == 1 else plural
          ''',
          '''
          def test_basic_forms():
              assert pluralize(1, 'item') == 'item'
              assert pluralize(2, 'item') == 'items'
              assert pluralize(2, 'goose', 'geese') == 'geese'
          ''',
          '''
          import pytest

          def test_zero_and_negative_take_plural():
              assert pluralize(0, 'item') == 'items'
              assert pluralize(-3, 'coin') == 'coins'

          def test_default_plural_suffix():
              assert pluralize(0, 'box') == 'boxs'
              assert pluralize(5, 'box', 'boxes') == 'boxes'

          def test_validation():
              with pytest.raises(ValueError):
                  pluralize(1.5, 'x')
              with pytest.raises(ValueError):
                  pluralize(1, '')
          '''),
    # ── 来源组 3：v2c_queue_semantics ──────────────────────────────────
    _task("rolling_queue", "queue_semantics",
          "修复类 RollingQueue(capacity)：capacity 正整数（bool 不视为整数）否则 "
          "ValueError。push(item)：队列未满时入队并返回 None；已满时先逐出最旧元素、"
          "入队新元素并返回**被逐出的元素**。items() 返回按入队顺序的元素新列表"
          "（修改返回列表不得影响内部状态）。",
          '''
          class RollingQueue:
              def __init__(self, capacity):
                  if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
                      raise ValueError('capacity must be a positive integer')
                  self._capacity = capacity
                  self._items = []

              def push(self, item):
                  if len(self._items) >= self._capacity:
                      self._items.pop(0)
                  self._items.append(item)
                  return item

              def items(self):
                  return list(self._items)
          ''',
          '''
          class RollingQueue:
              def __init__(self, capacity):
                  if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
                      raise ValueError('capacity must be a positive integer')
                  self._capacity = capacity
                  self._items = []

              def push(self, item):
                  if len(self._items) >= self._capacity:
                      evicted = self._items.pop(0)
                      self._items.append(item)
                      return evicted
                  self._items.append(item)
                  return None

              def items(self):
                  return list(self._items)
          ''',
          '''
          def test_fill_and_order():
              queue = RollingQueue(2)
              queue.push('a')
              queue.push('b')
              assert queue.items() == ['a', 'b']

          def test_capacity_validation():
              import pytest
              with pytest.raises(ValueError):
                  RollingQueue(0)
          ''',
          '''
          import pytest

          def test_overflow_returns_evicted():
              queue = RollingQueue(2)
              queue.push('a')
              queue.push('b')
              assert queue.push('c') == 'a'
              assert queue.items() == ['b', 'c']

          def test_underfull_returns_none():
              queue = RollingQueue(3)
              assert queue.push('x') is None

          def test_items_returns_copy():
              queue = RollingQueue(2)
              queue.push('a')
              snapshot = queue.items()
              snapshot.append('z')
              assert queue.items() == ['a']
          '''),
    _task("rotate_left", "queue_semantics",
          "修复 rotate_left(items, k)：返回列表向左循环移动 k 步后的**新列表**"
          "（不修改输入）；k 可为任意非负整数，超过长度时按 k % len 取模，"
          "结果等于 0 时返回原序列副本；空列表返回 []；"
          "k 为负数或非 int（bool 不视为整数）抛 ValueError。",
          '''
          def rotate_left(items, k):
              if isinstance(k, bool) or not isinstance(k, int) or k < 0:
                  raise ValueError('k must be a nonnegative integer')
              return items[k:] + items[:k]
          ''',
          '''
          def rotate_left(items, k):
              if isinstance(k, bool) or not isinstance(k, int) or k < 0:
                  raise ValueError('k must be a nonnegative integer')
              if not items:
                  return []
              shift = k % len(items)
              return items[shift:] + items[:shift]
          ''',
          '''
          def test_small_rotation():
              assert rotate_left([1, 2, 3, 4], 1) == [2, 3, 4, 1]
              assert rotate_left([], 3) == []

          def test_input_not_mutated():
              data = [1, 2, 3]
              rotate_left(data, 1)
              assert data == [1, 2, 3]
          ''',
          '''
          import pytest

          def test_wraparound_uses_modulo():
              assert rotate_left([1, 2, 3], 5) == [3, 1, 2]
              assert rotate_left([1, 2, 3], 3) == [1, 2, 3]
              assert rotate_left([1, 2, 3], 0) == [1, 2, 3]

          def test_validation():
              with pytest.raises(ValueError):
                  rotate_left([1], -1)
              with pytest.raises(ValueError):
                  rotate_left([1], True)
          '''),
    # ── 来源组 4：v2c_coordinate_moves ─────────────────────────────────
    _task("apply_moves", "coordinate_moves",
          "修复 apply_moves(start, moves)：start 为 (x, y) 元组，moves 为字符串，"
          "每个字符依次执行：'U' 使 y+1、'D' 使 y-1、'L' 使 x-1、'R' 使 x+1；"
          "返回终点 (x, y) 元组；出现其他字符抛 ValueError；空移动串返回 start。",
          '''
          def apply_moves(start, moves):
              x, y = start
              for move in moves:
                  if move == 'U':
                      y += 1
                  elif move == 'D':
                      y -= 1
                  elif move == 'L':
                      x += 1
                  elif move == 'R':
                      x -= 1
                  else:
                      raise ValueError(f'invalid move: {move}')
              return (x, y)
          ''',
          '''
          def apply_moves(start, moves):
              x, y = start
              for move in moves:
                  if move == 'U':
                      y += 1
                  elif move == 'D':
                      y -= 1
                  elif move == 'L':
                      x -= 1
                  elif move == 'R':
                      x += 1
                  else:
                      raise ValueError(f'invalid move: {move}')
              return (x, y)
          ''',
          '''
          def test_vertical_moves():
              assert apply_moves((0, 0), 'UU') == (0, 2)
              assert apply_moves((0, 0), '') == (0, 0)

          def test_invalid_move_rejected():
              import pytest
              with pytest.raises(ValueError):
                  apply_moves((0, 0), 'UX')
          ''',
          '''
          def test_horizontal_directions():
              assert apply_moves((1, 1), 'L') == (0, 1)
              assert apply_moves((0, 0), 'RR') == (2, 0)

          def test_mixed_roundtrip():
              assert apply_moves((3, 3), 'ULDR') == (3, 3)
          '''),
    _task("step_distance", "coordinate_moves",
          "修复 step_distance(start, target)：每次移动可使 x 或 y 增减 1，"
          "返回从 start 到 target 的最少步数（曼哈顿距离 |dx|+|dy|）；"
          "两点均为 (x, y) 元组。",
          '''
          def step_distance(start, target):
              dx = abs(target[0] - start[0])
              dy = abs(target[1] - start[1])
              return max(dx, dy)
          ''',
          '''
          def step_distance(start, target):
              dx = abs(target[0] - start[0])
              dy = abs(target[1] - start[1])
              return dx + dy
          ''',
          '''
          def test_axis_aligned():
              assert step_distance((0, 0), (3, 0)) == 3
              assert step_distance((1, 1), (1, 1)) == 0
          ''',
          '''
          def test_diagonal_requires_two_axes():
              assert step_distance((0, 0), (2, 3)) == 5
              assert step_distance((-1, -2), (1, 3)) == 7
              assert step_distance((5, 0), (0, 5)) == 10
          '''),
    # ── 来源组 5：v2c_csv_splitting ────────────────────────────────────
    _task("split_quoted", "csv_splitting",
          "修复 split_quoted(line)：按逗号切分为字段列表。双引号内的逗号不是分隔符；"
          "字段内连续两个双引号 '\"\"' 表示一个字面 '\"'；引号只在字段内部起转义作用，"
          "包裹字段的引号本身不出现在结果里；空字段保留为 ''；引号未闭合抛 ValueError。"
          "从左到右一次扫描，不使用任何 CSV 库。",
          '''
          def split_quoted(line):
              return line.split(',')
          ''',
          '''
          def split_quoted(line):
              fields = []
              current = []
              in_quotes = False
              i = 0
              n = len(line)
              while i < n:
                  ch = line[i]
                  if in_quotes:
                      if ch == '"':
                          if i + 1 < n and line[i + 1] == '"':
                              current.append('"')
                              i += 2
                              continue
                          in_quotes = False
                          i += 1
                      else:
                          current.append(ch)
                          i += 1
                  elif ch == '"':
                      in_quotes = True
                      i += 1
                  elif ch == ',':
                      fields.append(''.join(current))
                      current = []
                      i += 1
                  else:
                      current.append(ch)
                      i += 1
              if in_quotes:
                  raise ValueError('unterminated quoted field')
              fields.append(''.join(current))
              return fields
          ''',
          '''
          def test_plain_fields():
              assert split_quoted('a,b,c') == ['a', 'b', 'c']
              assert split_quoted('') == ['']
          ''',
          '''
          import pytest

          def test_quoted_comma_stays_one_field():
              assert split_quoted('"a,b",c') == ['a,b', 'c']

          def test_escaped_quotes():
              assert split_quoted('"say ""hi""",x') == ['say "hi"', 'x']

          def test_empty_fields_kept():
              assert split_quoted('a,,c') == ['a', '', 'c']

          def test_unterminated_quote_rejected():
              with pytest.raises(ValueError):
                  split_quoted('"abc')

          def test_quote_at_line_end_rejected():
              with pytest.raises(ValueError):
                  split_quoted('x,"')
              with pytest.raises(ValueError):
                  split_quoted('"a""')

          def test_quoted_empty_fields():
              assert split_quoted('"",x') == ['', 'x']
              assert split_quoted('x,""') == ['x', '']
          ''', difficulty="L3"),
    _task("env_lines", "csv_splitting",
          "修复 parse_env_lines(lines)：解析 'KEY=value' 行为 dict。"
          "空行与以 '#' 开头的行跳过；非空行必须含 '='，以第一个 '=' 分隔，"
          "键值均去两端空白；键为空、重复键、无 '=' 的非空行一律 ValueError。",
          '''
          def parse_env_lines(lines):
              result = {}
              for line in lines:
                  if not line.strip():
                      continue
                  key, _, value = line.partition('=')
                  result[key.strip()] = value.strip()
              return result
          ''',
          '''
          def parse_env_lines(lines):
              result = {}
              for line in lines:
                  stripped = line.strip()
                  if not stripped or stripped.startswith('#'):
                      continue
                  if '=' not in line:
                      raise ValueError(f'missing separator: {line!r}')
                  key, value = line.split('=', 1)
                  key = key.strip()
                  if not key:
                      raise ValueError('empty key')
                  if key in result:
                      raise ValueError(f'duplicate key: {key}')
                  result[key] = value.strip()
              return result
          ''',
          '''
          def test_basic_and_blank_lines():
              assert parse_env_lines(['A=1', 'B=2']) == {'A': '1', 'B': '2'}
              assert parse_env_lines(['A=1', '', '  ', 'B=2']) == {'A': '1', 'B': '2'}
          ''',
          '''
          import pytest

          def test_comments_skipped():
              assert parse_env_lines(['# comment', 'A=1']) == {'A': '1'}

          def test_value_may_contain_equals():
              assert parse_env_lines(['TOKEN=a=b']) == {'TOKEN': 'a=b'}

          def test_duplicates_rejected():
              with pytest.raises(ValueError):
                  parse_env_lines(['A=1', 'A=2'])

          def test_malformed_lines_rejected():
              with pytest.raises(ValueError):
                  parse_env_lines(['=value'])
              with pytest.raises(ValueError):
                  parse_env_lines(['oops'])
          '''),
    # ── 来源组 6：v2c_range_stepping ───────────────────────────────────
    _task("inclusive_range", "range_stepping",
          "修复 inclusive_range(start, stop, step)：返回从 start 开始、步长 step 的"
          "升序等差整数列表，闭区间——stop 恰好落在序列上时也包含 stop；"
          "start > stop 返回空列表；step 正整数（bool 不视为整数）否则 ValueError。",
          '''
          def inclusive_range(start, stop, step):
              if isinstance(step, bool) or not isinstance(step, int) or step < 1:
                  raise ValueError('step must be a positive integer')
              return list(range(start, stop, step))
          ''',
          '''
          def inclusive_range(start, stop, step):
              if isinstance(step, bool) or not isinstance(step, int) or step < 1:
                  raise ValueError('step must be a positive integer')
              result = []
              value = start
              while value <= stop:
                  result.append(value)
                  value += step
              return result
          ''',
          '''
          def test_stop_not_on_sequence():
              assert inclusive_range(0, 10, 3) == [0, 3, 6, 9]
              assert inclusive_range(4, 3, 1) == []
          ''',
          '''
          import pytest

          def test_stop_included_when_hit():
              assert inclusive_range(2, 10, 4) == [2, 6, 10]
              assert inclusive_range(5, 5, 2) == [5]
              assert inclusive_range(1, 1, 1) == [1]

          def test_step_validation():
              with pytest.raises(ValueError):
                  inclusive_range(0, 5, 0)
              with pytest.raises(ValueError):
                  inclusive_range(0, 5, True)
          '''),
    _task("chunk_list", "range_stepping",
          "修复 chunk_list(items, size)：把列表切成连续、长度至多为 size 的块"
          "（最后一块可以更短，不得丢弃）；items 必须 list、size 正整数"
          "（bool 不视为整数），否则 ValueError；空列表返回 []；不修改输入。",
          '''
          def chunk_list(items, size):
              if (not isinstance(items, list) or isinstance(size, bool)
                      or not isinstance(size, int) or size < 1):
                  raise ValueError('items must be a list and size a positive integer')
              chunks = []
              for start in range(0, len(items) - size + 1, size):
                  chunks.append(items[start:start + size])
              return chunks
          ''',
          '''
          def chunk_list(items, size):
              if (not isinstance(items, list) or isinstance(size, bool)
                      or not isinstance(size, int) or size < 1):
                  raise ValueError('items must be a list and size a positive integer')
              return [items[start:start + size]
                      for start in range(0, len(items), size)]
          ''',
          '''
          def test_even_chunks():
              assert chunk_list([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]
              assert chunk_list([], 2) == []
          ''',
          '''
          import pytest

          def test_tail_chunk_kept():
              assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
              assert chunk_list([1, 2, 3], 5) == [[1, 2, 3]]

          def test_input_unchanged():
              data = [1, 2, 3]
              chunk_list(data, 2)
              assert data == [1, 2, 3]

          def test_validation():
              with pytest.raises(ValueError):
                  chunk_list([1], 0)
              with pytest.raises(ValueError):
                  chunk_list('abc', 1)
          '''),
    # ── 来源组 7：v2c_banking_ledger ───────────────────────────────────
    _task("bank_account", "banking_ledger",
          "修复类 BankAccount(balance=0)：初始余额必须非负数字（int/float，"
          "bool 不视为数字）否则 ValueError。deposit(amount)：amount 正数否则 "
          "ValueError，入账并返回新余额。withdraw(amount)：amount 正数否则 "
          "ValueError；amount 超过当前余额抛 ValueError 且**余额保持不变**；"
          "成功时扣款并返回新余额。balance() 返回当前余额。",
          '''
          class BankAccount:
              def __init__(self, balance=0):
                  if (isinstance(balance, bool) or not isinstance(balance, (int, float))
                          or balance < 0):
                      raise ValueError('balance must be a nonnegative number')
                  self._balance = balance

              def deposit(self, amount):
                  if amount <= 0:
                      raise ValueError('amount must be positive')
                  self._balance += amount
                  return self._balance

              def withdraw(self, amount):
                  if amount <= 0:
                      raise ValueError('amount must be positive')
                  self._balance -= amount
                  return self._balance

              def balance(self):
                  return self._balance
          ''',
          '''
          class BankAccount:
              def __init__(self, balance=0):
                  if (isinstance(balance, bool) or not isinstance(balance, (int, float))
                          or balance < 0):
                      raise ValueError('balance must be a nonnegative number')
                  self._balance = balance

              def deposit(self, amount):
                  if amount <= 0:
                      raise ValueError('amount must be positive')
                  self._balance += amount
                  return self._balance

              def withdraw(self, amount):
                  if amount <= 0:
                      raise ValueError('amount must be positive')
                  if amount > self._balance:
                      raise ValueError('insufficient funds')
                  self._balance -= amount
                  return self._balance

              def balance(self):
                  return self._balance
          ''',
          '''
          def test_deposit_and_withdraw():
              account = BankAccount(50)
              assert account.deposit(10) == 60
              assert account.withdraw(20) == 40
              assert account.balance() == 40

          def test_amount_validation():
              import pytest
              account = BankAccount(10)
              with pytest.raises(ValueError):
                  account.deposit(0)
              with pytest.raises(ValueError):
                  account.withdraw(-5)
          ''',
          '''
          import pytest

          def test_overdraft_rejected_and_unchanged():
              account = BankAccount(100)
              with pytest.raises(ValueError):
                  account.withdraw(200)
              assert account.balance() == 100

          def test_exact_balance_withdrawable():
              account = BankAccount(100)
              assert account.withdraw(100) == 0
              with pytest.raises(ValueError):
                  account.withdraw(1)

          def test_initial_balance_validation():
              with pytest.raises(ValueError):
                  BankAccount(-1)

          def test_default_balance_is_zero():
              assert BankAccount().balance() == 0

          def test_amount_one_boundaries():
              account = BankAccount()
              assert account.deposit(1) == 1
              assert account.withdraw(1) == 0
          '''),
    _task("transfer_funds", "banking_ledger",
          "修复 transfer(source, target, amount)：source 与 target 为 "
          "(名称, 余额) 二元组；amount 正数（int/float，bool 不视为数字）否则 "
          "ValueError；amount 超过 source 余额抛 ValueError；"
          "返回 (source 新二元组, target 新二元组)，双方余额一减一增、名称不变；"
          "不得修改输入结构。",
          '''
          def transfer(source, target, amount):
              if amount <= 0:
                  raise ValueError('amount must be positive')
              if amount > source[1]:
                  raise ValueError('insufficient funds')
              return ((source[0], source[1] - amount), target)
          ''',
          '''
          def transfer(source, target, amount):
              if (isinstance(amount, bool) or not isinstance(amount, (int, float))
                      or amount <= 0):
                  raise ValueError('amount must be a positive number')
              if amount > source[1]:
                  raise ValueError('insufficient funds')
              return ((source[0], source[1] - amount),
                      (target[0], target[1] + amount))
          ''',
          '''
          def test_source_side_and_validation():
              result = transfer(('a', 100), ('b', 0), 40)
              assert result[0] == ('a', 60)

          def test_insufficient_funds():
              import pytest
              with pytest.raises(ValueError):
                  transfer(('a', 10), ('b', 0), 11)
              with pytest.raises(ValueError):
                  transfer(('a', 10), ('b', 0), 0)
          ''',
          '''
          import pytest

          def test_target_receives_funds():
              result = transfer(('a', 100), ('b', 5), 40)
              assert result == (('a', 60), ('b', 45))

          def test_exact_amount_allowed():
              assert transfer(('a', 3), ('b', 0), 3) == (('a', 0), ('b', 3))

          def test_amount_type_validation():
              with pytest.raises(ValueError):
                  transfer(('a', 10), ('b', 0), True)
          '''),
    # ── 来源组 8：v2c_ordinal_indexing ─────────────────────────────────
    _task("to_ordinal", "ordinal_indexing",
          "修复 to_ordinal(n)：正整数（bool 不视为整数）n < 1 或非 int 抛 "
          "ValueError；返回 '<n><后缀>'：n % 100 为 11/12/13 时后缀 'th'，"
          "否则按个位 1→'st'、2→'nd'、3→'rd'、其余→'th'。",
          '''
          def to_ordinal(n):
              if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                  raise ValueError('n must be a positive integer')
              suffixes = {1: 'st', 2: 'nd', 3: 'rd'}
              return f'{n}{suffixes.get(n % 10, "th")}'
          ''',
          '''
          def to_ordinal(n):
              if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                  raise ValueError('n must be a positive integer')
              if n % 100 in (11, 12, 13):
                  suffix = 'th'
              else:
                  suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
              return f'{n}{suffix}'
          ''',
          '''
          def test_units():
              assert to_ordinal(1) == '1st'
              assert to_ordinal(2) == '2nd'
              assert to_ordinal(7) == '7th'
          ''',
          '''
          import pytest

          def test_teens_use_th():
              assert to_ordinal(11) == '11th'
              assert to_ordinal(12) == '12th'
              assert to_ordinal(13) == '13th'
              assert to_ordinal(112) == '112th'
              assert to_ordinal(111) == '111th'

          def test_tens_revert():
              assert to_ordinal(21) == '21st'
              assert to_ordinal(102) == '102nd'
              assert to_ordinal(43) == '43rd'

          def test_validation():
              with pytest.raises(ValueError):
                  to_ordinal(0)
              with pytest.raises(ValueError):
                  to_ordinal(2.0)
          '''),
    _task("weekday_name", "ordinal_indexing",
          "修复 weekday_name(index)：0→'Monday'、1→'Tuesday'、2→'Wednesday'、"
          "3→'Thursday'、4→'Friday'、5→'Saturday'、6→'Sunday'；"
          "index 不做取模，非 0..6 的 int（bool 不视为整数）一律 ValueError。",
          '''
          _DAYS = ['Sunday', 'Monday', 'Tuesday', 'Wednesday',
                   'Thursday', 'Friday', 'Saturday']

          def weekday_name(index):
              if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index <= 6:
                  raise ValueError('index must be in 0..6')
              return _DAYS[index]
          ''',
          '''
          _DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                 'Friday', 'Saturday', 'Sunday']

          def weekday_name(index):
              if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index <= 6:
                  raise ValueError('index must be in 0..6')
              return _DAYS[index]
          ''',
          '''
          import pytest

          def test_returns_string():
              assert isinstance(weekday_name(3), str)

          def test_validation():
              with pytest.raises(ValueError):
                  weekday_name(7)
          ''',
          '''
          import pytest

          def test_full_week_order():
              assert weekday_name(0) == 'Monday'
              assert weekday_name(1) == 'Tuesday'
              assert weekday_name(2) == 'Wednesday'
              assert weekday_name(3) == 'Thursday'
              assert weekday_name(4) == 'Friday'
              assert weekday_name(5) == 'Saturday'
              assert weekday_name(6) == 'Sunday'

          def test_validation_edges():
              with pytest.raises(ValueError):
                  weekday_name(-1)
              with pytest.raises(ValueError):
                  weekday_name(True)
          '''),
    # ── 来源组 9：v2c_inventory_diff ───────────────────────────────────
    _task("diff_counts", "inventory_diff",
          "修复 diff_counts(before, after)：两 dict（值为 int），返回"
          "{key: after - before}，只包含差值非 0 的键；键可只在一边（缺失按 0 计）；"
          "不修改输入；任一输入非 dict 抛 ValueError。",
          '''
          def diff_counts(before, after):
              if not isinstance(before, dict) or not isinstance(after, dict):
                  raise ValueError('before and after must be dicts')
              keys = set(before) | set(after)
              return {key: after.get(key, 0) - before.get(key, 0) for key in keys}
          ''',
          '''
          def diff_counts(before, after):
              if not isinstance(before, dict) or not isinstance(after, dict):
                  raise ValueError('before and after must be dicts')
              keys = set(before) | set(after)
              changes = {key: after.get(key, 0) - before.get(key, 0) for key in keys}
              return {key: value for key, value in changes.items() if value != 0}
          ''',
          '''
          def test_changed_and_moved_keys():
              assert diff_counts({'a': 1}, {'a': 3}) == {'a': 2}
              assert diff_counts({'a': 1}, {'b': 2}) == {'a': -1, 'b': 2}
          ''',
          '''
          import pytest

          def test_zero_diff_keys_excluded():
              assert diff_counts({'a': 2}, {'a': 2}) == {}
              assert diff_counts({'a': 1, 'b': 2, 'c': 5}, {'a': 1, 'b': 4, 'd': 5}) == \
                  {'b': 2, 'c': -5, 'd': 5}

          def test_inputs_unchanged_and_validation():
              before = {'a': 1}
              after = {'a': 2}
              diff_counts(before, after)
              assert before == {'a': 1} and after == {'a': 2}
              with pytest.raises(ValueError):
                  diff_counts([], {})
          '''),
    _task("restock_needs", "inventory_diff",
          "修复 restock_needs(stock, required)：对 required 中每个键，"
          "需求 = max(0, required[key] - stock.get(key, 0))；返回包含 required 全部键"
          "的 dict（需求为 0 也保留，不得出现负数）；stock 缺该键按 0 库存计；"
          "不修改输入；任一输入非 dict 抛 ValueError。",
          '''
          def restock_needs(stock, required):
              if not isinstance(stock, dict) or not isinstance(required, dict):
                  raise ValueError('stock and required must be dicts')
              return {key: required[key] - stock.get(key, 0) for key in required}
          ''',
          '''
          def restock_needs(stock, required):
              if not isinstance(stock, dict) or not isinstance(required, dict):
                  raise ValueError('stock and required must be dicts')
              return {key: max(0, required[key] - stock.get(key, 0))
                      for key in required}
          ''',
          '''
          def test_shortage_and_missing_stock():
              assert restock_needs({'a': 1}, {'a': 3}) == {'a': 2}
              assert restock_needs({}, {'b': 2}) == {'b': 2}
          ''',
          '''
          import pytest

          def test_no_negative_needs():
              assert restock_needs({'a': 5}, {'a': 3}) == {'a': 0}
              assert restock_needs({'x': 9}, {'x': 9, 'y': 0}) == {'x': 0, 'y': 0}

          def test_inputs_unchanged_and_validation():
              stock = {'a': 1}
              required = {'a': 5}
              restock_needs(stock, required)
              assert stock == {'a': 1} and required == {'a': 5}
              with pytest.raises(ValueError):
                  restock_needs('no', {})
          '''),
    # ── 来源组 10：v2c_slug_generation ─────────────────────────────────
    _task("slugify", "slug_generation",
          "修复 slugify(text)：小写化后，把每个非 [a-z0-9-] 字符替换为 '-'，"
          "连续多个 '-' 合并为一个，并去掉首尾 '-'（结果可为空串）；"
          "text 非 str 抛 ValueError。",
          '''
          def slugify(text):
              if not isinstance(text, str):
                  raise ValueError('text must be a string')
              out = []
              for ch in text.lower():
                  if ch.isascii() and (ch.isalnum() or ch == '-'):
                      out.append(ch)
                  else:
                      out.append('-')
              return ''.join(out)
          ''',
          '''
          def slugify(text):
              if not isinstance(text, str):
                  raise ValueError('text must be a string')
              out = []
              for ch in text.lower():
                  if ch.isascii() and (ch.isalnum() or ch == '-'):
                      out.append(ch)
                  else:
                      out.append('-')
              collapsed = []
              for ch in out:
                  if ch == '-' and collapsed and collapsed[-1] == '-':
                      continue
                  collapsed.append(ch)
              return ''.join(collapsed).strip('-')
          ''',
          '''
          def test_basic_slugs():
              assert slugify('Hello World') == 'hello-world'
              assert slugify('abc') == 'abc'
              assert slugify('') == ''
          ''',
          '''
          import pytest

          def test_repeated_separators_collapsed():
              assert slugify('a  b') == 'a-b'
              assert slugify('Hello, World!') == 'hello-world'

          def test_leading_trailing_stripped():
              assert slugify('--hi--') == 'hi'
              assert slugify('!!!') == ''

          def test_non_ascii_replaced():
              assert slugify('café') == 'caf'

          def test_validation():
              with pytest.raises(ValueError):
                  slugify(123)
          '''),
    _task("truncate_words", "slug_generation",
          "修复 truncate_words(text, limit)：按任意空白（str.split()）切词，保留前 "
          "limit 个词并以单个空格连接；limit 为 0 返回 ''；limit 不小于词数时**原样"
          "返回 text**（保留原始空白）；text 非 str、limit 非 int（bool 不视为整数）"
          "或 limit 为负抛 ValueError。",
          '''
          def truncate_words(text, limit):
              if (not isinstance(text, str) or isinstance(limit, bool)
                      or not isinstance(limit, int) or limit < 0):
                  raise ValueError('text must be str and limit a nonnegative integer')
              words = text.split()
              return ' '.join(words[:limit + 1])
          ''',
          '''
          def truncate_words(text, limit):
              if (not isinstance(text, str) or isinstance(limit, bool)
                      or not isinstance(limit, int) or limit < 0):
                  raise ValueError('text must be str and limit a nonnegative integer')
              words = text.split()
              if limit >= len(words):
                  return text
              return ' '.join(words[:limit])
          ''',
          '''
          def test_no_truncation_needed():
              assert truncate_words('one two three', 10) == 'one two three'
              assert truncate_words('a b', 2) == 'a b'

          def test_validation():
              import pytest
              with pytest.raises(ValueError):
                  truncate_words('a b', -1)
          ''',
          '''
          def test_truncates_to_limit():
              assert truncate_words('a b c', 2) == 'a b'
              assert truncate_words('a b c', 0) == ''
              assert truncate_words('a   b', 1) == 'a'

          def test_original_whitespace_preserved_when_within_limit():
              assert truncate_words('a   b', 5) == 'a   b'
              assert truncate_words('  lead', 1) == '  lead'
          '''),
]
