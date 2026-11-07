"""Twenty new v2 repair tasks in ten fresh source groups (batch 1 of the scale-up).

Distinct from the pilot's twelve families: encoding/escaping, bracket logic,
retry/backoff contracts, cache invalidation, grid/matrix handling, fallback
lookup, money splitting, word boundaries, parent-array trees and state flags.
Every buggy version passes its public tests but fails hidden private tests, so
an unchanged candidate is graded ``partial`` instead of accidentally passing.
"""
from __future__ import annotations

from textwrap import dedent


def _task(name, group, instruction, buggy, fixed, public_tests, private_tests,
          difficulty="L2"):
    module = f"v2_{name}"
    return {
        "task_id": f"v2_{name}_001", "source_group": f"v2_{group}",
        "split": "train", "bug_family": group, "difficulty": difficulty,
        "instruction": instruction, "module": f"{module}.py",
        "buggy": dedent(buggy).strip() + "\n",
        "fixed": dedent(fixed).strip() + "\n",
        "public_test": f"from {module} import *\n\n" + dedent(public_tests).strip() + "\n",
        "private_tests": {"test_regression.py":
                          f"import pytest\nfrom {module} import *\n\n"
                          + dedent(private_tests).strip() + "\n"},
    }


V2_SPECS = [
    # ── 来源组 1：v2_encoding_escape ────────────────────────────────────
    _task("percent_encode", "encoding_escape",
          "修复 percent_encode(text, safe='')：对 text 的 UTF-8 字节做百分号编码。"
          "永不编码的未保留字符为 A-Z、a-z、0-9、'-'、'_'、'.'、'~' 以及 safe 中的字符；"
          "其余每个字节输出 '%' 加两个大写十六进制位。非 ASCII 字符先按 UTF-8 拆成多字节再逐字节编码"
          "（如 'é' 编码为 '%C3%A9'）。safe 只起免编码作用。",
          '''
          UNRESERVED = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                           "abcdefghijklmnopqrstuvwxyz"
                           "0123456789-_.")

          def percent_encode(text, safe=''):
              keep = UNRESERVED | set(safe)
              out = []
              for ch in text:
                  if ch in keep:
                      out.append(ch)
                  else:
                      out.append('%{:02X}'.format(ord(ch)))
              return ''.join(out)
          ''',
          '''
          UNRESERVED = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                           "abcdefghijklmnopqrstuvwxyz"
                           "0123456789-_.~")

          def percent_encode(text, safe=''):
              keep = UNRESERVED | set(safe)
              out = []
              for byte in text.encode('utf-8'):
                  ch = chr(byte)
                  if ch in keep:
                      out.append(ch)
                  else:
                      out.append('%{:02X}'.format(byte))
              return ''.join(out)
          ''',
          '''
          def test_space_and_slash():
              assert percent_encode('hello world') == 'hello%20world'
              assert percent_encode('a/b') == 'a%2Fb'
          ''',
          '''
          def test_unreserved_tilde_kept():
              assert percent_encode('a~b') == 'a~b'
          def test_utf8_multibyte():
              assert percent_encode('café') == 'caf%C3%A9'
              assert percent_encode('中文') == '%E4%B8%AD%E6%96%87'
          def test_safe_argument():
              assert percent_encode('a b', safe=' ') == 'a b'
          '''),
    _task("unescape_entities", "encoding_escape",
          "修复 unescape_entities(text)：从左到右单次扫描解码实体，解码结果不再参与后续解码。"
          "支持命名实体 '&amp;'→'&'、'&lt;'→'<'、'&gt;'→'>'、'&quot;'→'\"'、'&apos;'→\"'\"；"
          "十进制数字实体 '&#N;'（N 为纯数字且值不超过 1114111）解码为对应字符；"
          "未知命名实体、越界数字实体、未以分号结束的 '&' 片段一律保持原文。"
          "关键语义：'&amp;lt;' 解码为 '&lt;'（先得到字面 '&'，后跟原文 'lt;'），不得二次解码。",
          '''
          NAMED = {'amp': '&', 'lt': '<', 'gt': '>', 'quot': '"', 'apos': "'"}

          def unescape_entities(text):
              for entity, char in NAMED.items():
                  text = text.replace('&' + entity + ';', char)
              import re
              return re.sub(r'&#(\\d+);', lambda m: chr(int(m.group(1))), text)
          ''',
          '''
          NAMED = {'amp': '&', 'lt': '<', 'gt': '>', 'quot': '"', 'apos': "'"}

          def unescape_entities(text):
              out = []
              i = 0
              n = len(text)
              while i < n:
                  if text[i] == '&':
                      end = text.find(';', i + 1)
                      if end != -1:
                          body = text[i + 1:end]
                          if body in NAMED:
                              out.append(NAMED[body])
                              i = end + 1
                              continue
                          if (body.startswith('#') and body[1:].isdigit()
                                  and int(body[1:]) <= 0x10FFFF):
                              out.append(chr(int(body[1:])))
                              i = end + 1
                              continue
                  out.append(text[i])
                  i += 1
              return ''.join(out)
          ''',
          '''
          def test_named_roundtrip():
              assert unescape_entities('a&amp;b') == 'a&b'
              assert unescape_entities('&lt;em&gt;') == '<em>'
          ''',
          '''
          def test_no_double_decoding():
              assert unescape_entities('&amp;lt;') == '&lt;'
          def test_numeric_and_unknown():
              assert unescape_entities('&#65;') == 'A'
              assert unescape_entities('&nbsp;') == '&nbsp;'
              assert unescape_entities('&#1114112;') == '&#1114112;'
          def test_ampersand_alone():
              assert unescape_entities('a & b') == 'a & b'
          '''),
    # ── 来源组 2：v2_bracket_logic ─────────────────────────────────────
    _task("is_balanced", "bracket_logic",
          "修复 is_balanced(text)：判断三种括号 ()、[]、{} 是否全部正确匹配。"
          "正确匹配要求每个右括号与最近的未匹配同类型左括号对应，类型间不得交叉"
          "（如 '([)]' 为不匹配）；非括号字符忽略；空字符串匹配。只能使用一遍扫描与栈语义。",
          '''
          OPEN = {'(': 0, '[': 0, '{': 0}
          CLOSE = {')': '(', ']': '[', '}': '{'}

          def is_balanced(text):
              counts = {'(': 0, '[': 0, '{': 0}
              for ch in text:
                  if ch in OPEN:
                      counts[ch] += 1
                  elif ch in CLOSE:
                      counts[CLOSE[ch]] -= 1
                      if counts[CLOSE[ch]] < 0:
                          return False
              return all(value == 0 for value in counts.values())
          ''',
          '''
          PAIRS = {')': '(', ']': '[', '}': '{'}

          def is_balanced(text):
              stack = []
              for ch in text:
                  if ch in '([{':
                      stack.append(ch)
                  elif ch in PAIRS:
                      if not stack or stack[-1] != PAIRS[ch]:
                          return False
                      stack.pop()
              return not stack
          ''',
          '''
          def test_simple_nesting():
              assert is_balanced('a(b[c]d)e') is True
              assert is_balanced('(]') is False
          ''',
          '''
          def test_interleaving_rejected():
              assert is_balanced('([)]') is False
              assert is_balanced('{[}]') is False
          def test_deep_and_empty():
              assert is_balanced('{[()]}') is True
              assert is_balanced('') is True
              assert is_balanced('((') is False
          '''),
    _task("remove_pairs", "bracket_logic",
          "修复 remove_pairs(text)：反复删除紧邻的空括号对（直接相邻且互相匹配的 ()、[]、{}），"
          "直到字符串不再变化，返回最终结果；非括号字符保持不动。"
          "必须迭代到不动点：如 '(())' 先删内层得到 '()'，还需继续删除得到 ''。",
          '''
          def remove_pairs(text):
              for pair in ('()', '[]', '{}'):
                  text = text.replace(pair, '')
              return text
          ''',
          '''
          def remove_pairs(text):
              changed = True
              while changed:
                  changed = False
                  for pair in ('()', '[]', '{}'):
                      if pair in text:
                          text = text.replace(pair, '')
                          changed = True
              return text
          ''',
          '''
          def test_single_level():
              assert remove_pairs('a()b') == 'ab'
              assert remove_pairs('([)]') == '([)]'
          ''',
          '''
          def test_fixpoint_required():
              assert remove_pairs('(())') == ''
              assert remove_pairs('{{}}') == ''
              assert remove_pairs('{[()]}') == ''
          def test_partial_removal():
              assert remove_pairs('x(y)z') == 'x(y)z'
              assert remove_pairs('(()') == '('
          '''),
    # ── 来源组 3：v2_retry_logic ───────────────────────────────────────
    _task("backoff_delay", "retry_logic",
          "修复 backoff_delay(attempt, base=1.0, factor=2.0, cap=30.0)："
          "返回第 attempt 次重试（从 0 开始计数）的退避秒数 min(cap, base * factor ** attempt)。"
          "attempt 必须为非负整数（bool 不视为整数），base、factor、cap 必须为正数；"
          "任一参数不合法立即抛 ValueError，不得先做其他计算。",
          '''
          def backoff_delay(attempt, base=1.0, factor=2.0, cap=30.0):
              if base <= 0 or factor <= 0 or cap <= 0:
                  raise ValueError('base/factor/cap must be positive')
              return min(cap, base * factor ** attempt)
          ''',
          '''
          def backoff_delay(attempt, base=1.0, factor=2.0, cap=30.0):
              if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
                  raise ValueError('attempt must be a nonnegative integer')
              if base <= 0 or factor <= 0 or cap <= 0:
                  raise ValueError('base/factor/cap must be positive')
              return min(cap, base * factor ** attempt)
          ''',
          '''
          def test_basic_growth():
              assert backoff_delay(0) == 1.0
              assert backoff_delay(3) == 8.0
              assert backoff_delay(10, cap=30.0) == 30.0
          ''',
          '''
          import pytest

          def test_attempt_validation():
              with pytest.raises(ValueError):
                  backoff_delay(-1)
              with pytest.raises(ValueError):
                  backoff_delay(1.5)
              with pytest.raises(ValueError):
                  backoff_delay(True)
          def test_parameter_validation():
              with pytest.raises(ValueError):
                  backoff_delay(0, base=0)
              with pytest.raises(ValueError):
                  backoff_delay(0, factor=-1)
          '''),
    _task("retry_decision", "retry_logic",
          "修复 should_retry(failed_attempts, elapsed_s, next_delay_s, deadline_s, max_retries=3)："
          "当且仅当 failed_attempts < max_retries 且 elapsed_s + next_delay_s <= deadline_s 时返回 True。"
          "边界语义：重试次数恰好用尽（等于 max_retries）不再重试；"
          "下一次重试完成时刻恰好等于 deadline（elapsed+delay == deadline）仍允许重试。"
          "failed_attempts、max_retries 必须为非负整数（bool 不视为整数），"
          "elapsed_s、next_delay_s、deadline_s 必须为非负数字，否则 ValueError。",
          '''
          def should_retry(failed_attempts, elapsed_s, next_delay_s, deadline_s,
                           max_retries=3):
              if failed_attempts < 0 or max_retries < 0:
                  raise ValueError('attempt counts must be nonnegative')
              if failed_attempts <= max_retries and elapsed_s + next_delay_s < deadline_s:
                  return True
              return False
          ''',
          '''
          def should_retry(failed_attempts, elapsed_s, next_delay_s, deadline_s,
                           max_retries=3):
              if (isinstance(failed_attempts, bool) or not isinstance(failed_attempts, int)
                      or failed_attempts < 0
                      or isinstance(max_retries, bool) or not isinstance(max_retries, int)
                      or max_retries < 0):
                  raise ValueError('attempt counts must be nonnegative integers')
              for value in (elapsed_s, next_delay_s, deadline_s):
                  if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
                      raise ValueError('times must be nonnegative numbers')
              return (failed_attempts < max_retries
                      and elapsed_s + next_delay_s <= deadline_s)
          ''',
          '''
          def test_clear_cases():
              assert should_retry(0, 0, 1, 10) is True
              assert should_retry(2, 95, 10, 100) is False
          ''',
          '''
          import pytest

          def test_exhausted_budget_boundary():
              assert should_retry(3, 10, 5, 100) is False
              assert should_retry(4, 0, 0, 10) is False
          def test_deadline_boundary_inclusive():
              assert should_retry(2, 95, 5, 100) is True
          def test_validation():
              with pytest.raises(ValueError):
                  should_retry(-1, 0, 0, 10)
              with pytest.raises(ValueError):
                  should_retry(1.5, 0, 0, 10)
              with pytest.raises(ValueError):
                  should_retry(0, -1, 0, 10)
              with pytest.raises(ValueError):
                  should_retry(0, 0, 0, 10, max_retries=True)
          '''),
    # ── 来源组 4：v2_cache_invalidation ────────────────────────────────
    _task("lru_cache", "cache_invalidation",
          "修复类 LRUCache(capacity)：capacity 必须为正整数（bool 不视为整数）否则 ValueError。"
          "put(key, value) 写入并把该键标记为最新使用；若插入新键时容量已满，先逐出最久未使用的键。"
          "get(key) 命中返回对应值并把该键标记为最新使用，未命中返回 None（不抛异常）。"
          "对已存在键重复 put 只覆盖并刷新使用标记，不增加条目数。",
          '''
          class LRUCache:
              def __init__(self, capacity):
                  if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
                      raise ValueError('capacity must be a positive integer')
                  self._capacity = capacity
                  self._data = {}

              def get(self, key):
                  if key not in self._data:
                      return None
                  value = self._data.pop(key)
                  self._data[key] = value
                  return value

              def put(self, key, value):
                  if key not in self._data and len(self._data) >= self._capacity:
                      evict = list(self._data)[-1]
                      del self._data[evict]
                  self._data.pop(key, None)
                  self._data[key] = value
          ''',
          '''
          class LRUCache:
              def __init__(self, capacity):
                  if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
                      raise ValueError('capacity must be a positive integer')
                  self._capacity = capacity
                  self._data = {}

              def get(self, key):
                  if key not in self._data:
                      return None
                  value = self._data.pop(key)
                  self._data[key] = value
                  return value

              def put(self, key, value):
                  if key not in self._data and len(self._data) >= self._capacity:
                      oldest = next(iter(self._data))
                      del self._data[oldest]
                  self._data.pop(key, None)
                  self._data[key] = value
          ''',
          '''
          def test_basic_put_get():
              cache = LRUCache(3)
              cache.put('a', 1)
              cache.put('b', 2)
              cache.put('c', 3)
              assert cache.get('a') == 1
              assert cache.get('missing') is None

          def test_capacity_validation():
              import pytest
              with pytest.raises(ValueError):
                  LRUCache(0)
          ''',
          '''
          import pytest

          def test_evicts_least_recently_used():
              cache = LRUCache(2)
              cache.put('a', 1)
              cache.put('b', 2)
              cache.get('a')
              cache.put('c', 3)
              assert cache.get('b') is None
              assert cache.get('a') == 1
              assert cache.get('c') == 3

          def test_overwrite_refreshes_without_growth():
              cache = LRUCache(1)
              cache.put('x', 1)
              cache.put('x', 2)
              assert cache.get('x') == 2
              cache.put('y', 3)
              assert cache.get('y') == 3

          def test_put_updates_recency():
              cache = LRUCache(2)
              cache.put('a', 1)
              cache.put('b', 2)
              cache.put('a', 10)
              cache.put('c', 3)
              assert cache.get('b') is None
              assert cache.get('a') == 10
          '''),
    _task("ttl_store", "cache_invalidation",
          "修复类 TTLStore(ttl)：ttl 必须为正数否则 ValueError。"
          "put(key, value, now) 写入或覆盖条目并刷新该键的写入时刻。"
          "get(key, now)：键不存在返回 None；键存在且 now - 写入时刻 < ttl 时返回值；"
          "now - 写入时刻 >= ttl（恰好等于也算过期）返回 None。"
          "now 为非负数字（秒），不做额外类型校验。",
          '''
          class TTLStore:
              def __init__(self, ttl):
                  if ttl <= 0:
                      raise ValueError('ttl must be positive')
                  self._ttl = ttl
                  self._entries = {}

              def put(self, key, value, now):
                  if key in self._entries:
                      self._entries[key] = (value, self._entries[key][1])
                  else:
                      self._entries[key] = (value, now)

              def get(self, key, now):
                  entry = self._entries.get(key)
                  if entry is None:
                      return None
                  value, stamp = entry
                  if now - stamp > self._ttl:
                      return None
                  return value
          ''',
          '''
          class TTLStore:
              def __init__(self, ttl):
                  if ttl <= 0:
                      raise ValueError('ttl must be positive')
                  self._ttl = ttl
                  self._entries = {}

              def put(self, key, value, now):
                  self._entries[key] = (value, now)

              def get(self, key, now):
                  entry = self._entries.get(key)
                  if entry is None:
                      return None
                  value, stamp = entry
                  if now - stamp >= self._ttl:
                      return None
                  return value
          ''',
          '''
          def test_fresh_and_missing():
              store = TTLStore(10)
              store.put('k', 'v', 0)
              assert store.get('k', 9) == 'v'
              assert store.get('missing', 0) is None

          def test_ttl_validation():
              import pytest
              with pytest.raises(ValueError):
                  TTLStore(0)
          ''',
          '''
          import pytest

          def test_expires_at_exact_boundary():
              store = TTLStore(10)
              store.put('k', 'v', 0)
              assert store.get('k', 10) is None
              assert store.get('k', 11) is None

          def test_overwrite_refreshes_timestamp():
              store = TTLStore(10)
              store.put('k', 'v1', 0)
              store.put('k', 'v2', 5)
              assert store.get('k', 14) == 'v2'
              assert store.get('k', 15) is None
          '''),
    # ── 来源组 5：v2_grid_matrix ───────────────────────────────────────
    _task("grid_neighbors", "grid_matrix",
          "修复 in_bounds_neighbors(row, col, rows, cols)：返回 4 邻域中位于网格内的坐标列表，"
          "顺序固定为上 (row-1,col)、下 (row+1,col)、左 (row,col-1)、右 (row,col+1)，"
          "每个元素为 (r, c) 元组；网格内指 0 <= r < rows 且 0 <= c < cols（边界值不可取等）。"
          "rows、cols 必须为正整数（bool 不视为整数）否则 ValueError。",
          '''
          def in_bounds_neighbors(row, col, rows, cols):
              if rows < 1 or cols < 1:
                  raise ValueError('rows/cols must be positive')
              candidates = [(row - 1, col), (row + 1, col),
                            (row, col - 1), (row, col + 1)]
              return [(r, c) for r, c in candidates
                      if 0 <= r <= rows and 0 <= c <= cols]
          ''',
          '''
          def in_bounds_neighbors(row, col, rows, cols):
              if (isinstance(rows, bool) or not isinstance(rows, int) or rows < 1
                      or isinstance(cols, bool) or not isinstance(cols, int) or cols < 1):
                  raise ValueError('rows/cols must be positive integers')
              candidates = [(row - 1, col), (row + 1, col),
                            (row, col - 1), (row, col + 1)]
              return [(r, c) for r, c in candidates
                      if 0 <= r < rows and 0 <= c < cols]
          ''',
          '''
          def test_interior_order():
              assert in_bounds_neighbors(1, 1, 3, 3) == [(0, 1), (2, 1), (1, 0), (1, 2)]

          def test_corner():
              assert in_bounds_neighbors(0, 0, 2, 2) == [(1, 0), (0, 1)]
          ''',
          '''
          import pytest

          def test_last_cell_excludes_out_of_bounds():
              assert in_bounds_neighbors(2, 2, 3, 3) == [(1, 2), (2, 1)]
              assert in_bounds_neighbors(0, 1, 1, 2) == [(0, 0)]

          def test_validation():
              with pytest.raises(ValueError):
                  in_bounds_neighbors(0, 0, 0, 3)
              with pytest.raises(ValueError):
                  in_bounds_neighbors(0, 0, 3, True)
          '''),
    _task("transpose_rect", "grid_matrix",
          "修复 transpose(matrix)：matrix 为非空矩形二维列表（至少 1 行 1 列，且各行长度一致），"
          "返回其转置（n 列 m 行 → m 列 n 行，新列表，不修改原矩阵）；"
          "matrix 为 []、[[]] 或各行长度不一致时抛 ValueError。",
          '''
          def transpose(matrix):
              if not matrix or not matrix[0]:
                  raise ValueError('matrix must be nonempty and rectangular')
              size = len(matrix)
              return [[matrix[c][r] for c in range(size)] for r in range(size)]
          ''',
          '''
          def transpose(matrix):
              if (not matrix or not matrix[0]
                      or any(len(row) != len(matrix[0]) for row in matrix)):
                  raise ValueError('matrix must be nonempty and rectangular')
              return [list(column) for column in zip(*matrix)]
          ''',
          '''
          def test_square():
              original = [[1, 2], [3, 4]]
              assert transpose(original) == [[1, 3], [2, 4]]
              assert original == [[1, 2], [3, 4]]

          def test_single_element():
              assert transpose([[7]]) == [[7]]
          ''',
          '''
          import pytest

          def test_rectangular():
              assert transpose([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]
              assert transpose([[1, 2], [3, 4], [5, 6]]) == [[1, 3, 5], [2, 4, 6]]

          def test_invalid_shapes():
              with pytest.raises(ValueError):
                  transpose([])
              with pytest.raises(ValueError):
                  transpose([[]])
              with pytest.raises(ValueError):
                  transpose([[1, 2], [3]])
          '''),
    # ── 来源组 6：v2_fallback_lookup ───────────────────────────────────
    _task("config_get", "fallback_lookup",
          "修复 config_get(config, key_path, default=None)：按 '.' 分隔的路径逐层取嵌套字典的值。"
          "任何一层键缺失、或中间值不是字典而无法继续下钻时，返回 default（不抛异常）。"
          "config 不是字典、key_path 为空串或含有空段（'a..b'、'.a'、'a.'）时抛 ValueError。",
          '''
          def config_get(config, key_path, default=None):
              if not isinstance(config, dict):
                  raise ValueError('config must be a dict')
              current = config
              for part in key_path.split('.'):
                  current = current[part]
              return current
          ''',
          '''
          def config_get(config, key_path, default=None):
              if not isinstance(config, dict):
                  raise ValueError('config must be a dict')
              parts = key_path.split('.')
              if not key_path or any(part == '' for part in parts):
                  raise ValueError('key_path must be nonempty dot-separated segments')
              current = config
              for part in parts:
                  if not isinstance(current, dict) or part not in current:
                      return default
                  current = current[part]
              return current
          ''',
          '''
          def test_nested_hit():
              assert config_get({'a': {'b': 1}}, 'a.b') == 1
              assert config_get({'a': {'b': {'c': 2}}}, 'a.b.c') == 2
          ''',
          '''
          import pytest

          def test_missing_returns_default():
              assert config_get({}, 'a') is None
              assert config_get({'a': 1}, 'a.b', default=0) == 0
              assert config_get({'a': {'b': 1}}, 'x.y') is None

          def test_path_validation():
              with pytest.raises(ValueError):
                  config_get({'a': 1}, 'a..b')
              with pytest.raises(ValueError):
                  config_get({'a': 1}, '')
              with pytest.raises(ValueError):
                  config_get(['not', 'dict'], 'a')
          '''),
    _task("first_present", "fallback_lookup",
          "修复 first_present(mapping, keys, default=None)：按 keys 顺序返回第一个"
          "「存在于 mapping 且值不为 None」的键的值。"
          "值为假值（0、''、False、[]）视为存在并直接返回；键不存在或值为 None 则跳过；"
          "全部跳过返回 default。mapping 不是字典或 keys 不是列表（tuple 等也不接受）时抛 ValueError。",
          '''
          def first_present(mapping, keys, default=None):
              if not isinstance(mapping, dict) or not isinstance(keys, list):
                  raise ValueError('mapping must be dict and keys must be list')
              for key in keys:
                  value = mapping.get(key)
                  if value:
                      return value
              return default
          ''',
          '''
          def first_present(mapping, keys, default=None):
              if not isinstance(mapping, dict) or not isinstance(keys, list):
                  raise ValueError('mapping must be dict and keys must be list')
              for key in keys:
                  if key in mapping and mapping[key] is not None:
                      return mapping[key]
              return default
          ''',
          '''
          def test_first_hit_and_none_skip():
              assert first_present({'a': 1, 'b': 2}, ['a', 'b']) == 1
              assert first_present({'a': None, 'b': 2}, ['a', 'b']) == 2
          ''',
          '''
          import pytest

          def test_falsy_values_count_as_present():
              assert first_present({'a': 0, 'b': 2}, ['a', 'b']) == 0
              assert first_present({'a': '', 'b': 'x'}, ['a', 'b']) == ''
              assert first_present({'a': False}, ['a']) is False
              assert first_present({'a': []}, ['a'], default='d') == []

          def test_default_and_validation():
              assert first_present({}, ['a'], default='fallback') == 'fallback'
              with pytest.raises(ValueError):
                  first_present({'a': 1}, ('a',))
          '''),
    # ── 来源组 7：v2_money_split ───────────────────────────────────────
    _task("split_cents", "money_split",
          "修复 split_cents(total_cents, people)：把 total_cents 分平均分成 people 份。"
          "令 q = total_cents // people，r = total_cents % people，返回长度为 people 的列表："
          "前 r 份为 q+1、其余为 q（列表之和必须等于 total_cents）。"
          "total_cents 为非负整数、people 为正整数（bool 不视为整数），否则 ValueError；禁止浮点运算路径。",
          '''
          def split_cents(total_cents, people):
              if total_cents < 0 or people < 1:
                  raise ValueError('invalid total_cents or people')
              base = round(total_cents / people)
              shares = [base] * people
              remainder = total_cents - base * people
              for index in range(remainder):
                  shares[index] += 1
              return shares
          ''',
          '''
          def split_cents(total_cents, people):
              if (isinstance(total_cents, bool) or not isinstance(total_cents, int)
                      or total_cents < 0
                      or isinstance(people, bool) or not isinstance(people, int)
                      or people < 1):
                  raise ValueError('invalid total_cents or people')
              quotient, remainder = divmod(total_cents, people)
              return [quotient + 1 if index < remainder else quotient
                      for index in range(people)]
          ''',
          '''
          def test_even_split():
              assert split_cents(10, 2) == [5, 5]
              assert split_cents(10, 4) == [3, 3, 2, 2]
          ''',
          '''
          import pytest

          def test_uneven_split_sums_exactly():
              assert split_cents(20, 3) == [7, 7, 6]
              assert sum(split_cents(20, 3)) == 20
              assert split_cents(2, 3) == [1, 1, 0]
              assert split_cents(100, 7) == [15, 15, 14, 14, 14, 14, 14]

          def test_validation():
              with pytest.raises(ValueError):
                  split_cents(10.0, 2)
              with pytest.raises(ValueError):
                  split_cents(True, 2)
              with pytest.raises(ValueError):
                  split_cents(10, 0)
              with pytest.raises(ValueError):
                  split_cents(-1, 2)
          '''),
    _task("format_cents", "money_split",
          "修复 format_cents(cents, symbol='$')：把整数分格式化为货币字符串。"
          "非负数输出 '符号+元+.+两位分'（如 12345 → '$123.45'，5 → '$0.05'，0 → '$0.00'）；"
          "负数在整个字符串最前加负号、数值按绝对值处理（如 -5 → '-$0.05'，-12345 → '-$123.45'）。"
          "cents 必须为整数（bool 不视为整数）否则 ValueError。",
          '''
          def format_cents(cents, symbol='$'):
              if isinstance(cents, bool) or not isinstance(cents, int):
                  raise ValueError('cents must be an integer')
              return f'{symbol}{cents // 100}.{cents % 100:02d}'
          ''',
          '''
          def format_cents(cents, symbol='$'):
              if isinstance(cents, bool) or not isinstance(cents, int):
                  raise ValueError('cents must be an integer')
              sign = '-' if cents < 0 else ''
              absolute = abs(cents)
              return f'{sign}{symbol}{absolute // 100}.{absolute % 100:02d}'
          ''',
          '''
          def test_positive():
              assert format_cents(12345) == '$123.45'
              assert format_cents(5) == '$0.05'
              assert format_cents(0) == '$0.00'
              assert format_cents(100, symbol='¥') == '¥1.00'
          ''',
          '''
          import pytest

          def test_negative_uses_absolute_value():
              assert format_cents(-5) == '-$0.05'
              assert format_cents(-12345) == '-$123.45'
              assert format_cents(-100) == '-$1.00'

          def test_validation():
              with pytest.raises(ValueError):
                  format_cents(1.5)
              with pytest.raises(ValueError):
                  format_cents(False)
          '''),
    # ── 来源组 8：v2_word_boundary ────────────────────────────────────
    _task("find_standalone", "word_boundary",
          "修复 find_standalone(text, word)：返回 word 在 text 中所有「独立出现」的起始索引（升序）。"
          "独立出现指该子串紧邻前后都不是词字符；词字符定义为 ASCII 字母、数字或下划线"
          "（[A-Za-z0-9_]），其他字符（含标点、空格、非 ASCII 字母）都不是词字符。"
          "word 为空串时抛 ValueError。大小写敏感。只统计原 text 中的出现。",
          '''
          def find_standalone(text, word):
              if not word:
                  raise ValueError('word must be nonempty')
              result = []
              start = 0
              while True:
                  index = text.find(word, start)
                  if index == -1:
                      break
                  result.append(index)
                  start = index + 1
              return result
          ''',
          '''
          def _is_word_char(ch):
              return ch.isascii() and (ch.isalnum() or ch == '_')

          def find_standalone(text, word):
              if not word:
                  raise ValueError('word must be nonempty')
              result = []
              start = 0
              while True:
                  index = text.find(word, start)
                  if index == -1:
                      break
                  before_ok = index == 0 or not _is_word_char(text[index - 1])
                  after = index + len(word)
                  after_ok = after >= len(text) or not _is_word_char(text[after])
                  if before_ok and after_ok:
                      result.append(index)
                  start = index + 1
              return result
          ''',
          '''
          def test_plain_occurrences():
              assert find_standalone('cat dog cat', 'cat') == [0, 8]
              assert find_standalone('dog', 'cat') == []

          def test_punctuation_boundary_counts():
              assert find_standalone('(cat).', 'cat') == [1]
          ''',
          '''
          import pytest

          def test_ignores_embedded_matches():
              assert find_standalone('cat catalog cat', 'cat') == [0, 12]
              assert find_standalone('concatenate', 'cat') == []

          def test_underscore_is_word_char():
              assert find_standalone('my_cat cat', 'cat') == [7]

          def test_empty_word_rejected():
              with pytest.raises(ValueError):
                  find_standalone('text', '')
          '''),
    _task("mask_tokens", "word_boundary",
          "修复 mask_token(text, token)：把 text 中每个「独立出现」的 token 替换为等长的 '*' 串。"
          "独立出现的边界定义与词字符判据同 find_standalone：紧邻前后都不是"
          "ASCII 字母/数字/下划线（或到达字符串端点）才算独立；"
          "嵌在更长词中的出现保持原样。token 为空串抛 ValueError。"
          "只对原 text 做一次扫描，替换产生的 '*' 不再参与匹配。",
          '''
          def mask_token(text, token):
              if not token:
                  raise ValueError('token must be nonempty')
              return text.replace(token, '*' * len(token))
          ''',
          '''
          def _is_word_char(ch):
              return ch.isascii() and (ch.isalnum() or ch == '_')

          def mask_token(text, token):
              if not token:
                  raise ValueError('token must be nonempty')
              out = []
              i = 0
              n = len(text)
              length = len(token)
              while i < n:
                  if text.startswith(token, i):
                      before_ok = i == 0 or not _is_word_char(text[i - 1])
                      after = i + length
                      after_ok = after >= n or not _is_word_char(text[after])
                      if before_ok and after_ok:
                          out.append('*' * length)
                          i = after
                          continue
                  out.append(text[i])
                  i += 1
              return ''.join(out)
          ''',
          '''
          def test_whole_word_masking():
              assert mask_token('secret key secret', 'secret') == '****** key ******'
              assert mask_token('[secret]', 'secret') == '[******]'
          ''',
          '''
          import pytest

          def test_embedded_occurrences_untouched():
              assert mask_token('secrets and secret', 'secret') == 'secrets and ******'
              assert mask_token('my_secret', 'secret') == 'my_secret'
              assert mask_token('tokenize token', 'token') == 'tokenize *****'

          def test_empty_token_rejected():
              with pytest.raises(ValueError):
                  mask_token('text', '')
          '''),
    # ── 来源组 9：v2_tree_parent_array ─────────────────────────────────
    _task("depth_list", "tree_parent_array",
          "修复 depth_list(parents)：parents[i] 为节点 i 的父节点编号或 None（表示根），"
          "返回每个节点的深度列表（根深度 0）。"
          "父节点编号可能大于子节点编号（数组顺序与树结构无关），必须支持任意顺序；"
          "父节点编号越界（不在 0..n-1）或不是 int（bool 不算）时抛 ValueError；"
          "检测到环（含 parents[i] == i 自环，以及更长的祖先环）时抛 ValueError；空数组返回 []。",
          '''
          def depth_list(parents):
              n = len(parents)
              depths = []
              for i in range(n):
                  p = parents[i]
                  if p is None:
                      depths.append(0)
                  else:
                      depths.append(depths[p] + 1)
              return depths
          ''',
          '''
          def depth_list(parents):
              n = len(parents)
              depths = [None] * n
              for start in range(n):
                  if depths[start] is not None:
                      continue
                  walk = []
                  index = start
                  while depths[index] is None:
                      walk.append(index)
                      p = parents[index]
                      if p is None:
                          depths[index] = 0
                          break
                      if (isinstance(p, bool) or not isinstance(p, int)
                              or not 0 <= p < n):
                          raise ValueError('parent index out of range')
                      if p in walk:
                          raise ValueError('cycle detected')
                      index = p
                  base = depths[index]
                  for node in reversed(walk):
                      if depths[node] is None:
                          base += 1
                          depths[node] = base
              return depths
          ''',
          '''
          def test_parents_in_order():
              assert depth_list([None, 0, 0, 1]) == [0, 1, 1, 2]
              assert depth_list([]) == []
          ''',
          '''
          import pytest

          def test_parents_out_of_order():
              assert depth_list([2, None, 1]) == [2, 0, 1]
              assert depth_list([1, None]) == [1, 0]

          def test_cycle_detection():
              with pytest.raises(ValueError):
                  depth_list([0])
              with pytest.raises(ValueError):
                  depth_list([1, 0])
              with pytest.raises(ValueError):
                  depth_list([None, 2, 1])

          def test_out_of_range():
              with pytest.raises(ValueError):
                  depth_list([None, 7])
              with pytest.raises(ValueError):
                  depth_list([True])
          ''', difficulty="L3"),
    _task("preorder", "tree_parent_array",
          "修复 preorder(parents)：parents 含义同上（单根、编号 0..n-1、无环，输入视为合法不做校验）。"
          "返回从根开始的深度优先先序遍历节点编号列表：每个节点先于其子节点出现，"
          "同一节点的子节点按编号升序访问；空数组返回 []。",
          '''
          from collections import deque

          def preorder(parents):
              children = {}
              root = None
              for node, parent in enumerate(parents):
                  if parent is None:
                      root = node
                  else:
                      children.setdefault(parent, []).append(node)
              order = []
              queue = deque([root] if root is not None else [])
              while queue:
                  node = queue.popleft()
                  order.append(node)
                  queue.extend(sorted(children.get(node, [])))
              return order
          ''',
          '''
          def preorder(parents):
              children = {}
              root = None
              for node, parent in enumerate(parents):
                  if parent is None:
                      root = node
                  else:
                      children.setdefault(parent, []).append(node)
              for kids in children.values():
                  kids.sort()
              order = []
              stack = [root] if root is not None else []
              while stack:
                  node = stack.pop()
                  order.append(node)
                  for child in reversed(children.get(node, [])):
                      stack.append(child)
              return order
          ''',
          '''
          def test_chain_and_empty():
              assert preorder([None, 0, 0]) == [0, 1, 2]
              assert preorder([]) == []
              assert preorder([None]) == [0]
          ''',
          '''
          def test_branching_is_depth_first():
              assert preorder([None, 0, 0, 1, 1, 2]) == [0, 1, 3, 4, 2, 5]
              assert preorder([None, 0, 0, 2]) == [0, 1, 2, 3]

          def test_children_ascending():
              assert preorder([None, 0, 0, 0]) == [0, 1, 2, 3]
          '''),
    # ── 来源组 10：v2_state_flags ─────────────────────────────────────
    _task("flags_decode", "state_flags",
          "修复 flags_to_text(value)：位标志 'read'=1、'write'=2、'exec'=4。"
          "按固定顺序 read、write、exec 输出 value 中存在的标志名，逗号分隔、无空格；"
          "无任何标志输出空串。value 必须为非负 int（bool 不视为整数）且不得含未知位"
          "（即 value & ~7 必须为 0），否则 ValueError。",
          '''
          _BITS = {'read': 1, 'write': 2, 'exec': 4}

          def flags_to_text(value):
              found = {name for name, bit in _BITS.items() if value & bit}
              return ','.join(sorted(found))
          ''',
          '''
          _ORDER = [('read', 1), ('write', 2), ('exec', 4)]

          def flags_to_text(value):
              if isinstance(value, bool) or not isinstance(value, int):
                  raise ValueError('value must be an integer')
              if value < 0 or value & ~7:
                  raise ValueError('unknown or negative flag value')
              return ','.join(name for name, bit in _ORDER if value & bit)
          ''',
          '''
          def test_single_flags():
              assert flags_to_text(1) == 'read'
              assert flags_to_text(0) == ''
          ''',
          '''
          import pytest

          def test_canonical_order():
              assert flags_to_text(7) == 'read,write,exec'
              assert flags_to_text(6) == 'write,exec'
              assert flags_to_text(5) == 'read,exec'

          def test_unknown_and_invalid_rejected():
              with pytest.raises(ValueError):
                  flags_to_text(8)
              with pytest.raises(ValueError):
                  flags_to_text(-1)
              with pytest.raises(ValueError):
                  flags_to_text(True)
          '''),
    _task("doc_workflow", "state_flags",
          "修复 next_state(current, event)：文档工作流状态机，状态集合 "
          "{'draft','review','published','archived'}。"
          "唯一合法迁移：draft --submit--> review；review --approve--> published；"
          "review --reject--> draft；published --archive--> archived。"
          "archived 是终态，没有任何合法迁出；其余任何状态/事件组合（含未知状态、未知事件）"
          "一律抛 ValueError。",
          '''
          def next_state(current, event):
              if event == 'submit' and current == 'draft':
                  return 'review'
              if event == 'approve' and current == 'review':
                  return 'published'
              if event == 'reject' and current == 'review':
                  return 'rejected'
              if event == 'archive':
                  return 'archived'
              raise ValueError(f'illegal transition: {current} --{event}-->')
          ''',
          '''
          _TRANSITIONS = {
              ('draft', 'submit'): 'review',
              ('review', 'approve'): 'published',
              ('review', 'reject'): 'draft',
              ('published', 'archive'): 'archived',
          }

          def next_state(current, event):
              if (current, event) not in _TRANSITIONS:
                  raise ValueError(f'illegal transition: {current} --{event}-->')
              return _TRANSITIONS[(current, event)]
          ''',
          '''
          def test_forward_path():
              assert next_state('draft', 'submit') == 'review'
              assert next_state('review', 'approve') == 'published'
          ''',
          '''
          import pytest

          def test_reject_returns_to_draft():
              assert next_state('review', 'reject') == 'draft'

          def test_archive_only_from_published():
              assert next_state('published', 'archive') == 'archived'
              for state in ('draft', 'review', 'archived'):
                  with pytest.raises(ValueError):
                      next_state(state, 'archive')

          def test_illegal_combinations():
              with pytest.raises(ValueError):
                  next_state('review', 'submit')
              with pytest.raises(ValueError):
                  next_state('draft', 'approve')
              with pytest.raises(ValueError):
                  next_state('unknown', 'submit')
              with pytest.raises(ValueError):
                  next_state('draft', 'unknown')
          ''', difficulty="L3"),
]
