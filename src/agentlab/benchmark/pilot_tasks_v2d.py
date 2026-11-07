"""Twenty v2d repair tasks in ten more source groups (scale-up batch 4).

New families: roman numerals, combinatorics, nested structures, justify text,
matrix rotation, identifier rules, linked list, priority semantics, table join
and char frequency.  Every buggy version passes its public tests but fails
hidden private tests, so an unchanged candidate is graded ``partial``.
"""
from __future__ import annotations

from textwrap import dedent


def _task(name, group, instruction, buggy, fixed, public_tests, private_tests,
          difficulty="L2"):
    module = f"v2d_{name}"
    return {
        "task_id": f"v2d_{name}_001", "source_group": f"v2d_{group}",
        "split": "train", "bug_family": group, "difficulty": difficulty,
        "instruction": instruction, "module": f"{module}.py",
        "buggy": dedent(buggy).strip() + "\n",
        "fixed": dedent(fixed).strip() + "\n",
        "public_test": f"from {module} import *\n\n" + dedent(public_tests).strip() + "\n",
        "private_tests": {"test_regression.py":
                          f"import pytest\nfrom {module} import *\n\n"
                          + dedent(private_tests).strip() + "\n"},
    }


V2D_SPECS = [
    # ── 来源组 1：v2d_roman_numerals ───────────────────────────────────
    _task("to_roman", "roman_numerals",
          "修复 to_roman(n)：把 1..3999 的整数转为标准罗马数字（必须使用减法形式："
          "4→'IV'、9→'IX'、40→'XL'、90→'XC'、400→'CD'、900→'CM'）；"
          "n 非 int（bool 不视为整数）或越界抛 ValueError。",
          '''
          _VALUES = [(1000, 'M'), (500, 'D'), (100, 'C'), (50, 'L'),
                     (10, 'X'), (5, 'V'), (1, 'I')]

          def to_roman(n):
              if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 3999:
                  raise ValueError('n must be an integer in 1..3999')
              out = []
              for value, symbol in _VALUES:
                  while n >= value:
                      out.append(symbol)
                      n -= value
              return ''.join(out)
          ''',
          '''
          _PAIRS = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
                    (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
                    (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]

          def to_roman(n):
              if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 3999:
                  raise ValueError('n must be an integer in 1..3999')
              out = []
              for value, symbol in _PAIRS:
                  while n >= value:
                      out.append(symbol)
                      n -= value
              return ''.join(out)
          ''',
          '''
          def test_simple_values():
              assert to_roman(1) == 'I'
              assert to_roman(10) == 'X'
              assert to_roman(1000) == 'M'

          def test_validation():
              import pytest
              with pytest.raises(ValueError):
                  to_roman(0)
              with pytest.raises(ValueError):
                  to_roman(4000)
          ''',
          '''
          import pytest

          def test_subtractive_forms_required():
              assert to_roman(4) == 'IV'
              assert to_roman(9) == 'IX'
              assert to_roman(40) == 'XL'
              assert to_roman(90) == 'XC'
              assert to_roman(400) == 'CD'
              assert to_roman(900) == 'CM'

          def test_composite():
              assert to_roman(1994) == 'MCMXCIV'
              assert to_roman(3888) == 'MMMDCCCLXXXVIII'
              assert to_roman(58) == 'LVIII'

          def test_type_validation():
              with pytest.raises(ValueError):
                  to_roman(True)
              with pytest.raises(ValueError):
                  to_roman(3.0)
          '''),
    _task("from_roman", "roman_numerals",
          "修复 from_roman(text)：解析罗马数字字符串为 int，使用标准前缀减法规则"
          "（较小值出现在较大值之前则做减法，如 'IV'→4、'MCMXCIV'→1994；"
          "重复符号按加法，'IIII'→4 也接受）；空串、非字符串或含非法字符抛 ValueError。",
          '''
          _VALUES = {'I': 1, 'V': 5, 'X': 10, 'L': 50,
                     'C': 100, 'D': 500, 'M': 1000}

          def from_roman(text):
              if not isinstance(text, str) or not text:
                  raise ValueError('text must be a nonempty roman numeral string')
              total = 0
              for ch in text:
                  if ch not in _VALUES:
                      raise ValueError(f'invalid symbol: {ch!r}')
                  total += _VALUES[ch]
              return total
          ''',
          '''
          _VALUES = {'I': 1, 'V': 5, 'X': 10, 'L': 50,
                     'C': 100, 'D': 500, 'M': 1000}

          def from_roman(text):
              if not isinstance(text, str) or not text:
                  raise ValueError('text must be a nonempty roman numeral string')
              values = []
              for ch in text:
                  if ch not in _VALUES:
                      raise ValueError(f'invalid symbol: {ch!r}')
                  values.append(_VALUES[ch])
              total = 0
              for index, value in enumerate(values):
                  if index + 1 < len(values) and values[index + 1] > value:
                      total -= value
                  else:
                      total += value
              return total
          ''',
          '''
          def test_additive_forms():
              assert from_roman('X') == 10
              assert from_roman('III') == 3
              assert from_roman('MM') == 2000

          def test_invalid_input():
              import pytest
              with pytest.raises(ValueError):
                  from_roman('')
              with pytest.raises(ValueError):
                  from_roman('ABC')
          ''',
          '''
          import pytest

          def test_prefix_subtraction():
              assert from_roman('IV') == 4
              assert from_roman('IX') == 9
              assert from_roman('XL') == 40
              assert from_roman('MCMXCIV') == 1994

          def test_lowercase_rejected():
              with pytest.raises(ValueError):
                  from_roman('iv')

          def test_non_string_rejected():
              with pytest.raises(ValueError):
                  from_roman(4)
          '''),
    # ── 来源组 2：v2d_combinatorics ────────────────────────────────────
    _task("power_set", "combinatorics",
          "修复 power_set(items)：返回全部子集（每个子集是 tuple），"
          "子集内元素保持输入顺序；子集排列顺序：按长度升序，同长度按组合枚举序"
          "（即按原位置组合的先后）；空列表返回 [()]；items 必须为 list 否则 ValueError。",
          '''
          from itertools import combinations

          def power_set(items):
              if not isinstance(items, list):
                  raise ValueError('items must be a list')
              result = []
              for size in range(len(items) + 1):
                  result.extend(tuple(sorted(subset))
                                for subset in combinations(items, size))
              return result
          ''',
          '''
          from itertools import combinations

          def power_set(items):
              if not isinstance(items, list):
                  raise ValueError('items must be a list')
              result = []
              for size in range(len(items) + 1):
                  result.extend(combinations(items, size))
              return result
          ''',
          '''
          def test_sorted_input_and_empty():
              assert power_set([1, 2]) == [(), (1,), (2,), (1, 2)]
              assert power_set([]) == [()]

          def test_validation():
              import pytest
              with pytest.raises(ValueError):
                  power_set('ab')
          ''',
          '''
          def test_input_order_preserved():
              assert power_set([3, 1]) == [(), (3,), (1,), (3, 1)]
              assert power_set(['b', 'a', 'c']) == [
                  (), ('b',), ('a',), ('c',),
                  ('b', 'a'), ('b', 'c'), ('a', 'c'),
                  ('b', 'a', 'c')]

          def test_duplicates_by_position():
              assert power_set([2, 2]) == [(), (2,), (2,), (2, 2)]
          '''),
    _task("cross_pairs", "combinatorics",
          "修复 cross_pairs(a, b)：返回笛卡尔积 [(x, y), ...]；"
          "顺序以 a 为主序（外层），每个 x 内按 b 的顺序；a 或 b 为空列表返回 []；"
          "a、b 必须为 list 否则 ValueError。",
          '''
          def cross_pairs(a, b):
              if not isinstance(a, list) or not isinstance(b, list):
                  raise ValueError('a and b must be lists')
              return [(x, y) for y in b for x in a]
          ''',
          '''
          def cross_pairs(a, b):
              if not isinstance(a, list) or not isinstance(b, list):
                  raise ValueError('a and b must be lists')
              return [(x, y) for x in a for y in b]
          ''',
          '''
          def test_single_side():
              assert cross_pairs([1, 2], ['p']) == [(1, 'p'), (2, 'p')]
              assert cross_pairs([], ['p']) == []
              assert cross_pairs([1], []) == []

          def test_validation():
              import pytest
              with pytest.raises(ValueError):
                  cross_pairs((1,), ['p'])
          ''',
          '''
          def test_outer_order_is_a():
              assert cross_pairs([1, 2], ['p', 'q']) == [
                  (1, 'p'), (1, 'q'), (2, 'p'), (2, 'q')]
              assert cross_pairs(['x', 'y'], [10, 20, 30]) == [
                  ('x', 10), ('x', 20), ('x', 30),
                  ('y', 10), ('y', 20), ('y', 30)]
          '''),
    # ── 来源组 3：v2d_nested_structures ────────────────────────────────
    _task("deep_merge", "nested_structures",
          "修复 deep_merge(base, override)：递归合并两个 dict——"
          "override 中键对应的值若与 base 同键值都是 dict，则递归合并；"
          "否则 override 的值覆盖；单侧存在的键直接保留；"
          "返回全新 dict，不修改任何输入；任一输入非 dict 抛 ValueError。",
          '''
          def deep_merge(base, override):
              if not isinstance(base, dict) or not isinstance(override, dict):
                  raise ValueError('base and override must be dicts')
              base.update(override)
              return base
          ''',
          '''
          def deep_merge(base, override):
              if not isinstance(base, dict) or not isinstance(override, dict):
                  raise ValueError('base and override must be dicts')
              merged = dict(base)
              for key, value in override.items():
                  if (key in merged and isinstance(merged[key], dict)
                          and isinstance(value, dict)):
                      merged[key] = deep_merge(merged[key], value)
                  else:
                      merged[key] = value
              return merged
          ''',
          '''
          def test_flat_merge():
              assert deep_merge({'a': 1, 'b': 2}, {'b': 3, 'c': 4}) == \
                  {'a': 1, 'b': 3, 'c': 4}

          def test_type_replacement():
              assert deep_merge({'x': 1}, {'x': {'y': 2}}) == {'x': {'y': 2}}
          ''',
          '''
          import pytest

          def test_nested_dicts_merge_recursively():
              assert deep_merge({'x': {'p': 1, 'q': 2}}, {'x': {'q': 9, 'r': 3}}) == \
                  {'x': {'p': 1, 'q': 9, 'r': 3}}

          def test_deeply_nested():
              assert deep_merge({'a': {'b': {'c': 1}}}, {'a': {'b': {'d': 2}}}) == \
                  {'a': {'b': {'c': 1, 'd': 2}}}

          def test_inputs_not_mutated():
              base = {'x': {'p': 1}}
              override = {'x': {'q': 2}}
              deep_merge(base, override)
              assert base == {'x': {'p': 1}}
              assert override == {'x': {'q': 2}}

          def test_validation():
              with pytest.raises(ValueError):
                  deep_merge([], {})
          ''', difficulty="L3"),
    _task("flatten_once", "nested_structures",
          "修复 flatten_once(items)：只拉平一层——元素为 list 的展开其内容，"
          "非 list 元素原样保留；嵌套更深的子列表保持原样；"
          "返回新列表，不修改输入；items 必须为 list 否则 ValueError。",
          '''
          def flatten_once(items):
              def walk(value):
                  out = []
                  for element in value:
                      if isinstance(element, list):
                          out.extend(walk(element))
                      else:
                          out.append(element)
                  return out
              return walk(items)
          ''',
          '''
          def flatten_once(items):
              if not isinstance(items, list):
                  raise ValueError('items must be a list')
              result = []
              for element in items:
                  if isinstance(element, list):
                      result.extend(element)
                  else:
                      result.append(element)
              return result
          ''',
          '''
          def test_one_level_flattening():
              assert flatten_once([1, [2, 3], 4]) == [1, 2, 3, 4]
              assert flatten_once([[1], [2]]) == [1, 2]
              assert flatten_once([[]]) == []
          ''',
          '''
          def test_deeper_levels_left_intact():
              assert flatten_once([1, [2, [3]]]) == [1, 2, [3]]
              assert flatten_once([[1, [2, 3]], [4]]) == [1, [2, 3], 4]

          def test_scalars_untouched_and_validation():
              assert flatten_once([1, 'a', None]) == [1, 'a', None]
              import pytest
              with pytest.raises(ValueError):
                  flatten_once('nope')
          '''),
    # ── 来源组 4：v2d_justify_text ─────────────────────────────────────
    _task("center_text", "justify_text",
          "修复 center_text(text, width)：用空格把 text 居中到宽度 width；"
          "总填充量 padding = width - len(text)；padding <= 0 时原样返回；"
          "左侧空格数为 padding // 2（多余的一个空格给右侧）；"
          "text 非 str、width 非 int（bool 不视为整数）抛 ValueError。",
          '''
          def center_text(text, width):
              if not isinstance(text, str):
                  raise ValueError('text must be a string')
              if isinstance(width, bool) or not isinstance(width, int):
                  raise ValueError('width must be an integer')
              padding = width - len(text)
              if padding <= 0:
                  return text
              left = (padding + 1) // 2
              return ' ' * left + text + ' ' * (padding - left)
          ''',
          '''
          def center_text(text, width):
              if not isinstance(text, str):
                  raise ValueError('text must be a string')
              if isinstance(width, bool) or not isinstance(width, int):
                  raise ValueError('width must be an integer')
              padding = width - len(text)
              if padding <= 0:
                  return text
              left = padding // 2
              return ' ' * left + text + ' ' * (padding - left)
          ''',
          '''
          def test_even_padding_and_short_width():
              assert center_text('ab', 6) == '  ab  '
              assert center_text('abc', 3) == 'abc'
              assert center_text('abcd', 2) == 'abcd'

          def test_validation():
              import pytest
              with pytest.raises(ValueError):
                  center_text(5, 10)
              with pytest.raises(ValueError):
                  center_text('a', True)
          ''',
          '''
          def test_odd_padding_extra_space_on_right():
              assert center_text('ab', 5) == ' ab  '
              assert center_text('x', 4) == ' x  '
              assert center_text('abc', 8) == '  abc   '
              assert center_text('ab', 3) == 'ab '
          '''),
    _task("truncate_ellipsis", "justify_text",
          "修复 truncate_ellipsis(text, width)：len(text) <= width 时原样返回；"
          "否则返回前 width-3 个字符加 '...'（总长恰为 width）；"
          "text 非 str、width 非 int（bool 不视为整数）或 width < 3 抛 ValueError。",
          '''
          def truncate_ellipsis(text, width):
              if not isinstance(text, str):
                  raise ValueError('text must be a string')
              if isinstance(width, bool) or not isinstance(width, int) or width < 3:
                  raise ValueError('width must be an integer >= 3')
              if len(text) <= width:
                  return text
              return text[:width - 2] + '...'
          ''',
          '''
          def truncate_ellipsis(text, width):
              if not isinstance(text, str):
                  raise ValueError('text must be a string')
              if isinstance(width, bool) or not isinstance(width, int) or width < 3:
                  raise ValueError('width must be an integer >= 3')
              if len(text) <= width:
                  return text
              return text[:width - 3] + '...'
          ''',
          '''
          def test_within_width_returned_as_is():
              assert truncate_ellipsis('hello', 10) == 'hello'
              assert truncate_ellipsis('abc', 3) == 'abc'

          def test_validation():
              import pytest
              with pytest.raises(ValueError):
                  truncate_ellipsis('text', 2)
              with pytest.raises(ValueError):
                  truncate_ellipsis(123, 5)
          ''',
          '''
          def test_exact_truncation_length():
              assert truncate_ellipsis('abcdefg', 6) == 'abc...'
              assert truncate_ellipsis('abcde', 4) == 'a...'
              assert len(truncate_ellipsis('x' * 50, 20)) == 20

          def test_boundary_width():
              assert truncate_ellipsis('abcd', 4) == 'abcd'
              assert truncate_ellipsis('abcde', 4) == 'a...'
              with pytest.raises(ValueError):
                  truncate_ellipsis('t', True)
          '''),
    # ── 来源组 5：v2d_matrix_rotation ──────────────────────────────────
    _task("rotate_90", "matrix_rotation",
          "修复 rotate_90(matrix)：把非空矩形二维列表顺时针旋转 90°"
          "（m 行 n 列 → n 行 m 列，第一行变成最后一列）；返回新矩阵不修改输入；"
          "空矩阵、[[]] 或行长不一致抛 ValueError。",
          '''
          def rotate_90(matrix):
              if not matrix or not matrix[0] or any(len(row) != len(matrix[0]) for row in matrix):
                  raise ValueError('matrix must be nonempty and rectangular')
              return [list(row) for row in zip(*matrix)][::-1]
          ''',
          '''
          def rotate_90(matrix):
              if not matrix or not matrix[0] or any(len(row) != len(matrix[0]) for row in matrix):
                  raise ValueError('matrix must be nonempty and rectangular')
              return [list(row) for row in zip(*matrix[::-1])]
          ''',
          '''
          def test_symmetric_and_single():
              assert rotate_90([[1, 1], [1, 1]]) == [[1, 1], [1, 1]]
              assert rotate_90([[5]]) == [[5]]

          def test_validation():
              import pytest
              with pytest.raises(ValueError):
                  rotate_90([])
              with pytest.raises(ValueError):
                  rotate_90([[1, 2], [3]])
          ''',
          '''
          def test_clockwise_direction():
              assert rotate_90([[1, 2], [3, 4]]) == [[3, 1], [4, 2]]
              assert rotate_90([[1, 2, 3], [4, 5, 6]]) == [[4, 1], [5, 2], [6, 3]]

          def test_input_not_mutated():
              matrix = [[1, 2], [3, 4]]
              rotate_90(matrix)
              assert matrix == [[1, 2], [3, 4]]
          '''),
    _task("anti_diagonal", "matrix_rotation",
          "修复 anti_diagonal(matrix)：返回方阵的反对角线元素列表"
          "（从右上角到左下角，即 matrix[i][n-1-i]）；"
          "空矩阵、行数与列数不等（非方阵）或行长不一致抛 ValueError。",
          '''
          def anti_diagonal(matrix):
              size = len(matrix)
              if (not matrix or any(len(row) != size for row in matrix)):
                  raise ValueError('matrix must be a nonempty square matrix')
              return [matrix[i][i] for i in range(size)]
          ''',
          '''
          def anti_diagonal(matrix):
              size = len(matrix)
              if (not matrix or any(len(row) != size for row in matrix)):
                  raise ValueError('matrix must be a nonempty square matrix')
              return [matrix[i][size - 1 - i] for i in range(size)]
          ''',
          '''
          def test_symmetric_matrix():
              assert anti_diagonal([[1, 1], [1, 1]]) == [1, 1]
              assert anti_diagonal([[7]]) == [7]

          def test_validation():
              import pytest
              with pytest.raises(ValueError):
                  anti_diagonal([])
              with pytest.raises(ValueError):
                  anti_diagonal([[1, 2, 3], [4, 5, 6]])
          ''',
          '''
          def test_anti_diagonal_values():
              assert anti_diagonal([[1, 2], [3, 4]]) == [2, 3]
              assert anti_diagonal([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [3, 5, 7]
              assert anti_diagonal([[0, 0, 9], [0, 8, 0], [7, 0, 0]]) == [9, 8, 7]
          '''),
    # ── 来源组 6：v2d_identifier_rules ─────────────────────────────────
    _task("valid_identifier", "identifier_rules",
          "修复 valid_identifier(name)：判断是否为合法 Python 标识符——"
          "非空字符串、长度 1..63、首字符为 ASCII 字母或下划线、"
          "其余字符为 ASCII 字母/数字/下划线、且不是 Python 关键字"
          "（用 keyword.iskeyword 判断）；返回 True/False，输入非字符串返回 False。",
          '''
          import keyword

          def valid_identifier(name):
              if not isinstance(name, str) or not 1 <= len(name) <= 63:
                  return False
              for ch in name:
                  if not (ch.isascii() and (ch.isalnum() or ch == '_')):
                      return False
              return True
          ''',
          '''
          import keyword

          def valid_identifier(name):
              if not isinstance(name, str) or not 1 <= len(name) <= 63:
                  return False
              if not (name[0].isascii() and (name[0].isalpha() or name[0] == '_')):
                  return False
              for ch in name[1:]:
                  if not (ch.isascii() and (ch.isalnum() or ch == '_')):
                      return False
              return not keyword.iskeyword(name)
          ''',
          '''
          def test_plain_identifiers():
              assert valid_identifier('value_1') is True
              assert valid_identifier('_hidden') is True

          def test_clearly_invalid():
              assert valid_identifier('has-dash') is False
              assert valid_identifier('') is False
              assert valid_identifier(42) is False
          ''',
          '''
          def test_leading_digit_rejected():
              assert valid_identifier('1st') is False
              assert valid_identifier('2_fast') is False

          def test_keywords_rejected():
              assert valid_identifier('class') is False
              assert valid_identifier('lambda') is False
              assert valid_identifier('class_name') is True

          def test_length_limits():
              assert valid_identifier('a' * 63) is True
              assert valid_identifier('a' * 64) is False
              assert valid_identifier('名') is False

          def test_single_char_and_digit_last():
              assert valid_identifier('x') is True
              assert valid_identifier('x9') is True
              assert valid_identifier('x_') is True
          '''),
    _task("snake_camel", "identifier_rules",
          "修复 snake_to_camel(text)：把小写 snake_case 转为 lowerCamelCase——"
          "首段保持小写，后续每段首字母大写（其余小写）；"
          "前导下划线原样保留（'_private_field'→'_privateField'）；"
          "单个词原样返回；只允许 ASCII 小写字母与下划线，"
          "出现其他字符或空串抛 ValueError。",
          '''
          def snake_to_camel(text):
              if not isinstance(text, str) or not text:
                  raise ValueError('text must be a nonempty snake_case string')
              for ch in text:
                  if not (ch.isascii() and (ch.islower() or ch == '_')):
                      raise ValueError('only lowercase ascii letters and underscores allowed')
              parts = text.split('_')
              camel = parts[0] + ''.join(part.capitalize() for part in parts[1:] if part)
              return camel.lstrip('_')
          ''',
          '''
          def snake_to_camel(text):
              if not isinstance(text, str) or not text:
                  raise ValueError('text must be a nonempty snake_case string')
              for ch in text:
                  if not (ch.isascii() and (ch.islower() or ch == '_')):
                      raise ValueError('only lowercase ascii letters and underscores allowed')
              leading = len(text) - len(text.lstrip('_'))
              body = text[leading:]
              parts = body.split('_') if body else ['']
              return '_' * leading + parts[0] + ''.join(
                  part.capitalize() for part in parts[1:] if part)
          ''',
          '''
          def test_plain_conversion():
              assert snake_to_camel('private_field') == 'privateField'
              assert snake_to_camel('data_value_set') == 'dataValueSet'
              assert snake_to_camel('word') == 'word'
          ''',
          '''
          import pytest

          def test_leading_underscore_preserved():
              assert snake_to_camel('_private_field') == '_privateField'
              assert snake_to_camel('__dunder_name') == '__dunderName'

          def test_rejected_inputs():
              with pytest.raises(ValueError):
                  snake_to_camel('Has-Upper')
              with pytest.raises(ValueError):
                  snake_to_camel('')
              with pytest.raises(ValueError):
                  snake_to_camel(123)
          '''),
    # ── 来源组 7：v2d_linked_list ──────────────────────────────────────
    _task("list_length", "linked_list",
          "模块提供 ListNode(value, next=None)。修复 list_length(head)："
          "遍历链表返回节点总数；head 为 None 返回 0；不得使用递归。",
          '''
          class ListNode:
              def __init__(self, value, next=None):
                  self.value = value
                  self.next = next

          def list_length(head):
              count = 0
              node = head
              while node is not None and node.next is not None:
                  count += 1
                  node = node.next
              return count
          ''',
          '''
          class ListNode:
              def __init__(self, value, next=None):
                  self.value = value
                  self.next = next

          def list_length(head):
              count = 0
              node = head
              while node is not None:
                  count += 1
                  node = node.next
              return count
          ''',
          '''
          def from_values(values):
              head = None
              for value in reversed(values):
                  head = ListNode(value, head)
              return head

          def test_empty_list():
              assert list_length(None) == 0

          def test_result_is_integer():
              assert isinstance(list_length(from_values([1, 2, 3])), int)
          ''',
          '''
          def from_values(values):
              head = None
              for value in reversed(values):
                  head = ListNode(value, head)
              return head

          def test_exact_counts():
              assert list_length(from_values([9])) == 1
              assert list_length(from_values([1, 2])) == 2
              assert list_length(from_values(list(range(5)))) == 5
              assert list_length(from_values([0] * 50)) == 50
          '''),
    _task("reverse_list", "linked_list",
          "模块提供 ListNode(value, next=None)、from_list(values)、to_list(head)。"
          "修复 reverse_list(head)：反转链表并返回新的头节点（复用原节点重排 next 指针）；"
          "空链表（None）返回 None；不得新建节点。",
          '''
          class ListNode:
              def __init__(self, value, next=None):
                  self.value = value
                  self.next = next

          def from_list(values):
              head = None
              for value in reversed(values):
                  head = ListNode(value, head)
              return head

          def to_list(head):
              values = []
              node = head
              while node is not None:
                  values.append(node.value)
                  node = node.next
              return values

          def reverse_list(head):
              previous = None
              node = head
              while node is not None:
                  node.next, previous, node = previous, node, node.next
              return node
          ''',
          '''
          class ListNode:
              def __init__(self, value, next=None):
                  self.value = value
                  self.next = next

          def from_list(values):
              head = None
              for value in reversed(values):
                  head = ListNode(value, head)
              return head

          def to_list(head):
              values = []
              node = head
              while node is not None:
                  values.append(node.value)
                  node = node.next
              return values

          def reverse_list(head):
              previous = None
              node = head
              while node is not None:
                  node.next, previous, node = previous, node, node.next
              return previous
          ''',
          '''
          def test_empty_chain():
              assert reverse_list(None) is None
              assert to_list(reverse_list(from_list([]))) == []
          ''',
          '''
          def test_reversal_values():
              assert to_list(reverse_list(from_list([1, 2, 3]))) == [3, 2, 1]
              assert to_list(reverse_list(from_list([7]))) == [7]
              assert to_list(reverse_list(from_list(['a', 'b', 'c', 'd']))) == \
                  ['d', 'c', 'b', 'a']

          def test_returns_node_type():
              head = reverse_list(from_list([1, 2]))
              assert isinstance(head, ListNode)
              assert head.value == 2
          '''),
    # ── 来源组 8：v2d_priority_semantics ───────────────────────────────
    _task("priority_queue", "priority_semantics",
          "修复类 PriorityQueue：push(item, priority) 入队（priority 为整数，"
          "数值越小越先出队）；相同 priority 的元素按入队先后（FIFO）；"
          "pop() 返回并移除队首元素，空队 pop 抛 IndexError；size() 返回当前数量。",
          '''
          import heapq

          class PriorityQueue:
              def __init__(self):
                  self._heap = []
                  self._order = 0

              def push(self, item, priority):
                  heapq.heappush(self._heap, (-priority, self._order, item))
                  self._order += 1

              def pop(self):
                  if not self._heap:
                      raise IndexError('pop from empty priority queue')
                  return heapq.heappop(self._heap)[2]

              def size(self):
                  return len(self._heap)
          ''',
          '''
          import heapq

          class PriorityQueue:
              def __init__(self):
                  self._heap = []
                  self._order = 0

              def push(self, item, priority):
                  heapq.heappush(self._heap, (priority, self._order, item))
                  self._order += 1

              def pop(self):
                  if not self._heap:
                      raise IndexError('pop from empty priority queue')
                  return heapq.heappop(self._heap)[2]

              def size(self):
                  return len(self._heap)
          ''',
          '''
          def test_single_item_and_size():
              queue = PriorityQueue()
              assert queue.size() == 0
              queue.push('only', 1)
              assert queue.size() == 1
              assert queue.pop() == 'only'

          def test_empty_pop_raises():
              import pytest
              with pytest.raises(IndexError):
                  PriorityQueue().pop()
          ''',
          '''
          def test_lower_priority_number_pops_first():
              queue = PriorityQueue()
              queue.push('low', 5)
              queue.push('high', 1)
              queue.push('mid', 3)
              assert [queue.pop() for _ in range(3)] == ['high', 'mid', 'low']

          def test_tie_is_fifo():
              queue = PriorityQueue()
              queue.push('first', 2)
              queue.push('second', 2)
              queue.push('third', 2)
              assert [queue.pop() for _ in range(3)] == ['first', 'second', 'third']

          def test_tie_fifo_not_by_item_value():
              queue = PriorityQueue()
              queue.push('z', 2)
              queue.push('a', 2)
              assert queue.pop() == 'z'
              assert queue.pop() == 'a'

          def test_negative_priority_allowed():
              queue = PriorityQueue()
              queue.push('urgent', -10)
              queue.push('normal', 0)
              assert queue.pop() == 'urgent'
          ''', difficulty="L3"),
    _task("bounded_stack", "priority_semantics",
          "修复类 BoundedStack(maxsize)：maxsize 正整数（bool 不视为整数）否则 "
          "ValueError。push(item)：未满时入栈并返回 True，已满返回 False 且不入栈；"
          "pop()：后进先出，空栈返回 None。",
          '''
          class BoundedStack:
              def __init__(self, maxsize):
                  if isinstance(maxsize, bool) or not isinstance(maxsize, int) or maxsize < 1:
                      raise ValueError('maxsize must be a positive integer')
                  self._maxsize = maxsize
                  self._items = []

              def push(self, item):
                  self._items.append(item)
                  return True

              def pop(self):
                  if not self._items:
                      return None
                  return self._items.pop()
          ''',
          '''
          class BoundedStack:
              def __init__(self, maxsize):
                  if isinstance(maxsize, bool) or not isinstance(maxsize, int) or maxsize < 1:
                      raise ValueError('maxsize must be a positive integer')
                  self._maxsize = maxsize
                  self._items = []

              def push(self, item):
                  if len(self._items) >= self._maxsize:
                      return False
                  self._items.append(item)
                  return True

              def pop(self):
                  if not self._items:
                      return None
                  return self._items.pop()
          ''',
          '''
          def test_basic_push_pop():
              stack = BoundedStack(3)
              assert stack.push('a') is True
              assert stack.pop() == 'a'
              assert stack.pop() is None

          def test_maxsize_validation():
              import pytest
              with pytest.raises(ValueError):
                  BoundedStack(0)
          ''',
          '''
          def test_full_stack_rejects_push():
              stack = BoundedStack(2)
              assert stack.push('x') is True
              assert stack.push('y') is True
              assert stack.push('z') is False
              assert stack.pop() == 'y'
              assert stack.push('z') is True

          def test_lifo_order():
              stack = BoundedStack(5)
              for value in (1, 2, 3):
                  stack.push(value)
              assert [stack.pop() for _ in range(3)] == [3, 2, 1]
          '''),
    # ── 来源组 9：v2d_table_join ───────────────────────────────────────
    _task("join_rows", "table_join",
          "修复 join_rows(left, right, key)：两个 dict 行列表按 key 字段做内连接。"
          "返回合并行列表（{**左行, **右行}，右行字段覆盖同名冲突）；"
          "行序：按 left 顺序，同一左行匹配多个右行时按右行出现顺序展开；"
          "无匹配的左行跳过；left/right 必须为 list、key 为 str、"
          "任一行不是 dict 或缺 key 字段抛 ValueError。",
          '''
          def join_rows(left, right, key):
              if not isinstance(left, list) or not isinstance(right, list) or not isinstance(key, str):
                  raise ValueError('left/right must be lists and key a string')
              for row in [*left, *right]:
                  if not isinstance(row, dict) or key not in row:
                      raise ValueError(f'every row must be a dict containing {key!r}')
              indexed = {}
              for row in right:
                  indexed[row[key]] = row
              return [{**row, **indexed[row[key]]}
                      for row in left if row[key] in indexed]
          ''',
          '''
          def join_rows(left, right, key):
              if not isinstance(left, list) or not isinstance(right, list) or not isinstance(key, str):
                  raise ValueError('left/right must be lists and key a string')
              for row in [*left, *right]:
                  if not isinstance(row, dict) or key not in row:
                      raise ValueError(f'every row must be a dict containing {key!r}')
              indexed = {}
              for row in right:
                  indexed.setdefault(row[key], []).append(row)
              merged = []
              for row in left:
                  for match in indexed.get(row[key], []):
                      merged.append({**row, **match})
              return merged
          ''',
          '''
          def test_unique_join_and_no_match():
              left = [{'id': 1, 'name': 'a'}, {'id': 2, 'name': 'b'}]
              right = [{'id': 1, 'score': 9}]
              assert join_rows(left, right, 'id') == [
                  {'id': 1, 'name': 'a', 'score': 9}]

          def test_validation():
              import pytest
              with pytest.raises(ValueError):
                  join_rows([{'id': 1}], [{'no_id': 1}], 'id')
          ''',
          '''
          def test_multiple_matches_expand_in_right_order():
              left = [{'id': 1, 'name': 'a'}]
              right = [{'id': 1, 'score': 1}, {'id': 1, 'score': 2}]
              assert join_rows(left, right, 'id') == [
                  {'id': 1, 'name': 'a', 'score': 1},
                  {'id': 1, 'name': 'a', 'score': 2}]

          def test_right_overrides_and_left_order():
              left = [{'id': 1, 'x': 'L'}, {'id': 2, 'x': 'L'}]
              right = [{'id': 2, 'x': 'R'}, {'id': 1, 'x': 'R'}]
              assert join_rows(left, right, 'id') == [
                  {'id': 1, 'x': 'R'}, {'id': 2, 'x': 'R'}]

          def test_empty_inputs():
              assert join_rows([], [{'id': 1}], 'id') == []
              assert join_rows([{'id': 1}], [], 'id') == []
          '''),
    _task("index_by", "table_join",
          "修复 index_by(rows, key)：把 dict 行列表按 key 字段值建索引，"
          "返回 {键值: 行}；出现重复键值抛 ValueError；"
          "rows 必须为 list、key 为 str、任一行非 dict 或缺 key 字段抛 ValueError。",
          '''
          def index_by(rows, key):
              if not isinstance(rows, list) or not isinstance(key, str):
                  raise ValueError('rows must be a list and key a string')
              result = {}
              for row in rows:
                  if not isinstance(row, dict) or key not in row:
                      raise ValueError(f'every row must be a dict containing {key!r}')
                  result[row[key]] = row
              return result
          ''',
          '''
          def index_by(rows, key):
              if not isinstance(rows, list) or not isinstance(key, str):
                  raise ValueError('rows must be a list and key a string')
              result = {}
              for row in rows:
                  if not isinstance(row, dict) or key not in row:
                      raise ValueError(f'every row must be a dict containing {key!r}')
                  if row[key] in result:
                      raise ValueError(f'duplicate key value: {row[key]!r}')
                  result[row[key]] = row
              return result
          ''',
          '''
          def test_unique_keys():
              rows = [{'id': 1, 'n': 'a'}, {'id': 2, 'n': 'b'}]
              assert index_by(rows, 'id') == {1: {'id': 1, 'n': 'a'},
                                               2: {'id': 2, 'n': 'b'}}
              assert index_by([], 'id') == {}

          def test_row_validation():
              import pytest
              with pytest.raises(ValueError):
                  index_by([{'n': 'a'}], 'id')
          ''',
          '''
          import pytest

          def test_duplicate_keys_rejected():
              with pytest.raises(ValueError):
                  index_by([{'id': 1}, {'id': 1}], 'id')

          def test_container_validation():
              with pytest.raises(ValueError):
                  index_by('rows', 'id')
              with pytest.raises(ValueError):
                  index_by([{'id': 1}], 1)
          '''),
    # ── 来源组 10：v2d_char_frequency ──────────────────────────────────
    _task("top_frequent", "char_frequency",
          "修复 top_k_frequent(words, k)：统计词频，返回前 k 个 (word, count)："
          "按出现次数降序，次数相同按该词首次出现顺序；"
          "k 为正整数（bool 不视为整数），k 超过不同词数时返回全部；"
          "words 必须为字符串列表否则 ValueError。",
          '''
          def top_k_frequent(words, k):
              if not isinstance(words, list) or not all(isinstance(w, str) for w in words):
                  raise ValueError('words must be a list of strings')
              if isinstance(k, bool) or not isinstance(k, int) or k < 1:
                  raise ValueError('k must be a positive integer')
              counts = {}
              for word in words:
                  counts[word] = counts.get(word, 0) + 1
              ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
              return ranked[:k]
          ''',
          '''
          def top_k_frequent(words, k):
              if not isinstance(words, list) or not all(isinstance(w, str) for w in words):
                  raise ValueError('words must be a list of strings')
              if isinstance(k, bool) or not isinstance(k, int) or k < 1:
                  raise ValueError('k must be a positive integer')
              counts = {}
              for word in words:
                  counts[word] = counts.get(word, 0) + 1
              ranked = sorted(counts.items(),
                              key=lambda item: (-item[1], words.index(item[0])))
              return ranked[:k]
          ''',
          '''
          def test_distinct_counts():
              assert top_k_frequent(['a', 'a', 'b'], 2) == [('a', 2), ('b', 1)]
              assert top_k_frequent(['a'], 5) == [('a', 1)]
          ''',
          '''
          def test_ties_keep_first_seen_order():
              assert top_k_frequent(['b', 'a', 'b', 'a'], 2) == [('b', 2), ('a', 2)]
              assert top_k_frequent(['zz', 'm', 'm', 'zz', 'k'], 3) == [
                  ('zz', 2), ('m', 2), ('k', 1)]

          def test_k_boundaries_and_validation():
              assert top_k_frequent(['x', 'y'], 1) == [('x', 1)]
              import pytest
              with pytest.raises(ValueError):
                  top_k_frequent(['a'], 0)
              with pytest.raises(ValueError):
                  top_k_frequent(['a', 1], 1)
          '''),
    _task("count_distinct", "char_frequency",
          "修复 count_distinct(items)：返回不同字符串元素的个数（按值判等）；"
          "items 必须为字符串列表否则 ValueError；空列表返回 0。",
          '''
          def count_distinct(items):
              if not isinstance(items, list) or not all(isinstance(x, str) for x in items):
                  raise ValueError('items must be a list of strings')
              return len(items)
          ''',
          '''
          def count_distinct(items):
              if not isinstance(items, list) or not all(isinstance(x, str) for x in items):
                  raise ValueError('items must be a list of strings')
              return len(set(items))
          ''',
          '''
          def test_all_distinct_and_empty():
              assert count_distinct(['a', 'b', 'c']) == 3
              assert count_distinct([]) == 0

          def test_validation():
              import pytest
              with pytest.raises(ValueError):
                  count_distinct('abc')
          ''',
          '''
          def test_duplicates_counted_once():
              assert count_distinct(['a', 'a', 'b']) == 2
              assert count_distinct(['x'] * 10) == 1
              assert count_distinct(['', '']) == 1
          '''),
]
