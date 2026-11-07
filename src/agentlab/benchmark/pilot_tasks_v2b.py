"""Twenty v2b repair tasks in ten more source groups (scale-up batch 2).

New families distinct from all previous pools: accumulator state, duration
units, run encoding, sorted merging, bit operations, frozen defaults,
exception mapping, status codes, default merging and integer math.
Every buggy version passes its public tests but fails hidden private tests.
"""
from __future__ import annotations

from textwrap import dedent


def _task(name, group, instruction, buggy, fixed, public_tests, private_tests,
          difficulty="L2"):
    module = f"v2b_{name}"
    return {
        "task_id": f"v2b_{name}_001", "source_group": f"v2b_{group}",
        "split": "train", "bug_family": group, "difficulty": difficulty,
        "instruction": instruction, "module": f"{module}.py",
        "buggy": dedent(buggy).strip() + "\n",
        "fixed": dedent(fixed).strip() + "\n",
        "public_test": f"from {module} import *\n\n" + dedent(public_tests).strip() + "\n",
        "private_tests": {"test_regression.py":
                          f"import pytest\nfrom {module} import *\n\n"
                          + dedent(private_tests).strip() + "\n"},
    }


V2B_SPECS = [
    # ── 来源组 1：v2b_accumulator_state ─────────────────────────────────
    _task("stats_summary", "accumulator_state",
          "修复 summarize(numbers)：numbers 为非空列表且元素全部为数字（int/float，"
          "bool 不视为数字），否则 ValueError。返回 {'min': 最小值, 'max': 最大值, "
          "'mean': 平均值}；mean 一律为 float 除法结果。全负数列表的 max 必须正确"
          "（不得用 0 初始化）。",
          '''
          def summarize(numbers):
              if not isinstance(numbers, list) or not numbers:
                  raise ValueError('numbers must be a nonempty list')
              minimum = numbers[0]
              maximum = 0
              total = 0
              for value in numbers:
                  if value < minimum:
                      minimum = value
                  if value > maximum:
                      maximum = value
                  total += value
              return {'min': minimum, 'max': maximum,
                      'mean': total // len(numbers)}
          ''',
          '''
          def summarize(numbers):
              if not isinstance(numbers, list) or not numbers:
                  raise ValueError('numbers must be a nonempty list')
              for value in numbers:
                  if isinstance(value, bool) or not isinstance(value, (int, float)):
                      raise ValueError('numbers must contain only numbers')
              return {'min': min(numbers), 'max': max(numbers),
                      'mean': sum(numbers) / len(numbers)}
          ''',
          '''
          def test_basic_summary():
              assert summarize([1, 2, 3]) == {'min': 1, 'max': 3, 'mean': 2.0}
          ''',
          '''
          import pytest

          def test_mean_is_float_division():
              assert summarize([1, 2])['mean'] == 1.5
              assert summarize([2, 4, 6])['mean'] == 4.0

          def test_all_negative_max():
              assert summarize([-2, -5]) == {'min': -5, 'max': -2, 'mean': -3.5}

          def test_validation():
              with pytest.raises(ValueError):
                  summarize([])
              with pytest.raises(ValueError):
                  summarize([True, 1])
          '''),
    _task("weighted_mean", "accumulator_state",
          "修复类 WeightedMean：add(value, weight) 累加一个样本，weight 必须为正数"
          "否则 ValueError（已累加的样本不受失败调用影响）；value() 在无样本时返回 None，"
          "否则返回按权重加权平均 Σ(value×weight)/Σweight。",
          '''
          class WeightedMean:
              def __init__(self):
                  self._total = 0.0
                  self._count = 0.0

              def add(self, value, weight):
                  if weight <= 0:
                      raise ValueError('weight must be positive')
                  self._total += value
                  self._count += 1

              def value(self):
                  if self._count == 0:
                      return None
                  return self._total / self._count
          ''',
          '''
          class WeightedMean:
              def __init__(self):
                  self._weighted = 0.0
                  self._weight = 0.0

              def add(self, value, weight):
                  if weight <= 0:
                      raise ValueError('weight must be positive')
                  self._weighted += value * weight
                  self._weight += weight

              def value(self):
                  if self._weight == 0:
                      return None
                  return self._weighted / self._weight
          ''',
          '''
          def test_equal_weights():
              stat = WeightedMean()
              stat.add(2, 1)
              stat.add(4, 1)
              assert stat.value() == 3.0

          def test_empty_returns_none():
              assert WeightedMean().value() is None
          ''',
          '''
          import pytest

          def test_weighted_average():
              stat = WeightedMean()
              stat.add(1, 1)
              stat.add(5, 3)
              assert stat.value() == 4.0

          def test_single_heavy_sample():
              stat = WeightedMean()
              stat.add(10, 0.5)
              assert stat.value() == 10.0

          def test_invalid_weight_rejected():
              stat = WeightedMean()
              with pytest.raises(ValueError):
                  stat.add(1, 0)
              with pytest.raises(ValueError):
                  stat.add(1, -2)
              assert stat.value() is None
          '''),
    # ── 来源组 2：v2b_duration_units ───────────────────────────────────
    _task("parse_duration", "duration_units",
          "修复 parse_duration(text)：把时长字符串转为总秒数（int）。"
          "合法格式为若干段「非负整数 + 单位」，单位取 'h'(3600秒)、'm'(60秒)、's'(1秒)，"
          "段顺序任意、单位可缺省，但同一单位最多出现一次；每段必须有数字（'h' 非法）、"
          "结尾不得有无单位的数字；至少要有一个段；出现其他字符一律 ValueError。"
          "空串或非字符串 ValueError。",
          '''
          def parse_duration(text):
              hours = minutes = seconds = 0
              number = ''
              for ch in text:
                  if ch.isdigit():
                      number += ch
                  elif ch in 'hms':
                      value = int(number or 0)
                      if ch == 'h':
                          hours = value
                      elif ch == 'm':
                          minutes = value
                      else:
                          seconds = value
                      number = ''
                  else:
                      raise ValueError('invalid character')
              if number:
                  raise ValueError('trailing digits')
              return hours * 3600 + minutes * 60 + seconds
          ''',
          '''
          UNITS = {'h': 3600, 'm': 60, 's': 1}

          def parse_duration(text):
              if not isinstance(text, str) or not text:
                  raise ValueError('duration text must be a nonempty string')
              seen = set()
              total = 0
              number = ''
              for ch in text:
                  if ch.isdigit():
                      number += ch
                  elif ch in UNITS:
                      if not number:
                          raise ValueError('missing number before unit')
                      if ch in seen:
                          raise ValueError('duplicate unit')
                      seen.add(ch)
                      total += int(number) * UNITS[ch]
                      number = ''
                  else:
                      raise ValueError(f'invalid character: {ch!r}')
              if number:
                  raise ValueError('trailing digits without unit')
              if not seen:
                  raise ValueError('no duration segment found')
              return total
          ''',
          '''
          def test_standard_order():
              assert parse_duration('2h30m10s') == 9010
              assert parse_duration('90s') == 90
          ''',
          '''
          import pytest

          def test_any_unit_order():
              assert parse_duration('30s2h') == 7230
              assert parse_duration('5m1h') == 3900

          def test_duplicate_unit_rejected():
              with pytest.raises(ValueError):
                  parse_duration('1h2h')
              with pytest.raises(ValueError):
                  parse_duration('10s20s')

          def test_missing_number_rejected():
              with pytest.raises(ValueError):
                  parse_duration('h')
              with pytest.raises(ValueError):
                  parse_duration('2m m')
              with pytest.raises(ValueError):
                  parse_duration('')
          '''),
    _task("format_hms", "duration_units",
          "修复 to_hms(total_seconds)：非负 int（bool 不视为整数）转为 "
          "(hours, minutes, seconds) 三元组，minutes 与 seconds 必须落在 0..59；"
          "负数或非 int 抛 ValueError。",
          '''
          def to_hms(total_seconds):
              if (isinstance(total_seconds, bool) or not isinstance(total_seconds, int)
                      or total_seconds < 0):
                  raise ValueError('total_seconds must be a nonnegative integer')
              hours, rest = divmod(total_seconds, 3600)
              minutes, seconds = divmod(rest, 100)
              return (hours, minutes, seconds)
          ''',
          '''
          def to_hms(total_seconds):
              if (isinstance(total_seconds, bool) or not isinstance(total_seconds, int)
                      or total_seconds < 0):
                  raise ValueError('total_seconds must be a nonnegative integer')
              hours, rest = divmod(total_seconds, 3600)
              minutes, seconds = divmod(rest, 60)
              return (hours, minutes, seconds)
          ''',
          '''
          def test_exact_hours_and_small():
              assert to_hms(3600) == (1, 0, 0)
              assert to_hms(59) == (0, 0, 59)
          ''',
          '''
          import pytest

          def test_minutes_seconds_bounded():
              assert to_hms(3661) == (1, 1, 1)
              assert to_hms(7322) == (2, 2, 2)
              assert to_hms(359999) == (99, 59, 59)

          def test_validation():
              with pytest.raises(ValueError):
                  to_hms(-1)
              with pytest.raises(ValueError):
                  to_hms(60.0)
              with pytest.raises(ValueError):
                  to_hms(True)
          '''),
    # ── 来源组 3：v2b_run_encoding ─────────────────────────────────────
    _task("collapse_runs", "run_encoding",
          "修复 collapse_runs(text)：把「连续重复」的字符压缩为单个（'aabbcc'→'abc'）；"
          "不相邻的重复必须保留（'aba'→'aba'，'aabbaa'→'aba'）；空串返回空串。",
          '''
          def collapse_runs(text):
              seen = set()
              out = []
              for ch in text:
                  if ch not in seen:
                      seen.add(ch)
                      out.append(ch)
              return ''.join(out)
          ''',
          '''
          def collapse_runs(text):
              out = []
              previous = None
              for ch in text:
                  if ch != previous:
                      out.append(ch)
                      previous = ch
              return ''.join(out)
          ''',
          '''
          def test_adjacent_runs_collapse():
              assert collapse_runs('aabbcc') == 'abc'
              assert collapse_runs('') == ''
          ''',
          '''
          def test_nonadjacent_duplicates_kept():
              assert collapse_runs('aba') == 'aba'
              assert collapse_runs('aabbaa') == 'aba'
              assert collapse_runs('a b b a') == 'a b b a'
          '''),
    _task("run_length", "run_encoding",
          "修复 run_length(text)：游程编码，返回 [(字符, 连续出现次数), ...]，"
          "按出现顺序；同一字符不相邻的两段必须分开记录（'aabbaa'→"
          "[('a',2),('b',2),('a',2)]）；空串返回 []。",
          '''
          def run_length(text):
              counts = {}
              order = []
              for ch in text:
                  if ch not in counts:
                      counts[ch] = 0
                      order.append(ch)
                  counts[ch] += 1
              return [(ch, counts[ch]) for ch in order]
          ''',
          '''
          def run_length(text):
              runs = []
              for ch in text:
                  if runs and runs[-1][0] == ch:
                      runs[-1] = (ch, runs[-1][1] + 1)
                  else:
                      runs.append((ch, 1))
              return runs
          ''',
          '''
          def test_basic_runs():
              assert run_length('aaabbc') == [('a', 3), ('b', 2), ('c', 1)]
              assert run_length('') == []
          ''',
          '''
          def test_separated_runs_recorded_separately():
              assert run_length('aabbaa') == [('a', 2), ('b', 2), ('a', 2)]
              assert run_length('abc') == [('a', 1), ('b', 1), ('c', 1)]
          '''),
    # ── 来源组 4：v2b_sorted_merge ─────────────────────────────────────
    _task("k_smallest", "sorted_merge",
          "修复 k_smallest(numbers, k)：numbers 必须为非空列表，k 必须为 1..len(numbers) "
          "的整数（bool 不视为整数），否则 ValueError；返回最小的 k 个数（升序），"
          "不修改原列表。",
          '''
          def k_smallest(numbers, k):
              if not numbers or k < 0 or k > len(numbers):
                  raise ValueError('invalid numbers or k')
              return sorted(numbers)[:k]
          ''',
          '''
          def k_smallest(numbers, k):
              if (not isinstance(numbers, list) or not numbers
                      or isinstance(k, bool) or not isinstance(k, int)
                      or k < 1 or k > len(numbers)):
                  raise ValueError('numbers must be a nonempty list and '
                                   'k an integer in 1..len(numbers)')
              return sorted(numbers)[:k]
          ''',
          '''
          def test_basic_smallest():
              assert k_smallest([4, 1, 3], 2) == [1, 3]
              assert k_smallest([9], 1) == [9]
          ''',
          '''
          import pytest

          def test_input_unchanged():
              data = [5, 2, 8]
              k_smallest(data, 2)
              assert data == [5, 2, 8]

          def test_validation():
              with pytest.raises(ValueError):
                  k_smallest([1, 2], 0)
              with pytest.raises(ValueError):
                  k_smallest([1], True)
              with pytest.raises(ValueError):
                  k_smallest('abc', 1)
              with pytest.raises(ValueError):
                  k_smallest([], 1)
          '''),
    _task("merge_sorted", "sorted_merge",
          "修复 merge_sorted(a, b)：合并两个升序列表为一个升序列表，"
          "必须保留全部重复元素（不做去重）；任一列表可为空；不修改输入；"
          "用双指针线性扫描实现，禁止排序拼接。",
          '''
          def merge_sorted(a, b):
              result = []
              i = j = 0
              while i < len(a) and j < len(b):
                  if a[i] <= b[j]:
                      if not result or result[-1] != a[i]:
                          result.append(a[i])
                      i += 1
                  else:
                      if not result or result[-1] != b[j]:
                          result.append(b[j])
                      j += 1
              for rest, index in ((a, i), (b, j)):
                  while index < len(rest):
                      if not result or result[-1] != rest[index]:
                          result.append(rest[index])
                      index += 1
              return result
          ''',
          '''
          def merge_sorted(a, b):
              result = []
              i = j = 0
              while i < len(a) and j < len(b):
                  if a[i] <= b[j]:
                      result.append(a[i])
                      i += 1
                  else:
                      result.append(b[j])
                      j += 1
              result.extend(a[i:])
              result.extend(b[j:])
              return result
          ''',
          '''
          def test_disjoint_merge():
              assert merge_sorted([1, 3], [2, 4]) == [1, 2, 3, 4]
              assert merge_sorted([], [1]) == [1]
          ''',
          '''
          def test_duplicates_preserved():
              assert merge_sorted([1, 1, 2], [1, 3]) == [1, 1, 1, 2, 3]
              assert merge_sorted([2, 2], [2, 2]) == [2, 2, 2, 2]

          def test_inputs_unchanged():
              a = [1, 3]
              b = [2]
              merge_sorted(a, b)
              assert a == [1, 3] and b == [2]
          '''),
    # ── 来源组 5：v2b_bit_ops ──────────────────────────────────────────
    _task("count_bits", "bit_ops",
          "修复 count_set_bits(value)：返回 value 二进制表示中 1 的个数；"
          "value 必须为非负 int（bool 不视为整数），否则 ValueError。"
          "必须按二进制位统计，不得按十进制数字符统计。",
          '''
          def count_set_bits(value):
              if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                  raise ValueError('value must be a nonnegative integer')
              return str(value).count('1')
          ''',
          '''
          def count_set_bits(value):
              if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                  raise ValueError('value must be a nonnegative integer')
              return bin(value).count('1')
          ''',
          '''
          def test_trivial_values():
              assert count_set_bits(0) == 0
              assert count_set_bits(1) == 1
          ''',
          '''
          import pytest

          def test_binary_popcount():
              assert count_set_bits(7) == 3
              assert count_set_bits(10) == 2
              assert count_set_bits(255) == 8
              assert count_set_bits(1024) == 1

          def test_validation():
              with pytest.raises(ValueError):
                  count_set_bits(-1)
              with pytest.raises(ValueError):
                  count_set_bits(3.0)
          '''),
    _task("flip_bits", "bit_ops",
          "修复 flip_bits(value, mask)：返回 value 与 mask 按位异或的结果"
          "（mask 为 1 的位翻转 value 的对应位，为 0 的位保持）；"
          "两者必须为非负 int（bool 不视为整数），否则 ValueError。",
          '''
          def flip_bits(value, mask):
              if (isinstance(value, bool) or not isinstance(value, int) or value < 0
                      or isinstance(mask, bool) or not isinstance(mask, int)
                      or mask < 0):
                  raise ValueError('value and mask must be nonnegative integers')
              return value | mask
          ''',
          '''
          def flip_bits(value, mask):
              if (isinstance(value, bool) or not isinstance(value, int) or value < 0
                      or isinstance(mask, bool) or not isinstance(mask, int)
                      or mask < 0):
                  raise ValueError('value and mask must be nonnegative integers')
              return value ^ mask
          ''',
          '''
          def test_zero_value():
              assert flip_bits(0, 0b1111) == 15
              assert flip_bits(5, 0) == 5
          ''',
          '''
          import pytest

          def test_set_bits_are_toggled_off():
              assert flip_bits(0b1010, 0b1111) == 0b0101
              assert flip_bits(3, 1) == 2
              assert flip_bits(6, 6) == 0

          def test_validation():
              with pytest.raises(ValueError):
                  flip_bits(-1, 0)
              with pytest.raises(ValueError):
                  flip_bits(1, True)
          '''),
    # ── 来源组 6：v2b_frozen_defaults ──────────────────────────────────
    _task("append_log", "frozen_defaults",
          "修复 append_log(entry, log=None)：返回一个新列表 = （log 的浅拷贝）+ [entry]；"
          "log 为 None 时返回只含 entry 的新列表。不得修改调用方传入的 log，"
          "两次独立调用（不传 log）的结果互不影响，禁止可变默认参数累积状态。",
          '''
          def append_log(entry, log=[]):
              log.append(entry)
              return log
          ''',
          '''
          def append_log(entry, log=None):
              entries = list(log) if log is not None else []
              entries.append(entry)
              return entries
          ''',
          '''
          def test_single_call():
              assert append_log('a') == ['a']
              assert append_log('a', ['x']) == ['x', 'a']
          ''',
          '''
          def test_calls_are_isolated():
              first = append_log('a')
              second = append_log('b')
              assert first == ['a']
              assert second == ['b']

          def test_input_not_mutated():
              base = ['x']
              append_log('y', base)
              assert base == ['x']
          '''),
    _task("build_query", "frozen_defaults",
          "修复 build_query(base, params)：params 为 None 时原样返回 base；"
          "params 必须为 dict（否则 ValueError，不得让内部异常漏出）；"
          "返回 base + '?' + 查询串，查询参数按「键转字符串后升序」排列，"
          "每项格式 'k=v'（k、v 均转 str），用 '&' 连接；params 为空 dict 时返回 base + '?'；"
          "不得修改 params，也不得跨调用共享状态。",
          '''
          def build_query(base, params=None):
              if params is None:
                  return base
              pairs = [f'{key}={value}' for key, value in params.items()]
              return base + '?' + '&'.join(pairs)
          ''',
          '''
          def build_query(base, params=None):
              if params is None:
                  return base
              if not isinstance(params, dict):
                  raise ValueError('params must be a dict or None')
              pairs = [f'{key}={value}'
                       for key, value in sorted(params.items(),
                                                 key=lambda item: str(item[0]))]
              return base + '?' + '&'.join(pairs)
          ''',
          '''
          def test_none_and_single_param():
              assert build_query('/a') == '/a'
              assert build_query('/search', {'q': 'x'}) == '/search?q=x'
          ''',
          '''
          import pytest

          def test_keys_sorted_as_strings():
              assert build_query('/s', {'b': 2, 'a': 1}) == '/s?a=1&b=2'
              assert build_query('/s', {'z': 1, 'a': 2, 'm': 3}) == '/s?a=2&m=3&z=1'

          def test_empty_dict_keeps_separator():
              assert build_query('/s', {}) == '/s?'

          def test_params_not_mutated_and_validated():
              params = {'b': 2, 'a': 1}
              build_query('/s', params)
              assert params == {'b': 2, 'a': 1}
              with pytest.raises(ValueError):
                  build_query('/s', 'ab')
          ''', difficulty="L3"),
    # ── 来源组 7：v2b_exception_mapping ────────────────────────────────
    _task("safe_divide", "exception_mapping",
          "修复 safe_divide(a, b)：a、b 必须为数字（int/float，bool 不视为数字），"
          "类型不符抛 TypeError；b 为 0 时抛 ValueError('division by zero')"
          "（即把 ZeroDivisionError 翻译为 ValueError）；正常时返回 a / b。"
          "不得吞掉或错误翻译任何异常。",
          '''
          def safe_divide(a, b):
              try:
                  return a / b
              except Exception:
                  return None
          ''',
          '''
          def safe_divide(a, b):
              for value in (a, b):
                  if isinstance(value, bool) or not isinstance(value, (int, float)):
                      raise TypeError('safe_divide requires numeric arguments')
              if b == 0:
                  raise ValueError('division by zero')
              return a / b
          ''',
          '''
          def test_normal_division():
              assert safe_divide(6, 3) == 2.0
              assert safe_divide(2.5, 0.5) == 5.0
          ''',
          '''
          import pytest

          def test_zero_denominator_raises_value_error():
              with pytest.raises(ValueError):
                  safe_divide(1, 0)
              with pytest.raises(ValueError):
                  safe_divide(1, 0.0)

          def test_type_errors_propagate():
              with pytest.raises(TypeError):
                  safe_divide('a', 1)
              with pytest.raises(TypeError):
                  safe_divide(1, True)
              with pytest.raises(TypeError):
                  safe_divide(None, 2)
          ''', difficulty="L3"),
    _task("parse_int", "exception_mapping",
          "修复 parse_int(text, default=None)：text 必须为 str（否则 TypeError，"
          "包括 None）；对 str 尝试 int(text)，解析失败（ValueError）返回 default；"
          "解析成功返回 int。不得把 TypeError 也当成解析失败。",
          '''
          def parse_int(text, default=None):
              try:
                  return int(text)
              except Exception:
                  return default
          ''',
          '''
          def parse_int(text, default=None):
              if not isinstance(text, str):
                  raise TypeError('text must be a string')
              try:
                  return int(text)
              except ValueError:
                  return default
          ''',
          '''
          def test_parse_and_fallback():
              assert parse_int('42') == 42
              assert parse_int('x', default=-1) == -1
          ''',
          '''
          import pytest

          def test_non_string_raises_type_error():
              with pytest.raises(TypeError):
                  parse_int(None)
              with pytest.raises(TypeError):
                  parse_int(3.5)

          def test_whitespace_ints_parse():
              assert parse_int(' 7 ') == 7
              assert parse_int('', default=0) == 0
          '''),
    # ── 来源组 8：v2b_status_codes ─────────────────────────────────────
    _task("http_label", "status_codes",
          "修复 http_label(code)：200→'OK'、201→'Created'、204→'No Content'、"
          "404→'Not Found'、500→'Internal Server Error'（注意大小写完全一致）；"
          "其他整数返回 'Unknown'（不得返回 None）；code 非 int（bool 不视为整数）"
          "抛 ValueError。",
          '''
          _LABELS = {200: 'OK', 201: 'Created', 204: 'No content',
                     404: 'Not Found', 500: 'Internal Server Error'}

          def http_label(code):
              return _LABELS.get(code)
          ''',
          '''
          _LABELS = {200: 'OK', 201: 'Created', 204: 'No Content',
                     404: 'Not Found', 500: 'Internal Server Error'}

          def http_label(code):
              if isinstance(code, bool) or not isinstance(code, int):
                  raise ValueError('code must be an integer')
              return _LABELS.get(code, 'Unknown')
          ''',
          '''
          def test_known_labels():
              assert http_label(200) == 'OK'
              assert http_label(404) == 'Not Found'
          ''',
          '''
          import pytest

          def test_exact_casing():
              assert http_label(204) == 'No Content'
              assert http_label(500) == 'Internal Server Error'

          def test_unknown_and_invalid():
              assert http_label(302) == 'Unknown'
              assert http_label(0) == 'Unknown'
              with pytest.raises(ValueError):
                  http_label('200')
              with pytest.raises(ValueError):
                  http_label(True)
          '''),
    _task("severity_rank", "status_codes",
          "修复 severity_rank(name)：按严重级别返回等级整数：'info'→0、'warning'→1、"
          "'error'→2、'critical'→3；名称大小写不敏感；未知或非字符串输入一律 ValueError。",
          '''
          _LEVELS = ['critical', 'error', 'info', 'warning']

          def severity_rank(name):
              return _LEVELS.index(name.lower())
          ''',
          '''
          _RANKS = {'info': 0, 'warning': 1, 'error': 2, 'critical': 3}

          def severity_rank(name):
              if not isinstance(name, str):
                  raise ValueError(f'severity must be a string: {name!r}')
              key = name.lower()
              if key not in _RANKS:
                  raise ValueError(f'unknown severity: {name!r}')
              return _RANKS[key]
          ''',
          '''
          import pytest

          def test_returns_integer_and_case_insensitive():
              assert isinstance(severity_rank('error'), int)
              assert isinstance(severity_rank('ERROR'), int)

          def test_unknown_rejected():
              with pytest.raises(ValueError):
                  severity_rank('fatal')
          ''',
          '''
          import pytest

          def test_full_ranking():
              assert severity_rank('info') == 0
              assert severity_rank('warning') == 1
              assert severity_rank('error') == 2
              assert severity_rank('critical') == 3
              assert severity_rank('Info') == 0

          def test_non_string_rejected():
              with pytest.raises(ValueError):
                  severity_rank(3)
          '''),
    # ── 来源组 9：v2b_default_merge ────────────────────────────────────
    _task("fill_defaults", "default_merge",
          "修复 fill_defaults(settings, defaults)：返回新字典——settings 中"
          "「存在的键」一律保留原值（包括值为 None 或假值），defaults 只为 settings "
          "缺失的键补值；不修改任何输入；两者非 dict 抛 ValueError。",
          '''
          def fill_defaults(settings, defaults):
              if not isinstance(settings, dict) or not isinstance(defaults, dict):
                  raise ValueError('settings and defaults must be dicts')
              for key, value in defaults.items():
                  if settings.get(key) is None:
                      settings[key] = value
              return settings
          ''',
          '''
          def fill_defaults(settings, defaults):
              if not isinstance(settings, dict) or not isinstance(defaults, dict):
                  raise ValueError('settings and defaults must be dicts')
              merged = dict(defaults)
              merged.update(settings)
              return merged
          ''',
          '''
          def test_missing_keys_filled():
              assert fill_defaults({'a': 1}, {'b': 2}) == {'a': 1, 'b': 2}
              assert fill_defaults({}, {'b': 2}) == {'b': 2}
          ''',
          '''
          import pytest

          def test_present_none_is_kept():
              assert fill_defaults({'a': None}, {'a': 1, 'b': 2}) == {'a': None, 'b': 2}

          def test_inputs_not_mutated():
              settings = {'a': 1}
              defaults = {'b': 2}
              fill_defaults(settings, defaults)
              assert settings == {'a': 1}
              assert defaults == {'b': 2}

          def test_validation():
              with pytest.raises(ValueError):
                  fill_defaults(None, {})
          '''),
    _task("pick_changed", "default_merge",
          "修复 pick_changed(old, new)：返回新字典，只包含 new 中「old 里不存在」或"
          "「与 old 对应值不相等（== 比较）」的键，值取 new 的；仅 old 中存在的键忽略；"
          "不修改输入；两者非 dict 抛 ValueError。",
          '''
          def pick_changed(old, new):
              if not isinstance(old, dict) or not isinstance(new, dict):
                  raise ValueError('old and new must be dicts')
              return {key: new[key] for key in old
                      if key in new and new[key] != old[key]}
          ''',
          '''
          def pick_changed(old, new):
              if not isinstance(old, dict) or not isinstance(new, dict):
                  raise ValueError('old and new must be dicts')
              return {key: value for key, value in new.items()
                      if key not in old or value != old[key]}
          ''',
          '''
          def test_changed_values():
              assert pick_changed({'a': 1, 'b': 2}, {'a': 1, 'b': 3}) == {'b': 3}
              assert pick_changed({'a': 1}, {'a': 1}) == {}
          ''',
          '''
          import pytest

          def test_added_keys_included():
              assert pick_changed({'a': 1}, {'a': 1, 'b': 2}) == {'b': 2}
              assert pick_changed({}, {'x': 9}) == {'x': 9}

          def test_removed_keys_ignored():
              assert pick_changed({'a': 1, 'b': 2}, {'a': 1}) == {}

          def test_inputs_unchanged_and_validation():
              old = {'a': 1}
              new = {'a': 2}
              pick_changed(old, new)
              assert old == {'a': 1} and new == {'a': 2}
              with pytest.raises(ValueError):
                  pick_changed([], {})
          '''),
    # ── 来源组 10：v2b_integer_math ────────────────────────────────────
    _task("gcd_list", "integer_math",
          "修复 gcd_list(numbers)：numbers 为非空列表且元素全部为 int（bool 不视为整数），"
          "否则 ValueError；返回所有元素绝对值的最大公约数（结果非负）；"
          "必须对全部元素求公约数，不得只看部分元素。",
          '''
          import math

          def gcd_list(numbers):
              if not numbers:
                  raise ValueError('numbers must be a nonempty list')
              return math.gcd(abs(numbers[0]), abs(numbers[-1]))
          ''',
          '''
          import math

          def gcd_list(numbers):
              if not isinstance(numbers, list) or not numbers:
                  raise ValueError('numbers must be a nonempty list')
              result = 0
              for value in numbers:
                  if isinstance(value, bool) or not isinstance(value, int):
                      raise ValueError('numbers must contain only integers')
                  result = math.gcd(result, abs(value))
              return result
          ''',
          '''
          def test_pair_and_single():
              assert gcd_list([12, 18]) == 6
              assert gcd_list([7]) == 7
          ''',
          '''
          import pytest

          def test_reduces_over_all_elements():
              assert gcd_list([12, 18, 24]) == 6
              assert gcd_list([8, 12, 18]) == 2
              assert gcd_list([-4, 6]) == 2

          def test_validation():
              with pytest.raises(ValueError):
                  gcd_list([])
              with pytest.raises(ValueError):
                  gcd_list([True])
              with pytest.raises(ValueError):
                  gcd_list([4, 6.0])
          '''),
    _task("lcm_pair", "integer_math",
          "修复 lcm_pair(a, b)：a、b 必须为 int（bool 不视为整数），否则 ValueError；"
          "任一为 0 时返回 0（不得抛除零异常）；否则返回 |a×b| 除以 gcd(|a|,|b|)，"
          "结果非负。",
          '''
          import math

          def lcm_pair(a, b):
              return abs(a * b) // math.gcd(a, b)
          ''',
          '''
          import math

          def lcm_pair(a, b):
              if (isinstance(a, bool) or not isinstance(a, int)
                      or isinstance(b, bool) or not isinstance(b, int)):
                  raise ValueError('a and b must be integers')
              if a == 0 or b == 0:
                  return 0
              return abs(a * b) // math.gcd(abs(a), abs(b))
          ''',
          '''
          def test_coprime_and_composite():
              assert lcm_pair(4, 6) == 12
              assert lcm_pair(5, 7) == 35
          ''',
          '''
          import pytest

          def test_zero_returns_zero():
              assert lcm_pair(0, 5) == 0
              assert lcm_pair(0, 0) == 0

          def test_negative_uses_absolute_values():
              assert lcm_pair(-4, 6) == 12
              assert lcm_pair(4, -6) == 12

          def test_one_is_identity():
              assert lcm_pair(1, 5) == 5
              assert lcm_pair(5, 1) == 5
              assert lcm_pair(1, 1) == 1

          def test_negative_one_not_treated_as_zero():
              assert lcm_pair(-1, 6) == 6
              assert lcm_pair(6, -1) == 6

          def test_validation():
              with pytest.raises(ValueError):
                  lcm_pair(4.0, 6)
              with pytest.raises(ValueError):
                  lcm_pair(True, 6)
          '''),
]
