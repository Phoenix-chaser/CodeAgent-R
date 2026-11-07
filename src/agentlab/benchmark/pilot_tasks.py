"""Twenty-four self-authored pilot repairs, not a generalization benchmark.

Each source group contains two different repairs. Freeze group-level splits
before teacher collection. This small pack validates the training pipeline;
it is insufficient evidence of job-ready repository repair generalization.
"""
from __future__ import annotations

from textwrap import dedent


def _task(name, group, instruction, buggy, fixed, public_tests, private_tests):
    module = f"pilot_{name}"
    return {
        "task_id": f"pilot_{name}_001", "source_group": f"pilot_{group}",
        "split": "train", "bug_family": group, "difficulty": "L2",
        "instruction": instruction, "module": f"{module}.py",
        "buggy": dedent(buggy).strip() + "\n",
        "fixed": dedent(fixed).strip() + "\n",
        "public_test": f"from {module} import *\n\n" + dedent(public_tests).strip() + "\n",
        "private_tests": {"test_regression.py":
                          f"import pytest\nfrom {module} import *\n\n"
                          + dedent(private_tests).strip() + "\n"},
    }


PILOT_SPECS = [
    _task("option_pairs", "string_parsing",
          "修复 parse_options(text)：解析以分号分隔的 key=value，键和值去两端空白；"
          "值可包含等号，空白分段忽略，重复键最后值覆盖。缺少等号或空键抛 ValueError。",
          '''
          def parse_options(text):
              result = {}
              for part in text.split(';'):
                  if not part.strip():
                      continue
                  key, value = part.split('=')
                  result[key.strip()] = value.strip()
              return result
          ''',
          '''
          def parse_options(text):
              result = {}
              for part in text.split(';'):
                  if not part.strip():
                      continue
                  if '=' not in part:
                      raise ValueError('missing equals')
                  key, value = part.split('=', 1)
                  key = key.strip()
                  if not key:
                      raise ValueError('empty key')
                  result[key] = value.strip()
              return result
          ''',
          '''
          def test_equals_in_value():
              assert parse_options('token=a=b; mode=fast') == {'token': 'a=b', 'mode': 'fast'}
          def test_blank_input():
              assert parse_options(' ; ; ') == {}
          ''',
          '''
          def test_duplicate_and_empty_value():
              assert parse_options(' a=1;a= ;b=2 ') == {'a': '', 'b': '2'}
          def test_invalid_items():
              for text in ('missing', '=value', ' = value'):
                  with pytest.raises(ValueError):
                      parse_options(text)
          '''),
    _task("header_lines", "string_parsing",
          "修复 parse_headers(lines)：按第一个冒号拆分 HTTP 风格字段，键去空白并转小写，"
          "值去空白；重复键用 ', ' 合并且保持出现顺序。忽略空白行；无冒号或空键抛 ValueError。",
          '''
          def parse_headers(lines):
              result = {}
              for line in lines:
                  if not line.strip():
                      continue
                  key, value = line.split(':', 1)
                  result[key.strip().lower()] = value.strip()
              return result
          ''',
          '''
          def parse_headers(lines):
              result = {}
              for line in lines:
                  if not line.strip():
                      continue
                  if ':' not in line:
                      raise ValueError('missing colon')
                  key, value = line.split(':', 1)
                  key, value = key.strip().lower(), value.strip()
                  if not key:
                      raise ValueError('empty header')
                  result[key] = result[key] + ', ' + value if key in result else value
              return result
          ''',
          '''
          def test_duplicate_case():
              assert parse_headers(['X-Tag: a', 'x-tag: b']) == {'x-tag': 'a, b'}
          def test_colon_value():
              assert parse_headers(['Time: 10:20']) == {'time': '10:20'}
          ''',
          '''
          def test_generator_and_blank():
              assert parse_headers(iter([' ', ' A : ', 'A: next'])) == {'a': ', next'}
          def test_malformed():
              for lines in (['oops'], [': value']):
                  with pytest.raises(ValueError):
                      parse_headers(lines)
          '''),
    _task("utc_timestamp", "date_time",
          "修复 as_utc(text)：读取 ISO 8601 时间，将时区感知时间转换到 UTC（保留瞬间与微秒），"
          "无时区时间按 UTC 解释；支持 Z 后缀。输出 datetime，tzinfo 为 timezone.utc。",
          '''
          from datetime import datetime, timezone

          def as_utc(text):
              parsed = datetime.fromisoformat(text.replace('Z', '+00:00'))
              return parsed.replace(tzinfo=timezone.utc)
          ''',
          '''
          from datetime import datetime, timezone

          def as_utc(text):
              parsed = datetime.fromisoformat(text.replace('Z', '+00:00'))
              if parsed.tzinfo is None:
                  return parsed.replace(tzinfo=timezone.utc)
              return parsed.astimezone(timezone.utc)
          ''',
          '''
          def test_positive_offset():
              assert as_utc('2024-01-02T01:00:00+02:00').isoformat() == '2024-01-01T23:00:00+00:00'
          def test_naive():
              assert as_utc('2024-01-01T12:00:00').hour == 12
          ''',
          '''
          def test_negative_offset_microseconds():
              assert as_utc('2024-01-01T23:30:00.123456-02:00').isoformat() == '2024-01-02T01:30:00.123456+00:00'
          def test_z_and_invalid():
              assert as_utc('2024-02-29T00:00:00Z').tzinfo is timezone.utc
              with pytest.raises(ValueError):
                  as_utc('not-a-date')
          '''),
    _task("calendar_days", "date_time",
          "修复 calendar_days(start, end)：输入 datetime.date，返回含首尾日期的每日 date 列表，"
          "正确跨月、跨年和闰年；start>end 返回空列表。",
          '''
          from datetime import timedelta

          def calendar_days(start, end):
              return [start + timedelta(days=offset) for offset in range(end.day - start.day + 1)]
          ''',
          '''
          from datetime import timedelta

          def calendar_days(start, end):
              return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]
          ''',
          '''
          from datetime import date
          def test_cross_month():
              assert calendar_days(date(2024, 1, 31), date(2024, 2, 2)) == [date(2024, 1, 31), date(2024, 2, 1), date(2024, 2, 2)]
          def test_same_day():
              assert calendar_days(date(2024, 1, 1), date(2024, 1, 1)) == [date(2024, 1, 1)]
          ''',
          '''
          from datetime import date
          def test_leap_and_year():
              assert len(calendar_days(date(2024, 2, 28), date(2024, 3, 1))) == 3
              assert len(calendar_days(date(2023, 12, 31), date(2024, 1, 1))) == 2
          def test_reverse():
              assert calendar_days(date(2024, 2, 1), date(2024, 1, 31)) == []
          '''),
    _task("clip_numeric", "numeric_intervals",
          "修复 clip(value, lower, upper)：将有限实数限制到闭区间，保留区间内原值。"
          "lower>upper 或任一输入为 NaN/无穷大时抛 ValueError；lower==upper 合法。",
          '''
          import math

          def clip(value, lower, upper):
              return min(lower, max(upper, value))
          ''',
          '''
          import math

          def clip(value, lower, upper):
              if lower > upper or not all(math.isfinite(item) for item in (value, lower, upper)):
                  raise ValueError('invalid interval or value')
              return max(lower, min(upper, value))
          ''',
          '''
          def test_inside():
              assert clip(4, 1, 8) == 4
          def test_outside():
              assert clip(-2, 1, 8) == 1
              assert clip(20, 1, 8) == 8
          ''',
          '''
          def test_degenerate_and_float():
              assert clip(9, 3, 3) == 3
              assert clip(0.5, 0.1, 0.9) == 0.5
          def test_invalid():
              for values in ((1, 3, 2), (float('nan'), 0, 1), (1, 0, float('inf'))):
                  with pytest.raises(ValueError):
                      clip(*values)
          '''),
    _task("interval_overlap", "numeric_intervals",
          "修复 overlaps(first, second)：两个二元素有限数值端点表示半开区间 [start,end)。"
          "仅存在正长度交集时返回 True；相接、空区间均不重叠。任一区间 start>end 抛 ValueError。",
          '''
          def overlaps(first, second):
              return max(first[0], second[0]) <= min(first[1], second[1])
          ''',
          '''
          def overlaps(first, second):
              if first[0] > first[1] or second[0] > second[1]:
                  raise ValueError('reversed interval')
              return max(first[0], second[0]) < min(first[1], second[1])
          ''',
          '''
          def test_touching():
              assert not overlaps((1, 3), (3, 5))
          def test_inside():
              assert overlaps((1, 5), (2, 4))
          ''',
          '''
          def test_empty_negative():
              assert not overlaps((2, 2), (1, 3))
              assert overlaps((-4, -1), (-2, 0))
          def test_reversed():
              with pytest.raises(ValueError):
                  overlaps((4, 2), (0, 9))
              with pytest.raises(ValueError):
                  overlaps((0, 9), (4, 2))
          '''),
    _task("invert_multimap", "collection_transform",
          "修复 invert_labels(mapping)：输入按插入顺序的 key→可迭代label 映射，输出 label→key列表。"
          "同一key重复label只计一次，多个key按输入顺序保留，不能修改输入；label与key均可哈希。",
          '''
          def invert_labels(mapping):
              result = {}
              for key, labels in mapping.items():
                  for label in labels:
                      result[label] = [key]
              return result
          ''',
          '''
          def invert_labels(mapping):
              result = {}
              for key, labels in mapping.items():
                  seen = set()
                  for label in labels:
                      if label not in seen:
                          result.setdefault(label, []).append(key)
                          seen.add(label)
              return result
          ''',
          '''
          def test_shared_label():
              assert invert_labels({'a': ['x'], 'b': ['x']}) == {'x': ['a', 'b']}
          def test_empty():
              assert invert_labels({}) == {}
          ''',
          '''
          def test_duplicates_and_input():
              data = {'a': ['x', 'x', 'y'], 'b': ['y']}
              assert invert_labels(data) == {'x': ['a'], 'y': ['a', 'b']}
              assert data['a'] == ['x', 'x', 'y']
          def test_iterators():
              assert invert_labels({1: iter(['z']), 2: []}) == {'z': [1]}
          '''),
    _task("count_deltas", "collection_transform",
          "修复 apply_deltas(counts, deltas)：返回新字典，对每个键累计有正有负的整数增量。"
          "缺失键从0开始；最终值为0的键移除，负值保留；不得修改输入counts或deltas。",
          '''
          def apply_deltas(counts, deltas):
              result = counts
              for key, amount in deltas:
                  result[key] = amount
              return result
          ''',
          '''
          def apply_deltas(counts, deltas):
              result = dict(counts)
              for key, amount in deltas:
                  result[key] = result.get(key, 0) + amount
              return {key: amount for key, amount in result.items() if amount != 0}
          ''',
          '''
          def test_accumulation():
              assert apply_deltas({'a': 2}, [('a', 3), ('a', -1)]) == {'a': 4}
          def test_zero():
              assert apply_deltas({'a': 2}, [('a', -2)]) == {}
          ''',
          '''
          def test_no_mutation():
              counts = {'a': 1, 'empty': 0}
              assert apply_deltas(counts, [('b', -3)]) == {'a': 1, 'b': -3}
              assert counts == {'a': 1, 'empty': 0}
          def test_stream():
              assert apply_deltas({}, iter([('x', 1), ('x', -1), ('y', 2)])) == {'y': 2}
          '''),
    _task("moving_sums", "sequence_windows",
          "修复 moving_sums(values, width)：返回所有连续完整窗口的和，包含最后窗口。"
          "values为数值序列；width须为正整数（拒绝bool），否则ValueError；不足一个窗口返回空列表。",
          '''
          def moving_sums(values, width):
              return [sum(values[start:start + width]) for start in range(len(values) - width)]
          ''',
          '''
          def moving_sums(values, width):
              if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
                  raise ValueError('width must be a positive integer')
              return [sum(values[start:start + width]) for start in range(len(values) - width + 1)]
          ''',
          '''
          def test_last_window():
              assert moving_sums([1, 2, 3, 4], 2) == [3, 5, 7]
          def test_short():
              assert moving_sums([1], 2) == []
          ''',
          '''
          def test_full_and_one():
              assert moving_sums([-1, 5, -2], 3) == [2]
              assert moving_sums((3, 4), 1) == [3, 4]
          def test_bad_width():
              for width in (0, -1, 1.5, True):
                  with pytest.raises(ValueError):
                      moving_sums([1, 2], width)
          '''),
    _task("centered_window", "sequence_windows",
          "修复 neighbors(values,index,radius)：返回索引index左右各radius个邻居，排除index本身，"
          "保持原顺序，边界截断而非负索引回绕。index越界抛IndexError，radius负数抛ValueError；"
          "index与radius保证为整数。不得修改values。",
          '''
          def neighbors(values, index, radius):
              return values[index - radius:index] + values[index + 1:index + radius + 1]
          ''',
          '''
          def neighbors(values, index, radius):
              if radius < 0:
                  raise ValueError('negative radius')
              if index < 0 or index >= len(values):
                  raise IndexError('index outside sequence')
              return values[max(0, index - radius):index] + values[index + 1:index + radius + 1]
          ''',
          '''
          def test_left_boundary():
              assert neighbors([10, 20, 30], 1, 3) == [10, 30]
          def test_middle():
              assert neighbors([0, 1, 2, 3, 4], 2, 1) == [1, 3]
          ''',
          '''
          def test_zero_and_right():
              assert neighbors([1, 2], 1, 0) == []
              assert neighbors([1, 2], 1, 5) == [1]
          def test_invalid():
              for index in (-1, 2):
                  with pytest.raises(IndexError):
                      neighbors([1, 2], index, 1)
              with pytest.raises(ValueError):
                  neighbors([1], 0, -1)
          '''),
    _task("relative_join", "path_handling",
          "修复 join_relative(base, child)：base与child为POSIX路径字符串，使用PurePosixPath返回字符串。"
          "child须为相对路径且任一路径分量不能是 '..'，否则ValueError；允许'.'或空child并规范化'.'和重复斜线。"
          "不可将纯子串如'file..txt'误判为上级目录。",
          '''
          from pathlib import PurePosixPath

          def join_relative(base, child):
              return str(PurePosixPath(base) / child)
          ''',
          '''
          from pathlib import PurePosixPath

          def join_relative(base, child):
              path = PurePosixPath(child)
              if path.is_absolute() or '..' in path.parts:
                  raise ValueError('child escapes base')
              return str(PurePosixPath(base) / path)
          ''',
          '''
          import pytest
          def test_absolute_rejected():
              with pytest.raises(ValueError):
                  join_relative('/srv/data', '/etc/passwd')
          def test_normal_child():
              assert join_relative('/srv/data', 'reports//today.txt') == '/srv/data/reports/today.txt'
          ''',
          '''
          def test_parent_rejected():
              for child in ('../x', 'safe/../../x', 'a/../b'):
                  with pytest.raises(ValueError):
                      join_relative('/base', child)
          def test_dot_and_substring():
              assert join_relative('/base', '') == '/base'
              assert join_relative('/base', './file..txt') == '/base/file..txt'
          '''),
    _task("file_extension", "path_handling",
          "修复 extension(path)：按POSIX路径最后一个文件名返回最后扩展名（不含点、casefold小写）；"
          "无扩展名、隐藏文件'.env'、末尾点'file.'都返回空串。父目录中的点不属于文件扩展名。",
          '''
          def extension(path):
              return path.split('.', 1)[-1].lower() if '.' in path else ''
          ''',
          '''
          from pathlib import PurePosixPath

          def extension(path):
              name = PurePosixPath(path).name
              suffix = PurePosixPath(name).suffix
              return suffix[1:].casefold() if suffix != '.' else ''
          ''',
          '''
          def test_final_suffix():
              assert extension('/data/archive.tar.GZ') == 'gz'
          def test_no_suffix():
              assert extension('README') == ''
          ''',
          '''
          def test_hidden_and_parent_dot():
              assert extension('/a.b/.env') == ''
              assert extension('/a.b/file') == ''
          def test_trailing_and_unicode():
              assert extension('file.') == ''
              assert extension('.config.JSON') == 'json'
          '''),
    _task("adjacent_pairs", "iterator_protocol",
          "修复 adjacent_pairs(iterable)：返回相邻元素二元组的迭代器，[(a,b),(b,c)]，"
          "支持只能遍历一次的生成器，无元素或单元素时为空；值为None也应正常处理。不得先把输入整体转成list。",
          '''
          def adjacent_pairs(iterable):
              iterator = iter(iterable)
              for previous in iterator:
                  try:
                      current = next(iterator)
                  except StopIteration:
                      return
                  yield previous, current
          ''',
          '''
          def adjacent_pairs(iterable):
              iterator = iter(iterable)
              sentinel = object()
              previous = next(iterator, sentinel)
              if previous is sentinel:
                  return
              for current in iterator:
                  yield previous, current
                  previous = current
          ''',
          '''
          def test_adjacent():
              assert list(adjacent_pairs([1, 2, 3, 4])) == [(1, 2), (2, 3), (3, 4)]
          def test_empty():
              assert list(adjacent_pairs([])) == []
          ''',
          '''
          def test_single_and_none():
              assert list(adjacent_pairs([1])) == []
              assert list(adjacent_pairs(iter([None, 1, None]))) == [(None, 1), (1, None)]
          def test_lazy():
              source = iter([10, 20, 30, 40])
              pairs = adjacent_pairs(source)
              assert next(pairs) == (10, 20)
              assert next(source) == 30
          '''),
    _task("take_prefix", "iterator_protocol",
          "修复 take_exactly(iterable,count)：返回最多count个元素的list，输入不足则返回全部。"
          "不能额外消费第count+1个元素；count=0不消费输入，count<0抛ValueError。count保证为整数。",
          '''
          def take_exactly(iterable, count):
              result = []
              for item in iterable:
                  if len(result) >= count:
                      break
                  result.append(item)
              return result
          ''',
          '''
          from itertools import islice

          def take_exactly(iterable, count):
              if count < 0:
                  raise ValueError('negative count')
              return list(islice(iterable, count))
          ''',
          '''
          def test_no_overread():
              source = iter([1, 2, 3])
              assert take_exactly(source, 2) == [1, 2]
              assert next(source) == 3
          def test_short_input():
              assert take_exactly([1], 4) == [1]
          ''',
          '''
          def test_zero_no_consumption():
              source = iter([8])
              assert take_exactly(source, 0) == []
              assert next(source) == 8
          def test_negative():
              with pytest.raises(ValueError):
                  take_exactly(iter([1]), -1)
          '''),
    _task("boolean_field", "record_validation",
          "修复 parse_boolean(value)：仅接受bool本身，整数0/1，去空白且大小写无关的字符串"
          "true/false/yes/no/1/0；返回bool。所有其他输入（包括None、float、空串、2）抛ValueError。",
          '''
          def parse_boolean(value):
              return bool(value)
          ''',
          '''
          def parse_boolean(value):
              if isinstance(value, bool):
                  return value
              if isinstance(value, int) and value in (0, 1):
                  return value == 1
              if isinstance(value, str):
                  normalized = value.strip().casefold()
                  if normalized in {'true', 'yes', '1'}:
                      return True
                  if normalized in {'false', 'no', '0'}:
                      return False
              raise ValueError('invalid boolean')
          ''',
          '''
          def test_false_string():
              assert parse_boolean('false') is False
          def test_true_string():
              assert parse_boolean(' YES ') is True
          ''',
          '''
          def test_supported_types():
              assert parse_boolean(False) is False
              assert parse_boolean(1) is True
              assert parse_boolean(' 0 ') is False
          def test_invalid():
              for value in (None, 0.0, 1.0, '', 2, [], 'maybe'):
                  with pytest.raises(ValueError):
                      parse_boolean(value)
          '''),
    _task("person_record", "record_validation",
          "修复 validate_person(record)：record为dict，name须为去空白后非空字符串，age须为0到130"
          "的整数且不能是bool。返回只含规范化name及原age的新dict；缺字段或类型/范围错误均ValueError；不改输入。",
          '''
          def validate_person(record):
              if not record.get('name') or not isinstance(record.get('age'), int):
                  raise ValueError('invalid person')
              return record
          ''',
          '''
          def validate_person(record):
              name = record.get('name')
              age = record.get('age')
              if not isinstance(name, str) or not name.strip():
                  raise ValueError('invalid name')
              if isinstance(age, bool) or not isinstance(age, int) or not 0 <= age <= 130:
                  raise ValueError('invalid age')
              return {'name': name.strip(), 'age': age}
          ''',
          '''
          def test_normalized_copy():
              assert validate_person({'name': ' Ada ', 'age': 30, 'extra': 1}) == {'name': 'Ada', 'age': 30}
          def test_zero_age():
              assert validate_person({'name': 'Baby', 'age': 0})['age'] == 0
          ''',
          '''
          def test_no_mutation_and_upper_bound():
              source = {'name': ' Old ', 'age': 130}
              result = validate_person(source)
              assert result == {'name': 'Old', 'age': 130}
              assert source['name'] == ' Old '
              assert result is not source
          def test_invalid_fields():
              for source in ({}, {'name': ' ', 'age': 2}, {'name': 9, 'age': 2}, {'name': 'A', 'age': True}, {'name': 'A', 'age': -1}, {'name': 'A', 'age': 131}):
                  with pytest.raises(ValueError):
                      validate_person(source)
          '''),
    _task("priority_sort", "stable_sorting",
          "修复 sort_jobs(jobs, priority)：每个job含kind字段，priority为无重复kind列表。"
          "按priority中顺序排序，未知kind排最后；同优先级保持输入顺序，返回新list，不修改输入。",
          '''
          def sort_jobs(jobs, priority):
              ranks = {kind: index for index, kind in enumerate(priority)}
              return sorted(jobs, key=lambda job: ranks.get(job['kind'], -1))
          ''',
          '''
          def sort_jobs(jobs, priority):
              ranks = {kind: index for index, kind in enumerate(priority)}
              return sorted(jobs, key=lambda job: ranks.get(job['kind'], len(priority)))
          ''',
          '''
          def test_unknown_last():
              jobs = [{'kind': 'other'}, {'kind': 'urgent'}]
              assert sort_jobs(jobs, ['urgent']) == [jobs[1], jobs[0]]
          def test_priority():
              jobs = [{'kind': 'low'}, {'kind': 'high'}]
              assert sort_jobs(jobs, ['high', 'low']) == [jobs[1], jobs[0]]
          ''',
          '''
          def test_stable_and_unchanged():
              jobs = [{'kind': 'x', 'id': 1}, {'kind': 'a', 'id': 2}, {'kind': 'x', 'id': 3}]
              assert [item['id'] for item in sort_jobs(jobs, ['a'])] == [2, 1, 3]
              assert [item['id'] for item in jobs] == [1, 2, 3]
          def test_empty_priority():
              jobs = [{'kind': 'b'}, {'kind': 'a'}]
              assert sort_jobs(jobs, []) == jobs
              assert sort_jobs([], ['a']) == []
          '''),
    _task("version_sort", "stable_sorting",
          "修复 sort_versions(versions)：版本是由点分隔的非负十进制整数字符串，按各段数值排序，"
          "缺段按0比较（1和1.0等价），等价项稳定保留原顺序；允许前导0，输入保证格式合法，返回新list。",
          '''
          def sort_versions(versions):
              return sorted(versions)
          ''',
          '''
          def sort_versions(versions):
              versions = list(versions)
              width = max((len(version.split('.')) for version in versions), default=0)
              def key(version):
                  parts = tuple(int(part) for part in version.split('.'))
                  return parts + (0,) * (width - len(parts))
              return sorted(versions, key=key)
          ''',
          '''
          def test_numeric_segments():
              assert sort_versions(['1.10', '1.2', '2']) == ['1.2', '1.10', '2']
          def test_empty():
              assert sort_versions([]) == []
          ''',
          '''
          def test_equivalent_stable():
              assert sort_versions(['1.0', '1', '1.0.0', '0.9']) == ['0.9', '1.0', '1', '1.0.0']
          def test_leading_zero_and_input():
              source = ['2.0', '01.002', '1.1']
              assert sort_versions(source) == ['1.1', '01.002', '2.0']
              assert source == ['2.0', '01.002', '1.1']
          '''),
    _task("json_lines", "serialization",
          "修复 encode_json_lines(records)：每个可JSON序列化对象写为一个JSON行，ensure_ascii=False，"
          "非空结果末尾有一个换行，空输入返回空串；对象字符串内换行须由JSON转义；支持一次性迭代器。",
          '''
          import json

          def encode_json_lines(records):
              return json.dumps(list(records), ensure_ascii=False) + '\\n'
          ''',
          '''
          import json

          def encode_json_lines(records):
              return ''.join(json.dumps(record, ensure_ascii=False) + '\\n' for record in records)
          ''',
          '''
          def test_individual_lines():
              output = encode_json_lines([{'a': 1}, {'b': 2}])
              assert [json.loads(line) for line in output.splitlines()] == [{'a': 1}, {'b': 2}]
          def test_empty():
              assert encode_json_lines([]) == ''
          ''',
          '''
          def test_unicode_and_embedded_newline():
              output = encode_json_lines([{'text': '你好\\n世界'}])
              assert len(output.splitlines()) == 1
              assert '你好' in output
              assert json.loads(output) == {'text': '你好\\n世界'}
          def test_stream_and_scalar():
              output = encode_json_lines(iter([None, 0, False]))
              assert output.endswith('\\n')
              assert [json.loads(line) for line in output.splitlines()] == [None, 0, False]
          '''),
    _task("binary_envelope", "serialization",
          "修复 pack_bytes(payload) 与 unpack_bytes(envelope)：payload为bytes；封装为dict含"
          "base64 ASCII字符串data和原始字节数size。解封返回bytes，非法base64或长度不符抛ValueError，支持空bytes与任意二进制。",
          '''
          import base64

          def pack_bytes(payload):
              encoded = base64.b64encode(payload).decode('ascii')
              return {'data': encoded, 'size': len(encoded)}

          def unpack_bytes(envelope):
              return base64.b64decode(envelope['data'])
          ''',
          '''
          import base64

          def pack_bytes(payload):
              encoded = base64.b64encode(payload).decode('ascii')
              return {'data': encoded, 'size': len(payload)}

          def unpack_bytes(envelope):
              try:
                  payload = base64.b64decode(envelope['data'], validate=True)
              except (ValueError, TypeError) as error:
                  raise ValueError('invalid base64') from error
              if len(payload) != envelope['size']:
                  raise ValueError('size mismatch')
              return payload
          ''',
          '''
          def test_byte_count():
              assert pack_bytes(b'abc') == {'data': 'YWJj', 'size': 3}
          def test_roundtrip():
              assert unpack_bytes(pack_bytes(b'hello')) == b'hello'
          ''',
          '''
          def test_empty_and_binary():
              for payload in (b'', bytes(range(256))):
                  envelope = pack_bytes(payload)
                  assert envelope['size'] == len(payload)
                  assert unpack_bytes(envelope) == payload
          def test_reject_invalid():
              for envelope in ({'data': '@@@', 'size': 0}, {'data': 'YQ==', 'size': 2}):
                  with pytest.raises(ValueError):
                      unpack_bytes(envelope)
          '''),
    _task("search_normalization", "text_normalization",
          "修复 normalize_search(text)：先Unicode NFC规范化，再casefold，最后把任意连续Unicode空白"
          "压为单个ASCII空格且去两端空白；空白输入返回空串，不能移除重音。",
          '''
          def normalize_search(text):
              return text.lower().strip()
          ''',
          '''
          import unicodedata

          def normalize_search(text):
              normalized = unicodedata.normalize('NFC', text).casefold()
              return ' '.join(normalized.split())
          ''',
          '''
          def test_casefold():
              assert normalize_search(' Straße ') == 'strasse'
          def test_whitespace():
              assert normalize_search(' A\\t B\\n C ') == 'a b c'
          ''',
          '''
          def test_composed_accents():
              assert normalize_search('Cafe\\u0301') == 'café'
              assert normalize_search('CAFÉ') == 'café'
          def test_unicode_space_and_empty():
              assert normalize_search('A\\u00a0\\u2003B') == 'a b'
              assert normalize_search(' \\t\\n') == ''
          '''),
    _task("newline_normalization", "text_normalization",
          "修复 normalize_newlines(text)：将CRLF与单独CR规范化为LF；保留所有其他字符、"
          "连续空行与末尾换行数量，不能额外添加或去除换行。",
          '''
          def normalize_newlines(text):
              return text.replace('\\r', '\\n')
          ''',
          '''
          def normalize_newlines(text):
              return text.replace('\\r\\n', '\\n').replace('\\r', '\\n')
          ''',
          '''
          def test_windows():
              assert normalize_newlines('a\\r\\nb\\r\\n') == 'a\\nb\\n'
          def test_old_mac():
              assert normalize_newlines('a\\rb') == 'a\\nb'
          ''',
          '''
          def test_mixed_and_empty_lines():
              assert normalize_newlines('a\\r\\n\\r\\nb\\rc\\n') == 'a\\n\\nb\\nc\\n'
          def test_empty_and_unchanged():
              assert normalize_newlines('') == ''
              assert normalize_newlines('  a\\n\\n') == '  a\\n\\n'
              assert normalize_newlines('a') == 'a'
          '''),
    _task("batch_outcomes", "batch_processing",
          "修复 run_batch(items, handler)：按输入顺序逐个调用handler恰好一次；每项结果为"
          "{'ok':True,'value':返回值}或异常时{'ok':False,'error':str(exception)}。"
          "捕获Exception后继续后续项；不吞KeyboardInterrupt等BaseException；支持生成器。",
          '''
          def run_batch(items, handler):
              results = []
              for item in items:
                  try:
                      results.append({'ok': True, 'value': handler(item)})
                  except Exception as error:
                      results.append({'ok': False, 'error': str(error)})
                      break
              return results
          ''',
          '''
          def run_batch(items, handler):
              results = []
              for item in items:
                  try:
                      results.append({'ok': True, 'value': handler(item)})
                  except Exception as error:
                      results.append({'ok': False, 'error': str(error)})
              return results
          ''',
          '''
          def test_continue_after_failure():
              def handler(value):
                  if value == 0:
                      raise ValueError('zero')
                  return value * 2
              assert run_batch([1, 0, 3], handler) == [{'ok': True, 'value': 2}, {'ok': False, 'error': 'zero'}, {'ok': True, 'value': 6}]
          def test_empty():
              assert run_batch([], str) == []
          ''',
          '''
          def test_call_order_once():
              seen = []
              def handler(value):
                  seen.append(value)
                  if value == 2:
                      raise ValueError('bad item')
                  return value
              assert run_batch(iter([1, 2, 3]), handler) == [
                  {'ok': True, 'value': 1},
                  {'ok': False, 'error': 'bad item'},
                  {'ok': True, 'value': 3},
              ]
              assert seen == [1, 2, 3]
          def test_base_exception():
              def handler(value):
                  raise KeyboardInterrupt()
              with pytest.raises(KeyboardInterrupt):
                  run_batch([1], handler)
          '''),
    _task("payload_batches", "batch_processing",
          "修复 payload_batches(items, limit)：按输入顺序把bytes组成list批次，每批总字节数不超过"
          "正整数limit，不拆分单条且尽量填满当前批；超限单条抛ValueError，非正limit抛ValueError。"
          "空输入无批次，空bytes也保留，不能输出空批次；支持一次性迭代器。",
          '''
          def payload_batches(items, limit):
              batches, current = [], []
              size = 0
              for item in items:
                  if size + len(item) >= limit:
                      batches.append(current)
                      current, size = [], 0
                  current.append(item)
                  size += len(item)
              if current:
                  batches.append(current)
              return batches
          ''',
          '''
          def payload_batches(items, limit):
              if limit <= 0:
                  raise ValueError('nonpositive limit')
              batches, current = [], []
              size = 0
              for item in items:
                  if len(item) > limit:
                      raise ValueError('item too large')
                  if current and size + len(item) > limit:
                      batches.append(current)
                      current, size = [], 0
                  current.append(item)
                  size += len(item)
              if current:
                  batches.append(current)
              return batches
          ''',
          '''
          def test_exact_fit():
              assert payload_batches([b'ab', b'c', b'd'], 3) == [[b'ab', b'c'], [b'd']]
          def test_empty():
              assert payload_batches([], 3) == []
          ''',
          '''
          def test_empty_payloads_and_stream():
              assert payload_batches(iter([b'', b'abc', b'', b'd']), 3) == [[b'', b'abc', b''], [b'd']]
          def test_invalid_sizes():
              with pytest.raises(ValueError):
                  payload_batches([b'abcd'], 3)
              with pytest.raises(ValueError):
                  payload_batches([], 0)
          '''),
]
