#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合成出题的流水线：向教师要题族、逐个审计，把接受的题族冻结划分后并入题池。

题池只有 327 道题、难题只有 20 个来源族，RFT 看不出提升，根子在题太少。这里把 agentlab.synth 的四个部件
（themes、prompts、parse、audit）串成一条流水线，一批一批地出题：一个主题（出题方向 × 具体领域）向教师要一个
题族，也就是一个程序——参考解、题面、公开测试、按标签的隐藏测试、若干 mutant。回复解析后在沙箱里真判分
（audit_family）；没有被接受、但发现的问题可以返修时，带着全部发现让教师改一轮，轮次用完还没接受就是拒绝。
接受的题族 = 母题 1 道（带着全部可用的缺陷）+ 每个可用的 mutant 一道单缺陷变体。

四个子命令（平时不单独运行，scripts/run_synth_pool.sh 把它们串成一条命令）：
  run     向教师要这一批的题族，逐个解析、审计、返修；每个题族的记录写在 <输出目录>/families/<slug>.json
  status  一共接受了多少、各方向多少、因为什么拒绝
  show    抽几个已接受的题族，打印题面和原程序到参考解的 diff，供人过目
  export  冻结划分，写出新题文件，并入阶梯文件

续跑：已经有定论的题族（接受了，或者因为题族本身被拒绝）不再请求。教师调用出错、回复被截断或者没法解析、
判分没能完成、中途停下的不算定论——那不是题族的问题，重跑时会再要一次，新记录覆盖旧的。

回复没法用时不直接判这一族拒绝，算一轮返修（还有轮次的话）：回复被截断（finish_reason 是 length，或者正文
是空的——思考类模型把输出全花在推理上时就是这样）的，带着"上次被截断"请它重新输出；正文有、可是不是合法的
题族 JSON 的，带着"上次不是合法的 JSON"和解析器的说明请它改。每一轮的原始回复、结束原因、用量和推理字段的
长度都记在记录里，事后查得到原因。教师请求出错由 _shared_teacher.chat_text 长时间重试（429、5xx），重试用完
才算这一轮出错。

停下的办法：第一次 Ctrl-C 不再请求新的，正在做的题族做完就退（退出码 130）；再按一次，或者 kill，立刻退
（正在做的那几族的回答不保存，重跑会再要）。退出码 3 表示一族都没接受、而且没有一族是因为题族本身被拒的
（教师接口出错、回复全都没法用……）：原因修好以后原样重跑，就是把这一批的主题再要一次。

教师接口的读法同 scripts/collect_data.py；--max-tokens、--min-interval、--temperature、--timeout 不给时取
环境变量 TEACHER_MAX_TOKENS、TEACHER_MIN_INTERVAL、TEACHER_TEMPERATURE、TEACHER_TIMEOUT（没有就是 12000、
1、0.8、300）。默认值取得保守（要慢、不要成片出错）：教师会先写一大段推理再给题族（写技能时上限 900 的回复
多半没写完），一个题族的 JSON 本身就有一两千 token，所以上限放到 12000；限额按账户算，所以 --concurrency
默认 2（也是同一时刻在飞的请求上限），两次请求之间至少隔 1 秒。
接口特有的参数写进环境变量 TEACHER_EXTRA_JSON（一个 JSON 对象），原样合并进每次请求。
"""
from __future__ import annotations

import argparse
import contextlib
import difflib
import itertools
import json
import os
import random
import shutil
import signal
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agentlab.benchmark.protocol import load_specs, validate_specs
from agentlab.sampling import vllm_client
from agentlab.synth import audit, parse, prompts, themes
from agentlab.synth.specs import family_specs
from agentlab.training.preparation import freeze_splits

ROUNDS, CONCURRENCY, FAMILIES = 2, 2, 50
# 教师先写一大段推理再给题族，一个题族的 JSON 本身就有一两千 token：给推理留足余量，宁可慢
MAX_TOKENS, TEMPERATURE, TIMEOUT = 12000, 0.8, 300
MIN_INTERVAL = 1.0             # 两次教师请求之间至少隔几秒（全进程共用，见 _shared_teacher.Throttle）
DEV_RATIO, TEST_RATIO = 0.2, 0.2
# 进度条 24 格，样子同 scripts/run_rft.sh 的 progress()：进度面板 scripts/show_progress.sh 靠 "] k/n (p%)" 认进度
BAR, DASH = "########################", "------------------------"
# 审计要临时改判分器的 pytest 超时（进程里的全局变量，见 audit.py 的说明）：同一个进程里不能同时审计两个题族。
# 所以审计串行进行，向教师的请求照旧并发——请求才是慢的那一头
AUDIT_LOCK = threading.Lock()
# 连续这么多个题族都因为判分环境出问题被拒，说明环境坏了：停下这一批，不再白白调用教师
ENVIRONMENT_STREAK = 3
# freeze_splits 至少要三个来源族才分得开 train / dev / test
MIN_FREEZE_GROUPS = 3
SPLITS = ("train", "dev", "test")

PARSE_ERROR, TRUNCATED, TEACHER_ERROR, CRASHED, INTERRUPTED = (
    "解析失败", "回复被截断", "教师调用出错", "处理时程序出错", "停下时还没做完")
# 回复没法用时带进返修提示的开头：教师看得出上一轮出的是哪一种事
CUT_SHORT, NOT_JSON = "上次被截断", "上次不是合法的 JSON"
GRADING_ERROR = "判分没能完成"
# 这些原因说的不是题族内容的问题：教师接口出错、回复被截断或者没法解析、判分没能完成、流水线自己出错、
# 中途停下。这样的题族没有定论——排主题时不算用过，重跑时会再要一次（新记录盖掉旧的）
NOT_ABOUT_CONTENT = frozenset({PARSE_ERROR, TRUNCATED, TEACHER_ERROR, GRADING_ERROR, CRASHED, INTERRUPTED})
# 给人看的时候按这个先后排
NOT_ABOUT_CONTENT_ORDER = (TEACHER_ERROR, TRUNCATED, PARSE_ERROR, GRADING_ERROR, CRASHED, INTERRUPTED)
# 环境坏了时才会连续出现的几种原因：连续 ENVIRONMENT_STREAK 个就停下这一批
ENVIRONMENT_ERRORS = frozenset({GRADING_ERROR, CRASHED})
# run_synth_pool.sh 调用 run 时设上它：脚本被 kill -9 时没人给 python 发信号，父进程一没就立刻停下
WATCH_PARENT_ENV = "SYNTH_POOL_STOP_WITH_PARENT"

# 审计发现项的 code（audit.py 的模块说明里列的全部 code）→ 给人看的拒绝原因
REASONS = {
    "bad_slug": "slug 不合规",
    "invalid_family": "题族构造不出题目",
    "syntax_error": "源码不是合法的 Python",
    "no_test_functions": "测试文件里没有 pytest 收得到的测试函数",
    "mutant_anchor": "mutant 的 old 片段定位不到",
    "mutant_syntax": "mutant 把程序改坏成不合法的 Python",
    "mutant_duplicate": "mutant 没有引入新的缺陷",
    "mutant_overlap": "mutant 之间改动重叠",
    "invalid_spec": "题目规格不合法",
    "collision": "和现有题池撞了",
    "reference_fails": "参考解不通过自己的测试",
    "reference_timeout": "参考解的测试超时",
    "reference_no_results": "参考解的测试没有留下结果",
    "grading_error": GRADING_ERROR,
    "unstable_tests": "测试不稳定",
    "mutant_escapes": "有 mutant 没被隐藏测试拦住",
    "mutant_breaks_public": "有 mutant 让公开测试失败",
    "mutant_not_failed": "有 mutant 只报错、没有被断言拦住",
    "mutant_timeout": "有 mutant 让测试超时",
    "mutant_ungraded": "有 mutant 没能判分",
    "parent_escapes": "母题没被隐藏测试拦住",
    "parent_breaks_public": "母题让公开测试失败",
    "parent_not_failed": "母题只报错、没有被断言拦住",
    "parent_timeout": "母题的测试超时",
    "parent_ungraded": "母题没能判分",
    "parent_broken": "母题构造失败",
    "too_few_mutants": "可用的 mutant 太少",
    "indistinct_tests": "隐藏测试区分不了各条要求",
    "fixed_lines": "参考解的行数不对",
    "too_long_tokens": "题目太长",
    "unsafe_code": "有不安全的写法",
    "hint_comment": "注释里有提示缺陷的标记",
}
# 没被接受时，按这个顺序找"决定性的那一条"：题族是因为它才没过关的。各个 mutant 自己的发现
# （没被拦住、让公开测试失败……）排在最后：它们的结果是"可用的 mutant 太少"
DECISIVE = (
    "grading_error", "collision", "bad_slug", "invalid_spec", "invalid_family", "syntax_error", "no_test_functions",
    "reference_fails", "reference_timeout", "reference_no_results", "unstable_tests", "too_few_mutants",
    "parent_broken", "parent_escapes", "parent_breaks_public", "parent_not_failed", "parent_timeout", "parent_ungraded",
    "fixed_lines", "too_long_tokens", "unsafe_code",
)


class ExportError(Exception):
    """导出做不下去。消息是给用户看的；抛出来的时候还没有写任何文件。"""


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def print_line(text: str) -> None:
    print(text, flush=True)


def warn_line(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def dump_json(value, *, indent=None, sort_keys=False) -> bytes:
    """写成 JSON 的字节。中文原样写出；孤立的代理项（回复里偶尔混进来的半截转义）写成转义，不然写不进文件。"""
    text = json.dumps(value, ensure_ascii=False, indent=indent, sort_keys=sort_keys)
    return text.encode("utf-8", "backslashreplace")


def write_atomic(path: Path, data: bytes) -> None:
    """先写临时文件再改名：写到一半被打断，也不会留下读不出来的半截文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def dump_spec(spec: dict) -> str:
    """题池文件里的一行：非 ASCII 字符一律写成转义，一行里只会有 \\n 这一种换行（按行切的脚本才切得开）。"""
    return json.dumps(spec, sort_keys=True) + "\n"


class Reply(str):
    """教师的原始回复。它就是一个字符串，所以注入的假 ask 直接给字符串也行；顺带带上这一次的用量、结束原因，
    和正文以外文字字段（推理）各有多长。"""

    usage: dict | None
    finish_reason: str | None
    other_text: dict

    def __new__(cls, text, usage=None, finish_reason=None, other_text=None):
        reply = super().__new__(cls, text)
        reply.usage, reply.finish_reason, reply.other_text = usage, finish_reason, other_text or {}
        return reply


def make_ask(client, model, chat):
    """把教师接口包成 ask(messages, max_tokens, temperature) -> str。只在 main 里构造；单元测试里注入假的。

    chat 是 _shared_teacher.chat_text（429、5xx 长时间重试，连不上、超时带退避重试，每次请求先过进程内的节流）；
    回复做成 Reply，顺带带上这一次的用量。重试用完它才抛，所以 ask 抛异常时这一轮才记成教师调用出错。
    """
    def ask(messages, max_tokens, temperature):
        result = chat(client, model, messages, max_tokens=max_tokens, temperature=temperature)
        return Reply(result["text"], result["usage"], result["finish_reason"], result.get("other_text"))

    return ask


class AlreadyRunning(RuntimeError):
    """同一个目录上已经有一个 run 在跑。"""


@contextlib.contextmanager
def one_at_a_time(output_dir):
    """同一个目录同一时间只跑一个 run：两个一起跑会把同一批主题向教师各要一遍，钱花两份，记录互相覆盖。

    锁由操作系统握着，进程一退出（包括被 kill）就放开，不会留下死锁。拿不到锁抛 AlreadyRunning。
    """
    try:
        import fcntl
    except ImportError:                            # 没有 flock 的平台：不拦
        yield
        return
    lock_path = Path(output_dir) / ".run.lock"
    handle = lock_path.open("a", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise AlreadyRunning(f"另一个 synth_pool.py run 正在 {output_dir} 上运行：等它跑完再来；"
                                 f"确认它已经停了（看日志和进度面板）才需要处理") from None
        except OSError as error:                   # 文件系统不支持 flock：不拦，说一声
            warn_line(f"{lock_path} 上加不了锁（{error}）：请确认没有另一个 run 在跑")
        yield
    finally:
        handle.close()


# ── 主题 ────────────────────────────────────────────────────────────────

def parse_directions(value: str) -> frozenset[str]:
    """--directions 的取值：all 是全部方向，否则是逗号分隔的方向短名。认不出的抛 ValueError。"""
    if value.strip() == "all":
        return frozenset(themes.DIRECTION_NAMES)
    names = frozenset(piece.strip() for piece in value.split(",") if piece.strip())
    known = "、".join(f"{name}（{themes.DIRECTIONS[name].label}）" for name in themes.DIRECTION_NAMES)
    if not names:
        raise ValueError(f"--directions 要写 all，或者逗号分隔的方向短名；可选的是 {known}")
    unknown = sorted(names - set(themes.DIRECTION_NAMES))
    if unknown:
        raise ValueError(f"未知的方向 {'、'.join(unknown)}；可选的是 {known}")
    return names


def _walk(start, directions, *, seed, skip=frozenset()):
    """从固定序列的第 start 个位置起，一个一个地给出属于这些方向、slug 不在 skip 里的主题，直到序列走完。

    只用 plan_themes 的公开接口：每批 1 个的第 start + 1 批，窗口起点就是位置 start，取到的就是这个位置上的
    那一个主题。
    """
    for offset in range(len(themes.DIRECTIONS) * len(themes.DOMAINS)):
        try:
            theme = themes.plan_themes(start + 1 + offset, 1, seed=seed)[0]
        except ValueError:
            return                                  # 序列走到头了
        if theme[0].name in directions and themes.slug_of(*theme) not in skip:
            yield theme


def planned_themes(batch, families, directions, *, seed, used_slugs=()) -> list:
    """这一批向教师要的主题，每个主题一个题族：从这一批的位置起，跳过已经有定论的 slug，凑够 families 个。

    所以没定论的（教师调用出错、回复没法用……）排在最前，后面接着新主题。directions 是全部方向时就是
    plan_themes 的第 batch 批；只要某几个方向时同样从这个位置往后找，只收这些方向的。凑不够时抛 ValueError。
    """
    if directions == frozenset(themes.DIRECTION_NAMES):
        return themes.plan_themes(batch, families, seed=seed, used_slugs=used_slugs)
    picked = list(itertools.islice(_walk((batch - 1) * families, directions, seed=seed, skip=frozenset(used_slugs)),
                                   families))
    if len(picked) < families:
        wanted = "、".join(sorted(directions))
        raise ValueError(f"第 {batch} 批凑不够 {families} 个 {wanted} 方向的主题：序列里只剩 {len(picked)} 个")
    return picked


# ── 题族的记录 ───────────────────────────────────────────────────────────

def families_dir(output_dir) -> Path:
    return Path(output_dir) / "families"


def family_path(output_dir, slug) -> Path:
    return families_dir(output_dir) / f"{slug}.json"


def is_accepted(record: dict) -> bool:
    return record["status"] == "accepted" and bool(record.get("family"))


def read_records(output_dir) -> list[dict]:
    """families 目录里的全部记录，按 slug 排序。读不出来的跳过，在标准错误里说一声（它会被当成没有记录）。"""
    records = []
    for path in sorted(families_dir(output_dir).glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict) or not {"slug", "status", "rounds"} <= record.keys():
                raise ValueError("不是一个题族的记录")
        except (OSError, ValueError) as error:
            warn_line(f"{path.name} 读不出来，跳过（{error}）：重跑时这一族会再要一次")
            continue
        records.append(record)
    return records


def write_record(output_dir, record: dict) -> None:
    write_atomic(family_path(output_dir, record["slug"]), dump_json(record, indent=2))


def settled(record: dict) -> bool:
    """这个题族有没有定论：接受了，或者因为题族本身被拒绝。教师调用出错、回复没法用、没能判分的不算。"""
    return is_accepted(record) or record.get("reason") not in NOT_ABOUT_CONTENT


def used_slugs(output_dir) -> set[str]:
    """已经有定论的题族的 slug：排下一批主题时跳过它们。没定论的重跑时会再要一次。"""
    return {record["slug"] for record in read_records(output_dir) if settled(record)}


def accepted_order(record: dict) -> tuple:
    """题族的先后：按方向的固定顺序，再按 slug。"""
    names = themes.DIRECTION_NAMES
    return (names.index(record["direction"]) if record["direction"] in names else len(names), record["slug"])


def accepted_records(output_dir) -> list[dict]:
    """接受了、并且带着题族的记录，按 accepted_order 排序。"""
    return sorted((record for record in read_records(output_dir) if is_accepted(record)), key=accepted_order)


def direction_label(name: str) -> str:
    return themes.DIRECTIONS[name].label if name in themes.DIRECTIONS else name


def reason_of(entry: dict) -> str | None:
    """一轮下来题族为什么不行：教师调用出错、回复被截断或者没法解析，或者审计发现里决定性的那一条。

    没有问题返回 None。回复被截断的（到了输出上限，或者正文是空的，见 cut_short）单独说：多半还要把
    --max-tokens 调大（经 run_synth_pool.sh 跑时是环境变量 TEACHER_MAX_TOKENS）。旧记录里没有 truncated
    这一项，所以也看 finish_reason。
    """
    if entry.get("error"):
        return TEACHER_ERROR
    if entry.get("parse_error"):
        return TRUNCATED if entry.get("truncated") or entry.get("finish_reason") == "length" else PARSE_ERROR
    codes = [finding.get("code") for finding in entry.get("findings") or []]
    for code in DECISIVE:
        if code in codes:
            return REASONS.get(code, code)
    return REASONS.get(codes[0], codes[0]) if codes else None


def final_reason(record: dict) -> str:
    """一个被拒绝的题族的拒绝原因：最后一个说明了问题的轮次。"""
    reason = next((text for text in map(reason_of, reversed(record["rounds"])) if text), None)
    return reason or "审计没有接受，也没有说明原因"


def blank(reply) -> bool:
    return not isinstance(reply, str) or not reply.strip()


def side_note(other_text) -> str:
    """正文以外的文字字段有多长，写进给人和教师看的话里："，推理等其它字段：reasoning_content 1234 个字符"。"""
    if not other_text:
        return ""
    return "，推理等其它字段：" + "、".join(f"{name} {chars} 个字符" for name, chars in other_text.items())


def cut_short(raw, finish_reason, other_text=None) -> str | None:
    """回复没写完就没了的时候，返回带进返修提示的一句话；不是这种情况返回 None。

    两种情况算：finish_reason 是 length（到了输出上限），或者正文是空的。思考类模型把输出全花在推理上时正文
    就是空的——推理在单独的字段里，这里只带上它有多长（``other_text``，见 _shared_teacher.other_text），不记内容。
    这种回复不成 JSON 不是"格式不对"，所以先于解析判断，报出来的也不是"找不到 JSON 对象"。
    """
    if finish_reason != "length" and not blank(raw):
        return None
    side = side_note(other_text)
    if finish_reason == "length":
        return (f"{CUT_SHORT}：回复在写完之前到了输出上限（finish_reason=length，正文 {len(raw)} 个字符{side}），"
                "JSON 没有写完。请重新输出完整的 JSON 对象，写紧凑一些，不要别的内容")
    return (f"{CUT_SHORT}：上一次的回复是空的（finish_reason={finish_reason}，正文 0 个字符{side}），没有可用的"
            "内容。请输出完整的 JSON 对象")


class Accepted:
    """到现在为止接受的题族：后来审计的题族不能和它们撞（task_id、原程序、参考解）。几个工作线程共用，读写都加锁。"""

    def __init__(self, records):
        self.lock = threading.Lock()
        kept = [record for record in records if is_accepted(record)]
        self.hashes = [record["fixed_hash"] for record in kept if record.get("fixed_hash")]
        self.specs = [spec for record in kept for spec in family_specs(record["family"])]

    def audit_kwargs(self) -> dict:
        with self.lock:
            return {"known_fixed_hashes": list(self.hashes), "extra_specs": list(self.specs)}

    def add(self, record: dict) -> None:
        with self.lock:
            if record.get("fixed_hash"):
                self.hashes.append(record["fixed_hash"])
            self.specs += family_specs(record["family"])


def process_family(theme, ask, *, batch, rounds=ROUNDS, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, example=None,
                   auditor=None, work_dir=None, repeats=2, accepted=None, stop=None) -> dict:
    """一个题族走完全部轮次：请求 → 解析 → 审计 →（没接受、有可返修的发现、还有轮次）返修。返回它的记录。

    ask(messages, max_tokens, temperature) 返回教师的原始回复，抛异常时这一轮记为错误、算一次轮次（重试在
    _shared_teacher.chat_text 里做，抛出来就是重试用完了）。回复被截断或者解析不了也算一轮：还有轮次的话
    下一轮是返修，带着"上次被截断"或"上次不是合法的 JSON"，不直接判这一族拒绝。
    auditor 默认是 audit.audit_family（在沙箱里真判分）；它改判分器的全局超时，所以要过 AUDIT_LOCK。
    accepted 是这一批到这时为止已接受的题族（Accepted）：新题族审计时不能和它们撞。
    stop 是这一批的停止开关（Event）：置位以后，正在做的题族做完手上的这一轮就停，记成"停下时还没做完"。
    """
    direction, domain = theme
    slug = themes.slug_of(direction, domain)
    auditor = auditor or audit.audit_family
    record = {"slug": slug, "direction": direction.name, "domain": domain.name, "batch": batch,
              "status": "rejected", "reason": None, "rounds": [], "family": None, "mutants": [],
              "fixed_hash": None, "created_at": now()}
    previous, findings = None, None
    for number in range(1, rounds + 1):
        if number > 1 and stop is not None and stop.is_set():
            record["reason"] = INTERRUPTED          # 这一批被停下了：不再为这一族花一轮，重跑时会再要
            break
        if number == 1 or previous is None:
            # 第一轮，或者上一轮连回复都没拿到（教师调用出错）：没有可改的东西，重新出题
            kind, messages = "generation", prompts.generation_messages(theme, example, slug)
        else:
            kind, messages = "repair", prompts.repair_messages(previous, findings, theme, slug)
        entry = {"round": number, "kind": kind, "raw": None, "finish_reason": None, "usage": None,
                 "other_text": None, "truncated": False, "parse_error": None, "findings": None, "stats": None,
                 "error": None}
        record["rounds"].append(entry)
        try:
            raw = ask(messages, max_tokens, temperature)
        except Exception as error:                 # 教师接口出错：记下，不让一个题族打断整批
            entry["error"] = f"{type(error).__name__}: {error}"
            previous, findings = None, None        # 没拿到回复：下一轮没有可改的东西，重新出题
            continue
        entry.update(raw=raw, finish_reason=getattr(raw, "finish_reason", None), usage=getattr(raw, "usage", None),
                     other_text=getattr(raw, "other_text", None))
        problem = cut_short(raw, entry["finish_reason"], entry["other_text"])
        entry["truncated"] = problem is not None
        if problem is None:
            try:
                parsed = parse.parse_family(raw, slug)
            except ValueError as error:            # 格式不对：解析器的话本来就是写给教师看的，原样带进返修提示
                problem = f"{NOT_JSON}：{error}"
        if problem is not None:                    # 这一轮的回复没法用：还有轮次的话，下一轮带着这句话返修
            entry["parse_error"] = problem
            previous, findings = raw, problem
            continue
        with AUDIT_LOCK:
            result = auditor(parsed, slug=slug, direction=direction.name, work_dir=Path(work_dir) / slug,
                             repeats=repeats, **(accepted.audit_kwargs() if accepted else {}))
            entry["findings"], entry["stats"] = result["findings"], result.get("stats")
            if result["accepted"]:
                family = result["family"]
                record.update(status="accepted", family=family,
                              mutants=[parsed["mutants"][index] for index in family["mutants"]],
                              fixed_hash=result.get("stats", {}).get("fixed_hash"))
                if accepted:                       # 趁还握着 AUDIT_LOCK 就登记：下一个题族审计时已经看得到它
                    accepted.add(record)
                break
        previous, findings = raw, result["findings"]
        if not any(finding.get("repairable") for finding in result["findings"]):
            break                                  # 没有可返修的发现（撞车、判分没能完成……）：返修解决不了
    if record["status"] != "accepted":
        record["reason"] = record.get("reason") or final_reason(record)
    return record


def crashed_record(theme, error, *, batch) -> dict:
    """处理一个题族时流水线自己出了错：记成拒绝（可以重跑补上），不让整批断在这里。"""
    direction, domain = theme
    return {"slug": themes.slug_of(direction, domain), "direction": direction.name, "domain": domain.name,
            "batch": batch, "status": "rejected", "reason": CRASHED, "rounds": [], "family": None,
            "error": f"{type(error).__name__}: {error}", "created_at": now()}


def progress_line(done, total, label, started) -> str:
    """一行进度，样子同 scripts/run_rft.sh 的 progress()。"""
    filled = 24 * done // total if total else 24
    percent = done * 100 // total if total else 100
    return (f"[{BAR[:filled]}{DASH[:24 - filled]}] {done}/{total} ({percent}%) {label} "
            f"· 已用 {int((time.monotonic() - started) // 60)}m")


def outcome_label(record: dict) -> str:
    """一行进度里进度条后面的说明：接受在第几轮、有几个变体；拒绝了是因为什么。"""
    if is_accepted(record):
        return (f"{record['slug']}: 接受（第 {record['rounds'][-1]['round']} 轮，"
                f"变体 {len(record['family']['variants'])} 个）")
    return f"{record['slug']}: 拒绝（{record['reason']}）"


def result_line(summary: dict) -> str:
    counted = sorted(summary["reasons"].items(), key=lambda item: (-item[1], item[0]))
    reasons = "、".join(f"{label} {count}" for label, count in counted) or "无"
    return (f"结果：第 {summary['batch']} 批 接受 {summary['accepted']} 族（母题 {summary['accepted']} 道、"
            f"变体 {summary['variants']} 道）、拒绝 {summary['rejected']} 族（原因：{reasons}）、"
            f"教师调用 {summary['calls']} 次、用时 {int(summary['seconds'] // 60)} 分钟")


def next_batch_hints(summary: dict) -> list[str]:
    """跑完以后怎么接着跑。

    没定论的题族（教师调用出错、回复被截断或没法解析、没能判分）不算用掉：重跑同一批时它们排在最前，后面
    接着新主题，一直到凑满这一批的个数。只补它们、不要新主题，就把 FAMILIES 写成它们的个数。
    """
    batch, lines = summary["batch"], []
    if summary.get("unwritten") or summary.get("stopped"):
        lines.append(f"修好以后原样重跑这一批（没做完的排在最前，再要满这一批的 {summary['planned']} 个）："
                     f"BATCH={batch} bash scripts/run_synth_pool.sh")
    elif summary["unsettled"]:
        lines.append(f"其中 {summary['unsettled']} 族没有定论，不算用掉。只补它们、不要新主题："
                     f"FAMILIES={summary['unsettled']} BATCH={batch} bash scripts/run_synth_pool.sh")
    lines.append(f"下一批这样跑：BATCH={batch + 1} bash scripts/run_synth_pool.sh")
    return lines


class BatchStopped(Exception):
    """这一批被第一次 Ctrl-C 打断：正在做的题族做完、记录落盘以后抛出，其余的主题留给重跑。"""

    def __init__(self, planned: int):
        super().__init__(f"{planned} 个主题")
        self.planned = planned


def run_batch(plan, ask, output_dir, *, batch, rounds=ROUNDS, concurrency=CONCURRENCY, max_tokens=MAX_TOKENS,
              temperature=TEMPERATURE, repeats=2, auditor=None, work_dir=None, examples=None, stop=None,
              out=print_line, warn=warn_line) -> dict:
    """这一批的主题各向教师要一个题族，结果写进 <输出目录>/families/，边跑边打印进度。返回这一批的汇总。

    plan 是 planned_themes 给的主题；examples 是 {slug: 风格参照}（不给就由提示自己挑）。每个题族的请求和审计
    都在同一个工作线程里，同时处理 concurrency 个；写盘和打印加锁。审计串行（AUDIT_LOCK），请求照旧并发。
    auditor 默认是真的 audit_family，测试里可以换成假的。

    stop 是这一批的停止开关（Event，不给就自己建一个）。置位以后排着的主题不再请求，正在做的题族做完手上的
    这一轮就停。连续 ENVIRONMENT_STREAK 个题族都因为判分环境出问题被拒，或者有记录写不进盘（磁盘满、
    目录只读——再调用教师也只是白花钱）时由这里置位；Ctrl-C、kill、脚本被强行结束时由 stop_signals 置位。
    """
    plan, started = list(plan), time.monotonic()
    examples, stop = examples or {}, stop or threading.Event()
    lock = threading.Lock()
    accepted = Accepted(read_records(output_dir))
    results, reasons, streak, skipped, unwritten = [], Counter(), 0, 0, []
    progress = {"done": 0}

    def say(line):                                 # 标准输出断了（日志的管道被关掉）不影响这一批
        try:
            out(line)
        except OSError:
            pass

    def tell(line):                                # 标准错误同理
        try:
            warn(line)
        except OSError:
            pass

    def worker(theme):
        if stop.is_set():
            return None                             # 这一批停下了：这一族不请求，留给重跑
        try:
            slug = themes.slug_of(*theme)
            record = process_family(theme, ask, batch=batch, rounds=rounds, max_tokens=max_tokens,
                                    temperature=temperature, example=examples.get(slug), auditor=auditor,
                                    work_dir=work_dir, repeats=repeats, accepted=accepted, stop=stop)
        except Exception as error:                 # 流水线自己的错：记下来，不让整批断在一个题族上
            record = crashed_record(theme, error, batch=batch)
            tell(f"处理 {record['slug']} 时流水线出了错（已记成拒绝，重跑会再要一次）：{record['error']}")
        try:
            with lock:                             # 结果加锁写盘、加锁打印：几路输出不会串行
                write_record(output_dir, record)
                progress["done"] += 1
                say(progress_line(progress["done"], len(plan), outcome_label(record), started))
        except OSError as error:                   # 写不进盘：这一族的回答丢了，后面的再要也只是白花钱
            stop.set()
            unwritten.append(f"{record['slug']}（{error}）")
        return record

    try:
        for _, record in vllm_client.map_concurrent(worker, plan, concurrency):
            if record is None:
                skipped += 1
                continue
            results.append(record)
            if is_accepted(record):
                streak = 0
            else:
                reasons[record["reason"]] += 1
                streak = streak + 1 if record["reason"] in ENVIRONMENT_ERRORS else 0
            if streak >= ENVIRONMENT_STREAK:
                stop.set()
    except KeyboardInterrupt:
        # 第一次 Ctrl-C（stop_signals 已经把开关置上）：正在做的几个题族做完、记录落盘以后才走到这里
        raise BatchStopped(len(plan)) from None

    kept = [record for record in results if is_accepted(record)]
    summary = {
        "batch": batch, "planned": len(plan), "accepted": len(kept),
        "variants": sum(len(record["family"]["variants"]) for record in kept),
        "rejected": len(results) - len(kept), "skipped": skipped,
        "calls": sum(len(record["rounds"]) for record in results),
        "errors": sum(1 for record in results for entry in record["rounds"] if entry.get("error")),
        "unsettled": sum(1 for record in results if not settled(record)),
        "reasons": dict(sorted(reasons.items())), "stopped": stop.is_set(), "unwritten": list(unwritten),
        "seconds": time.monotonic() - started,
    }
    say(result_line(summary))
    if unwritten:
        tell(f"有 {len(unwritten)} 族的记录写不进 {families_dir(output_dir)}：{unwritten[0]}。磁盘满了，还是目录"
             f"只读？这一批停下了，不再调用教师。腾出空间或者改好权限以后原样重跑——没写进盘的题族会再要一次。")
    elif summary["stopped"]:
        tell(f"连续 {ENVIRONMENT_STREAK} 个题族都因为{GRADING_ERROR}或者程序出错被拒，这一批停下了："
             f"判分环境多半有问题。先检查沙箱和判分环境（agentlab doctor），修好后原样重跑——"
             f"这些题族会再要一次。")
    for line in next_batch_hints(summary):
        say(line)
    return summary


# ── status ──────────────────────────────────────────────────────────────

def status_lines(output_dir) -> list[str]:
    records = read_records(output_dir)
    if not records:
        return [f"结果：还没有题族记录（{families_dir(output_dir)} 里没有文件）："
                "先跑一批，bash scripts/run_synth_pool.sh"]
    kept = [record for record in records if is_accepted(record)]
    lost = [record for record in records if not is_accepted(record)]
    by_direction = Counter(record["direction"] for record in records)
    good = Counter(record["direction"] for record in kept)
    order = [name for name in themes.DIRECTION_NAMES if name in by_direction]
    order += sorted(set(by_direction) - set(themes.DIRECTION_NAMES))
    directions = "、".join(f"{direction_label(name)} {good[name]}/{by_direction[name]}" for name in order)
    variants = sum(len(record['family']['variants']) for record in kept)
    lines = [f"结果：共接受 {len(kept)} 族（母题 {len(kept)}、变体 {variants}）、拒绝 {len(lost)} 族；"
             f"各方向 {directions}"]
    if lost:
        counted = Counter(record.get("reason") or "未知原因" for record in lost)
        lines.append("拒绝原因：" + "、".join(f"{label} {count}" for label, count in counted.most_common()))
    retry = sorted(record["slug"] for record in lost if not settled(record))
    if retry:
        shown = "、".join(retry[:8]) + ("…" if len(retry) > 8 else "")
        lines.append(f"其中 {len(retry)} 族没有定论（教师调用出错、回复被截断或没法解析、没能判分），不是题族"
                     f"的问题、也不算用掉：重跑它们所在的那一批时排在最前（{shown}）")
    return lines


# ── show ────────────────────────────────────────────────────────────────

def unified_diff(spec: dict) -> str:
    """原程序 → 参考解的 diff。"""
    return "".join(difflib.unified_diff(
        spec["buggy"].splitlines(keepends=True), spec["fixed"].splitlines(keepends=True),
        fromfile=f"a/{spec['module']}", tofile=f"b/{spec['module']}"))


def show_lines(output_dir, sample=3, seed=42) -> list[str]:
    """抽查几个已接受的题族：题面、隐藏测试文件名、母题和一个变体各自的 diff。"""
    records = accepted_records(output_dir)
    if not records:
        return [f"结果：{families_dir(output_dir)} 里还没有已接受的题族，没有可看的"]
    picked = random.Random(seed).sample(records, min(sample, len(records)))
    picked = sorted(picked, key=lambda record: record["slug"])
    lines = []
    for number, record in enumerate(picked, 1):
        family, parent = record["family"], record["family"]["parent"]
        lines += [f"── {number}/{len(picked)} {record['slug']}（{direction_label(record['direction'])} · "
                  f"{record['domain']}；第 {record['rounds'][-1]['round']} 轮接受，母题 1 道、变体 "
                  f"{len(family['variants'])} 道）──",
                  "题面：", parent["instruction"],
                  "隐藏测试文件：" + "、".join(sorted(parent["private_tests"])),
                  f"母题 {parent['task_id']} 的原程序 → 参考解：", unified_diff(parent).rstrip("\n")]
        if family["variants"]:
            idea = (record.get("mutants") or [{}])[0].get("idea", "")
            what = f"（这处缺陷：{idea}）" if idea else ""
            lines += [f"变体 {family['variants'][0]['task_id']} 的原程序 → 参考解{what}：",
                      unified_diff(family["variants"][0]).rstrip("\n")]
    lines.append(f"结果：抽查 {len(picked)} 个已接受的题族（一共 {len(records)} 个；换个样本：--seed 另给一个数）")
    return lines


# ── export ──────────────────────────────────────────────────────────────

def load_splits(path: Path) -> dict:
    """splits.json 的全部内容：source_groups（来源组 → 划分）和 history（每次冻结的记录）。文件不存在就是空文档。"""
    if not path.exists():
        return {"schema_version": 1, "source_groups": {}, "history": []}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        groups = document["source_groups"]
        if not isinstance(groups, dict) or not all(
                isinstance(group, str) and split in SPLITS for group, split in groups.items()):
            raise TypeError("source_groups 不是 {来源组: 划分}")
        document["history"] = list(document.get("history", []))
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ExportError(f"{path} 读不出来（{error}）：它记着已经冻结的划分，不能丢。从备份恢复一份；"
                          f"没有备份就只能删掉它重新冻结，已接受题族的划分会变") from error
    return document


def read_ladder(path) -> tuple[list[str], list[dict]]:
    """阶梯文件：一行一道题的题池文件。返回原来的各行（不带换行）和解析出来的题。"""
    try:
        lines = [line for line in Path(path).read_text(encoding="utf-8").split("\n") if line.strip()]
        specs = [json.loads(line) for line in lines]
        validate_specs(specs)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ExportError(f"阶梯文件 {path} 读不出来，或者不是一行一道题的题池文件（{error}）") from error
    return lines, specs


def groups_crossing_splits(specs) -> list[str]:
    """同一个来源组的题落在了不同划分里的那些来源组。"""
    seen, crossed = {}, set()
    for spec in specs:
        group = spec["source_group"]
        if seen.setdefault(group, spec["split"]) != spec["split"]:
            crossed.add(group)
    return sorted(crossed)


def verify_written(path, label: str) -> int:
    """写完以后读回来核对：每道题都过 validate_specs，没有来源组跨划分。"""
    try:
        specs = load_specs(path)
        crossing = groups_crossing_splits(specs)
    except (OSError, ValueError) as error:
        raise ExportError(f"{label} {path} 写进去以后读回来有问题：{error}") from error
    if crossing:
        raise ExportError(f"{label} {path} 里有来源组跨了划分：{'、'.join(crossing)}")
    return len(specs)


def audit_document(records, splits: dict[str, str]) -> dict:
    """每个题族请求了几轮、调用教师几次；每道题的审计结论（接受那一轮的）。"""
    families, tasks = {}, {}
    for record in records:
        entry = next((item for item in reversed(record["rounds"]) if item.get("findings") is not None), {})
        stats, findings, slug = entry.get("stats") or {}, entry.get("findings") or [], record["slug"]
        usage = [item.get("usage") or {} for item in record["rounds"]]
        families[slug] = {
            "direction": record["direction"], "domain": record["domain"], "batch": record.get("batch"),
            "accepted_in_round": entry.get("round"), "rounds": len(record["rounds"]),
            "teacher_calls": len(record["rounds"]),
            "tokens": {name: sum(item.get(name) or 0 for item in usage)
                       for name in ("input_tokens", "output_tokens")},
            "fixed_lines": stats.get("fixed_lines"), "total_tokens": stats.get("total_tokens"),
            "isolation": stats.get("isolation"), "checks": stats.get("checks"), "findings": findings,
        }
        family = record["family"]
        parent, split = family["parent"], splits[family["parent"]["source_group"]]
        tasks[parent["task_id"]] = {
            "family": slug, "kind": "parent", "split": split, "checks": stats.get("checks"),
            "fixed_lines": stats.get("fixed_lines"), "total_tokens": stats.get("total_tokens"),
            "isolation": stats.get("isolation"), "reference": stats.get("reference"), "parent": stats.get("parent"),
            "findings": findings,
        }
        for variant, index in zip(family["variants"], family["mutants"]):
            graded = next((item for item in stats.get("variants", []) if item.get("index") == index), {})
            tasks[variant["task_id"]] = {
                "family": slug, "kind": "variant", "variant_of": parent["task_id"], "split": split,
                "mutant": index + 1, "requirement": graded.get("requirement"),
                "caught_by": graded.get("failed_tests", []),
            }
    return {"schema_version": 1, "created_at": now(), "families": families, "tasks": tasks}


def audit_path_for(tasks_out: Path) -> Path:
    """审计结论文件：<tasks-out 去掉 .jsonl>.audit.json。"""
    stem = tasks_out.name[:-len(".jsonl")] if tasks_out.name.endswith(".jsonl") else tasks_out.name
    return tasks_out.with_name(stem + ".audit.json")


def text_of(path: Path) -> str | None:
    """文件里的文字；读不出来（不存在、不是 UTF-8）返回 None。"""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def earlier_export(existing: str, carried: str, added: list[str]) -> bool:
    """阶梯输出文件里现在的内容，是不是我们自己以前导出的一个旧版本。

    是的条件：阶梯原来的行原样在最前；后面每一行都是这一次也要写进去的新题，先后也一致，只是少了一些
    （还没有后面几批）。这样的文件更新成新的内容不会丢任何一行，所以可以覆盖。别的文件一律不碰。
    """
    if not existing.startswith(carried):
        return False
    position = 0
    for line in existing[len(carried):].splitlines(keepends=True):
        try:
            position = added.index(line, position) + 1
        except ValueError:
            return False
    return True


def nothing_to_export(output_dir, records) -> str:
    """一篇已接受的题族都没有时说明为什么、接着怎么办。"""
    if not records:
        return f"{output_dir} 里还没有题族记录，没有可导出的题：先跑一批（bash scripts/run_synth_pool.sh）"
    unsettled = sorted(record["slug"] for record in records if not settled(record))
    if unsettled:
        advice = (f"；其中 {len(unsettled)} 族没有定论（不是题族的问题），照 run 上次打印的提示把它们补做"
                  f"（FAMILIES 写成那一批里这样的族数）再导出")
    else:
        advice = "；都是因为题族本身被拒绝的，换一批主题再跑（BATCH 写成下一批的批号）"
    return (f"{output_dir} 里没有已接受的题族（{len(records)} 份记录都被拒绝了），没有可导出的题。"
            f"各是为什么看 python scripts/synth_pool.py status --output-dir {output_dir}{advice}")


def split_reminder(groups: Counter, dev_ratio: float, test_ratio: float) -> str | None:
    """已经冻结的划分离要的比例太远时提醒一句（早先导出的族太少，dev、test 分不出来）。

    groups 是各划分里的来源族个数。dev 或者 test 的占比不到要的一半才提醒。
    """
    total = sum(groups.values())
    if not total:
        return None
    share = {"dev": groups["dev"] / total, "test": groups["test"] / total}
    wanted = {"dev": dev_ratio, "test": test_ratio}
    if not any(share[split] < wanted[split] / 2 for split in share):
        return None
    return (f"提醒：{total} 个来源族里 dev 占 {share['dev']:.0%}、test 占 {share['test']:.0%}，比要的 "
            f"{dev_ratio:.0%}、{test_ratio:.0%} 少得多。冻结过的族划分不再改（加了批次，旧族也不动），早先导出的"
            f"那几族留在 train 里，比例偏了以后只能靠新批次慢慢往回拉。想一次分得准，等接受的族多一些"
            f"（每个方向五六族以上）再导出")


def freeze_new(fresh: list, new_tasks: list, seed, dev_ratio, test_ratio) -> tuple[dict, str | None]:
    """还没冻结过的题族怎么分。返回 (来源组 → 划分, 要提醒的一句话；不用提醒就是 None)。

    不少于 MIN_FREEZE_GROUPS 族时交给 freeze_splits（按 stratum 分层，只传这些族的新题）。更少时 freeze_splits
    不肯分——它要至少三个来源族；而这么几族本来就分不出 dev 和 test（每个方向至少留一族在 train），所以都记在
    train 里，以后也不再改（冻结过的族的划分不变）。不为这个让整次导出失败：那几族的题照样能用。
    """
    if not 0 < dev_ratio < 1 or not 0 < test_ratio < 1 or dev_ratio + test_ratio >= 1:
        raise ExportError(f"dev 和 test 的比例都要在 0 和 1 之间、加起来不到 1：现在是 {dev_ratio:g} 和 {test_ratio:g}")
    if len(fresh) >= MIN_FREEZE_GROUPS:
        try:
            _, manifest = freeze_splits(new_tasks, seed, dev_ratio, test_ratio, stratify=True)
        except ValueError as error:
            raise ExportError(f"冻结 {len(fresh)} 个新题族的划分失败：{error}") from error
        return dict(manifest["source_groups"]), None
    note = (f"新接受的题族只有 {len(fresh)} 个：freeze_splits 至少要 {MIN_FREEZE_GROUPS} 个来源族才分得开，这么几族"
            f"也分不出 dev 和 test，都记在 train 里（以后不再改）")
    return {record["family"]["parent"]["source_group"]: "train" for record in fresh}, note


def export_pool(output_dir, ladder, tasks_out, ladder_out, *, dev_ratio=DEV_RATIO, test_ratio=TEST_RATIO, seed=42,
                out=print_line) -> dict:
    """把已接受的题族冻结划分，写出题池文件，并入阶梯。做不下去时抛 ExportError，这时还没有写任何文件。

    只冻结还没冻结过的题族（freeze_new）；冻结过的沿用 splits.json 里的划分，所以加了新批次以后旧族的划分不变。
    全部检查通过才写文件，写完再读回来核对一遍。
    """
    output_dir, tasks_out, ladder_out = Path(output_dir), Path(tasks_out), Path(ladder_out) if ladder_out else None
    everything = read_records(output_dir)
    records = sorted((record for record in everything if is_accepted(record)), key=accepted_order)
    if not records:
        raise ExportError(nothing_to_export(output_dir, everything))
    splits_path = output_dir / "splits.json"
    document = load_splits(splits_path)
    splits = document["source_groups"]
    fresh = [record for record in records if record["family"]["parent"]["source_group"] not in splits]
    new_tasks = [spec for record in fresh for spec in family_specs(record["family"])]
    small = None
    if fresh:
        try:
            validate_specs(new_tasks)
        except ValueError as error:
            raise ExportError(f"新接受的 {len(fresh)} 个题族里有不合法的题目：{error}") from error
        frozen, small = freeze_new(fresh, new_tasks, seed, dev_ratio, test_ratio)
        splits.update(frozen)
        document["history"].append({
            "created_at": now(), "families": len(fresh), "seed": seed, "dev_ratio": dev_ratio, "test_ratio": test_ratio,
            "task_counts": dict(Counter(spec["split"] for spec in new_tasks)),
            "how": "freeze_splits" if small is None else "train_only",
        })
    tasks = [{**spec, "split": splits[spec["source_group"]]}
             for record in records for spec in family_specs(record["family"])]
    validate_specs(tasks)
    audit_path = audit_path_for(tasks_out)

    # 阶梯：原有的题原样保留，新题加在后面。阶梯里已经有同一道题（按 task_id 认、内容相同）的就不再加一遍
    merged, note, repeated, added = None, None, 0, []
    if ladder and Path(ladder).exists():
        if ladder_out is None:
            raise ExportError(f"给了 --ladder {ladder} 就要给 --ladder-out（阶梯加上新题以后写到哪）")
        if os.path.realpath(ladder) == os.path.realpath(ladder_out):
            raise ExportError(f"阶梯文件和输出是同一个文件（{ladder}）：输出必须是另一个文件，阶梯本身不动")
        lines, existing = read_ladder(ladder)
        known = {spec["task_id"]: spec for spec in existing}
        for spec in tasks:
            old = known.get(spec["task_id"])
            if old is None:
                added.append(spec)
            elif old != spec:
                raise ExportError(f"阶梯文件 {ladder} 里已经有 {spec['task_id']}，但和要加的这道题不一样"
                                  f"（划分或者内容不同）：先把阶梯里的旧版本处理掉再导出")
            else:
                repeated += 1
        # 并进阶梯以后的整份题池也要合法：阶梯里已有的来源组不能和新题跨划分
        try:
            validate_specs([*existing, *added])
        except ValueError as error:
            raise ExportError(f"新题并进阶梯 {ladder} 以后不合法：{error}") from error
        crossing = groups_crossing_splits([*existing, *added])
        if crossing:
            raise ExportError(f"新题并进阶梯 {ladder} 以后有来源组跨了划分：{'、'.join(crossing)}")
        carried = "".join(line + "\n" for line in lines)
        added_lines = [dump_spec(spec) for spec in added]
        payload = carried + "".join(added_lines)
        current = text_of(ladder_out) if ladder_out.exists() else None
        if current == payload:
            note = f"阶梯文件 {ladder_out} 已经是这个内容，跳过"
        elif current is not None and not earlier_export(current, carried, added_lines):
            # 不是我们自己以前导出的旧版本：里面有的行是这次不会写进去的，覆盖就会丢东西
            raise ExportError(f"{ladder_out} 已经存在，内容和将要写的不一样，也不是以前导出的版本，不覆盖："
                              f"换一个还没有的文件名再导出（LADDER_OUT=…），或者确认不要它了先删掉")
        else:
            merged = payload.encode("utf-8")
            if current is None:
                note = f"并入阶梯：{ladder_out} = {ladder}（原有的 {len(existing)} 道原样保留）+ 新题 {len(added)} 道"
            else:
                note = (f"阶梯文件 {ladder_out} 是以前导出的（只有前几批的题），更新成现在的内容："
                        f"{ladder} 原有的 {len(lines)} 道原样保留，加新题 {len(added)} 道")
    elif ladder:
        note = (f"阶梯文件 {ladder} 不存在：只导出新题，没有写 {ladder_out or '阶梯文件（没有给 --ladder-out）'}。"
                f"要并入阶梯时把它拷到这个路径再运行 export")

    # 检查都过了才写文件；写完再读回来核对一遍
    if fresh:
        write_atomic(splits_path, dump_json(document, indent=2, sort_keys=True))
    write_atomic(tasks_out, "".join(dump_spec(spec) for spec in tasks).encode("utf-8"))
    write_atomic(audit_path, dump_json(audit_document(records, splits), indent=2, sort_keys=True) + b"\n")
    if merged is not None:
        write_atomic(ladder_out, merged)
        verify_written(ladder_out, "阶梯文件")

    counted = Counter(spec["split"] for spec in tasks)
    groups = Counter(splits.values())
    verified = verify_written(tasks_out, "新题文件")
    out(f"结果：导出 {verified} 道合成题（母题 {len(records)} 道、变体 {len(tasks) - len(records)} 道；"
        f"这次新冻结 {len(fresh)} 族）：train {counted['train']} 道、dev {counted['dev']} 道、"
        f"test {counted['test']} 道；读回来核对过，没有来源组跨划分")
    out(f"新题文件：{tasks_out}；每道题的审计结论：{audit_path.name}；划分：{splits_path}"
        f"（{len(splits)} 个来源族：train {groups['train']}、dev {groups['dev']}、test {groups['test']}）")
    reminder = split_reminder(groups, dev_ratio, test_ratio)
    if reminder:
        out(reminder)
    if small:
        out(small)
    if repeated:
        out(f"阶梯里已经有 {repeated} 道合成题，不再加一遍")
    if note:
        out(note)
    return {"tasks": verified, "families": len(records), "fresh": len(fresh), "splits": counted, "ladder": note,
            "small": small}


# ── 停下的办法 ──────────────────────────────────────────────────────────

@contextlib.contextmanager
def stop_signals(stop: threading.Event, work_dir):
    """run 跑着的时候让 Ctrl-C 和 kill 停得下来，用完把原来的信号处理器装回去。

    不装处理器的话，ThreadPoolExecutor 要等排着的主题全做完才退出：按一次 Ctrl-C，剩下的教师请求照样发出去。
      * 第一次 Ctrl-C：把开关置上（排着的主题不再请求），在主线程里打断；正在做的几个题族做完手上的这一轮、
        记录落盘以后，run 以 130 退出；
      * 再按一次 Ctrl-C，或者 kill（SIGTERM）、终端关闭（SIGHUP）：立刻退出，正在做的几个题族这一次的回答
        不保存（重跑会再要），审计用的临时目录顺手删掉；
      * run_synth_pool.sh 调用 run 时会设环境变量 WATCH_PARENT_ENV：脚本被 kill -9 时没人给 python 发信号，
      所以另有一个线程盯着父进程，父进程一没就照 kill 处理，不让 python 成了孤儿把这一批要完。
    """
    state = {"interrupts": 0}
    previous, finished = {}, threading.Event()

    def leave(signum, why):
        # 不走解释器正常的收尾：那会等正在跑的那几个线程做完
        warn_line(f"{why}：立刻停下。排着的主题没有请求；正在进行的几个题族这一次的回答没有保存，重跑会再要。")
        shutil.rmtree(work_dir, ignore_errors=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(128 + signum)

    def on_signal(signum, frame):
        stop.set()                                  # 排着的主题不再请求
        if signum != signal.SIGINT or state["interrupts"]:
            leave(signum, "收到 kill 信号" if signum == signal.SIGTERM else "终端关了")
        state["interrupts"] += 1
        raise KeyboardInterrupt()                   # 第一次 Ctrl-C：在主线程里打断，正在做的题族做完再退

    def watch_parent():
        original = os.getppid()
        while not finished.wait(2.0):
            if os.getppid() != original:
                leave(signal.SIGTERM, "启动这个进程的脚本已经没了（被强行结束）")

    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        number = getattr(signal, name, None)
        try:                                  # 信号只能在主线程里接：不在主线程（或者平台没有这个信号）就不接
            if number is not None:
                previous[number] = signal.signal(number, on_signal)
        except (ValueError, OSError):
            pass
    watcher = None
    if os.environ.get(WATCH_PARENT_ENV) == "1":
        watcher = threading.Thread(target=watch_parent, name="synth-pool-parent", daemon=True)
        watcher.start()
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
        finished.set()
        if watcher:
            watcher.join(timeout=5)


# ── 命令行 ──────────────────────────────────────────────────────────────

def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("至少是 1")
    return number


def teacher_default(name: str, kind: type, fallback, what: str):
    """教师接口的参数在命令行没给时的默认值：环境变量 TEACHER_<名字>，没有就是 fallback。

    读法同 scripts/write_skills.py：经 run_synth_pool.sh 跑时没有命令行可加，只能靠这些环境变量调（.env 里
    写上就行）。写得不是数抛 ValueError。
    """
    raw = os.environ.get(f"TEACHER_{name.upper()}")
    if raw is None or not raw.strip():
        return fallback
    try:
        return kind(raw.strip())
    except ValueError:
        raise ValueError(f"环境变量 TEACHER_{name.upper()} 要是{what}，现在是 {raw!r}") from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    try:
        max_tokens = teacher_default("max_tokens", int, MAX_TOKENS, "整数")
        min_interval = teacher_default("min_interval", float, MIN_INTERVAL, "非负数")
        temperature = teacher_default("temperature", float, TEMPERATURE, "数字")
        timeout = teacher_default("timeout", float, TIMEOUT, "数字")
    except ValueError as error:
        parser.error(str(error))
    if max_tokens < 1 or temperature <= 0 or timeout <= 0:
        parser.error("TEACHER_MAX_TOKENS、TEACHER_TEMPERATURE、TEACHER_TIMEOUT 都必须是正数")
    if min_interval < 0:
        parser.error("TEACHER_MIN_INTERVAL 不能是负数")
    commands = parser.add_subparsers(dest="command", required=True)

    def output(option):
        option.add_argument("--output-dir", required=True, help="合成题族的目录（默认 data/synth-pool）")

    run = commands.add_parser("run", help="向教师要这一批的题族，逐个解析、审计、返修")
    output(run)
    run.add_argument("--batch", type=positive, default=1, help="第几批（从 1 起；默认 1）")
    run.add_argument("--families", type=positive, default=FAMILIES, help="这一批向教师要多少个题族（默认 50）")
    run.add_argument("--directions", default="all", help="出题方向：all（默认），或者逗号分隔的方向短名")
    run.add_argument("--rounds", type=positive, default=ROUNDS, help="每个题族最多请求几次，含首次（默认 2）")
    run.add_argument("--concurrency", type=positive, default=CONCURRENCY,
                     help="同时处理几个题族，也就是同一时刻最多有几个教师请求在飞（默认 2）")
    run.add_argument("--max-tokens", type=positive, default=max_tokens,
                     help=f"教师每次回复的输出上限（默认 {MAX_TOKENS}；环境变量 TEACHER_MAX_TOKENS）")
    run.add_argument("--min-interval", type=float, default=min_interval,
                     help="两次教师请求之间至少隔几秒，全进程共用（默认 1；环境变量 TEACHER_MIN_INTERVAL）")
    run.add_argument("--temperature", type=float, default=temperature,
                     help="教师接口的温度（默认 0.8；环境变量 TEACHER_TEMPERATURE）")
    run.add_argument("--seed", type=int, default=42, help="决定主题的先后（默认 42）")
    run.add_argument("--model", default=None, help="教师模型名；不填时用环境变量 TEACHER_MODEL")
    run.add_argument("--base-url", default=None, help="教师接口地址；不填时用环境变量 TEACHER_BASE_URL")
    run.add_argument("--api-key-env", default=None, help="教师的密钥放在哪个环境变量里（默认 TEACHER_API_KEY）")
    run.add_argument("--timeout", type=positive, default=timeout,
                     help="教师接口的超时，秒（默认 300；环境变量 TEACHER_TIMEOUT）")

    status = commands.add_parser("status", help="一共接受了多少、各方向多少、因为什么拒绝")
    output(status)

    show = commands.add_parser("show", help="抽查几个已接受的题族，打印题面和 diff")
    output(show)
    show.add_argument("--sample", type=int, default=3, help="抽查几个（默认 3；0 表示只看数量）")
    show.add_argument("--seed", type=int, default=42, help="抽样用的种子（默认 42）")

    export = commands.add_parser("export", help="冻结划分，写出新题文件，并入阶梯文件")
    output(export)
    export.add_argument("--ladder", default=None, help="已有的题池文件（阶梯）；不给或者不存在就只导出新题")
    export.add_argument("--tasks-out", required=True, help="写出的新题文件（全部已接受并冻结的题，每行一道）")
    export.add_argument("--ladder-out", default=None,
                        help="阶梯加上新题以后写到哪；里面是别的内容（不是以前导出的版本）时拒绝覆盖")
    export.add_argument("--dev-ratio", type=float, default=DEV_RATIO, help="新冻结的题族里 dev 的比例（默认 0.2）")
    export.add_argument("--test-ratio", type=float, default=TEST_RATIO, help="新冻结的题族里 test 的比例（默认 0.2）")
    export.add_argument("--seed", type=int, default=42, help="冻结划分用的种子（默认 42）")
    return parser


def recorded(output_dir) -> set[str]:
    """families 里有记录的 slug（不管有没有定论）。"""
    return {path.stem for path in families_dir(output_dir).glob("*.json")}


def makeup_note(plan: list, batch: int, seen: set[str]) -> str:
    """run 开头那一行里说明重跑时这一批的来路：以前要过、还没有定论的排在最前，后面接着没有要过的。

    排主题时跳过有定论的 slug、凑满这一批的个数，所以原样重跑并不只是"把没做完的补上"：它会把这一批补满。
    只补以前要过的那几个，就把 FAMILIES 写成它们的个数——这只有在它们全排在前面时才准（没有要过的排在它们
    前面的话，FAMILIES 写几个都取不到该取的那几个），不准的时候不给这句话。
    """
    asked = [themes.slug_of(*theme) in seen for theme in plan]
    again = sum(asked)
    if not again:
        return ""                                   # 头一回要：没有什么要说明的
    if again == len(asked):
        return "；都是以前要过、还没有定论的（补做）"
    note = f"；其中 {again} 个是以前要过、还没有定论的，{len(asked) - again} 个是没有要过的"
    if all(asked[:again]):
        note += (f"（只补这 {again} 个、不要新主题：FAMILIES={again} BATCH={batch} bash scripts/run_synth_pool.sh）")
    return note


def hopeless_reason(summary: dict, max_tokens: int) -> str | None:
    """这一批一个都没接受，而且没有一族是因为题族本身被拒的：问题出在教师接口、回复或者判分环境上，
    重跑之前得先把原因修好。返回给用户看的一句话；不是这种情况返回 None。"""
    if summary["accepted"] or not summary["rejected"]:
        return None
    if not set(summary["reasons"]) <= NOT_ABOUT_CONTENT:
        return None
    counted = Counter(summary["reasons"])
    listed = "、".join(f"{label} {counted[label]} 族" for label in NOT_ABOUT_CONTENT_ORDER if counted[label])
    if counted[TRUNCATED] or counted[PARSE_ERROR]:
        return (f"教师有回复，但没有一族能用（{listed}）。回复被截断就把输出上限调大：环境变量 TEACHER_MAX_TOKENS"
                f"（现在 {max_tokens}）；回复没法解析多半是模型没按格式回答，换一个 TEACHER_MODEL，或者把温度调低"
                f"（TEACHER_TEMPERATURE）")
    if counted[TEACHER_ERROR] == sum(counted.values()):
        return "教师调用全部出错：检查 TEACHER_API_KEY、TEACHER_BASE_URL 和网络"
    return f"这一批一个都没接受（{listed}）"


def command_run(args, parser) -> int:
    # 真正要请求时才导入教师接口（本机和 CI 可能没装 openai：要到 make_client 里才 import 它）
    from _shared_teacher import KEY_ENV, chat_text, configure_throttle, extra_json, make_client, teacher_model

    try:
        extra_json()                     # 设了 TEACHER_EXTRA_JSON 就得是 JSON 对象：不对就在请求之前报
    except ValueError as error:
        parser.error(str(error))
    try:
        directions = parse_directions(args.directions)
    except ValueError as error:
        parser.error(str(error))
    if args.temperature <= 0:
        parser.error("--temperature 必须是正数")
    if args.min_interval < 0:                      # 命令行给了负数（build_parser 里只查了环境变量给的默认值）
        parser.error("--min-interval 不能是负数")
    output = Path(args.output_dir)
    families_dir(output).mkdir(parents=True, exist_ok=True)
    settled_slugs = used_slugs(output)
    try:
        plan = planned_themes(args.batch, args.families, directions, seed=args.seed, used_slugs=settled_slugs)
    except ValueError as error:
        print(f"排不出第 {args.batch} 批的主题：{error}", file=sys.stderr)
        return 1
    if directions == frozenset(themes.DIRECTION_NAMES):
        names = "全部方向"
    else:
        names = "、".join(direction_label(name) for name in themes.DIRECTION_NAMES if name in directions)
    print(f"第 {args.batch} 批：向教师要 {len(plan)} 个题族（{names}）"
          f"{makeup_note(plan, args.batch, recorded(output))}；每个最多 {args.rounds} 轮，"
          f"同时 {args.concurrency} 个、请求至少隔 {args.min_interval:g} 秒；"
          f"目录里已有定论的题族 {len(settled_slugs)} 个")
    try:
        with one_at_a_time(output):
            try:
                client = make_client(args.api_key_env or KEY_ENV, args.base_url, args.timeout)
            except RuntimeError as error:             # 没有密钥
                print(f"{error}；也可以用 --api-key-env 指定放在哪个环境变量里", file=sys.stderr)
                return 1
            except ImportError as error:              # 没装 openai SDK
                print(f"没装 openai SDK：{error}（pip install 'openai>=1.0'）", file=sys.stderr)
                return 1
            # 限额按账户算：在飞的请求数和间隔按这一次的参数来定，之后这一批所有的请求都走进程里的同一份节流
            configure_throttle(max_in_flight=args.concurrency, min_interval=args.min_interval)
            examples = {}
            for theme in plan:
                slug = themes.slug_of(*theme)
                examples[slug] = prompts.pick_example(theme[0], slug)
            with tempfile.TemporaryDirectory(prefix="synth-pool-") as temporary:
                stop = threading.Event()
                try:
                    with stop_signals(stop, temporary):
                        summary = run_batch(plan, make_ask(client, args.model or teacher_model(), chat_text), output,
                                            batch=args.batch, rounds=args.rounds, concurrency=args.concurrency,
                                            max_tokens=args.max_tokens, temperature=args.temperature,
                                            work_dir=Path(temporary), examples=examples, stop=stop)
                except BatchStopped as stopped:
                    # 第一次 Ctrl-C：正在做的题族都做完、记录落盘了，才走到这里
                    settled_now = used_slugs(output)
                    done = sum(1 for theme in plan if themes.slug_of(*theme) in settled_now)
                    left = [theme for theme in plan if themes.slug_of(*theme) not in settled_now]
                    print(f"收到 Ctrl-C，这一批停下了：{stopped.planned} 个主题里 {done} 个已经有了定论，"
                          f"其余的没有做完。原样重跑会接着做（没做完的排在最前，再要满这一批）；只补它们、不要"
                          f"新主题：FAMILIES={len(left)} BATCH={args.batch} bash scripts/run_synth_pool.sh",
                          file=sys.stderr)
                    return 130
    except AlreadyRunning as error:                   # 另一个 run 正在这个目录上跑
        print(error, file=sys.stderr)
        return 1
    if summary["unwritten"] or summary["stopped"]:
        return 1                                     # 原因和接着怎么办 run_batch 已经说了
    cause = hopeless_reason(summary, args.max_tokens)
    if cause:
        print(cause + "。这一批的主题都还没有定论：修好以后原样重跑，就是把它们再要一次，不会要新主题。",
              file=sys.stderr)
        return 3
    return 0


def command_status(args) -> int:
    for line in status_lines(args.output_dir):
        print(line)
    return 0


def command_show(args) -> int:
    if args.sample < 0:
        print("--sample 不能是负数", file=sys.stderr)
        return 1
    for line in show_lines(args.output_dir, args.sample, args.seed):
        print(line)
    return 0


def command_export(args) -> int:
    try:
        export_pool(args.output_dir, args.ladder, args.tasks_out, args.ladder_out, dev_ratio=args.dev_ratio,
                    test_ratio=args.test_ratio, seed=args.seed)
    except ExportError as error:
        print(f"导出没有完成：{error}", file=sys.stderr)
        return 1
    except OSError as error:                          # 写不进去（目录只读、磁盘满……）
        print(f"导出没有完成：写文件出错（{error}）", file=sys.stderr)
        return 1
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return command_run(args, parser)
    if args.command == "status":
        return command_status(args)
    if args.command == "show":
        return command_show(args)
    return command_export(args)


if __name__ == "__main__":
    raise SystemExit(main())
