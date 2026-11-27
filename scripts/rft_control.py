"""构造"不验证"的对照组数据（RFT 的随机信号对照）。

验证组里，每道被解出的题只留一份判分通过的答案。对照组里是同样的这些题，每题从它的
全部采样中随机取一份**格式完整**（能解析成 Python）的答案，不看它对不对。两组的题目、
条数、文风相同，唯一的区别是有没有用判分结果来筛选。如果对照组在 dev 上的变化和验证组
一样，说明变化不是"验证"带来的（参照 Spurious Rewards, arXiv:2506.10947 的思路）。

对照组的训练要加 ``train_sft.py --allow-unverified``；dev 集沿用验证组的，只用来算 loss。

两种模式，必须二选一：

- ``--verified-dir``：上面说的对照组，每道解出的题一份随机采样。
- ``--base-dir`` 加 ``--guided-from``：引导采样那一轮的**配对对照**。引导组（``--guided-from``）
  是基础集（上一轮的验证组，``--base-dir``）加上一批带着提示才解出来的题，每题一份判分通过
  的答案。配对对照是基础集的训练记录原样，加上这批题各一份模型写完了、格式完整、但判分**没通过**
  的答案：两组的题目逐题相同，区别只在答案对不对。一道题连一份这样的采样都没有时配不上，题号写进
  ``unmatched-tasks.txt``；引导组要用 ``prepare_data.py --drop-tasks-file`` 把它也去掉，否则
  两组的题目不一样。

按题对齐只保证到文件为止：训练入口会把超过 ``--max-length`` 的记录记成"超长"丢掉，退出码仍是 0。
所以训练前跑完 ``train_sft.py --dry-run`` 之后，要核对它打印的"加载"条数：对照组应等于 manifest
里的 ``base_samples + samples``，引导组应等于 prepare_data.py 的 manifest 里
``splits.train.samples``；对不上就说明两组又不一样了。
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agentlab.benchmark.tiers import LABELS, tier_of
from _shared_jsonl import jsonl_line, read_lines


def well_formed(record: dict) -> bool:
    """输出是完整、可解析的 Python。对照组只去掉"对不对"这一道筛选，不去掉格式筛选。"""
    code = record.get("target", {}).get("content", "")
    try:
        return bool(code) and bool(ast.parse(code).body)
    except (SyntaxError, ValueError):
        return False


def read_jsonl(path: Path) -> list[dict]:
    """读一个 JSONL 文件（sample_rft.py、prepare_data.py 写的：一条记录一行，行尾只认 "\n"），空行跳过。

    按 "\n" 切行，不用 splitlines()：记录里的 U+2028 这类字符不是换行，切开了这条记录就读不回来。
    """
    return [json.loads(line) for line in read_lines(path) if line.strip()]


def usable_as_control(record: dict) -> bool:
    """这条采样能不能当配对对照的答案：模型写完了、格式完整、没有判分通过。

    写到一半被截断的输出（``finish_reason`` 不是 ``stop``）不算。它没有判过分，``strict_pass`` 为假只是
    "没被接受"；而且它差不多用满了采样的长度上限（默认和训练的 ``--max-length`` 一样是 2048），加上
    提示几乎一定超长，训练入口会把它记成"超长"丢掉：对照组就这样悄悄少了这道题，引导组却还留着，
    两组不再逐题对齐。这样的题宁可记成配不上，让两组都不留。
    """
    return (record.get("finish_reason", "stop") == "stop"      # 没有这个字段的记录按写完了算
            and well_formed(record)
            and (record.get("provenance") or {}).get("strict_pass") is not True)


def build_control(graded: list[dict], verified_train: list[dict], seed: int) -> tuple[list[dict], dict]:
    """给验证组训练集里的每道题随机选一份采样。"""
    accepted = sorted({record["task_id"] for record in verified_train})
    pool = defaultdict(list)
    for record in graded:
        if record["task_id"] in accepted and well_formed(record):
            pool[record["task_id"]].append(record)
    samples = []
    for task in accepted:
        if not pool[task]:
            raise ValueError(f"{task}: 没有格式完整的采样，无法构造对照")
        chosen = random.Random(f"{seed}:{task}").choice(pool[task])
        samples.append({**chosen, "sample_id": chosen.get("sample_id") or f"{task}-control",
                        "selection": {"policy": "control_unverified_random", "pool": len(pool[task])},
                        "provenance": {**chosen["provenance"], "control": "unverified_random_sample"}})
    passing = sum(bool(sample["provenance"]["strict_pass"]) for sample in samples)
    stats = {
        "schema_version": 1, "samples": len(samples), "tasks": len(accepted),
        "actually_passing": passing,
        "by_tier": dict(Counter(LABELS[tier_of(sample["task_id"])] for sample in samples)),
        "label_validation": "none_by_design_control",
        "limitation": "对照组里随机选中的答案可能本来就是对的；它检验的是“按判分结果筛选”这一步"
                      "是否带来差别，不是“训练错误答案是否有害”。",
    }
    return samples, stats


def build_matched_control(graded: list[dict], tasks: list[str], seed: int) -> tuple[list[dict], dict]:
    """给每道引导题配一份判分**没通过**、格式完整的采样。

    ``tasks`` 是引导题的题号（引导组在基础集之外多出来的题）。每道题从 ``graded`` 里它自己的采样
    中取：模型写完了、格式完整、``strict_pass`` 不为真的那些（见 ``usable_as_control``），按它们在
    ``graded`` 里出现的先后排成池子，再用 ``f"{seed}:{task}:failing"`` 做种子随机选一份。种子按题
    分开，增减题目不影响其它题抽到的答案。一份都没有的题不硬配：记进 ``stats["unmatched"]``，由
    调用方把它从引导组里也去掉。``stats["tasks"]`` 是引导题的总数，``samples`` 是配上的记录数
    （每题一条），两者之差就是配不上的题数。

    ``stats["chosen"]`` 记下每道题抽到的是哪一份，连同采样时数出来的输入、输出 token 数。这里的按题
    对齐只到文件为止：训练入口还会丢掉写完了但超长的记录，核对"加载"条数时靠它们找出是哪一条。
    """
    wanted = set(tasks)
    pool = defaultdict(list)
    for record in graded:
        if record["task_id"] in wanted and usable_as_control(record):
            pool[record["task_id"]].append(record)
    samples = []
    unmatched = []
    for task in sorted(wanted):
        if not pool[task]:
            unmatched.append(task)
            continue
        chosen = random.Random(f"{seed}:{task}:failing").choice(pool[task])
        samples.append({**chosen, "sample_id": chosen.get("sample_id") or f"{task}-control",
                        "selection": {"policy": "control_failing_random", "pool": len(pool[task])},
                        "provenance": {**(chosen.get("provenance") or {}),
                                       "control": "failing_random_sample"}})
    stats = {
        "schema_version": 1, "samples": len(samples), "tasks": len(wanted),
        "by_tier": dict(Counter(LABELS[tier_of(sample["task_id"])] for sample in samples)),
        "unmatched": unmatched,
        "chosen": [{"task_id": sample["task_id"], "sample_id": sample["sample_id"],
                    "verdict": sample["provenance"].get("verdict"),
                    "finish_reason": sample.get("finish_reason"),
                    "input_tokens": (sample.get("usage") or {}).get("input_tokens"),
                    "output_tokens": (sample.get("usage") or {}).get("output_tokens")}
                   for sample in samples],
        "label_validation": "none_by_design_control",
        "limitation": "对照组里的答案只保证模型写完了、格式完整、没有判分通过（判分没过的，或者判分超时的），"
                      "也没有人工核对；它们和引导组的正确答案未必出自同样的采样条件（例如提示不同）。"
                      "它检验的是“同一道题训练正确答案还是错误答案”是否带来差别，不是“训练错误答案是否"
                      "有害”。这里的按题对齐只到文件为止：训练入口会丢掉超过 --max-length 的记录，训练前"
                      "要用 train_sft.py --dry-run 核对“加载”的条数等于 base_samples + samples。配不上的题"
                      "不在对照组里，引导组也要去掉它们，两组才按题对齐。",
    }
    return samples, stats


def write_matched_control(parser, args, graded: list[dict], output: Path) -> None:
    """配对对照的命令行模式：引导题逐题配一份没通过的采样。先全部算好再落盘。"""
    base = Path(args.base_dir)
    directories = [Path(directory) for directory in args.guided_from]
    missing = [path for path in (base / "train.jsonl", base / "dev.jsonl",
                                 *(directory / "train.jsonl" for directory in directories))
               if not path.is_file()]
    if missing:
        parser.error("找不到 " + "、".join(str(path) for path in missing))
    base_train = read_jsonl(base / "train.jsonl")
    base_tasks = {record["task_id"] for record in base_train}
    # 引导题 = 各引导组训练集里出现过的题的并集，减去基础集里已有的题
    groups = [{record["task_id"] for record in read_jsonl(directory / "train.jsonl")} - base_tasks
              for directory in directories]
    guided = set().union(*groups)
    if not guided:
        parser.error("--guided-from 的训练集里没有基础集之外的题：引导采样没有解出新题，或者目录给错了")
    samples, stats = build_matched_control(graded, sorted(guided), args.seed)
    if not samples:
        # 一道都配不上时，对照组只剩基础记录、和基础集一模一样；这多半是 --graded 给错了文件
        absent = len(guided - {record["task_id"] for record in graded})
        hint = ("先核对 --graded 是不是这次运行里 train 的 graded 文件" if absent
                else "这些题的采样里没有一份写完了、格式完整、又没判分通过的答案")
        parser.error(f"引导题一道也配不上（{len(guided)} 道里 {absent} 道在 --graded 里一条记录都没有）："
                     f"{hint}。全都配不上时对照组只剩基础记录，所以不写")
    matched = {sample["task_id"] for sample in samples}

    output.mkdir(parents=True, exist_ok=True)
    # 基础记录一个字节都不改：整份拷过来，再往后面追加
    base_bytes = (base / "train.jsonl").read_bytes()
    (output / "train.jsonl").write_bytes(base_bytes)
    with (output / "train.jsonl").open("a", encoding="utf-8") as handle:
        if base_bytes and not base_bytes.endswith(b"\n"):
            handle.write("\n")           # 末行没有换行时补上，免得和后面第一条对照记录粘成一行
        for sample in samples:
            handle.write(jsonl_line(sample) + "\n")          # 同 prepare_data.py：一条记录一行，按 "\n" 切得开
    shutil.copyfile(base / "dev.jsonl", output / "dev.jsonl")
    (output / "unmatched-tasks.txt").write_text("".join(f"{task}\n" for task in stats["unmatched"]),
                                                encoding="utf-8")
    stats.update(base_dir=str(base), guided_from=[str(directory) for directory in directories],
                 graded=str(Path(args.graded)), base_samples=len(base_train))
    (output / "dataset-manifest.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
    unmatched = stats["unmatched"]
    print(f"结果：配对对照 = 基础 {len(base_train)} 条 + 引导题 {len(samples)} 条"
          f"（来自 {sum(bool(tasks & matched) for tasks in groups)} 组）；"
          f"配不上的题 {len(unmatched)} 道 {unmatched[:8]}{'…' if len(unmatched) > 8 else ''}")
    if unmatched:
        print(f"配不上的题记在 {output / 'unmatched-tasks.txt'}：引导组要用 prepare_data.py "
              "--drop-tasks-file 把它们去掉，两组才按题对齐")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graded", required=True,
                        help="sample_rft.py 的 <split>-graded.jsonl；对照组的答案都从这里取")
    parser.add_argument("--verified-dir",
                        help="对照组模式：验证组的数据目录（prepare_data.py 的输出）")
    parser.add_argument("--base-dir",
                        help="配对对照模式：基础集的数据目录（上一轮 prepare_data.py 的输出，"
                             "含 train.jsonl 和 dev.jsonl）")
    parser.add_argument("--guided-from", action="append", nargs="+", default=[], metavar="DIR",
                        help="配对对照模式：引导组的数据目录，可以给多个（空格分开，或重复这个参数）")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # 两种写法都接受：--guided-from A B 和 --guided-from A --guided-from B。同一个目录只算一次，哪怕给了
    # 两次、或者绝对和相对的写法各给了一次；manifest 里记的是第一次出现的写法
    unique: dict[Path, str] = {}
    for group in args.guided_from:
        for directory in group:
            unique.setdefault(Path(directory).resolve(), directory)
    args.guided_from = list(unique.values())
    if bool(args.verified_dir) == bool(args.base_dir):
        parser.error("两种模式必须二选一：--verified-dir（对照组），或 --base-dir 加 --guided-from（配对对照）")
    if args.guided_from and not args.base_dir:
        parser.error("--guided-from 要和 --base-dir 一起用，不能和 --verified-dir 混用")
    if args.base_dir and not args.guided_from:
        parser.error("--base-dir 至少要配一个 --guided-from 目录")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        parser.error("output directory must be empty to protect existing datasets")
    graded = read_jsonl(Path(args.graded))
    if args.verified_dir:
        verified = Path(args.verified_dir)
        train = read_jsonl(verified / "train.jsonl")
        samples, stats = build_control(graded, train, args.seed)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "train.jsonl").open("w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(jsonl_line(sample) + "\n")
        shutil.copyfile(verified / "dev.jsonl", output / "dev.jsonl")
        stats["verified_dir"] = str(verified)
        (output / "dataset-manifest.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                                      encoding="utf-8")
        print(f"对照组：{stats['samples']} 条 / {stats['tasks']} 题，其中碰巧判分通过的 {stats['actually_passing']}"
              f" 条；分层 {stats['by_tier']}")
        return
    write_matched_control(parser, args, graded, output)


if __name__ == "__main__":
    main()
