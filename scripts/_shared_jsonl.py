"""按行读写的 JSONL：一条记录一行，行尾只认 "\\n"。RFT 这条链上的脚本（sample_rft.py、prepare_data.py、
rft_control.py）读写采样记录、训练数据都用它，彼此才对得上。

Python 把 U+2028、U+2029、U+0085 也当成换行（str.splitlines() 在它们处切行），json.dumps(ensure_ascii=False)
却把它们原样写出来。模型的回答、判分的输出里偶尔会有：原样写出去，按行切的脚本就把一条记录切成几条坏行，
这条记录就读不回来了。所以写的时候把这三个字符写成转义（在 JSON 里完全等价，读回来还是原来的字符），
读的时候按 "\\n" 切行。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

LINE_SEPARATORS = ("\u2028", "\u2029", "\u0085")


def jsonl_line(record) -> str:
    """一条记录写成一行，末尾不带换行：中文原样，只有 LINE_SEPARATORS 里的字符写成 ``\\\\uXXXX`` 转义。

    按 ``splitlines()`` 或按 ``"\\n"`` 切行的脚本都不会把它切开。
    """
    text = json.dumps(record, ensure_ascii=False)
    for character in LINE_SEPARATORS:
        text = text.replace(character, f"\\u{ord(character):04x}")
    return text


def read_lines(path: Path) -> list[str]:
    """按 ``"\\n"`` 把文件切成行。不用 splitlines()：它还把 U+2028、U+0085 这些字符当成行尾。

    截在某个汉字中间的坏字节换成替换字符，不做行尾转换（``newline=""``）：那一行下一刻就因解析不了被跳过。
    """
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        return handle.read().split("\n")


def read_jsonl(path: Path) -> list[dict]:
    """读 JSONL：每行一条记录。写到一半被中断的坏行跳过（追加写的文件；对应的槽位会被重做），文件不存在
    就是空列表。"""
    records = []
    try:
        lines = read_lines(path)
    except FileNotFoundError:
        return records
    for line in lines:
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"跳过没写完的一行：{path.name}", file=sys.stderr)
    return records


def cut_partial_tail(path: Path) -> int:
    """去掉追加写出的文件末尾没写完的那一行（最后一行后面没有换行符），返回去掉的字节数。

    上一次运行写到一半被打断就会留下这样的半行。不去掉，接下来追加的记录会接在它后面，两条粘成一条
    坏行：读不回来，那个槽位每次都被重做、每次都粘坏。记录里不会有原样的换行符，所以 ``"\\n"`` 只可能是行尾。
    """
    try:
        size = path.stat().st_size
    except OSError:
        return 0
    if not size:
        return 0
    with path.open("r+b") as handle:
        handle.seek(size - 1)
        if handle.read(1) == b"\n":
            return 0
        end = size
        while end > 0:
            start = max(0, end - (1 << 16))
            handle.seek(start)
            found = handle.read(end - start).rfind(b"\n")
            if found != -1:
                handle.truncate(start + found + 1)
                return size - (start + found + 1)
            end = start
        handle.truncate(0)
    return size
