# -*- coding: utf-8 -*-
"""确定性测试分词器：字符级 + 特殊多 token 字符（验证边界正确性）。"""
from __future__ import annotations


class StubTokenizer:
    """'错' 字符映射为 2 个 token——专门暴露多 token 边界错误。"""
    vocab_size = 300

    def tokenize(self, text: str) -> list[int]:
        ids: list[int] = []
        for ch in text:
            if ch == "错":
                ids += [100, 101]            # 多 token 字符陷阱
            else:
                ids.append(ord(ch) % 200 + 2)
        return ids

    def decode(self, ids: list[int]) -> str:
        out = []
        i = 0
        while i < len(ids):
            if ids[i] == 100 and i + 1 < len(ids) and ids[i + 1] == 101:
                out.append("错")
                i += 2
            else:
                out.append(chr((ids[i] - 2) % 200 + 0x4E00) if ids[i] >= 2 else "?")
                i += 1
        # 注：decode 仅用于一致性钩子；字符级往返在测试中单独校验
        return "".join(out)
