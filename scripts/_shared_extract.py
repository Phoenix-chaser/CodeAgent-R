"""Extract complete Python responses without dropping constants or strings."""
from __future__ import annotations

import ast
import re


_CODE_START = re.compile(
    r"^(?:async\s+def\s|def\s|class\s|import\s|from\s|@|#|"
    r"[A-Za-z_]\w*\s*(?::[^=]+)?=(?!=))")


def extract_code(response: str) -> str:
    if not response or not response.strip():
        return ""
    text = response.strip()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        tree = None
    if tree is not None and any(
        not isinstance(node, ast.Expr) or isinstance(node.value, ast.Constant)
        for node in tree.body
    ):
        return text + "\n"
    fence = re.search(r"```(?:python3?|py)?[ \t]*(?:\r?\n)?", text)
    if fence:
        body = text[fence.end():]
        closing = body.find("```")
        code = (body[:closing] if closing >= 0 else body).strip()
        return code + "\n" if code else ""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if _CODE_START.match(line.lstrip()):
            return "\n".join(lines[index:]).strip() + "\n"
    return ""
