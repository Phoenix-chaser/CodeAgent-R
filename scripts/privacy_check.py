#!/usr/bin/env python3
"""Pre-push privacy audit: block pushes containing sensitive content.

Scans all files in the current commit for personal information patterns
and forbidden file types. Exits 1 (blocks push) on any violation.

Usage: python3 scripts/privacy_check.py [--verbose]
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SELF = Path(__file__).resolve()


def _tracked_files() -> list[Path]:
    """Only scan files tracked by git (what would actually be pushed)."""
    result = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, cwd=ROOT)
    if result.returncode != 0:
        return []
    return [ROOT / line for line in result.stdout.splitlines() if line.strip()]

# 文件名/路径不应出现的关键词（名称, 正则，不区分大小写）
FORBIDDEN_PATH_PATTERNS = [
    ("zcode", r"\.zcode"),
    ("handoff", r"handoff"),
    ("server-guide", r"SERVER-STEP|SERVER-COMPLETE"),
    ("ops-manual", r"操作手册"),
    ("resource-guide", r"RESOURCE-GUIDE"),
    ("quickstart", r"QUICKSTART"),
    ("review-plan", r"审查与优化计划"),
    ("resume", r"简历"),
    ("job-hunting", r"求职"),
    ("env-file", r"\.env$"),
    ("key-file", r"\.key$"),
    ("credential", r"credential"),
    ("password", r"password"),
    ("secret", r"secret"),
    ("pem", r"\.pem$"),
    ("doctor-archive", r"p00-doctor"),
]

# 文件内容不应出现的内容模式（文件名, 正则）
CONTENT_PATTERNS = [
    ("硬件型号", r"RTX\s*4090|GTX\s*\d{4}"),
    ("个人路径", r"/home/[a-z]+\d|/Users/[a-z]+\d|C:\\\\Users\\\\[a-z]"),
    ("服务器账号", r"u\d{10}"),
    ("服务商-智谱", r"智谱|bigmodel\.cn|ZHIPU_"),
    ("服务商-GLM", r"glm-\d|GLM-\d"),
    ("服务商-DeepSeek", r"deepseek\.com"),
    ("求职/简历", r"简历|求职|面试官|实习岗"),
    ("API密钥格式", r"sk-[a-zA-Z0-9]{20,}"),
    ("私钥格式", r"-----BEGIN (RSA |EC )?PRIVATE KEY-----"),
]

# 内容扫描跳过的文件（大文件或二进制）
SKIP_EXTENSIONS = {".safetensors", ".bin", ".pt", ".pth", ".pyc", ".png",
                   ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2",
                   ".ttf", ".zip", ".tar.gz", ".tar", ".gz"}


def check_paths() -> list[str]:
    """Check file paths against forbidden patterns (git-tracked only)."""
    violations = []
    for path in _tracked_files():
        if path == SELF or not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        for name, pattern in FORBIDDEN_PATH_PATTERNS:
            if re.search(pattern, rel, re.IGNORECASE):
                violations.append(f"  路径违规 [{name}]: {rel}")
    return violations


def check_contents() -> list[str]:
    """Scan file contents for sensitive patterns (git-tracked only)."""
    violations = []
    for path in _tracked_files():
        if path == SELF or not path.is_file():
            continue
        if path.suffix.lower() in SKIP_EXTENSIONS:
            continue
        if path.stat().st_size > 500_000:
            continue
        rel = path.relative_to(ROOT).as_posix()
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for label, pattern in CONTENT_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                violations.append(
                    f"  内容违规 [{label}] ×{len(matches)}: {rel}")
    return violations


def main() -> int:
    verbose = "--verbose" in sys.argv
    print("── 隐私审查 ──")
    path_violations = check_paths()
    content_violations = check_contents()
    all_violations = path_violations + content_violations
    if all_violations:
        print(f"✗ 发现 {len(all_violations)} 项违规，阻止推送：")
        for v in all_violations:
            print(v)
        print("\n修复后重试，或 FORCE_PUSH=1 跳过（不建议）。")
        return 1
    print("✓ 无隐私违规，允许推送。")
    if verbose:
        print(f"  扫描路径模式: {len(FORBIDDEN_PATH_PATTERNS)}")
        print(f"  扫描内容模式: {len(CONTENT_PATTERNS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
