#!/usr/bin/env python3
"""Pre-push privacy audit: block pushes containing sensitive content.

Scans the files git tracks (what would actually be pushed) for personal
information patterns and forbidden file types. The project is also delivered
as an unpacked release, which has no .git: there the files to scan are the
whole tree minus what .gitignore excludes. Exits 1 (blocks push) on any
violation, and also when there is nothing to scan.

Usage: python3 scripts/privacy_check.py [--verbose]
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SELF = Path(__file__).resolve()


def _tracked_files() -> list[Path]:
    """The files to scan: the ones git tracks (what would actually be pushed)."""
    result = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, cwd=ROOT)
    if result.returncode == 0:
        return [ROOT / line for line in result.stdout.splitlines() if line.strip()]
    # 这里不是仓库（解压出来的发布包）：没有东西可跟踪，会公开出去的就是这棵树本身
    return _release_files()


def _release_files() -> list[Path]:
    """Files of an unpacked release: the whole tree minus what the ignore rules exclude.

    忽略规则交给 git 判，和 .gitignore 不会走样：临时建一个空仓库顶替缺的 .git，
    ROOT 下面不写任何东西。列不出来时返回空表，由 main 报"没有找到要检查的文件"。
    """
    with tempfile.TemporaryDirectory(prefix="privacy-check-") as git_dir:
        subprocess.run(["git", "init", "-q", "--bare", git_dir], check=True)
        result = subprocess.run(
            ["git", f"--git-dir={git_dir}", f"--work-tree={ROOT}", "ls-files",
             "--others", "--exclude-standard", "-z"],
            capture_output=True, text=True, cwd=ROOT)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        return []
    return sorted(ROOT / name for name in result.stdout.split("\0") if name)

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


def check_paths(files: list[Path]) -> list[str]:
    """Check file paths against forbidden patterns."""
    violations = []
    for path in files:
        if path == SELF or not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        for name, pattern in FORBIDDEN_PATH_PATTERNS:
            if re.search(pattern, rel, re.IGNORECASE):
                violations.append(f"  路径违规 [{name}]: {rel}")
    return violations


def check_contents(files: list[Path]) -> list[str]:
    """Scan file contents for sensitive patterns."""
    violations = []
    for path in files:
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
    files = [path for path in _tracked_files() if path != SELF and path.is_file()]
    if not files:
        # 一个文件都不查不能算通过：没有 .git 的目录里原先就是这样报的"无隐私违规"
        print("✗ 没有找到要检查的文件，无法判断能否推送。")
        return 1
    path_violations = check_paths(files)
    content_violations = check_contents(files)
    all_violations = path_violations + content_violations
    if all_violations:
        print(f"✗ 发现 {len(all_violations)} 项违规，阻止推送：")
        for v in all_violations:
            print(v)
        print("\n修复后重试，或 FORCE_PUSH=1 跳过（不建议）。")
        return 1
    print("✓ 无隐私违规，允许推送。")
    if verbose:
        print(f"  扫描文件: {len(files)}")
        print(f"  扫描路径模式: {len(FORBIDDEN_PATH_PATTERNS)}")
        print(f"  扫描内容模式: {len(CONTENT_PATTERNS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
