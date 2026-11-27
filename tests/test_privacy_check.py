# -*- coding: utf-8 -*-
"""隐私审查（scripts/privacy_check.py）查的是哪些文件。

仓库里查的是 git 跟踪的文件。项目也以压缩包交付，解包后的目录没有 .git：那里原先一个文件都不扫，照样
打印"无隐私违规"。现在解包目录里查的是整棵树里没被忽略规则排除的文件；要查的文件一个都没有时不报通过。
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="需要 git")

# 违规样本在运行时拼出来：写成字面量的话，这个文件自己就过不了隐私审查
HARDWARE = "gpu: " + "RTX" + " 40" + "90\n"
KEY = "teacher key: " + "sk" + "-" + "A" * 30 + "\n"


def release_at(directory, ignore_rules=True, checkout=False):
    """一个交付目录，带着隐私审查脚本；带着仓库的忽略规则，checkout 为真时它还是个刚建好的仓库。"""
    (directory / "scripts").mkdir(parents=True)
    if ignore_rules:
        shutil.copyfile(ROOT / ".gitignore", directory / ".gitignore")
    shutil.copyfile(ROOT / "scripts/privacy_check.py", directory / "scripts/privacy_check.py")
    if checkout:
        subprocess.run(["git", "init", "-q", str(directory)], check=True)
    return directory


def write(directory, relative, text):
    path = directory / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def audit(directory, *arguments):
    """在交付目录里照文档的写法运行一遍，返回退出码和全部输出。"""
    result = subprocess.run([sys.executable, "scripts/privacy_check.py", *arguments],
                            cwd=directory, capture_output=True, text=True, timeout=60)
    return result.returncode, result.stdout + result.stderr


def test_unpacked_release_is_scanned(tmp_path):
    """没有 .git 的目录里，包里的文件要真的查一遍。"""
    release = release_at(tmp_path)
    write(release, "docs/notes.md", HARDWARE)
    write(release, "BUILD-INFO.txt", KEY)
    code, output = audit(release)
    assert code == 1, output
    assert "内容违规 [硬件型号] ×1: docs/notes.md" in output
    assert "内容违规 [API密钥格式] ×1: BUILD-INFO.txt" in output
    assert "允许推送" not in output


@pytest.mark.parametrize("linked", [False, True], ids=["data-dir", "data-link"])
def test_unpacked_release_follows_the_ignore_rules(tmp_path, linked):
    """不进仓库的东西不查：.env、数据、结果、运行期状态、缓存，还有指向旧包的 data 链接；其余的都查。

    服务器上的解包目录就是这个样子：data 链接到上一个包的 data，密钥在 .env 里。
    """
    release = release_at(tmp_path / "release")
    old_data = tmp_path / "old-release"
    write(old_data, "train.jsonl", KEY)
    if linked:
        (release / "data").symlink_to(old_data, target_is_directory=True)
    else:
        write(release, "data/train.jsonl", KEY)
    for relative in (".env", "results/seed42/base.json", ".state/status.json",
                     "__pycache__/module.cpython-312.pyc", ".ruff_cache/0.17.4/some-id"):
        write(release, relative, KEY)
    write(release, "BUILD-INFO.txt", "CodeAgent-R v22\n")
    write(release, "docs/evidence/seed42/base.json", '{"passed": 1}\n')
    code, output = audit(release, "--verbose")
    assert code == 0, output
    assert "✓ 无隐私违规，允许推送。" in output
    assert "扫描文件: 3" in output        # .gitignore、BUILD-INFO.txt、docs/evidence/seed42/base.json


def test_checkout_scans_tracked_files_only(tmp_path):
    """仓库里的口径不变：只查 git 跟踪的文件，也就是会推上去的那些。"""
    repo = release_at(tmp_path, checkout=True)
    write(repo, "docs/notes.md", HARDWARE)
    write(repo, "docs/draft.md", KEY)
    subprocess.run(["git", "add", ".gitignore", "scripts/privacy_check.py", "docs/notes.md"], cwd=repo, check=True)
    code, output = audit(repo)
    assert code == 1, output
    assert "内容违规 [硬件型号] ×1: docs/notes.md" in output
    assert "docs/draft.md" not in output


@pytest.mark.parametrize("checkout", [False, True], ids=["unpacked", "checkout"])
def test_nothing_to_scan_is_not_a_pass(tmp_path, checkout):
    """要查的文件一个都没有时不能报通过：解包目录里和仓库里都一样。"""
    directory = release_at(tmp_path, ignore_rules=False, checkout=checkout)
    code, output = audit(directory)
    assert code == 1, output
    assert "没有找到要检查的文件" in output
    assert "允许推送" not in output
