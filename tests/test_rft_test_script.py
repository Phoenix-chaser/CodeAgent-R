# -*- coding: utf-8 -*-
"""rft_test.sh 的端到端演练：假的 python 代替评测、比较和分层汇总三个脚本。

演的是编排本身——读没读对选型结果、保留基座时评不评、两份报告各评一次、中断后接着评。
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not shutil.which("bash"), reason="needs Bash")

STUB = '''#!/usr/bin/env python3
"""假的 python：评测、比较、分层汇总给出像样的产物；其余命令交给真解释器。"""
import os, sys
from pathlib import Path
args = sys.argv[1:]

def log(name):
    with open(os.environ["STUB_LOG"], "a", encoding="utf-8") as handle:
        handle.write(name + " " + " ".join(args) + "\\n")

if args[0].endswith("evaluate.py"):
    log("evaluate")
    Path(args[args.index("--output") + 1]).parent.mkdir(parents=True, exist_ok=True)
    Path(args[args.index("--output") + 1]).write_text('{"results": []}')
elif args[0].endswith("compare.py"):
    log("compare")
    Path(args[args.index("--output") + 1]).write_text("# 比较（假的）\\n")
elif args[0].endswith("tier_report.py"):
    log("tiers")
    print("结果：（假的分层表）")
else:
    os.execv(sys.executable, [sys.executable, *args])
'''


@pytest.fixture
def stage(tmp_path):
    home = tmp_path / "home"
    (home / "repo/scripts").mkdir(parents=True)
    (home / "bin").mkdir()
    shutil.copy(ROOT / "scripts/rft_test.sh", home / "repo/scripts/rft_test.sh")
    (home / "bin/python").write_text(STUB, encoding="utf-8")
    (home / "bin/python").chmod(0o755)
    (home / "repo/data").mkdir()
    (home / "repo/data/frozen-ladder.jsonl").write_text('{"task_id": "t", "split": "test"}\n',
                                                        encoding="utf-8")
    checkpoint = home / "repo/data/rft-round1/checkpoints/A-verified"
    checkpoint.mkdir(parents=True)
    (checkpoint / "training-manifest.json").write_text(json.dumps(
        {"model": "org/model", "revision": "requested", "resolved_revision": "resolved-commit"}),
        encoding="utf-8")
    return home


def select(stage, adapter):
    results = stage / "repo/data/rft-round1/results/A-verified"
    results.mkdir(parents=True, exist_ok=True)
    (results / "selection.json").write_text(json.dumps({"selected_adapter": adapter}), encoding="utf-8")


def run(stage, log, *arguments):
    if log.exists():
        log.unlink()
    environment = {**os.environ, "HOME": str(stage), "STUB_LOG": str(log),
                   "PATH": f"{stage / 'bin'}{os.pathsep}{os.environ['PATH']}"}
    result = subprocess.run(["bash", "scripts/rft_test.sh", *arguments], cwd=stage / "repo",
                            env=environment, capture_output=True, text=True, timeout=120)
    calls = [line.split(" ", 1) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
    return result, calls


def test_test_is_evaluated_once_for_the_selected_adapter_and_resumed_if_interrupted(stage):
    log = stage / "calls.log"
    adapter = "/somewhere/data/rft-round1/checkpoints/A-verified/epoch-001/adapter"
    select(stage, adapter)
    result, calls = run(stage, log, "A-verified")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "模型 org/model @ resolved-commit；选中的 adapter：/somewhere/" in result.stdout
    assert [name for name, _ in calls] == ["evaluate", "evaluate", "compare", "tiers"]
    base, selected = calls[0][1], calls[1][1]
    for call in (base, selected):
        assert "--split test" in call and "--temperature 0" in call and "--quantization nf4" in call
        assert "--revision resolved-commit" in call
        assert "checkpoints/A-verified/training-manifest.json" in call       # 带上训练数据，做泄漏审计
    assert "--adapter" not in base and f"--adapter {adapter}" in selected
    assert base.endswith("--output data/rft-round1/results/A-verified/test/base.json")
    assert selected.endswith("--output data/rft-round1/results/A-verified/test/selected.json")
    assert "--baseline data/rft-round1/results/A-verified/test/base.json" in calls[2][1]
    assert "test 只在定了配置之后评" in (ROOT / "scripts/rft_test.sh").read_text(encoding="utf-8")

    (stage / "repo/data/rft-round1/results/A-verified/test/selected.json").unlink()      # 第二份评到一半断了
    result, calls = run(stage, log, "A-verified")
    assert result.returncode == 0 and [name for name, _ in calls] == ["evaluate", "compare", "tiers"]
    assert "基座的 test 报告已经有了，跳过" in result.stdout                 # 评完的那份不重评


def test_nothing_is_evaluated_when_dev_kept_the_base(stage):
    log = stage / "calls.log"
    select(stage, None)
    result, calls = run(stage, log, "A-verified")
    assert result.returncode == 0 and calls == []
    assert "保留了基座" in result.stdout and "===== 全部结束 =====" in result.stdout


def test_a_missing_selection_stops_before_touching_the_gpu(stage):
    log = stage / "calls.log"
    result, calls = run(stage, log, "B-oldhp")
    assert result.returncode == 1 and calls == [] and "还没有选完型" in result.stdout
    result, calls = run(stage, log)
    assert result.returncode == 1 and "用法：bash scripts/rft_test.sh" in result.stderr
