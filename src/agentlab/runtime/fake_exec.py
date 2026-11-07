# -*- coding: utf-8 -*-
"""fake 可执行：剧本驱动的外部运行器替身（P04 契约测试用）。

制造：合法流 / stderr 噪声 / 非零退出 / 重复事件 / 截断末行 /
缺 usage / 缺终止事件 / 超时挂死。
"""
from __future__ import annotations

import stat
from pathlib import Path

TEMPLATE = '''#!/usr/bin/env python3
import json, sys, time

scenario = {scenario!r}

def emit(ev, **kw):
    envelope = dict(schema_version=1, event=ev)
    envelope.update(kw)
    print(json.dumps(envelope, ensure_ascii=False), flush=True)

emit("runtime_episode_start", payload={{"task": "t"}})
if scenario.get("model_events"):
    emit("model_request", payload={{"i": 1}})
    u = scenario.get("usage")
    emit("model_response", payload={{"i": 1}}, usage=u)
for dup in range(scenario.get("duplicate_times", 1)):
    emit("tool_result", event_id="fixed-evt-1", payload={{"call_id": "c1"}})
if scenario.get("stderr_noise"):
    sys.stderr.write("noise: boot warnings\\n")
if scenario.get("sleep"):
    time.sleep(scenario["sleep"])
if scenario.get("truncate_last"):
    sys.stdout.write('{{"schema_version":1,"event":"tool_resu')
    sys.stdout.flush()
if scenario.get("termination", True):
    emit("runtime_termination", payload={{"reason": "done"}})
sys.exit(scenario.get("exit_code", 0))
'''


def write_fake_exec(path: Path, scenario: dict) -> Path:
    path.write_text(TEMPLATE.format(scenario=scenario), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path
