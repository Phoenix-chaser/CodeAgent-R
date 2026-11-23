# -*- coding: utf-8 -*-
"""测试套件的全局设置。"""
import os

# 判分默认拒绝以 root 执行不可信代码；有些 CI 容器只能以 root 运行测试。
if hasattr(os, "geteuid") and os.geteuid() == 0:
    os.environ.setdefault("AGENTLAB_ALLOW_ROOT", "1")
