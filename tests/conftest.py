# -*- coding: utf-8 -*-
"""测试套件的全局设置。"""
import os

import pytest

# 判分默认拒绝以 root 执行不可信代码；有些 CI 容器只能以 root 运行测试。
if hasattr(os, "geteuid") and os.geteuid() == 0:
    os.environ.setdefault("AGENTLAB_ALLOW_ROOT", "1")


@pytest.fixture
def dead_proxy(monkeypatch):
    """环境里设着指向空端口的代理变量：访问本机服务的请求不能被它截走。"""
    for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")
