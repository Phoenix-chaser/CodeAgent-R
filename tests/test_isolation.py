# -*- coding: utf-8 -*-
"""隔离层在真实子进程里的行为。依赖 Landlock 的用例在内核不支持时跳过。"""
import os
import sys

import pytest

from agentlab.runtime import isolation

needs_landlock = pytest.mark.skipif(
    not isolation.probe_landlock().available, reason="kernel Landlock unavailable")


@pytest.fixture
def sandbox(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "file.txt").write_text("original", encoding="utf-8")
    return workspace, isolation.build_policy(workspace, tmp_path / "scratch", mode="landlock")


def run(policy, workspace, code):
    return isolation.run_sandboxed(
        policy, [sys.executable, "-B", "-c", code], workspace, 30)


@needs_landlock
def test_environment_is_built_not_inherited(sandbox, monkeypatch):
    workspace, policy = sandbox
    monkeypatch.setenv("TEACHER_API_KEY", "should-not-leak")
    out = run(policy, workspace, (
        "import json, os\n"
        "print(json.dumps([os.environ.get('TEACHER_API_KEY'), os.environ['HOME'], os.getcwd()]))"))
    assert out.exit_status == "ok", out.stderr
    secret, home, cwd = __import__("json").loads(out.stdout)
    assert secret is None and cwd == str(workspace)
    assert home == str(policy.scratch / "home")


@needs_landlock
def test_workspace_is_read_only(sandbox):
    workspace, policy = sandbox
    out = run(policy, workspace, (
        "try:\n    open('file.txt', 'w')\n    print('writable')\n"
        "except OSError as error:\n    print(type(error).__name__, error.errno)"))
    assert out.exit_status == "ok" and out.stdout.strip() == "PermissionError 13"
    assert (workspace / "file.txt").read_text(encoding="utf-8") == "original"


@needs_landlock
def test_real_home_proc_and_etc_are_unreachable(sandbox):
    workspace, policy = sandbox
    out = run(policy, workspace, (
        "import os\n"
        "def probe(path):\n"
        "    try:\n"
        "        os.listdir(path) if os.path.isdir(path) else open(path, 'rb').read()\n"
        "        return 'allowed'\n"
        "    except OSError as error:\n"
        "        return 'denied'\n"
        f"print(probe({os.path.expanduser('~')!r}), probe('/proc/self/environ'), probe('/etc/passwd'))"))
    assert out.stdout.split() == ["denied", "denied", "denied"]


@needs_landlock
@pytest.mark.skipif(not isolation.probe_landlock().blocks_tcp, reason="Landlock ABI < 4")
def test_tcp_connect_is_denied(sandbox):
    workspace, policy = sandbox
    out = run(policy, workspace, (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('127.0.0.1', 9), timeout=3)\n"
        "    print('connected')\n"
        "except OSError as error:\n"
        "    print(type(error).__name__, error.errno)"))
    assert out.stdout.strip() == "PermissionError 13"


@needs_landlock
def test_only_the_interpreter_is_executable(sandbox):
    workspace, policy = sandbox
    out = run(policy, workspace, (
        "import subprocess, sys\n"
        "for argv in (['/bin/sh', '-c', 'true'], [sys.executable, '-c', \"print('child-ok')\"]):\n"
        "    try:\n"
        "        done = subprocess.run(argv, capture_output=True, text=True)\n"
        "        print(argv[0], done.returncode, done.stdout.strip())\n"
        "    except OSError as error:\n"
        "        print(argv[0], type(error).__name__, error.errno)"))
    shell, python = out.stdout.strip().splitlines()
    assert shell == "/bin/sh PermissionError 13"
    assert python.endswith("0 child-ok")


@needs_landlock
def test_bootstrap_failure_is_an_error_not_a_fallback(sandbox):
    workspace, policy = sandbox
    policy.rules.append(("/nonexistent/path", ["READ_FILE"]))
    with pytest.raises(isolation.SandboxError, match="/nonexistent/path"):
        run(policy, workspace, "print('must not run')")


def test_fallback_mode_is_reported_honestly(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "file.txt").write_text("original", encoding="utf-8")
    policy = isolation.build_policy(workspace, tmp_path / "scratch", mode="process")
    document = policy.document()
    assert document["isolation"] == "process"
    assert document["workspace_readonly"] is False and document["tcp_blocked"] is False
    # 降级模式确实不限制文件系统；详情里的记录与实际一致
    out = run(policy, workspace, "open('file.txt', 'w').write('changed')")
    assert out.exit_status == "ok", out.stderr
    assert (workspace / "file.txt").read_text(encoding="utf-8") == "changed"


def test_mode_is_resolved_from_environment(monkeypatch):
    monkeypatch.setattr(isolation, "probe_landlock",
                        lambda: isolation.SandboxProbe(0, "simulated: old kernel"))
    monkeypatch.setenv("AGENTLAB_SANDBOX", "off")
    assert isolation.resolve_mode() == "process"
    monkeypatch.setenv("AGENTLAB_SANDBOX", "auto")
    assert isolation.resolve_mode() == "process"             # 降级，不中断
    monkeypatch.setenv("AGENTLAB_REQUIRE_SANDBOX", "1")
    with pytest.raises(isolation.UnsupportedSandbox, match="simulated"):
        isolation.resolve_mode()
    monkeypatch.delenv("AGENTLAB_REQUIRE_SANDBOX")
    monkeypatch.setattr(isolation, "probe_landlock", lambda: isolation.SandboxProbe(8, "ok"))
    assert isolation.resolve_mode() == "landlock"
    monkeypatch.setenv("AGENTLAB_SANDBOX", "sometimes")
    with pytest.raises(ValueError):
        isolation.resolve_mode()


def test_command_must_use_an_absolute_interpreter(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    policy = isolation.build_policy(workspace, tmp_path / "scratch", mode="process")
    with pytest.raises(ValueError):
        isolation.run_sandboxed(policy, ["python3", "-c", "print(1)"], workspace, 5)


def test_undecodable_output_does_not_crash_the_caller(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    policy = isolation.build_policy(workspace, tmp_path / "scratch", mode="process")
    out = run(policy, workspace, "import sys\nsys.stdout.buffer.write(b'ok\\xff\\xfe')\n")
    assert out.exit_status == "ok" and out.stdout.startswith("ok")
