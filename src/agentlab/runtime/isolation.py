# -*- coding: utf-8 -*-
"""不可信候选代码的执行隔离：Landlock + rlimit + 清洁环境，在独立 bootstrap 进程里生效。

判分必须在宿主机上运行模型生成的代码。本模块把隔离放进一个单独的
bootstrap 进程，在加载任何不可信代码之前完成三件事：

1. Landlock（Linux ≥ 5.13，不需要 root 或容器）：文件系统只读白名单加少量
   可写目录；ABI ≥ 4 时同时拒绝全部 TCP connect/bind。
2. rlimit：CPU 时间、地址空间、单个文件大小、core dump、同用户进程数。
3. 清洁环境：白名单构造，不继承父进程的任何变量，API key 不会泄入。

bootstrap 以 ``python -I -c <源码>`` 启动，源码是本模块的常量而不是磁盘文件，
工作区里的文件无法影响隔离逻辑。Landlock 不可用时由调用方决定降级还是
拒绝；两种情况下判分详情都会如实记录实际生效的隔离。

已知边界：守护进程化（setsid）的孙进程不受进程组清理约束；Landlock 只覆盖
TCP，不覆盖 UDP 等其他协议；同一 pytest 进程内的对抗性代码仍可伪造输出。
需要更强的边界时应使用容器。
"""
from __future__ import annotations

import glob
import json
import os
import platform
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from agentlab.runtime.exec import ExecOutcome, run_command
from agentlab.runtime.sandbox import UnsupportedSandbox

# Linux 通用系统调用号：landlock_create_ruleset / add_rule / restrict_self
_SYS_CREATE_RULESET, _SYS_ADD_RULE, _SYS_RESTRICT_SELF = 444, 445, 446
_ABI_WITH_TCP = 4
_FAIL_PREFIX = "agentlab-sandbox: "
_EXIT_SANDBOX_FAILED = 125

MODES = ("auto", "required", "off")
_BOOTSTRAP_ENV = {"PATH": "/usr/bin:/bin"}
_READ = ("READ_FILE", "READ_DIR")
_READ_EXECUTE = ("READ_FILE", "READ_DIR", "EXECUTE")
_WRITE = ("READ_FILE", "READ_DIR", "WRITE_FILE", "REMOVE_DIR", "REMOVE_FILE",
          "MAKE_DIR", "MAKE_REG", "MAKE_SYM", "REFER", "TRUNCATE")

# 常量本身在子进程里执行，保持自包含：不 import agentlab，不从磁盘读配置。
_BOOTSTRAP = r'''
import json, os, sys

policy = json.loads(sys.argv[1])
command = sys.argv[2:]
parent = os.getppid()


def fail(message):
    sys.stderr.write("agentlab-sandbox: " + message + "\n")
    sys.exit(125)


def owned_tasks():
    # RLIMIT_NPROC 按同一用户的线程数计，所以要把每个进程的 task 也数进去
    uid, count = os.getuid(), 0
    with os.scandir("/proc") as processes:
        for process in processes:
            if not process.name.isdigit():
                continue
            try:
                if process.stat(follow_symlinks=False).st_uid != uid:
                    continue
                with os.scandir("/proc/" + process.name + "/task") as threads:
                    count += sum(1 for _ in threads)
            except OSError:
                pass
    return count


def apply_rlimits():
    try:
        import resource
    except ImportError:
        return
    limits = dict(policy["rlimits"])
    if policy["extra_processes"]:
        try:
            limits["RLIMIT_NPROC"] = owned_tasks() + policy["extra_processes"]
        except OSError:
            pass
    for name, value in limits.items():
        resource_id = getattr(resource, name)
        _, hard = resource.getrlimit(resource_id)
        value = value if hard == resource.RLIM_INFINITY else min(value, hard)
        resource.setrlimit(resource_id, (value, value))


def apply_landlock():
    import ctypes

    fs_bits = {
        "EXECUTE": 1 << 0, "WRITE_FILE": 1 << 1, "READ_FILE": 1 << 2,
        "READ_DIR": 1 << 3, "REMOVE_DIR": 1 << 4, "REMOVE_FILE": 1 << 5,
        "MAKE_CHAR": 1 << 6, "MAKE_DIR": 1 << 7, "MAKE_REG": 1 << 8,
        "MAKE_SOCK": 1 << 9, "MAKE_FIFO": 1 << 10, "MAKE_BLOCK": 1 << 11,
        "MAKE_SYM": 1 << 12,
        # v2 起才有 REFER，v3 起才有 TRUNCATE
        "REFER": 1 << 13, "TRUNCATE": 1 << 14,
    }
    introduced = {"REFER": 2, "TRUNCATE": 3}
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long

    def syscall(number, *arguments):
        return libc.syscall(ctypes.c_long(number), *arguments)

    def supported(names, abi):
        return [name for name in names if introduced.get(name, 1) <= abi]

    abi = syscall(444, ctypes.c_void_p(None), ctypes.c_size_t(0), ctypes.c_uint32(1))
    if abi < 1:
        fail("landlock_create_ruleset: " + os.strerror(ctypes.get_errno()))

    class RulesetAttr(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64),
                    ("handled_access_net", ctypes.c_uint64)]

    class PathBeneath(ctypes.Structure):
        # struct landlock_path_beneath_attr：u64 + s32 + s32，本身没有内部填充
        _fields_ = [("allowed_access", ctypes.c_uint64),
                    ("parent_fd", ctypes.c_int32), ("reserved", ctypes.c_int32)]

    handled_fs = 0
    for _, names in policy["rules"]:
        for name in supported(names, abi):
            if name not in fs_bits:
                fail("unknown access right: " + name)
            handled_fs |= fs_bits[name]
    # 不添加任何端口规则，等于拒绝全部 TCP bind/connect
    handled_net = 3 if policy["block_tcp"] and abi >= 4 else 0

    attr = RulesetAttr(handled_fs, handled_net)
    ruleset = syscall(444, ctypes.byref(attr),
                      ctypes.c_size_t(16 if handled_net else 8), ctypes.c_uint32(0))
    if ruleset < 0:
        fail("landlock_create_ruleset: " + os.strerror(ctypes.get_errno()))
    try:
        for path, names in policy["rules"]:
            allowed = 0
            for name in supported(names, abi):
                allowed |= fs_bits[name]
            if not allowed:
                continue
            try:
                path_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            except OSError as error:
                fail("open " + path + ": " + str(error.strerror))
            try:
                rule = PathBeneath(allowed, path_fd)
                code = syscall(445, ctypes.c_long(ruleset), ctypes.c_long(1),
                               ctypes.byref(rule), ctypes.c_uint32(0))
                if code != 0:
                    fail("landlock_add_rule " + path + ": " + os.strerror(ctypes.get_errno()))
            finally:
                os.close(path_fd)
        # restrict_self 的前提：之后 exec 的 setuid 程序也无法提权
        if libc.prctl(ctypes.c_long(38), ctypes.c_ulong(1), ctypes.c_ulong(0),
                      ctypes.c_ulong(0), ctypes.c_ulong(0)) != 0:
            fail("prctl(PR_SET_NO_NEW_PRIVS): " + os.strerror(ctypes.get_errno()))
        if syscall(446, ctypes.c_long(ruleset), ctypes.c_uint32(0)) != 0:
            fail("landlock_restrict_self: " + os.strerror(ctypes.get_errno()))
    finally:
        os.close(ruleset)


if not command or not os.path.isabs(command[0]):
    fail("command must start with an absolute path")
apply_rlimits()
if policy["landlock"]:
    apply_landlock()
try:
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    # 判分进程意外退出时，被测进程随之被杀（不覆盖另行 setsid 的孙进程）
    libc.prctl(ctypes.c_long(1), ctypes.c_ulong(9), ctypes.c_ulong(0),
               ctypes.c_ulong(0), ctypes.c_ulong(0))
except OSError:
    pass
if os.getppid() != parent:
    sys.exit(125)
os.execve(command[0], command, policy["env"])
'''


class SandboxError(RuntimeError):
    """隔离无法建立。调用方必须按基础设施错误处理，不能退回无隔离执行。"""


@dataclass(frozen=True)
class SandboxProbe:
    landlock_abi: int          # 0 表示不可用
    reason: str

    @property
    def available(self) -> bool:
        return self.landlock_abi >= 1

    @property
    def blocks_tcp(self) -> bool:
        return self.landlock_abi >= _ABI_WITH_TCP


@dataclass
class SandboxPolicy:
    mode: str                                  # landlock | process
    workspace: Path
    scratch: Path                              # 唯一可写区域：HOME、临时文件、junit 都在这里
    probe: SandboxProbe | None = None
    cpu_seconds: int = 90
    address_space_bytes: int = 8 << 30
    file_size_bytes: int = 64 << 20
    extra_processes: int = 64
    env: dict[str, str] = field(default_factory=dict)
    rules: list[tuple[str, list[str]]] = field(default_factory=list)

    @property
    def landlock(self) -> bool:
        return self.mode == "landlock"

    def document(self) -> dict:
        """写入判分详情的摘要：说明实际生效了哪些隔离，而不是声明用了哪种。"""
        return {
            "isolation": self.mode,
            "landlock_abi": self.probe.landlock_abi if self.probe else 0,
            "workspace_readonly": self.landlock,
            "tcp_blocked": bool(self.probe and self.probe.blocks_tcp),
            "rlimits": {"cpu_seconds": self.cpu_seconds,
                        "address_space_bytes": self.address_space_bytes,
                        "file_size_bytes": self.file_size_bytes,
                        "extra_processes": self.extra_processes},
            "inherited_environment": False,
        }

    def command(self, command: list[str]) -> list[str]:
        payload = {
            "landlock": self.landlock,
            "block_tcp": True,
            "rules": [[path, sorted(set(names))] for path, names in self.rules],
            "rlimits": {"RLIMIT_CPU": self.cpu_seconds,
                        "RLIMIT_AS": self.address_space_bytes,
                        "RLIMIT_FSIZE": self.file_size_bytes,
                        "RLIMIT_CORE": 0},
            "extra_processes": self.extra_processes,
            "env": self.env,
        }
        return [sys.executable, "-I", "-c", _BOOTSTRAP, json.dumps(payload), *command]


def probe_landlock() -> SandboxProbe:
    """向内核询问 Landlock ABI 版本。非 Linux、旧内核或被 seccomp 拦截时返回 0。"""
    if sys.platform != "linux":
        return SandboxProbe(0, f"unsupported platform: {platform.system()}")
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    abi = libc.syscall(ctypes.c_long(_SYS_CREATE_RULESET), ctypes.c_void_p(None),
                       ctypes.c_size_t(0), ctypes.c_uint32(1))
    if abi < 1:
        return SandboxProbe(0, f"landlock_create_ruleset: {os.strerror(ctypes.get_errno())}")
    return SandboxProbe(int(abi), "ok")


def resolve_mode() -> str:
    """返回实际可用的隔离模式：landlock，或退化为 process。

    AGENTLAB_SANDBOX=auto（默认）在 Landlock 不可用时降级并告警；
    =required 或 AGENTLAB_REQUIRE_SANDBOX=1 时拒绝执行；=off 显式关闭 Landlock。
    """
    requested = os.environ.get("AGENTLAB_SANDBOX", "auto").strip().lower()
    if requested == "auto" and os.environ.get("AGENTLAB_REQUIRE_SANDBOX") == "1":
        requested = "required"
    if requested not in MODES:
        raise ValueError(f"AGENTLAB_SANDBOX must be one of {MODES}, got {requested!r}")
    if requested == "off":
        return "process"
    probe = probe_landlock()
    if probe.available:
        return "landlock"
    if requested == "required":
        raise UnsupportedSandbox(
            f"kernel sandbox required but unavailable: {probe.reason}")
    print(f"警告：Landlock 不可用（{probe.reason}），候选代码只在 rlimit + 清洁环境下执行；"
          "文件系统与网络未受限制。", file=sys.stderr, flush=True)
    return "process"


_SYSTEM_LIBRARY_DIRS = ("/lib", "/lib64", "/usr/lib", "/usr/lib64", "/usr/local/lib")


def _elf_interpreter(path: str) -> str | None:
    """ELF 可执行文件的 PT_INTERP：它需要的动态加载器的绝对路径。"""
    import struct

    with open(path, "rb") as handle:
        header = handle.read(64)
        if len(header) < 64 or header[:4] != b"\x7fELF":
            return None
        wide, little = header[4] == 2, header[5] == 1
        fmt = "<" if little else ">"
        size = "Q" if wide else "I"
        if wide:
            phoff = struct.unpack_from(fmt + "Q", header, 32)[0]
            phentsize, phnum = struct.unpack_from(fmt + "HH", header, 54)
        else:
            phoff = struct.unpack_from(fmt + "I", header, 28)[0]
            phentsize, phnum = struct.unpack_from(fmt + "HH", header, 42)
        for index in range(phnum):
            handle.seek(phoff + index * phentsize)
            entry = handle.read(phentsize)
            if struct.unpack_from(fmt + "I", entry, 0)[0] != 3:        # PT_INTERP
                continue
            offset = struct.unpack_from(fmt + size, entry, 8 if wide else 4)[0]
            length = struct.unpack_from(fmt + size, entry, 32 if wide else 16)[0]
            handle.seek(offset)
            return handle.read(length).rstrip(b"\0").decode()
    return None


def _loader_libraries(loader: str | None, binary: str) -> list[str]:
    """让动态加载器自己报告 `binary` 会映射哪些共享库，返回它们的绝对路径。"""
    import subprocess

    if not loader:
        return []
    try:
        completed = subprocess.run([loader, "--list", binary], capture_output=True,
                                   text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    return re.findall(r"=>\s*(/\S+)", completed.stdout)


@lru_cache(maxsize=1)
def _interpreter_rules() -> tuple[tuple[str, list[str]], ...]:
    """被测进程启动所需的解释器、标准库、site-packages 与共享库。

    解释器的布局在一个进程内不会变，所以只计算一次。
    """
    import sysconfig

    granted: dict[str, set[str]] = {}

    def grant(path, names):
        if path and os.path.isdir(path):
            granted.setdefault(os.path.realpath(path), set()).update(names)

    for path in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix,
                 *sys.path):
        grant(path, _READ)
    config = sysconfig.get_config_vars()
    executable = os.path.realpath(sys.executable)
    loader = _elf_interpreter(executable)
    # 给执行权限的目录：解释器的扩展模块目录、它的库目录、加载器所在目录，以及
    # 加载器为解释器和每个扩展模块实际解析出的共享库所在目录。只按 sysconfig 推断
    # 不够：独立构建或 conda 的 Python 有自己的 lib 目录，libc 却在系统目录里。
    library_dirs = {config.get("LIBDIR"), config.get("DESTSHARED")}
    if loader:
        library_dirs.add(os.path.dirname(os.path.realpath(loader)))
    extensions = sorted(glob.glob(os.path.join(config.get("DESTSHARED") or "", "*.so")))
    for binary in [executable, *extensions]:
        for library in _loader_libraries(loader, binary):
            library_dirs.add(os.path.dirname(os.path.realpath(library)))
    for path in filter(None, library_dirs):
        grant(path, _READ_EXECUTE)
    # 系统库目录只给读权限：少见的依赖仍能加载，git、ssh 等程序却不可执行
    for path in _SYSTEM_LIBRARY_DIRS:
        grant(path, _READ)
    rules = [(path, sorted(names)) for path, names in sorted(granted.items())]
    # 只有解释器和动态加载器本身可作为程序执行
    executables = {executable} | ({os.path.realpath(loader)} if loader else set())
    rules += [(path, ["READ_FILE", "EXECUTE"]) for path in sorted(executables)]
    # 不可变：这个结果被缓存并在整个进程里复用，调用方不能往里追加
    return tuple(rules)


_SELF_TEST_CODE = ("import bz2, ctypes, decimal, hashlib, json, lzma, sqlite3, ssl, "
                   "unicodedata, zlib; print('ok')")


def self_test(mode: str | None = None) -> tuple[bool, str]:
    """在沙箱里真实启动一次解释器并导入判分用到的库，返回 (是否通过, 说明)。

    规则表是按当前解释器的布局算出来的。换一种安装方式（conda、独立构建、容器
    镜像）之后解释器可能缺某个库目录，只有真正启动一次才能发现。
    """
    import tempfile

    with tempfile.TemporaryDirectory(prefix="agentlab-sandbox-") as temporary:
        workspace = Path(temporary) / "ws"
        workspace.mkdir()
        try:
            policy = build_policy(workspace, Path(temporary) / "scratch", mode=mode)
            outcome = run_sandboxed(policy, [sys.executable, "-B", "-c", _SELF_TEST_CODE],
                                    workspace, 60)
        except SandboxError as error:
            return False, str(error)
        if outcome.exit_status != "ok" or not outcome.stdout.strip().endswith("ok"):
            detail = (outcome.stderr or outcome.stdout).strip()[-300:] \
                or f"exit status {outcome.exit_status}"
            return False, f"{policy.mode} sandbox cannot start {sys.executable}: {detail}"
        return True, (f"{policy.mode}: {sys.executable} starts and imports the grading "
                      "stack inside the sandbox")


def build_policy(workspace: Path, scratch: Path, mode: str | None = None,
                 cpu_seconds: int = 90) -> SandboxPolicy:
    """为一次判分构造隔离策略。workspace 只读，scratch 是唯一可写目录。"""
    mode = mode or resolve_mode()
    if hasattr(os, "geteuid") and os.geteuid() == 0 \
            and os.environ.get("AGENTLAB_ALLOW_ROOT") != "1":
        # 以 root 运行模型生成的代码，出问题时波及整台机器；默认拒绝
        raise SandboxError("refusing to execute untrusted candidate code as root; "
                           "run as a normal user or set AGENTLAB_ALLOW_ROOT=1")
    workspace, scratch = Path(workspace).resolve(), Path(scratch).resolve()
    for name in ("home", "tmp"):
        (scratch / name).mkdir(parents=True, exist_ok=True, mode=0o700)

    import site

    env = {
        "PATH": f"{os.path.dirname(sys.executable)}:/usr/bin:/bin",
        "HOME": str(scratch / "home"),
        "TMPDIR": str(scratch / "tmp"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    # 只放行与结果无关的区域设置；其余变量（含各类 API key）一律不继承
    for name in ("LANG", "LC_ALL", "LC_CTYPE"):
        if os.environ.get(name):
            env[name] = os.environ[name]
    # HOME 被重定向后解释器找不到真实用户目录下的 site-packages，显式带回
    user_site = site.getusersitepackages()
    if os.path.isdir(user_site) and user_site in sys.path:
        env["PYTHONPATH"] = user_site

    # 每次都新建列表。解释器规则是进程内缓存的；早先的写法直接往缓存的那张表上追加
    # 本次的工作区规则，一个进程里判分几百次之后，策略参数就超过单个参数的上限，
    # 之后这个进程里的沙箱进程全都起不来（E2BIG）。
    rules = [*_interpreter_rules(), (str(workspace), list(_READ)), (str(scratch), list(_WRITE)),
             *((device, ["READ_FILE", "WRITE_FILE"]) for device in ("/dev/null",))]
    return SandboxPolicy(
        mode=mode, workspace=workspace, scratch=scratch,
        probe=probe_landlock() if mode == "landlock" else None,
        cpu_seconds=cpu_seconds, env=env,
        rules=[(path, names) for path, names in rules if os.path.exists(path)])


def run_sandboxed(policy: SandboxPolicy, command: list[str], cwd: Path,
                  timeout: float) -> ExecOutcome:
    """在隔离环境中运行命令；超时杀整个进程组（见 runtime.exec.run_command）。"""
    if not command or not os.path.isabs(command[0]):
        raise ValueError("sandboxed command must start with an absolute path")
    # bootstrap 自身也不继承父进程环境；被测命令的环境由 policy.env 在 execve 时给定
    outcome = run_command(policy.command(command), cwd, timeout, env=_BOOTSTRAP_ENV)
    if _FAIL_PREFIX in outcome.stderr and outcome.exit_status != "ok":
        reason = next((line[len(_FAIL_PREFIX):] for line in outcome.stderr.splitlines()
                       if line.startswith(_FAIL_PREFIX)), outcome.stderr[-300:])
        raise SandboxError(f"sandbox setup failed: {reason}")
    return outcome
