# -*- coding: utf-8 -*-
"""交给用户粘贴的六条命令（docs/POSTTRAIN-GPU.md 里带 deliver 标记的六个代码块：引导采样的启动、看进度、取结果，
合成出题的启动、看进度、取结果）的演练。

不是检查文档里有没有哪句话：命令从文档里取出来，原样写进一个脚本文件交给 bash 执行，看它做过什么。家目录换成
临时的：~/jupyterlab 里放着上传上来的仓库包（里面的脚本是假的）和旧包的 data、.env。启动命令用的 tar、pgrep、
nohup 都是真的，后台跑起来的是假的 run_rft_guided.sh、run_synth_pool.sh（往日志里写一行，然后像在跑一样待
30 秒）；看进度、取结果的四条命令对着同一个家目录里解压出来的那一包跑，进度面板是假的（只记下收到的参数），
合成出题的看进度另有一遍用仓库里真的面板。仓库包的目录名以文档为准：从命令里取出来，这里不写死。
"""
import contextlib
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC = "docs/POSTTRAIN-GPU.md"
# 进程还活着没有、命令行是什么，都从 /proc 里读
pytestmark = pytest.mark.skipif(not Path("/proc/self/stat").is_file() or not shutil.which("bash"),
                               reason="需要 /proc 和 Bash")

PACKAGE = re.compile(r"CodeAgent-R-v\d+")        # 仓库包的目录名：CodeAgent-R-v 加上数字
BOARD = "show_progress.sh"                       # 进度面板，两条看进度的命令调的都是它
ENV = ("TEACHER_API_KEY=teacher-key-for-tests\n"
       "TEACHER_BASE_URL=https://teacher.invalid/v1\n"
       "TEACHER_MODEL=teacher-model-for-tests\n")
OTHER_ENV = ENV.replace("teacher-key-for-tests", "teacher-key-in-another-package") + "TEACHER_TIMEOUT=240\n"
# 终端的 locale 各有各的：一种照机器上的，一种是 C。按字截结果行、旧包目录名的先后次序都受它影响，受影响的用例两种都演
LOCALES = [pytest.param(None, id="terminal-locale"), pytest.param("C", id="LC_ALL=C")]

FAKE_RUNNER = r'''#!/usr/bin/env bash
# 假的启动脚本（演练用）：先把自己的 PID 追加到 @STARTS@（每启动一次占一行），往日志里写一行——就是往 stdout 打印，
# 日志是启动命令用重定向开出来的——然后像在跑一样待 30 秒
echo "$$" >> @STARTS@
echo "@LINE@"
sleep 30
'''

FAKE_BOARD = r'''#!/usr/bin/env bash
# 假的 show_progress.sh（演练用）：和真的一样先回到仓库根目录，但不看日志、不等待。收到的参数原样记进 @ARGS@
# （参数之间用 NUL 隔开，空串也占一格），它看到的 WATCH 和所在目录记进 @ENV@
cd "$(dirname "$0")/.."
: > @ARGS@
for argument in "$@"; do printf '%s\0' "$argument" >> @ARGS@; done
printf 'WATCH=%s\nPWD=%s\n' "${WATCH-unset}" "$PWD" > @ENV@
'''


@dataclass(frozen=True)
class Delivery:
    """一组交付命令（启动、看进度、取结果）围着转的后台脚本和它的日志。文档里的标记是 <!-- deliver:<组名>-<哪一条> -->。

    命令里的仓库包目录名以文档为准，从命令里取；脚本名、日志名和交给面板的参数以脚本为准，写在这里。"""
    key: str            # 标记里的组名
    script: str         # 启动命令在后台跑起来的脚本（仓库包 scripts/ 下的文件名）；看进度的命令把它当进程名交给面板
    log: str            # 启动命令给这个脚本开的日志（仓库包的根目录下）；看进度、取结果的两条命令读它
    verdicts: str       # 看进度的命令交给面板的结果行正则
    must_show: tuple    # 取结果的命令的输出里必须见到的几种行


GUIDED = Delivery("guided", "run_rft_guided.sh", "rft-guided.log", "", ("=====", "结果："))
SYNTH = Delivery("synth", "run_synth_pool.sh", "synth-batch1.log", "结果：|接受|拒绝", ("=====", "结果：", "接受"))
DELIVERIES = (GUIDED, SYNTH)
MARKERS = tuple(f"{delivery.key}-{step}" for delivery in DELIVERIES for step in ("launch", "progress", "results"))


def fake_line(delivery):
    """delivery 的假脚本启动后写进日志的那一行。"""
    return f"假的 {delivery.script} 开始跑了"


def names_in(command):
    """这条命令里出现的仓库包目录名，按出现的先后，不重复。"""
    return list(dict.fromkeys(PACKAGE.findall(command)))


def unpacked_by(command):
    """引导采样的启动命令解压出来的那一包：命令里那个压缩包（CodeAgent-R-v数字.tar.gz）的名字。"""
    found = re.findall(r"(CodeAgent-R-v\d+)\.tar\.gz", command)
    assert len(found) == 1, f"{DOC} 的引导采样启动命令里应当正好解压一个仓库包，现在写着 {found}"
    return found[0]


def data_from(command):
    """引导采样的启动命令把 data 接过来的那一包（旧包）：命令里除了解压出来的这一包，正好提到一个别的包。"""
    unpacked = unpacked_by(command)
    others = [name for name in names_in(command) if name != unpacked]
    assert len(others) == 1, (f"{DOC} 的引导采样启动命令里，除了解压出来的 {unpacked} 以外应当正好提到一个旧包"
                              f"（data 从它那里接过来），现在是 {others}")
    return others[0]


def version(name):
    """仓库包目录名里的版本号：CodeAgent-R-v19 是 19。"""
    return int(name[len("CodeAgent-R-v"):])


def neighbouring(lab):
    """另一个旧包的目录名：版本号比数据接过来的那一包大 1（那正是这一包的话大 2）。演"别的旧包里有 .env"用。"""
    gap = 2 if version(lab.older) + 1 == version(lab.unpacked) else 1
    return f"CodeAgent-R-v{version(lab.older) + gap}"


def delivered(name):
    """标记 <!-- deliver:name --> 后面紧跟着的那个 bash 代码块：用户要原样粘贴的命令。标记缺失，或者后面紧跟的不是
    bash 代码块，都让测试失败，并说明缺的是哪一个。"""
    marker = f"<!-- deliver:{name} -->"
    lines = (ROOT / DOC).read_text(encoding="utf-8").split("\n")
    found = [index for index, line in enumerate(lines) if line.strip() == marker]
    if len(found) != 1:
        pytest.fail(f"{DOC} 里标记 {marker} 应当正好出现一次，现在是 {len(found)} 次")
    rest = lines[found[0] + 1:]
    while rest and not rest[0].strip():                       # 标记和代码块之间可以隔空行
        rest.pop(0)
    if not rest or not rest[0].startswith("```"):
        pytest.fail(f"{DOC} 里标记 {marker} 后面没有紧跟代码块：缺了交给用户粘贴的命令")
    if rest[0].strip() != "```bash":
        pytest.fail(f"{DOC} 里标记 {marker} 后面紧跟的代码块不是 bash（围栏写的是 {rest[0].strip()}）")
    body = []
    for line in rest[1:]:
        if line.startswith("```"):
            return "\n".join(body)
        body.append(line)
    pytest.fail(f"{DOC} 里标记 {marker} 后面的代码块没有收尾的围栏")


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def eventually(condition, seconds=10.0):
    """等 condition 成立；到时间还不成立就返回 False（后台起来的那一份写日志要一点时间）。"""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return condition()


def holds(condition, seconds=1.0):
    """condition 在这段时间里一直是真的才算数。"""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not condition():
            return False
        time.sleep(0.02)
    return condition()


def running(pid):
    """这个 PID 还在跑吗（已经退出、还没被收回的僵尸不算）。"""
    try:
        state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state != "Z"


def cmdline(pid):
    """这个进程的命令行（各参数用空格连起来，同 pgrep -f 看到的一样）；进程不在了就是空串。"""
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8").strip()
    except OSError:
        return ""


def in_sessions(lab):
    """还在机器上的、属于这次演练哪一回粘贴的进程，按 PID。/proc/<pid>/stat 里进程名后面第 4 个字段是会话号：每一回
    粘贴单独占一个会话（见 paste），它起的后台脚本和脚本派生的 sleep 都在这个会话里。僵尸不算（已经退出，等人收回）。"""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:                                                    # 进程名放在括号里，里面可能有空格和括号
            fields = (entry / "stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        except OSError:                                          # 这个进程刚好退出了
            continue
        if len(fields) > 3 and fields[0] != "Z" and int(fields[3]) in lab.sessions:
            found.append(int(entry.name))
    return found


def started(lab, delivery):
    """到目前为止起过的假脚本（delivery 这一组的），每次启动一个 PID，按先后。"""
    path = lab.starts(delivery)
    if not path.exists():
        return []
    return [int(item) for item in path.read_text(encoding="utf-8").split()]


def stop_started(lab):
    """把这次演练起过的假脚本都关掉（连同它派生的 sleep），等它们都不在了才返回。清理时也用它：断言失败也不把它们
    留在机器上。按会话找，不按假脚本自己记下的 PID 找：刚起的那一份可能还没来得及把自己的 PID 记进文件，测试就
    已经失败了——它会待满 30 秒，后面要用启动命令的用例只能等它。"""
    def finish():
        for pid in in_sessions(lab):
            with contextlib.suppress(ProcessLookupError):         # 上一次已经杀掉了
                os.kill(pid, signal.SIGKILL)
        return not in_sessions(lab)

    return eventually(finish, seconds=5.0)


def need_tools(*names):
    """交付的命令自己要用到的命令行工具，有找不到的就不演了。"""
    missing = [name for name in names if not shutil.which(name)]
    if missing:
        pytest.skip(f"需要 {'、'.join(missing)}")


def strangers(script):
    """机器上命令行里带着这个脚本名的进程：pgrep -af 的原样输出，一行一个。演练自己起的假脚本也在里面，调用处自己分。"""
    listing = subprocess.run(["pgrep", "-af", script], capture_output=True, text=True)
    return [line for line in listing.stdout.splitlines() if line.strip()]


def need_quiet_machine(*scripts):
    """启动命令靠 pgrep 判断"已经在运行"：谁的命令行里带着脚本名都算。机器上本来就有这样的进程时（别的会话在跑这份
    脚本，或者有人正对着它跑命令），同一条命令只会回答"已经在运行"，演不了全新启动——等一会儿，还在就跳过并说明是谁。"""
    def busy():
        return [line for script in scripts for line in strangers(script)]

    if not eventually(lambda: not busy(), seconds=15.0):
        pytest.skip(f"机器上有命令行里带着 {'、'.join(scripts)} 的进程（不是这次演练起的），启动命令会把它们当成已经在"
                    f"运行：\n" + "\n".join(busy()))


def ours_running(lab, delivery):
    """这次演练起过的假脚本（delivery 这一组的）里还在跑的。"""
    return [pid for pid in started(lab, delivery) if delivery.script in cmdline(pid)]


def start_fresh(lab, delivery, command, reset=None, locale=None):
    """贴一次启动命令，要的是它全新启动（回答"已启动"），返回它的输出。

    启动命令靠 pgrep 认进程：机器上别的进程（命令行里带着脚本名，不是这次演练起的）正好在的时候，它只会回答
    "已经在运行"、什么都不做。被这样挡住就等那些进程走开再贴（最多等 15 秒），还不行就跳过并说明是谁。贴过一次，
    这一包就不是没贴过的样子了（引导采样的那条会解压、接数据、拷 .env），所以再贴以前先让 reset(lab) 把它恢复原样。
    没有别的进程还回答"已经在运行"，是命令把自己也算成了在运行：原样交回去，让调用处的断言失败。"""
    deadline = time.monotonic() + 15.0
    while True:
        result = paste(lab, command, locale)
        if "已经在运行" not in result.stdout or ours_running(lab, delivery):
            return result
        found = strangers(delivery.script)
        if not found:
            return result
        if time.monotonic() > deadline:
            why = (f"贴启动命令时机器上有命令行里带着 {delivery.script} 的进程（不是这次演练起的），15 秒内没走开，"
                   f"启动命令一直把它当成已经在运行：\n" + "\n".join(found))
            pytest.skip(why)
        if reset:
            reset(lab)
        eventually(lambda: not strangers(delivery.script), seconds=max(deadline - time.monotonic(), 0.0))


@dataclass
class Lab:
    """一次演练用的家目录。仓库包要先打包（pack）才有，解压出来是 ~/jupyterlab/<这一包>。"""
    home: Path
    unpacked: str       # 这一包的目录名（引导采样的启动命令解压出来的那个），从命令里取的
    older: str          # 数据接过来的那一包的目录名，同上
    notes: Path         # 假脚本、假面板记东西的目录：放在 ~/jupyterlab 外面，不算用户目录布局的一部分
    sessions: list      # 到目前为止每一回粘贴的会话号：清理时按它找出这次演练起过的所有进程

    @property
    def jupyterlab(self):
        return self.home / "jupyterlab"

    @property
    def package(self):
        return self.jupyterlab / self.unpacked

    @property
    def previous(self):
        return self.jupyterlab / self.older

    @property
    def tarball(self):
        return self.jupyterlab / f"{self.unpacked}.tar.gz"

    @property
    def board_args(self):
        """假的进度面板记参数的文件。"""
        return self.notes / "board.args"

    @property
    def board_env(self):
        """假的进度面板记 WATCH 和所在目录的文件。"""
        return self.notes / "board.env"

    def starts(self, delivery):
        """delivery 那一组：假脚本每次启动时追加自己 PID 的文件。"""
        return self.notes / f"{delivery.key}-started-pids.txt"

    def log(self, delivery):
        """启动命令给 delivery 的脚本开的日志。"""
        return self.package / delivery.log


@pytest.fixture
def lab(tmp_path):
    """临时的家目录：~/jupyterlab 里只有数据接过来的那一包（旧包，里面是第 1 轮的 data 和 .env）。仓库包由各个用例
    自己打包，缺了它正好演失败路径。"""
    launch = delivered("guided-launch")
    made = Lab(home=tmp_path / "home", unpacked=unpacked_by(launch), older=data_from(launch),
               notes=tmp_path / "rehearsal", sessions=[])
    made.notes.mkdir()
    write(made.previous / "data/rft-round1/sampling/train-manifest.json", '{"zero_pass_tasks": ["h3_t1"]}\n')
    write(made.previous / ".env", ENV)
    yield made
    stop_started(made)


def pack(lab, keep=100, real_board=False):
    """像发布时那样打包：~/jupyterlab/pkg/<这一包>/scripts/ 里放上两个假的启动脚本和进度面板，打成
    ~/jupyterlab/<这一包>.tar.gz，再删掉打包用的目录（仓库包只能靠解压得到）。keep 是百分比：只留压缩包的前这么多
    字节（上传没传完）。real_board：进度面板放仓库里真的那一个，给要看面板真正显示什么的那条用例。"""
    scripts = lab.jupyterlab / "pkg" / lab.unpacked / "scripts"
    for delivery in DELIVERIES:
        write(scripts / delivery.script, FAKE_RUNNER.replace("@STARTS@", shlex.quote(str(lab.starts(delivery))))
              .replace("@LINE@", fake_line(delivery)))
    if real_board:
        shutil.copy(ROOT / "scripts" / BOARD, scripts / BOARD)
    else:
        write(scripts / BOARD, FAKE_BOARD.replace("@ARGS@", shlex.quote(str(lab.board_args)))
              .replace("@ENV@", shlex.quote(str(lab.board_env))))
    subprocess.run(["tar", "-czf", str(lab.tarball), lab.unpacked], cwd=scripts.parents[1], check=True,
                   capture_output=True)
    shutil.rmtree(lab.jupyterlab / "pkg")
    if keep < 100:
        data = lab.tarball.read_bytes()
        lab.tarball.write_bytes(data[:len(data) * keep // 100])


def unpack(lab):
    """解压到 ~/jupyterlab：贴过引导采样的启动命令以后的样子（启动命令自己怎么解压由它那个用例演）。"""
    subprocess.run(["tar", "-xzf", lab.tarball.name], cwd=lab.jupyterlab, check=True, capture_output=True)


def remove_package(lab):
    """让这一包回到没贴过启动命令的样子：解压出来的整个目录删掉（压缩包和旧包不动）。"""
    shutil.rmtree(lab.package, ignore_errors=True)


def environment(lab, locale=None):
    """交给粘贴命令的环境：HOME 指向临时的家目录，WATCH 给 0（不用真等刷新）。bash、tar 启动时会读进去的几个变量
    不带进来——用户把它们贴进自己终端的时候没有这些（BASH_ENV 会让非交互的 bash 先 source 一个文件）。
    locale 给了就用它当 LC_ALL：终端未必设着 UTF-8 的 locale（服务不是从登录 shell 起的时候就没有）。"""
    clean = {name: value for name, value in os.environ.items()
             if name not in ("BASH_ENV", "ENV", "SHELLOPTS", "BASHOPTS", "CDPATH", "TAR_OPTIONS", "WATCH")}
    return {**clean, "HOME": str(lab.home), "WATCH": "0", **({"LC_ALL": locale} if locale else {})}


def paste(lab, command, locale=None, seconds=60.0):
    """像用户那样把一条命令交给 bash 执行，返回它的输出。命令原样写进一个脚本文件再运行，不放进命令行：启动命令用
    pgrep -f 找命令行里带着脚本名的进程，bash -c '…' 会匹配到它自己。

    这一回粘贴单独占一个会话（会话号记在 lab.sessions 里，清理时按它找）：过了 seconds 秒还没结束，就把整个会话
    一起收拾掉再判失败。真的进度面板在日志还没有、脚本却像在跑的时候会一直刷新下去，只杀最外面那个 bash 的话，
    面板和它的 sleep 会留在机器上。"""
    script = lab.home / "paste.sh"                            # 放在 ~/jupyterlab 外面：不是用户目录布局的一部分
    write(script, command + "\n")
    process = subprocess.Popen(["bash", str(script)], cwd=lab.home, env=environment(lab, locale),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
                               text=True, encoding="utf-8", errors="replace")
    lab.sessions.append(process.pid)                          # 新会话的会话号就是它的 PID
    try:
        out, err = process.communicate(timeout=seconds)
    except subprocess.TimeoutExpired:
        stop_started(lab)
        out, err = process.communicate(timeout=10)
        pytest.fail(f"这条命令过了 {seconds:g} 秒还没有结束。它到这时打出来的：\n{out}{err}")
    return subprocess.CompletedProcess(process.args, process.returncode, out, err)


def board_arguments(lab):
    """假的进度面板记下来的参数。"""
    data = lab.board_args.read_bytes() if lab.board_args.exists() else b""
    return [item.decode("utf-8") for item in data.split(b"\0")[:-1]] if data else []


def board_environment(lab):
    """假的进度面板记下来的 (WATCH, 所在目录)。"""
    settings = dict(line.split("=", 1) for line in lab.board_env.read_text(encoding="utf-8").splitlines())
    return settings["WATCH"], settings["PWD"]


def log_text(lab, delivery):
    return lab.log(delivery).read_text(encoding="utf-8") if lab.log(delivery).is_file() else None


@pytest.mark.parametrize("name", MARKERS, ids=MARKERS)
def test_each_delivered_command_is_one_line_of_bash_right_after_its_marker(name):
    """网页终端里粘贴：标记后面紧跟的必须是一个 bash 代码块，而且只有一行——多行的贴进去会被拆开、逐行执行。"""
    command = delivered(name)
    assert command.strip(), f"{DOC} 里 {name} 的代码块是空的"
    assert "\n" not in command, f"{DOC} 里 {name} 的命令不止一行"


def test_the_other_five_commands_work_in_the_package_the_guided_launch_command_unpacks():
    """引导采样的启动命令把上传的压缩包解压成 ~/jupyterlab/<这一包>。另外五条命令都得对着这同一个目录：合成出题的
    启动命令自己不解压（文档让它后贴），四条看进度、取结果的命令读的是这里的日志。目录名各自从命令里取出来比。"""
    launch = delivered("guided-launch")
    unpacked, older = unpacked_by(launch), data_from(launch)
    assert PACKAGE.fullmatch(unpacked) and PACKAGE.fullmatch(older) and unpacked != older
    for name in MARKERS:
        if name == "guided-launch":
            continue
        names = names_in(delivered(name))
        assert names == [unpacked], f"{DOC} 里 {name} 的命令写的仓库包是 {names}，启动命令解压出来的是 {unpacked}"


def test_launch_command_unpacks_links_data_copies_the_env_and_starts_in_the_background(lab):
    """全新贴一遍，趁它还在跑再贴一遍，它停了以后再贴一遍。"""
    need_tools("tar", "pgrep", "nohup")
    need_quiet_machine(GUIDED.script)
    pack(lab)
    command = delivered("guided-launch")
    line = fake_line(GUIDED)

    first = start_fresh(lab, GUIDED, command, reset=remove_package)
    assert first.returncode == 0, first.stderr
    # 解压、接数据都不出声；.env 是从旧包里找到拷来的，说一声是哪一个；出了别的声就是出了错
    assert first.stdout == f"已从 ../{lab.older}/.env 拷来 .env\n已启动\n" and first.stderr == ""
    assert eventually(lambda: log_text(lab, GUIDED) == line + "\n")     # 后台那一份写日志要一点时间
    # 第 1 轮的产物：data 是指向旧包的符号链接（相对路径），透过它看得到旧包里的东西
    link = lab.package / "data"
    assert link.is_symlink() and os.readlink(link) == f"../{lab.older}/data"
    assert (link / "rft-round1/sampling/train-manifest.json").is_file()
    # 教师的密钥：.env 拷了一份过来（是拷贝，不是链接；旧包的还在）
    copied = lab.package / ".env"
    assert not copied.is_symlink() and copied.read_bytes() == ENV.encode("utf-8")
    assert (lab.previous / ".env").read_bytes() == ENV.encode("utf-8")
    pids = started(lab, GUIDED)
    assert len(pids) == 1 and running(pids[0])
    assert cmdline(pids[0]) == f"bash scripts/{GUIDED.script}"           # 后台跑的是这一包里的那一份脚本

    # 趁它还在跑再贴一遍：不起第二个，正在跑的那份日志原样
    again = paste(lab, command)
    assert again.returncode == 0, again.stderr
    assert again.stdout == "已经在运行，不重复启动\n" and again.stderr == ""
    assert holds(lambda: started(lab, GUIDED) == pids)                # 再起一个的话，它一启动就先把自己的 PID 记下来
    assert running(pids[0]) and log_text(lab, GUIDED) == line + "\n"

    # 它停了以后再贴：重新启动，日志重写。.env 已经有了就不动——旧包的 .env 后来改了，也不跟过来
    assert stop_started(lab)
    with lab.log(GUIDED).open("a", encoding="utf-8") as handle:
        handle.write("上一次跑到一半留下的最后一行\n")
    write(lab.previous / ".env", ENV.replace("teacher-key-for-tests", "teacher-key-rotated"))
    third = start_fresh(lab, GUIDED, command)
    assert third.returncode == 0, third.stderr
    assert third.stdout == "已启动\n" and third.stderr == ""            # .env 已经有了：不用再找，也不多说什么
    assert eventually(lambda: len(started(lab, GUIDED)) == 2 and log_text(lab, GUIDED) == line + "\n")
    assert os.readlink(link) == f"../{lab.older}/data" and copied.read_bytes() == ENV.encode("utf-8")


@pytest.mark.parametrize("trouble", ["missing", "truncated"])
def test_launch_command_fails_before_starting_anything_when_the_tarball_cannot_be_read(lab, trouble):
    """压缩包没传上去，或者没传完：解压这一步就停下，什么都没启动，报错留在屏幕上。"""
    need_tools("tar", "pgrep", "nohup")
    if trouble == "truncated":
        pack(lab, keep=50)
    result = paste(lab, delivered("guided-launch"))
    assert result.returncode != 0
    assert "已启动" not in result.stdout and "已经在运行" not in result.stdout
    assert result.stderr.strip()                               # 报错留在屏幕上：用户看得出停在哪一步
    assert started(lab, GUIDED) == [] and log_text(lab, GUIDED) is None
    if trouble == "missing":
        assert f"{lab.unpacked}.tar.gz" in result.stderr         # 找不到压缩包时报错里带着它的名字
        assert not lab.package.exists()


@pytest.mark.parametrize("locale", LOCALES)
@pytest.mark.parametrize("both", [pytest.param(False, id="only-another-old-package-has-it"),
                                  pytest.param(True, id="both-old-packages-have-it")])
def test_launch_command_copies_the_env_from_whichever_old_package_has_one(lab, both, locale):
    """旧包里的 .env：启动命令到 ~/jupyterlab/CodeAgent-R-*/ 里找，哪个旧包里有就从它那里拷一份再启动，拷来的一字
    不差，也不凭空造一个。数据接过来的那一包里没有 .env 时（教师的参数当时靠环境变量给过）也不能卡在这一步。
    两个旧包里都有时（内容不同）取目录名排在最后的那个，旧包里的两份都不动。先后是 bash 按终端的 locale 排出来的，
    所以两种 locale 都演。"""
    need_tools("tar", "pgrep", "nohup")
    need_quiet_machine(GUIDED.script)
    pack(lab)
    second = lab.jupyterlab / neighbouring(lab)
    if not both:
        (lab.previous / ".env").unlink()
    write(second / ".env", OTHER_ENV)
    source = max(lab.older, second.name)                        # 多个旧包里都有时取目录名排在最后的那个
    contents = {lab.older: ENV, second.name: OTHER_ENV}

    result = start_fresh(lab, GUIDED, delivered("guided-launch"), reset=remove_package, locale=locale)
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"已从 ../{source}/.env 拷来 .env\n已启动\n" and result.stderr == ""
    assert eventually(lambda: log_text(lab, GUIDED) == fake_line(GUIDED) + "\n")
    copied = lab.package / ".env"
    assert not copied.is_symlink() and copied.read_bytes() == contents[source].encode("utf-8")
    assert (second / ".env").read_bytes() == OTHER_ENV.encode("utf-8")
    if both:
        assert (lab.previous / ".env").read_bytes() == ENV.encode("utf-8")
    else:
        assert not (lab.previous / ".env").exists()
    assert len(started(lab, GUIDED)) == 1
    assert os.readlink(lab.package / "data") == f"../{lab.older}/data"


def test_launch_command_starts_nothing_and_says_so_when_no_old_package_has_an_env_file(lab):
    """~/jupyterlab 里哪个 CodeAgent-R-* 都没有 .env：解压、把 data 接过来，然后停下——不启动、退出码不是 0，
    说清楚把 .env 放到哪一包里再粘一次，也不凭空造一个。照它说的把 .env 放进去再粘，就启动了。"""
    need_tools("tar", "pgrep", "nohup")
    need_quiet_machine(GUIDED.script)
    pack(lab)
    (lab.previous / ".env").unlink()
    command = delivered("guided-launch")
    refused = paste(lab, command)
    assert refused.returncode != 0
    assert "没有找到 .env" in refused.stdout and f"~/jupyterlab/{lab.unpacked}/" in refused.stdout
    assert "已启动" not in refused.stdout and "已经在运行" not in refused.stdout and refused.stderr == ""
    assert started(lab, GUIDED) == [] and log_text(lab, GUIDED) is None
    assert not os.path.lexists(lab.package / ".env")
    assert os.readlink(lab.package / "data") == f"../{lab.older}/data"      # 停下以前做完的：解压、接数据
    again = paste(lab, command)
    assert again.returncode != 0 and again.stdout == refused.stdout       # 不放 .env 就一直停在这里
    assert started(lab, GUIDED) == []

    # 照它说的做：把 .env 放进这一包再粘。已经有了就不找了，原样启动
    write(lab.package / ".env", OTHER_ENV)
    launched = start_fresh(lab, GUIDED, command)
    assert launched.returncode == 0, launched.stderr
    assert launched.stdout == "已启动\n" and launched.stderr == ""
    assert eventually(lambda: log_text(lab, GUIDED) == fake_line(GUIDED) + "\n")
    assert (lab.package / ".env").read_bytes() == OTHER_ENV.encode("utf-8")


def test_synth_launch_command_starts_the_pool_script_in_the_background_and_a_second_paste_starts_nothing_else(lab):
    """引导采样的启动命令贴过以后（仓库包解压出来了）再贴合成出题的启动命令：全新贴一遍，趁它还在跑再贴一遍，
    它停了以后再贴一遍。"""
    need_tools("pgrep", "nohup")
    need_quiet_machine(SYNTH.script)
    command = delivered("synth-launch")
    line = fake_line(SYNTH)

    # 仓库包还没解压出来（引导采样的启动命令没贴过，或者没成功）：进不了仓库目录，什么都没启动
    missing = paste(lab, command)
    assert missing.returncode != 0 and str(lab.package) in missing.stderr
    assert "已启动" not in missing.stdout and "已经在运行" not in missing.stdout
    assert started(lab, SYNTH) == [] and log_text(lab, SYNTH) is None

    pack(lab)
    unpack(lab)
    first = start_fresh(lab, SYNTH, command)
    assert first.returncode == 0, first.stderr
    assert first.stdout == "已启动\n" and first.stderr == ""
    assert eventually(lambda: log_text(lab, SYNTH) == line + "\n")       # 后台那一份写日志要一点时间
    pids = started(lab, SYNTH)
    assert len(pids) == 1 and running(pids[0])
    assert cmdline(pids[0]) == f"bash scripts/{SYNTH.script}"            # 后台跑的是这一包里的那一份脚本

    # 趁它还在跑再贴一遍：不起第二个，正在跑的那份日志原样
    again = paste(lab, command)
    assert again.returncode == 0, again.stderr
    assert again.stdout == "已经在运行，不重复启动\n" and again.stderr == ""
    assert holds(lambda: started(lab, SYNTH) == pids)                   # 再起一个的话，它一启动就先把自己的 PID 记下来
    assert running(pids[0]) and log_text(lab, SYNTH) == line + "\n"

    # 它停了以后再贴：重新启动，日志重写
    assert stop_started(lab)
    with lab.log(SYNTH).open("a", encoding="utf-8") as handle:
        handle.write("上一次跑到一半留下的最后一行\n")
    third = start_fresh(lab, SYNTH, command)
    assert third.returncode == 0, third.stderr
    assert third.stdout == "已启动\n" and third.stderr == ""
    assert eventually(lambda: len(started(lab, SYNTH)) == 2 and log_text(lab, SYNTH) == line + "\n")


@pytest.mark.parametrize("delivery", DELIVERIES, ids=[delivery.key for delivery in DELIVERIES])
def test_progress_command_points_the_panel_at_the_log_the_launch_command_wrote(lab, delivery):
    """看进度的命令要在启动命令解压出来的那一包里跑，把日志名、结果行的正则和进程名这三个参数交给面板。"""
    need_tools("tar")
    # 还没解压（启动命令没贴过，或者没成功）就来看进度：进不了仓库目录，面板不会被调用
    failed = paste(lab, delivered(f"{delivery.key}-progress"))
    assert failed.returncode != 0 and str(lab.package) in failed.stderr
    assert not lab.board_args.exists()
    pack(lab)
    unpack(lab)
    write(lab.log(delivery), fake_line(delivery) + "\n")                 # 启动命令跑过以后留下的日志
    # 记录是忠实的：直接调一次假面板，空串也原样记下来
    subprocess.run(["bash", str(lab.package / "scripts" / BOARD), "甲", "", "丙"], env=environment(lab), check=True)
    assert board_arguments(lab) == ["甲", "", "丙"]

    shown = paste(lab, delivered(f"{delivery.key}-progress"))
    assert shown.returncode == 0, shown.stderr
    # 三个参数照文档里写的传到：合成出题这一组就是 synth-batch1.log、'结果：|接受|拒绝'、run_synth_pool.sh
    assert board_arguments(lab) == [delivery.log, delivery.verdicts, delivery.script]
    watch, where = board_environment(lab)
    assert Path(where) == lab.package                          # 面板站在解压出来的仓库目录里
    # 它要看的正是启动命令写的那份日志
    assert (Path(where) / delivery.log).read_text(encoding="utf-8") == fake_line(delivery) + "\n"
    # 刷新的间隔是命令自己带的（外面给的是 0：只显示一次）
    assert watch.isdigit() and int(watch) > 0


def test_synth_progress_command_shows_the_steps_and_each_familys_verdict_on_the_real_panel(lab):
    """看进度的那条命令交给面板的三个参数（日志名、结果行的正则、进程名）对着仓库里真的 show_progress.sh 再跑一遍。
    跑完的那一批显示"已完成"、五步都标着完成，"已出的结果"是结果行和每个题族一行的接受、拒绝（只留最后 14 行）；
    第 2 步没做完就停下的那份日志，脚本已经不在了，面板说"没有在运行"并自己退出，不让人干等下一次刷新。"""
    need_tools("tar", "grep", "cut", "pgrep")
    pack(lab, real_board=True)
    unpack(lab)
    command = delivered("synth-progress")
    log = synth_log()
    write(lab.log(SYNTH), "".join(line + "\n" for line, _ in log))
    done = paste(lab, command, seconds=20)                        # 跑完的那一批：面板显示一次就退出，不该等下一次刷新
    assert done.returncode == 0, done.stderr
    assert f"进度：{SYNTH.log}" in done.stdout and "状态: 已完成" in done.stdout
    for step in ("1/5 检查前提", "2/5 向教师要题族并逐个审计（每族最多 2 轮）", "3/5 汇总这一池",
                 "4/5 抽查 3 个已接受的题族", "5/5 并入阶梯", "全部结束"):
        assert f"[完成]   {step}" in done.stdout
    assert LAST_VERDICT in done.stdout                          # "当前进度"是最后几行进度条
    results = done.stdout.split("已出的结果\n", 1)[1].split("\n日志最后 3 行", 1)[0]
    rows = [row.strip() for row in results.splitlines() if row.strip()]
    verdicts = [line for line, kept in log if kept and not line.startswith("=====")]
    # 只认 结果：/接受/拒绝 这几种行，只留最后 14 行
    assert [row[:40] for row in rows] == [line[:40] for line in verdicts[-14:]]

    stopped = log[:next(index for index, (line, _) in enumerate(log) if line.startswith("===== 3/5"))]
    write(lab.log(SYNTH), "".join(line + "\n" for line, _ in stopped))
    # 这一遍面板靠进程名判断脚本在不在：机器上有同名的别的进程时，它会一直"进行中"、等着刷新，演不了
    need_quiet_machine(SYNTH.script)
    shown = paste(lab, command, seconds=20)
    assert shown.returncode == 0, shown.stderr
    assert "状态: 没有在运行" in shown.stdout and "状态: 已完成" not in shown.stdout
    assert "[完成]   1/5 检查前提" in shown.stdout
    assert "[进行中] 2/5 向教师要题族并逐个审计（每族最多 2 轮）" in shown.stdout


def long_cjk_line():
    """一行很长的结果行（没写成的题和原因），汉字为主，超过 240 个字。"""
    return "结果：技能提示没写成的题和原因：" + "、".join(
        f"h3_p{index:02d}_001 教师接口超时，120 秒内没有回复" for index in range(1, 10))


def long_ids_line():
    """一行很长的结果行（对不上的题号），ASCII 为主，超过 240 个字符。"""
    return ("结果：训练时实际加载的题 D-prefix 和 F-control 不一样：只在 D-prefix 里的 "
            + str([f"h3_p{index:02d}_001" for index in range(1, 16)]) + "。")


def guided_log():
    """一份像模像样的日志：可用的技能太少、降到 11 步的一轮，探针那一组的自检没过。每行标明取结果的命令要不要带出来。"""
    return [
        ("===== 1/13 检查前提 =====", True),                                    # 步骤标记
        ("任务文件：6 题，train 3 / dev 2 / test 1；其中难题来源族 train 2 / dev 1", False),
        ("探针题：上一轮零通过的 train 题里属于 hard、variant 层的共 72 道，写在 data/rft-guided/probe-ids.txt", False),
        ("===== 3/13 启动 vLLM 并先评基座 =====", True),
        ("----- 先评基座（dev 上贪心加采样，再评探针；已经评过的部分会跳过）-----", False),
        ("[#####-------------------] 108/432 (25%) (r1) task: strict_pass · 已用 1.0m · 剩余≈3.0m", False),   # 进度条
        ("这一步在服务里加载的是基座模型，评完就关", False),                       # "模型"不在行首
        ("结果：base dev 贪心 29/66", True),
        ("===== 4/13 教师给零通过的题写技能提示 =====", True),
        ("写技能：待写 72 道", False),
        ("技能提示没全部写成，继续（没写成的题不做技能引导）", True),
        (long_cjk_line(), True),                                               # 太长：只留前 240 个字
        ("可用的技能只有 4 条，少于 MIN_SKILL_TASKS=5：技能引导的题太少，", True),
        ("不做技能引导采样，也不训 D-skill 组", True),
        ("===== 5/11 前缀引导采样（参考解的开头按 0.25,0.5,0.75 各给一部分，每种 4 次） =====", True),
        ("[########------------] 216/864 (25%) (r1) task: strict_pass · 已用 3.0m · 剩余≈9.0m", False),
        ("结果：train-guided-prefix 引导（前缀 0.25）解出 12/72 题（通过样本 31/288）、照抄了开头的 20/288"
         "（其中通过 9）", True),
        ("一次都没通过的题：48 道 ['h3_p03_001', 'h3_p04_001']", True),
        ("交给训练的候选：22 条（每题只取最短的、成功过的比例）", False),
        ("===== 6/11 关闭 vLLM，等显存释放 =====", True),
        ("===== 7/11 准备训练数据（第 1 轮的验证组 + 引导出的正确答案，以及逐题配对的对照） =====", True),
        ("结果：D-prefix 训练集 83 条 / 83 题，其中引导题 22 条（难题 12、变体 10）；dev 66 条", True),
        ("结果：F-control 训练集 105 条 / 105 题，其中引导题 22 条（难题 12、变体 10）；dev 66 条", True),
        ("", False),
        ("===== 8/11 D-prefix：训练（lr 1e-4、秩 32）并在 dev 上选型 =====", True),
        ("Dry run 完成", False),
        ("训练完成", False),
        ("选中的是 epoch 1", False),                                           # "epoch"不在行首
        ("结果：D-prefix dev 0.439 → 0.473（选中 epoch 1；变好 5 道、变差 3 道）", True),
        (long_ids_line(), True),                                               # 太长：只留前 240 个字符
        ("===== 10/11 汇总主结果（都在 dev 上；test 这一轮不评） =====", True),
        ("模型                    原有题    难题     变体     合计", True),       # HF 贪心表的表头和行
        ("base                   20/24     4/20     5/20    29/64", True),
        ("epoch 1（选中） · D-prefix       21/24     5/20     6/20    32/64", True),
        ("base · F-control             20/24     4/20     5/20    29/64", True),
        ("epoch 1（选中） · F-control       21/24     5/20     5/20    31/64", True),
        ("epoch 1（选中） · D-prefix 相对 base：变好 5 道、变差 3 道"
         "（各层变好/变差：原有题 +2/-1、难题 +1/-1、变体 +2/-1）", True),
        ("base · F-control 相对 base：变好 0 道、变差 0 道"
         "（各层变好/变差：原有题 +0/-0、难题 +0/-0、变体 +0/-0）", True),
        ("===== 11/11 经 vLLM 评各组选中的适配器（辅助指标；主结果是上面 HF 贪心那张表） =====", True),
        ("族内探针：data/rft-guided/probe-ids.txt；评探针前要自检的组：D-prefix", True),
        ("----- D-prefix：族内探针（贪心，先自检）-----", True),
        ("D-prefix：适配器在 vLLM 里没有生效（训练过的题复现不了），这组的 vLLM 评测作废", True),
        ("----- base：族内探针（贪心）-----", False),
        ("标签                 dev 贪心   采样通过率   pass@k", False),        # vLLM 的表里只带出 base 的行
        ("base                 29/66      31.2%       pass@4 48.5%", True),
        ("F-control            30/66      32.0%       pass@4 50.0%", False),
        ("  各层贪心：原有题 12/24、难题 4/20、变体 14/22", False),
        ("结果：dev 贪心 base 29/66、F-control 30/66；采样通过率 base 31.2%、F-control 32.0%；"
         "相对基座 F-control +1 道", True),
        ("结果：族内探针（贪心；每个组依次是训练过的题、没训练过的同族题，括号里是基座）：基座 1/72；"
         "F-control 评测作废或没做", True),
        ("── 怎么读 ──", False),
        ("  - 采样通过率比贪心灵敏，只作旁证；", False),
        ("===== 8/8 汇总 =====", True),                                         # 嵌在最后一步里的 vLLM 评测自己的步骤
        ("===== 全部结束 =====", True),
        ("===== 没有全部完成：技能引导没有做成：可用的技能只有 4 条、少于 MIN_SKILL_TASKS=5，其中 8 道出错，"
         "没有 D-skill 组；上面的 dev 选型结果不受影响，重新运行同一条命令只补没做完的 =====", True),
    ]


# 被拒绝的 19 个题族各是因为什么（都是 scripts/synth_pool.py 给人看的原因名），一个原因一族
REJECTIONS = (
    "参考解不通过自己的测试", "和现有题池撞了", "母题只报错、没有被断言拦住", "母题没被隐藏测试拦住",
    "母题的测试超时", "母题让公开测试失败", "测试不稳定", "有 mutant 让公开测试失败",
    "有 mutant 让测试超时", "有 mutant 只报错、没有被断言拦住", "有 mutant 没被隐藏测试拦住", "有 mutant 没能判分",
    "可用的 mutant 太少", "参考解的行数不对", "参考解的测试没有留下结果", "参考解的测试超时",
    "题目规格不合法", "题目太长", "题族构造不出题目",
)
LAST_VERDICT = "[########################] 50/50 (100%) cond_buffer: 接受（第 1 轮，变体 4 个） · 已用 61m"


def verdict_line(done, slug, outcome, minutes):
    """一行题族的进度，同 scripts/synth_pool.py 的 progress_line：进度条 24 格，后面是这一族的结论。"""
    filled = 24 * done // 50
    return f"[{'#' * filled}{'-' * (24 - filled)}] {done}/50 ({done * 2}%) {slug}: {outcome} · 已用 {minutes}m"


def listed_rejections():
    """拒绝的原因一项一项列出来，每项是"原因 数量"，同这一批的结果行和汇总里"拒绝原因："的写法。

    这里一族一个原因，数量都是 1；原因多的时候这一串很长，带它的几行都会超过 240 个字。"""
    return "、".join(f"{label} 1" for label in REJECTIONS)


def long_batch_line():
    """一行很长的结果行（这一批的结果，拒绝的原因一项一项列着），汉字为主，超过 240 个字。"""
    return (f"结果：第 1 批 接受 31 族（母题 31 道、变体 124 道）、拒绝 19 族（原因：{listed_rejections()}）、"
            f"教师调用 73 次、用时 63 分钟")


def long_crash_line():
    """一行很长的出错说明（处理某个题族时流水线自己出了错，后面是异常原文），ASCII 为主，超过 240 个字符。"""
    fields = ", ".join(f"s3_cond_pool_001: {field}: field required"
                       for field in ("instruction", "module", "buggy", "fixed", "public_test", "private_tests"))
    return f"处理 cond_pool 时流水线出了错（已记成拒绝，重跑会再要一次）：ValueError: 题目规格不合法：{fields}"


def synth_log():
    """一份像模像样的合成出题日志（第 1 批跑完）：五步都在，31 族接受、19 族拒绝。每行标明取结果的命令要不要带出来。"""
    return [
        ("===== 1/5 检查前提 =====", True),                                    # 步骤标记
        ("第 1 批：向教师要 50 个题族（方向 all），记录在 data/synth-pool", False),
        ("===== 2/5 向教师要题族并逐个审计（每族最多 2 轮）=====", True),
        ("第 1 批：向教师要 50 个题族（全部方向）；每个最多 2 轮，同时 2 个、请求至少隔 1 秒；"
         "目录里已有定论的题族 0 个", False),
        ("警告：Landlock 不可用（这台机器的内核不支持它），候选代码只在 rlimit + 清洁环境下执行；"
         "文件系统与网络未受限制", False),
        (verdict_line(10, "state_coupon", "接受（第 1 轮，变体 4 个）", 8), True),      # 进度条带着结论：要带出来
        (verdict_line(12, "tx_ledger", "拒绝（可用的 mutant 太少）", 11), True),
        (verdict_line(17, "exc_ticket", "接受（第 2 轮，变体 3 个）", 19), True),
        (long_crash_line(), True),                                             # 太长：只留前 240 个字符
        (verdict_line(19, "cond_pool", "拒绝（处理时程序出错）", 22), True),
        (verdict_line(21, "lock_seat", "接受（第 1 轮，变体 5 个）", 26), True),
        (verdict_line(25, "cond_queue", "拒绝（有 mutant 没被隐藏测试拦住）", 31), True),
        (verdict_line(29, "state_cart", "接受（第 1 轮，变体 4 个）", 37), True),
        (verdict_line(33, "tx_payment", "接受（第 1 轮，变体 6 个）", 42), True),
        (verdict_line(38, "exc_retry", "拒绝（题目太长）", 48), True),
        (verdict_line(40, "cond_coupon", "接受（第 2 轮，变体 4 个）", 53), True),
        (verdict_line(43, "lock_fence", "拒绝（参考解不通过自己的测试）", 55), True),
        (LAST_VERDICT, True),
        (long_batch_line(), True),                                             # 太长：只留前 240 个字
        ("其中 2 族没有定论，不算用掉。只补它们、不要新主题：FAMILIES=2 BATCH=1 bash scripts/run_synth_pool.sh", False),
        ("下一批这样跑：BATCH=2 bash scripts/run_synth_pool.sh", False),       # 怎么接着跑不在这条命令的输出里
        ("===== 3/5 汇总这一池 =====", True),
        ("结果：共接受 31 族（母题 31、变体 124）、拒绝 19 族；各方向 状态一致性 7/12、事务回滚 6/8、异常安全 7/9、"
         "锁序 5/7、条件变量 6/9", True),
        (f"拒绝原因：{listed_rejections()}", True),                        # 汇总里的；也带着"拒绝"两个字
        ("其中 2 族没有定论（教师调用出错、回复被截断或没法解析、没能判分），不是题族的问题、也不算用掉："
         "重跑它们所在的那一批时排在最前（lock_seat、tx_refund）", False),
        ("===== 4/5 抽查 3 个已接受的题族 =====", True),
        ("── 1/3 cond_coupon（条件变量 · 优惠券；第 2 轮接受，母题 1 道、变体 4 道）──", True),    # 标题行里带着"接受"
        ("题面：实现一个优惠券核销器：核销一张券要原子地检查它没用过、没过期；已过期的券拒绝核销并返回 False；"
         "同一张券并发核销时只能成功一次", True),                                # 题面里带了"拒绝"：也带出来，是正常的
        ("隐藏测试文件：test_cond_coupon.py", False),
        ("母题 s3_cond_coupon_001 的原程序 → 参考解：", False),
        ("--- a/cond_coupon.py", False),
        ("+++ b/cond_coupon.py", False),
        ("@@ -14,7 +14,9 @@", False),
        ("-        if coupon.used:", False),
        ("+        with self._lock:", False),
        ("+            if coupon.used or coupon.expired:", False),
        ("变体 s3v_cond_coupon_1 的原程序 → 参考解（这处缺陷：过期检查放在了加锁以前）：", False),
        ("结果：抽查 3 个已接受的题族（一共 31 个；换个样本：--seed 另给一个数）", True),
        ("===== 5/5 并入阶梯 =====", True),
        ("结果：导出 155 道合成题（母题 31 道、变体 124 道；这次新冻结 31 族）：train 93 道、dev 31 道、"
         "test 31 道；读回来核对过，没有来源组跨划分", True),
        ("新题文件：data/synth-pool/synth-pool.jsonl；每道题的审计结论：synth-pool.audit.json；划分：splits.json"
         "（31 个来源族：train 19、dev 6、test 6）", False),
        ("并入阶梯：data/frozen-ladder-v2.jsonl = data/frozen-ladder.jsonl（原有的 327 道原样保留）+ 新题 155 道",
         False),
        ("===== 全部结束 =====", True),
    ]


LOGS = {GUIDED: guided_log, SYNTH: synth_log}


@pytest.mark.parametrize("locale", LOCALES)
@pytest.mark.parametrize("delivery", DELIVERIES, ids=[delivery.key for delivery in DELIVERIES])
def test_results_command_keeps_the_markers_and_result_lines_cut_to_240_characters(lab, delivery, locale):
    """取结果的命令：带出步骤标记、结果行和这一组各自的要点行（日志里每行标明取结果的命令要不要带出来），别的说明
    不带，每行只留前 240 个字符。"""
    need_tools("grep", "python3")
    log = LOGS[delivery]()
    too_long = [line for line, kept in log if len(line) > 240 and kept]
    cjk = [line for line in too_long if len(line[:240].encode("utf-8")) > 400]
    assert cjk and len(too_long) > len(cjk)         # 夹具自检：确实有要截的行，汉字为主的和 ASCII 为主的都有
    write(lab.log(delivery), "".join(line + "\n" for line, _ in log))
    shown = paste(lab, delivered(f"{delivery.key}-results"), locale)
    assert shown.returncode == 0 and shown.stderr == "", shown.stderr
    lines = shown.stdout.split("\n")[:-1]                      # print 在末尾加的那个换行不算一行
    assert lines == [line[:240] for line, kept in log if kept]
    for piece in delivery.must_show:
        assert any(piece in line for line in lines), f"取结果的命令的输出里没有带{piece}的行"
    assert all(len(line) <= 240 for line in lines)
    # 汉字为主的那一行截出来正好 240 个字，远不止 240 个字节：是按字截的。换成 cut -c，在 C 的 locale 下就按字节截了
    assert lines.count(cjk[0][:240]) == 1


@pytest.mark.parametrize("delivery", DELIVERIES, ids=[delivery.key for delivery in DELIVERIES])
def test_results_command_names_the_missing_log_and_prints_no_results(lab, delivery):
    """脚本没起来、日志还没有：什么结果都不打印，报错里带着日志的路径。"""
    need_tools("grep", "python3")
    result = paste(lab, delivered(f"{delivery.key}-results"))
    assert "结果：" not in result.stdout and "=====" not in result.stdout
    assert delivery.log in result.stderr and "Traceback" not in result.stderr
