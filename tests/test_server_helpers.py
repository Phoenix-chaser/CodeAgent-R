# -*- coding: utf-8 -*-
"""给服务器用的辅助脚本：语法可用，内嵌的 Python 片段在本地用合成数据实际跑一遍。"""
import ast
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHELL_SCRIPTS = ("scripts/thinking_pilot.sh", "scripts/show_progress.sh")
BOARD = ["bash", str(ROOT / "scripts/show_progress.sh")]
HEREDOC = re.compile(r"python -(?: \"[^\"]*\")? <<'PY'\n(.*?)\nPY\n", re.S)


def pilot_text():
    return (ROOT / "scripts/thinking_pilot.sh").read_text(encoding="utf-8")


def pilot_snippets():
    """实测脚本里内嵌的 Python，按出现顺序：选题、结果是否有效、单组结果、汇总。"""
    return HEREDOC.findall(pilot_text())


def run_snippet(index, directory, *arguments):
    result = subprocess.run([sys.executable, "-", *arguments], input=pilot_snippets()[index], text=True,
                            capture_output=True, cwd=directory, timeout=60)
    assert result.returncode == 0, result.stderr
    return result.stdout


def write_frozen_split(directory):
    """40 道难题：前 24 道在 train、接着 8 道在 dev、其余在 test；每道带 4 个变体，另有原有题。"""
    (directory / "data").mkdir(parents=True, exist_ok=True)
    with (directory / "data/frozen-ladder.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(40):
            split = "train" if index < 24 else "dev" if index < 32 else "test"
            parent = f"h3_p{index:02d}_001"
            base = dict(source_group=f"h3_p{index:02d}", split=split, module=f"h3_p{index:02d}.py")
            handle.write(json.dumps({**base, "task_id": parent}) + "\n")
            for variant in range(4):
                handle.write(json.dumps({**base, "task_id": f"h3v_p{index:02d}_{variant + 1}",
                                         "variant_of": parent}) + "\n")
            handle.write(json.dumps({"task_id": f"v2_p{index:02d}_001", "split": split,
                                     "source_group": f"v2_p{index:02d}"}) + "\n")


def test_vllm_speed_probe_only_needs_the_standard_library():
    tree = ast.parse((ROOT / "scripts/vllm_speed.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= set(sys.stdlib_module_names)      # 新环境里没装任何项目依赖也能跑


def fake_vllm(bodies):
    """本机上假的服务：记下收到的请求体，一律回答“生成了 256 个 token”。"""
    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            bodies.append(self.rfile.read(int(self.headers["Content-Length"])))
            payload = json.dumps({"choices": [{"message": {"content": "code"}}],
                                  "usage": {"completion_tokens": 256}}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)      # 随机一个空闲端口
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, str(server.server_address[1])


def test_vllm_speed_probe_reaches_a_local_server_even_with_a_proxy_variable_set():
    """环境里设了 http_proxy 时，默认的 opener 会把发给本机服务的请求转给代理，得到“连接被拒”。"""
    bodies = []
    server, port = fake_vllm(bodies)
    try:
        environment = {**os.environ, "http_proxy": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9"}
        result = subprocess.run([sys.executable, str(ROOT / "scripts/vllm_speed.py"), port], env=environment,
                                capture_output=True, text=True, timeout=60)
    finally:
        server.shutdown()
    assert result.returncode == 0, result.stderr
    assert "单路：平均" in result.stdout and "16 路并发：共 4096 个 token" in result.stdout
    assert len(bodies) == 3 + 16 and all(json.loads(b) == json.loads(bodies[0]) for b in bodies)
    request = json.loads(bodies[0])
    assert request["max_tokens"] == 256 and request["temperature"] == 0
    assert request["chat_template_kwargs"] == {"enable_thinking": False}    # 测的是不思考的生成速度


def test_vllm_speed_probe_explains_a_missing_server_without_a_traceback():
    import socket
    with socket.socket() as probe:                       # 找一个确实没人监听的端口
        probe.bind(("127.0.0.1", 0))
        port = str(probe.getsockname()[1])
    result = subprocess.run([sys.executable, str(ROOT / "scripts/vllm_speed.py"), port],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 1 and "Traceback" not in result.stderr
    assert f"连不上 http://127.0.0.1:{port}" in result.stderr


def test_pilot_selects_twelve_hard_tasks_and_twelve_other_families_without_touching_test(tmp_path):
    """12 道难题，加上另外 12 道难题各自的第一个变体；只用 train 和 dev 的题。"""
    write_frozen_split(tmp_path)
    assert "共 24 道" in run_snippet(0, tmp_path)
    lines = (tmp_path / "data/thinking-pilot/pilot-tasks.jsonl").read_text(encoding="utf-8").splitlines()
    chosen = [json.loads(line) for line in lines]
    hard = [spec for spec in chosen if spec["task_id"].startswith("h3_")]
    steps = [spec for spec in chosen if spec["task_id"].startswith("h3v_")]
    assert len(chosen) == 24 and len(hard) == len(steps) == 12
    assert {spec["split"] for spec in chosen} == {"ood"}                  # 交出去的文件当作留出集
    frozen = {spec["task_id"]: spec["split"] for spec in map(
        json.loads, (tmp_path / "data/frozen-ladder.jsonl").read_text(encoding="utf-8").splitlines())}
    assert all(frozen[spec["task_id"]] in ("train", "dev") for spec in chosen)
    assert [spec["task_id"] for spec in hard] == [f"h3_p{index:02d}_001" for index in range(12)]   # 排在最前的 12 道
    assert [spec["task_id"] for spec in steps] == [f"h3v_p{index:02d}_1" for index in range(12, 24)]
    # 变体来自另外 12 道难题（各不相同），不和选中的难题同族
    assert len({spec["variant_of"] for spec in steps}) == 12
    assert not {spec["variant_of"] for spec in steps} & {spec["task_id"] for spec in hard}


def test_pilot_treats_a_result_full_of_infrastructure_errors_as_not_done(tmp_path):
    """显卡被占用时整份都是环境报错：这一组不能算做完，重新启动脚本会重做。"""
    def done(*failure_types):
        path = tmp_path / "result.json"
        path.write_text(json.dumps({"results": [{"failure_type": kind} for kind in failure_types]}),
                        encoding="utf-8")
        return path

    def exit_code(path=None):
        command = [sys.executable, "-"] + ([] if path is None else [str(path)])
        return subprocess.run(command, input=pilot_snippets()[1], text=True, capture_output=True,
                              cwd=tmp_path, timeout=60).returncode

    assert exit_code() == 1                                            # 文件不存在
    assert exit_code(done(*(["infrastructure_error"] * 3))) == 1
    assert exit_code(done("infrastructure_error", "private_test_failure")) == 0


def test_pilot_reports_each_arm_and_the_final_table(tmp_path):
    def record(task, passed, failure_type=None, output_tokens=100):
        return dict(task_id=task, strict_pass=passed, verdict="strict_pass" if passed else "partial",
                    failure_type=None if passed else failure_type, output_tokens=output_tokens)

    def payload(records, thinking=False):
        failures = [r["failure_type"] for r in records if not r["strict_pass"]]
        return {"results": records,
                "generation": {"max_new_tokens": 8192, **({"thinking": True} if thinking else {})},
                "metrics": {"failure_counts": {kind: failures.count(kind) for kind in set(failures)}}}

    results = tmp_path / "data/thinking-pilot/results"
    results.mkdir(parents=True)
    hard = [f"h3_p{index:02d}_001" for index in range(2)]
    steps = [f"h3v_p{index:02d}_1" for index in range(24, 26)]
    broken = {"task_id": steps[1], "strict_pass": False, "verdict": "error",          # 环境报错的记录没有 token 数
              "failure_type": "infrastructure_error"}
    on = [record(hard[0], False, "thinking_unfinished", output_tokens=8192),
          record(hard[1], True, output_tokens=900), record(steps[0], True, output_tokens=700), broken]
    (results / "on.json").write_text(json.dumps(payload(on, thinking=True)), encoding="utf-8")
    line = run_snippet(2, tmp_path, str(results / "on.json"))
    assert "结果：难题 1/2、变体 1/2" in line and "推理没写完 1 道" in line
    assert "环境报错 1 道 ['h3v_p25_1']" in line
    (results / "off-sampled.json").write_text(json.dumps(payload([
        record(hard[0], True, output_tokens=500), record(hard[1], False, "private_test_failure"),
        record(steps[0], True, output_tokens=500), record(steps[1], False, "public_test_failure")])),
        encoding="utf-8")
    (results / "off.json").write_text(json.dumps(payload([
        record(hard[0], False, "private_test_failure"), record(hard[1], False, "private_test_failure"),
        record(steps[0], False, "private_test_failure"), record(steps[1], True, output_tokens=400)])),
        encoding="utf-8")
    assert "推理没写完" not in run_snippet(2, tmp_path, str(results / "off.json"))
    table = run_snippet(3, tmp_path)
    assert "类别    A      B      C" in table
    assert "难题    0/12   1/12   1/12" in table and "变体    1/12   1/12   1/12" in table
    assert "A 组有" not in table and "B 组有" not in table and "C 组有 1 道是环境报错" in table
    assert "输出 token 中位 900 / 最大 8192；推理没写完 1 道 ['h3_p00_001']" in table
    assert re.search(r"h3_p00_001\s+private_test_failure\s+通过\s+thinking_unfinished\(8192\)", table)
    assert re.search(r"h3v_p25_1\s+通过\s+public_test_failure\s+infrastructure_error", table)


def test_pilot_arms_use_the_sampling_the_model_card_recommends():
    """三组对比；思考模式按模型卡的推荐采样（temperature 0.6 / top_p 0.95 / top_k 20），而不是贪心。"""
    assert re.findall(r"^arm (\S+) ([ABC]) ", pilot_text(), re.M) == [
        ("off", "A"), ("off-sampled", "B"), ("on", "C")]            # 结果文件名与汇总里读的一致
    assert '--thinking --temperature 0.6 --top-p 0.95 --top-k 20 --max-new-tokens "$PILOT_MAX"' in pilot_text()
    assert "--temperature 0.7 --top-p 0.8 --top-k 20 --max-new-tokens 2048" in pilot_text()
    assert 'PILOT_MAX="${PILOT_MAX:-8192}"' in pilot_text()            # 思考的输出长，3072 不够用


@pytest.mark.skipif(os.name == "nt" or not shutil.which("bash"), reason="needs Bash")
@pytest.mark.parametrize("relative", SHELL_SCRIPTS)
def test_shell_helpers_parse(relative):
    assert subprocess.run(["bash", "-n", str(ROOT / relative)], capture_output=True).returncode == 0


@pytest.mark.skipif(os.name == "nt" or not shutil.which("bash"), reason="needs Bash")
def test_progress_board_marks_steps_and_results(tmp_path):
    (tmp_path / "run.log").write_text(
        "===== 1/2 第一步 =====\n"
        "候选：难题 32 道\n"
        "===== 2/2 第二步 =====\n"
        "[#####-------------------] 5/20 (25%) (r1) task: strict_pass · 已用 1.0m · 剩余≈3.0m\n"
        "结果：难题 0/12、变体 3/12\n"
        "A = 现有协议：思考关闭 + 贪心\n"
        "类别    A      B      C\n"
        "难题    0/12   1/12   2/12\n"
        "变体    3/12   4/12   6/12\n"
        "C 组：输出 token 中位 1500 / 最大 8192\n"
        "  h3_p00_001        partial        partial        通过\n"      # 逐题表只在“日志最后几行”里出现
        "警告：Landlock 不可用\n", encoding="utf-8")
    result = subprocess.run([*BOARD, str(tmp_path / "run.log"), "", ""], capture_output=True, text=True,
                            timeout=30)
    board = result.stdout.replace("\x1b[2J\x1b[H", "")
    assert result.returncode == 0, result.stderr
    assert "状态: 进行中" in board
    assert "[完成]   1/2 第一步" in board and "[进行中] 2/2 第二步" in board
    assert "候选：难题 32 道" in board and "5/20 (25%)" in board and "结果：难题 0/12、变体 3/12" in board
    for line in ("A = 现有协议：思考关闭 + 贪心", "难题    0/12   1/12   2/12", "变体    3/12   4/12   6/12",
                 "C 组：输出 token 中位 1500 / 最大 8192"):            # 实测的汇总表要能在面板上直接看到
        assert line in board
    assert "Landlock" not in board


@pytest.mark.skipif(os.name == "nt" or not shutil.which("bash"), reason="needs Bash")
def test_progress_board_ignores_itself_and_stops_watching_when_the_job_is_over(tmp_path):
    """进程名也出现在面板自己的命令行里；不排除自己的话会永远显示“进行中”。"""
    log = tmp_path / "run.log"
    log.write_text("===== 1/1 第一步 =====\n结果：难题 0/12\n", encoding="utf-8")
    board = subprocess.run([*BOARD, str(log), "", "sleepy-job-xyz"], capture_output=True, text=True,
                           timeout=30).stdout
    assert "状态: 没有在运行" in board and "结果：难题 0/12" in board
    watched = subprocess.run([*BOARD, str(log), "", "sleepy-job-xyz"], capture_output=True, text=True,
                             timeout=30, env={**os.environ, "WATCH": "1"}).stdout
    assert watched.count("进度：") == 1                                # 任务不在了就不再等下一轮
    log.write_text("===== 1/1 第一步 =====\n结果：难题 3/12\n===== 全部结束 =====\n", encoding="utf-8")
    watched = subprocess.run([*BOARD, str(log), "", ""], capture_output=True, text=True, timeout=30,
                             env={**os.environ, "WATCH": "1"}).stdout
    assert watched.count("进度：") == 1 and "状态: 已完成" in watched
    missing = subprocess.run([*BOARD, str(tmp_path / "missing.log"), "", ""], capture_output=True, text=True,
                             timeout=30).stdout
    assert "还没有日志：任务尚未启动" in missing


@pytest.mark.skipif(os.name == "nt" or not shutil.which("bash"), reason="needs Bash")
def test_progress_board_tells_a_run_that_ended_with_things_undone_apart_from_a_complete_one(tmp_path):
    """run_rft.sh、run_rft_guided.sh 有没做完的事时以「===== 没有全部完成：… =====」结束（退出码 1）：面板不能显示
    "已完成"。状态看的是最后一个步骤标记——嵌在第 13 步里的 rft_vllm_eval.sh 自己也打「全部结束」。"""
    log = tmp_path / "run.log"
    tail = "===== 8/8 汇总 =====\n结果：vLLM 评测做完了；报告在 data/rft-guided/vllm\n===== 全部结束 =====\n"
    undone = ("===== 没有全部完成：vLLM 评测没有全部完成（rft_vllm_eval.sh 以退出码 1 结束）；"
              "上面的 dev 选型结果不受影响，重新运行同一条命令只补没做完的 =====\n")
    # 里面那次结束以后外层还有步骤：日志里有过「全部结束」不算完，看最后一个步骤标记
    log.write_text("===== 1/2 外层 =====\n" + tail + "===== 2/2 收尾 =====\n结果：收尾中\n", encoding="utf-8")
    board = subprocess.run([*BOARD, str(log), "", ""], capture_output=True, text=True, timeout=30).stdout
    assert "状态: 进行中" in board and "[完成]   全部结束" in board and "[进行中] 2/2 收尾" in board
    log.write_text("===== 1/2 外层 =====\n" + tail + undone, encoding="utf-8")
    board = subprocess.run([*BOARD, str(log), "", ""], capture_output=True, text=True, timeout=30).stdout
    assert "状态: 已结束，但没有全部完成" in board and "状态: 已完成" not in board
    assert "[没做完] 没有全部完成：vLLM 评测没有全部完成" in board      # 最后一步没做完，不是"进行中"
    watched = subprocess.run([*BOARD, str(log), "", ""], capture_output=True, text=True, timeout=30,
                             env={**os.environ, "WATCH": "1"}).stdout
    assert watched.count("进度：") == 1 and "状态: 已结束，但没有全部完成" in watched   # 结束了就不再等下一轮
    log.write_text("===== 1/2 外层 =====\n" + tail + "===== 2/2 收尾 =====\n===== 全部结束 =====\n",
                   encoding="utf-8")
    board = subprocess.run([*BOARD, str(log), "", ""], capture_output=True, text=True, timeout=30).stdout
    assert "状态: 已完成" in board and "[没做完]" not in board


STUB = '''#!/usr/bin/env python3
"""假的 python：模型名、版本号和评测结果都是现成的；其余命令（内嵌的 Python 片段）交给真解释器。"""
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if args[0] == "src/agentlab/experiments/run_state.py":
    print("Qwen/Qwen3.5-9B" if args[1] == "default-model" else "rev-123")
elif args[0] == "scripts/evaluate.py":
    with open(os.environ["STUB_LOG"], "a", encoding="utf-8") as handle:
        handle.write(" ".join(args) + "\\n")
    output = Path(args[args.index("--output") + 1])
    tasks = [json.loads(line)["task_id"] for line in
             Path(args[args.index("--tasks-file") + 1]).read_text(encoding="utf-8").splitlines()]
    if os.environ.get("STUB_BROKEN"):                      # 显卡被占用：整份都是环境报错
        results = [dict(task_id=task, strict_pass=False, verdict="error",
                        failure_type="infrastructure_error") for task in tasks]
    else:
        results = []
        for task in tasks:
            passed = task.startswith("h3_p00")
            kind = None if passed else "thinking_unfinished" if "--thinking" in args else "private_test_failure"
            results.append(dict(task_id=task, strict_pass=passed, verdict="strict_pass" if passed else "partial",
                                failure_type=kind, output_tokens=100))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"results": results, "metrics": {"failure_counts": {}},
                                  "generation": {"max_new_tokens": 1}}), encoding="utf-8")
else:
    os.execv(sys.executable, [sys.executable, *args])
'''


@pytest.mark.skipif(os.name == "nt" or not shutil.which("bash"), reason="needs Bash")
def test_pilot_runs_end_to_end_and_resumes(tmp_path):
    """整份脚本用假的评测器跑通：先是一次显卡被占着的失败，再是正常的一遍，再跑一遍全部跳过。"""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts/thinking_pilot.sh", repo / "scripts/thinking_pilot.sh")
    write_frozen_split(repo)
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin/python").write_text(STUB, encoding="utf-8")
    (tmp_path / "bin/python").chmod(0o755)
    log = tmp_path / "evaluate-calls.log"
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}",
           "STUB_LOG": str(log)}

    def pilot(**extra):
        return subprocess.run(["bash", "scripts/thinking_pilot.sh"], cwd=repo, env={**env, **extra},
                              capture_output=True, text=True, timeout=120)

    broken = pilot(STUB_BROKEN="1")
    assert broken.returncode == 0, broken.stderr
    assert broken.stdout.count("环境报错 24 道") == 3 and "===== 全部结束 =====" in broken.stdout
    assert log.read_text(encoding="utf-8").count("\n") == 3
    calls = log.read_text(encoding="utf-8").splitlines()
    assert "--thinking --temperature 0.6 --top-p 0.95 --top-k 20 --max-new-tokens 8192" in calls[2]
    assert "--temperature 0 --max-new-tokens 2048" in calls[0] and "--top-p 1.0" not in calls[0]
    assert calls[1].endswith("--temperature 0.7 --top-p 0.8 --top-k 20 --max-new-tokens 2048")
    for call in calls:                                    # 三组评的是同一批题，都当作留出集
        assert "--tasks-file data/thinking-pilot/pilot-tasks.jsonl --split ood --model Qwen/Qwen3.5-9B" in call

    good = pilot()                                        # 整份报错的三组都重做
    assert good.returncode == 0, good.stderr
    assert good.stdout.count("这一组已有结果，跳过") == 0 and "环境报错" not in good.stdout
    assert "结果：难题 1/12、变体 0/12" in good.stdout
    assert "难题    1/12   1/12   1/12" in good.stdout and "推理没写完 23 道" in good.stdout   # 通过的那道之外都没写完
    again = pilot()                                       # 结果都有效：一组都不重评
    assert again.stdout.count("这一组已有结果，跳过") == 3 and "===== 全部结束 =====" in again.stdout
    assert log.read_text(encoding="utf-8").count("\n") == 6
