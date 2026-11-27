# -*- coding: utf-8 -*-
"""vLLM 采样客户端和 RFT 采样脚本：服务是本机上的假服务，判分是真的。"""
import http.server
import json
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agentlab.sampling import vllm_client


def completion(text, finish="stop"):
    return {"choices": [{"message": {"content": text}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2}}


def closed_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class FakeVLLM:
    """假的 vLLM：GET /v1/models 报一个模型；POST 的回答由 answer(payload) 决定。"""

    def __init__(self, answer=None):
        self.requests = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self._reply(200, {"data": [{"id": "org/model"}]})

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(payload)
                status, body = answer(payload) if answer else (200, completion("ok"))
                self._reply(status, body)

            def _reply(self, status, body):
                data = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def dead_proxy(monkeypatch):
    """环境里设着指向空端口的代理变量：访问本机服务的请求不能被它截走。"""
    for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")


def test_chat_sends_the_sampling_settings_and_ignores_proxy_variables(dead_proxy):
    fake = FakeVLLM()
    try:
        result = vllm_client.chat(fake.url, "org/model", [{"role": "user", "content": "hi"}],
                                  temperature=0.7, top_p=0.95, top_k=20, seed=7, max_tokens=32)
        models = vllm_client.server_models(fake.url)
    finally:
        fake.close()
    assert result == {"text": "ok", "finish_reason": "stop", "prompt_tokens": 3, "completion_tokens": 2}
    assert models == ["org/model"]
    request, = [item for item in fake.requests if item["max_tokens"] == 32]
    assert request["chat_template_kwargs"] == {"enable_thinking": False}     # 与训练、评测一致
    assert {key: request[key] for key in ("model", "temperature", "top_p", "top_k", "seed")} == {
        "model": "org/model", "temperature": 0.7, "top_p": 0.95, "top_k": 20, "seed": 7}


def test_errors_are_reported_in_one_line_without_a_traceback():
    fake = FakeVLLM(lambda payload: (400, {"error": {"message": "unknown field top_k"}}))
    try:
        with pytest.raises(vllm_client.VLLMError, match="返回 400.*unknown field top_k") as refused:
            vllm_client.chat(fake.url, "org/model", [{"role": "user", "content": "x"}], top_k=20)
    finally:
        fake.close()
    assert not refused.value.transient and len(fake.requests) == 1     # 参数不对：重试没有用
    with pytest.raises(vllm_client.VLLMError, match="连不上") as offline:
        vllm_client.chat(f"http://127.0.0.1:{closed_port()}", "org/model", [])
    assert offline.value.transient


def test_transient_failures_are_retried_with_backoff(monkeypatch):
    sleeps = []
    monkeypatch.setattr(vllm_client.time, "sleep", sleeps.append)
    attempts = []

    def flaky(payload):
        attempts.append(payload)
        return (503, {"error": "busy"}) if len(attempts) < 3 else (200, completion("recovered"))

    fake = FakeVLLM(flaky)
    try:
        result = vllm_client.chat(fake.url, "org/model", [{"role": "user", "content": "x"}])
    finally:
        fake.close()
    assert result["text"] == "recovered" and len(attempts) == 3
    assert sleeps == [2.0, 4.0]


def test_wait_until_ready_gives_up_with_the_last_error(monkeypatch):
    monkeypatch.setattr(vllm_client.time, "sleep", lambda seconds: None)
    with pytest.raises(vllm_client.VLLMError, match="连不上"):
        vllm_client.wait_until_ready(f"http://127.0.0.1:{closed_port()}", timeout=0.2, interval=0.01)
    fake = FakeVLLM()
    try:
        assert vllm_client.wait_until_ready(fake.url, timeout=1) == ["org/model"]
    finally:
        fake.close()


@pytest.mark.parametrize("rejected, expected", [
    (set(), {"seed": True, "top_k": True}),
    ({"top_k"}, {"seed": True, "top_k": False}),
    ({"seed", "top_k"}, {"seed": False, "top_k": False}),
])
def test_sampling_options_are_probed_not_assumed(rejected, expected):
    def answer(payload):
        if rejected & set(payload):
            return 400, {"error": {"message": "extra fields not permitted"}}
        return 200, completion("")

    fake = FakeVLLM(answer)
    try:
        assert vllm_client.sampling_options(fake.url, "org/model") == expected
    finally:
        fake.close()
    assert all(request["max_tokens"] == 1 for request in fake.requests)    # 试参数只生成 1 个 token


def test_map_concurrent_reports_results_with_their_positions():
    import time
    items = [0.03, 0.01, 0.02]

    def sleep_then_double(seconds):
        time.sleep(seconds)
        return seconds * 2

    results = dict(vllm_client.map_concurrent(sleep_then_double, items, workers=3))
    assert results == {0: 0.06, 1: 0.02, 2: 0.04}
    with pytest.raises(ValueError, match="positive"):
        list(vllm_client.map_concurrent(sleep_then_double, items, workers=0))
    with pytest.raises(ZeroDivisionError):
        list(vllm_client.map_concurrent(lambda x: 1 / x, [0, 1], workers=2))


def tiny_task(task_id, split):
    """一道独立的小题：clamp 没做截断，公开测试过、隐藏测试挂。"""
    module = f"{task_id}.py"
    return {
        "task_id": task_id, "source_group": f"group-{task_id}", "split": split,
        "bug_family": "clamping", "difficulty": "L1",
        "instruction": "实现 clamp(values, low, high)：把每个数限制在 [low, high] 内，返回新列表。",
        "module": module,
        "buggy": "def clamp(values, low, high):\n    return list(values)\n",
        "fixed": "def clamp(values, low, high):\n"
                 "    return [min(max(v, low), high) for v in values]\n",
        "public_test": f"from {module[:-3]} import clamp\n\n"
                       "def test_public():\n    assert clamp([1, 5], 0, 10) == [1, 5]\n",
        "private_tests": {"test_hidden.py": f"from {module[:-3]} import clamp\n\n"
                          "def test_hidden():\n    assert clamp([-3, 3, 13], 0, 10) == [0, 3, 10]\n"},
    }


def write_tasks(path):
    """四道小题分到三层：训练三道（难题、变体、原有题各一）、dev 一道。"""
    specs = [tiny_task(name, "dev" if name == "h3_dev_001" else "train")
             for name in ("h3_demo_001", "h3v_demo_001", "demo_001", "h3_dev_001")]
    path.write_text("".join(json.dumps(spec, ensure_ascii=False) + "\n" for spec in specs), encoding="utf-8")
    return specs


def model_answers(specs):
    """假模型：难题和 dev 题在偶数号采样时答对，其余答成原来的有缺陷代码；变体题总是写不完。"""
    by_module = {spec["module"]: spec for spec in specs}

    def answer(payload):
        user = payload["messages"][-1]["content"]
        spec = next((item for module, item in by_module.items() if f"文件 {module}：" in user), None)
        if spec is None:                       # 探测参数用的请求（只生成 1 个 token）
            return 200, completion("")
        if spec["task_id"] == "h3v_demo_001":
            return 200, completion(spec["fixed"], finish="length")
        passes = spec["task_id"] != "demo_001" and payload["seed"] % 2 == 0
        return 200, completion(spec["fixed"] if passes else spec["buggy"])

    return answer


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_sampler_grades_every_sample_and_feeds_prepare_data(tmp_path, dead_proxy):
    specs = write_tasks(tmp_path / "tasks.jsonl")
    fake = FakeVLLM(model_answers(specs))
    output = tmp_path / "sampling"

    def sample(split, *extra):
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts/sample_rft.py"), "--tasks-file", str(tmp_path / "tasks.jsonl"),
             "--split", split, "--samples", "2", "--vllm-url", fake.url, "--output-dir", str(output),
             "--concurrency", "3", "--grade-workers", "2", "--model", "org/model", *extra],
            capture_output=True, text=True, timeout=300)

    try:
        for split in ("train", "dev"):
            result = sample(split)
            assert result.returncode == 0, result.stderr[-2000:]
            assert f"结果：{split} " in result.stdout and "失败类型：" in result.stdout
    finally:
        fake.close()
    manifest = json.loads((output / "train-manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol"]["sampling_options"] == {"seed": True, "top_k": True}
    assert manifest["protocol"]["model"] == "org/model" and manifest["protocol"]["split"] == "train"
    assert {tier: (row["tasks"], row["graded_samples"], row["passing_samples"], row["solved_tasks"])
            for tier, row in manifest["tiers"].items()} == {
        "hard": (1, 2, 1, 1), "variant": (1, 2, 0, 0), "default": (1, 2, 0, 0),
        "synth_hard": (0, 0, 0, 0), "synth_variant": (0, 0, 0, 0)}      # 清单里每层一项；合成两层没有题
    assert manifest["tiers"]["hard"]["pass_at_1"] == 0.5 and manifest["tiers"]["hard"]["pass_at_k"] == 1.0
    assert manifest["failure_types"]["incomplete_response"] == 2          # 写到一半被截断的两次
    assert sum(manifest["failure_types"].values()) == 5                    # 六次采样只有一次通过
    assert manifest["failure_types"] == {"incomplete_response": 2, "private_test_failure": 3}
    assert manifest["zero_pass_tasks"] == ["demo_001", "h3v_demo_001"]
    assert len(lines(output / "train-raw.jsonl")) == len(lines(output / "train-graded.jsonl")) == 6
    candidates = lines(output / "train-candidates.jsonl")
    assert [(row["task_id"], row["candidate_idx"]) for row in candidates] == [("h3_demo_001", 0)]
    assert candidates[0]["target"]["content"] == specs[0]["fixed"].strip() + "\n"
    assert candidates[0]["provenance"]["backend"] == "vllm" and candidates[0]["provenance"]["generator"] == "self"
    sent = [item for item in fake.requests if item["max_tokens"] == 2048]
    assert {42, 43, 1042, 2042} <= {item["seed"] for item in sent}     # 种子随题目和采样序号变化
    assert all(item["chat_template_kwargs"] == {"enable_thinking": False} for item in sent)

    # 通过的样本就是教师采集那种格式：prepare_data.py 重放验证后每题只留一条
    prepared = subprocess.run(
        [sys.executable, str(ROOT / "scripts/prepare_data.py"), "--tasks-file", str(tmp_path / "tasks.jsonl"),
         "--input", str(output / "train-candidates.jsonl"), str(output / "dev-candidates.jsonl"),
         "--output-dir", str(tmp_path / "sft"), "--max-per-task", "1"],
        capture_output=True, text=True, timeout=300)
    assert prepared.returncode == 0, prepared.stderr[-2000:]
    assert [row["task_id"] for row in lines(tmp_path / "sft/train.jsonl")] == ["h3_demo_001"]
    assert [row["task_id"] for row in lines(tmp_path / "sft/dev.jsonl")] == ["h3_dev_001"]
    assert json.loads((tmp_path / "sft/dataset-manifest.json").read_text(encoding="utf-8"))[
        "label_validation"] == "all_exported_samples_replayed_strict_pass"

    # 续跑：什么都不缺，就不再发请求、不再判分
    before = len(fake.requests)
    again = sample("train")
    assert again.returncode == 0 and len(fake.requests) == before
    assert "待生成 0 个" in again.stdout and "待判 0 个" in again.stdout
    # 换了采样次数就不能混进同一个目录
    changed = sample("train", "--samples", "3")
    assert changed.returncode == 2 and "采样协议" in changed.stderr


def test_sampler_resumes_after_the_server_drops_a_request(tmp_path, dead_proxy):
    specs = write_tasks(tmp_path / "tasks.jsonl")
    state = {"failing": True}

    def answer(payload):
        if state["failing"] and payload["max_tokens"] == 2048 and payload["seed"] % 2 == 1:
            return 400, {"error": {"message": "simulated rejection"}}    # 探测参数的请求（1 个 token）不受影响
        return model_answers(specs)(payload)

    fake = FakeVLLM(answer)
    output = tmp_path / "sampling"
    command = [sys.executable, str(ROOT / "scripts/sample_rft.py"), "--tasks-file", str(tmp_path / "tasks.jsonl"),
               "--split", "dev", "--samples", "2", "--vllm-url", fake.url, "--output-dir", str(output),
               "--model", "org/model"]
    try:
        broken = subprocess.run(command, capture_output=True, text=True, timeout=300)
        assert broken.returncode == 3 and "没能生成" in broken.stderr
        assert not (output / "dev-graded.jsonl").exists()           # 生成没凑齐就不开始判分
        state["failing"] = False
        fixed = subprocess.run(command, capture_output=True, text=True, timeout=300)
    finally:
        fake.close()
    assert fixed.returncode == 0, fixed.stderr[-2000:]
    assert "待生成 1 个" in fixed.stdout and len(lines(output / "dev-graded.jsonl")) == 2
    assert "结果：dev 难题 解出 1/1（通过样本 1/2）" in fixed.stdout


SEPARATORS = "\u2028\u0085\u2029"       # str.splitlines() 把它们当成换行，JSON 的字符串里却可以原样出现


def cut_last_line(path, keep=20):
    """把文件最后一行截掉后半截：像上一次运行写到一半被打断。"""
    whole = path.read_bytes()[:-1].split(b"\n")
    path.write_bytes(b"\n".join([*whole[:-1], whole[-1][:keep]]))


def test_line_separators_in_a_model_answer_do_not_lose_the_sample(tmp_path, dead_proxy):
    """回答里带着 U+2028、U+0085、U+2029 的采样不能丢：以前写进 raw 之后被 splitlines() 切成几条坏行、读不回来，
    槽位被悄悄丢掉——清单照写、退出码还是 0，答对的题记成"一次都没通过"。"""
    specs = write_tasks(tmp_path / "tasks.jsonl")
    marked = {"on": True}

    def answer(payload):
        user = payload["messages"][-1]["content"]
        spec = next((item for item in specs if f"文件 {item['module']}：" in user), None)
        if spec is None:                       # 探测参数用的请求（只生成 1 个 token）
            return 200, completion("")
        if spec["task_id"] == "h3v_demo_001":
            code, finish = spec["fixed"], "length"
        else:
            even = spec["task_id"] != "demo_001" and payload["seed"] % 2 == 0
            code, finish = spec["fixed"] if even else spec["buggy"], "stop"
        if marked["on"] and finish == "stop":
            code = f"说明：{SEPARATORS}。\n{code}"     # 回答前面的说明里带着这三个字符，代码在它们后面
        return 200, completion(code, finish)

    fake = FakeVLLM(answer)
    output = tmp_path / "sampling"
    command = [sys.executable, str(ROOT / "scripts/sample_rft.py"), "--tasks-file", str(tmp_path / "tasks.jsonl"),
               "--split", "train", "--samples", "2", "--vllm-url", fake.url, "--output-dir", str(output),
               "--concurrency", "3", "--grade-workers", "2", "--model", "org/model"]

    def sample():
        return subprocess.run(command, capture_output=True, text=True, timeout=300)

    try:
        first = sample()
        assert first.returncode == 0, first.stderr[-3000:]
        assert "Traceback" not in first.stderr and "跳过没写完的一行" not in first.stderr
        manifest = json.loads((output / "train-manifest.json").read_text(encoding="utf-8"))
        assert (manifest["slots"], manifest["raw_records"], manifest["graded_records"]) == (6, 6, 6)
        assert manifest["zero_pass_tasks"] == ["demo_001", "h3v_demo_001"]   # h3_demo_001 偶数号那一份答对了，没有丢
        assert ("原有题 解出 0/1（通过样本 0/2）、难题 解出 1/1（通过样本 1/2）、变体 解出 0/1（通过样本 0/2）"
                in first.stdout)                                   # 六份都判了，答对的那份没有被丢掉
        assert [(row["task_id"], row["candidate_idx"]) for row in lines(output / "train-candidates.jsonl")] == [
            ("h3_demo_001", 0)]
        for name, count in (("train-raw.jsonl", 6), ("train-graded.jsonl", 6), ("train-candidates.jsonl", 1)):
            text = (output / name).read_text(encoding="utf-8")
            assert len(text.splitlines()) == text.count("\n") == count     # 一条记录一行，怎么按行读都是这么多行
            assert not set(SEPARATORS) & set(text)                 # 写出来的是转义，按行读的工具切不开它
        records = lines(output / "train-raw.jsonl")
        assert sorted((row["task_id"], row["candidate_idx"]) for row in records) == sorted(
            (spec["task_id"], index) for spec in specs if spec["split"] == "train" for index in range(2))
        explained = [row["text"] for row in records if row["text"].startswith("说明：")]
        assert len(explained) == 4 and all(SEPARATORS in row for row in explained)   # 读回来还是原来的字符
        assert len([item for item in fake.requests if item["max_tokens"] == 2048]) == 6   # 每个槽位只生成一次
        sent = len(fake.requests)
        again = sample()
        # 旧版本的采样脚本把它们原样写出来：这样的文件现在也读得回来（按 "\n" 切行），不必重做
        for name in ("train-raw.jsonl", "train-graded.jsonl"):
            path = output / name
            path.write_text(path.read_text(encoding="utf-8").replace("\\u2028", "\u2028"), encoding="utf-8")
        old_format = sample()
    finally:
        fake.close()
    for result in (again, old_format):
        assert result.returncode == 0, result.stderr[-3000:]
        assert "Traceback" not in result.stderr and "跳过没写完的一行" not in result.stderr
        assert "待生成 0 个" in result.stdout and "待判 0 个" in result.stdout
    assert len(fake.requests) == sent                                 # 两次都一个请求都没发
    assert [row["task_id"] for row in lines(output / "train-candidates.jsonl")] == ["h3_demo_001"]


def test_a_tail_cut_mid_line_is_dropped_and_that_slot_is_redone_without_gluing(tmp_path, dead_proxy):
    """上一次运行写到一半被打断，raw 或 graded 的最后一行只有半截：那一行去掉、那个槽位重做，新记录接在完整的
    行后面。以前 raw 的半行让那个槽位读不回来却不重做；graded 的半行则让新记录粘在它后面，收尾时 KeyError。"""
    specs = write_tasks(tmp_path / "tasks.jsonl")
    fake = FakeVLLM(model_answers(specs))
    output = tmp_path / "sampling"
    command = [sys.executable, str(ROOT / "scripts/sample_rft.py"), "--tasks-file", str(tmp_path / "tasks.jsonl"),
               "--split", "train", "--samples", "2", "--vllm-url", fake.url, "--output-dir", str(output),
               "--concurrency", "3", "--grade-workers", "2", "--model", "org/model"]

    def sample():
        return subprocess.run(command, capture_output=True, text=True, timeout=300)

    try:
        first = sample()
        cut_last_line(output / "train-raw.jsonl")
        second = sample()
        cut_last_line(output / "train-graded.jsonl")
        third = sample()
    finally:
        fake.close()
    assert first.returncode == 0, first.stderr[-3000:]
    for result in (second, third):
        assert result.returncode == 0, result.stderr[-3000:]
        assert "Traceback" not in result.stderr
    assert "待生成 1 个" in second.stdout and "待判 0 个" in second.stdout
    assert len([item for item in fake.requests if item["max_tokens"] == 2048]) == 7     # 只重新生成被截掉的那一个
    assert "待生成 0 个" in third.stdout and "待判 1 个" in third.stdout
    for name in ("train-raw.jsonl", "train-graded.jsonl"):
        text = (output / name).read_text(encoding="utf-8")
        assert text.count("\n") == 6 and len(lines(output / name)) == 6       # 每一行都是完整的记录，没有半行粘着新记录
    manifest = json.loads((output / "train-manifest.json").read_text(encoding="utf-8"))
    assert (manifest["slots"], manifest["raw_records"], manifest["graded_records"]) == (6, 6, 6)
    assert manifest["zero_pass_tasks"] == ["demo_001", "h3v_demo_001"]


def test_a_raw_record_that_cannot_be_read_back_leaves_the_sampling_unfinished_and_unmanifested(
        tmp_path, monkeypatch, capsys, dead_proxy):
    """槽位都报了完成、收尾时却读不出某条的记录（比如那一行没写完）：按"没完成"处理——退出码 3、不开始判分、不写
    manifest，原样重跑只补这一条。以前这样的采样被悄悄丢掉：清单照写、退出码 0，这道题就记成"一次都没通过"。"""
    import sample_rft

    specs = write_tasks(tmp_path / "tasks.jsonl")
    fake = FakeVLLM(model_answers(specs))
    output = tmp_path / "sampling"
    monkeypatch.setattr(sys, "argv", [
        str(ROOT / "scripts/sample_rft.py"), "--tasks-file", str(tmp_path / "tasks.jsonl"), "--split", "train",
        "--samples", "2", "--vllm-url", fake.url, "--output-dir", str(output), "--model", "org/model",
        "--concurrency", "2", "--grade-workers", "2"])
    honest = sample_rft.jsonl_line
    output.mkdir()
    stale = output / "train-manifest.json"                                 # 上一次留下的清单，说这份采样是齐的
    stale.write_text(json.dumps({"slots": 6, "raw_records": 6, "graded_records": 6}), encoding="utf-8")
    try:
        monkeypatch.setattr(sample_rft, "jsonl_line", lambda record: (
            honest(record)[:30] if (record["task_id"], record["candidate_idx"]) == ("demo_001", 1) else honest(record)))
        with pytest.raises(SystemExit) as stopped:
            sample_rft.main()
        assert stopped.value.code == 3
        broken = capsys.readouterr()
        assert "有 1 条采样写进文件后读不出来，原样重跑会重做：['demo_001 [1]']" in broken.err
        assert "去掉了 train-manifest.json：采样没有凑齐，不能留一份说它是齐的清单" in broken.err
        assert "Traceback" not in broken.err and "判分 train" not in broken.out
        assert sorted(path.name for path in output.iterdir()) == ["train-protocol.json", "train-raw.jsonl"]
        assert not (output / "train-manifest.json").exists()                # 清单存在就表示采样是齐的：这时不能写
        monkeypatch.setattr(sample_rft, "jsonl_line", honest)
        sample_rft.main()
    finally:
        fake.close()
    final = capsys.readouterr()
    assert "Traceback" not in final.err
    assert "已有 5 个，待生成 1 个" in final.out and "待判 6 个" in final.out  # 只补这一条（上一回一条都没判）
    manifest = json.loads((output / "train-manifest.json").read_text(encoding="utf-8"))
    assert (manifest["slots"], manifest["raw_records"], manifest["graded_records"]) == (6, 6, 6)
    assert manifest["zero_pass_tasks"] == ["demo_001", "h3v_demo_001"]
    assert len(lines(output / "train-graded.jsonl")) == 6
