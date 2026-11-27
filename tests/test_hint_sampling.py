# -*- coding: utf-8 -*-
"""引导采样：提示怎么拼、采样器在提示模式下发了什么、存了什么；服务是假的，判分是真的。"""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from agentlab.benchmark.protocol import build_messages
from agentlab.sampling import hints
import sample_rft
from test_rft_sampling import FakeVLLM, lines

FIXED = '''def clamp(values, low, high):
    """Clamp every value."""
    result = []
    for value in values:
        if value < low:
            result.append(low)
        elif value > high:
            result.append(high)
        else:
            result.append(value)
    return result
'''
BUGGY = "def clamp(values, low, high):\n    return list(values)\n"


def task(task_id, split="train"):
    module = f"{task_id}.py"
    return {
        "task_id": task_id, "source_group": f"group-{task_id}", "split": split,
        "bug_family": "clamping", "difficulty": "L1",
        "instruction": "实现 clamp(values, low, high)：把每个数限制在 [low, high] 内，返回新列表。",
        "module": module, "buggy": BUGGY, "fixed": FIXED,
        "public_test": f"from {module[:-3]} import clamp\n\n"
                       "def test_public():\n    assert clamp([1, 5], 0, 10) == [1, 5]\n",
        "private_tests": {"test_hidden.py": f"from {module[:-3]} import clamp\n\n"
                          "def test_hidden():\n    assert clamp([-3, 3, 13], 0, 10) == [0, 3, 10]\n"},
    }


def test_prefix_hint_shows_a_fraction_of_the_reference_and_never_the_whole_file():
    assert [hints.prefix_lines(FIXED, fraction) for fraction in (0.25, 0.5, 0.75, 0.99)] == [3, 6, 9, 10]
    assert hints.prefix_lines("a = 1\n", 0.75) == 1 and hints.prefix_lines("a = 1\nb = 2\n", 0.75) == 1
    hint = hints.prefix_hint(FIXED, 0.25)
    assert hint["text"] == "\n".join(FIXED.splitlines()[:3])
    assert (hint["lines_shown"], hint["lines_total"], hint["fraction"], hint["kind"]) == (3, 11, 0.25, "prefix")
    assert "到第 3 行为止，参考解共 11 行" in hint["message"] and "原样保留" in hint["message"]


def test_with_hint_appends_to_the_user_message_and_leaves_the_canonical_prompt_alone():
    spec = task("h3_demo_001")
    canonical = build_messages(spec)
    copy = json.loads(json.dumps(canonical))
    hinted = hints.with_hint(canonical, hints.prefix_hint(spec["fixed"], 0.5))
    assert canonical == copy                                   # 标准提示没有被改动
    assert hinted[0] == canonical[0] and len(hinted) == 2
    user = canonical[1]["content"]
    assert hinted[1]["content"].startswith(user + "\n\n")
    assert hinted[1]["content"].endswith(hints.prefix_hint(spec["fixed"], 0.5)["text"])
    assert hints.with_hint(canonical, hints.skill_hint("  先想清楚再改  "))[1]["content"].endswith(
        hints.SKILL_HINT + "\n\n先想清楚再改")
    with pytest.raises(ValueError, match="final user message"):
        hints.with_hint([{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
                        hints.skill_hint("技能"))
    with pytest.raises(ValueError, match="empty skill"):
        hints.skill_hint("   ")


ONE_LINER = ("def clamp(values, low, high):\n"
             "    return [min(max(value, low), high) for value in values]\n")      # 正确，但没有照抄开头


def test_restated_checks_the_first_and_the_last_line_of_the_shown_prefix():
    half = hints.prefix_hint(FIXED, 0.5)
    quarter = hints.prefix_hint(FIXED, 0.25)
    assert hints.restated(FIXED, half) is True and hints.restated(FIXED, quarter) is True
    assert hints.restated(ONE_LINER, quarter) is False and hints.restated(ONE_LINER, half) is False
    assert hints.restated(BUGGY, quarter) is False
    assert hints.restated("    result = []\n" + FIXED, quarter) is False     # 顺序反了也不算
    assert hints.restated(FIXED, hints.skill_hint("技能")) is None


def guided_answers(specs):
    """假模型：按提示里给的行数决定答得对不对。h3_a 要看到一半以上才对，h3_b 怎么都不对，
    h3v_c 只要给了提示就对，dev 题在不引导的采样里偶数号答对；种子除 4 余 1 时换成另一种
    正确写法（答案仍正确，但没有照抄提示里的开头）。"""
    by_module = {spec["module"]: spec for spec in specs}

    def answer(payload):
        user = payload["messages"][-1]["content"]
        spec = next((item for module, item in by_module.items() if f"文件 {module}：" in user), None)
        if spec is None:                       # 探测参数用的请求（只生成 1 个 token）
            return 200, {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}
        shown = re.search(r"到第 (\d+) 行为止", user)
        shown = int(shown.group(1)) if shown else 0
        name = spec["task_id"]
        passes = (name == "h3v_c_001" or (name == "h3_a_001" and shown >= 6)
                  or (name == "h3_d_001" and payload["seed"] % 2 == 0))
        code = spec["fixed"] if passes else spec["buggy"]
        if passes and shown and payload["seed"] % 4 == 1:
            code = ONE_LINER
        return 200, {"choices": [{"message": {"content": code}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 9, "completion_tokens": 9}}

    return answer


def run_sampler(fake, tmp_path, *extra, split="train"):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts/sample_rft.py"), "--tasks-file", str(tmp_path / "tasks.jsonl"),
         "--split", split, "--vllm-url", fake.url, "--output-dir", str(tmp_path / "sampling"),
         "--concurrency", "4", "--grade-workers", "2", "--model", "org/model", *extra],
        capture_output=True, text=True, timeout=600)


def test_prefix_sampling_sends_hints_keeps_canonical_prompts_and_trains_on_the_shortest_success(tmp_path,
                                                                                               dead_proxy):
    specs = [task(name) for name in ("h3_a_001", "h3_b_001", "h3v_c_001")] + [task("h3_d_001", "dev")]
    (tmp_path / "tasks.jsonl").write_text(
        "".join(json.dumps(spec, ensure_ascii=False) + "\n" for spec in specs), encoding="utf-8")
    (tmp_path / "zero.json").write_text(json.dumps({
        "zero_pass_tasks": ["h3_a_001", "h3_b_001", "h3v_c_001", "h3_ghost_001"]}), encoding="utf-8")
    fake = FakeVLLM(guided_answers(specs))
    guided = ["--hint-kind", "prefix", "--samples", "2", "--only-zero-pass", str(tmp_path / "zero.json"),
              "--tag", "guided-prefix"]
    try:
        result = run_sampler(fake, tmp_path, *guided)
        assert result.returncode == 0, result.stderr[-3000:]
        dev = run_sampler(fake, tmp_path, "--samples", "2", split="dev")
        assert dev.returncode == 0, dev.stderr[-3000:]
    finally:
        fake.close()
    out = result.stdout
    assert "3 道题 × 6 次 = 18 个槽位" in out and "指定 4 道，其中 3 道在 train 里" in out
    assert "结果：train-guided-prefix 引导（前缀 0.25）解出 1/3 题（通过样本 2/6）" in out
    assert "结果：train-guided-prefix 引导（前缀 0.5）解出 2/3 题（通过样本 4/6）" in out
    assert "结果：train-guided-prefix 引导（前缀 0.75）解出 2/3 题（通过样本 4/6）" in out
    assert "（前缀 0.25）解出 1/3 题（通过样本 2/6）、照抄了开头的 2/6（其中通过 2）" in out
    assert "（前缀 0.5）解出 2/3 题（通过样本 4/6）、照抄了开头的 2/6（其中通过 2）" in out   # 另一种写法没照抄
    assert "（前缀 0.75）解出 2/3 题（通过样本 4/6）、照抄了开头的 4/6（其中通过 4）" in out
    assert "交给训练的候选：4 条" in out

    # 发出去的消息带提示，标准提示在前、提示在后（dev 那次不引导的采样不算）
    sent = [item for item in fake.requests
            if item["max_tokens"] == 2048 and "到第" in item["messages"][-1]["content"]]
    assert len(sent) == 18
    for request in sent:
        spec = next(item for item in specs if f"文件 {item['module']}：" in request["messages"][-1]["content"])
        canonical_user = build_messages(spec)[1]["content"]
        content = request["messages"][-1]["content"]
        assert content.startswith(canonical_user + "\n\n下面是参考修复后的主文件的开头部分")
        assert request["messages"][0] == build_messages(spec)[0]
        assert request["chat_template_kwargs"] == {"enable_thinking": False}
    assert {request["temperature"] for request in sent} == {0.7, 1.0}

    sampling = tmp_path / "sampling"
    raw = lines(sampling / "train-guided-prefix-raw.jsonl")
    graded = lines(sampling / "train-guided-prefix-graded.jsonl")
    assert len(raw) == len(graded) == 18 and all(record["hint"]["kind"] == "prefix" for record in raw)
    assert sorted((record["task_id"], record["candidate_idx"]) for record in raw) == sorted(
        (spec["task_id"], index) for spec in specs[:3] for index in range(6))
    by_id = {spec["task_id"]: spec for spec in specs}
    for record in graded:                       # 训练记录里的提示是不带提示的标准提示
        assert record["messages"] == build_messages(by_id[record["task_id"]])
        assert record["provenance"]["guided"]["kind"] == "prefix"
        assert record["provenance"]["guided"]["fraction"] in {0.25, 0.5, 0.75}
        assert "restated" in record["provenance"]["guided"]
    assert {(record["task_id"], record["candidate_idx"]) for record in graded
            if record["provenance"]["guided"]["fraction"] == 0.5
            and record["provenance"]["guided"]["lines_shown"] == 6
            and record["candidate_idx"] in (2, 3)} == {("h3_a_001", 2), ("h3_a_001", 3), ("h3_b_001", 2),
                                                        ("h3_b_001", 3), ("h3v_c_001", 2), ("h3v_c_001", 3)}
    not_restated = sorted((record["task_id"], record["candidate_idx"]) for record in graded
                          if record["provenance"]["strict_pass"]
                          and not record["provenance"]["guided"]["restated"])
    assert not_restated == [("h3_a_001", 3), ("h3v_c_001", 3)]    # 换了写法的两次
    assert all(record["sample_id"] == f"rft-{record['task_id']}-{record['candidate_idx']:03d}"
               for record in graded)

    # 每题只把"最短的、成功过的比例"交给训练：h3_a 是 0.5，h3v_c 是 0.25
    candidates = lines(sampling / "train-guided-prefix-candidates.jsonl")
    assert sorted((record["task_id"], record["provenance"]["guided"]["fraction"]) for record in candidates) == [
        ("h3_a_001", 0.5), ("h3_a_001", 0.5), ("h3v_c_001", 0.25), ("h3v_c_001", 0.25)]
    assert all(record["provenance"]["strict_pass"] for record in candidates)
    manifest = json.loads((sampling / "train-guided-prefix-manifest.json").read_text(encoding="utf-8"))
    assert [(row["fraction"], row["slots"], row["passing_samples"], row["solved_tasks"], row["restated"])
            for row in manifest["hint"]] == [(0.25, 6, 2, 1, 2), (0.5, 6, 4, 2, 2), (0.75, 6, 4, 2, 4)]
    assert (manifest["candidates"], manifest["candidates_kept_for_training"]) == (10, 4)
    assert manifest["tiers"]["hard"]["pass_at_k"] is None          # 多种提示混在一起，pass@k 没有意义
    protocol = json.loads((sampling / "train-guided-prefix-protocol.json").read_text(encoding="utf-8"))
    assert protocol["hint"]["kind"] == "prefix" and protocol["hint"]["fractions"] == [0.25, 0.5, 0.75]
    assert protocol["hint"]["samples_per_hint"] == 2 and len(protocol["hint"]["zero_pass_from"]) == 64
    assert "hint" not in json.loads((sampling / "dev-protocol.json").read_text(encoding="utf-8"))
    assert (sampling / "train-raw.jsonl").exists() is False       # 不带标记的文件名没有被占用

    # 引导出来的正确答案能直接走 prepare_data.py（重放验证、每题一条）
    prepared = subprocess.run(
        [sys.executable, str(ROOT / "scripts/prepare_data.py"), "--tasks-file", str(tmp_path / "tasks.jsonl"),
         "--input", str(sampling / "train-guided-prefix-candidates.jsonl"),
         str(sampling / "dev-candidates.jsonl"), "--output-dir", str(tmp_path / "sft"), "--max-per-task", "1"],
        capture_output=True, text=True, timeout=600)
    assert prepared.returncode == 0, prepared.stderr[-3000:]
    train = lines(tmp_path / "sft/train.jsonl")
    assert [record["task_id"] for record in train] == ["h3_a_001", "h3v_c_001"]
    assert [record["task_id"] for record in lines(tmp_path / "sft/dev.jsonl")] == ["h3_d_001"]
    assert train[0]["messages"] == build_messages(by_id["h3_a_001"])
    assert train[0]["provenance"]["guided"]["fraction"] == 0.5 and train[0]["provenance"]["revalidated"]

    # 续跑：什么都不缺就不再发请求；换了比例就不能混进同一个目录
    before = len(fake.requests)
    again = run_sampler(fake, tmp_path, *guided)
    assert again.returncode == 0 and len(fake.requests) == before
    assert "待生成 0 个" in again.stdout and "待判 0 个" in again.stdout
    changed = run_sampler(fake, tmp_path, "--hint-kind", "prefix", "--samples", "2",
                          "--hint-fractions", "0.5,0.75", "--only-zero-pass", str(tmp_path / "zero.json"),
                          "--tag", "guided-prefix")
    assert changed.returncode == 2 and "采样协议" in changed.stderr


def test_skill_sampling_uses_only_the_usable_skills(tmp_path, dead_proxy):
    specs = [task(name) for name in ("h3_a_001", "h3_b_001")]
    (tmp_path / "tasks.jsonl").write_text(
        "".join(json.dumps(spec, ensure_ascii=False) + "\n" for spec in specs), encoding="utf-8")
    skills = tmp_path / "skills.jsonl"
    skills.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in (
        {"task_id": "h3_a_001", "status": "ok", "text": "改动状态前先想清楚失败时怎么还原。"},
        {"task_id": "h3_b_001", "status": "leaky", "text": "把 clamp 的 return 改成 min(max(…))"})), encoding="utf-8")

    def answer(payload):
        user = payload["messages"][-1]["content"]
        spec = next((item for item in specs if f"文件 {item['module']}：" in user), None)
        if spec is None:
            return 200, {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}
        code = spec["fixed"] if payload["seed"] % 2 == 0 else spec["buggy"]
        return 200, {"choices": [{"message": {"content": code}, "finish_reason": "stop"}]}

    fake = FakeVLLM(answer)
    try:
        missing = run_sampler(fake, tmp_path, "--hint-kind", "skill", "--samples", "2")
        assert missing.returncode == 2 and "--skills-file" in missing.stderr
        result = run_sampler(fake, tmp_path, "--hint-kind", "skill", "--samples", "2",
                             "--skills-file", str(skills), "--tag", "guided-skill")
        assert result.returncode == 0, result.stderr[-3000:]
    finally:
        fake.close()
    assert "1 道题 × 2 次 = 2 个槽位" in result.stdout and "有可用技能的 1 道" in result.stdout
    assert "结果：train-guided-skill 引导（技能提示）解出 1/1 题（通过样本 1/2）" in result.stdout
    sent = [item for item in fake.requests if item["max_tokens"] == 2048]
    assert all(item["messages"][-1]["content"].endswith(
        hints.SKILL_HINT + "\n\n改动状态前先想清楚失败时怎么还原。") for item in sent)
    graded = lines(tmp_path / "sampling/train-guided-skill-graded.jsonl")
    assert [record["provenance"]["guided"] for record in graded] == [{"kind": "skill"}, {"kind": "skill"}]
    assert all(record["messages"] == build_messages(specs[0]) for record in graded)


def test_summary_helpers_pick_the_lowest_successful_fraction():
    def record(task_id, fraction, passed):
        return {"task_id": task_id, "provenance": {"strict_pass": passed, "guided": {
            "kind": "prefix", "fraction": fraction, "restated": passed}}}

    records = [record("a", 0.25, False), record("a", 0.5, True), record("a", 0.75, True),
               record("b", 0.25, True), record("b", 0.75, True), record("c", 0.5, False)]
    kept = sample_rft.lowest_fraction_only([row for row in records if row["provenance"]["strict_pass"]])
    assert [(row["task_id"], row["provenance"]["guided"]["fraction"]) for row in kept] == [
        ("a", 0.5), ("b", 0.25)]
    assert [(row["fraction"], row["slots"], row["passing_samples"], row["solved_tasks"], row["restated"],
             row["passing_restated"]) for row in sample_rft.hint_summary(records)] == [
        (0.25, 2, 1, 1, 1, 1), (0.5, 2, 1, 1, 1, 1), (0.75, 2, 2, 2, 2, 2)]
