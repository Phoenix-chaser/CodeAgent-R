# -*- coding: utf-8 -*-
"""第一个 API smoke：教师模型 生成代码候选 → 项目 grader 判分。

用法：
    export TEACHER_API_KEY="你的标准 API key"
    PYTHONPATH=src python3 scripts/run_first_episode.py

这是单轮候选生成 smoke，不是完整的多轮 Agent episode。它只运行前三个
内置 smoke 任务，不记录工具事件，也不产生训练数据集。
"""
import difflib
import json
import os
import sys
import tempfile
from pathlib import Path

# 设置项目路径
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agentlab.benchmark.builder import SPECS
from agentlab.benchmark.task_store import TaskStore
from agentlab.benchmark.grader import grade
from agentlab.runtime.workspace import Workspace
from agentlab.benchmark.builder import build_all
from _shared_extract import extract_code

# ═══ 配置 ═══
# API 端点（如果你用的是标准 API，export TEACHER_BASE_URL 覆盖）
BASE_URL = os.environ.get(
    "TEACHER_BASE_URL",
    "https://teacher-api.example/api/coding/paas/v4")
MODEL = os.environ.get("TEACHER_MODEL", "teacher-api-model")
API_KEY = os.environ.get("TEACHER_API_KEY", "")


def call_teacher(instruction: str, buggy_code: str) -> str:
    """调 教师模型 生成修复代码。"""
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL,
                    timeout=60, max_retries=0)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content":
                 "你是代码修复专家。只输出修复后的完整 Python 代码，"
                 "不要解释、不要 markdown 代码块标记。"},
                {"role": "user", "content":
                 f"{instruction}\n\n当前有缺陷的代码：\n\n{buggy_code}"}
            ],
            max_tokens=1024,
            temperature=0.1)
        if not resp.choices or not resp.choices[0].message.content:
            raise RuntimeError("provider returned no text completion")
        return resp.choices[0].message.content.strip()
    finally:
        client.close()


def make_patch(module_path: str, buggy: str, fixed: str) -> str:
    """生成 unified diff。"""
    return "".join(difflib.unified_diff(
        buggy.splitlines(keepends=True),
        fixed.splitlines(keepends=True),
        fromfile=f"a/{module_path}",
        tofile=f"b/{module_path}"))


def run_one_task(spec: dict, tmp_dir: Path) -> dict:
    """对一个任务跑完整 episode。"""
    task_id = spec["task_id"]
    print(f"\n{'='*60}")
    print(f"任务: {task_id}")
    print(f"指令: {spec['instruction']}")
    print(f"缺陷家族: {spec['bug_family']}")

    # 1. 构建任务目录
    task_dir = tmp_dir / "tasks"
    build_all(task_dir)
    store = TaskStore(task_dir)
    task = store.load(task_id)

    # 2. 调教师 API 生成修复
    print(f"\n[1/4] 调用 {MODEL} 生成修复...")
    raw_response = call_teacher(spec["instruction"], spec["buggy"])
    fixed_code = extract_code(raw_response)
    print(f"  回复长度: {len(raw_response)} 字符")

    # 3. 生成补丁
    patch_text = make_patch(spec["module"], spec["buggy"], fixed_code)
    patch_file = tmp_dir / f"candidate-{task_id}.diff"
    patch_file.write_text(patch_text, encoding="utf-8")
    print(f"[2/4] 补丁大小: {len(patch_text)} 字符")

    # 4. grader 判分
    print(f"[3/4] grader 判分中...")
    result, details = grade(task, patch_file, tmp_dir / "grade")

    # 5. 输出
    print(f"[4/4] 结果: {result.verdict}")
    if result.verdict == "strict_pass":
        print(f"  ✅ 修复成功！")
    elif result.verdict == "partial":
        print(f"  ⚠️ 部分通过（公开测试过，私有测试挂）")
    elif result.verdict == "invalid":
        print(f"  ❌ 非法修改: {details.get('layers', {}).get('legality', {})}")
    else:
        func = details.get("layers", {}).get("functional", {})
        print(f"  ❌ 修复失败")
        if func:
            print(f"     公开测试: {'过' if func.get('public_pass') else '挂'}")
            print(f"     私有测试: {'过' if func.get('private_pass') else '挂'}")

    return dict(task_id=task_id, verdict=result.verdict,
                strict_pass=result.strict_pass,
                response_preview=raw_response[:200])


def main():
    if not API_KEY:
        print("错误: 请先设置 TEACHER_API_KEY 环境变量")
        print("  export TEACHER_API_KEY='你的key'")
        sys.exit(1)

    print(f"教师模型 真实 Episode 测试")
    print(f"API: {BASE_URL}")
    print(f"Endpoint: {BASE_URL}")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        results = []
        # 只跑前 3 个任务（省 token）
        for spec in SPECS[:3]:
            try:
                r = run_one_task(spec, tmp)
                results.append(r)
            except Exception as e:
                print(f"\n❌ {spec['task_id']} 出错: {e}")
                results.append(dict(task_id=spec["task_id"],
                                   verdict="error", strict_pass=False,
                                   response_preview=str(e)[:200]))

    # 汇总
    print(f"\n{'='*60}")
    print(f"汇总: {len(results)} 个任务")
    passed = sum(1 for r in results if r["strict_pass"])
    print(f"通过: {passed}/{len(results)}")
    for r in results:
        status = "✅" if r["strict_pass"] else "❌"
        print(f"  {status} {r['task_id']}: {r['verdict']}")

    # 保存结果
    out = Path("run-logs")
    out.mkdir(exist_ok=True)
    (out / "first-episode.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: run-logs/first-episode.json")


if __name__ == "__main__":
    main()
