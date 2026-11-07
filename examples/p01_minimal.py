# -*- coding: utf-8 -*-
"""P01 最小可运行示例：私有任务 → 公开投影 → prompt（canary 不出现）。"""
from agentlab.contracts import Budgets, PrivateTask, PublicTask, build_prompt, to_public

private = PrivateTask(
    task_id="mini_alpha_bug_001", source_group="sg_001", split="smoke",
    bug_family="off_by_one", difficulty="L1",
    instruction="修复 sum_to() 在 n=0 时的行为：应返回 0，当前抛 ValueError。",
    allowed_paths=["src/**"], protected_paths=["tests/**"],
    public_check_profile="public_pytest",
    budgets=Budgets(max_model_requests=10, max_tool_calls=20, max_wall_seconds=300),
    repository_id="mini_package_alpha", repository_revision="rev-1",
    fixture_artifact="artifacts/alpha_rev1.tar.zst",
    private_grader_id="grader_hidden_001",
    canary="CANARY-7f3a9-private-never-in-prompt",
)

public = to_public(private)
prompt = build_prompt(public)
print(prompt)
print("\n--- 泄漏自检 ---")
print("canary 在 prompt 中:", "CANARY" in prompt,
      "| 私有字段数:", len(PrivateTask.model_fields) - len(PublicTask.model_fields))
