# -*- coding: utf-8 -*-
"""P01 契约验收测试：五条硬标准逐条对应。

1. 序列化往返
2. 非法字段/版本拒绝
3. 私有 canary 不进 prompt
4. 未知费用不被转为零
5. 运行器退出成功不能直接产生 strict_pass
"""
import pytest
from pydantic import ValidationError

from agentlab.contracts import (
    ArtifactRef, BudgetLedger, Budgets, ContractError, CostAmount,
    EpisodeConfig, GradeResult, PrivateTask, PublicTask, RunResult,
    RuntimeEvent, Usage, build_prompt, to_public,
)

CANARY = "CANARY-7f3a9-private-never-in-prompt"


def make_private() -> PrivateTask:
    return PrivateTask(
        task_id="repo_family_bug_001", task_version=1,
        source_group="source_group_001", split="train",
        bug_family="state_consistency", difficulty="L2",
        instruction="修复给定问题，并保持已有行为兼容。",
        allowed_paths=["src/**"], protected_paths=["tests/**", "pyproject.toml"],
        public_check_profile="public_pytest",
        budgets=Budgets(max_model_requests=30, max_tool_calls=60, max_wall_seconds=600),
        metadata={"license": "MIT"},
        repository_id="mini_package_alpha", repository_revision="rev-immutable-1",
        fixture_artifact="artifacts/fixture.tar.zst",
        private_grader_id="grader_001", reference_patch_id="private_reference_001",
        canary=CANARY,
    )


# ── 1. 序列化往返 ──────────────────────────────────────────
ALL_FACTORIES = [
    ("PrivateTask", make_private),
    ("PublicTask", lambda: to_public(make_private())),
    ("EpisodeConfig", lambda: EpisodeConfig(
        episode_id="ep-1", task_id="repo_family_bug_001",
        budgets=Budgets(max_model_requests=30))),
    ("RuntimeEvent", lambda: RuntimeEvent(
        episode_id="ep-1", attempt=1, step=0, source="model",
        event_type="model_request", timestamp="2026-10-31T12:00:00+00:00",
        model_revision="qwen2.5-coder:7b@rev1", payload={"prompt_sha256": "ab"},
        usage=Usage(input_tokens=120, output_tokens=35), seq=1)),
    ("RunResult", lambda: RunResult(
        episode_id="ep-1", task_id="t", exit_status="ok",
        usage=Usage(input_tokens=1000, output_tokens=200),
        cost=CostAmount(usd=0.013),
        artifacts=[ArtifactRef(path="out/diff.patch", kind="workspace_diff",
                               sha256="00" * 32)])),
    ("GradeResult", lambda: GradeResult(
        episode_id="ep-1", grader_id="grader_001", graded_at="2026-10-31T12:05:00+00:00",
        verdict="strict_fail", details={"failed_tests": ["test_x"]}),
    ),
    ("BudgetLedger", lambda: BudgetLedger(episode_id="ep-1")),
    ("CostAmount", lambda: CostAmount(usd=1.5)),
    ("ArtifactRef", lambda: ArtifactRef(path="a/b.txt")),
    ("Usage", lambda: Usage(input_tokens=5)),
    ("Budgets", lambda: Budgets(max_tool_calls=10)),
]

@pytest.mark.parametrize("name,factory", ALL_FACTORIES, ids=[n for n, _ in ALL_FACTORIES])
def test_roundtrip(name, factory):
    obj = factory()
    cls = type(obj)
    restored = cls.from_contract_dict(obj.to_contract_dict())
    assert restored == obj
    # JSON 序列化同样保持字段（None 保持 null，不落 0）
    import json
    dumped = json.loads(json.dumps(obj.to_contract_dict()))
    assert cls.from_contract_dict(dumped) == obj


# ── 2. 非法字段/版本拒绝 ───────────────────────────────────
def test_unsupported_version_rejected():
    with pytest.raises(ContractError, match="unsupported schema_version"):
        PublicTask.from_contract_dict({**_public_min(), "schema_version": 99})

def _public_min():
    # 保留 schema_version=1：非法字段测试需要通过版本检查后命中字段校验
    return to_public(make_private()).to_contract_dict()

def test_unknown_field_rejected():
    d = _public_min()
    d["turbo_mode"] = True
    with pytest.raises(ContractError, match="invalid fields"):
        PublicTask.from_contract_dict(d)

def test_missing_required_rejected():
    d = _public_min()
    d.pop("task_id")
    with pytest.raises(ContractError):
        PublicTask.from_contract_dict(d)

def test_bad_split_literal_rejected():
    d = _public_min()
    d["split"] = "validation"
    with pytest.raises(ContractError):
        PublicTask.from_contract_dict(d)


# ── 3. 私有 canary 不进 prompt ────────────────────────────
def test_canary_never_in_prompt():
    private = make_private()
    public = to_public(private)
    prompt = build_prompt(public)
    assert CANARY not in prompt
    # 私有侧字段同样不得出现
    for secret in (private.private_grader_id, private.fixture_artifact,
                   private.repository_revision):
        assert secret not in prompt

def test_build_prompt_rejects_private_task():
    with pytest.raises(ContractError, match="only accepts PublicTask"):
        build_prompt(make_private())

def test_to_public_strips_all_private_fields():
    public_dict = to_public(make_private()).to_contract_dict()
    for private_field in ("canary", "repository_id", "repository_revision",
                          "fixture_artifact", "private_grader_id",
                          "reference_patch_id", "environment_digest"):
        assert private_field not in public_dict


# ── 4. 未知费用不被转为零 ─────────────────────────────────
def test_unknown_cost_stays_none():
    assert CostAmount.unknown().usd is None
    assert CostAmount(usd=None).usd is None
    # 序列化后仍是 null，不是 0
    assert CostAmount.unknown().to_contract_dict()["usd"] is None

def test_ledger_unknown_usd_total_is_none():
    ledger = BudgetLedger(episode_id="ep-1")
    ledger.commit("usd", 0.05)
    ledger.commit("usd", None)          # 来源未报告费用
    assert ledger.spent_of("usd") is None
    assert ledger.total_usd() is None   # 存在 unknown 贡献 → None ≠ 0

def test_ledger_reserve_release_commit():
    ledger = BudgetLedger(episode_id="ep-1")
    ledger.reserve("model_requests", 5)
    ledger.commit("model_requests", 3)
    ledger.release("model_requests", 2)
    assert ledger.spent["model_requests"] == 3
    assert ledger.reserved["model_requests"] == 0


# ── 5. 退出成功 ≠ 评分通过 ────────────────────────────────
def test_run_result_has_no_pass_field():
    run = RunResult(episode_id="ep-1", task_id="t", exit_status="ok")
    assert run.exit_status == "ok"
    # RunResult 根本没有任何评分字段
    assert not any("pass" in f or "grade" in f for f in RunResult.model_fields)

def test_grade_default_not_pass_and_consistency():
    g = GradeResult(episode_id="ep-1", grader_id="g1", graded_at="now",
                    verdict="strict_fail")
    assert g.strict_pass is False
    # 直构校验失败 → pydantic ValidationError（分层设计：from_contract_dict
    # 才包装为 ContractError；两者皆为 ValueError 子类）
    with pytest.raises(ValidationError, match="strict_pass must equal"):
        GradeResult(episode_id="ep-1", grader_id="g1", graded_at="now",
                    verdict="strict_fail", strict_pass=True)
    # strict_pass 必须有 grader_id
    with pytest.raises(ValidationError, match="grader_id"):
        GradeResult(episode_id="ep-1", grader_id="", graded_at="now",
                    verdict="strict_pass", strict_pass=True)


# ── 事件去重键规则 ─────────────────────────────────────────
def test_event_dedup_key_rule():
    e1 = RuntimeEvent(episode_id="ep-1", attempt=1, source="runtime",
                      event_type="episode_start", timestamp="t", seq=1)
    e2 = RuntimeEvent(episode_id="ep-1", attempt=1, source="runtime",
                      event_type="episode_start", timestamp="t", seq=1)
    e3 = RuntimeEvent(episode_id="ep-1", attempt=2, source="runtime",
                      event_type="episode_start", timestamp="t", seq=1)
    key = lambda e: (e.episode_id, e.attempt, e.seq)
    assert key(e1) == key(e2)      # 重放 → 同键，可去重
    assert key(e1) != key(e3)      # 重试 → 新 attempt，永不等同
    assert e1.event_id is None     # 未提供 event_id 时保持 None，不合成
