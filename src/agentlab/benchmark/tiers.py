"""任务分层：原有题、难题（h3）、单缺陷变体（h3s），以及合成出来的题（母题 s3、单缺陷变体 s3v）。

报告和采样统计都按层分开看。这个文件只用标准库，也不 import agentlab 里的别的模块：
scripts/vllm_summary.py 按文件路径单独执行它来读层的定义，别的模块不跟着被拉进来。
"""
from __future__ import annotations

# 层的先后：原有三层照旧排在前头，合成题排在后面
TIERS = ("default", "hard", "variant", "synth_hard", "synth_variant")
LABELS = {"default": "原有题", "hard": "难题", "variant": "变体", "synth_hard": "合成难题", "synth_variant": "合成变体"}


def tier_of(task_id: str) -> str:
    if task_id.startswith("h3_"):
        return "hard"
    if task_id.startswith("h3v_"):
        return "variant"
    if task_id.startswith("s3v_"):
        return "synth_variant"
    if task_id.startswith("s3_"):
        return "synth_hard"
    return "default"
