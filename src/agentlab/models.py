# -*- coding: utf-8 -*-
"""支持的本地基座模型及其固定 revision。

只依赖标准库：实验阶段脚本在安装训练依赖之前就会读取它。
换模型必须同时换 revision，否则会按旧模型的提交去拉取新模型。
"""
from __future__ import annotations

# 当前实验的默认模型。README 里已发表的结果全部来自 QWEN25_MODEL，
# 要复现那些结果需要显式设置 MODEL=$QWEN25_MODEL。
DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
QWEN25_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

# 仓库核对过的提交。未列出的模型返回 None：跟随 main，并把实际解析到的
# 版本写进 manifest；要复现实验应显式指定 revision。
PINNED_REVISIONS = {
    DEFAULT_MODEL: "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    QWEN25_MODEL: "c03e6d358207e414f1eca0bb1891e29f1db0e242",
}

# 键是 config.model_type。Qwen3.5 的模型代码从 transformers v5.2.0 才开始有，
# 4.x 最后一个版本也没有。
TRANSFORMERS_MINIMUM = {"qwen3_5": (5, 2, 0), "qwen3_5_moe": (5, 2, 0)}


def default_revision(model: str | None) -> str | None:
    """该模型已核对的固定 revision；没有则为 None。"""
    return PINNED_REVISIONS.get(model or DEFAULT_MODEL)


def minimum_transformers(model_type: str) -> tuple[int, ...] | None:
    """该架构要求的最低 transformers 版本；不认识时为 None。"""
    return TRANSFORMERS_MINIMUM.get((model_type or "").replace(".", "_").lower())


def install_extra(model: str | None) -> str:
    """安装训练依赖时用的 extra 名。

    Qwen2.5 系列在 transformers 4.x 上验证过，单独一个 extra；其余模型
    （含默认的 Qwen3.5）用 ``train``。
    """
    return "train-qwen25" if "qwen2.5" in (model or "").lower() else "train"
