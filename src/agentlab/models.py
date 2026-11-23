# -*- coding: utf-8 -*-
"""支持的本地基座模型及其固定 revision。

只依赖标准库：实验阶段脚本在安装训练依赖之前就会读取它。
换模型时必须同时换 revision，否则会按旧模型的提交去拉取新模型。
"""
from __future__ import annotations

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

# 仓库核对过的提交。未列出的模型返回 None：跟随 main，并把实际解析到的
# 版本写进 manifest；要复现实验应显式指定 revision。
PINNED_REVISIONS = {
    "Qwen/Qwen2.5-Coder-7B-Instruct": "c03e6d358207e414f1eca0bb1891e29f1db0e242",
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
}

# Qwen3.5 的模型代码从 transformers v5.2.0 才开始有；4.x 最后一个版本也没有。
TRANSFORMERS_MINIMUM = {"qwen3.5": (5, 2, 0)}


def default_revision(model: str | None) -> str | None:
    """该模型已核对的固定 revision；没有则为 None。"""
    return PINNED_REVISIONS.get(model or DEFAULT_MODEL)


def needs_new_transformers(model: str | None) -> bool:
    """模型名属于需要 transformers ≥ 5.2 的家族。"""
    return "qwen3.5" in (model or "").lower()


def install_extra(model: str | None) -> str:
    """安装训练依赖时应该用的 extra 名。"""
    return "train-qwen35" if needs_new_transformers(model) else "train"
