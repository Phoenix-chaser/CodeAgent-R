# -*- coding: utf-8 -*-
"""实验合同：冻结协议的加载与校验（experiment-contract 模板实例化）。

规则：status 非 frozen 的合同只允许 pilot 性质运行；authorization
scope 未授权（max_real_episodes=0）时任何真实 episode 记录都会被
报表管线拒绝——不把未授权数据伪装成结果。
"""
from __future__ import annotations

from pathlib import Path

import yaml

from agentlab.contracts import ContractError

ALLOWED_STATUS = {"draft", "piloted", "frozen"}
REQUIRED_KEYS = {"schema_version", "experiment_id", "runtime", "dataset",
                 "models", "budget", "repetitions", "grading", "analysis",
                 "authorization", "status"}


class ExperimentContract:
    def __init__(self, data: dict, source: Path | None = None):
        self.data = data
        self.source = source

    @property
    def experiment_id(self) -> str:
        return self.data["experiment_id"]

    @property
    def status(self) -> str:
        return self.data["status"]

    @property
    def max_real_episodes(self) -> int:
        return int(self.data["authorization"].get("max_real_episodes", 0))

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentContract":
        p = Path(path)
        if not p.exists():
            raise ContractError(f"experiment contract not found: {p}")
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        missing = REQUIRED_KEYS - set(data or {})
        if missing:
            raise ContractError(
                f"experiment contract missing keys: {sorted(missing)}")
        if data["schema_version"] != 1:
            raise ContractError("unsupported contract schema_version")
        if data["status"] not in ALLOWED_STATUS:
            raise ContractError(f"invalid status: {data['status']}")
        if not data["models"]:
            raise ContractError("experiment contract requires >=1 model")
        return cls(data, p)
