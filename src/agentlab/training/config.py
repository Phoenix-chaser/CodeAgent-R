# -*- coding: utf-8 -*-
"""训练配置（LoRA/QLoRA/优化/预算）+ 回显。"""
from __future__ import annotations

from pathlib import Path

import yaml

from agentlab.contracts import ContractError


class TrainConfig:
    REQUIRED = {"base_model_revision", "mode", "lora", "optimizer",
                "context", "runtime"}

    @classmethod
    def load(cls, path: str | Path) -> dict:
        p = Path(path)
        if not p.exists():
            raise ContractError(f"train config not found: {p}")
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        missing = cls.REQUIRED - set(data or {})
        if missing:
            raise ContractError(f"train config missing: {sorted(missing)}")
        if data["mode"] not in {"full_ft", "lora", "qlora"}:
            raise ContractError(f"unknown mode: {data['mode']}")
        if data["mode"] in {"lora", "qlora"}:
            lora = data.get("lora") or {}
            for k in ("r", "alpha", "target_modules"):
                if k not in lora:
                    raise ContractError(f"lora config missing key: {k}")
        return data

    @staticmethod
    def echo(data: dict) -> str:
        """运行记录回显完整配置（可追溯）。"""
        import json
        return json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2)
