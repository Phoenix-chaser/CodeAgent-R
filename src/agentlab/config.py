# -*- coding: utf-8 -*-
"""配置加载入口：读取 YAML 配置并做字段校验。

P00 范围：仅加载与校验，不涉及模型/运行时配置语义（后续工作包扩展）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ExecutionLimits:
    """执行限额：unknown 与 0 语义不同（unknown=未设定，0=禁止）。"""
    mode: str = "offline"          # offline | pilot | full
    max_real_episodes: int | None = None
    max_api_cost: float | None = None
    gpu_job_authorized: bool = False


@dataclass
class ProjectConfig:
    schema_version: int = 1
    project_name: str = "CodeAgent-R"
    execution_limits: ExecutionLimits = field(default_factory=ExecutionLimits)
    source: Path | None = None


REQUIRED_KEYS = {"schema_version", "project_name"}


class ConfigError(ValueError):
    pass


def load_config(path: str | Path) -> ProjectConfig:
    """从 YAML 加载项目配置；缺失必需字段或版本不支持时抛 ConfigError。"""
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"config root must be a mapping: {p}")
    missing = REQUIRED_KEYS - raw.keys()
    if missing:
        raise ConfigError(f"missing required keys: {sorted(missing)}")
    if raw["schema_version"] != 1:
        raise ConfigError(f"unsupported schema_version: {raw['schema_version']}")
    limits_raw = raw.get("execution_limits") or {}
    known = set(ExecutionLimits.__dataclass_fields__)
    unknown = set(limits_raw) - known
    if unknown:
        raise ConfigError(f"unknown execution_limits keys: {sorted(unknown)}")
    limits = ExecutionLimits(**limits_raw)
    return ProjectConfig(
        schema_version=raw["schema_version"],
        project_name=raw["project_name"],
        execution_limits=limits,
        source=p,
    )
