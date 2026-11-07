# -*- coding: utf-8 -*-
"""契约版本化基类。

版本兼容策略（P01 冻结）：
- 每个契约对象携带 schema_version；当前支持 {1}。
- 读取端拒绝不支持的版本（显式异常，不静默降级）。
- 所有契约 extra="forbid"：未知字段拒绝，防止拼写错误的配置被静默忽略。
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

CONTRACT_VERSION = 1
SUPPORTED_VERSIONS = {1}


class ContractError(ValueError):
    """契约违例：版本不支持、字段非法、投影越权等。"""


class VersionedModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = CONTRACT_VERSION

    def to_contract_dict(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_contract_dict(cls, data: dict[str, Any]):
        """带版本检查的构造：先验 schema_version，再走字段验证。"""
        if not isinstance(data, dict):
            raise ContractError(f"{cls.__name__}: root must be a mapping")
        version = data.get("schema_version")
        if version not in SUPPORTED_VERSIONS:
            raise ContractError(
                f"{cls.__name__}: unsupported schema_version={version!r}, "
                f"supported={sorted(SUPPORTED_VERSIONS)}")
        try:
            return cls.model_validate(data)
        except ValidationError as e:
            raise ContractError(f"{cls.__name__}: invalid fields: {e}") from e
