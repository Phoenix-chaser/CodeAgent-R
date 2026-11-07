# -*- coding: utf-8 -*-
"""BudgetLedger：预算账本。

规则（计划书 §2/P00 延续）：
- reserve → 预扣；commit → 实际发生；release → 未发生的预扣释放。
- usd 维度允许 unknown（来源不给费用时），unknown 不计入数值总额，
  total_usd() 只要存在 unknown 贡献即返回 None——绝不把 unknown 当 0。
- attempt/restart 不清零账本（重建账本只发生在新 episode）。
"""
from __future__ import annotations

from agentlab.contracts.versioning import VersionedModel


class BudgetLedger(VersionedModel):
    episode_id: str
    reserved: dict[str, float] = {}
    spent: dict[str, float] = {}
    unknown_spent_dims: list[str] = []  # JSON 可序列化；成员查询用 in

    def reserve(self, dim: str, amount: float) -> None:
        self.reserved[dim] = self.reserved.get(dim, 0.0) + amount

    def release(self, dim: str, amount: float) -> None:
        cur = self.reserved.get(dim, 0.0) - amount
        self.reserved[dim] = max(cur, 0.0)

    def commit(self, dim: str, amount: float | None) -> None:
        """实际发生额；amount=None 记为 unknown（不改数值侧）。"""
        if amount is None:
            if dim not in self.unknown_spent_dims:
                self.unknown_spent_dims.append(dim)
            self.reserved[dim] = max(self.reserved.get(dim, 0.0) - 0.0, 0.0)
            return
        self.spent[dim] = self.spent.get(dim, 0.0) + amount
        self.release(dim, amount)

    def spent_of(self, dim: str) -> float | None:
        return None if dim in self.unknown_spent_dims else self.spent.get(dim, 0.0)

    def total_usd(self) -> float | None:
        """存在任何 unknown 贡献 → None（unknown ≠ 0）。"""
        if "usd" in self.unknown_spent_dims:
            return None
        return self.spent.get("usd", 0.0)
