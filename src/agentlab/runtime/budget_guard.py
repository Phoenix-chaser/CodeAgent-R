# -*- coding: utf-8 -*-
"""BudgetGuard：基于 P01 BudgetLedger 的原子准入 + 幂等结算。

准入规则（execution-state.md §2）：
    已结算 + 未结算预留 + 新预留 <= 维度上限（None = 不限）

- reserve/settle 以 operation_id 幂等：重复调用（事件重放）不重复入账。
- settle(actual=None) → 记 unknown（绝不释放为 0）。
- 线程锁保证并发准入原子性。
- attempt 递增不清零（账本属于 episode，由调用方持有同实例）。
"""
from __future__ import annotations

import math
import threading

from agentlab.contracts import BudgetLedger, Budgets


class BudgetExceeded(RuntimeError):
    pass


class BudgetGuard:
    def __init__(self, episode_id: str, budgets: Budgets,
                 ledger: BudgetLedger | None = None):
        self.budgets = budgets
        self.ledger = ledger or BudgetLedger(episode_id=episode_id)
        self._lock = threading.Lock()
        self._reservations: dict[str, tuple[str, float]] = {}
        self._settled: dict[str, str] = {}

    @staticmethod
    def _validate_amount(amount: float | None) -> None:
        if amount is not None and (not math.isfinite(amount) or amount < 0):
            raise ValueError("budget amount must be finite and nonnegative")

    def _limit(self, dim: str) -> float | None:
        return {
            "model_requests": self.budgets.max_model_requests,
            "tool_calls": self.budgets.max_tool_calls,
            "wall_seconds": self.budgets.max_wall_seconds,
            "usd": None,                      # 金额上限由实验授权决定，此处不限
        }.get(dim)

    def try_reserve(self, op_id: str, dim: str, amount: float) -> bool:
        """原子准入：满足则预留并返回 True；不满足返回 False（不抛）。"""
        self._validate_amount(amount)
        with self._lock:
            if op_id in self._reservations:
                if self._reservations[op_id] != (dim, amount):
                    raise ValueError("operation_id reused with different reservation")
                return True
            if op_id in self._settled:
                if self._settled[op_id] != dim:
                    raise ValueError("operation_id reused with different dimension")
                return True
            limit = self._limit(dim)
            if limit is not None:
                settled = self.ledger.spent.get(dim, 0.0)
                reserved = self.ledger.reserved.get(dim, 0.0)
                if settled + reserved + amount > limit:
                    return False
            self.ledger.reserve(dim, amount)
            self._reservations[op_id] = (dim, amount)
            return True

    def settle(self, op_id: str, dim: str, actual: float | None) -> None:
        """幂等结算：同 op_id 二次调用为 no-op；actual=None 记 unknown。"""
        self._validate_amount(actual)
        with self._lock:
            reservation = self._reservations.get(op_id)
            recorded_dim = self._settled.get(op_id)
            if ((reservation and reservation[0] != dim)
                    or (recorded_dim is not None and recorded_dim != dim)):
                raise ValueError("operation_id reused with different dimension")
            if op_id in self._settled:
                return
            if actual is None:
                self.ledger.commit(dim, None)
            else:
                if reservation is not None:
                    self.ledger.release(dim, reservation[1])
                self.ledger.spent[dim] = self.ledger.spent.get(dim, 0.0) + actual
            self._settled[op_id] = dim

    def admit_or_raise(self, op_id: str, dim: str, amount: float) -> None:
        if not self.try_reserve(op_id, dim, amount):
            raise BudgetExceeded(
                f"budget admission rejected: dim={dim} amount={amount} "
                f"settled={self.ledger.spent.get(dim, 0)} "
                f"reserved={self.ledger.reserved.get(dim, 0)}")
