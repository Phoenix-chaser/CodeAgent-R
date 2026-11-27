# -*- coding: utf-8 -*-
"""Opt-in task pools: the default 104-task benchmark never changes implicitly."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import export_tasks_v2 as pools  # noqa: E402


def test_default_pool_is_unchanged(monkeypatch):
    monkeypatch.delenv("EXTRA_POOLS", raising=False)
    assert pools.active_pools() == ["v2", "v2b", "v2c", "v2d"]


def test_opt_in_pool_is_listed_only_when_named(monkeypatch):
    monkeypatch.setenv("EXTRA_POOLS", "h3")
    assert pools.active_pools() == ["v2", "v2b", "v2c", "v2d", "h3"]


@pytest.mark.parametrize("value", ["v2", "h3 h3", "nope", "h3 v2d"])
def test_invalid_opt_in_list_is_rejected(monkeypatch, value):
    monkeypatch.setenv("EXTRA_POOLS", value)
    with pytest.raises(ValueError):
        pools.extra_pools()
