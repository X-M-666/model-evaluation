# -*- coding: utf-8 -*-
"""统计显著性单元测试（迭代二）：bootstrap 确定性、CI 判定、胜率、样本不足提示。"""
from __future__ import annotations

import pytest

from backend.engine.stats import (
    MIN_SAMPLE,
    paired_bootstrap_ci,
    significance_note,
    win_rate,
)


def test_bootstrap_seed_deterministic():
    x = [8.0, 7.0, 9.0, 6.0, 8.5, 7.5, 9.0, 8.0]
    y = [5.0, 4.0, 6.0, 5.5, 4.5, 6.0, 5.0, 5.5]
    assert paired_bootstrap_ci(x, y, seed=42) == paired_bootstrap_ci(x, y, seed=42)


def test_bootstrap_empty_returns_zero():
    assert paired_bootstrap_ci([], []) == (0.0, 0.0)


def test_ci_contains_zero_when_identical():
    x = [7.0] * 10
    ci = paired_bootstrap_ci(x, x, seed=1)
    assert ci[0] <= 0 <= ci[1]


def test_ci_excludes_zero_when_clearly_different():
    x = [9.0, 8.0, 9.5, 8.5, 9.0, 8.0, 9.0, 8.5, 9.5, 8.0]
    y = [3.0, 2.0, 4.0, 3.5, 2.5, 3.0, 2.0, 3.5, 2.0, 3.0]
    ci = paired_bootstrap_ci(x, y, seed=7)
    assert ci[0] > 0


def test_win_rate_counts():
    r = win_rate([10, 8, 7, 6], [5, 8, 9, 4])
    assert r["wins"] == 2 and r["ties"] == 1 and r["losses"] == 1
    assert r["total"] == 4 and r["win_rate"] == 0.5


def test_significance_insufficient_sample():
    x = [9.0, 8.0, 9.0, 8.0]
    y = [2.0, 3.0, 2.0, 3.0]
    r = significance_note(x, y, seed=1)
    assert r["significant"] is False
    assert r["reason"] == "insufficient_sample"
    assert "不足" in r["note"]


def test_significance_ci_overlaps_zero():
    x = [7.0] * MIN_SAMPLE
    r = significance_note(x, list(x), seed=1)
    assert r["significant"] is False
    assert r["reason"] == "ci_overlaps_zero"
    assert "包含 0" in r["note"]


def test_significance_clear_difference():
    x = [9.0, 8.0, 9.5, 8.5, 9.0, 8.0, 9.0, 8.5, 9.5, 8.0]
    y = [3.0, 2.0, 4.0, 3.5, 2.5, 3.0, 2.0, 3.5, 2.0, 3.0]
    r = significance_note(x, y, seed=7)
    assert r["significant"] is True
    assert r["reason"] == "significant"
    assert r["sample"] == len(x)
    assert r["ci"][0] > 0
