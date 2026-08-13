# -*- coding: utf-8 -*-
"""stats 扩展纯函数测试（迭代三）：Spearman / Cohen's Kappa / 金标锚定偏移 / 复评一致率。"""
from __future__ import annotations

import pytest

from backend.engine.stats import (
    calibration_offset,
    cohen_kappa,
    consistency_rate,
    spearman,
)


# ---- Spearman ----

def test_spearman_perfect_monotone():
    assert spearman([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == 1.0


def test_spearman_perfect_inverse():
    assert spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == -1.0


def test_spearman_nonlinear_monotone():
    assert spearman([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) == 1.0


def test_spearman_ties_average_rank():
    rho = spearman([1, 1, 2, 2, 3], [1, 1, 2, 2, 3])
    assert rho == 1.0


def test_spearman_unsorted_values():
    assert spearman([3, 1, 2], [3, 1, 2]) == 1.0
    assert spearman([1, 2, 3], [1, 3, 2]) == 0.5


def test_spearman_empty_or_mismatch():
    assert spearman([], []) == 0.0
    assert spearman([1, 2], [1]) == 0.0


def test_spearman_constant_series_zero():
    assert spearman([5, 5, 5], [1, 2, 3]) == 0.0


# ---- Cohen's Kappa ----

def test_kappa_perfect_agreement():
    assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0


def test_kappa_random_agreement_near_zero():
    # 类别边缘概率相等时，随机一致率的 Kappa 应接近 0
    k = cohen_kappa(["a", "b", "a", "b"], ["a", "b", "b", "a"])
    assert -0.5 < k < 0.5


def test_kappa_partial_agreement_between():
    k = cohen_kappa(["a", "a", "b", "b"], ["a", "a", "a", "b"])
    assert 0.0 < k < 1.0


def test_kappa_negative_on_inverse():
    k = cohen_kappa(["a", "a", "b", "b"], ["b", "b", "a", "a"])
    assert k < 0


def test_kappa_empty_or_mismatch():
    assert cohen_kappa([], []) == 0.0
    assert cohen_kappa(["a"], ["a", "b"]) == 0.0


def test_kappa_custom_categories():
    k = cohen_kappa([1, 2, 1], [1, 2, 1], categories=[1, 2])
    assert k == 1.0


# ---- 金标锚定偏移 ----

def test_calibration_offset_positive_when_judge_low():
    # 评审分系统性偏低 1 分
    assert calibration_offset([8, 7, 9], [7, 6, 8]) == 1.0


def test_calibration_offset_zero_when_aligned():
    assert calibration_offset([8, 7, 9], [8, 7, 9]) == 0.0


def test_calibration_offset_negative_when_judge_high():
    assert calibration_offset([6, 5], [8, 7]) == -2.0


def test_calibration_offset_empty_or_mismatch():
    assert calibration_offset([], []) == 0.0
    assert calibration_offset([1, 2], [1]) == 0.0


# ---- 复评一致率 ----

def test_consistency_rate_all_agree():
    assert consistency_rate(["model_a", "model_a", "model_a"]) == 1.0


def test_consistency_rate_all_disagree():
    assert consistency_rate(["model_a", "model_b", "tie"]) == 0.0


def test_consistency_rate_two_of_three():
    # 3 轮共 3 对：A/A 一致、A/B 不一致、A/B 不一致 → 1/3
    assert consistency_rate(["model_a", "model_a", "model_b"]) == pytest.approx(1 / 3, abs=1e-4)


def test_consistency_rate_single_round_zero():
    assert consistency_rate(["model_a"]) == 0.0


def test_consistency_rate_empty_zero():
    assert consistency_rate([]) == 0.0
