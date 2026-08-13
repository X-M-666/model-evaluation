# -*- coding: utf-8 -*-
"""迭代六：鲁棒性衰减曲线（build_robustness_curves）单测。"""
from backend.engine.perturb import build_robustness_curves


def _per_task(items):
    rows = []
    for it in items:
        row = {"origin_id": "T1", "mode": "原版", "intensity": 0.0}
        row.update(it)
        rows.append(row)
    return rows


def test_intensity_zero_uses_original():
    curves = build_robustness_curves(_per_task([
        {"mode": "原版", "intensity": 0.0, "score": 9.0},
        {"mode": "改写", "intensity": 1.0, "score": 7.0},
    ]), modes=["改写"])
    c = curves["curves"]["改写"]
    assert c["intensities"] == [0.0, 1.0]
    assert c["scores"] == [9.0, 7.0]
    assert c["n_tasks"] == [1, 1]


def test_mean_aggregation_over_tasks():
    curves = build_robustness_curves(_per_task([
        {"origin_id": "T1", "mode": "原版", "intensity": 0.0, "score": 8.0},
        {"origin_id": "T2", "mode": "原版", "intensity": 0.0, "score": 6.0},
        {"origin_id": "T1", "mode": "噪声注入", "intensity": 0.1, "score": 4.0},
        {"origin_id": "T2", "mode": "噪声注入", "intensity": 0.1, "score": 8.0},
        {"origin_id": "T1", "mode": "噪声注入", "intensity": 0.2, "score": 3.0},
        {"origin_id": "T2", "mode": "噪声注入", "intensity": 0.2, "score": None},
    ]), modes=["噪声注入"])
    c = curves["curves"]["噪声注入"]
    assert c["scores"] == [7.0, 6.0, 3.0]          # 0.0: (8+6)/2, 0.1: (4+8)/2, 0.2: 仅 T1
    assert c["n_tasks"] == [2, 2, 1]


def test_by_task_perspective():
    curves = build_robustness_curves(_per_task([
        {"mode": "原版", "intensity": 0.0, "score": 8.0},
        {"mode": "改写", "intensity": 1.0, "score": 5.0},
    ]), modes=["改写"])
    assert curves["by_task"]["T1"]["原版"][0.0] == 8.0
    assert curves["by_task"]["T1"]["改写"][1.0] == 5.0


def test_empty_input():
    out = build_robustness_curves([], modes=["改写"])
    assert out["curves"]["改写"]["scores"] == [None]
    assert out["curves"]["改写"]["n_tasks"] == [0]
