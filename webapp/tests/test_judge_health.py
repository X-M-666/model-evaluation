# -*- coding: utf-8 -*-
"""评审健康度（迭代三）：health_check 阈值、告警、空集边界。"""
from __future__ import annotations

import pytest

from backend.engine.judge import health_check


def _v(score: float, invalid: bool = False) -> dict:
    return {"id": "T1", "dimension": "数学能力", "score": score, "basis": "ok",
            **_invalid_marker(invalid)}


def _invalid_marker(v: bool) -> dict:
    return {"_invalid": True} if v else {}


def test_health_check_healthy():
    v_list = [_v(8), _v(7), _v(9)]
    h = health_check(v_list, threshold=0.2)
    assert h["healthy"] is True
    assert h["alarm"] is False
    assert h["invalid_rate"] == 0.0


def test_health_check_alarm_when_above_threshold():
    v_list = [_v(7), _v(3, invalid=True), _v(1, invalid=True), _v(9)]
    h = health_check(v_list, threshold=0.4)
    assert h["healthy"] is False
    assert h["alarm"] is True
    assert h["invalid_rate"] == pytest.approx(0.5, abs=1e-4)


def test_health_check_boundary_equal_threshold():
    v_list = [_v(5), _v(5, invalid=True)]
    h = health_check(v_list, threshold=0.5)
    assert h["alarm"] is False
    assert h["healthy"] is True
    assert h["invalid_rate"] == pytest.approx(0.5, abs=1e-4)


def test_health_check_empty_no_alarm():
    h = health_check([], threshold=0.1)
    assert h["healthy"] is True
    assert h["alarm"] is False
    assert h["invalid_rate"] == 0.0


def test_health_check_meta_dict_input():
    h = health_check({"total": 4, "invalid": 2}, threshold=0.3)
    assert h["alarm"] is True
    assert h["invalid_rate"] == pytest.approx(0.5, abs=1e-4)


def test_health_check_uses_invalid_flag_not_basis_text():
    v_list = [_v(6), {"id": "T2", "score": 0, "basis": "评审模型未能返回有效 verdict"}]
    h = health_check(v_list, threshold=0.1)
    assert h["invalid_rate"] == 0.0  # 无 _invalid 标记不计数