# -*- coding: utf-8 -*-
"""饱和度趋势（迭代五）：saturation_trend 纯函数 + /api/stats/saturation 路由 trend 段。

空态（无历史数据）、样本不足、上升饱和、平稳不饱和、按数据集过滤分组。
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from backend import main as main_module
from backend import storage
from backend.engine.stats import saturation_trend


def _job(job_id, dataset, updated_at, rates):
    """rates: [(task_id, rate0-1) ...] → entries（answer_x/y 对称，得分率=rate*10）。"""
    entries = []
    for tid, rate in rates:
        s = round(rate * 10, 2)
        entries.append({"id": tid, "dimension": "知识能力", "type": "判别式",
                        "answer_x": s, "answer_y": s, "winner": "tie"})
    return {"job_id": job_id, "dataset": dataset, "updated_at": updated_at, "entries": entries}


def test_empty_state():
    out = saturation_trend({"jobs": []})
    assert out["available"] is False
    assert out["datasets"] == {}
    assert "暂无" in out["note"]


def test_insufficient_samples():
    # 每任务只有 2 次记录（< 2*3=6）
    data = {"jobs": [
        _job("J1", "dsA", "2026-01-01T00:00:00Z", [("T1", 0.5)]),
        _job("J2", "dsA", "2026-01-02T00:00:00Z", [("T1", 0.8)]),
    ]}
    out = saturation_trend(data)
    assert out["available"] is True
    assert out["datasets"]["dsA"]["saturated"] is False
    assert "样本不足" in out["datasets"]["dsA"]["note"]


def test_rising_trend_saturated():
    data = {"jobs": [
        _job("J1", "dsA", "2026-01-01T00:00:00Z", [("T1", 0.4)]),
        _job("J2", "dsA", "2026-01-02T00:00:00Z", [("T1", 0.45)]),
        _job("J3", "dsA", "2026-01-03T00:00:00Z", [("T1", 0.42)]),
        _job("J4", "dsA", "2026-01-04T00:00:00Z", [("T1", 0.6)]),
        _job("J5", "dsA", "2026-01-05T00:00:00Z", [("T1", 0.65)]),
        _job("J6", "dsA", "2026-01-06T00:00:00Z", [("T1", 0.7)]),
    ]}
    out = saturation_trend(data)
    d = out["datasets"]["dsA"]
    assert d["saturated"] is True
    assert d["tasks"] == 1
    assert "过时" in d["note"]
    assert out["available"] is True


def test_flat_trend_not_saturated():
    data = {"jobs": [
        _job("J1", "dsA", "2026-01-01T00:00:00Z", [("T1", 0.6)]),
        _job("J2", "dsA", "2026-01-02T00:00:00Z", [("T1", 0.62)]),
        _job("J3", "dsA", "2026-01-03T00:00:00Z", [("T1", 0.6)]),
        _job("J4", "dsA", "2026-01-04T00:00:00Z", [("T1", 0.61)]),
        _job("J5", "dsA", "2026-01-05T00:00:00Z", [("T1", 0.62)]),
        _job("J6", "dsA", "2026-01-06T00:00:00Z", [("T1", 0.63)]),
    ]}
    d = saturation_trend(data)["datasets"]["dsA"]
    assert d["saturated"] is False
    assert "未" in d["note"]


def test_dataset_filter_and_grouping():
    data = {"jobs": [
        _job("J1", "dsA", "2026-01-01T00:00:00Z", [("T1", 0.3)]),
        _job("J2", "dsA", "2026-01-02T00:00:00Z", [("T1", 0.9)]),
        _job("J3", "dsB", "2026-01-01T00:00:00Z", [("T2", 0.5)]),
        _job("J4", "dsB", "2026-01-02T00:00:00Z", [("T2", 0.5)]),
    ]}
    out = saturation_trend(data, dataset="dsA")
    assert set(out["datasets"]) == {"dsA"}
    out2 = saturation_trend(data)
    assert set(out2["datasets"]) == {"dsA", "dsB"}


@pytest.fixture(autouse=True)
def _clean():
    main_module._jobs.clear()
    main_module._tasks.clear()
    yield
    main_module._jobs.clear()
    main_module._tasks.clear()


def test_saturation_route_trend_preserves_jobs_and_adds_trend():
    storage.update_saturation("20260101_000000_000011",
                              [{"id": "T1", "dimension": "知识能力", "type": "判别式",
                                "answer_x": 7.0, "answer_y": 7.0, "winner": "tie"}],
                              dataset="dsA")
    with TestClient(main_module.app) as client:
        resp = client.get("/api/stats/saturation")
        assert resp.status_code == 200
        data = resp.json()
    assert len(data["jobs"]) == 1
    assert "trend" in data
    assert data["trend"]["available"] is True
    assert "dsA" in data["trend"]["datasets"]


def test_saturation_route_empty_trend():
    storage.SATURATION_FILE.unlink(missing_ok=True)  # 隔离共享目录，确保空态
    with TestClient(main_module.app) as client:
        resp = client.get("/api/stats/saturation")
        data = resp.json()
    assert data["jobs"] == []
    assert data["trend"]["available"] is False
    assert "暂无" in data["trend"]["note"]