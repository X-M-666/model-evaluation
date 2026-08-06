# -*- coding: utf-8 -*-
"""平台代码验真能力上报测试（issue #11 复审 R2-008）。

保护不变量：
- /api/code-runner/status 返回平台、模式可用性与 probe 快检结果
- ?selfcheck=1 时触发真实受限进程自检，返回 selfcheck_ok（布尔）
- 共享模式（MODEL_DUEL_TOKEN）下该接口与其他 /api/* 同样受保护
"""
from __future__ import annotations

import sys

import pytest
from fastapi.testclient import TestClient

from backend import main as main_module


@pytest.fixture
def client():
    return TestClient(main_module.app)


def test_status_reports_platform_and_modes(client):
    r = client.get("/api/code-runner/status")
    assert r.status_code == 200
    data = r.json()
    assert data["default_mode"] == "off"
    assert data["platform"] == sys.platform
    assert "off" in data["modes"] and "native-sandbox" in data["modes"]
    for m in data["modes"].values():
        assert isinstance(m["available"], bool)
        assert isinstance(m["detail"], str)


def test_status_probe_is_boolean(client):
    data = client.get("/api/code-runner/status").json()
    native = data["native"]
    assert isinstance(native["probe_ok"], bool)
    assert isinstance(native["probe_detail"], str)
    assert "selfcheck_ok" not in native, "默认不执行真实自检"


def test_status_selfcheck_opt_in(client):
    data = client.get("/api/code-runner/status", params={"selfcheck": "1"}).json()
    native = data["native"]
    assert isinstance(native["selfcheck_ok"], bool)
    assert isinstance(native["selfcheck_detail"], str)


def test_status_protected_in_shared_mode(monkeypatch, client):
    monkeypatch.setenv("MODEL_DUEL_TOKEN", "shared-token-123")
    r = client.get("/api/code-runner/status")
    assert r.status_code == 401
    ok = client.get(
        "/api/code-runner/status",
        headers={"Authorization": "Bearer shared-token-123"},
    )
    assert ok.status_code == 200
