# -*- coding: utf-8 -*-
"""访问控制测试（issue #8）：认证 / Host 校验 / Origin 校验 / 写限流。"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend import main as main_module
from backend import access


@pytest.fixture
def client():
    access._hits.clear()
    return TestClient(main_module.app)


# ---------------- 单机模式（默认，无 MODEL_DUEL_TOKEN） ----------------

def test_single_mode_read_ok(client):
    assert client.get("/api/dims").status_code == 200


def test_single_mode_docs_open(client):
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_single_mode_rejects_non_loopback_host(client):
    assert client.get("/api/dims", headers={"host": "evil.com"}).status_code == 403
    assert client.get("/api/dims", headers={"host": "192.168.1.5:8910"}).status_code == 403


def test_single_mode_allows_loopback_host_with_port(client):
    for host in ("127.0.0.1:8910", "localhost:8910", "[::1]:8910"):
        assert client.get("/api/dims", headers={"host": host}).status_code == 200, host


def test_single_mode_rejects_cross_origin_write(client):
    r = client.post(
        "/api/eval/mock", headers={"origin": "https://evil.example"}
    )
    assert r.status_code == 403


def test_single_mode_allows_same_origin_write(client):
    r = client.post(
        "/api/eval/mock",
        headers={"origin": "http://testserver", "host": "testserver"},
    )
    assert r.status_code == 200


def test_single_mode_allows_no_origin_write(client):
    r = client.post("/api/eval/mock")
    assert r.status_code == 200


def test_single_mode_read_ignores_origin(client):
    assert client.get("/api/dims", headers={"origin": "https://evil.example"}).status_code == 200


def test_single_mode_referer_cross_origin_rejected(client):
    r = client.post(
        "/api/eval/mock", headers={"referer": "https://evil.example/x", "host": "testserver"}
    )
    assert r.status_code == 403


def test_single_mode_referer_same_origin_allowed(client):
    r = client.post(
        "/api/eval/mock", headers={"referer": "http://testserver/", "host": "testserver"}
    )
    assert r.status_code == 200


# ---------------- 共享模式（已设置 MODEL_DUEL_TOKEN） ----------------

@pytest.fixture
def shared_client(client, monkeypatch):
    monkeypatch.setenv("MODEL_DUEL_TOKEN", "secret-token-123")
    return client


def test_shared_mode_requires_token(shared_client):
    assert shared_client.get("/api/dims").status_code == 401


def test_shared_mode_wrong_token_rejected(shared_client):
    r = shared_client.get("/api/dims", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401
    # 非 Bearer 前缀同样拒绝
    assert shared_client.get(
        "/api/dims", headers={"Authorization": "Token secret-token-123"}
    ).status_code == 401


def test_shared_mode_valid_token_allowed(shared_client):
    r = shared_client.get("/api/dims", headers={"Authorization": "Bearer secret-token-123"})
    assert r.status_code == 200


def test_shared_mode_skips_host_check(shared_client):
    # 共享模式下局域网 IP 访问合法（Host 校验让位于令牌认证）
    r = shared_client.get(
        "/api/dims",
        headers={"Authorization": "Bearer secret-token-123", "host": "192.168.1.5:8910"},
    )
    assert r.status_code == 200


def test_shared_mode_cross_origin_still_rejected(shared_client):
    r = shared_client.post(
        "/api/eval/mock",
        headers={"Authorization": "Bearer secret-token-123", "origin": "https://evil.example"},
    )
    assert r.status_code == 403


def test_shared_mode_sse_accepts_query_token(shared_client, monkeypatch):
    # 先创建一个 job（mock），随后验证 events 路由的两种鉴权方式
    r = shared_client.post(
        "/api/eval/mock", headers={"Authorization": "Bearer secret-token-123"}
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    # 放入终止事件使 SSE 流尽快返回（默认 heartbeat 30s 会挂住测试）
    def terminal_get(path, headers=None):
        main_module._jobs[job_id]["sse_queue"].put_nowait({"type": "test", "state": "error"})
        return shared_client.get(path, headers=headers)

    assert shared_client.get(f"/api/eval/{job_id}/events").status_code == 401
    assert terminal_get(
        f"/api/eval/{job_id}/events",
        headers={"Authorization": "Bearer secret-token-123"},
    ).status_code == 200
    assert terminal_get(f"/api/eval/{job_id}/events?token=secret-token-123").status_code == 200
    assert shared_client.get(f"/api/eval/{job_id}/events?token=wrong").status_code == 401


def test_shared_mode_rate_limit_write(shared_client):
    headers = {"Authorization": "Bearer secret-token-123"}
    for i in range(30):
        r = shared_client.post("/api/eval/mock", headers=headers)
        assert r.status_code == 200, f"第 {i + 1} 次应放行"
    assert shared_client.post("/api/eval/mock", headers=headers).status_code == 429
    # 读请求不受限流影响
    assert shared_client.get("/api/dims", headers=headers).status_code == 200


def test_shared_mode_docs_and_openapi_protected(shared_client):
    headers = {"Authorization": "Bearer secret-token-123"}
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert shared_client.get(path).status_code == 401, path
        assert shared_client.get(path, headers=headers).status_code == 200, path


def test_strip_query_token_unit():
    assert access._strip_query_param(b"", b"token") == b""
    assert access._strip_query_param(b"token=x", b"token") == b""
    assert access._strip_query_param(b"a=1&token=x", b"token") == b"a=1"
    assert access._strip_query_param(b"token=x&a=1", b"token") == b"a=1"
    assert access._strip_query_param(b"a=1&b=2", b"token") == b"a=1&b=2"
    assert access._strip_query_param(b"token=&token=2", b"token") == b""


def test_rate_limit_tracking_cap(monkeypatch):
    access._hits.clear()
    monkeypatch.setattr(access, "MAX_TRACKED_IPS", 2)
    assert access._rate_limit("1.1.1.1") is True
    assert access._rate_limit("2.2.2.2") is True
    assert access._rate_limit("3.3.3.3") is True
    assert len(access._hits) <= 2
    assert "3.3.3.3" in access._hits
    access._hits.clear()


def test_shared_mode_rate_limit_isolated_per_ip(shared_client):
    headers = {"Authorization": "Bearer secret-token-123"}
    access._hits.clear()
    # 限流器按 IP 隔离（直接单测，TestClient 不暴露客户端 IP）
    ip1, ip2 = "10.0.0.1", "10.0.0.2"
    for _ in range(30):
        assert access._rate_limit(ip1) is True
    assert access._rate_limit(ip1) is False
    assert access._rate_limit(ip2) is True
