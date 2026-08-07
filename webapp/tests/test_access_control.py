# -*- coding: utf-8 -*-
"""访问控制测试（issue #8）：认证 / Host 校验 / Origin 校验 / 写限流。

SSE ticket 相关（issue #13 / R2-004）：长期 Token 不再经 URL 传递，
/events 仅接受 Authorization header 或短时单次 ticket。
"""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from backend import main as main_module
from backend import access
from backend import sse_ticket


@pytest.fixture
def client():
    access._hits.clear()
    sse_ticket._tickets.clear()
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


def test_shared_mode_sse_bearer_and_ticket(shared_client):
    """SSE 路由鉴权：Bearer header 合法；短时单次 ticket 合法；长期 Token 经 URL 一律拒绝。"""
    headers = {"Authorization": "Bearer secret-token-123"}
    r = shared_client.post("/api/eval/mock", headers=headers)
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    # 放入终止事件使 SSE 流尽快返回（默认 heartbeat 30s 会挂住测试）
    def terminal_get(path, req_headers=None):
        main_module._jobs[job_id]["sse_queue"].put_nowait({"type": "test", "state": "error"})
        return shared_client.get(path, headers=req_headers)

    # 无凭据且未携带 ticket：401 + 审计（Bearer 失败路径）
    assert shared_client.get(f"/api/eval/{job_id}/events").status_code == 401
    # 长期 Token 出现在 URL 中一律拒绝（R2-004 回归断言，未带 ticket 参数）
    assert shared_client.get(f"/api/eval/{job_id}/events?token=secret-token-123").status_code == 401
    # Bearer header 仍合法
    assert terminal_get(f"/api/eval/{job_id}/events", req_headers=headers).status_code == 200


def test_shared_mode_sse_ticket_flow(shared_client):
    """签发 ticket → 凭 ticket 连接 events → 重放同一 ticket 被静默拒绝。"""
    from backend import audit

    headers = {"Authorization": "Bearer secret-token-123"}
    job_id = shared_client.post("/api/eval/mock", headers=headers).json()["job_id"]

    # 签发接口需要认证
    assert shared_client.post(f"/api/eval/{job_id}/events/ticket").status_code == 401
    r = shared_client.post(f"/api/eval/{job_id}/events/ticket", headers=headers)
    assert r.status_code == 200
    ticket = r.json()["ticket"]
    assert ticket and r.json()["ttl_seconds"] > 0

    # 伪造 ticket：静默 204（R3-001 残余 2），不记审计
    n_events_before = len(audit.read_events())
    assert shared_client.get(f"/api/eval/{job_id}/events?ticket=forged").status_code == 204
    assert len(audit.read_events()) == n_events_before, "ticket 认证失败不应产生审计噪声"

    # 正确 ticket → 200，且 ticket 已消耗（条目即删）
    main_module._jobs[job_id]["sse_queue"].put_nowait({"type": "test", "state": "error"})
    assert shared_client.get(f"/api/eval/{job_id}/events?ticket={ticket}").status_code == 200
    assert ticket not in sse_ticket._tickets
    # 重放同一 ticket → 204（单次，条目已删除）
    assert shared_client.get(f"/api/eval/{job_id}/events?ticket={ticket}").status_code == 204


def test_shared_mode_sse_ticket_expired(shared_client):
    headers = {"Authorization": "Bearer secret-token-123"}
    job_id = shared_client.post("/api/eval/mock", headers=headers).json()["job_id"]
    ticket = shared_client.post(
        f"/api/eval/{job_id}/events/ticket", headers=headers
    ).json()["ticket"]
    sse_ticket._tickets[ticket]["exp"] = time.monotonic() - 1
    assert shared_client.get(f"/api/eval/{job_id}/events?ticket={ticket}").status_code == 204
    assert ticket not in sse_ticket._tickets


def test_shared_mode_sse_ticket_scope_mismatch(shared_client):
    """A job 的 ticket 不能访问 B job 的 events（静默 204）。"""
    headers = {"Authorization": "Bearer secret-token-123"}
    job_a = shared_client.post("/api/eval/mock", headers=headers).json()["job_id"]
    job_b = shared_client.post("/api/eval/mock", headers=headers).json()["job_id"]
    ticket = shared_client.post(
        f"/api/eval/{job_a}/events/ticket", headers=headers
    ).json()["ticket"]
    assert shared_client.get(f"/api/eval/{job_b}/events?ticket={ticket}").status_code == 204
    # 作用域不匹配不消耗 ticket：正确 job 仍可用（R3-001 保持语义）
    main_module._jobs[job_a]["sse_queue"].put_nowait({"type": "test", "state": "error"})
    assert shared_client.get(f"/api/eval/{job_a}/events?ticket={ticket}").status_code == 200


def test_shared_mode_sse_ticket_unknown_or_terminal_job(shared_client):
    headers = {"Authorization": "Bearer secret-token-123"}
    # 不存在的 job（job_id 必须为系统生成格式，issue #17；未知 ID → 404）
    assert shared_client.post(
        "/api/eval/20260101_120000_abcdef/events/ticket", headers=headers
    ).status_code == 404
    # 非法格式 → 400
    assert shared_client.post(
        "/api/eval/no-such-job/events/ticket", headers=headers
    ).status_code == 400
    # 终态 job（completed / error）拒绝签发，避免客户端挂在心跳上
    job_id = shared_client.post("/api/eval/mock", headers=headers).json()["job_id"]
    main_module._jobs[job_id]["state"] = "completed"
    assert shared_client.post(
        f"/api/eval/{job_id}/events/ticket", headers=headers
    ).status_code == 409
    main_module._jobs[job_id]["state"] = "error"
    assert shared_client.post(
        f"/api/eval/{job_id}/events/ticket", headers=headers
    ).status_code == 409


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


def test_strip_query_ticket_unit():
    assert access._strip_query_param(b"ticket=xyz", b"ticket") == b""
    assert access._strip_query_param(b"a=1&ticket=xyz", b"ticket") == b"a=1"
    assert access._strip_query_param(b"ticket=xyz&a=1", b"ticket") == b"a=1"
    assert access._strip_query_param(b"a=1&b=2", b"ticket") == b"a=1&b=2"


def test_job_id_from_events_path_unit():
    assert access._job_id_from_events_path("/api/eval/abc123/events") == "abc123"
    assert access._job_id_from_events_path("/api/eval/a/b/events") == ""
    assert access._job_id_from_events_path("/api/eval/abc123/events/ticket") == ""
    assert access._job_id_from_events_path("/api/dims") == ""


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
