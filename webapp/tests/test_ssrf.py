# -*- coding: utf-8 -*-
"""SSRF 防护测试（issue #8 + issue #12 复审 R2-003 + issue #19 多 IP 故障转移）。

issue #12：DNS 重绑定防护——除启动门禁校验外，连接前重新解析并逐 IP 过滤，
拒绝指向内网/IP 字面量的连接；302 重定向目标同样受限。
issue #19：httpcore 把 OSError 映射为 ConnectError（非 OSError 子类），
重试循环须显式捕获 ConnectError/ConnectTimeout；timeout 为总预算。
"""
from __future__ import annotations

import asyncio
import http.server
import httpcore
import json
import socket
import threading
import time

import anyio
import httpx
import pytest
from fastapi.testclient import TestClient

from backend import main as main_module
from backend import ssrf
from backend.ssrf import (
    UpstreamUrlError,
    build_upstream_client,
    resolve_validated,
    validate_upstream_url,
)


# ---------------- 单元：URL 结构校验 ----------------

def test_reject_non_http_schemes():
    for url in ("ftp://example.com", "file:///etc/passwd", "gopher://x", "ldap://x", "smb://x", ""):
        with pytest.raises(UpstreamUrlError):
            validate_upstream_url(url)


def test_reject_userinfo():
    with pytest.raises(UpstreamUrlError):
        validate_upstream_url("https://user:pass@example.com/v1")


def test_reject_missing_host():
    with pytest.raises(UpstreamUrlError):
        validate_upstream_url("https:///path")


def test_reject_non_string():
    with pytest.raises(UpstreamUrlError):
        validate_upstream_url(None)
    with pytest.raises(UpstreamUrlError):
        validate_upstream_url(123)


# ---------------- 单元：IP 直判（无需 DNS） ----------------

@pytest.mark.parametrize("ip", [
    "127.0.0.1", "127.1.2.3", "0.0.0.0",
    "10.0.0.1", "172.16.0.1", "172.31.255.255", "192.168.1.1",
    "169.254.169.254", "169.254.1.1",
    "100.64.0.1", "192.0.0.8", "198.18.0.1", "224.0.0.1", "240.0.0.1",
    "::1", "fe80::1", "fc00::1",
])
def test_reject_private_or_special_ips(ip):
    with pytest.raises(UpstreamUrlError):
        validate_upstream_url(f"http://{ip}/v1")


def test_ipv4_mapped_ipv6_loopback_rejected():
    with pytest.raises(UpstreamUrlError):
        validate_upstream_url("http://[::ffff:127.0.0.1]/v1")


def test_accept_public_ip():
    url = validate_upstream_url("http://8.8.8.8/v1")
    assert url == "http://8.8.8.8/v1"
    assert validate_upstream_url("https://1.1.1.1:8080/v1")

def test_reject_domain_resolving_to_private(monkeypatch):
    def fake_getaddrinfo(host, port, *args):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.10", port or 0))]
    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(UpstreamUrlError, match="非公网"):
        validate_upstream_url("https://internal.example.com/v1")


def test_reject_domain_with_any_private_ip(monkeypatch):
    def fake_getaddrinfo(host, port, *args):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", port or 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", port or 0)),
        ]
    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(UpstreamUrlError, match="非公网"):
        validate_upstream_url("https://mixed.example.com/v1")


def test_accept_domain_resolving_to_public(monkeypatch):
    def fake_getaddrinfo(host, port, *args):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", port or 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.4.4", port or 0)),
        ]
    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    assert validate_upstream_url("https://api.example.com/v1")


def test_reject_resolution_failure(monkeypatch):
    def fake_getaddrinfo(host, port, *args):
        raise socket.gaierror(-2, "Name or service not known")
    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    with pytest.raises(UpstreamUrlError, match="无法解析"):
        validate_upstream_url("https://nonexistent.invalid/v1")


# ---------------- 开关：ALLOW_PRIVATE=1 ----------------

def test_allow_private_switch_permits_private_ip(monkeypatch):
    monkeypatch.setattr(ssrf, "ALLOW_PRIVATE", True)
    assert validate_upstream_url("http://192.168.1.5:8000/v1")
    assert validate_upstream_url("http://127.0.0.1:8000/v1")
    # 结构校验仍然生效：协议 / userinfo 依然拒绝
    with pytest.raises(UpstreamUrlError):
        validate_upstream_url("ftp://192.168.1.5")
    with pytest.raises(UpstreamUrlError):
        validate_upstream_url("http://u:p@192.168.1.5/v1")


# ---------------- 集成：API 入口 ----------------

@pytest.fixture
def client():
    return TestClient(main_module.app)


def _start_payload(url: str) -> dict:
    return {
        "model_a": {"url": url, "key": "k", "name": "A", "temperature": 0.7, "max_tokens": 100},
        "model_b": {"url": url, "key": "k", "name": "B", "temperature": 0.7, "max_tokens": 100},
    }


def test_start_eval_rejects_private_url(client):
    r = client.post("/api/eval/start", json=_start_payload("http://127.0.0.1:8000/v1"))
    assert r.status_code == 400
    assert "校验失败" in r.json()["detail"]


def test_test_connection_reports_private_url(client):
    r = client.post("/api/test-connection", json={
        "models": [
            _start_payload("http://192.168.1.1:8000/v1")["model_a"],
            _start_payload("http://192.168.1.1:8000/v1")["model_b"],
        ],
    })
    assert r.status_code == 200
    results = r.json()["models"]
    assert len(results) == 2
    for item in results:
        assert item["ok"] is False
        assert "校验失败" in item["error"]


def test_start_eval_rejects_private_domain(monkeypatch, client):
    def fake_getaddrinfo(host, port, *args):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", port or 0))]
    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    r = client.post("/api/eval/start", json=_start_payload("http://internal.example/v1"))
    assert r.status_code == 400


# ---------------- 单元：resolve_validated（连接端复用同一判定） ----------------

def test_resolve_validated_ip_literal():
    assert resolve_validated("8.8.8.8") == ["8.8.8.8"]
    with pytest.raises(UpstreamUrlError, match="非公网"):
        resolve_validated("127.0.0.1")
    with pytest.raises(UpstreamUrlError, match="非公网"):
        resolve_validated("::1")


def test_resolve_validated_domain_revalidated_every_call(monkeypatch):
    calls = {"n": 0}

    def fake_getaddrinfo(host, port, *args):
        calls["n"] += 1
        ip = "8.8.8.8" if calls["n"] == 1 else "192.168.1.10"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    # 每次调用都重新解析（不缓存），第二次解析结果已变为内网
    assert resolve_validated("api.example.com") == ["8.8.8.8"]
    with pytest.raises(UpstreamUrlError, match="非公网"):
        resolve_validated("api.example.com")


# ---------------- 传输层：连接前解析校验（httpcore network_backend 扩展点） ----------------

class _FakeInnerBackend:
    """记录收到的连接参数，返回哨兵流。"""

    def __init__(self):
        self.calls = []
        self.raise_exc = None
        self.fail_count = 0    # 前 N 次调用抛 raise_exc（0 = 总是抛）
        self.slow = 0.0        # 失败调用先 sleep 该时长再抛（模拟耗时连接）
        self.hang = False      # 成功调用前挂起（模拟忽略 timeout 的挂死连接）
        self.cancel_on = None  # 第 N 次调用抛 asyncio.CancelledError

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        self.calls.append({
            "host": host, "port": port, "local_address": local_address, "timeout": timeout,
        })
        if self.cancel_on is not None and len(self.calls) == self.cancel_on:
            raise asyncio.CancelledError
        if self.raise_exc is not None and (self.fail_count == 0 or len(self.calls) <= self.fail_count):
            if self.slow:
                await asyncio.sleep(self.slow)
            raise self.raise_exc
        if self.hang:
            await asyncio.sleep(60)
        return "stream"

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):
        return "unix-stream"

    async def sleep(self, seconds):
        return None


def test_transport_connects_to_validated_ip(monkeypatch):
    inner = _FakeInnerBackend()
    backend = ssrf.ValidatingNetworkBackend(inner)
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, *args: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", port or 0)),
    ])

    async def run():
        return await backend.connect_tcp("api.example.com", 443, timeout=10)

    assert asyncio.run(run()) == "stream"
    # 内部后端收到的是校验后的 IP；TLS 由 httpcore 连接层用原域名 start_tls
    assert inner.calls == [{
        "host": "8.8.8.8", "port": 443, "local_address": None, "timeout": 10,
    }]


def test_transport_blocks_rebound_private_resolution(monkeypatch):
    inner = _FakeInnerBackend()
    backend = ssrf.ValidatingNetworkBackend(inner)
    # 门禁校验通过后，连接前一刻解析结果已回落到内网（重绑定）
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, *args: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port or 0)),
    ])

    async def run():
        await backend.connect_tcp("api.example.com", 443, timeout=10)

    with pytest.raises(httpx.ConnectError, match="SSRF 校验失败"):
        asyncio.run(run())
    assert inner.calls == [], "内网解析结果不得发起任何 TCP 连接"


def test_transport_blocks_resolution_failure(monkeypatch):
    inner = _FakeInnerBackend()
    backend = ssrf.ValidatingNetworkBackend(inner)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a: (_ for _ in ()).throw(
        socket.gaierror(-2, "Name or service not known"),
    ))

    async def run():
        await backend.connect_tcp("api.example.com", 443, timeout=10)

    with pytest.raises(httpx.ConnectError, match="SSRF 校验失败"):
        asyncio.run(run())
    assert inner.calls == []


def test_transport_rejects_mixed_public_private_resolution(monkeypatch):
    inner = _FakeInnerBackend()
    backend = ssrf.ValidatingNetworkBackend(inner)
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, *args: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", port or 0)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", port or 0)),
    ])

    async def run():
        await backend.connect_tcp("mixed.example.com", 443, timeout=10)

    with pytest.raises(httpx.ConnectError, match="SSRF 校验失败"):
        asyncio.run(run())
    assert inner.calls == [], "任一 IP 非公网即整体拒绝"


@pytest.mark.parametrize("exc_type", [OSError, httpcore.ConnectError, httpcore.ConnectTimeout])
def test_transport_retries_next_ip_on_connect_failure(monkeypatch, exc_type):
    # httpcore 生产环境把 OSError/超时映射为 ConnectError/ConnectTimeout，
    # 均非 OSError 子类；三种异常都必须能触发下一地址重试
    inner = _FakeInnerBackend()
    inner.raise_exc = exc_type("conn refused")
    backend = ssrf.ValidatingNetworkBackend(inner)
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, *args: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", port or 0)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.4.4", port or 0)),
    ])

    async def run():
        await backend.connect_tcp("api.example.com", 443, timeout=10)

    with pytest.raises(exc_type, match="conn refused"):
        asyncio.run(run())
    # 两个 IP 均尝试过；校验本身放行（均为公网）
    assert len(inner.calls) == 2


@pytest.mark.parametrize("first_failure", [
    httpcore.ConnectError("first ip refused"),
    httpcore.ConnectTimeout("first ip timed out"),
])
def test_transport_fails_over_to_second_ip(monkeypatch, first_failure):
    inner = _FakeInnerBackend()
    inner.raise_exc = first_failure
    inner.fail_count = 1
    backend = ssrf.ValidatingNetworkBackend(inner)
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, *args: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", port or 0)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.4.4", port or 0)),
    ])

    async def run():
        return await backend.connect_tcp("api.example.com", 443, timeout=10)

    # 首个地址失败后应返回第二个地址的连接结果
    assert asyncio.run(run()) == "stream"
    assert inner.calls == [
        {"host": "8.8.8.8", "port": 443, "local_address": None, "timeout": 5},
        {"host": "8.8.4.4", "port": 443, "local_address": None, "timeout": 5},
    ]
    assert len(inner.calls) == 2


def test_transport_propagates_cancellation(monkeypatch):
    inner = _FakeInnerBackend()
    inner.raise_exc = httpcore.ConnectError("must not mask cancellation")
    inner.cancel_on = 1
    backend = ssrf.ValidatingNetworkBackend(inner)
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, *args: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", port or 0)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.4.4", port or 0)),
    ])

    async def run():
        await backend.connect_tcp("api.example.com", 443, timeout=10)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())
    assert len(inner.calls) == 1, "取消异常不得被当作可重试连接失败"


def test_transport_total_timeout_budget_interrupts_hanging_attempt(monkeypatch):
    inner = _FakeInnerBackend()
    inner.hang = True
    backend = ssrf.ValidatingNetworkBackend(inner)
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, *args: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", port or 0)),
    ])

    async def run():
        await backend.connect_tcp("api.example.com", 443, timeout=0.3)

    start = time.monotonic()
    with pytest.raises(httpx.ConnectTimeout, match="总预算"):
        asyncio.run(run())
    elapsed = time.monotonic() - start
    assert len(inner.calls) == 1
    assert elapsed < 1.5, "挂死的连接尝试必须被总预算中断"


def test_transport_timeout_budget_shares_per_address(monkeypatch):
    # timeout 为总预算：每地址分得 timeout/len(ips)，首个地址超时后
    # 后续地址仍有剩余预算完成连接
    inner = _FakeInnerBackend()
    inner.raise_exc = httpcore.ConnectTimeout("first ip timed out at its share")
    inner.fail_count = 1
    inner.slow = 0.3
    backend = ssrf.ValidatingNetworkBackend(inner)
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, *args: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", port or 0)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.4.4", port or 0)),
    ])

    async def run():
        return await backend.connect_tcp("api.example.com", 443, timeout=2.0)

    start = time.monotonic()
    assert asyncio.run(run()) == "stream"
    elapsed = time.monotonic() - start
    # 每地址配额为 2.0/2 = 1.0s；首地址耗时 0.3s 后失败，次地址立即成功
    assert inner.calls == [
        {"host": "8.8.8.8", "port": 443, "local_address": None, "timeout": 1.0},
        {"host": "8.8.4.4", "port": 443, "local_address": None, "timeout": 1.0},
    ]
    assert 0.2 < elapsed < 1.5


def test_transport_allows_private_when_switch_on(monkeypatch):
    monkeypatch.setattr(ssrf, "ALLOW_PRIVATE", True)
    inner = _FakeInnerBackend()
    backend = ssrf.ValidatingNetworkBackend(inner)

    async def run():
        return await backend.connect_tcp("127.0.0.1", 8080, timeout=10)

    assert asyncio.run(run()) == "stream"
    assert inner.calls == [{
        "host": "127.0.0.1", "port": 8080, "local_address": None, "timeout": 10,
    }]


# ---------------- 客户端：302 重定向目标同样受限 ----------------

def test_client_does_not_follow_redirect_to_private_target(monkeypatch):
    monkeypatch.setattr(ssrf, "ALLOW_PRIVATE", True)
    hits = {"count": 0}

    class RedirectHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            hits["count"] += 1
            body = b""
            self.send_response(302)
            self.send_header("Location", "http://10.0.0.5/evil")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        async def run():
            async with build_upstream_client() as client:
                return await client.post(
                    f"http://127.0.0.1:{server.server_address[1]}/chat/completions",
                    json={"model": "x"}, timeout=10,
                )

        resp = asyncio.run(run())
        assert resp.status_code == 302
        assert hits["count"] == 1, "302 不得被跟随，内网重定向目标不得被访问"
    finally:
        server.shutdown()


# ---------------- 集成：重绑定攻击贯穿评测链路 ----------------
#
# 说明：TestClient 每次请求使用独立 portal 事件循环，start_eval 内 create_task
# 的后台协程不会在测试期间被调度（这正是 issue #12 评审点出的泄漏告警根源），
# 故此处直接驱动与生产完全相同的执行路径：
#   启动门禁校验（validate_upstream_url）→ execute_all → _execute_model
#   → build_upstream_client → ValidatingTransport（连接前解析校验）

from backend.engine.executor import execute_all
from backend.engine.tasks import build_task_set

def _run_execute_all(config_a, config_b, task_set):
    async def run():
        return await execute_all(task_set, config_a=config_a, config_b=config_b)

    return asyncio.run(run())


def test_execute_all_blocks_dns_rebinding_at_connect_time(monkeypatch):
    getaddrinfo_calls = {"n": 0}
    connect_calls = {"n": 0}

    def fake_getaddrinfo(host, port, *args):
        getaddrinfo_calls["n"] += 1
        # 第 1 次（启动门禁）解析为公网放行，之后回落内网（重绑定）
        ip = "8.8.8.8" if getaddrinfo_calls["n"] == 1 else "127.0.0.1"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 0))]

    async def fake_connect_tcp(**kwargs):
        connect_calls["n"] += 1
        raise OSError("unreachable")

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(anyio, "connect_tcp", fake_connect_tcp)

    # 启动门禁：此时解析仍为公网，放行
    assert validate_upstream_url("http://api.example.com/v1") == "http://api.example.com/v1"

    config = {
        "url": "http://api.example.com/v1", "key": "k", "name": "M",
        "temperature": 0.7, "max_tokens": 100, "top_p": None, "code_verify_mode": "off",
    }
    task_set = build_task_set(dims=["知识能力"], num_questions=1, seed=1)
    answers_a, answers_b = _run_execute_all(config, dict(config), task_set)

    # 连接前重新解析被触发，且全程未放行任何 TCP 连接
    assert getaddrinfo_calls["n"] >= 3
    assert connect_calls["n"] == 0
    for answers in (answers_a, answers_b):
        assert "SSRF 校验失败" in answers["answers"][0]["api_info"]["error"]


def test_execute_all_allows_private_upstream(monkeypatch):
    monkeypatch.setattr(ssrf, "ALLOW_PRIVATE", True)
    hits = {"count": 0}

    class ChatHandler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            hits["count"] += 1
            body = json.dumps({
                "choices": [{"message": {"content": "ok-answer"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                "model": "local",
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), ChatHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        config = {
            "url": f"http://127.0.0.1:{server.server_address[1]}/v1", "key": "k", "name": "M",
            "temperature": 0.7, "max_tokens": 100, "top_p": None, "code_verify_mode": "off",
        }
        task_set = build_task_set(dims=["知识能力"], num_questions=1, seed=1)
        answers_a, answers_b = _run_execute_all(config, dict(config), task_set)
        for answers in (answers_a, answers_b):
            assert answers["answers"][0]["api_info"]["status"] == "ok"
            assert answers["answers"][0]["raw_answer"] == "ok-answer"
        # 每个模型至少一次真实调用；Windows 本地 server 偶发瞬时读取失败由
        # 应用侧重试吸收（api_info 仍为 ok），故不做精确次数断言
        assert hits["count"] >= 2
    finally:
        server.shutdown()
