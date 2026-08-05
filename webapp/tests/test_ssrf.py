# -*- coding: utf-8 -*-
"""SSRF 防护测试（issue #8）：上游 URL 公网性校验。"""
from __future__ import annotations

import socket

import pytest
from fastapi.testclient import TestClient

from backend import main as main_module
from backend import ssrf
from backend.ssrf import UpstreamUrlError, validate_upstream_url


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
    r = client.post("/api/test-connection", json=_start_payload("http://192.168.1.1:8000/v1"))
    assert r.status_code == 200
    for label in ("model_a", "model_b"):
        assert r.json()[label]["ok"] is False
        assert "校验失败" in r.json()[label]["error"]


def test_start_eval_rejects_private_domain(monkeypatch, client):
    def fake_getaddrinfo(host, port, *args):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", port or 0))]
    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    r = client.post("/api/eval/start", json=_start_payload("http://internal.example/v1"))
    assert r.status_code == 400
