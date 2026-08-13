# -*- coding: utf-8 -*-
"""Issue #7：CSP 响应头与页面脚本/事件处理器静态审计（轻量断言）。

配合 test_xss_playwright.py 的浏览器级验证，此处保证：
- 三个页面均返回 CSP 头且 nonce 每请求轮换
- 页面内全部 <script>（含外链）均带 nonce
- 页面不存在任何内联事件处理器（onclick/onchange/...）
- echarts 已本地化（不再依赖外网 CDN）
"""
from __future__ import annotations

import re

from fastapi.testclient import TestClient
import pytest

from backend.main import app

client = TestClient(app)

PAGES = ["/", "/report.html", "/review.html", "/gen_review.html",
         "/badcases.html", "/perturb.html", "/leaderboard.html",
         "/dashboard.html"]

INLINE_EVENT_RE = re.compile(r'\son\w+\s*=')


def test_pages_serve_csp_header():
    for path in PAGES:
        r = client.get(path)
        assert r.status_code == 200
        csp = r.headers.get("content-security-policy")
        assert csp is not None, f"{path} 缺少 CSP 头"
        assert "script-src 'self' 'nonce-" in csp
        assert "connect-src 'self'" in csp
        assert "frame-ancestors 'none'" in csp
        assert "object-src 'none'" in csp
        # 'unsafe-inline' 仅允许出现在 style-src
        rest = csp.replace("style-src 'self' 'unsafe-inline'", "")
        assert "'unsafe-inline'" not in rest, f"{path} CSP 放行了 script 内联: {csp}"


def test_all_script_tags_carry_nonce():
    for path in PAGES:
        r = client.get(path)
        scripts = re.findall(r"<script\b[^>]*>", r.text)
        assert scripts, f"{path} 未找到任何 <script>"
        for tag in scripts:
            assert "nonce=" in tag, f"{path} 脚本缺少 nonce: {tag}"
        # 不允许 script 标签带不受控属性（如 onload）
        for tag in scripts:
            assert INLINE_EVENT_RE.search(tag) is None, f"{path} script 标签含内联处理器: {tag}"


def test_no_inline_event_handlers_in_markup():
    for path in PAGES:
        r = client.get(path)
        for m in re.finditer(r"<[a-z][a-z0-9-]*\b[^>]*>", r.text):
            tag = m.group(0)
            if INLINE_EVENT_RE.search(tag):
                raise AssertionError(f"{path} 存在内联事件处理器: {tag}")


def test_nonce_rotates_per_request():
    a = client.get("/")
    b = client.get("/")
    na = re.search(r'nonce="([^"]+)"', a.text).group(1)
    nb = re.search(r'nonce="([^"]+)"', b.text).group(1)
    assert na and nb
    assert na != nb, "nonce 未随请求轮换"


def test_echarts_served_locally():
    r = client.get("/report.html")
    assert "/static/echarts.min.js" in r.text, "echarts 应改为本地引用"
    assert "https://" not in r.text and "http://" not in r.text, "页面不应引用外部资源"
    sr = client.get("/static/echarts.min.js")
    assert sr.status_code == 200
    assert sr.text.lstrip().startswith("/*"), "echarts.min.js 内容异常（可能为错误页）"
