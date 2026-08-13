# -*- coding: utf-8 -*-
"""访问控制中间件：认证 / Host 校验 / Origin 校验 / 写请求限流（issue #8）。

两种部署模式：
- 单机模式（未设置 MODEL_DUEL_TOKEN）：不强制认证，但校验 Host 为回环别名
  （localhost/127.0.0.1/::1/testserver），阻止 DNS rebinding 与局域网直连。
- 共享模式（已设置 MODEL_DUEL_TOKEN）：全部 /api/* 需 Bearer token（compare_digest
  恒定时间比较）；跳过 Host 校验（局域网 IP 访问合法）；启用写请求限流。
- SSE 进度流（/events）：EventSource 无法携带自定义 header，改用短时单次
  ticket（POST /api/eval/{job_id}/events/ticket 签发，见 sse_ticket.py），
  长期 Token 不再出现在任何 URL 中。迭代八：任务视图流（/api/tasks/events）
  同机制，ticket 作用域固定为 "tasks"。

Origin 校验仅作用于写方法（POST/PUT/DELETE/PATCH）：Origin 存在则必须与请求
Host 同源；无 Origin 时 Referer 必须同源；两者皆无（curl/TestClient）放行，
由认证兜底。OPTIONS（CORS preflight）直接放行。
"""
from __future__ import annotations

import os
import re
import secrets
import time
from collections import deque
from urllib.parse import urlsplit

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from backend import audit
from backend import sse_ticket

# 单机模式允许的 Host（testserver 为 TestClient 默认别名）
SINGLE_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]", "testserver"})

_HOST_PORT_RE = re.compile(r"^(\[[^]]*\]|[^\[\]:]+)(?::\d{1,5})?$")

# 从 /api/eval/{job_id}/events 提取 job_id（ticket 作用域校验用）
_EVENTS_JOB_RE = re.compile(r"^/api/eval/([^/]+)/events$")
# 迭代八：任务视图 SSE 路径（ticket 作用域固定为 "tasks"）
_TASKS_EVENTS_RE = re.compile(r"^/api/tasks/events$")

WRITE_METHODS = frozenset({"POST", "PUT", "DELETE", "PATCH"})

RATE_LIMIT_MAX = int(os.environ.get("MODEL_DUEL_RATE_LIMIT", "30"))
RATE_LIMIT_WINDOW = 60.0

# 限流追踪的 IP 上限：防止恶意大量 IP 写入使 _hits 无限增长
MAX_TRACKED_IPS = 10_000

# 共享模式下同样受保护的接口文档路径（单机模式保持开放便于开发调试）
DOCS_PATHS = frozenset({"/docs", "/redoc", "/openapi.json"})

_hits: dict[str, deque[float]] = {}


def _host_allowed(host: str) -> bool:
    h = (host or "").strip().lower()
    # 剥端口（IPv6 字面量须带方括号）：只允许「主机[:端口]」形态
    m = _HOST_PORT_RE.match(h)
    if m:
        h = m.group(1)
    if h.startswith("[") and h.endswith("]"):
        h = h[1:-1]
    return h in SINGLE_HOSTS


def _same_origin(request: Request) -> bool:
    """Origin/Referer 必须与请求 Host 同源；两者皆无视为非浏览器请求，放行。"""
    req_host = (request.headers.get("host") or "").lower()
    origin = request.headers.get("origin")
    if origin:
        return urlsplit(origin).netloc.lower() == req_host
    referer = request.headers.get("referer")
    if referer:
        return urlsplit(referer).netloc.lower() == req_host
    return True


def _rate_limit(ip: str) -> bool:
    """滑动窗口限流：每 IP 每窗口最多 RATE_LIMIT_MAX 次，超限返回 False。

    追踪表有 MAX_TRACKED_IPS 容量上限（优先淘汰已过期的空队列），防无限增长。
    """
    now = time.monotonic()
    if ip not in _hits and len(_hits) >= MAX_TRACKED_IPS:
        stale = next((k for k, q in _hits.items() if not q), None)
        _hits.pop(stale if stale is not None else next(iter(_hits)))
    q = _hits.setdefault(ip, deque())
    while q and now - q[0] > RATE_LIMIT_WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT_MAX:
        return False
    q.append(now)
    return True


def _job_id_from_events_path(path: str) -> str:
    """从事件路径提取 ticket 作用域串：/api/eval/{job_id}/events → job_id；
    /api/tasks/events → "tasks"；格式不符返回空串（consume 必然失败）。"""
    m = _EVENTS_JOB_RE.match(path)
    if m:
        return m.group(1)
    if _TASKS_EVENTS_RE.match(path):
        return "tasks"
    return ""


def _strip_query_param(qs: bytes, name: bytes) -> bytes:
    """从 query string 中移除指定参数（uvicorn access log 读取 scope["query_string"]）。"""
    if not qs:
        return qs
    parts = [p for p in qs.split(b"&") if not p.startswith(name + b"=")]
    return b"&".join(parts)


async def security_middleware(request: Request, call_next):
    """保护 /api/* 与共享模式下的接口文档；页面与静态资源不拦截。"""
    path = request.url.path
    if not (path.startswith("/api/") or path in DOCS_PATHS):
        return await call_next(request)

    # 认证材料剥离（R3-001 残余 1）：无论认证成败、无论部署模式，
    # ticket / token 查询参数都不进入 uvicorn/代理访问日志。
    # 先取值再剥离——request.query_params 是 cached_property，
    # 首次访问缓存原始解析结果，后续 consume 仍能拿到 ticket。
    ticket = request.query_params.get("ticket", "")
    if ticket:
        request.scope["query_string"] = _strip_query_param(
            request.scope.get("query_string", b""), b"ticket"
        )
    if "token" in request.query_params:
        request.scope["query_string"] = _strip_query_param(
            request.scope.get("query_string", b""), b"token"
        )

    token = os.environ.get("MODEL_DUEL_TOKEN", "")
    shared = bool(token)

    # 1. 认证（共享模式）/ Host 校验（单机模式）
    if shared:
        auth = request.headers.get("authorization", "")
        ok = auth.startswith("Bearer ") and secrets.compare_digest(auth[7:].strip(), token)
        if not ok and path.endswith("/events") and ticket:
            # EventSource 无法禁用自动重连：断线重连会复用已消费 ticket。
            # 对 ticket 认证失败静默 204、不记审计（R3-001 残余 2）——
            # ticket 为 128 位随机单次凭证，重放/伪造无利用价值；
            # 未携带 ticket 的请求仍走下方 401 + 审计。
            ok = sse_ticket.consume(ticket, _job_id_from_events_path(path))
            if not ok:
                return Response(status_code=204)
        if not ok:
            audit.auth_failed(path)
            return JSONResponse({"detail": "未授权：需要有效的访问令牌"}, status_code=401)
    elif not _host_allowed(request.headers.get("host", "")):
        return JSONResponse({"detail": "非法 Host 头"}, status_code=403)

    # 2. Origin 校验（写方法）
    if request.method in WRITE_METHODS and not _same_origin(request):
        return JSONResponse({"detail": "跨源写请求被拒绝"}, status_code=403)

    # 3. 写限流（仅共享模式；单机本机信任）
    if shared and request.method in WRITE_METHODS:
        ip = request.client.host if request.client else "unknown"
        if not _rate_limit(ip):
            return JSONResponse({"detail": "请求过于频繁，请稍后重试"}, status_code=429)

    return await call_next(request)
