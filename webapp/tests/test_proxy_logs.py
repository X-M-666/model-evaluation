# -*- coding: utf-8 -*-
"""真实 uvicorn 访问日志验证（R3-001 残余 3 修复）。

同进程后台线程运行真实 uvicorn.Server（共享模式），捕获 uvicorn.access
日志，断言 /events 的 401/204/200 请求与正常 API 请求的访问日志中：
- 不出现 ticket 查询参数（无条件剥离，含认证失败路径）
- 不出现长期 Token（含误把 Token 放 URL 的 401 路径）

覆盖「代理/WAF 先看到 URL」的链路：middleware 对 scope["query_string"]
的剥离必须在 uvicorn 写访问日志前生效。
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest
import uvicorn

from backend import main as main_module

TOKEN = "proxy-test-token-abc123"


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _CaptureHandler(logging.StreamHandler):
    """uvicorn dictConfig 可实例化的日志捕获 handler（写入模块级缓冲）。"""


LOG_BUF = io.StringIO()


def _log_config() -> dict:
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"access": {"format": "%(message)s"}},
        "handlers": {
            "access": {"class": "test_proxy_logs._CaptureHandler",
                       "formatter": "access", "stream": "ext://sys.stderr"},
        },
        "loggers": {
            "uvicorn": {"level": "INFO", "propagate": False},
            "uvicorn.error": {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
        },
    }


@pytest.fixture(scope="module")
def live_server():
    """启动真实 uvicorn（共享模式），捕获 access 日志到 LOG_BUF。"""
    from backend import storage  # noqa: F401  确保模块级存储重定向已生效

    old_token = os.environ.get("MODEL_DUEL_TOKEN")
    os.environ["MODEL_DUEL_TOKEN"] = TOKEN
    port = _free_port()
    LOG_BUF.seek(0)
    LOG_BUF.truncate()

    config = uvicorn.Config(
        main_module.app, host="127.0.0.1", port=port,
        log_config=_log_config(), access_log=True,
    )
    server = uvicorn.Server(config)
    # 直接替换 handler 的 stream：dictConfig 已把 access 日志指向 _CaptureHandler
    for h in logging.getLogger("uvicorn.access").handlers:
        if isinstance(h, _CaptureHandler):
            h.setStream(LOG_BUF)

    thread = threading.Thread(target=lambda: asyncio.run(server.serve()), daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            urllib.request.urlopen(base + "/api/dims", timeout=1)
            break
        except urllib.error.HTTPError:
            break  # 401 说明服务已就绪
        except Exception:
            time.sleep(0.1)
    else:
        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError("uvicorn server 未能启动")

    yield {"url": base}

    server.should_exit = True
    thread.join(timeout=10)
    if old_token is None:
        os.environ.pop("MODEL_DUEL_TOKEN", None)
    else:
        os.environ["MODEL_DUEL_TOKEN"] = old_token


def _read_log() -> str:
    time.sleep(0.1)  # access 日志由服务端线程在响应后异步写入
    return LOG_BUF.getvalue()


def test_access_log_never_contains_ticket_or_token(live_server):
    import httpx

    base = live_server["url"]
    headers = {"Authorization": f"Bearer {TOKEN}"}

    with httpx.Client() as c:
        # 1) 无凭据 → 401（Bearer 失败路径，应保留审计与 401）
        assert c.get(base + "/api/dims").status_code == 401
        # 2) 合法认证 → 200
        assert c.get(base + "/api/dims", headers=headers).status_code == 200
        # 3) 创建 mock 任务
        r = c.post(base + "/api/eval/mock", headers=headers)
        assert r.status_code == 200
        job_id = r.json()["job_id"]
        # 4) 伪造 ticket → 204 静默
        assert c.get(f"{base}/api/eval/{job_id}/events?ticket=forged").status_code == 204
        # 5) 签发真 ticket → 200，随后重放 → 204
        tk = c.post(f"{base}/api/eval/{job_id}/events/ticket", headers=headers).json()["ticket"]
        main_module._jobs[job_id]["sse_queue"].put_nowait({"type": "test", "state": "error"})
        assert c.get(f"{base}/api/eval/{job_id}/events?ticket={tk}").status_code == 200
        assert c.get(f"{base}/api/eval/{job_id}/events?ticket={tk}").status_code == 204
        # 6) 误把长期 Token 放 URL → 401（URL 中的 token 必须被剥离后再进日志）
        assert c.get(f"{base}/api/eval/{job_id}/events?token={TOKEN}").status_code == 401

    log_text = _read_log()
    # 各状态码均已到达真实服务器（证明请求确实经过 access log）
    for status in ("204", "401", "200"):
        assert f'HTTP/1.1" {status}' in log_text, f"access log 缺少 {status} 状态记录:\n{log_text}"
    assert "ticket=" not in log_text, f"access log 泄露 ticket:\n{log_text}"
    assert TOKEN not in log_text, f"access log 泄露长期 Token:\n{log_text}"


def test_access_log_ticket_stripped_on_401_too(live_server):
    """401（认证失败）路径同样不记录 ticket（R3-001 残余 1 回归）。"""
    import httpx

    base = live_server["url"]
    with httpx.Client() as c:
        r = c.get(base + "/api/dims?ticket=stripped-please")
        assert r.status_code == 401
    log_text = _read_log()
    assert "ticket=" not in log_text, f"401 路径 access log 泄露 ticket:\n{log_text}"
