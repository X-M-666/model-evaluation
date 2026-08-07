# -*- coding: utf-8 -*-
"""历史/评测接口 job_id 路径穿越回归测试（issue #17 / R3-001）。

覆盖：
- 编码（%2E%2E）与非编码（..）的 `.`/`..`、非法 ID 均返回明确的 4xx；
- 任何历史接口都不能读取、创建或删除 BASE_DIR 之外的路径；
- 删除合法 job 仍只影响该 job 目录；
- 哨兵文件（模拟 .eval/audit.log）与父目录在攻击尝试后保持不变。

说明：httpx/TestClient 会先按 RFC 3986 折叠 URL 中的明文点段（.. 被
归一化），无法把明文 ``..`` 送达路由；issue 的实达路径是编码形式
（%2E%2E 由 ASGI 路由解码），故编码用例走 TestClient，明文用例用
真实 uvicorn + urllib（urllib 不做点段规范化）验证。
"""
from __future__ import annotations

import asyncio
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest
import uvicorn
from fastapi.testclient import TestClient

from backend import main as main_module
from backend import storage

# 编码形式绕过客户端规范化可到达应用路由（issue 复现路径）
ENCODED_BAD = ["%2E%2E", "%2E", "abc", "bad-id", "%2E%2E%2F%2E%2E"]

# 明文形式（raw HTTP 发送，不经点段规范化）
LITERAL_BAD = ["..", "."]


def _seed_sentinels():
    """在 BASE_DIR 父目录放置哨兵（模拟 .eval/audit.log 与邻接文件）。"""
    parent = storage.BASE_DIR.parent
    storage.BASE_DIR.mkdir(parents=True, exist_ok=True)
    audit_log = parent / "audit.log"
    sentinel = parent / "sentinel.txt"
    audit_log.write_text("audit-line\n", encoding="utf-8")
    sentinel.write_text("keep", encoding="utf-8")
    return audit_log, sentinel


def _assert_intact(audit_log, sentinel):
    assert storage.BASE_DIR.is_dir()
    assert audit_log.read_text(encoding="utf-8") == "audit-line\n"
    assert sentinel.read_text(encoding="utf-8") == "keep"


@pytest.fixture
def client():
    return TestClient(main_module.app)


# ---------------- 编码穿越（TestClient） ----------------

def test_history_get_encoded_bad_ids_4xx(client):
    audit_log, sentinel = _seed_sentinels()
    for bad in ENCODED_BAD:
        r = client.get(f"/api/history/{bad}")
        # %2F 被客户端解码为 / 时该路径不匹配任何路由 → 404，同为明确 4xx
        assert r.status_code in (400, 404), (bad, r.status_code)
    _assert_intact(audit_log, sentinel)


def test_history_delete_encoded_bad_ids_4xx_sentinels_kept(client):
    audit_log, sentinel = _seed_sentinels()
    for bad in ENCODED_BAD:
        r = client.delete(f"/api/history/{bad}")
        assert r.status_code in (400, 404), (bad, r.status_code)
    _assert_intact(audit_log, sentinel)


def test_eval_routes_bad_job_id_4xx(client):
    audit_log, sentinel = _seed_sentinels()
    bad = "%2E%2E"
    cases = [
        ("get", f"/api/eval/{bad}/status"),
        ("get", f"/api/eval/{bad}/review"),
        ("get", f"/api/eval/{bad}/report"),
        ("get", f"/api/eval/{bad}/events"),
        ("post", f"/api/eval/{bad}/events/ticket"),
    ]
    for method, path in cases:
        r = getattr(client, method)(path)
        assert r.status_code == 400, (method, path, r.status_code)
    # POST review 无请求体会先被 body 校验 422，同样属于明确 4xx
    r = client.post(f"/api/eval/{bad}/review")
    assert r.status_code in (400, 422)
    _assert_intact(audit_log, sentinel)


def test_delete_valid_job_only_removes_own_dir(client):
    audit_log, sentinel = _seed_sentinels()
    victim, survivor = storage.create_job_id(), storage.create_job_id()
    for jid in (victim, survivor):
        d = storage.BASE_DIR / jid
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text("{}", encoding="utf-8")

    r = client.delete(f"/api/history/{victim}")
    assert r.status_code == 200, r.status_code
    assert not (storage.BASE_DIR / victim).exists()
    assert (storage.BASE_DIR / survivor).exists()
    # audit.log 未被删除，仅被追加一条 history_deleted 审计事件（前缀保留）
    assert audit_log.read_text(encoding="utf-8").startswith("audit-line\n")
    assert sentinel.read_text(encoding="utf-8") == "keep"


# ---------------- 明文穿越（真实 uvicorn + urllib，无点段规范化） ----------------

def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def live_server():
    from backend import storage  # noqa: F401  确保模块级存储重定向已生效

    port = _free_port()
    config = uvicorn.Config(main_module.app, host="127.0.0.1", port=port, access_log=False)
    server = uvicorn.Server(config)
    thread = threading.Thread(
        target=lambda: asyncio.run(server.serve()), daemon=True
    )
    thread.start()

    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            urllib.request.urlopen(base + "/api/dims", timeout=1)
            break
        except urllib.error.HTTPError:
            break  # 4xx 说明服务已就绪
        except Exception:
            time.sleep(0.1)
    else:
        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError("uvicorn server 未能启动")

    yield {"url": base}

    server.should_exit = True
    thread.join(timeout=10)


def _raw_request(base: str, path: str, method: str = "GET"):
    try:
        urllib.request.urlopen(
            urllib.request.Request(base + path, method=method), timeout=5
        )
    except urllib.error.HTTPError as e:
        return e.code
    return 200


def test_literal_dotdot_raw_http_4xx(live_server):
    """明文 .. 经原始 HTTP 客户端送达应用路由，GET/DELETE 均被拒绝且哨兵完好。"""
    audit_log, sentinel = _seed_sentinels()
    base = live_server["url"]
    for raw in LITERAL_BAD:
        assert _raw_request(base, f"/api/history/{raw}") == 400, raw
    for raw in LITERAL_BAD:
        assert _raw_request(base, f"/api/history/{raw}", method="DELETE") == 400, raw
    _assert_intact(audit_log, sentinel)
