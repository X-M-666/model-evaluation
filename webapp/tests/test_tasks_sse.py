# -*- coding: utf-8 -*-
"""迭代八：任务视图 SSE（/api/tasks/events + ticket 作用域 "tasks"）。

- ticket 签发端点（scope="tasks"）；
- 广播器：订阅 → _notify_tasks_view → 收到 task_view ping；失效连接清理；
- access 双路径解析：_job_id_from_events_path 对 eval 路径与 tasks 路径；
- 共享模式：/api/tasks/events 未带 ticket 401+审计、带错作用域 ticket 204 静默；
- 触发点：priority 变更 / 排队取消 / batch cancel 后广播可被订阅者收到。
"""
from __future__ import annotations

import asyncio
import json
import os

import pytest
from fastapi.testclient import TestClient

from backend import access
from backend import audit
from backend import main as main_module
from backend import storage
from backend import sse_ticket
from backend.schemas import BenchmarkRequest


@pytest.fixture
def client():
    return TestClient(main_module.app)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    main_module._jobs.clear()
    main_module._tasks.clear()
    main_module._SCHEDULER.clear()
    main_module._TASKS_EVENT_SUBS.clear()
    audit._log_path().write_text("", encoding="utf-8")
    yield
    main_module._jobs.clear()
    main_module._tasks.clear()
    main_module._SCHEDULER.clear()
    main_module._TASKS_EVENT_SUBS.clear()


def test_ticket_endpoint_issues_tasks_scope(client):
    r = client.post("/api/tasks/events/ticket")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["ticket"]
    # 作用域校验：仅 tasks 路径可消费
    assert sse_ticket.consume(d["ticket"], "tasks") is True
    assert sse_ticket.consume(d["ticket"], "other") is False   # 消费即焚 + 作用域


def test_broadcast_delivers_task_view():
    async def _scenario():
        q = asyncio.Queue(maxsize=16)
        loop = asyncio.get_running_loop()
        main_module._TASKS_EVENT_SUBS.add((loop, q))
        main_module._notify_tasks_view()
        evt = await asyncio.wait_for(q.get(), timeout=1)
        main_module._TASKS_EVENT_SUBS.discard((loop, q))
        return evt
    evt = asyncio.run(_scenario())
    assert evt["type"] == "task_view"
    assert evt["at"]


def test_broadcast_ignores_stale_loop():
    """失效 loop（已关闭）广播不抛错、不影响其余订阅者。"""
    async def _scenario():
        q = asyncio.Queue(maxsize=16)
        loop = asyncio.get_running_loop()
        main_module._TASKS_EVENT_SUBS.add((loop, q))
        main_module._TASKS_EVENT_SUBS.add((object(), asyncio.Queue(maxsize=1)))  # 伪 loop
        main_module._notify_tasks_view()
        evt = await asyncio.wait_for(q.get(), timeout=1)
        return evt
    assert asyncio.run(_scenario())["type"] == "task_view"


def test_access_path_scope_parsing():
    assert access._job_id_from_events_path("/api/eval/20260101_000000_abc123/events") == "20260101_000000_abc123"
    assert access._job_id_from_events_path("/api/tasks/events") == "tasks"
    assert access._job_id_from_events_path("/api/other/events") == ""
    assert access._job_id_from_events_path("/api/tasks/events/extra") == ""


def test_events_route_auth_shared_mode(client, monkeypatch):
    """共享模式：/api/tasks/events 未带 ticket → 401+审计；错作用域 ticket → 204 静默。"""
    monkeypatch.setenv("MODEL_DUEL_TOKEN", "tok-secret")
    # 中间件在 TestClient 请求时读取 env（access 模块缓存在函数内读 env，需重置）
    r = client.get("/api/tasks/events")
    assert r.status_code == 401
    events = audit.read_events()
    assert any(e["event"] == "auth_failed" for e in events)

    # 用 eval 路径签发的 ticket 打错作用域 → consume 失败 → 204 静默（无审计增长）
    before = len(events)
    t = sse_ticket.issue("20260101_000000_abc123")
    r2 = client.get(f"/api/tasks/events?ticket={t}")
    assert r2.status_code == 204
    assert len(audit.read_events()) == before


def test_events_stream_single_loop(client):
    """流路由单循环直驱：订阅注册于当前循环，广播后流内收到 task_view 事件。
    （HTTP 全链路鉴权见 test_access_control::test_shared_mode_tasks_events_auth；
    TestClient 每请求独立 portal 循环与流式读取不兼容，故直驱路由函数验证机制。）"""
    async def _scenario():
        resp = await main_module.tasks_events()      # 同一循环内订阅（route 函数）
        assert len(main_module._TASKS_EVENT_SUBS) == 1, "订阅未注册"
        try:
            async def _ping():
                await asyncio.sleep(0.05)
                main_module._notify_tasks_view()
            asyncio.get_running_loop().create_task(_ping())
            got: list[str] = []
            async for chunk in resp.body_iterator:
                text = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
                got.append(text)
                if "task_view" in text:
                    break
            return got
        finally:
            main_module._TASKS_EVENT_SUBS.clear()
    got = asyncio.run(_scenario())
    assert any("task_view" in t for t in got)
    assert any('"type": "task_view"' in t for t in got)


def test_priority_change_queued_triggers_broadcast(client):
    from backend.scheduler import Scheduler
    sched = Scheduler(concurrency=1)
    client_orig = main_module._SCHEDULER
    main_module._SCHEDULER = sched
    jid = "20250901_000000_abc123"
    main_module._jobs[jid] = {"state": "queued", "config": {}}
    sched.submit(jid)
    q = asyncio.Queue(maxsize=16)
    loop = None
    try:
        async def _scenario():
            nonlocal loop
            loop = asyncio.get_running_loop()
            main_module._TASKS_EVENT_SUBS.add((loop, q))
            from backend.schemas import PriorityRequest
            res = await main_module.tasks_set_priority(jid, PriorityRequest(priority=5))
            evt = await asyncio.wait_for(q.get(), timeout=1)
            return res, evt
        res, evt = asyncio.run(_scenario())
        assert res["priority"] == 5
        assert evt["type"] == "task_view"
    finally:
        if loop is not None:
            main_module._TASKS_EVENT_SUBS.discard((loop, q))
        main_module._jobs.pop(jid, None)
        main_module._SCHEDULER = client_orig
