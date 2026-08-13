# -*- coding: utf-8 -*-
"""迭代七：任务队列视图 / 优先级 / 排队取消 / 重启沉降（tasks API）集成测试。

覆盖不变量：
- GET /api/tasks：queued（位置/优先级）+ running + quota + batches 视图
- PUT /api/tasks/{id}/priority：queued 可改（重排序 + 审计）；运行中/终态 409；
  不存在 404；非法 id 400
- DELETE /api/history/{id} 排队取消分支：出队 + 删目录 + 审计（无 task.cancel）
- lifespan 重启沉降：磁盘态 queued job → error（提示不自动恢复）
"""
from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend import audit
from backend import main as main_module
from backend import storage
from backend.schemas import PriorityRequest, StartRequest

PUBLIC_URL = "https://8.8.8.8/v1"


def _payload() -> dict:
    return {"model_a": {"url": PUBLIC_URL, "key": "k", "name": "A",
                        "temperature": 0.7, "max_tokens": 100},
            "model_b": {"url": PUBLIC_URL, "key": "k", "name": "B",
                        "temperature": 0.7, "max_tokens": 100}}


def _call(fn, *a, **k):
    return asyncio.run(fn(*a, **k))


@pytest.fixture
def client():
    return TestClient(main_module.app)


@pytest.fixture(autouse=True)
def _clean():
    main_module._jobs.clear()
    main_module._tasks.clear()
    main_module._SCHEDULER.clear()
    audit._log_path().write_text("", encoding="utf-8")
    yield
    main_module._jobs.clear()
    main_module._tasks.clear()
    main_module._SCHEDULER.clear()


def _queue_job(client, name="排队模型") -> str:
    """超配额提交 → 排队中的任务。"""
    main_module._SCHEDULER.clear()
    main_module._jobs.clear()
    for i in range(2):  # 填满配额
        jid = f"slot-{i}"
        main_module._jobs[jid] = {"state": "executing",
                                  "config": {"model_a": {"url": PUBLIC_URL, "name": "s"}}}
        main_module._SCHEDULER.submit(jid)
        main_module._SCHEDULER.next_batch()
    r = client.post("/api/eval/start", json=_payload())
    assert r.status_code == 200, r.text
    jid = r.json()["job_id"]
    assert main_module._jobs[jid]["state"] == "queued"
    return jid


# ---- GET /api/tasks ----

def test_tasks_view(client):
    jid = _queue_job(client)
    body = client.get("/api/tasks").json()
    assert body["quota"]["concurrency"] == 2
    assert body["quota"]["active"] == 2
    assert body["quota"]["queued"] == 1
    assert body["queued"][0]["job_id"] == jid
    assert body["queued"][0]["position"] == 1
    assert body["queued"][0]["priority"] == 0
    assert len(body["running"]) == 2
    assert isinstance(body["batches"], list)


def test_tasks_view_shows_batch_jobs(client, monkeypatch):
    """batch 执行单元在任务视图中标记 type=batch。"""
    from backend.schemas import BenchmarkRequest
    from backend.models_registry import register
    register("b模型1", PUBLIC_URL, key="k")
    register("b模型2", PUBLIC_URL, key="k")
    storage.save_dataset("批次集X", {"name": "批次集X", "tasks": [
        {"id": "T1", "type": "判别式", "dimension": "知识能力",
         "prompt": "p", "test_cases": [{"input": "i", "expected": "e"}]}]})
    main_module._SCHEDULER.clear()
    main_module._jobs.clear()
    resp = _call(main_module.create_benchmark, BenchmarkRequest(
        dataset_name="批次集X", model_ids=["b模型1", "b模型2"], rounds=1))
    body = client.get("/api/tasks").json()
    batch_ids = {x["job_id"] for x in body["running"] if x.get("type") == "batch"}
    assert set(resp["jobs"]) == batch_ids


# ---- PUT /api/tasks/{id}/priority ----

def test_set_priority_requeues(client):
    jid = _queue_job(client)
    r = client.put(f"/api/tasks/{jid}/priority",
                   json=PriorityRequest(priority=10).model_dump())
    assert r.status_code == 200
    assert r.json()["priority"] == 10
    assert main_module._SCHEDULER.queue_view()[0]["job_id"] == jid
    events = audit.read_events()
    assert any(e["event"] == "priority_changed" and e["target"] == jid for e in events)


def test_set_priority_running_409(client):
    jid = _queue_job(client)
    main_module._jobs[jid]["state"] = "executing"
    main_module._SCHEDULER.next_batch()  # 模拟已派发（出队）
    r = client.put(f"/api/tasks/{jid}/priority", json={"priority": 5})
    assert r.status_code == 409


def test_set_priority_not_found_and_bad_id(client):
    r = client.put("/api/tasks/00000000_000000_000000/priority", json={"priority": 1})
    assert r.status_code == 404
    r = client.put("/api/tasks/bad_id/priority", json={"priority": 1})
    assert r.status_code == 400


# ---- 排队取消（DELETE /api/history/{id} 独立分支）----

def test_delete_queued_job(client):
    jid = _queue_job(client)
    r = client.delete(f"/api/history/{jid}")
    assert r.status_code == 200
    assert main_module._jobs.get(jid) is None
    assert main_module._SCHEDULER.queue_view() == []
    assert storage.get_job_files(jid) is None  # 目录已删
    events = audit.read_events()
    assert any(e["event"] == "eval_cancelled" and e["job_id"] == jid for e in events)


def test_delete_queued_job_not_found(client):
    r = client.delete("/api/history/00000000_000000_000000")
    assert r.status_code == 404


# ---- lifespan 重启沉降 ----

def test_lifespan_settles_queued_jobs(client):
    """磁盘态 queued（仅 config.json）任务在启动时沉降 error（不自动恢复）。"""
    jid = storage.create_job_id()
    storage.save_config(jid, {"model_a": {"name": "A"}, "model_b": {"name": "B"}})
    assert storage._job_state(storage.BASE_DIR / jid) == "queued"
    main_module._settle_queued_on_restart()
    st = storage.get_job_status(jid)
    assert st["state"] == "error"
    with open(storage.BASE_DIR / jid / "error.json", encoding="utf-8") as f:
        assert "重启" in json.load(f)["error"]


def test_lifespan_skips_non_queued(client):
    jid = storage.create_job_id()
    storage.save_config(jid, {"model_a": {"name": "A"}, "model_b": {"name": "B"}})
    storage.save_task_set(jid, {"meta": {"total": 1}, "tasks": []})
    main_module._settle_queued_on_restart()
    st = storage.get_job_status(jid)
    assert st["state"] == "executing"  # 运行中任务不受沉降影响
