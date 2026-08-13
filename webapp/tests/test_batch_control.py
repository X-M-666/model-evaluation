# -*- coding: utf-8 -*-
"""迭代八：benchmark 批次整体取消与重跑。

- cancel：排队中 job 出队+删目录（无 task.cancel）、运行中 job 走 cancelling
  （真实 asyncio 任务）；终态 409、不存在/非法格式 404；审计 benchmark_cancelled
- rerun：复用 batch 文件（model_ids/rounds/dataset/name）+ job config 重建；
  运行中 409、缺 model_ids 400、原批次有 judge 必须重供评审配置、模型 Key
  从配置库重取（未补录 400）；审计 benchmark_rerun
"""
from __future__ import annotations

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from backend import audit
from backend import main as main_module
from backend import storage
from backend.schemas import BenchmarkRequest

PUBLIC_URL = "https://8.8.8.8/v1"

# 模块导入时捕获真实 create_task（测试 fixture 会全局打补丁为 _DummyTask）
_REAL_CREATE_TASK = asyncio.create_task


def _dataset() -> dict:
    return {
        "name": "批次控制集",
        "tasks": [
            {"id": "T1", "type": "判别式", "dimension": "数学能力",
             "prompt": "1+1=?", "test_cases": [{"input": "1+1=?", "expected": "2"}]},
            {"id": "T2", "type": "判别式", "dimension": "数学能力",
             "prompt": "2+2=?", "test_cases": [{"input": "2+2=?", "expected": "4"}]},
            {"id": "T3", "type": "判别式", "dimension": "逻辑推理能力",
             "prompt": "3+3=?", "test_cases": [{"input": "3+3=?", "expected": "6"}]},
        ],
    }


async def _fake_execute(model_label, config, tasks, stability_repeat,
                        progress_cb=None, embedding_cfg=None, skip_ids=None,
                        persist_cb=None):
    answers = []
    for t in tasks:
        if skip_ids and t["id"] in skip_ids:
            continue
        answers.append({
            "id": t["id"],
            "raw_answer": "答案是 2" if t.get("type") == "判别式" else "生成文本",
            "api_info": {"status": "ok", "attempts": 1, "truncated": False,
                         "error": None, "latency_ms": 100, "prompt_tokens": 50,
                         "completion_tokens": 20, "repeat_index": 1},
        })
    return {"model": config["name"], "api": {"name": config["name"]},
            "answers": answers}


async def _fake_single_arm(task_set, answers, judge_config,
                           progress_cb=None, max_retries=1):
    scores = [{"id": t["id"], "dimension": t.get("dimension", ""),
               "score": 8.0, "basis": "fake", "_invalid": False}
              for t in task_set["tasks"]]
    return {"meta": {"total": len(scores), "valid": len(scores), "invalid": 0},
            "scores": scores, "totals": {}, "health": {"healthy": True}}


def _call(fn, *a, **k):
    return asyncio.run(fn(*a, **k))


class _DummyTask:
    def __init__(self, coro):
        self.coro = coro

    def add_done_callback(self, fn):
        pass


def _drain_batch(batch_id: str) -> None:
    """直跑批次内全部 job 的后台协程（测试专用，幂等）。"""
    batch = storage.load_batch(batch_id)
    for jid in batch["jobs"]:
        j = main_module._jobs.get(jid)
        if j is None or j.get("state") in ("completed", "error", "cancelled"):
            continue
        main_module._tasks.pop(jid, None)
        asyncio.run(main_module._run_batch_job(jid))


@pytest.fixture
def client():
    return TestClient(main_module.app)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    from backend import models_registry
    main_module._jobs.clear()
    main_module._tasks.clear()
    main_module._SCHEDULER.clear()
    monkeypatch.setattr(main_module, "_execute_model", _fake_execute)
    monkeypatch.setattr(main_module, "run_single_arm_judge", _fake_single_arm)
    monkeypatch.setattr(main_module.asyncio, "create_task", _DummyTask)
    audit._log_path().write_text("", encoding="utf-8")
    models_registry.clear_memory_keys()
    for p in models_registry.MODELS_DIR.glob("*.json"):
        p.unlink(missing_ok=True)
    storage.save_dataset("批次控制集", _dataset())
    yield
    main_module._jobs.clear()
    main_module._tasks.clear()
    main_module._SCHEDULER.clear()
    models_registry.clear_memory_keys()
    for p in models_registry.MODELS_DIR.glob("*.json"):
        p.unlink(missing_ok=True)


def _register(client, name, key="k", url=PUBLIC_URL) -> str:
    r = client.post("/api/models", json={"name": name, "url": url, "key": key})
    assert r.status_code == 200, r.text
    return r.json()["model"]["id"]


def _start(client, n=3, **overrides) -> dict:
    ids = [_register(client, f"控制模型{i + 1}") for i in range(n)]
    base = {"dataset_name": "批次控制集", "model_ids": ids, "rounds": 1}
    base.update(overrides)
    return _call(main_module.create_benchmark, BenchmarkRequest(**base))


# ---- cancel ----

def test_cancel_queued_batch(client, monkeypatch):
    from backend.scheduler import Scheduler
    # 占满配额：批内 job 全部停留排队、不派发（无后台任务，走排队取消分支）
    sched = Scheduler(concurrency=1)
    monkeypatch.setattr(main_module, "_SCHEDULER", sched)
    filler = "20250901_000000_fill1"
    sched.submit(filler)
    sched.next_batch()
    main_module._jobs[filler] = {"state": "executing", "config": {}}
    res = _start(client, n=3)
    try:
        assert storage.load_batch(res["batch_id"])["state"] == "running"
        for jid in res["jobs"]:
            assert jid in main_module._jobs and main_module._jobs[jid]["state"] == "queued"
            assert jid not in main_module._tasks   # 未派发 → 无后台任务

        r = client.post(f"/api/benchmark/{res['batch_id']}/cancel")
        assert r.status_code == 200, r.text
        assert r.json()["state"] == "cancelled"

        batch = storage.load_batch(res["batch_id"])
        assert batch["state"] == "cancelled"
        assert batch["finished_at"]
        for jid in res["jobs"]:
            assert jid not in main_module._jobs       # 排队中 job 已出队
            assert not storage._job_path(jid).exists()  # 目录已删
        assert main_module._SCHEDULER.queue_view() == []
        events = audit.read_events()
        assert any(e["event"] == "benchmark_cancelled"
                   and e["target"] == res["batch_id"] for e in events)
        assert sum(1 for e in events if e["event"] == "eval_cancelled") == 3
        # 终态再取消 → 409
        r2 = client.post(f"/api/benchmark/{res['batch_id']}/cancel")
        assert r2.status_code == 409
    finally:
        main_module._jobs.pop(filler, None)


def test_cancel_missing_and_invalid(client):
    assert client.post("/api/benchmark/batch_20990101_000000_000000/cancel").status_code == 404
    assert client.post("/api/benchmark/..%2F..%2Fetc/cancel").status_code == 404


def test_cancel_running_batch_real_tasks(client, monkeypatch):
    """运行中 job 走 cancelling（真实 asyncio 任务，同事件循环内创建与取消）。

    TestClient 每次请求独立 portal 循环会终止跨请求的后台任务，故本场景
    在单次 asyncio.run 内完成创建→执行→取消→回收，模拟真实 uvicorn 常驻循环。
    """
    async def _slow_execute(model_label, config, tasks, stability_repeat,
                            progress_cb=None, **kw):
        await asyncio.sleep(0.5)
        return {"model": config["name"], "answers": []}
    monkeypatch.setattr(main_module, "_execute_model", _slow_execute)
    monkeypatch.setattr(main_module.asyncio, "create_task", _REAL_CREATE_TASK)

    ids = [_register(client, f"运行模型{i + 1}") for i in range(2)]

    async def _scenario():
        req = BenchmarkRequest(dataset_name="批次控制集", model_ids=ids, rounds=1)
        res = await main_module.create_benchmark(req)
        bid = res["batch_id"]
        await asyncio.sleep(0.2)                       # 任务已派发进入执行
        assert all(jid in main_module._tasks for jid in res["jobs"]), "后台任务未存活"
        rc = await main_module.benchmark_cancel(bid)
        assert rc["state"] == "cancelled"
        await asyncio.sleep(0.3)                       # 取消回收完成
        assert not any(jid in main_module._jobs for jid in res["jobs"])
        assert not any(jid in main_module._tasks for jid in res["jobs"])
        return bid

    bid = asyncio.run(_scenario())
    assert storage.load_batch(bid)["state"] == "cancelled"
    events = audit.read_events()
    assert any(e["event"] == "benchmark_cancelled" and e["target"] == bid for e in events)
    assert sum(1 for e in events if e["event"] == "eval_cancelled") == 2


# ---- rerun ----

def test_rerun_done_batch(client):
    res = _start(client, n=3)
    bid = res["batch_id"]
    _drain_batch(bid)
    batch = storage.load_batch(bid)
    assert batch["state"] == "done"

    r = client.post(f"/api/benchmark/{bid}/rerun", json={})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["from_batch"] == bid
    assert d["batch_id"] != bid
    assert len(d["jobs"]) == 3
    assert d["models"] == batch["models"]
    new_batch = storage.load_batch(d["batch_id"])
    assert new_batch["model_ids"] == batch["model_ids"]
    assert new_batch["rounds"] == batch["rounds"]
    assert new_batch["dataset"] == batch["dataset"]
    assert storage.load_batch(d["batch_id"])["state"] == "running"
    events = audit.read_events()
    assert any(e["event"] == "benchmark_rerun" and e["target"] == bid
               and e["path"] == d["batch_id"] for e in events)


def test_rerun_running_409(client):
    res = _start(client, n=2)
    r = client.post(f"/api/benchmark/{res['batch_id']}/rerun", json={})
    assert r.status_code == 409


def test_rerun_missing_404(client):
    r = client.post("/api/benchmark/batch_20990101_000000_000000/rerun", json={})
    assert r.status_code == 404


def test_rerun_old_batch_without_model_ids_400(client):
    res = _start(client, n=2)
    bid = res["batch_id"]
    _drain_batch(bid)
    batch = storage.load_batch(bid)
    batch.pop("model_ids", None)          # 模拟迭代七创建的历史批次
    storage.save_batch(bid, batch)
    r = client.post(f"/api/benchmark/{bid}/rerun", json={})
    assert r.status_code == 400
    assert "model_ids" in r.json()["detail"]


def test_rerun_requires_judge_reprovision(client):
    res = _start(client, n=2, review={"mode": "pure_agent", "judge": {
        "url": PUBLIC_URL, "name": "J", "key": "jk", "temperature": 0.0,
        "max_tokens": 256}})
    bid = res["batch_id"]
    _drain_batch(bid)
    r = client.post(f"/api/benchmark/{bid}/rerun", json={})
    assert r.status_code == 400
    assert "评审" in r.json()["detail"]
    r2 = client.post(f"/api/benchmark/{bid}/rerun", json={
        "review": {"mode": "pure_agent", "judge": {
            "url": PUBLIC_URL, "name": "J2", "key": "jk2", "temperature": 0.0,
            "max_tokens": 256}}})
    assert r2.status_code == 200, r2.text


def test_rerun_model_key_missing_400(client):
    from backend import models_registry
    res = _start(client, n=2)
    bid = res["batch_id"]
    _drain_batch(bid)
    models_registry.clear_memory_keys()   # 模拟重启后 Key 未补录
    r = client.post(f"/api/benchmark/{bid}/rerun", json={})
    assert r.status_code == 400
    assert "Key" in r.json()["detail"]


def test_rerun_dataset_deleted_404(client):
    res = _start(client, n=2)
    bid = res["batch_id"]
    _drain_batch(bid)
    storage.delete_dataset("批次控制集")
    r = client.post(f"/api/benchmark/{bid}/rerun", json={})
    assert r.status_code == 404
