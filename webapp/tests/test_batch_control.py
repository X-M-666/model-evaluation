# -*- coding: utf-8 -*-
"""迭代八：benchmark 批次整体取消。

- cancel：排队中 job 出队+删目录（无 task.cancel）、运行中 job 走 cancelling
  （真实 asyncio 任务）；终态 409、不存在/非法格式 404；审计 benchmark_cancelled
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
                        persist_cb=None, concurrency=1):
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

    def done(self):
        return True

    def cancel(self):
        pass


@pytest.fixture
def client():
    return TestClient(main_module.app)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    main_module._jobs.clear()
    main_module._tasks.clear()
    main_module._SCHEDULER.clear()
    monkeypatch.setattr(main_module, "_execute_model", _fake_execute)
    monkeypatch.setattr(main_module, "run_single_arm_judge", _fake_single_arm)
    monkeypatch.setattr(main_module.asyncio, "create_task", _DummyTask)
    audit._log_path().write_text("", encoding="utf-8")
    storage.save_dataset("批次控制集", _dataset())
    yield
    main_module._jobs.clear()
    main_module._tasks.clear()
    main_module._SCHEDULER.clear()


def _payload(n=3, **overrides) -> dict:
    base = {"dataset_name": "批次控制集", "models": [
        {"url": PUBLIC_URL, "key": "k", "name": f"控制模型{i + 1}",
         "temperature": 0.7, "max_tokens": 4096} for i in range(n)]}
    base.update(overrides)
    return base


def _start(n=3, **overrides) -> dict:
    return _call(main_module.create_benchmark, BenchmarkRequest(**_payload(n, **overrides)))


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
    res = _start(n=3)
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

    async def _scenario():
        req = BenchmarkRequest(**_payload(2))
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
