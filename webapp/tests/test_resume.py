# -*- coding: utf-8 -*-
"""迭代七：断点续跑（executor skip_ids/persist_cb + 增量落盘 + resume API）。

覆盖：
- _execute_model 的 persist_cb 逐题回调（含跳过题不回调）、skip_ids 跳过执行
- 增量落盘幂等 + resume 合并路径（磁盘答案并回答案池）
- resume API 语义：运行中 409 / 无部分答案 409 / 不存在 404 /
  磁盘态部分完成任务重新入队续跑（跳过已完成题）
"""
import asyncio

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend import audit
from backend import main as main_module
from backend import storage
from backend.engine.executor import _execute_model
from backend.schemas import StartRequest

PUBLIC_URL = "https://8.8.8.8/v1"

CONFIG = {"url": PUBLIC_URL, "key": "k", "name": "模型A",
          "temperature": 0.7, "max_tokens": 100, "top_p": None,
          "code_verify_mode": "off", "prompt_strategy": "cot"}

TASKS = [{"id": "T1", "dimension": "知识能力", "type": "判别式", "prompt": "p1",
          "test_cases": [{"input": "i", "expected": "e"}]},
         {"id": "T2", "dimension": "知识能力", "type": "判别式", "prompt": "p2",
          "test_cases": [{"input": "i", "expected": "e"}]},
         {"id": "T3", "dimension": "知识能力", "type": "判别式", "prompt": "p3",
          "test_cases": [{"input": "i", "expected": "e"}]}]


def _ok_handler(request):
    return httpx.Response(200, json={
        "choices": [{"message": {"content": "答案"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
    })


@pytest.fixture
def client():
    return TestClient(main_module.app)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    main_module._jobs.clear()
    main_module._tasks.clear()
    main_module._SCHEDULER.clear()
    audit._log_path().write_text("", encoding="utf-8")
    yield
    main_module._jobs.clear()
    main_module._tasks.clear()
    main_module._SCHEDULER.clear()


def _call(fn, *a, **k):
    return asyncio.run(fn(*a, **k))


# ---- executor：persist_cb / skip_ids ----

async def _run_executor(**kw):
    import backend.engine.executor as ex

    class _Ctx:
        def __init__(self, c):
            self._c = c

        async def __aenter__(self):
            return self._c

        async def __aexit__(self, *a):
            await self._c.aclose()

    transport = httpx.MockTransport(_ok_handler)
    client = httpx.AsyncClient(transport=transport)
    orig = ex.build_upstream_client
    ex.build_upstream_client = lambda: _Ctx(client)
    try:
        return await _execute_model("A", CONFIG, TASKS, None, **kw)
    finally:
        ex.build_upstream_client = orig
        await client.aclose()


def test_persist_cb_called_per_task(monkeypatch):
    calls = []

    def persist(label, tid, entry):
        calls.append((label, tid))

    result = _call(_run_executor, persist_cb=persist)
    assert [t for _, t in calls] == ["T1", "T2", "T3"]
    assert len(result["answers"]) == 3
    assert all(e["id"] in ("T1", "T2", "T3") for e in result["answers"])


def test_skip_ids_skips_execution(monkeypatch):
    calls = []

    def persist(label, tid, entry):
        calls.append(tid)

    result = _call(_run_executor, skip_ids={"T2"}, persist_cb=persist)
    assert calls == ["T1", "T3"]
    assert [e["id"] for e in result["answers"]] == ["T1", "T3"]


def test_skip_all_skips_nothing_calls(monkeypatch):
    calls = []
    result = _call(_run_executor, skip_ids={"T1", "T2", "T3"},
                   persist_cb=lambda l, t, e: calls.append(t))
    assert calls == []
    assert result["answers"] == []


# ---- 增量落盘与合并（resume 数据路径）----

def test_inc_persist_and_merge():
    jid = storage.create_job_id()
    storage.save_config(jid, {"model_a": CONFIG, "model_b": CONFIG})
    storage.save_answers_inc(jid, "a:T1", {"id": "T1", "raw_answer": "已有"})
    storage.save_answers_inc(jid, "b:T1", {"id": "T1", "raw_answer": "已有"})
    storage.save_answers_inc(jid, "a:T2", {"id": "T2", "raw_answer": "已有"})
    inc = storage.load_answers_inc(jid)
    assert set(inc) == {"a:T1", "b:T1", "a:T2"}
    assert storage.partial_answers_count(jid) == 2  # 按题去重
    # 合并：磁盘答案并回答案池（resume 路径语义，按侧补齐）
    answers_a = {"model": "m", "answers": []}
    answers_a["answers"] += [inc[k] for k in ("a:T1", "a:T2") if k in inc]
    assert [e["id"] for e in answers_a["answers"]] == ["T1", "T2"]


# ---- resume API ----

def _make_disk_job(partial=True) -> str:
    jid = storage.create_job_id()
    storage.save_config(jid, {"model_a": CONFIG, "model_b": CONFIG,
                              "dataset_name": None, "repeat_n": 1})
    storage.save_task_set(jid, {"meta": {"total": 3}, "tasks": TASKS})
    if partial:
        storage.save_answers_inc(jid, "a:T1", {"id": "T1", "raw_answer": "已有"})
        storage.save_answers_inc(jid, "b:T1", {"id": "T1", "raw_answer": "已有"})
    return jid


def test_resume_running_job_409(client):
    jid = _make_disk_job()
    main_module._jobs[jid] = {"state": "executing", "config": {"model_a": CONFIG}}
    with pytest.raises(HTTPException) as ei:
        _call(main_module.resume_eval, jid)
    assert ei.value.status_code == 409


def test_resume_no_partial_409(client):
    jid = _make_disk_job(partial=False)
    with pytest.raises(HTTPException) as ei:
        _call(main_module.resume_eval, jid)
    assert ei.value.status_code == 409


def test_resume_not_found_404(client):
    with pytest.raises(HTTPException) as ei:
        _call(main_module.resume_eval, "00000000_000000_000000")
    assert ei.value.status_code == 404


def test_resume_requeues_and_skips_done(client, monkeypatch):
    """磁盘态部分完成（T1 双侧已有）→ resume → 入队 → 续跑跳过 T1。"""
    jid = _make_disk_job(partial=True)
    assert storage._job_state(storage.BASE_DIR / jid) == "executing"
    # 桩化执行：验证 skip_ids 传递与增量合并（续跑只执行 T2/T3）
    seen = {}

    async def fake_execute(task_set, config_a=None, config_b=None, progress_cb=None,
                           **kwargs):
        skip = kwargs.get("skip_ids") or set()
        seen["skip"] = skip
        seen["persist"] = kwargs.get("persist_cb") is not None

        def mk(label):
            return {"model": config_a["name"], "answers": [
                {"id": t["id"], "raw_answer": "新答",
                 "api_info": {"status": "ok", "attempts": 1, "truncated": False,
                              "error": None, "latency_ms": 1,
                              "prompt_tokens": 1, "completion_tokens": 1,
                              "repeat_index": 1}}
                for t in task_set["tasks"] if t["id"] not in (skip or set())]}

        return mk("a"), mk("b")

    monkeypatch.setattr(main_module, "execute_all", fake_execute)
    resp = _call(main_module.resume_eval, jid)
    assert resp["job_id"] == jid
    assert seen["skip"] == {"T1"}
    assert seen["persist"] is True
    # 审计
    events = audit.read_events()
    assert any(e["event"] == "eval_resumed" and e["target"] == jid for e in events)
    main_module._jobs.clear()
    main_module._tasks.clear()
    main_module._SCHEDULER.clear()


def _payload() -> dict:
    return {"model_a": {"url": PUBLIC_URL, "key": "k", "name": "A",
                        "temperature": 0.7, "max_tokens": 100},
            "model_b": {"url": PUBLIC_URL, "key": "k", "name": "B",
                        "temperature": 0.7, "max_tokens": 100}}
