# -*- coding: utf-8 -*-
"""任务生命周期测试（issue #14 / R2-005）。

保护不变量：
- 删除运行中评测必须取消并等待后台任务安全终止，目录/文件不复活
- 后台任务被正常回收，不产生 KeyError / 未获取 Task 异常
- 协作取消路径（未调用 cancel）也不得在检查点后落盘
- 重复删除 404；取消/删除后的评审提交被拒绝；关闭时统一回收

说明：TestClient 每次请求使用独立 portal 事件循环，请求内 create_task 的
后台协程不会跨请求调度（见 test_ssrf.py:342），因此"运行中删除"用例在
单一 asyncio.run 循环内直接调用与生产完全相同的端点函数
（start_eval / delete_history）驱动；其余无后台任务的状态语义走 HTTP 层。
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from backend import audit
from backend import main as main_module
from backend import storage
from backend.schemas import StartRequest

PUBLIC_URL = "https://8.8.8.8/v1"


def _payload() -> dict:
    return {
        "model_a": {"url": PUBLIC_URL, "key": "k", "name": "A", "temperature": 0.7, "max_tokens": 100},
        "model_b": {"url": PUBLIC_URL, "key": "k", "name": "B", "temperature": 0.7, "max_tokens": 100},
    }


@pytest.fixture
def client():
    return TestClient(main_module.app)


@pytest.fixture(autouse=True)
def _clean_state():
    main_module._jobs.clear()
    main_module._tasks.clear()
    yield
    main_module._jobs.clear()
    main_module._tasks.clear()


# ---- 删除运行中任务：取消 + 等待 + 无复活 ----

def test_delete_running_job_cancels_task_and_no_resurrection(monkeypatch):
    async def _slow_execute_all(*args, **kwargs):
        await asyncio.sleep(3600)

    monkeypatch.setattr(main_module, "execute_all", _slow_execute_all)

    async def _scenario():
        main_module._jobs.clear()
        main_module._tasks.clear()
        resp = await main_module.start_eval(StartRequest(**_payload()))
        job_id = resp.job_id

        task = main_module._tasks.get(job_id)
        assert task is not None and not task.done()
        await asyncio.sleep(0.05)  # 让后台任务进入执行态

        r = await main_module.delete_history(job_id)
        assert r == {"ok": True}

        # 目录与内存状态彻底移除，任务引用回收
        assert storage.get_job_files(job_id) is None
        assert job_id not in main_module._jobs
        assert job_id not in main_module._tasks
        assert task.done()

        # 继续调度：目录不复活、无新文件
        await asyncio.sleep(0.2)
        assert storage.get_job_files(job_id) is None

        # 审计：取消 + 删除均记录
        events = {e["event"]: e for e in audit.read_events()}
        assert events["eval_cancelled"]["job_id"] == job_id
        assert events["history_deleted"]["job_id"] == job_id

    asyncio.run(_scenario())


def test_delete_running_job_after_grace_timeout_still_clean(monkeypatch):
    """任务吞掉首次取消（极端阻塞）超过兜底窗口时，删除照常完成且目录不复活。"""
    async def _stubborn_execute_all(*args, **kwargs):
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await asyncio.sleep(0.5)  # 拖延超过兜底窗口

    monkeypatch.setattr(main_module, "execute_all", _stubborn_execute_all)
    monkeypatch.setattr(main_module, "CANCEL_GRACE_SECONDS", 0.1)

    async def _scenario():
        main_module._jobs.clear()
        main_module._tasks.clear()
        resp = await main_module.start_eval(StartRequest(**_payload()))
        job_id = resp.job_id
        task = main_module._tasks[job_id]
        await asyncio.sleep(0.05)

        r = await main_module.delete_history(job_id)
        assert r == {"ok": True}
        assert storage.get_job_files(job_id) is None
        assert job_id not in main_module._jobs

        # 等待极端任务最终停止：job 已缺失，协作检查点兜底，不写任何文件
        await asyncio.sleep(1.0)
        assert task.done()
        assert storage.get_job_files(job_id) is None

    asyncio.run(_scenario())


# ---- 协作取消：不依赖 task.cancel 也不得落盘 ----

def test_cooperative_cancel_stops_before_persistence(monkeypatch):
    """execute_all 运行中状态被置为 cancelling（模拟并发删除的早期窗口）：
    返回后 _run_eval 复查取消标记，不写任何 answers 文件。"""
    async def _self_cancel_execute_all(task_set, config_a=None, config_b=None, progress_cb=None):
        for j in main_module._jobs.values():
            j["state"] = "cancelling"
        return {"answers": []}, {"answers": []}

    monkeypatch.setattr(main_module, "execute_all", _self_cancel_execute_all)

    async def _scenario():
        main_module._jobs.clear()
        main_module._tasks.clear()
        resp = await main_module.start_eval(StartRequest(**_payload()))
        job_id = resp.job_id
        await _wait_task_gone(job_id)

        # 任务已回收，job 仍存在（未被删除），状态为 cancelled
        assert job_id not in main_module._tasks
        assert main_module._jobs[job_id]["state"] == "cancelled"
        files = storage.get_job_files(job_id)
        assert files is not None
        assert not any("answers" in k or "reveal" in k for k in files)

        # 清理
        r = await main_module.delete_history(job_id)
        assert r == {"ok": True}

    asyncio.run(_scenario())


# ---- 删除语义：重复删除 / 评审提交（HTTP 层） ----

def test_delete_missing_job_404(client):
    r = client.delete("/api/history/不存在的任务")
    assert r.status_code == 404


def test_delete_twice_second_404(client):
    r = client.post("/api/eval/mock")
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    assert client.delete(f"/api/history/{job_id}").status_code == 200
    assert client.delete(f"/api/history/{job_id}").status_code == 404


def test_review_submit_after_delete_404(client):
    r = client.post("/api/eval/mock")
    job_id = r.json()["job_id"]
    assert client.delete(f"/api/history/{job_id}").status_code == 200
    r = client.post(f"/api/eval/{job_id}/review", json={"scores": []})
    assert r.status_code == 404


def test_review_submit_while_cancelling_409(client):
    r = client.post("/api/eval/mock")
    job_id = r.json()["job_id"]
    task_set = main_module._jobs[job_id]["task_set"]
    scores = [{"id": t["id"], "round": 1, "answer_x": 5, "answer_y": 5} for t in task_set["tasks"]]

    # 模拟删除流程已进入 cancelling
    main_module._jobs[job_id]["state"] = "cancelling"
    r = client.post(f"/api/eval/{job_id}/review", json={"scores": scores})
    assert r.status_code == 409


def test_delete_reviewing_job_no_task_ok(client):
    """无后台任务的 reviewing 任务（mock）直接删除。"""
    r = client.post("/api/eval/mock")
    job_id = r.json()["job_id"]
    assert job_id not in main_module._tasks
    r = client.delete(f"/api/history/{job_id}")
    assert r.status_code == 200
    assert storage.get_job_files(job_id) is None


# ---- 服务关闭回收 ----

def test_shutdown_cancels_all_tasks(monkeypatch):
    async def _slow_execute_all(*args, **kwargs):
        await asyncio.sleep(3600)

    monkeypatch.setattr(main_module, "execute_all", _slow_execute_all)

    async def _scenario():
        job_id = "shutdown-test"
        main_module._jobs.clear()
        main_module._tasks.clear()
        main_module._jobs[job_id] = {
            "state": "executing",
            "progress": "0/1",
            "task_set": {"meta": {"total": 1}, "tasks": []},
            "config": {"model_a": {"url": "mock://a", "name": "A"}, "model_b": {"url": "mock://b", "name": "B"}},
            "answers_a": None, "answers_b": None, "verdict": None,
            "rounds_answers": [], "reveal": None,
            "created_at": "x", "sse_queue": asyncio.Queue(), "repeat_n": 1,
        }
        task = asyncio.create_task(main_module._run_eval(job_id))
        main_module._tasks[job_id] = task
        await asyncio.sleep(0.05)
        assert not task.done()

        await main_module._shutdown_cancel_all()
        assert job_id not in main_module._tasks
        assert task.done()
        assert main_module._jobs[job_id]["state"] == "cancelled"
        assert not any("answers" in k for k in storage.get_job_files(job_id) or {})

    asyncio.run(_scenario())


async def _wait_task_gone(job_id: str, timeout: float = 5.0) -> None:
    """异步轮询等待后台任务从 _tasks 注销（让出事件循环，任务才能推进）。"""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while job_id in main_module._tasks and loop.time() < deadline:
        await asyncio.sleep(0.02)
