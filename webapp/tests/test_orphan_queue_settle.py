# -*- coding: utf-8 -*-
"""迭代十二：重启孤儿排队任务沉降（_settle_orphan_queued）。

覆盖：批次下某 job 磁盘态 queued 且不在内存（_jobs/调度器）时，
_batch 列表/详情查询自动沉降 error → 批次可收尾（partial/done），
不再永久 running 阻塞排行榜生成。
"""
from __future__ import annotations

import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

from backend import main as main_module
from backend import storage

PUBLIC_URL = "https://8.8.8.8/v1"


class _DummyTask:
    def __init__(self, coro):
        self.coro = coro

    def add_done_callback(self, fn):
        pass

    def done(self):
        return True


@pytest.fixture()
def client(tmp_path):
    """隔离存储 + 真实第三方库导出：仅替换 asyncio.create_task 为 _DummyTask。"""
    monkeypatch = pytest.MonkeyPatch()
    import backend.engine.battle as battle_mod
    import backend.storage as storage_mod

    orig_base = storage_mod.BASE_DIR
    orig_batches = storage_mod.BATCHES_DIR
    storage_mod.BASE_DIR = tmp_path / "hist"
    storage_mod.BATCHES_DIR = tmp_path / "batches"
    for jid in list(main_module._jobs):
        main_module._jobs.pop(jid)
    for jid in list(main_module._tasks):
        main_module._tasks.pop(jid)
    main_module._SCHEDULER.clear()
    created = []

    def _fake_create(coro):
        t = _DummyTask(coro)
        created.append(t)
        return t

    monkeypatch.setattr(main_module.asyncio, "create_task", _fake_create)
    monkeypatch.setattr(main_module, "build_task_set", lambda **k: {
        "meta": {"total": 1}, "tasks": [
            {"id": "T1", "type": "判别式", "dimension": "知识能力",
             "prompt": "p", "test_cases": [{"input": "?", "expected": "1"}]}]})
    monkeypatch.setattr(main_module, "build_task_set_from_dataset", lambda **k: {
        "meta": {"total": 1}, "tasks": [
            {"id": "T1", "type": "判别式", "dimension": "知识能力",
             "prompt": "p", "test_cases": [{"input": "?", "expected": "1"}]}]})
    client = TestClient(main_module.app)
    yield client
    storage_mod.BASE_DIR = orig_base
    storage_mod.BATCHES_DIR = orig_batches
    main_module._SCHEDULER.clear()
    monkeypatch.undo()


def _submit_batch(client, name="b1"):
    r = client.post("/api/benchmark", json={
        "name": name,
        "dataset_name": None,
        "models": [
            {"name": "模型甲", "url": "https://8.8.8.8/v1", "key": "k"},
            {"name": "模型乙", "url": "https://8.8.8.8/v1", "key": "k"},
        ],
        "dims": ["知识能力"],
        "rounds": 1,
        "code_verify_mode": "off",
    })
    assert r.status_code == 200, r.text
    return r.json()["batch_id"]


def test_orphan_queued_settled_on_batch_list(client):
    """批次 job 在磁盘仅 config.json（queued）且不在内存 → 列表查询沉降 error，
    批次自动收尾（不再永久 running）。"""
    batch_id = _submit_batch(client)
    batch = storage.load_batch(batch_id)
    jobs = batch["jobs"]
    assert len(jobs) == 2

# 模拟重启孤儿：清空内存并删除已执行 job 的所有产物，仅保留一个 config.json
    for jid in list(main_module._jobs):
        main_module._jobs.pop(jid, None)
    main_module._SCHEDULER.clear()
    for jid in list(main_module._tasks):
        main_module._tasks.pop(jid, None)
    # 让第一个 job 变为已完成（单臂产物齐全），第二个 job 保持孤儿（磁盘态 queued）
    storage.save_answers(jobs[0], "a", {"model": "模型甲", "answers": [
        {"id": "T1", "raw_answer": "1", "api_info": {"status": "ok",
         "latency_ms": 1, "prompt_tokens": 1, "completion_tokens": 1}}]})
    storage.save_verdict(jobs[0], {
        "scores": [{"id": "T1", "score": 8.0}],
        "meta": {"mode": "single_arm"},
    })
    storage.save_report(jobs[0], {"report": {
        "summary": {"mode": "single_arm", "n_tasks": 1},
        "kpi": {"total_tokens": {"x": 2}}}})
    # 第二个 job 保持孤儿（磁盘态 queued）

    # 列表接口自愈：先沉降孤儿 → 批次终态（partial，仅 1 个完成模型）
    r = client.get("/api/tasks")
    assert r.status_code == 200
    batch_out = next(x for x in r.json()["batches"]
                     if x["batch_id"] == batch_id)
    assert batch_out["state"] in ("done", "partial")

    orphan_error_file = storage._job_path(jobs[1]) / "error.json"
    assert orphan_error_file.exists()
    assert "进程重启" in json.loads(orphan_error_file.read_text(encoding="utf-8"))["error"]


def test_orphan_queued_not_settled_when_in_memory(client):
    """排队中（内存持有）任务不应被沉降——_jobs/调度器存在时保持 queued。"""
    batch_id = _submit_batch(client)
    batch = storage.load_batch(batch_id)
    jid = batch["jobs"][0]
    assert jid in main_module._jobs or main_module._SCHEDULER.is_queued(jid)
    main_module._settle_orphan_queued(jid)
    st = storage.get_job_files(jid)
    assert "error.json" not in st