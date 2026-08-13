# -*- coding: utf-8 -*-
"""迭代七：benchmark 批次 API 集成测试。

覆盖不变量：
- 创建校验：数据集 404、模型配置不存在/重复/未补录 Key 400、评审模型 SSRF、
  预算 hard（N 模型放大）400
- N=5 全链路（mock 单臂）→ done + 排行榜聚合（综合分/分维度/胜率矩阵/CI）
- 部分失败 → partial + failed_models N/A + 排行榜排除失败模型
- 详情/列表/排行榜端点；审计 benchmark_started/benchmark_done
- Key 不落盘（config 仅掩码字段）

使用同款模式：monkeypatch _execute_model / run_single_arm_judge + 假 create_task
（后台任务在 asyncio.run 收尾被取消），再直跑 _run_batch_job 完成管线。
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend import audit
from backend import main as main_module
from backend import storage
from backend.schemas import BenchmarkRequest

PUBLIC_URL = "https://8.8.8.8/v1"

MODEL_NAMES = ["模型1", "模型2", "模型3", "模型4", "模型5"]


def _dataset() -> dict:
    return {
        "name": "批次集A",
        "tasks": [
            {"id": "T1", "type": "判别式", "dimension": "数学能力",
             "prompt": "1+1=?", "test_cases": [{"input": "1+1=?", "expected": "2"}]},
            {"id": "T2", "type": "判别式", "dimension": "数学能力",
             "prompt": "2+2=?", "test_cases": [{"input": "2+2=?", "expected": "4"}]},
            {"id": "T3", "type": "生成式", "dimension": "语言能力",
             "prompt": "写一句话", "expected": "参考答案",
             "rubric_note": "满分10分"},
        ],
    }


async def _fake_execute(model_label, config, tasks, stability_repeat,
                        progress_cb=None, embedding_cfg=None, skip_ids=None,
                        persist_cb=None):
    if config.get("url", "").startswith("mock://fail"):
        raise RuntimeError("模型调用失败（测试）")
    answers = []
    for t in tasks:
        if skip_ids and t["id"] in skip_ids:
            continue
        cases = t.get("test_cases") or []
        exp = cases[0].get("expected", "") if cases else ""
        answers.append({
            "id": t["id"],
            "raw_answer": f"答案是 {exp}" if t.get("type") == "判别式" else "生成文本",
            "api_info": {"status": "ok", "attempts": 1, "truncated": False,
                         "error": None, "latency_ms": 100, "prompt_tokens": 50,
                         "completion_tokens": 20, "repeat_index": 1},
        })
        if progress_cb:
            await progress_cb(model_label, len(answers), len(tasks))
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
    # 模型配置库按用例隔离（conftest 为 module 级重定向，测试内需清空防重名）
    models_registry.clear_memory_keys()
    for p in models_registry.MODELS_DIR.glob("*.json"):
        p.unlink(missing_ok=True)
    storage.save_dataset("批次集A", _dataset())
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


def _payload(client, n=3, **overrides) -> dict:
    ids = [_register(client, f"模型{i + 1}") for i in range(n)]
    base = {"dataset_name": "批次集A", "model_ids": ids, "rounds": 1}
    base.update(overrides)
    return base


def _start(**kw):
    return _call(main_module.create_benchmark, BenchmarkRequest(**kw))


# ---- 创建校验 ----

def test_dataset_missing_404(client):
    with pytest.raises(HTTPException) as ei:
        _start(dataset_name="不存在集", model_ids=["m1", "m2"])
    assert ei.value.status_code == 404


def test_model_not_in_library_400(client):
    with pytest.raises(HTTPException) as ei:
        _start(**{"dataset_name": "批次集A", "model_ids": ["nope", "nope2"]})
    assert ei.value.status_code == 400


def test_model_without_key_400(client):
    _register(client, "无Key模型", key=None)
    with pytest.raises(HTTPException) as ei:
        _start(dataset_name="批次集A", model_ids=["无Key模型", "x2"])
    assert ei.value.status_code == 400
    assert "Key" in str(ei.value.detail)


def test_duplicate_model_ids_400(client):
    mid = _register(client, "重复模型")
    with pytest.raises(HTTPException) as ei:
        _start(dataset_name="批次集A", model_ids=[mid, mid])
    assert ei.value.status_code == 400


def test_judge_ssrf_400(client):
    payload = _payload(client, n=2)
    payload["review"] = {"mode": "pure_agent",
                         "judge": {"url": "http://127.0.0.1:9/v1", "key": "k",
                                   "name": "j", "temperature": 0.0,
                                   "max_tokens": 100}}
    with pytest.raises(HTTPException) as ei:
        _start(**payload)
    assert ei.value.status_code == 400


def test_budget_hard_400(client):
    payload = _payload(client, n=5)
    payload["budget"] = {"max_tokens": 100, "mode": "hard"}
    with pytest.raises(HTTPException) as ei:
        _start(**payload)
    assert ei.value.status_code == 400
    assert "预算超限" in str(ei.value.detail)


# ---- 全链路 ----

def test_full_pipeline_n5(client):
    payload = _payload(client, n=5, rounds=2,
                       review={"mode": "pure_agent",
                               "judge": {"url": PUBLIC_URL, "key": "jk",
                                         "name": "评审", "temperature": 0.0,
                                         "max_tokens": 100}})
    resp = _start(**payload)
    assert resp["ok"] is True
    batch_id = resp["batch_id"]
    assert len(resp["jobs"]) == 5
    assert resp["models"] == MODEL_NAMES
    events = audit.read_events()
    assert any(e["event"] == "benchmark_started" and e["target"] == batch_id
               for e in events)
    # Key 不落盘
    for jid in resp["jobs"]:
        cfg = storage.get_job_files(jid)["config.json"]
        assert "model_a_key_masked" in cfg and "k" not in str(cfg["model_a"].get("key_masked", ""))
    _drain_batch(batch_id)
    batch = storage.load_batch(batch_id)
    assert batch["state"] == "done"
    assert batch["leaderboard_id"]
    assert batch["failed_models"] == []
    events = audit.read_events()
    assert any(e["event"] == "benchmark_done" and e["target"] == batch_id
               for e in events)
    # 排行榜内容（单臂格式聚合）
    lb = storage.load_leaderboard(batch["leaderboard_id"])
    assert set(lb["models"]) == set(MODEL_NAMES)
    # 判别式满分（10×2 题×2 轮均值=10），生成式 8.0
    assert lb["composite"]["模型1"]["score"] == 28.0
    assert lb["ranks"]["模型1"] == 1
    assert lb["win_matrix"]["模型1"]["模型2"]["total"] == 3
    ci = lb["ci"]["模型1"]["模型2"]
    assert ci["n"] == 3 and ci["significant"] is False
    # 详情与排行榜端点
    detail = client.get(f"/api/benchmark/{batch_id}").json()
    assert detail["state"] == "done"
    assert detail["progress"] == "5/5"
    lr = client.get(f"/api/benchmark/{batch_id}/leaderboard")
    assert lr.status_code == 200
    assert set(lr.json()["models"]) == set(MODEL_NAMES)


def test_partial_failure_marks_na(client):
    """5 模型中 1 个调用失败 → batch partial + 失败模型 N/A + 排行榜排除。"""
    payload = _payload(client, n=5)
    # 最后一个模型 URL 指向失败
    from backend.models_registry import _MODEL_KEYS
    ids = payload["model_ids"]
    bad_id = ids[-1]
    main_module._jobs.clear()
    # 直接改内存 Key 指向 mock://fail（模型库无法存 mock URL，SSRF 会拦）
    _MODEL_KEYS[bad_id] = "k"
    from backend.models_registry import _model_path
    import json as _json
    p = _model_path(bad_id)
    info = _json.loads(p.read_text(encoding="utf-8"))
    info["url"] = "mock://fail/v1"
    p.write_text(_json.dumps(info, ensure_ascii=False), encoding="utf-8")

    resp = _start(**payload)
    batch_id = resp["batch_id"]
    _drain_batch(batch_id)
    batch = storage.load_batch(batch_id)
    assert batch["state"] == "partial"
    assert len(batch["failed_models"]) == 1
    assert batch["aggregation_error"] is None  # 4 个完成模型可聚合
    lb = storage.load_leaderboard(batch["leaderboard_id"])
    assert set(lb["models"]) == set(MODEL_NAMES[:4])
    detail = client.get(f"/api/benchmark/{batch_id}").json()
    assert detail["progress"] == "5/5"
    assert len(detail["failed_models"]) == 1


def test_leaderboard_404_before_completion(client):
    payload = _payload(client, n=2)
    resp = _start(**payload)
    r = client.get(f"/api/benchmark/{resp['batch_id']}/leaderboard")
    assert r.status_code == 404


def test_batch_validation_and_list(client):
    resp = _start(**_payload(client, n=2))
    assert resp["ok"] is True
    items = client.get("/api/benchmark").json()["batches"]
    assert any(x["batch_id"] == resp["batch_id"] for x in items)
    r = client.get("/api/benchmark/batch_bad")
    assert r.status_code == 400
    r = client.get("/api/benchmark/batch_00000000_000000_000000")
    assert r.status_code == 404
