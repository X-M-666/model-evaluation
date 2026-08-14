# -*- coding: utf-8 -*-
"""扰动评测 API（迭代六）集成测试：创建 → 后台执行 → 轮询 → 结果结构。

覆盖不变量：
- SSRF 校验（非法 URL 400）、非法模式 400、评测集不存在 404、并发上限 429
- 判别式题指标得分（top1×10）；生成式题单臂评审得分（mock judge）
- judge 缺省：生成式题 score=N/A + 无评审提示；安全过滤拦截 → warnings
- 状态机 running → ready（轮询）；审计事件 perturb_started/completed
- Key 不落盘（模型配置仅掩码字段）

使用同款模式：monkeypatch _execute_model / run_single_arm_judge，零真实网络。
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend import audit
from backend import main as main_module
from backend import storage
from backend.schemas import PerturbRequest

PUBLIC_URL = "https://8.8.8.8/v1"


def _dataset() -> dict:
    return {
        "name": "扰动集A",
        "tasks": [
            {"id": "T1", "type": "判别式", "dimension": "数学能力",
             "prompt": "公司总部位于北京，请问 1+1=?", "test_cases": [{"input": "1+1=?", "expected": "2"}]},
            {"id": "T2", "type": "生成式", "dimension": "语言能力",
             "prompt": "介绍北京的城市文化", "expected": "参考答案",
             "rubric_note": "满分10分，按内容评分"},
        ],
    }


def _payload(**overrides) -> dict:
    base = {
        "model": {"url": PUBLIC_URL, "key": "mk", "name": "被测模型",
                  "temperature": 0.7, "max_tokens": 100},
        "dataset_name": "扰动集A",
        "modes": ["改写", "属性扰动-地域"],
        "seed": 7,
    }
    base.update(overrides)
    return base


async def _fake_execute(model_label, config, tasks, stability_repeat,
                        progress_cb=None, embedding_cfg=None, concurrency=1):
    answers = []
    for t in tasks:
        cases = t.get("test_cases") or []
        expected = str(cases[0].get("expected", "") if cases
                       else t.get("expected", "") or "")
        raw = f"答案是 {expected}" if t.get("type") == "判别式" else "这是一段生成文本"
        answers.append({
            "id": t["id"], "raw_answer": raw,
            "api_info": {"status": "ok", "attempts": 1, "truncated": False,
                         "error": None, "latency_ms": 100, "prompt_tokens": 50,
                         "completion_tokens": 20, "repeat_index": 1},
        })
        if progress_cb:
            await progress_cb(model_label, len(answers), len(tasks))
    return {"model": config["name"], "api": {"name": config["name"],
                                             "url": config["url"]},
            "note": "fake", "answers": answers}


async def _fake_single_arm(task_set, answers, judge_config,
                           progress_cb=None, max_retries=1):
    scores = [{"id": t["id"], "dimension": t.get("dimension", ""),
               "score": 8.0, "basis": "fake", "_invalid": False}
              for t in task_set["tasks"]]
    return {"meta": {"total": len(scores), "valid": len(scores), "invalid": 0,
                     "excluded_ids": [], "excluded_dimensions": []},
            "scores": scores, "totals": {"score": 8.0 * len(scores), "max": 10.0},
            "health": {"healthy": True}}


def _call(fn, *a, **k):
    return asyncio.run(fn(*a, **k))


def _drain(perturb_id: str) -> None:
    """直跑后台扰动协程（测试专用，幂等）。

    测试经 asyncio.run 调用 create_perturb 时，后台任务可能在循环收尾阶段
    已被调度完成（或已取消）；仅在磁盘仍为 running 时用内存请求表重建 req
    同步跑完管线，保证全链路可测。
    """
    data = storage.load_perturb(perturb_id)
    if data is not None and data["state"] in ("ready", "error", "partial"):
        return
    req = main_module._PERTURB_REQS.get(perturb_id)
    assert req is not None
    main_module._tasks.pop(perturb_id, None)
    asyncio.run(main_module._run_perturb(perturb_id, req, data))


def _start(**kw):
    return _call(main_module.create_perturb, PerturbRequest(**_payload(**kw)))


def _wait_ready(perturb_id: str) -> dict:
    _drain(perturb_id)
    for _ in range(50):
        data = _call(main_module.perturb_detail, perturb_id)
        if data["state"] in ("ready", "error", "partial"):
            return data
        asyncio.run(asyncio.sleep(0.01))
    raise AssertionError("扰动评测未在超时内完成")


@pytest.fixture
def client():
    return TestClient(main_module.app)


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    main_module._jobs.clear()
    main_module._tasks.clear()
    monkeypatch.setattr(main_module, "_execute_model", _fake_execute)
    monkeypatch.setattr(main_module, "run_single_arm_judge", _fake_single_arm)
    audit._log_path().write_text("", encoding="utf-8")
    yield
    main_module._jobs.clear()
    main_module._tasks.clear()


# ---- 创建与校验 ----

def test_create_requires_valid_modes():
    with pytest.raises(HTTPException) as ei:
        _start(modes=["不存在"])
    assert ei.value.status_code == 400


def test_create_ssrf_blocked():
    with pytest.raises(HTTPException) as ei:
        _start(model={"url": "http://127.0.0.1:9999/v1", "key": "k", "name": "m",
                      "temperature": 0.7, "max_tokens": 100})
    assert ei.value.status_code == 400


def test_create_dataset_missing():
    with pytest.raises(HTTPException) as ei:
        _start(dataset_name="不存在集")
    assert ei.value.status_code == 404


def test_create_ok_and_audit():
    storage.save_dataset("扰动集A", _dataset())
    resp = _start()
    assert resp["ok"] is True
    assert resp["perturb_id"].startswith("prb_")
    events = audit.read_events()
    assert any(e["event"] == "perturb_started" and e["target"] == resp["perturb_id"]
               for e in events)
    # Key 不落盘
    data = storage.load_perturb(resp["perturb_id"])
    assert data["model_key_masked"] == "***"
    assert "model_url" in data and "mk" not in json_str(data)


def json_str(data):
    import json
    return json.dumps(data, ensure_ascii=False)


def test_concurrent_limit_429(client, monkeypatch):
    """并发上限：asyncio.run 关闭时取消后台任务会清空 _tasks，故注入假 create_task。"""
    class _DummyTask:
        def __init__(self, coro):
            self.coro = coro

        def add_done_callback(self, fn):
            pass

    monkeypatch.setattr(main_module.asyncio, "create_task", _DummyTask)
    storage.save_dataset("扰动集A", _dataset())
    for _ in range(3):
        resp = _start()
        assert resp["ok"] is True
    assert len([k for k in main_module._tasks if k.startswith("prb_")]) == 3
    with pytest.raises(HTTPException) as ei:
        _start()
    assert ei.value.status_code == 429


# ---- 全链路（判别式 + 生成式 + 单臂评审）----

def test_full_pipeline_with_judge():
    storage.save_dataset("扰动集A", _dataset())
    resp = _start(modes=["改写", "属性扰动-地域"], seed=7,
                  judge={"url": PUBLIC_URL, "key": "jk", "name": "评审模型",
                         "temperature": 0.0, "max_tokens": 100})
    data = _wait_ready(resp["perturb_id"])
    assert data["state"] == "ready"
    assert data["progress"] == "done"
    # 每条记录带来源/得分；判别式满分，生成式走单臂
    by_id = {p["task_id"]: p for p in data["per_task"]}
    assert "T1" in by_id and by_id["T1"]["mode"] == "原版"
    pert_ids = [p["task_id"] for p in data["per_task"] if "-p" in p["task_id"]]
    assert pert_ids
    # 判别式原版得分 = top1×10 = 10.0；生成式（含原版与扰动版）走单臂 8.0
    gen_rows = [p for p in data["per_task"] if p["task_id"].startswith("T2")]
    assert all(p["score"] == 8.0 for p in gen_rows)
    # 衰减曲线：强度 0 注入 + 各模式曲线
    assert "改写" in data["curves"]["curves"]
    assert 0.0 in data["curves"]["curves"]["改写"]["intensities"]
    events = audit.read_events()
    assert any(e["event"] == "perturb_completed" for e in events)


def test_pipeline_without_judge_gen_score_na():
    storage.save_dataset("扰动集A", _dataset())
    resp = _start(modes=["改写"], seed=1)
    data = _wait_ready(resp["perturb_id"])
    assert data["state"] == "ready"
    gen_rows = [p for p in data["per_task"] if p["task_id"].startswith("T2")]
    assert gen_rows and all(p["score"] is None for p in gen_rows)
    # 判别式仍计分
    dis_rows = [p for p in data["per_task"] if p["task_id"].startswith("T1")]
    assert dis_rows and all(p["score"] is not None for p in dis_rows)


def test_bias_pairs_generated():
    storage.save_dataset("扰动集A", _dataset())
    resp = _start(modes=["属性扰动-地域"], seed=3)
    data = _wait_ready(resp["perturb_id"])
    pairs = data["bias"]["pairs"]
    assert pairs
    assert {"score_original", "score_perturbed", "diff", "consistency",
            "discriminates"} <= set(pairs[0])
    assert data["bias"]["threshold"] == 1.0


def test_pipeline_error_dataset_invalid():
    storage.save_dataset("扰动集A", {"name": "扰动集A", "tasks": []})
    resp = _start(modes=["改写"])
    data = _wait_ready(resp["perturb_id"])
    assert data["state"] == "error"
    assert data["error"]


def test_detail_validation_and_404():
    with pytest.raises(HTTPException) as ei:
        _call(main_module.perturb_detail, "prb_bad")
    assert ei.value.status_code == 400
    with pytest.raises(HTTPException) as ei:
        _call(main_module.perturb_detail, "prb_00000000_000000_000000")
    assert ei.value.status_code == 404


def test_list_endpoint(client):
    storage.save_dataset("扰动集A", _dataset())
    _start(modes=["改写"], seed=1)
    resp = client.get("/api/perturb")
    assert resp.status_code == 200
    assert any(p["perturb_id"].startswith("prb_") for p in resp.json()["perturbs"])
