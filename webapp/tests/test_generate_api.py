# -*- coding: utf-8 -*-
"""出题批次 API（迭代四）集成测试：生成入队 → 校验 → 审核（approve/驳回）→
版本化入库 → 全链路可用于真实评测。

覆盖不变量：
- gen_config 必填、SSRF 校验、Key/URL 不明文落盘（spec 仅 gen_name + key_masked）
- 出题协程上限 429；批次 generating → ready；中断批次沉降 partial
- approve：单题校验（edits 字段白名单）、数据集追加 + 版本递增 + source=generated
- 超过 MAX_DATASET_TASKS 拒审；驳回终态；非 pending 409；未知批次/条目 404
- 生成数据集可直接用于 start_eval（全链路验收）

使用同款模式：monkeypatch run_generation_pipeline / execute_all，零真实网络。
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
from backend.schemas import GenerateRequest, ReviewDecisionRequest, StartRequest

PUBLIC_URL = "https://8.8.8.8/v1"


def _gen_payload(**overrides) -> dict:
    base = {
        "gen_config": {"url": PUBLIC_URL, "key": "gk", "name": "gen-model", "temperature": 0.7,
                       "max_tokens": 100},
        "task_type": "判别式",
        "dimension": "数学能力",
        "count": 2,
        "target_dataset": "生成集A",
        "options": {"cot": True},
    }
    base.update(overrides)
    return base


def _fake_item(prompt="生成的题目内容", expected="42", dimension="数学能力",
               task_type="判别式", ok=True, issues=None) -> dict:
    return {
        "task": {"prompt": prompt, "expected": expected, "difficulty": "medium",
                 "dimension": dimension, "type": task_type, "rubric_note": "满分10分"},
        "checks": {"dedup": {"ok": True, "sim": 0.1, "against": None},
                   "leakage": {"ok": True, "sim": 0.1, "hit": None},
                   "rubric": {"ok": True},
                   "solvable": {"status": "verified", "detail": ""},
                   "safety": {"status": "passed", "detail": ""}},
        "issues": issues or [],
        "ok": ok,
    }


async def _fake_pipeline(gen_config, spec, pool=None, leaked_extra=None,
                         client=None, progress_cb=None):
    return [_fake_item(), _fake_item(prompt="第二道生成题", expected="7")]


def _call(fn, *a, **k):
    """统一包装 async 路由函数（测试直调端点）。"""
    return asyncio.run(fn(*a, **k))


def _review(gen_id: str, item_id: str, action: str, edits=None):
    return _call(main_module.review_generated, gen_id, item_id,
                 ReviewDecisionRequest(action=action, edits=edits))


@pytest.fixture
def client():
    return TestClient(main_module.app)


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    main_module._jobs.clear()
    main_module._tasks.clear()
    monkeypatch.setattr(main_module, "run_generation_pipeline", _fake_pipeline)
    audit._log_path().write_text("", encoding="utf-8")  # 审计日志按用例隔离
    yield
    main_module._jobs.clear()
    main_module._tasks.clear()


def _settle(gen_id: str, state: str = "ready"):
    batch = storage.load_generation_batch(gen_id)
    assert batch is not None
    batch["state"] = state
    if state == "ready":
        batch["items"] = [
            {
                "item_id": f"{gen_id}-{i + 1}",
                "task": it["task"], "checks": it["checks"], "issues": it["issues"],
                "ok": it["ok"], "status": "pending", "edits": None, "reviewed_at": None,
            }
            for i, it in enumerate(asyncio.run(_fake_pipeline({}, {})))
        ]
    storage.save_generation_batch(gen_id, batch)
    return batch


def _start(req_kwargs: dict) -> str:
    resp = _call(main_module.generate_tasks, GenerateRequest(**req_kwargs))
    gen_id = resp["gen_id"]
    assert gen_id.startswith("gen_")
    return gen_id


# ---- 入队与校验 ----


def test_generate_requires_gen_config():
    with pytest.raises(HTTPException) as ei:
        _start(_gen_payload(gen_config=None))
    assert ei.value.status_code == 400


def test_generate_ssrf_rejects_private_url():
    with pytest.raises(HTTPException) as ei:
        _start(_gen_payload(
            gen_config={"url": "http://127.0.0.1:11434/v1", "key": "k", "name": "x"}))
    assert ei.value.status_code == 400


def test_generate_creates_batch_no_secret_persisted():
    gen_id = _start(_gen_payload())
    raw = json.loads((storage.GENERATED_DIR / f"{gen_id}.json").read_text(encoding="utf-8"))
    flat = json.dumps(raw["spec"], ensure_ascii=False)
    assert "gk" not in flat and PUBLIC_URL not in flat
    assert raw["spec"]["gen_key_masked"] == "***"

    detail = _call(main_module.generate_detail, gen_id)
    assert set(detail["item_stats"]) == {"total", "pending", "approved", "rejected"}


def test_generate_max_active_429():
    main_module._tasks["gen_a"] = object()
    main_module._tasks["gen_b"] = object()
    main_module._tasks["gen_c"] = object()
    with pytest.raises(HTTPException) as ei:
        _start(_gen_payload())
    assert ei.value.status_code == 429


def test_generation_background_fills_items():
    req = GenerateRequest(**_gen_payload())
    gen_id = _start(_gen_payload())
    _call(main_module._run_generation, gen_id, {"url": PUBLIC_URL, "key": "gk"}, req)
    detail = _call(main_module.generate_detail, gen_id)
    assert detail["state"] == "ready"
    assert detail["item_stats"]["total"] == 2
    assert all(it["status"] == "pending" for it in detail["items"])
    assert detail["items"][0]["item_id"] == f"{gen_id}-1"


def test_generation_error_state(monkeypatch):
    gen_id = _start(_gen_payload())

    async def _boom(*a, **k):
        raise RuntimeError("gen 模型不可用")

    monkeypatch.setattr(main_module, "run_generation_pipeline", _boom)
    _call(main_module._run_generation, gen_id, {"url": PUBLIC_URL, "key": "gk"},
          GenerateRequest(**_gen_payload()))
    detail = _call(main_module.generate_detail, gen_id)
    assert detail["state"] == "error"
    assert "gen 模型不可用" in detail["error"]


def test_generation_list_summary():
    gen_id = _start(_gen_payload())
    _settle(gen_id)
    entries = _call(main_module.generate_list)["batches"]
    entry = next(b for b in entries if b["gen_id"] == gen_id)
    assert entry["state"] == "ready"
    assert entry["items"]["total"] == 2
    assert entry["dimension"] == "数学能力"
    assert entry["gen_name"] == "gen-model"


def test_generate_multi_dimensions_persists_spec():
    resp = _call(main_module.generate_tasks, GenerateRequest(**_gen_payload(
        dimensions=["数学能力", "语言能力"], count=3)))
    assert resp["count"] == 6  # 每维度 count 题
    gen_id = resp["gen_id"]
    raw = json.loads((storage.GENERATED_DIR / f"{gen_id}.json").read_text(encoding="utf-8"))
    assert raw["spec"]["dimensions"] == ["数学能力", "语言能力"]
    assert raw["spec"]["dimension"] == "数学能力"  # 首维兼容字段


def test_generate_single_dimension_falls_back_to_list():
    resp = _call(main_module.generate_tasks, GenerateRequest(**_gen_payload()))
    gen_id = resp["gen_id"]
    raw = json.loads((storage.GENERATED_DIR / f"{gen_id}.json").read_text(encoding="utf-8"))
    assert raw["spec"]["dimensions"] == ["数学能力"]
    assert raw["spec"]["dimension"] == "数学能力"


# ---- 批次删除（迭代十一：审核完成可删除） ----

def test_generation_delete_success_and_list_gone():
    gen_id = _start(_gen_payload())
    resp = _call(main_module.generation_delete, gen_id)
    assert resp["deleted"] is True
    assert storage.load_generation_batch(gen_id) is None
    entries = _call(main_module.generate_list)["batches"]
    assert all(b["gen_id"] != gen_id for b in entries)


def test_generation_delete_missing_404():
    with pytest.raises(HTTPException) as ei:
        _call(main_module.generation_delete, "gen_20260101_000000_abcdef")
    assert ei.value.status_code == 404


def test_generation_delete_bad_id_400():
    with pytest.raises(HTTPException) as ei:
        _call(main_module.generation_delete, "../evil")
    assert ei.value.status_code == 400


def test_generation_delete_generating_409():
    gen_id = _start(_gen_payload())
    batch = storage.load_generation_batch(gen_id)
    batch["state"] = "generating"
    storage.save_generation_batch(gen_id, batch)
    with pytest.raises(HTTPException) as ei:
        _call(main_module.generation_delete, gen_id)
    assert ei.value.status_code == 409
    assert storage.load_generation_batch(gen_id) is not None


def test_generation_delete_audited():
    gen_id = _start(_gen_payload())
    _call(main_module.generation_delete, gen_id)
    events = audit.read_events()
    assert any(e["event"] == "generation_deleted" and e["target"] == gen_id
               for e in events)


def test_interrupted_generation_settles_partial():
    gen_id = _start(_gen_payload())
    _settle(gen_id, state="generating")
    detail = _call(main_module.generate_detail, gen_id)
    assert detail["state"] == "partial"


# ---- 审核 ----


def test_review_approve_appends_and_bumps_version():
    gen_id = _start(_gen_payload())
    _settle(gen_id)

    r = _review(gen_id, f"{gen_id}-1", "approve")
    assert r["status"] == "approved"
    assert r["dataset"] == "生成集A"
    assert r["version"] == "v1"  # 新数据集从 v1 起

    r2 = _review(gen_id, f"{gen_id}-2", "approve")
    assert r2["version"] == "v2"  # 已存在数据集递增

    ds = storage.load_dataset("生成集A")
    assert ds["version"] == "v2"
    assert ds["source"] == "generated"
    assert len(ds["tasks"]) == 2
    assert ds["tasks"][0]["prompt"].startswith("生成的题目内容")

    detail = _call(main_module.generate_detail, gen_id)
    assert detail["item_stats"] == {"total": 2, "pending": 0, "approved": 2, "rejected": 0}


def test_review_approve_with_edits_whitelist():
    gen_id = _start(_gen_payload())
    _settle(gen_id)
    r = _review(gen_id, f"{gen_id}-1", "approve",
                edits={"prompt": "人工修订后的题面", "difficulty": "hard"})
    assert r["status"] == "approved"
    ds = storage.load_dataset("生成集A")
    assert ds["tasks"][-1]["prompt"] == "人工修订后的题面"
    assert ds["tasks"][-1]["difficulty"] == "hard"


def test_review_edits_invalid_field_rejected():
    gen_id = _start(_gen_payload())
    _settle(gen_id)
    with pytest.raises(HTTPException) as ei:
        _review(gen_id, f"{gen_id}-1", "approve", edits={"evil": "x"})
    assert ei.value.status_code == 400
    assert "evil" in ei.value.detail


def test_review_edits_invalid_task_rejected():
    gen_id = _start(_gen_payload())
    _settle(gen_id)
    with pytest.raises(HTTPException) as ei:
        _review(gen_id, f"{gen_id}-1", "approve", edits={"prompt": ""})
    assert ei.value.status_code == 400


def test_review_reject_terminal():
    gen_id = _start(_gen_payload())
    _settle(gen_id)
    assert _review(gen_id, f"{gen_id}-1", "reject")["status"] == "rejected"
    with pytest.raises(HTTPException) as ei:
        _review(gen_id, f"{gen_id}-1", "approve")
    assert ei.value.status_code == 409


def test_review_generating_rejected():
    gen_id = _start(_gen_payload())
    main_module._tasks[gen_id] = object()  # 模拟出题协程仍在运行（_settle_generation 不沉降）
    _settle(gen_id, state="generating")
    with pytest.raises(HTTPException) as ei:
        _review(gen_id, f"{gen_id}-1", "approve")
    assert ei.value.status_code == 409


def test_review_over_capacity_400():
    gen_id = _start(_gen_payload())
    _settle(gen_id)
    storage.save_dataset("生成集A", {
        "name": "生成集A",
        "tasks": [
            {"id": f"T{i}", "prompt": f"p{i}",
             "test_cases": [{"input": "i", "expected": "e"}]}
            for i in range(200)
        ],
        "version": "v9",
    })
    with pytest.raises(HTTPException) as ei:
        _review(gen_id, f"{gen_id}-1", "approve")
    assert ei.value.status_code == 400


def test_review_not_found_and_bad_id():
    gen_id = _start(_gen_payload())
    _settle(gen_id)
    with pytest.raises(HTTPException) as ei:
        _review("gen_bad", "x-1", "approve")
    assert ei.value.status_code == 400
    with pytest.raises(HTTPException) as ei:
        _review(gen_id, "nope", "approve")
    assert ei.value.status_code == 404
    with pytest.raises(HTTPException) as ei:
        _call(main_module.generate_detail, "gen_不存在")
    assert ei.value.status_code == 400


# ---- 全链路：生成 → 审核 → 入库 → 真实评测 ----


async def _fake_execute_all(task_set, config_a=None, config_b=None, progress_cb=None, **kwargs):
    def mk(model):
        return {"model": model, "api": {"name": model, "url": PUBLIC_URL},
                "note": "fake", "answers": [
                    {"id": t["id"], "raw_answer": "42", "api_info": {"status": "ok",
                     "attempts": 1, "truncated": False, "error": None, "latency_ms": 5,
                     "prompt_tokens": 3, "completion_tokens": 2, "repeat_index": 1}}
                    for t in task_set["tasks"]]}
    return mk(config_a["name"]), mk(config_b["name"])


async def _fake_run_judge(task_set, answers_x, answers_y, judge_config, revealed=None,
                          progress_cb=None, **kwargs):
    scores = []
    for t in task_set["tasks"]:
        scores.append({"id": t["id"], "dimension": t["dimension"], "answer_x": 8.0,
                       "answer_y": 7.0, "winner": "answer_x", "basis": "fake"})
    return {"scores": scores,
            "meta": {"total": len(scores), "valid": len(scores), "invalid": 0},
            "health": {"healthy": True, "invalid_rate": 0.0, "alarm": False},
            "conclusion": "fake", "revealed": revealed or {"answer_x": "a", "answer_y": "b"},
            "source": "AI评审"}


def test_generate_to_eval_full_chain(monkeypatch):
    gen_id = _start(_gen_payload(target_dataset="全链路集"))
    _settle(gen_id)
    assert _review(gen_id, f"{gen_id}-1", "approve")["status"] == "approved"
    assert _review(gen_id, f"{gen_id}-2", "approve")["status"] == "approved"

    monkeypatch.setattr(main_module, "execute_all", _fake_execute_all)
    monkeypatch.setattr(main_module, "run_judge", _fake_run_judge)
    from backend.schemas import ModelConfig, ReviewConfig
    payload = {
        "model_a": {"url": PUBLIC_URL, "key": "k", "name": "M1", "temperature": 0.7, "max_tokens": 1000},
        "model_b": {"url": PUBLIC_URL, "key": "k", "name": "M2", "temperature": 0.7, "max_tokens": 1000},
        "dataset_name": "全链路集",
        "prompt_strategy": "direct",
        "review": ReviewConfig(mode="pure_agent", judge=ModelConfig(
            url=PUBLIC_URL, key="jk", name="J", temperature=0.0, max_tokens=256)).model_dump(),
    }
    started = _call(main_module.start_eval, StartRequest(**payload))
    assert started.job_id

    import time as _t
    st = None
    for _ in range(100):
        st = main_module.get_job_status(started.job_id)
        if st and st["state"] in ("completed", "error"):
            break
        _t.sleep(0.02)
    assert st and st["state"] == "completed"

    files = main_module.get_job_files(started.job_id)
    assert files["report.json"]["report"]["judge_mode"] == "pure_agent"
    assert files["report.json"]["tasks"]["meta"]["dataset_version"] == "v2"

    events = audit.read_events()
    assert any(e["event"] == "task_generate_started" for e in events)
    assert sum(1 for e in events if e["event"] == "task_reviewed") == 2