# -*- coding: utf-8 -*-
"""Bad Case API 集成（迭代五）：job 完成自动挖掘入库 → 列表/详情/统计/确认/导出。

- _finalize_job 接线：saturation 幂等写入 + bad case 同步入库 + 审计 badcase_mined；
- 异步 LLM 归因（monkeypatch attribute_badcase）→ attribution by=llm + 审计；
- 人工确认/改标/驳回语义（400/404）；
- 导出 JSON 下载、筛选参数。
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
from backend.schemas import ModelConfig, ReviewConfig, StartRequest

PUBLIC_URL = "https://8.8.8.8/v1"


def _payload():
    return {
        "model_a": {"url": PUBLIC_URL, "key": "k", "name": "M1", "temperature": 0.7, "max_tokens": 1000},
        "model_b": {"url": PUBLIC_URL, "key": "k", "name": "M2", "temperature": 0.7, "max_tokens": 1000},
        "dataset_name": None,
        "prompt_strategy": "direct",
        "review": ReviewConfig(mode="pure_agent", judge=ModelConfig(
            url=PUBLIC_URL, key="jk", name="J", temperature=0.0, max_tokens=256)).model_dump(),
    }


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
    for i, t in enumerate(task_set["tasks"]):
        x = 2.0 if i == 0 else 8.0  # 第一题低分 → 触发 low_score 挖掘（分差<3 避免分歧）
        y = 4.0 if i == 0 else 8.0
        scores.append({"id": t["id"], "dimension": t["dimension"], "answer_x": x,
                       "answer_y": y, "winner": "answer_x" if x > y else "answer_y",
                       "basis": "fake"})
    return {"scores": scores,
            "meta": {"total": len(scores), "valid": len(scores), "invalid": 0},
            "health": {"healthy": True, "invalid_rate": 0.0, "alarm": False},
            "conclusion": "fake", "revealed": revealed or {"answer_x": "a", "answer_y": "b"},
            "source": "AI评审"}


@pytest.fixture
def client():
    return TestClient(main_module.app)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    main_module._jobs.clear()
    main_module._tasks.clear()
    monkeypatch.setattr(main_module, "execute_all", _fake_execute_all)
    monkeypatch.setattr(main_module, "run_judge", _fake_run_judge)
    audit._log_path().write_text("", encoding="utf-8")
    yield
    main_module._jobs.clear()
    main_module._tasks.clear()


def _completed_job_id():
    started = asyncio.run(main_module.start_eval(StartRequest(**_payload())))
    import time as _t
    st = None
    for _ in range(100):
        st = main_module.get_job_status(started.job_id)
        if st and st["state"] in ("completed", "error"):
            break
        _t.sleep(0.02)
    assert st and st["state"] == "completed"
    return started.job_id


def test_job_completion_mines_bad_cases_and_saturation():
    job_id = _completed_job_id()
    cases = storage.list_badcases(job_id)
    assert len(cases) >= 1
    assert cases[0]["sources"] == ["low_score"]
    assert cases[0]["category"] == "未归类"
    assert cases[0]["model"] == "x"
    assert cases[0]["case_id"].startswith("bc_" + job_id)

    sat = storage.get_saturation()
    assert any(j["job_id"] == job_id for j in sat["jobs"])

    events = audit.read_events()
    assert any(e["event"] == "badcase_mined" and e["target"] == job_id for e in events)


def test_badcase_list_filters_and_detail():
    job_id = _completed_job_id()
    with TestClient(main_module.app) as client:
        r = client.get("/api/badcases")
        assert r.status_code == 200
        all_cases = r.json()
        assert all_cases["total"] >= 1

        r2 = client.get(f"/api/badcases?job_id={job_id}")
        assert r2.json()["total"] >= 1

        r3 = client.get(f"/api/badcases?job_id={job_id}&category=未归类")
        assert r3.json()["total"] >= 1

        r4 = client.get(f"/api/badcases?job_id={job_id}&category=推理错误")
        assert r4.json()["total"] == 0

        r5 = client.get(f"/api/badcases?confirmed=false")
        assert r5.json()["total"] >= 1

        case_id = r2.json()["cases"][0]["case_id"]
        d = client.get(f"/api/badcases/{case_id}")
        assert d.status_code == 200
        assert d.json()["evidence"]["basis"] == "fake"
        assert d.json()["attribution"]["by"] == "auto"


def test_badcase_stats_shape():
    _completed_job_id()
    with TestClient(main_module.app) as client:
        r = client.get("/api/badcases/stats")
        assert r.status_code == 200
        s = r.json()
    assert s["total"] >= 1
    assert "low_score" in s["by_source"]
    assert s["by_category"]["未归类"] >= 1
    assert "事实错误" in s["categories"]


def test_badcase_404_and_400():
    with TestClient(main_module.app) as client:
        assert client.get("/api/badcases/../evil").status_code == 404
        assert client.get("/api/badcases/bc_bad_id").status_code == 400
        assert client.get("/api/badcases/bc_20260101_000000_000001_ZZZ").status_code == 404
        r = client.post("/api/badcases/bc_20260101_000000_000001_ZZZ/attribution",
                        json={"category": "推理错误"})
        assert r.status_code == 404
        r2 = client.post("/api/badcases/bc_bad_id/attribution", json={"category": "x"})
        assert r2.status_code == 400


def test_human_attribution_confirm_and_revert():
    job_id = _completed_job_id()
    case = storage.list_badcases(job_id)[0]
    with TestClient(main_module.app) as client:
        r = client.post(f"/api/badcases/{case['case_id']}/attribution",
                        json={"category": "推理错误", "suggestion": "补充前提"})
        assert r.status_code == 200
        assert r.json()["category"] == "推理错误"

        r2 = client.post(f"/api/badcases/{case['case_id']}/attribution",
                        json={"category": "未知类别"})
        assert r2.status_code == 400

    loaded = storage.load_badcase(case["case_id"])
    assert loaded["attribution"]["label"] == "推理错误"
    assert loaded["attribution"]["confirmed"] is True
    assert loaded["attribution"]["by"] == "human"
    assert loaded["attribution"]["suggestion"] == "补充前提"

    events = audit.read_events()
    assert any(e["event"] == "badcase_attribution" and e["target"] == case["case_id"]
               and e["path"] == "confirm" for e in events)


def test_llm_attribution_async_updates(monkeypatch):
    from backend.engine.badcase import attribute_badcase as _real

    async def _fake_attr(case, task, judge_config, client=None):
        return {"label": "推理错误", "by": "llm", "confirmed": False,
                "basis": "步骤2出错", "suggestion": "补充条件", "updated_at": "now"}

    monkeypatch.setattr("backend.main.attribute_badcase", _fake_attr)
    job_id = _completed_job_id()
    # 等待后台归因协程完成
    import time as _t
    for _ in range(100):
        cases = storage.list_badcases(job_id)
        if cases and cases[0]["attribution_by"] == "llm":
            break
        _t.sleep(0.02)
    cases = storage.list_badcases(job_id)
    assert cases[0]["attribution_by"] == "llm"
    assert cases[0]["category"] == "推理错误"
    events = audit.read_events()
    assert any(e["event"] == "badcase_attribution" and e["path"] == "llm" for e in events)


def test_llm_attribution_failure_keeps_uncategorized(monkeypatch):
    async def _fake_attr(case, task, judge_config, client=None):
        return None  # 归因失败 → 保持未归类

    monkeypatch.setattr("backend.main.attribute_badcase", _fake_attr)
    job_id = _completed_job_id()
    import time as _t
    _t.sleep(0.1)
    cases = storage.list_badcases(job_id)
    assert cases[0]["category"] == "未归类"
    assert cases[0]["attribution_by"] == "auto"


def test_export_json():
    job_id = _completed_job_id()
    with TestClient(main_module.app) as client:
        r = client.get(f"/api/badcases/export?job_id={job_id}")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/json")
        assert f"badcases-{job_id}.json" in r.headers["content-disposition"]
        data = r.json()
    assert data["total"] >= 1
    assert data["cases"][0]["job_id"] == job_id