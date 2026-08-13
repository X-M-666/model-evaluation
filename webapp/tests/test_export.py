# -*- coding: utf-8 -*-
"""评测包导出（迭代四）：GET /api/eval/{job_id}/export。

- 仅 completed 可导出；运行中/未开始 409；非法/不存在 404；
- zip 内容 = 白名单文件 + MANIFEST.json（逐文件 sha256）；
- 逐文件 sha256 与内容可核验（完整性保证）；
- 审计事件 dataset_exported 记录。
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import zipfile

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
    scores = [{"id": t["id"], "dimension": t["dimension"], "answer_x": 8.0,
               "answer_y": 7.0, "winner": "answer_x", "basis": "fake"}
              for t in task_set["tasks"]]
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


def test_export_rejects_non_completed(monkeypatch):
    async def _never(*a, **k):
        await asyncio.sleep(3600)

    monkeypatch.setattr(main_module, "_run_eval", _never)
    started = asyncio.run(main_module.start_eval(StartRequest(**_payload())))
    with pytest.raises(HTTPException) as ei:
        asyncio.run(main_module.eval_export(started.job_id))
    assert ei.value.status_code == 409


def test_export_404_for_bad_or_missing():
    with pytest.raises(HTTPException) as ei:
        asyncio.run(main_module.eval_export("20250101_000000_000000"))
    assert ei.value.status_code == 404
    with pytest.raises(HTTPException) as ei:
        asyncio.run(main_module.eval_export("..\\evil"))
    assert ei.value.status_code == 400


def test_export_zip_manifest_and_hashes():
    job_id = _completed_job_id()
    with TestClient(main_module.app) as client:
        resp = client.get(f"/api/eval/{job_id}/export")
        assert resp.status_code == 200
        data = resp.content
    zf = zipfile.ZipFile(io.BytesIO(data))
    names = set(zf.namelist())
    assert "MANIFEST.json" in names
    manifest = json.loads(zf.read("MANIFEST.json"))
    assert manifest["job_id"] == job_id
    assert manifest["total"] >= 5
    for name, sha in manifest["files"].items():
        assert hashlib.sha256(zf.read(name)).hexdigest() == sha
    assert "config.json" in manifest["files"]
    assert "report.json" in manifest["files"]
    assert "env.json" in manifest["files"]


def test_export_zip_no_secrets():
    job_id = _completed_job_id()
    with TestClient(main_module.app) as client:
        resp = client.get(f"/api/eval/{job_id}/export")
        data = resp.content
    zf = zipfile.ZipFile(io.BytesIO(data))
    flat = "".join(zf.read(n).decode("utf-8", "replace") for n in zf.namelist())
    assert "api_key" not in flat.lower()
    assert '"key": "k"' not in flat  # 配置中的 Key 均以 key_masked 落盘


def test_export_via_client_and_audit():
    job_id = _completed_job_id()
    audit._log_path().write_text("", encoding="utf-8")
    with TestClient(main_module.app) as client:
        resp = client.get(f"/api/eval/{job_id}/export")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/zip")
        assert f"eval-{job_id}.zip" in resp.headers["content-disposition"]
        body = resp.content
    assert body.startswith(b"PK")  # zip 魔数
    assert any(e["event"] == "dataset_exported" for e in audit.read_events())