# -*- coding: utf-8 -*-
"""env.json 环境快照（迭代四）：评测可复现性元数据。

- start_eval 时写入 <job>/env.json（OS/Python/CPU/依赖版本，无密钥）；
- get_job_files 白名单包含 env.json；
- 报告 report.json 内嵌 env_snapshot 段（未写入时兜底实时采集）；
- 快照不包含任何 API Key / 敏感配置。
"""
from __future__ import annotations

import asyncio
import json
import platform

import pytest
from fastapi.testclient import TestClient

from backend import main as main_module
from backend import storage
from backend.schemas import ModelConfig, ReviewConfig, StartRequest

PUBLIC_URL = "https://8.8.8.8/v1"


def _payload():
    return {
        "model_a": {"url": PUBLIC_URL, "key": "k", "name": "M1", "temperature": 0.7, "max_tokens": 1000},
        "model_b": {"url": PUBLIC_URL, "key": "k", "name": "M2", "temperature": 0.7, "max_tokens": 1000},
        "dataset_name": None,  # 内置题库，无需预先准备数据集
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


def test_env_snapshot_written_on_start():
    started = asyncio.run(main_module.start_eval(StartRequest(**_payload())))
    env = storage.load_env_snapshot(started.job_id)
    assert env is not None
    assert env["platform"]["system"] == platform.system()
    assert env["platform"]["python_version"].startswith("3")
    assert isinstance(env["packages"], dict)
    assert "fastapi" in env["packages"]
    assert "cwd" in env


def test_env_snapshot_no_secrets():
    started = asyncio.run(main_module.start_eval(StartRequest(**_payload())))
    env = storage.load_env_snapshot(started.job_id)
    flat = json.dumps(env, ensure_ascii=False)
    assert '"k"' not in flat  # 不含 API Key 明文（M1/M2 的 key="k"）
    assert '"jk"' not in flat  # 评审模型 Key 明文同样不落盘
    assert "api_key" not in flat and "apikey" not in flat.lower()
    assert "sk-" not in flat


def test_get_job_files_includes_env():
    started = asyncio.run(main_module.start_eval(StartRequest(**_payload())))
    files = main_module.get_job_files(started.job_id)
    assert files["env.json"] is not None
    assert files["env.json"]["platform"]["system"] == platform.system()


def test_report_embeds_env_snapshot():
    started = asyncio.run(main_module.start_eval(StartRequest(**_payload())))
    import time as _t
    st = None
    for _ in range(100):
        st = main_module.get_job_status(started.job_id)
        if st and st["state"] in ("completed", "error"):
            break
        _t.sleep(0.02)
    assert st and st["state"] == "completed"
    files = main_module.get_job_files(started.job_id)
    env = files["report.json"]["report"]["env_snapshot"]
    assert env["platform"]["system"] == platform.system()


def test_env_snapshot_missing_job_returns_none():
    assert storage.load_env_snapshot("20250101_000000_000000") is None
    assert storage.load_env_snapshot("gen_1") is None