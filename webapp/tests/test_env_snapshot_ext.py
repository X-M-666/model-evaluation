# -*- coding: utf-8 -*-
"""env.json 环境快照扩展（迭代八）：评测参数段（seed/温度/模型名/评审指纹/数据集版本）。

- build_env_snapshot(config) 补齐 eval 段（全部新字段可选，向后兼容）；
- judge_fingerprint：剔除 key 后确定性 SHA-256 前 12 位（换 Key 指纹不变、换配置指纹变）；
- start_eval 落盘的 env.json 含 eval 段（D9：两调用点均传 config）；
- 旧快照（无 eval 段）load 零破坏。
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
        "model_a": {"url": PUBLIC_URL, "key": "k", "name": "M1", "temperature": 0.3, "max_tokens": 2000},
        "model_b": {"url": PUBLIC_URL, "key": "k", "name": "M2", "temperature": 0.7, "max_tokens": 1000},
        "dataset_name": None,
        "seed": 12345,
        "prompt_strategy": "cot",
        "repeat_n": 3,
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


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    main_module._jobs.clear()
    main_module._tasks.clear()
    monkeypatch.setattr(main_module, "execute_all", _fake_execute_all)
    monkeypatch.setattr(main_module, "run_judge", _fake_run_judge)
    yield
    main_module._jobs.clear()
    main_module._tasks.clear()


def test_build_env_snapshot_includes_eval_fields():
    snap = storage.build_env_snapshot({**_payload(), "code_verify_mode": "off"})
    ev = snap["eval"]
    assert ev["seed"] == 12345
    assert ev["repeat_n"] == 3
    assert ev["prompt_strategy"] == "cot"
    assert ev["code_verify_mode"] == "off"
    assert ev["dataset_name"] is None
    assert ev["dataset_version"] == "n/a"
    assert ev["model_names"] == {"a": "M1", "b": "M2"}
    assert ev["temperature"] == {"a": 0.3, "b": 0.7}
    assert ev["max_tokens"] == {"a": 2000, "b": 1000}
    assert ev["review_mode"] == "pure_agent"
    assert ev["judge_config_fingerprint"]
    assert ev["review_k_top_human"] == 5        # ReviewConfig 默认 k_top_human=5 随 model_dump 入 config
    # 既有段保持
    assert snap["platform"]["system"] == platform.system()


def test_judge_fingerprint_deterministic_and_key_stripped():
    j1 = {"url": "https://x/v1", "name": "J", "key": "secret-1", "temperature": 0.0, "max_tokens": 256}
    j2 = {"url": "https://x/v1", "name": "J", "key": "secret-2", "temperature": 0.0, "max_tokens": 256}
    j3 = {"url": "https://x/v1", "name": "J", "key": "secret-1", "temperature": 0.5, "max_tokens": 256}
    fp1 = storage.judge_fingerprint(j1)
    assert fp1 == storage.judge_fingerprint(j2)      # 换 Key 指纹不变
    assert fp1 != storage.judge_fingerprint(j3)      # 换配置指纹变
    assert len(fp1) == 12
    assert fp1 == storage.judge_fingerprint(j1)      # 确定性
    assert "secret" not in fp1
    assert storage.judge_fingerprint(None) is None
    assert storage.judge_fingerprint({"key": "only"}) is None


def test_start_eval_env_snapshot_has_eval_section():
    started = asyncio.run(main_module.start_eval(StartRequest(**_payload())))
    env = storage.load_env_snapshot(started.job_id)
    assert env is not None
    ev = env["eval"]
    assert ev["seed"] == 12345
    assert ev["review_mode"] == "pure_agent"
    assert ev["judge_config_fingerprint"]
    assert ev["temperature"]["a"] == 0.3
    flat = json.dumps(env, ensure_ascii=False)
    assert "jk" not in flat        # 评审 Key 不进快照
    assert "12345" in flat         # seed 可见（非敏感）


def test_old_snapshot_without_eval_section_backward_compatible(tmp_path, monkeypatch):
    from pathlib import Path
    jid = "20250801_010203_abc123"
    job_dir = Path(tmp_path) / jid
    job_dir.mkdir(parents=True)
    (job_dir / "env.json").write_text(json.dumps(
        {"platform": {"system": "TestOS"}, "cwd": "/tmp", "packages": {}},
        ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(storage, "BASE_DIR", tmp_path)
    env = storage.load_env_snapshot(jid)
    assert env is not None
    assert "eval" not in env        # 旧快照零破坏：无 eval 段
    assert env["platform"]["system"] == "TestOS"
    # 新写则补全
    p = storage.save_env_snapshot(jid, {"model_a": {"name": "M", "temperature": 0.5}})
    assert p.exists()
    env2 = json.loads(p.read_text(encoding="utf-8"))
    assert env2["eval"]["model_names"]["a"] == "M"


def test_save_env_snapshot_without_config_keeps_legacy_shape(tmp_path, monkeypatch):
    from pathlib import Path
    jid = "20250801_010203_abc124"
    monkeypatch.setattr(storage, "BASE_DIR", tmp_path)
    p = storage.save_env_snapshot(jid)   # config=None：保持迭代四旧形态（零破坏）
    env = json.loads(p.read_text(encoding="utf-8"))
    assert "eval" not in env             # 旧调用点形态不变
    assert env["platform"]["system"]
