# -*- coding: utf-8 -*-
"""安全回归测试：验证 API Key 永不落盘、不通过历史/报告接口外泄。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend import main as main_module
from backend import storage
from backend.security import SENSITIVE_KEYS, redact_sensitive, sanitize_config

SENTINEL = "sk-secret-TEST-123456"


def _assert_no_sensitive(data, path: str = "root"):
    if isinstance(data, dict):
        for k, v in data.items():
            assert k.lower() not in SENSITIVE_KEYS, f"{path}.{k} 泄露敏感字段"
            _assert_no_sensitive(v, f"{path}.{k}")
    elif isinstance(data, list):
        for i, v in enumerate(data):
            _assert_no_sensitive(v, f"{path}[{i}]")


def _iter_json_files(job_id: str) -> list[Path]:
    d = storage.BASE_DIR / job_id
    return sorted(p for p in d.rglob("*.json") if p.is_file())


@pytest.fixture
def client():
    return TestClient(main_module.app)


def test_redact_sensitive_nested():
    obj = {
        "model_a": {"key": "x", "api_key": "y", "name": "m", "nested": {"token": "z", "ok": 1}},
        "list": [{"authorization": "a", "keep": True}, "str"],
    }
    out = redact_sensitive(obj)
    _assert_no_sensitive(out)
    assert out["model_a"]["name"] == "m"
    assert out["model_a"]["nested"]["ok"] == 1
    assert out["list"][0]["keep"] is True


def test_redact_sensitive_does_not_mutate_source():
    obj = {"model_a": {"key": "x", "name": "m"}}
    redact_sensitive(obj)
    assert obj["model_a"]["key"] == "x"


def test_redact_sensitive_idempotent():
    obj = {"model_a": {"key": "x", "nested": [{"secret": "y"}]}}
    once = redact_sensitive(obj)
    twice = redact_sensitive(once)
    assert once == twice


def test_sanitize_config_removes_key_keeps_display_fields():
    cfg = {
        "model_a": {"name": "A", "url": "https://a/v1", "key": "x", "temperature": 0.7, "max_tokens": 4096},
        "model_b": {"name": "B", "url": "https://b/v1", "key": "y", "top_p": 1.0},
        "dims": ["知识能力"], "seed": 42, "repeat_n": 1, "dataset_name": None,
    }
    safe = sanitize_config(cfg)
    assert "key" not in safe["model_a"] and "key" not in safe["model_b"]
    assert safe["model_a"]["name"] == "A" and safe["model_a"]["url"] == "https://a/v1"
    assert safe["model_a"]["temperature"] == 0.7
    assert safe["model_b"]["top_p"] == 1.0
    assert safe["seed"] == 42 and safe["repeat_n"] == 1
    assert cfg["model_a"]["key"] == "x"  # 原配置不受影响


def _submit_review(client, job_id: str) -> None:
    task_set = main_module._jobs[job_id]["task_set"]
    scores = [{"id": t["id"], "round": 1, "answer_x": 5, "answer_y": 5} for t in task_set["tasks"]]
    r = client.post(f"/api/eval/{job_id}/review", json={"scores": scores})
    assert r.status_code == 200, r.text


def test_e2e_no_key_persisted_or_returned(client):
    r = client.post("/api/eval/mock")
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    # 注入哨兵 Key，覆盖 _finalize_job 写盘与报告接口的脱敏
    main_module._jobs[job_id]["config"]["model_a"]["key"] = SENTINEL
    main_module._jobs[job_id]["config"]["model_b"]["key"] = SENTINEL

    _submit_review(client, job_id)

    # 1) 落盘文件递归检查：不含 key 字段，也不含哨兵值
    files = _iter_json_files(job_id)
    assert files, "应存在历史 JSON 文件"
    for p in files:
        text = p.read_text(encoding="utf-8")
        assert SENTINEL not in text, f"{p} 含明文 Key"
        _assert_no_sensitive(json.loads(text), str(p))

    # 2) 报告响应（内存分支）：不泄露且保留展示字段
    for endpoint in (
        f"/api/eval/{job_id}/report",
        f"/api/history/{job_id}",
        f"/api/eval/{job_id}/status",
    ):
        resp = client.get(endpoint)
        assert resp.status_code == 200, f"{endpoint} -> {resp.status_code}"
        assert SENTINEL not in resp.text, f"{endpoint} 响应含明文 Key"
        _assert_no_sensitive(resp.json())

    report_resp = client.get(f"/api/eval/{job_id}/report").json()
    assert report_resp["config"]["model_a"]["name"]
    assert report_resp["config"]["model_b"]["name"]

    # 3) 报告响应（磁盘分支）：模拟服务重启后从文件恢复
    main_module._jobs.pop(job_id)
    for endpoint in (
        f"/api/eval/{job_id}/report",
        f"/api/eval/{job_id}/status",
        "/api/history",
        f"/api/history/{job_id}",
    ):
        resp = client.get(endpoint)
        assert resp.status_code == 200, f"{endpoint} -> {resp.status_code}"
        assert SENTINEL not in resp.text, f"{endpoint} 响应含明文 Key"
        _assert_no_sensitive(resp.json())


def test_legacy_plaintext_config_never_leaks_via_report_api(client):
    """回归（补强点2 / issue #1 验收3）：修复前遗留的明文 config.json
    （save_config 打码之前的旧记录）不得经报告接口泄露。

    构造一份修复前风格的 config.json（model_a.key / model_b.key 为明文），
    请求报告/历史/状态接口，断言响应中既无 key 字段也无明文值。
    """
    from backend.storage import create_job_id

    job_id = create_job_id()
    job_dir = storage.BASE_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    legacy_cfg = {
        "created_at": "2026-01-01T00:00:00Z",
        "model_a": {"name": "LegacyA", "url": "https://a/v1", "key": SENTINEL,
                    "temperature": 0.7, "max_tokens": 4096},
        "model_b": {"name": "LegacyB", "url": "https://b/v1", "key": SENTINEL,
                    "temperature": 0.7, "max_tokens": 4096},
        "repeat_n": 1,
    }
    (job_dir / "config.json").write_text(
        json.dumps(legacy_cfg, ensure_ascii=False), encoding="utf-8")
    # 手工构造"已完成"状态所需的最小文件集（report 分支从磁盘恢复）
    (job_dir / "tasks.json").write_text(
        json.dumps({"tasks": [{"id": "T1", "dimension": "知识能力",
                               "prompt": "p", "test_cases": []}]}, ensure_ascii=False),
        encoding="utf-8")
    answers = {"model": "LegacyA", "api": {"name": "LegacyA", "url": "https://a/v1"},
               "answers": [{"id": "T1", "raw_answer": "x", "api_info": {"status": "ok"}}]}
    (job_dir / "answers-a.json").write_text(
        json.dumps(answers, ensure_ascii=False), encoding="utf-8")
    (job_dir / "answers-b.json").write_text(
        json.dumps(answers, ensure_ascii=False), encoding="utf-8")
    (job_dir / "verdict.json").write_text(
        json.dumps({"scores": [{"id": "T1", "answer_x": 5, "answer_y": 5, "winner": "tie"}],
                    "totals": {"answer_x": 5, "answer_y": 5},
                    "revealed": {"answer_x": "LegacyA", "answer_y": "LegacyB",
                                 "answer_x_file": "a", "answer_y_file": "b"}},
                   ensure_ascii=False), encoding="utf-8")

    for endpoint in (
        f"/api/eval/{job_id}/report",
        f"/api/history/{job_id}",
        f"/api/eval/{job_id}/status",
    ):
        resp = client.get(endpoint)
        assert resp.status_code == 200, f"{endpoint} -> {resp.status_code}"
        assert SENTINEL not in resp.text, f"{endpoint} 响应含明文 Key"
        _assert_no_sensitive(resp.json())
