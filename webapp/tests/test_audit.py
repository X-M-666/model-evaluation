# -*- coding: utf-8 -*-
"""审计日志测试（补强方案 #1：审计日志）。

保护不变量：
- 事件字段白名单 + 写入前递归脱敏，API Key 永不进入日志
- 关键操作（评测启动/评审提交/历史删除/数据集上传与删除/鉴权失败）均有事件
- 日志文件随存储重定向到临时目录，不污染仓库
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from backend import audit
from backend import main as main_module
from backend import storage

SENTINEL = "sk-audit-test-secret-98765"


@pytest.fixture
def client():
    return TestClient(main_module.app)


@pytest.fixture(autouse=True)
def _clean_audit_log():
    p = audit._log_path()
    if p.exists():
        p.unlink()
    yield
    if p.exists():
        p.unlink()


@pytest.fixture
def mock_job(client):
    r = client.post("/api/eval/mock")
    assert r.status_code == 200
    return r.json()["job_id"]


def _submit_review(client, job_id: str) -> None:
    task_set = main_module._jobs[job_id]["task_set"]
    scores = [{"id": t["id"], "round": 1, "answer_x": 5, "answer_y": 5} for t in task_set["tasks"]]
    r = client.post(f"/api/eval/{job_id}/review", json={"scores": scores})
    assert r.status_code == 200, r.text


# ---- 单元：白名单与脱敏 ----

def test_log_path_redirected_with_storage():
    p = audit._log_path()
    assert p.name == "audit.log"
    assert p.parent == storage.BASE_DIR.parent


def test_append_keeps_only_whitelisted_keys():
    audit._append({
        "event": "eval_started", "job_id": "j1", "actor": "local",
        "hack": "被丢弃", "key": "sk-x", "api_key": "sk-y",
    })
    events = audit.read_events()
    assert len(events) == 1
    e = events[0]
    assert set(e.keys()) <= audit.ALLOWED_KEYS
    assert "hack" not in e and "key" not in e and "api_key" not in e
    assert e["event"] == "eval_started" and e["job_id"] == "j1"


def test_append_redacts_nested_sensitive_fields():
    audit._append({"event": "x", "actor": {"model_a": {"key": SENTINEL, "name": "m"}}})
    text = audit._log_path().read_text(encoding="utf-8")
    assert SENTINEL not in text
    e = audit.read_events()[0]
    assert e["actor"] == {"model_a": {"name": "m"}}


def test_append_timestamp_added():
    audit._append({"event": "x", "job_id": "j"})
    assert audit.read_events()[0]["ts"]


def test_append_errors_silently_and_reads_broken_lines():
    audit._append({"event": "ok", "job_id": "j1"})
    p = audit._log_path()
    p.write_text("broken json line\n" + p.read_text(encoding="utf-8"), encoding="utf-8")
    events = audit.read_events()
    assert len(events) == 1
    assert events[0]["event"] == "ok"


# ---- 集成：端点挂钩 ----

def test_mock_eval_and_review_audited(client, mock_job):
    _submit_review(client, mock_job)
    events = {e["event"]: e for e in audit.read_events()}
    assert events["eval_started"]["job_id"] == mock_job
    assert events["eval_started"]["actor"] == "mock"
    assert events["review_submitted"]["job_id"] == mock_job


def test_real_eval_start_audited(monkeypatch, client):
    async def _noop(job_id):
        pass
    monkeypatch.setattr(main_module, "_run_eval", _noop)
    payload = {
        "model_a": {"url": "https://8.8.8.8/v1", "key": "k", "name": "A", "temperature": 0.7, "max_tokens": 100},
        "model_b": {"url": "https://8.8.8.8/v1", "key": "k", "name": "B", "temperature": 0.7, "max_tokens": 100},
    }
    r = client.post("/api/eval/start", json=payload)
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    events = audit.read_events()
    assert any(e["event"] == "eval_started" and e["job_id"] == job_id for e in events)


def test_delete_history_audited(client, mock_job):
    r = client.delete(f"/api/history/{mock_job}")
    assert r.status_code == 200
    events = audit.read_events()
    assert any(e["event"] == "history_deleted" and e["job_id"] == mock_job for e in events)


def test_dataset_upload_and_delete_audited(client):
    ds = {"name": "审计测试集", "description": "d",
          "tasks": [{"id": "T1", "prompt": "1+1?", "expected": "2"}]}
    r = client.post("/api/datasets/upload-json", json={"content": json.dumps(ds, ensure_ascii=False)})
    assert r.status_code == 200, r.text
    r = client.delete("/api/datasets/审计测试集")
    assert r.status_code == 200, r.text
    events = {e["event"]: e for e in audit.read_events()}
    assert events["dataset_uploaded"]["target"] == "审计测试集"
    assert events["dataset_deleted"]["target"] == "审计测试集"


def test_auth_failure_audited(monkeypatch, client):
    monkeypatch.setenv("MODEL_DUEL_TOKEN", "shared-token-123")
    r = client.get("/api/history")
    assert r.status_code == 401
    events = audit.read_events()
    assert any(e["event"] == "auth_failed" and e["path"] == "/api/history" for e in events)


def test_audit_log_never_contains_key_after_full_flow(client, mock_job):
    # 注入哨兵 Key（仅内存），走完整评审流程后日志仍无明文
    main_module._jobs[mock_job]["config"]["model_a"]["key"] = SENTINEL
    main_module._jobs[mock_job]["config"]["model_b"]["key"] = SENTINEL
    _submit_review(client, mock_job)
    assert SENTINEL not in audit._log_path().read_text(encoding="utf-8")
