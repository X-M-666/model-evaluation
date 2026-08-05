# -*- coding: utf-8 -*-
"""资源限制测试（issue #8）：并发上限 / 上传大小 / 数据集题数。"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from backend import main as main_module

MB = 1024 * 1024


@pytest.fixture
def client():
    return TestClient(main_module.app)


# ---------------- 上传大小 ----------------

def test_upload_too_large_rejected(client):
    big = b'{"name":"big","tasks":[{"prompt":"x","expected":"y"}]}' + b" " * (5 * MB)
    r = client.post("/api/datasets/upload", files={"file": ("big.json", big)})
    assert r.status_code == 400
    assert "过大" in r.json()["detail"]


def test_upload_just_under_limit_ok(client):
    r = client.post(
        "/api/datasets/upload",
        files={"file": ("ok.json", b'{"name":"ok","tasks":[{"prompt":"x","expected":"y"}]}')},
    )
    assert r.status_code == 200


def test_upload_json_too_large_rejected(client):
    big = '{"name":"big","tasks":[{"prompt":"x","expected":"y"}]}' + " " * (2 * MB)
    r = client.post("/api/datasets/upload-json", json={"content": big})
    assert r.status_code == 400
    assert "过大" in r.json()["detail"]


def test_upload_very_large_file_rejected(client):
    # 远超上限的 body：验证截断读取路径（读满 MAX+1 即拒，不整体读入）
    big = b'{"name":"big","tasks":[{"prompt":"x","expected":"y"}]}' + b" " * (30 * MB)
    r = client.post("/api/datasets/upload", files={"file": ("big.json", big)})
    assert r.status_code == 400
    assert "过大" in r.json()["detail"]


def test_upload_json_raw_body_rejected(client):
    # 带 Content-Length 的大 body：快速预检路径直接 400
    big = b'{"content":"' + b"x" * (3 * MB) + b'"}'
    r = client.post(
        "/api/datasets/upload-json",
        content=big,
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 400
    assert "过大" in r.json()["detail"]


def test_upload_json_chunked_body_rejected(client):
    # 无 Content-Length 的分块传输大 body：流式累读截断路径 400
    def chunks():
        yield b'{"content":"' + b"x" * MB
        yield b"x" * MB
        yield b"x" * MB + b'"}'

    r = client.post(
        "/api/datasets/upload-json",
        content=chunks(),
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 400
    assert "过大" in r.json()["detail"]


def test_upload_json_invalid_body_rejected(client):
    # 非法 JSON 与非 dict 结构：400 而非 500
    for bad in ("not json at all", "[1,2,3]", "42", '"str"'):
        r = client.post(
            "/api/datasets/upload-json",
            content=bad,
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 400, repr(bad)
    r = client.post("/api/datasets/upload-json", json={})
    assert r.status_code == 400


def test_upload_json_normal_ok(client):
    inner = json.dumps({"name": "normal", "tasks": [{"prompt": "p1", "expected": "e1"}]}, ensure_ascii=False)
    body = json.dumps({"content": inner}).encode()
    r = client.post(
        "/api/datasets/upload-json",
        content=body,
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 200, r.text


# ---------------- 题数上限 ----------------

def test_dataset_task_count_over_limit_rejected(client):
    header = "id,dimension,prompt,expected,rubric_note,difficulty"
    tasks = header + "\n" + "\n".join(f"T{i},知识,问题{i}？,答案{i},,进阶" for i in range(201))
    r = client.post("/api/datasets/upload", files={"file": ("many.csv", tasks.encode())})
    assert r.status_code == 400
    assert "200" in r.json()["detail"]


def test_dataset_task_count_at_limit_ok(client):
    header = "id,dimension,prompt,expected,rubric_note,difficulty"
    tasks = header + "\n" + "\n".join(f"T{i},知识,问题{i}？,答案{i},,进阶" for i in range(200))
    r = client.post("/api/datasets/upload", files={"file": ("many.csv", tasks.encode())})
    assert r.status_code == 200


# ---------------- 并发上限 ----------------

PUBLIC_URL = "https://8.8.8.8/v1"


def _payload() -> dict:
    return {
        "model_a": {"url": PUBLIC_URL, "key": "k", "name": "A", "temperature": 0.7, "max_tokens": 100},
        "model_b": {"url": PUBLIC_URL, "key": "k", "name": "B", "temperature": 0.7, "max_tokens": 100},
    }


def test_concurrency_limit_reached_429(client):
    main_module._jobs.clear()
    for i in range(2):
        main_module._jobs[f"fake-job-{i}"] = {
            "state": "executing",
            "config": {"model_a": {"url": "https://example.com/v1", "name": "x"}},
        }
    r = client.post("/api/eval/start", json=_payload())
    assert r.status_code == 429
    assert "2" in r.json()["detail"]


def test_concurrency_pending_counts(client):
    main_module._jobs.clear()
    for i in range(2):
        main_module._jobs[f"fake-pending-{i}"] = {
            "state": "pending",
            "config": {"model_a": {"url": "https://example.com/v1", "name": "x"}},
        }
    r = client.post("/api/eval/start", json=_payload())
    assert r.status_code == 429


def test_concurrency_mock_not_counted(client):
    main_module._jobs.clear()
    for i in range(2):
        main_module._jobs[f"mock-job-{i}"] = {
            "state": "executing",
            "config": {"model_a": {"url": "mock://a", "name": "x"}},
        }
    r = client.post("/api/eval/start", json=_payload())
    assert r.status_code == 200
    assert "fake-job" not in r.json()["job_id"]
    main_module._jobs.clear()


def test_concurrency_under_limit_ok(client):
    main_module._jobs.clear()
    r = client.post("/api/eval/start", json=_payload())
    assert r.status_code == 200
    main_module._jobs.clear()
