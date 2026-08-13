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
    # 迭代一：上限提至 10MB（200 题 × 32KB 上下文场景），此处构造超出部分
    big = b'{"name":"big","tasks":[{"prompt":"x","expected":"y"}]}' + b" " * (11 * MB)
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


# ---------------- 数据集字段校验（issue #15 / R2-006） ----------------

def _upload_json(client, inner: str):
    body = json.dumps({"content": inner}).encode()
    return client.post(
        "/api/datasets/upload-json",
        content=body,
        headers={"Content-Type": "application/json"},
    )


def test_upload_json_non_string_name_400(client):
    r = _upload_json(client, '{"name":123,"tasks":[{"prompt":"p","expected":"e"}]}')
    assert r.status_code == 400
    assert "name" in r.json()["detail"]
    assert "500" not in r.text


def test_upload_json_non_string_prompt_400(client):
    r = _upload_json(client, '{"tasks":[{"prompt":123}]}')
    assert r.status_code == 400
    assert "tasks[0].prompt" in r.json()["detail"]


def test_upload_json_duplicate_ids_400_no_persist(client):
    inner = '{"name":"dup","tasks":[{"id":"X","prompt":"p1","expected":"e1"},' \
            '{"id":"X","prompt":"p2","expected":"e2"}]}'
    r = _upload_json(client, inner)
    assert r.status_code == 400
    assert "重复" in r.json()["detail"]
    assert "dup" not in {d["name"] for d in main_module.list_datasets()}


def test_upload_json_explicit_empty_id_400(client):
    r = _upload_json(client, '{"tasks":[{"id":"","prompt":"p","expected":"e"}]}')
    assert r.status_code == 400
    assert "tasks[0].id" in r.json()["detail"]


def test_upload_csv_blank_id_cell_400(client):
    csv_text = "id,prompt,expected\nA,q1,e1\n,q2,e2\n"
    r = client.post("/api/datasets/upload", files={"file": ("blankid.csv", csv_text.encode())})
    assert r.status_code == 400
    assert "tasks[1].id" in r.json()["detail"]


def test_upload_markdown_duplicate_ids_400(client):
    md_text = "### X\n**题目：** p\n**期望：** e\n### X\n**题目：** q\n**期望：** e2\n"
    r = client.post("/api/datasets/upload", files={"file": ("dup.md", md_text.encode())})
    assert r.status_code == 400
    assert "重复" in r.json()["detail"]


def test_upload_filename_stem_too_long_400(client):
    """超长文件名 stem 在名称覆盖时被拒（避免上传成功/启动评测失败不一致）。"""
    body = b'{"name":"ok","tasks":[{"prompt":"x","expected":"y"}]}'
    r = client.post("/api/datasets/upload", files={"file": ("a" * 201 + ".json", body)})
    assert r.status_code == 400
    assert "name" in r.json()["detail"]


def test_upload_whitespace_filename_stem_falls_back(client):
    """纯空白 stem 回退解析器生成名，上传仍成功。"""
    body = b'{"name":"ok","tasks":[{"prompt":"x","expected":"y"}]}'
    r = client.post("/api/datasets/upload", files={"file": ("   .json", body)})
    assert r.status_code == 200
    assert r.json()["name"] == "ok"


# ---------------- 并发调度（迭代七：排队替代 429） ----------------

PUBLIC_URL = "https://8.8.8.8/v1"


def _payload() -> dict:
    return {
        "model_a": {"url": PUBLIC_URL, "key": "k", "name": "A", "temperature": 0.7, "max_tokens": 100},
        "model_b": {"url": PUBLIC_URL, "key": "k", "name": "B", "temperature": 0.7, "max_tokens": 100},
    }


def test_concurrency_over_quota_queues(client):
    """并发超配额不再 429：新任务排队等待调度。"""
    main_module._SCHEDULER.clear()
    main_module._jobs.clear()
    for i in range(2):
        main_module._jobs[f"fake-job-{i}"] = {
            "state": "executing",
            "config": {"model_a": {"url": "https://example.com/v1", "name": "x"}},
        }
    # 调度器运行集同步填满配额（模拟两个任务占用槽位）
    for i in range(2):
        main_module._SCHEDULER.submit(f"fake-job-{i}")
        main_module._SCHEDULER.next_batch()
    r = client.post("/api/eval/start", json=_payload())
    assert r.status_code == 200
    jid = r.json()["job_id"]
    assert main_module._jobs[jid]["state"] == "queued"
    view = main_module._SCHEDULER.queue_view()
    assert view[0]["job_id"] == jid and view[0]["position"] == 1
    main_module._jobs.clear()
    main_module._SCHEDULER.clear()


def test_concurrency_pending_counts(client):
    main_module._SCHEDULER.clear()
    main_module._jobs.clear()
    for i in range(2):
        main_module._jobs[f"fake-pending-{i}"] = {
            "state": "pending",
            "config": {"model_a": {"url": "https://example.com/v1", "name": "x"}},
        }
        main_module._SCHEDULER.submit(f"fake-pending-{i}")
        main_module._SCHEDULER.next_batch()
    r = client.post("/api/eval/start", json=_payload())
    assert r.status_code == 200
    assert main_module._jobs[r.json()["job_id"]]["state"] == "queued"
    main_module._jobs.clear()
    main_module._SCHEDULER.clear()


def test_concurrency_mock_not_counted(client, monkeypatch):
    monkeypatch.setattr(main_module, "_run_eval", _stub_run_eval)
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


def test_concurrency_under_limit_ok(client, monkeypatch):
    monkeypatch.setattr(main_module, "_run_eval", _stub_run_eval)
    main_module._jobs.clear()
    r = client.post("/api/eval/start", json=_payload())
    assert r.status_code == 200
    main_module._jobs.clear()


async def _stub_run_eval(job_id):
    """桩化后台评测任务：200 路径测试只验证并发计数门禁，不发起真实模型调用。

    返回 200 后 main 会用 asyncio.create_task 启动 _run_eval（独立事件循环），
    不清除桩会导致真实连接 8.8.8.8 或任务抛 KeyError（issue #11 复审 R2-002）。
    """
    pass
