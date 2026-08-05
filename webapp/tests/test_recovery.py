# -*- coding: utf-8 -*-
"""服务重启后磁盘态任务可提交评审（issue #5）回归测试。

核心验证：_jobs 清空（模拟重启）后，提交评分不依赖进程内状态，
verdict/review/report 正常落盘且状态为 completed；重复提交被拒绝。
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi.testclient import TestClient

import pytest

from backend import storage
from backend.engine.human_review import make_reveal
from backend.engine.mock import prepare_mock_job
from backend.engine.tasks import build_task_set
from backend.main import app, _jobs
from backend.storage import (
    create_job_id,
    save_answers,
    save_config,
    save_reveal,
    save_task_set,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def _isolate_history(tmp_path: Path, monkeypatch):
    """将历史目录重定向到临时目录，并清理内存 _jobs，防止测试间污染。"""
    monkeypatch.setattr(storage, "BASE_DIR", tmp_path)
    yield
    for jid in list(_jobs):
        _jobs.pop(jid)


def _seed_disk_job(repeat: int = 1, dataset_name: str | None = None) -> tuple[str, dict]:
    """构造磁盘态任务（模拟已执行完成、服务重启后的磁盘状态）。"""
    job_id = create_job_id()
    config = {
        "model_a": {"name": "模型A", "url": "https://api.example.com/v1", "key": "secret-a"},
        "model_b": {"name": "模型B", "url": "https://api.example.com/v1", "key": "secret-b"},
        "dims": None,
        "seed": 7,
        "dataset_name": dataset_name,
        "repeat_n": repeat,
        "code_verify_mode": "off",
    }
    save_config(job_id, config)

    task_set = build_task_set(seed=7)
    save_task_set(job_id, task_set)

    answers = {
        "model": "模型A",
        "api": {"name": "模型A", "url": "https://example.com"},
        "note": "test",
        "answers": [
            {"id": t["id"], "raw_answer": f"answer-{t['id']}", "api_info": {"status": "ok"}}
            for t in task_set["tasks"]
        ],
    }

    for r in range(1, repeat + 1):
        save_answers(job_id, f"a-r{r}", answers)
        save_answers(job_id, f"b-r{r}", answers)
    save_answers(job_id, "a", answers)
    save_answers(job_id, "b", answers)
    save_reveal(job_id, make_reveal(repeat))
    return job_id, task_set


def _scores(task_set: dict, repeat: int) -> list[dict]:
    scores = []
    for r in range(1, repeat + 1):
        for t in task_set["tasks"]:
            scores.append({"id": t["id"], "round": r, "answer_x": 8.0, "answer_y": 2.0, "note": ""})
    return scores


def _assert_completed(job_id: str):
    d = storage.BASE_DIR / job_id
    assert (d / "verdict.json").exists()
    assert (d / "review.json").exists()
    assert (d / "report.json").exists()
    assert storage._job_state(d) == "completed"


def test_review_view_after_restart():
    """验收3：重启后评审页可恢复显示题目与答案。"""
    job_id, task_set = _seed_disk_job()
    resp = client.get(f"/api/eval/{job_id}/review")
    assert resp.status_code == 200
    body = resp.json()
    assert body["repeat_n"] == 1
    assert body["total_questions"] == len(task_set["tasks"])
    assert len(body["rounds"]) == 1
    assert len(body["rounds"][0]["items"]) == len(task_set["tasks"])


def test_submit_after_restart_single_round():
    """验收1-4：重启后提交完整评分，verdict/review/report 落盘，状态 completed。"""
    job_id, task_set = _seed_disk_job()
    resp = client.post(f"/api/eval/{job_id}/review", json={"scores": _scores(task_set, 1)})
    assert resp.status_code == 200, resp.text
    _assert_completed(job_id)
    report = json.loads((storage.BASE_DIR / job_id / "report.json").read_text(encoding="utf-8"))
    assert report["report"] is not None
    assert report["verdict"] == json.loads(
        (storage.BASE_DIR / job_id / "verdict.json").read_text(encoding="utf-8"))


def test_submit_after_restart_multi_round():
    """验收6-多轮：repeat_n=2 时的磁盘恢复提交。"""
    job_id, task_set = _seed_disk_job(repeat=2)
    resp = client.post(f"/api/eval/{job_id}/review", json={"scores": _scores(task_set, 2)})
    assert resp.status_code == 200, resp.text
    _assert_completed(job_id)
    verdict = json.loads((storage.BASE_DIR / job_id / "verdict.json").read_text(encoding="utf-8"))
    assert verdict["meta"]["repeat_n"] == 2


def test_submit_after_restart_custom_dataset():
    """验收6-自定义数据集。"""
    job_id, task_set = _seed_disk_job(dataset_name="my-dataset")
    resp = client.post(f"/api/eval/{job_id}/review", json={"scores": _scores(task_set, 1)})
    assert resp.status_code == 200, resp.text
    _assert_completed(job_id)


def test_submit_after_restart_mock():
    """验收6-模拟任务（prepare_mock_job 落盘后重启提交）。"""
    data = prepare_mock_job(seed=42)
    job_id = data["job_id"]
    task_set = data["task_set"]
    resp = client.post(f"/api/eval/{job_id}/review", json={"scores": _scores(task_set, 1)})
    assert resp.status_code == 200, resp.text
    _assert_completed(job_id)


def test_report_identical_memory_vs_disk():
    """验收5：同一数据下，内存路径与磁盘恢复路径生成的报告内容一致。"""
    job_id, task_set = _seed_disk_job()
    cfg = json.loads((storage.BASE_DIR / job_id / "config.json").read_text(encoding="utf-8"))
    answers_a = json.loads((storage.BASE_DIR / job_id / "answers-a.json").read_text(encoding="utf-8"))
    answers_b = json.loads((storage.BASE_DIR / job_id / "answers-b.json").read_text(encoding="utf-8"))
    reveal = json.loads((storage.BASE_DIR / job_id / "reveal.json").read_text(encoding="utf-8"))
    _jobs[job_id] = {
        "state": "reviewing", "progress": "0/0",
        "task_set": task_set, "config": cfg,
        "answers_a": answers_a, "answers_b": answers_b,
        "verdict": None, "rounds_answers": [{"a": answers_a, "b": answers_b}],
        "reveal": reveal, "created_at": "now", "sse_queue": asyncio.Queue(), "repeat_n": 1,
    }
    resp_mem = client.post(f"/api/eval/{job_id}/review", json={"scores": _scores(task_set, 1)})
    assert resp_mem.status_code == 200, resp_mem.text

    job_disk, _ = _seed_disk_job()
    resp_disk = client.post(f"/api/eval/{job_disk}/review", json={"scores": _scores(task_set, 1)})
    assert resp_disk.status_code == 200, resp_disk.text

    report_mem = json.loads((storage.BASE_DIR / job_id / "report.json").read_text(encoding="utf-8"))
    report_disk = json.loads((storage.BASE_DIR / job_disk / "report.json").read_text(encoding="utf-8"))
    assert report_mem["report"] == report_disk["report"]


def test_duplicate_submit_rejected():
    """验收7：重复提交被明确拒绝（409）。"""
    job_id, task_set = _seed_disk_job()
    payload = {"scores": _scores(task_set, 1)}
    first = client.post(f"/api/eval/{job_id}/review", json=payload)
    assert first.status_code == 200, first.text
    second = client.post(f"/api/eval/{job_id}/review", json=payload)
    assert second.status_code == 409
    report_files = list((storage.BASE_DIR / job_id).glob("report.json"))
    assert len(report_files) == 1