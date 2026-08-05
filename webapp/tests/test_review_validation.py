# -*- coding: utf-8 -*-
"""评审提交完整性/唯一性校验测试（issue #9）。

验收覆盖：
1. 完整单轮/多轮提交成功
2. 缺失题目 4xx，任务仍 reviewing
3. 同轮同题重复 4xx
4. 未知题号 4xx
5. 轮次 0 / 越界 repeat_n 4xx
6. 整轮缺失 4xx
7. 失败请求不产生/不修改 verdict、review、report 文件
8. 合法重复请求按冲突语义返回 409 且不改动既有文件
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend import storage
from backend.engine.human_review import make_reveal
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
    monkeypatch.setattr(storage, "BASE_DIR", tmp_path)
    yield
    for jid in list(_jobs):
        _jobs.pop(jid)


def _seed_job(repeat: int = 1) -> tuple[str, dict]:
    """构造磁盘态评审任务（config/tasks/answers/reveal 已落盘，等待人工评审）。"""
    job_id = create_job_id()
    save_config(job_id, {
        "model_a": {"name": "模型A", "url": "https://api.example.com/v1", "key": "k"},
        "model_b": {"name": "模型B", "url": "https://api.example.com/v1", "key": "k"},
        "repeat_n": repeat, "code_verify_mode": "off",
    })
    task_set = build_task_set(seed=7)
    save_task_set(job_id, task_set)
    answers = {
        "model": "模型A",
        "answers": [
            {"id": t["id"], "raw_answer": f"a-{t['id']}", "api_info": {"status": "ok"}}
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


def _full_scores(task_set: dict, repeat: int = 1) -> list[dict]:
    scores = []
    for r in range(1, repeat + 1):
        for t in task_set["tasks"]:
            scores.append({"id": t["id"], "round": r, "answer_x": 8.0, "answer_y": 2.0, "note": ""})
    return scores


def _submit(job_id: str, scores: list[dict]):
    return client.post(f"/api/eval/{job_id}/review", json={"scores": scores})


def _assert_no_final_files(job_id: str):
    d = storage.BASE_DIR / job_id
    assert not (d / "verdict.json").exists()
    assert not (d / "review.json").exists()
    assert not (d / "report.json").exists()


# ---------------- 验收 1：完整提交成功 ----------------

def test_full_submit_single_round_ok():
    job_id, task_set = _seed_job()
    r = _submit(job_id, _full_scores(task_set))
    assert r.status_code == 200, r.text
    assert storage._job_state(storage.BASE_DIR / job_id) == "completed"
    d = storage.BASE_DIR / job_id
    assert (d / "verdict.json").exists()
    assert (d / "review.json").exists()
    assert (d / "report.json").exists()


def test_full_submit_multi_round_ok():
    job_id, task_set = _seed_job(repeat=2)
    r = _submit(job_id, _full_scores(task_set, repeat=2))
    assert r.status_code == 200, r.text
    verdict = (storage.BASE_DIR / job_id / "verdict.json").read_text(encoding="utf-8")
    assert '"repeat_n": 2' in verdict


# ---------------- 验收 2-6：各类不完整/冲突被拒绝 ----------------

def test_missing_task_rejected_and_still_reviewing():
    job_id, task_set = _seed_job()
    scores = _full_scores(task_set)[1:]  # 缺失第一题
    r = _submit(job_id, scores)
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert any(e["type"] == "missing_task" and task_set["tasks"][0]["id"] in e["ids"] for e in detail["errors"])
    assert storage._job_state(storage.BASE_DIR / job_id) == "reviewing"
    _assert_no_final_files(job_id)


def test_duplicate_task_rejected():
    job_id, task_set = _seed_job()
    scores = _full_scores(task_set)
    tid = task_set["tasks"][0]["id"]
    scores.append({"id": tid, "round": 1, "answer_x": 5, "answer_y": 5, "note": ""})
    r = _submit(job_id, scores)
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert {"type": "duplicate_task", "round": 1, "id": tid} in detail["errors"]
    _assert_no_final_files(job_id)


def test_unknown_task_rejected():
    job_id, task_set = _seed_job()
    scores = _full_scores(task_set)
    scores[0]["id"] = "nonexistent-task"
    r = _submit(job_id, scores)
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert any(e["type"] == "unknown_task" and e["id"] == "nonexistent-task" for e in detail["errors"])
    _assert_no_final_files(job_id)


def test_round_out_of_range_rejected():
    job_id, task_set = _seed_job()
    # round=0：被 schema 层（ge=1）拦截 → 422（同为 4xx，双层防护）
    scores = _full_scores(task_set)
    scores[0]["round"] = 0
    assert _submit(job_id, scores).status_code == 422
    # round=2 > repeat_n=1：schema 无法表达上界，由服务层验证 → 400
    scores = _full_scores(task_set)
    scores[0]["round"] = 2
    r = _submit(job_id, scores)
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert {"type": "round_out_of_range", "round": 2} in detail["errors"]
    _assert_no_final_files(job_id)


def test_missing_round_rejected():
    job_id, task_set = _seed_job(repeat=2)
    r = _submit(job_id, _full_scores(task_set, repeat=1))  # 只交第 1 轮
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert any(e["type"] == "missing_round" and e["round"] == 2 for e in detail["errors"])
    _assert_no_final_files(job_id)


def test_empty_scores_rejected():
    job_id, _ = _seed_job()
    r = _submit(job_id, [])
    assert r.status_code == 400
    _assert_no_final_files(job_id)


# ---------------- 验收 8：重复提交冲突语义 ----------------

def test_duplicate_submit_conflict_keeps_files_unchanged():
    job_id, task_set = _seed_job()
    scores = _full_scores(task_set)
    assert _submit(job_id, scores).status_code == 200
    report_after_first = (storage.BASE_DIR / job_id / "report.json").read_bytes()
    verdict_after_first = (storage.BASE_DIR / job_id / "verdict.json").read_bytes()

    r = _submit(job_id, scores)
    assert r.status_code == 409
    assert (storage.BASE_DIR / job_id / "report.json").read_bytes() == report_after_first
    assert (storage.BASE_DIR / job_id / "verdict.json").read_bytes() == verdict_after_first
