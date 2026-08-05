# -*- coding: utf-8 -*-
"""报告页答案原文按 reveal 映射（issue #6）回归测试。

核心验证：后端 /api/eval/{job_id}/report 提供按 reveal 归一化的
answers_x/answers_y，前端只消费归一化结果，避免 X=B/Y=A 时归因反转。
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend import storage
from backend.engine.report_builder import build_report, reveal_answers
from backend.engine.tasks import build_task_set
from backend.main import app, _jobs
from backend.storage import (
    create_job_id,
    save_answers,
    save_config,
    save_task_set,
)

client = TestClient(app)

TEXT_A = "这是模型 A 的回答"
TEXT_B = "这是模型 B 的回答"


@pytest.fixture(autouse=True)
def _isolate_history(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(storage, "BASE_DIR", tmp_path)
    yield
    for jid in list(_jobs):
        _jobs.pop(jid)


def _seed_job(x_file: str, y_file: str) -> tuple[str, dict]:
    """构造已完成（含 verdict/report）的磁盘态任务，reveal 指定 X/Y 答卷文件。"""
    job_id = create_job_id()
    config = {
        "model_a": {"name": "模型A", "url": "https://example.com", "key": "k"},
        "model_b": {"name": "模型B", "url": "https://example.com", "key": "k"},
        "dims": None, "seed": 7, "dataset_name": None,
        "repeat_n": 1, "code_verify_mode": "off",
    }
    save_config(job_id, config)
    task_set = build_task_set(seed=7)
    save_task_set(job_id, task_set)

    answers_a = {
        "model": "模型A", "api": {"name": "模型A", "url": "https://example.com"},
        "note": "t", "answers": [
            {"id": t["id"], "raw_answer": f"{TEXT_A} [{t['id']}]", "api_info": {"status": "ok"}}
            for t in task_set["tasks"]
        ],
    }
    answers_b = {
        "model": "模型B", "api": {"name": "模型B", "url": "https://example.com"},
        "note": "t", "answers": [
            {"id": t["id"], "raw_answer": f"{TEXT_B} [{t['id']}]", "api_info": {"status": "ok"}}
            for t in task_set["tasks"]
        ],
    }
    save_answers(job_id, "a", answers_a)
    save_answers(job_id, "b", answers_b)

    verdict = {
        "revealed": {"answer_x_file": x_file, "answer_y_file": y_file,
                     "answer_x": "模型B" if x_file == "b" else "模型A",
                     "answer_y": "模型A" if y_file == "a" else "模型B"},
        "scores": [{"id": t["id"], "answer_x": 8.0, "answer_y": 2.0} for t in task_set["tasks"]],
        "totals": {"answer_x": 8.0 * len(task_set["tasks"]), "answer_y": 2.0 * len(task_set["tasks"])},
        "meta": {"repeat_n": 1},
    }
    (storage.BASE_DIR / job_id / "verdict.json").write_text(
        __import__("json").dumps(verdict, ensure_ascii=False), encoding="utf-8")
    (storage.BASE_DIR / job_id / "report.json").write_text(
        __import__("json").dumps({
            "config": config, "tasks": task_set,
            "answers_a": answers_a, "answers_b": answers_b,
            "verdict": verdict,
            "report": build_report(config, task_set, answers_a, answers_b, verdict),
        }, ensure_ascii=False), encoding="utf-8")
    return job_id, task_set


def _first_raw(answers: dict | None) -> str:
    return (answers or {}).get("answers", [{}])[0].get("raw_answer", "")


def test_reveal_answers_both_directions():
    """验收1/2/3：两种 reveal 方向下归一化结果正确。"""
    verdict_ab = {"revealed": {"answer_x_file": "a", "answer_y_file": "b"}}
    x, y = reveal_answers({"model": "A"}, {"model": "B"}, verdict_ab)
    assert x["model"] == "A" and y["model"] == "B"

    verdict_ba = {"revealed": {"answer_x_file": "b", "answer_y_file": "a"}}
    x, y = reveal_answers({"model": "A"}, {"model": "B"}, verdict_ba)
    assert x["model"] == "B" and y["model"] == "A"


def test_report_endpoint_reveal_swap():
    """验收2/3/5：X=B/Y=A 时报告响应 answers_x 为模型B原文、answers_y 为模型A原文，
    answers_a/answers_b 原样保留（导出 JSON 无歧义）。"""
    job_id, _ = _seed_job("b", "a")
    resp = client.get(f"/api/eval/{job_id}/report")
    assert resp.status_code == 200
    body = resp.json()
    assert TEXT_B in _first_raw(body["answers_x"])
    assert TEXT_A in _first_raw(body["answers_y"])
    assert TEXT_A in _first_raw(body["answers_a"])
    assert TEXT_B in _first_raw(body["answers_b"])
    assert body["report"]["summary"]["x_model"] == "模型B"
    assert body["report"]["summary"]["y_model"] == "模型A"


def test_chart_and_text_same_mapping():
    """验收4：归一化原文与报告标签/图表同源于同一 revealed 映射。"""
    job_id, _ = _seed_job("b", "a")
    resp = client.get(f"/api/eval/{job_id}/report")
    body = resp.json()
    s = body["report"]["summary"]
    assert s["x_model"] == "模型B"
    assert s["y_model"] == "模型A"
    x_entry = body["answers_x"]["answers"][0]
    y_entry = body["answers_y"]["answers"][0]
    assert x_entry["raw_answer"].startswith(TEXT_B)
    assert y_entry["raw_answer"].startswith(TEXT_A)
    first_tid = body["tasks"]["tasks"][0]["id"]
    assert x_entry["id"] == first_tid and y_entry["id"] == first_tid


def test_default_mapping_no_revealed():
    """验收1边界：verdict 无 revealed 时默认 X=A/Y=B，不崩溃。"""
    x, y = reveal_answers({"model": "A"}, {"model": "B"}, {"scores": []})
    assert x["model"] == "A" and y["model"] == "B"
    x, y = reveal_answers({"model": "A"}, {"model": "B"}, None)
    assert x["model"] == "A" and y["model"] == "B"
