# -*- coding: utf-8 -*-
"""逐题 reveal（迭代三，仅 agent）测试：make_task_reveal 构造、run_judge
逐题选池注入、human_review._stable_scores 按题归一化、旧 rounds-only 回退。"""
from __future__ import annotations

import asyncio
import json

import httpx

from backend.engine import judge as judge_module
from backend.engine.human_review import _stable_scores, build_final_verdict
from backend.engine.judge import (
    _normalize_task_reveal,
    health_check,
    make_task_reveal,
    run_judge,
)

TASK1 = {"id": "T1", "dimension": "数学能力", "prompt": "1+1=?",
         "rubric_note": "答案正确", "test_cases": []}
TASK2 = {"id": "T2", "dimension": "数学能力", "prompt": "2+2=?",
         "rubric_note": "答案正确", "test_cases": []}


# ---- make_task_reveal ----

def test_make_task_reveal_per_task_independent():
    r = make_task_reveal(["T1", "T2", "T3"], seed=42)
    assert len(r["per_task"]) == 3
    assert all("task_id" in item and item["answer_x"] in ("a", "b") for item in r["per_task"])
    for item in r["per_task"]:
        assert item["answer_y"] == ("b" if item["answer_x"] == "a" else "a")
    # rounds 轮级兜底 = 首题标签
    assert r["rounds"][0]["answer_x"] == r["per_task"][0]["answer_x"]


def test_make_task_reveal_deterministic_seed():
    a = make_task_reveal(["T1", "T2"], seed=7)
    b = make_task_reveal(["T1", "T2"], seed=7)
    assert a == b


def test_make_task_reveal_empty():
    r = make_task_reveal([])
    assert r["per_task"] == []
    assert r["rounds"][0]["answer_x"] == "a"


# ---- _normalize_task_reveal ----

def test_normalize_task_reveal():
    revealed = {"rounds": [{"answer_x": "a", "answer_y": "b"}],
                "per_task": [{"task_id": "T1", "answer_x": "b", "answer_y": "a"},
                             {"task_id": "T2", "answer_x": "a", "answer_y": "b"}]}
    m = _normalize_task_reveal(revealed)
    assert m == {"T1": "b", "T2": "a"}


def test_normalize_task_reveal_none_on_missing():
    assert _normalize_task_reveal(None) is None
    assert _normalize_task_reveal({"rounds": [{"answer_x": "a"}]}) is None
    assert _normalize_task_reveal({"per_task": []}) is None


def test_normalize_task_reveal_file_labels():
    revealed = {"per_task": [{"task_id": "T1", "answer_x": "answers-b.json"}]}
    assert _normalize_task_reveal(revealed) == {"T1": "b"}


# ---- run_judge 逐题注入 ----

def _judge_stub(verdicts_by_call: list[dict]):
    calls = {"n": 0}

    def factory(**kwargs):
        async def handler(request):
            calls["n"] += 1
            body = {"choices": [{"message": {"content": "___"} }]}
            return httpx.Response(200, json=body)

        # 每次调用返回不同 verdict：按调用次数轮换
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        idx = min(calls["n"] - 1, len(verdicts_by_call) - 1)
        judge_module._parse_verdict_calls = getattr(judge_module, "_parse_verdict_calls", [])
        return client

    return factory


async def _deal_response(handler, payload):
    return httpx.Response(200, json={"choices": [{"message": {"content": payload}}]})


def _run_judge_with_reveal(per_task: list[dict], answers, tasks):
    """用 MockTransport 驱动 run_judge，逐题按调用序号返回预置 verdict 文本。"""
    verdict_texts = [None, None]

    async def handler(request):
        # 按请求体内容无法区分题号；改用计数在闭包外驱动
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    captured = {}

    def factory(**kwargs):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return client

    async def _run():
        return await run_judge(
            {"meta": {"total": len(tasks)}, "tasks": tasks},
            answers["A"], answers["B"],
            {"url": "https://8.8.8.8/v1", "key": "k", "name": "J"},
            revealed={"rounds": [{"answer_x": "a", "answer_y": "b"}],
                      "per_task": per_task},
        )

    return _run()  # 由调用方 await


def test_run_judge_with_per_task_reveal_uses_correct_pool(monkeypatch):
    """T1 的 X=b（用 B 的答案）、T2 的 X=a（用 A 的答案）——评审 prompt 内容可区分。"""
    seen = []

    async def handler(request):
        body = json.loads(request.content)
        content = body["messages"][0]["content"]
        seen.append("B答:T1" if "B答:T1" in content else ("A答:T2" if "A答:T2" in content else "other"))
        # 根据题目 id 返回对应 score（winner 按 X 答得更好）
        if "T1" in content:
            payload = '{"id":"T1","dimension":"数学能力","answer_x":8,"answer_y":2,"winner":"answer_x","basis":"ok"}'
        else:
            payload = '{"id":"T2","dimension":"数学能力","answer_x":9,"answer_y":1,"winner":"answer_x","basis":"ok"}'
        return httpx.Response(200, json={"choices": [{"message": {"content": payload}}]})

    def factory(**kwargs):
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(judge_module, "build_upstream_client", factory)

    answers = {
        "A": {"model": "模型A", "answers": [
            {"id": "T1", "raw_answer": "A答:T1", "api_info": {}},
            {"id": "T2", "raw_answer": "A答:T2", "api_info": {}},
        ]},
        "B": {"model": "模型B", "answers": [
            {"id": "T1", "raw_answer": "B答:T1", "api_info": {}},
            {"id": "T2", "raw_answer": "B答:T2", "api_info": {}},
        ]},
    }

    async def _run():
        return await run_judge(
            {"meta": {"total": 2}, "tasks": [TASK1, TASK2]},
            answers["A"], answers["B"],
            {"url": "https://8.8.8.8/v1", "key": "k", "name": "J"},
            revealed={"rounds": [{"answer_x": "a", "answer_y": "b"}],
                      "per_task": [{"task_id": "T1", "answer_x": "b", "answer_y": "a"},
                                   {"task_id": "T2", "answer_x": "a", "answer_y": "b"}]},
        )

    r = asyncio.run(_run())
    assert "B答:T1" in seen and "A答:T2" in seen
    assert seen[0] == "B答:T1" and seen[1] == "A答:T2"


# ---- human_review 按题归一化 ----

def _round_verdict(x_files: dict[str, str]) -> dict:
    """构造一轮 verdict：scores 按题不同身份（x_files: task_id -> "a"|"b"）。"""
    scores = []
    for tid, xf in x_files.items():
        if xf == "a":
            scores.append({"id": tid, "dimension": "数学能力",
                           "answer_x": 8.0, "answer_y": 2.0, "_invalid": False})
        else:
            scores.append({"id": tid, "dimension": "数学能力",
                           "answer_x": 2.0, "answer_y": 8.0, "_invalid": False})
    return {
        "meta": {"total": len(scores), "valid": len(scores), "invalid": 0,
                 "excluded_ids": [], "excluded_dimensions": []},
        "scores": scores,
        "per_dimension": {}, "totals": {"answer_x": 0, "answer_y": 0},
        "revealed": {"answer_x": "模型A", "answer_y": "模型B",
                     "answer_x_file": "a", "answer_y_file": "b"},
        "conclusion": "", "winner_model": "tie",
    }


def test_stable_scores_per_task_normalization():
    per_task = {"T1": "b", "T2": "a"}  # T1 的 X=b，T2 的 X=a
    rows = _stable_scores(_round_verdict({"T1": "b", "T2": "a"}), per_task)
    by_id = {r["id"]: r for r in rows}
    # T1：answer_x=2 属于 B → model_a=B 分=8；T2：answer_x=8 属于 A → model_a=8
    assert by_id["T1"]["model_a"] == 8.0 and by_id["T1"]["model_b"] == 2.0
    assert by_id["T2"]["model_a"] == 8.0 and by_id["T2"]["model_b"] == 2.0


def test_stable_scores_rounds_only_fallback():
    # 无 per_task → 回退轮级 answer_x_file="a"
    rows = _stable_scores(_round_verdict({"T1": "a", "T2": "a"}), None)
    by_id = {r["id"]: r for r in rows}
    assert by_id["T1"]["model_a"] == 8.0 and by_id["T2"]["model_a"] == 8.0


def test_build_final_verdict_aggregates_per_task_multi_round():
    per_task = {"T1": "b", "T2": "a"}
    rv1 = _round_verdict({"T1": "b", "T2": "a"})
    rv2 = _round_verdict({"T1": "b", "T2": "a"})
    v = build_final_verdict([rv1, rv2], 2, per_task_reveal=per_task)
    by_id = {s["id"]: s for s in v["scores"]}
    assert by_id["T1"]["model_a"] == 8.0
    assert by_id["T2"]["model_a"] == 8.0
    assert v["meta"]["repeat_n"] == 2


def test_build_final_verdict_old_call_unchanged():
    rv1 = _round_verdict({"T1": "a", "T2": "a"})
    v = build_final_verdict([rv1], 1)
    by_id = {s["id"]: s for s in v["scores"]}
    assert by_id["T1"]["answer_x"] == 8.0