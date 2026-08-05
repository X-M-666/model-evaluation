# -*- coding: utf-8 -*-
"""双盲评审官单元测试（issue #11 残余风险 R3）。

覆盖 verdict 解析、代码验真格式化、双盲 prompt 构建（不泄露模型名）、
评审重试与失败兜底的核心不变量。网络通过 httpx.MockTransport 完全 mock。
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from backend.engine import judge as judge_module
from backend.engine.judge import (
    _build_blind_prompt,
    _build_conclusion,
    _fmt_code_verify,
    _parse_verdict,
    judge_task,
    run_judge,
)

TASK = {
    "id": "T1", "dimension": "知识能力",
    "prompt": "1+1?", "rubric_note": "满分10分",
    "test_cases": [{"input": "1+1", "expected": "2"}],
}
ANSWER_X = {"id": "T1", "raw_answer": "答案是2", "code_verify": {"status": "run", "passed": 5, "total": 5}}
ANSWER_Y = {"id": "T1", "raw_answer": "答案是3", "code_verify": {"status": "run", "passed": 4, "total": 5}}
JUDGE_CFG = {"name": "judge", "url": "https://j.example.com/v1", "key": "k"}
VALID_V = '{"id":"T1","dimension":"知识能力","answer_x":8,"answer_y":6,"winner":"answer_x","basis":"详","arbiter_note":""}'


# ---- _parse_verdict ----

def test_parse_verdict_plain_json():
    v = _parse_verdict('{"id":"T1","answer_x":8,"answer_y":6,"winner":"answer_x","basis":"x好"}')
    assert v["answer_x"] == 8 and v["winner"] == "answer_x"


def test_parse_verdict_with_json_fence():
    v = _parse_verdict('```json\n{"id":"T1","winner":"tie"}\n```')
    assert v["winner"] == "tie"


def test_parse_verdict_text_prefix_then_json():
    v = _parse_verdict('分析：以下是 verdict。\n{"id":"T1","winner":"answer_y"}')
    assert v["winner"] == "answer_y"


def test_parse_verdict_broken_json_returns_none():
    assert _parse_verdict("{not valid json}") is None


def test_parse_verdict_empty_returns_none():
    assert _parse_verdict("") is None
    assert _parse_verdict(None) is None


def test_parse_verdict_picks_first_object_with_winner():
    v = _parse_verdict('{"a":1}{"id":"t","winner":"tie"}')
    assert v["winner"] == "tie"


# ---- _fmt_code_verify ----

def test_fmt_code_verify_run():
    assert _fmt_code_verify({"status": "run", "passed": 3, "total": 5}) == "3/5"


def test_fmt_code_verify_error():
    assert _fmt_code_verify({"status": "error", "reason": "超时"}) == (
        "执行异常（可能已部分执行）：超时"
    )


def test_fmt_code_verify_disabled():
    assert "已禁用" in _fmt_code_verify({})
    assert "已禁用" in _fmt_code_verify(None)


def test_fmt_code_verify_missing_status_defaults():
    assert "已禁用" in _fmt_code_verify({"passed": 5, "total": 5})


# ---- _build_blind_prompt ----

def test_blind_prompt_uses_answer_x_y_labels():
    p = _build_blind_prompt(TASK, ANSWER_X, ANSWER_Y)
    assert "答案X" in p and "答案Y" in p


def test_blind_prompt_exposes_no_real_model_name():
    ax = {"raw_answer": "只要答案内容", "code_verify": {}}
    ay = {"raw_answer": "另一份答案", "code_verify": {}}
    p = _build_blind_prompt(TASK, ax, ay)
    assert "模型A" not in p and "模型B" not in p and "gpt" not in p.lower()


def test_blind_prompt_includes_test_cases():
    p = _build_blind_prompt(TASK, ANSWER_X, ANSWER_Y)
    assert "用例1" in p and "1+1" in p and "2" in p


def test_blind_prompt_no_test_cases_section_when_empty():
    t = {**TASK, "test_cases": []}
    p = _build_blind_prompt(t, ANSWER_X, ANSWER_Y)
    assert "测试用例参考" not in p


def test_blind_prompt_raw_answer_missing_placeholder():
    ax = {}  # 无 raw_answer 键 → 触发兜底 "(无回答)"
    p = _build_blind_prompt(TASK, ax, ANSWER_Y)
    assert "(无回答)" in p


def test_blind_prompt_code_verify_missing_shows_disabled():
    p = _build_blind_prompt(TASK, {"raw_answer": "x"}, {"raw_answer": "y"})
    assert "未执行" in p


# ---- _build_conclusion ----

def test_conclusion_x_wins():
    assert "答案X" in _build_conclusion([], {}, 10, 5)


def test_conclusion_y_wins():
    assert "答案Y" in _build_conclusion([], {}, 3, 8)


def test_conclusion_tie():
    assert "平局" in _build_conclusion([], {}, 5, 5)


# ---- judge_task (mock transport) ----

def _mock_handler_ok(request):
    return httpx.Response(200, json={"choices": [{"message": {"content": VALID_V}}]})


def _mock_handler_then_ok():
    count = [0]

    def h(request):
        count[0] += 1
        if count[0] < 2:
            return httpx.Response(500, json={})
        return httpx.Response(200, json={"choices": [{"message": {"content": VALID_V}}]})
    return h


def _mock_handler_always_fail(request):
    return httpx.Response(500, json={})


def test_judge_task_valid_verdict():
    async def _run():
        transport = httpx.MockTransport(_mock_handler_ok)
        async with httpx.AsyncClient(transport=transport) as c:
            return await judge_task(c, TASK, ANSWER_X, ANSWER_Y, JUDGE_CFG)
    v = asyncio.run(_run())
    assert v["winner"] == "answer_x"
    assert not v.get("_invalid")


def test_judge_task_retry_succeeds():
    async def _run():
        transport = httpx.MockTransport(_mock_handler_then_ok())
        async with httpx.AsyncClient(transport=transport) as c:
            return await judge_task(c, TASK, ANSWER_X, ANSWER_Y, JUDGE_CFG, max_retries=5)
    v = asyncio.run(_run())
    assert v["winner"] == "answer_x" and not v.get("_invalid")


def test_judge_task_all_fail_returns_invalid_fallback():
    async def _run():
        transport = httpx.MockTransport(_mock_handler_always_fail)
        async with httpx.AsyncClient(transport=transport) as c:
            return await judge_task(c, TASK, ANSWER_X, ANSWER_Y, JUDGE_CFG)
    v = asyncio.run(_run())
    assert v["_invalid"] is True
    assert v["winner"] == "tie"
    assert "评审模型未能返回有效" in v["basis"]


# ---- run_judge (monkeypatch _call_judge_model) ----

def test_run_judge_returns_valid_structure(monkeypatch):
    async def mock_call(client, judge_cfg, prompt, *args, **kwargs):
        return VALID_V
    monkeypatch.setattr(judge_module, "_call_judge_model", mock_call)

    task_set = {"tasks": [TASK]}
    answers_x = {"model": "模型A", "answers": [ANSWER_X]}
    answers_y = {"model": "模型B", "answers": [ANSWER_Y]}

    result = asyncio.run(run_judge(task_set, answers_x, answers_y, JUDGE_CFG))
    assert result["meta"]["total"] == 1
    assert result["meta"]["valid"] == 1
    assert result["meta"]["invalid"] == 0
    assert "revealed" in result
    assert "totals" in result
    assert isinstance(result["scores"], list)
    assert len(result["scores"]) == 1


def test_run_judge_revealed_uses_file_labels(monkeypatch):
    async def mock_call(client, judge_cfg, prompt, *args, **kwargs):
        return VALID_V
    monkeypatch.setattr(judge_module, "_call_judge_model", mock_call)

    task_set = {"tasks": [TASK]}
    answers_x = {"model": "模型A", "answers": [ANSWER_X]}
    answers_y = {"model": "模型B", "answers": [ANSWER_Y]}

    result = asyncio.run(run_judge(task_set, answers_x, answers_y, JUDGE_CFG))
    revealed = result["revealed"]
    assert revealed["answer_x"] in ("answers-a.json", "answers-b.json")
    assert revealed["answer_y"] in ("answers-a.json", "answers-b.json")
    assert revealed["answer_x"] != revealed["answer_y"]
