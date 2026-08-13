# -*- coding: utf-8 -*-
"""评审方抽象（迭代三）单测：Reviewer 协议、AgentReviewer pairwise/single
分发与参数透传、HumanReviewer 薄封装委托正确。"""
from __future__ import annotations

import asyncio

import pytest

from backend.engine.reviewer import (
    AgentReviewer,
    HumanReviewer,
    Reviewer,
    PAIRWISE_PROTOCOL,
    SINGLE_PROTOCOL,
)

TASKS = {"meta": {"total": 1}, "tasks": [
    {"id": "T1", "dimension": "数学能力", "prompt": "1+1=?",
     "rubric_note": "答案正确", "test_cases": []}]}
ANSWERS_A = {"model": "模型A", "answers": [
    {"id": "T1", "raw_answer": "2", "api_info": {}}]}
ANSWERS_B = {"model": "模型B", "answers": [
    {"id": "T1", "raw_answer": "3", "api_info": {}}]}
JUDGE_CFG = {"url": "https://8.8.8.8/v1", "key": "k", "name": "J"}


def test_reviewer_protocol_exists():
    assert getattr(Reviewer, "review_all") is not None


def test_agent_reviewer_default_pairwise():
    assert AgentReviewer().protocol == PAIRWISE_PROTOCOL


def test_agent_reviewer_single_protocol():
    assert AgentReviewer(SINGLE_PROTOCOL).protocol == SINGLE_PROTOCOL


def test_agent_reviewer_unknown_protocol():
    with pytest.raises(ValueError):
        AgentReviewer("triple")


def test_agent_reviewer_pairwise_forwards_to_run_judge(monkeypatch):
    captured = {}

    async def fake_run_judge(task_set, ax, ay, judge_config, revealed=None):
        captured["args"] = (task_set, ax, ay, judge_config, revealed)
        return {"meta": {"total": 1}, "scores": []}

    monkeypatch.setattr("backend.engine.reviewer.run_judge", fake_run_judge)
    rv = AgentReviewer()
    revealed = {"rounds": [{"answer_x": "a", "answer_y": "b"}],
                "per_task": [{"task_id": "T1", "answer_x": "b", "answer_y": "a"}]}

    async def _run():
        return await rv.review_all(
            TASKS, {"a": ANSWERS_A, "b": ANSWERS_B},
            {"judge": JUDGE_CFG, "revealed": revealed},
        )

    out = asyncio.run(_run())
    assert out == {"meta": {"total": 1}, "scores": []}
    ts, ax, ay, jc, rv_ = captured["args"]
    assert ts is TASKS and ax is ANSWERS_A and ay is ANSWERS_B
    assert jc == JUDGE_CFG and rv_ == revealed


def test_agent_reviewer_pairwise_without_revealed(monkeypatch):
    captured = {}

    async def fake_run_judge(task_set, ax, ay, judge_config, revealed=None):
        captured["revealed"] = revealed
        return {}

    monkeypatch.setattr("backend.engine.reviewer.run_judge", fake_run_judge)
    asyncio.run(AgentReviewer().review_all(
        TASKS, {"a": ANSWERS_A, "b": ANSWERS_B}, {"judge": JUDGE_CFG}))
    assert captured["revealed"] is None


def test_agent_reviewer_single_forwards_to_run_single_arm(monkeypatch):
    captured = {}

    async def fake_run_single(task_set, answer, judge_config):
        captured["args"] = (task_set, answer, judge_config)
        return {"meta": {"total": 1, "valid": 1, "invalid": 0}, "scores": []}

    monkeypatch.setattr("backend.engine.reviewer.run_single_arm_judge", fake_run_single)
    asyncio.run(AgentReviewer(SINGLE_PROTOCOL).review_all(
        TASKS, {"a": ANSWERS_A, "b": ANSWERS_B}, {"judge": JUDGE_CFG}))
    ts, ans, jc = captured["args"]
    assert ts is TASKS and ans is ANSWERS_A and jc == JUDGE_CFG


def test_human_reviewer_build_round_delegates():
    hr = HumanReviewer()
    round_scores = [{"id": "T1", "answer_x": 8, "answer_y": 2, "note": "A 更完整"}]
    v = hr.build_round(TASKS, round_scores,
                       {"answer_x": "a", "answer_y": "b"},
                       "模型A", "模型B", 0)
    assert v["totals"] == {"answer_x": 8.0, "answer_y": 2.0}
    assert v["winner_model"] == "模型A"
    assert v["scores"][0]["winner"] == "answer_x"
    assert v["revealed"]["answer_x_file"] == "a"


def test_human_reviewer_finalize_delegates():
    hr = HumanReviewer()
    rv = hr.build_round(TASKS, [{"id": "T1", "answer_x": 8, "answer_y": 2}],
                        {"answer_x": "a", "answer_y": "b"}, "模型A", "模型B", 0)
    v = hr.finalize([rv], 1)
    assert v["scores"][0]["answer_x"] == 8.0
    assert v["meta"]["repeat_n"] == 1


def test_human_reviewer_finalize_per_task_reveal():
    hr = HumanReviewer()
    rv1 = hr.build_round(TASKS, [{"id": "T1", "answer_x": 2, "answer_y": 8}],
                         {"answer_x": "b", "answer_y": "a"}, "模型A", "模型B", 0)
    rv2 = hr.build_round(TASKS, [{"id": "T1", "answer_x": 2, "answer_y": 8}],
                         {"answer_x": "b", "answer_y": "a"}, "模型A", "模型B", 1)
    # T1 的 X 实为 b 答卷 → 稳定空间 model_a = b 的分数 = 8（按题归一化）
    v = hr.finalize([rv1, rv2], 2, per_task_reveal={"T1": "b"})
    assert v["scores"][0]["model_a"] == 8.0
    assert v["scores"][0]["model_b"] == 2.0