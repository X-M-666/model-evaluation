# -*- coding: utf-8 -*-
"""单臂 rubric 评审协议测试（迭代三）：prompt 结构、结构化输出校验、
invalid 标记、健康度阈值、progress_cb、零网络。"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from backend.engine import judge as judge_module
from backend.engine.judge import (
    _build_single_arm_prompt,
    _parse_single_arm_verdict,
    health_check,
    run_single_arm_judge,
)

TASK = {
    "id": "T1", "dimension": "语言能力", "prompt": "把一句话翻译成英文",
    "rubric_note": "忠实原文、通顺、无语法错误得高分",
    "test_cases": [],
}
ANSWER_OK = {"id": "T1", "raw_answer": "Hello world.", "api_info": {}}


def _task_set(*tasks):
    return {"meta": {"total": len(tasks)}, "tasks": list(tasks)}


# ---- 单臂 prompt ----

def test_single_arm_prompt_has_rubric_and_fence():
    answer = {"id": "T1", "raw_answer": "x" * 99999, "api_info": {}}
    p = _build_single_arm_prompt(TASK, answer)
    assert "评分标准" in p and "Hello world" not in p
    assert "已截断" in p and "score" in p and "basis" in p
    assert "答案X" not in p and "答案Y" not in p


def test_single_arm_prompt_code_verify_status():
    answer = {"id": "T1", "raw_answer": "code", "api_info": {},
              "code_verify": {"status": "disabled", "reason": "已禁用"}}
    p = _build_single_arm_prompt(TASK, answer)
    assert "未执行（已禁用）" in p


# ---- 结构化输出校验 ----

def test_parse_single_arm_valid():
    v = _parse_single_arm_verdict(
        '{"id":"T1","dimension":"语言能力","score":8.5,"basis":"翻译忠实通顺"}'
    )
    assert v == {"id": "T1", "dimension": "语言能力", "score": 8.5,
                 "basis": "翻译忠实通顺"}


def test_parse_single_arm_rejects_bad_score():
    assert _parse_single_arm_verdict('{"id":"T1","score":12,"basis":"x"}') is None
    assert _parse_single_arm_verdict('{"id":"T1","score":"很高","basis":"x"}') is None
    assert _parse_single_arm_verdict('{"id":"T1","score":-1,"basis":"x"}') is None


def test_parse_single_arm_requires_basis():
    assert _parse_single_arm_verdict('{"id":"T1","score":5}') is None
    assert _parse_single_arm_verdict('{"id":"T1","score":5,"basis":""}') is None


def test_parse_single_arm_markdown_fence():
    v = _parse_single_arm_verdict('```json\n{"id":"T1","score":6,"basis":"ok"}\n```')
    assert v["score"] == 6.0


def test_parse_single_arm_garbage():
    assert _parse_single_arm_verdict("对不起，我无法评分") is None
    assert _parse_single_arm_verdict("") is None


# ---- run_single_arm_judge 全链路（MockTransport） ----

def _mock_client(raw_responses: list[str]) -> tuple[httpx.AsyncClient, int]:
    calls = {"n": 0}

    async def handler(request):
        calls["n"] += 1
        body = {"choices": [{"message": {"content": raw_responses[min(calls["n"] - 1, len(raw_responses) - 1)]}}]}
        return httpx.Response(200, json=body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client, calls


def _stub_client(raw_responses: list[str]):
    """同步替换 build_upstream_client（原函数为同步工厂）。"""
    return lambda **kwargs: _mock_client(raw_responses)[0]


def test_run_single_arm_full(monkeypatch):
    monkeypatch.setattr(judge_module, "build_upstream_client", _stub_client([
        '{"id":"T1","dimension":"语言能力","score":7,"basis":"通顺"}',
        '{"id":"T2","dimension":"语言能力","score":9,"basis":"优秀"}',
    ]))
    tasks = [TASK, {**TASK, "id": "T2"}]
    answers = {"model": "A", "answers": [ANSWER_OK, {**ANSWER_OK, "id": "T2"}]}
    progress = []

    async def _progress(done, total):
        progress.append((done, total))

    async def _run():
        return await run_single_arm_judge(
            _task_set(*tasks), answers,
            {"url": "https://8.8.8.8/v1", "key": "k", "name": "J"},
            progress_cb=_progress,
        )

    result = asyncio.run(_run())
    assert result["meta"]["total"] == 2 and result["meta"]["valid"] == 2
    assert [s["score"] for s in result["scores"]] == [7.0, 9.0]
    assert result["totals"]["score"] == 16.0
    assert result["totals"]["max"] == 20.0
    assert result["health"]["healthy"] is True
    assert progress == [(1, 2), (2, 2)]


def test_run_single_arm_invalid_retry_then_mark(monkeypatch):
    monkeypatch.setattr(judge_module, "build_upstream_client", _stub_client([
        "不是 JSON",             # 首次失败
        '{"id":"T1","score":5,"basis":"勉强"}',  # 重试成功
    ]))
    result = asyncio.run(run_single_arm_judge(
        _task_set(TASK), {"model": "A", "answers": [ANSWER_OK]},
        {"url": "https://8.8.8.8/v1", "key": "k", "name": "J"},
    ))
    assert result["meta"]["valid"] == 1
    assert result["scores"][0]["score"] == 5.0


def test_run_single_arm_all_invalid(monkeypatch):
    monkeypatch.setattr(judge_module, "build_upstream_client", _stub_client(["完全不可解析"]))
    result = asyncio.run(run_single_arm_judge(
        _task_set(TASK), {"model": "A", "answers": [ANSWER_OK]},
        {"url": "https://8.8.8.8/v1", "key": "k", "name": "J"},
    ))
    assert result["meta"]["invalid"] == 1 and result["meta"]["valid"] == 0
    assert result["scores"][0]["_invalid"] is True
    assert result["health"]["healthy"] is False
    assert result["health"]["alarm"] is True


def test_run_single_arm_excluded_from_total(monkeypatch):
    monkeypatch.setattr(judge_module, "build_upstream_client", _stub_client(
        ['{"id":"T7","score":8,"basis":"ok"}']))
    task = {**TASK, "id": "T7", "excluded_from_total": True}
    result = asyncio.run(run_single_arm_judge(
        _task_set(task), {"model": "A", "answers": [{**ANSWER_OK, "id": "T7"}]},
        {"url": "https://8.8.8.8/v1", "key": "k", "name": "J"},
    ))
    assert result["meta"]["excluded_ids"] == ["T7"]
    assert result["totals"]["score"] == 0.0  # 不计分题不纳入总分


# ---- 健康度 ----

def test_health_check_threshold():
    v_list = [
        {"id": "a", "_invalid": False}, {"id": "b", "_invalid": False},
        {"id": "c", "_invalid": True},
    ]
    h = health_check(v_list, threshold=0.1)
    assert h["invalid_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert h["alarm"] is True and h["healthy"] is False


def test_health_check_below_threshold():
    v_list = [{"_invalid": False}] * 9 + [{"_invalid": True}]
    h = health_check(v_list)  # 默认 10%
    assert h["alarm"] is False and h["healthy"] is True


def test_health_check_from_meta_dict():
    h = health_check({"total": 10, "invalid": 2}, threshold=0.1)
    assert h["alarm"] is True
    h2 = health_check({"total": 10, "invalid": 0})
    assert h2["alarm"] is False


def test_health_check_empty():
    h = health_check([], threshold=0.1)
    assert h["healthy"] is True and h["invalid_rate"] == 0.0