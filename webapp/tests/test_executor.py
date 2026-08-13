# -*- coding: utf-8 -*-
"""双模型执行器单元测试（issue #11 残余风险 R3）。

覆盖 API 调用重试、指标采集、代码验真编排、稳定性温度切换、
异常兜底的核心不变量。网络通过 httpx.MockTransport 完全 mock。
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from backend.engine.executor import _call_one, _empty_answers, execute_task

T_USAGE = {"prompt_tokens": 10, "completion_tokens": 5}
OK_BODY = {"choices": [{"message": {"content": "答"}, "finish_reason": "stop"}], "usage": T_USAGE}

CONFIG = {
    "url": "https://x.example.com/v1", "key": "k", "name": "m",
    "temperature": 0.7, "max_tokens": 4096, "top_p": 0.9,
    "code_verify_mode": "off",
}
BASIC_TASK = {"id": "T1", "dimension": "知识能力", "prompt": "1+1?"}
CODE_TASK = {"id": "C1", "dimension": "代码能力", "prompt": "写函数",
             "test_cases": [{"input": "f()", "expected": "1"}]}
STABILITY_TASK = {"id": "S1", "dimension": "长文本与效率稳定性", "prompt": "p"}


# ---- handlers ----

def _handler_ok(request):
    return httpx.Response(200, json=OK_BODY)


def _handler_length(request):
    body = {"choices": [{"message": {"content": "答"}, "finish_reason": "length"}], "usage": T_USAGE}
    return httpx.Response(200, json=body)


def _handler_400(request):
    return httpx.Response(400, json={"error": {"message": "bad request"}})


def _handler_timeout(request):
    raise httpx.TimeoutException("timeout")


def _handler_conn_error(request):
    raise httpx.ConnectError("refused")


# _raw helper

def _raw_call(handler, **kw):
    async def _run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as c:
            return await _call_one(c, "https://x/v1", "sk", "p", "m", **kw)
    return asyncio.run(_run())


# ---- _call_one ----

def test_call_one_ok():
    raw, info = _raw_call(_handler_ok)
    assert raw == "答"
    assert info["status"] == "ok" and info["latency_ms"] >= 0
    assert info["prompt_tokens"] == 10 and info["completion_tokens"] == 5
    assert info["truncated"] is False and info["attempts"] == 1


def test_call_one_truncated():
    _, info = _raw_call(_handler_length)
    assert info["truncated"] is True


def test_call_one_400_retry_then_error():
    raw, info = _raw_call(_handler_400)
    assert raw is None
    assert info["status"] == "error" and info["attempts"] == 2


def test_call_one_timeout_retry():
    raw, info = _raw_call(_handler_timeout)
    assert info["latency_ms"] == 120000
    assert info["attempts"] == 2


def test_call_one_connection_error():
    raw, info = _raw_call(_handler_conn_error)
    assert "HTTP 连接错误" in (info["error"] or "")


def test_call_one_top_p_only_when_given():
    cap = {}

    def h(req):
        cap["p"] = json.loads(req.content)
        return httpx.Response(200, json=OK_BODY)

    async def _both():
        transport = httpx.MockTransport(h)
        async with httpx.AsyncClient(transport=transport) as c:
            await _call_one(c, "https://x/v1", "sk", "p", "m", top_p=0.9)
            assert cap["p"]["top_p"] == 0.9
            cap.clear()
            await _call_one(c, "https://x/v1", "sk", "p", "m", top_p=None)
            assert "top_p" not in cap["p"]
    asyncio.run(_both())


def test_call_one_auth_header():
    cap = {}

    def h(req):
        cap["auth"] = req.headers.get("authorization", "")
        return httpx.Response(200, json=OK_BODY)

    _raw_call(h, top_p=None)
    assert cap["auth"] == "Bearer sk"


# ---- execute_task ----

def _exec(handler, task=None, is_repeat_task=False, **kw):
    async def _run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as c:
            return await execute_task(c, task or BASIC_TASK, CONFIG, is_repeat_task, **kw)
    return asyncio.run(_run())


def test_execute_task_basic():
    ans = _exec(_handler_ok)
    assert ans["id"] == "T1"
    assert ans["raw_answer"] == "答"
    assert ans["api_info"]["status"] == "ok"


def test_execute_task_code_max_tokens():
    cap = {}

    def h(req):
        cap["mt"] = json.loads(req.content).get("max_tokens")
        return httpx.Response(200, json=OK_BODY)

    ans = _exec(h, task=CODE_TASK, is_repeat_task=False)
    assert cap["mt"] >= 8192


def test_execute_task_off_mode_code_verify_present():
    ans = _exec(_handler_ok, task=CODE_TASK, is_repeat_task=False)
    assert "code_verify" in ans
    assert ans["code_verify"]["status"] == "disabled"


def test_execute_task_no_test_cases_no_code_verify():
    t = {"id": "C2", "dimension": "代码能力", "prompt": "p"}
    ans = _exec(_handler_ok, task=t, is_repeat_task=False)
    assert "code_verify" not in ans


def test_stability_repeat2_temp_zero():
    cap = {}

    def h(req):
        cap["temp"] = json.loads(req.content).get("temperature")
        return httpx.Response(200, json=OK_BODY)

    _exec(h, task=STABILITY_TASK, is_repeat_task=True, repeat_index=2)
    assert cap["temp"] == 0.0


# ---- _empty_answers ----

def test_empty_answers_structure():
    tasks = [{"id": "t1"}, {"id": "t2"}]
    ans = _empty_answers({"name": "模型A", "url": "https://x"}, "boom", tasks)
    assert ans["model"] == "模型A"
    assert ans["note"].startswith("webapp executor")
    assert len(ans["answers"]) == 2
    for a in ans["answers"]:
        assert a["api_info"]["status"] == "error"
        assert a["api_info"]["error"] == "boom"
        assert a["api_info"]["attempts"] == 0
