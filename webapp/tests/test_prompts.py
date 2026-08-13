# -*- coding: utf-8 -*-
"""提示策略与语义采集单元测试（迭代二 步骤5）。

覆盖：cot/direct/fewshot 三种策略的 prompt 构造（纯函数）、fewshot 无
test_cases 回退 direct、executor 实际 payload 携带策略、生成式题 embedding
采集写入 ans_entry["semantic"] 及其全部降级路径。
"""
from __future__ import annotations

import asyncio
import json

import httpx

from backend.engine.executor import (
    COT_SUFFIX,
    _embed_pair,
    _make_embedder,
    build_prompt,
    execute_task,
)
from backend.engine.embed import resolve_provider

BASIC_TASK = {"id": "T1", "dimension": "知识能力", "prompt": "1+1=?"}
FEWSHOT_TASK = {
    "id": "T2", "dimension": "知识能力", "prompt": "按示例作答",
    "test_cases": [
        {"input": "a+b", "expected": "c"},
        {"input": "x", "expected": "y"},
        {"input": "p", "expected": "q"},
    ],
}
GEN_TASK = {
    "id": "G1", "dimension": "语言能力", "type": "生成式",
    "prompt": "翻译", "expected": "reference text",
}
CONFIG = {
    "url": "https://x.example.com/v1", "key": "k", "name": "m",
    "temperature": 0.7, "max_tokens": 4096, "top_p": 0.9,
    "code_verify_mode": "off",
}


# ---- build_prompt 纯函数 ----

def test_cot_default_appends_suffix():
    assert build_prompt(BASIC_TASK) == "1+1=?" + COT_SUFFIX
    assert build_prompt(BASIC_TASK, "cot") == "1+1=?" + COT_SUFFIX


def test_direct_verbatim():
    assert build_prompt(BASIC_TASK, "direct") == "1+1=?"


def test_fewshot_injects_examples():
    p = build_prompt(FEWSHOT_TASK, "fewshot")
    assert p.startswith("按示例作答")
    assert "输入：a+b\n输出：c" in p
    assert "输入：x\n输出：y" in p
    # 只注入前 2 条示例
    assert "输入：p" not in p


def test_fewshot_without_cases_falls_back_verbatim():
    assert build_prompt(BASIC_TASK, "fewshot") == "1+1=?"


def test_unknown_strategy_uses_cot():
    assert build_prompt(BASIC_TASK, "weird") == "1+1=?" + COT_SUFFIX


def test_build_prompt_deterministic():
    assert build_prompt(FEWSHOT_TASK, "fewshot") == build_prompt(FEWSHOT_TASK, "fewshot")


# ---- execute_task payload 携带策略 ----

def _run_task(task, config=None, handler=None, embedder=None):
    def _ok(req):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "答"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })

    captured = {}

    def h(req):
        captured["content"] = json.loads(req.content)["messages"][0]["content"]
        return (handler or _ok)(req)

    async def _run():
        transport = httpx.MockTransport(h)
        async with httpx.AsyncClient(transport=transport) as c:
            return await execute_task(c, task, config or CONFIG, False, 1, embedder)

    return asyncio.run(_run()), captured


def test_payload_uses_cot_by_default():
    config = {**CONFIG}
    _, cap = _run_task(FEWSHOT_TASK, config=config)
    assert cap["content"].endswith(COT_SUFFIX)


def test_payload_direct_verbatim():
    config = {**CONFIG, "prompt_strategy": "direct"}
    ans, cap = _run_task(FEWSHOT_TASK, config=config)
    assert cap["content"] == FEWSHOT_TASK["prompt"]


def test_payload_fewshot_has_examples():
    config = {**CONFIG, "prompt_strategy": "fewshot"}
    _, cap = _run_task(FEWSHOT_TASK, config=config)
    assert "示例：" in cap["content"] and "输出：c" in cap["content"]


# ---- semantic 采集 ----

def _semantic_collected(embedder=None, task=None, raw="原生回答"):
    def _ok(req):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": raw}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })

    async def _run():
        transport = httpx.MockTransport(_ok)
        async with httpx.AsyncClient(transport=transport) as c:
            return await execute_task(c, task or GEN_TASK, CONFIG, False, 1, embedder)

    return asyncio.run(_run())


async def _emb_ok(texts):
    return [[1.0, 0.0], [0.0, 1.0]] if len(texts) == 2 else None


def test_semantic_collected_when_embedder_ok():
    ans = _semantic_collected(embedder=_emb_ok)
    assert ans["semantic"] == {"vector": [1.0, 0.0], "ref_vector": [0.0, 1.0]}


def test_semantic_absent_without_embedder():
    assert "semantic" not in _semantic_collected(embedder=None)


async def _emb_none(texts):
    return None


def test_semantic_absent_when_embedder_returns_none():
    assert "semantic" not in _semantic_collected(embedder=_emb_none)


async def _emb_short(texts):
    return [[1.0]]


def test_semantic_absent_when_bad_vector_count():
    assert "semantic" not in _semantic_collected(embedder=_emb_short)


async def _emb_broken(texts):
    raise RuntimeError("boom")


def test_semantic_absent_when_embedder_raises():
    assert "semantic" not in _semantic_collected(embedder=_emb_broken)


def test_semantic_absent_for_non_generative():
    t = {"id": "D1", "dimension": "知识能力", "type": "判别式", "prompt": "p",
         "expected": "x"}
    assert "semantic" not in _semantic_collected(embedder=_emb_ok, task=t)


def test_semantic_absent_for_generative_without_expected():
    t = {"id": "G2", "dimension": "语言能力", "type": "生成式", "prompt": "p"}
    assert "semantic" not in _semantic_collected(embedder=_emb_ok, task=t)


def test_semantic_absent_on_api_error():
    raw, ans = None, None

    def _err(req):
        return httpx.Response(500, json={})

    async def _run():
        nonlocal ans
        transport = httpx.MockTransport(_err)
        async with httpx.AsyncClient(transport=transport) as c:
            ans = await execute_task(c, GEN_TASK, CONFIG, False, 1, _emb_ok)
        return ans

    asyncio.run(_run())
    assert ans["api_info"]["status"] == "error"
    assert "semantic" not in ans


def test_embed_pair_empty_input_returns_none():
    async def _run():
        return await _embed_pair(None, "", "x")

    assert asyncio.run(_run()) is None


# ---- _make_embedder provider 解析 ----

def test_make_embedder_offline_returns_none(monkeypatch):
    monkeypatch.delenv("MODEL_DUEL_EMBEDDING_URL", raising=False)
    monkeypatch.setattr("backend.engine.embed._bge_importable", lambda: False)
    resolved = resolve_provider({"provider": "offline"})

    async def _run():
        transport = httpx.MockTransport(lambda r: httpx.Response(500))
        async with httpx.AsyncClient(transport=transport) as c:
            return await _make_embedder(resolved, c)

    assert asyncio.run(_run()) is None


def test_make_embedder_external_calls_embeddings():
    resolved = {"kind": "external", "cfg": {"url": "https://8.8.8.8/v1",
                                            "name": "bge", "key": "sk"}}
    cap = {}

    def h(req):
        assert req.url.path.endswith("/embeddings")
        cap["n"] = len(json.loads(req.content)["input"])
        return httpx.Response(200, json={"data": [{"embedding": [1.0]}, {"embedding": [2.0]}]})

    async def _run():
        transport = httpx.MockTransport(h)
        async with httpx.AsyncClient(transport=transport) as c:
            emb = await _make_embedder(resolved, c)
            assert emb is not None
            return await emb(["a", "b"])

    out = asyncio.run(_run())
    assert out == [[1.0], [2.0]] and cap["n"] == 2


def test_make_embedder_local_bge_without_model_degrades(monkeypatch):
    monkeypatch.delenv("MODEL_DUEL_BGE_MODEL_DIR", raising=False)
    resolved = {"kind": "local_bge", "cfg": {}}

    async def _run():
        transport = httpx.MockTransport(lambda r: httpx.Response(500))
        async with httpx.AsyncClient(transport=transport) as c:
            emb = await _make_embedder(resolved, c)
            return await emb(["x", "y"])

    assert asyncio.run(_run()) is None