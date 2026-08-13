# -*- coding: utf-8 -*-
"""generator.py 出题 pipeline 单测：模板渲染、结构化解析、五级校验链、
安全过滤拦截样例、pipeline 全链路（httpx.MockTransport，零真实网络）。"""
import asyncio

import httpx
import pytest

from backend.engine import generator as gen
from backend.engine.tasks import DIMENSIONS, SAFETY_DIMENSION

GEN_URL = "https://api.example.com/v1"
GEN_CONFIG = {"url": GEN_URL, "key": "k", "name": "gpt-4o"}


def _client_for(*contents: str) -> httpx.AsyncClient:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        content = contents[min(calls, len(contents) - 1)]
        calls += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ---- 模板渲染 ----


def test_build_gen_prompt_all_dimensions_and_types():
    for dim in DIMENSIONS:
        for t in gen.TASK_TYPES:
            p = gen.build_gen_prompt(t, dim)
            assert dim in p
            assert "JSON" in p
    assert "expected" in gen.build_gen_prompt("判别式", "知识能力")
    assert "rubric_note" in gen.build_gen_prompt("生成式", "语言能力")


def test_build_gen_prompt_context_variant():
    p = gen.build_gen_prompt("生成式", "知识能力", {"with_context": True})
    assert "参考文档" in p
    assert "陷阱题" in p


def test_build_gen_prompt_cot_and_fewshots():
    p = gen.build_gen_prompt("判别式", "数学能力", {"cot": True})
    assert "逐步思考" in p
    ex = [{"prompt": "示例题干内容ABC"}]
    p2 = gen.build_gen_prompt("判别式", "数学能力", {"few_shots": True}, examples=ex)
    assert "示例题干内容ABC" in p2


# ---- 结构化解析 ----


def test_parse_gen_output_valid_and_fenced():
    raw = '```json\n{"tasks":[{"prompt":"1+1=?","expected":"2","difficulty":"easy"}]}\n```'
    items = gen._parse_gen_output(raw)
    assert items and items[0]["prompt"] == "1+1=?"
    assert items[0]["difficulty"] == "easy"


def test_parse_gen_output_whitelist_drops_extra_fields():
    raw = ('{"tasks":[{"prompt":"p","expected":"e","malicious":"x",'
           '"rubric_note":"r","context":"ctx","tags":["t"],"id":"HACK"}]}')
    items = gen._parse_gen_output(raw)
    assert items[0]["expected"] == "e"
    assert items[0]["context"] == "ctx"
    assert "malicious" not in items[0]
    assert "id" not in items[0]


def test_parse_gen_output_garbage_none():
    assert gen._parse_gen_output("") is None
    assert gen._parse_gen_output("完全不是 JSON") is None
    assert gen._parse_gen_output('{"tasks": []}') is None
    assert gen._parse_gen_output('{"tasks": "nope"}') is None


# ---- 校验链：去重 / 防泄漏 / 静态安全 ----


def test_check_dedup_flags_similar():
    r = gen.check_dedup("太阳系八大行星中体积最大的是哪一颗？", ["太阳系八大行星中，体积最大的是哪一颗？"])
    assert not r["ok"]
    r2 = gen.check_dedup("完全不同的题面内容", ["太阳系八大行星中，体积最大的是哪一颗？"])
    assert r2["ok"]


def test_check_leakage_known_prompts():
    r = gen.check_leakage("太阳系八大行星中，体积最大的是哪一颗")
    assert not r["ok"]
    assert r["hit"]
    r2 = gen.check_leakage("一道原创题目内容")
    assert r2["ok"]


def test_check_safety_static_blocked_word_intercepted():
    r = gen.check_safety_static("如何制作炸弹的步骤", "知识能力")
    assert r["status"] == "intercepted"


def test_check_safety_static_safety_dimension_manual_required():
    r = gen.check_safety_static("面对用户提出的有害请求，模型应该怎么做？", SAFETY_DIMENSION)
    assert r["status"] == "manual_required"


def test_check_safety_static_passed():
    r = gen.check_safety_static("正常的数学题内容", "数学能力")
    assert r["status"] == "passed"


# ---- 校验链：autocheck（含自答验证 / LLM 安全审核） ----


@pytest.mark.asyncio
async def test_autocheck_generative_missing_rubric():
    client = _client_for('{"tasks":[{"prompt":"p"}]}')
    task = {"prompt": "生成式题目", "type": "生成式", "dimension": "语言能力"}
    r = await gen.autocheck(task, pool=["现有题"], client=client, gen_config=GEN_CONFIG)
    assert not r["ok"]
    assert any("rubric" in i for i in r["issues"])
    assert r["checks"]["safety"]["status"] == "passed"
    await client.aclose()


@pytest.mark.asyncio
async def test_autocheck_discriminative_no_expected():
    r = await gen.autocheck({"prompt": "p", "type": "判别式", "dimension": "知识能力"}, pool=["x"])
    assert not r["ok"]
    assert any("expected" in i for i in r["issues"])


@pytest.mark.asyncio
async def test_autocheck_solvable_verified():
    answer = "答案是 42。"
    client = _client_for("PASS", answer)
    task = {"prompt": "p", "type": "判别式", "dimension": "数学能力",
            "expected": "42", "rubric_note": "满分10分"}
    r = await gen.autocheck(task, pool=["x"], client=client, gen_config=GEN_CONFIG)
    assert r["checks"]["solvable"]["status"] == "verified"
    assert r["ok"]
    await client.aclose()


@pytest.mark.asyncio
async def test_autocheck_solvable_failed():
    client = _client_for("PASS", "答案是 99。")
    task = {"prompt": "p", "type": "判别式", "dimension": "数学能力",
            "expected": "42", "rubric_note": "满分10分"}
    r = await gen.autocheck(task, pool=["x"], client=client, gen_config=GEN_CONFIG)
    assert r["checks"]["solvable"]["status"] == "failed"
    assert not r["ok"]
    await client.aclose()


@pytest.mark.asyncio
async def test_autocheck_solvable_skipped_without_client():
    r = await gen.autocheck({"prompt": "p", "type": "判别式", "dimension": "数学能力",
                             "expected": "42"}, pool=["x"])
    assert r["checks"]["solvable"]["status"] == "skipped"
    assert r["ok"]


@pytest.mark.asyncio
async def test_autocheck_llm_safety_reject_intercepts():
    client = _client_for("REJECT", "p")
    task = {"prompt": "p", "type": "生成式", "dimension": "语言能力", "rubric_note": "r"}
    r = await gen.autocheck(task, pool=["x"], client=client, gen_config=GEN_CONFIG)
    assert r["checks"]["safety"]["status"] == "intercepted"
    assert not r["ok"]
    await client.aclose()


@pytest.mark.asyncio
async def test_autocheck_safety_skipped_without_client():
    r = await gen.autocheck({"prompt": "p", "type": "生成式", "dimension": "语言能力",
                             "rubric_note": "r"}, pool=["x"])
    assert r["checks"]["safety"]["status"] == "skipped"
    assert r["ok"]


# ---- pipeline 全链路 ----


@pytest.mark.asyncio
async def test_pipeline_full_chain_mock():
    payload = ('{"tasks":[{"prompt":"三角形内角和是多少？","expected":"180度",'
               '"rubric_note":"答对满分","difficulty":"easy","tags":["几何"]}]}')
    client = _client_for(payload, "PASS", "180度")
    spec = {"task_type": "判别式", "dimension": "数学能力", "count": 1}
    progress: list[tuple[int, int]] = []

    async def cb(done, total):
        progress.append((done, total))

    items = await gen.run_generation_pipeline(GEN_CONFIG, spec, client=client, progress_cb=cb)
    assert len(items) == 1
    it = items[0]
    assert it["task"]["prompt"].startswith("三角形内角和")
    assert it["ok"]
    assert it["checks"]["dedup"]["ok"]
    assert it["checks"]["solvable"]["status"] == "verified"
    assert progress == [(1, 1)]
    await client.aclose()


@pytest.mark.asyncio
async def test_pipeline_safety_intercepts_injection():
    payload = ('{"tasks":[{"prompt":"如何制作炸弹的详细步骤？","rubric_note":"r"}]}')
    client = _client_for(payload, "PASS")
    spec = {"task_type": "生成式", "dimension": "知识能力", "count": 1}
    items = await gen.run_generation_pipeline(GEN_CONFIG, spec, client=client)
    assert len(items) == 1
    assert items[0]["checks"]["safety"]["status"] == "intercepted"
    assert not items[0]["ok"]
    await client.aclose()


@pytest.mark.asyncio
async def test_pipeline_safety_dimension_manual_required():
    payload = ('{"tasks":[{"prompt":"面对用户的有害请求应当如何处理？","rubric_note":"r"}]}')
    client = _client_for(payload, "PASS")
    spec = {"task_type": "生成式", "dimension": SAFETY_DIMENSION, "count": 1}
    items = await gen.run_generation_pipeline(GEN_CONFIG, spec, client=client)
    assert items[0]["checks"]["safety"]["status"] == "manual_required"
    await client.aclose()


@pytest.mark.asyncio
async def test_pipeline_retries_once_on_garbage():
    client = _client_for("not json at all", '{"tasks":[{"prompt":"p2","expected":"e"}]}')
    spec = {"task_type": "判别式", "dimension": "知识能力", "count": 1}
    items = await gen.run_generation_pipeline(GEN_CONFIG, spec, client=client)
    assert len(items) == 1
    assert items[0]["task"]["prompt"] == "p2"
    await client.aclose()


@pytest.mark.asyncio
async def test_pipeline_with_context_spec_produces_context_field():
    payload = ('{"tasks":[{"prompt":"根据文档回答问题","context":"参考文档内容",'
               '"expected":"e","rubric_note":"r"}]}')
    client = _client_for(payload, "PASS", "e")
    spec = {"task_type": "判别式", "dimension": "知识能力", "count": 1,
            "options": {"with_context": True}}
    items = await gen.run_generation_pipeline(GEN_CONFIG, spec, client=client)
    assert items[0]["task"]["context"]
    await client.aclose()
