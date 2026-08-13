# -*- coding: utf-8 -*-
"""embedding provider 单元测试（迭代二）：auto 解析、external mock 调用、
local 缺失报错、offline n-gram 确定性。网络全部 MockTransport 注入。"""
from __future__ import annotations

import asyncio

import httpx
import pytest

from backend.engine import embed
from backend.engine.embed import cosine, embed_texts, ngram_vec, resolve_provider


def test_ngram_vec_deterministic_and_empty():
    a = ngram_vec("你好世界")
    b = ngram_vec("你好世界")
    assert a == b and a
    assert ngram_vec("") == {}


def test_cosine_dicts():
    assert cosine({"a": 1}, {"a": 1}) == 1.0
    assert cosine({"a": 1}, {"b": 1}) == 0.0
    assert cosine({}, {}) == 0.0


def test_cosine_lists():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine([], []) == 0.0
    assert cosine([1.0], [1.0, 0.0]) == 0.0


def test_resolve_offline_default(monkeypatch):
    monkeypatch.delenv("MODEL_DUEL_EMBEDDING_URL", raising=False)
    monkeypatch.setattr(embed, "_bge_importable", lambda: False)
    r = resolve_provider(None)
    assert r["kind"] == "offline"


def test_resolve_auto_env_prefers_external(monkeypatch):
    monkeypatch.setenv("MODEL_DUEL_EMBEDDING_URL", "https://8.8.8.8/v1")
    monkeypatch.setenv("MODEL_DUEL_EMBEDDING_KEY", "k")
    r = resolve_provider(None)
    assert r["kind"] == "external"
    assert r["cfg"]["url"] == "https://8.8.8.8/v1"


def test_resolve_auto_local_bge_when_importable(monkeypatch):
    monkeypatch.delenv("MODEL_DUEL_EMBEDDING_URL", raising=False)
    monkeypatch.setattr(embed, "_bge_importable", lambda: True)
    assert resolve_provider(None)["kind"] == "local_bge"


def test_resolve_explicit_offline_ignores_env(monkeypatch):
    monkeypatch.setenv("MODEL_DUEL_EMBEDDING_URL", "https://8.8.8.8/v1")
    r = resolve_provider({"provider": "offline"})
    assert r["kind"] == "offline"


def test_resolve_local_bge_missing_onnx(monkeypatch):
    monkeypatch.setattr(embed, "_bge_importable", lambda: False)
    r = resolve_provider({"provider": "local_bge"})
    assert r["kind"] == "local_bge"
    assert "onnxruntime" in (r["error"] or "")


def test_resolve_external_missing_url():
    r = resolve_provider({"provider": "external"})
    assert r["kind"] == "external"
    assert "缺少 URL" in (r["error"] or "")


def _run_embed_texts(handler, kind="external", cfg=None):
    async def _run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as c:
            return await embed_texts(kind, cfg or {"url": "https://8.8.8.8/v1", "name": "m"}, c, ["你好"])
    return asyncio.run(_run())


def test_embed_texts_external_ok():
    def h(req):
        return httpx.Response(200, json={"data": [{"embedding": [0.1, 0.2]}]})
    assert _run_embed_texts(h) == [[0.1, 0.2]]


def test_embed_texts_external_error_returns_none():
    def h(req):
        return httpx.Response(500, json={})
    assert _run_embed_texts(h) is None


def test_embed_texts_timeout_returns_none():
    def h(req):
        raise httpx.TimeoutException("t")
    assert _run_embed_texts(h) is None


def test_embed_texts_non_external_returns_none():
    assert _run_embed_texts(lambda r: pytest.fail("不应调用"), kind="offline") is None


def test_embed_texts_payload_shape():
    cap = {}

    def h(req):
        import json
        cap["body"] = json.loads(req.content)
        cap["auth"] = req.headers.get("authorization", "")
        return httpx.Response(200, json={"data": [{"embedding": [1.0]}, {"embedding": [2.0]}]})

    out = _run_embed_texts(h, cfg={"url": "https://8.8.8.8/v1", "key": "sk", "name": "bge"})
    assert out == [[1.0], [2.0]]
    assert cap["body"]["model"] == "bge"
    assert cap["body"]["input"] == ["你好"]
    assert cap["auth"] == "Bearer sk"
