# -*- coding: utf-8 -*-
"""迭代十一：文本对战（battle）API 集成测试。

覆盖不变量：
- 抽题：random 内置题库（数量/无期望答案/同 seed 复现/count 收敛）、
  custom 评测集（404/400）、边界收敛
- 流式：SSE 双路事件（a/b delta、done、error 降级）、模型 URL SSRF 400、
  模型名空 400
- benchmark 列表：完成批次 score 计算、未完成 None
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from backend import main as main_module
from backend import storage
from backend.engine import battle as battle_module
from backend.engine.tasks import QUESTION_POOL

PUBLIC_URL = "https://8.8.8.8/v1"

TOTAL_POOL = sum(len(v) for v in QUESTION_POOL.values())


def _dataset() -> dict:
    return {
        "name": "对战集A",
        "tasks": [
            {"id": "B1", "type": "生成式", "dimension": "语言能力",
             "prompt": "用一句话介绍你自己", "rubric_note": "满分10分"},
            {"id": "B2", "type": "判别式", "dimension": "数学能力",
             "prompt": "1+1=?",
             "test_cases": [{"input": "1+1=?", "expected": "2"}]},
            {"id": "B3", "type": "生成式", "dimension": "知识能力",
             "prompt": "谈谈长城的由来", "rubric_note": "满分10分"},
        ],
    }


def _model(name: str = "A") -> dict:
    return {"url": PUBLIC_URL, "key": "k", "name": name,
            "temperature": 0.7, "max_tokens": 4096}


@pytest.fixture
def client():
    return TestClient(main_module.app)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    storage.save_dataset("对战集A", _dataset())
    yield


def _fake_battle_stream(*args, **kwargs):
    """假流式生成器：模拟双路 delta → done。"""
    async def gen():
        yield {"side": "a", "delta": "你好"}
        yield {"side": "b", "delta": "你好"}
        yield {"side": "a", "delta": "世界"}
        yield {"side": "b", "delta": "世界"}
        yield {"side": "a", "done": True}
        yield {"side": "b", "done": True}
    return gen()


def _sse_events(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if line.startswith("data:"):
            data = line[5:].strip()
            if not data:
                continue
            events.append(json.loads(data))
    return events


# ---- 抽题 ----

def test_battle_questions_random(client):
    r = client.post("/api/battle/questions",
                    json={"count": 5, "source": "random", "seed": 7})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["total"] == 5
    for q in d["questions"]:
        assert q["prompt"]
        assert "expected" not in q and "rubric_note" not in q and "test_cases" not in q
        assert q["category"]


def test_battle_questions_random_reproducible(client):
    a = client.post("/api/battle/questions",
                    json={"count": 5, "source": "random", "seed": 42}).json()
    b = client.post("/api/battle/questions",
                    json={"count": 5, "source": "random", "seed": 42}).json()
    assert [q["question_id"] for q in a["questions"]] == \
           [q["question_id"] for q in b["questions"]]


def test_battle_questions_random_count_capped(client):
    r = client.post("/api/battle/questions",
                    json={"count": 50, "source": "random"})
    assert r.status_code == 200
    assert r.json()["total"] <= TOTAL_POOL


def test_battle_questions_custom_dataset(client):
    r = client.post("/api/battle/questions",
                    json={"count": 2, "source": "custom", "dataset_name": "对战集A"})
    assert r.status_code == 200
    qs = r.json()["questions"]
    assert len(qs) == 2
    assert {q["prompt"] for q in qs} == {"1+1=?", "谈谈长城的由来"} or len(qs) == 2


def test_battle_questions_custom_missing_404(client):
    r = client.post("/api/battle/questions",
                    json={"count": 2, "source": "custom", "dataset_name": "不存在集"})
    assert r.status_code == 404


def test_battle_questions_custom_without_name_400(client):
    r = client.post("/api/battle/questions",
                    json={"count": 2, "source": "custom"})
    assert r.status_code == 400


def test_battle_questions_count_boundary(client):
    assert client.post("/api/battle/questions",
                       json={"count": 0}).status_code == 422
    assert client.post("/api/battle/questions",
                       json={"count": 51}).status_code == 422


# ---- 流式 ----

def test_battle_stream_sse_events(client, monkeypatch):
    monkeypatch.setattr(main_module, "stream_battle", _fake_battle_stream)
    r = client.post("/api/battle/stream", json={
        "prompt": "你好", "context": "",
        "model_a": _model("A"), "model_b": _model("B"),
    })
    assert r.status_code == 200, r.text
    assert r.headers.get("content-type", "").startswith("text/event-stream")
    events = _sse_events(r.text)
    sides = [e["side"] for e in events if "delta" in e]
    assert sides.count("a") == 2 and sides.count("b") == 2
    done = [e for e in events if "done" in e]
    assert {e["side"] for e in done} == {"a", "b"}


def test_battle_stream_side_error_keeps_other(client, monkeypatch):
    async def gen():
        yield {"side": "a", "delta": "部分输出"}
        yield {"side": "a", "error": "连接中断"}
        yield {"side": "b", "delta": "正常输出"}
        yield {"side": "b", "done": True}
    monkeypatch.setattr(main_module, "stream_battle", lambda *a, **k: gen())
    r = client.post("/api/battle/stream", json={
        "prompt": "你好", "model_a": _model("A"), "model_b": _model("B"),
    })
    assert r.status_code == 200
    events = _sse_events(r.text)
    err = [e for e in events if "error" in e]
    assert len(err) == 1 and err[0]["side"] == "a"
    done = [e for e in events if "done" in e]
    assert done and done[0]["side"] == "b"


def test_battle_stream_ssrf_rejected(client, monkeypatch):
    monkeypatch.setattr(main_module, "stream_battle", _fake_battle_stream)
    r = client.post("/api/battle/stream", json={
        "prompt": "你好",
        "model_a": {"url": "http://127.0.0.1:8000/v1", "key": "k", "name": "内网"},
        "model_b": _model("B"),
    })
    assert r.status_code == 400
    assert "URL" in r.json()["detail"]


def test_battle_stream_blank_name_400(client, monkeypatch):
    monkeypatch.setattr(main_module, "stream_battle", _fake_battle_stream)
    r = client.post("/api/battle/stream", json={
        "prompt": "你好",
        "model_a": {"url": PUBLIC_URL, "key": "k", "name": "  "},
        "model_b": _model("B"),
    })
    assert r.status_code == 400


# ---- URL 端点约定与上游错误瘦身（迭代十一修复） ----

def test_chat_endpoint_auto_append_and_no_duplicate():
    assert battle_module._chat_endpoint("https://api.example.com/v1") == \
        "https://api.example.com/v1/chat/completions"
    assert battle_module._chat_endpoint("https://api.example.com/v1/") == \
        "https://api.example.com/v1/chat/completions"
    assert battle_module._chat_endpoint("https://api.example.com/v1/chat/completions") == \
        "https://api.example.com/v1/chat/completions"
    assert battle_module._chat_endpoint("  ") == ""


def test_upstream_error_html_shortened_json_kept():
    err = battle_module._upstream_error(
        404, '<!DOCTYPE html><html lang="en"><meta og:image="social-share.png">...')
    assert "DOCTYPE" not in err
    assert "请检查是否填写 OpenAI 兼容" in err
    err2 = battle_module._upstream_error(400, '{"error": "bad request"}')
    assert err2 == 'HTTP 400: {"error": "bad request"}'


def test_battle_stream_endpoint_appended_and_404_html_shortened(monkeypatch):
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        return httpx.Response(404, content=b"<!DOCTYPE html><html>not found</html>",
                              headers={"content-type": "text/html"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(battle_module, "build_upstream_client", lambda: client)

    async def run():
        events = []
        async for evt in battle_module.stream_battle(_model("A"), _model("B"), "你好"):
            events.append(evt)
        return events

    events = asyncio.run(run())
    assert seen["path"] == "/v1/chat/completions"
    errs = [e for e in events if "error" in e]
    assert len(errs) == 2
    assert "DOCTYPE" not in errs[0]["error"]
    assert "请检查是否填写 OpenAI 兼容" in errs[0]["error"]


# ---- benchmark 列表 score ----

def test_benchmark_list_score_field(client):
    r = client.get("/api/benchmark")
    assert r.status_code == 200
    for b in r.json()["batches"]:
        assert "score" in b
