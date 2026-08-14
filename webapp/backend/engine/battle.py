# -*- coding: utf-8 -*-
"""文本对战（battle）模块：双模型同题并排流式作答 + 人工 5 档投票。

迭代十一：对齐 mcbench /battle/text 交互。与评测链路（executor/judge）解耦：
- 抽题：内置题库（QUESTION_POOL 全局池抽样）或自定义评测集，仅返回题面（不含
  expected/rubric，防作弊）；
- 流式：双路 SSE 逐 token 转发（OpenAI 兼容 stream），单侧失败降级不中断另一侧；
- 无状态：不落库、无持久化；投票与统计全部在前端会话内完成。
"""
from __future__ import annotations

import asyncio
import json
import random
import re
from typing import Any, AsyncGenerator

import httpx

from backend.ssrf import build_upstream_client
from backend.engine.tasks import QUESTION_POOL
from backend.engine.tasks import build_task_set_from_dataset
from backend.engine.datasets import DatasetValidationError

MAX_BATTLE_QUESTIONS = 50

# 流式请求超时：connect 10s；read 按 chunk 间隔计（180s）；write 30s
STREAM_TIMEOUT = httpx.Timeout(connect=10, read=180, write=30, pool=10)


def _chat_endpoint(url: str) -> str:
    """统一 OpenAI 兼容端点约定：/v1 地址自动拼接 /chat/completions。

    与评测/出题链路（executor/generator）一致；已带完整端点时不重复拼接。
    """
    u = (url or "").strip().rstrip("/")
    if not u:
        return u
    if not u.endswith("/chat/completions"):
        u += "/chat/completions"
    return u


def _upstream_error(status: int, detail: str) -> str:
    """上游 4xx/5xx 错误精简：HTML 页面只给状态码 + 提示，原文保留前 300 字符。"""
    if re.search(r"<\s*(?:html|!DOCTYPE)", (detail or "")[:300], re.I):
        return (f"HTTP {status}（目标地址不存在或不是 API 端点，"
                "请检查是否填写 OpenAI 兼容 /v1 地址）")
    if detail:
        return f"HTTP {status}: {detail}"
    return f"HTTP {status}"


def sample_battle_questions(
    count: int,
    source: str = "random",
    dataset_name: str | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """抽取对战题目（仅题面，不含期望答案）。

    Args:
        count: 抽取题数（1..MAX_BATTLE_QUESTIONS，超出按边界收敛）。
        source: random=内置题库全局池抽样；custom=指定自定义评测集抽样。
        dataset_name: custom 源必填；不存在抛 FileNotFoundError。
        seed: 随机种子（同 seed 可复现同题集）。

    Returns:
        [{question_id, category, prompt, context}]，保持抽样顺序。
    """
    count = max(1, min(int(count), MAX_BATTLE_QUESTIONS))
    if source == "custom":
        if not dataset_name:
            raise ValueError("自定义源必须提供 dataset_name")
        from backend.storage import load_dataset

        dataset = load_dataset(dataset_name)
        if dataset is None:
            raise FileNotFoundError(dataset_name)
        try:
            task_set = build_task_set_from_dataset(
                dataset, num_questions=count, seed=seed)
        except DatasetValidationError:
            raise
        tasks = task_set["tasks"]
    else:
        if seed is not None:
            random.seed(seed)
        pool: list[dict[str, Any]] = []
        for dim, items in QUESTION_POOL.items():
            for q in items:
                qq = dict(q)
                qq["dimension"] = qq.get("dimension") or dim
                pool.append(qq)
        tasks = random.sample(pool, min(count, len(pool))) if pool else []

    out: list[dict[str, Any]] = []
    for i, t in enumerate(tasks):
        out.append({
            "question_id": t.get("id") or f"Q{i + 1}",
            "category": t.get("dimension") or t.get("benchmark") or "通用",
            "prompt": (t.get("prompt") or "").strip(),
            "context": (t.get("context") or "").strip(),
        })
    return out


def _chat_payload(model: dict[str, Any], prompt: str, context: str) -> dict[str, Any]:
    """构造 OpenAI 兼容流式请求体（context 以参考文档块置于题目前）。"""
    messages: list[dict[str, Any]] = []
    if context:
        messages.append({"role": "system",
                         "content": "请结合下方参考文档作答，只能引用文档中的事实。"})
        messages.append({"role": "user", "content": f"【参考文档】\n{context}"})
    messages.append({"role": "user", "content": prompt})
    payload: dict[str, Any] = {
        "model": model.get("name") or "",
        "messages": messages,
        "temperature": float(model.get("temperature") if model.get("temperature") is not None else 0.7),
        "max_tokens": int(model.get("max_tokens") or 4096),
        "stream": True,
    }
    if model.get("top_p") is not None:
        payload["top_p"] = model["top_p"]
    return payload


async def _produce_side(
    client: httpx.AsyncClient,
    side: str,
    model: dict[str, Any],
    prompt: str,
    context: str,
    queue: asyncio.Queue,
) -> None:
    """单侧流式消费：逐 chunk 提取 delta 推入共享队列，结束后发 done/error。"""
    headers = {"Authorization": f"Bearer {model.get('key') or ''}",
               "Content-Type": "application/json"}
    payload = _chat_payload(model, prompt, context)
    try:
        async with client.stream(
            "POST", _chat_endpoint(model["url"]), json=payload, headers=headers,
            timeout=STREAM_TIMEOUT,
        ) as resp:
            if resp.status_code >= 400:
                try:
                    body = await resp.aread()
                    detail = body[:300].decode("utf-8", errors="replace")
                except Exception:
                    detail = ""
                await queue.put({"side": side, "error": _upstream_error(resp.status_code, detail)})
                return
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    delta = (json.loads(data).get("choices") or [{}])[0].get("delta", {}).get("content")
                except (json.JSONDecodeError, IndexError, AttributeError):
                    continue
                if delta:
                    await queue.put({"side": side, "delta": delta})
        await queue.put({"side": side, "done": True})
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        await queue.put({"side": side, "error": f"{type(exc).__name__}: {exc}"})


async def stream_battle(
    model_a: dict[str, Any],
    model_b: dict[str, Any],
    prompt: str,
    context: str = "",
) -> AsyncGenerator[dict[str, Any], None]:
    """双路并发流式对战：逐事件产出 {side: a|b, delta|done|error}。

    客户端中断（生成器被关闭/取消）时取消两侧任务并干净释放连接。
    """
    queue: asyncio.Queue = asyncio.Queue()
    async with build_upstream_client() as client:
        t1 = asyncio.create_task(_produce_side(client, "a", model_a, prompt, context, queue))
        t2 = asyncio.create_task(_produce_side(client, "b", model_b, prompt, context, queue))
        tasks = (t1, t2)
        try:
            terminal: set[str] = set()
            while len(terminal) < 2:
                evt = await queue.get()
                yield evt
                if evt.get("done") or evt.get("error"):
                    terminal.add(evt["side"])
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
