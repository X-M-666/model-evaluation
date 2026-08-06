# -*- coding: utf-8 -*-
"""双模型并发执行器：向两个模型发送题目，采集 latency/tokens 指标，
D7 效率稳定性题自动 repeat N=2（第二次 temperature=0.0），
代码题跑沙箱验真，单题失败不中断整轮，失败重试 1 次（指数退避）。

API 契约：调用 OpenAI 兼容 POST {URL}/chat/completions。
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
from backend.ssrf import build_upstream_client

MAX_RETRIES = 1
RETRY_BASE_DELAY = 3.0

# 代码验真并发上限：避免多个 AppContainer/子进程同时压垮宿主
_CODE_SEM = asyncio.Semaphore(2)


async def _build_code_verify_safe(mode: str, raw_answer: str, task: dict[str, Any]) -> dict[str, Any]:
    """带信号量与异常兜底地构建 code_verify（沙箱异常不中断整轮评测）。"""
    from backend.engine.isolation.runners import build_code_verify

    async with _CODE_SEM:
        try:
            return await asyncio.to_thread(build_code_verify, mode, raw_answer, task)
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "reason": f"执行异常（可能已部分执行）: {exc}"}


async def _call_one(
    client: httpx.AsyncClient,
    url: str,
    api_key: str,
    prompt: str,
    model_name: str = "",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    top_p: float | None = None,
) -> dict[str, Any]:
    """向单个模型发请求并返回 api_info 结构。"""
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if top_p is not None:
        payload["top_p"] = top_p

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    latency_ms = 0
    result: dict[str, Any] = {
        "status": "error", "attempts": 1, "truncated": False, "error": None,
        "latency_ms": 0, "prompt_tokens": 0, "completion_tokens": 0, "repeat_index": 1,
    }
    last_err: str | None = None
    for attempt in range(1 + MAX_RETRIES):
        try:
            t0 = time.perf_counter()
            resp = await client.post(url, json=payload, headers=headers, timeout=120)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            body = resp.json()
            if resp.status_code >= 400:
                last_err = body.get("error", {}).get("message") or f"HTTP {resp.status_code}"
                continue
            usage = body.get("usage", {})
            raw_answer = body["choices"][0]["message"]["content"] or ""
            finish = body["choices"][0].get("finish_reason", "")
            result.update(
                status="ok", error=None, latency_ms=latency_ms,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                truncated=(finish == "length"),
            )
            return raw_answer, result
        except httpx.TimeoutException:
            last_err = "请求超时（>120s）"
            latency_ms = 120000
            continue
        except httpx.HTTPError as exc:
            last_err = f"HTTP 连接错误: {exc}"
            continue
        except Exception as exc:
            last_err = f"未知错误: {exc}"
            continue
    result["error"] = last_err
    result["latency_ms"] = latency_ms
    result["attempts"] = 1 + MAX_RETRIES
    return None, result


async def execute_task(
    client: httpx.AsyncClient,
    task: dict[str, Any],
    config: dict[str, Any],
    is_repeat_task: bool,
    repeat_index: int = 1,
) -> dict[str, Any]:
    """执行单个任务。config 含 url/key/name/temperature/max_tokens/top_p/code_verify_mode。"""
    tid = task["id"]
    prompt = task["prompt"]
    is_code_task = task["dimension"] == "代码能力"
    is_stability_task = task["dimension"] == "效率与稳定性"

    temperature = config.get("temperature", 0.7)
    max_tokens = config.get("max_tokens", 4096)
    top_p = config.get("top_p")
    if is_code_task:
        max_tokens = max(max_tokens, 8192)

    if is_repeat_task and is_stability_task:
        temperature = config.get("temperature", 0.7) if repeat_index == 1 else 0.0

    raw_answer, api_info = await _call_one(
        client, config["url"].rstrip("/") + "/chat/completions", config["key"], prompt,
        model_name=config["name"], temperature=temperature, max_tokens=max_tokens, top_p=top_p,
    )
    api_info["repeat_index"] = repeat_index

    code_verify = None
    if is_code_task and raw_answer is not None and task.get("test_cases"):
        mode = config.get("code_verify_mode", "off")
        code_verify = await _build_code_verify_safe(mode, raw_answer, task)

    ans_entry: dict[str, Any] = {
        "id": tid, "raw_answer": raw_answer or "", "api_info": api_info,
    }
    if code_verify is not None:
        ans_entry["code_verify"] = code_verify
    return ans_entry


async def _execute_model(
    model_label: str,
    config: dict[str, Any],
    tasks: list[dict[str, Any]],
    stability_repeat: dict[str, int] | None,
    progress_cb=None,
) -> dict[str, Any]:
    """串行跑完该模型的全部题目。config 含 url/key/name/temperature/max_tokens/top_p。"""
    url = config["url"].rstrip("/")
    model_name = config["name"]

    answers: list[dict[str, Any]] = []
    total = len(tasks)
    # 统一走 SSRF 校验客户端：连接前重新解析并按公网性过滤（DNS 重绑定防护）
    async with build_upstream_client() as client:
        for i, task in enumerate(tasks):
            tid = task["id"]
            is_repeat_task = stability_repeat is not None and tid in stability_repeat
            repeat_n = stability_repeat.get(tid, 1) if is_repeat_task else 1

            if is_repeat_task and repeat_n > 1:
                ans1 = await execute_task(client, task, config, is_repeat_task=True, repeat_index=1)
                ans1["repeat_index"] = 1
                ans2 = await execute_task(client, task, config, is_repeat_task=True, repeat_index=2)
                ans2["repeat_index"] = 2
                answers.append(ans1)
                answers.append(ans2)
            else:
                ans = await execute_task(client, task, config, is_repeat_task=False)
                answers.append(ans)

            if progress_cb:
                await progress_cb(model_label, i + 1, total)

    return {
        "model": model_name,
        "api": {"name": model_name, "url": config["url"]},
        "note": f"webapp executor: {model_label}",
        "answers": answers,
    }


async def execute_all(
    task_set: dict[str, Any],
    config_a: dict[str, Any],
    config_b: dict[str, Any],
    progress_cb=None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """并发跑完两个模型的全部题目。config_a/b 含 url/key/name/temperature/max_tokens/top_p。"""
    tasks = task_set["tasks"]
    stability_repeat = task_set.get("meta", {}).get("eval_flags", {}).get("stability_repeat")

    results = await asyncio.gather(
        _execute_model("A", config_a, tasks, stability_repeat, progress_cb),
        _execute_model("B", config_b, tasks, stability_repeat, progress_cb),
        return_exceptions=True,
    )
    if isinstance(results[0], Exception):
        results[0] = _empty_answers(config_a, str(results[0]), tasks)
    if isinstance(results[1], Exception):
        results[1] = _empty_answers(config_b, str(results[1]), tasks)
    return results[0], results[1]


def _empty_answers(config: dict[str, Any], error: str, tasks: list) -> dict:
    return {
        "model": config["name"],
        "api": {"name": config["name"], "url": config["url"]},
        "note": f"webapp executor: 整体异常 — {error}",
        "answers": [
            {
                "id": t["id"], "raw_answer": "",
                "api_info": {
                    "status": "error", "attempts": 0, "truncated": False,
                    "error": error, "latency_ms": 0, "prompt_tokens": 0,
                    "completion_tokens": 0, "repeat_index": 1,
                },
            }
            for t in tasks
        ],
    }
