# -*- coding: utf-8 -*-
"""双模型并发执行器：向两个模型发送题目，采集 latency/tokens 指标，
D7 效率稳定性题自动 repeat N=2（第二次 temperature=0.0），
代码题跑沙箱验真，单题失败不中断整轮，失败重试 1 次（指数退避）。

迭代二：提示策略（cot/direct/fewshot）与语义向量采集前置到执行阶段
（生成式题 optional embedding，失败静默降级 n-gram，不影响整轮）。

API 契约：调用 OpenAI 兼容 POST {URL}/chat/completions。
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Awaitable, Callable

import httpx
from backend.ssrf import build_upstream_client
from backend.engine.embed import embed_texts, local_bge_embed, resolve_provider
from backend.engine.tasks import CODE_DIMENSION, STABILITY_DIMENSION

MAX_RETRIES = 1
RETRY_BASE_DELAY = 3.0

# 代码验真并发上限：避免多个 AppContainer/子进程同时压垮宿主
_CODE_SEM = asyncio.Semaphore(2)

# 默认提示策略：Zero-shot CoT（历史可比性由 config.json 落盘 prompt_strategy 自描述）
COT_SUFFIX = "\n\n请一步一步思考，并在最后给出明确、独立的结论。"


def _context_block(task: dict[str, Any]) -> str:
    """任务携带参考文档（RAG/上下文忠实性，迭代四）：非空时置于题目之前。"""
    ctx = (task.get("context") or "").strip()
    if not ctx:
        return ""
    return f"【参考文档】\n{ctx}\n\n【任务】\n"


def build_prompt(task: dict[str, Any], strategy: str = "cot") -> str:
    """按提示策略构造发给模型的最终 prompt（纯函数，确定性可测）。

    - cot（默认）：追加逐步思考指示；
    - direct：原样透传 prompt；
    - fewshot：注入 test_cases 示例（输入/输出对）；无 test_cases 时回退 direct
      （与 cot 互斥，按用户选择执行）。

    迭代四：任务携带 context 时，任何策略下均以「【参考文档】…【任务】」块
    置于题面之前（RAG 忠实性评测要求模型可被观察到引用上下文）。
    """
    s = (strategy or "cot").strip().lower()
    prompt = _context_block(task) + (task.get("prompt") or "").strip()
    if s == "direct":
        return prompt
    if s == "fewshot":
        cases = task.get("test_cases") or []
        if not cases:
            return prompt
        examples = "\n\n示例：\n" + "\n".join(
            f"输入：{c.get('input', '')}\n输出：{c.get('expected', '')}"
            for c in cases[:2]
        )
        return prompt + examples
    return prompt + COT_SUFFIX


async def _embed_pair(embedder: Any, raw_answer: str, expected: str) -> dict[str, Any] | None:
    """调用注入的 embedder 采集双向量；任何异常静默降级（不中断整轮评测）。"""
    try:
        vecs = await embedder([raw_answer, expected])
    except Exception:
        return None
    if not vecs or len(vecs) != 2:
        return None
    return {"vector": vecs[0], "ref_vector": vecs[1]}


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
    embedder: Awaitable | None = None,
) -> dict[str, Any]:
    """执行单个任务。config 含 url/key/name/temperature/max_tokens/top_p/code_verify_mode。

    embedder：可选 async 可调用（入参 [answer, expected] 文本列表，出参向量列表；
    None 或返回 None 时语义指标由报告层 n-gram 兜底）。
    """
    tid = task["id"]
    prompt = build_prompt(task, config.get("prompt_strategy", "cot"))
    is_code_task = task["dimension"] == CODE_DIMENSION
    is_stability_task = task["dimension"] == STABILITY_DIMENSION

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
    if (
        embedder is not None
        and raw_answer
        and task.get("type") == "生成式"
        and (task.get("expected") or "").strip()
    ):
        sem = await _embed_pair(embedder, raw_answer, str(task["expected"]))
        if sem:
            ans_entry["semantic"] = sem
    return ans_entry


async def _make_embedder(resolved: dict[str, Any], client: httpx.AsyncClient):
    """按已解析 provider 构造 embedder（async: list[str] -> list[list[float]] | None）。

    external：复用 SSRF 校验客户端；local_bge：线程池懒加载模型；
    offline / 有 error：None（报告层 n-gram 兜底）。
    """
    kind = resolved.get("kind")
    if kind == "external":
        cfg = resolved.get("cfg") or {}

        async def emb(texts: list[str]):
            return await embed_texts(kind, cfg, client, texts)

        return emb
    if kind == "local_bge":
        async def emb(texts: list[str]):
            return await asyncio.to_thread(local_bge_embed, texts)

        return emb
    return None


async def _execute_model(
    model_label: str,
    config: dict[str, Any],
    tasks: list[dict[str, Any]],
    stability_repeat: dict[str, int] | None,
    progress_cb=None,
    embedding_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """串行跑完该模型的全部题目。config 含 url/key/name/temperature/max_tokens/top_p。

    embedding_cfg：embedding provider 配置（None=不采集语义向量；失败自动降级）。
    """
    url = config["url"].rstrip("/")
    model_name = config["name"]

    answers: list[dict[str, Any]] = []
    total = len(tasks)
    # 统一走 SSRF 校验客户端：连接前重新解析并按公网性过滤（DNS 重绑定防护）
    async with build_upstream_client() as client:
        embedder = None
        if embedding_cfg is not None:
            embedder = await _make_embedder(resolve_provider(embedding_cfg), client)
        for i, task in enumerate(tasks):
            tid = task["id"]
            is_repeat_task = stability_repeat is not None and tid in stability_repeat
            repeat_n = stability_repeat.get(tid, 1) if is_repeat_task else 1

            if is_repeat_task and repeat_n > 1:
                ans1 = await execute_task(client, task, config, is_repeat_task=True, repeat_index=1,
                                          embedder=embedder)
                ans1["repeat_index"] = 1
                ans2 = await execute_task(client, task, config, is_repeat_task=True, repeat_index=2,
                                          embedder=embedder)
                ans2["repeat_index"] = 2
                answers.append(ans1)
                answers.append(ans2)
            else:
                ans = await execute_task(client, task, config, is_repeat_task=False,
                                         embedder=embedder)
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
    embedding_cfg: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """并发跑完两个模型的全部题目。config_a/b 含 url/key/name/temperature/max_tokens/top_p。

    embedding_cfg：语义向量采集配置（None=不采集）；prompt 策略由 config 内
    prompt_strategy 字段控制（executor 不单独入参）。
    """
    tasks = task_set["tasks"]
    stability_repeat = task_set.get("meta", {}).get("eval_flags", {}).get("stability_repeat")

    results = await asyncio.gather(
        _execute_model("A", config_a, tasks, stability_repeat, progress_cb, embedding_cfg),
        _execute_model("B", config_b, tasks, stability_repeat, progress_cb, embedding_cfg),
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
