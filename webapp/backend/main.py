# -*- coding: utf-8 -*-
"""FastAPI 主服务：REST 路由 + 静态前端托管 + 内存进度追踪。

启动：uvicorn backend.main:app --host 127.0.0.1 --port 8910
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import sys
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.engine.tasks import build_task_set, build_task_set_from_dataset, DIMENSIONS
from backend.engine.executor import execute_all
from backend.engine.human_review import (
    make_reveal, resolve_round, build_review_view,
    build_round_verdict, build_final_verdict,
)
from backend.engine.report_builder import build_report, reveal_answers
from backend.engine.parsers import get_parser, supported_extensions
from backend.engine.datasets import validate_json_dataset
from backend import audit
from backend.access import security_middleware
from backend.ssrf import validate_upstream_url, UpstreamUrlError
from backend.storage import (
    create_job_id, save_config, save_task_set, save_answers,
    save_verdict, save_error, save_report, save_reveal, load_reveal,
    save_review, load_review, save_round_verdicts,
    get_job_status, list_jobs, get_job_files,
    save_dataset, load_dataset, list_datasets, delete_dataset,
)
from backend.schemas import StartRequest, StartResponse, ReviewSubmission
from backend.security import redact_sensitive, sanitize_config

app = FastAPI(title="模型对决评测平台", version="0.3.0")
app.middleware("http")(security_middleware)

_jobs: dict[str, dict] = {}

# 资源限制（issue #8）：并发执行任务上限 / 上传大小 / 数据集题数
MAX_ACTIVE_JOBS = 2
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_DATASET_TASKS = 200


async def _read_body_limited(request: Request, limit: int) -> bytes:
    """流式读取请求体，累计超过 limit 立即 400（防大 body 先占满内存再拒绝）。

    Content-Length 存在时先做快速预检；分块传输（无 Content-Length）走流式截断。
    """
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > limit:
        raise HTTPException(400, f"请求体过大：最大 {limit // (1024 * 1024)}MB")
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            raise HTTPException(400, f"请求体过大：最大 {limit // (1024 * 1024)}MB")
        chunks.append(chunk)
    return b"".join(chunks)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# CSP：仅放行本站内联脚本（nonce 每请求轮换）；内联事件处理器一律禁用，
# 前端已改为 addEventListener 挂载。style-src 保留 'unsafe-inline'（页面内联样式）。
CSP_DIRECTIVES = (
    "default-src 'self'; "
    "script-src 'self' 'nonce-{nonce}'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
)


def _page_response(name: str) -> HTMLResponse:
    """托管前端页面并注入 CSP：为全部 <script> 附加请求级 nonce。"""
    html = (FRONTEND_DIR / name).read_text(encoding="utf-8")
    nonce = secrets.token_urlsafe(16)
    html = re.sub(r"<script\b", f'<script nonce="{nonce}"', html)
    return HTMLResponse(
        html,
        headers={"Content-Security-Policy": CSP_DIRECTIVES.format(nonce=nonce)},
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    return _page_response("index.html")


@app.get("/report.html", response_class=HTMLResponse)
async def report_page():
    return _page_response("report.html")


@app.get("/review.html", response_class=HTMLResponse)
async def review_page():
    return _page_response("review.html")


@app.get("/api/dims")
async def get_dims():
    return {"dims": DIMENSIONS}


# ---- 代码验真模式状态 ----

@app.get("/api/code-runner/status")
async def code_runner_status():
    """报告当前可用的代码验真模式（off 恒可用；native-sandbox 视环境而定）。"""
    from backend.engine.isolation.runners import MODES, get_runner

    modes = {}
    for m in MODES:
        runner = get_runner(m)
        available, detail = runner.is_available()
        modes[m] = {"available": available, "detail": detail}
    return {"default_mode": "off", "modes": modes}


# ---- 数据集管理 API ----

@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """上传评测集文件（JSON / CSV / Markdown / TXT），按扩展名路由解析。"""
    # 截断读取：最多读 5MB+1 字节，超限立即拒绝，不将大文件整体读入内存
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, f"文件过大：最大 {MAX_UPLOAD_BYTES // (1024 * 1024)}MB")
    filename = file.filename or "unknown"
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(400, "文件编码不是 UTF-8")

    ext = Path(filename).suffix.lower()
    parser = get_parser(ext)
    if parser is None:
        raise HTTPException(
            400,
            f"不支持的文件类型: {ext or '(无扩展名)'}，支持: {', '.join(supported_extensions())}",
        )

    try:
        data = parser(text)
    except ValueError as e:
        raise HTTPException(400, f"数据集格式错误: {e}")
    if len(data.get("tasks", [])) > MAX_DATASET_TASKS:
        raise HTTPException(400, f"题目数量超过上限 {MAX_DATASET_TASKS}")

    # 用文件名（不含扩展名）作为数据集名称
    name = Path(filename).stem
    data["name"] = name
    save_dataset(name, data)
    audit.dataset_uploaded(name)

    tasks = data.get("tasks", [])
    dims = list({t.get("dimension", "自定义") for t in tasks})
    return {
        "ok": True,
        "name": name,
        "task_count": len(tasks),
        "dimensions": dims,
        "description": data.get("description", ""),
    }


@app.post("/api/datasets/upload-json")
async def upload_dataset_json(request: Request):
    """通过 JSON body 上传评测集（用于文本框粘贴）。"""
    body_bytes = await _read_body_limited(request, MAX_JSON_BYTES)
    try:
        body = json.loads(body_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(400, "请求体不是合法 JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "缺少 content 字段")
    raw = body.get("content", "")
    if not isinstance(raw, str) or not raw:
        raise HTTPException(400, "缺少 content 字段")
    if len(raw) > MAX_JSON_BYTES:
        raise HTTPException(400, f"JSON 内容过大：最大 {MAX_JSON_BYTES // (1024 * 1024)}MB")

    try:
        data = validate_json_dataset(raw)
    except ValueError as e:
        raise HTTPException(400, f"数据集格式错误: {e}")
    if len(data.get("tasks", [])) > MAX_DATASET_TASKS:
        raise HTTPException(400, f"题目数量超过上限 {MAX_DATASET_TASKS}")

    name = data.get("name", f"评测集_{int(time.time())}")
    save_dataset(name, data)
    audit.dataset_uploaded(name)

    tasks = data.get("tasks", [])
    dims = list({t.get("dimension", "自定义") for t in tasks})
    return {
        "ok": True,
        "name": name,
        "task_count": len(tasks),
        "dimensions": dims,
        "description": data.get("description", ""),
    }


@app.get("/api/datasets")
async def get_datasets():
    return {"datasets": list_datasets()}


@app.delete("/api/datasets/{name}")
async def remove_dataset(name: str):
    ok = delete_dataset(name)
    if not ok:
        raise HTTPException(404, "dataset not found")
    audit.dataset_deleted(name)
    return {"ok": True}


# ---- 连通性测试 ----

@app.post("/api/test-connection")
async def test_connection(config: StartRequest):
    import httpx
    results = {}
    for label, cfg in [("model_a", config.model_a), ("model_b", config.model_b)]:
        try:
            # SSRF 防护：仅允许公网 http/https 目标（内网场景需显式开关放行）
            validate_upstream_url(cfg.url)
            url = cfg.url.rstrip("/") + "/chat/completions"
            payload = {"model": cfg.name, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 8}
            headers = {"Authorization": f"Bearer {cfg.key}", "Content-Type": "application/json"}
            async with httpx.AsyncClient(follow_redirects=False) as client:
                resp = await client.post(url, json=payload, headers=headers, timeout=15)
                body = resp.json()
                if resp.status_code >= 400:
                    results[label] = {"ok": False, "error": body.get("error", {}).get("message", f"HTTP {resp.status_code}")}
                else:
                    usage = body.get("usage", {})
                    results[label] = {"ok": True, "model": body.get("model", cfg.name),
                                      "prompt_tokens": usage.get("prompt_tokens", 0),
                                      "completion_tokens": usage.get("completion_tokens", 0)}
        except UpstreamUrlError as e:
            results[label] = {"ok": False, "error": f"目标地址校验失败: {e}"}
        except httpx.TimeoutException:
            results[label] = {"ok": False, "error": "连接超时（>15s）"}
        except httpx.ConnectError as e:
            results[label] = {"ok": False, "error": f"连接失败: {e}"}
        except Exception as e:
            results[label] = {"ok": False, "error": f"未知错误: {e}"}
    return results


# ---- 评测启动 ----

@app.post("/api/eval/start", response_model=StartResponse)
async def start_eval(req: StartRequest):
    """启动一轮评测。支持 dataset_name（自定义评测集）和 repeat_n（重复次数）。"""
    # SSRF 防护：评测启动即校验上游目标（executor 每次调用前不再重复解析）
    try:
        validate_upstream_url(req.model_a.url)
        validate_upstream_url(req.model_b.url)
    except UpstreamUrlError as e:
        raise HTTPException(400, f"模型 URL 校验失败: {e}")

    # 并发限制：执行中任务数（mock 不消耗外部资源，不计入）
    active = sum(
        1 for j in _jobs.values()
        if j.get("state") in ("pending", "executing")
        and not str(j.get("config", {}).get("model_a", {}).get("url", "")).startswith("mock")
    )
    if active >= MAX_ACTIVE_JOBS:
        raise HTTPException(429, f"当前已有 {active} 个评测任务在执行，请等待完成后再启动")

    job_id = create_job_id()
    seed = req.seed if req.seed is not None else int(time.time()) % 100000

    # 代码验真模式校验：native-sandbox 必须真实可用，避免静默跳过验真
    if req.code_verify_mode == "native-sandbox":
        from backend.engine.isolation.runners import get_runner

        available, detail = get_runner("native-sandbox").is_available()
        if not available:
            raise HTTPException(
                400,
                f"原生沙箱当前不可用：{detail}（可先运行 python -m scripts.sandbox_selfcheck 检查环境）",
            )

    # 构建任务集：自定义评测集 or 内置题库
    if req.dataset_name:
        dataset = load_dataset(req.dataset_name)
        if dataset is None:
            raise HTTPException(404, f"评测集 '{req.dataset_name}' 不存在")
        task_set = build_task_set_from_dataset(dataset)
    else:
        task_set = build_task_set(dims=req.dims, seed=seed, num_questions=req.num_questions)

    # 构建 config（含模型参数）
    def _build_model_config(mc) -> dict[str, Any]:
        return {
            "name": mc.name, "url": mc.url, "key": mc.key,
            "temperature": mc.temperature, "max_tokens": mc.max_tokens, "top_p": mc.top_p,
            "code_verify_mode": req.code_verify_mode,
        }

    config_data = {
        "model_a": _build_model_config(req.model_a),
        "model_b": _build_model_config(req.model_b),
        "dims": req.dims,
        "seed": seed,
        "dataset_name": req.dataset_name,
        "repeat_n": req.repeat_n,
        "num_questions": req.num_questions,
        "code_verify_mode": req.code_verify_mode,
    }
    save_config(job_id, config_data)
    save_task_set(job_id, task_set)

    _jobs[job_id] = {
        "state": "executing",
        "progress": "0/0",
        "task_set": task_set,
        "config": config_data,
        "answers_a": None,
        "answers_b": None,
        "verdict": None,
        "rounds_answers": [],
        "reveal": None,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sse_queue": asyncio.Queue(),
        "repeat_n": req.repeat_n,
    }

    asyncio.create_task(_run_eval(job_id))
    audit.eval_started(job_id)
    return StartResponse(job_id=job_id)


async def _push_event(job_id: str, event: dict):
    if job_id in _jobs:
        await _jobs[job_id]["sse_queue"].put(event)


def _mark_error(job_id: str, message: str):
    """将错误信息落盘，使历史记录可显示 error 状态。"""
    try:
        save_error(job_id, message)
    except Exception:
        pass


async def _run_eval(job_id: str):
    """后台执行：出题→调用双模型→生成 X/Y 盲评映射→等待人工评审。

    评审阶段由用户在前端打分（POST /api/eval/{id}/review），
    本协程在作答完成后即退出，不再调用 AI 评审。
    """
    cfg = _jobs[job_id]["config"]
    task_set = _jobs[job_id]["task_set"]
    total_tasks = task_set["meta"]["total"]
    repeat_n = _jobs[job_id].get("repeat_n", 1)

    all_rounds: list[dict] = []
    for round_idx in range(repeat_n):
        round_label = f"第{round_idx+1}/{repeat_n}轮" if repeat_n > 1 else ""

        _jobs[job_id]["state"] = "executing"
        _jobs[job_id]["progress"] = f"0/{total_tasks}"
        await _push_event(job_id, {"state": "executing", "progress": f"0/{total_tasks}", "round": round_label})

        async def progress_cb(label, done, total):
            _jobs[job_id]["progress"] = f"{done}/{total}"
            await _push_event(job_id, {"state": "executing", "progress": f"{done}/{total}", "round": round_label})

        try:
            answers_a, answers_b = await execute_all(
                task_set, config_a=cfg["model_a"], config_b=cfg["model_b"], progress_cb=progress_cb,
            )
        except Exception as e:
            _jobs[job_id]["state"] = "error"
            _jobs[job_id]["error"] = f"模型调用失败: {e}"
            await _push_event(job_id, {"state": "error", "error": str(e)})
            _mark_error(job_id, f"模型调用失败: {e}")
            return

        # 保存本轮答卷（round 文件 + 覆盖当前答卷）
        save_answers(job_id, f"a-r{round_idx+1}", answers_a)
        save_answers(job_id, f"b-r{round_idx+1}", answers_b)
        save_answers(job_id, "a", answers_a)
        save_answers(job_id, "b", answers_b)
        _jobs[job_id]["answers_a"] = answers_a
        _jobs[job_id]["answers_b"] = answers_b
        all_rounds.append({"a": answers_a, "b": answers_b})

    _jobs[job_id]["rounds_answers"] = all_rounds

    # 生成并持久化 X/Y 身份映射（重启不丢）
    reveal = make_reveal(repeat_n)
    save_reveal(job_id, reveal)
    _jobs[job_id]["reveal"] = reveal

    # 进入人工评审阶段，等待用户打分
    _jobs[job_id]["state"] = "reviewing"
    _jobs[job_id]["progress"] = "0/0"
    await _push_event(job_id, {"state": "reviewing"})


def _finalize_job(
    job_id: str,
    verdict: dict,
    review_data: dict,
    round_verdicts: list[dict] | None,
    cfg: dict,
    task_set: dict,
    answers_a: dict,
    answers_b: dict,
    rounds_answers: list[dict] | None = None,
):
    """人工评审提交后：写 verdict/review/round-verdicts/report，置为 completed。

    数据一律来自显式参数（由 _load_job_state 恢复），不依赖进程内 _jobs，
    服务重启后（_jobs 清空）磁盘态任务仍可完成提交闭环。
    """
    save_verdict(job_id, verdict)
    save_review(job_id, review_data)
    if round_verdicts:
        save_round_verdicts(job_id, round_verdicts)
    report = build_report(cfg, task_set, answers_a, answers_b, verdict, rounds_answers)
    save_report(job_id, {
        "config": sanitize_config(cfg),
        "tasks": task_set,
        "answers_a": answers_a,
        "answers_b": answers_b,
        "verdict": verdict,
        "report": report,
    })
    if job_id in _jobs:
        j = _jobs[job_id]
        j["verdict"] = verdict
        j["answers_a"] = answers_a
        j["answers_b"] = answers_b
        j["rounds_answers"] = rounds_answers or j.get("rounds_answers")
        j["round_verdicts"] = round_verdicts
        j["state"] = "completed"
        j["progress"] = "done"


# ---- 人工双盲评审 ----

def _validate_review_scores(
    task_set: dict, repeat_n: int, scores_by_round: dict[int, list[dict]]
) -> list[dict]:
    """校验评分集合（issue #9）：每个 (round, task_id) 恰好一次、题号归属任务集、轮次在 1..repeat_n。

    返回结构化错误列表（空列表=完整有效）：
      round_out_of_range: {"type","round"}       轮次不在 1..repeat_n
      unknown_task:       {"type","round","id"}  题号不在任务集
      duplicate_task:     {"type","round","id"}  同轮同题重复
      missing_task:       {"type","round","ids"} 该轮缺失题目
      missing_round:      {"type","round","ids"} 整轮缺失（含该轮全部题目）
    """
    task_ids = {t["id"] for t in task_set["tasks"]}
    expected = set(range(1, repeat_n + 1))
    errors: list[dict] = []
    for round_no, items in sorted(scores_by_round.items()):
        if round_no not in expected:
            errors.append({"type": "round_out_of_range", "round": round_no})
            continue
        seen: set[str] = set()
        for s in items:
            sid = s["id"]
            if sid not in task_ids:
                errors.append({"type": "unknown_task", "round": round_no, "id": sid})
            elif sid in seen:
                errors.append({"type": "duplicate_task", "round": round_no, "id": sid})
            else:
                seen.add(sid)
        missing = sorted(task_ids - seen)
        if missing:
            errors.append({"type": "missing_task", "round": round_no, "ids": missing})
    for round_no in sorted(expected - set(scores_by_round)):
        errors.append({"type": "missing_round", "round": round_no, "ids": sorted(task_ids)})
    return errors


def _load_job_state(job_id: str) -> tuple[dict, dict, dict, int] | None:
    """从内存或磁盘恢复评审所需状态（config/task_set/rounds_answers/repeat_n）。"""
    if job_id in _jobs:
        j = _jobs[job_id]
        return j["config"], j["task_set"], j["rounds_answers"], j.get("repeat_n", 1)
    files = get_job_files(job_id)
    if files is None:
        return None
    cfg = files.get("config.json") or {}
    task_set = files.get("tasks.json") or {}
    rounds: list[dict] = []
    repeat_n = int(cfg.get("repeat_n", 1))
    for r in range(1, repeat_n + 1):
        a = files.get(f"answers-a-r{r}.json")
        b = files.get(f"answers-b-r{r}.json")
        if a is None or b is None:
            break
        rounds.append({"a": a, "b": b})
    if not rounds:
        a = files.get("answers-a.json")
        b = files.get("answers-b.json")
        if a is not None and b is not None:
            rounds.append({"a": a, "b": b})
    if not task_set or not rounds:
        return None
    return cfg, task_set, rounds, repeat_n


@app.get("/api/eval/{job_id}/review")
async def eval_review_view(job_id: str):
    """返回人工评审页数据：题目 + 答案X/答案Y（模型身份完全隐藏）。"""
    restored = _load_job_state(job_id)
    if restored is None:
        raise HTTPException(404, "job not found")
    cfg, task_set, rounds_answers, repeat_n = restored

    reveal = load_reveal(job_id) or make_reveal(repeat_n)
    rounds_view = []
    for r_idx, round_ans in enumerate(rounds_answers):
        x_model, y_model, x_pool, y_pool = resolve_round(
            reveal, r_idx, round_ans["a"], round_ans["b"]
        )
        items = [
            build_review_view(t, x_pool.get(t["id"], []), y_pool.get(t["id"], []))
            for t in task_set["tasks"]
        ]
        rounds_view.append({"round": r_idx + 1, "items": items})

    return {
        "job_id": job_id,
        "repeat_n": repeat_n,
        "total_questions": len(task_set["tasks"]),
        "rounds": rounds_view,
        "submitted": load_review(job_id) is not None,
    }


@app.post("/api/eval/{job_id}/review", response_model=StartResponse)
async def eval_review_submit(job_id: str, req: ReviewSubmission):
    """提交人工打分：按轮构建 verdict → 聚合 → 生成报告 → completed。"""
    restored = _load_job_state(job_id)
    if restored is None:
        raise HTTPException(404, "job not found")
    cfg, task_set, rounds_answers, repeat_n = restored

    if load_review(job_id) is not None:
        raise HTTPException(409, "该任务已提交评分，请勿重复提交")

    scores_by_round: dict[int, list[dict]] = {}
    for s in req.scores:
        scores_by_round.setdefault(s.round, []).append(s.model_dump())

    if not scores_by_round:
        raise HTTPException(400, "未提交任何打分")

    # 完整性/唯一性/归属校验（issue #9）：任一 (round, task_id) 缺失、重复、
    # 未知题号或轮次越界，整个请求 400 拒绝，不写任何文件、不改任务状态
    errors = _validate_review_scores(task_set, repeat_n, scores_by_round)
    if errors:
        raise HTTPException(400, {
            "message": f"评分集合不完整或存在冲突（{len(errors)} 处），请核对后重新提交",
            "errors": errors,
        })

    reveal = load_reveal(job_id)
    if reveal is None:
        reveal = make_reveal(repeat_n)
        save_reveal(job_id, reveal)
    round_verdicts = []
    for r_idx, round_ans in enumerate(rounds_answers):
        x_model, y_model, x_pool, y_pool = resolve_round(
            reveal, r_idx, round_ans["a"], round_ans["b"]
        )
        round_reveal = reveal["rounds"][r_idx] if r_idx < len(reveal["rounds"]) else {"answer_x": "a", "answer_y": "b"}
        round_scores = scores_by_round.get(r_idx + 1, [])
        v = build_round_verdict(
            task_set, round_scores, round_reveal, x_model, y_model, r_idx,
        )
        round_verdicts.append(v)

    verdict = build_final_verdict(round_verdicts, repeat_n)
    answers_a, answers_b = rounds_answers[-1]["a"], rounds_answers[-1]["b"]
    _finalize_job(
        job_id, verdict, {"scores": [s.model_dump() for s in req.scores]},
        round_verdicts, cfg, task_set, answers_a, answers_b, rounds_answers,
    )
    await _push_event(job_id, {"state": "completed"})
    audit.review_submitted(job_id)
    return StartResponse(job_id=job_id)


# ---- 模拟评测 ----

@app.post("/api/eval/mock")
async def mock_eval():
    """生成一条模拟评测记录（无需真实 API）。

    与真实评测一致：先生成任务集与双模型答卷进入 reviewing 状态，
    由用户在评审页打分，提交后才生成 verdict 与报告。
    """
    from backend.engine.mock import prepare_mock_job
    data = prepare_mock_job()
    job_id = data["job_id"]
    _jobs[job_id] = {
        "state": "reviewing",
        "progress": "0/0",
        "task_set": data["task_set"],
        "config": data["config"],
        "answers_a": data["answers_a"],
        "answers_b": data["answers_b"],
        "verdict": None,
        "rounds_answers": data["rounds_answers"],
        "reveal": data["reveal"],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sse_queue": asyncio.Queue(),
        "repeat_n": data["repeat_n"],
    }
    await _push_event(job_id, {"state": "reviewing"})
    audit.eval_started(job_id, actor="mock")
    return {"job_id": job_id, "mock": True}


# ---- 状态 / 报告 / 历史 ----

@app.get("/api/eval/{job_id}/status")
async def eval_status(job_id: str):
    if job_id not in _jobs:
        j = get_job_status(job_id)
        if j is None:
            raise HTTPException(404, "job not found")
        return j
    j = _jobs[job_id]
    return {
        "job_id": job_id, "state": j["state"], "progress": j.get("progress"),
        "model_a": j["config"]["model_a"]["name"], "model_b": j["config"]["model_b"]["name"],
        "created_at": j.get("created_at"), "verdict": j.get("verdict"),
    }


@app.get("/api/eval/{job_id}/events")
async def eval_events(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "job not found")
    queue = _jobs[job_id]["sse_queue"]

    async def stream():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30)
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                    continue
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("state") in ("completed", "error"):
                    break
        except asyncio.CancelledError:
            pass

    return StreamingResponse(stream(), media_type="text/event-stream")


def _rounds_view(
    rounds_answers: list[dict],
    reveal: dict | None,
) -> list[dict]:
    """把全部轮次答卷按各轮 reveal 归一化为 answers_x/answers_y（供前端按轮展示）。"""
    result = []
    for r_idx, round_ans in enumerate(rounds_answers):
        x_model, y_model, x_pool, y_pool = resolve_round(
            reveal, r_idx, round_ans["a"], round_ans["b"]
        )
        result.append({
            "round": r_idx + 1,
            "answers_x": {
                "model": x_model,
                "answers": [e for entries in x_pool.values() for e in entries],
            },
            "answers_y": {
                "model": y_model,
                "answers": [e for entries in y_pool.values() for e in entries],
            },
        })
    return result


def _round_scores_view(round_verdicts: list[dict] | None) -> list[dict]:
    """每轮每题原始打分（供报告逐题表展开每轮分数明细）。"""
    result = []
    for r_idx, rv in enumerate(round_verdicts or []):
        scores = [
            {
                "id": s.get("id", ""),
                "answer_x": s.get("answer_x", 0),
                "answer_y": s.get("answer_y", 0),
                "winner": s.get("winner", "tie"),
            }
            for s in rv.get("scores", [])
        ]
        result.append({"round": r_idx + 1, "scores": scores})
    return result


@app.get("/api/eval/{job_id}/report")
async def eval_report(job_id: str):
    files = get_job_files(job_id)
    if files is None:
        if job_id in _jobs:
            j = _jobs[job_id]
            if j["verdict"]:
                rounds_answers = j.get("rounds_answers") or [
                    {"a": j["answers_a"], "b": j["answers_b"]}
                ]
                answers_x, answers_y = reveal_answers(
                    j["answers_a"], j["answers_b"], j["verdict"],
                )
                payload = {
                    "job_id": job_id, "config": j["config"], "tasks": j["task_set"],
                    "answers_a": j["answers_a"], "answers_b": j["answers_b"],
                    "answers_x": answers_x, "answers_y": answers_y,
                    "rounds": _rounds_view(rounds_answers, j.get("reveal")),
                    "round_scores": _round_scores_view(j.get("round_verdicts")),
                    "verdict": j["verdict"],
                }
                payload["report"] = build_report(
                    j["config"], j["task_set"], j["answers_a"], j["answers_b"],
                    j["verdict"], rounds_answers,
                )
                return redact_sensitive(payload)
        raise HTTPException(404, "job not found or not completed")

    report_dict = files.get("report.json")
    verdict = files.get("verdict.json")
    tasks = files.get("tasks.json")
    answers_a = files.get("answers-a.json")
    answers_b = files.get("answers-b.json")
    cfg = files.get("config.json")

    # 从磁盘恢复全部轮次答卷（多轮时 answers-a-r{n}.json；旧记录回退单轮）
    rounds_answers: list[dict] = []
    repeat_n = int((cfg or {}).get("repeat_n", 1))
    for r in range(1, max(repeat_n, 1) + 1):
        ra = files.get(f"answers-a-r{r}.json")
        rb = files.get(f"answers-b-r{r}.json")
        if ra is None or rb is None:
            break
        rounds_answers.append({"a": ra, "b": rb})
    if not rounds_answers:
        rounds_answers = [{"a": answers_a, "b": answers_b}]

    # 旧记录（无 report 字段）时即时生成，保持历史可读
    rich = None
    if isinstance(report_dict, dict):
        rich = report_dict.get("report")
    if rich is None and verdict and tasks:
        try:
            rich = build_report(cfg, tasks, answers_a, answers_b, verdict, rounds_answers)
        except Exception:
            rich = None

    answers_x, answers_y = reveal_answers(answers_a, answers_b, verdict)
    reveal = load_reveal(job_id)
    return {
        "job_id": job_id,
        "config": cfg,
        "tasks": tasks,
        "answers_a": answers_a,
        "answers_b": answers_b,
        "answers_x": answers_x,
        "answers_y": answers_y,
        "rounds": _rounds_view(rounds_answers, reveal),
        "round_scores": _round_scores_view(files.get("round-verdicts.json")),
        "verdict": verdict,
        "report": rich,
    }


@app.delete("/api/history/{job_id}")
async def delete_history(job_id: str):
    from backend.storage import delete_job
    if job_id in _jobs:
        _jobs.pop(job_id)
    ok = delete_job(job_id)
    if not ok:
        raise HTTPException(404, "job not found")
    audit.history_deleted(job_id)
    return {"ok": True}


@app.get("/api/history")
async def history():
    return {"jobs": list_jobs()}


@app.get("/api/history/{job_id}")
async def history_detail(job_id: str):
    files = get_job_files(job_id)
    if files is None:
        raise HTTPException(404, "job not found")
    return files
