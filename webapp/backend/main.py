# -*- coding: utf-8 -*-
"""FastAPI 主服务：REST 路由 + 静态前端托管 + 内存进度追踪。

启动：uvicorn backend.main:app --host 127.0.0.1 --port 8910
"""
from __future__ import annotations

import asyncio
import json
import os
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
from backend.engine.report_builder import build_report
from backend.engine.parsers import get_parser, supported_extensions
from backend.engine.datasets import validate_json_dataset
from backend.storage import (
    create_job_id, save_config, save_task_set, save_answers,
    save_verdict, save_error, save_report, save_reveal, load_reveal,
    save_review, load_review,
    get_job_status, list_jobs, get_job_files,
    save_dataset, load_dataset, list_datasets, delete_dataset,
)
from backend.schemas import StartRequest, StartResponse, ReviewSubmission

app = FastAPI(title="模型对决评测平台", version="0.3.0")

_jobs: dict[str, dict] = {}

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/report.html", response_class=HTMLResponse)
async def report_page():
    return (FRONTEND_DIR / "report.html").read_text(encoding="utf-8")


@app.get("/review.html", response_class=HTMLResponse)
async def review_page():
    return (FRONTEND_DIR / "review.html").read_text(encoding="utf-8")


@app.get("/api/dims")
async def get_dims():
    return {"dims": DIMENSIONS}


# ---- 数据集管理 API ----

@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """上传评测集文件（JSON / CSV / Markdown / TXT），按扩展名路由解析。"""
    content = await file.read()
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

    # 用文件名（不含扩展名）作为数据集名称
    name = Path(filename).stem
    data["name"] = name
    save_dataset(name, data)

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
    body = await request.json()
    raw = body.get("content", "")
    if not raw:
        raise HTTPException(400, "缺少 content 字段")

    try:
        data = validate_json_dataset(raw)
    except ValueError as e:
        raise HTTPException(400, f"数据集格式错误: {e}")

    name = data.get("name", f"评测集_{int(time.time())}")
    save_dataset(name, data)

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
    return {"ok": True}


# ---- 连通性测试 ----

@app.post("/api/test-connection")
async def test_connection(config: StartRequest):
    import httpx
    results = {}
    for label, cfg in [("model_a", config.model_a), ("model_b", config.model_b)]:
        try:
            url = cfg.url.rstrip("/") + "/chat/completions"
            payload = {"model": cfg.name, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 8}
            headers = {"Authorization": f"Bearer {cfg.key}", "Content-Type": "application/json"}
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload, headers=headers, timeout=15)
                body = resp.json()
                if resp.status_code >= 400:
                    results[label] = {"ok": False, "error": body.get("error", {}).get("message", f"HTTP {resp.status_code}")}
                else:
                    usage = body.get("usage", {})
                    results[label] = {"ok": True, "model": body.get("model", cfg.name),
                                      "prompt_tokens": usage.get("prompt_tokens", 0),
                                      "completion_tokens": usage.get("completion_tokens", 0)}
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
    job_id = create_job_id()
    seed = req.seed if req.seed is not None else int(time.time()) % 100000

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
        }

    config_data = {
        "model_a": _build_model_config(req.model_a),
        "model_b": _build_model_config(req.model_b),
        "dims": req.dims,
        "seed": seed,
        "dataset_name": req.dataset_name,
        "repeat_n": req.repeat_n,
        "num_questions": req.num_questions,
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


def _finalize_job(job_id: str, verdict: dict, review_data: dict):
    """人工评审提交后：写 verdict/review/report，置为 completed。"""
    _jobs[job_id]["verdict"] = verdict
    save_verdict(job_id, verdict)
    save_review(job_id, review_data)

    cfg = _jobs[job_id]["config"]
    task_set = _jobs[job_id]["task_set"]
    report = build_report(
        cfg, task_set,
        _jobs[job_id]["answers_a"], _jobs[job_id]["answers_b"], verdict,
    )
    save_report(job_id, {
        "config": cfg,
        "tasks": task_set,
        "answers_a": _jobs[job_id]["answers_a"],
        "answers_b": _jobs[job_id]["answers_b"],
        "verdict": verdict,
        "report": report,
    })
    _jobs[job_id]["state"] = "completed"
    _jobs[job_id]["progress"] = "done"


# ---- 人工双盲评审 ----

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

    scores_by_round: dict[int, list[dict]] = {}
    for s in req.scores:
        scores_by_round.setdefault(s.round, []).append(s.model_dump())

    if not scores_by_round:
        raise HTTPException(400, "未提交任何打分")

    reveal = load_reveal(job_id) or make_reveal(repeat_n)
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
    _finalize_job(job_id, verdict, {"scores": [s.model_dump() for s in req.scores]})
    await _push_event(job_id, {"state": "completed"})
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


@app.get("/api/eval/{job_id}/report")
async def eval_report(job_id: str):
    files = get_job_files(job_id)
    if files is None:
        if job_id in _jobs:
            j = _jobs[job_id]
            if j["verdict"]:
                payload = {
                    "job_id": job_id, "config": j["config"], "tasks": j["task_set"],
                    "answers_a": j["answers_a"], "answers_b": j["answers_b"], "verdict": j["verdict"],
                }
                payload["report"] = build_report(
                    j["config"], j["task_set"], j["answers_a"], j["answers_b"], j["verdict"],
                )
                return payload
        raise HTTPException(404, "job not found or not completed")

    report_dict = files.get("report.json")
    verdict = files.get("verdict.json")
    tasks = files.get("tasks.json")
    answers_a = files.get("answers-a.json")
    answers_b = files.get("answers-b.json")
    cfg = files.get("config.json")

    # 旧记录（无 report 字段）时即时生成，保持历史可读
    rich = None
    if isinstance(report_dict, dict):
        rich = report_dict.get("report")
    if rich is None and verdict and tasks:
        try:
            rich = build_report(cfg, tasks, answers_a, answers_b, verdict)
        except Exception:
            rich = None

    return {
        "job_id": job_id,
        "config": cfg,
        "tasks": tasks,
        "answers_a": answers_a,
        "answers_b": answers_b,
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
