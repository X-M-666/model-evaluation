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
from backend.engine.judge import run_judge
from backend.engine.parsers import get_parser, supported_extensions
from backend.storage import (
    create_job_id, save_config, save_task_set, save_answers,
    save_verdict, save_error, get_job_status, list_jobs, get_job_files,
    save_dataset, load_dataset, list_datasets, delete_dataset,
)
from backend.schemas import StartRequest, StartResponse

app = FastAPI(title="模型对决评测平台", version="0.2.0")

_jobs: dict[str, dict] = {}

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/report.html", response_class=HTMLResponse)
async def report_page():
    return (FRONTEND_DIR / "report.html").read_text(encoding="utf-8")


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
            payload = {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 8}
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
        task_set = build_task_set(dims=req.dims, seed=seed)

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
    """后台执行：出题→调用双模型→评审→写文件。支持 repeat_n 重复。"""
    cfg = _jobs[job_id]["config"]
    task_set = _jobs[job_id]["task_set"]
    total_tasks = task_set["meta"]["total"]
    repeat_n = _jobs[job_id].get("repeat_n", 1)

    all_rounds_a: list[dict] = []
    all_rounds_b: list[dict] = []
    all_rounds_verdicts: list[dict] = []

    for round_idx in range(repeat_n):
        round_label = f"第{round_idx+1}/{repeat_n}轮" if repeat_n > 1 else ""

        # 阶段1：调用双模型
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

        all_rounds_a.append(answers_a)
        all_rounds_b.append(answers_b)

        # 保存最后一轮（或唯一一轮）的答卷
        save_answers(job_id, "a", answers_a)
        save_answers(job_id, "b", answers_b)
        _jobs[job_id]["answers_a"] = answers_a
        _jobs[job_id]["answers_b"] = answers_b

        # 阶段2：评审
        _jobs[job_id]["state"] = "judging"
        _jobs[job_id]["progress"] = "0/0"
        await _push_event(job_id, {"state": "judging", "round": round_label})

        judge_config = {
            "url": cfg["model_a"]["url"],
            "key": cfg["model_a"]["key"],
            "name": os.environ.get("JUDGE_MODEL_NAME", "opencode/big-pickle"),
        }

        try:
            verdict = await run_judge(task_set, answers_a, answers_b, judge_config)
        except Exception as e:
            _jobs[job_id]["state"] = "error"
            _jobs[job_id]["error"] = f"评审失败: {e}"
            await _push_event(job_id, {"state": "error", "error": str(e)})
            _mark_error(job_id, f"评审失败: {e}")
            return

        all_rounds_verdicts.append(verdict)
        save_verdict(job_id, verdict)
        _jobs[job_id]["verdict"] = verdict

    # 如果 repeat_n > 1，汇总平均分
    if repeat_n > 1:
        final_verdict = _aggregate_rounds(all_rounds_verdicts, repeat_n)
        save_verdict(job_id, final_verdict)
        _jobs[job_id]["verdict"] = final_verdict

    _jobs[job_id]["state"] = "completed"
    await _push_event(job_id, {"state": "completed"})

    # 落盘完整报告，供历史/重启后读取
    try:
        from backend.storage import save_report
        save_report(job_id, {
            "config": cfg,
            "tasks": task_set,
            "answers_a": _jobs[job_id]["answers_a"],
            "answers_b": _jobs[job_id]["answers_b"],
            "verdict": _jobs[job_id]["verdict"],
        })
    except Exception:
        pass


def _aggregate_rounds(verdicts: list[dict], repeat_n: int) -> dict:
    """汇总多轮评测的平均分和标准差。"""
    import statistics

    last = verdicts[-1]
    scores_map: dict[str, list[dict]] = {}  # task_id -> [score_per_round]

    for v in verdicts:
        for s in v.get("scores", []):
            tid = s["id"]
            if tid not in scores_map:
                scores_map[tid] = []
            scores_map[tid].append(s)

    avg_scores = []
    for tid, round_scores in scores_map.items():
        x_vals = [s["answer_x"] for s in round_scores]
        y_vals = [s["answer_y"] for s in round_scores]
        avg_scores.append({
            "id": tid,
            "dimension": round_scores[0]["dimension"],
            "answer_x": round(statistics.mean(x_vals), 2),
            "answer_y": round(statistics.mean(y_vals), 2),
            "answer_x_std": round(statistics.stdev(x_vals), 2) if len(x_vals) > 1 else 0,
            "answer_y_std": round(statistics.stdev(y_vals), 2) if len(y_vals) > 1 else 0,
            "winner": "tie" if abs(statistics.mean(x_vals) - statistics.mean(y_vals)) < 0.01 else (
                "answer_x" if statistics.mean(x_vals) > statistics.mean(y_vals) else "answer_y"
            ),
            "basis": f"{repeat_n}轮平均（原始{len(round_scores)}轮数据）",
            "arbiter_note": "",
        })

    dim_totals: dict[str, dict] = {}
    for s in avg_scores:
        dim = s["dimension"]
        if dim not in dim_totals:
            dim_totals[dim] = {"x": 0, "y": 0}
        dim_totals[dim]["x"] += s["answer_x"]
        dim_totals[dim]["y"] += s["answer_y"]

    total_x = round(sum(d["x"] for d in dim_totals.values()), 2)
    total_y = round(sum(d["y"] for d in dim_totals.values()), 2)

    return {
        "meta": {"total": len(avg_scores), "valid": len(avg_scores), "invalid": 0,
                 "tie_arbitrated": 0, "repeat_n": repeat_n},
        "scores": avg_scores,
        "per_dimension": dim_totals,
        "totals": {"answer_x": total_x, "answer_y": total_y},
        "revealed": last.get("revealed", {}),
        "conclusion": f"经过{repeat_n}轮评测取平均",
        "winner_model": last.get("winner_model", "tie"),
    }


# ---- 模拟评测 ----

@app.post("/api/eval/mock")
async def mock_eval():
    """生成一条模拟评测记录（无需真实 API），用于演示报告页。"""
    from backend.engine.mock import generate_mock_job
    job_id = generate_mock_job()
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
                return {
                    "job_id": job_id, "config": j["config"], "tasks": j["task_set"],
                    "answers_a": j["answers_a"], "answers_b": j["answers_b"], "verdict": j["verdict"],
                }
        raise HTTPException(404, "job not found or not completed")
    return {
        "job_id": job_id,
        "config": files.get("config.json"),
        "tasks": files.get("tasks.json"),
        "answers_a": files.get("answers-a.json"),
        "answers_b": files.get("answers-b.json"),
        "verdict": files.get("verdict.json"),
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
