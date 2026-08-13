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
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.engine.tasks import build_task_set, build_task_set_from_dataset, DIMENSIONS
from backend.engine.executor import execute_all
from backend.engine.budget import check_budget
from backend.engine.judge import (
    run_judge, make_task_reveal, _normalize_task_reveal, health_check,
)
from backend.engine.human_review import (
    make_reveal, resolve_round, build_review_view,
    build_round_verdict, build_final_verdict, merge_hybrid_verdicts,
)
from backend.engine.report_builder import build_report, reveal_answers
from backend.engine.parsers import get_parser, supported_extensions
from backend.engine.datasets import (
    _as_str,
    MAX_NAME_LEN,
    DatasetValidationError,
    validate_json_dataset,
)
from backend import audit
from backend import sse_ticket
from backend.access import security_middleware
from backend.ssrf import build_upstream_client, validate_upstream_url, UpstreamUrlError
from backend.storage import (
    create_job_id, save_config, save_task_set, save_answers,
    save_verdict, save_error, save_report, save_reveal, load_reveal,
    save_review, load_review, save_round_verdicts, load_round_verdicts,
    get_job_status, list_jobs, get_job_files,
    save_dataset, load_dataset, list_datasets, delete_dataset,
    update_saturation, get_saturation,
    is_valid_job_id,
    list_gold, save_gold, delete_gold, load_gold, save_hybrid_review,
    load_hybrid_review,
)
from backend.gold import ensure_demo_gold, compute_meta_eval
from backend.schemas import (
    StartRequest, StartResponse, ReviewSubmission, ModelRegisterRequest,
    GoldSetRequest,
)
from backend.security import redact_sensitive, sanitize_config
from backend.models_registry import (
    delete_model, get_key, get_model, list_models, register, ModelRegistryError,
)

# 资源限制（issue #8）：并发执行任务上限 / 上传大小（数据集题数上限在 datasets.py 校验层统一生效）
MAX_ACTIVE_JOBS = 2
# 迭代一：context 字段 32KB×200 题的理论上界 6.4MB，上限提升至 10MB
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_JSON_BYTES = 2 * 1024 * 1024

# 任务生命周期（issue #14 / R2-005）：运行中评测的删除必须先取消后台任务
CANCEL_GRACE_SECONDS = 5
TERMINAL_STATES = ("completed", "error", "cancelled")


class JobCancelled(asyncio.CancelledError):
    """协作取消标记：_run_eval 在持久化/状态转换检查点抛出。

    绝不从 execute_all 内部（progress_cb 等）抛出——asyncio.gather 的
    return_exceptions=True 会吞掉子协程的 CancelledError 并作为返回值返回。
    """


def _require_job_id(job_id: str) -> str:
    """校验路由 job_id 为系统生成格式（issue #17：URL 编码 %2E%2E 解码后
    可进入 {job_id}，未校验会拼入 BASE_DIR 造成路径穿越删除整个 .eval）。"""
    if not is_valid_job_id(job_id):
        raise HTTPException(400, "invalid job_id format")
    return job_id


def _task_done(job_id: str, task: asyncio.Task):
    """后台任务结束回调：回收引用并消费异常，避免未获取 Task 异常告警。"""
    _tasks.pop(job_id, None)
    if task.cancelled():
        return
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        return
    if exc is not None:
        print(f"[eval] job {job_id} 后台任务异常: {exc}", file=sys.stderr)


async def _shutdown_cancel_all():
    """服务关闭：统一取消并回收全部后台任务。"""
    pending = list(_tasks.values())
    for t in pending:
        t.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    _tasks.clear()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_demo_gold()  # 迭代三：gold 目录为空时载入 demo 金标（source="demo"）
    yield
    await _shutdown_cancel_all()


app = FastAPI(title="模型对决评测平台", version="0.3.0", lifespan=lifespan)
app.middleware("http")(security_middleware)

_jobs: dict[str, dict] = {}
_tasks: dict[str, asyncio.Task] = {}


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
async def code_runner_status(request: Request):
    """报告当前平台的代码验真能力（issue #11 复审 R2-008）。

    probe 为组件存在性快检（请求路径）；selfcheck 会真实创建受限进程，
    较重，仅按 ?selfcheck=1 显式触发（自检脚本 / CI 用）。
    """
    from backend.engine.isolation import windows_native
    from backend.engine.isolation.runners import MODES, get_runner

    modes = {}
    for m in MODES:
        runner = get_runner(m)
        available, detail = runner.is_available()
        modes[m] = {"available": available, "detail": detail}
    probe_ok, probe_detail = windows_native.probe()
    result = {
        "default_mode": "off",
        "platform": sys.platform,
        "modes": modes,
        "native": {"probe_ok": probe_ok, "probe_detail": probe_detail},
    }
    if request.query_params.get("selfcheck") == "1":
        selfcheck_ok, selfcheck_detail = windows_native.selfcheck()
        result["native"]["selfcheck_ok"] = selfcheck_ok
        result["native"]["selfcheck_detail"] = selfcheck_detail
    return result


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

    # 用文件名（不含扩展名）作为数据集名称。名称覆盖发生在解析校验之后，
    # 须对 stem 重新做名称校验：超长/非规范 stem 直接 400，避免上传成功
    # 却在启动评测时被拒的体验不一致（issue #15）；空/纯空白 stem 回退解析器名。
    stem = Path(filename).stem
    try:
        name = _as_str(stem, "name", max_len=MAX_NAME_LEN)
    except DatasetValidationError as e:
        raise HTTPException(400, f"数据集格式错误: {e}")
    name = name or data.get("name") or "dataset"
    data["name"] = name
    save_dataset(name, data)
    audit.dataset_uploaded(name)

    tasks = data.get("tasks", [])
    dims = list({t.get("dimension", "自定义") for t in tasks})
    type_counts: dict[str, int] = {}
    for t in tasks:
        ttype = t.get("type", "判别式")
        type_counts[ttype] = type_counts.get(ttype, 0) + 1
    return {
        "ok": True,
        "name": name,
        "task_count": len(tasks),
        "dimensions": dims,
        "type_counts": type_counts,
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

    name = data.get("name", f"评测集_{int(time.time())}")
    save_dataset(name, data)
    audit.dataset_uploaded(name)

    tasks = data.get("tasks", [])
    dims = list({t.get("dimension", "自定义") for t in tasks})
    type_counts: dict[str, int] = {}
    for t in tasks:
        ttype = t.get("type", "判别式")
        type_counts[ttype] = type_counts.get(ttype, 0) + 1
    return {
        "ok": True,
        "name": name,
        "task_count": len(tasks),
        "dimensions": dims,
        "type_counts": type_counts,
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


# ---- 金标集（迭代三） ----

@app.get("/api/gold")
async def gold_list():
    """金标集列表（含 source=demo/manual，供前端标注）。"""
    return {"gold": list_gold()}


@app.post("/api/gold")
async def gold_upsert(req: GoldSetRequest):
    """录入/覆盖金标集（source=manual；manual 覆盖同名 demo）。"""
    save_gold(req.name, {"items": [it.model_dump() for it in req.items],
                         "source": "manual"})
    audit.gold_added(req.name)
    return {"ok": True, "name": req.name}


@app.delete("/api/gold/{name}")
async def gold_remove(name: str):
    ok = delete_gold(name)
    if not ok:
        raise HTTPException(404, "gold set not found")
    return {"ok": True}


@app.get("/api/gold/{name}/meta-eval")
async def gold_meta_eval(name: str, job_id: str):
    """金标元评估：金标 vs 指定任务评审分的 Spearman/Kappa/锚定偏移。

    任务未完成（无 verdict）→ 404；金标与 job 题/模型不匹配 → 200 空态。
    """
    if not is_valid_job_id(job_id):
        raise HTTPException(400, "非法 job_id")
    files = get_job_files(job_id)
    if files is None:
        raise HTTPException(404, "job not found")
    verdict = files.get("verdict.json")
    if verdict is None:
        raise HTTPException(404, "任务尚未完成评审（无 verdict），无法计算元评估")
    task_set = files.get("tasks.json", {})
    gold = load_gold(name)
    if gold is None:
        raise HTTPException(404, "gold set not found")
    return {"meta_eval": compute_meta_eval(verdict, task_set, gold)}


# ---- 模型配置库（迭代一） ----

@app.post("/api/models")
async def models_create(req: ModelRegisterRequest):
    """注册模型配置。API Key 仅存进程内存，落盘文件只保留 key_masked（***）。"""
    try:
        info = register(
            name=req.name, url=req.url, key=req.key or "",
            temperature=req.temperature, max_tokens=req.max_tokens, top_p=req.top_p,
        )
    except ModelRegistryError as e:
        raise HTTPException(400, f"模型注册失败: {e}")
    audit.model_registered(info["id"])
    return {"ok": True, "model": info}


@app.get("/api/models")
async def models_list():
    return {"models": list_models()}


@app.get("/api/models/{model_id}")
async def models_get(model_id: str):
    info = get_model(model_id)
    if info is None:
        raise HTTPException(404, "model not found")
    # 单条读取：附带进程内存中的 Key（本地前端「填入」流程使用；
    # 列表接口 /api/models 仍不返回 Key）
    return {**info, "key": get_key(model_id)}


@app.delete("/api/models/{model_id}")
async def models_delete(model_id: str):
    ok = delete_model(model_id)
    if not ok:
        raise HTTPException(404, "model not found")
    audit.model_deleted(model_id)
    return {"ok": True}


# ---- 跨 job 历史汇总（迭代一：接口就绪，数据由后续迭代真实接入） ----

@app.get("/api/stats/saturation")
async def stats_saturation():
    """读取跨 job 逐题结果汇总表（.eval/stats/saturation.json，幂等追加）。"""
    return redact_sensitive(get_saturation())


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
            # 与正式评测同策略：连接前重新解析并过滤非公网 IP（DNS 重绑定防护）
            async with build_upstream_client() as client:
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
    """启动一轮评测。支持 dataset_name（自定义评测集）和 repeat_n（重复次数）。

    迭代二：prompt_strategy / review（pure_agent 评审）/ budget（预算熔断）/
    embedding（语义向量采集）随请求生效；budget hard 超限启动即 400 拒绝。
    """
    # SSRF 防护：评测启动即校验上游目标（executor 每次调用前不再重复解析）
    try:
        validate_upstream_url(req.model_a.url)
        validate_upstream_url(req.model_b.url)
    except UpstreamUrlError as e:
        raise HTTPException(400, f"模型 URL 校验失败: {e}")

    # 迭代二：pure_agent 时评审模型同样走 SSRF 校验；迭代三：hybrid 同要求
    review_mode = (req.review.mode if req.review else "pure_human")
    judge_cfg = None
    if review_mode in ("pure_agent", "hybrid"):
        if not req.review or req.review.judge is None:
            raise HTTPException(400, f"{review_mode} 评审模式必须提供 review.judge 模型配置")
        judge_cfg = {
            "name": req.review.judge.name,
            "url": req.review.judge.url,
            "key": req.review.judge.key,
        }
        try:
            validate_upstream_url(req.review.judge.url)
        except UpstreamUrlError as e:
            raise HTTPException(400, f"评审模型 URL 校验失败: {e}")

    # 并发限制：执行中任务数（mock 不消耗外部资源，不计入；judging 阶段
    # 评审模型真实调用，同样计入）
    active = sum(
        1 for j in _jobs.values()
        if j.get("state") in ("pending", "executing", "judging")
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
        try:
            task_set = build_task_set_from_dataset(dataset)
        except DatasetValidationError as e:
            raise HTTPException(400, f"数据集格式错误: {e}")
    else:
        task_set = build_task_set(dims=req.dims, seed=seed, num_questions=req.num_questions)

    # 迭代二：预算熔断（hard 超限启动即 400 拒绝；warn 由运行期幂等提示）
    budget_check = check_budget(
        req.budget.model_dump() if req.budget else None,
        task_set["meta"]["total"], req.repeat_n, review_mode,
    )
    if budget_check["limited"] and not budget_check["allowed"]:
        raise HTTPException(
            400,
            f"预算超限：预估消耗 {budget_check['estimated']} token，"
            f"超过上限 {budget_check['limit']}（hard 模式拒绝启动）",
        )

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
        "prompt_strategy": req.prompt_strategy,
        "review": {
            "mode": review_mode,
            "fail_open": bool(req.review.fail_open) if req.review else False,
            "judge": judge_cfg,
            "k_top_human": req.review.k_top_human if req.review else 0,
        },
        "budget": req.budget.model_dump() if req.budget else None,
        "embedding": req.embedding.model_dump() if req.embedding else None,
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
        "budget_warned": False,
        "health_warned": False,
        "embedding_cfg": req.embedding.model_dump() if req.embedding else None,
    }

    task = asyncio.create_task(_run_eval(job_id))
    _tasks[job_id] = task
    task.add_done_callback(lambda t, jid=job_id: _task_done(jid, t))
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


def _job_cancelled(job_id: str) -> bool:
    """任务是否应停止：job 已从内存移除（并发删除）或处于 cancelling。"""
    j = _jobs.get(job_id)
    return j is None or j.get("state") == "cancelling"


async def _request_cancel(job_id: str):
    """将运行中任务置为 cancelling 并推送事件，随后由删除方调用 task.cancel()。"""
    j = _jobs.get(job_id)
    if j is None or j.get("state") == "cancelling":
        return
    j["state"] = "cancelling"
    await _push_event(job_id, {"state": "cancelling"})


async def _run_eval(job_id: str):
    """后台执行入口：包装 _run_eval_impl，取消时统一收尾并终止任务。"""
    try:
        await _run_eval_impl(job_id)
    except asyncio.CancelledError:
        j = _jobs.get(job_id)
        if j is not None:
            j["state"] = "cancelled"
        await _push_event(job_id, {"state": "cancelled"})
        raise


async def _run_eval_impl(job_id: str):
    """后台执行：出题→调用双模型→生成 X/Y 盲评映射→等待人工评审。

    评审阶段由用户在前端打分（POST /api/eval/{id}/review），
    本协程在作答完成后即退出，不再调用 AI 评审。

    取消协议（issue #14 / R2-005）：每次持久化与状态转换前检查取消标记；
    删除方先置 cancelling 再 task.cancel()，本协程在任何 await 点被中断，
    故目录删除后不会复活任何文件。
    """
    j = _jobs.get(job_id)
    if j is None:
        return
    cfg = j["config"]
    task_set = j["task_set"]
    total_tasks = task_set["meta"]["total"]
    repeat_n = j.get("repeat_n", 1)

    all_rounds: list[dict] = []
    for round_idx in range(repeat_n):
        round_label = f"第{round_idx+1}/{repeat_n}轮" if repeat_n > 1 else ""

        if _job_cancelled(job_id):
            raise JobCancelled
        j = _jobs.get(job_id)
        if j is None:
            raise JobCancelled
        j["state"] = "executing"
        j["progress"] = f"0/{total_tasks}"
        await _push_event(job_id, {"state": "executing", "progress": f"0/{total_tasks}", "round": round_label})

        async def progress_cb(label, done, total):
            j = _jobs.get(job_id)
            if j is None or j.get("state") == "cancelling":
                if j is not None:
                    j["cancel_requested"] = True
                return
            j["progress"] = f"{done}/{total}"
            await _push_event(job_id, {"state": "executing", "progress": f"{done}/{total}", "round": round_label})

        try:
            _exec_kwargs = {}
            if j.get("embedding_cfg"):
                _exec_kwargs["embedding_cfg"] = j["embedding_cfg"]
            answers_a, answers_b = await execute_all(
                task_set, config_a=cfg["model_a"], config_b=cfg["model_b"],
                progress_cb=progress_cb, **_exec_kwargs,
            )
        except Exception as e:
            if _job_cancelled(job_id):
                raise JobCancelled
            j = _jobs.get(job_id)
            if j is None:
                raise JobCancelled
            j["state"] = "error"
            j["error"] = f"模型调用失败: {e}"
            await _push_event(job_id, {"state": "error", "error": str(e)})
            _mark_error(job_id, f"模型调用失败: {e}")
            return

        # execute_all 内部可能因取消标记而停止（gather 吞掉子协程异常），
        # 返回后必须复查：取消态一律不得落盘（防目录复活）。
        if _job_cancelled(job_id):
            raise JobCancelled
        if _jobs.get(job_id, {}).get("cancel_requested"):
            raise JobCancelled

        # 保存本轮答卷（round 文件 + 覆盖当前答卷）
        save_answers(job_id, f"a-r{round_idx+1}", answers_a)
        save_answers(job_id, f"b-r{round_idx+1}", answers_b)
        save_answers(job_id, "a", answers_a)
        save_answers(job_id, "b", answers_b)
        j = _jobs.get(job_id)
        if j is None:
            raise JobCancelled
        j["answers_a"] = answers_a
        j["answers_b"] = answers_b
        all_rounds.append({"a": answers_a, "b": answers_b})

    j = _jobs.get(job_id)
    if j is None:
        raise JobCancelled
    j["rounds_answers"] = all_rounds

    # 迭代二：warn 预算超限 → 每任务幂等推送一次预算告警
    budget_cfg = cfg.get("budget")
    if budget_cfg:
        b_check = check_budget(
            budget_cfg, task_set["meta"]["total"], repeat_n,
            (cfg.get("review") or {}).get("mode", "pure_human"),
        )
        if b_check["limited"] and not j.get("budget_warned") and b_check["exceed"] > 0:
            j["budget_warned"] = True
            await _push_event(job_id, {
                "type": "budget_warning",
                "message": f"预估 token 消耗 {b_check['estimated']} 超过预算上限 "
                           f"{b_check['limit']}（超出 {b_check['exceed']}），请关注成本",
                "estimated": b_check["estimated"],
                "limit": b_check["limit"],
            })

    # 生成并持久化 X/Y 身份映射（重启不丢）
    reveal = make_reveal(repeat_n)
    save_reveal(job_id, reveal)
    j = _jobs.get(job_id)
    if j is None:
        raise JobCancelled
    j["reveal"] = reveal

    review_mode = (cfg.get("review") or {}).get("mode")
    if review_mode in ("pure_agent", "hybrid"):
        await _run_agent_judging(job_id, all_rounds, reveal, cfg, task_set, review_mode)
    else:
        # 进入人工评审阶段，等待用户打分
        j["state"] = "reviewing"
        j["progress"] = "0/0"
        await _push_event(job_id, {"state": "reviewing"})


async def _run_agent_judging(
    job_id: str,
    all_rounds: list[dict],
    reveal: dict,
    cfg: dict,
    task_set: dict,
    mode: str = "pure_agent",
):
    """Agent 全量预评（pure_agent 直通 completed；hybrid 选出复核集进 reviewing）。

    - 与人工评审共用 build_final_verdict 聚合；迭代三起逐题独立随机交换
      （H1：make_task_reveal → per_round_reveal 按题归一化聚合）；
    - 评审模型整体异常 / 全部 verdict invalid：fail_open=True 降级人工评审
      （写盘 review.mode=pure_human + degraded 标注 H3），否则任务置 error；
    - hybrid：judging 结束选择复核集（H2：invalid 必选 → 分差降序 → 低分兜底，
      k=min(k_top_human, 候选, 总数) L3）；k==0 直通 completed（不 reviewing）；
      落盘 hybrid-review.json（M2 重启恢复）；SSE reviewing 带 mode/k；
    - 健康度：invalid 率超阈值 → 幂等推送 judge_health invalid_rate；
    - audit：judging 完成落 eval_judged（补迭代二缺口）；
    - Key 仅存进程内存（config 落盘时已打码）。
    """
    j = _jobs.get(job_id)
    if j is None:
        raise JobCancelled
    judge_cfg = (cfg.get("review") or {}).get("judge") or {}
    fail_open = bool((cfg.get("review") or {}).get("fail_open"))
    repeat_n = len(all_rounds) or 1
    k_top = int((cfg.get("review") or {}).get("k_top_human") or 0)

    def _raise_if_cancelled():
        if _job_cancelled(job_id):
            raise JobCancelled
        jj = _jobs.get(job_id)
        if jj is None or jj.get("cancel_requested"):
            raise JobCancelled

    async def judge_progress_cb(done, total):
        jj = _jobs.get(job_id)
        if jj is None:
            return
        jj["progress"] = f"{done}/{total}"
        await _push_event(job_id, {"state": "judging", "progress": f"{done}/{total}"})

    round_verdicts: list[dict] = []
    per_round_reveal: list[dict] = []
    try:
        for r_idx, round_ans in enumerate(all_rounds):
            _raise_if_cancelled()
            jj = _jobs.get(job_id)
            jj["state"] = "judging"
            jj["progress"] = "0/0"
            await _push_event(job_id, {"state": "judging", "progress": "0/0",
                                       "round": f"第{r_idx+1}/{repeat_n}轮"})

            x_model, y_model, _xp, _yp = resolve_round(
                reveal, r_idx, round_ans["a"], round_ans["b"]
            )
            round_reveal = reveal["rounds"][r_idx] if r_idx < len(reveal["rounds"]) \
                else {"answer_x": "a", "answer_y": "b"}
            # 迭代三（H1）：逐题独立随机交换（仅 agent）；旧轮级 reveal 保留为兜底
            task_reveal = make_task_reveal(
                [t["id"] for t in task_set["tasks"]],
                seed=int(cfg.get("seed", 0)) * 100 + r_idx,
            )
            per_round_reveal.append(_normalize_task_reveal(task_reveal) or {})
            judge_verdict = await run_judge(
                task_set, round_ans["a"], round_ans["b"], judge_cfg,
                revealed=task_reveal,
                progress_cb=judge_progress_cb,
            )
            valid = judge_verdict.get("meta", {}).get("valid", 0)
            if valid == 0:
                raise RuntimeError("评审模型未能返回任何有效 verdict")
            round_verdicts.append(_round_verdict_from_judge(
                judge_verdict, x_model, y_model, r_idx, round_reveal,
                per_task_reveal=per_round_reveal[-1],
            ))
    except JobCancelled:
        raise
    except Exception as e:
        _raise_if_cancelled()
        j = _jobs.get(job_id)
        if fail_open:
            j["state"] = "reviewing"
            j["progress"] = "0/0"
            await _push_event(job_id, {
                "state": "reviewing", "judge_failed": True,
            })
            await _push_event(job_id, {
                "type": "judge_health", "status": "degraded",
                "detail": f"AI 评审失败（fail_open 降级人工评审）：{e}",
            })
            await _mark_review_degraded(job_id, cfg)
            return
        j["state"] = "error"
        j["error"] = f"AI 评审失败: {e}"
        await _push_event(job_id, {"state": "error", "error": str(e)})
        _mark_error(job_id, f"AI 评审失败: {e}")
        return

    # 健康度告警（迭代三）：judging 结束后整体 invalid 率超阈值 →
    # 幂等推送 judge_health invalid_rate（不打断流程）
    j = _jobs.get(job_id)
    if j is not None:
        total_v = sum(int(rv.get("meta", {}).get("total", 0)) for rv in round_verdicts)
        total_inv = sum(int(rv.get("meta", {}).get("invalid", 0)) for rv in round_verdicts)
        h = health_check({"total": total_v, "invalid": total_inv}, threshold=0.1)
        if h["alarm"] and not j.get("health_warned"):
            j["health_warned"] = True
            await _push_event(job_id, {
                "type": "judge_health", "status": "invalid_rate",
                "rate": h["invalid_rate"], "threshold": 0.1,
            })

    if mode == "hybrid":
        review_set, k = _select_hybrid_review_set(round_verdicts, k_top, task_set)
        if k == 0:
            # L3 边界：k==0 直通 completed（不进入复核态）
            _finalize_agent_done(job_id, round_verdicts, per_round_reveal,
                                 cfg, task_set, all_rounds,
                                 review_data={"mode": "hybrid", "k": 0,
                                              "note": "k_top_human=0 或候选为空，未进入人工复核"})
            await _push_event(job_id, {"state": "completed", "mode": "hybrid", "k": 0})
            audit.eval_judged(job_id, actor="agent")
            audit.review_submitted(job_id, actor="agent")
            return
        save_round_verdicts(job_id, round_verdicts)  # M2：逐轮 agent 分落盘，供复核提交聚合
        save_hybrid_review(job_id, {
            "k": k,
            "review_set": review_set,
            "reveals": per_round_reveal,   # M2：逐题 reveal 落盘，重启恢复聚合
            "selected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "state": "reviewing",
        })
        j = _jobs.get(job_id)
        j["state"] = "reviewing"
        j["progress"] = "0/0"
        await _push_event(job_id, {"state": "reviewing", "mode": "hybrid", "k": k})
        audit.eval_judged(job_id, actor="agent")
        return

    _finalize_agent_done(job_id, round_verdicts, per_round_reveal,
                         cfg, task_set, all_rounds,
                         review_data={"mode": "pure_agent", "note": "agent 自动评审"})
    await _push_event(job_id, {"state": "completed"})
    audit.eval_judged(job_id, actor="agent")
    audit.review_submitted(job_id, actor="agent")


def _finalize_agent_done(
    job_id: str,
    round_verdicts: list[dict],
    per_round_reveal: list[dict],
    cfg: dict,
    task_set: dict,
    all_rounds: list[dict],
    review_data: dict,
):
    """agent 路径完成聚合（pure_agent / hybrid k==0 共用）。"""
    verdict = build_final_verdict(
        round_verdicts, len(all_rounds) or 1,
        per_round_reveal=per_round_reveal or None,
    )
    answers_a, answers_b = all_rounds[-1]["a"], all_rounds[-1]["b"]
    _finalize_job(
        job_id, verdict, review_data,
        round_verdicts, cfg, task_set, answers_a, answers_b, all_rounds,
        embedding_config=_jobs.get(job_id, {}).get("embedding_cfg"),
    )


async def _mark_review_degraded(job_id: str, cfg: dict):
    """fail_open 降级（H3）：review.mode 落盘 pure_human + degraded 标注。"""
    j = _jobs.get(job_id)
    if j is not None:
        j["config"] = dict(cfg)
        j["config"].setdefault("review", {})
        j["config"]["review"]["mode"] = "pure_human"
        j["config"]["review"]["degraded"] = True
        j["config"]["review"]["judge"] = None
    _re_save_config(job_id, cfg)


def _re_save_config(job_id: str, cfg: dict):
    """把降级后的 review 配置重新落盘（config.json），重启恢复可见。"""
    from backend.storage import save_config
    degraded_cfg = dict(cfg)
    review = dict(degraded_cfg.get("review") or {})
    review["mode"] = "pure_human"
    review["degraded"] = True
    review["judge"] = None
    degraded_cfg["review"] = review
    degraded_cfg["review_mode"] = "pure_human"  # 兼容扁平落盘读取
    save_config(job_id, degraded_cfg)


def _select_hybrid_review_set(
    round_verdicts: list[dict],
    k_top_human: int,
    task_set: dict,
) -> tuple[list[dict], int]:
    """选择 hybrid 人工复核集（H2/L3）：invalid 必选 → 其余分差降序 → 低分兜底。

    k = min(k_top_human, 候选总数)（L3）；返回 (review_set, k)。候选不含
    excluded_from_total 题（不计分题无需复核胜负）。
    """
    excluded = {t["id"] for t in task_set.get("tasks", [])
                if t.get("excluded_from_total")}
    candidates: list[dict] = []
    for rv in round_verdicts:
        for s in rv.get("scores", []):
            tid = s.get("id", "")
            if tid in excluded:
                continue
            x = float(s.get("answer_x", 0))
            y = float(s.get("answer_y", 0))
            candidates.append({
                "round": int(s.get("round", 1)),
                "task_id": tid,
                "agent_x": round(x, 2),
                "agent_y": round(y, 2),
                "winner": s.get("winner", "tie"),
                "basis": s.get("basis", ""),
                "gap": round(abs(x - y), 2),
                "low": round(min(x, y), 2),
                "invalid": bool(s.get("_invalid")),
            })
    if not candidates:
        return [], 0
    k = min(max(k_top_human, 0), len(candidates))
    invalids = sorted(
        [c for c in candidates if c["invalid"]],
        key=lambda c: (-c["gap"], c["low"]),
    )
    others = sorted(
        [c for c in candidates if not c["invalid"]],
        key=lambda c: (-c["gap"], c["low"]),
    )
    return (invalids + others)[:k], k


def _round_verdict_from_judge(
    judge_verdict: dict,
    x_model: str,
    y_model: str,
    round_idx: int,
    round_reveal: dict,
    per_task_reveal: dict[str, str] | None = None,
) -> dict:
    """把 run_judge 输出归一化为人工评审同构的轮次 verdict（供 build_final_verdict 聚合）。

    per_task_reveal（迭代三 H1）：逐题独立交换映射，写进 revealed 供
    report_builder 一致率在稳定空间计算时按题对齐（round-verdicts.json 落盘持久）。
    """
    scores = []
    for s in judge_verdict.get("scores", []):
        row = dict(s)
        row["round"] = round_idx + 1
        scores.append(row)
    revealed = {
        "answer_x": x_model,
        "answer_y": y_model,
        "answer_x_file": round_reveal.get("answer_x", "a"),
        "answer_y_file": round_reveal.get("answer_y", "b"),
    }
    if per_task_reveal:
        revealed["per_task"] = per_task_reveal
    return {
        "meta": {**judge_verdict.get("meta", {}), "repeat_n": 1},
        "scores": scores,
        "per_dimension": judge_verdict.get("per_dimension", {}),
        "totals": judge_verdict.get("totals", {}),
        "revealed": revealed,
        "conclusion": judge_verdict.get("conclusion", ""),
        "winner_model": judge_verdict.get("winner_model", "tie"),
    }


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
    embedding_config: dict | None = None,
):
    """人工评审提交后：写 verdict/review/round-verdicts/report，置为 completed。

    数据一律来自显式参数（由 _load_job_state 恢复），不依赖进程内 _jobs，
    服务重启后（_jobs 清空）磁盘态任务仍可完成提交闭环。
    """
    save_verdict(job_id, verdict)
    save_review(job_id, review_data)
    if round_verdicts:
        save_round_verdicts(job_id, round_verdicts)
    report = build_report(cfg, task_set, answers_a, answers_b, verdict, rounds_answers,
                          embedding_config)
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


def _restore_review_object(cfg: dict) -> None:
    """把扁平 config（落盘格式）重建为 review 对象；已含 review 对象则跳过。

    迭代三：k_top_human/degraded 仅在有落盘键时写入，与未配置（缺省）区分。
    """
    if isinstance(cfg.get("review"), dict):
        return
    if isinstance(cfg.get("review_mode"), str):
        review = {"mode": cfg.get("review_mode")}
        if "fail_open" in cfg:
            review["fail_open"] = bool(cfg.get("fail_open"))
        if "review_k_top_human" in cfg:
            review["k_top_human"] = int(cfg.get("review_k_top_human") or 0)
        if "review_degraded" in cfg:
            review["degraded"] = bool(cfg.get("review_degraded"))
        cfg["review"] = review


def _load_job_state(job_id: str) -> tuple[dict, dict, dict, int] | None:
    """从内存或磁盘恢复评审所需状态（config/task_set/rounds_answers/repeat_n）。"""
    if job_id in _jobs:
        j = _jobs[job_id]
        cfg = j["config"]
        _restore_review_object(cfg)
        return cfg, j["task_set"], j["rounds_answers"], j.get("repeat_n", 1)
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
    # 磁盘 config 为扁平结构（迭代三含 k_top_human/degraded），重建 review 对象
    _restore_review_object(cfg)
    return cfg, task_set, rounds, repeat_n


@app.get("/api/eval/{job_id}/review")
async def eval_review_view(job_id: str):
    """返回人工评审页数据：题目 + 答案X/答案Y（模型身份完全隐藏）。"""
    job_id = _require_job_id(job_id)
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

    # 迭代三（H4）：hybrid 任务附带复核集/agent 原分/K（纯人工任务缺省；
    # k==0 直通 completed 时无 hybrid-review.json，视为非复核态）
    hybrid_view = None
    if (cfg.get("review") or {}).get("mode") == "hybrid":
        hdata = load_hybrid_review(job_id)
        if hdata is not None:
            review_set = hdata.get("review_set", []) or []
            agent_scores = {
                f"{it.get('round')}:{it.get('task_id')}": {
                    "agent_x": it.get("agent_x"), "agent_y": it.get("agent_y"),
                    "winner": it.get("winner"),
                }
                for it in review_set
            }
            hybrid_view = {
                "review_set": review_set,
                "agent_scores": agent_scores,
                "k": hdata.get("k", 0 if review_set else None),
            }

    return {
        "job_id": job_id,
        "repeat_n": repeat_n,
        "total_questions": len(task_set["tasks"]),
        "rounds": rounds_view,
        "submitted": load_review(job_id) is not None,
        "hybrid": hybrid_view,
    }


@app.post("/api/eval/{job_id}/review", response_model=StartResponse)
async def eval_review_submit(job_id: str, req: ReviewSubmission):
    """提交人工打分：按轮构建 verdict → 聚合 → 生成报告 → completed。"""
    job_id = _require_job_id(job_id)
    restored = _load_job_state(job_id)
    if restored is None:
        raise HTTPException(404, "job not found")
    if _jobs.get(job_id, {}).get("state") in ("cancelling", "cancelled"):
        raise HTTPException(409, "该任务已被取消或删除，无法提交评分")
    cfg, task_set, rounds_answers, repeat_n = restored

    if load_review(job_id) is not None:
        raise HTTPException(409, "该任务已提交评分，请勿重复提交")

    # 迭代三：hybrid 任务的人工打分必须走 hybrid-review 复核接口（子集覆盖），
    # 防止绕过融合直接用全量人工分覆盖
    if (cfg.get("review") or {}).get("mode") == "hybrid":
        raise HTTPException(409, "该任务为 hybrid 评审模式，请走 AI 复核接口提交（hybrid-review）")

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


@app.post("/api/eval/{job_id}/hybrid-review", response_model=StartResponse)
async def eval_hybrid_review_submit(job_id: str, req: ReviewSubmission):
    """hybrid 复核提交：按 (round, task_id) 用人工分覆盖 agent 预评分 → 聚合 → completed。

    约束：配置/状态非 hybrid 复核态 → 409；已提交 → 409；复核子集内
    每 (round, task_id) 恰一条、round 在 1..repeat_n、task 属于复核集、
    题号属于任务集 → 否则 400；重启恢复（M2）从磁盘 config /
    hybrid-review.json / round-verdicts.json 重建。
    """
    job_id = _require_job_id(job_id)
    restored = _load_job_state(job_id)
    if restored is None:
        raise HTTPException(404, "job not found")
    if _jobs.get(job_id, {}).get("state") in ("cancelling", "cancelled"):
        raise HTTPException(409, "该任务已被取消或删除，无法提交复核")
    cfg, task_set, rounds_answers, repeat_n = restored

    if (cfg.get("review") or {}).get("mode") != "hybrid":
        raise HTTPException(409, "该任务不是 hybrid 评审模式，请走人工评审接口")
    if (cfg.get("review") or {}).get("degraded"):
        raise HTTPException(409, "该任务已因 AI 评审失败降级为纯人工评审，请走人工评审接口")
    if load_review(job_id) is not None:
        raise HTTPException(409, "该任务已提交评审结果，请勿重复提交")

    hybrid_data = load_hybrid_review(job_id)
    if hybrid_data is None:
        raise HTTPException(409, "该任务未进入 hybrid 复核态（无复核集）")
    review_keys = {(int(it["round"]), str(it["task_id"]))
                   for it in hybrid_data.get("review_set", [])}

    # 复核子集校验：round 在 1..repeat_n、task 属于复核集、每 (round, task) 恰一条
    task_ids = {t["id"] for t in task_set["tasks"]}
    seen: set[tuple[int, str]] = set()
    for s in req.scores:
        key = (s.round, s.id)
        if s.round not in range(1, repeat_n + 1):
            raise HTTPException(400, f"轮次越界：round={s.round}（有效 1..{repeat_n}）")
        if s.id not in task_ids:
            raise HTTPException(400, f"未知题号：{s.id}")
        if key not in review_keys:
            raise HTTPException(400, f"题目 {s.id}（round {s.round}）不属于 hybrid 复核集")
        if key in seen:
            raise HTTPException(400, f"重复提交：round {s.round} / 题 {s.id}")
        seen.add(key)
    missing = sorted(review_keys - seen)
    if missing:
        raise HTTPException(400, f"复核集未完整提交，缺失 {len(missing)} 条："
                                 f"{', '.join(f'r{r}/{tid}' for r, tid in missing[:5])}…")

    round_verdicts = load_round_verdicts(job_id)
    if not round_verdicts:
        raise HTTPException(409, "缺少 agent 预评的逐轮 verdict（round-verdicts.json 缺失）")

    merged = merge_hybrid_verdicts(
        round_verdicts, [s.model_dump() for s in req.scores]
    )
    per_round_reveal = hybrid_data.get("reveals") or None
    verdict = build_final_verdict(merged, repeat_n, per_round_reveal=per_round_reveal)
    answers_a, answers_b = rounds_answers[-1]["a"], rounds_answers[-1]["b"]
    _finalize_job(
        job_id, verdict,
        {"mode": "hybrid", "scores": [s.model_dump() for s in req.scores],
         "k": hybrid_data.get("k", 0)},
        merged, cfg, task_set, answers_a, answers_b, rounds_answers,
    )
    await _push_event(job_id, {"state": "completed"})
    audit.review_submitted(job_id, actor="human")
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
    job_id = _require_job_id(job_id)
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


@app.post("/api/eval/{job_id}/events/ticket")
async def eval_events_ticket(job_id: str):
    """为 SSE 进度流签发短时单次 ticket（issue #13）。

    认证由中间件以 Authorization header 兜底；ticket 仅限该 job 的 /events
    路由，使用一次后立即失效。终态 job 不再签发，避免客户端挂在心跳上。
    """
    job_id = _require_job_id(job_id)
    if job_id not in _jobs:
        raise HTTPException(404, "job not found")
    if _jobs[job_id]["state"] in TERMINAL_STATES or _jobs[job_id]["state"] == "cancelling":
        raise HTTPException(409, "job already finished")
    ticket = sse_ticket.issue(job_id)
    return {"ticket": ticket, "ttl_seconds": sse_ticket.TTL_SECONDS}


@app.get("/api/eval/{job_id}/events")
async def eval_events(job_id: str):
    job_id = _require_job_id(job_id)
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
                if event.get("state") in TERMINAL_STATES:
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
    job_id = _require_job_id(job_id)
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
                    j["verdict"], rounds_answers, j.get("embedding_cfg"),
                )
                return redact_sensitive(payload)
        raise HTTPException(404, "job not found or not completed")

    report_dict = files.get("report.json")
    verdict = files.get("verdict.json")
    tasks = files.get("tasks.json")
    answers_a = files.get("answers-a.json")
    answers_b = files.get("answers-b.json")
    cfg = files.get("config.json")

    # 纵深防御（issue #1 验收3）：修复前遗留的旧记录 config.json 可能含
    # 明文 Key（save_config 打码只对新记录生效），报告接口同样须脱敏
    cfg = redact_sensitive(cfg)

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
            # 磁盘 config 为扁平化结构（review_mode/judge 平铺），重建时还原
            # review.mode 供 build_report 输出正确的 judge_mode
            restore_cfg = dict(cfg or {})
            if isinstance(restore_cfg.get("review_mode"), str):
                restore_cfg["review"] = {"mode": restore_cfg["review_mode"]}
            rich = build_report(restore_cfg, tasks, answers_a, answers_b, verdict, rounds_answers)
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
    """删除历史记录。运行中的任务先取消并等待后台协程安全终止，再删内存与磁盘，
    保证目录/文件不会由后台流程复活（issue #14 / R2-005）。

    语义：运行中删除 = 置 cancelling → task.cancel() → 等待回收（含兜底超时）
    → 完整清理后返回 200；未知任务 404；目录删除只发生在任务停止之后。
    """
    from backend.storage import delete_job

    job_id = _require_job_id(job_id)
    task = _tasks.get(job_id)
    if task is not None:
        if not task.done():
            await _request_cancel(job_id)
            task.cancel()
            for _ in range(2):
                _, pending = await asyncio.wait({task}, timeout=CANCEL_GRACE_SECONDS)
                if not pending:
                    break
            else:
                # 极端场景（如阻塞线程未归位）：取消已送达，任务只能在下一个
                # await 点退出，且所有持久化前都有取消检查，删除后不会复活。
                print(f"[eval] job {job_id} 取消等待超时，继续删除（任务仍在回收）", file=sys.stderr)
            audit.eval_cancelled(job_id)
        _tasks.pop(job_id, None)

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
    job_id = _require_job_id(job_id)
    files = get_job_files(job_id)
    if files is None:
        raise HTTPException(404, "job not found")
    return files
