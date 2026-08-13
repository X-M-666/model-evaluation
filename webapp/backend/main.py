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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.engine.tasks import build_task_set, build_task_set_from_dataset, DIMENSIONS
from backend.engine.executor import execute_all, _execute_model
from backend.engine.budget import check_budget
from backend.engine.judge import (
    run_judge, run_single_arm_judge, make_task_reveal, _normalize_task_reveal,
    health_check,
)
from backend.engine.human_review import (
    make_reveal, resolve_round, build_review_view,
    build_round_verdict, build_final_verdict, merge_hybrid_verdicts,
)
from backend.engine.report_builder import build_report, reveal_answers
from backend.engine.parsers import get_parser, supported_extensions
from backend.engine.generator import run_generation_pipeline, EDIT_ALLOWED_FIELDS
from backend.engine.datasets import (
    _as_str,
    MAX_NAME_LEN,
    MAX_DATASET_TASKS,
    DatasetValidationError,
    validate_json_dataset,
    validate_standard_dataset,
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
    save_generation_batch, load_generation_batch, list_generation_batches,
    bump_dataset_version, is_valid_gen_id,
    save_env_snapshot, load_env_snapshot, collect_env_snapshot,
    build_export_zip,
    save_badcase, load_badcase, list_badcases, delete_badcase,
    update_badcase_attribution, export_badcases_json,
    save_perturb, load_perturb, list_perturbs, is_valid_perturb_id,
    save_leaderboard, load_leaderboard, list_leaderboards, is_valid_lb_id,
    save_batch, load_batch, list_batches, is_valid_batch_id,
    save_answers_inc, load_answers_inc, partial_answers_count,
)
from backend.gold import ensure_demo_gold, compute_meta_eval
from backend.engine.rag_demo import ensure_demo_rag_dataset
from backend.engine.badcase import (
    BAD_CASE_CATEGORIES, UNCATEGORIZED, mine_bad_cases, attribute_badcase,
)
from backend.engine.stats import saturation_trend
from backend.engine.perturb import (
    PERTURB_MODES, build_perturb_set, score_task_metric,
    build_robustness_curves, bias_analysis,
)
from backend.engine.leaderboard import build_leaderboard, LeaderboardError
from backend.engine.dashboard import build_jobs_trend
from backend.hwmon import collect_hw
from backend.scheduler import Scheduler
from backend.schemas import (
    StartRequest, StartResponse, ReviewSubmission, ModelRegisterRequest,
    GoldSetRequest, GenerateRequest, ReviewDecisionRequest,
    PerturbRequest, LeaderboardRequest, BenchmarkRequest, PriorityRequest,
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
    """后台任务结束回调：回收引用、释放调度配额并消费异常，避免未获取 Task 异常告警。"""
    _tasks.pop(job_id, None)
    _scheduler_release(job_id)  # 迭代七：终态/取消兜底释放配额
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
    ensure_demo_rag_dataset()  # 迭代四：内置 RAG 演示集（带 context 参考文档）
    _settle_queued_on_restart()  # 迭代七：重启后排队任务沉降 error（v1 内存队列）
    yield
    await _shutdown_cancel_all()


def _settle_queued_on_restart() -> None:
    """重启沉降：磁盘态 queued（仅 config.json）任务置 error，提示不自动恢复。

    v1 内存队列不持久化，排队任务随进程重启丢失；error 标记保证历史列表
    可见且不悬置（磁盘无 cancelled 态，error 可见性等价）。
    """
    for j in list_jobs():
        if j.get("state") == "queued":
            try:
                save_error(j["job_id"], "排队任务随重启取消（v1 内存队列，不自动恢复，可删除）")
            except Exception:
                continue


app = FastAPI(title="模型对决评测平台", version="0.3.0", lifespan=lifespan)
app.middleware("http")(security_middleware)

_jobs: dict[str, dict] = {}
_tasks: dict[str, asyncio.Task] = {}

# 迭代七：任务调度器（迭代 0 契约落地）——优先级队列 + 并发配额，
# 替代硬编码 MAX_ACTIVE_JOBS=2 与 429 拒绝；配额可由环境变量弹性调整
# （MODEL_DUEL_CONCURRENCY，默认 2，即 CPU 池化 v1 形态）。
_SCHEDULER = Scheduler(concurrency=int(os.environ.get("MODEL_DUEL_CONCURRENCY", "2") or 2))


def _job_running(j: dict) -> bool:
    """job 是否占用调度配额（执行+评审阶段）。"""
    return j.get("state") in ("executing", "judging") and not str(
        j.get("config", {}).get("model_a", {}).get("url", "")
    ).startswith("mock")


async def _dispatch_pending() -> None:
    """调度器派发胶水：配额内弹候选 → 建后台任务（沿用 _tasks/_task_done）。

    batch 执行单元（config.batch_id）走 _run_batch_job，其余走 _run_eval。
    """
    for job_id in _SCHEDULER.next_batch():
        j = _jobs.get(job_id)
        if j is None:
            continue
        if (j.get("config") or {}).get("batch_id"):
            task = asyncio.create_task(_run_batch_job(job_id))
        else:
            task = asyncio.create_task(_run_eval(job_id))
        _tasks[job_id] = task
        task.add_done_callback(lambda t, jid=job_id: _task_done(jid, t))


def _scheduler_release(job_id: str) -> None:
    """释放配额并补派发（幂等；reviewing/终态/取消路径统一调用）。"""
    if _SCHEDULER.release(job_id):
        asyncio.create_task(_dispatch_pending())


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


@app.get("/gen_review.html", response_class=HTMLResponse)
async def gen_review_page():
    """出题批次审核页（迭代四）：approve（含编辑）/reject → 数据集入库。"""
    return _page_response("gen_review.html")


@app.get("/badcases.html", response_class=HTMLResponse)
async def badcases_page():
    return _page_response("badcases.html")


@app.get("/perturb.html", response_class=HTMLResponse)
async def perturb_page():
    """对抗扰动评测页（迭代六）：配置 → 运行 → 衰减曲线/偏见对照。"""
    return _page_response("perturb.html")


@app.get("/leaderboard.html", response_class=HTMLResponse)
async def leaderboard_page():
    """N 模型排行榜页（迭代六）：历史 job 聚合 → 排名/胜率矩阵/图表。"""
    return _page_response("leaderboard.html")


@app.get("/dashboard.html", response_class=HTMLResponse)
async def dashboard_page():
    """KPI 看板页（迭代六）：耗时/token 趋势 + CPU/GPU 硬件利用率。"""
    return _page_response("dashboard.html")


@app.get("/tasks.html", response_class=HTMLResponse)
async def tasks_page():
    """任务调度页（迭代七）：排队/运行视图、优先级调整、benchmark 批次。"""
    return _page_response("tasks.html")


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


# ---- LLM 出题与待审核批次（迭代四） ----

_GENERATION_MAX_ACTIVE = 3


def _require_gen_id(gen_id: str) -> str:
    """校验 gen_id 为系统生成格式（防路径穿越）。"""
    if not is_valid_gen_id(gen_id):
        raise HTTPException(400, "invalid gen_id format")
    return gen_id


def _settle_generation(batch: dict) -> dict:
    """重启/取消遗留的 generating 批次落为 partial（进程中断的生成不复活）。"""
    if batch.get("state") == "generating" and batch.get("gen_id") not in _tasks:
        batch["state"] = "partial"
        batch["error"] = "生成协程中断（进程重启或服务关闭），已完成题目可继续审核"
        save_generation_batch(batch["gen_id"], batch)
    return batch


def _item_stats(items: list[dict]) -> dict:
    stats = {"total": len(items), "pending": 0, "approved": 0, "rejected": 0}
    for it in items:
        st = it.get("status")
        if st in stats:
            stats[st] += 1
    return stats


async def _run_generation(gen_id: str, gen_config: dict, req: GenerateRequest):
    """后台出题协程：生成 → 五级校验 → 批次置 ready。错误落盘不复活。"""
    try:
        pool = None
        if req.target_dataset:
            ds = load_dataset(req.target_dataset)
            if ds:
                pool = [str(t.get("prompt", "")) for t in ds.get("tasks", []) if isinstance(t, dict)]
        spec = {
            "task_type": req.task_type,
            "dimension": req.dimension or "知识能力",
            "count": req.count,
            "options": req.options or {},
            "target_dataset": req.target_dataset,
            "gen_name": gen_config.get("name", ""),
            "gen_key_masked": "***",
        }
        items = await run_generation_pipeline(gen_config, spec, pool=pool)
        batch = load_generation_batch(gen_id)
        if batch is None:
            return
        batch["items"] = [
            {
                "item_id": f"{gen_id}-{i + 1}",
                "task": it["task"],
                "checks": it["checks"],
                "issues": it["issues"],
                "ok": it["ok"],
                "status": "pending",
                "edits": None,
                "reviewed_at": None,
            }
            for i, it in enumerate(items)
        ]
        batch["state"] = "ready"
        save_generation_batch(gen_id, batch)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        batch = load_generation_batch(gen_id)
        if batch is not None:
            batch["state"] = "error"
            batch["error"] = f"出题失败: {exc}"
            save_generation_batch(gen_id, batch)


@app.post("/api/generate", response_model=None)
async def generate_tasks(req: GenerateRequest):
    """创建 LLM 出题批次（后台执行）。gen_config 必填（Key 仅内存，不落盘）。"""
    if req.gen_config is None:
        raise HTTPException(
            400,
            "必须提供 gen_config 出题模型配置（出题面板可一键复用评审模型配置）",
        )
    try:
        validate_upstream_url(req.gen_config.url)
    except UpstreamUrlError as e:
        raise HTTPException(400, f"出题模型 URL 校验失败: {e}")

    # 出题协程计入活动上限（与评测任务共池，避免并发模型调用失控）
    active = sum(1 for k in _tasks if str(k).startswith("gen_"))
    if active >= _GENERATION_MAX_ACTIVE:
        raise HTTPException(429, f"当前已有 {active} 个出题任务在执行，请稍后再试")

    gen_id = "gen_" + create_job_id()
    gen_cfg = {
        "name": req.gen_config.name,
        "url": req.gen_config.url,
        "key": req.gen_config.key or "",
    }
    batch = {
        "gen_id": gen_id,
        "state": "generating",
        "error": None,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "spec": {
            "task_type": req.task_type,
            "dimension": req.dimension,
            "count": req.count,
            "options": req.options or {},
            "target_dataset": req.target_dataset,
            "gen_name": gen_cfg["name"],
            "gen_key_masked": "***",
        },
        "items": [],
    }
    save_generation_batch(gen_id, batch)
    task = asyncio.create_task(_run_generation(gen_id, gen_cfg, req))
    _tasks[gen_id] = task
    task.add_done_callback(lambda t, gid=gen_id: _task_done(gid, t))
    audit.task_generate_started(gen_id)
    return {"ok": True, "gen_id": gen_id, "state": "generating", "count": req.count}


# ---- 对抗扰动评测（迭代六）----

_PERTURB_MAX_ACTIVE = 3

# 进程内存扰动请求表：{perturb_id: PerturbRequest}，仅内存不落盘
# （含 model/judge Key，随进程销毁；重启后遗留批次由 _settle_perturb 沉降 partial）
_PERTURB_REQS: dict[str, PerturbRequest] = {}


def _settle_perturb(data: dict) -> dict:
    """重启/取消遗留的 running 扰动评测沉降 partial（进程中断不复活）。"""
    if data.get("state") == "running" and data.get("perturb_id") not in _tasks:
        data["state"] = "partial"
        data["error"] = "扰动评测协程中断（进程重启或服务关闭），已生成部分不可用"
        save_perturb(data["perturb_id"], data)
    return data


async def _run_perturb(perturb_id: str, req: PerturbRequest, data: dict):
    """后台扰动评测：构建扰动集 → 单模型作答 → 判别式指标/生成式单臂评审 → 落盘。"""
    try:
        dataset = load_dataset(req.dataset_name)
        if dataset is None:
            raise ValueError(f"评测集 '{req.dataset_name}' 不存在")
        task_set = build_task_set_from_dataset(dataset)
        perturbed = build_perturb_set(task_set, req.modes, req.intensities, req.seed)

        model_cfg = {
            "name": req.model.name, "url": req.model.url, "key": req.model.key or "",
            "temperature": req.model.temperature, "max_tokens": req.model.max_tokens,
            "top_p": req.model.top_p, "code_verify_mode": "off",
            "prompt_strategy": req.prompt_strategy,
        }

        async def progress_cb(_label, done, total):
            data["progress"] = f"{done}/{total}"

        answers = await _execute_model("P", model_cfg, perturbed["tasks"], None,
                                       progress_cb, embedding_cfg=None)

        # 生成式题单臂评审（可选 judge；缺省该类题得分 N/A）
        single_scores: dict[str, float] = {}
        invalid = 0
        if req.judge is not None:
            judge_cfg = {"name": req.judge.name, "url": req.judge.url,
                         "key": req.judge.key or ""}
            gen_tasks = [t for t in perturbed["tasks"] if t.get("type") == "生成式"]
            answers_map = {a["id"]: a for a in answers.get("answers", [])}
            gen_answers = [answers_map[t["id"]] for t in gen_tasks
                           if t["id"] in answers_map]
            if gen_answers:
                sa = await run_single_arm_judge(
                    {"tasks": gen_tasks, "meta": {}},
                    {"model": model_cfg["name"], "answers": gen_answers},
                    judge_cfg)
                single_scores = {v["id"]: v.get("score")
                                 for v in sa.get("scores", [])}
                invalid = sa.get("meta", {}).get("invalid", 0)

        answers_map = {a["id"]: a for a in answers.get("answers", [])}
        per_task: list[dict] = []
        for t in perturbed["tasks"]:
            entry = answers_map.get(t["id"])
            meta = t.get("meta") or {}
            api = (entry or {}).get("api_info") or {}
            score = None
            if t.get("type") == "生成式":
                score = single_scores.get(t["id"])
            elif entry is not None:
                score = score_task_metric(t, entry)
            per_task.append({
                "task_id": t["id"],
                "origin_id": meta.get("origin_id", t["id"]),
                "mode": meta.get("perturb_mode", "原版"),
                "intensity": meta.get("perturb_intensity", 0.0),
                "score": score,
                "raw_answer": (entry or {}).get("raw_answer", "")[:2000] or None,
                "latency_ms": api.get("latency_ms"),
                "tokens": (api.get("prompt_tokens") or 0)
                          + (api.get("completion_tokens") or 0),
            })

        curves = build_robustness_curves(per_task, req.modes)
        bias = await bias_analysis(per_task)
        warnings = [
            {"code": "perturb_skipped", "task_id": s.get("task_id"),
             "mode": s.get("mode"), "message": s.get("reason", "")}
            for s in perturbed["meta"].get("skipped", [])
        ]
        if invalid:
            warnings.append({
                "code": "perturb_judge_invalid",
                "message": f"单臂评审 invalid {invalid} 题，对应生成式题得分按 N/A 处理",
            })

        data.update({
            "state": "ready",
            "per_task": per_task,
            "curves": curves,
            "bias": bias,
            "warnings": warnings,
            "progress": "done",
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        audit.perturb_completed(perturb_id, len(per_task))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        data["state"] = "error"
        data["error"] = f"扰动评测失败: {exc}"
    else:
        # 正常完成即清除内存请求（含 Key）；取消路径保留（进程将退出）
        _PERTURB_REQS.pop(perturb_id, None)
    finally:
        save_perturb(perturb_id, data)


@app.post("/api/perturb", response_model=None)
async def create_perturb(req: PerturbRequest):
    """创建对抗扰动评测（后台执行）。SSRF 校验 model/judge URL；非法模式 400。"""
    invalid = [m for m in req.modes if m not in PERTURB_MODES]
    if invalid:
        raise HTTPException(400, f"非法扰动模式: {invalid}（合法值 {PERTURB_MODES}）")
    try:
        validate_upstream_url(req.model.url)
        if req.judge is not None:
            validate_upstream_url(req.judge.url)
    except UpstreamUrlError as e:
        raise HTTPException(400, f"模型 URL 校验失败: {e}")
    if load_dataset(req.dataset_name) is None:
        raise HTTPException(404, f"评测集 '{req.dataset_name}' 不存在")

    active = sum(1 for k in _tasks if str(k).startswith("prb_"))
    if active >= _PERTURB_MAX_ACTIVE:
        raise HTTPException(429, f"当前已有 {active} 个扰动评测在执行，请稍后再试")

    perturb_id = "prb_" + create_job_id()
    data = {
        "perturb_id": perturb_id,
        "state": "running",
        "error": None,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_name": req.model.name,
        "model_url": req.model.url,
        "model_key_masked": "***",
        "dataset": req.dataset_name,
        "modes": req.modes,
        "intensities": req.intensities,
        "seed": req.seed,
        "has_judge": req.judge is not None,
        "judge_name": req.judge.name if req.judge else None,
        "progress": "0/0",
        "per_task": [],
        "curves": {},
        "bias": {},
        "warnings": [],
    }
    save_perturb(perturb_id, data)
    _PERTURB_REQS[perturb_id] = req  # 仅内存（含 Key），随进程销毁
    task = asyncio.create_task(_run_perturb(perturb_id, req, data))
    _tasks[perturb_id] = task
    task.add_done_callback(lambda t, pid=perturb_id: _task_done(pid, t))
    audit.perturb_started(perturb_id)
    return {"ok": True, "perturb_id": perturb_id, "state": "running"}


@app.get("/api/perturb")
async def perturb_list():
    return {"perturbs": list_perturbs()}


@app.get("/api/perturb/{perturb_id}")
async def perturb_detail(perturb_id: str):
    if not is_valid_perturb_id(perturb_id):
        raise HTTPException(400, "invalid perturb_id format")
    data = load_perturb(perturb_id)
    if data is None:
        raise HTTPException(404, "perturb not found")
    return _settle_perturb(data)


# ---- 排行榜（迭代六）----

@app.post("/api/leaderboard", response_model=None)
async def create_leaderboard(req: LeaderboardRequest):
    """由 N 个已完成 job（同一评测集）聚合排行榜。校验失败 400。"""
    try:
        data = build_leaderboard(req.job_ids, name=req.name)
    except LeaderboardError as e:
        raise HTTPException(400, str(e))
    lb_id = "lb_" + create_job_id()
    payload = {
        "lb_id": lb_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **data,
    }
    save_leaderboard(lb_id, payload)
    audit.leaderboard_created(lb_id)
    return {"ok": True, "lb_id": lb_id, "models": payload["models"]}


@app.get("/api/leaderboard")
async def leaderboard_list():
    return {"leaderboards": list_leaderboards()}


@app.get("/api/leaderboard/{lb_id}")
async def leaderboard_detail(lb_id: str):
    if not is_valid_lb_id(lb_id):
        raise HTTPException(400, "invalid lb_id format")
    data = load_leaderboard(lb_id)
    if data is None:
        raise HTTPException(404, "leaderboard not found")
    return data


# ---- KPI 看板（迭代六）----

@app.get("/api/dashboard")
async def dashboard():
    """KPI 看板：硬件利用率（CPU 增量采样，GPU N/A）+ 历史 job 耗时/token 趋势。"""
    return {
        "hw": collect_hw(),
        "jobs_trend": build_jobs_trend(list_jobs()),
    }


# ---- 任务调度视图（迭代七）----

def _task_view_entry(job_id: str, j: dict, kind: str) -> dict:
    cfg = j.get("config") or {}
    if cfg.get("batch_id"):
        kind = "batch"
    return {
        "job_id": job_id,
        "type": kind,
        "state": j.get("state"),
        "progress": j.get("progress"),
        "model_a": (cfg.get("model_a") or {}).get("name", "?"),
        "model_b": (cfg.get("model_b") or {}).get("name", "?"),
        "created_at": j.get("created_at"),
        "priority": j.get("priority", 0),
        "batch_id": cfg.get("batch_id"),
    }


@app.get("/api/tasks")
async def tasks_view():
    """任务队列视图：排队（含位置/优先级）+ 运行中 + 批次摘要。"""
    queued = []
    for item in _SCHEDULER.queue_view():
        j = _jobs.get(item["job_id"])
        if j is None:
            continue
        entry = _task_view_entry(item["job_id"], j, "eval")
        entry["position"] = item["position"]
        entry["priority"] = item["priority"]
        queued.append(entry)
    running = [
        _task_view_entry(job_id, j, "eval")
        for job_id, j in _jobs.items()
        if _job_running(j)
    ]
    return {
        "quota": {"concurrency": _SCHEDULER.concurrency(),
                  "active": _SCHEDULER.active_count(),
                  "queued": len(queued)},
        "queued": queued,
        "running": running,
        "batches": list_batches(),
    }


@app.put("/api/tasks/{job_id}/priority")
async def tasks_set_priority(job_id: str, req: PriorityRequest):
    """调整排队中任务优先级（重排序）；运行中/终态 409；不存在 404。"""
    job_id = _require_job_id(job_id)
    j = _jobs.get(job_id)
    if j is None:
        raise HTTPException(404, "job not found")
    if j.get("state") != "queued":
        raise HTTPException(409, f"仅排队中任务可调整优先级（当前 state={j.get('state')}）")
    if not _SCHEDULER.set_priority(job_id, req.priority):
        raise HTTPException(409, "任务不在调度队列中")
    j["priority"] = req.priority
    audit.priority_changed(job_id, str(req.priority))
    position = next((x["position"] for x in _SCHEDULER.queue_view()
                     if x["job_id"] == job_id), None)
    return {"ok": True, "job_id": job_id, "priority": req.priority, "position": position}


@app.post("/api/eval/{job_id}/resume")
async def resume_eval(job_id: str):
    """断点续跑（迭代七）：磁盘态部分完成任务重新入队，跳过已完成题。

    仅对运行中崩溃/异常遗留的磁盘态任务（_jobs 无条目、磁盘 state 为
    executing/error、增量答案 >0）生效；运行中任务 409，无部分答案 409。
    """
    job_id = _require_job_id(job_id)
    if job_id in _jobs:
        raise HTTPException(409, f"任务运行中（state={_jobs[job_id].get('state')}），无需续跑")
    st = get_job_status(job_id)
    if st is None:
        raise HTTPException(404, "job not found")
    if st["state"] not in ("executing", "error"):
        raise HTTPException(409, f"任务状态 {st['state']} 不可续跑（仅 executing/error 且存在部分答案）")
    partial = partial_answers_count(job_id)
    if partial <= 0:
        raise HTTPException(409, "磁盘无部分答案，无可续跑内容")

    cfg = st["config"]
    task_set = _load_job_task_set(job_id)
    if task_set is None:
        raise HTTPException(409, "任务集缺失，无法续跑")
    _jobs[job_id] = {
        "state": "queued",
        "progress": "0/0",
        "task_set": task_set,
        "config": cfg,
        "answers_a": None,
        "answers_b": None,
        "verdict": None,
        "rounds_answers": [],
        "reveal": None,
        "started_at": time.time(),
        "created_at": (cfg.get("created_at") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        "sse_queue": asyncio.Queue(),
        "repeat_n": int(cfg.get("repeat_n") or 1),
        "budget_warned": False,
        "health_warned": False,
        "embedding_cfg": cfg.get("embedding"),
        "priority": 0,
        "resumed": True,
    }
    _SCHEDULER.submit(job_id, priority=0)
    await _dispatch_pending()
    audit.eval_resumed(job_id)
    return {"ok": True, "job_id": job_id, "state": "queued", "partial": partial}


# ---- benchmark 批次（迭代七）----

def _on_batch_job_done(batch_id: str, job_id: str) -> None:
    """批次子任务终态回调（幂等计数）：全部终态时聚合排行榜并置批次终态。

    completed job 聚合进排行榜（build_leaderboard 复用）；error/cancelled
    模型标注 N/A 排除；不足 2 个完成模型 → 无排行榜 + note。
    """
    batch = load_batch(batch_id)
    if batch is None or batch.get("state") in ("done", "partial"):
        return
    states: dict[str, str] = {}
    for jid in batch.get("jobs", []):
        if jid in _jobs:
            states[jid] = _jobs[jid].get("state", "executing")
        else:
            st = get_job_status(jid)
            states[jid] = (st or {}).get("state", "executing")
    if any(states.get(jid) not in TERMINAL_STATES for jid in batch.get("jobs", [])):
        save_batch(batch_id, batch)
        return
    completed = [jid for jid in batch.get("jobs", []) if states.get(jid) == "completed"]
    failed = [jid for jid in batch.get("jobs", []) if states.get(jid) != "completed"]
    partial = bool(failed)
    batch["state"] = "partial" if partial else "done"
    batch["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    batch["failed_models"] = [
        (_jobs[jid].get("config", {}).get("model_a", {}).get("name", "?")
         if jid in _jobs and _jobs[jid].get("config")
         else (get_job_status(jid) or {}).get("config", {}).get("model_a", {}).get("name", "?"))
        for jid in failed
    ]
    if len(completed) >= 2:
        try:
            lb = build_leaderboard(completed, name=batch.get("name"))
            lb_id = "lb_" + create_job_id()
            save_leaderboard(lb_id, {
                "lb_id": lb_id,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "batch_id": batch_id,
                **lb,
            })
            batch["leaderboard_id"] = lb_id
            audit.leaderboard_created(lb_id)
        except LeaderboardError as e:
            batch["aggregation_error"] = str(e)
    else:
        batch["aggregation_error"] = "不足 2 个完成模型，无法生成排行榜"
    save_batch(batch_id, batch)
    audit.benchmark_done(batch_id, partial)


async def _run_batch_job(job_id: str):
    """batch 执行单元：单模型 M 轮作答 + 单臂评分 → 单臂 verdict + slim report。

    判别式/代码题 = score_task_metric 逐轮均值（0-10）；生成式题 =
    run_single_arm_judge 逐轮单臂分均值（judge 缺省该类题 N/A + warning）。
    """
    j = _jobs.get(job_id)
    if j is not None and j.get("state") == "queued":
        j["state"] = "executing"
        save_task_set(job_id, j["task_set"])
        save_env_snapshot(job_id)
        await _push_event(job_id, {"state": "executing", "progress": "0/0"})
    try:
        j = _jobs.get(job_id)
        if j is None:
            return
        cfg = j["config"]
        task_set = j["task_set"]
        model_cfg = cfg["model_a"]
        rounds = int(j.get("repeat_n", 1)) or 1

        async def progress_cb(_label, done, total):
            jj = _jobs.get(job_id)
            if jj is None or jj.get("state") == "cancelling":
                return
            jj["progress"] = f"{done}/{total}"
            await _push_event(job_id, {"state": "executing", "progress": f"{done}/{total}"})

        all_rounds: list[dict] = []
        for r_idx in range(rounds):
            if _job_cancelled(job_id):
                raise JobCancelled
            answers = await _execute_model("A", model_cfg, task_set["tasks"], None,
                                           progress_cb)
            save_answers(job_id, f"a-r{r_idx + 1}", answers)
            all_rounds.append(answers)
        save_answers(job_id, "a", all_rounds[-1])
        save_answers(job_id, "b", {"model": model_cfg["name"], "answers": []})

        if _job_cancelled(job_id):
            raise JobCancelled

        judge_cfg = (cfg.get("review") or {}).get("judge")
        gen_tasks = [t for t in task_set["tasks"] if t.get("type") == "生成式"]

        disc_scores: dict[str, list[float]] = {}
        gen_scores: dict[str, list[float]] = {}
        invalid = 0
        for round_ans in all_rounds:
            ans_map = {a["id"]: a for a in round_ans.get("answers", [])}
            for t in task_set["tasks"]:
                if t.get("type") != "生成式":
                    entry = ans_map.get(t["id"])
                    if entry is None:
                        continue
                    s = score_task_metric(t, entry)
                    if s is not None:
                        disc_scores.setdefault(t["id"], []).append(s)
            if judge_cfg and gen_tasks:
                gen_answers = [ans_map[t["id"]] for t in gen_tasks if t["id"] in ans_map]
                if gen_answers:
                    sa = await run_single_arm_judge(
                        {"tasks": gen_tasks, "meta": {}},
                        {"model": model_cfg["name"], "answers": gen_answers},
                        judge_cfg)
                    invalid += sa.get("meta", {}).get("invalid", 0)
                    for v in sa.get("scores", []):
                        if not v.get("_invalid"):
                            gen_scores.setdefault(v["id"], []).append(v.get("score", 0.0))

        scores: list[dict] = []
        warnings: list[dict] = []
        for t in task_set["tasks"]:
            tid = t["id"]
            if t.get("type") == "生成式":
                vals = gen_scores.get(tid, [])
                score = round(sum(vals) / len(vals), 2) if vals else None
                if score is None and judge_cfg is None:
                    warnings.append({"code": "batch_judge_missing", "task_id": tid,
                                     "message": "未提供 judge 配置，生成式题得分 N/A"})
            else:
                vals = disc_scores.get(tid, [])
                score = round(sum(vals) / len(vals), 2) if vals else None
            scores.append({"id": tid, "dimension": t.get("dimension", ""),
                           "score": score, "basis": "批内单臂评分（M 轮均值）",
                           "_invalid": score is None})

        valid = [s for s in scores if not s["_invalid"]]
        verdict = {
            "meta": {"total": len(scores), "valid": len(valid),
                     "invalid": len(scores) - len(valid),
                     "rounds": rounds, "mode": "single_arm",
                     "excluded_ids": [t["id"] for t in task_set["tasks"]
                                      if t.get("excluded_from_total")],
                     "excluded_dimensions": sorted({t.get("dimension", "")
                                                    for t in task_set["tasks"]
                                                    if t.get("excluded_from_total")})},
            "scores": scores,
            "totals": {"score": round(sum(s["score"] or 0 for s in valid), 2),
                       "max": round(10.0 * len(valid), 2)},
        }
        save_verdict(job_id, verdict)
        report = {
            "summary": {"model": model_cfg["name"], "n_tasks": len(scores),
                        "rounds": rounds, "mode": "single_arm",
                        "invalid": verdict["meta"]["invalid"]},
            "kpi": {"duration_sec": _job_duration_sec(job_id, cfg)},
            "warnings": warnings,
        }
        save_report(job_id, {
            "config": sanitize_config(cfg),
            "tasks": task_set,
            "answers_a": all_rounds[-1] if all_rounds else {},
            "answers_b": {"model": model_cfg["name"], "answers": []},
            "verdict": verdict,
            "report": report,
        })
        if job_id in _jobs:
            _jobs[job_id].update({"verdict": verdict, "state": "completed",
                                  "progress": "done"})
        await _push_event(job_id, {"state": "completed"})
        _on_batch_job_done(cfg["batch_id"], job_id)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        j = _jobs.get(job_id)
        if j is not None:
            j["state"] = "error"
            j["error"] = f"批次执行失败: {e}"
            await _push_event(job_id, {"state": "error", "error": str(e)})
        _mark_error(job_id, f"批次执行失败: {e}")
        cfg = (_jobs.get(job_id, {}).get("config") or {})
        if cfg.get("batch_id"):
            _on_batch_job_done(cfg["batch_id"], job_id)


@app.post("/api/benchmark", response_model=None)
async def create_benchmark(req: BenchmarkRequest):
    """创建 benchmark 批次：1 任务集 × N 模型（配置库）× M 轮。

    每模型一个执行单元（单臂评审），全部经调度器入队；Key 取进程内存。
    """
    dataset = load_dataset(req.dataset_name)
    if dataset is None:
        raise HTTPException(404, f"评测集 '{req.dataset_name}' 不存在")
    try:
        task_set = build_task_set_from_dataset(dataset)
    except DatasetValidationError as e:
        raise HTTPException(400, f"数据集格式错误: {e}")

    seen_ids: set[str] = set()
    models: list[dict] = []
    for mid in req.model_ids:
        if mid in seen_ids:
            raise HTTPException(400, f"模型配置重复：{mid}")
        seen_ids.add(mid)
        info = get_model(mid)
        if info is None:
            raise HTTPException(400, f"模型配置不存在：{mid}")
        key = get_key(mid)
        if not key:
            raise HTTPException(
                400, f"模型「{info.get('name')}」未补录 API Key（配置库重启后需重新填写）")
        models.append({
            "id": mid, "name": info["name"], "url": info["url"], "key": key,
            "temperature": info.get("temperature", 0.7),
            "max_tokens": info.get("max_tokens", 4096),
            "top_p": info.get("top_p"),
        })

    judge_cfg = None
    if req.review is not None and req.review.judge is not None:
        judge_cfg = {"name": req.review.judge.name, "url": req.review.judge.url,
                     "key": req.review.judge.key or ""}
        try:
            validate_upstream_url(req.review.judge.url)
        except UpstreamUrlError as e:
            raise HTTPException(400, f"评审模型 URL 校验失败: {e}")

    budget_check = check_budget(
        req.budget.model_dump() if req.budget else None,
        task_set["meta"]["total"], req.rounds,
        "pure_agent" if judge_cfg else "human", len(models),
    )
    if budget_check["limited"] and not budget_check["allowed"]:
        raise HTTPException(
            400,
            f"预算超限：预估消耗 {budget_check['estimated']} token"
            f"（{len(models)} 模型 × {req.rounds} 轮），超过上限 "
            f"{budget_check['limit']}（hard 模式拒绝启动）",
        )

    batch_id = "batch_" + create_job_id()
    job_ids: list[str] = []
    for m in models:
        jid = create_job_id()
        config_data = {
            "model_a": {k: m[k] for k in ("name", "url", "key", "temperature",
                                          "max_tokens", "top_p")},
            "model_a_key_masked": "***",
            "dims": None, "seed": None, "dataset_name": req.dataset_name,
            "repeat_n": req.rounds, "num_questions": None,
            "code_verify_mode": req.code_verify_mode,
            "prompt_strategy": req.prompt_strategy,
            "review": {"mode": "single_arm", "fail_open": False,
                       "judge": judge_cfg, "k_top_human": 0},
            "budget": req.budget.model_dump() if req.budget else None,
            "embedding": req.embedding.model_dump() if req.embedding else None,
            "batch_id": batch_id,
            "model_id": m["id"],
        }
        save_config(jid, config_data)
        _jobs[jid] = {
            "state": "queued",
            "progress": "0/0",
            "task_set": task_set,
            "config": config_data,
            "answers_a": None, "answers_b": None, "verdict": None,
            "rounds_answers": [], "reveal": None,
            "started_at": time.time(),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sse_queue": asyncio.Queue(),
            "repeat_n": req.rounds,
            "budget_warned": False, "health_warned": False,
            "embedding_cfg": req.embedding.model_dump() if req.embedding else None,
            "priority": req.priority,
        }
        _SCHEDULER.submit(jid, priority=req.priority)
        job_ids.append(jid)

    batch = {
        "batch_id": batch_id,
        "name": req.name or "",
        "state": "running",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": req.dataset_name,
        "models": [m["name"] for m in models],
        "rounds": req.rounds,
        "jobs": job_ids,
        "leaderboard_id": None,
        "failed_models": [],
        "aggregation_error": None,
        "finished_at": None,
    }
    save_batch(batch_id, batch)
    audit.benchmark_started(batch_id, len(models))
    await _dispatch_pending()
    return {"ok": True, "batch_id": batch_id, "jobs": job_ids,
            "models": batch["models"], "state": "running"}


@app.get("/api/benchmark")
async def benchmark_list():
    return {"batches": list_batches()}


@app.get("/api/benchmark/{batch_id}/leaderboard")
async def benchmark_leaderboard(batch_id: str):
    if not is_valid_batch_id(batch_id):
        raise HTTPException(400, "invalid batch_id format")
    batch = load_batch(batch_id)
    if batch is None:
        raise HTTPException(404, "batch not found")
    lb_id = batch.get("leaderboard_id")
    if not lb_id:
        raise HTTPException(404, "排行榜尚未生成（批次未完成或完成模型不足 2 个）")
    data = load_leaderboard(lb_id)
    if data is None:
        raise HTTPException(404, "leaderboard not found")
    return data


@app.get("/api/benchmark/{batch_id}")
async def benchmark_detail(batch_id: str):
    if not is_valid_batch_id(batch_id):
        raise HTTPException(400, "invalid batch_id format")
    batch = load_batch(batch_id)
    if batch is None:
        raise HTTPException(404, "batch not found")
    jobs = []
    for jid in batch.get("jobs", []):
        if jid in _jobs:
            st = _jobs[jid].get("state")
            prog = _jobs[jid].get("progress")
        else:
            st = (get_job_status(jid) or {}).get("state")
            prog = None
        jobs.append({"job_id": jid, "state": st, "progress": prog})
    n_terminal = sum(1 for x in jobs if x["state"] in TERMINAL_STATES)
    n_total = len(jobs)
    return {
        **batch,
        "progress": f"{n_terminal}/{n_total}",
        "terminal": n_terminal,
        "jobs": jobs,
    }


@app.get("/api/generate")
async def generate_list():
    return {"batches": list_generation_batches()}


@app.get("/api/generate/{gen_id}")
async def generate_detail(gen_id: str):
    _require_gen_id(gen_id)
    batch = load_generation_batch(gen_id)
    if batch is None:
        raise HTTPException(404, "generation batch not found")
    batch = _settle_generation(batch)
    return {
        "gen_id": batch["gen_id"],
        "state": batch["state"],
        "error": batch.get("error"),
        "created_at": batch.get("created_at", ""),
        "spec": batch.get("spec", {}),
        "item_stats": _item_stats(batch.get("items", [])),
        "items": batch.get("items", []),
    }


@app.post("/api/generate/{gen_id}/items/{item_id}/review")
async def review_generated(gen_id: str, item_id: str, req: ReviewDecisionRequest):
    """审核提交：approve（可选 edits）→ 目标数据集入库 + 版本递增；reject 终态。"""
    _require_gen_id(gen_id)
    batch = load_generation_batch(gen_id)
    if batch is None:
        raise HTTPException(404, "generation batch not found")
    batch = _settle_generation(batch)
    if batch["state"] not in ("ready", "partial"):
        raise HTTPException(409, f"批次尚未完成生成（state={batch['state']}），无法审核")
    if not item_id.startswith(gen_id + "-"):
        raise HTTPException(404, "item not found")
    item = next((it for it in batch.get("items", []) if it.get("item_id") == item_id), None)
    if item is None:
        raise HTTPException(404, "item not found")
    if item.get("status") != "pending":
        raise HTTPException(409, f"该题目已审核（status={item.get('status')}），驳回为终态")

    if req.action == "reject":
        item["status"] = "rejected"
        item["reviewed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save_generation_batch(gen_id, batch)
        audit.task_reviewed(item_id, "reject")
        return {"ok": True, "status": "rejected"}

    # approve：合并人工编辑（字段白名单 E6）
    task = dict(item.get("task", {}))
    if req.edits:
        bad = [k for k in req.edits if k not in EDIT_ALLOWED_FIELDS]
        if bad:
            raise HTTPException(400, f"不允许编辑字段: {', '.join(sorted(bad))}")
        task = {**task, **{k: v for k, v in req.edits.items() if v is not None}}

    # 单题校验（E1：包装为整集校验路径，不调私有函数）
    try:
        validated = validate_standard_dataset({"name": "pending", "tasks": [task]})
        task = validated["tasks"][0]
    except DatasetValidationError as e:
        raise HTTPException(400, f"题目校验失败: {e}")

    # 目标数据集：缺省自动命名；追加 + 版本递增 + 来源标注
    spec = batch.get("spec", {})
    ds_name = spec.get("target_dataset") or f"LLM生成集_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    dataset = load_dataset(ds_name) or {
        "name": ds_name,
        "description": "由出题审核生成（LLM 出题 + 人工审核）",
        "tasks": [],
    }
    if len(dataset.get("tasks", [])) >= MAX_DATASET_TASKS:
        raise HTTPException(400, f"数据集题目数已达上限 {MAX_DATASET_TASKS}，无法追加")
    # 单题校验赋予的自动 id（T1）可能与既有题目冲突：改为按现有 id 扫描取最小空闲号
    used_ids = {str(t.get("id", "")) for t in dataset.get("tasks", []) if isinstance(t, dict)}
    n = 1
    while f"T{n}" in used_ids:
        n += 1
    task["id"] = f"T{n}"
    next_version = bump_dataset_version(dataset.get("version") or "v0")
    dataset["version"] = next_version
    dataset["source"] = "generated"
    dataset["tasks"] = list(dataset.get("tasks", [])) + [task]
    try:
        save_dataset(ds_name, dataset)
    except DatasetValidationError as e:
        raise HTTPException(400, f"数据集校验失败: {e}")

    item["status"] = "approved"
    item["edits"] = req.edits
    item["dataset"] = ds_name
    item["version"] = next_version
    item["reviewed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_generation_batch(gen_id, batch)
    audit.task_reviewed(item_id, "approve")
    return {"ok": True, "status": "approved", "dataset": ds_name, "version": next_version}


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


# ---- 跨 job 历史汇总与饱和度监测（迭代一接口 + 迭代五趋势） ----

@app.get("/api/stats/saturation")
async def stats_saturation():
    """跨 job 逐题结果汇总表 + 饱和度趋势（jobs 字段兼容旧客户端，trend 为迭代五新增）。"""
    data = get_saturation()
    return redact_sensitive({
        "jobs": data.get("jobs", []),
        "trend": saturation_trend(data),
    })


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

    # 迭代七：并发超限不再 429 拒绝，改由调度器排队（优先级队列 + 配额派发）
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
    # 迭代七：tasks.json/env 快照延后到调度派发时落盘（排队中磁盘态 = queued）

    _jobs[job_id] = {
        "state": "queued",
        "progress": "0/0",
        "task_set": task_set,
        "config": config_data,
        "answers_a": None,
        "answers_b": None,
        "verdict": None,
        "rounds_answers": [],
        "reveal": None,
        "started_at": time.time(),  # 迭代六：KPI 看板评测耗时（duration_sec）
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sse_queue": asyncio.Queue(),
        "repeat_n": req.repeat_n,
        "budget_warned": False,
        "health_warned": False,
        "embedding_cfg": req.embedding.model_dump() if req.embedding else None,
        "priority": 0,
    }

    _SCHEDULER.submit(job_id, priority=0)
    audit.eval_started(job_id)
    await _dispatch_pending()
    j = _jobs.get(job_id)
    # 已派发（running 集含本 job）则不推 queued；仅真正排队时推送
    if (j is not None and j.get("state") == "queued"
            and job_id not in _SCHEDULER.running()):
        position = next((x["position"] for x in _SCHEDULER.queue_view()
                         if x["job_id"] == job_id), None)
        await _push_event(job_id, {"state": "queued", "position": position})
        audit.eval_queued(job_id)
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


def _load_job_task_set(job_id: str) -> dict | None:
    """从磁盘读取任务集（resume 路径：进程重启后 _jobs 无条目）。"""
    files = get_job_files(job_id)
    tasks = files.get("tasks.json") if files else None
    return tasks if isinstance(tasks, dict) else None


def _merge_inc_answers(answers: dict, inc_by_task: dict) -> dict:
    """把磁盘增量答案按 task_id 并回答案池（resume 跳过题补齐两侧）。"""
    if not inc_by_task:
        return answers
    have = {e.get("id") for e in answers.get("answers", [])}
    merged = list(answers.get("answers", []))
    for tid, entry in sorted(inc_by_task.items()):
        if tid not in have:
            merged.append(entry)
    return {**answers, "answers": merged}


async def _run_eval(job_id: str):
    """后台执行入口：包装 _run_eval_impl，取消时统一收尾并终止任务。

    迭代七：派发时由 queued → executing，并在此刻落盘 tasks.json 与环境快照
    （排队中磁盘态仅 config.json，保证状态推断一致）。
    """
    j = _jobs.get(job_id)
    if j is not None and j.get("state") == "queued":
        j["state"] = "executing"
        save_task_set(job_id, j["task_set"])
        save_env_snapshot(job_id)  # 迭代四：环境快照（OS/Python/依赖版本，无密钥）
        await _push_event(job_id, {"state": "executing", "progress": "0/0"})
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
            # 迭代七断点续跑：resume 任务跳过磁盘已完成题（双侧均有答案才跳过），
            # 并通过 persist_cb 继续增量落盘（本轮新完成题随时可再次续跑）
            if j.get("resumed"):
                inc = load_answers_inc(job_id)
                skip_ids = {
                    tid for tid in {k.rsplit(":", 1)[1] for k in inc}
                    if f"a:{tid}" in inc and f"b:{tid}" in inc
                }
                if skip_ids:
                    _exec_kwargs["skip_ids"] = skip_ids
                _exec_kwargs["persist_cb"] = lambda label, tid, e: save_answers_inc(
                    job_id, f"{label}:{tid}", e)
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

        # resume 续跑：跳过题的两侧答案从磁盘增量合并回答案池
        if j.get("resumed"):
            inc = load_answers_inc(job_id)
            if inc:
                by_side = {"a": {}, "b": {}}
                for key, entry in inc.items():
                    side, _, tid = key.rpartition(":")
                    if side in by_side:
                        by_side[side][tid] = entry
                answers_a = _merge_inc_answers(answers_a, by_side["a"])
                answers_b = _merge_inc_answers(answers_b, by_side["b"])

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
        # 进入人工评审阶段，等待用户打分（迭代七：释放调度配额，后续任务可派发）
        _scheduler_release(job_id)
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
            _scheduler_release(job_id)  # 迭代七：降级人工评审即释放配额
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
        _scheduler_release(job_id)  # 迭代七：hybrid 进入人工复核即释放配额
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


def _job_duration_sec(job_id: str, cfg: dict) -> float | None:
    """评测耗时（秒）：内存 started_at 优先；重启恢复路径回退 config.created_at。

    两处均缺失（极旧历史记录）→ None（前端空态 N/A）。
    """
    t0 = None
    j = _jobs.get(job_id)
    if j and j.get("started_at"):
        t0 = float(j["started_at"])
    else:
        ca = (cfg or {}).get("created_at") or ""
        if ca:
            try:
                t0 = datetime.fromisoformat(ca.replace("Z", "+00:00")).timestamp()
            except ValueError:
                t0 = None
    if t0 is None:
        return None
    return round(time.time() - t0, 1)


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
    # 迭代六：KPI 看板评测耗时（内存 started_at 优先，磁盘路径回退 config.created_at）
    report["kpi"]["duration_sec"] = _job_duration_sec(job_id, cfg)
    report["env_snapshot"] = load_env_snapshot(job_id) or collect_env_snapshot()
    save_report(job_id, {
        "config": sanitize_config(cfg),
        "tasks": task_set,
        "answers_a": answers_a,
        "answers_b": answers_b,
        "verdict": verdict,
        "report": report,
    })
    _finalize_badcases(job_id, verdict, task_set, answers_a, answers_b, report, cfg)
    if job_id in _jobs:
        j = _jobs[job_id]
        j["verdict"] = verdict
        j["answers_a"] = answers_a
        j["answers_b"] = answers_b
        j["rounds_answers"] = rounds_answers or j.get("rounds_answers")
        j["round_verdicts"] = round_verdicts
        j["state"] = "completed"
        j["progress"] = "done"


# ---- Bad Case 与饱和度（迭代五） ----

def _finalize_badcases(
    job_id: str, verdict: dict, task_set: dict,
    answers_a: dict, answers_b: dict, report: dict, cfg: dict,
) -> None:
    """job 完成收尾：saturation 幂等写入 + bad case 同步挖掘入库 + 后台异步归因。

    挖掘为纯规则（零网络，即时产出满足「一次评测自动产出 bad case 库」）；
    LLM 归因复用 review.judge 配置异步跑批，失败逐条静默降级「未归类」。
    """
    # 1. 跨 job 历史汇总（饱和度监测数据源，按 job_id 幂等）
    type_map = {t.get("id"): t.get("type", "判别式") for t in task_set.get("tasks", [])}
    entries = [
        {
            "id": s.get("id"),
            "dimension": s.get("dimension", ""),
            "type": type_map.get(s.get("id"), "判别式"),
            "answer_x": s.get("answer_x"),
            "answer_y": s.get("answer_y"),
            "winner": s.get("winner", "tie"),
        }
        for s in verdict.get("scores", [])
    ]
    update_saturation(job_id, entries, dataset=cfg.get("dataset_name"))

    # 2. bad case 挖掘入库（同步）
    answers_x, answers_y = reveal_answers(answers_a, answers_b, verdict)
    per_task_metrics = (report.get("metrics") or {}).get("per_task") if isinstance(report, dict) else None
    cases = mine_bad_cases(job_id, task_set, verdict, answers_x, answers_y,
                           per_task_metrics, dataset_name=cfg.get("dataset_name"))
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for c in cases:
        c["created_at"] = now
        try:
            save_badcase(c)
        except Exception:
            continue
    if cases:
        audit.badcase_mined(job_id, len(cases))

    # 3. 后台异步 LLM 归因（复用 review.judge；无配置/失败则保持未归类）
    review = cfg.get("review") or {}
    judge_cfg = review.get("judge") if isinstance(review, dict) else None
    if cases and judge_cfg and judge_cfg.get("url"):
        task_map = {t.get("id"): t for t in task_set.get("tasks", [])}
        _spawn_attribution(job_id, cases, task_map, judge_cfg)


def _spawn_attribution(
    job_id: str, cases: list[dict], task_map: dict, judge_cfg: dict,
) -> None:
    """在独立 daemon 线程的事件循环中跑归因跑批。

    _finalize_badcases 处于同步链深处，若直接 create_task，归因协程可能
    在宿主事件循环收尾（asyncio.run 驱动的测试）前从未被调度。独立线程
    自带 loop（asyncio.run），测试与真实环境均确定运行；daemon 不阻塞主线程。
    """
    import threading

    def _runner():
        try:
            asyncio.run(_attribute_cases_async(job_id, cases, task_map, judge_cfg))
        except Exception:
            pass

    threading.Thread(target=_runner, name=f"badcase-{job_id}", daemon=True).start()


async def _attribute_cases_async(
    job_id: str, cases: list[dict], task_map: dict, judge_cfg: dict,
) -> None:
    """后台归因跑批：逐条 LLM 归因，失败保持未归类；整体不抛异常。"""
    for c in cases:
        try:
            task = task_map.get(c["task_id"]) or {}
            result = await attribute_badcase(c, task, judge_cfg)
            if result is None:
                continue
            updated = update_badcase_attribution(c["case_id"], result)
            if updated is not None:
                audit.badcase_attribution(c["case_id"], "llm")
        except Exception:
            continue


def _require_badcase_id(case_id: str) -> str:
    """校验 bad case id 为系统生成格式（防路径穿越）。"""
    from backend.storage import is_valid_badcase_id
    if not is_valid_badcase_id(case_id):
        raise HTTPException(400, "invalid case_id format")
    return case_id


@app.get("/api/badcases")
async def badcases_list(job_id: str | None = None, category: str | None = None,
                        confirmed: str | None = None):
    """bad case 列表（按 job/分类/确认态筛选，新→旧）。"""
    cases = list_badcases(job_id)
    if category:
        cases = [c for c in cases if c["category"] == category]
    if confirmed is not None:
        want = confirmed.lower() in ("1", "true", "yes")
        cases = [c for c in cases if c["confirmed"] == want]
    return {"total": len(cases), "cases": cases}


@app.get("/api/badcases/stats")
async def badcases_stats(job_id: str | None = None):
    """分类分布 + 来源分布 + 总数（图表数据源）。"""
    cases = list_badcases(job_id)
    by_category: dict[str, int] = {}
    by_source: dict[str, int] = {}
    confirmed = 0
    for c in cases:
        by_category[c["category"]] = by_category.get(c["category"], 0) + 1
        for s in c.get("sources", []):
            by_source[s] = by_source.get(s, 0) + 1
        if c.get("confirmed"):
            confirmed += 1
    return {
        "total": len(cases),
        "confirmed": confirmed,
        "by_category": by_category,
        "by_source": by_source,
        "categories": list(BAD_CASE_CATEGORIES) + [UNCATEGORIZED],
    }


@app.get("/api/badcases/export")
async def badcases_export(job_id: str | None = None):
    """导出 bad case 清单 JSON（含修订建议），可指定 job。"""
    from fastapi.responses import Response
    content = export_badcases_json(job_id)
    fname = f"badcases-{job_id}.json" if job_id else "badcases-all.json"
    return Response(
        content=content,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@app.get("/api/badcases/{case_id}")
async def badcase_detail(case_id: str):
    """bad case 详情（含证据全量，供归因报告与跳转报告原文）。"""
    _require_badcase_id(case_id)
    case = load_badcase(case_id)
    if case is None:
        raise HTTPException(404, "bad case not found")
    return redact_sensitive(case)


@app.post("/api/badcases/{case_id}/attribution")
async def badcase_attribution_confirm(case_id: str, req: dict):
    """人工确认/改标归因（body: {category, suggestion?}；驳回传 category=null）。"""
    _require_badcase_id(case_id)
    case = load_badcase(case_id)
    if case is None:
        raise HTTPException(404, "bad case not found")
    category = req.get("category")
    if category is not None and category not in BAD_CASE_CATEGORIES:
        raise HTTPException(400, f"category 必须是五类之一: {', '.join(BAD_CASE_CATEGORIES)}")
    label = category or UNCATEGORIZED
    suggestion = req.get("suggestion")
    if suggestion is not None and not isinstance(suggestion, str):
        raise HTTPException(400, "suggestion 必须是字符串")
    updated = update_badcase_attribution(case_id, {
        "label": label,
        "by": "human",
        "confirmed": True,
        "basis": req.get("basis") or (case.get("attribution") or {}).get("basis") or "",
        "suggestion": suggestion if suggestion is not None else (case.get("attribution") or {}).get("suggestion") or "",
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    audit.badcase_attribution(case_id, "confirm" if category else "revert")
    return {"ok": True, "case_id": case_id, "category": label, "confirmed": True}


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
                payload["report"]["env_snapshot"] = (
                    load_env_snapshot(job_id) or collect_env_snapshot()
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


@app.get("/api/eval/{job_id}/export")
async def eval_export(job_id: str):
    """导出评测包（迭代四）：仅 completed 可导出 zip + MANIFEST sha256。

    未完成/运行中 409；非法 job_id 或任务不存在 404（复用 _require_job_id
    与 get_job_status 的磁盘态判定，重启后仍可导出）。
    """
    job_id = _require_job_id(job_id)
    st = get_job_status(job_id)
    if st is None:
        raise HTTPException(404, "job not found")
    if st["state"] != "completed":
        raise HTTPException(409, f"任务未完成（state={st['state']}），完成后方可导出")

    data = build_export_zip(job_id)
    audit.dataset_exported(job_id)
    filename = f"eval-{job_id}.zip"
    return StreamingResponse(
        iter([data]),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


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
    # 迭代七：排队中取消走独立分支（出队 + 删目录 + 审计，无 task.cancel）
    j_entry = _jobs.get(job_id)
    if task is None and j_entry is not None and j_entry.get("state") == "queued":
        _SCHEDULER.cancel_queued(job_id)
        _jobs.pop(job_id, None)
        ok = delete_job(job_id)
        if not ok:
            raise HTTPException(404, "job not found")
        audit.eval_cancelled(job_id)
        return {"ok": True}
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
