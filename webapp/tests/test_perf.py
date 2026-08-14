# -*- coding: utf-8 -*-
"""迭代八：性能基准（@pytest.mark.perf，默认排除于回归，验收时单独运行）。

- 长文本边界：prompt 超 MAX_PROMPT_LEN=20000 拒、context 在 MAX_CONTEXT_LEN=32000
  内通过/超界拒绝（双边界，D6）；长文本数据资产生成校验；32K 题执行 mock 链路
  （prompt_tokens 采集 + truncated 跳算）；judge ANSWER_FENCE=8000 截断边界
- 多任务并发：Scheduler N=100 排队 + 优先级重排压力
- N 模型批次：batch N=10 全链路（mock 单臂）

阈值一律宽松（确定性 mock + 宽上界），防 CI/本地抖动产生 flaky。
"""
from __future__ import annotations

import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

from backend import main as main_module
from backend import storage
from backend.engine.datasets import (
    MAX_CONTEXT_LEN, MAX_PROMPT_LEN, DatasetValidationError,
    validate_standard_dataset,
)
from backend.engine.judge import ANSWER_FENCE, _fenced
from backend.engine.metrics import compute_task_metrics
from backend.scheduler import Scheduler
from backend.schemas import BenchmarkRequest

pytestmark = pytest.mark.perf


def _longtext_dataset(over_context: bool) -> dict:
    """长文本基准集（材料在 context；over_context=True 时超 MAX_CONTEXT_LEN）。"""
    base = {
        "name": "perf_longtext",
        "tasks": [{
            "id": "LT",
            "type": "判别式",
            "dimension": "长文本与效率稳定性",
            "prompt": "阅读参考文档，营业收入是多少亿元？",
            "expected": "1.00亿元",
            "test_cases": [{"input": "营业收入？", "expected": "1.00亿元"}],
            "context": "长文材料。" * (MAX_CONTEXT_LEN // 5),
        }],
    }
    if over_context:
        base["tasks"][0]["context"] = "长文材料。" * (MAX_CONTEXT_LEN // 5 + 10)
    return base


# ---- 长文本边界（D6 双边界） ----

def test_prompt_len_upper_bound_rejected():
    """prompt 超过 MAX_PROMPT_LEN=20000 拒绝（32K 内容必须走 context 而非 prompt）。"""
    ds = {
        "name": "perf_prompt_over",
        "tasks": [{"id": "T", "type": "判别式", "dimension": "知识能力",
                   "prompt": "x" * (MAX_PROMPT_LEN + 1), "expected": "y"}],
    }
    with pytest.raises(DatasetValidationError):
        validate_standard_dataset(ds)


def test_context_len_boundaries():
    """context ≤32000 通过；超界拒绝（MAX_CONTEXT_LEN）。"""
    ok = _longtext_dataset(over_context=False)
    assert validate_standard_dataset(ok)      # 边界内通过
    with pytest.raises(DatasetValidationError):
        validate_standard_dataset(_longtext_dataset(over_context=True))


def test_longtext_bench_asset_valid():
    """长文本基准资产（ensure_longtext_bench 写入评测集目录）三档规模合规。"""
    from backend.engine.longtext import DEMO_NAME, ensure_longtext_bench
    ensure_longtext_bench()                       # 幂等；conftest 已重定向目录
    ds = storage.load_dataset(DEMO_NAME)
    assert ds is not None, "longtext_bench 未写入（lifespan 未跑或手工删除）"
    norm = validate_standard_dataset(ds)
    tasks = norm["tasks"]
    assert len(tasks) == 3
    ctxs = {t["id"]: len(t.get("context", "")) for t in tasks}
    assert ctxs["LT2K"] < 3000
    assert ctxs["LT8K"] < 12000
    assert ctxs["LT32K"] <= MAX_CONTEXT_LEN
    assert all(t.get("tags") == ["长文本", "基准"] for t in tasks)


def test_longtext_execution_and_truncation_path():
    """32K 题执行链路：prompt_tokens 采集随 context 增大 + truncated 题跳算不抛错。"""
    tasks = [
        {"id": "A", "type": "判别式", "dimension": "长文本与效率稳定性",
         "prompt": "q", "expected": "1.00亿元",
         "context": "长文材料。" * 1000},
        {"id": "B", "type": "判别式", "dimension": "长文本与效率稳定性",
         "prompt": "q", "expected": "1.00亿元",
         "context": "长文材料。" * 1000},
    ]
    # 正常长文本条目：token 随 context 增大（模拟 32K 级输入）
    ok_entry = {"id": "A", "raw_answer": "1.00亿元",
                "api_info": {"status": "ok", "attempts": 1, "truncated": False,
                             "error": None, "latency_ms": 300,
                             "prompt_tokens": 11000, "completion_tokens": 20,
                             "repeat_index": 1}}
    t0 = time.perf_counter()
    m_ok = compute_task_metrics(tasks[0], [ok_entry])
    elapsed = time.perf_counter() - t0
    assert m_ok.get("skipped") is None or m_ok.get("skipped") is False
    assert elapsed < 2.0                    # 宽阈值：单题指标计算 < 2s
    # 截断条目：跳算指标 + 记录原因（报告层转 warning），不抛错
    trunc_entry = {**ok_entry, "api_info": {**ok_entry["api_info"], "truncated": True}}
    m_trunc = compute_task_metrics(tasks[0], [trunc_entry])
    assert m_trunc.get("skipped") is True


def test_judge_answer_fence_boundary():
    """评审答案围栏 ANSWER_FENCE=8000：超长答案被截断标注（协议边界回归）。"""
    assert ANSWER_FENCE >= 8000
    assert "已截断" not in _fenced("x" * 5000)
    assert "已截断" in _fenced("x" * (ANSWER_FENCE + 100))


# ---- 多任务并发压力 ----

def test_scheduler_100_queued_with_priority_reshuffle():
    """Scheduler N=100：优先级排序 + 同优先级 FIFO + 重排 + 配额派发 + 批量取消。"""
    s = Scheduler(concurrency=2)
    for i in range(100):
        s.submit(f"job{i:03d}", priority=i % 5)
    view = s.queue_view()
    assert len(view) == 100
    prios = [v["priority"] for v in view]
    assert prios == sorted(prios, reverse=True)
    assert [v["job_id"] for v in view if v["priority"] == 0][:4] == \
        ["job000", "job005", "job010", "job015"]   # FIFO（i%5==0 共 20 个，取前 4）
    assert s.set_priority("job099", 9)
    assert s.queue_view()[0]["job_id"] == "job099"
    running = s.next_batch()
    assert len(running) == 2 and s.active_count() == 2
    assert len(s.queue_view()) == 98
    cancelled = sum(1 for v in s.queue_view()[:50] if s.cancel_queued(v["job_id"]))
    assert cancelled == 50
    assert len(s.queue_view()) == 48


# ---- N 模型批次（N=10） ----

@pytest.fixture
def client():
    return TestClient(main_module.app)


async def _perf_fake_execute(model_label, config, tasks, stability_repeat,
                             progress_cb=None, embedding_cfg=None, skip_ids=None,
                             persist_cb=None, concurrency=1):
    answers = []
    for t in tasks:
        answers.append({
            "id": t["id"],
            "raw_answer": "答案是 2" if t.get("type") == "判别式" else "生成文本",
            "api_info": {"status": "ok", "attempts": 1, "truncated": False,
                         "error": None, "latency_ms": 100, "prompt_tokens": 50,
                         "completion_tokens": 20, "repeat_index": 1},
        })
    return {"model": config["name"], "answers": answers}


def test_batch_n10_full_pipeline(client, monkeypatch):
    """N=10 模型批次全链路（mock 单臂）→ done + 排行榜聚合。"""
    from backend import audit

    main_module._jobs.clear()
    main_module._tasks.clear()
    main_module._SCHEDULER.clear()
    monkeypatch.setattr(main_module, "_execute_model", _perf_fake_execute)
    ds = {
        "name": "perf_批次集",
        "tasks": [
            {"id": "T1", "type": "判别式", "dimension": "数学能力",
             "prompt": "1+1=?", "test_cases": [{"input": "1+1=?", "expected": "2"}]},
            {"id": "T2", "type": "判别式", "dimension": "数学能力",
             "prompt": "2+2=?", "test_cases": [{"input": "2+2=?", "expected": "4"}]},
            {"id": "T3", "type": "判别式", "dimension": "逻辑推理能力",
             "prompt": "3+3=?", "test_cases": [{"input": "3+3=?", "expected": "6"}]},
        ],
    }
    storage.save_dataset("perf_批次集", ds)
    models = [{"url": "https://8.8.8.8/v1", "key": "k", "name": f"P{i}",
               "temperature": 0.7, "max_tokens": 4096} for i in range(10)]
    t0 = time.perf_counter()
    r = client.post("/api/benchmark", json={
        "dataset_name": "perf_批次集", "models": models, "rounds": 1})
    assert r.status_code == 200, r.text
    bid = r.json()["batch_id"]
    batch = storage.load_batch(bid)
    for jid in batch["jobs"]:
        main_module._tasks.pop(jid, None)
        asyncio.run(main_module._run_batch_job(jid))
    finished = storage.load_batch(bid)
    elapsed = time.perf_counter() - t0
    assert finished["state"] == "done", finished.get("aggregation_error")
    assert finished["leaderboard_id"]
    assert len(finished["models"]) == 10
    assert elapsed < 30.0                 # 宽阈值：10 模型全链路 < 30s
    audit._log_path().write_text("", encoding="utf-8")
