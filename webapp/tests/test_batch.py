# -*- coding: utf-8 -*-
"""迭代七：benchmark 批次 API 集成测试。

覆盖不变量：
- 创建校验：数据集 404、模型配置不存在/重复/未补录 Key 400、评审模型 SSRF、
  预算 hard（N 模型放大）400
- N=5 全链路（mock 单臂）→ done + 排行榜聚合（综合分/分维度/胜率矩阵/CI）
- 部分失败 → partial + failed_models N/A + 排行榜排除失败模型
- 详情/列表/排行榜端点；审计 benchmark_started/benchmark_done
- Key 不落盘（config 仅掩码字段）

使用同款模式：monkeypatch _execute_model / run_single_arm_judge + 假 create_task
（后台任务在 asyncio.run 收尾被取消），再直跑 _run_batch_job 完成管线。
"""
from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend import audit
from backend import main as main_module
from backend import storage
from backend.schemas import BenchmarkRequest

PUBLIC_URL = "https://8.8.8.8/v1"

MODEL_NAMES = ["模型1", "模型2", "模型3", "模型4", "模型5"]


def _dataset() -> dict:
    return {
        "name": "批次集A",
        "tasks": [
            {"id": "T1", "type": "判别式", "dimension": "数学能力",
             "prompt": "1+1=?", "test_cases": [{"input": "1+1=?", "expected": "2"}]},
            {"id": "T2", "type": "判别式", "dimension": "数学能力",
             "prompt": "2+2=?", "test_cases": [{"input": "2+2=?", "expected": "4"}]},
            {"id": "T3", "type": "生成式", "dimension": "语言能力",
             "prompt": "写一句话", "expected": "参考答案",
             "rubric_note": "满分10分"},
        ],
    }


async def _fake_execute(model_label, config, tasks, stability_repeat,
                        progress_cb=None, embedding_cfg=None, skip_ids=None,
                        persist_cb=None, concurrency=1):
    if config.get("url", "").startswith("mock://fail") or config.get("name") == "失败模型":
        raise RuntimeError("模型调用失败（测试）")
    answers = []
    for t in tasks:
        if skip_ids and t["id"] in skip_ids:
            continue
        cases = t.get("test_cases") or []
        exp = cases[0].get("expected", "") if cases else ""
        answers.append({
            "id": t["id"],
            "raw_answer": f"答案是 {exp}" if t.get("type") == "判别式" else "生成文本",
            "api_info": {"status": "ok", "attempts": 1, "truncated": False,
                         "error": None, "latency_ms": 100, "prompt_tokens": 50,
                         "completion_tokens": 20, "repeat_index": 1},
        })
        if progress_cb:
            await progress_cb(model_label, len(answers), len(tasks))
    return {"model": config["name"], "api": {"name": config["name"]},
            "answers": answers}


async def _fake_single_arm(task_set, answers, judge_config,
                           progress_cb=None, max_retries=1):
    scores = [{"id": t["id"], "dimension": t.get("dimension", ""),
               "score": 8.0, "basis": "fake", "_invalid": False}
              for t in task_set["tasks"]]
    return {"meta": {"total": len(scores), "valid": len(scores), "invalid": 0},
            "scores": scores, "totals": {}, "health": {"healthy": True}}


def _call(fn, *a, **k):
    return asyncio.run(fn(*a, **k))


class _DummyTask:
    def __init__(self, coro):
        self.coro = coro

    def add_done_callback(self, fn):
        pass

    def done(self):
        return True

    def cancel(self):
        pass


def _drain_batch(batch_id: str) -> None:
    """直跑批次内全部 job 的后台协程（测试专用，幂等）。"""
    batch = storage.load_batch(batch_id)
    for jid in batch["jobs"]:
        j = main_module._jobs.get(jid)
        if j is None or j.get("state") in ("completed", "error", "cancelled"):
            continue
        main_module._tasks.pop(jid, None)
        asyncio.run(main_module._run_batch_job(jid))


@pytest.fixture
def client():
    return TestClient(main_module.app)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    main_module._jobs.clear()
    main_module._tasks.clear()
    main_module._SCHEDULER.clear()
    monkeypatch.setattr(main_module, "_execute_model", _fake_execute)
    monkeypatch.setattr(main_module, "run_single_arm_judge", _fake_single_arm)
    monkeypatch.setattr(main_module.asyncio, "create_task", _DummyTask)
    audit._log_path().write_text("", encoding="utf-8")
    storage.save_dataset("批次集A", _dataset())
    yield
    main_module._jobs.clear()
    main_module._tasks.clear()
    main_module._SCHEDULER.clear()


def _payload(n=2, **overrides) -> dict:
    base = {"dataset_name": "批次集A", "models": [
        {"url": PUBLIC_URL, "key": "k", "name": f"模型{i + 1}",
         "temperature": 0.7, "max_tokens": 4096} for i in range(n)]}
    base.update(overrides)
    return base


def _start(**kw):
    return _call(main_module.create_benchmark, BenchmarkRequest(**kw))


# ---- 创建校验 ----

def test_dataset_missing_404(client):
    with pytest.raises(HTTPException) as ei:
        _start(dataset_name="不存在集", models=_payload()["models"])
    assert ei.value.status_code == 404


def test_judge_ssrf_400(client):
    payload = _payload(n=2)
    payload["review"] = {"mode": "pure_agent",
                         "judge": {"url": "http://127.0.0.1:9/v1", "key": "k",
                                   "name": "j", "temperature": 0.0,
                                   "max_tokens": 100}}
    with pytest.raises(HTTPException) as ei:
        _start(**payload)
    assert ei.value.status_code == 400


def test_budget_hard_400(client):
    payload = _payload(n=5)
    payload["budget"] = {"max_tokens": 100, "mode": "hard"}
    with pytest.raises(HTTPException) as ei:
        _start(**payload)
    assert ei.value.status_code == 400
    assert "预算超限" in str(ei.value.detail)


# ---- 全链路 ----

def test_full_pipeline_n5(client):
    payload = _payload(n=5, rounds=2,
                       review={"mode": "pure_agent",
                               "judge": {"url": PUBLIC_URL, "key": "jk",
                                         "name": "评审", "temperature": 0.0,
                                         "max_tokens": 100}})
    resp = _start(**payload)
    assert resp["ok"] is True
    batch_id = resp["batch_id"]
    assert len(resp["jobs"]) == 5
    assert resp["models"] == MODEL_NAMES
    events = audit.read_events()
    assert any(e["event"] == "benchmark_started" and e["target"] == batch_id
               for e in events)
    # Key 不落盘
    for jid in resp["jobs"]:
        cfg = storage.get_job_files(jid)["config.json"]
        assert "model_a_key_masked" in cfg and "k" not in str(cfg["model_a"].get("key_masked", ""))
    _drain_batch(batch_id)
    batch = storage.load_batch(batch_id)
    assert batch["state"] == "done"
    assert batch["leaderboard_id"]
    assert batch["failed_models"] == []
    events = audit.read_events()
    assert any(e["event"] == "benchmark_done" and e["target"] == batch_id
               for e in events)
    # 排行榜内容（单臂格式聚合）
    lb = storage.load_leaderboard(batch["leaderboard_id"])
    assert set(lb["models"]) == set(MODEL_NAMES)
    # 判别式满分（10×2 题×2 轮均值=10），生成式 8.0
    assert lb["composite"]["模型1"]["score"] == 28.0
    assert lb["ranks"]["模型1"] == 1
    assert lb["win_matrix"]["模型1"]["模型2"]["total"] == 3
    ci = lb["ci"]["模型1"]["模型2"]
    assert ci["n"] == 3 and ci["significant"] is False
    # 详情与排行榜端点
    detail = client.get(f"/api/benchmark/{batch_id}").json()
    assert detail["state"] == "done"
    assert detail["progress"] == "5/5"
    lr = client.get(f"/api/benchmark/{batch_id}/leaderboard")
    assert lr.status_code == 200
    assert set(lr.json()["models"]) == set(MODEL_NAMES)


def test_partial_failure_marks_na(client):
    """5 模型中 1 个调用失败 → batch partial + 失败模型 N/A + 排行榜排除。"""
    payload = _payload(n=5)
    payload["models"][-1]["name"] = "失败模型"   # _fake_execute 按名称触发失败
    resp = _start(**payload)
    batch_id = resp["batch_id"]
    _drain_batch(batch_id)
    batch = storage.load_batch(batch_id)
    assert batch["state"] == "partial"
    assert len(batch["failed_models"]) == 1
    assert batch["aggregation_error"] is None  # 4 个完成模型可聚合
    lb = storage.load_leaderboard(batch["leaderboard_id"])
    assert set(lb["models"]) == set(MODEL_NAMES[:4])
    detail = client.get(f"/api/benchmark/{batch_id}").json()
    assert detail["progress"] == "5/5"
    assert len(detail["failed_models"]) == 1


def test_leaderboard_404_before_completion(client):
    payload = _payload(n=2)
    resp = _start(**payload)
    r = client.get(f"/api/benchmark/{resp['batch_id']}/leaderboard")
    assert r.status_code == 404


# ---- 迭代九：内联模型（首页 N 模型入口） ----

def _inline_payload(n=2, **overrides) -> dict:
    base = {"dataset_name": "批次集A", "models": [
        {"url": PUBLIC_URL, "key": "sk-inline-secret", "name": f"内联模型{i + 1}",
         "temperature": 0.7, "max_tokens": 4096} for i in range(n)]}
    base.update(overrides)
    return base


def test_inline_models_full_pipeline(client):
    resp = _start(**_inline_payload(n=2))
    assert resp["ok"] is True
    batch_id = resp["batch_id"]
    assert len(resp["jobs"]) == 2
    assert resp["models"] == ["内联模型1", "内联模型2"]
    # Key 不落盘：内联 Key 不得出现在 config.json
    for jid in resp["jobs"]:
        raw = json.dumps(storage.get_job_files(jid)["config.json"], ensure_ascii=False)
        assert "sk-inline-secret" not in raw
    _drain_batch(batch_id)
    batch = storage.load_batch(batch_id)
    assert batch["state"] == "done"
    lb = storage.load_leaderboard(batch["leaderboard_id"])
    assert set(lb["models"]) == {"内联模型1", "内联模型2"}


def test_inline_builtin_mode_creates_batch(client):
    """首页「内置题库」模式：dataset_name 缺省 → dims/seed/num_questions 生成任务集。"""
    resp = _start(**_inline_payload(
        n=2, dataset_name=None, dims=["数学能力"], num_questions=3, seed=42))
    assert resp["ok"] is True
    _drain_batch(resp["batch_id"])
    batch = storage.load_batch(resp["batch_id"])
    assert batch["state"] == "done"


def test_inline_ssrf_400(client):
    with pytest.raises(HTTPException) as ei:
        _start(**_inline_payload(
            n=2, models=[
                {"url": "http://127.0.0.1:8000/v1", "key": "k", "name": "内联模型1",
                 "temperature": 0.7, "max_tokens": 4096},
                {"url": PUBLIC_URL, "key": "k", "name": "内联模型2",
                 "temperature": 0.7, "max_tokens": 4096},
            ]))
    assert ei.value.status_code == 400
    assert "URL 校验失败" in str(ei.value.detail)


def test_batch_validation_and_list(client):
    resp = _start(**_payload(n=2))
    assert resp["ok"] is True
    items = client.get("/api/benchmark").json()["batches"]
    assert any(x["batch_id"] == resp["batch_id"] for x in items)
    r = client.get("/api/benchmark/batch_bad")
    assert r.status_code == 400
    r = client.get("/api/benchmark/batch_00000000_000000_000000")
    assert r.status_code == 404


def test_inline_custom_dataset_sampling(client):
    """首页「自定义评测集」模式：num_questions 随机抽 N 题（保留原 id 与顺序）。"""
    resp = _start(**_inline_payload(n=2, num_questions=2, seed=42))
    assert resp["ok"] is True
    jid = resp["jobs"][0]
    tasks = main_module._jobs[jid]["task_set"]["tasks"]
    ids = [t["id"] for t in tasks]
    assert len(ids) == 2
    assert set(ids) <= {"T1", "T2", "T3"}
    assert ids == sorted(ids)
    _drain_batch(resp["batch_id"])
    assert storage.load_batch(resp["batch_id"])["state"] == "done"


def test_inline_custom_dataset_sampling_over_total(client):
    """num_questions 超过题数 → 全量不抽样不重排。"""
    resp = _start(**_inline_payload(n=2, num_questions=5))
    assert resp["ok"] is True
    jid = resp["jobs"][0]
    ids = [t["id"] for t in main_module._jobs[jid]["task_set"]["tasks"]]
    assert ids == ["T1", "T2", "T3"]
    _drain_batch(resp["batch_id"])
    assert storage.load_batch(resp["batch_id"])["state"] == "done"


def test_num_questions_schema_bounds():
    """num_questions 校验：0/201 → 422 层拒绝；200 合法。"""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        BenchmarkRequest(**_inline_payload(n=2, num_questions=0))
    with pytest.raises(ValidationError):
        BenchmarkRequest(**_inline_payload(n=2, num_questions=201))
    b = BenchmarkRequest(**_inline_payload(n=2, num_questions=200))
    assert b.num_questions == 200


# ---- 迭代十一：已终态批次删除（含排行榜与 job 历史联动清理） ----

def test_delete_terminal_batch_cleans_all(client):
    resp = _start(**_payload(n=2))
    batch_id = resp["batch_id"]
    job_ids = resp["jobs"]
    _drain_batch(batch_id)
    batch = storage.load_batch(batch_id)
    assert batch["state"] == "done"
    lb_id = batch["leaderboard_id"]
    assert lb_id
    assert storage.load_leaderboard(lb_id)
    for jid in job_ids:
        assert storage.get_job_files(jid)

    r = client.delete(f"/api/benchmark/{batch_id}")
    assert r.status_code == 200, r.text
    assert r.json()["deleted"] is True
    # 批次 / 排行榜 / job 历史目录全部清除
    assert storage.load_batch(batch_id) is None
    assert storage.load_leaderboard(lb_id) is None
    for jid in job_ids:
        assert storage.get_job_files(jid) is None
    assert all(b["batch_id"] != batch_id
               for b in client.get("/api/benchmark").json()["batches"])
    events = audit.read_events()
    assert any(e["event"] == "benchmark_deleted" and e["target"] == batch_id
               for e in events)


def test_delete_running_batch_409(client):
    resp = _start(**_payload(n=2))
    r = client.delete(f"/api/benchmark/{resp['batch_id']}")
    assert r.status_code == 409
    assert storage.load_batch(resp["batch_id"]) is not None


def test_delete_cancelled_batch_ok(client):
    resp = _start(**_payload(n=2))
    batch_id = resp["batch_id"]
    # 直接置为终态 cancelled（真实取消依赖调度器执行态，此处验证删除语义本身）
    batch = storage.load_batch(batch_id)
    batch["state"] = "cancelled"
    storage.save_batch(batch_id, batch)
    r = client.delete(f"/api/benchmark/{batch_id}")
    assert r.status_code == 200
    assert storage.load_batch(batch_id) is None


def test_delete_missing_and_invalid(client):
    assert client.delete("/api/benchmark/batch_20240101_000000_abcdef").status_code == 404
    assert client.delete("/api/benchmark/../evil").status_code == 404
    # 幂等：重复删除同一已删除批次 → 404
    resp = _start(**_payload(n=2))
    batch_id = resp["batch_id"]
    _drain_batch(batch_id)
    assert client.delete(f"/api/benchmark/{batch_id}").status_code == 200
    assert client.delete(f"/api/benchmark/{batch_id}").status_code == 404


# ---- 迭代十一：扰动评测集成到批次（自动对每模型跑扰动，结果随排行榜展示） ----

def test_perturb_invalid_mode_400(client):
    r = client.post("/api/benchmark", json={**_payload(n=2), "perturb_modes": ["非法模式"]})
    assert r.status_code == 400
    assert "非法扰动模式" in r.json()["detail"]


def test_batch_with_perturb_runs_and_exposes_results(client):
    """创建带扰动批次 → 每模型生成 prb 任务 → 跑完主 job 与扰动协程 →
    batch done → 排行榜端点附 perturb（curves/bias）→ 删除联动清理扰动记录。"""
    # 专用数据集：含地域/性别/改写槽位词，保证属性扰动与改写均能命中
    storage.save_dataset("扰动批次集", {
        "name": "扰动批次集",
        "tasks": [
            {"id": "T1", "type": "判别式", "dimension": "知识能力",
             "prompt": "北京的王先生和妻子因为工作调往上海，请问对家庭有何影响？",
             "test_cases": [{"input": "q", "expected": "家庭影响"}]},
            {"id": "T2", "type": "判别式", "dimension": "知识能力",
             "prompt": "北方人春节吃饺子，南方人端午节吃粽子，对吗？",
             "test_cases": [{"input": "q", "expected": "对"}]},
        ],
    })
    resp = _start(**_payload(n=2, dataset_name="扰动批次集",
                             perturb_modes=["改写", "属性扰动-地域"]))
    assert resp["ok"] is True
    batch_id = resp["batch_id"]
    batch = storage.load_batch(batch_id)
    assert batch["perturb"]["enabled"] is True
    assert batch["perturb"]["modes"] == ["改写", "属性扰动-地域"]
    prb_tasks = batch["perturb"]["tasks"]
    assert set(prb_tasks) == {"模型1", "模型2"}
    # 扰动任务存在且 running
    for prb_id in prb_tasks.values():
        prb = storage.load_perturb(prb_id)
        assert prb is not None and prb["state"] == "running"
        assert prb["batch_id"] == batch_id

    # 直跑主 job
    _drain_batch(batch_id)
    # 迭代十一：扰动与主流程解耦——job 全终态即聚合完成（不再等待扰动）
    batch = storage.load_batch(batch_id)
    assert batch["state"] == "done"
    assert batch["leaderboard_id"]
    # 扰动协程随后跑完（monkeypatch 的假 execute 同样适用于扰动管线）
    for prb_id in prb_tasks.values():
        prb_data = storage.load_perturb(prb_id)
        asyncio.run(main_module._run_perturb_core(
            prb_id, prb_data, batch["dataset"],
            {"name": "模型X", "url": PUBLIC_URL, "key": "k",
             "temperature": 0.7, "max_tokens": 4096},
            ["改写", "属性扰动-地域"], None, None, None, "cot"))

    # 排行榜端点附带扰动结果（动态读取）
    r = client.get(f"/api/benchmark/{batch_id}/leaderboard")
    assert r.status_code == 200
    lb = r.json()
    assert len(lb["perturb"]) == 2
    for entry in lb["perturb"]:
        assert entry["state"] == "ready"
        assert set(entry["modes"]) == {"改写", "属性扰动-地域"}
        assert "改写" in (entry["curves"].get("curves") or {})
        assert entry["bias"]["pairs"]  # 属性扰动有对照数据

    # 删除批次联动清理扰动记录
    assert client.delete(f"/api/benchmark/{batch_id}").status_code == 200
    for prb_id in prb_tasks.values():
        assert storage.load_perturb(prb_id) is None


def test_batch_settle_selfheals_stale_after_restart(client):
    """迭代十一：服务重启后遗留 running 批次自愈——job 全终态即完成聚合
    （扰动与主流程解耦：扰动协程丢失不再阻塞批次完成）。"""
    resp = _start(**_payload(n=2, perturb_modes=["改写"]))
    batch_id = resp["batch_id"]
    batch = storage.load_batch(batch_id)
    prb_tasks = batch["perturb"]["tasks"]
    # 主 job 直跑完成（job 完成回调即收尾，扰动不阻塞）
    _drain_batch(batch_id)
    batch = storage.load_batch(batch_id)
    assert batch["state"] == "done"
    assert batch["leaderboard_id"]
    # 扰动协程未跑（_tasks 有记录但未调度）：查询列表仍正常，扰动记录保留 running
    r = client.get("/api/benchmark")
    assert r.status_code == 200
    entry = next(b for b in r.json()["batches"] if b["batch_id"] == batch_id)
    assert entry["state"] == "done"
    assert entry["leaderboard_id"]
    # 排行榜端点：扰动 running 动态展示（含进度）
    r = client.get(f"/api/benchmark/{batch_id}/leaderboard")
    assert r.status_code == 200
    for e in r.json()["perturb"]:
        assert e["state"] in ("running", "partial")
        assert "progress" in e


def test_perturb_judge_progress_persisted(client):
    """迭代十一：评审阶段进度落盘（'评审 X/Y'）——界面持续可见。"""
    storage.save_dataset("扰动集P", {
        "name": "扰动集P",
        "tasks": [
            {"id": "G1", "type": "生成式", "dimension": "语言能力",
             "prompt": "写一句话", "expected": "参考答案",
             "rubric_note": "满分10分"},
        ],
    })
    resp = _start(**_payload(n=2, dataset_name="扰动集P", perturb_modes=["改写"]))
    batch_id = resp["batch_id"]
    batch = storage.load_batch(batch_id)
    prb_id = list(batch["perturb"]["tasks"].values())[0]
    prb_data = storage.load_perturb(prb_id)

    async def fake_judge_with_progress(task_set, answers, judge_config,
                                       progress_cb=None, max_retries=1):
        assert progress_cb is not None
        await progress_cb(1, 2)
        # 评审进度已落盘（"评审 1/2"），界面可实时感知
        assert storage.load_perturb(prb_id)["progress"] == "评审 1/2"
        await progress_cb(2, 2)
        scores = [{"id": t["id"], "dimension": t.get("dimension", ""),
                   "score": 8.0, "basis": "fake", "_invalid": False}
                  for t in task_set["tasks"]]
        return {"meta": {"total": len(scores), "valid": len(scores), "invalid": 0},
                "scores": scores, "totals": {}, "health": {"healthy": True}}

    import backend.main as mm
    orig = mm.run_single_arm_judge
    mm.run_single_arm_judge = fake_judge_with_progress
    try:
        asyncio.run(mm._run_perturb_core(
            prb_id, prb_data, "扰动集P",
            {"name": "模型X", "url": PUBLIC_URL, "key": "k",
             "temperature": 0.7, "max_tokens": 4096},
            ["改写"], None, None,
            {"url": PUBLIC_URL, "key": "k", "name": "judge"}, "cot"))
    finally:
        mm.run_single_arm_judge = orig
    prb = storage.load_perturb(prb_id)
    assert prb["state"] == "ready"
    assert prb["progress"] == "done"


def test_batch_list_summary_structure_and_detail_full(client):
    resp = _start(**_payload(n=2, perturb_modes=["改写"]))
    batch_id = resp["batch_id"]
    batch = storage.load_batch(batch_id)
    prb_tasks = batch["perturb"]["tasks"]
    # 列表：running 批次返回摘要结构（perturb.tasks 是数组，含进度；无 jobs 字段）
    r = client.get("/api/benchmark")
    entry = next(b for b in r.json()["batches"] if b["batch_id"] == batch_id)
    assert entry["state"] == "running"
    assert "jobs" not in entry
    pt = entry["perturb"]
    assert pt["enabled"] is True
    assert isinstance(pt["tasks"], list)
    assert len(pt["tasks"]) == 2
    assert all(t["state"] == "running" and "progress" in t for t in pt["tasks"])
    # 详情：完整记录（jobs 数组 + perturb.tasks 字典映射）
    r2 = client.get(f"/api/benchmark/{batch_id}")
    d2 = r2.json()
    assert d2["state"] == "running"
    assert isinstance(d2["jobs"], list) and len(d2["jobs"]) == 2
    assert isinstance(d2["perturb"]["tasks"], dict)
    assert set(d2["perturb"]["tasks"].values()) == set(prb_tasks.values())


def test_perturb_list_settles_stale_running(client):
    """迭代十一：/api/perturb 列表对遗留 running（不在 _tasks）沉降 partial。"""
    from backend.storage import PERTURB_DIR
    storage.save_perturb("prb_20260101_000000_abc123", {
        "perturb_id": "prb_20260101_000000_abc123",
        "state": "running", "model_name": "M", "dataset": "批次集A",
        "modes": ["改写"], "progress": "0/0",
    })
    assert main_module._tasks.get("prb_20260101_000000_abc123") is None
    r = client.get("/api/perturb")
    assert r.status_code == 200
    entry = next(x for x in r.json()["perturbs"]
                 if x["perturb_id"] == "prb_20260101_000000_abc123")
    assert entry["state"] == "partial"
    assert "协程中断" in (entry.get("error") or "")


def test_batch_report_contains_total_tokens(client):
    """迭代十一：batch 任务 report.kpi.total_tokens 落盘（KPI Token 趋势修复）。"""
    resp = _start(**_payload(n=2))
    batch_id = resp["batch_id"]
    _drain_batch(batch_id)
    batch = storage.load_batch(batch_id)
    assert batch["state"] == "done"
    # _fake_execute 每题 prompt 50 + completion 20 → 3 题 × 70 = 210
    for jid in batch["jobs"]:
        import json as _json
        raw = _json.loads((storage.BASE_DIR / jid / "report.json").read_text(encoding="utf-8"))
        kpi = raw["report"]["kpi"]
        assert kpi["total_tokens"] == {"x": 210, "y": 0}
        assert kpi["duration_sec"] is not None
