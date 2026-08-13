# -*- coding: utf-8 -*-
"""Agent 评审（pure_agent）接入主流程的集成测试（迭代二 步骤9）。

覆盖不变量：
- 作答完成后自动进入 judging → completed，report 标记 judge_mode=pure_agent
- agent 评审与人工评审共用同一 reveal（身份对齐），round_verdicts 可聚合
- 评审模型整体异常：fail_open=True 降级 reviewing（judge_health 事件），
  fail_open=False 任务置 error
- 全部 verdict invalid 视为评审失败，走同一降级/报错分支
- budget hard 超限 start 即 400；warn 超限运行期幂等推送 budget_warning
- pure_agent 缺 judge 配置 / 评审 URL SSRF 不合规 → 400
- embedding_cfg 透传执行器；judge key 落盘打码

说明：与 test_task_lifecycle 同款——单一 asyncio.run 循环内直调端点函数，
monkeypatch execute_all / run_judge（评审全程无真实网络）。
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
from backend.engine.tasks import build_task_set
from backend.schemas import ReviewSubmission, StartRequest

PUBLIC_URL = "https://8.8.8.8/v1"


def _payload(**overrides) -> dict:
    base = {
        "model_a": {"url": PUBLIC_URL, "key": "k", "name": "A", "temperature": 0.7, "max_tokens": 100},
        "model_b": {"url": PUBLIC_URL, "key": "k", "name": "B", "temperature": 0.7, "max_tokens": 100},
    }
    base.update(overrides)
    return base


def _agent_review(fail_open: bool = True, judge_url: str = PUBLIC_URL) -> dict:
    return {
        "mode": "pure_agent",
        "fail_open": fail_open,
        "judge": {"url": judge_url, "key": "jk", "name": "J", "temperature": 0.0, "max_tokens": 256},
    }


@pytest.fixture
def client():
    return TestClient(main_module.app)


@pytest.fixture(autouse=True)
def _clean_state():
    main_module._jobs.clear()
    main_module._tasks.clear()
    yield
    main_module._jobs.clear()
    main_module._tasks.clear()


def _fake_answers(task_set: dict, model: str) -> dict:
    return {
        "model": model,
        "answers": [
            {
                "id": t["id"],
                "raw_answer": f"{model}: {t['id']} 的作答",
                "api_info": {
                    "provider": "external", "status": "ok", "latency_ms": 12,
                    "prompt_tokens": 10, "completion_tokens": 8, "repeat_index": 1,
                },
            }
            for t in task_set["tasks"]
        ],
    }


async def _fake_execute_all(task_set, config_a=None, config_b=None, progress_cb=None, **kwargs):
    captured["execute_kwargs"] = kwargs
    answers_a = _fake_answers(task_set, config_a["name"])
    answers_b = _fake_answers(task_set, config_b["name"])
    total = task_set["meta"]["total"]
    if progress_cb:
        for i in range(1, total + 1):
            await progress_cb("a", i, total)
    return answers_a, answers_b


async def _fake_run_judge(task_set, answers_x, answers_y, judge_config,
                          revealed=None, progress_cb=None):
    captured["judge_cfg"] = judge_config
    captured["revealed"] = revealed
    tasks = task_set["tasks"]
    scores, dim_totals = [], {}
    for t in tasks:
        winner = "x" if t["id"] not in ("T3B", "T3C") else "y"
        ax, ay = (5.0, 4.0) if winner == "x" else (4.0, 5.0)
        scores.append({
            "id": t["id"], "dimension": t.get("dimension", ""), "winner": winner,
            "answer_x": ax, "answer_y": ay, "basis": "fake", "arbiter_note": None,
            "_invalid": False,
        })
        if not t.get("excluded_from_total"):
            dim_totals.setdefault(t.get("dimension", ""), {"x": 0.0, "y": 0.0})
            dim_totals[t.get("dimension", "")]["x"] += ax
            dim_totals[t.get("dimension", "")]["y"] += ay
    total_x = sum(d["x"] for d in dim_totals.values())
    total_y = sum(d["y"] for d in dim_totals.values())
    if progress_cb:
        for i in range(1, len(tasks) + 1):
            await progress_cb(i, len(tasks))
    return {
        "meta": {"total": len(tasks), "valid": len(tasks), "invalid": 0,
                 "tie_arbitrated": 0, "excluded_ids": [], "excluded_dimensions": []},
        "scores": scores,
        "per_dimension": dim_totals,
        "totals": {"answer_x": total_x, "answer_y": total_y},
        "revealed": {"answer_x": "answers-a.json", "answer_y": "answers-b.json",
                     "answer_x_file": "a", "answer_y_file": "b"},
        "conclusion": "fake conclusion",
        "winner_model": answers_x["model"] if total_x > total_y else (
            answers_y["model"] if total_y > total_x else "tie"
        ),
    }


captured: dict = {}


def _drain_events(job_id: str) -> list[dict]:
    q = main_module._jobs[job_id]["sse_queue"]
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    return events


async def _wait_task_done(job_id: str, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while job_id in main_module._tasks and loop.time() < deadline:
        await asyncio.sleep(0.02)


async def _run_agent_job(**overrides) -> str:
    main_module._jobs.clear()
    main_module._tasks.clear()
    captured.clear()
    review = _agent_review()
    if "fail_open" in overrides:
        review["fail_open"] = overrides.pop("fail_open")
    payload = _payload(review=review)
    payload.update(overrides)
    resp = await main_module.start_eval(StartRequest(**payload))
    job_id = resp.job_id
    await _wait_task_done(job_id)
    return job_id


# ---- 全链路：executing → judging → completed ----

def test_agent_judge_full_chain_completes(monkeypatch):
    monkeypatch.setattr(main_module, "execute_all", _fake_execute_all)
    monkeypatch.setattr(main_module, "run_judge", _fake_run_judge)

    async def _scenario():
        job_id = await _run_agent_job()
        j = main_module._jobs[job_id]

        # 状态机走完并停在 completed；报告可用
        assert j["state"] == "completed"
        files = storage.get_job_files(job_id)
        for k in ("config.json", "tasks.json", "answers-a.json", "answers-b.json",
                  "verdict.json", "round-verdicts.json", "report.json"):
            assert k in files, f"缺少 {k}"
        assert storage.load_review(job_id)["mode"] == "pure_agent"

        # SSE 事件序列：executing → judging → completed
        events = _drain_events(job_id)
        states = [e["state"] for e in events if "state" in e]
        assert "judging" in states
        assert states[-1] == "completed"
        judge_events = [e for e in events if e.get("state") == "judging"]
        assert judge_events and all("progress" in e for e in judge_events)

        # 报告：judge_mode=pure_agent、prompt_strategy=cot、四段齐备
        report = storage.get_job_files(job_id)["report.json"]["report"]
        assert report["judge_mode"] == "pure_agent"
        assert report["prompt_strategy"] == "cot"
        for k in ("summary", "charts", "analysis", "metrics", "kpi", "significance", "warnings"):
            assert k in report

        # round_verdicts：revealed 身份与 reveal 文件一致（answer_x=模型名，_file=标签）
        verdicts = storage.get_job_files(job_id)["round-verdicts.json"]
        assert len(verdicts) == 1
        j_reveal = j["reveal"]["rounds"][0]
        assert verdicts[0]["revealed"]["answer_x_file"] == j_reveal["answer_x"]
        assert verdicts[0]["revealed"]["answer_y_file"] == j_reveal["answer_y"]

        # 审计：agent 评审提交
        events_log = {e["event"]: e for e in audit.read_events() if e.get("job_id") == job_id}
        assert events_log["review_submitted"]["actor"] == "agent"

    asyncio.run(_scenario())


def test_agent_judge_uses_shared_reveal_and_injects(monkeypatch):
    monkeypatch.setattr(main_module, "execute_all", _fake_execute_all)
    monkeypatch.setattr(main_module, "run_judge", _fake_run_judge)

    async def _scenario():
        job_id = await _run_agent_job()
        j = main_module._jobs[job_id]
        round_reveal = j["reveal"]["rounds"][0]

        # 迭代三（H1）：run_judge 收到逐题独立随机交换 reveal（仅 agent）
        rv = captured["revealed"]
        assert set(rv) == {"rounds", "per_task"}
        per_task = {pt["task_id"]: pt for pt in rv["per_task"]}
        # rounds 字段 = per_task 首项（轮级兜底），与逐题映射一致
        first = rv["per_task"][0]
        assert rv["rounds"][0]["answer_x"] == first["answer_x"]
        assert rv["rounds"][0]["answer_y"] == first["answer_y"]
        task_ids = {t["id"] for t in j["task_set"]["tasks"]}
        assert set(per_task) == task_ids
        assert all(pt["answer_x"] in ("a", "b") and pt["answer_y"] in ("a", "b")
                   and pt["answer_x"] != pt["answer_y"] for pt in per_task.values())
        # 聚合结果未被逐题交换破坏（单轮不归一化，totals 即 answer_x/answer_y）
        verdict = storage.get_job_files(job_id)["verdict.json"]
        assert verdict["totals"] and len(verdict["totals"]) == 2

        # round_verdicts 中 x 模型名 = 轮级 reveal 标签对应模型；文件标签同 reveal
        expected_x = ("A", "B") if round_reveal["answer_x"] == "a" else ("B", "A")
        verdicts = storage.get_job_files(job_id)["round-verdicts.json"]
        assert verdicts[0]["revealed"]["answer_x"] == expected_x[0]
        assert verdicts[0]["revealed"]["answer_y"] == expected_x[1]
        assert verdicts[0]["revealed"]["answer_x_file"] == round_reveal["answer_x"]
        assert verdicts[0]["meta"]["repeat_n"] == 1

    asyncio.run(_scenario())


def test_agent_judge_judge_cfg_and_masked_key(monkeypatch):
    monkeypatch.setattr(main_module, "execute_all", _fake_execute_all)
    monkeypatch.setattr(main_module, "run_judge", _fake_run_judge)

    async def _scenario():
        job_id = await _run_agent_job()
        assert captured["judge_cfg"]["name"] == "J"
        assert captured["judge_cfg"]["url"] == PUBLIC_URL

        # config 落盘：judge key 打码，被测模型 key 同样打码；扁平化字段自描述
        cfg = storage.get_job_files(job_id)["config.json"]
        assert cfg["judge"]["key_masked"] == "***"
        assert cfg["judge"]["name"] == "J"
        assert cfg["review_mode"] == "pure_agent"
        assert cfg["prompt_strategy"] == "cot"
        assert cfg["model_a"]["key_masked"] == "***"
        # 内存 config 保留明文（agent 评审运行时仍可调用）
        assert main_module._jobs[job_id]["config"]["review"]["judge"]["key"] == "jk"

    asyncio.run(_scenario())


def test_agent_judge_repeat_n_multiple_rounds(monkeypatch):
    monkeypatch.setattr(main_module, "execute_all", _fake_execute_all)
    monkeypatch.setattr(main_module, "run_judge", _fake_run_judge)

    async def _scenario():
        job_id = await _run_agent_job(repeat_n=3)
        j = main_module._jobs[job_id]
        assert j["state"] == "completed"
        verdicts = storage.get_job_files(job_id)["round-verdicts.json"]
        assert len(verdicts) == 3
        assert [v["meta"]["repeat_n"] for v in verdicts] == [1, 1, 1]
        # 每轮注入的 reveal 与逐轮身份一致（文件标签）
        for r_idx, v in enumerate(verdicts):
            rr = j["reveal"]["rounds"][r_idx]
            assert v["revealed"]["answer_x_file"] == rr["answer_x"]
        verdict = storage.get_job_files(job_id)["verdict.json"]
        assert verdict["meta"]["repeat_n"] == 3

    asyncio.run(_scenario())


# ---- 评审失败：fail_open 降级 / 报错 ----

def test_agent_judge_fail_open_degrades_to_reviewing(monkeypatch):
    monkeypatch.setattr(main_module, "execute_all", _fake_execute_all)

    async def _boom_run_judge(*args, **kwargs):
        raise RuntimeError("评审上游 500")

    monkeypatch.setattr(main_module, "run_judge", _boom_run_judge)

    async def _scenario():
        job_id = await _run_agent_job(fail_open=True)
        j = main_module._jobs[job_id]
        assert j["state"] == "reviewing"

        events = _drain_events(job_id)
        assert any(e.get("state") == "reviewing" for e in events)
        health = [e for e in events if e.get("type") == "judge_health"]
        assert health and health[0]["status"] == "degraded"

        # 未产出任何评审产物，人工评审可继续
        files = storage.get_job_files(job_id)
        assert "verdict.json" not in files
        assert "report.json" not in files
        # 作答与 reveal 仍在，降级后人工提交闭环可走通
        task_set = j["task_set"]
        scores = [{"id": t["id"], "round": 1, "answer_x": 5, "answer_y": 5} for t in task_set["tasks"]]
        r = await main_module.eval_review_submit(job_id, ReviewSubmission(scores=scores))
        assert r.job_id == job_id
        assert main_module._jobs[job_id]["state"] == "completed"

    asyncio.run(_scenario())


def test_agent_judge_fail_open_false_marks_error(monkeypatch):
    monkeypatch.setattr(main_module, "execute_all", _fake_execute_all)

    async def _boom_run_judge(*args, **kwargs):
        raise RuntimeError("评审上游 500")

    monkeypatch.setattr(main_module, "run_judge", _boom_run_judge)

    async def _scenario():
        job_id = await _run_agent_job(fail_open=False)
        j = main_module._jobs[job_id]
        assert j["state"] == "error"
        assert j["error"].startswith("AI 评审失败")
        events = _drain_events(job_id)
        assert any(e.get("state") == "error" for e in events)
        # 审计无 review_submitted
        evs = [e for e in audit.read_events()
               if e.get("job_id") == job_id and e["event"] == "review_submitted"]
        assert not evs

    asyncio.run(_scenario())


def test_agent_judge_all_invalid_verdicts_degrades(monkeypatch):
    monkeypatch.setattr(main_module, "execute_all", _fake_execute_all)

    async def _invalid_run_judge(task_set, answers_x, answers_y, judge_config,
                                 revealed=None, progress_cb=None):
        return {
            "meta": {"total": len(task_set["tasks"]), "valid": 0, "invalid": len(task_set["tasks"]),
                     "tie_arbitrated": 0, "excluded_ids": [], "excluded_dimensions": []},
            "scores": [], "per_dimension": {},
            "totals": {"answer_x": 0, "answer_y": 0},
            "revealed": {}, "conclusion": "全 invalid", "winner_model": "tie",
        }

    monkeypatch.setattr(main_module, "run_judge", _invalid_run_judge)

    async def _scenario():
        job_id = await _run_agent_job(fail_open=True)
        assert main_module._jobs[job_id]["state"] == "reviewing"
        health = [e for e in _drain_events(job_id) if e.get("type") == "judge_health"]
        assert health and health[0]["status"] == "degraded"

    asyncio.run(_scenario())


# ---- 预算熔断 ----

def test_agent_judge_budget_hard_exceed_rejected(monkeypatch):
    monkeypatch.setattr(main_module, "execute_all", _fake_execute_all)
    monkeypatch.setattr(main_module, "run_judge", _fake_run_judge)

    async def _scenario():
        main_module._jobs.clear()
        main_module._tasks.clear()
        payload = _payload(review=_agent_review(), budget={"max_tokens": 1000, "mode": "hard"})
        with pytest.raises(HTTPException) as exc:
            await main_module.start_eval(StartRequest(**payload))
        assert exc.value.status_code == 400
        assert "预算超限" in exc.value.detail

    asyncio.run(_scenario())


def test_agent_judge_budget_warn_ok_and_event(monkeypatch):
    monkeypatch.setattr(main_module, "execute_all", _fake_execute_all)
    monkeypatch.setattr(main_module, "run_judge", _fake_run_judge)

    async def _scenario():
        job_id = await _run_agent_job(
            fail_open=True, budget={"max_tokens": 1000, "mode": "warn"},
        )
        j = main_module._jobs[job_id]
        assert j["state"] == "completed"
        assert j["budget_warned"] is True
        events = _drain_events(job_id)
        warns = [e for e in events if e.get("type") == "budget_warning"]
        assert len(warns) == 1
        assert warns[0]["estimated"] > warns[0]["limit"]

    asyncio.run(_scenario())


# ---- 参数校验 ----

def test_agent_judge_requires_judge_config():
    async def _scenario():
        payload = _payload(review={"mode": "pure_agent", "fail_open": False})
        with pytest.raises(HTTPException) as exc:
            await main_module.start_eval(StartRequest(**payload))
        assert exc.value.status_code == 400
        assert "judge" in exc.value.detail

    asyncio.run(_scenario())


def test_agent_judge_judge_url_ssrf_rejected():
    async def _scenario():
        payload = _payload(review=_agent_review(judge_url="http://127.0.0.1:11434/v1"))
        with pytest.raises(HTTPException) as exc:
            await main_module.start_eval(StartRequest(**payload))
        assert exc.value.status_code == 400
        assert "评审模型 URL" in exc.value.detail

    asyncio.run(_scenario())


# ---- embedding 透传 ----

def test_agent_judge_embedding_cfg_forwarded(monkeypatch):
    monkeypatch.setattr(main_module, "execute_all", _fake_execute_all)
    monkeypatch.setattr(main_module, "run_judge", _fake_run_judge)

    async def _scenario():
        job_id = await _run_agent_job(embedding={"provider": "offline"})
        assert captured["execute_kwargs"]["embedding_cfg"]["provider"] == "offline"
        report = storage.get_job_files(job_id)["report.json"]["report"]
        assert report["metrics"]["provider"]["kind"] == "offline"
        assert main_module._jobs[job_id]["state"] == "completed"

    asyncio.run(_scenario())
