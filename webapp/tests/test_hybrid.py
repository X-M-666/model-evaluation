# -*- coding: utf-8 -*-
"""Hybrid 评审接入主流程的集成测试（迭代三 步骤7）。

覆盖不变量：
- 全链路：executing → judging → reviewing(hybrid) → hybrid-review 提交 → completed
- 复核集选择（H2）：invalid 必选 → 分差降序 → 低分兜底；k=min(k_top, 候选)（L3）
- k==0 直通 completed（不进入复核态）
- 提交校验：轮次/题号/复核集归属/完整性/去重；409 状态机与降级后拒绝
- 覆盖语义：人工分覆盖被选题（winner/分数），未选题保留 agent 分
- fail_open 降级（H3）：judge 全败 → reviewing + config 落盘 degraded（纯人工可接管）
- 健康度告警：invalid 率超阈值 → judge_health invalid_rate 幂等推送
- M2 重启恢复：清空 _jobs 后凭磁盘态完成复核提交

说明：与 test_agent_judge 同款——单一 asyncio.run 循环内直调端点函数，
monkeypatch execute_all / run_judge（全程无真实网络）。
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


def _hybrid_review(k_top_human: int = 3, fail_open: bool = True,
                   judge_url: str = PUBLIC_URL) -> dict:
    return {
        "mode": "hybrid",
        "fail_open": fail_open,
        "k_top_human": k_top_human,
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
    answers_a = _fake_answers(task_set, config_a["name"])
    answers_b = _fake_answers(task_set, config_b["name"])
    total = task_set["meta"]["total"]
    if progress_cb:
        for i in range(1, total + 1):
            await progress_cb("a", i, total)
    return answers_a, answers_b


def _make_judge_stub(gaps: dict[str, float] | None = None,
                     invalid: set[str] | None = None,
                     fail: bool = False):
    """可控分差的 run_judge 替身：gap>0 答X胜（分差 gap），默认 gap=1.0。

    fail=True 时整体抛异常（模拟评审模型全败）；invalid 题返回 0/0+tie+_invalid。
    """
    gaps = gaps or {}
    invalid = invalid or set()

    async def _stub(task_set, answers_x, answers_y, judge_config,
                    revealed=None, progress_cb=None):
        captured["revealed"] = revealed
        if fail:
            raise RuntimeError("评审模型全败（模拟）")
        tasks = task_set["tasks"]
        scores, dim_totals = [], {}
        for t in tasks:
            tid = t["id"]
            if tid in invalid:
                scores.append({"id": tid, "dimension": t.get("dimension", ""),
                               "winner": "tie", "answer_x": 0.0, "answer_y": 0.0,
                               "basis": "评审模型未能返回有效 verdict",
                               "arbiter_note": None, "_invalid": True})
                continue
            gap = gaps.get(tid, 1.0)
            winner = "x" if gap >= 0 else "y"
            ax = 5.0 + gap / 2 if gap >= 0 else 5.0 + gap / 2
            ay = 5.0 - gap / 2 if gap >= 0 else 5.0 - gap / 2
            scores.append({"id": tid, "dimension": t.get("dimension", ""),
                           "winner": winner, "answer_x": ax, "answer_y": ay,
                           "basis": "fake", "arbiter_note": None, "_invalid": False})
            if not t.get("excluded_from_total"):
                dim_totals.setdefault(t.get("dimension", ""), {"x": 0.0, "y": 0.0})
                dim_totals[t.get("dimension", "")]["x"] += ax
                dim_totals[t.get("dimension", "")]["y"] += ay
        valid = sum(1 for s in scores if not s["_invalid"])
        if progress_cb:
            for i in range(1, len(tasks) + 1):
                await progress_cb(i, len(tasks))
        return {
            "meta": {"total": len(scores), "valid": valid,
                     "invalid": len(scores) - valid,
                     "tie_arbitrated": 0, "excluded_ids": [],
                     "excluded_dimensions": []},
            "scores": scores,
            "per_dimension": dim_totals,
            "totals": {"answer_x": sum(d["x"] for d in dim_totals.values()),
                       "answer_y": sum(d["y"] for d in dim_totals.values())},
            "revealed": {"answer_x": "answers-a.json", "answer_y": "answers-b.json",
                         "answer_x_file": "a", "answer_y_file": "b"},
            "conclusion": "fake conclusion",
            "winner_model": "tie",
        }

    return _stub


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


async def _run_hybrid_job(k_top_human: int = 3, **overrides) -> str:
    main_module._jobs.clear()
    main_module._tasks.clear()
    captured.clear()
    review = _hybrid_review(k_top_human=k_top_human)
    if "fail_open" in overrides:
        review["fail_open"] = overrides.pop("fail_open")
    payload = _payload(review=review)
    payload.update(overrides)
    resp = await main_module.start_eval(StartRequest(**payload))
    job_id = resp.job_id
    await _wait_task_done(job_id)
    return job_id


def _review_set_of(job_id: str) -> list[dict]:
    hdata = storage.load_hybrid_review(job_id)
    assert hdata is not None, "hybrid-review.json 未落盘"
    return hdata["review_set"]


def _submit_scores(job_id: str, human_map: dict[str, tuple[float, float, str]]) -> ReviewSubmission:
    """human_map: task_id -> (x分, y分, winner) 构造完整复核集提交。"""
    items = []
    for it in _review_set_of(job_id):
        tid = it["task_id"]
        if tid in human_map:
            ax, ay, win = human_map[tid]
        else:
            ax, ay, win = it["agent_x"], it["agent_y"], it["winner"]
        items.append({"id": tid, "round": it["round"], "answer_x": ax,
                      "answer_y": ay, "winner": win, "note": ""})
    return ReviewSubmission(scores=items)


# ---- 全链路：executing → judging → reviewing → 复核提交 → completed ----

def test_hybrid_full_chain_review_then_submit(monkeypatch):
    monkeypatch.setattr(main_module, "execute_all", _fake_execute_all)
    monkeypatch.setattr(main_module, "run_judge", _make_judge_stub())

    async def _scenario():
        job_id = await _run_hybrid_job(k_top_human=3)
        j = main_module._jobs[job_id]

        # 停在 reviewing，SSE 带 hybrid 信息
        assert j["state"] == "reviewing"
        events = _drain_events(job_id)
        assert "judging" in [e["state"] for e in events]
        review_evt = [e for e in events if e.get("state") == "reviewing"][0]
        assert review_evt.get("mode") == "hybrid" and review_evt.get("k") == 3

        # 复核集落盘且 k=3
        review_set = _review_set_of(job_id)
        assert len(review_set) == 3
        assert all(it["round"] == 1 for it in review_set)

        # review 视图（H4）：hybrid 字段带 agent 原分/K
        view = await main_module.eval_review_view(job_id)
        assert view["hybrid"]["k"] == 3
        assert len(view["hybrid"]["review_set"]) == 3
        keys = {f"{it['round']}:{it['task_id']}" for it in review_set}
        assert set(view["hybrid"]["agent_scores"]) == keys
        sample = review_set[0]
        ag = view["hybrid"]["agent_scores"][f"{sample['round']}:{sample['task_id']}"]
        assert ag["agent_x"] == sample["agent_x"]

        # 人工覆盖 top1 胜负、其余保留 agent 分
        top1 = review_set[0]
        top1_id = top1["task_id"]
        human = {top1_id: (9.0, 1.0, "answer_x")}
        await main_module.eval_hybrid_review_submit(
            job_id, _submit_scores(job_id, human))

        j = main_module._jobs[job_id]
        assert j["state"] == "completed"
        verdict = storage.get_job_files(job_id)["verdict.json"]
        scored = {s["id"]: s for s in verdict["scores"]}
        assert scored[top1_id]["winner"] == "answer_x"
        assert scored[top1_id]["answer_x"] == 9.0
        other = [it for it in review_set if it["task_id"] != top1_id]
        for it in other:
            expected_win = ("answer_x" if it["winner"] == "x"
                            else "answer_y" if it["winner"] == "y" else "tie")
            assert scored[it["task_id"]]["winner"] == expected_win
        # review 记录 mode=hybrid；报告 review.mode=hybrid
        assert storage.load_review(job_id)["mode"] == "hybrid"
        report = storage.get_job_files(job_id)["report.json"]["report"]
        assert report["review"]["mode"] == "hybrid"
        # 审计：agent 预评 + human 复核提交
        log = {e["event"]: e for e in audit.read_events() if e.get("job_id") == job_id}
        assert log["eval_judged"]["actor"] == "agent"
        assert log["review_submitted"]["actor"] == "human"

    asyncio.run(_scenario())


# ---- 复核集选择（H2/L3） ----

def test_hybrid_review_set_selection_invalid_first(monkeypatch):
    task_set = build_task_set()
    ids = [t["id"] for t in task_set["tasks"]]
    inv = set(ids[:2])
    monkeypatch.setattr(main_module, "execute_all", _fake_execute_all)
    monkeypatch.setattr(main_module, "run_judge",
                        _make_judge_stub(gaps={}, invalid=inv))

    async def _scenario():
        job_id = await _run_hybrid_job(k_top_human=3)
        review_set = _review_set_of(job_id)
        assert len(review_set) == 3
        # invalid 必选（H2）
        assert {it["task_id"] for it in review_set[:2]} == inv
        # 其余按分差降序（默认 gap=1.0 平分差 → 低分兜底优先）
        assert len(review_set) == 3
        assert all(it["invalid"] for it in review_set[:2])
        # k=min(k_top, 候选)（L3）：候选=全部题
        hdata = storage.load_hybrid_review(job_id)
        assert hdata["k"] == 3

    asyncio.run(_scenario())


def test_hybrid_review_set_gap_desc_then_low_score(monkeypatch):
    task_set = build_task_set()
    ids = [t["id"] for t in task_set["tasks"]]
    gaps = {ids[0]: 8.0, ids[1]: 2.0, ids[2]: 5.0}
    monkeypatch.setattr(main_module, "execute_all", _fake_execute_all)
    monkeypatch.setattr(main_module, "run_judge", _make_judge_stub(gaps=gaps))

    async def _scenario():
        job_id = await _run_hybrid_job(k_top_human=3)
        picked = [it["task_id"] for it in _review_set_of(job_id)]
        assert picked == [ids[0], ids[2], ids[1]]

    asyncio.run(_scenario())


def test_hybrid_k0_direct_completed(monkeypatch):
    monkeypatch.setattr(main_module, "execute_all", _fake_execute_all)
    monkeypatch.setattr(main_module, "run_judge", _make_judge_stub())

    async def _scenario():
        job_id = await _run_hybrid_job(k_top_human=0)
        j = main_module._jobs[job_id]
        assert j["state"] == "completed"
        # 无 reviewing 事件；review 记录 mode=hybrid 但 k=0 直通
        events = _drain_events(job_id)
        assert "reviewing" not in [e.get("state") for e in events]
        assert storage.load_hybrid_review(job_id) is None
        assert storage.load_review(job_id)["mode"] == "hybrid"
        # 视图：直通任务无复核态（hybrid 字段应缺省而非空对象）
        view = await main_module.eval_review_view(job_id)
        assert view["hybrid"] is None
        report = storage.get_job_files(job_id)["report.json"]["report"]
        assert report["review"]["k_top_human"] == 0

    asyncio.run(_scenario())


# ---- 提交校验 ----

def test_hybrid_submit_validation_and_409(monkeypatch):
    monkeypatch.setattr(main_module, "execute_all", _fake_execute_all)
    monkeypatch.setattr(main_module, "run_judge", _make_judge_stub())

    async def _scenario():
        # 非 hybrid 任务 → 409
        main_module._jobs.clear()
        main_module._tasks.clear()
        captured.clear()
        pure = _payload(review={"mode": "pure_human"})
        jid_pure = (await main_module.start_eval(StartRequest(**pure))).job_id
        with pytest.raises(HTTPException) as e1:
            await main_module.eval_hybrid_review_submit(jid_pure, ReviewSubmission(scores=[]))
        assert e1.value.status_code == 409

        job_id = await _run_hybrid_job(k_top_human=3)
        review_set = _review_set_of(job_id)
        ids = [it["task_id"] for it in review_set]

        # 409：hybrid 任务走纯人工端点被拒（须走 hybrid-review 覆盖接口）
        with pytest.raises(HTTPException) as e0:
            await main_module.eval_review_submit(job_id, _submit_scores(job_id, {}))
        assert e0.value.status_code == 409

        # 400：不完整（缺 1 条）
        items = _submit_scores(job_id, {}).scores[:-1]
        with pytest.raises(HTTPException) as e2:
            await main_module.eval_hybrid_review_submit(job_id, ReviewSubmission(scores=items))
        assert e2.value.status_code == 400
        assert "缺失" in e2.value.detail

        # 400：重复
        full = _submit_scores(job_id, {}).scores
        dup = full + [full[0]]
        with pytest.raises(HTTPException) as e3:
            await main_module.eval_hybrid_review_submit(job_id, ReviewSubmission(scores=dup))
        assert e3.value.status_code == 400
        assert "重复" in e3.value.detail

        # 400：不在复核集
        all_ids = [t["id"] for t in build_task_set()["tasks"]]
        outsider = next(t for t in all_ids if t not in set(ids))
        bad = [dict(s) for s in full]
        bad[0]["id"] = outsider
        with pytest.raises(HTTPException) as e4:
            await main_module.eval_hybrid_review_submit(job_id, ReviewSubmission(scores=bad))
        assert e4.value.status_code == 400
        assert "不属于 hybrid 复核集" in e4.value.detail

        # 400：轮次越界
        bad_round = [dict(s) for s in full]
        bad_round[0]["round"] = 9
        with pytest.raises(HTTPException) as e5:
            await main_module.eval_hybrid_review_submit(job_id, ReviewSubmission(scores=bad_round))
        assert e5.value.status_code == 400
        assert "轮次越界" in e5.value.detail

        # 首次提交成功
        await main_module.eval_hybrid_review_submit(job_id, _submit_scores(job_id, {}))
        # 409：已提交
        with pytest.raises(HTTPException) as e6:
            await main_module.eval_hybrid_review_submit(job_id, _submit_scores(job_id, {}))
        assert e6.value.status_code == 409

    asyncio.run(_scenario())


# ---- fail_open 降级（H3） ----

def test_hybrid_fail_open_degrades_to_pure_human(monkeypatch):
    monkeypatch.setattr(main_module, "execute_all", _fake_execute_all)
    monkeypatch.setattr(main_module, "run_judge", _make_judge_stub(fail=True))

    async def _scenario():
        job_id = await _run_hybrid_job(k_top_human=3, fail_open=True)
        j = main_module._jobs[job_id]
        assert j["state"] == "reviewing"
        events = _drain_events(job_id)
        health = [e for e in events if e.get("type") == "judge_health"
                  and e.get("status") == "degraded"]
        assert health

        # config 落盘降级标注（M2 可见）
        cfg = storage.get_job_files(job_id)["config.json"]
        assert cfg["review_mode"] == "pure_human"
        assert cfg["review_degraded"] is True
        # 降级后 hybrid 提交被拒；人工评审可接管
        with pytest.raises(HTTPException) as e:
            await main_module.eval_hybrid_review_submit(job_id, ReviewSubmission(scores=[]))
        assert e.value.status_code == 409
        tasks = build_task_set()
        human_scores = [{"id": t["id"], "round": 1, "answer_x": 7.0,
                         "answer_y": 6.0, "winner": "answer_x", "note": ""}
                        for t in tasks["tasks"]]
        await main_module.eval_review_submit(job_id, ReviewSubmission(scores=human_scores))
        assert main_module._jobs[job_id]["state"] == "completed"

    asyncio.run(_scenario())


def test_hybrid_judge_fail_no_fail_open_errors(monkeypatch):
    monkeypatch.setattr(main_module, "execute_all", _fake_execute_all)
    monkeypatch.setattr(main_module, "run_judge", _make_judge_stub(fail=True))

    async def _scenario():
        job_id = await _run_hybrid_job(k_top_human=3, fail_open=False)
        j = main_module._jobs[job_id]
        assert j["state"] == "error"
        assert "AI 评审失败" in j.get("error", "")

    asyncio.run(_scenario())


# ---- 健康度告警：invalid 率超阈值 ----

def test_hybrid_invalid_rate_alarm_sse(monkeypatch):
    task_set = build_task_set()
    ids = [t["id"] for t in task_set["tasks"]]
    # 部分 invalid（valid>0 不触发"评审失败"降级），整体 invalid 率 5/26≈0.19 > 0.1
    monkeypatch.setattr(main_module, "execute_all", _fake_execute_all)
    monkeypatch.setattr(main_module, "run_judge",
                        _make_judge_stub(invalid=set(ids[:5])))

    async def _scenario():
        job_id = await _run_hybrid_job(k_top_human=3)
        # 无效题必先进复核集（H2）
        review_set = _review_set_of(job_id)
        assert review_set and review_set[0]["invalid"]
        events = _drain_events(job_id)
        health = [e for e in events if e.get("type") == "judge_health"]
        alarms = [e for e in health if e.get("status") == "invalid_rate"]
        assert alarms
        assert alarms[0]["rate"] == pytest.approx(5 / 8, abs=1e-6)
        assert alarms[0]["threshold"] == 0.1

    asyncio.run(_scenario())


# ---- M2 重启恢复 ----

def test_hybrid_restart_recovery_submit_after_jobs_cleared(monkeypatch):
    monkeypatch.setattr(main_module, "execute_all", _fake_execute_all)
    monkeypatch.setattr(main_module, "run_judge", _make_judge_stub())

    async def _scenario():
        job_id = await _run_hybrid_job(k_top_human=2)
        assert main_module._jobs[job_id]["state"] == "reviewing"
        review_set = _review_set_of(job_id)
        assert len(review_set) == 2
        # 模拟重启：清空进程态，SSE 队列随之消失
        main_module._jobs.clear()

        # 凭磁盘态恢复并提交复核
        await main_module.eval_hybrid_review_submit(
            job_id, _submit_scores(job_id, {}))
        files = storage.get_job_files(job_id)
        assert files["verdict.json"]["totals"]
        report = files["report.json"]["report"]
        assert report["review"]["mode"] == "hybrid"

    asyncio.run(_scenario())


def test_hybrid_restart_recovery_multi_round(monkeypatch):
    """M2 多轮：repeat_n=2 的 hybrid 重启恢复后按 (round, task) 子集复核提交。"""
    monkeypatch.setattr(main_module, "execute_all", _fake_execute_all)
    monkeypatch.setattr(main_module, "run_judge", _make_judge_stub())

    async def _scenario():
        job_id = await _run_hybrid_job(k_top_human=3, repeat_n=2)
        assert main_module._jobs[job_id]["state"] == "reviewing"
        review_set = _review_set_of(job_id)
        assert len(review_set) == 3
        assert {it["round"] for it in review_set} <= {1, 2}
        main_module._jobs.clear()

        await main_module.eval_hybrid_review_submit(
            job_id, _submit_scores(job_id, {}))
        files = storage.get_job_files(job_id)
        assert files["verdict.json"]["meta"]["repeat_n"] == 2
        # 逐轮 reveals 落盘且与 round-verdicts 对齐（聚合用逐题映射）
        hdata = storage.load_hybrid_review(job_id)
        assert len(hdata["reveals"]) == 2
        assert len(files["round-verdicts.json"]) == 2

    asyncio.run(_scenario())


# ---- 纯人工视图差缺 hybrid 字段 ----

def test_review_view_hybrid_null_for_pure_human(monkeypatch):
    async def _scenario():
        main_module._jobs.clear()
        main_module._tasks.clear()
        captured.clear()
        pure = _payload(review={"mode": "pure_human"})
        resp = await main_module.start_eval(StartRequest(**pure))
        job_id = resp.job_id
        await _wait_task_done(job_id)
        view = await main_module.eval_review_view(job_id)
        assert view["hybrid"] is None

    asyncio.run(_scenario())


# ---- ReviewConfig 校验（迭代三：mode 枚举 / k_top_human 范围） ----

def test_review_config_validation():
    from pydantic import ValidationError
    async def _scenario():
        # 非法 mode → 校验错误（pydantic，直调端点不包装为 HTTPException）
        with pytest.raises(ValidationError):
            await main_module.start_eval(StartRequest(**_payload(review={"mode": "foo"})))
        # hybrid 缺 judge → 400（业务校验）
        with pytest.raises(HTTPException) as e2:
            await main_module.start_eval(
                StartRequest(**_payload(review={"mode": "hybrid", "k_top_human": 2})))
        assert e2.value.status_code == 400
        # k_top_human 越界 → 校验错误
        with pytest.raises(ValidationError):
            await main_module.start_eval(
                StartRequest(**_payload(review=_hybrid_review(k_top_human=21))))
        with pytest.raises(ValidationError):
            await main_module.start_eval(
                StartRequest(**_payload(review=_hybrid_review(k_top_human=-1))))

    asyncio.run(_scenario())