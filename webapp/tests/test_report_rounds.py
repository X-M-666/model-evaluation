# -*- coding: utf-8 -*-
"""多轮报告聚合回归测试（issue #10）。

核心验证：多轮（repeat_n>1）时报告不得混用"平均评分 + 最后一轮效率/答卷"，
效率/成本/代码/原文必须按稳定模型跨轮聚合，且逐轮明细可查。
"""
from __future__ import annotations

import statistics
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend import storage
from backend.engine.human_review import (
    build_final_verdict,
    build_round_verdict,
    make_reveal,
)
from backend.engine.report_builder import build_report
from backend.main import app, _jobs
from backend.storage import (
    create_job_id,
    save_answers,
    save_config,
    save_reveal,
    save_task_set,
)

client = TestClient(app)

MODEL_A = "模型A"
MODEL_B = "模型B"

TASK_SET = {
    "tasks": [
        {"id": "t1", "dimension": "知识"},
        {"id": "t2", "dimension": "代码能力"},
    ]
}


def _round_answers(label: str, latencies: dict[str, int], tokens: dict[str, tuple],
                   code: dict[str, tuple] | None = None,
                   raw: str | None = None) -> dict:
    """构造一轮答卷：latencies/tokens 按 {task_id: 值}，code 按 {task_id: (passed, total)}。"""
    answers = []
    for t in TASK_SET["tasks"]:
        tid = t["id"]
        entry = {
            "id": tid,
            "raw_answer": f"{label}-{tid}-{raw or 'r'}",
            "api_info": {
                "status": "ok",
                "latency_ms": latencies.get(tid, 100),
                "prompt_tokens": tokens.get(tid, (10, 20))[0],
                "completion_tokens": tokens.get(tid, (10, 20))[1],
                "repeat_index": 1,
            },
        }
        if code and tid in code:
            entry["code_verify"] = {"status": "run", "passed": code[tid][0], "total": code[tid][1]}
        answers.append(entry)
    return {"model": label, "answers": answers}


def _scores(mapping: tuple[str, str], x_vals: dict[str, float], y_vals: dict[str, float]) -> list[dict]:
    return [
        {"id": t["id"], "answer_x": x_vals.get(t["id"], 8.0), "answer_y": y_vals.get(t["id"], 2.0), "note": ""}
        for t in TASK_SET["tasks"]
    ]


def _make_round_verdict(mapping: tuple[str, str], round_idx: int, x_vals: dict, y_vals: dict) -> dict:
    x_file, y_file = mapping
    x_model = MODEL_A if x_file == "a" else MODEL_B
    y_model = MODEL_B if y_file == "b" else MODEL_A
    return build_round_verdict(
        TASK_SET, _scores(mapping, x_vals, y_vals),
        {"answer_x": x_file, "answer_y": y_file}, x_model, y_model, round_idx,
    )


@pytest.fixture(autouse=True)
def _isolate_history(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(storage, "BASE_DIR", tmp_path)
    yield
    for jid in list(_jobs):
        _jobs.pop(jid)


def _multi_job() -> tuple[dict, dict, list[dict], dict, dict]:
    """构造两轮任务：轮1 X=A/Y=B，轮2 X=B/Y=A（身份交换）。

    稳定模型 A 每轮延迟 100ms、token (10,20)；B 延迟 200ms、token (30,40)。
    轮1 代码：A 5/5、B 4/5；轮2 代码：A 3/5、B 5/5。
    """
    cfg = {
        "model_a": {"name": MODEL_A, "url": "https://example.com", "key": "k"},
        "model_b": {"name": MODEL_B, "url": "https://example.com", "key": "k"},
        "repeat_n": 2, "seed": 7, "dataset_name": None,
        "dims": None, "code_verify_mode": "mock",
    }
    reveal = {"rounds": [{"answer_x": "a", "answer_y": "b"}, {"answer_x": "b", "answer_y": "a"}]}

    rounds_answers = [
        {
            "a": _round_answers(MODEL_A, {"t1": 100, "t2": 100}, {"t1": (10, 20), "t2": (10, 20)},
                                code={"t2": (5, 5)}, raw="r1"),
            "b": _round_answers(MODEL_B, {"t1": 200, "t2": 200}, {"t1": (30, 40), "t2": (30, 40)},
                                code={"t2": (4, 5)}, raw="r1"),
        },
        {
            "a": _round_answers(MODEL_A, {"t1": 100, "t2": 100}, {"t1": (10, 20), "t2": (10, 20)},
                                code={"t2": (3, 5)}, raw="r2"),
            "b": _round_answers(MODEL_B, {"t1": 200, "t2": 200}, {"t1": (30, 40), "t2": (30, 40)},
                                code={"t2": (5, 5)}, raw="r2"),
        },
    ]
    round_verdicts = [
        _make_round_verdict(("a", "b"), 0, {"t1": 9.0, "t2": 8.0}, {"t1": 1.0, "t2": 2.0}),
        _make_round_verdict(("b", "a"), 1, {"t1": 1.0, "t2": 2.0}, {"t1": 9.0, "t2": 8.0}),
    ]
    verdict = build_final_verdict(round_verdicts, 2)
    report = build_report(cfg, TASK_SET, rounds_answers[1]["a"], rounds_answers[1]["b"],
                          verdict, rounds_answers)
    return cfg, verdict, rounds_answers, round_verdicts, report


# ---- 验收1：总 Token = 两轮实际之和 ----

def test_total_tokens_is_sum_of_all_rounds():
    _, _, _, _, report = _multi_job()
    # 最后一轮 reveal X=B：A 每轮 (10,20)×2 轮 → prompt 20/completion 40；B 每轮 (30,40)×2 → 60/80
    ch = report["charts"]["tokens_by_question"]
    assert ch["x_prompt"] == [60, 60]
    assert ch["x_completion"] == [80, 80]
    assert ch["y_prompt"] == [20, 20]
    assert ch["y_completion"] == [40, 40]
    # 分析文本含"2 轮合计"且数值为跨轮之和（X=B：120+160=280）
    paras = [p for sec in report["analysis"] for p in sec["paragraphs"] if "总 token 消耗" in p]
    assert any("（2 轮合计）" in p and "280" in p for p in paras)


# ---- 验收2：每轮均值/标准差/中位数与手算一致 ----

def test_median_and_std_match_manual_calc():
    _, verdict, _, _, report = _multi_job()
    by_id = {s["id"]: s for s in verdict["scores"]}
    # t1: A 两轮 9/9 → mean 9, std 0, median 9；B 两轮 1/1 → 1/0/1
    assert by_id["t1"]["model_a"] == 9.0
    assert by_id["t1"]["model_b"] == 1.0
    assert by_id["t1"]["model_a_median"] == 9.0
    assert by_id["t1"]["model_b_median"] == 1.0
    assert by_id["t1"]["model_a_std"] == 0
    # 投影到最后一轮 reveal（X=B）：X 显示 B 的分数
    assert by_id["t1"]["answer_x"] == 1.0
    assert by_id["t1"]["answer_y"] == 9.0
    # 逐题表中位数与手算一致
    ch = report["charts"]["stability"]
    assert ch["x_median"] == [1.0, 2.0]
    assert ch["y_median"] == [9.0, 8.0]
    # 延迟中位数：A 稳定 100ms、B 稳定 200ms → 聚合后 X(B)=200, Y(A)=100
    # 通过 latency 图表断言平均（单值轮次平均值=中位数）
    lat = report["charts"]["latency_by_question"]
    assert lat["x"] == [200, 200]
    assert lat["y"] == [100, 100]


# ---- 验收3：调用成功率 = 成功轮次/总轮次 ----

def test_success_rate_per_model():
    cfg, verdict, rounds_answers, _, _ = _multi_job()
    # 把 B 轮1 的 t1 改成失败（status fail）
    rounds_answers[0]["b"]["answers"][0]["api_info"]["status"] = "fail"
    rounds_answers[0]["b"]["answers"][0]["api_info"]["latency_ms"] = 0
    report = build_report(cfg, TASK_SET, rounds_answers[1]["a"], rounds_answers[1]["b"],
                          verdict, rounds_answers)
    # 最后一轮 reveal：X=B → B 失败 1 轮（跨 2 题×2 轮汇总 3/4），A 4/4
    text = report["analysis"]
    success_paras = [p for sec in text for p in sec["paragraphs"] if "调用成功率" in p]
    assert success_paras, "多轮报告应包含调用成功率段落"
    assert MODEL_B in success_paras[0] and "3/4" in success_paras[0]
    assert MODEL_A in success_paras[0] and "4/4" in success_paras[0]


# ---- 验收4：两轮代码通过率均可查看，聚合符合语义 ----

def test_code_pass_rates_per_round_and_aggregated():
    _, _, _, _, report = _multi_job()
    # X=B 聚合：B 4/5 + 5/5 = 9/10；Y=A：5/5 + 3/5 = 8/10
    ch = report["charts"]["code_pass_rate"]
    by_id = {q["id"]: q for q in ch}
    assert by_id["t2"]["x_passed"] == 9 and by_id["t2"]["x_total"] == 10
    assert by_id["t2"]["y_passed"] == 8 and by_id["t2"]["y_total"] == 10
    # 逐轮明细可查（分析文本）
    code_paras = [p for sec in report["analysis"] for p in sec["paragraphs"] if "逐轮明细" in p]
    assert any("第1轮 5/5" in p for p in code_paras)
    assert any("第1轮 4/5" in p and "第2轮 5/5" in p for p in code_paras)


# ---- 验收5：答案原文能区分轮次（接口 rounds 字段逐轮归一化，见接口级测试） ----


# ---- 验收6：单轮（repeat_n=1）行为与修复前一致 ----

def test_single_round_unchanged():
    cfg = {"repeat_n": 1, "model_a": {"name": MODEL_A}, "model_b": {"name": MODEL_B}}
    answers_a = _round_answers(MODEL_A, {"t1": 100, "t2": 100}, {"t1": (10, 20), "t2": (10, 20)})
    answers_b = _round_answers(MODEL_B, {"t1": 200, "t2": 200}, {"t1": (30, 40), "t2": (30, 40)})
    v = _make_round_verdict(("a", "b"), 0, {"t1": 8.0}, {"t1": 2.0})
    verdict = build_final_verdict([v], 1)
    # 不传 rounds_answers：单轮原路径（t1 显式 100/200，t2 默认 100/200）
    report = build_report(cfg, TASK_SET, answers_a, answers_b, verdict)
    lat = report["charts"]["latency_by_question"]
    assert lat["x"] == [100, 100]
    assert lat["y"] == [200, 200]
    # 传单元素 rounds_answers：同样走单轮路径
    report2 = build_report(cfg, TASK_SET, answers_a, answers_b, verdict,
                           [{"a": answers_a, "b": answers_b}])
    assert report2["charts"]["latency_by_question"]["x"] == [100, 100]
    assert "调用成功率" not in [p for sec in report2["analysis"] for p in sec["paragraphs"]]
    # 单轮 verdict 也带 median 与 round_reveals
    assert verdict["meta"]["round_reveals"] == [{"answer_x_file": "a", "answer_y_file": "b"}]
    assert verdict["scores"][0]["answer_x_median"] == 8.0


# ---- 验收7：X/Y 每轮交换时指标归属正确真实模型 ----

def test_metric_attribution_under_round_swap():
    """轮1 X=A、轮2 X=B：稳定模型 A 的延迟/Token 必须投影为对应轮次的真实模型，
    聚合后 X(B)=200ms、Y(A)=100ms，与 build_final_verdict 的投影一致。"""
    cfg, verdict, rounds_answers, _, report = _multi_job()
    lat = report["charts"]["latency_by_question"]
    assert lat["x"] == [200, 200]
    assert lat["y"] == [100, 100]
    # 与反向映射（轮1 X=B、轮2 X=A）的结果必须一致（稳定空间与映射无关）
    rev_rounds = [
        {
            "a": rounds_answers[0]["a"], "b": rounds_answers[0]["b"],
        },
        {
            "a": rounds_answers[1]["a"], "b": rounds_answers[1]["b"],
        },
    ]
    # 直接构造反向 reveal 的 verdict：仅展示投影不同，聚合指标（延迟/token）按模型不变
    rev_verdicts = [
        _make_round_verdict(("b", "a"), 0, {"t1": 9.0, "t2": 8.0}, {"t1": 1.0, "t2": 2.0}),
        _make_round_verdict(("a", "b"), 1, {"t1": 9.0, "t2": 8.0}, {"t1": 1.0, "t2": 2.0}),
    ]
    rev_verdict = build_final_verdict(rev_verdicts, 2)
    rev_report = build_report(cfg, TASK_SET, rev_rounds[1]["a"], rev_rounds[1]["b"],
                              rev_verdict, rev_rounds)
    rev_lat = rev_report["charts"]["latency_by_question"]
    # 反向 reveal 下 X=A → X=100ms；两个报告里真实模型 A 的指标恒为 100ms
    assert rev_lat["x"] == [100, 100]
    assert rev_lat["y"] == [200, 200]
    assert rev_lat["y"] == lat["x"]


# ---- 接口级：磁盘态/内存态报告均返回 rounds 与聚合值 ----

def _seed_disk_job(repeat_n: int, rounds_answers: list[dict]) -> str:
    job_id = create_job_id()
    cfg = {
        "model_a": {"name": MODEL_A, "url": "https://example.com", "key": "k"},
        "model_b": {"name": MODEL_B, "url": "https://example.com", "key": "k"},
        "dims": None, "seed": 7, "dataset_name": None,
        "repeat_n": repeat_n, "code_verify_mode": "off",
    }
    save_config(job_id, cfg)
    save_task_set(job_id, TASK_SET)
    for i, ra in enumerate(rounds_answers):
        save_answers(job_id, f"a-r{i+1}", ra["a"])
        save_answers(job_id, f"b-r{i+1}", ra["b"])
    save_answers(job_id, "a", rounds_answers[-1]["a"])
    save_answers(job_id, "b", rounds_answers[-1]["b"])
    # 固定 reveal：轮1 X=A/Y=B，轮2 X=B/Y=A（与 _multi_job 一致，保证断言确定性）
    reveal = {"rounds": [
        {"answer_x": "a", "answer_y": "b"},
        {"answer_x": "b", "answer_y": "a"},
    ][:repeat_n]}
    save_reveal(job_id, reveal)
    # 完成闭环：落盘 verdict/round-verdicts/报告（磁盘态接口才能生成报告）
    from backend.storage import save_report, save_round_verdicts, save_verdict
    round_verdicts = []
    for r_idx, ra in enumerate(rounds_answers):
        x_file = reveal["rounds"][r_idx]["answer_x"]
        x_vals, y_vals = ({"t1": 9.0, "t2": 8.0}, {"t1": 1.0, "t2": 2.0}) if x_file == "a" \
            else ({"t1": 1.0, "t2": 2.0}, {"t1": 9.0, "t2": 8.0})
        round_verdicts.append(_make_round_verdict(
            (x_file, reveal["rounds"][r_idx]["answer_y"]),
            r_idx, x_vals, y_vals,
        ))
    verdict = build_final_verdict(round_verdicts, repeat_n)
    save_verdict(job_id, verdict)
    save_round_verdicts(job_id, round_verdicts)
    save_report(job_id, {
        "config": cfg, "tasks": TASK_SET,
        "answers_a": rounds_answers[-1]["a"], "answers_b": rounds_answers[-1]["b"],
        "verdict": verdict,
        "report": build_report(cfg, TASK_SET, rounds_answers[-1]["a"],
                               rounds_answers[-1]["b"], verdict, rounds_answers),
    })
    return job_id


def test_report_endpoint_rounds_disk_state():
    job_id = _seed_disk_job(2, _multi_job()[2])
    resp = client.get(f"/api/eval/{job_id}/report")
    assert resp.status_code == 200
    body = resp.json()
    # rounds 逐轮归一化：与 reveal 一致
    assert len(body["rounds"]) == 2
    assert body["rounds"][0]["answers_x"]["answers"][0]["raw_answer"].endswith("r1")
    assert body["rounds"][1]["answers_x"]["answers"][0]["raw_answer"].endswith("r2")
    # 报告聚合：最后一轮 reveal X=B → 两轮 B token 求和（30+30/40+40）
    toks = body["report"]["charts"]["tokens_by_question"]
    assert toks["x_prompt"] == [60, 60]
    assert toks["x_completion"] == [80, 80]


def test_report_endpoint_rounds_in_memory_state():
    job_id = _seed_disk_job(2, _multi_job()[2])
    # 进程内注册任务（模拟运行中任务），接口走内存态分支
    cfg, verdict, rounds_answers, _, _ = _multi_job()
    _jobs[job_id] = {
        "state": "completed", "progress": "done",
        "config": cfg, "task_set": TASK_SET,
        "answers_a": rounds_answers[1]["a"], "answers_b": rounds_answers[1]["b"],
        "verdict": verdict, "rounds_answers": rounds_answers,
        "reveal": {"rounds": [{"answer_x": "a", "answer_y": "b"}, {"answer_x": "b", "answer_y": "a"}]},
        "created_at": "t", "sse_queue": None, "repeat_n": 2,
    }
    resp = client.get(f"/api/eval/{job_id}/report")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["rounds"]) == 2
    assert body["rounds"][1]["answers_x"]["answers"][0]["raw_answer"].endswith("r2")
    assert body["report"]["charts"]["latency_by_question"]["x"] == [200, 200]


def test_old_report_without_rounds_still_works():
    """旧记录（无轮次文件、verdict 无 round_reveals）必须可读且不崩溃。"""
    job_id = _seed_disk_job(1, [_multi_job()[2][0]])
    # 制造旧记录：无 r1 文件，仅 answers-a/b + verdict + report
    from backend.storage import save_report, save_verdict
    answers_a = _round_answers(MODEL_A, {"t1": 100}, {"t1": (10, 20)})
    answers_b = _round_answers(MODEL_B, {"t1": 200}, {"t1": (30, 40)})
    v = _make_round_verdict(("a", "b"), 0, {"t1": 8.0}, {"t1": 2.0})
    verdict = build_final_verdict([v], 1)
    report = build_report({"repeat_n": 1}, TASK_SET, answers_a, answers_b, verdict)
    save_verdict(job_id, verdict)
    save_report(job_id, {
        "config": {"repeat_n": 1}, "tasks": TASK_SET,
        "answers_a": answers_a, "answers_b": answers_b,
        "verdict": verdict, "report": report,
    })
    resp = client.get(f"/api/eval/{job_id}/report")
    assert resp.status_code == 200
    body = resp.json()
    assert body["report"]["summary"]["repeat_n"] == 1
    assert len(body["rounds"]) == 1


def test_round_scores_endpoint_disk_state():
    """验收补充：报告接口返回每轮每题原始打分（磁盘态）。"""
    job_id = _seed_disk_job(2, _multi_job()[2])
    resp = client.get(f"/api/eval/{job_id}/report")
    assert resp.status_code == 200
    body = resp.json()
    rs = body["round_scores"]
    assert len(rs) == 2
    assert [r["round"] for r in rs] == [1, 2]
    # 固定 reveal：轮1 X=A（t1 得 9）、轮2 X=B（t1 得 1）
    t1_r1 = next(s for s in rs[0]["scores"] if s["id"] == "t1")
    assert t1_r1["answer_x"] == 9.0 and t1_r1["answer_y"] == 1.0
    t1_r2 = next(s for s in rs[1]["scores"] if s["id"] == "t1")
    assert t1_r2["answer_x"] == 1.0 and t1_r2["answer_y"] == 9.0
    # 聚合展示（最后一轮 reveal X=B）与每轮明细一致
    assert body["verdict"]["scores"][0]["answer_x"] == 1.0


def test_round_scores_endpoint_in_memory_state():
    """验收补充：报告接口返回每轮原始打分（内存态）。"""
    job_id = _seed_disk_job(2, _multi_job()[2])
    cfg, verdict, rounds_answers, round_verdicts, _ = _multi_job()
    _jobs[job_id] = {
        "state": "completed", "progress": "done",
        "config": cfg, "task_set": TASK_SET,
        "answers_a": rounds_answers[1]["a"], "answers_b": rounds_answers[1]["b"],
        "verdict": verdict, "rounds_answers": rounds_answers,
        "round_verdicts": round_verdicts,
        "reveal": {"rounds": [{"answer_x": "a", "answer_y": "b"}, {"answer_x": "b", "answer_y": "a"}]},
        "created_at": "t", "sse_queue": None, "repeat_n": 2,
    }
    resp = client.get(f"/api/eval/{job_id}/report")
    assert resp.status_code == 200
    body = resp.json()
    rs = body["round_scores"]
    assert len(rs) == 2
    t2_r2 = next(s for s in rs[1]["scores"] if s["id"] == "t2")
    assert t2_r2["answer_x"] == 2.0 and t2_r2["answer_y"] == 8.0


def test_old_multi_round_report_without_rounds_files():
    """旧多轮记录：磁盘仅最后一轮答卷 + verdict 无 round_reveals，报告可读且不崩溃。

    历史任务（升级前完成）磁盘上没有 answers-a-r{n}.json 与 round_reveals，
    报告必须回退为单轮口径渲染，且接口 rounds 兜底为 1 轮。
    """
    from backend.storage import save_report, save_verdict
    job_id = create_job_id()
    cfg = {
        "model_a": {"name": MODEL_A, "url": "https://example.com", "key": "k"},
        "model_b": {"name": MODEL_B, "url": "https://example.com", "key": "k"},
        "dims": None, "seed": 7, "dataset_name": None,
        "repeat_n": 2, "code_verify_mode": "off",
    }
    save_config(job_id, cfg)
    save_task_set(job_id, TASK_SET)
    answers_a = _round_answers(MODEL_A, {"t1": 100, "t2": 100}, {"t1": (10, 20), "t2": (10, 20)})
    answers_b = _round_answers(MODEL_B, {"t1": 200, "t2": 200}, {"t1": (30, 40), "t2": (30, 40)})
    save_answers(job_id, "a", answers_a)
    save_answers(job_id, "b", answers_b)
    # 旧格式 verdict：无 round_reveals、无 median 字段
    verdict = {
        "meta": {"repeat_n": 2, "total": 2, "valid": 2, "invalid": 0},
        "revealed": {"answer_x_file": "a", "answer_y_file": "b",
                     "answer_x": MODEL_A, "answer_y": MODEL_B},
        "scores": [
            {"id": "t1", "answer_x": 8.0, "answer_y": 2.0, "winner": "answer_x", "basis": "旧记录"},
            {"id": "t2", "answer_x": 8.0, "answer_y": 2.0, "winner": "answer_x", "basis": "旧记录"},
        ],
        "totals": {"answer_x": 16.0, "answer_y": 4.0},
    }
    save_verdict(job_id, verdict)
    save_report(job_id, {
        "config": cfg, "tasks": TASK_SET,
        "answers_a": answers_a, "answers_b": answers_b,
        "verdict": verdict,
        "report": build_report(cfg, TASK_SET, answers_a, answers_b, verdict),
    })
    resp = client.get(f"/api/eval/{job_id}/report")
    assert resp.status_code == 200
    body = resp.json()
    assert body["report"] is not None
    assert body["report"]["summary"]["repeat_n"] == 2
    # 兜底：仅 1 轮可读（answers-a.json 覆盖版），不冒充多轮
    assert len(body["rounds"]) == 1
    # 无 round-verdicts 文件 → round_scores 为空
    assert body["round_scores"] == []
