# -*- coding: utf-8 -*-
"""迭代六：N 模型排行榜聚合（leaderboard.py）单测。

覆盖：reveal 归一化抽取、综合分/分维度排名、胜率矩阵、paired bootstrap CI
（显著与「差异不显著」）、K-召回率、雷达均分、排除维不计分、错误语义
（不存在/未完成/评测集不一致）、API 端点。
"""
import asyncio

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend import audit
from backend import main as main_module
from backend import storage
from backend.engine.leaderboard import (
    LeaderboardError,
    build_leaderboard,
    extract_job_model_scores,
)
from backend.engine.perturb import k_recall_curve
from backend.schemas import LeaderboardRequest

TASKS = [
    {"id": "T1", "dimension": "数学能力", "type": "判别式"},
    {"id": "T2", "dimension": "数学能力", "type": "判别式"},
    {"id": "T3", "dimension": "语言能力", "type": "判别式"},
    {"id": "T4", "dimension": "安全与价值观", "type": "判别式",
     "excluded_from_total": True},
]


def _task_set():
    return {"meta": {"total": len(TASKS)}, "tasks": TASKS}


def _verdict(scores, x="模型A", y="模型B", x_file="a", y_file="b"):
    return {
        "meta": {"total": len(scores), "valid": len(scores), "invalid": 0,
                 "repeat_n": 1, "excluded_ids": ["T4"],
                 "excluded_dimensions": ["安全与价值观"]},
        "scores": scores,
        "per_dimension": {},
        "totals": {"answer_x": 0, "answer_y": 0},
        "revealed": {"answer_x": x, "answer_y": y,
                     "answer_x_file": x_file, "answer_y_file": y_file},
        "conclusion": "", "winner_model": "tie",
    }


def _answer(label, scores):
    """构造 answers 文件：每模型侧全部题（score 用于 api_info 生成）。"""
    entries = []
    for tid, sc in scores.items():
        entries.append({
            "id": tid, "raw_answer": f"答案 {label}-{tid}",
            "api_info": {"status": "ok", "truncated": False, "error": None,
                         "latency_ms": int(sc * 100), "prompt_tokens": 100,
                         "completion_tokens": 50, "repeat_index": 1},
        })
    return {"model": label, "api": {"name": label}, "answers": entries}


def _make_job(name, scores_a, scores_b, x="模型A", y="模型B", x_file="a",
              y_file="b", with_report=True):
    jid = storage.create_job_id()
    storage.save_config(jid, {"model_a": {"name": x, "url": "https://example.com/v1"},
                              "model_b": {"name": y, "url": "https://example.com/v1"},
                              "dataset_name": "demo"})
    storage.save_task_set(jid, _task_set())
    storage.save_answers(jid, "a", _answer("a", scores_a))
    storage.save_answers(jid, "b", _answer("b", scores_b))
    scores = [{"id": tid, "answer_x": a, "answer_y": b}
              for tid, a, b in zip(sorted(scores_a), [scores_a[t] for t in sorted(scores_a)],
                                   [scores_b[t] for t in sorted(scores_b)])]
    storage.save_verdict(jid, _verdict(scores, x, y, x_file, y_file))
    if with_report:
        storage.save_report(jid, {"config": {}, "tasks": _task_set(),
                                  "answers_a": {}, "answers_b": {},
                                  "verdict": {}, "report": {}})
    return jid


class TestExtract:
    def test_reveal_normalization_cross_files(self):
        jid = storage.create_job_id()
        storage.save_config(jid, {"model_a": {"name": "A"}, "model_b": {"name": "B"}})
        storage.save_task_set(jid, _task_set())
        storage.save_answers(jid, "a", _answer("a", {"T1": 5}))
        storage.save_answers(jid, "b", _answer("b", {"T1": 9}))
        storage.save_verdict(jid, _verdict(
            [{"id": "T1", "answer_x": 7.5, "answer_y": 6.0}],
            x="模型X", y="模型Y", x_file="b", y_file="a"))  # X 侧文件 = b
        out = extract_job_model_scores(jid)
        assert set(out) == {"模型X", "模型Y"}
        assert out["模型X"]["T1"]["score"] == 7.5
        assert out["模型X"]["T1"]["latency_ms"] == 900  # 取自 b 文件
        assert out["模型Y"]["T1"]["score"] == 6.0
        assert out["模型Y"]["T1"]["latency_ms"] == 500

    def test_missing_job_returns_none(self):
        assert extract_job_model_scores("nope") is None


class TestBuildLeaderboard:
    def test_composite_and_ranks(self):
        j1 = _make_job("j1", {"T1": 8.0, "T2": 7.0, "T3": 6.0, "T4": 5.0},
                       {"T1": 6.0, "T2": 5.0, "T3": 4.0, "T4": 9.0})
        j2 = _make_job("j2", {"T1": 6.0, "T2": 5.0, "T3": 4.0, "T4": 9.0},
                       {"T1": 9.0, "T2": 8.0, "T3": 7.0, "T4": 3.0},
                       x="模型B", y="模型C", x_file="a", y_file="b")
        lb = build_leaderboard([j1, j2], name="demo榜", seed=1)
        assert lb["models"] == ["模型A", "模型B", "模型C"]
        assert lb["name"] == "demo榜"
        assert lb["dataset"] == "demo"
        # 综合分 = 计分题合计（T4 不计分）
        assert lb["composite"]["模型A"]["score"] == 21.0   # 8+7+6
        assert lb["composite"]["模型B"]["score"] == 15.0  # 6+5+4
        assert lb["composite"]["模型C"]["score"] == 24.0  # 9+8+7
        assert lb["ranks"]["模型C"] == 1
        assert lb["ranks"]["模型A"] == 2
        assert lb["ranks"]["模型B"] == 3
        assert lb["excluded_dimensions"] == ["安全与价值观"]
        assert lb["note"]

    def test_per_dim_ranking(self):
        j1 = _make_job("j1", {"T1": 8.0, "T2": 7.0, "T3": 6.0, "T4": 5.0},
                       {"T1": 6.0, "T2": 5.0, "T3": 4.0, "T4": 9.0})
        lb = build_leaderboard([j1], seed=1)
        assert lb["dims"] == ["安全与价值观", "数学能力", "语言能力"]
        assert lb["per_dim"]["数学能力"]["模型A"] == 15.0
        assert lb["per_dim"]["语言能力"]["模型A"] == 6.0

    def test_win_matrix_and_ci(self):
        j1 = _make_job("j1", {"T1": 9.0, "T2": 9.0, "T3": 8.0, "T4": 5.0},
                       {"T1": 5.0, "T2": 5.0, "T3": 4.0, "T4": 9.0})
        lb = build_leaderboard([j1], seed=7)
        wm = lb["win_matrix"]["模型A"]["模型B"]
        assert wm["wins"] == 3 and wm["losses"] == 0 and wm["total"] == 3
        ci = lb["ci"]["模型A"]["模型B"]
        assert ci["n"] == 3
        assert ci["significant"] is False            # 题数不足
        assert "差异不显著" in ci["note"]

    def test_ci_significant_with_large_sample(self):
        scores_a = {f"T{i}": 8.0 for i in range(1, 13)}
        scores_a["T4"] = 5.0
        scores_b = {f"T{i}": 4.0 for i in range(1, 13)}
        scores_b["T4"] = 9.0
        tasks = [{"id": f"T{i}", "dimension": "数学能力", "type": "判别式"}
                 for i in range(1, 13)]
        tasks[3]["id"] = "T4"
        tasks[3]["excluded_from_total"] = True
        jid = storage.create_job_id()
        storage.save_config(jid, {"model_a": {"name": "A"}, "model_b": {"name": "B"}})
        storage.save_task_set(jid, {"meta": {"total": 12}, "tasks": tasks})
        storage.save_answers(jid, "a", _answer("a", scores_a))
        storage.save_answers(jid, "b", _answer("b", scores_b))
        scores = [{"id": t, "answer_x": scores_a[t], "answer_y": scores_b[t]}
                  for t in sorted(scores_a)]
        storage.save_verdict(jid, _verdict(scores))
        storage.save_report(jid, {"report": {}})
        lb = build_leaderboard([jid], seed=3)
        ci = lb["ci"]["模型A"]["模型B"]
        assert ci["n"] == 11                      # 计分题
        assert ci["significant"] is True
        assert ci["mean"] > 3.0

    def test_k_recall_curve(self):
        curve = k_recall_curve({"T1": 9.0, "T2": 8.0, "T3": 2.0, "T4": 7.0},
                               threshold=6.0)
        assert curve["n_passed"] == 3
        assert curve["ks"] == [1, 2, 3, 4]
        assert curve["recalls"] == [0.3333, 0.6667, 1.0, 1.0]
        assert curve["threshold"] == 6.0

    def test_k_recall_in_leaderboard(self):
        j1 = _make_job("j1", {"T1": 9.0, "T2": 8.0, "T3": 2.0, "T4": 5.0},
                       {"T1": 1.0, "T2": 1.0, "T3": 1.0, "T4": 9.0})
        lb = build_leaderboard([j1], seed=1)
        kr = lb["k_recall"]["模型A"]
        assert kr["n_passed"] == 2
        assert kr["recalls"][-1] == 1.0

    def test_radar_and_box(self):
        j1 = _make_job("j1", {"T1": 8.0, "T2": 7.0, "T3": 6.0, "T4": 5.0},
                       {"T1": 6.0, "T2": 5.0, "T3": 4.0, "T4": 9.0})
        lb = build_leaderboard([j1], seed=1)
        assert lb["radar"]["avg"]["模型A"]["数学能力"] == 7.5
        box = lb["score_dist"]["模型A"]
        assert box["min"] == 6.0 and box["max"] == 8.0
        assert len(lb["scatter"]["模型A"]) == 3  # 仅计分题

    def test_missing_job_raises(self):
        with pytest.raises(LeaderboardError) as ei:
            build_leaderboard(["00000000_000000_000000"])
        assert "不存在" in str(ei.value)

    def test_unfinished_job_raises(self):
        jid = _make_job("j1", {"T1": 8.0, "T2": 7.0, "T3": 6.0, "T4": 5.0},
                        {"T1": 6.0, "T2": 5.0, "T3": 4.0, "T4": 9.0},
                        with_report=False)
        with pytest.raises(LeaderboardError) as ei:
            build_leaderboard([jid])
        assert "未完成" in str(ei.value)

    def test_inconsistent_datasets_raise(self):
        j1 = _make_job("j1", {"T1": 8.0, "T2": 7.0, "T3": 6.0, "T4": 5.0},
                       {"T1": 6.0, "T2": 5.0, "T3": 4.0, "T4": 9.0})
        jid = storage.create_job_id()
        storage.save_config(jid, {"model_a": {"name": "A"}, "model_b": {"name": "B"}})
        storage.save_task_set(jid, {"meta": {"total": 1},
                                    "tasks": [{"id": "X1", "dimension": "数学能力"}]})
        storage.save_verdict(jid, _verdict([{"id": "X1", "answer_x": 1.0,
                                             "answer_y": 2.0}]))
        storage.save_report(jid, {"report": {}})
        with pytest.raises(LeaderboardError) as ei:
            build_leaderboard([j1, jid])
        assert "不一致" in str(ei.value)

    def test_same_model_across_jobs_merges(self):
        j1 = _make_job("j1", {"T1": 8.0, "T2": 7.0, "T3": 6.0, "T4": 5.0},
                       {"T1": 6.0, "T2": 5.0, "T3": 4.0, "T4": 9.0},
                       x="模型A", y="模型B")
        j2 = _make_job("j2", {"T1": 9.0, "T2": 9.0, "T3": 9.0, "T4": 9.0},
                       {"T1": 1.0, "T2": 1.0, "T3": 1.0, "T4": 1.0},
                       x="模型C", y="模型B")
        lb = build_leaderboard([j1, j2], seed=1)
        # 模型B 同题 last-wins（来自 j2）
        assert lb["composite"]["模型B"]["score"] == 3.0
        assert lb["models"] == ["模型A", "模型B", "模型C"]


# ---- API 端点（迭代六）----


def _call(fn, *a, **k):
    return asyncio.run(fn(*a, **k))


@pytest.fixture(autouse=True)
def _clean_audit():
    audit._log_path().write_text("", encoding="utf-8")
    yield


@pytest.fixture
def client():
    return TestClient(main_module.app)


def test_api_create_and_list(client):
    j1 = _make_job("j1", {"T1": 8.0, "T2": 7.0, "T3": 6.0, "T4": 5.0},
                   {"T1": 6.0, "T2": 5.0, "T3": 4.0, "T4": 9.0},
                   x="模型A", y="模型B")
    resp = _call(main_module.create_leaderboard, LeaderboardRequest(
        name="榜单1", job_ids=[j1]))
    assert resp["ok"] is True
    lb_id = resp["lb_id"]
    assert lb_id.startswith("lb_")
    assert set(resp["models"]) == {"模型A", "模型B"}
    events = audit.read_events()
    assert any(e["event"] == "leaderboard_created" and e["target"] == lb_id
               for e in events)
    # 落盘与详情
    detail = _call(main_module.leaderboard_detail, lb_id)
    assert detail["name"] == "榜单1"
    assert detail["composite"]["模型A"]["score"] == 21.0
    assert detail["note"]
    # 列表
    items = client.get("/api/leaderboard").json()["leaderboards"]
    assert any(x["lb_id"] == lb_id for x in items)


def test_api_delete_leaderboard(client):
    """迭代十一：已有排行榜删除（DELETE /api/leaderboard/{lb_id}）。"""
    j1 = _make_job("j1", {"T1": 8.0, "T2": 7.0, "T3": 6.0, "T4": 5.0},
                   {"T1": 6.0, "T2": 5.0, "T3": 4.0, "T4": 9.0},
                   x="模型A", y="模型B")
    resp = _call(main_module.create_leaderboard, LeaderboardRequest(
        name="待删榜单", job_ids=[j1]))
    lb_id = resp["lb_id"]

    r = client.delete(f"/api/leaderboard/{lb_id}")
    assert r.status_code == 200
    assert r.json()["deleted"] is True
    assert storage.load_leaderboard(lb_id) is None
    items = client.get("/api/leaderboard").json()["leaderboards"]
    assert all(x["lb_id"] != lb_id for x in items)
    events = audit.read_events()
    assert any(e["event"] == "leaderboard_deleted" and e["target"] == lb_id
               for e in events)


def test_api_delete_leaderboard_missing_and_invalid(client):
    assert client.delete("/api/leaderboard/lb_20260101_000000_abcdef").status_code == 404
    # 路径穿越/非法格式：路由层规范化或 is_valid_lb_id 拒绝（均不删除任何文件）
    assert client.delete("/api/leaderboard/../evil").status_code in (400, 404)
    assert client.delete("/api/leaderboard/not-a-lb").status_code == 400


def test_api_rejects_unfinished_job(client):
    jid = _make_job("j1", {"T1": 8.0, "T2": 7.0, "T3": 6.0, "T4": 5.0},
                    {"T1": 6.0, "T2": 5.0, "T3": 4.0, "T4": 9.0},
                    with_report=False)
    with pytest.raises(HTTPException) as ei:
        _call(main_module.create_leaderboard,
              LeaderboardRequest(job_ids=[jid]))
    assert ei.value.status_code == 400
    assert "未完成" in str(ei.value.detail)


def test_api_rejects_inconsistent_dataset(client):
    j1 = _make_job("j1", {"T1": 8.0, "T2": 7.0, "T3": 6.0, "T4": 5.0},
                   {"T1": 6.0, "T2": 5.0, "T3": 4.0, "T4": 9.0})
    jid = storage.create_job_id()
    storage.save_config(jid, {"model_a": {"name": "A"}, "model_b": {"name": "B"}})
    storage.save_task_set(jid, {"meta": {"total": 1},
                                "tasks": [{"id": "X1", "dimension": "数学能力"}]})
    storage.save_verdict(jid, _verdict([{"id": "X1", "answer_x": 1.0,
                                         "answer_y": 2.0}]))
    storage.save_report(jid, {"report": {}})
    with pytest.raises(HTTPException) as ei:
        _call(main_module.create_leaderboard,
              LeaderboardRequest(job_ids=[j1, jid]))
    assert ei.value.status_code == 400
    assert "不一致" in str(ei.value.detail)


def test_api_detail_validation():
    with pytest.raises(HTTPException) as ei:
        _call(main_module.leaderboard_detail, "lb_bad")
    assert ei.value.status_code == 400
    with pytest.raises(HTTPException) as ei:
        _call(main_module.leaderboard_detail, "lb_00000000_000000_000000")
    assert ei.value.status_code == 404


def test_api_dashboard(client):
    j1 = _make_job("j1", {"T1": 8.0, "T2": 7.0, "T3": 6.0, "T4": 5.0},
                   {"T1": 6.0, "T2": 5.0, "T3": 4.0, "T4": 9.0})
    resp = client.get("/api/dashboard")
    assert resp.status_code == 200
    body = resp.json()
    assert "hw" not in body  # 迭代十一：硬件利用率已移除
    # mock job 的 report 无 kpi → 入列但字段 N/A（历史记录空态）
    trend0 = {x["job_id"]: x for x in body["jobs_trend"]}
    assert trend0[j1]["duration_sec"] is None
    assert trend0[j1]["total_tokens"] is None
    # 落盘完整 report 后 kpi 生效
    report = {"config": {}, "tasks": _task_set(), "answers_a": {},
              "answers_b": {}, "verdict": {},
              "report": {"kpi": {"duration_sec": 3.0,
                                 "total_tokens": {"x": 7, "y": 8}}}}
    storage.save_report(j1, report)
    body = client.get("/api/dashboard").json()
    trend = {x["job_id"]: x for x in body["jobs_trend"]}
    assert trend[j1]["duration_sec"] == 3.0
    assert trend[j1]["total_tokens"] == 15
