# -*- coding: utf-8 -*-
"""金标集（迭代三）单测：存储 CRUD/demo 自动初始化/manual 覆盖、
meta-eval 计算（Spearman/Kappa/偏移）、不匹配空态、API 端点。"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend import storage
from backend.gold import _demo_items, compute_meta_eval, ensure_demo_gold
from backend.main import app


@pytest.fixture()
def empty_gold_dir(tmp_path, monkeypatch):
    """把 GOLD_DIR 重定向到空临时目录（conftest 模块级重定向之上再覆盖）。"""
    monkeypatch.setattr(storage, "GOLD_DIR", tmp_path / "gold")
    return tmp_path / "gold"


def _task_set() -> dict:
    return {"meta": {"total": 2}, "tasks": [
        {"id": "T1", "dimension": "数学能力", "type": "生成式"},
        {"id": "T2", "dimension": "语言能力", "type": "生成式"},
    ]}


def _verdict(scores: list[dict], revealed: dict | None = None) -> dict:
    return {
        "meta": {"total": len(scores), "valid": len(scores), "invalid": 0,
                 "repeat_n": 1, "excluded_ids": [], "excluded_dimensions": []},
        "scores": scores,
        "per_dimension": {},
        "totals": {"answer_x": 0, "answer_y": 0},
        "revealed": revealed or {"answer_x": "模型A", "answer_y": "模型B",
                                 "answer_x_file": "a", "answer_y_file": "b"},
        "conclusion": "", "winner_model": "tie",
    }


def _gold() -> dict:
    return {"name": "demo", "source": "demo", "items": [
        {"task_id": "T1", "model_name": "模型A", "score": 90.0},
        {"task_id": "T1", "model_name": "模型B", "score": 70.0},
        {"task_id": "T2", "model_name": "模型A", "score": 80.0},
        {"task_id": "T2", "model_name": "模型B", "score": 60.0},
        {"task_id": "T1", "model_name": "无关模型", "score": 50.0},
    ]}


# ---- 存储 ----

def test_save_and_load_gold_roundtrip(empty_gold_dir):
    storage.save_gold("g1", {"items": [{"task_id": "T1", "model_name": "M",
                                        "score": 80.0}], "source": "manual"})
    g = storage.load_gold("g1")
    assert g["source"] == "manual"
    assert g["items"][0]["score"] == 80.0


def test_load_gold_missing(empty_gold_dir):
    assert storage.load_gold("nope") is None


def test_list_gold_summary(empty_gold_dir):
    storage.save_gold("a", {"items": [], "source": "demo"})
    storage.save_gold("b", {"items": [{"task_id": "T1", "model_name": "M",
                                       "score": 1.0}], "source": "manual"})
    names = {g["name"]: g for g in storage.list_gold()}
    assert set(names) == {"a", "b"}
    assert names["a"]["source"] == "demo"
    assert names["b"]["item_count"] == 1


def test_delete_gold(empty_gold_dir):
    storage.save_gold("x", {"items": [], "source": "manual"})
    assert storage.delete_gold("x") is True
    assert storage.delete_gold("x") is False


# ---- demo 初始化 ----

def test_ensure_demo_gold_when_empty(empty_gold_dir):
    ensure_demo_gold()
    demo = storage.load_gold("demo")
    assert demo is not None
    assert demo["source"] == "demo"
    assert len(demo["items"]) == 8  # 4 题 × 2 模型
    note = demo["items"][0]["note"]
    assert "演示" in note


def test_ensure_demo_gold_skips_when_present(empty_gold_dir):
    storage.save_gold("mine", {"items": [], "source": "manual"})
    ensure_demo_gold()
    assert storage.load_gold("demo") is None  # 已有金标不再自动种 demo


def test_manual_overrides_demo(empty_gold_dir):
    ensure_demo_gold()
    storage.save_gold("demo", {"items": [], "source": "manual"})
    assert storage.load_gold("demo")["source"] == "manual"


# ---- meta-eval 计算 ----

def test_meta_eval_happy_path():
    gold = {"name": "demo", "source": "demo", "items": [
        {"task_id": "T1", "model_name": "模型A", "score": 90.0},
        {"task_id": "T1", "model_name": "模型B", "score": 70.0},
        {"task_id": "T2", "model_name": "模型A", "score": 75.0},
        {"task_id": "T2", "model_name": "模型B", "score": 75.0},
    ]}
    v = _verdict([
        {"id": "T1", "dimension": "数学能力",
         "model_a": 9.0, "model_b": 7.0},   # → model_a
        {"id": "T2", "dimension": "语言能力",
         "model_a": 7.5, "model_b": 7.5},   # → tie
    ])
    m = compute_meta_eval(v, _task_set(), gold)
    assert m["available"] is True
    assert m["matched"] == 2
    assert m["gold_source"] == "demo"
    # 评审对（90/70/75/75）与金标对（90/70/75/75）完全同序 → Spearman=1
    assert m["spearman"] == 1.0
    assert m["kappa"] == pytest.approx(1.0, abs=1e-4)  # winner 全部一致且含 tie 类别
    assert m["gold_offset"] == pytest.approx(0.0, abs=1e-4)


def test_meta_eval_offset_positive_when_judge_low():
    gold = _gold()
    v = _verdict([
        {"id": "T1", "dimension": "数学能力",
         "model_a": 6.0, "model_b": 5.0},
        {"id": "T2", "dimension": "语言能力",
         "model_a": 5.0, "model_b": 4.0},
    ])
    m = compute_meta_eval(v, _task_set(), gold)
    # 评审（映射 ×10）=60/50/50/40；金标=90/70/80/60 → 偏移 = mean(75) - mean(50) = +25
    assert m["gold_offset"] == pytest.approx(25.0, abs=1e-4)


def test_meta_eval_single_round_field_fallback():
    """单轮 verdict（无 model_a/model_b）回退 answer_x/answer_y + reveal 归一化。"""
    gold = _gold()
    v = _verdict(
        [{"id": "T1", "dimension": "数学能力", "answer_x": 7.0, "answer_y": 9.0},
         {"id": "T2", "dimension": "语言能力", "answer_x": 6.0, "answer_y": 8.0}],
        revealed={"answer_x": "模型A", "answer_y": "模型B",
                  "answer_x_file": "b", "answer_y_file": "a"},  # X 实为 B
    )
    m = compute_meta_eval(v, _task_set(), gold)
    # 按 reveal：model_a(=A)=answer_y=9/8（×10=90/80），与金标一致 → 偏移 0
    assert m["available"] is True
    assert m["gold_offset"] == pytest.approx(0.0, abs=1e-4)


def test_meta_eval_unmatched_returns_empty_state():
    gold = _gold()
    v = _verdict([{"id": "OTHER", "dimension": "数学能力",
                   "model_a": 8.0, "model_b": 6.0}])
    m = compute_meta_eval(v, _task_set(), gold)
    assert m["available"] is False
    assert m["matched"] == 0
    assert m["spearman"] is None and m["kappa"] is None
    assert "不匹配" in m["note"]


def test_meta_eval_no_gold_items():
    v = _verdict([{"id": "T1", "dimension": "数学能力",
                   "model_a": 8.0, "model_b": 6.0}])
    m = compute_meta_eval(v, _task_set(), {"items": [], "source": "manual"})
    assert m["available"] is False


# ---- API 端点 ----

def test_gold_api_flow(empty_gold_dir, monkeypatch):
    # lifespan 在 TestClient 上下文里执行（demo 种子被空目录重定向覆盖）
    with TestClient(app) as client:
        r = client.get("/api/gold")
        assert r.status_code == 200
        names = {g["name"]: g for g in r.json()["gold"]}
        assert names["demo"]["source"] == "demo"

        r = client.post("/api/gold", json={
            "name": "demo",  # manual 覆盖 demo
            "items": [{"task_id": "T1", "model_name": "M", "score": 88.0,
                       "note": "人工录入"}],
        })
        assert r.status_code == 200
        g = storage.load_gold("demo")
        assert g["source"] == "manual"
        assert g["items"][0]["score"] == 88.0

        r = client.get("/api/gold")
        assert r.json()["gold"][0]["source"] == "manual"

        r = client.delete("/api/gold/demo")
        assert r.status_code == 200
        assert storage.load_gold("demo") is None


def test_meta_eval_endpoint_404_unfinished_job(empty_gold_dir, monkeypatch):
    with TestClient(app) as client:
        client.post("/api/gold", json={"name": "g", "items": [
            {"task_id": "T1", "model_name": "M", "score": 80.0}]})
        r = client.get("/api/gold/g/meta-eval", params={"job_id": "bad_id"})
        assert r.status_code == 400


def test_meta_eval_endpoint_completed_job(empty_gold_dir, monkeypatch):
    from backend import main as main_module

    with TestClient(app) as client:
        client.post("/api/gold", json={"name": "g", "items": [
            {"task_id": "T1", "model_name": "模型A", "score": 90.0},
            {"task_id": "T1", "model_name": "模型B", "score": 70.0},
            {"task_id": "T2", "model_name": "模型A", "score": 70.0},
            {"task_id": "T2", "model_name": "模型B", "score": 70.0}]})
        # 手工造一个"已完成" job（config + tasks + verdict）
        jid = storage.create_job_id()
        storage.save_config(jid, {"model_a": {"name": "模型A", "url": "https://x",
                                              "key": "k"},
                                  "model_b": {"name": "模型B", "url": "https://x",
                                              "key": "k"},
                                  "repeat_n": 1})
        storage.save_task_set(jid, _task_set())
        storage.save_verdict(jid, _verdict([
            {"id": "T1", "dimension": "数学能力",
             "model_a": 9.0, "model_b": 7.0},
            {"id": "T2", "dimension": "语言能力",
             "model_a": 7.0, "model_b": 7.0},
        ]))
        # 未完成任务（有 config 无 verdict）→ 404
        jid2 = storage.create_job_id()
        storage.save_config(jid2, {"model_a": {"name": "A", "url": "https://x",
                                               "key": "k"},
                                   "model_b": {"name": "B", "url": "https://x",
                                               "key": "k"}})
        storage.save_task_set(jid2, _task_set())

        r = client.get("/api/gold/g/meta-eval", params={"job_id": jid2})
        assert r.status_code == 404

        r = client.get("/api/gold/g/meta-eval", params={"job_id": jid})
        assert r.status_code == 200
        m = r.json()["meta_eval"]
        assert m["available"] is True and m["matched"] == 2
        assert m["gold_source"] == "manual"
        assert m["spearman"] == 1.0
        assert m["kappa"] == pytest.approx(1.0, abs=1e-4)
        assert m["gold_offset"] == pytest.approx(0.0, abs=1e-4)