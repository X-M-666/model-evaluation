# -*- coding: utf-8 -*-
"""报告扩展（迭代三）单测：meta_eval 段（金标传入/空态）、consistency 段
（repeat_n≥2 一致率、单轮 null）、review 段（mode/degraded/k_top_human）。"""
from __future__ import annotations

from backend.engine.report_builder import _build_consistency, _build_meta_eval, build_report

CONFIG = {"model_a": {"name": "A"}, "model_b": {"name": "B"},
          "review": {"mode": "hybrid", "k_top_human": 3}, "repeat_n": 2}
TASKS = {"meta": {"total": 2}, "tasks": [
    {"id": "T1", "dimension": "数学能力", "type": "判别式", "prompt": "1+1=?",
     "test_cases": [{"input": "x", "expected": "2"}], "rubric_note": "r",
     "expected": "2"},
    {"id": "T2", "dimension": "数学能力", "type": "判别式", "prompt": "2+2=?",
     "test_cases": [{"input": "x", "expected": "4"}], "rubric_note": "r",
     "expected": "4"},
]}


def _verdict(round_reveals: list[dict] | None = None) -> dict:
    return {
        "meta": {"total": 2, "valid": 2, "invalid": 0, "repeat_n": 2,
                 "excluded_ids": [], "excluded_dimensions": [],
                 "round_reveals": round_reveals or [
                     {"answer_x_file": "a", "answer_y_file": "b"}] * 2},
        "scores": [
            {"id": "T1", "dimension": "数学能力", "model_a": 8.0, "model_b": 6.0},
            {"id": "T2", "dimension": "数学能力", "model_a": 7.0, "model_b": 7.0},
        ],
        "per_dimension": {"数学能力": {"a": 15.0, "b": 13.0}},
        "totals": {"answer_x": 15.0, "answer_y": 13.0},
        "revealed": {"answer_x": "A", "answer_y": "B",
                     "answer_x_file": "a", "answer_y_file": "b"},
        "conclusion": "", "winner_model": "A",
    }


def _answers(label: str) -> dict:
    return {"model": label, "answers": [
        {"id": "T1", "raw_answer": "2", "api_info": {}},
        {"id": "T2", "raw_answer": "4", "api_info": {}},
    ]}


def _gold() -> dict:
    return {"name": "demo", "source": "demo", "items": [
        {"task_id": "T1", "model_name": "A", "score": 80.0},
        {"task_id": "T1", "model_name": "B", "score": 60.0},
        {"task_id": "T2", "model_name": "A", "score": 70.0},
        {"task_id": "T2", "model_name": "B", "score": 70.0},
    ]}


def _round_verdict(x_file_a: bool, scores: list[tuple]) -> dict:
    """构造逐轮 verdict（轮级 reveal 固定为 x_file）。"""
    rows = []
    for tid, x, y in scores:
        rows.append({"id": tid, "dimension": "数学能力",
                     "answer_x": x, "answer_y": y, "winner": "",
                     "_invalid": False})
    return {
        "meta": {"total": 2, "valid": 2, "invalid": 0, "excluded_ids": [],
                 "excluded_dimensions": []},
        "scores": rows,
        "per_dimension": {}, "totals": {},
        "revealed": {"answer_x": "A", "answer_y": "B",
                     "answer_x_file": "a" if x_file_a else "b",
                     "answer_y_file": "b" if x_file_a else "a"},
        "conclusion": "", "winner_model": "",
    }


def test_build_meta_eval_with_gold():
    m = _build_meta_eval(_gold(), _verdict(), TASKS)
    assert m["available"] is True
    assert m["matched"] == 2
    assert m["gold_source"] == "demo"
    assert m["spearman"] == 1.0


def test_build_meta_eval_without_gold_empty_state():
    m = _build_meta_eval(None, _verdict(), TASKS)
    assert m["available"] is False
    assert "未配置金标集" in m["note"]


def test_build_consistency_multi_round_stable_winner():
    """T1 两轮 winner 稳定为 model_a；T2 一轮 tie 一轮 model_a → 一致率 0。"""
    rvs = [
        _round_verdict(True, [("T1", 8, 6), ("T2", 7, 7)]),
        _round_verdict(True, [("T1", 8, 6), ("T2", 8, 7)]),
    ]
    c = _build_consistency(rvs, 2)
    assert c is not None
    assert c["repeat_n"] == 2
    assert c["per_task"]["T1"]["rate"] == 1.0  # 两轮一致
    assert c["per_task"]["T2"]["rate"] == 0.0  # 一轮 tie 一轮 model_a


def test_build_consistency_single_round_none():
    assert _build_consistency([_round_verdict(True, [("T1", 8, 6)])], 1) is None
    assert _build_consistency(None, 2) is None


def test_build_consistency_reveal_swap_stable_space():
    """轮级 reveal 互换时 winner 在稳定空间一致（M4 口径）。"""
    rvs = [
        _round_verdict(True, [("T1", 8, 6)]),   # X=a: A 8, B 6 → A 胜
        _round_verdict(False, [("T1", 6, 8)]),  # X=b: A 8(B), B 6(A) → 稳定 A 胜
    ]
    c = _build_consistency(rvs, 2)
    assert c["per_task"]["T1"]["rate"] == 1.0


def test_build_report_includes_new_sections():
    r = build_report(CONFIG, TASKS, _answers("A"), _answers("B"), _verdict(),
                     gold=_gold(),
                     round_verdicts=[
                         _round_verdict(True, [("T1", 8, 6), ("T2", 7, 7)]),
                         _round_verdict(True, [("T1", 8, 6), ("T2", 8, 7)]),
                     ])
    assert r["judge_mode"] == "hybrid"
    assert r["review"]["mode"] == "hybrid"
    assert r["review"]["degraded"] is False
    assert r["review"]["k_top_human"] == 3
    assert r["meta_eval"]["available"] is True
    assert r["consistency"]["per_task"]["T1"]["rate"] == 1.0
    # 既有字段不破坏
    assert r["summary"]["repeat_n"] == 2
    assert r["prompt_strategy"] == "cot"


def test_build_report_review_degraded_flag():
    cfg = dict(CONFIG)
    cfg["review"] = {"mode": "pure_human", "degraded": True}
    r = build_report(cfg, TASKS, _answers("A"), _answers("B"), _verdict())
    assert r["review"]["degraded"] is True
    assert r["judge_mode"] == "pure_human"


def test_consistency_uses_per_task_reveal_alignment():
    """迭代三：agent 逐题独立交换落盘 per_task 时，一致率按题归一化到稳定空间。"""
    # r1：视角分 X/Y 对应 a/b 文件（T1 X=a 8分→A、T2 X=a 6分→B 胜）
    r1 = _round_verdict(True, [("T1", 8, 6), ("T2", 6, 8)])
    # r2：T1 交换（X=b → A 得 Y 分 8），T2 未交换（X=a）——稳定空间两题与 r1 同胜者
    r2 = _round_verdict(True, [("T1", 6, 8), ("T2", 6, 8)])
    r2["revealed"]["per_task"] = {"T1": "b", "T2": "a"}
    c = _build_consistency([r1, r2], 2)
    # 按题对齐稳定空间：T1 两轮 A 均 8 分 → model_a 胜，一致 1.0
    assert c["per_task"]["T1"]["rate"] == 1.0
    # T2 两轮 A 6 / B 8 → model_b 胜，一致 1.0
    assert c["per_task"]["T2"]["rate"] == 1.0
    assert c["overall"] == 1.0
    # 对照组：r2 的 per_task 缺失时回退轮级 x_file=a → T1 第二轮被误判为 A 6 分
    #（实为 B 8 分）→ model_b 胜，与 r1 分歧，一致率 0
    r2b = _round_verdict(True, [("T1", 6, 8), ("T2", 6, 8)])
    c2 = _build_consistency([r1, r2b], 2)
    assert c2["per_task"]["T1"]["rate"] == 0.0
    assert c2["per_task"]["T2"]["rate"] == 1.0