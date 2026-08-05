# -*- coding: utf-8 -*-
"""报告构建器空数据/异常状态回归测试（issue #11）。

核心验证：report_builder 在空数据、缺失字段、答卷不完整等异常输入下
必须保持稳定输出（不抛异常、结构完整、聚合口径合理），而非崩溃。
"""
from __future__ import annotations

import pytest

from backend.engine.report_builder import (
    _round_reveals,
    build_report,
    reveal_answers,
)

MODEL_A = "模型A"
MODEL_B = "模型B"

TASK_SET = {
    "tasks": [
        {"id": "t1", "dimension": "知识"},
        {"id": "t2", "dimension": "代码能力"},
    ]
}

CFG = {
    "model_a": {"name": MODEL_A, "url": "https://example.com", "key": "k"},
    "model_b": {"name": MODEL_B, "url": "https://example.com", "key": "k"},
    "repeat_n": 2, "seed": 7, "dataset_name": None,
    "dims": None, "code_verify_mode": "mock",
}

# 两轮：轮1 X=A/Y=B，轮2 X=B/Y=A；round_reveals 与 revealed 一致
VERDICT = {
    "revealed": {
        "answer_x": MODEL_B, "answer_y": MODEL_A,
        "answer_x_file": "b", "answer_y_file": "a",
    },
    "meta": {
        "repeat_n": 2,
        "round_reveals": [
            {"answer_x_file": "a", "answer_y_file": "b"},
            {"answer_x_file": "b", "answer_y_file": "a"},
        ],
    },
    "totals": {"answer_x": 16.0, "answer_y": 4.0},
    "scores": [
        {"id": "t1", "winner": "answer_x"},
        {"id": "t2", "winner": "answer_x"},
    ],
}


def _round_answers(label: str, raw: str = "r") -> dict:
    """正常一轮答卷：每题一条 entry。"""
    return {
        "model": label,
        "answers": [
            {
                "id": t["id"],
                "raw_answer": f"{label}-{t['id']}-{raw}",
                "api_info": {
                    "status": "ok",
                    "latency_ms": 100,
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "repeat_index": 1,
                },
            }
            for t in TASK_SET["tasks"]
        ],
    }


def _rounds_a_b() -> list[dict]:
    return [
        {"a": _round_answers(MODEL_A, "r1"), "b": _round_answers(MODEL_B, "r1")},
        {"a": _round_answers(MODEL_A, "r2"), "b": _round_answers(MODEL_B, "r2")},
    ]


# ---- 空数据 / 异常状态 ----

def test_empty_tasks_and_scores_no_crash():
    """空任务集 + 空评分：报告可生成，计数为 0，均值/比率有守卫。"""
    report = build_report(CFG, {"tasks": []}, {}, {}, {})
    assert report["summary"]["total_questions"] == 0
    assert report["summary"]["avg_x"] == 0
    assert report["summary"]["avg_y"] == 0
    assert report["summary"]["score_ratio"] is None
    assert report["summary"]["max_score"] == 0
    assert report["summary"]["winner"] in ("answer_x", "answer_y", "tie")


def test_empty_verdict_fields_defaulted():
    """verdict 缺 revealed/totals/scores/meta：全部走默认值，不崩溃。"""
    report = build_report(CFG, TASK_SET, {}, {}, {})
    s = report["summary"]
    assert s["total_x"] == 0 and s["total_y"] == 0
    assert s["win_x"] == 0 and s["win_y"] == 0 and s["ties"] == 0
    assert s["repeat_n"] == 1
    assert s["x_model"] == "" and s["y_model"] == ""


def test_single_round_empty_answers_ok():
    """单轮路径：answers 为空 dict 时 _pick_entry 兜底为空对象。"""
    report = build_report(CFG, TASK_SET, {}, {}, VERDICT)
    assert report["summary"]["total_questions"] == 2
    rows = report["summary"]["total_x"]
    assert rows == 16.0


def test_multi_round_missing_task_in_last_round_no_crash():
    """多轮路径：最后一轮答卷缺某题（entry 列表为空）不得抛异常。"""
    rounds_answers = _rounds_a_b()
    rounds_answers[1]["a"]["answers"] = [
        e for e in rounds_answers[1]["a"]["answers"] if e["id"] != "t2"
    ]
    report = build_report(CFG, TASK_SET, rounds_answers[1]["a"], rounds_answers[1]["b"],
                          VERDICT, rounds_answers)
    assert report["summary"]["total_questions"] == 2


def test_multi_round_empty_round_no_crash():
    """多轮路径：某一轮整个模型答卷为空列表，不得抛异常。"""
    rounds_answers = _rounds_a_b()
    rounds_answers[0]["a"] = {"model": MODEL_A, "answers": []}
    report = build_report(CFG, TASK_SET, rounds_answers[1]["a"], rounds_answers[1]["b"],
                          VERDICT, rounds_answers)
    assert report["summary"]["total_questions"] == 2


def test_multi_round_missing_round_model_no_crash():
    """多轮路径：某一轮缺失模型答卷（key 不存在 → None），不得抛异常。"""
    rounds_answers = _rounds_a_b()
    del rounds_answers[1]["b"]
    report = build_report(CFG, TASK_SET, rounds_answers[1]["a"], {},
                          VERDICT, rounds_answers)
    assert report["summary"]["total_questions"] == 2


# ---- reveal 映射兜底 ----

def test_reveal_answers_defaults_to_a_b():
    x, y = reveal_answers(None, None, None)
    assert x is None and y is None

    answers_a, answers_b = {"m": "a"}, {"m": "b"}
    x, y = reveal_answers(answers_a, answers_b, None)
    assert x is answers_a and y is answers_b

    x, y = reveal_answers(answers_a, answers_b, {"revealed": {"answer_x_file": "b", "answer_y_file": "a"}})
    assert x is answers_b and y is answers_a


def test_round_reveals_fallback_when_verdict_old():
    """旧 verdict 无 round_reveals：用最后一轮 revealed 兜底全轮。"""
    verdict = {"revealed": {"answer_x_file": "b", "answer_y_file": "a"}}
    reveals = _round_reveals(verdict, 3)
    assert reveals == [{"answer_x_file": "b", "answer_y_file": "a"}] * 3


def test_round_reveals_prefers_meta_round_reveals():
    reveals = _round_reveals(VERDICT, 2)
    assert reveals[0] == {"answer_x_file": "a", "answer_y_file": "b"}
    assert reveals[1] == {"answer_x_file": "b", "answer_y_file": "a"}
