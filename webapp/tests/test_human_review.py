# -*- coding: utf-8 -*-
"""人工双盲评审多轮聚合回归测试（issue #2）。

核心验证：每轮 X/Y 身份独立随机，跨轮统计必须以稳定模型
（model_a/model_b）为主键；X/Y 仅作最终固定展示映射。
"""
from __future__ import annotations

import statistics

import pytest

from backend.engine.human_review import (
    build_final_verdict,
    build_round_verdict,
)
from backend.engine.report_builder import build_report

MODEL_A = "模型A"
MODEL_B = "模型B"

TASK_SET = {
    "tasks": [
        {"id": "t1", "dimension": "知识"},
        {"id": "t2", "dimension": "推理"},
    ]
}

# 每轮真实 A/B 分数保持不变（t1/t2 两道题）
REAL_ROUNDS = [
    ({"a": 10.0, "b": 0.0}, {"a": 6.0, "b": 4.0}),  # 轮1
    ({"a": 10.0, "b": 0.0}, {"a": 6.0, "b": 4.0}),  # 轮2
]


def _round_verdict(mapping: tuple[str, str], round_idx: int, real: list[dict] | None = None):
    """按 mapping（X 文件, Y 文件）构造一轮 verdict；real 为真实 A/B 逐题分。"""
    x_file, y_file = mapping
    if real is None:
        real = REAL_ROUNDS[round_idx]
    scores = [
        {"id": "t1", "answer_x": real[0][x_file], "answer_y": real[0][y_file], "note": ""},
        {"id": "t2", "answer_x": real[1][x_file], "answer_y": real[1][y_file], "note": ""},
    ]
    x_model = MODEL_A if x_file == "a" else MODEL_B
    y_model = MODEL_B if y_file == "b" else MODEL_A
    return build_round_verdict(
        TASK_SET, scores, {"answer_x": x_file, "answer_y": y_file},
        x_model, y_model, round_idx,
    )


def _aggregate(mappings: list[tuple[str, str]]) -> dict:
    round_verdicts = [
        _round_verdict(mapping, idx) for idx, mapping in enumerate(mappings)
    ]
    return build_final_verdict(round_verdicts, len(mappings))


@pytest.mark.parametrize("mappings", [
    [("a", "b"), ("a", "b")],  # 所有轮次均为 X=A
    [("b", "a"), ("b", "a")],  # 所有轮次均为 X=B
    [("a", "b"), ("b", "a")],  # X=A 与 X=B 交替
    [("b", "a"), ("a", "b")],  # X=A 与 X=B 交替（反向）
])
def test_stable_aggregation_identical_under_mapping_permutation(mappings):
    """真实 A/B 分数不变时，无论 X/Y 映射如何变化，聚合结果必须一致。"""
    verdict = _aggregate(mappings)
    by_id = {s["id"]: s for s in verdict["scores"]}

    # t1: A=10, B=0 → 均值 10/0、跨轮标准差 0
    t1 = by_id["t1"]
    assert t1["model_a"] == 10.0
    assert t1["model_b"] == 0.0
    assert t1["model_a_std"] == 0
    assert t1["model_b_std"] == 0
    assert t1["winner"] == "answer_x" if mappings[-1][0] == "a" else t1["winner"] == "answer_y"

    # t2: A=6, B=4 → 均值 6/4
    t2 = by_id["t2"]
    assert t2["model_a"] == 6.0
    assert t2["model_b"] == 4.0

    # 总分与胜方以稳定模型判定，且与映射无关
    assert verdict["totals"]["answer_x"] + verdict["totals"]["answer_y"] == 20.0
    assert verdict["winner_model"] == MODEL_A


@pytest.mark.parametrize("mappings", [
    [("a", "b"), ("a", "b")],
    [("b", "a"), ("b", "a")],
    [("a", "b"), ("b", "a")],
])
def test_issue_repro_minimal(mappings):
    """issue #2 最小复现：单题两轮，真实 A=10、B=0，不得误判为平局。"""
    task_set = {"tasks": [{"id": "t1", "dimension": "知识"}]}
    scores = [{"id": "t1", "answer_x": 10.0, "answer_y": 0.0, "note": ""}]
    if mappings[0][0] == "b":
        scores = [{"id": "t1", "answer_x": 0.0, "answer_y": 10.0, "note": ""}]

    round_verdicts = []
    for idx, (x_file, y_file) in enumerate(mappings):
        x_model = MODEL_A if x_file == "a" else MODEL_B
        y_model = MODEL_B if y_file == "b" else MODEL_A
        sc = [{"id": "t1", "answer_x": 10.0, "answer_y": 0.0, "note": ""}]
        if x_file == "b":
            sc = [{"id": "t1", "answer_x": 0.0, "answer_y": 10.0, "note": ""}]
        round_verdicts.append(build_round_verdict(
            task_set, sc, {"answer_x": x_file, "answer_y": y_file},
            x_model, y_model, idx,
        ))

    verdict = build_final_verdict(round_verdicts, len(mappings))
    s = verdict["scores"][0]

    assert s["model_a"] == 10.0
    assert s["model_b"] == 0.0
    assert s["model_a_std"] == 0
    assert s["model_b_std"] == 0

    # 修复前：X=5、Y=5、std=7.07、平局；修复后：真实 A/B 为准
    assert verdict["winner_model"] == MODEL_A
    assert verdict["totals"]["answer_x"] + verdict["totals"]["answer_y"] == 10.0


def test_stable_aggregation_with_variance():
    """跨轮分数有波动时，标准差按稳定模型序列计算。"""
    round1 = [({"a": 10.0, "b": 0.0}, {"a": 6.0, "b": 4.0})]
    round2 = [({"a": 6.0, "b": 4.0}, {"a": 10.0, "b": 0.0})]
    mappings = [("a", "b"), ("b", "a")]

    round_verdicts = []
    for idx, ((x_file, y_file), real) in enumerate(zip(mappings, [round1[0], round2[0]])):
        round_verdicts.append(_round_verdict((x_file, y_file), idx, real))

    verdict = build_final_verdict(round_verdicts, 2)
    t1 = next(s for s in verdict["scores"] if s["id"] == "t1")

    assert t1["model_a"] == 8.0
    assert t1["model_b"] == 2.0
    assert t1["model_a_std"] == round(statistics.stdev([10.0, 6.0]), 2)
    assert t1["model_b_std"] == round(statistics.stdev([0.0, 4.0]), 2)


def test_report_consistent_with_stable_scores():
    """报告中的胜方/逐题/维度/稳定性数据与归一化 A/B 一致。"""
    round1 = [({"a": 10.0, "b": 0.0}, {"a": 6.0, "b": 4.0})]
    round2 = [({"a": 6.0, "b": 4.0}, {"a": 10.0, "b": 0.0})]
    mappings = [("a", "b"), ("b", "a")]

    round_verdicts = []
    for idx, ((x_file, y_file), real) in enumerate(zip(mappings, [round1[0], round2[0]])):
        round_verdicts.append(_round_verdict((x_file, y_file), idx, real))
    verdict = build_final_verdict(round_verdicts, 2)

    def _answers(model_name: str, t1: float, t2: float) -> dict:
        return {
            "model": model_name,
            "answers": [
                {"id": "t1", "raw_answer": f"answer-{model_name}-1", "api_info": {
                    "status": "ok", "latency_ms": 10, "prompt_tokens": 1,
                    "completion_tokens": 2}},
                {"id": "t2", "raw_answer": f"answer-{model_name}-2", "api_info": {
                    "status": "ok", "latency_ms": 20, "prompt_tokens": 3,
                    "completion_tokens": 4}},
            ],
        }

    answers_a = _answers(MODEL_A, "a1", "a2")
    answers_b = _answers(MODEL_B, "b1", "b2")
    report = build_report({"repeat_n": 2}, TASK_SET, answers_a, answers_b, verdict)
    summary = report["summary"]

    # 最后一轮映射为 X=B、Y=A：展示上 X 为模型B、Y 为模型A
    assert summary["x_model"] == MODEL_B
    assert summary["y_model"] == MODEL_A
    assert summary["winner_model"] == MODEL_A
    assert summary["winner"] == "answer_y"

    # 稳定分：A 总分 16、B 总分 4；展示投影后 answer_y(A)=16
    assert summary["total_x"] == 4.0
    assert summary["total_y"] == 16.0
    assert summary["win_x"] == 0
    assert summary["win_y"] == 2
    assert summary["ties"] == 0

    # 逐题分与标准差与归一化 A/B 一致
    sby = report["charts"]["score_by_question"]
    rows = {
        tid: {"x": sby["x"][i], "y": sby["y"][i]}
        for i, tid in enumerate(sby["categories"])
    }
    assert rows["t1"]["x"] == 2.0 and rows["t1"]["y"] == 8.0
    assert rows["t2"]["x"] == 2.0 and rows["t2"]["y"] == 8.0
    stability = report["charts"]["stability"]
    assert stability["x_std"] == [round(statistics.stdev([0.0, 4.0]), 2),
                                  round(statistics.stdev([4.0, 0.0]), 2)]
    assert stability["y_std"] == [round(statistics.stdev([10.0, 6.0]), 2),
                                  round(statistics.stdev([6.0, 10.0]), 2)]

    # 维度聚合与稳定分一致
    score_by_dim = report["charts"]["score_by_dimension"]
    dims = dict(zip(score_by_dim["dimensions"], zip(score_by_dim["x"], score_by_dim["y"])))
    assert dims["知识"] == (2.0, 8.0)
    assert dims["推理"] == (2.0, 8.0)


def test_single_round_unchanged():
    """单轮（repeat_n=1）行为与修复前一致：直接返回该轮 verdict。"""
    v = _round_verdict(("a", "b"), 0)
    verdict = build_final_verdict([v], 1)
    assert verdict is v
    assert verdict["meta"]["repeat_n"] == 1
    assert verdict["winner_model"] == MODEL_A
    assert verdict["totals"] == {"answer_x": 16.0, "answer_y": 4.0}
