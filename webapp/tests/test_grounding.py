# -*- coding: utf-8 -*-
"""D4 grounding 指标（迭代四）：忠实度/相关性（纯 n-gram）+ 报告段。

- metric_grounding_faithfulness：答案 vs 参考文档 n-gram 余弦；
- metric_answer_relevancy：答案 vs 题面 n-gram 余弦；
- GROUNDING_SUPPORT_THRESHOLD=0.35：faithfulness 低于阈值 → grounded=False；
- 报告 metrics.grounding 段：仅当任务集存在带 context 的题目时出现；
- 无 context / 无答案：grounding 段缺省或 grounded=False + reason。
"""
from __future__ import annotations

import json
from pathlib import Path

from backend.engine.metrics import (
    GROUNDING_SUPPORT_THRESHOLD,
    METRICS, metric_grounding_faithfulness, metric_answer_relevancy,
)
from backend.engine.report_builder import _build_metrics

CTX = (
    "恒星中央核区通过氢聚变释放能量，其核心温度可达1500万开尔文。太阳属于黄矮星"
    "（G2V光谱型），表面温度约5500摄氏度，质量约为地球的33万倍。恒星在主序阶段的"
    "停留时间与其质量成反比。大质量恒星主序结束后经历红超巨星阶段，随后以超新星"
    "爆发结束生命。低质量恒星缓慢演化为红巨星，核心最终成为白矮星，白矮星依靠"
    "电子简并压支撑自身。"
)
PROMPT = "请仅依据参考文档回答：太阳的演化终点是什么？"


def _task(**kw):
    base = {"id": "T1", "dimension": "知识能力", "prompt": PROMPT, "expected": "",
            "rubric_note": "满分10分", "type": "生成式", "context": CTX}
    base.update(kw)
    return base


def _entry(raw, status="ok"):
    return {"id": "T1", "raw_answer": raw,
            "api_info": {"status": status, "latency_ms": 1, "prompt_tokens": 1,
                         "completion_tokens": 1, "repeat_index": 1}}


def test_metrics_registered():
    assert "grounding_faithfulness" in METRICS
    assert "answer_relevancy" in METRICS


def test_faithfulness_high_when_quoting_doc():
    ans = CTX[:60]  # 直接引用文档开头
    f = metric_grounding_faithfulness(ans, CTX)
    assert f >= GROUNDING_SUPPORT_THRESHOLD
    assert 0.0 <= f <= 1.0


def test_faithfulness_low_when_unrelated():
    f = metric_grounding_faithfulness("今天天气不错，我们去公园散步吧", CTX)
    assert f < GROUNDING_SUPPORT_THRESHOLD


def test_faithfulness_empty_handling():
    assert metric_grounding_faithfulness("", CTX) == 0.0
    assert metric_grounding_faithfulness("有内容", "") == 0.0


def test_answer_relevancy_reflects_question_overlap():
    rel_high = metric_answer_relevancy("太阳的演化终点是白矮星。", PROMPT)
    rel_low = metric_answer_relevancy("今天天气不错。", PROMPT)
    assert rel_high > rel_low


def _build_metrics_ctx(answers):
    return _build_metrics(
        task_set={"tasks": [_task()]},
        rounds_answers=None,
        answers_a={"answers": [_entry(answers[0])]},
        answers_b={"answers": [_entry(answers[1])]},
        verdict={"revealed": {"answer_x_file": "a", "answer_y_file": "b"}, "scores": []},
        embedding_config=None,
    )


def test_report_grounding_section_present_with_context():
    m = _build_metrics_ctx([CTX[:60], "天气不错，无关回答"])
    assert m["grounding"]["context_tasks"] == 1
    assert m["grounding"]["grounded_x"] == 1
    assert m["grounding"]["grounded_y"] == 0
    assert m["grounding"]["threshold"] == GROUNDING_SUPPORT_THRESHOLD
    pt = m["per_task"][0]
    assert pt["grounding"]["x"]["grounded"] is True
    assert pt["grounding"]["y"]["grounded"] is False
    assert pt["grounding"]["x"]["faithfulness"] >= GROUNDING_SUPPORT_THRESHOLD


def test_report_no_grounding_without_context():
    m = _build_metrics_ctx.__wrapped__ if hasattr(_build_metrics_ctx, "__wrapped__") else None
    out = _build_metrics(
        task_set={"tasks": [_task(context="")]},
        rounds_answers=None,
        answers_a={"answers": [_entry(CTX[:60])]},
        answers_b={"answers": [_entry(CTX[:60])]},
        verdict={"revealed": {"answer_x_file": "a", "answer_y_file": "b"}, "scores": []},
        embedding_config=None,
    )
    assert "grounding" not in out
    assert "grounding" not in out["per_task"][0]


def test_report_grounding_no_answer_side():
    m = _build_metrics_ctx([CTX[:60], ""])
    assert m["grounding"]["grounded_y"] == 0
    assert m["per_task"][0]["grounding"]["y"] == {
        "faithfulness": None, "answer_relevancy": None,
        "grounded": False, "reason": "no_answer"}


def test_grounding_values_serializable():
    m = _build_metrics_ctx([CTX[:60], CTX[:30]])
    json.dumps(m, ensure_ascii=False)  # 报告层需可 JSON 序列化


def test_grounding_persist_shape_used_in_packet():
    """build_report 打包路径兼容：grounding 段随 metrics 进 report.json。"""
    from backend.engine.report_builder import build_report
    task_set = {"name": "d", "tasks": [_task()]}
    verdict = {
        "scores": [{"id": "T1", "dimension": "知识能力", "answer_x": 8.0, "answer_y": 7.0}],
        "revealed": {"answer_x_file": "a", "answer_y_file": "b",
                     "answer_x": "M1", "answer_y": "M2", "per_task": {"T1": "a"}},
        "meta": {"total": 1, "valid": 1, "invalid": 0, "repeat_n": 1},
    }
    config = {
        "model_a": {"name": "M1", "url": "https://8.8.8.8/v1"},
        "model_b": {"name": "M2", "url": "https://8.8.8.8/v1"},
        "prompt_strategy": "cot",
        "review": {"mode": "pure_agent"},
    }
    ans_a = {"model": "M1", "answers": [_entry(CTX[:60])]}
    ans_b = {"model": "M2", "answers": [_entry("无关")]}
    r = build_report(config, task_set, ans_a, ans_b, verdict,
                     [{"a": ans_a, "b": ans_b}], embedding_config=None)
    assert r["metrics"]["grounding"]["grounded_x"] == 1
    assert r["metrics"]["grounding"]["grounded_y"] == 0