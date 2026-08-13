# -*- coding: utf-8 -*-
"""指标引擎单元测试（迭代二）：全部指标纯函数 + 注册表路由 + 边界语义。

覆盖：BLEU/ROUGE 已知值、判别式多子题比对、数值容差、生成式语义相似度
（离线 n-gram 兜底）、consistency、代码题 N/A、截断/失败跳算。
"""
from __future__ import annotations

import pytest

from backend.engine.metrics import (
    METRICS,
    bleu_score,
    compute_task_metrics,
    rouge_l,
)

DISC_TASK = {
    "id": "T1", "dimension": "知识能力", "type": "判别式",
    "prompt": "三道单选",
    "test_cases": [
        {"input": "Q1 答案？", "expected": "A（木星）"},
        {"input": "Q2 答案？", "expected": "A（ATP）"},
        {"input": "Q3 答案？", "expected": "A（苏必利尔湖）"},
    ],
}
GEN_TASK = {
    "id": "G1", "dimension": "语言能力", "type": "生成式",
    "prompt": "翻译", "expected": "好的产品需要用心打磨，没有捷径。",
}
CODE_TASK = {
    "id": "C1", "dimension": "代码能力", "type": "判别式", "prompt": "写函数",
    "test_cases": [{"input": "f()", "expected": "1"}],
}


def _entry(raw, status="ok", truncated=False, error=None, semantic=None):
    return {
        "id": "x", "raw_answer": raw,
        "api_info": {"status": status, "truncated": truncated, "error": error},
        **({"semantic": semantic} if semantic else {}),
    }


# ---- 注册表 ----

def test_registry_contains_all_metrics():
    for name in ("top1", "exact_match", "f1", "relaxed_accuracy",
                 "semantic_sim", "rubric_score", "consistency", "bleu", "rouge_l"):
        assert name in METRICS


# ---- BLEU / ROUGE（已知值） ----

def test_bleu_identical_is_one():
    assert bleu_score("the cat sat on the mat", "the cat sat on the mat") == 1.0


def test_bleu_empty_pred_is_zero():
    assert bleu_score("", "the cat sat on the mat") == 0.0
    assert bleu_score("", "") == 0.0


def test_bleu_disjoint_is_zero():
    assert bleu_score("zzz yyy xxx", "the cat sat on the mat") == 0.0


def test_bleu_half_overlap_between_zero_and_one():
    v = bleu_score("the cat sat on the mat", "the cat sat on the sofa")
    assert 0.0 < v < 1.0


def test_rouge_identical_one_disjoint_zero():
    assert rouge_l("a b c d", "a b c d") == 1.0
    assert rouge_l("w x y z", "a b c d") == 0.0
    assert rouge_l("", "a b c d") == 0.0


def test_rouge_partial():
    v = rouge_l("the cat sat", "the cat sat on the mat")
    assert 0.0 < v < 1.0


# ---- 判别式 ----

def test_discriminative_full_match():
    ans = _entry("Q1-A 木星\nQ2-A ATP\nQ3-A 苏必利尔湖")
    m = compute_task_metrics(DISC_TASK, [ans])
    assert m["top1"] == 1.0
    assert m["exact_match"] == 1.0
    assert m["f1"] is not None and 0.0 < m["f1"] < 1.0


def test_discriminative_partial():
    ans = _entry("Q1-A 木星\nQ2-B 其他\nQ3-C 其他")
    m = compute_task_metrics(DISC_TASK, [ans])
    assert m["top1"] == pytest.approx(1 / 3, abs=0.001)
    assert m["exact_match"] == 0.0


def test_relaxed_accuracy_numeric():
    task = {"id": "N1", "dimension": "数学能力", "type": "判别式",
            "test_cases": [{"input": "总和", "expected": "44440"}]}
    assert compute_task_metrics(task, [_entry("结论：44439")])["relaxed_accuracy"] == 1.0
    assert compute_task_metrics(task, [_entry("结论：12345")])["relaxed_accuracy"] == 0.0
    # 非数值期望 → relaxed 不适用
    assert compute_task_metrics(DISC_TASK, [_entry("Q1-A\nQ2-A\nQ3-A")])["relaxed_accuracy"] is None


def test_discriminative_no_expected_skipped():
    task = {"id": "X1", "dimension": "知识能力", "type": "判别式", "prompt": "p"}
    m = compute_task_metrics(task, [_entry("随便")])
    assert m["skipped"] is True
    assert m["reason"] == "no_expected"


# ---- 生成式 ----

def test_generative_semantic_similar_reference():
    near = _entry("好的产品需要用心打磨，没有捷径。")
    m = compute_task_metrics(GEN_TASK, [near])
    assert m["semantic_sim"] is not None and m["semantic_sim"] > 0.5
    assert m["bleu"] is not None and m["rouge_l"] is not None


def test_generative_semantic_unrelated_lower():
    far = _entry("今天天气不错，适合去公园散步。")
    near = _entry("好的产品需要用心打磨，没有捷径。")
    m_far = compute_task_metrics(GEN_TASK, [far])
    m_near = compute_task_metrics(GEN_TASK, [near])
    assert m_far["semantic_sim"] < m_near["semantic_sim"]


def test_generative_uses_precomputed_vectors():
    v = [1.0, 0.0, 0.0]
    ans = _entry("任意文本", semantic={"vector": v, "ref_vector": v})
    m = compute_task_metrics(GEN_TASK, [ans])
    assert m["semantic_sim"] == 1.0


def test_generative_no_expected_n_a():
    task = {"id": "G2", "dimension": "语言能力", "type": "生成式",
            "prompt": "写作文", "rubric_note": "结构完整"}
    m = compute_task_metrics(task, [_entry("我的作文内容")])
    assert m["semantic_sim"] is None
    assert m["bleu"] is None and m["rouge_l"] is None


def test_rubric_score_passthrough():
    m = compute_task_metrics(GEN_TASK, [_entry("内容")], verdict_score=8.5)
    assert m["rubric_score"] == 8.5
    m2 = compute_task_metrics(GEN_TASK, [_entry("内容")])
    assert m2["rubric_score"] is None


def test_consistency_two_identical_runs():
    entries = [_entry("相同答案"), _entry("相同答案")]
    m = compute_task_metrics(GEN_TASK, entries)
    assert m["consistency"] == 1.0


def test_consistency_single_run_none():
    m = compute_task_metrics(GEN_TASK, [_entry("只有一次")])
    assert m["consistency"] is None


# ---- 代码题 ----

def test_code_task_uses_code_verify():
    ans = _entry("```python\npass\n```", semantic=None)
    ans["code_verify"] = {"status": "run", "passed": 5, "total": 5}
    m = compute_task_metrics(CODE_TASK, [ans])
    assert m["code_verify"] == {"passed": 5, "total": 5}
    assert m["top1"] is None and m["semantic_sim"] is None and m["bleu"] is None


def test_code_task_not_run_skipped():
    ans = _entry("```python\npass\n```")
    ans["code_verify"] = {"status": "disabled", "passed": 0, "total": 0}
    m = compute_task_metrics(CODE_TASK, [ans])
    assert m["skipped"] is True and m["reason"] == "code_not_run"


# ---- 截断 / 失败 ----

def test_truncated_answer_skipped():
    m = compute_task_metrics(DISC_TASK, [_entry("Q1-A", truncated=True)])
    assert m["skipped"] is True
    assert "截断" in m["reason"]


def test_api_error_skipped():
    m = compute_task_metrics(DISC_TASK, [_entry("", status="error", error="boom")])
    assert m["skipped"] is True
    assert m["reason"] == "boom"


def test_error_entry_with_ok_fallback():
    entries = [_entry("", status="error", error="boom"), _entry("Q1-A\nQ2-A\nQ3-A")]
    m = compute_task_metrics(DISC_TASK, entries)
    assert m["top1"] == 1.0


def test_empty_entries_skipped():
    assert compute_task_metrics(DISC_TASK, [])["skipped"] is True
