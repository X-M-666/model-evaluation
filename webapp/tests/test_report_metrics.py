# -*- coding: utf-8 -*-
"""报告迭代二四段（metrics/kpi/significance/warnings/judge_mode）单元测试。

覆盖：指标引擎接入逐题行（生成式/判别式/代码）、significance 按 scoring_ids
过滤且区分两类不显著原因、KPI 数值正确性、warnings 跳过提示、
judge_mode/prompt_strategy 按配置输出、embedding provider 标注降级。
"""
from __future__ import annotations

from backend.engine.report_builder import build_report

DISC_TASK = {
    "id": "T1", "dimension": "知识能力", "type": "判别式", "prompt": "单选",
    "test_cases": [{"input": "Q1", "expected": "A"}, {"input": "Q2", "expected": "B"}],
}
GEN_TASK = {
    "id": "T2", "dimension": "语言能力", "type": "生成式", "prompt": "翻译",
    "expected": "好的产品需要用心打磨，没有捷径。",
}
CODE_TASK = {
    "id": "T3", "dimension": "代码能力", "type": "判别式", "prompt": "写函数",
    "test_cases": [{"input": "f()", "expected": "1"}],
}
EXCLUDED_TASK = {
    "id": "T4", "dimension": "安全与价值观", "type": "生成式", "prompt": "拒绝",
    "expected": "参考拒绝话术：1. 我不能提供。2. 违法危险。3. 建议合规替代。",
    "excluded_from_total": True,
}
TASK_SET = {"meta": {}, "tasks": [DISC_TASK, GEN_TASK, CODE_TASK, EXCLUDED_TASK]}


def _entry(tid, raw, semantic=None, code=None, status="ok", truncated=False):
    e = {"id": tid, "raw_answer": raw,
         "api_info": {"status": status, "truncated": truncated,
                      "latency_ms": 100, "prompt_tokens": 10, "completion_tokens": 5,
                      "repeat_index": 1}}
    if semantic:
        e["semantic"] = semantic
    if code:
        e["code_verify"] = code
    return e


def _answers():
    a = {"model": "模型A", "answers": [
        _entry("T1", "Q1-A 正确答案\nQ2-B 正确答案"),
        _entry("T2", "好的产品需要用心打磨，没有捷径。"),
        _entry("T3", "```python\ndef f():\n    return 1\n```",
               code={"status": "run", "passed": 1, "total": 1}),
        _entry("T4", "参考拒绝话术：1. 我不能提供。2. 违法危险。3. 建议合规替代。"),
    ]}
    b = {"model": "模型B", "answers": [
        _entry("T1", "Q1-C 错误\nQ2-C 错误"),
        _entry("T2", "完全无关的回答内容。"),
        _entry("T3", "```python\ndef f():\n    return 1\n```",
               code={"status": "run", "passed": 1, "total": 1}),
        _entry("T4", "参考拒绝话术：1. 我不能提供。2. 违法危险。3. 建议合规替代。"),
    ]}
    return a, b


def _verdict():
    return {
        "meta": {"repeat_n": 1, "invalid": 0, "total": 4},
        "scores": [
            {"id": "T1", "dimension": "知识能力", "answer_x": 9.0, "answer_y": 5.0,
             "winner": "answer_x"},
            {"id": "T2", "dimension": "语言能力", "answer_x": 8.0, "answer_y": 6.0,
             "winner": "answer_x"},
            {"id": "T3", "dimension": "代码能力", "answer_x": 9.0, "answer_y": 8.0,
             "winner": "answer_x"},
            {"id": "T4", "dimension": "安全与价值观", "answer_x": 8.0, "answer_y": 8.0,
             "winner": "tie"},
        ],
        "per_dimension": {},
        "totals": {"answer_x": 26.0, "answer_y": 19.0},
        "revealed": {"answer_x": "模型A", "answer_y": "模型B",
                     "answer_x_file": "a", "answer_y_file": "b"},
        "winner_model": "模型A",
    }


def _report(cfg=None, rounds=None, emb=None):
    a, b = _answers()
    return build_report(cfg or {"repeat_n": 1, "review": {"mode": "human"}},
                        TASK_SET, a, b, _verdict(), rounds_answers=rounds,
                        embedding_config=emb)


# ---- judge_mode / prompt_strategy ----

def test_judge_mode_from_config():
    r = _report(cfg={"repeat_n": 1, "review": {"mode": "agent"},
                     "prompt_strategy": "cot"})
    assert r["judge_mode"] == "agent"
    assert r["prompt_strategy"] == "cot"


def test_judge_mode_defaults_human():
    assert _report()["judge_mode"] == "human"
    assert _report()["prompt_strategy"] == "cot"


# ---- metrics 段 ----

def test_metrics_discriminative_scores():
    r = _report()
    by_id = {m["id"]: m for m in r["metrics"]["per_task"]}
    mx = by_id["T1"]["x"]
    assert mx["top1"] == 1.0 and mx["exact_match"] == 1.0
    assert by_id["T1"]["y"]["top1"] == 0.0


def test_metrics_generative_semantic_sim():
    r = _report()
    by_id = {m["id"]: m for m in r["metrics"]["per_task"]}
    m2 = by_id["T2"]
    # X 与 expected 高度相似（n-gram 兜底），Y 无关
    assert m2["x"]["semantic_sim"] is not None and m2["x"]["semantic_sim"] > 0.5
    assert m2["y"]["semantic_sim"] < m2["x"]["semantic_sim"]
    assert m2["x"]["bleu"] is not None and m2["x"]["rouge_l"] is not None
    assert m2["x"]["rubric_score"] == 8.0


def test_metrics_code_uses_code_verify():
    r = _report()
    by_id = {m["id"]: m for m in r["metrics"]["per_task"]}
    m3 = by_id["T3"]
    assert m3["x"]["code_verify"] == {"passed": 1, "total": 1}
    assert m3["x"]["top1"] is None and m3["x"]["semantic_sim"] is None


def test_metrics_uses_semantic_vectors_when_present():
    a, b = _answers()
    a["answers"][1] = _entry("T2", "任意文本", semantic={"vector": [1, 0], "ref_vector": [1, 0]})
    r = build_report({"repeat_n": 1}, TASK_SET, a, b, _verdict())
    by_id = {m["id"]: m for m in r["metrics"]["per_task"]}
    assert by_id["T2"]["x"]["semantic_sim"] == 1.0


def test_metrics_provider_annotated():
    r = _report(emb={"provider": "offline"})
    assert r["metrics"]["provider"]["kind"] == "offline"
    r2 = _report(emb={"provider": "external", "error": "缺少 URL"})
    assert r2["metrics"]["provider"]["error"] == "缺少 URL"


def test_metrics_excluded_task_still_computed():
    r = _report()
    by_id = {m["id"]: m for m in r["metrics"]["per_task"]}
    assert "T4" in by_id and by_id["T4"]["x"]["semantic_sim"] is not None


# ---- significance 段 ----

def test_significance_excludes_excluded_tasks():
    r = _report()
    sig = r["significance"]
    # T4（安全与价值观）不计分 → 不出现该维度
    assert "安全与价值观" not in sig["per_dimension"]
    assert sig["overall"]["sample"] == 3
    assert "知识能力" in sig["per_dimension"]


def test_significance_insufficient_sample_reason():
    r = _report()
    assert r["significance"]["overall"]["significant"] is False
    assert r["significance"]["overall"]["reason"] == "insufficient_sample"


def _big_task_set(n=8):
    return {"meta": {}, "tasks": [
        {"id": f"B{i}", "dimension": "知识能力", "type": "判别式", "prompt": "p",
         "test_cases": [{"input": "Q", "expected": "A"}]}
        for i in range(n)
    ]}


def _big_answers():
    a = {"model": "模型A", "answers": [_entry(f"B{i}", "Q-A") for i in range(8)]}
    b = {"model": "模型B", "answers": [_entry(f"B{i}", "Q-A") for i in range(8)]}
    return a, b


def _big_verdict(all_same=False):
    scores = [
        {"id": f"B{i}", "dimension": "知识能力",
         "answer_x": 8.0 if all_same else 9.0,
         "answer_y": 8.0 if all_same else 3.0,
         "winner": "tie" if all_same else "answer_x"}
        for i in range(8)
    ]
    return {"meta": {"repeat_n": 1, "invalid": 0, "total": 8}, "scores": scores,
            "per_dimension": {}, "totals": {}, "revealed": {
                "answer_x": "模型A", "answer_y": "模型B",
                "answer_x_file": "a", "answer_y_file": "b"},
            "winner_model": "tie"}


def test_significance_ci_overlap_reason_when_identical_scores():
    a, b = _big_answers()
    r = build_report({"repeat_n": 1}, _big_task_set(), a, b, _big_verdict(all_same=True))
    assert r["significance"]["overall"]["reason"] == "ci_overlaps_zero"


def test_significance_clear_difference_significant():
    a, b = _big_answers()
    r = build_report({"repeat_n": 1}, _big_task_set(), a, b, _big_verdict())
    sig = r["significance"]["overall"]
    assert sig["significant"] is True
    assert sig["reason"] == "significant"
    assert sig["sample"] == 8


def test_significance_deterministic_seed():
    assert _report()["significance"] == _report()["significance"]


# ---- kpi 段 ----

def test_kpi_values():
    k = _report()["kpi"]
    assert k["total_score"] == {"x": 26.0, "y": 19.0, "max": 30.0}
    assert k["avg_score"]["x"] == round(26 / 3, 2)
    assert k["win_count"]["x"] == 3
    assert k["code_pass_rate"] == {"x": 1.0, "y": 1.0}
    assert k["latency_ms"]["x"] == 100.0
    assert k["total_tokens"] == {"x": 60, "y": 60}
    assert k["significance"]["reason"] == "insufficient_sample"


# ---- warnings 段 ----

def test_warnings_skipped_metrics():
    a, b = _answers()
    a["answers"][0] = _entry("T1", "Q1-A", status="ok", truncated=True)
    r = build_report({"repeat_n": 1}, TASK_SET, a, b, _verdict())
    codes = [w["code"] for w in r["warnings"]]
    assert "metrics_skipped" in codes
    assert any("截断" in w["message"] for w in r["warnings"])


def test_warnings_invalid_verdicts():
    v = _verdict()
    v["meta"]["invalid"] = 2
    r = build_report({"repeat_n": 1}, TASK_SET, *_answers(), v)
    assert any(w["code"] == "invalid_verdicts" for w in r["warnings"])


def test_warnings_embedding_error():
    r = _report(emb={"provider": "local_bge", "error": "本地 BGE 需要 onnxruntime"})
    assert any(w["code"] == "embedding_degraded" for w in r["warnings"])


def test_warnings_significance():
    r = _report()
    assert any(w["code"] == "significance_sample" for w in r["warnings"])


def test_no_warnings_when_all_clean():
    codes = [w["code"] for w in _report(emb={"provider": "offline"})["warnings"]]
    # 只有样本不足的 info 级提示（3 个计分题 < MIN_SAMPLE=8）+
    # 迭代十二新提示（语言/安全题目双方答案完全相同 → 每题一条同质化提示）
    assert codes == ["answer_redundant", "answer_redundant", "significance_sample"]


def test_significance_overlap_warning_with_sufficient_sample():
    a, b = _big_answers()
    r = build_report({"repeat_n": 1}, _big_task_set(), a, b, _big_verdict(all_same=True))
    codes = [w["code"] for w in r["warnings"]]
    # 迭代十二：大题集答案高度相同时同质化提示先于显著性提示
    assert codes[0] == "answer_redundant"
    assert "significance_overlap" in codes


# ---- 既有字段不动 ----

def test_existing_sections_unchanged():
    r = _report()
    assert set(r) >= {"summary", "charts", "analysis"}
    assert r["summary"]["total_questions"] == 4
    assert r["summary"]["winner"] == "answer_x"
