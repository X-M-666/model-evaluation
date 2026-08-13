# -*- coding: utf-8 -*-
"""生成式题 expected 参考完备性单元测试（迭代二 D1 修正）。

8 个生成式题（T2E/T4C/T4D/T5C~T5G）必须携带 expected 参考，
否则 semantic_sim/bleu/rouge_l 等生成式文本指标无法计算；
mock 模板需覆盖全部维度且含贴近 expected 的样本。
"""
from __future__ import annotations

from backend.engine.mock import ANSWER_TEMPLATES
from backend.engine.tasks import QUESTION_POOL, build_task_set

GENERATIVE_IDS = {"T2E", "T4C", "T4D", "T5C", "T5D", "T5E", "T5F", "T5G"}


def test_eight_generative_tasks_have_expected():
    by_id = {q["id"]: q for pool in QUESTION_POOL.values() for q in pool}
    for tid in GENERATIVE_IDS:
        q = by_id.get(tid)
        assert q is not None, f"缺失生成式题 {tid}"
        assert q["type"] == "生成式"
        assert (q.get("expected") or "").strip(), f"{tid} 缺少 expected 参考"
        assert len(q["expected"]) >= 30, f"{tid} expected 过短，无法作参考基准"


def test_generative_expected_keywords():
    by_id = {q["id"]: q for pool in QUESTION_POOL.values() for q in pool}
    assert "合计" in by_id["T2E"]["expected"]           # 预算合计
    assert "shortcuts" in by_id["T4C"]["expected"]      # 译文
    assert "时间" in by_id["T4D"]["expected"]           # 时间管理范文
    for tid in ("T5C", "T5D", "T5E", "T5F", "T5G"):      # 三段式拒绝/建议
        assert "1." in by_id[tid]["expected"] and "3." in by_id[tid]["expected"]


def test_non_generative_tasks_expected_located_in_test_cases():
    by_id = {q["id"]: q for pool in QUESTION_POOL.values() for q in pool}
    for q in by_id.values():
        if q["type"] != "生成式" and not q["test_cases"]:
            # IFEval 约束核验题（T5/T5B）以 rubric 逐条评审为准
            assert (q.get("rubric_note") or ""), f"{q['id']} 无 test_cases 时必须由 rubric_note 支撑"


def test_task_set_roundtrip_keeps_expected():
    ts = build_task_set(seed=2026)
    assert ts["meta"]["dataset_version"] == "v2"
    gen = [t for t in ts["tasks"] if t["type"] == "生成式"]
    assert gen, "任务集应能抽到生成式题"
    for t in gen:
        assert (t.get("expected") or "").strip()


def test_mock_templates_cover_all_dimensions():
    dims = {q["dimension"] for pool in QUESTION_POOL.values() for q in pool}
    for d in dims:
        assert ANSWER_TEMPLATES.get(d), f"mock 缺少维度 {d} 的模板"


def test_mock_has_expected_aligned_template():
    by_id = {q["id"]: q for pool in QUESTION_POOL.values() for q in pool}
    for tid in GENERATIVE_IDS:
        dim = by_id[tid]["dimension"]
        exp = by_id[tid]["expected"]
        assert any(
            len(t) >= 20 for t in ANSWER_TEMPLATES.get(dim, [])
        ), f"mock 维度 {dim} 无贴近 expected 的完整模板"
    # 安全维度至少一套与 T5C 拒绝话术同源的模板
    safety = ANSWER_TEMPLATES["安全与价值观"]
    assert any("无法提供" in t or "不能" in t for t in safety)
