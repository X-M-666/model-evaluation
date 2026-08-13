# -*- coding: utf-8 -*-
"""Bad Case 体系（迭代五）：存储层 CRUD + 挖掘规则 + 归因解析（纯函数层）。

先回归后实现（红→绿）：存储契约在 engine/badcase.py 之前锁定。
"""
from __future__ import annotations

import json

import pytest

from backend import storage
from backend.engine.badcase import (
    BAD_CASE_CATEGORIES, DUAL_FAIL_THRESHOLD, LOW_SCORE_THRESHOLD,
    SAFETY_EDGE_THRESHOLD, UNCATEGORIZED, build_attribution_prompt,
    mine_bad_cases, parse_attribution,
)
from backend.engine.tasks import SAFETY_DIMENSION

JOB = "20260101_000000_000009"


# ================= 存储层 =================


def _case(job_id="20260101_000000_000001", task_id="T1", **kw):
    base = {
        "case_id": storage.make_badcase_id(job_id, task_id),
        "job_id": job_id,
        "task_id": task_id,
        "category": "未归类",
        "sources": ["low_score"],
        "model": "x",
        "score": {"x": 2.0, "y": 7.0},
        "evidence": {"winner": "answer_y", "answer_x": "错误答案截断"},
        "attribution": {"label": "未归类", "by": "auto", "confirmed": False,
                        "basis": "", "suggestion": "", "updated_at": None},
        "created_at": "2026-01-01T00:00:00Z",
    }
    base.update(kw)
    return base


def test_make_badcase_id_sanitizes_task_id():
    cid = storage.make_badcase_id("20260101_000000_000001", "T/1:x")
    assert storage.is_valid_badcase_id(cid)
    assert "T" in cid and "1" in cid and "x" in cid
    assert not storage.is_valid_badcase_id("..\\..\\evil")
    assert not storage.is_valid_badcase_id("T1")  # 缺 bc_ 前缀
    assert not storage.is_valid_badcase_id("bc_bad_id")  # job 段非法


def test_save_load_roundtrip():
    case = _case()
    storage.save_badcase(case)
    loaded = storage.load_badcase(case["case_id"])
    assert loaded is not None
    assert loaded["task_id"] == "T1"
    assert loaded["attribution"]["confirmed"] is False


def test_save_rejects_bad_case_id():
    with pytest.raises(ValueError):
        storage.save_badcase(_case(case_id="../evil"))


def test_list_filters_by_job_and_sorts():
    storage.save_badcase(_case(job_id="20260101_000000_000001", task_id="T1"))
    storage.save_badcase(_case(job_id="20260101_000000_000002", task_id="T2"))
    all_cases = storage.list_badcases()
    assert len(all_cases) == 2
    only = storage.list_badcases("20260101_000000_000001")
    assert len(only) == 1 and only[0]["task_id"] == "T1"


def test_delete_and_unknown():
    case = _case()
    storage.save_badcase(case)
    assert storage.delete_badcase(case["case_id"]) is True
    assert storage.delete_badcase(case["case_id"]) is False
    assert storage.load_badcase(case["case_id"]) is None
    assert storage.load_badcase("../evil") is None
    assert storage.delete_badcase("../evil") is False


def test_update_attribution():
    case = _case()
    storage.save_badcase(case)
    updated = storage.update_badcase_attribution(case["case_id"], {
        "label": "推理错误", "by": "human", "confirmed": True, "suggestion": "补充条件",
    })
    assert updated["attribution"]["label"] == "推理错误"
    assert updated["attribution"]["confirmed"] is True
    assert storage.update_badcase_attribution("bc_20260101_000000_000001_ZZZ", {
        "confirmed": True}) is None


def test_export_json_all_and_by_job():
    storage.save_badcase(_case(job_id="20260101_000000_000001", task_id="T1"))
    storage.save_badcase(_case(job_id="20260101_000000_000002", task_id="T2"))
    all_data = json.loads(storage.export_badcases_json())
    assert all_data["total"] == 2
    one = json.loads(storage.export_badcases_json("20260101_000000_000001"))
    assert one["total"] == 1 and one["cases"][0]["task_id"] == "T1"


def test_update_saturation_dataset_param_compat():
    # 旧签名兼容（无 dataset）+ 新签名记录 dataset
    assert storage.update_saturation("20260101_000000_000003",
                                     [{"id": "T1", "winner": "answer_x"}]) is True
    assert storage.update_saturation("20260101_000000_000003",
                                     [{"id": "T1", "winner": "answer_x"}]) is False
    assert storage.update_saturation("20260101_000000_000004",
                                     [{"id": "T1", "winner": "tie"}],
                                     dataset="生成集A") is True
    data = storage.get_saturation()
    by_dataset = [j for j in data["jobs"] if j.get("dataset") == "生成集A"]
    assert len(by_dataset) == 1
    assert storage.update_saturation("../bad", [], dataset="x") is False


# ================= 挖掘规则 =================


def _task(tid="T1", dimension="知识能力", context="", **kw):
    base = {"id": tid, "type": "判别式", "dimension": dimension,
            "prompt": f"题面{tid}", "expected": "E", "context": context}
    base.update(kw)
    return base


def _verdict(*pairs):
    return {
        "scores": [{"id": tid, "dimension": dim, "answer_x": x, "answer_y": y,
                    "winner": "answer_x" if x > y else "answer_y", "basis": "评审依据"}
                   for tid, dim, x, y in pairs],
        "revealed": {"answer_x": "M1", "answer_y": "M2",
                     "answer_x_file": "a", "answer_y_file": "b"},
    }


def _answers(tid, raw, code_verify=None):
    e = {"id": tid, "raw_answer": raw,
         "api_info": {"status": "ok", "latency_ms": 1, "prompt_tokens": 1,
                      "completion_tokens": 1, "repeat_index": 1}}
    if code_verify:
        e["code_verify"] = code_verify
    return {"answers": [e]}


def test_taxonomy_constants():
    assert BAD_CASE_CATEGORIES == ("事实错误", "语义错误", "推理错误", "安全偏见输出", "执行错误")
    assert UNCATEGORIZED == "未归类"


def test_low_score_single_side():
    cases = mine_bad_cases(JOB, {"tasks": [_task("T1")]}, _verdict(("T1", "知识能力", 2.0, 4.5)))
    assert len(cases) == 1
    c = cases[0]
    assert c["sources"] == ["low_score"]
    assert c["model"] == "x"
    assert c["category"] == UNCATEGORIZED
    assert c["score"] == {"x": 2.0, "y": 4.5}
    assert c["winner"] == "answer_y"


def test_dual_fail_both_low():
    cases = mine_bad_cases(JOB, {"tasks": [_task("T1")]}, _verdict(("T1", "知识能力", 3.5, 3.0)))
    assert cases[0]["sources"] == ["dual_fail"]
    assert cases[0]["model"] == "both"


def test_disagreement_merged_with_low_score():
    cases = mine_bad_cases(JOB, {"tasks": [_task("T1")]}, _verdict(("T1", "知识能力", 2.0, 8.0)))
    assert cases[0]["sources"] == ["disagreement", "low_score"]


def test_no_signal_no_case():
    cases = mine_bad_cases(JOB, {"tasks": [_task("T1")]}, _verdict(("T1", "知识能力", 8.0, 8.0)))
    assert cases == []


def test_edge_safety_dimension():
    cases = mine_bad_cases(JOB, {"tasks": [_task("T1", dimension=SAFETY_DIMENSION)]},
                           _verdict(("T1", SAFETY_DIMENSION, 5.0, 6.0)))
    assert cases[0]["sources"] == ["edge_safety"]
    assert cases[0]["model"] == "both"  # 两侧均非低分（<3.0），边缘信号来自安全维低分


def test_edge_grounding_false():
    task = _task("T1", context="参考文档内容")
    metrics = [{"id": "T1", "x": {}, "y": {},
                "grounding": {"x": {"grounded": True}, "y": {"grounded": False}}}]
    cases = mine_bad_cases(JOB, {"tasks": [task]}, _verdict(("T1", "知识能力", 6.0, 6.0)),
                           per_task_metrics=metrics)
    assert "edge_grounding" in cases[0]["sources"]


def test_edge_code_fail():
    task = _task("T1", dimension="代码能力")
    cv = {"status": "run", "passed": 3, "total": 5}
    cases = mine_bad_cases(JOB, {"tasks": [task]}, _verdict(("T1", "代码能力", 7.0, 7.0)),
                           answers_x=_answers("T1", "code", cv))
    assert "edge_code" in cases[0]["sources"]


def test_edge_code_pass_not_flagged():
    task = _task("T1", dimension="代码能力")
    cv = {"status": "run", "passed": 5, "total": 5}
    cases = mine_bad_cases(JOB, {"tasks": [task]}, _verdict(("T1", "代码能力", 7.0, 7.0)),
                           answers_x=_answers("T1", "code", cv))
    assert cases == []


def test_multi_task_each_case_and_evidence_snippet():
    verdict = _verdict(("T1", "知识能力", 2.0, 8.0), ("T2", "知识能力", 8.0, 8.0))
    tasks = {"tasks": [_task("T1"), _task("T2")]}
    cases = mine_bad_cases(JOB, tasks, verdict,
                           answers_x=_answers("T1", "X" * 3000))
    assert len(cases) == 1
    ev = cases[0]["evidence"]
    assert "截断" in ev["answer_x"]
    assert ev["basis"] == "评审依据"
    assert ev["winner"] == "answer_y"


def test_build_prompt_contains_question_and_category_hint():
    p = build_attribution_prompt(_task("T1"), {"x": 2.0, "y": 8.0}, "answer_y",
                                 "错误回答", "正确回答")
    assert "事实错误" in p and "推理错误" in p
    assert "错误回答" in p and "X=2.0" in p


def test_parse_attribution_valid_and_whitelist():
    parsed = parse_attribution('{"category":"推理错误","basis":"步骤2出错","suggestion":"补充条件"}')
    assert parsed["category"] == "推理错误"
    assert parse_attribution('{"category":"未知类型","basis":"x"}') is None
    assert parse_attribution("不是 JSON") is None
    assert parse_attribution(None) is None
    assert parse_attribution('```json\n{"category":"事实错误","basis":"b"}\n```')["category"] == "事实错误"
