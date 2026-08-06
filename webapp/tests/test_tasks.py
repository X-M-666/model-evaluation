# -*- coding: utf-8 -*-
"""内置题库与任务集生成器单元测试（补强方案 #9：tasks.py 覆盖率盲区）。

保护不变量：
- 七大维度完整、每维度 T/TB 双题、字段齐全、rubric 仅供评审可见
- 代码题 5 组用例且 input 为函数调用形式（sandbox 打印包裹执行分支）
- 效率题 5 组用例且 input 为 stdin 文本形式（绝不误判为函数调用）
- generate_tasks 抽样/复现/维度过滤/编号语义
- build_task_set 稳定性 repeat 标记与 meta 语义
- build_task_set_from_dataset 字段补全、编号与稳定性检测
"""
from __future__ import annotations

import re

import pytest

from backend.engine.datasets import DatasetValidationError
from backend.engine.tasks import (
    DIMENSIONS,
    QUESTION_POOL,
    build_task_set,
    build_task_set_from_dataset,
    generate_tasks,
)

# 与 sandbox.verify_code_task 的函数型判定正则保持一致
_FUNC_CALL_RE = re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_]*\(")

CODE_DIMS = {"代码能力", "效率与稳定性"}
REQUIRED_FIELDS = {
    "id", "dimension", "benchmark", "difficulty", "prompt",
    "test_cases", "rubric_note",
}


def _all_pool_questions():
    for dim, pool in QUESTION_POOL.items():
        for q in pool:
            yield dim, q


# ---- 题库池完整性 ----

def test_dimensions_complete_and_pool_aligned():
    assert len(DIMENSIONS) == 7
    assert set(DIMENSIONS) == set(QUESTION_POOL.keys())


def test_each_dimension_has_t_and_tb_pair():
    seen = set()
    for dim in DIMENSIONS:
        ids = sorted(q["id"] for q in QUESTION_POOL[dim])
        assert len(ids) == 2, f"{dim} 应恰好 T/TB 两题: {ids}"
        seen.update(ids)
    assert len(seen) == 14
    assert all(re.fullmatch(r"T[1-7]B?", i) for i in seen)


def test_pool_question_required_fields():
    for dim, q in _all_pool_questions():
        assert REQUIRED_FIELDS <= set(q), f"{q['id']} 缺少字段: {REQUIRED_FIELDS - set(q)}"
        assert q["dimension"] == dim
        assert q["rubric_note"].startswith("【仅评审可见】"), f"{q['id']} rubric 未标注仅供评审可见"


def test_code_questions_have_five_function_call_cases():
    for dim, q in _all_pool_questions():
        if dim != "代码能力":
            continue
        assert len(q["test_cases"]) == 5, f"{q['id']} 应为 5 组用例（与 mock/executor total=5 对齐）"
        for tc in q["test_cases"]:
            assert _FUNC_CALL_RE.match(tc["input"]), f"{q['id']} 用例应为函数调用形式: {tc['input']!r}"


def test_efficiency_questions_have_five_stdin_cases():
    for dim, q in _all_pool_questions():
        if dim != "效率与稳定性":
            continue
        assert len(q["test_cases"]) == 5, f"{q['id']} 应为 5 组用例"
        for tc in q["test_cases"]:
            # 程序型：输入作为 stdin 原样传入，绝不能被误判为函数调用
            assert not _FUNC_CALL_RE.match(tc["input"]), f"{q['id']} 程序型用例不应是函数调用: {tc['input']!r}"


def test_non_code_questions_cases_well_formed():
    for dim, q in _all_pool_questions():
        if dim in CODE_DIMS:
            continue
        # test_cases 必须为列表（可空：T5/T5B 指令约束由 rubric 逐条核验）
        assert isinstance(q["test_cases"], list), f"{q['id']} test_cases 类型错误"
        for tc in q["test_cases"]:
            assert "input" in tc and "expected" in tc and tc["expected"], f"{q['id']} 用例缺少字段"


def test_pool_ids_unique_globally():
    ids = [q["id"] for _, q in _all_pool_questions()]
    assert len(ids) == len(set(ids))


# ---- generate_tasks ----

def test_generate_default_all_dimensions():
    tasks = generate_tasks(seed=42)
    assert len(tasks) == 7
    assert [t["dimension"] for t in tasks] == DIMENSIONS
    assert [t["id"] for t in tasks] == [f"T{i+1}" for i in range(7)]
    assert len({t["dimension"] for t in tasks}) == 7


def test_generate_seed_reproducible():
    assert generate_tasks(seed=2026) == generate_tasks(seed=2026)


def test_generate_dims_filter():
    tasks = generate_tasks(dims=["知识能力", "代码能力"], seed=1)
    assert {t["dimension"] for t in tasks} == {"知识能力", "代码能力"}
    assert len(tasks) == 2


def test_generate_unknown_dimension_skipped():
    tasks = generate_tasks(dims=["不存在的维度", "知识能力"], seed=1)
    assert [t["dimension"] for t in tasks] == ["知识能力"]


def test_generate_num_questions_sampled():
    tasks = generate_tasks(seed=7, num_questions=3)
    assert len(tasks) == 3
    assert len({t["dimension"] for t in tasks}) == 3


def test_generate_num_questions_clamped():
    tasks = generate_tasks(seed=7, num_questions=99)
    assert len(tasks) == 7


def test_generate_num_questions_zero_returns_empty():
    assert generate_tasks(seed=7, num_questions=0) == []


def test_generate_ids_renumbered():
    tasks = generate_tasks(seed=11, num_questions=3)
    assert [t["id"] for t in tasks] == ["T1", "T2", "T3"]


# ---- build_task_set ----

def test_build_task_set_meta():
    ts = build_task_set(seed=5, num_questions=7)
    assert ts["meta"]["total"] == len(ts["tasks"]) == 7
    assert ts["meta"]["scope"] == "七大能力维度"
    assert ts["meta"]["num_questions"] == 7
    assert ts["meta"]["created_by"] == "webapp"


def test_build_task_set_stability_flag_when_efficiency_selected():
    ts = build_task_set(dims=["效率与稳定性"], seed=5)
    assert ts["meta"]["eval_flags"] == {"stability_repeat": {"T1": 2}}


def test_build_task_set_no_stability_flag_otherwise():
    ts = build_task_set(dims=["知识能力", "代码能力"], seed=5)
    assert "eval_flags" not in ts["meta"]


def test_build_task_set_tasks_valid_pool_shapes():
    ts = build_task_set(seed=99, num_questions=4)
    for t in ts["tasks"]:
        assert REQUIRED_FIELDS <= set(t)
        assert "id" in t and t["dimension"] in DIMENSIONS


# ---- build_task_set_from_dataset ----

def _raw_dataset(tasks, name="测试集", description="描述"):
    return {"name": name, "description": description, "tasks": tasks}


def test_dataset_empty_tasks_raises():
    with pytest.raises(ValueError):
        build_task_set_from_dataset({"name": "x", "tasks": []})
    with pytest.raises(ValueError):
        build_task_set_from_dataset({"name": "x"})


def test_dataset_missing_fields_filled():
    ts = build_task_set_from_dataset(_raw_dataset([{"prompt": "1+1?"}]))
    t = ts["tasks"][0]
    assert t["id"] == "T1"
    assert t["dimension"] == "自定义"
    assert t["benchmark"] == "自定义评测集"
    assert t["difficulty"] == "进阶"
    assert t["test_cases"] == []
    assert t["rubric_note"]
    assert ts["meta"]["total"] == 1
    assert ts["meta"]["scope"] == "自定义评测集"
    assert ts["meta"]["dataset_name"] == "测试集"
    assert ts["meta"]["dataset_description"] == "描述"


def test_dataset_existing_ids_and_fields_kept():
    tasks = [{
        "id": "Q9", "dimension": "知识能力", "prompt": "p",
        "test_cases": [{"input": "i", "expected": "e"}],
    }]
    ts = build_task_set_from_dataset(_raw_dataset(tasks))
    t = ts["tasks"][0]
    assert t["id"] == "Q9"
    assert t["dimension"] == "知识能力"
    assert t["test_cases"] == [{"input": "i", "expected": "e"}]


def test_dataset_ids_numbered_for_missing_only():
    tasks = [
        {"id": "A", "prompt": "p1"},
        {"prompt": "p2"},
        {"prompt": "p3"},
    ]
    ts = build_task_set_from_dataset(_raw_dataset(tasks))
    assert [t["id"] for t in ts["tasks"]] == ["A", "T2", "T3"]


def test_dataset_stability_flag_detected():
    tasks = [
        {"id": "S1", "dimension": "效率与稳定性", "prompt": "p"},
        {"id": "S2", "dimension": "知识能力", "prompt": "p"},
    ]
    ts = build_task_set_from_dataset(_raw_dataset(tasks))
    assert ts["meta"]["eval_flags"] == {"stability_repeat": {"S1": 2}}


def test_dataset_multiple_dimensions_total():
    tasks = [
        {"dimension": "知识能力", "prompt": "p"},
        {"dimension": "代码能力", "prompt": "p"},
        {"dimension": "知识能力", "prompt": "p"},
    ]
    ts = build_task_set_from_dataset(_raw_dataset(tasks))
    assert ts["meta"]["total"] == 3


# ---- build_task_set_from_dataset：篡改数据拒绝（issue #15 / R2-006） ----

def test_dataset_duplicate_ids_rejected():
    tasks = [
        {"id": "X", "prompt": "p1"},
        {"id": "X", "prompt": "p2"},
    ]
    with pytest.raises(DatasetValidationError, match="重复"):
        build_task_set_from_dataset(_raw_dataset(tasks))


def test_dataset_non_string_prompt_rejected():
    tasks = [{"id": "X", "prompt": 123}]
    with pytest.raises(DatasetValidationError, match=r"tasks\[0\]\.prompt: 必须是字符串"):
        build_task_set_from_dataset(_raw_dataset(tasks))


def test_dataset_non_string_id_rejected():
    tasks = [{"id": 123, "prompt": "p"}]
    with pytest.raises(DatasetValidationError, match=r"tasks\[0\]\.id: 必须是字符串"):
        build_task_set_from_dataset(_raw_dataset(tasks))


def test_dataset_auto_ids_avoid_explicit_collision():
    """篡改/手工数据集：缺省 id 自动补全时不得与显式 id 冲突。"""
    tasks = [
        {"id": "T2", "prompt": "p1"},
        {"prompt": "p2"},
    ]
    ts = build_task_set_from_dataset(_raw_dataset(tasks))
    assert [t["id"] for t in ts["tasks"]] == ["T2", "T3"]
