# -*- coding: utf-8 -*-
"""数据集验证、格式转换、解析逻辑。

支持两种输入格式：
- JSON：完整格式（含 id/dimension/prompt/test_cases/rubric_note）或简化格式
- CSV：简化格式（prompt,expected 两列）或完整格式（id,dimension,prompt,expected,rubric_note,difficulty）
"""
from __future__ import annotations

import csv
import io
import json
import re
from datetime import datetime, timezone
from typing import Any


def validate_json_dataset(raw: str) -> dict[str, Any]:
    """校验并规范化 JSON 评测集，返回标准格式 {name, description, tasks}。

    支持两种 JSON 格式：
    1. 完整格式：{name, description, tasks: [{id, dimension, prompt, test_cases, rubric_note, ...}]}
    2. 简化格式：{name, tasks: [{prompt, expected}]}  → 自动补全其他字段
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 解析失败: {e}")

    if not isinstance(data, dict):
        raise ValueError("JSON 顶层必须是对象")

    tasks_raw = data.get("tasks")
    if not tasks_raw or not isinstance(tasks_raw, list) or len(tasks_raw) == 0:
        raise ValueError("缺少 tasks 数组或为空")

    name = data.get("name", "").strip()
    description = data.get("description", "").strip()
    if not name:
        name = f"评测集_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    tasks = []
    for i, t in enumerate(tasks_raw):
        if not isinstance(t, dict):
            raise ValueError(f"tasks[{i}] 不是对象")
        prompt = t.get("prompt", "").strip()
        if not prompt:
            raise ValueError(f"tasks[{i}] 缺少 prompt")

        # 判断是完整格式还是简化格式
        has_test_cases = "test_cases" in t and isinstance(t["test_cases"], list) and len(t["test_cases"]) > 0
        has_rubric = "rubric_note" in t and t["rubric_note"]
        is_full_format = has_test_cases or has_rubric or ("dimension" in t and t["dimension"])

        task: dict[str, Any] = {
            "id": t.get("id", f"T{i+1}"),
            "dimension": t.get("dimension", "自定义"),
            "benchmark": t.get("benchmark", "自定义评测集"),
            "difficulty": t.get("difficulty", "进阶"),
            "prompt": prompt,
            "test_cases": [],
            "rubric_note": t.get("rubric_note", ""),
        }

        if is_full_format:
            # 完整格式：直接使用提供的字段
            if has_test_cases:
                task["test_cases"] = t["test_cases"]
            if has_rubric:
                task["rubric_note"] = t["rubric_note"]
        else:
            # 简化格式：从 expected 构建 test_cases
            expected = t.get("expected", "").strip()
            if expected:
                task["test_cases"] = [{"input": prompt[:50], "expected": expected}]
            task["rubric_note"] = f"【仅评审可见】满分10分。根据回答与期望答案的匹配程度评分。"

        tasks.append(task)

    return {"name": name, "description": description, "tasks": tasks}


def parse_csv_dataset(content: str) -> dict[str, Any]:
    """解析 CSV 评测集，返回标准格式。

    支持两种 CSV 格式：
    1. 简化格式（2列）：prompt,expected
    2. 完整格式（3-6列）：id,dimension,prompt,expected,rubric_note,difficulty
    """
    reader = csv.DictReader(io.StringIO(content))
    if reader.fieldnames is None:
        raise ValueError("CSV 为空或格式错误")

    headers = [h.strip().lower() for h in reader.fieldnames]

    # 检测格式
    has_prompt = "prompt" in headers
    has_expected = "expected" in headers
    has_id = "id" in headers
    has_dimension = "dimension" in headers

    if not has_prompt:
        raise ValueError("CSV 缺少 prompt 列")
    if not has_expected:
        raise ValueError("CSV 缺少 expected 列")

    tasks = []
    for i, raw_row in enumerate(reader):
        # 列名大小写/空白不敏感：统一 strip+小写后按规范名取值
        row = {k.strip().lower(): v for k, v in raw_row.items()}
        prompt = (row.get("prompt", "") or "").strip()
        expected = (row.get("expected", "") or "").strip()
        if not prompt:
            continue

        task_id = (row.get("id", "") or "").strip() or f"T{i+1}"
        dimension = (row.get("dimension", "") or "").strip() or "自定义"
        rubric = (row.get("rubric_note", "") or "").strip()
        difficulty = (row.get("difficulty", "") or "").strip() or "进阶"

        if not rubric:
            rubric = "【仅评审可见】满分10分。根据回答与期望答案的匹配程度评分。"

        task: dict[str, Any] = {
            "id": task_id,
            "dimension": dimension,
            "benchmark": "自定义评测集",
            "difficulty": difficulty,
            "prompt": prompt,
            "test_cases": [{"input": prompt[:50], "expected": expected}] if expected else [],
            "rubric_note": rubric,
        }
        tasks.append(task)

    if not tasks:
        raise ValueError("CSV 中没有有效的题目行")

    return {
        "name": f"CSV评测集_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "description": f"从 CSV 导入，共 {len(tasks)} 题",
        "tasks": tasks,
    }
