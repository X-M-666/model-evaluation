# -*- coding: utf-8 -*-
"""数据集验证、格式转换、解析逻辑。

支持两种输入格式：
- JSON：完整格式（含 id/dimension/prompt/test_cases/rubric_note）或简化格式
- CSV：简化格式（prompt,expected 两列）或完整格式（id,dimension,prompt,expected,rubric_note,difficulty）

issue #15（R2-006）：所有解析器输出的标准格式统一经 validate_standard_dataset
做类型/必填/长度/ID 唯一性校验。非法输入抛 DatasetValidationError（带字段路径），
由 API 层转换为 400，杜绝裸 .strip() 造成的 AttributeError→500 与重复 ID
导致的评审死锁（答卷/评审/报告均按任务 id 分组）。
"""
from __future__ import annotations

import csv
import io
import json
import re
from datetime import datetime, timezone
from typing import Any

# 资源/字段上限（issue #15）：校验层统一生效，main.py 不再重复检查题数
MAX_DATASET_TASKS = 200
MAX_NAME_LEN = 200
MAX_DESCRIPTION_LEN = 2000
MAX_PROMPT_LEN = 20000
MAX_RUBRIC_LEN = 5000
MAX_ID_LEN = 64
MAX_SHORT_FIELD_LEN = 100
MAX_TEST_CASES_PER_TASK = 50
MAX_TEST_CASE_FIELD_LEN = 5000


class DatasetValidationError(ValueError):
    """数据集校验错误：携带字段路径（如 tasks[2].prompt），由 API 层渲染为 400。"""

    def __init__(self, field: str, message: str):
        super().__init__(message)
        self.field = field
        self.message = message

    def __str__(self) -> str:
        return f"{self.field}: {self.message}"


def _type_name(v: Any) -> str:
    return type(v).__name__


def _as_str(value: Any, field: str, *, max_len: int | None = None) -> str:
    """类型安全字符串取值：非 str 抛 DatasetValidationError（替代裸 .strip() 的 AttributeError）。"""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise DatasetValidationError(field, f"必须是字符串，实际为 {_type_name(value)}")
    s = value.strip()
    if max_len is not None and len(s) > max_len:
        raise DatasetValidationError(field, f"长度超过上限 {max_len}")
    return s


def _next_free_id(explicit: set[str], taken: set[str], preferred: str) -> str:
    """按 preferred（形如 T{n}）生成自动 id，与显式/已用 id 冲突时递增后缀。"""
    m = re.match(r"^T(\d+)$", preferred)
    n = int(m.group(1)) if m else 1
    while True:
        candidate = f"T{n}"
        if candidate not in explicit and candidate not in taken:
            return candidate
        n += 1


def _collect_explicit_ids(tasks_raw: list[Any]) -> set[str]:
    """预扫全部显式任务 id（含类型/非空/长度校验），供自动编号避让与重复检测。"""
    explicit: set[str] = set()
    for i, t in enumerate(tasks_raw):
        if not isinstance(t, dict):
            continue
        raw_id = t.get("id")
        if raw_id is None:
            continue
        if not isinstance(raw_id, str):
            raise DatasetValidationError(f"tasks[{i}].id", f"必须是字符串，实际为 {_type_name(raw_id)}")
        sid = raw_id.strip()
        if not sid:
            raise DatasetValidationError(f"tasks[{i}].id", "去空白后不能为空")
        if len(sid) > MAX_ID_LEN:
            raise DatasetValidationError(f"tasks[{i}].id", f"长度超过上限 {MAX_ID_LEN}")
        explicit.add(sid)
    return explicit


def _validate_task(t: dict[str, Any], i: int, explicit_ids: set[str], seen_ids: dict[str, int]) -> None:
    """校验单任务：id 规范字符串且唯一（缺省自动生成并避让）；prompt/test_cases 类型与长度。"""
    prefix = f"tasks[{i}]"

    raw_id = t.get("id")
    if raw_id is None:
        t["id"] = _next_free_id(explicit_ids, set(seen_ids), f"T{i+1}")
        seen_ids[t["id"]] = i
    else:
        if not isinstance(raw_id, str):
            raise DatasetValidationError(f"{prefix}.id", f"必须是字符串，实际为 {_type_name(raw_id)}")
        sid = raw_id.strip()
        if not sid:
            raise DatasetValidationError(f"{prefix}.id", "去空白后不能为空")
        if len(sid) > MAX_ID_LEN:
            raise DatasetValidationError(f"{prefix}.id", f"长度超过上限 {MAX_ID_LEN}")
        if sid in seen_ids:
            raise DatasetValidationError(f"{prefix}.id", f"与 tasks[{seen_ids[sid]}].id 重复（'{sid}'）")
        seen_ids[sid] = i
        t["id"] = sid

    prompt = t.get("prompt")
    if prompt is None:
        raise DatasetValidationError(f"{prefix}.prompt", "不能为空")
    if not isinstance(prompt, str):
        raise DatasetValidationError(f"{prefix}.prompt", f"必须是字符串，实际为 {_type_name(prompt)}")
    prompt = prompt.strip()
    if not prompt:
        raise DatasetValidationError(f"{prefix}.prompt", "不能为空")
    if len(prompt) > MAX_PROMPT_LEN:
        raise DatasetValidationError(f"{prefix}.prompt", f"长度超过上限 {MAX_PROMPT_LEN}")
    t["prompt"] = prompt

    for f, max_len in (
        ("dimension", MAX_SHORT_FIELD_LEN),
        ("benchmark", MAX_SHORT_FIELD_LEN),
        ("difficulty", MAX_SHORT_FIELD_LEN),
        ("rubric_note", MAX_RUBRIC_LEN),
    ):
        if f in t:
            t[f] = _as_str(t[f], f"{prefix}.{f}", max_len=max_len)

    tc_raw = t.get("test_cases", [])
    if not isinstance(tc_raw, list):
        raise DatasetValidationError(f"{prefix}.test_cases", "必须是数组")
    if len(tc_raw) > MAX_TEST_CASES_PER_TASK:
        raise DatasetValidationError(f"{prefix}.test_cases", f"用例数量超过上限 {MAX_TEST_CASES_PER_TASK}")
    for j, tc in enumerate(tc_raw):
        if not isinstance(tc, dict):
            raise DatasetValidationError(f"{prefix}.test_cases[{j}]", "必须是对象")
        for k in ("input", "expected"):
            v = tc.get(k)
            if v is None:
                raise DatasetValidationError(f"{prefix}.test_cases[{j}].{k}", "不能为空")
            if not isinstance(v, str):
                raise DatasetValidationError(f"{prefix}.test_cases[{j}].{k}", f"必须是字符串，实际为 {_type_name(v)}")
            s = v.strip()
            if not s:
                raise DatasetValidationError(f"{prefix}.test_cases[{j}].{k}", "去空白后不能为空")
            if len(s) > MAX_TEST_CASE_FIELD_LEN:
                raise DatasetValidationError(f"{prefix}.test_cases[{j}].{k}", f"长度超过上限 {MAX_TEST_CASE_FIELD_LEN}")
            tc[k] = s


def validate_standard_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    """统一语义校验（issue #15）：各解析器输出标准格式后调用；就地规范化并返回。

    校验项：顶层 name/description 类型与长度；任务字段类型/必填/长度；
    任务 id 非空规范字符串且唯一；test_cases 嵌套结构；任务总数上限。
    """
    dataset["name"] = _as_str(dataset.get("name", ""), "name", max_len=MAX_NAME_LEN)
    if not dataset["name"]:
        raise DatasetValidationError("name", "去空白后不能为空")
    dataset["description"] = _as_str(dataset.get("description", ""), "description", max_len=MAX_DESCRIPTION_LEN)

    tasks = dataset.get("tasks")
    if not isinstance(tasks, list):
        raise DatasetValidationError("tasks", "必须是数组")
    if not tasks:
        raise DatasetValidationError("tasks", "不能为空")
    if len(tasks) > MAX_DATASET_TASKS:
        raise DatasetValidationError("tasks", f"题目数量超过上限 {MAX_DATASET_TASKS}")

    explicit_ids = _collect_explicit_ids(tasks)
    seen_ids: dict[str, int] = {}
    for i, t in enumerate(tasks):
        if not isinstance(t, dict):
            raise DatasetValidationError(f"tasks[{i}]", "必须是对象")
        _validate_task(t, i, explicit_ids, seen_ids)
    return dataset


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

    name = _as_str(data.get("name", ""), "name", max_len=MAX_NAME_LEN)
    if not name:
        name = f"评测集_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    description = _as_str(data.get("description", ""), "description", max_len=MAX_DESCRIPTION_LEN)

    # 预扫显式 id：自动编号须避开显式 id（issue #15）
    explicit_ids = _collect_explicit_ids(tasks_raw)
    used_ids: set[str] = set()

    tasks = []
    for i, t in enumerate(tasks_raw):
        if not isinstance(t, dict):
            raise ValueError(f"tasks[{i}] 不是对象")

        prompt_raw = t.get("prompt")
        if prompt_raw is None:
            raise ValueError(f"tasks[{i}] 缺少 prompt")
        if not isinstance(prompt_raw, str):
            raise DatasetValidationError(f"tasks[{i}].prompt", f"必须是字符串，实际为 {_type_name(prompt_raw)}")
        prompt = prompt_raw.strip()
        if not prompt:
            raise ValueError(f"tasks[{i}] 缺少 prompt")

        # 判断是完整格式还是简化格式
        if "test_cases" in t and not isinstance(t["test_cases"], list):
            raise DatasetValidationError(f"tasks[{i}].test_cases", "必须是数组")
        has_test_cases = isinstance(t.get("test_cases"), list) and len(t["test_cases"]) > 0
        has_rubric = isinstance(t.get("rubric_note"), str) and bool(t["rubric_note"].strip())
        is_full_format = has_test_cases or has_rubric or bool(t.get("dimension"))

        raw_id = t.get("id")
        if raw_id is None:
            task_id = _next_free_id(explicit_ids, used_ids, f"T{i+1}")
            used_ids.add(task_id)
        else:
            task_id = raw_id.strip()  # 类型/非空/长度已在 _collect_explicit_ids 校验

        task: dict[str, Any] = {
            "id": task_id,
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
            expected_raw = t.get("expected", "")
            if expected_raw is None:
                expected_raw = ""
            if not isinstance(expected_raw, str):
                raise DatasetValidationError(f"tasks[{i}].expected", f"必须是字符串，实际为 {_type_name(expected_raw)}")
            expected = expected_raw.strip()
            if expected:
                task["test_cases"] = [{"input": prompt[:50], "expected": expected}]
            task["rubric_note"] = f"【仅评审可见】满分10分。根据回答与期望答案的匹配程度评分。"

        tasks.append(task)

    dataset = {"name": name, "description": description, "tasks": tasks}
    return validate_standard_dataset(dataset)


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

        if has_id:
            # id 列存在：单元格必须为非空规范字符串（issue #15）
            task_id = (row.get("id", "") or "").strip()
            if not task_id:
                raise DatasetValidationError(f"tasks[{i}].id", "去空白后不能为空")
        else:
            task_id = f"T{i+1}"
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

    result = {
        "name": f"CSV评测集_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "description": f"从 CSV 导入，共 {len(tasks)} 题",
        "tasks": tasks,
    }
    return validate_standard_dataset(result)
