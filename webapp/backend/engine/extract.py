# -*- coding: utf-8 -*-
"""答案提取分层（迭代二）：规则链优先（精确/模式匹配），LLM 解析兜底（可注入回调）。

VLMEvalKit 模式：先低成本规则/精确匹配，未命中才走 LLM 解析。
LLM 兜底默认关闭（无配置绝不发起网络），由评测配置显式注入回调。
规则链全部为纯函数，确定性可测。
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

# 逐子题「题号 + 选项字母」：Q1-A / Q1：A / Q1 A / 第1题：B 等
_QUESTION_LETTER_RE = re.compile(
    r"(?:Q\s*(\d+)|[（(]?第\s*([一二三四五六七八九十1-9])\s*[题问]|(?:[①②③④⑤⑥])|(?:\(\s*(\d+)\s*\)))"
    r"[)）]?[\s:：.-]*([A-Da-d])\b"
)
# 按出现顺序提取选项字母
_LETTERS_RE = re.compile(r"\b([A-Da-d])\b")
# 结论句数值：数字（含小数/百分号/负数）
_NUMBER_RE = re.compile(r"(-?\d+(?:\.\d+)?(?:%|％)?)")
_JSON_ANSWER_KEYS = ("答案", "answer", "result", "结论", "conclusion")
_CODE_BLOCK_RE = re.compile(r"```(?:python|py|java|cpp|c|javascript|js|ts|go|rust|sql)?\s*\n?(.*?)```", re.DOTALL)

LLMCallable = Callable[[str], str | None]


def _norm(s: str) -> str:
    """归一化文本：去空白与常见分隔符，统一小写。"""
    return re.sub(r"[\s，。、；：:()（）\[\]【】\"''`~\-—]+", "", (s or "")).lower()


def extract_json_answer(raw: str) -> str | None:
    """从 JSON 结构中提取答案字段（{"答案": ...} / {"answer": ...}）。"""
    m = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    for key in _JSON_ANSWER_KEYS:
        v = obj.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def extract_code_block(raw: str) -> str | None:
    """提取代码块内容（代码维度题）。"""
    m = _CODE_BLOCK_RE.search(raw or "")
    return m.group(1).strip() if m else None


def extract_option_letters(raw: str) -> list[str] | None:
    """按出现顺序提取全部选项字母（A~D）。"""
    letters = [m.group(1).upper() for m in _LETTERS_RE.finditer(raw or "")]
    return letters or None


def extract_number(raw: str) -> float | None:
    """提取文本中的数值（取最后一个，通常为结论）。"""
    matches = _NUMBER_RE.findall(raw or "")
    if not matches:
        return None
    try:
        return float(matches[-1].rstrip("%％"))
    except ValueError:
        return None


def _subquestion_letters(raw: str) -> dict[int, str] | None:
    """逐子题题号 → 选项字母映射（Q1-A 木星 / 第1题：B 等）。"""
    result: dict[int, str] = {}
    for m in _QUESTION_LETTER_RE.finditer(raw or ""):
        qno = m.group(1) or m.group(2) or m.group(3)
        letter = m.group(4).upper()
        try:
            n = int(qno) if qno.isdigit() else int("一二三四五六七八九十".index(qno) + 1)
        except ValueError:
            continue
        result[n] = letter
    return result or None


def extract_per_case(task: dict[str, Any], raw: str) -> list[str | None]:
    """按 test_cases 逐子题提取答案（与 test_cases 对齐）。

    判别式多子题：优先「题号+字母」逐题提取；无题号时按字母出现顺序对齐；
    单题/生成式：整体提取一次。
    """
    cases = task.get("test_cases") or []
    if not cases or task.get("type") == "生成式":
        return [extract_answer(task, raw)]

    by_qno = _subquestion_letters(raw)
    if by_qno:
        return [by_qno.get(i + 1) for i in range(len(cases))]

    letters = extract_option_letters(raw)
    if letters:
        aligned = letters[: len(cases)]
        return aligned + [None] * (len(cases) - len(aligned))

    whole = extract_answer(task, raw)
    return [whole] + [None] * (len(cases) - 1)


def extract_answer(
    task: dict[str, Any],
    raw_answer: str,
    llm_call: LLMCallable | None = None,
) -> str:
    """分层提取：JSON → 代码块 → 选项字母 → 数值 → LLM 兜底 → 原文。

    llm_call 默认 None（不启用 LLM 解析，绝不静默外连）；由评测配置显式注入。
    """
    raw = (raw_answer or "").strip()
    if not raw:
        return ""

    j = extract_json_answer(raw)
    if j is not None:
        return j

    code = extract_code_block(raw)
    if code is not None:
        return code

    if task.get("dimension") == "代码能力":
        return raw

    letters = extract_option_letters(raw)
    if letters:
        return ",".join(letters)

    num = extract_number(raw)
    if num is not None:
        return str(int(num)) if num.is_integer() else str(num)

    if llm_call is not None:
        try:
            parsed = llm_call(raw)
            if isinstance(parsed, str) and parsed.strip():
                return parsed.strip()
        except Exception:
            pass

    return raw
