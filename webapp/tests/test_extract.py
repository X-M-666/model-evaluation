# -*- coding: utf-8 -*-
"""答案提取分层单元测试（迭代二）：规则链各分支、LLM 兜底默认关闭、失败回退原文。"""
from __future__ import annotations

from backend.engine.extract import (
    extract_answer,
    extract_code_block,
    extract_json_answer,
    extract_number,
    extract_option_letters,
    extract_per_case,
)

TASK = {"id": "T1", "dimension": "知识能力", "type": "判别式",
        "test_cases": [{"input": "Q1 答案？", "expected": "A"},
                       {"input": "Q2 答案？", "expected": "B"}]}


def test_json_answer_extraction():
    assert extract_json_answer('{"答案": "A"}') == "A"
    assert extract_json_answer('{"answer": "42", "其他": 1}') == "42"
    assert extract_json_answer("没有 JSON 结构") is None
    assert extract_json_answer('{"broken": ') is None


def test_code_block_extraction():
    raw = "以下是代码：\n```python\ndef f():\n    return 1\n```"
    assert "def f():" in extract_code_block(raw)
    assert extract_code_block("无代码") is None


def test_option_letters_sequence():
    assert extract_option_letters("答案是 A，然后是 B、C") == ["A", "B", "C"]
    assert extract_option_letters("没有字母") is None


def test_number_extraction_takes_last():
    assert extract_number("先有 3 个，结论是 42") == 42
    assert extract_number("总和 44440") == 44440
    assert extract_number("无数字") is None
    assert extract_number("50% 的人") == 50


def test_extract_answer_priority_json_over_letters():
    raw = '{"答案": "C"}\n另外我倾向 A'
    assert extract_answer(TASK, raw) == "C"


def test_extract_answer_letters():
    assert extract_answer(TASK, "Q1-A 木星 Q2-B 土星") == "A,B"


def test_extract_answer_number():
    assert extract_answer(TASK, "最终的结论是 44440") == "44440"


def test_extract_answer_empty():
    assert extract_answer(TASK, "") == ""
    assert extract_answer(TASK, None) == ""


def test_extract_answer_llm_fallback_invoked_when_injected():
    calls = []

    def llm(raw):
        calls.append(raw)
        return "归一化答案"

    raw = "一段完全自由格式的回答内容"
    assert extract_answer(TASK, raw, llm_call=llm) == "归一化答案"
    assert calls == [raw]


def test_extract_answer_no_llm_by_default():
    """LLM 兜底默认关闭：无注入回调时绝不调用（保持纯规则，无网络）。"""
    raw = "一段完全自由格式的回答内容"
    assert extract_answer(TASK, raw) == raw


def test_extract_answer_llm_failure_falls_back_raw():
    def llm(raw):
        raise RuntimeError("LLM 失败")

    raw = "自由文本"
    assert extract_answer(TASK, raw, llm_call=llm) == raw


def test_extract_per_case_question_numbers():
    raw = "Q1-A 木星\nQ2-B 土星"
    assert extract_per_case(TASK, raw) == ["A", "B"]


def test_extract_per_case_letters_sequence_fallback():
    raw = "答案是 A、B"
    assert extract_per_case(TASK, raw) == ["A", "B"]


def test_extract_per_case_shortfall_pads_none():
    raw = "Q1-A"
    out = extract_per_case(TASK, raw)
    assert out[0] == "A"
    assert out[1] is None


def test_extract_per_case_generative_single():
    gt = {"id": "G1", "dimension": "语言能力", "type": "生成式", "prompt": "p",
          "test_cases": []}
    assert extract_per_case(gt, "全文内容") == ["全文内容"]
