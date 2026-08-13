# -*- coding: utf-8 -*-
"""D3 context 注入（迭代四）：执行侧 / 评审侧 / 人工评审视图三处透传。

- executor.build_prompt：任一策略（direct/cot/fewshot）下 context 均以
  「【参考文档】…【任务】」块置于题面之前；无 context 行为不变（回归）。
- judge 双盲/单臂 prompt：携带 context 时追加参考文档段（供核对忠实性）。
- human_review.build_review_view：评审视图透出 context 字段（前端渲染）。
"""
from __future__ import annotations

from backend.engine.executor import build_prompt
from backend.engine.judge import _build_blind_prompt, _build_single_arm_prompt
from backend.engine.human_review import build_review_view

DOC = "某参考文档内容：恒星核心温度可达1500万开尔文，白矮星依靠电子简并压支撑。"
PROMPT_TEXT = "请仅依据参考文档回答：太阳的演化终点是什么？"
TASK = {
    "id": "T9",
    "dimension": "知识能力",
    "prompt": PROMPT_TEXT,
    "expected": "白矮星",
    "rubric_note": "满分10分：准确引用文档为高分档。",
    "type": "生成式",
    "context": DOC,
}
NO_CTX = {**TASK, "context": ""}


def _ans(text="白矮星依靠电子简并压支撑（文档原文）。"):
    return {"raw_answer": text, "api_info": {"status": "ok"}}


def test_build_prompt_direct_injects_context():
    out = build_prompt(TASK, "direct")
    assert out.startswith("【参考文档】")
    assert DOC in out
    assert "【任务】" in out
    assert PROMPT_TEXT in out


def test_build_prompt_cot_and_fewshot_inject_context():
    cot = build_prompt(TASK, "cot")
    assert "【参考文档】" in cot and "请一步一步思考" in cot
    fs = build_prompt({**TASK, "test_cases": [
        {"input": "i", "expected": "e"}]}, "fewshot")
    assert "【参考文档】" in fs and "示例：" in fs


def test_build_prompt_no_context_unchanged():
    assert build_prompt(NO_CTX, "direct") == PROMPT_TEXT
    assert build_prompt(NO_CTX, "cot") == PROMPT_TEXT + "\n\n请一步一步思考，并在最后给出明确、独立的结论。"
    assert build_prompt(NO_CTX, "fewshot") == PROMPT_TEXT


def test_build_prompt_empty_context_treated_as_missing():
    out = build_prompt({**TASK, "context": "   "}, "direct")
    assert out == PROMPT_TEXT


def test_judge_blind_prompt_carries_context():
    p = _build_blind_prompt(TASK, _ans(), _ans("不同答案"))
    assert "【参考文档（题目携带的上下文材料，供核对答案忠实性）】" in p
    assert DOC in p
    assert "【评分标准】" in p


def test_judge_blind_prompt_no_context():
    p = _build_blind_prompt(NO_CTX, _ans(), _ans())
    assert "【参考文档（题目携带的上下文材料" not in p


def test_judge_single_arm_carries_context():
    p = _build_single_arm_prompt(TASK, _ans())
    assert "【参考文档（题目携带的上下文材料，供核对答案忠实性）】" in p
    assert DOC in p


def test_review_view_carries_context():
    v = build_review_view(TASK, [_ans()], [_ans()])
    assert v["context"] == DOC
    assert v["prompt"] == PROMPT_TEXT
    v2 = build_review_view(NO_CTX, [_ans()], [_ans()])
    assert v2["context"] == ""