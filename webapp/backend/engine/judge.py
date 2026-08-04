# -*- coding: utf-8 -*-
"""双盲评审官：将双方答卷随机打乱为答案X/答案Y，调用评审模型逐题评分，
按分差判定 winner（tie/answer_x/answer_y），分差 ≤ 1 需仲裁备注，
最终写入 verdict 结构（与 .eval/verdict.json 兼容）。

评审模型调用 OpenAI 兼容 API（与 executor 相同协议）。
"""
from __future__ import annotations

import json
import random
import re
import time
from typing import Any

import httpx


async def _call_judge_model(
    client: httpx.AsyncClient,
    judge_config: dict[str, str],
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> str | None:
    url = judge_config["url"].rstrip("/")
    api_key = judge_config["key"]
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        resp = await client.post(url, json=payload, headers=headers, timeout=180)
        body = resp.json()
        if resp.status_code >= 400:
            return None
        return body["choices"][0]["message"]["content"] or ""
    except Exception:
        return None


def _build_blind_prompt(
    task: dict[str, Any],
    answer_x: dict[str, Any],
    answer_y: dict[str, Any],
) -> str:
    """为单个任务构造双盲评审 prompt（与 judge agent 定义一致）。"""
    dim = task["dimension"]
    prompt_text = task["prompt"]
    rubric_note = task.get("rubric_note", "")
    test_cases = task.get("test_cases", [])

    code_verify_x = answer_x.get("code_verify", {})
    code_verify_y = answer_y.get("code_verify", {})

    prompt = f"""你是一个严格、客观、公平的 AI 评测双盲评审官。
你将看到一道评测题、该题的评分标准（rubric），以及两个被测模型的回答（分别标记为「答案X」和「答案Y」）。

注意：你不知道答案X和答案Y分别对应哪个模型，这是双盲评审的核心要求。
在写入 verdict 时，只使用「答案X」「答案Y」这两个标签，绝不揭示其对应的真实模型身份。

请严格按以下 JSON 格式输出 verdict，不要输出任何其他文字：
{{"id":"{task['id']}","dimension":"{dim}","answer_x":<X分0-10>,"answer_y":<Y分0-10>,"winner":"tie或answer_x或answer_y","basis":"详细评分依据","arbiter_note":"仲裁备注（分差≤1时必填）"}}

---

【评测题 id={task['id']}】
维度：{dim}
题目：
{prompt_text}

【评分标准】
{rubric_note}
"""
    if test_cases:
        prompt += "\n【测试用例参考（已验证）】\n"
        for i, tc in enumerate(test_cases):
            prompt += f"  用例{i+1}：输入={tc['input']}，期望={tc['expected']}\n"

    prompt += f"""
【代码验真结果】
答案X 通过率：{code_verify_x.get('passed', '?')}/{code_verify_x.get('total', '?')}
答案Y 通过率：{code_verify_y.get('passed', '?')}/{code_verify_y.get('total', '?')}

【答案X】
{answer_x.get('raw_answer', '(无回答)')}

【答案Y】
{answer_y.get('raw_answer', '(无回答)')}
"""
    return prompt


def _parse_verdict(raw: str) -> dict[str, Any] | None:
    """从评审模型输出中提取 JSON verdict。"""
    if not raw:
        return None
    # 去掉可能的 markdown 围栏
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # 尝试提取第一个 {...}
        m = re.search(r"\{[^{}]*\"winner\"[^{}]*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


async def judge_task(
    client: httpx.AsyncClient,
    task: dict[str, Any],
    answer_x: dict[str, Any],
    answer_y: dict[str, Any],
    judge_config: dict[str, str],
    max_retries: int = 1,
) -> dict[str, Any]:
    """评审单个任务，失败重试 max_retries 次。"""
    prompt = _build_blind_prompt(task, answer_x, answer_y)
    verdict = None
    for attempt in range(1 + max_retries):
        raw = await _call_judge_model(client, judge_config, prompt)
        verdict = _parse_verdict(raw)
        if verdict and "winner" in verdict:
            break
    if verdict is None:
        # 评审失败：标记 invalid
        verdict = {
            "id": task["id"],
            "dimension": task["dimension"],
            "answer_x": 0,
            "answer_y": 0,
            "winner": "tie",
            "basis": "评审模型未能返回有效 verdict",
            "arbiter_note": "评审失败，按 tie 处理但标记为 invalid",
            "_invalid": True,
        }
    return verdict


async def run_judge(
    task_set: dict[str, Any],
    answers_x: dict[str, Any],
    answers_y: dict[str, Any],
    judge_config: dict[str, str],
) -> dict[str, Any]:
    """对全部任务集执行双盲评审，返回完整 verdict 结构。

    - 随机打乱 X/Y 对应关系（答案X/答案Y身份随机）
    - 逐题评分
    - 统计 totals / per_dimension / revealed / meta
    """
    tasks = task_set["tasks"]
    answers_list_x = {a["id"]: a for a in answers_x["answers"]}
    answers_list_y = {a["id"]: a for a in answers_y["answers"]}

    # 随机打乱：决定哪份答卷是答案X，哪份是答案Y
    if random.random() < 0.5:
        x_pool, y_pool = answers_list_x, answers_list_y
        revealed = {"answer_x": "answers-a.json", "answer_y": "answers-b.json"}
    else:
        x_pool, y_pool = answers_list_y, answers_list_x
        revealed = {"answer_x": "answers-b.json", "answer_y": "answers-a.json"}

    verdicts: list[dict[str, Any]] = []
    async with httpx.AsyncClient() as client:
        for task in tasks:
            tid = task["id"]
            # 效率稳定性题有 repeat_1/repeat_2 两条，用 repeat_1 送审
            ax = x_pool.get(tid) or x_pool.get(f"{tid}_repeat_1", {"raw_answer": "", "api_info": {}})
            ay = y_pool.get(tid) or y_pool.get(f"{tid}_repeat_1", {"raw_answer": "", "api_info": {}})
            v = await judge_task(client, task, ax, ay, judge_config)
            verdicts.append(v)

    # 统计
    valid = [v for v in verdicts if not v.get("_invalid")]
    invalid_count = len(verdicts) - len(valid)
    tie_arbitrated = sum(
        1 for v in valid
        if v.get("winner") == "tie" and v.get("arbiter_note")
    )

    # 按维度汇总
    dim_totals: dict[str, dict[str, float]] = {}
    for v in valid:
        dim = v.get("dimension", "")
        if dim not in dim_totals:
            dim_totals[dim] = {"x": 0, "y": 0}
        dim_totals[dim]["x"] += v.get("answer_x", 0)
        dim_totals[dim]["y"] += v.get("answer_y", 0)

    total_x = sum(d["x"] for d in dim_totals.values())
    total_y = sum(d["y"] for d in dim_totals.values())

    return {
        "meta": {
            "total": len(verdicts),
            "valid": len(valid),
            "invalid": invalid_count,
            "tie_arbitrated": tie_arbitrated,
        },
        "scores": verdicts,
        "per_dimension": dim_totals,
        "totals": {"answer_x": total_x, "answer_y": total_y},
        "revealed": revealed,
        "conclusion": _build_conclusion(valid, dim_totals, total_x, total_y),
        "winner_model": answers_x["model"] if total_x > total_y else (
            answers_y["model"] if total_y > total_x else "tie"
        ),
    }


def _build_conclusion(
    valid: list[dict],
    dim_totals: dict,
    total_x: float,
    total_y: float,
) -> str:
    if total_x > total_y:
        return f"答案X 以 {int(total_x)}:{int(total_y)} 胜出"
    elif total_y > total_x:
        return f"答案Y 以 {int(total_y)}:{int(total_x)} 胜出"
    else:
        return f"平局 {int(total_x)}:{int(total_y)}"
