# -*- coding: utf-8 -*-
"""双盲评审官：将双方答卷随机打乱为答案X/答案Y，调用评审模型逐题评分，
按分差判定 winner（tie/answer_x/answer_y），分差 ≤ 1 需仲裁备注，
最终写入 verdict 结构（与 .eval/verdict.json 兼容）。

迭代二：revealed 可显式注入（与人工评审 reveal 打通，缺省仍随机打乱）、
评审网络走 SSRF 校验客户端、答案长度围栏（超长截断避免 prompt 爆炸）、
judging SSE 进度回调。评审模型调用 OpenAI 兼容 API（与 executor 相同协议）。
"""
from __future__ import annotations

import json
import random
import re
import time
from typing import Any, Awaitable, Callable

import httpx

from backend.ssrf import build_upstream_client

# 送审答案长度围栏：超长截断并在 prompt 中标注（防 prompt 爆炸，保证可审性）
ANSWER_FENCE = 8000


def _fenced(raw: str | None) -> str:
    """答案围栏：超长截断并标注；空值显示占位符。"""
    if not raw:
        return "(无回答)"
    if len(raw) <= ANSWER_FENCE:
        return raw
    return raw[:ANSWER_FENCE] + "\n\n……（答案过长已截断，完整内容见答卷文件）"


async def _call_judge_model(
    client: httpx.AsyncClient,
    judge_config: dict[str, str],
    prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> str | None:
    url = judge_config["url"].rstrip("/") + "/chat/completions"
    api_key = judge_config["key"]
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if judge_config.get("name"):
        payload["model"] = judge_config["name"]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        resp = await client.post(url, json=payload, headers=headers, timeout=180)
        body = resp.json()
        if resp.status_code >= 400:
            return None
        return body["choices"][0]["message"]["content"] or ""
    except Exception:
        return None


def _fmt_code_verify(cv: dict) -> str:
    """格式化代码验真结果：未执行/异常时明确标注，避免误读为 0/N。"""
    if not isinstance(cv, dict):
        return "未执行（已禁用）"
    if cv.get("status") != "run":
        if cv.get("status") == "error":
            return f"执行异常（可能已部分执行）：{cv.get('reason') or '未知错误'}"
        return f"未执行（{cv.get('reason') or '已禁用'}）"
    return f"{cv.get('passed', '?')}/{cv.get('total', '?')}"


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
"""
    ctx = (task.get("context") or "").strip()
    if ctx:
        prompt += f"""
【参考文档（题目携带的上下文材料，供核对答案忠实性）】
{ctx}
"""
    prompt += f"""
【评分标准】
{rubric_note}
"""
    if test_cases:
        prompt += "\n【测试用例参考（已验证）】\n"
        for i, tc in enumerate(test_cases):
            prompt += f"  用例{i+1}：输入={tc['input']}，期望={tc['expected']}\n"

    prompt += f"""
【代码验真结果】
答案X 通过率：{_fmt_code_verify(code_verify_x)}
答案Y 通过率：{_fmt_code_verify(code_verify_y)}

【答案X】
{_fenced(answer_x.get('raw_answer'))}

【答案Y】
{_fenced(answer_y.get('raw_answer'))}
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


def _file_label(v: Any) -> str:
    """revealed 标签归一化：支持 "a"/"b"、文件全名 "answers-a.json" 等形式。"""
    s = str(v or "")
    if s in ("a", "b"):
        return s
    if "answers-a" in s or "a.json" in s:
        return "a"
    return "b"


def make_task_reveal(task_ids: list[str], seed: int | None = None) -> dict[str, Any]:
    """逐题独立随机交换（迭代三，仅 agent 评审）：每题独立决定 答案X 对应
    答卷 a 或 b，消除位置偏差。

    Returns:
        {"rounds": [{"answer_x": "a", "answer_y": "b"}],   # 轮级兜底（取首题）
         "per_task": [{"task_id": "...", "answer_x": "a", "answer_y": "b"}, ...]}
    """
    rng = random.Random(seed)
    per_task = []
    for tid in task_ids:
        x = "a" if rng.random() < 0.5 else "b"
        per_task.append({"task_id": tid, "answer_x": x, "answer_y": "b" if x == "a" else "a"})
    first = per_task[0] if per_task else {"answer_x": "a", "answer_y": "b"}
    return {"rounds": [{"answer_x": first["answer_x"], "answer_y": first["answer_y"]}],
            "per_task": per_task}


def _normalize_task_reveal(revealed: dict[str, Any] | None) -> dict[str, str] | None:
    """把注入 revealed 的 per_task 归一化为 {task_id: "a"|"b"}；缺失返回 None。"""
    if not isinstance(revealed, dict):
        return None
    per_task = revealed.get("per_task")
    if not isinstance(per_task, list) or not per_task:
        return None
    mapping: dict[str, str] = {}
    for item in per_task:
        if not isinstance(item, dict):
            continue
        tid = item.get("task_id")
        label = _file_label(item.get("answer_x"))
        if tid:
            mapping[str(tid)] = label
    return mapping or None


def _normalize_reveal(revealed: dict[str, Any] | None) -> tuple[str, str] | None:
    """归一化外部注入的 revealed（支持 human_review 的 rounds 结构与直接映射）。"""
    if not isinstance(revealed, dict):
        return None
    r = revealed.get("rounds", [{}])[0] if revealed.get("rounds") else revealed
    x, y = r.get("answer_x"), r.get("answer_y")
    if x is None and y is None:
        return None
    return _file_label(x) or "a", _file_label(y) or "b"


async def run_judge(
    task_set: dict[str, Any],
    answers_x: dict[str, Any],
    answers_y: dict[str, Any],
    judge_config: dict[str, str],
    revealed: dict[str, Any] | None = None,
    progress_cb: Callable[[int, int], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    """对全部任务集执行双盲评审，返回完整 verdict 结构。

    - 随机打乱 X/Y 对应关系（答案X/答案Y身份随机）；revealed 非 None 时
      显式注入身份映射（与人工评审 reveal 打通，人工/agent 评审结果可对齐）
    - 逐题评分（送审答案带长度围栏），每审完一题回调 progress_cb(done, total)
    - 统计 totals / per_dimension / revealed / meta
    - 网络统一走 build_upstream_client()（SSRF 校验）
    """
    tasks = task_set["tasks"]
    answers_list_x = {a["id"]: a for a in answers_x["answers"]}
    answers_list_y = {a["id"]: a for a in answers_y["answers"]}

    # 决定哪份答卷是答案X，哪份是答案Y
    injected = _normalize_reveal(revealed)
    task_map = _normalize_task_reveal(revealed)
    if injected is not None:
        x_label, y_label = injected
    else:
        if random.random() < 0.5:
            x_label, y_label = "a", "b"
        else:
            x_label, y_label = "b", "a"

    revealed_map = {
        "answer_x": f"answers-{x_label}.json",
        "answer_y": f"answers-{y_label}.json",
        "answer_x_file": x_label,
        "answer_y_file": y_label,
    }

    verdicts: list[dict[str, Any]] = []
    async with build_upstream_client() as client:
        for i, task in enumerate(tasks):
            tid = task["id"]
            # 迭代三：逐题独立随机交换（per_task 注入时每题按题选池，消除位置偏差）；
            # 无题级映射时回退轮级标签
            task_x = task_map.get(tid, x_label) if task_map else x_label
            if task_x == "a":
                x_pool, y_pool = (answers_list_x, answers_list_y)
            else:
                x_pool, y_pool = (answers_list_y, answers_list_x)
            # 效率稳定性题有 repeat_1/repeat_2 两条，用 repeat_1 送审
            ax = x_pool.get(tid) or x_pool.get(f"{tid}_repeat_1", {"raw_answer": "", "api_info": {}})
            ay = y_pool.get(tid) or y_pool.get(f"{tid}_repeat_1", {"raw_answer": "", "api_info": {}})
            v = await judge_task(client, task, ax, ay, judge_config)
            verdicts.append(v)
            if progress_cb:
                await progress_cb(i + 1, len(tasks))

    # 统计
    valid = [v for v in verdicts if not v.get("_invalid")]
    invalid_count = len(verdicts) - len(valid)
    tie_arbitrated = sum(
        1 for v in valid
        if v.get("winner") == "tie" and v.get("arbiter_note")
    )

    # 按维度汇总（排除不计分题：excluded_from_total 的任务打分仅作展示，
    # 不参与 totals / 胜负 / per_dimension 聚合；判题仍逐题执行）
    scoring_verdicts = [
        v for v in verdicts
        if not _task_flag(task_set, v.get("id"), "excluded_from_total")
    ]
    dim_totals: dict[str, dict[str, float]] = {}
    for v in scoring_verdicts:
        dim = v.get("dimension", "")
        if dim not in dim_totals:
            dim_totals[dim] = {"x": 0, "y": 0}
        dim_totals[dim]["x"] += v.get("answer_x", 0)
        dim_totals[dim]["y"] += v.get("answer_y", 0)

    total_x = sum(d["x"] for d in dim_totals.values())
    total_y = sum(d["y"] for d in dim_totals.values())

    excluded_ids = [
        v["id"] for v in verdicts
        if _task_flag(task_set, v.get("id"), "excluded_from_total")
    ]
    excluded_dims = sorted({v.get("dimension", "") for v in verdicts
                            if _task_flag(task_set, v.get("id"), "excluded_from_total")})

    return {
        "meta": {
            "total": len(verdicts),
            "valid": len(valid),
            "invalid": invalid_count,
            "tie_arbitrated": tie_arbitrated,
            "excluded_ids": excluded_ids,
            "excluded_dimensions": excluded_dims,
        },
        "scores": verdicts,
        "per_dimension": dim_totals,
        "totals": {"answer_x": total_x, "answer_y": total_y},
        "revealed": revealed_map,
        "conclusion": _build_conclusion(valid, dim_totals, total_x, total_y),
        "winner_model": answers_x["model"] if total_x > total_y else (
            answers_y["model"] if total_y > total_x else "tie"
        ),
    }


def _task_flag(task_set: dict[str, Any], task_id: Any, key: str) -> bool:
    """按任务 id 从任务集取布尔标记（excluded_from_total 等），缺失一律 False。"""
    if not task_id:
        return False
    for t in task_set.get("tasks", []):
        if t.get("id") == task_id:
            return bool(t.get(key, False))
    return False


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


def _build_single_arm_prompt(task: dict[str, Any], answer: dict[str, Any]) -> str:
    """单臂 rubric 评审 prompt（迭代三，N 模型评审协议层）：单答案独立评分。

    与双盲评审共用评分标准（rubric_note），输出结构化 JSON 打分协议。
    """
    dim = task["dimension"]
    rubric_note = task.get("rubric_note", "")
    test_cases = task.get("test_cases", [])
    code_verify = answer.get("code_verify", {})

    prompt = f"""你是一个严格、客观、公平的 AI 评测评审官。
你将看到一道评测题、该题的评分标准（rubric），以及一个被测模型的回答。
请依据 rubric 独立评分，不要与其他模型比较。

请严格按以下 JSON 格式输出，不要输出任何其他文字：
{{"id":"{task['id']}","dimension":"{dim}","score":<0-10分>,"basis":"详细评分依据"}}

---
【评测题 id={task['id']}】
维度：{dim}
题目：
{task['prompt']}
"""
    ctx = (task.get("context") or "").strip()
    if ctx:
        prompt += f"""
【参考文档（题目携带的上下文材料，供核对答案忠实性）】
{ctx}
"""
    prompt += f"""
【评分标准】
{rubric_note}
"""
    if test_cases:
        prompt += "\n【测试用例参考（已验证）】\n"
        for i, tc in enumerate(test_cases):
            prompt += f"  用例{i+1}：输入={tc['input']}，期望={tc['expected']}\n"

    prompt += f"""
【代码验真结果】
通过率：{_fmt_code_verify(code_verify)}

【回答】
{_fenced(answer.get('raw_answer'))}
"""
    return prompt


def _parse_single_arm_verdict(raw: str) -> dict[str, Any] | None:
    """单臂评审输出结构校验：score 0-10 数值、basis 非空、id 与题一致由调用方核对。"""
    parsed = _parse_verdict(raw)
    if not parsed:
        return None
    score = parsed.get("score")
    if not isinstance(score, (int, float)) or not (0 <= float(score) <= 10):
        return None
    if not isinstance(parsed.get("basis"), str) or not parsed["basis"].strip():
        return None
    return {
        "id": str(parsed.get("id", "")),
        "dimension": str(parsed.get("dimension", "")),
        "score": round(float(score), 2),
        "basis": parsed["basis"].strip(),
    }


async def run_single_arm_judge(
    task_set: dict[str, Any],
    answers: dict[str, Any],
    judge_config: dict[str, str],
    progress_cb: Callable[[int, int], Awaitable[None]] | None = None,
    max_retries: int = 1,
) -> dict[str, Any]:
    """单臂 rubric 评审（迭代三，评审协议层，不接主流程；迭代 7 benchmark 接线）。

    每题对单个模型独立评分（无 X/Y 配对，成本 O(N)），金标锚定由元评估
    段经 stats.calibration_offset 计算展示，本函数不做自动校正。

    Returns:
        {"meta": {"total", "valid", "invalid", "excluded_ids", "excluded_dimensions"},
         "scores": [{"id", "dimension", "score", "basis", "_invalid"}...],
         "totals": {"score": 汇总分, "max": 满分}, "health": {...}}
    """
    tasks = task_set["tasks"]
    answers_map = {a["id"]: a for a in answers.get("answers", [])}
    verdicts: list[dict[str, Any]] = []
    async with build_upstream_client() as client:
        for i, task in enumerate(tasks):
            tid = task["id"]
            answer = (answers_map.get(tid) or answers_map.get(f"{tid}_repeat_1")
                      or {"raw_answer": "", "api_info": {}})
            prompt = _build_single_arm_prompt(task, answer)
            v = None
            for _attempt in range(1 + max_retries):
                raw = await _call_judge_model(client, judge_config, prompt)
                v = _parse_single_arm_verdict(raw)
                if v:
                    break
            if v is None:
                v = {
                    "id": tid,
                    "dimension": task.get("dimension", ""),
                    "score": 0.0,
                    "basis": "评审模型未能返回有效 verdict",
                    "_invalid": True,
                }
            verdicts.append(v)
            if progress_cb:
                await progress_cb(i + 1, len(tasks))

    valid = [v for v in verdicts if not v.get("_invalid")]
    excluded_ids = [t["id"] for t in tasks if t.get("excluded_from_total")]
    scoring_scores = [v for v in verdicts if v["id"] not in excluded_ids]
    return {
        "meta": {
            "total": len(verdicts),
            "valid": len(valid),
            "invalid": len(verdicts) - len(valid),
            "excluded_ids": excluded_ids,
            "excluded_dimensions": sorted({t.get("dimension", "") for t in tasks
                                           if t.get("excluded_from_total")}),
        },
        "scores": verdicts,
        "totals": {
            "score": round(sum(s.get("score", 0) for s in scoring_scores), 2),
            "max": round(10 * len(scoring_scores), 2),
        },
        "health": health_check(verdicts),
    }


def health_check(verdicts: list[dict[str, Any]] | dict[str, Any],
                 threshold: float = 0.1) -> dict[str, Any]:
    """评审健康度（迭代三）：invalid 率阈值判定。

    入参可为 verdicts 列表（自带 _invalid 标记）或 run_judge/单臂输出的
    meta dict（含 total/invalid）。
    Returns:
        {"healthy": bool, "invalid_rate": float, "alarm": bool, "threshold": float}
    """
    if isinstance(verdicts, dict) and "total" in verdicts:
        total = int(verdicts.get("total", 0))
        invalid = int(verdicts.get("invalid", 0))
    elif isinstance(verdicts, list):
        total = len(verdicts)
        invalid = sum(1 for v in verdicts if v.get("_invalid"))
    else:
        total = invalid = 0
    rate = round(invalid / total, 4) if total else 0.0
    return {
        "healthy": rate <= threshold,
        "invalid_rate": rate,
        "alarm": rate > threshold,
        "threshold": threshold,
    }
