# -*- coding: utf-8 -*-
"""人工双盲评审模块：X/Y 身份打乱、盲评视图构建、用户打分 → verdict。

与 AI 评审（judge.py）完全解耦：本模块不调用任何评审模型，
仅负责双盲组织、verdict 统计与多轮聚合，保证整个流程免费可离线。
"""
from __future__ import annotations

import random
import statistics
from typing import Any

EPS = 0.001


def make_reveal(rounds: int) -> dict[str, Any]:
    """为每一轮随机决定 答案X/答案Y 对应哪份答卷（a 或 b）。

    Returns:
        {"rounds": [{"answer_x": "a", "answer_y": "b"}, ...]}
    """
    result = []
    for _ in range(max(rounds, 1)):
        if random.random() < 0.5:
            result.append({"answer_x": "a", "answer_y": "b"})
        else:
            result.append({"answer_x": "b", "answer_y": "a"})
    return {"rounds": result}


def _group_by_id(ans_dict: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    """按题目 id 分组答卷条目（稳定性题同 id 有 repeat_index 1/2 两条）。"""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for e in (ans_dict or {}).get("answers", []):
        grouped.setdefault(e["id"], []).append(e)
    return grouped


def resolve_round(
    reveal: dict[str, Any] | None,
    round_idx: int,
    answers_a: dict[str, Any] | None,
    answers_b: dict[str, Any] | None,
) -> tuple[str, str, dict[str, list[dict]], dict[str, list[dict]]]:
    """解析某一轮的 X/Y 身份。

    Returns:
        (x_model, y_model, x_pool, y_pool)
        pool 为 {task_id: [entry, ...]}，answer 内容未暴露模型身份。
    """
    if reveal and "rounds" in reveal and round_idx < len(reveal["rounds"]):
        r = reveal["rounds"][round_idx]
        x_label, y_label = r["answer_x"], r["answer_y"]
    else:
        x_label, y_label = "a", "b"

    pool_a = _group_by_id(answers_a)
    pool_b = _group_by_id(answers_b)
    model_a = (answers_a or {}).get("model", "模型A")
    model_b = (answers_b or {}).get("model", "模型B")
    if x_label == "a":
        return model_a, model_b, pool_a, pool_b
    return model_b, model_a, pool_b, pool_a


def _entry_view(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """把答卷条目转成评审页视图（不含任何模型身份字段）。"""
    main = entries[0] if entries else {}
    api = main.get("api_info", {})
    runs = [e.get("raw_answer", "") for e in entries if e.get("raw_answer")]
    view: dict[str, Any] = {
        "raw_answer": main.get("raw_answer", ""),
        "status": api.get("status", "ok"),
        "latency_ms": api.get("latency_ms", 0),
        "prompt_tokens": api.get("prompt_tokens", 0),
        "completion_tokens": api.get("completion_tokens", 0),
        "truncated": api.get("truncated", False),
        "repeat_index": api.get("repeat_index", 1),
        "code_verify": main.get("code_verify"),
        "repeat_runs": runs if len(runs) > 1 else None,
    }
    return view


def build_review_view(
    task: dict[str, Any],
    x_entries: list[dict[str, Any]],
    y_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """构建单题评审视图：仅题目 + 答案X/答案Y 内容，模型身份完全隐藏。"""
    return {
        "id": task["id"],
        "dimension": task.get("dimension", ""),
        "benchmark": task.get("benchmark", ""),
        "difficulty": task.get("difficulty", ""),
        "prompt": task.get("prompt", ""),
        "rubric_note": task.get("rubric_note", ""),
        "test_cases": task.get("test_cases", []),
        "answer_x": _entry_view(x_entries),
        "answer_y": _entry_view(y_entries),
    }


def build_round_verdict(
    task_set: dict[str, Any],
    round_scores: list[dict[str, Any]],
    round_reveal: dict[str, str],
    x_model: str,
    y_model: str,
    round_idx: int,
) -> dict[str, Any]:
    """把一轮的人工打分转成完整 verdict（revealed 含模型身份，评审结束才揭示）。"""
    score_by_id = {s["id"]: s for s in round_scores}
    scores: list[dict[str, Any]] = []
    dim_totals: dict[str, dict[str, float]] = {}

    for t in task_set["tasks"]:
        sid = t["id"]
        sc = score_by_id.get(sid)
        if sc is None:
            x, y, note, invalid = 0.0, 0.0, "未打分", True
        else:
            x, y = float(sc["answer_x"]), float(sc["answer_y"])
            note = str(sc.get("note", "")).strip()
            invalid = False
        if x - y > EPS:
            winner = "answer_x"
        elif y - x > EPS:
            winner = "answer_y"
        else:
            winner = "tie"
        dim = t.get("dimension", "自定义")
        if dim not in dim_totals:
            dim_totals[dim] = {"x": 0.0, "y": 0.0}
        dim_totals[dim]["x"] += x
        dim_totals[dim]["y"] += y
        scores.append({
            "id": sid, "dimension": dim,
            "answer_x": round(x, 2), "answer_y": round(y, 2),
            "winner": winner,
            "basis": note or "人工双盲评审打分",
            "arbiter_note": "",
            "_invalid": invalid,
            "round": round_idx + 1,
        })

    total_x = round(sum(d["x"] for d in dim_totals.values()), 2)
    total_y = round(sum(d["y"] for d in dim_totals.values()), 2)
    if total_x > total_y:
        winner_model = x_model
    elif total_y > total_x:
        winner_model = y_model
    else:
        winner_model = "tie"

    return {
        "meta": {
            "total": len(scores),
            "valid": sum(1 for s in scores if not s["_invalid"]),
            "invalid": sum(1 for s in scores if s["_invalid"]),
            "tie_arbitrated": 0,
        },
        "scores": scores,
        "per_dimension": dim_totals,
        "totals": {"answer_x": total_x, "answer_y": total_y},
        "revealed": {
            "answer_x": x_model,
            "answer_y": y_model,
            "answer_x_file": round_reveal["answer_x"],
            "answer_y_file": round_reveal["answer_y"],
        },
        "conclusion": f"第{round_idx+1}轮人工评审完成",
        "winner_model": winner_model,
    }


def _stable_scores(round_verdict: dict[str, Any]) -> list[dict[str, Any]]:
    """把一轮 verdict 的 X/Y 分数按该轮 reveal 归一化为稳定模型分数。

    每轮的 X/Y 身份独立随机（answer_x_file 可为 "a" 或 "b"），
    跨轮统计必须以此处归一化后的 model_a/model_b 为主键。

    Returns:
        [{"id": ..., "dimension": ..., "model_a": float, "model_b": float}, ...]
    """
    revealed = round_verdict.get("revealed", {})
    x_file = revealed.get("answer_x_file", "a")
    rows = []
    for s in round_verdict.get("scores", []):
        if x_file == "a":
            model_a, model_b = s["answer_x"], s["answer_y"]
        else:
            model_a, model_b = s["answer_y"], s["answer_x"]
        rows.append({
            "id": s["id"],
            "dimension": s.get("dimension", "自定义"),
            "model_a": model_a,
            "model_b": model_b,
        })
    return rows


def _stable_names(round_verdict: dict[str, Any]) -> dict[str, str]:
    """从一轮 reveal 反查稳定模型名：{"a": 模型A名, "b": 模型B名}。"""
    revealed = round_verdict.get("revealed", {})
    if revealed.get("answer_x_file") == "a":
        return {
            "a": revealed.get("answer_x", "模型A"),
            "b": revealed.get("answer_y", "模型B"),
        }
    return {
        "b": revealed.get("answer_x", "模型A"),
        "a": revealed.get("answer_y", "模型B"),
    }


def build_final_verdict(
    round_verdicts: list[dict[str, Any]],
    repeat_n: int,
) -> dict[str, Any]:
    """多轮聚合（repeat_n>1 时取平均分与标准差）；单轮直接返回。

    跨轮统计始终以稳定模型（model_a/model_b）为主键：每轮先按该轮
    reveal 归一化，再聚合均值/标准差/胜负/维度汇总；胜方在稳定空间
    判定后，用固定的展示映射（最后一轮 reveal）把结果投影回
    answer_x/answer_y 展示字段，保证报告与前端无需感知身份变化。
    """
    if repeat_n <= 1:
        v = round_verdicts[0]
        v["meta"] = {**v.get("meta", {}), "repeat_n": 1}
        for s in v.get("scores", []):
            s.pop("_invalid", None)
            s["answer_x_median"] = s.get("answer_x", 0)
            s["answer_y_median"] = s.get("answer_y", 0)
        v["meta"]["round_reveals"] = [
            {
                "answer_x_file": v.get("revealed", {}).get("answer_x_file", "a"),
                "answer_y_file": v.get("revealed", {}).get("answer_y_file", "b"),
            }
        ]
        return v

    last = round_verdicts[-1]
    names = _stable_names(last)
    stable_map: dict[str, list[dict]] = {}
    for v in round_verdicts:
        for r in _stable_scores(v):
            stable_map.setdefault(r["id"], []).append(r)

    avg_scores: list[dict[str, Any]] = []
    dim_totals: dict[str, dict[str, float]] = {}
    for tid, round_scores in stable_map.items():
        a_vals = [r["model_a"] for r in round_scores]
        b_vals = [r["model_b"] for r in round_scores]
        a_mean = statistics.mean(a_vals)
        b_mean = statistics.mean(b_vals)
        dim = round_scores[0]["dimension"]
        if dim not in dim_totals:
            dim_totals[dim] = {"a": 0.0, "b": 0.0}
        dim_totals[dim]["a"] += a_mean
        dim_totals[dim]["b"] += b_mean
        avg_scores.append({
            "id": tid,
            "dimension": dim,
            "model_a": round(a_mean, 2),
            "model_b": round(b_mean, 2),
            "model_a_std": round(statistics.stdev(a_vals), 2) if len(a_vals) > 1 else 0,
            "model_b_std": round(statistics.stdev(b_vals), 2) if len(b_vals) > 1 else 0,
            "model_a_median": round(statistics.median(a_vals), 2),
            "model_b_median": round(statistics.median(b_vals), 2),
        })

    total_a = round(sum(d["a"] for d in dim_totals.values()), 2)
    total_b = round(sum(d["b"] for d in dim_totals.values()), 2)
    if total_a > total_b:
        winner_model = names["a"]
    elif total_b > total_a:
        winner_model = names["b"]
    else:
        winner_model = "tie"

    x_file = last.get("revealed", {}).get("answer_x_file", "a")

    def _project(avg: dict[str, Any], field_a: str, field_b: str) -> float:
        return avg[field_a] if x_file == "a" else avg[field_b]

    for avg in avg_scores:
        avg["answer_x"] = round(_project(avg, "model_a", "model_b"), 2)
        avg["answer_y"] = round(_project(avg, "model_b", "model_a"), 2)
        avg["answer_x_std"] = round(_project(avg, "model_a_std", "model_b_std"), 2)
        avg["answer_y_std"] = round(_project(avg, "model_b_std", "model_a_std"), 2)
        avg["answer_x_median"] = round(_project(avg, "model_a_median", "model_b_median"), 2)
        avg["answer_y_median"] = round(_project(avg, "model_b_median", "model_a_median"), 2)
        avg["winner"] = "tie" if abs(avg["answer_x"] - avg["answer_y"]) < EPS else (
            "answer_x" if avg["answer_x"] > avg["answer_y"] else "answer_y"
        )
        avg["basis"] = f"{repeat_n}轮人工评审平均（原始{len(stable_map[avg['id']])}轮数据）"
        avg["arbiter_note"] = ""

    per_dimension = {
        d: {
            "answer_x": round(vals["a"] if x_file == "a" else vals["b"], 2),
            "answer_y": round(vals["b"] if x_file == "a" else vals["a"], 2),
        }
        for d, vals in dim_totals.items()
    }
    total_x = round(total_a if x_file == "a" else total_b, 2)
    total_y = round(total_b if x_file == "a" else total_a, 2)
    revealed = last.get("revealed", {})

    return {
        "meta": {
            "total": len(avg_scores),
            "valid": len(avg_scores),
            "invalid": 0,
            "tie_arbitrated": 0,
            "repeat_n": repeat_n,
            "round_reveals": [
                {
                    "answer_x_file": rv.get("revealed", {}).get("answer_x_file", "a"),
                    "answer_y_file": rv.get("revealed", {}).get("answer_y_file", "b"),
                }
                for rv in round_verdicts
            ],
        },
        "scores": avg_scores,
        "per_dimension": per_dimension,
        "totals": {"answer_x": total_x, "answer_y": total_y},
        "revealed": revealed,
        "conclusion": f"经过{repeat_n}轮人工评审取平均",
        "winner_model": winner_model,
    }
