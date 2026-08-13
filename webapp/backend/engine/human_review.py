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
        "context": (task.get("context") or "").strip(),
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
    """把一轮的人工打分转成完整 verdict（revealed 含模型身份，评审结束才揭示）。

    不计分题（excluded_from_total）仍逐题记录分数供展示，但不计入
    totals / per_dimension / 胜负判定（安全与价值观维度）。
    """
    score_by_id = {s["id"]: s for s in round_scores}
    excluded_ids = {
        t["id"] for t in task_set["tasks"] if t.get("excluded_from_total")
    }
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
        if sid not in excluded_ids:
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

    excluded_dims = sorted({t.get("dimension", "") for t in task_set["tasks"]
                            if t.get("excluded_from_total")})

    return {
        "meta": {
            "total": len(scores),
            "valid": sum(1 for s in scores if not s["_invalid"]),
            "invalid": sum(1 for s in scores if s["_invalid"]),
            "tie_arbitrated": 0,
            "excluded_ids": sorted(excluded_ids),
            "excluded_dimensions": excluded_dims,
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


def _task_x_file(round_verdict: dict[str, Any], task_id: str,
                  per_task_reveal: dict[str, str] | None) -> str:
    """取单题的 answer_x 文件标签：题级 reveal 优先（迭代三，仅 agent），
    无 per_task 时回退轮级字段（人工评审与旧数据路径，行为不变）。"""
    if per_task_reveal and task_id in per_task_reveal:
        return per_task_reveal[task_id]
    return round_verdict.get("revealed", {}).get("answer_x_file", "a")


def _stable_scores(
    round_verdict: dict[str, Any],
    per_task_reveal: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """把一轮 verdict 的 X/Y 分数按该轮 reveal 归一化为稳定模型分数。

    每轮的 X/Y 身份独立随机（answer_x_file 可为 "a" 或 "b"；迭代三 agent
    评审可细到逐题 per_task_reveal），跨轮统计必须以此处归一化后的
    model_a/model_b 为主键。

    Returns:
        [{"id": ..., "dimension": ..., "model_a": float, "model_b": float}, ...]
    """
    rows = []
    for s in round_verdict.get("scores", []):
        if _task_x_file(round_verdict, s.get("id"), per_task_reveal) == "a":
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


def _stable_names(
    round_verdict: dict[str, Any],
    per_task_reveal: dict[str, str] | None = None,
) -> dict[str, str]:
    """从一轮 reveal 反查稳定模型名：{"a": 模型A名, "b": 模型B名}。

    逐题模式下不同题可映射到不同模型文件；取该轮首题映射（展示用），
    聚合统计以 _stable_scores 按题归一化为准。
    """
    first_id = None
    for s in round_verdict.get("scores", []):
        first_id = s.get("id")
        break
    x_file = _task_x_file(round_verdict, first_id, per_task_reveal) if first_id \
        else round_verdict.get("revealed", {}).get("answer_x_file", "a")
    revealed = round_verdict.get("revealed", {})
    if x_file == "a":
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
    per_task_reveal: dict[str, str] | None = None,
    per_round_reveal: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """多轮聚合（repeat_n>1 时取平均分与标准差）；单轮直接返回。

    跨轮统计始终以稳定模型（model_a/model_b）为主键：每轮先按该轮
    reveal 归一化，再聚合均值/标准差/胜负/维度汇总；胜方在稳定空间
    判定后，用固定的展示映射（最后一轮 reveal）把结果投影回
    answer_x/answer_y 展示字段，保证报告与前端无需感知身份变化。

    迭代三：per_round_reveal=[{task_id: "a"|"b"}...]（与 round_verdicts
    逐轮对齐的逐题 reveal，仅 agent 评审）优先；缺省回退 per_task_reveal
    （全员同映射）或轮级 reveal（人工/旧数据，零破坏）。
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
    last_per_task = None
    if per_round_reveal and len(per_round_reveal) == len(round_verdicts):
        last_per_task = per_round_reveal[-1]
    names = _stable_names(last, last_per_task or per_task_reveal)
    stable_map: dict[str, list[dict]] = {}
    for i, v in enumerate(round_verdicts):
        round_map = None
        if per_round_reveal and i < len(per_round_reveal):
            round_map = per_round_reveal[i]
        for r in _stable_scores(v, round_map or per_task_reveal):
            stable_map.setdefault(r["id"], []).append(r)

    # 不计分题（安全与价值观维度）：仍保留平均分展示，但不计入总分/胜负
    excluded_ids = set(round_verdicts[0].get("meta", {}).get("excluded_ids", []))
    excluded_dims = sorted(set(round_verdicts[0].get("meta", {}).get("excluded_dimensions", [])))

    avg_scores: list[dict[str, Any]] = []
    dim_totals: dict[str, dict[str, float]] = {}
    for tid, round_scores in stable_map.items():
        a_vals = [r["model_a"] for r in round_scores]
        b_vals = [r["model_b"] for r in round_scores]
        a_mean = statistics.mean(a_vals)
        b_mean = statistics.mean(b_vals)
        dim = round_scores[0]["dimension"]
        if tid not in excluded_ids:
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
            "excluded_ids": sorted(excluded_ids),
            "excluded_dimensions": excluded_dims,
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


def merge_hybrid_verdicts(
    round_verdicts: list[dict[str, Any]],
    human_scores: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """hybrid 复核（迭代三）：按 (round, task_id) 用人工分覆盖 agent verdict。

    - 人工为准：answer_x/answer_y/winner/basis（basis=人工备注或默认文案）、
      arbiter_note 清空、_invalid=False；未覆盖题原样保留 agent 分
    - 覆盖后重算该轮 per_dimension / totals / meta（与 run_judge 同口径：
      excluded_from_total 不计入聚合）
    - 纯函数：不修改入参，返回新结构供 build_final_verdict 聚合
    """
    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for h in human_scores:
        r = int(h.get("round", 0))
        tid = str(h.get("id", ""))
        by_key[(r, tid)] = h

    out: list[dict[str, Any]] = []
    for rv in round_verdicts:
        scores: list[dict[str, Any]] = []
        for s in rv.get("scores", []):
            row = dict(s)
            h = by_key.get((int(row.get("round", 1)), str(row.get("id", ""))))
            if h is not None:
                x = float(h.get("answer_x", 0))
                y = float(h.get("answer_y", 0))
                note = str(h.get("note", "")).strip()
                row["answer_x"] = round(x, 2)
                row["answer_y"] = round(y, 2)
                row["winner"] = (
                    "tie" if abs(x - y) <= EPS
                    else ("answer_x" if x > y else "answer_y")
                )
                row["basis"] = note or "人工复核打分（覆盖 AI 评审）"
                row["arbiter_note"] = ""
                row["_invalid"] = False
                row["reviewed"] = True
            else:
                row.pop("reviewed", None)
            scores.append(row)

        excluded_ids = set(rv.get("meta", {}).get("excluded_ids", []))
        dim_totals: dict[str, dict[str, float]] = {}
        for sc in scores:
            if sc.get("id") in excluded_ids:
                continue
            dim = sc.get("dimension", "")
            if dim not in dim_totals:
                dim_totals[dim] = {"x": 0.0, "y": 0.0}
            dim_totals[dim]["x"] += float(sc.get("answer_x", 0))
            dim_totals[dim]["y"] += float(sc.get("answer_y", 0))
        total_x = round(sum(d["x"] for d in dim_totals.values()), 2)
        total_y = round(sum(d["y"] for d in dim_totals.values()), 2)
        valid = [sc for sc in scores if not sc.get("_invalid")]
        revealed = rv.get("revealed", {})
        if total_x > total_y:
            winner_model = revealed.get("answer_x", "answer_x")
        elif total_y > total_x:
            winner_model = revealed.get("answer_y", "answer_y")
        else:
            winner_model = "tie"

        merged_meta = dict(rv.get("meta", {}))
        merged_meta.update({
            "total": len(scores),
            "valid": len(valid),
            "invalid": len(scores) - len(valid),
        })
        out.append({
            **rv,
            "meta": merged_meta,
            "scores": scores,
            "per_dimension": {d: {"x": round(v["x"], 2), "y": round(v["y"], 2)}
                              for d, v in dim_totals.items()},
            "totals": {"answer_x": total_x, "answer_y": total_y},
            "winner_model": winner_model,
            "conclusion": f"第{int(scores[0].get('round', 1))}轮人工复核覆盖完成"
                          if scores else rv.get("conclusion", ""),
        })
    return out
