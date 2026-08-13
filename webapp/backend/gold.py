# -*- coding: utf-8 -*-
"""金标集（迭代三）：demo 种子初始化 + meta-eval 计算。

- demo 金标：内置题库生成式题（T2E/T4C/T4D 计入总分 + T5D 安全题）的
  人工编写"近似权威分"（0-100），source="demo" 明确标注；manual 录入
  同名覆盖（M3）。
- meta-eval：金标与单次评测的评审分逐题匹配——Spearman 相关、评审 vs
  金标逐题 winner 的 Cohen's Kappa、锚定偏移（金标 0-100 vs 评审 0-10
  映射到 0-100 后比较）。题库 id 调整导致失配 → available=False 空态
  兜底（不报错）。
"""
from __future__ import annotations

from typing import Any

from backend.engine.stats import calibration_offset, cohen_kappa, spearman
from backend import storage

# 评审分 0-10 → 金标 0-100 的映射系数
_GOLD_SCALE = 10.0

# 内置题库生成式题（计入总分）；安全题 T5D 不计总分但可展示
_DEMO_TASKS = ("T2E", "T4C", "T4D")
_DEMO_MODELS = ("GPT-4o", "Claude Sonnet 4")

# 人工编写近似权威分（0-100）：
_DEMO_SCORES: dict[tuple[str, str], float] = {
    # (task_id, model_name) -> score
    ("T2E", "GPT-4o"): 84.0,
    ("T2E", "Claude Sonnet 4"): 90.0,
    ("T4C", "GPT-4o"): 95.0,
    ("T4C", "Claude Sonnet 4"): 88.0,
    ("T4D", "GPT-4o"): 82.0,
    ("T4D", "Claude Sonnet 4"): 92.0,
    ("T5D", "GPT-4o"): 96.0,
    ("T5D", "Claude Sonnet 4"): 94.0,
}


def _demo_items() -> list[dict[str, Any]]:
    return [
        {
            "task_id": tid,
            "model_name": model,
            "score": _DEMO_SCORES[(tid, model)],
            "note": "内置演示金标（人工编写的近似权威分）",
        }
        for tid in _DEMO_TASKS + ("T5D",)
        for model in _DEMO_MODELS
    ]


def ensure_demo_gold() -> None:
    """gold 目录为空时载入 demo 金标（source="demo"）。

    目录已有任何金标集则不写入；manual 录入同名覆盖后不再回退（M3）。
    """
    if storage.list_gold():
        return
    storage.save_gold("demo", {"items": _demo_items(), "source": "demo"})


def _normalize_model_name(name: str) -> str:
    return (name or "").strip()


def compute_meta_eval(verdict: dict[str, Any], task_set: dict[str, Any],
                      gold: dict[str, Any]) -> dict[str, Any]:
    """金标元评估（纯函数，不依赖存储）。

    verdict：最终 verdict（scores 含 model_a/model_b 稳定空间分 0-10，
    或单轮 answer_x/answer_y + revealed；逐题归一化）。
    task_set：任务集（取模型无法得知时用 verdict.revealed 的模型名）。
    gold：金标集 record（items: [{task_id, model_name, score}]）。

    Returns:
        {"available", "spearman", "kappa", "gold_offset", "gold_source",
         "matched", "gold_total", "note"}
        金标与 job 题不匹配或样本不足 → available=False + note 提示。
    """
    items = gold.get("items", []) if isinstance(gold, dict) else []
    gold_map: dict[tuple[str, str], float] = {}
    for it in items:
        tid = str(it.get("task_id", ""))
        mn = _normalize_model_name(it.get("model_name", ""))
        if tid and mn:
            gold_map[(tid, mn)] = float(it.get("score", 0.0))

    scores = verdict.get("scores", [])
    revealed = verdict.get("revealed", {})
    name_x = _normalize_model_name(revealed.get("answer_x", ""))
    name_y = _normalize_model_name(revealed.get("answer_y", ""))

    # 逐题稳定空间分：多轮聚合已有 model_a/model_b；单轮回退 X/Y + reveal 归一化
    matched: list[dict[str, Any]] = []
    for s in scores:
        tid = str(s.get("id", ""))
        if "model_a" in s:
            ma, mb = float(s["model_a"]), float(s["model_b"])
            x_file = revealed.get("answer_x_file", "a")
            na, nb = (name_x, name_y) if x_file == "a" else (name_y, name_x)
        else:
            x_file = revealed.get("answer_x_file", "a")
            if x_file == "a":
                ma, mb = float(s.get("answer_x", 0)), float(s.get("answer_y", 0))
                na, nb = name_x, name_y
            else:
                ma, mb = float(s.get("answer_y", 0)), float(s.get("answer_x", 0))
                na, nb = name_y, name_x
        ga = gold_map.get((tid, na))
        gb = gold_map.get((tid, nb))
        if ga is not None or gb is not None:
            matched.append({
                "task_id": tid,
                "name_a": na, "name_b": nb,
                "review_a": round(ma * _GOLD_SCALE, 2),
                "review_b": round(mb * _GOLD_SCALE, 2),
                "gold_a": ga, "gold_b": gb,
            })

    n_judge = len(matched)
    if not matched:
        return {
            "available": False, "spearman": None, "kappa": None,
            "gold_offset": None, "gold_source": None,
            "matched": 0, "gold_total": len(gold_map),
            "note": "金标集与该任务题目/模型不匹配（题库 id 或模型名不同），无法计算元评估",
        }

    review_pairs: list[tuple[float, float]] = []
    gold_pairs: list[tuple[float, float]] = []
    for m in matched:
        if m["gold_a"] is not None:
            review_pairs.append(m["review_a"])
            gold_pairs.append(float(m["gold_a"]))
        if m["gold_b"] is not None:
            review_pairs.append(m["review_b"])
            gold_pairs.append(float(m["gold_b"]))

    def _winner(a: float | None, b: float | None) -> str:
        if a is None or b is None:
            return "tie"
        eps = 1e-6
        if a - b > eps:
            return "model_a"
        if b - a > eps:
            return "model_b"
        return "tie"

    judge_winners = [_winner(m["review_a"], m["review_b"]) for m in matched]
    gold_winners = [_winner(m["gold_a"], m["gold_b"]) for m in matched]

    base = {
        "available": True,
        "matched": len(matched),
        "gold_total": len(gold_map),
        "gold_source": gold.get("source", "manual"),
        "spearman": round(spearman(review_pairs, gold_pairs), 4)
        if len(review_pairs) >= 3 else None,
        "kappa": round(cohen_kappa(judge_winners, gold_winners,
                                   ["model_a", "tie", "model_b"]), 4)
        if len(matched) >= 2 else None,
        "gold_offset": calibration_offset(gold_pairs, review_pairs)
        if review_pairs else None,
    }
    if len(matched) < 2 or len(review_pairs) < 3:
        base["note"] = "样本不足：部分指标暂缺（完整指标需匹配题数≥2 且分对≥3）"
    else:
        base["note"] = ""
    return base