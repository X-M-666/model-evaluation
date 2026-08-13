# -*- coding: utf-8 -*-
"""统计显著性（迭代二）：胜率、配对 bootstrap 置信区间、「差异不显著」提示。

全部纯函数（随机源可注入 seed），确定性可测。
注意：调用方须先按 scoring_ids 过滤（排除 excluded_from_total 不计分题），
再传入本模块计算，否则安全与价值观维度会污染胜率/显著性。
"""
from __future__ import annotations

import random
from typing import Any

# 显著性所需最小题数（与迭代计划「题数不足标注」对齐）
MIN_SAMPLE = 8

DEFAULT_N_BOOT = 1000


def paired_bootstrap_ci(
    x_scores: list[float],
    y_scores: list[float],
    seed: int | None = None,
    n_boot: int = DEFAULT_N_BOOT,
) -> tuple[float, float]:
    """配对 bootstrap 均值差置信区间（对每题采样 Δ=x−y 的均值分布）。

    返回 (ci_lo, ci_hi) 为 95% 分位；空输入返回 (0.0, 0.0)。
    """
    pairs = list(zip(x_scores, y_scores))
    n = len(pairs)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    deltas = [a - b for a, b in pairs]
    means: list[float] = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += deltas[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[max(0, int(n_boot * 0.025))]
    hi = means[min(n_boot - 1, int(n_boot * 0.975))]
    return (round(lo, 3), round(hi, 3))


def win_rate(x_scores: list[float], y_scores: list[float]) -> dict[str, Any]:
    """逐题胜/平/负统计（含占比）。"""
    wins = ties = losses = 0
    for a, b in zip(x_scores, y_scores):
        if a > b:
            wins += 1
        elif a < b:
            losses += 1
        else:
            ties += 1
    total = wins + ties + losses
    return {
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "total": total,
        "win_rate": round(wins / total, 4) if total else 0.0,
        "lose_rate": round(losses / total, 4) if total else 0.0,
    }


def significance_note(
    x_scores: list[float],
    y_scores: list[float],
    seed: int | None = None,
    n_boot: int = DEFAULT_N_BOOT,
) -> dict[str, Any]:
    """显著性结论：CI 含 0（样本充足）或样本不足时标注「差异不显著」。

    Returns:
        {"significant": bool, "reason": "ci_overlaps_zero"|"insufficient_sample"|"significant",
         "ci": [lo, hi], "sample": n, "note": 展示文案}
    """
    x = list(x_scores)
    y = list(y_scores)
    n = min(len(x), len(y))
    ci = paired_bootstrap_ci(x, y, seed=seed, n_boot=n_boot)
    if n < MIN_SAMPLE:
        reason = "insufficient_sample"
        significant = False
        note = (
            f"样本仅 {n} 题，不足 {MIN_SAMPLE} 题最小阈值，"
            f"均值差 95% 置信区间 [{ci[0]}, {ci[1]}]，差异不显著。"
        )
    elif ci[0] <= 0 <= ci[1]:
        reason = "ci_overlaps_zero"
        significant = False
        note = (
            f"均值差 95% 置信区间 [{ci[0]}, {ci[1]}] 包含 0，"
            f"两模型差异不显著（n={n} 题）。"
        )
    else:
        reason = "significant"
        significant = True
        direction = "X 显著更高" if sum(x) > sum(y) else "Y 显著更高"
        note = (
            f"均值差 95% 置信区间 [{ci[0]}, {ci[1]}] 不含 0，"
            f"{direction}（n={n} 题）。"
        )
    return {
        "significant": significant,
        "reason": reason,
        "ci": [ci[0], ci[1]],
        "sample": n,
        "note": note,
    }
