# -*- coding: utf-8 -*-
"""统计（迭代二/三）：胜率、配对 bootstrap 置信区间、「差异不显著」提示、
秩相关（Spearman）、类别一致性（Cohen's Kappa）、金标锚定偏移、复评一致率。

全部纯函数（随机源可注入 seed），确定性可测。
注意：调用方须先按 scoring_ids 过滤（排除 excluded_from_total 不计分题），
再传入本模块计算，否则安全与价值观维度会污染胜率/显著性。
"""
from __future__ import annotations

import random
import statistics
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


def _rank(values: list[float]) -> list[float]:
    """平均秩（并列取均值）。"""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman 秩相关系数（迭代三：评审分 vs 金标分）。

    并列取平均秩；空或长度不等返回 0.0。|rho| 越接近 1 排序一致性越强。
    """
    if not x or len(x) != len(y):
        return 0.0
    rx = _rank(list(x))
    ry = _rank(list(y))
    mean_x = sum(rx) / len(rx)
    mean_y = sum(ry) / len(ry)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    den_x = sum((a - mean_x) ** 2 for a in rx) ** 0.5
    den_y = sum((b - mean_y) ** 2 for b in ry) ** 0.5
    if den_x == 0 or den_y == 0:
        return 0.0
    return round(num / (den_x * den_y), 4)


def cohen_kappa(
    labels_a: list[Any],
    labels_b: list[Any],
    categories: list[Any] | None = None,
) -> float:
    """Cohen's Kappa（迭代三：评审-评审/评审-金标类别一致性）。

    观察一致率减去期望一致率（按类别边缘概率），再归一化；
    空/长度不等返回 0.0；完全一致=1.0，随机≈0.0，低于随机可为负。
    """
    if not labels_a or len(labels_a) != len(labels_b):
        return 0.0
    cats = categories or sorted(set(labels_a) | set(labels_b))
    if not cats:
        return 0.0
    n = len(labels_a)
    counts = {c: [0, 0] for c in cats}
    observed = 0
    for la, lb in zip(labels_a, labels_b):
        a_key = la if la in cats else None
        b_key = lb if lb in cats else None
        if a_key is not None:
            counts[a_key][0] += 1
        if b_key is not None:
            counts[b_key][1] += 1
        if a_key is not None and a_key == b_key:
            observed += 1
    p_o = observed / n
    p_e = sum((counts[c][0] / n) * (counts[c][1] / n) for c in cats)
    if p_e >= 1.0:
        return 0.0
    return round((p_o - p_e) / (1 - p_e), 4)


def calibration_offset(gold_scores: list[float], judge_scores: list[float]) -> float:
    """金标锚定线性偏移（迭代三，v1）：mean(gold) - mean(judge)。

    正值表示评审系统性偏低（需上调）；偏移仅用于元评估展示，
    本轮不自动校正评审分（复杂校准/分段回归后置）。
    """
    if not gold_scores or len(gold_scores) != len(judge_scores):
        return 0.0
    return round(
        statistics.fmean(gold_scores) - statistics.fmean(judge_scores), 4
    )


def consistency_rate(
    round_winners: list[str],
    categories: tuple[str, ...] = ("model_a", "tie", "model_b"),
) -> float:
    """复评一致率（迭代三）：多轮稳定空间 winner 一致的比例。

    调用方须先把每轮 winner 归一化到稳定模型名（model_a/tie/model_b）；
    单轮/空返回 0.0；n 轮中任意两轮一致的占比（C(n,2) 对）。
    """
    n = len(round_winners)
    if n < 2:
        return 0.0
    agreed = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if round_winners[i] == round_winners[j]:
                agreed += 1
    return round(agreed / total, 4) if total else 0.0
