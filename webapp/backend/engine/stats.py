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


def clustered_bootstrap_ci(
    cluster_deltas: dict[Any, list[float]],
    seed: int | None = None,
    n_boot: int = DEFAULT_N_BOOT,
) -> tuple[float, float]:
    """聚类稳健 bootstrap 配对均值差置信区间（迭代十二，论文聚类推断落地）。

    以「簇」为单位重采样（bootstrap clusters）：cluster_deltas 为
    {cluster_id: [该簇内逐题配对差 Δ=x−y]}，每次抽出整簇的全部差值参与均值
    计算。簇内题目共享相关性（同数据集/同维度），抽样不能拆散它们；
    该 CI 通常比逐题 bootstrap 更宽（标准误膨胀，伪重复校正）。

    返回 (ci_lo, ci_hi) 95% 分位；无任何簇数据返回 (0.0, 0.0)。
    """
    clusters = [list(v) for v in cluster_deltas.values() if v]
    if not clusters:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(clusters)
    means: list[float] = []
    for _ in range(n_boot):
        sample: list[float] = []
        for _ in range(n):
            sample.extend(rng.choice(clusters))
        means.append(sum(sample) / len(sample) if sample else 0.0)
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


# 基准饱和度（迭代五）：得分率升幅阈值与最少样本（可配常量）
SATURATION_RISE = 0.15
SATURATION_MIN_SAMPLE = 3


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def saturation_trend(
    saturation: dict[str, Any],
    dataset: str | None = None,
    rise: float = SATURATION_RISE,
    min_sample: int = SATURATION_MIN_SAMPLE,
) -> dict[str, Any]:
    """基准饱和度趋势监测（纯函数，迭代五）：历史得分率上升 → 提示题库过时。

    输入为 storage.get_saturation()（jobs: [{job_id, updated_at, dataset?,
    entries: [{id, dimension, type, answer_x, answer_y, winner}]}]）。
    得分率 = 双侧评审分均值 / 10（评审分 0-10；多轮 entries 已是稳定空间均值）。

    每 task 按时间序取得分率序列，前一半均值 vs 后一半均值：
    升幅 ≥ rise 且两侧各 ≥ min_sample 次记录 → 该 task 判「饱和（题库过时）」。
    dataset 非空时仅统计该数据集；否则按数据集分组（无 dataset 记录归 "__all__"）。

    返回：
        {"available", "datasets": {name: {"saturated", "delta", "tasks",
         "note", "per_task"}}, "note"}
        available=False（无历史数据）时 datasets 为空、note 提示待数据积累。
    """
    jobs = saturation.get("jobs", []) if isinstance(saturation, dict) else []
    if not jobs:
        return {"available": False, "datasets": {}, "note": "暂无历史评测数据，饱和度监测待数据积累后生效"}

    ordered = sorted(jobs, key=lambda j: str(j.get("updated_at", "")))

    def _match(ds: str) -> bool:
        if dataset is None:
            return True
        return ds == dataset or ds == "__all__"

    datasets: dict[str, Any] = {}
    names = sorted({str(j.get("dataset") or "__all__") for j in ordered if _match(str(j.get("dataset") or "__all__"))})
    for ds in names:
        per_task: dict[str, list[tuple[str, float]]] = {}
        for j in ordered:
            if str(j.get("dataset") or "__all__") != ds:
                continue
            ts = str(j.get("updated_at", ""))
            for e in j.get("entries", []):
                if not isinstance(e, dict) or e.get("id") is None:
                    continue
                try:
                    rate = (float(e.get("answer_x", 0)) + float(e.get("answer_y", 0))) / 2.0 / 10.0
                except (TypeError, ValueError):
                    continue
                per_task.setdefault(str(e.get("id")), []).append((ts, rate))
        task_results: dict[str, dict[str, Any]] = {}
        for tid, history in per_task.items():
            series = [r for _, r in sorted(history, key=lambda kv: kv[0])]
            n = len(series)
            if n < 2 * min_sample:
                continue
            split = n // 2
            first = _mean(series[:split])
            second = _mean(series[split:])
            delta = second - first
            task_results[tid] = {
                "saturated": delta >= rise,
                "delta": round(delta, 4),
                "samples": n,
                "rate": round(second, 4),
            }
        if not task_results:
            datasets[ds] = {"saturated": False, "delta": 0.0, "tasks": 0,
                            "note": f"样本不足（每个题目至少 {min_sample} 次历史记录方可判定）"}
            continue
        best = max(task_results.values(), key=lambda v: v["delta"])
        saturated_any = any(v["saturated"] for v in task_results.values())
        datasets[ds] = {
            "saturated": saturated_any,
            "delta": best["delta"],
            "tasks": len(task_results),
            "note": "题库可能过时：历史通过率持续上升" if saturated_any
                    else "历史得分率未见持续上升趋势",
            "per_task": task_results,
        }

    if not datasets:
        return {"available": False, "datasets": {}, "note": "暂无满足样本要求的历史数据，饱和度监测待数据积累后生效"}
    return {"available": True, "datasets": datasets,
            "note": "按数据集分组监测历史得分率升幅（阈值 +15%、每任务样本≥3）"}
