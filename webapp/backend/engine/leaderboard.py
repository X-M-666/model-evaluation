# -*- coding: utf-8 -*-
"""N 模型排行榜聚合（迭代六）：从 N 个已完成 job（同一评测集）抽取每模型
每题得分，生成分维度排名表 + 综合分 + 胜率矩阵 + bootstrap CI + K-召回率 +
箱线/散点/雷达数据。

口径说明：各 job 为双盲成对评审，得分经 reveal 归一化到模型名后跨 job 聚合
（同一模型多 job 时同题 last-wins 合并）；综合分 = 计分题得分合计（满分
10×题数），安全与价值观维度不计入排名。迭代 7 batch + 单臂评审接线后可
替换数据源，聚合逻辑不变。

全部计算确定性（bootstrap seed 可注入），零网络，纯函数可测。
"""
from __future__ import annotations

import statistics
from typing import Any

from backend import storage
from backend.engine.metrics import efficiency_frontier, normalize_efficiency, token_efficiency
from backend.engine.perturb import k_recall_curve
from backend.engine.stats import MIN_SAMPLE, clustered_bootstrap_ci, paired_bootstrap_ci, win_rate


class LeaderboardError(ValueError):
    """排行榜构建错误（任务不存在/未完成/评测集不一致），API 层渲染 400。"""


def _job_entry(files: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    ans = files.get(f"answers-{label}.json")
    if not isinstance(ans, dict):
        return {}
    return {e.get("id"): e for e in ans.get("answers", []) if isinstance(e, dict)}


def extract_job_model_scores(job_id: str) -> dict[str, dict[str, dict[str, Any]]] | None:
    """抽取单个 job 的 {model: {task_id: {score, latency_ms, tokens}}}。

    双盲格式：经 verdict.revealed 归一化 X/Y 侧到真实模型名（answer_x_file/
    answer_y_file 决定 a/b 文件标签）。
    单臂格式（迭代七 batch：verdict 无 revealed）：模型名取 config.model_a.name，
    得分取 scores[].score，延迟/Token 从 answers-a.json 聚合。
    无 verdict/无法解析 → None。
    """
    files = storage.get_job_files(job_id)
    if not files:
        return None
    verdict = files.get("verdict.json")
    if not isinstance(verdict, dict):
        return None
    revealed = verdict.get("revealed") or {}
    if not revealed.get("answer_x") or not revealed.get("answer_y"):
        return _extract_single_arm(files, verdict)
    x_name = revealed.get("answer_x")
    y_name = revealed.get("answer_y")
    x_file = revealed.get("answer_x_file") or "a"
    y_file = revealed.get("answer_y_file") or "b"
    ans_a = _job_entry(files, x_file)
    ans_b = _job_entry(files, y_file)
    out: dict[str, dict[str, dict[str, Any]]] = {x_name: {}, y_name: {}}
    for s in verdict.get("scores", []):
        tid = s.get("id")
        if not tid:
            continue
        entry_x = ans_a.get(tid) or {}
        entry_y = ans_b.get(tid) or {}
        out[x_name][tid] = {
            "score": s.get("answer_x"),
            "latency_ms": (entry_x.get("api_info") or {}).get("latency_ms"),
            "tokens": ((entry_x.get("api_info") or {}).get("prompt_tokens") or 0)
                      + ((entry_x.get("api_info") or {}).get("completion_tokens") or 0),
        }
        out[y_name][tid] = {
            "score": s.get("answer_y"),
            "latency_ms": (entry_y.get("api_info") or {}).get("latency_ms"),
            "tokens": ((entry_y.get("api_info") or {}).get("prompt_tokens") or 0)
                      + ((entry_y.get("api_info") or {}).get("completion_tokens") or 0),
        }
    return out


def _extract_single_arm(
    files: dict[str, Any],
    verdict: dict[str, Any],
) -> dict[str, dict[str, dict[str, Any]]] | None:
    """单臂格式（迭代七 batch 执行单元）：scores[{id,score}] + answers-a。"""
    cfg = files.get("config.json") or {}
    model_name = (cfg.get("model_a") or {}).get("name")
    if not model_name:
        return None
    entries = _job_entry(files, "a")
    out: dict[str, dict[str, dict[str, Any]]] = {model_name: {}}
    for s in verdict.get("scores", []):
        tid = s.get("id")
        if not tid:
            continue
        entry = entries.get(tid) or {}
        out[model_name][tid] = {
            "score": s.get("score"),
            "latency_ms": (entry.get("api_info") or {}).get("latency_ms"),
            "tokens": ((entry.get("api_info") or {}).get("prompt_tokens") or 0)
                      + ((entry.get("api_info") or {}).get("completion_tokens") or 0),
        }
    return out


def _box(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"min": None, "q1": None, "median": None, "q3": None,
                "max": None, "n": 0, "values": []}
    v = sorted(values)
    n = len(v)
    q1 = statistics.median(v[: n // 2]) if n >= 2 else None
    q3 = statistics.median(v[(n + 1) // 2:]) if n >= 2 else None
    return {
        "min": round(v[0], 2),
        "q1": round(q1, 2) if q1 is not None else None,
        "median": round(statistics.median(v), 2),
        "q3": round(q3, 2) if q3 is not None else None,
        "max": round(v[-1], 2),
        "n": n,
        "values": [round(x, 2) for x in v],
    }


def build_leaderboard(
    job_ids: list[str],
    name: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """聚合 N 个已完成 job 构建排行榜数据（结构见报告视图消费约定）。

    Raises:
        LeaderboardError: 任务不存在/未完成/无评审数据/评测集不一致。
    """
    job_ids = [j for j in job_ids if j]
    if not job_ids:
        raise LeaderboardError("job_ids 为空")

    all_scores: dict[str, dict[str, dict[str, Any]]] = {}
    per_job_tasks: dict[str, dict[str, Any]] = {}
    per_job_ids: dict[str, set[str]] = {}
    per_job_extract: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    per_job_dataset: dict[str, str | None] = {}
    dataset = None
    errors: list[str] = []
    for jid in job_ids:
        files = storage.get_job_files(jid)
        if not files:
            errors.append(f"{jid}: 任务不存在")
            continue
        if not isinstance(files.get("report.json"), dict):
            errors.append(f"{jid}: 任务未完成（无报告）")
            continue
        tasks = files.get("tasks.json")
        task_list = tasks.get("tasks", []) if isinstance(tasks, dict) else []
        per_job_tasks[jid] = {t["id"]: t for t in task_list}
        per_job_ids[jid] = set(per_job_tasks[jid])
        if dataset is None:
            dataset = (files.get("config.json") or {}).get("dataset_name")
        per_job_dataset[jid] = (files.get("config.json") or {}).get("dataset_name")
        extracted = extract_job_model_scores(jid)
        if not extracted:
            errors.append(f"{jid}: 无有效评审数据")
            continue
        per_job_extract[jid] = extracted
        for model, per_task in extracted.items():
            bucket = all_scores.setdefault(model, {})
            for tid, entry in per_task.items():
                bucket[tid] = entry  # 同模型多 job 时同题 last-wins
    if errors:
        raise LeaderboardError("；".join(errors))

    base_ids = None
    base_tasks: dict[str, dict[str, Any]] = {}
    for jid in job_ids:
        if base_ids is None:
            base_ids = per_job_ids[jid]
            base_tasks = per_job_tasks[jid]
        elif per_job_ids[jid] != base_ids:
            diff = sorted(base_ids ^ per_job_ids[jid])[:10]
            raise LeaderboardError(
                f"评测集不一致：{jid} 与其它任务的题集差异 {diff}")
    if base_ids is None:
        raise LeaderboardError("无法解析任务集")

    models = sorted(all_scores)
    scoring_ids = [tid for tid in base_tasks
                   if not base_tasks[tid].get("excluded_from_total")]
    excluded_ids = [tid for tid in base_tasks
                    if base_tasks[tid].get("excluded_from_total")]
    dim_of = {tid: t.get("dimension", "自定义") for tid, t in base_tasks.items()}
    dims = sorted({d for d in dim_of.values()})

    # ---- 综合分（计分题合计，满分 10×题数） ----
    composite: dict[str, dict[str, Any]] = {}
    for m in models:
        total = 0.0
        n = 0
        for tid in scoring_ids:
            sc = all_scores[m].get(tid, {}).get("score")
            if sc is not None:
                total += sc
                n += 1
        composite[m] = {"score": round(total, 2), "n_scored": n,
                        "max": round(10.0 * n, 2)}

    ranking = sorted(models, key=lambda m: composite[m]["score"], reverse=True)
    ranks = {m: i + 1 for i, m in enumerate(ranking)}

    # ---- 分维度排名表（每维度各模型合计分） ----
    per_dim: dict[str, dict[str, float]] = {}
    for d in dims:
        tids = [tid for tid, dd in dim_of.items() if dd == d]
        row: dict[str, float] = {}
        for m in models:
            total = 0.0
            for tid in tids:
                sc = all_scores[m].get(tid, {}).get("score")
                if sc is not None:
                    total += sc
            row[m] = round(total, 2)
        per_dim[d] = row

    # ---- 胜率矩阵 + paired bootstrap CI ----
    win_matrix: dict[str, dict[str, Any]] = {}
    ci_matrix: dict[str, dict[str, Any]] = {}
    empty = {"wins": 0, "ties": 0, "losses": 0, "total": 0, "win_rate": 0.0}
    for m1 in models:
        win_matrix[m1] = {}
        ci_matrix[m1] = {}
        for m2 in models:
            if m1 == m2:
                win_matrix[m1][m2] = dict(empty)
                ci_matrix[m1][m2] = {"n": 0, "significant": False, "ci": [0.0, 0.0],
                                     "mean": 0.0, "note": "自身对比"}
                continue
            pairs = []
            for tid in scoring_ids:
                a = all_scores[m1].get(tid, {}).get("score")
                b = all_scores[m2].get(tid, {}).get("score")
                if a is not None and b is not None:
                    pairs.append((a, b))
            if not pairs:
                win_matrix[m1][m2] = dict(empty)
                ci_matrix[m1][m2] = {"n": 0, "significant": False,
                                     "ci": [0.0, 0.0], "mean": None,
                                     "note": "无共同计分题"}
                continue
            wa, wb = zip(*pairs)
            wr = win_rate(list(wa), list(wb))
            win_matrix[m1][m2] = {
                "wins": wr["wins"], "ties": wr["ties"], "losses": wr["losses"],
                "total": wr["total"], "win_rate": wr["win_rate"],
            }
            deltas = [a - b for a, b in pairs]
            mean = sum(deltas) / len(deltas)
            ci_lo, ci_hi = paired_bootstrap_ci(list(wa), list(wb), seed=seed)
            n = len(pairs)
            significant = n >= MIN_SAMPLE and not (ci_lo <= 0 <= ci_hi)
            ci_matrix[m1][m2] = {
                "mean": round(mean, 3),
                "ci": [ci_lo, ci_hi],
                "n": n,
                "significant": significant,
                "note": ("差异显著" if significant
                         else (f"题数不足 {MIN_SAMPLE}，差异不显著" if n < MIN_SAMPLE
                               else "置信区间包含 0，差异不显著")),
            }
            # 迭代十二（改动 F 精神）：按 job（数据集实例）聚类稳健 CI 对照
            c_deltas: dict[str, list[float]] = {}
            for jid, ex in per_job_extract.items():
                e1, e2 = ex.get(m1) or {}, ex.get(m2) or {}
                shared = [
                    tid for tid in scoring_ids
                    if (e1.get(tid) or {}).get("score") is not None
                    and (e2.get(tid) or {}).get("score") is not None
                ]
                if len(shared) >= 2:
                    c_deltas[jid] = [
                        float(e1[tid]["score"] - e2[tid]["score"]) for tid in shared
                    ]
            if len(c_deltas) >= 2:
                cl_lo, cl_hi = clustered_bootstrap_ci(c_deltas, seed=seed)
                ci_matrix[m1][m2]["ci_clustered"] = {
                    "ci": [cl_lo, cl_hi],
                    "n_clusters": len(c_deltas),
                    "cluster_unit": "job（数据集实例）",
                    "note": "聚类 bootstrap 通常给出更宽 CI（伪重复校正），与逐题 CI 对照呈现",
                }

    # ---- K-召回率（Top-K 达标覆盖） ----
    k_recall: dict[str, dict[str, Any]] = {}
    for m in models:
        scores = {tid: all_scores[m].get(tid, {}).get("score")
                  for tid in scoring_ids}
        k_recall[m] = k_recall_curve(scores)

    # ---- 箱线（得分分布）/ 散点（得分-延迟/Token）/ 雷达（分维度均分） ----
    score_dist: dict[str, dict[str, Any]] = {}
    scatter: dict[str, list[dict[str, Any]]] = {}
    radar_avg: dict[str, dict[str, float]] = {}
    for m in models:
        values = [all_scores[m][tid]["score"] for tid in scoring_ids
                  if all_scores[m].get(tid, {}).get("score") is not None]
        score_dist[m] = _box(values)
        scatter[m] = [{
            "task_id": tid,
            "score": all_scores[m][tid]["score"],
            "latency_ms": all_scores[m][tid].get("latency_ms"),
            "tokens": all_scores[m][tid].get("tokens"),
        } for tid in scoring_ids if all_scores[m].get(tid, {}).get("score") is not None]
        radar_avg[m] = {}
        for d in dims:
            tids = [tid for tid, dd in dim_of.items() if dd == d]
            vals = [all_scores[m][tid]["score"] for tid in tids
                    if all_scores[m].get(tid, {}).get("score") is not None]
            radar_avg[m][d] = round(sum(vals) / len(vals), 2) if vals else None

    excluded_dimensions = sorted({dim_of[tid] for tid in excluded_ids})

    # ---- 迭代十二：效率归一化（每 1K token 得分 + 跨评测集共用归一基准） ----
    efficiency: dict[str, dict[str, Any]] = {}
    eff_raw: dict[str, float | None] = {}
    for m in models:
        tokens = 0
        for tid in scoring_ids:
            entry = all_scores[m].get(tid)
            if entry is not None and entry.get("score") is not None:
                tokens += int(entry.get("tokens") or 0)
        te = token_efficiency(composite[m]["score"], completion=tokens)
        efficiency[m] = {
            "tokens": te["tokens"],
            "score": composite[m]["score"],
            "score_per_1k_tokens": te["score_per_1k_tokens"],
            "cost_per_score": te["cost_per_score"],
            "norm_score": None,
        }
        eff_raw[m] = te["score_per_1k_tokens"]
    normed = normalize_efficiency([eff_raw[m] for m in models])
    for m, nv in zip(models, normed):
        efficiency[m]["norm_score"] = nv

    # ---- 迭代十二（改动 E）：成本–性能前沿线（累计 token → 累计得分 + 拐点） ----
    efficiency_frontier_data: dict[str, dict[str, Any]] = {}
    for m in models:
        pts = [
            (float(all_scores[m][tid].get("tokens") or 0), float(all_scores[m][tid]["score"]))
            for tid in scoring_ids
            if all_scores[m].get(tid, {}).get("score") is not None
        ]
        efficiency_frontier_data[m] = efficiency_frontier(pts)

    # ---- 迭代十二（改动 C）：结论跨评测集可迁移性标注（LODO 精神） ----
    transferability = _build_transferability(
        models, scoring_ids, per_job_extract, per_job_dataset)

    # ---- 迭代十二（N3）：任务锚定能力指数（ACI）----
    # ACI = 模型跨评测集综合平均得分率（composite.score / (10×计分题数)），
    # 作为跨任务集横向比较的归一基准（论文推荐 ACI 为主要能力指标）。
    aci = _build_aci(models, composite, per_job_extract, per_job_dataset)

    return {
        "name": name or "",
        "jobs": job_ids,
        "dataset": dataset,
        "models": models,
        "ranks": ranks,
        "composite": composite,
        "dims": dims,
        "per_dim": per_dim,
        "win_matrix": win_matrix,
        "ci": ci_matrix,
        "k_recall": k_recall,
        "score_dist": score_dist,
        "scatter": scatter,
        "radar": {"dimensions": dims, "avg": radar_avg},
        "efficiency": efficiency,
        "efficiency_frontier": efficiency_frontier_data,
        "transferability": transferability,
        "aci": aci,
        "excluded_dimensions": excluded_dimensions,
        "seed": seed,
        "note": "口径：各 job 为双盲成对评审，得分经 reveal 归一化到模型名跨 job "
                "聚合（同模型多 job 同题 last-wins）；综合分=计分题得分合计（满分 "
                "10×题数）；安全与价值观维度不计入排名；CI 为配对 bootstrap 95% "
                "区间，题数不足标注差异不显著。efficiency 基于该评测集合计分/合计 "
                "token，norm_score 以当前榜单最高效率为 1.0 共用归一基准。"
                "绝对分数与排序基于当前评测集，跨评测集可能翻转（LODO 精神），"
                "选型需谨慎。aci（能力指数）= 模型跨评测集综合平均得分率，作为跨任务集"
                "横向比较的归一基准。",
    }


def _build_aci(
    models: list[str],
    composite: dict[str, dict[str, Any]],
    per_job_extract: dict[str, dict[str, dict[str, dict[str, Any]]]],
    per_job_dataset: dict[str, str | None],
) -> dict[str, dict[str, Any]]:
    """任务锚定能力指数（迭代十二 N3，论文 ACI）。

    每模型 aci = 综合平均得分率（composite.score / max(10×n_scored, 1)）。
    多评测集（不同 dataset_name 的 job 混合）时同时给出按评测集分别统计的
    平均得分率（per_dataset），供"模型×任务集"横向比较。
    无计分数据 → {"aci": None, "per_dataset": {}}（不输出段内数值）。
    """
    per_ds_scores: dict[str, dict[str, dict[str, float]]] = {}
    for jid, ex in per_job_extract.items():
        ds = per_job_dataset.get(jid) or "__all__"
        bucket = per_ds_scores.setdefault(ds, {})
        for m, per_task in ex.items():
            mb = bucket.setdefault(m, {})
            for tid, entry in per_task.items():
                if entry.get("score") is not None:
                    mb[tid] = float(entry["score"])
    out: dict[str, dict[str, Any]] = {}
    for m in models:
        c = composite.get(m) or {}
        n = int(c.get("n_scored") or 0)
        aci_value = round(float(c.get("score") or 0) / max(10.0 * n, 1.0), 4) if n else None
        per_dataset: dict[str, float | None] = {}
        for ds, bucket in sorted(per_ds_scores.items()):
            scores = bucket.get(m) or {}
            if scores:
                per_dataset[ds] = round(
                    sum(scores.values()) / (10.0 * len(scores)), 4)
            else:
                per_dataset[ds] = None
        out[m] = {"aci": aci_value, "per_dataset": per_dataset}
    return out


def _build_transferability(
    models: list[str],
    scoring_ids: list[str],
    per_job_extract: dict[str, dict[str, dict[str, dict[str, Any]]]],
    per_job_dataset: dict[str, str | None],
) -> dict[str, Any]:
    """跨评测集胜者方向一致性标注（改动 C）。

    对每对模型，按数据集分组统计“各评测集中胜者方向”：方向一致 →
    consistent；出现相反方向 → flips_across_datasets 并列出相悖的评测集。
    单臂（跨 job 配对）/双盲（同 job 配对）均按“同一评测集内逐题配对”口径。
    """
    by_dataset: dict[str, dict[str, dict[str, float]]] = {}
    for jid, ex in per_job_extract.items():
        ds = per_job_dataset.get(jid) or "__all__"
        bucket = by_dataset.setdefault(ds, {})
        for m, per_task in ex.items():
            model_bucket = bucket.setdefault(m, {})
            for tid, entry in per_task.items():
                if entry.get("score") is not None:
                    model_bucket[tid] = float(entry["score"])
    per_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for i, m1 in enumerate(models):
        per_pair[m1] = {}
        for m2 in models[i + 1:]:
            directions: dict[str, str] = {}
            flipped: dict[str, str] = {}
            for ds, bucket in sorted(by_dataset.items()):
                s1 = bucket.get(m1) or {}
                s2 = bucket.get(m2) or {}
                shared = [t for t in scoring_ids if t in s1 and t in s2]
                if not shared:
                    continue
                d = sum(s1[t] - s2[t] for t in shared)
                directions[ds] = "m1_ahead" if d > 1e-9 else ("m2_ahead" if d < -1e-9 else "tie")
            signs = [v for v in directions.values() if v != "tie"]
            if not signs:
                status = "consistent"  # 无有效方向（全平局或数据缺失）
                flipped = {}
            elif all(s == signs[0] for s in signs):
                status = "consistent"
            else:
                status = "flips_across_datasets"
                for ds, v in directions.items():
                    if v != "tie" and v != signs[0]:
                        flipped[ds] = v
            per_pair[m1][m2] = {
                "status": status,
                "datasets": sorted(directions),
                "directions": directions,
                "flipped": dict(sorted(flipped.items())),
            }
    has_multi = len(by_dataset) > 1
    return {
        "per_pair": per_pair,
        "datasets": sorted(by_dataset),
        "note": ("跨评测集结论迁移性：按数据集分组统计胜者方向，方向相反表示"
                 "排序可能随评测集翻转（论文 LODO：绝对分数不可跨域迁移）。"
                 + ("" if has_multi else "当前仅一个评测集，无可迁移性对照。")),
    }
