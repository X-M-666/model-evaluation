# -*- coding: utf-8 -*-
"""报告生成器（纯规则化统计，无 API）：从任务集/答卷/verdict 计算全部指标，
生成 8 类图表数据与逐条引用数值的精确分析文本，写入 report JSON 供前端渲染。

迭代二：新增 metrics（逐题指标引擎）/ kpi / significance（bootstrap 显著性）/
warnings（跳过与降级提示）四段；judge_mode 按实际评审模式输出；
significance 与 win_rate 均按 scoring_ids（排除 excluded_from_total）过滤。
"""
from __future__ import annotations

import statistics
from typing import Any

from backend.engine.metrics import (
    GROUNDING_SUPPORT_THRESHOLD, compute_task_metrics,
    metric_answer_relevancy, metric_grounding_faithfulness,
)
from backend.engine.stats import MIN_SAMPLE, significance_note
from backend.engine.tasks import STABILITY_DIMENSION

EPS = 0.001
SIG_SEED = 2026


def _group_by_id(ans: dict[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for e in (ans or {}).get("answers", []):
        grouped.setdefault(e["id"], []).append(e)
    return grouped


def _pick_entry(entries: list[dict[str, Any]]) -> dict[str, Any]:
    return entries[0] if entries else {}


def _api(entry: dict[str, Any]) -> dict[str, Any]:
    return entry.get("api_info", {}) or {}


def _fmt(n: float, digits: int = 2) -> str:
    return f"{n:.{digits}f}".rstrip("0").rstrip(".") if isinstance(n, (int, float)) else str(n)


def _entries_of(round_ans: dict[str, Any] | None, task_id: str) -> list[dict[str, Any]]:
    """取某一轮答卷中指定题目的全部 entry（稳定性题同 id 有两次运行）。"""
    return [e for e in (round_ans or {}).get("answers", []) if e["id"] == task_id]


def _round_reveals(verdict: dict[str, Any], round_count: int) -> list[dict[str, str]]:
    """每轮 X/Y 身份映射；旧 verdict 无 round_reveals 时用最后一轮 reveal 兜底。"""
    reveals = (verdict.get("meta") or {}).get("round_reveals")
    if isinstance(reveals, list) and len(reveals) >= round_count:
        return reveals[:round_count]
    revealed = verdict.get("revealed", {})
    fallback = {
        "answer_x_file": revealed.get("answer_x_file", "a"),
        "answer_y_file": revealed.get("answer_y_file", "b"),
    }
    return [fallback] * round_count


def _stable_agg(
    rounds_answers: list[dict[str, Any]],
    task_id: str,
    label: str,
) -> dict[str, Any]:
    """把某一稳定模型（"a"/"b"）在全部轮次的数据聚合成报告指标。

    聚合在稳定空间进行（轮次间 X/Y 身份交换不影响结果），返回：
      latency / latency_median / prompt / completion（跨轮）
      latency_rounds / tokens_rounds（逐轮明细）
      ok_rounds / total_rounds（成功率）
      code 聚合 passed/total + code_rounds 逐轮
      rounds（逐轮原文，带轮次标记）
      runs（最后一轮全部 raw_answer，含稳定性题两次运行）
    """
    entries = [
        _entries_of(ra.get(label), task_id) for ra in rounds_answers
    ]
    latencies = []
    latency_rounds = []
    tokens_rounds = []
    code_rounds = []
    rounds = []
    for i, entry_list in enumerate(entries):
        e = entry_list[0] if entry_list else {}
        api = _api(e)
        status = api.get("status", "ok")
        lat = api.get("latency_ms", 0)
        latency_rounds.append({"round": i + 1, "latency_ms": lat, "status": status})
        tokens_rounds.append({
            "round": i + 1,
            "prompt_tokens": api.get("prompt_tokens", 0),
            "completion_tokens": api.get("completion_tokens", 0),
            "status": status,
        })
        if status == "ok" and lat > 0:
            latencies.append(lat)
        cv = (e or {}).get("code_verify") or {}
        if cv.get("status") == "run":
            code_rounds.append({
                "round": i + 1, "status": "run",
                "passed": cv.get("passed", 0), "total": cv.get("total", 0),
            })
        runs = [x.get("raw_answer", "") for x in entry_list if x.get("raw_answer")]
        if runs:
            rounds.append({
                "round": i + 1,
                "raw_answer": runs[0],
                "repeat_runs": runs if len(runs) > 1 else None,
            })
    return {
        "latency": round(statistics.mean(latencies), 1) if latencies else 0,
        "latency_median": round(statistics.median(latencies), 1) if latencies else 0,
        "latency_rounds": latency_rounds,
        "tokens_rounds": tokens_rounds,
        "prompt": sum(t["prompt_tokens"] for t in tokens_rounds),
        "completion": sum(t["completion_tokens"] for t in tokens_rounds),
        "ok_rounds": sum(1 for t in latency_rounds if t["status"] == "ok"),
        "total_rounds": len(latency_rounds),
        "code_passed": sum(c["passed"] for c in code_rounds),
        "code_total": sum(c["total"] for c in code_rounds),
        "code_rounds": code_rounds,
        "rounds": rounds,
    }


def _aggregate_rounds(
    rounds_answers: list[dict[str, Any]],
    task_id: str,
    verdict: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """跨轮聚合并投影为 答案X/答案Y 两套指标（按 verdict 最后一轮 reveal 投影）。"""
    reveals = _round_reveals(verdict, len(rounds_answers))
    last_x = reveals[-1]["answer_x_file"]
    agg_a = _stable_agg(rounds_answers, task_id, "a")
    agg_b = _stable_agg(rounds_answers, task_id, "b")
    if last_x == "a":
        return agg_a, agg_b
    return agg_b, agg_a


def reveal_answers(
    answers_a: dict[str, Any] | None,
    answers_b: dict[str, Any] | None,
    verdict: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """按 verdict.revealed 的 X/Y 映射返回归一化的 (answers_x, answers_y)。

    与 build_report 内部 pool 选择、summary.x_model 同源于同一 revealed，
    保证报告标签、图表、代码通过率与答案原文使用同一映射。
    """
    revealed = (verdict or {}).get("revealed", {})
    x_file = revealed.get("answer_x_file", "a")
    y_file = revealed.get("answer_y_file", "b")
    answers_x = answers_a if x_file == "a" else answers_b
    answers_y = answers_b if y_file == "b" else answers_a
    return answers_x, answers_y


def build_report(
    config: dict[str, Any],
    task_set: dict[str, Any],
    answers_a: dict[str, Any] | None,
    answers_b: dict[str, Any] | None,
    verdict: dict[str, Any],
    rounds_answers: list[dict[str, Any]] | None = None,
    embedding_config: dict[str, Any] | None = None,
    gold: dict[str, Any] | None = None,
    round_verdicts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """生成完整报告结构：summary + charts + analysis + 迭代二四段 + 元评估/一致率。

    多轮（rounds_answers 长度>1）时效率/成本/代码/原文按稳定模型跨轮聚合：
    延迟取平均、Token 求和、成功率=成功轮/总轮、代码通过率逐轮+聚合、
    原文带轮次标记；单轮/缺省时保持原逻辑。

    embedding_config：embedding provider 配置（仅用于报告标注，报告本身零网络）。
    gold（迭代三）：金标记录；None/不匹配 → meta_eval 段空态。
    round_verdicts（迭代三）：逐轮 verdict，供复评一致率（repeat_n≥2 时）。
    """
    tasks = task_set["tasks"]
    if rounds_answers and len(rounds_answers) > 1:
        rows = _build_rows_multi(tasks, rounds_answers, verdict)
    else:
        rows = _build_rows(tasks, answers_a, answers_b, verdict)

    summary = _build_summary(config, tasks, rows, verdict)
    charts = _build_charts(tasks, rows, summary)
    analysis = _build_analysis(tasks, rows, summary, charts)
    metrics = _build_metrics(task_set, rounds_answers, answers_a, answers_b, verdict,
                             embedding_config)
    significance = _build_significance(task_set, verdict)
    kpi = _build_kpi(summary, charts, significance, rows)
    warnings = _build_warnings(task_set, verdict, metrics, significance)
    meta_eval = _build_meta_eval(gold, verdict, task_set)
    consistency = _build_consistency(round_verdicts,
                                     verdict.get("meta", {}).get("repeat_n", 1))

    review_cfg = config.get("review") or {}
    review = {
        "mode": review_cfg.get("mode") or "pure_human",
        "degraded": bool(review_cfg.get("degraded")) or False,
        "k_top_human": review_cfg.get("k_top_human"),
    }
    return {
        "judge_mode": review_cfg.get("mode") or "human",  # 旧字段：缺省 human 兼容
        "review": review,
        "prompt_strategy": config.get("prompt_strategy", "cot"),
        "summary": summary,
        "charts": charts,
        "analysis": analysis,
        "metrics": metrics,
        "kpi": kpi,
        "significance": significance,
        "warnings": warnings,
        "meta_eval": meta_eval,
        "consistency": consistency,
    }


def _build_meta_eval(gold: dict[str, Any] | None,
                     verdict: dict[str, Any],
                     task_set: dict[str, Any]) -> dict[str, Any]:
    """金标元评估段（迭代三）：金标传入且可匹配时计算；否则空态提示。"""
    if not isinstance(gold, dict) or not gold.get("items"):
        return {
            "available": False, "spearman": None, "kappa": None,
            "gold_offset": None, "gold_source": None,
            "matched": 0, "gold_total": 0,
            "note": "未配置金标集，暂无元评估（可在页面录入或直接调用金标 API）",
        }
    from backend.gold import compute_meta_eval
    return compute_meta_eval(verdict, task_set, gold)


def _build_consistency(round_verdicts: list[dict[str, Any]] | None,
                       repeat_n: int) -> dict[str, Any] | None:
    """复评一致率（迭代三，M4）：repeat_n≥2 时稳定空间逐轮 winner 一致率。

    逐轮 winner 在稳定模型空间判定（model_a/tie/model_b）；单轮/缺数据 → None。
    """
    if repeat_n < 2 or not round_verdicts or len(round_verdicts) < 2:
        return None
    from backend.engine.human_review import _stable_scores
    from backend.engine.stats import consistency_rate

    per_id: dict[str, list[str]] = {}
    for rv in round_verdicts:
        # 迭代三：agent 逐题独立交换时 round-verdicts 落盘 per_task 映射，
        # 一致率按题归一化到稳定模型空间（避免轮级 reveal 失配）
        per_task = rv.get("revealed", {}).get("per_task")
        for r in _stable_scores(rv, per_task):
            a, b = r["model_a"], r["model_b"]
            eps = 1e-6
            winner = "model_a" if a - b > eps else ("model_b" if b - a > eps else "tie")
            per_id.setdefault(r["id"], []).append(winner)
    per_task = {
        tid: {"rounds": wins,
              "rate": consistency_rate(wins)}
        for tid, wins in sorted(per_id.items())
    }
    task_rates = [v["rate"] for v in per_task.values() if v["rate"] is not None]
    overall_rate = round(
        sum(task_rates) / len(task_rates), 4
    ) if task_rates else None
    return {
        "repeat_n": repeat_n,
        "per_task": per_task,
        "overall": overall_rate,
        "note": "稳定空间逐轮 winner 一致率（两两轮次比对）",
    }


def _build_metrics(
    task_set: dict[str, Any],
    rounds_answers: list[dict[str, Any]] | None,
    answers_a: dict[str, Any] | None,
    answers_b: dict[str, Any] | None,
    verdict: dict[str, Any],
    embedding_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """逐题指标（指标引擎）：X/Y 两侧分别计算；纯规则化、零网络。

    语义向量优先执行期采集（ans_entry.semantic），缺失时 n-gram 兜底；
    截断/调用失败/无 expected/代码未执行 → {skipped, reason} 由 warnings 段承接。
    """
    revealed = verdict.get("revealed", {})
    x_file = revealed.get("answer_x_file", "a")
    y_file = revealed.get("answer_y_file", "b")

    if rounds_answers and len(rounds_answers) > 1:
        last_round = rounds_answers[-1]
    else:
        last_round = {"a": answers_a, "b": answers_b}

    def entries(label: str, task_id: str) -> list[dict[str, Any]]:
        ra = last_round.get(label) if isinstance(last_round, dict) else None
        return [e for e in (ra or {}).get("answers", []) if e.get("id") == task_id]

    score_map = {s["id"]: s for s in verdict.get("scores", [])}
    per_task: list[dict[str, Any]] = []
    grounding_ctx_tasks = 0
    grounding_grounded_x = 0
    grounding_grounded_y = 0
    for t in task_set.get("tasks", []):
        sid = t["id"]
        sc = score_map.get(sid, {})
        item: dict[str, Any] = {
            "id": sid,
            "dimension": t.get("dimension", "自定义"),
            "x": compute_task_metrics(t, entries("a" if x_file == "a" else "b", sid),
                                      sc.get("answer_x")),
            "y": compute_task_metrics(t, entries("b" if y_file == "b" else "a", sid),
                                      sc.get("answer_y")),
        }
        ctx = (t.get("context") or "").strip()
        if ctx:
            gx = _grounding_side(ctx, entries("a" if x_file == "a" else "b", sid),
                                 t.get("prompt", ""))
            gy = _grounding_side(ctx, entries("b" if y_file == "b" else "a", sid),
                                 t.get("prompt", ""))
            item["grounding"] = {"x": gx, "y": gy}
            grounding_ctx_tasks += 1
            grounding_grounded_x += int(gx["grounded"])
            grounding_grounded_y += int(gy["grounded"])
        per_task.append(item)
    provider = {
        "kind": (embedding_config or {}).get("provider") or "auto",
        "error": (embedding_config or {}).get("error"),
    }
    result: dict[str, Any] = {"provider": provider, "per_task": per_task}
    if grounding_ctx_tasks:
        result["grounding"] = {
            "context_tasks": grounding_ctx_tasks,
            "grounded_x": grounding_grounded_x,
            "grounded_y": grounding_grounded_y,
            "threshold": GROUNDING_SUPPORT_THRESHOLD,
        }
    return result


def _grounding_side(context: str, side_entries: list[dict[str, Any]],
                    prompt: str) -> dict[str, Any]:
    """单侧 RAG 忠实性：最近一次成功运行取 raw_answer；缺失视为不通过。"""
    raw = ""
    for e in (side_entries or []):
        if (e.get("api_info") or {}).get("status") == "ok" and e.get("raw_answer"):
            raw = e["raw_answer"]
    if not raw.strip():
        return {"faithfulness": None, "answer_relevancy": None,
                "grounded": False, "reason": "no_answer"}
    faith = metric_grounding_faithfulness(raw, context)
    rel = metric_answer_relevancy(raw, prompt)
    return {
        "faithfulness": faith,
        "answer_relevancy": rel,
        "grounded": round(faith, 4) >= GROUNDING_SUPPORT_THRESHOLD,
        "reason": None,
    }


def _build_significance(
    task_set: dict[str, Any],
    verdict: dict[str, Any],
) -> dict[str, Any]:
    """bootstrap 配对显著性（迭代二）：稳定模型空间，按 scoring_ids 过滤。

    整体 + 分维度；样本 < MIN_SAMPLE 或 CI 含 0 → significant=False 并区分原因。
    """
    excluded = {t["id"] for t in task_set.get("tasks", []) if t.get("excluded_from_total")}
    x_file = verdict.get("revealed", {}).get("answer_x_file", "a")
    rows: list[dict[str, Any]] = []
    for s in verdict.get("scores", []):
        if s.get("id") in excluded:
            continue
        if x_file == "a":
            ma, mb = s.get("answer_x", 0), s.get("answer_y", 0)
        else:
            ma, mb = s.get("answer_y", 0), s.get("answer_x", 0)
        rows.append({"dimension": s.get("dimension", "自定义"), "ma": ma, "mb": mb})

    by_dim: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_dim.setdefault(r["dimension"], []).append(r)
    per_dimension = {
        d: significance_note([r["ma"] for r in rs], [r["mb"] for r in rs], seed=SIG_SEED)
        for d, rs in sorted(by_dim.items())
    }
    overall = (
        significance_note([r["ma"] for r in rows], [r["mb"] for r in rows], seed=SIG_SEED)
        if rows else {"significant": False, "reason": "no_scoring_tasks",
                      "ci": [0.0, 0.0], "sample": 0, "note": "无计分题，无法进行显著性检验"}
    )
    return {"overall": overall, "per_dimension": per_dimension}


def _build_kpi(
    summary: dict[str, Any],
    charts: dict[str, Any],
    significance: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """KPI 汇总：分数/胜场/代码通过率/延迟/token + 显著性结论。"""
    code_rates = {"x": None, "y": None}
    total_px = total_tx = total_py = total_ty = 0
    for c in charts.get("code_pass_rate", []):
        total_px += c.get("x_passed", 0)
        total_tx += c.get("x_total", 0)
        total_py += c.get("y_passed", 0)
        total_ty += c.get("y_total", 0)
    if total_tx:
        code_rates["x"] = round(total_px / total_tx, 4)
    if total_ty:
        code_rates["y"] = round(total_py / total_ty, 4)

    lat_x = [r["latency_x"] for r in rows if r["status_x"] == "ok" and r["latency_x"] > 0]
    lat_y = [r["latency_y"] for r in rows if r["status_y"] == "ok" and r["latency_y"] > 0]

    return {
        "total_score": {"x": summary["total_x"], "y": summary["total_y"],
                        "max": summary["max_score"]},
        "avg_score": {"x": summary["avg_x"], "y": summary["avg_y"]},
        "win_count": {"x": summary["win_x"], "y": summary["win_y"], "ties": summary["ties"]},
        "code_pass_rate": code_rates,
        "latency_ms": {
            "x": round(statistics.mean(lat_x), 1) if lat_x else None,
            "y": round(statistics.mean(lat_y), 1) if lat_y else None,
        },
        "total_tokens": {
            "x": sum(r["prompt_x"] + r["completion_x"] for r in rows),
            "y": sum(r["prompt_y"] + r["completion_y"] for r in rows),
        },
        "significance": significance["overall"],
    }


def _build_warnings(
    task_set: dict[str, Any],
    verdict: dict[str, Any],
    metrics: dict[str, Any],
    significance: dict[str, Any],
) -> list[dict[str, Any]]:
    """告警清单：指标跳过（截断/失败/无 expected/代码未执行）、评审失败、
    embedding 降级、显著性未达样本/CI 含 0。"""
    warnings: list[dict[str, Any]] = []
    meta = verdict.get("meta", {})
    if meta.get("invalid"):
        warnings.append({
            "code": "invalid_verdicts", "level": "warning",
            "message": f"评审失败 {meta['invalid']} 题，已按平局标记 invalid 处理",
        })
    for pt in metrics.get("per_task", []):
        for side in ("x", "y"):
            m = pt.get(side) or {}
            if m.get("skipped"):
                reason = m.get("reason", "")
                warnings.append({
                    "code": "metrics_skipped", "level": "info",
                    "message": f"{pt['id']} 侧{side}指标跳过：{reason}",
                })
    emb_err = (metrics.get("provider") or {}).get("error")
    if emb_err:
        warnings.append({
            "code": "embedding_degraded", "level": "warning",
            "message": f"embedding provider 不可用（{emb_err}），语义相似度已用 n-gram 兜底",
        })
    sig = significance.get("overall", {})
    if sig.get("reason") == "insufficient_sample":
        warnings.append({
            "code": "significance_sample", "level": "warning",
            "message": f"计分样本 {sig.get('sample', 0)} < {MIN_SAMPLE}，显著性结论不可靠",
        })
    elif sig.get("reason") == "ci_overlaps_zero":
        warnings.append({
            "code": "significance_overlap", "level": "info",
            "message": "总分差异不显著（bootstrap CI 包含 0）",
        })
    return warnings


def _build_rows(
    tasks: list[dict[str, Any]],
    answers_a: dict[str, Any] | None,
    answers_b: dict[str, Any] | None,
    verdict: dict[str, Any],
) -> list[dict[str, Any]]:
    """单轮（或未提供多轮答卷）时的逐题行：与修复前完全一致。"""
    revealed = verdict.get("revealed", {})
    x_file = revealed.get("answer_x_file", "a")
    y_file = revealed.get("answer_y_file", "b")
    pool_x = _group_by_id(answers_a if x_file == "a" else answers_b)
    pool_y = _group_by_id(answers_b if y_file == "b" else answers_a)

    rows: list[dict[str, Any]] = []
    for t in tasks:
        tid = t["id"]
        ex = _pick_entry(pool_x.get(tid, []))
        ey = _pick_entry(pool_y.get(tid, []))
        ax, ay = _api(ex), _api(ey)
        score_map = {s["id"]: s for s in verdict.get("scores", [])}
        sc = score_map.get(tid, {})
        rows.append({
            "id": tid,
            "dimension": t.get("dimension", "自定义"),
            "score_x": sc.get("answer_x", 0),
            "score_y": sc.get("answer_y", 0),
            "score_x_std": sc.get("answer_x_std", 0),
            "score_y_std": sc.get("answer_y_std", 0),
            "score_x_median": sc.get("answer_x_median", 0),
            "score_y_median": sc.get("answer_y_median", 0),
            "latency_x": ax.get("latency_ms", 0),
            "latency_y": ay.get("latency_ms", 0),
            "prompt_x": ax.get("prompt_tokens", 0),
            "prompt_y": ay.get("prompt_tokens", 0),
            "completion_x": ax.get("completion_tokens", 0),
            "completion_y": ay.get("completion_tokens", 0),
            "status_x": ax.get("status", "ok"),
            "status_y": ay.get("status", "ok"),
            "code_x": ex.get("code_verify"),
            "code_y": ey.get("code_verify"),
            "runs_x": [e.get("raw_answer", "") for e in pool_x.get(tid, []) if e.get("raw_answer")],
            "runs_y": [e.get("raw_answer", "") for e in pool_y.get(tid, []) if e.get("raw_answer")],
        })
    return rows


def _build_rows_multi(
    tasks: list[dict[str, Any]],
    rounds_answers: list[dict[str, Any]],
    verdict: dict[str, Any],
) -> list[dict[str, Any]]:
    """多轮时的逐题行：效率/成本/代码/原文按稳定模型跨轮聚合后投影 X/Y。"""
    score_map = {s["id"]: s for s in verdict.get("scores", [])}
    rows: list[dict[str, Any]] = []
    for t in tasks:
        tid = t["id"]
        agg_x, agg_y = _aggregate_rounds(rounds_answers, tid, verdict)
        sc = score_map.get(tid, {})
        ok_x, total_x = agg_x["ok_rounds"], agg_x["total_rounds"]
        ok_y, total_y = agg_y["ok_rounds"], agg_y["total_rounds"]
        code_x = {
            "status": "run",
            "passed": agg_x["code_passed"],
            "total": agg_x["code_total"],
        } if agg_x["code_rounds"] else {"status": "skip", "passed": 0, "total": 0}
        code_y = {
            "status": "run",
            "passed": agg_y["code_passed"],
            "total": agg_y["code_total"],
        } if agg_y["code_rounds"] else {"status": "skip", "passed": 0, "total": 0}
        rows.append({
            "id": tid,
            "dimension": t.get("dimension", "自定义"),
            "score_x": sc.get("answer_x", 0),
            "score_y": sc.get("answer_y", 0),
            "score_x_std": sc.get("answer_x_std", 0),
            "score_y_std": sc.get("answer_y_std", 0),
            "score_x_median": sc.get("answer_x_median", 0),
            "score_y_median": sc.get("answer_y_median", 0),
            "latency_x": agg_x["latency"],
            "latency_y": agg_y["latency"],
            "latency_x_median": agg_x["latency_median"],
            "latency_y_median": agg_y["latency_median"],
            "latency_x_rounds": agg_x["latency_rounds"],
            "latency_y_rounds": agg_y["latency_rounds"],
            "tokens_x_rounds": agg_x["tokens_rounds"],
            "tokens_y_rounds": agg_y["tokens_rounds"],
            "prompt_x": agg_x["prompt"],
            "prompt_y": agg_y["prompt"],
            "completion_x": agg_x["completion"],
            "completion_y": agg_y["completion"],
            "status_x": "ok" if ok_x > 0 else "fail",
            "status_y": "ok" if ok_y > 0 else "fail",
            "success_x": f"{ok_x}/{total_x}",
            "success_y": f"{ok_y}/{total_y}",
            "code_x": code_x,
            "code_y": code_y,
            "code_x_rounds": agg_x["code_rounds"],
            "code_y_rounds": agg_y["code_rounds"],
            "rounds_x": agg_x["rounds"],
            "rounds_y": agg_y["rounds"],
            "runs_x": [r["raw_answer"] for r in agg_x["rounds"]],
            "runs_y": [r["raw_answer"] for r in agg_y["rounds"]],
        })
    return rows


def _build_summary(
    config: dict[str, Any],
    tasks: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    verdict: dict[str, Any],
) -> dict[str, Any]:
    totals = verdict.get("totals", {})
    total_x = round(totals.get("answer_x", 0), 2)
    total_y = round(totals.get("answer_y", 0), 2)

    # 不计分题（excluded_from_total，如安全与价值观维度）：展示但不算战绩/平均分
    scoring_ids = {
        t["id"] for t in tasks if not t.get("excluded_from_total")
    }
    scoring_scores = [s for s in verdict.get("scores", []) if s.get("id") in scoring_ids]
    win_x = sum(1 for s in scoring_scores if s.get("winner") == "answer_x")
    win_y = sum(1 for s in scoring_scores if s.get("winner") == "answer_y")
    ties = sum(1 for s in scoring_scores if s.get("winner") == "tie")
    if total_x > total_y:
        winner = "answer_x"
    elif total_y > total_x:
        winner = "answer_y"
    else:
        winner = "tie"

    dims = list({r["dimension"] for r in rows})
    n_scoring = len(scoring_scores)
    avg_x = round(total_x / n_scoring, 2) if n_scoring else 0
    avg_y = round(total_y / n_scoring, 2) if n_scoring else 0
    excluded_dims = sorted({t["dimension"] for t in tasks if t.get("excluded_from_total")})

    return {
        "total_questions": len(rows),
        "dimensions": dims,
        "repeat_n": verdict.get("meta", {}).get("repeat_n", 1),
        "x_model": verdict.get("revealed", {}).get("answer_x", ""),
        "y_model": verdict.get("revealed", {}).get("answer_y", ""),
        "total_x": total_x,
        "total_y": total_y,
        "max_score": round(n_scoring * 10, 2),
        "avg_x": avg_x,
        "avg_y": avg_y,
        "win_x": win_x,
        "win_y": win_y,
        "ties": ties,
        "winner": winner,
        "excluded_dimensions": excluded_dims,
        "scoring_count": n_scoring,
        "winner_model": verdict.get("winner_model", "tie"),
        "score_ratio": round(total_x / total_y, 3) if total_y else None,
    }


def _build_charts(
    tasks: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    ids = [r["id"] for r in rows]
    dim_names = summary["dimensions"]

    score_by_question = {
        "categories": ids,
        "x": [r["score_x"] for r in rows],
        "y": [r["score_y"] for r in rows],
    }
    dim_map: dict[str, dict[str, float]] = {}
    for r in rows:
        d = r["dimension"]
        if d not in dim_map:
            dim_map[d] = {"x": 0.0, "y": 0.0}
        dim_map[d]["x"] += r["score_x"]
        dim_map[d]["y"] += r["score_y"]
    score_by_dimension = {
        "dimensions": dim_names,
        "x": [round(dim_map.get(d, {}).get("x", 0), 2) for d in dim_names],
        "y": [round(dim_map.get(d, {}).get("y", 0), 2) for d in dim_names],
    }
    latency_by_question = {
        "categories": ids,
        "x": [r["latency_x"] for r in rows],
        "y": [r["latency_y"] for r in rows],
    }
    tokens_by_question = {
        "categories": ids,
        "x_prompt": [r["prompt_x"] for r in rows],
        "x_completion": [r["completion_x"] for r in rows],
        "y_prompt": [r["prompt_y"] for r in rows],
        "y_completion": [r["completion_y"] for r in rows],
    }
    code_pass_rate = [
        {
            "id": r["id"],
            "x_passed": (r["code_x"] or {}).get("passed", 0) if (r["code_x"] or {}).get("status") == "run" else 0,
            "x_total": (r["code_x"] or {}).get("total", 0) if (r["code_x"] or {}).get("status") == "run" else 0,
            "y_passed": (r["code_y"] or {}).get("passed", 0) if (r["code_y"] or {}).get("status") == "run" else 0,
            "y_total": (r["code_y"] or {}).get("total", 0) if (r["code_y"] or {}).get("status") == "run" else 0,
        }
        for r in rows
        if (r["code_x"] or {}).get("status") == "run" or (r["code_y"] or {}).get("status") == "run"
    ]
    efficiency_scatter = {
        "x_points": [
            {"name": r["id"], "latency_ms": r["latency_x"],
             "completion_tokens": r["completion_x"], "score": r["score_x"]}
            for r in rows
        ],
        "y_points": [
            {"name": r["id"], "latency_ms": r["latency_y"],
             "completion_tokens": r["completion_y"], "score": r["score_y"]}
            for r in rows
        ],
    }
    win_distribution = {
        "win_x": summary["win_x"],
        "win_y": summary["win_y"],
        "ties": summary["ties"],
    }
    stability = None
    if summary["repeat_n"] > 1:
        stability = {
            "categories": ids,
            "x": [r["score_x"] for r in rows],
            "y": [r["score_y"] for r in rows],
            "x_std": [r["score_x_std"] for r in rows],
            "y_std": [r["score_y_std"] for r in rows],
            "x_median": [r["score_x_median"] for r in rows],
            "y_median": [r["score_y_median"] for r in rows],
        }

    return {
        "score_by_question": score_by_question,
        "score_by_dimension": score_by_dimension,
        "latency_by_question": latency_by_question,
        "tokens_by_question": tokens_by_question,
        "code_pass_rate": code_pass_rate,
        "efficiency_scatter": efficiency_scatter,
        "win_distribution": win_distribution,
        "stability": stability,
    }


def _build_analysis(
    tasks: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    charts: dict[str, Any],
) -> list[dict[str, Any]]:
    """纯规则化分析：7 个段落，逐条引用具体数值。"""
    sections: list[dict[str, Any]] = []

    # ---- 1. 总体结论 ----
    paras = []
    wname = summary["x_model"] or "答案X"
    lname = summary["y_model"] or "答案Y"
    if summary["winner"] == "tie":
        paras.append(
            f"双方总分战平：{wname} {_fmt(summary['total_x'])} 分 vs {lname} {_fmt(summary['total_y'])} 分，"
            f"满分 {_fmt(summary['max_score'])} 分。"
        )
    else:
        wscore = summary["total_x"] if summary["winner"] == "answer_x" else summary["total_y"]
        lscore = summary["total_y"] if summary["winner"] == "answer_x" else summary["total_x"]
        winner_txt = summary["winner_model"]
        if winner_txt == wname:
            winner_txt = f"{summary['winner_model']}（答案X）"
        elif winner_txt == lname:
            winner_txt = f"{summary['winner_model']}（答案Y）"
        paras.append(
            f"胜方为 {winner_txt}，总分 {_fmt(wscore)} 分，"
            f"领先 {lname}（{lscore} 分）共 {_fmt(wscore - lscore)} 分，"
            f"满分 {_fmt(summary['max_score'])} 分，领先幅度 {_fmt((wscore - lscore) / summary['max_score'] * 100)}%。"
        )
    paras.append(
        f"逐题战绩：{wname if summary['winner']!='tie' else '答案X'} 胜 {summary['win_x']} 题、"
        f"{lname if summary['winner']!='tie' else '答案Y'} 胜 {summary['win_y']} 题、平局 {summary['ties']} 题，"
        f"共 {summary.get('scoring_count', summary['total_questions'])} 题（计分）。"
    )
    excluded_dims = summary.get("excluded_dimensions", [])
    if excluded_dims:
        paras.append(
            f"不计分维度（{'、'.join(excluded_dims)}）：相关题目仍逐题打分供参考，"
            f"但不计入总分、平均分与胜负判定。"
        )
    if summary["avg_x"] > summary["avg_y"]:
        paras.append(f"平均分 {_fmt(summary['avg_x'])} vs {_fmt(summary['avg_y'])}（每题满分 10 分），{wname} 每题平均高出 {_fmt(summary['avg_x'] - summary['avg_y'])} 分。")
    elif summary["avg_y"] > summary["avg_x"]:
        paras.append(f"平均分 {_fmt(summary['avg_x'])} vs {_fmt(summary['avg_y'])}（每题满分 10 分），{lname} 每题平均高出 {_fmt(summary['avg_y'] - summary['avg_x'])} 分。")
    else:
        paras.append(f"平均分均为 {_fmt(summary['avg_x'])} 分（每题满分 10 分），整体实力相当。")
    sections.append({"title": "总体结论", "paragraphs": paras})

    # ---- 2. 得分剖析 ----
    paras = []
    sorted_rows = sorted(rows, key=lambda r: (r["score_x"] - r["score_y"]), reverse=True)
    best_x_row = max(rows, key=lambda r: r["score_x"]) if rows else None
    best_y_row = max(rows, key=lambda r: r["score_y"]) if rows else None
    worst_x_row = min(rows, key=lambda r: r["score_x"]) if rows else None
    worst_y_row = min(rows, key=lambda r: r["score_y"]) if rows else None
    if best_x_row:
        paras.append(f"{wname} 最佳表现：{best_x_row['id']}（{best_x_row['dimension']}）得 {_fmt(best_x_row['score_x'])} 分；最弱环节：{worst_x_row['id']}（{worst_x_row['dimension']}）仅得 {_fmt(worst_x_row['score_x'])} 分。")
    if best_y_row:
        paras.append(f"{lname} 最佳表现：{best_y_row['id']}（{best_y_row['dimension']}）得 {_fmt(best_y_row['score_y'])} 分；最弱环节：{worst_y_row['id']}（{worst_y_row['dimension']}）仅得 {_fmt(worst_y_row['score_y'])} 分。")
    gaps = sorted(rows, key=lambda r: abs(r["score_x"] - r["score_y"]), reverse=True)
    if gaps:
        g = gaps[0]
        paras.append(f"分差最大的题目：{g['id']}（{g['dimension']}），{_fmt(g['score_x'])} vs {_fmt(g['score_y'])}，相差 {_fmt(abs(g['score_x'] - g['score_y']))} 分。")
    close = sorted(rows, key=lambda r: abs(r["score_x"] - r["score_y"]))
    if close:
        c = close[0]
        if abs(c["score_x"] - c["score_y"]) < EPS:
            paras.append(f"最胶着的题目：{c['id']}（{c['dimension']}）双方均为 {_fmt(c['score_x'])} 分，难分高下。")
        else:
            paras.append(f"最胶着的题目：{c['id']}（{c['dimension']}）仅相差 {_fmt(abs(c['score_x'] - c['score_y']))} 分（{_fmt(c['score_x'])} vs {_fmt(c['score_y'])}）。")
    full_x = [r for r in rows if r["score_x"] >= 10 - EPS]
    full_y = [r for r in rows if r["score_y"] >= 10 - EPS]
    if full_x or full_y:
        paras.append(f"满分表现：{wname} 得满分 {len(full_x)} 题（{', '.join(r['id'] for r in full_x) or '无'}）；{lname} 得满分 {len(full_y)} 题（{', '.join(r['id'] for r in full_y) or '无'}）。")
    sections.append({"title": "得分剖析", "paragraphs": paras})

    # ---- 3. 效率剖析 ----
    paras = []
    multi = summary["repeat_n"] > 1 and any("success_x" in r for r in rows)
    lat_x = [r["latency_x"] for r in rows if r["status_x"] == "ok" and r["latency_x"] > 0]
    lat_y = [r["latency_y"] for r in rows if r["status_y"] == "ok" and r["latency_y"] > 0]
    avg_lx = round(statistics.mean(lat_x), 1) if lat_x else 0
    avg_ly = round(statistics.mean(lat_y), 1) if lat_y else 0
    if lat_x and lat_y:
        scope = f"（跨 {summary['repeat_n']} 轮平均，仅统计成功轮次）" if multi else ""
        paras.append(f"平均响应延迟：{wname} {_fmt(avg_lx)} ms vs {lname} {_fmt(avg_ly)} ms{scope}。")
        med_x = [r.get("latency_x_median", 0) for r in rows if r["status_x"] == "ok" and r["latency_x"] > 0] if multi else []
        med_y = [r.get("latency_y_median", 0) for r in rows if r["status_y"] == "ok" and r["latency_y"] > 0] if multi else []
        if multi and med_x and med_y:
            paras.append(f"延迟中位数：{wname} {_fmt(statistics.median(med_x))} ms vs {lname} {_fmt(statistics.median(med_y))} ms。")
        if avg_lx < avg_ly:
            paras.append(f"{wname} 整体更快，平均每题少花 {_fmt(avg_ly - avg_lx)} ms（约 {_fmt((avg_ly - avg_lx) / avg_ly * 100)}%）。")
        elif avg_ly < avg_lx:
            paras.append(f"{lname} 整体更快，平均每题少花 {_fmt(avg_lx - avg_ly)} ms（约 {_fmt((avg_lx - avg_ly) / avg_lx * 100)}%）。")
        else:
            paras.append("双方平均延迟完全相同。")
    fast_rows = [r for r in rows if r["status_x"] == "ok" and r["status_y"] == "ok" and r["latency_x"] > 0 and r["latency_y"] > 0]
    if fast_rows:
        fastest = min(fast_rows, key=lambda r: min(r["latency_x"], r["latency_y"]))
        slowest = max(fast_rows, key=lambda r: max(r["latency_x"], r["latency_y"]))
        paras.append(f"单题耗时最快：{fastest['id']}（{fastest['dimension']}）最短 {min(fastest['latency_x'], fastest['latency_y'])} ms；最慢：{slowest['id']} 最长 {max(slowest['latency_x'], slowest['latency_y'])} ms。")
    lag_rows = [r for r in fast_rows if abs(r["latency_x"] - r["latency_y"]) / max(r["latency_x"], r["latency_y"], 1) > 0.3]
    if lag_rows:
        lag = max(lag_rows, key=lambda r: abs(r["latency_x"] - r["latency_y"]))
        paras.append(f"延迟差距最悬殊的题目：{lag['id']}，{_fmt(lag['latency_x'])} ms vs {_fmt(lag['latency_y'])} ms，相差 {_fmt(abs(lag['latency_x'] - lag['latency_y']))} ms。")
    if multi:
        x_ok = sum(int(r["success_x"].split("/")[0]) for r in rows)
        x_tot = sum(int(r["success_x"].split("/")[1]) for r in rows)
        y_ok = sum(int(r["success_y"].split("/")[0]) for r in rows)
        y_tot = sum(int(r["success_y"].split("/")[1]) for r in rows)
        if x_tot or y_tot:
            paras.append(f"调用成功率：{wname} {x_ok}/{x_tot} 轮成功，{lname} {y_ok}/{y_tot} 轮成功。")
    sections.append({"title": "效率剖析", "paragraphs": paras})

    # ---- 4. 成本剖析 ----
    paras = []
    p_x = sum(r["prompt_x"] for r in rows)
    c_x = sum(r["completion_x"] for r in rows)
    p_y = sum(r["prompt_y"] for r in rows)
    c_y = sum(r["completion_y"] for r in rows)
    cost_scope = f"（{summary['repeat_n']} 轮合计）" if multi else ""
    paras.append(f"输入 token（prompt）：{wname} {p_x} vs {lname} {p_y}；输出 token（completion）：{wname} {c_x} vs {lname} {c_y}。")
    paras.append(f"总 token 消耗{cost_scope}：{wname} {p_x + c_x}（输入 {p_x} / 输出 {c_x}），{lname} {p_y + c_y}（输入 {p_y} / 输出 {c_y}）。")
    if c_x + c_y > 0:
        paras.append(f"输出占比：{wname} {_fmt(c_x / (p_x + c_x) * 100 if p_x + c_x else 0)}%，{lname} {_fmt(c_y / (p_y + c_y) * 100 if p_y + c_y else 0)}%（输出 token 越多通常意味着回答越长、潜在成本越高）。")
    if summary["total_y"] > 0:
        paras.append(f"单位得分成本：{wname} 每得 1 分消耗 {_fmt((p_x + c_x) / summary['total_x'] if summary['total_x'] else 0)} token；{lname} 每得 1 分消耗 {_fmt((p_y + c_y) / summary['total_y'])} token。")
    sections.append({"title": "成本剖析", "paragraphs": paras})

    # ---- 5. 代码质量 ----
    code_rows = [
        r for r in rows
        if (r["code_x"] or {}).get("status") == "run" or (r["code_y"] or {}).get("status") == "run"
    ]
    if code_rows:
        paras = []
        for r in code_rows:
            cx, cy = r["code_x"] or {}, r["code_y"] or {}
            tx = cx.get("total", 0) if cx.get("status") == "run" else 0
            ty = cy.get("total", 0) if cy.get("status") == "run" else 0
            px = cx.get("passed", 0) if cx.get("status") == "run" else 0
            py = cy.get("passed", 0) if cy.get("status") == "run" else 0
            if tx or ty:
                rate_x = f"{px}/{tx}"
                rate_y = f"{py}/{ty}"
                result = "X 全过" if px == tx and py < ty else ("Y 全过" if py == ty and px < tx else ("双方全过" if px == tx and py == ty else ("均未全过")))
                paras.append(f"{r['id']}（代码题）：{wname} 通过 {rate_x} 组用例，{lname} 通过 {rate_y} 组用例——{result}。")
                if summary["repeat_n"] > 1:
                    def _rounds_detail(rounds: list, model_name: str) -> str:
                        if not rounds:
                            return f"{model_name} 无执行轮次"
                        parts = ["第{0}轮 {1}/{2}".format(item["round"], item["passed"], item["total"]) for item in rounds]
                        return f"{model_name} 各轮：{'、'.join(parts)}"
                    paras.append(f"  {r['id']} 逐轮明细：{_rounds_detail(r.get('code_x_rounds', []), wname)}；{_rounds_detail(r.get('code_y_rounds', []), lname)}。")
        if paras:
            paras.append("说明：代码题通过率来自隔离环境自动验证（executor 已按测试用例逐组执行），人工评分时已同步展示；未执行的代码题不参与该统计。")
        sections.append({"title": "代码质量", "paragraphs": paras})

    # ---- 6. 稳定性 ----
    paras = []
    if summary["repeat_n"] > 1 and charts["stability"]:
        st = charts["stability"]
        vol = sorted(rows, key=lambda r: max(r["score_x_std"], r["score_y_std"]), reverse=True)
        paras.append(f"本次评测重复 {summary['repeat_n']} 轮取平均；标准差反映各轮波动，中位数反映典型水平。")
        if vol and max(vol[0]["score_x_std"], vol[0]["score_y_std"]) > 0:
            v = vol[0]
            paras.append(f"波动最大的题目：{v['id']}（{v['dimension']}），X 标准差 {_fmt(v['score_x_std'])}、Y 标准差 {_fmt(v['score_y_std'])}，说明该题得分跨轮不稳定。")
        else:
            paras.append("所有题目各轮得分零波动（标准差为 0），结果高度稳定。")
        x_volatile = [r for r in rows if r["score_x_std"] > 0]
        y_volatile = [r for r in rows if r["score_y_std"] > 0]
        paras.append(f"{wname} 有 {len(x_volatile)} 题跨轮波动（{'、'.join(r['id'] for r in x_volatile) or '无'}）；{lname} 有 {len(y_volatile)} 题跨轮波动（{'、'.join(r['id'] for r in y_volatile) or '无'}）。")
    else:
        paras.append("本次评测为单轮（repeat_n=1），未做跨轮稳定性统计。")

    def _repeat_runs(r: dict[str, Any], key: str) -> list[str]:
        rounds_list = r.get(key)
        if isinstance(rounds_list, list) and rounds_list:
            last = rounds_list[-1]
            rr = last.get("repeat_runs") if isinstance(last, dict) else None
            return rr or []
        return r.get("runs_x" if key == "runs_x" else "runs_y", []) if key == "runs_x" else r.get("runs_y", [])

    same_rows = [
        r for r in rows
        if len(_repeat_runs(r, "runs_x")) > 1 or len(_repeat_runs(r, "runs_y")) > 1
    ]
    if same_rows:
        paras.append(f"{STABILITY_DIMENSION}维度题目内置两次运行（temperature 0.7 → 0.0）校验输出一致性：")
        for r in same_rows:
            rx = _repeat_runs(r, "runs_x")
            ry = _repeat_runs(r, "runs_y")
            x_same = len(set(rx)) == 1 if len(rx) > 1 else None
            y_same = len(set(ry)) == 1 if len(ry) > 1 else None
            x_txt = "一致" if x_same else ("不一致" if x_same is False else "无数据")
            y_txt = "一致" if y_same else ("不一致" if y_same is False else "无数据")
            paras.append(f"  {r['id']}：{wname} 两次输出{x_txt}，{lname} 两次输出{y_txt}。")
    sections.append({"title": "稳定性", "paragraphs": paras})

    # ---- 7. 建议 ----
    paras = []
    wname = summary["x_model"] or "答案X"
    lname = summary["y_model"] or "答案Y"
    weak_x = sorted([r for r in rows if r["score_x"] <= 6.5], key=lambda r: r["score_x"])
    weak_y = sorted([r for r in rows if r["score_y"] <= 6.5], key=lambda r: r["score_y"])
    if weak_x:
        parts = [f"{r['id']}（{r['dimension']}，{_fmt(r['score_x'])} 分）" for r in weak_x]
        paras.append(f"{wname} 需重点补强：{'、'.join(parts)}，建议针对对应维度优化提示词或微调。")
    if weak_y:
        parts = [f"{r['id']}（{r['dimension']}，{_fmt(r['score_y'])} 分）" for r in weak_y]
        paras.append(f"{lname} 需重点补强：{'、'.join(parts)}，建议针对对应维度优化提示词或微调。")
    if summary["winner"] == "tie":
        paras.append("双方实力接近，若要分出高下，可增大题目量或重复轮次（repeat_n）做稳定性对决。")
    else:
        paras.append(f"当前胜方 {summary['winner_model']} 领先有限，若作为生产选型建议补充更多场景题验证后再定论。")
    lag = None
    for r in rows:
        if r["latency_x"] > 0 and r["latency_y"] > 0:
            gap = abs(r["latency_x"] - r["latency_y"]) / max(r["latency_x"], r["latency_y"])
            if gap > 0.5:
                lag = r
                break
    if lag:
        paras.append(f"延迟敏感场景下，{lag['id']} 两模型耗时相差 {_fmt(abs(lag['latency_x'] - lag['latency_y']))} ms，可考虑为慢速模型启用流式输出或更高并发。")
    if not paras:
        paras.append("无突出短板，双方表现均衡。")
    sections.append({"title": "建议", "paragraphs": paras})

    return sections
