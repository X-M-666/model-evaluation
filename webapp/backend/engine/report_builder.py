# -*- coding: utf-8 -*-
"""报告生成器（纯规则化统计，无 API）：从任务集/答卷/verdict 计算全部指标，
生成 8 类图表数据与逐条引用数值的精确分析文本，写入 report JSON 供前端渲染。"""
from __future__ import annotations

import statistics
from typing import Any

EPS = 0.001


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


def build_report(
    config: dict[str, Any],
    task_set: dict[str, Any],
    answers_a: dict[str, Any] | None,
    answers_b: dict[str, Any] | None,
    verdict: dict[str, Any],
) -> dict[str, Any]:
    """生成完整报告结构：summary + charts + analysis。"""
    tasks = task_set["tasks"]
    revealed = verdict.get("revealed", {})
    x_file = revealed.get("answer_x_file", "a")
    y_file = revealed.get("answer_y_file", "b")
    x_model = revealed.get("answer_x") or (answers_a or {}).get("model", "答案X")
    y_model = revealed.get("answer_y") or (answers_b or {}).get("model", "答案Y")

    pool_x = _group_by_id(answers_a if x_file == "a" else answers_b)
    pool_y = _group_by_id(answers_b if y_file == "b" else answers_a)

    # ---- 逐题数据 ----
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

    summary = _build_summary(config, tasks, rows, verdict)
    charts = _build_charts(tasks, rows, summary)
    analysis = _build_analysis(tasks, rows, summary, charts)

    return {
        "judge_mode": "human",
        "summary": summary,
        "charts": charts,
        "analysis": analysis,
    }


def _build_summary(
    config: dict[str, Any],
    tasks: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    verdict: dict[str, Any],
) -> dict[str, Any]:
    totals = verdict.get("totals", {})
    total_x = round(totals.get("answer_x", 0), 2)
    total_y = round(totals.get("answer_y", 0), 2)
    win_x = sum(1 for s in verdict.get("scores", []) if s.get("winner") == "answer_x")
    win_y = sum(1 for s in verdict.get("scores", []) if s.get("winner") == "answer_y")
    ties = sum(1 for s in verdict.get("scores", []) if s.get("winner") == "tie")
    if total_x > total_y:
        winner = "answer_x"
    elif total_y > total_x:
        winner = "answer_y"
    else:
        winner = "tie"

    dims = list({r["dimension"] for r in rows})
    avg_x = round(total_x / len(rows), 2) if rows else 0
    avg_y = round(total_y / len(rows), 2) if rows else 0

    return {
        "total_questions": len(rows),
        "dimensions": dims,
        "repeat_n": verdict.get("meta", {}).get("repeat_n", 1),
        "x_model": verdict.get("revealed", {}).get("answer_x", ""),
        "y_model": verdict.get("revealed", {}).get("answer_y", ""),
        "total_x": total_x,
        "total_y": total_y,
        "max_score": round(len(rows) * 10, 2),
        "avg_x": avg_x,
        "avg_y": avg_y,
        "win_x": win_x,
        "win_y": win_y,
        "ties": ties,
        "winner": winner,
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
            "x_passed": (r["code_x"] or {}).get("passed", 0),
            "x_total": (r["code_x"] or {}).get("total", 0),
            "y_passed": (r["code_y"] or {}).get("passed", 0),
            "y_total": (r["code_y"] or {}).get("total", 0),
        }
        for r in rows if r["code_x"] or r["code_y"]
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
        f"共 {summary['total_questions']} 题。"
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
    lat_x = [r["latency_x"] for r in rows if r["status_x"] == "ok" and r["latency_x"] > 0]
    lat_y = [r["latency_y"] for r in rows if r["status_y"] == "ok" and r["latency_y"] > 0]
    avg_lx = round(statistics.mean(lat_x), 1) if lat_x else 0
    avg_ly = round(statistics.mean(lat_y), 1) if lat_y else 0
    if lat_x and lat_y:
        paras.append(f"平均响应延迟：{wname} {_fmt(avg_lx)} ms vs {lname} {_fmt(avg_ly)} ms。")
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
    sections.append({"title": "效率剖析", "paragraphs": paras})

    # ---- 4. 成本剖析 ----
    paras = []
    p_x = sum(r["prompt_x"] for r in rows)
    c_x = sum(r["completion_x"] for r in rows)
    p_y = sum(r["prompt_y"] for r in rows)
    c_y = sum(r["completion_y"] for r in rows)
    paras.append(f"输入 token（prompt）：{wname} {p_x} vs {lname} {p_y}；输出 token（completion）：{wname} {c_x} vs {lname} {c_y}。")
    paras.append(f"总 token 消耗：{wname} {p_x + c_x}（输入 {p_x} / 输出 {c_x}），{lname} {p_y + c_y}（输入 {p_y} / 输出 {c_y}）。")
    if c_x + c_y > 0:
        paras.append(f"输出占比：{wname} {_fmt(c_x / (p_x + c_x) * 100 if p_x + c_x else 0)}%，{lname} {_fmt(c_y / (p_y + c_y) * 100 if p_y + c_y else 0)}%（输出 token 越多通常意味着回答越长、潜在成本越高）。")
    if summary["total_y"] > 0:
        paras.append(f"单位得分成本：{wname} 每得 1 分消耗 {_fmt((p_x + c_x) / summary['total_x'] if summary['total_x'] else 0)} token；{lname} 每得 1 分消耗 {_fmt((p_y + c_y) / summary['total_y'])} token。")
    sections.append({"title": "成本剖析", "paragraphs": paras})

    # ---- 5. 代码质量 ----
    code_rows = [r for r in rows if r["code_x"] or r["code_y"]]
    if code_rows:
        paras = []
        for r in code_rows:
            cx, cy = r["code_x"] or {}, r["code_y"] or {}
            tx, ty = cx.get("total", 0), cy.get("total", 0)
            px, py = cx.get("passed", 0), cy.get("passed", 0)
            if tx or ty:
                rate_x = f"{px}/{tx}"
                rate_y = f"{py}/{ty}"
                result = "X 全过" if px == tx and py < ty else ("Y 全过" if py == ty and px < tx else ("双方全过" if px == tx and py == ty else ("均未全过")))
                paras.append(f"{r['id']}（代码题）：{wname} 通过 {rate_x} 组用例，{lname} 通过 {rate_y} 组用例——{result}。")
        if paras:
            paras.append("说明：代码题通过率来自沙箱自动验证（executor 已按测试用例逐组执行），人工评分时已同步展示。")
        sections.append({"title": "代码质量", "paragraphs": paras})

    # ---- 6. 稳定性 ----
    paras = []
    if summary["repeat_n"] > 1 and charts["stability"]:
        st = charts["stability"]
        vol = sorted(rows, key=lambda r: max(r["score_x_std"], r["score_y_std"]), reverse=True)
        paras.append(f"本次评测重复 {summary['repeat_n']} 轮取平均；标准差反映各轮波动。")
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
    same_rows = [r for r in rows if len(r["runs_x"]) > 1 or len(r["runs_y"]) > 1]
    if same_rows:
        paras.append("效率与稳定性维度题目内置两次运行（temperature 0.7 → 0.0）校验输出一致性：")
        for r in same_rows:
            x_same = len(set(r["runs_x"])) == 1 if len(r["runs_x"]) > 1 else None
            y_same = len(set(r["runs_y"])) == 1 if len(r["runs_y"]) > 1 else None
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
