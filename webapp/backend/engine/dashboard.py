# -*- coding: utf-8 -*-
"""KPI 看板聚合（迭代六）：从历史 job 汇总耗时/token 趋势。

build_jobs_trend 为纯函数：入参 storage.list_jobs() 输出 + 可注入的
report 加载器（缺省读磁盘 report.json），便于确定性测试；历史记录
（迭代五之前无 kpi.duration_sec）相应字段置 None（前端空态 N/A）。
"""
from __future__ import annotations

from typing import Any, Callable


def _sum_tokens(kpi: dict[str, Any]) -> int | None:
    t = kpi.get("total_tokens") or {}
    x, y = t.get("x"), t.get("y")
    if x is None and y is None:
        return None
    return int((x or 0) + (y or 0))


def _default_loader(job_id: str) -> dict[str, Any] | None:
    from backend import storage

    files = storage.get_job_files(job_id)
    if not files or not isinstance(files.get("report.json"), dict):
        return None
    report = files["report.json"].get("report")
    return report if isinstance(report, dict) else None


def build_jobs_trend(
    jobs: list[dict[str, Any]],
    report_loader: Callable[[str], dict[str, Any] | None] | None = None,
) -> list[dict[str, Any]]:
    """按创建时间聚合已完成的 job 序列（KPI 看板数据源）。

    jobs：storage.list_jobs() 输出；report_loader(job_id) -> report 或 None。
    仅 completed 且有 kpi 的 job 入列；duration_sec/total_tokens 缺失 → None。
    """
    loader = report_loader or _default_loader
    out: list[dict[str, Any]] = []
    for j in jobs:
        if j.get("state") != "completed":
            continue
        report = loader(j["job_id"])
        kpi = (report or {}).get("kpi") or {}
        out.append({
            "job_id": j["job_id"],
            "model_a": j.get("model_a"),
            "model_b": j.get("model_b"),
            "created_at": j.get("created_at"),
            "duration_sec": kpi.get("duration_sec"),
            "total_tokens": _sum_tokens(kpi),
        })
    return out
