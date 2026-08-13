# -*- coding: utf-8 -*-
"""预算熔断（迭代二）：启动前预估 token 消耗，warn/hard 两档策略。

- 预估 = 执行阶段（题数 × repeat_n × 每题执行 token 参考）+ 评审阶段
  （pure_agent 时 题数 × 每题评审 token 参考；pure_human 无评审消耗）；
- max_tokens=0 → 不限制（向后兼容）；
- hard：启动即拒绝（HTTP 400）；warn：每轮结束后幂等发一次预算告警
  （幂等由调用方按 job 标记控制，本模块只做纯计算）。

纯函数、确定性可测，无任何网络调用。
"""
from __future__ import annotations

from typing import Any

# 单题执行 token 参考（prompt 题目 + 答案输出，取模型 max_tokens 均值）
EXEC_PER_TASK_TOKENS = 5000
# 单题评审 token 参考（双盲 prompt 含双份答案 + verdict 输出）
JUDGE_PER_TASK_TOKENS = 8000


def estimate_tokens(
    task_count: int,
    repeat_n: int,
    judge_mode: str = "human",
    n_models: int = 1,
) -> dict[str, Any]:
    """预估总 token 消耗（执行 + 评审两段）。

    n_models（迭代七）：benchmark 批次按模型数放大（执行段 ×N、
    评审段 ×N×题数）；旧调用缺省 1，零破坏。
    """
    n = max(task_count, 0)
    rounds = max(repeat_n, 1)
    models = max(n_models, 1)
    exec_tokens = n * rounds * models * EXEC_PER_TASK_TOKENS
    judge_tokens = n * models * JUDGE_PER_TASK_TOKENS if judge_mode == "pure_agent" else 0
    return {
        "execution_tokens": exec_tokens,
        "judging_tokens": judge_tokens,
        "total": exec_tokens + judge_tokens,
        "tasks": n,
        "rounds": rounds,
        "models": models,
        "judge_mode": judge_mode,
    }


def check_budget(
    budget_config: dict[str, Any] | None,
    task_count: int,
    repeat_n: int,
    judge_mode: str = "human",
    n_models: int = 1,
) -> dict[str, Any]:
    """预算检查（纯计算）。

    Returns:
        {
            "limited": bool,          # 是否配置了预算上限
            "allowed": bool,          # 是否放行（未配置/未超限为 True）
            "mode": "warn"|"hard",    # 超限时的策略
            "estimated": int,         # 预估总 token
            "limit": int,             # 配置上限（0=不限制）
            "exceed": int,            # 超出量（未超限为 0）
        }
    """
    cfg = budget_config or {}
    limit = int(cfg.get("max_tokens") or 0)
    mode = cfg.get("mode") or "warn"
    est = estimate_tokens(task_count, repeat_n, judge_mode, n_models)
    if limit <= 0:
        return {"limited": False, "allowed": True, "mode": mode,
                "estimated": est["total"], "limit": 0, "exceed": 0}
    exceed = est["total"] - limit
    # warn：超限仍放行（由调用方在运行中幂等提示）；hard：超限即拒绝
    allowed = mode == "warn" or exceed <= 0
    return {
        "limited": True,
        "allowed": allowed,
        "mode": mode,
        "estimated": est["total"],
        "limit": limit,
        "exceed": max(exceed, 0),
    }
