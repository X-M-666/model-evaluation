# -*- coding: utf-8 -*-
"""预算熔断单元测试（迭代二 步骤8）。

覆盖：估算公式（执行+评审两段）、max_tokens=0 不限制、hard 超限拒绝、
warn 超限放行、刚好等于上限放行、向后兼容缺省。
"""
from __future__ import annotations

from backend.engine.budget import (
    EXEC_PER_TASK_TOKENS,
    JUDGE_PER_TASK_TOKENS,
    check_budget,
    estimate_tokens,
)


def test_estimate_human_no_judge_cost():
    e = estimate_tokens(10, 1, "pure_human")
    assert e["execution_tokens"] == 10 * EXEC_PER_TASK_TOKENS
    assert e["judging_tokens"] == 0
    assert e["total"] == e["execution_tokens"]
    assert e["tasks"] == 10 and e["rounds"] == 1


def test_estimate_agent_includes_judge():
    e = estimate_tokens(10, 1, "pure_agent")
    assert e["judging_tokens"] == 10 * JUDGE_PER_TASK_TOKENS
    assert e["total"] == 10 * (EXEC_PER_TASK_TOKENS + JUDGE_PER_TASK_TOKENS)


def test_estimate_rounds_multiplies_execution():
    e = estimate_tokens(10, 3, "pure_agent")
    assert e["execution_tokens"] == 10 * 3 * EXEC_PER_TASK_TOKENS
    assert e["judging_tokens"] == 10 * JUDGE_PER_TASK_TOKENS


def test_estimate_zero_tasks():
    e = estimate_tokens(0, 1, "pure_agent")
    assert e["total"] == 0


def test_check_budget_unlimited_by_default():
    r = check_budget(None, 10, 1, "pure_agent")
    assert r["limited"] is False and r["allowed"] is True and r["limit"] == 0


def test_check_budget_zero_means_unlimited():
    r = check_budget({"max_tokens": 0, "mode": "hard"}, 10, 1, "pure_agent")
    assert r["limited"] is False and r["allowed"] is True


def test_check_budget_hard_over_rejects():
    r = check_budget({"max_tokens": 100_000, "mode": "hard"}, 10, 1, "pure_agent")
    assert r["limited"] is True
    assert r["allowed"] is False
    assert r["mode"] == "hard"
    assert r["exceed"] == 10 * (EXEC_PER_TASK_TOKENS + JUDGE_PER_TASK_TOKENS) - 100_000
    assert r["estimated"] == 130_000


def test_check_budget_warn_over_allows_but_flags():
    r = check_budget({"max_tokens": 100_000, "mode": "warn"}, 10, 1, "pure_agent")
    assert r["limited"] is True
    assert r["allowed"] is True
    assert r["mode"] == "warn"
    assert r["exceed"] > 0


def test_check_budget_exactly_at_limit_allowed():
    total = 10 * (EXEC_PER_TASK_TOKENS + JUDGE_PER_TASK_TOKENS)
    r = check_budget({"max_tokens": total, "mode": "hard"}, 10, 1, "pure_agent")
    assert r["allowed"] is True and r["exceed"] == 0


def test_check_budget_human_cheaper_may_pass():
    r = check_budget({"max_tokens": 60_000, "mode": "hard"}, 10, 1, "pure_human")
    assert r["allowed"] is True
    assert r["estimated"] == 50_000


def test_check_budget_within_limit_allowed():
    r = check_budget({"max_tokens": 200_000, "mode": "hard"}, 10, 1, "pure_agent")
    assert r["allowed"] is True and r["exceed"] == 0
