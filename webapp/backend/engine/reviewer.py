# -*- coding: utf-8 -*-
"""评审方抽象（迭代三）：Reviewer 协议 + AgentReviewer / HumanReviewer。

本轮仅提供协议与实现 + 单测，main.py 不引入分发（分发在步骤 7 以
hybrid 状态流接入）。不做 plugin 机制：评审方固定两类，避免过度设计
（L2：ReviewConfig 不加 arm 字段，单臂能力由 AgentReviewer(protocol=
"single") 构造参数表达）。
"""
from __future__ import annotations

from typing import Any, Protocol

from backend.engine.human_review import build_final_verdict, build_round_verdict
from backend.engine.judge import run_judge, run_single_arm_judge

PAIRWISE_PROTOCOL = "pairwise"
SINGLE_PROTOCOL = "single"

_PROTOCOLS = frozenset({PAIRWISE_PROTOCOL, SINGLE_PROTOCOL})


class Reviewer(Protocol):
    """评审方统一协议：对任务集与答卷执行一轮完整评审，返回 verdict 结构。

    answers 为 {"a": 答卷a, "b": 答卷b | None}；pairwise 需要双份，
    single 仅用 "a"。config 各实现自定（Agent 喂 judge/revealed，
    Human 喂打分/揭示）。
    """

    async def review_all(self, task_set: dict[str, Any],
                         answers: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        ...


class AgentReviewer:
    """Agent 评审方：包装 judge.py（pairwise → run_judge / single →
    run_single_arm_judge）。

    config 约定：
        {"judge": {"url", "key", "name"},
         "revealed": {"rounds": [...], "per_task": [...]} | None,   # pairwise
         "arm": True}                                               # single 必填
    """

    def __init__(self, protocol: str = PAIRWISE_PROTOCOL) -> None:
        if protocol not in _PROTOCOLS:
            raise ValueError(f"未知评审协议：{protocol!r}（可选 {sorted(_PROTOCOLS)}）")
        self.protocol = protocol

    async def review_all(self, task_set: dict[str, Any],
                         answers: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        judge_config = config["judge"]
        if self.protocol == PAIRWISE_PROTOCOL:
            return await run_judge(
                task_set,
                answers["a"], answers["b"],
                judge_config,
                revealed=config.get("revealed"),
            )
        return await run_single_arm_judge(
            task_set, answers["a"], judge_config,
        )


class HumanReviewer:
    """人工评审方：薄封装 human_review 的提交语义（无网络）。

    协议先行：本轮提供结构与单测，main.py 不引入分发；步骤 7 的
    hybrid 复核提交直接复用 build_round_verdict / merge_hybrid_verdicts。
    """

    def __init__(self) -> None:
        self.protocol = "human"

    @staticmethod
    def build_round(task_set: dict[str, Any], round_scores: list[dict[str, Any]],
                    round_reveal: dict[str, str], x_model: str, y_model: str,
                    round_idx: int) -> dict[str, Any]:
        return build_round_verdict(task_set, round_scores, round_reveal,
                                   x_model, y_model, round_idx)

    @staticmethod
    def finalize(round_verdicts: list[dict[str, Any]], repeat_n: int,
                 per_task_reveal: dict[str, str] | None = None) -> dict[str, Any]:
        return build_final_verdict(round_verdicts, repeat_n,
                                   per_task_reveal=per_task_reveal)