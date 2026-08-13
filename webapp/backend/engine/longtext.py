# -*- coding: utf-8 -*-
"""内置长文本基准集（迭代八）：2K/8K/32K 三档材料内嵌细节提取。

与 rag_demo / gold demo 同一模式：评测集目录尚无 longtext_bench 时在
lifespan 启动写入（source="builtin_demo"），用户同名覆盖后不回退。

设计约束（D1/D6）：材料全部放 context（≤MAX_CONTEXT_LEN=32000），
prompt 保持简短（≤MAX_PROMPT_LEN=20000）；不入内置题库池（build_task_set
按维度抽题无 tags 过滤，入池会污染普通评测成本），仅作为数据资产按需选用。
内容为确定性生成的虚构公司年报，每档嵌入唯一事实（营收/城市/项目/员工数），
判别式题，tags=["长文本","基准"]。
"""
from __future__ import annotations

from typing import Any

from backend import storage
from backend.engine.datasets import validate_standard_dataset

DEMO_NAME = "longtext_bench"
DEMO_DESC = (
    "长文本基准（迭代八）：2K/8K/32K 三档材料内嵌细节提取，"
    "材料在 context 字段，source=builtin_demo"
)

_FACTS: dict[str, dict[str, Any]] = {
    "LT2K": {"revenue": "4.26", "city": "长春", "project": "北极光",
             "staff": 1280, "year": 2025},
    "LT8K": {"revenue": "9.83", "city": "泉州", "project": "鲲鹏引擎",
             "staff": 3420, "year": 2024},
    "LT32K": {"revenue": "17.05", "city": "贵阳", "project": "深海算网",
              "staff": 5600, "year": 2026},
}


def _paragraph(no: int, year: int) -> str:
    """一段约 160 字的填充段（确定性模板，含大量无关数字以增加提取难度）。"""
    return (
        f"第{no}节 经营回顾（{year}年度）本期集团完成营业收入目标值的"
        f"87.3%～94.6%区间波动，其中华南区贡献{no % 7 + 3}.{no % 5}1亿元，"
        f"华北区{no % 9 + 1}.{no % 3}4亿元，海外业务占比{no % 13 + 5}.{no % 7}%。"
        f"研发投入合计{no % 11 + 2}.{no % 6}2亿元，同比增长{no % 8 + 6}.{no % 4}%。"
        f"全年累计召开董事会{no % 4 + 6}次，审议议案{no % 17 + 22}项，"
        f"均获全票通过。重要在建项目共{no % 5 + 9}个，总投资规模约"
        f"{no % 9 + 12}.{no % 7}亿元。\n"
    )


def _fact_paragraph(no: int, year: int, key: str, value: Any) -> str:
    label = {
        "revenue": f"本财年经审计合并营业收入为 {value} 亿元",
        "city": f"新设立的首个区域研发中心位于 {value}",
        "project": f"旗舰级人工智能平台项目代号为「{value}」",
        "staff": f"期末在职员工总数达到 {value} 人",
        "year": f"本报告覆盖财年为 {value} 年",
    }[key]
    return (
        f"第{no}节 关键披露：{label}（该数据经独立审计机构核验，"
        f"与上年同期相比口径一致，可交叉验证）。其余经营数据以附录为准。\n"
    )


def build_context(facts: dict[str, Any], target_chars: int) -> str:
    """确定性生成长文材料（≈target_chars 字符，绝不超限）。"""
    parts: list[str] = []
    total = 0
    i = 1
    keys = list(facts.keys())
    ki = 0
    while total < target_chars:
        if i % 11 == 3 and ki < len(keys):
            para = _fact_paragraph(i, facts["year"], keys[ki], str(facts[keys[ki]]))
            ki += 1
        else:
            para = _paragraph(i, facts["year"])
        if total + len(para) > target_chars and parts:
            break
        parts.append(para)
        total += len(para)
        i += 1
    text = "".join(parts)
    return text[:target_chars]


def build_tasks() -> list[dict[str, Any]]:
    tasks = []
    for tid, facts in _FACTS.items():
        chars = int(tid[2:-1]) * 1000      # LT2K→2000, LT8K→8000, LT32K→32000
        ctx = build_context(facts, chars)
        tasks.append({
            "id": tid,
            "type": "判别式",
            "dimension": "长文本与效率稳定性",
            "difficulty": "hard",
            "benchmark": "长文本基准（材料内嵌细节提取）",
            "tags": ["长文本", "基准"],
            "prompt": "阅读以下参考文档，回答：本财年经审计的合并营业收入是多少亿元？"
                      "（仅输出数字与单位）",
            "expected": f"{facts['revenue']}亿元",
            "test_cases": [
                {"input": "阅读参考文档，营业收入是多少亿元？",
                 "expected": f"{facts['revenue']}亿元"},
            ],
            "context": ctx,
        })
    return validate_standard_dataset({
        "name": DEMO_NAME,
        "description": DEMO_DESC,
        "tasks": tasks,
    })["tasks"]


def ensure_longtext_bench() -> None:
    """评测集目录尚无 longtext_bench 数据集时写入内置长文本基准（幂等）。"""
    if storage.load_dataset(DEMO_NAME) is not None:
        return
    storage.save_dataset(DEMO_NAME, {
        "name": DEMO_NAME,
        "description": DEMO_DESC,
        "source": "builtin_demo",
        "version": "v1",
        "tasks": build_tasks(),
    })
