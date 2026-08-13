# -*- coding: utf-8 -*-
"""评测集解析器注册表：各文件类型解析函数统一输出标准格式。

标准格式：{name, description, tasks: [{id, dimension, benchmark, difficulty,
prompt, test_cases: [{input, expected}], rubric_note}]}

注册表按小写扩展名路由，未知扩展名返回 None（由调用方报错）。
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from backend.engine.datasets import (
    _next_free_id,
    parse_csv_dataset,
    validate_json_dataset,
    validate_standard_dataset,
)


def parse_json(raw: str) -> dict[str, Any]:
    """JSON：包装 datasets.validate_json_dataset。"""
    return validate_json_dataset(raw)


def parse_csv(raw: str) -> dict[str, Any]:
    """CSV：包装 datasets.parse_csv_dataset。"""
    return parse_csv_dataset(raw)


def _strip_md_label(raw: str) -> str:
    """剥离 **标签：** 前缀，返回纯内容。

    兼容 `**题目：** 内容`（闭合星号+空格）与 `**题目：内容**`（尾部闭合星号）。
    """
    body = raw.strip().lstrip("*").strip().rstrip("*").strip()
    for colon in ("：", ":"):
        if colon in body:
            body = body.split(colon, 1)[1].strip()
            break
    return body.lstrip("*").strip().rstrip("*").strip()


def parse_markdown(raw: str) -> dict[str, Any]:
    """Markdown 评测集解析。

    格式约定：
    - `# 标题` → 数据集名称（缺省时自动生成）
    - `> 描述` → 描述
    - `## 维度名` → 当前维度，影响后续所有题目
    - `### 题号` → 题目开始（题号可省略，省略则自动编号）
    - `**题目：** ...` → prompt
    - `**期望：** ...` → 期望答案（写入 test_cases[0].expected）
    - `**评分标准：** ...` → rubric_note（可选）
    - `**类型：** 判别式|生成式` → 任务类型（可选，缺省判别式）
    - `**上下文：**` → context（迭代一，可选；支持多行块：直到下一个
      `**标签：**`、`#`/`##`/`###` 标题或文件尾，块内禁止空行）
    - 题目块内普通段落 → prompt 追加行
    """
    lines = raw.splitlines()
    name = ""
    description = ""
    tasks: list[dict[str, Any]] = []
    cur_dim = "自定义"
    cur_id: str | None = None
    cur_prompt: list[str] = []
    cur_expected: list[str] = []
    cur_rubric: list[str] = []
    cur_type: str = "判别式"
    cur_context: list[str] = []
    in_context_block = False
    auto_no = 0
    md_taken: set[str] = set()

    # 预扫显式 `### X` id：自动编号须避开显式 id（issue #15）
    md_explicit: set[str] = set()
    for line in lines:
        s = line.strip()
        if (s == "###" or s.startswith("### ")) and s[3:].strip():
            md_explicit.add(s[3:].strip())

    def _flush():
        nonlocal cur_id, cur_prompt, cur_expected, cur_rubric, cur_type, cur_context, in_context_block
        if cur_id is None:
            return
        prompt = "\n".join(s for s in cur_prompt if s.strip()).strip()
        if prompt:
            expected = "\n".join(s for s in cur_expected if s.strip()).strip()
            tasks.append({
                "id": cur_id,
                "dimension": cur_dim,
                "benchmark": name or "自定义评测集",
                "difficulty": "进阶",
                "prompt": prompt,
                "test_cases": [{"input": prompt[:50], "expected": expected}] if expected else [],
                "rubric_note": "\n".join(s for s in cur_rubric if s.strip()).strip()
                or "【仅评审可见】满分10分。根据回答与期望答案的匹配程度评分。",
                "type": cur_type,
                "context": "\n".join(s for s in cur_context if s.strip()).strip(),
            })
        cur_id = None
        cur_prompt, cur_expected, cur_rubric = [], [], []
        cur_type = "判别式"
        cur_context = []
        in_context_block = False

    def _flush_labels():
        """开始新标签（题目/期望/评分标准/类型/上下文）时退出上下文多行块。"""
        nonlocal in_context_block
        in_context_block = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("# "):
            _flush()
            name = stripped[2:].strip()
        elif stripped.startswith("## "):
            _flush()
            cur_dim = stripped[3:].strip() or "自定义"
        elif stripped == "###" or stripped.startswith("### "):
            _flush()
            auto_no += 1
            label = stripped[3:].strip()
            if label:
                cur_id = label
            else:
                cur_id = _next_free_id(md_explicit, md_taken, f"T{auto_no}")
                md_taken.add(cur_id)
        elif stripped.startswith(">"):
            if not tasks and not cur_id:
                description = (description + " " + stripped[1:].strip()).strip()
        elif stripped.startswith("**题目：") or stripped.startswith("**题目:"):
            _flush_labels()
            cur_prompt = [_strip_md_label(stripped)]
        elif stripped.startswith("**期望：") or stripped.startswith("**期望:"):
            _flush_labels()
            cur_expected = [_strip_md_label(stripped)]
        elif stripped.startswith("**评分标准：") or stripped.startswith("**评分标准:"):
            _flush_labels()
            cur_rubric = [_strip_md_label(stripped)]
        elif stripped.startswith("**类型：") or stripped.startswith("**类型:"):
            _flush_labels()
            cur_type = _strip_md_label(stripped) or "判别式"
        elif stripped.startswith("**上下文：") or stripped.startswith("**上下文:"):
            _flush_labels()
            in_context_block = True
            first = _strip_md_label(stripped)
            if first:
                cur_context.append(first)
        elif stripped.startswith("**") or stripped.startswith("#"):
            # 其他未知标签/标题：退出上下文块
            in_context_block = False
            if cur_id is not None:
                cur_prompt.append(stripped)
        elif in_context_block and cur_id is not None:
            # 上下文多行块：直到下一个标签或标题
            cur_context.append(stripped)
        elif cur_id is not None:
            cur_prompt.append(stripped)
    _flush()

    if not tasks:
        raise ValueError("Markdown 中未解析到任何题目（需要 **题目：** 或 ### 块）")

    if not name:
        name = f"评测集_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    return validate_standard_dataset({"name": name, "description": description, "tasks": tasks})


def parse_txt(raw: str) -> dict[str, Any]:
    """纯文本评测集解析。

    格式约定：
    - 首行 `# 标题`（可选）→ 数据集名称
    - `# 维度名`（可选，可多处出现）→ 当前维度
    - 其余每行一条：`题目` `期望` `类型` `上下文` 之间用 `|` 或 TAB 分隔
      （2~4 段均可：`题目|期望`、`题目|期望|类型`、`题目|期望|类型|上下文`；
      类型缺省=判别式；上下文段内禁止 `|`；
      无分隔符的行视为上一题的 prompt 续行）
    """
    lines = raw.splitlines()
    name = ""
    description = ""
    tasks: list[dict[str, Any]] = []
    cur_dim = "自定义"
    cur: dict[str, Any] | None = None
    auto_no = 0

    def _flush():
        nonlocal cur
        if cur is None:
            return
        prompt = cur["prompt"].strip()
        if prompt:
            expected = cur["expected"].strip()
            tasks.append({
                "id": cur["id"],
                "dimension": cur["dimension"],
                "benchmark": name or "自定义评测集",
                "difficulty": "进阶",
                "prompt": prompt,
                "test_cases": [{"input": prompt[:50], "expected": expected}] if expected else [],
                "rubric_note": "【仅评审可见】满分10分。根据回答与期望答案的匹配程度评分。",
                "type": cur["type"],
                "context": cur["context"],
            })
        cur = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if not tasks and not cur and not name:
                name = stripped.lstrip("#").strip()
                continue
            _flush()
            cur_dim = stripped.lstrip("#").strip() or "自定义"
            continue
        parts = re.split(r"\s*\|\s*|\t", stripped, maxsplit=3)
        if len(parts) >= 2:
            _flush()
            auto_no += 1
            ttype = parts[2].strip() if len(parts) >= 3 else "判别式"
            context = parts[3].strip() if len(parts) >= 4 else ""
            cur = {"id": f"T{auto_no}", "dimension": cur_dim,
                   "prompt": parts[0].strip(), "expected": parts[1].strip(),
                   "type": ttype or "判别式", "context": context}
        else:
            if cur is None:
                auto_no += 1
                cur = {"id": f"T{auto_no}", "dimension": cur_dim, "prompt": "",
                       "expected": "", "type": "判别式", "context": ""}
            cur["prompt"] += " " + stripped
    _flush()

    if not tasks:
        raise ValueError("TXT 中未解析到任何题目（需要 `题目 | 期望` 行）")

    if not name:
        name = f"评测集_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    return validate_standard_dataset({"name": name, "description": description, "tasks": tasks})


PARSER_REGISTRY: dict[str, Any] = {
    ".json": parse_json,
    ".csv": parse_csv,
    ".md": parse_markdown,
    ".markdown": parse_markdown,
    ".txt": parse_txt,
}


def get_parser(ext: str):
    """按扩展名取解析函数；不支持返回 None。ext 需为小写（含点）。"""
    return PARSER_REGISTRY.get(ext)


def supported_extensions() -> list[str]:
    """返回支持的文件扩展名列表（用于报错信息/前端提示）。"""
    return sorted(PARSER_REGISTRY.keys())
