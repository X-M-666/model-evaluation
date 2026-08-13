# -*- coding: utf-8 -*-
"""迭代八：生成/重生成内置长文本基准集（longtext_bench）。

直接调用 backend.engine.longtext.build_tasks() 生成并写入评测集目录
（启动时 lifespan 也会幂等确保存在；本脚本用于显式重建或人工核对）。

用法：python -m scripts.gen_longtext_bench
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import storage  # noqa: E402
from backend.engine.longtext import DEMO_NAME, build_tasks  # noqa: E402


def main() -> int:
    tasks = build_tasks()
    storage.save_dataset(DEMO_NAME, {
        "name": DEMO_NAME,
        "description": "长文本基准（迭代八）：2K/8K/32K 三档材料内嵌细节提取",
        "source": "builtin_demo",
        "version": "v1",
        "tasks": tasks,
    })
    for t in tasks:
        print(f"  {t['id']}: context={len(t['context'])} chars, prompt={len(t['prompt'])} chars")
    print(f"written longtext_bench（{len(tasks)} 题）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
