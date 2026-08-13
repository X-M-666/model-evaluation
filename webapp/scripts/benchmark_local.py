# -*- coding: utf-8 -*-
"""迭代八：本地性能基准脚本（可选运行，不进 CI）。

复用 mock 引擎跑纯管线吞吐（零真实网络）：
- 单模型 N 题执行（mock 上游，模拟 2K/8K/32K 长文本 context 的 token 采集）
- 判别式指标计算 / 生成式单臂评分（mock）
- 输出 JSON 报告：各规模题耗时、token 统计、单题中位数/均值

用法：python -m scripts.benchmark_local [--tasks 20] [--rounds 3]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _longtext_tasks(n_each: int) -> list[dict]:
    """构造 2K/8K/32K 三档长文本题（材料入 context，与数据资产同形态）。"""
    tasks: list[dict] = []
    for size, chars in (("2K", 2000), ("8K", 8000), ("32K", 31900)):
        for i in range(n_each):
            tasks.append({
                "id": f"LT{size}-{i}",
                "type": "判别式",
                "dimension": "长文本与效率稳定性",
                "prompt": "阅读参考文档，营业收入是多少亿元？",
                "expected": "1.00亿元",
                "context": "长文材料。" * (chars // 5),
            })
    return tasks


async def _mock_execute(model_label, config, tasks, stability_repeat,
                        progress_cb=None, embedding_cfg=None, skip_ids=None,
                        persist_cb=None):
    """mock 执行：同步返回，token 与 context 规模成正比（模拟真实读取成本）。"""
    answers = []
    for t in tasks:
        ctx_chars = len(t.get("context", ""))
        answers.append({
            "id": t["id"],
            "raw_answer": "1.00亿元",
            "api_info": {"status": "ok", "attempts": 1, "truncated": False,
                         "error": None, "latency_ms": ctx_chars // 40,
                         "prompt_tokens": ctx_chars // 3 + 500,
                         "completion_tokens": 20, "repeat_index": 1},
        })
    return {"model": config["name"], "answers": answers}


async def _run_benchmark(tasks: list[dict], rounds: int) -> dict:
    """逐档执行并统计：每档聚合各轮单题耗时（ms）。"""
    from backend.engine.metrics import compute_task_metrics

    groups: dict[str, list[dict]] = {}
    for t in tasks:
        groups.setdefault(t["id"].rsplit("-", 1)[0], []).append(t)

    results: dict[str, dict] = {}
    for label, group in groups.items():
        per_round: list[float] = []
        tokens_total = 0
        for _ in range(rounds):
            t0 = time.perf_counter()
            out = await _mock_execute("A", {"name": label}, group, None)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            per_round.append(elapsed_ms / max(1, len(group)))
            for a in out["answers"]:
                tokens_total += a["api_info"]["prompt_tokens"] + a["api_info"]["completion_tokens"]
            # 指标计算链路（含 32K context 不抛错）
            for t, a in zip(group, out["answers"]):
                compute_task_metrics(t, [a])
        results[label] = {
            "n_tasks": len(group),
            "rounds": rounds,
            "per_task_ms_avg": round(statistics.fmean(per_round), 2),
            "per_task_ms_median": round(statistics.median(per_round), 2),
            "tokens_per_round": tokens_total,
            "tokens_per_task": round(tokens_total / (rounds * len(group))),
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="本地性能基准（mock，零网络）")
    parser.add_argument("--tasks", type=int, default=20, help="每档题数（默认 20）")
    parser.add_argument("--rounds", type=int, default=3, help="轮次（默认 3）")
    args = parser.parse_args()

    tasks = _longtext_tasks(args.tasks)
    print(f"基准开始：3 档 × {args.tasks} 题 × {args.rounds} 轮（mock 上游，零网络）")
    t0 = time.perf_counter()
    results = asyncio.run(_run_benchmark(tasks, args.rounds))
    elapsed = time.perf_counter() - t0
    report = {
        "config": {"tasks_per_scale": args.tasks, "rounds": args.rounds},
        "total_elapsed_sec": round(elapsed, 2),
        "by_scale": results,
    }
    out = Path(__file__).resolve().parent.parent / "benchmark_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"报告已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
