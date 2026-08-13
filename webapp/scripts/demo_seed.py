# -*- coding: utf-8 -*-
"""一键演示数据生成（迭代八，D3：全程零网络，离线直调既有函数）。

产出四类演示数据（写入 .eval/ 与 webapp/data/ 运行产物目录，不碰代码）：
1. 双模型 completed 报告 ×3（不同模型组合，同一 seed → 同一任务集，
   含金标元评估段 + bad case 库 + 饱和度数据）——供 report.html 演示
2. N 模型排行榜：3 个 mock job 聚合 → leaderboard（分维度/胜率矩阵/CI/雷达）——
   供 leaderboard.html 演示（含 N 模型 benchmark 形态）
3. 扰动评测 ready 记录（curves/bias 数据）——供 perturb.html 演示
4. KPI 看板数据：由上述 history 自动聚合——供 dashboard.html 演示

用法：
  python -m scripts.demo_seed            # 写入真实运行目录
  python -m scripts.demo_seed --tmp      # 写入临时目录（不污染正式数据）
"""
from __future__ import annotations

import argparse
import random
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---- 覆盖既有任务后恢复的存储目录（--tmp 模式） ----
_ORIG = {}


def _redirect_tmp(tmp: bool) -> None:
    if not tmp:
        return
    from backend import storage
    from backend import models_registry
    import tempfile
    root = Path(tempfile.mkdtemp(prefix="demo_seed_"))
    _ORIG["storage.BASE_DIR"] = storage.BASE_DIR
    _ORIG["storage.PERTURB_DIR"] = storage.PERTURB_DIR
    _ORIG["storage.BADCASES_DIR"] = storage.BADCASES_DIR
    _ORIG["storage.STATS_DIR"] = storage.STATS_DIR
    _ORIG["models_registry.MODELS_DIR"] = models_registry.MODELS_DIR
    storage.BASE_DIR = root / "history"
    storage.PERTURB_DIR = root / "perturb"
    storage.BADCASES_DIR = root / "badcases"
    storage.STATS_DIR = root / "stats"
    models_registry.MODELS_DIR = root / "models"


def _restore_dirs() -> None:
    if not _ORIG:
        return
    import backend.storage as storage
    import backend.models_registry as mr
    storage.BASE_DIR = _ORIG["storage.BASE_DIR"]
    storage.PERTURB_DIR = _ORIG["storage.PERTURB_DIR"]
    storage.BADCASES_DIR = _ORIG["storage.BADCASES_DIR"]
    storage.STATS_DIR = _ORIG["storage.STATS_DIR"]
    mr.MODELS_DIR = _ORIG["models_registry.MODELS_DIR"]


def _seed_dual_job(seed: int) -> str:
    """生成一条 completed 双模型报告（复用 prepare_mock_job + 评审提交等价路径）。"""
    from backend.engine.mock import prepare_mock_job
    from backend.engine.human_review import build_round_verdict, build_final_verdict, resolve_round
    from backend.main import _finalize_job
    from backend import storage

    data = prepare_mock_job(seed=seed)
    job_id = data["job_id"]
    task_set = data["task_set"]
    answers_a, answers_b = data["answers_a"], data["answers_b"]
    reveal = data["reveal"]
    rounds_answers = data["rounds_answers"]

    rng = random.Random(seed)
    # 确定性打分：模型 A 略优（模拟评测结果形态）
    round_scores = []
    for t in task_set["tasks"]:
        ax = round(rng.uniform(6.5, 9.5), 1)
        ay = round(max(3.0, ax - rng.uniform(0.5, 2.0)), 1)
        round_scores.append({"id": t["id"], "answer_x": ax, "answer_y": ay, "note": "演示数据"})

    round_verdicts = []
    for r_idx, round_ans in enumerate(rounds_answers):
        x_model, y_model, _x_pool, _y_pool = resolve_round(reveal, r_idx, round_ans["a"], round_ans["b"])
        round_reveal = reveal["rounds"][r_idx]
        v = build_round_verdict(task_set, round_scores, round_reveal, x_model, y_model, r_idx)
        round_verdicts.append(v)
    verdict = build_final_verdict(round_verdicts, 1)

    _finalize_job(
        job_id, verdict, {"scores": round_scores, "note": "demo_seed"},
        round_verdicts, data["config"], task_set, answers_a, answers_b, rounds_answers,
    )
    storage.save_env_snapshot(job_id, data["config"])   # 环境快照含评测参数段（演示报告卡片）
    return job_id


def _seed_leaderboard(job_ids: list[str], name: str) -> str:
    """由 3 个同任务集 completed job 聚合排行榜（复用迭代六聚合器）。"""
    from backend.engine.leaderboard import build_leaderboard
    from backend import storage
    from backend.main import create_job_id

    lb = build_leaderboard(job_ids, name=name)
    lb_id = "lb_" + create_job_id()
    storage.save_leaderboard(lb_id, {
        "lb_id": lb_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "batch_id": None,
        **lb,
    })
    return lb_id


def _seed_perturb(dataset: str, model_name: str) -> str:
    """落一条 ready 扰动记录（含衰减曲线与偏见对照，纯演示数据）。"""
    from backend import storage
    from backend.main import create_job_id

    perturb_id = "prb_" + create_job_id()
    storage.save_perturb(perturb_id, {
        "perturb_id": perturb_id,
        "state": "ready",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_name": model_name,
        "model": model_name,
        "dataset": dataset,
        "modes": ["改写", "噪声注入", "属性扰动-地域"],
        "progress": "done",
        "seed": 20260801,
        "curves": {
            "curves": {
                "改写": {"intensities": [0, 1.0], "scores": [8.2, 7.4], "n_tasks": [8, 8]},
                "噪声注入": {"intensities": [0, 0.1, 0.2, 0.3],
                             "scores": [8.2, 7.9, 7.2, 6.5], "n_tasks": [8, 8, 8, 8]},
                "属性扰动-地域": {"intensities": [0, 1.0], "scores": [8.2, 8.0], "n_tasks": [6, 6]},
            }
        },
        "bias": {
            "threshold": 1.0,
            "n_flagged": 1,
            "pairs": [
                {"task_id": "T3", "mode": "属性扰动-地域", "score_original": 8.0,
                 "score_perturbed": 6.5, "diff": 1.5, "consistency": 0.81, "discriminates": True},
                {"task_id": "T7", "mode": "属性扰动-地域", "score_original": 8.5,
                 "score_perturbed": 8.4, "diff": 0.1, "consistency": 0.95, "discriminates": False},
            ],
        },
        "warnings": [],
        "per_task": [
            {"task_id": f"P{i}", "origin_id": f"T{i % 8 + 1}", "mode": "改写",
             "intensity": 1.0, "score": round(8.2 - 0.1 * i, 2),
             "latency_ms": 500, "tokens": 300, "raw_answer": "演示答案摘要……"}
            for i in range(4)
        ],
    })
    return perturb_id


def main() -> int:
    parser = argparse.ArgumentParser(description="一键生成演示数据（零网络）")
    parser.add_argument("--tmp", action="store_true", help="写入临时目录（不污染正式数据）")
    parser.add_argument("--seed", type=int, default=20260813, help="演示随机种子")
    args = parser.parse_args()

    _redirect_tmp(args.tmp)
    try:
        t0 = time.perf_counter()
        seed = args.seed

        # 1. 双模型报告 ×3（同一任务集）
        job_ids = []
        for i in range(3):
            jid = _seed_dual_job(seed + i)
            job_ids.append(jid)
            print(f"  [1/4] 双模型报告 {i + 1}: {jid}（completed）")

        # 2. N 模型排行榜（3 个 job 聚合，6 个模型条目）
        lb_id = _seed_leaderboard(job_ids, f"演示榜单-{seed}")
        print(f"  [2/4] 排行榜: {lb_id}")

        # 3. 扰动记录
        from backend.engine.mock import MODEL_POOL
        prb_id = _seed_perturb("rag_demo", MODEL_POOL[0])
        print(f"  [3/4] 扰动记录: {prb_id}（ready）")

        # 4. KPI 看板：由 history 自动聚合（无需额外写入）
        print(f"  [4/4] KPI 看板：{len(job_ids)} 条历史自动聚合")

        elapsed = time.perf_counter() - t0
        mode = "（临时目录，不污染正式数据）" if args.tmp else "（写入正式运行目录）"
        print(f"\n演示数据就绪（{elapsed:.1f}s）{mode}，演示路线：")
        print("  1. 首页  http://localhost:8910/            → 历史记录（3 条 completed）")
        print("  2. 报告  http://localhost:8910/report.html?job=<job_id>  → 全指标/环境快照/元评估/Bad Case")
        print("  3. 排行榜 http://localhost:8910/leaderboard.html?id=" + lb_id)
        print("  4. 扰动  http://localhost:8910/perturb.html? id 见列表")
        print("  5. 看板  http://localhost:8910/dashboard.html")
        return 0
    finally:
        _restore_dirs()


if __name__ == "__main__":
    raise SystemExit(main())
