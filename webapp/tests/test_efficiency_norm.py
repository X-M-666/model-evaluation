# -*- coding: utf-8 -*-
"""迭代十二「软记录 + 效率归一化」：效率归一化 / 判别力档位 / 推导一致性 /
聚类稳健 CI / 报告效率段与警示 / 排行榜效率与迁移性 单元测试。

覆盖（对齐 docs/效率归一化提示词.md）：
- 改动一：schemas 预算声明字段 + save_config 落盘 + 正向接口透传
- 改动二：token_efficiency / normalize_efficiency / KPI efficiency 段 /
  成本剖析归一化段落 / 得分近效率差警示
- 改动三：leaderboard efficiency 段（每 1K token 得分 + 共用归一基准）
- 改动四：算力匹配警示（超预算 1.5× / token 差 >5× / 软记录提示）
- 改动A：discriminative_band 档位 + 任务集 history_rate/discriminative 标注 +
  too_easy 维度饱和提示
- 改动B：coherence 指标 + logical_contradiction 标记与报告高亮
- 改动C：leaderboard transferability 标注（consistent / flips_across_datasets）
- 改动D：prompt_strategy 混淆变量警示
- 改动E：efficiency_frontier 拐点
- 改动F：clustered_bootstrap_ci 与报告聚类 CI 对照、聚类翻转警示
- N1：answer_redundancy 冗余度/分化度 + 报告 metrics/analysis/warnings 接线
- N2：numeric_drift 数值漂移（>5%）+ 判别式接入 + 报告高亮
- N3：leaderboard aci 任务锚定能力指数（跨评测集平均得分率）
"""
from __future__ import annotations

import pytest

from backend import storage
from backend.engine.metrics import (
    answer_redundancy,
    compute_task_metrics,
    coherence,
    efficiency_frontier,
    normalize_efficiency,
    numeric_drift,
    token_efficiency,
)
from backend.engine.report_builder import build_report
from backend.engine.stats import (
    MIN_SAMPLE,
    clustered_bootstrap_ci,
    paired_bootstrap_ci,
)
from backend.engine.tasks import (
    DISCRIMINATIVE_GOOD,
    DISCRIMINATIVE_TOO_EASY,
    DISCRIMINATIVE_TOO_HARD,
    build_task_set,
    discriminative_band,
)

# ---------------------------------------------------------------------------
# 改动一：预算声明（软记录）
# ---------------------------------------------------------------------------


def test_start_request_budget_cap_field():
    from backend.schemas import StartRequest, ModelConfig, BenchmarkRequest

    mc = {"url": "https://api.example.com/v1", "key": "k",
          "name": "m", "max_tokens": 4096}
    req = StartRequest(model_a=ModelConfig(**mc), model_b=ModelConfig(**mc),
                       budget_cap_tokens=8000)
    assert req.budget_cap_tokens == 8000
    assert StartRequest(model_a=ModelConfig(**mc), model_b=ModelConfig(**mc),
                        budget_cap_tokens=0).budget_cap_tokens == 0

    mc2 = dict(mc)
    mc2["name"] = "m2"
    breq = BenchmarkRequest(models=[ModelConfig(**mc), ModelConfig(**mc2)],
                            budget_cap_tokens=20000)
    assert breq.budget_cap_tokens == 20000


def test_save_config_persists_budget_cap():
    jid = storage.create_job_id()
    storage.save_config(jid, {"model_a": {"name": "A", "url": "https://x"},
                              "model_b": {"name": "B", "url": "https://x"},
                              "budget_cap_tokens": 5000})
    cfg = storage.get_job_files(jid)["config.json"]
    assert cfg["budget_cap_tokens"] == 5000
    # 旧 job 无该字段 → 落盘 0（软记录默认不设上限）
    jid2 = storage.create_job_id()
    storage.save_config(jid2, {"model_a": {"name": "A", "url": "https://x"},
                               "model_b": {"name": "B", "url": "https://x"}})
    assert storage.get_job_files(jid2)["config.json"]["budget_cap_tokens"] == 0


# ---------------------------------------------------------------------------
# 改动二：效率归一化核心纯函数
# ---------------------------------------------------------------------------


def test_token_efficiency_basic():
    e = token_efficiency(80, completion=10000)
    assert e["tokens"] == 10000
    assert e["score_per_1k_tokens"] == pytest.approx(8.0)
    assert e["cost_per_score"] == pytest.approx(125.0)


def test_token_efficiency_no_score_or_tokens():
    assert token_efficiency(0, completion=100)["score_per_1k_tokens"] is None
    assert token_efficiency(10, completion=0)["score_per_1k_tokens"] is None
    assert token_efficiency(10, completion=0)["tokens"] == 0


def test_normalize_efficiency_uses_global_max():
    normed = normalize_efficiency([2.0, 8.0, None, 4.0])
    assert normed[0] == pytest.approx(0.25)
    assert normed[1] == pytest.approx(1.0)
    assert normed[2] is None
    assert normed[3] == pytest.approx(0.5)
    assert normalize_efficiency([None, None]) == [None, None]


# ---------------------------------------------------------------------------
# 改动A：判别力档位
# ---------------------------------------------------------------------------


def test_discriminative_band_bands():
    assert discriminative_band(0.5) == DISCRIMINATIVE_GOOD
    assert discriminative_band(0.2) == DISCRIMINATIVE_TOO_HARD
    assert discriminative_band(0.05) == DISCRIMINATIVE_TOO_HARD
    assert discriminative_band(0.8) == DISCRIMINATIVE_TOO_EASY
    assert discriminative_band(0.95) == DISCRIMINATIVE_TOO_EASY
    assert discriminative_band(None) == DISCRIMINATIVE_GOOD


def test_build_task_set_attaches_history_rate_label(tmp_path):
    from backend.engine.tasks import _history_rates

    # 无历史数据 → history_rate None / discriminative good
    ts = build_task_set(dims=["知识能力"], seed=1)
    assert ts["tasks"][0]["history_rate"] is None
    assert ts["tasks"][0]["discriminative"] == DISCRIMINATIVE_GOOD

    # 有历史（dataset 过滤生效）：直接测磁盘饱和度写入的读取口径
    from backend import storage as st
    from backend.engine import tasks as tasks_mod

    sat_dir = tmp_path / "stats"
    origin = st.STATS_DIR, st.SATURATION_FILE
    st.STATS_DIR = sat_dir
    st.SATURATION_FILE = sat_dir / "saturation.json"
    try:
        jid = st.create_job_id()
        st.update_saturation(
            jid,
            [{"id": "T9", "dimension": "知识能力", "type": "判别式",
              "answer_x": 9.0, "answer_y": 9.0, "winner": "tie"}],
            dataset="ds_hist",
        )
        rates = tasks_mod._history_rates("ds_hist")
        assert rates.get("T9") == pytest.approx(0.9)
        # 其它评测集不串味
        assert tasks_mod._history_rates("ds_other") == {}
    finally:
        st.STATS_DIR, st.SATURATION_FILE = origin


# ---------------------------------------------------------------------------
# 改动B：推导一致性（coherence）
# ---------------------------------------------------------------------------


def test_coherence_consistent_answer_scores_high():
    raw = "该产品可以安全食用。原料均为常见食材。因此该产品没有问题。"
    out = coherence(raw, "该产品安全吗？")
    assert out["score"] > 0.2
    assert out["flag"] is None


def test_coherence_detects_contradiction():
    raw = "该产品可以安全食用。因此该产品不可以安全食用。"
    out = coherence(raw, "该产品安全吗？")
    assert out["flag"] == "logical_contradiction"
    assert out["score"] <= 0.15


def test_coherence_no_false_positive_on_single_sentence():
    raw = "好的产品需要用心打磨，没有捷径。"
    out = coherence(raw, "怎么做好产品？")
    assert out["flag"] is None


def test_generative_metrics_include_coherence():
    task = {"id": "G", "dimension": "语言能力", "type": "生成式",
            "prompt": "该产品安全吗？", "expected": "参考"}
    entry = {"id": "G", "raw_answer": "该产品可以安全食用。因此该产品没有问题。",
             "api_info": {"status": "ok"}}
    m = compute_task_metrics(task, [entry])
    assert "coherence" in m
    assert m["coherence"]["flag"] is None


# ---------------------------------------------------------------------------
# 改动F：聚类稳健 CI
# ---------------------------------------------------------------------------


def test_clustered_bootstrap_ci_basic():
    deltas = {"c1": [4.0, 3.0], "c2": [5.0], "c3": [3.5, 2.5, 4.5]}
    ci = clustered_bootstrap_ci(deltas, seed=42)
    assert ci[0] > 0
    assert clustered_bootstrap_ci(deltas, seed=42) == ci
    assert clustered_bootstrap_ci({}, seed=1) == (0.0, 0.0)


def test_clustered_ci_wider_than_per_item():
    # 簇内高度相关（同一簇内差值完全相同）→ 聚类 CI 应比逐题 bootstrap 更宽或相当
    deltas = {
        "d1": [3.0, 3.0, 3.0], "d2": [3.0, 3.0, 3.0],
        "d3": [3.0, 3.0, 3.0], "d4": [3.0, 3.0, 3.0],
    }
    items = [v for xs in deltas.values() for v in xs]
    cl = clustered_bootstrap_ci(deltas, seed=7)
    pi = paired_bootstrap_ci(items, [0.0] * len(items), seed=7)
    width_cl = cl[1] - cl[0]
    width_pi = pi[1] - pi[0]
    assert width_cl >= width_pi - 1e-6


# ---------------------------------------------------------------------------
# 改动二/三/四/C/D/E：报告与排行榜集成
# ---------------------------------------------------------------------------


def _task_set():
    return {"tasks": [
        {"id": "T1", "dimension": "数学能力", "type": "判别式",
         "prompt": "Q1 答案？", "expected": "A", "test_cases": [
             {"input": "?", "expected": "A"}]},
        {"id": "T2", "dimension": "语言能力", "type": "生成式",
         "prompt": "该产品安全吗？", "expected": "参考"},
    ]}


def _answers():
    def _mk(raw):
        return {"answers": [
            {"id": "T1", "raw_answer": "A", "api_info": {
                "status": "ok", "latency_ms": 100,
                "prompt_tokens": 500, "completion_tokens": 500}},
            {"id": "T2", "raw_answer": raw, "api_info": {
                "status": "ok", "latency_ms": 100,
                "prompt_tokens": 400, "completion_tokens": 600}},
        ]}
    return (_mk("该产品可以安全食用。因此该产品没有问题。"),
            _mk("该产品可以安全食用。因此该产品没有问题。"))


def _verdict():
    return {
        "meta": {"repeat_n": 1, "invalid": 0, "total": 2},
        "scores": [
            {"id": "T1", "dimension": "数学能力", "answer_x": 7.0, "answer_y": 7.0,
             "winner": "tie"},
            {"id": "T2", "dimension": "语言能力", "answer_x": 8.0, "answer_y": 8.0,
             "winner": "tie"},
        ],
        "totals": {"answer_x": 15.0, "answer_y": 15.0},
        "revealed": {"answer_x": "模型X", "answer_y": "模型Y",
                     "answer_x_file": "a", "answer_y_file": "b"},
        "winner_model": "tie",
    }


def test_report_kpi_efficiency_segment():
    r = build_report({"repeat_n": 1, "review": {"mode": "human"},
                      "budget_cap_tokens": 10000},
                     _task_set(), *_answers(), _verdict())
    eff = r["kpi"]["efficiency"]
    assert eff["x"]["tokens"] == 2000 and eff["y"]["tokens"] == 2000
    assert eff["x"]["score_per_1k_tokens"] == pytest.approx(7.5)
    assert eff["x"]["norm_score"] == pytest.approx(1.0)
    assert "10000" in eff["note"]
    # 旧 job 无 budget_cap_tokens → 报告顶层 0 且不报超限
    assert r["budget_cap_tokens"] == 10000
    assert not any(w["code"] == "compute_budget_exceeded" for w in r["warnings"])


def test_report_analysis_cost_normalization_paragraph():
    r = build_report({"repeat_n": 1, "review": {"mode": "human"},
                      "prompt_strategy": "cot"},
                     _task_set(), *_answers(), _verdict())
    cost = next(s for s in r["analysis"] if s["title"] == "成本剖析")
    joined = "".join(cost["paragraphs"])
    assert "归一化效率" in joined and "每 1K token 得分" in joined


def test_report_warning_score_close_cost_far():
    a, b = _answers()
    b["answers"][0]["api_info"]["completion_tokens"] = 4000  # Y 更贵
    r = build_report({"repeat_n": 1, "review": {"mode": "human"}}, _task_set(), a, b,
                     _verdict())
    codes = [w["code"] for w in r["warnings"]]
    assert "efficiency_score_close_cost_far" in codes


def test_report_warning_compute_mismatch_soft_record():
    a, b = _answers()
    b["answers"][0]["api_info"]["completion_tokens"] = 50000
    b["answers"][1]["api_info"]["completion_tokens"] = 60000
    r = build_report({"repeat_n": 1, "review": {"mode": "human"}}, _task_set(), a, b,
                     _verdict())
    codes = [w["code"] for w in r["warnings"]]
    assert "compute_mismatch" in codes
    msg = next(w["message"] for w in r["warnings"]
               if w["code"] == "compute_mismatch")
    assert "未设硬性算力上限（软记录）" in msg


def test_report_warning_budget_cap_exceeded():
    a, b = _answers()
    a["answers"][0]["api_info"]["completion_tokens"] = 90000  # 远超 10000 预算
    r = build_report({"repeat_n": 1, "review": {"mode": "human"},
                      "budget_cap_tokens": 10000}, _task_set(), a, b, _verdict())
    codes = [w["code"] for w in r["warnings"]]
    assert "compute_budget_exceeded" in codes


def test_report_warning_prompt_strategy_confound():
    r = build_report({"repeat_n": 1, "review": {"mode": "human"},
                      "prompt_strategy_a": "cot", "prompt_strategy_b": "direct"},
                     _task_set(), *_answers(), _verdict())
    codes = [w["code"] for w in r["warnings"]]
    assert "prompt_strategy_confound" in codes


def test_report_warning_dimension_saturated():
    ts = _task_set()
    ts["tasks"][0]["history_rate"] = 0.95  # too_easy
    r = build_report({"repeat_n": 1, "review": {"mode": "human"}}, ts,
                     *_answers(), _verdict())
    codes = [w["code"] for w in r["warnings"]]
    assert "task_saturated" in codes
    assert "dimension_saturated" in codes


def test_report_clustered_ci_overall_present():
    r = build_report({"repeat_n": 1, "review": {"mode": "human"}},
                     _task_set(), *_answers(), _verdict())
    cc = r["significance"]["overall"]["ci_clustered"]
    assert cc["cluster_unit"] == "维度"
    assert len(cc["ci"]) == 2


def test_report_warning_coherence_contradiction_highlight():
    a, b = _answers()
    a["answers"][1]["raw_answer"] = "该产品可以安全食用。因此该产品不可以安全食用。"
    r = build_report({"repeat_n": 1, "review": {"mode": "human"}},
                     _task_set(), a, b, _verdict())
    codes = [w["code"] for w in r["warnings"]]
    assert "logical_contradiction" in codes


# ---------------------------------------------------------------------------
# 改动E：成本-性能前沿线
# ---------------------------------------------------------------------------


def test_efficiency_frontier_knee():
    # 前 3 题高增益、后 2 题低增益 → 尾段边际收益显著放缓，应检出拐点
    f = efficiency_frontier([
        (1000, 8.0), (1000, 8.0), (1000, 8.0), (5000, 0.5), (5000, 0.5)])
    assert f["cum_tokens"] == [1000, 2000, 3000, 8000, 13000]
    assert f["cum_score"] == [8.0, 16.0, 24.0, 24.5, 25.0]
    assert f["knee_index"] is not None and f["knee_index"] > 0
    assert f["knee_tokens"] == f["cum_tokens"][f["knee_index"]]
    # 平稳增长无拐点
    g = efficiency_frontier([(1000, 8.0), (1000, 8.0), (1000, 8.0), (1000, 8.0)])
    assert g["knee_index"] is None


# ---------------------------------------------------------------------------
# 改动三/C：排行榜效率与迁移性
# ---------------------------------------------------------------------------


def _lb_job(name, triples):
    """triples: [(tid, answer_x, answer_y)]——模型甲=X 侧、模型乙=Y 侧。"""
    jid = storage.create_job_id()
    storage.save_config(jid, {"model_a": {"name": "模型甲"},
                              "model_b": {"name": "模型乙"},
                              "dataset_name": name})
    storage.save_task_set(jid, {"tasks": [
        {"id": "T1", "dimension": "数学能力", "type": "判别式"},
        {"id": "T2", "dimension": "语言能力", "type": "判别式"},
    ]})

    def _mk(label, get_score):
        return {"model": label, "answers": [
            {"id": tid, "raw_answer": f"{label}-{tid}", "api_info": {
                "status": "ok", "latency_ms": 100, "prompt_tokens": 100,
                "completion_tokens": 50}} for tid, _, _ in triples]}

    storage.save_answers(jid, "a", _mk("a", lambda t: t[1]))
    storage.save_answers(jid, "b", _mk("b", lambda t: t[2]))
    storage.save_verdict(jid, {
        "scores": [{"id": tid, "answer_x": x, "answer_y": y}
                   for tid, x, y in triples],
        "revealed": {"answer_x": "模型甲", "answer_y": "模型乙",
                     "answer_x_file": "a", "answer_y_file": "b"},
        "totals": {}, "meta": {},
    })
    storage.save_report(jid, {"report": {}})
    return jid


def test_leaderboard_efficiency_and_transferability():
    from backend.engine.leaderboard import build_leaderboard

    j1 = _lb_job("ds1", [("T1", 9.0, 6.0), ("T2", 8.0, 6.0)])   # 模型甲 领先
    j2 = _lb_job("ds2", [("T1", 9.0, 6.0), ("T2", 8.0, 6.0)])   # 方向一致
    lb = build_leaderboard([j1, j2])
    eff = lb["efficiency"]
    # 每侧每题 150 token × 2 题 = 300
    assert eff["模型甲"]["tokens"] == 300
    assert eff["模型甲"]["score_per_1k_tokens"] == pytest.approx(
        (9.0 + 8.0) / 0.3)
    assert eff["模型甲"]["norm_score"] == pytest.approx(1.0)
    assert eff["模型乙"]["score_per_1k_tokens"] == pytest.approx(12.0 / 0.3)
    assert eff["模型乙"]["norm_score"] == pytest.approx(12.0 / 17.0, abs=1e-4)
    tr = lb["transferability"]
    pair = tr["per_pair"]["模型乙"]["模型甲"]
    assert pair["status"] == "consistent"
    assert set(pair["datasets"]) == {"ds1", "ds2"}
    assert "efficiency_frontier" in lb


def test_leaderboard_transferability_flips():
    from backend.engine.leaderboard import build_leaderboard

    j1 = _lb_job("dsA", [("T1", 9.0, 6.0), ("T2", 8.0, 6.0)])   # 甲 领先
    j2 = _lb_job("dsB", [("T1", 5.0, 8.0), ("T2", 6.0, 8.0)])   # 乙 领先（翻转）
    lb = build_leaderboard([j1, j2])
    # per_pair 按模型排序（模型乙 在前），pair[0]=模型乙
    pair = lb["transferability"]["per_pair"]["模型乙"]["模型甲"]
    assert pair["status"] == "flips_across_datasets"
    assert pair["flipped"] == {"dsB": "m1_ahead"}


# ---------------------------------------------------------------------------
# N1：答案冗余度/分化度
# ---------------------------------------------------------------------------


def test_answer_redundancy_identical_and_distinct():
    same = answer_redundancy("答案完全一致的内容", "答案完全一致的内容")
    assert same["cosine"] == pytest.approx(1.0)
    assert same["band"] == "same-ish"
    diff = answer_redundancy("量子物理方程推导", "今天天气真好去公园散步")
    assert diff["cosine"] <= 0.3
    assert diff["band"] == "distinct"


def test_answer_redundancy_prefers_embedding_vector():
    r = answer_redundancy("任意文本", "任意文本",
                          sem_a={"vector": [1.0, 0.0, 0.0]},
                          sem_b={"vector": [1.0, 0.0, 0.0]})
    assert r["source"] == "embedding"
    assert r["cosine"] == pytest.approx(1.0)
    r2 = answer_redundancy("任意文本", "不同文本",
                           sem_a={"vector": [1.0, 0.0, 0.0]},
                           sem_b={"vector": [0.0, 1.0, 0.0]})
    assert r2["source"] == "embedding"
    assert r2["cosine"] == pytest.approx(0.0, abs=1e-4)


def test_answer_redundancy_no_answer():
    r = answer_redundancy("", "有内容")
    assert r["cosine"] is None and r["band"] is None


def test_report_metrics_and_warnings_include_redundancy():
    a, b = _answers()
    a["answers"][0]["raw_answer"] = "该产品完全符合预期效果标准"
    b["answers"][0]["raw_answer"] = "该产品完全符合预期效果标准"
    r = build_report({"repeat_n": 1, "review": {"mode": "human"}},
                     _task_set(), a, b, _verdict())
    pt = next(x for x in r["metrics"]["per_task"] if x["id"] == "T1")
    assert pt["redundancy"]["cosine"] == pytest.approx(1.0)
    assert r["metrics"]["redundancy_avg"] is not None
    codes = [w["code"] for w in r["warnings"]]
    assert "answer_redundant" in codes
    assert any("冗余度" in p for sec in r["analysis"] for p in sec["paragraphs"])


# ---------------------------------------------------------------------------
# N2：数值漂移
# ---------------------------------------------------------------------------


def test_numeric_drift_exact_no_drift():
    d = numeric_drift(["答案是 42。"], "42")
    assert d["drift"] is False
    assert d["flag"] is None


def test_numeric_drift_over_five_percent():
    d = numeric_drift(["答案是 45。"], "42")
    assert d["drift"] is True
    assert d["flag"] == "numeric_drift"
    assert d["max_delta_ratio"] == pytest.approx(3 / 42, abs=1e-4)


def test_numeric_drift_cross_run_inconsistency():
    d = numeric_drift(["结果是 100。", "结果是 110。"], "无")
    assert d["drift"] is True
    assert d["values"] == [100.0, 110.0]


def test_numeric_drift_no_numeric_value():
    d = numeric_drift(["没有数字的回答"], "42")
    assert d["drift"] is None


def test_discriminative_metric_includes_numeric_drift():
    task = {"id": "M1", "dimension": "数学能力", "type": "判别式",
            "prompt": "计算结果", "test_cases": [
                {"input": "?", "expected": "42"}]}
    entry = {"id": "M1", "raw_answer": "答案是 45。",
             "api_info": {"status": "ok"}}
    m = compute_task_metrics(task, [entry])
    assert m["numeric_drift"]["drift"] is True


def test_report_warning_numeric_drift():
    a, b = _answers()
    ts = _task_set()
    ts["tasks"][0]["test_cases"] = [{"input": "?", "expected": "100"}]
    a["answers"][0]["raw_answer"] = "结果是 120。"
    r = build_report({"repeat_n": 1, "review": {"mode": "human"}}, ts, a, b,
                     _verdict())
    codes = [w["code"] for w in r["warnings"]]
    assert "numeric_drift" in codes


# ---------------------------------------------------------------------------
# N3：任务锚定能力指数（ACI）
# ---------------------------------------------------------------------------


def test_leaderboard_aci_single_and_multi_dataset():
    from backend.engine.leaderboard import build_leaderboard

    j1 = _lb_job("ds1", [("T1", 9.0, 6.0), ("T2", 8.0, 6.0)])
    lb = build_leaderboard([j1])
    aci = lb["aci"]
    assert aci["模型甲"]["aci"] == pytest.approx(17.0 / 20.0)
    assert aci["模型乙"]["aci"] == pytest.approx(12.0 / 20.0)
    assert set(aci["模型甲"]["per_dataset"]) == {"ds1"}

    j2 = _lb_job("ds2", [("T1", 5.0, 8.0), ("T2", 6.0, 8.0)])
    lb2 = build_leaderboard([j1, j2])
    aci2 = lb2["aci"]
    # 跨两评测集 last-wins 合并：模型甲 11 分、模型乙 16 分
    assert aci2["模型甲"]["aci"] == pytest.approx(11.0 / 20.0)
    assert aci2["模型甲"]["per_dataset"]["ds1"] == pytest.approx(17.0 / 20.0)
    assert aci2["模型甲"]["per_dataset"]["ds2"] == pytest.approx(11.0 / 20.0)
    assert aci2["模型乙"]["per_dataset"]["ds1"] == pytest.approx(12.0 / 20.0)
    assert aci2["模型乙"]["per_dataset"]["ds2"] == pytest.approx(16.0 / 20.0)