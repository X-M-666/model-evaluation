# -*- coding: utf-8 -*-
"""迭代六：偏见与公平性对照（bias_analysis）单测。

覆盖：分差计算、一致性相似度（embedder 注入与缺省 n-gram 兜底）、
歧视性阈值判定、缺失原版跳过。
"""
import pytest

from backend.engine.perturb import bias_analysis


def _row(mode, score, raw, origin_id="T1"):
    return {"origin_id": origin_id, "mode": mode, "intensity": 1.0,
            "score": score, "raw_answer": raw}


@pytest.mark.asyncio
async def test_pairs_with_diff_and_discrimination():
    rows = [
        _row("原版", 8.0, "这位医生非常专业"),
        _row("属性扰动-性别", 6.0, "这位医生非常专业"),
        _row("属性扰动-地域", 7.5, "这位医生非常专业"),
    ]
    out = await bias_analysis(rows)
    assert len(out["pairs"]) == 2
    by_mode = {p["mode"]: p for p in out["pairs"]}
    assert by_mode["属性扰动-性别"]["score_original"] == 8.0
    assert by_mode["属性扰动-性别"]["score_perturbed"] == 6.0
    assert by_mode["属性扰动-性别"]["diff"] == -2.0
    assert by_mode["属性扰动-性别"]["discriminates"] is True
    assert by_mode["属性扰动-地域"]["discriminates"] is False
    assert out["n_flagged"] == 1


@pytest.mark.asyncio
async def test_consistency_ngram_default():
    rows = [
        _row("原版", 8.0, "医生给出专业建议"),
        _row("属性扰动-性别", 8.0, "医生给出专业建议"),
    ]
    out = await bias_analysis(rows)
    cons = out["pairs"][0]["consistency"]
    assert cons is not None and 0.0 <= cons <= 1.0
    assert cons > 0.9  # 相同文本 n-gram 余弦 ≈ 1


@pytest.mark.asyncio
async def test_consistency_with_injected_embedder():
    calls = []

    async def fake_embedder(texts):
        calls.append(texts)
        return [[1.0, 0.0], [0.6, 0.8]]

    rows = [
        _row("原版", 8.0, "答案一"),
        _row("属性扰动-文化", 8.0, "答案二"),
    ]
    out = await bias_analysis(rows, embedder=fake_embedder)
    assert calls  # embedder 被调用
    assert out["pairs"][0]["consistency"] == 0.6


@pytest.mark.asyncio
async def test_embedder_failure_falls_back_ngram():
    async def bad_embedder(texts):
        raise RuntimeError("embedding 不可用")

    rows = [
        _row("原版", 8.0, "完全一致的答案文本"),
        _row("属性扰动-性别", 8.0, "完全一致的答案文本"),
    ]
    out = await bias_analysis(rows, embedder=bad_embedder)
    assert out["pairs"][0]["consistency"] is not None  # 兜底生效


@pytest.mark.asyncio
async def test_missing_original_skipped():
    rows = [
        _row("原版", 8.0, "A", origin_id="T1"),
        _row("属性扰动-性别", 7.0, "B", origin_id="T2"),  # 无 T2 原版
    ]
    out = await bias_analysis(rows)
    assert out["pairs"] == []
    assert out["n_flagged"] == 0


@pytest.mark.asyncio
async def test_threshold_configurable():
    rows = [
        _row("原版", 8.0, "x"),
        _row("属性扰动-地域", 7.0, "x"),
    ]
    out = await bias_analysis(rows, diff_threshold=0.5)
    assert out["pairs"][0]["discriminates"] is True
    assert out["threshold"] == 0.5
