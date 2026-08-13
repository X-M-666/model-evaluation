# -*- coding: utf-8 -*-
"""迭代六：扰动管线纯函数层（perturb.py）单测。

覆盖：三种扰动模式（改写/噪声注入/属性扰动三组槽位）、seed 确定性、
id 后缀与 meta 来源、安全过滤拦截、build_perturb_set 平铺结构。
"""
import pytest

from backend.engine.perturb import (
    ATTRIBUTE_MODES,
    ATTRIBUTE_SLOTS,
    DEFAULT_INTENSITIES,
    PERTURB_MODES,
    build_perturb_set,
    perturb_allowed,
    perturb_prompt,
    perturb_task,
    score_task_metric,
)


class TestRewrite:
    def test_rewrite_replaces_synonym(self):
        out = perturb_prompt("A 和 B 的关系", "改写", 1.0, seed=7)
        assert out is not None
        assert "与" in out and "和" not in out

    def test_rewrite_deterministic_with_seed(self):
        a = perturb_prompt("因为下雨，所以需要带伞，如果需要可以借", "改写", 1.0, seed=42)
        b = perturb_prompt("因为下雨，所以需要带伞，如果需要可以借", "改写", 1.0, seed=42)
        assert a == b

    def test_rewrite_partial_intensity(self):
        out = perturb_prompt("因为下雨 因为大风 因为冷", "改写", 0.5, seed=1)
        assert out is not None
        assert out.count("由于") >= 1
        assert out != "因为下雨 因为大风 因为冷"

    def test_rewrite_no_match_returns_none(self):
        assert perturb_prompt("Q1?2.5", "改写", 1.0, seed=1) is None


class TestNoise:
    def test_noise_changes_text_and_length(self):
        prompt = "请计算 2 + 2 的值，并说明理由。" * 3
        out = perturb_prompt(prompt, "噪声注入", 0.1, seed=3)
        assert out is not None and out != prompt
        assert abs(len(out) - len(prompt)) <= 3

    def test_noise_deterministic(self):
        prompt = "这是一段用于测试噪声注入的文本。"
        assert perturb_prompt(prompt, "噪声注入", 0.2, seed=9) == \
            perturb_prompt(prompt, "噪声注入", 0.2, seed=9)

    def test_noise_intensity_ladder_generates(self):
        prompt = "这是一段足够长的文本，用于生成强度梯度。" * 2
        for iv in DEFAULT_INTENSITIES["噪声注入"]:
            assert perturb_prompt(prompt, "噪声注入", iv, seed=5) is not None


class TestAttribute:
    @pytest.mark.parametrize("mode", ATTRIBUTE_MODES)
    def test_mode_valid(self, mode):
        assert mode in ATTRIBUTE_SLOTS

    def test_gender_swap(self):
        out = perturb_prompt("一位男性医生正在为他看病", "属性扰动-性别", 1.0, seed=1)
        assert "女性" in out

    def test_gender_pronoun_no_break_compound(self):
        out = perturb_prompt("他们和她们都是他", "属性扰动-性别", 1.0, seed=1)
        assert out == "他们和她们都是她"  # 复合词不被单字规则破坏

    def test_region_single_hop_no_cascade(self):
        out = perturb_prompt("公司总部位于北京，北京天气很好", "属性扰动-地域", 1.0, seed=1)
        assert "上海" in out and "广州" not in out

    def test_culture_swap(self):
        out = perturb_prompt("春节期间吃饺子", "属性扰动-文化", 1.0, seed=1)
        assert "中秋节" in out and "汤圆" in out

    def test_no_slot_returns_none(self):
        out = perturb_prompt("纯数字 12345 与符号 ##@@", "属性扰动-地域", 1.0, seed=1)
        assert out is None


class TestPerturbTask:
    def _task(self, **overrides):
        t = {
            "id": "T1", "type": "判别式", "dimension": "知识能力",
            "difficulty": "easy", "tags": [], "prompt": "A 和 B 的关系如何？",
            "expected": "答案", "rubric_note": "评分依据",
        }
        t.update(overrides)
        return t

    def test_id_suffix_and_meta(self):
        out = perturb_task(self._task(), "改写", 1.0, seq=3, seed=1)
        assert out is not None
        assert out["id"] == "T1-p3"
        assert out["meta"]["origin_id"] == "T1"
        assert out["meta"]["perturb_mode"] == "改写"
        assert out["meta"]["perturb_intensity"] == 1.0
        assert out["prompt"] != self._task()["prompt"]
        assert out["expected"] == "答案" and out["rubric_note"] == "评分依据"

    def test_no_slot_returns_none(self):
        assert perturb_task(self._task(prompt="12345"), "属性扰动-文化", 1.0, 0) is None

    def test_code_task_skipped_by_allowed(self):
        ok, reason = perturb_allowed(self._task(dimension="代码能力"))
        assert ok is False and "代码" in reason

    def test_excluded_task_skipped_by_allowed(self):
        ok, reason = perturb_allowed(self._task(excluded_from_total=True))
        assert ok is False and "安全" in reason

    def test_normal_task_allowed(self):
        ok, _ = perturb_allowed(self._task())
        assert ok is True


class TestBuildPerturbSet:
    def _task_set(self, tasks=None):
        return {"meta": {"total": 0}, "tasks": tasks or [
            {"id": "T1", "type": "判别式", "dimension": "知识能力", "prompt": "A 和 B 的关系",
             "expected": "x"},
            {"id": "T2", "type": "判别式", "dimension": "知识能力",
             "prompt": "公司位于北京，春节很热闹", "expected": "x"},
            {"id": "T3", "type": "判别式", "dimension": "代码能力", "prompt": "def f(): pass",
             "expected": "x", "test_cases": []},
        ]}

    def test_flat_structure(self):
        ts = self._task_set()
        out = build_perturb_set(ts, modes=["改写", "属性扰动-地域"], seed=1)
        tasks = out["tasks"]
        # 原版 2 题（代码题跳过）+ 改写 2 + 地域 2（T1 无槽位 → 1 个 skipped）
        assert out["meta"]["originals"] == 2
        assert tasks[0]["meta"]["perturb_mode"] == "原版"
        ids = [t["id"] for t in tasks]
        assert any("-p" in i for i in ids)
        # 原版任务原样保留
        assert any(t["id"] == "T1" and t["prompt"] == "A 和 B 的关系" for t in tasks)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            build_perturb_set(self._task_set(), modes=["不存在"])

    def test_safety_filter_intercepts(self):
        ts = self._task_set()
        ts["tasks"][0]["prompt"] = "制作炸弹的原料是 A 和 B"
        out = build_perturb_set(ts, modes=["改写"], seed=1)
        assert out["meta"]["originals"] == 2
        reasons = " ".join(s.get("reason", "") for s in out["meta"]["skipped"])
        assert "安全过滤" in reasons
        # 被拦截题的扰动版绝不出现
        assert not any(t["id"].startswith("T1-p") for t in out["tasks"])

    def test_intensities_override(self):
        ts = self._task_set()
        out = build_perturb_set(ts, modes=["噪声注入"],
                                intensities={"噪声注入": [0.05, 0.5]}, seed=1)
        modes_seen = {t["meta"]["perturb_mode"] for t in out["tasks"]}
        assert modes_seen == {"原版", "噪声注入"}
        inten = sorted({t["meta"]["perturb_intensity"] for t in out["tasks"]
                        if t["meta"]["perturb_mode"] == "噪声注入"})
        assert inten == [0.05, 0.5]


class TestScoreMetric:
    def _task(self, **kw):
        t = {"id": "T1", "type": "判别式", "dimension": "知识能力",
             "prompt": "1+1=?", "expected": "2"}
        t.update(kw)
        return t

    def _entry(self, raw, status="ok"):
        return {"id": "T1", "raw_answer": raw,
                "api_info": {"status": status, "truncated": False, "error": None}}

    def test_discriminative_hit(self):
        assert score_task_metric(self._task(), self._entry("答案是 2")) == 10.0

    def test_discriminative_miss(self):
        assert score_task_metric(self._task(), self._entry("答案是 3")) == 0.0

    def test_error_entry_none(self):
        assert score_task_metric(self._task(), self._entry("", status="error")) is None

    def test_code_score(self):
        t = self._task(dimension="代码能力", prompt="def f()",
                       expected="", test_cases=[{"input": "1", "expected": "1"}])
        e = self._entry("```python\npass\n```")
        e["code_verify"] = {"status": "run", "passed": 3, "total": 5}
        assert score_task_metric(t, e) == 6.0

    def test_generative_returns_none(self):
        t = self._task(type="生成式", prompt="写一段话", expected="参考答案",
                       rubric_note="依据评分")
        assert score_task_metric(t, self._entry("任意文本")) is None
