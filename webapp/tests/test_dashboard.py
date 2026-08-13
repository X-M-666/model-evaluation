# -*- coding: utf-8 -*-
"""迭代六：KPI 看板（hwmon + duration_sec + jobs_trend）单测。

覆盖：_util_from_deltas 纯函数、CpuSampler 注入 fake reader（首采 None/
增量值/平台不支持）、GPU N/A、_job_duration_sec 内存与磁盘两路径、
build_jobs_trend 聚合（含历史记录空态）。
"""
import json
import time

import pytest

from backend import hwmon
from backend.engine.dashboard import build_jobs_trend


class TestUtilFromDeltas:
    def test_busy_ratio(self):
        assert hwmon._util_from_deltas(30.0, 100.0) == 70.0

    def test_zero_total(self):
        assert hwmon._util_from_deltas(0.0, 0.0) == 0.0

    def test_idle_equal_total(self):
        assert hwmon._util_from_deltas(100.0, 100.0) == 0.0

    def test_clamped_100(self):
        assert hwmon._util_from_deltas(-5.0, 100.0) == 100.0


class TestCpuSampler:
    def test_first_sample_returns_none(self):
        s = hwmon.CpuSampler(reader=lambda: (10.0, 100.0))
        assert s.sample() is None

    def test_incremental_utilization(self):
        data = iter([(10.0, 100.0), (30.0, 200.0)])
        s = hwmon.CpuSampler(reader=lambda: next(data))
        assert s.sample() is None
        assert s.sample() == 80.0  # busy 70 / total 100

    def test_unsupported_platform_returns_none(self):
        s = hwmon.CpuSampler(reader=lambda: None)
        assert s.sample() is None

    def test_injected_kind(self):
        s = hwmon.CpuSampler(reader=lambda: (1.0, 2.0))
        assert s.kind == "injected"


class TestCollectHw:
    def test_gpu_na(self):
        assert hwmon.collect_gpu_util() is None

    def test_collect_hw_structure(self, monkeypatch):
        monkeypatch.setattr(hwmon, "_SAMPLER", hwmon.CpuSampler(
            reader=lambda: None))
        out = hwmon.collect_hw()
        assert out["gpu"] is None
        assert out["cpu"]["percent"] is None
        assert "N/A" in out["cpu"]["note"]
        assert "generated_at" in out


class TestJobDurationSec:
    def test_memory_started_at(self):
        from backend import main as m
        m._jobs["dur_job"] = {"started_at": time.time() - 5}
        try:
            sec = m._job_duration_sec("dur_job", {})
            assert sec is not None and 4.5 <= sec <= 6.0
        finally:
            m._jobs.pop("dur_job", None)

    def test_disk_created_at_fallback(self):
        from backend import main as m
        cfg = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                           time.gmtime(time.time() - 10))}
        sec = m._job_duration_sec("nonexistent_job_xyz", cfg)
        assert sec is not None and 9.0 <= sec <= 11.0

    def test_no_timestamp_returns_none(self):
        from backend import main as m
        m._jobs.pop("dur_job2", None)
        assert m._job_duration_sec("dur_job2", {}) is None


class TestBuildJobsTrend:
    def _kpi(self, duration=None, tokens=None):
        kpi = {}
        if duration is not None:
            kpi["duration_sec"] = duration
        if tokens is not None:
            kpi["total_tokens"] = {"x": tokens[0], "y": tokens[1]}
        return {"kpi": kpi}

    def test_only_completed_with_loader(self):
        jobs = [
            {"job_id": "j1", "state": "completed", "model_a": "m1", "model_b": "m2",
             "created_at": "2026-01-01T00:00:00Z"},
            {"job_id": "j2", "state": "executing", "model_a": "m3", "model_b": "m4",
             "created_at": "2026-01-01T00:00:01Z"},
        ]
        store = {"j1": self._kpi(12.5, (100, 50))}

        def loader(jid):
            return store.get(jid)

        out = build_jobs_trend(jobs, loader)
        assert len(out) == 1
        assert out[0]["job_id"] == "j1"
        assert out[0]["duration_sec"] == 12.5
        assert out[0]["total_tokens"] == 150

    def test_missing_kpi_fields_are_none(self):
        jobs = [{"job_id": "old", "state": "completed",
                 "model_a": "a", "model_b": "b", "created_at": ""}]
        out = build_jobs_trend(jobs, lambda jid: {"report": {"kpi": {}}})
        assert out[0]["duration_sec"] is None
        assert out[0]["total_tokens"] is None

    def test_default_loader_reads_disk(self, tmp_path, monkeypatch):
        from backend import storage
        monkeypatch.setattr(storage, "BASE_DIR", tmp_path)
        job_id = storage.create_job_id()
        job_dir = tmp_path / job_id
        job_dir.mkdir()
        report = {"config": {}, "tasks": {}, "answers_a": {}, "answers_b": {},
                  "verdict": {}, "report": {"kpi": {"duration_sec": 3.0,
                                                    "total_tokens": {"x": 7, "y": 8}}}}
        (job_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False),
                                             encoding="utf-8")
        jobs = [{"job_id": job_id, "state": "completed", "model_a": "a",
                 "model_b": "b", "created_at": "2026-01-01T00:00:00Z"}]
        out = build_jobs_trend(jobs)  # 缺省加载器读磁盘
        assert out[0]["duration_sec"] == 3.0
        assert out[0]["total_tokens"] == 15
