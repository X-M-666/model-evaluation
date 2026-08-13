# -*- coding: utf-8 -*-
"""硬件利用率采集（迭代六 KPI 看板）。零新依赖。

- CPU：系统级增量利用率采样。Windows 用 ctypes.GetSystemTimes（100ns 单位），
  Linux 用 /proc/stat（jiffies，idle+iowait）；macOS 及未知平台 → None（N/A）。
  首次 sample() 仅建基线返回 None，第二次起返回两次采样间的利用率。
- GPU：v1 仅预留接口（无硬件探测），恒返回 None（N/A）。

核心换算 _util_from_deltas 为纯函数（确定性可测），平台读取可注入 fake。
"""
from __future__ import annotations

import ctypes
import platform
import time
from typing import Any, Callable


class _FileTime(ctypes.Structure):
    _fields_ = [("dwLowDateTime", ctypes.c_uint32),
                ("dwHighDateTime", ctypes.c_uint32)]


def _read_win() -> tuple[float, float] | None:
    """Windows GetSystemTimes：返回 (idle, total) 100ns tick。"""
    try:
        idle, kernel, user = _FileTime(), _FileTime(), _FileTime()
        ok = ctypes.windll.kernel32.GetSystemTimes(
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user))
        if not ok:
            return None

        def to_u64(ft: _FileTime) -> float:
            return float((ft.dwHighDateTime << 32) | ft.dwLowDateTime)

        i, k, u = to_u64(idle), to_u64(kernel), to_u64(user)
        return i, k + u
    except Exception:
        return None


def _read_linux() -> tuple[float, float] | None:
    """/proc/stat 首行 cpu 行：返回 (idle, total) jiffies。"""
    try:
        with open("/proc/stat", encoding="utf-8") as f:
            line = f.readline()
        parts = line.split()
        if not parts or parts[0] != "cpu" or len(parts) < 8:
            return None
        nums = [float(x) for x in parts[1:8]]
        idle = nums[3] + nums[4]  # idle + iowait
        return idle, sum(nums)
    except Exception:
        return None


def _read_cpu_counters() -> tuple[float, float] | None:
    """按平台读取 (idle, total) 计数器；不支持平台返回 None。"""
    system = platform.system().lower()
    if system == "windows":
        return _read_win()
    if system == "linux":
        return _read_linux()
    return None


def _detect_kind(real: bool) -> str:
    if not real:
        return "injected"
    system = platform.system().lower()
    if system == "windows":
        return "windows"
    if system == "linux":
        return "linux"
    return "unsupported"


class CpuSampler:
    """增量 CPU 利用率采样器。reader 可注入（测试用）。"""

    def __init__(self, reader: Callable[[], tuple[float, float] | None] | None = None):
        self._reader = reader or _read_cpu_counters
        self._prev: tuple[float, float] | None = None
        self.kind = _detect_kind(reader is None)

    def sample(self) -> float | None:
        """返回两次采样间的系统 CPU 利用率 0-100；首次/平台不支持 → None。"""
        now = self._reader()
        if now is None:
            return None
        if self._prev is None:
            self._prev = now
            return None
        idle_delta = max(0.0, now[0] - self._prev[0])
        total_delta = max(0.0, now[1] - self._prev[1])
        self._prev = now
        return _util_from_deltas(idle_delta, total_delta)


def _util_from_deltas(idle_delta: float, total_delta: float) -> float:
    """纯函数：busy/total 百分比（0-100，1 位小数）。"""
    if total_delta <= 0:
        return 0.0
    busy = max(0.0, total_delta - idle_delta)
    return round(min(100.0, busy / total_delta * 100.0), 1)


_SAMPLER = CpuSampler()


def collect_gpu_util() -> dict[str, Any] | None:
    """GPU 利用率（预留接口）：v1 无硬件探测，恒返回 None（N/A）。"""
    return None


def collect_hw() -> dict[str, Any]:
    """KPI 看板硬件快照：cpu 增量利用率 + gpu N/A。"""
    percent = _SAMPLER.sample()
    cpu = {
        "percent": percent,
        "unit": "%",
        "scope": "system",
        "platform": _SAMPLER.kind,
        "note": ("系统级 CPU 利用率（增量采样，首次采样返回空基线）"
                 if percent is not None else
                 "当前平台不支持系统 CPU 采样（N/A）"),
    }
    return {
        "cpu": cpu,
        "gpu": None,
        "gpu_note": "GPU 利用率预留接口：当前无硬件探测（N/A）",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
