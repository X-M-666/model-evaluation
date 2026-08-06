# -*- coding: utf-8 -*-
"""运行时配置校验测试（issue #16：移除 Python 自动下载）。

保护不变量：
- 未配置 / 相对路径 / 不存在 / 目录 / 不可执行 / 版本不兼容 / stdlib 越界
  一律 fail closed，返回可操作错误信息；
- 任何失败路径均不触发网络或替代品回退（代码级断言 + conftest 网络封锁）；
- 应用代码中不存在运行时下载（urllib / zipfile）实现。
"""
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path

import pytest

from backend.engine.isolation import bootstrap


class _FakeProc:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _fake_version(exe: Path) -> str:
    """构造与真实解释器同格式的冒烟输出（major minor base_prefix）。"""
    return f"{sys.version_info[0]} {sys.version_info[1]} {exe.parent}"


def _fake_run(monkeypatch, proc: _FakeProc) -> None:
    monkeypatch.setattr(bootstrap.subprocess, "run", lambda *a, **k: proc)


def _mk_exe(tmp_path: Path, name: str = "python.exe") -> Path:
    exe = tmp_path / name
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_bytes(b"MZ")  # 只保证存在且是文件，校验链不依赖真实可执行
    return exe


# ---- 未配置 ----

def test_unconfigured_fails_closed(monkeypatch):
    monkeypatch.delenv(bootstrap.RUNTIME_ENV, raising=False)
    ok, detail = bootstrap.runtime_ready()
    assert ok is False
    assert bootstrap.RUNTIME_ENV in detail, f"错误信息应指引变量名: {detail}"
    with pytest.raises(RuntimeError) as ei:
        bootstrap.resolve_runtime()
    assert bootstrap.RUNTIME_ENV in str(ei.value)
    assert "不会自动下载" in str(ei.value)


def test_configured_exe_none_when_unset(monkeypatch):
    monkeypatch.delenv(bootstrap.RUNTIME_ENV, raising=False)
    assert bootstrap.configured_exe() is None


def test_configured_exe_live_read(monkeypatch, tmp_path):
    """env 实时读取：配置变更立即生效，不做模块级缓存。"""
    a = _mk_exe(tmp_path / "a")
    b = _mk_exe(tmp_path / "b")
    monkeypatch.setenv(bootstrap.RUNTIME_ENV, str(a))
    assert bootstrap.configured_exe() == a
    monkeypatch.setenv(bootstrap.RUNTIME_ENV, str(b))
    assert bootstrap.configured_exe() == b


# ---- 路径校验 ----

def test_relative_path_rejected(monkeypatch):
    monkeypatch.setenv(bootstrap.RUNTIME_ENV, "python.exe")
    ok, detail = bootstrap.runtime_ready()
    assert ok is False
    assert "绝对路径" in detail


def test_nonexistent_rejected(monkeypatch, tmp_path):
    target = tmp_path / "missing" / "python.exe"
    monkeypatch.setenv(bootstrap.RUNTIME_ENV, str(target))
    ok, detail = bootstrap.runtime_ready()
    assert ok is False
    assert "不存在" in detail


def test_directory_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv(bootstrap.RUNTIME_ENV, str(tmp_path))
    ok, detail = bootstrap.runtime_ready()
    assert ok is False
    assert "不是文件" in detail


def test_blank_value_treated_as_unconfigured(monkeypatch):
    monkeypatch.setenv(bootstrap.RUNTIME_ENV, "   ")
    assert bootstrap.configured_exe() is None


# ---- 冒烟执行校验 ----

def test_not_executable_rejected(monkeypatch, tmp_path):
    exe = _mk_exe(tmp_path)
    monkeypatch.setenv(bootstrap.RUNTIME_ENV, str(exe))
    _fake_run(monkeypatch, _FakeProc(returncode=1, stderr="not a python\n"))
    ok, detail = bootstrap.runtime_ready()
    assert ok is False
    assert "不是可运行的 Python 解释器" in detail


def test_version_incompatible_rejected(monkeypatch, tmp_path):
    exe = _mk_exe(tmp_path)
    monkeypatch.setenv(bootstrap.RUNTIME_ENV, str(exe))
    _fake_run(monkeypatch, _FakeProc(stdout="2 7 C:\\old"))
    ok, detail = bootstrap.runtime_ready()
    assert ok is False
    assert "不兼容" in detail
    assert f">= {bootstrap.MIN_VERSION[0]}.{bootstrap.MIN_VERSION[1]}" in detail


def test_garbage_version_output_rejected(monkeypatch, tmp_path):
    exe = _mk_exe(tmp_path)
    monkeypatch.setenv(bootstrap.RUNTIME_ENV, str(exe))
    _fake_run(monkeypatch, _FakeProc(stdout="not-a-version"))
    ok, detail = bootstrap.runtime_ready()
    assert ok is False
    assert "版本探测输出异常" in detail


@pytest.mark.skipif(os.name != "nt", reason="stdlib 越界检查仅 Windows 沙箱生效")
def test_stdlib_outside_tree_rejected(monkeypatch, tmp_path):
    exe = _mk_exe(tmp_path)
    monkeypatch.setenv(bootstrap.RUNTIME_ENV, str(exe))
    _fake_run(monkeypatch, _FakeProc(stdout="3 12 C:\\outside\\prefix"))
    ok, detail = bootstrap.runtime_ready()
    assert ok is False
    assert "stdlib 位于授权目录之外" in detail


# ---- 合法配置 ----

def test_valid_absolute_path_accepted(monkeypatch, tmp_path):
    exe = _mk_exe(tmp_path)
    monkeypatch.setenv(bootstrap.RUNTIME_ENV, str(exe))
    _fake_run(monkeypatch, _FakeProc(stdout=_fake_version(exe)))
    ok, detail = bootstrap.runtime_ready()
    assert ok is True
    assert "Python" in detail
    assert str(exe) in detail
    assert bootstrap.resolve_runtime() == exe


def test_real_interpreter_validated_on_posix():
    """POSIX 上真实解释器应通过路径/版本校验（stdlib 检查不生效）。"""
    ok, detail = bootstrap.validate_runtime(Path(sys.executable))
    assert ok is True, detail


# ---- 无下载不变量 ----

def test_no_download_code_in_bootstrap():
    """bootstrap 模块不得包含任何下载/解压实现（issue #16 验收）。"""
    src = inspect.getsource(bootstrap)
    for banned in ("urllib", "urlopen", "zipfile", "requests", "http"):
        assert banned not in src, f"bootstrap 源码不得包含 {banned!r}"
    assert not hasattr(bootstrap, "urllib")
    assert not hasattr(bootstrap, "zipfile")


def test_validation_failure_does_not_contact_network(monkeypatch, tmp_path):
    """失败路径无网络调用：conftest 网络封锁 fixture 会在任何真实外连时抛错，
    此处仅确认失败路径返回的是校验错误而非网络异常。"""
    monkeypatch.setenv(bootstrap.RUNTIME_ENV, str(tmp_path / "nope.exe"))
    ok, detail = bootstrap.runtime_ready()
    assert ok is False
    assert "网络" not in detail
