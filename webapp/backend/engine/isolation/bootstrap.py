# -*- coding: utf-8 -*-
"""embeddable Python 运行时引导。

native-sandbox 执行器的解释器使用自包含的 Python embeddable 发行包
（置于用户目录 %LOCALAPPDATA%\\arena_python），理由：
- AppContainer 容器进程默认无权限访问系统 Python 安装目录；
- 自包含运行时只向容器 SID 授予本目录读取权限，不触碰系统路径。
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

PYTHON_VERSION = "3.12.10"
EMBED_URL = (
    f"https://www.python.org/ftp/python/{PYTHON_VERSION}/"
    f"python-{PYTHON_VERSION}-embed-amd64.zip"
)

ARENA_ROOT = Path(
    os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")
) / "arena_python"


def runtime_dir() -> Path:
    return ARENA_ROOT / PYTHON_VERSION


def python_exe() -> Path:
    return runtime_dir() / "python.exe"


def runtime_ready() -> bool:
    return python_exe().exists()


def ensure_runtime(force: bool = False, quiet: bool = True) -> Path:
    """确保 embeddable Python 可用；缺失时自动下载解压。返回 python.exe 路径。"""
    exe = python_exe()
    if exe.exists() and not force:
        return exe

    runtime_dir().mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="arena_py_dl_"))
    zpath = tmp / "python-embed.zip"
    try:
        if not quiet:
            print(f"下载 {EMBED_URL} ...")
        with urllib.request.urlopen(EMBED_URL, timeout=120) as resp, open(zpath, "wb") as f:
            shutil.copyfileobj(resp, f)
        if not quiet:
            print("解压中...")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(runtime_dir())
        if not exe.exists():
            raise RuntimeError(f"解压后未找到 python.exe: {exe}")
        _smoke_test(exe)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return exe


def _smoke_test(exe: Path) -> None:
    """运行解释器做冒烟测试（无害代码），确认自包含运行时可用。"""
    import subprocess

    proc = subprocess.run(
        [str(exe), "-c", "import sys; print(sys.version.split()[0])"],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        shutil.rmtree(runtime_dir(), ignore_errors=True)
        raise RuntimeError(f"embeddable python 冒烟测试失败: {proc.stderr.strip()}")


if __name__ == "__main__":
    print(ensure_runtime(quiet=False))
