# -*- coding: utf-8 -*-
"""native-sandbox 解释器运行时：仅引用部署方明确配置的 Python。

产品约束（issue #16，R2-007）：
- 应用不得自行下载、解压、安装或更新 Python；仓库内不存在任何运行时
  下载/解压的网络代码路径。
- 部署方通过环境变量 MODEL_DUEL_SANDBOX_PYTHON 提供绝对路径，指向
  部署方预装的 python.exe；运行器只引用该路径，绝不静默回退到网络下载、
  系统 PATH 中任意 Python 或未配置的缓存副本。
- 未配置、路径非法、目标不是文件、版本不兼容或无法运行时一律 fail
  closed，并返回可操作的错误信息。
- 运行时的安装、补丁、来源与完整性验证由部署流程负责（信任边界）。

校验要求：
- 目标须为自包含单目录运行时（stdlib 位于 python.exe 所在目录内，
  如完整安装根目录或 embeddable 发行包目录）；venv 等跨目录运行时因
  授权目录之外无法读取 stdlib，在沙箱内不可用，故被拒绝。
- 冒烟执行仅用于校验部署方指定的解释器（宿主侧、无害代码），
  绝不执行任何下载或缓存内容。
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

#: 部署方必须设置的运行时路径环境变量（绝对路径，指向 python.exe）
RUNTIME_ENV = "MODEL_DUEL_SANDBOX_PYTHON"

#: 支持的最低 Python 主版本/次版本
MIN_VERSION = (3, 10)

#: 冒烟执行超时（秒）
SMOKE_TIMEOUT = 30

_SMOKE_CODE = (
    "import sys; "
    "print(sys.version_info[0], sys.version_info[1], end=' '); "
    "print(sys.base_prefix)"
)


def configured_exe() -> Path | None:
    """返回部署方配置的解释器路径；未配置时返回 None。

    实时读取环境变量（不做模块级缓存），保证配置变更即时生效。
    """
    raw = os.environ.get(RUNTIME_ENV, "").strip()
    if not raw:
        return None
    return Path(raw)


def _smoke_test(exe: Path) -> tuple[bool, str]:
    """冒烟执行部署方指定的解释器，返回 (是否成功, 版本信息/错误)。

    仅执行部署方显式配置的路径；失败时给出可操作错误信息，不做任何
    下载或替代品回退。
    """
    try:
        proc = subprocess.run(
            [str(exe), "-c", _SMOKE_CODE],
            capture_output=True, text=True, timeout=SMOKE_TIMEOUT,
        )
    except OSError as exc:
        return False, f"无法执行 {exe}（{exc}）"
    except subprocess.TimeoutExpired:
        return False, f"{exe} 冒烟执行超时（{SMOKE_TIMEOUT}s）"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[:200]
        return False, f"{exe} 不是可运行的 Python 解释器（{detail}）"
    return True, proc.stdout.strip()


def validate_runtime(exe: Path | None) -> tuple[bool, str]:
    """校验部署方配置的运行时，返回 (是否可用, 明细)。

    校验链（任一失败即 fail closed）：
    1. 已配置（环境变量非空）；
    2. 绝对路径（拒绝相对路径，防路径劫持）；
    3. 存在且是文件（拒绝目录、符号链接悬空等）；
    4. 可执行且为 Python（冒烟执行解析版本）；
    5. 版本兼容（major==3 且 minor >= MIN_VERSION）；
    6. 自包含单目录（仅 Windows：stdlib base_prefix 位于 exe 父目录内，
       保证 AppContainer 授权该目录后沙箱内可运行）。
    """
    if exe is None:
        return False, (
            f"未配置 {RUNTIME_ENV}（部署方需提供 Python 运行时绝对路径，"
            "应用不会自动下载）"
        )
    if not exe.is_absolute():
        return False, (
            f"配置的 {RUNTIME_ENV} 必须为绝对路径：{exe}（拒绝相对路径，"
            "防止路径劫持）"
        )
    if not exe.exists():
        return False, (
            f"配置的 {RUNTIME_ENV} 不存在：{exe}（请确认部署方已安装运行时）"
        )
    if not exe.is_file():
        return False, (
            f"配置的 {RUNTIME_ENV} 不是文件：{exe}（预期为 python.exe 绝对路径）"
        )

    ok, version_line = _smoke_test(exe)
    if not ok:
        return False, version_line
    try:
        major_s, minor_s, base_prefix = version_line.split(" ", 2)
        version = (int(major_s), int(minor_s))
    except (ValueError, IndexError):
        return False, f"{exe} 版本探测输出异常：{version_line[:100]!r}"
    if version < MIN_VERSION:
        return False, (
            f"Python 版本 {version[0]}.{version[1]} 不兼容："
            f"native-sandbox 要求 >= {MIN_VERSION[0]}.{MIN_VERSION[1]}"
        )
    if os.name == "nt":
        base = Path(base_prefix)
        try:
            in_tree = base.resolve() == exe.parent.resolve() or (
                base.resolve() in exe.parent.resolve().parents)
        except OSError:
            in_tree = False
        if not in_tree:
            return False, (
                f"{exe} 的 stdlib 位于授权目录之外（base_prefix={base_prefix}），"
                "沙箱内不可运行；请使用自包含单目录运行时"
                "（完整安装根目录或 embeddable 发行包目录），venv 不支持"
            )
    return True, (
        f"Python {version[0]}.{version[1]}（{Path(os.path.abspath(exe))}）"
    )


def runtime_ready() -> tuple[bool, str]:
    """状态探针：区分“未配置 / 配置无效 / 可用”，返回 (是否可用, 明细)。"""
    return validate_runtime(configured_exe())


def resolve_runtime() -> Path:
    """返回部署方配置且校验通过的 python.exe 路径；否则抛异常（fail closed）。"""
    exe = configured_exe()
    ok, detail = validate_runtime(exe)
    if not ok:
        raise RuntimeError(
            f"native-sandbox 运行时不可用：{detail}（设置环境变量 "
            f"{RUNTIME_ENV} 为部署方预装 python.exe 的绝对路径；"
            "应用不会自动下载运行时）"
        )
    return exe


if __name__ == "__main__":
    ok, detail = runtime_ready()
    print(f"{'可用' if ok else '不可用'}: {detail}")
    raise SystemExit(0 if ok else 1)
