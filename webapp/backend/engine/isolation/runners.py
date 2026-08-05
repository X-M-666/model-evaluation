# -*- coding: utf-8 -*-
"""代码执行后端注册表：off（仅展示/语法检查）与 native-sandbox（Windows 原生隔离）。

新增后端（如 Docker）只需实现 CodeRunner 接口并注册到 MODES/get_runner。
"""
from __future__ import annotations

import abc
from typing import Any

MODES = ("off", "native-sandbox")


class CodeRunner(abc.ABC):
    """代码执行后端抽象。run() 只应在 is_available() 为 True 时调用。"""

    name: str

    @abc.abstractmethod
    def is_available(self) -> tuple[bool, str]:
        """返回 (是否可用, 说明)。"""

    @abc.abstractmethod
    def run(self, code: str, stdin_text: str = "") -> dict[str, Any]:
        """执行代码，返回 {ok, stdout, stderr, timed_out, error, returncode}。"""


class OffRunner(CodeRunner):
    """off 模式：仅展示与语法检查，不执行任何代码。"""

    name = "off"

    def is_available(self) -> tuple[bool, str]:
        return True, "仅展示与语法检查，不执行代码"

    def run(self, code: str, stdin_text: str = "") -> dict[str, Any]:  # pragma: no cover
        raise RuntimeError("off 模式不执行代码")


class NativeRunner(CodeRunner):
    """native-sandbox：Windows AppContainer + Job Object 系统级隔离。"""

    name = "native-sandbox"

    def __init__(self, limits: dict[str, Any] | None = None):
        self.limits = limits

    def is_available(self) -> tuple[bool, str]:
        from backend.engine.isolation import windows_native

        return windows_native.probe()

    def run(self, code: str, stdin_text: str = "") -> dict[str, Any]:
        from backend.engine.isolation import windows_native

        return windows_native.run_code(code, stdin_text=stdin_text, limits=self.limits)


def get_runner(mode: str) -> CodeRunner:
    if mode == "off":
        return OffRunner()
    if mode == "native-sandbox":
        return NativeRunner()
    raise ValueError(f"未知的代码验真模式: {mode}")


def build_code_verify(mode: str, raw_answer: str, task: dict[str, Any]) -> dict[str, Any]:
    """根据模式构建 code_verify 结果。

    - off：语法检查（compile 不执行），status=disabled；
    - native-sandbox：逐测试用例在隔离环境中执行，status=run；
    - 后端不可用/代码缺失：status=disabled + reason。
    """
    from backend.engine.sandbox import extract_code, syntax_check, verify_code_task

    code = extract_code(raw_answer)
    if not code:
        return {"status": "disabled", "reason": "未提取到代码"}

    if mode == "off":
        return {"status": "disabled", "syntax_ok": syntax_check(code)}

    runner = get_runner(mode)
    available, detail = runner.is_available()
    if not available:
        return {"status": "disabled", "reason": f"{mode} 不可用：{detail}"}

    return verify_code_task(code, task.get("test_cases", []), runner)
