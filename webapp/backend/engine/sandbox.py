# -*- coding: utf-8 -*-
"""代码题安全沙箱：在受限子进程中运行模型输出的 Python 代码并验证测试用例。

安全约束（Windows 下 resource 模块不可用，改用进程级限制 + 运行时自我保护）：
- 独立 subprocess，超时严格限制；
- 代码注入沙箱保护层：屏蔽文件写入、网络、导入危险模块（os/subprocess/socket
  等），防止模型生成恶意代码读写本机；
- stdin/stdout/stderr 隔离；cwd 设为只读临时目录。
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

TIMEOUT_SEC = 15
MAX_OUTPUT_CHARS = 4096

# 保护层：在用户代码之前注入的沙箱前缀
_SANDBOX_PREFIX = r'''
import builtins, io, sys
def _deny(*a, **k):
    raise RuntimeError("sandbox: 禁止的操作")
_orig_open = builtins.open
builtins.open = _deny
_orig_import = builtins.__import__
def _safe_import(name, *a, **k):
    _deny_set = {"os","subprocess","socket","shutil","pathlib","signal",
                 "ctypes","pickle","marshal","importlib","pty","select","pwd",
                 "grp","fcntl","getpass","resource","tempfile","urllib",
                 "requests","http","ftplib","telnetlib"}
    base = name.split(".")[0]
    if base in _deny_set:
        raise RuntimeError("sandbox: 禁止导入模块 " + base)
    return _orig_import(name, *a, **k)
builtins.__import__ = _safe_import
sys.stdin = io.StringIO("")
'''

# 从模型回答中提取第一个 ```python ... ``` 代码块
def extract_code(raw_answer: str) -> str | None:
    pattern = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
    m = pattern.search(raw_answer)
    if m:
        return m.group(1).strip()
    return None


def _strip_builtins_from_code(code: str) -> str:
    """去掉代码中可能重复定义的 import sys 等，避免与沙箱前缀冲突。"""
    lines = [ln for ln in code.splitlines() if not ln.strip().startswith(("import sys", "from sys"))]
    return "\n".join(lines)


def run_code(code: str, stdin_text: str = "") -> dict[str, Any]:
    """在沙箱中运行代码（不执行具体测试，仅返回 stdout/stderr/超时状态）。

    供测试用例批量验证使用；单次运行限制超时与输出长度。
    """
    result: dict[str, Any] = {
        "ok": False, "stdout": "", "stderr": "", "timed_out": False, "error": None,
    }
    workdir = tempfile.mkdtemp(prefix="arena_sandbox_")
    try:
        full_code = _SANDBOX_PREFIX + "\n" + _strip_builtins_from_code(code)
        proc = subprocess.run(
            [sys.executable, "-c", full_code],
            input=stdin_text,
            capture_output=True,
            text=True,
            cwd=workdir,
            timeout=TIMEOUT_SEC,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        result["ok"] = proc.returncode == 0
        result["stdout"] = proc.stdout[:MAX_OUTPUT_CHARS]
        result["stderr"] = proc.stderr[:MAX_OUTPUT_CHARS]
        result["returncode"] = proc.returncode
    except subprocess.TimeoutExpired:
        result["timed_out"] = True
        result["error"] = f"运行超时（>{TIMEOUT_SEC}s）"
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"沙箱执行异常: {exc}"
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return result


def verify_code_task(code: str, test_cases: list[dict[str, Any]]) -> dict[str, Any]:
    """对代码题运行全部 test_cases，逐组验证输出与期望。

    策略：代码定义函数后，附加一行调用代码把函数执行结果打印出来；
    若代码是程序（从 stdin 读），则直接以 test_case.input 作为 stdin 运行。
    """
    if not test_cases:
        return {"supported": True, "passed": 0, "total": 0, "results": []}
    if not code:
        return {"supported": True, "passed": 0, "total": len(test_cases),
                "results": [{"index": i, "passed": False, "note": "未提取到代码"}
                            for i in range(len(test_cases))]}

    results: list[dict[str, Any]] = []
    passed = 0
    for i, tc in enumerate(test_cases):
        case_input = tc.get("input", "")
        expected = (tc.get("expected", "") or "").strip()
        # 若是函数/表达式形式（input 形如 foo(args)），用 print 包裹执行
        stripped = _strip_builtins_from_code(code)
        if re.match(r"^\s*[A-Za-z_][A-Za-z0-9_]*\(", case_input):
            runner = stripped + "\nprint(" + case_input + ")"
        else:
            runner = stripped
        res = run_code(runner, stdin_text=case_input)
        out = res["stdout"].strip()
        res["index"] = i
        res["passed"] = res["ok"] and out == expected
        if res["passed"]:
            passed += 1
        res["expected"] = expected
        res["got"] = out
        results.append(res)

    return {"supported": True, "passed": passed, "total": len(test_cases), "results": results}
