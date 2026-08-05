# -*- coding: utf-8 -*-
"""代码题验真编排：从模型回答提取代码，按隔离后端执行测试用例。

安全说明：
- 本模块不构成安全边界。安全边界由 backend/engine/isolation/ 提供
  （Windows AppContainer + Job Object 系统级隔离）。
- 默认模式（off）仅做语法检查、不执行任何不可信代码；
  任何执行行为都必须在显式开启的隔离后端中进行。
"""
from __future__ import annotations

import re
from typing import Any

MAX_OUTPUT_CHARS = 4096

# 从模型回答中提取第一个 ```python ... ``` 代码块
def extract_code(raw_answer: str) -> str | None:
    pattern = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
    m = pattern.search(raw_answer)
    if m:
        return m.group(1).strip()
    return None


def syntax_check(code: str) -> bool:
    """仅做语法编译检查（compile 不执行代码），用于 off 模式的安全反馈。"""
    try:
        compile(code, "<code>", "exec")
        return True
    except SyntaxError:
        return False


def verify_code_task(code: str, test_cases: list[dict[str, Any]], runner) -> dict[str, Any]:
    """对代码题运行全部 test_cases，逐组验证输出与期望。

    策略：代码定义函数后，附加一行调用代码把函数执行结果打印出来；
    若代码是程序（从 stdin 读），则直接以 test_case.input 作为 stdin 运行。
    runner 为 isolation.runners.CodeRunner 实例（off 模式不会走到这里）。
    """
    if not test_cases:
        return {"status": "run", "passed": 0, "total": 0, "results": []}
    if not code:
        return {"status": "run", "passed": 0, "total": len(test_cases),
                "results": [{"index": i, "passed": False, "note": "未提取到代码"}
                            for i in range(len(test_cases))]}

    results: list[dict[str, Any]] = []
    passed = 0
    for i, tc in enumerate(test_cases):
        case_input = tc.get("input", "")
        expected = (tc.get("expected", "") or "").strip()
        # 若是函数/表达式形式（input 形如 foo(args)），用 print 包裹执行
        if re.match(r"^\s*[A-Za-z_][A-Za-z0-9_]*\(", case_input):
            runner_code = code + "\nprint(" + case_input + ")"
        else:
            runner_code = code
        res = runner.run(runner_code, stdin_text=case_input)
        out = res["stdout"].strip()
        res["index"] = i
        res["passed"] = res["ok"] and out == expected
        if res["passed"]:
            passed += 1
        res["expected"] = expected
        res["got"] = out
        results.append(res)

    return {"status": "run", "passed": passed, "total": len(test_cases), "results": results}
