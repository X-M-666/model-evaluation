# -*- coding: utf-8 -*-
"""代码隔离安全回归测试：off 模式不执行；native-sandbox 逃逸/资源限制。

对照 SECURITY.md 验收项：
- 宿主敏感文件读取被拒
- 工作目录外写入被拒、目录内写入允许
- 网络连接被拒
- 内存炸弹 / 死循环 / 进程炸弹在配额内终止
- 宿主环境变量（含凭据）不继承
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from backend.engine.isolation import windows_native
from backend.engine.isolation.runners import build_code_verify, get_runner
from backend.engine.sandbox import extract_code, syntax_check

HOSTS_FILE = windows_native.HOSTS_FILE
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_TMP = os.environ.get("TEMP") or os.environ.get("TMP") or "/tmp"
OUTSIDE_FILE = Path(_TMP) / "arena_test_pwn.txt"
USER_SECRET = Path(os.environ.get("LOCALAPPDATA") or _TMP) / "arena_test_secret.txt"


# ---- off 模式：默认不执行 ----

def test_extract_code():
    assert extract_code("前置\n```python\nprint(1)\n```\n后置") == "print(1)"
    assert extract_code("无代码块") is None
    assert extract_code("```\nx=1\n```") == "x=1"


def test_syntax_check():
    assert syntax_check("def f():\n    return 1\n") is True
    assert syntax_check("def f(:\n") is False


def test_off_mode_no_execution():
    task = {"test_cases": [{"input": "f()", "expected": "1"}]}
    cv = build_code_verify("off", "```python\nprint('hello')\n```", task)
    assert cv["status"] == "disabled"
    assert cv.get("syntax_ok") is True
    assert "passed" not in cv


def test_off_mode_syntax_error_reported():
    task = {"test_cases": [{"input": "f()", "expected": "1"}]}
    cv = build_code_verify("off", "```python\ndef f(:\n```", task)
    assert cv["status"] == "disabled"
    assert cv.get("syntax_ok") is False


def test_off_mode_no_code():
    task = {"test_cases": [{"input": "f()", "expected": "1"}]}
    cv = build_code_verify("off", "没有代码", task)
    assert cv["status"] == "disabled"
    assert cv.get("reason") == "未提取到代码"


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        build_code_verify("bogus", "```python\nprint(1)\n```", {"test_cases": []})


# ---- native-sandbox：逃逸与资源限制 ----

@pytest.fixture(scope="module")
def native_runner():
    runner = get_runner("native-sandbox")
    available, detail = runner.is_available()
    if not available:
        pytest.skip(
            f"native-sandbox 不可用：{detail}"
            "（先配置 MODEL_DUEL_SANDBOX_PYTHON 并运行 python -m scripts.sandbox_selfcheck）"
        )
    return runner


def test_native_normal_execution(native_runner):
    res = native_runner.run("def add(a, b): return a + b\nprint(add(1, 2))")
    assert res["ok"] is True
    assert res["stdout"].strip() == "3"


def test_native_stdin_program(native_runner):
    res = native_runner.run("import sys\nprint(sys.stdin.read().strip().upper())", stdin_text="hello")
    assert res["ok"] is True
    assert res["stdout"].strip() == "HELLO"


def test_native_stdin_input_single_line(native_runner):
    """Issue #4 验收1：input() 能读取单行输入（含 EOF 无换行场景）。"""
    res = native_runner.run("print(input().upper())", stdin_text="hello")
    assert res["ok"] is True
    assert res["stdout"].strip() == "HELLO"


def test_native_stdin_read_multiline_utf8(native_runner):
    """Issue #4 验收2：sys.stdin.read() 能读取多行及 UTF-8 输入。"""
    text = "第一行\n第二行 中文\n"
    res = native_runner.run("import sys\nprint(sys.stdin.read(), end='')", stdin_text=text)
    assert res["ok"] is True
    assert res["stdout"] == text


def test_native_stdin_empty_is_eof(native_runner):
    """Issue #4 验收3：空输入时程序收到正常 EOF，而不是伪造数据或异常。"""
    res = native_runner.run("import sys\nprint(repr(sys.stdin.read()))", stdin_text="")
    assert res["ok"] is True
    assert res["stdout"].strip() == "''"


def test_native_read_hosts_allowed_residual(native_runner):
    # AppContainer（同 UWP）允许读取世界可读的系统公共文件（如 hosts），
    # 属文档化残留（SECURITY.md 边界说明）；此处仅断言不崩溃且输出不泄露用户数据
    res = native_runner.run(f'print(open(r"{HOSTS_FILE}").read())')
    assert res["ok"] is True
    assert "localhost" in res["stdout"]


def test_native_read_user_profile_blocked(native_runner):
    USER_SECRET.write_text("top-secret", encoding="utf-8")
    try:
        res = native_runner.run(f'print(open(r"{USER_SECRET}").read())')
    finally:
        USER_SECRET.unlink(missing_ok=True)
    assert res["ok"] is False, f"不应能读取宿主用户文件：{res}"
    assert "top-secret" not in res["stdout"]


def test_native_read_project_source_blocked(native_runner):
    target = REPO_ROOT / "README.md"
    res = native_runner.run(f'print(open(r"{target}").read())')
    assert res["ok"] is False, f"不应能读取项目源码：{res}"
    assert "模型对决评测平台" not in res["stdout"]


def test_native_write_outside_blocked(native_runner):
    if OUTSIDE_FILE.exists():
        OUTSIDE_FILE.unlink()
    res = native_runner.run(f'open(r"{OUTSIDE_FILE}", "w").write("pwn"); print("written")')
    assert res["ok"] is False, f"不应能写工作目录外：{res}"
    assert not OUTSIDE_FILE.exists(), "目录外文件不应被创建"
    if OUTSIDE_FILE.exists():
        OUTSIDE_FILE.unlink()


def test_native_write_inside_allowed(native_runner):
    res = native_runner.run('open("allowed.txt", "w").write("ok"); print("written")')
    assert res["ok"] is True
    assert res["stdout"].strip() == "written"


def test_native_network_blocked(native_runner):
    res = native_runner.run('import socket; socket.create_connection(("1.1.1.1", 80))')
    assert res["ok"] is False, f"网络连接不应成功：{res}"
    assert res["timed_out"] is False


def test_native_memory_bomb_killed(native_runner):
    res = native_runner.run("x = [0] * 10**9")
    assert res["ok"] is False
    assert res["timed_out"] is False


def test_native_infinite_loop_killed(native_runner):
    res = native_runner.run("while True: pass")
    assert res["ok"] is False
    # 纯 CPU 循环会被 CPU 时间配额（5s）提前终止（rc!=0），
    # 睡眠循环才会触达墙钟超时；两者任一都符合配额终止预期
    assert res["timed_out"] is True or res["returncode"] != 0


def test_native_process_bomb_killed(native_runner):
    code = (
        "import subprocess, sys, time\n"
        "while True:\n"
        "    subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
    )
    res = native_runner.run(code)
    assert res["ok"] is False
    assert res["timed_out"] is False, "进程炸弹应被进程数配额拒绝，而非等墙钟超时"


# 慢速持续写入脚本：避免单次大写入在 watcher 轮询前完成导致测试竞态。
# 监视对象为整个工作目录总占用（0.05s 轮询），慢速脚本保证超限发生在
# 运行期内、能被 watcher 捕获（生产环境恶意代码持续写入同样会被捕获）。
_SLOW_STDOUT = (
    "import sys, time\n"
    "for _ in range(200):\n"
    "    sys.stdout.write('x' * 6000)\n"
    "    sys.stdout.flush()\n"
    "    time.sleep(0.02)\n"
    "print('done')\n"
)
_SLOW_STDERR = (
    "import sys, time\n"
    "for _ in range(200):\n"
    "    sys.stderr.write('y' * 6000)\n"
    "    sys.stderr.flush()\n"
    "    time.sleep(0.02)\n"
    "print('done')\n"
)
_SLOW_BLOB = (
    "import time\n"
    "with open('blob.bin', 'wb') as f:\n"
    "    for _ in range(200):\n"
    "        f.write(b'x' * 6000)\n"
    "        f.flush()\n"
    "        time.sleep(0.02)\n"
    "print('done')\n"
)
# 单次高速大写入后立即退出：可能在 0.05s 轮询采样前完成（进程随即退出），
# 依赖进程退出后的目录总占用复核兜底（issue #18 缺口2 回归）
_BIG_SINGLE_WRITE = (
    "with open('blob.bin', 'wb') as f:\n"
    "    f.write(b'x' * (4 * 1024 * 1024))\n"
    "    f.flush()\n"
)
# stdout/stderr/普通文件各自 < 1MB，但累计超限（每轮 18KB × 70 轮 = 1.26MB）
_MULTI_FILE_SLOW = (
    "import sys, time\n"
    "with open('blob.bin', 'wb') as bf:\n"
    "    for _ in range(70):\n"
    "        sys.stdout.write('x' * 6000)\n"
    "        sys.stdout.flush()\n"
    "        sys.stderr.write('y' * 6000)\n"
    "        sys.stderr.flush()\n"
    "        bf.write(b'z' * 6000)\n"
    "        bf.flush()\n"
    "        time.sleep(0.02)\n"
    "print('done')\n"
)


def test_native_output_overflow_stdout_killed(native_runner):
    """输出配额（补强点3）：超限 stdout 在配额内被终止，不撑爆宿主磁盘。"""
    res = native_runner.run(_SLOW_STDOUT)
    assert res["ok"] is False
    assert res.get("output_overflow") is True, res
    assert res["timed_out"] is False
    assert "配额" in (res.get("error") or "")


def test_native_output_overflow_stderr_killed(native_runner):
    """输出配额（补强点3）：超限 stderr 同样被终止。"""
    res = native_runner.run(_SLOW_STDERR)
    assert res["ok"] is False
    assert res.get("output_overflow") is True, res


def test_native_output_quota_configurable(native_runner):
    """输出配额可配置：小配额下小输出也会被终止（验证配额参数生效）。"""
    from backend.engine.isolation import windows_native

    res = windows_native.run_code(_SLOW_STDOUT, limits={"max_output_bytes": 8 * 1024})
    assert res["ok"] is False
    assert res.get("output_overflow") is True, res


def test_native_regular_file_overflow_killed(native_runner):
    """Issue #18 验收1：持续写普通文件（非 stdout/stderr）同样在配额内被终止。"""
    res = native_runner.run(_SLOW_BLOB)
    assert res["ok"] is False
    assert res.get("output_overflow") is True, res
    assert res["timed_out"] is False
    assert "配额" in (res.get("error") or "")


def test_native_multiple_files_cumulative(native_runner):
    """Issue #18 验收2：stdout/stderr/普通文件各自未超限但累计超限时被终止（同一总量约束）。"""
    res = native_runner.run(_MULTI_FILE_SLOW)
    assert res["ok"] is False
    assert res.get("output_overflow") is True, res
    assert res["timed_out"] is False


def test_native_single_big_write_detected(native_runner):
    """Issue #18 缺口2：单次高速大写入在轮询采样前完成，退出后复核兜底判定超限。"""
    res = native_runner.run(_BIG_SINGLE_WRITE)
    assert res["ok"] is False
    assert res.get("output_overflow") is True, res


def test_native_peak_dir_bytes_bounded(native_runner):
    """Issue #18 验收3：终止时目录总字节数（peak_dir_bytes）的最大超额符合文档承诺。

    慢速写入每轮 6KB（20ms），0.05s 轮询窗口内最多再写约 15KB；
    64KB 上界远大于理论峰值（8KB 配额 + 窗口写入 + 基准文件），
    用于避免机器抖动导致的 flaky，同时能捕获实现回归（如只查单文件）。
    """
    from backend.engine.isolation import windows_native

    res = windows_native.run_code(_SLOW_BLOB, limits={"max_output_bytes": 8 * 1024})
    assert res.get("output_overflow") is True, res
    peak = res.get("peak_dir_bytes", 0)
    assert isinstance(peak, int) and peak > 0, res
    assert peak <= 8 * 1024 + 64 * 1024, res


def test_native_large_but_within_quota_output_ok(native_runner):
    """配额内的大输出（100KB < 1MB）不被误杀；展示截断 max_output_chars 生效。"""
    from backend.engine.isolation import windows_native

    res = native_runner.run("print('x' * 100_000)")
    assert res["ok"] is True
    assert res.get("output_overflow") is False
    assert len(res["stdout"]) <= windows_native.DEFAULT_LIMITS["max_output_chars"]
    assert res["stdout"].startswith("xxxxx")


def test_native_env_not_inherited(native_runner):
    os.environ["ARENA_TEST_SECRET"] = "secret-123"
    try:
        res = native_runner.run("import os; print(os.environ.get('ARENA_TEST_SECRET'))")
    finally:
        os.environ.pop("ARENA_TEST_SECRET", None)
    assert "secret-123" not in res["stdout"]


def test_profile_pool_distinct_sids(native_runner):
    """并发运行必须从池中分配到互异的 profile/SID。"""
    p1 = windows_native._PROFILE_POOL.acquire()
    p2 = windows_native._PROFILE_POOL.acquire()
    try:
        kernel32 = windows_native._api()["kernel32"]
        h1, s1 = windows_native._appcontainer_sid(p1)
        h2, s2 = windows_native._appcontainer_sid(p2)
        kernel32.LocalFree(h1)
        kernel32.LocalFree(h2)
        assert p1 != p2
        assert s1 != s2
    finally:
        windows_native._PROFILE_POOL.release(p1)
        windows_native._PROFILE_POOL.release(p2)


def test_native_concurrent_isolation(native_runner):
    """并发任务使用互异 SID：任务 B 的沙箱代码不能读取任务 A 的一次性工作目录。

    修复前（共享 SID）任务 B 可读到任务 A 的 marker.txt，此测试会失败。
    """
    from backend.engine.isolation import bootstrap

    exe = bootstrap.resolve_runtime()
    wd_a = Path(tempfile.mkdtemp(prefix="arena_xrun_a_"))
    wd_b = Path(tempfile.mkdtemp(prefix="arena_xrun_b_"))
    results: dict[str, dict] = {}

    def worker(key: str, wd: Path, code: str) -> None:
        results[key] = windows_native._spawn(exe, wd, code, "", windows_native.DEFAULT_LIMITS)

    try:
        ta = threading.Thread(target=worker, args=(
            "a", wd_a, 'open("marker.txt", "w").write("A")\nimport time\ntime.sleep(1.2)'))
        tb = threading.Thread(target=worker, args=(
            "b", wd_b,
            f'import time\ntime.sleep(0.6)\n'
            f'try:\n    print(open(r"{wd_a.as_posix()}\\marker.txt").read())\n'
            f'except OSError:\n    print("DENIED")'))
        ta.start()
        time.sleep(0.5)
        tb.start()
        ta.join()
        tb.join()
    finally:
        shutil.rmtree(wd_a, ignore_errors=True)
        shutil.rmtree(wd_b, ignore_errors=True)

    assert results["a"]["ok"] is True, results["a"]
    assert results["b"]["ok"] is True, f"任务 B 应能正常运行：{results['b']}"
    assert "A" not in results["b"]["stdout"], "任务 B 不应能读取任务 A 的工作目录"
    assert "DENIED" in results["b"]["stdout"], f"任务 B 应被拒绝访问：{results['b']}"


def test_build_code_verify_safe_error_status(monkeypatch):
    """R3：沙箱异常时兜底返回 status=error（而非 disabled），
    提示可能已部分执行，避免误导为"未执行"。"""
    from backend.engine import executor
    from backend.engine.isolation import runners

    def boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("boom")

    monkeypatch.setattr(runners, "build_code_verify", boom)
    cv = asyncio.run(executor._build_code_verify_safe("native-sandbox", "x", {"test_cases": []}))
    assert cv["status"] == "error"
    assert "boom" in cv["reason"]
    assert "部分执行" in cv["reason"]


def test_fmt_code_verify_error_label():
    """R3：评审提示中 error 状态显示"执行异常（可能已部分执行）"。"""
    from backend.engine.judge import _fmt_code_verify

    assert "执行异常" in _fmt_code_verify({"status": "error", "reason": "boom"})
    assert _fmt_code_verify({"status": "disabled"}) == "未执行（已禁用）"
    assert _fmt_code_verify({"status": "run", "passed": 2, "total": 3}) == "2/3"


def test_native_verify_code_task_integration(native_runner):
    code = (
        "def merge(intervals):\n"
        "    if not intervals: return []\n"
        "    intervals.sort()\n"
        "    out = [intervals[0]]\n"
        "    for s, e in intervals[1:]:\n"
        "        if s <= out[-1][1]: out[-1][1] = max(out[-1][1], e)\n"
        "        else: out.append([s, e])\n"
        "    return out\n"
    )
    from backend.engine.sandbox import verify_code_task

    test_cases = [
        {"input": "merge([[1,3],[2,6],[8,10],[15,18]])", "expected": "[[1, 6], [8, 10], [15, 18]]"},
        {"input": "merge([[1,4],[4,5]])", "expected": "[[1, 5]]"},
        {"input": "merge([])", "expected": "[]"},
    ]
    result = verify_code_task(code, test_cases, native_runner)
    assert result["status"] == "run"
    assert result["passed"] == 3
    assert result["total"] == 3


def test_build_code_verify_native_run(native_runner):
    raw = "```python\ndef f():\n    return 42\n```"
    task = {"test_cases": [{"input": "f()", "expected": "42"}]}
    cv = build_code_verify("native-sandbox", raw, task)
    assert cv["status"] == "run"
    assert cv["passed"] == 1


# ---- Issue #4 验收 4/5：内置 T7/T7B 标准输入型代码题的正确实现必须全通过 ----

T7_REFERENCE = """\
import sys
from collections import Counter


def main():
    text = sys.stdin.read()
    counts = Counter(ch for ch in text if "\\u4e00" <= ch <= "\\u9fff")
    if not counts:
        print("EMPTY")
        return
    for ch, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]:
        print(f"{ch} {n}")


main()
"""

T7B_REFERENCE = """\
import sys
from collections import Counter


def main():
    counts = Counter(word for line in sys.stdin for word in line.split())
    for word, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]:
        print(f"{word} {n}")


main()
"""


def _builtin_task_cases(tid: str) -> list[dict]:
    from backend.engine.tasks import QUESTION_POOL

    return next(
        t["test_cases"] for t in QUESTION_POOL["效率与稳定性"] if t["id"] == tid
    )


def test_verify_t7_correct_implementation_passes(native_runner):
    """Issue #4 验收4：T7 正确实现通过全部示例用例（含 EMPTY 与并列码点序边界）。"""
    assert syntax_check(T7_REFERENCE) is True, "参考答案自身必须语法合法"
    cv = build_code_verify(
        "native-sandbox", f"```python\n{T7_REFERENCE}\n```",
        {"test_cases": _builtin_task_cases("T7")},
    )
    assert cv["status"] == "run"
    assert cv["passed"] == cv["total"] == 5
    for r in cv["results"]:
        assert r["passed"] is True, f"T7 用例 {r['index']} 失败: {r}"


def test_verify_t7b_correct_implementation_passes(native_runner):
    """Issue #4 验收5：T7B 正确流式实现通过全部示例用例（含空输入无输出）。"""
    assert syntax_check(T7B_REFERENCE) is True, "参考答案自身必须语法合法"
    cv = build_code_verify(
        "native-sandbox", f"```python\n{T7B_REFERENCE}\n```",
        {"test_cases": _builtin_task_cases("T7B")},
    )
    assert cv["status"] == "run"
    assert cv["passed"] == cv["total"] == 5
    for r in cv["results"]:
        assert r["passed"] is True, f"T7B 用例 {r['index']} 失败: {r}"


def test_verify_function_case_no_regression(native_runner):
    """Issue #4 验收6：函数调用型测试方式不因 stdin 处理而回归。"""
    from backend.engine.sandbox import verify_code_task

    code = "def add(a, b):\n    return a + b\n"
    result = verify_code_task(code, [{"input": "add(2, 3)", "expected": "5"}], native_runner)
    assert result["passed"] == 1
    assert result["results"][0]["stdout"].strip() == "5"
