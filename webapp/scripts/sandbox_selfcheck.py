# -*- coding: utf-8 -*-
"""代码隔离环境自检：检测模式可用性、校验配置运行时、实测逃逸与资源限制。

用法（在 webapp/ 目录下，须先配置 MODEL_DUEL_SANDBOX_PYTHON）：
    python -m scripts.sandbox_selfcheck
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.engine.isolation import bootstrap  # noqa: E402
from backend.engine.isolation import windows_native  # noqa: E402
from backend.engine.isolation.runners import MODES, get_runner  # noqa: E402

HOSTS = windows_native.HOSTS_FILE
REPO_README = Path(__file__).resolve().parent.parent.parent / "README.md"
OUTSIDE_FILE = Path(os.environ.get("TEMP", r"C:\Windows\Temp")) / "arena_selfcheck_pwn.txt"
USER_SECRET = Path(os.environ.get("LOCALAPPDATA", os.environ.get("TEMP"))) / "arena_selfcheck_secret.txt"


def _check(label: str, fn) -> tuple[bool, str]:
    t0 = time.time()
    try:
        ok, detail = fn()
        return ok, f"{detail}（{time.time() - t0:.1f}s）"
    except Exception as exc:  # noqa: BLE001
        return False, f"异常: {exc}"


def _mk_case(label: str, code: str, bad_out: bool = True) -> tuple[str, str, object]:
    """返回 (label, code, 判定函数)。"""

    def run_check() -> tuple[bool, str]:
        res = windows_native.run_code(code)
        if res.get("timed_out"):
            return True, "在配额内被终止（timed_out）"
        if res["ok"]:
            return (False, f"意外成功: stdout={res['stdout'][:80]!r}") if bad_out \
                else (True, f"stdout={res['stdout'][:80]!r}")
        return (True, f"被拒绝/终止: rc={res.get('returncode')} err={res.get('error')}") if bad_out \
            else (False, f"执行失败: rc={res.get('returncode')} err={res.get('error')}")

    return label, code, run_check


def main() -> int:
    print("== 代码隔离环境自检 ==")

    # 1) 模式可用性
    print("\n[1] 模式可用性")
    for m in MODES:
        runner = get_runner(m)
        available, detail = runner.is_available()
        print(f"    {m}: {'可用' if available else '不可用'} — {detail}")

    # 2) 运行时配置校验
    print("\n[2] 运行时配置校验")
    ready, detail = bootstrap.runtime_ready()
    print(f"    {bootstrap.RUNTIME_ENV}: {detail}")
    if not ready:
        print(f"    运行时不可用：应用不会自动下载 Python。")
        print(f"    请部署方预装自包含 Python（完整安装或 embeddable 发行包），")
        print(f"    并将 {bootstrap.RUNTIME_ENV} 设为 python.exe 的绝对路径后重试。")
        return 1
    exe = bootstrap.resolve_runtime()

    # 3) 逃逸/资源实测（native-sandbox）
    runner = get_runner("native-sandbox")
    available, detail = runner.is_available()
    if not available:
        print(f"\n[3] native-sandbox 不可用（{detail}），跳过实测")
        return 1

    print("\n[3] 逃逸与资源限制实测")
    cases = [
        _mk_case("正常执行", "print(1+1)", bad_out=False),
        _mk_case("读项目源码", f'print(open(r"{REPO_README}").read())'),
        _mk_case("读宿主用户文件", f'print(open(r"{USER_SECRET}").read())'),
        _mk_case("写工作目录外", f'open(r"{OUTSIDE_FILE}", "w").write("pwn"); print("written")'),
        _mk_case("网络连接", 'import socket; socket.create_connection(("1.1.1.1", 80))'),
        _mk_case("内存炸弹", "x = [0] * 10**9"),
        _mk_case("死循环", "while True: pass"),
        _mk_case("进程炸弹", "import subprocess, sys, time\n"
                            "while True: subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])"),
    ]
    if OUTSIDE_FILE.exists():
        OUTSIDE_FILE.unlink()
    USER_SECRET.write_text("top-secret", encoding="utf-8")

    all_ok = True
    for label, code, check in cases:
        ok, detail = _check(label, check)
        all_ok = all_ok and ok
        print(f"    {'[OK]' if ok else '[FAIL]'} {label}: {detail}")

    # hosts 属世界可读的系统公共文件：AppContainer（同 UWP）允许读取，
    # 属预期残留；仅作信息项，不计入通过判定
    hosts_res = windows_native.run_code(f'print(open(r"{HOSTS}").read())')
    print(f"    [INFO] 读宿主 hosts: {'可读（预期残留）' if hosts_res['ok'] else '被拒绝'}")
    if not hosts_res["ok"]:
        all_ok = False
        print("    [FAIL] hosts 不可读与预期不符，请检查运行环境")

    if OUTSIDE_FILE.exists():
        OUTSIDE_FILE.unlink()
        print("    [FAIL] 检测到目录外文件被创建！")
        all_ok = False
    if USER_SECRET.exists():
        USER_SECRET.unlink()

    # 4) 环境变量不继承
    os.environ["ARENA_TEST_SECRET"] = "secret-123"

    def env_check() -> tuple[bool, str]:
        res = windows_native.run_code(
            "import os; print(os.environ.get('ARENA_TEST_SECRET'))")
        leaked = "secret-123" in res["stdout"]
        return not leaked, ("泄露！" if leaked else "宿主环境变量未继承")

    ok, detail = _check("环境不继承", env_check)
    all_ok = all_ok and ok
    print(f"    {'[OK]' if ok else '[FAIL]'} 环境不继承: {detail}")

    # 5) 并发隔离（互异 SID）
    def concurrent_check() -> tuple[bool, str]:
        wd_a = Path(tempfile.mkdtemp(prefix="arena_xrun_a_"))
        wd_b = Path(tempfile.mkdtemp(prefix="arena_xrun_b_"))
        res: dict[str, dict] = {}

        def worker(key: str, wd: Path, code: str) -> None:
            res[key] = windows_native._spawn(exe, wd, code, "", windows_native.DEFAULT_LIMITS)

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
        shutil.rmtree(wd_a, ignore_errors=True)
        shutil.rmtree(wd_b, ignore_errors=True)

        if not res.get("a", {}).get("ok"):
            return False, f"任务 A 执行失败: {res['a']}"
        if "A" in res.get("b", {}).get("stdout", ""):
            return False, "任务 B 读到了任务 A 的工作目录文件（SID 未隔离）"
        if "DENIED" not in res.get("b", {}).get("stdout", ""):
            return False, f"任务 B 读取结果异常: {res['b']}"
        return True, "并发任务 SID 互异，跨任务工作目录互相不可读"

    ok, detail = _check("并发隔离", concurrent_check)
    all_ok = all_ok and ok
    print(f"    {'[OK]' if ok else '[FAIL]'} 并发隔离: {detail}")

    print("\n== 自检" + ("通过" if all_ok else "发现失败项") + " ==")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
