# -*- coding: utf-8 -*-
"""Windows 原生代码隔离：AppContainer（文件/网络硬隔离）+ Job Object（资源配额）。

安全边界（均为操作系统级强制，非 Python 黑名单，不可被 io.open/反射绕过）：
- AppContainer：以空能力集创建低权限容器进程。容器 SID 未显式授权的路径
  一律拒绝访问；默认无网络能力，网络连接被系统拒绝。
- Job Object：内存上限、CPU 时间上限、活动进程数上限、KILL_ON_JOB_CLOSE；
  超时后 TerminateJobObject 强杀整棵进程树。
- 环境变量白名单：不继承宿主凭据与多余环境变量。
- 运行解释器为部署方配置的自包含 Python 运行时（MODEL_DUEL_SANDBOX_PYTHON，
  校验见 bootstrap；应用不下载、不解压、不更新任何运行时）。

仅支持 Windows（win32）。标准用户即可运行，无需管理员。
"""
from __future__ import annotations

import ctypes
import os
import shutil
import sys
import tempfile
import threading
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any

HOSTS_FILE = r"C:\Windows\System32\drivers\etc\hosts"

DEFAULT_LIMITS = {
    "timeout_sec": 15,
    "memory_mb": 256,
    "cpu_time_sec": 5,
    "max_active_processes": 8,
    "max_output_chars": 4096,
    # 整个一次性工作目录的总占用上限（运行期监视的软限制，非 OS 硬配额；
    # Job Object 无 IO 配额，标准用户下 Windows 亦无每任务磁盘配额）。
    # 超额按 0.05s 轮询终止，单轮窗口最大超额 ≈ 间隔 × 峰值写入速率；
    # 进程退出后复核兜底，终止后目录即清理。
    "max_output_bytes": 1024 * 1024,
}

APPCONTAINER_NAME = "arena.code-sandbox"

# 磁盘占用轮询间隔（秒）：越短则轮询窗口内的最大超额越小，
# 但会提高每次执行的开销；正常任务目录仅数个小文件，0.05s 开销可忽略。
WATCH_INTERVAL = 0.05

# 并发运行隔离池大小：>= executor._CODE_SEM(2) 即保证并发运行的 SID 互异
PROFILE_POOL_SIZE = 4

# ---- Windows 常量 ----
CREATE_SUSPENDED = 0x00000004
CREATE_NO_WINDOW = 0x08000000
CREATE_UNICODE_ENVIRONMENT = 0x00000400
EXTENDED_STARTUPINFO_PRESENT = 0x00080000
STARTF_USESTDHANDLES = 0x00000100
PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES = 0x00020009
GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
FILE_SHARE_READ = 0x00000001
FILE_SHARE_WRITE = 0x00000002
OPEN_EXISTING = 3
CREATE_ALWAYS = 2
FILE_ATTRIBUTE_NORMAL = 0x80
WAIT_TIMEOUT = 0x00000102

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_LIMIT_PROCESS_TIME = 0x00000002
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
JobObjectExtendedLimitInformation = 9

# ---- ctypes 结构 ----
class _LARGE_INTEGER(ctypes.Structure):
    _fields_ = [("QuadPart", ctypes.c_longlong)]


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", _LARGE_INTEGER),
        ("PerJobUserTimeLimit", _LARGE_INTEGER),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _SECURITY_CAPABILITIES(ctypes.Structure):
    _fields_ = [
        ("AppContainerSid", ctypes.c_void_p),
        ("CapabilitySids", ctypes.c_void_p),
        ("CapabilityCount", wintypes.DWORD),
        ("Reserved", wintypes.DWORD),
    ]


class _SECURITY_ATTRIBUTES(ctypes.Structure):
    _fields_ = [
        ("nLength", wintypes.DWORD),
        ("lpSecurityDescriptor", wintypes.LPVOID),
        ("bInheritHandle", wintypes.BOOL),
    ]


class _STARTUPINFOW(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.POINTER(wintypes.BYTE)),
        ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class _STARTUPINFOEX(ctypes.Structure):
    _fields_ = [
        ("StartupInfo", _STARTUPINFOW),
        ("lpAttributeList", ctypes.c_void_p),
    ]


class _PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
    ]


_API_CACHE: dict[str, Any] = {}


def _api() -> dict[str, Any]:
    """惰性加载 Win32 API（仅 Windows）。"""
    if _API_CACHE:
        return _API_CACHE
    if os.name != "nt":
        raise RuntimeError("Windows 原生隔离仅支持 Windows")

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    userenv = ctypes.WinDLL("userenv", use_last_error=True)

    userenv.DeriveAppContainerSidFromAppContainerName.argtypes = [
        wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p)]
    userenv.DeriveAppContainerSidFromAppContainerName.restype = ctypes.c_long
    userenv.CreateAppContainerProfile.argtypes = [
        wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.LPCWSTR,
        ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p)]
    userenv.CreateAppContainerProfile.restype = ctypes.c_long

    advapi32.ConvertSidToStringSidW.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)]
    advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL

    kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [
        wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.TerminateJobObject.restype = wintypes.BOOL
    kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.TerminateProcess.restype = wintypes.BOOL
    kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
    kernel32.ResumeThread.restype = wintypes.DWORD
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [wintypes.HANDLE]
    kernel32.LocalFree.restype = wintypes.HANDLE

    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
        ctypes.POINTER(_SECURITY_ATTRIBUTES), wintypes.DWORD,
        wintypes.DWORD, wintypes.HANDLE]
    kernel32.CreateFileW.restype = wintypes.HANDLE

    kernel32.CreateProcessW.argtypes = [
        wintypes.LPCWSTR, wintypes.LPWSTR,
        wintypes.LPVOID, wintypes.LPVOID, wintypes.BOOL,
        wintypes.DWORD, wintypes.LPVOID, wintypes.LPCWSTR,
        ctypes.POINTER(_STARTUPINFOW), ctypes.POINTER(_PROCESS_INFORMATION)]
    kernel32.CreateProcessW.restype = wintypes.BOOL

    kernel32.InitializeProcThreadAttributeList.argtypes = [
        wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(ctypes.c_size_t)]
    kernel32.InitializeProcThreadAttributeList.restype = wintypes.BOOL
    kernel32.UpdateProcThreadAttribute.argtypes = [
        wintypes.LPVOID, wintypes.DWORD, ctypes.c_size_t,
        wintypes.LPVOID, ctypes.c_size_t, wintypes.LPVOID, ctypes.POINTER(ctypes.c_size_t)]
    kernel32.UpdateProcThreadAttribute.restype = wintypes.BOOL
    kernel32.DeleteProcThreadAttributeList.argtypes = [wintypes.LPVOID]
    kernel32.DeleteProcThreadAttributeList.restype = None

    _API_CACHE.update(
        kernel32=kernel32, advapi32=advapi32, userenv=userenv,
    )
    return _API_CACHE


# ---- AppContainer SID / 配置文件 ----

class _ProfilePool:
    """固定 N 个 AppContainer profile 轮换分配，保证并发运行 SID 互异。

    背景：若所有运行共享同一 SID，则每个运行的工作目录 ACL（对该 SID
    授予 RXWM）都会被其他并发运行的沙箱代码读写，造成跨任务数据访问。
    池大小（>= executor._CODE_SEM）即保证并发互异；
    跨时间复用无风险——工作目录每次执行后即删除。
    """

    def __init__(self, size: int = PROFILE_POOL_SIZE):
        self._names = [f"arena.codesb.{i}" for i in range(size)]
        self._free: list[str] = list(self._names)
        self._lock = threading.Lock()

    def acquire(self) -> str:
        with self._lock:
            if not self._free:
                raise RuntimeError("AppContainer profile 池耗尽（并发超过上限）")
            return self._free.pop()

    def release(self, name: str) -> None:
        with self._lock:
            if name in self._names and name not in self._free:
                self._free.append(name)


_PROFILE_POOL = _ProfilePool()


def _appcontainer_sid(name: str) -> tuple[ctypes.c_void_p, str]:
    """按容器名推导/创建 AppContainer SID，返回 (SID 句柄, SID 字符串)。

    CreateAppContainerProfile 幂等：已存在时返回错误可忽略，SID 由推导得到。
    """
    api = _api()
    userenv, advapi32, kernel32 = api["userenv"], api["advapi32"], api["kernel32"]

    sid = ctypes.c_void_p()
    hr = userenv.DeriveAppContainerSidFromAppContainerName(
        name, ctypes.byref(sid))
    if hr < 0:
        raise RuntimeError(f"DeriveAppContainerSid 失败 HRESULT={hr:#x}")

    sid2 = ctypes.c_void_p()
    userenv.CreateAppContainerProfile(
        name, "Arena Code Sandbox", "模型代码隔离容器",
        None, 0, ctypes.byref(sid2))

    sid_str = wintypes.LPWSTR()
    if not advapi32.ConvertSidToStringSidW(sid, ctypes.byref(sid_str)):
        kernel32.LocalFree(sid)
        raise RuntimeError("ConvertSidToStringSid 失败")
    return sid, sid_str.value


# ---- 文件/ACL ----

def _grant_sid(workdir: Path, sid_str: str, perms: str, timeout: int = 30) -> None:
    """用 icacls 将目录 ACL 授予容器 SID（目录为当前用户所有，无需管理员）。"""
    import subprocess

    proc = subprocess.run(
        ["icacls", str(workdir), "/grant", f"*{sid_str}:{perms}", "/T", "/Q"],
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"icacls 授权失败({workdir}): {proc.stderr.strip()}")


#: 运行时目录授权缓存：key=(SID, 目录, mtime_ns)。ACL 持久化在目录上，
#: 无需每次执行重复递归授权（完整安装树 icacls /T 可能超过 30s）。
#: 目录被部署方替换（mtime 变化）时自动失效，下次执行重新授权。
_GRANT_CACHE: set[tuple[str, str, int]] = set()
_GRANT_LOCK = threading.Lock()


def _grant_runtime_sid(runtime_dir: Path, sid_str: str) -> None:
    """为容器 SID 授权运行时目录读取/执行（进程内缓存，仅首次实际执行）。"""
    try:
        stamp = (sid_str, str(runtime_dir), runtime_dir.stat().st_mtime_ns)
    except OSError:
        stamp = (sid_str, str(runtime_dir), 0)
    with _GRANT_LOCK:
        if stamp in _GRANT_CACHE:
            return
    _grant_sid(runtime_dir, sid_str, "(OI)(CI)RX", timeout=120)
    with _GRANT_LOCK:
        _GRANT_CACHE.add(stamp)


def _build_env_block(env: dict[str, str]) -> bytes:
    data = "".join(f"{k}={v}\0" for k, v in env.items())
    return (data + "\0").encode("utf-16le")


def _read_capped(path: Path, max_chars: int) -> str:
    """限量读取子进程输出文件：最多读 max_chars+1 字符，避免大文件整体入内存。"""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            return fh.read(max_chars + 1)[:max_chars]
    except OSError:
        return ""


def _dir_size(path: Path) -> int:
    """统计目录内全部文件 st_size 之和（含子目录，不跟随符号链接）。

    这是"工作目录磁盘占用"的直接度量，覆盖 stdout/stderr/普通文件及
    多文件累计；遍历中目录被增删导致的 OSError 不影响安全（本次放弃，
    下轮重试），单独文件的 stat 失败（如句柄独占）跳过。
    """
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += (Path(root) / name).stat().st_size
                except OSError:
                    pass
    except OSError:
        return 0
    return total


# ---- 进程启动 ----

def _create_io_file(path: Path, read: bool) -> wintypes.HANDLE:
    api = _api()
    kernel32 = api["kernel32"]
    sa = _SECURITY_ATTRIBUTES()
    sa.nLength = ctypes.sizeof(_SECURITY_ATTRIBUTES)
    sa.bInheritHandle = True
    access = GENERIC_READ if read else GENERIC_WRITE
    # 输出文件允许共享读写：子进程退出后父进程需重新打开读取，
    # 且 Job 终止期间继承句柄的孙进程可能仍存活，禁止共享会导致
    # ERROR_SHARING_VIOLATION（不影响安全：ACL 与容器限制不变）
    share = FILE_SHARE_READ | FILE_SHARE_WRITE
    disposition = OPEN_EXISTING if read else CREATE_ALWAYS
    h = kernel32.CreateFileW(
        str(path), access, share, ctypes.byref(sa), disposition,
        FILE_ATTRIBUTE_NORMAL, None)
    if h == wintypes.HANDLE(-1).value or h is None:
        raise ctypes.WinError(ctypes.get_last_error())
    return h


def probe() -> tuple[bool, str]:
    """检测当前环境是否可运行 native-sandbox（不实际执行用户代码）。"""
    if os.name != "nt":
        return False, "仅支持 Windows"
    try:
        _appcontainer_sid(APPCONTAINER_NAME)
    except Exception as exc:
        return False, f"AppContainer 初始化失败: {exc}"
    from backend.engine.isolation import bootstrap

    ready, detail = bootstrap.runtime_ready()
    if not ready:
        return False, f"运行时不完整（{detail}），请按 README 配置 {bootstrap.RUNTIME_ENV} 后重试"
    return True, "AppContainer + Job Object 可用"


def selfcheck() -> tuple[bool, str]:
    """真实执行一次最小无害任务，验证受限进程可启动且输出回传正常。

    与 probe() 的区别（R2-008 复审）：probe() 只验证组件存在（快检，
    用于请求路径）；selfcheck() 实际创建受限进程，验证安全边界生效
    （用于 sandbox_selfcheck 脚本与 Windows CI）。
    """
    ok, detail = probe()
    if not ok:
        return False, detail
    try:
        res = run_code("print(1+1)")
    except Exception as exc:  # noqa: BLE001
        return False, f"最小任务执行异常: {exc}"
    if not res.get("ok"):
        return False, f"最小任务执行失败: {res}"
    if res.get("stdout", "").strip() != "2":
        return False, f"stdout 异常: {res.get('stdout')!r}"
    return True, "受限进程启动 + 输出回传正常"


def _spawn(
    python_exe: Path,
    workdir: Path,
    code: str,
    stdin_text: str,
    limits: dict[str, Any],
) -> dict[str, Any]:
    """以 AppContainer + Job Object 启动部署方配置的 python 执行 code。

    通过文件传递代码/输入/输出（stdin 重定向自 input.txt，
    stdout/stderr 重定向至容器可写的 stdout.txt/stderr.txt），
    避免管道句柄继承问题。
    """
    api = _api()
    kernel32 = api["kernel32"]

    code_file = workdir / "main.py"
    in_file = workdir / "input.txt"
    out_file = workdir / "stdout.txt"
    err_file = workdir / "stderr.txt"
    code_file.write_text(code, encoding="utf-8")
    in_file.write_text(stdin_text or "", encoding="utf-8")

    profile = _PROFILE_POOL.acquire()
    sid = None
    try:
        sid, sid_str = _appcontainer_sid(profile)
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())

        handles: list[wintypes.HANDLE] = []
        try:
            # ---- Job Object 资源配额 ----
            jeli = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            jeli.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if limits.get("memory_mb"):
                jeli.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_JOB_MEMORY
                jeli.JobMemoryLimit = int(limits["memory_mb"]) * 1024 * 1024
            if limits.get("cpu_time_sec"):
                jeli.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_PROCESS_TIME
                jeli.BasicLimitInformation.PerProcessUserTimeLimit.QuadPart = (
                    int(limits["cpu_time_sec"] * 10_000_000))
            if limits.get("max_active_processes"):
                jeli.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_ACTIVE_PROCESS
                jeli.BasicLimitInformation.ActiveProcessLimit = int(limits["max_active_processes"])
            if not kernel32.SetInformationJobObject(
                    job, JobObjectExtendedLimitInformation, ctypes.byref(jeli),
                    ctypes.sizeof(_JOBOBJECT_EXTENDED_LIMIT_INFORMATION)):
                raise ctypes.WinError(ctypes.get_last_error())

            # ---- 工作目录授权（容器 SID 可读写，其余路径仍被系统拒绝）----
            _grant_sid(workdir, sid_str, "(OI)(CI)RXWM")

            # ---- 运行时授权：AppContainer 需可读取部署方配置的运行时目录----
            # （python.exe 与 DLL 位于部署方配置的目录，宿主 ACL
            # 不含容器 SID，否则子进程以 STATUS_DLL_NOT_FOUND 启动失败；
            # 授权按目录缓存，避免每次执行都对完整安装树递归 icacls）
            _grant_runtime_sid(python_exe.parent, sid_str)

            # ---- 标准输入/输出文件 ----
            h_in = _create_io_file(in_file, read=True)
            h_out = _create_io_file(out_file, read=False)
            h_err = _create_io_file(err_file, read=False)
            handles += [h_in, h_out, h_err]

            # ---- 环境白名单（不继承宿主凭据）----
            # LOCALAPPDATA 必须存在：AppContainer 进程创建时内核需用它
            # 推导包状态目录（%LOCALAPPDATA%\Packages\<SID>），缺失会导致
            # ERROR_ENVVAR_NOT_FOUND (203)；指向工作目录而非宿主路径。
            env = {
                "TEMP": str(workdir),
                "TMP": str(workdir),
                "LOCALAPPDATA": str(workdir),
                "SYSTEMROOT": os.environ.get("SYSTEMROOT", r"C:\Windows"),
                "WINDIR": os.environ.get("WINDIR", r"C:\Windows"),
                "PYTHONDONTWRITEBYTECODE": "1",
                # 强制 UTF-8 模式：子进程 stdin/stdout/stderr 编码确定性，
                # 不受宿主 locale（如 GBK）影响，否则中文输出经 stdout.txt
                # 按 UTF-8 回读时被替换为 U+FFFD，导致逐字节比对误判
                "PYTHONUTF8": "1",
            }
            env_block = _build_env_block(env)

            # ---- 线程属性列表（SECURITY_CAPABILITIES）----
            # 首次调用（缓冲区为 NULL）仅用于查询所需尺寸，返回 ERROR_INSUFFICIENT_BUFFER 属正常
            size = ctypes.c_size_t(0)
            kernel32.InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(size))
            if not size.value:
                raise ctypes.WinError(ctypes.get_last_error())
            attr_buf = ctypes.create_string_buffer(size.value)
            if not kernel32.InitializeProcThreadAttributeList(attr_buf, 1, 0, ctypes.byref(size)):
                raise ctypes.WinError(ctypes.get_last_error())
            caps = _SECURITY_CAPABILITIES()
            caps.AppContainerSid = sid
            caps.CapabilitySids = None
            caps.CapabilityCount = 0
            caps.Reserved = 0
            if not kernel32.UpdateProcThreadAttribute(
                    attr_buf, 0, PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES,
                    ctypes.byref(caps), ctypes.sizeof(_SECURITY_CAPABILITIES), None, None):
                raise ctypes.WinError(ctypes.get_last_error())

            # ---- 启动进程（CREATE_SUSPENDED → 入 Job → 恢复）----
            si = _STARTUPINFOEX()
            si.StartupInfo.cb = ctypes.sizeof(_STARTUPINFOEX)
            si.StartupInfo.dwFlags = STARTF_USESTDHANDLES
            si.StartupInfo.hStdInput = h_in
            si.StartupInfo.hStdOutput = h_out
            si.StartupInfo.hStdError = h_err
            si.lpAttributeList = ctypes.cast(attr_buf, ctypes.c_void_p).value

            pi = _PROCESS_INFORMATION()
            flags = CREATE_SUSPENDED | CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT
            cmdline = f'"{python_exe}" main.py'
            ok = kernel32.CreateProcessW(
                str(python_exe), cmdline, None, None, True, flags,
                env_block, str(workdir),
                ctypes.cast(ctypes.byref(si), ctypes.POINTER(_STARTUPINFOW)),
                ctypes.byref(pi))
            if not ok:
                raise ctypes.WinError(ctypes.get_last_error())

            try:
                if not kernel32.AssignProcessToJobObject(job, pi.hProcess):
                    err = ctypes.get_last_error()
                    kernel32.TerminateProcess(pi.hProcess, 1)
                    raise RuntimeError(
                        f"AssignProcessToJobObject 失败（宿主进程可能已处于不可嵌套的 Job）：WinError {err}")
                kernel32.ResumeThread(pi.hThread)

                # ---- 工作目录磁盘占用监视（Job Object 无 IO 配额，需自行兜底）----
                # 恶意代码可无限写任意文件（含 stdout/stderr）填满宿主磁盘；
                # 监视线程在运行期间轮询整个工作目录总占用（而非仅输出文件），
                # 超限即 TerminateJobObject 强杀。这是软限制：单轮窗口内的
                # 最大超额 ≈ WATCH_INTERVAL × 峰值写入速率，进程退出后复核兜底。
                max_bytes = int(limits.get("max_output_bytes", 1024 * 1024))
                overflowed = False
                peak_dir_bytes = 0
                stop = threading.Event()
                watcher = None

                def _watch_workdir():
                    nonlocal overflowed, peak_dir_bytes
                    while not stop.is_set():
                        current = _dir_size(workdir)
                        if current > peak_dir_bytes:
                            peak_dir_bytes = current
                        if current > max_bytes:
                            overflowed = True
                            kernel32.TerminateJobObject(job, 1)
                            return
                        time.sleep(WATCH_INTERVAL)

                try:
                    watcher = threading.Thread(target=_watch_workdir, daemon=True)
                    watcher.start()

                    # ---- 等待 / 超时强杀 ----
                    timeout_ms = int(float(limits.get("timeout_sec", 15)) * 1000)
                    wait = kernel32.WaitForSingleObject(pi.hProcess, timeout_ms)
                    timed_out = wait == WAIT_TIMEOUT
                    if timed_out:
                        kernel32.TerminateJobObject(job, 1)
                        kernel32.WaitForSingleObject(pi.hProcess, 5000)
                finally:
                    stop.set()
                    if watcher is not None:
                        watcher.join(timeout=5)
                # 无论正常退出与否都终止 Job，确保子孙进程不残留
                kernel32.TerminateJobObject(job, 1)
                # 等待 Job 内全部进程退出后复核目录总占用：单次高速大写入
                # 可能在轮询采样前完成（进程随即退出），此处兜底捕获，
                # 确保最终磁盘占用不超过配额；统计值同时纳入 peak 记录。
                kernel32.WaitForSingleObject(pi.hProcess, 5000)
                final_bytes = _dir_size(workdir)
                if final_bytes > peak_dir_bytes:
                    peak_dir_bytes = final_bytes
                if final_bytes > max_bytes:
                    overflowed = True
                rc = wintypes.DWORD()
                kernel32.GetExitCodeProcess(pi.hProcess, ctypes.byref(rc))
                # 子进程已退出，先关闭输出句柄（share=0），否则父进程
                # 重新打开 stdout.txt 读取会因共享冲突失败
                for h in handles:
                    kernel32.CloseHandle(h)
                handles.clear()
            finally:
                kernel32.CloseHandle(pi.hThread)
                kernel32.CloseHandle(pi.hProcess)
                kernel32.DeleteProcThreadAttributeList(attr_buf)

            max_chars = int(limits.get("max_output_chars", 4096))
            if overflowed:
                error_msg = f"输出超出配额上限（{max_bytes} 字节）"
            elif timed_out:
                error_msg = "运行超时（超过隔离配额）"
            else:
                error_msg = None
            return {
                "ok": not timed_out and not overflowed and rc.value == 0,
                "stdout": _read_capped(out_file, max_chars),
                "stderr": _read_capped(err_file, max_chars),
                "timed_out": timed_out,
                "output_overflow": overflowed,
                "peak_dir_bytes": peak_dir_bytes,
                "returncode": rc.value,
                "error": error_msg,
            }
        finally:
            for h in handles:
                kernel32.CloseHandle(h)
            if job:
                kernel32.CloseHandle(job)
    finally:
        if sid:
            kernel32.LocalFree(sid)
        _PROFILE_POOL.release(profile)


def run_code(code: str, stdin_text: str = "", limits: dict[str, Any] | None = None) -> dict[str, Any]:
    """在 Windows 原生隔离环境中运行一段 Python 代码。"""
    if os.name != "nt":
        return {"ok": False, "stdout": "", "stderr": "", "timed_out": False,
                "returncode": None, "error": "仅支持 Windows"}
    from backend.engine.isolation import bootstrap

    limits = {**DEFAULT_LIMITS, **(limits or {})}
    python_exe_path = bootstrap.resolve_runtime()
    workdir = Path(tempfile.mkdtemp(prefix="arena_iso_"))
    try:
        result = _spawn(python_exe_path, workdir, code, stdin_text, limits)
    except Exception as exc:
        result = {"ok": False, "stdout": "", "stderr": "", "timed_out": False,
                  "returncode": None, "error": f"沙箱执行异常: {exc}"}
    # 终止后可靠清理；失败不静默（重试一次仍失败则标记可观测，
    # 部署方可据此人工处置残留目录）
    try:
        shutil.rmtree(workdir)
    except OSError as exc:
        time.sleep(0.5)
        try:
            shutil.rmtree(workdir)
        except OSError:
            result["cleanup_failed"] = True
            result["error"] = (result.get("error") or "") + f"；沙箱目录清理失败: {exc}"
    return result
