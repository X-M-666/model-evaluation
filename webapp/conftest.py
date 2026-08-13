# -*- coding: utf-8 -*-
"""pytest 根配置：保证无论从哪个目录运行都能导入 webapp/backend。

全局存储隔离（issue #11）：所有测试模块共享同一约束——存储目录重定向到
临时目录并自动清理，绝不污染仓库内的 .eval/ 与 webapp/data/。
全局网络封锁（issue #11 复审 R2-002）：未显式 mock 的真实外连立即失败
并指出调用位置，强制 README 的"所有测试零真实网络"约定。
"""
import socket
import sys
import traceback
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


# 网络封锁放行清单：仅回环（Playwright 本地 server / 浏览器 CDP）。
# 无任何外部主机例外（issue #16：运行时由部署方提供，测试与代码零下载）。
_ALLOWED_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


@pytest.fixture(autouse=True)
def _block_real_network(monkeypatch):
    """禁止测试期间的真实外部网络连接（R2-002 复审）。

    双层拦截：
    1. socket.socket.connect —— 覆盖 socket.create_connection、urllib 等同步路径；
    2. asyncio.BaseEventLoop.create_connection —— 覆盖所有异步 TCP（Windows
       Proactor 走 IOCP ConnectEx、Linux/macOS Selector 走 sock.connect_ex，
       两者均不经过 socket.socket.connect；httpx/anyio/Playwright 之外的
       asyncio 连接一律从此入口经过）。
    被拦截时抛异常并附调用栈，便于定位未 mock 的外连点。
    """
    import asyncio

    _orig_connect = socket.socket.connect

    def _blocked_connect(self, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else str(address)
        if host in _ALLOWED_HOSTS:
            return _orig_connect(self, address, *args, **kwargs)
        stack = " <- ".join(
            f"{f.name}:{f.lineno}" for f in traceback.extract_stack()[-4:-1]
        )
        raise RuntimeError(
            f"测试网络封锁：禁止真实外连 {address!r}（调用位置 {stack}）"
        )

    monkeypatch.setattr(socket.socket, "connect", _blocked_connect)

    _orig_create_connection = asyncio.BaseEventLoop.create_connection

    async def _blocked_create_connection(self, protocol_factory, host=None, port=None, *args, **kwargs):
        if host is not None and host not in _ALLOWED_HOSTS:
            stack = " <- ".join(
                f"{f.name}:{f.lineno}" for f in traceback.extract_stack()[-4:-1]
            )
            raise RuntimeError(
                f"测试网络封锁：禁止真实外连 {host!r}:{port}（调用位置 {stack}）"
            )
        return await _orig_create_connection(
            self, protocol_factory, host=host, port=port, *args, **kwargs
        )

    monkeypatch.setattr(
        asyncio.BaseEventLoop, "create_connection", _blocked_create_connection
    )
    yield


def pytest_collection_modifyitems(items):
    """按 fixture 依赖自动标记 native 测试（R2-008 复审）。

    依赖 native_runner fixture 的测试即为 Windows 专属隔离测试，统一打
    `native` 标记，供 CI 在 Linux/macOS 上以 -m "not native" 排除、在
    Windows 上全量真实执行；新增 native 测试无需再手动打标。
    """
    for item in items:
        if "native_runner" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.native)


@pytest.fixture(scope="module", autouse=True)
def _isolate_storage(tmp_path_factory):
    """module 级：将 BASE_DIR/DATASETS_DIR 重定向到临时目录，清空内存任务。

    module 级（而非 function 级）以兼容浏览器 e2e 测试的跨用例数据持久化；
    各测试文件自带的 function 级重定向可叠加覆盖，互不冲突。
    """
    from backend import main as main_module
    from backend import storage
    from backend import models_registry

    orig_base = storage.BASE_DIR
    orig_datasets = storage.DATASETS_DIR
    orig_stats = storage.STATS_DIR
    orig_generated = storage.GENERATED_DIR
    orig_badcases = storage.BADCASES_DIR
    orig_perturb = storage.PERTURB_DIR
    orig_leaderboards = storage.LEADERBOARD_DIR
    orig_models = models_registry.MODELS_DIR
    storage.BASE_DIR = tmp_path_factory.mktemp("history")
    storage.DATASETS_DIR = tmp_path_factory.mktemp("datasets")
    storage.STATS_DIR = tmp_path_factory.mktemp("stats")
    storage.SATURATION_FILE = storage.STATS_DIR / "saturation.json"
    storage.GENERATED_DIR = tmp_path_factory.mktemp("generated")
    storage.BADCASES_DIR = tmp_path_factory.mktemp("badcases")
    storage.PERTURB_DIR = tmp_path_factory.mktemp("perturb")
    storage.LEADERBOARD_DIR = tmp_path_factory.mktemp("leaderboards")
    models_registry.MODELS_DIR = tmp_path_factory.mktemp("models")
    models_registry.clear_memory_keys()
    for jid in list(main_module._jobs):
        main_module._jobs.pop(jid)
    for jid in list(main_module._tasks):
        main_module._tasks.pop(jid)

    yield

    storage.BASE_DIR = orig_base
    storage.DATASETS_DIR = orig_datasets
    storage.STATS_DIR = orig_stats
    storage.SATURATION_FILE = orig_stats / "saturation.json"
    storage.GENERATED_DIR = orig_generated
    storage.BADCASES_DIR = orig_badcases
    storage.PERTURB_DIR = orig_perturb
    storage.LEADERBOARD_DIR = orig_leaderboards
    models_registry.MODELS_DIR = orig_models
