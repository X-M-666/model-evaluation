# -*- coding: utf-8 -*-
"""pytest 根配置：保证无论从哪个目录运行都能导入 webapp/backend。

全局存储隔离（issue #11）：所有测试模块共享同一约束——存储目录重定向到
临时目录并自动清理，绝不污染仓库内的 .eval/ 与 webapp/data/。
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(scope="module", autouse=True)
def _isolate_storage(tmp_path_factory):
    """module 级：将 BASE_DIR/DATASETS_DIR 重定向到临时目录，清空内存任务。

    module 级（而非 function 级）以兼容浏览器 e2e 测试的跨用例数据持久化；
    各测试文件自带的 function 级重定向可叠加覆盖，互不冲突。
    """
    from backend import main as main_module
    from backend import storage

    orig_base = storage.BASE_DIR
    orig_datasets = storage.DATASETS_DIR
    storage.BASE_DIR = tmp_path_factory.mktemp("history")
    storage.DATASETS_DIR = tmp_path_factory.mktemp("datasets")
    for jid in list(main_module._jobs):
        main_module._jobs.pop(jid)

    yield

    storage.BASE_DIR = orig_base
    storage.DATASETS_DIR = orig_datasets
