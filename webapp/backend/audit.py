# -*- coding: utf-8 -*-
"""审计日志：以 JSONL 追加记录关键操作（issue #8 补强方案）。

- 文件位置：`<存储根>/.eval/audit.log`（即 storage.BASE_DIR 的父目录，
  conftest 的存储重定向自动生效，测试不污染仓库）。
- 事件字段白名单：只允许预定义字段，写入前统一经 redact_sensitive 递归脱敏
  （纵深防御），保证 API Key 等敏感内容永不进入日志。
- 审计失败静默：日志异常不得影响主流程（写盘错误、磁盘满等一律吞掉）。
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any

from backend.security import redact_sensitive

ALLOWED_KEYS = frozenset({"ts", "event", "job_id", "target", "path", "actor"})

_lock = threading.Lock()


def _log_path() -> Any:
    from backend import storage
    return storage.BASE_DIR.parent / "audit.log"


def _append(event: dict[str, Any]) -> None:
    try:
        entry = {k: v for k, v in event.items() if k in ALLOWED_KEYS}
        entry = redact_sensitive(entry)
        entry.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        p = _log_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with _lock:
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def eval_started(job_id: str, actor: str = "local") -> None:
    _append({"event": "eval_started", "job_id": job_id, "actor": actor})


def review_submitted(job_id: str, actor: str = "local") -> None:
    _append({"event": "review_submitted", "job_id": job_id, "actor": actor})


def history_deleted(job_id: str, actor: str = "local") -> None:
    _append({"event": "history_deleted", "job_id": job_id, "actor": actor})


def eval_cancelled(job_id: str, actor: str = "local") -> None:
    """运行中评测被删除时记录取消事件（issue #14 / R2-005）。"""
    _append({"event": "eval_cancelled", "job_id": job_id, "actor": actor})


def dataset_uploaded(name: str, actor: str = "local") -> None:
    _append({"event": "dataset_uploaded", "target": name, "actor": actor})


def dataset_deleted(name: str, actor: str = "local") -> None:
    _append({"event": "dataset_deleted", "target": name, "actor": actor})


def model_registered(model_id: str, actor: str = "local") -> None:
    """模型配置库新增配置（迭代一；Key 不落盘，仅存内存）。"""
    _append({"event": "model_registered", "target": model_id, "actor": actor})


def model_deleted(model_id: str, actor: str = "local") -> None:
    """模型配置库删除配置。"""
    _append({"event": "model_deleted", "target": model_id, "actor": actor})


def auth_failed(path: str, actor: str = "unknown") -> None:
    _append({"event": "auth_failed", "path": path, "actor": actor})


def read_events() -> list[dict[str, Any]]:
    """读取全部审计事件（测试与运维排查用）。"""
    p = _log_path()
    if not p.exists():
        return []
    events = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events
