# -*- coding: utf-8 -*-
"""任务调度器（迭代七，迭代 0 契约落地）：优先级队列 + 并发配额 + 排队管理。

纯逻辑模块（零 asyncio，确定性可测）：
- submit：入队，按 (priority 降序, enq_seq 升序) 排序（同优先级 FIFO）
- next_batch：配额内弹出可派发候选（高优先级插队 = 排序天然实现）
- release：任务释放配额后返回可补派发候选（幂等）
- cancel_queued：排队中移除；set_priority：仅排队中可改（重排序）
- queue_view：任务列表页数据（含排队位置）

v1 内存队列：进程重启后排队任务不自动恢复（由调用方在启动时沉降 error）；
运行中任务不可被抢占（优先级仅对排队中生效）。
"""
from __future__ import annotations

import time
from typing import Any


class Scheduler:
    """优先级队列调度器。concurrency 为并发配额（执行+评审中任务数上限）。"""

    def __init__(self, concurrency: int = 2):
        self._concurrency = max(1, int(concurrency))
        self._items: dict[str, dict[str, Any]] = {}   # job_id -> {priority, enq_seq, enq_at}
        self._running: set[str] = set()
        self._seq = 0

    # ---- 视图 ----

    def queue_view(self) -> list[dict[str, Any]]:
        """排队中任务视图（按调度顺序），含位置（1 起）与入队时间。"""
        ordered = self._sorted_ids()
        return [
            {
                "job_id": job_id,
                "priority": self._items[job_id]["priority"],
                "position": i + 1,
                "enq_at": self._items[job_id]["enq_at"],
            }
            for i, job_id in enumerate(ordered)
        ]

    def running(self) -> set[str]:
        return set(self._running)

    def active_count(self) -> int:
        return len(self._running)

    def concurrency(self) -> int:
        return self._concurrency

    def is_queued(self, job_id: str) -> bool:
        return job_id in self._items

    # ---- 操作 ----

    def submit(self, job_id: str, priority: int = 0) -> bool:
        """入队。已在队列/运行中 → False（幂等防重复入队）。"""
        if job_id in self._items or job_id in self._running:
            return False
        self._seq += 1
        self._items[job_id] = {
            "priority": int(priority),
            "enq_seq": self._seq,
            "enq_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        return True

    def next_batch(self) -> list[str]:
        """配额内弹出可派发候选（按调度顺序取前 N=配额-运行数 个）。"""
        free = self._concurrency - len(self._running)
        if free <= 0:
            return []
        ids = self._sorted_ids()[:free]
        for job_id in ids:
            self._items.pop(job_id, None)
            self._running.add(job_id)
        return ids

    def release(self, job_id: str) -> list[str]:
        """任务结束/离开执行阶段：释放配额，返回可补派发候选（幂等）。"""
        if job_id in self._running:
            self._running.discard(job_id)
        return self.next_batch()

    def cancel_queued(self, job_id: str) -> bool:
        """排队中移除（排队取消专用）。运行中 → False（走 cancelling 语义）。"""
        if job_id in self._items:
            self._items.pop(job_id, None)
            return True
        return False

    def set_priority(self, job_id: str, priority: int) -> bool:
        """修改排队中任务的优先级（重排序）。运行中/不存在 → False。"""
        item = self._items.get(job_id)
        if item is None:
            return False
        item["priority"] = int(priority)
        return True

    def clear(self) -> None:
        """清空队列与运行集（测试隔离用）。"""
        self._items.clear()
        self._running.clear()
        self._seq = 0

    def _sorted_ids(self) -> list[str]:
        return sorted(
            self._items,
            key=lambda jid: (-self._items[jid]["priority"], self._items[jid]["enq_seq"]),
        )
