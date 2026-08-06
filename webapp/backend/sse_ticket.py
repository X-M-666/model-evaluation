# -*- coding: utf-8 -*-
"""SSE 短时单次 ticket（issue #13 / R2-004 修复）。

共享模式下长期管理员 Token 不再经 URL 传递：前端先通过已认证的
POST /api/eval/{job_id}/events/ticket 换取随机 ticket，再以原生
EventSource 携带 `?ticket=...` 建立 SSE 连接。

- 短时：默认 TTL 60 秒，仅覆盖「签发 → 建立连接」的间隙，断线重连由
  前端重新签发。
- 单次：consume 通过 threading.Lock 原子校验并立即删除条目（消费即焚），
  同一 ticket 的并发/重放请求只会有一个成功。
- 作用域：ticket 与签发时的 job_id 绑定，只能访问该 job 的 /events 路由，
  不能用于任何其他 API。
"""
from __future__ import annotations

import os
import secrets
import threading
import time

TTL_SECONDS = float(os.environ.get("MODEL_DUEL_SSE_TICKET_TTL", "60"))
MAX_TICKETS = 10_000

_tickets: dict[str, dict] = {}
_lock = threading.Lock()


def issue(job_id: str) -> str:
    """为指定 job 签发一个短时单次 ticket，返回随机凭证串。"""
    ticket = secrets.token_urlsafe(32)
    with _lock:
        _evict_expired()
        if len(_tickets) >= MAX_TICKETS:
            # 容量上限：优先淘汰最早签发的（dict 保持插入序）
            oldest = next(iter(_tickets))
            _tickets.pop(oldest)
        _tickets[ticket] = {
            "job_id": job_id,
            "exp": time.monotonic() + TTL_SECONDS,
        }
    return ticket


def consume(ticket: str, job_id: str) -> bool:
    """单次消费：原子校验（存在/未过期/job 匹配）通过后立即删除条目。

    消费即焚：成功后条目即不存在，重放/并发只能得到 False；
    已用条目不驻留内存（R3-001 残余 4）。调用方不得记录 ticket 本身。
    """
    with _lock:
        rec = _tickets.get(ticket)
        if rec is None:
            return False
        if time.monotonic() > rec["exp"]:
            _tickets.pop(ticket, None)
            return False
        if rec["job_id"] != job_id:
            return False
        _tickets.pop(ticket, None)
        return True


def _evict_expired() -> None:
    """惰性清理过期条目（仅在持有锁时调用）。"""
    now = time.monotonic()
    for k in [k for k, r in _tickets.items() if now > r["exp"]]:
        _tickets.pop(k, None)
