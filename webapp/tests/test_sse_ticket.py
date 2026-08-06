# -*- coding: utf-8 -*-
"""SSE 短时单次 ticket 单元测试（issue #13 / R2-004）。"""
from __future__ import annotations

import threading
import time

import pytest

from backend import sse_ticket


@pytest.fixture(autouse=True)
def _clear_tickets():
    sse_ticket._tickets.clear()
    yield
    sse_ticket._tickets.clear()


def test_issue_returns_unique_tickets():
    a = sse_ticket.issue("job-1")
    b = sse_ticket.issue("job-1")
    assert a and b and a != b


def test_consume_ok():
    t = sse_ticket.issue("job-1")
    assert sse_ticket.consume(t, "job-1") is True


def test_consume_removes_entry():
    """R3-001 残余 4：消费成功后条目立即删除，内存零残留。"""
    t = sse_ticket.issue("job-1")
    assert sse_ticket.consume(t, "job-1") is True
    assert t not in sse_ticket._tickets


def test_consume_twice_fails():
    t = sse_ticket.issue("job-1")
    assert sse_ticket.consume(t, "job-1") is True
    assert sse_ticket.consume(t, "job-1") is False


def test_consume_unknown_ticket():
    assert sse_ticket.consume("no-such-ticket", "job-1") is False


def test_consume_job_mismatch():
    t = sse_ticket.issue("job-1")
    assert sse_ticket.consume(t, "job-2") is False
    # 作用域不匹配不消耗 ticket，正确 job 仍可用
    assert sse_ticket.consume(t, "job-1") is True


def test_consume_expired():
    t = sse_ticket.issue("job-1")
    sse_ticket._tickets[t]["exp"] = time.monotonic() - 1
    assert sse_ticket.consume(t, "job-1") is False
    # 过期条目被惰性清理
    assert t not in sse_ticket._tickets


def test_issue_evicts_expired():
    stale = sse_ticket.issue("job-1")
    sse_ticket._tickets[stale]["exp"] = time.monotonic() - 1
    sse_ticket.issue("job-2")
    assert stale not in sse_ticket._tickets


def test_issue_cap_evicts_oldest(monkeypatch):
    monkeypatch.setattr(sse_ticket, "MAX_TICKETS", 2)
    first = sse_ticket.issue("job-1")
    sse_ticket.issue("job-2")
    sse_ticket.issue("job-3")
    assert len(sse_ticket._tickets) == 2
    assert first not in sse_ticket._tickets


def test_concurrent_single_use():
    """并发消费同一 ticket 只允许一次成功（原子性）。"""
    t = sse_ticket.issue("job-1")
    results = []
    lock = threading.Lock()

    def worker():
        ok = sse_ticket.consume(t, "job-1")
        with lock:
            results.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert results.count(True) == 1


def test_ttl_env_configurable(monkeypatch):
    monkeypatch.setattr(sse_ticket, "TTL_SECONDS", 5)
    t = sse_ticket.issue("job-1")
    assert abs(sse_ticket._tickets[t]["exp"] - (time.monotonic() + 5)) < 1
