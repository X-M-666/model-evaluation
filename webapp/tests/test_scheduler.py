# -*- coding: utf-8 -*-
"""迭代七：调度器（scheduler.py）纯逻辑单测。

覆盖：submit 排序（优先级/FIFO）、配额内派发、超配额排队、release 释放补派发
（幂等）、cancel_queued、set_priority 重排、运行集/视图（位置）、clear。
"""
from backend.scheduler import Scheduler


class TestSubmitAndOrder:
    def test_fifo_same_priority(self):
        s = Scheduler(concurrency=1)
        assert s.submit("j1")
        assert s.submit("j2")
        assert s.submit("j3")
        assert [x["job_id"] for x in s.queue_view()] == ["j1", "j2", "j3"]

    def test_higher_priority_earlier(self):
        s = Scheduler(concurrency=1)
        s.submit("j1", priority=0)
        s.submit("j2", priority=5)
        s.submit("j3", priority=-1)
        assert [x["job_id"] for x in s.queue_view()] == ["j2", "j1", "j3"]

    def test_duplicate_submit_rejected(self):
        s = Scheduler(concurrency=1)
        assert s.submit("j1") is True
        assert s.submit("j1") is False
        assert len(s.queue_view()) == 1

    def test_positions(self):
        s = Scheduler(concurrency=1)
        s.submit("j1", priority=1)
        s.submit("j2", priority=2)
        view = s.queue_view()
        assert view[0]["position"] == 1 and view[0]["job_id"] == "j2"
        assert view[1]["position"] == 2 and view[1]["job_id"] == "j1"


class TestDispatch:
    def test_dispatch_within_quota(self):
        s = Scheduler(concurrency=2)
        s.submit("j1")
        s.submit("j2")
        s.submit("j3")
        assert s.next_batch() == ["j1", "j2"]   # 配额 2 立即派发
        assert s.running() == {"j1", "j2"}
        assert s.active_count() == 2
        assert len(s.queue_view()) == 1         # j3 排队

    def test_high_priority_queued_skips(self):
        s = Scheduler(concurrency=1)
        s.submit("low1", priority=0)
        s.submit("low2", priority=0)
        s.next_batch()                          # low1 运行
        s.submit("high", priority=9)
        # low2 与 high 排队，high 优先
        assert [x["job_id"] for x in s.queue_view()] == ["high", "low2"]

    def test_release_dispatches_next(self):
        s = Scheduler(concurrency=1)
        s.submit("a")
        s.submit("b")
        s.next_batch()
        assert s.running() == {"a"}
        assert s.release("a") == ["b"]          # 释放后补派发
        assert s.running() == {"b"}
        assert s.release("b") == []             # 无排队 → 空
        assert s.active_count() == 0

    def test_release_idempotent(self):
        s = Scheduler(concurrency=1)
        s.submit("a")
        s.next_batch()
        assert s.release("a") == []
        assert s.release("a") == []             # 幂等
        assert s.active_count() == 0

    def test_next_batch_no_quota(self):
        s = Scheduler(concurrency=1)
        s.submit("a")
        s.next_batch()
        assert s.next_batch() == []             # 配额已满


class TestCancelAndPriority:
    def test_cancel_queued(self):
        s = Scheduler(concurrency=1)
        s.submit("a")
        s.next_batch()
        s.submit("b")
        assert s.cancel_queued("b") is True
        assert s.queue_view() == []
        assert s.cancel_queued("b") is False    # 已移除
        assert s.cancel_queued("a") is False    # 运行中不可排队取消

    def test_set_priority_only_queued(self):
        s = Scheduler(concurrency=1)
        s.submit("a", priority=0)
        s.submit("b", priority=0)
        s.next_batch()                          # a 运行
        assert s.set_priority("b", 10) is True
        assert s.set_priority("a", 10) is False  # 运行中不可改
        assert s.set_priority("nope", 1) is False
        assert [x["job_id"] for x in s.queue_view()] == ["b"]

    def test_clear(self):
        s = Scheduler(concurrency=1)
        s.submit("a")
        s.next_batch()
        s.clear()
        assert s.running() == set()
        assert s.queue_view() == []
        assert s.active_count() == 0
