"""Web 端写操作队列。

Web 请求线程不直接写 pageboygent.db，而是把写操作封装成任务提交到这里，
统一由 heartbeat 线程执行。这样所有写入都落在 heartbeat 线程的同一个连接上，
不会再出现跨连接的 WAL 读快照过期（SQLITE_BUSY_SNAPSHOT / "database is locked"）。

任务分两类：
- quick（增/改/删等快速写入）：heartbeat 空闲时执行；heart_beat 运行中也会在
  checkpoint() 处“插队”执行；进程关闭前会处理完。
- slow（Trace、Update Theme Papers 等耗时任务）：只在 heartbeat 空闲时执行，
  关闭时直接丢弃。

队列只在进程内存中。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Callable


@dataclass
class WriteTask:
    description: str
    func: Callable[[], object]
    quick: bool = True
    submitted_at: float = 0.0


_cond = threading.Condition()
_quick: deque[WriteTask] = deque()
_slow: deque[WriteTask] = deque()
_consumer_thread: threading.Thread | None = None
_in_task = False  # 只由消费线程读写，防止任务内部再触发 checkpoint 嵌套执行


def submit(description: str, func: Callable[[], object], *, quick: bool = True) -> int:
    """提交一个写任务，返回提交后队列中的待处理任务数。"""
    task = WriteTask(description, func, quick, time.time())
    with _cond:
        (_quick if quick else _slow).append(task)
        _cond.notify_all()
    return pending_count()


def pending_count() -> int:
    with _cond:
        return len(_quick) + len(_slow)


def set_consumer_thread() -> None:
    """由 heartbeat 线程调用，声明自己是唯一的任务执行线程。"""
    global _consumer_thread
    _consumer_thread = threading.current_thread()


def wake() -> None:
    """唤醒正在 process_until 中等待的 heartbeat 线程（关闭时调用）。"""
    with _cond:
        _cond.notify_all()


def _pop(include_slow: bool) -> WriteTask | None:
    with _cond:
        if _quick:
            return _quick.popleft()
        if include_slow and _slow:
            return _slow.popleft()
        return None


def _run(task: WriteTask) -> None:
    global _in_task
    waited = time.time() - task.submitted_at
    _in_task = True
    try:
        task.func()
        print(f"[write_queue] done: {task.description} (queued {waited:.1f}s)")
    except Exception as exc:
        print(f"[write_queue] failed: {task.description}: {exc}")
    finally:
        _in_task = False


def checkpoint() -> None:
    """heart_beat 内部的安全点：让排队中的 quick 任务插队执行。

    仅在 heartbeat 线程上、且不在执行任务时生效；调用方需保证此处没有未提交的显式事务。
    """
    if _in_task or threading.current_thread() is not _consumer_thread:
        return
    while (task := _pop(include_slow=False)) is not None:
        _run(task)


def process_until(deadline: float, stop_event: threading.Event) -> None:
    """heartbeat 空闲期：在 deadline（time.monotonic）之前持续处理任务，无任务时阻塞等待。"""
    while not stop_event.is_set():
        timeout = deadline - time.monotonic()
        if timeout <= 0:
            return
        with _cond:
            _cond.wait_for(lambda: stop_event.is_set() or bool(_quick) or bool(_slow), timeout)
        task = _pop(include_slow=True)
        if task is not None:
            _run(task)


def drain(*, shutting_down: bool = False) -> None:
    """非阻塞地处理完当前队列中的任务；关闭时丢弃 slow 任务。"""
    while (task := _pop(include_slow=not shutting_down)) is not None:
        _run(task)
    if shutting_down:
        with _cond:
            dropped = list(_slow)
            _slow.clear()
        for task in dropped:
            print(f"[write_queue] dropped on shutdown: {task.description}")
