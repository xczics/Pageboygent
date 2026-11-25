"""
LLM 用量账本：记录对话 / 文献检索之外的所有模型调用（AI 快速总结、主题总结、精读笔记、综述流水线……）。

对话与文献检索的用量逐轮保存在各自的会话文件里（带逐次调用明细），这里不重复记录，
数据库看板把两者合起来展示。账本是独立的 SQLite 文件（workstation_dir/llm_usage.db），
避免与 heartbeat 写 pageboygent.db 争锁。
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import get_config

_db_path = Path(get_config("workstation_dir")).expanduser() / "llm_usage.db"
_lock = threading.Lock()

# 类别的显示顺序（看板按这个顺序列出；未登记的类别排在后面）
CATEGORIES = ["快速总结", "主题总结", "精读笔记", "子课题综述", "综述流水线"]


def _connect(path: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path or _db_path), timeout=10)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS llm_usage (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               at TEXT NOT NULL,
               category TEXT NOT NULL,
               model TEXT NOT NULL,
               prompt_tokens INTEGER NOT NULL DEFAULT 0,
               cache_hit_tokens INTEGER NOT NULL DEFAULT 0,
               completion_tokens INTEGER NOT NULL DEFAULT 0,
               ref TEXT
           )"""
    )
    return conn


def usage_numbers(usage: Any) -> dict:
    """把 OpenAI 兼容的 usage 对象转成数字（兼容 DeepSeek 的 prompt_cache_hit_tokens）。"""
    if usage is None:
        return {"prompt_tokens": 0, "cache_hit_tokens": 0, "completion_tokens": 0}
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    cached = getattr(usage, "prompt_cache_hit_tokens", None)
    if cached is None:
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) if details is not None else 0
    return {"prompt_tokens": prompt, "cache_hit_tokens": int(cached or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0)}


def record(category: str, model: str, usage: Any, ref: str | None = None, *, path: Path | None = None) -> None:
    """记一笔。usage 可以是 API 返回的 usage 对象，也可以是 usage_numbers() 的结果。记账失败不影响主流程。"""
    numbers = usage if isinstance(usage, dict) else usage_numbers(usage)
    try:
        with _lock:
            conn = _connect(path)
            with conn:
                conn.execute(
                    "INSERT INTO llm_usage (at, category, model, prompt_tokens, cache_hit_tokens, completion_tokens, ref)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (datetime.now().isoformat(timespec="seconds"), category, model,
                     numbers.get("prompt_tokens", 0), numbers.get("cache_hit_tokens", 0),
                     numbers.get("completion_tokens", 0), ref),
                )
            conn.close()
    except Exception as exc:  # pragma: no cover - 记账绝不能打断总结/综述
        print(f"[llm_usage] failed to record usage: {exc}")


def summarize(*, path: Path | None = None) -> dict:
    """按类别 × 模型汇总，并按各自模型的费率估算费用。"""
    from .lit_search import usage_breakdown

    try:
        conn = _connect(path)
        rows = conn.execute(
            "SELECT category, model, COUNT(*), SUM(prompt_tokens), SUM(cache_hit_tokens), SUM(completion_tokens),"
            " MIN(at) FROM llm_usage GROUP BY category, model"
        ).fetchall()
        conn.close()
    except Exception as exc:  # pragma: no cover
        print(f"[llm_usage] failed to read usage: {exc}")
        rows = []
    categories: dict[str, dict] = {}
    since = None
    for category, model, calls, prompt, hit, completion, first_at in rows:
        breakdown = usage_breakdown({"llm_calls": calls, "prompt_tokens": prompt or 0,
                                     "cache_hit_tokens": hit or 0, "completion_tokens": completion or 0}, model)
        entry = categories.setdefault(category, {"category": category, "calls": 0, "tokens": 0, "cost": 0.0,
                                                 "unpriced_calls": 0})
        entry["calls"] += calls
        entry["tokens"] += breakdown["total_tokens"]
        if breakdown["cost"] is None:
            entry["unpriced_calls"] += calls
        else:
            entry["cost"] += breakdown["cost"]["total"]
        since = min(since, first_at) if since else first_at
    order = {name: index for index, name in enumerate(CATEGORIES)}
    items = sorted(categories.values(), key=lambda e: (order.get(e["category"], len(order)), e["category"]))
    return {
        "categories": items,
        "tokens": sum(e["tokens"] for e in items),
        "cost": sum(e["cost"] for e in items),
        "calls": sum(e["calls"] for e in items),
        "unpriced_calls": sum(e["unpriced_calls"] for e in items),
        "since": since[:10] if since else None,
    }
