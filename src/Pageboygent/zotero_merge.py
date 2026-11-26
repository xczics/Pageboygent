"""
处理 Zotero 里“合并重复条目”之后留下的旧 key。

Zotero 合并重复条目时，保留一个主条目，其余条目移入回收站（之后可能被清空），并在主条目的
relations["dc:replaces"] 里记下被合并掉的条目地址。Pageboygent 按 key 追踪文献，于是会出现：
- 旧 key 在 Zotero 里已经查不到（读取作者、更新信息时报 404）；
- 主条目往往也已经被单独追踪，库里同一篇文献有两条记录。

reconcile() 按 dc:replaces 把旧 key 映射到主条目：
- 主条目还没被追踪：把这条记录的 key 改成主条目的 key（笔记记录一起改）；
- 主条目已被追踪：把旧记录合并进去——主记录缺的字段、主题、笔记记录从旧记录补过去
  （同一类笔记两边都有时保留主记录的），然后删掉旧记录的 PDF/MD 状态、向量与缓存，最后删掉旧记录。
不修改 Zotero 里的任何内容。
"""
from __future__ import annotations

import datetime as dt
import time
from typing import Any

RECONCILE_INTERVAL_SECONDS = 24 * 3600
_last_run = 0.0


def replaced_key_map(items: list[dict]) -> dict[str, str]:
    """旧 key → 现存主条目 key（顺着多次合并的链一直找到底）。"""
    direct: dict[str, str] = {}
    for item in items:
        relation = (item.get("data", {}).get("relations") or {}).get("dc:replaces")
        for uri in ([relation] if isinstance(relation, str) else (relation or [])):
            direct[str(uri).rstrip("/").rsplit("/", 1)[-1]] = item["key"]
    resolved = {}
    for old in direct:
        key, seen = old, set()
        while key in direct and key not in seen:
            seen.add(key)
            key = direct[key]
        resolved[old] = key
    return resolved


def _fill_missing_fields(master: Any, duplicate: Any) -> list[str]:
    filled = []
    for field in ("abstract", "doi", "year", "first_author", "quick_summary", "md_ready_date"):
        if not getattr(master, field) and getattr(duplicate, field):
            setattr(master, field, getattr(duplicate, field))
            filled.append(field)
    if filled:
        master.save()
    return filled


def _move_notes(old_key: str, new_key: str) -> dict[str, int]:
    from .note_storage import NoteRecord

    moved = dropped = 0
    for note in list(NoteRecord.select().where(NoteRecord.item_key == old_key)):
        clash = NoteRecord.get_or_none((NoteRecord.item_key == new_key) & (NoteRecord.tags_key == note.tags_key))
        if clash is None:
            note.item_key = new_key
            note.save()
            moved += 1
        else:
            note.delete_instance()   # 主条目已有同一类笔记，保留主条目的
            dropped += 1
    return {"moved": moved, "dropped": dropped}


def _drop_derived(paper_id: int) -> None:
    """删掉旧记录的派生数据：PDF/MD 状态、全文向量、文献追踪的缓存。主记录有自己的一份。"""
    from . import rag
    from .storage import MDStatus, PDFStatus, ThemePaper

    PDFStatus.delete().where(PDFStatus.paper == paper_id).execute()
    MDStatus.delete().where(MDStatus.paper == paper_id).execute()
    ThemePaper.delete().where(ThemePaper.paper == paper_id).execute()
    rag.EmbedStatus.delete().where(rag.EmbedStatus.paper_id == paper_id).execute()
    try:
        rag.DBcollection.delete(where={"paper_id": paper_id})
    except Exception as exc:
        print(f"[zotero_merge] cannot delete chunks of paper {paper_id}: {exc}")
    try:
        from . import paper_feed

        paper_feed.init_db()
        paper_feed.LibraryVector.delete().where(paper_feed.LibraryVector.pid == paper_id).execute()
        paper_feed.LibraryAuthor.delete().where(paper_feed.LibraryAuthor.pid == paper_id).execute()
    except Exception as exc:
        print(f"[zotero_merge] cannot clear feed caches of paper {paper_id}: {exc}")


def reconcile(*, dry_run: bool = False, items: list[dict] | None = None) -> list[dict]:
    """检查并修复因 Zotero 合并而失效的 key。返回每篇的处理结果。"""
    from .storage import Paper, PaperAlias, ThemePaper
    from .zt_tools import zot

    items = items if items is not None else zot.everything(zot.top())
    present = {item["key"] for item in items}
    if not present:      # 没读到任何条目（Zotero 没开或返回异常）：什么都不做
        return []
    replaced = replaced_key_map(items)
    if not dry_run:  # 重复条目在 Zotero 里被合并或删掉后，它的别名记录就没用了
        for alias in list(PaperAlias.select()):
            if alias.key not in present:
                alias.delete_instance()
    report = []
    for paper in [p for p in Paper.select() if p.key_in_zotero not in present]:
        old_key = str(paper.key_in_zotero)
        new_key = replaced.get(old_key)
        if not new_key or new_key not in present:
            report.append({"pid": paper.id, "old_key": old_key, "action": "missing", "title": paper.title})
            continue
        master = Paper.get_or_none(Paper.key_in_zotero == new_key)
        entry = {"pid": paper.id, "old_key": old_key, "new_key": new_key, "title": paper.title,
                 "action": "rekey" if master is None else "merge", "into": master.id if master else None}
        report.append(entry)
        if dry_run:
            continue
        with Paper._meta.database.atomic():
            if master is None:
                entry["notes"] = _move_notes(old_key, new_key)
                paper.key_in_zotero = new_key
                paper.save()
            else:
                entry["filled"] = _fill_missing_fields(master, paper)
                for link in list(ThemePaper.select().where(ThemePaper.paper == paper.id)):
                    if ThemePaper.get_or_none((ThemePaper.paper == master.id) & (ThemePaper.theme == link.theme_id)) is None:
                        link.paper = master.id
                        link.save()
                entry["notes"] = _move_notes(old_key, new_key)
                PaperAlias.update(paper=master.id).where(PaperAlias.paper == paper.id).execute()
                _drop_derived(paper.id)
                paper.delete_instance()
        print(f"[zotero_merge] {entry['action']} paper {paper.id} {old_key} -> {new_key}"
              + (f" (into {master.id})" if master else ""))
    return report


def maybe_reconcile() -> None:
    """heart_beat 里调用：每天检查一次。"""
    global _last_run
    if time.time() - _last_run < RECONCILE_INTERVAL_SECONDS:
        return
    _last_run = time.time()
    try:
        reconcile()
    except Exception as exc:
        print(f"[zotero_merge] reconcile failed at {dt.datetime.now():%H:%M}: {exc}")
