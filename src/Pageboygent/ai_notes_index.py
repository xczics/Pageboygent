"""
AI 总结与笔记的 BM25 索引：快速总结 + 主题总结 + 精读笔记（及其他 AI 笔记），统一检索。

原来"检索 AI 总结"只索引快速总结，主题总结和精读笔记只能在已知文献编号时用 read_paper_notes 读到，
检索阶段几乎用不上。这里把它们放进同一个 Whoosh 索引，每篇文档带来源类别，按内容摘要增量同步。

文档 key 与 lit_search.SearchBackend.paper_notes 保持一致（qs / theme<id> / note<id>），
这样检索命中的段落与 read_paper_notes 读到的段落是同一条证据（同一个 E#）。
"""
from __future__ import annotations

import hashlib
import re
import threading
import time
from pathlib import Path
from typing import Iterator

from .config import get_config

_index_dir = Path(get_config("workstation_dir")).expanduser() / "ai_notes_index"
_lock = threading.RLock()
_ix = None
_last_sync = 0.0
_SYNC_INTERVAL_SECONDS = 60

KINDS = ("summary", "theme", "note")
KIND_LABELS = {"summary": "AI 快速总结", "theme": "主题总结", "note": "笔记"}


def _schema():
    from whoosh.analysis import StemmingAnalyzer
    from whoosh.fields import ID, NUMERIC, STORED, TEXT, Schema

    return Schema(
        doc_id=ID(stored=True, unique=True),   # "<paper_id>:<key>"
        paper_id=NUMERIC(stored=True),
        key=STORED,                            # qs / theme<id> / note<id>
        kind=ID(stored=True),                  # summary / theme / note
        label=STORED,
        digest=STORED,
        content=TEXT(analyzer=StemmingAnalyzer(), stored=True),
    )


def _get_index(index_dir: Path | None = None):
    global _ix
    from whoosh.index import create_in, exists_in, open_dir

    directory = index_dir or _index_dir
    if index_dir is not None:  # 测试用的临时目录：不缓存
        directory.mkdir(parents=True, exist_ok=True)
        return open_dir(str(directory)) if exists_in(str(directory)) else create_in(str(directory), _schema())
    if _ix is None:
        directory.mkdir(parents=True, exist_ok=True)
        _ix = open_dir(str(directory)) if exists_in(str(directory)) else create_in(str(directory), _schema())
    return _ix


def _note_label(tags: list[str], ai_tag: str, detailed_prefix: str) -> str:
    detailed = [t for t in tags if t.startswith(detailed_prefix + "/")]
    if detailed:
        return "精读笔记 · " + detailed[0][len(detailed_prefix) + 1:]
    rest = [t for t in tags if t != ai_tag]
    return "笔记" + (" · " + "; ".join(rest) if rest else "")


def iter_documents() -> Iterator[dict]:
    """全库的 AI 总结与笔记，去掉重复副本（Zotero 里的快速总结副本、按主题标签存的主题总结副本、内容完全相同的笔记）。"""
    from .note_storage import NoteRecord
    from .storage import Paper, Theme, ThemePaper

    ai_tag = get_config("tag_for_ai_notes")
    summary_tag = get_config("tag_for_quick_summary")
    detailed_prefix = get_config("tag_for_detailed_reading")
    seen: set[tuple[int, str]] = set()

    def _doc(pid: int, key: str, kind: str, label: str, content: str | None) -> dict | None:
        text = (content or "").strip()
        if not text:
            return None
        digest = hashlib.md5(re.sub(r"\s+", " ", text).encode()).hexdigest()
        if (pid, digest) in seen:
            return None
        seen.add((pid, digest))
        return {"doc_id": f"{pid}:{key}", "paper_id": pid, "key": key, "kind": kind, "label": label,
                "digest": digest, "content": text}

    key_to_pid: dict[str, int] = {}
    for paper in Paper.select(Paper.id, Paper.key_in_zotero, Paper.quick_summary):
        key_to_pid[str(paper.key_in_zotero)] = int(paper.id)
        if (doc := _doc(int(paper.id), "qs", "summary", "AI 快速总结", paper.quick_summary)):
            yield doc
    theme_tags: set[str] = {summary_tag}
    for tp in ThemePaper.select(ThemePaper, Theme).join(Theme):
        if tp.theme_specific_summary:
            theme_tags.add(str(tp.theme.note_tag))
            if (doc := _doc(int(tp.paper_id), f"theme{tp.id}", "theme", f"主题总结 · {tp.theme.name}",
                            tp.theme_specific_summary)):
                yield doc
    for record in NoteRecord.select():
        pid = key_to_pid.get(str(record.item_key))
        tags = [t for t in str(record.tags_key).split(";") if t]
        if pid is None or theme_tags & set(tags):
            continue  # 不在库中的条目，或快速总结/主题总结在 Zotero 里的副本
        if (doc := _doc(pid, f"note{record.id}", "note", _note_label(tags, ai_tag, detailed_prefix),
                        str(record.content_md))):
            yield doc


def sync(force: bool = False, *, documents: list[dict] | None = None, index_dir: Path | None = None) -> dict:
    """按内容摘要增量同步：新增/变化的文档写入，已删除的移除。返回 {added, updated, removed}。"""
    global _last_sync
    with _lock:
        if not force and documents is None and time.monotonic() - _last_sync < _SYNC_INTERVAL_SECONDS:
            return {"added": 0, "updated": 0, "removed": 0}
        docs = {d["doc_id"]: d for d in (documents if documents is not None else iter_documents())}
        ix = _get_index(index_dir)
        with ix.searcher() as searcher:
            existing = {fields["doc_id"]: fields.get("digest") for _, fields in searcher.reader().iter_docs()}
        removed = [doc_id for doc_id in existing if doc_id not in docs]
        changed = [d for doc_id, d in docs.items() if existing.get(doc_id) != d["digest"]]
        if removed or changed:
            writer = ix.writer()
            try:
                for doc_id in removed:
                    writer.delete_by_term("doc_id", doc_id)
                for doc in changed:
                    writer.update_document(**doc)
                writer.commit()
            except BaseException:
                writer.cancel()
                raise
        if index_dir is None:
            _last_sync = time.monotonic()
        updated = sum(1 for d in changed if d["doc_id"] in existing)
        return {"added": len(changed) - updated, "updated": updated, "removed": len(removed)}


def search(query: str, *, mode: str = "bm25", kinds: list[str] | None = None, limit: int = 20,
           index_dir: Path | None = None) -> list[dict]:
    """检索 AI 总结与笔记。mode=bm25：词间 OR，按相关度排序；mode=keywords：布尔匹配，词间默认 AND。"""
    from whoosh.qparser import AndGroup, OrGroup, QueryParser
    from whoosh.query import Or, Term

    if not (query or "").strip():
        return []
    if index_dir is None:
        sync()
    ix = _get_index(index_dir)
    group = AndGroup if mode == "keywords" else OrGroup.factory(0.9)
    parsed = QueryParser("content", ix.schema, group=group).parse(query)
    kinds = [k for k in (kinds or KINDS) if k in KINDS] or list(KINDS)
    restrict = Or([Term("kind", k) for k in kinds]) if len(kinds) < len(KINDS) else None
    with ix.searcher() as searcher:
        return [{"paper_id": hit["paper_id"], "key": hit["key"], "kind": hit["kind"], "label": hit["label"],
                 "content": hit["content"], "score": float(hit.score)}
                for hit in searcher.search(parsed, limit=limit, filter=restrict)]
