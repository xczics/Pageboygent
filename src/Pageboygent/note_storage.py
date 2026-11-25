from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, List, Sequence, cast
import time

import peewee
from markdown import markdown
from markdownify import markdownify

from .config import get_config
from .write_queue import checkpoint as _write_queue_checkpoint
from .zt_tools import retreive_note_keys_from_item, zot, zot_local, _create_note_in_zotero, _perform_local_write

_db_path = Path(get_config("workstation_dir")).expanduser() / "pageboygent.db"
_db = peewee.SqliteDatabase(
    str(_db_path),
    pragmas={
        "journal_mode": "wal",
        "busy_timeout": 5000,
        "foreign_keys": 1,
    },
)


class _BaseModel(peewee.Model):
    class Meta:
        database = _db


class NoteRecord(_BaseModel):
    item_key = peewee.CharField(index=True)
    tags_key = peewee.CharField(index=True)
    tags_raw = peewee.TextField(default="")
    content_md = peewee.TextField()
    zotero_note_key = peewee.CharField(null=True)
    needs_sync = peewee.BooleanField(default=True)
    updated_at = peewee.DateTimeField(default=datetime.now)
    synced_at = peewee.DateTimeField(null=True)

    class Meta:  # type: ignore[assignment]
        indexes = ((('item_key', 'tags_key'), True),)


class NoteStorageService:
    def __init__(self) -> None:
        with _db:
            _db.create_tables([NoteRecord], safe=True)

    @staticmethod
    def _normalize_tags(tags: Sequence[str] | None) -> List[str]:
        if not tags:
            return []
        clean = [tag.strip() for tag in tags if isinstance(tag, str) and tag.strip()]
        return sorted(set(clean))

    @classmethod
    def _tags_to_key(cls, tags: Sequence[str] | None) -> str:
        return ";".join(cls._normalize_tags(tags))

    @classmethod
    def _parse_tags_key(cls, tags_key: str) -> List[str]:
        if not tags_key:
            return []
        return cls._normalize_tags(tags_key.split(";"))

    @classmethod
    def _tags_match(cls, note_tags_key: str, filter_tag: Sequence[str] | None, filter_logic: str) -> bool:
        filters = cls._normalize_tags(filter_tag)
        if not filters:
            return True
        note_tags = set(cls._parse_tags_key(note_tags_key))
        if filter_logic == "and":
            return all(tag in note_tags for tag in filters)
        if filter_logic == "or":
            return any(tag in note_tags for tag in filters)
        raise ValueError(f"Invalid filter_logic: {filter_logic}. Use 'and' or 'or'.")

    @staticmethod
    def _md_to_html(content_md: str) -> str:
        return markdown(content_md, extensions=["fenced_code", "tables", "nl2br"])

    @staticmethod
    def _run_with_db_retry(func, *args, **kwargs) -> Any:
        last_exc = None
        for _ in range(5):
            try:
                return func(*args, **kwargs)
            except peewee.OperationalError as exc:
                last_exc = exc
                if "database is locked" not in str(exc).lower():
                    raise
                time.sleep(0.2)
        if last_exc is not None:
            raise last_exc

    @staticmethod
    def _is_html_content(content: str) -> bool:
        text = content.strip().lower()
        return ("<p" in text) or ("<div" in text) or ("<h1" in text) or ("<h2" in text) or ("<ul" in text) or ("<ol" in text)

    @staticmethod
    def _html_to_md(content: str) -> str:
        if NoteStorageService._is_html_content(content):
            return markdownify(content).strip()
        return content.strip()

    def ingest_note_from_zotero(
        self,
        item_key: str,
        note_key: str,
        note_content_raw: str,
        tags: Sequence[str] | None = None,
    ) -> NoteRecord:
        """Ingest one Zotero note into local storage and mark whether it needs html sync."""
        tags_key = self._tags_to_key(tags)
        tags_raw = ";".join(self._normalize_tags(tags))
        note_md = self._html_to_md(note_content_raw)
        needs_sync = not self._is_html_content(note_content_raw)
        now = datetime.now()
        defaults = {
            "content_md": note_md,
            "tags_raw": tags_raw,
            "zotero_note_key": note_key,
            "needs_sync": needs_sync,
            "updated_at": now,
            "synced_at": None if needs_sync else now,
        }
        record, created = cast(
            tuple[NoteRecord, bool],
            self._run_with_db_retry(
                NoteRecord.get_or_create,
                item_key=item_key,
                tags_key=tags_key,
                defaults=defaults,
            ),
        )
        if not created:
            # Preserve local unsynced edits if any.
            if not record.needs_sync:
                record.content_md = note_md  # type: ignore[assignment]
                record.tags_raw = tags_raw  # type: ignore[assignment]
                record.zotero_note_key = note_key  # type: ignore[assignment]
                record.needs_sync = needs_sync  # type: ignore[assignment]
                record.updated_at = now  # type: ignore[assignment]
                record.synced_at = None if needs_sync else now  # type: ignore[assignment]
                self._run_with_db_retry(record.save)
        return record

    def sync_record_to_zotero(self, record: NoteRecord) -> bool:
        """Sync one local note record to Zotero as html."""
        return self._sync_record_to_zotero(record)

    def save_note(self, item_key: str, note_content_md: str, tags: Sequence[str] | None = None) -> bool:
        tags_key = self._tags_to_key(tags)
        tags_raw = ";".join(self._normalize_tags(tags))
        now = datetime.now()
        defaults = {
            "content_md": note_content_md,
            "tags_raw": tags_raw,
            "needs_sync": True,
            "updated_at": now,
        }
        record, created = cast(
            tuple[NoteRecord, bool],
            self._run_with_db_retry(
                NoteRecord.get_or_create,
                item_key=item_key,
                tags_key=tags_key,
                defaults=defaults,
            ),
        )
        if not created:
            record.content_md = note_content_md  # type: ignore[assignment]
            record.tags_raw = tags_raw  # type: ignore[assignment]
            record.needs_sync = True  # type: ignore[assignment]
            record.updated_at = now  # type: ignore[assignment]
            self._run_with_db_retry(record.save)
        return True

    def _find_local_note(self, item_key: str, filter_tag: Sequence[str] | None, filter_logic: str) -> NoteRecord | None:
        records = (
            NoteRecord.select()
            .where(NoteRecord.item_key == item_key)
            .order_by(NoteRecord.updated_at.desc())
        )
        for record in records:
            if self._tags_match(str(record.tags_key), filter_tag, filter_logic):
                return record
        return None

    def _sync_record_to_zotero(self, record: NoteRecord) -> bool:
        note_html = self._md_to_html(str(record.content_md))
        tags = self._parse_tags_key(str(record.tags_key))
        try:
            note_key = str(record.zotero_note_key) if record.zotero_note_key else None
            if note_key:
                def _update():
                    note_item = zot.item(note_key)
                    note_item["data"]["note"] = note_html
                    note_item["data"]["tags"] = [{"tag": tag} for tag in tags]
                    return zot.update_item(note_item)
                result = _perform_local_write(f"note_storage.update_item('{note_key}')", _update)
                if not result:
                    return False
            else:
                # 建note这块逻辑跟 zt_tools.add_note_to_item 用的是同一个
                # _create_note_in_zotero,不在这里重新手写一遍(本地API没有
                # /items/new,以前这里单独调用 zot.item_template("note") 在
                # 本地模式下会直接报错——两处各写一份同样的逻辑,改的时候很容易
                # 只改一处漏了另一处,这是上次犯过的错)。
                note_key = _create_note_in_zotero(str(record.item_key), note_html, tags)
                if not note_key:
                    return False
                record.zotero_note_key = note_key  # type: ignore[assignment]
            record.needs_sync = False  # type: ignore[assignment]
            record.synced_at = datetime.now()  # type: ignore[assignment]
            record.updated_at = datetime.now()  # type: ignore[assignment]
            self._run_with_db_retry(record.save)
            return True
        except Exception as exc:
            print(f"Failed to sync note to Zotero for item {record.item_key}: {exc}")
            return False

    def _pull_notes_from_zotero(self, item_key: str, filter_tag: Sequence[str] | None, filter_logic: str) -> int:
        tags = self._normalize_tags(filter_tag)
        if not tags:
            return 0
        try:
            note_keys = retreive_note_keys_from_item(item_key, tags, filter_logic=filter_logic)
        except Exception as exc:
            print(f"Failed to pull notes from Zotero for item {item_key}: {exc}")
            return 0
        if not note_keys:
            return 0

        upserted = 0
        for note_key in note_keys:
            try:
                note_item = zot_local.item(note_key)
                note_data = note_item.get("data", {})
                note_html = str(note_data.get("note", "")).strip()
                note_md = self._html_to_md(note_html)
                note_tags = [tag["tag"] for tag in note_data.get("tags", []) if tag.get("tag")]
                tags_key = self._tags_to_key(note_tags)
                tags_raw = ";".join(self._normalize_tags(note_tags))
                now = datetime.now()
                defaults = {
                    "content_md": note_md,
                    "tags_raw": tags_raw,
                    "zotero_note_key": note_key,
                    "needs_sync": False,
                    "updated_at": now,
                    "synced_at": now,
                }
                record, created = cast(
                    tuple[NoteRecord, bool],
                    self._run_with_db_retry(
                        NoteRecord.get_or_create,
                        item_key=item_key,
                        tags_key=tags_key,
                        defaults=defaults,
                    ),
                )
                if not created and not record.needs_sync:
                    record.content_md = note_md  # type: ignore[assignment]
                    record.tags_raw = tags_raw  # type: ignore[assignment]
                    record.zotero_note_key = note_key  # type: ignore[assignment]
                    record.updated_at = now  # type: ignore[assignment]
                    record.synced_at = now  # type: ignore[assignment]
                    self._run_with_db_retry(record.save)
                upserted += 1
            except Exception as exc:
                print(f"Failed to read note {note_key} from Zotero for item {item_key}: {exc}")
        return upserted

    def get_note(
        self,
        item_key: str,
        filter_tag: Sequence[str] | None,
        filter_logic: str = "and",
        sync_when_needed: bool = True,
    ) -> str | None:
        record = self._find_local_note(item_key, filter_tag, filter_logic)
        if record:
            if sync_when_needed and record.needs_sync:
                self._sync_record_to_zotero(record)
                record = self._find_local_note(item_key, filter_tag, filter_logic)
            return str(record.content_md).strip() if record else None

        if sync_when_needed:
            self._pull_notes_from_zotero(item_key, filter_tag, filter_logic)
            record = self._find_local_note(item_key, filter_tag, filter_logic)
            if record:
                return str(record.content_md).strip()
        return None

    def has_note(
        self,
        item_key: str,
        filter_tag: Sequence[str] | None,
        filter_logic: str = "and",
        sync_when_needed: bool = True,
    ) -> bool:
        content = self.get_note(item_key, filter_tag, filter_logic=filter_logic, sync_when_needed=sync_when_needed)
        return content is not None and len(content) > 0

    def sync_pending_notes(self, limit: int | None = None) -> int:
        query = NoteRecord.select().where(NoteRecord.needs_sync == True).order_by(NoteRecord.updated_at.asc())  # noqa: E712
        if limit is not None:
            query = query.limit(limit)
        synced = 0
        for record in query:
            if self._sync_record_to_zotero(record):
                synced += 1
            _write_queue_checkpoint()
        return synced


note_storage = NoteStorageService()
