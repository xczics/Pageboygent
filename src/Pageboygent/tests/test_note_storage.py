from __future__ import annotations

from pathlib import Path

import peewee
import pytest

import Pageboygent.note_storage as ns
import Pageboygent.zt_tools as zt


@pytest.fixture
def isolated_note_service(tmp_path: Path):
    test_db_path = tmp_path / "note_storage_test.db"
    test_db = peewee.SqliteDatabase(str(test_db_path))

    original_db = ns._db
    ns._db = test_db
    test_db.bind([ns.NoteRecord], bind_refs=False, bind_backrefs=False)
    test_db.connect(reuse_if_open=True)
    test_db.create_tables([ns.NoteRecord], safe=True)

    service = ns.NoteStorageService()
    try:
        yield service
    finally:
        test_db.drop_tables([ns.NoteRecord], safe=True)
        if not test_db.is_closed():
            test_db.close()
        ns._db = original_db


def test_get_note_syncs_pending_local_note_to_zotero(isolated_note_service, monkeypatch):
    service = isolated_note_service

    class FakeZot:
        # 需要有这个属性,_ensure_local_write_authorized() 靠它判断"已经授权过了",
        # 不然会尝试真的去弹Zotero的授权对话框/连真实的本地API。
        local_api_key = "fake-local-api-key"

        def create_items(self, items, parentid=None):
            assert parentid == "ITEM-001"
            assert len(items) == 1
            assert "<p>" in items[0]["note"]
            return {
                "success": ["0"],
                "successful": {"0": {"key": "NOTE-001"}},
            }

        def item(self, note_key: str):
            return {"key": note_key, "data": {"tags": []}}

        def add_tags(self, note_item, *tags):
            assert note_item["key"] == "NOTE-001"
            assert set(tags) == {"AI Notes", "Simple Notes"}

    fake_zot = FakeZot()
    # note_storage.py 里"更新已有note"那条分支还在用自己模块里的 zot;而"新建note"
    # 那条分支现在委托给了 zt_tools._create_note_in_zotero,它读的是 zt_tools 模块
    # 自己的 zot 全局变量——两边都要patch,否则新建note还是会打到真实的本地Zotero上。
    monkeypatch.setattr(ns, "zot", fake_zot)
    monkeypatch.setattr(zt, "zot", fake_zot)

    assert service.save_note(
        "ITEM-001",
        "# Summary\n\nThis is local markdown note.",
        tags=["AI Notes", "Simple Notes"],
    )

    content = service.get_note(
        "ITEM-001",
        filter_tag=["AI Notes", "Simple Notes"],
        filter_logic="and",
        sync_when_needed=True,
    )

    assert content is not None
    assert "local markdown note" in content

    record = ns.NoteRecord.get(ns.NoteRecord.item_key == "ITEM-001")
    assert record.needs_sync is False
    assert record.zotero_note_key == "NOTE-001"


def test_get_note_pulls_from_zotero_when_local_missing(isolated_note_service, monkeypatch):
    service = isolated_note_service

    monkeypatch.setattr(
        ns,
        "retreive_note_keys_from_item",
        lambda item_key, tags, filter_logic="and": ["NOTE-REMOTE-1"],
    )

    class FakeZotLocal:
        def item(self, note_key: str):
            assert note_key == "NOTE-REMOTE-1"
            return {
                "data": {
                    "note": "<h1>Remote Title</h1><p>Remote note body.</p>",
                    "tags": [{"tag": "AI Notes"}, {"tag": "Simple Notes"}],
                }
            }

    monkeypatch.setattr(ns, "zot_local", FakeZotLocal())

    content = service.get_note(
        "ITEM-002",
        filter_tag=["AI Notes", "Simple Notes"],
        filter_logic="and",
        sync_when_needed=True,
    )

    assert content is not None
    assert "Remote Title" in content
    assert "Remote note body" in content

    record = ns.NoteRecord.get(ns.NoteRecord.item_key == "ITEM-002")
    assert record.needs_sync is False
    assert record.zotero_note_key == "NOTE-REMOTE-1"
