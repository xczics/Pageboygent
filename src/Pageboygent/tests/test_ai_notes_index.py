from __future__ import annotations

from Pageboygent import ai_notes_index
from Pageboygent.lit_search import LitSearchToolbox
from Pageboygent.tests.test_lit_search import FakeBackend

DOCS = [
    {"doc_id": "1:qs", "paper_id": 1, "key": "qs", "kind": "summary", "label": "AI 快速总结", "digest": "a",
     "content": "**Quick Summary** Zinc isotopes in lunar basalts are heavy.\n\n## Methodology\nMC-ICP-MS."},
    {"doc_id": "1:theme7", "paper_id": 1, "key": "theme7", "kind": "theme", "label": "主题总结 · MVE", "digest": "b",
     "content": "Theme view: evaporative loss of zinc during the lunar magma ocean stage."},
    {"doc_id": "2:note9", "paper_id": 2, "key": "note9", "kind": "note", "label": "精读笔记 · Lunar", "digest": "c",
     "content": "Detailed note: potassium isotopes, kinetic fractionation factor beta = 0.5, vapor saturation 99%."},
]


def test_sync_is_incremental_and_search_filters_by_source(tmp_path):
    index_dir = tmp_path / "idx"
    assert ai_notes_index.sync(documents=DOCS, index_dir=index_dir) == {"added": 3, "updated": 0, "removed": 0}
    assert ai_notes_index.sync(documents=DOCS, index_dir=index_dir) == {"added": 0, "updated": 0, "removed": 0}
    changed = [dict(DOCS[0], digest="a2", content=DOCS[0]["content"] + " Rb too."), DOCS[2]]
    assert ai_notes_index.sync(documents=changed, index_dir=index_dir) == {"added": 0, "updated": 1, "removed": 1}

    ai_notes_index.sync(documents=DOCS, index_dir=index_dir)
    hits = ai_notes_index.search("zinc evaporation lunar", index_dir=index_dir)
    assert [h["key"] for h in hits][:2] == ["theme7", "qs"]  # 主题总结同时命中 3 个词，排在前面
    assert [h["kind"] for h in ai_notes_index.search("zinc", kinds=["summary"], index_dir=index_dir)] == ["summary"]
    assert [h["paper_id"] for h in ai_notes_index.search("kinetic fractionation", index_dir=index_dir)] == [2]
    assert ai_notes_index.search("zinc AND potassium", mode="keywords", index_dir=index_dir) == []


class NotesBackend(FakeBackend):
    def ai_notes(self, query, *, mode, kinds, limit):
        return [{"paper_id": 1, "key": "theme7", "kind": "theme", "label": "主题总结 · MVE",
                 "content": DOCS[1]["content"], "score": 3.0},
                {"paper_id": 1, "key": "qs", "kind": "summary", "label": "AI 快速总结",
                 "content": DOCS[0]["content"], "score": 2.0}]

    def paper_notes(self, paper_id):
        return [{"key": "qs", "kind": "summary", "label": "AI 快速总结", "content": DOCS[0]["content"]},
                {"key": "theme7", "kind": "theme", "label": "主题总结 · MVE", "content": DOCS[1]["content"]}]


def test_toolbox_labels_sources_and_shares_evidence_ids_with_read_notes():
    tb = LitSearchToolbox(backend=NotesBackend())
    out = tb.search_summaries("zinc evaporation")
    assert "命中文档：主题总结 1、AI 快速总结 1" in out and "主题总结 · MVE §1" in out
    theme_ev = next(e for e in tb.evidence.values() if e.kind == "theme")
    tb.read_paper_notes(1)
    # 检索命中的段落与 read_paper_notes 读到的是同一条证据，不会重复编号
    assert [e.eid for e in tb.evidence.values() if e.kind == "theme"] == [theme_ev.eid]
    assert "read-notes" in theme_ev.methods and "notes-bm25" in theme_ev.methods
