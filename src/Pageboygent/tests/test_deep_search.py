import numpy as np
import pytest

from Pageboygent import deep_search as ds


def test_parse_query_syntax():
    q = ds.parse_query('zinc "late accretion" -Mars author:"James M. D. Day" year:2015-2024')
    assert q.terms == ["zinc"] and q.phrases == ["late accretion"] and q.excluded == ["Mars"]
    assert q.author == "James M. D. Day" and (q.year_from, q.year_to) == (2015, 2024)
    assert q.whoosh() == 'zinc "late accretion"'               # 排除词不交给 Whoosh
    assert ds.parse_query("https://doi.org/10.1016/J.EPSL.2026.120451").doi == "10.1016/j.epsl.2026.120451"
    assert (ds.parse_query("year:>2018").year_from, ds.parse_query("year:<2010").year_to) == (2019, 2009)
    assert ds.parse_query("isotope isotopes").stems == ["isotop"]            # 词干相同只算一个


def test_rrf_fusion_rewards_agreement_across_channels():
    fused = ds.fuse({
        "fulltext_bm25": [{"pid": 1}, {"pid": 2}, {"pid": 2}],
        "fulltext_semantic": [{"pid": 2}, {"pid": 3}],
        "meta": [{"pid": 3}],
    })
    assert fused[2]["ranks"] == {"fulltext_bm25": 2, "fulltext_semantic": 1}
    # 两个渠道都靠前的，排在只在一个渠道第一的前面
    assert fused[2]["score"] > fused[1]["score"]


def test_author_matching_rules():
    names = ["James M.D. Day", "Holliday, A."]
    assert ds.author_matches("Day", names) and not ds.author_matches("Holli", names)
    assert ds.author_matches("James M. D. Day", ["James M.D. Day"])      # 同一个人的不同写法
    assert not ds.author_matches("Day", ["Holliday, A."])                # 只写姓：要求姓完全一致


def test_highlight_marks_stems_and_keeps_math():
    q = ds.parse_query("evaporation zinc")
    out = ds.highlight("Zinc isotopes record evaporative loss; $\\delta^{66}$Zn evaporation shifts.", q)
    assert "<b>Zinc</b>" in out and "<b>evaporation</b>" in out and "<b>evaporative</b>" in out
    assert "$\\delta^{66}$" in out                                       # 公式里不插标记


class FakeBackend(ds.Backend):
    papers_ = [
        {"pid": 1, "key": "A", "title": "Zinc isotope evaporation on the Moon", "year": 2012, "first_author": "Paniello",
         "doi": "10.1/a", "abstract": "Zinc evaporation.", "summary": "", "authors": ["Randal C. Paniello", "James M. D. Day"]},
        {"pid": 2, "key": "B", "title": "Martian magma ocean", "year": 2020, "first_author": "Smith",
         "doi": "10.1/b", "abstract": "Mars mantle.", "summary": "", "authors": ["A. Smith"]},
        {"pid": 3, "key": "C", "title": "Volatile loss in planetesimals", "year": 2018, "first_author": "Day",
         "doi": "", "abstract": "", "summary": "", "authors": ["James M.D. Day"]},
    ]

    def papers(self):
        return self.papers_

    def embed(self, text):
        return np.array([1.0, 0.0], dtype=np.float32)

    def fulltext_bm25(self, query, limit):
        return [{"paper_id": 3, "text": "Zinc evaporation during accretion of planetesimals.", "line_number": 12, "score": 9.0},
                {"paper_id": 2, "text": "On Mars, zinc is depleted.", "line_number": 4, "score": 3.0}]

    def fulltext_semantic(self, vec, limit):
        return [{"paper_id": 1, "text": "Moon zinc.", "line_number": 1, "score": 0.8}]

    def notes(self, query, limit):
        return [{"paper_id": 1, "kind": "summary", "label": "", "content": "Zinc evaporation on the Moon.", "score": 5.0}]

    def paper_vectors(self):
        return [1, 2, 3], np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype=np.float32)

    def groups(self):
        return []

    def stats(self):
        return {"chunks": 3}


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(ds, "backend", FakeBackend())
    ds._papers_cache.clear()
    ds._result_cache.clear()
    yield
    ds._papers_cache.clear()
    ds._result_cache.clear()


def test_search_end_to_end_with_filters(fake):
    r = ds.search("zinc evaporation")
    assert [x["pid"] for x in r["results"]][:2] == [1, 3]       # 多渠道都靠前的排第一
    top = r["results"][0]
    assert top["snippet"] and "<b>" in top["snippet"]["html"] and top["zotero_url"].endswith("/A")
    assert {c["key"] for c in top["channels"]} >= {"meta", "notes", "fulltext_semantic"}
    assert [x["pid"] for x in ds.search("zinc -Mars")["results"]].count(2) == 0              # 排除词
    assert [x["pid"] for x in ds.search("zinc author:Day")["results"]] == [1, 3]             # 两种写法都命中
    assert [x["pid"] for x in ds.search("zinc year:2015-2020")["results"]] == [3, 2]
    assert [x["pid"] for x in ds.search("doi:10.1/b")["results"]] == [2]
    assert [x["pid"] for x in ds.search("author:Day")["results"]] == [3, 1]                  # 只有过滤条件：按年份列出
    assert ds.search("zinc", scope="title")["channels"] == [{"key": "meta", "label": "标题与作者"}]


def test_suggest_groups_author_variants(fake):
    items = ds.suggest("day")
    assert items[0]["type"] == "author" and items[0]["count"] == 2


def test_web_merge_dedupes_across_sources_and_ranks():
    from Pageboygent import web_search as ws

    rec = lambda source, rank, title, doi=None, abstract="": ws._record(source, rank, title=title, authors=["A. B"], year=2020,  # noqa: E731
                                                                       journal="J", doi=doi, abstract=abstract, url=None)
    merged = ws.merge({
        "openalex": [rec("openalex", 1, "Zinc isotopes on the Moon", "10.1/x"), rec("openalex", 2, "Other paper")],
        "crossref": [rec("crossref", 1, "Zinc Isotopes on the Moon.", "https://doi.org/10.1/X", "A long abstract.")],
    })
    assert len(merged) == 2 and merged[0]["ranks"] == {"openalex": 1, "crossref": 1}
    assert merged[0]["abstract"] == "A long abstract."      # 合并时补上缺的摘要


def test_web_scope_marks_library_papers_and_survives_source_errors(fake, monkeypatch):
    from Pageboygent import paper_feed, web_search as ws

    def openalex(query, **kw):
        return [ws._record("openalex", 1, title="Zinc isotope evaporation on the Moon", authors=["R. Paniello"], year=2012,
                           journal="Nature", doi="10.1/a", abstract="Zinc evaporation.", url="https://doi.org/10.1/a"),
                ws._record("openalex", 2, title="New Martian zinc study", authors=["X. Y"], year=2025, journal="Icarus",
                           doi="10.9/new", abstract="Mars zinc evaporation.", url="https://doi.org/10.9/new")]

    def broken(query, **kw):
        raise RuntimeError("down")

    def ads(query, **kw):
        raise LookupError("not-configured")

    monkeypatch.setattr(ws, "SEARCHERS", {"openalex": openalex, "crossref": broken, "ads": ads})
    monkeypatch.setattr(paper_feed.backend, "library_doi_map", lambda: {"10.1/a": 1})
    ws._cache.clear()
    r = ds.search("zinc evaporation", scope="web")
    assert r["total"] == 2 and r["results"][0]["in_library"] and r["results"][0]["zotero_url"].endswith("/A")
    new = r["results"][1]
    assert not new["in_library"] and new["record"]["doi"] == "10.9/new" and new["url"] == "https://doi.org/10.9/new"
    states = {s["key"]: s["state"] for s in r["web"]["status"]}
    assert states["openalex"] == "ok" and states["ads"] == "not-configured" and states["crossref"].startswith("error")
    ws._cache.clear()
    assert [x["title"] for x in ds.search("zinc evaporation -Mars", scope="web")["results"]] == ["Zinc isotope evaporation on the Moon"]
