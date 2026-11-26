import peewee
import pytest

from Pageboygent import storage
from Pageboygent.storage import MDStatus, Paper, PaperAlias, PDFStatus, Theme, ThemePaper, normalize_doi


def test_normalize_doi():
    assert normalize_doi("https://doi.org/10.1016/J.GCA.2020.01.001") == "10.1016/j.gca.2020.01.001"
    assert normalize_doi(" doi:10.1029/2020JE006 ") == "10.1029/2020je006"
    assert normalize_doi("") is None and normalize_doi(None) is None and normalize_doi("   ") is None
    assert normalize_doi("not a doi") is None


@pytest.fixture
def mem_db(monkeypatch):
    db = peewee.SqliteDatabase(":memory:", pragmas={"foreign_keys": 1})
    models = [Paper, PDFStatus, MDStatus, Theme, ThemePaper, PaperAlias]
    with db.bind_ctx(models):
        db.create_tables(models)
        zotero = {}
        monkeypatch.setattr(storage, "check_literature_exists", lambda key, *a, **k: key in zotero)
        monkeypatch.setattr(storage, "get_info", lambda key, fields: zotero[key])
        monkeypatch.setattr(Paper, "update_from_zotero", lambda self: None)
        yield zotero


def info(title, doi):
    return (title, "2020-01-01", [{"creatorType": "author", "lastName": "Smith", "firstName": "A"}], "", doi)


def test_same_doi_in_two_zotero_items_is_tracked_once(mem_db):
    mem_db["AAA"] = info("Zinc isotopes", "10.1/ABC")
    mem_db["BBB"] = info("Zinc isotopes (dup)", "https://doi.org/10.1/abc")
    first = Paper.trace_paper("AAA")
    second = Paper.trace_paper("BBB")
    assert second.id == first.id and Paper.select().count() == 1
    assert PaperAlias.get(PaperAlias.key == "BBB").paper.id == first.id
    assert Paper.trace_paper("BBB").id == first.id         # 之后直接按别名找到，不再查 Zotero


def test_missing_doi_is_never_a_duplicate(mem_db):
    mem_db["AAA"] = info("Paper one", None)
    mem_db["BBB"] = info("Paper two", "")
    mem_db["CCC"] = info("Paper three", "   ")
    ids = {Paper.trace_paper(k).id for k in ("AAA", "BBB", "CCC")}
    assert len(ids) == 3 and PaperAlias.select().count() == 0


def test_same_doi_after_old_item_left_zotero_takes_new_key(mem_db):
    mem_db["AAA"] = info("Zinc isotopes", "10.1/abc")
    paper = Paper.trace_paper("AAA")
    del mem_db["AAA"]                     # 旧条目被删、又重新导入成新条目
    mem_db["NEW"] = info("Zinc isotopes", "10.1/abc")
    again = Paper.trace_paper("NEW")
    assert again.id == paper.id and Paper.get_by_id(paper.id).key_in_zotero == "NEW"
    assert PaperAlias.select().count() == 0
