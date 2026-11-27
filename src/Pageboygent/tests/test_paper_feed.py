from __future__ import annotations

import datetime as dt
import hashlib
import json
from types import SimpleNamespace

import feedparser
import numpy as np
import pytest

from Pageboygent import feed_ranker, llm_usage, paper_feed, settings
from Pageboygent.feed_ranker import Facet, RankerBackend, author_key

TODAY = dt.datetime.now()
RSS = f"""<?xml version="1.0"?>
<rss version="2.0" xmlns:prism="http://prismstandard.org/namespaces/basic/2.0/"><channel><title>Icarus</title>
<item><title>Zinc isotope evaporation on the Moon</title><link>https://doi.org/10.1016/j.icarus.2026.1</link>
<guid>a1</guid><pubDate>{TODAY:%a, %d %b %Y} 10:00:00 GMT</pubDate><author>Sean N. Raymond</author>
<description>&lt;p&gt;{"We measured zinc isotopes in lunar basalts and found evaporative loss during the magma ocean stage. " * 4}&lt;/p&gt;</description></item>
<item><title>Bird migration in Europe</title><guid>a2</guid><pubDate>{TODAY:%a, %d %b %Y} 10:00:00 GMT</pubDate>
<description>Short.</description></item>
<item><title>Zinc isotope evaporation on the Moon</title><guid>dup</guid><link>https://doi.org/10.1016/j.icarus.2026.1</link></item>
<item><title>Ancient paper</title><guid>old</guid><pubDate>Mon, 01 Jan 2001 10:00:00 GMT</pubDate></item>
</channel></rss>"""

VOCAB = ["zinc", "isotope", "moon", "lunar", "evaporation", "bird", "migration", "planet"]


def fake_vec(text: str) -> np.ndarray:
    words = text.lower()
    vec = np.array([words.count(w) for w in VOCAB], dtype=np.float32) + 0.01
    return vec / np.linalg.norm(vec)


class FakeRanker(RankerBackend):
    def embed(self, text):
        return fake_vec(text)

    def library_papers(self):
        return [{"pid": i, "key": f"K{i}", "title": t, "first_author": "Raymond", "year": 2020, "abstract": t, "summary": ""}
                for i, t in enumerate(["zinc isotope lunar evaporation", "moon zinc isotope", "planet lunar moon",
                                       "zinc evaporation isotope moon", "lunar isotope zinc", "planet formation"], start=1)]

    def library_facets(self, pids, vectors):
        return [Facet(id="cluster:1", name="MVE", source="cluster", description="volatile loss", pids=[1, 2, 4, 5])]

    def library_authors(self, papers):
        return {p["pid"]: ["Sean N. Raymond"] for p in papers}

    def lexical_score(self, text):
        return 15.0 if "zinc" in text.lower() else 0.0

    def vector_cache_get(self, pid, digest, kind="full"):
        return None

    def vector_cache_put(self, pid, digest, vec, kind="full"):
        pass


class FakeClient:
    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        user = kwargs["messages"][-1]["content"]
        if "items" in kwargs["messages"][0]["content"]:  # 判断
            ids = [line[1:line.index("]")] for line in user.split("\n") if line.startswith("[F")]
            content = json.dumps({"items": [{"id": i, "score": 9, "reason": "与 MVE 方向直接相关"} for i in ids]})
        else:
            content = "本周 [F1] 很重要，另见 [F999]。"
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20, prompt_cache_hit_tokens=0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=usage)


class FakeFeedBackend(paper_feed.FeedBackend):
    def __init__(self):
        self.client = FakeClient()
        self.zotero = []
        self.tracked = []

    def parse_feed(self, url):
        return feedparser.parse(RSS)

    def enrich(self, doi):
        return {"abstract": "", "source": "none", "authors": ["Sean N. Raymond", "Alice Smith"], "journal": "Icarus"}

    title_hits: dict = {}

    def enrich_by_title(self, title):
        return self.title_hits.get(title)

    def llm_client(self):
        return self.client

    def library_doi_map(self):
        return {}

    def zotero_collection(self, root, child):
        return f"{root}/{child}"

    def zotero_add_item(self, collection_key, item):
        self.zotero.append((collection_key, item["title"]))
        self.added.append(item)
        return "ZKEY0001"

    def track_in_pageboygent(self, zotero_key, root):
        self.tracked.append((zotero_key, root))

    added: list = []
    patched: list = []

    def zotero_patch_item(self, key, fields):
        self.patched.append((key, fields))
        return True


@pytest.fixture
def feed_env(tmp_path, monkeypatch):
    monkeypatch.setattr(paper_feed, "_initialized_path", None)
    monkeypatch.setattr(paper_feed, "_data_dir", tmp_path / "pf")
    monkeypatch.setattr(paper_feed, "_model_cache", None)
    monkeypatch.setattr(paper_feed, "ranker_backend", FakeRanker())
    fb = FakeFeedBackend()
    monkeypatch.setattr(paper_feed, "backend", fb)
    monkeypatch.setattr(llm_usage, "_db_path", tmp_path / "usage.db")
    paper_feed.init_db(tmp_path / "feed.db")
    yield fb
    paper_feed._initialized_path = None


def test_author_key_normalizes_name_orders():
    assert author_key("Raymond, Sean N.") == author_key("Sean N. Raymond") == "raymond s"
    assert author_key("Albarède, F.") == "albarede f"


def test_fetch_dedupes_skips_old_and_enriches(feed_env):
    feed = paper_feed.add_feed("Icarus", "https://example.org/rss")
    assert paper_feed.fetch_feed(feed, {}) == 2          # 重复 DOI 与超过保留期的条目被跳过
    items = {i.title: i for i in paper_feed.Item.select()}
    zinc = items["Zinc isotope evaporation on the Moon"]
    assert zinc.doi == "10.1016/j.icarus.2026.1" and zinc.abstract_source == "rss" and "<p>" in zinc.raw_html
    assert items["Bird migration in Europe"].abstract is None   # 太短，不算摘要
    paper_feed.enrich_items()
    assert json.loads(paper_feed.Item.get_by_id(zinc.id).authors) == ["Sean N. Raymond", "Alice Smith"]


def test_scoring_bands_auto_ingest_and_no_self_reinforcement(feed_env):
    feed = paper_feed.add_feed("Icarus", "https://example.org/rss")
    paper_feed.fetch_feed(feed, {})
    paper_feed.score_items()
    zinc = paper_feed.Item.get(paper_feed.Item.title.contains("Zinc"))
    bird = paper_feed.Item.get(paper_feed.Item.title.contains("Bird"))
    features = json.loads(zinc.features)
    assert features["facet_name"] == "MVE" and features["author_match"] == "raymond s"
    assert zinc.llm_score == 9 and zinc.band == "auto"
    assert bird.llm_score is None and bird.band == "low"        # 基础分低，不送 LLM
    assert zinc.info == "abstract" and bird.info == "title"
    # 只有标题的文章也能自动入库，但需要至少一项强证据（标题与库很近，或作者常见）
    assert feed_ranker.assign_band(0.9, info="title", llm_score=9, features={"dense": 0.7, "author": 0}) == "auto"
    assert feed_ranker.assign_band(0.9, info="title", llm_score=9, features={"dense": 0.2, "author": 0.8}) == "auto"
    assert feed_ranker.assign_band(0.9, info="title", llm_score=9, features={"dense": 0.2, "author": 0.1}) == "confirm"
    assert feed_ranker.assign_band(0.9, info="abstract", llm_score=9, features={}) == "auto"
    assert paper_feed.auto_ingest() == 1
    zinc = paper_feed.Item.get_by_id(zinc.id)
    assert zinc.status == "ingested" and zinc.ingest_mode == "auto" and zinc.feedback is None  # 自动入库不算正例
    assert feed_env.zotero == [("RSS_TRACK/Icarus", zinc.title)] and feed_env.tracked == [("ZKEY0001", "RSS_TRACK")]
    assert llm_usage.summarize()["categories"][0]["category"] == "文献追踪"


def test_feedback_changes_future_scores(feed_env):
    feed = paper_feed.add_feed("Icarus", "https://example.org/rss")
    paper_feed.fetch_feed(feed, {})
    paper_feed.score_items()
    bird = paper_feed.Item.get(paper_feed.Item.title.contains("Bird"))
    paper_feed.set_feedback(bird.id, 1)
    paper_feed.Item.create(feed=feed, guid="b2", dedupe_key="title:birds2", title="Bird migration patterns", authors="[]")
    paper_feed.score_items()
    new = paper_feed.Item.get(paper_feed.Item.guid == "b2")
    assert json.loads(new.features)["feedback"] > 0      # 与你 👍 过的文章相近 → 反馈项加分


def test_prune_keeps_rated_and_ingested(feed_env):
    feed = paper_feed.add_feed("Icarus", "https://example.org/rss")
    old = dt.datetime.now() - dt.timedelta(days=200)
    plain = paper_feed.Item.create(feed=feed, guid="1", dedupe_key="t:1", title="old plain", created_at=old, raw_html="<p>x</p>")
    rated = paper_feed.Item.create(feed=feed, guid="2", dedupe_key="t:2", title="old rated", created_at=old, raw_html="<p>x</p>", feedback=-1)
    assert paper_feed.prune_old_items() == {"deleted": 1, "slimmed": 1}
    assert paper_feed.Item.get_or_none(paper_feed.Item.id == plain.id) is None
    assert paper_feed.Item.get_by_id(rated.id).raw_html is None


def test_weekly_report_only_cites_listed_items(feed_env):
    feed = paper_feed.add_feed("Icarus", "https://example.org/rss")
    paper_feed.fetch_feed(feed, {})
    paper_feed.score_items()
    paper_feed.auto_ingest()
    week = paper_feed.iso_week(dt.datetime.now())
    report = paper_feed.generate_report(week)
    data = json.loads(report.data)
    assert data["stats"]["items"] == 2 and data["stats"]["auto"] == 1
    assert "[F999]" not in report.overview
    assert report.read_at is None


def test_add_feed_validation(feed_env):
    with pytest.raises(ValueError):
        paper_feed.add_feed("x", "ftp://bad")
    paper_feed.add_feed("A", "https://example.org/a")
    with pytest.raises(ValueError):
        paper_feed.add_feed("B", "https://example.org/a")
    with pytest.raises(ValueError):
        paper_feed.add_feed("A/B", "https://example.org/b")


# --------------------------- 设置 ---------------------------

def test_settings_save_roundtrip_keeps_secrets_and_profile_layout(tmp_path):
    import tomllib

    path = tmp_path / "Config.toml"
    path.write_text('ds_url = "https://api.deepseek.com"\n\n[profiles.default]\nds_api_key = "sk-old"\nzotero_id = "123"\n')
    result = settings.save({"ds_api_key": "", "zotero_id": "456", "feed_retention_days": "30",
                            "mineru_upload_accelerate": False, "embedding_dim": "abc"}, path=path)
    assert result["errors"] == {"embedding_dim": "格式不正确"}
    data = tomllib.loads(path.read_text())
    assert data["profiles"]["default"]["ds_api_key"] == "sk-old"   # 密钥留空 = 不修改
    assert data["profiles"]["default"]["zotero_id"] == "456"       # 原本在 profile 段的仍写回 profile 段
    assert data["feed_retention_days"] == 30 and data["mineru_upload_accelerate"] is False
    assert oct(path.stat().st_mode)[-3:] == "600"
    assert list(tmp_path.glob("Config.toml.bak-*"))
    settings.save({}, clear=["feed_retention_days"], path=path)
    assert "feed_retention_days" not in tomllib.loads(path.read_text())


def test_settings_describe_never_exposes_secret_values():
    described = settings.describe()
    rows = {r["name"]: r for g in described["groups"] for r in g["items"]}
    assert rows["ds_api_key"]["secret"] and rows["ds_api_key"]["required"]
    from Pageboygent.config import get_config

    real = get_config("ds_api_key")
    if real:
        assert real not in json.dumps(described)


def test_title_only_items_compare_against_library_titles(feed_env):
    model, rb, _ = paper_feed.library_model(describe=False)
    vec = fake_vec("zinc moon")
    as_title = feed_ranker.compute_features(model, rb, vec=vec, text="zinc moon", authors=[], positives=[], negatives=[],
                                            title_only=True)
    as_full = feed_ranker.compute_features(model, rb, vec=vec, text="zinc moon", authors=[], positives=[], negatives=[])
    assert as_title["info"] == "title" and as_full["info"] == "abstract"
    # 标题对标题比较、按库内标题自己的锚点换算：只有标题的文章不再被系统性压分
    assert as_title["dense"] >= as_full["dense"] - 0.05
    assert model.anchors["title"] != model.anchors["abstract"]


def test_unrelated_negative_feedback_does_not_penalize(feed_env):
    """只是“随便一篇”那样像被 👎 的文章：不扣分；很像才扣分。"""
    model, rb, _ = paper_feed.library_model(describe=False)
    vec = fake_vec("zinc isotope lunar evaporation")
    far = feed_ranker.compute_features(model, rb, vec=vec, text="", authors=[], positives=[],
                                       negatives=[fake_vec("bird migration")])
    near = feed_ranker.compute_features(model, rb, vec=vec, text="", authors=[], positives=[],
                                        negatives=[fake_vec("zinc isotope lunar evaporation moon")])
    assert far["feedback"] == 0.0 and near["feedback"] < -0.5


def test_enrichment_by_title_and_retry_until_abstract_arrives(feed_env):
    feed = paper_feed.add_feed("GCA", "https://example.org/gca")
    item = paper_feed.Item.create(feed=feed, guid="g1", dedupe_key="title:gca1", title="Zinc isotopes in chondrules",
                                  authors="[]", info="title")
    paper_feed.enrich_items()
    item = paper_feed.Item.get_by_id(item.id)
    assert item.enrich_attempts == 1 and item.abstract is None
    paper_feed.score_items()
    # 三天后 OpenAlex 有了摘要：重试补全，并按“有摘要”重新打分
    paper_feed.Item.update(last_enrich_at=dt.datetime.now() - dt.timedelta(days=4)).where(paper_feed.Item.id == item.id).execute()
    feed_env.title_hits["Zinc isotopes in chondrules"] = {"doi": "10.1/gca", "abstract": "zinc isotope " * 40,
                                                          "source": "openalex", "authors": ["A B"], "journal": "GCA"}
    paper_feed.enrich_items()
    item = paper_feed.Item.get_by_id(item.id)
    assert item.doi == "10.1/gca" and item.info == "abstract" and item.status == "new" and item.enrich_attempts == 2


def test_recalibration_is_separate_per_class(feed_env):
    feed = paper_feed.add_feed("X", "https://example.org/x")
    now = dt.datetime.now()
    for n in range(6):  # 仅标题的自动入库：6 篇里 4 篇被你 👎 → 门槛收紧
        paper_feed.Item.create(feed=feed, guid=f"t{n}", dedupe_key=f"t:{n}", title=f"t{n}", info="title",
                               ingest_mode="auto", status="ingested", ingested_at=now, feedback=1 if n < 2 else -1)
    data = paper_feed.recalibrate()
    assert data["auto"]["title"] > feed_ranker.AUTO_THRESHOLD["title"]
    assert data["auto"]["abstract"] == feed_ranker.AUTO_THRESHOLD["abstract"]   # 另一类不受影响
    assert paper_feed.precision_stats()["auto_precision"]["title"]["rated"] == 6


def test_save_profile_merges_and_keeps_other_groups(feed_env):
    paper_feed.save_profile({"facets": {"cluster:1": {"enabled": False, "weight": 1}, "cluster:2": {"enabled": True, "weight": 1.5}}})
    paper_feed.save_profile({"facets": {"cluster:2": {"enabled": True, "weight": 2}}, "notes": "x"})
    facets = paper_feed.load_profile()["facets"]
    assert facets["cluster:1"]["enabled"] is False and facets["cluster:2"]["weight"] == 2
    paper_feed.save_profile({"must_follow_authors": ["A. B"]})      # 只存补充画像，不动各组设置
    assert paper_feed.load_profile()["facets"]["cluster:1"]["enabled"] is False
    assert paper_feed.load_profile()["notes"] == "x"


def test_library_self_grouping_finds_topics_and_keeps_ids_stable():
    rng = np.random.default_rng(1)
    centers = np.eye(3, 16, dtype=np.float32)
    vectors = np.vstack([c + rng.normal(0, 0.05, (20, 16)) for c in centers]).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    pids = list(range(1, 61))
    first = feed_ranker.cluster_library(pids, vectors, k_range=(2, 6))
    assert len(first) == 3 and sorted(len(c["pids"]) for c in first) == [20, 20, 20]
    # 加入几篇新文献后重新分组：同一主题的分组沿用原编号
    extra = (centers[0] + rng.normal(0, 0.05, (4, 16))).astype(np.float32)
    extra /= np.linalg.norm(extra, axis=1, keepdims=True)
    second = feed_ranker.cluster_library(pids + [61, 62, 63, 64], np.vstack([vectors, extra]), previous=first, k_range=(2, 6))
    old_of = {pid: c["id"] for c in first for pid in c["pids"]}
    for cluster in second:
        assert {old_of[p] for p in cluster["pids"] if p in old_of} == {cluster["id"]}
    assert any({61, 62}.issubset(c["pids"]) for c in second)


def test_run_survives_concurrent_writer_during_fetch(feed_env, monkeypatch):
    """抓取各订阅源期间另一线程写库（例如画像页的“分组”在写向量缓存）：不能报 database is locked。"""
    import threading

    paper_feed.add_feed("A", "https://example.org/a")
    paper_feed.add_feed("B", "https://example.org/b")
    original = paper_feed.fetch_feed

    def fetch_with_other_writer(feed, dois):
        t = threading.Thread(target=lambda: paper_feed.LibraryVector.replace(
            pid=feed.id, kind="full", digest="x", vec=b"\0").execute())
        t.start()
        t.join()
        return original(feed, dois)

    monkeypatch.setattr(paper_feed, "fetch_feed", fetch_with_other_writer)
    result = paper_feed.run_once()
    assert not [e for e in result["errors"] if "locked" in e], result["errors"]
    assert all(f.last_fetch_at for f in paper_feed.Feed.select())


def test_ingest_rechecks_doi_but_never_matches_empty_doi(feed_env, monkeypatch):
    feed = paper_feed.add_feed("Icarus", "https://example.org/rss")
    paper_feed.fetch_feed(feed, {})
    zinc = paper_feed.Item.get(paper_feed.Item.title.contains("Zinc"))
    bird = paper_feed.Item.get(paper_feed.Item.title.contains("Bird"))
    assert bird.doi is None
    # 抓取之后，文献库里有了这篇（DOI 写法不同）；另有一篇没有 DOI 的库内文献
    monkeypatch.setattr(feed_env, "library_doi_map", lambda: {"10.1016/j.icarus.2026.1": 42})
    item = paper_feed.ingest(zinc.id)
    assert item.status == "in_library" and item.in_library_pid == 42 and feed_env.zotero == []
    item = paper_feed.ingest(bird.id)            # 没有 DOI：不查重，照常入库
    assert item.status == "ingested" and feed_env.zotero == [("RSS_TRACK/Icarus", "Bird migration in Europe")]


def test_split_authors_handles_wiley_and_rejects_glued_strings():
    assert paper_feed.split_authors([{"name": "N. Agarwal, \nT. A. Smith, \nL. C. Slivinski"}]) == ["N. Agarwal", "T. A. Smith", "L. C. Slivinski"]
    glued = "M. Colin MarvinAlex GleasonClément Narteauahttps://ror.org/00f54p054Department of Earth Sciences"
    assert paper_feed.split_authors(glued) == []          # 无法可靠拆分：留空，交给 OpenAlex 补全
    assert paper_feed.split_authors("Sean N. Raymond, Alice Smith") == ["Sean N. Raymond", "Alice Smith"]
    assert paper_feed.split_authors([{"name": "Donald J. DePaolo"}, {"name": "William F. McDonough"}]) == ["Donald J. DePaolo", "William F. McDonough"]


def test_report_citations_become_clickable_chips(feed_env):
    from Pageboygent import ui

    feed = paper_feed.add_feed("Icarus", "https://example.org/rss")
    paper_feed.fetch_feed(feed, {})
    zinc = paper_feed.Item.get(paper_feed.Item.title.contains("Zinc"))
    zinc.authors = json.dumps(["Sean N. Raymond", "Alice Smith"])
    zinc.save()
    html = ui._report_overview_html(f"锌同位素蒸发 [F{zinc.id}]，另见 [F99999]。")
    assert f'data-open="{zinc.id}"' in html and "Raymond 等" in html and "[F" not in html
    assert "已清理" in html                                  # 已被清理的条目不留下裸编号



def test_journal_names_are_cleaned():
    assert paper_feed.clean_journal("ScienceDirect Publication: Icarus") == "Icarus"
    assert paper_feed.clean_journal("Wiley: Journal of Geophysical Research: Planets: Table of Contents") == "Journal of Geophysical Research: Planets"
    assert paper_feed.clean_journal("Earth and Planetary Science Letters") == "Earth and Planetary Science Letters"
    assert paper_feed._bad_creators([{"creatorType": "author", "firstName": "", "lastName": "93106"}])
    assert paper_feed._bad_creators([])
    assert not paper_feed._bad_creators([{"creatorType": "author", "firstName": "James M D", "lastName": "Day"}])


def test_title_only_item_is_enriched_before_ingest(feed_env, monkeypatch):
    feed = paper_feed.add_feed("Icarus", "https://example.org/rss")
    paper_feed.fetch_feed(feed, {})
    bird = paper_feed.Item.get(paper_feed.Item.title.contains("Bird"))
    assert bird.doi is None and bird.authors_list() == []
    feed_env.title_hits = {bird.title: {"doi": "10.1/bird", "abstract": "", "source": "none",
                                        "authors": ["Jane Doe"], "journal": "Icarus", "published": "2026-11-20"}}
    feed_env.added.clear()
    paper_feed.ingest(bird.id)                      # 入库前当场补全：DOI、作者、刊名、日期一并写进 Zotero
    sent = feed_env.added[-1]
    assert sent["doi"] == "10.1/bird" and sent["authors"] == ["Jane Doe"] and sent["journal"] == "Icarus"
    assert sent["published"] == TODAY.strftime("%Y-%m-%d")   # RSS 自带日期：补全只填缺的，不覆盖
    assert paper_feed.sync_ingested_to_zotero() == 1 and feed_env.patched[-1][1]["DOI"] == "10.1/bird"


def _pending_item(feed, title, abstract, band="low"):
    vec = fake_vec(f"{title}\n{abstract}")
    return paper_feed.Item.create(feed=feed, guid=title, dedupe_key=f"title:{title}", title=title, abstract=abstract,
                                  info="abstract", vector=feed_ranker.vector_to_blob(vec), features="{}",
                                  base_score=0.0, score=0.0, band=band, status="scored")


def test_rescoring_promotes_at_most_to_confirm_and_reuses_llm(feed_env):
    feed = paper_feed.add_feed("Icarus", "https://example.org/rss")
    long_text = "We measured zinc isotope evaporation in lunar samples and moon basalts. " * 4
    item = _pending_item(feed, "Zinc isotope evaporation in lunar samples", long_text)
    result = paper_feed.rescore_pending()
    item = paper_feed.Item.get_by_id(item.id)
    assert result["judged"] == 1 and item.llm_score == 9          # 新进入短名单：补做 LLM 判断
    assert item.band == "confirm" and item.status == "scored"     # 分数够自动入库，但重新评估最多到“待确认”
    assert item.rerank_note and item.rerank_note.endswith("上升")
    calls = len(feed_env.client.calls)
    assert paper_feed.rescore_pending()["judged"] == 0             # 画像没变：沿用原判断，不再花钱
    assert len(feed_env.client.calls) == calls


def test_rescoring_reflects_new_negative_feedback(feed_env):
    feed = paper_feed.add_feed("Icarus", "https://example.org/rss")
    text = "Zinc isotope evaporation on the moon. " * 8
    target = _pending_item(feed, "Zinc isotope evaporation on the moon", text, band="confirm")
    paper_feed.rescore_pending()
    before = json.loads(paper_feed.Item.get_by_id(target.id).features)["feedback"]
    disliked = _pending_item(feed, "Zinc isotope evaporation on the moon again", text)
    paper_feed.set_feedback(disliked.id, -1)
    paper_feed.rescore_pending()
    after = json.loads(paper_feed.Item.get_by_id(target.id).features)["feedback"]
    assert after < before                                          # 👎 了一篇很像的，这篇随之扣分



def test_title_only_item_matches_library_by_exact_title(feed_env, monkeypatch):
    feed = paper_feed.add_feed("Icarus", "https://example.org/rss")
    paper_feed.fetch_feed(feed, {})
    bird = paper_feed.Item.get(paper_feed.Item.title.contains("Bird"))
    long_title = "Bird migration in Europe and the magnetic compass of songbirds"
    bird.title = long_title
    bird.save()
    library = {f"title:{paper_feed._norm_title(long_title.upper() + '.')}": 7}   # 大小写、标点不同也算同一篇
    monkeypatch.setattr(feed_env, "library_doi_map", lambda: library)
    feed_env.added.clear()
    item = paper_feed.ingest(bird.id)
    assert item.status == "in_library" and item.in_library_pid == 7 and feed_env.added == []
    assert paper_feed.library_match(library, None, "Short") is None       # 太短的标题不拿来查重


def test_mark_low_read_only_touches_unread_low_items(feed_env):
    feed = paper_feed.add_feed("Icarus", "https://example.org/rss")
    other = paper_feed.add_feed("GCA", "https://example.org/gca")
    low = _pending_item(feed, "Bird migration patterns in Europe", "Birds.", band="low")
    low_other = _pending_item(other, "Bird song in Asia", "Birds.", band="low")
    keep = _pending_item(feed, "Zinc isotope evaporation", "Zinc.", band="confirm")
    assert paper_feed.count_unread_low(feed.id) == 1 and paper_feed.count_unread_low() == 2
    assert paper_feed.mark_low_read(feed.id) == 1                    # 只动当前订阅源
    assert paper_feed.Item.get_by_id(low.id).read_at is not None
    assert paper_feed.Item.get_by_id(low_other.id).read_at is None
    assert paper_feed.Item.get_by_id(keep.id).read_at is None        # 待确认的不动
    assert paper_feed.Item.get_by_id(low.id).feedback is None        # 只是已读，不算 👎



def test_ingested_and_in_library_items_count_as_read(feed_env, monkeypatch):
    feed = paper_feed.add_feed("Icarus", "https://example.org/rss")
    paper_feed.fetch_feed(feed, {"10.1016/j.icarus.2026.1": 5})            # 库里已有：抓到时就算已读
    zinc = paper_feed.Item.get(paper_feed.Item.title.contains("Zinc"))
    assert zinc.status == "in_library" and zinc.read_at is not None
    bird = paper_feed.Item.get(paper_feed.Item.title.contains("Bird"))
    assert bird.read_at is None
    assert paper_feed.ingest(bird.id).read_at is not None                  # 入库即已读
    old = _pending_item(feed, "Old ingested paper", "x", band="confirm")
    paper_feed.Item.update(status="ingested").where(paper_feed.Item.id == old.id).execute()
    assert paper_feed.mark_settled_read() == 1                             # 以前入库的也补上


def test_ingest_during_background_rescoring_is_not_overwritten(feed_env, monkeypatch):
    """后台重新评估已经把条目读进内存，这时你点了入库：评估写回时不能把“已入库”覆盖成“待确认”。"""
    feed = paper_feed.add_feed("Icarus", "https://example.org/rss")
    text = "We measured zinc isotope evaporation in lunar samples and moon basalts. " * 4
    item = _pending_item(feed, "Zinc isotope evaporation in lunar samples", text)
    original = paper_feed._judge

    def judge_while_user_ingests(pairs, model, profile):
        paper_feed.ingest(item.id)                   # 用户在后台评估进行中点了“入库”
        return original(pairs, model, profile)

    monkeypatch.setattr(paper_feed, "_judge", judge_while_user_ingests)
    paper_feed.rescore_pending()
    after = paper_feed.Item.get_by_id(item.id)
    assert after.status == "ingested" and after.zotero_key == "ZKEY0001" and after.read_at is not None


def test_ingest_during_scoring_is_not_overwritten(feed_env, monkeypatch):
    feed = paper_feed.add_feed("Icarus", "https://example.org/rss")
    paper_feed.fetch_feed(feed, {})
    zinc = paper_feed.Item.get(paper_feed.Item.title.contains("Zinc"))
    original = paper_feed._judge

    def judge_while_user_ingests(pairs, model, profile):
        paper_feed.ingest(zinc.id)
        return original(pairs, model, profile)

    monkeypatch.setattr(paper_feed, "_judge", judge_while_user_ingests)
    paper_feed.score_items()
    assert paper_feed.Item.get_by_id(zinc.id).status == "ingested"


@pytest.fixture
def shutdown_flag():
    from Pageboygent import write_queue

    yield write_queue
    write_queue._shutdown_event.clear()


def test_shutdown_stops_background_loops_at_safe_points(feed_env, shutdown_flag, monkeypatch):
    import threading

    write_queue = shutdown_flag
    paper_feed.add_feed("A", "https://example.org/a")
    paper_feed.add_feed("B", "https://example.org/b")
    fetched = []
    original = paper_feed.fetch_feed

    def fetch_then_ctrl_c(feed, dois):
        fetched.append(feed.name)
        write_queue.request_shutdown()           # 抓第一个源的时候按了 Ctrl-C
        return original(feed, dois)

    monkeypatch.setattr(paper_feed, "fetch_feed", fetch_then_ctrl_c)
    with pytest.raises(KeyboardInterrupt):
        paper_feed.run_once()
    assert fetched == ["A"]                       # 第二个源没有再抓
    assert paper_feed._run_lock.acquire(blocking=False)   # 锁已释放，下次能正常运行
    paper_feed._run_lock.release()

    # heartbeat 线程上的 checkpoint（快速总结、主题总结等循环里都在调用）同样会停下
    caught = []

    def heartbeat_like():
        write_queue.set_consumer_thread()
        try:
            write_queue.checkpoint()
        except KeyboardInterrupt:
            caught.append(True)

    thread = threading.Thread(target=heartbeat_like)
    thread.start()
    thread.join()
    assert caught == [True]
    write_queue.checkpoint()                      # 其他线程（网页请求）上调用不受影响
