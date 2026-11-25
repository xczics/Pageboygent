"""
文献追踪（RSS）：订阅 → 抓取 → 补全元数据 → 排序分档 → 自动/确认入库 → 每周报告；同时是一个 RSS 阅读器。

数据：workstation_dir/paper_feed.db（与 pageboygent.db 分离，避免与 heartbeat 写库争锁）
画像：workstation_dir/paper_feed/profile.json（结构化，用户可编辑）

分档（见 feed_ranker）：
    auto     高分且有摘要 → 自动入库（进 Zotero 的 RSS_TRACK/<订阅源> 子集合，并纳入 Pageboygent 追踪）
    confirm  中分 → 等你确认后入库
    low      只留在候选池 / 阅读器里
反馈：👍 / 👎 / 确认入库（=👍）。AI 自动入库但你没表态的文章不当作正例，避免模型自我强化。

替代了另一个 AI 写的未审查草稿 rss_tracker.py（已删除）。
"""
from __future__ import annotations

import datetime as dt
import hashlib
import html
import json
import re
import threading
import time
from pathlib import Path
from typing import Any

import peewee

from .config import get_config

_workstation = Path(get_config("workstation_dir")).expanduser()
_data_dir = _workstation / "paper_feed"
_db = peewee.SqliteDatabase(None, pragmas={"journal_mode": "wal", "busy_timeout": 10000, "foreign_keys": 1},
                            check_same_thread=False)
_init_lock = threading.Lock()
_run_lock = threading.Lock()
_initialized_path: str | None = None


class _Base(peewee.Model):
    class Meta:
        database = _db


class Feed(_Base):
    name = peewee.CharField()
    url = peewee.CharField(unique=True)
    enabled = peewee.BooleanField(default=True)
    broad = peewee.BooleanField(default=False)  # 综合期刊：先按内容预筛，省 LLM 调用（不影响分数）
    created_at = peewee.DateTimeField(default=dt.datetime.now)
    last_fetch_at = peewee.DateTimeField(null=True)
    last_error = peewee.TextField(null=True)
    last_new_count = peewee.IntegerField(default=0)


class Item(_Base):
    feed = peewee.ForeignKeyField(Feed, backref="items", on_delete="CASCADE")
    guid = peewee.CharField(index=True)
    dedupe_key = peewee.CharField(unique=True)        # doi:… 或 title:…
    doi = peewee.CharField(null=True, index=True)
    title = peewee.TextField()
    authors = peewee.TextField(default="[]")          # JSON list
    journal = peewee.CharField(null=True)
    published = peewee.DateTimeField(null=True)
    link = peewee.CharField(null=True)
    raw_html = peewee.TextField(null=True)            # RSS 原始内容（阅读器展示用）
    abstract = peewee.TextField(null=True)
    abstract_source = peewee.CharField(default="none")  # rss / openalex / crossref / none
    enriched = peewee.BooleanField(default=False)
    enrich_attempts = peewee.IntegerField(default=0)
    last_enrich_at = peewee.DateTimeField(null=True)
    info = peewee.CharField(default="title")          # abstract / title：按信息量分开打分、分开校准
    vector = peewee.BlobField(null=True)
    features = peewee.TextField(null=True)            # JSON
    base_score = peewee.FloatField(null=True)
    llm_score = peewee.FloatField(null=True)
    llm_reason = peewee.TextField(null=True)
    score = peewee.FloatField(null=True, index=True)
    band = peewee.CharField(null=True, index=True)    # auto / confirm / low
    facet_name = peewee.CharField(null=True)
    status = peewee.CharField(default="new", index=True)  # new / scored / ingested / in_library / dismissed / error
    in_library_pid = peewee.IntegerField(null=True)
    ingest_mode = peewee.CharField(null=True)         # auto / manual
    ingested_at = peewee.DateTimeField(null=True)
    zotero_key = peewee.CharField(null=True)
    ingest_error = peewee.TextField(null=True)
    feedback = peewee.IntegerField(null=True)         # +1 / -1
    feedback_at = peewee.DateTimeField(null=True)
    exploration = peewee.BooleanField(default=False)  # 周报里的“边界样本”
    read_at = peewee.DateTimeField(null=True)         # 在阅读器里打开过（隐式信号，第二期使用）
    created_at = peewee.DateTimeField(default=dt.datetime.now, index=True)
    scored_at = peewee.DateTimeField(null=True)

    def authors_list(self) -> list[str]:
        try:
            return json.loads(self.authors or "[]")
        except json.JSONDecodeError:
            return []


class Report(_Base):
    week = peewee.CharField(unique=True)              # 2026-W48
    generated_at = peewee.DateTimeField(default=dt.datetime.now)
    read_at = peewee.DateTimeField(null=True)
    overview = peewee.TextField(default="")
    data = peewee.TextField(default="{}")             # JSON：统计、各分组的文章 id、边界样本


class LibraryVector(_Base):
    pid = peewee.IntegerField()
    kind = peewee.CharField()                          # full（标题+摘要/总结）/ title（仅标题）
    digest = peewee.CharField()
    vec = peewee.BlobField()

    class Meta:
        primary_key = peewee.CompositeKey("pid", "kind")


class LibraryAuthor(_Base):
    pid = peewee.IntegerField(primary_key=True)
    key = peewee.CharField()
    authors = peewee.TextField(default="[]")


def init_db(path: Path | None = None) -> None:
    """初始化数据库。不带参数时：已经初始化过就保持当前数据库（不能悄悄切回默认路径——测试里会写到真实数据）。"""
    global _initialized_path
    with _init_lock:
        if path is None and _initialized_path is not None:
            return
        path = path or (_workstation / "paper_feed.db")
        if _initialized_path == str(path):
            return
        if not _db.is_closed():
            _db.close()
        path.parent.mkdir(parents=True, exist_ok=True)
        _db.init(str(path))
        models = [Feed, Item, Report, LibraryVector, LibraryAuthor]
        _db.create_tables(models, safe=True)
        _add_missing_columns(models)
        _initialized_path = str(path)


def _add_missing_columns(models: list) -> None:
    """给已有的表补上新增的字段（只增不删），避免加字段后旧数据库报错。"""
    from playhouse.migrate import SqliteMigrator, migrate

    migrator = SqliteMigrator(_db)
    for model in models:
        existing = {c.name for c in _db.get_columns(model._meta.table_name)}
        ops = [migrator.add_column(model._meta.table_name, f.column_name, f)
               for f in model._meta.sorted_fields if f.column_name not in existing]
        if ops:
            migrate(*ops)


# ---------------------------------------------------------------------------
# 文本清洗
# ---------------------------------------------------------------------------

_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>&]+)", re.I)


def strip_html(text: str | None) -> str:
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text or "", flags=re.S | re.I)
    text = re.sub(r"<br\s*/?>|</p>", "\n", text, flags=re.I)
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n\n", text)).strip()


def extract_doi(*texts: str | None) -> str | None:
    for text in texts:
        match = _DOI_RE.search(html.unescape(text or ""))
        if match:
            return match.group(1).rstrip(".);,]").lower()
    return None


def _norm_title(title: str) -> str:
    return re.sub(r"[\W_]+", "", (title or "").lower())[:200]


def split_authors(value: Any) -> list[str]:
    if isinstance(value, list):
        names = [a.get("name", "") if isinstance(a, dict) else str(a) for a in value]
    else:
        text = strip_html(str(value or ""))
        names = re.split(r";|\band\b|&|,(?=\s*[A-Z][a-z]+\s+[A-Z])", text)
    return [n.strip(" ,.") for n in names if n and len(n.strip()) > 1][:60]


# ---------------------------------------------------------------------------
# 外部访问（RSS、OpenAlex/Crossref、Zotero、LLM）集中在这里，测试时整体替换
# ---------------------------------------------------------------------------

class FeedBackend:
    user_agent = "Pageboygent/0.1 (literature tracker)"

    def parse_feed(self, url: str) -> Any:
        import feedparser

        return feedparser.parse(url, agent=self.user_agent)

    def enrich(self, doi: str) -> dict | None:
        """OpenAlex → Crossref。返回 {abstract, authors, journal, published, source}，取不到时 None。"""
        import requests

        try:
            r = requests.get(f"https://api.openalex.org/works/doi:{doi}", timeout=20,
                             params={"select": "title,publication_date,authorships,abstract_inverted_index,primary_location"},
                             headers={"User-Agent": self.user_agent})
            if r.ok:
                d = r.json()
                words: dict[int, str] = {}
                for word, positions in (d.get("abstract_inverted_index") or {}).items():
                    for pos in positions:
                        words[pos] = word
                abstract = " ".join(words[i] for i in sorted(words))
                source = ((d.get("primary_location") or {}).get("source") or {})
                return {"abstract": abstract, "source": "openalex" if abstract else "none",
                        "authors": [a["author"]["display_name"] for a in d.get("authorships", []) if a.get("author")],
                        "journal": source.get("display_name"), "published": d.get("publication_date")}
        except Exception as exc:
            print(f"[paper_feed] OpenAlex failed for {doi}: {exc}")
        try:
            r = requests.get(f"https://api.crossref.org/works/{doi}", timeout=20, headers={"User-Agent": self.user_agent})
            if r.ok:
                m = r.json().get("message", {})
                abstract = strip_html(m.get("abstract") or "")
                return {"abstract": abstract, "source": "crossref" if abstract else "none",
                        "authors": [" ".join(filter(None, [a.get("given"), a.get("family")])) for a in m.get("author", [])],
                        "journal": (m.get("container-title") or [None])[0], "published": None}
        except Exception as exc:
            print(f"[paper_feed] Crossref failed for {doi}: {exc}")
        return None

    def enrich_by_title(self, title: str) -> dict | None:
        """没有 DOI 时按标题去 OpenAlex 找（标题几乎完全一致才采用）。返回 {doi, abstract, authors, journal, source}。"""
        import difflib

        import requests

        try:
            r = requests.get("https://api.openalex.org/works", timeout=20, headers={"User-Agent": self.user_agent},
                             params={"search": title[:300], "per-page": 3,
                                     "select": "doi,title,authorships,abstract_inverted_index,primary_location"})
            if not r.ok:
                return None
            for work in r.json().get("results", []):
                ratio = difflib.SequenceMatcher(None, _norm_title(title), _norm_title(work.get("title") or "")).ratio()
                if ratio < 0.92:
                    continue
                words: dict[int, str] = {}
                for word, positions in (work.get("abstract_inverted_index") or {}).items():
                    for pos in positions:
                        words[pos] = word
                abstract = " ".join(words[i] for i in sorted(words))
                doi = (work.get("doi") or "").replace("https://doi.org/", "").lower() or None
                source = ((work.get("primary_location") or {}).get("source") or {})
                return {"doi": doi, "abstract": abstract, "source": "openalex" if abstract else "none",
                        "authors": [a["author"]["display_name"] for a in work.get("authorships", []) if a.get("author")],
                        "journal": source.get("display_name")}
        except Exception as exc:
            print(f"[paper_feed] OpenAlex title search failed: {exc}")
        return None

    def llm_client(self) -> Any:
        from openai import OpenAI

        return OpenAI(api_key=get_config("ds_api_key"), base_url=get_config("ds_url"))

    def library_doi_map(self) -> dict[str, int]:
        from .storage import Paper

        return {str(p.doi).lower(): int(p.id) for p in Paper.select(Paper.id, Paper.doi) if p.doi}

    # ---- Zotero ----
    def zotero_collection(self, root: str, child: str) -> str:
        """确保 <root>/<child> 子集合存在，返回其 key。"""
        from .zt_tools import _perform_local_write, zot

        def _find(name: str, parent: str | None) -> str | None:
            for c in zot.everything(zot.collections()):
                data = c["data"]
                if data["name"] == name and (data.get("parentCollection") or None) == parent:
                    return data["key"]
            return None

        root_key = _find(root, None)
        if root_key is None:
            _perform_local_write(f"create collection {root}", zot.create_collections, [{"name": root}])
            root_key = _find(root, None)
        child_key = _find(child, root_key)
        if child_key is None:
            _perform_local_write(f"create collection {root}/{child}", zot.create_collections,
                                 [{"name": child, "parentCollection": root_key}])
            child_key = _find(child, root_key)
        if not child_key:
            raise RuntimeError(f"无法在 Zotero 中创建 {root}/{child} 集合（请在 Zotero 中手动创建后重试）")
        return child_key

    def zotero_add_item(self, collection_key: str, item: dict) -> str:
        """在集合中新建条目（已有同 DOI 条目则直接加入集合），返回条目 key。"""
        from .zt_tools import _perform_local_write, zot

        if item.get("doi"):
            for existing in zot.everything(zot.items(q=item["doi"], qmode="everything")):
                if str(existing["data"].get("DOI", "")).lower() == item["doi"].lower():
                    _perform_local_write("add to collection", zot.addto_collection, collection_key, existing)
                    return existing["data"]["key"]
        creators = []
        for name in item.get("authors", [])[:50]:
            parts = name.rsplit(" ", 1)
            creators.append({"creatorType": "author", "firstName": parts[0] if len(parts) > 1 else "",
                             "lastName": parts[-1]})
        data = {"itemType": "journalArticle", "title": item["title"], "creators": creators, "tags": [],
                "collections": [collection_key], "relations": {}, "DOI": item.get("doi") or "",
                "url": item.get("link") or "", "abstractNote": item.get("abstract") or "",
                "publicationTitle": item.get("journal") or "", "date": item.get("published") or ""}
        result = _perform_local_write("create rss item", zot.create_items, [data])
        created = (result or {}).get("successful") or (result or {}).get("success") or {}
        first = next(iter(created.values()), None) if isinstance(created, dict) else None
        key = first.get("key") if isinstance(first, dict) else first
        if not key:
            raise RuntimeError(f"Zotero 未返回新条目的 key：{result}")
        return str(key)

    def track_in_pageboygent(self, zotero_key: str, root: str) -> None:
        """纳入 Pageboygent 追踪：经 write_queue 交给 heartbeat 线程执行（PDF → 全文 → 总结 流水线自动接手）。"""
        from . import write_queue
        from .storage import Paper, TracedCollection

        def _task() -> None:
            TracedCollection.get_or_create(name=root, defaults={"note": "文献追踪（RSS）自动入库的文献"})
            Paper.trace_paper(zotero_key)

        write_queue.submit(f"trace RSS paper {zotero_key}", _task)


backend = FeedBackend()


# ---------------------------------------------------------------------------
# 画像（结构化，用户可编辑）
# ---------------------------------------------------------------------------

def _profile_path() -> Path:
    return _data_dir / "profile.json"


def load_profile() -> dict:
    path = _profile_path()
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"facets": {}, "must_follow_authors": [], "avoid_topics": [], "notes": ""}


def save_profile(profile: dict) -> None:
    """保存画像里你的设置（各组的启用 / 权重、必看作者、不感兴趣的方向、补充说明）。
    各组的名称和逐行说明不在这里保存，见 facet_store.edit_claims。只更新传进来的组，其余原样保留。"""
    path = _profile_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    old = load_profile()
    merged = dict(old.get("facets") or {})
    touched = []
    for k, v in (profile.get("facets") or {}).items():
        entry = {"enabled": bool(v.get("enabled", True)), "weight": float(v.get("weight", 1.0))}
        before = merged.get(k) or {"enabled": True, "weight": 1.0}
        if (entry["enabled"], entry["weight"]) != (bool(before.get("enabled", True)), float(before.get("weight", 1.0))):
            touched.append(k)
        merged[k] = entry
    clean = {
        "facets": merged,
        "must_follow_authors": [a.strip() for a in profile.get("must_follow_authors", old.get("must_follow_authors", [])) if a.strip()][:200],
        "avoid_topics": [t.strip() for t in profile.get("avoid_topics", old.get("avoid_topics", [])) if t.strip()][:100],
        "notes": str(profile.get("notes", old.get("notes", "")))[:3000],
    }
    path.write_text(json.dumps(clean, ensure_ascii=False, indent=1), encoding="utf-8")
    if touched:
        facet_store_ready().mark_touched(touched)
    invalidate_model()


def invalidate_model() -> None:
    global _model_cache
    _model_cache = None


def facet_store_ready():
    """facet_store 指向当前数据目录（并在第一次时迁移旧格式的分组文件）。"""
    from . import facet_store

    facet_store.set_data_dir(_data_dir)
    facet_store.migrate_legacy(load_profile())
    return facet_store


# ---------------------------------------------------------------------------
# 库内数据缓存（供 feed_ranker 使用）
# ---------------------------------------------------------------------------

def vector_cache_get(pid: int, digest: str, kind: str = "full"):
    from .feed_ranker import blob_to_vector

    init_db()
    row = LibraryVector.get_or_none((LibraryVector.pid == pid) & (LibraryVector.kind == kind))
    return blob_to_vector(row.vec) if row and row.digest == digest else None


def vector_cache_put(pid: int, digest: str, vec, kind: str = "full") -> None:
    from .feed_ranker import vector_to_blob

    init_db()
    LibraryVector.replace(pid=pid, kind=kind, digest=digest, vec=vector_to_blob(vec)).execute()


def library_authors(papers: list[dict]) -> dict[int, list[str]]:
    """库内文献的完整作者列表（Zotero creators，按条目缓存；读不到时用第一作者）。"""
    init_db()
    cached = {row.pid: row for row in LibraryAuthor.select()}
    out: dict[int, list[str]] = {}
    missing = [p for p in papers if p["pid"] not in cached or cached[p["pid"]].key != p["key"]]
    if missing:
        try:
            from .zt_tools import get_info

            for paper in missing:
                creators, = get_info(paper["key"], ["creators"])
                names = [" ".join(filter(None, [c.get("firstName"), c.get("lastName")])) or c.get("name", "")
                         for c in (creators or []) if c.get("creatorType") == "author"]
                LibraryAuthor.replace(pid=paper["pid"], key=paper["key"], authors=json.dumps(names)).execute()
            cached = {row.pid: row for row in LibraryAuthor.select()}
        except Exception as exc:
            print(f"[paper_feed] cannot read creators from Zotero, falling back to first authors: {exc}")
    for paper in papers:
        row = cached.get(paper["pid"])
        names = json.loads(row.authors) if row else []
        out[paper["pid"]] = names or ([paper["first_author"]] if paper.get("first_author") else [])
    return out


_model_cache = None
_model_signature = None
ranker_backend = None  # 测试时替换为伪造的 RankerBackend


def library_model(*, describe: bool = True):
    """库内排序模型（库内文献或画像变化时重建）。describe=True 时顺便让 AI 为新组提炼说明、为有新文献的组提建议。"""
    global _model_cache, _model_signature
    from . import feed_ranker

    rb = ranker_backend or feed_ranker.RankerBackend()
    profile = load_profile()
    papers = rb.library_papers()
    signature = (len(papers), json.dumps(profile, sort_keys=True))
    if _model_cache is None or _model_signature != signature:
        _model_cache = feed_ranker.build_library_model(rb, profile)
        _model_signature = signature
    store = facet_store_ready()
    data = describe_facets(_model_cache, rb, papers) if describe else store.load()
    refresh_facet_text(_model_cache, data)
    return _model_cache, rb, profile


def refresh_facet_text(model, data: dict) -> None:
    """把各组最新的名称和说明（逐行编辑后）同步进已构建好的排序模型，不必重建向量。"""
    current = {f.id: f for f in facet_store_ready().facets_for_ranking(data)}
    for facet in model.facets:
        if facet.id in current:
            facet.name, facet.description, facet.user_note = (current[facet.id].name, current[facet.id].description,
                                                              current[facet.id].user_note)


# ---------------------------------------------------------------------------
# 兴趣方向：从库里自动分组（规则见 facet_store：已有分组固定，AI 只能加组；说明逐行，AI 只能提议新增）
# ---------------------------------------------------------------------------

_force_regroup = False


def request_regroup() -> None:
    """下次构建排序模型时整理分组（用户点“整理分组”：只重新划分没动过的组）。"""
    global _force_regroup
    _force_regroup = True
    invalidate_model()


def ensure_clusters(pids: list[int], vectors) -> list:
    global _force_regroup
    store = facet_store_ready()
    data = store.sync_groups(pids, vectors, reorganize=_force_regroup)
    _force_regroup = False
    return store.facets_for_ranking(data)


def _quick_line(summary: str) -> str:
    text = (summary or "").split("## Labels")[0]
    match = re.search(r"\*\*Quick Summary\*\*\s*(.+?)(?:\n\s*\n|$)", text, re.S)
    line = match.group(1) if match else text
    return re.sub(r"\s+", " ", line).strip()[:220]


def _ask_json(system: str, user: str) -> dict | None:
    from .lit_search import parse_tool_arguments

    try:
        client = backend.llm_client()
        model_name = get_config("model_name_cheap")
        response = client.chat.completions.create(
            model=model_name, temperature=0.2, max_tokens=800, stream=False,
            response_format={"type": "json_object"}, extra_body={"thinking": {"type": "disabled"}},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        _record_llm(getattr(response, "usage", None), model_name)
        parsed, _ = parse_tool_arguments(response.choices[0].message.content or "")
        return parsed if isinstance(parsed, dict) else None
    except Exception as exc:
        print(f"[paper_feed] facet description failed: {exc}")
        return None


def describe_facets(model, rb, papers: list[dict], *, force: bool = False) -> dict:
    return facet_store_ready().describe(papers, ask=_ask_json, embed=rb.embed, quick_line=_quick_line,
                                        pids=model.pids, vectors=model.vectors, force=force)


# ---------------------------------------------------------------------------
# 订阅源
# ---------------------------------------------------------------------------

def preview_feed(url: str) -> dict:
    parsed = backend.parse_feed(url)
    entries = getattr(parsed, "entries", []) or []
    if getattr(parsed, "bozo", False) and not entries:
        raise ValueError(f"无法解析这个 RSS：{getattr(parsed, 'bozo_exception', '')}")
    title = (getattr(parsed, "feed", {}) or {}).get("title") or ""
    with_abstract = sum(1 for e in entries if len(strip_html(e.get("summary") or "")) > 200)
    with_doi = sum(1 for e in entries if extract_doi(e.get("link"), e.get("id"), e.get("summary"),
                                                     e.get("prism_doi"), e.get("dc_identifier")))
    return {"title": title, "entries": len(entries), "with_abstract": with_abstract, "with_doi": with_doi,
            "samples": [strip_html(e.get("title", ""))[:120] for e in entries[:5]]}


def add_feed(name: str, url: str, broad: bool = False) -> Feed:
    init_db()
    url = url.strip()
    if not re.match(r"^https?://", url):
        raise ValueError("RSS 地址需要以 http:// 或 https:// 开头")
    if Feed.get_or_none(Feed.url == url):
        raise ValueError("这个 RSS 已经订阅过了")
    name = (name or "").strip() or preview_feed(url)["title"] or url
    if "/" in name:
        raise ValueError("名称里不能有 “/”（会用作 Zotero 子集合名）")
    return Feed.create(name=name[:120], url=url, broad=broad)


def update_feed(feed_id: int, **changes: Any) -> Feed:
    init_db()
    feed = Feed.get_by_id(feed_id)
    for field in ("name", "enabled", "broad"):
        if field in changes and changes[field] is not None:
            setattr(feed, field, changes[field])
    feed.save()
    return feed


def delete_feed(feed_id: int) -> None:
    """删除订阅源及其未入库、未反馈的条目（已入库或有反馈的条目保留，用于画像学习）。"""
    init_db()
    feed = Feed.get_by_id(feed_id)
    kept = Item.select().where((Item.feed == feed) & ((Item.status == "ingested") | Item.feedback.is_null(False)))
    if kept.exists():
        feed.enabled = False
        feed.name = feed.name + "（已删除）"
        feed.save()
        Item.delete().where((Item.feed == feed) & (Item.status != "ingested") & Item.feedback.is_null(True)).execute()
    else:
        feed.delete_instance(recursive=True)


# ---------------------------------------------------------------------------
# 抓取 → 补全 → 排序 → 入库
# ---------------------------------------------------------------------------

def _entry_published(entry: Any) -> dt.datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        value = entry.get(key)
        if value:
            try:
                return dt.datetime(*value[:6])
            except (TypeError, ValueError):
                pass
    return None


def fetch_feed(feed: Feed, library_dois: dict[str, int]) -> int:
    parsed = backend.parse_feed(feed.url)
    entries = getattr(parsed, "entries", []) or []
    if getattr(parsed, "bozo", False) and not entries:
        raise ValueError(str(getattr(parsed, "bozo_exception", "解析失败")))
    feed_title = (getattr(parsed, "feed", {}) or {}).get("title")
    cutoff = dt.datetime.now() - dt.timedelta(days=retention_days())
    new = 0
    for entry in entries:
        title = strip_html(entry.get("title", ""))
        if not title:
            continue
        published = _entry_published(entry)
        if published and published < cutoff:
            continue  # 已超过保留期：不抓，免得清理后又被抓回来
        raw = ""
        if entry.get("content"):
            raw = entry["content"][0].get("value", "")
        raw = raw or entry.get("summary", "") or ""
        doi = extract_doi(entry.get("prism_doi"), entry.get("dc_identifier"), entry.get("link"), entry.get("id"), raw)
        key = f"doi:{doi}" if doi else f"title:{_norm_title(title)}"
        if Item.get_or_none(Item.dedupe_key == key):
            continue  # 同一篇文章出现在多个订阅源里只保留一条
        abstract = strip_html(raw)
        in_lib = library_dois.get(doi) if doi else None
        Item.create(
            feed=feed, guid=str(entry.get("id") or entry.get("link") or key)[:250], dedupe_key=key[:250], doi=doi,
            title=title, authors=json.dumps(split_authors(entry.get("authors") or entry.get("author") or "")),
            journal=(entry.get("prism_publicationname") or feed_title or feed.name), published=published,
            link=entry.get("link"), raw_html=raw[:50000],
            abstract=abstract if len(abstract) > 200 else None,
            abstract_source="rss" if len(abstract) > 200 else "none",
            info="abstract" if len(abstract) > 200 else "title",
            status="in_library" if in_lib else "new", in_library_pid=in_lib,
        )
        new += 1
    return new


ENRICH_RETRY_DAYS = 3      # 只有标题的条目：每隔几天再补一次（OpenAlex 的摘要往往晚几天才有）
ENRICH_MAX_ATTEMPTS = 5
ENRICH_WINDOW_DAYS = 21


def enrich_items(limit: int = 200) -> int:
    """补全摘要与作者：有 DOI 查 OpenAlex/Crossref，没有 DOI 按标题查 OpenAlex。
    只有标题的条目会在之后几天里重试；补到摘要后，未入库、未表态的条目重新打分。"""
    now = dt.datetime.now()
    retry = (Item.abstract.is_null(True) & (Item.enrich_attempts < ENRICH_MAX_ATTEMPTS)
             & (Item.last_enrich_at < now - dt.timedelta(days=ENRICH_RETRY_DAYS))
             & (Item.created_at > now - dt.timedelta(days=ENRICH_WINDOW_DAYS))
             & Item.status.in_(["new", "scored"]) & Item.feedback.is_null(True))
    query = Item.select().where(((Item.enriched == False) & Item.status.in_(["new", "scored"])) | retry).limit(limit)  # noqa: E712
    count = 0
    for item in query:
        info = backend.enrich(item.doi) if item.doi else backend.enrich_by_title(item.title)
        had_abstract = bool(item.abstract)
        if info:
            if info.get("doi") and not item.doi:
                item.doi = info["doi"]
            if info.get("abstract") and len(info["abstract"]) > len(item.abstract or "") and len(info["abstract"]) > 200:
                item.abstract, item.abstract_source, item.info = info["abstract"], info["source"], "abstract"
            if info.get("authors") and len(info["authors"]) >= len(item.authors_list()):
                item.authors = json.dumps(info["authors"][:60])
            item.journal = item.journal or info.get("journal")
        if item.abstract and not had_abstract and item.status == "scored":
            item.status = "new"  # 补到了摘要：按“有摘要”重新打分
        item.enriched = True
        item.enrich_attempts += 1
        item.last_enrich_at = now
        item.save()
        count += 1
        time.sleep(0.1)  # 对公共 API 客气一点
    return count


# ---------------------------------------------------------------------------
# 自动入库门槛的校准：两类文章（有摘要 / 仅标题）各自统计、各自调整
# ---------------------------------------------------------------------------

def _calibration_path() -> Path:
    return _data_dir / "calibration.json"


def load_calibration() -> dict:
    from . import feed_ranker

    data = {"auto": dict(feed_ranker.AUTO_THRESHOLD), "confirm": dict(feed_ranker.CONFIRM_THRESHOLD), "history": []}
    path = _calibration_path()
    if path.exists():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            data["auto"].update(stored.get("auto", {}))
            data["confirm"].update(stored.get("confirm", {}))
            data["history"] = stored.get("history", [])[-50:]
        except json.JSONDecodeError:
            pass
    return data


def auto_precision(info: str, weeks: int = 8) -> dict:
    """自动入库的文章中，你表了态的有多少是 👍（你对自动入库的文章点 👎 就是一次误判）。"""
    since = dt.datetime.now() - dt.timedelta(weeks=weeks)
    rated = list(Item.select().where((Item.ingest_mode == "auto") & (Item.info == info)
                                     & Item.feedback.is_null(False) & (Item.ingested_at > since)))
    good = sum(1 for i in rated if i.feedback > 0)
    return {"rated": len(rated), "good": good, "precision": round(good / len(rated), 3) if rated else None}


def recalibrate() -> dict:
    """准确率低于 80% 就收紧门槛，高于 90%（且样本足够）就略微放宽；两类文章分开调整。"""
    from . import feed_ranker

    data = load_calibration()
    changes = []
    for info in ("abstract", "title"):
        stats = auto_precision(info)
        current = data["auto"][info]
        if stats["rated"] >= 5 and stats["precision"] < 0.8:
            data["auto"][info] = min(0.95, round(current + 0.03, 3))
        elif stats["rated"] >= 10 and stats["precision"] >= 0.9:
            data["auto"][info] = max(feed_ranker.AUTO_THRESHOLD[info] - 0.05, round(current - 0.02, 3))
        if data["auto"][info] != current:
            changes.append({"at": dt.datetime.now().isoformat(timespec="seconds"), "info": info,
                            "from": current, "to": data["auto"][info], **stats})
    if changes:
        data["history"] = (data["history"] + changes)[-50:]
        _calibration_path().parent.mkdir(parents=True, exist_ok=True)
        _calibration_path().write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return data


def _record_llm(usage: Any, model: str) -> None:
    from .llm_usage import record

    record("文献追踪", model, usage, ref="paper_feed")


def score_items(limit: int = 400) -> int:
    from . import feed_ranker

    items = list(Item.select().where(Item.status == "new").limit(limit))
    if not items:
        return 0
    model, rb, profile = library_model()
    labelled = Item.select().where(Item.feedback.is_null(False) & Item.vector.is_null(False))
    positives = [feed_ranker.blob_to_vector(i.vector) for i in labelled if i.feedback > 0]
    negatives = [feed_ranker.blob_to_vector(i.vector) for i in labelled if i.feedback < 0]
    shortlist = []
    for item in items:
        title_only = not item.abstract
        item.info = "title" if title_only else "abstract"
        text = item.title if title_only else f"{item.title}\n{item.abstract}"
        vec = rb.embed(text)
        features = feed_ranker.compute_features(model, rb, vec=vec, text=text, authors=item.authors_list(),
                                                positives=positives, negatives=negatives, title_only=title_only)
        item.vector = feed_ranker.vector_to_blob(vec)
        item.features = feed_ranker.features_json(features)
        item.base_score = feed_ranker.base_score(features)
        item.facet_name = features.get("facet_name")
        # 综合期刊：内容离你的库太远就不送 LLM（只为省钱，不改变这篇的分数）
        prefiltered = item.feed.broad and features["dense"] < 0.05
        if item.base_score >= feed_ranker.SHORTLIST_MIN_BASE[item.info] and not prefiltered:
            shortlist.append((item, features))
        item.save()

    batch_size = int(get_config("feed_llm_batch_size") or 8)
    if shortlist:
        client = backend.llm_client()
        model_name = get_config("model_name_cheap")
        profile_text = feed_ranker.render_profile(model, profile)
        for start in range(0, len(shortlist), batch_size):
            batch = shortlist[start:start + batch_size]
            payload = [{"id": f"F{item.id}", "title": item.title, "abstract": item.abstract,
                        "authors": "; ".join(item.authors_list()[:8]), "journal": item.journal,
                        "nearest": features.get("nearest", [])} for item, features in batch]
            try:
                judged, usage = feed_ranker.judge_batch(client, model_name, profile_text, payload)
                _record_llm(usage, model_name)
            except Exception as exc:
                print(f"[paper_feed] LLM judge failed: {exc}")
                judged = {}
            for item, _ in batch:
                result = judged.get(f"F{item.id}")
                if result:
                    item.llm_score, item.llm_reason = result["score"], result["reason"]
                    item.save()

    thresholds = load_calibration()
    for item in items:
        item.score = feed_ranker.final_score(item.base_score or 0.0, item.llm_score)
        item.band = feed_ranker.assign_band(item.score, info=item.info, llm_score=item.llm_score,
                                            features=json.loads(item.features or "{}"), thresholds=thresholds)
        item.status = "scored"
        item.scored_at = dt.datetime.now()
        item.save()
    return len(items)


def ingest(item_id: int, mode: str = "manual") -> Item:
    """入库 = 进 Zotero 的 RSS_TRACK/<订阅源> 子集合，并纳入 Pageboygent 追踪。"""
    init_db()
    item = Item.get_by_id(item_id)
    if item.status == "ingested":
        return item
    root = get_config("feed_root_collection") or "RSS_TRACK"
    try:
        collection = backend.zotero_collection(root, item.feed.name.replace("（已删除）", ""))
        key = backend.zotero_add_item(collection, {
            "title": item.title, "authors": item.authors_list(), "doi": item.doi, "link": item.link,
            "abstract": item.abstract, "journal": item.journal,
            "published": item.published.strftime("%Y-%m-%d") if item.published else ""})
        backend.track_in_pageboygent(key, root)
        item.status, item.zotero_key, item.ingest_error = "ingested", key, None
        item.ingest_mode, item.ingested_at = mode, dt.datetime.now()
    except Exception as exc:
        item.ingest_error = f"{type(exc).__name__}: {exc}"
    item.save()
    return item


def auto_ingest(limit: int | None = None) -> int:
    limit = limit or int(get_config("feed_auto_ingest_per_run") or 10)
    done = 0
    for item in Item.select().where((Item.band == "auto") & (Item.status == "scored")).order_by(Item.score.desc()).limit(limit):
        if ingest(item.id, mode="auto").status == "ingested":
            done += 1
    return done


def set_feedback(item_id: int, value: int) -> Item:
    """👍 = +1，👎 = -1，0 = 撤销。确认入库另走 confirm_ingest（同时记 +1）。"""
    init_db()
    item = Item.get_by_id(item_id)
    item.feedback = value if value in (1, -1) else None
    item.feedback_at = dt.datetime.now() if item.feedback else None
    if value == -1 and item.status == "scored":
        item.status = "dismissed"
    elif value != -1 and item.status == "dismissed":
        item.status = "scored"
    item.save()
    return item


def confirm_ingest(item_id: int) -> Item:
    set_feedback(item_id, 1)
    return ingest(item_id, mode="manual")


def mark_read(item_id: int) -> None:
    init_db()
    Item.update(read_at=dt.datetime.now()).where((Item.id == item_id) & Item.read_at.is_null(True)).execute()


# ---------------------------------------------------------------------------
# 周报
# ---------------------------------------------------------------------------

def iso_week(day: dt.date | dt.datetime) -> str:
    year, week, _ = day.isocalendar()
    return f"{year}-W{week:02d}"


def week_range(week: str) -> tuple[dt.datetime, dt.datetime]:
    year, num = week.split("-W")
    start = dt.datetime.fromisocalendar(int(year), int(num), 1)
    return start, start + dt.timedelta(days=7)


REPORT_SYSTEM = """
你是一位地球化学、地球物理、宇宙化学与行星科学方向研究者的文献助理，正在写本周的新文献周报开头。
只依据下面列出的文章（标题、摘要、推荐理由）写作，不要提及列表之外的文章或你记忆中的研究；用 [F编号] 指代文章。
输出 Markdown：先用 3-5 句话概括本周值得关注的内容与趋势（例如几篇文章不约而同地讨论了同一问题），
再按兴趣方向各写一句要点。总长度 300 字以内。
""".strip()


def generate_report(week: str | None = None, *, use_llm: bool = True) -> Report:
    """生成（或重新生成）某一周的周报。默认是上一周；也可以传本周，生成“截至目前”的版本。"""
    init_db()
    week = week or iso_week(dt.date.today() - dt.timedelta(days=7))
    start, end = week_range(week)
    items = list(Item.select().where((Item.created_at >= start) & (Item.created_at < end)))
    scored = [i for i in items if i.score is not None]
    auto = sorted([i for i in scored if i.status == "ingested" and i.ingest_mode == "auto"], key=lambda i: -i.score)
    confirm = sorted([i for i in scored if i.band == "confirm" and i.status == "scored"], key=lambda i: -i.score)
    manual = [i for i in scored if i.status == "ingested" and i.ingest_mode == "manual"]
    # 边界样本：未表态、分数最接近“确认线”的两篇（探索 + 收集最有用的反馈）
    confirm_line = load_calibration()["confirm"]
    pool = [i for i in scored if i.feedback is None and i.status == "scored" and i not in confirm[:15]]
    boundary = sorted(pool, key=lambda i: abs(i.score - confirm_line.get(i.info, 0.5)))[:2]
    for item in boundary:
        Item.update(exploration=True).where(Item.id == item.id).execute()
    picks = (auto + manual + confirm)[:30]
    groups: dict[str, list[int]] = {}
    for item in picks:
        groups.setdefault(item.facet_name or "其他", []).append(item.id)
    by_feed: dict[str, dict] = {}
    for item in items:
        stats = by_feed.setdefault(item.feed.name, {"items": 0, "picked": 0})
        stats["items"] += 1
        stats["picked"] += int(item.band in ("auto", "confirm"))
    data = {
        "week": week, "range": [start.strftime("%m-%d"), (end - dt.timedelta(days=1)).strftime("%m-%d")],
        "stats": {"items": len(items), "scored": len(scored), "auto": len(auto), "manual": len(manual),
                  "confirm": len(confirm), "in_library": sum(1 for i in items if i.status == "in_library")},
        "groups": groups, "confirm": [i.id for i in confirm[:15]], "boundary": [i.id for i in boundary],
        "by_feed": by_feed,
    }
    overview = ""
    if use_llm and picks:
        lines = [f"[F{i.id}] {i.title}（方向：{i.facet_name or '其他'}；推荐理由：{i.llm_reason or '无'}）\n摘要：{(i.abstract or '')[:600]}"
                 for i in picks[:20]]
        try:
            client = backend.llm_client()
            model_name = get_config("model_name_cheap")
            response = client.chat.completions.create(
                model=model_name, temperature=0.3, max_tokens=1500, stream=False,
                extra_body={"thinking": {"type": "disabled"}},
                messages=[{"role": "system", "content": REPORT_SYSTEM}, {"role": "user", "content": "\n\n".join(lines)}])
            _record_llm(getattr(response, "usage", None), model_name)
            overview = (response.choices[0].message.content or "").strip()
            # 只允许引用本周列表里的文章
            valid = {f"F{i.id}" for i in picks}
            overview = re.sub(r"\[(F\d+)\]", lambda m: m.group(0) if m.group(1) in valid else "", overview)
        except Exception as exc:
            print(f"[paper_feed] report overview failed: {exc}")
    report, created = Report.get_or_create(week=week, defaults={"overview": overview, "data": json.dumps(data)})
    if not created:
        report.overview, report.data, report.generated_at, report.read_at = overview, json.dumps(data), dt.datetime.now(), None
        report.save()
    return report


# ---------------------------------------------------------------------------
# 心跳与状态
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {"running": False, "last_run_at": None, "last_result": None, "last_error": None}


def run_once(force: bool = False) -> dict:
    """抓取 → 补全 → 排序 → 自动入库 → 需要时生成上周周报。由 heartbeat 定时调用，也可在页面上手动触发。"""
    init_db()
    if not _run_lock.acquire(blocking=False):
        return {"skipped": "已在运行"}
    try:
        _state["running"] = True
        library_dois = backend.library_doi_map()
        result: dict[str, Any] = {"fetched": 0, "errors": []}
        for feed in Feed.select().where(Feed.enabled == True):  # noqa: E712
            try:
                new = fetch_feed(feed, library_dois)
                feed.last_new_count, feed.last_error = new, None
                result["fetched"] += new
            except Exception as exc:
                feed.last_error = f"{type(exc).__name__}: {exc}"
                result["errors"].append(f"{feed.name}: {exc}")
            feed.last_fetch_at = dt.datetime.now()
            feed.save()
        result["enriched"] = enrich_items()
        result["thresholds"] = recalibrate()["auto"]
        result["scored"] = score_items()
        result["auto_ingested"] = auto_ingest()
        result["pruned"] = prune_old_items()["deleted"]
        last_week = iso_week(dt.date.today() - dt.timedelta(days=7))
        if not Report.get_or_none(Report.week == last_week) and Item.select().where(
                (Item.created_at >= week_range(last_week)[0]) & (Item.created_at < week_range(last_week)[1])).exists():
            generate_report(last_week)
            result["report"] = last_week
        _state.update(last_run_at=dt.datetime.now().isoformat(timespec="seconds"), last_result=result, last_error=None)
        return result
    except Exception as exc:
        _state["last_error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _state["running"] = False
        _run_lock.release()


def retention_days() -> int:
    return max(7, int(get_config("feed_retention_days") or 60))


def prune_old_items() -> dict:
    """自动清理过旧的 RSS 条目（与传统阅读器不同：不长期堆积）。
    已入库、有反馈的条目是画像学习的样本，不删，只清掉原始 HTML。"""
    init_db()
    cutoff = dt.datetime.now() - dt.timedelta(days=retention_days())
    keep = (Item.status == "ingested") | Item.feedback.is_null(False)
    deleted = Item.delete().where((Item.created_at < cutoff) & ~keep).execute()
    slimmed = Item.update(raw_html=None).where((Item.created_at < cutoff) & keep & Item.raw_html.is_null(False)).execute()
    return {"deleted": deleted, "slimmed": slimmed}


def heart_beat() -> None:
    init_db()
    interval = float(get_config("feed_fetch_interval_hours") or 6)
    last = Feed.select(peewee.fn.MAX(Feed.last_fetch_at)).where(Feed.enabled == True).scalar()  # noqa: E712
    if not Feed.select().where(Feed.enabled == True).exists():  # noqa: E712
        return
    if last and (dt.datetime.now() - last).total_seconds() < interval * 3600:
        return
    run_once()


def status() -> dict:
    init_db()
    last = Feed.select(peewee.fn.MAX(Feed.last_fetch_at)).scalar()
    interval = float(get_config("feed_fetch_interval_hours") or 6)
    return {
        **_state,
        "last_fetch_at": last.strftime("%Y-%m-%d %H:%M") if last else None,
        "next_fetch_at": (last + dt.timedelta(hours=interval)).strftime("%Y-%m-%d %H:%M") if last else "添加订阅源后开始",
        "interval_hours": interval,
    }


def precision_stats(weeks: int = 8) -> dict:
    """每周推荐质量（推荐且你表了态的文章中 👍 的比例），有摘要 / 仅标题分开统计；附当前自动入库门槛。"""
    init_db()
    today = dt.date.today()
    series: dict[str, list[dict]] = {"abstract": [], "title": []}
    for back in range(weeks - 1, -1, -1):
        week = iso_week(today - dt.timedelta(days=7 * back))
        start, end = week_range(week)
        for info in series:
            rated = list(Item.select().where((Item.created_at >= start) & (Item.created_at < end) & (Item.info == info)
                                             & Item.band.in_(["auto", "confirm"]) & Item.feedback.is_null(False)))
            good = sum(1 for i in rated if i.feedback > 0)
            series[info].append({"week": week, "rated": len(rated), "good": good,
                                 "precision": round(good / len(rated), 3) if rated else None})
    calibration = load_calibration()
    return {"series": series, "auto_threshold": calibration["auto"], "confirm_threshold": calibration["confirm"],
            "auto_precision": {info: auto_precision(info) for info in series}}
