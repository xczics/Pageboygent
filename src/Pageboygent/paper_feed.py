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

from .i18n import t, answer_language_name
from .write_queue import stop_point
from .config import get_config, research_field

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
    """一篇 RSS 文章。后台任务（补全、打分、重新评估）和页面操作（入库、反馈）会同时改同一条记录：
    只保存改动过的字段（only_save_dirty），后台的状态变化一律带条件更新（见 _save_if），
    避免后台用旧数据把你刚做的入库 / 反馈覆盖掉。"""

    class Meta:
        only_save_dirty = True

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
    llm_profile = peewee.CharField(null=True)         # 做 LLM 判断时画像文字的摘要：画像变了才考虑重判
    rerank_note = peewee.CharField(null=True)         # 重新评估后档位变化的原因（页面上显示）
    rescored_at = peewee.DateTimeField(null=True)

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


_FEED_TITLE_NOISE = re.compile(r"^(?:ScienceDirect Publication|Wiley|AAAS|Recent articles published in)\s*:\s*|\s*:\s*Table of Contents$|\s*-\s*latest papers$", re.I)


def _openalex_auth() -> dict:
    """OpenAlex 自 2026 年 2 月起要求 API key（免费注册）；不带 key 时每天的额度很小。"""
    key = get_config("openalex_api_key")
    return {"api_key": key} if key else {}


def library_match(library: dict, doi: str | None, title: str | None) -> int | None:
    """文献库里是否已有这篇：先比 DOI；没有 DOI（或 DOI 对不上）时比规范化后的完整标题（只认精确一致）。"""
    from .storage import normalize_doi

    doi = normalize_doi(doi)
    if doi and doi in library:
        return library[doi]
    key = _norm_title(title or "")
    return library.get(f"title:{key}") if len(key) >= 30 else None


def clean_journal(name: str | None) -> str | None:
    """RSS 源标题当期刊名时去掉平台前后缀，例如 “ScienceDirect Publication: Icarus” → “Icarus”。"""
    if not name:
        return None
    text = str(name).strip()
    for _ in range(2):
        text = _FEED_TITLE_NOISE.sub("", text).strip()
    return text or None


def _looks_like_feed_title(name: str | None) -> bool:
    return bool(name) and clean_journal(name) != str(name).strip()


def _norm_title(title: str) -> str:
    return re.sub(r"[\W_]+", "", (title or "").lower())[:200]


_JUNK_AUTHOR = re.compile(r"https?://|ror\.org|Department|University|Institute", re.I)


def _looks_like_junk(name: str) -> bool:
    """RSS 里粘在一起的作者串（例如 PNAS/Science 的“名字名字名字+机构+网址”）：无法可靠拆分。"""
    return len(name) > 60 or bool(_JUNK_AUTHOR.search(name)) or len(re.findall(r"[a-z\u00e0-\u017f][A-Z]", name)) >= 2


def split_authors(value: Any) -> list[str]:
    """把 RSS 里的作者字段拆成名单。拆出来的有任何一项像是粘连的乱串，就整体放弃（返回空），交给 OpenAlex 补全。"""
    if isinstance(value, list):
        raw = [a.get("name", "") if isinstance(a, dict) else str(a) for a in value]
    else:
        raw = [str(value or "")]
    names: list[str] = []
    for chunk in raw:
        text = strip_html(chunk)
        # Wiley：“N. Agarwal, \nT. A. Smith, \n…”；其他：分号、and、&、“, 名 姓”
        for part in re.split(r"\s*\n\s*|;|\band\b|&|,(?=\s*[A-Z][a-z]+\s+[A-Z])", text):
            part = part.strip(" ,.")
            if part and len(part) > 1:
                names.append(part)
    if any(_looks_like_junk(n) for n in names):
        return []
    return names[:60]


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
                             params={"select": "title,publication_date,authorships,abstract_inverted_index,primary_location",
                                     **_openalex_auth()},
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
                             params={"search": title[:300], "per-page": 3, **_openalex_auth(),
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
        # 新文章 OpenAlex 往往要晚几天才收录；出版商上线时就向 Crossref 登记了 DOI
        try:
            r = requests.get("https://api.crossref.org/works", timeout=20, headers={"User-Agent": self.user_agent},
                             params={"query.bibliographic": title[:300], "rows": 3,
                                     "select": "DOI,title,container-title,author,abstract,issued"})
            if not r.ok:
                return None
            for work in r.json().get("message", {}).get("items", []):
                ratio = difflib.SequenceMatcher(None, _norm_title(title), _norm_title((work.get("title") or [""])[0])).ratio()
                if ratio < 0.92:
                    continue
                abstract = strip_html(work.get("abstract") or "")
                parts = ((work.get("issued") or {}).get("date-parts") or [[None]])[0]
                published = "-".join(f"{v:02d}" if i else str(v) for i, v in enumerate(parts) if v) or None
                return {"doi": (work.get("DOI") or "").lower() or None, "abstract": abstract,
                        "source": "crossref" if abstract else "none",
                        "authors": [" ".join(filter(None, [a.get("given"), a.get("family")])) for a in work.get("author", [])],
                        "journal": (work.get("container-title") or [None])[0], "published": published}
        except Exception as exc:
            print(f"[paper_feed] Crossref title search failed: {exc}")
        return None

    def llm_client(self) -> Any:
        from openai import OpenAI

        return OpenAI(api_key=get_config("ds_api_key"), base_url=get_config("ds_url"))

    def library_doi_map(self) -> dict[str, int]:
        from .storage import Paper

        from .storage import normalize_doi

        out = {}
        for p in Paper.select(Paper.id, Paper.doi, Paper.title):
            doi = normalize_doi(p.doi)
            if doi:            # 没有 DOI 的不按 DOI 查重
                out.setdefault(doi, int(p.id))
            title = _norm_title(p.title or "")
            if len(title) >= 30:   # 标题太短（如 “Introduction”）不拿来查重
                out.setdefault(f"title:{title}", int(p.id))
        return out

    # ---- Zotero ----
    def zotero_collection(self, root: str, child: str | None = None) -> str:
        """确保 <root>/<child> 子集合（child 为空时就是 <root>）存在，返回其 key。"""
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
        if not child:
            if not root_key:
                raise RuntimeError(t("feed:error.create_collection", path=root))
            return root_key
        child_key = _find(child, root_key)
        if child_key is None:
            _perform_local_write(f"create collection {root}/{child}", zot.create_collections,
                                 [{"name": child, "parentCollection": root_key}])
            child_key = _find(child, root_key)
        if not child_key:
            raise RuntimeError(t("feed:error.create_collection", path=f"{root}/{child}"))
        return child_key

    def zotero_add_item(self, collection_key: str, item: dict) -> str:
        """在集合中新建条目（已有同 DOI 条目则直接加入集合），返回条目 key。"""
        from .zt_tools import _perform_local_write, zot

        if item.get("doi"):
            for existing in zot.everything(zot.items(q=item["doi"], qmode="everything")):
                if str(existing["data"].get("DOI", "")).lower() == item["doi"].lower():
                    _perform_local_write("add to collection", zot.addto_collection, collection_key, existing)
                    return existing["data"]["key"]
        creators = author_creators(item.get("authors", []))
        data = {"itemType": "journalArticle", "title": item["title"], "creators": creators, "tags": [],
                "collections": [collection_key], "relations": {}, "DOI": item.get("doi") or "",
                "url": item.get("link") or "", "abstractNote": item.get("abstract") or "",
                "publicationTitle": clean_journal(item.get("journal")) or "", "date": item.get("published") or ""}
        result = _perform_local_write("create rss item", zot.create_items, [data])
        created = (result or {}).get("successful") or (result or {}).get("success") or {}
        first = next(iter(created.values()), None) if isinstance(created, dict) else None
        key = first.get("key") if isinstance(first, dict) else first
        if not key:
            raise RuntimeError(f"Zotero 未返回新条目的 key：{result}")
        return str(key)

    def zotero_patch_item(self, key: str, fields: dict) -> bool:
        """按“只填空、改明显错误”的规则更新一个 Zotero 条目。有改动返回 True。"""
        from .zt_tools import _perform_local_write, zot

        data = zot.item(key)["data"]
        changes = {}
        for name, value in fields.items():
            if not value:
                continue
            current = data.get(name)
            if name == "creators":
                if _bad_creators(current or []):
                    changes[name] = value
            elif name == "publicationTitle":
                if not current or _looks_like_feed_title(current):
                    changes[name] = value
            elif not current:
                changes[name] = value
        if not changes:
            return False
        data.update(changes)
        _perform_local_write(f"sync rss item {key}", zot.update_item, data)
        return True

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
            from pyzotero.zotero_errors import ResourceNotFoundError as ResourceNotFound

            from .zt_tools import get_info
        except Exception as exc:
            print(f"[paper_feed] cannot read creators from Zotero, falling back to first authors: {exc}")
            missing = []
        not_found, failures = 0, 0
        for paper in missing:
            try:
                creators, = get_info(paper["key"], ["creators"])
            except ResourceNotFound:
                # Zotero 里已经没有这个条目（被删或移走）：记为空，之后不再反复去查；用第一作者
                creators = []
                not_found += 1
            except Exception as exc:
                failures += 1
                if failures >= 3:   # 连续出错多半是 Zotero 没开：这次先用第一作者，下次再补
                    print(f"[paper_feed] cannot read creators from Zotero, falling back to first authors: {exc}")
                    break
                continue
            failures = 0
            names = [" ".join(filter(None, [c.get("firstName"), c.get("lastName")])) or c.get("name", "")
                     for c in (creators or []) if c.get("creatorType") == "author"]
            LibraryAuthor.replace(pid=paper["pid"], key=paper["key"], authors=json.dumps(names)).execute()
        if not_found:
            print(f"[paper_feed] {not_found} library papers are no longer in Zotero; using first authors for them")
        cached = {row.pid: row for row in LibraryAuthor.select()}
    for paper in papers:
        row = cached.get(paper["pid"])
        names = json.loads(row.authors) if row else []
        out[paper["pid"]] = names or ([paper["first_author"]] if paper.get("first_author") else [])
    return out


_model_cache = None
_model_signature = None
ranker_backend = None  # 测试时替换为伪造的 RankerBackend


_model_build_lock = threading.Lock()


def library_model(*, describe: bool = True):
    """库内排序模型（库内文献或画像变化时重建）。describe=True 时顺便让 AI 为新组提炼说明、为有新文献的组提建议。
    构建同一时间只进行一次：画像页的“分组”与后台打分同时触发时，后到的等前一个建完直接复用。"""
    global _model_cache, _model_signature
    from . import feed_ranker

    rb = ranker_backend or feed_ranker.RankerBackend()
    with _model_build_lock:
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
        raise ValueError(t("feed:error.parse", error=getattr(parsed, "bozo_exception", "")))
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
        raise ValueError(t("RSS 地址需要以 http:// 或 https:// 开头"))
    if Feed.get_or_none(Feed.url == url):
        raise ValueError(t("这个 RSS 已经订阅过了"))
    name = (name or "").strip() or preview_feed(url)["title"] or url
    if "/" in name:
        raise ValueError(t("名称里不能有 “/”（会用作 Zotero 子集合名）"))
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
        raise ValueError(str(getattr(parsed, "bozo_exception", t("解析失败"))))
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
        in_lib = library_match(library_dois, doi, title)
        Item.create(
            feed=feed, guid=str(entry.get("id") or entry.get("link") or key)[:250], dedupe_key=key[:250], doi=doi,
            title=title, authors=json.dumps(split_authors(entry.get("authors") or entry.get("author") or "")),
            journal=clean_journal(entry.get("prism_publicationname") or feed_title or feed.name), published=published,
            link=entry.get("link"), raw_html=raw[:50000],
            abstract=abstract if len(abstract) > 200 else None,
            abstract_source="rss" if len(abstract) > 200 else "none",
            info="abstract" if len(abstract) > 200 else "title",
            status="in_library" if in_lib else "new", in_library_pid=in_lib,
            read_at=dt.datetime.now() if in_lib else None,   # 库里已有的不算未读
        )
        new += 1
    return new


ENRICH_RETRY_DAYS = 3      # 只有标题的条目：每隔几天再补一次（OpenAlex 的摘要往往晚几天才有）
ENRICH_MAX_ATTEMPTS = 5
ENRICH_WINDOW_DAYS = 21


ENRICH_BUDGET_SECONDS = 180   # 每次运行最多花在补全上的时间：剩下的下次接着补，不耽误打分


def _save_if(item: "Item", condition) -> bool:
    """只在数据库里这条记录仍满足 condition 时，才写入这次改动过的字段（后台任务用：你在此期间入库或表态过就跳过）。"""
    fields = item.dirty_fields
    if not fields:
        return True
    data = {field: getattr(item, field.name) for field in fields}
    updated = Item.update(data).where((Item.id == item.id) & condition).execute()
    item._dirty.clear()
    return bool(updated)


def enrich_one(item: "Item", now: dt.datetime | None = None) -> bool:
    """补全一篇：DOI、摘要、作者、正式刊名、发表日期。只填缺的，或替换明显不对的（RSS 源标题当刊名）。返回是否有变化。"""
    now = now or dt.datetime.now()
    info = backend.enrich(item.doi) if item.doi else backend.enrich_by_title(item.title)
    had_abstract = bool(item.abstract)
    before = (item.doi, item.abstract, item.authors, item.journal, item.published)
    if info:
        if info.get("doi") and not item.doi:
            item.doi = info["doi"]
        if info.get("abstract") and len(info["abstract"]) > len(item.abstract or "") and len(info["abstract"]) > 200:
            item.abstract, item.abstract_source, item.info = info["abstract"], info["source"], "abstract"
        if info.get("authors") and len(info["authors"]) >= len(item.authors_list()):
            item.authors = json.dumps(info["authors"][:60])
        if info.get("journal") and (not item.journal or _looks_like_feed_title(item.journal)):
            item.journal = info["journal"]
        if not item.published and info.get("published"):
            try:
                item.published = dt.datetime.fromisoformat(str(info["published"])[:10])
            except ValueError:
                pass
    rescore = bool(item.abstract and not had_abstract and item.status == "scored")
    item.enriched = True
    item.enrich_attempts += 1
    item.last_enrich_at = now
    item.save()
    if rescore:   # 补到了摘要：按“有摘要”重新打分（只在它仍是“已评分”时；你刚入库或表态过就不动）
        Item.update(status="new").where((Item.id == item.id) & (Item.status == "scored")).execute()
    return before != (item.doi, item.abstract, item.authors, item.journal, item.published)


def enrich_items(limit: int = 200, budget_seconds: float | None = None) -> int:
    """补全摘要与作者：有 DOI 查 OpenAlex/Crossref，没有 DOI 按标题查 OpenAlex。
    只有标题的条目会在之后几天里重试；补到摘要后，未入库、未表态的条目重新打分。
    补全逐篇访问外部 API、比较慢，每次运行限时；没补到的先按“仅标题”打分，下次补到摘要后自动重打。"""
    now = dt.datetime.now()
    deadline = time.monotonic() + (ENRICH_BUDGET_SECONDS if budget_seconds is None else budget_seconds)
    missing = (Item.abstract.is_null(True) | (Item.authors == "[]") | Item.authors.is_null(True)
               | (Item.doi.is_null(True) & (Item.status == "ingested")))
    retry = (missing & (Item.enrich_attempts < ENRICH_MAX_ATTEMPTS)
             & (Item.last_enrich_at < now - dt.timedelta(days=ENRICH_RETRY_DAYS))
             & (Item.created_at > now - dt.timedelta(days=ENRICH_WINDOW_DAYS))
             & (Item.status.in_(["new", "scored"]) & Item.feedback.is_null(True) | (Item.status == "ingested")))
    query = Item.select().where(((Item.enriched == False) & Item.status.in_(["new", "scored"])) | retry).limit(limit)  # noqa: E712
    count = 0
    todo = list(query)
    # 先把结果取完再逐条处理：边遍历未结束的查询边写入，另一线程在此期间写过库时 SQLite 会直接报 database is locked
    for n, item in enumerate(todo, start=1):
        stop_point()
        if time.monotonic() > deadline:
            break
        _state["progress"] = f"{n}/{len(todo)}"
        enrich_one(item, now)
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


def feedback_vectors(rb) -> tuple[list, list]:
    """正例：你 👍 的、你手动入库或确认入库的文章；负例：你 👎 的。自动入库的不算（不拿系统自己的选择训练自己）。
    还没算过向量的（例如抓到后立刻手动入库的）现场补算并存下。"""
    from . import feed_ranker

    labelled = list(Item.select().where(Item.feedback.is_null(False) |
                                        ((Item.status == "ingested") & (Item.ingest_mode != "auto"))))
    positives, negatives = [], []
    for item in labelled:
        if item.vector is None:
            text = item.title if not item.abstract else f"{item.title}\n{item.abstract}"
            item.vector = feed_ranker.vector_to_blob(rb.embed(text))
            item.save()
        vec = feed_ranker.blob_to_vector(item.vector)
        if (item.feedback or 0) < 0:
            negatives.append(vec)
        else:
            positives.append(vec)
    return positives, negatives


def _item_text(item: "Item") -> str:
    return item.title if not item.abstract else f"{item.title}\n{item.abstract}"


def _profile_digest(profile_text: str) -> str:
    return hashlib.md5(profile_text.encode()).hexdigest()[:12]


def _judge(pairs: list[tuple], model, profile: dict) -> int:
    """对 [(item, features)] 做 LLM 判断，写回 llm_score / llm_reason / llm_profile。返回判断成功的篇数。"""
    from . import feed_ranker

    if not pairs:
        return 0
    client = backend.llm_client()
    model_name = get_config("model_name_cheap")
    profile_text = feed_ranker.render_profile(model, profile)
    digest = _profile_digest(profile_text)
    batch_size = int(get_config("feed_llm_batch_size") or 8)
    judged_count = 0
    for start in range(0, len(pairs), batch_size):
        stop_point()
        _state["progress"] = t("feed:progress.llm", done=min(start + batch_size, len(pairs)), total=len(pairs))
        batch = pairs[start:start + batch_size]
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
                item.llm_score, item.llm_reason, item.llm_profile = result["score"], result["reason"], digest
                item.save()
                judged_count += 1
    return judged_count


def score_items(limit: int = 400) -> int:
    from . import feed_ranker

    items = list(Item.select().where(Item.status == "new").limit(limit))
    if not items:
        return 0
    _state["progress"] = t("准备库内模型")
    model, rb, profile = library_model()
    positives, negatives = feedback_vectors(rb)
    shortlist = []
    for n, item in enumerate(items, start=1):
        stop_point()
        _state["progress"] = t("feed:progress.similarity", done=n, total=len(items))
        title_only = not item.abstract
        item.info = "title" if title_only else "abstract"
        vec = rb.embed(_item_text(item))
        features = feed_ranker.compute_features(model, rb, vec=vec, text=_item_text(item), authors=item.authors_list(),
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
    _judge(shortlist, model, profile)

    thresholds = load_calibration()
    for item in items:
        item.score = feed_ranker.final_score(item.base_score or 0.0, item.llm_score)
        item.band = feed_ranker.assign_band(item.score, info=item.info, llm_score=item.llm_score,
                                            features=json.loads(item.features or "{}"), thresholds=thresholds)
        item.status = "scored"
        item.scored_at = dt.datetime.now()
        _save_if(item, Item.status == "new")      # 打分期间你已入库 / 表态：保留你的操作
    return len(items)


RERANK_NEAR = 0.10        # 离档位门槛这么近、且画像已变的文章，才重做 LLM 判断
RERANK_MAX_LLM = 40       # 每次重新评估最多重判 / 补判多少篇（控制花费）


def rescore_pending() -> dict:
    """重新评估还没处理的文章（保留期内、待确认或低分、你还没表态）：
    - 基础分每次都按当前的画像、文献库和你的反馈重算（用存好的向量，几乎不花时间）；
    - 新进入短名单、还没做过 LLM 判断的补做；画像文字变了且分数离门槛不远的重做；其余沿用原判断；
    - 重新划分档位，但最多升到“待确认”：老文章因你最近的反馈变得相关，让你看一眼再入库，
      也不掺进“自动入库准确率”的统计。档位变化的原因写在 rerank_note 里。"""
    from . import feed_ranker

    cutoff = dt.datetime.now() - dt.timedelta(days=retention_days())
    pending = list(Item.select().where((Item.status == "scored") & Item.band.in_(["confirm", "low"])
                                       & Item.feedback.is_null(True) & Item.vector.is_null(False)
                                       & (Item.created_at > cutoff)))
    if not pending:
        return {"rescored": 0, "judged": 0, "up": 0, "down": 0}
    _state["progress"] = t("准备库内模型")
    model, rb, profile = library_model(describe=False)
    positives, negatives = feedback_vectors(rb)
    digest = _profile_digest(feed_ranker.render_profile(model, profile))
    thresholds = load_calibration()
    to_judge, previous = [], {}
    for n, item in enumerate(pending, start=1):
        stop_point()
        _state["progress"] = t("feed:progress.rescore", done=n, total=len(pending))
        old_features = json.loads(item.features or "{}")
        previous[item.id] = (item.band, old_features)
        features = feed_ranker.compute_features(model, rb, vec=feed_ranker.blob_to_vector(item.vector),
                                                text=_item_text(item), authors=item.authors_list(),
                                                positives=positives, negatives=negatives, title_only=item.info == "title")
        item.features = feed_ranker.features_json(features)
        item.base_score = feed_ranker.base_score(features)
        item.facet_name = features.get("facet_name")
        shortlisted = item.base_score >= feed_ranker.SHORTLIST_MIN_BASE[item.info] and not (
            item.feed.broad and features["dense"] < 0.05)
        if item.llm_score is None and shortlisted:
            to_judge.append((item, features))
        elif item.llm_score is not None and item.llm_profile != digest:
            score = feed_ranker.final_score(item.base_score, item.llm_score)
            gates = [thresholds["confirm"][item.info], thresholds["auto"][item.info]]
            if any(abs(score - gate) < RERANK_NEAR for gate in gates):
                to_judge.append((item, features))
    to_judge.sort(key=lambda pair: -pair[0].base_score)
    judged = _judge(to_judge[:RERANK_MAX_LLM], model, profile)

    up = down = 0
    order = {"low": 0, "confirm": 1, "auto": 2}
    for item in pending:
        old_band, old_features = previous[item.id]
        item.score = feed_ranker.final_score(item.base_score or 0.0, item.llm_score)
        band = feed_ranker.assign_band(item.score, info=item.info, llm_score=item.llm_score,
                                       features=json.loads(item.features or "{}"), thresholds=thresholds)
        band = "confirm" if band == "auto" else band          # 重新评估最多升到“待确认”
        if band != old_band:
            new_features = json.loads(item.features or "{}")
            rose = order[band] > order[old_band]
            up, down = up + rose, down + (not rose)
            if abs(new_features.get("feedback", 0) - old_features.get("feedback", 0)) >= 0.1:
                cause = "你最近的反馈"
            elif abs(new_features.get("dense", 0) - old_features.get("dense", 0)) >= 0.1:
                cause = "画像或文献库的变化"
            else:
                cause = "重新评估"
            item.rerank_note = f"因{cause}{'上升' if rose else '下降'}"
            item.band = band
        item.rescored_at = dt.datetime.now()
        _save_if(item, (Item.status == "scored") & Item.feedback.is_null(True))
    return {"rescored": len(pending), "judged": judged, "up": up, "down": down}


def ingest(item_id: int, mode: str = "manual") -> Item:
    """入库 = 进 Zotero 的 RSS_TRACK/<订阅源> 子集合，并纳入 Pageboygent 追踪。"""
    init_db()
    item = Item.get_by_id(item_id)
    if item.status == "ingested":
        return item
    # 入库前再按 DOI 查一次重：抓取之后文献库可能已经有了这篇（没有 DOI 的不查）
    from .storage import normalize_doi

    doi = normalize_doi(item.doi)
    in_library = library_match(backend.library_doi_map(), doi, item.title)
    if in_library:
        item.status, item.in_library_pid, item.ingest_error = "in_library", in_library, None
        item.read_at = item.read_at or dt.datetime.now()
        item.save()
        return item
    if not item.doi or not item.authors_list():
        try:
            enrich_one(item)   # 抓到后还没轮到补全：入库前当场补，免得 Zotero 里只有一个标题
        except Exception as exc:
            print(f"[paper_feed] enrich before ingest failed for item {item.id}: {exc}")
        doi = normalize_doi(item.doi)
        in_library = library_match(backend.library_doi_map(), doi, item.title)
        if in_library:
            item.status, item.in_library_pid, item.ingest_error = "in_library", in_library, None
            item.read_at = item.read_at or dt.datetime.now()
            item.save()
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
        item.read_at = item.read_at or dt.datetime.now()   # 入库了就不必再提醒你读：标为已读
        item.ingest_mode, item.ingested_at = mode, dt.datetime.now()
    except Exception as exc:
        item.ingest_error = f"{type(exc).__name__}: {exc}"
    item.save()
    return item


def zotero_fields(item: "Item") -> dict:
    """Pageboygent 里这篇文章的著录信息，按 Zotero 字段名。"""
    return {"DOI": item.doi or "", "abstractNote": item.abstract or "", "publicationTitle": clean_journal(item.journal) or "",
            "date": item.published.strftime("%Y-%m-%d") if item.published else "", "url": item.link or "",
            "creators": author_creators(item.authors_list())}


def author_creators(authors: list[str]) -> list[dict]:
    creators = []
    for name in authors[:50]:
        if "," in name:
            last, first = [part.strip() for part in name.split(",", 1)]
        else:
            parts = name.strip().rsplit(" ", 1)
            first, last = (parts[0], parts[1]) if len(parts) > 1 else ("", parts[0])
        creators.append({"creatorType": "author", "firstName": first, "lastName": last})
    return creators


def _bad_creators(creators: list[dict]) -> bool:
    """Zotero 里的作者明显不对：没有作者，或者有“作者”是纯数字 / 网址 / 粘连串（来自旧的 RSS 作者解析）。"""
    if not creators:
        return True
    for c in creators:
        name = " ".join(filter(None, [c.get("firstName"), c.get("lastName"), c.get("name")]))
        if re.fullmatch(r"[\d\s-]+", name or "") or _looks_like_junk(name or ""):
            return True
    return False


def sync_ingested_to_zotero(limit: int = 100) -> int:
    """把补全后的信息写回已入库的 Zotero 条目：只填 Zotero 里空着的字段，或改正明显错误的
    （刊名是 RSS 源标题、作者是乱码）。你在 Zotero 里改过的内容不动。返回更新的条目数。"""
    updated = 0
    since = dt.datetime.now() - dt.timedelta(days=ENRICH_WINDOW_DAYS * 2)
    for item in list(Item.select().where((Item.status == "ingested") & Item.zotero_key.is_null(False)
                                         & (Item.ingested_at > since)).limit(limit)):
        if backend.zotero_patch_item(item.zotero_key, zotero_fields(item)):
            updated += 1
    return updated


def auto_ingest(limit: int | None = None) -> int:
    limit = limit or int(get_config("feed_auto_ingest_per_run") or 10)
    done = 0
    for item in list(Item.select().where((Item.band == "auto") & (Item.status == "scored")).order_by(Item.score.desc()).limit(limit)):
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


def mark_settled_read() -> int:
    """已入库、库中已有的条目一律算已读（含以前留下的），不占未读数。返回本次标注的条数。"""
    init_db()
    return Item.update(read_at=dt.datetime.now()).where(
        Item.read_at.is_null(True) & Item.status.in_(["ingested", "in_library"])).execute()


def mark_read(item_id: int) -> None:
    init_db()
    Item.update(read_at=dt.datetime.now()).where((Item.id == item_id) & Item.read_at.is_null(True)).execute()


def _unread_low(feed_id: int | None = None):
    query = (Item.read_at.is_null(True)) & (Item.band == "low") & (Item.status == "scored")
    if feed_id:
        query &= Item.feed == int(feed_id)
    return query


def count_unread_low(feed_id: int | None = None) -> int:
    init_db()
    return Item.select().where(_unread_low(feed_id)).count()


def mark_low_read(feed_id: int | None = None) -> int:
    """把（某个订阅源或全部的）未读低分条目一次标为已读。只改“已读”，不算 👎、不影响画像。返回标注的条数。"""
    init_db()
    return Item.update(read_at=dt.datetime.now()).where(_unread_low(feed_id)).execute()


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
你是一位〔研究领域〕方向研究者的文献助理，正在写本周的新文献周报开头。
只依据下面列出的文章（标题、摘要、推荐理由）写作，不要提及列表之外的文章或你记忆中的研究；用 [F编号] 指代文章。
输出 Markdown：先用 3-5 句话概括本周值得关注的内容与趋势（例如几篇文章不约而同地讨论了同一问题），
再按兴趣方向各写一句要点。总长度 300 字以内。
所有写给用户看的文字一律用〔回答语言〕书写（专业术语、文献标题、检索词可保留英文原文）。
""".strip().replace("〔研究领域〕", research_field()).replace("〔回答语言〕", answer_language_name())


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

_state: dict[str, Any] = {"running": False, "stage": None, "progress": None, "last_run_at": None, "last_result": None, "last_error": None}


def _stage(result: dict, name: str, fn, *args):
    """运行一个阶段；出错只记下来，不影响后面的阶段。遇到短暂的 database is locked 重试两次。"""
    import traceback

    _state.update(stage=name, progress=None)
    for attempt in range(3):
        try:
            return fn(*args)
        except peewee.OperationalError as exc:
            if "locked" in str(exc) and attempt < 2:
                time.sleep(2 + 3 * attempt)
                continue
            error = exc
        except Exception as exc:  # noqa: BLE001
            error = exc
        break
    where = traceback.extract_tb(error.__traceback__)[-1]
    message = t("feed:error.stage", stage=t(name), error=f"{type(error).__name__}: {error}",
                where=f"{Path(where.filename).name}:{where.lineno}")
    print(f"[paper_feed] {message}")
    traceback.print_exception(error)
    result["errors"].append(message)
    result.setdefault("stage_errors", []).append(message)
    return None


def _fetch_all(result: dict, library_dois: dict) -> int:
    fetched = 0
    for feed in list(Feed.select().where(Feed.enabled == True)):  # noqa: E712
        stop_point()
        try:
            new = fetch_feed(feed, library_dois)
            feed.last_new_count, feed.last_error = new, None
            fetched += new
        except Exception as exc:
            feed.last_error = f"{type(exc).__name__}: {exc}"
            result["errors"].append(f"{feed.name}: {exc}")
        feed.last_fetch_at = dt.datetime.now()
        feed.save()
    return fetched


def run_once(force: bool = False) -> dict:
    """抓取 → 补全 → 排序 → 自动入库 → 需要时生成上周周报。由 heartbeat 定时调用，也可在页面上手动触发。
    各阶段互相隔离：某一步出错只记下来，其余照常进行（例如补全失败不影响打分）。"""
    init_db()
    if not _run_lock.acquire(blocking=False):
        return {"skipped": t("已在运行")}
    try:
        _state["running"] = True
        result: dict[str, Any] = {"fetched": 0, "errors": []}
        library_dois = _stage(result, "读取文献库 DOI", backend.library_doi_map) or {}
        result["fetched"] = _stage(result, "抓取", _fetch_all, result, library_dois) or 0
        result["enriched"] = _stage(result, "补全摘要", enrich_items)
        calibration = _stage(result, "校准门槛", recalibrate)
        result["thresholds"] = calibration["auto"] if calibration else None
        result["scored"] = _stage(result, "打分", score_items)
        result["rescored"] = _stage(result, "重新评估待处理文章", rescore_pending)
        result["auto_ingested"] = _stage(result, "自动入库", auto_ingest)
        result["zotero_synced"] = _stage(result, "同步到 Zotero", sync_ingested_to_zotero)
        _stage(result, "整理已读", mark_settled_read)
        pruned = _stage(result, "清理旧条目", prune_old_items)
        result["pruned"] = pruned["deleted"] if pruned else None
        last_week = iso_week(dt.date.today() - dt.timedelta(days=7))
        if not Report.get_or_none(Report.week == last_week) and Item.select().where(
                (Item.created_at >= week_range(last_week)[0]) & (Item.created_at < week_range(last_week)[1])).exists():
            if _stage(result, "生成周报", generate_report, last_week) is not None:
                result["report"] = last_week
        _state.update(last_run_at=dt.datetime.now().isoformat(timespec="seconds"), last_result=result,
                      last_error=t("common:list_sep_strong").join(result.get("stage_errors", [])) or None)
        return result
    finally:
        _state.update(running=False, stage=None, progress=None)
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
        "stage": t(_state["stage"]) if _state.get("stage") else None,   # 阶段名在内部用中文，显示时翻译
        "last_fetch_at": last.strftime("%Y-%m-%d %H:%M") if last else None,
        "next_fetch_at": (last + dt.timedelta(hours=interval)).strftime("%Y-%m-%d %H:%M") if last else t("添加订阅源后开始"),
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
