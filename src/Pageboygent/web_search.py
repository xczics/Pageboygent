"""
深度搜索的“全网”范围：查外部文献数据库（只有用户明确选择“全网”时才会发请求）。

来源（并行请求，各自限时；某个来源出错不影响其他来源）：
- OpenAlex   覆盖最广、相关度排序、多数带摘要。2026 年起需要免费 API key（openalex_api_key），不带 key 额度很小。
- NASA ADS   天文、行星科学、地球物理收录最全，摘要齐全。需要免费 token（ads_api_token），没有就跳过。
- Crossref   出版商注册 DOI 时就收录，新文章最快；摘要较少。不需要 key。

合并：按 DOI（没有 DOI 时按规范化标题）合并同一篇文献，用倒数排名融合（各来源的名次），来源越多、越靠前越靠前。
已在你库里的文献标出“库中已有”（按 DOI 或完整标题匹配）。
"""
from __future__ import annotations

import difflib
import html
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import requests

from .config import get_config

TIMEOUT = 15
PER_SOURCE = 30
RRF_K = 60
CACHE_SECONDS = 600
SOURCES = {"openalex": "OpenAlex", "ads": "NASA ADS", "crossref": "Crossref"}
# Crossref 的相关度排序明显弱于 OpenAlex / ADS（它主要按书目字段匹配），名次的权重打折
SOURCE_WEIGHTS = {"openalex": 1.0, "ads": 1.0, "crossref": 0.6}
USER_AGENT = "Pageboygent (literature search; https://github.com/xczics/Pageboygent)"

_cache_lock = threading.Lock()
_cache: "OrderedDict[str, tuple[float, dict]]" = OrderedDict()


def _strip_tags(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text or ""))).strip()


def _norm_title(title: str) -> str:
    return re.sub(r"[\W_]+", "", (title or "").lower())[:200]


def _record(source: str, rank: int, *, title: str, authors: list[str], year: Any, journal: str | None,
            doi: str | None, abstract: str, url: str | None, cited_by: int | None = None) -> dict:
    from .storage import normalize_doi

    year_int = None
    try:
        year_int = int(str(year)[:4]) if year else None
    except ValueError:
        pass
    return {"source": source, "rank": rank, "title": _strip_tags(title), "authors": [a for a in authors if a][:60],
            "year": year_int, "journal": _strip_tags(journal or "") or None, "doi": normalize_doi(doi),
            "abstract": _strip_tags(abstract), "url": url, "cited_by": cited_by}


# ---------------------------------------------------------------------------
# 各来源（query：自然语言 + 短语；year_from / year_to / author 为可选过滤）
# ---------------------------------------------------------------------------

def search_openalex(query: str, *, year_from: int | None, year_to: int | None, author: str | None) -> list[dict]:
    params: dict[str, Any] = {"search": query, "per-page": PER_SOURCE,
                              "select": "doi,title,publication_year,authorships,primary_location,abstract_inverted_index,cited_by_count,id"}
    filters = []
    if year_from or year_to:
        filters.append(f"publication_year:{year_from or ''}-{year_to or ''}")
    if author:
        filters.append(f"raw_author_name.search:{author}")
    if filters:
        params["filter"] = ",".join(filters)
    key = get_config("openalex_api_key")
    if key:
        params["api_key"] = key
    r = requests.get("https://api.openalex.org/works", params=params, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    out = []
    for rank, w in enumerate(r.json().get("results", []), start=1):
        words: dict[int, str] = {}
        for word, positions in (w.get("abstract_inverted_index") or {}).items():
            for pos in positions:
                words[pos] = word
        source = ((w.get("primary_location") or {}).get("source") or {})
        doi = (w.get("doi") or "").replace("https://doi.org/", "") or None
        out.append(_record("openalex", rank, title=w.get("title") or "", year=w.get("publication_year"),
                           authors=[(a.get("author") or {}).get("display_name", "") for a in w.get("authorships", [])],
                           journal=source.get("display_name"), doi=doi, abstract=" ".join(words[i] for i in sorted(words)),
                           url=f"https://doi.org/{doi}" if doi else w.get("id"), cited_by=w.get("cited_by_count")))
    return out


def search_ads(query: str, *, year_from: int | None, year_to: int | None, author: str | None) -> list[dict]:
    token = get_config("ads_api_token")
    if not token:
        raise LookupError("not-configured")
    q = query
    if author:
        q += f' author:"{author}"'
    if year_from or year_to:
        q += f" year:{year_from or 1000}-{year_to or 3000}"
    r = requests.get("https://api.adsabs.harvard.edu/v1/search/query", timeout=TIMEOUT,
                     headers={"Authorization": f"Bearer {token}", "User-Agent": USER_AGENT},
                     params={"q": q, "rows": PER_SOURCE, "sort": "score desc",
                             "fl": "title,author,year,pub,doi,abstract,bibcode,citation_count"})
    r.raise_for_status()
    out = []
    for rank, d in enumerate(r.json().get("response", {}).get("docs", []), start=1):
        doi = (d.get("doi") or [None])[0]
        out.append(_record("ads", rank, title=(d.get("title") or [""])[0], authors=d.get("author") or [], year=d.get("year"),
                           journal=d.get("pub"), doi=doi, abstract=d.get("abstract") or "",
                           url=f"https://doi.org/{doi}" if doi else f"https://ui.adsabs.harvard.edu/abs/{d.get('bibcode')}",
                           cited_by=d.get("citation_count")))
    return out


def search_crossref(query: str, *, year_from: int | None, year_to: int | None, author: str | None) -> list[dict]:
    params: dict[str, Any] = {"query.bibliographic": query, "rows": PER_SOURCE,
                              "select": "DOI,title,author,issued,container-title,abstract,is-referenced-by-count,URL"}
    filters = []
    if year_from:
        filters.append(f"from-pub-date:{year_from}")
    if year_to:
        filters.append(f"until-pub-date:{year_to}")
    if filters:
        params["filter"] = ",".join(filters)
    if author:
        params["query.author"] = author
    r = requests.get("https://api.crossref.org/works", params=params, timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    out = []
    for rank, w in enumerate(r.json().get("message", {}).get("items", []), start=1):
        parts = ((w.get("issued") or {}).get("date-parts") or [[None]])[0]
        out.append(_record("crossref", rank, title=(w.get("title") or [""])[0],
                           authors=[" ".join(filter(None, [a.get("given"), a.get("family")])) for a in w.get("author", [])],
                           year=parts[0] if parts else None, journal=(w.get("container-title") or [None])[0],
                           doi=w.get("DOI"), abstract=w.get("abstract") or "", url=w.get("URL"),
                           cited_by=w.get("is-referenced-by-count")))
    return out


SEARCHERS: dict[str, Callable[..., list[dict]]] = {"openalex": search_openalex, "ads": search_ads, "crossref": search_crossref}


# ---------------------------------------------------------------------------
# 合并与排序
# ---------------------------------------------------------------------------

def merge(per_source: dict[str, list[dict]]) -> list[dict]:
    """同一篇文献（DOI 相同，或没有 DOI 时标题几乎一致）合并；按各来源名次做倒数排名融合。"""
    merged: list[dict] = []
    by_doi: dict[str, dict] = {}
    by_title: dict[str, dict] = {}
    for source, records in per_source.items():
        for rec in records:
            if not rec["title"]:
                continue
            entry = by_doi.get(rec["doi"]) if rec["doi"] else None
            title_key = _norm_title(rec["title"])
            if entry is None and title_key:
                entry = by_title.get(title_key)
                if entry is None:   # 标题几乎一致（标点、大小写不同）也算同一篇
                    for key, candidate in by_title.items():
                        if abs(len(key) - len(title_key)) < 8 and difflib.SequenceMatcher(None, key, title_key).ratio() > 0.95:
                            entry = candidate
                            break
            if entry is None:
                entry = {**rec, "ranks": {}, "score": 0.0}
                merged.append(entry)
            else:   # 补全缺的字段：摘要取最长的，DOI、期刊、年份取先到的
                for field in ("doi", "journal", "year", "url"):
                    entry[field] = entry.get(field) or rec[field]
                if len(rec["abstract"]) > len(entry.get("abstract") or ""):
                    entry["abstract"] = rec["abstract"]
                if len(rec["authors"]) > len(entry.get("authors") or []):
                    entry["authors"] = rec["authors"]
                entry["cited_by"] = max(filter(None, [entry.get("cited_by"), rec.get("cited_by")]), default=None)
            if source not in entry["ranks"]:
                entry["ranks"][source] = rec["rank"]
                entry["score"] += SOURCE_WEIGHTS.get(source, 1.0) / (RRF_K + rec["rank"])
            if entry.get("doi"):
                by_doi[entry["doi"]] = entry
            by_title[title_key] = entry
    return sorted(merged, key=lambda e: -e["score"])


def search(query: str, *, year_from: int | None = None, year_to: int | None = None,
           author: str | None = None) -> dict:
    """并行查询各来源并合并。结果缓存 10 分钟（翻页、切换排序不重复请求）。"""
    cache_key = f"{query}\x00{year_from}\x00{year_to}\x00{author}"
    with _cache_lock:
        hit = _cache.get(cache_key)
        if hit and hit[0] > time.time() - CACHE_SECONDS:
            return hit[1]
    per_source: dict[str, list[dict]] = {}
    status: dict[str, str] = {}

    def _run(name: str) -> None:
        try:
            per_source[name] = SEARCHERS[name](query, year_from=year_from, year_to=year_to, author=author)
            status[name] = "ok"
        except LookupError:
            status[name] = "not-configured"
        except Exception as exc:  # noqa: BLE001
            status[name] = f"error: {type(exc).__name__}: {str(exc)[:200]}"

    with ThreadPoolExecutor(max_workers=len(SEARCHERS)) as pool:
        list(pool.map(_run, SEARCHERS))
    result = {"results": merge(per_source), "status": status,
              "counts": {name: len(records) for name, records in per_source.items()}}
    with _cache_lock:
        _cache[cache_key] = (time.time(), result)
        while len(_cache) > 32:
            _cache.popitem(last=False)
    return result
