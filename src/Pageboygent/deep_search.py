"""
深度搜索：多渠道检索 + 联合排序（不调用大模型）。

渠道（各自独立召回，再在“文献”这一级合并）：
- meta              标题 / 作者 / DOI（按词干匹配，词越稀有权重越高）
- fulltext_bm25     原文切块 BM25（Whoosh，支持引号短语、NOT）
- fulltext_semantic 原文切块语义（Chroma 向量）
- notes             AI 快速总结、主题总结、笔记（BM25）
- paper_semantic    整篇文献的语义向量（文献追踪已为全库算好，标题 + 摘要/总结）

联合排序：加权倒数排名融合（RRF，k = 60）——每个渠道只看名次不看原始分数，不同量纲的分数无需校准；
同一渠道里多处命中略加分；查询词覆盖越全越靠前；整句短语出现再加分；DOI 精确命中直接置顶。
每条结果给出最能说明问题的几段原文（查询词高亮），以及它在各渠道里的名次（为什么排在这里）。

查询语法：普通词（词干匹配，evaporation ≈ evaporated）、"引号短语"、-排除词、author:姓名、
year:2015-2024 / year:>2018 / year:2020、doi:10.xxxx/…（直接粘贴 DOI 也行）。
"""
from __future__ import annotations

import html
import math
import re
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .i18n import t

RRF_K = 60
PAGE_SIZE = 10
MAX_RESULTS = 300
CHANNELS: dict[str, tuple[str, float]] = {
    "meta": ("标题与作者", 1.2),
    "fulltext_bm25": ("原文 · 关键词", 1.0),
    "fulltext_semantic": ("原文 · 语义", 1.0),
    "notes": ("AI 总结与笔记", 0.8),
    "paper_semantic": ("整篇 · 语义", 0.9),
}
SCOPES: dict[str, tuple[str, list[str]]] = {
    "all": ("全部", list(CHANNELS)),
    "fulltext": ("原文", ["fulltext_bm25", "fulltext_semantic"]),
    "notes": ("AI 总结与笔记", ["notes"]),
    "title": ("标题与作者", ["meta"]),
    "web": ("全网", []),      # 外部数据库（OpenAlex / NASA ADS / Crossref），只有选了才查
}
SEMANTIC_WINDOW = 0.12   # 语义渠道只保留与最佳结果相差不超过这么多的：不让“总能算出一个相似度”的向量把无关文献带进来
_STOPWORDS = set("""a an and are as at be by for from has have in into is it its of on or that the their this to was
were which with within without we our using use used via than then these those between during after before
can may also about over under new study""".split())
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)", re.I)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)*|\d+[A-Za-z]+[A-Za-z0-9]*|[一-鿿]+")
_KIND_LABELS = {"summary": "AI 快速总结", "theme": "主题总结", "note": "笔记"}


# ---------------------------------------------------------------------------
# 查询解析
# ---------------------------------------------------------------------------

@dataclass
class ParsedQuery:
    raw: str
    terms: list[str] = field(default_factory=list)      # 普通词（原样）
    phrases: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    author: str | None = None
    year_from: int | None = None
    year_to: int | None = None
    doi: str | None = None

    @property
    def free_text(self) -> str:
        """给语义渠道用的自然语言文本（不含语法符号和排除词）。"""
        return " ".join(self.phrases + self.terms).strip()

    @property
    def stems(self) -> list[str]:
        words = [w for t in self.terms + self.phrases for w in tokenize(t)]
        return list(dict.fromkeys(stem(w) for w in words if w not in _STOPWORDS))

    def whoosh(self) -> str:
        """给 Whoosh（BM25）用的查询串：词与短语之间 OR（命中越多越靠前）。"""
        clean = lambda s: re.sub(r"[()\[\]{}:^~*?\\/!]+", " ", s).strip()  # noqa: E731
        parts = [clean(t) for t in self.terms if clean(t) and clean(t).upper() not in {"AND", "OR", "NOT"}]
        parts += [f'"{clean(p)}"' for p in self.phrases if clean(p)]
        # 排除词不交给 Whoosh，在合并结果时按标题与命中片段过滤（更直观，也避开 Whoosh 的 NOT 评分问题）
        return " ".join(parts).strip()

    def describe(self) -> list[str]:
        chips = []
        if self.author:
            chips.append(t("search:filter.author", value=self.author))
        if self.year_from or self.year_to:
            chips.append(t("search:filter.year", value=(f"{self.year_from or '…'}–{self.year_to or '…'}"
                                                         if self.year_from != self.year_to else str(self.year_from))))
        if self.doi:
            chips.append(f"DOI：{self.doi}")
        chips += [t("search:filter.exclude", value=w) for w in self.excluded]
        return chips


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def stem(word: str) -> str:
    from whoosh.lang.porter import stem as porter

    return porter(word) if word.isascii() else word


def parse_query(raw: str) -> ParsedQuery:
    q = ParsedQuery(raw=raw or "")
    for token in re.findall(r'-?"[^"]*"|-?\S+:"[^"]*"|\S+', q.raw):
        negative = token.startswith("-") and len(token) > 1
        body = token[1:] if negative else token
        field_match = re.match(r'(?i)(author|作者|year|年份|doi):(.*)$', body)
        if field_match:
            name, value = field_match.group(1).lower(), field_match.group(2).strip('"').strip()
            if name in {"author", "作者"} and value:
                q.author = value
            elif name in {"year", "年份"}:
                _apply_year(q, value)
            elif name == "doi" and value:
                q.doi = value.lower()
            continue
        doi = _DOI_RE.search(body)
        if doi and not negative:
            q.doi = doi.group(1).rstrip(".,;)").lower()
            continue
        if body.startswith('"') and body.endswith('"') and len(body) > 2:
            (q.excluded if negative else q.phrases).append(body.strip('"').strip())
        elif negative:
            q.excluded.append(body)
        elif body:
            q.terms.append(body)
    return q


def _apply_year(q: ParsedQuery, value: str) -> None:
    value = value.replace("—", "-").replace("–", "-")
    if m := re.fullmatch(r"(\d{4})?-(\d{4})?", value):
        q.year_from = int(m.group(1)) if m.group(1) else None
        q.year_to = int(m.group(2)) if m.group(2) else None
    elif m := re.fullmatch(r"(>=?|<=?)(\d{4})", value):
        year = int(m.group(2))
        if m.group(1).startswith(">"):
            q.year_from = year + (0 if "=" in m.group(1) else 1)
        else:
            q.year_to = year - (0 if "=" in m.group(1) else 1)
    elif re.fullmatch(r"\d{4}", value):
        q.year_from = q.year_to = int(value)


# ---------------------------------------------------------------------------
# 外部依赖（测试时整体替换）
# ---------------------------------------------------------------------------

class Backend:
    def papers(self) -> list[dict]:
        """全库文献：pid, key, title, year, first_author, doi, abstract, summary, authors（完整作者名单，取不到为空）。"""
        from .storage import Paper

        authors = {}
        try:
            from . import paper_feed

            paper_feed.init_db()
            import json

            authors = {row.pid: json.loads(row.authors or "[]") for row in paper_feed.LibraryAuthor.select()}
        except Exception:
            pass
        return [{"pid": int(p.id), "key": str(p.key_in_zotero), "title": str(p.title or ""), "year": p.year,
                 "first_author": p.first_author or "", "doi": (p.doi or "").lower(), "abstract": p.abstract or "",
                 "summary": p.quick_summary or "", "authors": authors.get(int(p.id), [])}
                for p in Paper.select(Paper.id, Paper.key_in_zotero, Paper.title, Paper.year, Paper.first_author,
                                      Paper.doi, Paper.abstract, Paper.quick_summary)]

    def embed(self, text: str) -> np.ndarray:
        from . import rag

        return np.asarray(rag.embed_query(text), dtype=np.float32)

    def fulltext_bm25(self, query: str, limit: int) -> list[dict]:
        from . import rag

        return rag.bm25_retrieve(query, top_k=limit)

    def fulltext_semantic(self, vec: np.ndarray, limit: int) -> list[dict]:
        from . import rag

        res = rag.DBcollection.query(query_embeddings=[vec.tolist()], n_results=limit,
                                     include=["documents", "metadatas", "distances"])
        return [{"paper_id": int(m["paper_id"]), "line_number": m.get("line_number"), "text": d, "score": 1.0 - dist}
                for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]

    def notes(self, query: str, limit: int) -> list[dict]:
        from . import ai_notes_index

        return ai_notes_index.search(query, mode="bm25", limit=limit)

    def paper_vectors(self) -> tuple[list[int], np.ndarray]:
        from . import paper_feed
        from .feed_ranker import blob_to_vector

        paper_feed.init_db()
        rows = list(paper_feed.LibraryVector.select().where(paper_feed.LibraryVector.kind == "full"))
        if not rows:
            return [], np.zeros((0, 1), dtype=np.float32)
        matrix = np.vstack([blob_to_vector(r.vec) for r in rows]).astype(np.float32)
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)
        return [int(r.pid) for r in rows], matrix

    def groups(self) -> list[dict]:
        """兴趣分组（文献追踪自动分组的结果），用来给出“相关主题”。"""
        try:
            from . import paper_feed

            return paper_feed.facet_store_ready().load().get("groups", [])
        except Exception:
            return []

    def stats(self) -> dict:
        out = {}
        try:
            from . import rag

            out["chunks"] = rag.DBcollection.count()
        except Exception:
            pass
        return out


backend = Backend()
_cache_lock = threading.Lock()
_result_cache: "OrderedDict[str, tuple[float, dict]]" = OrderedDict()
_papers_cache: dict[str, Any] = {}
CACHE_SECONDS = 600


def _library(force: bool = False) -> dict:
    """文献元数据与标题词干的 IDF（标题渠道用），缓存 10 分钟。"""
    now = time.time()
    if not force and _papers_cache.get("at", 0) > now - CACHE_SECONDS:
        return _papers_cache
    papers = backend.papers()
    title_stems = {p["pid"]: {stem(w) for w in tokenize(p["title"]) if w not in _STOPWORDS} for p in papers}
    df = Counter(s for stems in title_stems.values() for s in stems)
    n = max(1, len(papers))
    _papers_cache.update(at=now, by_pid={p["pid"]: p for p in papers}, title_stems=title_stems,
                         idf={s: math.log(1 + n / c) for s, c in df.items()}, n=n)
    _papers_cache.pop("vectors", None)
    return _papers_cache


def _paper_vectors() -> tuple[list[int], np.ndarray]:
    lib = _library()
    if "vectors" not in lib:
        lib["vectors"] = backend.paper_vectors()
    return lib["vectors"]


# ---------------------------------------------------------------------------
# 各渠道：返回按相关度排好序的命中 [{pid, text?, where?, kind}]，同一文献可以出现多次
# ---------------------------------------------------------------------------

def _channel_meta(q: ParsedQuery, lib: dict) -> list[dict]:
    stems = q.stems
    idf = lib["idf"]
    phrase_norm = [re.sub(r"\s+", " ", p.lower()) for p in q.phrases]
    total = sum(idf.get(s, math.log(1 + lib["n"])) for s in stems) or 1.0
    query_words = {w for t in q.terms for w in tokenize(t)}
    scored = []
    for pid, paper in lib["by_pid"].items():
        score = 0.0
        if q.doi and paper["doi"] and paper["doi"] == q.doi:
            score += 100.0
        title_stems = lib["title_stems"][pid]
        if stems:
            score += sum(idf.get(s, 0.0) for s in stems if s in title_stems) / total
        title_lower = paper["title"].lower()
        score += 0.5 * sum(1 for p in phrase_norm if p and p in title_lower)
        surnames = {_surname(a).lower() for a in (paper["authors"] or [paper["first_author"]]) if a}
        if query_words & surnames:
            score += 0.6
        if score >= 0.34:
            scored.append((score, pid))
    scored.sort(reverse=True)
    return [{"pid": pid, "kind": "title", "score": s} for s, pid in scored[:200]]


def _channel_fulltext_bm25(q: ParsedQuery, lib: dict) -> list[dict]:
    query = q.whoosh()
    if not query:
        return []
    return [{"pid": int(h["paper_id"]), "kind": "fulltext", "text": h["text"], "where": h.get("line_number"),
             "score": h["score"]} for h in backend.fulltext_bm25(query, 300)]


def _channel_fulltext_semantic(q: ParsedQuery, lib: dict, vec: np.ndarray | None) -> list[dict]:
    if vec is None:
        return []
    hits = backend.fulltext_semantic(vec, 150)
    if not hits:
        return []
    best = max(h["score"] for h in hits)
    return [{"pid": int(h["paper_id"]), "kind": "fulltext", "text": h["text"], "where": h.get("line_number"),
             "score": h["score"]} for h in hits if h["score"] >= best - SEMANTIC_WINDOW]


def _channel_notes(q: ParsedQuery, lib: dict) -> list[dict]:
    query = q.whoosh()
    if not query:
        return []
    return [{"pid": int(h["paper_id"]), "kind": h["kind"], "label": h.get("label"), "text": h["content"],
             "score": h["score"]} for h in backend.notes(query, 200) if h.get("paper_id") is not None]


def _channel_paper_semantic(q: ParsedQuery, lib: dict, vec: np.ndarray | None) -> list[dict]:
    if vec is None:
        return []
    pids, matrix = _paper_vectors()
    if not pids:
        return []
    unit = vec / max(float(np.linalg.norm(vec)), 1e-9)
    sims = matrix @ unit
    order = np.argsort(-sims)[:80]
    best = float(sims[order[0]])
    return [{"pid": pids[i], "kind": "abstract", "score": float(sims[i])} for i in order
            if float(sims[i]) >= best - SEMANTIC_WINDOW]


# ---------------------------------------------------------------------------
# 联合排序
# ---------------------------------------------------------------------------

def fuse(channel_hits: dict[str, list[dict]]) -> dict[int, dict]:
    """加权 RRF：每个渠道内按文献的最佳名次计分，同一渠道多处命中略加分。返回 {pid: {score, ranks, hits}}。"""
    fused: dict[int, dict] = {}
    for channel, hits in channel_hits.items():
        weight = CHANNELS[channel][1]
        order: list[int] = []
        counts: Counter = Counter()
        for hit in hits:
            if hit["pid"] not in counts:
                order.append(hit["pid"])
            counts[hit["pid"]] += 1
        for rank, pid in enumerate(order, start=1):
            entry = fused.setdefault(pid, {"score": 0.0, "ranks": {}, "hits": []})
            entry["score"] += weight / (RRF_K + rank) * (1 + 0.15 * math.log(counts[pid]))
            entry["ranks"][channel] = rank
        for hit in hits:
            fused[hit["pid"]]["hits"].append({**hit, "channel": channel})
    return fused


def _coverage(q: ParsedQuery, texts: list[str]) -> float:
    stems = q.stems
    if len(stems) < 2:
        return 1.0
    seen = {stem(w) for t in texts for w in tokenize(t)}
    return sum(1 for s in stems if s in seen) / len(stems)


def _passes_filters(q: ParsedQuery, paper: dict, texts: list[str]) -> bool:
    year = paper.get("year")
    if q.year_from and (not year or year < q.year_from):
        return False
    if q.year_to and (not year or year > q.year_to):
        return False
    if q.author and not author_matches(q.author, (paper["authors"] or []) + [paper["first_author"]]):
        return False
    if q.doi and paper["doi"] != q.doi and not (q.terms or q.phrases):
        return False
    if q.excluded:
        blob = " ".join([paper["title"]] + texts).lower()
        if any(word.lower() in blob for word in q.excluded):
            return False
    return True


def search(raw: str, *, scope: str = "all", sort: str = "relevance", page: int = 1) -> dict:
    started = time.perf_counter()
    q = parse_query(raw)
    scope = scope if scope in SCOPES else "all"
    if scope == "web":
        return _search_web(raw, q, sort=sort, page=page, started=started)
    cache_key = f"{scope}\x00{raw.strip()}"
    with _cache_lock:
        cached = _result_cache.get(cache_key)
        if cached and cached[0] > time.time() - CACHE_SECONDS:
            _result_cache.move_to_end(cache_key)
            ranked = cached[1]
        else:
            ranked = None
    if ranked is None:
        ranked = _rank(q, scope)
        with _cache_lock:
            _result_cache[cache_key] = (time.time(), ranked)
            while len(_result_cache) > 32:
                _result_cache.popitem(last=False)
    results = ranked["results"]
    if sort == "newest":
        results = sorted(results, key=lambda r: (-(r["year"] or 0), -r["score"]))
    pages = max(1, math.ceil(len(results) / PAGE_SIZE))
    page = min(max(1, page), pages)
    shown = [_present(r, q) for r in results[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]]
    return {"query": raw, "scope": scope, "sort": sort, "page": page, "pages": pages, "total": len(results),
            "took_ms": round((time.perf_counter() - started) * 1000), "errors": ranked["errors"],
            "filters": q.describe(), "results": shown, "facets": ranked["facets"], "related": ranked["related"],
            "channels": [{"key": c, "label": t(CHANNELS[c][0])} for c in SCOPES[scope][1]]}


def _rank(q: ParsedQuery, scope: str) -> dict:
    started = time.perf_counter()
    lib = _library()
    channels = SCOPES[scope][1]
    vec = None
    if q.free_text and ({"fulltext_semantic", "paper_semantic"} & set(channels)):
        try:
            vec = backend.embed(q.free_text)
        except Exception as exc:
            print(f"[deep_search] embedding failed, semantic channels skipped: {exc}")
    runners = {
        "meta": lambda: _channel_meta(q, lib),
        "fulltext_bm25": lambda: _channel_fulltext_bm25(q, lib),
        "fulltext_semantic": lambda: _channel_fulltext_semantic(q, lib, vec),
        "notes": lambda: _channel_notes(q, lib),
        "paper_semantic": lambda: _channel_paper_semantic(q, lib, vec),
    }
    channel_hits: dict[str, list[dict]] = {}
    errors = {}
    for name in channels:
        try:
            channel_hits[name] = [h for h in runners[name]() if h["pid"] in lib["by_pid"]]
        except Exception as exc:
            errors[name] = str(exc)
            print(f"[deep_search] channel {name} failed: {exc}")
    fused = fuse(channel_hits)
    phrase_norm = [p.lower() for p in q.phrases if p]
    results = []
    for pid, entry in fused.items():
        paper = lib["by_pid"][pid]
        texts = [h.get("text") or "" for h in entry["hits"]]
        if not _passes_filters(q, paper, texts):
            continue
        cover = _coverage(q, [paper["title"]] + texts)
        score = entry["score"] * (0.7 + 0.3 * cover)
        if phrase_norm and any(p in t.lower() for p in phrase_norm for t in [paper["title"]] + texts):
            score *= 1.15
        if q.doi and paper["doi"] == q.doi:
            score += 10.0
        results.append({"pid": pid, "score": score, "ranks": entry["ranks"], "hits": entry["hits"],
                        "year": paper.get("year"), "coverage": round(cover, 2)})
    # 只有过滤条件、没有检索词：按年份列出符合条件的全部文献
    if not (q.terms or q.phrases or q.doi) and (q.author or q.year_from or q.year_to):
        results = [{"pid": p["pid"], "score": 0.0, "ranks": {}, "hits": [], "year": p.get("year"), "coverage": 1.0}
                   for p in lib["by_pid"].values() if _passes_filters(q, p, [])]
        results.sort(key=lambda r: -(r["year"] or 0))
    else:
        results.sort(key=lambda r: -r["score"])
    results = results[:MAX_RESULTS]
    return {"results": results, "facets": _facets(results, lib, channel_hits), "related": _related(results, lib),
            "errors": errors, "took_ms": (time.perf_counter() - started) * 1000}


# ---------------------------------------------------------------------------
# 结果展示：片段挑选、高亮、侧栏统计、相关主题
# ---------------------------------------------------------------------------

_MD_NOISE = re.compile(r"!\[[^\]]*\]\([^)]*\)|<[^>]+>|^#+\s*|\*\*|__|`", re.M)


def _plain(text: str) -> str:
    text = _MD_NOISE.sub(" ", text or "")
    text = text.split("## Labels")[0]
    return re.sub(r"\s+", " ", text).strip()


def highlight(text: str, q: ParsedQuery, width: int = 300) -> str:
    """截取查询词最密集的一段，查询词加粗。公式（$…$）里不加标记，也不从中间截断。"""
    plain = _plain(text)
    stems = set(q.stems)
    phrases = [p.lower() for p in q.phrases if p]
    positions = [m.start() for m in _WORD_RE.finditer(plain) if stem(m.group(0).lower()) in stems]
    for phrase in phrases:
        idx = plain.lower().find(phrase)
        if idx >= 0:
            positions.append(idx)
    start = 0
    if len(plain) > width and positions:
        best, best_count = positions[0], 0
        for p in positions:
            count = sum(1 for x in positions if p <= x < p + width)
            if count > best_count:
                best, best_count = p, count
        start = max(0, best - width // 4)
        space = plain.rfind(" ", 0, start)
        start = space + 1 if space > start - 30 else start
    end = min(len(plain), start + width)
    if plain[start:end].count("$") % 2:
        nxt = plain.find("$", end)
        end = nxt + 1 if 0 <= nxt < end + 120 else plain.rfind("$", start, end)
    window = plain[start:end]
    pieces = re.split(r"(\$[^$]+\$)", window)
    out = []
    for piece in pieces:
        if piece.startswith("$") and piece.endswith("$") and len(piece) > 1:
            out.append(html.escape(piece))
            continue
        last = 0
        chunk = []
        for m in _WORD_RE.finditer(piece):
            word = m.group(0)
            if stem(word.lower()) in stems:
                chunk.append(html.escape(piece[last:m.start()]))
                chunk.append(f"<b>{html.escape(word)}</b>")
                last = m.end()
        chunk.append(html.escape(piece[last:]))
        out.append("".join(chunk))
    prefix = "… " if start > 0 else ""
    suffix = " …" if end < len(plain) else ""
    return prefix + "".join(out) + suffix


def _where_label(hit: dict) -> str:
    if hit["kind"] == "fulltext":
        return t("search:where.fulltext_line", line=hit["where"]) if hit.get("where") is not None else t("search:where.fulltext")
    if hit["kind"] == "abstract":
        return t("search:where.abstract")
    label = t(_KIND_LABELS.get(hit["kind"], hit["kind"]))
    return f"{label} · {hit['label']}" if hit.get("label") and hit["kind"] != "summary" else label


def _present(result: dict, q: ParsedQuery) -> dict:
    lib = _library()
    paper = lib["by_pid"][result["pid"]]
    # 片段：优先原文关键词命中（含查询词），其次 AI 总结与笔记、原文语义，最后摘要
    priority = {"fulltext_bm25": 0, "notes": 1, "fulltext_semantic": 2, "meta": 3, "paper_semantic": 4}
    hits = sorted((h for h in result["hits"] if h.get("text")), key=lambda h: (priority[h["channel"]], -h["score"]))
    passages, seen = [], set()
    for hit in hits:
        key = _plain(hit["text"])[:80]
        if key in seen:
            continue
        seen.add(key)
        passages.append({"where": _where_label(hit), "html": highlight(hit["text"], q),
                         "channel": t(CHANNELS[hit["channel"]][0])})
        if len(passages) >= 4:
            break
    if not passages:
        fallback = paper["abstract"] or paper["summary"]
        if fallback:
            passages.append({"where": t("search:where.abstract") if paper["abstract"] else t("AI 快速总结"), "html": highlight(fallback, q),
                             "channel": ""})
    authors = paper["authors"] or ([paper["first_author"]] if paper["first_author"] else [])
    shown_authors = t("search:authors.sep").join(authors[:3]) + (t("search:authors.more") if len(authors) > 3 else "")
    return {
        "pid": paper["pid"], "key": paper["key"], "title": paper["title"], "title_html": highlight(paper["title"], q, 400),
        "authors": shown_authors, "first_author": paper["first_author"], "year": paper.get("year"), "doi": paper["doi"],
        "zotero_url": f"zotero://select/library/items/{paper['key']}",
        "doi_url": f"https://doi.org/{paper['doi']}" if paper["doi"] else None,
        "snippet": passages[0] if passages else None, "more": passages[1:],
        "channels": [{"key": c, "label": t(CHANNELS[c][0]), "rank": r} for c, r in sorted(result["ranks"].items(), key=lambda x: x[1])],
        "coverage": result["coverage"],
    }


def _facets(results: list[dict], lib: dict, channel_hits: dict[str, list[dict]]) -> dict:
    from .feed_ranker import author_key

    years = Counter(r["year"] for r in results if r["year"])
    authors: Counter = Counter()
    names_seen: list[str] = []
    for r in results[:100]:
        paper = lib["by_pid"][r["pid"]]
        names = [n for n in (paper["authors"] or [paper["first_author"]])[:6] if n]
        names_seen += names
        for key in {author_key(n) for n in names}:
            authors[key] += 1
    canonical = _canonical_authors(names_seen)
    in_results = {r["pid"] for r in results}
    return {
        "years": sorted(({"year": y, "count": c} for y, c in years.items()), key=lambda x: x["year"]),
        "authors": [{"name": canonical[k], "count": c} for k, c in authors.most_common(8) if c > 1 and k in canonical],
        "channels": [{"key": c, "label": t(CHANNELS[c][0]),
                      "count": len({h["pid"] for h in hits if h["pid"] in in_results})}
                     for c, hits in channel_hits.items()],
    }


def _related(results: list[dict], lib: dict) -> list[dict]:
    """相关主题：结果前 20 篇最集中的兴趣分组，给出它的名字和关键词（点一下就以它为新查询）。"""
    top = {r["pid"] for r in results[:20]}
    if not top:
        return []
    out = []
    for group in backend.groups():
        overlap = len(top & set(group.get("pids", [])))
        if overlap >= 3 and group.get("name"):
            keywords = [c["text"] for c in group.get("claims", []) if c.get("kind") == "keyword" and c.get("status") == "active"]
            out.append({"name": group["name"], "overlap": overlap, "keywords": keywords[:6]})
    return sorted(out, key=lambda g: -g["overlap"])[:3]


def author_matches(query: str, names: list[str]) -> bool:
    """author: 过滤。只写了姓（如 Day）：要求姓完全一致（不会匹配到 Holliday）；
    写了全名：按“姓 + 名首字母”比较，同一个人的不同写法（James M.D. Day / James M. D. Day）都算。"""
    from .feed_ranker import author_key

    query = query.strip()
    if not query:
        return True
    if " " not in query and "," not in query and "." not in query:
        return any(_surname(n).lower() == query.lower() for n in names if n)
    target = author_key(query)
    return any(author_key(n) == target for n in names if n)


def _canonical_authors(names: list[str]) -> dict[str, str]:
    """把同一个人的不同写法归到一起：{author_key: 最常见的写法}。"""
    from .feed_ranker import author_key

    variants: dict[str, Counter] = {}
    for name in names:
        if name:
            variants.setdefault(author_key(name), Counter())[name] += 1
    return {key: counts.most_common(1)[0][0] for key, counts in variants.items()}


def _surname(name: str) -> str:
    name = (name or "").strip()
    if "," in name:
        return name.split(",")[0].strip()
    return name.split()[-1] if name.split() else ""


def suggest(prefix: str, limit: int = 8) -> list[dict]:
    """输入联想：标题里包含这段文字的文献、名字以它开头的作者。"""
    text = (prefix or "").strip().lower()
    if len(text) < 2:
        return []
    lib = _library()
    from .feed_ranker import author_key

    titles, authors, names_seen = [], Counter(), []
    for paper in lib["by_pid"].values():
        title = paper["title"]
        idx = title.lower().find(text)
        if idx >= 0:
            titles.append((idx != 0 and not title.lower()[idx - 1].isspace(), idx, title, paper["pid"]))
        for name in {n for n in (paper["authors"] or [paper["first_author"]]) if n}:
            if _surname(name).lower().startswith(text) or name.lower().startswith(text):
                authors[author_key(name)] += 1
                names_seen.append(name)
    titles.sort()
    canonical = _canonical_authors(names_seen)
    out = [{"type": "author", "text": canonical[k], "query": f'author:"{canonical[k]}"', "count": c}
           for k, c in authors.most_common(3) if k in canonical]
    out += [{"type": "title", "text": t, "query": f'"{t}"', "pid": pid} for _, _, t, pid in titles[: limit - len(out)]]
    return out[:limit]


def stats() -> dict:
    lib = _library()
    return {"papers": lib["n"], **backend.stats()}


def warmup() -> None:
    """服务启动时预热：文献元数据与标题索引、整篇向量矩阵、向量模型（第一次查询不必等这些）。"""
    _library(force=True)
    _paper_vectors()
    try:
        backend.embed("warm up")
    except Exception:
        pass



# ---------------------------------------------------------------------------
# 全网（外部数据库）
# ---------------------------------------------------------------------------

def _search_web(raw: str, q: ParsedQuery, *, sort: str, page: int, started: float) -> dict:
    from . import paper_feed, web_search

    text = " ".join(q.terms + [f'"{p}"' for p in q.phrases]).strip() or (q.doi or "")
    if not text:
        return {"query": raw, "scope": "web", "sort": sort, "page": 1, "pages": 1, "total": 0, "took_ms": 0, "errors": {},
                "filters": q.describe(), "results": [], "facets": {"years": [], "authors": [], "channels": []}, "related": [],
                "channels": [], "web": {"status": []}}
    found = web_search.search(text, year_from=q.year_from, year_to=q.year_to, author=q.author)
    # 各来源只看名次；再按查询词在标题与摘要里的覆盖程度调整（与库内搜索一致），压下只沾边的结果
    results = sorted(({**r, "score": r["score"] * (0.4 + 0.6 * _coverage(q, [r["title"], r["abstract"]]))}
                      for r in found["results"]), key=lambda r: -r["score"])
    if q.excluded:
        results = [r for r in results if not any(w.lower() in (r["title"] + " " + r["abstract"]).lower() for w in q.excluded)]
    try:
        library = paper_feed.backend.library_doi_map()
    except Exception:  # noqa: BLE001
        library = {}
    lib = _library()
    for r in results:
        pid = paper_feed.library_match(library, r.get("doi"), r["title"])
        r["in_library_pid"] = pid if pid in lib["by_pid"] else None
    if sort == "newest":
        results = sorted(results, key=lambda r: (-(r["year"] or 0), -r["score"]))
    pages = max(1, math.ceil(len(results) / PAGE_SIZE))
    page = min(max(1, page), pages)
    shown = [_present_web(r, q, lib) for r in results[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]]
    years = Counter(r["year"] for r in results if r["year"])
    authors: Counter = Counter(a for r in results[:100] for a in r["authors"][:6])
    status = [{"key": k, "label": web_search.SOURCES[k], "state": v, "count": found["counts"].get(k, 0)}
              for k, v in found["status"].items()]
    return {"query": raw, "scope": "web", "sort": sort, "page": page, "pages": pages, "total": len(results),
            "took_ms": round((time.perf_counter() - started) * 1000), "errors": {}, "filters": q.describe(),
            "results": shown, "related": [],
            "facets": {"years": sorted(({"year": y, "count": c} for y, c in years.items()), key=lambda x: x["year"]),
                       "authors": [{"name": n, "count": c} for n, c in authors.most_common(8) if c > 1],
                       "channels": [{"key": s["key"], "label": s["label"], "count": s["count"]} for s in status if s["state"] == "ok"]},
            "channels": [{"key": s["key"], "label": s["label"]} for s in status],
            "web": {"status": status}}


def _present_web(r: dict, q: ParsedQuery, lib: dict) -> dict:
    from . import web_search

    authors = r["authors"]
    shown_authors = t("search:authors.sep").join(authors[:3]) + (t("search:authors.more") if len(authors) > 3 else "")
    paper = lib["by_pid"].get(r.get("in_library_pid")) if r.get("in_library_pid") else None
    snippet = {"where": r.get("journal") or web_search.SOURCES[r["source"]], "html": highlight(r["abstract"], q),
               "channel": ""} if r["abstract"] else ({"where": r.get("journal"), "html": "", "channel": ""} if r.get("journal") else None)
    return {
        "pid": paper["pid"] if paper else None, "key": paper["key"] if paper else None,
        "title": r["title"], "title_html": highlight(r["title"], q, 400), "authors": shown_authors,
        "first_author": authors[0] if authors else "", "year": r["year"], "doi": r.get("doi") or "",
        "zotero_url": f"zotero://select/library/items/{paper['key']}" if paper else None,
        "doi_url": f"https://doi.org/{r['doi']}" if r.get("doi") else None, "url": r.get("url"),
        "snippet": snippet, "more": [],
        "channels": [{"key": s, "label": web_search.SOURCES[s], "rank": rank} for s, rank in sorted(r["ranks"].items(), key=lambda x: x[1])],
        "coverage": 1.0, "web": True, "in_library": bool(paper), "cited_by": r.get("cited_by"),
        # 加入 Zotero 时用的著录信息
        "record": {"title": r["title"], "authors": authors, "year": r["year"], "journal": r.get("journal"),
                   "doi": r.get("doi"), "abstract": r["abstract"], "url": r.get("url")},
    }
