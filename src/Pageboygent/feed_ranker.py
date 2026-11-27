"""
文献追踪的排序：判断一篇新文章你"大概率会感兴趣"的程度。

按推荐系统常用的"召回 → 排序 → 重排"组织，只依赖 RSS 能提供的信息（标题、摘要、作者）：

    多兴趣匹配   每个主题、每个综述子课题各是一个"兴趣方向"（facet），新文章与最接近的方向比，
                 取最大值而不是把全库平均成一个口味——小众但重要的方向不会被冲淡。
    稠密 + 关键词 嵌入向量管语义，BM25 关键词管专有名词（δ66Zn、Grand Tack 这类嵌入模型抓不准的词）。
    时间衰减     近期加入的库内文献、近期的反馈权重更高。
    作者亲和度   作者在你库里出现的次数（平滑），以及画像里"必看作者"。
    负反馈       和你点过 👎 的文章越像，扣分越多。
    LLM 判断     只对短名单调用，作为一项特征，并给出推荐理由。

刻意不用的信号（与用户讨论后放弃）：引用耦合（新文章的参考文献通常还没被收录）、
来源/期刊先验（会把综合期刊里的相关文章压进低分池）。

冷启动用人工权重；反馈积累后可替换为学习到的权重（第二期）。所有训练信号只来自用户的真实反馈和库内文献，
AI 自动入库但用户未表态的文章不算正例，避免模型自我强化。
"""
from __future__ import annotations

from .i18n import answer_language_name
from .config import research_field

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# 两类文章分开处理：有摘要（abstract）/ 只有标题（title）。
# 只有标题是常态而不是例外：Elsevier 系期刊（如 GCA）的摘要在开放数据库里普遍拿不到。
# 只有标题时向量信息少、作者信号相对更可靠，所以权重不同；门槛也分开，并按各自的反馈分别校准。
COLD_START_WEIGHTS = {
    "abstract": {"dense": 0.55, "lexical": 0.20, "author": 0.15, "feedback": 0.10},
    "title": {"dense": 0.45, "lexical": 0.15, "author": 0.30, "feedback": 0.10},
}
LLM_BLEND = 0.5                                     # 最终分 = (1-LLM_BLEND)*基础分 + LLM_BLEND*LLM 分
SHORTLIST_MIN_BASE = {"abstract": 0.35, "title": 0.35}  # 两类都按同一把锚点标尺换算后，门槛一致
AUTO_THRESHOLD = {"abstract": 0.75, "title": 0.78}      # 初始值；运行中按各自的自动入库准确率校准
CONFIRM_THRESHOLD = {"abstract": 0.50, "title": 0.45}
# 只有标题的文章自动入库时，还要求至少一项“强证据”：标题与库很近，或作者在库中常见
TITLE_STRONG_DENSE = 0.60
TITLE_STRONG_AUTHOR = 0.50

_STOP_AUTHOR = re.compile(r"[^a-z]")


def author_key(name: str) -> str:
    """作者归一化为 "姓 + 名首字母"：'Raymond, Sean N.' / 'Sean N. Raymond' → 'raymond s'。"""
    from .lit_search import _fold

    name = _fold(name or "").strip()
    if not name:
        return ""
    if "," in name:
        last, first = [p.strip() for p in name.split(",", 1)]
    else:
        parts = name.split()
        last, first = (parts[-1], " ".join(parts[:-1])) if len(parts) > 1 else (parts[0], "")
    last = _STOP_AUTHOR.sub("", last)
    initial = _STOP_AUTHOR.sub("", first)[:1]
    return f"{last} {initial}".strip()


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec


@dataclass
class Facet:
    """一个兴趣方向：主题、综述子课题，或兜底的"全库"。"""

    id: str
    name: str
    source: str            # theme / review / library
    description: str = ""  # AI 维护的说明（从库内文献提炼；用户可改，文献变化较大时 AI 会重新提炼）
    user_note: str = ""    # 用户的补充（只有用户能改，AI 永远不动）
    pids: list[int] = field(default_factory=list)
    enabled: bool = True
    weight: float = 1.0


@dataclass
class LibraryModel:
    """排序所需的库内数据：文献向量、近期权重、兴趣方向、作者计数。"""

    pids: list[int]
    vectors: np.ndarray                 # (n, d)，已归一化
    recency: np.ndarray                 # (n,)，0..1，越新越大
    meta: dict[int, dict]               # pid -> {cite, title, key}
    facets: list[Facet]
    author_counts: dict[str, int]
    in_domain_scores: np.ndarray        # 库内文献与最近 3 篇邻居的平均相似度（已按时间衰减），供页面解释用
    must_follow: set[str] = field(default_factory=set)
    title_vectors: np.ndarray | None = None       # 只用标题算的向量：与"只有标题"的新文章对称比较
    in_domain_title_scores: np.ndarray | None = None
    # 相似度换算的两个锚点 {kind: (lo, hi)}：lo = 库内随便两篇有多像，hi = 一篇典型库内文献与它最近邻居有多像。
    # 新文章的原始相似度按 (raw - lo) / (hi - lo) 换算到 0..1：像“随便一篇”就是 0，像“典型库内文献”就是 1。
    anchors: dict = field(default_factory=dict)

    def index_of(self) -> dict[int, int]:
        return {pid: i for i, pid in enumerate(self.pids)}


class RankerBackend:
    """外部依赖集中在这里（嵌入模型、文献库、关键词索引），测试时整体替换。"""

    def embed(self, text: str) -> np.ndarray:
        from . import rag

        return np.asarray(rag.embed_query(text), dtype=np.float32)

    def library_papers(self) -> list[dict]:
        from .storage import Paper

        return [{"pid": int(p.id), "key": str(p.key_in_zotero), "title": str(p.title or ""),
                 "first_author": p.first_author, "year": p.year, "abstract": p.abstract or "",
                 "summary": p.quick_summary or ""}
                for p in Paper.select(Paper.id, Paper.key_in_zotero, Paper.title, Paper.first_author, Paper.year,
                                      Paper.abstract, Paper.quick_summary)]

    def library_facets(self, pids: list[int], vectors: np.ndarray) -> list[Facet]:
        """兴趣方向：对全库文献的向量自动分组（不依赖主题或 Collection），分组编号跨重建保持稳定。"""
        from . import paper_feed

        return paper_feed.ensure_clusters(pids, vectors)

    def library_authors(self, papers: list[dict]) -> dict[int, list[str]]:
        """库内文献的作者（Zotero creators，按条目缓存在 paper_feed 数据库里；取不到时退回第一作者）。"""
        from . import paper_feed

        return paper_feed.library_authors(papers)

    def lexical_score(self, text: str) -> float:
        """新文章与库内 AI 总结的 BM25 最高分。"""
        from . import ai_notes_index

        words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}|\d+[A-Za-z]+[A-Za-z0-9]*|δ\S+", text or "")[:60]
        if not words:
            return 0.0
        query = " OR ".join(dict.fromkeys(words))
        try:
            hits = ai_notes_index.search(query, kinds=["summary"], limit=1)
        except Exception:
            return 0.0
        return float(hits[0]["score"]) if hits else 0.0

    def vector_cache_get(self, pid: int, digest: str, kind: str = "full") -> np.ndarray | None:
        from . import paper_feed

        return paper_feed.vector_cache_get(pid, digest, kind)

    def vector_cache_put(self, pid: int, digest: str, vec: np.ndarray, kind: str = "full") -> None:
        from . import paper_feed

        paper_feed.vector_cache_put(pid, digest, vec, kind)


def paper_text(paper: dict) -> str:
    """库内文献用于嵌入的文本：标题 + 摘要（没有摘要时用 AI 快速总结的开头）。"""
    body = paper.get("abstract") or (paper.get("summary") or "").split("## Labels")[0]
    return f"{paper.get('title', '')}\n{body[:1500]}".strip()


def build_library_model(backend: RankerBackend, profile: dict | None = None) -> LibraryModel:
    profile = profile or {}
    papers = [p for p in backend.library_papers() if p["title"]]
    papers.sort(key=lambda p: p["pid"])
    def _vectors(texts: list[str], kind: str) -> np.ndarray:
        out = []
        for paper, text in zip(papers, texts):
            digest = hashlib.md5(text.encode()).hexdigest()
            vec = backend.vector_cache_get(paper["pid"], digest, kind)
            if vec is None:
                vec = backend.embed(text)
                backend.vector_cache_put(paper["pid"], digest, vec, kind)
            out.append(_normalize(np.asarray(vec, dtype=np.float32)))
        return np.vstack(out) if out else np.zeros((0, 1), dtype=np.float32)

    matrix = _vectors([paper_text(p) for p in papers], "full")
    title_matrix = _vectors([p["title"] for p in papers], "title")
    n = len(papers)
    # 库里没有"加入时间"，用编号（入库顺序）近似：最新的 1.0，最早的 0.0
    recency = np.linspace(0.0, 1.0, n, dtype=np.float32) if n > 1 else np.ones(n, dtype=np.float32)
    pids = [p["pid"] for p in papers]
    meta = {p["pid"]: {"cite": f"{p.get('first_author') or 'Unknown'} ({p.get('year') or 'n.d.'})",
                       "title": p["title"], "key": p["key"]} for p in papers}

    # 兴趣方向：从库里自动分组 + 兜底的"全库"；用户在画像里的启用/权重/补充覆盖默认值
    settings = profile.get("facets", {})
    facets = backend.library_facets(pids, matrix) + [Facet(id="library", name="全库", source="library",
                                                           description="整个文献库", pids=pids, weight=0.8)]
    known = set(pids)
    for facet in facets:
        facet.pids = [pid for pid in facet.pids if pid in known]
        custom = settings.get(facet.id, {})
        facet.enabled = bool(custom.get("enabled", True))
        facet.weight = float(custom.get("weight", facet.weight))
        facet.user_note = str(custom.get("user_note") or "")
    facets = [f for f in facets if f.pids]

    authors = backend.library_authors(papers)
    counts: dict[str, int] = {}
    for names in authors.values():
        for key in {author_key(name) for name in names}:
            if key:
                counts[key] = counts.get(key, 0) + 1

    # 库内文献与"其余库内文献中最近 3 篇"的平均相似度：新文章的相似度按这个分布换算成百分位，
    # 这样分数的含义是"它和你的库有多近——像一篇典型的库内文献那样近，就是 50%"
    # 与新文章完全相同的算法：相似度乘时间衰减，取最近 3 篇的平均
    decay = (0.85 + 0.15 * recency).astype(np.float32)

    def _calibrate(m: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
        if n <= 4:
            return np.array([0.5], dtype=np.float32), (0.3, 0.7)
        sims = (m @ m.T) * decay[None, :]
        np.fill_diagonal(sims, np.nan)
        top3 = np.sort(np.nan_to_num(sims, nan=-1.0), axis=1)[:, -3:].mean(axis=1)
        lo = float(np.nanmedian(sims))
        hi = float(np.median(top3))
        if hi - lo < 0.05:
            hi = lo + 0.05
        return np.sort(top3), (round(lo, 4), round(hi, 4))

    in_domain, full_anchor = _calibrate(matrix)
    in_domain_title, title_anchor = _calibrate(title_matrix)
    must_follow = {author_key(a) for a in profile.get("must_follow_authors", []) if author_key(a)}
    return LibraryModel(pids=pids, vectors=matrix, recency=recency, meta=meta, facets=facets,
                        author_counts=counts, in_domain_scores=in_domain, must_follow=must_follow,
                        title_vectors=title_matrix, in_domain_title_scores=in_domain_title,
                        anchors={"abstract": full_anchor, "title": title_anchor})


def cluster_library(pids: list[int], vectors: np.ndarray, previous: list[dict] | None = None,
                    k_range: tuple[int, int] = (6, 30), seed: int = 0) -> list[dict]:
    """把库内文献按内容自动分组（KMeans，归一化向量上等价于余弦距离），用轮廓系数在 k_range 内选分组数。

    previous：上一次的分组 [{id, pids}]。新分组与旧分组按文献重合度（Jaccard ≥ 0.5）一一对应，
    对得上的沿用旧编号——用户写在分组上的补充、权重不会因为重新分组而丢失。
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = len(pids)
    if n < 8:
        return [{"id": "cluster:1", "pids": list(pids)}] if n else []
    lo, hi = max(2, k_range[0]), max(2, min(k_range[1], n // 8))
    lo = min(lo, hi)
    sample = np.random.default_rng(seed).choice(n, size=min(n, 1500), replace=False)
    best = None
    for k in range(lo, hi + 1):
        labels = KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(vectors)
        score = silhouette_score(vectors[sample], labels[sample], metric="cosine") if k > 1 else -1
        if best is None or score > best[0] + 0.002:  # 分数相近时偏向更少的分组
            best = (score, k, labels)
    _, k, labels = best
    groups = [[pids[i] for i in range(n) if labels[i] == c] for c in range(k)]
    groups = [g for g in groups if g]

    # 编号稳定：按重合度贪心匹配旧分组
    previous = previous or []
    used_old: set[str] = set()
    result: list[dict] = []
    pairs = []
    for gi, group in enumerate(groups):
        for old in previous:
            a, b = set(group), set(old.get("pids", []))
            jaccard = len(a & b) / max(1, len(a | b))
            if jaccard >= 0.5:
                pairs.append((jaccard, gi, old["id"]))
    assigned: dict[int, str] = {}
    for _, gi, old_id in sorted(pairs, reverse=True):
        if gi not in assigned and old_id not in used_old:
            assigned[gi] = old_id
            used_old.add(old_id)
    existing_numbers = [int(str(o["id"]).split(":")[1]) for o in previous if str(o.get("id", "")).startswith("cluster:")]
    next_number = max(existing_numbers, default=0) + 1
    for gi, group in enumerate(groups):
        if gi not in assigned:
            assigned[gi] = f"cluster:{next_number}"
            next_number += 1
        result.append({"id": assigned[gi], "pids": sorted(group)})
    return sorted(result, key=lambda g: -len(g["pids"]))


def scale_similarity(raw: float, anchor: tuple[float, float]) -> float:
    """原始余弦相似度 → 0..1：等于“库内随便两篇”的相似度为 0，等于“典型库内文献与最近邻居”为 1。"""
    lo, hi = anchor
    return float(min(1.0, max(0.0, (raw - lo) / max(hi - lo, 1e-6))))


def _percentile(sorted_values: np.ndarray, value: float) -> float:
    if not len(sorted_values):
        return 0.0
    return float(np.searchsorted(sorted_values, value, side="right")) / len(sorted_values)


def compute_features(model: LibraryModel, backend: RankerBackend, *, vec: np.ndarray, text: str,
                     authors: list[str], positives: list[np.ndarray], negatives: list[np.ndarray],
                     title_only: bool = False) -> dict:
    """一篇新文章的全部排序特征（以及用于解释的"最相近的库内文献"）。

    title_only=True：新文章只有标题，与库内文献的"标题向量"比，百分位也按库内标题之间的相近程度换算——
    避免"只有标题 vs 标题+摘要"的不对称把分数系统性压低。
    """
    vec = _normalize(np.asarray(vec, dtype=np.float32))
    features: dict[str, Any] = {"info": "title" if title_only else "abstract"}
    use_titles = title_only and model.title_vectors is not None and len(model.title_vectors)
    library = model.title_vectors if use_titles else model.vectors
    anchor = model.anchors.get("title" if use_titles else "abstract") or (0.3, 0.7)
    if len(model.pids):
        # 时间衰减：越新的库内文献，相似度打折越少
        sims = (library @ vec) * (0.85 + 0.15 * model.recency)
        index = model.index_of()
        best_facet, best_score, facet_scores = None, -1.0, {}
        for facet in model.facets:
            if not facet.enabled:
                continue
            rows = [index[pid] for pid in facet.pids if pid in index]
            if not rows:
                continue
            top = np.sort(sims[rows])[-3:]
            score = float(top.mean()) * facet.weight
            facet_scores[facet.id] = round(score, 4)
            if score > best_score:
                best_facet, best_score = facet, score
        nearest = np.argsort(-sims)[:3]
        features["dense_raw"] = round(max(best_score, 0.0), 4)
        features["dense"] = round(scale_similarity(best_score, anchor), 4)
        features["facet"] = best_facet.id if best_facet else None
        features["facet_name"] = best_facet.name if best_facet else None
        features["facet_scores"] = facet_scores
        features["nearest"] = [{"pid": model.pids[i], "sim": round(float(sims[i]), 3), **model.meta[model.pids[i]]}
                               for i in nearest]
    else:
        features.update(dense_raw=0.0, dense=0.0, facet=None, facet_name=None, facet_scores={}, nearest=[])

    raw = backend.lexical_score(text)
    features["lexical_raw"] = round(raw, 3)
    features["lexical"] = round(raw / (raw + 20.0), 4) if raw > 0 else 0.0  # BM25 分数饱和映射到 0..1

    keys = [author_key(a) for a in authors if author_key(a)]
    peak = max(model.author_counts.values(), default=1)
    best_author, best_value = None, 0.0
    for key in keys:
        value = 1.0 if key in model.must_follow else math.log1p(model.author_counts.get(key, 0)) / math.log1p(peak)
        if value > best_value:
            best_author, best_value = key, value
    features["author"] = round(best_value, 4)
    features["author_match"] = best_author if best_value > 0 else None
    features["author_count"] = model.author_counts.get(best_author, 0) if best_author else 0

    # 反馈项：与你 👍 / 👎 过的文章的接近程度，用同一把标尺换算——只是“随便一篇”那样像不加也不扣分
    full_anchor = model.anchors.get("abstract") or (0.3, 0.7)
    pos = max((float(_normalize(p) @ vec) for p in positives), default=0.0)
    neg = max((float(_normalize(n) @ vec) for n in negatives), default=0.0)
    features["pos_sim"], features["neg_sim"] = round(pos, 4), round(neg, 4)
    closeness = lambda s: scale_similarity(s, full_anchor)  # noqa: E731
    features["feedback"] = round((closeness(pos) if positives else 0.0) - (closeness(neg) if negatives else 0.0), 4)
    return features


def base_score(features: dict, weights: dict | None = None) -> float:
    w = weights or COLD_START_WEIGHTS[features.get("info", "abstract")]
    score = (w["dense"] * features["dense"] + w["lexical"] * features["lexical"]
             + w["author"] * features["author"] + w["feedback"] * features["feedback"])
    return round(max(0.0, min(1.0, score)), 4)


def final_score(base: float, llm_score: float | None) -> float:
    if llm_score is None:
        return base
    return round((1 - LLM_BLEND) * base + LLM_BLEND * max(0.0, min(10.0, llm_score)) / 10.0, 4)


def assign_band(score: float, *, info: str, llm_score: float | None, features: dict | None = None,
                thresholds: dict | None = None) -> str:
    """auto = 自动入库；confirm = 等你确认；low = 只留在候选池。门槛按信息量（abstract / title）分开。"""
    auto = (thresholds or {}).get("auto", {}).get(info, AUTO_THRESHOLD[info])
    confirm = (thresholds or {}).get("confirm", {}).get(info, CONFIRM_THRESHOLD[info])
    if score >= auto and llm_score is not None and llm_score >= 8:
        if info == "abstract":
            return "auto"
        f = features or {}
        if f.get("dense", 0) >= TITLE_STRONG_DENSE or f.get("author", 0) >= TITLE_STRONG_AUTHOR:
            return "auto"
    if score >= confirm:
        return "confirm"
    return "low"


# ---------------------------------------------------------------------------
# LLM 判断（只对短名单，批量调用：画像放在前面，同一批次共享前缀缓存）
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """
你是一位〔研究领域〕方向研究者的文献助理，负责判断新发表的文章对他是否值得一读。
下面是他的研究画像（由他的文献库自动生成，并经他本人修改）：

{profile}

对每篇文章给出 0-10 分（10 = 必读，7-9 = 很可能想读，4-6 = 擦边，0-3 = 无关），并用一句〔回答语言〕说明理由。
理由要具体：说明它与哪个兴趣方向、哪篇库内文献相关，或为什么无关。只依据给出的标题、摘要、作者判断，不要编造文章内容。
如果文章属于画像中"不感兴趣"的方向，分数不超过 3。
只输出 JSON：{{"items": [{{"id": "F12", "score": 8, "reason": "……"}}]}}
""".strip().replace("〔研究领域〕", research_field()).replace("〔回答语言〕", answer_language_name())


def render_profile(model: LibraryModel, profile: dict) -> str:
    lines = ["兴趣方向："]
    for facet in model.facets:
        if facet.enabled and facet.source != "library":
            desc = f"：{facet.description[:300]}" if facet.description else ""
            note = f"\n  用户补充（优先）：{facet.user_note[:300]}" if facet.user_note else ""
            lines.append(f"- {facet.name}（库内 {len(facet.pids)} 篇）{desc}{note}")
    if profile.get("must_follow_authors"):
        lines.append("必看作者：" + "、".join(profile["must_follow_authors"]))
    if profile.get("avoid_topics"):
        lines.append("不感兴趣的方向：" + "；".join(profile["avoid_topics"]))
    if profile.get("notes"):
        lines.append("补充说明：" + profile["notes"])
    return "\n".join(lines)


def judge_batch(client: Any, model_name: str, profile_text: str, items: list[dict]) -> tuple[dict[str, dict], Any]:
    """items: [{id, title, abstract, authors, journal, nearest}]。返回 ({id: {score, reason}}, usage)。"""
    from .lit_search import parse_tool_arguments

    blocks = []
    for item in items:
        nearest = "；".join(f"{n['cite']} {n['title'][:80]}" for n in item.get("nearest", [])[:3])
        blocks.append(f"[{item['id']}] {item['title']}\n作者：{item.get('authors') or '未知'}\n来源：{item.get('journal') or '未知'}\n"
                      f"摘要：{(item.get('abstract') or '（无摘要）')[:1800]}\n库内最相近的文献：{nearest or '无'}")
    response = client.chat.completions.create(
        model=model_name, temperature=0.1, max_tokens=4096, stream=False,
        response_format={"type": "json_object"}, extra_body={"thinking": {"type": "disabled"}},
        messages=[{"role": "system", "content": JUDGE_SYSTEM.format(profile=profile_text)},
                  {"role": "user", "content": "\n\n".join(blocks)}],
    )
    parsed, _ = parse_tool_arguments(response.choices[0].message.content or "")
    out: dict[str, dict] = {}
    for entry in (parsed or {}).get("items", []):
        try:
            out[str(entry["id"])] = {"score": float(entry["score"]), "reason": str(entry.get("reason", "")).strip()}
        except (KeyError, TypeError, ValueError):
            continue
    return out, getattr(response, "usage", None)


def vector_to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def blob_to_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


def features_json(features: dict) -> str:
    return json.dumps(features, ensure_ascii=False)
