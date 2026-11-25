"""
兴趣方向（分组）与每组说明（逐行论点）的存储与维护规则。

分组的规则——AI 只能加组，删组 / 合并 / 拆分由用户决定：
- 第一次从全库自动聚类。之后已有分组固定：新进库的文献离某组足够近（不比该组 90% 的成员更远）就归进去，
  否则进入“未归组”。未归组的文献攒够一定数量、且彼此足够紧密时，AI 才新建一组。
- 用户删掉的分组留档：AI 不会再从同一批文献重新建出它（可在“已删除的分组”里恢复）。
- “整理分组”由用户触发，只重新划分用户从没动过的分组；动过的分组原样保留。
- 合并 / 拆分只作为建议出现，用户点了才执行。

每组说明的规则——一行一条论点或关键词，AI 只能提议新增，删除和修改只有用户能做：
- 新组第一次由 AI 提炼几条论点和关键词。之后 AI 只在这组新进了足够多文献时，
  只看“现有论点 + 新进的文献”提出新增论点，作为建议等用户采纳。
- 用户删掉的行留档，但**不会写进提示词**：AI 的新建议由程序在模型之外与留档内容比对，太像的直接丢掉。
- 删掉的行在回收站里可以随时恢复。普通删除满一定天数、且之后新进的文献确实支持这一点时，
  由程序（不经过模型）把原文重新放回建议区，注明原因；用户再次忽略则等待时间翻倍。“不再提示”则永久屏蔽。
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

MIN_GROUP = 8                 # 新建一组至少需要的文献数
RADIUS_PERCENTILE = 10        # 离组中心比该组 90% 的成员更远 → 不归进这组
NEW_PAPERS_FOR_UPDATE = 5     # 一组新进多少篇文献后，AI 才看一次是否有新论点
DELETE_WAIT_DAYS = 60         # 普通删除后，至少多少天才可能被重新提起
IGNORE_WAIT_DAYS = 30         # 忽略一条建议后，至少多少天才可能被重新提起
RESURFACE_MIN_SUPPORT = 3     # 重新提起需要的“支持这一点的新文献”篇数
DUPLICATE_SIMILARITY = 0.85   # AI 新建议与已有行（含回收站）的向量相似度达到这个值就视为重复
SMALL_GROUP = 5               # 少于这个篇数的组在页面上提示
K_RANGE = (6, 30)             # 全库分组 / 整理分组时，分组数的选择范围

_lock = threading.RLock()
_data_dir: Path | None = None
_now: Callable[[], dt.datetime] = dt.datetime.now   # 测试时替换
_claim_vector_cache: dict[str, np.ndarray] = {}


def set_data_dir(path: Path) -> None:
    global _data_dir
    _data_dir = Path(path)


def _path() -> Path:
    if _data_dir is None:
        raise RuntimeError("facet_store 尚未设置数据目录")
    return _data_dir / "facets.json"


def _iso(t: dt.datetime | None = None) -> str:
    return (t or _now()).isoformat(timespec="seconds")


def _parse(text: str | None) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(text) if text else None
    except ValueError:
        return None


def empty() -> dict:
    return {"version": 2, "groups": [], "unassigned": [], "library_pids": [], "grouped_at": None,
            "suggestions": [], "dismissed_suggestions": [], "deleted_groups": [],
            "next_group": 1, "next_claim": 1}


def load() -> dict:
    path = _path()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            base = empty()
            base.update(data)
            return base
        except json.JSONDecodeError:
            pass
    return empty()


def save(data: dict) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def group_of(data: dict, group_id: str) -> dict:
    for group in data["groups"]:
        if group["id"] == group_id:
            return group
    raise ValueError("找不到这个分组")


def _new_group(data: dict, pids: list[int], created_by: str = "ai") -> dict:
    group = {"id": f"cluster:{data['next_group']}", "pids": sorted(pids), "representative": [],
             "created_by": created_by, "created_at": _iso(), "touched": False,
             "name": "", "name_by": "ai", "claims": [], "described_pids": [], "described_at": None}
    data["next_group"] += 1
    return group


def _new_claim(data: dict, text: str, kind: str, by: str, status: str) -> dict:
    claim = {"id": f"c{data['next_claim']}", "text": text.strip(), "kind": kind, "by": by,
             "status": status, "created_at": _iso(), "changed_at": _iso()}
    data["next_claim"] += 1
    return claim


# ---------------------------------------------------------------------------
# 分组
# ---------------------------------------------------------------------------

def _unit(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


def _centroid(vectors: np.ndarray) -> np.ndarray:
    return _unit(vectors.mean(axis=0))


def _member_sims(vectors: np.ndarray) -> np.ndarray:
    return vectors @ _centroid(vectors)


def _representative(group: dict, index: dict[int, int], vectors: np.ndarray, top: int = 5) -> list[int]:
    rows = [index[p] for p in group["pids"] if p in index]
    if not rows:
        return []
    order = np.argsort(-_member_sims(vectors[rows]))[:top]
    members = [p for p in group["pids"] if p in index]
    return [members[i] for i in order]


def _cohesive_subgroups(pids: list[int], vectors: np.ndarray, min_cohesion: float, seed: int = 0) -> list[list[int]]:
    """从未归组文献里找出“够多、够紧密”的子群：每个子群至少 MIN_GROUP 篇，紧密程度不低于现有最松散的分组。"""
    from sklearn.cluster import KMeans

    n = len(pids)
    if n < MIN_GROUP:
        return []
    best: list[list[int]] = []
    for k in range(1, max(1, n // MIN_GROUP) + 1):
        labels = np.zeros(n, dtype=int) if k == 1 else KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(vectors)
        found = []
        for c in range(k):
            rows = np.where(labels == c)[0]
            if len(rows) >= MIN_GROUP and float(_member_sims(vectors[rows]).mean()) >= min_cohesion:
                found.append([pids[i] for i in rows])
        if sum(map(len, found)) > sum(map(len, best)):
            best = found
    return best


def _blocked_by_deleted(data: dict, pids: list[int]) -> bool:
    candidate = set(pids)
    for old in data.get("deleted_groups", []):
        previous = set(old.get("pids", []))
        if len(candidate & previous) / max(1, len(candidate | previous)) >= 0.5:
            return True
    return False


def sync_groups(pids: list[int], vectors: np.ndarray, *, reorganize: bool = False) -> dict:
    """按当前文献库更新分组（规则见模块说明）。vectors 与 pids 一一对应、已归一化。"""
    from . import feed_ranker

    with _lock:
        data = load()
        index = {pid: i for i, pid in enumerate(pids)}
        library = set(pids)
        changed = False

        if not data["groups"] and not data["unassigned"]:
            for cluster in feed_ranker.cluster_library(pids, vectors, k_range=K_RANGE):
                data["groups"].append(_new_group(data, cluster["pids"]))
            data["grouped_at"] = _iso()
            changed = True
        elif reorganize:
            changed = _reorganize(data, pids, vectors, index) or changed

        # 移出文献库的文献
        for group in data["groups"]:
            kept = [p for p in group["pids"] if p in library]
            if len(kept) != len(group["pids"]):
                group["pids"], changed = kept, True
        before = len(data["groups"])
        data["groups"] = [g for g in data["groups"] if g["pids"] or g.get("touched") or g["claims"]]
        changed = changed or len(data["groups"]) != before

        # 新进库的文献 + 之前未归组的文献：离某组足够近就归进去
        placed = {p for g in data["groups"] for p in g["pids"]}
        pending = [p for p in dict.fromkeys(list(data["unassigned"]) + pids) if p in library and p not in placed]
        groups = [g for g in data["groups"] if len([p for p in g["pids"] if p in index]) >= 2]
        unassigned = pending
        if groups and pending:
            centroids = np.vstack([_centroid(vectors[[index[p] for p in g["pids"] if p in index]]) for g in groups])
            radius = np.array([np.percentile(_member_sims(vectors[[index[p] for p in g["pids"] if p in index]]),
                                             RADIUS_PERCENTILE) for g in groups])
            sims = vectors[[index[p] for p in pending]] @ centroids.T
            unassigned = []
            for row, pid in enumerate(pending):
                best = int(np.argmax(sims[row]))
                if sims[row, best] >= radius[best]:
                    groups[best]["pids"] = sorted(groups[best]["pids"] + [pid])
                    changed = True
                else:
                    unassigned.append(pid)

        # 未归组的攒够了、彼此够紧密 → AI 新建一组（与用户删过的分组高度重合的不建）
        if len(unassigned) >= MIN_GROUP:
            cohesion = [float(_member_sims(vectors[[index[p] for p in g["pids"] if p in index]]).mean())
                        for g in groups] or [0.0]
            for sub in _cohesive_subgroups(unassigned, vectors[[index[p] for p in unassigned]], min(cohesion)):
                if _blocked_by_deleted(data, sub):
                    continue
                data["groups"].append(_new_group(data, sub))
                unassigned = [p for p in unassigned if p not in set(sub)]
                changed = True
        if sorted(unassigned) != sorted(data["unassigned"]):
            changed = True
        data["unassigned"] = sorted(unassigned)

        if changed or sorted(data["library_pids"]) != sorted(pids):
            for group in data["groups"]:
                group["representative"] = _representative(group, index, vectors)
            data["suggestions"] = _suggestions(data, index, vectors)
            data["library_pids"] = sorted(pids)
            save(data)
        return data


def _reorganize(data: dict, pids: list[int], vectors: np.ndarray, index: dict[int, int]) -> bool:
    """用户点“整理分组”：只重新划分从没动过的分组（连同未归组文献），动过的原样保留。"""
    from . import feed_ranker

    fixed = [g for g in data["groups"] if g.get("touched")]
    free = [g for g in data["groups"] if not g.get("touched")]
    taken = {p for g in fixed for p in g["pids"]}
    pool = [p for p in pids if p not in taken]
    if len(pool) < MIN_GROUP:
        return False
    clusters = feed_ranker.cluster_library(pool, vectors[[index[p] for p in pool]],
                                           previous=[{"id": g["id"], "pids": g["pids"]} for g in free], k_range=K_RANGE)
    by_id = {g["id"]: g for g in free}
    rebuilt = []
    for cluster in clusters:
        old = by_id.get(cluster["id"])
        if old is not None:               # 与旧组对得上：沿用编号、名称和论点（组内文献变了，AI 随后会看新进的文献）
            old["pids"] = cluster["pids"]
            rebuilt.append(old)
        else:
            group = _new_group(data, cluster["pids"])
            rebuilt.append(group)
    # cluster_library 给新组的编号可能与 next_group 冲突：统一改用 _new_group 分配的编号
    data["groups"] = fixed + rebuilt
    data["unassigned"] = []
    data["grouped_at"] = _iso()
    return True


def _silhouette(vectors: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import silhouette_score

    if len(set(labels.tolist())) < 2 or len(labels) < 4:
        return 0.0
    rng = np.random.default_rng(0)
    rows = rng.choice(len(labels), size=min(len(labels), 1500), replace=False)
    if len(set(labels[rows].tolist())) < 2:
        return 0.0
    return float(silhouette_score(vectors[rows], labels[rows], metric="cosine"))


def _suggestions(data: dict, index: dict[int, int], vectors: np.ndarray) -> list[dict]:
    """合并 / 拆分建议：与“现有分组之间的区分度”比——
    某组内部能分成两块、且两块之间的区分度不低于现有分组之间的平均区分度 → 建议拆分；
    两组之间的区分度不到平均水平的一半 → 建议合并。"""
    from sklearn.cluster import KMeans

    groups = [g for g in data["groups"] if len([p for p in g["pids"] if p in index]) >= 2]
    if len(groups) < 2:
        return []
    rows_of = {g["id"]: [index[p] for p in g["pids"] if p in index] for g in groups}
    all_rows = [r for g in groups for r in rows_of[g["id"]]]
    labels = np.concatenate([[i] * len(rows_of[g["id"]]) for i, g in enumerate(groups)])
    overall = max(0.02, _silhouette(vectors[all_rows], labels))
    dismissed = set(data.get("dismissed_suggestions", []))
    out = []
    centroids = np.vstack([_centroid(vectors[rows_of[g["id"]]]) for g in groups])
    sims = centroids @ centroids.T
    np.fill_diagonal(sims, -1)
    seen = set()
    for i, group in enumerate(groups):
        j = int(np.argmax(sims[i]))
        key = "merge:" + "+".join(sorted([group["id"], groups[j]["id"]]))
        if key in seen or key in dismissed:
            continue
        seen.add(key)
        rows = rows_of[group["id"]] + rows_of[groups[j]["id"]]
        pair = np.array([0] * len(rows_of[group["id"]]) + [1] * len(rows_of[groups[j]["id"]]))
        score = _silhouette(vectors[rows], pair)
        if score < overall * 0.5:
            out.append({"key": key, "type": "merge", "groups": sorted([group["id"], groups[j]["id"]]),
                        "reason": f"两组内容区分度只有 {score:.2f}（分组之间平均 {overall:.2f}）"})
    for group in groups:
        rows = rows_of[group["id"]]
        key = f"split:{group['id']}:{len(rows)}"
        if len(rows) < 3 * MIN_GROUP or key in dismissed:
            continue
        halves = KMeans(n_clusters=2, n_init=5, random_state=0).fit_predict(vectors[rows])
        sizes = np.bincount(halves, minlength=2)
        score = _silhouette(vectors[rows], halves)
        if min(sizes) >= MIN_GROUP and score >= overall:
            members = [p for p in group["pids"] if p in index]
            parts = [[members[k] for k in range(len(members)) if halves[k] == c] for c in (0, 1)]
            out.append({"key": key, "type": "split", "groups": [group["id"]], "parts": parts,
                        "reason": f"组内两块之间的区分度 {score:.2f}，不低于分组之间的平均水平 {overall:.2f}"})
    return out


# ---------------------------------------------------------------------------
# 给排序用的兴趣方向
# ---------------------------------------------------------------------------

def facets_for_ranking(data: dict) -> list:
    from .feed_ranker import Facet

    facets = []
    for group in data["groups"]:
        if not group["pids"]:
            continue
        active = [c for c in group["claims"] if c["status"] == "active"]
        ai_lines = [c["text"] for c in active if c["kind"] == "claim" and c["by"] == "ai" and not c.get("edited")]
        user_lines = [c["text"] for c in active if c["kind"] == "claim" and (c["by"] == "user" or c.get("edited"))]
        keywords = [c["text"] for c in active if c["kind"] == "keyword"]
        description = "；".join(ai_lines) + (f"\n关键词：{', '.join(keywords)}" if keywords else "")
        facets.append(Facet(id=group["id"], name=group.get("name") or f"未命名分组 {group['id'].split(':')[1]}",
                            source="cluster", description=description.strip(), user_note="；".join(user_lines),
                            pids=list(group["pids"])))
    if data.get("unassigned"):
        facets.append(Facet(id="unassigned", name="未归组文献", source="unassigned",
                            description="还没归进任何一组的库内文献", pids=list(data["unassigned"]), weight=0.9))
    return facets


# ---------------------------------------------------------------------------
# AI 提炼与新增建议（模型只看到现有论点和文献；回收站内容只在程序里比对）
# ---------------------------------------------------------------------------

INITIAL_SYSTEM = """
下面是一位地球化学、宇宙化学与行星科学研究者文献库中，按内容自动分到同一组的文献（标题 + AI 总结的核心句；前几篇最有代表性）。
请为这一组起一个简短的中文名称（不超过 14 个字），写 3-5 条中文论点，每条一句，概括它关注的科学问题、对象与方法；再给出 8-15 个英文关键词。
只依据给出的文献，不要加入列表里没有的内容。只输出 JSON：{"name": "……", "claims": ["……"], "keywords": ["…"]}
""".strip()

UPDATE_SYSTEM = """
下面是研究者文献库中一组文献的现有说明（每行一条），以及这组最近新进的文献（标题 + AI 总结的核心句）。
只写出这些新文献带来的、现有说明还没有覆盖的内容：0-3 条中文论点（每条一句）和 0-5 个英文关键词。
没有新内容就返回空数组。只依据给出的文献。只输出 JSON：{"claims": ["……"], "keywords": ["…"]}
""".strip()


def _norm(text: str) -> str:
    return re.sub(r"[\s，,。；;：:、.\-_]+", "", text or "").lower()


def _claim_vector(text: str, embed: Callable[[str], np.ndarray]) -> np.ndarray:
    key = hashlib.md5(text.encode()).hexdigest()
    if key not in _claim_vector_cache:
        _claim_vector_cache[key] = _unit(np.asarray(embed(text), dtype=np.float32))
    return _claim_vector_cache[key]


def _is_duplicate(text: str, kind: str, existing: list[dict], embed: Callable[[str], np.ndarray] | None) -> bool:
    """与已有行（任何状态，包括回收站和“不再提示”）重复？关键词只比字面，论点再比语义。"""
    same_kind = [c for c in existing if c["kind"] == kind]
    if any(_norm(c["text"]) == _norm(text) for c in same_kind):
        return True
    if kind != "claim" or embed is None or not same_kind:
        return False
    vec = _claim_vector(text, embed)
    return any(float(vec @ _claim_vector(c["text"], embed)) >= DUPLICATE_SIMILARITY for c in same_kind)


def _paper_line(paper: dict, quick_line: Callable[[str], str]) -> str:
    summary = quick_line(paper.get("summary", "")) if paper.get("summary") else ""
    return f"- {paper['title']}" + (f"：{summary}" if summary else "")


def describe(papers: list[dict], *, ask: Callable[[str, str], dict | None], embed: Callable[[str], np.ndarray] | None,
             quick_line: Callable[[str], str], pids: list[int] | None = None, vectors: np.ndarray | None = None,
             force: bool = False, max_calls: int = 40) -> dict:
    """为新组提炼论点；为新进了足够多文献的组提出新增建议；把满足条件的回收站内容重新放回建议区。
    LLM 调用在锁外进行，结果再合并进最新的数据（期间用户的编辑不会被覆盖）。"""
    by_pid = {p["pid"]: p for p in papers}
    snapshot = load()
    jobs = []
    for group in snapshot["groups"]:
        if not group["pids"]:
            continue
        if not group["claims"] and not group.get("described_at"):
            jobs.append(("initial", group["id"], group))
            continue
        new = [p for p in group["pids"] if p not in set(group.get("described_pids", []))]
        if len(new) >= NEW_PAPERS_FOR_UPDATE or (force and new):
            jobs.append(("update", group["id"], group))
    results = {}
    for kind, group_id, group in jobs[:max_calls]:
        if kind == "initial":
            ordered = list(dict.fromkeys(group.get("representative", []) + group["pids"][::-1]))
            lines = [_paper_line(by_pid[p], quick_line) for p in ordered if p in by_pid][:120]
            parsed = ask(INITIAL_SYSTEM, "\n".join(lines))
        else:
            current = [c["text"] for c in group["claims"] if c["status"] == "active" and c["kind"] == "claim"]
            keywords = [c["text"] for c in group["claims"] if c["status"] == "active" and c["kind"] == "keyword"]
            new = [p for p in group["pids"] if p not in set(group.get("described_pids", []))]
            lines = [_paper_line(by_pid[p], quick_line) for p in new[::-1] if p in by_pid][:60]
            prompt = ("现有说明：\n" + "\n".join(f"- {t}" for t in current) +
                      (f"\n关键词：{', '.join(keywords)}" if keywords else "") +
                      "\n\n最近新进的文献：\n" + "\n".join(lines))
            parsed = ask(UPDATE_SYSTEM, prompt)
        if parsed is not None:
            results[group_id] = (kind, parsed, list(group["pids"]))

    with _lock:
        data = load()
        for group in data["groups"]:
            if group["id"] in results:
                kind, parsed, seen_pids = results[group["id"]]
                status = "active" if kind == "initial" and not group["claims"] else "suggested"
                if kind == "initial" and group.get("name_by") != "user" and parsed.get("name"):
                    group["name"] = str(parsed["name"]).strip()[:20]
                for text in [str(t) for t in parsed.get("claims", []) if str(t).strip()][:5]:
                    if not _is_duplicate(text, "claim", group["claims"], embed):
                        group["claims"].append(_new_claim(data, text[:300], "claim", "ai", status))
                for text in [str(t) for t in parsed.get("keywords", []) if str(t).strip()][:15]:
                    if not _is_duplicate(text, "keyword", group["claims"], embed):
                        group["claims"].append(_new_claim(data, text[:60], "keyword", "ai", status))
                group["described_pids"] = seen_pids
                group["described_at"] = _iso()
            if embed is not None and pids is not None and vectors is not None:
                _resurface(group, {pid: i for i, pid in enumerate(pids)}, vectors, embed)
        save(data)
        return data


def _resurface(group: dict, index: dict[int, int], vectors: np.ndarray, embed: Callable[[str], np.ndarray]) -> None:
    """回收站 / 被忽略的行：等待期已过，且之后新进的文献里至少有几篇明显支持它 → 原文放回建议区（不经过模型）。
    “支持”：这篇新文献与该行的相似度，不低于删除时组内文献的中位数（至少像一篇典型的组内文献那样相关）。"""
    now = _now()
    for claim in group["claims"]:
        if claim["status"] not in {"deleted", "ignored"} or claim["kind"] != "claim":
            continue
        eligible = _parse(claim.get("eligible_at"))
        if eligible is None or now < eligible:
            continue
        before = [p for p in claim.get("snapshot_pids", []) if p in index]
        new = [p for p in group["pids"] if p in index and p not in set(claim.get("snapshot_pids", []))]
        if len(before) < 3 or len(new) < RESURFACE_MIN_SUPPORT:
            continue
        vec = _claim_vector(claim["text"], embed)
        threshold = float(np.percentile(vectors[[index[p] for p in before]] @ vec, 50))
        support = [p for p in new if float(vectors[index[p]] @ vec) >= threshold]
        if len(support) >= RESURFACE_MIN_SUPPORT:
            claim["resurfaced"] = {"from": claim["status"], "at": _iso(now), "support": support[:6],
                                   "support_count": len(support)}
            claim["status"] = "suggested"
            claim["changed_at"] = _iso(now)


# ---------------------------------------------------------------------------
# 用户操作
# ---------------------------------------------------------------------------

def _touch(group: dict) -> None:
    group["touched"] = True


def _claim(group: dict, claim_id: str) -> dict:
    for claim in group["claims"]:
        if claim["id"] == claim_id:
            return claim
    raise ValueError("找不到这一行")


def _shelve(claim: dict, group: dict, status: str, wait_days: int) -> None:
    claim.update(status=status, changed_at=_iso(), wait_days=wait_days,
                 eligible_at=_iso(_now() + dt.timedelta(days=wait_days)), snapshot_pids=list(group["pids"]))


def edit_claims(group_id: str, op: str, **kw: Any) -> dict:
    """逐行编辑。op：
    - edit {claim, text}：改一行（改成空等于删除）；
    - insert {after, text, kind}：在 after 那一行后面插入你写的一行（after 为空则插在最前）；
    - split {claim, before, after}：在一行中间回车——这一行改成 before，后面插入 after；
    - delete {claim}：删除（你写的行直接删掉；AI 写的行进回收站，等待期后有新证据可能被重新提起）；
    - block {claim}：不再提示（永久屏蔽，除非你在回收站恢复）；
    - restore {claim}：从回收站恢复；
    - accept {claim} / ignore {claim}：采纳 / 忽略一条 AI 建议（忽略被重新提起的行，等待时间翻倍）；
    - rename {name}：改组名（改过之后 AI 不再给这组起名）。
    返回 {"group": 更新后的分组, "claim": 新插入行的 id（如有）}。"""
    with _lock:
        data = load()
        group = group_of(data, group_id)
        result: dict[str, Any] = {}
        if op == "rename":
            group["name"], group["name_by"] = str(kw.get("name") or "").strip()[:20], "user"
        elif op in {"edit", "split"}:
            claim = _claim(group, kw["claim"])
            text = str(kw.get("before" if op == "split" else "text") or "").strip()
            if text != claim["text"]:
                if not text and op == "edit":
                    return edit_claims_locked_delete(data, group, claim)
                if text:
                    claim.update(text=text[:300], edited=True if claim["by"] == "ai" else claim.get("edited", False),
                                 changed_at=_iso())
            if op == "split":
                tail = str(kw.get("after") or "").strip()
                result["claim"] = _insert(data, group, claim["id"], tail, claim["kind"])
        elif op == "insert":
            result["claim"] = _insert(data, group, kw.get("after"), str(kw.get("text") or "").strip(),
                                      kw.get("kind") or "claim")
        elif op == "delete":
            return edit_claims_locked_delete(data, group, _claim(group, kw["claim"]))
        elif op == "block":
            claim = _claim(group, kw["claim"])
            claim.update(status="blocked", changed_at=_iso())
            claim.pop("resurfaced", None)
        elif op == "restore":
            claim = _claim(group, kw["claim"])
            claim.update(status="active", changed_at=_iso())
            claim.pop("resurfaced", None)
        elif op == "accept":
            claim = _claim(group, kw["claim"])
            if claim["status"] != "suggested":
                raise ValueError("这一行不是待采纳的建议")
            claim.update(status="active", changed_at=_iso())
            claim.pop("resurfaced", None)
        elif op == "ignore":
            claim = _claim(group, kw["claim"])
            if claim["status"] != "suggested":
                raise ValueError("这一行不是待采纳的建议")
            again = claim.pop("resurfaced", None)
            if again:
                _shelve(claim, group, again["from"], int(claim.get("wait_days") or DELETE_WAIT_DAYS) * 2)
            else:
                _shelve(claim, group, "ignored", IGNORE_WAIT_DAYS)
        else:
            raise ValueError(f"未知操作：{op}")
        _touch(group)
        save(data)
        return {"group": group, **result}


def edit_claims_locked_delete(data: dict, group: dict, claim: dict) -> dict:
    if claim["by"] == "user":
        group["claims"] = [c for c in group["claims"] if c["id"] != claim["id"]]
    else:
        _shelve(claim, group, "deleted", DELETE_WAIT_DAYS)
        claim.pop("resurfaced", None)
    _touch(group)
    save(data)
    return {"group": group}


def _insert(data: dict, group: dict, after: str | None, text: str, kind: str) -> str:
    claim = _new_claim(data, text[:300], "keyword" if kind == "keyword" else "claim", "user", "active")
    claims = group["claims"]
    position = 0
    if after:
        position = next((i + 1 for i, c in enumerate(claims) if c["id"] == after), len(claims))
    claims.insert(position, claim)
    return claim["id"]


def delete_group(group_id: str) -> dict:
    """删除分组：组内文献重新归到最近的组或“未归组”；分组留档，AI 不会从同一批文献重新建出它。"""
    with _lock:
        data = load()
        group = group_of(data, group_id)
        data["groups"] = [g for g in data["groups"] if g["id"] != group_id]
        data["deleted_groups"] = (data.get("deleted_groups", []) + [{**group, "deleted_at": _iso()}])[-50:]
        data["unassigned"] = sorted(set(data["unassigned"]) | set(group["pids"]))
        data["suggestions"] = [s for s in data["suggestions"] if group_id not in s["groups"]]
        save(data)
        return data


def restore_group(group_id: str) -> dict:
    """从“已删除的分组”恢复：取回仍在库里的文献（从它们现在所在的组里移出来）。"""
    with _lock:
        data = load()
        old = next((g for g in data.get("deleted_groups", []) if g["id"] == group_id), None)
        if old is None:
            raise ValueError("找不到这个已删除的分组")
        library = set(data["library_pids"])
        pids = [p for p in old["pids"] if p in library]
        for group in data["groups"]:
            group["pids"] = [p for p in group["pids"] if p not in set(pids)]
        data["unassigned"] = [p for p in data["unassigned"] if p not in set(pids)]
        old = {k: v for k, v in old.items() if k != "deleted_at"}
        old["pids"] = sorted(pids)
        data["groups"].append(old)
        data["deleted_groups"] = [g for g in data["deleted_groups"] if g["id"] != group_id]
        save(data)
        return data


def merge_groups(source_id: str, target_id: str) -> dict:
    """把 source 合并进 target：target 保留编号和名称，source 的所有行（含回收站）接在后面。"""
    if source_id == target_id:
        raise ValueError("不能合并到自己")
    with _lock:
        data = load()
        source, target = group_of(data, source_id), group_of(data, target_id)
        target["pids"] = sorted(set(target["pids"]) | set(source["pids"]))
        existing = target["claims"]
        for claim in source["claims"]:
            if not any(_norm(c["text"]) == _norm(claim["text"]) and c["kind"] == claim["kind"] for c in existing):
                existing.append(claim)
        target["described_pids"] = sorted(set(target.get("described_pids", [])) | set(source.get("described_pids", [])))
        _touch(target)
        data["groups"] = [g for g in data["groups"] if g["id"] != source_id]
        data["suggestions"] = [s for s in data["suggestions"] if source_id not in s["groups"] and target_id not in s["groups"]]
        save(data)
        return data


def apply_suggestion(key: str) -> dict:
    with _lock:
        data = load()
        suggestion = next((s for s in data["suggestions"] if s["key"] == key), None)
        if suggestion is None:
            raise ValueError("这条建议已经失效")
    if suggestion["type"] == "merge":
        a, b = suggestion["groups"]
        groups = {g["id"]: g for g in load()["groups"]}
        # 并入篇数更多的那组
        source, target = (a, b) if len(groups[a]["pids"]) <= len(groups[b]["pids"]) else (b, a)
        return merge_groups(source, target)
    with _lock:
        data = load()
        group = group_of(data, suggestion["groups"][0])
        big, small = sorted(suggestion["parts"], key=len, reverse=True)
        current = set(group["pids"])
        group["pids"] = sorted(p for p in big if p in current)
        _touch(group)
        new = _new_group(data, [p for p in small if p in current])
        data["groups"].append(new)
        data["suggestions"] = [s for s in data["suggestions"] if s["key"] != key]
        save(data)
        return data


def dismiss_suggestion(key: str) -> dict:
    with _lock:
        data = load()
        data["dismissed_suggestions"] = (data.get("dismissed_suggestions", []) + [key])[-200:]
        data["suggestions"] = [s for s in data["suggestions"] if s["key"] != key]
        save(data)
        return data


def mark_touched(group_ids: list[str]) -> None:
    """用户在画像里改了启用 / 权重：这些组也算“动过”，整理分组时保留。"""
    with _lock:
        data = load()
        hit = False
        for group in data["groups"]:
            if group["id"] in group_ids and not group.get("touched"):
                group["touched"] = hit = True
        if hit:
            save(data)


# ---------------------------------------------------------------------------
# 旧格式迁移（clusters.json + facet_descriptions.json + profile.json 里的 user_note）
# ---------------------------------------------------------------------------

def migrate_legacy(profile: dict) -> bool:
    if _data_dir is None or _path().exists():
        return False
    clusters_path, desc_path = _data_dir / "clusters.json", _data_dir / "facet_descriptions.json"
    if not clusters_path.exists():
        return False
    try:
        clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
        descriptions = json.loads(desc_path.read_text(encoding="utf-8")) if desc_path.exists() else {}
    except json.JSONDecodeError:
        return False
    data = empty()
    numbers = [int(c["id"].split(":")[1]) for c in clusters.get("clusters", []) if str(c.get("id", "")).startswith("cluster:")]
    data["next_group"] = max(numbers, default=0) + 1
    for cluster in clusters.get("clusters", []):
        desc = descriptions.get(cluster["id"], {})
        edited = bool(desc.get("edited_by_user_at"))
        group = {"id": cluster["id"], "pids": cluster.get("pids", []), "representative": cluster.get("representative", []),
                 "created_by": "ai", "created_at": clusters.get("created_at") or _iso(), "touched": edited,
                 "name": desc.get("name", ""), "name_by": "user" if edited else "ai", "claims": [],
                 "described_pids": desc.get("pids", []), "described_at": desc.get("generated_at")}
        text = (desc.get("text") or "").split("\n关键词：")[0]
        for sentence in [s.strip() for s in re.split(r"(?<=[。；！？])", text) if s.strip()]:
            group["claims"].append(_new_claim(data, sentence, "claim", "user" if edited else "ai", "active"))
        for keyword in desc.get("keywords", []):
            group["claims"].append(_new_claim(data, keyword, "keyword", "ai", "active"))
        note = ((profile.get("facets") or {}).get(cluster["id"]) or {}).get("user_note")
        if note:
            group["claims"].append(_new_claim(data, note, "claim", "user", "active"))
            group["touched"] = True
        data["groups"].append(group)
    data["library_pids"] = clusters.get("pids", [])
    data["grouped_at"] = clusters.get("created_at")
    save(data)
    return True
