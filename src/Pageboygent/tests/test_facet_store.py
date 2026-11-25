from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from Pageboygent import facet_store, feed_ranker

DIM = 16
CENTERS = np.eye(5, DIM, dtype=np.float32)


def blob(center: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = CENTERS[center] + rng.normal(0, 0.05, (n, DIM))
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


# 论点文本 → 向量：让“讲同一件事”的论点与对应主题的文献落在同一方向
CLAIM_TOPIC = {"挥发分在岩浆洋阶段蒸发亏损": 0, "锌同位素记录蒸发": 0, "Zn 同位素记录了蒸发过程": 0,
               "行星形成的数值模拟": 1, "月壳的年代学": 2}


def embed(text: str) -> np.ndarray:
    topic = CLAIM_TOPIC.get(text, 4)
    return CENTERS[topic] + 0.01


@pytest.fixture
def store(tmp_path, monkeypatch):
    facet_store.set_data_dir(tmp_path)
    facet_store._claim_vector_cache.clear()
    clock = {"now": dt.datetime(2026, 1, 1)}
    monkeypatch.setattr(facet_store, "_now", lambda: clock["now"])
    monkeypatch.setattr(facet_store, "K_RANGE", (2, 6))
    yield clock
    facet_store._claim_vector_cache.clear()


def library(*blocks):
    vectors = np.vstack([b for b in blocks])
    return list(range(1, len(vectors) + 1)), vectors


class Asker:
    """只回答提到某篇文献（默认第 3 篇，属于主题 0 那组）的请求，其余返回空。"""

    def __init__(self, reply=None, about="- paper 3\n"):
        self.reply, self.about = reply, about
        self.prompts = []

    def __call__(self, system, user):
        if self.about in user + "\n":
            self.prompts.append(user)
            return self.reply or {"claims": [], "keywords": []}
        return {"claims": [], "keywords": []}


def papers_for(pids):
    return [{"pid": p, "title": f"paper {p}", "summary": ""} for p in pids]


def group_with(data, pid):
    return next(g for g in data["groups"] if pid in g["pids"])


def test_existing_groups_stay_fixed_and_ai_only_adds_groups(store):
    pids, vectors = library(blob(0, 20, 1), blob(1, 20, 2), blob(2, 20, 3))
    data = facet_store.sync_groups(pids, vectors)
    assert len(data["groups"]) == 3
    before = {g["id"]: set(g["pids"]) for g in data["groups"]}
    # 新进的文献：3 篇属于已有主题，9 篇是全新主题
    more_pids, more = library(blob(0, 20, 1), blob(1, 20, 2), blob(2, 20, 3), blob(0, 3, 9), blob(3, 9, 10))
    data = facet_store.sync_groups(more_pids, more)
    groups = {g["id"]: set(g["pids"]) for g in data["groups"]}
    for gid, members in before.items():
        assert members <= groups[gid]                        # 已有分组的成员一个不少，编号不变
    assert {61, 62, 63} <= groups[group_with(data, 1)["id"]]  # 近的归进去
    new = [g for g in data["groups"] if g["id"] not in before]
    assert len(new) == 1 and set(new[0]["pids"]) == set(range(64, 73))   # 离群的攒够了才新建一组
    assert data["unassigned"] == []


def test_scattered_outliers_stay_unassigned_until_enough(store):
    pids, vectors = library(blob(0, 20, 1), blob(1, 20, 2))
    facet_store.sync_groups(pids, vectors)
    more_pids, more = library(blob(0, 20, 1), blob(1, 20, 2), blob(3, 3, 5))
    data = facet_store.sync_groups(more_pids, more)
    assert sorted(data["unassigned"]) == [41, 42, 43] and len(data["groups"]) == 2
    assert any(f.id == "unassigned" for f in facet_store.facets_for_ranking(data))   # 评分时照样参与比较


def test_deleted_group_is_not_recreated_and_can_be_restored(store):
    pids, vectors = library(blob(0, 20, 1), blob(1, 20, 2), blob(3, 12, 3))
    data = facet_store.sync_groups(pids, vectors)
    victim = group_with(data, 45)["id"]
    facet_store.delete_group(victim)
    data = facet_store.sync_groups(pids + [], vectors)   # 再同步一次：同一批文献不会被重新建成一组
    assert victim not in {g["id"] for g in data["groups"]}
    assert all(45 not in g["pids"] for g in data["groups"]) and 45 in data["unassigned"]
    data = facet_store.restore_group(victim)
    assert 45 in group_with(data, 45)["pids"] and group_with(data, 45)["id"] == victim


def test_reorganize_keeps_touched_groups(store):
    pids, vectors = library(blob(0, 20, 1), blob(1, 20, 2), blob(2, 20, 3))
    data = facet_store.sync_groups(pids, vectors)
    kept = group_with(data, 1)
    facet_store.edit_claims(kept["id"], "rename", name="挥发分")
    data = facet_store.sync_groups(pids, vectors, reorganize=True)
    again = group_with(data, 1)
    assert again["id"] == kept["id"] and again["name"] == "挥发分" and set(again["pids"]) == set(kept["pids"])
    assert len({g["id"] for g in data["groups"]}) == len(data["groups"])   # 编号不冲突


def _described_group(store):
    pids, vectors = library(blob(0, 20, 1), blob(1, 20, 2))
    data = facet_store.sync_groups(pids, vectors)
    asker = Asker({"name": "MVE", "claims": ["挥发分在岩浆洋阶段蒸发亏损", "月壳的年代学"], "keywords": ["zinc", "moon"]})
    data = facet_store.describe(papers_for(pids), ask=asker, embed=embed, quick_line=str, pids=pids, vectors=vectors)
    return group_with(data, 1), pids, vectors


def test_user_deletions_never_reach_the_prompt_and_block_lookalikes(store):
    group, pids, vectors = _described_group(store)
    rows = {c["text"]: c for c in group["claims"]}
    assert group["name"] == "MVE" and rows["挥发分在岩浆洋阶段蒸发亏损"]["status"] == "active"
    facet_store.edit_claims(group["id"], "delete", claim=rows["挥发分在岩浆洋阶段蒸发亏损"]["id"])
    # 这组新进了 6 篇；AI 提出的新论点里有一条和删掉的那条讲的是同一件事
    more_pids, more = library(blob(0, 20, 1), blob(1, 20, 2), blob(0, 6, 7))
    facet_store.sync_groups(more_pids, more)
    asker = Asker({"claims": ["锌同位素记录蒸发", "行星形成的数值模拟"], "keywords": ["moon", "impact"]}, about="- paper 41\n")
    data = facet_store.describe(papers_for(more_pids), ask=asker, embed=embed, quick_line=str, pids=more_pids, vectors=more)
    assert "挥发分在岩浆洋阶段蒸发亏损" not in asker.prompts[0]          # 删掉的内容不进提示词
    assert "月壳的年代学" in asker.prompts[0] and "paper 41" in asker.prompts[0] and "paper 1\n" not in asker.prompts[0]
    group = group_with(data, 1)
    texts = {c["text"]: c["status"] for c in group["claims"]}
    assert "锌同位素记录蒸发" not in texts                            # 与回收站里的那条太像，程序挡掉
    assert texts["行星形成的数值模拟"] == "suggested"                  # 新内容只是建议
    assert texts["impact"] == "suggested" and list(texts).count("moon") == 1


def test_deleted_line_resurfaces_with_new_evidence_and_backs_off(store):
    group, pids, vectors = _described_group(store)
    claim = next(c for c in group["claims"] if c["text"] == "挥发分在岩浆洋阶段蒸发亏损")
    facet_store.edit_claims(group["id"], "delete", claim=claim["id"])
    more_pids, more = library(blob(0, 20, 1), blob(1, 20, 2), blob(0, 10, 7))
    facet_store.sync_groups(more_pids, more)
    run = lambda: facet_store.describe(papers_for(more_pids), ask=Asker(), embed=embed, quick_line=str,
                                       pids=more_pids, vectors=more)
    store["now"] += dt.timedelta(days=30)
    row = next(c for c in group_with(run(), 1)["claims"] if c["id"] == claim["id"])
    assert row["status"] == "deleted"                                   # 等待期没到
    store["now"] += dt.timedelta(days=31)
    row = next(c for c in group_with(run(), 1)["claims"] if c["id"] == claim["id"])
    assert row["status"] == "suggested" and row["resurfaced"]["support_count"] >= 3
    facet_store.edit_claims(group["id"], "ignore", claim=claim["id"])   # 再次忽略：等待时间翻倍
    row = next(c for c in group_with(facet_store.load(), 1)["claims"] if c["id"] == claim["id"])
    assert row["status"] == "deleted" and row["wait_days"] == 120
    facet_store.edit_claims(group["id"], "block", claim=claim["id"])
    store["now"] += dt.timedelta(days=400)
    row = next(c for c in group_with(run(), 1)["claims"] if c["id"] == claim["id"])
    assert row["status"] == "blocked"                                   # 不再提示：永久


def test_line_editing_split_insert_restore(store):
    group, _, _ = _described_group(store)
    first, second = [c for c in group["claims"] if c["kind"] == "claim"]
    # 在第一行中间回车：断成两行，后半行是你写的
    out = facet_store.edit_claims(group["id"], "split", claim=first["id"], before="挥发分在岩浆洋阶段", after="蒸发亏损")
    claims = [c for c in out["group"]["claims"] if c["kind"] == "claim"]
    assert [c["text"] for c in claims] == ["挥发分在岩浆洋阶段", "蒸发亏损", "月壳的年代学"]
    assert claims[0]["edited"] and claims[1]["by"] == "user" and claims[1]["id"] == out["claim"]
    # 你写的行删掉就没了；AI 的行进回收站，可恢复
    facet_store.edit_claims(group["id"], "delete", claim=claims[1]["id"])
    facet_store.edit_claims(group["id"], "delete", claim=second["id"])
    claims = facet_store.edit_claims(group["id"], "restore", claim=second["id"])["group"]["claims"]
    assert "蒸发亏损" not in [c["text"] for c in claims]
    assert next(c for c in claims if c["id"] == second["id"])["status"] == "active"
    # 插到最前面
    out = facet_store.edit_claims(group["id"], "insert", after=None, text="只看实验约束")
    assert out["group"]["claims"][0]["text"] == "只看实验约束"
    facets = {f.id: f for f in facet_store.facets_for_ranking(facet_store.load())}
    facet = facets[group["id"]]
    assert "只看实验约束" in facet.user_note and "挥发分在岩浆洋阶段" in facet.user_note   # 你写的 / 改过的
    assert "月壳的年代学" in facet.description and "关键词：zinc, moon" in facet.description
    text = feed_ranker.render_profile(feed_ranker.LibraryModel(pids=[], vectors=np.zeros((0, 1)), recency=np.zeros(0), meta={},
                                                               facets=[facet], author_counts={}, in_domain_scores=np.zeros(1)), {})
    assert "用户补充（优先）：只看实验约束" in text


def test_merge_and_split_suggestions(store):
    # 两组其实是同一主题（被人为切开）→ 建议合并；一组里明显有两块 → 建议拆分
    pids, vectors = library(blob(0, 30, 1), blob(1, 15, 2), blob(4, 15, 3))
    data = facet_store.load()
    data["groups"] = [facet_store._new_group(data, list(range(1, 16))), facet_store._new_group(data, list(range(16, 31))),
                      facet_store._new_group(data, list(range(31, 61)))]
    facet_store.save(data)
    data = facet_store.sync_groups(pids, vectors)
    kinds = {s["type"]: s for s in data["suggestions"]}
    assert set(kinds) == {"merge", "split"}
    facet_store.apply_suggestion(kinds["split"]["key"])
    data = facet_store.apply_suggestion(kinds["merge"]["key"])
    sizes = sorted(len(g["pids"]) for g in data["groups"])
    assert sizes == [15, 15, 30]
