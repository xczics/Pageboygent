from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from Pageboygent import llm_usage, review_pipeline
from Pageboygent.review_pipeline import ReviewBackend, ReviewProjectStore, ReviewRunner, parse_plan

PAPERS = {
    "K1": {"pid": 1, "key": "K1", "title": "Zinc isotopes on the Moon", "first_author": "Paniello", "year": 2012,
           "doi": "10.1/zn", "summary": "Zn heavy.\n## Labels\nx;y", "abstract": "", "md_path": "/k1.md"},
    "K2": {"pid": 2, "key": "K2", "title": "Potassium isotopes", "first_author": "Wang", "year": 2016,
           "doi": None, "summary": "K heavy.", "abstract": "", "md_path": "/k2.md"},
    "K3": {"pid": 3, "key": "K3", "title": "No fulltext", "first_author": "Day", "year": 2014,
           "doi": None, "summary": "x", "abstract": "", "md_path": None},
}

PLAN = """<subprojects>
<subproject><label>Lunar Zn</label><subprojectsummary>Zn & K on the Moon</subprojectsummary>
<biblist><bib><pid>P1</pid><author>Paniello</author><year>2012</year></bib>
<bib><pid>P99</pid><author>Wang</author><year>2016</year></bib>
<bib><pid>P42</pid><author>Nobody</author><year>1999</year></bib></biblist>
<readprompt>Read carefully.</readprompt><noteformat>- point</noteformat><enhancenoteprompt>Be brief.</enhancenoteprompt>
<reviewprompt>Write a review.</reviewprompt><reviewformat># Review</reviewformat><enhancereviewprompt>Cite.</enhancereviewprompt>
</subproject></subprojects>"""


class FakeBackend(ReviewBackend):
    def __init__(self):
        self.saved_notes = []

    def collection_names(self):
        return ["Moon"]

    def collection_keys(self, name):
        return ["K1", "K2", "K3", "K404"]

    def papers_by_keys(self, keys):
        return {k: dict(PAPERS[k]) for k in keys if k in PAPERS}

    def bibliography(self, key):
        return f"EPSL ref of {key}."

    def read_fulltext(self, md_path):
        return f"FULLTEXT of {md_path}"

    def save_note_to_zotero(self, key, content, tags):
        self.saved_notes.append((key, tags))


class FakeClient:
    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        messages = kwargs["messages"]
        text = messages[-1]["content"]
        if "<subprojects>" in text or "候选文献的略读笔记" in text:
            content = PLAN
        elif text.startswith("Please read the full text"):
            content = "NOTE for " + text.split("]")[0] + "]"
        elif "子话题综述" in text:
            content = "# Final review\nAs shown by Paniello et al. (2012) and Ghost et al. (2001)."
        else:
            content = "# Sub review\nSee Paniello et al. (2012)."
        usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=100, total_tokens=1100, prompt_cache_hit_tokens=400)
        message = SimpleNamespace(content=content, reasoning_content="thinking...")
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=usage)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_usage, "_db_path", tmp_path / "usage.db")
    client = FakeClient()
    backend = FakeBackend()
    st = ReviewProjectStore(directory=tmp_path / "projects", backend=backend,
                            runner_factory=lambda s, pid: ReviewRunner(s, pid, client=client))
    st.client = client
    return st


def _wait(store, project_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if store.status(project_id) != "running":
            return store.load(project_id)
        time.sleep(0.02)
    raise AssertionError("pipeline did not finish")


def test_parse_plan_validates_ids_and_fixes_by_author_year():
    papers = {1: PAPERS["K1"], 2: PAPERS["K2"]}
    topics, warnings = parse_plan("noise ```xml\n" + PLAN + "\n```", papers)
    assert topics[0]["pids"] == [1, 2]  # P99 按 Wang 2016 纠正为 P2；Nobody 1999 剔除
    assert topics[0]["dir"] == "01-Lunar-Zn" and topics[0]["noteformat"] == "- point"
    assert any("纠正为 P2" in w for w in warnings) and any("剔除" in w for w in warnings)


def test_full_pipeline_writes_organized_files_and_tracks_usage(store, tmp_path):
    pid = store.create(name="Moon volatiles", collections=["Moon"], topic="月球挥发分", use_pro=False)
    store.start(pid)
    project = _wait(store, pid)
    assert project["status"] == "done", project.get("error")
    base = tmp_path / "projects" / pid
    # 规整的目录结构
    assert (base / "plan" / "plan.xml").exists()
    assert (base / "subtopics" / "01-Lunar-Zn" / "notes" / "P1.md").exists()
    assert (base / "subtopics" / "01-Lunar-Zn" / "notes" / "P2.md").exists()
    assert (base / "subtopics" / "01-Lunar-Zn" / "review.md").exists()
    assert (base / "subtopics" / "01-Lunar-Zn" / "reasoning" / "P1.md").exists()
    assert (base / "final" / "review.md").read_text().startswith("# Final review")
    # 没有全文 / 未被追踪的文献被排除
    assert sorted(project["papers"]) == ["1", "2"]
    assert {e["reason"][:4] for e in project["excluded"]} == {"未被 P", "没有本地"}
    # 精读笔记写回 Zotero，标签带项目名，避免与旧项目同名子课题互相覆盖
    assert ("K1", ["AI Notes", "Detailed Reading/Moon-volatiles/Lunar Zn"]) in store.backend.saved_notes
    # 用量：分阶段 + 全局账本
    assert set(project["usage"]) == {"plan", "notes:Lunar Zn", "review:Lunar Zn", "final"}
    assert project["usage"]["notes:Lunar Zn"]["llm_calls"] == 2
    ledger = llm_usage.summarize(path=tmp_path / "usage.db")
    assert ledger["categories"][0]["category"] == "综述流水线" and ledger["calls"] == 5
    # 最终综述里不在文献列表中的引用 → 警告
    assert any("Ghost et al" in t["text"] for t in project["trace"] if t["kind"] == "warn")
    # 书目信息来自 Zotero 导出，而不是模型
    final_prompt = store.client.calls[-1]["messages"][-1]["content"]
    assert "[P1] EPSL ref of K1." in final_prompt


def test_resume_reuses_finished_artifacts(store, tmp_path):
    pid = store.create(name="x", collections=["Moon"], topic="t", use_pro=False)
    store.start(pid)
    _wait(store, pid)
    calls_before = len(store.client.calls)
    (tmp_path / "projects" / pid / "final" / "review.md").unlink()
    store.start(pid)
    project = _wait(store, pid)
    assert project["status"] == "done"
    assert len(store.client.calls) == calls_before + 1  # 只重跑了最终综述


def test_note_prompts_share_prefix_within_subtopic(store):
    pid = store.create(name="x", collections=["Moon"], topic="t", use_pro=True)
    store.start(pid)
    _wait(store, pid)
    note_calls = [c for c in store.client.calls if c["messages"][-1]["content"].startswith("Please read")]
    assert len(note_calls) == 2
    assert note_calls[0]["messages"][0] == note_calls[1]["messages"][0]  # 同一系统提示（含略读总结）→ 前缀缓存
    assert "Zn heavy." in note_calls[0]["messages"][0]["content"] and "Labels" not in note_calls[0]["messages"][0]["content"]


def test_read_document_is_confined_to_project_dir(store, tmp_path):
    pid = store.create(name="x", collections=["Moon"], topic="t", use_pro=False)
    store.start(pid)
    _wait(store, pid)
    assert store.read_document(pid, "final/review.md").startswith("# Final")
    assert store.read_document(pid, "../../usage.db") is None
    assert store.read_document(pid, "project.json") is None


def test_review_api_end_to_end(store, monkeypatch):
    from Pageboygent import ui

    monkeypatch.setattr(review_pipeline, "_store", store)
    client = ui.create_storage_admin_app().test_client()
    page = client.get("/admin/review/").get_data(as_text=True)
    assert 'id="review-root"' in page and "pbUsagePanel" in page
    assert client.get("/admin-api/review/collections").get_json()["collections"] == ["Moon"]
    preview = client.post("/admin-api/review/preview", json={"collections": ["Moon"]}).get_json()
    assert (preview["items"], preview["usable"], preview["no_fulltext"], preview["untracked"]) == (4, 2, 1, 1)
    assert client.post("/admin-api/review/create", json={"collections": [], "topic": ""}).status_code == 400
    state = client.post("/admin-api/review/create",
                        json={"name": "Moon", "collections": ["Moon"], "topic": "月球", "use_pro": False}).get_json()
    pid = state["project"]["id"]
    _wait(store, pid)
    state = client.get(f"/admin-api/review/state?project_id={pid}").get_json()
    project = state["project"]
    assert [s["state"] for s in project["stages"]] == ["done"] * 5
    assert project["subtopics"][0]["papers"][0]["status"] == "done"
    assert "<h1>Final review</h1>" in project["final"]["html"]
    panel = state["usage_panel"]
    assert panel["session"]["llm_calls"] == 5 and panel["session"]["turn_count"] == 4
    doc = client.get(f"/admin-api/review/doc?project_id={pid}&path=subtopics/01-Lunar-Zn/review.md").get_json()
    assert "Sub review" in doc["html"]
    download = client.get(f"/admin-api/review/doc?project_id={pid}&path=final/review.md&download=1")
    assert download.headers["Content-Disposition"].endswith("Moon-review.md")
    assert client.get(f"/admin-api/review/doc?project_id={pid}&path=../x.md").status_code == 404
