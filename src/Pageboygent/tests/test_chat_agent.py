from __future__ import annotations

import json
import time
from pathlib import Path

from Pageboygent import chat_agent
from Pageboygent.chat_agent import ChatAgent, migrate_legacy_histories, unsupported_author_year_mentions
from Pageboygent.tests.test_lit_search import FakeBackend, ScriptedClient, _tool_call
from Pageboygent.lit_search import LitSearchSessionStore, LitSearchToolbox


def _agent(script, **kwargs):
    client = ScriptedClient(script)
    return ChatAgent(client=client, model="fake", toolbox=LitSearchToolbox(backend=FakeBackend()),
                     max_rounds=4, **kwargs), client


def test_plain_reply_needs_no_tools_and_no_finish():
    agent, client = _agent([("你好！有什么可以帮你？", None)])
    result = agent.run("你好")
    assert result["answer"] == "你好！有什么可以帮你？"
    assert result["warnings"] == [] and result["cited_papers"] == []
    assert all(t["function"]["name"] != "finish" for t in client.requests[0]["tools"])
    assert client.requests[0]["messages"][1] == {"role": "user", "content": "你好"}


def test_answer_after_search_validates_citations_and_flags_memory_references():
    agent, _ = _agent([
        ("", [_tool_call("c1", "search_fulltext_bm25", {"query": "zinc moon"})]),
        ("月球玄武岩的 Zn 更重 [E1]，见 [P1]；另见编造的 [E99]。"
         "Moynier et al. (2006) 也有报道，Paniello et al. (2012) 给出了解释。", None),
    ])
    result = agent.run("月球的 Zn 同位素偏轻还是偏重？")
    assert "[E1]" in result["answer"] and "[E99]" not in result["answer"] and "[?]" in result["answer"]
    assert [p["paper_id"] for p in result["cited_papers"]] == [1]
    assert list(result["evidence"]) == ["E1"]
    joined = " ".join(result["warnings"])
    assert "Moynier et al" in joined  # 本对话检索结果里没有 → 警告
    assert "Paniello" not in joined   # P1 = Paniello (2012)，在检索结果里 → 不警告


def test_follow_up_reuses_transcript_and_evidence(tmp_path):
    agent, _ = _agent([
        ("", [_tool_call("c1", "search_fulltext_bm25", {"query": "zinc moon"})]),
        ("更重 [E1]。", None),
    ])
    agent.run("Zn 同位素？")
    state = json.loads(json.dumps(agent.export_state()))
    agent2, client2 = _agent([("如上，依据是 [E1]。", None)])
    agent2.load_state(state)
    result = agent2.run("依据是什么？")
    sent = client2.requests[0]["messages"]
    assert sent[:len(state["messages"])] == state["messages"]
    assert sent[len(state["messages"])] == {"role": "user", "content": "依据是什么？"}
    assert list(result["evidence"]) == ["E1"]


def test_unsupported_author_year_ignores_months_and_known_papers():
    tb = LitSearchToolbox(backend=FakeBackend())
    tb.search_fulltext_bm25("zinc")  # 使 P1 Paniello (2012) 成为已见文献
    text = "In January, 2020 we met. Paniello et al. (2012) and Wang & Li, 2019 disagree; see Fig, 2010."
    assert unsupported_author_year_mentions(text, tb) == ["Wang & Li, 2019"]


def test_migrate_legacy_histories_moves_files_out_of_code_dir(tmp_path):
    source = tmp_path / "repo"
    source.mkdir()
    target = tmp_path / "chat_sessions"
    target.mkdir()
    sid = "83b8f475b43a0110314d72d1664a5169"
    (source / f"history_{sid}.json").write_text(json.dumps({"context": [
        {"role": "system", "content": "old system"},
        {"role": "user", "content": "Question from the user: BSE 的 Rb 同位素？"},
        {"role": "assistant", "content": "<queries><query>x</query></queries>"},
        {"role": "user", "content": "<queried_informations>...</queried_informations>"},
        {"role": "assistant", "content": "比球粒陨石重（Wang 2020）。"},
        {"role": "user", "content": "Question from the user: 没回答的问题"},
    ]}), encoding="utf-8")
    assert migrate_legacy_histories(target, [source]) == 1
    assert not list(source.glob("history_*.json"))
    assert (target / "legacy_history" / f"history_{sid}.json").exists()
    session = json.loads((target / f"{sid}.json").read_text())
    assert [t["question"] for t in session["turns"]] == ["BSE 的 Rb 同位素？", "没回答的问题"]
    assert session["turns"][0]["result"]["answer"] == "比球粒陨石重（Wang 2020）。"
    assert session["turns"][1]["status"] == "error"
    state = json.loads((target / f"{sid}.agent.json").read_text())
    assert [m["role"] for m in state["messages"]] == ["system", "user", "assistant"]
    assert migrate_legacy_histories(target, [source]) == 0  # 幂等


def test_chat_page_and_api_use_workstation_store(tmp_path, monkeypatch):
    from Pageboygent import ui

    class FakeChat:
        def __init__(self, model, on_event):
            pass

        def run(self, question, history=""):
            return {"answer": f"答：{question}", "cited_papers": [], "evidence": {}, "paper_cites": {},
                    "warnings": [], "stats": {}}

    store = LitSearchSessionStore(directory=tmp_path / "chat_sessions", agent_factory=FakeChat)
    monkeypatch.setattr(chat_agent, "_chat_store", store)
    client = ui.create_storage_admin_app().test_client()
    page = client.get("/admin/?tab=chat").get_data(as_text=True)
    assert 'id="chat-root"' in page and "pbAgentView" in page and "history_hash" not in page
    sid = client.post("/admin-api/chat/send", json={"question": "你好"}).get_json()["session_id"]
    for _ in range(100):
        state = client.get(f"/admin-api/chat/state?session_id={sid}").get_json()
        if not state["is_running"]:
            break
        time.sleep(0.01)
    assert state["turns"][0]["result"]["answer_html"] == "<p>答：你好</p>"
    assert (tmp_path / "chat_sessions" / f"{sid}.json").exists()
    assert not list(Path.cwd().glob(f"history_{sid}.json"))
