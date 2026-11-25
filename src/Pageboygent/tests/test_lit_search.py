from __future__ import annotations

import json
from types import SimpleNamespace

from Pageboygent import lit_search
from Pageboygent.lit_search import LitSearchAgent, LitSearchToolbox, build_paper_meta, validate_finish


SUMMARY_1 = """**Quick Summary** Zinc isotopes in lunar basalts record volatile loss.

## Core Value
Heavy δ66Zn in lunar mare basalts indicates evaporative loss of zinc during the magma ocean stage.

## Methodology
Natural sample analysis with MC-ICP-MS."""


class FakeBackend(lit_search.SearchBackend):
    papers = {
        1: build_paper_meta(paper_id=1, title="Zinc isotopes and lunar volatile loss", year=2012,
                            first_author="Paniello", doi="10.1/zn", zotero_key="ABCD1234",
                            pdf_key="PDFK0001", md_path="/x.md", has_summary=True),
        2: build_paper_meta(paper_id=2, title="Potassium isotopes of the Moon", year=2016,
                            first_author="Wang", doi=None, zotero_key="EFGH5678"),
    }

    def summary_bm25(self, query, limit):
        return [(1, 5.2, SUMMARY_1), (999, 3.0, "stale paper not in db")]

    def summary_keywords(self, expression, limit):
        return [(1, None, SUMMARY_1)]

    def fulltext_bm25(self, query, limit):
        return [
            {"paper_id": 1, "line_number": 42, "text": "The δ66Zn values of lunar basalts are ~1.4‰ heavier than Earth.", "score": 9.1},
            {"paper_id": 2, "line_number": 10, "text": "K isotopes of lunar samples are heavier.", "score": 4.0},
        ]

    def fulltext_semantic(self, query, limit):
        # 与 BM25 命中同一切块：应复用同一证据编号
        return [{"paper_id": 1, "line_number": 42, "text": "The δ66Zn values of lunar basalts are ~1.4‰ heavier than Earth.", "score": 0.8}]

    def fulltext_keywords(self, expression, limit):
        return []

    def paper_meta(self, paper_id):
        return self.papers.get(paper_id)

    def paper_notes(self, paper_id):
        return [{"key": "qs", "kind": "summary", "label": "AI 快速总结", "content": SUMMARY_1}] if paper_id == 1 else []

    def md_lines(self, meta):
        return ["# Title", "", "Abstract: zinc evaporation on the Moon is discussed in detail here.", "# References", "zinc ref"]


def _toolbox() -> LitSearchToolbox:
    return LitSearchToolbox(backend=FakeBackend())


def test_evidence_registry_dedupes_and_skips_unknown_papers():
    tb = _toolbox()
    out = tb.search_summaries("lunar zinc evaporation")
    assert "P999" not in out and "[P1]" in out
    tb.search_fulltext_bm25("zinc isotopes moon")
    tb.search_fulltext_semantic("isotopic evidence for volatile loss on the Moon")
    chunk_evidence = [e for e in tb.evidence.values() if e.kind == "fulltext" and e.line_number == 42]
    assert len(chunk_evidence) == 1
    assert set(chunk_evidence[0].methods) == {"fulltext-bm25", "fulltext-semantic"}
    assert tb.seen_papers == [1, 2]


def test_read_passages_stops_at_references():
    tb = _toolbox()
    out = tb.read_paper_passages(1, keywords=["zinc"])
    assert "L3" in out and "zinc ref" not in out


def test_validate_finish_drops_fabricated_papers_and_evidence():
    tb = _toolbox()
    tb.search_fulltext_bm25("zinc")
    e_zn = next(e.eid for e in tb.evidence.values() if e.paper_id == 1)
    e_k = next(e.eid for e in tb.evidence.values() if e.paper_id == 2)
    result = validate_finish({
        "overall_judgement": f"Zn is heavier [{e_zn}] and see [E77] and [P1], also [P404].",
        "confidence": "medium",
        "candidates": [
            {"paper_id": 1, "relevance": "high", "reason": f"direct [{e_zn}]", "evidence_ids": [e_zn, e_k, "E99"]},
            {"paper_id": 404, "relevance": "high", "reason": "made up", "evidence_ids": [e_zn]},
            {"paper_id": 2, "relevance": "medium", "reason": "no cites", "evidence_ids": []},
        ],
        "gaps": "无",
        "search_summary": "bm25",
    }, tb)
    assert [c["paper"]["paper_id"] for c in result["candidates"]] == [1, 2]
    assert result["candidates"][0]["evidence_ids"] == [e_zn]  # 不属于该文献/不存在的证据被剔除
    assert result["candidates"][1]["auto_evidence"] and result["candidates"][1]["relevance"] == "low"
    assert "[E77]" not in result["overall_judgement"] and "[?]" in result["overall_judgement"]
    assert "[P404]" not in result["overall_judgement"] and "[P1]" in result["overall_judgement"]
    assert any("P404" in w for w in result["warnings"])
    # 书目信息来自数据库，而非模型
    assert result["candidates"][0]["paper"]["zotero_select_url"] == "zotero://select/library/items/ABCD1234"


def _tool_call(call_id, name, args):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


class ScriptedClient:
    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        content, calls = self.script.pop(0)
        message = SimpleNamespace(content=content, tool_calls=calls)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_agent_self_check_forces_more_search_then_finishes():
    early_finish = {"overall_judgement": "only summary [E1]", "confidence": "low",
                    "candidates": [{"paper_id": 1, "relevance": "high", "reason": "x [E1]", "evidence_ids": ["E1"]}],
                    "gaps": "", "search_summary": ""}
    client = ScriptedClient([
        ("【评估】先查总结", [_tool_call("c1", "search_summaries", {"query": "lunar zinc"})]),
        ("", [_tool_call("c2", "finish", early_finish)]),  # 太早：自检应退回
        ("【评估】补充原文", [_tool_call("c3", "search_fulltext_bm25", {"query": "zinc isotopes moon"}),
                          _tool_call("c4", "search_fulltext_semantic", {"query": "zinc loss moon"})]),
        ("", [_tool_call("c5", "finish", {**early_finish, "overall_judgement": "heavier Zn [E2]",
                                          "candidates": [{"paper_id": 1, "relevance": "high", "reason": "[E2]",
                                                          "evidence_ids": ["E2", "E1"]}]})]),
    ])
    events = []
    agent = LitSearchAgent(client=client, model="fake", toolbox=_toolbox(), max_rounds=6,
                           on_event=lambda kind, text: events.append((kind, text)))
    result = agent.run("月球上 Zn 同位素与挥发分丢失")
    assert not client.script
    assert any(kind == "system" and "自检" in text for kind, text in events)
    # 自检提示以 finish 的 tool 结果返回
    finish_replies = [m["content"] for m in client.requests[-1]["messages"]
                      if m.get("role") == "tool" and m.get("tool_call_id") == "c2"]
    assert lit_search.SELF_CHECK_HEADER in finish_replies[0]
    cand = result["candidates"][0]
    assert cand["paper"]["paper_id"] == 1
    assert cand["evidence_ids"][0] == "E2"  # 原文证据排在前
    assert result["stats"]["tool_calls"] == 3


def test_agent_forced_finish_keeps_tool_list_constant_for_prefix_cache():
    search = ("", [_tool_call("c1", "search_fulltext_bm25", {"query": "zinc"})])
    client = ScriptedClient([search, ("still thinking, no tool", None)])
    agent = LitSearchAgent(client=client, model="fake", toolbox=_toolbox(), max_rounds=2)
    result = agent.run("zinc")
    # 工具列表不变（换工具列表会让前缀缓存失效），靠提示要求提交
    assert client.requests[0]["tools"] == client.requests[-1]["tools"]
    assert client.requests[-1]["messages"][-2]["content"] == lit_search.FORCE_FINISH_PROMPT
    assert result["warnings"][0].startswith("模型未调用 finish")
    assert {c["paper"]["paper_id"] for c in result["candidates"]} == {1, 2}


def test_session_store_runs_turn_and_builds_history(tmp_path):
    class FakeAgent:
        def __init__(self, model, on_event):
            self.on_event = on_event

        def run(self, question, history=""):
            self.on_event("tool", "search")
            return {"overall_judgement": f"answer [E1] ({history[:5]})", "candidates": [
                {"paper": FakeBackend.papers[1], "relevance": "high"}]}

    store = lit_search.LitSearchSessionStore(directory=tmp_path, agent_factory=FakeAgent)
    sid = store.new_session_id()
    store.start_turn(sid, "first question")
    for _ in range(100):
        if not store.get(sid)["is_running"]:
            break
        import time
        time.sleep(0.02)
    session = store.get(sid)
    assert session["turns"][0]["status"] == "done"
    assert session["turns"][0]["trace"][0]["text"] == "search"
    digest = lit_search._history_digest(session["turns"])
    assert "[E1]" not in digest and "P1 Paniello (2012)" in digest
    assert store.list_sessions()[0]["id"] == sid


def test_agent_accumulates_token_usage_including_cache_hits():
    class UsageClient(ScriptedClient):
        def _create(self, **kwargs):
            response = super()._create(**kwargs)
            response.usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=50, total_tokens=1050,
                                             prompt_cache_hit_tokens=600)
            return response

    finish = {"overall_judgement": "x", "confidence": "low", "candidates": [], "gaps": "", "search_summary": ""}
    client = UsageClient([
        ("", [_tool_call("c1", "search_fulltext_bm25", {"query": "zinc"})]),
        ("", [_tool_call("c2", "finish", finish)]),
    ])
    events = []
    agent = LitSearchAgent(client=client, model="fake", toolbox=_toolbox(), max_rounds=2,
                           on_event=lambda kind, text: events.append((kind, text)))
    result = agent.run("zinc")
    usage = result["stats"]["usage"]
    # 第 2 次调用的前缀应是第 1 次的全部输入（按 64 token 取整 = 960），实际只命中 600 → 本可命中却丢失 360
    assert usage == {"llm_calls": 2, "prompt_tokens": 2000, "completion_tokens": 100,
                     "total_tokens": 2100, "cache_hit_tokens": 1200, "lost_cache_tokens": 360}
    assert [c["lost"] for c in result["stats"]["calls"]] == [0, 360]
    assert json.loads([t for k, t in events if k == "usage"][-1])["total_tokens"] == 2100


# --------------------------- 二级文献 ---------------------------

REF_MD = [
    "# Zinc and the Moon",
    "Zinc in lunar rocks is isotopically heavy (Moynier et al., 2006), as also shown by ref. 2.",
    "## References",
    "Herzog, G.F., Moynier, F., Albarède, F., 2009. Isotopic and elemental abundances of copper and zinc. GCA 73, 5884–5904.",
    "Moynier, F., Albarède, F., Herzog, G.F., 2006. Isotopic composition of zinc, copper, and iron in lunar samples. "
    "Geochim. Cosmochim. Acta 70, 6103–6117. https://doi.org/10.1016/ j.gca.2006.02.030 White, W.M., Hofmann, A.W., "
    "1982. Sr and Nd isotope geochemistry of oceanic basalts and mantle evolution. Nature 296, 821–825.",
    "Paniello, R.C., Day, J.M.D., Moynier, F., 2012. Zinc isotopic evidence for the origin of the Moon. Nature 490, 376.",
    "## Acknowledgments",
    "Wang, X., 2020. Not a reference.",
]


class RefBackend(FakeBackend):
    def md_lines(self, meta):
        return REF_MD

    def find_in_library(self, doi, raw_reference):
        return 1 if "Paniello" in raw_reference else None


def test_parse_reference_entries_splits_merged_lines_and_stops_at_next_section():
    entries = lit_search.parse_reference_entries(REF_MD)
    texts = [t for _, t in entries]
    assert len(entries) == 4  # 合并行被拆开，Acknowledgments 之后不算
    assert texts[1].startswith("Moynier, F.") and texts[2].startswith("White, W.M.")
    numbered = lit_search.parse_reference_entries(["intro", "1. Smith, A. Title one. Nature 1 (2001).",
                                                   "2. Doe, B. Title two (2002).", "3. Roe, C. T3 (2003).",
                                                   "4. Poe, D. T4 (2004).", "5. Moe, E. T5 (2005)."])
    assert [n for n, _ in numbered] == [1, 2, 3, 4, 5]
    assert [numbered[i][0] for i in lit_search.match_reference_entries(numbered, citation="refs 2-4")] == [2, 3, 4]


def test_match_reference_requires_first_author_and_year():
    entries = lit_search.parse_reference_entries(REF_MD)
    hits = lit_search.match_reference_entries(entries, citation="Hofmann and White, 1982")
    assert hits == []  # 第一作者是 White，不是 Hofmann
    hits = lit_search.match_reference_entries(entries, citation="Moynier et al., 2006")
    assert [entries[i][1][:7] for i in hits] == ["Moynier"]
    assert lit_search.extract_doi(entries[hits[0]][1]) == "10.1016/j.gca.2006.02.030"


def test_secondary_reference_fields_must_come_from_reference_text():
    tb = LitSearchToolbox(backend=RefBackend())
    tb.read_paper_passages(1, keywords=["Moynier"])
    context = next(e.eid for e in tb.evidence.values() if e.kind == "fulltext")
    out = tb.lookup_cited_reference(1, citation="Moynier et al., 2006")
    assert "R1" in out
    out_lib = tb.lookup_cited_reference(1, citation="Paniello et al., 2012")
    assert "已在库中" in out_lib
    result = validate_finish({
        "overall_judgement": "见 [R1]，以及编造的 [R9]", "confidence": "medium", "candidates": [],
        "secondary_references": [
            {"ref_id": "R1", "context_evidence_ids": [context, "E99"], "relevance": "high",
             "reason": f"据 [P1] 所述 [{context}]",
             "authors": "Moynier, F., Albarède, F., Herzog, G.F.",
             "title": "Isotopic composition of zinc, copper, and iron in lunar samples",
             "journal": "Earth Planet. Sci. Lett.",  # 原文是 GCA：应剔除
             "doi": "10.1016/j.gca.2006.99.999"},      # 编造的 DOI：应以原文提取的为准
            {"ref_id": "E1", "context_evidence_ids": [], "relevance": "high", "reason": "x"},
            {"ref_id": "R2", "context_evidence_ids": [], "relevance": "low", "reason": "in library"},
        ],
        "gaps": "", "search_summary": "",
    }, tb)
    refs = result["secondary_references"]
    assert len(refs) == 2
    first = refs[0]
    assert first["fields"]["doi"] == "10.1016/j.gca.2006.02.030"
    assert "journal" not in first["fields"] and first["fields"]["year"] == "2006"
    assert first["fields"]["title"].startswith("Isotopic composition")
    assert first["context_evidence_ids"] == [context]
    assert first["cited_by"][0]["paper_id"] == 1
    assert refs[1]["in_library"]["paper_id"] == 1  # 已在库中的被标注并排到后面
    assert "[R1]" in result["overall_judgement"] and "[R9]" not in result["overall_judgement"]
    joined = " ".join(result["warnings"])
    assert "journal" in joined and "E1 不是" in joined and "DOI 与参考文献原文不符" in joined


# --------------------------- token 费用预估 ---------------------------

def test_usage_breakdown_uses_peak_rates_and_skips_unpriced_models(monkeypatch):
    prices = {"price_cheap_input_cache_hit": 0.04, "price_cheap_input_cache_miss": 2.0, "price_cheap_output": 8.0,
              "price_professional_input_cache_hit": None, "price_professional_input_cache_miss": None,
              "price_professional_output": None, "model_name_professional": "pro-model"}
    real_get = lit_search.get_config
    monkeypatch.setattr(lit_search, "get_config", lambda key, *a: prices[key] if key in prices else real_get(key, *a))

    usage = {"llm_calls": 10, "prompt_tokens": 208450, "completion_tokens": 5600, "cache_hit_tokens": 150016}
    flash = lit_search.usage_breakdown(usage, "flash-model")
    assert flash["tokens"] == {"input_cache_hit": 150016, "input_cache_miss": 58434, "output": 5600}
    # 0.150016*0.04 + 0.058434*2 + 0.0056*8
    assert abs(flash["cost"]["total"] - (0.00600064 + 0.116868 + 0.0448)) < 1e-9

    pro = lit_search.usage_breakdown(usage, "pro-model")
    assert pro["cost"] is None and pro["rates"] is None
    session = lit_search.merge_breakdowns([flash, pro, lit_search.usage_breakdown({}, "flash-model")])
    assert session["total_tokens"] == 2 * 214050
    assert session["unpriced_turns"] == 1
    assert abs(session["cost"]["total"] - flash["cost"]["total"]) < 1e-12


# --------------------------- 回归：空结果 / 引号 / 跨文献引用语境 ---------------------------

def test_parse_tool_arguments_handles_sloppy_json_and_never_returns_silent_empty():
    parse = lit_search.parse_tool_arguments
    assert parse('{"overall_judgement": "line1\nline2"}', "finish")[0]["overall_judgement"] == "line1\nline2"
    assert parse(json.dumps(json.dumps({"overall_judgement": "x"})), "finish")[0] == {"overall_judgement": "x"}
    assert parse('{"finish": {"overall_judgement": "x"}}', "finish")[0] == {"overall_judgement": "x"}
    assert parse('```json\n{"query": "zinc"}\n```', "search")[0] == {"query": "zinc"}
    args, err = parse('{"overall_judgement": "unterminated', "finish")
    assert args is None and "JSON" in err
    args, err = parse('["not", "a", "dict"]', "finish")
    assert args is None and "list" in err


def test_empty_finish_is_rejected_and_model_resubmits():
    good = {"short_answer": "有，见 [P1]。", "overall_judgement": "found [E1]", "confidence": "high", "gaps": "无",
            "search_summary": "s",
            "candidates": [{"paper_id": 1, "relevance": "high", "reason": "[E1]", "evidence_ids": ["E1"]}]}
    client = ScriptedClient([
        ("", [_tool_call("c1", "search_fulltext_bm25", {"query": "zinc"}),
              _tool_call("c2", "search_summaries", {"query": "zinc"}),
              _tool_call("c3", "search_fulltext_semantic", {"query": "zinc"})]),
        ("", [SimpleNamespace(id="c4", function=SimpleNamespace(name="finish", arguments="{}"))]),
        ("", [_tool_call("c5", "finish", good)]),
    ])
    events = []
    agent = LitSearchAgent(client=client, model="fake", toolbox=_toolbox(), max_rounds=6,
                           on_event=lambda kind, text: events.append((kind, text)))
    result = agent.run("zinc")
    assert [c["paper"]["paper_id"] for c in result["candidates"]] == [1]
    assert any(kind == "warn" and "finish 参数无效" in text for kind, text in events)
    assert result["stats"]["anomalies"][0]["raw_arguments"] == "{}"


def test_reference_matching_folds_curly_quotes_and_accents():
    entries = [(None, "O’Brien, D. P., Morbidelli, A., & Levison, H. F. 2006, Icarus, 184, 39"),
               (None, "Moynier F., Albarède F. and Herzog G. F. (2006a) Isotopic composition of zinc.")]
    assert lit_search.match_reference_entries(entries, citation="O'Brien et al., 2006") == [0]
    assert lit_search.match_reference_entries(entries, keywords=["O'Brien"]) == [0]
    assert lit_search.match_reference_entries(entries, citation="Moynier, Albarede & Herzog 2006a") == [1]


def test_secondary_context_from_another_citing_paper_requires_author_and_year():
    tb = LitSearchToolbox(backend=RefBackend())
    tb.lookup_cited_reference(1, citation="Moynier et al., 2006")  # R1 来自 P1
    ok = tb.add_evidence(("x", 1), paper_id=2, kind="fulltext", label="原文 L5", method="t",
                         text="As shown by Moynier et al. (2006), lunar Zn is heavy.")
    bad = tb.add_evidence(("x", 2), paper_id=2, kind="fulltext", label="原文 L9", method="t",
                          text="Potassium isotopes are heavy too.")
    tb.paper(2)
    result = validate_finish({"overall_judgement": "x", "confidence": "low", "candidates": [], "gaps": "",
                              "search_summary": "", "secondary_references": [
                                  {"ref_id": "R1", "context_evidence_ids": [ok.eid, bad.eid],
                                   "relevance": "high", "reason": "r"}]}, tb)
    ref = result["secondary_references"][0]
    assert ref["context_evidence_ids"] == [ok.eid]
    assert [c["paper_id"] for c in ref["cited_by"]] == [1, 2]
    assert any(bad.eid in w for w in result["warnings"])


def test_find_in_library_falls_back_to_unique_first_author_and_year():
    backend = lit_search.SearchBackend()
    backend._library_index = ({}, {}, {("chambers", "2001"): [159], ("clement", "2019"): [185, 187, 164], ("obrien", "2006"): [153]})
    assert backend.find_in_library(None, "O’Brien, D. P., Morbidelli, A., & Levison, H. F. 2006, Icarus, 184, 39") == 153
    assert backend.find_in_library(None, "Chambers, J. E. 2001, Icar, 152, 205") == 159
    assert backend.find_in_library(None, "Clement, M. S., et al. 2019, Icar, 321, 778") is None  # 不唯一，不下结论
    assert backend.find_in_library(None, "Chambers, J. E. 2004, EPSL, 223, 241") is None



# --------------------------- 简短回答 ---------------------------

def test_short_answer_required_by_self_check_and_derived_when_missing():
    tb = _toolbox()
    for q in ("a", "b", "c"):
        tb.dispatch("search_fulltext_bm25", {"query": q})
    tb.dispatch("search_summaries", {"query": "zinc"})
    base = {"overall_judgement": "## 结论：库中有一篇直接相关的文献\n\n细节 [E1]", "confidence": "high", "candidates": [],
            "gaps": "", "search_summary": ""}
    assert any("short_answer" in issue for issue in lit_search.self_check(base, tb))
    assert any("太长" in issue for issue in lit_search.self_check({**base, "short_answer": "长" * 400}, tb))
    assert not any("short_answer" in issue for issue in lit_search.self_check({**base, "short_answer": "有，见 [P1]。"}, tb))

    result = validate_finish(base, tb)  # 兜底：取详细报告里的结论标题
    assert result["short_answer"] == "结论：库中有一篇直接相关的文献"
    assert any("简短回答" in w for w in result["warnings"])
    result = validate_finish({**base, "short_answer": "有 [P1]，见 [E1] 与编造的 [E404]"}, tb)
    assert "[E404]" not in result["short_answer"] and "E1" in result["evidence"]


def test_render_drops_author_year_repeated_after_paper_cite():
    from Pageboygent import ui

    result = {"paper_cites": {"184": "Hansen (2009)", "203": "Chambers (1998)"}, "evidence": {}}
    html = ui._lit_render_with_citations("是 [P184] Hansen (2009)（窄环），前身 [P203] Chambers & Wetherill (1998)。",
                                          result, "t")
    assert html.count("Hansen (2009)") == 1 and html.count("1998") == 1
    assert "（窄环）" in html



# --------------------------- 同一对话：完整上下文 + 共用证据编号 ---------------------------

def _finish_call(call_id, eid, short="有，见 [P1]。"):
    return _tool_call(call_id, "finish", {
        "short_answer": short, "overall_judgement": f"详见 [{eid}]", "confidence": "high", "gaps": "无",
        "search_summary": "s", "candidates": [{"paper_id": 1, "relevance": "high", "reason": f"[{eid}]",
                                               "evidence_ids": [eid]}]})


def _first_turn_agent(client):
    return LitSearchAgent(client=client, model="fake", toolbox=_toolbox(), max_rounds=6)


def test_follow_up_reuses_transcript_prefix_and_evidence_ids():
    client = ScriptedClient([
        ("", [_tool_call("a2", "search_fulltext_bm25", {"query": "zinc isotopes"}),
              _tool_call("a1", "search_summaries", {"query": "zinc"}),
              _tool_call("a3", "search_fulltext_semantic", {"query": "zinc loss"})]),
        ("", [_finish_call("a4", "E1")]),
    ])
    agent = _first_turn_agent(client)
    agent.run("月球 Zn 同位素")
    state = json.loads(json.dumps(agent.export_state()))  # 模拟落盘再读回
    first_turn_messages = state["messages"]
    e1_text = agent.toolbox.evidence["E1"].text

    client2 = ScriptedClient([
        # 追问：直接引用上一轮的 E2；重复上一轮的检索不会被再次执行
        ("", [_tool_call("b1", "search_fulltext_bm25", {"query": "zinc isotopes"}),
              _tool_call("b2", "recall_evidence", {"ids": ["E1"]})]),
        ("", [_finish_call("b3", "E1", short="上一轮的 [P1] 就是答案。")]),
    ])
    agent2 = LitSearchAgent(client=client2, model="fake", toolbox=_toolbox(), max_rounds=6)
    agent2.load_state(state)
    result = agent2.run("它的结论是什么？")

    sent = client2.requests[0]["messages"]
    assert sent[:len(first_turn_messages)] == first_turn_messages  # 前缀逐字一致 → 可命中缓存
    assert sent[len(first_turn_messages)]["content"].startswith("用户追问：")
    tool_replies = [m["content"] for m in client2.requests[1]["messages"] if m.get("tool_call_id") in {"b1", "b2"}]
    assert "已执行过" in tool_replies[0] and e1_text[:20] in tool_replies[1]
    # 追问不强求重新做足检索；E2 编号跨轮有效
    assert result["candidates"][0]["evidence_ids"] == ["E1"]
    assert result["evidence"]["E1"]["text"] == e1_text
    assert len(agent2.turn_summaries) == 2


def test_compaction_when_context_exceeds_budget_keeps_evidence_ids():
    client = ScriptedClient([
        ("", [_tool_call("a2", "search_fulltext_bm25", {"query": "zinc isotopes"}),
              _tool_call("a1", "search_summaries", {"query": "zinc"}),
              _tool_call("a3", "search_fulltext_semantic", {"query": "zinc loss"})]),
        ("", [_finish_call("a4", "E1")]),
    ])
    agent = _first_turn_agent(client)
    agent.run("月球 Zn 同位素")
    state = agent.export_state()
    state["last_prompt_tokens"] = 10 ** 9  # 超出预算

    client2 = ScriptedClient([
        ("", [_tool_call("b1", "search_fulltext_bm25", {"query": "zinc isotopes"})]),
        ("", [_finish_call("b2", "E1")]),
    ])
    events = []
    agent2 = LitSearchAgent(client=client2, model="fake", toolbox=_toolbox(), max_rounds=6,
                            on_event=lambda k, t: events.append((k, t)))
    agent2.load_state(state)
    agent2.run("再查一下")
    sent = client2.requests[0]["messages"]
    # 压缩后：system + 上一轮（问题, 结论摘要）+ 本轮追问；上一轮的工具往来全部丢弃
    assert [m["role"] for m in sent[:4]] == ["system", "user", "assistant", "user"]
    assert sent[3]["content"].startswith("用户追问：") and not any(m.get("tool_call_id", "").startswith("a") for m in sent)
    assert "E1" in sent[2]["content"] and "recall_evidence" in sent[2]["content"]
    assert any("已把之前各轮压缩" in t for _, t in events)
    # 压缩后旧的工具结果不在上下文里，同一检索允许重新执行
    reply = next(m["content"] for m in client2.requests[1]["messages"] if m.get("tool_call_id") == "b1")
    assert "已执行过" not in reply and "E1" in reply


def test_session_store_persists_agent_state_between_turns(tmp_path):
    scripts = [
        [("", [_tool_call("a2", "search_fulltext_bm25", {"query": "zinc isotopes"}),
               _tool_call("a1", "search_summaries", {"query": "zinc"}),
               _tool_call("a3", "search_fulltext_semantic", {"query": "zinc loss"})]),
         ("", [_finish_call("a4", "E1")])],
        [("", [_finish_call("b1", "E1", short="同上 [P1]。")])],
    ]
    clients = []

    def factory(model, on_event):
        client = ScriptedClient(scripts[len(clients)])
        clients.append(client)
        return LitSearchAgent(client=client, model=model, toolbox=_toolbox(), max_rounds=6, on_event=on_event)

    store = lit_search.LitSearchSessionStore(directory=tmp_path, agent_factory=factory)
    sid = store.new_session_id()
    import time
    for question in ("第一问", "追问"):
        store.start_turn(sid, question)
        for _ in range(200):
            if not store.get(sid)["is_running"]:
                break
            time.sleep(0.01)
    session = store.get(sid)
    assert [t["status"] for t in session["turns"]] == ["done", "done"]
    assert session["turns"][1]["result"]["candidates"][0]["evidence_ids"] == ["E1"]
    assert (tmp_path / f"{sid}.agent.json").exists()
    assert [s["id"] for s in store.list_sessions()] == [sid]  # agent 状态文件不会被当成会话
    assert clients[1].requests[0]["messages"][1]["content"] == "用户的检索需求：第一问"


# --------------------------- 上下文控制：折叠 / 硬上限 / 精读子 agent ---------------------------

def test_compress_ids_and_quote_location():
    assert lit_search._compress_ids(["E101", "E102", "E103", "E107", "R2"]) == "E101-E103, E107, R2"
    lines = ["# Title", "The δ66Zn values of *lunar* basalts are ~1.4‰ heavier than Earth’s mantle.", "Next."]
    assert lit_search.locate_quote(lines, "The δ66Zn values of lunar basalts are ~1.4‰ heavier than Earth's")[0] == 2
    assert lit_search.locate_quote(lines, "lunar basalts are much lighter than terrestrial ones overall") is None
    text, truncated = lit_search.prepare_fulltext(["a"] * 10 + ["## References", "Ref 1."], 10_000)
    assert "Ref 1." not in text and not truncated


def test_deep_read_registers_only_verified_quotes_and_keeps_main_context_small():
    tb = LitSearchToolbox(backend=RefBackend())
    real = "Zinc in lunar rocks is isotopically heavy (Moynier et al., 2006), as also shown by ref. 2."
    tb.deep_reader = lambda fulltext, question: {
        "answer": "月球岩石 Zn 同位素偏重。", "found": True,
        "quotes": [{"quote": real, "note": "直接陈述"},
                   {"quote": "Zinc was found to be isotopically light in all lunar samples measured.", "note": "编造"}],
    }
    out = tb.deep_read_paper(1, "月球 Zn 同位素是轻还是重？")
    assert "E1 (原文 L2)" in out and "1 条引句在原文中找不到" in out
    assert tb.evidence["E1"].kind == "fulltext" and "deep-read" in tb.evidence["E1"].methods
    assert len(out) < 2000  # 主上下文只拿到结论与引句，不含全文
    for _ in range(2):
        tb.deep_read_paper(1, f"另一个问题 {_}")
    assert "上限" in tb.deep_read_paper(1, "第四次")


def test_turn_start_folds_old_tool_results_before_summarizing():
    client = ScriptedClient([
        ("", [_tool_call("a2", "search_fulltext_bm25", {"query": "zinc isotopes"}),
              _tool_call("a1", "search_summaries", {"query": "zinc"}),
              _tool_call("a3", "search_fulltext_semantic", {"query": "zinc loss"})]),
        ("", [_finish_call("a4", "E1")]),
    ])
    agent = _first_turn_agent(client)
    agent.run("月球 Zn 同位素")
    state = json.loads(json.dumps(agent.export_state()))
    state["last_prompt_tokens"] = 250_000  # 超过软阈值 200k，但折叠后能降下来
    state["chars_per_token"] = 0.001        # 伪造的检索结果很短：放大"折叠掉的字符"折算出的 tokens

    client2 = ScriptedClient([("", [_finish_call("b1", "E1", short="见上轮 [P1]。")])])
    events = []
    agent2 = LitSearchAgent(client=client2, model="fake", toolbox=_toolbox(), max_rounds=6,
                            on_event=lambda k, t: events.append((k, t)))
    agent2.load_state(state)
    agent2.run("追问")
    sent = client2.requests[0]["messages"]
    tool_msgs = [m for m in sent if m.get("role") == "tool" and m["tool_call_id"].startswith("a")]
    assert tool_msgs and all(m["content"].startswith("[已折叠") for m in tool_msgs if m["tool_call_id"] != "a4")
    assert "E1" in next(m["content"] for m in tool_msgs if m["tool_call_id"] == "a2")
    assert any(m.get("role") == "assistant" and m.get("tool_calls") for m in sent)  # 推理与调用记录保留，没有压成摘要
    assert not any("压缩为结论摘要" in t for _, t in events)


def test_hard_limit_folds_current_turn_tool_results_mid_turn():
    class UsageClient(ScriptedClient):
        def _create(self, **kwargs):
            response = super()._create(**kwargs)
            response.usage = SimpleNamespace(prompt_tokens=460_000, completion_tokens=10, total_tokens=460_010,
                                             prompt_cache_hit_tokens=0)
            return response

    client = UsageClient([
        ("", [_tool_call("a1", "search_fulltext_bm25", {"query": "zinc"})]),
        ("", [_tool_call("a2", "search_summaries", {"query": "zinc"}),
              _tool_call("a4", "search_fulltext_semantic", {"query": "zinc"})]),
        ("", [_finish_call("a3", "E1")]),
    ])
    events = []
    agent = LitSearchAgent(client=client, model="fake", toolbox=_toolbox(), max_rounds=6,
                           on_event=lambda k, t: events.append((k, t)))
    agent.run("zinc")
    assert any("接近硬上限" in t for _, t in events)
    first_result = next(m for m in agent.messages if m.get("tool_call_id") == "a1")
    assert first_result["content"].startswith("[已折叠")
