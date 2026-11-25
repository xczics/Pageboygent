"""
AI 文献检索 Agent：通过自然语言对话，在追踪的文献库中检索文献。

与 paper_query.PaperQueryChatter（固定流水线：生成查询 -> 检索 -> 逐篇阅读 -> 作答）不同，
这里是一个工具调用循环（tool-calling loop）：LLM 自己决定用哪种检索、查什么词、何时回原文核实、
何时停止。检索手段全部复用已有代码：

    - AI 快速总结  BM25          storage.search_quick_summary_bm25
    - AI 快速总结  布尔关键词     storage.search_quick_summary_by_key_words
    - 原文切块     BM25          rag.bm25_retrieve
    - 原文切块     精确关键词     rag.keywords_retrieve
    - 原文切块     向量语义(RAG)  rag.vector_retrieve
    - 预生成笔记   主题总结/精读笔记（ThemePaper / note_storage.NoteRecord）
    - 原文 Markdown 回读（out_mds/{pdf_key}.md）

防幻觉设计（提示词约束 + 程序校验双保险）：
    1. 所有检索结果中的片段都注册为证据 E#，文献编号即 paper_id（P#）。LLM 只能"引用编号"，
       不能自己写文献信息或片段原文——最终展示的书目信息来自数据库，片段文字来自证据库原文。
    2. finish 时程序校验：未在本轮工具结果中出现过的 P#、不存在或不属于该文献的 E# 一律剔除，
       判断文本中的无效引用替换为 [?] 并给出警告。
    3. 提交前做一次"充分性自检"，检索太少/只靠 AI 总结支撑时退回让 LLM 补充检索。
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .config import get_config


_workstation = Path(get_config("workstation_dir")).expanduser()

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
你是 Pageboygent 的「文献检索 Agent」，服务于地球化学、地球物理、行星科学与天体物理方向的研究者。
你的唯一任务：在用户本地追踪的文献库中找出与用户需求相关的文献，并给出有据可查的判断。
你不是百科全书：库外知识一律不能作为结论或候选文献的来源。

## 文献库
每篇文献可能有：原文全文（PDF 转换的 Markdown，已按段切块）、AI 预生成的快速总结（quick summary），
部分文献还有主题总结与精读笔记。库内文献绝大多数为英文。

## 工具
- search_summaries(query, mode): 检索每篇文献的 AI 快速总结。mode=bm25 为相关度排序（自然语言/多词，词间为 OR，命中越多越靠前）；
  mode=keywords 为布尔匹配（支持 AND/OR/NOT/括号/"引号短语"，词间默认 AND），适合精确限定。最快定位候选文献。
- search_fulltext_bm25(query): 原文切块 BM25 检索（已做词干化，evaporation 可命中 evaporated）。
- search_fulltext_semantic(query): 原文切块向量语义检索（RAG），适合换一种说法表达同一概念、或用一句完整的英文描述来找。
- search_fulltext_keywords(expression): 原文切块精确子串匹配（区分大小写！），支持 AND/OR/NOT/"引号短语"。
  适合专有名词、同位素符号（如 "δ66Zn"、"87Rb"）、矿物名、陨石名、样品编号、模型名。
- read_paper_notes(paper_id): 阅读某篇文献的全部 AI 总结、主题总结与精读笔记。用于快速判断文献是否相关、主要讲什么。
- read_paper_passages(paper_id, keywords | line_number): 回到原文核实。给 keywords 返回含这些词的原文段落；
  给 line_number 返回该行附近的上下文；都不给时返回摘要和章节目录。
- lookup_cited_reference(paper_id, citation | keywords): 查库内文献 P# 的参考文献表，返回被引文献条目原文（R#）。
- finish(...): 提交最终结果。

- deep_read_paper(paper_id, question): 派子 agent 通读全文回答具体问题，返回结论与经核实的原文引句。较贵，关键词回读不够时再用。
- recall_evidence(ids): 取回本对话中之前已获得的证据原文（不重新检索）。

工具结果中，P# 是文献编号（即 paper_id），E# 是证据片段编号，R# 是参考文献条目编号。
这些编号在整个对话中始终有效：用户追问时，之前轮次找到的 E#/R#/P# 可以直接引用，不必重新检索；
只有需要新信息时才检索，重复同一个检索不会得到新结果。
为控制上下文长度，较早的工具结果可能显示为"[已折叠] …→ 证据 E101-E130"：这些编号依然有效，
需要看原文时用 recall_evidence 取回，不要为此重新检索。
可以在一次回复中并行调用多个检索工具以节省轮数。

## 检索策略（像一个严谨的研究助理那样工作）
1. 先拆解需求：核心概念、研究对象（天体/样品/元素体系）、过程与机制、方法；列出英文术语、同义词、缩写、化学/同位素符号。
   中文需求一律翻译成英文术语检索，必要时中英都试。
2. 广撒网：先用 search_summaries 定位候选，再用 search_fulltext_bm25 和 search_fulltext_semantic 从原文召回
   （总结可能漏掉文中的次要内容）；专有名词再用 search_fulltext_keywords 精确命中。
3. 每次调用工具前，先在回复正文中用 2-4 句话写出【评估】（目前找到了什么、还缺什么、证据强弱）和【下一步】（准备怎么扩展或收窄）。
4. 自行判断检索是否足够。同时满足以下条件才算够：
   - 需求的每个关键方面，都至少用两种不同检索方式或两种不同措辞查过；
   - 每篇高相关候选都有原文片段（来源为"原文"的 E#）直接支撑，而不只是 AI 总结；只有总结支撑的关键结论，用 read_paper_passages 回原文核实；
   - 最近一轮检索没有再出现新的高相关文献（结果已趋于饱和）。
   不够就扩展：换同义词、上位/下位概念、相关机制/方法/天体、作者的常用表述；结果太杂就收窄：加限定词或用布尔关键词。
5. 不要重复完全相同的查询。轮数有限，优先做信息增益最大的检索。

## 防幻觉铁律（违反任何一条都视为任务失败）
1. 只依据本轮工具返回的内容作答。禁止凭记忆或常识补充任何文献、作者、年份、期刊、数据或结论。
   即使你"知道"某篇著名文献，只要工具没有返回它，就不能提及。
2. 候选文献只能是工具结果中出现过的 P#；证据只能是工具结果中出现过的 E#。不得编造、猜测或改写编号。
3. 每个事实性论断都必须紧跟 [E#] 引用，多个引用写成 [E3][E8]。没有证据的内容要么删掉，要么明确标注"（推断，未见原文直接表述）"。
4. 区分证据等级：原文片段 > 主题总结/精读笔记 > AI 快速总结。AI 总结是二手转述，可能有误；与原文冲突时以原文为准并指出冲突。
5. 不要夸大相关性：文献"提到"某概念 ≠ 文献"研究了"该问题。如实给出相关度。
6. 找不到就直说"库中未找到直接相关的文献"，并在 gaps 中写明缺什么、试过哪些检索。宁缺毋滥，不要为了凑数列入弱相关文献。
7. 不要在结果里大段复述片段原文。系统会按你给出的 E# 自动展示原文片段，你只需挑选最能支撑判断的 E#。
8. 书目信息（作者、年份、标题）由系统从数据库生成，你在文本中用 [P#] 指代文献即可，不要自己写作者年份。

## 二级文献（库中未收录、但被库内文献引用的文献）
当库内文献的原文片段中引用了与需求直接相关的工作（例如 "(Moynier et al., 2006)"、"ref. 14"、上标序号），
而该工作不在库中时，可以作为二级文献列出：
1. 必须先用 lookup_cited_reference 在引用它的那篇库内文献的参考文献表中查到条目（R#），才能列出。没有 R# 的一律不能列。
2. authors/title/year/journal/doi 只能从 R# 条目原文中逐字摘取，条目里没有的字段留空，绝不凭记忆补全（尤其是 DOI）。
   系统会逐字段核对，摘不到原文的字段会被剔除。
3. 你没有读过二级文献本身。对它的描述只能转述库内文献的说法，写成"据 [P#] 所述，……[E#]"，并在 context_evidence_ids 中给出引用处的原文片段 E#。
4. 如果工具提示某条参考文献其实已在库中，就按库内文献处理，不要列为二级文献。
5. 只列与需求直接相关、被库内文献实质讨论的工作，不要把参考文献表整段列出来。
6. 在 overall_judgement 中提到二级文献时用 [R#] 指代（写成方括号形式，系统会渲染成可点击的引用），不要自己写作者和年份。

## 输出
检索充分后调用 finish。读者最先看到的是 short_answer，详细报告默认折叠，所以：
- short_answer（最重要）：先给结论。1-3 句、约 120 字以内，用用户提问的语言，像同事当面回答一样简单直白：
  直接回答问题（有/没有、是哪几篇、核心结论是什么），点名最关键的 1-3 篇文献 [P#]，可带 1-2 个最关键的 [E#]。
  不要铺垫、不要复述问题、不要写方法细节和参数。证据不足或库中没有时，直接说"库中没有找到……"。
  如果是追问，优先回答本轮新发现了什么、与上一轮结论相比有什么变化。
- overall_judgement：详细报告（Markdown，可分点）。展开说明：库里有哪些相关文献、各自说了什么、
  共识与分歧、证据强弱。每个论断带 [E#]，用 [P#] 指代文献。不必重复 short_answer 的原话。
- candidates：按相关度从高到低，每篇给出 paper_id、relevance（high/medium/low）、reason（一两句说明它与需求的具体关系，带 [E#]）、
  evidence_ids（1-4 个最关键的 E#，必须属于该文献，优先原文片段）。
- confidence / gaps / search_summary：见字段说明。
""".strip()

SELF_CHECK_HEADER = "【系统自检】提交前请确认检索是否充分，发现以下问题："
SELF_CHECK_FOOTER = (
    "请针对上述问题继续检索或回原文核实。如果你确认检索已经充分（例如库中确实没有更多相关内容），"
    "可以再次调用 finish（可原样提交）。"
)

FORCE_FINISH_PROMPT = (
    "已达到最大检索轮数。请立即调用 finish，仅基于目前工具已返回的内容提交结果；"
    "证据不足之处如实写入 gaps，不要补充任何工具未返回的信息。"
)

FORMAT_NUDGE_PROMPT = (
    "请不要直接用文字作答。若需继续检索请调用检索工具；若检索已充分，请调用 finish 工具提交结构化结果。"
)


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


_TOP_K = {"type": "integer", "description": "返回条数，默认 12，最大 30", "minimum": 1, "maximum": 30}

SEARCH_TOOLS = [
    _tool(
        "search_summaries",
        "检索每篇文献的 AI 快速总结（quick summary），按文献返回最相关的总结段落。最快定位候选文献。",
        {
            "query": {"type": "string", "description": "英文为主的检索词或布尔表达式"},
            "mode": {
                "type": "string",
                "enum": ["bm25", "keywords"],
                "description": "bm25=相关度排序（词间 OR）；keywords=布尔匹配（词间默认 AND，支持 AND/OR/NOT/括号/引号短语）",
            },
            "top_k": _TOP_K,
        },
        ["query"],
    ),
    _tool(
        "search_fulltext_bm25",
        "原文切块 BM25 检索（词干化，词间 OR，命中越多得分越高；也支持 AND/OR/NOT/引号短语）。",
        {"query": {"type": "string", "description": "英文检索词"}, "top_k": _TOP_K},
        ["query"],
    ),
    _tool(
        "search_fulltext_semantic",
        "原文切块向量语义检索（RAG）。用一句完整、具体的英文描述效果最好。",
        {"query": {"type": "string", "description": "一句英文描述"}, "top_k": _TOP_K},
        ["query"],
    ),
    _tool(
        "search_fulltext_keywords",
        "原文切块精确子串匹配（区分大小写），支持 AND/OR/NOT/括号/\"引号短语\"，词间默认 AND。适合专有名词与符号。",
        {"expression": {"type": "string", "description": "例如: \"δ66Zn\" AND evaporation"}, "top_k": _TOP_K},
        ["expression"],
    ),
    _tool(
        "read_paper_notes",
        "阅读某篇文献全部预生成的 AI 快速总结、主题总结与精读笔记（按段落编号为 E#）。",
        {"paper_id": {"type": "integer", "description": "P# 中的数字"}},
        ["paper_id"],
    ),
    _tool(
        "read_paper_passages",
        "回到原文核实：给 keywords 返回含这些词（不区分大小写）的原文段落；给 line_number 返回该行附近上下文；都不给时返回摘要与章节目录。",
        {
            "paper_id": {"type": "integer", "description": "P# 中的数字"},
            "keywords": {"type": "array", "items": {"type": "string"}, "description": "要在原文中查找的词或短语"},
            "line_number": {"type": "integer", "description": "原文行号（来自 E# 的 L 行号）"},
        },
        ["paper_id"],
    ),
    _tool(
        "lookup_cited_reference",
        "查找库内文献 P# 的参考文献表，返回与正文引文标记匹配的条目原文（编号为 R#）。用于获取库中未收录、但被库内文献引用的二级文献的书目信息。",
        {
            "paper_id": {"type": "integer", "description": "引用该文献的库内文献 P# 的数字"},
            "citation": {
                "type": "string",
                "description": "正文中的引文标记，如 \"Moynier et al., 2006\"、\"Paniello and Day 2012\"、\"ref. 14\"、\"[3,5-7]\"",
            },
            "keywords": {"type": "array", "items": {"type": "string"}, "description": "没有明确引文标记时，用标题/作者关键词在参考文献表中查找"},
        },
        ["paper_id"],
    ),
]

SEARCH_TOOLS.append(_tool(
    "deep_read_paper",
    "派一个子 agent 通读某篇库内文献的全文（参考文献表除外），回答一个具体问题。返回阅读结论和经程序核实的原文引句（登记为 E#）。"
    "适用于关键词回读不够、需要通读才能回答的问题（方法细节、参数设置、完整论证链、作者的保留意见）。"
    "比 read_paper_passages 贵得多（每次约 2-4 万 tokens），每轮最多 3 次；能用关键词回读解决的不要用它。",
    {
        "paper_id": {"type": "integer", "description": "P# 中的数字"},
        "question": {"type": "string", "description": "要在全文中寻找答案的具体问题（越具体越好）"},
    },
    ["paper_id", "question"],
))

SEARCH_TOOLS.append(_tool(
    "recall_evidence",
    "按编号取回本对话中之前已获得的证据原文（E#/R#，任何一轮的都可以），不重新检索、编号不变。上下文被压缩后用它查看旧证据。",
    {"ids": {"type": "array", "items": {"type": "string"}, "description": "例如 [\"E12\", \"R3\"]，最多 20 个"}},
    ["ids"],
))

FINISH_TOOL = _tool(
    "finish",
    "检索充分后提交最终结果。只能引用本轮工具结果中出现过的 P# 与 E#。",
    {
        "short_answer": {
            "type": "string",
            "description": "简单直白的回答，1-3 句、约 120 字以内：直接回答问题并点名最关键的 1-3 篇 [P#]；不写方法细节。追问时优先说本轮新发现",
        },
        "overall_judgement": {
            "type": "string",
            "description": "详细报告（Markdown，用户提问的语言），每个论断带 [E#] 引用，用 [P#] 指代文献",
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
            "description": "对'库中相关文献已基本找全且判断有据'的把握",
        },
        "candidates": {
            "type": "array",
            "description": "候选文献，按相关度从高到低；没有相关文献时为空数组",
            "items": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "integer"},
                    "relevance": {"type": "string", "enum": ["high", "medium", "low"]},
                    "reason": {"type": "string", "description": "与需求的具体关系，一两句，带 [E#]"},
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "1-3 个属于该文献的关键 E#，优先原文片段；第 1 个最重要（默认只展示前 2 个）",
                    },
                },
                "required": ["paper_id", "relevance", "reason", "evidence_ids"],
            },
        },
        "secondary_references": {
            "type": "array",
            "description": "二级文献：库中未收录、但被库内文献引用且与需求直接相关的文献。只能来自 lookup_cited_reference 返回的 R#；没有则为空数组",
            "items": {
                "type": "object",
                "properties": {
                    "ref_id": {"type": "string", "description": "R#"},
                    "context_evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "库内文献中引用该文献的原文片段 E#（说明它为什么相关）",
                    },
                    "relevance": {"type": "string", "enum": ["high", "medium", "low"]},
                    "reason": {"type": "string", "description": "据引用它的库内文献所述，该文献与需求的关系（转述，带 [E#]）"},
                    "authors": {"type": "string", "description": "从 R# 条目原文中逐字摘取；条目中没有则留空"},
                    "title": {"type": "string", "description": "从 R# 条目原文中逐字摘取；没有则留空"},
                    "year": {"type": "string", "description": "从 R# 条目原文中摘取；没有则留空"},
                    "journal": {"type": "string", "description": "从 R# 条目原文中逐字摘取；没有则留空"},
                    "doi": {"type": "string", "description": "从 R# 条目原文中摘取；没有则留空，绝不能自行补全"},
                },
                "required": ["ref_id", "context_evidence_ids", "relevance", "reason"],
            },
        },
        "gaps": {"type": "string", "description": "库中缺失或证据不足之处、未能核实的内容；没有则写'无'"},
        "search_summary": {"type": "string", "description": "一两句概述做过哪些检索（检索方式与主要关键词）"},
    },
    ["short_answer", "overall_judgement", "confidence", "candidates", "gaps", "search_summary"],
)

SEARCH_TOOL_NAMES = {
    "search_summaries",
    "search_fulltext_bm25",
    "search_fulltext_semantic",
    "search_fulltext_keywords",
}

# ---------------------------------------------------------------------------
# Search backend: thin adapters over existing storage / rag / note code
# ---------------------------------------------------------------------------

_ZOTERO_KEY_RE = re.compile(r"^[A-Z0-9]{8}$")


class SearchBackend:
    """所有数据访问都集中在这里，方便测试时替换。"""

    def summary_bm25(self, query: str, limit: int) -> list[tuple[int, float, str]]:
        from .storage import search_quick_summary_bm25

        return search_quick_summary_bm25(query, limit=limit)

    def summary_keywords(self, expression: str, limit: int) -> list[tuple[int, float | None, str]]:
        from .storage import search_quick_summary_by_key_words

        if not expression.strip():  # 空查询会返回全部文献
            return []
        hits = search_quick_summary_by_key_words(expression)
        return [(pid, None, summary) for pid, summary in list(hits.items())[:limit]]

    def fulltext_bm25(self, query: str, limit: int) -> list[dict]:
        from . import rag

        return rag.bm25_retrieve(query, top_k=limit)

    def fulltext_semantic(self, query: str, limit: int) -> list[dict]:
        from . import rag

        return rag.vector_retrieve(query, top_k=limit)

    def fulltext_keywords(self, expression: str, limit: int) -> list[dict]:
        from . import rag

        return rag.keywords_retrieve(expression)[:limit]

    def paper_meta(self, paper_id: int) -> dict | None:
        from .storage import Paper, PDFStatus

        paper = Paper.get_or_none(Paper.id == paper_id)
        if paper is None:
            return None
        pdf_status = PDFStatus.get_or_none(PDFStatus.paper == paper.id)
        pdf_key = str(pdf_status.pdf_key_in_zotero) if pdf_status and pdf_status.pdf_key_in_zotero else None
        md_path = _workstation / "out_mds" / f"{pdf_key}.md" if pdf_key else None
        return build_paper_meta(
            paper_id=int(paper.id),
            title=str(paper.title or ""),
            year=paper.year,
            first_author=paper.first_author,
            doi=paper.doi,
            zotero_key=str(paper.key_in_zotero),
            pdf_key=pdf_key,
            md_path=str(md_path) if md_path and md_path.exists() else None,
            has_summary=bool(paper.quick_summary),
        )

    def paper_notes(self, paper_id: int) -> list[dict]:
        """快速总结 + 主题总结 + 其他 AI 笔记（精读笔记等），按内容去重。"""
        from .note_storage import NoteRecord
        from .storage import Paper, Theme, ThemePaper, ai_tag, summary_tag

        paper = Paper.get_or_none(Paper.id == paper_id)
        if paper is None:
            return []
        docs: list[dict] = []
        seen_digests: set[str] = set()

        def _add(key: str, kind: str, label: str, content: str | None) -> None:
            text = (content or "").strip()
            if not text:
                return
            digest = hashlib.md5(re.sub(r"\s+", " ", text).encode()).hexdigest()
            if digest in seen_digests:
                return
            seen_digests.add(digest)
            docs.append({"key": key, "kind": kind, "label": label, "content": text})

        _add("qs", "summary", "AI 快速总结", paper.quick_summary)
        covered_tags = {summary_tag}
        query = ThemePaper.select(ThemePaper, Theme).join(Theme).where(ThemePaper.paper == paper.id)
        for theme_paper in query:
            if theme_paper.theme_specific_summary:
                covered_tags.add(str(theme_paper.theme.note_tag))
                _add(f"theme{theme_paper.id}", "theme", f"主题总结 · {theme_paper.theme.name}",
                     theme_paper.theme_specific_summary)
        for record in NoteRecord.select().where(NoteRecord.item_key == str(paper.key_in_zotero)):
            tags = [t for t in str(record.tags_key).split(";") if t]
            if covered_tags & set(tags):
                continue  # 与快速总结/主题总结重复的笔记副本
            label_tags = [t for t in tags if t != ai_tag] or tags
            _add(f"note{record.id}", "note", "笔记 · " + "; ".join(label_tags), str(record.content_md))
        return docs

    _library_index: tuple[dict[str, int], dict[str, int], dict[tuple[str, str], list[int]]] | None = None

    def _build_library_index(self) -> tuple[dict[str, int], dict[str, int], dict[tuple[str, str], list[int]]]:
        from .storage import Paper

        by_doi: dict[str, int] = {}
        by_title: dict[str, int] = {}
        by_author_year: dict[tuple[str, str], list[int]] = {}
        for paper in Paper.select(Paper.id, Paper.doi, Paper.title, Paper.first_author, Paper.year):
            if paper.doi:
                by_doi[str(paper.doi).lower()] = int(paper.id)
            title = _norm_alnum(str(paper.title or ""))
            if len(title) >= 25:
                by_title[title] = int(paper.id)
            if paper.first_author and paper.year:
                # 只比字母数字：数据库里可能是 "Obrien"，参考文献里是 "O’Brien"
                key = (_norm_alnum(_fold(str(paper.first_author))), str(paper.year))
                by_author_year.setdefault(key, []).append(int(paper.id))
        return by_doi, by_title, by_author_year

    def find_in_library(self, doi: str | None, raw_reference: str) -> int | None:
        """判断被引文献是否其实已在库中。依次按 DOI、标题、"第一作者 + 年份"（库中唯一匹配时）核对。

        天文类期刊的参考文献常只有 "Chambers, J. E. 2001, Icar, 152, 205"，没有标题和 DOI，
        只能靠第一作者 + 年份；库中同一作者同年有多篇时不下结论。
        """
        if self._library_index is None:
            self._library_index = self._build_library_index()
        by_doi, by_title, by_author_year = self._library_index
        if doi and doi.lower() in by_doi:
            return by_doi[doi.lower()]
        raw_norm = _norm_alnum(raw_reference)
        for title, paper_id in by_title.items():
            if title in raw_norm:
                return paper_id
        surname, year = _reference_signature(raw_reference, None)
        matches = by_author_year.get((_norm_alnum(surname or ""), year or ""), [])
        return matches[0] if len(matches) == 1 else None

    def md_lines(self, meta: dict) -> list[str] | None:
        md_path = meta.get("md_path")
        if not md_path:
            return None
        try:
            return Path(md_path).read_text(encoding="utf-8").splitlines()
        except OSError:
            return None


def build_paper_meta(
    *,
    paper_id: int,
    title: str,
    year: int | None,
    first_author: str | None,
    doi: str | None,
    zotero_key: str,
    pdf_key: str | None = None,
    md_path: str | None = None,
    has_summary: bool = False,
) -> dict:
    author = (first_author or "").strip() or "Unknown"
    year_text = str(year) if year else "n.d."
    zotero_ok = bool(_ZOTERO_KEY_RE.match(zotero_key or ""))
    pdf_ok = bool(pdf_key and _ZOTERO_KEY_RE.match(pdf_key))
    return {
        "paper_id": paper_id,
        "title": title,
        "year": year,
        "first_author": first_author,
        "doi": doi,
        "zotero_key": zotero_key,
        "pdf_key": pdf_key,
        "md_path": md_path,
        "has_fulltext": bool(md_path),
        "has_summary": has_summary,
        "short_cite": f"{author} ({year_text})",
        # zotero://select 在 Zotero 桌面端中定位并选中该条目；open-pdf 直接打开附件 PDF
        "zotero_select_url": f"zotero://select/library/items/{zotero_key}" if zotero_ok else None,
        "zotero_pdf_url": f"zotero://open-pdf/library/items/{pdf_key}" if pdf_ok else None,
        "doi_url": f"https://doi.org/{doi}" if doi else None,
    }


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_BOOLEAN_WORDS = {"AND", "OR", "NOT", "ANDNOT", "ANDMAYBE"}
_REFERENCES_HEADER_RE = re.compile(r"^#+\s*(references|bibliography|references and notes|参考文献)\b", re.I)


def _analyzer():
    from whoosh.analysis import StemmingAnalyzer

    return StemmingAnalyzer()


def _stems(text: str) -> set[str]:
    return {token.text for token in _analyzer()(text)}


def _query_stems(query: str) -> set[str]:
    words = [w for w in re.findall(r"[^\s()\"']+", query) if w not in _BOOLEAN_WORDS]
    return _stems(" ".join(words))


def split_paragraphs(text: str, min_chars: int = 40) -> list[str]:
    """按空行切段；标题行/过短的行并入下一段，避免出现只有标题的证据。"""
    raw = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    paragraphs: list[str] = []
    carry = ""
    for para in raw:
        merged = f"{carry}\n{para}".strip() if carry else para
        if len(merged) < min_chars or (merged.startswith("#") and "\n" not in merged):
            carry = merged
            continue
        paragraphs.append(merged)
        carry = ""
    if carry:
        if paragraphs:
            paragraphs[-1] = f"{paragraphs[-1]}\n{carry}"
        else:
            paragraphs.append(carry)
    return paragraphs


def _best_paragraphs(text: str, query: str, top: int = 2) -> list[tuple[int, str]]:
    """在总结中挑出与查询词重合最多的段落（按词干匹配），返回 (段落序号, 段落)。"""
    paragraphs = split_paragraphs(text)
    terms = _query_stems(query)
    scored = []
    for index, para in enumerate(paragraphs):
        if re.match(r"^#*\s*\**labels\**", para, re.I):
            continue  # 标签列表不是有效证据
        overlap = len(terms & _stems(para)) if terms else 0
        if overlap:
            scored.append((overlap, -index, index, para))
    scored.sort(reverse=True)
    picked = [(index, para) for _, _, index, para in scored[:top]]
    if not picked and paragraphs:
        picked = [(0, paragraphs[0])]
    return sorted(picked)


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " …"


def _window_around(text: str, keywords: list[str], limit: int) -> str:
    """长段落时截取第一个关键词附近的窗口。"""
    if len(text) <= limit:
        return text
    lowered = text.lower()
    positions = [lowered.find(k.lower()) for k in keywords if k and lowered.find(k.lower()) >= 0]
    center = min(positions) if positions else 0
    start = max(0, center - limit // 3)
    end = min(len(text), start + limit)
    return ("… " if start > 0 else "") + text[start:end].strip() + (" …" if end < len(text) else "")


# ---------------------------------------------------------------------------
# Reference list parsing (secondary literature)
# ---------------------------------------------------------------------------

_REF_SECTION_RE = re.compile(
    r"^#+\s*(references|bibliography|references and notes|literature cited|reference list|参考文献)\b", re.I)
_REF_SECTION_END_RE = re.compile(
    r"^#+\s*(appendix|supplementary|supporting information|acknowledg|author contributions|"
    r"competing interests|extended data|figure|table)", re.I)
_NUMBERED_REF_RE = re.compile(r"^\s*(?:\[(\d{1,3})\]|(\d{1,3})\.)\s+(?=\S)")
# 一行里挤了多条 author-year 文献时的切分点：". Surname, X." 之前
_MERGED_REF_SPLIT_RE = re.compile(
    r"(?<=[.\d])\s+(?=[A-Z][A-Za-z'’\-]+,\s(?:[A-Z]\.|[A-Z][a-z]{1,2}\.)[^\n]{0,200}?\b(?:19|20)\d{2})")
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s,;\"<>]+", re.I)
_CITE_STOPWORDS = {"et", "al", "and", "in", "see", "e.g", "i.e", "ref", "refs", "cf", "also", "amp"}


def parse_reference_entries(lines: list[str]) -> list[tuple[int | None, str]]:
    """从原文 Markdown 中解析参考文献条目，返回 [(序号或 None, 条目原文), ...]。

    支持两种格式：`# References` 标题之后的列表（author-year 为主）；以及没有标题、
    以 "1." / "[1]" 编号的列表（Nature 类期刊常见，取最长的连续编号段）。
    """
    entries: list[tuple[int | None, str]] = []
    header_idx = [i for i, line in enumerate(lines) if _REF_SECTION_RE.match(line.strip())]
    if header_idx:
        for line in lines[header_idx[-1] + 1:]:
            text = line.strip()
            if not text:
                continue
            if _REF_SECTION_END_RE.match(text):
                break
            if text.startswith("#"):
                continue
            pieces = _MERGED_REF_SPLIT_RE.split(text) if len(text) > 200 else [text]
            for piece in pieces:
                piece = piece.strip()
                if len(piece) >= 15:
                    match = _NUMBERED_REF_RE.match(piece)
                    number = int(match.group(1) or match.group(2)) if match else None
                    entries.append((number, piece))
    if any(number is not None for number, _ in entries):
        return entries

    # 无标题的编号列表：找从 1 开始、序号递增的最长连续段
    best: list[tuple[int, str]] = []
    run: list[tuple[int, str]] = []
    for line in lines:
        match = _NUMBERED_REF_RE.match(line)
        if not match:
            continue
        number = int(match.group(1) or match.group(2))
        if number == 1:
            if len(run) > len(best):
                best = run
            run = [(1, line.strip())]
        elif run and number == run[-1][0] + 1:
            run.append((number, line.strip()))
    if len(run) > len(best):
        best = run
    if len(best) >= 5 and len(best) > len(entries):
        return [(number, text) for number, text in best]
    return entries


_FOLD_TABLE = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'", "ʼ": "'", "“": '"', "”": '"',
                             "‐": "-", "‑": "-", "–": "-", "—": "-", "−": "-"})


def _fold(text: str) -> str:
    """匹配用的归一化：弯引号→直引号、各类连字符→"-"、去重音（Albarède→albarede）、小写。"""
    import unicodedata

    text = unicodedata.normalize("NFKD", (text or "").translate(_FOLD_TABLE))
    return "".join(ch for ch in text if not unicodedata.combining(ch)).lower()


def _norm_alnum(text: str) -> str:
    return re.sub(r"[\W_]+", "", (text or "").lower())


def extract_doi(text: str) -> str | None:
    # MinerU 有时会在 DOI 的 "/" 后插入空格，或把 "-" 转成 Unicode 连字符
    cleaned = re.sub(r"(10\.\d{4,9}/)\s+", r"\1", text or "")
    cleaned = cleaned.replace("‐", "-").replace("‑", "-").replace("–", "-")
    match = _DOI_RE.search(cleaned)
    if not match:
        return None
    return match.group(0).rstrip(".)]").lower()


def _expand_numbers(citation: str) -> set[int]:
    numbers: set[int] = set()
    for a, b in re.findall(r"(\d{1,3})\s*[–\-−]\s*(\d{1,3})", citation):
        lo, hi = int(a), int(b)
        if 0 < lo <= hi and hi - lo < 30:
            numbers.update(range(lo, hi + 1))
    numbers.update(int(n) for n in re.findall(r"\d{1,3}", re.sub(r"\b(19|20)\d{2}\b", "", citation)))
    return numbers


def match_reference_entries(entries: list[tuple[int | None, str]], citation: str | None = None,
                            keywords: list[str] | None = None, limit: int = 4) -> list[int]:
    """按正文中的引文标记（"Moynier et al., 2006" / "ref. 14" / "[3,5-7]"）或关键词匹配参考文献条目，返回下标。"""
    citation = _fold(citation or "").strip()
    if citation:
        years = re.findall(r"\b(?:19|20)\d{2}[a-z]?\b", citation)
        # citation 已小写：按"非停用词、长度>1、含字母"取姓氏
        surnames = [w for w in re.findall(r"[a-z][a-z'\-]+", citation)
                    if w not in _CITE_STOPWORDS and len(w) > 1]
        if not years and not surnames:
            numbers = _expand_numbers(citation)
            return [i for i, (number, _) in enumerate(entries) if number in numbers][:max(limit, len(numbers))]
        scored = []
        for index, (_, raw_text) in enumerate(entries):
            text = _fold(raw_text)
            score = 0
            if years:
                if not any(y[:4] in text for y in years):
                    continue
                score += 3
            if surnames:
                # 去掉编号前缀后，第一作者姓氏必须位于条目最开头，前面最多是名字缩写（"K. Lodders"、"J.M.D. Day"）
                body = _NUMBERED_REF_RE.sub("", text, count=1)
                if re.match(r"(?:[a-z]\.\s?-?)*" + re.escape(surnames[0]), body):
                    score += 4
                else:
                    continue
                score += sum(1 for s in surnames[1:] if s in text)
            scored.append((score, -index, index))
        scored.sort(reverse=True)
        return [index for _, _, index in scored[:limit]]
    keys = [_fold(k) for k in (keywords or []) if isinstance(k, str) and k.strip()]
    if not keys:
        return []
    scored = []
    for index, (_, text) in enumerate(entries):
        low = _fold(text)
        score = sum(1 for k in keys if k in low)
        if score:
            scored.append((score, -index, index))
    scored.sort(reverse=True)
    return [index for _, _, index in scored[:limit]]


def validate_reference_fields(raw: str, fields: dict) -> tuple[dict, list[str]]:
    """模型给出的引文字段必须能在参考文献原文中找到，否则剔除。返回 (保留的字段, 被剔除的字段名)。"""
    raw_norm = _norm_alnum(raw)
    kept: dict = {}
    dropped: list[str] = []
    for name in ("authors", "title", "year", "journal", "doi"):
        value = fields.get(name)
        if value is None or str(value).strip() == "":
            continue
        value = str(value).strip()
        if name == "authors":
            tokens = [t for t in re.findall(r"[A-Za-zÀ-ÿ'’\-]{2,}", value) if t.lower() not in _CITE_STOPWORDS]
            ok = bool(tokens) and all(_norm_alnum(t) in raw_norm for t in tokens)
        elif name == "doi":
            ok = _norm_alnum(value) in _norm_alnum(extract_doi(raw) or "") or _norm_alnum(value) in raw_norm
        else:
            ok = bool(_norm_alnum(value)) and _norm_alnum(value) in raw_norm
        if ok:
            kept[name] = value
        else:
            dropped.append(name)
    doi = extract_doi(raw)
    if doi:
        kept["doi"] = doi  # DOI 一律以程序从原文提取的为准
    if "year" not in kept:
        year = re.search(r"\b(?:19|20)\d{2}\b", raw)
        if year:
            kept["year"] = year.group(0)
    return kept, dropped


# ---------------------------------------------------------------------------
# Deep reading sub-agent（fork：全文只进子 agent 的上下文，主上下文只拿到结论与核实过的引句）
# ---------------------------------------------------------------------------

DEEP_READ_SYSTEM_PROMPT = """
你是文献精读助手。你会收到一篇学术论文的全文（Markdown，参考文献表已删除）和一个问题。
只依据这篇论文的原文回答，不得使用论文以外的知识，不得猜测。

输出一个 JSON 对象：
{
  "answer": "用提问的语言回答，300 字以内。原文没有讨论这个问题就明确说'原文未涉及'。",
  "found": true 或 false（原文是否直接讨论了这个问题）,
  "quotes": [
    {"quote": "从原文逐字复制的句子或短段落（40-400 个字符），必须与原文完全一致：不得翻译、改写、缩写或用省略号拼接", "note": "这句说明了什么（一句话）"}
  ]
}
quotes 给 1-6 条最能支撑 answer 的原文引句，优先正文与图表说明，不要摘录标题或参考文献。
系统会逐条核对引句是否真的出现在原文中，核对不上的会被丢弃。
""".strip()


def _norm_quote(text: str) -> str:
    """引句比对用：折叠引号/连字符/重音、去 Markdown 强调与行内公式定界符、压缩空白。"""
    text = _fold(text)
    text = re.sub(r"[*_`$]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def locate_quote(lines: list[str], quote: str) -> tuple[int, str] | None:
    """在原文中定位引句，返回 (行号, 原文中对应的文字)。只接受逐字（归一化后）出现的引句。"""
    needle = _norm_quote(quote)
    if len(needle) < 20:
        return None
    for index, line in enumerate(lines, start=1):
        if needle in _norm_quote(line):
            return index, _window_around(line.strip(), [quote[:40]], 1500)
    # 跨行引句：与下一行拼接后再找
    for index in range(1, len(lines)):
        joined = lines[index - 1].strip() + " " + lines[index].strip()
        if needle in _norm_quote(joined):
            return index, _window_around(joined, [quote[:40]], 1500)
    return None


def prepare_fulltext(lines: list[str], max_chars: int) -> tuple[str, bool]:
    """去掉参考文献表（只在后半部分出现时才截断），超长时截断。返回 (文本, 是否截断)。"""
    cut = len(lines)
    for index, line in enumerate(lines):
        if index > len(lines) // 2 and _REF_SECTION_RE.match(line.strip()):
            cut = index
            break
    text = "\n".join(lines[:cut])
    if len(text) > max_chars:
        return text[:max_chars], True
    return text, False


# ---------------------------------------------------------------------------
# Evidence registry + toolbox
# ---------------------------------------------------------------------------

KIND_LABELS = {"fulltext": "原文", "summary": "AI 快速总结", "theme": "主题总结", "note": "笔记", "reference": "参考文献"}


@dataclass
class Evidence:
    eid: str
    paper_id: int
    kind: str  # fulltext | summary | theme | note
    label: str
    text: str
    line_number: int | None = None
    methods: list[str] = field(default_factory=list)
    score: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class LitSearchToolbox:
    """执行工具调用，把所有返回片段登记为证据，并把结果格式化成给 LLM 读的文本。"""

    LLM_SNIPPET_CHARS = 600
    MAX_RESULT_CHARS = 12000
    MAX_PER_PAPER = 3

    def __init__(self, backend: SearchBackend | None = None) -> None:
        self.backend = backend or SearchBackend()
        self.evidence: dict[str, Evidence] = {}
        self._evidence_by_key: dict[tuple, str] = {}
        self._counters: dict[str, int] = {}
        self._reference_cache: dict[int, list[tuple[int | None, str]]] = {}
        self._meta_cache: dict[int, dict | None] = {}
        self.seen_papers: list[int] = []
        self.calls: list[dict] = []  # 本轮的工具调用（每轮清空，自检用）
        self.turn_index = 1
        # 同一对话里已执行过的检索：key -> 第几轮。上下文完整保留时，重复检索直接指回上文，不再执行
        self.call_cache: dict[str, int] = {}
        self.deep_reader: Callable[[str, str], dict] | None = None  # (全文, 问题) -> {"answer","found","quotes"}
        self.deep_reads_this_turn = 0
        self.last_touched: list[str] = []  # 最近一次工具调用涉及的证据编号（折叠旧工具结果时用）
        self._touched: list[str] = []

    # -- persistence（同一对话共用一套证据编号）-----------------------------
    def to_state(self) -> dict:
        return {
            "evidence": [ev.to_dict() for ev in self.evidence.values()],
            "evidence_by_key": [[list(key), eid] for key, eid in self._evidence_by_key.items()],
            "counters": dict(self._counters),
            "seen_papers": list(self.seen_papers),
            "call_cache": dict(self.call_cache),
        }

    def load_state(self, state: dict) -> None:
        self.evidence = {d["eid"]: Evidence(**d) for d in state.get("evidence", [])}
        self._evidence_by_key = {tuple(key): eid for key, eid in state.get("evidence_by_key", [])}
        self._counters = dict(state.get("counters", {}))
        self.seen_papers = [int(pid) for pid in state.get("seen_papers", [])]
        self.call_cache = {k: int(v) for k, v in state.get("call_cache", {}).items()}

    def new_turn(self, turn_index: int) -> None:
        self.calls = []
        self.turn_index = turn_index
        self.deep_reads_this_turn = 0

    # -- registry ---------------------------------------------------------
    def paper(self, paper_id: int) -> dict | None:
        if paper_id not in self._meta_cache:
            try:
                self._meta_cache[paper_id] = self.backend.paper_meta(paper_id)
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[lit_search] failed to load paper {paper_id}: {exc}")
                self._meta_cache[paper_id] = None
        meta = self._meta_cache[paper_id]
        if meta is not None and paper_id not in self.seen_papers:
            self.seen_papers.append(paper_id)
        return meta

    def add_evidence(
        self,
        key: tuple,
        *,
        paper_id: int,
        kind: str,
        label: str,
        text: str,
        method: str,
        line_number: int | None = None,
        score: float | None = None,
        prefix: str = "E",
    ) -> Evidence:
        eid = self._evidence_by_key.get(key)
        if eid is not None:
            if eid not in self._touched:
                self._touched.append(eid)
            ev = self.evidence[eid]
            if method not in ev.methods:
                ev.methods.append(method)
            if score is not None and (ev.score is None or score > ev.score):
                ev.score = score
            return ev
        self._counters[prefix] = self._counters.get(prefix, 0) + 1
        eid = f"{prefix}{self._counters[prefix]}"
        ev = Evidence(eid=eid, paper_id=paper_id, kind=kind, label=label, text=text.strip(),
                      line_number=line_number, methods=[method], score=score)
        self.evidence[eid] = ev
        self._evidence_by_key[key] = eid
        self._touched.append(eid)
        return ev

    # -- formatting -------------------------------------------------------
    def _paper_line(self, meta: dict) -> str:
        flags = []
        if not meta.get("has_fulltext"):
            flags.append("无本地原文")
        if not meta.get("has_summary"):
            flags.append("无AI总结")
        flag_text = f"  [{', '.join(flags)}]" if flags else ""
        return f"[P{meta['paper_id']}] {meta['short_cite']} {meta['title']}{flag_text}"

    def _evidence_line(self, ev: Evidence) -> str:
        where = f"原文 L{ev.line_number}" if ev.kind == "fulltext" and ev.line_number else ev.label
        score = f" | score={ev.score:.3g}" if ev.score is not None else ""
        return f"  {ev.eid} ({where}{score}): {_clip(ev.text, self.LLM_SNIPPET_CHARS)}"

    def _format_grouped(self, header: str, grouped: dict[int, list[Evidence]]) -> str:
        if not grouped:
            return f"{header}\n无结果。可以换同义词、放宽条件或换一种检索方式。"
        n_evidence = sum(len(v) for v in grouped.values())
        lines = [f"{header}\n共 {n_evidence} 个片段，涉及 {len(grouped)} 篇文献："]
        for paper_id, items in grouped.items():
            meta = self.paper(paper_id)
            if meta is None:
                continue
            lines.append(self._paper_line(meta))
            lines.extend(self._evidence_line(ev) for ev in items)
        return _clip("\n".join(lines), self.MAX_RESULT_CHARS)

    # -- tools ------------------------------------------------------------
    def _group_chunks(self, hits: list[dict], method: str) -> dict[int, list[Evidence]]:
        grouped: dict[int, list[Evidence]] = {}
        for hit in hits:
            try:
                paper_id = int(hit["paper_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if self.paper(paper_id) is None:
                continue  # 数据库中已不存在的文献（陈旧切块），不提供给 LLM
            bucket = grouped.setdefault(paper_id, [])
            if len(bucket) >= self.MAX_PER_PAPER:
                continue
            text = str(hit.get("text") or "")
            line = hit.get("line_number")
            line = int(line) if line is not None else None
            key = ("chunk", paper_id, line, hashlib.md5(text.encode()).hexdigest())
            ev = self.add_evidence(key, paper_id=paper_id, kind="fulltext", label=f"原文 L{line}",
                                   text=text, method=method, line_number=line, score=hit.get("score"))
            if ev not in bucket:
                bucket.append(ev)
        return grouped

    def search_summaries(self, query: str, mode: str = "bm25", top_k: int = 12) -> str:
        mode = "keywords" if mode == "keywords" else "bm25"
        if mode == "bm25":
            hits = self.backend.summary_bm25(query, top_k)
        else:
            hits = self.backend.summary_keywords(query, top_k)
        grouped: dict[int, list[Evidence]] = {}
        for paper_id, score, summary in hits:
            paper_id = int(paper_id)
            if self.paper(paper_id) is None:
                continue
            for index, para in _best_paragraphs(summary or "", query):
                ev = self.add_evidence(("note", paper_id, "qs", index), paper_id=paper_id, kind="summary",
                                       label=f"AI 快速总结 §{index + 1}", text=para,
                                       method=f"summary-{mode}", score=score)
                grouped.setdefault(paper_id, []).append(ev)
        return self._format_grouped(f"search_summaries(mode={mode}, query={query!r})", grouped)

    def search_fulltext_bm25(self, query: str, top_k: int = 12) -> str:
        # 多取一些再按文献限流，保证结果覆盖更多文献
        hits = self.backend.fulltext_bm25(query, top_k * 2)
        grouped = self._trim(self._group_chunks(hits, "fulltext-bm25"), top_k)
        return self._format_grouped(f"search_fulltext_bm25(query={query!r})", grouped)

    def search_fulltext_semantic(self, query: str, top_k: int = 12) -> str:
        hits = self.backend.fulltext_semantic(query, top_k * 2)
        grouped = self._trim(self._group_chunks(hits, "fulltext-semantic"), top_k)
        return self._format_grouped(f"search_fulltext_semantic(query={query!r})", grouped)

    def search_fulltext_keywords(self, expression: str, top_k: int = 12) -> str:
        hits = self.backend.fulltext_keywords(expression, max(top_k * 4, 40))
        grouped = self._trim(self._group_chunks(hits, "fulltext-keywords"), top_k)
        return self._format_grouped(f"search_fulltext_keywords(expression={expression!r})", grouped)

    @staticmethod
    def _trim(grouped: dict[int, list[Evidence]], top_k: int) -> dict[int, list[Evidence]]:
        out: dict[int, list[Evidence]] = {}
        count = 0
        for paper_id, items in grouped.items():
            if count >= top_k:
                break
            out[paper_id] = items[: max(1, top_k - count)]
            count += len(out[paper_id])
        return out

    def read_paper_notes(self, paper_id: int) -> str:
        meta = self.paper(int(paper_id))
        if meta is None:
            return f"P{paper_id} 不存在于文献库中，请只使用工具结果中出现过的 P#。"
        docs = self.backend.paper_notes(int(paper_id))
        if not docs:
            return f"{self._paper_line(meta)}\n该文献没有预生成的 AI 总结或笔记。可用 read_paper_passages 阅读原文。"
        lines = [self._paper_line(meta)]
        budget = self.MAX_RESULT_CHARS
        for doc in docs:
            lines.append(f"## {doc['label']}")
            for index, para in enumerate(split_paragraphs(doc["content"])):
                ev = self.add_evidence(("note", meta["paper_id"], doc["key"], index), paper_id=meta["paper_id"],
                                       kind=doc["kind"], label=f"{doc['label']} §{index + 1}", text=para,
                                       method="read-notes")
                line = f"  {ev.eid}: {_clip(para, 900)}"
                budget -= len(line)
                if budget < 0:
                    lines.append("  …（笔记过长，其余段落已省略）")
                    break
                lines.append(line)
            if budget < 0:
                break
        return "\n".join(lines)

    def read_paper_passages(self, paper_id: int, keywords: list[str] | None = None,
                            line_number: int | None = None, max_passages: int = 6) -> str:
        meta = self.paper(int(paper_id))
        if meta is None:
            return f"P{paper_id} 不存在于文献库中，请只使用工具结果中出现过的 P#。"
        lines = self.backend.md_lines(meta)
        if not lines:
            return f"{self._paper_line(meta)}\n该文献没有可读取的本地原文。只能依据 AI 总结，并在结论中注明未经原文核实。"
        pid = meta["paper_id"]
        keywords = [k for k in (keywords or []) if isinstance(k, str) and k.strip()]
        out = [self._paper_line(meta)]

        def _register(start: int, end: int, text: str) -> Evidence:
            return self.add_evidence(("passage", pid, start, end), paper_id=pid, kind="fulltext",
                                     label=f"原文 L{start}-{end}" if end > start else f"原文 L{start}",
                                     text=text, method="read-passages", line_number=start)

        if line_number:
            n = int(line_number)
            start, end = max(1, n - 8), min(len(lines), n + 4)
            text = "\n".join(l for l in lines[start - 1:end] if l.strip())
            ev = _register(start, end, _clip(text, 3000))
            out.append(f"  {ev.eid} (原文 L{start}-{end}): {ev.text}")
            return "\n".join(out)

        if keywords:
            lowered_keys = [_fold(k) for k in keywords]
            scored = []
            for index, line in enumerate(lines, start=1):
                if _REFERENCES_HEADER_RE.match(line.strip()):
                    break  # 参考文献列表之后不再匹配
                low = _fold(line)
                matched = sum(1 for k in lowered_keys if k in low)
                if matched and len(line.strip()) >= 30:
                    scored.append((matched, -index, index, line.strip()))
            if not scored:
                return "\n".join(out + [f"  原文中未找到 {keywords}。可换同义词、单复数或缩写再试。"])
            scored.sort(reverse=True)
            for _, _, index, text in sorted(scored[:max_passages], key=lambda item: item[2]):
                ev = _register(index, index, _window_around(text, keywords, 1500))
                out.append(f"  {ev.eid} (原文 L{index}): {ev.text}")
            if len(scored) > max_passages:
                out.append(f"  …另有 {len(scored) - max_passages} 处匹配未列出，可用更具体的关键词。")
            return "\n".join(out)

        # 无参数：返回开头（标题/摘要）与章节目录，便于定位
        head_text, head_end = [], 0
        for index, line in enumerate(lines, start=1):
            if line.strip():
                head_text.append(line.strip())
            head_end = index
            if sum(len(t) for t in head_text) > 1800:
                break
        ev = _register(1, head_end, _clip("\n".join(head_text), 2000))
        out.append(f"  {ev.eid} (原文 L1-{head_end}，开头/摘要): {ev.text}")
        headings = [f"L{i}: {l.strip()}" for i, l in enumerate(lines, start=1) if l.lstrip().startswith("#")]
        if headings:
            out.append("章节目录（可用 line_number 读取对应位置）：")
            out.extend(f"  {h}" for h in headings[:60])
        return "\n".join(out)

    def lookup_cited_reference(self, paper_id: int, citation: str | None = None,
                               keywords: list[str] | None = None) -> str:
        meta = self.paper(int(paper_id))
        if meta is None:
            return f"P{paper_id} 不存在于文献库中，请只使用工具结果中出现过的 P#。"
        pid = meta["paper_id"]
        if pid not in self._reference_cache:
            lines = self.backend.md_lines(meta) or []
            self._reference_cache[pid] = parse_reference_entries(lines)
        entries = self._reference_cache[pid]
        head = self._paper_line(meta)
        if not entries:
            return f"{head}\n未能从该文献原文中解析出参考文献表（可能缺失或格式无法识别），无法给出被引文献的书目信息。"
        query_text = citation or ", ".join(keywords or [])
        indexes = match_reference_entries(entries, citation=citation, keywords=keywords)
        if not indexes:
            return (f"{head}\n参考文献表（共 {len(entries)} 条）中未匹配到 {query_text!r}。"
                    "可换成第一作者姓氏 + 年份，或编号式文献的序号再试。")
        out = [f"{head}\n参考文献表中与 {query_text!r} 匹配的条目（逐字摘自原文）："]
        for index in indexes:
            number, text = entries[index]
            ev = self.add_evidence(("ref", pid, index), paper_id=pid, kind="reference",
                                   label=f"参考文献{f' [{number}]' if number else ''} · 引自 P{pid}",
                                   text=text, method="lookup-reference", prefix="R")
            in_library = self.backend.find_in_library(extract_doi(text), text)
            note = ""
            if in_library is not None and self.paper(in_library) is not None:
                lib_meta = self.paper(in_library)
                note = (f"\n     ⚠ 该文献已在库中：[P{in_library}] {lib_meta['short_cite']} {lib_meta['title']}。"
                        "请按库内文献处理（检索/阅读它的原文与总结），不要列为二级文献。")
            out.append(f"  {ev.eid}: {_clip(text, 700)}{note}")
        return "\n".join(out)

    def recall_evidence(self, ids: list[str]) -> str:
        """按编号取回本对话中已登记证据的原文（不重新检索、编号不变）。"""
        ids = [str(i).strip() for i in (ids or []) if str(i).strip()][:20]
        if not ids:
            return "请给出要取回的 E#/R# 编号。"
        out = []
        for eid in ids:
            ev = self.evidence.get(eid)
            if ev is None:
                out.append(f"{eid}: 本对话中不存在该编号。")
                continue
            meta = self.paper(ev.paper_id)
            head = self._paper_line(meta) if meta else f"[P{ev.paper_id}]"
            out.append(f"{head}\n  {ev.eid} ({ev.label}): {_clip(ev.text, 1500)}")
        return "\n".join(out)

    MAX_DEEP_READS_PER_TURN = 3

    def deep_read_paper(self, paper_id: int, question: str) -> str:
        meta = self.paper(int(paper_id))
        if meta is None:
            return f"P{paper_id} 不存在于文献库中，请只使用工具结果中出现过的 P#。"
        if self.deep_reader is None:
            return "当前环境未启用精读子 agent，请改用 read_paper_passages。"
        if self.deep_reads_this_turn >= self.MAX_DEEP_READS_PER_TURN:
            return f"本轮精读次数已达上限（{self.MAX_DEEP_READS_PER_TURN} 次），请改用 read_paper_passages 按关键词回读。"
        lines = self.backend.md_lines(meta)
        if not lines:
            return f"{self._paper_line(meta)}\n该文献没有可读取的本地原文，无法精读。"
        question = (question or "").strip()
        if not question:
            return "请给出具体的精读问题。"
        max_chars = int(get_config("lit_search_deep_read_max_chars") or 400000)
        fulltext, truncated = prepare_fulltext(lines, max_chars)
        self.deep_reads_this_turn += 1
        reply = self.deep_reader(fulltext, question)
        pid = meta["paper_id"]
        verified, dropped = [], 0
        for item in (reply.get("quotes") or [])[:8]:
            quote = str((item or {}).get("quote") or "")
            located = locate_quote(lines, quote)
            if located is None:
                dropped += 1
                continue
            line_no, original = located
            ev = self.add_evidence(("deep", pid, line_no, hashlib.md5(_norm_quote(quote).encode()).hexdigest()),
                                   paper_id=pid, kind="fulltext", label=f"原文 L{line_no}（精读）", text=original,
                                   method="deep-read", line_number=line_no)
            verified.append((ev, quote, str((item or {}).get("note") or "")))
        out = [self._paper_line(meta), f"精读问题：{question}" + ("（原文过长，只读了前面部分）" if truncated else "")]
        out.append(f"子 agent 结论（{'原文有直接讨论' if reply.get('found') else '原文未直接讨论'}；以下方核实过的引句为准）：\n"
                   f"  {_clip(str(reply.get('answer') or '（无）'), 900)}")
        if verified:
            out.append("经核实的原文引句：")
            for ev, quote, note in verified:
                out.append(f"  {ev.eid} (原文 L{ev.line_number}): \"{_clip(quote, 400)}\"" + (f" —— {note}" if note else ""))
        else:
            out.append("没有可核实的原文引句：上面的结论没有原文支撑，不能作为证据引用。")
        if dropped:
            out.append(f"另有 {dropped} 条引句在原文中找不到（可能被改写或编造），已丢弃。")
        return "\n".join(out)

    def dispatch(self, name: str, args: dict) -> str:
        self._touched = []
        result = self._dispatch_cached(name, args)
        self.last_touched = list(self._touched)
        return result

    def _dispatch_cached(self, name: str, args: dict) -> str:
        cache_key = json.dumps([name, args], ensure_ascii=False, sort_keys=True)
        if name != "recall_evidence" and cache_key in self.call_cache:
            earlier = self.call_cache[cache_key]
            self.calls.append({"tool": name, "args": args, "cached": True})
            where = "本轮" if earlier == self.turn_index else f"第 {earlier} 轮"
            return (f"（这个调用与{where}已执行过的完全相同，结果见上文，证据编号不变，不再重复执行。"
                    "需要某条证据的原文可用 recall_evidence；需要新结果请换检索词或检索方式。）")
        result = self._dispatch(name, args)
        if name != "recall_evidence" and not result.startswith(f"工具 {name} 执行出错"):
            self.call_cache[cache_key] = self.turn_index
        return result

    def _dispatch(self, name: str, args: dict) -> str:
        top_k = args.get("top_k")
        try:
            top_k = max(1, min(30, int(top_k))) if top_k is not None else 12
        except (TypeError, ValueError):
            top_k = 12
        try:
            if name == "search_summaries":
                result = self.search_summaries(str(args.get("query", "")), str(args.get("mode", "bm25")), top_k)
            elif name == "search_fulltext_bm25":
                result = self.search_fulltext_bm25(str(args.get("query", "")), top_k)
            elif name == "search_fulltext_semantic":
                result = self.search_fulltext_semantic(str(args.get("query", "")), top_k)
            elif name == "search_fulltext_keywords":
                result = self.search_fulltext_keywords(str(args.get("expression", "")), top_k)
            elif name == "read_paper_notes":
                result = self.read_paper_notes(int(args["paper_id"]))
            elif name == "read_paper_passages":
                line = args.get("line_number")
                result = self.read_paper_passages(int(args["paper_id"]), args.get("keywords"),
                                                  int(line) if line else None)
            elif name == "lookup_cited_reference":
                result = self.lookup_cited_reference(int(args["paper_id"]), args.get("citation"),
                                                     args.get("keywords"))
            elif name == "deep_read_paper":
                result = self.deep_read_paper(int(args["paper_id"]), str(args.get("question", "")))
            elif name == "recall_evidence":
                ids = args.get("ids") or args.get("evidence_ids") or []
                result = self.recall_evidence([ids] if isinstance(ids, str) else ids)
            else:
                result = f"未知工具 {name}。"
        except Exception as exc:
            result = f"工具 {name} 执行出错：{exc}。请调整参数或换一种检索方式。"
        self.calls.append({"tool": name, "args": args})
        return result


# ---------------------------------------------------------------------------
# Result validation (anti-hallucination, program side)
# ---------------------------------------------------------------------------

_CITE_RE = re.compile(r"\[\s*([EPR]\d+(?:\s*[,，;；]\s*[EPR]?\d+)*)\s*\]")
_RELEVANCE_ORDER = {"high": 0, "medium": 1, "low": 2}


def _sanitize_citations(text: str, toolbox: LitSearchToolbox, warnings: list[str], where: str,
                        allowed_paper: int | None = None) -> tuple[str, list[str]]:
    """校验文本里的 [E#]/[P#]：无效编号替换为 [?]，返回 (新文本, 有效 E# 列表)。"""
    cited: list[str] = []

    def _replace(match: re.Match) -> str:
        parts = [p.strip() for p in re.split(r"[,，;；]", match.group(1)) if p.strip()]
        prefix = parts[0][0]
        out = []
        for part in parts:
            if part[0] not in "EPR":
                part = prefix + part
            if part[0] in "ER":
                ev = toolbox.evidence.get(part)
                if ev is None or (allowed_paper is not None and ev.paper_id != allowed_paper):
                    warnings.append(f"{where} 引用了无效的证据编号 {part}，已标记为 [?]")
                    out.append("[?]")
                    continue
                cited.append(part)
                out.append(f"[{part}]")
            else:
                pid = int(part[1:])
                if pid not in toolbox.seen_papers:
                    warnings.append(f"{where} 提及了检索结果中不存在的文献 {part}，已标记为 [?]")
                    out.append("[?]")
                    continue
                out.append(f"[{part}]")
        return "".join(out)

    return _CITE_RE.sub(_replace, text or ""), cited


SHORT_ANSWER_SOFT_LIMIT = 300  # 字符；超过则在自检时提醒模型压缩


def _derive_short_answer(judgement: str, limit: int = 160) -> str:
    """模型没给 short_answer 时的兜底：取详细报告里第一段正文（跳过标题）。"""
    for block in re.split(r"\n\s*\n", judgement or ""):
        text = re.sub(r"^#+\s*", "", block.strip())
        text = re.sub(r"\*\*", "", text)
        if text and not block.strip().startswith("#"):
            return _clip(text.replace("\n", " "), limit)
        if text and block.strip().startswith("#") and (len(text) > 20 or re.match(r"(结论|conclusion)", text, re.I)):
            return _clip(text, limit)  # 形如 "## 结论：……" 的标题本身就是结论
    return ""


def validate_finish(args: dict, toolbox: LitSearchToolbox) -> dict:
    warnings: list[str] = []
    judgement, judgement_cites = _sanitize_citations(
        str(args.get("overall_judgement") or ""), toolbox, warnings, "详细报告")
    short_answer, short_cites = _sanitize_citations(
        str(args.get("short_answer") or ""), toolbox, warnings, "简短回答")
    if not short_answer.strip():
        short_answer = _derive_short_answer(judgement)
        if short_answer:
            warnings.append("模型未给出简短回答，已截取详细报告开头代替")
    judgement_cites = list(dict.fromkeys(short_cites + judgement_cites))
    gaps, _ = _sanitize_citations(str(args.get("gaps") or ""), toolbox, warnings, "gaps")

    candidates: list[dict] = []
    seen_candidate_ids: set[int] = set()
    raw_candidates = args.get("candidates") or []
    if not isinstance(raw_candidates, list):
        raw_candidates = []
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            continue
        try:
            paper_id = int(str(raw.get("paper_id")).lstrip("Pp"))
        except (TypeError, ValueError):
            warnings.append(f"候选文献编号无法识别：{raw.get('paper_id')!r}，已剔除")
            continue
        if paper_id in seen_candidate_ids:
            continue
        meta = toolbox.paper(paper_id) if paper_id in toolbox.seen_papers else None
        if meta is None:
            warnings.append(f"候选文献 P{paper_id} 未出现在本轮任何检索结果中，已剔除（疑似编造）")
            continue
        seen_candidate_ids.add(paper_id)

        relevance = str(raw.get("relevance") or "low").lower()
        if relevance not in _RELEVANCE_ORDER:
            relevance = "low"
        reason, reason_cites = _sanitize_citations(str(raw.get("reason") or ""), toolbox, warnings,
                                                   f"P{paper_id} 的理由", allowed_paper=paper_id)
        evidence_ids: list[str] = []
        raw_ids = raw.get("evidence_ids") or []
        if isinstance(raw_ids, str):
            raw_ids = re.findall(r"E\d+", raw_ids)
        for eid in [str(e).strip() for e in raw_ids] + reason_cites:
            ev = toolbox.evidence.get(eid)
            if ev is None:
                warnings.append(f"P{paper_id} 的证据 {eid} 不存在，已剔除")
                continue
            if ev.paper_id != paper_id or ev.kind == "reference":
                warnings.append(f"证据 {eid} 不属于 P{paper_id}（或是参考文献条目），已剔除")
                continue
            if eid not in evidence_ids:
                evidence_ids.append(eid)
        auto_evidence = False
        if not evidence_ids:
            # 模型没给出有效证据：自动附上检索得分最高的片段，并降级为低相关、显式标注
            own = sorted((ev for ev in toolbox.evidence.values() if ev.paper_id == paper_id),
                         key=lambda ev: (ev.kind != "fulltext", -(ev.score or 0)))
            evidence_ids = [ev.eid for ev in own[:2]]
            auto_evidence = True
            relevance = "low"
            warnings.append(f"P{paper_id} 未给出有效证据，已自动附上检索命中片段并降为低相关")
        # 原文片段优先展示
        evidence_ids.sort(key=lambda eid: toolbox.evidence[eid].kind != "fulltext")
        candidates.append({
            "paper": meta,
            "relevance": relevance,
            "reason": reason,
            "evidence_ids": evidence_ids[:5],
            "auto_evidence": auto_evidence,
            "summary_only": all(toolbox.evidence[e].kind != "fulltext" for e in evidence_ids),
        })
    candidates.sort(key=lambda c: _RELEVANCE_ORDER[c["relevance"]])

    secondary = _validate_secondary(args.get("secondary_references"), toolbox, warnings,
                                    {c["paper"]["paper_id"] for c in candidates})

    # 页面上实际展示的证据：候选文献的片段 + 二级文献的条目与引用语境
    displayed: set[str] = set()
    for candidate in candidates:
        displayed.update(candidate["evidence_ids"])
    for ref in secondary:
        displayed.update(ref["ref_ids"])
        displayed.update(ref["context_evidence_ids"])
    # 判断文本引用了、但没有在任何卡片中展示的片段，单独列出，保证每个角标都能跳转
    orphan_ids = [eid for eid in judgement_cites if eid not in displayed]
    referenced = displayed | set(judgement_cites)
    papers = {pid: toolbox.paper(pid) for pid in toolbox.seen_papers}

    confidence = str(args.get("confidence") or "low").lower()
    if confidence not in _RELEVANCE_ORDER:
        confidence = "low"
    return {
        "short_answer": short_answer.strip(),
        "overall_judgement": judgement.strip(),
        "confidence": confidence,
        "gaps": gaps.strip(),
        "search_summary": str(args.get("search_summary") or "").strip(),
        "candidates": candidates,
        "secondary_references": secondary,
        "evidence": {eid: toolbox.evidence[eid].to_dict() for eid in sorted(referenced, key=lambda e: (e[0], int(e[1:])))},
        "orphan_evidence_ids": list(dict.fromkeys(orphan_ids)),
        # 判断文本里 [P#] 渲染成短引用时用
        "paper_cites": {str(pid): meta["short_cite"] for pid, meta in papers.items() if meta},
        "warnings": list(dict.fromkeys(warnings)),
    }


def _reference_signature(raw_reference: str, year: str | None) -> tuple[str | None, str | None]:
    """参考文献条目的 (第一作者姓氏, 年份)，用于在正文片段中核对引用。"""
    body = _NUMBERED_REF_RE.sub("", _fold(raw_reference), count=1)
    surname = next((w for w in re.findall(r"[a-z][a-z'\-]+", body) if len(w) > 1 and w not in _CITE_STOPWORDS), None)
    match = re.search(r"\b(?:19|20)\d{2}\b", str(year or "")) or re.search(r"\b(?:19|20)\d{2}\b", body)
    return surname, match.group(0) if match else None


def _validate_secondary(raw_list: Any, toolbox: LitSearchToolbox, warnings: list[str],
                        candidate_ids: set[int]) -> list[dict]:
    """校验二级文献：R# 必须真实存在；字段必须摘自条目原文；DOI 由程序提取；已在库中的单独标注。"""
    if not isinstance(raw_list, list):
        return []
    merged: dict[str, dict] = {}
    for raw in raw_list:
        if not isinstance(raw, dict):
            continue
        rid = str(raw.get("ref_id") or "").strip()
        ref_ev = toolbox.evidence.get(rid)
        if ref_ev is None or ref_ev.kind != "reference":
            warnings.append(f"二级文献 {rid or '(无编号)'} 不是 lookup_cited_reference 返回的参考文献条目，已剔除（疑似编造）")
            continue
        fields, dropped = validate_reference_fields(ref_ev.text, raw)
        for name in dropped:
            if name == "doi":
                warnings.append(f"二级文献 {rid} 的 DOI 与参考文献原文不符，已剔除"
                                + ("，改用原文中的 DOI" if fields.get("doi") else ""))
            else:
                warnings.append(f"二级文献 {rid} 的 {name} 字段在参考文献原文中找不到，已剔除")

        contexts: list[str] = []
        other_citing: list[int] = []
        surname, ref_year = _reference_signature(ref_ev.text, fields.get("year"))
        for eid in [str(e).strip() for e in (raw.get("context_evidence_ids") or [])]:
            ev = toolbox.evidence.get(eid)
            if ev is None or ev.kind == "reference":
                warnings.append(f"二级文献 {rid} 的引用语境 {eid} 无效，已剔除")
                continue
            if ev.paper_id != ref_ev.paper_id:
                # 其他库内文献也可能引用同一篇：片段中同时出现第一作者姓氏与年份才算数
                folded = _fold(ev.text)
                if not (surname and ref_year and surname in folded and ref_year in folded):
                    warnings.append(f"引用语境 {eid}（P{ev.paper_id}）中找不到 {rid} 的第一作者与年份，已剔除")
                    continue
                if ev.paper_id not in other_citing:
                    other_citing.append(ev.paper_id)
            contexts.append(eid)
        reason, reason_cites = _sanitize_citations(str(raw.get("reason") or ""), toolbox, warnings,
                                                   f"二级文献 {rid} 的理由")
        for eid in reason_cites:
            ev = toolbox.evidence[eid]
            if eid not in contexts and ev.kind != "reference" and ev.paper_id == ref_ev.paper_id:
                contexts.append(eid)

        in_library = toolbox.backend.find_in_library(fields.get("doi"), ref_ev.text)
        library_meta = toolbox.paper(in_library) if in_library is not None else None
        relevance = str(raw.get("relevance") or "low").lower()
        if relevance not in _RELEVANCE_ORDER:
            relevance = "low"
        citing = toolbox.paper(ref_ev.paper_id)

        # 同一篇被引文献可能出现在多篇库内文献的参考文献表中：按 DOI / 标题 / 条目原文合并
        key = fields.get("doi") or _norm_alnum(fields.get("title", ""))[:80] or _norm_alnum(ref_ev.text)[:80]
        entry = merged.get(key)
        if entry is None:
            entry = merged[key] = {
                "ref_ids": [], "raw": ref_ev.text, "fields": fields, "relevance": relevance, "reason": reason,
                "cited_by": [], "context_evidence_ids": [], "in_library": library_meta,
            }
        else:
            if _RELEVANCE_ORDER[relevance] < _RELEVANCE_ORDER[entry["relevance"]]:
                entry["relevance"] = relevance
            if reason and reason not in entry["reason"]:
                entry["reason"] = f"{entry['reason']}\n\n{reason}".strip()
            for name, value in fields.items():
                entry["fields"].setdefault(name, value)
        entry["ref_ids"].append(rid)
        for citing_meta in [citing] + [toolbox.paper(pid) for pid in other_citing]:
            if citing_meta and all(c["paper_id"] != citing_meta["paper_id"] for c in entry["cited_by"]):
                entry["cited_by"].append({**citing_meta, "is_candidate": citing_meta["paper_id"] in candidate_ids})
        entry["context_evidence_ids"].extend(e for e in contexts if e not in entry["context_evidence_ids"])
        if not contexts:
            warnings.append(f"二级文献 {rid} 没有给出引用处的原文片段，相关性仅为模型判断")
        if library_meta is not None:
            warnings.append(f"二级文献 {rid} 实际已在库中（P{library_meta['paper_id']} {library_meta['short_cite']}），已标注")

    refs = list(merged.values())
    for ref in refs:
        ref["doi_url"] = f"https://doi.org/{ref['fields']['doi']}" if ref["fields"].get("doi") else None
    refs.sort(key=lambda r: (r["in_library"] is not None, _RELEVANCE_ORDER[r["relevance"]]))
    return refs


def self_check(args: dict, toolbox: LitSearchToolbox, follow_up: bool = False) -> list[str]:
    """程序化的检索充分性检查。返回问题列表，空列表表示通过。

    follow_up=True（追问，且之前的证据仍在上下文中）时，不再要求本轮重新做足检索次数。
    """
    issues: list[str] = []
    search_calls = [c for c in toolbox.calls if c["tool"] in SEARCH_TOOL_NAMES and not c.get("cached")]
    tools_used = {c["tool"] for c in search_calls}
    if not follow_up:
        if len(search_calls) < 3:
            issues.append(f"只做了 {len(search_calls)} 次检索，建议换同义词/相关术语再查。")
        if "search_summaries" not in tools_used:
            issues.append("还没有检索 AI 总结（search_summaries），可能漏掉以该主题为核心的文献。")
        if not tools_used & {"search_fulltext_bm25", "search_fulltext_semantic", "search_fulltext_keywords"}:
            issues.append("还没有检索原文（search_fulltext_bm25 / search_fulltext_semantic），可能漏掉在正文中讨论该问题的文献。")
    candidates = [c for c in (args.get("candidates") or []) if isinstance(c, dict)]
    unverified = []
    for cand in candidates:
        if str(cand.get("relevance", "")).lower() != "high":
            continue
        ids = [str(e) for e in (cand.get("evidence_ids") or [])]
        kinds = {toolbox.evidence[e].kind for e in ids if e in toolbox.evidence}
        if "fulltext" not in kinds:
            unverified.append(f"P{cand.get('paper_id')}")
    if unverified:
        issues.append(f"高相关文献 {', '.join(unverified)} 只有 AI 总结/笔记支撑，建议用 read_paper_passages 回原文核实关键结论，并引用原文 E#。")
    short = str(args.get("short_answer") or "").strip()
    if not short:
        issues.append("缺少 short_answer：请在最前面用 1-3 句简单直白地回答用户的问题（点名最关键的 1-3 篇 [P#]）。")
    elif len(short) > SHORT_ANSWER_SOFT_LIMIT:
        issues.append(f"short_answer 太长（{len(short)} 字）：请压缩到 1-3 句、约 120 字，细节放到 overall_judgement。")
    unknown = [f"P{c.get('paper_id')}" for c in candidates
               if _safe_int(c.get("paper_id")) not in toolbox.seen_papers]
    if unknown:
        issues.append(f"候选 {', '.join(unknown)} 没有出现在任何工具结果中，不允许列入。")
    return issues


def _safe_int(value: Any) -> int | None:
    try:
        return int(str(value).lstrip("Pp"))
    except (TypeError, ValueError):
        return None


_FINISH_KEYS = {"short_answer", "overall_judgement", "candidates", "secondary_references", "gaps", "confidence",
                "search_summary"}


def parse_tool_arguments(raw: str, tool_name: str = "") -> tuple[dict | None, str | None]:
    """容错解析工具参数。返回 (参数对象, 错误说明)；无法得到 JSON 对象时参数为 None，绝不静默变成 {}。

    兼容几类常见的不规范输出：字符串中的裸换行/制表符（strict=False）、被包在 ```json 代码块里、
    整个 JSON 被再编码成字符串、多套一层 {"finish": {...}} / {"arguments": {...}}。
    """
    text = (raw or "").strip()
    if not text:
        return {}, None
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if fenced:
        text = fenced.group(1)
    try:
        value: Any = json.loads(text, strict=False)
    except json.JSONDecodeError as exc:
        return None, f"JSON 解析失败：{exc.msg}（第 {exc.pos} 个字符）"
    for _ in range(2):  # 双重编码："{\"overall_judgement\": ...}"
        if isinstance(value, str):
            try:
                value = json.loads(value, strict=False)
            except json.JSONDecodeError:
                break
    if not isinstance(value, dict):
        return None, f"参数是 {type(value).__name__}，不是 JSON 对象"
    for wrapper in (tool_name, "arguments", "parameters", "input", "args"):
        inner = value.get(wrapper) if wrapper else None
        if len(value) == 1 and isinstance(inner, dict):
            value = inner
            break
        if len(value) == 1 and isinstance(inner, str):
            unwrapped, _ = parse_tool_arguments(inner, "")
            if unwrapped:
                value = unwrapped
                break
    return value, None


def _finish_has_content(args: dict) -> bool:
    if not (_FINISH_KEYS & set(args)):
        return False
    return bool(str(args.get("short_answer") or "").strip() or str(args.get("overall_judgement") or "").strip()
                or args.get("candidates")
                or args.get("secondary_references"))


def _extract_json_object(text: str) -> dict | None:
    """模型偶尔把 finish 参数写在正文里，尝试从中解析出 JSON。"""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [fenced.group(1)] if fenced else []
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start:end + 1])
    for raw in candidates:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "overall_judgement" in data:
            return data
    return None


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

EventCallback = Callable[[str, str], None]

_TOOL_LABELS = {
    "search_summaries": "检索 AI 总结",
    "search_fulltext_bm25": "原文 BM25",
    "search_fulltext_semantic": "原文语义检索",
    "search_fulltext_keywords": "原文关键词",
    "read_paper_notes": "阅读笔记",
    "read_paper_passages": "回读原文",
    "lookup_cited_reference": "查参考文献",
    "recall_evidence": "取回证据",
    "deep_read_paper": "精读全文",
}


def describe_tool_call(name: str, args: dict) -> str:
    label = _TOOL_LABELS.get(name, name)
    if name == "search_summaries":
        return f"{label}（{'布尔关键词' if args.get('mode') == 'keywords' else 'BM25'}）：{args.get('query', '')}"
    if name in {"search_fulltext_bm25", "search_fulltext_semantic"}:
        return f"{label}：{args.get('query', '')}"
    if name == "search_fulltext_keywords":
        return f"{label}：{args.get('expression', '')}"
    if name == "read_paper_notes":
        return f"{label}：P{args.get('paper_id')}"
    if name == "read_paper_passages":
        detail = args.get("keywords") or (f"L{args['line_number']}" if args.get("line_number") else "摘要与目录")
        return f"{label}：P{args.get('paper_id')} · {detail}"
    if name == "recall_evidence":
        return f"{label}：{', '.join(map(str, args.get('ids') or []))}"
    if name == "deep_read_paper":
        return f"{label}：P{args.get('paper_id')} · {args.get('question', '')}"
    if name == "lookup_cited_reference":
        return f"{label}：P{args.get('paper_id')} · {args.get('citation') or args.get('keywords') or ''}"
    return f"{label}：{json.dumps(args, ensure_ascii=False)}"


def _messages_chars(messages: list[dict]) -> int:
    total = 0
    for message in messages:
        total += len(str(message.get("content") or ""))
        for call in message.get("tool_calls") or []:
            total += len(call["function"].get("arguments") or "") + 40
    return total


def _compress_ids(eids: list[str]) -> str:
    """E101,E102,E103,E107 → E101-E103, E107"""
    by_prefix: dict[str, list[int]] = {}
    for eid in eids:
        by_prefix.setdefault(eid[0], []).append(int(eid[1:]))
    parts = []
    for prefix, numbers in by_prefix.items():
        numbers = sorted(set(numbers))
        start = prev = numbers[0]
        for n in numbers[1:] + [None]:
            if n is not None and n == prev + 1:
                prev = n
                continue
            parts.append(f"{prefix}{start}" if start == prev else f"{prefix}{start}-{prefix}{prev}")
            if n is not None:
                start = prev = n
    return ", ".join(parts)


class LitSearchAgent:
    """文献检索 agent。子类（如 chat_agent.ChatAgent）通过覆盖下面几个钩子改变提示词与"何时算答完"。"""

    system_prompt = SYSTEM_PROMPT
    force_finish_prompt = FORCE_FINISH_PROMPT
    final_round_tool_reply = "已达到最大检索轮数，该调用未执行。请立即调用 finish 提交结果。"
    format_nudge_prompt = FORMAT_NUDGE_PROMPT

    def tool_definitions(self) -> list[dict]:
        return SEARCH_TOOLS + [FINISH_TOOL]

    def _user_content(self, question: str, follow_up: bool, history: str) -> str:
        if follow_up:
            return (f"用户追问：{question.strip()}\n（之前找到的 E#/R#/P# 继续有效，可直接引用；"
                    "只在需要新信息时检索。short_answer 优先说明本轮的新发现。）")
        user_content = f"用户的检索需求：{question.strip()}"
        if history:  # 旧会话（没有保存完整记录）：退回到结论摘要
            user_content = (
                "以下是本次对话中之前几轮检索的结论（仅供理解上下文；其中的证据编号已失效，"
                "若要引用相关内容，必须在本轮重新检索或阅读后引用新的 E#。"
                "short_answer 请优先说明本轮的新发现，不要重复上一轮已经给出的结论）：\n"
                f"{history}\n\n{user_content}"
            )
        return user_content

    def _handle_text_reply(self, content: str, question: str, *, final_round: bool,
                           already_nudged: bool) -> tuple[str, dict | None]:
        """模型没有调用工具、直接输出文字时怎么办。返回 ("result", 结果) / ("nudge", None) / ("stop", None)。"""
        parsed = _extract_json_object(content)
        if parsed is not None:
            return "result", self._finalize(parsed, question)
        if not already_nudged and not final_round:
            return "nudge", None
        return "stop", None

    def _turn_digest(self, question: str, result: dict) -> str:
        cands = "; ".join(f"P{c['paper']['paper_id']}({c['relevance']}: {', '.join(c['evidence_ids'])})"
                          for c in result.get("candidates", []))
        refs = ", ".join(rid for r in result.get("secondary_references", []) for rid in r["ref_ids"])
        return (f"简短回答：{result.get('short_answer', '')}\n候选：{cands or '无'}"
                + (f"\n二级文献：{refs}" if refs else ""))

    def __init__(
        self,
        *,
        client: Any = None,
        model: str | None = None,
        toolbox: LitSearchToolbox | None = None,
        max_rounds: int | None = None,
        on_event: EventCallback | None = None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=get_config("ds_api_key"), base_url=get_config("ds_url"))
        self.client = client
        self.model = model or get_config("model_name_cheap")
        self.toolbox = toolbox or LitSearchToolbox()
        self.max_rounds = max(2, int(max_rounds or get_config("lit_search_max_rounds") or 10))
        self.on_event = on_event or (lambda kind, text: None)
        self.rounds = 0
        self.usage = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                      "total_tokens": 0, "cache_hit_tokens": 0, "lost_cache_tokens": 0}
        # 逐次调用明细：用来区分"新增内容本来就无法命中"与"本该命中却没命中"
        self.call_log: list[dict] = []
        self._prev_main_prompt = 0  # 主 agent 上一次请求的输入长度（下一次请求的前缀应能全部命中）
        self.anomalies: list[dict] = []  # 无法解析/内容为空的工具参数原文，用于事后排查
        # 跨轮次持久化的对话状态（见 load_state / export_state）
        self.messages: list[dict] = []
        self.turn_summaries: list[dict] = []
        self.last_prompt_tokens = 0
        self.context_budget = int(get_config("lit_search_context_budget_tokens") or 200000)   # 软阈值
        self.context_hard_limit = int(get_config("lit_search_context_hard_tokens") or 500000)  # 硬上限
        # tool_call_id -> {"desc", "eids", "turn", "folded"}：折叠旧工具结果时，把它换成一行存根
        self.tool_meta: dict[str, dict] = {}
        self._chars_at_last_call = 0
        self._chars_per_token = 3.0
        if self.toolbox.deep_reader is None:
            self.toolbox.deep_reader = self._deep_read

    def _emit(self, kind: str, text: str) -> None:
        try:
            self.on_event(kind, text)
        except Exception:  # pragma: no cover - UI callback must never break the agent
            pass

    def _call_llm(self, messages: list[dict], tools: list[dict]) -> Any:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            temperature=0.1,
            max_tokens=8192,
            stream=False,
            # 工具调用循环里关闭深度思考：每轮的【评估】【下一步】已经写在正文中，且更稳定
            extra_body={"thinking": {"type": "disabled"}},
        )
        self._record_usage(getattr(response, "usage", None))
        chars = _messages_chars(messages)
        if self.last_prompt_tokens > 1000:
            self._chars_per_token = max(1.0, chars / self.last_prompt_tokens)
        self._chars_at_last_call = chars
        return response.choices[0].message

    def _log_call(self, usage: Any, kind: str) -> None:
        if usage is None:
            return
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        cached = getattr(usage, "prompt_cache_hit_tokens", None)
        if cached is None:
            details = getattr(usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) if details is not None else 0
        hit = int(cached or 0)
        entry = {"n": len(self.call_log) + 1, "kind": kind, "prompt": prompt, "hit": hit, "miss": prompt - hit,
                 "completion": int(getattr(usage, "completion_tokens", 0) or 0)}
        if kind == "main":
            # 只追加、不改动历史消息时，上一次请求的全部输入都应是本次的前缀（缓存按 64 token 为单位存储）
            ideal_hit = min(prompt, self._prev_main_prompt // 64 * 64)
            entry["new_tokens"] = prompt - ideal_hit
            entry["lost"] = max(0, ideal_hit - hit)
            self.usage["lost_cache_tokens"] += entry["lost"]
            self._prev_main_prompt = prompt
        self.call_log.append(entry)

    def _deep_read(self, fulltext: str, question: str) -> dict:
        """精读子 agent：独立上下文（阅读说明 + 全文 + 问题），全文放在问题前面，同一篇再问时前缀可命中缓存。"""
        self._emit("system", f"精读子 agent 开始通读全文（约 {len(fulltext):,} 字符）……")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": DEEP_READ_SYSTEM_PROMPT},
                {"role": "user", "content": f"<paper>\n{fulltext}\n</paper>"},
                {"role": "user", "content": f"问题：{question}\n请只输出 JSON 对象。"},
            ],
            temperature=0.1,
            max_tokens=4096,
            stream=False,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
        )
        prompt_before = self.last_prompt_tokens
        self._record_usage(getattr(response, "usage", None), kind="deep_read")
        self.last_prompt_tokens = prompt_before  # 子 agent 的上下文不计入主上下文长度
        content = response.choices[0].message.content or ""
        parsed, _ = parse_tool_arguments(content)
        return parsed or {"answer": content[:900], "found": False, "quotes": []}

    def estimated_context_tokens(self) -> int:
        """估算下一次请求的输入长度：上次实测值 + 之后新增内容按实测的字符/token 比折算。"""
        # 增量可以为负：折叠旧工具结果后，估算值要随之下降
        delta = _messages_chars(self.messages) - self._chars_at_last_call
        return max(0, int(self.last_prompt_tokens + delta / self._chars_per_token))

    def _fold_tool_results(self, *, before_index: int | None = None, turns: set[int] | None = None) -> int:
        """把旧的工具结果换成一行存根（列出它产生的证据编号）。证据库里原文仍在，可用 recall_evidence 取回。"""
        folded = 0
        limit = len(self.messages) if before_index is None else before_index
        for message in self.messages[:limit]:
            meta = self.tool_meta.get(message.get("tool_call_id", "")) if message.get("role") == "tool" else None
            if not meta or meta.get("folded") or (turns is not None and meta["turn"] not in turns):
                continue
            eids = meta.get("eids") or []
            message["content"] = (
                f"[已折叠以节省上下文] {meta['desc']} → "
                + (f"证据 {_compress_ids(eids)}；原文可用 recall_evidence 取回。" if eids else "无新证据。"))
            meta["folded"] = True
            folded += 1
        return folded

    def _record_usage(self, usage: Any, kind: str = "main") -> None:
        self.usage["llm_calls"] += 1
        self._log_call(usage, kind)
        if usage is not None:
            prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion = int(getattr(usage, "completion_tokens", 0) or 0)
            # DeepSeek: prompt_cache_hit_tokens；OpenAI 兼容: prompt_tokens_details.cached_tokens
            cached = getattr(usage, "prompt_cache_hit_tokens", None)
            if cached is None:
                details = getattr(usage, "prompt_tokens_details", None)
                cached = getattr(details, "cached_tokens", 0) if details is not None else 0
            self.last_prompt_tokens = prompt
            self.usage["prompt_tokens"] += prompt
            self.usage["completion_tokens"] += completion
            self.usage["total_tokens"] += int(getattr(usage, "total_tokens", 0) or (prompt + completion))
            self.usage["cache_hit_tokens"] += int(cached or 0)
        self._emit("usage", json.dumps(self.usage))

    # -- 跨轮次状态 ---------------------------------------------------------
    def load_state(self, state: dict) -> None:
        """恢复同一对话之前的完整消息记录与证据库，使追问能命中前缀缓存、证据编号保持不变。"""
        self.messages = list(state.get("messages") or [])
        self.turn_summaries = list(state.get("turn_summaries") or [])
        self.last_prompt_tokens = int(state.get("last_prompt_tokens") or 0)
        self._prev_main_prompt = self.last_prompt_tokens
        self.tool_meta = dict(state.get("tool_meta") or {})
        self._chars_per_token = float(state.get("chars_per_token") or 3.0)
        self._chars_at_last_call = _messages_chars(self.messages) if self.messages else 0
        self.toolbox.load_state(state.get("toolbox") or {})

    def export_state(self) -> dict:
        return {
            "version": 1,
            "messages": self.messages,
            "turn_summaries": self.turn_summaries,
            "last_prompt_tokens": self.last_prompt_tokens,
            "tool_meta": self.tool_meta,
            "chars_per_token": self._chars_per_token,
            "toolbox": self.toolbox.to_state(),
        }

    def _compacted_messages(self) -> list[dict]:
        """上下文超出预算时：之前各轮只保留"问题 + 结论摘要"，工具往来全部丢弃（证据库仍保留，编号仍有效）。"""
        messages: list[dict] = [{"role": "system", "content": self.system_prompt}]
        for index, summary in enumerate(self.turn_summaries, start=1):
            digest = summary.get("digest")
            if digest is None:  # 旧格式的状态文件
                cands = "; ".join(f"P{c['paper_id']}({c['relevance']}: {', '.join(c['evidence_ids'])})"
                                  for c in summary.get("candidates", []))
                refs = ", ".join(summary.get("secondary_ref_ids", []))
                digest = (f"简短回答：{summary.get('short_answer', '')}\n候选：{cands or '无'}"
                          + (f"\n二级文献：{refs}" if refs else ""))
            messages.append({"role": "user", "content": f"（第 {index} 轮）{summary['question']}"})
            messages.append({"role": "assistant", "content": (
                f"【第 {index} 轮结论（上下文已压缩；其中的 E#/R#/P# 仍然有效，可用 recall_evidence 取回原文）】\n"
                + digest)})
        return messages

    def run(self, question: str, history: str = "") -> dict:
        turn_index = len(self.turn_summaries) + 1
        self.toolbox.new_turn(turn_index)
        follow_up = bool(self.messages)
        if follow_up and self.last_prompt_tokens > self.context_budget:
            # 第一层：折叠之前各轮的工具结果（推理与结论保留，证据可按编号取回）
            folded = self._fold_tool_results(turns=set(range(1, turn_index)))
            estimate = self.estimated_context_tokens()
            self._emit("system", f"上下文约 {self.last_prompt_tokens:,} tokens，超过软阈值 {self.context_budget:,}：已折叠之前各轮的 "
                                 f"{folded} 条工具结果（证据编号不变，可用 recall_evidence 取回），估计降到约 {estimate:,}。")
            if estimate > self.context_budget:
                # 第二层：仍然太长，之前各轮只保留问题与结论摘要
                self._emit("system", "折叠后仍超过软阈值，已把之前各轮压缩为结论摘要（证据编号不变）。")
                self.messages = self._compacted_messages()
                self.tool_meta = {}
            # 旧工具结果已不在上下文里，允许重新执行同样的检索
            self.toolbox.call_cache.clear()
            self._prev_main_prompt = 0  # 前缀已被改动，本轮第一次请求的未命中属于预期
        elif follow_up:
            self._emit("system", f"沿用本对话前 {turn_index - 1} 轮的完整检索记录（约 {self.last_prompt_tokens:,} tokens，"
                                 "可命中前缀缓存）；之前的证据编号继续有效。")
        if not self.messages:
            self.messages = [{"role": "system", "content": self.system_prompt}]
        user_content = self._user_content(question, follow_up, history)
        messages = self.messages
        messages.append({"role": "user", "content": user_content})
        self_checked = False
        format_nudged = False
        invalid_finishes = 0
        last_text = ""
        # 工具列表全程不变（换工具列表会让前缀缓存失效）；最后一轮靠提示要求提交，并多给一次宽限
        tools = self.tool_definitions()

        for round_index in range(self.max_rounds + 1):
            self.rounds = round_index + 1
            final_round = round_index >= self.max_rounds - 1
            if round_index == self.max_rounds - 1:
                messages.append({"role": "user", "content": self.force_finish_prompt})
                self._emit("system", "已达到最大检索轮数，要求模型提交结果。")
            estimate = self.estimated_context_tokens()
            if estimate > self.context_hard_limit * 0.9:
                # 只保留最近一次模型回复对应的工具结果，其余（含本轮较早的）全部折叠
                last_assistant = max((i for i, m in enumerate(messages) if m.get("role") == "assistant"),
                                     default=len(messages))
                folded = self._fold_tool_results(before_index=last_assistant)
                self._prev_main_prompt = 0
                self._emit("system", f"上下文估计约 {estimate:,} tokens，接近硬上限 {self.context_hard_limit:,}，"
                                     f"已折叠 {folded} 条较早的工具结果。")
            message = self._call_llm(messages, tools)
            content = (message.content or "").strip()
            tool_calls = list(message.tool_calls or [])
            if content:
                last_text = content
                self._emit("thought", content)

            assistant: dict[str, Any] = {"role": "assistant", "content": content}
            reasoning = getattr(message, "reasoning_content", None)
            if reasoning:
                assistant["reasoning_content"] = reasoning
            if tool_calls:
                assistant["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}}
                    for tc in tool_calls
                ]
            messages.append(assistant)

            if not tool_calls:
                action, outcome = self._handle_text_reply(content, question, final_round=final_round,
                                                          already_nudged=format_nudged)
                if action == "result":
                    return outcome  # type: ignore[return-value]
                if action == "nudge":
                    format_nudged = True
                    messages.append({"role": "user", "content": self.format_nudge_prompt})
                    continue
                break

            finish_args: dict | None = None
            finish_call_id: str | None = None
            finish_error: str | None = None
            for tc in tool_calls:
                name = tc.function.name
                raw_args = tc.function.arguments or ""
                args, parse_error = parse_tool_arguments(raw_args, name)
                if name == "finish":
                    finish_call_id = tc.id
                    if args is None or not _finish_has_content(args):
                        finish_error = parse_error or "参数中没有 overall_judgement / candidates 等任何内容"
                        self.anomalies.append({"tool": name, "error": finish_error, "raw_arguments": raw_args})
                        self._emit("warn", f"finish 参数无效（{finish_error}），已退回让模型重新提交。原始参数前 300 字：{raw_args[:300]!r}")
                    else:
                        finish_args = args
                    continue
                if final_round:
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": self.final_round_tool_reply})
                    continue
                if args is None:
                    self.anomalies.append({"tool": name, "error": parse_error, "raw_arguments": raw_args})
                    self._emit("warn", f"{name} 参数无法解析（{parse_error}）")
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": f"参数不是合法的 JSON 对象（{parse_error}），请重新调用。"})
                    continue
                self._emit("tool", describe_tool_call(name, args))
                result = self.toolbox.dispatch(name, args)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                self.tool_meta[tc.id] = {"desc": describe_tool_call(name, args), "eids": self.toolbox.last_touched,
                                         "turn": turn_index, "folded": False}

            if finish_error is not None:
                invalid_finishes += 1
                messages.append({"role": "tool", "tool_call_id": finish_call_id, "content": (
                    f"finish 未被接受：{finish_error}。请重新调用 finish，参数必须是一个 JSON 对象，"
                    "直接包含 short_answer、overall_judgement、confidence、candidates、secondary_references、gaps、"
                    "search_summary 字段（不要再套一层、不要把整个 JSON 写成字符串）。")})
                if invalid_finishes >= 3:
                    break  # 连续无效：退回到程序兜底，至少给出检索命中的文献
                continue

            if finish_args is not None:
                issues = [] if (self_checked or final_round) else self_check(finish_args, self.toolbox, follow_up)
                if not issues:
                    messages.append({"role": "tool", "tool_call_id": finish_call_id, "content": "已提交。"})
                    return self._finalize(finish_args, question)
                self_checked = True
                self._emit("system", "自检未通过，继续检索：" + "；".join(issues))
                messages.append({
                    "role": "tool",
                    "tool_call_id": finish_call_id,
                    "content": "\n".join([SELF_CHECK_HEADER] + [f"- {i}" for i in issues] + [SELF_CHECK_FOOTER]),
                })

        return self._fallback(last_text, question)

    def _remember_turn(self, question: str, result: dict) -> None:
        self.turn_summaries.append({"question": question.strip(), "digest": self._turn_digest(question, result)})

    def _finalize(self, args: dict, question: str = "") -> dict:
        result = validate_finish(args, self.toolbox)
        result["stats"] = self._stats()
        self._remember_turn(question, result)
        for warning in result["warnings"]:
            self._emit("warn", warning)
        self._emit("system", f"完成：{len(result['candidates'])} 篇候选文献。")
        return result

    def _fallback(self, last_text: str, question: str = "") -> dict:
        """模型始终没有调用 finish：按检索命中自动列出候选，并明确标注未经模型确认。"""
        hits: dict[int, list[Evidence]] = {}
        for ev in self.toolbox.evidence.values():
            hits.setdefault(ev.paper_id, []).append(ev)
        ranked = sorted(hits.items(), key=lambda item: -len(item[1]))[:8]
        args = {
            "short_answer": "模型未能提交结构化结果。以下候选按检索命中次数自动列出，未经模型判断，请自行核实。",
            "overall_judgement": last_text or "",
            "confidence": "low",
            "candidates": [
                {"paper_id": pid, "relevance": "low", "reason": "按检索命中次数自动列出（未经模型确认）。",
                 "evidence_ids": [e.eid for e in sorted(evs, key=lambda e: e.kind != "fulltext")[:2]]}
                for pid, evs in ranked
            ],
            "gaps": "模型未按要求提交结果，以上内容为程序自动整理。",
            "search_summary": "",
        }
        result = validate_finish(args, self.toolbox)
        result["warnings"].insert(0, "模型未调用 finish 提交结果，候选文献为程序按命中次数自动列出。")
        result["stats"] = self._stats()
        self._remember_turn(question, result)
        return result

    def _stats(self) -> dict:
        return {
            "rounds": self.rounds,
            "tool_calls": len(self.toolbox.calls),
            "papers_seen": len(self.toolbox.seen_papers),
            "context_tokens": self.last_prompt_tokens,
            "evidence_count": len(self.toolbox.evidence),
            "model": self.model,
            "usage": dict(self.usage),
            "calls": list(self.call_log),
            "anomalies": [{**a, "raw_arguments": a["raw_arguments"][:4000]} for a in self.anomalies],
        }


# ---------------------------------------------------------------------------
# Token usage & cost estimate
# ---------------------------------------------------------------------------

def price_table(model: str | None) -> dict | None:
    """按模型取费率（元/百万 tokens）。Pro 模型默认未配置费率，返回 None 表示无法预估。"""
    tier = "professional" if model and model == get_config("model_name_professional") else "cheap"
    rates = {kind: get_config(f"price_{tier}_{kind}") for kind in ("input_cache_hit", "input_cache_miss", "output")}
    if any(value is None for value in rates.values()):
        return None
    return {kind: float(value) for kind, value in rates.items()}


def usage_breakdown(usage: dict | None, model: str | None) -> dict:
    """把累计 usage 拆成 缓存命中输入 / 未命中输入 / 输出，并按费率估算费用。"""
    usage = usage or {}
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    hit = min(int(usage.get("cache_hit_tokens", 0) or 0), prompt)
    output = int(usage.get("completion_tokens", 0) or 0)
    parts = {"input_cache_hit": hit, "input_cache_miss": prompt - hit, "output": output}
    rates = price_table(model)
    cost = None
    if rates is not None:
        cost = {kind: parts[kind] / 1_000_000 * rates[kind] for kind in parts}
        cost["total"] = sum(cost.values())
    return {
        "tokens": parts,
        "lost_cache_tokens": int(usage.get("lost_cache_tokens", 0) or 0),
        "total_tokens": prompt + output,
        "llm_calls": int(usage.get("llm_calls", 0) or 0),
        "rates": rates,
        "cost": cost,
    }


def merge_breakdowns(items: list[dict]) -> dict:
    """会话级汇总：各轮可能用了不同模型（不同费率），所以按轮分别算费用再相加。"""
    tokens = {"input_cache_hit": 0, "input_cache_miss": 0, "output": 0}
    cost: dict | None = {"input_cache_hit": 0.0, "input_cache_miss": 0.0, "output": 0.0, "total": 0.0}
    calls = 0
    unpriced = 0
    lost = 0
    for item in items:
        calls += item["llm_calls"]
        lost += item.get("lost_cache_tokens", 0)
        for kind in tokens:
            tokens[kind] += item["tokens"][kind]
        if item["cost"] is None:
            if item["total_tokens"]:
                unpriced += 1
            continue
        for kind in cost:
            cost[kind] += item["cost"][kind]
    return {
        "tokens": tokens,
        "total_tokens": sum(tokens.values()),
        "llm_calls": calls,
        "lost_cache_tokens": lost,
        "cost": cost,
        "unpriced_turns": unpriced,
    }


# ---------------------------------------------------------------------------
# Sessions (multi-turn conversation, persisted as JSON)
# ---------------------------------------------------------------------------

_SESSION_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def safe_session_id(raw: str | None) -> str | None:
    value = (raw or "").strip().lower()
    return value if _SESSION_ID_RE.match(value) else None


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _history_digest(turns: list[dict], max_turns: int = 3) -> str:
    """把之前几轮的结论压缩成上下文：去掉 E# 编号（本轮无效），保留 P# 与书目短引用。"""
    parts = []
    for turn in [t for t in turns if t.get("result")][-max_turns:]:
        result = turn["result"]
        short = re.sub(r"\[[ER]\d+\]", "", result.get("short_answer", ""))
        judgement = re.sub(r"\[E\d+\]", "", result.get("overall_judgement", ""))
        cands = "; ".join(
            f"P{c['paper']['paper_id']} {c['paper']['short_cite']} {c['paper']['title'][:80]} ({c['relevance']})"
            for c in result.get("candidates", [])
        )
        judgement = re.sub(r"\[R\d+\]", "", judgement)
        secondary = "; ".join(
            _clip(" ".join(str(r["fields"].get(k, "")) for k in ("authors", "year", "title") if r["fields"].get(k))
                  or r["raw"], 100) + " (引自 " + ", ".join(f"P{c['paper_id']}" for c in r.get("cited_by", [])) + ")"
            for r in result.get("secondary_references", [])
        )
        parts.append(f"- 需求：{turn['question']}\n  简短回答：{short or '（无）'}\n  详细报告摘录：{_clip(judgement, 800)}"
                     f"\n  候选：{cands or '无'}"
                     + (f"\n  二级文献（库外）：{secondary}" if secondary else ""))
    return "\n".join(parts)


class LitSearchSessionStore:
    def __init__(self, directory: Path | None = None,
                 agent_factory: Callable[..., LitSearchAgent] | None = None,
                 history_fn: Callable[[list[dict]], str] | None = None) -> None:
        self.directory = directory or (_workstation / "lit_search_sessions")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.agent_factory = agent_factory or LitSearchAgent
        # 没有保存完整记录的旧会话，追问时用它把之前几轮压成摘要
        self.history_fn = history_fn or _history_digest
        self._lock = threading.RLock()
        self._sessions: dict[str, dict] = {}
        self._running: dict[str, str] = {}  # session_id -> turn_id

    def _path(self, session_id: str) -> Path:
        return self.directory / f"{session_id}.json"

    def _agent_state_path(self, session_id: str) -> Path:
        # 完整消息记录与证据库较大，单独存放，避免每条进度更新都重写
        return self.directory / f"{session_id}.agent.json"

    def _load_agent_state(self, session_id: str) -> dict | None:
        path = self._agent_state_path(session_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[lit_search] failed to load agent state {session_id}: {exc}")
            return None

    def _save_agent_state(self, session_id: str, state: dict) -> None:
        path = self._agent_state_path(session_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def new_session_id(self) -> str:
        return uuid.uuid4().hex

    def _load(self, session_id: str) -> dict:
        session = self._sessions.get(session_id)
        if session is not None:
            return session
        path = self._path(session_id)
        session = None
        if path.exists():
            try:
                session = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"[lit_search] failed to load session {session_id}: {exc}")
        if session is None:
            session = {"id": session_id, "created_at": _now(), "updated_at": _now(), "turns": []}
        for turn in session["turns"]:
            # 进程重启后，未完成的轮次不会再继续
            if turn.get("status") in {"queued", "running"} and self._running.get(session_id) != turn["id"]:
                turn["status"] = "error"
                turn["error"] = turn.get("error") or "检索被中断（服务已重启）。"
        self._sessions[session_id] = session
        return session

    def _save(self, session: dict) -> None:
        session["updated_at"] = _now()
        if not session["turns"]:
            return  # 空会话不落盘
        tmp = self._path(session["id"]).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(session, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self._path(session["id"]))

    def get(self, session_id: str) -> dict:
        with self._lock:
            session = copy.deepcopy(self._load(session_id))
        session["is_running"] = session_id in self._running
        return session

    def list_sessions(self, limit: int = 100) -> list[dict]:
        files = sorted(self.directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
        previews = []
        for path in files:
            session_id = safe_session_id(path.stem)
            if not session_id:
                continue
            with self._lock:
                session = self._load(session_id)
                turns = session.get("turns", [])
                title = turns[0]["question"] if turns else "新检索"
                previews.append({
                    "id": session_id,
                    "title": _clip(title, 28),
                    "turn_count": len(turns),
                    "updated_at": str(session.get("updated_at", ""))[:16].replace("T", " "),
                })
        return previews

    def start_turn(self, session_id: str, question: str, use_pro: bool = False) -> str:
        with self._lock:
            if session_id in self._running:
                raise RuntimeError("该会话仍有检索在进行中，请等待完成。")
            session = self._load(session_id)
            agent_state = self._load_agent_state(session_id)
            # 没有保存完整记录的旧会话，退回到结论摘要
            history = "" if agent_state else self.history_fn(session["turns"])
            turn = {
                "id": uuid.uuid4().hex[:12],
                "question": question.strip(),
                "use_pro": use_pro,
                "status": "running",
                "created_at": _now(),
                "finished_at": None,
                "trace": [],
                "result": None,
                "error": "",
            }
            session["turns"].append(turn)
            self._running[session_id] = turn["id"]
            self._save(session)

        def _on_event(kind: str, text: str) -> None:
            with self._lock:
                if kind == "usage":
                    # token 统计是累计值，直接覆盖，不进检索过程列表
                    try:
                        turn["usage"] = json.loads(text)
                    except json.JSONDecodeError:
                        return
                else:
                    turn["trace"].append({"kind": kind, "text": text, "at": time.strftime("%H:%M:%S")})
                self._save(session)

        def _worker() -> None:
            try:
                model = get_config("model_name_professional") if use_pro else get_config("model_name_cheap")
                with self._lock:
                    turn["model"] = model
                agent = self.agent_factory(model=model, on_event=_on_event)
                if agent_state and hasattr(agent, "load_state"):
                    agent.load_state(agent_state)
                result = agent.run(turn["question"], history=history)
                if hasattr(agent, "export_state"):
                    # 只在本轮成功结束后保存：出错时消息记录可能不完整，保留上一轮的状态
                    self._save_agent_state(session_id, agent.export_state())
                with self._lock:
                    turn["result"] = result
                    turn["status"] = "done"
            except Exception as exc:
                with self._lock:
                    turn["status"] = "error"
                    turn["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                with self._lock:
                    turn["finished_at"] = _now()
                    self._running.pop(session_id, None)
                    self._save(session)

        threading.Thread(target=_worker, name=f"lit-search-{turn['id']}", daemon=True).start()
        return turn["id"]


_store: LitSearchSessionStore | None = None
_store_lock = threading.Lock()


def get_store() -> LitSearchSessionStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = LitSearchSessionStore()
        return _store


def warmup_indexes() -> None:
    """服务启动时在后台预建全文 BM25 索引，避免第一次检索等待太久。"""
    def _run() -> None:
        try:
            from . import rag

            added = rag.sync_bm25_index(force=True, on_progress=print)
            if added:
                print(f"[lit_search] full-text BM25 index synced: +{added} chunks")
        except Exception as exc:
            print(f"[lit_search] BM25 index warmup failed: {exc}")

    threading.Thread(target=_run, name="lit-search-bm25-warmup", daemon=True).start()
