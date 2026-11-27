"""
AI 对话：与"AI 文献检索"共用同一套引擎（lit_search.LitSearchAgent）。

- 工具、证据编号（E#/R#/P#，整个对话共用）、完整上下文（前缀可命中缓存）、分层压缩、精读子 agent
  全部沿用文献检索的实现；
- 区别只在于：这里是自然对话。模型可以自行决定要不要检索；不调用工具、直接用文字回复就算答完，
  不需要 finish 这样的结构化提交；
- 防幻觉：回答里的 [E#]/[R#]/[P#] 同样逐个校验；另外检查回答中形如 "Wang et al. (2020)" 的作者-年份引用，
  在本对话的检索结果中找不到的给出警告（大概率来自模型记忆）。

聊天记录存放在 workstation_dir/chat_sessions/（不再写到代码库或当前目录）。
"""
from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from .i18n import t, answer_language_name
from .config import get_config, research_field
from .lit_search import (
    SEARCH_TOOLS,
    LitSearchAgent,
    LitSearchSessionStore,
    LitSearchToolbox,
    _clip,
    _fold,
    _sanitize_citations,
    _workstation,
)

CHAT_SYSTEM_PROMPT = """
你是 Pageboygent 的研究助理，服务于〔研究领域〕方向的研究者，正在和用户对话。
你可以调用工具检索用户本地追踪的文献库：每篇文献有原文全文（Markdown，已切块）、AI 预生成的快速总结，
部分还有主题总结与精读笔记。库内文献绝大多数为英文。

## 工具
- search_summaries(query, mode, sources): 检索预生成的 AI 总结与笔记（快速总结、主题总结、精读笔记；bm25 相关度 / keywords 布尔匹配），
  最快定位文献；精读笔记信息最密，适合回答方法、参数、具体结论类问题。
- search_fulltext_bm25(query) / search_fulltext_semantic(query) / search_fulltext_keywords(expression):
  原文切块的 BM25、向量语义、精确子串（区分大小写）检索。
- read_paper_notes(paper_id): 读某篇文献的 AI 总结与笔记。
- read_paper_passages(paper_id, keywords | line_number): 回原文读关键词所在段落或某行上下文。
- deep_read_paper(paper_id, question): 派子 agent 通读全文回答具体问题（较贵，关键词回读不够时再用）。
- lookup_cited_reference(paper_id, citation | keywords): 查库内文献的参考文献表（库外文献只能这样获得，编号 R#）。
- recall_evidence(ids): 取回本对话中之前已获得的证据原文。
P# 是文献编号（paper_id），E# 是证据片段编号，R# 是参考文献条目编号，在整个对话中始终有效。
较早的工具结果可能显示为"[已折叠] …→ 证据 E101-E130"，编号依然有效，需要原文时用 recall_evidence 取回。

## 什么时候检索
- 问题涉及具体研究、数据、机制、某篇文献的观点、"库里有没有"：先检索、读证据，再回答。
- 寒暄、改写润色、翻译、解释用户自己给出的文字、讨论写作安排等：直接回答，不必检索。
- 追问时，之前找到的 E#/R#/P# 可直接引用；只在需要新信息时再检索，不要重复同样的检索。
- 可以在一次回复中并行调用多个工具。检索足够后，直接用文字回答（不要调用任何"提交"工具）。

## 防幻觉铁律
1. 来自文献库的事实性内容必须紧跟 [E#] 引用（多个写成 [E3][E8]）；用 [P#] 指代库内文献，
   系统会把它渲染成"作者 (年份)"并附 Zotero 链接，所以不要自己写作者和年份。
2. 禁止凭记忆列举文献、作者、年份、期刊、DOI 或具体数据。库外文献只能用 lookup_cited_reference 得到的 [R#] 提及，
   且只能转述引用它的库内文献的说法。
3. 可以用通用的学科背景知识帮助解释概念，但要和文献证据分开，并标注"（通用知识，非库内文献）"；
   这类内容不得包含具体数值、出处或"某某研究发现"。
4. 区分证据等级：原文片段 > 主题总结/精读笔记 > AI 快速总结；冲突时以原文为准。不要夸大文献的相关性。
5. 库中找不到就直说"库中没有找到……"，并简要说明查过什么。

## 回答方式
- 先用一两句话直接回答，再按需要展开（Markdown）。长度与问题匹配，简单问题简短回答。
- 不要复述大段原文，系统会把你引用的 E# 原文附在回答下方。不要输出 JSON。
默认用〔回答语言〕回答；如果用户明确用另一种语言提问，就用用户的语言（专业术语、文献标题、检索词可保留英文原文）。
""".strip().replace("〔研究领域〕", research_field()).replace("〔回答语言〕", answer_language_name())

CHAT_FORCE_FINISH_PROMPT = (
    "已达到本轮的工具调用上限。请不要再调用工具，直接根据目前已获得的证据用文字回答；"
    "证据不足之处如实说明。"
)

# "Wang et al. (2020)" / "Wang et al., 2020" / "Wang and Li (2019)" / "Wang & Li, 2019" / "Wang (2020)"
_AUTHOR_YEAR_RE = re.compile(
    r"\b([A-Z][A-Za-zÀ-ÿ'’\-]+)(?:\s+et\s+al\.?|\s+(?:and|&)\s+[A-Z][A-Za-zÀ-ÿ'’\-]+)?\s*[,(（]\s*((?:19|20)\d{2})[a-z]?\s*[)）]?")
_NOT_SURNAMES = {"In", "See", "Table", "Fig", "Figure", "Eq", "Section", "Chapter", "Since", "From", "Before",
                 "After", "Until", "By", "As", "Of", "The", "And", "Vol", "Icarus", "Nature", "Science", "Apollo",
                 "January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                 "November", "December", "Spring", "Summer", "Autumn", "Fall", "Winter"}


def unsupported_author_year_mentions(answer: str, toolbox: LitSearchToolbox) -> list[str]:
    """回答中出现、但在本对话的检索结果（库内文献书目、证据原文）中找不到的作者-年份引用。"""
    # 先去掉系统会渲染的 [P#]/[E#]/[R#]，避免把它们旁边的文字误判
    text = re.sub(r"\[(?:[EPR]\d+|\?)\]", " ", answer or "")
    known_papers = [(_fold(str(meta.get("first_author") or "")), str(meta.get("year") or ""))
                    for meta in (toolbox.paper(pid) for pid in toolbox.seen_papers) if meta]
    evidence_texts = [_fold(ev.text) for ev in toolbox.evidence.values()]
    unsupported: list[str] = []
    for match in _AUTHOR_YEAR_RE.finditer(text):
        surname, year = match.group(1), match.group(2)
        if surname in _NOT_SURNAMES:
            continue
        name = _fold(surname)
        if any(name == author and year == paper_year for author, paper_year in known_papers):
            continue
        if any(name in ev and year in ev for ev in evidence_texts):
            continue
        mention = match.group(0).strip(" ,(（")
        if mention not in unsupported:
            unsupported.append(mention)
    return unsupported


def validate_chat_answer(text: str, toolbox: LitSearchToolbox) -> dict:
    warnings: list[str] = []
    answer, cited = _sanitize_citations(text or "", toolbox, warnings, "回答")
    paper_ids: list[int] = []
    for pid in re.findall(r"\[P(\d+)\]", answer):
        if int(pid) in toolbox.seen_papers and int(pid) not in paper_ids:
            paper_ids.append(int(pid))
    for eid in cited:
        pid = toolbox.evidence[eid].paper_id
        if pid not in paper_ids:
            paper_ids.append(pid)
    for mention in unsupported_author_year_mentions(answer, toolbox):
        warnings.append(t("chat:warn.unsupported_mention", mention=mention))
    papers = [meta for meta in (toolbox.paper(pid) for pid in paper_ids) if meta]
    return {
        "answer": answer.strip(),
        "cited_papers": papers,
        "evidence": {eid: toolbox.evidence[eid].to_dict() for eid in dict.fromkeys(cited)},
        "paper_cites": {str(meta["paper_id"]): meta["short_cite"] for meta in
                        (toolbox.paper(pid) for pid in toolbox.seen_papers) if meta},
        "warnings": list(dict.fromkeys(warnings)),
    }


class ChatAgent(LitSearchAgent):
    system_prompt = CHAT_SYSTEM_PROMPT
    force_finish_prompt = CHAT_FORCE_FINISH_PROMPT
    final_round_tool_reply = "已达到本轮的工具调用上限，该调用未执行。请直接用文字回答。"
    format_nudge_prompt = "请直接用文字回答用户；如果还需要信息，请调用检索工具。"

    def tool_definitions(self) -> list[dict]:
        return list(SEARCH_TOOLS)  # 没有 finish：直接用文字回答就算答完

    def _user_content(self, question: str, follow_up: bool, history: str) -> str:
        if history and not follow_up:
            return f"（之前的对话摘要，仅供理解上下文）\n{history}\n\n{question.strip()}"
        return question.strip()

    def _handle_text_reply(self, content: str, question: str, *, final_round: bool,
                           already_nudged: bool) -> tuple[str, dict | None]:
        if not content.strip():
            return ("stop", None) if (already_nudged or final_round) else ("nudge", None)
        return "result", self._finalize_answer(content, question)

    def _finalize_answer(self, content: str, question: str) -> dict:
        result = validate_chat_answer(content, self.toolbox)
        result["stats"] = self._stats()
        self._remember_turn(question, result)
        for warning in result["warnings"]:
            self._emit("warn", warning)
        self._emit("system", t("chat:done", evidence=len(result['evidence']), papers=len(result['cited_papers'])))
        return result

    def _fallback(self, last_text: str, question: str = "") -> dict:
        text = last_text or t("抱歉，这一轮没能整理出回答（工具调用已达上限）。可以换个问法，或让我基于已找到的证据再总结一次。")
        result = self._finalize_answer(text, question)
        result["warnings"].insert(0, t("模型未能在工具调用上限内给出完整回答，以上为最后的中间输出。"))
        return result

    def _turn_digest(self, question: str, result: dict) -> str:
        cited = ", ".join(result.get("evidence", {}).keys())
        return f"回答：{_clip(result.get('answer', ''), 1200)}" + (f"\n引用证据：{cited}" if cited else "")


def _chat_history_digest(turns: list[dict], max_turns: int = 6) -> str:
    parts = []
    for turn in [t for t in turns if t.get("result")][-max_turns:]:
        answer = re.sub(r"\[[ER]\d+\]", "", turn["result"].get("answer", ""))
        parts.append(f"用户：{turn['question']}\n助理：{_clip(answer, 800)}")
    return "\n\n".join(parts)


_chat_store: LitSearchSessionStore | None = None


def get_chat_store() -> LitSearchSessionStore:
    global _chat_store
    if _chat_store is None:
        _chat_store = LitSearchSessionStore(directory=_workstation / "chat_sessions", agent_factory=ChatAgent,
                                            history_fn=_chat_history_digest)
        migrate_legacy_histories(_chat_store.directory)
    return _chat_store


# ---------------------------------------------------------------------------
# 旧版 AI 对话（paper_query.PaperQueryChatter）的历史记录迁移
# ---------------------------------------------------------------------------

_LEGACY_QUESTION_PREFIX = "Question from the user: "


def _legacy_pairs(context: list[dict]) -> list[tuple[str, str]]:
    """从旧格式的 context 中提取 (用户问题, 最终回答)。中间的检索/分析消息（XML 形式）全部丢弃。"""
    pairs: list[tuple[str, str]] = []
    question: str | None = None
    answer = ""
    for message in context:
        role, content = message.get("role"), str(message.get("content") or "").strip()
        if role == "user" and content.startswith(_LEGACY_QUESTION_PREFIX):
            if question is not None:
                pairs.append((question, answer))
            question, answer = content[len(_LEGACY_QUESTION_PREFIX):].strip(), ""
        elif role == "assistant" and question is not None and content and not content.startswith("<"):
            answer = content  # 同一问题下取最后一条非内部消息作为最终回答
    if question is not None:
        pairs.append((question, answer))
    return pairs


def migrate_legacy_histories(target_dir: Path, source_dirs: list[Path] | None = None) -> int:
    """把旧版 history_*.json（写在启动目录/代码库里）转换成新会话，原文件移到 target_dir/legacy_history/ 备份。

    转换后的会话保留原来的问答，并附带一份 agent 状态（系统提示 + 原问答），所以可以直接接着追问。
    """
    source_dirs = source_dirs or [Path.cwd(), Path(__file__).resolve().parents[2]]
    backup_dir = target_dir / "legacy_history"
    migrated = 0
    seen: set[Path] = set()
    for directory in source_dirs:
        for path in sorted(directory.glob("history_*.json")):
            path = path.resolve()
            if path in seen:
                continue
            seen.add(path)
            session_id = path.stem.replace("history_", "", 1).lower()
            if not re.fullmatch(r"[0-9a-f]{32}", session_id):
                continue
            try:
                context = json.loads(path.read_text(encoding="utf-8")).get("context", [])
            except (OSError, json.JSONDecodeError) as exc:
                print(f"[chat] skip legacy history {path.name}: {exc}")
                continue
            target = target_dir / f"{session_id}.json"
            if not target.exists():
                when = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
                pairs = _legacy_pairs(context)
                turns = [{
                    "id": f"legacy{index:02d}{session_id[:6]}",
                    "question": question,
                    "use_pro": False,
                    "status": "done" if answer else "error",
                    "created_at": when,
                    "finished_at": when,
                    "trace": [],
                    "result": {"answer": answer, "cited_papers": [], "evidence": {}, "paper_cites": {},
                               "warnings": [], "legacy": True, "stats": {}} if answer else None,
                    "error": "" if answer else t("旧版对话中这一问没有得到回答。"),
                } for index, (question, answer) in enumerate(pairs)]
                if not turns:
                    continue
                target.write_text(json.dumps({"id": session_id, "created_at": when, "updated_at": when,
                                              "legacy": True, "turns": turns}, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
                messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
                for question, answer in pairs:
                    if answer:
                        messages += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
                state = {"version": 1, "messages": messages,
                         "turn_summaries": [{"question": q, "digest": f"回答：{_clip(a, 1200)}"} for q, a in pairs if a],
                         "last_prompt_tokens": 0, "tool_meta": {}, "toolbox": {}}
                (target_dir / f"{session_id}.agent.json").write_text(json.dumps(state, ensure_ascii=False),
                                                                     encoding="utf-8")
                # 会话列表按文件修改时间排序：保留旧记录原来的时间
                mtime = path.stat().st_mtime
                os.utime(target, (mtime, mtime))
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(backup_dir / path.name))
            migrated += 1
    if migrated:
        print(f"[chat] migrated {migrated} legacy chat history file(s) into {target_dir}")
    return migrated
