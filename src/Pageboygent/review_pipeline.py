"""
综述流水线：选择 Zotero Collection(s) + 主题描述 → 规划 → 逐篇精读 → 子课题综述 → 最终综述，全自动。

沿用原来手工流程（网页端规划 + 已删除的 review_sub_agent.py）的设计：
    1. 规划：把候选文献的 AI 略读总结交给模型，拆分成若干子课题，并为每个子课题量身定制
       精读提示词、笔记格式、综述提示词与综述格式（XML，字段与原 prompts.txt 一致）。
    2. 逐篇精读：每篇文献单独读全文，上下文只有"精读提示词 + 本子课题文献的略读总结 + 这篇全文"。
       笔记同时写回 Zotero（标签 Detailed Reading/<项目>/<子课题>）。
    3. 子课题综述：只看本子课题的精读笔记，按定制格式写子话题综述。
    4. 最终综述：整合各子课题综述，写完整综述。

相对原流程的改进：
    - 文献用库内编号 P# 指代，不再让模型抄 DOI；程序校验编号，书目信息（EPSL 格式）由 Zotero/数据库生成。
    - 每个项目一个文件夹，按子课题分目录存放笔记、思考过程与综述；任一步完成即落盘，可中断后继续。
    - 提示词结构对前缀缓存友好：同一子课题内的精读请求共享"系统提示 + 略读总结"前缀。
    - 用量逐阶段统计（右栏 Token 仪表），并记入全局用量账本。
"""
from __future__ import annotations

import concurrent.futures
import copy
import json
import re
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .i18n import t
from .config import get_config, research_field

_workstation = Path(get_config("workstation_dir")).expanduser()

# ---------------------------------------------------------------------------
# Prompts（规划提示词沿用原 prompts.txt 的结构，泛化了领域与主题，并改用 P# 指代文献）
# ---------------------------------------------------------------------------

PLAN_PROMPT = """
你现在是〔研究领域〕领域的专家。
你正在完成一篇关于"{topic}"的综述。下面给出了候选文献的 AI 略读笔记，每篇以 [P#] 标注编号。
你需要根据这些略读笔记进行任务拆分，以便 AI Agent 工具执行：

- 每个子任务先逐篇精读一组文献，生成精读笔记。此时子任务的上下文中仅包含任务提示词、该组文献的略读笔记和当前精读文献的全文。
- 之后子任务整合精读笔记，生成子话题综述。此时上下文中仅包含任务提示词和该组文献的精读笔记，文献全文不再出现。
- 所有子任务完成后，它们的综述报告会重新作为你的上下文，由你整合成完整综述。请确保任务拆分和提示词能帮助未来的你写出一篇可发表级别的综述。

请为每一个子任务生成：
1. 精读的重点；
2. 需要精读的文献列表（{per_topic} 篇左右；一篇文献可以出现在多个子任务中）；
3. 为该子任务量身定制的精读笔记格式；
4. 子任务完成精读后整理成子话题综述的格式，同样量身定制。
子任务数量建议 {min_topics}-{max_topics} 个，覆盖主题的主要方面，彼此之间尽量少重叠。

输出格式（XML）：
<subprojects>
<subproject>
<label>英文短标签，不超过 3 个单词（会用作文件夹名和 Zotero 标签）</label>
<subprojectsummary>子课题简介（子任务看不到，在所有子任务完成后回到你的上下文中）</subprojectsummary>
<biblist>
<bib><pid>P123</pid><author>第一作者姓</author><year>2013</year><keyword>标题中的 2-3 个关键词，用 ; 分隔</keyword></bib>
<bib>……</bib>
</biblist>
<readprompt>You are ..., you need to read the user-uploaded paper in detail and take notes. You need to focus on: ...（请写出高质量的提示词）</readprompt>
<noteformat>精读笔记格式</noteformat>
<enhancenoteprompt>简短而关键的强化提示，确保子任务按要求输出（例如：Generate output directly and make sure the format fits the requirement.）</enhancenoteprompt>
<reviewprompt>You are ..., you have read several related papers in detail and need to summarize them into a review. In this review, you need to focus on: ...</reviewprompt>
<reviewformat>子话题综述格式</reviewformat>
<enhancereviewprompt>简短而关键的强化提示（例如：cite references in author-year format; list references at the end; generate output directly.）</enhancereviewprompt>
</subproject>
</subprojects>

规则：
- 文献只能用下面给出的 [P#] 编号指代（写在 <pid> 中，形如 P123），不得使用列表之外的文献，不要输出 DOI。
- 每个 subproject 中不要出现未规定的字段。
- 直接输出 XML，不要其他文字。

候选文献的略读笔记：
{summaries}
""".strip().replace("〔研究领域〕", research_field())

NOTE_SYSTEM_SUFFIX = """

The skim-reading notes of all papers in this sub-task are provided for reference:
```markdown
{summaries}
```
The note format you are expected to use is:
```markdown
{noteformat}
```
Base the note strictly on the paper's own text. Do not invent data, references or conclusions."""

REVIEW_SYSTEM_SUFFIX = """

The expected review format is:
```markdown
{reviewformat}
```"""

REVIEW_USER = """Below are the detailed reading notes of every paper in this sub-topic.

{notes}

Bibliography (the ONLY works you may cite; use exactly these bibliographic details, never invent references):
{bibliography}

{enhance}
Generate the review directly."""

FINAL_SYSTEM = """
你是〔研究领域〕领域的专家，正在完成一篇关于"{topic}"的综述。
此前你把工作拆分成了若干子课题，由子任务逐篇精读文献并写出子话题综述。现在请整合这些子话题综述，写出一篇可发表级别的完整综述：
- 有清晰的整体结构（引言、按逻辑组织的主体章节、讨论/悬而未决的问题与展望、结论），而不是子综述的简单拼接；
- 梳理不同研究之间的联系、共识与分歧，指出证据强弱；
- 在正文中以 author-year 格式引用，文末按 EPSL 格式列出参考文献；
- 只能引用下面给出的参考文献列表中的文献，书目信息以列表为准，不得编造；子综述中转引的库外文献，如需提及须说明是转引；
- 语言与子话题综述保持一致（若子综述为英文则用英文写作）。
直接输出综述正文（Markdown）。
""".strip().replace("〔研究领域〕", research_field())

FINAL_USER = """# 子课题规划
{plan}

# 子话题综述
{reviews}

# 参考文献列表（只能引用这些）
{bibliography}
"""

STAGES = ["collect", "plan", "notes", "reviews", "final"]
STAGE_LABELS = {"collect": "收集文献", "plan": "规划", "notes": "逐篇精读", "reviews": "子课题综述", "final": "最终综述"}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def slugify(text: str, fallback: str = "topic") -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", text or "").strip("-")
    return slug[:40] or fallback


# ---------------------------------------------------------------------------
# Plan parsing / validation
# ---------------------------------------------------------------------------

PLAN_FIELDS = ["label", "subprojectsummary", "readprompt", "noteformat", "enhancenoteprompt",
               "reviewprompt", "reviewformat", "enhancereviewprompt"]


def _escape_amp(xml_text: str) -> str:
    return re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9A-Fa-f]+;)", "&amp;", xml_text)


def parse_plan(text: str, papers: dict[int, dict]) -> tuple[list[dict], list[str]]:
    """解析规划 XML。返回 (子课题列表, 警告)。编号不在候选集中的文献会尝试按 作者+年份 纠正，否则剔除。"""
    warnings: list[str] = []
    match = re.search(r"<subprojects>.*</subprojects>", text or "", re.S)
    if not match:
        raise ValueError(t("规划输出中找不到 <subprojects> … </subprojects>"))
    root = ET.fromstring(_escape_amp(match.group(0)))
    by_author_year: dict[tuple[str, str], list[int]] = {}
    for pid, meta in papers.items():
        key = ((meta.get("first_author") or "").strip().lower(), str(meta.get("year") or ""))
        by_author_year.setdefault(key, []).append(pid)
    subtopics: list[dict] = []
    used_slugs: set[str] = set()
    for index, node in enumerate(root.findall("subproject"), start=1):
        item = {field: (node.findtext(field) or "").strip() for field in PLAN_FIELDS}
        # 兼容旧提示词里的拼写（noteformt / enhancedprompt）
        item["noteformat"] = item["noteformat"] or (node.findtext("noteformt") or "").strip()
        item["enhancenoteprompt"] = item["enhancenoteprompt"] or (node.findtext("enhancedprompt") or "").strip()
        label = item["label"] or f"Topic {index}"
        pids: list[int] = []
        for bib in node.findall("biblist/bib"):
            raw = (bib.findtext("pid") or "").strip()
            pid = int(re.sub(r"\D", "", raw)) if re.search(r"\d", raw) else None
            if pid not in papers:
                author = (bib.findtext("author") or "").strip().lower()
                year = (bib.findtext("year") or "").strip()
                candidates = by_author_year.get((author, year), [])
                if len(candidates) == 1:
                    warnings.append(t("review:warn.id_fixed", label=label, raw=raw or t("（空）"), author=author, year=year, pid=candidates[0]))
                    pid = candidates[0]
                else:
                    warnings.append(t("review:warn.not_candidate", label=label, raw=raw or t("（空）"), author=author, year=year))
                    continue
            if pid not in pids:
                pids.append(pid)
        if not pids:
            warnings.append(t("review:warn.no_papers", label=label))
            continue
        slug = slugify(label, f"topic-{index}")
        while slug in used_slugs:
            slug += "-x"
        used_slugs.add(slug)
        subtopics.append({**item, "label": label, "index": len(subtopics) + 1, "slug": slug,
                          "dir": f"{len(subtopics) + 1:02d}-{slug}", "pids": pids})
    if not subtopics:
        raise ValueError(t("规划中没有任何有效的子课题"))
    return subtopics, warnings


_AUTHOR_YEAR_RE = re.compile(
    r"\b([A-Z][A-Za-zÀ-ÿ'’\-]+)(?:\s+et\s+al\.?|\s+(?:and|&)\s+[A-Z][A-Za-zÀ-ÿ'’\-]+)?\s*[,(（]\s*((?:19|20)\d{2})[a-z]?\s*[)）]?")


def uncited_mentions(text: str, papers: dict[int, dict]) -> list[str]:
    """综述中出现、但不在本项目文献列表中的作者-年份引用（可能是转引，也可能是编造）。"""
    from .lit_search import _fold

    known = {(_fold(str(m.get("first_author") or "")), str(m.get("year") or "")) for m in papers.values()}
    found: list[str] = []
    for match in _AUTHOR_YEAR_RE.finditer(text or ""):
        name, year = _fold(match.group(1)), match.group(2)
        if (name, year) in known or match.group(1) in {"In", "See", "Table", "Fig", "Figure", "Eq", "Since", "From"}:
            continue
        mention = match.group(0).strip(" ,(（")
        if mention not in found:
            found.append(mention)
    return found


# ---------------------------------------------------------------------------
# Data access（集中在这里，测试时可整体替换）
# ---------------------------------------------------------------------------

class ReviewBackend:
    def collection_names(self) -> list[str]:
        names: list[str] = []
        try:
            from .zt_tools import zot

            names = sorted({c["data"]["name"] for c in zot.everything(zot.collections())})
        except Exception as exc:
            print(f"[review] cannot list Zotero collections: {exc}")
        try:
            from .storage import Theme, TracedCollection

            known = {c.name for c in TracedCollection.select()}
            for theme in Theme.select():
                known.update(n.strip() for n in str(theme.collection_names).split(";") if n.strip())
            names = sorted(set(names) | known)
        except Exception:
            pass
        return names

    def collection_keys(self, name: str) -> list[str]:
        from .zt_tools import fetch_keys_by_collection

        return fetch_keys_by_collection(name)

    def papers_by_keys(self, keys: list[str]) -> dict[str, dict]:
        from .storage import MDStatus, Paper, PDFStatus

        out: dict[str, dict] = {}
        rows = list(Paper.select().where(Paper.key_in_zotero.in_(keys))) if keys else []
        pdf = {r.paper_id: r for r in PDFStatus.select().where(PDFStatus.paper.in_([p.id for p in rows]))}
        mds = {r.paper_id: r for r in MDStatus.select().where(MDStatus.paper.in_([p.id for p in rows]))}
        for paper in rows:
            pdf_key = pdf[paper.id].pdf_key_in_zotero if paper.id in pdf else None
            md_path = _workstation / "out_mds" / f"{pdf_key}.md" if pdf_key else None
            has_md = bool(paper.id in mds and mds[paper.id].has_md_locally and md_path and md_path.exists())
            out[str(paper.key_in_zotero)] = {
                "pid": int(paper.id), "key": str(paper.key_in_zotero), "title": str(paper.title or ""),
                "first_author": paper.first_author, "year": paper.year, "doi": paper.doi,
                "summary": paper.quick_summary or "", "abstract": paper.abstract or "",
                "md_path": str(md_path) if has_md else None,
            }
        return out

    def bibliography(self, key: str) -> str | None:
        try:
            from .zt_tools import bib_export

            text = bib_export(key)
            return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text or "")).strip() or None
        except Exception:
            return None

    def read_fulltext(self, md_path: str) -> str:
        return Path(md_path).read_text(encoding="utf-8")

    def save_note_to_zotero(self, key: str, content: str, tags: list[str]) -> None:
        """写回 Zotero 笔记：经由 write_queue 交给 heartbeat 线程执行，避免和后台写库争锁。"""
        from . import write_queue
        from .note_storage import note_storage

        write_queue.submit(f"save review note for {key}",
                           lambda: note_storage.save_note(key, content, tags=tags))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class PipelineCancelled(Exception):
    pass


class ReviewRunner:
    """执行（或续跑）一个项目。每一步完成后立即落盘；已完成的产物直接复用。"""

    def __init__(self, store: "ReviewProjectStore", project_id: str, *, client: Any = None,
                 backend: ReviewBackend | None = None) -> None:
        self.store = store
        self.project_id = project_id
        self.backend = backend or store.backend
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=get_config("ds_api_key"), base_url=get_config("ds_url"))
        self.client = client
        self.dir = store.project_dir(project_id)
        self._usage_lock = threading.Lock()
        self.max_tokens = int(get_config("review_pipeline_max_output_tokens") or 32768)
        self.concurrency = max(1, int(get_config("review_pipeline_concurrency") or 3))

    # -- helpers -------------------------------------------------------------
    def project(self) -> dict:
        return self.store.load(self.project_id)

    def update(self, fn: Callable[[dict], None]) -> dict:
        return self.store.update(self.project_id, fn)

    def log(self, kind: str, text: str) -> None:
        def _apply(p: dict) -> None:
            p["trace"].append({"at": time.strftime("%H:%M:%S"), "kind": kind, "text": text})
            p["trace"] = p["trace"][-400:]
        self.update(_apply)

    def check_cancel(self) -> None:
        if self.store.cancel_requested(self.project_id):
            raise PipelineCancelled()

    def write(self, relative: str, text: str) -> None:
        path = self.dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)

    def read(self, relative: str) -> str | None:
        path = self.dir / relative
        return path.read_text(encoding="utf-8") if path.exists() else None

    def llm(self, stage: str, messages: list[dict], *, max_tokens: int | None = None) -> tuple[str, str]:
        """一次模型调用：开启思考模式（与原手工流程一致），返回 (正文, 思考过程)，并记账。"""
        from .llm_usage import record, usage_numbers

        project = self.project()
        model = project["model"]
        response = self.client.chat.completions.create(
            model=model, messages=messages, temperature=0.2, max_tokens=max_tokens or self.max_tokens, stream=False,
            extra_body={"thinking": {"type": "enabled"}},
        )
        numbers = usage_numbers(getattr(response, "usage", None))
        record("综述流水线", model, numbers, ref=self.project_id)

        def _apply(p: dict) -> None:
            usage = p["usage"].setdefault(stage, {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                                   "total_tokens": 0, "cache_hit_tokens": 0})
            usage["llm_calls"] += 1
            usage["prompt_tokens"] += numbers["prompt_tokens"]
            usage["cache_hit_tokens"] += numbers["cache_hit_tokens"]
            usage["completion_tokens"] += numbers["completion_tokens"]
            usage["total_tokens"] += numbers["prompt_tokens"] + numbers["completion_tokens"]
            p["current_stage_key"] = stage
        with self._usage_lock:
            self.update(_apply)
        choice = response.choices[0]
        content = (choice.message.content or "").strip()
        reasoning = getattr(choice.message, "reasoning_content", None) or ""
        if getattr(choice, "finish_reason", None) == "length":
            self.log("warn", t("review:warn.truncated", stage=stage, max_tokens=max_tokens or self.max_tokens))
        return content, reasoning

    @staticmethod
    def _summary_block(meta: dict) -> str:
        summary = meta.get("summary") or meta.get("abstract") or "（无 AI 总结）"
        summary = summary.split("## Labels")[0].split("**Labels**")[0].strip()
        return f"[P{meta['pid']}] {meta.get('first_author') or 'Unknown'} ({meta.get('year') or 'n.d.'}) {meta['title']}\n{summary}"

    @staticmethod
    def _bib_line(meta: dict) -> str:
        if meta.get("ref"):
            return f"- [P{meta['pid']}] {meta['ref']}"
        return (f"- [P{meta['pid']}] {meta.get('first_author') or 'Unknown'} ({meta.get('year') or 'n.d.'}). "
                f"{meta['title']}." + (f" doi:{meta['doi']}" if meta.get("doi") else ""))

    # -- stages --------------------------------------------------------------
    def run(self) -> None:
        try:
            for stage in STAGES:
                self.check_cancel()
                self.update(lambda p, s=stage: p.update(stage=s))
                getattr(self, f"stage_{stage}")()
            self.update(lambda p: p.update(status="done", stage="done", finished_at=_now(), current_stage_key=None))
            self.log("system", t("综述流水线全部完成。"))
        except PipelineCancelled:
            self.update(lambda p: p.update(status="cancelled"))
            self.log("system", t("已停止。可随时点击“继续”从中断处接着跑，已完成的部分会直接复用。"))
        except Exception as exc:
            self.update(lambda p: p.update(status="error", error=f"{type(exc).__name__}: {exc}"))
            self.log("warn", t("review:error.run", error=f"{type(exc).__name__}: {exc}"))

    def stage_collect(self) -> None:
        project = self.project()
        if project.get("papers"):
            return
        self.log("system", t("review:log.read_collections", names=', '.join(project['collections'])))
        keys: list[str] = []
        for name in project["collections"]:
            keys.extend(self.backend.collection_keys(name))
        keys = list(dict.fromkeys(keys))
        found = self.backend.papers_by_keys(keys)
        papers: dict[str, dict] = {}
        excluded: list[dict] = []
        for key in keys:
            meta = found.get(key)
            if meta is None:
                excluded.append({"key": key, "reason": t("未被 Pageboygent 追踪（不在数据库中）")})
            elif not meta["md_path"]:
                excluded.append({"key": key, "pid": meta["pid"], "title": meta["title"], "reason": t("没有本地全文 Markdown")})
            else:
                meta["ref"] = self.backend.bibliography(key)
                papers[str(meta["pid"])] = meta
        if not papers:
            raise ValueError(t("所选 Collection 中没有可用的文献（需要已被追踪且有全文 Markdown）"))
        self.update(lambda p: p.update(papers=papers, excluded=excluded))
        self.log("system", t("review:log.collected", items=len(keys), usable=len(papers), excluded=len(excluded)))

    def stage_plan(self) -> None:
        project = self.project()
        if project.get("plan"):
            return
        papers = {int(pid): meta for pid, meta in project["papers"].items()}
        n = len(papers)
        prompt = PLAN_PROMPT.format(
            topic=project["topic"], per_topic=10,
            min_topics=max(2, min(4, n // 10 or 2)), max_topics=max(3, min(10, n // 6 or 3)),
            summaries="\n\n".join(self._summary_block(m) for m in sorted(papers.values(), key=lambda m: m["pid"])),
        )
        self.log("system", t("review:log.planning", n=n))
        content, reasoning = self.llm("plan", [{"role": "user", "content": prompt}], max_tokens=self.max_tokens)
        self.write("plan/plan.xml", content)
        if reasoning:
            self.write("plan/reasoning.md", reasoning)
        subtopics, warnings = parse_plan(content, papers)
        for warning in warnings:
            self.log("warn", warning)
        self.update(lambda p: p.update(plan={"subtopics": subtopics, "warnings": warnings, "created_at": _now()}))
        self.log("system", t("review:log.planned", topics=len(subtopics), reads=sum(len(topic['pids']) for topic in subtopics)))

    def _note_path(self, topic: dict, pid: int) -> str:
        return f"subtopics/{topic['dir']}/notes/P{pid}.md"

    def _read_one(self, topic: dict, pid: int, system: str) -> None:
        self.check_cancel()
        project = self.project()
        meta = project["papers"][str(pid)]
        stage = f"notes:{topic['label']}"
        cite = f"{meta.get('first_author') or 'Unknown'} ({meta.get('year') or 'n.d.'})"
        self.store.set_status(self.project_id, "notes", topic["dir"], str(pid), "running")
        try:
            fulltext = self.backend.read_fulltext(meta["md_path"])
            user = (f"Please read the full text of the paper below in detail: [P{pid}] {cite} {meta['title']}\n"
                    f"```markdown\n{fulltext}\n```\n{topic.get('enhancenoteprompt') or ''}\nOutput the note content directly!")
            content, reasoning = self.llm(stage, [{"role": "system", "content": system},
                                                  {"role": "user", "content": user}])
            if not content:
                raise ValueError(t("模型没有返回笔记内容"))
            header = f"<!-- P{pid} · {cite} · {meta['title']} · 子课题 {topic['label']} -->\n\n"
            self.write(self._note_path(topic, pid), header + content)
            if reasoning:
                self.write(f"subtopics/{topic['dir']}/reasoning/P{pid}.md", reasoning)
            tags = [get_config("tag_for_ai_notes"),
                    f"{get_config('tag_for_detailed_reading')}/{project['slug']}/{topic['label']}"]
            self.backend.save_note_to_zotero(meta["key"], content, tags)
            self.store.set_status(self.project_id, "notes", topic["dir"], str(pid), "done")
            self.log("tool", t("review:log.note_done", label=topic['label'], pid=pid, cite=cite))
        except PipelineCancelled:
            self.store.set_status(self.project_id, "notes", topic["dir"], str(pid), "pending")
            raise
        except Exception as exc:
            self.store.set_status(self.project_id, "notes", topic["dir"], str(pid), "error")
            self.log("warn", t("review:log.note_failed", label=topic['label'], pid=pid, cite=cite, error=exc))

    def stage_notes(self) -> None:
        project = self.project()
        papers = project["papers"]
        jobs = []
        for topic in project["plan"]["subtopics"]:
            summaries = "\n\n".join(self._summary_block(papers[str(pid)]) for pid in topic["pids"])
            system = (topic.get("readprompt") or "") + NOTE_SYSTEM_SUFFIX.format(
                summaries=summaries, noteformat=topic.get("noteformat") or "Markdown")
            for pid in topic["pids"]:
                if self.read(self._note_path(topic, pid)) is not None:
                    self.store.set_status(self.project_id, "notes", topic["dir"], str(pid), "done")
                    continue
                jobs.append((topic, pid, system))
        if not jobs:
            return
        self.log("system", t("review:log.notes_start", n=len(jobs), concurrency=self.concurrency))
        # 按子课题顺序提交：同一子课题的请求共享"系统提示 + 略读总结"前缀，能命中缓存
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = [pool.submit(self._read_one, *job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                exc = future.exception()
                if isinstance(exc, PipelineCancelled):
                    for f in futures:
                        f.cancel()
                    raise exc
        failed = sum(1 for status in self._statuses("notes").values() for s in status.values() if s == "error")
        if failed:
            self.log("warn", t("review:log.notes_failed", n=failed))

    def _statuses(self, kind: str) -> dict:
        return self.project().get("progress", {}).get(kind, {})

    def stage_reviews(self) -> None:
        project = self.project()
        papers = project["papers"]
        for topic in project["plan"]["subtopics"]:
            self.check_cancel()
            path = f"subtopics/{topic['dir']}/review.md"
            if self.read(path) is not None:
                self.store.set_status(self.project_id, "reviews", topic["dir"], None, "done")
                continue
            notes = []
            for pid in topic["pids"]:
                note = self.read(self._note_path(topic, pid))
                if note:
                    meta = papers[str(pid)]
                    notes.append(f"## [P{pid}] {meta.get('first_author') or 'Unknown'} ({meta.get('year') or 'n.d.'}) "
                                 f"{meta['title']}\n\n{note}")
            if not notes:
                self.log("warn", t("review:log.no_notes", label=topic['label']))
                self.store.set_status(self.project_id, "reviews", topic["dir"], None, "error")
                continue
            self.store.set_status(self.project_id, "reviews", topic["dir"], None, "running")
            self.log("system", t("review:log.writing_sub", label=topic['label'], n=len(notes)))
            system = (topic.get("reviewprompt") or "") + REVIEW_SYSTEM_SUFFIX.format(
                reviewformat=topic.get("reviewformat") or "Markdown")
            user = REVIEW_USER.format(notes="\n\n---\n\n".join(notes),
                                      bibliography="\n".join(self._bib_line(papers[str(pid)]) for pid in topic["pids"]),
                                      enhance=topic.get("enhancereviewprompt") or "")
            content, reasoning = self.llm(f"review:{topic['label']}", [{"role": "system", "content": system},
                                                                      {"role": "user", "content": user}])
            self.write(path, content)
            if reasoning:
                self.write(f"subtopics/{topic['dir']}/reasoning/review.md", reasoning)
            for mention in uncited_mentions(content, {int(k): v for k, v in papers.items()}):
                self.log("warn", t("review:log.sub_mention", label=topic['label'], mention=mention))
            self.store.set_status(self.project_id, "reviews", topic["dir"], None, "done")

    def stage_final(self) -> None:
        if self.read("final/review.md") is not None:
            return
        project = self.project()
        papers = project["papers"]
        topics = project["plan"]["subtopics"]
        reviews = []
        for topic in topics:
            review = self.read(f"subtopics/{topic['dir']}/review.md")
            if review:
                reviews.append(f"## 子课题 {topic['index']}：{topic['label']}\n\n{review}")
        if not reviews:
            raise ValueError(t("没有任何子课题综述，无法撰写最终综述"))
        self.log("system", t("review:log.writing_final", n=len(reviews)))
        used = sorted({pid for t in topics for pid in t["pids"]})
        plan_text = "\n".join(f"- {t['label']}：{t.get('subprojectsummary', '')}" for t in topics)
        content, reasoning = self.llm("final", [
            {"role": "system", "content": FINAL_SYSTEM.format(topic=project["topic"])},
            {"role": "user", "content": FINAL_USER.format(
                plan=plan_text, reviews="\n\n---\n\n".join(reviews),
                bibliography="\n".join(self._bib_line(papers[str(pid)]) for pid in used))},
        ])
        self.write("final/review.md", content)
        if reasoning:
            self.write("final/reasoning.md", reasoning)
        for mention in uncited_mentions(content, {int(k): v for k, v in papers.items()}):
            self.log("warn", t("review:log.final_mention", mention=mention))
        self.update(lambda p: p.update(current_stage_key=None))


# ---------------------------------------------------------------------------
# Project store
# ---------------------------------------------------------------------------

_ID_RE = re.compile(r"^[0-9a-f]{12}$")


def safe_project_id(raw: str | None) -> str | None:
    value = (raw or "").strip().lower()
    return value if _ID_RE.match(value) else None


class ReviewProjectStore:
    """review_projects/<id>/ 下：project.json（元数据与进度）、plan/、subtopics/NN-label/{notes,reasoning}/、final/。"""

    def __init__(self, directory: Path | None = None, backend: ReviewBackend | None = None,
                 runner_factory: Callable[..., ReviewRunner] | None = None) -> None:
        self.directory = directory or (_workstation / "review_projects")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.backend = backend or ReviewBackend()
        self.runner_factory = runner_factory or ReviewRunner
        self._lock = threading.RLock()
        self._cache: dict[str, dict] = {}
        self._running: dict[str, threading.Thread] = {}
        self._cancel: set[str] = set()

    def project_dir(self, project_id: str) -> Path:
        return self.directory / project_id

    def _path(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "project.json"

    def load(self, project_id: str) -> dict:
        with self._lock:
            if project_id not in self._cache:
                project = json.loads(self._path(project_id).read_text(encoding="utf-8"))
                if project.get("status") == "running" and project_id not in self._running:
                    project["status"] = "cancelled"  # 服务重启：未完成的项目可以“继续”
                self._cache[project_id] = project
            return copy.deepcopy(self._cache[project_id])

    def _save(self, project: dict) -> None:
        project["updated_at"] = _now()
        path = self._path(project["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(project, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)

    def update(self, project_id: str, fn: Callable[[dict], None]) -> dict:
        with self._lock:
            self.load(project_id)
            project = self._cache[project_id]
            fn(project)
            self._save(project)
            return copy.deepcopy(project)

    def set_status(self, project_id: str, kind: str, topic_dir: str, pid: str | None, status: str) -> None:
        def _apply(p: dict) -> None:
            progress = p.setdefault("progress", {}).setdefault(kind, {})
            if pid is None:
                progress[topic_dir] = status
            else:
                progress.setdefault(topic_dir, {})[pid] = status
        self.update(project_id, _apply)

    def list_projects(self) -> list[dict]:
        items = []
        for path in self.directory.glob("*/project.json"):
            project_id = safe_project_id(path.parent.name)
            if not project_id:
                continue
            try:
                p = self.load(project_id)
            except (OSError, json.JSONDecodeError):
                continue
            items.append({"id": project_id, "name": p["name"], "status": self.status(project_id, p),
                          "updated_at": str(p.get("updated_at", ""))[:16].replace("T", " "),
                          "stage": p.get("stage")})
        return sorted(items, key=lambda i: i["updated_at"], reverse=True)

    def status(self, project_id: str, project: dict | None = None) -> str:
        if project_id in self._running and self._running[project_id].is_alive():
            return "running"
        project = project or self.load(project_id)
        return project.get("status", "idle")

    def create(self, *, name: str, collections: list[str], topic: str, use_pro: bool) -> str:
        project_id = uuid.uuid4().hex[:12]
        model = get_config("model_name_professional") if use_pro else get_config("model_name_cheap")
        project = {
            "id": project_id, "name": name.strip() or topic.strip()[:30], "created_at": _now(), "updated_at": _now(),
            "slug": slugify(name or topic, project_id), "collections": collections, "topic": topic.strip(),
            "use_pro": use_pro, "model": model, "status": "idle", "stage": "collect", "error": "",
            "papers": {}, "excluded": [], "plan": None, "progress": {}, "usage": {}, "current_stage_key": None,
            "trace": [],
        }
        with self._lock:
            self._cache[project_id] = project
            self._save(project)
        return project_id

    def start(self, project_id: str, *, retry_failed: bool = True) -> None:
        with self._lock:
            if self.status(project_id) == "running":
                raise RuntimeError(t("该项目正在运行中"))
            if self.load(project_id).get("imported"):
                # 历史导入的项目缺少原始上下文（提示词组装方式、模型、用量），重跑缺失部分只会产生不一致的结果和费用
                raise RuntimeError(t("这是从旧流程导入的历史记录，只供查看，不能继续运行。"))
            self._cancel.discard(project_id)

            def _prepare(p: dict) -> None:
                p.update(status="running", error="")
                if retry_failed:  # 失败的精读/综述清空状态，以便重试（已完成的产物文件会被直接复用）
                    for kind in ("notes", "reviews"):
                        for topic_dir, value in p.get("progress", {}).get(kind, {}).items():
                            if isinstance(value, dict):
                                for pid, s in list(value.items()):
                                    if s in {"error", "running"}:
                                        value[pid] = "pending"
                            elif value in {"error", "running"}:
                                p["progress"][kind][topic_dir] = "pending"
            self.update(project_id, _prepare)
            runner = self.runner_factory(self, project_id)
            thread = threading.Thread(target=runner.run, name=f"review-{project_id}", daemon=True)
            self._running[project_id] = thread
            thread.start()

    def cancel(self, project_id: str) -> None:
        self._cancel.add(project_id)

    def cancel_requested(self, project_id: str) -> bool:
        return project_id in self._cancel

    def read_document(self, project_id: str, relative: str) -> str | None:
        """只允许读取项目文件夹内的 .md/.xml 文件。"""
        base = self.project_dir(project_id).resolve()
        target = (base / relative).resolve()
        if base not in target.parents or target.suffix not in {".md", ".xml"} or not target.exists():
            return None
        return target.read_text(encoding="utf-8")


_store: ReviewProjectStore | None = None
_store_lock = threading.Lock()


def get_store() -> ReviewProjectStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = ReviewProjectStore()
        return _store


def migrate_legacy_review_notes(target: Path | None = None) -> int:
    """把旧的 review_notes/ 平铺文件按“项目/子课题”整理进 review_projects/legacy/，原文件夹保留不动。"""
    source = _workstation / "review_notes"
    target = target or (_workstation / "review_projects" / "legacy")
    if not source.exists():
        return 0
    pattern = re.compile(r"^(detailed_note|review_note|reasoning_review|report_review)_(.+?)_(.+?)(?:_([0-9a-f]{8}))?\.md$")
    names = {"detailed_note": "detailed_notes.md", "review_note": "review.md",
             "reasoning_review": "review_reasoning.md", "report_review": "report.md"}
    moved = 0
    for path in sorted(source.glob("*.md")):
        match = pattern.match(path.name)
        if not match:
            continue
        kind, agent, label, _ = match.groups()
        dest = target / slugify(agent, "agent") / slugify(label, "topic") / names[kind]
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        moved += 1
    return moved
