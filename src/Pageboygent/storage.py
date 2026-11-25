"""
用于管理文献元数据和状态
用peewee调用sqlite,实现增删改查等功能
表结构：
- 文献表（id, key_in_zotero, title, year, first_author, abstract, doi, quick_summary）
- 文献PDF状态表（paper_id, has_pdf_in_zotero, pdf_require_submitted, pdf_key_in_zotero, has_pdf_locally, local_pdf_path）
- 文献MD全文状态表（paper_id,  has_md_in_zotero, md_key_in_zotero, has_md_locally, pdf_2_md_submitted）
"""
from pathlib import Path
from peewee import Model, CharField, IntegerField, BooleanField, TextField, ForeignKeyField, ManyToManyField, DateField, DateTimeField
from .config import get_config
from .zt_tools import dump_md, dump_pdf, get_info, get_pdf_from_item
from .zt_tools import check_md_full_text_exits,check_literature_exists,search_literature_by_doi
from .zt_tools import fetch_keys_by_collection,add_item_to_collection,rm_item_from_collection
from .zt_tools import upload_md_to_zotero
from .zt_tools import ZoteroActionPaused
from .ai_interface import ai_summary, summary_paper_for_themes, summary_papers_for_theme, get_summary_sys_prompt
from .note_storage import note_storage, _db as _pageboygent_db
from .write_queue import checkpoint as _write_queue_checkpoint
from playhouse.migrate import SqliteMigrator, migrate
import shutil
import re
from typing import List, Dict
from whoosh.qparser import QueryParser
from whoosh.fields import Schema, TEXT
from whoosh.index import create_in, open_dir
from whoosh.query import Every
import tempfile


def _get_workstation_dir() -> Path:
    return Path(get_config("workstation_dir")).expanduser()


# 确保工作站目录存在
_workstation = _get_workstation_dir()
_workstation.mkdir(parents=True, exist_ok=True)
mineru_ouput_dir = get_config("mineru_output_dir")
mineru_ouput_dir = Path(mineru_ouput_dir).expanduser()
mineru_glob_pattern = get_config("mineru_result_glob_pattern")
ai_tag = get_config("tag_for_ai_notes")
summary_tag = get_config("tag_for_quick_summary")

# Whoosh 索引配置
_search_index_dir = _workstation / "summary_search_index"
_search_index_schema = Schema(content=TEXT(stored=True), doc_id=TEXT(stored=True))

def _get_search_index():
    """获取或创建持久化搜索索引"""
    _search_index_dir.mkdir(exist_ok=True)
    try:
        return open_dir(str(_search_index_dir))
    except Exception:
        return create_in(str(_search_index_dir), _search_index_schema)

def _update_search_index_doc(paper_id: int, quick_summary: str | None) -> None:
    """更新索引中的单条文档"""
    if not quick_summary:
        return
    try:
        ix = _get_search_index()
        writer = ix.writer()
        # 删除旧的文档（如果存在）
        writer.delete_by_term("doc_id", str(paper_id))
        # 添加新文档
        writer.add_document(content=quick_summary, doc_id=str(paper_id))
        writer.commit()
    except Exception as e:
        print(f"Warning: Failed to update search index for paper {paper_id}: {e}")

def _remove_search_index_doc(paper_id: int) -> None:
    """从索引中删除文档"""
    try:
        ix = _get_search_index()
        writer = ix.writer()
        writer.delete_by_term("doc_id", str(paper_id))
        writer.commit()
    except Exception as e:
        print(f"Warning: Failed to remove paper {paper_id} from search index: {e}")

def _sync_missing_quick_summary_docs() -> Dict[int, str]:
    """查询前同步索引：补齐缺失的 paper_id 文档，并返回当前可检索摘要映射。"""
    # 先拿到数据库里有 quick_summary 的论文
    base_query = Paper.select(Paper.id, Paper.quick_summary).where(~Paper.quick_summary.is_null())  # type: ignore
    all_papers = {paper.id: paper.quick_summary for paper in base_query}

    # 再读取索引里已有的 doc_id
    indexed_ids: set[int] = set()
    try:
        ix = _get_search_index()
        with ix.searcher() as searcher:
            for hit in searcher.search(Every(), limit=None):
                fields = hit.fields()
                doc_id_raw = fields.get("doc_id")
                if doc_id_raw is None:
                    continue
                try:
                    indexed_ids.add(int(doc_id_raw))
                except (TypeError, ValueError):
                    continue
    except Exception as e:
        print(f"Warning: Failed to read search index doc ids: {e}")
        return all_papers

    # 找出“数据库有摘要，但索引缺失”的 paper_id
    missing_ids = set(all_papers.keys()) - indexed_ids
    if missing_ids:
        try:
            ix = _get_search_index()
            writer = ix.writer()
            for paper_id in missing_ids:
                summary = all_papers.get(paper_id)
                # 分支：Paper 表中 quick_summary 为空/无效时跳过，不写入索引
                if not summary:
                    continue
                writer.add_document(content=summary, doc_id=str(paper_id))
            writer.commit()
        except Exception as e:
            print(f"Warning: Failed to sync missing quick_summary docs: {e}")

    return all_papers

# 作为"第一作者"的优先级：作者 > 其他贡献者 > 编者/系列编者/译者。
# 书的章节（bookSection）、会议论文集里，Zotero 的 creators 常把编者排在作者前面，不能直接取 creators[0]。
_SECONDARY_CREATOR_TYPES = {"editor", "seriesEditor", "bookAuthor", "translator", "reviewedAuthor", "contributor"}


def first_author_from_creators(creators) -> str | None:
    if not creators or not isinstance(creators, list):
        return creators if isinstance(creators, str) else None

    def _name(creator: dict) -> str | None:
        # 双字段（lastName/firstName）或单字段（name，如机构作者）
        last = (creator.get("lastName") or "").strip()
        if last:
            return last
        name = (creator.get("name") or "").strip()
        # 单字段里录成 "Agnor, C" 这种"姓, 名"写法时只取姓；机构名（如 "IPCC"）原样保留
        if re.match(r"^[^,]+,\s*[A-Z][a-z]?\.?(\s*[A-Z]\.?)*$", name):
            name = name.split(",")[0].strip()
        return name or None

    ranked = (
        [c for c in creators if c.get("creatorType") == "author"]
        or [c for c in creators if c.get("creatorType") not in _SECONDARY_CREATOR_TYPES]
        or creators
    )
    for creator in ranked:
        name = _name(creator)
        if name:
            return name
    return None


class BaseModel(Model):
    class Meta:
        # 与 note_storage 共用同一个 SqliteDatabase 实例（同线程同一连接）。
        # 若各开一个连接，遍历 select 游标期间 note_storage 写入会让本连接的 WAL 读快照过期，
        # 再 save() 时直接报 "database is locked"（SQLITE_BUSY_SNAPSHOT，busy_timeout 无效）。
        database = _pageboygent_db

class Paper(BaseModel):
    key_in_zotero = CharField(unique=True)
    title = CharField()
    year = IntegerField(null=True)
    first_author = CharField(null=True)
    abstract = TextField(null=True)
    doi = CharField(null=True)
    quick_summary = TextField(null=True)
    # 添加一个日期字段, 只有md_ready_date十天以后才被AI用来生成总结。
    md_ready_date = DateField(null=True)
    @classmethod
    def trace_paper(cls, key_in_zotero: str):
        # check if the paper already exists in the database
        try:
            paper = cls.get(cls.key_in_zotero == key_in_zotero)
            paper.update_from_zotero()
            return paper
        except cls.DoesNotExist:  # type: ignore[attr-defined]
            # if not, create a new entry
            if not check_literature_exists(key_in_zotero):
                raise Exception(f"Item with key {key_in_zotero} does not exist in Zotero or is not a supported literature type.")
            title, date, first_author, abstract, doi = get_info(key_in_zotero, ["title", "date", "creators", "abstractNote", "DOI"])
            year = None
            if isinstance(date, str) and len(date) >= 4 and date[:4].isdigit():
                year = int(date[:4])
            first_author = first_author_from_creators(first_author)
            paper = cls.create(
                key_in_zotero=key_in_zotero,
                title=title,
                year=year,
                first_author=first_author,
                abstract=abstract,
                doi=doi
            )
            paper.save()
            return paper
    @classmethod
    def update_all(cls):
        for paper in cls.select():
            paper.quick_update()
            _write_queue_checkpoint()
    def quick_update(self):
        # 检查doi是否全小写，如果不是，转换为小写并保存
        if self.doi and self.doi != self.doi.lower(): #type: ignore
            self.doi = self.doi.lower() #type: ignore
            self.save()
        if self.quick_summary is None:
            md_filter = self.md_status #type: ignore
            if md_filter.exists():
                md_status = md_filter.get() 
                if md_status.has_md_locally:
                    local_md = md_status.check_md_locally()
                    if local_md:
                        # 检查md_ready_date，如果为空，将其设置为今天，如果不为空，检查是否已经超过10天，如果超过10天了，才更新quick_summary
                        from datetime import date, timedelta
                        if self.md_ready_date is None:
                            self.md_ready_date = date.today()
                            self.save()
                            print(f"Set md_ready_date for paper '{self.title}' to {self.md_ready_date}. Quick summary will be generated after 10 days.")
                        elif date.today() >= self.md_ready_date + timedelta(days=10):
                            self.quick_summary = ai_summary(local_md)
                            self.save()
                            note_saved = note_storage.save_note(str(self.key_in_zotero), self.quick_summary, tags=[ai_tag, summary_tag])
                            if note_saved:
                                print(f"Stored quick summary note in local note storage for paper '{self.title}'.")
                            else:
                                print(f"Failed to store quick summary note for paper '{self.title}'.")
        if self.quick_summary is not None:
            self.save()
        # 检查是否注册了PDF状态跟踪器, 如果注册了, 同步更新PDF状态，如果没有注册，注册PDF状态跟踪器
        if not self.pdf_status.exists(): #type: ignore
            print(f"Paper '{self.title}' does not have a PDFStatus tracker. Registering one now.")
            self.register_pdf_tracer()
    def update_from_zotero(self):
        item_key = str(self.key_in_zotero)
        if not check_literature_exists(item_key):
            print(f"Item with key {item_key} no longer exists in Zotero or is not a supported literature type. Try to fetch items with the same DOI if available.")
            current_doi = str(self.doi) if self.doi else None
            if current_doi:
                candidates = search_literature_by_doi(current_doi)
                if candidates:
                    print(f"Found {len(candidates)} candidate(s) with the same DOI: {', '.join(candidates)}. Updating the paper record with the first candidate.")
                    item_key = candidates[0]
                else:
                    print(f"No candidates found with the same DOI {current_doi}. Marking the paper as missing in Zotero.")
                    return
            else:
                print(f"No DOI available for this paper. Marking the paper as missing in Zotero.")
                return
        title, date, first_author, abstract, doi = get_info(item_key, ["title", "date", "creators", "abstractNote", "DOI"])
        year = None
        if isinstance(date, str) and len(date) >= 4 and date[:4].isdigit():
            year = int(date[:4])
        first_author = first_author_from_creators(first_author)
        self.title = title
        self.year = year
        self.first_author = first_author
        self.abstract = abstract
        self.doi = doi.lower() if doi else None
        self.save()
        if self.quick_summary is None:
            self.quick_summary = self.load_summary_from_zotero_note()
        if self.quick_summary is not None:
            self.save()
        self.quick_update()
    
    @classmethod
    def auto_clean(cls,min_id:int=300):
        # 对于没有MD的文献，如果它时24小时内添加的，就删除它
        from datetime import datetime, timedelta
        threshold_time = datetime.now() - timedelta(hours=16)
        for paper in cls.select():
            md_filter = paper.md_status #type: ignore
            if not md_filter.exists():
                # 检查paper的创建时间，如果id> min_id，删除它
                if paper.id > min_id:
                    print(f"Deleting paper '{paper.title}' (key: {paper.key_in_zotero}) because it has no MD and was created within the last 16 hours.")
                    #删除PDF追踪器
                    if paper.pdf_status.exists(): #type: ignore
                        pdf_status = paper.pdf_status
                        for pdf_status in pdf_status:
                            pdf_status.delete_instance()
                    paper.delete_instance()
                # 如果有MD状态追踪器，但没有MD文件，也删除它
            elif md_filter.exists():
                #还是检查paper的创建时间，如果是16小时内创建的，删除它
                if paper.id > min_id:
                    md_status = md_filter.get()
                    if not md_status.has_md_locally:
                        print(f"Deleting paper '{paper.title}' (key: {paper.key_in_zotero}) because it has no local MD and was created within the last 16 hours.")
                        #删除PDF追踪器
                        if paper.pdf_status.exists(): #type: ignore
                            pdf_status = paper.pdf_status
                            for pdf_status in pdf_status:
                                pdf_status.delete_instance()
                        #删除MD状态追踪器
                        md_status.delete_instance()
                        paper.delete_instance()

    @classmethod
    def trace_collection(cls, collection_name: str) -> int | None:
        # 返回追踪的文献数量；Zotero 连接暂停时返回 None
        try:
            keys = fetch_keys_by_collection(collection_name)
        except ZoteroActionPaused as e:
            print(f"Paused trace_collection('{collection_name}') this round: {e}")
            return None
        print(f"Tracing {len(keys)} papers from collection '{collection_name}'...")
        print(f"Found {len(keys)} papers in collection '{collection_name}'. Updating database...")
        for key in keys:
            cls.trace_paper(key)
        return len(keys)

    def load_summary_from_zotero_note(self, filter_tag: List[str] = [ai_tag, summary_tag], filter_logic: str="and") -> str|None:
        item_key = str(self.key_in_zotero)
        note_content = note_storage.get_note(item_key, filter_tag=filter_tag, filter_logic=filter_logic, sync_when_needed=True)
        return note_content
    
    def register_pdf_tracer(self):
        pdf_status, created= PDFStatus.get_or_create(paper=self)
        if created:
            pdf_status.check_pdf_in_zotero()
        pdf_status.update_pdf_status()
        return pdf_status
    
    def get_md_text(self) -> str|None:
        md_filter = self.md_status #type: ignore
        if not md_filter.exists():
            print(f"Paper '{self.title}' does not have an MDStatus tracker.")
            return None
        md_status = md_filter.get()
        if not md_status.has_md_locally:
            print(f"Paper '{self.title}' does not have the MD full text available locally.")
            return None
        local_md = md_status.check_md_locally()
        if not local_md:
            print(f"Paper '{self.title}' is marked as having the MD full text locally, but the file was not found. Please check if the MD file exists in Zotero and if it has been downloaded.")
            return None
        with open(local_md, "r", encoding="utf-8") as f:
            return f.read()

    @classmethod
    def get_md_path_by_doi(cls, doi: str) -> str|None:
        # 注意，doi的大小写可能不一致，需要统一处理
        doi = doi.lower()
        paper = cls.select().where(cls.doi == doi).first()
        if not paper:
            print(f"No paper found with DOI {doi}.")
            return None
        md_filter = paper.md_status #type: ignore
        if not md_filter.exists():
            print(f"Paper with DOI {doi} does not have an MDStatus tracker.")
            return None
        md_status = md_filter.get()
        if not md_status.has_md_locally:
            print(f"Paper with DOI {doi} does not have the MD full text available locally.")
            return None
        local_md = md_status.check_md_locally()
        if not local_md:
            print(f"Paper with DOI {doi} is marked as having the MD full text locally, but the file was not found. Please check if the MD file exists in Zotero and if it has been downloaded.")
            return None
        return local_md
    def bib_info(self) -> str:
        return f"{self.first_author} + {self.year}, {self.title}, doi: {self.doi}"
    def bib_full_info(self) -> str:
        from .zt_tools import bib_export
        return bib_export(str(self.key_in_zotero))
    
    @classmethod
    def get_summary_from_note_by_doi(cls, doi: str, filter_tag: List[str] = [ai_tag, summary_tag], filter_logic: str="and") -> str|None:
        doi = doi.lower()
        paper = cls.select().where(cls.doi == doi).first()
        if not paper:
            print(f"No paper found with DOI {doi}.")
            return None
        summary_note = paper.quick_summary
        if summary_note:
            # 尝试删去**Labels**及其后的内容
            if "**Labels**" in summary_note:
                summary_note = summary_note.split("**Labels**")[0].strip()
            return "Paper: " + paper.bib_info() + "\n" + summary_note
        else:
            print(f"Paper with DOI {doi} does not have a quick summary note.")
            return None
    
    @classmethod
    def add_note_to_paper_by_doi(cls, doi: str, note_content: str, tags: List[str] = []) -> str|None:
        doi = doi.lower()
        paper = cls.select().where(cls.doi == doi).first()
        if not paper:
            print(f"No paper found with DOI {doi}.")
            return None
        item_key = str(paper.key_in_zotero)
        saved = note_storage.save_note(item_key, note_content, tags=tags)
        if saved:
            print(f"Stored note in local note storage for paper with DOI {doi}.")
            return item_key
        else:
            print(f"Failed to store note for paper with DOI {doi}.")
            return None
    
    @classmethod
    def did_doi_have_note_with_tags(cls, doi: str, filter_tag: List[str], filter_logic: str="and") -> bool:
        doi = doi.lower()
        paper = cls.select().where(cls.doi == doi).first()
        if not paper:
            print(f"No paper found with DOI {doi}.")
            return False
        item_key = str(paper.key_in_zotero)
        return note_storage.has_note(item_key, filter_tag=filter_tag, filter_logic=filter_logic, sync_when_needed=True)

class Theme(BaseModel):
    name = CharField(unique=True)
    collection_names = TextField() # 存储与该主题相关的Zotero文献库集合名称，多个集合名称用分号分隔
    summary_prompt = TextField() # 存储该主题的总结提示词
    note_tag = CharField(default="ai_summary") # 存储与该主题相关的Zotero笔记标签，用于区分不同主题的总结笔记

    def update_theme_papers(self):
        collection_names = [name.strip() for name in str(self.collection_names).split(";") if name.strip()]
        all_keys = set()
        for collection_name in collection_names:
            try:
                keys = fetch_keys_by_collection(collection_name)
            except ZoteroActionPaused as e:
                print(
                    f"Paused update_theme_papers for theme '{self.name}' this round due to '{collection_name}': {e}"
                )
                return
            all_keys.update(keys)
        #检测keys必须不为空
        if not all_keys:
            print(f"Warning: No papers found in any of the collections for theme '{self.name}'. It may due to the connection issue with Zotero. Pause this update for now.")
            return
        #对每个key进行追踪.
        # 先提取出目前Paper表中已经存在并绑定了该主题的paper
        existing_papers = Paper.select().join(ThemePaper).where(ThemePaper.theme == self)
        # 做两个检查，1，对于已经存在的paper, 如果它的key不在all_keys中，说明这个paper不应该再绑定这个主题了；2，对于all_keys中的key，如果它没有被绑定这个主题，说明需要新增一个绑定。
        existing_keys = set(paper.key_in_zotero for paper in existing_papers)
        no_longer_relevant_keys = existing_keys - all_keys
        for paper in existing_papers:
            if paper.key_in_zotero in no_longer_relevant_keys:
                # 取消这个paper与主题的绑定
                theme_paper = ThemePaper.get((ThemePaper.theme == self) & (ThemePaper.paper == paper))
                theme_paper.delete_instance()
                print(f"Removed paper '{paper.title}' from theme '{self.name}' because it is no longer in the relevant collections.")
        new_relevant_keys = all_keys - existing_keys
        #print(all_keys)
        #print(existing_keys)
        #print(new_relevant_keys)
        # 检查新key,如果Paper表中已经存在这个key的paper了，直接绑定；如果没有，需要先创建paper记录，再绑定。
        for key in new_relevant_keys:
            try:
                paper = Paper.get(Paper.key_in_zotero == key)
            except Paper.DoesNotExist: #type: ignore[attr-defined]
                paper = Paper.trace_paper(key)
            ThemePaper.create(theme=self, paper=paper)
            print(f"Added paper '{paper.title}' to theme '{self.name}' because it is in the relevant collections.")
    
    @classmethod
    def update_all(cls):
        for theme in cls.select():
            theme.update_theme_papers()
            _write_queue_checkpoint()

    def retreive_summary(self, attach_quick_summary: bool = True) -> str|None:
        summaries = []
        for theme_paper in self.theme_papers: #type: ignore
            paper = theme_paper.paper
            note_content = theme_paper.theme_specific_summary
            if note_content:
                summaries.append(f"Paper: {paper.bib_info()}\nTheme Specific Summary: \n {note_content}\n")
            if attach_quick_summary and paper.quick_summary:
                summaries.append(f"Quick Summary: \n {paper.quick_summary}\n")
        return "\n-----\n".join(summaries) if summaries else None

class ThemePaper(BaseModel):
    theme = ForeignKeyField(Theme, backref='theme_papers')
    paper = ForeignKeyField(Paper, backref='paper_themes')
    theme_specific_summary = TextField(null=True) # 存储针对该主题的总结内容

    @classmethod
    def retreive_summary_from_theme(cls, theme_name: str, attach_quick_summary: bool = False) -> str|None:
        try:
            theme = Theme.get(Theme.name == theme_name)
        except Theme.DoesNotExist: #type: ignore[attr-defined]
            print(f"No theme found with name '{theme_name}'.")
            return None
        filter_tag = [ai_tag,theme.note_tag]
        summaries = []
        for theme_paper in theme.theme_papers: #type: ignore
            paper = theme_paper.paper
            note_content = paper.load_summary_from_zotero_note(filter_tag=filter_tag, filter_logic="and")
            if note_content:
                summaries.append(f"Paper: {paper.bib_info()}\nTheme Specific Summary: \n {note_content}\n")
            if attach_quick_summary and paper.quick_summary:
                summaries.append(f"Quick Summary: \n {paper.quick_summary}\n")
        return "\n-----\n".join(summaries) if summaries else None
    
    @classmethod
    def update_summary(cls):
        #先统计有哪些需要更新的记录
        all_theme_papers = cls.select().join(Paper).join(MDStatus).where(MDStatus.has_md_locally == True, cls.theme_specific_summary.is_null(True))
        #注意，按以下顺序执行，先找有没有一篇文章有多个待总结的summary的情况，就调用summary_paper_for_themes
        #如果没有，就调用summary_papers_for_theme来总结每篇文章，然后更新记录。
        #先按检查有没有一篇文章有多个待总结的summary的情况
        paper_id_to_theme_papers = {}
        has_summaried = set()
        for theme_paper in all_theme_papers:
            paper_id = theme_paper.paper.id
            if paper_id not in paper_id_to_theme_papers:
                paper_id_to_theme_papers[paper_id] = []
            paper_id_to_theme_papers[paper_id].append(theme_paper)
        for paper_id, theme_papers in paper_id_to_theme_papers.items():
            paper = theme_papers[0].paper
            if len(theme_papers) > 1:
                theme_names = [tp.theme.name for tp in theme_papers]
                md_path = paper.md_status.get().check_md_locally() #type: ignore
                assert Path.exists(Path(md_path)), "The MD file has not been found, although the MD status is marked as having MD locally"
                print(f"Paper '{paper.title}' is associated with multiple themes ({', '.join(theme_names)}). Generating a combined summary for these themes.")
                for tp in theme_papers:
                    tp.theme_specific_summary = summary_paper_for_themes(md_path, tp.theme.summary_prompt)
                    tp.save()
                    note_storage.save_note(str(paper.key_in_zotero), tp.theme_specific_summary, tags=[ai_tag, tp.theme.note_tag])
                    tp.save()
                    has_summaried.add(paper_id)
                    _write_queue_checkpoint()
                # 检查quick_summary是否存在，如果不存在，生成quick_summary
                if not paper.quick_summary:
                    paper.quick_summary = summary_paper_for_themes(md_path, get_summary_sys_prompt())
                    paper.save()
                    note_saved = note_storage.save_note(str(paper.key_in_zotero), paper.quick_summary, tags=[ai_tag, summary_tag])
                    paper.save()
                    if note_saved:
                        print(f"Stored quick summary note in local note storage for paper '{paper.title}'.")
                    else:
                        print(f"Failed to store quick summary note for paper '{paper.title}'.")
                    paper.save()
                    _write_queue_checkpoint()
        #整理未被总结的记录，按主题分别总结
        theme_to_theme_papers = {}
        for theme_paper in all_theme_papers:
            if theme_paper.paper.id in has_summaried:
                continue
            theme_name = theme_paper.theme.name
            if theme_name not in theme_to_theme_papers:
                theme_to_theme_papers[theme_name] = []
            theme_to_theme_papers[theme_name].append(theme_paper)
        #print(theme_to_theme_papers)
        for theme_name, theme_papers in theme_to_theme_papers.items():
            theme = theme_papers[0].theme
            print(f"Generating summaries for theme '{theme_name}' with {len(theme_papers)} papers.")
            # 检查这个主题下的每篇文章是否没有quick_summary，如果没有，提取出来，生成quick_summary
            no_quick_summary_papers = [tp for tp in theme_papers if not tp.paper.quick_summary]
            has_quick_summary_papers = [tp for tp in theme_papers if tp.paper.quick_summary]
            for tp in no_quick_summary_papers:
                paper = tp.paper
                md_path = paper.md_status.get().check_md_locally() #type: ignore
                assert Path.exists(Path(md_path)), "The MD file has not been found, although the MD status is marked as having MD locally"
                paper.quick_summary = summary_paper_for_themes(md_path, get_summary_sys_prompt())
                paper.save()
                note_saved = note_storage.save_note(str(paper.key_in_zotero), paper.quick_summary, tags=[ai_tag, summary_tag])
                paper.save()
                if note_saved:
                    print(f"Stored quick summary note in local note storage for paper '{paper.title}'.")
                else:
                    print(f"Failed to store quick summary note for paper '{paper.title}'.")
                tp.theme_specific_summary = summary_paper_for_themes(md_path, theme.summary_prompt)
                tp.save()
                note_storage.save_note(str(paper.key_in_zotero), tp.theme_specific_summary, tags=[ai_tag, theme.note_tag])
                _write_queue_checkpoint()
            for theme_paper in has_quick_summary_papers:
                paper = theme_paper.paper
                md_path = paper.md_status.get().check_md_locally() #type: ignore
                assert Path.exists(Path(md_path)), "The MD file has not been found, although the MD status is marked as having MD locally"
                theme_paper.theme_specific_summary = summary_papers_for_theme(md_path, theme.summary_prompt)
                theme_paper.save()
                note_storage.save_note(str(paper.key_in_zotero), theme_paper.theme_specific_summary, tags=[ai_tag, theme.note_tag])
                _write_queue_checkpoint()

    @classmethod
    def auto_fix(cls):
        #对于没有theme或paper的theme_paper记录，删除它
        for theme_paper in cls.select():
            try:
                # Trigger the DB lookup for both relations
                theme = theme_paper.theme
                paper = theme_paper.paper
            except cls.theme.rel_model.DoesNotExist:
                print(f"Deleting ThemePaper {theme_paper.id}: Associated Theme does not exist.")
                #theme_paper.delete_instance()
            except cls.paper.rel_model.DoesNotExist:
                print(f"Deleting ThemePaper {theme_paper.id}: Associated Paper does not exist.")
                theme_paper.delete_instance()
pdf_require_collection = get_config("pdf_require_collection")

class PDFStatus(BaseModel):
    paper = ForeignKeyField(Paper, backref='pdf_status', unique=True)
    has_pdf_in_zotero = BooleanField(default=False)
    pdf_require_submitted = BooleanField(default=False)
    pdf_key_in_zotero = CharField(null=True)
    has_pdf_locally = BooleanField(default=False)
    @classmethod
    def update_all(cls):
        for pdf_status in cls.select():
            pdf_status.update_pdf_status()
            _write_queue_checkpoint()
    def check_pdf_in_zotero(self):
        item_key = str(self.paper.key_in_zotero)
        try:
            pdf_key = get_pdf_from_item(item_key)
        except Exception as e:
            pdf_key = None
        if pdf_key:
            self.has_pdf_in_zotero = True
            self.pdf_key_in_zotero = pdf_key
            self.save()
            print(f"Found PDF for paper '{self.paper.title}' in Zotero with key {pdf_key}.")
        else:
            self.has_pdf_in_zotero = False
            self.pdf_key_in_zotero = None
            self.save()
            #print(f"No PDF found for paper '{self.paper.title}' in Zotero. Please check the Zotero entry.")
    def check_submited_requirement_finished(self):
        if not self.pdf_require_submitted:
            return
        self.check_pdf_in_zotero()
        if self.has_pdf_in_zotero:
            item_key = str(self.paper.key_in_zotero)
            try:
                rm_result = rm_item_from_collection(item_key, pdf_require_collection)
                if rm_result:
                    self.pdf_require_submitted = False
                    self.save()
                    print(f"PDF for paper '{self.paper.title}' has been added to Zotero. Removed from PDF requirement collection '{pdf_require_collection}'.")
            except Exception as e:
                print(f"Failed to remove paper '{self.paper.title}' from PDF requirement collection in Zotero with error: {e}. Please check if the collection name is correct and if the Zotero API key has sufficient permissions.")
                rm_result = False
            if not rm_result:
                print(f"Failed to check if the PDF requirement for paper '{self.paper.title}' is finished. Please check if the collection name is correct and if the Zotero API key has sufficient permissions.")
                self.pdf_require_submitted = True
                self.save()
    def check_pdf_locally(self) -> str|None:
        # return the local path if exists, otherwise return None
        if not self.pdf_key_in_zotero:
            return None
        pdf_key = str(self.pdf_key_in_zotero)
        local_pdf_path = _workstation / "input_pdfs" / f"{pdf_key}.pdf"
        if local_pdf_path.exists():
            print(f"-----Found local PDF for paper '{self.paper.title}' at {local_pdf_path}.")
            return str(local_pdf_path)
        local_pdf_path = _workstation / "human_upload_pdfs" / f"{pdf_key}.pdf"
        if local_pdf_path.exists():
            print(f"-----Found local PDF for paper '{self.paper.title}' at {local_pdf_path}.")
            return str(local_pdf_path)
        local_pdf_path = _workstation / "old_human_pdfs" / f"{pdf_key}.pdf"
        if local_pdf_path.exists():
            print(f"-----Found local PDF for paper '{self.paper.title}' at {local_pdf_path}.")
            return str(local_pdf_path)
        #import os
        #print(os.system(f"ls -l {_workstation / 'input_pdfs' / f'{pdf_key}.pdf'}"))
        #print(os.system(f"ls -l {_workstation / 'human_upload_pdfs' / f'{pdf_key}.pdf'}"))
        #print(os.system(f"ls -l {_workstation / 'old_human_pdfs' / f'{pdf_key}.pdf'}"))
        #raise FileNotFoundError(f"PDF file for paper '{self.paper.title}' with key {pdf_key} not found locally in any of the expected directories. Please check if the PDF file exists in Zotero and if it has been downloaded.")
        return None
    def download_pdf(self):
        if not self.pdf_key_in_zotero:
            print(f"Cannot download PDF for paper '{self.paper.title}' because no PDF key is available in Zotero.")
            return
        pdf_key = str(self.pdf_key_in_zotero)
        local_pdf_path = _workstation / "input_pdfs" / f"{pdf_key}.pdf"
        try:
            dump_pdf(pdf_key, local_pdf_path)
            self.has_pdf_locally = True
            self.save()
            print(f"Downloaded PDF for paper '{self.paper.title}' from Zotero and saved locally at {local_pdf_path}.")
        except Exception as e:
            print(f"Failed to download PDF for paper '{self.paper.title}' from Zotero with error: {e}. Please check if the PDF file exists in Zotero and if the Zotero API key has sufficient permissions.")
            self.has_pdf_locally = False
            self.save()
    def sub_item_to_pdf_require_collection(self):
        if not self.pdf_require_submitted:
            item_key = str(self.paper.key_in_zotero)
            if not check_literature_exists(item_key):
                print(f"Cannot submit paper '{self.paper.title}' to PDF requirement collection because it does not exist in Zotero or is not a supported literature type.")
                return
            try:
                add_result = add_item_to_collection(item_key, pdf_require_collection)
                if add_result:
                    self.pdf_require_submitted = True
                    self.save()
                    print(f"Submitted paper '{self.paper.title}' to PDF requirement collection '{pdf_require_collection}' in Zotero.")
            except Exception as e:
                add_result = False
            if not add_result:
                print(f"Failed to submit paper '{self.paper.title}' to PDF requirement collection in Zotero. Please check if the collection name is correct and if the Zotero API key has sufficient permissions.")
                self.pdf_require_submitted = False
                self.save()
    def update_pdf_status(self):
        has_pdf_in_zotero = bool(self.has_pdf_in_zotero)
        pdf_require_submitted = bool(self.pdf_require_submitted)
        pdf_key_in_zotero = str(self.pdf_key_in_zotero) if self.pdf_key_in_zotero else None
        has_pdf_locally = bool(self.has_pdf_locally)
        if has_pdf_locally:
            if self.paper.id == 103:
                #print(f"Paper '{self.paper.title}' has local PDF. ")
                # print(self.paper.md_status.exists())
                pass
            if not self.paper.md_status.exists():
                self.register_md_tracer()
            return
        if has_pdf_in_zotero:
            if not pdf_key_in_zotero:
                self.check_pdf_in_zotero()
                pdf_key_in_zotero = str(self.pdf_key_in_zotero) if self.pdf_key_in_zotero else None
                if not pdf_key_in_zotero:
                    print(f"PDF for paper '{self.paper.title}' is marked as existing in Zotero but no PDF key found. Please check the Zotero entry.")
                    return
            # if any dir contains {pdf_key}.pdf in workstation_dir/input_pdfs, workstation_dir/human_upload_pdfs, workstation_dir/old_human_pdfs, label has_pdf_locally = True.
            local_pdf = self.check_pdf_locally()
            self.has_pdf_locally = local_pdf is not None
            self.save()
            has_pdf_locally = bool(self.has_pdf_locally)
            if not has_pdf_locally:
                print(f"PDF for paper '{self.paper.title}' exists in Zotero but not found locally. Try to download it.")
                self.download_pdf()
        elif not pdf_require_submitted:
            self.sub_item_to_pdf_require_collection()
        elif pdf_require_submitted:
            self.check_submited_requirement_finished()

    def register_md_tracer(self):
        if not self.has_pdf_locally:
            #print(f"Cannot register MD tracer for paper '{self.paper.title}' because the PDF file is not available locally. Please check if the PDF file exists in Zotero and if it has been downloaded.")
            return
        md_status, _ = MDStatus.get_or_create(paper=self.paper)
        md_status.update_md_status()
        return md_status


class MDStatus(BaseModel):
    paper = ForeignKeyField(Paper, backref='md_status', unique=True)
    has_md_in_zotero = BooleanField(default=False)
    md_key_in_zotero = CharField(null=True)
    has_md_locally = BooleanField(default=False)
    pdf_2_md_submitted = BooleanField(default=False)
    submit_2_rag = BooleanField(default=False) # 是否已经提交到RAG系统了
    @classmethod
    def update_all(cls):
        for md_status in cls.select():
            md_status.update_md_status()
            _write_queue_checkpoint()
    def update_md_status(self):
        # 0) 判断是否可以直接跳过？
        if self.has_md_in_zotero and self.has_md_locally:
            return
        
        # 1) 先与真实状态同步
        self.check_md_in_zotero()
        local_md = self.check_md_locally()
        self.has_md_locally = local_md is not None
        self.save()

        # 2) 分支A：Zotero有 + 本地有 -> 已完成
        if self.has_md_in_zotero and self.has_md_locally:
            return

        # 3) 分支B：Zotero有 + 本地无 -> 下载
        if self.has_md_in_zotero and not self.has_md_locally:
            self.download_md()
            return

        # 4) 分支C：Zotero无 + 本地有 -> 上传
        if not self.has_md_in_zotero and self.has_md_locally:
            assert local_md is not None, "Only for code logic."
            upload_result = upload_md_to_zotero(str(self.paper.key_in_zotero), local_md)
            if upload_result:
                self.has_md_in_zotero = True
                self.md_key_in_zotero = upload_result
                self.save()
                print(f"Uploaded MD full text for paper '{self.paper.title}' to Zotero and linked to the paper entry.")
            else:
                print(f"Failed to upload MD full text for paper '{self.paper.title}' to Zotero. Please check API permissions.")
            return

        # 5) 分支D：Zotero无 + 本地无 -> 走 PDF->MD 任务流
        if not self.pdf_2_md_submitted:
            self.submit_pdf_2_md_task()
            return

    def get_pdf_key(self) -> str|None:
        pdf_status = self.paper.pdf_status.get()
        return str(pdf_status.pdf_key_in_zotero) if pdf_status else None
    
    def check_md_in_zotero(self):
        item_key = str(self.paper.key_in_zotero)
        md_key = check_md_full_text_exits(item_key)
        if md_key:
            self.has_md_in_zotero = True
            self.md_key_in_zotero = md_key
            self.save()
            #print(f"Found MD full text for paper '{self.paper.title}' in Zotero with key {md_key}.")
        else:
            self.has_md_in_zotero = False
            self.md_key_in_zotero = None
            self.save()
            #print(f"No MD full text found for paper '{self.paper.title}' in Zotero. Please check the Zotero entry.")

    def check_md_locally(self) -> str|None:
        # return the local path if exists, otherwise return None
        # Get the pdf_key_in_zotero from the related PDFStatus entry
        pdf_status = self.paper.pdf_status.get()
        if not pdf_status.pdf_key_in_zotero:
            return None
        pdf_key = str(pdf_status.pdf_key_in_zotero)
        local_md_path = _workstation / "out_mds" / f"{pdf_key}.md"
        if local_md_path.exists():
            return str(local_md_path)
        return None
    
    def download_md(self):
        pdf_key = self.get_pdf_key()
        if not pdf_key:
            print(f"Cannot download MD full text for paper '{self.paper.title}' because no PDF key is available in Zotero.")
            return
        local_md_path = _workstation / "out_mds" / f"{pdf_key}.md"
        try:
            dump_md(str(self.md_key_in_zotero), local_md_path)
            self.has_md_locally = True
            self.save()
            print(f"Downloaded MD full text for paper '{self.paper.title}' from Zotero and saved locally at {local_md_path}.")
        except Exception as e:
            print(f"Failed to download MD full text for paper '{self.paper.title}' from Zotero with error: {e}. Please check if the MD file exists in Zotero and if the Zotero API key has sufficient permissions.")
            self.has_md_locally = False
            self.save()

    def check_pdf_2_md_task_exits(self) -> bool:
        pdf_key = self.get_pdf_key()
        if not pdf_key:
            return False
        local_pdf_path = _workstation / "human_upload_pdfs" / f"{pdf_key}.pdf"
        task_exits = local_pdf_path.exists()
        if task_exits and not self.pdf_2_md_submitted:
            self.pdf_2_md_submitted = True
            self.save()
            return True
        return local_pdf_path.exists()

    def submit_pdf_2_md_task(self):
        if not self.pdf_2_md_submitted:
            if self.check_pdf_2_md_task_exits():
                print(f"PDF to MD conversion task for paper '{self.paper.title}' has already been submitted.")
                return
            # move the local PDF file to "human_upload_pdfs" folder.
            pdf_status = self.paper.pdf_status.get()
            local_pdf_path = pdf_status.check_pdf_locally()
            if not local_pdf_path:
                print(f"Cannot submit PDF to MD conversion task for paper '{self.paper.title}' because the PDF file is not available locally. Please check if the PDF file exists in Zotero and if it has been downloaded.")
                return
            local_pdf_path = Path(local_pdf_path)
            if get_config("pdf2md_mode") == "api":
                # 只提交任务，上传/轮询/下载由独立的 mineru-worker 进程完成，结果写到 out_mds/{pdf_key}.md
                from .mineru_tasks import enqueue
                enqueue(str(pdf_status.pdf_key_in_zotero), str(local_pdf_path), str(self.paper.title))
                self.pdf_2_md_submitted = True
                self.save()
                print(f"Submitted PDF to MD conversion task (MinerU API) for paper '{self.paper.title}'.")
                return
            target_path = _workstation / "human_upload_pdfs" / local_pdf_path.name
            shutil.move(str(local_pdf_path), str(target_path))
            self.pdf_2_md_submitted = True
            self.save()
            print(f"Submitted PDF to MD conversion task for paper '{self.paper.title}'.")
    

class TracedCollection(BaseModel):
    # 只追踪文献（Paper/PDF/MD/quick summary），不进入主题summary流程；仅在UI中手动触发，不参与heart_beat
    name = CharField(unique=True) # Zotero集合名称（包含子集合）
    note = TextField(null=True) # 备注
    last_traced_at = DateTimeField(null=True) # 上次成功追踪的时间
    last_paper_count = IntegerField(null=True) # 上次追踪到的文献数量

    def trace(self) -> int | None:
        from datetime import datetime
        count = Paper.trace_collection(str(self.name))
        if count is not None:
            self.last_traced_at = datetime.now()
            self.last_paper_count = count
            self.save()
        return count


# check if need to create tables
if not Paper.table_exists():
    with BaseModel._meta.database: #type: ignore
        BaseModel._meta.database.create_tables([Paper, PDFStatus, MDStatus, Theme, ThemePaper]) #type: ignore
if not TracedCollection.table_exists():
    TracedCollection.create_table()


def check_pdf2md_tasks():
    def _move_pdf_to_old(pdf_file: Path, md_path: Path):
        # 路径显式传入：md 已在 out_mds（如 MinerU API 产出）时不会走复制分支，不能依赖外层循环变量
        target_pdf_path = _workstation / "old_human_pdfs" / pdf_file.name
        shutil.move(str(pdf_file), str(target_pdf_path))
        print(f"PDF to MD conversion task for PDF {pdf_file.stem} is finished. MD file saved to {md_path}")
    # 对于每一个human_upload_pdfs中的PDF文件，检查转化任务是否完成，如果已完成，复制MD文件到out_mds目录，并将其移动到old_human_pdfs目录中。
    human_upload_dir = _workstation / "human_upload_pdfs"
    for pdf_file in human_upload_dir.glob("*.pdf"):
        pdf_name = pdf_file.stem
        md_path = _workstation / "out_mds" / f"{pdf_name}.md"
        if md_path.exists():
            _move_pdf_to_old(pdf_file, md_path)
            continue
        matched_files = list(mineru_ouput_dir.glob(mineru_glob_pattern.format(pdf_name=pdf_name)))
        if len(matched_files) > 1:
            raise Exception(f"Multiple mineru result files found for PDF {pdf_name}: {matched_files}. Please check the mineru output directory and glob pattern.")
        if not matched_files:
            #print(f"No mineru result file found for PDF {pdf_name}. The PDF to MD conversion task may not be finished yet. Please check the mineru output directory and glob pattern.")
            continue
        md_file = matched_files[0]
        target_md_path = _workstation / "out_mds" / f"{pdf_name}.md"
        shutil.copy(md_file, target_md_path)
        # move the processed PDF file to old_human_pdfs directory
        _move_pdf_to_old(pdf_file, target_md_path)

def add_one_paper_to_rag():
    #def append_paper(cls, paper_id: int, md_key: str, ref_detail: str):
    print("trying to add one paper to RAG system...")
    from .rag import EmbedStatus
    #从MD_Status中取一个解析好了的记录
    md_status = MDStatus.select().where(MDStatus.has_md_locally == True, MDStatus.submit_2_rag == False).first()
    if not md_status:
        return
    paper_id = md_status.paper.id
    md_key = md_status.paper.pdf_status.get().pdf_key_in_zotero #type: ignore
    ref_detail = md_status.paper.bib_full_info()
    EmbedStatus.append_paper(paper_id, md_key, ref_detail)
    md_status.submit_2_rag = True
    md_status.save()
    print(f"Added paper '{md_status.paper.title}' to RAG system with paper ID.")


def heart_beat():
    # 自动化任务
    add_one_paper_to_rag()
    note_storage.sync_pending_notes()
    check_pdf2md_tasks()
    _write_queue_checkpoint()
    Theme.update_all()
    Paper.update_all()
    PDFStatus.update_all()
    MDStatus.update_all()
    PDFStatus.update_all()
    Paper.update_all()
    ThemePaper.update_summary()
    add_one_paper_to_rag()
    pass

def search_quick_summary_by_key_words(
    query: str,
) -> Dict[int, str]:
    """
    使用 Whoosh 搜索文献摘要。支持括号、AND、OR、NOT 等完整语法。
    
    Args:
        query (str): 搜索查询，支持:
            - "deep learning" AND reinforcement  (完整短语 + 关键词)
            - (quantum OR classical) AND computing  (括号改变优先级)
            - NOT deprecated OR legacy  (否定操作)
            - "neural network" 3-gram  (多个短语和关键词)
    Example:
        - query = "deep learning" AND reinforcement
          搜索既包含"deep learning"短语，又包含"reinforcement"单词的论文
        - query = (quantum OR classical) AND computing
          搜索包含"quantum"或"classical"，且包含"computing"的论文
        - query = "neural network" AND (NOT deprecated)
          搜索包含"neural network"但不包含"deprecated"的论文
    Return:
        Dict[int, str]: 论文ID到quick_summary的映射
    """
    # 查询前先同步索引缺失的 paper_id 文档（不修改 Paper 代码）
    all_papers = _sync_missing_quick_summary_docs()

    if not query.strip():
        return all_papers
    
    # 使用持久化索引执行查询
    try:
        ix = _get_search_index()
        with ix.searcher() as searcher:
            qp = QueryParser("content", ix.schema)
            parsed_query = qp.parse(query)
            results = searcher.search(parsed_query, limit=None)
            
            # 收集结果
            result_dict = {}
            for hit in results:
                paper_id = int(hit['doc_id'])
                if paper_id in all_papers:
                    result_dict[paper_id] = all_papers[paper_id]
            
            return result_dict
    except Exception as e:
        print(f"Search error: {e}")
        return {}

def search_quick_summary_bm25(query: str, limit: int = 20) -> List[tuple[int, float, str]]:
    """
    复用 quick_summary 的 Whoosh 持久化索引做 BM25 相关度排序检索（Whoosh 默认打分即 BM25F）。
    与 search_quick_summary_by_key_words 的区别：词之间是 OR 关系（多命中者得分更高），并返回得分，
    适合自然语言/多词查询。仍支持 AND/OR/NOT/引号短语语法。

    Return:
        [(paper_id, bm25_score, quick_summary), ...]，按得分从高到低
    """
    from whoosh.qparser import OrGroup

    all_papers = _sync_missing_quick_summary_docs()
    if not query.strip():
        return []
    try:
        ix = _get_search_index()
        with ix.searcher() as searcher:
            parser = QueryParser("content", ix.schema, group=OrGroup.factory(0.9))
            results = searcher.search(parser.parse(query), limit=limit)
            output: List[tuple[int, float, str]] = []
            for hit in results:
                paper_id = int(hit["doc_id"])
                if paper_id in all_papers:
                    output.append((paper_id, float(hit.score), all_papers[paper_id]))
            return output
    except Exception as e:
        print(f"BM25 search error: {e}")
        return []


def rebuild_search_index():
    """重建全部搜索索引（在数据库已有但索引缺失时调用）"""
    print("Rebuilding search index...")
    # 清理旧索引
    if _search_index_dir.exists():
        shutil.rmtree(str(_search_index_dir), ignore_errors=True)

    # create_in 要求目标目录已存在
    _search_index_dir.mkdir(parents=True, exist_ok=True)

    # 重建索引
    ix = create_in(str(_search_index_dir), _search_index_schema)
    writer = ix.writer()

    count = 0
    for paper in Paper.select().where(~Paper.quick_summary.is_null()):  # type: ignore
        writer.add_document(content=paper.quick_summary, doc_id=str(paper.id))
        count += 1

    writer.commit()
    print(f"Search index rebuilt with {count} papers.")

def test_quick_summary_index_sync() -> None:
    """仅调用 search_quick_summary_by_key_words 的交互式测试入口。"""

    def _print_search_results(pattern: str, result: Dict[int, str], max_show: int = 10) -> None:
        print(f"[test] pattern={pattern!r} -> {len(result)} result(s)")
        if not result:
            print("[test]   (no hits)")
            return

        shown = 0
        for paper_id, summary in result.items():
            preview = str(summary).replace("\n", " ").strip()
            if len(preview) > 140:
                preview = preview[:140] + "..."
            print(f"[test]   - paper_id={paper_id}: {preview}")
            shown += 1
            if shown >= max_show:
                break

        if len(result) > max_show:
            print(f"[test]   ... and {len(result) - max_show} more")

    preset_patterns = [
        '"fractionation"',
        '"deep learning" OR "machine learning"',
        '(quantum OR classical) AND computing',
    ]

    print("[test] Run preset patterns...")
    for pattern in preset_patterns:
        result = search_quick_summary_by_key_words(pattern)
        _print_search_results(pattern, result)

    print("[test] Enter interactive query mode.")
    print("[test] Input empty string or q/quit/exit to finish.")
    while True:
        try:
            user_pattern = input("[test] pattern> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[test] Interactive mode terminated.")
            break

        if not user_pattern or user_pattern.lower() in {"q", "quit", "exit"}:
            print("[test] Interactive mode finished.")
            break

        result = search_quick_summary_by_key_words(user_pattern)
        _print_search_results(user_pattern, result)

if __name__ == "__main__":
    #Paper.trace_collection("pebble theoretical core")
    # register the first paper to pdf tracer
    #paper = Paper.select().first()
    #if paper:
    #    pdf_status = paper.register_pdf_tracer()
    #    pdf_status.pdf_require_submitted = False
    #    pdf_status.save()
    #check_pdf2md_tasks()
    #MDStatus.update_all()
    #PDFStatus.update_all()
    #Paper.update_all()
    #临时测试代码，为Paper表添加新的md_ready_date字段, 用sqlmigrate 完成
    #migrator = SqliteMigrator(BaseModel._meta.database) #type: ignore
    #md_ready_date_field = DateField(null=True)
    #migrate(
    #    migrator.add_column('paper', 'md_ready_date', md_ready_date_field)
    #)
    #print("Added md_ready_date field to Paper table.")
    #heart_beat()
    #添加ubmit_2_rag 列
    #migrator = SqliteMigrator(BaseModel._meta.database) #type: ignore
    #submit_2_rag_field = BooleanField(default=False)
    #try:
    #    migrate(
    #        migrator.add_column('mdstatus', 'submit_2_rag', submit_2_rag_field)
    #    )
    #    print("Added submit_2_rag field to MDStatus table.")
    #except Exception as e:
    #    print(f"Failed to add submit_2_rag field to MDStatus table. It may already exist. Error: {e}")
    #heart_beat()
    #check_pdf2md_tasks()
    #Paper.auto_clean()
    #ThemePaper.auto_fix()
    #test_quick_summary_index_sync()
    pass