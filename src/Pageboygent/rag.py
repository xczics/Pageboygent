from typing import Tuple,List,Iterator,Self, Dict, Any, cast
from pathlib import Path
from peewee import Model, SqliteDatabase
from .config import get_config
from .write_queue import checkpoint as _write_queue_checkpoint
import chromadb
from chromadb.api.types import WhereDocument
import datetime
import re
from whoosh.qparser import QueryParser, AndGroup
import uuid
import peewee
import hashlib
import threading
import time
from worthy import Worthy, LLMConfig, EmbeddingConfig, Thresholds

def _get_workstation_dir() -> Path:
    return Path(get_config("workstation_dir")).expanduser()
_workstation = _get_workstation_dir()
_workstation.mkdir(parents=True, exist_ok=True)
embedding_dim = get_config("embedding_dim")
max_chunk_size = get_config("max_chunk_size")

embedding_model = get_config("embedding_model")
embedding_url = get_config("embedding_url")
# if embedding_key is not set, use default "ollama"
try:
    embedding_key = get_config("embedding_key")
except KeyError:
    embedding_key = None
if embedding_key is None:
    embedding_key = "ollama"


def check_embedding_service(timeout: float = 3.0) -> str | None:
    """检查 embedding_url 上的 OpenAI 兼容接口是否可用（Ollama、OpenAI 和各类云平台都走这一个接口）。
    可用返回 None，否则返回原因。以前这里写死检查 localhost:11434 的 Ollama，
    换成云端 embedding 或 Ollama 不在本机时会误报。"""
    import requests

    try:
        resp = requests.get(str(embedding_url).rstrip("/") + "/models", timeout=timeout,
                            headers={"Authorization": f"Bearer {embedding_key}"})
        if resp.status_code >= 400:
            return f"{embedding_url} 返回 HTTP {resp.status_code}"
    except Exception as exc:
        return f"{embedding_url} 无法连接：{exc}"
    return None


# 只提示不退出：embedding 服务不可用时，embedding 入库会失败，但其他功能（UI、问答、summary）不受影响
_embedding_problem = check_embedding_service()
if _embedding_problem:
    print("⚠️ Embedding 服务不可用，embedding 相关功能将不可用：", _embedding_problem)

def hash_file(file_path: Path) -> str:
    #计算文件的hash值，使用sha256算法
    sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

# ---------------------------------------------------------------------------
# Worthy(掌眼,见 ~/code/worthy): 统一负责"这段文本值不值得embedding入库"的判断，
# 以及embedding本身的计算。
#
# 以前这里单独注册、单独维护一个"tiny_model"(独立的base_url/key/model_name配置，
# 专门跑一个小模型来给段落打分)。现在不再单独养这个小模型：判断时统一复用已经
# 注册好的flash模型(model_name_cheap，当前是deepseek-v4-flash)。worthy会在后台
# 用这些判断结果自动训练一个专属当前语料的轻量分类器，大部分case之后就不用再
# 调用flash模型了——比之前手写的tiny_model方案更省成本，也不用再单独维护一份
# "小模型专用"的配置。
# ---------------------------------------------------------------------------

ds_api_key = get_config("ds_api_key")
ds_url = get_config("ds_url")
flash_model_name = get_config("model_name_cheap")

_RAG_CRITERIA = """判断文本是否为学术论文的正文段落。
规则：
    - 如果是完整的叙述性句子（像在讲故事或讲原理），值得入库。
    - 图注和表标题也识别为正文段落，值得入库。
    - 如果是参考文献、标题、[数字]、Fig、网址、乱码或无意义片段，不值得入库。
"""

def _rag_rule_filter(text: str) -> bool | None:
    """零成本的确定性规则，命中时直接判定，不用麻烦模型。"""
    # 如果text足够长，且"<td>"出现特别多次，直接判定不值得入库，因为大概率是表格
    if len(text) > 100 and (text.count("<td>") > 8 or text.count("<tr>") > 10):
        return False
    if "![](images/" in text:
        return False
    return None  # 交给 专用小模型(worthy自己训练的)/flash模型 继续判断

worthy = Worthy(
    llm_config=LLMConfig(
        base_url=ds_url,
        api_key=ds_api_key or "not-configured",   # 未配置时先占位，服务照常启动
        model=flash_model_name,
        temperature=0.05,
        max_tokens=100,
        extra_body={
            # 关闭deepseek的深度思考：只是个简单的0-10打分任务，不需要推理
            "thinking": {
                "type": "disabled",
            },
        },
    ),
    embedding_config=EmbeddingConfig(
        provider="openai_compatible",
        base_url=embedding_url,
        api_key=embedding_key,
        model=embedding_model,
        dimensions=embedding_dim,
    ),
    criteria=_RAG_CRITERIA,
    thresholds=Thresholds(score_binarize_threshold=6),  # 沿用之前"score >= 6 才入库"的标准
    db_path=str(_workstation / "rag_worthy.db"),
    model_dir=str(_workstation / "rag_worthy_models"),
    rule_filter_fn=_rag_rule_filter,
)

def embed_query(text: str) -> list[float]:
    # 检索用的query不需要"值不值得入库"的判断，只要embedding本身，
    # 所以直接用worthy内部的embedder，不走完整的judge()流程。
    return worthy.embedder.embed(text).tolist()

def parse_keyword_query(query: str) -> WhereDocument | None:
    """
    使用 Whoosh 解析关键词语法，再转换为 Chroma 的 where_document。
    支持：引号短语、AND/OR/NOT、括号、隐式 AND。
    """
    if not query.strip():
        return None

    def _to_where(node) -> dict | None:
        cls = node.__class__.__name__

        if cls == "Term":
            text = getattr(node, "text", "")
            return {"$contains": text} if text else None

        if cls == "Phrase":
            words = getattr(node, "words", [])
            phrase = " ".join(words).strip()
            return {"$contains": phrase} if phrase else None

        if cls in {"And", "Or"}:
            children = [_to_where(child) for child in getattr(node, "subqueries", [])]
            children = [c for c in children if c]
            if not children:
                return None
            if len(children) == 1:
                return children[0]
            return {"$and" if cls == "And" else "$or": children}

        if cls == "Not":
            child = _to_where(getattr(node, "query", None))
            if not child:
                return None
            if "$contains" in child:
                return {"$not_contains": child["$contains"]}
            if "$and" in child:
                return {"$or": [{"$not_contains": c["$contains"]} if "$contains" in c else {"$not": c} for c in child["$and"]]}
            if "$or" in child:
                return {"$and": [{"$not_contains": c["$contains"]} if "$contains" in c else {"$not": c} for c in child["$or"]]}
            return {"$not": child}

        # 未知节点：回退为包含字符串表示
        text = str(node).strip()
        return {"$contains": text} if text else None

    try:
        parsed = QueryParser("content", schema=None, group=AndGroup).parse(query)
    except Exception:
        return None

    where_doc = _to_where(parsed)
    return cast(WhereDocument, where_doc) if where_doc else None

def vector_retrieve(
    query: str,
    top_k: int = 10,
    filters: dict | None = None,
) -> list[dict]:
    """
    纯向量召回
    """
    query_vec = embed_query(query)

    filters = filters or {}
    where_clause = {}

    # paper_id 过滤
    if "paper_ids" in filters:
        where_clause["paper_id"] = {"$in": filters["paper_ids"]}

    # md_key 过滤
    if "md_keys" in filters:
        where_clause["md_key"] = {"$in": filters["md_keys"]}

    # embedding_model 过滤
    if "embedding_model" in filters:
        where_clause["embedding_model"] = filters["embedding_model"]

    results = DBcollection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        where=where_clause if where_clause else None,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0] #type: ignore
    metas = results["metadatas"][0] #type: ignore
    dists = results["distances"][0] #type: ignore

    output = []
    for doc, meta, dist in zip(docs, metas, dists):
        output.append({
            "id": f"{meta['paper_id']}_{meta['line_number']}",
            "paper_id": meta["paper_id"],
            "md_key": meta["md_key"],
            "line_number": meta["line_number"],
            "text": doc,
            "score": 1.0 - dist,  # cosine distance → similarity
            "source": meta.get("source"),
        })

    return sorted(output, key=lambda x: x["score"], reverse=True)

def keywords_retrieve(
    query: str,
    filters: dict | None = None,
) -> list[dict]:
    """
    关键词检索
    """
    filters = filters or {}
    where_clause = {}

    # paper_id 过滤
    if "paper_ids" in filters:
        where_clause["paper_id"] = {"$in": filters["paper_ids"]}

    # md_key 过滤
    if "md_keys" in filters:
        where_clause["md_key"] = {"$in": filters["md_keys"]}

    # embedding_model 过滤
    if "embedding_model" in filters:
        where_clause["embedding_model"] = filters["embedding_model"]

    # 关键字召回
    where_doc = parse_keyword_query(query)
    if where_doc is None:
        return []

    results = DBcollection.get(
        where=where_clause if where_clause else None,
        where_document=where_doc,
        include=["documents", "metadatas"],
    )

    docs = results.get("documents") or []
    metas = results.get("metadatas") or []

    output = []
    for doc, meta in zip(docs, metas):
        output.append({
            "id": f"{meta['paper_id']}_{meta['line_number']}",
            "paper_id": meta["paper_id"],
            "md_key": meta["md_key"],
            "line_number": meta["line_number"],
            "text": doc,
            "score": 1.0,  
            "source": meta.get("source"),
        })

    return output

# ---------------------------------------------------------------------------
# 全文 BM25 检索：在 Chroma 已入库的原文切块上建立一个 Whoosh 倒排索引（BM25F 打分）。
# 切块本身不重新计算，索引只是 Chroma 文档的镜像，每次检索前按 id 增量同步
# （新增的切块补进来、Chroma 中已删除的切块移除），首次使用时会全量建一次。
# ---------------------------------------------------------------------------
_bm25_index_dir = _workstation / "fulltext_bm25_index"
_bm25_lock = threading.RLock()
_bm25_ix = None
_bm25_last_sync = 0.0
_BM25_SYNC_INTERVAL_SECONDS = 120


def _bm25_schema():
    from whoosh.analysis import StemmingAnalyzer
    from whoosh.fields import Schema, ID, TEXT, NUMERIC, STORED

    return Schema(
        chunk_id=ID(stored=True, unique=True),
        paper_id=NUMERIC(stored=True),
        line_number=NUMERIC(stored=True),
        md_key=STORED,
        source=STORED,
        # 英文为主的论文正文：词干化 + 去停用词，"evaporation"能命中"evaporated"
        content=TEXT(analyzer=StemmingAnalyzer(), stored=True),
    )


def _get_bm25_index():
    global _bm25_ix
    from whoosh.index import create_in, exists_in, open_dir

    if _bm25_ix is None:
        _bm25_index_dir.mkdir(parents=True, exist_ok=True)
        if exists_in(str(_bm25_index_dir)):
            _bm25_ix = open_dir(str(_bm25_index_dir))
        else:
            _bm25_ix = create_in(str(_bm25_index_dir), _bm25_schema())
    return _bm25_ix


def sync_bm25_index(force: bool = False, batch_size: int = 1000, on_progress=None) -> int:
    """把 Chroma 中的切块增量同步进 BM25 索引，返回本次新增的切块数。"""
    global _bm25_last_sync
    with _bm25_lock:
        if not force and time.monotonic() - _bm25_last_sync < _BM25_SYNC_INTERVAL_SECONDS:
            return 0
        ix = _get_bm25_index()
        chroma_ids = set(DBcollection.get(include=[])["ids"])
        with ix.searcher() as searcher:
            reader = searcher.reader()
            if reader.has_deletions():
                indexed_ids = {fields["chunk_id"] for _, fields in reader.iter_docs()}
            else:
                indexed_ids = set(reader.field_terms("chunk_id"))
        to_add = sorted(chroma_ids - indexed_ids)
        to_remove = indexed_ids - chroma_ids
        if to_add or to_remove:
            if on_progress and len(to_add) > batch_size:
                on_progress(f"正在为 {len(to_add)} 个原文切块建立 BM25 索引（仅首次较慢）...")
            writer = ix.writer(limitmb=256)
            try:
                for chunk_id in to_remove:
                    writer.delete_by_term("chunk_id", chunk_id)
                for start in range(0, len(to_add), batch_size):
                    batch = to_add[start:start + batch_size]
                    got = DBcollection.get(ids=batch, include=["documents", "metadatas"])
                    for chunk_id, doc, meta in zip(got["ids"], got["documents"] or [], got["metadatas"] or []):
                        if not doc or not meta:
                            continue
                        writer.update_document(
                            chunk_id=chunk_id,
                            paper_id=int(meta["paper_id"]),
                            line_number=int(meta["line_number"]),
                            md_key=meta.get("md_key"),
                            source=meta.get("source"),
                            content=doc,
                        )
                writer.commit()
            except BaseException:
                writer.cancel()
                raise
        _bm25_last_sync = time.monotonic()
        return len(to_add)


def bm25_retrieve(
    query: str,
    top_k: int = 20,
    filters: dict | None = None,
) -> list[dict]:
    """
    全文 BM25 检索，输出格式与 vector_retrieve 一致。
    查询词之间是 OR 关系（命中越多得分越高），也支持 AND/OR/NOT/引号短语。
    filters 支持 paper_ids。
    """
    from whoosh.qparser import QueryParser, OrGroup
    from whoosh.query import Or, NumericRange

    if not query.strip():
        return []
    sync_bm25_index()
    with _bm25_lock:
        ix = _get_bm25_index()
    parsed = QueryParser("content", ix.schema, group=OrGroup.factory(0.9)).parse(query)
    restrict = None
    paper_ids = (filters or {}).get("paper_ids")
    if paper_ids:
        restrict = Or([NumericRange("paper_id", int(pid), int(pid)) for pid in paper_ids])
    output = []
    with ix.searcher() as searcher:
        try:
            hits = searcher.search(parsed, limit=top_k, filter=restrict)
        except ZeroDivisionError:
            # Whoosh 的已知问题：OR 加覆盖度加权（OrGroup.factory）遇到“单个短语 + NOT”会除以零，换成普通 OR 重查
            parsed = QueryParser("content", ix.schema, group=OrGroup).parse(query)
            hits = searcher.search(parsed, limit=top_k, filter=restrict)
        for hit in hits:
            output.append({
                "id": hit["chunk_id"],
                "paper_id": hit["paper_id"],
                "md_key": hit.get("md_key"),
                "line_number": hit["line_number"],
                "text": hit["content"],
                "score": float(hit.score),
                "source": hit.get("source"),
            })
    return output


def retrieve(
    query: str,
    top_k: int = 10,
    filters: dict | None = None,
    ) -> list[dict]:
    """
    暂时先仅用向量召回
    """
    return vector_retrieve(query, top_k, filters)

def smart_split_text(
    text: str,
    max_chunk_size: int = 512,
) -> List[str]:
    """
    中英文混合 + LaTeX公式保护的智能分句
    """
    if len(text) <= max_chunk_size:
        return [text]
    
    # 1️⃣ 保护 $...$ 公式
    parts = []
    last = 0
    for m in re.finditer(r'\$(.*?)\$', text):
        parts.append(text[last:m.start()])
        parts.append(m.group(0))  # 公式整体
        last = m.end()
    parts.append(text[last:])

    # 2️⃣ 分句正则
    # 中文强终止
    # 英文：. ? ! 后面必须跟空格+大写字母
    sentence_end = re.compile(
        r'([。！？；…\n]|'
        r'(?<!\.\d)'                  # 防 1.23
        r'(?<!\b[A-Za-z]\.)'          # 防 A. B.
        r'(?<!Dr\.)'
        r'(?<!Mr\.)'
        r'(?<!Ms\.)'
        r'(?<!Mrs\.)'
        r'(?<!Fig\.)'
        r'(?<!Eq\.)'
        r'(?<!Ref\.)'
        r'[.!?]'
        r'(?=\s+[A-Z])'
        r'(?![a-z]))'
    )

    chunks = []
    buffer = ""

    for part in parts:
        if part.startswith('$'):
            # 公式直接追加
            buffer += part
            continue

        sentences = sentence_end.split(part)
        for i in range(0, len(sentences)-1, 2):
            sent = sentences[i] + (sentences[i+1] or "")
            buffer += sent

            if len(buffer) >= max_chunk_size:
                chunks.append(buffer.strip())
                buffer = ""

        if len(sentences) % 2 == 1 and sentences[-1]:
            buffer += sentences[-1]

    if buffer.strip():
        chunks.append(buffer.strip())

    final_chunks = []
    for c in chunks:
        if not final_chunks:
            final_chunks.append(c)
            continue
        prev = final_chunks[-1]
        # 只有当前 chunk 很小，才考虑合并
        if len(c) < 0.2 * max_chunk_size:
            if len(prev) + len(c) + 1 <= max_chunk_size:
                # ✅ 整体吞并，而不是拼接句子
                final_chunks[-1] = prev + " " + c
            else:
                final_chunks.append(c)
        else:
            final_chunks.append(c)
    return final_chunks

def embedding_md(md_path: str|Path, start_line: int = 0 ,max_chunk: int = max_chunk_size) -> Iterator[Tuple[List[float], int, str, int]]:
    with open(md_path, "r") as f:
        line_number = 0
        have_last_line = False
        last_line_message = ""
        while True:
            if have_last_line:
                line = last_line_message
                last_line_message = ""
                line_number += 1
                have_last_line = False
            else:
                line = f.readline()
                if not line:
                    break
                line_number += 1
                if line_number <= start_line:
                    continue  # 跳过已经embedding过的行
            #检查是否以句子结束符号结尾，如果没有，就继续下一行，直到遇到句子结束符号或者文件结尾
            while not re.search(r'[。！？；….!?]$', line.strip()) and \
                (not line.strip().startswith("!")) and (not line.strip().startswith("#")):
                next_line = f.readline()
                if not next_line :
                    break
                if next_line.startswith("#"):
                    have_last_line = True
                    last_line_message = next_line.strip()
                    break # 遇到下一个标题了，也停止继续合并了
                if next_line.strip().startswith("![](im"): # 如果是图片了，也停止继续合并了
                    line+= next_line
                    break
                line += next_line
                line_number += 1
            line = line.strip()
            chunks = smart_split_text(line, max_chunk_size)
            for subline_index, chunk in enumerate(chunks):
                if len(chunk) < 5:
                    continue  # 太短的行不embedding
                verdict = worthy.judge(chunk)
                print(f"#{line_number}-{subline_index} Line: {chunk}, \n"
                      f"Worthy: should_index={verdict.should_index} (source={verdict.source})")
                if verdict.should_index:
                    # embedding是judge()顺带算出来的，不用再单独调用一次embedding接口
                    yield verdict.embedding.tolist(), line_number, chunk, subline_index

def test_score():
    text1 = """
# Thermodynamics of Element Volatility and its Application to Planetary Processes
"""
    text2 = """
author@example.org
Bruce Fegley, Jr.
Planetary Chemistry Laboratory, Department of Earth and Planetary Sciences and McDonnell Center for the Space Sciences, Washington University, St. Louis, Missouri, USA
"""
    verdict1 = worthy.judge(text1)
    verdict2 = worthy.judge(text2)
    print(f"Text: {text1}\nShould index: {verdict1.should_index} (source={verdict1.source})")
    print(f"Text: {text2}\nShould index: {verdict2.should_index} (source={verdict2.source})")

db_path = str(_workstation / "embed_db")

if not Path(db_path).exists():
    Path(db_path).mkdir(parents=True, exist_ok=True)

EmbedDatabase = chromadb.PersistentClient(path=db_path)
DBcollection = EmbedDatabase.get_or_create_collection(name="embeddings")

class BaseModel(Model):
    class Meta:
        database = SqliteDatabase(str(_workstation / "EmbedStatus.db"))

class EmbedStatus(BaseModel):
    STATUS_DONE = 0
    STATUS_PROCESSING = 1
    STATUS_PENDING = 2
    STATUS_CURRENT = 3 # 当前正在被某个进程embedding.
    paper_id = peewee.IntegerField(unique=True,index=True) #论文ID，唯一索引
    last_embedded_line = peewee.IntegerField(default=0) #已经embedding过的行号，初始值为0，每次embedding后更新为当前行号
    ref_detail = peewee.TextField() #参考文献信息，以EPSL格式存储
    md_key = peewee.CharField() #形似“AXYDDAB”,与pdf的key相同,其path由key唯一确定
    file_hash = peewee.CharField(max_length=64, index=True) #md文件的hash值，如果文件变化，则自动重算
    last_embedded_at = peewee.DateTimeField(null=True) #embedding的时间, 只有当文件最后修改时间晚于embed_time时检验hash值
    embedding_model = peewee.CharField(max_length=128) #使用的embedding模型，默认为配置文件中的embedding_model，如果配置文件中embedding_model发生变化，则自动重算
    updated_at = peewee.DateTimeField() #记录每次更新的时间
    status = peewee.IntegerField(
        default=STATUS_PENDING,
        choices=(
            (STATUS_DONE, "done"),
            (STATUS_PROCESSING, "processing"),
            (STATUS_PENDING, "pending"),
            (STATUS_CURRENT, "current"),
        )
    )

    @classmethod
    def append_paper(cls, paper_id: int, md_key: str, ref_detail: str):
        #将论文添加到数据库，如果已经存在，则不添加
        obj, created = cls.get_or_create(
            paper_id=paper_id,
            defaults={
                "md_key": md_key,
                "ref_detail": ref_detail,
                "file_hash": "",
                "embedding_model": embedding_model,
                "status": cls.STATUS_PENDING,
                "updated_at": datetime.datetime.now(),
            }
        )
        if created:
            hash_value = hash_file(_workstation / "out_mds" /f"{md_key}.md")
            obj.file_hash = hash_value
            obj.save()
    @classmethod
    def acquire_and_lock_task(cls, status: int = STATUS_PROCESSING) -> Self|None:
        from .lock import lock_it
        with lock_it("embedding_task_lock"):
            obj = (
                cls.select()
                .where(cls.status == status)
                .order_by(cls.updated_at)
                .get_or_none()
            )
            if obj is None:
                return None
            cls.update(
                status=cls.STATUS_CURRENT,
                updated_at=datetime.datetime.now(),
            ).where(cls.id == obj.id).execute() #type: ignore
            obj.status = cls.STATUS_CURRENT
            obj.updated_at = datetime.datetime.now()
            return obj
    @classmethod
    def embedding_processing_papers(cls):
        #获取第一个状态为processing的论文，返回其md_key和已经embedding的行号
        obj = cls.acquire_and_lock_task(cls.STATUS_PROCESSING)
        if obj is None:
            return None
        # 继续embedding这个论文
        obj.embedding_paper()
    @classmethod
    def embedding_pending_papers(cls):
        #获取第一个状态为pending的论文，返回其md_key和已经embedding的行号
        obj = cls.acquire_and_lock_task(cls.STATUS_PENDING)
        if obj is None:
            return None
        # 继续embedding这个论文
        obj.embedding_paper()

    def embedding_paper(self):
        paper_id:int = self.paper_id #type: ignore
        md_key = str(self.md_key)
        md_path = _workstation / "out_mds" / f"{md_key}.md"
        last_line:int = self.last_embedded_line #type: ignore
        last_line_processed = last_line
        last_commited_line = last_line
        try:
            for vec, line_number, line, subline_index in embedding_md(md_path, start_line=last_line):
                # 将embedding结果保存到chromadb中，id为"{md_key}_{line_number}"
                DBcollection.add(
                    ids=f"{paper_id}_{line_number}_{subline_index}",
                    documents=line.strip(),
                    embeddings=vec,
                    metadatas={
                        "paper_id": paper_id,
                        "line_number": line_number,
                        "md_key": md_key,
                        "embedding_model": embedding_model,
                        "source": str(md_path),
                    },
                    )
                # 更新数据库中的last_embedded_line
                last_line_processed = line_number
                if line_number > last_commited_line + 8: # 每8行更新一次数据库，减少数据库写入次数
                    self.last_embedded_line = line_number
                    self.last_embedded_at = datetime.datetime.now()
                    self.updated_at = datetime.datetime.now()
                    last_commited_line = line_number
                    self.save()
                # 检查优雅关闭标志，通知任务在下一个 chunk 边界中断
                if _shutdown_event.is_set():
                    raise KeyboardInterrupt("Graceful shutdown requested")
                # chunk 边界：让 Web 端排队的快速写操作插队（写的是 pageboygent.db，与 EmbedStatus.db 无关）
                _write_queue_checkpoint()
        except BaseException as e:
            if not isinstance(e, KeyboardInterrupt):
                print(f"Error embedding paper {paper_id}: {e}")
            self.status = self.STATUS_PROCESSING # 遇到错误，设置状态为processing，以便下次继续处理
            self.save()
            raise e
        else:
            # embedding完成，更新状态为done
            self.status = self.STATUS_DONE
        finally:
            self.last_embedded_line = last_line_processed
            self.last_embedded_at = datetime.datetime.now()
            self.updated_at = datetime.datetime.now()
            self.save()

#检查EmbedStatus表是否存在，如果不存在则创建
if not EmbedStatus.table_exists():
    EmbedStatus.create_table()

# 优雅关闭事件：主进程设置此事件后，正在 embedding 的任务会尽快中断并保存状态
_shutdown_event = threading.Event()

def request_shutdown() -> None:
    """通知正在进行的 embedding 任务尽快保存进度后退出。"""
    _shutdown_event.set()

def heart_beat():
    EmbedStatus.embedding_processing_papers()
    EmbedStatus.embedding_pending_papers()
    EmbedStatus.embedding_processing_papers()
    EmbedStatus.embedding_pending_papers()

def test_treive():
    while True:
        query = input("请输入查询内容（输入exit退出）：")
        if query.lower() == "exit":
            break
        results = vector_retrieve(
            query=query,
            top_k=3,
            )
        for r in results:
            print(f"[{r['score']:.3f}] {r['text']}")

def test_keywords_retrieve():
    while True:
        query = input("请输入查询内容（输入exit退出）：")
        if query.lower() == "exit":
            break
        results = keywords_retrieve(
            query=query,
            )
        for r in results:
            print(f"[{r['score']:.3f}] {r['text']}")

def print_all_embedding_status():
    # print title of the table
    print(f"{'Paper ID':<10} {'MD Key':<10} {'Last Embedded Line':<20} {'Status':<10} {'Updated At':<20}")
    for obj in EmbedStatus.select():
        status_str = {
            EmbedStatus.STATUS_DONE: "done",
            EmbedStatus.STATUS_PROCESSING: "processing",
            EmbedStatus.STATUS_PENDING: "pending",
            EmbedStatus.STATUS_CURRENT: "current",
        }.get(obj.status, "unknown")
        print(f"{obj.paper_id:<10} {obj.md_key:<10} {obj.last_embedded_line:<20} {status_str:<10} {obj.updated_at:<20}")

def fix_embedding_status():
    # 把所有状态为current的论文改为processing
    EmbedStatus.update(status=EmbedStatus.STATUS_PROCESSING).where(EmbedStatus.status == EmbedStatus.STATUS_CURRENT).execute()

def rag_database_searching(
    vector_queries: List[str],
    keyword_queries: List[str],
    filters: dict | None = None,
    vector_top_k: int = 50,
    vector_min_score: float = 0.2,
    vector_weight: float = 0.7,
) -> Tuple[Dict[int, List[str]], List[int]]:
    """
    分别使用向量检索和关键词检索，然后将结果合并，返回:
    - 一个字典，key为paper_id，value为该paper_id对应的所有检索到的文本列表
    - 一个列表，包含所有检索到的paper_id, 按检索到的文本数量排序，多的排在前面
    """
    filters = filters or {}
    paper_texts: Dict[int, List[str]] = {}
    paper_scores: Dict[int, float] = {}

    # 1. 向量检索
    for vq in vector_queries:
        vector_results = vector_retrieve(vq, top_k=vector_top_k, filters=filters)
        for res in vector_results:
            if res["score"] < vector_min_score:
                continue
            pid = res["paper_id"]
            if pid not in paper_texts:
                paper_texts[pid] = []
                paper_scores[pid] = 0.0
            paper_texts[pid].append(res["text"])
            paper_scores[pid] += res["score"]/vector_min_score * vector_weight

    # 2. 关键词检索
    for kq in keyword_queries:
        keyword_results = keywords_retrieve(kq, filters=filters)
        for res in keyword_results:
            pid = res["paper_id"]
            if pid not in paper_texts:
                paper_texts[pid] = []
                paper_scores[pid] = 0.0
            paper_texts[pid].append(res["text"])
            paper_scores[pid] += res["score"] * (1 - vector_weight)

    # 3. 按检索到的文本数量排序paper_id
    sorted_paper_ids = sorted(paper_texts.keys(), key=lambda pid: len(paper_texts[pid]), reverse=True)

    return paper_texts, sorted_paper_ids

if __name__ == "__main__":
    #test_score()
    #test_embedding()
    #heart_beat()
    # 打印所有的embedding状态
    #for obj in EmbedStatus.select():
    #    print(f"Paper ID: {obj.paper_id}, MD Key: {obj.md_key}, Last Embedded Line: {obj.last_embedded_line}, Status: {obj.status}, Updated At: {obj.updated_at}")
    # 把所有状态改成pending，重新embedding
    # EmbedStatus.update(status=EmbedStatus.STATUS_PENDING).execute()
    #print_all_embedding_status()
    #fix_embedding_status()
    pass
    #test_treive()
    #results = vector_retrieve(
    #    query="What determines element volatility in planetary processes?",
    #    top_k=10,
    #)
    # print(parse_keyword_query("volatility AND planetary OR processes"))
    # heart_beat()
    test_keywords_retrieve()