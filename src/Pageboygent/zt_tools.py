from .config import get_config
from pyzotero import Zotero
import glob
import shutil
from pathlib import Path
from typing import Dict, Tuple, List
from pathlib import Path
from markdownify import markdownify as md
import random
import time
import peewee
import logging

# 注意: 本地写入相关的异常 (LocalAPIKeyRequiredError / LocalAPIDeniedError / ServerIDMismatchError ...)
# 只在 pyzotero.errors 里,旧的 pyzotero.zotero_errors 兼容模块并没有重新导出它们。
# 见 zt_api_tracking.md 里的记录,升级pyzotero时要重新确认这一点。
try:
    from pyzotero import errors as ze
except Exception:
    ze = None  # type: ignore

random.seed(time.time())

ZOTERO_LOCAL_APP_NAME = "Pageboygent"

# 本地 API 里用户 ID 填 0 表示“本机这个 Zotero 库”，没配置时就用它
zt_id = str(get_config("zotero_id") or "0")


class ZoteroActionPaused(RuntimeError):
    pass


_zotero_guard_log = Path(get_config("workstation_dir")).expanduser() / "zotero_guard.log"
_zotero_guard_log.parent.mkdir(parents=True, exist_ok=True)   # 第一次运行时数据目录还不存在
_guard_logger = logging.getLogger("Pageboygent.ZoteroGuard")
if not _guard_logger.handlers:
    _guard_logger.setLevel(logging.INFO)
    _guard_handler = logging.FileHandler(_zotero_guard_log, encoding="utf-8")
    _guard_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | action=%(action)s | reason=%(reason)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    _guard_logger.addHandler(_guard_handler)


def _is_connection_issue(exc: BaseException) -> bool:
    text = str(exc).lower()
    keywords = [
        "timeout",
        "timed out",
        "connection",
        "network",
        "unreachable",
        "refused",
        "429",
        "503",
        "502",
        "backoff",
    ]
    if any(k in text for k in keywords):
        return True
    if ze is not None:
        known = (
            getattr(ze, "CouldNotReachURLError", tuple()),
            getattr(ze, "TooManyRequestsError", tuple()),
            getattr(ze, "TooManyRetriesError", tuple()),
            getattr(ze, "HTTPError", tuple()),
        )
        if isinstance(exc, known):
            return True
    return False


def _pause_action(action: str, exc: BaseException) -> None:
    msg = (
        f"[PAUSED] {action} paused because the local Zotero API is unreachable or unstable "
        f"(is the Zotero desktop app running, with Settings > Advanced > \"Allow other applications "
        f"on this computer to communicate with Zotero\" enabled?): {exc}"
    )
    print(msg)
    _guard_logger.error(
        "paused",
        extra={
            "action": action,
            "reason": str(exc).replace("\n", " ").strip(),
        },
    )
    raise ZoteroActionPaused(msg) from exc


# ---------------------------------------------------------------------------
# 本地API写入授权 (authorize_local)
#
# Zotero的本地API现在原生支持写入,但每次写入都必须带上一个"本地API key",这个key
# 与zotero.org的Web API key毫无关系,而是通过 zot.authorize_local(app_name) 触发
# Zotero客户端弹出一个"Allow / Always Allow / Deny"的授权对话框拿到的:
#   - 选择 "Allow"(仅一次): 返回的key只能用于接下来的第一次成功写入,用掉即失效。
#   - 选择 "Always Allow"(始终允许): 返回的key长期有效,可以缓存下来,以后每次启动
#     程序时都可以直接复用,不用再弹窗确认。
#
# 我们把拿到的"Always Allow" key缓存在 workstation_dir 下的一个文本文件里,这样只需要
# 在Zotero里点一次"始终允许",以后就不用再管了。也支持通过 zotero_local_api_key 配置项
# (环境变量 PAGEBOYGENT_ZOTERO_LOCAL_API_KEY) 预先注入一个已经拿到的key,方便无人值守
# 的环境(例如服务器/CI)跳过交互式弹窗。
# ---------------------------------------------------------------------------

_local_api_key_cache_path = Path(get_config("workstation_dir")).expanduser() / "zotero_local_api_key.txt"


def _load_local_api_key() -> str | None:
    configured = get_config("zotero_local_api_key")
    if configured:
        return configured
    if _local_api_key_cache_path.exists():
        cached = _local_api_key_cache_path.read_text(encoding="utf-8").strip()
        return cached or None
    return None


def _save_local_api_key(key: str) -> None:
    _local_api_key_cache_path.parent.mkdir(parents=True, exist_ok=True)
    _local_api_key_cache_path.write_text(key, encoding="utf-8")


def _clear_local_api_key_cache() -> None:
    try:
        _local_api_key_cache_path.unlink()
    except FileNotFoundError:
        pass


# 全部读写都走本机 Zotero 客户端的本地 API（不需要 zotero.org 的 Web API key）
zot = Zotero(zt_id, "user", None, local=True, local_api_key=_load_local_api_key())
# 兼容旧代码: 以前 zot 是远程Web API客户端(负责写), zot_local 是只读的本地API客户端(负责读),
# 需要靠"远程写入 -> 拿到新item的key -> 再手动把文件抄一份进本地storage目录"这种别扭的方式
# 才能让本地Zotero马上"看到"新写入的内容。现在Zotero本地API本身就能写,两者被合并成同一个
# 本地客户端,不再有真正意义上的"双API"。保留 zot_local 这个名字只是为了不用同时改掉
# note_storage.py / migrate_paper_notes_html.py 等其他模块里的旧引用。
zot_local = zot


def _is_local_api_denied(exc: BaseException) -> bool:
    if ze is None:
        return False
    return isinstance(exc, getattr(ze, "LocalAPIDeniedError", tuple()))


def _is_local_api_key_required(exc: BaseException) -> bool:
    if ze is None:
        return False
    return isinstance(exc, getattr(ze, "LocalAPIKeyRequiredError", tuple()))


def _is_server_id_mismatch(exc: BaseException) -> bool:
    if ze is None:
        return False
    return isinstance(exc, getattr(ze, "ServerIDMismatchError", tuple()))


def _ensure_local_write_authorized() -> None:
    """确保 zot 持有一个可用的本地API写入key,必要时触发一次性的Zotero授权弹窗。"""
    if zot.local_api_key:
        return
    cached_key = _load_local_api_key()
    if cached_key:
        zot.local_api_key = cached_key
        return
    print(
        f"[Zotero] 正在请求本地API写入权限,请在Zotero弹出的窗口中选择\"始终允许 (Always Allow)\","
        f"这样以后就不用再次确认了..."
    )
    authorized = zot.authorize_local(ZOTERO_LOCAL_APP_NAME)
    if authorized.get("remember") and authorized.get("key"):
        _save_local_api_key(authorized["key"])
        print("[Zotero] 已获得永久本地写入授权,凭证已缓存到本地,后续启动无需再次确认。")
    else:
        print("[Zotero] 仅获得了一次性写入授权,下一次写入操作会再次弹窗,建议在弹窗中选择\"始终允许\"。")


def _perform_local_write(action: str, func, *args, **kwargs):
    """
    统一执行一次本地Zotero写入操作,替代原来分散在各个函数里、只针对"远程API"的写入调用。

    - 写入前自动确保已经拿到本地API写入授权(见 _ensure_local_write_authorized)。
    - 授权key失效了(LocalAPIKeyRequiredError,比如用户在Zotero里手动收回了授权)时,
      清掉缓存、重新走一次授权流程,再重试一次。
    - 连接到了不同的Zotero实例/数据库(ServerIDMismatchError,例如换了Zotero profile
      或者重装了Zotero)时,重置缓存的server_id再重试一次。
    - 用户在弹窗里点了"拒绝",或者本地Zotero API整体连不上,统一走 _pause_action
      暂停并记录到 zotero_guard.log,而不是让异常裸抛出去打断整个流程。
    """
    try:
        _ensure_local_write_authorized()
        return func(*args, **kwargs)
    except Exception as e:
        if _is_local_api_denied(e):
            _pause_action(action, e)
        if _is_local_api_key_required(e):
            zot.local_api_key = None
            _clear_local_api_key_cache()
            try:
                _ensure_local_write_authorized()
                return func(*args, **kwargs)
            except Exception as e2:
                if _is_local_api_denied(e2):
                    _pause_action(action, e2)
                if _is_connection_issue(e2):
                    _pause_action(action, e2)
                raise
        if _is_server_id_mismatch(e):
            zot.server_id = None
            return func(*args, **kwargs)
        if _is_connection_issue(e):
            _pause_action(action, e)
        raise


def _create_note_in_zotero(item_key: str, note_content: str, tags: List[str]) -> str | None:
    """在item_key下创建一条note,并按需打标签。返回新note的key,失败返回None。

    被 add_note_to_item 和 NoteCache.check_and_update_cache 共用,避免重复维护两份
    几乎一样的"建note -> 判断成功 -> 打标签"逻辑。
    """
    def _create():
        # 本地API没有 /items/new,不能用 zot.item_template("note") 拿模板
        # (会抛 CallDoesNotExistError),这里照 pyzotero 自己 mcp_server.py
        # 里手写note模板的方式,直接手动拼一个最小的note条目字典。
        note_item = {"itemType": "note", "note": note_content, "tags": [], "relations": {}}
        return zot.create_items([note_item], parentid=item_key)

    result = _perform_local_write(f"create_note('{item_key}')", _create)
    success = result.get("success", {})
    unchanged = result.get("unchanged", {})
    if not success and not unchanged:
        print(f"Note was not created for item {item_key}. Response: {result}")
        return None
    if success:
        note_key = result["successful"]["0"]["key"]
    else:
        note_key = zot.item(unchanged["0"])["key"]
    if tags:
        def _tag():
            note_full = zot.item(note_key)
            return zot.add_tags(note_full, *tags)
        _perform_local_write(f"tag_note('{note_key}')", _tag)
    return note_key


# 建立一个临时的sqlite数据库，用来缓存待提交到Zotero上的笔记信息，避免提交失败导致信息丢失
db_path = Path(get_config("workstation_dir")) / "note_cache.db"
db = peewee.SqliteDatabase(db_path)

class NoteCache(peewee.Model):
    item_key = peewee.CharField()
    note_content = peewee.TextField()
    tags = peewee.TextField()  # 用分号分隔的标签列表

    class Meta:
        database = db

    @classmethod
    def check_and_update_cache(cls) -> None:
        """
        检查数据库中是否有未提交的笔记,如果有, 尝试提交到Zotero, 并在成功后从数据库中删除。
        """
        unsent_notes = cls.select()
        for note in unsent_notes:
            try:
                tags = note.tags.split(";") if note.tags else []
                note_key = _create_note_in_zotero(note.item_key, note.note_content, tags)
                if note_key:
                    note.delete_instance()
            except Exception as e:
                print(f"Error occurred while submitting note from cache: {e}. Will try again later.")

# 在模块加载时：看看有没有这个数据库文件，如果没有就创建一个，如果有就连接它
if not db_path.exists():
    db.create_tables([NoteCache])
else:
    db.connect()
# 尝试提交一次缓存中的笔记，看看能不能成功。
NoteCache.check_and_update_cache()

def get_pdf_from_item(item_key: str, first: bool = True) -> Dict[str, bytes] | str:
    #print(f"Fetching PDF attachments for item {item_key}...")
    attachments = zot.children(item_key, format="keys", itemType="attachment")
    attachments = attachments.decode("utf-8").split("\n")
    attachments = [att for att in attachments if att]
    pdfs = {}
    for att_key in attachments:
        att_data = zot.item(att_key)
        if att_data["data"]["contentType"] == "application/pdf":
            pdfs[att_data['data']['title'].lower()] = att_key
    if first:
        #print(pdfs)
        # 按title "PDF" > "Full Text PDF" > others > "已提交版本"的顺序，返回第一个key
        if len(pdfs) == 0:
            raise Exception(f"No PDF attachment found for item {item_key}")
        if "PDF".lower() in pdfs:
            return pdfs["pdf"]
        elif "Full Text PDF".lower() in pdfs:
            return pdfs["full text pdf"]
        else:
            for key in pdfs:
                if key != "已提交版本".lower():
                    return pdfs[key]
            if "已提交版本".lower() in pdfs:
                return pdfs["已提交版本"]
        #print(pdfs)
    return pdfs

def dump_file_from_zotero(file_key: str, save_path:str|Path):
    """经由 Zotero API 的 /file 接口下载附件内容。
    不直接读 Zotero 的 storage 目录：那是 Zotero 的内部存储，布局随版本和同步方式变化，只通过 API 访问。"""
    f = zot.file(file_key)
    with open(save_path, "wb") as out:
        out.write(f)


def dump_file(file_key: str, save_path:str|Path, *args, **kwargs):
    dump_file_from_zotero(file_key, save_path)


def dump_pdf(pdf_key: str, save_path:str|Path, *args, **kwargs):
    dump_file(pdf_key, save_path, *args, **kwargs)

def check_md_full_text_exits(item_key: str) -> str|None:
    attachments = zot.children(item_key, format="keys", itemType="attachment")
    attachments = attachments.decode("utf-8").split("\n")
    attachments = [att for att in attachments if att]
    for att_key in attachments:
        att_data = zot.item(att_key)
        if att_data["data"]["title"] == "Full Text MD":
            return att_key
    return None

def dump_md(md_key: str, save_path:str|Path, *args, **kwargs):
    dump_file(md_key, save_path, *args, **kwargs)

def check_literature_exists(item_key: str, type_filter: str = "book || journalArticle || thesis || preprint") -> bool:
    item = zot.item(item_key, itemType=type_filter)
    return item is not None

def search_literature_by_doi(doi: str, type_filter: str = "book || journalArticle || thesis || preprint") -> list[str]:
    items = zot.everything(zot.items(itemType=type_filter))
    results = []
    for item in items:
        if item["data"].get("DOI", "").lower() == doi.lower():
            results.append(item["data"]["key"])
    return results

def upload_md_to_zotero(item_key: str, md_path: str|Path) -> str|None:
    if type(md_path) != Path:
        md_path = Path(md_path)
    existing_key = check_md_full_text_exits(item_key)
    if existing_key:
        # 已经有Full Text MD了，直接返回
        print(f"Item {item_key} already has a 'Full Text MD' attachment. Skipping upload.")
        return existing_key
    # 没有Full Text MD，上传新的。本地API写入后Zotero自己就会把文件落盘到storage目录、
    # 立刻反映在本地数据库里，不再需要额外手动把md_path拷贝进storage目录的那套补丁逻辑。
    #
    # 注意: 本地API没有 /items/new,zot.attachment_simple() 内部依赖 item_template()
    # 拿附件模板,在本地模式下会直接抛 CallDoesNotExistError。这里改成照 pyzotero
    # mcp_server.py 里手写附件模板的方式,自己拼一个"imported_file"附件字典,
    # 再用 upload_attachments() 上传——这个函数才是本地/远程都通用的底层写入方法。
    # 顺便把标题直接设成"Full Text MD",不用像以前那样先传再改名、多一次写入往返。
    attachment_item = {
        "itemType": "attachment",
        "linkMode": "imported_file",
        "title": "Full Text MD",
        "filename": str(md_path),
        "note": "",
        "tags": [],
        "relations": {},
    }
    upload_result = _perform_local_write(
        f"upload_md_to_zotero('{item_key}')", zot.upload_attachments, [attachment_item], item_key
    )
    new_items = upload_result.get("success") or upload_result.get("unchanged")
    if not new_items:
        print(f"No new attachment was created for item {item_key}. Response: {upload_result}")
        return None
    new_att_key = new_items[0]["key"]
    print(f"Successfully uploaded MD for item {item_key}. New attachment key: {new_att_key}")
    return new_att_key


def get_info(item_key: str, info_list: list[str]) -> Tuple:
    item = zot.item(item_key)
    data = item["data"]
    return tuple(data.get(info, None) for info in info_list)

def get_collections(root_collection_name: str) -> List[str]:
    # check the connection of zot, if connection is unstable, raise ZoteroActionPaused
    try:
        collections = zot.collections()  # returns a list of all collections in the library
        root_collection = None
        for item in collections:
            if item["data"]["name"] == root_collection_name:
                root_collection = item["data"]
                break
        assert root_collection is not None, f"Collection '{root_collection_name}' not found"
        related_collections = [root_collection["key"]]
        if root_collection:
            sub_collections = zot.collections_sub(root_collection["key"], format="keys")
            sub_collections = sub_collections.decode("utf-8").split("\n")
            sub_collections = [col for col in sub_collections if col]
            if len(sub_collections) == 100:
                while True:
                    more_sub_cols = zot.collections_sub(
                        root_collection["key"], format="keys", start=len(related_collections)
                    )
                    more_sub_cols = more_sub_cols.decode("utf-8").split("\n")
                    more_sub_cols = [col for col in more_sub_cols if col]
                    if not more_sub_cols:
                        break
                    related_collections += more_sub_cols
                    if len(more_sub_cols) < 100:
                        break
            related_collections += sub_collections
        related_collections = [col for col in related_collections if col]
        return related_collections
    except Exception as e:
        if _is_connection_issue(e):
            _pause_action(f"get_collections('{root_collection_name}')", e)
        raise

def fetch_keys_by_collection(collection_name: str) -> list[str]:
    try:
        related_collections = get_collections(collection_name)
        keys = []
        for col_key in related_collections:
            items = zot.collection_items(col_key, format="keys", itemType="book || journalArticle || thesis || preprint || bookSection")
            items = items.decode("utf-8").split("\n")
            items = [item for item in items if item]
            if len(items) == 100:
                while True:
                    more_items = zot.collection_items(col_key, format="keys", itemType="book || journalArticle || thesis || preprint || bookSection", start=len(keys))
                    more_items = more_items.decode("utf-8").split("\n")
                    more_items = [item for item in more_items if item]
                    if not more_items:
                        break
                    keys += more_items
                    if len(more_items) < 100:
                        break
            keys += items
        keys = list(set(keys))
        keys = [key for key in keys if key]
        return keys
    except Exception as e:
        if _is_connection_issue(e):
            _pause_action(f"fetch_keys_by_collection('{collection_name}')", e)
        raise

def fetch_ref_keys_by_collection(collection_key: str) -> list[str]:
    try:
        items = zot.collection_items(collection_key, format="keys", itemType="book || journalArticle || thesis || preprint")
        items = items.decode("utf-8").split("\n")
        items = [item for item in items if item]
        if len(items) == 100:
            while True:
                more_items = zot.collection_items(collection_key, format="keys", itemType="book || journalArticle || thesis || preprint", start=len(items))
                more_items = more_items.decode("utf-8").split("\n")
                more_items = [item for item in more_items if item]
                if not more_items:
                    break
                items += more_items
                if len(more_items) < 100:
                    break
        return items
    except Exception as e:
        if _is_connection_issue(e):
            _pause_action(f"fetch_ref_keys_by_collection('{collection_key}')", e)
        raise

def _find_collection_key(collection_name: str) -> str:
    related_collections = zot.collections(q=collection_name)
    for col in related_collections or []:
        if col["data"]["name"] == collection_name:
            return col["data"]["key"]
    raise Exception(f"Collection '{collection_name}' not found")

def add_item_to_collection(item_key: str,collection_name: str) -> bool:
    collection_key = _find_collection_key(collection_name)
    item = zot.item(item_key)
    return _perform_local_write(
        f"add_item_to_collection('{item_key}','{collection_name}')",
        zot.addto_collection, collection_key, item,
    )

def rm_item_from_collection(item_key: str,collection_name: str) -> bool:
    collection_key = _find_collection_key(collection_name)
    item = zot.item(item_key)
    return _perform_local_write(
        f"rm_item_from_collection('{item_key}','{collection_name}')",
        zot.deletefrom_collection, collection_key, item,
    )

def retreive_note_keys_from_item(item_key: str, filter_tag: List[str], filter_logic: str = "and") -> List[str]|None:
    attachments = zot.children(item_key, format="keys", itemType="note")
    attachments = attachments.decode("utf-8").split("\n")
    attachments = [att for att in attachments if att]
    note_keys = []
    for att_key in attachments:
        att_data = zot.item(att_key)
        tags = [tag["tag"] for tag in att_data["data"].get("tags", [])]
        if filter_logic == "and":
            if all(tag in tags for tag in filter_tag):
                note_keys.append(att_key)
        elif filter_logic == "or":
            if any(tag in tags for tag in filter_tag):
                note_keys.append(att_key)
        else:
            raise ValueError(f"Invalid filter_logic: {filter_logic}. Use 'and' or 'or'.")
    if note_keys:
        return note_keys
    return None

def retreive_note_from_item(item_key: str,filter_tag: List[str],filter_logic: str="and") -> str|None:
    attachments = zot.children(item_key, format="keys", itemType="note")
    attachments = attachments.decode("utf-8").split("\n")
    attachments = [att for att in attachments if att]
    note_context = None
    for att_key in attachments:
        att_data = zot.item(att_key)
        tags = [tag["tag"] for tag in att_data["data"].get("tags", [])]
        if filter_logic == "and":
            if all(tag in tags for tag in filter_tag):
                note_context = att_data["data"]["note"]
                break
        elif filter_logic == "or":
            if any(tag in tags for tag in filter_tag):
                note_context = att_data["data"]["note"]
                break
        else:
            raise ValueError(f"Invalid filter_logic: {filter_logic}. Use 'and' or 'or'.")
    if note_context is not None:
        # check if note_context is in HTML format, if so, convert it to markdown
        if "<p>" in note_context or "<div>" in note_context:
            note_context = md(note_context)
        return note_context.strip()
    return None

def add_note_to_item(item_key: str, note_content: str, tags: List[str] = []) -> str|None:
    # 先新建缓存，避免提交失败导致笔记内容丢失
    cache_obj = NoteCache.create(item_key=item_key, note_content=note_content, tags=";".join(tags))
    cache_obj.save()
    try:
        note_key = _create_note_in_zotero(item_key, note_content, tags)
    except Exception as e:
        note_key = None
        print(f"Error occurred while creating note: {e}")
    if note_key:
        # 提交成功后从缓存中删除
        cache_obj.delete_instance()
    else:
        print("Note was not created; it stays cached and will be retried on next startup.")
    return note_key

def bib_export(item_key: str,style: str = "earth-and-planetary-science-letters") -> str:
    bib_return = md(zot.item(item_key, format="bib", style=style).decode("utf-8"))
    return bib_return.replace("xml version=\"1.0\"?", "").strip()

def note_export(collection_name: str, dump_to_file_name: str, filter_tag: List[str], filter_logic: str = "and") -> None:
    related_collections = get_collections(collection_name)
    item_keys = []
    for col_key in related_collections:
        items = zot.collection_items(col_key, format="keys", itemType="book || journalArticle || thesis || preprint")
        items = items.decode("utf-8").split("\n")
        items = [item for item in items if item]
        item_keys += items
    item_keys = list(set(item_keys))
    with open(dump_to_file_name, "w", encoding="utf-8") as f:
        for item_key in item_keys:
            note_content = retreive_note_from_item(item_key, filter_tag, filter_logic)
            if note_content:
                f.write(f"**Reference:** {bib_export(item_key)}\n\n")
                f.write(f"**Note:**\n\n")
                f.write(f"```markdown\n")
                # 剔除note中**Label**以后的内容
                if "**Labels**" in note_content:
                    note_content = note_content.split("**Label**")[0].strip()
                f.write(note_content)
                f.write(f"\n```\n\n")

def get_notes_by_tags(filter_tag: List[str]|str, filter_logic: str = "and") -> str:
    if isinstance(filter_tag, str):
        filter_tag = [filter_tag]
    if filter_logic == "and":
        tag_query = filter_tag
    elif filter_logic == "or":
        tag_query = " || ".join(filter_tag)
    else:
        raise ValueError(f"Invalid filter_logic: {filter_logic}. Use 'and' or 'or'.")
    note_request= zot.items(itemType="note", tag=tag_query)
    #print(note_request[0])
    # get all note items, make sure to re-request if there are more than 100 items
    note_items = []
    if note_request:
        note_items = note_request
        note_items = [item["data"] for item in note_items if item]
        if len(note_items) == 100:
            while True:
                more_note_request = zot.items(itemType="note", tags=tag_query, start=len(note_items))
                more_note_request = more_note_request.decode("utf-8").split("\n")
                more_note_request = [item for item in more_note_request if item]
                if not more_note_request:
                    break
                note_items += more_note_request
                if len(more_note_request) < 100:
                    break
    parent_key_2_note = {}
    num_parent_key_notes = {}
    for note_item in note_items:
        parent_key = note_item.get("parentItem", None)
        if parent_key:
            note_content = note_item.get("note", "")
            # check if note_content is in HTML format, if so, convert it to markdown
            if "<p>" in note_content or "<div>" in note_content:
                note_content = md(note_content)
            if parent_key not in parent_key_2_note:
                parent_key_2_note[parent_key] = note_content.strip()
                num_parent_key_notes[parent_key] = 1
            if parent_key in parent_key_2_note:
                num_parent_key_notes[parent_key] += 1
                # 如果同一parent_key下有多个note，随机选择一个note的内容作为
                # 置随机数，保证同一parent_key下不同的note被选中的概率相同
                if random.random() < 1 / num_parent_key_notes[parent_key]:
                    parent_key_2_note[parent_key] = note_content.strip()
    # 拼接notes内容，格式如下:
    # Paper: epsl style reference
    # Note:
    #
    # ```markdown
    # note content
    # ```
    return_note = []
    for parent_key, note_content in parent_key_2_note.items():
        try:
            bib_info = bib_export(parent_key)
            return_note.append(f"**Reference:** {bib_info}\n\n")
            return_note.append(f"**Note:**\n\n")
            return_note.append(f"```markdown\n{note_content}\n```\n\n")
        except Exception as e:
            print(f"Error occurred while exporting bib for item {parent_key}: {e}")
            continue
    return '\n'.join(return_note)


def _label_notes_in_collections(collection_name: str):
    """
    A debug function, to label all the notes under the collection with tags "AI Notes", "Simple Notes" if they don't have such tags.
    """
    related_collections = get_collections(collection_name)
    for col_key in related_collections:
        note_items = zot.collection_items(col_key, format="keys", itemType="note")
        note_items = note_items.decode("utf-8").split("\n")
        note_items = [item for item in note_items if item]
        for note_key in note_items:
            tags = zot.item_tags(note_key)
            if "AI Notes" in tags and "Simple Notes" in tags:
                continue
            tags_to_add = [t for t in ("AI Notes", "Simple Notes") if t not in tags]
            note = zot.item(note_key)
            _perform_local_write(f"_label_notes_in_collections('{note_key}')", zot.add_tags, note, *tags_to_add)

def re_label_note_tags(note_key: str, new_tags: List[str]) -> bool:
    note = zot.item(note_key)
    note["data"]["tags"] = [{"tag": tag} for tag in new_tags]
    try:
        result = _perform_local_write(f"re_label_note_tags('{note_key}')", zot.update_item, note)
        if result:
            print(f"Successfully updated tags for note {note_key}. New tags: {new_tags}")
            return True
        else:
            print(f"Failed to update tags for note {note_key}. Response: {result}")
            return False
    except Exception as e:
        print(f"Error occurred while updating tags for note {note_key}: {e}")
        return False

def insert_reference_to_collection(collection_name: str, type: str = "journalArticle", journal: str|None = None, author: str|None = None, doi: str|None = None, title: str|None = None) -> str|None:
    # 定义一个静态变量，如果上次调用时的collection_name与这次相同，就不再次调用时重新查询collection_key，直接使用上次查询到的collection_key
    if not hasattr(insert_reference_to_collection, "last_collection_name"):
        insert_reference_to_collection.last_collection_name = None #type: ignore
        insert_reference_to_collection.last_collection_key = None #type: ignore
        insert_reference_to_collection.collection_items = None #type: ignore
    if collection_name != insert_reference_to_collection.last_collection_name: #type: ignore
        collection_key = _find_collection_key(collection_name)
        insert_reference_to_collection.last_collection_name = collection_name #type: ignore
        insert_reference_to_collection.last_collection_key = collection_key #type: ignore
        insert_reference_to_collection.collection_items = fetch_ref_keys_by_collection(collection_key) #type: ignore
    else:
        collection_key = insert_reference_to_collection.last_collection_key #type: ignore
    collection_items = insert_reference_to_collection.collection_items #type: ignore
    # 如果doi不为None,有限根据doi检查文献是否已经存在，如果存在，就直接把它加入collection，如果不存在，就创建一个新的item，并加入collection
    assert collection_items is not None, "collection_items should not be None"
    if doi:
        existing_items = search_literature_by_doi(doi)
        if existing_items:
            item_key = existing_items[0]
            # 先检查这个item是否已经在collection里了，如果没有，就加入collection
            if item_key not in collection_items:
                _perform_local_write(
                    f"insert_reference_to_collection.addto_collection('{item_key}')",
                    zot.addto_collection, collection_key, zot.item(item_key),
                )
            return item_key
    #没有doi号或者根据doi没有找到文献，就创建一个新的item，并加入collection
    # 本地API没有 /items/new,不能用 zot.item_template(type) 拿模板(会抛
    # CallDoesNotExistError),这里手动拼一个最小的条目字典交给 create_items——
    # 服务端本来就只按itemType接受该类型认识的字段,不需要模板里那些空字段占位。
    new_item = {"itemType": type, "tags": [], "creators": [], "relations": {}}
    if journal:
        new_item["publicationTitle"] = journal
    if author:
        new_item["creators"] = [{"creatorType": "author", "name": author}]
    if title:
        new_item["title"] = title
    if doi:
        new_item["DOI"] = doi
    try:
        result = _perform_local_write(
            f"insert_reference_to_collection.create_items('{collection_name}')",
            zot.create_items, [new_item],
        )
        sucess = result.get("success", {})
        #print(sucess)
        if len(sucess.keys()) == 0:
            print(f"Failed to create new item for collection {collection_name}. Response: {result}")
            return None
        new_item_key = sucess['0']
        new_item = zot.item(new_item_key)
        print(f"Successfully created new item with key {new_item_key} and added to collection {collection_name}")
        # add the new item to collection
        _perform_local_write(
            f"insert_reference_to_collection.addto_collection('{new_item_key}')",
            zot.addto_collection, collection_key, new_item,
        )
        return new_item_key
    except Exception as e:
        print(f"Error occurred while creating new item for collection {collection_name}: {e}")
        return None

if __name__ == "__main__":
    pass
