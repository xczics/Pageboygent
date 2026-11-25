"""MinerU 官方 API 的 PDF->MD 任务队列与独立 worker 进程。

- 主流程（heartbeat）只调用 enqueue() 往 mineru_tasks.db 插一条任务，不做任何网络请求。
- worker 进程（`pageboygent mineru-worker`，由 serve_storage 自动拉起）负责：
  申请上传链接 -> PUT 上传 PDF -> 批量轮询 -> 下载结果 zip -> 把 full.md 写到
  workstation_dir/out_mds/{pdf_key}.md。之后由 heartbeat 现有的 MDStatus 流程接手
  （上传 Zotero、进入 RAG）。
- worker 只写 mineru_tasks.db 和文件，不碰 pageboygent.db；mineru_tasks.db 与 pageboygent.db
  刻意分开，避免两个进程写同一个库。所有状态迁移都带 WHERE state 条件，Web 端操作与 worker 并发安全。

API 文档：https://mineru.net/apiManage/docs
"""
from __future__ import annotations

import base64
from datetime import datetime, timedelta
import fcntl
import hmac
import io
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import signal
import threading
import time
import urllib.parse
from typing import Any, Callable, Iterable
import zipfile

import peewee
import requests

from .config import get_config

_workstation = Path(get_config("workstation_dir")).expanduser()
_out_md_dir = _workstation / "out_mds"
_human_upload_dir = _workstation / "human_upload_pdfs"

_db = peewee.SqliteDatabase(
    str(_workstation / "mineru_tasks.db"),
    pragmas={"journal_mode": "wal", "busy_timeout": 10000},
)

# 任务状态（本地）
STATE_QUEUED = "queued"          # 等待 worker 上传
STATE_UPLOADING = "uploading"    # worker 已领取，正在申请链接/上传
STATE_SUBMITTED = "submitted"    # 已上传，等待 MinerU 解析
STATE_DONE = "done"              # full.md 已写入 out_mds
STATE_FAILED = "failed"          # 超过重试次数或不可重试的错误
STATE_MANUAL = "manual"          # 已转为手动流程（PDF 已移到 human_upload_pdfs）
STATES = (STATE_QUEUED, STATE_UPLOADING, STATE_SUBMITTED, STATE_DONE, STATE_FAILED, STATE_MANUAL)

MAX_FILE_BYTES = 200 * 1024 * 1024   # 官方限制 200MB
MAX_BATCH_FILES = 50                 # 单次申请上传链接最多 50 个
TASK_TIMEOUT = timedelta(hours=24)   # 上传链接有效期 24h，超过视为卡死
_BATCH_NOT_FOUND_CODE = "-60012"     # task not found or expire
_BATCH_NOT_FOUND_GRACE = timedelta(minutes=10)

# 鉴权失败（token 错误/过期）：不消耗任务重试次数，worker 降低频率等待 token 更新
_AUTH_ERROR_CODES = {"A0202", "A0211"}


class MinerUTask(peewee.Model):
    pdf_key = peewee.CharField(unique=True)
    paper_title = peewee.TextField(default="")
    pdf_path = peewee.TextField()
    state = peewee.CharField(default=STATE_QUEUED, index=True, choices=[(s, s) for s in STATES])
    remote_state = peewee.CharField(null=True)        # MinerU 返回的 state：pending/running/converting/...
    batch_id = peewee.CharField(null=True, index=True)
    model_version = peewee.CharField(null=True)
    extracted_pages = peewee.IntegerField(null=True)
    total_pages = peewee.IntegerField(null=True)
    attempts = peewee.IntegerField(default=0)
    err_msg = peewee.TextField(null=True)
    created_at = peewee.DateTimeField(default=datetime.now)
    submitted_at = peewee.DateTimeField(null=True)
    finished_at = peewee.DateTimeField(null=True)
    updated_at = peewee.DateTimeField(default=datetime.now)

    class Meta:
        database = _db
        table_name = "mineru_task"


def init_db() -> None:
    with _db.connection_context():
        _db.create_tables([MinerUTask], safe=True)


init_db()


def _transition(task_id: int, from_states: Iterable[str], **fields: Any) -> bool:
    """带条件的状态迁移：仅当任务仍处于 from_states 之一时才更新，返回是否成功。"""
    fields.setdefault("updated_at", datetime.now())
    rows = (
        MinerUTask.update(**fields)
        .where(MinerUTask.id == task_id, MinerUTask.state.in_(list(from_states)))
        .execute()
    )
    return rows == 1


# ---------------------------------------------------------------------------
# 主流程 / Web 端 API
# ---------------------------------------------------------------------------

def enqueue(pdf_key: str, pdf_path: str, paper_title: str = "") -> MinerUTask:
    """提交一个 PDF->MD 任务（幂等：同一 pdf_key 已存在则直接返回已有任务）。"""
    task, created = MinerUTask.get_or_create(
        pdf_key=pdf_key,
        defaults={"pdf_path": pdf_path, "paper_title": paper_title},
    )
    if created:
        print(f"[mineru] queued PDF->MD task for '{paper_title or pdf_key}'.")
    return task


def retry(task_ids: Iterable[int]) -> int:
    """把 failed 任务重新放回队列（重置重试次数）。"""
    count = 0
    for task_id in task_ids:
        if _transition(task_id, [STATE_FAILED], state=STATE_QUEUED, attempts=0, err_msg=None,
                       batch_id=None, remote_state=None):
            count += 1
    return count


def fallback_to_manual(task_ids: Iterable[int]) -> int:
    """把 queued/failed 任务转为手动流程：PDF 移到 human_upload_pdfs，由 check_pdf2md_tasks 接手。"""
    _human_upload_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for task_id in task_ids:
        task = MinerUTask.get_or_none(MinerUTask.id == task_id)
        if task is None:
            continue
        # 先抢占状态，防止 worker 同时领取上传
        if not _transition(task_id, [STATE_QUEUED, STATE_FAILED], state=STATE_MANUAL):
            continue
        src = Path(str(task.pdf_path))
        target = _human_upload_dir / f"{task.pdf_key}.pdf"
        try:
            if src.exists() and src.resolve() != target.resolve():
                shutil.move(str(src), str(target))
            MinerUTask.update(pdf_path=str(target), updated_at=datetime.now()).where(MinerUTask.id == task_id).execute()
            count += 1
        except Exception as exc:
            _transition(task_id, [STATE_MANUAL], state=STATE_FAILED, err_msg=f"fallback to manual failed: {exc}")
    return count


# ---------------------------------------------------------------------------
# MinerU API 客户端
# ---------------------------------------------------------------------------

class MinerUError(Exception):
    def __init__(self, message: str, code: str | None = None, http_status: int | None = None):
        super().__init__(message)
        self.code = code
        self.http_status = http_status

    @property
    def is_auth(self) -> bool:
        return self.http_status == 401 or self.code in _AUTH_ERROR_CODES


_OPENXLAB_AUTH_BASE = "https://openapi.openxlab.org.cn/api/v1/sso-be/api/v1/open/"


class OpenXLabJwtProvider:
    """用 OpenXLab AK/SK 换取 MinerU 可用的短期 JWT（约 1 小时），过期前自动重新换取。

    流程与官方 openxlab SDK 一致：POST auth {ak} 得到 nonce/algorithm，
    用 SK 对 nonce 做 HMAC 并 base64 得到 d，再 POST getJwt {ak, d}。
    """

    REFRESH_MARGIN = timedelta(minutes=5)

    def __init__(self, access_key: str, secret_key: str, session: Any = None, auth_base: str = _OPENXLAB_AUTH_BASE):
        self.access_key = access_key.strip()
        self.secret_key = secret_key.strip()
        self.session = session or requests.Session()
        self.auth_base = auth_base
        self._jwt: str | None = None
        self._expires_at: datetime | None = None

    def _post(self, path: str, payload: dict) -> dict:
        resp = self.session.post(self.auth_base + path, json=payload,
                                 headers={"Content-Type": "application/json", "accept": "application/json"}, timeout=30)
        try:
            body = resp.json()
        except ValueError:
            raise MinerUError(f"OpenXLab {path}: HTTP {resp.status_code}", http_status=401 if resp.status_code in (401, 403) else resp.status_code)
        inner = body.get("data") or {}
        if body.get("msgCode") != "10000" or inner.get("msgCode") != "10000":
            # AK/SK 被拒视为鉴权错误：不消耗任务重试次数
            raise MinerUError(f"OpenXLab {path} rejected: {inner.get('msg') or body.get('msg')}", http_status=401)
        return inner["data"]

    def _fetch(self) -> None:
        auth = self._post("auth", {"ak": self.access_key})
        algorithm = str(auth["algorithm"])
        digest = algorithm[4:] if algorithm.startswith("Hmac") else algorithm
        d = base64.b64encode(hmac.new(self.secret_key.encode(), str(auth["nonce"]).encode(), digest).digest()).decode()
        jwt = str(self._post("getJwt", {"ak": self.access_key, "d": d})["jwt"])
        payload = jwt.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        self._jwt = jwt
        self._expires_at = datetime.fromtimestamp(int(claims["exp"]))

    def __call__(self) -> str:
        if self._jwt is None or self._expires_at is None or datetime.now() >= self._expires_at - self.REFRESH_MARGIN:
            self._fetch()
        assert self._jwt is not None
        return self._jwt

    def invalidate(self) -> None:
        self._jwt = None


_OSS_REGIONAL_HOST = re.compile(r"^(?P<bucket>[^.]+)\.oss-[a-z0-9-]+\.aliyuncs\.com$")


def _accelerated_oss_url(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    m = _OSS_REGIONAL_HOST.match(parts.netloc)
    if not m or parts.netloc.startswith(f"{m['bucket']}.oss-accelerate"):
        return url
    return urllib.parse.urlunsplit((parts.scheme, f"{m['bucket']}.oss-accelerate.aliyuncs.com", parts.path, parts.query, parts.fragment))


# 每种鉴权方式（"static" / "jwt"）上次上传成功时是否带了 Content-Type；进程内共享（worker 每轮都会重建 client）
_upload_style_cache: dict[str, bool] = {}


class MinerUClient:
    def __init__(self, token: str | Callable[[], str], base_url: str, model_version: str, session: Any = None, timeout: float = 60):
        self.base_url = base_url.rstrip("/")
        self.model_version = model_version
        self.session = session or requests.Session()
        self.timeout = timeout
        # token 可以是固定字符串，也可以是 OpenXLabJwtProvider 这类每次调用返回当前有效 JWT 的函数
        self._token = token

    def _headers(self) -> dict:
        token = self._token() if callable(self._token) else self._token
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def _call(self, method: str, path: str, **kwargs) -> Any:
        resp = self.session.request(method, f"{self.base_url}{path}", headers=self._headers(), timeout=self.timeout, **kwargs)
        try:
            body = resp.json()
        except ValueError:
            raise MinerUError(f"HTTP {resp.status_code}: {resp.text[:200]}", http_status=resp.status_code)
        code = body.get("code")
        if resp.status_code != 200 or code not in (0, "0"):
            raise MinerUError(
                f"HTTP {resp.status_code} code={code} msg={body.get('msg')}",
                code=None if code is None else str(code),
                http_status=resp.status_code,
            )
        return body.get("data") or {}

    def request_upload_urls(self, files: list[dict]) -> tuple[str, list[str]]:
        data = self._call("POST", "/file-urls/batch", json={"files": files, "model_version": self.model_version})
        return str(data["batch_id"]), list(data["file_urls"])

    def upload(self, url: str, path: Path) -> None:
        # 跨洲直连上海 OSS 丢包严重（实测约 6KB/s），默认先走 OSS 全球加速域名（实测快约 10 倍）。
        # OSS V1 预签名只签 bucket/对象名、不含域名，换域名签名仍有效；加速失败再回退原链接。
        targets = [url]
        if get_config("mineru_upload_accelerate"):
            accelerated = _accelerated_oss_url(url)
            if accelerated != url:
                targets.insert(0, accelerated)
        for i, target in enumerate(targets):
            try:
                self._upload_to(target, path)
                return
            except MinerUError as exc:
                if i == len(targets) - 1:
                    raise
                print(f"[mineru] accelerated upload failed for {path.name}, falling back to regional endpoint: {exc}", flush=True)

    def _upload_to(self, url: str, path: Path) -> None:
        # 预签名 URL 是否签了 Content-Type 取决于鉴权方式（实测 2026-10）：
        # - 静态 API token：按文档不带 Content-Type 才能上传，带了反而 403；
        # - AK/SK 换来的 JWT：必须带 application/pdf，不带则 403 SignatureDoesNotMatch。
        # 先试上次成功的方式（按鉴权方式记住，避免每个文件都白传一遍），签名不匹配时
        # （OSS 不会接收文件，URL 可复用）再换另一种。
        content_type = mimetypes.guess_type(path.name)[0] or "application/pdf"
        style_key = "jwt" if callable(self._token) else "static"
        without_ct, with_ct = {}, {"Content-Type": content_type}
        candidates = [with_ct, without_ct] if _upload_style_cache.get(style_key) else [without_ct, with_ct]
        resp = None
        last_exc: Exception | None = None
        for headers in candidates:
            try:
                with open(path, "rb") as fh:
                    resp = self.session.put(url, data=fh, headers=headers, timeout=max(self.timeout, 600))
            except requests.ConnectionError as exc:
                # 签名不匹配时 OSS 会在请求体传完前直接断开连接（大文件常见 BrokenPipe），按签名不匹配处理
                last_exc, resp = exc, None
                continue
            if resp.status_code in (200, 201, 204):
                _upload_style_cache[style_key] = "Content-Type" in headers
                return
            if not (resp.status_code == 403 and "SignatureDoesNotMatch" in resp.text):
                break
        if resp is None:
            raise MinerUError(f"upload failed: {last_exc}")
        raise MinerUError(f"upload HTTP {resp.status_code}: {resp.text[:200]}", http_status=resp.status_code)

    def batch_results(self, batch_id: str) -> list[dict]:
        data = self._call("GET", f"/extract-results/batch/{batch_id}")
        return list(data.get("extract_result") or [])

    def download_full_md(self, zip_url: str) -> str:
        resp = self.session.get(zip_url, timeout=max(self.timeout, 600))
        if resp.status_code != 200:
            raise MinerUError(f"download HTTP {resp.status_code}", http_status=resp.status_code)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = [n for n in zf.namelist() if n == "full.md" or n.endswith("/full.md")]
            if not names:
                raise MinerUError("full.md not found in result zip")
            return zf.read(sorted(names, key=len)[0]).decode("utf-8")


_jwt_provider: OpenXLabJwtProvider | None = None
# 被 MinerU 拒绝过的静态 token（过期/失效）；配置里换成新 token 后自动恢复优先使用
_rejected_static_token: str | None = None


def _client_from_config(session: Any = None) -> MinerUClient | None:
    """优先用固定的 mineru_token；它缺失或被拒绝后，退回 AK/SK（自动换取并刷新短期 JWT）。

    AK/SK 换 JWT 依赖 openapi.openxlab.org.cn，实测偶有超时，所以只作为后备。
    """
    global _jwt_provider
    static_token = str(get_config("mineru_token") or "").strip()
    access_key, secret_key = get_config("mineru_access_key"), get_config("mineru_secret_key")
    token: str | Callable[[], str]
    if static_token and static_token != _rejected_static_token:
        token = static_token
    elif access_key and secret_key:
        if _jwt_provider is None or (_jwt_provider.access_key, _jwt_provider.secret_key) != (str(access_key).strip(), str(secret_key).strip()):
            _jwt_provider = OpenXLabJwtProvider(str(access_key), str(secret_key))
        token = _jwt_provider
    elif static_token:
        token = static_token  # 没有后备凭据：继续用（可能已失效的）token，等用户更新配置
    else:
        return None
    return MinerUClient(
        token=token,
        base_url=str(get_config("mineru_api_base")),
        model_version=str(get_config("mineru_model_version")),
        session=session,
    )


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------

def _fail_or_requeue(task: MinerUTask, from_states: Iterable[str], err: str) -> None:
    attempts = int(task.attempts) + 1  # type: ignore[arg-type]
    max_attempts = int(get_config("mineru_max_attempts"))
    if attempts >= max_attempts:
        _transition(task.id, from_states, state=STATE_FAILED, attempts=attempts, err_msg=err, finished_at=datetime.now())
        print(f"[mineru] task failed for '{task.paper_title or task.pdf_key}': {err}")
    else:
        _transition(task.id, from_states, state=STATE_QUEUED, attempts=attempts, err_msg=err, batch_id=None, remote_state=None)
        print(f"[mineru] task for '{task.paper_title or task.pdf_key}' will retry ({attempts}/{max_attempts}): {err}")


def submit_queued(client: MinerUClient) -> int:
    """领取 queued 任务，申请上传链接并上传。返回成功上传的数量。"""
    candidates = list(MinerUTask.select().where(MinerUTask.state == STATE_QUEUED).order_by(MinerUTask.created_at).limit(MAX_BATCH_FILES))
    claimed: list[MinerUTask] = []
    for task in candidates:
        path = Path(str(task.pdf_path))
        if not path.exists():
            _transition(task.id, [STATE_QUEUED], state=STATE_FAILED, err_msg=f"PDF not found: {path}", finished_at=datetime.now())
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            _transition(task.id, [STATE_QUEUED], state=STATE_FAILED, err_msg="PDF larger than 200MB (MinerU API limit)", finished_at=datetime.now())
            continue
        if _transition(task.id, [STATE_QUEUED], state=STATE_UPLOADING):
            claimed.append(task)
    if not claimed:
        return 0

    files = [{"name": f"{t.pdf_key}.pdf", "data_id": str(t.pdf_key)} for t in claimed]
    try:
        batch_id, urls = client.request_upload_urls(files)
    except Exception:
        # 申请链接失败（网络/鉴权/配额）：整批放回队列，不消耗重试次数，交给上层退避
        for t in claimed:
            _transition(t.id, [STATE_UPLOADING], state=STATE_QUEUED)
        raise

    uploaded = 0
    for task, url in zip(claimed, urls):
        try:
            client.upload(url, Path(str(task.pdf_path)))
        except Exception as exc:
            _fail_or_requeue(task, [STATE_UPLOADING], f"upload failed: {exc}")
            continue
        if _transition(task.id, [STATE_UPLOADING], state=STATE_SUBMITTED, batch_id=batch_id, remote_state="pending",
                       model_version=client.model_version, submitted_at=datetime.now(), err_msg=None):
            uploaded += 1
    print(f"[mineru] uploaded {uploaded}/{len(claimed)} PDF(s) in batch {batch_id}.")
    return uploaded


def _write_md_atomically(pdf_key: str, content: str) -> Path:
    _out_md_dir.mkdir(parents=True, exist_ok=True)
    target = _out_md_dir / f"{pdf_key}.md"
    tmp = target.with_name(f".{target.name}.tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, target)  # 原子替换，heartbeat 不会读到写了一半的文件
    return target


def poll_submitted(client: MinerUClient) -> int:
    """查询所有 submitted 任务的批量结果；完成的下载 full.md。返回本轮完成的数量。"""
    submitted = list(MinerUTask.select().where(MinerUTask.state == STATE_SUBMITTED))
    by_batch: dict[str, list[MinerUTask]] = {}
    for task in submitted:
        by_batch.setdefault(str(task.batch_id), []).append(task)

    finished = 0
    for batch_id, tasks in by_batch.items():
        try:
            raw_results = client.batch_results(batch_id)
        except MinerUError as exc:
            if exc.code != _BATCH_NOT_FOUND_CODE:
                raise
            # 刚上传完的批次短时间内会返回“task not found”（实测几秒后即可查到）；超过宽限期仍查不到才算失败
            for task in tasks:
                submitted_at = task.submitted_at or task.updated_at
                if submitted_at and datetime.now() - submitted_at > _BATCH_NOT_FOUND_GRACE:  # type: ignore[operator]
                    _fail_or_requeue(task, [STATE_SUBMITTED], f"MinerU batch not found: {exc}")
            continue
        # 实测批量结果不回传 data_id，按 file_name（上传时为 {pdf_key}.pdf）匹配
        results = {str(r.get("data_id") or Path(str(r.get("file_name", ""))).stem): r for r in raw_results}
        for task in tasks:
            result = results.get(str(task.pdf_key))
            if result is None:
                continue
            state = str(result.get("state") or "")
            progress = result.get("extract_progress") or {}
            if state == "done":
                try:
                    md = client.download_full_md(str(result["full_zip_url"]))
                    path = _write_md_atomically(str(task.pdf_key), md)
                except Exception as exc:
                    _fail_or_requeue(task, [STATE_SUBMITTED], f"download result failed: {exc}")
                    continue
                if _transition(task.id, [STATE_SUBMITTED], state=STATE_DONE, remote_state=state, err_msg=None, finished_at=datetime.now()):
                    finished += 1
                    print(f"[mineru] PDF->MD done for '{task.paper_title or task.pdf_key}': {path}")
            elif state == "failed":
                _fail_or_requeue(task, [STATE_SUBMITTED], f"MinerU: {result.get('err_msg') or 'failed'}")
            else:
                submitted_at = task.submitted_at or task.updated_at
                if submitted_at and datetime.now() - submitted_at > TASK_TIMEOUT:  # type: ignore[operator]
                    _fail_or_requeue(task, [STATE_SUBMITTED], f"timed out in remote state '{state}'")
                    continue
                MinerUTask.update(
                    remote_state=state,
                    extracted_pages=progress.get("extracted_pages"),
                    total_pages=progress.get("total_pages"),
                    updated_at=datetime.now(),
                ).where(MinerUTask.id == task.id, MinerUTask.state == STATE_SUBMITTED).execute()
    return finished


def _on_auth_failure(client: MinerUClient) -> None:
    """鉴权失败：静态 token 标记为已拒绝（下一轮改用 AK/SK）；JWT 则丢弃缓存，下一轮重新换取。"""
    global _rejected_static_token
    if callable(client._token):
        if _jwt_provider is not None:
            _jwt_provider.invalidate()
    elif _rejected_static_token != client._token:
        _rejected_static_token = client._token
        print("[mineru] mineru_token was rejected (expired?); falling back to mineru_access_key/mineru_secret_key if configured.", flush=True)


def _recover_interrupted() -> None:
    # 上次 worker 在上传途中退出：上传链接对应的批次作废，重新排队即可
    MinerUTask.update(state=STATE_QUEUED, updated_at=datetime.now()).where(MinerUTask.state == STATE_UPLOADING).execute()


def _acquire_single_instance_lock():
    lock_path = _workstation / "mineru_worker.lock"
    fh = open(lock_path, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        return None
    fh.write(str(os.getpid()))
    fh.flush()
    return fh


def run_worker(stop_event: threading.Event | None = None, parent_pid: int | None = None) -> int:
    """mineru-worker 主循环。parent_pid 给定时，父进程退出后 worker 自动退出。"""
    stop_event = stop_event or threading.Event()
    lock = _acquire_single_instance_lock()
    if lock is None:
        print("[mineru] another mineru-worker is already running; exiting.", flush=True)
        return 1

    def _on_signal(signum, _frame):
        stop_event.set()

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, _on_signal)
        signal.signal(signal.SIGINT, _on_signal)

    interval = int(get_config("mineru_poll_interval_seconds"))
    _recover_interrupted()
    print(f"[mineru] worker started (pid={os.getpid()}, interval={interval}s, model={get_config('mineru_model_version')}).", flush=True)
    backoff = interval
    last_auth_warning = 0.0
    try:
        while not stop_event.is_set():
            if parent_pid is not None and os.getppid() != parent_pid:
                print("[mineru] parent process exited; stopping worker.", flush=True)
                break
            delay = interval
            client = _client_from_config()
            if client is None:
                if time.monotonic() - last_auth_warning > 3600:
                    print("[mineru] neither mineru_access_key/mineru_secret_key nor mineru_token is configured; waiting.", flush=True)
                    last_auth_warning = time.monotonic()
            else:
                try:
                    submit_queued(client)
                    poll_submitted(client)
                    backoff = interval
                except MinerUError as exc:
                    if exc.is_auth:
                        _on_auth_failure(client)
                        if time.monotonic() - last_auth_warning > 3600:
                            print(f"[mineru] authentication failed, check mineru_access_key/mineru_secret_key or mineru_token: {exc}", flush=True)
                            last_auth_warning = time.monotonic()
                        delay = max(interval, 300)
                    else:
                        backoff = min(backoff * 2, 1800)
                        delay = backoff
                        print(f"[mineru] API error, backing off {delay}s: {exc}", flush=True)
                except requests.RequestException as exc:
                    backoff = min(backoff * 2, 1800)
                    delay = backoff
                    print(f"[mineru] network error, backing off {delay}s: {exc}", flush=True)
                except Exception as exc:
                    print(f"[mineru] unexpected error: {exc!r}", flush=True)
            stop_event.wait(delay)
    finally:
        print("[mineru] worker stopped.", flush=True)
        lock.close()
    return 0
