from __future__ import annotations

import io
from pathlib import Path
import zipfile

import peewee
import pytest

import Pageboygent.mineru_tasks as mt


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch):
    test_db = peewee.SqliteDatabase(str(tmp_path / "mineru_tasks_test.db"))
    test_db.bind([mt.MinerUTask], bind_refs=False, bind_backrefs=False)
    test_db.connect(reuse_if_open=True)
    test_db.create_tables([mt.MinerUTask], safe=True)
    monkeypatch.setattr(mt, "_out_md_dir", tmp_path / "out_mds")
    monkeypatch.setattr(mt, "_human_upload_dir", tmp_path / "human_upload_pdfs")
    real_get_config = mt.get_config
    monkeypatch.setattr(mt, "get_config", lambda k, *a: 2 if k == "mineru_max_attempts" else real_get_config(k, *a))
    try:
        yield tmp_path
    finally:
        test_db.close()
        mt.MinerUTask.bind(mt._db, bind_refs=False, bind_backrefs=False)


def _make_pdf(tmp_path: Path, key: str) -> Path:
    path = tmp_path / "input_pdfs" / f"{key}.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4 fake")
    return path


def _zip_with_full_md(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("layout.json", "{}")
        zf.writestr("full.md", text)
    return buf.getvalue()


class FakeResp:
    def __init__(self, status=200, json_body=None, content=b""):
        self.status_code = status
        self._json = json_body
        self.content = content
        self.text = str(json_body)

    def _with_text(self, text):
        self.text = text
        return self

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


class FakeMinerU:
    """模拟 MinerU API：记录上传，按 results 返回批量结果。"""

    def __init__(self):
        self.uploads: list[str] = []
        self.results: dict[str, dict] = {}   # data_id -> extract_result item
        self.batch_files: dict[str, list[dict]] = {}
        self.auth_ok = True
        self.not_found_batches: set[str] = set()

    def request(self, method, url, headers=None, timeout=None, json=None):
        if not self.auth_ok:
            return FakeResp(401, {"msg": "user authenticate failed"})
        if method == "POST" and url.endswith("/file-urls/batch"):
            batch_id = f"b{len(self.batch_files) + 1}"
            self.batch_files[batch_id] = json["files"]
            return FakeResp(200, {"code": 0, "data": {"batch_id": batch_id, "file_urls": [f"https://up/{f['data_id']}" for f in json["files"]]}})
        if method == "GET" and "/extract-results/batch/" in url:
            batch_id = url.rsplit("/", 1)[1]
            if batch_id in self.not_found_batches:
                return FakeResp(200, {"code": -60012, "msg": "task not found or expire"})
            # 与真实接口一致：结果里不回传 data_id，只有 file_name
            items = [
                {k: v for k, v in {"file_name": f["name"], "state": "running", **self.results.get(f["data_id"], {})}.items() if k != "data_id"}
                for f in self.batch_files[batch_id]
            ]
            return FakeResp(200, {"code": 0, "data": {"batch_id": batch_id, "extract_result": items}})
        raise AssertionError(f"unexpected {method} {url}")

    # 模拟预签名 URL 的签名方式：None 表示按文档不带 Content-Type（静态 token），否则需带该类型（AK/SK JWT）
    signed_content_type: str | None = None

    def put(self, url, data=None, headers=None, timeout=None):
        if (headers or {}).get("Content-Type") != self.signed_content_type:
            return FakeResp(403, content=b"")._with_text("<Error><Code>SignatureDoesNotMatch</Code></Error>")
        self.uploads.append(url.rsplit("/", 1)[1])
        return FakeResp(200)

    def get(self, url, timeout=None):
        return FakeResp(200, content=_zip_with_full_md(f"# md for {url}"))


def _client(fake: FakeMinerU) -> mt.MinerUClient:
    return mt.MinerUClient(token="t", base_url="https://api", model_version="vlm", session=fake)


def test_full_flow_writes_md_and_marks_done(isolated):
    fake = FakeMinerU()
    task = mt.enqueue("KEY1", str(_make_pdf(isolated, "KEY1")), "Paper 1")
    assert mt.enqueue("KEY1", "ignored", "dup").id == task.id  # 幂等

    assert mt.submit_queued(_client(fake)) == 1
    assert fake.uploads == ["KEY1"]
    t = mt.MinerUTask.get_by_id(task.id)
    assert t.state == mt.STATE_SUBMITTED and t.batch_id == "b1" and t.model_version == "vlm"

    fake.results["KEY1"] = {"data_id": "KEY1", "state": "running", "extract_progress": {"extracted_pages": 3, "total_pages": 10}}
    assert mt.poll_submitted(_client(fake)) == 0
    t = mt.MinerUTask.get_by_id(task.id)
    assert (t.remote_state, t.extracted_pages, t.total_pages) == ("running", 3, 10)

    fake.results["KEY1"] = {"data_id": "KEY1", "state": "done", "full_zip_url": "https://zip/KEY1"}
    assert mt.poll_submitted(_client(fake)) == 1
    assert mt.MinerUTask.get_by_id(task.id).state == mt.STATE_DONE
    assert (isolated / "out_mds" / "KEY1.md").read_text() == "# md for https://zip/KEY1"


def test_remote_failure_retries_then_fails(isolated):
    fake = FakeMinerU()
    task = mt.enqueue("KEY2", str(_make_pdf(isolated, "KEY2")))
    fake.results["KEY2"] = {"data_id": "KEY2", "state": "failed", "err_msg": "boom"}

    mt.submit_queued(_client(fake)); mt.poll_submitted(_client(fake))
    t = mt.MinerUTask.get_by_id(task.id)
    assert t.state == mt.STATE_QUEUED and t.attempts == 1 and "boom" in t.err_msg

    mt.submit_queued(_client(fake)); mt.poll_submitted(_client(fake))   # max_attempts=2
    t = mt.MinerUTask.get_by_id(task.id)
    assert t.state == mt.STATE_FAILED and t.attempts == 2

    assert mt.retry([task.id]) == 1
    t = mt.MinerUTask.get_by_id(task.id)
    assert t.state == mt.STATE_QUEUED and t.attempts == 0


def test_auth_error_requeues_without_consuming_attempts(isolated):
    fake = FakeMinerU()
    fake.auth_ok = False
    task = mt.enqueue("KEY3", str(_make_pdf(isolated, "KEY3")))
    with pytest.raises(mt.MinerUError) as exc_info:
        mt.submit_queued(_client(fake))
    assert exc_info.value.is_auth
    t = mt.MinerUTask.get_by_id(task.id)
    assert t.state == mt.STATE_QUEUED and t.attempts == 0


def test_missing_pdf_fails_immediately(isolated):
    task = mt.enqueue("KEY4", str(isolated / "nope.pdf"))
    assert mt.submit_queued(_client(FakeMinerU())) == 0
    assert mt.MinerUTask.get_by_id(task.id).state == mt.STATE_FAILED


def test_fallback_to_manual_moves_pdf(isolated):
    pdf = _make_pdf(isolated, "KEY5")
    task = mt.enqueue("KEY5", str(pdf))
    assert mt.fallback_to_manual([task.id]) == 1
    t = mt.MinerUTask.get_by_id(task.id)
    assert t.state == mt.STATE_MANUAL
    assert not pdf.exists() and (isolated / "human_upload_pdfs" / "KEY5.pdf").exists()
    # 已转手动的任务不会被 worker 领取，也不能再次 fallback
    assert mt.submit_queued(_client(FakeMinerU())) == 0
    assert mt.fallback_to_manual([task.id]) == 0


def _fake_jwt(exp: int) -> str:
    import base64, json
    enc = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).decode().rstrip("=")
    return f"{enc({'alg': 'HS512'})}.{enc({'iss': 'OpenXLab', 'exp': exp})}.sig"


class FakeOpenXLab:
    def __init__(self, exp_offsets):
        import time
        self.exp_offsets = list(exp_offsets)
        self.calls: list[str] = []
        self.issued: list[str] = []
        self.now = time.time

    def post(self, url, json=None, headers=None, timeout=None):
        import base64, hmac
        path = url.rsplit("/", 1)[1]
        self.calls.append(path)
        if path == "auth":
            assert json == {"ak": "AK"}
            return FakeResp(200, {"msgCode": "10000", "data": {"msgCode": "10000", "data": {"nonce": "n1", "algorithm": "HmacSHA256"}}})
        expected_d = base64.b64encode(hmac.new(b"SK", b"n1", "SHA256").digest()).decode()
        if json["d"] != expected_d:
            return FakeResp(200, {"msgCode": "10000", "data": {"msgCode": "40001", "msg": "bad signature"}})
        jwt = _fake_jwt(int(self.now()) + self.exp_offsets.pop(0))
        self.issued.append(jwt)
        return FakeResp(200, {"msgCode": "10000", "data": {"msgCode": "10000", "data": {"jwt": jwt}}})


def test_openxlab_jwt_provider_caches_and_refreshes():
    fake = FakeOpenXLab([3600, 3600])
    provider = mt.OpenXLabJwtProvider(" AK ", "SK", session=fake)
    first = provider()
    assert provider() == first and fake.calls == ["auth", "getJwt"]   # 有效期内复用
    provider._expires_at = mt.datetime.now()                          # 模拟即将过期
    second = provider()
    assert second != first or fake.calls.count("getJwt") == 2
    assert fake.calls == ["auth", "getJwt", "auth", "getJwt"]


def test_openxlab_bad_secret_is_auth_error():
    provider = mt.OpenXLabJwtProvider("AK", "WRONG", session=FakeOpenXLab([3600]))
    with pytest.raises(mt.MinerUError) as exc_info:
        provider()
    assert exc_info.value.is_auth


def test_client_sends_current_jwt_from_provider():
    seen = []

    class Session(FakeMinerU):
        def request(self, method, url, headers=None, timeout=None, json=None):
            seen.append(headers["Authorization"])
            return super().request(method, url, headers=headers, timeout=timeout, json=json)

    tokens = iter(["jwt-1", "jwt-2"])
    client = mt.MinerUClient(token=lambda: next(tokens), base_url="https://api", model_version="vlm", session=Session())
    client.request_upload_urls([{"name": "a.pdf", "data_id": "a"}])
    client.request_upload_urls([{"name": "b.pdf", "data_id": "b"}])
    assert seen == ["Bearer jwt-1", "Bearer jwt-2"]


def test_batch_not_found_is_tolerated_within_grace_period(isolated):
    fake = FakeMinerU()
    task = mt.enqueue("KEY6", str(_make_pdf(isolated, "KEY6")))
    mt.submit_queued(_client(fake))
    fake.not_found_batches.add("b1")
    assert mt.poll_submitted(_client(fake)) == 0
    assert mt.MinerUTask.get_by_id(task.id).state == mt.STATE_SUBMITTED   # 宽限期内：继续等

    mt.MinerUTask.update(submitted_at=mt.datetime.now() - mt.timedelta(minutes=11)).where(mt.MinerUTask.id == task.id).execute()
    mt.poll_submitted(_client(fake))
    t = mt.MinerUTask.get_by_id(task.id)
    assert t.state == mt.STATE_QUEUED and t.attempts == 1 and "not found" in t.err_msg


@pytest.mark.parametrize("signed_content_type", [None, "application/pdf"])
def test_upload_handles_both_presigned_signature_styles(isolated, signed_content_type):
    fake = FakeMinerU()
    fake.signed_content_type = signed_content_type
    task = mt.enqueue("KEY7", str(_make_pdf(isolated, "KEY7")))
    assert mt.submit_queued(_client(fake)) == 1 and fake.uploads == ["KEY7"]
    assert mt.MinerUTask.get_by_id(task.id).state == mt.STATE_SUBMITTED


def test_upload_treats_connection_reset_as_signature_mismatch_and_remembers_style(isolated, monkeypatch):
    import requests as _requests
    monkeypatch.setattr(mt, "_upload_style_cache", {})
    fake = FakeMinerU()
    fake.signed_content_type = "application/pdf"
    attempts = []
    real_put = fake.put

    def put(url, data=None, headers=None, timeout=None):
        attempts.append((headers or {}).get("Content-Type"))
        if not headers:   # 大文件签名不匹配时 OSS 直接断开连接
            raise _requests.ConnectionError("Broken pipe")
        return real_put(url, data=data, headers=headers, timeout=timeout)

    fake.put = put
    mt.enqueue("KEY8", str(_make_pdf(isolated, "KEY8")))
    mt.enqueue("KEY9", str(_make_pdf(isolated, "KEY9")))
    assert mt.submit_queued(_client(fake)) == 2
    # 第一个文件先试不带、断开后带 Content-Type 成功；第二个文件直接用记住的方式
    assert attempts == [None, "application/pdf", "application/pdf"]


def test_accelerated_oss_url_rewrites_only_regional_oss_hosts():
    u = "https://mineru.oss-cn-shanghai.aliyuncs.com/api-upload/x/y.pdf?Expires=1&OSSAccessKeyId=a&Signature=b%2Bc"
    assert mt._accelerated_oss_url(u) == "https://mineru.oss-accelerate.aliyuncs.com/api-upload/x/y.pdf?Expires=1&OSSAccessKeyId=a&Signature=b%2Bc"
    assert mt._accelerated_oss_url(mt._accelerated_oss_url(u)) == mt._accelerated_oss_url(u)
    assert mt._accelerated_oss_url("https://up/KEY1") == "https://up/KEY1"


def test_upload_falls_back_to_regional_endpoint(isolated, monkeypatch):
    import requests as _requests
    real_get_config = mt.get_config
    monkeypatch.setattr(mt, "get_config", lambda k, *a: True if k == "mineru_upload_accelerate" else real_get_config(k, *a))
    hosts = []

    class Session(FakeMinerU):
        def request(self, method, url, headers=None, timeout=None, json=None):
            resp = super().request(method, url, headers=headers, timeout=timeout, json=json)
            if url.endswith("/file-urls/batch"):
                resp._json["data"]["file_urls"] = [f"https://mineru.oss-cn-shanghai.aliyuncs.com/{f['data_id']}?Signature=s" for f in json["files"]]
            return resp

        def put(self, url, data=None, headers=None, timeout=None):
            hosts.append(url.split("/")[2])
            if "oss-accelerate" in url:
                raise _requests.ConnectionError("accelerate unreachable")
            return super().put(url.split("?")[0], data=data, headers=headers, timeout=timeout)

    mt.enqueue("KEYA", str(_make_pdf(isolated, "KEYA")))
    assert mt.submit_queued(_client(Session())) == 1
    assert hosts[0] == "mineru.oss-accelerate.aliyuncs.com" and hosts[-1] == "mineru.oss-cn-shanghai.aliyuncs.com"


def test_client_prefers_static_token_and_falls_back_to_aksk_after_rejection(monkeypatch):
    cfg = {"mineru_token": "STATIC", "mineru_access_key": "AK", "mineru_secret_key": "SK",
           "mineru_api_base": "https://api", "mineru_model_version": "vlm"}
    monkeypatch.setattr(mt, "get_config", lambda k, *a: cfg.get(k))
    monkeypatch.setattr(mt, "_rejected_static_token", None)
    monkeypatch.setattr(mt, "_jwt_provider", None)

    client = mt._client_from_config()
    assert client._token == "STATIC"                                  # 优先静态 token
    mt._on_auth_failure(client)
    fallback = mt._client_from_config()
    assert isinstance(fallback._token, mt.OpenXLabJwtProvider)       # 被拒后改用 AK/SK

    cfg["mineru_token"] = "NEW_STATIC"                                # 用户换了新 token
    assert mt._client_from_config()._token == "NEW_STATIC"

    cfg.pop("mineru_access_key")                                       # 没有后备凭据时继续用 token
    mt._on_auth_failure(mt._client_from_config())
    assert mt._client_from_config()._token == "NEW_STATIC"
