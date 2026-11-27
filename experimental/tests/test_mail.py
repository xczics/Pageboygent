"""Tests for the bookboy internal mail system (mail.py)."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import pytest

# We'll test via the public API
from experimental.mail import (
    BookBoyMailError,
    send,
    check,
    read,
    reply,
    _ensure_mailbox,
    _mail_path,
    _is_read,
)

# ===== Fixtures =====


@pytest.fixture
def test_dir():
    """Create a temporary directory for mailbox files."""
    tmp = Path(tempfile.mkdtemp(prefix="bookboy_test_"))
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


# ===== _mail_path tests =====


class TestMailPath:
    def test_generates_correct_path(self, test_dir):
        path = _mail_path(test_dir, "student1", "inbox")
        assert path == test_dir / "student1" / "inbox"
        assert isinstance(path, Path)

    def test_accepts_sent_folder(self, test_dir):
        path = _mail_path(test_dir, "student1", "sent")
        assert path == test_dir / "student1" / "sent"


# ===== _ensure_mailbox tests =====


class TestEnsureMailbox:
    def test_creates_mailbox_dirs(self, test_dir):
        _ensure_mailbox(test_dir, "xiaoming")
        assert (test_dir / "xiaoming" / "inbox").is_dir()
        assert (test_dir / "xiaoming" / "sent").is_dir()
        assert (test_dir / "xiaoming" / "read.json").is_file()

    def test_idempotent(self, test_dir):
        _ensure_mailbox(test_dir, "xiaoming")
        _ensure_mailbox(test_dir, "xiaoming")  # should not raise
        assert (test_dir / "xiaoming" / "inbox").is_dir()


# ===== send tests =====


class TestSend:
    def test_send_creates_mail_in_recipient_inbox(self, test_dir):
        mail_id = send(test_dir, "zhangsan", "lisi", "Hello", "World")
        inbox = test_dir / "lisi" / "inbox"
        files = list(inbox.iterdir())
        assert len(files) == 1
        with open(files[0]) as f:
            msg = json.load(f)
        assert msg["from"] == "zhangsan"
        assert msg["to"] == ["lisi"]
        assert msg["subject"] == "Hello"
        assert msg["content"] == "World"
        # Read status tracked via read.json, not in file
        assert _is_read(test_dir, "lisi", msg["id"]) is False

    def test_send_creates_copy_in_sender_sent(self, test_dir):
        mail_id = send(test_dir, "zhangsan", "lisi", "Hi", "There")
        sent = test_dir / "zhangsan" / "sent"
        files = list(sent.iterdir())
        assert len(files) == 1
        with open(files[0]) as f:
            msg = json.load(f)
        assert msg["from"] == "zhangsan"
        assert msg["to"] == ["lisi"]

    def test_send_to_multiple_recipients(self, test_dir):
        send(test_dir, "zhangsan", ["lisi", "wangwu"], "Multi", "Test")
        assert len(list((test_dir / "lisi" / "inbox").iterdir())) == 1
        assert len(list((test_dir / "wangwu" / "inbox").iterdir())) == 1

    def test_send_auto_creates_mailboxes(self, test_dir):
        send(test_dir, "new_sender", "new_recipient", "Auto", "Create")
        assert (test_dir / "new_sender" / "sent").is_dir()
        assert (test_dir / "new_recipient" / "inbox").is_dir()

    def test_send_returns_string_id(self, test_dir):
        mail_id = send(test_dir, "a", "b", "Test", "Content")
        assert isinstance(mail_id, str)
        assert len(mail_id) > 0

    def test_send_content_defaults_to_empty(self, test_dir):
        mail_id = send(test_dir, "a", "b", "No content")
        inbox = test_dir / "b" / "inbox"
        msg_file = list(inbox.iterdir())[0]
        with open(msg_file) as f:
            msg = json.load(f)
        assert msg["content"] == ""


# ===== check tests =====


class TestCheck:
    def test_check_returns_empty_list_for_new_mailbox(self, test_dir):
        _ensure_mailbox(test_dir, "empty_one")
        result = check(test_dir, "empty_one")
        assert result == []

    def test_check_returns_unread_mails(self, test_dir):
        send(test_dir, "zhangsan", "lisi", "Subject", "Body")
        result = check(test_dir, "lisi")
        assert len(result) == 1
        assert result[0]["subject"] == "Subject"
        assert result[0]["from"] == "zhangsan"

    def test_check_mark_as_read(self, test_dir):
        send(test_dir, "a", "b", "Read me", "Content")
        check(test_dir, "b")  # This should mark as read
        # Read status tracked via read.json
        inbox = test_dir / "b" / "inbox"
        msg_file = list(inbox.iterdir())[0]
        with open(msg_file) as f:
            msg = json.load(f)
        assert _is_read(test_dir, "b", msg["id"]) is True

    def test_check_filters_by_folder(self, test_dir):
        send(test_dir, "a", "b", "Inbox mail", "Content")
        result_sent = check(test_dir, "a", folder="sent")
        assert len(result_sent) == 1
        result_inbox = check(test_dir, "a", folder="inbox")
        assert len(result_inbox) == 0

    def test_check_raises_on_invalid_folder(self, test_dir):
        _ensure_mailbox(test_dir, "u")
        with pytest.raises(BookBoyMailError, match="Invalid folder"):
            check(test_dir, "u", folder="invalid")


# ===== read tests =====


class TestRead:
    def test_read_returns_full_content(self, test_dir):
        mail_id = send(test_dir, "a", "b", "Full read", "Detailed content here")
        msg = read(test_dir, "b", mail_id)
        assert msg["subject"] == "Full read"
        assert msg["content"] == "Detailed content here"
        assert msg["from"] == "a"

    def test_read_marks_as_read(self, test_dir):
        mail_id = send(test_dir, "a", "b", "Mark read", "Content")
        read(test_dir, "b", mail_id)
        assert _is_read(test_dir, "b", mail_id) is True

    def test_read_raises_on_nonexistent_id(self, test_dir):
        _ensure_mailbox(test_dir, "u")
        with pytest.raises(BookBoyMailError, match="not found"):
            read(test_dir, "u", "nonexistent_id")


# ===== reply tests =====


class TestReply:
    def test_reply_sends_back_to_sender(self, test_dir):
        mail_id = send(test_dir, "zhangsan", "lisi", "Question?", "What is AI?")
        reply(test_dir, "lisi", mail_id, "AI is artificial intelligence")
        # Check lisi's sent folder for the reply
        sent = test_dir / "lisi" / "sent"
        msgs = list(sent.iterdir())
        assert len(msgs) == 1
        with open(msgs[0]) as f:
            msg = json.load(f)
        assert msg["from"] == "lisi"
        assert msg["to"] == ["zhangsan"]
        assert msg["subject"] == "Re: Question?"
        assert msg["content"] == "AI is artificial intelligence"

    def test_reply_delivers_to_original_sender_inbox(self, test_dir):
        mail_id = send(test_dir, "a", "b", "Q", "Body")
        reply(test_dir, "b", mail_id, "Answer")
        inbox = test_dir / "a" / "inbox"
        assert len(list(inbox.iterdir())) == 1

    def test_reply_raises_on_nonexistent_mail(self, test_dir):
        _ensure_mailbox(test_dir, "u")
        with pytest.raises(BookBoyMailError, match="not found"):
            reply(test_dir, "u", "bad_id", "Reply")


# ===== Edge cases =====


class TestEdgeCases:
    def test_send_to_self(self, test_dir):
        mail_id = send(test_dir, "a", "a", "Self mail", "Content")
        inbox = test_dir / "a" / "inbox"
        assert len(list(inbox.iterdir())) == 1

    def test_special_chars_in_content(self, test_dir):
        content = "你好，世界！\nLine 2\n\tTab\n{\"json\": true}"
        send(test_dir, "a", "b", "Special", content)
        result = check(test_dir, "b")
        assert result[0]["content"] == content

    def test_multiple_mails_in_order(self, test_dir):
        ids = []
        for i in range(5):
            mid = send(test_dir, "a", "b", f"Mail {i}", f"Body {i}")
            ids.append(mid)
        result = check(test_dir, "b")
        assert len(result) == 5
        # Should be sorted newest first by default
        assert result[0]["subject"] == "Mail 4"
