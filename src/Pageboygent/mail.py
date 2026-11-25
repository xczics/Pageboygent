"""Bookboy internal mail system.

Each bookboy has a mailbox directory under the pageboys root:
    <workstation_dir>/pageboys/<bookboy_name>/
        ├── inbox/     # Received mails (one JSON file per mail)
        ├── sent/      # Sent mails (one JSON file per mail)
        └── read.json  # Tracks which mail IDs have been read

Public API:
    send(base_dir, sender, recipients, subject, content="") -> mail_id
    check(base_dir, bookboy_name, folder="inbox") -> list[dict]
    read(base_dir, bookboy_name, mail_id) -> dict
    reply(base_dir, bookboy_name, mail_id, content) -> mail_id
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class BookBoyMailError(RuntimeError):
    """Raised on mail operation failures."""
    pass


def _mail_path(base_dir: Path, bookboy_name: str, folder: str) -> Path:
    """Get the path to a bookboy's mailbox folder."""
    return base_dir / bookboy_name / folder


def _ensure_mailbox(base_dir: Path, bookboy_name: str) -> None:
    """Create mailbox directories for a bookboy if they don't exist."""
    for folder in ("inbox", "sent"):
        path = _mail_path(base_dir, bookboy_name, folder)
        path.mkdir(parents=True, exist_ok=True)
    # Track read status
    read_file = base_dir / bookboy_name / "read.json"
    if not read_file.exists():
        read_file.write_text("[]")


def _next_mail_id(base_dir: Path, bookboy_name: str) -> str:
    """Generate a unique mail ID based on timestamp."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    return f"{ts}"


def _write_mail(base_dir: Path, bookboy_name: str, folder: str,
                mail_data: dict[str, Any]) -> str:
    """Write a mail JSON file and return its ID."""
    _ensure_mailbox(base_dir, bookboy_name)
    mail_id = mail_data.get("id") or _next_mail_id(base_dir, bookboy_name)
    mail_data["id"] = mail_id
    folder_path = _mail_path(base_dir, bookboy_name, folder)
    file_path = folder_path / f"{mail_id}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(mail_data, f, ensure_ascii=False)
    return mail_id


def _read_mail_file(base_dir: Path, bookboy_name: str, folder: str,
                    mail_id: str) -> dict[str, Any] | None:
    """Read a specific mail file, return None if not found."""
    file_path = _mail_path(base_dir, bookboy_name, folder) / f"{mail_id}.json"
    if not file_path.exists():
        return None
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)


def _mark_read(base_dir: Path, bookboy_name: str, mail_id: str) -> None:
    """Mark a mail as read in the read.json tracker."""
    read_file = base_dir / bookboy_name / "read.json"
    try:
        read_ids = json.loads(read_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, FileNotFoundError):
        read_ids = []
    if mail_id not in read_ids:
        read_ids.append(mail_id)
    read_file.write_text(json.dumps(read_ids, ensure_ascii=False))


def _is_read(base_dir: Path, bookboy_name: str, mail_id: str) -> bool:
    """Check if a mail has been marked as read."""
    read_file = base_dir / bookboy_name / "read.json"
    try:
        read_ids = json.loads(read_file.read_text(encoding="utf-8"))
        return mail_id in read_ids
    except (json.JSONDecodeError, FileNotFoundError):
        return False


def _list_mails(base_dir: Path, bookboy_name: str, folder: str
                ) -> list[dict[str, Any]]:
    """List all mail files in a folder, newest first."""
    folder_path = _mail_path(base_dir, bookboy_name, folder)
    if not folder_path.is_dir():
        return []
    mails = []
    for file_path in sorted(folder_path.iterdir(), reverse=True):
        if file_path.suffix == ".json":
            with open(file_path, encoding="utf-8") as f:
                mail = json.load(f)
            mail["read"] = _is_read(base_dir, bookboy_name, mail.get("id", ""))
            mails.append(mail)
    return mails


# ===== Public API =====


def send(base_dir: str | Path,
         sender: str,
         recipients: str | list[str],
         subject: str,
         content: str = "") -> str:
    """Send a mail from one bookboy to one or more recipients.

    Args:
        base_dir: Root directory for all bookboy mailboxes.
        sender: Name of the sending bookboy.
        recipients: Single name or list of recipient names.
        subject: Mail subject line.
        content: Mail body text (default: "").

    Returns:
        The mail ID string.

    Raises:
        BookBoyMailError: On invalid input.
    """
    base = Path(base_dir)
    if isinstance(recipients, str):
        recipients = [recipients]
    if not recipients:
        raise BookBoyMailError("At least one recipient is required")
    if not sender:
        raise BookBoyMailError("Sender name is required")
    if not subject:
        raise BookBoyMailError("Subject is required")

    mail_id = _next_mail_id(base, sender)
    now = datetime.now(timezone.utc).isoformat()

    mail_data = {
        "id": mail_id,
        "from": sender,
        "to": recipients,
        "subject": subject,
        "content": content,
        "time": now,
    }

    # Write to each recipient's inbox
    for recipient in recipients:
        _write_mail(base, recipient, "inbox", dict(mail_data))

    # Write copy to sender's sent folder
    mail_data["to"] = recipients  # Keep original recipients list
    _write_mail(base, sender, "sent", dict(mail_data))

    return mail_id


def check(base_dir: str | Path,
          bookboy_name: str,
          folder: str = "inbox") -> list[dict[str, Any]]:
    """Check mails in a bookboy's mailbox folder.

    After checking, unread mails are marked as read.

    Args:
        base_dir: Root directory for all bookboy mailboxes.
        bookboy_name: Name of the bookboy.
        folder: Folder to check ('inbox', 'sent', or 'archive'). Default: 'inbox'.

    Returns:
        List of mail dicts (newest first). Each dict has keys:
        id, from, to, subject, content, time, read.

    Raises:
        BookBoyMailError: If folder name is invalid.
    """
    base = Path(base_dir)
    valid_folders = {"inbox", "sent", "archive"}
    if folder not in valid_folders:
        raise BookBoyMailError(
            f"Invalid folder: '{folder}'. Valid options: {', '.join(sorted(valid_folders))}"
        )

    _ensure_mailbox(base, bookboy_name)
    mails = _list_mails(base, bookboy_name, folder)

    # Mark unread mails as read (for inbox only)
    if folder == "inbox":
        for mail in mails:
            if not mail.get("read", True):
                _mark_read(base, bookboy_name, mail["id"])
                mail["read"] = True

    return mails


def read(base_dir: str | Path,
         bookboy_name: str,
         mail_id: str) -> dict[str, Any]:
    """Read a specific mail by ID.

    Automatically marks the mail as read.

    Args:
        base_dir: Root directory for all bookboy mailboxes.
        bookboy_name: Name of the bookboy.
        mail_id: The mail ID to read.

    Returns:
        The mail dict with full content.

    Raises:
        BookBoyMailError: If mail not found.
    """
    base = Path(base_dir)
    # Search inbox first, then sent, then archive
    for folder in ("inbox", "sent", "archive"):
        mail = _read_mail_file(base, bookboy_name, folder, mail_id)
        if mail is not None:
            # Mark as read
            _mark_read(base, bookboy_name, mail_id)
            mail["read"] = True
            return mail

    raise BookBoyMailError(f"Mail '{mail_id}' not found for bookboy '{bookboy_name}'")


def reply(base_dir: str | Path,
          bookboy_name: str,
          mail_id: str,
          content: str) -> str:
    """Reply to a mail.

    Finds the original mail, then sends a reply back to the original sender
    with subject prefixed by 'Re: '.

    Args:
        base_dir: Root directory for all bookboy mailboxes.
        bookboy_name: Name of the replying bookboy.
        mail_id: ID of the mail to reply to.
        content: Reply body text.

    Returns:
        The reply mail ID.

    Raises:
        BookBoyMailError: If original mail not found.
    """
    original = read(base_dir, bookboy_name, mail_id)
    original_sender = original["from"]
    original_subject = original["subject"]

    reply_subject = f"Re: {original_subject}"
    return send(base_dir, bookboy_name, original_sender, reply_subject, content)
