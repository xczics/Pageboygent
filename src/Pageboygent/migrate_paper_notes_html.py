from __future__ import annotations

import argparse
import errno
from typing import Any

from .note_storage import note_storage
from .storage import Paper
from .zt_tools import zot, zot_local


def _get_note_keys_for_item(item_key: str) -> list[str]:
    try:
        note_keys = zot_local.children(item_key, format="keys", itemType="note")
    except Exception:
        note_keys = zot.children(item_key, format="keys", itemType="note")
    if isinstance(note_keys, bytes):
        text = note_keys.decode("utf-8")
    else:
        text = str(note_keys)
    return [line.strip() for line in text.split("\n") if line.strip()]


def _get_note_item(note_key: str) -> dict[str, Any]:
    try:
        return zot_local.item(note_key)
    except Exception:
        return zot.item(note_key)


def _is_conn_refused(exc: Exception) -> bool:
    text = str(exc).lower()
    return ("connection refused" in text) or (str(getattr(exc, "errno", "")) == str(errno.ECONNREFUSED))


def migrate_paper_notes_md_to_html(
    limit_notes: int | None = None,
    dry_run: bool = False,
    max_failures: int = 20,
) -> dict[str, Any]:
    migrated = 0
    scanned = 0
    skipped_html = 0
    failed = 0
    aborted = False

    # Preflight: ensure at least one Zotero endpoint is reachable.
    try:
        zot_local.top(limit=1)
    except Exception:
        try:
            zot.top(limit=1)
        except Exception as exc:
            print(f"[fatal] Zotero APIs are not reachable: {exc}")
            return {
                "scanned": scanned,
                "migrated": migrated,
                "skipped_html": skipped_html,
                "failed": failed,
                "dry_run": dry_run,
                "aborted": True,
            }

    for paper in Paper.select().order_by(Paper.id.asc()):
        item_key = str(paper.key_in_zotero)
        try:
            note_keys = _get_note_keys_for_item(item_key)
        except Exception as exc:
            print(f"[warn] failed to list notes for paper {paper.id} ({item_key}): {exc}")
            failed += 1
            if failed >= max_failures and _is_conn_refused(exc):
                print(f"[fatal] too many connectivity failures ({failed}), aborting this run")
                aborted = True
                break
            continue

        for note_key in note_keys:
            if limit_notes is not None and scanned >= limit_notes:
                return {
                    "scanned": scanned,
                    "migrated": migrated,
                    "skipped_html": skipped_html,
                    "failed": failed,
                    "dry_run": dry_run,
                }

            scanned += 1
            try:
                note_item = _get_note_item(note_key)
                note_data = note_item.get("data", {})
                note_raw = str(note_data.get("note", ""))
                note_tags = [tag["tag"] for tag in note_data.get("tags", []) if tag.get("tag")]

                record = note_storage.ingest_note_from_zotero(
                    item_key=item_key,
                    note_key=note_key,
                    note_content_raw=note_raw,
                    tags=note_tags,
                )

                if not record.needs_sync:
                    skipped_html += 1
                    print(f"[skip] note {note_key} already html-like; no sync needed")
                    continue

                if dry_run:
                    print(f"[dry-run] note {note_key} will be converted md->html and synced")
                    continue

                ok = note_storage.sync_record_to_zotero(record)
                if ok:
                    migrated += 1
                    print(f"[ok] migrated note {note_key} for paper item {item_key}")
                else:
                    failed += 1
                    print(f"[fail] sync failed for note {note_key} (paper item {item_key})")
            except Exception as exc:
                failed += 1
                print(f"[fail] note {note_key} migration error: {exc}")
                if failed >= max_failures and _is_conn_refused(exc):
                    print(f"[fatal] too many connectivity failures ({failed}), aborting this run")
                    aborted = True
                    break

        if aborted:
            break

    return {
        "scanned": scanned,
        "migrated": migrated,
        "skipped_html": skipped_html,
        "failed": failed,
        "dry_run": dry_run,
        "aborted": aborted,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pageboygent-migrate-paper-notes-html",
        description="Migrate paper notes from md-like Zotero content into local storage and sync back as html.",
    )
    parser.add_argument(
        "--limit-notes",
        type=int,
        default=None,
        help="Only process the first N notes for small-batch validation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report only; do not sync anything to Zotero",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=20,
        help="Abort run after this number of failures (default: 20)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    summary = migrate_paper_notes_md_to_html(
        limit_notes=args.limit_notes,
        dry_run=args.dry_run,
        max_failures=args.max_failures,
    )
    print("\n=== Migration Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
