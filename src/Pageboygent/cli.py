from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import (
    ConfigError,
    load_settings,
    list_all_configs,
    masked,
    set_config_paths,
    _registry,
)


def _import_all_package_modules() -> None:
    """Import every module in this package so all register_config() calls are executed."""
    import importlib
    import pkgutil
    from . import __name__ as pkg_name
    from . import __path__ as pkg_path
    for module_info in pkgutil.iter_modules(pkg_path):
        full_name = f"{pkg_name}.{module_info.name}"
        try:
            importlib.import_module(full_name)
        except Exception:  # noqa: BLE001
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pageboygent",
        description="Pageboygent command line interface",
    )
    parser.add_argument("--profile", default="default", help="Configuration profile name")
    parser.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help="Path to user config TOML (default: platform user config path)",
    )
    parser.add_argument(
        "--project-config-file",
        type=Path,
        default=None,
        help="Path to project config TOML (default: ./pageboygent.toml)",
    )
    parser.add_argument(
        "--dotenv-file",
        type=Path,
        default=None,
        help="Path to dotenv file (default: ./.env)",
    )
    parser.add_argument(
        "--require",
        nargs="*",
        default=[],
        help="Required keys to validate, e.g. --require ds_api_key zotero_key",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print effective config (secrets masked) as JSON",
    )

    subparsers = parser.add_subparsers(dest="command")
    serve_parser = subparsers.add_parser(
        "serve-storage",
        help="Start a lightweight web server to display storage tables",
    )
    serve_parser.add_argument(
        "bind",
        nargs="?",
        default=None,
        help="Bind endpoint in host:port format, e.g. 127.0.0.1:8765",
    )
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind (default: 127.0.0.1)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to bind (default: 8765)",
    )
    serve_parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open browser automatically after server starts",
    )

    mineru_worker_parser = subparsers.add_parser(
        "mineru-worker",
        help="Run the MinerU API PDF->MD worker (started automatically by serve-storage)",
    )
    mineru_worker_parser.add_argument(
        "--parent-pid",
        type=int,
        default=None,
        help="Exit when this parent process is gone (used by serve-storage)",
    )

    migrate_parser = subparsers.add_parser(
        "migrate-paper-notes-html",
        help="Pull paper notes to local storage and sync back to Zotero as html",
    )
    migrate_parser.add_argument(
        "--limit-notes",
        type=int,
        default=None,
        help="Only process first N notes for small-batch validation",
    )
    migrate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report only; do not sync anything to Zotero",
    )
    migrate_parser.add_argument(
        "--max-failures",
        type=int,
        default=20,
        help="Abort run after this number of failures (default: 20)",
    )

    rag_status_parser = subparsers.add_parser(
        "rag-status",
        help="Show the status of the Worthy specialist classifier/data used for RAG chunk filtering",
    )
    rag_status_parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON (as returned by worthy's own Worthy.status()) instead of a formatted summary",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    set_config_paths(
        user_config=args.config_file,
        project_config=args.project_config_file,
        dotenv=args.dotenv_file,
    )

    try:
        load_settings(profile=args.profile)
        if args.require:
            from .config import get_config, ConfigError as CE
            missing = [k for k in args.require if not get_config(k)]
            if missing:
                raise CE(
                    f"Missing required configuration: {', '.join(missing)}. "
                    "Set environment variables or config file values."
                )
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if args.show_config:
        _import_all_package_modules()
        data: dict[str, object] = {"profile": args.profile}
        for key, value in list_all_configs(include_defaults=True).items():
            item = _registry._items.get(key)
            if item and item.is_secret and value is not None:
                data[key] = masked(value)
            else:
                data[key] = value
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0

    if args.command == "serve-storage":
        from .ui import serve_storage

        # 透传全局配置参数，保证自动拉起的 mineru-worker 子进程读到同一套配置
        global_args = ["--profile", args.profile]
        for flag, value in (
            ("--config-file", args.config_file),
            ("--project-config-file", args.project_config_file),
            ("--dotenv-file", args.dotenv_file),
        ):
            if value is not None:
                global_args += [flag, str(value)]

        return serve_storage(
            bind=args.bind,
            host=args.host,
            port=args.port,
            open_browser=args.open_browser,
            worker_global_args=global_args,
        )

    if args.command == "mineru-worker":
        from .mineru_tasks import run_worker

        return run_worker(parent_pid=args.parent_pid)

    if args.command == "rag-status":
        from .rag import worthy as rag_worthy

        # 状态数据完全来自worthy自己的 Worthy.status()，这里不重新去读sqlite/模型文件。
        status = rag_worthy.status()

        if args.json:
            print(json.dumps(status, ensure_ascii=False, indent=2, default=str))
            return 0

        # 不硬编码字段名/类型：worthy的status()字段可能随版本调整，这里只假设它是dict，逐项打印。
        def _fmt(value) -> str:
            if value is None:
                return "N/A"
            if isinstance(value, float):
                return f"{value:.3f}"
            return str(value)

        print("=== RAG Worthy Specialist Status ===")
        print(f"db_path: {getattr(getattr(rag_worthy, 'store', None), 'db_path', 'N/A')}")
        print(f"model_dir: {getattr(getattr(rag_worthy, 'specialist', None), 'model_dir', 'N/A')}")
        for key, value in status.items():
            print(f"{key}: {_fmt(value)}")
        return 0

    if args.command == "migrate-paper-notes-html":
        from .migrate_paper_notes_html import migrate_paper_notes_md_to_html

        summary = migrate_paper_notes_md_to_html(
            limit_notes=args.limit_notes,
            dry_run=args.dry_run,
            max_failures=args.max_failures,
        )
        print("=== Migration Summary ===")
        for key, value in summary.items():
            print(f"{key}: {value}")
        return 0 if summary.get("failed", 0) == 0 else 1

    print("Configuration loaded successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
